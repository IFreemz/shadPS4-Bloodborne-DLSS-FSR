// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Threaded renderer (Bloodborne): a two-stage draw pipeline, after bbport's design
// (github.com/deadinside28/bloodborne_pc, docs/parallel_gpu.md).
//
// The GPU command thread (stage A) decodes PM4, keeps the register file and selects pipelines.
// For a draw or dispatch it writes a packet (the register blocks changed since the previous
// packet, each stage's user data, the draw parameters) into this ring and goes on decoding.
// The draw recording thread (stage B) applies each packet to its own copy of the registers and
// runs the rest of the draw: textures, buffers, render targets, barriers, command recording.
// While packets are in flight stage B owns that state; anything else stage A does on it first
// waits for stage B to run dry (Drain) and then runs on stage A as before.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "common/assert.h"
#include "common/guest_stats.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "common/types.h"

namespace Vulkan {

class DrawPipe {
public:
    /// Runs a packet on stage B.
    using Handler = void (*)(void* context, const u8* packet, u32 size);

    DrawPipe(Handler handler_, void* context_) : handler{handler_}, context{context_} {
        ring = std::make_unique<u8[]>(Capacity);
        const u32 threads = std::thread::hardware_concurrency();
        spin_time = std::chrono::microseconds(threads >= 12 ? 200 : 50);
        thread = std::jthread([this](std::stop_token stop) { Run(stop); });
    }

    ~DrawPipe() {
        Drain();
        thread.request_stop();
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }

    DrawPipe(const DrawPipe&) = delete;
    DrawPipe& operator=(const DrawPipe&) = delete;

    /// The GPU command thread marks itself as stage A.
    static void SetStageA() noexcept {
        role = Role::StageA;
    }
    [[nodiscard]] static bool OnStageA() noexcept {
        return role == Role::StageA;
    }
    [[nodiscard]] static bool OnStageB() noexcept {
        return role == Role::StageB;
    }

    /// Stage A: space for a packet of `size` bytes, valid until Commit().
    u8* Begin(u32 size) {
        size = Align(size + sizeof(Header));
        ASSERT(size <= Capacity / 4);
        u64 at = head;
        const u64 offset = at % Capacity;
        if (offset + size > Capacity) {
            // Wrap: the rest of the ring is skipped (a header with zero size marks it).
            WaitForSpace(at, Capacity - offset);
            reinterpret_cast<Header*>(ring.get() + offset)->size = 0;
            at += Capacity - offset;
            head = at;
            Publish();
        }
        WaitForSpace(at, size);
        pending_size = size;
        auto* header = reinterpret_cast<Header*>(ring.get() + at % Capacity);
        header->size = size;
        return reinterpret_cast<u8*>(header + 1);
    }

    /// Stage A: hands the packet from Begin() to stage B.
    void Commit(u32 payload_size) {
        reinterpret_cast<Header*>(ring.get() + head % Capacity)->payload = payload_size;
        head += pending_size;
        ++packets;
        Publish();
    }

    /// Stage A: waits until stage B has run every committed packet.
    void Drain() {
        if (consumed.load(std::memory_order_acquire) == head) {
            return;
        }
        ++drains;
        Common::GuestStats::CmdStateScope state{Common::GuestStats::CmdState::DrainWait};
        const auto start = std::chrono::steady_clock::now();
        for (u32 spins = 0; consumed.load(std::memory_order_acquire) != head; ++spins) {
            if (spins < 4096) {
                Pause();
            } else {
                std::this_thread::yield();
            }
        }
        drain_time += std::chrono::steady_clock::now() - start;
    }

    [[nodiscard]] bool Idle() const noexcept {
        return consumed.load(std::memory_order_acquire) == head;
    }

    /// Packets stage B has finished (packets are numbered from 1 in commit order).
    [[nodiscard]] u64 ConsumedPackets() const noexcept {
        return consumed_packets.load(std::memory_order_acquire);
    }

    /// Statistics (read on stage A).
    u64 packets = 0;
    u64 drains = 0;
    std::chrono::nanoseconds drain_time{};
    [[nodiscard]] std::chrono::nanoseconds BusyTime() const noexcept {
        return std::chrono::nanoseconds(busy_ns.load(std::memory_order_relaxed));
    }

private:
    enum class Role : u8 { None, StageA, StageB };

    struct Header {
        u32 size;    ///< bytes to the next packet (0: wrap to the ring start)
        u32 payload; ///< bytes written by the producer
    };
    static constexpr u64 Capacity = 16ull << 20;

    static constexpr u32 Align(u32 size) {
        return (size + 63) & ~63u;
    }

    static void Pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }

    void Publish() {
        published.store(head, std::memory_order_seq_cst);
        if (sleeping.load(std::memory_order_seq_cst)) {
            wake.fetch_add(1, std::memory_order_seq_cst);
            wake.notify_one();
        }
    }

    void WaitForSpace(u64 at, u64 size) {
        for (u32 spins = 0; at + size - consumed.load(std::memory_order_acquire) > Capacity;
             ++spins) {
            if (spins < 4096) {
                Pause();
            } else {
                std::this_thread::yield();
            }
        }
    }

    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:GpuDrawRecorder");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
        role = Role::StageB;
        u64 at = 0;
        while (true) {
            // Packets follow each other within microseconds while a frame is decoded: spin,
            // and sleep only after a longer pause (between frames).
            u64 available = published.load(std::memory_order_acquire);
            if (available == at) {
                const auto spin_until = std::chrono::steady_clock::now() + spin_time;
                for (u32 spins = 1; available == at; ++spins) {
                    if (stop.stop_requested()) {
                        return;
                    }
                    Pause();
                    if ((spins & 255) == 0 && std::chrono::steady_clock::now() >= spin_until) {
                        const u32 seen = wake.load(std::memory_order_seq_cst);
                        sleeping.store(true, std::memory_order_seq_cst);
                        if (published.load(std::memory_order_seq_cst) == at &&
                            !stop.stop_requested()) {
                            wake.wait(seen, std::memory_order_seq_cst);
                        }
                        sleeping.store(false, std::memory_order_relaxed);
                    }
                    available = published.load(std::memory_order_acquire);
                }
            }
            const auto* header = reinterpret_cast<const Header*>(ring.get() + at % Capacity);
            if (header->size == 0) {
                at += Capacity - at % Capacity;
                consumed.store(at, std::memory_order_release);
                continue;
            }
            const auto start = std::chrono::steady_clock::now();
            Common::GuestStats::recorder_busy.store(true, std::memory_order_relaxed);
            handler(context, reinterpret_cast<const u8*>(header + 1), header->payload);
            Common::GuestStats::recorder_busy.store(false, std::memory_order_relaxed);
            busy_ns.fetch_add((std::chrono::steady_clock::now() - start).count(),
                              std::memory_order_relaxed);
            at += header->size;
            consumed_packets.fetch_add(1, std::memory_order_release);
            consumed.store(at, std::memory_order_release);
        }
    }

    static inline thread_local Role role = Role::None;
    Handler handler;
    void* context;
    std::unique_ptr<u8[]> ring;
    std::chrono::microseconds spin_time{};
    u64 head = 0; ///< stage A's write position
    u32 pending_size = 0;
    alignas(64) std::atomic<u64> published{0};
    alignas(64) std::atomic<u64> consumed{0};
    std::atomic<u64> consumed_packets{0};
    alignas(64) std::atomic<u32> wake{0};
    std::atomic<bool> sleeping{false};
    std::atomic<s64> busy_ns{0};
    std::jthread thread;
};

} // namespace Vulkan
