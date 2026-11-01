// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>
#include <boost/preprocessor/stringize.hpp>
#include <xxhash.h>

#include "common/assert.h"
#include "common/debug.h"
#include "common/guest_stats.h"
#include "common/polyfill_thread.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_draw_pipe.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

namespace {
// Command thread frame timing (Stutter detail): where the GPU command thread's time goes between
// two flips, including what the per-packet timing doesn't see (the constant engine, which runs
// as its own task, and the command loop around the tasks). Only touched on the command thread;
// no timestamps are taken while Stutter detail is off.
struct CmdFrameTiming {
    using Clock = std::chrono::steady_clock;
    Clock::time_point frame_start{};
    u64 gfx_ns{}, ce_ns{}, ce_in_gfx_ns{}, asc_ns{}, idle_ns{};
    u32 gfx_runs{}, ce_runs{}, frames{}, reports{};
    bool in_gfx{}, flip_seen{};
    double avg_ms{};
    std::array<u64, 256> ce_op_ns{};
    std::array<u32, 256> ce_op_count{};
    std::array<u64, 256> last_opcode_ns{};

    static bool On() noexcept {
        return Common::GuestStats::detail_timing.load(std::memory_order_relaxed);
    }

    static u64 Since(Clock::time_point start) {
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    }

    void Report() {
        const auto now = Clock::now();
        if (frame_start == Clock::time_point{}) {
            frame_start = now;
            Reset();
            return;
        }
        const double ms = std::chrono::duration<double, std::milli>(now - frame_start).count();
        if (ms > 1000.0) {
            // Stutter detail was off in between: start a fresh frame.
            for (size_t i = 0; i < last_opcode_ns.size(); ++i) {
                last_opcode_ns[i] =
                    Common::GuestStats::cmd_opcode_ns[i].load(std::memory_order_relaxed);
            }
            frame_start = now;
            Reset();
            return;
        }
        u64 packet_ns = 0;
        for (size_t i = 0; i < last_opcode_ns.size(); ++i) {
            const u64 v = Common::GuestStats::cmd_opcode_ns[i].load(std::memory_order_relaxed);
            packet_ns += v >= last_opcode_ns[i] ? v - last_opcode_ns[i] : v;
            last_opcode_ns[i] = v;
        }
        const bool slow = frames > 60 && ms > std::max(1.4 * avg_ms, avg_ms + 5.0);
        if (!slow && ms < 250.0) {
            avg_ms = frames == 0 ? ms : avg_ms * 0.95 + ms * 0.05;
        }
        ++frames;
        if ((slow && reports < 400) || frames % 600 == 0) {
            if (slow) {
                ++reports;
            }
            const double gfx = (gfx_ns - std::min(gfx_ns, ce_in_gfx_ns)) / 1e6;
            const double ce = ce_ns / 1e6;
            const double asc = asc_ns / 1e6;
            const double idle = idle_ns / 1e6;
            const double packets = packet_ns / 1e6;
            std::array<std::pair<u64, size_t>, 256> ops{};
            for (size_t i = 0; i < ops.size(); ++i) {
                ops[i] = {ce_op_ns[i], i};
            }
            std::ranges::sort(ops, std::greater<>{});
            std::string ce_ops;
            for (size_t i = 0; i < 4 && ops[i].first >= 50'000; ++i) {
                ce_ops +=
                    fmt::format("{}{:#x} {:.1f} ({})", ce_ops.empty() ? "" : ", ", ops[i].second,
                                ops[i].first / 1e6, ce_op_count[ops[i].second]);
            }
            const auto text = fmt::format(
                "Command thread frame {:.1f} ms (usual {:.1f}): graphics tasks {:.1f} in {} runs "
                "(timed packets {:.1f}, outside packets {:.1f}) | constant engine {:.1f} in {} "
                "runs, {:.1f} of it inside graphics; by opcode: {} | compute queues {:.1f} | "
                "idle {:.1f} | command loop outside tasks {:.1f}",
                ms, avg_ms, gfx, gfx_runs, packets, std::max(0.0, gfx - packets), ce, ce_runs,
                ce_in_gfx_ns / 1e6, ce_ops.empty() ? "-" : ce_ops, asc, idle,
                std::max(0.0, ms - gfx - ce - asc - idle));
            if (slow) {
                LOG_WARNING(Render, "{}", text);
            } else {
                LOG_INFO(Render, "{}", text);
            }
        }
        frame_start = now;
        Reset();
    }

    void Reset() {
        gfx_ns = ce_ns = ce_in_gfx_ns = asc_ns = idle_ns = 0;
        gfx_runs = ce_runs = 0;
        ce_op_ns.fill(0);
        ce_op_count.fill(0);
    }
};
CmdFrameTiming cmd_frame;
} // namespace

// Threaded renderer: graphics packets that leave the draws queued for the draw recording thread.
// Every other packet runs after that thread has run dry (DrainDrawPipe), as without it. Draws
// and dispatches decide in the rasterizer; some of the packets below drain in their handlers
// when they have to (flips, markers, real DMA, unmet waits).
static bool KeepsDrawPipe(PM4ItOpcode opcode) {
    switch (opcode) {
    case PM4ItOpcode::Nop:
    case PM4ItOpcode::ContextControl:
    case PM4ItOpcode::ClearState:
    case PM4ItOpcode::SetConfigReg:
    case PM4ItOpcode::SetContextReg:
    case PM4ItOpcode::SetShReg:
    case PM4ItOpcode::SetUconfigReg:
    case PM4ItOpcode::SetPredication:
    case PM4ItOpcode::IndexType:
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
    case PM4ItOpcode::DrawIndirect:
    case PM4ItOpcode::DrawIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti:
    case PM4ItOpcode::DispatchDirect:
    case PM4ItOpcode::DispatchIndirect:
    case PM4ItOpcode::NumInstances:
    case PM4ItOpcode::IndexBase:
    case PM4ItOpcode::IndexBufferSize:
    case PM4ItOpcode::SetBase:
    case PM4ItOpcode::EventWrite:
    case PM4ItOpcode::DmaData:
    case PM4ItOpcode::CopyData:
    case PM4ItOpcode::MemSemaphore:
    case PM4ItOpcode::AcquireMem:
    case PM4ItOpcode::Rewind:
    case PM4ItOpcode::WaitRegMem:
    case PM4ItOpcode::IndirectBuffer:
    case PM4ItOpcode::IncrementDeCounter:
    case PM4ItOpcode::WaitOnCeCounter:
    case PM4ItOpcode::PfpSyncMe:
    case PM4ItOpcode::StrmoutBufferUpdate:
    case PM4ItOpcode::GetLodStats:
    case PM4ItOpcode::EventWriteEop:
    case PM4ItOpcode::EventWriteEos:
    case PM4ItOpcode::WriteData:
        return true;
    default:
        return false;
    }
}

void Liverpool::NotePendingFence(u64 packet, VAddr address, u32 value) {
    if (packet == 0) {
        return; // Written already.
    }
    while (!pending_fences.empty() && rasterizer->DrawPipeRan(pending_fences.front().packet)) {
        pending_fences.pop_front();
    }
    pending_fences.push_back({packet, address, value});
}

bool Liverpool::MetByPendingFence(const PM4CmdWaitRegMem& wait) {
    if (wait.mem_space.Value() != PM4CmdWaitRegMem::MemSpace::Memory) {
        return false;
    }
    const VAddr address = wait.Address<VAddr>();
    for (auto it = pending_fences.rbegin(); it != pending_fences.rend(); ++it) {
        if (it->address != address) {
            continue;
        }
        if (rasterizer->DrawPipeRan(it->packet)) {
            return false; // Written by now; the memory holds the answer.
        }
        const u32 value = it->value & wait.mask;
        switch (wait.function.Value()) {
        case PM4CmdWaitRegMem::Function::Always:
            return true;
        case PM4CmdWaitRegMem::Function::LessThan:
            return value < wait.ref;
        case PM4CmdWaitRegMem::Function::LessThanEqual:
            return value <= wait.ref;
        case PM4CmdWaitRegMem::Function::Equal:
            return value == wait.ref;
        case PM4CmdWaitRegMem::Function::NotEqual:
            return value != wait.ref;
        case PM4CmdWaitRegMem::Function::GreaterThanEqual:
            return value >= wait.ref;
        case PM4CmdWaitRegMem::Function::GreaterThan:
            return value > wait.ref;
        default:
            return false;
        }
    }
    return false;
}

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
    while (!ce_submits.empty()) {
        ce_submits.front().destroy();
        ce_submits.pop();
    }
    for (auto& queue : mapped_queues) {
        while (!queue.submits.empty()) {
            queue.submits.front().destroy();
            queue.submits.pop();
        }
    }
}

void Liverpool::ProcessCeQueue() {
    auto& queue = mapped_queues[GfxQueueId];
    Task::Handle task{};
    {
        std::scoped_lock lock{queue.m_access};
        if (ce_submits.empty()) {
            return;
        }
        task = ce_submits.front();
    }
    const bool timed = CmdFrameTiming::On();
    const auto ce_start =
        timed ? CmdFrameTiming::Clock::now() : CmdFrameTiming::Clock::time_point{};
    task.resume();
    if (timed) {
        const u64 ce_ns = CmdFrameTiming::Since(ce_start);
        cmd_frame.ce_ns += ce_ns;
        ++cmd_frame.ce_runs;
        if (cmd_frame.in_gfx) {
            cmd_frame.ce_in_gfx_ns += ce_ns;
        }
    }
    if (task.done()) {
        task.destroy();
        std::scoped_lock lock{queue.m_access};
        ce_submits.pop();
        --num_submits;
        std::scoped_lock lock2{submit_mutex};
        submit_cv.notify_all();
    }
}

void Liverpool::ProcessCommands() {
    // Process incoming commands with high priority
    if (num_commands && rasterizer) {
        rasterizer->DrainDrawPipe();
    }
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        Common::GuestStats::CmdStateScope state{Common::GuestStats::CmdState::Command};
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();
#ifdef __linux__
    gpu_tid = gettid();
#endif

    Common::GuestStats::is_cmd_thread = true;
    using Common::GuestStats::CmdState;
    using Common::GuestStats::CmdStateScope;
    const auto set_state = [](CmdState state) {
        Common::GuestStats::cmd_state.store(static_cast<u8>(state), std::memory_order_relaxed);
    };
    while (!stoken.stop_requested()) {
        {
            set_state(CmdState::Idle);
            const bool idle_timed = CmdFrameTiming::On();
            const auto idle_start =
                idle_timed ? CmdFrameTiming::Clock::now() : CmdFrameTiming::Clock::time_point{};
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken, [this] {
                return num_commands || num_submits || submit_done || !frame_ends.empty();
            });
            if (idle_timed) {
                cmd_frame.idle_ns += CmdFrameTiming::Since(idle_start);
            }
            set_state(CmdState::Work);
        }
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();
        Vulkan::DrawPipe::SetStageA();

        curr_qid = -1;

        while ((num_submits || num_commands) && !stoken.stop_requested()) {
            ProcessCommands();
            ProcessCeQueue();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            const bool task_timed = CmdFrameTiming::On();
            const auto task_start =
                task_timed ? CmdFrameTiming::Clock::now() : CmdFrameTiming::Clock::time_point{};
            const bool gfx_task = curr_qid == GfxQueueId;
            cmd_frame.in_gfx = gfx_task;
            task.resume();
            cmd_frame.in_gfx = false;
            if (task_timed) {
                if (gfx_task) {
                    cmd_frame.gfx_ns += CmdFrameTiming::Since(task_start);
                    ++cmd_frame.gfx_runs;
                } else {
                    cmd_frame.asc_ns += CmdFrameTiming::Since(task_start);
                }
            }
            if (cmd_frame.flip_seen) {
                cmd_frame.flip_seen = false;
                if (task_timed) {
                    cmd_frame.Report();
                }
            }

            if (task.done()) {
                // Threaded renderer: the next submission is decoded while the draw recorder
                // still records this one (as bbport does). The guest learns of finished work
                // only from fences and labels, which the recorder writes in order after the
                // draws, and from GPU idle, which waits for the recorder below.
                // Overlap GPU submissions off (or SHADPS4_SUBMIT_DRAIN=1): wait for the recorder
                // after each submission, as before.
                static const bool drain_env = [] {
                    const char* value = std::getenv("SHADPS4_SUBMIT_DRAIN");
                    return value && value[0] == '1';
                }();
                if (rasterizer && (drain_env || !EmulatorSettings.IsBbSubmitOverlap())) {
                    rasterizer->DrainDrawPipe();
                }
                task.destroy();

                {
                    std::scoped_lock lock{queue.m_access};
                    queue.submits.pop();
                    queue.submit_seqs.pop();

                    --num_submits;
                    std::scoped_lock lock2{submit_mutex};
                    submit_cv.notify_all();
                }
                if (rasterizer) {
                    rasterizer->OnSubmissionDecoded();
                }
                RetireFrames();
            }
        }
        RetireFrames();

        if (submit_done) {
            if (rasterizer) {
                rasterizer->OnSubmit();
                rasterizer->Flush();
                if (VideoCore::IsCapturing()) {
                    // The frame's last work is in the capture too.
                    rasterizer->DrainDrawPipe();
                    rasterizer->Flush();
                }
            }
            VideoCore::EndCapture();
            submit_done = false;
        }

        if (rasterizer) {
            rasterizer->DrainDrawPipe();
        }
        {
            std::scoped_lock lk{submit_mutex};
            if (num_submits == 0) {
                work_retired = true;
                submit_cv.notify_all();
            }
        }
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

// Stutter detail: the first kinds of constant dumps that wait for the draw recorder (address,
// size, why), up to 32 lines.
static void LogConstantDumpWait(VAddr address, u32 size, const Vulkan::Rasterizer::QueuedUse& use) {
    if (!Common::GuestStats::detail_timing.load(std::memory_order_relaxed)) {
        return;
    }
    static std::mutex mutex;
    static std::vector<std::tuple<VAddr, u32, u32>> seen;
    const u32 why = u32(use.read_later) | u32(use.written) << 1 | u32(use.overflow) << 2;
    {
        std::scoped_lock lock{mutex};
        if (seen.size() >= 32 ||
            std::ranges::find(seen, std::tuple{address, size, why}) != seen.end()) {
            return;
        }
        seen.emplace_back(address, size, why);
    }
    LOG_INFO(Render_Vulkan,
             "Constant dump waits for the draw recorder: {:#x}, {} bytes | queued draws read it "
             "later: {} | too many queued reads to tell: {} | queued work writes it: {}{}",
             address, size, use.read_later && !use.overflow, use.overflow, use.written,
             use.written ? fmt::format(" ({}{}, {} bytes, queued {} packets earlier)",
                                       use.writer ? use.writer : "not tracked",
                                       use.writer_shader ? fmt::format(" {:#x}", use.writer_shader)
                                                         : std::string{},
                                       use.writer_bytes, use.writer_packets_back)
                         : std::string{});
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb, CmdCopy copy) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type == 2) {
            ccb = NextPacket(ccb, 1);
            continue;
        }
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        // Read the size before running the packet: once it signals the guest, the guest may
        // already reuse the buffer.
        const u32 packet_words = header->type3.NumWords() + 1;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        const bool time_ce_op =
            CmdFrameTiming::On() && opcode != PM4ItOpcode::WaitOnDeCounterDiff &&
            opcode != PM4ItOpcode::IndirectBufferConst && opcode != PM4ItOpcode::Rewind;
        const auto ce_op_start =
            time_ce_op ? CmdFrameTiming::Clock::now() : CmdFrameTiming::Clock::time_point{};
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            // Queued draws may still read the memory the dump overwrites, or queued tasks (DMA,
            // fences) write it after. The draws copy what they read when they are queued, so
            // without either the dump is written now, without waiting for them.
            namespace GS = Common::GuestStats;
            const bool timed = GS::detail_timing.load(std::memory_order_relaxed);
            const auto now_if = [timed] {
                return timed ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
            };
            const auto ns_since = [](std::chrono::steady_clock::time_point start) {
                return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - start)
                                            .count());
            };
            GS::ce_dumps.fetch_add(1, std::memory_order_relaxed);
            if (rasterizer) {
                const auto use =
                    rasterizer->CheckQueuedUse(dump_const->Address<VAddr>(), dump_const->Size());
                const bool must_wait =
                    !EmulatorSettings.IsBbConstantsNoWait() || use.read_later || use.written;
                if (!must_wait && use.huge_writes_skipped != 0) {
                    GS::ce_huge_write_skips.fetch_add(1, std::memory_order_relaxed);
                }
                if (must_wait) {
                    GS::ce_wait_reads.fetch_add(use.read_later && !use.overflow,
                                                std::memory_order_relaxed);
                    GS::ce_wait_writes.fetch_add(use.written, std::memory_order_relaxed);
                    GS::ce_wait_overflow.fetch_add(use.overflow, std::memory_order_relaxed);
                    LogConstantDumpWait(dump_const->Address<VAddr>(), dump_const->Size(), use);
                }
                if (must_wait) {
                    const auto drain_start = now_if();
                    rasterizer->DrainDrawPipe();
                    GS::ce_dump_drains.fetch_add(1, std::memory_order_relaxed);
                    if (timed) {
                        GS::ce_dump_drain_ns.fetch_add(ns_since(drain_start),
                                                       std::memory_order_relaxed);
                    }
                }
            }
            const auto write_start = now_if();
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            if (timed) {
                GS::ce_dump_write_ns.fetch_add(ns_since(write_start), std::memory_order_relaxed);
            }
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while (cblock.ce_count >= cblock.de_count &&
                   (cblock.ce_count - cblock.de_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            SCOPE_EXIT {
                task.handle.destroy();
            };
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const auto* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            if (copy.guest && ccb.data() >= copy.words.data() &&
                ccb.data() < copy.words.data() + copy.words.size()) {
                // The game sets the valid bit in its own buffer later, not in our copy.
                rewind = reinterpret_cast<const PM4CmdRewind*>(copy.guest +
                                                               (ccb.data() - copy.words.data()));
            }
            while (!rewind->Valid()) {
                YIELD_CE();
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        if (time_ce_op) {
            const size_t index = static_cast<size_t>(opcode) & 0xff;
            cmd_frame.ce_op_ns[index] += CmdFrameTiming::Since(ce_op_start);
            ++cmd_frame.ce_op_count[index];
        }
        ccb = NextPacket(ccb, packet_words);
    }

    FIBER_EXIT;
}

// DmaData into or from memory and GDS (on the draw recorder with the threaded renderer).
static void ExecuteDmaData(Vulkan::Rasterizer& rasterizer, const PM4DmaData& dma) {
    const auto* dma_data = &dma;
    if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
        rasterizer.FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data, true);
    } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
               dma_data->dst_sel == DmaDataDst::Gds) {
        rasterizer.CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                              dma_data->NumBytes(), true, false);
    } else if (dma_data->src_sel == DmaDataSrc::Data &&
               (dma_data->dst_sel == DmaDataDst::Memory ||
                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
        rasterizer.FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(), dma_data->data,
                              false);
    } else if (dma_data->src_sel == DmaDataSrc::Gds &&
               (dma_data->dst_sel == DmaDataDst::Memory ||
                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
        rasterizer.CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                              dma_data->NumBytes(), false, true);
    } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
               (dma_data->dst_sel == DmaDataDst::Memory ||
                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
        rasterizer.CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->SrcAddress<VAddr>(),
                              dma_data->NumBytes(), false, false);
    } else {
        UNREACHABLE_MSG("DmaData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                        u32(dma_data->dst_sel));
    }
}

namespace {

struct PacketTrail {
    static constexpr u32 Capacity = 16;
    std::array<std::pair<u32, u32>, Capacity> packets{};
    u32 total{};

    void Add(u32 dword_offset, u32 header) {
        packets[total++ % Capacity] = {dword_offset, header};
    }
};

u64 HashCommands(std::span<const u32> words) {
    return XXH3_64bits(words.data(), words.size_bytes());
}

void LogBadPacket(std::span<const u32> commands, const u32* bad, u64 start_hash,
                  const PacketTrail& trail) {
    const u64 at = static_cast<u64>(bad - commands.data());
    const u64 first = at > 8 ? at - 8 : 0;
    const u64 last = std::min<u64>(commands.size(), at + 9);
    std::string around;
    for (u64 i = first; i < last; ++i) {
        if (i == at) {
            fmt::format_to(std::back_inserter(around), "[{:08x}] ", commands[i]);
        } else {
            fmt::format_to(std::back_inserter(around), "{:08x} ", commands[i]);
        }
    }
    const u32 shown = std::min(trail.total, PacketTrail::Capacity);
    std::string recent;
    for (u32 i = trail.total - shown; i < trail.total; ++i) {
        const auto [offset, raw] = trail.packets[i % PacketTrail::Capacity];
        const auto* header = reinterpret_cast<const PM4Header*>(&raw);
        const u32 type = header->type;
        if (type == 3) {
            const PM4ItOpcode opcode = header->type3.opcode;
            fmt::format_to(std::back_inserter(recent), "{}: {} ({:#x}, {} dwords); ", offset,
                           magic_enum::enum_name(opcode), static_cast<u32>(opcode),
                           header->type3.NumWords() + 1);
        } else {
            fmt::format_to(std::back_inserter(recent), "{}: type {} ({:08x}); ", offset, type, raw);
        }
    }
    LOG_CRITICAL(Render,
                 "Bad PM4 packet at dword {} of {} in the draw command buffer at {:#x}, header "
                 "{:08x} | the buffer {} since the submission started | words around it: {}| "
                 "last {} packets, oldest first: {}",
                 at, commands.size(), reinterpret_cast<uintptr_t>(commands.data()), *bad,
                 HashCommands(commands) == start_hash ? "is unchanged" : "CHANGED", around, shown,
                 recent);
}

// A bad packet means the decoder has lost its place in the buffer (or the buffer was changed under
// it): nothing after it can be trusted, so the rest of that command buffer is dropped.
void StopAtBadPacket(std::span<const u32> commands, std::span<const u32>& dcb, u64 start_hash,
                     const PacketTrail& trail, std::string_view what) {
    static std::atomic<u32> bad_buffers{0};
    const u32 seen = ++bad_buffers;
    if (seen <= 8) {
        LogBadPacket(commands, dcb.data(), start_hash, trail);
    }
    LOG_ERROR(Render, "{}: dropped the last {} of {} dwords of this command buffer ({} so far)",
              what, dcb.size(), commands.size(), seen);
    dcb = {};
}

} // namespace

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, CmdCopy copy) {
    FIBER_ENTER(dcb_task_name);

    auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    std::span<const u32> commands = dcb;
    u64 commands_hash = HashCommands(commands);
    PacketTrail trail;
    while (!dcb.empty()) {
        current_packet_opcode = ~0u;
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;
        trail.Add(static_cast<u32>(dcb.data() - commands.data()), header->raw);

        switch (type) {
        default:
            StopAtBadPacket(commands, dcb, commands_hash, trail,
                            fmt::format("Wrong PM4 type {}", type));
            continue;
        case 0:
            StopAtBadPacket(commands, dcb, commands_hash, trail,
                            fmt::format("PM4 type 0 packet (base reg {:#x}, {} dwords)",
                                        header->type0.base.Value(), header->type0.NumWords()));
            continue;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            current_packet_opcode = static_cast<u32>(opcode);
            // Stutter detail: this packet's time on the command thread, by type.
            const bool time_packet =
                Common::GuestStats::detail_timing.load(std::memory_order_relaxed);
            if (rasterizer && !KeepsDrawPipe(opcode)) {
                const auto drain_start = time_packet ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{};
                rasterizer->DrainDrawPipe(static_cast<u32>(opcode));
                if (time_packet) {
                    Common::GuestStats::cmd_drain_ns.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - drain_start)
                            .count(),
                        std::memory_order_relaxed);
                }
            }
            const auto packet_start = time_packet ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }
                const bool is_flip = nop->data_block[0] == PM4CmdNop::PayloadType::PatchedFlip;
                const auto flip_start = time_packet && is_flip
                                            ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
                if (rasterizer && (guest_markers_enabled || is_flip)) {
                    rasterizer->DrainDrawPipe();
                }
                const auto signal_start = time_packet && is_flip
                                              ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    cmd_frame.flip_seen = true;
                    if (time_packet) {
                        namespace GS = Common::GuestStats;
                        const auto end = std::chrono::steady_clock::now();
                        GS::flips.fetch_add(1, std::memory_order_relaxed);
                        GS::flip_drain_ns.fetch_add(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(signal_start -
                                                                                 flip_start)
                                .count(),
                            std::memory_order_relaxed);
                        GS::flip_signal_ns.fetch_add(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(end - signal_start)
                                .count(),
                            std::memory_order_relaxed);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                pipe_dirty.Reset();
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                pipe_dirty.Mark(reg_addr, count - 1);
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                pipe_dirty.Mark(reg_addr, count - 1);

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                        const u64 address = regs.depth_buffer.DepthAddress();
                        const auto found = std::ranges::find(db_hints, address, &DbHint::address);
                        auto& slot = found != db_hints.end()
                                         ? *found
                                         : db_hints[db_hint_next++ % db_hints.size()];
                        slot = {address, last_db_extent};
                        break;
                    }
                    last_db_extent.raw = 0;
                    if (EmulatorSettings.IsBbRealOcclusion()) {
                        // A depth buffer set up again without the size hint (the game's
                        // occlusion test boxes): the size it was last given, so the boxes test
                        // against the scene's depth and not a new, empty image of the whole
                        // allocation.
                        const u64 address = regs.depth_buffer.DepthAddress();
                        const auto found = std::ranges::find(db_hints, address, &DbHint::address);
                        if (address != 0 && found != db_hints.end()) {
                            last_db_extent = found->extent;
                        }
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                    pipe_dirty.Mark(Regs::ShRegWordOffset + set_data->reg_offset, count - 1);
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                            header + 2, (count - 1) * sizeof(u32));
                pipe_dirty.Mark(Regs::UconfigRegWordOffset + set_data->reg_offset, count - 1);
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                MarkRegWritten(regs.index_buffer_type);
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                MarkRegWritten(regs.max_index_size);
                MarkRegWritten(regs.index_base_address);
                MarkRegWritten(regs.num_indices);
                MarkRegWritten(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DrawIndex2", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(true); });
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                MarkRegWritten(regs.max_index_size);
                MarkRegWritten(regs.num_indices);
                MarkRegWritten(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexOffset2", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->Draw(true, draw_index_off->index_offset); });
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                MarkRegWritten(regs.num_indices);
                MarkRegWritten(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DrawIndexAuto", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(false); });
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndirectCountMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectCountMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirectCountMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            false, indirect_args_addr, offset, draw_indirect->stride,
                            draw_indirect->count, draw_indirect->count_addr,
                            draw_indirect->base_vtx_loc, draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                regs.max_index_size = index_buffer_num_indices;
                MarkRegWritten(regs.max_index_size);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_index_indirect->base_vtx_loc,
                                                 draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                regs.max_index_size = index_buffer_num_indices;
                MarkRegWritten(regs.max_index_size);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count, 0, draw_index_indirect->base_vtx_loc,
                            draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                regs.max_index_size = index_buffer_num_indices;
                MarkRegWritten(regs.max_index_size);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectCountMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count,
                            draw_index_indirect->count_indirect_enable.Value()
                                ? draw_index_indirect->count_addr
                                : 0,
                            draw_index_indirect->base_vtx_loc, draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DispatchDirect", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->DispatchDirect(); });
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DispatchIndirect", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->DispatchIndirect(indirect_args_addr, offset, size); });
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                MarkRegWritten(regs.num_instances);
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                MarkRegWritten(regs.index_base_address);
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                index_buffer_num_indices = index_size->num_indices;
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                    MarkRegWritten(regs.cp_strmout_cntl);
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        // Writes guest memory in order with the queued draws (the game clears
                        // each query's block with a DMA before the dump): run by the draw
                        // recorder after them, instead of waiting here for all of them.
                        struct Dump {
                            u64* results;
                            u64 value;
                            u32 pairs;
                        };
                        const Dump dump{event->Address<u64*>(),
                                        pixel_counter | OcclusionCounterValidMask,
                                        num_counter_pairs};
                        pixel_counter += OcclusionCounterStep;
                        const auto write = [](Vulkan::Rasterizer&, const u8* data) {
                            const auto& d = *reinterpret_cast<const Dump*>(data);
                            u64* results = d.results;
                            for (u32 i = 0; i < d.pairs; ++i, results += 2) {
                                *results = d.value;
                            }
                        };
                        if (rasterizer && EmulatorSettings.IsBbRealOcclusion()) {
                            // Real occlusion tests: written by the draw recorder with the GPU's
                            // count once the GPU has run the test (until then: not ready).
                            rasterizer->OcclusionDump(dump.results, dump.value, dump.pairs);
                        } else if (rasterizer && EmulatorSettings.IsBbOcclusionInOrder()) {
                            rasterizer->RunInOrder(
                                write, &dump, sizeof(dump), reinterpret_cast<VAddr>(dump.results),
                                u64(dump.pairs) * 2 * sizeof(u64), "occlusion result");
                        } else {
                            // Off: wait here for every queued draw, then write (as before).
                            if (rasterizer) {
                                rasterizer->DrainDrawPipe(static_cast<u32>(opcode));
                            }
                            u64* results = dump.results;
                            for (u32 i = 0; i < dump.pairs; ++i, results += 2) {
                                *results = dump.value;
                            }
                        }
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (rasterizer && event_eos->command != PM4CmdEventWriteEos::Command::GdsStore) {
                    // Threaded renderer: signalled in order with the queued draws.
                    const u64 packet = rasterizer->RunInOrder(
                        [](Vulkan::Rasterizer& r, const u8* data) {
                            r.OnFence();
                            reinterpret_cast<const PM4CmdEventWriteEos*>(data)->SignalFence(
                                [](void* address, u64 value, u32 num_bytes) {
                                    auto* memory = Core::Memory::Instance();
                                    ASSERT(memory->TryWriteBacking(address, &value, num_bytes));
                                });
                        },
                        event_eos, sizeof(PM4CmdEventWriteEos),
                        reinterpret_cast<VAddr>(event_eos->Address()), sizeof(u32),
                        "EventWriteEos fence");
                    NotePendingFence(packet, reinterpret_cast<VAddr>(event_eos->Address()),
                                     event_eos->DataDWord());
                    break;
                }
                if (rasterizer) {
                    rasterizer->DrainDrawPipe(static_cast<u32>(opcode));
                    rasterizer->OnFence();
                }
                event_eos->SignalFence([](void* address, u64 data, u32 num_bytes) {
                    auto* memory = Core::Memory::Instance();
                    ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                });
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        rasterizer->Finish();
                        const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                        *event_eos->Address() = value;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                if (rasterizer) {
                    // Threaded renderer: the fence (and its interrupt: flips, guest events) is
                    // signalled in order with the queued draws, by the draw recorder.
                    const u64 packet = rasterizer->RunInOrder(
                        [](Vulkan::Rasterizer& r, const u8* data) {
                            r.OnFence();
                            reinterpret_cast<const PM4CmdEventWriteEop*>(data)->SignalFence(
                                [](void* address, u64 value, u32 num_bytes) {
                                    auto* memory = Core::Memory::Instance();
                                    ASSERT(memory->TryWriteBacking(address, &value, num_bytes));
                                },
                                [] {
                                    Platform::IrqC::Instance()->Signal(
                                        Platform::InterruptId::GfxEop);
                                });
                        },
                        event_eop, sizeof(PM4CmdEventWriteEop),
                        reinterpret_cast<VAddr>(event_eop->Address<u8>()), sizeof(u64),
                        "EventWriteEop fence");
                    const auto address = reinterpret_cast<VAddr>(event_eop->Address<u8>());
                    switch (event_eop->data_sel.Value()) {
                    case DataSelect::Data32Low:
                        NotePendingFence(packet, address, event_eop->DataDWord());
                        break;
                    case DataSelect::Data64:
                        NotePendingFence(packet, address, event_eop->data_lo);
                        NotePendingFence(packet, address + 4, event_eop->data_hi);
                        break;
                    default:
                        break;
                    }
                    break;
                }
                event_eop->SignalFence(
                    [](void* address, u64 data, u32 num_bytes) {
                        auto* memory = Core::Memory::Instance();
                        ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                    },
                    [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                ASSERT(dma_data->command.das == 0);
                if (!Vulkan::Rasterizer::DrainForSpecialPackets()) {
                    // Threaded renderer: run by the draw recorder, in order with the draws.
                    const bool to_memory = dma_data->dst_sel == DmaDataDst::Memory ||
                                           dma_data->dst_sel == DmaDataDst::MemoryUsingL2;
                    rasterizer->RunInOrder(
                        [](Vulkan::Rasterizer& r, const u8* data) {
                            ExecuteDmaData(r, *reinterpret_cast<const PM4DmaData*>(data));
                        },
                        dma_data, sizeof(PM4DmaData), to_memory ? dma_data->DstAddress<VAddr>() : 0,
                        to_memory ? dma_data->NumBytes() : 0, "DmaData");
                    break;
                }
                rasterizer->DrainDrawPipe();
                ExecuteDmaData(*rasterizer, *dma_data);
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                if (!write_data->wr_one_addr.Value() && rasterizer) {
                    // Threaded renderer: written in order with the queued draws.
                    const u64 packet = rasterizer->RunInOrder(
                        [](Vulkan::Rasterizer& r, const u8* data) {
                            const auto* packet = reinterpret_cast<const PM4CmdWriteData*>(data);
                            const u32 bytes = (packet->header.count.Value() - 2) * 4;
                            r.OnFence();
                            std::memcpy(packet->Address<u64*>(), packet->data, bytes);
                        },
                        write_data, (header->type3.NumWords() + 1) * sizeof(u32),
                        reinterpret_cast<VAddr>(address), data_size, "WriteData");
                    for (u32 i = 0; i < std::min<u32>(data_size / 4, 4); ++i) {
                        NotePendingFence(packet, reinterpret_cast<VAddr>(address) + i * 4,
                                         write_data->data[i]);
                    }
                } else if (!write_data->wr_one_addr.Value()) {
                    std::memcpy(address, write_data->data, data_size);
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (mem_semaphore->IsSignaling()) {
                    if (rasterizer) {
                        rasterizer->DrainDrawPipe();
                    }
                    mem_semaphore->Signal();
                } else {
                    while (!mem_semaphore->Signaled()) {
                        if (rasterizer) {
                            rasterizer->DrainDrawPipe();
                        }
                        Common::GuestStats::cmd_state.store(
                            u8(Common::GuestStats::CmdState::GuestWait), std::memory_order_relaxed);
                        YIELD_GFX();
                        Common::GuestStats::cmd_state.store(u8(Common::GuestStats::CmdState::Work),
                                                            std::memory_order_relaxed);
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                if (copy.guest && dcb.data() >= copy.words.data() &&
                    dcb.data() < copy.words.data() + copy.words.size()) {
                    // The game sets the valid bit in its own buffer later, not in our copy.
                    rewind = reinterpret_cast<const PM4CmdRewind*>(
                        copy.guest + (dcb.data() - copy.words.data()));
                }
                while (!rewind->Valid()) {
                    rasterizer->DrainDrawPipe();
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (rasterizer && !wait_reg_mem->Test(regs.reg_array)) {
                    if (MetByPendingFence(*wait_reg_mem)) {
                        break;
                    }
                    rasterizer->DrainDrawPipe();
                }
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    Common::GuestStats::CmdStateScope state{
                        Common::GuestStats::CmdState::DisplayWait};
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    break;
                }
                const bool vo_label = vo_port->IsVoLabel(wait_addr);
                while (!wait_reg_mem->Test(regs.reg_array)) {
                    Common::GuestStats::cmd_state.store(
                        u8(vo_label ? Common::GuestStats::CmdState::DisplayWait
                                    : Common::GuestStats::CmdState::GuestWait),
                        std::memory_order_relaxed);
                    YIELD_GFX();
                    Common::GuestStats::cmd_state.store(u8(Common::GuestStats::CmdState::Work),
                                                        std::memory_order_relaxed);
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                if (indirect_buffer->chain) {
                    // A chained buffer replaces the current one: the CP never comes back to the
                    // rest of this buffer, which in reused segments still holds old commands.
                    dcb = {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size};
                    base_addr = reinterpret_cast<uintptr_t>(dcb.data());
                    // Bad packet reports describe the buffer being read now.
                    commands = dcb;
                    commands_hash = HashCommands(commands);
                    trail = {};
                    continue;
                }
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
                SCOPE_EXIT {
                    task.handle.destroy();
                };
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                // The game hands constants from the constant engine to draws thousands of times a
                // frame: run the oldest CE task here until it catches up, as before 0.21.0, instead
                // of a trip around the whole command loop for every hand-off. Yield only when no
                // CE work is queued or a CE step made no progress (it waits on this side).
                while (cblock.ce_count <= cblock.de_count) {
                    const auto ce_before = cblock.ce_count;
                    FIBER_EXIT;
                    ProcessCeQueue();
                    FIBER_ENTER(dcb_task_name);
                    if (cblock.ce_count == ce_before) {
                        YIELD_GFX();
                    }
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                const u32 buf_idx = strmout->buffer_select.Value();
                LOG_DEBUG(Render,
                          "IT_STRMOUT_BUFFER_UPDATE buffer_select = {}, source_select = {}, "
                          "update_memory = {}",
                          buf_idx, magic_enum::enum_name(strmout->source_select.Value()),
                          strmout->update_memory.Value());
                u32 offset = vgt_strmout_buffer_filled_size[buf_idx];
                switch (strmout->source_select.Value()) {
                case SourceSelect::BufferOffset:
                    offset = strmout->buffer_offset;
                    break;
                case SourceSelect::VgtStrmoutBufferFilledSize:
                    offset = vgt_strmout_buffer_filled_size[buf_idx];
                    break;
                case SourceSelect::SrcAddress: {
                    const VAddr src_addr = strmout->SrcAddress<VAddr>();
                    auto* memory = Core::Memory::Instance();
                    if (src_addr && memory && memory->IsValidMapping(src_addr, sizeof(offset))) {
                        std::memcpy(&offset, reinterpret_cast<const void*>(src_addr),
                                    sizeof(offset));
                    }
                    break;
                }
                case SourceSelect::None:
                    break;
                }
                if (strmout->source_select.Value() != SourceSelect::None) {
                    vgt_strmout_buffer_filled_size[buf_idx] = offset;
                }
                if (strmout->update_memory.Value()) {
                    const VAddr dst_addr = strmout->DstAddress<VAddr>();
                    auto* memory = Core::Memory::Instance();
                    if (dst_addr && memory && memory->IsValidMapping(dst_addr, sizeof(offset))) {
                        std::memcpy(reinterpret_cast<void*>(dst_addr),
                                    &vgt_strmout_buffer_filled_size[buf_idx], sizeof(offset));
                    }
                }
                regs.cp_strmout_cntl.offset_update_done = 1;
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                const auto* get_lod = reinterpret_cast<const PM4CmdGetLodStats*>(header);
                HandleGetLodStats(get_lod);
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                StopAtBadPacket(commands, dcb, commands_hash, trail,
                                fmt::format("Unknown PM4 type 3 opcode {:#x} with count {}",
                                            static_cast<u32>(opcode), count));
                continue;
            }
            if (time_packet) {
                const auto index = static_cast<size_t>(opcode) & 0xff;
                Common::GuestStats::cmd_opcode_ns[index].fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - packet_start)
                        .count(),
                    std::memory_order_relaxed);
                Common::GuestStats::cmd_opcode_count[index].fetch_add(1, std::memory_order_relaxed);
            }
            // The size read before running the packet: a flip or an EOP interrupt lets the guest
            // reuse this buffer while we are still on it, so the header may be gone by now.
            dcb = NextPacket(dcb, count + 1);
            break;
        }
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;

        // Threaded renderer: compute queues run with the draw recording thread idle.
        const bool time_packet = Common::GuestStats::detail_timing.load(std::memory_order_relaxed);
        if (rasterizer && opcode != PM4ItOpcode::Nop) {
            const auto drain_start = time_packet ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
            rasterizer->DrainDrawPipe(static_cast<u32>(opcode));
            if (time_packet) {
                Common::GuestStats::cmd_drain_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - drain_start)
                        .count(),
                    std::memory_order_relaxed);
            }
        }
        // Stutter detail: compute queue packets' time (an indirect buffer's are counted inside).
        const auto packet_start = time_packet ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            if constexpr (is_indirect) {
                if (indirect_buffer->chain) {
                    // As for graphics, a chained buffer replaces the current one.
                    acb = {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size};
                    base_addr = reinterpret_cast<VAddr>(acb.data());
                    acb_size = acb.size_bytes();
                    continue;
                }
            }
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            SCOPE_EXIT {
                task.handle.destroy();
            };
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            ASSERT(dma_data->command.das == 0);
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                u32(dma_data->dst_sel));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
                pipe_dirty.Mark(Regs::ShRegWordOffset + set_data->reg_offset,
                                static_cast<u32>(set_size / sizeof(u32)));
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchDirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchDirect(); });
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchIndirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchIndirect(ib_address, 0, size); });
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (!write_data->wr_one_addr.Value()) {
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
            } else {
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            while (!wait_reg_mem->Test(regs.reg_array)) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (rasterizer) {
                rasterizer->OnFence();
            }
            release_mem->SignalFence(
                [pipe_id = queue.pipe_id] {
                    Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
                },
                [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                    rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
                });
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        case PM4ItOpcode::GetLodStats: {
            const auto* get_lod = reinterpret_cast<const PM4CmdGetLodStats*>(header);
            HandleGetLodStats(get_lod);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        if (time_packet && opcode != PM4ItOpcode::IndirectBuffer) {
            Common::GuestStats::cmd_compute_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - packet_start)
                    .count(),
                std::memory_order_relaxed);
            Common::GuestStats::cmd_compute_packets.fetch_add(1, std::memory_order_relaxed);
        }
        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

void Liverpool::HandleGetLodStats(const PM4CmdGetLodStats* get_lod) {
    LOG_DEBUG(Render, "IT_GET_LOD_STATS: buffer = {:#x}, offset = {:#x}", get_lod->Address(),
              get_lod->DataOffset());
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::RetireFrames() {
    {
        std::scoped_lock lk{submit_mutex};
        if (frame_ends.empty()) {
            return;
        }
    }
    // The oldest submission not decoded yet, on any queue (queues finish out of order).
    u64 oldest_pending = ~0ull;
    for (u32 qid = 0; qid < num_mapped_queues; ++qid) {
        auto& queue = mapped_queues[qid];
        std::scoped_lock lock{queue.m_access};
        if (!queue.submit_seqs.empty()) {
            oldest_pending = std::min(oldest_pending, queue.submit_seqs.front());
        }
    }
    u64 frame = 0;
    {
        std::scoped_lock lk{submit_mutex};
        while (!frame_ends.empty() && frame_ends.front().first < oldest_pending) {
            frame = frame_ends.front().second;
            frame_ends.pop_front();
        }
    }
    if (frame == 0) {
        return;
    }
    // What the end of a frame and GPU idle did before: the frame's cache work, then everything
    // queued for the draw recorder recorded.
    if (rasterizer) {
        rasterizer->OnSubmit();
        rasterizer->Flush();
        rasterizer->DrainDrawPipe();
        if (VideoCore::IsCapturing()) {
            rasterizer->Flush(); // the frame's last work is in the capture too
        }
    }
    VideoCore::EndCapture();
    if (on_frame_retired) {
        on_frame_retired(frame);
    }
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    // Overlapped frames: the game may reuse a command buffer's memory once the next frame's
    // submissions start, which can be before the command thread has read it. Each submission gets
    // copies of its own then, each freed with the task that reads it: the dcb's with the graphics
    // task, the ccb's with the constant engine's, which runs on its own and can finish later.
    // (Indirect buffers they call are still read in place.)
    static const bool copy_for_overlap =
        EmulatorSettings.IsThreadedRenderer() && EmulatorSettings.IsBbOverlapFrames();
    CmdCopy dcb_copy{};
    CmdCopy ccb_copy{};
    if (copy_for_overlap) {
        dcb_copy.guest = dcb.data();
        dcb_copy.words.assign(dcb.begin(), dcb.end());
        dcb = {dcb_copy.words.data(), dcb_copy.words.size()};
        if (!ccb.empty()) {
            ccb_copy.guest = ccb.data();
            ccb_copy.words.assign(ccb.begin(), ccb.end());
            ccb = {ccb_copy.words.data(), ccb_copy.words.size()};
        }
    } else if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    auto task = ProcessGraphics(dcb, std::move(dcb_copy));
    u64 seq;
    {
        std::scoped_lock lk{submit_mutex};
        seq = ++submissions_total;
    }
    {
        std::scoped_lock lock{submit_mutex, queue.m_access};
        if (!ccb.empty()) {
            ce_submits.emplace(ProcessCeUpdate(ccb, std::move(ccb_copy)).handle);
            ++num_submits;
        }
        queue.submits.emplace(task.handle);
        queue.submit_seqs.emplace(seq);
        ++num_submits;
        work_retired = false;
        submit_cv.notify_one();
    }
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    const auto& task = ProcessCompute(acb, vqid);
    u64 seq;
    {
        std::scoped_lock lk{submit_mutex};
        seq = ++submissions_total;
    }
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
        queue.submit_seqs.emplace(seq);
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    work_retired = false;
    submit_cv.notify_one();
}

} // namespace AmdGpu
