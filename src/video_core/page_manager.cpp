// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include "common/adaptive_mutex.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/error.h"
#include "common/game_waits.h"
#include "common/guest_stats.h"
#include "common/multi_level_page_table.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#else
#include <windows.h>
#endif

#ifdef __linux__
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers;
        u8 num_read_watchers;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            // Reads watched: no access at all (a page can't be write-only).
            if (num_read_watchers != 0) {
                return Core::MemoryPermission::None;
            }
            return ReadPerm() | WritePerm();
        }

        template <bool is_read>
        u8 GetPage() const {
            if constexpr (is_read) {
                return num_read_watchers;
            } else {
                return num_write_watchers;
            }
        }

        constexpr Core::MemoryPermission Update(PageOp write_op, bool update_write = true,
                                                PageOp read_op = PageOp::None,
                                                bool update_read = false) {
            if (update_read) {
                if (read_op == PageOp::Track) {
                    // (A count: the buffer cache's readbacks and occlusion answers can both
                    // watch a page's reads.)
                    ASSERT_MSG(num_read_watchers < 255, "Too many watchers");
                } else if (read_op == PageOp::Untrack) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                }
                num_read_watchers += std::to_underlying(read_op);
            }
            if (update_write) {
                if (write_op == PageOp::Track) {
                    ASSERT_MSG(num_write_watchers < 255, "Too many watchers");
                } else if (write_op == PageOp::Untrack) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                }
                num_write_watchers += std::to_underlying(write_op);
            }
            return Perms();
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / NUM_REGION_PAGES;
    inline static Vulkan::Rasterizer* rasterizer;

    Impl() = default;
    virtual ~Impl() = default;

    virtual void OnMap(VAddr address, size_t size) {
        // No-op
        EnsurePages(address, address + size);
    }

    virtual void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    virtual void Protect(VAddr address, size_t size, Core::MemoryPermission perms) = 0;

    void EnsurePages(VAddr begin, VAddr end) {
        end = std::min(end, VAddr{1} << ADDRESS_BITS) - 1;
        const size_t start_page = begin >> PM_PAGE_BITS;
        const size_t end_page = end >> PM_PAGE_BITS;
        cached_pages.reserve(start_page, end_page);
        locks.reserve(start_page, end_page);
    }

    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op, ProtectRanges* deferred) {
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        Core::MemoryPermission perms{};
        u64 range_begin = page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;
        // Windows' protection cost is per page: only runs of pages whose protection changes.
        const bool exact = EmulatorSettings.IsBbExactProtection();
        // Deferred protection: tracking only (a page made writable again is never left for
        // later).
        if (write_op != PageOp::Track) {
            deferred = nullptr;
        }

        const auto release_pending = [&] {
            if (range_pages > 0) {
                const VAddr range_addr = range_begin << PM_PAGE_BITS;
                const u64 range_size = range_pages << PM_PAGE_BITS;
                if (deferred) {
                    if (!deferred->empty() &&
                        deferred->back().first + deferred->back().second == range_addr) {
                        deferred->back().second += range_size;
                    } else {
                        deferred->emplace_back(range_addr, range_size);
                    }
                } else {
                    Protect(range_addr, range_size, perms);
                }
                range_pages = 0;
                potential_pages = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page_start << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
            EnsurePages(aligned_addr, aligned_end);
        }

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                continue;
            }

            locks[page].lock();

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const auto new_perms = state->Update(write_op);
            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (exact && new_perms == old_perms) {
                // Already right: not part of the next call (see bb_exact_protection).
                release_pending();
            } else if (range_pages != 0) {
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_pages = 1;
                }
                // Extend current range up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(page)) {
                lock->unlock();
            }
        }
    }

    bool UpdatePageReadWatcher(VAddr addr, PageOp read_op) {
        const u64 page = addr >> PM_PAGE_BITS;
        PageState* state = cached_pages.find(page);
        auto* lock = locks.find(page);
        if (!state || !lock) {
            return false;
        }
        lock->lock();
        const auto old_perms = state->Perms();
        const auto new_perms = state->Update(PageOp::None, false, read_op, true);
        if (new_perms != old_perms) {
            Protect(page << PM_PAGE_BITS, PM_PAGE_SIZE, new_perms);
        }
        lock->unlock();
        return true;
    }

    void ApplyProtection(VAddr addr, u64 size) {
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);
        // Under the pages' locks: the protection matches their states as they are now (a fault or
        // an untrack since the deferred tracking already changed them).
        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(page); lock && cached_pages.find(page)) {
                lock->lock();
            }
        }
        Core::MemoryPermission perms{};
        u64 run_begin = page_start;
        u64 run_pages = 0;
        const auto release = [&] {
            if (run_pages > 0) {
                Protect(run_begin << PM_PAGE_BITS, run_pages << PM_PAGE_BITS, perms);
                run_pages = 0;
            }
        };
        for (u64 page = page_start; page != page_end; ++page) {
            const PageState* state = cached_pages.find(page);
            if (!state || !locks.find(page)) {
                release();
                continue;
            }
            const auto page_perms = state->Perms();
            if (run_pages != 0 && page_perms != perms) {
                release();
            }
            if (run_pages == 0) {
                run_begin = page;
                perms = page_perms;
            }
            ++run_pages;
        }
        release();
        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(page); lock && cached_pages.find(page)) {
                lock->unlock();
            }
        }
    }

    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) {
        const u64 base_page = base_addr >> PM_PAGE_BITS;
        const u64 page_start = bounds.start_word * PAGES_PER_WORD + bounds.start_page;
        const u64 page_end = bounds.end_word * PAGES_PER_WORD + bounds.end_page + 1;

        Core::MemoryPermission perms{};
        u64 range_begin = base_page + page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;
        // Windows' protection cost is per page: only runs of pages whose protection changes.
        const bool exact = EmulatorSettings.IsBbExactProtection();

        const auto release_pending = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(base_page + page);
            if (!state) {
                continue;
            }

            locks[base_page + page].lock();

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const bool update_write = write_op != PageOp::None && write_mask.GetPage(page);
            const bool update_read = read_op != PageOp::None && read_mask.GetPage(page);
            const auto new_perms = state->Update(write_op, update_write, read_op, update_read);

            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (exact && new_perms == old_perms) {
                // Already right: not part of the next call (see bb_exact_protection).
                release_pending();
            } else if (range_pages != 0) {
                // If the protection did not change, extend the potential range
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_pages = 1;
                }
                // Extend current rango up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(base_page + page)) {
                lock->unlock();
            }
        }
    }

    struct PageTraits {
        using Entry = PageState;
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<PageTraits> cached_pages;
    struct MutexTraits {
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
        using Entry = Common::AdaptiveMutex;
#else
        using Entry = std::mutex;
#endif
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<MutexTraits> locks;
};

#ifdef __linux__
struct UffdImpl : public PageManager::Impl {
private:
    std::jthread ufd_thread;
    int uffd;

public:
    UffdImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;
        uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        if (uffd == -1) {
            LOG_ERROR(Common_Memory,
                      "userfaultfd syscall failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("userfaultfd");
        }

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        if (ret != 0) {
            LOG_ERROR(Common_Memory,
                      "uffdio_api call failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("uffdio_api");
        }

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    ~UffdImpl() = default;

    void OnMap(VAddr address, size_t size) override {
        PageManager::Impl::OnMap(address, size);
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed with error: {}", Common::GetLastErrorMsg());
    }

    void OnUnmap(VAddr address, size_t size) override {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed with error: {}", Common::GetLastErrorMsg());
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? UFFDIO_WRITEPROTECT_MODE_DONTWAKE : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        Common::SetCurrentThreadName("shadPS4:Uffd");

        auto regions = Core::Memory::Instance()->GetAddressSpace().GetUsableRegions();
        for (auto& region : regions) {
            OnMap(region.lower(), region.upper());
        }
        LOG_INFO(Common_Memory, "registered reserved memory with userfaultfd");

        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            if (readret == -1) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue;
                }
                LOG_ERROR(Common_Memory, "Unexpected result of uffd read: {}",
                          Common::GetLastErrorMsg());
                break;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            const auto ptid = msg.arg.pagefault.feat.ptid;
            rasterizer->InvalidateMemory(addr, 1,
                                         ptid == rasterizer->GetGpuCommandProcessorThreadId());

            // Some calls to InvalidateMemory never reach the UFFDIO_WRITEPROTECT ioctl in
            // ::Protect, therefore we use MODE_DONTWAKE and wake the thread with UFFDIO_WAKE here
            uffdio_range wake;
            wake.start = msg.arg.pagefault.address;
            wake.len = PageManager::PM_PAGE_SIZE;
            const int ret = ioctl(uffd, UFFDIO_WAKE, &wake);
            ASSERT_MSG(ret != -1, "Waking thread {} failed with: {}", ptid,
                       Common::GetLastErrorMsg());
        }
    }
};
#endif // __linux__

struct SignalImpl : public PageManager::Impl {
    SignalImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        const auto start = std::chrono::steady_clock::now();
        impl.Protect(address, size, perms);
        Common::GuestStats::protect_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                 start)
                .count(),
            std::memory_order_relaxed);
        Common::GuestStats::protect_calls.fetch_add(1, std::memory_order_relaxed);
        Common::GuestStats::protect_pages.fetch_add(size >> 12, std::memory_order_relaxed);
        if (Common::GuestStats::on_protect_helper) {
            namespace GS = Common::GuestStats;
            GS::helper_protect_calls.fetch_add(1, std::memory_order_relaxed);
            GS::helper_protect_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::steady_clock::now() - start)
                                                .count(),
                                            std::memory_order_relaxed);
        }
        if (Common::GuestStats::is_cmd_thread) {
            namespace GS = Common::GuestStats;
            GS::cmd_protect_calls.fetch_add(1, std::memory_order_relaxed);
            GS::cmd_protect_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - start)
                                             .count(),
                                         std::memory_order_relaxed);
        }
        if (Common::GuestStats::on_recorder) {
            namespace GS = Common::GuestStats;
            const auto by = static_cast<size_t>(GS::current_detail);
            GS::recorder_protect_calls_by[by].fetch_add(1, std::memory_order_relaxed);
            GS::recorder_protect_ns_by[by].fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count(),
                std::memory_order_relaxed);
            GS::recorder_protect_calls.fetch_add(1, std::memory_order_relaxed);
            GS::detail_ns[static_cast<size_t>(GS::Detail::RecorderProtect)].fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count(),
                std::memory_order_relaxed);
        }
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        const auto size = std::min<u64>(8, PageManager::GetNextPageAddr(addr) - addr);
        const auto is_gpu_thread = rasterizer->IsGpuSideThread();
        // Occlusion results kept from the game until answered (first: no other wait).
        if (rasterizer->OcclusionTrapFault(addr, Common::IsWriteError(context))) {
            return true;
        }
        if (Common::IsWriteError(context)) {
            const Common::GameWaits::Scope wait{"write fault"};
            return rasterizer->OnWriteFault(addr, size, is_gpu_thread);
        } else {
            const Common::GameWaits::Scope wait{"read fault"};
            return rasterizer->ReadMemory(addr, size, is_gpu_thread);
        }
        return false;
    }
};

struct PageManager::ProtectWorker {
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<std::pair<u64, ProtectRanges>> jobs;
    u64 queued{};
    std::atomic<u64> done{};
    std::jthread thread;
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_) {
#ifdef __linux__
    if (EmulatorSettings.IsUserfaultfdTracking()) {
        try {
            impl = std::make_unique<UffdImpl>(rasterizer_);
            LOG_INFO(Config, "Memory tracking method: userfaultfd");
            return;
        } catch (const std::runtime_error& e) {
            // if uffd is unsupported, falls back to SignalImpl
        }
    }
    LOG_INFO(Config, "Memory tracking method: signals");
#endif
    impl = std::make_unique<SignalImpl>(rasterizer_);
}

PageManager::~PageManager() = default;

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

void PageManager::UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op,
                                     ProtectRanges* deferred) const {
    impl->UpdatePageWatchers(addr, size, write_op, deferred);
}

bool PageManager::UpdatePageReadWatcher(VAddr addr, PageOp read_op) const {
    return impl->UpdatePageReadWatcher(addr, read_op);
}

void PageManager::ApplyProtection(VAddr addr, u64 size) const {
    impl->ApplyProtection(addr, size);
}

u64 PageManager::QueueProtection(ProtectRanges ranges) {
    if (!protect_worker) {
        protect_worker = std::make_unique<ProtectWorker>();
        protect_worker->thread = std::jthread([this](std::stop_token stop) {
            Common::SetCurrentThreadName("shadPS4:PageProtect");
            Common::GuestStats::on_protect_helper = true;
            auto& worker = *protect_worker;
            while (true) {
                std::pair<u64, ProtectRanges> job;
                {
                    std::unique_lock lock{worker.mutex};
                    worker.cv.wait(lock, stop, [&] { return !worker.jobs.empty(); });
                    if (worker.jobs.empty()) {
                        return; // stop requested
                    }
                    job = std::move(worker.jobs.front());
                    worker.jobs.pop_front();
                }
                for (const auto& [address, size] : job.second) {
                    impl->ApplyProtection(address, size);
                }
                worker.done.store(job.first, std::memory_order_release);
                worker.done.notify_all();
            }
        });
    }
    auto& worker = *protect_worker;
    u64 ticket;
    {
        std::scoped_lock lock{worker.mutex};
        ticket = ++worker.queued;
        worker.jobs.emplace_back(ticket, std::move(ranges));
    }
    worker.cv.notify_one();
    return ticket;
}

void PageManager::WaitProtection(u64 ticket) {
    if (!protect_worker) {
        return;
    }
    auto& done = protect_worker->done;
    for (u64 seen = done.load(std::memory_order_acquire); seen < ticket;
         seen = done.load(std::memory_order_acquire)) {
        done.wait(seen, std::memory_order_acquire);
    }
}

void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                              const RegionBits& write_mask,
                                              const RegionBits& read_mask, PageOp write_op,
                                              PageOp read_op) const {
    impl->UpdatePageWatchersForRegion(base_addr, bounds, write_mask, read_mask, write_op, read_op);
}

} // namespace VideoCore
