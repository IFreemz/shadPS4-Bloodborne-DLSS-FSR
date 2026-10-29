// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "common/copy_pool.h"
#include "common/thread.h"

namespace Common::CopyPool {

namespace {

std::atomic<u32> requested_workers{0};

class Pool {
public:
    Pool() {
        // A few threads: memory bandwidth, not cores, limits copies, and the game needs its own.
        // Copy threads (setting): more, for traversal bursts of 200-440 MiB of new textures.
        const u32 hardware = std::max(1u, std::thread::hardware_concurrency());
        const u32 requested = requested_workers.load(std::memory_order_relaxed);
        const u32 workers = requested != 0
                                ? std::clamp(requested, 1u, hardware > 2 ? hardware - 2 : 1u)
                                : std::clamp(hardware / 4, 1u, 4u);
        for (u32 i = 0; i < workers; ++i) {
            threads.emplace_back([this](std::stop_token stop) { WorkerLoop(stop); });
        }
    }

    ~Pool() {
        {
            std::scoped_lock lock{mutex};
            stopping = true;
        }
        cv.notify_all();
        for (auto& thread : threads) {
            thread.request_stop();
        }
        threads.clear();
    }

    void Run(std::size_t count, const std::function<void(std::size_t)>& task) {
        std::unique_lock job{job_mutex, std::try_to_lock};
        if (!job || count < 2 || threads.empty()) {
            for (std::size_t i = 0; i < count; ++i) {
                task(i);
            }
            return;
        }
        {
            std::scoped_lock lock{mutex};
            job_task = &task;
            job_count = count;
            next.store(0, std::memory_order_relaxed);
            done.store(0, std::memory_order_relaxed);
            ++generation;
        }
        cv.notify_all();
        Work(task, count);
        while (done.load(std::memory_order_acquire) < count) {
            std::this_thread::yield();
        }
        // Workers that joined must be out of this job before the task goes away.
        std::unique_lock lock{mutex};
        job_task = nullptr;
        idle_cv.wait(lock, [this] { return active == 0; });
    }

    void Async(std::function<void()> task) {
        if (threads.empty()) {
            task();
            return;
        }
        async_pending.fetch_add(1, std::memory_order_relaxed);
        {
            std::scoped_lock lock{mutex};
            async_queue.push_back(std::move(task));
        }
        cv.notify_one();
    }

    bool WaitAsync() {
        if (async_pending.load(std::memory_order_acquire) == 0) {
            return false;
        }
        while (async_pending.load(std::memory_order_acquire) != 0) {
            // Helps with queued tasks instead of only waiting.
            std::function<void()> task;
            {
                std::scoped_lock lock{mutex};
                if (!async_queue.empty()) {
                    task = std::move(async_queue.front());
                    async_queue.pop_front();
                }
            }
            if (task) {
                task();
                async_pending.fetch_sub(1, std::memory_order_release);
            } else {
                std::this_thread::yield();
            }
        }
        return true;
    }

    bool AsyncPending() const {
        return async_pending.load(std::memory_order_acquire) != 0;
    }

private:
    void Work(const std::function<void(std::size_t)>& task, std::size_t count) {
        while (true) {
            const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= count) {
                return;
            }
            task(i);
            done.fetch_add(1, std::memory_order_release);
        }
    }

    void WorkerLoop(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:CopyPool");
        u64 seen = 0;
        while (true) {
            const std::function<void(std::size_t)>* task = nullptr;
            std::size_t count = 0;
            std::function<void()> async_task;
            {
                std::unique_lock lock{mutex};
                cv.wait(lock,
                        [&] { return stopping || generation != seen || !async_queue.empty(); });
                if (stopping || stop.stop_requested()) {
                    return;
                }
                if (generation != seen) {
                    // Parallel jobs first: their caller is waiting for them.
                    seen = generation;
                    if (job_task != nullptr) {
                        task = job_task;
                        count = job_count;
                        ++active;
                    }
                } else {
                    async_task = std::move(async_queue.front());
                    async_queue.pop_front();
                }
            }
            if (task) {
                Work(*task, count);
                {
                    std::scoped_lock lock{mutex};
                    --active;
                }
                idle_cv.notify_all();
            } else if (async_task) {
                async_task();
                async_pending.fetch_sub(1, std::memory_order_release);
            }
        }
    }

public:
    u32 Workers() const {
        return static_cast<u32>(threads.size());
    }

private:
    std::vector<std::jthread> threads;
    std::mutex job_mutex; ///< one job at a time; other callers copy inline
    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    const std::function<void(std::size_t)>* job_task = nullptr;
    std::size_t job_count = 0;
    u64 generation = 0;
    u32 active = 0;
    bool stopping = false;
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> done{0};
    std::deque<std::function<void()>> async_queue;
    std::atomic<u64> async_pending{0};
};

Pool& GetPool() {
    static Pool pool;
    return pool;
}

} // namespace

void ParallelFor(std::size_t count, const std::function<void(std::size_t)>& task) {
    GetPool().Run(count, task);
}

void SplitCopy(u64 size, const std::function<void(u64, u64)>& copy) {
    if (size < ParallelThreshold) {
        copy(0, size);
        return;
    }
    const std::size_t chunks = static_cast<std::size_t>((size + ChunkSize - 1) / ChunkSize);
    ParallelFor(chunks, [&](std::size_t i) {
        const u64 offset = i * ChunkSize;
        copy(offset, std::min(ChunkSize, size - offset));
    });
}

void Async(std::function<void()> task) {
    GetPool().Async(std::move(task));
}

bool WaitAsync() {
    return GetPool().WaitAsync();
}

bool AsyncPending() {
    return GetPool().AsyncPending();
}

u32 AutoWorkers() {
    const u32 hardware = std::max(1u, std::thread::hardware_concurrency());
    return std::clamp(hardware / 2, 2u, 8u);
}

u32 Configure(u32 workers) {
    requested_workers.store(workers, std::memory_order_relaxed);
    return GetPool().Workers();
}

} // namespace Common::CopyPool
