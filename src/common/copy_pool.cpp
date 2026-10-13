// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "common/copy_pool.h"
#include "common/thread.h"

namespace Common::CopyPool {

namespace {

class Pool {
public:
    Pool() {
        // A few threads: memory bandwidth, not cores, limits copies, and the game needs its own.
        const u32 hardware = std::max(1u, std::thread::hardware_concurrency());
        const u32 workers = std::clamp(hardware / 4, 1u, 4u);
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
            const std::function<void(std::size_t)>* task;
            std::size_t count;
            {
                std::unique_lock lock{mutex};
                cv.wait(lock, [&] { return stopping || generation != seen; });
                if (stopping || stop.stop_requested()) {
                    return;
                }
                seen = generation;
                if (job_task == nullptr) {
                    continue;
                }
                task = job_task;
                count = job_count;
                ++active;
            }
            Work(*task, count);
            {
                std::scoped_lock lock{mutex};
                --active;
            }
            idle_cv.notify_all();
        }
    }

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

} // namespace Common::CopyPool
