// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: MIT
//
// The Win32 threading AMD's Vulkan frame generation swapchain (FidelityFX SDK v1.1.4) uses,
// on Linux: events, threads, critical sections and the performance counter.

#pragma once

#include <pthread.h>
#include <time.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

typedef int BOOL;
typedef uint64_t UINT64;
typedef void* HINSTANCE;
typedef void* HWND;
typedef void* HMONITOR;
typedef const wchar_t* LPCWSTR;
struct SECURITY_ATTRIBUTES;
typedef unsigned long DWORD;
typedef void* LPVOID;
typedef void* HANDLE;
typedef const void* LPSECURITY_ATTRIBUTES;
typedef DWORD (*LPTHREAD_START_ROUTINE)(LPVOID);

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#define WINAPI
#define INFINITE 0xFFFFFFFFul
#define WAIT_OBJECT_0 0ul
#define WAIT_TIMEOUT 258ul
#define WAIT_FAILED 0xFFFFFFFFul
#define THREAD_PRIORITY_HIGHEST 2
#ifndef TEXT
#define TEXT(text) text
#endif

typedef union _LARGE_INTEGER {
    struct {
        uint32_t LowPart;
        int32_t HighPart;
    };
    int64_t QuadPart;
} LARGE_INTEGER;

inline BOOL QueryPerformanceCounter(LARGE_INTEGER* count) {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    count->QuadPart = int64_t(now.tv_sec) * 1000000000 + now.tv_nsec;
    return TRUE;
}

inline BOOL QueryPerformanceFrequency(LARGE_INTEGER* frequency) {
    frequency->QuadPart = 1000000000;
    return TRUE;
}

struct CRITICAL_SECTION {
    std::recursive_mutex* mutex;
};

inline void InitializeCriticalSection(CRITICAL_SECTION* section) {
    section->mutex = new std::recursive_mutex;
}
inline BOOL InitializeCriticalSectionAndSpinCount(CRITICAL_SECTION* section, DWORD) {
    InitializeCriticalSection(section);
    return TRUE;
}
inline void EnterCriticalSection(CRITICAL_SECTION* section) {
    section->mutex->lock();
}
inline BOOL TryEnterCriticalSection(CRITICAL_SECTION* section) {
    return section->mutex->try_lock() ? TRUE : FALSE;
}
inline void LeaveCriticalSection(CRITICAL_SECTION* section) {
    section->mutex->unlock();
}
inline void DeleteCriticalSection(CRITICAL_SECTION* section) {
    delete section->mutex;
    section->mutex = nullptr;
}

namespace Win32Shim {

struct Object {
    virtual ~Object() = default;
    virtual DWORD Wait(DWORD milliseconds) = 0;
};

struct Event final : Object {
    std::mutex mutex;
    std::condition_variable signalled;
    bool manual_reset;
    bool state;

    Event(bool manual_reset_, bool initial_state) : manual_reset{manual_reset_}, state{initial_state} {}

    void Set() {
        {
            std::scoped_lock lock{mutex};
            state = true;
        }
        if (manual_reset)
            signalled.notify_all();
        else
            signalled.notify_one();
    }

    void Reset() {
        std::scoped_lock lock{mutex};
        state = false;
    }

    DWORD Wait(DWORD milliseconds) override {
        std::unique_lock lock{mutex};
        const auto ready = [this] { return state; };
        if (milliseconds == INFINITE)
            signalled.wait(lock, ready);
        else if (!signalled.wait_for(lock, std::chrono::milliseconds{milliseconds}, ready))
            return WAIT_TIMEOUT;
        if (!manual_reset)
            state = false;
        return WAIT_OBJECT_0;
    }
};

struct Thread final : Object {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable finished_cv;
    bool finished = false;

    Thread(LPTHREAD_START_ROUTINE routine, LPVOID parameter) {
        thread = std::thread([this, routine, parameter] {
            routine(parameter);
            std::scoped_lock lock{mutex};
            finished = true;
            finished_cv.notify_all();
        });
    }

    ~Thread() override {
        if (thread.joinable())
            thread.join();
    }

    DWORD Wait(DWORD milliseconds) override {
        {
            std::unique_lock lock{mutex};
            const auto done = [this] { return finished; };
            if (milliseconds == INFINITE)
                finished_cv.wait(lock, done);
            else if (!finished_cv.wait_for(lock, std::chrono::milliseconds{milliseconds}, done))
                return WAIT_TIMEOUT;
        }
        if (thread.joinable())
            thread.join();
        return WAIT_OBJECT_0;
    }
};

} // namespace Win32Shim

inline HANDLE CreateEvent(LPSECURITY_ATTRIBUTES, BOOL manual_reset, BOOL initial_state,
                          const void*) {
    return static_cast<Win32Shim::Object*>(new Win32Shim::Event(manual_reset, initial_state));
}
#define CreateEventA CreateEvent
#define CreateEventW CreateEvent

inline BOOL SetEvent(HANDLE event) {
    static_cast<Win32Shim::Event*>(static_cast<Win32Shim::Object*>(event))->Set();
    return TRUE;
}

inline BOOL ResetEvent(HANDLE event) {
    static_cast<Win32Shim::Event*>(static_cast<Win32Shim::Object*>(event))->Reset();
    return TRUE;
}

inline HANDLE CreateThread(LPSECURITY_ATTRIBUTES, size_t, LPTHREAD_START_ROUTINE routine,
                           LPVOID parameter, DWORD, DWORD*) {
    return static_cast<Win32Shim::Object*>(new Win32Shim::Thread(routine, parameter));
}

inline DWORD WaitForSingleObject(HANDLE handle, DWORD milliseconds) {
    return handle ? static_cast<Win32Shim::Object*>(handle)->Wait(milliseconds) : WAIT_FAILED;
}

inline BOOL CloseHandle(HANDLE handle) {
    delete static_cast<Win32Shim::Object*>(handle);
    return TRUE;
}

inline BOOL SetThreadPriority(HANDLE, int) {
    return TRUE;
}

inline long SetThreadDescription(HANDLE handle, const wchar_t* description) {
    auto* thread = static_cast<Win32Shim::Thread*>(static_cast<Win32Shim::Object*>(handle));
    char name[16]{};
    for (size_t i = 0; i + 1 < sizeof(name) && description[i]; ++i)
        name[i] = char(description[i]);
    pthread_setname_np(thread->thread.native_handle(), name);
    return 0;
}
