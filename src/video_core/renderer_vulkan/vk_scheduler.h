// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <source_location>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <queue>

#include "common/interval_set.h"
#include "common/polyfill_thread.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_primitive.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "vulkan/vulkan.hpp"

namespace tracy {
class VkCtxScope;
}

namespace Vulkan {

class Instance;

struct RenderAttachment {
    vk::ImageView image_view;
    vk::ImageLayout image_layout;
    std::array<u32, 4> clear_value;
    union {
        u32 is_clear;
        struct {
            bool has_depth;
            bool depth_clear;
            bool has_stencil;
            bool stencil_clear;
        };
    };
};
static_assert(std::has_unique_object_representations_v<RenderAttachment>);

struct RenderState {
    std::array<RenderAttachment, 8> color_attachments;
    RenderAttachment depth_stencil_attachment;
    u16 width;
    u16 height;
    u16 num_layers;
    u16 num_color_attachments;

    bool operator==(const RenderState& other) const noexcept {
        return std::memcmp(this, &other, sizeof(RenderState)) == 0;
    }
};
static_assert(std::has_unique_object_representations_v<RenderState>);

struct SubmitInfo {
    std::array<vk::Semaphore, 4> wait_semas;
    std::array<u64, 4> wait_ticks;
    std::array<vk::Semaphore, 4> signal_semas;
    std::array<u64, 4> signal_ticks;
    vk::Fence fence;
    u32 num_wait_semas;
    u32 num_signal_semas;

    void AddWait(vk::Semaphore semaphore, u64 tick = 1) {
        wait_semas[num_wait_semas] = semaphore;
        wait_ticks[num_wait_semas++] = tick;
    }

    void AddSignal(vk::Semaphore semaphore, u64 tick = 1) {
        signal_semas[num_signal_semas] = semaphore;
        signal_ticks[num_signal_semas++] = tick;
    }

    void AddSignal(vk::Fence fence) {
        this->fence = fence;
    }
};

using Viewports = boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS>;
using Scissors = boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS>;
using ColorWriteMasks = std::array<vk::ColorComponentFlags, AmdGpu::NUM_COLOR_BUFFERS>;
struct StencilOps {
    vk::StencilOp fail_op{};
    vk::StencilOp pass_op{};
    vk::StencilOp depth_fail_op{};
    vk::CompareOp compare_op{};

    bool operator==(const StencilOps& other) const {
        return fail_op == other.fail_op && pass_op == other.pass_op &&
               depth_fail_op == other.depth_fail_op && compare_op == other.compare_op;
    }
};
struct DynamicState {
    struct {
        bool viewports : 1;
        bool scissors : 1;

        bool depth_test_enabled : 1;
        bool depth_write_enabled : 1;
        bool depth_compare_op : 1;

        bool depth_bounds_test_enabled : 1;
        bool depth_bounds : 1;

        bool depth_bias_enabled : 1;
        bool depth_bias : 1;

        bool stencil_test_enabled : 1;
        bool stencil_front_ops : 1;
        bool stencil_front_reference : 1;
        bool stencil_front_write_mask : 1;
        bool stencil_front_compare_mask : 1;
        bool stencil_back_ops : 1;
        bool stencil_back_reference : 1;
        bool stencil_back_write_mask : 1;
        bool stencil_back_compare_mask : 1;

        bool primitive_restart_enable : 1;
        bool rasterizer_discard_enable : 1;
        bool cull_mode : 1;
        bool front_face : 1;

        bool blend_constants : 1;
        bool color_write_masks : 1;
        bool line_width : 1;
        bool feedback_loop_enabled : 1;
    } dirty_state{};

    Viewports viewports{};
    Scissors scissors{};

    bool depth_test_enabled{};
    bool depth_write_enabled{};
    vk::CompareOp depth_compare_op{};

    bool depth_bounds_test_enabled{};
    float depth_bounds_min{};
    float depth_bounds_max{};

    bool depth_bias_enabled{};
    float depth_bias_constant{};
    float depth_bias_clamp{};
    float depth_bias_slope{};

    bool stencil_test_enabled{};
    StencilOps stencil_front_ops{};
    u32 stencil_front_reference{};
    u32 stencil_front_write_mask{};
    u32 stencil_front_compare_mask{};
    StencilOps stencil_back_ops{};
    u32 stencil_back_reference{};
    u32 stencil_back_write_mask{};
    u32 stencil_back_compare_mask{};

    bool primitive_restart_enable{};
    bool rasterizer_discard_enable{};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};

    std::array<float, 4> blend_constants{};
    ColorWriteMasks color_write_masks{};
    float line_width{};
    bool feedback_loop_enabled{};

    /// Commits the dynamic state to the provided command buffer.
    void Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf);

    /// Invalidates all dynamic state to be flushed into the next command buffer.
    void Invalidate() {
        std::memset(&dirty_state, 0xFF, sizeof(dirty_state));
    }

    void SetViewports(const Viewports& viewports_) {
        if (!std::ranges::equal(viewports, viewports_)) {
            viewports = viewports_;
            dirty_state.viewports = true;
        }
    }

    void SetScissors(const Scissors& scissors_) {
        if (!std::ranges::equal(scissors, scissors_)) {
            scissors = scissors_;
            dirty_state.scissors = true;
        }
    }

    void SetDepthTestEnabled(const bool enabled) {
        if (depth_test_enabled != enabled) {
            depth_test_enabled = enabled;
            dirty_state.depth_test_enabled = true;
        }
    }

    void SetDepthWriteEnabled(const bool enabled) {
        if (depth_write_enabled != enabled) {
            depth_write_enabled = enabled;
            dirty_state.depth_write_enabled = true;
        }
    }

    void SetDepthCompareOp(const vk::CompareOp compare_op) {
        if (depth_compare_op != compare_op) {
            depth_compare_op = compare_op;
            dirty_state.depth_compare_op = true;
        }
    }

    void SetDepthBoundsTestEnabled(const bool enabled) {
        if (depth_bounds_test_enabled != enabled) {
            depth_bounds_test_enabled = enabled;
            dirty_state.depth_bounds_test_enabled = true;
        }
    }

    void SetDepthBounds(const float min, const float max) {
        if (depth_bounds_min != min || depth_bounds_max != max) {
            depth_bounds_min = min;
            depth_bounds_max = max;
            dirty_state.depth_bounds = true;
        }
    }

    void SetDepthBiasEnabled(const bool enabled) {
        if (depth_bias_enabled != enabled) {
            depth_bias_enabled = enabled;
            dirty_state.depth_bias_enabled = true;
        }
    }

    void SetDepthBias(const float constant, const float clamp, const float slope) {
        if (depth_bias_constant != constant || depth_bias_clamp != clamp ||
            depth_bias_slope != slope) {
            depth_bias_constant = constant;
            depth_bias_clamp = clamp;
            depth_bias_slope = slope;
            dirty_state.depth_bias = true;
        }
    }

    void SetStencilTestEnabled(const bool enabled) {
        if (stencil_test_enabled != enabled) {
            stencil_test_enabled = enabled;
            dirty_state.stencil_test_enabled = true;
        }
    }

    void SetStencilOps(const StencilOps& front_ops, const StencilOps& back_ops) {
        if (stencil_front_ops != front_ops) {
            stencil_front_ops = front_ops;
            dirty_state.stencil_front_ops = true;
        }
        if (stencil_back_ops != back_ops) {
            stencil_back_ops = back_ops;
            dirty_state.stencil_back_ops = true;
        }
    }

    void SetStencilReferences(const u32 front_reference, const u32 back_reference) {
        if (stencil_front_reference != front_reference) {
            stencil_front_reference = front_reference;
            dirty_state.stencil_front_reference = true;
        }
        if (stencil_back_reference != back_reference) {
            stencil_back_reference = back_reference;
            dirty_state.stencil_back_reference = true;
        }
    }

    void SetStencilWriteMasks(const u32 front_write_mask, const u32 back_write_mask) {
        if (stencil_front_write_mask != front_write_mask) {
            stencil_front_write_mask = front_write_mask;
            dirty_state.stencil_front_write_mask = true;
        }
        if (stencil_back_write_mask != back_write_mask) {
            stencil_back_write_mask = back_write_mask;
            dirty_state.stencil_back_write_mask = true;
        }
    }

    void SetStencilCompareMasks(const u32 front_compare_mask, const u32 back_compare_mask) {
        if (stencil_front_compare_mask != front_compare_mask) {
            stencil_front_compare_mask = front_compare_mask;
            dirty_state.stencil_front_compare_mask = true;
        }
        if (stencil_back_compare_mask != back_compare_mask) {
            stencil_back_compare_mask = back_compare_mask;
            dirty_state.stencil_back_compare_mask = true;
        }
    }

    void SetPrimitiveRestartEnabled(const bool enabled) {
        if (primitive_restart_enable != enabled) {
            primitive_restart_enable = enabled;
            dirty_state.primitive_restart_enable = true;
        }
    }

    void SetCullMode(const vk::CullModeFlags cull_mode_) {
        if (cull_mode != cull_mode_) {
            cull_mode = cull_mode_;
            dirty_state.cull_mode = true;
        }
    }

    void SetFrontFace(const vk::FrontFace front_face_) {
        if (front_face != front_face_) {
            front_face = front_face_;
            dirty_state.front_face = true;
        }
    }

    void SetBlendConstants(const std::array<float, 4> blend_constants_) {
        if (blend_constants != blend_constants_) {
            blend_constants = blend_constants_;
            dirty_state.blend_constants = true;
        }
    }

    void SetRasterizerDiscardEnabled(const bool enabled) {
        if (rasterizer_discard_enable != enabled) {
            rasterizer_discard_enable = enabled;
            dirty_state.rasterizer_discard_enable = true;
        }
    }

    void SetColorWriteMasks(const ColorWriteMasks& color_write_masks_) {
        if (!std::ranges::equal(color_write_masks, color_write_masks_)) {
            color_write_masks = color_write_masks_;
            dirty_state.color_write_masks = true;
        }
    }

    void SetLineWidth(const float width) {
        if (line_width != width) {
            line_width = width;
            dirty_state.line_width = true;
        }
    }

    void SetAttachmentFeedbackLoopEnabled(const bool enabled) {
        if (feedback_loop_enabled != enabled) {
            feedback_loop_enabled = enabled;
            dirty_state.feedback_loop_enabled = true;
        }
    }

    /// Clears the dirty flags exactly as Commit() would, without recording: the commands are
    /// recorded later from a copy taken before (threaded recording).
    void MarkCommitted(const Instance& instance);
};

/// Threaded command recording (after bbport): Vulkan commands passed to Scheduler::Record()
/// are stored as small closures in chunks and recorded into the command buffer by a worker
/// thread, in order. The thread that issues them only decides what to record.
class CommandRecorder {
public:
    CommandRecorder();
    ~CommandRecorder();

    CommandRecorder(const CommandRecorder&) = delete;
    CommandRecorder& operator=(const CommandRecorder&) = delete;

    template <typename Func>
    void Push(Func&& func) {
        using T = std::decay_t<Func>;
        static_assert(alignof(T) <= EntryAlign, "recorded command is over-aligned");
        constexpr u32 size =
            static_cast<u32>((sizeof(Entry) + sizeof(T) + EntryAlign - 1) & ~(EntryAlign - 1));
        static_assert(size <= ChunkCapacity, "recorded command is too large");
        if (!current || current->used + size > ChunkCapacity) {
            Kick(true);
            current = AcquireChunk();
        }
        u8* at = current->data + current->used;
        auto* entry = new (at) Entry{&RunEntry<T>, size};
        new (entry + 1) T(std::forward<Func>(func));
        current->used += size;
    }

    /// Hands the current chunk to the recording thread (when it holds enough, or `force`).
    void Kick(bool force);
    /// Waits until every pushed command is in the command buffer.
    void Sync();
    /// Nanoseconds the recording thread spent recording (statistics).
    std::atomic<s64> busy_ns{0};
    /// The command buffer the thread records into; only changed while synced.
    void SetCommandBuffer(vk::CommandBuffer cmdbuf_) {
        cmdbuf = cmdbuf_;
    }

private:
    static constexpr size_t EntryAlign = 16;
    static constexpr size_t ChunkCapacity = 64 * 1024;
    static constexpr size_t KickBytes = 8 * 1024;
    static constexpr u64 RingSize = 64;

    struct alignas(EntryAlign) Entry {
        void (*run)(void* payload, vk::CommandBuffer cmdbuf);
        u32 size;
    };
    struct Chunk {
        size_t used = 0;
        alignas(EntryAlign) u8 data[ChunkCapacity];
    };

    template <typename T>
    static void RunEntry(void* payload, vk::CommandBuffer cmdbuf) {
        auto* func = static_cast<T*>(payload);
        (*func)(cmdbuf);
        func->~T();
    }

    Chunk* AcquireChunk();
    void Run(std::stop_token stop);

    vk::CommandBuffer cmdbuf{};
    Chunk* current{};
    std::vector<std::unique_ptr<Chunk>> chunks;
    std::mutex free_mutex;
    std::vector<Chunk*> free_chunks;
    std::array<Chunk*, RingSize> ring{};
    alignas(64) std::atomic<u64> head{0}; ///< chunks handed over (producer)
    alignas(64) std::atomic<u64> tail{0}; ///< chunks recorded (recording thread)
    std::atomic<bool> sleeping{false};
    std::jthread thread;
};

using SessionFunc = Common::UniqueFunction<void>;
using SubmitFunc = Common::UniqueFunction<void, SubmitInfo&>;

class Scheduler {
public:
    explicit Scheduler(const Instance& instance);
    ~Scheduler();

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush(SubmitInfo& info);

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush();

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish(std::source_location where = std::source_location::current());

    /// Waits for the given tick to trigger on the GPU.
    void Wait(u64 tick, std::source_location where = std::source_location::current());

    /// Attempts to execute operations whose tick the GPU has caught up with.
    void PopPendingOperations();

    /// Starts a new rendering scope with provided state.
    void BeginRendering(const RenderState& new_state);
    static void RecordBeginRendering(vk::CommandBuffer cmdbuf, const RenderState& state);

    /// Ends current rendering scope.
    void EndRendering();

    /// Starts a new session.
    void BeginSession();

    /// Returns the current command buffer used for uploads.
    vk::CommandBuffer UploadCommandBuffer();

    /// Sets a function to be called on every session finalization.
    void SetSessionCallback(SessionFunc&& on_session) {
        this->on_session = std::move(on_session);
    }

    /// Sets a function to be called on every scheduler submission.
    void SetSubmitCallback(SubmitFunc&& on_submit) {
        this->on_submit = std::move(on_submit);
    }

    /// Returns the current render state.
    const RenderState& GetRenderState() const {
        return render_state;
    }

    /// Whether a render pass with exactly this state is open.
    [[nodiscard]] bool IsRenderingWith(const RenderState& state) const {
        return is_rendering && render_state == state;
    }

    /// Returns the current pipeline dynamic state tracking.
    DynamicState& GetDynamicState() {
        return dynamic_state;
    }

    /// Returns the current command buffer for recording on the calling thread. With threaded
    /// recording this first waits until every command passed to Record() is recorded, then
    /// records directly (Record() included) until the next KickRecording() or submission.
    vk::CommandBuffer CommandBuffer(
        std::source_location where = std::source_location::current()) const {
        if (recorder && !direct_mode) {
            NoteSync(where);
            recorder->Sync();
            direct_mode = true;
        }
        return sessions.back().primary;
    }

    /// Statistics: where recording had to wait for the recording thread (per call site), and
    /// how many commands went through Record() deferred or direct. Resets the counters.
    std::string TakeRecordingStats();

    /// Records `func(vk::CommandBuffer)` in order with the other commands. The closure must
    /// own everything it uses (capture by value): it may run later on the recording thread.
    template <typename Func>
    void Record(Func&& func) {
        if (capture) {
            // Captured for a later replay (velocity mirror batches), not recorded now.
            capture->emplace_back(std::decay_t<Func>(std::forward<Func>(func)));
            return;
        }
        if (!recorder || direct_mode) {
            if (recorder) {
                direct_records.fetch_add(1, std::memory_order_relaxed);
            }
            func(sessions.back().primary);
            return;
        }
        deferred_records.fetch_add(1, std::memory_order_relaxed);
        recorder->Push(std::forward<Func>(func));
    }

    /// While set, Record() appends to `out` instead of recording (see BbVelocityMirror).
    void BeginCapture(std::vector<Common::UniqueFunction<void, vk::CommandBuffer>>* out) {
        capture = out;
    }
    void EndCapture() {
        capture = nullptr;
    }

    /// Runs after a render pass ends (EndRendering), not recursively: the velocity mirror
    /// records its deferred draws there.
    void SetPassEndHook(Common::UniqueFunction<void> hook) {
        pass_end_hook = std::move(hook);
    }

    /// Starts threaded recording on this scheduler (Bloodborne threaded renderer).
    void EnableThreadedRecording();

    /// GPU time measurement (threaded renderer): each submission's execution time and the GPU
    /// idle time between submissions, from timestamps. Returns {busy ns, idle ns, submissions}
    /// since the last call.
    std::array<u64, 3> TakeGpuTiming();
    /// Timestamps for TakeGpuTiming only while diagnostics are on.
    void SetGpuTiming(bool enabled) {
        gpu_timing = enabled;
    }

    /// Lets Record() defer again; call where nobody holds a CommandBuffer().
    void KickRecording() {
        if (recorder) {
            recorder->Kick(false);
            direct_mode = false;
        }
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return work_semaphore.CurrentTick();
    }

    /// Returns true when a tick has been triggered by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) noexcept {
        if (work_semaphore.IsFree(tick)) {
            return true;
        }
        work_semaphore.Refresh();
        return work_semaphore.IsFree(tick);
    }

    /// Returns the scheduler timeline semaphore.
    [[nodiscard]] Semaphore* GetWorkSemaphore() noexcept {
        return &work_semaphore;
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Will be run when submitting or calling PopPendingOperations.
    void DeferOperation(Common::UniqueFunction<void>&& func) {
        std::unique_lock lk(pending_ops_mutex);
        pending_ops.emplace(std::move(func), CurrentTick());
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Runs as soon as possible in another thread.
    void DeferPriorityOperation(Common::UniqueFunction<void>&& func) {
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops.emplace(std::move(func), CurrentTick());
        }
        priority_pending_ops_cv.notify_one();
    }

    static std::mutex submit_mutex;
    static std::mutex present_mutex;

    /// An operation on the graphics queue (a submission, a sparse binding), called with
    /// submit_mutex held: run now, or, while a frame is presented with Keep rendering while
    /// presenting, right after the present returns, in order.
    static void QueueOp(Common::UniqueFunction<void>&& op);
    /// Keep rendering while presenting: the present runs without holding submit_mutex, so a
    /// frame limiter sleeping in it does not stop the GPU threads; their queue operations wait
    /// in order and run when it returns. Begin before the present call, End after it.
    static void BeginDeferringQueueOps();
    static void EndDeferringQueueOps();

private:
    void EndSession();

    void SubmitExecution(SubmitInfo& info);

    void PriorityPendingOpsThread(std::stop_token stoken);

    static std::mutex queue_ops_mutex;
    static bool deferring_queue_ops;
    static std::vector<Common::UniqueFunction<void>> deferred_queue_ops;

private:
    const Instance& instance;
    Semaphore work_semaphore;
    CommandPool command_pool;
    DynamicState dynamic_state;
    SessionFunc on_session{};
    SubmitFunc on_submit{};
    struct Session {
        vk::CommandBuffer upload{};
        vk::CommandBuffer primary{};
    };
    std::vector<Session> sessions;
    std::condition_variable_any event_cv;
    struct PendingOp {
        Common::UniqueFunction<void> callback;
        u64 gpu_tick;
    };
    std::queue<PendingOp> pending_ops;
    std::recursive_mutex pending_ops_mutex;
    std::queue<PendingOp> priority_pending_ops;
    std::mutex priority_pending_ops_mutex;
    std::condition_variable_any priority_pending_ops_cv;
    std::jthread priority_pending_ops_thread;
    RenderState render_state;
    bool is_rendering = false;
    tracy::VkCtxScope* profiler_scope{};
    std::unique_ptr<CommandRecorder> recorder;
    std::vector<Common::UniqueFunction<void, vk::CommandBuffer>>* capture{};
    Common::UniqueFunction<void> pass_end_hook;
    bool in_pass_end_hook = false;
    // GPU timing (see TakeGpuTiming): a timestamp pair per submission.
    static constexpr u32 TimerPairs = 512;
    vk::UniqueQueryPool timer_pool;
    double timer_period_ns = 1.0;
    u32 timer_slot = 0;
    bool timer_open = false;
    bool gpu_timing = false;
    struct PendingTiming {
        u64 tick;
        u32 slot;
    };
    std::deque<PendingTiming> pending_timings;
    u64 timer_last_end = 0;
    std::atomic<u64> gpu_busy_ns{0};
    std::atomic<u64> gpu_idle_ns{0};
    std::atomic<u64> gpu_submits{0};
    void CollectGpuTiming();
    mutable bool direct_mode = true;
    void NoteSync(const std::source_location& where) const;
    mutable std::mutex sync_stats_mutex;
    mutable std::unordered_map<u64, std::pair<std::string, u64>> sync_sites;
    mutable std::atomic<u64> deferred_records{0};
    mutable std::atomic<u64> direct_records{0};
};

} // namespace Vulkan
