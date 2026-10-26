// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <source_location>
#include <unordered_map>

#include "common/guest_stats.h"
#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_bb_object_motion.h"
#include "video_core/renderer_vulkan/vk_bb_temporal_dlss.h"
#include "video_core/renderer_vulkan/vk_bb_velocity_mirror.h"
#include "video_core/renderer_vulkan/vk_draw_pipe.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class ComputePipeline;
class GraphicsPipeline;
class Runtime;

class Rasterizer {
public:
    /// The Bloodborne threaded renderer is in use.
    [[nodiscard]] bool ThreadedRendererActive() const noexcept {
        return draw_pipe != nullptr && pipe_enabled.load(std::memory_order_relaxed);
    }

    /// SHADPS4_PIPE_DRAINS=1: wait for the draw recorder before draws FilterDraw handles and
    /// before DmaData, as before (instead of running them in order on the recorder).
    static bool DrainForSpecialPackets();

    /// Once per game frame (the presenter, when the frame is prepared): stutter report.
    void OnGameFrame();

    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    BbTemporalDlss& GetTemporalDlss() {
        return temporal_dlss;
    }
    BbVelocityMirror& GetVelocityMirror() {
        return velocity_mirror;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
    bool OnWriteFault(VAddr addr, u64 size, bool assume_locks = false);
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    /// GPU command thread, after each decoded submission: early write-back (see
    /// BufferCache::SetEarlyWriteback) sends a batch that writes hot memory, and copies back
    /// the hot writes whose GPU work has finished.
    void OnSubmissionDecoded();
    void Finish();
    void OnSubmit();
    void OnFence();

    /// Threaded renderer: on the GPU command thread, waits until the draw recording thread
    /// has run every draw handed to it (vk_draw_pipe.h); a no-op on every other thread.
    /// `reason`: the PM4 opcode that drained, if any (statistics).
    void DrainDrawPipe(u32 reason = NoDrainReason,
                       std::source_location where = std::source_location::current());
    static constexpr u32 NoDrainReason = 0xFFFF;
    /// The GPU command thread or the draw recording thread.
    [[nodiscard]] bool IsGpuSideThread() const;

    /// Threaded renderer: runs `task` with a copy of `data` in order with the draws handed to
    /// the draw recording thread (on that thread), or here at once when none are. `write_addr`
    /// and `write_size`: guest memory the task writes, kept from early reads by the constant
    /// ring until it has run.
    using OrderedTask = void (*)(Rasterizer& rasterizer, const u8* data);
    /// Returns the number of the queued packet, or 0 when the task ran here.
    u64 RunInOrder(OrderedTask task, const void* data, u32 size, VAddr write_addr = 0,
                   u64 write_size = 0);
    /// Constant RAM dumps: what queued, not yet recorded work does with a guest range. Stage A
    /// only.
    struct QueuedUse {
        bool read_later; ///< a queued draw or dispatch reads a buffer there when recorded
        bool written;    ///< a queued task (DMA, fence) writes there
    };
    QueuedUse CheckQueuedUse(VAddr address, u64 size);
    /// Whether the draw recording thread has run the packet with this number.
    [[nodiscard]] bool DrawPipeRan(u64 packet) const {
        return !draw_pipe || draw_pipe->ConsumedPackets() >= packet;
    }

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    RenderState BeginRenderingFull(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();
    bool FilterDrawPasses() const;
    /// Stage A: true for a draw that cannot write anything (no pixel shader, no depth or
    /// stencil test, no memory writes in its shaders, no stream out), e.g. occlusion query
    /// geometry, whose results are not emulated. Such draws are skipped (setting).
    bool DrawWritesNothing(const GraphicsPipeline* pipeline);
    const GraphicsPipeline* no_write_pipeline{}; ///< stage A memo: last pipeline checked
    bool no_write_shaders{};
    u64 empty_draws_skipped{};   ///< stage A
    u64 depth_test_only_draws{}; ///< stage A: no pixel shader, depth/stencil read only
    u64 pipe_stats_empty_draws_skipped{};
    u64 pipe_stats_depth_test_only_draws{};

    // Threaded renderer (vk_draw_pipe.h).
    struct IndirectArgs {
        VAddr address;
        VAddr count_address;
        u32 offset;
        u32 stride;
        u32 max_count;
    };
    enum class PacketKind : u8 { Draw, DrawIndirect, Dispatch, DispatchIndirect, Task };
    bool UseDrawPipe() const;
    void QueueDraw(PacketKind kind, const Pipeline* pipeline, bool is_indexed, u32 index_offset,
                   const IndirectArgs& args);
    static void RunDrawPacket(void* context, const u8* data, u32 size);
    static void RunDrawPacketBody(Rasterizer& self, const u8* data, u32 size);
    /// "Detailed stutter reports" (with diagnostics): every draw timed by section, every packet
    /// timed by kind, and both shown for each stutter. Slower; for finding stutter causes.
    static bool HitchProfile();
    std::array<std::atomic<u64>, 5> packet_kind_ns{}; ///< draw recorder, by PacketKind
    void DrawRecord(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset);
    void DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed,
                            const IndirectArgs& args);
    void DispatchRecord(const ComputePipeline* pipeline);
    void DispatchIndirectRecord(const ComputePipeline* pipeline, const IndirectArgs& args);
    void LogDrawPipeStats();
    /// Logs a game frame that took noticeably longer than usual, with what happened during it.
    void ReportHitch();
    /// Every 10 s with diagnostics on: where VRAM goes (images, buffers, the rest).
    void LogVramReport();
    /// GPU profile (vk_gpu_profiler.h): a timestamp where a render pass or dispatch starts.
    void MarkPass(const GraphicsPipeline* pipeline, const RenderState& state);
    void MarkDispatch(const ComputePipeline* pipeline);
    struct HitchSnapshot {
        std::chrono::steady_clock::time_point time{};
        std::chrono::nanoseconds busy{}, drain{};
        u64 packets{}, gpu_wait_ns{}, gpu_waits{}, pipelines{}, background_pipelines{},
            background_shaders{}, background_skips{}, background_wait_ns{}, arena_binds{},
            protect_ns{}, protect_calls{}, faults{};
        std::array<u64, static_cast<size_t>(Common::GuestStats::CmdState::Count) * 2> states{};
        u64 present_acquire_ns{}, present_queue_ns{}, present_frame_wait_ns{}, presents{},
            queue_ops_behind_present{}, gpu_busy_ns{}, gpu_idle_ns{}, gpu_submits{},
            queue_submit_ns{};
        u64 mutex_waits{}, sleeps{}, sleep_ns{}, uploads{}, upload_bytes{}, upload_ns{},
            image_copy_ns{}, buffer_bytes{}, buffer_ns{}, resident_ns{}, staging_count{},
            staging_bytes{}, staging_ns{}, async_copies{}, async_wait_ns{}, file_opens{},
            resident_bytes{}, diag_ns{};
        std::array<u64, 4> housekeeping{};
        std::array<s64, 20> sections{}; ///< draw recorder profile (Section), HitchProfile
        std::array<u64, 5> packet_kinds{};
        std::array<u64, static_cast<size_t>(Common::GuestStats::Detail::Count)> details{};
        u64 recorder_protect_calls{};
        VideoCore::TextureCache::CleanupCounters cleanup{};
    };
    HitchSnapshot hitch_last{};
    /// Diagnostics setting (statistics, time breakdown, hitch reports, draw profile).
    std::atomic<bool> diagnostics{false};
    /// Samples (every ~250 us) of the command thread's state x draw recorder busy/idle.
    std::array<std::atomic<u64>, static_cast<size_t>(Common::GuestStats::CmdState::Count) * 2>
        state_samples{};
    std::jthread state_sampler;
    double hitch_avg_ms{};
    u64 hitch_frames{};
    u32 hitch_run{};                      ///< slow frames in a row
    std::array<s64, 4> housekeeping_ns{}; ///< previous frame: buffer tick, downloads, GC, runtime
    std::array<std::atomic<u64>, 4> housekeeping_total{}; ///< the same, added up
    std::atomic<u64> diag_log_ns{}; ///< time spent writing the 10 s statistics and VRAM report
    /// Registers, render target size hints and compute registers of the draw being recorded:
    /// the draw recording thread's copy there, the live ones elsewhere.
    const AmdGpu::Regs& Regs() const;
    AmdGpu::CbDbExtent CbExtent(u32 cb) const;
    AmdGpu::CbDbExtent DbExtent() const;
    const AmdGpu::ComputeProgram& CsRegs() const;

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    void BindSamplers(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0, bool is_indirect = false);

    void ResetBindings(bool is_compute);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    BbVelocityMirror velocity_mirror;
    BbTemporalDlss temporal_dlss;
    std::unique_ptr<BbObjectMotion> object_motion; // character motion vectors (temporal DLSS)
    std::array<float, 2> draw_jitter{}; // viewport offset of the current draw (temporal DLSS)
    void PrepareObjectMotion(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset,
                             u32 vertex_offset, u32 instance_offset);
    /// Character motion inputs of a skinned draw that need no history: worked out on stage A
    /// with the threaded renderer, so the draw recorder only does the bookkeeping.
    struct MotionPrep {
        u64 geometry; ///< hash of the vertex stream V#s
        u64 palette;  ///< hash of small bone palettes (gated draws)
        VAddr index_address;
        u32 vertex_offset, instance_offset;
        u32 index_size, num_indices, num_instances;
        u64 topology;              ///< hash of the index list
        Motion::VertexRange range; ///< vertices the draw references
        bool gated;                ///< no character-size skeleton: history only while bones move
        bool valid;                ///< false: no motion for this draw (still, or no usable indices)
    };
    MotionPrep ComputeMotionPrep(const GraphicsPipeline* pipeline, bool is_indexed,
                                 u32 index_offset);
    // Character motion state of whichever thread prepares draws (stage A when threaded): the
    // last bone palette per small-skeleton draw, and the index ranges.
    struct MotionGate {
        u64 palette;
        u64 frame; ///< submission (game frame) it was last seen in
    };
    tsl::robin_map<u64, MotionGate> motion_gate;
    tsl::robin_map<u64, u32> motion_gate_counts; ///< draws per identity this frame
    u64 motion_gate_frame{};
    Motion::IndexRangeCache motion_index_ranges;
    u64 motion_preps{};
    MotionPrep packet_motion_storage{};
    const MotionPrep* packet_motion{}; ///< stage B: the current packet's, if prepared
    void TemporalDlssDraw(const GraphicsPipeline* pipeline, bool indirect);
    void CopyInsteadOfDraw(const BbTemporalDlss::DrawReplacement& source);
    void ReplayDisplayCopy(const GraphicsPipeline* pipeline, const RenderState& state,
                           const std::function<void()>& draw);
    void ReplayVelocityMirror(const GraphicsPipeline* pipeline, const RenderState& state,
                              bool is_indexed, const std::function<void()>& draw);
    /// Records the current draw's vertex input, index buffer, descriptors and full dynamic
    /// state (with the given viewport/scissor) from the values it last bound.
    void RecordDrawBindings(const GraphicsPipeline* pipeline, bool is_indexed,
                            const Viewports* viewports, const Scissors* scissors);
    void RebindAfterMirror(const GraphicsPipeline* pipeline, const RenderState& state,
                           bool is_indexed, u64 flushes_before);
    void RecordVertexBindings();
    void RecordIndexBinding();
    struct LastVertexBind {
        bool dynamic = false;
        VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
        VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
        u32 num_buffers = 0;
        VertexInputs<vk::Buffer> buffers;
        VertexInputs<vk::DeviceSize> offsets;
        VertexInputs<vk::DeviceSize> sizes;
        VertexInputs<vk::DeviceSize> strides;
    } last_vertex;
    struct LastIndexBind {
        vk::Buffer handle;
        u64 offset = 0;
        u32 size = 0;
        vk::IndexType type = vk::IndexType::eUint16;
    } last_index;
    bool mirror_rebind = false;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;

    struct ImageBinding {
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<ImageBinding, Shader::NUM_IMAGES> image_bindings;
    std::array<ImageBinding, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    ImageBinding db_desc;

    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    // Index in image_infos of each shader's first texture, for the RenoDX display copy replay.
    boost::container::static_vector<std::pair<u64, u32>, 4> first_texture_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;

    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        vk::AccessFlags2 src_access;
    };
    std::vector<BoundBuffer> bound_buffers;
    std::vector<VideoCore::ImageId> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    bool attachment_feedback_loop{};
    bool needs_barrier{};

    // Threaded renderer state. The draw recording thread's copies below are only written by it,
    // or by the GPU command thread while it is idle. alignas(64) in this class keeps fields
    // the two threads write every draw on separate cache lines.
    std::unique_ptr<AmdGpu::Regs> pipe_regs;
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> pipe_cb_extent{};
    AmdGpu::CbDbExtent pipe_db_extent{};
    AmdGpu::ComputeProgram pipe_cs{};
    alignas(64) bool pipe_synced{};   ///< stage A from here (draw recorder fields above)
    std::atomic<bool> pipe_enabled{}; // read by the presentation thread
    std::vector<u16> pipe_blocks;
    std::chrono::steady_clock::time_point pipe_stats_time{};
    std::chrono::steady_clock::time_point vram_report_time{};
    u64 pipe_stats_packets{};
    u64 pipe_stats_drains{};
    std::chrono::nanoseconds pipe_stats_drain_time{};
    std::chrono::nanoseconds pipe_stats_busy{};
    struct DrainSite {
        const char* function;
        u32 line;
        u32 reason;
        u64 count;
        std::chrono::nanoseconds time;
    };
    std::unordered_map<u64, DrainSite> pipe_drain_sites;

    // Render state memo (after bbport): most draws continue the render pass the previous draw
    // opened, with the same targets; such a draw reuses that draw's render state instead of
    // looking up the target views, metadata and layouts again.
    struct TargetSignature {
        std::array<VideoCore::ImageId, AmdGpu::NUM_COLOR_BUFFERS + 1> ids{};
        std::array<VideoCore::ImageViewInfo, AmdGpu::NUM_COLOR_BUFFERS + 1> views{};
        std::array<u8, AmdGpu::NUM_COLOR_BUFFERS> color_samples{};
        u32 mrt_mask{};
        u32 depth_control{};
        bool depth_valid{};
        bool stencil_valid{};
        bool operator==(const TargetSignature&) const = default;
    };
    TargetSignature MakeTargetSignature(const GraphicsPipeline* pipeline) const;
    bool TargetMemoUsable(const TargetSignature& signature);
    struct TargetMemo {
        bool valid{};
        TargetSignature signature;
        RenderState state;
    } target_memo;
    std::atomic<bool> target_memo_enabled{};
    std::atomic<u64> target_memo_hits{};
    std::atomic<u64> target_memo_misses{};
    u64 pipe_stats_memo_hits{};
    u64 pipe_stats_memo_misses{};

    // Image lookup memos (after bbport): FindImage walks the texture cache's page table for
    // every texture and render target of every draw. Its result for the same descriptor holds
    // while no image is registered or unregistered (TextureCache::RegistryGeneration).
    using ImageDesc = VideoCore::TextureCache::ImageDesc;
    struct TextureMemoEntry {
        std::array<u64, 4> sharp{};
        u32 flags{};
        bool valid{};
        u64 generation{};
        VideoCore::ImageId image_id{};
        ImageDesc desc{};
    };
    static constexpr u32 TextureMemoSize = 4096;
    std::unique_ptr<std::array<TextureMemoEntry, TextureMemoSize>> texture_memo;
    // Texture set memo (after bbport): a stage's resolved sampled textures (image after the
    // depth redirect, view, backing, view info) remembered by its T#s. A hit only redoes the
    // per-draw effects (found tick, binding flags, layout transition, usage) and writes the
    // descriptors: no image lookup, view search or image update per texture.
    struct TextureSetEntry {
        VideoCore::ImageId id{}; ///< null: null descriptor
        vk::ImageView view{};
        const void* backing{};
        VideoCore::ImageViewInfo view_info{};
    };
    struct TextureSet {
        static constexpr u32 MaxImages = 16; ///< larger sets are not remembered
        u64 key{};
        const Shader::Info* stage{};
        u64 generation{~0ull};
        u32 count{};
        std::array<std::array<u64, 4>, MaxImages> sharps{};
        std::array<TextureSetEntry, MaxImages> entries{};
    };
    // Direct-mapped by key: with ~2300 sets a frame (4700 draws, BB PC Remaster) 4096 slots lost
    // ~40% to two sets sharing a slot (71% reused); 16384 keep collisions near 13%. Only the
    // slots in use are touched, so the larger table costs memory (~21 MiB), not cache.
    static constexpr u32 TextureSetCount = 16384;
    std::unique_ptr<std::array<TextureSet, TextureSetCount>> texture_sets;
    std::atomic<u64> texture_set_hits{};
    std::atomic<u64> texture_set_misses{};
    u64 pipe_stats_set_hits{};
    u64 pipe_stats_set_misses{};
    struct TargetLookupMemo {
        std::array<u32, 32> key{};
        u32 key_words{};
        bool valid{};
        u64 generation{};
        VideoCore::ImageId image_id{};
        ImageDesc desc{};
    };
    std::array<TargetLookupMemo, AmdGpu::NUM_COLOR_BUFFERS + 1> target_lookup_memo;
    VideoCore::ImageId FindTargetImage(u32 slot, ImageDesc& desc, const void* key, u32 key_bytes);
    // Sampler memo: GetSampler hashes the S# and takes a lock for every sampler of every draw.
    struct SamplerMemoEntry {
        std::array<u64, 2> sharp{};
        bool is_depth{};
        float lod_bias{};
        bool valid{};
        u64 generation{};
        u64 touched_gc_tick{};
        vk::Sampler handle{};
    };
    static constexpr u32 SamplerMemoSize = 1024;
    std::unique_ptr<std::array<SamplerMemoEntry, SamplerMemoSize>> sampler_memo;
    alignas(64) std::atomic<u64> write_faults{}; ///< guest fault handlers
    alignas(64) u64 stat_draws{};                ///< stage A: draws seen (queued or run here)
    u64 stat_frames{};                           ///< stage A: submissions done (one per game frame)
    u64 pipe_stats_draws{};
    u64 pipe_stats_frames{};

public:
    /// Real occlusion tests (GPU command thread): a counter dump the game asked for.
    void OcclusionDump(u64* results, u64 value, u32 pairs);

private:
    struct OcclusionTask {
        u64* results;
        u64 begin_value;
        u32 pairs;
        u32 stage_a_draws;
        bool begin;
    };
    struct {
        VAddr begin_address{};
        u64 begin_value{};
        u32 draws{};
        bool open{};
    } occlusion_a; ///< GPU command thread
    struct {
        bool open{};
        bool dropped{};
        u32 pairs{};
        u32 draws_seen{};
        u64* results{};
        u64 begin_value{};
        boost::container::small_vector<u32, 4> queries;
    } occlusion_b; ///< draw recorder
    struct PendingOcclusion {
        u64* results;
        u64 begin_value;
        u32 pairs;
        u64 tick;
        boost::container::small_vector<u32, 4> queries;
        bool known;
    };
    std::deque<PendingOcclusion> occlusion_pending;
    static constexpr u32 OcclusionPoolSize = 4096;
    static constexpr u32 NoOcclusionQuery = ~0u;
    vk::UniqueQueryPool occlusion_pool;
    std::vector<bool> occlusion_busy;
    u32 occlusion_next{};
    std::atomic<u64> occlusion_tests{};
    std::atomic<u64> occlusion_answered{};
    std::atomic<u64> occlusion_hidden{};
    std::atomic<u64> occlusion_unknown{};
    u32 occlusion_poll_counter{};
    std::array<u64, 5> pipe_stats_occlusion{};
    void OcclusionTaskRun(const OcclusionTask& task);
    bool OcclusionRedirectBegin();
    void OcclusionRedirectEnd();
    void OcclusionMarkRedirected();
    struct {
        bool valid{};
        AmdGpu::DepthBuffer depth_buffer{};
        AmdGpu::DepthView depth_view{};
        AmdGpu::Address htile{};
        AmdGpu::CbDbExtent extent{};
    } scene_depth; ///< GPU command thread
    struct {
        AmdGpu::DepthBuffer depth_buffer{};
        AmdGpu::DepthView depth_view{};
        AmdGpu::Address htile{};
        AmdGpu::CbDbExtent extent{};
        std::array<AmdGpu::ViewportBounds, AmdGpu::NUM_VIEWPORTS> viewports{};
        AmdGpu::Scissor screen_scissor{};
        AmdGpu::ViewportScissor window_scissor{};
        AmdGpu::ViewportScissor generic_scissor{};
        std::array<AmdGpu::ViewportScissor, AmdGpu::NUM_VIEWPORTS> viewport_scissors{};
    } occlusion_saved;
    std::atomic<u64> occlusion_redirects{};
    void OcclusionClose(u32 stage_a_draws);
    u32 OcclusionQueryBegin();
    void OcclusionQueryEnd(u32 query);
    void PollOcclusion();

    u64 pipe_stats_faults{};
    alignas(64) std::atomic<u64> image_memo_hits{}; ///< draw recorder
    std::atomic<u64> image_memo_misses{};
    u64 pipe_stats_image_hits{};
    u64 pipe_stats_image_misses{};

    // Constant ring (after bbport): the GPU command thread copies the small read-only buffers
    // of a draw (the stream path of BufferCache::ObtainBuffer, and the flattened user data)
    // into a ring of its own when it queues the draw; the draw recorder only binds them.
    struct PrefetchedBuffer {
        u8 stage;  ///< ordinal among the pipeline's present stages
        u8 buffer; ///< index in the stage's buffer list
        u16 pad;
        u32 size;
        u64 offset; ///< in const_ring
    };
    static constexpr u64 ConstRingSize = 32ull << 20;
    std::unique_ptr<VideoCore::Buffer> const_ring;
    /// Stage A: next position (grows forever; offset = position % size).
    alignas(64) u64 ring_head{};
    u64 ring_free_until{}; ///< stage A: positions below this are no longer read by the GPU
    struct RingRetire {
        u64 end;
        u64 tick;
    };
    static constexpr u32 RingRetireSize = 4096;
    std::array<RingRetire, RingRetireSize> ring_retire{};
    // Fields of the two threads on separate cache lines: a write by one thread to a line the
    // other reads every draw costs both a cache miss.
    alignas(64) std::atomic<u64> ring_retire_w{}; ///< written by stage B
    u64 ring_b_tick{}; ///< stage B: command buffer of its latest ring bytes
    u64 ring_b_end{};  ///< stage B: ring position its recorded draws reach
    alignas(64) std::atomic<u64> ring_retire_r{}; ///< written by stage A
    /// Stage A: guest ranges that queued, not yet recorded draws write (storage buffers): a
    /// prefetch overlapping one would read the memory before those writes.
    struct PendingWrite {
        u64 packet;
        VAddr begin;
        VAddr end;
    };
    std::deque<PendingWrite> pending_writes;
    /// Guest ranges queued packets read buffers from when recorded (not copied ahead).
    std::deque<PendingWrite> queued_later_reads;
    bool queued_reads_overflow{};
    void NoteQueuedReads(const Pipeline* pipeline);
    std::vector<PrefetchedBuffer> prefetch_scratch;
    alignas(64) const PrefetchedBuffer* packet_prefetch{}; ///< stage B: the current packet's
    u32 packet_num_prefetch{};
    u32 bind_stage_ordinal{};
    u32 prefetch_cursor{}; ///< stage B: next entry of packet_prefetch (same order as binding)
    alignas(64) std::atomic<u64> ring_prefetches{}; ///< stage A
    std::atomic<u64> ring_fallbacks{};
    u64 pipe_stats_ring_prefetches{};
    u64 pipe_stats_ring_fallbacks{};
    u8* RingAlloc(u64 size, u64 alignment, u64& offset);
    /// Stage A: takes in what the draw recorder reports as no longer read by the GPU.
    void RingRetire();
    void PrefetchBuffers(const Pipeline* pipeline);
    /// Stage A: small or CPU-changed vertex/index data (what stage B would copy to the stream
    /// buffer) copied into the constant ring now. Returns the ring offset, or NoRing.
    static constexpr u64 NoRing = ~0ull;
    u64 RingCopyGeometry(VAddr address, u64 size);
    std::atomic<u64> geometry_ring_copies{};
    u64 pipe_stats_geometry_ring_copies{};
    /// The current packet's index buffer, copied into the ring by stage A.
    struct RingIndex {
        VAddr address{};
        u32 size{};
        u64 offset{NoRing};
    } packet_index_ring;

    /// Vertex inputs of a draw, worked out on stage A (which waits for stage B about half of
    /// the time): the dynamic vertex input state, each stream's V#, and the stream memory
    /// merged into ranges (sizes clamped to mapped memory). Stage B only obtains the buffers
    /// and records the bindings.
    struct PreparedRange {
        VAddr base;
        u64 size;
        u64 ring; ///< offset in the constant ring when stage A copied it there, else NoRing
    };
    struct PreparedVertex {
        u32 count;
        u32 num_ranges;
        const vk::VertexInputAttributeDescription2EXT* attributes;
        const vk::VertexInputBindingDescription2EXT* bindings;
        const AmdGpu::Buffer* buffers;
        const u8* range_index; ///< per stream: its entry in `ranges`, NoRange without memory
        const PreparedRange* ranges;
    };
    static constexpr u8 NoRange = 0xFF;
    /// Stage A: fills the vtx_* scratch; false when the draw has no dynamic vertex input.
    bool PrepareVertexInputs(const GraphicsPipeline* pipeline);
    VertexInputs<vk::VertexInputAttributeDescription2EXT> vtx_attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> vtx_bindings;
    VertexInputs<AmdGpu::Buffer> vtx_buffers;
    VertexInputs<u8> vtx_range_index;
    VertexInputs<PreparedRange> vtx_ranges;
    alignas(64) PreparedVertex packet_vertex_storage{}; ///< draw recorder
    const PreparedVertex* packet_vertex{}; ///< stage B: the current packet's, if prepared
    std::atomic<u64> vertex_prepared{};
    std::atomic<u64> vertex_unprepared{};
    u64 pipe_stats_vertex_prepared{};
    u64 pipe_stats_vertex_unprepared{};
    const PrefetchedBuffer* FindPrefetched(u32 buffer_index);

    // Draw recorder profile: every 8th direct draw on stage B is timed by section.
    enum class Section : u8 {
        Packet,
        Pending,
        Targets,
        Buffers,
        TexSetHit,
        TexLookup,
        TexView,
        TexLayout,
        Textures,
        Samplers,
        Dlss,
        BeginRendering,
        VertexBuffers,
        Vertex,
        Barriers,
        Descriptors,
        Motion,
        Dynamic,
        Record,
        Reset,
        Count,
    };
    static constexpr std::array<const char*, static_cast<size_t>(Section::Count)> SectionNames{
        "packet",
        "pending ops",
        "render targets",
        "buffers",
        "tex: set memo hit",
        "tex: image lookup",
        "tex: update+view",
        "tex: layout+usage",
        "tex: descriptor writes",
        "samplers",
        "dlss",
        "begin pass",
        "vertex buffers",
        "index buffer",
        "barriers",
        "descriptors",
        "character motion",
        "dynamic",
        "record+mirror",
        "reset"};
    // GPU command thread (stage A) profile: every 8th queued draw, by section; "decode" is the
    // time since the previous draw was queued, less waits for the draw recorder.
    enum class ASection : u8 { Decode, Pipeline, Prefetch, Vertex, Motion, Packet, Count };
    static constexpr std::array<const char*, static_cast<size_t>(ASection::Count)> ASectionNames{
        "decode+other", "pipeline select",        "constant ring copies",
        "vertex prep",  "index copy+motion prep", "packet write"};
    bool a_profile{};
    u32 a_profile_counter{};
    std::chrono::steady_clock::time_point a_last{}, a_lap{};
    std::chrono::nanoseconds a_drain_mark{};
    std::array<std::atomic<s64>, static_cast<size_t>(ASection::Count)> a_profile_ns{};
    std::atomic<u64> a_profile_draws{};
    std::array<s64, static_cast<size_t>(ASection::Count)> pipe_stats_a_ns{};
    u64 pipe_stats_a_draws{};
    void ALap(ASection finished) {
        if (!a_profile) [[likely]] {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        a_profile_ns[static_cast<size_t>(finished)].fetch_add((now - a_lap).count(),
                                                              std::memory_order_relaxed);
        a_lap = now;
    }
    void ABeginDraw();
    void AEndDraw();
    alignas(64) bool profile_draw{}; ///< draw recorder
    u32 profile_counter{};
    std::chrono::steady_clock::time_point profile_last{};
    std::array<std::atomic<s64>, static_cast<size_t>(Section::Count)> profile_ns{};
    std::atomic<u64> profile_draws{};
    std::array<s64, static_cast<size_t>(Section::Count)> pipe_stats_profile_ns{};
    u64 pipe_stats_profile_draws{};
    void ProfileLap(Section finished) {
        if (!profile_draw) [[likely]] {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        profile_ns[static_cast<size_t>(finished)].fetch_add((now - profile_last).count(),
                                                            std::memory_order_relaxed);
        profile_last = now;
    }
    float scene_lod_bias{}; ///< this draw's upscaler texture LOD bias (G-buffer draws only)
    // Copy thread (bb_copy_thread): stage A's copies of guest memory into the constant ring
    // (small read-only buffers, vertex and index data) made on their own thread. Each packet
    // carries the number of jobs queued before it; the draw recorder runs it once those are done.
    struct CopyJob {
        VAddr guest;
        u8* dest;
        u64 offset;
        u64 size;
    };
    static constexpr u64 CopyJobCount = 1ull << 16;
    void RingCopy(VAddr guest, u8* dest, u64 offset, u64 size);
    void RunCopyThread(std::stop_token stop);
    void WaitForCopies(u64 end);
    std::unique_ptr<CopyJob[]> copy_jobs;
    bool copy_thread_on{};
    u64 copy_head{}; ///< stage A
    alignas(64) std::atomic<u64> copy_published{};
    alignas(64) std::atomic<u64> copy_done{};
    alignas(64) std::atomic<u32> copy_wake{};
    std::atomic<bool> copy_sleeping{};
    std::atomic<s64> copy_busy_ns{};
    std::atomic<u64> copy_waits{};
    std::atomic<s64> copy_wait_ns{};
    u64 pipe_stats_copy_jobs{};
    s64 pipe_stats_copy_busy{};
    u64 pipe_stats_copy_waits{};
    s64 pipe_stats_copy_wait_ns{};
    std::jthread copy_thread; ///< stopped after the draw recorder (below)
    // Last member: destroyed (stopped) first, while everything the draw recording thread uses
    // still exists.
    std::unique_ptr<DrawPipe> draw_pipe;
};

} // namespace Vulkan
