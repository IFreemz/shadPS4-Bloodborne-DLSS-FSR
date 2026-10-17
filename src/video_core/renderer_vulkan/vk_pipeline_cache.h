// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <variant>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
    }
};

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
    bool is_indirect;
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);
    Program* PreloadVariant(Serialization::Archive& ar, size_t& perm_idx);

    /// `may_defer` (GPU command thread, threaded renderer): a new pipeline for a scene draw
    /// (depth tested, into a depth target used in each of the last 8 game frames) is built on a
    /// background thread and null returned, so the draw is skipped until it is ready (Background
    /// pipeline compiles setting).
    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {},
                                                bool may_defer = false);

    /// Counts game frames (any thread): render target history for deferred compiles.
    void OnGameFrame() noexcept {
        game_frames.fetch_add(1, std::memory_order_relaxed);
    }

    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

    // Background pipeline compiles.
    void NoteDepthTarget();
    [[nodiscard]] bool MayDefer() const;
    void StartBackgroundCompile();
    void StartBackgroundShader(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                               const Shader::ShaderParams& params,
                               const Shader::RuntimeInfo& runtime_info,
                               const Shader::Backend::Bindings& binding,
                               std::optional<Shader::StageSpecialization> spec);
    [[nodiscard]] bool IsShaderPending(u64 hash,
                                       const Shader::StageSpecialization* spec = nullptr) const;
    void CollectBackgroundPipelines(bool shaders = true);
    void CompileWorker(std::stop_token stop);
    // The driver's own pipeline cache, kept between sessions.
    [[nodiscard]] std::filesystem::path DriverCachePath() const;
    vk::UniquePipelineCache CreateDriverCache();
    void SaveDriverCache();
    void SaveDriverCacheInBackground();

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    bool motion_select{}; // the stages being built write character motion vectors
    // Character motion choice per vertex shader (by code address), so a G-buffer draw builds
    // its stages once instead of twice.
    tsl::robin_map<u64, bool> motion_choices;
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    struct FinishedShader {
        u64 hash;
        Shader::HwStage hw_stage;
        std::unique_ptr<Program> program; ///< a new program (null: a new variant)
        std::optional<Shader::StageSpecialization> spec;
        std::vector<u32> spv;
        vk::ShaderModule module;
        double ms;
    };
    struct FinishedPipeline {
        GraphicsPipelineKey key;
        u64 hash;
        std::unique_ptr<GraphicsPipeline> pipeline;
        GraphicsPipeline::SerializationSupport sdata;
        double ms;
    };
    std::mutex compile_mutex;
    std::condition_variable_any compile_cv;
    std::deque<std::function<void()>> compile_jobs;
    std::mutex finished_mutex;
    std::vector<FinishedPipeline> finished_pipelines;
    std::vector<FinishedShader> finished_shaders;
    void CollectBackgroundShader(FinishedShader& done);
    std::atomic<u32> num_finished{0};
    tsl::robin_map<GraphicsPipelineKey, u8> pending_pipelines; ///< GPU command thread only
    /// Shaders being translated in the background (GPU command thread only): a new program,
    /// and new variants of known ones.
    struct PendingShader {
        bool new_program{};
        std::vector<Shader::StageSpecialization> variants;
    };
    tsl::robin_map<u64, PendingShader> pending_shaders;
    bool translate_in_background{}; ///< the draw being prepared may skip a new shader
    bool stage_deferred{};          ///< ... and did
    Program* fetch_program{};       ///< vertex shader program and variant with a fetch shader
    size_t fetch_perm{};
    struct DepthTargetUse {
        VAddr address;
        u32 frames; ///< bit n: used n game frames ago
    };
    std::array<DepthTargetUse, 32> depth_targets{};
    u64 depth_target_frame{};
    VAddr last_depth_target{};
    std::atomic<u64> game_frames{0};
    std::atomic<bool> driver_save_queued{false};
    u32 driver_saved_pipelines{};
    std::chrono::steady_clock::time_point driver_saved_at{};

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
    // Last: stopped (joined) first, while everything their jobs use still exists.
    std::vector<std::jthread> compile_threads;
};

} // namespace Vulkan
