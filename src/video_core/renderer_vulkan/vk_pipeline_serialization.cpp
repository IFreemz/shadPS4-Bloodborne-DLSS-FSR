// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
// Experimental build: different from upstream's (binary 9, meta 6, key 3) and from earlier
// experimental builds, so no build loads the others' character-motion shaders or keys.
static constexpr u32 ShaderBinaryVersion = 11u;
static constexpr u32 ShaderMetaVersion = 13u;
static constexpr u32 PipelineKeyVersion = 6u;
} // namespace Serialization

namespace {
// Cache layout of this build. 1: every shader variant preloaded, new ones saved after all known
// ones (earlier caches may hold variants saved over each other, so they are rebuilt once).
constexpr u32 CacheLayout = 1;
} // namespace

namespace {
// Why cached pipelines were not preloaded (logged after the warm-up).
enum StaleReason : u32 {
    KeyVersion,
    MetaMissing,
    MetaVersion,
    BinaryMissing,
    BinaryBroken,
    NumStaleReasons,
};
std::array<u32, NumStaleReasons> stale_reasons{};
} // namespace

namespace Vulkan {

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(Serialization::ShaderMetaVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(Serialization::ShaderMetaVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    meta.Write(Serialization::ShaderBinaryVersion);

    meta.Write(perm_hash);
    meta.Write(perm_idx);

    spec.Serialize(ar);
    info.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    Shader::StageSpecialization& spec, size_t& perm_idx) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion) {
        ++stale_reasons[MetaVersion];
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    if (binary_version != Serialization::ShaderBinaryVersion) {
        ++stale_reasons[MetaVersion];
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    meta.Read(perm_idx);

    spec.Deserialize(ar);
    info.Deserialize(ar);
    return true;
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    // Nothing here yet
    return;
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    // Nothing here yet
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    compute_key.Deserialize(ar);

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    if (meta_blob.empty()) {
        ++stale_reasons[MetaMissing];
        return false;
    }

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0)) {
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    graphics_key.Deserialize(ar);

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        std::vector<u8> meta_blob;
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                           fmt::format("{:#018x}", hash), meta_blob);
        if (meta_blob.empty()) {
            ++stale_reasons[MetaMissing];
            return false;
        }

        Serialization::Archive meta_ar{std::move(meta_blob)};

        if (!LoadPipelineStage(meta_ar, stage_idx)) {
            return false;
        }
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    ASSERT(is_new);

    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader = nullptr;

    return true;
}

// Loads one cached shader variant (description and binary) into the program cache at the index
// it was saved under, unless it is there already. Returns its program, or null.
Program* PipelineCache::PreloadVariant(Serialization::Archive& ar, size_t& perm_idx) {
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    if (!LoadShaderMeta(ar, program->info, spec, perm_idx)) {
        return nullptr;
    }
    const u64 pgm_hash = program->info.pgm_hash;
    if (const auto it = program_cache.find(pgm_hash); it != program_cache.end()) {
        auto& modules = it.value()->modules;
        if (perm_idx < modules.size() && modules[perm_idx].module) {
            return it.value().get();
        }
    }

    std::vector<u32> spv{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx), spv);
    if (spv.empty()) {
        ++stale_reasons[BinaryMissing];
        return nullptr;
    }
    // A binary cut short or damaged (the game closed while it was written): rebuilt in game
    // instead of stopping the launch.
    static constexpr u32 SpirvMagic = 0x07230203;
    if (spv.size() < 5 || spv[0] != SpirvMagic) {
        ++stale_reasons[BinaryBroken];
        return nullptr;
    }
    auto [module_result, module] = instance.GetDevice().createShaderModule(
        vk::ShaderModuleCreateInfo{.codeSize = spv.size() * sizeof(u32), .pCode = spv.data()});
    if (module_result != vk::Result::eSuccess) {
        LOG_WARNING(Render, "Cached shader {:#018x}_{} was rejected by the driver ({})", pgm_hash,
                    perm_idx, vk::to_string(module_result));
        ++stale_reasons[BinaryBroken];
        return nullptr;
    }

    auto [it_pgm, new_program] = program_cache.try_emplace(pgm_hash);
    if (new_program) {
        it_pgm.value() = std::move(program);
    }
    auto* target = it_pgm.value().get();
    spec.info = &target->info;
    // At the index it was saved under: pipeline keys name variants by index.
    target->InsertPermut(module, std::move(spec), perm_idx);
    return target;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t stage) {
    size_t perm_idx{};
    auto* program = PreloadVariant(ar, perm_idx);
    if (!program) {
        return false;
    }
    infos[stage] = &program->info;
    modules[stage] = program->modules[perm_idx].module;
    // The variant's own runtime info: the pipeline reads from it what its shaders do (the
    // upscaler jitter in the vertex shader), not left over from the last pipeline built.
    runtime_infos[stage] = program->modules[perm_idx].spec.runtime_info;
    if (auto& fetch = program->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        fetch_shader = &fetch;
    }
    return true;
}

void PipelineCache::WarmUp() {
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        return;
    }

    Storage::DataBase::Instance().Open();

    // Check if cache is compatible: same shader profile (build and GPU) and cache layout.
    auto& db = Storage::DataBase::Instance();
    const auto start_fresh = [&] {
        db.FinishPreload();
        std::vector<u8> profile_data(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        db.Save(Storage::BlobType::ShaderProfile, "profile", std::move(profile_data));
        std::vector<u8> layout(sizeof(CacheLayout));
        std::memcpy(layout.data(), &CacheLayout, sizeof(CacheLayout));
        db.Save(Storage::BlobType::ShaderProfile, "layout", std::move(layout));
    };
    std::vector<u8> profile_data{};
    db.Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    if (profile_data.empty()) {
        start_fresh();
        return;
    }
    Shader::Profile cached_profile{};
    if (profile_data.size() == sizeof(Shader::Profile)) {
        std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    }
    std::vector<u8> layout_data{};
    db.Load(Storage::BlobType::ShaderProfile, "layout", layout_data);
    u32 cached_layout{};
    if (layout_data.size() == sizeof(cached_layout)) {
        std::memcpy(&cached_layout, layout_data.data(), sizeof(cached_layout));
    }
    if (profile_data.size() != sizeof(Shader::Profile) || cached_profile != profile ||
        cached_layout != CacheLayout) {
        // The cache used to be closed for the session here, so it was never rewritten and
        // every later session compiled every shader again (a hitch on each new area). Start a
        // fresh cache for this build and GPU instead (as bbport does). An older layout may hold
        // shader variants that later sessions saved over each other: rebuilt as well.
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system (profile {} bytes, "
                    "expected {}; layout {}, expected {}): rebuilding it",
                    profile_data.size(), sizeof(Shader::Profile), cached_layout, CacheLayout);
        db.Clear();
        start_fresh();
        return;
    }

    // Every cached shader variant first, whether or not a pipeline using it loads: a variant the
    // game needs again is then found, not translated again, and a new one is saved after all
    // known ones instead of over one of them.
    u32 num_variants{};
    db.ForEachBlob(Storage::BlobType::ShaderMeta, [&](std::vector<u8>&& data) {
        Serialization::Archive ar{std::move(data)};
        size_t perm_idx{};
        if (PreloadVariant(ar, perm_idx)) {
            ++num_variants;
        }
    });
    LOG_INFO(Render, "Preloaded {} shader variants", num_variants);
    stale_reasons.fill(0);

    u32 num_pipelines{};
    u32 num_total_pipelines{};

    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
            ++num_total_pipelines;

            Serialization::Archive ar{std::move(data)};
            Serialization::Reader pldata{ar};

            u32 version{};
            pldata.Read(version);
            if (version != Serialization::PipelineKeyVersion) {
                ++stale_reasons[KeyVersion];
                return;
            }
            // Only pipelines saved with this build's shader variants: shader variants are saved
            // under the same names by every build, so a pipeline from another build would
            // otherwise come back once this build has saved its own variants under those names.
            u32 meta_version{};
            pldata.Read(meta_version);
            if (meta_version != Serialization::ShaderMetaVersion) {
                ++stale_reasons[KeyVersion];
                return;
            }

            u32 is_compute{};
            pldata.Read(is_compute);

            bool result{};
            if (is_compute) {
                result = LoadComputePipeline(ar);
            } else {
                result = LoadGraphicsPipeline(ar);
            }

            if (result) {
                ++num_pipelines;
            }
        });

    LOG_INFO(Render, "Preloaded {} pipelines", num_pipelines);
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
        LOG_WARNING(Render,
                    "Stale pipelines: {} from older builds, {} shader info missing, {} shader "
                    "info from older builds, {} shader binary missing, {} shader binary damaged",
                    stale_reasons[KeyVersion], stale_reasons[MetaMissing],
                    stale_reasons[MetaVersion], stale_reasons[BinaryMissing],
                    stale_reasons[BinaryBroken]);
    }

    Storage::DataBase::Instance().FinishPreload();
}

void PipelineCache::Sync() {
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    srt.Write(this, sizeof(*this));
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};

    srt.Read(this, sizeof(*this));

    if (walker_func_size) {
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size);
        ar.Advance(walker_func_size);
    }

    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    spec.Write(runtime_info);

    spec.Write(bitset.to_string());

    if (!fetch_shader_data.Empty()) {
        spec.Write(sizeof(fetch_shader_data));
        fetch_shader_data.Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader
