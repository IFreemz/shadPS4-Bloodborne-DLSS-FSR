// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <span>
#include <vector>

#include <zlib.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <bcrypt.h>
#endif

#include "common/logging/log.h"
#include "core/file_sys/bb_hdr_compat.h"

namespace Core::FileSys::BbHdrCompat {
namespace {

using Hash = std::array<u8, 32>;
struct BytePatch {
    u32 offset;
    u8 before;
    u8 after;
};
struct Profile {
    std::string_view path;
    bool moon;
    u32 size;
    Hash original;
    Hash corrected;
    std::span<const BytePatch> patches;
};

#include "core/file_sys/bb_hdr_profiles.h"

constexpr size_t HeaderSize = 0x4c;
constexpr size_t MaxFileSize = 2 * 1024 * 1024;
std::atomic_bool enabled{false};
std::atomic_bool restore_moon{false};
std::atomic_bool remaster_detected{false};
std::atomic_bool remaster_present{false}; // ProbeRemaster: found whatever the HDR settings

struct CachedFile {
    Hash source_hash;
    std::shared_ptr<const std::vector<u8>> data;
};
std::array<std::optional<CachedFile>, Profiles.size()> cache;
std::mutex cache_mutex;

std::optional<Hash> Sha256(std::span<const u8> bytes) {
#ifdef _WIN32
    struct Algorithm {
        BCRYPT_ALG_HANDLE handle{};
        Algorithm() {
            BCryptOpenAlgorithmProvider(&handle, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
        }
        ~Algorithm() {
            if (handle) {
                BCryptCloseAlgorithmProvider(handle, 0);
            }
        }
    };
    static const Algorithm algorithm;
    Hash hash;
    if (algorithm.handle &&
        BCryptHash(algorithm.handle, nullptr, 0, const_cast<PUCHAR>(bytes.data()),
                   static_cast<ULONG>(bytes.size()), hash.data(), hash.size()) >= 0) {
        return hash;
    }
#endif
    return std::nullopt;
}

u32 ReadBE(std::span<const u8> data, size_t offset) {
    return (u32(data[offset]) << 24) | (u32(data[offset + 1]) << 16) |
           (u32(data[offset + 2]) << 8) | data[offset + 3];
}

bool ValidHeader(std::span<const u8> data, const Profile& profile) {
    if (data.size() <= HeaderSize || data.size() > MaxFileSize) {
        return false;
    }
    // Only the DFLT layout used by the fingerprinted remaster bundles is supported.
    constexpr std::array<u8, HeaderSize> header{
        0x44, 0x43, 0x58, 0,    0,    1,    0,    0,    0,    0,    0,    0x18, 0, 0, 0, 0x24,
        0,    0,    0,    0x44, 0,    0,    0,    0x4c, 0x44, 0x43, 0x53, 0,    0, 0, 0, 0,
        0,    0,    0,    0,    0x44, 0x43, 0x50, 0,    0x44, 0x46, 0x4c, 0x54, 0, 0, 0, 0x20,
        9,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0, 0, 0,
        0,    1,    1,    0,    0x44, 0x43, 0x41, 0,    0,    0,    0,    8};
    for (size_t i = 0; i < HeaderSize; ++i) {
        if (i >= 0x1c && i < 0x24) {
            continue;
        }
        if (data[i] != header[i]) {
            return false;
        }
    }
    return profile.size < MaxFileSize && ReadBE(data, 0x1c) == profile.size &&
           ReadBE(data, 0x20) == data.size() - HeaderSize;
}

class CorrectedFile final : public IFile {
public:
    CorrectedFile(std::unique_ptr<IFile> source, std::shared_ptr<const std::vector<u8>> data)
        : position(source->Tell()), source(std::move(source)), data(std::move(data)) {}

    s64 Read(void* dst, u64 size) override {
        std::scoped_lock lock{mutex};
        if (size && !dst) {
            return -1;
        }
        const u64 count = position < data->size() ? std::min(size, data->size() - position) : 0;
        if (count) {
            std::memcpy(dst, data->data() + position, count);
            position += count;
        }
        return count;
    }
    s64 Write(const void*, u64) override {
        return -1;
    }
    bool Seek(s64 offset, Common::FS::SeekOrigin origin) override {
        std::scoped_lock lock{mutex};
        u64 base{};
        switch (origin) {
        case Common::FS::SeekOrigin::SetOrigin:
            break;
        case Common::FS::SeekOrigin::CurrentPosition:
            base = position;
            break;
        case Common::FS::SeekOrigin::End:
            base = data->size();
            break;
        default:
            return false;
        }
        if (offset < 0) {
            const u64 distance = u64(-(offset + 1)) + 1;
            if (distance > base) {
                return false;
            }
            position = base - distance;
        } else {
            if (u64(offset) > u64(std::numeric_limits<s64>::max()) - base) {
                return false;
            }
            position = base + offset;
        }
        return true;
    }
    u64 Tell() const override {
        std::scoped_lock lock{mutex};
        return position;
    }
    u64 Size() const override {
        return data->size();
    }
    bool Flush() override {
        return true;
    }
    bool IsOpen() const override {
        return source->IsOpen();
    }
    std::optional<std::filesystem::path> GetMetadataHostPath() const override {
        return source->GetMetadataHostPath();
    }
    void Stat(FileStat& out) override {
        source->Stat(out);
        out.size = data->size();
    }
    MmapPolicy GetMmapPolicy() const override {
        return MmapPolicy::Copy;
    }
    bool Map(u8* addr, u64 size, u64 offset, u32 prot, const FileMapContext& ctx) override {
        if (!addr || !ctx.map_anonymous || !ctx.protect ||
            size > std::numeric_limits<size_t>::max()) {
            return false;
        }
        ctx.map_anonymous(addr, size);
        const u64 count = offset < data->size() ? std::min(size, data->size() - offset) : 0;
        if (count) {
            std::memcpy(addr, data->data() + offset, count);
        }
        std::memset(addr + count, 0, size - count);
        ctx.protect(addr, size, prot);
        return true;
    }

private:
    mutable std::mutex mutex;
    u64 position;
    std::unique_ptr<IFile> source;
    std::shared_ptr<const std::vector<u8>> data;
};

const Profile* FindProfile(std::string_view path) {
    if (!enabled.load(std::memory_order_relaxed) ||
        (!path.starts_with("dvdroot_ps4/param/drawparam/") &&
         path != "dvdroot_ps4/mtd/allmaterialbnd.mtdbnd.dcx")) {
        return nullptr;
    }
    for (const auto& profile : Profiles) {
        if (profile.path == path &&
            (!profile.moon || restore_moon.load(std::memory_order_relaxed))) {
            return &profile;
        }
    }
    return nullptr;
}

} // namespace

void Configure(bool supported_game, bool enable, bool moon) {
    enabled.store(supported_game && enable, std::memory_order_relaxed);
    restore_moon.store(moon, std::memory_order_relaxed);
    remaster_detected.store(false, std::memory_order_relaxed);
    std::scoped_lock lock{cache_mutex};
    cache.fill(std::nullopt);
}

void Initialize(bool supported_game, const std::filesystem::path& settings_path) {
    bool enable = true;
    bool moon = false;
    std::ifstream settings{settings_path};
    for (std::string line; std::getline(settings, line);) {
        if (line.starts_with("hdr_mod_compat=")) {
            enable = line.substr(15) != "0";
        } else if (line.starts_with("hdr_moon_restore=")) {
            moon = line.substr(17) == "1";
        }
    }
    Configure(supported_game, enable, moon);
}

bool IsCandidate(std::string_view path) {
    return FindProfile(path) != nullptr;
}

bool ShouldSkipPatch(std::string_view name) {
    return enabled.load(std::memory_order_relaxed) &&
           remaster_detected.load(std::memory_order_relaxed) &&
           (name == "Disable AA" || name == "Disable Bloom" ||
            name == "Disable Chromatic Aberration");
}

std::unique_ptr<IFile> Open(std::unique_ptr<IFile> source, std::string_view path) {
    const auto* profile = FindProfile(path);
    if (!profile || !source || !source->IsOpen() || !source->IsReadOnly()) {
        return source;
    }
    const auto file_size = source->Size();
    if (file_size <= HeaderSize || file_size > MaxFileSize) {
        return source;
    }
    const u64 position = source->Tell();
    if (position > u64(std::numeric_limits<s64>::max()) ||
        !source->Seek(0, Common::FS::SeekOrigin::SetOrigin)) {
        return source;
    }
    std::vector<u8> compressed(file_size);
    size_t read{};
    while (read < compressed.size()) {
        const auto count = source->Read(compressed.data() + read, compressed.size() - read);
        if (count <= 0 || u64(count) > compressed.size() - read) {
            break;
        }
        read += count;
    }
    if (!source->Seek(position, Common::FS::SeekOrigin::SetOrigin)) {
        return nullptr;
    }
    if (read != compressed.size() || !ValidHeader(compressed, *profile)) {
        return source;
    }
    const auto source_hash = Sha256(compressed);
    if (!source_hash) {
        return source;
    }
    std::scoped_lock lock{cache_mutex};
    auto& entry = cache[profile - Profiles.data()];
    if (entry && entry->source_hash == *source_hash) {
        return std::make_unique<CorrectedFile>(std::move(source), entry->data);
    }
    std::vector<u8> raw(profile->size);
    uLongf size = raw.size();
    uLong compressed_size = compressed.size() - HeaderSize;
    if (uncompress2(raw.data(), &size, compressed.data() + HeaderSize, &compressed_size) != Z_OK ||
        size != raw.size() || compressed_size != compressed.size() - HeaderSize) {
        return source;
    }
    const auto raw_hash = Sha256(raw);
    if (!profile->moon && (raw_hash == profile->original || raw_hash == profile->corrected)) {
        remaster_detected.store(true, std::memory_order_relaxed);
    }
    if (raw_hash != profile->original) {
        return source;
    }
    for (const auto& patch : profile->patches) {
        if (patch.offset >= raw.size() || raw[patch.offset] != patch.before) {
            return source;
        }
        raw[patch.offset] = patch.after;
    }
    if (Sha256(raw) != profile->corrected) {
        return source;
    }
    auto result = std::make_shared<std::vector<u8>>(HeaderSize + compressBound(raw.size()));
    std::copy_n(compressed.begin(), HeaderSize, result->begin());
    size = result->size() - HeaderSize;
    if (compress2(result->data() + HeaderSize, &size, raw.data(), raw.size(), 9) != Z_OK) {
        return source;
    }
    result->resize(HeaderSize + size);
    for (size_t i = 0; i < 4; ++i) {
        (*result)[0x20 + i] = (size >> ((3 - i) * 8)) & 0xff;
    }
    entry = CachedFile{*source_hash, result};
    LOG_INFO(Common_Filesystem, "[HDR-COMPAT] {}: {}", path,
             profile->moon ? "moon brightness restored" : "remaster HDR parameters restored");
    return std::make_unique<CorrectedFile>(std::move(source), std::move(result));
}

void ProbeRemaster(std::unique_ptr<IFile> file, std::string_view path) {
    remaster_present.store(false, std::memory_order_relaxed);
    const auto profile = std::ranges::find_if(Profiles, [path](const Profile& candidate) {
        return candidate.path == path && !candidate.moon;
    });
    if (profile == Profiles.end() || !file || !file->IsOpen()) {
        return;
    }
    const auto file_size = file->Size();
    if (file_size <= HeaderSize || file_size > MaxFileSize) {
        return;
    }
    std::vector<u8> compressed(file_size);
    size_t read{};
    while (read < compressed.size()) {
        const auto count = file->Read(compressed.data() + read, compressed.size() - read);
        if (count <= 0 || u64(count) > compressed.size() - read) {
            break;
        }
        read += count;
    }
    if (read != compressed.size() || !ValidHeader(compressed, *profile)) {
        return;
    }
    std::vector<u8> raw(profile->size);
    uLongf size = raw.size();
    uLong compressed_size = compressed.size() - HeaderSize;
    if (uncompress2(raw.data(), &size, compressed.data() + HeaderSize, &compressed_size) != Z_OK ||
        size != raw.size()) {
        return;
    }
    const auto raw_hash = Sha256(raw);
    if (raw_hash && (*raw_hash == profile->original || *raw_hash == profile->corrected)) {
        remaster_present.store(true, std::memory_order_relaxed);
        LOG_INFO(Common_Filesystem, "BB PC Remaster found ({})", path);
    }
}

bool RemasterPresent() {
    return remaster_present.load(std::memory_order_relaxed);
}

} // namespace Core::FileSys::BbHdrCompat
