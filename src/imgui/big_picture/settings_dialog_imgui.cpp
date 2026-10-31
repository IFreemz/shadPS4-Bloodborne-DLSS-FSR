//  SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
//  SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <map>
#include <ranges>
#include <utility>
#include <ImGuiFileDialog.h>
#include <SDL3/SDL_events.h>
#include <cmrc/cmrc.hpp>
#include <stb_image.h>

#include "common/copy_pool.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/cpu_patches.h"
#include "core/devtools/layer.h"
#include "imgui/imgui_std.h"
#include "input/input_handler.h"
#include "settings_dialog_imgui.h"

CMRC_DECLARE(res);

constexpr float gameImageSize = 200.f;
constexpr float settingsIconSize = 125.f;

namespace ImGuiEmuSettings {

int SettingsWindow::GetComboIndex(std::string selection, std::vector<std::string> options) {
    for (int i = 0; i < options.size(); i++) {
        if (selection == options[i])
            return i;
    }

    return 0;
}

void SettingsWindow::LoadSettings(std::string profile) {
    // Global-only settings
    EmulatorSettings.Load();
    scanDepthSetting = EmulatorSettings.GetBigPictureFolderDepth();

    const bool isSpecific = currentProfile != "Global";
    if (isSpecific) {
        EmulatorSettings.Load(profile);
    }

    /////////// General Tab
    int languageIndex = EmulatorSettings.GetConsoleLanguage();
    std::string language;
    for (const auto& [key, value] : languageMap) {
        if (value == languageIndex) {
            language = key;
        }
    }
    consoleLanguageSetting = GetComboIndex(language, languageOptions);
    volumeSetting = EmulatorSettings.GetVolumeSlider();
    showSplashSetting = EmulatorSettings.IsShowSplash();
    audioBackendSetting = EmulatorSettings.GetAudioBackend();

    /////////// Graphics Tab
    fullscreenModeSetting =
        GetComboIndex(EmulatorSettings.GetFullScreenMode(), fullscreenModeOptions);
    presentModeSetting = GetComboIndex(EmulatorSettings.GetPresentMode(), presentModeOptions);
    windowHeightSetting = EmulatorSettings.GetWindowHeight();
    windowWidthSetting = EmulatorSettings.GetWindowWidth();
    hdrAllowedSetting = EmulatorSettings.IsHdrAllowed();
    fsrEnabledSetting = EmulatorSettings.IsFsrEnabled();
    rcasEnabledSetting = EmulatorSettings.IsRcasEnabled();
    rcasAttenuationSetting = static_cast<float>(EmulatorSettings.GetRcasAttenuation() * 0.001f);
    threadedRendererSetting = EmulatorSettings.IsThreadedRenderer();
    threadedRendererDiagnosticsSetting = EmulatorSettings.IsThreadedRendererDiagnostics();
    threadedRendererGpuProfileSetting = EmulatorSettings.IsThreadedRendererGpuProfile();
    bbReadbackQueueSetting = EmulatorSettings.IsBbReadbackQueue();
    bbStreamingSetting = EmulatorSettings.IsBbStreaming();
    bbMergedCopiesSetting = EmulatorSettings.IsBbMergedCopies();
    bbMergedNoStreamSourcesSetting = EmulatorSettings.IsBbMergedNoStreamSources();
    bbMergedLargeBatchesSetting = EmulatorSettings.IsBbMergedLargeBatches();
    bbFastLocksSetting = EmulatorSettings.IsBbFastLocks();
    bbFlipWhenReadySetting = EmulatorSettings.IsBbFlipWhenReady();
    bbVramReclaimSetting = EmulatorSettings.IsBbVramReclaim();
    bbBufferEvictionSetting = EmulatorSettings.IsBbBufferEviction();
    bbLoadScreenCleanupSetting = EmulatorSettings.IsBbLoadScreenCleanup();
    bbMoreVramSetting = EmulatorSettings.IsBbMoreVram();
    bbVramPastBudgetSetting = EmulatorSettings.IsBbVramPastBudget();
    bbBackgroundTextureCopiesSetting = EmulatorSettings.IsBbBackgroundTextureCopies();
    bbGeometryInPlaceSetting = EmulatorSettings.IsBbGeometryInPlace();
    bbSkipEmptyDrawsSetting = EmulatorSettings.IsBbSkipEmptyDraws();
    bbBackgroundPipelinesSetting = EmulatorSettings.IsBbBackgroundPipelines();
    bbExactProtectionSetting = EmulatorSettings.IsBbExactProtection();
    bbBatchedUploadsSetting = EmulatorSettings.IsBbBatchedUploads();
    bbTextureSetMemoSetting = EmulatorSettings.IsBbTextureSetMemo();
    bbRingGeometrySetting = EmulatorSettings.IsBbRingGeometry();
    bbSubmitOverlapSetting = EmulatorSettings.IsBbSubmitOverlap();
    bbOverlapFramesSetting = EmulatorSettings.IsBbOverlapFrames();
    bbEarlyWritebackSetting = EmulatorSettings.IsBbEarlyWriteback();
    bbOcclusionInOrderSetting = EmulatorSettings.IsBbOcclusionInOrder();
    bbFastPixelMathSetting = EmulatorSettings.IsBbFastPixelMath();
    bbConstantsNoWaitSetting = EmulatorSettings.IsBbConstantsNoWait();
    bbHardwareInterpolationSetting = EmulatorSettings.IsBbHardwareInterpolation();
    bbRealOcclusionSetting = EmulatorSettings.IsBbRealOcclusion();
    bbCopyThreadSetting = EmulatorSettings.IsBbCopyThread();
    bbRenderDuringPresentSetting = EmulatorSettings.IsBbRenderDuringPresent();
    bbSkipLeftOutLevelsSetting = EmulatorSettings.IsBbSkipLeftOutLevels();
    bbDirectTextureCopiesSetting = EmulatorSettings.IsBbDirectTextureCopies();
    {
        const int threads = EmulatorSettings.GetBbCopyThreads();
        bbCopyThreadsSetting = threads >= 8 ? 3 : threads >= 6 ? 2 : threads >= 4 ? 1 : 0;
    }
    bbCachedTextureStagingSetting = EmulatorSettings.IsBbCachedTextureStaging();
    renderdocSetting = EmulatorSettings.IsRenderdocEnabled();
    bbRdocCaptureFramesSetting = EmulatorSettings.GetBbRdocCaptureFrames();
    bbLowerDetailFirstSetting = EmulatorSettings.IsBbLowerDetailFirst();
    bbClearShaderWrittenSetting = EmulatorSettings.IsBbClearShaderWritten();
    bbCeSkipHugeWritesSetting = EmulatorSettings.IsBbCeSkipHugeWrites();
    bbEvictKeepProtectionSetting = EmulatorSettings.IsBbEvictKeepProtection();
    bbEvictKeepReturningSetting = EmulatorSettings.IsBbEvictKeepReturning();
    bbProtectHelperThreadSetting = EmulatorSettings.IsBbProtectHelperThread();
    bbCpuCopyShaderSetting = EmulatorSettings.IsBbCpuCopyShader();
    bbCopySkipSameSetting = EmulatorSettings.IsBbCopySkipSame();
    bbClearOneLayerSetting = EmulatorSettings.IsBbClearOneLayer();
    bbSpreadLoadsSetting = EmulatorSettings.IsBbSpreadLoads();
    {
        // 0 (the game's own), 2, 4, 8, 16 -> 0..4
        const int samples = EmulatorSettings.GetBbAnisotropy();
        bbAnisotropySetting = samples >= 16  ? 4
                              : samples >= 8 ? 3
                              : samples >= 4 ? 2
                              : samples >= 2 ? 1
                                             : 0;
    }
    bbDrawDataMemorySetting = std::clamp(EmulatorSettings.GetBbDrawDataMemory(), 0, 2);
    bbTextureQualitySetting = std::clamp(EmulatorSettings.GetBbTextureQuality(), 0, 2);
    bbIgnoreDisplayScaleSetting = EmulatorSettings.IsBbIgnoreDisplayScale();
    bbStutterDetailSetting = EmulatorSettings.IsBbStutterDetail();
    bbVramBudgetTestSetting = EmulatorSettings.GetBbVramBudgetTestMb();

    /////////// Input Tab
    motionControlsSetting = EmulatorSettings.IsMotionControlsEnabled();
    backgroundControllerSetting = EmulatorSettings.IsBackgroundControllerInput();
    cursorStateSetting = EmulatorSettings.GetCursorState();
    cursorTimeoutSetting = EmulatorSettings.GetCursorHideTimeout();

    /////////// Trophy Tab
    trophyPopupDisabledSetting = EmulatorSettings.IsTrophyPopupDisabled();
    trophySideSetting =
        GetComboIndex(EmulatorSettings.GetTrophyNotificationSide(), trophySideOptions);
    trophyDurationSetting = static_cast<float>(EmulatorSettings.GetTrophyNotificationDuration());

    /////////// Log Tab
    logEnableSetting = EmulatorSettings.IsLogEnable();
    logSeparateSetting = EmulatorSettings.IsLogSeparate();
    logSyncSetting = EmulatorSettings.IsLogSync();

    /////////// Experimental Tab
    if (isSpecific) {
        readbacksModeSetting = EmulatorSettings.GetReadbacksMode();
        readbackLinearImagesSetting = EmulatorSettings.IsReadbackLinearImagesEnabled();
        directMemoryAccessSetting = EmulatorSettings.IsDirectMemoryAccessEnabled();
        windowsGuestRedZoneProtectionModeSetting = EmulatorSettings.IsRedZonePatchingEnabled();
        devkitConsoleSetting = EmulatorSettings.IsDevKit();
        neoModeSetting = EmulatorSettings.IsNeo();
        shadnetEnabledSetting = EmulatorSettings.IsShadNetEnabledSetting();
        connectedNetworkSetting = EmulatorSettings.IsConnectedToNetwork();
        pipelineCacheEnabledSetting = EmulatorSettings.IsPipelineCacheEnabled();
        pipelineCacheArchiveSetting = EmulatorSettings.IsPipelineCacheArchived();
        extraDmemSetting = EmulatorSettings.GetExtraDmemInMBytes();
        vblankFrequencySetting = EmulatorSettings.GetVblankFrequency();
    }
}

void SettingsWindow::SaveSettings(std::string profile) {
    const bool isSpecific = currentProfile != "Global";

    /////////// General Tab
    EmulatorSettings.SetConsoleLanguage(languageMap.at(languageOptions.at(consoleLanguageSetting)),
                                        isSpecific);
    EmulatorSettings.SetVolumeSlider(volumeSetting, isSpecific);
    EmulatorSettings.SetShowSplash(showSplashSetting, isSpecific);
    EmulatorSettings.SetAudioBackend(audioBackendSetting, isSpecific);

    /////////// Graphics Tab
    bool isFullscreen = fullscreenModeSetting != 0;
    EmulatorSettings.SetFullScreen(isFullscreen, isSpecific);
    EmulatorSettings.SetFullScreenMode(fullscreenModeOptions.at(fullscreenModeSetting), isSpecific);
    EmulatorSettings.SetPresentMode(presentModeOptions.at(presentModeSetting), isSpecific);
    EmulatorSettings.SetWindowHeight(windowHeightSetting, isSpecific);
    EmulatorSettings.SetWindowWidth(windowWidthSetting, isSpecific);
    EmulatorSettings.SetHdrAllowed(hdrAllowedSetting, isSpecific);
    EmulatorSettings.SetFsrEnabled(fsrEnabledSetting, isSpecific);
    EmulatorSettings.SetRcasEnabled(rcasEnabledSetting, isSpecific);
    EmulatorSettings.SetRcasAttenuation(static_cast<int>(rcasAttenuationSetting * 1000),
                                        isSpecific);
    EmulatorSettings.SetThreadedRenderer(threadedRendererSetting, isSpecific);
    EmulatorSettings.SetThreadedRendererDiagnostics(threadedRendererDiagnosticsSetting, isSpecific);
    EmulatorSettings.SetThreadedRendererGpuProfile(threadedRendererGpuProfileSetting, isSpecific);
    EmulatorSettings.SetBbReadbackQueue(bbReadbackQueueSetting, isSpecific);
    EmulatorSettings.SetBbStreaming(bbStreamingSetting, isSpecific);
    EmulatorSettings.SetBbMergedCopies(bbMergedCopiesSetting, isSpecific);
    EmulatorSettings.SetBbMergedNoStreamSources(bbMergedNoStreamSourcesSetting, isSpecific);
    EmulatorSettings.SetBbMergedLargeBatches(bbMergedLargeBatchesSetting, isSpecific);
    EmulatorSettings.SetBbFastLocks(bbFastLocksSetting, isSpecific);
    EmulatorSettings.SetBbFlipWhenReady(bbFlipWhenReadySetting, isSpecific);
    EmulatorSettings.SetBbVramReclaim(bbVramReclaimSetting, isSpecific);
    EmulatorSettings.SetBbBufferEviction(bbBufferEvictionSetting, isSpecific);
    EmulatorSettings.SetBbLoadScreenCleanup(bbLoadScreenCleanupSetting, isSpecific);
    EmulatorSettings.SetBbMoreVram(bbMoreVramSetting, isSpecific);
    EmulatorSettings.SetBbVramPastBudget(bbVramPastBudgetSetting, isSpecific);
    EmulatorSettings.SetBbBackgroundTextureCopies(bbBackgroundTextureCopiesSetting, isSpecific);
    EmulatorSettings.SetBbGeometryInPlace(bbGeometryInPlaceSetting, isSpecific);
    EmulatorSettings.SetBbSkipEmptyDraws(bbSkipEmptyDrawsSetting, isSpecific);
    EmulatorSettings.SetBbBackgroundPipelines(bbBackgroundPipelinesSetting, isSpecific);
    EmulatorSettings.SetBbExactProtection(bbExactProtectionSetting, isSpecific);
    EmulatorSettings.SetBbBatchedUploads(bbBatchedUploadsSetting, isSpecific);
    EmulatorSettings.SetBbTextureSetMemo(bbTextureSetMemoSetting, isSpecific);
    EmulatorSettings.SetBbRingGeometry(bbRingGeometrySetting, isSpecific);
    EmulatorSettings.SetBbSubmitOverlap(bbSubmitOverlapSetting, isSpecific);
    EmulatorSettings.SetBbOverlapFrames(bbOverlapFramesSetting, isSpecific);
    EmulatorSettings.SetBbEarlyWriteback(bbEarlyWritebackSetting, isSpecific);
    EmulatorSettings.SetBbOcclusionInOrder(bbOcclusionInOrderSetting, isSpecific);
    EmulatorSettings.SetBbFastPixelMath(bbFastPixelMathSetting, isSpecific);
    EmulatorSettings.SetBbConstantsNoWait(bbConstantsNoWaitSetting, isSpecific);
    EmulatorSettings.SetBbHardwareInterpolation(bbHardwareInterpolationSetting, isSpecific);
    EmulatorSettings.SetBbRealOcclusion(bbRealOcclusionSetting, isSpecific);
    EmulatorSettings.SetBbCopyThread(bbCopyThreadSetting, isSpecific);
    EmulatorSettings.SetBbRenderDuringPresent(bbRenderDuringPresentSetting, isSpecific);
    EmulatorSettings.SetBbSkipLeftOutLevels(bbSkipLeftOutLevelsSetting, isSpecific);
    EmulatorSettings.SetBbDirectTextureCopies(bbDirectTextureCopiesSetting, isSpecific);
    EmulatorSettings.SetBbCopyThreads(
        std::array{0, 4, 6, 8}[std::clamp(bbCopyThreadsSetting, 0, 3)], isSpecific);
    EmulatorSettings.SetBbCachedTextureStaging(bbCachedTextureStagingSetting, isSpecific);
    EmulatorSettings.SetRenderdocEnabled(renderdocSetting, isSpecific);
    EmulatorSettings.SetBbRdocCaptureFrames(std::clamp(bbRdocCaptureFramesSetting, 1, 120),
                                            isSpecific);
    EmulatorSettings.SetBbLowerDetailFirst(bbLowerDetailFirstSetting, isSpecific);
    EmulatorSettings.SetBbClearShaderWritten(bbClearShaderWrittenSetting, isSpecific);
    EmulatorSettings.SetBbCeSkipHugeWrites(bbCeSkipHugeWritesSetting, isSpecific);
    EmulatorSettings.SetBbEvictKeepProtection(bbEvictKeepProtectionSetting, isSpecific);
    EmulatorSettings.SetBbEvictKeepReturning(bbEvictKeepReturningSetting, isSpecific);
    EmulatorSettings.SetBbProtectHelperThread(bbProtectHelperThreadSetting, isSpecific);
    EmulatorSettings.SetBbCpuCopyShader(bbCpuCopyShaderSetting, isSpecific);
    EmulatorSettings.SetBbCopySkipSame(bbCopySkipSameSetting, isSpecific);
    EmulatorSettings.SetBbClearOneLayer(bbClearOneLayerSetting, isSpecific);
    EmulatorSettings.SetBbSpreadLoads(bbSpreadLoadsSetting, isSpecific);
    EmulatorSettings.SetBbAnisotropy(bbAnisotropySetting == 0 ? 0 : 1 << bbAnisotropySetting,
                                     isSpecific);
    EmulatorSettings.SetBbDrawDataMemory(bbDrawDataMemorySetting, isSpecific);
    EmulatorSettings.SetBbTextureQuality(bbTextureQualitySetting, isSpecific);
    EmulatorSettings.SetBbIgnoreDisplayScale(bbIgnoreDisplayScaleSetting, isSpecific);
    EmulatorSettings.SetBbStutterDetail(bbStutterDetailSetting, isSpecific);
    EmulatorSettings.SetBbVramBudgetTestMb(bbVramBudgetTestSetting, isSpecific);

    /////////// Input Tab
    EmulatorSettings.SetMotionControlsEnabled(motionControlsSetting, isSpecific);
    EmulatorSettings.SetBackgroundControllerInput(backgroundControllerSetting, isSpecific);
    EmulatorSettings.SetCursorState(cursorStateSetting, isSpecific);
    EmulatorSettings.SetCursorHideTimeout(cursorTimeoutSetting, isSpecific);

    /////////// Trophy Tab
    EmulatorSettings.SetTrophyPopupDisabled(trophyPopupDisabledSetting, isSpecific);
    EmulatorSettings.SetTrophyNotificationSide(trophySideOptions.at(trophySideSetting), isSpecific);
    EmulatorSettings.SetTrophyNotificationDuration(static_cast<double>(trophyDurationSetting));

    /////////// Log Tab
    EmulatorSettings.SetLogEnable(logEnableSetting, isSpecific);
    EmulatorSettings.SetLogSeparate(logSeparateSetting, isSpecific);
    EmulatorSettings.SetLogSync(logSyncSetting, isSpecific);

    /////////// Experimental Tab
    if (isSpecific) {
        EmulatorSettings.SetReadbacksMode(readbacksModeSetting, true);
        EmulatorSettings.SetReadbackLinearImagesEnabled(readbackLinearImagesSetting, true);
        EmulatorSettings.SetDirectMemoryAccessEnabled(directMemoryAccessSetting, true);
        // Windows static guest red-zone protection
        EmulatorSettings.SetRedZonePatchingEnabled(windowsGuestRedZoneProtectionModeSetting, true);
        EmulatorSettings.SetDevKit(devkitConsoleSetting, true);
        EmulatorSettings.SetNeo(neoModeSetting, true);
        EmulatorSettings.SetShadNetEnabled(shadnetEnabledSetting, true);
        EmulatorSettings.SetConnectedToNetwork(connectedNetworkSetting, true);
        EmulatorSettings.SetPipelineCacheEnabled(pipelineCacheEnabledSetting, true);
        EmulatorSettings.SetPipelineCacheArchived(pipelineCacheArchiveSetting, true);
        EmulatorSettings.SetExtraDmemInMBytes(extraDmemSetting, true);
        EmulatorSettings.SetVblankFrequency(vblankFrequencySetting, true);
    }

    isSpecific ? EmulatorSettings.Save(profile) : EmulatorSettings.Save();

    // Switch the open window to the chosen display mode now rather than at the next launch.
    SDL_Event event{};
    event.type = SDL_EVENT_APPLY_DISPLAY_MODE;
    SDL_PushEvent(&event);
}

void SettingsWindow::SaveInstallDirs() {
    std::string profile;
    const bool isGlobal = currentProfile == "Global";
    if (!isGlobal) {
        profile = currentProfile.substr(0, 9);
        EmulatorSettings.Load();
    }

    EmulatorSettings.SetAllGameInstallDirs(m_GameInstallDirs);
    EmulatorSettings.Save();

    if (!isGlobal) {
        EmulatorSettings.Load(profile);
    }

    if (!isGameRunning) {
        GetProfileInfo();
    }
}

void SettingsWindow::GetProfileInfo() {
    GetGameIconInfo(profileIcons);

    BigPictureMode::IconInfo global;
    global.title = "Global";
    profileIcons.emplace(profileIcons.begin(), global);
}

SettingsWindow::SettingsWindow(bool gameRunning) : isGameRunning(gameRunning) {
    auto resource = cmrc::res::get_filesystem();
    auto loadTexture = [&](const std::string& resourcePath,
                           std::variant<SDL_Texture*, ImGui::RefCountedTexture>& texture) {
        auto file = resource.open(resourcePath);
        std::vector<u8> texData = std::vector<u8>(file.begin(), file.end());
        gameRunning ? texture = ImGui::RefCountedTexture::DecodePngTexture(texData)
                    : texture = BigPictureMode::LoadSdlTextureData(texData);
    };

    loadTexture("src/resources/big_picture/settings.png", generalTexture);
    loadTexture("src/resources/big_picture/experimental.png", experimentalTexture);
    loadTexture("src/resources/big_picture/graphics.png", graphicsTexture);
    loadTexture("src/resources/big_picture/bloodborne.png", bloodborneTexture);
    loadTexture("src/resources/big_picture/controller.png", inputTexture);
    loadTexture("src/resources/big_picture/trophy.png", trophyTexture);
    loadTexture("src/resources/big_picture/log.png", logTexture);
    loadTexture("src/resources/big_picture/folder.png", foldersTexture);
    loadTexture("src/resources/big_picture/profiles.png", profilesTexture);

    auto languageKeys = std::views::keys(languageMap);
    languageOptions.assign(languageKeys.begin(), languageKeys.end());

    currentProfile = "Global";
    m_GameInstallDirs = EmulatorSettings.GetAllGameInstallDirs();
    currentCategory = isGameRunning ? SettingsCategory::General : SettingsCategory::Profiles;

    bool customConfigFound = false;
    if (isGameRunning) {
        runningGameSerial = std::string(Common::ElfInfo::Instance().GameSerial());
        std::filesystem::path customConfigFile =
            Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs) /
            (runningGameSerial + ".json");

        if (std::filesystem::exists(customConfigFile)) {
            customConfigFound = true;
            currentProfile =
                runningGameSerial + " - " + std::string(Common::ElfInfo::Instance().Title());
        }
    } else {
        GetProfileInfo();
    }

    customConfigFound ? LoadSettings(runningGameSerial) : LoadSettings("Global");
}

void SettingsWindow::Prepare() {
    uiScale = EmulatorSettings.GetBigPictureScale() / 1000.f;
    openedFrame = ImGui::GetFrameCount();
}

// Esc, Circle on a controller, or (in game) F3 again: closes the window like Cancel. Only while
// it has the focus (not another window such as the F1 one), not while a popup, a dropdown or an
// edited field takes the key, nor in the first frames, which still see the key press that
// opened it.
bool SettingsWindow::CloseKeyPressed() const {
    if (ImGui::GetFrameCount() <= openedFrame + 2 ||
        !ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) ||
        ImGui::IsAnyItemActive() ||
        ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
        return false;
    }
    return ImGui::IsKeyPressed(ImGuiKey_Escape, false) ||
           ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false) ||
           (isGameRunning && ImGui::IsKeyPressed(ImGuiKey_F3, false));
}

void SettingsWindow::DeInit() {
    EmulatorSettings.Load();
    EmulatorSettings.SetBigPictureScale(static_cast<int>(uiScale * 1000));
    EmulatorSettings.Save();

    if (isGameRunning && !runningGameSerial.empty()) {
        EmulatorSettings.Load(runningGameSerial);
    }
}
void SettingsWindow::DrawSettings(bool* open, const std::function<void()>& applySettings) {
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f, 0.06f, 0.06f, 1.00f)); // black
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.20f, 0.40f, 0.70f, 1.00f));   // blue
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                          ImVec4(0.25f, 0.50f, 0.85f, 1.00f)); // lighter blue
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive,
                          ImVec4(0.26f, 0.59f, 0.98f, 0.80f)); // another light blue
    ImGui::PushStyleColor(ImGuiCol_SliderGrab,
                          ImVec4(0.26f, 0.59f, 0.98f, 0.80f)); // another light blue

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f * uiScale);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f * uiScale, 10.0f * uiScale));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f * uiScale, 10.0f * uiScale));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2.5f * uiScale);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f * uiScale, 20.0f * uiScale));
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 20.0f * uiScale);

    rowFocused = false;
    SetupWindow();
    DrawCategoryTabs();
    DrawMainContent(open, applySettings);

    ImGui::PopStyleVar(8);
    ImGui::PopStyleColor(5);

    ImGui::End();
}

void SettingsWindow::SetupWindow() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);

    ImGui::Begin("Settings", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::DrawPrettyBackground();
    ImGui::SetWindowFontScale(uiScale);

    if (ImGui::IsWindowAppearing()) {
        ImGui::SetKeyboardFocusHere();
    }

    // Navigate categories with Tab / R1 / L1, in the order the tabs are shown
    const bool next = ImGui::IsKeyPressed(ImGuiKey_GamepadR1) || ImGui::IsKeyPressed(ImGuiKey_Tab);
    const bool previous = ImGui::IsKeyPressed(ImGuiKey_GamepadL1);
    if (next || previous) {
        const auto categories = VisibleCategories();
        const auto it = std::ranges::find(categories, currentCategory);
        const int count = static_cast<int>(categories.size());
        const int index = it == categories.end() ? 0 : static_cast<int>(it - categories.begin());
        currentCategory = categories[(index + (next ? 1 : count - 1)) % count];
    }
}

std::vector<SettingsWindow::SettingsCategory> SettingsWindow::VisibleCategories() const {
    std::vector<SettingsCategory> categories;
    if (!isGameRunning) {
        categories.push_back(SettingsCategory::Profiles);
    }
    categories.insert(categories.end(), {SettingsCategory::General, SettingsCategory::Graphics,
                                         SettingsCategory::Input, SettingsCategory::Trophy,
                                         SettingsCategory::Folders, SettingsCategory::Log});
    if (currentProfile != "Global") {
        categories.push_back(SettingsCategory::Experimental);
    }
    categories.push_back(SettingsCategory::Bloodborne);
    return categories;
}

void SettingsWindow::DrawCategoryTabs() {
    ImVec4 settingsColor = ImVec4(0.1f, 0.1f, 0.12f, 0.8f); // Darker gray
    ImGui::PushStyleColor(ImGuiCol_ChildBg, settingsColor);

    float vertSize = (settingsIconSize * uiScale + ImGui::CalcTextSize("Profiles").y) +
                     ImGui::GetStyle().FramePadding.y * 4.f + 20.0 * uiScale;
    ImGuiChildFlags child_flags = ImGuiChildFlags_Borders | ImGuiChildFlags_NavFlattened;

    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

    ImGui::BeginChild("Categories", ImVec2(0, vertSize), child_flags, window_flags);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

    struct Tab {
        const char* name;
        std::variant<SDL_Texture*, ImGui::RefCountedTexture>* texture;
    };
    const auto tab_of = [&](SettingsCategory category) -> Tab {
        switch (category) {
        case SettingsCategory::Profiles:
            return {"Profiles", &profilesTexture};
        case SettingsCategory::General:
            return {"General", &generalTexture};
        case SettingsCategory::Graphics:
            return {"Graphics", &graphicsTexture};
        case SettingsCategory::Input:
            return {"Input", &inputTexture};
        case SettingsCategory::Trophy:
            return {"Trophy", &trophyTexture};
        case SettingsCategory::Folders:
            return {"Game Folders", &foldersTexture};
        case SettingsCategory::Log:
            return {"Log", &logTexture};
        case SettingsCategory::Experimental:
            return {"Experimental", &experimentalTexture};
        case SettingsCategory::Bloodborne:
        default:
            return {"Bloodborne", &bloodborneTexture};
        }
    };

    // Icons evenly spaced 30 apart; a name wider than its tab is shortened with "..." (the full
    // name shows when the icon is hovered).
    const auto categories = VisibleCategories();
    const float icon_width = settingsIconSize * uiScale + ImGui::GetStyle().FramePadding.x * 2.0f;
    const float pitch = icon_width + 30.0f * uiScale;
    const float max_text_width = pitch - 8.0f * uiScale;
    const float first_center = ImGui::GetCursorPosX() + icon_width * 0.5f;
    for (size_t i = 0; i < categories.size(); ++i) {
        const Tab tab = tab_of(categories[i]);
        AddCategory(tab.name, *tab.texture, categories[i],
                    first_center + pitch * static_cast<float>(i), max_text_width);
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImGui::EndChild();
}

void SettingsWindow::AddCategory(std::string name,
                                 std::variant<SDL_Texture*, ImGui::RefCountedTexture> texture,
                                 SettingsCategory category, float center_x, float max_text_width) {
    ImGui::SameLine();
    ImGui::BeginGroup();

    // Icon and name centred on center_x (DrawCategoryTabs spaces the tabs).
    const float icon_width = settingsIconSize * uiScale + ImGui::GetStyle().FramePadding.x * 2.0f;
    std::string label = name;
    if (ImGui::CalcTextSize(label.c_str()).x > max_text_width) {
        while (!label.empty() && ImGui::CalcTextSize((label + "...").c_str()).x > max_text_width) {
            label.pop_back();
        }
        while (!label.empty() && label.back() == ' ') {
            label.pop_back();
        }
        label += "...";
    }
    const float text_width = ImGui::CalcTextSize(label.c_str()).x;
    ImGui::SetCursorPosX(center_x - icon_width * 0.5f);

    // make button appear hovered as long as category is selected, otherwise dull its hovered color
    currentCategory == category
        ? ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered])
        : ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.235f, 0.392f, 0.624f, 1.00f));

    ImTextureID id = nullptr;
    if (std::holds_alternative<SDL_Texture*>(texture)) {
        id = ImTextureID(std::get<SDL_Texture*>(texture));
    } else if (std::holds_alternative<ImGui::RefCountedTexture>(texture)) {
        id = std::get<ImGui::RefCountedTexture>(texture).GetTexture().im_id;
    }

    if (id != nullptr) {
        if (ImGui::ImageButton(name.c_str(), id,
                               ImVec2(settingsIconSize * uiScale, settingsIconSize * uiScale))) {
            currentCategory = category;
        }
        if (label != name && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", name.c_str());
        }
    }

    ImGui::PopStyleColor();

    ImGui::SetCursorPosX(center_x - text_width * 0.5f);
    ImGui::Text("%s", label.c_str());
    ImGui::EndGroup();
}

void SettingsWindow::DrawMainContent(bool* open, const std::function<void()>& applySettings) {
    ImVec4 settingsColor = ImVec4(0.1f, 0.1f, 0.12f, 0.8f); // Darker gray
    ImGui::PushStyleColor(ImGuiCol_ChildBg, settingsColor);

    std::string centeredText;
    currentCategory == SettingsCategory::Folders
        ? centeredText = "Manage shadPS4 Game Folders"
        : centeredText = "Selected Profile: " + currentProfile;

    ImGui::Separator();
    Overlay::TextCentered(centeredText.c_str());
    ImGui::Separator();

    if (currentCategory == SettingsCategory::Profiles) {
        DrawProfileSelector();
    } else if (currentCategory == SettingsCategory::Folders) {
        DrawGameFolderManager();
    } else {
        DrawSettingsTable(currentCategory);
    }

    ImGui::PopStyleColor();

    ImGui::Separator();
    ImGui::SetNextItemWidth(300.0f * uiScale);
    static float sliderScale2 = 1.0f;
    if (ImGui::IsWindowAppearing()) {
        sliderScale2 = uiScale;
    }

    ImGui::SliderFloat("UI Scale", &sliderScale2, 0.25f, 3.0f);
    // Only update when user is not interacting with slider
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        uiScale = sliderScale2;
    }
    ImGui::SameLine();

    // Align buttons right
    float buttonsWidth = ImGui::CalcTextSize("Save").x + ImGui::CalcTextSize("Cancel").x +
                         ImGui::CalcTextSize("Apply").x + ImGui::GetStyle().FramePadding.x * 6.0f +
                         ImGui::GetStyle().ItemSpacing.x * 2;
    ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - buttonsWidth);

    // Controller shortcuts: d-pad right on a setting jumps to Save (right again: Apply,
    // Cancel); Options saves and closes (with the usual confirmation).
    const bool shortcuts =
        ImGui::GetFrameCount() > openedFrame + 2 && !ImGui::IsAnyItemActive() &&
        !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
    if (shortcuts && rowFocused && ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight, false)) {
        ImGui::SetKeyboardFocusHere();
    }
    if (ImGui::Button("Save") || (shortcuts && ImGui::IsKeyPressed(ImGuiKey_GamepadStart, false))) {
        closeOnSave = true;
        ImGui::OpenPopup("Save Confirmation");
    }

    ImGui::SameLine();
    if (ImGui::Button("Apply")) {
        ImGui::OpenPopup("Save Confirmation");
    }

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Save Confirmation", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("%s", ("Profile Saved:\n" + currentProfile).c_str());
        ImGui::Separator();

        if (ImGui::Button("OK", ImVec2(250 * uiScale, 0))) {
            std::string profile = currentProfile;
            if (currentProfile != "Global") {
                profile = currentProfile.substr(0, 9);
            }

            SaveSettings(profile);
            if (closeOnSave) {
                DeInit();
                *open = false;
                closeOnSave = false;
                ImGui::CloseCurrentPopup();
            } else {
                ImGui::CloseCurrentPopup();
            }
            if (applySettings) {
                applySettings();
            }
        }

        ImGui::EndPopup();
    }

    ImGui::SameLine();
    if (ImGui::Button("Cancel") || CloseKeyPressed()) {
        DeInit();
        *open = false;
        if (applySettings) {
            applySettings();
        }
    }
}

void SettingsWindow::DrawProfileSelector() {
    ImGuiChildFlags child_flags = ImGuiChildFlags_Borders | ImGuiChildFlags_NavFlattened;

    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;

    ImVec4 settingsColor = ImVec4(0.1f, 0.1f, 0.12f, 0.8f); // Darker gray
    ImGui::PushStyleColor(ImGuiCol_ChildBg, settingsColor);
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);

    ImGui::BeginChild("Profile Selection", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()),
                      child_flags, window_flags);

    ImGui::PopStyleColor();

    if (ImGui::BeginTable("ProfilesTable", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("CreateDeleteButton", ImGuiTableColumnFlags_WidthFixed,
                                400.0f * uiScale);
        ImGui::TableSetupColumn("Profile");

        for (int i = 0; i < profileIcons.size(); i++) {
            const std::filesystem::path customConfigFile =
                Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs) /
                (profileIcons[i].serial + ".json");
            const bool gameConfigExists = std::filesystem::exists(customConfigFile);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            std::string deleteButtonLabel = "Delete game-specific config##" + std::to_string(i);

            if (gameConfigExists) {
                // Different shades of red depending on button state
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.0f, 0.0f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.0f, 0.0f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.5f, 0.0f, 0.0f, 1.0f));

                if (ImGui::Button(deleteButtonLabel.c_str(),
                                  ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
                    deleteProfileIndex = i;
                }

                ImGui::PopStyleColor(3);
            }

            ImGui::TableNextColumn();
            std::string profileLabel = i == 0
                                           ? "Global"
                                           : profileIcons[i].serial + " - " +
                                                 profileIcons[i].title + "##" + std::to_string(i);
            if (ImGui::Button(profileLabel.c_str(),
                              ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {

                currentProfile = profileLabel;
                if (currentProfile == "Global") {
                    LoadSettings("Global");
                } else {
                    LoadSettings(profileIcons[i].serial);
                    if (!gameConfigExists) {
                        SaveSettings(profileIcons[i].serial);
                    }
                }
            }
        }

        ImGui::EndTable();
    }

    if (deleteProfileIndex != -1) {
        ImGui::OpenPopup("Confirm Delete");
    }

    ImGui::PopStyleVar(3);
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Confirm Delete", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        const std::string title = profileIcons[deleteProfileIndex].title;
        const std::string message = "Delete game-specific config file for " + title + "?";
        const std::filesystem::path path =
            Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs) /
            (profileIcons[deleteProfileIndex].serial + ".json");

        ImGui::Text("%s", message.c_str());
        ImGui::Separator();

        if (ImGui::Button("OK", ImVec2(120 * uiScale, 0))) {
            try {
                std::filesystem::remove(path);
            } catch (const std::exception& e) {
                LOG_ERROR(ImGui, "Could not delete config file {}: {}", path.string(), e.what());
            }

            deleteProfileIndex = -1;
            currentProfile = "Global";
            LoadSettings("Global");
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(120 * uiScale, 0))) {
            deleteProfileIndex = -1;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    ImGui::EndChild();
}

void SettingsWindow::DrawGameFolderManager() {
    ImGuiChildFlags child_flags = ImGuiChildFlags_Borders | ImGuiChildFlags_NavFlattened;

    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;

    ImGui::BeginChild("ContentRegion", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()), child_flags,
                      window_flags);

    if (ImGui::Button("Add Folder", ImVec2(400.f * uiScale, 0))) {
        ImGuiFileDialog::Instance()->OpenDialog("OpenFolder", "Add shadPS4 game folder", nullptr,
                                                ".", 1, nullptr,
                                                ImGuiFileDialogFlags_DisableCreateDirectoryButton |
                                                    ImGuiFileDialogFlags_DontShowHiddenFiles);

        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
    }

    float buttonWidth = ImGui::GetItemRectSize().x;
    ImGui::PushItemWidth(buttonWidth);
    if (ImGui::SliderInt("Folder Scan Depth", &scanDepthSetting, 0, 5)) {
        std::string profile;
        const bool isGlobal = currentProfile == "Global";
        if (!isGlobal) {
            profile = currentProfile.substr(0, 9);
            EmulatorSettings.Load();
        }

        EmulatorSettings.SetBigPictureFolderDepth(scanDepthSetting);
        EmulatorSettings.Save();

        if (!isGlobal) {
            EmulatorSettings.Load(profile);
        }

        if (!isGameRunning) {
            GetProfileInfo();
        }
    }
    ImGui::PopItemWidth();

    if (ImGuiFileDialog::Instance()->Display("OpenFolder", child_flags | ImGuiWindowFlags_NoMove)) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            GameInstallDir dir;
            dir.path = ImGuiFileDialog::Instance()->GetCurrentPath();
            dir.enabled = true;

#ifdef WIN32
            // replace \ with / on windows
            std::string pathString = dir.path.string();
            size_t pos = 0;
            while ((pos = pathString.find("\\", pos)) != std::string::npos) {
                pathString.replace(pos, 1, "/");
                pos += 2;
            }

            dir.path = pathString;
#endif

            m_GameInstallDirs.push_back(dir);
            SaveInstallDirs();
        }

        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::BeginChild("Game Folder List", ImVec2(0, 0), true, child_flags);

    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0, 5.0f * uiScale));
    if (ImGui::BeginTable("FoldersTable", 3)) {
        ImGui::TableSetupColumn("FolderButton", ImGuiTableColumnFlags_WidthFixed, 300.0f * uiScale);
        ImGui::TableSetupColumn("FolderEnabled", ImGuiTableColumnFlags_WidthFixed, 0.0f);
        ImGui::TableSetupColumn("FolderPath");

        for (int i = 0; i < m_GameInstallDirs.size(); i++) {
            std::string buttonLabel = "Remove Folder##" + std::to_string(i);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ImGui::Button(buttonLabel.c_str(), ImVec2(280 * uiScale, 0))) {
                m_GameInstallDirs.erase(m_GameInstallDirs.begin() + i);
                SaveInstallDirs();
            }

            ImGui::TableNextColumn();
            std::string checkboxLabel = "##EnableFolder" + std::to_string(i);
            bool previousState = m_GameInstallDirs[i].enabled;
            ImGui::Checkbox(checkboxLabel.c_str(), &m_GameInstallDirs[i].enabled);
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(5.0 * uiScale, 0));

            if (m_GameInstallDirs[i].enabled != previousState) {
                SaveInstallDirs();
            }

            ImGui::TableNextColumn();
            ImGui::TextWrapped("%s", m_GameInstallDirs[i].path.string().c_str());
        }

        ImGui::EndTable();
    }

    ImGui::PopStyleVar();
    ImGui::EndChild();
    ImGui::EndChild();
}

void SettingsWindow::DrawSettingsTable(SettingsCategory category) {
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f * uiScale, 10.0f * uiScale));
    ImGuiChildFlags child_flags = ImGuiChildFlags_Borders | ImGuiChildFlags_NavFlattened;

    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;

    ImGui::BeginChild("ContentRegion", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()), child_flags,
                      window_flags);

    if (category == SettingsCategory::General) {
        if (ImGui::BeginTable("SettingsTable", 2)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 500.0f * uiScale);
            ImGui::TableSetupColumn("Value");

            AddSettingCombo("Console Language", consoleLanguageSetting, languageOptions);
            AddSettingSliderInt("Volume", volumeSetting, 0, 500);
            AddSettingCheckbox("Show Splash Screen When Launching Game", showSplashSetting);
            AddSettingCombo("Audio Backend", audioBackendSetting, audioBackendOptions);

            ImGui::EndTable();
        }
    } else if (category == SettingsCategory::Graphics) {
        if (ImGui::BeginTable("SettingsTable", 2)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 500.0f * uiScale);
            ImGui::TableSetupColumn("Value");

            AddSettingCombo("Display Mode", fullscreenModeSetting, fullscreenModeOptions);
            AddSettingCombo("Present Mode", presentModeSetting, presentModeOptions);
            AddSettingSliderInt("Window Width", windowWidthSetting, 0, 8000);
            AddSettingSliderInt("Window Height", windowHeightSetting, 0, 7000);
            AddSettingCheckbox("Enable HDR", hdrAllowedSetting);
            AddSettingCheckbox("Enable FSR", fsrEnabledSetting);

            if (fsrEnabledSetting) {
                AddSettingCheckbox("Enable RCAS", rcasEnabledSetting);
            }

            if (rcasEnabledSetting && fsrEnabledSetting) {
                AddSettingSliderFloat("RCAS Attenuation", rcasAttenuationSetting, 0.0f, 3.0f, 3);
            }

            ImGui::EndTable();
        }
    } else if (category == SettingsCategory::Bloodborne) {
        if (ImGui::BeginTable("SettingsTable", 2)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 500.0f * uiScale);
            ImGui::TableSetupColumn("Value");

            // Bloodborne-only, experimental: everything the threaded renderer work added. Off
            // (the switch, or one toggle under it) is the standard shadPS4 behaviour.
            AddSettingSection("Threaded Renderer",
                              "Used by Bloodborne only. If something goes wrong, turn off "
                              "Threaded Renderer to get standard shadPS4 behaviour.",
                              true);
            {
                // Advanced settings (the threaded renderer's parts, developer tools): listed only
                // when shown (remembered).
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextWrapped("Threaded Renderer");
                ImGui::TableNextColumn();
                ImGui::Checkbox("##Threaded Renderer", &threadedRendererSetting);
                // With the Show/Hide button beside it, d-pad right goes to the button first
                // (ImGui's own navigation); from the button it jumps to Save as usual.
                const bool shown = EmulatorSettings.IsBbShowThreadedSettings();
                ImGui::SameLine();
                if (ImGui::SmallButton(shown ? "Hide advanced settings##threaded_parts"
                                             : "Show advanced settings##threaded_parts")) {
                    EmulatorSettings.SetBbShowThreadedSettings(!shown);
                    EmulatorSettings.Save();
                }
                NoteRowFocus();
                ImGui::SameLine();
                AddHint("Faster rendering and smoother frame pacing. Switches all below");
            }
            const bool advanced = EmulatorSettings.IsBbShowThreadedSettings();
            if (threadedRendererSetting) {
                AddSettingCheckbox("    Real occlusion tests", bbRealOcclusionSetting,
                                   "More fps in busy areas; can make textures flicker white");
                AddSettingCheckbox("    Merged copy shaders", bbMergedCopiesSetting,
                                   "Less GPU copy work; can cause white dots to flicker");
                if (advanced && bbMergedCopiesSetting) {
                    AddSettingCheckbox("        Sources from buffer memory",
                                       bbMergedNoStreamSourcesSetting,
                                       "White dots test: small copy sources not snapshotted "
                                       "from game memory");
                    AddSettingCheckbox("        Large batches", bbMergedLargeBatchesSetting,
                                       "White dots test: the old 64 MB batches instead of 64 KB");
                }
                AddSettingCheckbox("    Background shader compiles", bbBackgroundPipelinesSetting,
                                   "New objects appear a frame late instead of a stutter");
                AddSettingCheckbox("    Show frames when ready", bbFlipWhenReadySetting,
                                   "Smoother above 60 fps. Off: frames on vblank ticks");
            }
            if (threadedRendererSetting && advanced) {
                // Each off: the emulator's original behaviour for that part (troubleshooting).
                AddSettingCheckbox("    Streaming optimisations", bbStreamingSetting,
                                   "Smaller hitches entering new areas");
                AddSettingCheckbox("    GPU readbacks on the copy queue", bbReadbackQueueSetting,
                                   "Fewer small stutters (needs a restart)");
                AddSettingCheckbox("    Faster Windows locks and sleeps", bbFastLocksSetting,
                                   "Less CPU overhead (needs a restart)");
                AddSettingCheckbox("    Geometry in place", bbGeometryInPlaceSetting,
                                   "Static vertex/index data not copied for every draw");
                AddSettingCheckbox("    Background copies", bbBackgroundTextureCopiesSetting,
                                   "Copies textures and new geometry for the GPU on other threads");
                AddSettingCheckbox("    Skip draws with no output", bbSkipEmptyDrawsSetting,
                                   "Occlusion test geometry the emulator does not use");
                AddSettingCheckbox("    Reuse texture lookups", bbTextureSetMemoSetting,
                                   "A draw's textures found once, not every draw");
                AddSettingCheckbox("    Copy geometry ahead", bbRingGeometrySetting,
                                   "Small vertex/index data copied by the decoding thread");
                AddSettingCheckbox("    Overlap GPU submissions", bbSubmitOverlapSetting,
                                   "Next submission decoded while the last is recorded");
                AddSettingCheckbox("    Overlap game and GPU frames", bbOverlapFramesSetting,
                                   "The game starts its next frame while the last one finishes "
                                   "(needs a restart)");
                AddSettingCheckbox("    Early GPU data write-back", bbEarlyWritebackSetting,
                                   "GPU data the game writes over is copied back before it waits");
                AddSettingCheckbox("    Occlusion results in order", bbOcclusionInOrderSetting,
                                   "Occlusion results queued behind the draws, no wait");
                AddSettingCheckbox("    Fast pixel shader math", bbFastPixelMathSetting,
                                   "Driver fuses multiply-adds; shaders rebuilt (needs a restart)");
                AddSettingCheckbox("    Constant copies without waiting", bbConstantsNoWaitSetting,
                                   "Game's constant copies don't wait for queued draws");
                AddSettingCheckbox("    GPU interpolation where safe",
                                   bbHardwareInterpolationSetting,
                                   "NVIDIA: less pixel shader code (needs a restart)");
                AddSettingCheckbox("    Copy thread", bbCopyThreadSetting,
                                   "Draw memory copies on their own thread (needs a restart)");
                AddSettingCheckbox("    Keep rendering while presenting",
                                   bbRenderDuringPresentSetting,
                                   "Frame limiters like RTSS don't hold up rendering");
                AddSettingCheckbox("    Spread area loading", bbSpreadLoadsSetting,
                                   "Respawns and area loads upload textures over a few frames");
                AddSettingCheckbox("    Copy only kept texture levels", bbSkipLeftOutLevelsSetting,
                                   "Texture quality High/Medium: less data copied per upload");
                AddSettingCheckbox("    Direct texture copies", bbDirectTextureCopiesSetting,
                                   "Texture data read straight from game memory (faster loads)");
                AddSettingCombo(
                    "    Copy threads", bbCopyThreadsSetting,
                    {fmt::format("Auto ({})", Common::CopyPool::AutoWorkers()), "4", "6", "8"});
                if (ImGui::IsItemHovered()) {
                    ShowTooltip("Threads copying texture data when an area streams in: more "
                                "copy faster but compete with the game's threads (needs a "
                                "restart)");
                }
                AddSettingCheckbox("    Cached texture staging", bbCachedTextureStagingSetting,
                                   "Texture uploads staged in cached memory (faster copies)");
                AddSettingCheckbox("    Lower detail first", bbLowerDetailFirstSetting,
                                   "Streaming bursts: a new texture's largest level arrives a "
                                   "few frames later (softer meanwhile, shorter stutter)");
                AddSettingCheckbox("    Clear shader-written images", bbClearShaderWrittenSetting,
                                   "New images a compute shader fully overwrites are cleared "
                                   "instead of uploaded (shorter stutter)");
                AddSettingCheckbox("    Constant dumps past huge writes", bbCeSkipHugeWritesSetting,
                                   "The GPU thread keeps preparing draws in area loads instead of "
                                   "waiting for the draw recorder");
                AddSettingCheckbox("    Cheaper buffer refill", bbEvictKeepProtectionSetting,
                                   "With buffer eviction: evicted buffers come back without "
                                   "protecting their memory again (shorter stutter)");
                AddSettingCheckbox("    Keep returning buffers", bbEvictKeepReturningSetting,
                                   "With buffer eviction: loading screens free only what VRAM "
                                   "needs; buffers the game keeps using are freed last");
                AddSettingCheckbox("    Protect on a helper thread", bbProtectHelperThreadSetting,
                                   "New textures' memory protected by a helper thread instead "
                                   "of the draw recorder (shorter stutter)");
                AddSettingCheckbox("    Copy shader on the CPU", bbCpuCopyShaderSetting,
                                   "The game's small GPU copies are done in game memory, so the "
                                   "game doesn't wait for the GPU (stutters while running)");
                AddSettingCheckbox("      Skip copies that change nothing", bbCopySkipSameSetting,
                                   "Fewer buffer uploads");
                AddSettingCheckbox("    Clear shadow cube faces", bbClearOneLayerSetting,
                                   "No wedges of light in lamp shadows");
            }

            AddSettingSection("Graphics", "Picture quality.", false);
            AddSettingCombo("Anisotropic filtering", bbAnisotropySetting, anisotropyOptions);
            if (ImGui::IsItemHovered()) {
                ShowTooltip("Sharper ground and stairs at an angle (needs a restart)");
            }
            AddSettingCombo("Texture quality", bbTextureQualitySetting, textureQualityOptions);
            if (ImGui::IsItemHovered()) {
                ShowTooltip("Less VRAM for big textures (the BB Remaster mod's) by leaving "
                            "out their sharpest level:\nHigh: textures over 2048, Medium: "
                            "over 1024 (about 500 MB less in busy areas).\nApplies to "
                            "textures loaded after a change: from the next loading screen.");
            }

            AddSettingSection("Performance", "How the game's data reaches the graphics card.",
                              false);
            // The per-draw data the CPU writes every frame: in graphics card memory with Resizable
            // BAR / Smart Access Memory, else system memory. The log says which was used.
            AddSettingCombo("Resizable BAR / Smart Access Memory", bbDrawDataMemorySetting,
                            drawDataMemoryOptions);
            if (ImGui::IsItemHovered()) {
                ShowTooltip("Puts the data the game changes every frame straight into "
                            "graphics card memory.\nNeeds Resizable BAR (NVIDIA, Intel) or "
                            "Smart Access Memory (AMD) enabled in the BIOS.\nAuto detects "
                            "it. Try Off if you get stutter or crashes. (needs a restart)");
            }
            AddSettingCheckbox("Protect only changed pages", bbExactProtectionSetting,
                               "Less CPU time protecting game memory the GPU reads");
            AddSettingCheckbox("Batched texture uploads", bbBatchedUploadsSetting,
                               "New textures of a draw uploaded together");

            AddSettingSection(
                "VRAM", "Graphics card memory in long sessions and with texture mods.", false);
            AddSettingCheckbox("VRAM cleanup", bbVramReclaimSetting,
                               "Frees unused VRAM in long sessions (needs a restart)");
            if (bbVramReclaimSetting) {
                AddSettingCheckbox(
                    "    Also free idle buffers", bbBufferEvictionSetting,
                    "Buffer memory unused for 30 s near the limit (needs a restart)");
                AddSettingCheckbox("    Clean up at loading screens", bbLoadScreenCleanupSetting,
                                   "Frees what the last area used; the next one loads its own");
                AddSettingCheckbox("    Use more VRAM", bbMoreVramSetting,
                                   "Cleans up only close to the driver's VRAM budget");
                if (bbMoreVramSetting) {
                    AddSettingCheckbox("        Go past the budget", bbVramPastBudgetSetting,
                                       "Up to the card's size; Windows moves other programs out "
                                       "(needs a restart)");
                }
            }
            AddSettingSection("Interface", "The emulator's in-game windows (F1, F3, pop-ups).",
                              false);
            AddSettingCheckbox("Menus sized by screen pixels", bbIgnoreDisplayScaleSetting,
                               "Ignores Windows display scaling (125%, 150%), as games do");

            AddSettingSection("Diagnostics", "For troubleshooting. Slower when on: leave off.",
                              false);
            AddSettingCheckbox("Diagnostics in the log", threadedRendererDiagnosticsSetting,
                               "Frame time, stutter, VRAM and readback reports");
            AddSettingCheckbox("Detailed stutter reports", bbStutterDetailSetting,
                               "With diagnostics: where each stutter's time went");
            // Times every draw: heavy on the CPU.
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.2f, 1.0f), "(costs fps)");
            if (advanced) {
                AddSettingCheckbox("GPU time per pass in the log",
                                   threadedRendererGpuProfileSetting,
                                   "Which render passes take the GPU's time");
                // Testing the VRAM cleanup: act as if the GPU had this much VRAM (0 = real).
                AddSettingSliderInt("VRAM budget override (MB, 0 = real)", bbVramBudgetTestSetting,
                                    0, 24000);
                AddSettingCheckbox("RenderDoc captures (F12)", renderdocSetting,
                                   "Loads an installed RenderDoc (needs a restart)");
                AddSettingSliderInt("Frames per RenderDoc capture", bbRdocCaptureFramesSetting, 1,
                                    120);
            }

            ImGui::EndTable();
        }
    } else if (category == SettingsCategory::Input) {
        if (ImGui::BeginTable("SettingsTable", 2)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 500.0f * uiScale);
            ImGui::TableSetupColumn("Value");

            AddSettingCheckbox("Enable Motion Controls", motionControlsSetting);
            AddSettingCheckbox("Enable Background Controller Input", backgroundControllerSetting);
            AddSettingCombo("Hide Cursor", cursorStateSetting, hideCursorOptions);

            if (cursorStateSetting == 1) {
                AddSettingSliderInt("Hide Cursor Idle Timeout", cursorTimeoutSetting, 1, 10);
            }

            ImGui::EndTable();
        }
    } else if (category == SettingsCategory::Trophy) {
        if (ImGui::BeginTable("SettingsTable", 2)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 500.0f * uiScale);
            ImGui::TableSetupColumn("Value");

            AddSettingCheckbox("Disable Trophy Notification", trophyPopupDisabledSetting);
            if (!trophyPopupDisabledSetting) {
                AddSettingCombo("Trophy Notification Position", trophySideSetting,
                                trophySideOptions);
                AddSettingSliderFloat("Trophy Notification Duration", trophyDurationSetting, 0.f,
                                      10.f, 1);
            }

            ImGui::EndTable();
        }
    } else if (category == SettingsCategory::Log) {
        if (ImGui::BeginTable("SettingsTable", 2)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 500.0f * uiScale);
            ImGui::TableSetupColumn("Value");

            AddSettingCheckbox("Enable Logging", logEnableSetting);
            if (logEnableSetting) {
                AddSettingCheckbox("Separate Log Files", logSeparateSetting);
                AddSettingCheckbox("Log Sync", logSyncSetting);
            }

            ImGui::EndTable();
        }
    } else if (category == SettingsCategory::Experimental) {
        if (ImGui::BeginTable("SettingsTable", 2)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 500.0f * uiScale);
            ImGui::TableSetupColumn("Value");

            AddSettingSliderInt("Additional DMem Allocation", extraDmemSetting, 0, 20000);
            AddSettingSliderInt("Vblank Frequency", vblankFrequencySetting, 30, 360);
            AddSettingCombo("Readbacks Mode", readbacksModeSetting, readbacksModeOptions);
            AddSettingCheckbox("Enable Readback Linear Images", readbackLinearImagesSetting);
            AddSettingCheckbox("Enable Direct Memory Access", directMemoryAccessSetting);
#ifdef _WIN32
            // Windows static guest red-zone protection
            AddSettingCheckbox("Windows Guest Red Zone Protection (Requires Restart)",
                               windowsGuestRedZoneProtectionModeSetting);
#endif
            AddSettingCheckbox("Enable Devkit Console Mode", devkitConsoleSetting);
            AddSettingCheckbox("Enable PS4 Neo Mode", neoModeSetting);
            AddSettingCheckbox("Enable ShadNet", shadnetEnabledSetting);
            AddSettingCheckbox("Set Network Connected to True", connectedNetworkSetting);
            AddSettingCheckbox("Enable Shader Cache", pipelineCacheEnabledSetting);

            if (pipelineCacheEnabledSetting) {
                AddSettingCheckbox("Compress Shader Cache to Zip File",
                                   pipelineCacheArchiveSetting);
            }

            ImGui::EndTable();
        }
    }

    ImGui::PopStyleVar();
    ImGui::EndChild();
}

void SettingsWindow::AddSettingCheckbox(std::string name, bool& value) {
    std::string label = "##" + name;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    ImGui::TextWrapped("%s", name.c_str());
    ImGui::TableNextColumn();
    ImGui::Checkbox(label.c_str(), &value);
    NoteRowFocus();
}

// "(needs a restart)" in a hint or tooltip stands out in amber, without the brackets.
constexpr std::string_view RestartNote = "(needs a restart)";
constexpr ImVec4 RestartColour{1.0f, 0.78f, 0.35f, 1.0f};

static std::pair<std::string_view, bool> SplitRestartNote(std::string_view text) {
    const auto at = text.find(RestartNote);
    if (at == std::string_view::npos) {
        return {text, false};
    }
    auto before = text.substr(0, at);
    while (!before.empty() && before.back() == ' ') {
        before.remove_suffix(1);
    }
    return {before, true};
}

void SettingsWindow::AddHint(std::string_view hint) {
    const auto [text, restart] = SplitRestartNote(hint);
    ImGui::TextDisabled("%.*s", static_cast<int>(text.size()), text.data());
    if (restart) {
        ImGui::SameLine();
        ImGui::TextColored(RestartColour, "needs a restart");
    }
}

void SettingsWindow::ShowTooltip(std::string_view tooltip) {
    const auto [text, restart] = SplitRestartNote(tooltip);
    ImGui::BeginTooltip();
    ImGui::TextUnformatted(text.data(), text.data() + text.size());
    if (restart) {
        ImGui::TextColored(RestartColour, "Needs a restart");
    }
    ImGui::EndTooltip();
}

void SettingsWindow::AddSettingCheckbox(std::string name, bool& value, const char* hint) {
    AddSettingCheckbox(name, value);
    ImGui::SameLine();
    AddHint(hint);
}

void SettingsWindow::AddSettingSection(const char* title, const char* note, bool experimental) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::Dummy(ImVec2(0.0f, ImGui::GetTextLineHeight() * 0.5f));
    ImGui::Separator();
    // Amber for experimental settings, the usual text colour otherwise.
    const ImVec4 colour =
        experimental ? ImVec4(1.0f, 0.72f, 0.25f, 1.0f) : ImGui::GetStyleColorVec4(ImGuiCol_Text);
    ImGui::TextColored(colour, "%s", title);
    ImGui::TableNextColumn();
    ImGui::Dummy(ImVec2(0.0f, ImGui::GetTextLineHeight() * 0.5f));
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", note);
    ImGui::PopTextWrapPos();
}

void SettingsWindow::AddSettingSliderInt(std::string name, int& value, int min, int max) {
    std::string label = "##" + name;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextWrapped("%s", name.c_str());

    ImGui::TableNextColumn();
    ImGui::SliderInt(label.c_str(), &value, min, max);
    NoteRowFocus();
}

void SettingsWindow::AddSettingSliderFloat(std::string name, float& value, int min, int max,
                                           int precision) {
    std::string label = "##" + name;
    std::string precisionString = "%." + std::to_string(precision) + "f";

    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextWrapped("%s", name.c_str());

    ImGui::TableNextColumn();
    ImGui::SliderFloat(label.c_str(), &value, min, max, precisionString.c_str());
    NoteRowFocus();
}

void SettingsWindow::AddSettingCombo(std::string name, int& value,
                                     std::vector<std::string> options) {
    std::string label = "##" + name;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextWrapped("%s", name.c_str());

    ImGui::TableNextColumn();
    const char* combo_value = options[value].c_str();
    const bool combo_open = ImGui::BeginCombo(label.c_str(), combo_value);
    if (!combo_open) {
        NoteRowFocus();
    }
    if (combo_open) {
        for (int i = 0; i < options.size(); i++) {
            const bool selected = (i == value);
            if (ImGui::Selectable(options[i].c_str(), selected))
                value = i;

            // Set the initial focus when opening the combo
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
}

} // namespace ImGuiEmuSettings
