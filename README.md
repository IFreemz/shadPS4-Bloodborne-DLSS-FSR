<!--
SPDX-FileCopyrightText: Copyright 2026 IFreemz
SPDX-License-Identifier: GPL-2.0-or-later
-->

# shadPS4 Bloodborne DLSS & FSR

> [!NOTE]
> **This project is now maintained by [@eFMann](https://github.com/eFMann).** It was created by
> [@IFreemz](https://github.com/IFreemz), who built the DLSS and FSR integration, frame generation
> and keyboard and mouse controls. Huge thanks to eFMann for taking it over, and
> for the threaded renderer, frame pacing, character motion vectors and VRAM work he has already
> contributed. Old links to `IFreemz/shadPS4-Bloodborne-DLSS-FSR` redirect here.

> **New: [2.1.0-pre.4](https://github.com/eFMann/shadPS4-Bloodborne-DLSS-FSR/releases/tag/v2.1.0-pre.4) (experimental).** Resolution in F1 (no patch file editing), faster rendering and fewer stutters: a lighter threaded renderer, background shader compiles and a saved driver cache, batched texture uploads. Should fix the white screen edge on AMD with upscaling (#24) and the crash when loading a save. The stable build is still [2.0.1](https://github.com/eFMann/shadPS4-Bloodborne-DLSS-FSR/releases/tag/v2.0.1). Feedback is welcome in the [issues](https://github.com/eFMann/shadPS4-Bloodborne-DLSS-FSR/issues).

Temporal upscaling for Bloodborne on [shadPS4](https://github.com/shadps4-emu/shadPS4): NVIDIA
DLSS on GeForce RTX cards, AMD FSR 3.1 on everything else, plus frame generation (DLSS on RTX 40
and 50 series, up to 4x on RTX 50; FSR on every other GPU).

This is a modified shadPS4 0.20.0. The game renders at the resolution set by its resolution patch
(say 2560x1440) and is upscaled to your window (say 3840x2160). It is real temporal upscaling: the
scene is jittered and the upscaler gets depth plus camera and character motion, so it rebuilds
detail instead of sharpening a stretched image. The HUD is drawn on top afterwards and stays
crisp.

DLSS looks better, especially from low render resolutions; FSR 3.1 runs on any GPU. Without the
upscaler DLLs the build behaves exactly like shadPS4 0.20.0.

| 1280x720, DLSS off | 1280x720 upscaled to 3840x2160 with DLSS |
|---|---|
| ![720p native](documents/dlss/720p-native.webp) | ![720p to 4K with DLSS](documents/dlss/720p-to-4k-dlss.webp) |

Both rendered at 1280x720; open them full size to compare.

**Jump to:** [Install](#install) · [Frame generation](#frame-generation) ·
[HDR](#hdr-sdr-to-hdr) · [Keyboard and mouse](#keyboard-and-mouse) · [Tested setups](#tested-setups) · [Settings](#settings) ·
[Troubleshooting](#troubleshooting) · [FAQ](#faq)

## Requirements

- Windows 10 or 11, 64-bit
- Bloodborne v1.09 already running in shadPS4 (any region; tested with CUSA03173)
- DLSS: an RTX 20 series or newer card and a recent driver (ships DLSS 310.9.1, DLSS 4.5)
- DLSS Frame Generation: an RTX 40 or 50 series card, a recent driver, Windows 10 20H1 or newer
  with *Hardware-accelerated GPU scheduling* on (Settings > Display > Graphics)
- FSR 3.1 and FSR frame generation: any GPU that runs shadPS4
- FSR 4 add-on (optional): a GPU with INT8 dot products and compute shader derivatives, such as
  a Radeon RX 6000 or newer, an RTX card or an Intel Arc card

## Install

1. Download the latest zip from [Releases](../../releases).
2. Copy `shadPS4.exe` and all the DLLs into your shadPS4 folder, replacing its `shadPS4.exe`
   (back it up if you want to switch back).
   - `shadps4_dlss.dll`, `nvngx_dlss.dll`: DLSS, used on RTX cards only
   - `amd_fidelityfx_vk.dll`: FSR 3.1 upscaling and FSR frame generation
   - `sl.*.dll`, `nvngx_dlssg.dll`: DLSS Frame Generation (NVIDIA Streamline), loaded only when
     frame generation is on
3. In the Bloodborne patches (shadPS4 patch manager or BBLauncher's Patches tab), enable a
   **Resolution Patch** (see the table below), **Disable AA**, **Disable Chromatic Aberration**
   and **Disable DoF**, and leave **Disable Motion Blur** off (see below).
4. Start the game and press **F1** in gameplay. The panel should say *Active* and show the
   upscaler and resolutions.

Without a launcher, double-click `shadPS4.exe` and pick Bloodborne from the game list.

To go back to normal rendering, delete the DLLs or untick *Upscaling enabled* in F1.

**With BBLauncher:** unzip this build into its own folder. In BBLauncher, click *Manage Builds* >
*Add Local Build*, pick this `shadPS4.exe`, tick it under *Selected* and start the game as
usual. Don't copy the files over a build BBLauncher downloaded: updating it brings back the
normal `shadPS4.exe`.

### Optional: FSR 4 add-on

FSR 4 is AMD's machine-learning upscaler, sharper and steadier than FSR 3.1, mainly for Radeon
and Intel Arc cards (on RTX cards DLSS is the better fit). It is a separate download from
[Releases](../../releases), `shadPS4-Bloodborne-FSR4-addon-…-win64.zip`.

1. Put the `fsr4` folder from the zip next to `shadPS4.exe` (the folder, not its contents).
2. Start the game. *Automatic* now uses FSR 4 on cards without DLSS; you can also pick
   *FSR 4 (add-on)* in F1.

Delete the folder to remove it. It works up to 3840x2160 output and costs more GPU time than
FSR 3.1. For 2x and 3x upscaling it uses the balanced model, since the performance models leave
trails behind moving characters in this game.

It runs the FSR 4 "v07" INT8 model through [FSR-Vulkan](https://github.com/FireBurn/FSR-Vulkan),
with model files built by the Q2RTX project from the FSR 4 source AMD published under the MIT
license. It is not AMD's official FSR 4.1 DLL, which exists only for DirectX 12.

### Frame generation

Tick *Frame generation* in F1 and restart. The game then shows generated frames between the ones
it renders, so 60 fps becomes 120 (or more with Multi Frame Generation). The game's HUD, the F1
panel and the fps counter stay still in generated frames. F1 shows which frame generation runs
and both frame rates (for example "FSR 60 -> 120"); *Frame gen FPS counter* shows them in the
top-left corner of the game.

*Frame gen type* picks which one runs (restart to apply):

| | Cards | Notes |
|---|---|---|
| **Automatic** (default) | all | DLSS on RTX 40/50, FSR on everything else |
| **DLSS** | RTX 40/50 | NVIDIA's AI frame generation; Multi Frame Generation on RTX 50 |
| **FSR** | any GPU | AMD FSR 3.1 frame generation, 2x only |

With DLSS on RTX 50, *Frame gen multiplier* offers 2x, 3x or 4x. Aim for the game's frame rate
times the multiplier to be close to your refresh rate (60 x 4 on a 240 Hz screen); frames above
it are wasted. *Lower input delay (Reflex)* (DLSS only) does what it says but can lower the
frame rate when the GPU is saturated or the generated output is limited by VSync.

Tips:

- **Use a 60 fps base**: *Uncap FPS++* or *60 FPS++* with vblank 60 in shadPS4's settings. Frame
  generation holds back a frame, which adds input delay; small at 60 fps, noticeable at 30, where
  moving characters also smear, more so at 3x and above.
- **Leave GPU headroom**: frame generation costs about 2-3 ms per frame at 4K. If the base frame
  rate drops below 60 and gets uneven, it stutters instead of smoothing. Lower the render
  resolution if needed, for example native 1080p on a 4K screen, or the 1280x720 patch on 1440p.
- **Close overlays** (NVIDIA app overlay, RivaTuner/RTSS): they break frame generation.
- **Use Fullscreen or Borderless** and keep the game focused; frame generation pauses in the
  background. With DLSS and *Fullscreen*, the build takes the display exclusively.
- **Present mode**: DLSS frame generation uses Immediate. Disable driver-forced VSync for
  `shadPS4.exe`; it can lower the base frame rate. FSR keeps your selected present mode.
- **Other base rates work too**: vblank 45 with *Uncap FPS++* gives 45 to 90. A steady base
  matters more than a high one. Set the rate with vblank: outside limiters (RTSS, NVIDIA app)
  don't cap generated frames reliably.

It turns itself off in menus, loading screens and while the window resizes. Multi Frame
Generation is DLSS only, and with the RenoDX HDR mod frame generation always uses FSR.

Setups players tested are under [Tested setups](#tested-setups).

#### DLSS frame generation on other NVIDIA cards (unofficial mods)

NVIDIA limits DLSS frame generation to RTX 40/50 and Multi Frame Generation to RTX 50.
Third-party mods lift those limits. This build doesn't include or support them, but doesn't
block them either: it asks Streamline what the card can do, like any game.

| Card | Mod | Gives | Status |
|---|---|---|---|
| RTX 20 / 30 | [RTX30MFG-Unlock](https://github.com/mcsoderh/RTX30MFG-Unlock) | frame generation and MFG | reported working on RTX 30; RTX 20 untested by its author |
| RTX 40 | [RTX40MFG-Unlock](https://github.com/dashdogy/RTX40MFG-Unlock) | MFG (3x to 6x) | tested here on an RTX 4090 |
| RTX 50 | none needed | MFG up to 4x | native |

On AMD and Intel, use FSR frame generation.

To install one, put its DLL next to `shadPS4.exe` as `winmm.dll` (or `version.dll`; shadPS4
also loads `dbghelp.dll` and `dxgi.dll`, but not `winhttp.dll` or `dinput8.dll`). Don't replace
any of this build's DLLs, and install **only one** unlock: they patch the same code and clash.
Set the mod's menu (Backspace) to *Follow game* so F1's *Frame gen multiplier* controls it. On
RTX 40, 3x and up show a little more ghosting than 2x, since it runs RTX 50 code on older
hardware.

These mods are unofficial and experimental: use them at your own risk, download them only from
their original page and report their problems to their authors.

Frame generation uses NVIDIA [Streamline](https://github.com/NVIDIA-RTX/Streamline) (DLSS-G and
Reflex) or AMD's FidelityFX SDK (FSR 3.1.4), loaded only when the setting is on.

### Which resolution patch

The resolution patch sets the render resolution; the upscaler goes from there to your window
size, keeping the aspect ratio.

| Window | Resolution Patch | Result |
|---|---|---|
| 3840x2160 | 2560x1440 | Quality mode (recommended for 4K) |
| 3840x2160 | none (native 1920x1080) | Performance mode, faster and a bit softer |
| 2560x1440 | none (native 1920x1080) | Quality mode |
| 2560x1440 | 2560x1440 | Anti-aliasing only (DLAA / FSR Native AA) |
| 1920x1080 | 1280x720 | Quality mode, for slower GPUs |
| 1920x1080 | none (native 1920x1080) | Anti-aliasing only |

16:10 and ultrawide work with a matching Resolution Patch. Explicit `output=WxH` values
also keep the patched aspect ratio instead of stretching the image.

FSR struggles more than DLSS from low resolutions: with FSR, render at 1920x1080 or more for a
4K window.

A render resolution above the window is anti-aliased and scaled down. The upscaler goes up to 3x
per side; beyond that the rest is plain scaling. Higher render resolutions need more VRAM, and
the resolution patch notes about raising dmem still apply.

### Decoupled 1080p UI patches

These patches render the scene lower (for example 1280x720) and keep the UI at 1080p. The build
detects them: the scene is upscaled to 1080p, then the game adds its effects and its own sharp
UI. Use one instead of a regular Resolution Patch, not with one. The 640x360, 960x540, 1280x720
and 1600x900 versions are supported; with 320x180 and those above 1080p the upscaler stays off
(F1 says why).

They are meant for 1080p screens and low VRAM. The result is 1080p scaled to your window, so on
1440p or 4K the regular Resolution Patches look sharper. FSR shimmers more on thin detail from
720p and below; DLSS copes well. Sharpness applies here too.

### HDR: SDR to HDR

Turn on **Use HDR** in the Windows display settings, then tick **SDR to HDR** in F1 and restart
the game. It works on any GPU (AMD, Intel, NVIDIA) and with DLSS and FSR frame generation;
brightness and peak sliders are in F1. Thanks to @zeroznet, who wrote it.

It converts the finished image like Special K's SDR to HDR does, so it can't bring back detail the
game already clipped: very bright lights, fire and the moon stay flat white, just brighter, and
the HUD gets brighter with the game. RenoDX (below) rebuilds the game's lighting in HDR and looks
better where it works.

### Keyboard and mouse

F1 → **Keyboard & mouse** → **Use keyboard & mouse controls**, then restart the game once. WASD
moves, the mouse turns the camera like a PC game (update 1.09; on other versions it acts as the
right stick), left click attacks, right click is the firearm, Space dodges, the arrow keys and
Enter work the menus. Every action can be rebound in F1 (click a cell, press the key), with mouse
sensitivity and invert. F7 turns the mouse camera off and on. Your previous controls are kept:
**Restore previous controls** puts them back. A controller keeps working alongside.

### HDR with RenoDX

> [!WARNING]
> **RenoDX doesn't work with 2.0.0, 2.0.1 and the 2.1.0-pre.2 and later builds yet.** shadPS4 0.20.0 compiles the game's shaders
> differently, so RenoDX doesn't recognise them until it gets an update. For HDR, use
> [1.8.6](https://github.com/eFMann/shadPS4-Bloodborne-DLSS-FSR/releases/tag/v1.8.6) or [2.1.0-pre.1](https://github.com/eFMann/shadPS4-Bloodborne-DLSS-FSR/releases/tag/v2.1.0-pre.1) (experimental) for now.

The [RenoDX](https://github.com/clshortfuse/renodx) HDR mod for Bloodborne (ReShade 6.8.0 or
newer) works with 1.8.6 and 2.1.0-pre.1, with every upscaler, patch type and frame generation. It's out of
early access: get it free from the Bloodborne channel on the
[RenoDX Discord](https://discord.gg/F6AUTeWJHM). If you enjoy it, donate to the RenoDX devs, they
deserve it.

Use 1.8.2 or newer. In 1.8.0 and 1.8.1, upscaling with a regular Resolution Patch capped
highlights at about 200 nits; 1.7.0 and earlier didn't support RenoDX.

1.8.5 restores RenoDX's Game/UI Brightness and Bloom controls with BB PC Remaster 1.01 + 1.07.
Recognized remaster files are corrected in memory, whether installed as a mod overlay or merged
into the game. No game files are changed; SDR and unmodded rendering are left alone.

- Turn the **Disable AA** patch off: with RenoDX it turns skin blue. With the recognized remaster,
  1.8.5 automatically ignores **Disable AA**, **Disable Bloom** and **Disable Chromatic Aberration**
  for HDR, so RenoDX can control those passes.
- **Restore original moon lighting** in F1 is optional and off by default. Restart after changing it.
- Frame generation always uses FSR with RenoDX.
- G-SYNC/FreeSync screens can flicker in HDR with frame generation off.

**Disclaimer:** RenoDX for Bloodborne is NVIDIA only for now, and a future RenoDX update may
break it here until this build catches up. Report problems with this build here, not to the
RenoDX authors.

### Other patches

Keep *Better AA* and *Enable TAA* off: the upscaler replaces the game's anti-aliasing.
*60FPS++ (no deltatime)* may crash while an area loads; use *60 FPS++*. *Enable Screen Space
Reflections* turns off object motion for the upscaler, so characters and weapons can smear; keep
it off for the sharpest image. Patches and mods that
players tested with this build are under [Tested setups](#tested-setups).

### Why the three "Disable" patches, and not Disable Motion Blur

The game's AA, chromatic aberration and depth of field blur the image before the upscaler sees
it. With them off, it gets a clean image.

**Disable Motion Blur (Perf Increase) must stay off.** It also removes the depth and motion data
the upscaler needs, and F1 stays on *Waiting for gameplay*. This build turns the game's motion
blur off by itself (*Game motion blur* below) and keeps the data.

F1 checks these patches and has a button that sets all four (applies after a restart).

## Tested setups

Setups tested with this build. Tested something else? Open an issue and it can go here.

**Graphics mods** (by EnglishDave_): the BB PC Remaster, upscaled armour, more blood, upscaled
UI and better cloth physics mods work with this build.

**Patches**: *Performance Patch*, *rgba8f color space*, *lower specific renders*, *HD Motion
Blur*, *Disable SSAO*, *Enable Screen Space Reflections*, *Model LOD -2*, *Increased Graphics
Heap Sizes*, *Increased camera distance*, *Restore Change Appearance Feature*, *1440p Light Grid*
and *Optimal 1080p*.

**Frame generation**

- RTX 4090, 4K 240 Hz, native 1080p, *60 FPS++*, vblank 60, v1.8.4: DLSS and FSR 60 to 120.
  With an unlock mod (see *Frame generation*), DLSS holds 60 base fps up to 6x with Reflex
  sleep on. 2x and 3x look clean; 4x and above show vibration or tearing-like artifacts during
  movement on this setup. The cause is still under investigation.
- [@TheFurryMonk](https://github.com/TheFurryMonk)
  ([#4](https://github.com/eFMann/shadPS4-Bloodborne-DLSS-FSR/issues/4)): RTX 5070 Ti, 1440p
  144 Hz G-Sync, 2560x1440 patch (DLAA), *1440p Light Grid*. Smoothest: vblank 60 with
  *Uncap FPS++* or *60 FPS++*, capped at 120 with RTSS (NVIDIA Reflex limiter, inject after
  frame presentation). vblank 90 gives 180 fps, above 144 Hz, so it judders and tears now and
  then; better on 180 Hz or faster.
- EnglishDave_, RTX 5080: 2x is the most stable, 3x the best compromise. *Lower input delay*
  may be needed on RTX 50 cards. *Readbacks Mode* on causes a little stutter for him.
- [@sunlighthell](https://github.com/sunlighthell)
  ([#11](https://github.com/eFMann/shadPS4-Bloodborne-DLSS-FSR/issues/11)): RTX 5080,
  Ryzen 7 9800X3D, 64 GB RAM, DLSS preset M, 2x frame generation. Reported smooth 60 base /
  120 displayed fps, with occasional dips to 118-119, from character creation through Cleric
  Beast. Shader cache on, Immediate, vblank 120, driver G-SYNC/VSync on, extra memory `9999`,
  Special K for HDR. Patches: *Performance Patch*, *Disable AA*, *Model LOD -2*, *60 FPS++*,
  *4K Light Grid*, *HD Motion Blur 4K*. Mods: *Bloodborne Visual Upgrade* and Xbox buttons/UI.

## Settings

Press **F1** in game. Mouse, keyboard and controller work; the game ignores the controller while
the panel is open.

- **Upscaling enabled**: compare with the normal image at any time.
- **Upscaler**: *Automatic* uses DLSS if the card supports it, else FSR 4 if its add-on is
  installed, else FSR 3.1. Any of them can be picked by hand.
- **DLSS model**: M (default) and L are DLSS 4.5, K (NVIDIA's recommendation for this mode) and
  J are DLSS 4.
- **Sharpness**: after upscaling, default 0.6; 0 for the plain look.
- **Game motion blur**: off by default. The game's blur leaves ghost trails around your
  character when the camera moves; tick it to bring it back.
- **Frame generation**: off by default; applies after a restart, then the tick toggles it live.
- **Frame gen type**: Automatic, DLSS or FSR; applies after a restart.
- **Frame gen multiplier**: 2x to 4x with DLSS on cards with Multi Frame Generation.
- **Lower input delay (Reflex)**: DLSS frame generation; less input delay, possibly fewer fps.
- **Frame gen FPS counter**: the game's frame rate and the displayed one, top left.
- **Menu fix**: on by default. While a menu covers most of the screen, the game's own image is
  shown and jitter pauses, so nothing shimmers behind it. Same frame rate; small popups keep the
  upscaled image.

Settings live in `dlss.ini` in the shadPS4 user folder (`user` next to `shadPS4.exe`, or
`%APPDATA%\shadPS4`). Hand edits apply within a second.

## Troubleshooting

F1 shows why upscaling isn't running.

- **"not an NVIDIA GPU" / "does not support DLSS"**: DLSS needs an RTX card; *Automatic* uses FSR
  instead. On laptops, make sure shadPS4 runs on the NVIDIA GPU (Windows graphics settings or
  NVIDIA Control Panel).
- **"NGX initialization failed"**: update your NVIDIA driver.
- **"shadps4_dlss.dll is missing or from a different version"**: copy all files from the same
  zip.
- **"FSR needs amd_fidelityfx_vk.dll"**: copy that DLL next to `shadPS4.exe`.
- **"FSR 4 needs the FSR 4 add-on" / "FSR 4 could not start"**: check the `fsr4` folder is next
  to `shadPS4.exe` with all its files and that your GPU meets the requirements. The log says why
  (search for `[FSR4]`).
- **Crash at boot after a "Config Migration" prompt**: you copied the files into a build that
  uses `config.toml` (ShadLix and other forks). This release and newer carry the extra dmem over;
  otherwise set it again (`"extra_dmem_in_mbytes": 8000` under `"General"` in
  `user/custom_configs/CUSA03173.json`). Pick *Copy* if it asks about saves.
- **Stretched polygons, flickering, missing parts of the world**: readbacks are off (shadPS4's
  default), with or without the upscaler. Press **F3** > *Experimental*, set *Readbacks Mode* to
  *Relaxed* and restart. F1 warns about it.
- **Missing faces or arms (the Doll, your character) or other glitches**: close RivaTuner / MSI
  Afterburner (a per-game profile isn't enough) and the NVIDIA app overlay, and check Readbacks
  Mode is *Relaxed*. An FPS patch matching the vblank (*60 FPS++* with 60, *90 FPS++* with 90)
  also helps.
- **"DLSS Frame generation needs …"**: F1 names what's missing (RTX 40/50 card, newer driver or
  Hardware-accelerated GPU scheduling). *Automatic* uses FSR frame generation instead.
- **Crash on AMD Radeon when the upscaler starts**: fixed in 1.7.0
  ([#7](https://github.com/eFMann/shadPS4-Bloodborne-DLSS-FSR/issues/7)). If it still happens,
  attach `user/log/shad_log.txt` there.
- **Frame generation on, but not smoother**: close overlays (NVIDIA app overlay, RivaTuner), keep
  the game focused and check the GPU isn't maxed out (see *Frame generation*).
- **Memory use grows over a long session**: a lower render resolution slows it down.
- **"Waiting for gameplay"**: expected in menus, the title screen and loading screens. If it stays
  in gameplay, check that Disable Motion Blur is off.
- **F1 doesn't open**: check `input_config/global.ini` in the user folder for
  `hotkey_toggle_dlss = f1`; you can bind another key there.
- **G-SYNC / FreeSync flicker**: uneven frame pacing in emulation, not DLSS. Use an FPS patch
  matching your vblank (60 FPS++ with 60) rather than *Uncap FPS++*.
- **No sound effects in menus**: if Windows uses *Dolby Atmos for Home Theater* (or another
  spatial sound format) for your audio device, turn it off in Windows' sound settings. Thanks to
  @DarthBane768 for tracking this down
  ([#15](https://github.com/eFMann/shadPS4-Bloodborne-DLSS-FSR/issues/15)).

## FAQ

**FSR 4?** Yes, as the optional add-on above. AMD's official FSR 4.1 is DirectX 12 only; the
add-on uses the community Vulkan port of the v07 model.

**XeSS?** No; FSR 3.1 covers Intel cards. XeSS frame generation is DirectX 12 only.

**Frame generation?** Built in, see *Frame generation*, including mods for DLSS frame
generation on RTX 20, 30 and 40.

**HDR?** Yes: SDR to HDR in F1 on any GPU (see *HDR: SDR to HDR*), NVIDIA RTX HDR, or RenoDX on 1.8.6 or 2.1.0-pre.1 (it doesn't work on the shadPS4 0.20.0 builds yet).

**Other games?** No. It depends on how Bloodborne renders: which shaders draw the HUD, where the
camera matrices live and which passes write motion data.

**Linux or Steam Deck?** No, Windows only for now.

**Is this an official shadPS4 release?** No, it is a separate build based on shadPS4 0.20.0.
Please report problems here, not to the shadPS4 team.

## How it works

For the curious, or for porting this to a newer shadPS4:

- **Jitter**: draws into the main scene depth buffer get a sub-pixel viewport offset (Halton
  2,3). Full-screen passes and HUD draws aren't jittered.
- **Camera motion**: built in double precision from the view and projection matrices in the
  game's scene constants. The game's own reprojection matrix for motion blur is float32 with
  large world coordinates, and its ~0.6 px rounding noise made the image shimmer.
- **Object motion**: the game's velocity draws are replayed at full resolution into a separate
  buffer, with camera motion filling in elsewhere. The replay needs the depth scaled to full
  resolution: a blit where the GPU can blit depth, a few small draws elsewhere (AMD).
- **Where the upscaler runs**: at the first HUD draw, on a copy of the finished scene. At the
  game's copy to the screen, the HUD is added back from its own render and the game's
  brightness/gamma table is applied at output resolution.
- **HUD and menus**: the HUD is treated as a dimming of the scene plus an overlay, so darkening
  overlays dim the upscaled image. When the HUD covers most of the screen (a menu), the game's
  own frame is shown and jitter pauses.
- **Frame pacing**: the vblank timer sleeps until shortly before each deadline and waits out the
  rest precisely, for even intervals on VRR displays.
- **Decoupled UI patches**: the game scales its HDR scene up to the UI resolution in one pass
  before post-processing. When that pass gets a smaller scene, DLSS/FSR (with automatic
  exposure) replaces it. Scene depth for jitter comes from the draws into that HDR scene.
- **Motion blur**: two full-screen passes along a motion buffer built from depth and velocity.
  With *Game motion blur* off they become plain copies, so the motion data stays.
- **Upscaler inputs**: colour, depth, motion vectors and jitter in render pixels, the same for
  DLSS and FSR. FSR also gets the camera's near/far planes and field of view.
- **NVIDIA code**: shadPS4 itself has none. `shadps4_dlss.dll` (source in
  [`dlss_bridge/`](dlss_bridge)) is a small bridge to the DLSS SDK, loaded at runtime if
  present.
- **AMD code**: `amd_fidelityfx_vk.dll` is AMD's signed FidelityFX SDK 1.1.4 build, loaded at
  runtime if present; only its headers are here ([`externals/ffx-api`](externals/ffx-api)).
- **FSR 4 add-on**: `fsr4/shadps4_fsr4.dll` (source in [`fsr4_addon/`](fsr4_addon)) links
  FSR-Vulkan's v07 provider and loads the model files next to it. With the folder present, the
  device also enables INT8 dot products and compute shader derivatives. Its colour input is
  converted to linear light, since FSR 4 has no setting for display-encoded input.
- **DLSS frame generation**: Streamline is loaded before the Vulkan instance and hands out its
  versions of the instance, device and swapchain functions. Each upscaled frame keeps its depth,
  motion and camera matrices in a small ring until presented. DLSS-G also gets the window without
  the emulator's overlays and a mask of them. Reflex markers are set around presenting.
- **FSR frame generation**: the device gets extra queues (present, acquire, compute) and the
  swapchain goes through AMD's frame generation swapchain. It gets the same data as DLSS-G and
  the frame without the HUD, and presents the in-between frame from its own threads.

Menus, the title screen and loading screens aren't upscaled.

## Building

The emulator builds like normal shadPS4 (see [documents/building-windows.md](documents/building-windows.md))
and doesn't need the DLSS SDK.

The bridge needs the [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS):

```
cmake -S dlss_bridge -B build-dlss -G Ninja -DCMAKE_BUILD_TYPE=Release -DDLSS_SDK_ROOT=<path to DLSS SDK>
cmake --build build-dlss
```

The FSR 4 add-on needs [FSR-Vulkan](https://github.com/FireBurn/FSR-Vulkan) and the Vulkan SDK
(or a `vulkan-1` import library via `-DVULKAN_LIBRARY`):

```
cmake -S fsr4_addon -B build-fsr4 -G Ninja -DCMAKE_BUILD_TYPE=Release -DFSR_VULKAN_ROOT=<path to FSR-Vulkan>
cmake --build build-fsr4
```

Its model files are `fsr4_model_v07_i8_*`, `rcas.spv` and `spd_auto_exposure.spv` from
[Q2RTX's `baseq2/fsr4_shaders`](https://github.com/FireBurn/Q2RTX/tree/ae8d628fae208813172446d1e49ed94150b04658/baseq2/fsr4_shaders).

Frame generation needs the [Streamline SDK](https://github.com/NVIDIA-RTX/Streamline/releases)
2.14.1 headers: add `-DBB_STREAMLINE_DIR=<path to the Streamline SDK>` when configuring the
emulator. `sl.*.dll` and `nvngx_dlssg.dll` come from its `bin/x64` folder.

`nvngx_dlss.dll` comes from the DLSS SDK (`lib/Windows_x86_64/rel`), `amd_fidelityfx_vk.dll`
from the [AMD FidelityFX SDK v1.1.4](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tree/v1.1.4)
(`PrebuiltSignedDLL`).

## AI disclosure

An AI coding agent was used in the development of this project.

## Credits

- IFreemz, the original author: the DLSS and FSR upscaling and frame generation integration
  (camera and object motion, jitter, HUD handling, Reflex), the F1 panel and keyboard and mouse
  controls with the mouse camera.
- eFMann, the current maintainer: the threaded renderer, frame pacing, character motion vectors,
  the VRAM cleanup rounds and buffer eviction, the shadPS4 0.20.0 port, the RTX 3060 DLSS
  frame-pacing diagnosis and the Reflex options reapplication patch.
- Cainhurst27 (u/EnglishDave_), for finding and fixing shadPS4's long-standing VRAM leak, and for
  testing.
- [shadPS4](https://github.com/shadps4-emu/shadPS4) and its contributors, for the emulator.
- deadinside28's [bloodborne_pc](https://github.com/deadinside28/bloodborne_pc), whose notes on
  the scene constants and viewport jitter were the starting point for camera motion and jitter
  here, and whose FSR 4 integration the add-on follows.
- FireBurn's [FSR-Vulkan](https://github.com/FireBurn/FSR-Vulkan), which runs FSR 4 on Vulkan, and
  the Q2RTX project, which built the FSR 4 v07 model files.
- The authors of the Bloodborne patches for shadPS4 (Kyo, illusion and others), including the
  resolution patches and the AA, chromatic aberration and depth-of-field patches.
- The [RenoDX](https://github.com/clshortfuse/renodx) authors, for the HDR mod.
- zeroznet, for SDR to HDR.
- NVIDIA, for DLSS, Streamline and Reflex, and AMD, for FSR and the open FidelityFX SDK.

## License

- shadPS4 and the changes to it: GPL-2.0-or-later, see [LICENSE](LICENSE).
- The DLSS bridge in `dlss_bridge/`: MIT, see [dlss_bridge/LICENSE.txt](dlss_bridge/LICENSE.txt).
- `nvngx_dlss.dll` and `nvngx_dlssg.dll`: NVIDIA DLSS SDK license, not covered by the licenses
  above.
- The Streamline DLLs (`sl.*.dll`): NVIDIA's Streamline SDK, MIT.
- `amd_fidelityfx_vk.dll` and `externals/ffx-api/`: AMD FidelityFX SDK, MIT, see
  [externals/ffx-api/LICENSE.txt](externals/ffx-api/LICENSE.txt).
- The FSR 4 add-on: its wrapper is GPL-2.0-or-later and FSR-Vulkan inside it is MIT. The model
  files carry the MIT notice of the FSR 4 source they were built from; the add-on zip includes
  all three license texts.

NVIDIA, GeForce RTX and DLSS are trademarks of NVIDIA Corporation. AMD and FidelityFX are
trademarks of Advanced Micro Devices, Inc. Bloodborne is a trademark of Sony Interactive
Entertainment; this project is not affiliated with Sony, FromSoftware, NVIDIA, AMD or the
shadPS4 team, and includes no game files.
