# DLC start area: stutters after loading (investigation notes)

Status as of 24 Oct 2026: load spikes and camera stutters fixed; the long "judder" after a
respawn is **not solved**. These notes are so the next attempt doesn't start from zero.

Test setup: RTX 3060 Laptop (6 GB), i7-11800H, 32 GB RAM, Windows 10, BB PC Remaster mod,
1280x720 render to 1920x1080 (DLSS), window 2560x1440, present mode Immediate.

## Fixed (committed)

| Problem | Cause | Fix |
|---|---|---|
| Camera-turn stutters (GPU waits 50-630 ms) | ~1 GB of VRAM held but unused by the allocator (256 MiB blocks half empty), pushing the rest out of VRAM | VMA blocks 64 MiB |
| Load-in spike (242 ms) | 67 upload blocks (1 GiB) allocated in the load frame | 1 GiB of upload memory kept ready on 16 GB+ PCs |
| Camera bursts over the VRAM pressure mark | GPU-drawn images all written back at once | Write-backs paced (128 MiB/s, 16 MiB at once) |
| VRAM over budget after respawns | New area loaded while the old one's textures were still there | Freeing ahead of an area load (2 s idle, 512 MiB below the pressure mark) |
| Respawn load frame (130-165 ms) | ~1000 textures uploaded in one frame | Spread area loading: 64 MiB of textures per frame for 6 s after a load (new ones black meanwhile; nobody noticed) — load frame now ~85 ms |

Diagnostics added (diagnostics on): GPU profile logs single slow frames (50 ms or twice the
usual); stutter lines show the GPU timeline (executing / idle between submissions), queue submit
calls, and presentation (acquire, present, frame waits); the swapchain's present mode is logged.

## Tried and dropped

- Waiting for critical VRAM before writing back GPU-drawn images: worse long term (bursts later).
- Reusing freed image memory for 10 s: never kicked in (VRAM always above the allowance).
- Keeping identical textures on reload (reload check): 80-90 % of a respawn's textures are the
  same data, but **at new addresses**; hashing everything costs as much CPU as it saves.
- Batching page protection: no effect (tried earlier).
- Filler GPU work on loading screens (GPU clocks): ran, no change.
- GPU frames ahead 2 and 3: no change.

## The long judder (unsolved)

What the player sees: 2-3 s of heavy judder (~30 ms frames, about 33 fps, against 12-14 ms
normal), shortly after a respawn or the first load **in the DLC start area only**, and **only if
you start running straight away**. Waiting 1-2 s before running avoids it.

What the logs show (diagnostics on, judder frames):

- CPU side is light and normal: preparing draws ~6-8 ms, draw recorder busy ~10 ms, no
  texture uploads, ~2 MiB buffer uploads, no staging allocations.
- The GPU command thread waits 17-26 ms per frame for the GPU (frames-ahead wait,
  `vk_presenter.cpp` PrepareFrame).
- With frames ahead 2, the GPU timeline shows **~30 ms executing per frame, ~0.3 ms idle**: the
  GPU really is busy the whole time (with frames ahead 1 only one of the two submissions per
  frame gets measured, which once looked like "11 ms of work").
- Not presentation: acquire 0 ms, present calls 0.3 ms, no frame waits (Immediate mode).
- Not the driver blocking us: queue submit calls 0.1-0.2 ms.
- Not VRAM pressure (3.6-4.2 GB of 5.2 GB budget; a lower VRAM budget test changed nothing), not
  the 270 MB burst of new render targets 2-3 s after loading (judder also when that burst was
  cheap), not GPU clocks (filler work), not frames ahead.
- Render passes per judder frame: ~150-160 ended, ~36-43 split and resumed — **the same number of
  splits as normal frames**. The clear difference: **about 25 more barrier flushes per frame**
  (`Runtime::FlushBarriers`, vk_runtime.cpp) during the judder (49-53 against 24-29).
- The per-pass GPU profile (10 s averages) barely changes between normal and judder windows; the
  slow-frame lines with "twice the usual" miss these frames because the usual adapts upwards.
- The game prints `== Stall during rendering at flush N` (its own message) around the load and
  the following bursts.

Leading ideas, untested:

1. Barriers: something during the judder adds ~25 full barriers a frame, each making the GPU wait
   for earlier work. Find who requests them (image layout transitions vs buffer accesses), e.g.
   count `Runtime::Transit`/`AccessBuffer` callers per frame, as the pass-end counting did.
2. Real occlusion tests: this area is where they matter most and the game issues more while
   moving. **First thing to try: respawn, run at once, with "Real occlusion tests" off in F3.**
3. Then the same with Spread area loading, Copy thread, Background texture copies off, and DLSS
   switched to FSR/native — one switch per run.
4. RenderDoc: hard to use here (lower fps, RTSS overlay missing, hard to time F12). If used: start
   BBLauncher from RenderDoc's Launch Application with "Capture Child Processes" ticked; shadPS4's
   own F12 then captures a frame (it takes a screenshot instead when RenderDoc isn't loaded).
   Compare Statistics and GPU Duration of a judder frame against a normal one.

Also seen: two "Device lost" crashes at the start of a respawn loading screen, both after the
same last draws/uploads. Rare. Suspects: buffer eviction (see the #40 fix) or the paced
write-backs, which run at that moment.
