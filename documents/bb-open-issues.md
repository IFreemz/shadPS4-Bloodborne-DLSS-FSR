<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Open issues (notes)

Status as of 29 Oct 2026. Test setup as in `bb-dlc-judder-investigation.md` (RTX 3060 Laptop
6 GB, BB PC Remaster mod, DLSS).

The debugging tools named below (F1 "Log motion blur inputs" and "Reload textures from game
memory", F3's threaded renderer parts, GPU time per pass, RenderDoc) are listed only after F3
"Show advanced settings". "Real occlusion tests" is off by default since BbDefaultsVersion 4, and
F1's screen-space reflections choice is hidden (the game's own reflections).

## Whole picture blurred after a respawn — fix under test

- Cause found: the game's 640x360 motion target (drawn and read as 8_8_8_8 unorm) shares memory
  with other targets, so the texture cache sometimes keeps it as an R32 or R16G16 float image.
  Its "no motion" clear (0.498, 0.498, 1, 0) was done in the image's format, so it held float
  bytes, which the game's motion blur read back as huge motion. Draws were fine: their views were
  8-bit.
- Fix: "Clear in the game's format" (Graphics, off by default). Fast clears and compute clears
  go through a view in the game's format. The log lists (first 16) every clear of an image kept
  in another format.
- Not confirmed yet: no run so far has logged a mismatched clear of the motion target itself.
- F1 "Log motion blur inputs" logs the passes' inputs, reads their textures back (as their
  format and as bytes) and follows the motion target's writers.

## Whole scene white after a respawn — seen once, not reproduced

- Seen once in the first run with "Clear in the game's format" on (with "Constant dumps past
  huge writes" and "Cheaper buffer refill" off). HUD fine, 60 fps; turning motion blur off and
  "Reload textures from game memory" did not fix it.
- The clear fix acted once in that run: a 512x512 sRGB texture cleared to black as unorm (the
  same bytes), before the respawn. No buffer evictions. The loading screen freed 1611 textures
  and kept 435 GPU-written ones.
- May be related to the blur bug (memory reused at a respawn holding data in the wrong
  format), most likely the game's exposure data, which carries over between frames.
- Next time: press "Log motion blur inputs" (scene colour before and after DLSS), turn DLSS off
  in F1, then respawn.

## White flash of the blood echo effect (5-10x larger) — not solved

- Sometimes, for a split second. Size and brightness come from constants and vertex data, so
  the suspects are "Constant dumps past huge writes" and "Cheaper buffer refill" first, then
  "Lower detail first" (texel fetches past the edge of the smaller level), then "Protect on a
  helper thread". A one-frame bad motion target (the blur bug) would also smear it.
- Test plan: halves (those first two off, then the other two).

## Leg armour with rainbow stripes after a respawn — seen once

- Looks like wrong data or memory that was never written. Suspects: "Lower detail first" (70 of
  640 textures never got their largest level in that run), upload trimming/skipping, "Protect
  on a helper thread". F1 "Reload textures from game memory" tells a stale cached copy from bad
  game data.

## Metadata reads

- The game reads compressed textures at addresses still registered as the CMask of a 640x360
  render target from an earlier area (stale metadata registration). The warning names both.

## Real occlusion: one-frame flash of sky/white round corners and through doorways — paused

- Seen only with "Real occlusion tests" on. With the F3 debug option "tell the game every test is
  visible" (the tests still run, are waited for and logged) it never happens, so the game is told
  "hidden" for an area that is visible: the cause is an answer, not the tests' handling.
- Ruled out by tests (it still flashed): render resolution and moving the tests to the render-size
  depth (also at native), the upscaler and its jitter (upscaler off), old depth (100% same-frame),
  one-pixel gaps ("tolerant of a pixel" removed those zeros), late answers or early reads (every
  read held: 0-7/s let go once the clear waits were answered past the protection), pixel counts
  too small (counts at the game's resolution), and a group of data settings (geometry in place,
  copy geometry ahead, constant copies, constant dumps, copy shader on the CPU).
- Not found by the logs: the game hands out result slots differently from frame to frame, so a
  test cannot be followed by its slot or its place in the frame. The count may also only drop
  below a threshold of the game's, not to 0.
- Next: a RenderDoc capture of a normal frame at the doorway to find the room's test box (its
  shader, geometry, constants), then a log that follows that box and starts a RenderDoc capture
  by itself when its count drops sharply (the flash frame), with the box's draw state that frame
  and the frame before.
- Already tried, did not fix the flash, and removed (code in commit 47f61bf; do not try these
  again as fixes):
  - Quick occlusion answers (a thread writing each answer as soon as the GPU has it): answers
    reached the game in the same frame 93% of the time, still flashed.
  - Visible until answered (a test reads "visible" until its answer arrives): still flashed.
  - Submit occlusion tests at once (tests sent to the GPU right after the game makes them):
    answers back sooner, still flashed.
  - Wait for occlusion answers (results held from the game until answered, up to 6 ms): still
    flashed, and it ruined frame pacing with Uncap FPS++: answers take ~8 ms, so the game's
    render thread lost the full 6 ms in some frames (game frames alternating ~10/14 ms, 80 fps
    feeling like 40-50).
  - Occlusion counts at the game's resolution (counts scaled from the render size): still
    flashed.
  - Occlusion tests tolerant of a pixel (tests against a dilated depth copy): removed the
    one-pixel-gap zeros, still flashed.
  - The debug option "tell the game every test is visible" (removed earlier): confirmed the
    cause is a "hidden" answer.
- Settings now: Real occlusion tests and Occlusion results in order only. The logs still count
  one-frame hidden blips and how late answers reach the game (the starting point next time).
