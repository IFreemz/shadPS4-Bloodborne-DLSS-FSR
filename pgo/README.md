<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# PGO profiles

Profile-guided optimisation: the compiler lays out and inlines code by how it is really used.

1. **Training build:** run the Build workflow by hand (Actions → Build → Run workflow) with
   **pgo = generate**. The artifact ends in `-pgo-training`. It runs somewhat slower than a normal build.
2. **Train:** play with it for 10-20 minutes in typical places (areas, combat, menus, loading).
   Each run writes `shadps4-<number>.profraw` into the emulator's `user/pgo` folder
   (`%APPDATA%\shadPS4\pgo` for a normal install) when the emulator is closed normally.
3. **Commit** the `.profraw` files into this folder (several runs are merged together).
4. **Release builds** (pgo = auto, the default) merge every `.profraw` / `.profdata` here with the
   runner's `llvm-profdata` and build with the result; the artifact ends in `-pgo`.

A profile from an older version still helps, but only a little: every function that changed since
gets no profile at all (silently). Make a new training build and profile after large changes, and
delete the old `.profraw` files then. Delete the files here to build without PGO.

## Local builds

`x64-Clang-Release` stays a quick build without LTO or PGO. Two presets build with ThinLTO, as the
release builds do, each in a folder of its own:

1. Training build: `cmake --preset x64-Clang-PGO-Training`, then
   `cmake --build Build/pgo-training-x64-Clang-Release --parallel`.
2. Play with `Build\pgo-training-x64-Clang-Release\shadPS4.exe` as above (DLSS, frame generation and
   the frame rate as you normally play), and quit the game normally. Each run leaves a
   `.profraw` in `%APPDATA%\shadPS4\pgo`.
3. Merge them with the LLVM that built them (Visual Studio's, in `VC\Tools\Llvm\x64\bin`):
   `llvm-profdata merge -output=pgo\bb.profdata %APPDATA%\shadPS4\pgo\*.profraw`
4. Build with it: `cmake --preset x64-Clang-Release-PGO`, then
   `cmake --build Build/pgo-x64-Clang-Release --parallel`.

`pgo/bb.profdata` is not committed; copy the `.profraw` files here for the release builds.

## What it can speed up

PGO and LTO make the emulator's own code faster: the GPU command thread, the draw recorder, the
copy threads, the shader recompiler and the system libraries the game calls. The game's own code
runs natively and is not changed by them. So the frame rate only gains when one of the emulator's
threads limits it, not when the GPU does. To see the difference, compare where the GPU has room
(a lower output resolution, frame generation off, the FPS cap Uncapped, the same spot standing
still) and compare the log's `Frame pacing` rate and the `GPU command thread waited` and
`draw recorder busy` figures.
