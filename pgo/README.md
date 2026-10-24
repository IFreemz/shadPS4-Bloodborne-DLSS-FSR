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

A profile from an older version still helps: code that changed since is just not optimised by it.
Make a new training build and profile after large changes. Delete the files here to build without PGO.
