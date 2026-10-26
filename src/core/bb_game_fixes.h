// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Core::BbGameFixes {

// Fixes in Bloodborne 1.09's own code, installed after the patch file at game load. Each checks
// the game's bytes first and is left out when they differ (another version, or a patch there).
void Install();

// True while the game shows a loading screen (respawn, travel, loading a save, quitting to the
// title). Always false when the game's code differs from 1.09's there.
bool IsLoadingScreen();

} // namespace Core::BbGameFixes
