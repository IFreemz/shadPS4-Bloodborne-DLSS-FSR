// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: MIT
//
// Forced into every source of the Linux FSR 3.1 library: what AMD's MSVC-only sources expect.

#pragma once

#ifdef __cplusplus
#include <new>
#endif

#define __declspec(x) __attribute__((visibility("default")))
