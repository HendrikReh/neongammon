// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bindings_common.h"
#include "bgbot/neural_net.h"

PYBIND11_MODULE(bgbot_cpp, m) {
    using namespace bgbot::python;
    m.doc() = "Backgammon bot C++ engine";
    m.attr("ABI_VERSION") = kAbiVersion;
    m.attr("ACCELERATE_ENABLED") = bgbot::accelerate_enabled();

    bind_core(m);
    bind_training(m);
    bind_rollout(m);
    bind_eval(m);
    bind_cube_rollout(m);
}
