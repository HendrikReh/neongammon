#!/bin/sh
# SPDX-License-Identifier: MPL-2.0
# Keep the compiler, flags, model assets and all unrelated native objects fixed.
set -eu
: "${ENGINE_PYTHON:?Set ENGINE_PYTHON to the verified 3.2.1 interpreter}"
out=${1:?Supply a fresh output directory}
mkdir -p "$(dirname "$out")"
mkdir "$out"
out=$(cd "$out" && pwd)
experiment=$(cd "$(dirname "$0")" && pwd)
root=$(git -C "$experiment" rev-parse --show-toplevel)
mkdir "$out/src"
git -C "$root" archive 34558d2 cpp | tar -xf - -C "$out/src"
cmake -S "$out/src/cpp" -B "$out/native" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DBGBOT_USE_ACCELERATE=ON -DBGBOT_BUILD_APPS=OFF -DPYBIND11_FINDPYTHON=ON \
    -DPython_EXECUTABLE="$ENGINE_PYTHON" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
for variant in baseline candidate dot_only trace; do
    if [ "$variant" != baseline ]; then
        # The LUT control changes only the candidate's sigmoid. Trace the pristine kernel.
        if [ "$variant" = trace ]; then
            git -C "$root" show 34558d2:cpp/src/neural_net.cpp > "$out/src/cpp/src/neural_net.cpp"
        fi
        patch -d "$out/src" -p1 < "$experiment/$variant.patch"
    fi
    cmake --build "$out/native" --target bgbot_cpp
    mkdir "$out/$variant"
    cp "$out/native"/bgbot_cpp*.so "$out/$variant/"
done
