#!/bin/sh
# SPDX-License-Identifier: MPL-2.0
# Rebuild the same source tree, changing only the candidate grouping.

set -eu
: "${ENGINE_PYTHON:?Set ENGINE_PYTHON to the verified 3.2.1 interpreter}"
out=${1:?Supply a fresh output directory}
mkdir -p "$(dirname "$out")"
mkdir "$out"
out=$(cd "$out" && pwd)
experiment=$(cd "$(dirname "$0")" && pwd)
root=$(git -C "$experiment" rev-parse --show-toplevel)
mkdir "$out/src"
git -C "$root" archive a4bb46b cpp | tar -xf - -C "$out/src"
cmake -S "$out/src/cpp" -B "$out/native" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DBGBOT_USE_ACCELERATE=ON -DBGBOT_BUILD_APPS=OFF -DPYBIND11_FINDPYTHON=ON \
    -DPython_EXECUTABLE="$ENGINE_PYTHON" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
for variant in baseline candidate; do
    if [ "$variant" = candidate ]; then
        patch -d "$out/src" -p1 < "$experiment/candidate.patch"
    fi
    cmake --build "$out/native" --target bgbot_cpp --parallel 5
    mkdir "$out/$variant"
    cp "$out/native"/bgbot_cpp*.so "$out/$variant/"
done
