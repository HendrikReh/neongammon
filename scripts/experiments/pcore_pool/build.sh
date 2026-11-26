#!/bin/sh
# SPDX-License-Identifier: MPL-2.0
# Use one build tree so the compiler, flags and unmodified objects match.
set -eu
: "${ENGINE_PYTHON:?Set ENGINE_PYTHON to the verified 3.2.1 interpreter}"
out=${1:?Supply a new build directory}
mkdir -p "$(dirname "$out")"
mkdir "$out"
out=$(cd "$out" && pwd)
experiment=$(cd "$(dirname "$0")" && pwd)
root=$(git -C "$experiment" rev-parse --show-toplevel)
assets=$("$ENGINE_PYTHON" -IB -c 'from pathlib import Path; import neongammon as ng; print(Path(ng.__file__).parent / "_assets")')
test -d "$assets/models"
mkdir "$out/src"
git -C "$root" archive 1b6e682 cpp python | tar -xf - -C "$out/src"
cmake -S "$out/src/cpp" -B "$out/native" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DBGBOT_BUILD_APPS=OFF -DPYBIND11_FINDPYTHON=ON \
    -DPython_EXECUTABLE="$ENGINE_PYTHON" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
for variant in baseline candidate; do
    if [ "$variant" = candidate ]; then
        patch -d "$out/src" -p1 < "$experiment/candidate.patch"
    fi
    cmake --build "$out/native" --target bgbot_cpp
    mkdir "$out/$variant"
    cp "$out/native"/bgbot_cpp*.so "$out/$variant/"
    cp -R "$out/src/python/neongammon" "$out/$variant/"
    ln -s "$assets" "$out/$variant/neongammon/_assets"
done
