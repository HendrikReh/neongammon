#!/bin/sh
# SPDX-License-Identifier: MPL-2.0
# Build disposable probes with a common compiler, SDK, interpreter and objects.
set -eu
: "${ENGINE_PYTHON:?Set ENGINE_PYTHON to the official 3.2.1 wheel interpreter}"
out=${1:?Supply a fresh output directory, e.g. build/ng63}
mkdir -p "$(dirname "$out")"
mkdir "$out"
out=$(cd "$out" && pwd)
experiment=$(cd "$(dirname "$0")" && pwd)
root=$(git -C "$experiment" rev-parse --show-toplevel)
mkdir "$out/src"
git -C "$root" archive v3.2.1 cpp > "$out/cpp.tar"
tar -xf "$out/cpp.tar" -C "$out/src"
cmake -S "$out/src/cpp" -B "$out/native" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DBGBOT_BUILD_APPS=OFF -DPYBIND11_FINDPYTHON=ON \
    -DPython_EXECUTABLE="$ENGINE_PYTHON" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"

save_extension() {
    cmake --build "$out/native" --target bgbot_cpp
    mkdir "$out/$1"
    cp "$out/native"/bgbot_cpp*.so "$out/$1/"
}

src="$out/src/cpp/src/cube_eval.cpp"
restore_source() {
    git -C "$root" show v3.2.1:cpp/src/cube_eval.cpp > "$src"
}

save_extension pristine
sed -i '' 's/const bool share_survivors = false;/const bool share_survivors = true;/' "$src"
save_extension subset
restore_source
patch -d "$out/src" -p1 < "$experiment/candidate-probe.patch"
save_extension ab
sed -i '' '1i\
#define NG63_TRACE 1
' "$src"
save_extension trace
restore_source
patch -d "$out/src" -p1 < "$experiment/phase-timers.patch"
save_extension phases
