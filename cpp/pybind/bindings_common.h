// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins

#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bgbot/benchmark.h"
#include "bgbot/strategy.h"

namespace bgbot::python {
namespace py = pybind11;

constexpr int kAbiVersion = 1;

template <typename T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
const std::shared_ptr<T>& require_non_null(const std::shared_ptr<T>& ptr, const char* name) {
    if (!ptr) {
        throw std::invalid_argument(std::string(name) + " must not be None");
    }
    return ptr;
}

void bind_core(py::module_& module);
void bind_eval(py::module_& module);
void bind_rollout(py::module_& module);
void bind_cube_rollout(py::module_& module);
void bind_training(py::module_& module);

// Convert a Python list of 26 ints to a Board
inline Board list_to_board(const std::vector<int>& v) {
    if (v.size() != 26) {
        throw std::runtime_error("Board must have exactly 26 elements");
    }
    Board b;
    std::copy(v.begin(), v.end(), b.begin());
    return b;
}

// Convert a Board to a Python-friendly vector
inline std::vector<int> board_to_list(const Board& b) {
    return std::vector<int>(b.begin(), b.end());
}

// C++ container for benchmark scenarios — avoids pybind11 vector copy issues
// with large datasets. Scenarios are built and stored entirely in C++.
struct ScenarioSet {
    std::vector<BenchmarkScenario> scenarios;

    void reserve(int n) { scenarios.reserve(n); }
    int size() const { return static_cast<int>(scenarios.size()); }

    void add(const std::vector<int>& start_board,
             int die1, int die2,
             const std::vector<std::vector<int>>& ranked_boards,
             const std::vector<double>& ranked_errors) {
        BenchmarkScenario s;
        s.start_board = list_to_board(start_board);
        s.die1 = die1;
        s.die2 = die2;
        s.ranked_boards.reserve(ranked_boards.size());
        for (const auto& rb : ranked_boards) {
            s.ranked_boards.push_back(list_to_board(rb));
        }
        s.ranked_errors = ranked_errors;
        scenarios.push_back(std::move(s));
    }
};

std::shared_ptr<Strategy> make_strategy_from_type(
    const std::string& strategy_type,
    const std::vector<std::string>& weight_paths,
    const std::vector<int>& hidden_sizes);

} // namespace bgbot::python
