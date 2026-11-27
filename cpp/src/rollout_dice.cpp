// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// Stratified quasi-random dice generation and thread-count resolution for
// RolloutStrategy.
#include "rollout_internal.h"
#include <random>

namespace bgbot {

// ======================== Static Data ========================

const std::array<RolloutStrategy::DiceRoll, 21> RolloutStrategy::ALL_ROLLS = {{
    {1,1,1}, {2,2,1}, {3,3,1}, {4,4,1}, {5,5,1}, {6,6,1},
    {1,2,2}, {1,3,2}, {1,4,2}, {1,5,2}, {1,6,2},
    {2,3,2}, {2,4,2}, {2,5,2}, {2,6,2},
    {3,4,2}, {3,5,2}, {3,6,2},
    {4,5,2}, {4,6,2},
    {5,6,2}
}};

// ======================== Stratified Dice (GNUbg-style) ========================

void RolloutStrategy::PerArray::init(uint32_t s) {
    if (seed == static_cast<int>(s)) return;

    // Use a simple PRNG to generate permutations (equivalent to GNUbg's ISAAC)
    std::mt19937 rng(s);

    for (int i = 0; i < 6; ++i) {
        // j starts at i (no need for permutations below the diagonal)
        for (int j = i; j < 128; ++j) {
            // Initialize identity permutation
            for (uint8_t k = 0; k < 36; ++k) {
                perm[i][j][k] = k;
            }
            // Fisher-Yates shuffle
            for (int k = 0; k < 35; ++k) {
                int r = rng() % (36 - k);
                std::swap(perm[i][j][k], perm[i][j][k + r]);
            }
        }
    }
    seed = static_cast<int>(s);
}

void RolloutStrategy::generate_stratified_dice(
    int n_trials, int max_moves, uint32_t seed,
    std::vector<std::vector<std::pair<int,int>>>& dice_out)
{
    dice_out.resize(n_trials);

    // Initialize hierarchical permutation array
    PerArray pa;
    pa.seed = -1;
    pa.init(seed);

    for (int t = 0; t < n_trials; ++t) {
        dice_out[t].resize(max_moves);

        // Per-trial RNG for rolls beyond the stratified range
        std::mt19937 trial_rng(seed + static_cast<uint32_t>(t) * 1000003u + 7u);
        std::uniform_int_distribution<int> die(1, 6);

        for (int m = 0; m < max_moves; ++m) {
            if (m < 128) {
                // Quasi-random: compose hierarchical permutations.
                // For 36^N trials, the first N rolls are jointly stratified.
                // Level i uses turn=m and composes from the previous level's output.
                unsigned int j = 0;
                unsigned int k = 1;  // 36^i
                int max_level = std::min(5, m);
                for (int i = 0; i <= max_level; ++i) {
                    j = pa.perm[i][m][((t / k) + j) % 36];
                    k *= 36;
                }
                dice_out[t][m] = {static_cast<int>(j / 6) + 1,
                                  static_cast<int>(j % 6) + 1};
            } else {
                // Beyond 128 turns: truly random
                dice_out[t][m] = {die(trial_rng), die(trial_rng)};
            }
        }
    }
}

int RolloutStrategy::rollout_thread_count(int n_trials) const
{
    int n_threads = config_.n_threads;
    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;

        // Truncated rollouts with N-ply move selection share many early-tree
        // subproblems. Keeping them on one worker preserves thread-local
        // MultiPly caches and is often faster than splitting the trials.
        // Allow opt-in via config_.parallelize_trials; default stays serial
        // to preserve cache locality and historical behavior.
        if (!config_.parallelize_trials && config_.truncation_depth > 0 && config_.decision_ply > 1) {
            n_threads = 1;
        }
    }

    n_threads = std::min(n_threads, n_trials);
    return std::max(1, n_threads);
}

} // namespace bgbot
