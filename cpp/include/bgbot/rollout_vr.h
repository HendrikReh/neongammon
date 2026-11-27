// SPDX-License-Identifier: MPL-2.0
#pragma once

#include "bearoff.h"
#include "cube.h"
#include "encoding.h"
#include "neural_net.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace bgbot {

inline uint32_t rollout_board_hash(const Board& board) {
    uint32_t hash = 2166136261u;
    for (int point : board) hash = (hash ^ static_cast<uint32_t>(point)) * 16777619u;
    return hash ^ (hash >> 16);
}

#if defined(BGBOT_USE_ACCELERATE)
// Generation-stamped hash table for cross-roll endpoint deduplication in one
// 21-roll VR fan. Keeps O(1) reset between VR steps and grows if needed.
struct VrEndpointDedup {
    static constexpr uint32_t INIT_CAP = 2048;
    std::vector<int32_t> slot_val = std::vector<int32_t>(INIT_CAP, -1);
    std::vector<uint32_t> stamp = std::vector<uint32_t>(INIT_CAP, 0);
    uint32_t gen = 0;
    uint32_t mask = INIT_CAP - 1;

    void begin() {
        ++gen;
        if (gen == 0) {
            std::fill(stamp.begin(), stamp.end(), 0u);
            gen = 1;
        }
    }

    void grow(const std::vector<const Board*>& boards) {
        const size_t new_cap = slot_val.size() * 2;
        slot_val.resize(new_cap);
        stamp.assign(new_cap, 0u);
        mask = static_cast<uint32_t>(new_cap - 1);
        for (size_t i = 1; i < boards.size(); ++i) {
            uint32_t pos = rollout_board_hash(*boards[i]) & mask;
            while (stamp[pos] == gen) pos = (pos + 1) & mask;
            stamp[pos] = gen;
            slot_val[pos] = static_cast<int32_t>(i);
        }
    }

    // Returns existing unique index (1..n_unique) or assigns and returns a new one.
    int lookup_or_insert(const Board* b, std::vector<const Board*>& unique_boards) {
        if (unique_boards.size() >= mask - mask / 4) grow(unique_boards);
        uint32_t pos = rollout_board_hash(*b) & mask;
        for (;;) {
            if (stamp[pos] != gen) {
                const int idx = static_cast<int>(unique_boards.size());
                unique_boards.push_back(b);
                stamp[pos] = gen;
                slot_val[pos] = idx;
                return idx;
            }
            const int existing = slot_val[pos];
            if (*unique_boards[existing] == *b) return existing;
            pos = (pos + 1) & mask;
        }
    }
};
#endif

// All 21 rolls have the same pre-move routing board. Evaluates both cubeful
// (cubes != nullptr && n_cubes > 0, ranking by cubes[0]) and cubeless VR fans.
inline bool try_rollout_vr_rolls(
    const Strategy& base, const BearoffDB* db, const Board& board,
    const std::array<std::vector<Board>, 21>& candidates,
    const CubeInfo* cubes, int n_cubes, float cube_x, std::array<int, 21>& picks,
    std::array<std::array<float, NUM_OUTPUTS>, 21>& probs)
{
    if (check_game_over(board) != GameResult::NOT_OVER) return false;
    // The opponent cannot move home on our roll. If already home, delegate
    // the entire fan, including mixed NN/database boundary rolls.
    if (db && board[0] == 0 &&
        std::none_of(board.begin() + 1, board.begin() + 19, [](int n) { return n < 0; }))
        return false;
    const auto [network, is_pr] = base.rollout_vr_network(board);
    if (!network) return false;
    const auto& nn = *network;
    const bool cubeful = (cubes != nullptr && n_cubes > 0);

#if defined(BGBOT_USE_ACCELERATE)
    // Apple Silicon: deduplicate all non-terminal endpoints across the 21 rolls,
    // encode incrementally from the pre-roll board, and run one Accelerate SGEMM.
    const int ni = nn.n_inputs();
    const int slot_width = is_pr ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS;
    if (ni != slot_width) return false;

    thread_local VrEndpointDedup dedup;
    thread_local std::vector<const Board*> unique_boards;
    thread_local std::array<std::vector<int>, 21> roll_refs;
    thread_local std::vector<float> encoded;
    thread_local std::vector<std::array<float, NUM_OUTPUTS>> unique_out;
    thread_local std::vector<float> values;
    thread_local std::vector<std::array<float, NUM_OUTPUTS>> outputs;

    dedup.begin();
    unique_boards.clear();
    unique_boards.push_back(&board);  // index 0 = pass board

    // Negative refs encode terminal GameResult: -(static_cast<int>(gr) + 10)
    // so terminal values never collide with unique_boards indices (>= 0).
    for (size_t roll = 0; roll < candidates.size(); ++roll) {
        const auto& moves = candidates[roll];
        auto& refs = roll_refs[roll];
        refs.resize(moves.size());
        for (size_t i = 0; i < moves.size(); ++i) {
            const GameResult gr = check_game_over(moves[i]);
            if (gr != GameResult::NOT_OVER) {
                refs[i] = -(static_cast<int>(gr) + 10);
            } else {
                refs[i] = dedup.lookup_or_insert(&moves[i], unique_boards);
            }
        }
    }

    const int n_eval = static_cast<int>(unique_boards.size());
    encoded.resize(static_cast<size_t>(n_eval) * ni);
    if (is_pr) {
        auto inp0 = compute_tesauro_inputs(board);
        std::copy_n(inp0.data(), ni, encoded.data());
    } else {
        auto inp0 = compute_extended_contact_inputs(board);
        std::copy_n(inp0.data(), ni, encoded.data());
    }
    for (int u = 1; u < n_eval; ++u) {
        encode_board_from_base(*unique_boards[u], board, encoded.data(), is_pr,
                               &encoded[static_cast<size_t>(u) * ni]);
    }

    unique_out.resize(n_eval);
    nn.forward_group(encoded.data(), n_eval, unique_out.data());

    auto pass = unique_out[0];
    clamp_probs_to_board(pass, board);

    if (cubeful) {
        for (size_t roll = 0; roll < candidates.size(); ++roll) {
            const auto& moves = candidates[roll];
            if (moves.empty()) {
                picks[roll] = -1;
                probs[roll] = pass;
                continue;
            }
            const auto& refs = roll_refs[roll];
            values.resize(moves.size());
            outputs.resize(moves.size());
            for (size_t i = 0; i < moves.size(); ++i) {
                const int ref = refs[i];
                auto& p = outputs[i];
                if (ref < 0) {
                    p = terminal_probs(static_cast<GameResult>(-ref - 10));
                } else {
                    p = unique_out[ref];
                    clamp_probs_to_board(p, moves[i]);
                }
                values[i] = cl2cf(p, cubes[0], cube_x);
            }
            const int best = moves.size() == 1 ? 0 :
                pick_best_with_tiebreak(values.data(), outputs.data(), static_cast<int>(moves.size()));
            picks[roll] = best;
            probs[roll] = outputs[best];
        }
    } else {
        for (size_t roll = 0; roll < candidates.size(); ++roll) {
            const auto& moves = candidates[roll];
            if (moves.empty()) {
                picks[roll] = -1;
                probs[roll] = pass;
                continue;
            }
            const auto& refs = roll_refs[roll];
            int best = 0;
            double best_eq = -1e30;
            std::array<float, NUM_OUTPUTS> best_p{};
            for (size_t i = 0; i < moves.size(); ++i) {
                const int ref = refs[i];
                std::array<float, NUM_OUTPUTS> p;
                double eq;
                if (ref < 0) {
                    const auto gr = static_cast<GameResult>(-ref - 10);
                    p = terminal_probs(gr);
                    eq = static_cast<double>(static_cast<int>(gr));
                } else {
                    p = unique_out[ref];
                    eq = NeuralNetwork::compute_equity(p);
                }
                if (eq > best_eq) {
                    best_eq = eq;
                    best = static_cast<int>(i);
                    best_p = p;
                }
            }
            if (refs[best] >= 0) clamp_probs_to_board(best_p, moves[best]);
            picks[roll] = best;
            probs[roll] = best_p;
        }
    }
    return true;
#else
    if (!cubeful) return false;
    thread_local std::vector<float> hidden, inputs, values;
    thread_local std::vector<std::array<float, NUM_OUTPUTS>> outputs;
    hidden.resize(nn.n_hidden());
    inputs.resize(nn.n_inputs());
    const auto save = [&](const auto& encoded_base) {
        return nn.forward_save_base(encoded_base.data(), hidden.data(), inputs.data());
    };
    auto pass = is_pr ? save(compute_tesauro_inputs(board))
                      : save(compute_extended_contact_inputs(board));
    clamp_probs_to_board(pass, board);
    struct EvaluatedMove {
        const Board* board = nullptr;
        std::array<float, NUM_OUTPUTS> probabilities;
        float value;
    };
    std::array<EvaluatedMove, 256> evaluated;
    for (size_t roll = 0; roll < candidates.size(); ++roll) {
        const auto& moves = candidates[roll];
        if (moves.empty()) {
            picks[roll] = -1;
            probs[roll] = pass;
            continue;
        }
        values.resize(moves.size());
        outputs.resize(moves.size());
        for (size_t i = 0; i < moves.size(); ++i) {
            auto& entry = evaluated[rollout_board_hash(moves[i]) & (evaluated.size() - 1)];
            if (entry.board && *entry.board == moves[i]) {
                outputs[i] = entry.probabilities;
                values[i] = entry.value;
                continue;
            }
            const auto result = check_game_over(moves[i]);
            auto& p = outputs[i];
            p = result == GameResult::NOT_OVER
                ? nn.forward_board_from_base(moves[i], board, hidden.data(), inputs.data())
                : terminal_probs(result);
            clamp_probs_to_board(p, moves[i]);
            values[i] = cl2cf(p, cubes[0], cube_x);
            entry = {&moves[i], p, values[i]};
        }
        const int best = moves.size() == 1 ? 0 :
            pick_best_with_tiebreak(values.data(), outputs.data(), static_cast<int>(moves.size()));
        picks[roll] = best;
        probs[roll] = outputs[best];
    }
    return true;
#endif
}

} // namespace bgbot
