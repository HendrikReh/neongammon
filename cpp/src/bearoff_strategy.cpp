// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// BearoffStrategy: Strategy decorator delegating bearoff positions to BearoffDB
// and sharing a single split-and-merge template across all four batch overrides.
#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/cube.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include <array>
#include <vector>

namespace bgbot {

namespace {

enum class BatchEvalMode {
    Equity,
    BatchEquity,
    EquityProbs,
    BestProb,
};

bool has_any_bearoff(const BearoffDB& db, const std::vector<Board>& candidates) {
    for (const auto& c : candidates) {
        if (db.is_bearoff(c)) return true;
    }
    return false;
}

int pick_all_bearoff(const BearoffDB& db, const std::vector<Board>& candidates) {
    int best_idx = 0;
    double best_eq = -1e30;
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        const GameResult r = check_game_over(candidates[i]);
        const auto probs = (r != GameResult::NOT_OVER)
            ? terminal_probs(r)
            : db.lookup_probs(candidates[i], /*post_move=*/true);
        const double eq = NeuralNetwork::compute_equity(probs);
        if (eq > best_eq) {
            best_eq = eq;
            best_idx = i;
        }
    }
    return best_idx;
}

template <BatchEvalMode Mode>
int eval_candidates_impl(
    const Strategy& base,
    const BearoffDB* db,
    bool wraps_same_db,
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities,
    std::array<float, NUM_OUTPUTS>* probs_out)
{
    auto forward_to_base = [&](const std::vector<Board>& boards,
                               double* eq_ptr,
                               std::array<float, NUM_OUTPUTS>* p_ptr) -> int {
        if constexpr (Mode == BatchEvalMode::Equity) {
            return base.evaluate_candidates_equity(boards, pre_move_board, eq_ptr);
        } else if constexpr (Mode == BatchEvalMode::BatchEquity) {
            return base.batch_evaluate_candidates_equity(boards, pre_move_board, eq_ptr);
        } else if constexpr (Mode == BatchEvalMode::EquityProbs) {
            return base.batch_evaluate_candidates_equity_probs(boards, pre_move_board, eq_ptr, p_ptr);
        } else {
            return base.batch_evaluate_candidates_best_prob(boards, pre_move_board, eq_ptr, p_ptr);
        }
    };

    if (!db || !db->is_loaded() || wraps_same_db || candidates.empty()) {
        return forward_to_base(candidates, equities, probs_out);
    }

    const int n = static_cast<int>(candidates.size());
    bool any_bearoff = false;
    bool all_handled = true;
    for (int i = 0; i < n; ++i) {
        if (db->is_bearoff(candidates[i])) {
            any_bearoff = true;
        } else if (check_game_over(candidates[i]) == GameResult::NOT_OVER) {
            all_handled = false;
        }
    }
    if (!any_bearoff) {
        return forward_to_base(candidates, equities, probs_out);
    }

    // Fast path: every candidate is terminal or in the bearoff DB.
    if (all_handled) {
        int best_idx = 0;
        double best_eq = -1e30;
        std::array<float, NUM_OUTPUTS> best_probs{};
        for (int i = 0; i < n; ++i) {
            const GameResult r = check_game_over(candidates[i]);
            const auto probs = (r != GameResult::NOT_OVER)
                ? terminal_probs(r)
                : db->lookup_probs(candidates[i], /*post_move=*/true);
            const double eq = compute_equity(probs);
            if (equities) equities[i] = eq;
            if constexpr (Mode == BatchEvalMode::EquityProbs) {
                if (probs_out) probs_out[i] = probs;
            }
            if (eq > best_eq) {
                best_eq = eq;
                best_idx = i;
                if constexpr (Mode == BatchEvalMode::BestProb) {
                    best_probs = probs;
                }
            }
        }
        if constexpr (Mode == BatchEvalMode::BestProb) {
            if (probs_out) *probs_out = best_probs;
        }
        return best_idx;
    }

    // Mixed bearoff + non-bearoff: use local buffers for re-entrancy safety.
    std::vector<int> nn_indices;
    std::vector<Board> nn_boards;
    nn_indices.reserve(n);
    nn_boards.reserve(n);

    if constexpr (Mode == BatchEvalMode::BestProb) {
        bool has_best = false;
        int best_idx = 0;
        double best_eq = -1e30;
        std::array<float, NUM_OUTPUTS> best_probs{};

        for (int i = 0; i < n; ++i) {
            const GameResult r = check_game_over(candidates[i]);
            if (r != GameResult::NOT_OVER || db->is_bearoff(candidates[i])) {
                const auto probs = (r != GameResult::NOT_OVER)
                    ? terminal_probs(r)
                    : db->lookup_probs(candidates[i], /*post_move=*/true);
                const double eq = compute_equity(probs);
                if (equities) equities[i] = eq;
                if (!has_best || eq > best_eq) {
                    has_best = true;
                    best_eq = eq;
                    best_idx = i;
                    best_probs = probs;
                }
            } else {
                nn_indices.push_back(i);
                nn_boards.push_back(candidates[i]);
            }
        }

        std::vector<double> nn_eq(nn_boards.size());
        std::array<float, NUM_OUTPUTS> nn_best_probs{};
        const int nn_best = forward_to_base(
            nn_boards, nn_eq.data(), probs_out ? &nn_best_probs : nullptr);
        if (equities) {
            for (size_t j = 0; j < nn_indices.size(); ++j) {
                equities[nn_indices[j]] = nn_eq[j];
            }
        }
        if (nn_best >= 0 && nn_best < static_cast<int>(nn_indices.size())) {
            const int orig_idx = nn_indices[nn_best];
            const double nn_best_eq = nn_eq[nn_best];
            if (!has_best || nn_best_eq > best_eq ||
                (nn_best_eq == best_eq && orig_idx < best_idx)) {
                best_eq = nn_best_eq;
                best_idx = orig_idx;
                best_probs = nn_best_probs;
            }
        }
        if (probs_out) *probs_out = best_probs;
        return best_idx;
    } else {
        std::vector<double> local_eq;
        double* eq_dst = equities;
        if (!eq_dst) {
            local_eq.resize(n);
            eq_dst = local_eq.data();
        }

        for (int i = 0; i < n; ++i) {
            const GameResult r = check_game_over(candidates[i]);
            if (r != GameResult::NOT_OVER || db->is_bearoff(candidates[i])) {
                const auto probs = (r != GameResult::NOT_OVER)
                    ? terminal_probs(r)
                    : db->lookup_probs(candidates[i], /*post_move=*/true);
                eq_dst[i] = compute_equity(probs);
                if constexpr (Mode == BatchEvalMode::EquityProbs) {
                    if (probs_out) probs_out[i] = probs;
                }
            } else {
                nn_indices.push_back(i);
                nn_boards.push_back(candidates[i]);
            }
        }

        std::vector<double> nn_eq(nn_boards.size());
        std::vector<std::array<float, NUM_OUTPUTS>> nn_probs;
        std::array<float, NUM_OUTPUTS>* nn_probs_ptr = nullptr;
        if constexpr (Mode == BatchEvalMode::EquityProbs) {
            if (probs_out) {
                nn_probs.resize(nn_boards.size());
                nn_probs_ptr = nn_probs.data();
            }
        }
        forward_to_base(nn_boards, nn_eq.data(), nn_probs_ptr);

        for (size_t j = 0; j < nn_indices.size(); ++j) {
            eq_dst[nn_indices[j]] = nn_eq[j];
            if constexpr (Mode == BatchEvalMode::EquityProbs) {
                if (probs_out) probs_out[nn_indices[j]] = nn_probs[j];
            }
        }

        int best_idx = 0;
        double best_eq = -1e30;
        for (int i = 0; i < n; ++i) {
            if (eq_dst[i] > best_eq) {
                best_eq = eq_dst[i];
                best_idx = i;
            }
        }
        return best_idx;
    }
}

} // namespace

// --- BearoffStrategy ---

BearoffStrategy::BearoffStrategy(std::shared_ptr<Strategy> base, const BearoffDB* db)
    : base_(std::move(base)), db_(db)
{
    if (!base_) {
        throw std::invalid_argument("base strategy must not be null");
    }
    const auto inner = base_->bearoff_layer();
    wraps_same_db_ = inner.base && inner.db == db_;
}

uint64_t BearoffStrategy::eval_identity() const {
    if (wraps_same_db_) return base_->eval_identity();
    const uint64_t base_id = base_->eval_identity();
    const uint64_t db_id = db_ ? db_->identity() : 0;
    if (db_id == 0) return base_id;
    uint64_t h = db_id * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 29;
    return base_id ^ (h | 1ULL << 63);
}

double BearoffStrategy::evaluate(const Board& board, bool pre_move_is_race) const {
    if (db_ && db_->is_loaded() && db_->is_bearoff(board)) {
        auto probs = db_->lookup_probs(board, /*post_move=*/true);
        return NeuralNetwork::compute_equity(probs);
    }
    return base_->evaluate(board, pre_move_is_race);
}

std::array<float, NUM_OUTPUTS> BearoffStrategy::evaluate_probs(
    const Board& board, bool pre_move_is_race) const
{
    if (db_ && db_->is_loaded() && db_->is_bearoff(board)) {
        return db_->lookup_probs(board, /*post_move=*/true);
    }
    return base_->evaluate_probs(board, pre_move_is_race);
}

std::array<float, NUM_OUTPUTS> BearoffStrategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    if (db_ && db_->is_loaded() && db_->is_bearoff(board)) {
        return db_->lookup_probs(board, /*post_move=*/true);
    }
    return base_->evaluate_probs(board, pre_move_board);
}

std::array<float, NUM_OUTPUTS> BearoffStrategy::evaluate_pinned_probs(
    const Board& board, bool race, int pinned) const
{
    if (db_ && db_->is_loaded() && db_->is_bearoff(board)) {
        return db_->lookup_probs(board, /*post_move=*/true);
    }
    return base_->evaluate_pinned_probs(board, race, pinned);
}

int BearoffStrategy::best_move_index(const std::vector<Board>& candidates,
                                      bool pre_move_is_race) const
{
    if (!db_ || !db_->is_loaded() || wraps_same_db_ || candidates.empty()) {
        return base_->best_move_index(candidates, pre_move_is_race);
    }
    bool all_bearoff = true;
    for (const auto& c : candidates) {
        if (!db_->is_bearoff(c) && check_game_over(c) == GameResult::NOT_OVER) {
            all_bearoff = false;
            break;
        }
    }
    if (all_bearoff) {
        return pick_all_bearoff(*db_, candidates);
    }
    return base_->best_move_index(candidates, pre_move_is_race);
}

int BearoffStrategy::best_move_index(const std::vector<Board>& candidates,
                                      const Board& pre_move_board) const
{
    if (!db_ || !db_->is_loaded() || wraps_same_db_ || candidates.empty()) {
        return base_->best_move_index(candidates, pre_move_board);
    }
    bool all_bearoff = true;
    for (const auto& c : candidates) {
        if (!db_->is_bearoff(c) && check_game_over(c) == GameResult::NOT_OVER) {
            all_bearoff = false;
            break;
        }
    }
    if (all_bearoff) {
        return pick_all_bearoff(*db_, candidates);
    }
    return base_->best_move_index(candidates, pre_move_board);
}

// ----- Cube-aware overrides -----

int BearoffStrategy::best_move_index_cubeful(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    const CubeInfo& ci,
    float cube_x) const
{
    if (candidates.empty()) return 0;
    if (!db_ || !db_->is_loaded() || wraps_same_db_ || !has_any_bearoff(*db_, candidates)) {
        return base_->best_move_index_cubeful(candidates, pre_move_board, ci, cube_x);
    }
    return Strategy::best_move_index_cubeful(candidates, pre_move_board, ci, cube_x);
}

void BearoffStrategy::best_move_index_cubeful_multi(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    const CubeInfo* cubes,
    int n_cubes,
    float cube_x,
    int* out_indices) const
{
    if (candidates.empty() || n_cubes <= 0) return;
    if (!db_ || !db_->is_loaded() || wraps_same_db_ || !has_any_bearoff(*db_, candidates)) {
        base_->best_move_index_cubeful_multi(candidates, pre_move_board,
                                             cubes, n_cubes, cube_x, out_indices);
        return;
    }
    Strategy::best_move_index_cubeful_multi(candidates, pre_move_board,
                                            cubes, n_cubes, cube_x, out_indices);
}

// ----- Batch evaluation overrides -----

int BearoffStrategy::evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    return eval_candidates_impl<BatchEvalMode::Equity>(
        *base_, db_, wraps_same_db_, candidates, pre_move_board, equities, nullptr);
}

int BearoffStrategy::batch_evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    return eval_candidates_impl<BatchEvalMode::BatchEquity>(
        *base_, db_, wraps_same_db_, candidates, pre_move_board, equities, nullptr);
}

int BearoffStrategy::batch_evaluate_candidates_equity_probs(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities,
    std::array<float, NUM_OUTPUTS>* probs_out) const
{
    return eval_candidates_impl<BatchEvalMode::EquityProbs>(
        *base_, db_, wraps_same_db_, candidates, pre_move_board, equities, probs_out);
}

int BearoffStrategy::batch_evaluate_candidates_best_prob(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities,
    std::array<float, NUM_OUTPUTS>* best_probs_out) const
{
    return eval_candidates_impl<BatchEvalMode::BestProb>(
        *base_, db_, wraps_same_db_, candidates, pre_move_board, equities, best_probs_out);
}

} // namespace bgbot
