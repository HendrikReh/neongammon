// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// RolloutStrategy construction, configuration, cancellation, VR evaluation,
// and Move0/Move1 cache prefill.
#include "rollout_internal.h"

namespace bgbot {

using namespace rollout_detail;

// ======================== Cache Management ========================

RolloutStrategy::Move1Cache::Move1Cache(const RolloutConfig& config) {
    for (int i = 0; i < Move0Cache::N_ROLLS; ++i) {
        state[i].store(0, std::memory_order_relaxed);
        cd_state[i].store(0, std::memory_order_relaxed);
    }
    // Allocate once per rollout. Short or low-trial runs have limited reuse.
    if (config.enable_vr && config.n_trials >= 288 &&
        (config.truncation_depth == 0 || config.truncation_depth > 2)) {
        vr_cache = std::make_unique<VarianceReductionCache>();
    }
}

RolloutStrategy::Move1Cache::~Move1Cache() = default;

void RolloutStrategy::clear_internal_caches() const {
    std::lock_guard<std::recursive_mutex> lock(rollout_mutex_);
    if (checker_strat_) checker_strat_->clear_cache();
    if (shared_pos_cache_) {
        shared_pos_cache_->clear();
    }
    clear_cubeful_eval_cache();
}

void RolloutStrategy::set_bearoff_db(const BearoffDB* db) {
    std::lock_guard<std::recursive_mutex> lock(rollout_mutex_);
    bearoff_db_ = db;
    if (db && db->is_loaded()) {
        base_bearoff_ = std::make_shared<BearoffStrategy>(base_, db);
    } else {
        base_bearoff_ = nullptr;
    }
    if (checker_strat_) checker_strat_->set_bearoff_db(db);
    if (checker_late_strat_) checker_late_strat_->set_bearoff_db(db);
    if (truncation_strat_) truncation_strat_->set_bearoff_db(db);
    if (cube_inner_rollout_) cube_inner_rollout_->set_bearoff_db(db);
}

void RolloutStrategy::set_move_filter(std::shared_ptr<Strategy> filter) {
    std::lock_guard<std::recursive_mutex> lock(rollout_mutex_);
    move_filter_ = filter;
    if (checker_strat_) checker_strat_->set_move_prefilter(filter);
    if (checker_late_strat_) checker_late_strat_->set_move_prefilter(filter);
    if (truncation_strat_) truncation_strat_->set_move_prefilter(filter);
}

// ======================== Cancellation ========================

void RolloutStrategy::cancel() {
    owned_cancel_flag_.store(true, std::memory_order_relaxed);
}

void RolloutStrategy::reset_cancel() {
    owned_cancel_flag_.store(false, std::memory_order_relaxed);
}

bool RolloutStrategy::is_cancelled() const {
    return owned_cancel_flag_.load(std::memory_order_relaxed) ||
           (config_.cancel_flag &&
            config_.cancel_flag->load(std::memory_order_relaxed));
}

// ======================== Constructor ========================

namespace {

std::shared_ptr<RolloutStrategy> make_inner_rollout_strategy(
    const TrialEvalConfig& cfg,
    const std::shared_ptr<Strategy>& base,
    const std::shared_ptr<Strategy>& filter_base)
{
    if (!cfg.is_rollout()) return nullptr;
    RolloutConfig inner;
    inner.n_trials = cfg.rollout_trials;
    inner.truncation_depth = cfg.rollout_depth;
    inner.decision_ply = cfg.rollout_ply;
    inner.n_threads = 1;
    inner.enable_vr = true;
    inner.seed = 42;
    if (filter_base) {
        return std::make_shared<RolloutStrategy>(base, filter_base, inner);
    }
    return std::make_shared<RolloutStrategy>(base, inner);
}

std::shared_ptr<Strategy> build_eval_strategy(
    const TrialEvalConfig& eval,
    int effective_ply,
    const std::shared_ptr<Strategy>& base,
    const std::shared_ptr<Strategy>& filter_base,
    const MoveFilter& internal_filter)
{
    if (eval.is_rollout()) {
        return make_inner_rollout_strategy(eval, base, filter_base);
    }

    int ply = eval.is_set() ? eval.ply : effective_ply;
    if (ply > 1) {
        if (filter_base) {
            return std::make_shared<MultiPlyStrategy>(
                base, filter_base, ply, internal_filter);
        }
        return std::make_shared<MultiPlyStrategy>(
            base, ply, internal_filter);
    }
    return base;
}

TrialEvalConfig resolve_cube_eval_config(
    const TrialEvalConfig& cfg, int default_ply)
{
    if (cfg.is_set()) return cfg;
    TrialEvalConfig resolved;
    resolved.ply = default_ply;
    return resolved;
}

void build_rollout_strategies(
    const std::shared_ptr<Strategy>& base,
    const std::shared_ptr<Strategy>& filter_base,
    const RolloutConfig& config,
    std::shared_ptr<Strategy>& checker_strat,
    std::shared_ptr<Strategy>& checker_late_strat,
    TrialEvalConfig& cube_eval_config,
    std::shared_ptr<RolloutStrategy>& cube_inner_rollout,
    std::shared_ptr<Strategy>& truncation_strat)
{
    MoveFilter internal_filter = {2, 0.03f};

    int checker_ply = config.decision_ply;
    int checker_late_ply_eff = (config.late_ply >= 1) ? config.late_ply : config.decision_ply;

    checker_strat = build_eval_strategy(
        config.checker, checker_ply, base, filter_base, internal_filter);

    if (!config.checker_late.is_set() && !config.checker.is_set()
        && checker_late_ply_eff == checker_ply) {
        checker_late_strat = checker_strat;
    } else {
        checker_late_strat = build_eval_strategy(
            config.checker_late.is_set() ? config.checker_late : config.checker,
            checker_late_ply_eff, base, filter_base, internal_filter);
    }

    cube_eval_config = resolve_cube_eval_config(config.cube, checker_ply);
    cube_inner_rollout = make_inner_rollout_strategy(cube_eval_config, base, filter_base);

    truncation_strat = base;
}

} // namespace

void RolloutStrategy::init_strategy(std::shared_ptr<Strategy> filter_base) {
    if (!config_.cancel_flag) {
        config_.cancel_flag = &owned_cancel_flag_;
    }
    build_rollout_strategies(base_, filter_base, config_,
                             checker_strat_, checker_late_strat_,
                             cube_eval_config_, cube_inner_rollout_,
                             truncation_strat_);

    truncation_ply_ = (config_.truncation_ply >= 1) ? config_.truncation_ply : config_.decision_ply;
    vr_enabled_ = config_.enable_vr;

    if (config_.n_trials > 0 && cached_max_moves_ > 0) {
        generate_stratified_dice(
            config_.n_trials, cached_max_moves_, config_.seed, cached_dice_);
    }

    if (rollout_thread_count(config_.n_trials) > 1) {
        shared_pos_cache_ = std::make_unique<SharedPosCache>();
    }
}

RolloutStrategy::RolloutStrategy(std::shared_ptr<Strategy> base, RolloutConfig config)
    : base_(std::move(base))
    , config_(config)
    , cached_max_moves_((config.truncation_depth > 0)
                       ? config.truncation_depth + 10
                       : 200)
{
    init_strategy(nullptr);
}

RolloutStrategy::RolloutStrategy(std::shared_ptr<Strategy> base,
                                 std::shared_ptr<Strategy> filter_base,
                                 RolloutConfig config)
    : base_(std::move(base))
    , config_(config)
    , cached_max_moves_((config.truncation_depth > 0)
                       ? config.truncation_depth + 10
                       : 200)
{
    init_strategy(std::move(filter_base));
}

// ======================== VR Helper ========================

std::array<float, NUM_OUTPUTS> RolloutStrategy::best_move_probs(
    const Board& board, int d1, int d2,
    const Strategy& strat) const
{
    TrialMoveScratch<std::vector<Board>> scratch;
    auto& candidates = scratch.get();
    possible_boards(board, d1, d2, candidates);
    return best_move_probs_for_candidates(board, candidates, strat);
}

std::array<float, NUM_OUTPUTS> RolloutStrategy::best_move_probs_for_candidates(
    const Board& board, const std::vector<Board>& candidates,
    const Strategy& strat, int* best_index,
    const CubeInfo* sel_cubes, int n_sel_cubes, float sel_cube_x) const
{
    if (sel_cubes && n_sel_cubes > 0 && candidates.size() > 1) {
        const bool is_base = (&strat == base_.get());
        const bool is_base_bearoff = (base_bearoff_ && &strat == base_bearoff_.get());
        if (is_base || is_base_bearoff) {
            const int n = static_cast<int>(candidates.size());
            thread_local std::vector<std::array<float, NUM_OUTPUTS>> probs_per;
            thread_local std::vector<float> cf_vals;
            probs_per.resize(n);
            cf_vals.resize(n);
            strat.batch_evaluate_candidates_equity_probs(
                candidates, board, nullptr, probs_per.data());
            for (int i = 0; i < n; ++i) {
                clamp_probs_to_board(probs_per[i], candidates[i]);
                cf_vals[i] = cl2cf(probs_per[i], sel_cubes[0], sel_cube_x);
            }
            int idx = pick_best_with_tiebreak(cf_vals.data(), probs_per.data(), n);
            if (best_index) *best_index = idx;
            return probs_per[idx];
        }
        std::array<int, 4> picks{0, 0, 0, 0};
        strat.best_move_index_cubeful_multi(
            candidates, board, sel_cubes,
            std::min(n_sel_cubes, 4), sel_cube_x, picks.data());
        int idx = picks[0];
        if (idx < 0 || idx >= static_cast<int>(candidates.size())) idx = 0;
        if (best_index) *best_index = idx;
        GameResult tr = check_game_over(candidates[idx]);
        if (tr != GameResult::NOT_OVER) return terminal_probs(tr);
        auto probs = strat.evaluate_probs(candidates[idx], board);
        clamp_probs_to_board(probs, candidates[idx]);
        return probs;
    }

    if (candidates.empty()) {
        if (best_index) *best_index = -1;
        auto probs = strat.evaluate_probs(board, board);
        clamp_probs_to_board(probs, board);
        return probs;
    }

    if (candidates.size() == 1) {
        GameResult r = check_game_over(candidates[0]);
        if (best_index) *best_index = 0;
        if (r != GameResult::NOT_OVER) {
            return terminal_probs(r);
        }
        auto probs = strat.evaluate_probs(candidates[0], board);
        clamp_probs_to_board(probs, candidates[0]);
        return probs;
    }

    const bool is_base = (&strat == base_.get());
    const bool is_base_bearoff = (base_bearoff_ && &strat == base_bearoff_.get());
    if (is_base || is_base_bearoff) {
        if (is_base_bearoff && bearoff_db_) {
            bool all_bearoff = true, any_bearoff = false;
            for (const auto& c : candidates) {
                if (bearoff_db_->is_bearoff(c)) any_bearoff = true;
                else all_bearoff = false;
            }
            if (any_bearoff) {
                thread_local std::vector<double> nn_eq;
                if (!all_bearoff) {
                    nn_eq.resize(candidates.size());
                    base_->batch_evaluate_candidates_equity(
                        candidates, board, nn_eq.data());
                }
                double best_eq = -1e30;
                int best_db_idx = 0;
                for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
                    double eq;
                    GameResult gr = check_game_over(candidates[i]);
                    if (gr != GameResult::NOT_OVER) {
                        eq = static_cast<double>(static_cast<int>(gr));
                    } else if (bearoff_db_->is_bearoff(candidates[i])) {
                        auto p = bearoff_db_->lookup_probs(
                            candidates[i], /*post_move=*/true);
                        clamp_probs_to_board(p, candidates[i]);
                        eq = compute_equity(p);
                    } else {
                        eq = nn_eq[i];
                    }
                    if (eq > best_eq) { best_eq = eq; best_db_idx = i; }
                }
                std::array<float, NUM_OUTPUTS> best_probs{};
                GameResult gr = check_game_over(candidates[best_db_idx]);
                if (gr != GameResult::NOT_OVER) {
                    best_probs = terminal_probs(gr);
                } else if (bearoff_db_->is_bearoff(candidates[best_db_idx])) {
                    best_probs = bearoff_db_->lookup_probs(
                        candidates[best_db_idx], /*post_move=*/true);
                    clamp_probs_to_board(best_probs, candidates[best_db_idx]);
                } else {
                    best_probs = base_->evaluate_probs(
                        candidates[best_db_idx], board);
                    clamp_probs_to_board(best_probs, candidates[best_db_idx]);
                }
                if (best_index) *best_index = best_db_idx;
                return best_probs;
            }
        }
        std::array<float, NUM_OUTPUTS> best_probs{};
        int idx = base_->batch_evaluate_candidates_best_prob(
            candidates, board, nullptr, &best_probs);
        if (idx >= 0 && idx < static_cast<int>(candidates.size())) {
            clamp_probs_to_board(best_probs, candidates[idx]);
        }
        if (best_index) *best_index = idx;
        return best_probs;
    }

    constexpr double VR_FILTER_THRESHOLD = 0.12;
    constexpr int VR_FILTER_MAX = 8;

    thread_local std::vector<double> eq_buf;
    eq_buf.resize(candidates.size());
    base_->batch_evaluate_candidates_equity(candidates, board, eq_buf.data());

    double best_1ply = -1e30;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (eq_buf[i] > best_1ply) best_1ply = eq_buf[i];
    }

    thread_local std::vector<std::pair<double, int>> ranked;
    ranked.clear();
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (eq_buf[i] >= best_1ply - VR_FILTER_THRESHOLD) {
            ranked.push_back({eq_buf[i], static_cast<int>(i)});
        }
    }

    if (static_cast<int>(ranked.size()) > VR_FILTER_MAX) {
        std::partial_sort(ranked.begin(), ranked.begin() + VR_FILTER_MAX,
                          ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        ranked.resize(VR_FILTER_MAX);
    }

    double best_eq = -1e30;
    std::array<float, NUM_OUTPUTS> best_probs = {};
    int best_original_idx = ranked.empty() ? 0 : ranked[0].second;

    for (const auto& [eq0, idx] : ranked) {
        GameResult r = check_game_over(candidates[idx]);
        std::array<float, NUM_OUTPUTS> probs;
        double eq;
        if (r != GameResult::NOT_OVER) {
            probs = terminal_probs(r);
            eq = static_cast<double>(static_cast<int>(r));
        } else {
            probs = strat.evaluate_probs(candidates[idx], board);
            clamp_probs_to_board(probs, candidates[idx]);
            eq = compute_equity(probs);
        }
        if (eq > best_eq) {
            best_eq = eq;
            best_probs = probs;
            best_original_idx = idx;
        }
    }

    if (best_index) *best_index = best_original_idx;
    return best_probs;
}

void RolloutStrategy::evaluate_vr_rolls(
    const Board& board, const std::array<std::vector<Board>, 21>& candidates,
    const CubeInfo* cubes, int cube_count, float cube_x, std::array<int, 21>& picks,
    std::array<std::array<float, NUM_OUTPUTS>, 21>& probabilities) const
{
    if (try_rollout_vr_rolls(*base_, bearoff_db_, board, candidates, cubes, cube_count,
                             cube_x, picks, probabilities)) return;
    const Strategy& evaluator = base_bearoff_ ? *base_bearoff_ : *base_;
    for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
        probabilities[i] = best_move_probs_for_candidates(
            board, candidates[i], evaluator, &picks[i], cubes, cube_count, cube_x);
    }
}

// ======================== Move0 / Move1 Prefill ========================

Board RolloutStrategy::select_move0_for_roll(
    const Board& start_board,
    int roll_idx,
    const CubeInfo* select_cubes,
    int n_select_cubes,
    float cube_x_stamp) const
{
    const auto& current_strat = *checker_strat_;
    const bool using_base = (&current_strat == base_.get());
    const bool cube_aware = (select_cubes != nullptr && n_select_cubes > 0);

    TrialMoveScratch<std::vector<Board>> scratch;
    auto& candidates = scratch.get();
    candidates.clear();
    possible_boards_unsorted(start_board,
                             ALL_ROLLS[roll_idx].d1,
                             ALL_ROLLS[roll_idx].d2,
                             candidates);

    if (candidates.empty()) {
        return start_board;
    }
    if (candidates.size() == 1) {
        return candidates[0];
    }
    if (cube_aware) {
        std::array<int, 4> picks{0, 0, 0, 0};
        current_strat.best_move_index_cubeful_multi(
            candidates, start_board,
            select_cubes, n_select_cubes,
            cube_x_stamp, picks.data());
        return candidates[picks[0]];
    }
    if (using_base) {
        return candidates[base_->best_move_index(candidates, start_board)];
    }
    return candidates[current_strat.rollout_cubeless_move(candidates, start_board)];
}

void RolloutStrategy::prefill_move0_cache(
    const Board& start_board, Move0Cache& cache, int n_threads,
    SharedPosCache* shared, const CubeInfo* select_cubes, int n_select_cubes) const
{
    const bool cube_aware = (select_cubes != nullptr && n_select_cubes > 0);
    float cube_x_stamp = 0.0f;
    if (cube_aware) {
        std::array<float, NUM_OUTPUTS> dummy{};
        auto [pp, op] = pip_counts(start_board);
        cube_x_stamp = cube_efficiency(dummy, is_race(start_board), pp, op);
    }

    auto compute_roll = [&](int roll_idx) {
        ScopedSharedCache shared_guard(shared, /*apply_null=*/false);
        cache.chosen[roll_idx] = select_move0_for_roll(
            start_board, roll_idx, select_cubes, n_select_cubes, cube_x_stamp);
        cache.state[roll_idx].store(2, std::memory_order_release);
    };

    const int workers = std::min<int>(Move0Cache::N_ROLLS, std::max(1, n_threads));
    if (workers <= 1) {
        for (int i = 0; i < Move0Cache::N_ROLLS; ++i) {
            if (is_cancelled()) throw RolloutCancelled();
            compute_roll(i);
        }
    } else {
        multipy_parallel_for(Move0Cache::N_ROLLS, workers, compute_roll);
    }
}

void RolloutStrategy::populate_move1_cache_entry(
    const Move0Cache& move0_cache, int first_roll_idx, Move1Cache::Entry& entry,
    const CubeInfo* select_cubes, int n_select_cubes) const
{
    const Board& move0_chosen = move0_cache.chosen[first_roll_idx];
    const Board move1_board = flip(move0_chosen);
    entry.race = is_race(move1_board);

    const bool cube_aware = (select_cubes != nullptr && n_select_cubes > 0);
    std::array<CubeInfo, 4> m1_cubes;
    if (cube_aware) {
        for (int i = 0; i < n_select_cubes && i < 4; ++i) {
            m1_cubes[i] = flip_cube_perspective(select_cubes[i]);
        }
    }

    const Board opp_board = flip(move1_board);
    if (bearoff_db_ && bearoff_db_->is_bearoff(move1_board)) {
        entry.mover_probs = bearoff_db_->lookup_probs(move1_board);
    } else {
        const Strategy& mover_strat = base_bearoff_ ? *base_bearoff_ : *base_;
        entry.mover_probs = invert_probs(mover_strat.evaluate_probs(opp_board, opp_board));
    }
    clamp_probs_to_board(entry.mover_probs, move1_board);
    {
        auto [pp, op] = pip_counts(move1_board);
        entry.cube_x = cube_efficiency(entry.mover_probs, entry.race, pp, op);
    }

    TrialMoveScratch<std::array<std::vector<Board>, 21>> scratch;
    auto& reply_candidates = scratch.get();
    for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
        possible_boards_unsorted(move1_board, ALL_ROLLS[i].d1, ALL_ROLLS[i].d2,
                                 reply_candidates[i]);
    }
    const CubeInfo* cubes = cube_aware ? m1_cubes.data() : nullptr;
    evaluate_vr_rolls(move1_board, reply_candidates, cubes, n_select_cubes, entry.cube_x,
                      entry.best_candidate_idx, entry.roll_best_probs);
    for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
        const int pick = entry.best_candidate_idx[i];
        entry.chosen[i] = pick < 0 ? move1_board : reply_candidates[i][pick];
    }

    entry.cl_mean_probs = {0, 0, 0, 0, 0};
    for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
        for (int k = 0; k < NUM_OUTPUTS; ++k) {
            entry.cl_mean_probs[k] += ALL_ROLLS[i].weight * entry.roll_best_probs[i][k];
        }
    }
    for (int k = 0; k < NUM_OUTPUTS; ++k) {
        entry.cl_mean_probs[k] /= 36.0;
    }
    entry.cl_mean_eq =
        2.0 * entry.cl_mean_probs[0] - 1.0 +
        entry.cl_mean_probs[1] - entry.cl_mean_probs[3] +
        entry.cl_mean_probs[2] - entry.cl_mean_probs[4];
}

void RolloutStrategy::prefill_move1_cache(
    const Move0Cache& move0_cache, Move1Cache& cache, int n_threads,
    SharedPosCache* shared, const CubeInfo* select_cubes, int n_select_cubes) const
{
    auto populate_entry = [&](int roll_idx) {
        ScopedSharedCache shared_guard(shared, /*apply_null=*/false);
        populate_move1_cache_entry(move0_cache, roll_idx, cache.entries[roll_idx],
                                   select_cubes, n_select_cubes);
        cache.state[roll_idx].store(2, std::memory_order_release);
    };

    const int workers = std::min<int>(Move0Cache::N_ROLLS, std::max(1, n_threads));
    if (workers <= 1) {
        for (int i = 0; i < Move0Cache::N_ROLLS; ++i) {
            if (is_cancelled()) throw RolloutCancelled();
            populate_entry(i);
        }
    } else {
        multipy_parallel_for(Move0Cache::N_ROLLS, workers, populate_entry);
    }
}

} // namespace bgbot
