// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// Parallel trial runners (cubeless, unified single/batched cubeful cube
// decision, cubeful post-move rollout) and public Strategy overrides for
// RolloutStrategy.
#include "rollout_internal.h"
#include <numeric>

namespace bgbot {

using namespace rollout_detail;

// ======================== Parallel Trial Execution (Cubeless) ========================

RolloutResult RolloutStrategy::run_trials_parallel(
    const Board& board,
    RolloutProgressCallback progress) const
{
    std::lock_guard<std::recursive_mutex> instance_lock(rollout_mutex_);
    MultiPlyStrategy::get_cache().clear();
    clear_cubeful_eval_cache();

    const int n_trials = config_.n_trials;
    const int max_moves = (config_.truncation_depth > 0)
        ? config_.truncation_depth + 10
        : 200;

    if (cached_dice_.empty() || cached_max_moves_ != max_moves) {
        cached_dice_.clear();
        cached_max_moves_ = max_moves;
        generate_stratified_dice(n_trials, max_moves, config_.seed, cached_dice_);
    }
    const auto& all_dice = cached_dice_;

    const int n_threads = rollout_thread_count(n_trials);
    const int chunk_size = trial_chunk_size(config_.truncation_depth, n_trials, n_threads);

    Move0Cache move0_cache;
    auto move1_cache_owner = std::make_unique<Move1Cache>(config_);
    Move1Cache& move1_cache = *move1_cache_owner;
    const bool uses_move1_cache =
        (config_.truncation_depth == 0) || (config_.truncation_depth > 1);

    prefill_move0_cache(flip(board), move0_cache, n_threads);
    if (uses_move1_cache) {
        prefill_move1_cache(move0_cache, move1_cache, n_threads);
    }

    RolloutStatsAccumulator stats;
    const int report_interval = progress
        ? trial_report_interval(config_.truncation_depth, n_trials, n_threads) : 0;

    if (n_threads == 1) {
        if (is_cancelled()) throw RolloutCancelled();
        for (int t = 0; t < n_trials; ++t) {
            if (is_cancelled()) throw RolloutCancelled();
            auto r = run_trial_unified(board, true, nullptr, 0, all_dice[t].data(), max_moves,
                                       &move0_cache, &move1_cache);
            stats.add_cubeless(r.probs, r.equity, r.scalar_vr_equity);
            if (report_interval && ((t + 1) % report_interval == 0 || t + 1 == n_trials)) {
                progress(t + 1, n_trials);
            }
        }
        return stats.finalize_cubeless(/*equity_from_mean_probs=*/false);
    }

    std::vector<TrialResult> trial_results(n_trials);
    {
        if (!shared_pos_cache_) {
            shared_pos_cache_ = std::make_unique<SharedPosCache>();
        }
        if (shared_pos_cache_->inserts.load(std::memory_order_relaxed) >=
            (SharedPosCache::CAPACITY * 3) / 4) {
            shared_pos_cache_->clear();
        }
        SharedPosCache* shared_cache = shared_pos_cache_.get();
        std::atomic<int> next_trial{0};
        MonotonicProgressReporter progress_reporter(progress, n_trials, report_interval, chunk_size);

        multipy_parallel_run(n_threads, [&]() {
            MultiPlyStrategy::get_cache().clear();
            clear_cubeful_eval_cache();
            ScopedSharedCache shared_guard(shared_cache);
            int start;
            while ((start = next_trial.fetch_add(chunk_size, std::memory_order_relaxed))
                   < n_trials) {
                if (is_cancelled()) {
                    break;
                }
                int end = std::min(start + chunk_size, n_trials);
                for (int t = start; t < end; ++t) {
                    trial_results[t] = run_trial_unified(
                        board, true, nullptr, 0,
                        all_dice[t].data(), max_moves,
                        &move0_cache, &move1_cache);
                }
                progress_reporter.report_chunk(end - start);
            }
        });

        if (is_cancelled()) {
            throw RolloutCancelled();
        }
    }

    for (int t = 0; t < n_trials; ++t) {
        stats.add_cubeless(
            trial_results[t].probs,
            trial_results[t].equity,
            trial_results[t].scalar_vr_equity);
    }
    return stats.finalize_cubeless(/*equity_from_mean_probs=*/false);
}

// ======================== Unified Cubeful Cube Decision Runner ========================

RolloutStrategy::CubefulRolloutResult RolloutStrategy::cubeful_cube_decision_impl(
    const Board& pre_roll_board,
    const CubeInfo& cube,
    RolloutProgressCallback progress,
    int max_batches,
    double target_se) const
{
    std::lock_guard<std::recursive_mutex> instance_lock(rollout_mutex_);
    MultiPlyStrategy::get_cache().clear();
    clear_cubeful_eval_cache();

    const int n_trials = config_.n_trials;
    const int max_moves = (config_.truncation_depth > 0)
        ? config_.truncation_depth + 10 : 200;
    const int batch_limit = std::max(1, max_batches);

    // Reuse cached stratified dice for batch 0 (seed == config_.seed).
    if (cached_dice_.empty() || cached_max_moves_ != max_moves) {
        cached_dice_.clear();
        cached_max_moves_ = max_moves;
        generate_stratified_dice(n_trials, max_moves, config_.seed, cached_dice_);
    }

    // Branch templates
    CubefulBranch nd_template{};
    nd_template.cube = cube;
    nd_template.basis_cube = cube.cube_value;

    CubefulBranch dt_template{};
    dt_template.cube = cube;
    dt_template.cube.cube_value = 2 * cube.cube_value;
    dt_template.cube.owner = CubeOwner::OPPONENT;
    dt_template.basis_cube = cube.cube_value;

    struct CubefulTrialResult {
        double nd_equity;
        double dt_equity;
        TrialResult cubeless;
    };
    std::vector<CubefulTrialResult> trial_results(n_trials);

    const int n_threads = rollout_thread_count(n_trials);
    const int chunk_size = trial_chunk_size(config_.truncation_depth, n_trials, n_threads);
    const bool uses_move1_cache =
        (config_.truncation_depth == 0) || (config_.truncation_depth > 1);

    std::array<CubeInfo, 2> select_cubes_for_cache{nd_template.cube, dt_template.cube};
    const CubeInfo* select_cubes_ptr = nullptr;
    int n_select_cubes = 0;
    if (config_.cubeful_trial_moves) {
        select_cubes_ptr = select_cubes_for_cache.data();
        n_select_cubes = 2;
    }

    Move0Cache move0_cache;
    auto move1_cache_owner = std::make_unique<Move1Cache>(config_);
    Move1Cache& move1_cache = *move1_cache_owner;

    SharedPosCache* shared_cache = nullptr;
    if (n_threads > 1) {
        if (!shared_pos_cache_) {
            shared_pos_cache_ = std::make_unique<SharedPosCache>();
        }
        if (shared_pos_cache_->inserts.load(std::memory_order_relaxed) >=
            (SharedPosCache::CAPACITY * 3) / 4) {
            shared_pos_cache_->clear();
        }
        shared_cache = shared_pos_cache_.get();

        // Reserve every scheduled entry before any batch-0 worker can start a
        // trial, preventing early trials from racing prefill writers.
        for (int r = 0; r < Move0Cache::N_ROLLS; ++r) {
            move0_cache.state[r].store(1, std::memory_order_relaxed);
            if (uses_move1_cache) {
                move1_cache.state[r].store(1, std::memory_order_relaxed);
            }
        }
    } else {
        if (is_cancelled()) throw RolloutCancelled();
        prefill_move0_cache(pre_roll_board, move0_cache, 1, nullptr,
                            select_cubes_ptr, n_select_cubes);
        if (uses_move1_cache) {
            for (int i = 0; i < Move0Cache::N_ROLLS; ++i) {
                if (is_cancelled()) throw RolloutCancelled();
                populate_move1_cache_entry(move0_cache, i, move1_cache.entries[i],
                                           select_cubes_ptr, n_select_cubes);
                move1_cache.state[i].store(2, std::memory_order_release);
            }
        }
    }

    float cube_x_stamp = 0.0f;
    if (select_cubes_ptr) {
        std::array<float, NUM_OUTPUTS> dummy{};
        auto [pp, op] = pip_counts(pre_roll_board);
        cube_x_stamp = cube_efficiency(dummy, is_race(pre_roll_board), pp, op);
    }

    const int report_interval = progress
        ? trial_report_interval(config_.truncation_depth, n_trials, n_threads) : 0;

    RolloutStatsAccumulator stats;
    int batches_done = 0;
    bool converged = false;
    std::vector<std::vector<std::pair<int, int>>> extra_batch_dice;

    for (int b = 0; b < batch_limit; ++b) {
        const std::vector<std::vector<std::pair<int, int>>>* dice_ptr = &cached_dice_;
        if (b > 0) {
            const uint32_t batch_seed =
                config_.seed + static_cast<uint32_t>(b) * 0x9E3779B9u;
            generate_stratified_dice(n_trials, max_moves, batch_seed, extra_batch_dice);
            dice_ptr = &extra_batch_dice;
        }
        const auto& batch_dice = *dice_ptr;

        if (n_threads == 1) {
            for (int t = 0; t < n_trials; ++t) {
                if (is_cancelled()) throw RolloutCancelled();
                CubefulBranch branches[2] = {nd_template, dt_template};
                trial_results[t].cubeless = run_trial_unified(
                    pre_roll_board, false, branches, 2,
                    batch_dice[t].data(), max_moves, &move0_cache, &move1_cache);
                trial_results[t].nd_equity = branches[0].final_equity;
                trial_results[t].dt_equity = branches[1].final_equity;
                if (report_interval && ((t + 1) % report_interval == 0 || t + 1 == n_trials)) {
                    progress(t + 1, n_trials);
                }
            }
        } else {
            std::atomic<int> next_roll{0};
            std::atomic<int> next_trial{0};
            MonotonicProgressReporter progress_reporter(
                progress, n_trials, report_interval, chunk_size);
            const bool do_prefill = (b == 0);

            auto rollback_remaining_prefill = [&](int failed_roll) {
                auto mark_failed = [&](int idx) {
                    if (move0_cache.state[idx].load(std::memory_order_relaxed) == 1) {
                        move0_cache.state[idx].store(-1, std::memory_order_release);
                    }
                    if (uses_move1_cache &&
                        move1_cache.state[idx].load(std::memory_order_relaxed) == 1) {
                        move1_cache.state[idx].store(-1, std::memory_order_release);
                    }
                };
                mark_failed(failed_roll);
                int rem;
                while ((rem = next_roll.fetch_add(1, std::memory_order_relaxed))
                       < Move0Cache::N_ROLLS) {
                    mark_failed(rem);
                }
            };

            multipy_parallel_run(n_threads, [&]() {
                MultiPlyStrategy::get_cache().clear();
                clear_cubeful_eval_cache();
                ScopedSharedCache shared_guard(shared_cache);

                if (do_prefill) {
                    int r;
                    while ((r = next_roll.fetch_add(1, std::memory_order_relaxed))
                           < Move0Cache::N_ROLLS) {
                        if (is_cancelled()) {
                            rollback_remaining_prefill(r);
                            return;
                        }
                        try {
                            move0_cache.chosen[r] = select_move0_for_roll(
                                pre_roll_board, r, select_cubes_ptr, n_select_cubes,
                                cube_x_stamp);
                            move0_cache.state[r].store(2, std::memory_order_release);

                            if (uses_move1_cache) {
                                populate_move1_cache_entry(
                                    move0_cache, r, move1_cache.entries[r],
                                    select_cubes_ptr, n_select_cubes);
                                move1_cache.state[r].store(2, std::memory_order_release);
                            }
                        } catch (...) {
                            rollback_remaining_prefill(r);
                            throw;
                        }
                    }
                }

                int start;
                while ((start = next_trial.fetch_add(chunk_size, std::memory_order_relaxed))
                       < n_trials) {
                    if (is_cancelled()) {
                        break;
                    }
                    int end = std::min(start + chunk_size, n_trials);
                    for (int t = start; t < end; ++t) {
                        CubefulBranch branches[2] = {nd_template, dt_template};
                        trial_results[t].cubeless = run_trial_unified(
                            pre_roll_board, false, branches, 2,
                            batch_dice[t].data(), max_moves, &move0_cache, &move1_cache);
                        trial_results[t].nd_equity = branches[0].final_equity;
                        trial_results[t].dt_equity = branches[1].final_equity;
                    }
                    progress_reporter.report_chunk(end - start);
                }
            });

            if (is_cancelled()) {
                throw RolloutCancelled();
            }
        }

        for (int t = 0; t < n_trials; ++t) {
            stats.add_cubeful(
                trial_results[t].nd_equity,
                trial_results[t].dt_equity,
                trial_results[t].cubeless.probs,
                trial_results[t].cubeless.equity);
        }
        batches_done = b + 1;

        if (target_se <= 0.0 || stats.current_nd_se() <= target_se) {
            converged = true;
            break;
        }
    }

    return stats.finalize_cubeful(batches_done, converged);
}

RolloutStrategy::CubefulRolloutResult RolloutStrategy::cubeful_cube_decision(
    const Board& pre_roll_board,
    const CubeInfo& cube,
    RolloutProgressCallback progress) const
{
    return cubeful_cube_decision_impl(
        pre_roll_board, cube, std::move(progress), /*max_batches=*/1, /*target_se=*/0.0);
}

RolloutStrategy::CubefulRolloutResult RolloutStrategy::cubeful_cube_decision_batched(
    const Board& pre_roll_board,
    const CubeInfo& cube,
    RolloutProgressCallback progress) const
{
    const int max_batches = config_.max_batches > 0 ? config_.max_batches : 1;
    return cubeful_cube_decision_impl(
        pre_roll_board, cube, std::move(progress), max_batches, config_.target_se);
}

// ======================== Cubeful Position Rollout (post-move) ========================

RolloutStrategy::CubefulPositionResult RolloutStrategy::cubeful_rollout_position(
    const Board& post_move_board,
    const CubeInfo& cube,
    RolloutProgressCallback progress) const
{
    std::lock_guard<std::recursive_mutex> instance_lock(rollout_mutex_);

    {
        GameResult gr_term = check_game_over(post_move_board);
        if (gr_term != GameResult::NOT_OVER) {
            auto t_probs = terminal_probs(gr_term);
            CubefulPositionResult tr;
            if (cube.is_money()) {
                tr.cubeful_equity = cube.jacoby_active()
                    ? (2.0 * static_cast<double>(t_probs[0]) - 1.0)
                    : static_cast<double>(cubeless_equity(t_probs));
            } else {
                float mwc = cubeless_mwc(t_probs, cube.match.away1, cube.match.away2,
                                         cube.cube_value, cube.match.is_crawford);
                tr.cubeful_equity = static_cast<double>(
                    mwc2eq(mwc, cube.match.away1, cube.match.away2,
                           cube.cube_value, cube.match.is_crawford));
            }
            tr.cubeful_se = 0.0;
            tr.cubeless.mean_probs = t_probs;
            tr.cubeless.equity = static_cast<double>(cubeless_equity(t_probs));
            tr.cubeless.std_error = 0.0;
            tr.cubeless.scalar_vr_equity = tr.cubeless.equity;
            tr.cubeless.scalar_vr_se = 0.0;
            return tr;
        }
    }

    Board opp_board = flip(post_move_board);
    CubeInfo opp_cube = flip_cube_perspective(cube);

    // Honor target_se consistently with RolloutStrategy.cube_decision so
    // checker_play and cube_action preserve exact symmetry under target_se.
    CubefulRolloutResult cfr = (config_.target_se > 0.0)
        ? cubeful_cube_decision_batched(opp_board, opp_cube, std::move(progress))
        : cubeful_cube_decision(opp_board, opp_cube, std::move(progress));

    const float nd_raw = static_cast<float>(cfr.nd_equity);
    const float dt_raw = static_cast<float>(cfr.dt_equity);
    const CubeDecision cd = decision_from_raw_equities(
        opp_cube, nd_raw, dt_raw, /*dt_is_basis_scaled=*/true);

    double opt_se = 0.0;
    if (opp_cube.is_money()) {
        if (cd.should_double && !cd.should_take) {
            opt_se = 0.0;
        } else if (cd.should_double) {
            opt_se = cfr.dt_se;
        } else {
            opt_se = cfr.nd_se;
        }
    } else {
        const int a1 = opp_cube.match.away1;
        const int a2 = opp_cube.match.away2;
        const int cv = opp_cube.cube_value;
        const bool craw = opp_cube.match.is_crawford;
        if (cd.should_double && !cd.should_take) {
            opt_se = 0.0;
        } else if (cd.should_double) {
            opt_se = mwc_se_to_eq_se(
                static_cast<float>(cfr.dt_se), dt_raw, a1, a2, cv, craw);
        } else {
            opt_se = mwc_se_to_eq_se(
                static_cast<float>(cfr.nd_se), nd_raw, a1, a2, cv, craw);
        }
    }

    CubefulPositionResult result;
    result.cubeful_equity = -static_cast<double>(cd.optimal_equity);
    result.cubeful_se = opt_se;

    result.cubeless.mean_probs = invert_probs(cfr.cubeless.mean_probs);
    result.cubeless.prob_std_errors[0] = cfr.cubeless.prob_std_errors[0];
    result.cubeless.prob_std_errors[1] = cfr.cubeless.prob_std_errors[3];
    result.cubeless.prob_std_errors[2] = cfr.cubeless.prob_std_errors[4];
    result.cubeless.prob_std_errors[3] = cfr.cubeless.prob_std_errors[1];
    result.cubeless.prob_std_errors[4] = cfr.cubeless.prob_std_errors[2];
    result.cubeless.equity = -cfr.cubeless.equity;
    result.cubeless.std_error = cfr.cubeless.std_error;
    result.cubeless.scalar_vr_equity = -cfr.cubeless.scalar_vr_equity;
    result.cubeless.scalar_vr_se = cfr.cubeless.scalar_vr_se;

    return result;
}

// ======================== Public Interface ========================

double RolloutStrategy::evaluate(const Board& board, bool /*pre_move_is_race*/) const {
    auto r = rollout_position(board);
    return r.equity;
}

std::array<float, NUM_OUTPUTS> RolloutStrategy::evaluate_probs(
    const Board& board, bool /*pre_move_is_race*/) const
{
    auto r = rollout_position(board);
    return r.mean_probs;
}

std::array<float, NUM_OUTPUTS> RolloutStrategy::evaluate_probs(
    const Board& board, const Board& /*pre_move_board*/) const
{
    auto r = rollout_position(board);
    return r.mean_probs;
}

RolloutResult RolloutStrategy::rollout_position(
    const Board& board,
    RolloutProgressCallback progress) const
{
    GameResult gr_term = check_game_over(board);
    if (gr_term != GameResult::NOT_OVER) {
        RolloutResult tr;
        tr.mean_probs = terminal_probs(gr_term);
        tr.equity = static_cast<double>(cubeless_equity(tr.mean_probs));
        tr.scalar_vr_equity = tr.equity;
        return tr;
    }
    rollout_profile::reset();
    auto result = run_trials_parallel(board, std::move(progress));
    rollout_profile::print();
    return result;
}

int RolloutStrategy::best_move_index(const std::vector<Board>& candidates,
                                      bool pre_move_is_race) const
{
    if (candidates.empty()) return 0;
    return base_->best_move_index(candidates, pre_move_is_race);
}

int RolloutStrategy::best_move_index(const std::vector<Board>& candidates,
                                      const Board& pre_move_board) const
{
    std::lock_guard<std::recursive_mutex> instance_lock(rollout_mutex_);
    const int n = static_cast<int>(candidates.size());
    if (n <= 1) return 0;

    clear_internal_caches();

    std::vector<double> equities(n);
    double best_1ply = -1e30;

    base_->batch_evaluate_candidates_equity(
        candidates, pre_move_board, equities.data());
    for (int i = 0; i < n; ++i) {
        if (equities[i] > best_1ply) best_1ply = equities[i];
    }

    std::vector<int> sorted_indices(n);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](int a, int b) { return equities[a] > equities[b]; });

    std::vector<int> survivors;
    survivors.reserve(std::min(n, config_.filter.max_moves));
    for (int idx : sorted_indices) {
        if (static_cast<int>(survivors.size()) >= config_.filter.max_moves) break;
        if (best_1ply - equities[idx] > config_.filter.threshold) break;
        survivors.push_back(idx);
    }

    if (survivors.size() == 1) return survivors[0];

    double best_rollout = -1e30;
    int best_idx = survivors[0];

    for (int idx : survivors) {
        auto r = rollout_position(candidates[idx]);
        if (r.equity > best_rollout) {
            best_rollout = r.equity;
            best_idx = idx;
        }
    }

    return best_idx;
}

// ========================== Cube-aware best_move_index ==========================

int RolloutStrategy::best_move_index_cubeful(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    const CubeInfo& ci,
    float cube_x) const
{
    int out = 0;
    best_move_index_cubeful_multi(candidates, pre_move_board, &ci, 1, cube_x, &out);
    return out;
}

void RolloutStrategy::best_move_index_cubeful_multi(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    const CubeInfo* cubes,
    int n_cubes,
    float cube_x,
    int* out_indices) const
{
    std::lock_guard<std::recursive_mutex> instance_lock(rollout_mutex_);
    const int n = static_cast<int>(candidates.size());
    if (n == 0 || n_cubes <= 0) return;
    if (n == 1) {
        for (int c = 0; c < n_cubes; ++c) out_indices[c] = 0;
        return;
    }

    clear_internal_caches();

    // Shared per-cube 1-ply cubeful filter with deterministic cubeless_equity
    // tie-breaking (matching MultiPlyStrategy::best_move_index_cubeful_multi).
    std::vector<int> survivors = filter_candidates_cubeful_1ply(
        *base_, candidates, pre_move_board, cubes, n_cubes, cube_x,
        config_.filter.max_moves, config_.filter.threshold);

    if (survivors.size() == 1) {
        for (int c = 0; c < n_cubes; ++c) out_indices[c] = survivors[0];
        return;
    }

    const int n_surv = static_cast<int>(survivors.size());
    std::vector<std::vector<double>> cf_equities(n_surv, std::vector<double>(n_cubes));
    for (int i = 0; i < n_surv; ++i) {
        for (int c = 0; c < n_cubes; ++c) {
            auto r = cubeful_rollout_position(candidates[survivors[i]], cubes[c]);
            cf_equities[i][c] = r.cubeful_equity;
        }
    }

    for (int c = 0; c < n_cubes; ++c) {
        double best_cf = -1e30;
        int best_surv_i = 0;
        for (int i = 0; i < n_surv; ++i) {
            if (cf_equities[i][c] > best_cf) {
                best_cf = cf_equities[i][c];
                best_surv_i = i;
            }
        }
        out_indices[c] = survivors[best_surv_i];
    }
}

} // namespace bgbot
