// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// Unified rollout trial execution (run_trial_unified) decomposed into explicit
// per-phase helpers: Move1 cache lookup, Phase 1 cube checks, Phase 3 VR mean,
// Phase 4 move selection, Phase 4b VR luck, Phase 5 terminal scoring, and
// truncation evaluation.
#include "rollout_internal.h"

namespace bgbot {

using namespace rollout_detail;

namespace {

// End-of-game test gating the dead-cube retry of the cube-decision screen.
// True only when both sides are down to at most two checkers (bar included)
// and at least one side is within 24 pips.
bool is_endgame_cube_position(const Board& board) {
    int mover = board[25];
    int opp   = board[0];
    for (int i = 1; i <= 24; ++i) {
        if (board[i] > 0)      mover += board[i];
        else if (board[i] < 0) opp   -= board[i];
        if (mover > 2 || opp > 2) return false;
    }
    auto [player_pips, opp_pips] = pip_counts(board);
    return player_pips <= 24 || opp_pips <= 24;
}

} // namespace

// ======================== Cube Decision Application ========================

void RolloutStrategy::apply_cube_decision(
    CubefulBranch& branch,
    const CubeDecision& cd,
    bool is_match,
    bool is_sp_turn)
{
    if (!cd.should_double) return;
    if (cd.is_beaver) {
        branch.cube.cube_value *= 4;
        branch.cube.owner = CubeOwner::OPPONENT;
    } else if (cd.should_take) {
        branch.cube.cube_value *= 2;
        branch.cube.owner = CubeOwner::OPPONENT;
    } else {
        double sp_val;
        if (is_match) {
            float mwc = dp_mwc(
                branch.cube.match.away1,
                branch.cube.match.away2,
                branch.cube.cube_value,
                branch.cube.match.is_crawford);
            sp_val = is_sp_turn ? static_cast<double>(mwc)
                                : (1.0 - static_cast<double>(mwc));
        } else {
            sp_val = static_cast<double>(branch.cube.cube_value)
                     / branch.basis_cube;
            if (!is_sp_turn) sp_val = -sp_val;
        }
        branch.final_equity = sp_val - branch.vr_luck;
        branch.finished = true;
    }
}

// ======================== Move1 Cache Acquire ========================

const RolloutStrategy::Move1Cache::Entry* RolloutStrategy::acquire_move1_entry(
    int move0_roll_idx,
    const CubefulBranch branches[], int n_branches,
    bool use_cubeful_select,
    Move0Cache* move0_cache,
    Move1Cache* move1_cache) const
{
    if (!move1_cache || move0_roll_idx < 0 || move0_roll_idx >= Move0Cache::N_ROLLS) {
        return nullptr;
    }
    const int st = cas_claim_or_wait(move1_cache->state[move0_roll_idx], [&]() {
        // At move_num==1, Phase 6 already flipped each branch's cube to move-1
        // mover perspective; flip back so populate_move1_cache_entry's internal
        // flip restores move-1 alignment.
        std::array<CubeInfo, 4> sel_storage;
        const CubeInfo* sel = nullptr;
        int n_sel = 0;
        if (use_cubeful_select && n_branches > 0) {
            for (int b = 0; b < n_branches && b < 4; ++b) {
                if (!branches[b].finished) {
                    sel_storage[n_sel++] = flip_cube_perspective(branches[b].cube);
                }
            }
            sel = (n_sel > 0) ? sel_storage.data() : nullptr;
        }
        populate_move1_cache_entry(*move0_cache, move0_roll_idx,
                                   move1_cache->entries[move0_roll_idx],
                                   sel, n_sel);
    });
    if (st != 2) {
        if (is_cancelled()) throw RolloutCancelled();
        throw std::runtime_error("Move1Cache prefill failed");
    }
    return &move1_cache->entries[move0_roll_idx];
}

// ======================== Phase 1: Cube Check ========================

bool RolloutStrategy::run_phase1_cube_check(
    const Board& board, int move_num, int move0_roll_idx,
    bool race, bool is_match, bool is_sp_turn,
    CubefulBranch branches[], int n_branches,
    const Move1Cache::Entry* move1_entry,
    Move1Cache* move1_cache,
    float& cube_x, bool& cube_x_ready) const
{
    const TrialEvalConfig* cube_cfg = &cube_eval_config_;
    const bool use_cube_1ply = (cube_cfg->ply <= 1 && !cube_cfg->is_rollout());

    if (use_cube_1ply) {
        std::array<float, NUM_OUTPUTS> mover_probs;
        if (move1_entry) {
            mover_probs = move1_entry->mover_probs;
            cube_x = move1_entry->cube_x;
            cube_x_ready = true;
        } else if (bearoff_db_ && bearoff_db_->is_bearoff(board)) {
            mover_probs = bearoff_db_->lookup_probs(board);
            auto [pp, op] = pip_counts(board);
            cube_x = cube_efficiency(mover_probs, race, pp, op);
            cube_x_ready = true;
        } else {
            Board opp_board = flip(board);
            auto opp_probs = base_->evaluate_probs(opp_board, opp_board);
            mover_probs = invert_probs(opp_probs);
            auto [pp, op] = pip_counts(board);
            cube_x = cube_efficiency(mover_probs, race, pp, op);
            cube_x_ready = true;
        }
        clamp_probs_to_board(mover_probs, board);

        for (int b = 0; b < n_branches; ++b) {
            if (branches[b].finished) continue;
            if (!can_double(branches[b].cube)) continue;
            CubeDecision cd = cube_decision_1ply(mover_probs, branches[b].cube, cube_x);
            apply_cube_decision(branches[b], cd, is_match, is_sp_turn);
        }
    } else {
        MoveFilter cube_filter = {2, 0.03f};
        const RolloutStrategy* cube_rollout =
            cube_cfg->is_rollout() ? cube_inner_rollout_.get() : nullptr;

        if (!cube_x_ready) {
            if (move1_entry) {
                cube_x = move1_entry->cube_x;
            } else {
                std::array<float, NUM_OUTPUTS> dummy_probs{};
                auto [pp, op] = pip_counts(board);
                cube_x = cube_efficiency(dummy_probs, race, pp, op);
            }
            cube_x_ready = true;
        }

        CubeDecision cds[8];
        int cd_branch[8];
        int n_cd = 0;

        if (cube_rollout == nullptr) {
            // 1-ply screen + escalate (with dead-cube retry in near-terminal endgames).
            std::array<float, NUM_OUTPUTS> screen_probs;
            bool screen_ready = false;
            bool endgame = false, endgame_ready = false;
            CubeInfo cubes_in[8];
            for (int b = 0; b < n_branches; ++b) {
                if (branches[b].finished) continue;
                if (!can_double(branches[b].cube)) continue;
                if (!screen_ready) {
                    if (move1_entry) {
                        screen_probs = move1_entry->mover_probs;
                    } else if (bearoff_db_ && bearoff_db_->is_bearoff(board)) {
                        screen_probs = bearoff_db_->lookup_probs(board);
                    } else {
                        Board opp_board = flip(board);
                        auto opp_probs = base_->evaluate_probs(opp_board, opp_board);
                        screen_probs = invert_probs(opp_probs);
                    }
                    clamp_probs_to_board(screen_probs, board);
                    screen_ready = true;
                }
                CubeDecision cd1 = cube_decision_1ply(screen_probs, branches[b].cube, cube_x);
                if (!cd1.should_double) {
                    if (!endgame_ready) {
                        endgame = is_endgame_cube_position(board);
                        endgame_ready = true;
                    }
                    if (!endgame) continue;
                    CubeDecision cd0 = cube_decision_1ply(screen_probs, branches[b].cube, 0.0f);
                    if (!cd0.should_double) continue;
                }
                cd_branch[n_cd] = b;
                cubes_in[n_cd]  = branches[b].cube;
                ++n_cd;
            }
            if (n_cd > 0) {
                const Strategy& cf_base = base_bearoff_ ? *base_bearoff_ : *base_;
                bool served = false;
                if (move_num == 1 && move1_cache &&
                    move0_roll_idx >= 0 &&
                    move0_roll_idx < Move0Cache::N_ROLLS &&
                    n_branches <= 2) {
                    const int st = cas_claim_or_wait(
                        move1_cache->cd_state[move0_roll_idx], [&]() {
                            CubeDecision tmp[8];
                            cube_decision_nply_multi(
                                board, cubes_in, n_cd, cf_base,
                                cube_cfg->ply, tmp, cube_filter,
                                1 /*serial*/, move_filter_.get(),
                                /*deep_prefilter=*/true, {}, config_.cube_eval_options);
                            uint8_t mask = 0;
                            for (int k = 0; k < n_cd; ++k) {
                                int b = cd_branch[k];
                                move1_cache->cd[move0_roll_idx][b] = tmp[k];
                                move1_cache->cd_fp[move0_roll_idx][b] =
                                    cube_state_fingerprint(&cubes_in[k], 1);
                                mask |= static_cast<uint8_t>(1u << b);
                            }
                            move1_cache->cd_mask[move0_roll_idx] = mask;
                        });
                    if (st == 2) {
                        bool ok = true;
                        for (int k = 0; k < n_cd; ++k) {
                            int b = cd_branch[k];
                            if (!(move1_cache->cd_mask[move0_roll_idx] & (1u << b)) ||
                                move1_cache->cd_fp[move0_roll_idx][b] !=
                                    cube_state_fingerprint(&cubes_in[k], 1)) {
                                ok = false;
                                break;
                            }
                        }
                        if (ok) {
                            for (int k = 0; k < n_cd; ++k) {
                                cds[k] = move1_cache->cd[move0_roll_idx][cd_branch[k]];
                            }
                            served = true;
                        }
                    }
                }
                if (!served) {
                    cube_decision_nply_multi(
                        board, cubes_in, n_cd, cf_base,
                        cube_cfg->ply, cds, cube_filter,
                        1 /*serial*/, move_filter_.get(),
                        /*deep_prefilter=*/true, {}, config_.cube_eval_options);
                }
            }
        } else {
            // Truncated-rollout cube decisions: per-branch, routed through the
            // canonical decision_from_raw_equities helper so money beaver and
            // match post-Crawford auto_double stay in lockstep with N-ply/1-ply.
            for (int b = 0; b < n_branches; ++b) {
                if (branches[b].finished) continue;
                if (!can_double(branches[b].cube)) continue;

                auto cfr = cube_rollout->cubeful_cube_decision(board, branches[b].cube);
                cd_branch[n_cd] = b;
                cds[n_cd] = decision_from_raw_equities(
                    branches[b].cube,
                    static_cast<float>(cfr.nd_equity),
                    static_cast<float>(cfr.dt_equity),
                    /*dt_is_basis_scaled=*/true);
                ++n_cd;
            }
        }

        for (int k = 0; k < n_cd; ++k) {
            apply_cube_decision(branches[cd_branch[k]], cds[k], is_match, is_sp_turn);
        }
    }

    for (int b = 0; b < n_branches; ++b) {
        if (!branches[b].finished) return false;
    }
    return true;
}

RolloutStrategy::TrialResult RolloutStrategy::finalize_all_dp_early_exit(
    const Board& board, bool is_sp_turn,
    const std::array<double, NUM_OUTPUTS>& cl_accumulated_luck,
    double cl_scalar_eq_luck) const
{
    Board opp_board_early = flip(board);
    const Strategy& early_strat = base_bearoff_ ? *base_bearoff_ : *base_;
    auto opp_probs_early = early_strat.evaluate_probs(opp_board_early, opp_board_early);
    auto early_probs = invert_probs(opp_probs_early);
    clamp_probs_to_board(early_probs, board);

    std::array<float, NUM_OUTPUTS> sp_probs =
        is_sp_turn ? early_probs : invert_probs(early_probs);
    double raw_eq = compute_equity(sp_probs);
    std::array<float, NUM_OUTPUTS> vr_probs;
    for (int k = 0; k < NUM_OUTPUTS; ++k) {
        vr_probs[k] = static_cast<float>(sp_probs[k] - cl_accumulated_luck[k]);
    }
    return {vr_probs, raw_eq - cl_scalar_eq_luck, raw_eq - cl_scalar_eq_luck};
}

// ======================== Phase 3: VR Mean ========================

void RolloutStrategy::run_phase3_vr_mean(
    const Board& board, int d1, int d2, int actual_idx,
    bool do_vr, bool cube_active, bool is_match, bool use_cubeful_select_now,
    float cube_x,
    const CubefulBranch branches[], int n_branches,
    const Move1Cache::Entry* move1_entry,
    Move1Cache* move1_cache,
    std::array<std::vector<Board>, 21>& move_candidates,
    std::array<std::array<float, NUM_OUTPUTS>, 21>& roll_best_probs,
    std::array<int, 21>& best_candidate_idx,
    std::array<double, NUM_OUTPUTS>& cl_mean_probs,
    double& cl_mean_eq,
    double mean_cf_branch[2]) const
{
    ROLLOUT_TIMER_START;
    std::array<CubeInfo, 4> vr_sel_cubes;
    int vr_n_sel = 0;
    if (do_vr && !move1_entry && use_cubeful_select_now) {
        for (int b = 0; b < n_branches && vr_n_sel < 4; ++b) {
            if (!branches[b].finished) vr_sel_cubes[vr_n_sel++] = branches[b].cube;
        }
    }
    const CubeInfo* vr_sel_ptr = vr_n_sel ? vr_sel_cubes.data() : nullptr;
    auto* cache = do_vr && !move1_entry && move1_cache ? move1_cache->vr_cache.get() : nullptr;
    const size_t slot = cache ? VarianceReductionCache::slot_for(board) : 0;
    const auto* entry = cache ? cache->lookup(slot, board, vr_sel_ptr, vr_n_sel, cube_x) : nullptr;

    if (!move1_entry) {
        if (do_vr && !entry) {
            for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                possible_boards_unsorted(board, ALL_ROLLS[i].d1, ALL_ROLLS[i].d2,
                                         move_candidates[i]);
            }
        } else {
            const auto& roll = ALL_ROLLS[actual_idx];
            possible_boards_unsorted(board, do_vr ? roll.d1 : d1, do_vr ? roll.d2 : d2,
                                     move_candidates[actual_idx]);
        }
    }

    mean_cf_branch[0] = 0.0;
    mean_cf_branch[1] = 0.0;
    cl_mean_probs = {0, 0, 0, 0, 0};
    cl_mean_eq = 0.0;

    if (do_vr) {
        if (move1_entry) {
            roll_best_probs = move1_entry->roll_best_probs;
            best_candidate_idx = move1_entry->best_candidate_idx;
            cl_mean_probs = move1_entry->cl_mean_probs;
            cl_mean_eq = move1_entry->cl_mean_eq;
        } else {
            if (entry) {
                roll_best_probs = entry->probabilities;
                best_candidate_idx = entry->picks;
            } else {
                evaluate_vr_rolls(board, move_candidates, vr_sel_ptr, vr_n_sel, cube_x,
                                  best_candidate_idx, roll_best_probs);
                if (cache) {
                    cache->publish(slot, board, vr_sel_ptr, vr_n_sel, cube_x,
                                   roll_best_probs, best_candidate_idx);
                }
            }

            for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                for (int k = 0; k < NUM_OUTPUTS; ++k)
                    cl_mean_probs[k] += ALL_ROLLS[i].weight * roll_best_probs[i][k];
            }
            for (int k = 0; k < NUM_OUTPUTS; ++k) cl_mean_probs[k] /= 36.0;
            cl_mean_eq = 2.0*cl_mean_probs[0]-1.0 + cl_mean_probs[1]-cl_mean_probs[3]
                         + cl_mean_probs[2]-cl_mean_probs[4];
        }

        if (cube_active) {
            for (int b = 0; b < n_branches; ++b) {
                if (branches[b].finished) continue;
                double mean_cf = 0.0;
                for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                    double val;
                    if (is_match) {
                        val = cl2cf_match(roll_best_probs[i], branches[b].cube, cube_x);
                    } else {
                        float cf = cl2cf_money(roll_best_probs[i], branches[b].cube.owner, cube_x,
                                                branches[b].cube.jacoby_active());
                        val = cf * branches[b].cube.cube_value / branches[b].basis_cube;
                    }
                    mean_cf += ALL_ROLLS[i].weight * val;
                }
                mean_cf_branch[b] = mean_cf / 36.0;
            }
        }
    }

    ROLLOUT_TIMER_ADD(vr_time_ns);
}

// ======================== Phase 4: Move Selection ========================

Board RolloutStrategy::run_phase4_select_move(
    const Board& board, int move_num, int actual_idx,
    bool do_vr, bool using_base, bool can_reuse_vr_idx,
    bool use_cubeful_select_now, float cube_x,
    const Strategy& current_strat,
    const CubefulBranch branches[], int n_branches,
    const Move1Cache::Entry* move1_entry,
    Move0Cache* move0_cache,
    const std::vector<Board>& candidates,
    const std::array<int, 21>& best_candidate_idx) const
{
#ifdef ROLLOUT_PROFILE
    auto _rp_bmi_timer = std::chrono::high_resolution_clock::now();
#endif
    Board chosen;

    if (use_cubeful_select_now) {
        if (move1_entry) {
            chosen = move1_entry->chosen[actual_idx];
        } else if (candidates.empty()) {
            chosen = board;
        } else if (candidates.size() == 1) {
            chosen = candidates[0];
        } else if (do_vr && using_base && !(move_num == 0 && move0_cache) &&
                   (!base_bearoff_ || std::any_of(candidates.begin(), candidates.end(),
                       [&](const Board& c) { return !bearoff_db_->is_bearoff(c); }))) {
            chosen = candidates[best_candidate_idx[actual_idx]];
        } else if (move0_cache && move_num == 0) {
            const int st = cas_claim_or_wait(move0_cache->state[actual_idx], [&]() {
                std::array<CubeInfo, 4> active_cubes;
                int n_active = 0;
                for (int b = 0; b < n_branches && b < 4; ++b) {
                    if (!branches[b].finished) {
                        active_cubes[n_active++] = branches[b].cube;
                    }
                }
                if (n_active == 0) {
                    move0_cache->chosen[actual_idx] =
                        candidates[current_strat.rollout_cubeless_move(candidates, board)];
                } else {
                    std::array<int, 4> picks{0, 0, 0, 0};
                    current_strat.best_move_index_cubeful_multi(
                        candidates, board,
                        active_cubes.data(), n_active, cube_x,
                        picks.data());
                    move0_cache->chosen[actual_idx] = candidates[picks[0]];
                }
            });
            if (st != 2) {
                if (is_cancelled()) throw RolloutCancelled();
                throw std::runtime_error("Move0Cache prefill failed");
            }
            chosen = move0_cache->chosen[actual_idx];
        } else {
            std::array<CubeInfo, 2> active_cubes;
            int n_active = 0;
            for (int b = 0; b < n_branches; ++b) {
                if (!branches[b].finished) {
                    active_cubes[n_active++] = branches[b].cube;
                }
            }
            if (n_active == 0) {
                chosen = candidates[current_strat.rollout_cubeless_move(candidates, board)];
            } else {
                std::array<int, 2> picks{0, 0};
                current_strat.best_move_index_cubeful_multi(
                    candidates, board,
                    active_cubes.data(), n_active, cube_x,
                    picks.data());
                chosen = candidates[picks[0]];
            }
        }
    } else if (move1_entry) {
        chosen = move1_entry->chosen[actual_idx];
    } else if (move0_cache && move_num == 0 && !using_base) {
        const int st = cas_claim_or_wait(move0_cache->state[actual_idx], [&]() {
            if (candidates.empty()) {
                move0_cache->chosen[actual_idx] = board;
            } else if (candidates.size() == 1) {
                move0_cache->chosen[actual_idx] = candidates[0];
            } else {
                move0_cache->chosen[actual_idx] =
                    candidates[current_strat.rollout_cubeless_move(candidates, board)];
            }
        });
        if (st != 2) {
            if (is_cancelled()) throw RolloutCancelled();
            throw std::runtime_error("Move0Cache prefill failed");
        }
        chosen = move0_cache->chosen[actual_idx];
    } else {
        if (candidates.empty()) {
            chosen = board;
        } else if (candidates.size() == 1) {
            chosen = candidates[0];
        } else if (using_base) {
            if (can_reuse_vr_idx && best_candidate_idx[actual_idx] >= 0 &&
                best_candidate_idx[actual_idx] < static_cast<int>(candidates.size())) {
                chosen = candidates[best_candidate_idx[actual_idx]];
            } else {
                chosen = candidates[base_->best_move_index(candidates, board)];
            }
        } else {
            chosen = candidates[current_strat.rollout_cubeless_move(candidates, board)];
        }
    }

#ifdef ROLLOUT_PROFILE
    rollout_profile::bmi_time_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - _rp_bmi_timer).count(),
        std::memory_order_relaxed);
#endif
    return chosen;
}

// ======================== Phase 4b: VR Luck ========================

void RolloutStrategy::run_phase4b_vr_luck(
    const Board& board, const Board& chosen, int actual_idx,
    bool cube_active, bool is_match, bool is_sp_turn, bool using_base,
    float cube_x,
    const Move1Cache::Entry* move1_entry,
    const std::vector<Board>& actual_candidates,
    const std::array<std::array<float, NUM_OUTPUTS>, 21>& roll_best_probs,
    const std::array<int, 21>& best_candidate_idx,
    const std::array<double, NUM_OUTPUTS>& cl_mean_probs,
    double cl_mean_eq,
    const double mean_cf_branch[2],
    CubefulBranch branches[], int n_branches,
    std::array<double, NUM_OUTPUTS>& cl_accumulated_luck,
    double& cl_scalar_eq_luck) const
{
    std::array<float, NUM_OUTPUTS> actual_probs;
    if (move1_entry) {
        actual_probs = move1_entry->roll_best_probs[actual_idx];
    } else if (using_base &&
               best_candidate_idx[actual_idx] >= 0 &&
               best_candidate_idx[actual_idx] < static_cast<int>(actual_candidates.size()) &&
               chosen == actual_candidates[best_candidate_idx[actual_idx]]) {
        actual_probs = roll_best_probs[actual_idx];
    } else {
        GameResult r = check_game_over(chosen);
        if (r != GameResult::NOT_OVER) {
            actual_probs = terminal_probs(r);
        } else {
            const Strategy& vr_base = base_bearoff_ ? *base_bearoff_ : *base_;
            actual_probs = vr_base.evaluate_probs(chosen, board);
            clamp_probs_to_board(actual_probs, chosen);
        }
    }

    if (cube_active) {
        for (int b = 0; b < n_branches; ++b) {
            if (branches[b].finished) continue;
            double actual_val;
            if (is_match) {
                actual_val = cl2cf_match(actual_probs, branches[b].cube, cube_x);
            } else {
                float cf = cl2cf_money(actual_probs, branches[b].cube.owner, cube_x,
                                        branches[b].cube.jacoby_active());
                actual_val = cf * branches[b].cube.cube_value / branches[b].basis_cube;
            }
            double luck = actual_val - mean_cf_branch[b];
            if (is_sp_turn) {
                branches[b].vr_luck += luck;
            } else {
                branches[b].vr_luck -= luck;
            }
        }
    }

    std::array<double, NUM_OUTPUTS> luck_mover;
    for (int k = 0; k < NUM_OUTPUTS; ++k)
        luck_mover[k] = actual_probs[k] - cl_mean_probs[k];
    double actual_eq = compute_equity(actual_probs);
    double luck_eq = actual_eq - cl_mean_eq;

    if (is_sp_turn) {
        for (int k = 0; k < NUM_OUTPUTS; ++k)
            cl_accumulated_luck[k] += luck_mover[k];
        cl_scalar_eq_luck += luck_eq;
    } else {
        cl_accumulated_luck[0] -= luck_mover[0];
        cl_accumulated_luck[1] += luck_mover[3];
        cl_accumulated_luck[2] += luck_mover[4];
        cl_accumulated_luck[3] += luck_mover[1];
        cl_accumulated_luck[4] += luck_mover[2];
        cl_scalar_eq_luck -= luck_eq;
    }
}

// ======================== Phase 5: Terminal ========================

RolloutStrategy::TrialResult RolloutStrategy::run_phase5_terminal(
    GameResult result,
    bool cube_active, bool is_match, bool is_sp_turn,
    CubefulBranch branches[], int n_branches,
    const std::array<double, NUM_OUTPUTS>& cl_accumulated_luck,
    double cl_scalar_eq_luck) const
{
    auto t_probs = terminal_probs(result);

    if (cube_active) {
        double mover_eq_full = compute_equity(t_probs);
        for (int b = 0; b < n_branches; ++b) {
            if (branches[b].finished) continue;
            double sp_val;
            if (is_match) {
                float mwc = cubeless_mwc(
                    t_probs,
                    branches[b].cube.match.away1,
                    branches[b].cube.match.away2,
                    branches[b].cube.cube_value,
                    branches[b].cube.match.is_crawford);
                sp_val = is_sp_turn ? static_cast<double>(mwc)
                                    : (1.0 - static_cast<double>(mwc));
            } else {
                double mover_eq = branches[b].cube.jacoby_active()
                    ? (2.0 * t_probs[0] - 1.0) : mover_eq_full;
                double points = mover_eq * branches[b].cube.cube_value;
                sp_val = points / branches[b].basis_cube;
                if (!is_sp_turn) sp_val = -sp_val;
            }
            branches[b].final_equity = sp_val - branches[b].vr_luck;
            branches[b].finished = true;
        }
    } else if (n_branches > 0) {
        std::array<float, NUM_OUTPUTS> sp_probs_t =
            is_sp_turn ? t_probs : invert_probs(t_probs);
        double raw_sp = compute_equity(sp_probs_t);
        double vr_sp = raw_sp - cl_scalar_eq_luck;
        for (int b = 0; b < n_branches; ++b) {
            if (branches[b].finished) continue;
            branches[b].final_equity = vr_sp
                * branches[b].cube.cube_value / branches[b].basis_cube;
            branches[b].finished = true;
        }
    }

    std::array<float, NUM_OUTPUTS> sp_probs =
        is_sp_turn ? t_probs : invert_probs(t_probs);
    double raw_eq = compute_equity(sp_probs);
    std::array<float, NUM_OUTPUTS> vr_probs;
    for (int k = 0; k < NUM_OUTPUTS; ++k)
        vr_probs[k] = static_cast<float>(sp_probs[k] - cl_accumulated_luck[k]);
    return {vr_probs, raw_eq - cl_scalar_eq_luck, raw_eq - cl_scalar_eq_luck};
}

// ======================== Truncation Evaluation ========================

RolloutStrategy::TrialResult RolloutStrategy::run_truncation_phase(
    const Board& board, int truncation, int max_moves, int sp_parity_offset,
    bool cube_active, bool is_match,
    CubefulBranch branches[], int n_branches,
    const std::array<double, NUM_OUTPUTS>& cl_accumulated_luck,
    double cl_scalar_eq_luck) const
{
    ROLLOUT_TIMER_START;
    Board last_mover_board = flip(board);
    int trunc_move = std::min(truncation, max_moves);
    bool trunc_race = is_race(last_mover_board);
    const auto& trunc_strat = *truncation_strat_;
    bool last_mover_is_sp = ((trunc_move - 1) % 2 == sp_parity_offset);

    std::array<float, NUM_OUTPUTS> last_mover_probs;
    bool cubeful_done = false;
    if (cube_active && truncation_ply_ > 1) {
        int active_idx[8];
        CubeInfo active_cubes[8];
        int n_active = 0;
        for (int b = 0; b < n_branches; ++b) {
            if (branches[b].finished) continue;
            active_idx[n_active] = b;
            active_cubes[n_active] = branches[b].cube;
            ++n_active;
        }
        if (n_active > 0) {
            float cfs[8];
            std::array<float, NUM_OUTPUTS> tree_probs{};
            const Strategy& cf_strat = base_bearoff_ ? *base_bearoff_ : *base_;
            cubeful_equity_nply_multi(
                board, active_cubes, n_active, cf_strat,
                truncation_ply_, cfs, MoveFilters::TINY,
                1 /*serial*/, move_filter_.get(),
                false, &tree_probs, /*deep_prefilter=*/true, nullptr,
                config_.cube_eval_options);
            last_mover_probs = invert_probs(tree_probs);
            clamp_probs_to_board(last_mover_probs, last_mover_board);

            for (int k = 0; k < n_active; ++k) {
                int b = active_idx[k];
                float cf = cfs[k];
                double sp_val;
                if (is_match) {
                    float mwc = eq2mwc(cf,
                        branches[b].cube.match.away1,
                        branches[b].cube.match.away2,
                        branches[b].cube.cube_value,
                        branches[b].cube.match.is_crawford);
                    sp_val = last_mover_is_sp ? (1.0 - static_cast<double>(mwc))
                                              : static_cast<double>(mwc);
                } else {
                    double points = cf * branches[b].cube.cube_value;
                    sp_val = points / branches[b].basis_cube;
                    if (last_mover_is_sp) sp_val = -sp_val;
                }
                branches[b].final_equity = sp_val - branches[b].vr_luck;
                branches[b].finished = true;
            }
            cubeful_done = true;
        }
    }

    if (!cubeful_done) {
        if (bearoff_db_ && bearoff_db_->is_bearoff(last_mover_board)) {
            last_mover_probs =
                bearoff_db_->lookup_probs(last_mover_board, /*post_move=*/true);
        } else if (truncation_ply_ > 1) {
            float cf_unused;
            std::array<float, NUM_OUTPUTS> tree_probs{};
            const Strategy& cf_strat = base_bearoff_ ? *base_bearoff_ : *base_;
            cubeful_equity_nply_multi(
                board, &dead_cube_info(), 1, cf_strat,
                truncation_ply_, &cf_unused, MoveFilters::TINY,
                1 /*serial*/, move_filter_.get(),
                false, &tree_probs, /*deep_prefilter=*/true, nullptr,
                config_.cube_eval_options);
            last_mover_probs = invert_probs(tree_probs);
        } else {
            last_mover_probs =
                trunc_strat.evaluate_probs(last_mover_board, last_mover_board);
        }
        clamp_probs_to_board(last_mover_probs, last_mover_board);
    }
    ROLLOUT_TIMER_ADD(trunc_time_ns);
#ifdef ROLLOUT_PROFILE
    rollout_profile::trial_count.fetch_add(1, std::memory_order_relaxed);
#endif

    if (!cubeful_done && cube_active) {
        auto [last_pp, last_op] = pip_counts(last_mover_board);
        float trunc_x = cube_efficiency(last_mover_probs, trunc_race, last_pp, last_op);
        for (int b = 0; b < n_branches; ++b) {
            if (branches[b].finished) continue;
            CubeInfo last_cube = flip_cube_perspective(branches[b].cube);
            double sp_val;
            if (is_match) {
                float mwc = cl2cf_match(last_mover_probs, last_cube, trunc_x);
                sp_val = last_mover_is_sp ? static_cast<double>(mwc)
                                          : (1.0 - static_cast<double>(mwc));
            } else {
                float cf = cl2cf_money(last_mover_probs, last_cube.owner, trunc_x,
                                        last_cube.jacoby_active());
                double points = cf * last_cube.cube_value;
                sp_val = points / branches[b].basis_cube;
                if (!last_mover_is_sp) sp_val = -sp_val;
            }
            branches[b].final_equity = sp_val - branches[b].vr_luck;
            branches[b].finished = true;
        }
    } else if (!cubeful_done && n_branches > 0) {
        std::array<float, NUM_OUTPUTS> sp_probs_t =
            last_mover_is_sp ? last_mover_probs : invert_probs(last_mover_probs);
        double raw_sp = compute_equity(sp_probs_t);
        double vr_sp = raw_sp - cl_scalar_eq_luck;
        for (int b = 0; b < n_branches; ++b) {
            if (branches[b].finished) continue;
            branches[b].final_equity = vr_sp
                * branches[b].cube.cube_value / branches[b].basis_cube;
            branches[b].finished = true;
        }
    }

    std::array<float, NUM_OUTPUTS> sp_probs =
        last_mover_is_sp ? last_mover_probs : invert_probs(last_mover_probs);
    double raw_eq = compute_equity(sp_probs);
    std::array<float, NUM_OUTPUTS> vr_probs;
    for (int k = 0; k < NUM_OUTPUTS; ++k)
        vr_probs[k] = static_cast<float>(sp_probs[k] - cl_accumulated_luck[k]);
    return {vr_probs, raw_eq - cl_scalar_eq_luck, raw_eq - cl_scalar_eq_luck};
}

// ======================== Unified Trial Driver ========================

RolloutStrategy::TrialResult RolloutStrategy::run_trial_unified(
    const Board& start_board,
    bool start_post_move,
    CubefulBranch branches[], int n_branches,
    const std::pair<int,int>* dice_seq,
    int max_moves,
    Move0Cache* move0_cache,
    Move1Cache* move1_cache) const
{
    TrialMoveScratch<std::array<std::vector<Board>, ALL_ROLLS.size()>> scratch;
    auto& move_candidates = scratch.get();

    bool cube_active = false;
    for (int b = 0; b < n_branches; ++b) {
        if (!cube_is_dead(branches[b].cube)) { cube_active = true; break; }
    }
    const bool is_match = cube_active && n_branches > 0 && !branches[0].cube.is_money();
    const bool use_cubeful_select =
        (config_.cubeful_trial_moves && cube_active && n_branches > 0);
    const int cubeful_late_threshold =
        (config_.cubeful_late_threshold > 0)
            ? config_.cubeful_late_threshold
            : config_.ultra_late_threshold;

    Board board;
    int sp_parity_offset;
    if (start_post_move) {
        board = flip(start_board);
        sp_parity_offset = 1;
        for (int b = 0; b < n_branches; ++b) {
            branches[b].cube.owner = flip_owner(branches[b].cube.owner);
            if (is_match) {
                std::swap(branches[b].cube.match.away1, branches[b].cube.match.away2);
            }
        }
    } else {
        board = start_board;
        sp_parity_offset = 0;
    }

    std::array<double, NUM_OUTPUTS> cl_accumulated_luck = {0, 0, 0, 0, 0};
    double cl_scalar_eq_luck = 0.0;
    const bool vr_enabled = vr_enabled_;
    const int truncation = (config_.truncation_depth > 0) ? config_.truncation_depth : 9999;
    int move0_roll_idx = -1;

    for (int move_num = 0; move_num < truncation && move_num < max_moves; ++move_num) {
        const Move1Cache::Entry* move1_entry = (move_num == 1)
            ? acquire_move1_entry(move0_roll_idx, branches, n_branches,
                                  use_cubeful_select, move0_cache, move1_cache)
            : nullptr;

        const bool is_sp_turn = (move_num % 2 == sp_parity_offset);
        const bool race = move1_entry ? move1_entry->race : is_race(board);
        const bool is_late = (move_num >= config_.late_threshold);
        float cube_x = 0.0f;
        bool cube_x_ready = false;

        // Phase 1: Cube check (cubeful only, skip on move 0)
        if (cube_active && move_num > 0) {
            if (run_phase1_cube_check(board, move_num, move0_roll_idx,
                                      race, is_match, is_sp_turn,
                                      branches, n_branches,
                                      move1_entry, move1_cache,
                                      cube_x, cube_x_ready)) {
                return finalize_all_dp_early_exit(
                    board, is_sp_turn, cl_accumulated_luck, cl_scalar_eq_luck);
            }
        }

        // Phase 2: Dice index + VR/selection gating
        const int d1 = dice_seq[move_num].first;
        const int d2 = dice_seq[move_num].second;
        int a_die = d1, b_die = d2;
        if (a_die > b_die) std::swap(a_die, b_die);
        int actual_idx = kOrderedRollToIndex[a_die][b_die];
        if (actual_idx < 0) actual_idx = 0;

        constexpr int kVrThinningMinTrials = 288;
        const bool skip_vr_this_move =
            (move_num == 0 && config_.n_trials % 36 == 0) ||
            (move_num >= config_.ultra_late_threshold && (move_num % 2 == 1) &&
             config_.n_trials >= kVrThinningMinTrials);
        const bool do_vr = vr_enabled && !skip_vr_this_move;
        const bool use_cubeful_select_now =
            use_cubeful_select && (move_num < cubeful_late_threshold);

        const auto& current_strat =
            (move_num >= config_.ultra_late_threshold ? *base_
                : (is_late ? *checker_late_strat_ : *checker_strat_));
        const bool using_base = (&current_strat == base_.get());
        const bool can_reuse_vr_idx = do_vr;

        if (cube_active && !cube_x_ready) {
            if (move1_entry) {
                cube_x = move1_entry->cube_x;
            } else {
                std::array<float, NUM_OUTPUTS> dummy_probs{};
                auto [pp, op] = pip_counts(board);
                cube_x = cube_efficiency(dummy_probs, race, pp, op);
            }
            cube_x_ready = true;
        }

        // Phase 3: VR mean
        std::array<std::array<float, NUM_OUTPUTS>, 21> roll_best_probs;
        std::array<int, 21> best_candidate_idx{};
        std::array<double, NUM_OUTPUTS> cl_mean_probs{};
        double cl_mean_eq = 0.0;
        double mean_cf_branch[2] = {0.0, 0.0};
        run_phase3_vr_mean(board, d1, d2, actual_idx,
                           do_vr, cube_active, is_match, use_cubeful_select_now,
                           cube_x, branches, n_branches,
                           move1_entry, move1_cache,
                           move_candidates, roll_best_probs, best_candidate_idx,
                           cl_mean_probs, cl_mean_eq, mean_cf_branch);

        // Phase 4: Select move for actual roll
        Board chosen = run_phase4_select_move(
            board, move_num, actual_idx,
            do_vr, using_base, can_reuse_vr_idx, use_cubeful_select_now,
            cube_x, current_strat, branches, n_branches,
            move1_entry, move0_cache,
            move_candidates[actual_idx], best_candidate_idx);
        if (move_num == 0) move0_roll_idx = actual_idx;

        // Phase 4b: VR luck
        if (do_vr) {
            run_phase4b_vr_luck(board, chosen, actual_idx,
                                cube_active, is_match, is_sp_turn, using_base,
                                cube_x, move1_entry,
                                move_candidates[actual_idx],
                                roll_best_probs, best_candidate_idx,
                                cl_mean_probs, cl_mean_eq, mean_cf_branch,
                                branches, n_branches,
                                cl_accumulated_luck, cl_scalar_eq_luck);
        }

        // Phase 5: Terminal check
        GameResult result = check_game_over(chosen);
        if (result != GameResult::NOT_OVER) {
            return run_phase5_terminal(result, cube_active, is_match, is_sp_turn,
                                       branches, n_branches,
                                       cl_accumulated_luck, cl_scalar_eq_luck);
        }

        // Phase 6: Flip board + cube ownership
        board = flip(chosen);
        if (cube_active) {
            for (int b = 0; b < n_branches; ++b) {
                if (!branches[b].finished) {
                    branches[b].cube.owner = flip_owner(branches[b].cube.owner);
                    if (is_match) {
                        std::swap(branches[b].cube.match.away1,
                                  branches[b].cube.match.away2);
                    }
                }
            }
        }
    }

    return run_truncation_phase(board, truncation, max_moves, sp_parity_offset,
                                cube_active, is_match, branches, n_branches,
                                cl_accumulated_luck, cl_scalar_eq_luck);
}

} // namespace bgbot
