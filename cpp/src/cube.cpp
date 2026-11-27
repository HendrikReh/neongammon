// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/cube.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include "bgbot/multipy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace bgbot {

void compute_WL(const std::array<float, NUM_OUTPUTS>& probs, float& W, float& L) {
    float p_win = probs[0];
    float p_gw = probs[1];
    float p_bw = probs[2];
    float p_gl = probs[3];
    float p_bl = probs[4];

    // W = average value of wins: 1 (single) + fraction that are gammon/bg
    if (p_win > 1e-7f) {
        W = 1.0f + (p_gw + p_bw) / p_win;
    } else {
        W = 1.0f;  // No wins — W doesn't matter, but avoid division by zero
    }

    // L = average value of losses
    float p_lose = 1.0f - p_win;
    if (p_lose > 1e-7f) {
        L = 1.0f + (p_gl + p_bl) / p_lose;
    } else {
        L = 1.0f;  // No losses
    }
}

float money_live(float W, float L, float p_win, CubeOwner owner,
                 bool jacoby) {
    // Take point and cash point (live cube)
    float TP = (L - 0.5f) / (W + L + 0.5f);
    float CP = (L + 1.0f) / (W + L + 0.5f);

    float p = p_win;

    switch (owner) {
        case CubeOwner::CENTERED: {
            // Interpolate through (0, -L), (TP, -1), (CP, +1), (1, +W)
            //
            // Under Jacoby, the extreme segments are clamped to ±1.0
            // (matching GNUbg's MoneyLive).  Rationale: above the cash
            // point, the player doubles and the opponent passes → +1.0;
            // below the take point, the opponent doubles and the player
            // passes → -1.0.  There is no "too good" / "too weak" premium
            // because gammons are worthless while the cube is centered.
            // The middle segment (doubling window) is unchanged because
            // once the cube is turned, gammons count — the real W/L apply.
            if (p < TP) {
                // Segment: (0, -L) → (TP, -1)
                return jacoby ? -1.0f
                              : -L + (-1.0f + L) * p / TP;
            } else if (p < CP) {
                // Segment: (TP, -1) → (CP, +1)
                return -1.0f + 2.0f * (p - TP) / (CP - TP);
            } else {
                // Segment: (CP, +1) → (1, +W)
                return jacoby ? 1.0f
                              : 1.0f + (W - 1.0f) * (p - CP) / (1.0f - CP);
            }
        }
        case CubeOwner::PLAYER: {
            // Player owns cube: interpolate through (0, -L), (CP, +1), (1, +W)
            if (p < CP) {
                return -L + (1.0f + L) * p / CP;
            } else {
                return 1.0f + (W - 1.0f) * (p - CP) / (1.0f - CP);
            }
        }
        case CubeOwner::OPPONENT: {
            // Opponent owns cube: interpolate through (0, -L), (TP, -1), (1, +W)
            if (p < TP) {
                return -L + (-1.0f + L) * p / TP;
            } else {
                return -1.0f + (W + 1.0f) * (p - TP) / (1.0f - TP);
            }
        }
    }
    return 0.0f;  // unreachable
}

float cl2cf_money(const std::array<float, NUM_OUTPUTS>& probs,
                  CubeOwner owner, float cube_x) {
    return cl2cf_money(probs, owner, cube_x, false);
}

float cl2cf_money(const std::array<float, NUM_OUTPUTS>& probs,
                  CubeOwner owner, float cube_x, bool jacoby_active) {
    float W, L;
    // Always compute real W/L from the full gammon/backgammon rates.
    // The live-cube equity uses real W/L because once the cube is turned,
    // gammons count (Jacoby deactivates).
    compute_WL(probs, W, L);

    // Dead-cube equity: under Jacoby with cube centered, the cube can
    // never be turned again means it stays centered forever, so gammons
    // never count → 2*P(win)-1.  Otherwise (Jacoby off, or cube already
    // turned) full cubeless equity with gammons/backgammons.
    float e_dead = jacoby_active ? (2.0f * probs[0] - 1.0f)
                                 : cubeless_equity(probs);

    // Live-cube equity: uses real W/L but with Jacoby clamping on the
    // extreme segments (p < TP → -1, p > CP → +1).  Matching GNUbg's
    // MoneyLive: the middle segment (doubling window) is unchanged because
    // doubling activates gammons; the extreme segments are clamped because
    // there is no "too good/too weak" premium under Jacoby (you'd just
    // double to cash or get doubled out).
    float e_live = money_live(W, L, probs[0], owner, jacoby_active);

    return e_dead * (1.0f - cube_x) + e_live * cube_x;
}

// ---------------------------------------------------------------------------
// Match play cubeful evaluation (Janowski in MWC space)
// ---------------------------------------------------------------------------

// Check if the cube is "dead" — no useful cube actions possible.
// This happens when the cube value is enough for both players to win the match
// on any win, or during Crawford game.
static bool is_dead_cube(const CubeInfo& cube) {
    if (cube_is_dead(cube)) return true;          // structural cap
    if (cube.is_money()) return false;
    if (cube.match.is_crawford) return true;
    // A cube can only be turned again by a player who BOTH may double it
    // (ownership) AND gains by doing so -- a player whose win at the current
    // cube value already takes the match gains nothing. Ownership matters:
    // with an owned cube only the owner can turn it, so the owner's away score
    // alone decides. Requiring BOTH away scores to be covered is correct only
    // for a CENTERED cube, and left e.g. 3-away/2-away with the cube on 2 owned
    // by the trailer looking live.
    const bool player_can_turn =
        (cube.owner == CubeOwner::CENTERED || cube.owner == CubeOwner::PLAYER)
        && cube.match.away1 > cube.cube_value;
    const bool opponent_can_turn =
        (cube.owner == CubeOwner::CENTERED || cube.owner == CubeOwner::OPPONENT)
        && cube.match.away2 > cube.cube_value;
    return !player_can_turn && !opponent_can_turn;
}

// Compute MWC-space take point and cash point for match play.
// These are the P(win) thresholds analogous to money game TP/CP.
//
// For the player's perspective:
// TP = opponent's take point (P(win) below which opponent should pass)
//    = (MWC_lose_cube - MWC_dp) / (MWC_lose_cube - MWC_win_cube)
// CP = player's cash point (P(win) above which player should play on for gammon)
//    = (MWC_dp - MWC_win_cube_lose) / (MWC_win_cube_win - MWC_win_cube_lose)
//
// But we use the Janowski formulation which works in MWC space directly.
// The take point and cash point in MWC space map to specific P(win) values.

// Compute cubeful MWC for match play — centered cube.
// Piecewise-linear interpolation with 3 regions:
//   p < opponent's TG: opponent too good to double
//   opponent's TG < p < player's TG: in doubling window
//   p > player's TG: player too good to double
// Compute live-cube cash points for both players simultaneously using
// GNUbg's recursive GetPoints algorithm.
//
// The algorithm works from the "dead" cube level (highest level where at
// least one player can't survive a recube) DOWN to the current cube level,
// computing cash points recursively.
//
// At dead cube levels: CP = (DTL - DP) / (DTL - DTW)
//   with the doubler's own prime for BOTH DTW and DTL.
// At live cube levels: CP = 1 - opp_CP_above * (DP - DTW) / (rRDP - DTW)
//   where rRDP = MWC after losing at the doubled value (recube-drop payoff).
//
// Parameters use 1-indexed away values. Gammon ratios are from the player's
// perspective: rG0/rBG0 for player's wins, rG1/rBG1 for player's losses.
void get_match_points(
    int away1, int away2, int cv, bool craw,
    float rG0, float rBG0,        // gammon/bg ratios for player's wins
    float rG1, float rBG1,        // gammon/bg ratios for player's losses
    float& player_cp,             // out: player's cash point
    float& opp_cp)                // out: opponent's cash point
{
    validate_met_away_scores(away1, away2);
    // 0-indexed away values (GNUbg convention)
    int idx0 = away1 - 1;
    int idx1 = away2 - 1;

    // Gammon/bg ratios per side (0=player, 1=opponent)
    float arG[2] = {rG0, rG1};
    float arBG[2] = {rBG0, rBG1};

    // Find the dead cube level: highest cube where both can survive a recube
    int nDead = cv;
    int nMax = 0;
    while (idx0 >= 2 * nDead && idx1 >= 2 * nDead) {
        nMax++;
        nDead *= 2;
    }

    // Cash points at each level (k=0: player, k=1: opponent)
    constexpr int MAX_LEVELS = 16;
    float arCPLive[2][MAX_LEVELS] = {};

    for (int nCubeValue = nDead, n = nMax; nCubeValue >= cv;
         nCubeValue >>= 1, n--) {
        // Auto-redouble primes at this cube level (0-indexed comparisons)
        int prime[2];
        prime[0] = (idx0 < 2*nCubeValue && idx1 >= 2*nCubeValue)
                   ? 2*nCubeValue : nCubeValue;
        prime[1] = (idx1 < 2*nCubeValue && idx0 >= 2*nCubeValue)
                   ? 2*nCubeValue : nCubeValue;

        for (int k = 0; k < 2; k++) {
            // k=0: player doubles, k=1: opponent doubles
            int away_d = (k == 0) ? away1 : away2;
            int away_t = (k == 0) ? away2 : away1;
            int id = (k == 0) ? idx0 : idx1;
            int it = (k == 0) ? idx1 : idx0;

            float gW = arG[k], bgW = arBG[k];       // doubler's wins
            float gL = arG[1-k], bgL = arBG[1-k];   // doubler's losses

            if (id < 2*nCubeValue || it < 2*nCubeValue) {
                // Dead cube: at least one player can't survive a recube.
                // Use doubler's own prime for BOTH DTW and DTL.
                // The effective cube after DT (with possible auto-redouble) is
                // 2*prime[k], so outcomes are at 2*p, 4*p, 6*p (not p, 2p, 3p).
                int p = prime[k];
                int dp = 2 * p;  // effective cube level after DT + auto-redouble

                float rDP = get_met_after(away_d, away_t, nCubeValue, true, craw);

                float rDTW = (1.0f - gW - bgW)
                                * get_met_after(away_d, away_t, dp, true, craw)
                           + gW * get_met_after(away_d, away_t, 2*dp, true, craw)
                           + bgW * get_met_after(away_d, away_t, 3*dp, true, craw);

                float rDTL = (1.0f - gL - bgL)
                                * get_met_after(away_d, away_t, dp, false, craw)
                           + gL * get_met_after(away_d, away_t, 2*dp, false, craw)
                           + bgL * get_met_after(away_d, away_t, 3*dp, false, craw);

                float denom = rDTL - rDTW;
                if (std::abs(denom) < 1e-10f)
                    arCPLive[k][n] = 0.5f;
                else
                    arCPLive[k][n] = std::clamp(
                        (rDTL - rDP) / denom, 0.0f, 1.0f);
            } else {
                // Live cube: both players can survive a recube.
                // Use standard doubled value, recursive formula.
                int dcv = 2 * nCubeValue;

                float rDP = get_met_after(away_d, away_t,
                                          nCubeValue, true, craw);
                float rRDP = get_met_after(away_d, away_t,
                                           dcv, false, craw);

                float rDTW = (1.0f - gW - bgW)
                                * get_met_after(away_d, away_t, dcv, true, craw)
                           + gW * get_met_after(away_d, away_t, 2*dcv, true, craw)
                           + bgW * get_met_after(away_d, away_t, 3*dcv, true, craw);

                float denom = rRDP - rDTW;
                if (std::abs(denom) < 1e-10f)
                    arCPLive[k][n] = 0.5f;
                else
                    arCPLive[k][n] = std::clamp(
                        1.0f - arCPLive[1-k][n+1] * (rDP - rDTW) / denom,
                        0.0f, 1.0f);
            }
        }
    }

    player_cp = arCPLive[0][0];
    opp_cp = arCPLive[1][0];
}

// Compute cubeful MWC for a live match cube (centered / owned / unavailable).
// Computes gammon/backgammon ratios, MET anchors, and GNUbg recursive cash
// points once, then applies piecewise-linear interpolation by cube ownership.
static float cl2cf_match_live(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    float p_win = probs[0];

    // Gammon/backgammon ratios
    float rG0, rBG0, rG1, rBG1;
    if (p_win > 1e-7f) {
        rG0 = (probs[1] - probs[2]) / p_win;
        rBG0 = probs[2] / p_win;
    } else {
        rG0 = 0.0f;
        rBG0 = 0.0f;
    }
    if (p_win < 1.0f - 1e-7f) {
        rG1 = (probs[3] - probs[4]) / (1.0f - p_win);
        rBG1 = probs[4] / (1.0f - p_win);
    } else {
        rG1 = 0.0f;
        rBG1 = 0.0f;
    }

    // Dead cube MWC (uses MET-weighted outcome values, not linear approximation)
    float mwc_dead = cubeless_mwc(probs, away1, away2, cv, craw);

    // MET lookups at current cube value (for no-double outcomes)
    float mwc_win_s  = get_met_after(away1, away2, cv, true, craw);
    float mwc_win_g  = get_met_after(away1, away2, 2*cv, true, craw);
    float mwc_win_b  = get_met_after(away1, away2, 3*cv, true, craw);
    float mwc_lose_s = get_met_after(away1, away2, cv, false, craw);
    float mwc_lose_g = get_met_after(away1, away2, 2*cv, false, craw);
    float mwc_lose_b = get_met_after(away1, away2, 3*cv, false, craw);

    // D/P MWC (player cashes cv points) and opponent's D/P MWC (player loses cv)
    float mwc_cash = mwc_win_s;
    float mwc_opp_cash = mwc_lose_s;

    // Weighted MWCs for all win/loss types
    float mwc_win_all = (1.0f - rG0 - rBG0) * mwc_win_s
                      + rG0 * mwc_win_g + rBG0 * mwc_win_b;
    float mwc_lose_all = (1.0f - rG1 - rBG1) * mwc_lose_s
                       + rG1 * mwc_lose_g + rBG1 * mwc_lose_b;

    // Compute live-cube cash points using recursive algorithm (GNUbg GetPoints).
    float player_cp, opp_cp;
    get_match_points(away1, away2, cv, craw, rG0, rBG0, rG1, rBG1,
                     player_cp, opp_cp);

    float opp_tg = 1.0f - opp_cp;   // Below this P(win), opponent is too good
    float player_tg = player_cp;    // Above this P(win), player is too good

    float mwc_live = 0.0f;
    switch (cube.owner) {
        case CubeOwner::CENTERED: {
            if (p_win <= opp_tg) {
                if (opp_tg > 1e-10f) {
                    mwc_live = mwc_lose_all + (mwc_opp_cash - mwc_lose_all) * p_win / opp_tg;
                } else {
                    mwc_live = mwc_lose_all;
                }
            } else if (p_win < player_tg) {
                float range = player_tg - opp_tg;
                if (range > 1e-10f) {
                    mwc_live = mwc_opp_cash + (mwc_cash - mwc_opp_cash) * (p_win - opp_tg) / range;
                } else {
                    mwc_live = mwc_opp_cash;
                }
            } else {
                float range = 1.0f - player_tg;
                if (range > 1e-10f) {
                    mwc_live = mwc_cash + (mwc_win_all - mwc_cash) * (p_win - player_tg) / range;
                } else {
                    mwc_live = mwc_win_all;
                }
            }
            break;
        }
        case CubeOwner::PLAYER: {
            if (p_win <= player_tg) {
                if (player_tg > 1e-10f) {
                    mwc_live = mwc_lose_all + (mwc_cash - mwc_lose_all) * p_win / player_tg;
                } else {
                    mwc_live = mwc_lose_all;
                }
            } else {
                float range = 1.0f - player_tg;
                if (range > 1e-10f) {
                    mwc_live = mwc_cash + (mwc_win_all - mwc_cash) * (p_win - player_tg) / range;
                } else {
                    mwc_live = mwc_win_all;
                }
            }
            break;
        }
        case CubeOwner::OPPONENT: {
            if (p_win <= opp_tg) {
                if (opp_tg > 1e-10f) {
                    mwc_live = mwc_lose_all + (mwc_opp_cash - mwc_lose_all) * p_win / opp_tg;
                } else {
                    mwc_live = mwc_lose_all;
                }
            } else {
                float range = 1.0f - opp_tg;
                if (range > 1e-10f) {
                    mwc_live = mwc_opp_cash + (mwc_win_all - mwc_opp_cash) * (p_win - opp_tg) / range;
                } else {
                    mwc_live = mwc_win_all;
                }
            }
            break;
        }
    }

    // Janowski interpolation in MWC space
    return mwc_dead * (1.0f - cube_x) + mwc_live * cube_x;
}

// Post-Crawford automatic double. With one side 1-away after the Crawford
// game, the trailer doubles at the first opportunity: the leader cannot gain
// from the cube, and the trailer can only win on their own turns, all of which
// come after that next pre-roll opportunity. So a cube the trailer may still
// turn (centered or trailer-owned, and not yet worth the trailer's match) is
// valued as doubled once rather than through Janowski: the dead-cube anchor at
// the current value scores a trailer's gammon at 3-away/1-away as 2 points
// instead of the match, which put No Double up to 0.18 equity below
// Double/Take at 2-ply although rollouts (GNUbg's and our own) show them equal.
//
// The trailer may also never double (play on at the current cube), and after
// the double the leader takes (2x cube, then dead -- only the 1-away leader
// could turn it) or passes. Returns true and sets `mwc` (player's view) when
// the rule applies.
static bool post_crawford_auto_double_mwc(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float& mwc)
{
    const MatchInfo& m = cube.match;
    if (cube.is_money() || m.is_crawford) return false;
    const bool player_trails = (m.away2 == 1 && m.away1 > 1);
    const bool opponent_trails = (m.away1 == 1 && m.away2 > 1);
    if (!player_trails && !opponent_trails) return false;

    const int cv = cube.cube_value;
    const int trailer_away = player_trails ? m.away1 : m.away2;
    const CubeOwner trailer_owns =
        player_trails ? CubeOwner::PLAYER : CubeOwner::OPPONENT;
    if (cube.owner != CubeOwner::CENTERED && cube.owner != trailer_owns)
        return false;
    if (trailer_away <= cv) return false;  // any trailer win already takes the match
    if (cube.max_cube_value > 0 && 2 * cv > cube.max_cube_value) return false;

    const float play_on = cubeless_mwc(probs, m.away1, m.away2, cv, false);
    const float take = cubeless_mwc(probs, m.away1, m.away2, 2 * cv, false);
    // The trailer cashes cv points when the leader passes.
    const float pass = get_met_after(m.away1, m.away2, cv, player_trails, false);
    mwc = player_trails ? std::max(play_on, std::min(take, pass))
                        : std::min(play_on, std::max(take, pass));
    return true;
}

float cl2cf_match(const std::array<float, NUM_OUTPUTS>& probs,
                  const CubeInfo& cube, float cube_x)
{
    // Dead cube: return cubeless MWC directly
    if (is_dead_cube(cube)) {
        return cubeless_mwc(probs, cube.match.away1, cube.match.away2,
                            cube.cube_value, cube.match.is_crawford);
    }

    float auto_double_mwc;
    if (post_crawford_auto_double_mwc(probs, cube, auto_double_mwc)) {
        return auto_double_mwc;
    }

    return cl2cf_match_live(probs, cube, cube_x);
}

float cl2cf_raw(const std::array<float, NUM_OUTPUTS>& probs,
                const CubeInfo& cube, float cube_x)
{
    if (cube_is_dead(cube)) {
        return cubeless_value(probs, cube);
    }
    if (cube.is_money()) {
        return cl2cf_money(probs, cube.owner, cube_x, cube.jacoby_active());
    }
    return cl2cf_match(probs, cube, cube_x);
}

float cl2cf(const std::array<float, NUM_OUTPUTS>& probs,
            const CubeInfo& cube, float cube_x)
{
    const float raw = cl2cf_raw(probs, cube, cube_x);
    if (cube.is_money()) {
        return raw;
    }
    // Match play: cl2cf_raw returns MWC, convert to equity
    return mwc2eq(raw, cube.match.away1, cube.match.away2,
                  cube.cube_value, cube.match.is_crawford);
}

float cube_efficiency(
    const std::array<float, NUM_OUTPUTS>& /*probs*/,
    bool is_race_pos,
    int player_pips,
    int opponent_pips)
{
    // Current implementation ignores probs; reserved for a future ML formula
    // trained on (probs, is_race, pips) -> implied Janowski cube life index.
    if (!is_race_pos) {
        return 0.68f;  // Contact/crashed
    }
    // Race: linear in average pip count (both players), clamped to [0.6, 0.7].
    // Using the average instead of just the roller's pips makes the Janowski
    // cubeful conversion perspective-independent, eliminating a systematic
    // bias at odd ply levels in N-ply match play evaluation.
    float avg_pips = static_cast<float>(player_pips + opponent_pips) * 0.5f;
    float x = 0.55f + 0.00125f * avg_pips;
    return std::clamp(x, 0.6f, 0.7f);
}

float contact_cube_x(const CubeInfo& cube, int player_pips, int opponent_pips)
{
    if (cube.contact_x_pip_slope == 0.0f) return cube.contact_x;
    float avg_pips = static_cast<float>(player_pips + opponent_pips) * 0.5f;
    float x = cube.contact_x
              + cube.contact_x_pip_slope * (avg_pips - CONTACT_X_PIP_PIVOT);
    return std::clamp(x, 0.5f, 1.0f);
}

float resolve_cube_x(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    bool is_race_pos,
    int player_pips,
    int opponent_pips)
{
    if (cube.cube_x_override >= 0.0f) return cube.cube_x_override;
    if (!is_race_pos && cube.contact_x >= 0.0f)
        return contact_cube_x(cube, player_pips, opponent_pips);
    return cube_efficiency(probs, is_race_pos, player_pips, opponent_pips);
}

CubeDecision decision_from_raw_equities(
    const CubeInfo& cube,
    float nd_raw,
    float dt_raw,
    bool dt_is_basis_scaled)
{
    const bool is_money = cube.is_money();
    CubeDecision result{};

    if (is_money) {
        result.equity_nd = nd_raw;
        const float actual_dt = dt_is_basis_scaled ? dt_raw : (2.0f * dt_raw);
        result.equity_dp = 1.0f;

        if (can_beaver_after_double(cube) && actual_dt < 0.0f) {
            result.equity_dt = 2.0f * actual_dt;  // DB equity
            result.is_beaver = true;
        } else {
            result.equity_dt = actual_dt;
            result.is_beaver = false;
        }
    } else {
        const int away1 = cube.match.away1;
        const int away2 = cube.match.away2;
        const int cv = cube.cube_value;
        const bool craw = cube.match.is_crawford;
        const float dp_m = dp_mwc(away1, away2, cv, craw);
        result.equity_nd = mwc2eq(nd_raw, away1, away2, cv, craw);
        result.equity_dt = mwc2eq(dt_raw, away1, away2, cv, craw);
        result.equity_dp = mwc2eq(dp_m, away1, away2, cv, craw);
        result.is_beaver = false;
    }

    const bool player_can_double = can_double(cube);
    if (!player_can_double) {
        result.should_double = false;
        result.should_take = true;
        result.optimal_equity = result.equity_nd;
        return result;
    }

    if (is_money) {
        const float best_double = std::min(result.equity_dt, result.equity_dp);
        result.should_double = (best_double > result.equity_nd);
        result.should_take = (result.equity_dt <= result.equity_dp);
        result.optimal_equity = result.should_double ? best_double : result.equity_nd;
    } else {
        const int away1 = cube.match.away1;
        const int away2 = cube.match.away2;
        const int cv = cube.cube_value;
        const bool craw = cube.match.is_crawford;
        const float dp_m = dp_mwc(away1, away2, cv, craw);
        const float best_mwc = std::min(dt_raw, dp_m);
        // Prefer doubling on a post-Crawford tie. The leader can also pass,
        // so playing on may be strictly better (e.g. a match-winning gammon).
        const bool trailer = !craw && away1 > 1 && away2 == 1;
        constexpr float tie_tolerance = std::numeric_limits<float>::epsilon();
        result.should_take = (dt_raw <= dp_m);
        result.should_double = best_mwc > nd_raw
            || (trailer && nd_raw - best_mwc <= tie_tolerance);
        result.optimal_equity = result.should_double
            ? std::min(result.equity_dt, result.equity_dp)
            : result.equity_nd;
    }

    return result;
}

float mwc_se_to_eq_se(float mwc_se, float mean_mwc,
                      int away1, int away2, int cube_value,
                      bool is_crawford)
{
    const float eps = std::max(1e-3f, mwc_se);
    const float eq_plus = mwc2eq(mean_mwc + eps, away1, away2, cube_value, is_crawford);
    const float eq_minus = mwc2eq(mean_mwc - eps, away1, away2, cube_value, is_crawford);
    const float slope = (eq_plus - eq_minus) / (2.0f * eps);
    return std::abs(slope) * mwc_se;
}

std::vector<int> filter_candidates_cubeful_1ply(
    const Strategy& filter_strategy,
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    const CubeInfo* cubes,
    int n_cubes,
    float cube_x,
    int max_moves,
    float threshold,
    std::vector<std::array<float, NUM_OUTPUTS>>* probs_out)
{
    const int n = static_cast<int>(candidates.size());
    std::vector<int> survivors;
    if (n == 0 || n_cubes <= 0) {
        if (probs_out) probs_out->clear();
        return survivors;
    }

    std::vector<std::array<float, NUM_OUTPUTS>> local_probs;
    auto& probs_per = probs_out ? *probs_out : local_probs;
    probs_per.resize(n);

    filter_strategy.batch_evaluate_candidates_equity_probs(
        candidates, pre_move_board, nullptr, probs_per.data());
    for (int i = 0; i < n; ++i) {
        clamp_probs_to_board(probs_per[i], candidates[i]);
    }

    std::vector<bool> in_set(n, false);
    survivors.reserve(std::min(n, max_moves * n_cubes));

    std::vector<std::pair<float, int>> cube_eqs;
    cube_eqs.reserve(n);
    for (int c = 0; c < n_cubes; ++c) {
        cube_eqs.clear();
        for (int i = 0; i < n; ++i) {
            float cf = cl2cf(probs_per[i], cubes[c], cube_x);
            cube_eqs.emplace_back(cf, i);
        }
        // Break exact ties on money cubeless equity so survival is
        // deterministic even when every cubeful key is bit-identical.
        // Uses exact equality (strict weak ordering for std::sort).
        std::sort(cube_eqs.begin(), cube_eqs.end(),
                  [&](const std::pair<float, int>& a,
                      const std::pair<float, int>& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return cubeless_equity(probs_per[a.second])
                           > cubeless_equity(probs_per[b.second]);
                  });
        float best_cf = cube_eqs[0].first;
        int keep = 0;
        for (const auto& p : cube_eqs) {
            if (keep >= max_moves) break;
            if (best_cf - p.first > threshold) break;
            if (!in_set[p.second]) {
                in_set[p.second] = true;
                survivors.push_back(p.second);
            }
            ++keep;
        }
    }

    return survivors;
}

CubeDecision cube_decision_1ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    if (cube_is_dead(cube)) {
        CubeDecision result{};
        if (cube.is_money()) {
            const float cl_eq = cubeless_equity(probs);
            result.equity_nd = cl_eq;
            result.equity_dt = cl_eq;
            result.equity_dp = 1.0f;
            result.optimal_equity = cl_eq;
        } else {
            const int away1 = cube.match.away1;
            const int away2 = cube.match.away2;
            const int cv = cube.cube_value;
            const bool craw = cube.match.is_crawford;
            const float mwc = cubeless_mwc(probs, away1, away2, cv, craw);
            const float eq = mwc2eq(mwc, away1, away2, cv, craw);
            result.equity_nd = eq;
            result.equity_dt = eq;
            result.equity_dp = mwc2eq(dp_mwc(away1, away2, cv, craw), away1, away2, cv, craw);
            result.optimal_equity = eq;
        }
        result.should_double = false;
        result.should_take = true;
        result.is_beaver = false;
        return result;
    }

    const float nd_raw = cl2cf_raw(probs, cube, cube_x);
    CubeInfo dt_cube = cube;
    dt_cube.cube_value = 2 * cube.cube_value;
    dt_cube.owner = CubeOwner::OPPONENT;
    const float dt_raw = cl2cf_raw(probs, dt_cube, cube_x);
    return decision_from_raw_equities(cube, nd_raw, dt_raw, /*dt_is_basis_scaled=*/false);
}

CubeDecision cube_decision_1ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    const Board& board,
    bool is_race_pos)
{
    auto [pp, op] = pip_counts(board);
    return cube_decision_1ply(probs, cube,
                              resolve_cube_x(probs, cube, is_race_pos, pp, op));
}

// ---------------------------------------------------------------------------
// N-ply cubeful evaluation
// ---------------------------------------------------------------------------

float cubeful_equity_nply(
    const Board& board,
    CubeOwner owner,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    CubeInfo cube;
    cube.cube_value = 1;
    cube.owner = owner;
    float out = 0.0f;
    cubeful_equity_nply_multi(board, &cube, 1, strategy, n_plies, &out,
                              filter, n_threads, move_filter);
    return out;
}

// Match play overload: returns cubeful equity (NOT MWC).
float cubeful_equity_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    float out = 0.0f;
    cubeful_equity_nply_multi(board, &cube, 1, strategy, n_plies, &out,
                              filter, n_threads, move_filter);
    return out;
}

// Returns post-roll probabilities from the N-ply CUBE-AWARE tree, in the
// player-on-roll's perspective at `board`. Same traversal as
// cubeful_equity_nply (the probs are accumulated through the tree), so the
// returned probs reflect cube/match-aware interior move picks throughout.
//
// Used by `post_move_analytics` when the inner analyzer is N-ply and the
// caller wants probs that depend on cube state (e.g. for a 1-away match score).
std::array<float, NUM_OUTPUTS> cubeful_probs_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    float out = 0.0f;
    std::array<float, NUM_OUTPUTS> probs{};
    cubeful_equity_nply_multi(board, &cube, 1, strategy, n_plies, &out,
                              filter, n_threads, move_filter, false, &probs);
    return probs;
}

// Returns BOTH the cube-aware probs and the cubeful equity from a single
// traversal of the cubeful tree. Avoids a second tree walk when callers want
// both (per-candidate checker play analytics).
CubefulProbsAndEquity cubeful_probs_and_equity_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter,
    const Board* root_board,
    const std::function<void()>& check_budget)
{
    float out = 0.0f;
    std::array<float, NUM_OUTPUTS> probs{};
    cubeful_equity_nply_multi(board, &cube, 1, strategy, n_plies, &out,
                              filter, n_threads, move_filter, false, &probs,
                              /*deep_prefilter=*/false, root_board, {}, check_budget);
    return {probs, out};
}


CubeDecision cube_decision_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter, const std::function<void()>& check_budget)
{
    // The multi entry with n=1 applies the identical 1-ply path, money/match
    // unpack and post-Crawford auto-double logic.
    CubeDecision out{};
    cube_decision_nply_multi(board, &cube, 1, strategy, n_plies, &out,
                             filter, n_threads, move_filter, false, check_budget);
    return out;
}

} // namespace bgbot
