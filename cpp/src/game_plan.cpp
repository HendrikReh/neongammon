// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// Game-plan classification and 17-NN pair routing helpers (declared in encoding.h).
#include "bgbot/encoding.h"
#include "bgbot/board.h"

namespace bgbot {

const char* game_plan_name(GamePlan gp) {
    switch (gp) {
        case GamePlan::PURERACE:  return "purerace";
        case GamePlan::RACING:    return "racing";
        case GamePlan::ATTACKING: return "attacking";
        case GamePlan::PRIMING:   return "priming";
        case GamePlan::ANCHORING: return "anchoring";
        default:                  return "unknown";
    }
}

GamePlan classify_game_plan(const Board& board) {
    if (is_race(board)) return GamePlan::PURERACE;
    auto [player_pips, opponent_pips] = pip_counts(board);
    return classify_contact_game_plan(board, player_pips, opponent_pips);
}

GamePlan classify_contact_game_plan(const Board& board, int player_pips, int opponent_pips) {
    // Count opponent back checkers (opponent checkers in player's home board + bar)
    int n_opp_back = board[0];  // opponent on bar
    for (int i = 1; i <= 6; ++i) {
        if (board[i] < 0) n_opp_back -= board[i];
    }

    if (n_opp_back == 0) {
        // No opponent back checkers to prime or blitz
        if (player_pips < opponent_pips) {
            return GamePlan::RACING;
        }

        // Does player have at least two back checkers?
        int n_player_back = board[25];  // bar counts as back too
        for (int i = 19; i <= 24; ++i) {
            if (board[i] > 0) n_player_back += board[i];
        }

        if (n_player_back > 1) {
            return GamePlan::ANCHORING;
        }

        // Check if opponent has more borne off and no offside checkers
        int n_player_borne = player_borne_off(board);
        int n_opp_borne = opponent_borne_off(board);
        int n_opp_offside = board[0];
        for (int i = 1; i <= 13; ++i) {
            if (board[i] < 0) n_opp_offside -= board[i];
        }

        if (n_opp_borne > n_player_borne && n_opp_offside == 0) {
            return GamePlan::ATTACKING;
        }
        return GamePlan::RACING;
    }

    // Opponent has back checkers
    if (player_pips < opponent_pips) {
        // Player is ahead in the race
        int n_out = 0;
        for (int i = 7; i <= 12; ++i) {
            if (board[i] > 0) n_out += board[i];
        }
        int n_rest = board[25];
        for (int i = 13; i <= 24; ++i) {
            if (board[i] > 0) n_rest += board[i];
        }

        if (n_rest == 0 && n_out < 4) {
            return GamePlan::RACING;
        }

        // Count player checkers in their zone (points 1-12)
        int n_zone = 0;
        for (int i = 1; i <= 12; ++i) {
            if (board[i] > 0) n_zone += board[i];
        }

        if (n_zone >= 8) {
            // Opponent holding at least two anchors (back game)? -> Race
            int n_opp_anchors = 0;
            for (int i = 1; i <= 6; ++i) {
                if (board[i] < -1) n_opp_anchors++;
            }
            if (n_opp_anchors >= 2) {
                return GamePlan::RACING;
            }

            // Opponent has blot on the bar? -> Attack
            if (board[0] > 0) {
                return GamePlan::ATTACKING;
            }

            // Opponent has blot in player's home board?
            int opp_blot_pnt = -1;
            for (int i = 1; i <= 6; ++i) {
                if (board[i] == -1) {
                    opp_blot_pnt = i;
                    break;
                }
            }

            if (opp_blot_pnt >= 0) {
                bool player_checker_above = false;
                for (int i = opp_blot_pnt + 1; i <= 6; ++i) {
                    if (board[i] > 0) {
                        player_checker_above = true;
                        break;
                    }
                }
                return player_checker_above ? GamePlan::ATTACKING : GamePlan::RACING;
            }

            // Check for priming possibility
            int n_player_back = board[25];
            for (int i = 19; i <= 24; ++i) {
                if (board[i] > 0) n_player_back += board[i];
            }

            if (n_player_back < 2) {
                // Prime if at least a 3-prime in front of opponent's highest checker
                int opp_max = board[0] > 0 ? 0 : -1;
                for (int i = 1; i <= 5; ++i) {
                    if (board[i] < 0) opp_max = i;
                }

                if (opp_max != -1) {
                    int max_prime = 0, n_prime = 0;
                    for (int i = opp_max + 1; i <= 11; ++i) {
                        if (board[i] > 1) {
                            n_prime++;
                            if (n_prime > max_prime) max_prime = n_prime;
                        } else {
                            n_prime = 0;
                        }
                    }
                    if (max_prime >= 3) {
                        return GamePlan::PRIMING;
                    }
                }
                return GamePlan::RACING;
            }

            return GamePlan::ATTACKING;
        }
        return GamePlan::RACING;
    }

    // Player is behind in the race (player_pips >= opponent_pips)
    int n_player_borne = player_borne_off(board);
    int n_opp_borne = opponent_borne_off(board);
    int n_player_offside = board[25];
    for (int i = 13; i <= 24; ++i) {
        if (board[i] > 0) n_player_offside += board[i];
    }

    if (n_player_borne > n_opp_borne && n_player_offside == 0) {
        return GamePlan::RACING;
    }

    // Check for deep home board point with gap
    int n_home_anchors = 0;
    for (int i = 1; i <= 6; ++i) {
        if (board[i] > 1) n_home_anchors++;
    }

    bool has_deep_point = false;
    for (int i = 1; i <= 3; ++i) {
        if (board[i] >= 2) { has_deep_point = true; break; }
    }

    bool has_gap = false;
    for (int i = 4; i <= 6; ++i) {
        if (board[i] <= 0) { has_gap = true; break; }
    }

    const double pip_diff_ratio = (player_pips + opponent_pips > 0)
        ? 2.0 * (player_pips - opponent_pips) / (player_pips + opponent_pips)
        : 0.0;

    auto count_player_back = [&]() {
        int n_back = board[25];
        for (int i = 19; i <= 24; ++i) {
            if (board[i] > 0) n_back += board[i];
        }
        return n_back;
    };

    if (n_home_anchors < 4 && has_deep_point && has_gap) {
        if (pip_diff_ratio <= 0.1) {
            return GamePlan::ATTACKING;
        }
        return (count_player_back() < 2) ? GamePlan::RACING : GamePlan::ANCHORING;
    }

    // Check for opponent high anchor blocking priming
    bool opp_has_high_anchor = false;
    for (int i = 4; i <= 6; ++i) {
        if (board[i] <= -2) { opp_has_high_anchor = true; break; }
    }

    auto count_opp_blots = [&]() {
        int n_blots = board[0] > 0 ? 1 : 0;
        for (int i = 1; i <= 12; ++i) {
            if (board[i] == -1) n_blots++;
        }
        return n_blots;
    };

    if (opp_has_high_anchor) {
        if (count_player_back() >= 2) {
            return GamePlan::ANCHORING;
        }
        if (count_opp_blots() > 0 && pip_diff_ratio <= 0.1) {
            return GamePlan::ATTACKING;
        }
        return GamePlan::RACING;
    }

    // No opponent high anchor - check for priming position
    int n_primers = 0;
    for (int i = 4; i <= 12; ++i) {
        if (board[i] >= 1) n_primers++;
    }
    int n_primers_bar = 0;
    for (int i = 5; i <= 8; ++i) {
        if (board[i] >= 1) n_primers_bar++;
    }

    if (n_primers >= 4 || n_primers_bar >= 3) {
        return GamePlan::PRIMING;
    }

    // Attack, anchor, or prime fallback
    if (count_opp_blots() > 0 && pip_diff_ratio <= 0.1) {
        return GamePlan::ATTACKING;
    }

    return (count_player_back() >= 2) ? GamePlan::ANCHORING : GamePlan::PRIMING;
}

int game_plan_pair_index(GamePlan player, GamePlan opponent) {
    int p = static_cast<int>(player);
    int o = static_cast<int>(opponent);
    return (p - 1) * 4 + (o - 1);
}

static const std::array<const char*, NUM_PAIR_NNS> PAIR_NAMES = {
    "purerace",
    "race_race",  "race_att",  "race_prim",  "race_anch",
    "att_race",   "att_att",   "att_prim",   "att_anch",
    "prim_race",  "prim_att",  "prim_prim",  "prim_anch",
    "anch_race",  "anch_att",  "anch_prim",  "anch_anch",
};

const char* game_plan_pair_name(int pair_idx) {
    if (pair_idx < 0 || pair_idx >= NUM_PAIR_NNS) return "unknown";
    return PAIR_NAMES[pair_idx];
}

const std::array<const char*, NUM_PAIR_NNS>& game_plan_pair_names() {
    return PAIR_NAMES;
}

} // namespace bgbot
