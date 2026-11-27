// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// BearoffDB: one-sided 15-checker 6-point bearoff database generation,
// combinatorial indexing, persistence, and exact two-sided probability lookup.
#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

namespace bgbot {

namespace {

struct BearoffDiceRoll { int d1, d2, weight; };
constexpr BearoffDiceRoll kBearoffRolls[21] = {
    {1,1,1}, {2,2,1}, {3,3,1}, {4,4,1}, {5,5,1}, {6,6,1},
    {1,2,2}, {1,3,2}, {1,4,2}, {1,5,2}, {1,6,2},
    {2,3,2}, {2,4,2}, {2,5,2}, {2,6,2},
    {3,4,2}, {3,5,2}, {3,6,2},
    {4,5,2}, {4,6,2},
    {5,6,2}
};

float one_roll_gammon_loss(const Board& board) {
    int failing_rolls = 0;
    std::vector<Board> moves;
    for (const auto& roll : kBearoffRolls) {
        possible_boards_unsorted(board, roll.d1, roll.d2, moves);
        const bool saves = std::any_of(moves.begin(), moves.end(), [](const Board& move) {
            return player_borne_off(move) > 0;
        });
        if (!saves) failing_rolls += roll.weight;
    }
    return failing_rolls / 36.0f;
}

} // namespace

// --- Binomial coefficient table ---

unsigned int BearoffDB::binom_table_[MAX_N][MAX_K] = {};
bool BearoffDB::binom_initialized_ = false;

void BearoffDB::init_binom() {
    for (int n = 0; n < MAX_N; ++n) {
        binom_table_[n][0] = 1;
        for (int k = 1; k < MAX_K && k <= n; ++k) {
            binom_table_[n][k] = binom_table_[n-1][k-1] + binom_table_[n-1][k];
        }
    }
    binom_initialized_ = true;
}

// --- Position indexing ---
//
// Uses the combinatorial number system (stars-and-bars). Choosing POINTS=6
// separator positions p[0] > p[1] > ... > p[5] >= 0 out of CHECKERS+POINTS=21:
//   p[0] = sum(checkers[0..5]) + 5
//   p[i] = p[i-1] - checkers[i-1] - 1
//   Rank = C(p[0], 6) + C(p[1], 5) + ... + C(p[5], 1).

unsigned int BearoffDB::position_index(const int checkers[POINTS]) {
    if (!binom_initialized_) init_binom();

    unsigned int index = 0;
    int pos = POINTS - 1;
    for (int i = 0; i < POINTS; ++i) {
        pos += checkers[i];
    }

    int k = POINTS;
    for (int i = 0; i < POINTS; ++i) {
        index += binom(pos, k);
        pos -= checkers[i] + 1;
        --k;
    }

    return index;
}

void BearoffDB::index_to_position(unsigned int index, int checkers[POINTS]) {
    if (!binom_initialized_) init_binom();

    int positions[POINTS];
    unsigned int remaining = index;

    for (int k = POINTS; k >= 1; --k) {
        int p = k - 1;
        while (binom(p + 1, k) <= remaining) {
            ++p;
        }
        positions[POINTS - k] = p;
        remaining -= binom(p, k);
    }

    for (int i = 0; i < POINTS - 1; ++i) {
        checkers[i] = positions[i] - positions[i + 1] - 1;
    }
    checkers[POINTS - 1] = positions[POINTS - 1];
}

unsigned int BearoffDB::board_to_player_index(const Board& board) {
    int checkers[POINTS];
    for (int i = 0; i < POINTS; ++i) {
        checkers[i] = std::max(0, board[i + 1]);  // points 1-6
    }
    return position_index(checkers);
}

unsigned int BearoffDB::board_to_opponent_index(const Board& board) {
    int checkers[POINTS];
    for (int i = 0; i < POINTS; ++i) {
        checkers[i] = std::max(0, -board[24 - i]);  // board[24]=opp's 1, ..., board[19]=opp's 6
    }
    return position_index(checkers);
}

int BearoffDB::checkers_on_board(unsigned int index) const {
    int checkers[POINTS];
    index_to_position(index, checkers);
    int sum = 0;
    for (int i = 0; i < POINTS; ++i) sum += checkers[i];
    return sum;
}

// --- is_bearoff ---

bool BearoffDB::is_bearoff(const Board& board) const {
    if (!loaded_) return false;
    if (board[0] != 0 || board[25] != 0) return false;

    for (int i = 7; i <= 24; ++i) {
        if (board[i] > 0) return false;
    }
    for (int i = 1; i <= 18; ++i) {
        if (board[i] < 0) return false;
    }
    return true;
}

// --- Two-sided lookups ---

std::array<float, NUM_OUTPUTS> BearoffDB::lookup_probs(
    const Board& board, bool post_move) const
{
    unsigned int p_idx = board_to_player_index(board);
    unsigned int o_idx = board_to_opponent_index(board);

    const auto& p_dist = distributions_[p_idx];
    const auto& o_dist = distributions_[o_idx];
    const auto* player_gammon = get_gammon_distribution(p_idx);
    const auto* opp_gammon = get_gammon_distribution(o_idx);

    // With all 15 checkers left against a guaranteed next-roll finish, the
    // side on roll can only play to save the gammon. The stored gammon
    // distribution follows the fastest finish policy. If it already saves
    // on all rolls, there is no need to enumerate legal plays.
    if (!post_move && o_dist[1] == 65535 && player_gammon && (*player_gammon)[1] != 0) {
        return {0.0f, 0.0f, 0.0f, one_roll_gammon_loss(board), 0.0f};
    }
    if (post_move && p_dist[1] == 65535 && opp_gammon && (*opp_gammon)[1] != 0) {
        return {1.0f, one_roll_gammon_loss(flip(board)), 0.0f, 0.0f, 0.0f};
    }

    constexpr float SCALE = 1.0f / 65535.0f;

    float opp_cdf[MAX_ROLLS];
    float player_cdf[MAX_ROLLS];
    {
        float cum_o = 0.0f, cum_p = 0.0f;
        for (int k = 0; k < MAX_ROLLS; ++k) {
            cum_o += o_dist[k] * SCALE;
            opp_cdf[k] = cum_o;
            cum_p += p_dist[k] * SCALE;
            player_cdf[k] = cum_p;
        }
    }

    float p_win = p_dist[0] * SCALE;
    if (!post_move) {
        for (int i = 1; i < MAX_ROLLS; ++i) {
            float p_player_i = p_dist[i] * SCALE;
            if (p_player_i == 0.0f) continue;
            p_win += p_player_i * (1.0f - opp_cdf[i - 1]);
        }
    } else {
        for (int i = 1; i < MAX_ROLLS; ++i) {
            float p_player_i = p_dist[i] * SCALE;
            if (p_player_i == 0.0f) continue;
            p_win += p_player_i * (1.0f - opp_cdf[i]);
        }
    }

    float p_gammon_win = 0.0f;
    float p_gammon_loss = 0.0f;

    if (!post_move) {
        if (opp_gammon) {
            p_gammon_win += p_dist[0] * SCALE;
            for (int i = 1; i < MAX_ROLLS; ++i) {
                float p_player_i = p_dist[i] * SCALE;
                if (p_player_i == 0.0f) continue;
                p_gammon_win += p_player_i * ((*opp_gammon)[i - 1] * SCALE);
            }
        }
        if (player_gammon) {
            for (int j = 1; j < MAX_ROLLS; ++j) {
                float p_opp_j = o_dist[j] * SCALE;
                if (p_opp_j == 0.0f) continue;
                p_gammon_loss += p_opp_j * ((*player_gammon)[j] * SCALE);
            }
        }
    } else {
        if (opp_gammon) {
            p_gammon_win += p_dist[0] * SCALE;
            for (int i = 1; i < MAX_ROLLS; ++i) {
                float p_player_i = p_dist[i] * SCALE;
                if (p_player_i == 0.0f) continue;
                p_gammon_win += p_player_i * ((*opp_gammon)[i] * SCALE);
            }
        }
        if (player_gammon) {
            for (int j = 1; j < MAX_ROLLS; ++j) {
                float p_opp_j = o_dist[j] * SCALE;
                if (p_opp_j == 0.0f) continue;
                float player_zero = (j - 1 >= 0 && j - 1 < MAX_ROLLS)
                    ? (*player_gammon)[j - 1] * SCALE : 0.0f;
                p_gammon_loss += p_opp_j * player_zero;
            }
        }
    }

    return {p_win, p_gammon_win, 0.0f, p_gammon_loss, 0.0f};
}

float BearoffDB::lookup_epc(const Board& board, int player) const {
    unsigned int idx = (player == 0)
        ? board_to_player_index(board)
        : board_to_opponent_index(board);
    return mean_rolls_[idx] * (49.0f / 6.0f);
}

const std::array<uint16_t, BearoffDB::MAX_ROLLS>*
BearoffDB::get_gammon_distribution(unsigned int index) const {
    int gi = gammon_index_[index];
    if (gi < 0) return nullptr;
    return &gammon_distributions_[gi];
}

// --- Generation ---

namespace {

Board checkers_to_board(const int checkers[BearoffDB::POINTS]) {
    Board b{};
    for (int i = 0; i < BearoffDB::POINTS; ++i) {
        b[i + 1] = checkers[i];
    }
    return b;
}

void board_to_checkers(const Board& board, int checkers[BearoffDB::POINTS]) {
    for (int i = 0; i < BearoffDB::POINTS; ++i) {
        checkers[i] = std::max(0, board[i + 1]);
    }
}

int total_pips(const int checkers[BearoffDB::POINTS]) {
    int pips = 0;
    for (int i = 0; i < BearoffDB::POINTS; ++i) {
        pips += checkers[i] * (i + 1);
    }
    return pips;
}

int total_checkers(const int checkers[BearoffDB::POINTS]) {
    int sum = 0;
    for (int i = 0; i < BearoffDB::POINTS; ++i) {
        sum += checkers[i];
    }
    return sum;
}

} // namespace

void BearoffDB::generate() {
    if (!binom_initialized_) init_binom();

    struct PosOrder {
        unsigned int index;
        int pips;
    };
    std::vector<PosOrder> order(N_POSITIONS);

    for (unsigned int idx = 0; idx < N_POSITIONS; ++idx) {
        int checkers[POINTS];
        index_to_position(idx, checkers);
        order[idx] = {idx, total_pips(checkers)};
    }

    std::sort(order.begin(), order.end(),
              [](const PosOrder& a, const PosOrder& b) { return a.pips < b.pips; });

    int zero_checkers[POINTS] = {};
    unsigned int terminal_idx = position_index(zero_checkers);
    distributions_[terminal_idx].fill(0);
    distributions_[terminal_idx][0] = 65535;
    mean_rolls_[terminal_idx] = 0.0f;

    std::vector<bool> computed(N_POSITIONS, false);
    computed[terminal_idx] = true;

    std::vector<Board> move_buf;

    for (const auto& po : order) {
        if (computed[po.index]) continue;

        int checkers[POINTS];
        index_to_position(po.index, checkers);

        if (total_checkers(checkers) == 0) {
            computed[po.index] = true;
            continue;
        }

        Board board = checkers_to_board(checkers);
        double acc_dist[MAX_ROLLS] = {};
        double total_weight = 0.0;

        for (const auto& roll : kBearoffRolls) {
            possible_boards(board, roll.d1, roll.d2, move_buf);

            float best_mean = 1e9f;
            int best_move_idx = -1;

            for (int m = 0; m < static_cast<int>(move_buf.size()); ++m) {
                int next_checkers[POINTS];
                board_to_checkers(move_buf[m], next_checkers);
                unsigned int next_idx = position_index(next_checkers);
                assert(computed[next_idx]);
                float next_mean = mean_rolls_[next_idx];
                if (next_mean < best_mean) {
                    best_mean = next_mean;
                    best_move_idx = m;
                }
            }

            int best_checkers[POINTS];
            board_to_checkers(move_buf[best_move_idx], best_checkers);
            unsigned int best_idx = position_index(best_checkers);
            const auto& next_dist = distributions_[best_idx];

            for (int k = 0; k < MAX_ROLLS - 1; ++k) {
                acc_dist[k + 1] += static_cast<double>(next_dist[k]) * roll.weight;
            }
            total_weight += roll.weight;
        }

        double frac_dist[MAX_ROLLS];
        for (int k = 0; k < MAX_ROLLS; ++k) {
            frac_dist[k] = acc_dist[k] / (total_weight * 65535.0);
        }

        int32_t quant[MAX_ROLLS];
        int32_t quant_sum = 0;
        int max_idx = 0;
        double max_frac = 0.0;
        for (int k = 0; k < MAX_ROLLS; ++k) {
            double val = frac_dist[k] * 65535.0;
            quant[k] = static_cast<int32_t>(val + 0.5);
            quant_sum += quant[k];
            if (frac_dist[k] > max_frac) {
                max_frac = frac_dist[k];
                max_idx = k;
            }
        }
        quant[max_idx] += (65535 - quant_sum);

        for (int k = 0; k < MAX_ROLLS; ++k) {
            distributions_[po.index][k] = static_cast<uint16_t>(
                std::max(0, std::min(65535, quant[k])));
        }

        double mean = 0.0;
        for (int k = 0; k < MAX_ROLLS; ++k) {
            mean += k * (distributions_[po.index][k] / 65535.0);
        }
        mean_rolls_[po.index] = static_cast<float>(mean);

        computed[po.index] = true;
    }

    generate_gammon_distributions();

    loaded_ = true;
    identity_ = next_identity();
}

void BearoffDB::generate_gammon_distributions() {
    gammon_index_.fill(-1);
    int gammon_count = 0;
    for (unsigned int idx = 0; idx < N_POSITIONS; ++idx) {
        int checkers[POINTS];
        index_to_position(idx, checkers);
        if (total_checkers(checkers) == CHECKERS) {
            gammon_index_[idx] = gammon_count++;
        }
    }
    gammon_distributions_.resize(gammon_count);

    struct PosOrder { unsigned int index; int pips; };
    std::vector<PosOrder> order;
    order.reserve(gammon_count);
    for (unsigned int idx = 0; idx < N_POSITIONS; ++idx) {
        if (gammon_index_[idx] >= 0) {
            int checkers[POINTS];
            index_to_position(idx, checkers);
            order.push_back({idx, total_pips(checkers)});
        }
    }
    std::sort(order.begin(), order.end(),
              [](const PosOrder& a, const PosOrder& b) { return a.pips < b.pips; });

    std::vector<Board> move_buf;

    for (const auto& po : order) {
        int gi = gammon_index_[po.index];
        auto& zero_off = gammon_distributions_[gi];
        zero_off.fill(0);
        zero_off[0] = 65535;

        int checkers[POINTS];
        index_to_position(po.index, checkers);
        Board board = checkers_to_board(checkers);

        double acc[MAX_ROLLS] = {};

        for (const auto& roll : kBearoffRolls) {
            possible_boards(board, roll.d1, roll.d2, move_buf);

            float best_mean = 1e9f;
            int best_move_idx = 0;
            for (int m = 0; m < static_cast<int>(move_buf.size()); ++m) {
                int next_checkers[POINTS];
                board_to_checkers(move_buf[m], next_checkers);
                unsigned int next_idx = position_index(next_checkers);
                if (mean_rolls_[next_idx] < best_mean) {
                    best_mean = mean_rolls_[next_idx];
                    best_move_idx = m;
                }
            }

            int next_checkers[POINTS];
            board_to_checkers(move_buf[best_move_idx], next_checkers);
            if (total_checkers(next_checkers) == CHECKERS) {
                unsigned int next_idx = position_index(next_checkers);
                int next_gi = gammon_index_[next_idx];
                assert(next_gi >= 0);
                const auto& next_zero = gammon_distributions_[next_gi];
                for (int k = 0; k < MAX_ROLLS - 1; ++k) {
                    acc[k + 1] += static_cast<double>(next_zero[k]) * roll.weight;
                }
            }
        }

        zero_off[0] = 65535;
        for (int k = 1; k < MAX_ROLLS; ++k) {
            double val = (acc[k] / (36.0 * 65535.0)) * 65535.0;
            zero_off[k] = static_cast<uint16_t>(
                std::max(0.0, std::min(65535.0, val + 0.5)));
        }
    }
}

// --- Save/Load ---

bool BearoffDB::save(const std::string& path) const {
    if (!loaded_) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    out.write(reinterpret_cast<const char*>(&MAGIC), 4);
    out.write(reinterpret_cast<const char*>(&VERSION), 4);
    uint32_t points = POINTS;
    uint32_t checkers = CHECKERS;
    uint32_t max_rolls = MAX_ROLLS;
    uint32_t n_positions = N_POSITIONS;
    out.write(reinterpret_cast<const char*>(&points), 4);
    out.write(reinterpret_cast<const char*>(&checkers), 4);
    out.write(reinterpret_cast<const char*>(&max_rolls), 4);
    out.write(reinterpret_cast<const char*>(&n_positions), 4);

    uint32_t n_gammon = static_cast<uint32_t>(gammon_distributions_.size());
    out.write(reinterpret_cast<const char*>(&n_gammon), 4);

    out.write(reinterpret_cast<const char*>(distributions_.data()),
              N_POSITIONS * MAX_ROLLS * sizeof(uint16_t));
    out.write(reinterpret_cast<const char*>(mean_rolls_.data()),
              N_POSITIONS * sizeof(float));
    out.write(reinterpret_cast<const char*>(gammon_index_.data()),
              N_POSITIONS * sizeof(int));

    if (n_gammon > 0) {
        out.write(reinterpret_cast<const char*>(gammon_distributions_.data()),
                  n_gammon * MAX_ROLLS * sizeof(uint16_t));
    }

    return out.good();
}

bool BearoffDB::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    uint32_t magic, version, points, checkers_val, max_rolls, n_positions, n_gammon;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&points), 4);
    in.read(reinterpret_cast<char*>(&checkers_val), 4);
    in.read(reinterpret_cast<char*>(&max_rolls), 4);
    in.read(reinterpret_cast<char*>(&n_positions), 4);
    in.read(reinterpret_cast<char*>(&n_gammon), 4);

    if (magic != MAGIC || version != VERSION) return false;
    if (points != POINTS || checkers_val != CHECKERS) return false;
    if (max_rolls != MAX_ROLLS || n_positions != N_POSITIONS) return false;

    in.read(reinterpret_cast<char*>(distributions_.data()),
            N_POSITIONS * MAX_ROLLS * sizeof(uint16_t));
    in.read(reinterpret_cast<char*>(mean_rolls_.data()),
            N_POSITIONS * sizeof(float));
    in.read(reinterpret_cast<char*>(gammon_index_.data()),
            N_POSITIONS * sizeof(int));

    gammon_distributions_.resize(n_gammon);
    if (n_gammon > 0) {
        in.read(reinterpret_cast<char*>(gammon_distributions_.data()),
                n_gammon * MAX_ROLLS * sizeof(uint16_t));
    }

    if (!binom_initialized_) init_binom();
    loaded_ = in.good();
    identity_ = loaded_ ? next_identity() : 0;
    return loaded_;
}

uint64_t BearoffDB::next_identity() {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

} // namespace bgbot
