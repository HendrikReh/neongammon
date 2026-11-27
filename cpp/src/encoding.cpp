// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/encoding.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "bgbot/board.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <utility>

namespace bgbot {

// ======================== Escape Lookup Tables (GNUbg-style) ========================
// Precomputed tables mapping a 12-bit bitmap of blocked points to escape roll counts.
// anPoint: maps checker count to "is blocked" (>=2 means blocked = 1).
static const int anPoint[16] = {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

static int anEscapes[0x1000];
static int anEscapes1[0x1000];
static uint8_t back_escape_counts[6][4096];
static int entry_losses[2][64];
static std::once_flag escape_tables_init_flag;

static void compute_escape_table0() {
    for (int i = 0; i < 0x1000; i++) {
        int c = 0;
        for (int n0 = 0; n0 <= 5; n0++)
            for (int n1 = 0; n1 <= n0; n1++)
                if (!(i & (1 << (n0 + n1 + 1))) &&
                    !((i & (1 << n0)) && (i & (1 << n1))))
                    c += (n0 == n1) ? 1 : 2;
        anEscapes[i] = c;
    }
}

static void compute_escape_table1() {
    anEscapes1[0] = 0;
    for (int i = 1; i < 0x1000; i++) {
        int c = 0;
        int low = 0;
        while (!(i & (1 << low))) ++low;

        for (int n0 = 0; n0 <= 5; n0++)
            for (int n1 = 0; n1 <= n0; n1++)
                if ((n0 + n1 + 1 > low) &&
                    !(i & (1 << (n0 + n1 + 1))) &&
                    !((i & (1 << n0)) && (i & (1 << n1))))
                    c += (n0 == n1) ? 1 : 2;
        anEscapes1[i] = c;
    }
}

// Raw table builders called strictly inside std::call_once (never re-enter init_escape_tables).
static uint8_t build_back_escapes_entry(int back, unsigned mask) {
    // Bit b (0..11) marks opponent anchor at point (back - 12 + b).
    auto is_blocked = [back, mask](int pt) -> bool {
        const int bit = pt - (back - 12);
        return bit >= 0 && bit < 12 && ((mask & (1u << bit)) != 0);
    };

    int n_escapes = 0;
    for (int d1 = 1; d1 <= 6; ++d1) {
        for (int d2 = 1; d2 <= d1; ++d2) {
            if (d1 != d2) {
                if (back - d1 - d2 > 15) continue;
                if (is_blocked(back - d1) && is_blocked(back - d2)) continue;
                if (is_blocked(back - d1 - d2)) continue;
                n_escapes += 2;
            } else {
                if (back - 4 * d1 > 15) continue;
                if (is_blocked(back - d1)) continue;
                if (is_blocked(back - 2 * d1)) continue;

                if (back - 2 * d1 <= 15) {
                    n_escapes += 1;
                } else if (back - 3 * d1 <= 15 && back - 3 * d1 > 0 && !is_blocked(back - 3 * d1)) {
                    n_escapes += 1;
                } else if (back - 4 * d1 <= 15 && back - 4 * d1 > 0 &&
                           !is_blocked(back - 3 * d1) && !is_blocked(back - 4 * d1)) {
                    n_escapes += 1;
                }
            }
        }
    }
    return static_cast<uint8_t>(n_escapes);
}

static int build_entry_loss_entry(bool two_on_bar, unsigned opp_home_mask) {
    int loss = 0;
    for (int i = 0; i < 6; ++i) {
        if (opp_home_mask & (1u << i)) {
            loss += 4 * (i + 1);
            for (int j = i + 1; j < 6; ++j) {
                if (opp_home_mask & (1u << j)) {
                    loss += 2 * (i + j + 2);
                } else if (two_on_bar) {
                    loss += 2 * (i + 1);
                }
            }
        } else if (two_on_bar) {
            for (int j = i + 1; j < 6; ++j) {
                if (opp_home_mask & (1u << j)) {
                    loss += 2 * (j + 1);
                }
            }
        }
    }
    return loss;
}

void init_escape_tables() {
    std::call_once(escape_tables_init_flag, []() {
        compute_escape_table0();
        compute_escape_table1();
        // The trained back-escape feature tests the 12 points immediately
        // in front of the back checker. A longer double already escapes
        // by its second step, so no farther blocker can change its result.
        for (int back = 19; back <= 24; ++back) {
            for (unsigned mask = 0; mask < 4096; ++mask) {
                back_escape_counts[back - 19][mask] = build_back_escapes_entry(back, mask);
            }
        }
        for (int two = 0; two <= 1; ++two) {
            for (unsigned mask = 0; mask < 64; ++mask) {
                entry_losses[two][mask] = build_entry_loss_entry(two != 0, mask);
            }
        }
    });
}

// Escapes: count rolls that let a checker at position n (0-indexed, GNUbg style)
// escape, given half_board[0..24] (one player's checkers, unsigned counts).
int escapes(const int half_board[25], int n) {
    init_escape_tables();
    int af = 0;
    int m = (n < 12) ? n : 12;
    for (int i = 0; i < m; i++) {
        int idx = 24 + i - n;
        if (idx >= 0 && idx < 25)
            af |= (anPoint[half_board[idx] < 0 ? 0 : (half_board[idx] > 15 ? 15 : half_board[idx])] << i);
    }
    return anEscapes[af];
}

int escapes1(const int half_board[25], int n) {
    init_escape_tables();
    int af = 0;
    int m = (n < 12) ? n : 12;
    for (int i = 0; i < m; i++) {
        int idx = 24 + i - n;
        if (idx >= 0 && idx < 25)
            af |= (anPoint[half_board[idx] < 0 ? 0 : (half_board[idx] > 15 ? 15 : half_board[idx])] << i);
    }
    return anEscapes1[af];
}

namespace {
// Point encoding is: [n>=1, n>=2, n>=3, (n-3)/2] with hard cap at 15.
// Precomputing avoids hot-path branches in input encoding loops.
static constexpr float k_point_features[16][4] = {
    {0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 1.0f, 0.0f, 0.0f},
    {1.0f, 1.0f, 1.0f, 0.0f},
    {1.0f, 1.0f, 1.0f, 0.5f},
    {1.0f, 1.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 1.0f, 1.5f},
    {1.0f, 1.0f, 1.0f, 2.0f},
    {1.0f, 1.0f, 1.0f, 2.5f},
    {1.0f, 1.0f, 1.0f, 3.0f},
    {1.0f, 1.0f, 1.0f, 3.5f},
    {1.0f, 1.0f, 1.0f, 4.0f},
    {1.0f, 1.0f, 1.0f, 4.5f},
    {1.0f, 1.0f, 1.0f, 5.0f},
    {1.0f, 1.0f, 1.0f, 5.5f},
    {1.0f, 1.0f, 1.0f, 6.0f},
};

static inline const float* point_feature_row(int n) {
    if (n < 0) return k_point_features[0];
    if (n > 15) return k_point_features[15];
    return k_point_features[n];
}
} // namespace

// ======================== Tesauro 196-input encoding ========================

std::array<float, TESAURO_INPUTS> compute_tesauro_inputs(const Board& board) {
    std::array<float, TESAURO_INPUTS> inputs;

    for (int i = 0; i < 24; ++i) {
        int n = board[i + 1];  // point i+1; positive = player 1, negative = player 2

        const float* p_feat = point_feature_row(n);
        const int p0 = 4 * i;
        inputs[p0 + 0] = p_feat[0];
        inputs[p0 + 1] = p_feat[1];
        inputs[p0 + 2] = p_feat[2];
        inputs[p0 + 3] = p_feat[3];

        int m = -n;
        const float* m_feat = point_feature_row(m);
        const int p1 = p0 + 98;
        inputs[p1 + 0] = m_feat[0];
        inputs[p1 + 1] = m_feat[1];
        inputs[p1 + 2] = m_feat[2];
        inputs[p1 + 3] = m_feat[3];
    }

    inputs[96]  = board[25] / 2.0f;
    inputs[194] = board[0]  / 2.0f;
    inputs[97]  = player_borne_off(board)   / 15.0f;
    inputs[195] = opponent_borne_off(board) / 15.0f;

    return inputs;
}

// ======================== Shared Contact Features ========================

float backg(const Board& board) {
    int nAc = 0;
    for (int pt = 19; pt <= 24; ++pt) {
        if (board[pt] >= 2) ++nAc;
    }

    if (nAc > 1) {
        int tot = board[25];
        for (int pt = 19; pt <= 24; ++pt) {
            if (board[pt] > 0) tot += board[pt];
        }
        return (tot - 3) / 4.0f;
    }
    return 0.0f;
}

float backg1(const Board& board) {
    int nAc = 0;
    for (int pt = 19; pt <= 24; ++pt) {
        if (board[pt] >= 2) ++nAc;
    }

    if (nAc == 1) {
        int tot = board[25];
        for (int pt = 19; pt <= 24; ++pt) {
            if (board[pt] > 0) tot += board[pt];
        }
        return tot / 8.0f;
    }
    return 0.0f;
}

int moment2(const Board& board) {
    int checkers[25];
    for (int gnubg_i = 0; gnubg_i < 24; ++gnubg_i) {
        const int n = board[gnubg_i + 1];
        checkers[gnubg_i] = n > 0 ? n : 0;
    }
    checkers[24] = board[25] > 0 ? board[25] : 0;

    int total_checkers = 0;
    int total_weighted = 0;
    for (int gnubg_i = 0; gnubg_i < 25; ++gnubg_i) {
        total_checkers += checkers[gnubg_i];
        total_weighted += gnubg_i * checkers[gnubg_i];
    }

    int mean = 0;
    if (total_checkers > 0) {
        mean = (total_weighted + total_checkers - 1) / total_checkers;
    }

    int k = 0;
    int j = 0;
    for (int gnubg_i = mean + 1; gnubg_i < 25; ++gnubg_i) {
        j += checkers[gnubg_i];
        k += checkers[gnubg_i] * (gnubg_i - mean) * (gnubg_i - mean);
    }

    if (j > 0) {
        k = (k + j - 1) / j;
    }

    return k;
}

// ======================== Production Bitmask Feature Kernels ========================

static int break_contact_fast(const Board& board, int opp_back) {
    if (opp_back == 0) return 0;
    int n = 0;
    for (int pt = opp_back + 1; pt <= 24; ++pt) {
        if (board[pt] > 0)
            n += (pt - opp_back) * board[pt];
    }
    if (board[25] > 0)
        n += (25 - opp_back) * board[25];
    return n;
}

static int free_pip_fast(const Board& board, int opp_back) {
    if (opp_back == 0) return 0;
    int p = 0;
    for (int pt = 1; pt < opp_back; ++pt) {
        if (board[pt] > 0)
            p += pt * board[pt];
    }
    return p;
}

static int timing_fast(const Board& board, int opp_back) {
    int gnubg_opp_back = opp_back > 0 ? opp_back - 1 : -1;

    int t = 0;
    int no = 0;

    t += 24 * board[25];
    no += board[25];

    for (int pt = 24; pt >= 13 && (pt - 1) > gnubg_opp_back; --pt) {
        int checkers = (board[pt] > 0) ? board[pt] : 0;
        if (checkers > 0 && checkers != 2) {
            int n = (checkers > 2) ? (checkers - 2) : 1;
            no += n;
            t += (pt - 1) * n;
        }
    }

    int start_pt = (13 > opp_back) ? 12 : opp_back;
    for (int pt = start_pt; pt >= 7; --pt) {
        int checkers = (board[pt] > 0) ? board[pt] : 0;
        if (checkers > 0) {
            no += checkers;
            t += (pt - 1) * checkers;
        }
    }

    for (int pt = 6; pt >= 1; --pt) {
        int checkers = (board[pt] > 0) ? board[pt] : 0;
        if (checkers > 2) {
            t += (pt - 1) * (checkers - 2);
            no += (checkers - 2);
        } else if (checkers < 2) {
            int deficit = 2 - checkers;
            if (no >= deficit) {
                t -= (pt - 1) * deficit;
                no -= deficit;
            }
        }
    }

    if (t < 0) t = 0;
    return t;
}

static float acontainment_fast(unsigned player_blocks, int opp_back) {
    int gnubg_nOppBack = opp_back > 0 ? opp_back - 1 : -1;

    int n = 36;
    int limit = (gnubg_nOppBack >= 0) ? 24 - gnubg_nOppBack : 24;
    for (int gnubg_i = 15; gnubg_i < limit; ++gnubg_i) {
        int j = anEscapes[(player_blocks >> (24 - gnubg_i)) & 4095];
        if (j < n) n = j;
    }
    return 36.0f - n;
}

static float containment_fast(unsigned player_blocks) {
    int n = 36;
    for (int gnubg_i = 15; gnubg_i < 24; ++gnubg_i) {
        int j = anEscapes[(player_blocks >> (24 - gnubg_i)) & 4095];
        if (j < n) n = j;
    }
    return 36.0f - n;
}

static int mobility_fast(const Board& board, unsigned opp_blocks) {
    int n = 0;
    for (int gnubg_i = 6; gnubg_i < 25; ++gnubg_i) {
        int our_pt = gnubg_i + 1;
        int checkers = 0;
        if (our_pt <= 24) {
            checkers = (board[our_pt] > 0) ? board[our_pt] : 0;
        } else {
            checkers = board[25];
        }
        if (checkers > 0) {
            n += (gnubg_i - 5) * checkers * anEscapes[(opp_blocks >> (24 - gnubg_i)) & 4095];
        }
    }
    return n;
}

static int back_rescue_escapes_fast(unsigned player_blocks, int opp_back) {
    int gnubg_nOppBack = opp_back > 0 ? opp_back - 1 : -1;
    if (gnubg_nOppBack < 0) return 36;
    int escape_point = 23 - gnubg_nOppBack;
    if (escape_point < 0) return 0;
    return anEscapes1[(player_blocks >> (24 - escape_point)) & 4095];
}

// Point 1 is bit 0. A shift to the right moves a source checker towards
// home; all blots on a given roll can then be tested together.
struct ContactPointMasks {
    unsigned occupied = 0;
    unsigned made = 0;
    unsigned enemy_blots = 0;
    unsigned enemy_blocks = 0;
    unsigned pip_loss_sources = 0;
    unsigned pip_loss_targets = 0;
};

static ContactPointMasks contact_point_masks(const Board& board) {
    ContactPointMasks masks;
    for (int i = 0; i < 24; ++i) {
        const int checkers = board[i + 1];
        masks.occupied |= static_cast<unsigned>(checkers > 0) << i;
        masks.made |= static_cast<unsigned>(checkers >= 2) << i;
        masks.enemy_blots |= static_cast<unsigned>(checkers == -1) << i;
        masks.enemy_blocks |= static_cast<unsigned>(checkers <= -2) << i;
    }
    int occupied_home = 0;
    unsigned two_checker_home_points = 0;
    for (int i = 0; i < 6; ++i) {
        occupied_home += board[i + 1] > 0;
        two_checker_home_points |= static_cast<unsigned>(board[i + 1] == 2) << i;
    }
    // GNUbg's pip-loss feature preserves two-checker home points. With at
    // most two occupied home points it also skips enemy blots on points 1–2.
    masks.pip_loss_sources = masks.occupied & ~two_checker_home_points;
    masks.pip_loss_targets = occupied_home > 2 ? masks.enemy_blots : masks.enemy_blots & ~3u;
    return masks;
}

// Each bit describes one of the 24 points, with the ace point at bit zero.
#if defined(__AVX2__)
static unsigned reverse_points(unsigned bits) {
    static constexpr auto reversed_byte = [] {
        std::array<unsigned, 256> table{};
        for (unsigned i = 1; i < table.size(); ++i)
            table[i] = (table[i >> 1] >> 1) | ((i & 1u) << 7);
        return table;
    }();
    return (reversed_byte[bits & 255] << 16) |
           (reversed_byte[(bits >> 8) & 255] << 8) | reversed_byte[bits >> 16];
}

static std::pair<ContactPointMasks, ContactPointMasks> both_contact_point_masks(const Board& board) {
    unsigned player_points = 0, player_made = 0, enemy_points = 0, enemy_made = 0;
    unsigned player_twos = 0, enemy_twos = 0;
    const auto mask = [](__m256i v) { return static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(v))); };
    const auto zero = _mm256_setzero_si256();
    for (int i = 0; i < 24; i += 8) {
        const auto points = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(board.data() + i + 1));
        player_points |= mask(_mm256_cmpgt_epi32(points, zero)) << i;
        player_made |= mask(_mm256_cmpgt_epi32(points, _mm256_set1_epi32(1))) << i;
        enemy_points |= mask(_mm256_cmpgt_epi32(zero, points)) << i;
        enemy_made |= mask(_mm256_cmpgt_epi32(_mm256_set1_epi32(-1), points)) << i;
        if (i == 0) player_twos = mask(_mm256_cmpeq_epi32(points, _mm256_set1_epi32(2))) & 63;
        if (i == 16) enemy_twos = (mask(_mm256_cmpeq_epi32(points, _mm256_set1_epi32(-2))) & 252) << 16;
    }
    ContactPointMasks player{}, opponent{};
    const auto three_home_points = [](unsigned occupied) {
        unsigned home = occupied & 63;
        home &= home - 1;
        return (home & (home - 1)) != 0;
    };
    player.occupied = player_points;
    player.made = player_made;
    player.enemy_blots = enemy_points ^ enemy_made;
    player.enemy_blocks = enemy_made;
    player.pip_loss_sources = player_points & ~player_twos;
    player.pip_loss_targets = three_home_points(player_points) ? player.enemy_blots : player.enemy_blots & ~3u;
    opponent.occupied = reverse_points(enemy_points);
    opponent.made = reverse_points(enemy_made);
    opponent.enemy_blots = reverse_points(player_points ^ player_made);
    opponent.enemy_blocks = reverse_points(player_made);
    opponent.pip_loss_sources = reverse_points(enemy_points & ~enemy_twos);
    opponent.pip_loss_targets = three_home_points(opponent.occupied) ? opponent.enemy_blots : opponent.enemy_blots & ~3u;
    return {player, opponent};
}
#endif

// One-based point number, with zero denoting the empty mask.
static inline int highest_set_point(unsigned mask) {
    if (mask == 0) return 0;
    return std::numeric_limits<unsigned>::digits - __builtin_clz(mask);
}

static int forward_anchor_from_mask(unsigned made) {
    const unsigned on_opponent_side = made >> 12;
    if (on_opponent_side == 0) return 0;
    // Lowest of points 13–24, expressed from the opponent's perspective.
    return 13 - highest_set_point(on_opponent_side & (0u - on_opponent_side));
}

static float blocked_entry_probability(unsigned enemy_blocks) {
    const int blocked = __builtin_popcount((enemy_blocks >> 18) & 63u);
    return static_cast<float>(blocked * blocked) / 36.0f;
}

static float backbone_from_mask(const Board& board, unsigned made) {
    made &= ~1u;  // The backbone feature excludes point 1.
    int previous = 0;
    int weighted_count = 0;
    int checker_count = 0;
    while (made) {
        const int point = highest_set_point(made);
        made ^= 1u << (point - 1);
        if (previous) {
            const int gap = previous - point;
            const int weight = gap <= 6 ? 11 : (gap <= 11 ? 13 - gap : 0);
            weighted_count += weight * board[previous];
            checker_count += board[previous];
        }
        previous = point;
    }
    if (checker_count == 0) return 0.0f;
    return 1.0f - static_cast<float>(weighted_count) / static_cast<float>(11 * checker_count);
}

// Each intermediate blocker mask has already been shifted to the target
// point's coordinates. A path of length k needs k-1 open intermediate points.
static unsigned double_roll_targets(unsigned sources, unsigned blots, int die,
                                    unsigned block1, unsigned block2, unsigned block3) {
    const unsigned from_two = sources >> (2 * die);
    const unsigned from_three = sources >> (3 * die);
    const unsigned from_four = sources >> (4 * die);
    const unsigned indirect = ~block1 & (from_two | (~block2 & (from_three | (~block3 & from_four))));
    return blots & ((sources >> die) | indirect);
}

static bool different_targets(unsigned first, unsigned second) {
    const unsigned both = first | second;
    return first && second && (both & (both - 1));
}

static bool two_hits_on_doubles(const ContactPointMasks& masks, unsigned sources,
                                unsigned direct, int die, unsigned block1,
                                unsigned block2, unsigned block3) {
    const unsigned blots = masks.enemy_blots;
    const unsigned after_two = blots & ~block1 & (sources >> (2 * die));
    const unsigned after_three = blots & ~block1 & ~block2 & (sources >> (3 * die));
    if (different_targets(direct, direct | after_two | after_three)) return true;

    const unsigned lower_one = blots << die;
    const unsigned lower_two = blots << (2 * die);
    const unsigned lower_three = blots << (3 * die);
    const unsigned lower_targets = lower_one | (~(masks.enemy_blocks << die) &
        (lower_two | (~(masks.enemy_blocks << (2 * die)) & lower_three)));
    if (direct & lower_targets) return true;

    const unsigned higher_one = blots >> die;
    const unsigned higher_two = blots >> (2 * die);
    const unsigned from_two = masks.occupied >> (2 * die);
    const unsigned from_three = masks.occupied >> (3 * die);
    const unsigned from_four = masks.occupied >> (4 * die);
    const unsigned via_blot = (higher_one & from_two) |
        (~block2 & (higher_one | higher_two) & (from_three | (~block3 & from_four)));
    return (blots & ~block1 & via_blot) != 0;
}

static bool two_hits_from_separate_checkers(unsigned high_targets, unsigned low_targets,
                                            int high_die, int low_die, unsigned stacked) {
    if (low_targets == 0) return false;
    while (high_targets) {
        const unsigned target = high_targets & (0u - high_targets);
        const unsigned source = target << high_die;
        const unsigned same_checker = (source & stacked) ? 0u : source >> low_die;
        if (low_targets & ~(target | same_checker)) return true;
        high_targets ^= target;
    }
    return false;
}

static int entry_pip_loss(const Board& board, const ContactPointMasks& masks) {
    constexpr unsigned bar = 1u << 24;
    const unsigned targets = masks.pip_loss_targets;
    if (!targets) return 0;
    const unsigned blocks = masks.enemy_blocks;
    unsigned direct[7], entered[7];
    int total = 0;
    const bool one_on_bar = board[25] == 1;
    for (int die = 1; die <= 6; ++die) {
        entered[die] = (bar >> die) & targets;
        direct[die] = (masks.pip_loss_sources >> die) & targets;
        unsigned hits = entered[die];
        if (one_on_bar) {
            hits = double_roll_targets(bar, targets, die, blocks >> die, blocks >> (2 * die),
                                       blocks >> (3 * die));
            if (!(blocks & (bar >> (2 * die)))) hits |= direct[die];
        }
        total += highest_set_point(hits);
    }
    for (int high = 2; high <= 6; ++high) {
        for (int low = 1; low < high; ++low) {
            unsigned hits = entered[low] | entered[high];
            if (one_on_bar) {
                const bool enter_high = !(blocks & (bar >> high));
                const bool enter_low = !(blocks & (bar >> low));
                if (enter_high || enter_low) hits |= (bar >> (high + low)) & targets;
                if (enter_high && direct[low]) hits |= direct[low];
                else if (enter_low) hits |= direct[high];
            }
            total += 2 * highest_set_point(hits);
        }
    }
    return total;
}

struct ShotFeatures {
    int single_hits = 0;
    int double_hits = 0;
    int pip_loss = 0;
};

#if defined(__AVX2__)
// Modal only (Linux x86-64): eight independent 24-point target masks, one per dice roll.
struct RollMasks {
    __m256i bits;
    explicit RollMasks(unsigned value) : bits(_mm256_set1_epi32(value)) {}
    RollMasks(__m256i value) : bits(value) {}
    friend RollMasks operator&(RollMasks a, RollMasks b) { return _mm256_and_si256(a.bits, b.bits); }
    friend RollMasks operator|(RollMasks a, RollMasks b) { return _mm256_or_si256(a.bits, b.bits); }
    friend RollMasks operator~(RollMasks a) { return _mm256_xor_si256(a.bits, _mm256_set1_epi32(-1)); }
    friend RollMasks operator+(RollMasks a, RollMasks b) { return _mm256_add_epi32(a.bits, b.bits); }
    friend RollMasks operator-(RollMasks a, RollMasks b) { return _mm256_sub_epi32(a.bits, b.bits); }
    friend RollMasks operator>>(RollMasks a, RollMasks b) { return _mm256_srlv_epi32(a.bits, b.bits); }
    friend RollMasks operator<<(RollMasks a, RollMasks b) { return _mm256_sllv_epi32(a.bits, b.bits); }
    RollMasks first() const { return *this & (RollMasks(0u) - *this); }
    RollMasks nonempty() const { return _mm256_cmpgt_epi32(bits, _mm256_setzero_si256()); }
    int hit_rolls() const {
        return _mm_popcnt_u32(_mm256_movemask_ps(_mm256_castsi256_ps(nonempty().bits)));
    }
    RollMasks highest_points() const {
        const __m256i exp = _mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(bits)), 23);
        return _mm256_max_epi32(_mm256_setzero_si256(), _mm256_sub_epi32(exp, _mm256_set1_epi32(126)));
    }
    int sum() const {
        int lanes[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), bits);
        return std::accumulate(lanes, lanes + 8, 0);
    }
};

static ShotFeatures off_bar_shots(const ContactPointMasks& masks, unsigned reachable_blots) {
    ShotFeatures counts;
    const RollMasks occupied(masks.occupied), made(masks.made), blocks(masks.enemy_blocks);
    const RollMasks pip_sources(masks.pip_loss_sources), pip_targets(masks.pip_loss_targets);
    const RollMasks blots(reachable_blots);
    const bool multiple = (reachable_blots & (reachable_blots - 1)) != 0;

    const RollMasks die = _mm256_setr_epi32(1,2,3,4,5,6,32,32);
    const RollMasks twice = die + die, three = twice + die, four = twice + twice;
    const RollMasks b1 = blocks >> die, b2 = blocks >> twice, b3 = blocks >> three;
    const auto double_paths = [&](RollMasks sources) {
        return (sources >> die) | (~b1 & ((sources >> twice) |
               (~b2 & ((sources >> three) | (~b3 & (sources >> four))))));
    };
    counts.single_hits = (blots & double_paths(occupied)).hit_rolls();
    if (multiple) {
        const RollMasks direct = blots & (occupied >> die);
        const RollMasks from2 = occupied >> twice, from3 = occupied >> three, from4 = occupied >> four;
        const RollMasks all = direct | (blots & ~b1 & (from2 | (~b2 & from3)));
        const RollMasks distinct = direct.nonempty() & all & (all - RollMasks(1u));
        const RollMasks lower = (blots << die) | (~(blocks << die) &
            ((blots << twice) | (~(blocks << twice) & (blots << three))));
        const RollMasks higher = blots >> die;
        const RollMasks via_blot = (higher & from2) | (~b2 & (higher | (blots >> twice)) & (from3 | (~b3 & from4)));
        counts.double_hits = (distinct | (direct & lower) | (blots & ~b1 & via_blot)).hit_rolls();
    }
    RollMasks pip_total = (pip_targets & double_paths(pip_sources)).highest_points();

    const RollMasks high[] = {_mm256_setr_epi32(2,3,3,4,4,4,5,5), _mm256_setr_epi32(5,5,6,6,6,6,6,32)};
    const RollMasks low[] = {_mm256_setr_epi32(1,1,2,1,2,3,1,2), _mm256_setr_epi32(3,4,1,2,3,4,5,32)};
    for (int group = 0; group < 2; ++group) {
        const RollMasks h = high[group], l = low[group], sum = h + l;
        const RollMasks open = ~((blocks >> h) & (blocks >> l));
        const RollMasks hd = blots & (occupied >> h), ld = blots & (occupied >> l);
        counts.single_hits += 2 * (hd | ld | (blots & open & (occupied >> sum))).hit_rolls();
        if (multiple) {
            const RollMasks through = (hd & (blots << l)) | (ld & (blots << h));
            const RollMasks first_h = hd.first(), first_l = ld.first();
            const RollMasks lone_h = ~made & (first_h << h), lone_l = ~made & (first_l << l);
            const RollMasks other_l = ld & ~(first_h | (lone_h >> l));
            const RollMasks other_h = hd & ~(first_l | (lone_l >> h));
            counts.double_hits += 2 * (through | (hd.nonempty() & other_l) |
                                       (ld.nonempty() & other_h)).hit_rolls();
        }
        const RollMasks hits = pip_targets &
            ((pip_sources >> h) | (pip_sources >> l) | (open & (pip_sources >> sum)));
        const RollMasks points = hits.highest_points();
        pip_total = pip_total + points + points;
    }
    counts.pip_loss = pip_total.sum();
    return counts;
}
#endif

template<bool OnBar>
static ShotFeatures shot_features(const Board& board, const ContactPointMasks& masks) {
    ShotFeatures counts;
    unsigned blots = masks.enemy_blots;
    if (!OnBar && blots) {
        if (!masks.occupied) return counts;
        blots &= (1u << (highest_set_point(masks.occupied) - 1)) - 1;
    }
    if (blots == 0) return counts;
#if defined(__AVX2__)
    if constexpr (!OnBar) return off_bar_shots(masks, blots);
#endif
    const bool multiple_blots = (blots & (blots - 1)) != 0;
    const unsigned sources = masks.occupied | (OnBar ? 1u << 24 : 0u);
    const unsigned stacked = masks.made | (static_cast<unsigned>(board[25] > 1) << 24);
    unsigned direct[7]{}, direct_with_bar[7]{}, pip_direct[7]{};
    for (int die = 1; die <= 6; ++die) {
        const unsigned block1 = masks.enemy_blocks >> die;
        const unsigned block2 = masks.enemy_blocks >> (2 * die);
        const unsigned block3 = masks.enemy_blocks >> (3 * die);
        direct[die] = blots & (masks.occupied >> die);
        direct_with_bar[die] = blots & (sources >> die);
        counts.single_hits += double_roll_targets(masks.occupied, blots, die, block1, block2, block3) != 0;
        if (multiple_blots) {
            counts.double_hits += two_hits_on_doubles(masks, sources, direct_with_bar[die],
                                                      die, block1, block2, block3);
        }
        if constexpr (!OnBar) {
            pip_direct[die] = masks.pip_loss_targets & (masks.pip_loss_sources >> die);
            counts.pip_loss += highest_set_point(double_roll_targets(masks.pip_loss_sources,
                masks.pip_loss_targets, die, block1, block2, block3));
        }
    }
    for (int high = 2; high <= 6; ++high) {
        for (int low = 1; low < high; ++low) {
            const unsigned open_path = ~((masks.enemy_blocks >> high) & (masks.enemy_blocks >> low));
            const unsigned combined = blots & open_path & (masks.occupied >> (high + low));
            counts.single_hits += 2 * ((direct[high] | direct[low] | combined) != 0);
            if (multiple_blots) {
                const bool step_through = (direct_with_bar[high] & (blots << low)) ||
                                          (direct_with_bar[low] & (blots << high));
                counts.double_hits += 2 * (step_through || two_hits_from_separate_checkers(
                    direct_with_bar[high], direct_with_bar[low], high, low, stacked));
            }
            if constexpr (!OnBar) {
                const unsigned pip_combined = masks.pip_loss_targets & open_path &
                                              (masks.pip_loss_sources >> (high + low));
                counts.pip_loss += 2 * highest_set_point(pip_direct[high] | pip_direct[low] | pip_combined);
            }
        }
    }
    if constexpr (OnBar) counts.pip_loss = entry_pip_loss(board, masks);
    return counts;
}

static ShotFeatures compute_shot_features(const Board& board) {
    const auto masks = contact_point_masks(board);
    return board[25] ? shot_features<true>(board, masks)
                     : shot_features<false>(board, masks);
}

// ======================== Public Standalone Feature Helpers ========================
// All standalone feature functions delegate directly to the production bitmask
// and lookup-table kernels used by compute_extended_contact_inputs, ensuring a
// single authoritative implementation.

BarExitProbs prob_no_enter_from_bar(const Board& board) {
    const auto masks = contact_point_masks(board);
    const int player_home_anchors = __builtin_popcount(masks.made & 63u);
    return {
        blocked_entry_probability(masks.enemy_blocks),
        static_cast<float>(player_home_anchors * player_home_anchors) / 36.0f,
    };
}

ForwardAnchorPoints forward_anchor_points(const Board& board) {
    const auto player_masks = contact_point_masks(board);
    const auto opp_masks = contact_point_masks(flip(board));
    return {
        forward_anchor_from_mask(player_masks.made),
        forward_anchor_from_mask(opp_masks.made),
    };
}

int hitting_shots(const Board& board) {
    return compute_shot_features(board).single_hits;
}

int double_hitting_shots(const Board& board) {
    return compute_shot_features(board).double_hits;
}

int compute_piploss(const Board& board) {
    return compute_shot_features(board).pip_loss;
}

int back_escapes(const Board& board) {
    init_escape_tables();
    const auto masks = contact_point_masks(board);
    const int back = highest_set_point(masks.occupied);
    return back < 19 ? 36 : back_escape_counts[back - 19][(masks.enemy_blocks >> (back - 13)) & 4095];
}

int max_point(const Board& board) {
    return highest_set_point(contact_point_masks(board).occupied);
}

int max_anchor_point(const Board& board) {
    return highest_set_point(contact_point_masks(board).made);
}

int break_contact(const Board& board) {
    const auto masks = contact_point_masks(board);
    return break_contact_fast(board, highest_set_point(masks.enemy_blots | masks.enemy_blocks));
}

int free_pip(const Board& board) {
    const auto masks = contact_point_masks(board);
    return free_pip_fast(board, highest_set_point(masks.enemy_blots | masks.enemy_blocks));
}

int timing(const Board& board) {
    const auto masks = contact_point_masks(board);
    return timing_fast(board, highest_set_point(masks.enemy_blots | masks.enemy_blocks));
}

float backbone(const Board& board) {
    return backbone_from_mask(board, contact_point_masks(board).made);
}

int enter_loss(const Board& board) {
    if (board[25] <= 0) return 0;
    init_escape_tables();
    unsigned opp_home = 0;
    for (int bit = 0; bit < 6; ++bit) {
        opp_home |= static_cast<unsigned>(board[24 - bit] <= -2) << bit;
    }
    return entry_losses[board[25] > 1][opp_home];
}

float containment(const Board& board) {
    init_escape_tables();
    return containment_fast(contact_point_masks(board).made) / 36.0f;
}

float acontainment(const Board& board) {
    init_escape_tables();
    const auto masks = contact_point_masks(board);
    return acontainment_fast(masks.made, highest_set_point(masks.enemy_blots | masks.enemy_blocks)) / 36.0f;
}

int mobility(const Board& board) {
    init_escape_tables();
    return mobility_fast(board, contact_point_masks(flip(board)).made);
}

int back_rescue_escapes(const Board& board) {
    init_escape_tables();
    const auto masks = contact_point_masks(board);
    return back_rescue_escapes_fast(masks.made, highest_set_point(masks.enemy_blots | masks.enemy_blocks));
}

// ======================== Extended 244-input Encoding ========================

// A saved half describes exactly the same checkers and bar. Its one-sided
// features survive unchanged. Interaction features also depend on the
// opposing checkers, their made points, and the position of their back man.
static void contact_half(const Board& board, const ContactPointMasks& side, unsigned other_made,
                         float* out, const float* saved = nullptr, int previous_opp_back = -1) {
    const int back = highest_set_point(side.occupied);
    const int opp_back = highest_set_point(side.enemy_blots | side.enemy_blocks);
    if (saved) {
        if (saved != out) std::copy_n(saved, 26, out);
    } else {
        const int off = player_borne_off(board);
        const int forward = forward_anchor_from_mask(side.made);
        const float contain = containment_fast(side.made);
        out[0] = board[25] / 2.0f;
        out[1] = off >= 5 ? 1.0f : off / 5.0f;
        out[2] = off >= 10 ? 1.0f : (off <= 5 ? 0.0f : (off - 5) / 5.0f);
        out[3] = off <= 10 ? 0.0f : (off - 10) / 5.0f;
        out[5] = forward == 0 ? 2.0f : forward / 6.0f;
        out[9] = back / 24.0f;
        out[10] = highest_set_point(side.made) / 24.0f;
        out[16] = contain * (1.0f / 36.0f);
        out[17] = (contain * contain) * (1.0f / 1296.0f);
        out[19] = moment2(board) / 400.0f;
        out[22] = backbone_from_mask(board, side.made);
        out[23] = backg(board);
        out[24] = backg1(board);
    }
    if (!saved || previous_opp_back != opp_back) {
        const float contain = acontainment_fast(side.made, opp_back);
        out[11] = break_contact_fast(board, opp_back) / 167.0f;
        out[12] = free_pip_fast(board, opp_back) / 100.0f;
        out[14] = contain * (1.0f / 36.0f);
        out[15] = (contain * contain) * (1.0f / 1296.0f);
        out[21] = timing_fast(board, opp_back) / 100.0f;
        out[25] = back_rescue_escapes_fast(side.made, opp_back) / 36.0f;
    }
    const auto shots = board[25] ? shot_features<true>(board, side) : shot_features<false>(board, side);
    const int escapes = back < 19 ? 36 : back_escape_counts[back - 19][(side.enemy_blocks >> (back - 13)) & 4095];
    const int entry_loss = board[25] ? entry_losses[board[25] > 1][other_made & 63] : 0;
    out[4] = blocked_entry_probability(side.enemy_blocks);
    out[6] = shots.single_hits / 36.0f;
    out[7] = shots.double_hits / 36.0f;
    out[8] = escapes / 36.0f;
    out[13] = shots.pip_loss / (12.0f * 36.0f);
    out[18] = mobility_fast(board, other_made) / 3600.0f;
    out[20] = entry_loss / (36.0f * (49.0f / 6.0f));
}

static void contact_aux(const Board& board, float* p_aux, float* o_aux,
                        const float* saved_opponent = nullptr, int previous_front_in_opponent_view = -1) {
    const Board flipped = flip(board);
#if defined(__AVX2__)
    const auto [player, opponent] = both_contact_point_masks(board);
#else
    const auto player = contact_point_masks(board);
    const auto opponent = contact_point_masks(flipped);
#endif
    init_escape_tables();
    contact_half(board, player, opponent.made, p_aux);
    contact_half(flipped, opponent, player.made, o_aux, saved_opponent, previous_front_in_opponent_view);
}

std::array<float, EXTENDED_CONTACT_INPUTS> compute_extended_contact_inputs(const Board& board) {
    static_assert(EXTENDED_CONTACT_INPUTS == 244, "Expected 244 inputs");
    std::array<float, EXTENDED_CONTACT_INPUTS> inputs;

    constexpr int P2_OFFSET = 122;

    for (int i = 0; i < 24; ++i) {
        int n = board[i + 1];

        const float* p_feat = point_feature_row(n);
        const int p0 = 4 * i;
        inputs[p0 + 0] = p_feat[0];
        inputs[p0 + 1] = p_feat[1];
        inputs[p0 + 2] = p_feat[2];
        inputs[p0 + 3] = p_feat[3];

        int m = -n;
        const float* m_feat = point_feature_row(m);
        const int p1 = 4 * i + P2_OFFSET;
        inputs[p1 + 0] = m_feat[0];
        inputs[p1 + 1] = m_feat[1];
        inputs[p1 + 2] = m_feat[2];
        inputs[p1 + 3] = m_feat[3];
    }

    contact_aux(board, inputs.data() + 96, inputs.data() + P2_OFFSET + 96);
    return inputs;
}

int board_input_deltas(const Board& board, const Board& base, const float* saved_inputs,
                       bool tesauro, int* indices, float* values) {
    unsigned changed = 0;
    for (int i = 0; i < 24; ++i)
        changed |= static_cast<unsigned>(board[i + 1] != base[i + 1]) << i;

    float p_aux[26], o_aux[26];
    if (tesauro) {
        p_aux[0] = board[25] / 2.0f;
        p_aux[1] = player_borne_off(board) / 15.0f;
        o_aux[0] = board[0] / 2.0f;
        o_aux[1] = opponent_borne_off(board) / 15.0f;
    } else {
        bool same_opponent = board[0] == base[0];
        for (unsigned bits = changed; same_opponent && bits; bits &= bits - 1) {
            const int point = highest_set_point(bits & (0u - bits));
            same_opponent = board[point] >= 0 && base[point] >= 0;
        }
        int previous_front = 0;
        if (same_opponent) {
            for (int point = 1; point <= 24; ++point) {
                if (base[point] > 0) {
                    previous_front = 25 - point;
                    break;
                }
            }
        }
        constexpr int opponent_aux_offset = EXTENDED_CONTACT_INPUTS / 2 + 24 * 4;
        contact_aux(board, p_aux, o_aux, same_opponent ? saved_inputs + opponent_aux_offset : nullptr,
                    previous_front);
    }
    int count = 0;
    auto append = [&](int index, float current) {
        const float delta = current - saved_inputs[index];
        if (delta != 0.0f && count < INPUT_DELTA_LIMIT) {
            indices[count] = index;
            values[count++] = delta;
        }
    };
    const int aux_count = tesauro ? 2 : 26;
    for (int side = 0; side < 2; ++side) {
        const int offset = side * (96 + aux_count);
        for (unsigned bits = changed; bits; bits &= bits - 1) {
            const unsigned first = bits & (0u - bits);
            const int point = highest_set_point(first);
            const int n = side ? -board[point] : board[point];
            const int previous = side ? -base[point] : base[point];
            if (n <= 0 && previous <= 0) continue;
            const float* row = point_feature_row(n);
            for (int k = 0; k < 4; ++k) append(offset + 4 * (point - 1) + k, row[k]);
        }
        const float* aux = side ? o_aux : p_aux;
        for (int k = 0; k < aux_count; ++k) append(offset + 96 + k, aux[k]);
    }
    return count;
}

void encode_board_from_base(const Board& board, const Board& base, const float* saved_inputs,
                            bool tesauro, float* out) {
    const int width = tesauro ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS;
    std::memcpy(out, saved_inputs, width * sizeof(float));

    unsigned changed = 0;
    for (int i = 0; i < 24; ++i)
        changed |= static_cast<unsigned>(board[i + 1] != base[i + 1]) << i;

    const int aux_count = tesauro ? 2 : 26;
    for (int side = 0; side < 2; ++side) {
        const int offset = side * (96 + aux_count);
        for (unsigned bits = changed; bits; bits &= bits - 1) {
            const unsigned first = bits & (0u - bits);
            const int point = highest_set_point(first);
            const int n = side ? -board[point] : board[point];
            const int previous = side ? -base[point] : base[point];
            if (n <= 0 && previous <= 0) continue;
            const float* row = point_feature_row(n);
            float* dst = out + offset + 4 * (point - 1);
            dst[0] = row[0];
            dst[1] = row[1];
            dst[2] = row[2];
            dst[3] = row[3];
        }
    }

    if (tesauro) {
        out[96] = board[25] / 2.0f;
        out[97] = player_borne_off(board) / 15.0f;
        out[194] = board[0] / 2.0f;
        out[195] = opponent_borne_off(board) / 15.0f;
    } else {
        bool same_opponent = board[0] == base[0];
        for (unsigned bits = changed; same_opponent && bits; bits &= bits - 1) {
            const int point = highest_set_point(bits & (0u - bits));
            same_opponent = board[point] >= 0 && base[point] >= 0;
        }
        int previous_front = 0;
        if (same_opponent) {
            for (int point = 1; point <= 24; ++point) {
                if (base[point] > 0) {
                    previous_front = 25 - point;
                    break;
                }
            }
        }
        constexpr int opponent_aux_offset = EXTENDED_CONTACT_INPUTS / 2 + 24 * 4;
        contact_aux(board, out + 96, out + opponent_aux_offset,
                    same_opponent ? saved_inputs + opponent_aux_offset : nullptr,
                    previous_front);
    }
}

} // namespace bgbot
