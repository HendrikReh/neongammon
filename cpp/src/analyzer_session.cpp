// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/analyzer_session.h"
#include "bgbot/bearoff.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include "bgbot/pubeval.h"
#include <algorithm>
#include <numeric>

namespace bgbot {

AnalyzerSession::AnalyzerSession(const Strategy& base, const MultiPlyStrategy& search,
        MoveFilter filter, bool jacoby, bool beaver,
        ContactCubeLife leaf, ContactCubeLife tree, bool filtered)
    : strategy_1ply(base), strategy_nply(search), filter(filter), jacoby(jacoby), beaver(beaver),
      leaf(leaf), tree(tree), filtered(filtered) {}

ContactCubeLife AnalyzerSession::contact_life(int plies, const MatchInfo& match) {
    if (match.is_money() && plies >= 1 && plies <= static_cast<int>(money_contact_life.size()))
        return money_contact_life[plies - 1];
    return {-1.0, 0.0};
}

// The N-ply strategy the create_multipy* factories return: `base` under the
// move filter, with the PubEval prefilter that narrows opponent candidates
// before 1-ply scoring when a roll has more than 20.
std::shared_ptr<MultiPlyStrategy> make_multipy(
    std::shared_ptr<Strategy> base, int n_plies,
    int filter_max_moves, float filter_threshold, bool full_depth_opponent,
    bool parallel_evaluate, int parallel_threads)
{
    MoveFilter filter{filter_max_moves, filter_threshold};
    auto strat = std::make_shared<MultiPlyStrategy>(
        std::move(base), n_plies, filter, full_depth_opponent,
        parallel_evaluate, parallel_threads);
    static auto pubeval = std::make_shared<PubEval>();
    strat->set_move_prefilter(pubeval);
    return strat;
}

CubeEvaluation evaluate_cube_leaf(const Board& board, const CubeInfo& ci,
                                  const Strategy& base) {
    CubeEvaluation out;
    const bool race = is_race(board);
    out.probs = invert_probs(base.evaluate_probs(flip(board), race));
    clamp_probs_to_board(out.probs, board);
    auto [pp, op] = pip_counts(board);
    out.cube_x = resolve_cube_x(out.probs, ci, race, pp, op);
    out.decision = cube_decision_1ply(out.probs, ci, out.cube_x);
    return out;
}

// Both the public single-position and batch APIs call this same search.
CubeEvaluation evaluate_cube_tree(const Board& board, const CubeInfo& ci,
        const std::shared_ptr<Strategy>& base, const BearoffDB* db,
        int plies, MoveFilter filter, int threads, bool decision_only,
        TwoPlyDetails* details, const std::function<void()>& check_budget) {
    if (check_budget) check_budget();
    static PubEval pubeval;
    CubeEvaluation out;
    if (details) {
        out.decision = cube_decision_nply_with_details(
            board, ci, *base, plies, filter, threads, *details, &pubeval, check_budget);
    } else if (decision_only) {
        out.decision = cube_decision_nply(board, ci, *base, plies, filter, threads, &pubeval, check_budget);
    }
    if (!decision_only) {
        MultiPlyStrategy multipy(base, plies, filter, false, threads > 1, threads);
        const ClearSearchCacheOnExit cleanup{multipy};
        if (db && db->is_loaded()) multipy.set_bearoff_db(db);
        if (details) {
            const Board flipped = flip(board);
            out.probs = invert_probs(multipy.evaluate_probs(flipped, flipped));
        } else {
            out.decision = multipy.cube_decision_with_cubeless_probs(board, ci, &pubeval, out.probs, check_budget);
        }
    }
    return out;
}

// ======================== N-ply checker play ========================
//
// Single-position analysis, batch checker play and the mobile C ABI share
// the following N-ply search:
//
//  * At the depths in `filtered_plies` (Python passes CUBEFUL_FILTERED_PLIES):
//    1-ply cubeful scoring and the move filter (plus any forced boards); the
//    cubeless N-ply search of those moves with its stale-top-two promotion,
//    which decides which moves the cube tree sees; the cube-aware N-ply tree
//    for them, while the others keep their 1-ply values and 1-ply Janowski
//    equity; then the cubeful stale-top-two promotion, one cube-tree call
//    per promoted move.
//  * At every other depth: every legal move through the cube-aware tree.
//
// The cube tree is cubeful_probs_and_equity_nply on the opponent's pre-roll
// board, at the position's own cube (and cap) and routed from the decision's
// pre-move board (root routing). Cubeless equities are computed in double,
// in the order Python used, so every caller breaks ties the same way.

// The analyzer's _move_order_key: cubeful equity, then cubeless equity,
// then the board, so the ranking never depends on generation order.
static bool compare_scored_moves_by_cubeful(const ScoredMove& a, const ScoredMove& b) {
    if (a.cubeful_equity != b.cubeful_equity)
        return a.cubeful_equity > b.cubeful_equity;
    if (a.cubeless_equity != b.cubeless_equity)
        return a.cubeless_equity > b.cubeless_equity;
    return a.board < b.board;
}

// Shared 1-ply candidate scoring for BOTH 1-ply batch_checker_play and the
// 1-ply filter stage of score_checker_position. Evaluates all legal candidates
// in one batched NN pass (single-threaded per position via BLAS SGEMM on
// Accelerate), clamps impossible gammon/backgammon probabilities against each
// post-move candidate board, and computes cubeless + cubeful equity in double.
// Returns candidates in generation order; callers sort as needed.
static std::vector<ScoredMove> score_checker_1ply(
        const CheckerPosition& pos,
        const Strategy& strategy_1ply,
        bool jacoby,
        bool beaver,
        float contact_x_1ply,
        float contact_x_pip_slope_1ply,
        const std::function<void()>& check_budget) {
    auto candidates = possible_boards(pos.board, pos.die1, pos.die2);
    std::vector<ScoredMove> out;
    if (candidates.empty()) return out;

    if (check_budget) check_budget();
    const size_t n = candidates.size();
    std::vector<std::array<float, NUM_OUTPUTS>> probs_per(n);
    strategy_1ply.batch_evaluate_candidates_equity_probs(
        candidates, pos.board, nullptr, probs_per.data());

    const bool money = pos.match.is_money();
    const bool dead_cube = pos.max_cube_value > 0 && pos.cube_value >= pos.max_cube_value;
    const CubeInfo ci{pos.cube_value, pos.owner, pos.match, -1.0f,
                      money && jacoby, money && beaver,
                      pos.max_cube_value,
                      money ? contact_x_1ply : -1.0f,
                      money ? contact_x_pip_slope_1ply : 0.0f};

    out.reserve(n);
    for (size_t j = 0; j < n; ++j) {
        if (check_budget) check_budget();
        auto probs = probs_per[j];
        clamp_probs_to_board(probs, candidates[j]);
        std::array<double, NUM_OUTPUTS> p;
        std::copy(probs.begin(), probs.end(), p.begin());
        const double cl = NeuralNetwork::compute_equity(probs);
        double cf;
        if (dead_cube) {
            cf = money
                ? cl
                : static_cast<double>(mwc2eq(
                    cubeless_mwc(probs, pos.match.away1, pos.match.away2,
                                 pos.cube_value, pos.match.is_crawford),
                    pos.match.away1, pos.match.away2, pos.cube_value,
                    pos.match.is_crawford));
        } else {
            auto [pp, op] = pip_counts(candidates[j]);
            cf = static_cast<double>(
                cl2cf(probs, ci, resolve_cube_x(probs, ci, is_race(candidates[j]), pp, op)));
        }
        out.push_back({candidates[j], p, cl, cf, false});
    }
    return out;
}

// Probs, cubeless and cubeful equity of a post-move board from the
// cube-aware N-ply tree.
static void apply_cube_tree(ScoredMove& m, const CubefulProbsAndEquity& r) {
    const auto& o = r.probs;
    m.probs = {1.0 - o[0], o[3], o[4], o[1], o[2]};
    m.cubeless_equity = 2.0 * m.probs[0] - 1.0
                        + m.probs[1] - m.probs[3]
                        + m.probs[2] - m.probs[4];
    m.cubeful_equity = -r.equity;
    m.searched = true;
}

std::vector<ScoredMove> ranked_checker_moves_1ply(const CheckerPosition& pos,
        const Strategy& strategy, bool jacoby, bool beaver, ContactCubeLife life,
        const std::function<void()>& check_budget) {
    auto moves = score_checker_1ply(pos, strategy, jacoby, beaver, life.value, life.pip_slope,
                                    check_budget);
    std::sort(moves.begin(), moves.end(), compare_scored_moves_by_cubeful);
    return moves;
}

std::vector<ScoredMove> score_checker_endpoints(
        const Board& before, const std::vector<Board>& endpoints, const CubeInfo& opp_ci,
        const Strategy& strategy, int plies, int threads, const std::function<void()>& check_budget) {
    const auto scores = cubeful_checker_endpoints_nply(before, endpoints, opp_ci, strategy,
                                                      plies, threads, check_budget);
    std::vector<ScoredMove> out(endpoints.size());
    for (size_t i = 0; i < endpoints.size(); ++i) {
        out[i].board = endpoints[i];
        apply_cube_tree(out[i], scores[i]);
    }
    std::sort(out.begin(), out.end(), compare_scored_moves_by_cubeful);
    return out;
}

CubeInfo AnalyzerSession::opponent_cube(const CheckerPosition& pos) const {
    const bool money = pos.match.is_money();
    return {pos.cube_value, flip_owner(pos.owner), pos.match.flip(), -1.0f,
            money && jacoby, money && beaver, pos.max_cube_value,
            money ? static_cast<float>(tree.value) : -1.0f,
            money ? static_cast<float>(tree.pip_slope) : 0.0f};
}

// Every 3-ply endpoint is independent. Queue endpoints across the whole batch
// so idle lanes can start the next roll while a long double is still running.
// Keep the root for model routing and write each score into its original row.
std::vector<std::vector<ScoredMove>> AnalyzerSession::score_unfiltered_checker_batch(
        const std::vector<CheckerPosition>& positions, int threads) const {
    std::vector<std::vector<ScoredMove>> results(positions.size());
    std::vector<CubeInfo> cubes;
    std::vector<std::pair<size_t, size_t>> work;
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto& pos = positions[i];
        cubes.push_back(opponent_cube(pos));
        const auto endpoints = possible_boards(pos.board, pos.die1, pos.die2);
        results[i].resize(endpoints.size());
        for (size_t j = 0; j < endpoints.size(); ++j) {
            results[i][j].board = endpoints[j];
        }
    }
    // Start wide rolls first; shorter rolls can fill lanes as they finish.
    std::vector<size_t> order(positions.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return results[a].size() > results[b].size();
    });
    for (size_t row : order)
        for (size_t move = 0; move < results[row].size(); ++move)
            work.emplace_back(row, move);
    const int count = static_cast<int>(work.size());
    const int lanes = std::min(count, threads);
    if (lanes > 0) {
        multipy_cooperative_for(count, lanes, [&](int k) {
            const auto [row, move] = work[k];
            auto& m = results[row][move];
            const auto score = cubeful_probs_and_equity_nply(
                flip(m.board), cubes[row], strategy_1ply, strategy_nply.n_plies(),
                MoveFilters::TINY, threads / lanes, nullptr, &positions[row].board);
            apply_cube_tree(m, score);
        });
    }
    for (auto& moves : results)
        std::sort(moves.begin(), moves.end(), compare_scored_moves_by_cubeful);
    return results;
}

// The ranked move list of one position. `candidate_threads` > 1 spreads the
// cube-tree calls of the searched moves over the shared pool. Batch callers
// can also reserve a share of their thread budget for this position.
std::vector<ScoredMove> AnalyzerSession::score_checker_position(
        const CheckerPosition& pos, int candidate_threads,
        const std::function<void()>& check_budget) const {
    const int n_plies = strategy_nply.n_plies();
    const CubeInfo opp_ci = opponent_cube(pos);

    auto cube_tree = [&](ScoredMove& m) {
        if (check_budget) check_budget();
        const int roll_threads = candidate_threads > 1 ? std::min(candidate_threads, 21) : 1;
        auto r = cubeful_probs_and_equity_nply(
            flip(m.board), opp_ci, strategy_1ply, n_plies, filter,
            roll_threads, /*move_filter=*/nullptr, &pos.board);
        apply_cube_tree(m, r);
    };
    // The cube tree for several moves, using dynamic work-stealing, cooperative
    // roll splitting and shared 2-ply subtrees at 4-ply when threads permit.
    auto cube_trees = [&](std::vector<ScoredMove*>& moves) {
        if (moves.empty()) return;
        std::vector<Board> endpoints;
        endpoints.reserve(moves.size());
        for (const auto* m : moves) endpoints.push_back(m->board);
        auto res = cubeful_checker_endpoints_nply(
            pos.board, endpoints, opp_ci, strategy_1ply, n_plies,
            candidate_threads, check_budget);
        for (size_t i = 0; i < moves.size(); ++i) {
            apply_cube_tree(*moves[i], res[i]);
        }
    };

    if (!filtered) {
        return score_checker_endpoints(pos.board, possible_boards(pos.board, pos.die1, pos.die2),
                                       opp_ci, strategy_1ply, n_plies, candidate_threads, check_budget);
    }

    // 1-ply cubeful scoring (shared with 1-ply batch_checker_play).
    std::vector<ScoredMove> out = score_checker_1ply(
        pos, strategy_1ply, jacoby, beaver,
        leaf.value, leaf.pip_slope, check_budget);
    if (out.empty()) return out;

    // The filter, on the 1-ply cubeful values sorted stably as Python sorted
    // them (equal values keep generation order).
    std::stable_sort(out.begin(), out.end(),
        [](const ScoredMove& a, const ScoredMove& b) {
            return a.cubeful_equity > b.cubeful_equity;
        });
    const double best = out[0].cubeful_equity;
    size_t keep = 0;
    while (keep < out.size() && keep < static_cast<size_t>(filter.max_moves)
           && best - out[keep].cubeful_equity < filter.threshold)
        ++keep;
    if (keep == 0) keep = 1;

    // The cubeless N-ply pass: survivors and forced boards at N-ply, the rest
    // on 1-ply cubeless values, ranked by cubeless equity then board, stale
    // top two promoted. Only which moves it searched is kept; the cube tree
    // replaces their values.
    struct Cubeless { size_t idx; double equity; bool searched; };
    std::vector<Cubeless> cubeless(out.size());
    auto nply_equity = [&](size_t j) {
        if (check_budget) check_budget();
        return NeuralNetwork::compute_equity(
            strategy_nply.evaluate_probs(out[j].board, pos.board));
    };
    auto forced = [&](size_t j) {
        return std::find(pos.force_boards.begin(), pos.force_boards.end(),
                         out[j].board) != pos.force_boards.end();
    };
    for (size_t j = 0; j < out.size(); ++j) {
        cubeless[j] = (j < keep || forced(j)) ? Cubeless{j, nply_equity(j), true}
                                              : Cubeless{j, out[j].cubeless_equity, false};
    }
    auto by_cubeless = [&](const Cubeless& a, const Cubeless& b) {
        if (a.equity != b.equity) return a.equity > b.equity;
        return out[a.idx].board < out[b.idx].board;
    };
    std::sort(cubeless.begin(), cubeless.end(), by_cubeless);
    for (;;) {
        Cubeless* stale = nullptr;
        for (size_t k = 0; k < cubeless.size() && k < 2; ++k)
            if (!cubeless[k].searched) { stale = &cubeless[k]; break; }
        if (!stale) break;
        stale->equity = nply_equity(stale->idx);
        stale->searched = true;
        std::sort(cubeless.begin(), cubeless.end(), by_cubeless);
    }

    std::vector<ScoredMove*> searched;
    for (const auto& c : cubeless)
        if (c.searched) searched.push_back(&out[c.idx]);
    cube_trees(searched);
    std::sort(out.begin(), out.end(), compare_scored_moves_by_cubeful);

    // No 1-ply value may sit in the top two: a pruned move's 1-ply equity is
    // not on the searched moves' scale, and when the deeper search lowers
    // every searched move, the pruned one would win on a number the search
    // never checked (snake benchmark, 2026-09-04: 312 of 795 cubeless 2-ply
    // picks). Promote whichever of #1/#2 is stale, re-sort, repeat.
    for (;;) {
        ScoredMove* stale = nullptr;
        for (size_t k = 0; k < out.size() && k < 2; ++k)
            if (!out[k].searched) { stale = &out[k]; break; }
        if (!stale) break;
        cube_tree(*stale);
        std::sort(out.begin(), out.end(), compare_scored_moves_by_cubeful);
    }
    return out;
}

} // namespace bgbot
