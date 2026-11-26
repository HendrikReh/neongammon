// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#pragma once

#include "cube.h"
#include <functional>

namespace bgbot {

struct CheckerPosition {
    Board board;
    int die1 = 0, die2 = 0;
    int cube_value = 1;
    CubeOwner owner = CubeOwner::CENTERED;
    MatchInfo match;
    int max_cube_value = 0;            // 0 = no cap
    std::vector<Board> force_boards;   // searched even if the filter drops them
};

struct ScoredMove {
    Board board;
    std::array<double, NUM_OUTPUTS> probs;
    double cubeless_equity;
    double cubeful_equity;
    bool searched;   // false: still on its 1-ply values
};

// Search caches are per-thread. A failed or interrupted request ends their
// lifetime just like a successfully scored one.
struct ClearSearchCacheOnExit {
    const MultiPlyStrategy& strategy;
    ~ClearSearchCacheOnExit() { strategy.clear_cache(); }
};

struct CubeEvaluation {
    CubeDecision decision{};
    std::array<float, NUM_OUTPUTS> probs{};
    float cube_x = -1.0f;
};

// Janowski calibration at 120 average pips and the change per pip, for money
// contact positions. Other depths and match play use the engine default.
struct ContactCubeLife {
    double value, pip_slope;
};

class AnalyzerSession {
public:
    AnalyzerSession(const Strategy& base, const MultiPlyStrategy& search,
                    MoveFilter filter, bool jacoby, bool beaver,
                    ContactCubeLife leaf, ContactCubeLife tree, bool filtered);

    std::vector<ScoredMove> score_checker_position(const CheckerPosition& pos,
        int candidate_threads, const std::function<void()>& check_budget = {}) const;

    bool can_share_checker_endpoints() const { return !filtered && strategy_nply.n_plies() == 3; }
    std::vector<std::vector<ScoredMove>> score_unfiltered_checker_batch(
        const std::vector<CheckerPosition>& positions, int threads) const;

    static constexpr std::array<ContactCubeLife, 2> money_contact_life{{
        {0.7375, 0.00175}, {0.7625, 0.00125}
    }};
    static constexpr std::array<int, 1> filtered_plies{{2}};
    static ContactCubeLife contact_life(int plies, const MatchInfo& match);

private:
    CubeInfo opponent_cube(const CheckerPosition& pos) const;
    const Strategy& strategy_1ply;
    const MultiPlyStrategy& strategy_nply;
    MoveFilter filter;
    bool jacoby, beaver;
    ContactCubeLife leaf, tree;
    bool filtered;
};

std::shared_ptr<MultiPlyStrategy> make_multipy(std::shared_ptr<Strategy> base, int n_plies,
    int filter_max_moves, float filter_threshold, bool full_depth_opponent,
    bool parallel_evaluate, int parallel_threads);

std::vector<ScoredMove> ranked_checker_moves_1ply(const CheckerPosition& pos,
    const Strategy& strategy, bool jacoby, bool beaver, ContactCubeLife life,
    const std::function<void()>& check_budget = {});

std::vector<ScoredMove> score_checker_endpoints(const Board& before,
    const std::vector<Board>& endpoints, const CubeInfo& opp_ci, const Strategy& strategy,
    int plies, int threads, const std::function<void()>& check_budget = {});

CubeEvaluation evaluate_cube_leaf(const Board& board, const CubeInfo& cube, const Strategy& base);
CubeEvaluation evaluate_cube_tree(const Board& board, const CubeInfo& cube,
    const std::shared_ptr<Strategy>& base, const BearoffDB* db,
    int plies, MoveFilter filter, int threads, bool decision_only, TwoPlyDetails* details = nullptr,
    const std::function<void()>& check_budget = {});

} // namespace bgbot
