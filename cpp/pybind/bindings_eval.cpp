// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bindings_common.h"
#include <thread>
#include "bgbot/analyzer_session.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/multipy.h"
#include "bgbot/cube.h"
#include "bgbot/bearoff.h"
#include "bgbot/neural_net.h"
#include "bgbot/pubeval.h"

namespace bgbot::python {

namespace {

std::shared_ptr<Strategy> with_bearoff_db(std::shared_ptr<Strategy> strategy,
                                                 const BearoffDB* db) {
    if (!db || !db->is_loaded()) return strategy;
    if (strategy->bearoff_layer().db == db) return strategy;
    return std::make_shared<BearoffStrategy>(std::move(strategy), db);
}

void validate_search_depth(int plies) {
    if (plies < 1) throw std::invalid_argument("n_plies must be at least 1");
}

py::list player_rolls_to_list(const std::vector<PlayerRollDetail>& rolls) {
    py::list result;
    for (const auto& pr : rolls) {
        py::dict d;
        d["die1"] = pr.die1;
        d["die2"] = pr.die2;
        d["checkers"] = board_to_list(pr.post_move_board);
        d["cubeful_equity"] = pr.cubeful_equity;
        d["probs"] = pr.probs;
        if (pr.is_terminal) {
            // Terminal: no opponent_rolls field
        } else if (pr.opponent_dp) {
            // Opponent D/P: no opponent_rolls field
            d["opponent_dp"] = true;
        } else if (pr.opponent_rolls.empty()) {
            // 2-ply analysis: opponent-roll level not captured — omit the field
        } else {
            py::list opp_list;
            for (const auto& opp : pr.opponent_rolls) {
                py::dict od;
                od["die1"] = opp.die1;
                od["die2"] = opp.die2;
                od["checkers"] = board_to_list(opp.post_move_board);
                od["cubeful_equity"] = opp.cubeful_equity;
                od["probs"] = opp.probs;
                opp_list.append(od);
            }
            d["opponent_rolls"] = opp_list;
        }
        result.append(d);
    }
    return result;
}

// Convert TwoPlyDetails to a Python dict with "nd" and "dt" sections
py::dict two_ply_details_to_dict(const TwoPlyDetails& details) {
    py::dict result;
    result["nd"] = player_rolls_to_list(details.nd_player_rolls);
    result["dt"] = player_rolls_to_list(details.dt_player_rolls);
    return result;
}

bool checker_root_is_filtered(int plies, const std::vector<int>& filtered_plies) {
    return std::find(filtered_plies.begin(), filtered_plies.end(), plies) != filtered_plies.end();
}

void validate_batch_move_filter(int plies, int max_moves, float threshold, bool filtered_root) {
    if (plies >= 3 && !filtered_root &&
        (max_moves != MoveFilters::TINY.max_moves || threshold != MoveFilters::TINY.threshold)) {
        throw std::invalid_argument(std::to_string(plies) + "-ply batch search has no root move filter. "
            "Use the default filter_max_moves=5 and filter_threshold=0.08.");
    }
}

AnalyzerSession make_checker_search(
        const Strategy& strategy_1ply, const MultiPlyStrategy& strategy_nply,
        int filter_max_moves, float filter_threshold, bool jacoby, bool beaver,
        float contact_x_1ply, float contact_x_pip_slope_1ply,
        float contact_x_nply, float contact_x_pip_slope_nply,
        bool filtered_root) {
    return AnalyzerSession{
        strategy_1ply, strategy_nply, MoveFilter{filter_max_moves, filter_threshold}, jacoby, beaver,
        {contact_x_1ply, contact_x_pip_slope_1ply}, {contact_x_nply, contact_x_pip_slope_nply},
        filtered_root};
}

int batch_thread_budget(int requested) {
    return requested > 0 ? requested : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
}

template <typename Fn>
void run_parallel_for(int n, int n_threads, Fn&& fn) {
    if (n <= 0) return;
    n_threads = std::min(batch_thread_budget(n_threads), n);
    if (n <= 1 || n_threads <= 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    multipy_cooperative_for(n, n_threads, std::forward<Fn>(fn));
}

// Spend the budget on rows or on the candidates/rolls of a row. Deep
// checker work is highly skewed (one double can have over 100 endpoints).
// For row-parallel batches, surplus lanes go to the individual searches.
template <typename Fn>
void run_search_batch(int rows, int threads, bool within_row, Fn&& evaluate) {
    if (rows <= 0) return;
    const int budget = batch_thread_budget(threads);
    const int outer = within_row ? 1 : std::min(rows, budget);
    const int inner = budget / outer;
    const int spare = budget % outer;
    run_parallel_for(rows, outer, [&](int i) { evaluate(i, inner + (i < spare)); });
}

py::dict cube_evaluation_dict(const CubeEvaluation& evaluation, bool decision_only) {
    py::dict result;
    if (!decision_only) {
        result["probs"] = evaluation.probs;
        result["cubeless_equity"] = cubeless_equity(evaluation.probs);
    }
    const auto& cd = evaluation.decision;
    result["equity_nd"] = cd.equity_nd;
    result["equity_dt"] = cd.equity_dt;
    result["equity_dp"] = cd.equity_dp;
    result["should_double"] = cd.should_double;
    result["should_take"] = cd.should_take;
    result["optimal_equity"] = cd.optimal_equity;
    result["is_beaver"] = cd.is_beaver;
    return result;
}

py::list scored_moves_to_list(const std::vector<ScoredMove>& moves, int n_plies) {
    const std::string nply_label = std::to_string(n_plies) + "-ply";
    const double best_eq = moves.empty() ? 0.0 : moves[0].cubeful_equity;
    py::list out;
    for (const auto& m : moves) {
        py::dict d;
        d["board"] = board_to_list(m.board);
        d["equity"] = m.cubeful_equity;
        d["cubeless_equity"] = m.cubeless_equity;
        d["probs"] = m.probs;
        d["equity_diff"] = m.cubeful_equity - best_eq;
        d["eval_level"] = m.searched ? nply_label : std::string("1-ply");
        out.append(d);
    }
    return out;
}

CheckerPosition checker_position_from_dict(const py::dict& inp) {
    CheckerPosition p;
    p.board = list_to_board(inp["board"].cast<std::vector<int>>());
    p.die1 = inp["die1"].cast<int>();
    p.die2 = inp["die2"].cast<int>();
    validate_die(p.die1);
    validate_die(p.die2);
    p.cube_value = inp["cube_value"].cast<int>();
    p.owner = inp["cube_owner"].cast<CubeOwner>();
    if (inp.contains("away1") || inp.contains("away2")) {
        p.match.away1 = inp.contains("away1") ? inp["away1"].cast<int>() : 0;
        p.match.away2 = inp.contains("away2") ? inp["away2"].cast<int>() : 0;
        validate_met_away_scores(p.match.away1, p.match.away2);
        if (inp.contains("is_crawford"))
            p.match.is_crawford = inp["is_crawford"].cast<bool>();
    }
    if (inp.contains("max_cube_value"))
        p.max_cube_value = inp["max_cube_value"].cast<int>();
    if (inp.contains("force_boards"))
        for (const auto& b : inp["force_boards"].cast<std::vector<std::vector<int>>>())
            p.force_boards.push_back(list_to_board(b));
    return p;
}

// 1-ply batch checker play: score_checker_1ply per position, the positions
// spread cooperatively over n_threads workers.
py::list batch_checker_play_1ply(
        py::list inputs,
        const Strategy& strategy_1ply,
        int filter_max_moves,
        float filter_threshold,
        int n_threads,
        bool jacoby,
        bool beaver,
        float contact_x_1ply,
        float contact_x_pip_slope_1ply) {
    (void)filter_max_moves;
    (void)filter_threshold;
    const int n = static_cast<int>(py::len(inputs));
    if (n == 0) return py::list();
    std::vector<CheckerPosition> parsed(n);
    for (int i = 0; i < n; ++i) parsed[i] = checker_position_from_dict(inputs[i].cast<py::dict>());
    std::vector<std::vector<ScoredMove>> results(n);
    {
        py::gil_scoped_release release;
        run_parallel_for(n, n_threads, [&](int i) {
            results[i] = ranked_checker_moves_1ply(parsed[i], strategy_1ply, jacoby, beaver,
                                                 {contact_x_1ply, contact_x_pip_slope_1ply});
        });
    }
    py::list out;
    for (int i = 0; i < n; ++i) {
        py::dict pos_result;
        pos_result["moves"] = scored_moves_to_list(results[i], 1);
        out.append(pos_result);
    }
    return out;
}

// N-ply batch checker play: score_checker_position per position, the
// thread budget shared between positions and their candidate searches.
py::list batch_checker_play_nply(
        py::list inputs,
        const Strategy& strategy_1ply,
        const std::shared_ptr<MultiPlyStrategy>& strategy_nply,
        int filter_max_moves,
        float filter_threshold,
        int n_threads,
        bool jacoby,
        bool beaver,
        float contact_x_1ply,
        float contact_x_pip_slope_1ply,
        float contact_x_nply,
        float contact_x_pip_slope_nply,
        const std::vector<int>& filtered_plies) {
    require_non_null(strategy_nply, "strategy_nply");
    const int plies = strategy_nply->n_plies();
    const bool filtered_root = checker_root_is_filtered(plies, filtered_plies);
    validate_batch_move_filter(plies, filter_max_moves, filter_threshold, filtered_root);
    const int n = static_cast<int>(py::len(inputs));
    if (n == 0) return py::list();
    std::vector<CheckerPosition> parsed(n);
    for (int i = 0; i < n; ++i) parsed[i] = checker_position_from_dict(inputs[i].cast<py::dict>());
    const AnalyzerSession session = make_checker_search(
        strategy_1ply, *strategy_nply, filter_max_moves, filter_threshold, jacoby, beaver,
        contact_x_1ply, contact_x_pip_slope_1ply, contact_x_nply, contact_x_pip_slope_nply,
        filtered_root);
    std::vector<std::vector<ScoredMove>> results(n);
    {
        py::gil_scoped_release release;
        const int threads = batch_thread_budget(n_threads);
        if (n > 1 && threads > 1 && session.can_share_checker_endpoints()) {
            results = session.score_unfiltered_checker_batch(parsed, threads);
        } else {
            // Keep 4-ply candidates together for their shared subtree cache.
            run_search_batch(n, threads, plies >= 3, [&](int i, int inner) {
                results[i] = session.score_checker_position(parsed[i], inner);
            });
        }
        strategy_nply->clear_cache();
    }
    py::list out;
    for (int i = 0; i < n; ++i) {
        py::dict pos_result;
        pos_result["moves"] = scored_moves_to_list(results[i], strategy_nply->n_plies());
        out.append(pos_result);
    }
    return out;
}

py::list batch_evaluate_positions_impl(
        py::list positions,
        const Strategy& strategy,
        MultiPlyStrategy* multipy_to_clear,
        int n_threads,
        bool jacoby,
        bool beaver,
        float contact_x,
        float contact_x_pip_slope) {
    struct PosInput {
        Board board;
        int cube_value;
        CubeOwner owner;
        MatchInfo match;
        int max_cube_value = 0;
    };
    const int n = static_cast<int>(py::len(positions));
    if (n == 0) return py::list();
    std::vector<PosInput> inputs(n);
    for (int i = 0; i < n; ++i) {
        py::tuple pos = positions[i].cast<py::tuple>();
        auto checkers = pos[0].cast<std::vector<int>>();
        inputs[i].board = list_to_board(checkers);
        inputs[i].cube_value = pos[1].cast<int>();
        inputs[i].owner = pos[2].cast<CubeOwner>();
        if (py::len(pos) >= 6) {
            inputs[i].match.away1 = pos[3].cast<int>();
            inputs[i].match.away2 = pos[4].cast<int>();
            validate_met_away_scores(inputs[i].match.away1, inputs[i].match.away2);
            inputs[i].match.is_crawford = pos[5].cast<bool>();
            if (py::len(pos) >= 7) {
                inputs[i].max_cube_value = pos[6].cast<int>();
            }
        }
    }

    struct PosResult {
        std::array<float, NUM_OUTPUTS> probs;
        float cubeless_equity;
        float cubeful_equity;
        CubeDecision cube_decision;
    };
    std::vector<PosResult> results(n);

    {
        py::gil_scoped_release release;
        run_parallel_for(n, n_threads, [&](int i) {
            const auto& inp = inputs[i];
            auto& out = results[i];

            // Pre-roll probs: flip -> evaluate -> invert -> clamp against board
            Board flipped = flip(inp.board);
            bool race = is_race(inp.board);
            auto post_probs = strategy.evaluate_probs(flipped, race);
            out.probs = invert_probs(post_probs);
            clamp_probs_to_board(out.probs, inp.board);
            out.cubeless_equity = cubeless_equity(out.probs);

            // Cubeful equity via Janowski (unified money/match)
            const bool money = inp.match.is_money();
            auto [_pp, _op] = pip_counts(inp.board);
            CubeInfo ci{inp.cube_value, inp.owner, inp.match, -1.0f,
                        money && jacoby, money && beaver,
                        inp.max_cube_value, money ? contact_x : -1.0f,
                        money ? contact_x_pip_slope : 0.0f};
            float x = resolve_cube_x(out.probs, ci, race, _pp, _op);
            out.cubeful_equity = cl2cf(out.probs, ci, x);

            // Cube decision
            out.cube_decision = cube_decision_1ply(out.probs, ci, x);
        });
        if (multipy_to_clear) multipy_to_clear->clear_cache();
    }

    py::list out;
    for (int i = 0; i < n; ++i) {
        const auto& r = results[i];
        const auto& cd = r.cube_decision;
        py::dict d;
        d["probs"] = r.probs;
        d["cubeless_equity"] = r.cubeless_equity;
        d["cubeful_equity"] = r.cubeful_equity;
        d["equity_nd"] = cd.equity_nd;
        d["equity_dt"] = cd.equity_dt;
        d["equity_dp"] = cd.equity_dp;
        d["should_double"] = cd.should_double;
        d["should_take"] = cd.should_take;
        d["is_beaver"] = cd.is_beaver;
        const char* action;
        if (!cd.should_double) {
            action = "No Double";
        } else if (cd.is_beaver) {
            action = "Double/Beaver";
        } else if (cd.should_take) {
            action = "Double/Take";
        } else {
            action = "Double/Pass";
        }
        d["optimal_action"] = action;
        out.append(d);
    }
    return out;
}

py::list batch_evaluate_post_move_impl(
        py::list positions,
        const Strategy& strategy,
        MultiPlyStrategy* multipy_to_clear,
        int n_threads,
        bool jacoby,
        float contact_x,
        float contact_x_pip_slope) {
    struct PosInput {
        Board board;
        CubeOwner owner;
        int cube_value;
        MatchInfo match;
        int max_cube_value = 0;
    };
    const int n = static_cast<int>(py::len(positions));
    if (n == 0) return py::list();
    std::vector<PosInput> inputs(n);
    for (int i = 0; i < n; ++i) {
        py::tuple pos = positions[i].cast<py::tuple>();
        auto checkers = pos[0].cast<std::vector<int>>();
        inputs[i].board = list_to_board(checkers);
        inputs[i].owner = pos[1].cast<CubeOwner>();
        inputs[i].cube_value = 1;
        const size_t plen = py::len(pos);
        if (plen >= 3) {
            inputs[i].cube_value = pos[2].cast<int>();
        }
        if (plen >= 5) {
            inputs[i].match.away1 = pos[3].cast<int>();
            inputs[i].match.away2 = pos[4].cast<int>();
            validate_met_away_scores(inputs[i].match.away1, inputs[i].match.away2);
            if (plen >= 6)
                inputs[i].match.is_crawford = pos[5].cast<bool>();
            if (plen >= 7)
                inputs[i].max_cube_value = pos[6].cast<int>();
        }
    }

    struct PosResult {
        std::array<float, NUM_OUTPUTS> probs;
        float cubeless_equity;
        float cubeful_equity;
    };
    std::vector<PosResult> results(n);

    {
        py::gil_scoped_release release;
        run_parallel_for(n, n_threads, [&](int i) {
            const auto& inp = inputs[i];
            auto& out = results[i];

            // Post-move: evaluate directly and clamp impossible gammons/backgammons
            bool race = is_race(inp.board);
            out.probs = strategy.evaluate_probs(inp.board, inp.board);
            clamp_probs_to_board(out.probs, inp.board);
            out.cubeless_equity = NeuralNetwork::compute_equity(out.probs);

            // Cubeful equity via Janowski (unified money/match; cl2cf handles dead cubes)
            const bool money = inp.match.is_money();
            auto [_pp, _op] = pip_counts(inp.board);
            CubeInfo ci{inp.cube_value, inp.owner, inp.match, -1.0f,
                        money && jacoby, false,
                        inp.max_cube_value, money ? contact_x : -1.0f,
                        money ? contact_x_pip_slope : 0.0f};
            float x = resolve_cube_x(out.probs, ci, race, _pp, _op);
            out.cubeful_equity = cl2cf(out.probs, ci, x);
        });
        if (multipy_to_clear) multipy_to_clear->clear_cache();
    }

    py::list out;
    for (int i = 0; i < n; ++i) {
        const auto& r = results[i];
        py::dict d;
        d["probs"] = r.probs;
        d["cubeless_equity"] = r.cubeless_equity;
        d["cubeful_equity"] = r.cubeful_equity;
        out.append(d);
    }
    return out;
}

} // namespace

void bind_eval(py::module_& m) {
    py::dict money_contact_life, money_contact_slope;
    for (size_t i = 0; i < AnalyzerSession::money_contact_life.size(); ++i) {
        auto calibration = AnalyzerSession::money_contact_life[i];
        money_contact_life[py::int_(i + 1)] = calibration.value;
        money_contact_slope[py::int_(i + 1)] = calibration.pip_slope;
    }
    m.attr("MONEY_CONTACT_CUBE_LIFE") = money_contact_life;
    m.attr("MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE") = money_contact_slope;
    m.attr("CUBEFUL_FILTERED_PLIES") = AnalyzerSession::filtered_plies;
    m.def("_validate_batch_move_filter", &validate_batch_move_filter);

    // ======================== Doubling Cube (Janowski) ========================

    py::enum_<CubeOwner>(m, "CubeOwner")
        .value("CENTERED", CubeOwner::CENTERED)
        .value("PLAYER", CubeOwner::PLAYER)
        .value("OPPONENT", CubeOwner::OPPONENT);

    py::class_<CubeInfo>(m, "CubeInfo")
        .def(py::init<>())
        .def(py::init([](int value, CubeOwner owner, int away1, int away2, bool crawford,
                         bool jacoby, bool beaver, int max_cube, float contact_x, float slope) {
            validate_met_away_scores(away1, away2);
            return CubeInfo{value, owner, {away1, away2, crawford}, -1.0f, jacoby, beaver,
                            max_cube, contact_x, slope};
        }), py::arg("cube_value") = 1, py::arg("owner") = CubeOwner::CENTERED,
            py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
            py::arg("jacoby") = false, py::arg("beaver") = false, py::arg("max_cube_value") = 0,
            py::arg("contact_x") = -1.0f, py::arg("contact_x_pip_slope") = 0.0f)
        .def_readwrite("cube_value", &CubeInfo::cube_value)
        .def_readwrite("owner", &CubeInfo::owner);

    py::class_<CubeDecision>(m, "CubeDecision")
        .def_readonly("equity_nd", &CubeDecision::equity_nd)
        .def_readonly("equity_dt", &CubeDecision::equity_dt)
        .def_readonly("equity_dp", &CubeDecision::equity_dp)
        .def_readonly("should_double", &CubeDecision::should_double)
        .def_readonly("should_take", &CubeDecision::should_take)
        .def_readonly("optimal_equity", &CubeDecision::optimal_equity);

    m.def("cubeless_equity", [](const std::array<float, NUM_OUTPUTS>& probs) {
        return cubeless_equity(probs);
    }, "Compute cubeless equity from 5 probabilities",
       py::arg("probs"));

    // --- Match equity utilities ---
    m.def("_validate_met_away_scores", &validate_met_away_scores);

    m.def("get_met", [](int away1, int away2, bool is_crawford) {
        return get_met(away1, away2, is_crawford);
    }, "MET lookup: MWC for player needing away1 points vs opponent needing away2.",
       py::arg("away1"), py::arg("away2"), py::arg("is_crawford") = false);

    m.def("_get_met_after", &get_met_after,
          "MWC after scoring points from the given current match state.",
          py::arg("away1"), py::arg("away2"), py::arg("points"),
          py::arg("player_wins"), py::arg("is_crawford"));

    m.def("can_double_match", [](int away1, int away2, int cube_value,
                                  CubeOwner owner, bool is_crawford, int max_cube_value) {
        validate_met_away_scores(away1, away2);
        CubeInfo cube{cube_value, owner, {away1, away2, is_crawford}};
        cube.max_cube_value = max_cube_value;
        return can_double(cube);
    }, "Can the player on roll double at this match score or in money play (0, 0)?",
       py::arg("away1"), py::arg("away2"), py::arg("cube_value"),
       py::arg("owner"), py::arg("is_crawford"), py::arg("max_cube_value") = 0);

    m.def("cubeless_mwc", [](const std::array<float, 5>& probs,
                              int away1, int away2, int cube_value, bool is_crawford) {
        return cubeless_mwc(probs, away1, away2, cube_value, is_crawford);
    }, "Cubeless MWC from NN probs and match state.",
       py::arg("probs"), py::arg("away1"), py::arg("away2"),
       py::arg("cube_value") = 1, py::arg("is_crawford") = false);

    m.def("eq2mwc", [](float equity, int away1, int away2, int cube_value, bool is_crawford) {
        return eq2mwc(equity, away1, away2, cube_value, is_crawford);
    }, "Convert cubeless equity to MWC.",
       py::arg("equity"), py::arg("away1"), py::arg("away2"),
       py::arg("cube_value") = 1, py::arg("is_crawford") = false);

    m.def("mwc2eq", [](float mwc, int away1, int away2, int cube_value, bool is_crawford) {
        return mwc2eq(mwc, away1, away2, cube_value, is_crawford);
    }, "Convert MWC to cubeless equity.",
       py::arg("mwc"), py::arg("away1"), py::arg("away2"),
       py::arg("cube_value") = 1, py::arg("is_crawford") = false);

    m.def("cl2cf_money", [](const std::array<float, NUM_OUTPUTS>& probs,
                             CubeOwner owner, float cube_x, bool jacoby_active) {
        return cl2cf_money(probs, owner, cube_x, jacoby_active);
    }, "Cubeless-to-cubeful conversion (Janowski, money game). Returns cubeful equity normalized to cube=1.",
       py::arg("probs"), py::arg("owner"), py::arg("cube_x"),
       py::arg("jacoby_active") = false);

    // Unified cl2cf: money or match play
    m.def("cl2cf", [](const std::array<float, NUM_OUTPUTS>& probs,
                       int cube_value, CubeOwner owner, float cube_x,
                       int away1, int away2, bool is_crawford, bool jacoby) {
        validate_met_away_scores(away1, away2);
        CubeInfo ci{cube_value, owner, {away1, away2, is_crawford}, -1.0f, jacoby};
        return cl2cf(probs, ci, cube_x);
    }, "Cubeless-to-cubeful conversion (money or match). Returns cubeful equity.",
       py::arg("probs"), py::arg("cube_value"), py::arg("owner"), py::arg("cube_x"),
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("jacoby") = true);

    // Live-cube cash points for match play (GNUbg GetPoints recursion).
    // Returns (player_cp, opp_cp) each in the respective player's P(win) space.
    // Player's live-cube take point = 1 - opp_cp.
    m.def("match_live_cash_points", [](int away1, int away2, int cube_value,
                                        bool is_crawford,
                                        float rG0, float rBG0,
                                        float rG1, float rBG1) {
        float player_cp = 0.0f, opp_cp = 0.0f;
        get_match_points(away1, away2, cube_value, is_crawford,
                         rG0, rBG0, rG1, rBG1, player_cp, opp_cp);
        return std::make_tuple(player_cp, opp_cp);
    }, "Live-cube cash points for match play (recursive GetPoints). "
       "Returns (player_cp, opp_cp) in each player's P(win) space.",
       py::arg("away1"), py::arg("away2"), py::arg("cube_value"),
       py::arg("is_crawford") = false,
       py::arg("rG0") = 0.0f, py::arg("rBG0") = 0.0f,
       py::arg("rG1") = 0.0f, py::arg("rBG1") = 0.0f);

    m.def("contact_cube_x",
          [](float contact_x, float contact_x_pip_slope,
             int player_pips, int opponent_pips) {
              CubeInfo ci;
              ci.contact_x = contact_x;
              ci.contact_x_pip_slope = contact_x_pip_slope;
              return contact_cube_x(ci, player_pips, opponent_pips);
          },
          "Contact cube life for a CubeInfo with contact_x set: contact_x + "
          "contact_x_pip_slope * (average pips - 120), clamped to [0.5, 1.0].",
          py::arg("contact_x"), py::arg("contact_x_pip_slope"),
          py::arg("player_pips"), py::arg("opponent_pips"));

    m.def("cube_efficiency",
          [](const std::vector<float>& probs_vec, bool is_race_pos,
             int player_pips, int opponent_pips) {
              std::array<float, NUM_OUTPUTS> probs{};
              for (size_t i = 0; i < std::min(probs_vec.size(), probs.size()); ++i) {
                  probs[i] = probs_vec[i];
              }
              return cube_efficiency(probs, is_race_pos, player_pips, opponent_pips);
          },
          "Cube life index (Janowski x). Current impl: 0.68 for contact, "
          "clamp(0.55 + 0.00125*avg_pips, 0.6, 0.7) for race; probs reserved "
          "for a future ML-based formula.",
          py::arg("probs"), py::arg("is_race"),
          py::arg("player_pips"), py::arg("opponent_pips"));

    m.def("pip_counts",
          [](const std::vector<int>& board) {
              auto [pp, op] = pip_counts(list_to_board(board));
              return std::make_tuple(pp, op);
          },
          "Return (player_pips, opponent_pips) for a 26-element board.",
          py::arg("board"));

    m.def("cube_decision_1ply", [](const std::array<float, NUM_OUTPUTS>& probs,
                                    int cube_value, CubeOwner owner, float cube_x,
                                    int away1, int away2, bool is_crawford,
                                    bool jacoby) {
        validate_met_away_scores(away1, away2);
        CubeInfo ci{cube_value, owner, {away1, away2, is_crawford}, -1.0f, jacoby};
        return cube_decision_1ply(probs, ci, cube_x);
    }, "Compute 1-ply cube decision from cubeless pre-roll probs (Janowski).\n"
       "Returns CubeDecision with equity_nd, equity_dt, equity_dp, should_double, should_take, optimal_equity.",
       py::arg("probs"), py::arg("cube_value") = 1,
       py::arg("owner") = CubeOwner::CENTERED, py::arg("cube_x") = 0.68f,
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("jacoby") = true);

    // Convenience: evaluate pre-roll probs + cube decision in one call
    m.def("evaluate_cube_decision", [](const std::vector<int>& checkers,
                                        int cube_value, CubeOwner owner,
                                        const std::string& purerace_w,
                                        const std::string& racing_w,
                                        const std::string& attacking_w,
                                        const std::string& priming_w,
                                        const std::string& anchoring_w,
                                        int n_h_purerace,
                                        int n_h_racing,
                                        int n_h_attacking,
                                        int n_h_priming,
                                        int n_h_anchoring,
                                        int away1, int away2, bool is_crawford,
                                        float cube_x_override, bool jacoby,
                                        bool beaver, int max_cube_value,
                                        const BearoffDB* bearoff_db) {
        validate_met_away_scores(away1, away2);
        Board board = list_to_board(checkers);
        auto gps = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        std::shared_ptr<Strategy> strat = gps;
        if (bearoff_db && bearoff_db->is_loaded()) {
            strat = std::make_shared<BearoffStrategy>(gps, bearoff_db);
        }

        // Pre-roll probs: flip → evaluate → invert → clamp
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_move_probs = strat->evaluate_probs(flipped, race);
        auto pre_roll_probs = invert_probs(post_move_probs);
        clamp_probs_to_board(pre_roll_probs, board);

        // Cube efficiency (use override if provided)
        float x;
        if (cube_x_override >= 0.0f) {
            x = cube_x_override;
        } else {
            auto [pp, op] = pip_counts(board);
            x = cube_efficiency(pre_roll_probs, race, pp, op);
        }

        // Cube decision
        CubeInfo ci{cube_value, owner, {away1, away2, is_crawford}, cube_x_override, jacoby, beaver, max_cube_value};
        auto cd = cube_decision_1ply(pre_roll_probs, ci, x);

        // Cubeless equity
        float cl_eq = cubeless_equity(pre_roll_probs);

        py::dict result;
        result["probs"] = pre_roll_probs;
        result["cubeless_equity"] = cl_eq;
        result["cube_x"] = x;
        result["equity_nd"] = cd.equity_nd;
        result["equity_dt"] = cd.equity_dt;
        result["equity_dp"] = cd.equity_dp;
        result["should_double"] = cd.should_double;
        result["should_take"] = cd.should_take;
        result["optimal_equity"] = cd.optimal_equity;
        result["is_beaver"] = cd.is_beaver;
        result["is_race"] = race;
        return result;
    }, "Evaluate pre-roll probs and cube decision for a position.\n"
       "Returns dict with probs, cubeless_equity, cube_x, equity_nd/dt/dp, should_double, should_take, optimal_equity, is_beaver.",
       py::arg("checkers"),
       py::arg("cube_value") = 1,
       py::arg("owner") = CubeOwner::CENTERED,
       py::arg("purerace_weights") = "",
       py::arg("racing_weights") = "",
       py::arg("attacking_weights") = "",
       py::arg("priming_weights") = "",
       py::arg("anchoring_weights") = "",
       py::arg("n_hidden_purerace") = 200,
       py::arg("n_hidden_racing") = 400,
       py::arg("n_hidden_attacking") = 400,
       py::arg("n_hidden_priming") = 400,
       py::arg("n_hidden_anchoring") = 400,
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("cube_x_override") = -1.0f, py::arg("jacoby") = true,
       py::arg("beaver") = true, py::arg("max_cube_value") = 0,
       py::arg("bearoff_db") = nullptr);

    // N-ply cube decision (standalone — creates its own strategy, serial by default)
    m.def("cube_decision_nply", [](const std::vector<int>& checkers,
                                    int cube_value, CubeOwner owner,
                                    int n_plies,
                                    const std::string& purerace_w,
                                    const std::string& racing_w,
                                    const std::string& attacking_w,
                                    const std::string& priming_w,
                                    const std::string& anchoring_w,
                                    int n_h_purerace,
                                    int n_h_racing,
                                    int n_h_attacking,
                                    int n_h_priming,
                                    int n_h_anchoring,
                                    int filter_max_moves,
                                    float filter_threshold,
                                    int n_threads,
                                    int away1, int away2, bool is_crawford,
                                    float cube_x_override, bool jacoby,
                                    bool beaver, int max_cube_value,
                                    const BearoffDB* bearoff_db,
                                    bool incl_2ply_details) {
        validate_search_depth(n_plies);
        if (incl_2ply_details && n_plies < 2) {
            throw std::invalid_argument(
                "incl_2ply_details requires n_plies >= 2 (analysis depth not deep enough)");
        }
        validate_met_away_scores(away1, away2);

        Board board = list_to_board(checkers);
        auto gps = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);

        // Wrap in BearoffStrategy if DB is available
        std::shared_ptr<Strategy> base_strat = gps;
        if (bearoff_db && bearoff_db->is_loaded()) {
            base_strat = std::make_shared<BearoffStrategy>(gps, bearoff_db);
        }

        CubeInfo ci{cube_value, owner, {away1, away2, is_crawford}, cube_x_override, jacoby, beaver, max_cube_value};
        MoveFilter filter{filter_max_moves, filter_threshold};

        CubeDecision cd;
        TwoPlyDetails details;
        std::array<float, NUM_OUTPUTS> pre_roll_probs;
        float cl_eq;
        {
            py::gil_scoped_release release;
            if (incl_2ply_details) {
                cd = cube_decision_nply_with_details(board, ci, *base_strat, n_plies, filter, n_threads, details);
            } else {
                cd = cube_decision_nply(board, ci, *base_strat, n_plies, filter, n_threads);
            }

            // Get N-ply cubeless pre-roll probs: flip board, evaluate at N-ply, invert.
            Board flipped = flip(board);
            MultiPlyStrategy multipy(base_strat, n_plies, filter);
            if (bearoff_db && bearoff_db->is_loaded()) {
                multipy.set_bearoff_db(bearoff_db);
            }
            auto post_probs = multipy.evaluate_probs(flipped, flipped);
            multipy.clear_cache();
            pre_roll_probs = invert_probs(post_probs);
            cl_eq = cubeless_equity(pre_roll_probs);
        }

        py::dict result;
        result["probs"] = pre_roll_probs;
        result["cubeless_equity"] = cl_eq;
        result["equity_nd"] = cd.equity_nd;
        result["equity_dt"] = cd.equity_dt;
        result["equity_dp"] = cd.equity_dp;
        result["should_double"] = cd.should_double;
        result["should_take"] = cd.should_take;
        result["optimal_equity"] = cd.optimal_equity;
        result["is_beaver"] = cd.is_beaver;
        result["n_plies"] = n_plies;
        if (incl_2ply_details) {
            result["details"] = two_ply_details_to_dict(details);
        }
        return result;
    }, "N-ply cube decision for a pre-roll position.\n"
       "Returns dict with probs, cubeless_equity, equity_nd/dt/dp, should_double, should_take, optimal_equity, is_beaver.\n"
       "When incl_2ply_details=True, also returns player_rolls with per-roll details.",
       py::arg("checkers"),
       py::arg("cube_value") = 1,
       py::arg("owner") = CubeOwner::CENTERED,
       py::arg("n_plies") = 2,
       py::arg("purerace_weights") = "",
       py::arg("racing_weights") = "",
       py::arg("attacking_weights") = "",
       py::arg("priming_weights") = "",
       py::arg("anchoring_weights") = "",
       py::arg("n_hidden_purerace") = 200,
       py::arg("n_hidden_racing") = 400,
       py::arg("n_hidden_attacking") = 400,
       py::arg("n_hidden_priming") = 400,
       py::arg("n_hidden_anchoring") = 400,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 1,
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("cube_x_override") = -1.0f, py::arg("jacoby") = true,
       py::arg("beaver") = true, py::arg("max_cube_value") = 0,
       py::arg("bearoff_db") = nullptr,
       py::arg("incl_2ply_details") = false);

    // ======================== Batch position evaluation ========================

    // Evaluate a batch of pre-roll positions in parallel.
    // Each position is a (board, cube_value, cube_owner[, away1, away2, is_crawford]) tuple.
    // Uses a MultiPlyStrategy or any 1-ply Strategy shared across workers.
    // MultiPlyStrategy overload is registered first so pybind11 clears its cache.
    m.def("batch_evaluate_positions", [](
            py::list positions,
            std::shared_ptr<MultiPlyStrategy> strategy,
            int n_threads,
            bool jacoby,
            bool beaver,
            float contact_x,
            float contact_x_pip_slope) {
        require_non_null(strategy, "strategy");
        return batch_evaluate_positions_impl(
            positions, *strategy, strategy.get(), n_threads,
            jacoby, beaver, contact_x, contact_x_pip_slope);
    }, "Evaluate a batch of pre-roll positions in parallel.\n"
       "positions: list of (board, cube_value, CubeOwner) tuples.\n"
       "strategy: MultiPlyStrategy (1-ply uses n_plies=1 wrapper).\n"
       "Returns list of dicts with probs, cubeless_equity, cubeful_equity, cube decision fields.",
       py::arg("positions"),
       py::arg("strategy"),
       py::arg("n_threads") = 0,
       py::arg("jacoby") = true,
       py::arg("beaver") = true,
       py::arg("contact_x") = -1.0f,
       py::arg("contact_x_pip_slope") = 0.0f);

    m.def("batch_evaluate_positions", [](
            py::list positions,
            Strategy& strategy,
            int n_threads,
            bool jacoby,
            bool beaver,
            float contact_x,
            float contact_x_pip_slope) {
        return batch_evaluate_positions_impl(
            positions, strategy, nullptr, n_threads,
            jacoby, beaver, contact_x, contact_x_pip_slope);
    }, "Evaluate a batch of pre-roll positions at 1-ply in parallel.\n"
       "positions: list of (board, cube_value, CubeOwner) tuples.\n"
       "strategy: any 1-ply Strategy.\n"
       "Returns list of dicts with probs, cubeless_equity, cubeful_equity, cube decision fields.",
       py::arg("positions"),
       py::arg("strategy"),
       py::arg("n_threads") = 0,
       py::arg("jacoby") = true,
       py::arg("beaver") = true,
       py::arg("contact_x") = -1.0f,
       py::arg("contact_x_pip_slope") = 0.0f);

    // ======================== Batch post-move position evaluation ========================

    // Evaluate a batch of post-move positions in parallel.
    // Each position is a (board, cube_owner[, cube_value, away1, away2, is_crawford]) tuple.
    // MultiPlyStrategy overload is registered first so pybind11 clears its cache.
    m.def("batch_evaluate_post_move", [](
            py::list positions,
            std::shared_ptr<MultiPlyStrategy> strategy,
            int n_threads,
            bool jacoby,
            float contact_x,
            float contact_x_pip_slope) {
        require_non_null(strategy, "strategy");
        return batch_evaluate_post_move_impl(
            positions, *strategy, strategy.get(), n_threads,
            jacoby, contact_x, contact_x_pip_slope);
    }, "Evaluate a batch of post-move positions at N-ply in parallel.\n"
       "positions: list of (board, CubeOwner) tuples.\n"
       "strategy: MultiPlyStrategy.\n"
       "Returns list of dicts with probs, cubeless_equity, cubeful_equity.",
       py::arg("positions"),
       py::arg("strategy"),
       py::arg("n_threads") = 0,
       py::arg("jacoby") = true,
       py::arg("contact_x") = -1.0f,
       py::arg("contact_x_pip_slope") = 0.0f);

    m.def("batch_evaluate_post_move", [](
            py::list positions,
            Strategy& strategy,
            int n_threads,
            bool jacoby,
            float contact_x,
            float contact_x_pip_slope) {
        return batch_evaluate_post_move_impl(
            positions, strategy, nullptr, n_threads,
            jacoby, contact_x, contact_x_pip_slope);
    }, "Evaluate a batch of post-move positions at 1-ply in parallel.\n"
       "positions: list of (board, CubeOwner) tuples.\n"
       "strategy: any 1-ply Strategy.\n"
       "Returns list of dicts with probs, cubeless_equity, cubeful_equity.",
       py::arg("positions"),
       py::arg("strategy"),
       py::arg("n_threads") = 0,
       py::arg("jacoby") = true,
       py::arg("contact_x") = -1.0f,
       py::arg("contact_x_pip_slope") = 0.0f);

    // ======================== Batch checker play ========================

    // 1-ply batch checker play (unified across all 1-ply Strategy subclasses).
    // Delegates to batch_checker_play_1ply -> score_checker_1ply (batched NN
    // evaluation, probability clamping, double-precision dead-cube handling).
    m.def("batch_checker_play", [](
            py::list inputs,
            Strategy& strategy_1ply,
            int filter_max_moves,
            float filter_threshold,
            int n_threads,
            bool jacoby,
            bool beaver,
            float contact_x_1ply,
            float contact_x_pip_slope_1ply) {
        return batch_checker_play_1ply(
            inputs, strategy_1ply, filter_max_moves, filter_threshold,
            n_threads, jacoby, beaver, contact_x_1ply, contact_x_pip_slope_1ply);
    }, "Batch checker play at 1-ply (any 1-ply strategy).\n"
       "inputs: list of dicts {board, die1, die2, cube_value, cube_owner[, away1, away2, is_crawford, max_cube_value, force_boards]}.\n"
       "Returns list of dicts, each with 'moves' list sorted by equity desc.",
       py::arg("inputs"),
       py::arg("strategy_1ply"),
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 0,
       py::arg("jacoby") = true,
       py::arg("beaver") = true,
       py::arg("contact_x_1ply") = -1.0f,
       py::arg("contact_x_pip_slope_1ply") = 0.0f);

    // One position's N-ply checker play: the routine the batch runs per
    // position (score_checker_position), with the analyzer's options.
    m.def("checker_play_nply", [](
            const std::vector<int>& board, int die1, int die2,
            const Strategy& strategy_1ply,
            std::shared_ptr<MultiPlyStrategy> strategy_nply,
            int cube_value, CubeOwner cube_owner,
            int away1, int away2, bool is_crawford, int max_cube_value,
            const std::vector<std::vector<int>>& force_boards,
            int filter_max_moves, float filter_threshold,
            bool jacoby, bool beaver,
            float contact_x_1ply, float contact_x_pip_slope_1ply,
            float contact_x_nply, float contact_x_pip_slope_nply,
            const std::vector<int>& cubeful_filtered_plies,
            int n_threads) {
        require_non_null(strategy_nply, "strategy_nply");
        CheckerPosition pos;
        pos.board = list_to_board(board);
        pos.die1 = die1;
        pos.die2 = die2;
        validate_die(pos.die1);
        validate_die(pos.die2);
        pos.cube_value = cube_value;
        pos.owner = cube_owner;
        validate_met_away_scores(away1, away2);
        pos.match = MatchInfo{away1, away2, is_crawford};
        pos.max_cube_value = max_cube_value;
        for (const auto& b : force_boards) pos.force_boards.push_back(list_to_board(b));
        const AnalyzerSession session = make_checker_search(
            strategy_1ply, *strategy_nply, filter_max_moves, filter_threshold, jacoby, beaver,
            contact_x_1ply, contact_x_pip_slope_1ply, contact_x_nply, contact_x_pip_slope_nply,
            checker_root_is_filtered(strategy_nply->n_plies(), cubeful_filtered_plies));
        std::vector<ScoredMove> moves;
        {
            py::gil_scoped_release release;
            moves = session.score_checker_position(pos, n_threads);
            strategy_nply->clear_cache();
        }
        return scored_moves_to_list(moves, strategy_nply->n_plies());
    }, "N-ply cubeful checker play of one position: the ranked move list\n"
       "(dicts with board, equity, cubeless_equity, probs, equity_diff,\n"
       "eval_level), as batch_checker_play computes it per position.\n"
       "n_threads spreads the cube-tree calls of the searched moves.",
       py::arg("board"), py::arg("die1"), py::arg("die2"),
       py::arg("strategy_1ply"), py::arg("strategy_nply"),
       py::arg("cube_value") = 1, py::arg("cube_owner") = CubeOwner::CENTERED,
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("max_cube_value") = 0,
       py::arg("force_boards") = std::vector<std::vector<int>>{},
       py::arg("filter_max_moves") = 5, py::arg("filter_threshold") = 0.08f,
       py::arg("jacoby") = true, py::arg("beaver") = true,
       py::arg("contact_x_1ply") = -1.0f, py::arg("contact_x_pip_slope_1ply") = 0.0f,
       py::arg("contact_x_nply") = -1.0f, py::arg("contact_x_pip_slope_nply") = 0.0f,
       py::arg("cubeful_filtered_plies") = std::vector<int>{},
       py::arg("n_threads") = 1);

    // One N-ply overload accepts every derived 1-ply strategy.
    m.def("batch_checker_play", [](
            py::list inputs,
            Strategy& strategy_1ply,
            std::shared_ptr<MultiPlyStrategy> strategy_nply,
            int filter_max_moves,
            float filter_threshold,
            int n_threads,
            bool jacoby,
            bool beaver,
            float contact_x_1ply,
            float contact_x_pip_slope_1ply,
            float contact_x_nply,
            float contact_x_pip_slope_nply,
            const std::vector<int>& cubeful_filtered_plies) {
        return batch_checker_play_nply(
            inputs, strategy_1ply, strategy_nply, filter_max_moves, filter_threshold,
            n_threads, jacoby, beaver, contact_x_1ply, contact_x_pip_slope_1ply,
            contact_x_nply, contact_x_pip_slope_nply, cubeful_filtered_plies);
    }, "Batch checker play at N-ply (any 1-ply strategy).\n"
       "inputs: list of dicts {board, die1, die2, cube_value, cube_owner}.\n"
       "strategy_nply: MultiPlyStrategy whose depth is the search depth.\n"
       "cubeful_filtered_plies: depths at which only the 1-ply filter's survivors\n"
       "(and stale-top-two promotions) go through the cube-aware tree; at other\n"
       "depths every legal move does. Python passes CUBEFUL_FILTERED_PLIES.\n"
       "At 3-ply and above custom filter_max_moves/filter_threshold require a filtered root.\n"
       "Returns list of dicts, each with 'moves' list sorted by equity desc,\n"
       "the same list BgBotAnalyzer.checker_play gives (no bearoff database).",
       py::arg("inputs"),
       py::arg("strategy_1ply"),
       py::arg("strategy_nply"),
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 0,
       py::arg("jacoby") = true,
       py::arg("beaver") = true,
       py::arg("contact_x_1ply") = -1.0f,
       py::arg("contact_x_pip_slope_1ply") = 0.0f,
       py::arg("contact_x_nply") = -1.0f,
       py::arg("contact_x_pip_slope_nply") = 0.0f,
       py::arg("cubeful_filtered_plies") = std::vector<int>{});

    // --- Unified cubeful_equity_nply (accepts any Strategy via shared_ptr) ---
    m.def("cubeful_equity_nply", [](const std::vector<int>& board_vec,
                                     CubeOwner owner,
                                     std::shared_ptr<Strategy> strategy,
                                     int n_plies,
                                     int filter_max_moves,
                                     float filter_threshold,
                                     int n_threads,
                                     int cube_value,
                                     int away1, int away2, bool is_crawford,
                                     bool jacoby, bool beaver,
                                     const BearoffDB* bearoff_db, int max_cube_value,
                                     float contact_x,
            float contact_x_pip_slope) {
        Board board = list_to_board(board_vec);
        validate_search_depth(n_plies);
        validate_met_away_scores(away1, away2);
        MoveFilter filter{filter_max_moves, filter_threshold};
        require_non_null(strategy, "strategy");

        auto eval_strat = with_bearoff_db(std::move(strategy), bearoff_db);

        py::gil_scoped_release release;
        CubeInfo ci{cube_value, owner,
                    {away1, away2, away1 > 0 && is_crawford},
                    -1.0f, jacoby, beaver, max_cube_value, contact_x, contact_x_pip_slope};
        return cubeful_equity_nply(board, ci, *eval_strat, n_plies, filter, n_threads);
    }, "Compute cubeful equity for a pre-roll position at N-ply depth (unified, any Strategy).",
       py::arg("board"), py::arg("owner"),
       py::arg("strategy"),
       py::arg("n_plies") = 2,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 1,
       py::arg("cube_value") = 1,
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("jacoby") = true, py::arg("beaver") = true,
       py::arg("bearoff_db") = nullptr, py::arg("max_cube_value") = 0,
       py::arg("contact_x") = -1.0f,
       py::arg("contact_x_pip_slope") = 0.0f);

    // --- Unified cubeful_probs_nply (accepts any Strategy via shared_ptr) ---
    m.def("cubeful_probs_nply", [](const std::vector<int>& board_vec,
                                    CubeOwner owner,
                                    std::shared_ptr<Strategy> strategy,
                                    int n_plies,
                                    int filter_max_moves,
                                    float filter_threshold,
                                    int n_threads,
                                    int cube_value,
                                    int away1, int away2, bool is_crawford,
                                    bool jacoby, bool beaver,
                                    const BearoffDB* bearoff_db, int max_cube_value) {
        Board board = list_to_board(board_vec);
        validate_search_depth(n_plies);
        validate_met_away_scores(away1, away2);
        MoveFilter filter{filter_max_moves, filter_threshold};
        require_non_null(strategy, "strategy");

        auto eval_strat = with_bearoff_db(std::move(strategy), bearoff_db);

        std::array<float, NUM_OUTPUTS> probs;
        {
            py::gil_scoped_release release;
            CubeInfo ci{cube_value, owner,
                        {away1, away2, away1 > 0 && is_crawford},
                        -1.0f, jacoby, beaver, max_cube_value};
            probs = cubeful_probs_nply(board, ci, *eval_strat, n_plies, filter, n_threads);
        }
        return probs;
    }, "Post-roll cubeless probs from the N-ply CUBE-AWARE tree (unified, any Strategy).",
       py::arg("board"), py::arg("owner"),
       py::arg("strategy"),
       py::arg("n_plies") = 2,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 1,
       py::arg("cube_value") = 1,
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("jacoby") = true, py::arg("beaver") = true,
       py::arg("bearoff_db") = nullptr, py::arg("max_cube_value") = 0);

    // --- Unified cubeful_probs_and_equity_nply ---
    // Returns BOTH cube-aware probs and cubeful equity from a single tree
    // traversal — saves a redundant cubeful_recursive_multi call when callers
    // (e.g. per-candidate checker play analytics) want both values.
    m.def("cubeful_probs_and_equity_nply",
          [](const std::vector<int>& board_vec,
             CubeOwner owner,
             std::shared_ptr<Strategy> strategy,
             int n_plies,
             int filter_max_moves,
             float filter_threshold,
             int n_threads,
             int cube_value,
             int away1, int away2, bool is_crawford,
             bool jacoby, bool beaver,
             const BearoffDB* bearoff_db,
             const std::vector<int>& root_board, int max_cube_value,
             float contact_x,
            float contact_x_pip_slope) {
        Board board = list_to_board(board_vec);
        validate_search_depth(n_plies);
        validate_met_away_scores(away1, away2);
        MoveFilter filter{filter_max_moves, filter_threshold};
        require_non_null(strategy, "strategy");
        Board root{};
        const Board* root_ptr = nullptr;
        if (!root_board.empty()) {
            root = list_to_board(root_board);
            root_ptr = &root;
        }

        auto eval_strat = with_bearoff_db(std::move(strategy), bearoff_db);

        CubefulProbsAndEquity result;
        {
            py::gil_scoped_release release;
            CubeInfo ci{cube_value, owner,
                        {away1, away2, away1 > 0 && is_crawford},
                        -1.0f, jacoby, beaver, max_cube_value, contact_x, contact_x_pip_slope};
            result = cubeful_probs_and_equity_nply(
                board, ci, *eval_strat, n_plies, filter, n_threads, nullptr, root_ptr);
        }
        py::dict d;
        d["probs"] = result.probs;
        d["equity"] = result.equity;
        return d;
    }, "Post-roll probs AND cubeful equity from the N-ply CUBE-AWARE tree (single traversal).",
       py::arg("board"), py::arg("owner"),
       py::arg("strategy"),
       py::arg("n_plies") = 2,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 1,
       py::arg("cube_value") = 1,
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("jacoby") = true, py::arg("beaver") = true,
       py::arg("bearoff_db") = nullptr,
       py::arg("root_board") = std::vector<int>{}, py::arg("max_cube_value") = 0,
       py::arg("contact_x") = -1.0f,
       py::arg("contact_x_pip_slope") = 0.0f);

    // --- Unified 1-ply cube decision ---
    m.def("evaluate_cube_decision_unified", [](const std::vector<int>& checkers,
                                                int cube_value, CubeOwner owner,
                                                const std::string& strategy_type,
                                                const std::vector<std::string>& weight_paths,
                                                const std::vector<int>& hidden_sizes,
                                                int away1, int away2, bool is_crawford,
                                                float cube_x_override, bool jacoby,
                                                bool beaver, int max_cube_value, float contact_x, float contact_x_pip_slope,
                                                const BearoffDB* bearoff_db,
                                                std::shared_ptr<Strategy> strategy) {
        validate_met_away_scores(away1, away2);
        Board board = list_to_board(checkers);
        const CubeInfo ci{cube_value, owner, {away1, away2, is_crawford}, cube_x_override,
                          jacoby, beaver, max_cube_value, contact_x, contact_x_pip_slope};
        CubeEvaluation evaluation;
        {
            py::gil_scoped_release release;
            auto base = strategy ? std::move(strategy) :
                make_strategy_from_type(strategy_type, weight_paths, hidden_sizes);
            auto strat = with_bearoff_db(std::move(base), bearoff_db);
            evaluation = evaluate_cube_leaf(board, ci, *strat);
        }
        auto result = cube_evaluation_dict(evaluation, false);
        result["cube_x"] = evaluation.cube_x;
        result["is_race"] = is_race(board);
        return result;
    }, "Unified 1-ply cube decision for any strategy type.",
       py::arg("checkers"),
       py::arg("cube_value") = 1,
       py::arg("owner") = CubeOwner::CENTERED,
       py::arg("strategy_type"),
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("cube_x_override") = -1.0f, py::arg("jacoby") = true,
       py::arg("beaver") = true, py::arg("max_cube_value") = 0,
       py::arg("contact_x") = -1.0f,
       py::arg("contact_x_pip_slope") = 0.0f,
       py::arg("bearoff_db") = nullptr, py::arg("strategy") = nullptr);

    m.def("score_checker_endpoints", [](const Board& board, const std::vector<Board>& endpoints,
             const CubeInfo& opponent_cube, const Strategy& strategy, int plies, int threads) {
        validate_search_depth(plies);
        std::vector<ScoredMove> moves;
        {
            py::gil_scoped_release release;
            moves = score_checker_endpoints(board, endpoints, opponent_cube, strategy, plies, threads);
        }
        return scored_moves_to_list(moves, plies);
    }, py::arg("board"), py::arg("endpoints"), py::arg("opponent_cube"), py::arg("strategy"),
       py::arg("n_plies"), py::arg("n_threads") = 1);

    m.def("batch_cube_action_nply", [](const std::vector<std::pair<Board, CubeInfo>>& positions,
            const std::shared_ptr<Strategy>& base, const BearoffDB* db, int plies,
            int threads, bool decision_only, int max_moves, float threshold) {
        validate_search_depth(plies);
        validate_batch_move_filter(plies, max_moves, threshold, false);
        require_non_null(base, "base");
        auto strategy = with_bearoff_db(base, db);
        std::vector<CubeEvaluation> results(positions.size());
        {
            py::gil_scoped_release release;
            const int rows = static_cast<int>(positions.size());
            auto evaluate = [&](int i, int inner) {
                const auto& [board, ci] = positions[i];
                results[i] = plies == 1 ? evaluate_cube_leaf(board, ci, *strategy)
                    : evaluate_cube_tree(board, ci, strategy, db, plies,
                                         {max_moves, threshold}, inner, decision_only);
            };
            if (plies >= 3 && rows > 1 && db && db->is_loaded()) {
                // A cheap bearoff row should not take a lane away from a long
                // contact search. Share the budget among comparable rows.
                std::array<std::vector<int>, 2> groups;
                for (int i = 0; i < rows; ++i)
                    groups[db->is_bearoff(positions[i].first)].push_back(i);
                for (size_t g = 0; g < groups.size(); ++g) {
                    const auto& group = groups[g];
                    // Even a bearoff cube has to search its cubeful dice tree.
                    const bool short_bearoff = g == 1 && static_cast<int>(group.size()) <= batch_thread_budget(threads);
                    run_search_batch(static_cast<int>(group.size()), threads, short_bearoff,
                        [&](int i, int inner) { evaluate(group[i], inner); });
                }
            } else {
                run_search_batch(rows, threads, false, evaluate);
            }
        }
        py::list out;
        for (const auto& r : results) out.append(cube_evaluation_dict(r, decision_only));
        return out;
    }, "Recursive cube batch. At 3-ply and above use the default filter_max_moves and filter_threshold.",
       py::arg("positions"), py::arg("base"), py::arg("bearoff_db"), py::arg("n_plies"),
       py::arg("n_threads") = 0, py::arg("decision_only") = false,
       py::arg("filter_max_moves") = 5, py::arg("filter_threshold") = 0.08f);

    // --- Unified N-ply cube decision ---
    m.def("cube_decision_nply_unified", [](const std::vector<int>& checkers,
                                            int cube_value, CubeOwner owner,
                                            int n_plies,
                                            const std::string& strategy_type,
                                            const std::vector<std::string>& weight_paths,
                                            const std::vector<int>& hidden_sizes,
                                            int filter_max_moves,
                                            float filter_threshold,
                                            int n_threads,
                                            int away1, int away2, bool is_crawford,
                                            float cube_x_override, bool jacoby,
                                            bool beaver, int max_cube_value, float contact_x, float contact_x_pip_slope,
                                            const BearoffDB* bearoff_db,
                                            bool incl_2ply_details,
                                            std::shared_ptr<Strategy> strategy,
                                            bool decision_only) {
        validate_search_depth(n_plies);
        if (incl_2ply_details && n_plies < 2) {
            throw std::invalid_argument(
                "incl_2ply_details requires n_plies >= 2");
        }
        validate_met_away_scores(away1, away2);
        Board board = list_to_board(checkers);
        std::shared_ptr<Strategy> base_strat;
        {
            py::gil_scoped_release release;
            auto base = strategy ? std::move(strategy) :
                make_strategy_from_type(strategy_type, weight_paths, hidden_sizes);
            base_strat = with_bearoff_db(std::move(base), bearoff_db);
        }
        // A caller may supply the ready-to-use wrapper without a separate DB.
        // The cubeless evaluator also needs that DB for its root shortcut.
        const Strategy* layer = base_strat.get();
        for (auto wrap = layer->bearoff_layer(); wrap.base; wrap = layer->bearoff_layer()) {
            if (wrap.db && wrap.db->is_loaded()) {
                bearoff_db = wrap.db;
                break;
            }
            layer = wrap.base;
        }
        CubeInfo ci{cube_value, owner, {away1, away2, is_crawford}, cube_x_override, jacoby, beaver, max_cube_value, contact_x, contact_x_pip_slope};
        MoveFilter filter{filter_max_moves, filter_threshold};
        TwoPlyDetails details;
        CubeEvaluation evaluation;
        {
            py::gil_scoped_release release;
            evaluation = evaluate_cube_tree(board, ci, base_strat, bearoff_db, n_plies,
                                            filter, n_threads, decision_only,
                                            incl_2ply_details ? &details : nullptr);
        }
        auto result = cube_evaluation_dict(evaluation, decision_only);
        result["n_plies"] = n_plies;
        if (incl_2ply_details) {
            result["details"] = two_ply_details_to_dict(details);
        }
        return result;
    }, "Unified N-ply cube decision for any strategy type.",
       py::arg("checkers"),
       py::arg("cube_value") = 1,
       py::arg("owner") = CubeOwner::CENTERED,
       py::arg("n_plies") = 2,
       py::arg("strategy_type"),
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 1,
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("cube_x_override") = -1.0f, py::arg("jacoby") = true,
       py::arg("beaver") = true, py::arg("max_cube_value") = 0,
       py::arg("contact_x") = -1.0f,
       py::arg("contact_x_pip_slope") = 0.0f,
       py::arg("bearoff_db") = nullptr,
       py::arg("incl_2ply_details") = false, py::arg("strategy") = nullptr,
       py::arg("decision_only") = false);
}

} // namespace bgbot::python
