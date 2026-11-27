// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bindings_common.h"
#include <pybind11/numpy.h>
#include <limits>
#include "move_arrays.h"
#include "bgbot/analyzer_session.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/pubeval.h"
#include "bgbot/game.h"
#include "bgbot/encoding.h"
#include "bgbot/neural_net.h"
#include "bgbot/multipy.h"
#include "bgbot/bearoff.h"

namespace bgbot::python {

std::shared_ptr<Strategy> make_strategy_from_type(
    const std::string& strategy_type,
    const std::vector<std::string>& weight_paths,
    const std::vector<int>& hidden_sizes)
{
    if (strategy_type == "5nn")
        return std::make_shared<GamePlanStrategy>(weight_paths, hidden_sizes);
    if (strategy_type == "pair")
        return std::make_shared<GamePlanPairStrategy>(weight_paths, hidden_sizes);
    if (strategy_type == "backgame_pair")
        return std::make_shared<BackgameAwarePairStrategy>(weight_paths, hidden_sizes);
    if (strategy_type == "backgame_pair_phased")
        return std::make_shared<BackgameAwarePairStrategy>(weight_paths, hidden_sizes, true);
    throw std::invalid_argument("Unknown strategy type: " + strategy_type +
        ". Valid types: 5nn, pair, backgame_pair, backgame_pair_phased");
}

namespace {

// Benchmark PR: parse decision dicts ('board', 'dice', 'candidates',
// 'rollout_equities') from Python.
std::vector<PRDecision> pr_decisions_from_py(py::list decisions_py) {
    std::vector<PRDecision> decisions;
    decisions.reserve(decisions_py.size());
    for (auto item : decisions_py) {
        PRDecision dec;
        auto d = item.cast<py::dict>();
        dec.board = list_to_board(d["board"].cast<std::vector<int>>());
        auto dice = d["dice"].cast<std::vector<int>>();
        dec.die1 = dice[0];
        dec.die2 = dice[1];
        auto cands = d["candidates"].cast<std::vector<std::vector<int>>>();
        for (auto& c : cands) {
            dec.candidates.push_back(list_to_board(c));
        }
        dec.rollout_equities = d["rollout_equities"].cast<std::vector<double>>();
        decisions.push_back(std::move(dec));
    }
    return decisions;
}

// Benchmark PR: overall and per-plan results as a Python dict.
py::dict pr_result_to_py(const PRResult& result) {
    py::dict out;
    out["overall_pr"] = result.overall_pr();
    out["total_decisions"] = result.total_decisions();
    out["total_error"] = result.total_error();
    out["n_outside"] = result.n_outside;
    out["n_skipped"] = result.n_skipped;
    py::dict gp_pr, gp_n, gp_n_err;
    const char* gp_names[] = {"purerace", "racing", "attacking", "priming", "anchoring"};
    for (int i = 0; i < 5; ++i) {
        gp_pr[gp_names[i]] = result.gp_pr(i);
        gp_n[gp_names[i]] = result.gp_n_decisions[i];
        gp_n_err[gp_names[i]] = result.gp_n_with_error[i];
    }
    out["per_plan_pr"] = gp_pr;
    out["per_plan_n"] = gp_n;
    out["per_plan_n_with_error"] = gp_n_err;
    return out;
}

py::dict evaluate_board_dict(const Strategy& strat,
                                    const std::vector<int>& board,
                                    const std::vector<int>& pre_move_board) {
    auto b = list_to_board(board);
    auto pmb = list_to_board(pre_move_board);
    std::array<float, NUM_OUTPUTS> probs;
    {
        py::gil_scoped_release release;
        probs = strat.evaluate_probs(b, pmb);
        clamp_probs_to_board(probs, b);
    }
    double eq = NeuralNetwork::compute_equity(probs);
    py::dict result;
    result["probs"] = probs;
    result["equity"] = eq;
    return result;
}

} // namespace

void bind_core(py::module_& m) {
    // --- ScenarioSet (preferred for large benchmark sets) ---
    py::class_<ScenarioSet, std::shared_ptr<ScenarioSet>>(m, "ScenarioSet")
        .def(py::init<>())
        .def("reserve", &ScenarioSet::reserve)
        .def("size", &ScenarioSet::size)
        .def("add", &ScenarioSet::add,
             py::arg("start_board"), py::arg("die1"), py::arg("die2"),
             py::arg("ranked_boards"), py::arg("ranked_errors"))
        .def("__len__", &ScenarioSet::size);

    // --- BenchmarkScenario (kept for backward compat) ---
    py::class_<BenchmarkScenario>(m, "BenchmarkScenario")
        .def(py::init<>())
        .def_readwrite("die1", &BenchmarkScenario::die1)
        .def_readwrite("die2", &BenchmarkScenario::die2);

    // Factory function to build a BenchmarkScenario from Python data
    m.def("make_scenario", [](const std::vector<int>& start_board,
                               int die1, int die2,
                               const std::vector<std::vector<int>>& ranked_boards,
                               const std::vector<double>& ranked_errors) {
        BenchmarkScenario s;
        s.start_board = list_to_board(start_board);
        s.die1 = die1;
        s.die2 = die2;
        s.ranked_boards.reserve(ranked_boards.size());
        for (const auto& rb : ranked_boards) {
            s.ranked_boards.push_back(list_to_board(rb));
        }
        s.ranked_errors = ranked_errors;
        return s;
    }, "Create a BenchmarkScenario from Python data");

    // --- BenchmarkResult ---
    py::class_<BenchmarkResult>(m, "BenchmarkResult")
        .def(py::init<>())
        .def_readonly("total_error", &BenchmarkResult::total_error)
        .def_readonly("count", &BenchmarkResult::count)
        .def("score", &BenchmarkResult::score);

    // --- PubEval ---
    py::enum_<PubEval::WeightSource>(m, "PubEvalWeights")
        .value("TESAURO", PubEval::WeightSource::TESAURO)
        .value("LISTNET", PubEval::WeightSource::LISTNET);

    py::class_<PubEval>(m, "PubEval")
        .def(py::init<PubEval::WeightSource>(),
             py::arg("weights") = PubEval::WeightSource::TESAURO);

    // --- Benchmark scoring ---
    m.def("score_benchmarks_pubeval", [](const ScenarioSet& ss,
                                          PubEval::WeightSource weights,
                                          int n_threads) {
        PubEval strat(weights);
        py::gil_scoped_release release;
        return score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmark scenarios using PubEval strategy",
       py::arg("scenarios"),
       py::arg("weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    m.def("score_benchmarks_pubeval", [](const std::vector<BenchmarkScenario>& scenarios,
                                          PubEval::WeightSource weights,
                                          int n_threads) {
        PubEval strat(weights);
        py::gil_scoped_release release;
        return score_benchmarks(strat, scenarios, n_threads);
    }, "Score benchmark scenarios using PubEval strategy (list)",
       py::arg("scenarios"),
       py::arg("weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- Game simulation ---
    py::class_<GameStats>(m, "GameStats")
        .def_readonly("n_games", &GameStats::n_games)
        .def_readonly("p1_wins", &GameStats::p1_wins)
        .def_readonly("p1_gammons", &GameStats::p1_gammons)
        .def_readonly("p1_backgammons", &GameStats::p1_backgammons)
        .def_readonly("p2_wins", &GameStats::p2_wins)
        .def_readonly("p2_gammons", &GameStats::p2_gammons)
        .def_readonly("p2_backgammons", &GameStats::p2_backgammons)
        .def("avg_ppg", &GameStats::avg_ppg);

    m.def("play_games_pubeval", [](int n_games, uint32_t seed,
                                    PubEval::WeightSource w1,
                                    PubEval::WeightSource w2,
                                    int n_threads) {
        PubEval strat1(w1);
        PubEval strat2(w2);
        py::gil_scoped_release release;
        return play_games_parallel(strat1, strat2, n_games, seed, n_threads);
    }, "Play games between two PubEval strategies",
       py::arg("n_games"), py::arg("seed") = 42,
       py::arg("weights1") = PubEval::WeightSource::TESAURO,
       py::arg("weights2") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- NeuralNetwork ---
    py::class_<NeuralNetwork, std::shared_ptr<NeuralNetwork>>(m, "NeuralNetwork")
        .def(py::init<int, int, float, uint32_t>(),
             py::arg("n_hidden") = 120,
             py::arg("n_inputs") = NN_INPUTS,
             py::arg("eps") = 0.1f,
             py::arg("seed") = 42)
        .def("save_weights", &NeuralNetwork::save_weights,
             py::arg("filepath"), py::arg("versioned") = false)
        .def("load_weights", &NeuralNetwork::load_weights)
        .def("n_hidden", &NeuralNetwork::n_hidden)
        .def("n_inputs", &NeuralNetwork::n_inputs)
        .def("_get_weights", [](const NeuralNetwork& nn) {
            py::array_t<float> hidden({nn.n_hidden(), nn.n_inputs() + 1});
            py::array_t<float> output({NN_OUTPUTS, nn.n_hidden() + 1});
            const auto& hidden_weights = nn.hidden_weights();
            const auto& output_weights = nn.output_weights();
            std::copy(hidden_weights.begin(), hidden_weights.end(), hidden.mutable_data());
            std::copy(output_weights.begin(), output_weights.end(), output.mutable_data());
            return py::make_tuple(hidden, output);
        })
        .def_static("compute_equity", &NeuralNetwork::compute_equity);

    // --- NNStrategy ---
    py::class_<NNStrategy>(m, "NNStrategy")
        .def(py::init<std::shared_ptr<NeuralNetwork>>())
        .def(py::init<const std::string&, int, int>(),
             py::arg("weights_path"),
             py::arg("n_hidden") = 120,
             py::arg("n_inputs") = NN_INPUTS)
        .def("evaluate_board", [](NNStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            return evaluate_board_dict(self, board, pre_move_board);
        }, "Evaluate board, returns probs and equity",
           py::arg("board"), py::arg("pre_move_board"));

    // --- MultiNNStrategy ---
    py::class_<MultiNNStrategy>(m, "MultiNNStrategy")
        .def(py::init<const std::string&, const std::string&, int>(),
             py::arg("contact_weights"),
             py::arg("race_weights"),
             py::arg("n_hidden") = 120)
        .def(py::init<const std::string&, const std::string&, const std::string&, int>(),
             py::arg("contact_weights"),
             py::arg("crashed_weights"),
             py::arg("race_weights"),
             py::arg("n_hidden") = 120)
        .def(py::init<const std::string&, const std::string&, const std::string&, int, int, int>(),
             py::arg("contact_weights"),
             py::arg("crashed_weights"),
             py::arg("race_weights"),
             py::arg("n_hidden_contact"),
             py::arg("n_hidden_crashed"),
             py::arg("n_hidden_race"))
        .def("evaluate_board", [](MultiNNStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            return evaluate_board_dict(self, board, pre_move_board);
        }, "Evaluate board, returns probs and equity",
           py::arg("board"), py::arg("pre_move_board"));

    // --- Benchmark scoring with NNStrategy ---
    m.def("score_benchmarks_nn", [](const ScenarioSet& ss,
                                     const std::string& weights_path,
                                     int n_hidden,
                                     int n_inputs,
                                     int n_threads) {
        NNStrategy strat(weights_path, n_hidden, n_inputs);
        py::gil_scoped_release release;
        return score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmark scenarios using NN strategy",
       py::arg("scenarios"),
       py::arg("weights_path"),
       py::arg("n_hidden") = 120,
       py::arg("n_inputs") = 196,
       py::arg("n_threads") = 0);

    m.def("score_benchmarks_nn", [](const std::vector<BenchmarkScenario>& scenarios,
                                     const std::string& weights_path,
                                     int n_hidden,
                                     int n_inputs,
                                     int n_threads) {
        NNStrategy strat(weights_path, n_hidden, n_inputs);
        py::gil_scoped_release release;
        return score_benchmarks(strat, scenarios, n_threads);
    }, "Score benchmark scenarios using NN strategy (list)",
       py::arg("scenarios"),
       py::arg("weights_path"),
       py::arg("n_hidden") = 120,
       py::arg("n_inputs") = 196,
       py::arg("n_threads") = 0);

    // The MLX trainer already has the two bias-augmented matrices on the host.
    m.def("_score_benchmarks_nn_arrays", [](
        const py::object& scenarios,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& hidden,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& output,
        int n_threads
    ) {
        if (hidden.ndim() != 2 || output.ndim() != 2) {
            throw py::value_error("hidden and output weights must be matrices");
        }
        const auto n_hidden = hidden.shape(0);
        const auto n_inputs = hidden.shape(1) - 1;
        if ((n_inputs != NN_INPUTS && n_inputs != EXTENDED_CONTACT_INPUTS)
            || n_hidden < 1 || n_hidden > std::numeric_limits<int>::max() / (n_inputs + 1)
            || output.shape(0) != NN_OUTPUTS || output.shape(1) != n_hidden + 1) {
            throw py::value_error("invalid benchmark network weight shapes");
        }
        auto nn = std::make_shared<NeuralNetwork>(static_cast<int>(n_hidden),
                                                  static_cast<int>(n_inputs));
        std::copy_n(hidden.data(), hidden.size(), nn->hidden_weights().begin());
        std::copy_n(output.data(), output.size(), nn->output_weights().begin());
        NNStrategy strat(std::move(nn));
        if (py::isinstance<ScenarioSet>(scenarios)) {
            const auto& ss = scenarios.cast<const ScenarioSet&>();
            py::gil_scoped_release release;
            return score_benchmarks(strat, ss.scenarios, n_threads);
        }
        const auto legacy_scenarios = scenarios.cast<std::vector<BenchmarkScenario>>();
        py::gil_scoped_release release;
        return score_benchmarks(strat, legacy_scenarios, n_threads);
    }, "Score benchmark scenarios directly from weight matrices",
       py::arg("scenarios"), py::arg("hidden"), py::arg("output"),
       py::arg("n_threads") = 0);

    // --- Benchmark scoring with MultiNNStrategy (2-NN: contact+race) ---
    m.def("score_benchmarks_multi_nn", [](const ScenarioSet& ss,
                                           const std::string& contact_weights,
                                           const std::string& race_weights,
                                           int n_hidden,
                                           int n_threads) {
        MultiNNStrategy strat(contact_weights, race_weights, n_hidden);
        py::gil_scoped_release release;
        return score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmarks using MultiNNStrategy (contact+race)",
       py::arg("scenarios"),
       py::arg("contact_weights"),
       py::arg("race_weights"),
       py::arg("n_hidden") = 120,
       py::arg("n_threads") = 0);

    // --- Benchmark scoring with MultiNNStrategy (3-NN: contact+crashed+race, separate hidden sizes) ---
    m.def("score_benchmarks_3nn", [](const ScenarioSet& ss,
                                      const std::string& contact_weights,
                                      const std::string& crashed_weights,
                                      const std::string& race_weights,
                                      int n_hidden_contact,
                                      int n_hidden_crashed,
                                      int n_hidden_race,
                                      int n_threads) {
        MultiNNStrategy strat(contact_weights, crashed_weights, race_weights,
                              n_hidden_contact, n_hidden_crashed, n_hidden_race);
        py::gil_scoped_release release;
        return score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmarks using 3-NN MultiNNStrategy (separate hidden sizes)",
       py::arg("scenarios"),
       py::arg("contact_weights"),
       py::arg("crashed_weights"),
       py::arg("race_weights"),
       py::arg("n_hidden_contact") = 120,
       py::arg("n_hidden_crashed") = 120,
       py::arg("n_hidden_race") = 80,
       py::arg("n_threads") = 0);

    // --- NN vs PubEval game simulation (parallelized) ---
    m.def("play_games_nn_vs_pubeval", [](const std::string& weights_path,
                                          int n_hidden,
                                          int n_games,
                                          uint32_t seed,
                                          PubEval::WeightSource pe_weights,
                                          int n_threads) {
        auto nn_ptr = std::make_shared<NeuralNetwork>(n_hidden);
        nn_ptr->load_weights(weights_path);
        NNStrategy nn_strat(nn_ptr);
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(nn_strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: NN (p1) vs PubEval (p2)",
       py::arg("weights_path"),
       py::arg("n_hidden") = 120,
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- MultiNN vs PubEval (2-NN) ---
    m.def("play_games_multi_nn_vs_pubeval", [](const std::string& contact_weights,
                                                const std::string& race_weights,
                                                int n_hidden,
                                                int n_games,
                                                uint32_t seed,
                                                PubEval::WeightSource pe_weights,
                                                int n_threads) {
        MultiNNStrategy nn_strat(contact_weights, race_weights, n_hidden);
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(nn_strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: MultiNN (p1) vs PubEval (p2)",
       py::arg("contact_weights"),
       py::arg("race_weights"),
       py::arg("n_hidden") = 120,
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- 3-NN vs PubEval (separate hidden sizes) ---
    m.def("play_games_3nn_vs_pubeval", [](const std::string& contact_weights,
                                           const std::string& crashed_weights,
                                           const std::string& race_weights,
                                           int n_hidden_contact,
                                           int n_hidden_crashed,
                                           int n_hidden_race,
                                           int n_games,
                                           uint32_t seed,
                                           PubEval::WeightSource pe_weights,
                                           int n_threads) {
        MultiNNStrategy nn_strat(contact_weights, crashed_weights, race_weights,
                                 n_hidden_contact, n_hidden_crashed, n_hidden_race);
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(nn_strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: 3-NN (p1) vs PubEval (p2), separate hidden sizes",
       py::arg("contact_weights"),
       py::arg("crashed_weights"),
       py::arg("race_weights"),
       py::arg("n_hidden_contact") = 120,
       py::arg("n_hidden_crashed") = 120,
       py::arg("n_hidden_race") = 80,
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- Utility: weight transfer from 196-input to 214-input network ---
    m.def("create_extended_from_tesauro", [](const std::string& src_weights,
                                              const std::string& dst_weights,
                                              int n_hidden,
                                              uint32_t seed) {
        // Load 196-input network
        auto src_nn = std::make_shared<NeuralNetwork>(n_hidden, TESAURO_INPUTS);
        if (!src_nn->load_weights(src_weights)) {
            throw std::runtime_error("Failed to load source weights: " + src_weights);
        }

        // Create 214-input network with small random weights
        auto dst_nn = std::make_shared<NeuralNetwork>(n_hidden, EXTENDED_CONTACT_INPUTS, 0.01f, seed);

        // Copy output weights (identical: n_hidden -> 5)
        dst_nn->output_weights() = src_nn->output_weights();

        // Copy hidden weights: map Tesauro indices to extended indices
        // Tesauro layout:  [0-95] player points, [96] bar, [97] borne-off, [98-193] opp points, [194] bar, [195] borne-off
        // Extended layout:  [0-95] player points, [96] bar, [97-99] 3 borne-off buckets, [100-106] features,
        //                   [107-202] opp points, [203] bar, [204-206] 3 borne-off buckets, [207-213] features
        const int src_stride = TESAURO_INPUTS + 1;  // 197
        const int dst_stride = EXTENDED_CONTACT_INPUTS + 1;  // 215

        for (int h = 0; h < n_hidden; ++h) {
            // Player point weights: src[0-95] -> dst[0-95]
            for (int i = 0; i < 96; ++i) {
                dst_nn->hidden_weights()[h * dst_stride + i] = src_nn->hidden_weights()[h * src_stride + i];
            }
            // Player bar: src[96] -> dst[96]
            dst_nn->hidden_weights()[h * dst_stride + 96] = src_nn->hidden_weights()[h * src_stride + 96];
            // Player borne-off: src[97] -> dst[97] (first bucket), others stay random small
            dst_nn->hidden_weights()[h * dst_stride + 97] = src_nn->hidden_weights()[h * src_stride + 97];
            // New features [98-106] stay random (small init)

            // Opponent point weights: src[98-193] -> dst[107-202]
            for (int i = 0; i < 96; ++i) {
                dst_nn->hidden_weights()[h * dst_stride + 107 + i] = src_nn->hidden_weights()[h * src_stride + 98 + i];
            }
            // Opponent bar: src[194] -> dst[203]
            dst_nn->hidden_weights()[h * dst_stride + 203] = src_nn->hidden_weights()[h * src_stride + 194];
            // Opponent borne-off: src[195] -> dst[204] (first bucket)
            dst_nn->hidden_weights()[h * dst_stride + 204] = src_nn->hidden_weights()[h * src_stride + 195];
            // New features [205-213] stay random (small init)

            // Bias: src[196] -> dst[214]
            dst_nn->hidden_weights()[h * dst_stride + EXTENDED_CONTACT_INPUTS] =
                src_nn->hidden_weights()[h * src_stride + TESAURO_INPUTS];
        }

        // Save
        if (!dst_nn->save_weights(dst_weights)) {
            throw std::runtime_error("Failed to save extended weights: " + dst_weights);
        }
        return true;
    }, "Create 214-input network weights from 196-input Tesauro weights",
       py::arg("src_weights"),
       py::arg("dst_weights"),
       py::arg("n_hidden") = 120,
       py::arg("seed") = 42);

    // --- Strategy base class (for polymorphic dispatch in unified bindings) ---
    py::class_<Strategy, std::shared_ptr<Strategy>>(m, "Strategy")
        .def("evaluate_board", [](Strategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            return evaluate_board_dict(self, board, pre_move_board);
        }, "Evaluate post-move board, returns probs and equity (polymorphic).",
           py::arg("board"), py::arg("pre_move_board"));

    // --- GamePlanStrategy ---
    py::class_<GamePlanStrategy, Strategy, std::shared_ptr<GamePlanStrategy>>(m, "GamePlanStrategy")
        .def(py::init<const std::string&, const std::string&,
                      const std::string&, const std::string&,
                      const std::string&,
                      int, int, int, int, int>(),
             py::arg("purerace_weights"),
             py::arg("racing_weights"),
             py::arg("attacking_weights"),
             py::arg("priming_weights"),
             py::arg("anchoring_weights"),
             py::arg("n_hidden_purerace") = 80,
             py::arg("n_hidden_racing") = 120,
             py::arg("n_hidden_attacking") = 120,
             py::arg("n_hidden_priming") = 120,
             py::arg("n_hidden_anchoring") = 120)
        .def("evaluate_board", [](GamePlanStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            return evaluate_board_dict(self, board, pre_move_board);
        }, "Evaluate board at 1-ply (raw NN), returns probs and equity",
           py::arg("board"), py::arg("pre_move_board"));

    // --- Benchmark scoring with GamePlanStrategy (5-NN) ---
    m.def("score_benchmarks_5nn", [](const ScenarioSet& ss,
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
                                      int n_threads) {
        GamePlanStrategy strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                               n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        py::gil_scoped_release release;
        return score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmarks using 5-NN GamePlanStrategy",
       py::arg("scenarios"),
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 80,
       py::arg("n_hidden_racing") = 120,
       py::arg("n_hidden_attacking") = 120,
       py::arg("n_hidden_priming") = 120,
       py::arg("n_hidden_anchoring") = 120,
       py::arg("n_threads") = 0);

    // --- 5-NN vs PubEval ---
    m.def("play_games_5nn_vs_pubeval", [](const std::string& purerace_w,
                                           const std::string& racing_w,
                                           const std::string& attacking_w,
                                           const std::string& priming_w,
                                           const std::string& anchoring_w,
                                           int n_h_purerace,
                                           int n_h_racing,
                                           int n_h_attacking,
                                           int n_h_priming,
                                           int n_h_anchoring,
                                           int n_games,
                                           uint32_t seed,
                                           PubEval::WeightSource pe_weights,
                                           int n_threads) {
        GamePlanStrategy nn_strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                                  n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(nn_strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: 5-NN GamePlan (p1) vs PubEval (p2)",
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 80,
       py::arg("n_hidden_racing") = 120,
       py::arg("n_hidden_attacking") = 120,
       py::arg("n_hidden_priming") = 120,
       py::arg("n_hidden_anchoring") = 120,
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // 5-NN self-play (same strategy vs itself)
    m.def("play_games_5nn_vs_self", [](const std::string& purerace_w,
                                       const std::string& racing_w,
                                       const std::string& attacking_w,
                                       const std::string& priming_w,
                                       const std::string& anchoring_w,
                                       int n_h_purerace,
                                       int n_h_racing,
                                       int n_h_attacking,
                                       int n_h_priming,
                                       int n_h_anchoring,
                                       int n_games,
                                       uint32_t seed,
                                       int n_threads) {
        GamePlanStrategy strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                               n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        py::gil_scoped_release release;
        return play_games_parallel(strat, strat, n_games, seed, n_threads);
    }, "Play games: 5-NN GamePlan vs itself (self-play outcome distribution)",
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 80,
       py::arg("n_hidden_racing") = 120,
       py::arg("n_hidden_attacking") = 120,
       py::arg("n_hidden_priming") = 120,
       py::arg("n_hidden_anchoring") = 120,
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("n_threads") = 0);

    // --- Encoding utilities for testing/validation ---
    m.def("is_race", [](const std::vector<int>& board) {
        return is_race(list_to_board(board));
    });
    m.def("is_crashed", [](const std::vector<int>& board) {
        return is_crashed(list_to_board(board));
    });
    m.def("classify_position", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        PosType pt = classify_position(b);
        switch (pt) {
            case PosType::CONTACT: return std::string("contact");
            case PosType::CRASHED: return std::string("crashed");
            case PosType::RACE: return std::string("race");
            default: return std::string("unknown");
        }
    });
    m.def("classify_game_plan", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        GamePlan gp = classify_game_plan(b);
        return std::string(game_plan_name(gp));
    });
    m.def("backgame_category", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        switch (backgame_category(b)) {
            case BackgameCategory::DEEP: return std::string("deep");
            case BackgameCategory::MIDDLE: return std::string("middle");
            case BackgameCategory::DOUBLE_ANCHOR: return std::string("double");
            default: return std::string("none");
        }
    }, "Stage 11 backgame category of the position (either side): 'deep' "
       "(21/31/32), 'middle' (41/42/51/52), 'double' (43/53/54), or 'none'. "
       "Applies Stage 9's backgame detection first.",
       py::arg("board"));
    m.def("containment_category", [](const std::vector<int>& board) {
        return containment_category(list_to_board(board));
    }, "True when the position is a containment game (either side the "
       "escaper): the escaper has >= 3 checkers off and 1-3 stragglers that a "
       "container checker can still hit. See scripts/containment_rule.py.",
       py::arg("board"));
    m.def("snake_category", [](const std::vector<int>& board) {
        return snake_category(list_to_board(board));
    }, "True when the position is a snake (either side the holder): a run of "
       ">= 4 consecutive made points on the opponent's half of the board "
       "trapping >= 1 straggler while the opponent has >= 10 checkers home. "
       "See scripts/snake_rule.py.",
       py::arg("board"));
    m.def("massive_category", [](const std::vector<int>& board) {
        return massive_category(list_to_board(board));
    }, "True when the position is a massive back game (either side the "
       "holder): >= 3 anchors in the opponent's home board, or >= 2 anchors "
       "and >= 7 checkers back, behind on pips; never a containment game or a "
       "snake. The benchmark's 'massive backgame' family filter.",
       py::arg("board"));
    m.def("backgame_phase", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        switch (backgame_phase(b)) {
            case BackgamePhase::WAITING: return std::string("waiting");
            case BackgamePhase::BEAR_IN: return std::string("bear_in");
            case BackgamePhase::EARLY_CONTAINMENT: return std::string("early_containment");
            case BackgamePhase::LATE_CONTAINMENT: return std::string("late_containment");
            default: return std::string("none");
        }
    }, "Stage 11 back-game phase of the position (either side), independent of "
       "the plan-pair gate: 'waiting', 'bear_in', 'early_containment', "
       "'late_containment', or 'none' when no side qualifies as the holder.",
       py::arg("board"));
    m.def("classify_game_plans_batch", [](py::array_t<int32_t> boards_np) {
        auto info = boards_np.request();
        if (info.ndim != 2 || info.shape[1] != 26) {
            throw std::runtime_error("boards must be shape [N, 26]");
        }
        int n = static_cast<int>(info.shape[0]);
        const int32_t* ptr = static_cast<const int32_t*>(info.ptr);
        py::array_t<int32_t> result(n);
        auto result_ptr = static_cast<int32_t*>(result.request().ptr);
        for (int i = 0; i < n; ++i) {
            Board b;
            for (int j = 0; j < 26; ++j) b[j] = ptr[i * 26 + j];
            result_ptr[i] = static_cast<int32_t>(classify_game_plan(b));
        }
        return result;
    }, "Classify game plan for a batch of boards. Returns int array: 0=purerace, 1=racing, 2=attacking, 3=priming, 4=anchoring",
       py::arg("boards"));
    m.def("compute_extended_contact_inputs", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        auto inp = compute_extended_contact_inputs(b);
        py::array_t<float> result(EXTENDED_CONTACT_INPUTS);
        std::copy(inp.begin(), inp.end(), result.mutable_data());
        return result;
    });
    // Batch encoding: boards [N, 26] int32 -> inputs [N, n_inputs] float32
    m.def("encode_boards_batch", [](py::array_t<int32_t> boards_np, int n_inputs) {
        auto info = boards_np.request();
        if (info.ndim != 2 || info.shape[1] != 26)
            throw std::runtime_error("boards must be shape [N, 26]");
        int n = static_cast<int>(info.shape[0]);
        const int32_t* data = static_cast<const int32_t*>(info.ptr);

        py::array_t<float> result({n, n_inputs});
        float* out = static_cast<float*>(result.request().ptr);

        {
            py::gil_scoped_release release;
            for (int i = 0; i < n; ++i) {
                Board board;
                for (int j = 0; j < 26; ++j) board[j] = data[i * 26 + j];
                if (n_inputs == TESAURO_INPUTS) {
                    auto inp = compute_tesauro_inputs(board);
                    std::copy(inp.begin(), inp.end(), out + i * n_inputs);
                } else {
                    auto inp = compute_extended_contact_inputs(board);
                    std::copy(inp.data(), inp.data() + n_inputs, out + i * n_inputs);
                }
            }
        }
        return result;
    }, "Batch-encode boards [N,26] int32 -> [N,n_inputs] float32",
       py::arg("boards"), py::arg("n_inputs") = EXTENDED_CONTACT_INPUTS);
    m.def("compute_tesauro_inputs", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        auto inp = compute_tesauro_inputs(b);
        py::array_t<float> result(196);
        std::copy(inp.begin(), inp.end(), result.mutable_data());
        return result;
    });

    // Expose helper functions for validation
    m.def("hitting_shots", [](const std::vector<int>& board) {
        return hitting_shots(list_to_board(board));
    });
    m.def("double_hitting_shots", [](const std::vector<int>& board) {
        return double_hitting_shots(list_to_board(board));
    });
    m.def("back_escapes", [](const std::vector<int>& board) {
        return back_escapes(list_to_board(board));
    });
    m.def("max_point", [](const std::vector<int>& board) {
        return max_point(list_to_board(board));
    });
    m.def("max_anchor_point", [](const std::vector<int>& board) {
        return max_anchor_point(list_to_board(board));
    });
    m.def("prob_no_enter_from_bar", [](const std::vector<int>& board) {
        auto r = prob_no_enter_from_bar(list_to_board(board));
        return py::make_tuple(r.player, r.opponent);
    });
    m.def("forward_anchor_points", [](const std::vector<int>& board) {
        auto r = forward_anchor_points(list_to_board(board));
        return py::make_tuple(r.player, r.opponent);
    });

    // New GNUbg feature helpers for testing/validation
    m.def("break_contact", [](const std::vector<int>& board) {
        return break_contact(list_to_board(board));
    });
    m.def("free_pip", [](const std::vector<int>& board) {
        return free_pip(list_to_board(board));
    });
    m.def("compute_piploss", [](const std::vector<int>& board) {
        return compute_piploss(list_to_board(board));
    });
    m.def("timing_feature", [](const std::vector<int>& board) {
        return timing(list_to_board(board));
    });
    m.def("backbone_feature", [](const std::vector<int>& board) {
        return backbone(list_to_board(board));
    });
    m.def("backg_feature", [](const std::vector<int>& board) {
        return backg(list_to_board(board));
    });
    m.def("backg1_feature", [](const std::vector<int>& board) {
        return backg1(list_to_board(board));
    });
    m.def("enter_loss", [](const std::vector<int>& board) {
        return enter_loss(list_to_board(board));
    });
    m.def("containment_feature", [](const std::vector<int>& board) {
        return containment(list_to_board(board));
    });
    m.def("acontainment_feature", [](const std::vector<int>& board) {
        return acontainment(list_to_board(board));
    });
    m.def("mobility_feature", [](const std::vector<int>& board) {
        return mobility(list_to_board(board));
    });
    m.def("moment2_feature", [](const std::vector<int>& board) {
        return moment2(list_to_board(board));
    });
    m.def("back_rescue_escapes", [](const std::vector<int>& board) {
        return back_rescue_escapes(list_to_board(board));
    });
    m.def("init_escape_tables", &init_escape_tables,
          "Initialize the escape lookup tables (called automatically on first use)");

    // --- Utility functions for debugging ---
    m.def("flip_board", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        auto f = flip(b);
        std::vector<int> result(f.begin(), f.end());
        return result;
    }, "Flip board perspective");

    m.def("possible_moves", [](const std::vector<int>& board, int d1, int d2) {
        auto b = list_to_board(board);
        std::vector<Board> results;
        possible_boards(b, d1, d2, results);
        std::vector<std::vector<int>> out;
        for (const auto& r : results) {
            out.emplace_back(r.begin(), r.end());
        }
        return out;
    }, "Generate all legal post-move boards");

    m.def("possible_moves_unsorted", [](const std::vector<int>& board, int d1, int d2) {
        auto b = list_to_board(board);
        std::vector<Board> results;
        possible_boards_unsorted(b, d1, d2, results);
        std::vector<std::vector<int>> out;
        for (const auto& r : results) {
            out.emplace_back(r.begin(), r.end());
        }
        return out;
    }, "The legal post-move boards in first-seen generation order, as the\n"
       "evaluation hot paths see them (possible_moves sorts them). Same set\n"
       "as possible_moves; exposed so tests can pin the order.");

    neongammon_python::bind_move_arrays(m);

    m.def("possible_single_die_moves", [](const std::vector<int>& board, int die) {
        auto b = list_to_board(board);
        std::vector<Board> results;
        possible_boards_one_die(b, die, results);

        // Build list of {from, to, board} dicts by comparing each result to input
        py::list out;
        for (const auto& nb : results) {
            int from_pt = -1, to_pt = -1;

            // Find which point lost a checker (from)
            if (b[25] > 0 && nb[25] < b[25]) {
                from_pt = 25; // entered from bar
            } else {
                for (int i = 1; i <= 24; ++i) {
                    if (nb[i] < b[i] && b[i] > 0) {
                        from_pt = i;
                        break;
                    }
                }
            }

            // Find which point gained a checker (to), or 0 for bear-off
            for (int i = 1; i <= 24; ++i) {
                // Check if player 1 gained a checker here
                int p1_before = b[i] > 0 ? b[i] : 0;
                int p1_after = nb[i] > 0 ? nb[i] : 0;
                // Also handle hitting: b[i] == -1, nb[i] == 1
                if (b[i] < 0 && nb[i] > 0) p1_after = nb[i];
                if (p1_after > p1_before) {
                    to_pt = i;
                    break;
                }
            }
            if (to_pt == -1) {
                // Must be bearing off
                to_pt = 0;
            }

            py::dict d;
            d["from"] = from_pt;
            d["to"] = to_pt;
            d["board"] = std::vector<int>(nb.begin(), nb.end());
            out.append(d);
        }
        return out;
    }, "Generate all legal single-die moves with from/to info",
       py::arg("board"), py::arg("die"));

    m.def("check_game_over", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        return static_cast<int>(check_game_over(b));
    }, "Check if game is over. Returns 0=not over, 1/2/3=win single/gammon/bg, -1/-2/-3=loss",
       py::arg("board"));

    m.def("invert_probs_py", [](const std::array<float, NUM_OUTPUTS>& p) {
        return invert_probs(p);
    }, "Invert probabilities from one perspective to the other");

    // --- Multi-Ply Strategy ---
    py::class_<MultiPlyStrategy, Strategy, std::shared_ptr<MultiPlyStrategy>>(m, "MultiPlyStrategy")
        .def("n_plies", &MultiPlyStrategy::n_plies)
        .def("clear_cache", &MultiPlyStrategy::clear_cache)
        .def("cache_size", &MultiPlyStrategy::cache_size)
        .def("cache_hits", &MultiPlyStrategy::cache_hits)
        .def("cache_misses", &MultiPlyStrategy::cache_misses)
        .def("set_prefilter_params", &MultiPlyStrategy::set_prefilter_params,
             py::arg("threshold"), py::arg("keep"),
             "PubEval pre-filter of the opponent's replies inside N-ply: when a reply "
             "list is longer than `threshold`, PubEval keeps the best `keep` before the "
             "NN sees them. A huge threshold disables the pre-filter.")
        .def("set_cache_enabled", &MultiPlyStrategy::set_cache_enabled,
             py::arg("enabled") = true)
        .def("cache_enabled", &MultiPlyStrategy::cache_enabled)
        .def("best_move_index", [](MultiPlyStrategy& self,
                                    const py::list& candidates_list,
                                    const std::vector<int>& pre_move_board) {
            std::vector<Board> candidates;
            candidates.reserve(candidates_list.size());
            for (const auto& item : candidates_list) {
                candidates.push_back(list_to_board(item.cast<std::vector<int>>()));
            }
            auto pmb = list_to_board(pre_move_board);
            int idx;
            {
                py::gil_scoped_release release;
                idx = self.best_move_index(candidates, pmb);
            }
            return idx;
        }, "Find best move index using iterative deepening filter chain",
           py::arg("candidates"), py::arg("pre_move_board"))
        .def("filter_chain", [](const MultiPlyStrategy& self) {
            py::list result;
            for (const auto& step : self.filter_chain()) {
                py::dict d;
                d["ply"] = step.ply;
                d["max_moves"] = step.max_moves;
                d["threshold"] = step.threshold;
                result.append(d);
            }
            return result;
        }, "Return the iterative deepening filter chain")
        .def("evaluate_board", [](MultiPlyStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            std::array<float, NUM_OUTPUTS> probs;
            double eq;
            {
                py::gil_scoped_release release;
                auto b = list_to_board(board);
                auto pmb = list_to_board(pre_move_board);
                probs = self.evaluate_probs(b, pmb);
                clamp_probs_to_board(probs, b);
                eq = NeuralNetwork::compute_equity(probs);
            }
            py::dict result;
            result["probs"] = probs;
            result["equity"] = eq;
            return result;
        }, "Evaluate board at N-ply depth, returns probs and equity",
           py::arg("board"), py::arg("pre_move_board"));

    py::class_<MoveFilter>(m, "MoveFilter")
        .def(py::init<>())
        .def(py::init<int, float>(),
             py::arg("max_moves"), py::arg("threshold"))
        .def_readwrite("max_moves", &MoveFilter::max_moves)
        .def_readwrite("threshold", &MoveFilter::threshold);

    m.attr("MOVE_FILTER_TINY")   = MoveFilters::TINY;
    m.attr("MOVE_FILTER_NARROW") = MoveFilters::NARROW;
    m.attr("MOVE_FILTER_NORMAL") = MoveFilters::NORMAL;
    m.attr("MOVE_FILTER_LARGE")  = MoveFilters::LARGE;
    m.attr("MOVE_FILTER_HUGE")   = MoveFilters::HUGE_;

    // Create N-ply strategy wrapping a 5-NN GamePlanStrategy
    m.def("create_multipy_5nn", [](const std::string& purerace_w,
                                    const std::string& racing_w,
                                    const std::string& attacking_w,
                                    const std::string& priming_w,
                                    const std::string& anchoring_w,
                                    int n_h_purerace,
                                    int n_h_racing,
                                    int n_h_attacking,
                                    int n_h_priming,
                                    int n_h_anchoring,
                                    int n_plies,
                                    int filter_max_moves,
                                    float filter_threshold,
                                    bool full_depth_opponent,
                                    bool parallel_evaluate,
                                    int parallel_threads) {
        auto base = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        return make_multipy(std::move(base), n_plies, filter_max_moves,
                            filter_threshold, full_depth_opponent,
                            parallel_evaluate, parallel_threads);
    }, "Create N-ply strategy wrapping 5-NN GamePlanStrategy",
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 120,
       py::arg("n_hidden_racing") = 250,
       py::arg("n_hidden_attacking") = 250,
       py::arg("n_hidden_priming") = 250,
       py::arg("n_hidden_anchoring") = 250,
       py::arg("n_plies") = 2,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("full_depth_opponent") = false,
       py::arg("parallel_evaluate") = false,
       py::arg("parallel_threads") = 0);

    // Create hybrid N-ply strategy with separate filter and leaf 5-NN strategies.
    // The filter strategy (small/fast) is used for 1-ply filtering and opponent
    // move selection. The leaf strategy (large/accurate) is used for leaf evaluations.
    m.def("create_multipy_hybrid_5nn", [](
            // Leaf (accurate) strategy weights
            const std::string& purerace_w,
            const std::string& racing_w,
            const std::string& attacking_w,
            const std::string& priming_w,
            const std::string& anchoring_w,
            int n_h_purerace, int n_h_racing, int n_h_attacking,
            int n_h_priming, int n_h_anchoring,
            // Filter (fast) strategy weights
            const std::string& filter_purerace_w,
            const std::string& filter_racing_w,
            const std::string& filter_attacking_w,
            const std::string& filter_priming_w,
            const std::string& filter_anchoring_w,
            int filter_n_h_purerace, int filter_n_h_racing, int filter_n_h_attacking,
            int filter_n_h_priming, int filter_n_h_anchoring,
            // Multi-ply config
            int n_plies, int filter_max_moves, float filter_threshold,
            bool full_depth_opponent, bool parallel_evaluate, int parallel_threads) {
        auto leaf = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        auto filter = std::make_shared<GamePlanStrategy>(
            filter_purerace_w, filter_racing_w, filter_attacking_w,
            filter_priming_w, filter_anchoring_w,
            filter_n_h_purerace, filter_n_h_racing, filter_n_h_attacking,
            filter_n_h_priming, filter_n_h_anchoring);
        MoveFilter mf{filter_max_moves, filter_threshold};
        auto strat = std::make_shared<MultiPlyStrategy>(
            leaf, filter, n_plies, mf, full_depth_opponent,
            parallel_evaluate, parallel_threads);
        static auto pubeval = std::make_shared<PubEval>();
        strat->set_move_prefilter(pubeval);
        return strat;
    }, "Create hybrid N-ply strategy with separate filter and leaf 5-NN strategies",
       py::arg("purerace_weights"), py::arg("racing_weights"),
       py::arg("attacking_weights"), py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 200, py::arg("n_hidden_racing") = 400,
       py::arg("n_hidden_attacking") = 400, py::arg("n_hidden_priming") = 400,
       py::arg("n_hidden_anchoring") = 400,
       py::arg("filter_purerace_weights"), py::arg("filter_racing_weights"),
       py::arg("filter_attacking_weights"), py::arg("filter_priming_weights"),
       py::arg("filter_anchoring_weights"),
       py::arg("filter_n_hidden_purerace") = 100, py::arg("filter_n_hidden_racing") = 200,
       py::arg("filter_n_hidden_attacking") = 200, py::arg("filter_n_hidden_priming") = 200,
       py::arg("filter_n_hidden_anchoring") = 200,
       py::arg("n_plies") = 2, py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f, py::arg("full_depth_opponent") = false,
       py::arg("parallel_evaluate") = false, py::arg("parallel_threads") = 0);

    // Create N-ply strategy wrapping a single NNStrategy
    m.def("create_multipy_nn", [](const std::string& weights_path,
                                   int n_hidden,
                                   int n_inputs,
                                   int n_plies,
                                   int filter_max_moves,
                                   float filter_threshold,
                                   bool full_depth_opponent,
                                   bool parallel_evaluate,
                                   int parallel_threads) {
        auto base = std::make_shared<NNStrategy>(weights_path, n_hidden, n_inputs);
        MoveFilter filter{filter_max_moves, filter_threshold};
        return std::make_shared<MultiPlyStrategy>(
            base, n_plies, filter, full_depth_opponent,
            parallel_evaluate, parallel_threads);
    }, "Create N-ply strategy wrapping a single NNStrategy",
       py::arg("weights_path"),
       py::arg("n_hidden") = 120,
       py::arg("n_inputs") = 196,
       py::arg("n_plies") = 2,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("full_depth_opponent") = false,
       py::arg("parallel_evaluate") = false,
       py::arg("parallel_threads") = 0);

    // Benchmark scoring with MultiPlyStrategy
    m.def("score_benchmarks_multipy", [](const ScenarioSet& ss,
                                          std::shared_ptr<MultiPlyStrategy> strat,
                                          int n_threads) {
        strat->clear_cache();
        py::gil_scoped_release release;
        return score_benchmarks(*strat, ss.scenarios, n_threads);
    }, "Score benchmarks using multi-ply strategy",
       py::arg("scenarios"),
       py::arg("strategy"),
       py::arg("n_threads") = 0);

    // Multi-ply vs PubEval game simulation
    m.def("play_games_multipy_vs_pubeval", [](std::shared_ptr<MultiPlyStrategy> strat,
                                               int n_games,
                                               uint32_t seed,
                                               PubEval::WeightSource pe_weights,
                                               int n_threads) {
        strat->clear_cache();
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(*strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: multi-ply strategy (p1) vs PubEval (p2)",
       py::arg("strategy"),
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // ======================== Per-scenario benchmark scoring ========================

    m.def("score_benchmarks_per_scenario_5nn", [](const ScenarioSet& ss,
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
                                                   int n_threads) {
        GamePlanStrategy strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                               n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        const int n = static_cast<int>(ss.scenarios.size());
        std::vector<double> errors(n);
        {
            py::gil_scoped_release release;
            score_benchmarks_per_scenario(strat, ss.scenarios, errors.data(), n_threads);
        }
        return errors;
    }, "Score benchmarks per-scenario using 5-NN, returns list of errors",
       py::arg("scenarios"),
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 120,
       py::arg("n_hidden_racing") = 250,
       py::arg("n_hidden_attacking") = 250,
       py::arg("n_hidden_priming") = 250,
       py::arg("n_hidden_anchoring") = 250,
       py::arg("n_threads") = 0);

    // ======================== Benchmark PR scoring ========================

    // Score a 1-ply strategy against benchmark PR data.
    // decisions: list of dicts with 'board', 'dice', 'candidates', 'rollout_equities'
    // Returns a dict with overall PR, per-plan PR, per-plan counts, etc.
    m.def("score_benchmark_pr_1ply", [](py::list decisions_py,
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
                                         int n_threads) {
        std::vector<PRDecision> decisions = pr_decisions_from_py(decisions_py);

        GamePlanStrategy strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                               n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        MoveFilter filter{5, 0.08f};  // TINY (unused for 1-ply but needed by signature)
        PRResult result;
        {
            py::gil_scoped_release release;
            result = score_benchmark_pr(strat, nullptr, decisions, filter, n_threads);
        }

        return pr_result_to_py(result);
    }, "Score 1-ply strategy against benchmark PR data (parallel)",
       py::arg("decisions"),
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 200,
       py::arg("n_hidden_racing") = 400,
       py::arg("n_hidden_attacking") = 400,
       py::arg("n_hidden_priming") = 400,
       py::arg("n_hidden_anchoring") = 400,
       py::arg("n_threads") = 0);

    // Score an N-ply strategy against benchmark PR data.
    // Uses 1-ply pre-filter + full-depth scoring on survivors.
    m.def("score_benchmark_pr_nply", [](py::list decisions_py,
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
                                         int n_plies,
                                         int filter_max_moves,
                                         float filter_threshold,
                                         int n_threads) {
        std::vector<PRDecision> decisions = pr_decisions_from_py(decisions_py);

        auto base = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        MoveFilter filter{filter_max_moves, filter_threshold};
        MultiPlyStrategy strat(base, n_plies, filter);
        strat.clear_cache();

        PRResult result;
        {
            py::gil_scoped_release release;
            result = score_benchmark_pr(strat, base.get(), decisions, filter, n_threads);
        }

        return pr_result_to_py(result);
    }, "Score N-ply strategy against benchmark PR data (parallel)",
       py::arg("decisions"),
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 200,
       py::arg("n_hidden_racing") = 400,
       py::arg("n_hidden_attacking") = 400,
       py::arg("n_hidden_priming") = 400,
       py::arg("n_hidden_anchoring") = 400,
       py::arg("n_plies") = 2,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 0);

    // Score any Strategy object (create_strategy / create_multipy, so every
    // registered model type) against benchmark PR data. For N-ply, pass the
    // N-ply strategy and a 1-ply base_strategy for the pre-filter.
    m.def("score_benchmark_pr_strategy", [](py::list decisions_py,
                                             std::shared_ptr<Strategy> strategy,
                                             std::shared_ptr<Strategy> base_strategy,
                                             int filter_max_moves,
                                             float filter_threshold,
                                             int n_threads) {
        std::vector<PRDecision> decisions = pr_decisions_from_py(decisions_py);
        MoveFilter filter{filter_max_moves, filter_threshold};
        PRResult result;
        {
            py::gil_scoped_release release;
            result = score_benchmark_pr(*strategy, base_strategy.get(), decisions,
                                        filter, n_threads);
        }
        return pr_result_to_py(result);
    }, "Score a Strategy object against benchmark PR data (parallel)",
       py::arg("decisions"),
       py::arg("strategy"),
       py::arg("base_strategy") = nullptr,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 0);

    // ======================== Bearoff Database ========================

    py::class_<BearoffDB, std::shared_ptr<BearoffDB>>(m, "BearoffDB")
        .def(py::init<>())
        .def("load", &BearoffDB::load, "Load bearoff database from file",
             py::arg("path"))
        .def("save", &BearoffDB::save, "Save bearoff database to file",
             py::arg("path"))
        .def("generate", [](BearoffDB& self) {
            py::gil_scoped_release release;
            self.generate();
        }, "Generate the database via backward induction")
        .def("is_loaded", &BearoffDB::is_loaded)
        .def("is_bearoff", [](const BearoffDB& self, const std::vector<int>& board_vec) {
            return self.is_bearoff(list_to_board(board_vec));
        }, "Check if a board position is covered by the bearoff DB",
           py::arg("board"))
        .def("lookup_probs", [](const BearoffDB& self, const std::vector<int>& board_vec,
                                bool post_move) {
            auto probs = self.lookup_probs(list_to_board(board_vec), post_move);
            return std::vector<float>(probs.begin(), probs.end());
        }, "Compute exact cubeless probabilities for a bearoff position.\n"
           "post_move=False: player on points 1-6 rolls first (on-roll/pre-roll).\n"
           "post_move=True: opponent rolls first (post-move, matches NN semantics).",
           py::arg("board"), py::arg("post_move") = false)
        .def("lookup_epc", [](const BearoffDB& self, const std::vector<int>& board_vec,
                               int player) {
            return self.lookup_epc(list_to_board(board_vec), player);
        }, "Get EPC (Effective Pip Count) for one side",
           py::arg("board"), py::arg("player") = 0)
        .def("get_mean_rolls", [](const BearoffDB& self, const std::vector<int>& board_vec,
                                   int player) {
            Board board = list_to_board(board_vec);
            unsigned int idx = (player == 0)
                ? BearoffDB::board_to_player_index(board)
                : BearoffDB::board_to_opponent_index(board);
            return self.get_mean_rolls(idx);
        }, "Get expected rolls to bear off for one side",
           py::arg("board"), py::arg("player") = 0)
        .def_static("position_index", [](const std::vector<int>& checkers) {
            if (checkers.size() != 6)
                throw std::runtime_error("checkers must have exactly 6 elements");
            return BearoffDB::position_index(checkers.data());
        }, "Map a 6-element checker array to a position index",
           py::arg("checkers"))
        .def_static("index_to_position", [](unsigned int index) {
            int checkers[6];
            BearoffDB::index_to_position(index, checkers);
            return std::vector<int>(checkers, checkers + 6);
        }, "Map a position index back to a 6-element checker array",
           py::arg("index"));

    py::class_<BearoffStrategy, Strategy, std::shared_ptr<BearoffStrategy>>(m, "BearoffStrategy")
        .def(py::init<std::shared_ptr<Strategy>, const BearoffDB*>(),
             py::arg("base"), py::arg("db"),
             py::keep_alive<1, 2>(),  // nested wrappers retain their DB through Python
             py::keep_alive<1, 3>())  // BearoffStrategy keeps db alive
        // Root-routing controls, forwarded to the wrapped pair strategy (the
        // analyzer hands out its 1-ply strategy wrapped in this class).
        .def("set_root_pinned", [](BearoffStrategy& self, std::vector<int> nn_indices) {
            auto* bg = dynamic_cast<BackgameAwarePairStrategy*>(
                const_cast<Strategy*>(&self.base()));
            if (!bg) throw std::runtime_error("wrapped strategy has no root routing");
            bg->set_root_pinned(std::move(nn_indices));
        }, py::arg("nn_indices"))
        .def("root_pinned", [](const BearoffStrategy& self) {
            auto* bg = dynamic_cast<const BackgameAwarePairStrategy*>(&self.base());
            return bg ? bg->root_pinned() : std::vector<int>{};
        })
        .def("root_pin_for", [](const BearoffStrategy& self, const std::vector<int>& root) {
            return self.root_pin_for(list_to_board(root));
        }, py::arg("root"));

    // Setter for MultiPlyStrategy bearoff DB
    m.def("multipy_set_bearoff_db", [](MultiPlyStrategy& strat, const BearoffDB* db) {
        strat.set_bearoff_db(db);
    }, "Set bearoff DB on a MultiPlyStrategy for exact bearoff evaluation",
       py::arg("strategy"), py::arg("db"));

    py::class_<GamePlanPairStrategy, Strategy, std::shared_ptr<GamePlanPairStrategy>>(m, "GamePlanPairStrategy")
        .def(py::init<const std::vector<std::string>&, const std::vector<int>&>(),
             py::arg("weight_paths"), py::arg("hidden_sizes"),
             "Create 17-NN strategy: weight_paths[0]=purerace, [1-16]=contact pairs; "
             "hidden_sizes[0-16] corresponding hidden layer sizes")
        .def("evaluate_board", [](GamePlanPairStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            return evaluate_board_dict(self, board, pre_move_board);
        }, "Evaluate board at 1-ply (raw NN), returns probs and equity",
           py::arg("board"), py::arg("pre_move_board"));

    // --- Pair strategy scoring ---
    m.def("score_benchmarks_pair", [](const ScenarioSet& ss,
                                      const std::vector<std::string>& weight_paths,
                                      const std::vector<int>& hidden_sizes,
                                      int n_threads) {
        GamePlanPairStrategy strat(weight_paths, hidden_sizes);
        py::gil_scoped_release release;
        return score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmarks using 17-NN GamePlanPairStrategy",
       py::arg("scenarios"),
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("n_threads") = 0);

    m.def("score_benchmarks_per_scenario_pair", [](const ScenarioSet& ss,
                                                   const std::vector<std::string>& weight_paths,
                                                   const std::vector<int>& hidden_sizes,
                                                   int n_threads) {
        GamePlanPairStrategy strat(weight_paths, hidden_sizes);
        const int n = static_cast<int>(ss.scenarios.size());
        std::vector<double> errors(n);
        {
            py::gil_scoped_release release;
            score_benchmarks_per_scenario(strat, ss.scenarios, errors.data(), n_threads);
        }
        return errors;
    }, "Score benchmarks per-scenario using 17-NN Pair strategy, returns list of errors",
       py::arg("scenarios"),
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("n_threads") = 0);

    // --- Pair strategy vs PubEval ---
    m.def("play_games_pair_vs_pubeval", [](const std::vector<std::string>& weight_paths,
                                            const std::vector<int>& hidden_sizes,
                                            int n_games, uint32_t seed,
                                            PubEval::WeightSource pe_weights,
                                            int n_threads) {
        GamePlanPairStrategy nn_strat(weight_paths, hidden_sizes);
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(nn_strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: 17-NN Pair (p1) vs PubEval (p2)",
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- Pair strategy self-play ---
    m.def("play_games_pair_vs_self", [](const std::vector<std::string>& weight_paths,
                                         const std::vector<int>& hidden_sizes,
                                         int n_games, uint32_t seed,
                                         int n_threads) {
        GamePlanPairStrategy strat(weight_paths, hidden_sizes);
        py::gil_scoped_release release;
        return play_games_parallel(strat, strat, n_games, seed, n_threads);
    }, "Play games: 17-NN Pair vs itself (self-play outcome distribution)",
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("n_threads") = 0);

    // --- Pair strategy multipy creation ---
    m.def("create_multipy_pair", [](const std::vector<std::string>& weight_paths,
                                     const std::vector<int>& hidden_sizes,
                                     int n_plies,
                                     int filter_max_moves, float filter_threshold,
                                     bool full_depth_opponent,
                                     bool parallel_evaluate, int parallel_threads) {
        auto base = std::make_shared<GamePlanPairStrategy>(weight_paths, hidden_sizes);
        MoveFilter filter{filter_max_moves, filter_threshold};
        auto strat = std::make_shared<MultiPlyStrategy>(
            base, n_plies, filter, full_depth_opponent,
            parallel_evaluate, parallel_threads);
        static auto pubeval = std::make_shared<PubEval>();
        strat->set_move_prefilter(pubeval);
        return strat;
    }, "Create MultiPlyStrategy from 17-NN Pair base strategy",
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("n_plies") = 2,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("full_depth_opponent") = false,
       py::arg("parallel_evaluate") = false,
       py::arg("parallel_threads") = 0);

    // --- Pair classification batch ---
    m.def("classify_game_plan_pairs_batch", [](py::array_t<int32_t> boards_np) {
        auto info = boards_np.request();
        if (info.ndim != 2 || info.shape[1] != 26)
            throw std::runtime_error("boards must be shape [N, 26]");
        int n = static_cast<int>(info.shape[0]);
        const int32_t* data = static_cast<const int32_t*>(info.ptr);

        py::array_t<int32_t> player_out(n);
        py::array_t<int32_t> opponent_out(n);
        auto p_ptr = static_cast<int32_t*>(player_out.request().ptr);
        auto o_ptr = static_cast<int32_t*>(opponent_out.request().ptr);

        for (int i = 0; i < n; ++i) {
            Board board;
            for (int j = 0; j < 26; ++j) board[j] = data[i * 26 + j];
            p_ptr[i] = static_cast<int32_t>(classify_game_plan(board));
            o_ptr[i] = static_cast<int32_t>(classify_game_plan(flip(board)));
        }
        return py::make_tuple(player_out, opponent_out);
    }, "Classify (player, opponent) game plans for a batch of boards. "
       "Returns (player_plans, opponent_plans) as int32 arrays (0-4).",
       py::arg("boards"));

    // --- Pair name utilities ---
    m.def("game_plan_pair_names", []() {
        const auto& names = game_plan_pair_names();
        py::list result;
        for (int i = 0; i < NUM_PAIR_NNS; ++i)
            result.append(std::string(names[i]));
        return result;
    }, "Return list of 17 pair names (purerace, race_race, race_att, ...)");

    py::class_<BackgameAwarePairStrategy, Strategy, std::shared_ptr<BackgameAwarePairStrategy>>(m, "BackgameAwarePairStrategy")
        .def(py::init<const std::vector<std::string>&, const std::vector<int>&, bool>(),
             py::arg("weight_paths"), py::arg("hidden_sizes"),
             py::arg("phase_containment") = false,
             "Create 19-NN strategy: [0]=purerace, [1-16]=contact pairs, "
             "[17]=player backgame, [18]=opponent backgame. 20 NNs = the Stage 11 "
             "category trio; 21 NNs = the Stage 10 hybrid, or with "
             "phase_containment=True the Stage 11 phased layout (trio + early-"
             "containment NN at 20 + containment NN at 21; 23 NNs add the snake "
             "NN at 22)")
        .def("evaluate_board", [](BackgameAwarePairStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            return evaluate_board_dict(self, board, pre_move_board);
        }, "Evaluate board at 1-ply (raw NN), returns probs and equity",
           py::arg("board"), py::arg("pre_move_board"))
        .def("set_root_pinned", &BackgameAwarePairStrategy::set_root_pinned, py::arg("nn_indices"),
             "Root routing: NN indices that pin a whole search tree rooted in their region "
             "(default [22], the snake net, on the 23-NN layout); [] disables it")
        .def("root_pinned", [](const BackgameAwarePairStrategy& self) { return self.root_pinned(); })
        .def("root_pin_for", [](const BackgameAwarePairStrategy& self, const std::vector<int>& root) {
            return self.root_pin_for(list_to_board(root));
        }, py::arg("root"), "The NN index a search tree rooted at `root` is pinned to, or -1")
        .def("select_nn_idx", [](BackgameAwarePairStrategy& self,
                                  const std::vector<int>& board) {
            // Delegate to the real routing (public accessor) so this never
            // drifts from evaluate_*. Returns 0-16 contact, 17/18 backgame; on
            // 21-NN hybrids, the blend sentinels 21 (player BG) / 22 (opp BG).
            return self.nn_index_for(list_to_board(board));
        }, "Return the NN index the strategy uses for a board position "
           "(17/18 = backgame; 21/22 = blended backgame on hybrid models).",
           py::arg("board"));

    // --- Stage 9 scoring ---
    m.def("score_benchmarks_stage9", [](const ScenarioSet& ss,
                                        const std::vector<std::string>& weight_paths,
                                        const std::vector<int>& hidden_sizes,
                                        int n_threads) {
        BackgameAwarePairStrategy strat(weight_paths, hidden_sizes);
        py::gil_scoped_release release;
        return score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmarks using 19-NN BackgameAwarePairStrategy",
       py::arg("scenarios"),
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("n_threads") = 0);

    // --- Stage 9 multipy creation ---
    m.def("create_multipy_stage9", [](const std::vector<std::string>& weight_paths,
                                       const std::vector<int>& hidden_sizes,
                                       int n_plies,
                                       int filter_max_moves, float filter_threshold,
                                       bool full_depth_opponent,
                                       bool parallel_evaluate, int parallel_threads) {
        auto base = std::make_shared<BackgameAwarePairStrategy>(weight_paths, hidden_sizes);
        MoveFilter filter{filter_max_moves, filter_threshold};
        auto strat = std::make_shared<MultiPlyStrategy>(
            base, n_plies, filter, full_depth_opponent,
            parallel_evaluate, parallel_threads);
        static auto pubeval = std::make_shared<PubEval>();
        strat->set_move_prefilter(pubeval);
        return strat;
    }, "Create MultiPlyStrategy from 19-NN Backgame-Aware Pair base strategy",
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("n_plies") = 2,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("full_depth_opponent") = false,
       py::arg("parallel_evaluate") = false,
       py::arg("parallel_threads") = 0);

    m.def("create_strategy", [](const std::string& strategy_type,
                                               const std::vector<std::string>& weight_paths,
                                               const std::vector<int>& hidden_sizes) {
        py::gil_scoped_release release;
        return make_strategy_from_type(strategy_type, weight_paths, hidden_sizes);
    }, "Create a base Strategy by type (5nn, pair, backgame_pair)",
       py::arg("strategy_type"),
       py::arg("weight_paths"),
       py::arg("hidden_sizes"));

    m.def("create_multipy", [](const std::string& strategy_type,
                                              const std::vector<std::string>& weight_paths,
                                              const std::vector<int>& hidden_sizes,
                                              int n_plies,
                                              int filter_max_moves, float filter_threshold,
                                              bool full_depth_opponent,
                                              bool parallel_evaluate, int parallel_threads) {
        py::gil_scoped_release release;
        return make_multipy(
            make_strategy_from_type(strategy_type, weight_paths, hidden_sizes),
            n_plies, filter_max_moves, filter_threshold, full_depth_opponent,
            parallel_evaluate, parallel_threads);
    }, "Create MultiPlyStrategy from any base strategy type",
       py::arg("strategy_type"),
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("n_plies") = 2,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("full_depth_opponent") = false,
       py::arg("parallel_evaluate") = false,
       py::arg("parallel_threads") = 0);

    m.def("create_multipy_from_base", [](std::shared_ptr<Strategy> base,
                                          int n_plies,
                                          int filter_max_moves, float filter_threshold,
                                          bool full_depth_opponent,
                                          bool parallel_evaluate, int parallel_threads) {
        if (!base) throw std::invalid_argument("base strategy is None");
        return make_multipy(std::move(base), n_plies, filter_max_moves,
                            filter_threshold, full_depth_opponent,
                            parallel_evaluate, parallel_threads);
    }, "Create MultiPlyStrategy around an already loaded base strategy (e.g.\n"
       "from create_strategy), so no weights are loaded again. The result is\n"
       "what create_multipy returns for the same weights.",
       py::arg("base"),
       py::arg("n_plies") = 2,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("full_depth_opponent") = false,
       py::arg("parallel_evaluate") = false,
       py::arg("parallel_threads") = 0);
}

} // namespace bgbot::python
