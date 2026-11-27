// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bindings_common.h"
#include <pybind11/numpy.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include "bgbot/board.h"
#include "bgbot/encoding.h"
#include "bgbot/neural_net.h"
#include "bgbot/training.h"

namespace bgbot::python {

void bind_training(py::module_& m) {
    // --- TD Training ---
    py::class_<TrainingHistoryEntry>(m, "TrainingHistoryEntry")
        .def_readonly("game_number", &TrainingHistoryEntry::game_number)
        .def_readonly("contact_score", &TrainingHistoryEntry::contact_score)
        .def_readonly("elapsed_seconds", &TrainingHistoryEntry::elapsed_seconds);

    py::class_<TDTrainResult>(m, "TDTrainResult")
        .def_readonly("games_played", &TDTrainResult::games_played)
        .def_readonly("total_seconds", &TDTrainResult::total_seconds)
        .def_readonly("history", &TDTrainResult::history);

    m.def("td_train", [](int n_games, float alpha, int n_hidden, float eps,
                          uint32_t seed, int benchmark_interval,
                          const std::string& model_name,
                          const std::string& models_dir,
                          const std::string& resume_from,
                          std::shared_ptr<ScenarioSet> ss) {
        TDTrainConfig config;
        config.n_games = n_games;
        config.alpha = alpha;
        config.n_hidden = n_hidden;
        config.weight_init_eps = eps;
        config.seed = seed;
        config.benchmark_interval = benchmark_interval;
        config.model_name = model_name;
        config.models_dir = models_dir;
        config.resume_from = resume_from;
        if (ss && !ss->scenarios.empty()) {
            config.benchmark_scenarios = &ss->scenarios;
        }
        return td_train(config);
    }, "Run TD(0) self-play training",
       py::arg("n_games") = 5000,
       py::arg("alpha") = 0.1f,
       py::arg("n_hidden") = 120,
       py::arg("eps") = 0.1f,
       py::arg("seed") = 42,
       py::arg("benchmark_interval") = 1000,
       py::arg("model_name") = "td_test",
       py::arg("models_dir") = "models",
       py::arg("resume_from") = "",
       py::arg("scenarios") = std::shared_ptr<ScenarioSet>(nullptr));

    // --- Paskogammon single-NN TD Training (244-input extended contact) ---
    m.def("td_train_pasko", [](int n_games, float alpha, int n_hidden, float eps,
                                uint32_t seed, int benchmark_interval,
                                const std::string& model_name,
                                const std::string& models_dir,
                                const std::string& resume_from,
                                const std::vector<int>& start_board,
                                const std::vector<std::vector<int>>& bench_boards,
                                const std::vector<float>& bench_targets) {
        PaskoTDTrainConfig config;
        config.n_games = n_games;
        config.alpha = alpha;
        config.n_hidden = n_hidden;
        config.weight_init_eps = eps;
        config.seed = seed;
        config.benchmark_interval = benchmark_interval;
        config.model_name = model_name;
        config.models_dir = models_dir;
        config.resume_from = resume_from;
        config.start_board = list_to_board(start_board);

        if (bench_boards.size() != bench_targets.size()) {
            throw std::runtime_error(
                "td_train_pasko: bench_boards and bench_targets size mismatch");
        }
        std::vector<EquityBenchmarkEntry> bench;
        bench.reserve(bench_boards.size());
        for (size_t i = 0; i < bench_boards.size(); ++i) {
            bench.push_back({list_to_board(bench_boards[i]), bench_targets[i]});
        }
        if (!bench.empty()) config.benchmark = &bench;

        py::gil_scoped_release release;
        return td_train_pasko(config);
    }, "Run single-NN (244-input) TD self-play from a fixed start position "
       "(Paskogammon: positive player always on roll, opening doubles allowed)",
       py::arg("n_games") = 5000,
       py::arg("alpha") = 0.1f,
       py::arg("n_hidden") = 400,
       py::arg("eps") = 0.1f,
       py::arg("seed") = 42,
       py::arg("benchmark_interval") = 10000,
       py::arg("model_name") = "td_pasko",
       py::arg("models_dir") = "models",
       py::arg("resume_from") = "",
       py::arg("start_board"),
       py::arg("bench_boards") = std::vector<std::vector<int>>(),
       py::arg("bench_targets") = std::vector<float>());

    // --- Stage 11 truncated backgame TD training ---
    m.def("td_train_backgame_truncated", [](
            int n_games, float alpha, int n_hidden, float eps,
            uint32_t seed, int benchmark_interval,
            const std::string& model_name,
            const std::string& models_dir,
            const std::string& resume_from,
            const std::vector<std::vector<int>>& start_boards,
            bool randomize_first_mover,
            int max_half_moves,
            bool anchor_boundary,
            int ref_move_games,
            int boundary_extra_plies,
            const std::vector<std::string>& ref_weight_paths,
            const std::vector<int>& ref_hidden_sizes,
            int ref_plies,
            int ref_threads,
            const std::vector<std::vector<int>>& bench_boards,
            const std::vector<float>& bench_targets) {
        BackgameTDTrainConfig config;
        config.n_games = n_games;
        config.alpha = alpha;
        config.n_hidden = n_hidden;
        config.weight_init_eps = eps;
        config.seed = seed;
        config.benchmark_interval = benchmark_interval;
        config.model_name = model_name;
        config.models_dir = models_dir;
        config.resume_from = resume_from;
        config.randomize_first_mover = randomize_first_mover;
        config.max_half_moves = max_half_moves;
        config.anchor_boundary = anchor_boundary;
        config.ref_move_games = ref_move_games;
        config.boundary_extra_plies = boundary_extra_plies;
        config.ref_weight_paths = ref_weight_paths;
        config.ref_hidden_sizes = ref_hidden_sizes;
        config.ref_plies = ref_plies;
        config.ref_threads = ref_threads;

        if (start_boards.empty())
            throw std::runtime_error(
                "td_train_backgame_truncated: start_boards is empty");
        config.start_boards.reserve(start_boards.size());
        for (const auto& b : start_boards)
            config.start_boards.push_back(list_to_board(b));

        if (bench_boards.size() != bench_targets.size()) {
            throw std::runtime_error(
                "td_train_backgame_truncated: bench_boards and bench_targets "
                "size mismatch");
        }
        std::vector<EquityBenchmarkEntry> bench;
        bench.reserve(bench_boards.size());
        for (size_t i = 0; i < bench_boards.size(); ++i) {
            bench.push_back({list_to_board(bench_boards[i]), bench_targets[i]});
        }
        if (!bench.empty()) config.benchmark = &bench;

        py::gil_scoped_release release;
        return td_train_backgame_truncated(config);
    }, "Truncated TD self-play for one Stage 11 backgame category NN: games "
       "start from the given backgame positions and end (with a reference-"
       "model N-ply cubeless eval as the terminal target) when the position "
       "leaves every backgame category",
       py::arg("n_games") = 5000,
       py::arg("alpha") = 0.1f,
       py::arg("n_hidden") = 400,
       py::arg("eps") = 0.1f,
       py::arg("seed") = 42,
       py::arg("benchmark_interval") = 10000,
       py::arg("model_name") = "td_s11_bg",
       py::arg("models_dir") = "models",
       py::arg("resume_from") = "",
       py::arg("start_boards"),
       py::arg("randomize_first_mover") = true,
       py::arg("max_half_moves") = 2000,
       py::arg("anchor_boundary") = false,
       py::arg("ref_move_games") = 0,
       py::arg("boundary_extra_plies") = 0,
       py::arg("ref_weight_paths"),
       py::arg("ref_hidden_sizes"),
       py::arg("ref_plies") = 3,
       py::arg("ref_threads") = 0,
       py::arg("bench_boards") = std::vector<std::vector<int>>(),
       py::arg("bench_targets") = std::vector<float>());

    // --- Multi-Network TD Training ---
    m.def("td_train_multi", [](int n_games, float alpha,
                                int n_hidden_contact, int n_hidden_crashed, int n_hidden_race,
                                float eps,
                                uint32_t seed, int benchmark_interval,
                                const std::string& model_name,
                                const std::string& models_dir,
                                const std::string& resume_contact,
                                const std::string& resume_crashed,
                                const std::string& resume_race,
                                std::shared_ptr<ScenarioSet> contact_ss) {
        MultiTDTrainConfig config;
        config.n_games = n_games;
        config.alpha = alpha;
        config.n_hidden_contact = n_hidden_contact;
        config.n_hidden_crashed = n_hidden_crashed;
        config.n_hidden_race = n_hidden_race;
        config.weight_init_eps = eps;
        config.seed = seed;
        config.benchmark_interval = benchmark_interval;
        config.model_name = model_name;
        config.models_dir = models_dir;
        config.resume_contact = resume_contact;
        config.resume_crashed = resume_crashed;
        config.resume_race = resume_race;
        if (contact_ss && !contact_ss->scenarios.empty()) {
            config.contact_benchmark = &contact_ss->scenarios;
        }
        py::gil_scoped_release release;
        return td_train_multi(config);
    }, "Run multi-network TD(0) self-play training (contact+crashed 214-input, race 196-input)",
       py::arg("n_games") = 5000,
       py::arg("alpha") = 0.1f,
       py::arg("n_hidden_contact") = 120,
       py::arg("n_hidden_crashed") = 120,
       py::arg("n_hidden_race") = 80,
       py::arg("eps") = 0.1f,
       py::arg("seed") = 42,
       py::arg("benchmark_interval") = 1000,
       py::arg("model_name") = "td_multi",
       py::arg("models_dir") = "models",
       py::arg("resume_contact") = "",
       py::arg("resume_crashed") = "",
       py::arg("resume_race") = "",
       py::arg("contact_benchmark") = std::shared_ptr<ScenarioSet>(nullptr));

    // --- Supervised Learning ---
    m.def("supervised_train", [](CArray<int32_t> boards_np,
                                  CArray<float> targets_np,
                                  const std::string& weights_path,
                                  int n_hidden,
                                  int n_inputs,
                                  float alpha,
                                  int epochs,
                                  int batch_size,
                                  uint32_t seed,
                                  int print_interval,
                                  const std::string& save_path,
                                  std::shared_ptr<ScenarioSet> benchmark_ss) {
        if (n_inputs < 1 || n_inputs > EXTENDED_CONTACT_INPUTS) {
            throw std::invalid_argument(
                "n_inputs must be between 1 and " + std::to_string(EXTENDED_CONTACT_INPUTS));
        }
        // Validate input shapes
        auto boards_info = boards_np.request();
        auto targets_info = targets_np.request();
        if (boards_info.ndim != 2 || boards_info.shape[1] != 26) {
            throw std::runtime_error("boards must be shape [N, 26]");
        }
        if (targets_info.ndim != 2 || targets_info.shape[1] != 5) {
            throw std::runtime_error("targets must be shape [N, 5]");
        }
        int n_positions = static_cast<int>(boards_info.shape[0]);
        if (targets_info.shape[0] != n_positions) {
            throw std::runtime_error("boards and targets must have same number of rows");
        }

        const int32_t* boards_ptr = static_cast<const int32_t*>(boards_info.ptr);
        const float* targets_ptr = static_cast<const float*>(targets_info.ptr);

        // Create or load neural network
        auto nn = std::make_shared<NeuralNetwork>(n_hidden, n_inputs, 0.1f, seed);
        if (!weights_path.empty()) {
            if (!nn->load_weights(weights_path)) {
                throw std::runtime_error("Failed to load weights: " + weights_path);
            }
        }

        // Pre-encode all positions
        std::vector<float> all_inputs(n_positions * n_inputs);
        for (int i = 0; i < n_positions; ++i) {
            Board b;
            for (int j = 0; j < 26; ++j) {
                b[j] = boards_ptr[i * 26 + j];
            }
            if (n_inputs == TESAURO_INPUTS) {
                auto inp = compute_tesauro_inputs(b);
                std::copy(inp.begin(), inp.end(), &all_inputs[i * n_inputs]);
            } else {
                auto inp = compute_extended_contact_inputs(b);
                std::copy(inp.data(), inp.data() + n_inputs, &all_inputs[i * n_inputs]);
            }
        }

        // Create strategy for benchmarking
        // For 214-input networks, use MultiNNStrategy so it encodes correctly
        std::shared_ptr<Strategy> strat_ptr;
        if (n_inputs == EXTENDED_CONTACT_INPUTS) {
            // Contact network: wrap in MultiNNStrategy with a dummy race NN
            auto dummy_race = std::make_shared<NeuralNetwork>(n_hidden, TESAURO_INPUTS, 0.1f, seed);
            strat_ptr = std::make_shared<MultiNNStrategy>(nn, nullptr, dummy_race);
        } else {
            strat_ptr = std::make_shared<NNStrategy>(nn);
        }
        const Strategy& strat = *strat_ptr;
        const std::vector<BenchmarkScenario>* bm_scenarios = nullptr;
        if (benchmark_ss && !benchmark_ss->scenarios.empty()) {
            bm_scenarios = &benchmark_ss->scenarios;
        }

        // Shuffle indices
        std::vector<int> indices(n_positions);
        for (int i = 0; i < n_positions; ++i) indices[i] = i;
        std::mt19937 rng(seed);

        py::gil_scoped_release release;

        auto t_start = std::chrono::steady_clock::now();
        double best_score = 1e9;
        std::string best_weights_path = save_path.empty() ? "" : save_path + ".best";

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Shuffle
            std::shuffle(indices.begin(), indices.end(), rng);

            // Mini-batch SGD
            double epoch_loss = 0.0;
            for (int start = 0; start < n_positions; start += batch_size) {
                int end = std::min(start + batch_size, n_positions);
                for (int b = start; b < end; ++b) {
                    int idx = indices[b];
                    const float* inp = &all_inputs[idx * n_inputs];
                    std::array<float, NN_OUTPUTS> target;
                    for (int k = 0; k < 5; ++k) {
                        target[k] = targets_ptr[idx * 5 + k];
                    }
                    auto pred = nn->forward_with_gradients(inp);
                    nn->td_update(target, alpha);

                    // Accumulate loss for monitoring
                    for (int k = 0; k < 5; ++k) {
                        float d = pred[k] - target[k];
                        epoch_loss += d * d;
                    }
                }
            }
            epoch_loss /= n_positions;

            auto t_now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();

            // Benchmark
            double bm_score = -1.0;
            if (bm_scenarios && !bm_scenarios->empty()) {
                BenchmarkResult bm = score_benchmarks(strat, *bm_scenarios, 0);
                bm_score = bm.score();
            }

            if (epoch % print_interval == 0 || epoch == epochs - 1) {
                std::ostringstream message;
                message << "Epoch " << std::setw(4) << epoch
                        << "  loss=" << std::fixed << std::setprecision(6) << epoch_loss
                        << "  score=" << std::setprecision(2) << bm_score
                        << "  time=" << std::setprecision(1) << elapsed << "s";
                std::cout << message.str() << std::endl;
            }

            // Save best
            if (bm_score >= 0 && bm_score < best_score) {
                best_score = bm_score;
                if (!best_weights_path.empty()) {
                    nn->save_weights(best_weights_path);
                }
            }

            // Save periodic
            if (!save_path.empty() && (epoch % print_interval == 0 || epoch == epochs - 1)) {
                nn->save_weights(save_path);
            }
        }

        // Return results as dict
        py::gil_scoped_acquire acquire;
        py::dict result;
        result["best_score"] = best_score;
        result["epochs"] = epochs;
        auto t_end = std::chrono::steady_clock::now();
        result["total_seconds"] = std::chrono::duration<double>(t_end - t_start).count();
        if (!save_path.empty()) {
            result["weights_path"] = save_path;
            result["best_weights_path"] = best_weights_path;
        }
        return result;
    }, "Run supervised learning training",
       py::arg("boards"),
       py::arg("targets"),
       py::arg("weights_path") = "",
       py::arg("n_hidden") = 120,
       py::arg("n_inputs") = NN_INPUTS,
       py::arg("alpha") = 0.01f,
       py::arg("epochs") = 100,
       py::arg("batch_size") = 512,
       py::arg("seed") = 42,
       py::arg("print_interval") = 1,
       py::arg("save_path") = "",
       py::arg("benchmark_scenarios") = std::shared_ptr<ScenarioSet>(nullptr));

    // --- Game Plan TD Training ---
    m.def("td_train_gameplan", [](int n_games, float alpha,
                                   int n_hidden_purerace,
                                   int n_hidden_racing, int n_hidden_attacking,
                                   int n_hidden_priming, int n_hidden_anchoring,
                                   float eps, uint32_t seed, int benchmark_interval,
                                   const std::string& model_name,
                                   const std::string& models_dir,
                                   const std::string& resume_purerace,
                                   const std::string& resume_racing,
                                   const std::string& resume_attacking,
                                   const std::string& resume_priming,
                                   const std::string& resume_anchoring,
                                   std::shared_ptr<ScenarioSet> purerace_ss,
                                   std::shared_ptr<ScenarioSet> attacking_ss,
                                   std::shared_ptr<ScenarioSet> priming_ss,
                                   std::shared_ptr<ScenarioSet> anchoring_ss,
                                   std::shared_ptr<ScenarioSet> race_ss) {
        GamePlanTDTrainConfig config;
        config.n_games = n_games;
        config.alpha = alpha;
        config.n_hidden_purerace = n_hidden_purerace;
        config.n_hidden_racing = n_hidden_racing;
        config.n_hidden_attacking = n_hidden_attacking;
        config.n_hidden_priming = n_hidden_priming;
        config.n_hidden_anchoring = n_hidden_anchoring;
        config.weight_init_eps = eps;
        config.seed = seed;
        config.benchmark_interval = benchmark_interval;
        config.model_name = model_name;
        config.models_dir = models_dir;
        config.resume_purerace = resume_purerace;
        config.resume_racing = resume_racing;
        config.resume_attacking = resume_attacking;
        config.resume_priming = resume_priming;
        config.resume_anchoring = resume_anchoring;
        if (purerace_ss && !purerace_ss->scenarios.empty())
            config.purerace_benchmark = &purerace_ss->scenarios;
        if (attacking_ss && !attacking_ss->scenarios.empty())
            config.attacking_benchmark = &attacking_ss->scenarios;
        if (priming_ss && !priming_ss->scenarios.empty())
            config.priming_benchmark = &priming_ss->scenarios;
        if (anchoring_ss && !anchoring_ss->scenarios.empty())
            config.anchoring_benchmark = &anchoring_ss->scenarios;
        if (race_ss && !race_ss->scenarios.empty())
            config.race_benchmark = &race_ss->scenarios;
        py::gil_scoped_release release;
        return td_train_gameplan(config);
    }, "Run 5-network game plan TD(0) self-play training",
       py::arg("n_games") = 5000,
       py::arg("alpha") = 0.1f,
       py::arg("n_hidden_purerace") = 80,
       py::arg("n_hidden_racing") = 120,
       py::arg("n_hidden_attacking") = 120,
       py::arg("n_hidden_priming") = 120,
       py::arg("n_hidden_anchoring") = 120,
       py::arg("eps") = 0.1f,
       py::arg("seed") = 42,
       py::arg("benchmark_interval") = 1000,
       py::arg("model_name") = "td_gameplan",
       py::arg("models_dir") = "models",
       py::arg("resume_purerace") = "",
       py::arg("resume_racing") = "",
       py::arg("resume_attacking") = "",
       py::arg("resume_priming") = "",
       py::arg("resume_anchoring") = "",
       py::arg("purerace_benchmark") = std::shared_ptr<ScenarioSet>(nullptr),
       py::arg("attacking_benchmark") = std::shared_ptr<ScenarioSet>(nullptr),
       py::arg("priming_benchmark") = std::shared_ptr<ScenarioSet>(nullptr),
       py::arg("anchoring_benchmark") = std::shared_ptr<ScenarioSet>(nullptr),
       py::arg("race_benchmark") = std::shared_ptr<ScenarioSet>(nullptr));

    // --- Pair TD training ---
    m.def("td_train_gameplan_pair", [](int n_games, float alpha,
                                        std::vector<int> hidden_sizes,
                                        float eps, uint32_t seed,
                                        int benchmark_interval,
                                        const std::string& model_name,
                                        const std::string& models_dir,
                                        std::vector<std::string> resume_paths,
                                        py::list benchmarks_py,
                                        std::vector<int> canonical_map) {
        if (static_cast<int>(hidden_sizes.size()) != NUM_PAIR_NNS)
            throw std::runtime_error("hidden_sizes must have " + std::to_string(NUM_PAIR_NNS) + " elements");

        GamePlanPairTDTrainConfig config;
        config.n_games = n_games;
        config.alpha = alpha;
        for (int i = 0; i < NUM_PAIR_NNS; ++i)
            config.hidden_sizes[i] = hidden_sizes[i];
        config.weight_init_eps = eps;
        config.seed = seed;
        config.benchmark_interval = benchmark_interval;
        config.model_name = model_name;
        config.models_dir = models_dir;

        // Pad resume_paths to 17 if shorter
        resume_paths.resize(NUM_PAIR_NNS, "");
        for (int i = 0; i < NUM_PAIR_NNS; ++i)
            config.resume_paths[i] = resume_paths[i];

        // Canonical map for NN sharing
        if (!canonical_map.empty()) {
            if (static_cast<int>(canonical_map.size()) != NUM_PAIR_NNS)
                throw std::runtime_error("canonical_map must have " + std::to_string(NUM_PAIR_NNS) + " elements");
            for (int i = 0; i < NUM_PAIR_NNS; ++i)
                config.canonical_map[i] = canonical_map[i];
        }

        // Convert benchmarks list (ScenarioSet or None for each)
        std::vector<std::shared_ptr<ScenarioSet>> bm_holders;
        for (int i = 0; i < NUM_PAIR_NNS && i < static_cast<int>(py::len(benchmarks_py)); ++i) {
            auto item = benchmarks_py[i];
            if (!item.is_none()) {
                auto ss = item.cast<std::shared_ptr<ScenarioSet>>();
                bm_holders.push_back(ss);
                if (!ss->scenarios.empty())
                    config.benchmarks[i] = &ss->scenarios;
            }
        }

        py::gil_scoped_release release;
        return td_train_gameplan_pair(config);
    }, "Run 17-network game plan pair TD(0) self-play training",
       py::arg("n_games") = 5000,
       py::arg("alpha") = 0.1f,
       py::arg("hidden_sizes"),
       py::arg("eps") = 0.1f,
       py::arg("seed") = 42,
       py::arg("benchmark_interval") = 1000,
       py::arg("model_name") = "td_s7",
       py::arg("models_dir") = "models",
       py::arg("resume_paths") = std::vector<std::string>(),
       py::arg("benchmarks") = py::list(),
       py::arg("canonical_map") = std::vector<int>());
}

} // namespace bgbot::python
