// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/training.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/encoding.h"
#include "bgbot/multipy.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace bgbot {

namespace {

// RAII guard that resets stream formatting to defaultfloat on entry and
// restores the caller's original flags/precision/fill on exit.
class ScopedStreamFormat {
    std::ostream& os_;
    std::ios::fmtflags flags_;
    std::streamsize precision_;
    char fill_;
public:
    explicit ScopedStreamFormat(std::ostream& os)
        : os_(os)
        , flags_(os.flags())
        , precision_(os.precision())
        , fill_(os.fill())
    {
        os_ << std::defaultfloat;
    }
    ~ScopedStreamFormat() {
        os_.flags(flags_);
        os_.precision(precision_);
        os_.fill(fill_);
    }
    ScopedStreamFormat(const ScopedStreamFormat&) = delete;
    ScopedStreamFormat& operator=(const ScopedStreamFormat&) = delete;
};

struct OutcomeStats {
    int games = 0;
    int singles = 0;
    int gammons = 0;
    int backgammons = 0;

    void record(GameResult result) {
        ++games;
        switch (result) {
            case GameResult::WIN_SINGLE:
            case GameResult::LOSS_SINGLE:
                ++singles;
                break;
            case GameResult::WIN_GAMMON:
            case GameResult::LOSS_GAMMON:
                ++gammons;
                break;
            case GameResult::WIN_BACKGAMMON:
            case GameResult::LOSS_BACKGAMMON:
                ++backgammons;
                break;
            default:
                break;
        }
    }

    void print_to(std::ostream& os) const {
        if (games <= 0) return;
        os << std::setprecision(1)
           << "  out: s=" << (100.0 * singles / games)
           << "% g=" << (100.0 * gammons / games)
           << "% b=" << (100.0 * backgammons / games) << "%";
    }

    void reset() {
        games = singles = gammons = backgammons = 0;
    }
};

void save_history_csv(const std::vector<TrainingHistoryEntry>& history,
                      const std::string& filepath)
{
    std::ofstream f(filepath);
    f << "game,contact_score,elapsed_seconds\n";
    for (const auto& e : history) {
        f << e.game_number << ","
          << e.contact_score << ","
          << e.elapsed_seconds << "\n";
    }
}

void load_weights_or_throw(NeuralNetwork& nn,
                           const std::string& path,
                           const std::string& label = "",
                           bool log = false)
{
    if (path.empty()) return;
    if (!nn.load_weights(path)) {
        if (label.empty()) {
            throw std::runtime_error("Failed to load weights: " + path);
        }
        throw std::runtime_error("Failed to load " + label + " weights: " + path);
    }
    if (log) {
        std::cout << "  Resumed " << label << " from: " << path << std::endl;
    }
}

void encode_into(const Board& board, bool extended, float* out) {
    if (extended) {
        auto inputs = compute_extended_contact_inputs(board);
        std::copy(inputs.begin(), inputs.end(), out);
    } else {
        auto inputs = compute_tesauro_inputs(board);
        std::copy(inputs.begin(), inputs.end(), out);
    }
}

// Standard backgammon opening: roll until non-double, flip board if d2 > d1.
// Returns true because (d1, d2) already hold the opening roll.
bool init_standard_game(std::mt19937& rng,
                        std::uniform_int_distribution<int>& die,
                        Board& board, int& d1, int& d2)
{
    board = STARTING_BOARD;
    do {
        d1 = die(rng);
        d2 = die(rng);
    } while (d1 == d2);
    if (d2 > d1) {
        board = flip(board);
    }
    return true;
}

// Unified TD(0) self-play training driver shared by all 6 trainers.
// Preserves the exact per-game RNG draw sequence and checkpoint cadence of
// each trainer via small policy callbacks.
struct PreStepRef {
    NeuralNetwork* nn = nullptr;
    int bucket = 0;
};

template <typename CheckpointFn,
          typename InitGameFn,
          typename PreparePreFn,
          typename PickMoveFn,
          typename OnUpdateFn,
          typename OnTerminalFn,
          typename MaybeTruncateFn,
          typename EvalPostFn>
TDTrainResult run_td_loop(
    int n_games,
    int benchmark_interval,
    float alpha,
    uint32_t seed,
    std::chrono::steady_clock::time_point t_start,
    std::vector<TrainingHistoryEntry>& history,
    CheckpointFn&& on_checkpoint,
    InitGameFn&& init_game,
    PreparePreFn&& prepare_pre,
    PickMoveFn&& pick_move,
    OnUpdateFn&& on_update,
    OnTerminalFn&& on_terminal,
    MaybeTruncateFn&& maybe_truncate,
    EvalPostFn&& eval_post)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> die(1, 6);

    std::vector<Board> candidates;
    candidates.reserve(32);
    std::array<float, EXTENDED_CONTACT_INPUTS> pre_buf;
    std::array<float, EXTENDED_CONTACT_INPUTS> post_buf;

    for (int game_idx = 0; game_idx < n_games; ++game_idx) {
        if (game_idx % benchmark_interval == 0) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();
            on_checkpoint(game_idx, elapsed, false);
        }

        Board board{};
        int d1 = 0, d2 = 0;
        bool has_opening_roll = init_game(game_idx, rng, die, board, d1, d2);
        bool first_move = true;
        int half_moves = 0;

        for (;;) {
            if (!first_move || !has_opening_roll) {
                d1 = die(rng);
                d2 = die(rng);
            }
            first_move = false;

            // Step A: pre-roll evaluation from opponent's perspective
            Board flipped = flip(board);
            PreStepRef pre = prepare_pre(flipped, board, pre_buf.data());
            pre.nn->forward_with_gradients(pre_buf.data());

            // Step B: generate moves and pick best
            possible_boards(board, d1, d2, candidates);
            if (candidates.size() == 1) {
                board = candidates[0];
            } else if (!candidates.empty()) {
                int idx = pick_move(game_idx, candidates, board);
                board = candidates[idx];
            }
            ++half_moves;

            // Step C: terminal check
            GameResult result = check_game_over(board);
            if (result != GameResult::NOT_OVER) {
                pre.nn->td_update(invert_probs(terminal_probs(result)), alpha);
                on_update(pre.bucket);
                on_terminal(result, half_moves);
                break;
            }

            // Optional truncation hook (used by truncated backgame trainer)
            if (maybe_truncate(*pre.nn, pre.bucket, board, half_moves, post_buf.data())) {
                break;
            }

            // Step D: non-terminal TD(0) bootstrap
            auto post_outputs = eval_post(board, post_buf.data());
            pre.nn->td_update(invert_probs(post_outputs), alpha);
            on_update(pre.bucket);

            // Step E: flip for next player
            board = flip(board);
        }
    }

    {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        on_checkpoint(n_games, elapsed, true);
    }

    TDTrainResult train_result;
    train_result.history = std::move(history);
    train_result.games_played = n_games;
    train_result.total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    return train_result;
}

double score_equity_benchmark(
    const NNStrategy& strat,
    const std::vector<EquityBenchmarkEntry>& bench)
{
    if (bench.empty()) return -1.0;
    double total_err = 0.0;
    for (const auto& e : bench) {
        auto probs = strat.evaluate_probs(e.board, false);
        double eq = NeuralNetwork::compute_equity(probs);
        total_err += std::abs(eq - static_cast<double>(e.target_equity));
    }
    return (total_err / bench.size()) * 1000.0;
}

} // namespace

// ======================== TD Training (single 196-input NN) ========================

TDTrainResult td_train(const TDTrainConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();
    ScopedStreamFormat fmt_guard(std::cout);

    auto nn = std::make_shared<NeuralNetwork>(
        config.n_hidden, NN_INPUTS, config.weight_init_eps, config.seed);
    load_weights_or_throw(*nn, config.resume_from);

    NNStrategy strat(nn);

    const std::string weights_path = config.models_dir + "/" + config.model_name + ".weights";
    const std::string history_path = config.models_dir + "/" + config.model_name + ".history.csv";

    std::vector<TrainingHistoryEntry> history;
    OutcomeStats outcomes;

    auto on_checkpoint = [&](int game_idx, double elapsed, bool is_final) {
        double bm_score = -1.0;
        if (config.benchmark_scenarios != nullptr && !config.benchmark_scenarios->empty()) {
            bm_score = score_benchmarks(strat, *config.benchmark_scenarios, 0).score();
        }
        history.push_back({game_idx, bm_score, elapsed});

        std::cout << "Game " << std::setw(6) << game_idx
                  << "  contact=" << std::fixed << std::setprecision(2) << bm_score;
        outcomes.print_to(std::cout);
        std::cout << "  time=" << std::setprecision(1) << elapsed << "s";
        if (is_final) std::cout << "  (final)";
        std::cout << std::endl;

        outcomes.reset();
        nn->save_weights(weights_path);
        save_history_csv(history, history_path);
    };

    return run_td_loop(
        config.n_games, config.benchmark_interval, config.alpha, config.seed,
        t_start, history,
        on_checkpoint,
        [](int, std::mt19937& rng, std::uniform_int_distribution<int>& die,
           Board& board, int& d1, int& d2) {
            return init_standard_game(rng, die, board, d1, d2);
        },
        [&](const Board& flipped, const Board&, float* buf) -> PreStepRef {
            encode_into(flipped, false, buf);
            return {nn.get(), 0};
        },
        [&](int, const std::vector<Board>& candidates, const Board& board) {
            return strat.best_move_index(candidates, board);
        },
        [](int) {},
        [&](GameResult result, int) { outcomes.record(result); },
        [](NeuralNetwork&, int, const Board&, int, float*) { return false; },
        [&](const Board& board, float* buf) {
            encode_into(board, false, buf);
            return nn->forward(buf);
        });
}

// ======================== Multi-Network TD Training (3-NN) ========================

TDTrainResult td_train_multi(const MultiTDTrainConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();
    ScopedStreamFormat fmt_guard(std::cout);
    init_escape_tables();

    std::cout << "=== Multi-Network TD Training ===" << std::endl;
    std::cout << "  Contact: " << config.n_hidden_contact << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Crashed: " << config.n_hidden_crashed << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Race:    " << config.n_hidden_race    << " hidden, " << TESAURO_INPUTS << " inputs" << std::endl;
    std::cout << "  Alpha:   " << config.alpha << std::endl;
    std::cout << "  Games:   " << config.n_games << std::endl;
    std::cout << "  Seed:    " << config.seed << std::endl;
    std::cout << std::endl;

    auto contact_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_contact, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed);
    auto crashed_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_crashed, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed + 3);
    auto race_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_race, TESAURO_INPUTS, config.weight_init_eps, config.seed + 7);

    load_weights_or_throw(*contact_nn, config.resume_contact, "contact", true);
    load_weights_or_throw(*crashed_nn, config.resume_crashed, "crashed", true);
    load_weights_or_throw(*race_nn, config.resume_race, "race", true);

    MultiNNStrategy strat(contact_nn, crashed_nn, race_nn);

    const std::string contact_path = config.models_dir + "/" + config.model_name + "_contact.weights";
    const std::string crashed_path = config.models_dir + "/" + config.model_name + "_crashed.weights";
    const std::string race_path    = config.models_dir + "/" + config.model_name + "_race.weights";
    const std::string history_path = config.models_dir + "/" + config.model_name + ".history.csv";

    auto select_nn = [&](PosType ptype) -> NeuralNetwork* {
        switch (ptype) {
            case PosType::RACE:    return race_nn.get();
            case PosType::CRASHED: return crashed_nn.get();
            case PosType::CONTACT:
            default:               return contact_nn.get();
        }
    };

    std::array<int, 3> update_counts = {};  // 0=contact, 1=crashed, 2=race
    auto ptype_bucket = [](PosType ptype) {
        switch (ptype) {
            case PosType::RACE:    return 2;
            case PosType::CRASHED: return 1;
            default:               return 0;
        }
    };

    std::vector<TrainingHistoryEntry> history;
    OutcomeStats outcomes;

    auto on_checkpoint = [&](int game_idx, double elapsed, bool is_final) {
        double bm_score = -1.0;
        if (config.contact_benchmark != nullptr && !config.contact_benchmark->empty()) {
            bm_score = score_benchmarks(strat, *config.contact_benchmark, 0).score();
        }
        history.push_back({game_idx, bm_score, elapsed});

        const int total_updates = update_counts[0] + update_counts[1] + update_counts[2];
        const double c_pct = total_updates > 0 ? 100.0 * update_counts[0] / total_updates : 0;
        const double x_pct = total_updates > 0 ? 100.0 * update_counts[1] / total_updates : 0;
        const double r_pct = total_updates > 0 ? 100.0 * update_counts[2] / total_updates : 0;

        std::cout << "Game " << std::setw(6) << game_idx
                  << "  contact=" << std::fixed << std::setprecision(2) << bm_score
                  << "  upd%: c=" << std::setprecision(0) << c_pct
                  << " x=" << x_pct
                  << " r=" << r_pct;
        outcomes.print_to(std::cout);
        std::cout << "  time=" << std::setprecision(1) << elapsed << "s";
        if (is_final) std::cout << "  (final)";
        std::cout << std::endl;

        outcomes.reset();
        contact_nn->save_weights(contact_path);
        crashed_nn->save_weights(crashed_path);
        race_nn->save_weights(race_path);
        save_history_csv(history, history_path);
    };

    return run_td_loop(
        config.n_games, config.benchmark_interval, config.alpha, config.seed,
        t_start, history,
        on_checkpoint,
        [](int, std::mt19937& rng, std::uniform_int_distribution<int>& die,
           Board& board, int& d1, int& d2) {
            return init_standard_game(rng, die, board, d1, d2);
        },
        [&](const Board& flipped, const Board&, float* buf) -> PreStepRef {
            PosType pre_ptype = classify_position(flipped);
            encode_into(flipped, pre_ptype != PosType::RACE, buf);
            return {select_nn(pre_ptype), ptype_bucket(pre_ptype)};
        },
        [&](int, const std::vector<Board>& candidates, const Board& board) {
            return strat.best_move_index(candidates, board);
        },
        [&](int bucket) { update_counts[bucket]++; },
        [&](GameResult result, int) { outcomes.record(result); },
        [](NeuralNetwork&, int, const Board&, int, float*) { return false; },
        [&](const Board& board, float* buf) {
            PosType post_ptype = classify_position(board);
            encode_into(board, post_ptype != PosType::RACE, buf);
            return select_nn(post_ptype)->forward(buf);
        });
}

// ======================== Game Plan TD Training (5-NN) ========================

TDTrainResult td_train_gameplan(const GamePlanTDTrainConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();
    ScopedStreamFormat fmt_guard(std::cout);
    init_escape_tables();

    std::cout << "=== Game Plan TD Training (5-NN) ===" << std::endl;
    std::cout << "  PureRace:  " << config.n_hidden_purerace  << " hidden, " << TESAURO_INPUTS << " inputs" << std::endl;
    std::cout << "  Racing:    " << config.n_hidden_racing     << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Attacking: " << config.n_hidden_attacking  << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Priming:   " << config.n_hidden_priming    << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Anchoring: " << config.n_hidden_anchoring  << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Alpha:     " << config.alpha << std::endl;
    std::cout << "  Games:     " << config.n_games << std::endl;
    std::cout << "  Seed:      " << config.seed << std::endl;
    std::cout << std::endl;

    auto purerace_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_purerace, TESAURO_INPUTS, config.weight_init_eps, config.seed);
    auto racing_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_racing, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed + 1);
    auto attacking_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_attacking, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed + 2);
    auto priming_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_priming, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed + 3);
    auto anchoring_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_anchoring, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed + 7);

    load_weights_or_throw(*purerace_nn, config.resume_purerace, "purerace", true);
    load_weights_or_throw(*racing_nn, config.resume_racing, "racing", true);
    load_weights_or_throw(*attacking_nn, config.resume_attacking, "attacking", true);
    load_weights_or_throw(*priming_nn, config.resume_priming, "priming", true);
    load_weights_or_throw(*anchoring_nn, config.resume_anchoring, "anchoring", true);

    GamePlanStrategy strat(purerace_nn, racing_nn, attacking_nn, priming_nn, anchoring_nn);

    const std::array<std::shared_ptr<NeuralNetwork>, 5> nns = {
        purerace_nn, racing_nn, attacking_nn, priming_nn, anchoring_nn
    };
    const std::array<const char*, 5> suffixes = {
        "_purerace.weights", "_racing.weights", "_attacking.weights",
        "_priming.weights", "_anchoring.weights"
    };
    std::array<std::string, 5> weight_paths, best_paths;
    for (int i = 0; i < 5; ++i) {
        weight_paths[i] = config.models_dir + "/" + config.model_name + suffixes[i];
        best_paths[i]   = weight_paths[i] + ".best";
    }
    const std::string history_path = config.models_dir + "/" + config.model_name + ".history.csv";

    auto gp_bucket = [](GamePlan gp) -> int {
        switch (gp) {
            case GamePlan::PURERACE:  return 0;
            case GamePlan::RACING:    return 1;
            case GamePlan::ATTACKING: return 2;
            case GamePlan::PRIMING:   return 3;
            case GamePlan::ANCHORING: return 4;
            default:                  return 2;
        }
    };

    double best_atk_score = 1e9;
    std::array<int, 5> update_counts = {};
    std::vector<TrainingHistoryEntry> history;
    OutcomeStats outcomes;

    auto on_checkpoint = [&](int game_idx, double elapsed, bool is_final) {
        auto score_opt = [&](const std::vector<BenchmarkScenario>* bm) {
            return (bm && !bm->empty()) ? score_benchmarks(strat, *bm, 0).score() : -1.0;
        };
        const double pr_score   = score_opt(config.purerace_benchmark);
        const double atk_score  = score_opt(config.attacking_benchmark);
        const double prm_score  = score_opt(config.priming_benchmark);
        const double anc_score  = score_opt(config.anchoring_benchmark);
        const double race_score = score_opt(config.race_benchmark);

        history.push_back({game_idx, atk_score, elapsed});

        int total_upd = 0;
        for (int c : update_counts) total_upd += c;
        auto pct = [&](int i) { return total_upd > 0 ? 100.0 * update_counts[i] / total_upd : 0.0; };

        std::cout << "Game " << std::setw(7) << game_idx
                  << std::fixed << std::setprecision(2)
                  << "  pr=" << pr_score
                  << "  atk=" << atk_score
                  << "  prm=" << prm_score
                  << "  anc=" << anc_score
                  << "  race=" << race_score
                  << std::setprecision(0)
                  << "  upd%: pr=" << pct(0) << " r=" << pct(1) << " a=" << pct(2)
                  << " p=" << pct(3) << " n=" << pct(4);
        outcomes.print_to(std::cout);
        std::cout << "  time=" << std::setprecision(1) << elapsed << "s";
        if (is_final) std::cout << "  (final)";
        std::cout << std::endl;

        outcomes.reset();
        for (int i = 0; i < 5; ++i) nns[i]->save_weights(weight_paths[i]);

        if (atk_score >= 0 && atk_score < best_atk_score && game_idx > 0) {
            best_atk_score = atk_score;
            for (int i = 0; i < 5; ++i) nns[i]->save_weights(best_paths[i]);
            std::cout << "  ** New best (atk=" << atk_score << ")" << std::endl;
        }
        save_history_csv(history, history_path);
    };

    return run_td_loop(
        config.n_games, config.benchmark_interval, config.alpha, config.seed,
        t_start, history,
        on_checkpoint,
        [](int, std::mt19937& rng, std::uniform_int_distribution<int>& die,
           Board& board, int& d1, int& d2) {
            return init_standard_game(rng, die, board, d1, d2);
        },
        [&](const Board& flipped, const Board&, float* buf) -> PreStepRef {
            const int bucket = gp_bucket(classify_game_plan(flipped));
            encode_into(flipped, bucket != 0, buf);
            return {nns[bucket].get(), bucket};
        },
        [&](int, const std::vector<Board>& candidates, const Board& board) {
            return strat.best_move_index(candidates, board);
        },
        [&](int bucket) { update_counts[bucket]++; },
        [&](GameResult result, int) { outcomes.record(result); },
        [](NeuralNetwork&, int, const Board&, int, float*) { return false; },
        [&](const Board& board, float* buf) {
            const int bucket = gp_bucket(classify_game_plan(board));
            encode_into(board, bucket != 0, buf);
            return nns[bucket]->forward(buf);
        });
}

// ======================== 17-NN Pair TD Training ========================

TDTrainResult td_train_gameplan_pair(const GamePlanPairTDTrainConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();
    ScopedStreamFormat fmt_guard(std::cout);
    init_escape_tables();

    const auto& pair_names = game_plan_pair_names();

    std::cout << "=== Game Plan Pair TD Training (17-NN) ===" << std::endl;
    for (int i = 0; i < NUM_PAIR_NNS; ++i) {
        int n_inp = (i == 0) ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS;
        std::cout << "  " << std::setw(12) << std::left << pair_names[i]
                  << ": " << config.hidden_sizes[i] << " hidden, " << n_inp << " inputs" << std::endl;
    }
    std::cout << std::right;
    std::cout << "  Alpha:     " << config.alpha << std::endl;
    std::cout << "  Games:     " << config.n_games << std::endl;
    std::cout << "  Seed:      " << config.seed << std::endl;
    std::cout << std::endl;

    bool has_sharing = false;
    for (int i = 0; i < NUM_PAIR_NNS; ++i) {
        if (config.canonical_map[i] != i) { has_sharing = true; break; }
    }
    if (has_sharing) {
        std::cout << "  NN sharing:" << std::endl;
        for (int i = 0; i < NUM_PAIR_NNS; ++i) {
            if (config.canonical_map[i] != i) {
                std::cout << "    " << pair_names[i] << " -> " << pair_names[config.canonical_map[i]] << std::endl;
            }
        }
        std::cout << std::endl;
    }

    std::array<std::shared_ptr<NeuralNetwork>, NUM_PAIR_NNS> nns;
    for (int i = 0; i < NUM_PAIR_NNS; ++i) {
        int canonical = config.canonical_map[i];
        if (canonical == i) {
            int n_inp = (i == 0) ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS;
            nns[i] = std::make_shared<NeuralNetwork>(
                config.hidden_sizes[i], n_inp, config.weight_init_eps, config.seed + i);
        } else {
            nns[i] = nns[canonical];
        }
    }

    for (int i = 0; i < NUM_PAIR_NNS; ++i) {
        if (config.canonical_map[i] != i) continue;
        load_weights_or_throw(*nns[i], config.resume_paths[i], pair_names[i], true);
    }

    GamePlanPairStrategy strat(nns);

    std::array<std::string, NUM_PAIR_NNS> weight_paths, best_paths;
    for (int i = 0; i < NUM_PAIR_NNS; ++i) {
        weight_paths[i] = config.models_dir + "/" + config.model_name + "_" + pair_names[i] + ".weights";
        best_paths[i]   = weight_paths[i] + ".best";
    }
    const std::string history_path = config.models_dir + "/" + config.model_name + ".history.csv";

    auto get_nn_idx = [&config](const Board& flipped_board, const Board& original_board) -> int {
        GamePlan player_gp = classify_game_plan(flipped_board);
        if (player_gp == GamePlan::PURERACE) return 0;
        GamePlan opponent_gp = classify_game_plan(original_board);
        if (opponent_gp == GamePlan::PURERACE) opponent_gp = GamePlan::RACING;
        int idx = 1 + game_plan_pair_index(player_gp, opponent_gp);
        return config.canonical_map[idx];
    };

    double best_contact_score = 1e9;
    std::array<int, NUM_PAIR_NNS> update_counts = {};
    std::vector<TrainingHistoryEntry> history;
    OutcomeStats outcomes;

    auto on_checkpoint = [&](int game_idx, double elapsed, bool is_final) {
        std::array<double, NUM_PAIR_NNS> scores;
        scores.fill(-1.0);
        for (int i = 0; i < NUM_PAIR_NNS; ++i) {
            if (config.canonical_map[i] != i) continue;
            if (config.benchmarks[i] && !config.benchmarks[i]->empty()) {
                scores[i] = score_benchmarks(strat, *config.benchmarks[i], 1).score();
            }
        }

        double contact_avg = 0.0;
        int n_contact = 0;
        for (int i = 1; i < NUM_PAIR_NNS; ++i) {
            if (scores[i] >= 0) { contact_avg += scores[i]; n_contact++; }
        }
        if (n_contact > 0) contact_avg /= n_contact;

        history.push_back({game_idx, contact_avg, elapsed});

        int total_upd = 0;
        for (int c : update_counts) total_upd += c;

        std::cout << "Game " << std::setw(7) << game_idx << std::fixed << std::setprecision(2);
        if (scores[0] >= 0) std::cout << "  pr=" << scores[0];
        static constexpr const char* plan_labels[] = {"race", "att", "prim", "anch"};
        for (int p = 0; p < 4; ++p) {
            double plan_avg = 0; int cnt = 0;
            for (int o = 0; o < 4; ++o) {
                int idx = 1 + p * 4 + o;
                if (scores[idx] >= 0) { plan_avg += scores[idx]; cnt++; }
            }
            if (cnt > 0) std::cout << "  " << plan_labels[p] << "=" << (plan_avg / cnt);
        }

        std::cout << std::setprecision(0) << "  upd%:";
        for (int i = 0; i < NUM_PAIR_NNS; ++i) {
            double pct = total_upd > 0 ? 100.0 * update_counts[i] / total_upd : 0;
            if (pct >= 0.5) std::cout << " " << pair_names[i][0] << pair_names[i][1] << "=" << pct;
        }
        outcomes.print_to(std::cout);
        std::cout << "  time=" << std::setprecision(1) << elapsed << "s";
        if (is_final) std::cout << "  (final)";
        std::cout << std::endl;

        outcomes.reset();
        for (int i = 0; i < NUM_PAIR_NNS; ++i) {
            if (config.canonical_map[i] == i) nns[i]->save_weights(weight_paths[i]);
        }

        if (contact_avg >= 0 && contact_avg < best_contact_score && game_idx > 0) {
            best_contact_score = contact_avg;
            for (int i = 0; i < NUM_PAIR_NNS; ++i) {
                if (config.canonical_map[i] == i) nns[i]->save_weights(best_paths[i]);
            }
            std::cout << "  ** New best (contact_avg=" << contact_avg << ")" << std::endl;
        }
        save_history_csv(history, history_path);
    };

    return run_td_loop(
        config.n_games, config.benchmark_interval, config.alpha, config.seed,
        t_start, history,
        on_checkpoint,
        [](int, std::mt19937& rng, std::uniform_int_distribution<int>& die,
           Board& board, int& d1, int& d2) {
            return init_standard_game(rng, die, board, d1, d2);
        },
        [&](const Board& flipped, const Board& original, float* buf) -> PreStepRef {
            const int pre_idx = get_nn_idx(flipped, original);
            encode_into(flipped, pre_idx != 0, buf);
            return {nns[pre_idx].get(), pre_idx};
        },
        [&](int, const std::vector<Board>& candidates, const Board& board) {
            return strat.best_move_index(candidates, board);
        },
        [&](int bucket) { update_counts[bucket]++; },
        [&](GameResult result, int) { outcomes.record(result); },
        [](NeuralNetwork&, int, const Board&, int, float*) { return false; },
        [&](const Board& board, float* buf) {
            const int post_idx = get_nn_idx(board, flip(board));
            encode_into(board, post_idx != 0, buf);
            return nns[post_idx]->forward(buf);
        });
}

// ======================== Paskogammon TD Training (single 244-input NN) ========

TDTrainResult td_train_pasko(const PaskoTDTrainConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();
    ScopedStreamFormat fmt_guard(std::cout);
    init_escape_tables();

    std::cout << "=== Paskogammon TD Training (single 244-input NN) ===" << std::endl;
    std::cout << "  Hidden:  " << config.n_hidden << ", " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Alpha:   " << config.alpha << std::endl;
    std::cout << "  Games:   " << config.n_games << std::endl;
    std::cout << "  Seed:    " << config.seed << std::endl;
    if (!config.resume_from.empty())
        std::cout << "  Resume:  " << config.resume_from << std::endl;
    std::cout << std::endl;

    auto nn = std::make_shared<NeuralNetwork>(
        config.n_hidden, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed);
    load_weights_or_throw(*nn, config.resume_from);

    NNStrategy strat(nn);

    const std::string weights_path = config.models_dir + "/" + config.model_name + ".weights";
    const std::string best_path    = weights_path + ".best";
    const std::string history_path = config.models_dir + "/" + config.model_name + ".history.csv";

    double best_bg_score = 1e9;
    std::vector<TrainingHistoryEntry> history;
    OutcomeStats outcomes;

    auto on_checkpoint = [&](int game_idx, double elapsed, bool is_final) {
        double bg_score = -1.0;
        if (config.benchmark != nullptr && !config.benchmark->empty()) {
            bg_score = score_equity_benchmark(strat, *config.benchmark);
        }
        history.push_back({game_idx, bg_score, elapsed});

        std::cout << "Game " << std::setw(7) << game_idx
                  << std::fixed << std::setprecision(2)
                  << "  bg=" << bg_score;
        outcomes.print_to(std::cout);
        std::cout << "  time=" << std::setprecision(1) << elapsed << "s";
        if (is_final) std::cout << "  (final)";
        std::cout << std::endl;

        outcomes.reset();
        nn->save_weights(weights_path);

        if (bg_score >= 0 && bg_score < best_bg_score && game_idx > 0) {
            best_bg_score = bg_score;
            nn->save_weights(best_path);
            std::cout << "  ** New best (bg=" << std::setprecision(2) << bg_score << ")" << std::endl;
        }
        save_history_csv(history, history_path);
    };

    return run_td_loop(
        config.n_games, config.benchmark_interval, config.alpha, config.seed,
        t_start, history,
        on_checkpoint,
        [&](int, std::mt19937&, std::uniform_int_distribution<int>&,
            Board& board, int&, int&) {
            board = config.start_board;
            return false;  // doubles allowed on first roll; drawn at top of loop
        },
        [&](const Board& flipped, const Board&, float* buf) -> PreStepRef {
            encode_into(flipped, true, buf);
            return {nn.get(), 0};
        },
        [&](int, const std::vector<Board>& candidates, const Board& board) {
            return strat.best_move_index(candidates, board);
        },
        [](int) {},
        [&](GameResult result, int) { outcomes.record(result); },
        [](NeuralNetwork&, int, const Board&, int, float*) { return false; },
        [&](const Board& board, float* buf) {
            encode_into(board, true, buf);
            return nn->forward(buf);
        });
}

// ======================== Stage 11 truncated backgame TD ========================

TDTrainResult td_train_backgame_truncated(const BackgameTDTrainConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();
    ScopedStreamFormat fmt_guard(std::cout);
    init_escape_tables();

    if (config.start_boards.empty())
        throw std::runtime_error("td_train_backgame_truncated: no start boards");
    if (config.ref_weight_paths.empty())
        throw std::runtime_error("td_train_backgame_truncated: no reference weights");

    std::cout << "=== Stage 11 truncated backgame TD (single 244-input NN) ===" << std::endl;
    std::cout << "  Hidden:  " << config.n_hidden << ", " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Alpha:   " << config.alpha << std::endl;
    std::cout << "  Games:   " << config.n_games << std::endl;
    std::cout << "  Seeds:   " << config.start_boards.size() << " start positions" << std::endl;
    std::cout << "  Target:  reference " << config.ref_plies << "-ply cubeless post-move eval on region exit" << std::endl;
    std::cout << "  Seed:    " << config.seed << std::endl;
    if (!config.resume_from.empty())
        std::cout << "  Resume:  " << config.resume_from << std::endl;
    std::cout << std::endl;

    auto nn = std::make_shared<NeuralNetwork>(
        config.n_hidden, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed);
    load_weights_or_throw(*nn, config.resume_from);

    NNStrategy strat(nn);

    auto ref_base = std::make_shared<BackgameAwarePairStrategy>(
        config.ref_weight_paths, config.ref_hidden_sizes);
    MultiPlyStrategy ref(ref_base, config.ref_plies, MoveFilters::TINY,
                         /*full_depth_opponent=*/false,
                         config.ref_parallel, config.ref_threads);

    std::uniform_int_distribution<int> coin(0, 1);

    const std::string weights_path = config.models_dir + "/" + config.model_name + ".weights";
    const std::string best_path    = weights_path + ".best";
    const std::string history_path = config.models_dir + "/" + config.model_name + ".history.csv";

    double best_bg_score = 1e9;
    std::vector<TrainingHistoryEntry> history;
    int interval_truncated = 0, interval_terminal = 0, interval_capped = 0;
    long long interval_half_moves = 0;
    int interval_games = 0;
    int exit_run = 0;

    auto on_checkpoint = [&](int game_idx, double elapsed, bool is_final) {
        double bg_score = -1.0;
        if (config.benchmark != nullptr && !config.benchmark->empty()) {
            bg_score = score_equity_benchmark(strat, *config.benchmark);
        }
        history.push_back({game_idx, bg_score, elapsed});

        std::cout << "Game " << std::setw(7) << game_idx
                  << std::fixed << std::setprecision(2)
                  << "  bg=" << bg_score;
        if (interval_games > 0) {
            std::cout << std::setprecision(1)
                      << "  trunc=" << (100.0 * interval_truncated / interval_games)
                      << "% term=" << (100.0 * interval_terminal / interval_games)
                      << "%";
            if (interval_capped > 0)
                std::cout << " capped=" << interval_capped;
            std::cout << "  len=" << std::setprecision(1)
                      << (static_cast<double>(interval_half_moves) / interval_games);
        }
        std::cout << "  time=" << std::setprecision(1) << elapsed << "s";
        if (is_final) std::cout << "  (final)";
        std::cout << std::endl;

        interval_truncated = interval_terminal = interval_capped = 0;
        interval_half_moves = 0;
        interval_games = 0;

        nn->save_weights(weights_path);
        if (bg_score >= 0 && bg_score < best_bg_score && game_idx > 0) {
            best_bg_score = bg_score;
            nn->save_weights(best_path);
            std::cout << "  ** New best (bg=" << std::setprecision(2) << bg_score << ")" << std::endl;
        }
        save_history_csv(history, history_path);
    };

    return run_td_loop(
        config.n_games, config.benchmark_interval, config.alpha, config.seed,
        t_start, history,
        on_checkpoint,
        [&](int game_idx, std::mt19937& rng, std::uniform_int_distribution<int>&,
            Board& board, int&, int&) {
            board = config.start_boards[game_idx % config.start_boards.size()];
            if (config.randomize_first_mover && coin(rng)) {
                board = flip(board);
            }
            exit_run = 0;
            return false;
        },
        [&](const Board& flipped, const Board&, float* buf) -> PreStepRef {
            encode_into(flipped, true, buf);
            return {nn.get(), 0};
        },
        [&](int game_idx, const std::vector<Board>& candidates, const Board& board) {
            return (game_idx < config.ref_move_games)
                ? ref_base->best_move_index(candidates, board)
                : strat.best_move_index(candidates, board);
        },
        [](int) {},
        [&](GameResult, int half_moves) {
            interval_games++;
            interval_terminal++;
            interval_half_moves += half_moves;
        },
        [&](NeuralNetwork& pre_nn, int, const Board& board, int half_moves, float* post_buf) {
            if (backgame_category(board) == BackgameCategory::NONE) {
                ++exit_run;
            } else {
                exit_run = 0;
            }
            const bool capped = half_moves >= config.max_half_moves;
            if (exit_run > config.boundary_extra_plies || capped) {
                auto ref_probs = ref.evaluate_probs(board, board);
                pre_nn.td_update(invert_probs(ref_probs), config.alpha);
                if (config.anchor_boundary) {
                    encode_into(board, true, post_buf);
                    pre_nn.forward_with_gradients(post_buf);
                    pre_nn.td_update(ref_probs, config.alpha);
                }
                interval_games++;
                interval_truncated++;
                if (capped) interval_capped++;
                interval_half_moves += half_moves;
                return true;
            }
            return false;
        },
        [&](const Board& board, float* buf) {
            encode_into(board, true, buf);
            return nn->forward(buf);
        });
}

} // namespace bgbot
