// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// Neural-network strategy wrappers and multi-network routing (NNStrategy,
// MultiNNStrategy, GamePlanStrategy, GamePlanPairStrategy, RoutedStrategy,
// and backgame / containment / snake / massive classifiers).
#include "bgbot/neural_net.h"
#include "bgbot/board.h"
#include "bgbot/encoding.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace bgbot {

static double terminal_equity(GameResult result) {
    switch (result) {
        case GameResult::WIN_SINGLE:      return  1.0;
        case GameResult::WIN_GAMMON:       return  2.0;
        case GameResult::WIN_BACKGAMMON:   return  3.0;
        case GameResult::LOSS_SINGLE:      return -1.0;
        case GameResult::LOSS_GAMMON:      return -2.0;
        case GameResult::LOSS_BACKGAMMON:  return -3.0;
        default: return 0.0;
    }
}

// ======================== NNStrategy ========================

NNStrategy::NNStrategy(std::shared_ptr<NeuralNetwork> nn)
    : nn_(std::move(nn))
{}

NNStrategy::NNStrategy(const std::string& weights_path, int n_hidden, int n_inputs)
    : nn_(std::make_shared<NeuralNetwork>(n_hidden, n_inputs))
{
    if (!nn_->load_weights(weights_path)) {
        throw std::runtime_error("Failed to load weights from: " + weights_path);
    }
}

double NNStrategy::evaluate(const Board& board, bool /*pre_move_is_race*/) const {
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_equity(result);
    }

    if (nn_->n_inputs() == EXTENDED_CONTACT_INPUTS) {
        auto inputs = compute_extended_contact_inputs(board);
        auto outputs = nn_->forward(inputs.data());
        return NeuralNetwork::compute_equity(outputs);
    } else {
        auto inputs = compute_tesauro_inputs(board);
        auto outputs = nn_->forward(inputs);
        return NeuralNetwork::compute_equity(outputs);
    }
}

std::array<float, NN_OUTPUTS> NNStrategy::evaluate_probs(
    const Board& board, bool /*pre_move_is_race*/) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }

    if (nn_->n_inputs() == EXTENDED_CONTACT_INPUTS) {
        auto inputs = compute_extended_contact_inputs(board);
        return nn_->forward(inputs.data());
    } else {
        auto inputs = compute_tesauro_inputs(board);
        return nn_->forward(inputs);
    }
}

int NNStrategy::best_move_index(const std::vector<Board>& candidates,
                                bool pre_move_is_race) const {
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    const int ni = nn_->n_inputs();
    const bool use_extended = (ni == EXTENDED_CONTACT_INPUTS);

    constexpr int STACK_MAX = 32;

    // Flat input buffer: ni floats per position
    std::vector<float> inputs_flat(n * ni);
    std::array<float, NN_OUTPUTS> outputs_stack[STACK_MAX];
    int eval_indices_stack[STACK_MAX];

    std::array<float, NN_OUTPUTS>* outputs_buf;
    int* eval_idx_buf;
    std::vector<std::array<float, NN_OUTPUTS>> outputs_heap;
    std::vector<int> eval_indices_heap;

    if (n <= STACK_MAX) {
        outputs_buf = outputs_stack;
        eval_idx_buf = eval_indices_stack;
    } else {
        outputs_heap.resize(n);
        eval_indices_heap.resize(n);
        outputs_buf = outputs_heap.data();
        eval_idx_buf = eval_indices_heap.data();
    }

    double best_val = -1e30;
    int best_idx = 0;
    int n_to_eval = 0;

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        if (result != GameResult::NOT_OVER) {
            double val = terminal_equity(result);
            if (val > best_val) {
                best_val = val;
                best_idx = i;
            }
        } else {
            float* dest = inputs_flat.data() + n_to_eval * ni;
            if (use_extended) {
                auto ext = compute_extended_contact_inputs(candidates[i]);
                std::memcpy(dest, ext.data(), ni * sizeof(float));
            } else {
                auto tes = compute_tesauro_inputs(candidates[i]);
                std::memcpy(dest, tes.data(), ni * sizeof(float));
            }
            eval_idx_buf[n_to_eval] = i;
            n_to_eval++;
        }
    }

    if (n_to_eval > 0) {
        nn_->forward_batch(inputs_flat.data(), outputs_buf, n_to_eval);

        for (int j = 0; j < n_to_eval; ++j) {
            double val = NeuralNetwork::compute_equity(outputs_buf[j]);
            if (val > best_val) {
                best_val = val;
                best_idx = eval_idx_buf[j];
            }
        }
    }

    return best_idx;
}

// ======================== MultiNNStrategy ========================

MultiNNStrategy::MultiNNStrategy(std::shared_ptr<NeuralNetwork> contact_nn,
                                 std::shared_ptr<NeuralNetwork> crashed_nn,
                                 std::shared_ptr<NeuralNetwork> race_nn)
    : contact_nn_(std::move(contact_nn))
    , crashed_nn_(std::move(crashed_nn))
    , race_nn_(std::move(race_nn))
{}

MultiNNStrategy::MultiNNStrategy(const std::string& contact_weights,
                                 const std::string& race_weights,
                                 int n_hidden)
{
    contact_nn_ = std::make_shared<NeuralNetwork>(n_hidden, EXTENDED_CONTACT_INPUTS);
    if (!contact_nn_->load_weights(contact_weights)) {
        throw std::runtime_error("Failed to load contact weights from: " + contact_weights);
    }
    crashed_nn_ = nullptr;  // use contact_nn for crashed positions
    race_nn_ = std::make_shared<NeuralNetwork>(n_hidden, TESAURO_INPUTS);
    if (!race_nn_->load_weights(race_weights)) {
        throw std::runtime_error("Failed to load race weights from: " + race_weights);
    }
}

MultiNNStrategy::MultiNNStrategy(const std::string& contact_weights,
                                 const std::string& crashed_weights,
                                 const std::string& race_weights,
                                 int n_hidden)
{
    contact_nn_ = std::make_shared<NeuralNetwork>(n_hidden, EXTENDED_CONTACT_INPUTS);
    if (!contact_nn_->load_weights(contact_weights)) {
        throw std::runtime_error("Failed to load contact weights from: " + contact_weights);
    }
    crashed_nn_ = std::make_shared<NeuralNetwork>(n_hidden, EXTENDED_CONTACT_INPUTS);
    if (!crashed_nn_->load_weights(crashed_weights)) {
        throw std::runtime_error("Failed to load crashed weights from: " + crashed_weights);
    }
    race_nn_ = std::make_shared<NeuralNetwork>(n_hidden, TESAURO_INPUTS);
    if (!race_nn_->load_weights(race_weights)) {
        throw std::runtime_error("Failed to load race weights from: " + race_weights);
    }
}

MultiNNStrategy::MultiNNStrategy(const std::string& contact_weights,
                                 const std::string& crashed_weights,
                                 const std::string& race_weights,
                                 int n_hidden_contact,
                                 int n_hidden_crashed,
                                 int n_hidden_race)
{
    contact_nn_ = std::make_shared<NeuralNetwork>(n_hidden_contact, EXTENDED_CONTACT_INPUTS);
    if (!contact_nn_->load_weights(contact_weights)) {
        throw std::runtime_error("Failed to load contact weights from: " + contact_weights);
    }
    crashed_nn_ = std::make_shared<NeuralNetwork>(n_hidden_crashed, EXTENDED_CONTACT_INPUTS);
    if (!crashed_nn_->load_weights(crashed_weights)) {
        throw std::runtime_error("Failed to load crashed weights from: " + crashed_weights);
    }
    race_nn_ = std::make_shared<NeuralNetwork>(n_hidden_race, TESAURO_INPUTS);
    if (!race_nn_->load_weights(race_weights)) {
        throw std::runtime_error("Failed to load race weights from: " + race_weights);
    }
}

const NeuralNetwork& MultiNNStrategy::select_nn(PosType ptype) const {
    switch (ptype) {
        case PosType::RACE:
            return *race_nn_;
        case PosType::CRASHED:
            return crashed_nn_ ? *crashed_nn_ : *contact_nn_;
        case PosType::CONTACT:
        default:
            return *contact_nn_;
    }
}

double MultiNNStrategy::evaluate_with_nn(const Board& board, PosType ptype) const {
    const auto& nn = select_nn(ptype);

    if (ptype == PosType::RACE) {
        auto inputs = compute_tesauro_inputs(board);
        auto outputs = nn.forward(inputs);
        return NeuralNetwork::compute_equity(outputs);
    } else {
        auto inputs = compute_extended_contact_inputs(board);
        auto outputs = nn.forward(inputs.data());
        return NeuralNetwork::compute_equity(outputs);
    }
}

std::array<float, NN_OUTPUTS> MultiNNStrategy::probs_with_nn(
    const Board& board, PosType ptype) const
{
    const auto& nn = select_nn(ptype);
    if (ptype == PosType::RACE) {
        auto inputs = compute_tesauro_inputs(board);
        return nn.forward(inputs);
    } else {
        auto inputs = compute_extended_contact_inputs(board);
        return nn.forward(inputs.data());
    }
}

double MultiNNStrategy::evaluate(const Board& board, bool pre_move_is_race) const {
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_equity(result);
    }

    PosType ptype = classify_position(board);
    return evaluate_with_nn(board, ptype);
}

std::array<float, NN_OUTPUTS> MultiNNStrategy::evaluate_probs(
    const Board& board, bool /*pre_move_is_race*/) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }
    PosType ptype = classify_position(board);
    return probs_with_nn(board, ptype);
}

int MultiNNStrategy::best_move_index(const std::vector<Board>& candidates,
                                     bool pre_move_is_race) const {
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    double best_val = -1e30;
    int best_idx = 0;

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val;
        if (result != GameResult::NOT_OVER) {
            val = terminal_equity(result);
        } else {
            PosType ptype = classify_position(candidates[i]);
            val = evaluate_with_nn(candidates[i], ptype);
        }
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    return best_idx;
}

// ======================== GamePlanStrategy ========================

GamePlanStrategy::GamePlanStrategy(std::shared_ptr<NeuralNetwork> purerace_nn,
                                   std::shared_ptr<NeuralNetwork> racing_nn,
                                   std::shared_ptr<NeuralNetwork> attacking_nn,
                                   std::shared_ptr<NeuralNetwork> priming_nn,
                                   std::shared_ptr<NeuralNetwork> anchoring_nn)
    : purerace_nn_(std::move(purerace_nn))
    , racing_nn_(std::move(racing_nn))
    , attacking_nn_(std::move(attacking_nn))
    , priming_nn_(std::move(priming_nn))
    , anchoring_nn_(std::move(anchoring_nn))
{}

GamePlanStrategy::GamePlanStrategy(const std::string& purerace_weights,
                                   const std::string& racing_weights,
                                   const std::string& attacking_weights,
                                   const std::string& priming_weights,
                                   const std::string& anchoring_weights,
                                   int n_hidden_purerace,
                                   int n_hidden_racing,
                                   int n_hidden_attacking,
                                   int n_hidden_priming,
                                   int n_hidden_anchoring)
{
    purerace_nn_ = std::make_shared<NeuralNetwork>(n_hidden_purerace, TESAURO_INPUTS);
    if (!purerace_nn_->load_weights(purerace_weights))
        throw std::runtime_error("Failed to load purerace weights: " + purerace_weights);

    racing_nn_ = std::make_shared<NeuralNetwork>(n_hidden_racing, EXTENDED_CONTACT_INPUTS);
    if (!racing_nn_->load_weights(racing_weights))
        throw std::runtime_error("Failed to load racing weights: " + racing_weights);

    attacking_nn_ = std::make_shared<NeuralNetwork>(n_hidden_attacking, EXTENDED_CONTACT_INPUTS);
    if (!attacking_nn_->load_weights(attacking_weights))
        throw std::runtime_error("Failed to load attacking weights: " + attacking_weights);

    priming_nn_ = std::make_shared<NeuralNetwork>(n_hidden_priming, EXTENDED_CONTACT_INPUTS);
    if (!priming_nn_->load_weights(priming_weights))
        throw std::runtime_error("Failed to load priming weights: " + priming_weights);

    anchoring_nn_ = std::make_shared<NeuralNetwork>(n_hidden_anchoring, EXTENDED_CONTACT_INPUTS);
    if (!anchoring_nn_->load_weights(anchoring_weights))
        throw std::runtime_error("Failed to load anchoring weights: " + anchoring_weights);

    // Prebuild transposed weights to avoid races in concurrent delta evaluation
    // (forward_from_base builds this lazily and is not synchronized).
    purerace_nn_->ensure_transposed_weights();
    racing_nn_->ensure_transposed_weights();
    attacking_nn_->ensure_transposed_weights();
    priming_nn_->ensure_transposed_weights();
    anchoring_nn_->ensure_transposed_weights();
}

GamePlanStrategy::GamePlanStrategy(const std::vector<std::string>& weight_paths,
                                   const std::vector<int>& hidden_sizes)
    : GamePlanStrategy(weight_paths.at(0), weight_paths.at(1), weight_paths.at(2),
                       weight_paths.at(3), weight_paths.at(4),
                       hidden_sizes.at(0), hidden_sizes.at(1), hidden_sizes.at(2),
                       hidden_sizes.at(3), hidden_sizes.at(4))
{
    if (weight_paths.size() != 5 || hidden_sizes.size() != 5)
        throw std::invalid_argument(
            "GamePlanStrategy vector constructor requires exactly 5 weight paths and 5 hidden sizes");
}

const NeuralNetwork& GamePlanStrategy::select_nn(GamePlan gp) const {
    switch (gp) {
        case GamePlan::PURERACE:  return *purerace_nn_;
        case GamePlan::RACING:    return *racing_nn_;
        case GamePlan::ATTACKING: return *attacking_nn_;
        case GamePlan::PRIMING:   return *priming_nn_;
        case GamePlan::ANCHORING: return *anchoring_nn_;
        default:                  return *attacking_nn_;
    }
}

double GamePlanStrategy::evaluate_with_nn(const Board& board, GamePlan gp) const {
    const auto& nn = select_nn(gp);
    if (gp == GamePlan::PURERACE) {
        auto inputs = compute_tesauro_inputs(board);
        auto outputs = nn.forward(inputs);
        return NeuralNetwork::compute_equity(outputs);
    } else {
        // RACING, ATTACKING, PRIMING, ANCHORING all use extended inputs
        auto inputs = compute_extended_contact_inputs(board);
        auto outputs = nn.forward(inputs.data());
        return NeuralNetwork::compute_equity(outputs);
    }
}

std::array<float, NN_OUTPUTS> GamePlanStrategy::probs_with_nn(
    const Board& board, GamePlan gp) const
{
    const auto& nn = select_nn(gp);
    if (gp == GamePlan::PURERACE) {
        auto inputs = compute_tesauro_inputs(board);
        return nn.forward(inputs);
    } else {
        auto inputs = compute_extended_contact_inputs(board);
        return nn.forward(inputs.data());
    }
}

double GamePlanStrategy::evaluate(const Board& board, bool /*pre_move_is_race*/) const {
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_equity(result);
    }
    GamePlan gp = classify_game_plan(board);
    return evaluate_with_nn(board, gp);
}

std::array<float, NN_OUTPUTS> GamePlanStrategy::evaluate_probs(
    const Board& board, bool /*pre_move_is_race*/) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }
    GamePlan gp = classify_game_plan(board);
    return probs_with_nn(board, gp);
}

std::array<float, NN_OUTPUTS> GamePlanStrategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }
    GamePlan gp = classify_game_plan(pre_move_board);
    return probs_with_nn(board, gp);
}

int GamePlanStrategy::best_move_index(const std::vector<Board>& candidates,
                                      bool pre_move_is_race) const {
    // Fallback: use PURERACE for race, ATTACKING for non-race
    GamePlan gp = pre_move_is_race ? GamePlan::PURERACE : GamePlan::ATTACKING;
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    double best_val = -1e30;
    int best_idx = 0;

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val;
        if (result != GameResult::NOT_OVER) {
            val = terminal_equity(result);
        } else {
            val = evaluate_with_nn(candidates[i], gp);
        }
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    return best_idx;
}

int GamePlanStrategy::best_move_index(const std::vector<Board>& candidates,
                                      const Board& pre_move_board) const {
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    // Classify the PRE-MOVE board to determine which NN to use for all candidates
    GamePlan gp = classify_game_plan(pre_move_board);

    double best_val = -1e30;
    int best_idx = 0;

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val;
        if (result != GameResult::NOT_OVER) {
            val = terminal_equity(result);
        } else {
            val = evaluate_with_nn(candidates[i], gp);
        }
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    return best_idx;
}

int GamePlanStrategy::evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    const int n = static_cast<int>(candidates.size());
    if (n == 0) return -1;
    if (n == 1) {
        if (equities) {
            GameResult result = check_game_over(candidates[0]);
            if (result != GameResult::NOT_OVER) {
                equities[0] = terminal_equity(result);
            } else {
                GamePlan gp = classify_game_plan(pre_move_board);
                equities[0] = evaluate_with_nn(candidates[0], gp);
            }
        }
        return 0;
    }

    // Classify once for all candidates
    GamePlan gp = classify_game_plan(pre_move_board);

    double best_val = -1e30;
    int best_idx = 0;

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val;
        if (result != GameResult::NOT_OVER) {
            val = terminal_equity(result);
        } else {
            val = evaluate_with_nn(candidates[i], gp);
        }
        if (equities) equities[i] = val;
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    return best_idx;
}


namespace {

template<typename SelectModel>
int score_game_plan_candidates(
    const std::vector<Board>& candidates, double* equities,
    std::array<float, NUM_OUTPUTS>* per_move_probs,
    std::array<float, NUM_OUTPUTS>* best_probs, SelectModel select_model)
{
    const int count = static_cast<int>(candidates.size());
    if (count == 0) return -1;
    if (count == 1 && !equities && !per_move_probs && !best_probs) return 0;

    const auto [network, pure_race, evaluate_one] = select_model();
    double best_value = -1e30;
    int best_index = 0;
    std::array<float, NUM_OUTPUTS> best_result{};
    const auto record = [&](int index, double value, const auto& probs) {
        if (equities) equities[index] = value;
        if (per_move_probs) per_move_probs[index] = probs;
        if (value > best_value) {
            best_value = value;
            best_index = index;
            if (best_probs) best_result = probs;
        }
    };

    if (count == 1 || !network) {
        // A blended route has position-dependent weights and cannot reuse a
        // neural base. Compare its candidates in generation order.
        for (int i = 0; i < count; ++i) {
            const auto result = check_game_over(candidates[i]);
            if (result != GameResult::NOT_OVER) {
                record(i, terminal_equity(result), terminal_probs(result));
            } else {
                const auto probs = evaluate_one(candidates[i]);
                record(i, NeuralNetwork::compute_equity(probs), probs);
            }
        }
    } else {
        thread_local std::vector<int> indices;
        thread_local std::vector<std::array<float, NUM_OUTPUTS>> probabilities;
        indices.clear();
        indices.reserve(count);
        // Terminal moves win ties against evaluated moves. Within each group
        // the first equal value wins, as in the original batch methods.
        for (int i = 0; i < count; ++i) {
            const auto result = check_game_over(candidates[i]);
            if (result != GameResult::NOT_OVER) {
                record(i, terminal_equity(result), terminal_probs(result));
            } else {
                indices.push_back(i);
            }
        }
        const int group_size = static_cast<int>(indices.size());
        if (group_size > 0) {
            probabilities.resize(group_size);
            evaluate_board_group(*network, pure_race, candidates,
                                 indices.data(), group_size, probabilities.data());
            for (int j = 0; j < group_size; ++j) {
                const auto& probs = probabilities[j];
                record(indices[j], NeuralNetwork::compute_equity(probs), probs);
            }
        }
    }
    if (best_probs) *best_probs = best_result;
    return best_index;
}

} // namespace

int GamePlanStrategy::batch_evaluate_candidates_equity(
    const std::vector<Board>& candidates, const Board& pre_move_board, double* equities) const
{
    return score_candidate_batch(candidates, pre_move_board, equities, nullptr, nullptr);
}

int GamePlanStrategy::batch_evaluate_candidates_equity_probs(
    const std::vector<Board>& candidates, const Board& pre_move_board,
    double* equities, std::array<float, NUM_OUTPUTS>* probs_out) const
{
    return score_candidate_batch(candidates, pre_move_board, equities, probs_out, nullptr);
}

int GamePlanStrategy::batch_evaluate_candidates_best_prob(
    const std::vector<Board>& candidates, const Board& pre_move_board,
    double* equities, std::array<float, NUM_OUTPUTS>* best_probs_out) const
{
    return score_candidate_batch(candidates, pre_move_board, equities, nullptr, best_probs_out);
}

int GamePlanStrategy::score_candidate_batch(
    const std::vector<Board>& candidates, const Board& pre_move_board, double* equities,
    std::array<float, NUM_OUTPUTS>* per_move_probs,
    std::array<float, NUM_OUTPUTS>* best_probs) const
{
    return score_game_plan_candidates(candidates, equities, per_move_probs, best_probs, [&] {
        const auto plan = classify_game_plan(pre_move_board);
        return std::tuple{&select_nn(plan), plan == GamePlan::PURERACE,
                          [this, plan](const Board& board) { return probs_with_nn(board, plan); }};
    });
}

// ===========================================================================
// GamePlanPairStrategy implementation
// ===========================================================================

GamePlanPairStrategy::GamePlanPairStrategy(
    const std::vector<std::string>& weight_paths,
    const std::vector<int>& hidden_sizes)
{
    if (weight_paths.size() != NUM_PAIR_NNS || hidden_sizes.size() != NUM_PAIR_NNS) {
        throw std::runtime_error(
            "GamePlanPairStrategy requires exactly " + std::to_string(NUM_PAIR_NNS) +
            " weight paths and hidden sizes, got " +
            std::to_string(weight_paths.size()) + "/" +
            std::to_string(hidden_sizes.size()));
    }
    for (int i = 0; i < NUM_PAIR_NNS; ++i) {
        int n_inputs = (i == 0) ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS;
        nns_[i] = std::make_shared<NeuralNetwork>(hidden_sizes[i], n_inputs);
        if (!nns_[i]->load_weights(weight_paths[i]))
            throw std::runtime_error(
                "Failed to load weights for " + std::string(game_plan_pair_name(i)) +
                ": " + weight_paths[i]);
        nns_[i]->ensure_transposed_weights();
    }
}

GamePlanPairStrategy::GamePlanPairStrategy(
    std::array<std::shared_ptr<NeuralNetwork>, NUM_PAIR_NNS> nns)
    : nns_(std::move(nns))
{
    // Prebuild transposed weights for thread-safe concurrent evaluation
    // (forward_from_base builds this lazily and is not synchronized).
    for (int i = 0; i < NUM_PAIR_NNS; ++i) {
        if (!nns_[i]) continue;
        const int width = i == 0 ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS;
        if (nns_[i]->n_inputs() > width)
            throw std::invalid_argument("Pair network input width exceeds its slot encoding");
        nns_[i]->ensure_transposed_weights();
    }
}

int GamePlanPairStrategy::select_nn_idx(const Board& board) const {
    if (is_race(board)) return 0;
    const auto [player_pips, opp_pips] = pip_counts(board);
    const GamePlan player_gp = classify_contact_game_plan(board, player_pips, opp_pips);
    const GamePlan opponent_gp = classify_contact_game_plan(flip(board), opp_pips, player_pips);
    return 1 + game_plan_pair_index(player_gp, opponent_gp);
}

double GamePlanPairStrategy::evaluate_with_nn(const Board& board, int nn_idx) const {
    const auto& nn = *nns_[nn_idx];
    if (nn_idx == 0) {
        auto inputs = compute_tesauro_inputs(board);
        auto outputs = nn.forward(inputs);
        return NeuralNetwork::compute_equity(outputs);
    } else {
        auto inputs = compute_extended_contact_inputs(board);
        auto outputs = nn.forward(inputs.data());
        return NeuralNetwork::compute_equity(outputs);
    }
}

std::array<float, NN_OUTPUTS> GamePlanPairStrategy::probs_with_nn(
    const Board& board, int nn_idx) const
{
    const auto& nn = *nns_[nn_idx];
    if (nn_idx == 0) {
        auto inputs = compute_tesauro_inputs(board);
        return nn.forward(inputs);
    } else {
        auto inputs = compute_extended_contact_inputs(board);
        return nn.forward(inputs.data());
    }
}

double GamePlanPairStrategy::evaluate(const Board& board, bool /*pre_move_is_race*/) const {
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) return terminal_equity(result);
    return evaluate_with_nn(board, select_nn_idx(board));
}

std::array<float, NN_OUTPUTS> GamePlanPairStrategy::evaluate_probs(
    const Board& board, bool /*pre_move_is_race*/) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) return terminal_probs(result);
    return probs_with_nn(board, select_nn_idx(board));
}

std::array<float, NN_OUTPUTS> GamePlanPairStrategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) return terminal_probs(result);
    return probs_with_nn(board, select_nn_idx(pre_move_board));
}

int GamePlanPairStrategy::best_move_index(const std::vector<Board>& candidates,
                                           bool pre_move_is_race) const {
    int nn_idx = pre_move_is_race ? 0 : 6;  // fallback: purerace or att_att
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    double best_val = -1e30;
    int best_idx = 0;
    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val = (result != GameResult::NOT_OVER)
            ? terminal_equity(result) : evaluate_with_nn(candidates[i], nn_idx);
        if (val > best_val) { best_val = val; best_idx = i; }
    }
    return best_idx;
}

int GamePlanPairStrategy::best_move_index(const std::vector<Board>& candidates,
                                           const Board& pre_move_board) const {
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;
    int nn_idx = select_nn_idx(pre_move_board);

    double best_val = -1e30;
    int best_idx = 0;
    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val = (result != GameResult::NOT_OVER)
            ? terminal_equity(result) : evaluate_with_nn(candidates[i], nn_idx);
        if (val > best_val) { best_val = val; best_idx = i; }
    }
    return best_idx;
}

int GamePlanPairStrategy::evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    const int n = static_cast<int>(candidates.size());
    if (n == 0) return -1;
    if (n == 1 && !equities) return 0;
    int nn_idx = select_nn_idx(pre_move_board);

    double best_val = -1e30;
    int best_idx = 0;
    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val = (result != GameResult::NOT_OVER)
            ? terminal_equity(result) : evaluate_with_nn(candidates[i], nn_idx);
        if (equities) equities[i] = val;
        if (val > best_val) { best_val = val; best_idx = i; }
    }
    return best_idx;
}

int GamePlanPairStrategy::batch_evaluate_candidates_equity(
    const std::vector<Board>& candidates, const Board& pre_move_board, double* equities) const
{
    return score_candidate_batch(candidates, pre_move_board, equities, nullptr, nullptr);
}

int GamePlanPairStrategy::batch_evaluate_candidates_equity_probs(
    const std::vector<Board>& candidates, const Board& pre_move_board,
    double* equities, std::array<float, NUM_OUTPUTS>* probs_out) const
{
    return score_candidate_batch(candidates, pre_move_board, equities, probs_out, nullptr);
}

int GamePlanPairStrategy::batch_evaluate_candidates_best_prob(
    const std::vector<Board>& candidates, const Board& pre_move_board,
    double* equities, std::array<float, NUM_OUTPUTS>* best_probs_out) const
{
    return score_candidate_batch(candidates, pre_move_board, equities, nullptr, best_probs_out);
}

int GamePlanPairStrategy::score_candidate_batch(
    const std::vector<Board>& candidates, const Board& pre_move_board, double* equities,
    std::array<float, NUM_OUTPUTS>* per_move_probs,
    std::array<float, NUM_OUTPUTS>* best_probs) const
{
    return score_game_plan_candidates(candidates, equities, per_move_probs, best_probs, [&] {
        const int index = select_nn_idx(pre_move_board);
        const auto* network = nns_[index].get();
        return std::tuple{network, index == 0,
                          [this, index](const Board& board) { return probs_with_nn(board, index); }};
    });
}

// ===========================================================================
// BackgameAwarePairStrategy implementation
// ===========================================================================

namespace {
// Precision gate for the extra Paskogammon backgame NNs: deep (3+ anchor)
// backgames, or massive 2-anchor ones (big back mass, high pips). Everything
// outside the gate evaluates bit-identically to the 19-NN model.
inline bool backgame_gate(int anchors, int back_checkers, int bg_pips) {
    return anchors >= BACKGAME_GATE_MIN_ANCHORS ||
           (anchors == 2 && back_checkers >= BACKGAME_GATE_ARM2_BC &&
            bg_pips >= BACKGAME_GATE_ARM2_PIPS);
}

// Extra-NN weight inside the gate: zone base by anchor depth, scaled by a pip
// ramp (half weight at/below PIP_LO, full at/above PIP_HI).
inline float backgame_blend_weight(int anchors, int bg_pips) {
    const float base = (anchors >= 4)   ? BACKGAME_BLEND_W_A4
                       : (anchors == 3) ? BACKGAME_BLEND_W_A3
                                        : BACKGAME_BLEND_W_ARM2;
    float ramp;
    if (bg_pips <= BACKGAME_BLEND_PIP_LO) {
        ramp = 0.0f;
    } else if (bg_pips >= BACKGAME_BLEND_PIP_HI) {
        ramp = 1.0f;
    } else {
        ramp = static_cast<float>(bg_pips - BACKGAME_BLEND_PIP_LO) /
               static_cast<float>(BACKGAME_BLEND_PIP_HI - BACKGAME_BLEND_PIP_LO);
    }
    return base * (0.5f + 0.5f * ramp);
}

// Category from the backgame side's anchor depths (depth d = an anchor on the
// opponent's d-point, d in 1..6; bit d of `mask` set when held). Exactly two
// anchors map by their pair — {1,2},{1,3},{2,3} deep; {3,4},{3,5},{4,5} (and
// any pair no deeper than the 3-point) double-anchor; the rest middle. Three
// or more anchors are DEEP when at least two sit on the 1/2/3 points and
// MIDDLE otherwise — never DOUBLE_ANCHOR.
inline BackgameCategory category_from_anchor_mask(int mask) {
    int n = 0, deep = 0, min_depth = 7;
    for (int d = 1; d <= 6; ++d) {
        if (mask & (1 << d)) {
            ++n;
            if (d <= 3) ++deep;
            if (d < min_depth) min_depth = d;
        }
    }
    if (n < 2) return BackgameCategory::NONE;
    if (deep >= 2) return BackgameCategory::DEEP;
    if (n == 2 && min_depth >= 3) return BackgameCategory::DOUBLE_ANCHOR;
    return BackgameCategory::MIDDLE;
}

// Stage 9's backgame detection (see select_nn_idx below — the two MUST agree),
// returning the backgame side's anchor-depth mask, or NONE. `player_gp` /
// `opponent_gp` are the precomputed plans, with the opponent's PURERACE
// already coerced to RACING as select_nn_idx does.
BackgameCategory backgame_category_given_plans(
    const Board& board, GamePlan player_gp, GamePlan opponent_gp,
    int player_pips, int opp_pips)
{
    if (player_gp == GamePlan::ANCHORING && opponent_gp == GamePlan::RACING) {
        if (player_pips > opp_pips) {
            int mask = 0;
            for (int pt = 19; pt <= 24; ++pt)
                if (board[pt] >= 2) mask |= 1 << (25 - pt);
            return category_from_anchor_mask(mask);
        }
    }
    if (player_gp == GamePlan::RACING && opponent_gp == GamePlan::ANCHORING) {
        if (opp_pips > player_pips) {
            int mask = 0;
            for (int pt = 1; pt <= 6; ++pt)
                if (board[pt] <= -2) mask |= 1 << pt;
            return category_from_anchor_mask(mask);
        }
    }
    return BackgameCategory::NONE;
}
}  // namespace

BackgameCategory backgame_category(const Board& board) {
    if (is_race(board)) return BackgameCategory::NONE;
    const auto [p1, p2] = pip_counts(board);
    const auto player_gp = classify_contact_game_plan(board, p1, p2);
    const auto opponent_gp = classify_contact_game_plan(flip(board), p2, p1);
    return backgame_category_given_plans(board, player_gp, opponent_gp, p1, p2);
}

namespace {
// Player 2 (negative, moving upward) is the escaper of a containment game.
bool p2_is_escaper(const Board& b) {
    int on_board = b[0];
    for (int i = 1; i <= 24; ++i) if (b[i] < 0) on_board -= b[i];
    if (15 - on_board < CONTAINMENT_E_OFF_MIN) return false;
    int p1_max = -1;                       // highest P1 index; the bar (25) counts
    for (int i = 1; i <= 25; ++i) if (b[i] > 0) p1_max = i;
    int stragglers = (p1_max > 0) ? b[0] : 0;
    for (int i = 1; i <= 18; ++i)
        if (b[i] < 0 && p1_max > i) stragglers -= b[i];
    return stragglers >= 1 && stragglers <= CONTAINMENT_STRAGGLERS_MAX;
}

// Player 1 (positive) holds a snake: a far-side prime with a P2 straggler
// behind it while P2's other checkers are crunched home.
bool p1_holds_snake(const Board& b) {
    int run = 0, best = 0;
    for (int i = 13; i <= 24; ++i) {
        run = (b[i] >= 2) ? run + 1 : 0;
        if (run > best) best = run;
    }
    if (best < SNAKE_PRIME_MIN_POINTS) return false;
    int straggler = b[0];                  // P2 bar, or P2 checkers in P1's home
    for (int i = 1; i <= 6; ++i) if (b[i] < 0) straggler -= b[i];
    if (straggler < 1) return false;
    int home = 0;
    for (int i = 19; i <= 24; ++i) if (b[i] < 0) home -= b[i];
    return home >= SNAKE_MIN_HOME;
}
}  // namespace

bool containment_category(const Board& board) {
    return p2_is_escaper(board) || p2_is_escaper(flip(board));
}

bool snake_category(const Board& board) {
    return p1_holds_snake(board) || p1_holds_snake(flip(board));
}

namespace {
// Player 1 plays a massive back game (scripts/backgame_benchmark.py
// _massive_p1_holder): behind in pips with >= MASSIVE_ALT_ANCHORS anchors in
// the opponent's home board, or >= MASSIVE_MIN_ANCHORS anchors and
// >= MASSIVE_BACK_MIN checkers back (opponent's home board + bar).
bool p1_has_massive_structure(const Board& b) {
    int anchors = 0, back = b[25];
    for (int i = 19; i <= 24; ++i) {
        if (b[i] >= 2) ++anchors;
        if (b[i] > 0) back += b[i];
    }
    return anchors >= MASSIVE_MIN_ANCHORS && (anchors >= MASSIVE_ALT_ANCHORS || back >= MASSIVE_BACK_MIN);
}
}  // namespace

bool massive_category(const Board& board) {
    // Disjoint from the two more specific regions: a late containment game
    // and a snake both read as "many checkers back" but are their own games.
    if (containment_category(board) || snake_category(board)) return false;
    const auto [p1, p2] = pip_counts(board);
    return (p1 > p2 && p1_has_massive_structure(board)) ||
           (p2 > p1 && p1_has_massive_structure(flip(board)));
}

static BackgamePhase backgame_phase(const Board& board, int p1_pips, int p2_pips) {
    int a1 = 0, a2 = 0;
    for (int pt = 19; pt <= 24; ++pt) if (board[pt] >= 2) ++a1;
    for (int pt = 1; pt <= 6; ++pt) if (board[pt] <= -2) ++a2;

    // Which side is the holder B: two or more anchors and behind on pips.
    // A single-anchor fallback was tried and rejected (2026-09-02): it routed
    // 3,927 ordinary money decisions — hit a blot while holding one anchor —
    // to the containment NN and cost +0.75 PR on the money benchmark, while
    // the two-anchor subset carried the whole folder gain at no money cost.
    bool holder_p1;
    if (a1 >= 2 && p1_pips > p2_pips) holder_p1 = true;
    else if (a2 >= 2 && p2_pips > p1_pips) holder_p1 = false;
    else return BackgamePhase::NONE;

    int r_bar, r_in_b_home, r_on_board, r_home;
    if (holder_p1) {
        r_bar = board[0];
        r_in_b_home = r_on_board = r_home = 0;
        for (int pt = 1; pt <= 24; ++pt) {
            if (board[pt] >= 0) continue;
            r_on_board -= board[pt];
            if (pt <= 6) r_in_b_home -= board[pt];
            if (pt >= 19) r_home -= board[pt];
        }
        r_on_board += board[0];
    } else {
        r_bar = board[25];
        r_in_b_home = r_on_board = r_home = 0;
        for (int pt = 1; pt <= 24; ++pt) {
            if (board[pt] <= 0) continue;
            r_on_board += board[pt];
            if (pt >= 19) r_in_b_home += board[pt];
            if (pt <= 6) r_home += board[pt];
        }
        r_on_board += board[25];
    }
    const int r_off = 15 - r_on_board;
    const int stragglers = r_bar + r_in_b_home;
    if (stragglers >= 1)
        return r_off <= 2 ? BackgamePhase::EARLY_CONTAINMENT
                          : BackgamePhase::LATE_CONTAINMENT;
    if (r_home >= 10 || r_off >= 1) return BackgamePhase::BEAR_IN;
    return BackgamePhase::WAITING;
}

BackgamePhase backgame_phase(const Board& board) {
    const auto [p1, p2] = pip_counts(board);
    return backgame_phase(board, p1, p2);
}


namespace {

RoutingManifest legacy_routing(size_t weights, size_t hiddens, bool phase_containment) {
    const int n = static_cast<int>(weights);
    if (phase_containment) {
        if ((n != NUM_BACKGAME_PAIR_NNS_PHASED &&
             n != NUM_BACKGAME_PAIR_NNS_PHASED_SNAKE &&
             n != NUM_BACKGAME_PAIR_NNS_PHASED_MASSIVE) ||
            hiddens != weights) {
            throw std::runtime_error(
                "Phased BackgameAwarePairStrategy requires " +
                std::to_string(NUM_BACKGAME_PAIR_NNS_PHASED) + ", " +
                std::to_string(NUM_BACKGAME_PAIR_NNS_PHASED_SNAKE) + " or " +
                std::to_string(NUM_BACKGAME_PAIR_NNS_PHASED_MASSIVE) +
                " weight paths and matching hidden sizes, got " +
                std::to_string(weights) + "/" +
                std::to_string(hiddens));
        }
    } else if ((n != NUM_BACKGAME_PAIR_NNS && n != NUM_BACKGAME_PAIR_NNS_HYBRID &&
                n != NUM_BACKGAME_PAIR_NNS_CATEGORIZED) ||
               hiddens != weights) {
        throw std::runtime_error(
            "BackgameAwarePairStrategy requires " +
            std::to_string(NUM_BACKGAME_PAIR_NNS) + ", " +
            std::to_string(NUM_BACKGAME_PAIR_NNS_CATEGORIZED) + " or " +
            std::to_string(NUM_BACKGAME_PAIR_NNS_HYBRID) +
            " weight paths and matching hidden sizes, got " +
            std::to_string(weights) + "/" +
            std::to_string(hiddens));
    }

    RoutingManifest routing;
    if (phase_containment || n == NUM_BACKGAME_PAIR_NNS_CATEGORIZED) {
        routing.backgame_categories = {17, 18, 19};
    } else {
        routing.side_backgame = {17, 18};
        if (n == NUM_BACKGAME_PAIR_NNS_HYBRID) routing.blend_extra = {19, 20};
    }
    if (phase_containment) {
        routing.early_containment = 20;
        routing.containment = 21;
        if (n >= NUM_BACKGAME_PAIR_NNS_PHASED_SNAKE) {
            routing.snake = 22;
            routing.root_pins = {22};
        }
        if (n == NUM_BACKGAME_PAIR_NNS_PHASED_MASSIVE) routing.massive = 23;
    }
    return routing;
}

} // namespace

RoutedStrategy::RoutedStrategy(const std::vector<std::string>& weight_paths,
                               const std::vector<int>& hidden_sizes, bool phase_containment)
    : RoutedStrategy(weight_paths, hidden_sizes,
                     legacy_routing(weight_paths.size(), hidden_sizes.size(), phase_containment))
{}

RoutedStrategy::RoutedStrategy(const std::vector<std::string>& weight_paths,
                               const std::vector<int>& hidden_sizes, RoutingManifest routing)
    : routing_(std::move(routing))
{
    if (weight_paths.empty() || weight_paths.size() >= BLENDED_SENTINEL_BASE ||
        weight_paths.size() != hidden_sizes.size()) {
        throw std::invalid_argument("Routing requires equally sized, nonempty lists of weights and hidden sizes");
    }
    const int n = static_cast<int>(weight_paths.size());
    const auto validate_contact = [n](int slot) {
        if (slot <= 0 || slot >= n) {
            throw std::invalid_argument("Contact routes require an existing non-race network slot");
        }
    };
    const auto validate_optional = [&](int slot) {
        if (slot != -1) validate_contact(slot);
    };
    for (int slot : routing_.contact_pairs) validate_contact(slot);
    for (int slot : routing_.root_pins) validate_contact(slot);
    for (int slot : routing_.backgame_categories) validate_optional(slot);
    for (int slot : routing_.side_backgame) validate_optional(slot);
    for (int slot : {routing_.snake, routing_.containment, routing_.massive,
                     routing_.early_containment}) validate_optional(slot);
    for (int side = 0; side < 2; ++side) {
        const int extra = routing_.blend_extra[side];
        validate_optional(extra);
        if (extra >= 0 && routing_.side_backgame[side] < 0) {
            throw std::invalid_argument("A backgame blend requires a base slot for the same side");
        }
    }
    categorized_backgame_ = std::any_of(routing_.backgame_categories.begin(),
                                        routing_.backgame_categories.end(),
                                        [](int slot) { return slot >= 0; });
    nns_.resize(n);
    for (int i = 0; i < n; ++i) {
        if (hidden_sizes[i] <= 0 || hidden_sizes[i] > std::numeric_limits<int>::max() / (EXTENDED_CONTACT_INPUTS + 1)) {
            throw std::invalid_argument("Invalid network hidden size");
        }
        int n_inputs = (i == 0) ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS;
        nns_[i] = std::make_shared<NeuralNetwork>(hidden_sizes[i], n_inputs);
        if (!nns_[i]->load_weights(weight_paths[i]))
            throw std::runtime_error(
                "Failed to load weights for index " + std::to_string(i) +
                ": " + weight_paths[i]);
        nns_[i]->ensure_transposed_weights();
    }
}

std::array<float, NUM_OUTPUTS> RoutedStrategy::evaluate_pinned_probs(
    const Board& board, bool race, int pinned) const
{
    if (pinned >= 0 && !race) return probs_with_nn(board, pinned);
    return evaluate_probs(board, race);
}

RolloutVRNetwork RoutedStrategy::rollout_vr_network(const Board& board) const {
    const int index = nn_index_for(board);
    if (index >= BLENDED_SENTINEL_BASE) return {};
    return {nns_[index].get(), index == 0};
}

int RoutedStrategy::root_pin_for(const Board& root) const {
    if (routing_.root_pins.empty()) return -1;
    const int idx = select_nn_idx(root);
    for (int p : routing_.root_pins) if (p == idx) return idx;
    return -1;
}

int RoutedStrategy::select_nn_idx(const Board& board) const {
    if (is_race(board)) return 0;
    const Board opponent = flip(board);
    if (routing_.snake >= 0 && (p1_holds_snake(board) || p1_holds_snake(opponent)))
        return routing_.snake;
    if (routing_.containment >= 0 && (p2_is_escaper(board) || p2_is_escaper(opponent)))
        return routing_.containment;
    const auto [player_pips, opp_pips] = pip_counts(board);
    if (routing_.massive >= 0 &&
        ((player_pips > opp_pips && p1_has_massive_structure(board)) ||
         (opp_pips > player_pips && p1_has_massive_structure(opponent))))
        return routing_.massive;
    const GamePlan player_gp = classify_contact_game_plan(board, player_pips, opp_pips);
    const GamePlan opponent_gp = classify_contact_game_plan(opponent, opp_pips, player_pips);

    if (categorized_backgame_) {
        const auto category =
            backgame_category_given_plans(board, player_gp, opponent_gp, player_pips, opp_pips);
        if (category != BackgameCategory::NONE) {
            const int slot = routing_.backgame_categories[static_cast<int>(category)];
            if (slot >= 0) return slot;
        }
    }
    if (routing_.early_containment >= 0 &&
        backgame_phase(board, player_pips, opp_pips) == BackgamePhase::EARLY_CONTAINMENT)
        return routing_.early_containment;

    // Player back game: (anchoring, racing) + player behind + 2+ anchors in opp home
    if (routing_.side_backgame[0] >= 0 &&
        player_gp == GamePlan::ANCHORING && opponent_gp == GamePlan::RACING) {
        if (player_pips > opp_pips) {
            int anchors = 0, back = board[25];
            for (int pt = 19; pt <= 24; ++pt) {
                if (board[pt] >= 2) ++anchors;
                if (board[pt] > 0) back += board[pt];
            }
            if (anchors >= 2) {
                if (routing_.blend_extra[0] >= 0 &&
                    backgame_gate(anchors, back, player_pips))
                    return BLENDED_PLAYER_BG_IDX;
                return routing_.side_backgame[0];
            }
        }
    }

    // Opponent back game: (racing, anchoring) + opp behind + 2+ opp anchors in player home
    if (routing_.side_backgame[1] >= 0 &&
        player_gp == GamePlan::RACING && opponent_gp == GamePlan::ANCHORING) {
        if (opp_pips > player_pips) {
            int opp_anchors = 0, opp_back = board[0];
            for (int pt = 1; pt <= 6; ++pt) {
                if (board[pt] <= -2) ++opp_anchors;
                if (board[pt] < 0) opp_back -= board[pt];
            }
            if (opp_anchors >= 2) {
                if (routing_.blend_extra[1] >= 0 &&
                    backgame_gate(opp_anchors, opp_back, opp_pips))
                    return BLENDED_OPPONENT_BG_IDX;
                return routing_.side_backgame[1];
            }
        }
    }

    return routing_.contact_pairs[game_plan_pair_index(player_gp, opponent_gp)];
}

std::array<float, NN_OUTPUTS> RoutedStrategy::blended_backgame_probs(
    const Board& board, bool player_bg) const
{
    auto inputs = compute_extended_contact_inputs(board);
    const int side = player_bg ? 0 : 1;
    auto base = nns_[routing_.side_backgame[side]]->forward(inputs.data());
    auto extra = nns_[routing_.blend_extra[side]]->forward(inputs.data());
    auto [player_pips, opp_pips] = pip_counts(board);
    int anchors = 0;
    if (player_bg) {
        for (int pt = 19; pt <= 24; ++pt)
            if (board[pt] >= 2) ++anchors;
    } else {
        for (int pt = 1; pt <= 6; ++pt)
            if (board[pt] <= -2) ++anchors;
    }
    const float w = backgame_blend_weight(anchors,
                                          player_bg ? player_pips : opp_pips);
    std::array<float, NN_OUTPUTS> out;
    for (int i = 0; i < NN_OUTPUTS; ++i)
        out[i] = (1.0f - w) * base[i] + w * extra[i];
    return out;
}

double RoutedStrategy::evaluate_with_nn(const Board& board, int nn_idx) const {
    if (nn_idx >= BLENDED_SENTINEL_BASE)
        return NeuralNetwork::compute_equity(
            blended_backgame_probs(board, nn_idx == BLENDED_PLAYER_BG_IDX));
    const auto& nn = *nns_[nn_idx];
    if (nn_idx == 0) {
        auto inputs = compute_tesauro_inputs(board);
        auto outputs = nn.forward(inputs);
        return NeuralNetwork::compute_equity(outputs);
    } else {
        auto inputs = compute_extended_contact_inputs(board);
        auto outputs = nn.forward(inputs.data());
        return NeuralNetwork::compute_equity(outputs);
    }
}

std::array<float, NN_OUTPUTS> RoutedStrategy::probs_with_nn(
    const Board& board, int nn_idx) const
{
    if (nn_idx >= BLENDED_SENTINEL_BASE)
        return blended_backgame_probs(board, nn_idx == BLENDED_PLAYER_BG_IDX);
    const auto& nn = *nns_[nn_idx];
    if (nn_idx == 0) {
        auto inputs = compute_tesauro_inputs(board);
        return nn.forward(inputs);
    } else {
        auto inputs = compute_extended_contact_inputs(board);
        return nn.forward(inputs.data());
    }
}

double RoutedStrategy::evaluate(const Board& board, bool /*pre_move_is_race*/) const {
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) return terminal_equity(result);
    return evaluate_with_nn(board, select_nn_idx(board));
}

std::array<float, NN_OUTPUTS> RoutedStrategy::evaluate_probs(
    const Board& board, bool /*pre_move_is_race*/) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) return terminal_probs(result);
    return probs_with_nn(board, select_nn_idx(board));
}

std::array<float, NN_OUTPUTS> RoutedStrategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) return terminal_probs(result);
    return probs_with_nn(board, select_nn_idx(pre_move_board));
}

int RoutedStrategy::best_move_index(const std::vector<Board>& candidates,
                                                bool pre_move_is_race) const {
    int nn_idx = pre_move_is_race
        ? 0
        : routing_.contact_pairs[game_plan_pair_index(GamePlan::ATTACKING, GamePlan::ATTACKING)];
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    double best_val = -1e30;
    int best_idx = 0;
    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val = (result != GameResult::NOT_OVER)
            ? terminal_equity(result) : evaluate_with_nn(candidates[i], nn_idx);
        if (val > best_val) { best_val = val; best_idx = i; }
    }
    return best_idx;
}

int RoutedStrategy::best_move_index(const std::vector<Board>& candidates,
                                                const Board& pre_move_board) const {
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;
    int nn_idx = select_nn_idx(pre_move_board);

    double best_val = -1e30;
    int best_idx = 0;
    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val = (result != GameResult::NOT_OVER)
            ? terminal_equity(result) : evaluate_with_nn(candidates[i], nn_idx);
        if (val > best_val) { best_val = val; best_idx = i; }
    }
    return best_idx;
}

int RoutedStrategy::evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    const int n = static_cast<int>(candidates.size());
    if (n == 0) return -1;
    if (n == 1 && !equities) return 0;
    int nn_idx = select_nn_idx(pre_move_board);

    double best_val = -1e30;
    int best_idx = 0;
    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val = (result != GameResult::NOT_OVER)
            ? terminal_equity(result) : evaluate_with_nn(candidates[i], nn_idx);
        if (equities) equities[i] = val;
        if (val > best_val) { best_val = val; best_idx = i; }
    }
    return best_idx;
}

int RoutedStrategy::batch_evaluate_candidates_equity(
    const std::vector<Board>& candidates, const Board& pre_move_board, double* equities) const
{
    return score_candidate_batch(candidates, pre_move_board, equities, nullptr, nullptr);
}

int RoutedStrategy::batch_evaluate_candidates_equity_probs(
    const std::vector<Board>& candidates, const Board& pre_move_board,
    double* equities, std::array<float, NUM_OUTPUTS>* probs_out) const
{
    return score_candidate_batch(candidates, pre_move_board, equities, probs_out, nullptr);
}

int RoutedStrategy::batch_evaluate_candidates_best_prob(
    const std::vector<Board>& candidates, const Board& pre_move_board,
    double* equities, std::array<float, NUM_OUTPUTS>* best_probs_out) const
{
    return score_candidate_batch(candidates, pre_move_board, equities, nullptr, best_probs_out);
}

int RoutedStrategy::score_candidate_batch(
    const std::vector<Board>& candidates, const Board& pre_move_board, double* equities,
    std::array<float, NUM_OUTPUTS>* per_move_probs,
    std::array<float, NUM_OUTPUTS>* best_probs) const
{
    return score_game_plan_candidates(candidates, equities, per_move_probs, best_probs, [&] {
        const int index = select_nn_idx(pre_move_board);
        const auto* network = index >= BLENDED_SENTINEL_BASE ? nullptr : nns_[index].get();
        return std::tuple{network, index == 0,
                          [this, index](const Board& board) { return probs_with_nn(board, index); }};
    });
}


} // namespace bgbot
