// SPDX-License-Identifier: MPL-2.0
// Fast native unit tests for low-level C++ invariants (designed to run under
// ASan/UBSan in < 2s).
#include "bgbot/analyzer_session.h"
#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/cube.h"
#include "bgbot/encoding.h"
#include "bgbot/moves.h"
#include "bgbot/multipy.h"
#include "bgbot/neural_net.h"
#include "bgbot/pubeval.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace bgbot;

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error("FAILED: " + msg);
}

class ConstantStrategy : public Strategy {
public:
    explicit ConstantStrategy(std::array<float, NUM_OUTPUTS> probs) : probs_(probs) {}

    double evaluate(const Board&, bool) const override {
        return NeuralNetwork::compute_equity(probs_);
    }

    std::array<float, NUM_OUTPUTS> evaluate_probs(const Board&, bool) const override {
        return probs_;
    }

    std::array<float, NUM_OUTPUTS> evaluate_probs(const Board&, const Board&) const override {
        return probs_;
    }

    int best_move_index(const std::vector<Board>&, bool) const override { return 0; }
    int best_move_index(const std::vector<Board>&, const Board&) const override { return 0; }

private:
    std::array<float, NUM_OUTPUTS> probs_;
};

Board make_starting_board() {
    return {0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5,
            5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0};
}

void test_shared_pos_cache_abandonment_and_probing() {
    SharedPosCache cache;
    const Board b1 = make_starting_board();
    Board b2 = b1;
    b2[6] = 4;
    b2[5] = 1;

    // Craft colliding hashes that map to the exact same primary slot.
    const std::size_t h1 = 0x123400ULL;
    const std::size_t h2 = h1 + SharedPosCache::CAPACITY;  // same index mod CAPACITY, different key

    // 1. Reserve slot for b1 and abandon it via exception unwinding.
    struct Cancelled {};
    try {
        auto res = cache.lookup_or_reserve(h1, 2, b1);
        require(res.probs == nullptr && res.reservation != nullptr,
                "first reservation must succeed");
        SharedPosCache::ReservationGuard guard(&cache, res.reservation);
        // Insert colliding key b2 while b1's slot is occupied, placing b2 at slot+1.
        const std::array<float, NUM_OUTPUTS> p2{0.61f, 0.12f, 0.01f, 0.09f, 0.0f};
        cache.insert(h2, 2, b2, p2);
        throw Cancelled{};
    } catch (const Cancelled&) {
    }

    // 2. Colliding key b2 at slot+1 MUST still be found past the tombstone!
    const auto* got2 = cache.lookup(h2, 2, b2);
    require(got2 != nullptr && std::abs((*got2)[0] - 0.61f) < 1e-6f,
            "tombstone must preserve linear-probe chain for colliding keys");

    // 3. b1 itself is NOT ready, and can be cleanly re-reserved and published.
    require(cache.lookup(h1, 2, b1) == nullptr, "abandoned entry must not read as ready");
    {
        auto res = cache.lookup_or_reserve(h1, 2, b1);
        require(res.probs == nullptr && res.reservation != nullptr,
                "abandoned slot must be reclaimable by same key");
        SharedPosCache::ReservationGuard guard(&cache, res.reservation);
        const std::array<float, NUM_OUTPUTS> p1{0.55f, 0.14f, 0.01f, 0.11f, 0.0f};
        guard.publish(p1);
    }
    const auto* got1 = cache.lookup(h1, 2, b1);
    require(got1 != nullptr && std::abs((*got1)[0] - 0.55f) < 1e-6f,
            "reclaimed entry must publish cleanly");

    // 4. Concurrent waiter spinning on STATE_COMPUTING unblocks on abandonment.
    const std::size_t h3 = 0x567800ULL;
    auto res3 = cache.lookup_or_reserve(h3, 2, b1);
    require(res3.reservation != nullptr, "third reservation must succeed");
    std::atomic<bool> waiter_started{false};
    std::atomic<bool> waiter_done{false};
    std::thread waiter([&] {
        waiter_started.store(true, std::memory_order_release);
        auto r = cache.lookup_or_reserve(h3, 2, b1);
        require(r.probs == nullptr, "waiter should observe abandoned miss");
        if (r.reservation) {
            cache.abandon(r.reservation);
        }
        waiter_done.store(true, std::memory_order_release);
    });
    while (!waiter_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    cache.abandon(res3.reservation);
    waiter.join();
    require(waiter_done.load(std::memory_order_acquire),
            "concurrent waiter must unblock cleanly after abandonment");
}

void test_multipy_set_bearoff_db_detach_and_swap(const std::string& bearoff_path) {
    BearoffDB db_loaded;
    require(db_loaded.load(bearoff_path), "failed to load bearoff DB at " + bearoff_path);
    BearoffDB db_unloaded;

    const std::array<float, NUM_OUTPUTS> mock_probs{0.25f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto base = std::make_shared<ConstantStrategy>(mock_probs);
    auto filter = std::make_shared<ConstantStrategy>(mock_probs);

    // Bearoff post-move board (both sides inside home board, non-terminal).
    Board bearoff{};
    bearoff[1] = 2;
    bearoff[2] = 1;
    bearoff[24] = -2;
    bearoff[23] = -1;
    require(db_loaded.is_bearoff(bearoff), "test board must be in bearoff DB");
    const auto exact = db_loaded.lookup_probs(bearoff, /*post_move=*/true);
    require(std::abs(exact[0] - mock_probs[0]) > 0.1f,
            "DB probs must differ clearly from mock base probs");

    // Non-bearoff board whose leaves enter bearoff (tests leaf wrapping + cache salt).
    Board near_bearoff{};
    near_bearoff[1] = 2;
    near_bearoff[2] = 2;
    near_bearoff[24] = -2;
    near_bearoff[17] = -1;  // one opponent checker outside home board

    for (bool hybrid : {false, true}) {
        auto mp = hybrid
            ? std::make_shared<MultiPlyStrategy>(base, filter, 2, MoveFilters::TINY, false, false, 1)
            : std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY, false, false, 1);
        mp->clear_cache();

        const auto plain_bearoff = mp->evaluate_probs(bearoff, bearoff);
        const auto plain_near = mp->evaluate_probs(near_bearoff, near_bearoff);
        require(std::abs(plain_bearoff[0] - exact[0]) > 0.05f,
                "initial bearoff eval must use base strategy instead of bearoff DB");

        // Attach loaded DB -> exact bearoff probs.
        mp->set_bearoff_db(&db_loaded);
        const auto with_db_bearoff = mp->evaluate_probs(bearoff, bearoff);
        const auto with_db_near = mp->evaluate_probs(near_bearoff, near_bearoff);
        require(std::abs(with_db_bearoff[0] - exact[0]) < 1e-5f,
                "attached DB must evaluate exact bearoff probs");
        require(std::abs(with_db_near[0] - plain_near[0]) > 0.05f,
                "attached DB must affect leaves without explicit clear_cache()");

        // Detach DB (nullptr) -> must return to base without calling clear_cache()!
        mp->set_bearoff_db(nullptr);
        const auto detached_bearoff = mp->evaluate_probs(bearoff, bearoff);
        const auto detached_near = mp->evaluate_probs(near_bearoff, near_bearoff);
        require(std::abs(detached_bearoff[0] - plain_bearoff[0]) < 1e-5f,
                "detached DB must restore base strategy on bearoff board");
        require(std::abs(detached_near[0] - plain_near[0]) < 1e-5f,
                "detached DB must restore base strategy and refresh cache salt on leaves");

        // Attach db_loaded then swap to db_unloaded -> must NOT keep db_loaded!
        mp->set_bearoff_db(&db_loaded);
        require(std::abs(mp->evaluate_probs(bearoff, bearoff)[0] - exact[0]) < 1e-5f,
                "re-attached DB must evaluate exact bearoff probs");
        mp->set_bearoff_db(&db_unloaded);
        const auto swapped_near = mp->evaluate_probs(near_bearoff, near_bearoff);
        require(std::abs(swapped_near[0] - plain_near[0]) < 1e-5f,
                "swapping DB must re-wrap raw_base_ instead of keeping previous DB");
    }
}

void test_routed_strategy_small_manifest_and_pin_bounds() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / "neongammon_test_native_core";
    fs::create_directories(dir);
    const auto race_path = (dir / "race.weights").string();
    const auto contact_path = (dir / "contact.weights").string();

    NeuralNetwork race_nn(4, TESAURO_INPUTS, 0.1f, 11);
    NeuralNetwork contact_nn(4, EXTENDED_CONTACT_INPUTS, 0.1f, 22);
    require(race_nn.save_weights(race_path), "save race weights");
    require(contact_nn.save_weights(contact_path), "save contact weights");

    RoutingManifest routing;
    routing.contact_pairs.fill(1);
    RoutedStrategy routed({race_path, contact_path}, {4, 4}, routing);

    const Board start = make_starting_board();
    std::vector<Board> candidates;
    possible_boards(start, 3, 1, candidates);
    require(candidates.size() > 1, "starting board 3-1 must have multiple moves");

    // Previously OOB indexed nns_[6] on < 7-net manifests.
    const int contact_idx = routed.best_move_index(candidates, /*pre_move_is_race=*/false);
    const int race_idx = routed.best_move_index(candidates, /*pre_move_is_race=*/true);
    require(contact_idx >= 0 && contact_idx < static_cast<int>(candidates.size()),
            "contact best_move_index out of range");
    require(race_idx >= 0 && race_idx < static_cast<int>(candidates.size()),
            "race best_move_index out of range");

    // set_root_pinned must reject invalid slots (0 = race, < 0, >= n).
    for (int bad : {-1, 0, 2, 99}) {
        bool threw = false;
        try {
            routed.set_root_pinned({bad});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw, "set_root_pinned must reject invalid slot " + std::to_string(bad));
    }
    routed.set_root_pinned({1});
    require(routed.root_pinned().size() == 1 && routed.root_pinned()[0] == 1,
            "valid root pin must be accepted");
    routed.set_root_pinned({});
    require(routed.root_pinned().empty(), "empty root pins must be accepted");

    fs::remove_all(dir);
}

void test_neural_network_inputs_above_256() {
    constexpr int kInputs = 300;
    NeuralNetwork nn(8, kInputs, 0.1f, 7);
    std::vector<float> single(kInputs, 0.5f);
    const auto out = nn.forward(single.data());
    for (float v : out) {
        require(std::isfinite(v) && v >= 0.0f && v <= 1.0f, "forward output invalid");
    }

    constexpr int kBatch = 3;
    std::vector<float> batch(kBatch * kInputs, 0.25f);
    std::array<std::array<float, NN_OUTPUTS>, kBatch> outs{};
    nn.forward_batch(batch.data(), outs.data(), kBatch);
    for (int i = 0; i < kBatch; ++i) {
        for (float v : outs[i]) {
            require(std::isfinite(v) && v >= 0.0f && v <= 1.0f, "forward_batch output invalid");
        }
    }
}

void test_budget_forwarding_and_cache_raii() {
    const Board board = make_starting_board();
    std::vector<Board> endpoints;
    possible_boards(board, 3, 1, endpoints);
    require(endpoints.size() >= 4, "need multiple endpoints for budget test");

    auto pubeval = std::make_shared<PubEval>();
    CubeInfo opp_ci{1, CubeOwner::CENTERED, {0, 0, false}, -1.0f, true, true};

    // 1. cubeful_checker_endpoints_nply must honor check_budget mid-roll (inside endpoint 0).
    struct Expired {};
    {
        std::atomic<int> checks{0};
        bool threw = false;
        try {
            cubeful_checker_endpoints_nply(board, endpoints, opp_ci, *pubeval, 2, 1, [&] {
                if (checks.fetch_add(1, std::memory_order_relaxed) >= 2) throw Expired{};
            });
        } catch (const Expired&) {
            threw = true;
        }
        require(threw, "cubeful_checker_endpoints_nply must throw when budget expires");
        require(checks.load() <= 4,
                "check_budget must fire mid-roll rather than scanning every endpoint (saw " +
                std::to_string(checks.load()) + ")");
    }

    // 2. cubeful_probs_and_equity_nply must forward check_budget mid-roll.
    {
        int checks = 0;
        bool threw = false;
        try {
            cubeful_probs_and_equity_nply(flip(endpoints[0]), opp_ci, *pubeval, 2,
                                          MoveFilters::TINY, 1, nullptr, &board, [&] {
                if (++checks >= 3) throw Expired{};
            });
        } catch (const Expired&) {
            threw = true;
        }
        require(threw && checks == 3,
                "cubeful_probs_and_equity_nply must forward check_budget mid-roll");
    }

    // 3. evaluate_cube_tree must honor check_budget and clean cache in decision_only and details modes.
    CubeInfo money_ci{1, CubeOwner::CENTERED, {0, 0, false}, -1.0f, true, true};
    const auto ref_decision = evaluate_cube_tree(
        board, money_ci, pubeval, nullptr, 2, MoveFilters::TINY, 1, true, nullptr);
    TwoPlyDetails ref_details;
    const auto ref_with_details = evaluate_cube_tree(
        board, money_ci, pubeval, nullptr, 2, MoveFilters::TINY, 1, false, &ref_details);
    require(ref_details.nd_player_rolls.size() == 21 && ref_details.dt_player_rolls.size() == 21,
            "2-ply details must populate 21 ND and DT rolls");

    for (bool decision_only : {true, false}) {
        int checks = 0;
        TwoPlyDetails partial;
        bool threw = false;
        try {
            evaluate_cube_tree(board, money_ci, pubeval, nullptr, 2, MoveFilters::TINY, 1,
                               decision_only, decision_only ? nullptr : &partial, [&] {
                if (++checks == 4) throw Expired{};
            });
        } catch (const Expired&) {
            threw = true;
        }
        require(threw, "evaluate_cube_tree must abort when budget expires");
    }

    const auto after_decision = evaluate_cube_tree(
        board, money_ci, pubeval, nullptr, 2, MoveFilters::TINY, 1, true, nullptr);
    TwoPlyDetails after_details;
    const auto after_with_details = evaluate_cube_tree(
        board, money_ci, pubeval, nullptr, 2, MoveFilters::TINY, 1, false, &after_details);
    require(after_decision.decision.equity_nd == ref_decision.decision.equity_nd &&
            after_decision.decision.equity_dt == ref_decision.decision.equity_dt,
            "decision_only must match reference after budget abort");
    require(after_with_details.decision.equity_nd == ref_with_details.decision.equity_nd &&
            after_with_details.probs == ref_with_details.probs,
            "details path must match reference after budget abort");
}

void test_two_ply_details_weighted_average() {
    const Board board = make_starting_board();
    auto pubeval = std::make_shared<PubEval>();
    CubeInfo ci{1, CubeOwner::CENTERED, {0, 0, false}, -1.0f, true, true};

    for (int plies : {2, 3}) {
        TwoPlyDetails details;
        const auto cd = cube_decision_nply_with_details(
            board, ci, *pubeval, plies, MoveFilters::TINY, 1, details);
        require(details.nd_player_rolls.size() == 21 && details.dt_player_rolls.size() == 21,
                "expected 21 rolls in details");
        double nd_sum = 0.0, dt_sum = 0.0;
        for (int r = 0; r < 21; ++r) {
            const auto& nd = details.nd_player_rolls[r];
            const auto& dt = details.dt_player_rolls[r];
            const int w = (nd.die1 == nd.die2) ? 1 : 2;
            nd_sum += w * nd.cubeful_equity;
            dt_sum += w * dt.cubeful_equity;
            if (plies == 2) {
                require(nd.opponent_rolls.empty() && dt.opponent_rolls.empty(),
                        "2-ply details should not populate opponent_rolls");
            } else if (!nd.is_terminal && !nd.opponent_dp) {
                require(nd.opponent_rolls.size() == 21,
                        "3-ply non-terminal non-DP ND roll should have 21 opponent_rolls");
            }
        }
        require(std::abs(nd_sum / 36.0 - cd.equity_nd) < 1e-4,
                "ND weighted average must match equity_nd");
        const double expected_dt = cd.is_beaver ? 2.0 * (dt_sum / 36.0) : (dt_sum / 36.0);
        require(std::abs(expected_dt - cd.equity_dt) < 1e-4,
                "DT weighted average must match equity_dt");
    }
}

void test_cube_nply_multi_bounds() {
    const Board board = make_starting_board();
    auto pubeval = std::make_shared<PubEval>();

    std::vector<CubeInfo> too_many_eq(65);
    std::vector<float> eq_out(65, 0.0f);
    bool threw_eq = false;
    try {
        cubeful_equity_nply_multi(board, too_many_eq.data(), 65, *pubeval, 1, eq_out.data());
    } catch (const std::invalid_argument&) {
        threw_eq = true;
    }
    require(threw_eq, "cubeful_equity_nply_multi must throw when n_cubes > EVAL_MAX_CCI");

    std::vector<CubeInfo> too_many_cd(33);
    std::vector<CubeDecision> cd_out(33);
    bool threw_cd = false;
    try {
        cube_decision_nply_multi(board, too_many_cd.data(), 33, *pubeval, 1, cd_out.data());
    } catch (const std::invalid_argument&) {
        threw_cd = true;
    }
    require(threw_cd, "cube_decision_nply_multi must throw when n_cubes > EVAL_MAX_CCI / 2");
}

void test_dead_cube_efficiency_skip_predicates_and_eval() {
    // Live money & live match use efficiency.
    CubeInfo money_live{1, CubeOwner::CENTERED, {0, 0, false}, -1.0f, true, true};
    require(cube_uses_efficiency(money_live), "live money cube uses efficiency");

    CubeInfo match_live{1, CubeOwner::CENTERED, {5, 5, false}};
    require(cube_uses_efficiency(match_live), "5a-5a centered cube uses efficiency");

    // Structural cap, Crawford, DMP, match-dead, and post-Crawford auto-double skip efficiency.
    CubeInfo capped{1, CubeOwner::CENTERED, {0, 0, false}, -1.0f, false, false, 1};
    require(!cube_uses_efficiency(capped), "max_cube_value=1 skips efficiency");

    CubeInfo crawford{1, CubeOwner::CENTERED, {3, 1, true}};
    require(!cube_uses_efficiency(crawford), "Crawford skips efficiency");

    CubeInfo dmp{1, CubeOwner::CENTERED, {1, 1, false}};
    require(!cube_uses_efficiency(dmp), "DMP skips efficiency");

    CubeInfo match_dead_centered{2, CubeOwner::CENTERED, {2, 2, false}};
    require(!cube_uses_efficiency(match_dead_centered), "2a-2a cube=2 skips efficiency");

    CubeInfo match_dead_owned{2, CubeOwner::OPPONENT, {3, 2, false}};
    require(!cube_uses_efficiency(match_dead_owned), "3a-2a cube=2 owned by 2-away skips efficiency");

    CubeInfo post_crawford{1, CubeOwner::CENTERED, {3, 1, false}};
    require(is_post_crawford_auto_double(post_crawford), "3a-1a post-Crawford is auto-double");
    require(!cube_uses_efficiency(post_crawford), "post-Crawford auto-double skips efficiency");

    // 2-ply evaluation across dead/auto-double/override cubes stays finite and consistent.
    const Board board = make_starting_board();
    auto pubeval = std::make_shared<PubEval>();
    for (const CubeInfo& ci : {crawford, dmp, match_dead_centered, match_dead_owned, post_crawford}) {
        const float eq = cubeful_equity_nply(board, ci, *pubeval, 2);
        require(std::isfinite(eq), "2-ply dead/auto-double match equity must be finite");
    }
    CubeInfo override_ci = money_live;
    override_ci.cube_x_override = 0.0f;
    const float eq_override_zero = cubeful_equity_nply(board, override_ci, *pubeval, 2);
    require(std::isfinite(eq_override_zero), "2-ply cube_x_override equity must be finite");
}

void test_bearoff_concurrent_indexing(const std::string& db_path) {
    auto db = std::make_unique<BearoffDB>();
    require(db->load(db_path), "failed to load bearoff DB for concurrency test");

    Board b{};
    b[1] = 2; b[2] = 3; b[3] = 1;
    b[24] = -2; b[23] = -2; b[22] = -2;
    const auto ref = db->lookup_probs(b, false);

    std::atomic<bool> ok{true};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&, t]() {
            for (int iter = 0; iter < 500; ++iter) {
                const unsigned int idx = static_cast<unsigned int>((t * 997 + iter * 31) % BearoffDB::N_POSITIONS);
                int checkers[BearoffDB::POINTS];
                BearoffDB::index_to_position(idx, checkers);
                if (BearoffDB::position_index(checkers) != idx) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                if (db->lookup_probs(b, false) != ref) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    require(ok.load(std::memory_order_relaxed), "concurrent BearoffDB indexing/lookup mismatch");
}

} // namespace

int main(int argc, char** argv) {
    init_escape_tables();

    std::string bearoff_path;
    if (argc > 1) {
        bearoff_path = argv[1];
    } else {
        namespace fs = std::filesystem;
        const auto from_file = fs::path(__FILE__).parent_path().parent_path().parent_path()
                               / "data" / "bearoff_1sided.db";
        if (fs::exists(from_file)) {
            bearoff_path = from_file.string();
        } else if (fs::exists("data/bearoff_1sided.db")) {
            bearoff_path = "data/bearoff_1sided.db";
        }
    }

    test_shared_pos_cache_abandonment_and_probing();
    if (!bearoff_path.empty()) {
        test_multipy_set_bearoff_db_detach_and_swap(bearoff_path);
        test_bearoff_concurrent_indexing(bearoff_path);
    } else {
        throw std::runtime_error("bearoff_1sided.db not found; pass its path as argv[1]");
    }
    test_routed_strategy_small_manifest_and_pin_bounds();
    test_neural_network_inputs_above_256();
    test_budget_forwarding_and_cache_raii();
    test_two_ply_details_weighted_average();
    test_cube_nply_multi_bounds();
    test_dead_cube_efficiency_skip_predicates_and_eval();

    std::cout << "test_native_core: all 9 test groups passed\n";
    return 0;
}
