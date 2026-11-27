// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bindings_common.h"
#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include "bgbot/analyzer_session.h"
#include "bgbot/board.h"
#include "bgbot/bearoff.h"
#include "bgbot/cube.h"
#include "bgbot/multipy.h"
#include "bgbot/neural_net.h"
#include "bgbot/pubeval.h"
#include "bgbot/rollout.h"

namespace bgbot::python {

namespace {

// Bridges rollout progress reports to a Python callable.
//
// Progress is reported from the rollout's WORKER threads: the binding releases
// the GIL for the rollout and each reporting thread re-acquires it to call into
// Python. If that call raises, pybind11 turns it into a C++ py::error_already_set
// — and letting that unwind out of the worker lambda reaches the thread runner,
// which calls std::terminate. When the faulted worker
// still holds the GIL it wedges instead: every other reporting thread blocks in
// gil_scoped_acquire forever. Either way the process is lost, and a caller that
// simply mistyped its callback signature gets no diagnosis at all.
//
// So the first exception is captured here rather than escaping, the rollout is
// asked to stop, and no later report calls into Python. The binding rethrows on
// the main thread once the GIL is back, so the error still reaches user code as
// an ordinary Python exception.
class ProgressBridge {
public:
    // `cancel` is invoked once, from the failing worker thread, to stop the
    // rollout early; it may be null when the caller has nothing to cancel.
    ProgressBridge(py::object cb, std::function<void()> cancel)
        : cb_(std::move(cb)), cancel_(std::move(cancel)) {}

    // The returned callback captures `this`, so the bridge must outlive the
    // rollout call and must not be copied or moved.
    ProgressBridge(const ProgressBridge&) = delete;
    ProgressBridge& operator=(const ProgressBridge&) = delete;

    RolloutProgressCallback callback() {
        if (cb_.is_none()) return nullptr;
        return [this](int completed, int total) {
            // Once one report has raised, stop calling into Python entirely:
            // the caller is going to get that first error regardless, and a
            // callback that raises once generally raises every time.
            if (failed_.load(std::memory_order_acquire)) return;
            py::gil_scoped_acquire acquire;
            try {
                cb_(completed, total);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mu_);
                if (!err_) err_ = std::current_exception();
                failed_.store(true, std::memory_order_release);
                if (cancel_) cancel_();
            }
        };
    }

    bool failed() const { return failed_.load(std::memory_order_acquire); }

    // Call on the main thread with the GIL held, after the rollout returns.
    // Rethrowing here destroys the captured py::error_already_set under the
    // GIL, which is why it cannot be done from the worker thread.
    void rethrow_if_failed() const {
        if (err_) std::rethrow_exception(err_);
    }

private:
    py::object cb_;
    std::function<void()> cancel_;
    std::atomic<bool> failed_{false};
    std::mutex mu_;
    std::exception_ptr err_;
};

// Scale a standard error in MWC space to equity space by the local slope of
// mwc2eq around `mean_mwc`. Used wherever rollout aggregates produce per-trial
// values in MWC for match play (cubeful_evaluate_board, cube_decision rollout
// bindings) so the reported SE is in the same units as the converted mean.
// mwc2eq is piecewise linear, so a small numerical derivative is exact within
// the active segment.
float mwc_se_to_eq_se(float mwc_se, float mean_mwc,
                             int away1, int away2, int cube_value,
                             bool is_crawford) {
    float eps = std::max(1e-3f, mwc_se);
    float eq_plus = mwc2eq(mean_mwc + eps, away1, away2, cube_value, is_crawford);
    float eq_minus = mwc2eq(mean_mwc - eps, away1, away2, cube_value, is_crawford);
    float slope = (eq_plus - eq_minus) / (2.0f * eps);
    return std::abs(slope) * mwc_se;
}

// Convert a vector of PlayerRollDetail to a Python list of dicts

} // namespace

void bind_rollout(py::module_& m) {
    py::register_exception<RolloutCancelled>(m, "RolloutCancelled");

    // ======================== Rollout Strategy ========================

    py::class_<RolloutConfig>(m, "RolloutConfig")
        .def(py::init<>())
        .def_readwrite("n_trials", &RolloutConfig::n_trials)
        .def_readwrite("truncation_depth", &RolloutConfig::truncation_depth)
        .def_readwrite("decision_ply", &RolloutConfig::decision_ply)
        .def_readwrite("truncation_ply", &RolloutConfig::truncation_ply)
        .def_readwrite("enable_vr", &RolloutConfig::enable_vr)
        .def_readwrite("filter", &RolloutConfig::filter)
        .def_readwrite("n_threads", &RolloutConfig::n_threads)
        .def_readwrite("seed", &RolloutConfig::seed)
        .def_readwrite("ultra_late_threshold", &RolloutConfig::ultra_late_threshold)
        .def_readwrite("cubeful_trial_moves", &RolloutConfig::cubeful_trial_moves)
        .def_readwrite("cubeful_late_threshold", &RolloutConfig::cubeful_late_threshold)
        .def_readwrite("target_se", &RolloutConfig::target_se)
        .def_readwrite("max_batches", &RolloutConfig::max_batches);

    // TrialEvalConfig: per-purpose evaluation config for rollout trials
    py::class_<TrialEvalConfig>(m, "TrialEvalConfig")
        .def(py::init<>())
        .def(py::init([](int ply, int rollout_trials, int rollout_depth, int rollout_ply) {
            TrialEvalConfig c;
            c.ply = ply;
            c.rollout_trials = rollout_trials;
            c.rollout_depth = rollout_depth;
            c.rollout_ply = rollout_ply;
            return c;
        }), "Create TrialEvalConfig (set ply for N-ply, or rollout_trials>0 for truncated rollout)",
           py::arg("ply") = 0,
           py::arg("rollout_trials") = 0,
           py::arg("rollout_depth") = 5,
           py::arg("rollout_ply") = 1)
        .def_readwrite("ply", &TrialEvalConfig::ply)
        .def_readwrite("rollout_trials", &TrialEvalConfig::rollout_trials)
        .def_readwrite("rollout_depth", &TrialEvalConfig::rollout_depth)
        .def_readwrite("rollout_ply", &TrialEvalConfig::rollout_ply)
        .def("is_set", &TrialEvalConfig::is_set)
        .def("is_rollout", &TrialEvalConfig::is_rollout);

    py::class_<RolloutResult>(m, "RolloutResult")
        .def_readonly("equity", &RolloutResult::equity)
        .def_readonly("std_error", &RolloutResult::std_error)
        .def_readonly("mean_probs", &RolloutResult::mean_probs)
        .def_readonly("prob_std_errors", &RolloutResult::prob_std_errors)
        .def_readonly("scalar_vr_equity", &RolloutResult::scalar_vr_equity)
        .def_readonly("scalar_vr_se", &RolloutResult::scalar_vr_se);

    py::class_<RolloutStrategy, Strategy, std::shared_ptr<RolloutStrategy>>(m, "RolloutStrategy")
        .def("config", &RolloutStrategy::config, py::return_value_policy::reference_internal)
        .def("clear_internal_caches", &RolloutStrategy::clear_internal_caches,
             py::call_guard<py::gil_scoped_release>(),
             "Clear thread-local N-ply caches to prevent state accumulation")
        .def("cancel", &RolloutStrategy::cancel,
             "Request cancellation of in-progress rollout. Thread-safe.")
        .def("reset_cancel", &RolloutStrategy::reset_cancel,
             "Clear cancellation flag. Call before reusing for a new rollout.")
        .def("is_cancelled", &RolloutStrategy::is_cancelled,
             "Check if cancellation was requested.")
        .def("set_move_filter", [](RolloutStrategy& self,
                                     PubEval& filter) {
            self.set_move_filter(std::make_shared<PubEval>(filter));
        }, py::call_guard<py::gil_scoped_release>(),
           "Set PubEval or other cheap strategy as pre-filter for N-ply opponent moves",
           py::arg("filter"))
        .def("rollout_position", [](const RolloutStrategy& self,
                                     const std::vector<int>& board_vec) {
            Board board = list_to_board(board_vec);
            py::gil_scoped_release release;
            return self.rollout_position(board);
        }, "Rollout a single post-move position",
           py::arg("board"))
        .def("evaluate_board", [](RolloutStrategy& self,
                                   const std::vector<int>& board_vec,
                                   const std::vector<int>& pre_move_board,
                                   py::object progress_callback) {
            auto b = list_to_board(board_vec);
            ProgressBridge bridge(progress_callback, [&self]() { self.cancel(); });
            RolloutProgressCallback cpp_progress = bridge.callback();
            RolloutResult r;
            {
                py::gil_scoped_release release;
                try {
                    r = self.rollout_position(b, cpp_progress);
                } catch (const RolloutCancelled&) {
                    // Our own cancel, if the callback raised; rethrown below as
                    // the user's exception. A genuine external cancel re-throws.
                    if (!bridge.failed()) throw;
                }
            }
            if (bridge.failed()) {
                self.reset_cancel();
                bridge.rethrow_if_failed();
            }
            py::dict result;
            result["probs"] = r.mean_probs;
            result["equity"] = r.equity;
            result["std_error"] = r.std_error;
            result["prob_std_errors"] = r.prob_std_errors;
            result["scalar_vr_equity"] = r.scalar_vr_equity;
            result["scalar_vr_se"] = r.scalar_vr_se;
            return result;
        }, "Evaluate board via rollout, returns probs, equity, std_error, prob_std_errors, scalar_vr_equity/se",
           py::arg("board"), py::arg("pre_move_board"),
           py::arg("progress") = py::none())
        .def("cubeful_evaluate_board", [](RolloutStrategy& self,
                                           const std::vector<int>& board_vec,
                                           const std::vector<int>& /*pre_move_board*/,
                                           int cube_value,
                                           CubeOwner owner,
                                           int away1,
                                           int away2,
                                           bool is_crawford,
                                           float cube_x_override,
                                           bool jacoby,
                                           bool beaver,
                                           int max_cube_value,
                                           py::object progress_callback) {
            auto b = list_to_board(board_vec);
            CubeInfo ci{cube_value, owner, {away1, away2, is_crawford},
                        cube_x_override, jacoby, beaver, max_cube_value};

            ProgressBridge bridge(progress_callback, [&self]() { self.cancel(); });
            RolloutProgressCallback cpp_progress = bridge.callback();

            RolloutStrategy::CubefulPositionResult cpr;
            {
                py::gil_scoped_release release;
                try {
                    cpr = self.cubeful_rollout_position(b, ci, cpp_progress);
                } catch (const RolloutCancelled&) {
                    if (!bridge.failed()) throw;
                }
            }
            if (bridge.failed()) {
                self.reset_cancel();
                bridge.rethrow_if_failed();
            }

            const auto& cl = cpr.cubeless;
            py::dict result;
            // Cubeless from the same rollout (post-move probs/equity in
            // SP's perspective — cubeful_rollout_position has already
            // inverted from opp's pre-roll perspective).
            result["probs"] = cl.mean_probs;
            result["equity"] = cl.equity;
            result["std_error"] = cl.std_error;
            result["prob_std_errors"] = cl.prob_std_errors;
            result["scalar_vr_equity"] = cl.scalar_vr_equity;
            result["scalar_vr_se"] = cl.scalar_vr_se;
            // Cubeful equity in basis-cube equity units (already converted
            // from MWC for match play inside cubeful_rollout_position).
            result["cubeful_equity"] = static_cast<float>(cpr.cubeful_equity);
            result["cubeful_se"] = static_cast<float>(cpr.cubeful_se);
            return result;
        }, "Rollout-level cubeful equity of a post-move position, including "
           "the opponent's optimal cube action at the start of their turn. "
           "Mirrors the multi-ply cubeful_equity_nply(opp_perspective) pattern "
           "so checker_play and cube_action stay consistent at the same "
           "eval level. Returns cubeless probs/equity from the same trials.",
           py::arg("board"), py::arg("pre_move_board"),
           py::arg("cube_value") = 1,
           py::arg("owner"),
           py::arg("away1") = 0,
           py::arg("away2") = 0,
           py::arg("is_crawford") = false,
           py::arg("cube_x_override") = -1.0f,
           py::arg("jacoby") = true,
           py::arg("beaver") = true,
           py::arg("max_cube_value") = 0,
           py::arg("progress") = py::none())
        .def("set_seed", &RolloutStrategy::set_seed,
             py::call_guard<py::gil_scoped_release>(), py::arg("seed"))
        .def("cube_decision", [](RolloutStrategy& self,
                                  const std::vector<int>& checkers,
                                  int cube_value,
                                  CubeOwner owner,
                                  int away1,
                                  int away2,
                                  bool is_crawford,
                                  float cube_x_override,
                                  bool jacoby,
                                  bool beaver,
                                  int max_cube_value,
                                  py::object progress_callback) {
            Board board = list_to_board(checkers);
            bool race = is_race(board);
            CubeInfo ci{cube_value, owner, {away1, away2, is_crawford},
                        cube_x_override, jacoby, beaver, max_cube_value};

            ProgressBridge bridge(progress_callback, [&self]() { self.cancel(); });
            RolloutProgressCallback cpp_progress = bridge.callback();

            RolloutStrategy::CubefulRolloutResult cfr;
            {
                py::gil_scoped_release release;
                try {
                    if (self.config().target_se > 0.0)
                        cfr = self.cubeful_cube_decision_batched(board, ci, cpp_progress);
                    else
                        cfr = self.cubeful_cube_decision(board, ci, cpp_progress);
                } catch (const RolloutCancelled&) {
                    if (!bridge.failed()) throw;
                }
            }
            if (bridge.failed()) {
                self.reset_cancel();
                bridge.rethrow_if_failed();
            }

            const auto& cl = cfr.cubeless;
            auto pre_roll_probs = cl.mean_probs;
            float cl_eq = cl.equity;
            double cl_se = cl.std_error;

            float equity_nd, equity_dt, equity_dp;
            float equity_nd_se, equity_dt_se;
            bool should_double, should_take;
            float optimal_equity;

            // The DT branch of `cubeful_cube_decision` always simulates "SP
            // doubles, opp takes", but that's an illegal move when SP doesn't
            // own (or share) the cube — equity_dt then represents an
            // impossible scenario and must not drive the optimal decision.
            // Mirrors the can_double guard in cube_decision_nply.
            const bool sp_can_double = can_double(ci);

            if (!ci.is_money()) {
                int a1 = ci.match.away1, a2 = ci.match.away2;
                int cv = ci.cube_value;
                bool craw = ci.match.is_crawford;

                float nd_m = static_cast<float>(cfr.nd_equity);
                float dt_m = static_cast<float>(cfr.dt_equity);
                float dp_m = dp_mwc(a1, a2, cv, craw);

                if (!sp_can_double) {
                    should_double = false;
                    should_take = true;
                } else {
                    bool auto_double = (!craw && a1 > 1 && a2 == 1);
                    if (auto_double) {
                        should_double = true;
                        should_take = (dt_m <= dp_m);
                    } else {
                        float best_mwc = std::min(dt_m, dp_m);
                        should_double = (best_mwc > nd_m);
                        should_take = (dt_m <= dp_m);
                    }
                }

                equity_nd = mwc2eq(nd_m, a1, a2, cv, craw);
                equity_dt = mwc2eq(dt_m, a1, a2, cv, craw);
                equity_dp = mwc2eq(dp_m, a1, a2, cv, craw);
                optimal_equity = should_double ? std::min(equity_dt, equity_dp)
                                               : equity_nd;

                // cfr.nd_se / dt_se are in MWC space because the per-trial
                // cubeful values from run_trial_unified are MWC for match
                // play. Scale to equity space by the local mwc2eq slope so
                // the SE matches the units of the converted means.
                equity_nd_se = mwc_se_to_eq_se(
                    static_cast<float>(cfr.nd_se), nd_m, a1, a2, cv, craw);
                equity_dt_se = mwc_se_to_eq_se(
                    static_cast<float>(cfr.dt_se), dt_m, a1, a2, cv, craw);
            } else {
                equity_dp = 1.0f;
                float actual_dt = static_cast<float>(cfr.dt_equity);
                equity_nd = static_cast<float>(cfr.nd_equity);

                if (ci.beaver && actual_dt < 0.0f) {
                    equity_dt = 2.0f * actual_dt;
                } else {
                    equity_dt = actual_dt;
                }

                if (!sp_can_double) {
                    should_double = false;
                    should_take = true;
                    optimal_equity = equity_nd;
                } else {
                    float best_double = std::min(equity_dt, equity_dp);
                    should_double = (best_double > equity_nd);
                    should_take = (equity_dt <= equity_dp);
                    optimal_equity = should_double ? best_double : equity_nd;
                }

                // Money-game SE is already in equity-per-basis-cube units.
                equity_nd_se = static_cast<float>(cfr.nd_se);
                equity_dt_se = static_cast<float>(cfr.dt_se);
            }

            bool is_beaver_result = false;
            if (ci.is_money() && ci.beaver && sp_can_double) {
                float actual_dt_check = static_cast<float>(cfr.dt_equity);
                is_beaver_result = (actual_dt_check < 0.0f);
            }

            py::dict result;
            result["probs"] = pre_roll_probs;
            result["prob_std_errors"] = cl.prob_std_errors;
            result["cubeless_equity"] = cl_eq;
            result["cubeless_se"] = cl_se;
            result["equity_nd"] = equity_nd;
            result["equity_nd_se"] = equity_nd_se;
            result["equity_dt"] = equity_dt;
            result["equity_dt_se"] = equity_dt_se;
            result["equity_dp"] = equity_dp;
            result["should_double"] = should_double;
            result["should_take"] = should_take;
            result["optimal_equity"] = optimal_equity;
            result["is_beaver"] = is_beaver_result;
            result["is_race"] = race;
            return result;
        }, "Cubeful cube-action rollout using an existing rollout strategy",
           py::arg("checkers"),
           py::arg("cube_value") = 1,
           py::arg("owner"),
           py::arg("away1") = 0,
           py::arg("away2") = 0,
           py::arg("is_crawford") = false,
           py::arg("cube_x_override") = -1.0f,
           py::arg("jacoby") = true,
           py::arg("beaver") = true,
           py::arg("max_cube_value") = 0,
           py::arg("progress") = py::none());

    // Create rollout strategy wrapping a 5-NN GamePlanStrategy
    m.def("create_rollout_5nn", [](const std::string& purerace_w,
                                    const std::string& racing_w,
                                    const std::string& attacking_w,
                                    const std::string& priming_w,
                                    const std::string& anchoring_w,
                                    int n_h_purerace,
                                    int n_h_racing,
                                    int n_h_attacking,
                                    int n_h_priming,
                                    int n_h_anchoring,
                                    int n_trials,
                                    int truncation_depth,
                                    int decision_ply,
                                    int filter_max_moves,
                                    float filter_threshold,
                                    int n_threads,
                                    uint32_t seed,
                                    int late_ply,
                                    int late_threshold,
                                    bool enable_vr,
                                    bool parallelize_trials,
                                    const TrialEvalConfig& checker,
                                    const TrialEvalConfig& checker_late,
                                    const TrialEvalConfig& cube,
                                    const TrialEvalConfig& cube_late,
                                    int ultra_late_threshold) {
        auto base = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        RolloutConfig config;
        config.n_trials = n_trials;
        config.truncation_depth = truncation_depth;
        config.decision_ply = decision_ply;
        config.enable_vr = enable_vr;
        config.parallelize_trials = parallelize_trials;
        config.filter = {filter_max_moves, filter_threshold};
        config.n_threads = n_threads;
        config.seed = seed;
        config.late_ply = late_ply;
        config.late_threshold = late_threshold;
        config.checker = checker;
        config.checker_late = checker_late;
        config.cube = cube;
        config.cube_late = cube_late;
        config.ultra_late_threshold = ultra_late_threshold;
        return std::make_shared<RolloutStrategy>(base, config);
    }, "Create rollout strategy wrapping 5-NN GamePlanStrategy",
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
       py::arg("n_trials") = 36,
       py::arg("truncation_depth") = 7,
       py::arg("decision_ply") = 1,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 0,
       py::arg("seed") = 42,
       py::arg("late_ply") = -1,
       py::arg("late_threshold") = 20,
       py::arg("enable_vr") = true,
       py::arg("parallelize_trials") = false,
       py::arg("checker") = TrialEvalConfig{},
       py::arg("checker_late") = TrialEvalConfig{},
       py::arg("cube") = TrialEvalConfig{},
       py::arg("cube_late") = TrialEvalConfig{},
       py::arg("ultra_late_threshold") = 2);

    // Create hybrid rollout strategy with separate filter and leaf 5-NN strategies.
    m.def("create_rollout_hybrid_5nn", [](
            // Leaf (accurate) strategy weights
            const std::string& purerace_w, const std::string& racing_w,
            const std::string& attacking_w, const std::string& priming_w,
            const std::string& anchoring_w,
            int n_h_purerace, int n_h_racing, int n_h_attacking,
            int n_h_priming, int n_h_anchoring,
            // Filter (fast) strategy weights
            const std::string& filter_purerace_w, const std::string& filter_racing_w,
            const std::string& filter_attacking_w, const std::string& filter_priming_w,
            const std::string& filter_anchoring_w,
            int filter_n_h_purerace, int filter_n_h_racing, int filter_n_h_attacking,
            int filter_n_h_priming, int filter_n_h_anchoring,
            // Rollout config
            int n_trials, int truncation_depth, int decision_ply,
            int filter_max_moves, float filter_threshold,
            int n_threads, uint32_t seed, int late_ply, int late_threshold,
            bool enable_vr, bool parallelize_trials,
            const TrialEvalConfig& checker, const TrialEvalConfig& checker_late,
            const TrialEvalConfig& cube, const TrialEvalConfig& cube_late,
            int ultra_late_threshold) {
        auto leaf = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        auto filter = std::make_shared<GamePlanStrategy>(
            filter_purerace_w, filter_racing_w, filter_attacking_w,
            filter_priming_w, filter_anchoring_w,
            filter_n_h_purerace, filter_n_h_racing, filter_n_h_attacking,
            filter_n_h_priming, filter_n_h_anchoring);
        RolloutConfig config;
        config.n_trials = n_trials;
        config.truncation_depth = truncation_depth;
        config.decision_ply = decision_ply;
        config.enable_vr = enable_vr;
        config.parallelize_trials = parallelize_trials;
        config.filter = {filter_max_moves, filter_threshold};
        config.n_threads = n_threads;
        config.seed = seed;
        config.late_ply = late_ply;
        config.late_threshold = late_threshold;
        config.checker = checker;
        config.checker_late = checker_late;
        config.cube = cube;
        config.cube_late = cube_late;
        config.ultra_late_threshold = ultra_late_threshold;
        return std::make_shared<RolloutStrategy>(leaf, filter, config);
    }, "Create hybrid rollout strategy with separate filter and leaf 5-NN strategies",
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
       py::arg("n_trials") = 36, py::arg("truncation_depth") = 7,
       py::arg("decision_ply") = 1, py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f, py::arg("n_threads") = 0,
       py::arg("seed") = 42, py::arg("late_ply") = -1,
       py::arg("late_threshold") = 20, py::arg("enable_vr") = true,
       py::arg("parallelize_trials") = false,
       py::arg("checker") = TrialEvalConfig{},
       py::arg("checker_late") = TrialEvalConfig{},
       py::arg("cube") = TrialEvalConfig{},
       py::arg("cube_late") = TrialEvalConfig{},
       py::arg("ultra_late_threshold") = 2);

    // Score benchmarks using rollout strategy
    // n_threads=1 for outer loop because parallelism is within each scenario
    m.def("score_benchmarks_rollout", [](const ScenarioSet& ss,
                                          std::shared_ptr<RolloutStrategy> strat,
                                          int n_threads) {
        py::gil_scoped_release release;
        return score_benchmarks(*strat, ss.scenarios, n_threads);
    }, "Score benchmarks using rollout strategy",
       py::arg("scenarios"),
       py::arg("strategy"),
       py::arg("n_threads") = 1);

    // Setter for RolloutStrategy bearoff DB
    m.def("rollout_set_bearoff_db", [](RolloutStrategy& strat, const BearoffDB* db) {
        strat.set_bearoff_db(db);
    }, py::call_guard<py::gil_scoped_release>(),
       "Set bearoff DB on a RolloutStrategy for exact bearoff evaluation",
       py::arg("strategy"), py::arg("db"));

    // --- Pair strategy rollout creation ---
    m.def("create_rollout_pair", [](const std::vector<std::string>& weight_paths,
                                     const std::vector<int>& hidden_sizes,
                                     int n_trials, int truncation_depth, int decision_ply,
                                     int filter_max_moves, float filter_threshold,
                                     int n_threads, uint32_t seed,
                                     int late_ply, int late_threshold,
                                     bool enable_vr, bool parallelize_trials,
                                     const TrialEvalConfig& checker,
                                     const TrialEvalConfig& checker_late,
                                     const TrialEvalConfig& cube,
                                     const TrialEvalConfig& cube_late,
                                     int ultra_late_threshold) {
        auto base = std::make_shared<GamePlanPairStrategy>(weight_paths, hidden_sizes);
        RolloutConfig rc;
        rc.n_trials = n_trials;
        rc.truncation_depth = truncation_depth;
        rc.decision_ply = decision_ply;
        rc.filter = {filter_max_moves, filter_threshold};
        rc.n_threads = n_threads;
        rc.seed = seed;
        rc.late_ply = late_ply;
        rc.late_threshold = late_threshold;
        rc.enable_vr = enable_vr;
        rc.parallelize_trials = parallelize_trials;
        rc.checker = checker;
        rc.checker_late = checker_late;
        rc.cube = cube;
        rc.cube_late = cube_late;
        rc.ultra_late_threshold = ultra_late_threshold;
        return std::make_shared<RolloutStrategy>(base, rc);
    }, "Create RolloutStrategy from 17-NN Pair base strategy",
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("n_trials") = 1296,
       py::arg("truncation_depth") = 0,
       py::arg("decision_ply") = 1,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 1,
       py::arg("seed") = 42,
       py::arg("late_ply") = -1,
       py::arg("late_threshold") = 20,
       py::arg("enable_vr") = true,
       py::arg("parallelize_trials") = true,
       py::arg("checker") = TrialEvalConfig{},
       py::arg("checker_late") = TrialEvalConfig{},
       py::arg("cube") = TrialEvalConfig{},
       py::arg("cube_late") = TrialEvalConfig{},
       py::arg("ultra_late_threshold") = 2);

    // --- Stage 9 rollout creation ---
    m.def("create_rollout_stage9", [](const std::vector<std::string>& weight_paths,
                                       const std::vector<int>& hidden_sizes,
                                       int n_trials, int truncation_depth, int decision_ply,
                                       int filter_max_moves, float filter_threshold,
                                       int n_threads, uint32_t seed,
                                       int late_ply, int late_threshold,
                                       bool enable_vr, bool parallelize_trials,
                                       const TrialEvalConfig& checker,
                                       const TrialEvalConfig& checker_late,
                                       const TrialEvalConfig& cube,
                                       const TrialEvalConfig& cube_late,
                                       int ultra_late_threshold) {
        auto base = std::make_shared<BackgameAwarePairStrategy>(weight_paths, hidden_sizes);
        RolloutConfig rc;
        rc.n_trials = n_trials;
        rc.truncation_depth = truncation_depth;
        rc.decision_ply = decision_ply;
        rc.filter = {filter_max_moves, filter_threshold};
        rc.n_threads = n_threads;
        rc.seed = seed;
        rc.late_ply = late_ply;
        rc.late_threshold = late_threshold;
        rc.enable_vr = enable_vr;
        rc.parallelize_trials = parallelize_trials;
        rc.checker = checker;
        rc.checker_late = checker_late;
        rc.cube = cube;
        rc.cube_late = cube_late;
        rc.ultra_late_threshold = ultra_late_threshold;
        return std::make_shared<RolloutStrategy>(base, rc);
    }, "Create RolloutStrategy from 19-NN Backgame-Aware Pair base strategy",
       py::arg("weight_paths"),
       py::arg("hidden_sizes"),
       py::arg("n_trials") = 1296,
       py::arg("truncation_depth") = 0,
       py::arg("decision_ply") = 1,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 1,
       py::arg("seed") = 42,
       py::arg("late_ply") = -1,
       py::arg("late_threshold") = 20,
       py::arg("enable_vr") = true,
       py::arg("parallelize_trials") = true,
       py::arg("checker") = TrialEvalConfig{},
       py::arg("checker_late") = TrialEvalConfig{},
       py::arg("cube") = TrialEvalConfig{},
       py::arg("cube_late") = TrialEvalConfig{},
       py::arg("ultra_late_threshold") = 2);

    m.def("create_rollout_from_base", [](std::shared_ptr<Strategy> base,
                                              int n_trials, int truncation_depth, int decision_ply,
                                              int filter_max_moves, float filter_threshold,
                                              int n_threads, uint32_t seed,
                                              int late_ply, int late_threshold,
                                              bool enable_vr, bool parallelize_trials,
                                              const TrialEvalConfig& checker,
                                              const TrialEvalConfig& checker_late,
                                              const TrialEvalConfig& cube,
                                              const TrialEvalConfig& cube_late,
                                              int ultra_late_threshold,
                                              bool cubeful_trial_moves,
                                              int cubeful_late_threshold,
                                              double target_se,
                                              int max_batches, int truncation_ply) {
        if (!base) throw std::invalid_argument("base strategy is None");
        py::gil_scoped_release release;
        RolloutConfig rc;
        rc.n_trials = n_trials;
        rc.truncation_depth = truncation_depth;
        rc.decision_ply = decision_ply;
        rc.filter = {filter_max_moves, filter_threshold};
        rc.n_threads = n_threads;
        rc.seed = seed;
        rc.late_ply = late_ply;
        rc.late_threshold = late_threshold;
        rc.enable_vr = enable_vr;
        rc.parallelize_trials = parallelize_trials;
        rc.checker = checker;
        rc.checker_late = checker_late;
        rc.cube = cube;
        rc.cube_late = cube_late;
        rc.ultra_late_threshold = ultra_late_threshold;
        rc.cubeful_trial_moves = cubeful_trial_moves;
        rc.cubeful_late_threshold = cubeful_late_threshold;
        rc.target_se = target_se;
        rc.max_batches = max_batches;
        rc.truncation_ply = truncation_ply;
        return std::make_shared<RolloutStrategy>(base, rc);
    }, "Create RolloutStrategy around loaded weights",
       py::arg("base"),
       py::arg("n_trials") = 1296,
       py::arg("truncation_depth") = 0,
       py::arg("decision_ply") = 1,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 1,
       py::arg("seed") = 42,
       py::arg("late_ply") = -1,
       py::arg("late_threshold") = 20,
       py::arg("enable_vr") = true,
       py::arg("parallelize_trials") = true,
       py::arg("checker") = TrialEvalConfig{},
       py::arg("checker_late") = TrialEvalConfig{},
       py::arg("cube") = TrialEvalConfig{},
       py::arg("cube_late") = TrialEvalConfig{},
       py::arg("ultra_late_threshold") = 2,
       py::arg("cubeful_trial_moves") = true,
       py::arg("cubeful_late_threshold") = 0,
       py::arg("target_se") = 0.0,
       py::arg("max_batches") = 50,
       py::arg("truncation_ply") = -1);

    m.def("create_rollout", [factory = py::object(m.attr("create_rollout_from_base"))]
          (const std::string& strategy_type, const std::vector<std::string>& weight_paths,
           const std::vector<int>& hidden_sizes, py::args args, py::kwargs kwargs) {
        std::shared_ptr<Strategy> base;
        {
            py::gil_scoped_release release;
            base = make_strategy_from_type(strategy_type, weight_paths, hidden_sizes);
        }
        return factory(base, *args, **kwargs);
    }, py::arg("strategy_type"), py::arg("weight_paths"), py::arg("hidden_sizes"));
}

void bind_cube_rollout(py::module_& m) {
    // Cubeful rollout: simulates cube decisions during trial games.
    // Two branches (ND and DT) share the same board evolution and dice sequences.
    // Cube decisions (double/take/pass) evaluated at 1-ply Janowski at each half-move.
    m.def("cube_decision_rollout", [](const std::vector<int>& checkers,
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
                                       int n_trials,
                                       int truncation_depth,
                                       int decision_ply,
                                       int filter_max_moves,
                                       float filter_threshold,
                                       int n_threads,
                                       uint32_t seed,
                                       int late_ply,
                                       int late_threshold,
                                       int away1, int away2, bool is_crawford,
                                       float cube_x_override,
                                       bool enable_vr, bool jacoby,
                                       bool beaver, int max_cube_value,
                                       const TrialEvalConfig& checker,
                                       const TrialEvalConfig& checker_late,
                                       const TrialEvalConfig& cube_cfg,
                                       const TrialEvalConfig& cube_late,
                                       int ultra_late_threshold,
                                       py::object progress_callback) {
        Board board = list_to_board(checkers);
        bool race = is_race(board);

        auto base = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        RolloutConfig config;
        config.n_trials = n_trials;
        config.truncation_depth = truncation_depth;
        config.decision_ply = decision_ply;
        config.enable_vr = enable_vr;
        config.filter = {filter_max_moves, filter_threshold};
        config.n_threads = n_threads;
        config.seed = seed;
        config.late_ply = late_ply;
        config.late_threshold = late_threshold;
        config.checker = checker;
        config.checker_late = checker_late;
        config.cube = cube_cfg;
        config.cube_late = cube_late;
        config.ultra_late_threshold = ultra_late_threshold;
        auto rollout = std::make_shared<RolloutStrategy>(base, config);

        CubeInfo ci{cube_value, owner, {away1, away2, is_crawford}, cube_x_override, jacoby, beaver, max_cube_value};

        // Build C++ progress callback that acquires GIL to call Python
        ProgressBridge bridge(progress_callback, [&rollout]() { rollout->cancel(); });
        RolloutProgressCallback cpp_progress = bridge.callback();

        RolloutStrategy::CubefulRolloutResult cfr;
        {
            py::gil_scoped_release release;
            try {
                cfr = rollout->cubeful_cube_decision(board, ci, cpp_progress);
            } catch (const RolloutCancelled&) {
                if (!bridge.failed()) throw;
            }
        }
        if (bridge.failed()) {
            bridge.rethrow_if_failed();
        }

        // Cubeless probs come from the cubeful rollout's integrated cubeless rollout
        const auto& cl = cfr.cubeless;
        auto pre_roll_probs = cl.mean_probs;
        float cl_eq = cl.equity;
        double cl_se = cl.std_error;

        // Decision logic: match play works in MWC space, money in equity
        float equity_nd, equity_dt, equity_dp;
        float equity_nd_se, equity_dt_se;
        bool should_double, should_take;
        float optimal_equity;

        // The DT branch of `cubeful_cube_decision` always simulates "SP
        // doubles, opp takes", but that's an illegal move when SP can't
        // legally double — equity_dt then represents an impossible scenario
        // and must not drive the optimal decision. Mirrors the can_double
        // guard in cube_decision_nply.
        const bool sp_can_double = can_double(ci);

        if (!ci.is_money()) {
            // Match play: cfr.nd_equity / dt_equity are MWC from SP perspective
            int a1 = ci.match.away1, a2 = ci.match.away2;
            int cv = ci.cube_value;
            bool craw = ci.match.is_crawford;

            float nd_m = static_cast<float>(cfr.nd_equity);
            float dt_m = static_cast<float>(cfr.dt_equity);
            float dp_m = dp_mwc(a1, a2, cv, craw);

            if (!sp_can_double) {
                should_double = false;
                should_take = true;
            } else {
                // Post-Crawford automatic double: trailing player always doubles
                bool auto_double = (!craw && a1 > 1 && a2 == 1);
                // Decision in MWC space
                if (auto_double) {
                    should_double = true;
                    should_take = (dt_m <= dp_m);
                } else {
                    float best_mwc = std::min(dt_m, dp_m);
                    should_double = (best_mwc > nd_m);
                    should_take = (dt_m <= dp_m);
                }
            }

            // Convert to equity at original cv for display
            equity_nd = mwc2eq(nd_m, a1, a2, cv, craw);
            equity_dt = mwc2eq(dt_m, a1, a2, cv, craw);
            equity_dp = mwc2eq(dp_m, a1, a2, cv, craw);
            optimal_equity = should_double ? std::min(equity_dt, equity_dp)
                                           : equity_nd;

            // cfr.nd_se / dt_se are in MWC space; scale to equity by the
            // local mwc2eq slope so SE matches the units of the means.
            equity_nd_se = mwc_se_to_eq_se(
                static_cast<float>(cfr.nd_se), nd_m, a1, a2, cv, craw);
            equity_dt_se = mwc_se_to_eq_se(
                static_cast<float>(cfr.dt_se), dt_m, a1, a2, cv, craw);
        } else {
            // Money game: existing logic
            equity_dp = 1.0f;
            float actual_dt = static_cast<float>(cfr.dt_equity);
            equity_nd = static_cast<float>(cfr.nd_equity);

            // Beaver: if enabled and DT < 0, opponent beavers (DB = 2*DT)
            bool is_beaver_flag = false;
            if (ci.beaver && actual_dt < 0.0f) {
                equity_dt = 2.0f * actual_dt;  // DB equity
                is_beaver_flag = true;
            } else {
                equity_dt = actual_dt;
            }

            if (!sp_can_double) {
                should_double = false;
                should_take = true;
                optimal_equity = equity_nd;
            } else {
                float best_double = std::min(equity_dt, equity_dp);
                should_double = (best_double > equity_nd);
                should_take = (equity_dt <= equity_dp);
                optimal_equity = should_double ? best_double : equity_nd;
            }

            // Money-game SE is already in equity-per-basis-cube units.
            equity_nd_se = static_cast<float>(cfr.nd_se);
            equity_dt_se = static_cast<float>(cfr.dt_se);
        }

        // Determine is_beaver for non-money path (always false for match play)
        bool is_beaver_result = false;
        if (ci.is_money() && ci.beaver && sp_can_double) {
            float actual_dt_check = static_cast<float>(cfr.dt_equity);
            is_beaver_result = (actual_dt_check < 0.0f);
        }

        py::dict result;
        result["probs"] = pre_roll_probs;
        result["prob_std_errors"] = cl.prob_std_errors;
        result["cubeless_equity"] = cl_eq;
        result["cubeless_se"] = cl_se;
        result["equity_nd"] = equity_nd;
        result["equity_nd_se"] = equity_nd_se;
        result["equity_dt"] = equity_dt;
        result["equity_dt_se"] = equity_dt_se;
        result["equity_dp"] = equity_dp;
        result["should_double"] = should_double;
        result["should_take"] = should_take;
        result["optimal_equity"] = optimal_equity;
        result["is_beaver"] = is_beaver_result;
        result["is_race"] = race;
        return result;
    }, "Cubeful rollout with separate checker/cube strategies during trial games.\n"
       "Two branches (ND and DT) share same board/dice. Cube decisions use configured cube strategy.\n"
       "Pass checker/cube TrialEvalConfig for separate checker/cube evaluation strengths.",
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
       py::arg("n_trials") = 1296,
       py::arg("truncation_depth") = 0,
       py::arg("decision_ply") = 1,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 0,
       py::arg("seed") = 42,
       py::arg("late_ply") = -1,
       py::arg("late_threshold") = 20,
       py::arg("away1") = 0, py::arg("away2") = 0, py::arg("is_crawford") = false,
       py::arg("cube_x_override") = -1.0f,
       py::arg("enable_vr") = true,
       py::arg("jacoby") = true,
       py::arg("beaver") = true,
       py::arg("max_cube_value") = 0,
       py::arg("checker") = TrialEvalConfig{},
       py::arg("checker_late") = TrialEvalConfig{},
       py::arg("cube") = TrialEvalConfig{},
       py::arg("cube_late") = TrialEvalConfig{},
       py::arg("ultra_late_threshold") = 2,
       py::arg("progress") = py::none());
}

} // namespace bgbot::python
