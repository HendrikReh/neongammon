// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// Internal shared helpers for rollout_{dice,cache,trial,runner}.cpp.
#pragma once

#include "bgbot/rollout.h"
#include "bgbot/rollout_vr.h"
#include "bgbot/multipy.h"
#include "bgbot/cube.h"
#include "bgbot/match_equity.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/encoding.h"
#include "bgbot/bearoff.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// Define ROLLOUT_PROFILE to enable lightweight per-phase timing counters.
// #define ROLLOUT_PROFILE
#ifdef ROLLOUT_PROFILE
#include <chrono>
#include <cstdio>
namespace rollout_profile {
    inline std::atomic<int64_t> vr_time_ns{0};
    inline std::atomic<int64_t> trunc_time_ns{0};
    inline std::atomic<int64_t> cube_time_ns{0};
    inline std::atomic<int64_t> movegen_time_ns{0};
    inline std::atomic<int64_t> bmi_time_ns{0};
    inline std::atomic<int64_t> trial_count{0};

    inline void reset() {
        vr_time_ns = 0; trunc_time_ns = 0; cube_time_ns = 0;
        movegen_time_ns = 0; bmi_time_ns = 0; trial_count = 0;
    }
    inline void print() {
        int64_t n = trial_count.load();
        printf("  Profile: vr=%.1fms bmi=%.1fms trunc=%.1fms cube=%.1fms movegen=%.1fms trials=%lld\n",
               vr_time_ns / 1e6, bmi_time_ns / 1e6, trunc_time_ns / 1e6, cube_time_ns / 1e6,
               movegen_time_ns / 1e6, (long long)n);
    }
}
#define ROLLOUT_TIMER_START auto _rp_timer = std::chrono::high_resolution_clock::now()
#define ROLLOUT_TIMER_ADD(counter) rollout_profile::counter.fetch_add( \
    std::chrono::duration_cast<std::chrono::nanoseconds>( \
        std::chrono::high_resolution_clock::now() - _rp_timer).count(), \
    std::memory_order_relaxed)
#else
namespace rollout_profile {
    inline void reset() {}
    inline void print() {}
}
#define ROLLOUT_TIMER_START (void)0
#define ROLLOUT_TIMER_ADD(counter) (void)0
#endif

namespace bgbot {

// The same board can occur in several trials even after the opening rolls.
// The VR policy depends on the board and the active selection cubes. The base
// evaluator and bearoff database are fixed for the lifetime of this table.
struct RolloutStrategy::VarianceReductionCache {
    static constexpr int capacity = 2048;
    using Probabilities = std::array<std::array<float, NUM_OUTPUTS>, Move0Cache::N_ROLLS>;
    using Picks = std::array<int, Move0Cache::N_ROLLS>;

    struct Entry {
        Board board;
        std::array<CubeInfo, 4> cubes;
        int cube_count;
        float cube_x;
        Probabilities probabilities;
        Picks picks;
    };
    std::array<std::atomic<uint8_t>, capacity> state;
    std::array<Entry, capacity> entries;

    VarianceReductionCache() {
        for (auto& slot : state) slot.store(0, std::memory_order_relaxed);
    }

    static size_t slot_for(const Board& board) {
        return rollout_board_hash(board) & (capacity - 1);
    }

    static bool same_cube(const CubeInfo& a, const CubeInfo& b) {
        return a.cube_value == b.cube_value && a.owner == b.owner &&
            a.match.away1 == b.match.away1 && a.match.away2 == b.match.away2 &&
            a.match.is_crawford == b.match.is_crawford &&
            a.cube_x_override == b.cube_x_override && a.jacoby == b.jacoby &&
            a.beaver == b.beaver && a.max_cube_value == b.max_cube_value &&
            a.contact_x == b.contact_x &&
            a.contact_x_pip_slope == b.contact_x_pip_slope;
    }

    const Entry* lookup(size_t slot, const Board& board, const CubeInfo* cubes,
                        int cube_count, float cube_x) const {
        if (state[slot].load(std::memory_order_acquire) != 2) return nullptr;
        const Entry& entry = entries[slot];
        if (entry.board != board || entry.cube_count != cube_count || entry.cube_x != cube_x)
            return nullptr;
        for (int b = 0; b < cube_count; ++b) {
            if (!same_cube(entry.cubes[b], cubes[b])) return nullptr;
        }
        return &entry;
    }

    void publish(size_t slot, const Board& board, const CubeInfo* cubes, int cube_count,
                 float cube_x, const Probabilities& probabilities, const Picks& picks) {
        // Every slot has a single writer; a concurrent fill or a collision
        // leaves the original evaluation intact. Published data is immutable.
        uint8_t empty = 0;
        if (!state[slot].compare_exchange_strong(empty, 1, std::memory_order_relaxed)) return;
        Entry& entry = entries[slot];
        entry.board = board;
        entry.cube_count = cube_count;
        for (int b = 0; b < cube_count; ++b) entry.cubes[b] = cubes[b];
        entry.cube_x = cube_x;
        entry.probabilities = probabilities;
        entry.picks = picks;
        state[slot].store(2, std::memory_order_release);
    }
};

namespace rollout_detail {

inline constexpr std::array<std::array<int, 7>, 7> kOrderedRollToIndex = {{
    {-1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  6,  7,  8,  9, 10},
    {-1,  6,  1, 11, 12, 13, 14},
    {-1,  7, 11,  2, 15, 16, 17},
    {-1,  8, 12, 15,  3, 18, 19},
    {-1,  9, 13, 16, 18,  4, 20},
    {-1, 10, 14, 17, 19, 20,  5},
}};

inline int trial_chunk_size(int depth, int trials, int workers) {
    // Full games vary widely in length. Let each idle worker take the next
    // game, as for short truncated batches with few trials per worker.
    return depth <= 0 || trials / workers < 32 ? 1 : 8;
}

inline int trial_report_interval(int depth, int trials, int workers) {
    // Full-game workers claim individual paths, but a caller can publish
    // every progress event. Keep updates at least eight paths apart in
    // parallel full rollouts, including each batch of an adaptive cube call.
    const int minimum = depth <= 0 && workers > 1 ? 8 : 1;
    return std::max(minimum, trials / 100);
}

// A dead money cube: cube_is_dead() is true, so the cubeful evaluation
// engine bypasses Janowski everywhere and its tree reduces exactly to a
// cubeless N-ply evaluation.
inline const CubeInfo& dead_cube_info() {
    static const CubeInfo dead = [] {
        CubeInfo c;
        c.cube_value = 1;
        c.max_cube_value = 1;
        return c;
    }();
    return dead;
}

// A nested checker rollout can enter the same generator on the same worker
// while its caller still holds a candidate index. An active frame owns its
// buffer. Sequential frames reuse the idle allocation on this worker.
template<class T>
class TrialMoveScratch {
    inline static thread_local std::unique_ptr<T> idle_;
    std::unique_ptr<T> storage_ = std::move(idle_);
public:
    TrialMoveScratch() {
        if (!storage_) storage_ = std::make_unique<T>();
    }
    ~TrialMoveScratch() { idle_ = std::move(storage_); }
    T& get() { return *storage_; }
};

// Restores the thread-local SharedPosCache pointer on scope exit (including
// exception unwind), preventing dangling pointers on persistent pool threads.
class ScopedSharedCache {
    SharedPosCache* prev_ = nullptr;
    bool active_ = false;
public:
    explicit ScopedSharedCache(SharedPosCache* cache, bool apply_null = true) {
        if (cache || apply_null) {
            prev_ = MultiPlyStrategy::get_shared_cache();
            MultiPlyStrategy::set_shared_cache(cache);
            active_ = true;
        }
    }
    ~ScopedSharedCache() {
        if (active_) MultiPlyStrategy::set_shared_cache(prev_);
    }
    ScopedSharedCache(const ScopedSharedCache&) = delete;
    ScopedSharedCache& operator=(const ScopedSharedCache&) = delete;
};

// Serializes progress callbacks across rollout workers and guarantees strictly
// monotonic completed counts while preserving exact report cadence.
class MonotonicProgressReporter {
    const RolloutProgressCallback& progress_;
    int total_;
    int report_interval_;
    int chunk_size_;
    std::mutex mutex_;
    int completed_ = 0;
    int last_reported_ = 0;
public:
    MonotonicProgressReporter(const RolloutProgressCallback& progress,
                              int total, int report_interval, int chunk_size)
        : progress_(progress)
        , total_(total)
        , report_interval_(report_interval)
        , chunk_size_(chunk_size) {}

    void report_chunk(int delta) {
        if (!report_interval_ || delta <= 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        completed_ += delta;
        const int clamped = std::min(completed_, total_);
        if ((completed_ % report_interval_ < chunk_size_ || completed_ >= total_)
            && clamped > last_reported_) {
            last_reported_ = clamped;
            progress_(clamped, total_);
        }
    }
};

// Claim-or-wait helper for tri-state atomic cache slots:
//   0 = empty, 1 = computing, 2 = ready, -1 = failed.
// Returns the resulting state (2 on success, or non-2 if another writer
// failed). When `compute` throws, stores -1 and rethrows.
template<typename ComputeFn>
int cas_claim_or_wait(std::atomic<int>& state, ComputeFn&& compute) {
    int s = state.load(std::memory_order_acquire);
    if (s == 2) return 2;
    int expected = 0;
    if (state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
        try {
            compute();
        } catch (...) {
            state.store(-1, std::memory_order_release);
            throw;
        }
        state.store(2, std::memory_order_release);
        return 2;
    }
    int wait_s = state.load(std::memory_order_acquire);
    while (wait_s == 1) {
        std::this_thread::yield();
        wait_s = state.load(std::memory_order_acquire);
    }
    return wait_s;
}

// Accumulates per-trial statistics (cubeless probabilities, equities, and
// cubeful ND/DT branch equities) across single or multi-batch rollouts.
struct RolloutStatsAccumulator {
    std::array<double, NUM_OUTPUTS> sum_probs = {0, 0, 0, 0, 0};
    std::array<double, NUM_OUTPUTS> sum_probs_sq = {0, 0, 0, 0, 0};
    double sum_eq = 0.0;
    double sum_eq_sq = 0.0;
    double sum_svr_eq = 0.0;
    double sum_svr_eq_sq = 0.0;
    double sum_nd = 0.0;
    double sum_nd_sq = 0.0;
    double sum_dt = 0.0;
    double sum_dt_sq = 0.0;
    long long count = 0;

    void add_cubeless(const std::array<float, NUM_OUTPUTS>& probs,
                      double equity,
                      double scalar_vr_equity) {
        for (int k = 0; k < NUM_OUTPUTS; ++k) {
            const double v = probs[k];
            sum_probs[k] += v;
            sum_probs_sq[k] += v * v;
        }
        sum_eq += equity;
        sum_eq_sq += equity * equity;
        sum_svr_eq += scalar_vr_equity;
        sum_svr_eq_sq += scalar_vr_equity * scalar_vr_equity;
        ++count;
    }

    void add_cubeful(double nd_equity,
                     double dt_equity,
                     const std::array<float, NUM_OUTPUTS>& probs,
                     double cl_equity) {
        sum_nd += nd_equity;
        sum_nd_sq += nd_equity * nd_equity;
        sum_dt += dt_equity;
        sum_dt_sq += dt_equity * dt_equity;
        for (int k = 0; k < NUM_OUTPUTS; ++k) {
            const double v = probs[k];
            sum_probs[k] += v;
            sum_probs_sq[k] += v * v;
        }
        sum_eq += cl_equity;
        sum_eq_sq += cl_equity * cl_equity;
        ++count;
    }

    static double std_error_from_sums(double sum, double sum_sq, long long n) {
        if (n <= 0) return 0.0;
        const double mean = sum / n;
        double var = (sum_sq / n) - (mean * mean);
        if (var < 0.0) var = 0.0;
        return std::sqrt(var / n);
    }

    double current_nd_se() const {
        return std_error_from_sums(sum_nd, sum_nd_sq, count);
    }

    RolloutResult finalize_cubeless(bool equity_from_mean_probs) const {
        RolloutResult result;
        if (count <= 0) return result;
        for (int k = 0; k < NUM_OUTPUTS; ++k) {
            const double mean_k = sum_probs[k] / count;
            result.mean_probs[k] = static_cast<float>(mean_k);
            result.prob_std_errors[k] = static_cast<float>(
                std_error_from_sums(sum_probs[k], sum_probs_sq[k], count));
        }
        const double mean_eq = sum_eq / count;
        result.std_error = std_error_from_sums(sum_eq, sum_eq_sq, count);
        if (equity_from_mean_probs) {
            result.equity = cubeless_equity(result.mean_probs);
            result.scalar_vr_equity = mean_eq;
            result.scalar_vr_se = result.std_error;
        } else {
            result.equity = mean_eq;
            result.scalar_vr_equity = sum_svr_eq / count;
            result.scalar_vr_se = std_error_from_sums(sum_svr_eq, sum_svr_eq_sq, count);
        }
        return result;
    }

    RolloutStrategy::CubefulRolloutResult finalize_cubeful(
        int n_batches, bool se_converged) const {
        RolloutStrategy::CubefulRolloutResult result;
        if (count > 0) {
            result.nd_equity = sum_nd / count;
            result.nd_se = std_error_from_sums(sum_nd, sum_nd_sq, count);
            result.dt_equity = sum_dt / count;
            result.dt_se = std_error_from_sums(sum_dt, sum_dt_sq, count);
            result.cubeless = finalize_cubeless(/*equity_from_mean_probs=*/true);
        }
        result.n_batches = n_batches;
        result.total_trials = count;
        result.se_converged = se_converged;
        return result;
    }
};

} // namespace rollout_detail

} // namespace bgbot
