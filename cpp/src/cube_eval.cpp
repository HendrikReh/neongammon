// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// N-ply cubeful evaluation engine: the evaluate-all-and-decide recursion
// behind cube_decision_nply(_multi), cubeful_equity_nply(_multi),
// cubeful_probs_nply and cubeful_probs_and_equity_nply (declared in cube.h;
// the single-cube wrappers live in cube.cpp and compose the multi entries
// defined here).
//
// Key mechanics (see MULTI-PLY.md for the full specification):
//   * Interior move-picks evaluate all candidates of a roll through a batched
//     delta-evaluation kernel (encode base candidate once, sparse hidden-layer
//     updates per sibling) with per-candidate game-plan NN selection, then
//     pick by 1-ply cubeful equity against the primary cube state.
//   * At plies==2 internal nodes the leaf value of the picked move reuses the
//     pick batch's NN output (leaf = invert(NN(chosen))) � no leaf NN evals.
//   * The cubeless probs of the tree are accumulated through the same
//     traversal and can be returned via probs_out � one walk serves both
//     cubeful equities and cubeless probabilities.
//   * Memoization is a per-thread cache keyed by (board, plies, cci, fTop,
//     cube-state fingerprint); entries stay valid across calls and threads
//     never invalidate each other. clear_cubeful_eval_cache() resets it.
//   * The 21 rolls at the top tree levels run in parallel when the caller
//     passes n_threads > 1 (standalone analytics); rollout-internal calls run
//     serial because trial dispatch already saturates the cores.

#include "bgbot/cube.h"
#include "bgbot/neural_net.h"
#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/encoding.h"
#include "bgbot/match_equity.h"
#include "bgbot/pubeval.h"
#include "bgbot/multipy.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace bgbot {

namespace {

constexpr int EVAL_MAX_CCI = 64;   // matches MAX_CCI in cube.cpp

void validate_eval_options(EvalOptions options) {
    if (options.deep_prefilter_keep < 1 ||
        options.deep_prefilter_threshold < options.deep_prefilter_keep) {
        throw std::invalid_argument("Deep cube filter requires 1 <= keep <= threshold");
    }
}

// ======================== Cube-state helpers ========================

float resolve_cube_x_local(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube, const Board& board, bool race)
{
    auto [pp, op] = pip_counts(board);
    return resolve_cube_x(probs, cube, race, pp, op);
}

void make_cube_pos_local(
    const CubeInfo input[], int cci, bool fTop,
    CubeInfo output[], bool fInvert)
{
    for (int ici = 0, i = 0; ici < cci; ici++) {
        if (input[ici].cube_value > 0) {
            output[i] = fInvert ? flip_cube_perspective(input[ici]) : input[ici];
        } else {
            output[i].cube_value = -1;
        }
        i++;

        if (!fTop && input[ici].cube_value > 0 && can_double(input[ici])) {
            CubeInfo dt = input[ici];
            dt.cube_value = 2 * input[ici].cube_value;
            dt.owner = CubeOwner::OPPONENT;
            output[i] = fInvert ? flip_cube_perspective(dt) : dt;
        } else {
            output[i].cube_value = -1;
        }
        i++;
    }
}

void get_ecf3_local(
    float arCubeful[], int cci,
    const float arCf[], const CubeInfo aci[])
{
    for (int ici = 0, i = 0; ici < cci; ici++, i += 2) {
        if (aci[i + 1].cube_value > 0) {
            float rND = arCf[i];
            float rDT;
            bool is_money = aci[i].is_money();

            if (is_money) {
                rDT = 2.0f * arCf[i + 1];
            } else {
                rDT = arCf[i + 1];
            }

            if (is_money && aci[i].beaver && rDT < 0.0f) {
                rDT = 2.0f * rDT;
            }

            float rDP;
            if (is_money) {
                rDP = 1.0f;
            } else {
                rDP = dp_mwc(aci[i].match.away1, aci[i].match.away2,
                             aci[i].cube_value, aci[i].match.is_crawford);
            }

            if (rDT >= rND && rDP >= rND) {
                arCubeful[ici] = (rDT >= rDP) ? rDP : rDT;
            } else {
                arCubeful[ici] = rND;
            }
        } else {
            arCubeful[ici] = arCf[i];
        }
    }
}

// ======================== Per-thread cache ========================
//
// Open-addressing, epoch-free. The key incorporates a fingerprint of the
// cube-state array, so entries remain valid across calls and across cube
// contexts — no global invalidation needed (unlike the old cubeful cache,
// whose process-global epoch bump per top-level call destroyed every other
// thread's in-flight memoization under parallel trial dispatch).

uint64_t fp_mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 12) + (h >> 4);
    return h;
}

static uint32_t float_bits(float v) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(v));
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

uint64_t cube_fp_one(uint64_t h, const CubeInfo& c) {
    h = fp_mix(h, static_cast<uint32_t>(c.cube_value));
    h = fp_mix(h, static_cast<int>(c.owner) + 3);
    h = fp_mix(h, static_cast<uint32_t>(c.match.away1) * 131u
                  + static_cast<uint32_t>(c.match.away2));
    h = fp_mix(h, c.match.is_crawford ? 7 : 11);
    h = fp_mix(h, c.jacoby ? 13 : 17);
    h = fp_mix(h, c.beaver ? 19 : 23);
    h = fp_mix(h, static_cast<uint32_t>(c.max_cube_value) + 29);
    h = fp_mix(h, float_bits(c.cube_x_override));
    h = fp_mix(h, float_bits(c.contact_x));
    h = fp_mix(h, float_bits(c.contact_x_pip_slope));
    return h;
}

uint64_t hash_board64(const Board& b) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 26; ++i) {
        h ^= static_cast<uint32_t>(b[i] + 32);
        h *= 0x100000001b3ULL;
    }
    return h;
}

struct CubefulEvalCacheEntry {
    uint64_t key = 0;
    uint64_t cube_fp = 0;
    Board board{};
    int16_t plies = 0;
    int16_t cci = 0;
    uint8_t ftop = 0;
    uint32_t generation = 0;
    float vals[16];
    float probs[NUM_OUTPUTS];
};

struct CubefulEvalCache {
    static constexpr size_t SIZE = 1 << 15;   // 32768 entries (~7 MB)
    static constexpr size_t MASK = SIZE - 1;
    static constexpr int MAX_PROBE = 4;
    std::vector<CubefulEvalCacheEntry> entries;
    uint32_t generation = 1;

    void clear() {
        if (++generation == 0) {
            for (auto& e : entries) e.generation = 0;
            generation = 1;
        }
    }

    bool get(uint64_t key, uint64_t fp, const Board& b, int plies, int cci,
             bool ftop, float* vals, std::array<float, NUM_OUTPUTS>* probs) const
    {
        if (entries.empty()) return false;
        size_t idx = static_cast<size_t>(key) & MASK;
        for (int p = 0; p < MAX_PROBE; ++p, idx = (idx + 1) & MASK) {
            const auto& e = entries[idx];
            // Inserts stop at the first vacant slot. No occupied slots are removed
            // within a generation, so a vacant slot terminates the probe.
            if (e.generation != generation) return false;
            if (e.key != key) continue;
            if (e.cube_fp != fp || e.plies != plies || e.cci != cci ||
                e.ftop != (ftop ? 1 : 0) || !(e.board == b)) continue;
            for (int i = 0; i < cci; ++i) vals[i] = e.vals[i];
            if (probs) {
                for (int k = 0; k < NUM_OUTPUTS; ++k) (*probs)[k] = e.probs[k];
            }
            return true;
        }
        return false;
    }

    void put(uint64_t key, uint64_t fp, const Board& b, int plies, int cci,
             bool ftop, const float* vals,
             const std::array<float, NUM_OUTPUTS>& probs)
    {
        // Threads that only clear the cache or miss a lookup need no table.
        if (entries.empty()) entries.resize(SIZE);
        size_t idx = static_cast<size_t>(key) & MASK;
        size_t target = idx;
        for (int p = 0; p < MAX_PROBE; ++p, idx = (idx + 1) & MASK) {
            const auto& e = entries[idx];
            if (e.generation != generation || e.key == key) { target = idx; break; }
        }
        auto& e = entries[target];
        e.generation = generation;
        e.key = key;
        e.cube_fp = fp;
        e.board = b;
        e.plies = static_cast<int16_t>(plies);
        e.cci = static_cast<int16_t>(cci);
        e.ftop = ftop ? 1 : 0;
        for (int i = 0; i < cci; ++i) e.vals[i] = vals[i];
        for (int k = 0; k < NUM_OUTPUTS; ++k) e.probs[k] = probs[k];
    }
};

thread_local CubefulEvalCache g_cubeful_eval_cache;

uint64_t make_cache_key(const Board& b, uint64_t fp, int plies, int cci, bool ftop) {
    uint64_t h = hash_board64(b);
    h = fp_mix(h, fp);
    h = fp_mix(h, static_cast<uint64_t>(plies) * 0x517cc1b727220a95ULL);
    h = fp_mix(h, static_cast<uint64_t>(cci) * 0x6c62272e07bb0142ULL);
    h = fp_mix(h, ftop ? 0x2545F4914F6CDD1DULL : 0x9E6C63D0876A9ULL);
    if (h == 0) h = 1;
    return h;
}

// Shared PubEval instance for the aggressive deep-node pre-filter.
const PubEval& static_pubeval() {
    static const PubEval pe(PubEval::WeightSource::TESAURO);
    return pe;
}

// ======================== Batched candidate evaluation ========================

// A node may distribute its 21 rolls across workers. Each worker saves a
// pre-roll base for this node the first time it evaluates a roll here:
//   - On Apple Silicon (Accelerate): caches the pre-roll input vector per
//     encoding kind (race / contact) so every candidate encodes incrementally
//     before one cblas_sgemm forward_group call.
//   - On non-Accelerate builds (e.g. Linux x86-64 Modal): caches hidden sums
//     per network and runs sparse delta inference from the pre-roll board.
// An outer base survives recursive children because their depths differ.
std::atomic<uint64_t> next_dice_node{1};

#if defined(BGBOT_USE_ACCELERATE)
struct DiceEncodedBase {
    std::array<float, EXTENDED_CONTACT_INPUTS> inputs{};
    uint64_t node = 0;
};
#else
struct DiceNetworkBase {
    std::vector<float> hidden, inputs;
    uint64_t node = 0;
};
#endif

template <class S>
void eval_groups_pair(const S& s, const std::vector<Board>& cands,
                      const int* idxs, int n, const CubeSearchNode& node,
                      std::array<float, NUM_OUTPUTS>* out, int pinned = -1)
{
    thread_local std::vector<int> nn_idx_of;
    thread_local std::vector<char> done;

    nn_idx_of.resize(n);
    done.assign(n, 0);
    for (int j = 0; j < n; ++j) {
        nn_idx_of[j] = (pinned >= 0 && !is_race(cands[idxs[j]]))
            ? pinned : s.nn_index_for(cands[idxs[j]]);
        if (nn_idx_of[j] >= BLENDED_SENTINEL_BASE) {
            // Blended backgame sentinel (21-NN hybrid): no shared base,
            // evaluate directly via the strategy's blend-aware probs.
            out[idxs[j]] = s.probs_with_nn(cands[idxs[j]], nn_idx_of[j]);
            done[j] = 1;
        }
    }

#if defined(BGBOT_USE_ACCELERATE)
    thread_local std::vector<int> members;
    thread_local std::vector<float> encoded;
    thread_local std::vector<std::array<float, NUM_OUTPUTS>> group_out;
    thread_local std::vector<std::array<DiceEncodedBase, 2>> base_by_depth;
    if (static_cast<int>(base_by_depth.size()) <= node.depth)
        base_by_depth.resize(node.depth + 1);

    for (int j = 0; j < n; ++j) {
        if (done[j]) continue;
        const int nidx = nn_idx_of[j];

        members.clear();
        for (int k = j; k < n; ++k) {
            if (done[k] || nn_idx_of[k] != nidx) continue;
            members.push_back(idxs[k]);
            done[k] = 1;
        }
        const int m = static_cast<int>(members.size());
        const NeuralNetwork& nn = *s.nn(nidx);
        const bool is_pr = (nidx == 0);
        const int ni = nn.n_inputs();
        const int slot_width = is_pr ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS;
        const bool standard_width = !std::is_same_v<S, GamePlanPairStrategy> || ni == slot_width;

        encoded.resize(static_cast<size_t>(m) * ni);
        if (standard_width) {
            auto& base_slot = base_by_depth[node.depth][is_pr ? 1 : 0];
            if (base_slot.node != node.id) {
                if (is_pr) {
                    auto inp = compute_tesauro_inputs(node.board);
                    std::copy_n(inp.data(), TESAURO_INPUTS, base_slot.inputs.data());
                } else {
                    auto inp = compute_extended_contact_inputs(node.board);
                    std::copy_n(inp.data(), EXTENDED_CONTACT_INPUTS, base_slot.inputs.data());
                }
                base_slot.node = node.id;
            }
            for (int g = 0; g < m; ++g) {
                encode_board_from_base(cands[members[g]], node.board,
                                       base_slot.inputs.data(), is_pr,
                                       &encoded[static_cast<size_t>(g) * ni]);
            }
        } else {
            for (int g = 0; g < m; ++g) {
                float* row = &encoded[static_cast<size_t>(g) * ni];
                if (is_pr) {
                    auto inp = compute_tesauro_inputs(cands[members[g]]);
                    std::copy_n(inp.data(), ni, row);
                } else {
                    auto inp = compute_extended_contact_inputs(cands[members[g]]);
                    std::copy_n(inp.data(), ni, row);
                }
            }
        }
        group_out.resize(m);
        nn.forward_group(encoded.data(), m, group_out.data());
        for (int g = 0; g < m; ++g) out[members[g]] = group_out[g];
    }
#else
    thread_local std::vector<std::vector<DiceNetworkBase>> by_depth;
    if (static_cast<int>(by_depth.size()) <= node.depth)
        by_depth.resize(node.depth + 1);

    for (int j = 0; j < n; ++j) {
        if (done[j]) continue;
        const int nidx = nn_idx_of[j];
        const NeuralNetwork& nn = *s.nn(nidx);
        const int ni = nn.n_inputs();
        const int nh = nn.n_hidden();
        auto& nets = by_depth[node.depth];
        if (static_cast<int>(nets.size()) <= nidx) nets.resize(nidx + 1);
        auto& slot = nets[nidx];
        auto& saved_base = slot.hidden;
        auto& saved_inputs = slot.inputs;
        const bool is_pr = (nidx == 0);
        // Only the constructor taking prebuilt pair nets can supply a network
        // that consumes a prefix of its slot's standard encoding.
        const int slot_width = is_pr ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS;
        const bool standard_width = !std::is_same_v<S, GamePlanPairStrategy> || ni == slot_width;

        const Board& base = node.board;
        if (slot.node != node.id) {
            saved_base.resize(nh);
            saved_inputs.resize(ni);
            const auto save = [&](const auto& inputs) {
                nn.save_hidden_base(inputs.data(), saved_base.data(), saved_inputs.data());
            };
            if (is_pr) save(compute_tesauro_inputs(base));
            else save(compute_extended_contact_inputs(base));
            slot.node = node.id;
        }
        for (int k = j; k < n; ++k) {
            if (done[k] || nn_idx_of[k] != nidx) continue;
            if (standard_width) {
                out[idxs[k]] = nn.forward_board_from_base(
                    cands[idxs[k]], base, saved_base.data(), saved_inputs.data());
            } else {
                const auto from_prefix = [&](const auto& inputs) {
                    return nn.forward_from_base(inputs.data(), saved_base.data(), saved_inputs.data());
                };
                out[idxs[k]] = is_pr ? from_prefix(compute_tesauro_inputs(cands[idxs[k]]))
                                     : from_prefix(compute_extended_contact_inputs(cands[idxs[k]]));
            }
            done[k] = 1;
        }
    }
#endif
}

void eval_each_impl(const Strategy& strat, const std::vector<Board>& cands,
                    const int* idxs, int n, const CubeSearchNode& node,
                    std::array<float, NUM_OUTPUTS>* out, int pinned = -1)
{
    if (n <= 0) return;

    // Frame-local storage avoids thread_local aliasing across nested
    // BearoffStrategy wrappers and re-entrant fallback calls, with zero heap
    // allocation for typical roll sizes (<= 64 candidates).
    constexpr int kStackRestCap = 64;
    int stack_rest[kStackRestCap];
    std::vector<int> heap_rest;

    const Strategy* model = &strat;
    for (auto layer = model->bearoff_layer(); layer.base; layer = model->bearoff_layer()) {
        const BearoffDB& db = *layer.db;
        int* dst = (n <= kStackRestCap) ? stack_rest : (heap_rest.resize(n), heap_rest.data());
        int remaining = 0;
        for (int j = 0; j < n; ++j) {
            const int idx = idxs[j];
            const Board& c = cands[idx];
            if (db.is_bearoff(c)) {
                out[idx] = db.lookup_probs(c, /*post_move=*/true);
            } else {
                dst[remaining++] = idx;
            }
        }
        if (remaining == 0) return;
        // An explicitly nested wrapper compacts surviving indices in place or
        // into stack_rest without reallocating the active input buffer.
        idxs = dst;
        n = remaining;
        model = layer.base;
    }

    if (model->cube_candidate_probs(cands, idxs, n, node, out, pinned)) return;

    // Fallback: per-candidate loop with the interior-pick convention.
    for (int j = 0; j < n; ++j) {
        const Board& c = cands[idxs[j]];
        out[idxs[j]] = model->evaluate_probs(c, is_race(c));
    }
}

// ======================== Core recursion ========================

struct CandidateBatchCache;

// Different legal checker moves can reach the same board two turns later.
// Only the 2-ply layer of one 4-ply checker call uses this table. Every
// endpoint starts with the same cube, model, filters and pre-move root pin.
class CheckerSubtrees {
    struct Entry {
        Board board;
        uint64_t cube_fp;
        std::array<float, 4> equities;
        std::array<float, NUM_OUTPUTS> probs;
    };
    struct Shard {
        std::mutex mutex;
        std::unordered_map<uint64_t, Entry> nodes;
    };
    std::array<Shard, 64> shards_;

public:
    bool get(uint64_t key, uint64_t fp, const Board& board, float* equities,
             std::array<float, NUM_OUTPUTS>* probs) {
        auto& shard = shards_[key % shards_.size()];
        std::lock_guard lock(shard.mutex);
        auto it = shard.nodes.find(key);
        if (it == shard.nodes.end() || it->second.cube_fp != fp || it->second.board != board)
            return false;
        std::copy(it->second.equities.begin(), it->second.equities.end(), equities);
        if (probs) *probs = it->second.probs;
        return true;
    }

    void put(uint64_t key, uint64_t fp, const Board& board, const float* equities,
             const std::array<float, NUM_OUTPUTS>& probs) {
        auto& shard = shards_[key % shards_.size()];
        std::lock_guard lock(shard.mutex);
        if (shard.nodes.size() < 128) {
            shard.nodes.try_emplace(key, Entry{board, fp,
                {equities[0], equities[1], equities[2], equities[3]}, probs});
        }
    }
};

struct EvalCtx {
    const Strategy* strategy;
    const Strategy* move_filter;
    int top_plies;       // plies at the entry node — enables the aggressive
                         // PubEval pre-filter at strictly deeper nodes only
    int n_threads = 1;   // top-level roll parallelism (1 = serial; rollout
                         // trial-internal calls are always serial)
    bool deep_prefilter = false;  // aggressive PubEval prune at deep nodes
                                  // (rollout-internal evaluations only)
    CandidateBatchCache* batches = nullptr;
    int pinned_nn = -1;   // root routing: the net every node uses (-1 = per node)
    CheckerSubtrees* checker = nullptr;
    const std::function<void()>* check_budget = nullptr;
    EvalOptions options;
};

// Compute leaf values from precomputed post-move probs of the previous
// mover's chosen board. `board` is the leaf pre-roll position (leaf mover's
// POV); post_probs == strategy.evaluate_probs(flip(board), ...), i.e. the NN
// output for the previous mover's post-move board � the same NN evaluation
// the parent's pick batch already performed, so plies==2 subtrees evaluate
// their leaves for free.
void leaf_from_post_probs(
    const Board& board,
    const CubeInfo aci_in[], int cci, bool fTop,
    const std::array<float, NUM_OUTPUTS>& post_probs,
    float arCubeful[],
    std::array<float, NUM_OUTPUTS>* arProbsOut)
{
    bool race = is_race(board);
    auto pre_roll_probs = invert_probs(post_probs);
    clamp_probs_to_board(pre_roll_probs, board);
    float default_x = resolve_cube_x_local(
        pre_roll_probs,
        (cci > 0 && aci_in[0].cube_value > 0) ? aci_in[0] : CubeInfo{},
        board, race);
    auto [leaf_pp, leaf_op] = pip_counts(board);

    CubeInfo aci[EVAL_MAX_CCI * 2];
    make_cube_pos_local(aci_in, cci, fTop, aci, false);

    float arCf[EVAL_MAX_CCI * 2];
    for (int i = 0; i < 2 * cci; i++) {
        if (aci[i].cube_value <= 0) {
            arCf[i] = 0.0f;
            continue;
        }
        if (cube_is_dead(aci[i])) {
            arCf[i] = cubeless_value(pre_roll_probs, aci[i]);
            continue;
        }
        float x = (aci[i].cube_x_override >= 0.0f) ? aci[i].cube_x_override
                   : (!race && aci[i].contact_x >= 0.0f)
                       ? contact_cube_x(aci[i], leaf_pp, leaf_op)
                   : default_x;
        if (aci[i].is_money()) {
            arCf[i] = cl2cf_money(pre_roll_probs, aci[i].owner, x,
                                   aci[i].jacoby_active());
        } else {
            arCf[i] = cl2cf_match(pre_roll_probs, aci[i], x);
        }
    }

    get_ecf3_local(arCubeful, cci, arCf, aci);
    if (arProbsOut) *arProbsOut = pre_roll_probs;
}

// Terminal accumulation for a chosen move that ends the game. `tp` is the
// terminal probs in opp-of-mover POV (already inverted). Writes weighted
// values into arCfLocal[0..expanded_cci-1].
void accumulate_terminal(
    const std::array<float, NUM_OUTPUTS>& tp,
    const CubeInfo aci[], int expanded_cci, int weight,
    float arCfLocal[])
{
    for (int i = 0; i < expanded_cci; i++) {
        if (aci[i].cube_value <= 0) {
            arCfLocal[i] = 0.0f;
            continue;
        }
        if (cube_is_dead(aci[i])) {
            arCfLocal[i] = weight * cubeless_value(tp, aci[i]);
            continue;
        }
        if (aci[i].is_money()) {
            if (aci[i].jacoby_active()) {
                arCfLocal[i] = weight * (2.0f * tp[0] - 1.0f);
            } else {
                arCfLocal[i] = weight * cubeless_equity(tp);
            }
        } else {
            arCfLocal[i] = weight * cubeless_mwc(tp,
                aci[i].match.away1, aci[i].match.away2,
                aci[i].cube_value, aci[i].match.is_crawford);
        }
    }
}

// What one tree node did on each of its 21 rolls, for
// cube_decision_nply_with_details: the board each roll chose and the value
// and probs its child returned, plus the node's values before the cube
// collapse. Filled only at the captured top levels; capturing bypasses the
// cache lookups of those nodes (a cached node has no per-roll record) but
// changes no arithmetic, so the values equal an uncaptured evaluation.
struct NodeCapture;
struct RollCapture {
    Board chosen{};                          // post-move board, mover's POV
    bool terminal = false;                   // the chosen move ends the game
    float cf[EVAL_MAX_CCI * 2] = {};         // child value per expanded cube state, unweighted
    std::array<float, NUM_OUTPUTS> probs{};  // child's pre-roll probs (the next mover's POV)
    NodeCapture* child = nullptr;            // when set, the child node is captured too
};
struct NodeCapture {
    std::array<RollCapture, 21> rolls;
    int expanded_cci = 0;
    float arCf[EVAL_MAX_CCI * 2] = {};       // after the perspective flip, before the collapse
    CubeInfo aci[EVAL_MAX_CCI * 2];          // expanded cube states, this node's perspective
};

struct DiceRoll21 { int d1, d2, weight; };
constexpr std::array<DiceRoll21, 21> ALL_ROLLS_21 = {{
    {1,1,1}, {2,2,1}, {3,3,1}, {4,4,1}, {5,5,1}, {6,6,1},
    {1,2,2}, {1,3,2}, {1,4,2}, {1,5,2}, {1,6,2},
    {2,3,2}, {2,4,2}, {2,5,2}, {2,6,2},
    {3,4,2}, {3,5,2}, {3,6,2},
    {4,5,2}, {4,6,2},
    {5,6,2}
}};

// Candidate probabilities shared by the live and dead searches in a cube
// call. Each path has a single writer, and the live search joins before the
// dead search reads. A skipped or divergent path falls back to evaluation.
struct CandidateBatchCache {
    static constexpr int ROOT = 0;
    static constexpr int MAX_CANDIDATES = 16;
    using Probs = std::array<float, NUM_OUTPUTS>;

    struct Roll {
        std::array<Probs, MAX_CANDIDATES> probs;
        std::array<int, MAX_CANDIDATES> original;
        int count = 0;
        int total = 0;
        bool subset = false;
    };
    struct Node {
        std::array<Roll, 21> rolls;
        Board board;
        const Strategy* model = nullptr;
        int pin = -1;
    };
    const bool share_survivors;
    explicit CandidateBatchCache(int plies, bool share_survivors)
        : share_survivors(share_survivors),
          node_count(plies == 4 ? 1 + 21 + 21 * 21 : (plies == 3 ? 1 + 21 : 1)) {}
    const int node_count;
    std::vector<Node> nodes;
    bool reading = false;

    Node* match_node(int path, const Board& board, const EvalCtx& ctx) {
        // A live root already in the tree cache needs no candidate storage.
        // Allocate on the calling thread before any parallel roll dispatch.
        if (!reading && path == ROOT) nodes.resize(node_count);
        if (nodes.empty()) return nullptr;
        Node& node = nodes[path];
        if (reading) {
            if (node.model != ctx.strategy || node.pin != ctx.pinned_nn || node.board != board)
                return nullptr;
        } else {
            node.board = board;
            node.pin = ctx.pinned_nn;
            node.model = ctx.strategy;
        }
        return &node;
    }
};

template<bool ShareBatches>
void cubeful_eval_recursive(
    const Board& board,
    const CubeInfo aciCubePos[],
    int cci,
    const EvalCtx& ctx,
    int plies,
    bool fTop,
    float arCubeful[],
    std::array<float, NUM_OUTPUTS>* arProbsOut,
    bool allow_parallel = false,
    NodeCapture* capture = nullptr,
    int roll_path = CandidateBatchCache::ROOT)
{
    bool is_money = (cci > 0 && aciCubePos[0].cube_value > 0)
                    ? aciCubePos[0].is_money() : true;

    // Cache lookup — internal nodes only (entries always carry probs).
    // Leaf/terminal nodes are cheap in this engine (leaf NN evals are reused
    // from the parent's pick batch), so caching them would cost more in key
    // hashing than it saves.
    uint64_t fp = 0, key = 0;
    const bool can_cache = (plies >= 2 && cci > 0 && cci <= 16);
    if (can_cache) {
        // Seed with the evaluator identity: the per-thread cache (and the
        // SharedPosCache keys derived from it) is shared across all engine
        // callers on a thread, so entries from different models must never
        // collide. The identity is a process-unique monotonic id (never an
        // address — freed strategies' addresses get reused). A
        // BearoffStrategy mixes its database's data identity into the
        // base's, so evaluations with and without the database never serve
        // each other, while nested wrappers over the same database share
        // one identity and its entries. The deep-prefilter context is folded
        // in too — the same node evaluated with and without the deep prune
        // has different values, so the two contexts must never serve each
        // other's entries.
        fp = 0xcbf29ce484222325ULL
             ^ (ctx.strategy->eval_identity() * 0x9e3779b97f4a7c15ULL)
             ^ (ctx.deep_prefilter ? 0x6a09e667f3bcc909ULL : 0)
             // Root routing changes every value below the root, so a node
             // pinned to one net must never serve an unpinned (or
             // differently pinned) evaluation of the same board.
             ^ (ctx.pinned_nn >= 0
                    ? (static_cast<uint64_t>(ctx.pinned_nn) + 1) * 0xbb67ae8584caa73bULL
                    : 0);
        // A persistent evaluator can see the same node both with and without an
        // entry move filter. Deep pruning also changes when a node at the same
        // remaining ply is visited as the root.
        fp = fp_mix(fp, ctx.move_filter ? ctx.move_filter->eval_identity() : 0);
        if (ctx.deep_prefilter) {
            fp = fp_mix(fp, plies < ctx.top_plies);
            fp = fp_mix(fp, ctx.options.deep_prefilter_threshold);
            fp = fp_mix(fp, ctx.options.deep_prefilter_keep);
        }
        for (int i = 0; i < cci; ++i) fp = cube_fp_one(fp, aciCubePos[i]);
        key = make_cache_key(board, fp, plies, cci, fTop);
        if (!capture && g_cubeful_eval_cache.get(key, fp, board, plies, cci, fTop,
                                                 arCubeful, arProbsOut)) {
            return;
        }
        if (!capture && ctx.checker && plies == 2 && cci == 4 &&
            ctx.checker->get(key, fp, board, arCubeful, arProbsOut)) {
            return;
        }
    }

    // Terminal check
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        auto t_probs = terminal_probs(result);
        for (int ici = 0; ici < cci; ici++) {
            if (aciCubePos[ici].cube_value <= 0) {
                arCubeful[ici] = 0.0f;
                continue;
            }
            if (cube_is_dead(aciCubePos[ici])) {
                arCubeful[ici] = cubeless_value(t_probs, aciCubePos[ici]);
                continue;
            }
            if (aciCubePos[ici].is_money()) {
                if (aciCubePos[ici].jacoby_active()) {
                    arCubeful[ici] = 2.0f * t_probs[0] - 1.0f;
                } else {
                    arCubeful[ici] = cubeless_equity(t_probs);
                }
            } else {
                arCubeful[ici] = cubeless_mwc(t_probs,
                    aciCubePos[ici].match.away1, aciCubePos[ici].match.away2,
                    aciCubePos[ici].cube_value, aciCubePos[ici].match.is_crawford);
            }
        }
        if (arProbsOut) *arProbsOut = t_probs;
        if (can_cache) {
            g_cubeful_eval_cache.put(key, fp, board, plies, cci, fTop, arCubeful, t_probs);
        }
        return;
    }

    // Leaf node (plies <= 1): single NN evaluation + Janowski.
    if (plies <= 1) {
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_probs = ctx.strategy->evaluate_pinned_probs(flipped, race, ctx.pinned_nn);
        std::array<float, NUM_OUTPUTS> pre_out{};
        leaf_from_post_probs(board, aciCubePos, cci, fTop, post_probs,
                             arCubeful, &pre_out);
        if (arProbsOut) *arProbsOut = pre_out;
        if (can_cache) {
            g_cubeful_eval_cache.put(key, fp, board, plies, cci, fTop, arCubeful, pre_out);
        }
        return;
    }

    // Dead-cube (cubeless) node sharing across rollout trial threads.
    // When every valid cube state at this node is dead, the node's equities
    // are fully determined by its cubeless probs (arCubeful[i] ==
    // cubeless_equity(probs) for valid slots, 0 for invalidated slots), so
    // the node is shared through the rollout's cross-thread SharedPosCache —
    // the same structure the cubeless N-ply recursion uses. The key is
    // salted so tree-produced entries can never be confused with entries
    // produced by MultiPlyStrategy::evaluate_probs_nply (same cache,
    // different evaluator => not bit-identical values).
    bool all_dead = can_cache && !fTop && !capture;
    if (all_dead) {
        for (int i = 0; i < cci; ++i) {
            if (aciCubePos[i].cube_value <= 0) continue;   // invalidated slot
            if (!cube_is_dead(aciCubePos[i])) { all_dead = false; break; }
        }
    }
    SharedPosCache* shared_cache =
        all_dead ? MultiPlyStrategy::get_shared_cache() : nullptr;
    SharedPosCache::Entry* shared_reservation = nullptr;
    if (shared_cache) {
        const uint64_t shared_key = key ^ 0x9e3779b97f4a7c15ULL;
        auto res = shared_cache->lookup_or_reserve(shared_key, 0, board);
        if (res.probs) {
            const auto p = *res.probs;
            for (int i = 0; i < cci; ++i) {
                arCubeful[i] = (aciCubePos[i].cube_value <= 0)
                    ? 0.0f : cubeless_value(p, aciCubePos[i]);
            }
            if (arProbsOut) *arProbsOut = p;
            g_cubeful_eval_cache.put(key, fp, board, plies, cci, fTop,
                                     arCubeful, p);
            return;
        }
        shared_reservation = res.reservation;
    }

    // Internal node: expand cube states, iterate 21 rolls.
    CubeInfo aci[EVAL_MAX_CCI * 2];
    make_cube_pos_local(aciCubePos, cci, fTop, aci, true);
    const int expanded_cci = 2 * cci;

    float arCf[EVAL_MAX_CCI * 2] = {};
    std::array<float, NUM_OUTPUTS> probsAccum = {};

    static constexpr int MOVE_FILTER_THRESHOLD = 16;
    static constexpr int MOVE_FILTER_KEEP = 15;
    // Deep-node pre-filter (PubEval): nodes strictly below the entry node's
    // ply are averaged over many rolls/trials, so pruning the PubEval-worst
    // candidates costs little accuracy while removing most of the per-node
    // eval work on big candidate sets (doubles rolls generate 30-90 moves).
    // keep=14 prunes only the deep tail, which is almost never the NN-best
    // candidate; tighter settings produce a measurable systematic negative
    // equity bias (weaker simulated play shrinks equities toward zero).
    // Race nodes skip the aggressive filter: their candidate evaluations hit
    // the bearoff DB / cheap race NN, so PubEval scoring would cost more than
    // it saves.
    const bool deep_node = ctx.deep_prefilter && (plies < ctx.top_plies)
                           && !is_race(board);
    const Strategy* prefilter = ctx.move_filter;
    int pf_threshold = MOVE_FILTER_THRESHOLD;
    int pf_keep = MOVE_FILTER_KEEP;
    if (deep_node) {
        if (!prefilter) prefilter = &static_pubeval();
        pf_threshold = ctx.options.deep_prefilter_threshold;
        pf_keep = ctx.options.deep_prefilter_keep;
    }

    // Per-roll evaluation. Writes the WEIGHTED contribution of roll `r` into
    // rollCf[0..expanded_cci-1] and rollProbs (so serial and parallel
    // execution accumulate the identical floating-point sequence below).
    //
    // Child-level parallelism only pays when there are more threads than the
    // 21 top-level roll tasks; below that it just stacks BLOCKING
    // parallel_for waits inside pool workers (a nested caller sleeps until
    // its queued chunks run), which can exhaust the shared pool and
    // deadlock. Values are thread-count independent (fixed accumulation
    // order), so this is purely a scheduling decision.
    const bool child_parallel = allow_parallel && (plies - 1 > 2)
                                && ctx.n_threads > 21;
    CandidateBatchCache::Node* batch_node = nullptr;
    if constexpr (ShareBatches)
        batch_node = ctx.batches->match_node(roll_path, board, ctx);

    // Each worker can reuse its pre-move base across all 21 rolls,
    // including the shortlisted moves of a rollout-internal search.
    const CubeSearchNode dice_node{board, plies, next_dice_node.fetch_add(1, std::memory_order_relaxed)};
    auto eval_roll = [&](int r, float* rollCf,
                         std::array<float, NUM_OUTPUTS>& rollProbs) {
        thread_local std::vector<Board> candidates;
        thread_local std::vector<Board> filtered;
        thread_local std::vector<std::array<float, NUM_OUTPUTS>> cand_probs;
        thread_local std::vector<int> live_idx;

        if (ctx.check_budget) (*ctx.check_budget)();
        const auto& roll = ALL_ROLLS_21[r];
        const int child_path = ShareBatches ? 21 * roll_path + r + 1 : 0;
        const float w = static_cast<float>(roll.weight);
        float arCfLocal[EVAL_MAX_CCI * 2];
        RollCapture* rc = capture ? &capture->rolls[r] : nullptr;
        NodeCapture* child_capture = rc ? rc->child : nullptr;

        candidates.clear();
        if (candidates.capacity() < 32) candidates.reserve(32);
        possible_boards_unsorted(board, roll.d1, roll.d2, candidates);
        const int n_cand = static_cast<int>(candidates.size());

        if (n_cand == 0) {
            // Standing pat: flip board, recurse / leaf.
            Board opp_board = flip(board);
            std::array<float, NUM_OUTPUTS> probsTmp{};
            if (plies - 1 <= 1) {
                // Leaf on opp_board. Terminal impossible (board not terminal,
                // no move played). NN eval of flip(opp_board) == board.
                bool lrace = is_race(opp_board);
                auto post = ctx.strategy->evaluate_pinned_probs(flip(opp_board), lrace, ctx.pinned_nn);
                leaf_from_post_probs(opp_board, aci, expanded_cci, false, post,
                                     arCfLocal, &probsTmp);
            } else {
                cubeful_eval_recursive<ShareBatches>(opp_board, aci, expanded_cci, ctx, plies - 1,
                               false, arCfLocal, &probsTmp, child_parallel,
                               child_capture, child_path);
            }
            for (int i = 0; i < expanded_cci; i++) rollCf[i] = w * arCfLocal[i];
            for (int k = 0; k < NUM_OUTPUTS; k++)
                rollProbs[k] = w * probsTmp[k];
            if (rc) {
                rc->chosen = board;
                std::copy(arCfLocal, arCfLocal + expanded_cci, rc->cf);
                rc->probs = probsTmp;
            }
            return;
        }

        // Determine eval candidate set (cheap pre-filter for big sets).
        const std::vector<Board>* eval_candidates = &candidates;
        thread_local std::vector<std::pair<double, int>> filter_scores;
        if (prefilter && n_cand > pf_threshold) {
            filter_scores.clear();
            filter_scores.reserve(n_cand);
            bool pre_move_race = is_race(board);
            for (int c = 0; c < n_cand; c++) {
                GameResult gr = check_game_over(candidates[c]);
                double eq = (gr != GameResult::NOT_OVER)
                    ? 1e30
                    : prefilter->evaluate(candidates[c], pre_move_race);
                filter_scores.push_back({-eq, c});
            }
            int keep = std::min(pf_keep, n_cand);
            std::partial_sort(filter_scores.begin(), filter_scores.begin() + keep,
                              filter_scores.end());
            filtered.clear();
            filtered.reserve(keep);
            for (int k = 0; k < keep; k++) {
                filtered.push_back(candidates[filter_scores[k].second]);
            }
            eval_candidates = &filtered;
        }

        const int n_eval = static_cast<int>(eval_candidates->size());
        Board chosen;
        std::array<float, NUM_OUTPUTS> chosen_post_probs{};
        bool chosen_terminal = false;

        if (n_eval == 1) {
            chosen = (*eval_candidates)[0];
            chosen_terminal = (check_game_over(chosen) != GameResult::NOT_OVER);
            // chosen_post_probs unused on this path unless plies-1 <= 1 and
            // non-terminal — compute it lazily below.
        } else {
            // Batched per-candidate probs (terminals get terminal_probs;
            // others per-candidate NN classification — pick convention).
            cand_probs.resize(n_eval);
            using SavedRoll = CandidateBatchCache::Roll;
            auto evaluate_batch = [&](const SavedRoll* partial = nullptr) {
                int reused = 0;
                live_idx.clear();
                live_idx.reserve(n_eval);
                for (int c = 0; c < n_eval; ++c) {
                    if constexpr (ShareBatches) {
                        if (partial && reused < partial->count && partial->original[reused] == c) {
                            cand_probs[c] = partial->probs[reused++];
                            continue;
                        }
                    }
                    GameResult gr = check_game_over((*eval_candidates)[c]);
                    if (gr != GameResult::NOT_OVER) {
                        cand_probs[c] = terminal_probs(gr);
                    } else {
                        live_idx.push_back(c);
                    }
                }
                eval_each_impl(*ctx.strategy, *eval_candidates,
                               live_idx.data(), static_cast<int>(live_idx.size()), dice_node,
                               cand_probs.data(), ctx.pinned_nn);
            };
            if constexpr (ShareBatches) {
                auto* saved = batch_node ? &batch_node->rolls[r] : nullptr;
                const bool full = eval_candidates == &candidates;
                const bool hit = saved && ctx.batches->reading && full && saved->total == n_cand;
                if (hit && !saved->subset && saved->count == n_eval) {
                    std::copy_n(saved->probs.begin(), n_eval, cand_probs.begin());
                } else {
                    evaluate_batch(hit && saved->subset ? saved : nullptr);
                    // A first-sibling delta or a custom evaluator may depend on
                    // the ordered group. Only a built-in pair with a node base
                    // shares surviving indices between differently ordered lists.
                    if (saved && !ctx.batches->reading && n_eval <= CandidateBatchCache::MAX_CANDIDATES &&
                        (full || ctx.batches->share_survivors)) {
                        saved->count = n_eval;
                        saved->total = n_cand;
                        saved->subset = !full;
                        if (full) {
                            std::copy_n(cand_probs.begin(), n_eval, saved->probs.begin());
                        } else {
                            // Record survivors in the generator's order, independently of
                            // PubEval rank. The reader scans them alongside its full list.
                            std::array<int, CandidateBatchCache::MAX_CANDIDATES> ranks;
                            for (int k = 0; k < n_eval; ++k) ranks[k] = k;
                            std::sort(ranks.begin(), ranks.begin() + n_eval,
                                [&](int a, int b) { return filter_scores[a].second < filter_scores[b].second; });
                            for (int k = 0; k < n_eval; ++k) {
                                saved->original[k] = filter_scores[ranks[k]].second;
                                saved->probs[k] = cand_probs[ranks[k]];
                            }
                        }
                    }
                }
            } else {
                evaluate_batch();
            }

            // Cubeful pick vs the primary cube state: every interior pick
            // is match/cube-aware without forking the recursion per cube.
            // Ties on the cubeful value are broken by money cubeless equity
            // (pick_best_with_tiebreak in cube.h): at scores where a gammon
            // and a backgammon lose the match alike, EVERY move can score
            // identically, and an arbitrary pick here leaves checkers in the
            // opponent's home board and invents backgammon losses at the leaf.
            thread_local std::vector<float> cf_vals;
            cf_vals.resize(n_eval);
            for (int c = 0; c < n_eval; ++c) {
                const Board& cand = (*eval_candidates)[c];
                bool crace = is_race(cand);
                auto [pp, op] = pip_counts(cand);
                float cube_x = cube_efficiency(cand_probs[c], crace, pp, op);
                cf_vals[c] = cl2cf(cand_probs[c], aciCubePos[0], cube_x);
            }
            const int local_best =
                pick_best_with_tiebreak(cf_vals.data(), cand_probs.data(), n_eval);
            chosen = (*eval_candidates)[local_best];
            chosen_post_probs = cand_probs[local_best];
            chosen_terminal = (check_game_over(chosen) != GameResult::NOT_OVER);
        }

        if (chosen_terminal) {
            auto tp = invert_probs(terminal_probs(check_game_over(chosen)));
            accumulate_terminal(tp, aci, expanded_cci, roll.weight, rollCf);
            for (int k = 0; k < NUM_OUTPUTS; k++)
                rollProbs[k] = w * tp[k];
            if (rc) {
                rc->chosen = chosen;
                rc->terminal = true;
                accumulate_terminal(tp, aci, expanded_cci, 1, rc->cf);
                rc->probs = tp;
            }
            return;
        }

        Board opp_pre_roll = flip(chosen);
        std::array<float, NUM_OUTPUTS> probsTmp{};
        if (plies - 1 <= 1) {
            if (n_eval == 1) {
                // Forced move: no pick batch ran — evaluate the leaf NN now.
                bool lrace = is_race(opp_pre_roll);
                chosen_post_probs =
                    ctx.strategy->evaluate_pinned_probs(chosen, lrace, ctx.pinned_nn);
            }
            leaf_from_post_probs(opp_pre_roll, aci, expanded_cci, false,
                                 chosen_post_probs, arCfLocal, &probsTmp);
        } else {
            cubeful_eval_recursive<ShareBatches>(opp_pre_roll, aci, expanded_cci, ctx, plies - 1,
                           false, arCfLocal, &probsTmp, child_parallel,
                           child_capture, child_path);
        }
        for (int i = 0; i < expanded_cci; i++) rollCf[i] = w * arCfLocal[i];
        for (int k = 0; k < NUM_OUTPUTS; k++)
            rollProbs[k] = w * probsTmp[k];
        if (rc) {
            rc->chosen = chosen;
            std::copy(arCfLocal, arCfLocal + expanded_cci, rc->cf);
            rc->probs = probsTmp;
        }
    };

    // Roll work varies substantially. At ordinary parallel widths, workers
    // claim rolls starting with the expensive doubles and reduce in fixed
    // dice order. Keep the existing chunk dispatch for small or nested jobs.
    std::array<std::array<float, NUM_OUTPUTS>, 21> roll_probs{};
    float roll_cf[21][EVAL_MAX_CCI * 2];

    if (ctx.checker && allow_parallel && ctx.n_threads >= 2 && ctx.n_threads <= 21) {
        // Each checker endpoint owns this many lanes. Its caller may already
        // occupy a pool worker, so it must be able to finish its own rolls.
        multipy_cooperative_for(21, ctx.n_threads, [&](int r) {
            eval_roll(r, roll_cf[r], roll_probs[r]);
        });
    } else if (allow_parallel && ctx.n_threads >= 4 && ctx.n_threads <= 21) {
        std::atomic<int> next_roll{0};
        multipy_parallel_run(ctx.n_threads, [&] {
            for (int r = next_roll.fetch_add(1, std::memory_order_relaxed); r < 21;
                 r = next_roll.fetch_add(1, std::memory_order_relaxed)) {
                eval_roll(r, roll_cf[r], roll_probs[r]);
            }
        });
    } else if (allow_parallel && ctx.n_threads > 1) {
        multipy_parallel_for(21, std::min(ctx.n_threads, 21), [&](int r) {
            eval_roll(r, roll_cf[r], roll_probs[r]);
        });
    } else {
        for (int r = 0; r < 21; ++r) {
            eval_roll(r, roll_cf[r], roll_probs[r]);
        }
    }

    for (int r = 0; r < 21; ++r) {
        for (int i = 0; i < expanded_cci; i++) arCf[i] += roll_cf[r][i];
        for (int k = 0; k < NUM_OUTPUTS; k++) probsAccum[k] += roll_probs[r][k];
    }

    // Average over 36 and flip perspective back to current player.
    for (int i = 0; i < expanded_cci; i++) {
        if (is_money) {
            arCf[i] = -arCf[i] / 36.0f;
        } else {
            arCf[i] = 1.0f - arCf[i] / 36.0f;
        }
    }

    // Probs are accumulated unconditionally (cheap — leaves produce them as a
    // byproduct) so cache entries always carry valid probs regardless of
    // whether THIS caller requested them.
    std::array<float, NUM_OUTPUTS> probs_final{};
    for (int k = 0; k < NUM_OUTPUTS; k++)
        probsAccum[k] /= 36.0f;
    probs_final = invert_probs(probsAccum);
    if (arProbsOut) *arProbsOut = probs_final;

    // Un-invert cube states back to current player's perspective.
    for (int i = 0; i < expanded_cci; i++) {
        if (aci[i].cube_value > 0) {
            aci[i] = flip_cube_perspective(aci[i]);
        }
    }

    if (capture) {
        capture->expanded_cci = expanded_cci;
        std::copy(arCf, arCf + expanded_cci, capture->arCf);
        std::copy(aci, aci + expanded_cci, capture->aci);
    }

    get_ecf3_local(arCubeful, cci, arCf, aci);
    if (all_dead) {
        // Derive dead-node equities from the node probs so the value is
        // bit-identical whether computed locally or served from the shared
        // cache (the two accumulation orders differ in the last ulp).
        for (int i = 0; i < cci; ++i) {
            arCubeful[i] = (aciCubePos[i].cube_value <= 0)
                ? 0.0f : cubeless_value(probs_final, aciCubePos[i]);
        }
    }
    if (can_cache) {
        g_cubeful_eval_cache.put(key, fp, board, plies, cci, fTop, arCubeful, probs_final);
        if (ctx.checker && plies == 2 && cci == 4)
            ctx.checker->put(key, fp, board, arCubeful, probs_final);
    }
    if (shared_reservation) {
        shared_cache->publish(shared_reservation, probs_final);
    }
}

// 1-ply pre-roll cubeful evaluation: one NN evaluation shared across all
// cube states, then Janowski per state.
void eval_pre_roll_1ply_multi(
    const Board& board,
    const CubeInfo* cubes,
    int n_cubes,
    const Strategy& strategy,
    float* out_raw,                  // money equity / match MWC per cube
    std::array<float, NUM_OUTPUTS>* probs_out,
    int pinned = -1)
{
    Board flipped = flip(board);
    bool race = is_race(board);

    GameResult result = check_game_over(flipped);
    std::array<float, NUM_OUTPUTS> pre_roll_probs;
    bool terminal = (result != GameResult::NOT_OVER);
    if (terminal) {
        pre_roll_probs = invert_probs(terminal_probs(result));
    } else {
        auto post_probs = strategy.evaluate_pinned_probs(flipped, race, pinned);
        pre_roll_probs = invert_probs(post_probs);
        clamp_probs_to_board(pre_roll_probs, board);
    }

    for (int i = 0; i < n_cubes; ++i) {
        const CubeInfo& cube = cubes[i];
        if (terminal) {
            if (cube.is_money()) {
                out_raw[i] = cube.jacoby_active()
                    ? (2.0f * pre_roll_probs[0] - 1.0f)
                    : cubeless_equity(pre_roll_probs);
            } else {
                out_raw[i] = cubeless_mwc(pre_roll_probs,
                    cube.match.away1, cube.match.away2,
                    cube.cube_value, cube.match.is_crawford);
            }
            continue;
        }
        float x = resolve_cube_x_local(pre_roll_probs, cube, board, race);
        if (cube.is_money()) {
            out_raw[i] = cl2cf_money(pre_roll_probs, cube.owner, x,
                                      cube.jacoby_active());
        } else {
            out_raw[i] = cl2cf_match(pre_roll_probs, cube, x);
        }
    }
    if (probs_out) *probs_out = pre_roll_probs;
}

} // namespace

bool GamePlanPairStrategy::cube_candidate_probs(
    const std::vector<Board>& candidates, const int* indices, int count,
    const CubeSearchNode& node, std::array<float, NUM_OUTPUTS>* out, int pinned) const
{
    eval_groups_pair(*this, candidates, indices, count, node, out, pinned);
    return true;
}

bool BackgameAwarePairStrategy::cube_candidate_probs(
    const std::vector<Board>& candidates, const int* indices, int count,
    const CubeSearchNode& node, std::array<float, NUM_OUTPUTS>* out, int pinned) const
{
    eval_groups_pair(*this, candidates, indices, count, node, out, pinned);
    return true;
}

// ======================== Public API ========================

void clear_cubeful_eval_cache() {
    g_cubeful_eval_cache.clear();
}

uint64_t cube_state_fingerprint(const CubeInfo* cubes, int n) {
    uint64_t fp = 0xcbf29ce484222325ULL;
    for (int i = 0; i < n; ++i) fp = cube_fp_one(fp, cubes[i]);
    return fp;
}

// A shallow race has few replies to evaluate. Bound the cost of waking
// pool workers while leaving deeper searches and serial rollout trials alone.
static int root_roll_threads(const Board& board, int plies, int threads) {
#if defined(__linux__)
    constexpr int race_threads = 1;
    if (plies == 2 && threads > race_threads && is_race(board)) return race_threads;
#endif
    return threads;
}

template<bool ShareBatches>
static void cubeful_equity_nply_multi_impl(
    const Board& board,
    const CubeInfo* cubes,
    int n_cubes,
    const Strategy& strategy,
    int n_plies,
    float* out,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter,
    bool fTop,
    std::array<float, NUM_OUTPUTS>* probs_out,
    bool deep_prefilter,
    const Board* root_board,
    CandidateBatchCache* batches,
    CheckerSubtrees* checker = nullptr,
    const std::function<void()>& check_budget = {},
    EvalOptions options = {})
{
    (void)filter;  // candidate selection inside the tree is single-pick
    if (n_cubes <= 0) return;
    if (n_cubes > EVAL_MAX_CCI) n_cubes = EVAL_MAX_CCI;
    const int pinned = root_board ? strategy.root_pin_for(*root_board) : -1;

    if (n_plies <= 1) {
        float raw[EVAL_MAX_CCI];
        eval_pre_roll_1ply_multi(board, cubes, n_cubes, strategy, raw, probs_out, pinned);
        for (int i = 0; i < n_cubes; ++i) {
            if (cubes[i].is_money()) {
                out[i] = raw[i];
            } else {
                out[i] = mwc2eq(raw[i], cubes[i].match.away1, cubes[i].match.away2,
                                cubes[i].cube_value, cubes[i].match.is_crawford);
            }
        }
        return;
    }

    CubeInfo aciCubePos[EVAL_MAX_CCI];
    for (int i = 0; i < n_cubes; ++i) aciCubePos[i] = cubes[i];

    n_threads = root_roll_threads(board, n_plies, n_threads);
    EvalCtx ctx{&strategy, move_filter, n_plies, n_threads, deep_prefilter};
    ctx.options = options;
    ctx.pinned_nn = pinned;
    ctx.batches = batches;
    ctx.check_budget = check_budget ? &check_budget : nullptr;
    ctx.checker = checker;
    const bool allow_parallel = n_threads > 1 && n_plies >= 2;
    float arCubeful[EVAL_MAX_CCI];
    cubeful_eval_recursive<ShareBatches>(board, aciCubePos, n_cubes, ctx, n_plies, fTop,
                   arCubeful, probs_out, allow_parallel);

    for (int i = 0; i < n_cubes; ++i) {
        if (cubes[i].is_money()) {
            out[i] = arCubeful[i];
        } else {
            out[i] = mwc2eq(arCubeful[i], cubes[i].match.away1, cubes[i].match.away2,
                            cubes[i].cube_value, cubes[i].match.is_crawford);
        }
    }
}

void cubeful_equity_nply_multi(
    const Board& board,
    const CubeInfo* cubes,
    int n_cubes,
    const Strategy& strategy,
    int n_plies,
    float* out,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter,
    bool fTop,
    std::array<float, NUM_OUTPUTS>* probs_out,
    bool deep_prefilter,
    const Board* root_board,
    EvalOptions options)
{
    validate_eval_options(options);
    cubeful_equity_nply_multi_impl<false>(
        board, cubes, n_cubes, strategy, n_plies, out, filter, n_threads,
        move_filter, fTop, probs_out, deep_prefilter, root_board, nullptr, nullptr, {}, options);
}

std::vector<CubefulProbsAndEquity> cubeful_checker_endpoints_nply(
    const Board& before, const std::vector<Board>& endpoints, const CubeInfo& cube,
    const Strategy& strategy, int plies, int threads, const std::function<void()>& check_budget)
{
    const int count = static_cast<int>(endpoints.size());
    if (count <= 0) return {};
    const bool split_rolls = plies == 4 && count >= 2 && threads >= 8 && threads <= 21;
    // Long 4P endpoints can finish their 21 rolls sooner with at least two
    // lanes each. All concurrent endpoint groups share the same thread budget.
    const int workers = split_rolls ? std::min(count, threads / 2)
                                    : std::clamp(threads, 1, std::max(count, 1));
    const int roll_threads = split_rolls ? threads / workers
                                         : (count == 1 ? std::min(threads, 21) : 1);
    std::vector<CubefulProbsAndEquity> results(count);
    std::vector<std::exception_ptr> errors(count);
    std::unique_ptr<CheckerSubtrees> cache;
    if (workers > 1 && plies == 4) cache = std::make_unique<CheckerSubtrees>();
    std::atomic<int> next{0};
    multipy_parallel_run(workers, [&] {
        for (int i = next.fetch_add(1, std::memory_order_relaxed); i < count;
             i = next.fetch_add(1, std::memory_order_relaxed)) {
            try {
                if (check_budget) check_budget();
                auto& result = results[i];
                cubeful_equity_nply_multi_impl<false>(
                    flip(endpoints[i]), &cube, 1, strategy, plies, &result.equity,
                    MoveFilters::TINY, roll_threads, nullptr, false, &result.probs,
                    false, &before, nullptr, cache.get());
            } catch (...) {
                errors[i] = std::current_exception();
            }
        }
    });
    for (const auto& error : errors) {
        if (error) std::rethrow_exception(error);
    }
    return results;
}

// The cube decision from the collapsed root values of a cube's ND and DT
// states (money equity or match MWC, per the cube's own cube value).
static CubeDecision decision_from_root_values(const CubeInfo& cube, float nd_raw, float dt_raw)
{
    bool is_money = cube.is_money();

    CubeDecision result = {};

    if (is_money) {
        result.equity_nd = nd_raw;
        float actual_dt = 2.0f * dt_raw;
        result.equity_dp = 1.0f;

        if (cube.beaver && actual_dt < 0.0f) {
            result.equity_dt = 2.0f * actual_dt;
            result.is_beaver = true;
        } else {
            result.equity_dt = actual_dt;
        }
    } else {
        int away1 = cube.match.away1;
        int away2 = cube.match.away2;
        int cv = cube.cube_value;
        bool craw = cube.match.is_crawford;
        float dp_m = dp_mwc(away1, away2, cv, craw);
        result.equity_nd = mwc2eq(nd_raw, away1, away2, cv, craw);
        result.equity_dt = mwc2eq(dt_raw, away1, away2, cv, craw);
        result.equity_dp = mwc2eq(dp_m,   away1, away2, cv, craw);
    }

    bool player_can_double = can_double(cube);
    bool auto_double = (!is_money && !cube.match.is_crawford &&
                        player_can_double &&
                        cube.match.away1 > 1 && cube.match.away2 == 1);

    if (!player_can_double) {
        result.should_double = false;
        result.should_take = true;
        result.optimal_equity = result.equity_nd;
    } else if (is_money) {
        float best_double = std::min(result.equity_dt, result.equity_dp);
        result.should_double = (best_double > result.equity_nd);
        result.should_take = (result.equity_dt <= result.equity_dp);
        result.optimal_equity = result.should_double
            ? std::min(result.equity_dt, result.equity_dp)
            : result.equity_nd;
    } else if (auto_double) {
        float dt_m = dt_raw;
        float dp_m_val = dp_mwc(cube.match.away1, cube.match.away2,
                                cube.cube_value, cube.match.is_crawford);
        result.should_double = true;
        result.should_take = (dt_m <= dp_m_val);
        result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
    } else {
        float nd_m = nd_raw;
        float dt_m = dt_raw;
        float dp_m_val = dp_mwc(cube.match.away1, cube.match.away2,
                                cube.cube_value, cube.match.is_crawford);
        float best_double = std::min(dt_m, dp_m_val);
        result.should_double = (best_double > nd_m);
        result.should_take = (dt_m <= dp_m_val);
        result.optimal_equity = result.should_double
            ? std::min(result.equity_dt, result.equity_dp)
            : result.equity_nd;
    }

    return result;
}

template<bool ShareBatches>
static void cube_decision_nply_multi_impl(
    const Board& board,
    const CubeInfo* cubes,
    int n_cubes,
    const Strategy& strategy,
    int n_plies,
    CubeDecision* out,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter,
    bool deep_prefilter,
    CandidateBatchCache* batches, const std::function<void()>& check_budget = {},
    EvalOptions options = {})
{
    (void)filter;  // candidate selection inside the tree is single-pick
    if (n_cubes <= 0) return;
    if (n_cubes > EVAL_MAX_CCI / 2) n_cubes = EVAL_MAX_CCI / 2;
    const int pinned = strategy.root_pin_for(board);  // a cube decision's root is the position itself

    if (n_plies <= 1) {
        // 1-ply path: pre-roll probs ONCE (shared), then Janowski per cube.
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_probs = strategy.evaluate_pinned_probs(flipped, race, pinned);
        auto pre_roll_probs = invert_probs(post_probs);
        clamp_probs_to_board(pre_roll_probs, board);
        for (int i = 0; i < n_cubes; ++i) {
            float x = resolve_cube_x_local(pre_roll_probs, cubes[i], board, race);
            out[i] = cube_decision_1ply(pre_roll_probs, cubes[i], x);
        }
        return;
    }

    const int n_states = 2 * n_cubes;
    CubeInfo aciCubePos[EVAL_MAX_CCI];
    for (int i = 0; i < n_cubes; ++i) {
        aciCubePos[2 * i]     = cubes[i];
        aciCubePos[2 * i + 1] = cubes[i];
        aciCubePos[2 * i + 1].cube_value = 2 * cubes[i].cube_value;
        aciCubePos[2 * i + 1].owner      = CubeOwner::OPPONENT;
    }

    n_threads = root_roll_threads(board, n_plies, n_threads);
    EvalCtx ctx{&strategy, move_filter, n_plies, n_threads, deep_prefilter};
    ctx.options = options;
    ctx.pinned_nn = pinned;
    ctx.batches = batches;
    ctx.check_budget = check_budget ? &check_budget : nullptr;
    const bool allow_parallel = n_threads > 1 && n_plies >= 2;
    float arCubeful[EVAL_MAX_CCI];
    cubeful_eval_recursive<ShareBatches>(board, aciCubePos, n_states, ctx, n_plies, /*fTop=*/true,
                   arCubeful, nullptr, allow_parallel);

    // Unpack per-branch decisions.
    for (int i = 0; i < n_cubes; ++i) {
        out[i] = decision_from_root_values(cubes[i], arCubeful[2 * i],
                                           arCubeful[2 * i + 1]);
    }
}

void cube_decision_nply_multi(
    const Board& board,
    const CubeInfo* cubes,
    int n_cubes,
    const Strategy& strategy,
    int n_plies,
    CubeDecision* out,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter,
    bool deep_prefilter, const std::function<void()>& check_budget,
    EvalOptions options)
{
    validate_eval_options(options);
    cube_decision_nply_multi_impl<false>(
        board, cubes, n_cubes, strategy, n_plies, out, filter, n_threads,
        move_filter, deep_prefilter, nullptr, check_budget, options);
}

// ======================== Cube decision with per-roll details ========================

// Opponent-roll equities of one captured player roll, for one cube scenario
// (ND: states nd_idx/dt_idx = 0/1, DT: 4/5 of the opponent node). The
// opponent's own cube decision in that scenario picks which branch the
// per-roll values come from: D/P ends the game (no opponent rolls), D/T
// takes the redoubled branch, otherwise the branch as it stands. Values are
// scaled to per-initial-cube units, player's perspective.
static void opponent_roll_equities(
    PlayerRollDetail& detail, const NodeCapture& opp,
    int nd_idx, int dt_idx, int branch_cv, int redouble_cv, int initial_cv,
    int away1, int away2, bool is_crawford, bool is_money)
{
    int idx = nd_idx, cv = branch_cv;
    if (opp.aci[dt_idx].cube_value > 0) {
        const CubeInfo& nd = opp.aci[nd_idx];
        float rND = opp.arCf[nd_idx];
        float rDT = is_money ? 2.0f * opp.arCf[dt_idx] : opp.arCf[dt_idx];
        float rDP = is_money ? 1.0f
                             : dp_mwc(nd.match.away1, nd.match.away2,
                                      nd.cube_value, nd.match.is_crawford);
        if (is_money && nd.beaver && rDT < 0.0f) rDT = 2.0f * rDT;
        const bool doubles = (rDT >= rND && rDP >= rND);
        if (doubles && rDT > rDP) {
            detail.opponent_dp = true;
            detail.opponent_rolls.clear();
            return;
        }
        if (doubles) { idx = dt_idx; cv = redouble_cv; }
    }
    const float scale = static_cast<float>(cv) / static_cast<float>(initial_cv);
    for (int o = 0; o < 21; ++o) {
        const float raw = opp.rolls[o].cf[idx];
        detail.opponent_rolls[o].cubeful_equity = is_money
            ? scale * raw : mwc2eq(raw, away1, away2, cv, is_crawford);
    }
}

CubeDecision cube_decision_nply_with_details(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    TwoPlyDetails& details,
    const Strategy* move_filter, const std::function<void()>& check_budget)
{
    (void)filter;  // candidate selection inside the tree is single-pick
    if (n_plies < 2) {
        throw std::invalid_argument("cube_decision_nply_with_details requires n_plies >= 2");
    }

    // The tree cube_decision_nply_multi walks for one cube, with the player
    // rolls (and, from 3-ply, the opponent rolls below them) captured.
    CubeInfo aciCubePos[2] = {cube, cube};
    aciCubePos[1].cube_value = 2 * cube.cube_value;
    aciCubePos[1].owner = CubeOwner::OPPONENT;

    auto root = std::make_unique<NodeCapture>();
    std::vector<std::unique_ptr<NodeCapture>> children;
    if (n_plies >= 3) {
        for (int r = 0; r < 21; ++r) {
            children.push_back(std::make_unique<NodeCapture>());
            root->rolls[r].child = children.back().get();
        }
    }

    n_threads = root_roll_threads(board, n_plies, n_threads);
    EvalCtx ctx{&strategy, move_filter, n_plies, n_threads, /*deep_prefilter=*/false};
    ctx.check_budget = check_budget ? &check_budget : nullptr;
    ctx.pinned_nn = strategy.root_pin_for(board);
    const bool allow_parallel = (n_threads > 1 && n_plies >= 2);
    float arCubeful[EVAL_MAX_CCI];
    cubeful_eval_recursive<false>(board, aciCubePos, 2, ctx, n_plies, /*fTop=*/true,
                                  arCubeful, nullptr, allow_parallel, root.get());
    CubeDecision cd = decision_from_root_values(cube, arCubeful[0], arCubeful[1]);

    // Root states after the fTop expansion: 0 = ND, 2 = DT (1 and 3 unused).
    // A roll's child value in those slots is the opponent's collapsed value
    // (opponent's perspective), which already includes the opponent's own
    // cube decision.
    const bool is_money = cube.is_money();
    const int cv = cube.cube_value;
    const int a1 = cube.match.away1, a2 = cube.match.away2;
    const bool craw = cube.match.is_crawford;

    details.nd_player_rolls.assign(21, PlayerRollDetail{});
    details.dt_player_rolls.assign(21, PlayerRollDetail{});
    for (int r = 0; r < 21; ++r) {
        const RollCapture& rc = root->rolls[r];
        auto& nd = details.nd_player_rolls[r];
        auto& dt = details.dt_player_rolls[r];
        for (auto* d : {&nd, &dt}) {
            d->die1 = ALL_ROLLS_21[r].d1;
            d->die2 = ALL_ROLLS_21[r].d2;
            d->post_move_board = rc.chosen;
            d->is_terminal = rc.terminal;
            // Player's perspective: the terminal probs of the player's own
            // winning move, else the opponent node's pre-roll probs inverted.
            d->probs = rc.terminal ? terminal_probs(check_game_over(rc.chosen))
                                   : invert_probs(rc.probs);
        }
        if (is_money) {
            nd.cubeful_equity = -rc.cf[0];
            dt.cubeful_equity = -2.0f * rc.cf[2];
        } else {
            nd.cubeful_equity = mwc2eq(1.0f - rc.cf[0], a1, a2, cv, craw);
            dt.cubeful_equity = mwc2eq(1.0f - rc.cf[2], a1, a2, 2 * cv, craw);
        }

        if (rc.terminal || !rc.child) continue;   // no opponent-roll level
        const NodeCapture& opp = *rc.child;
        for (auto* d : {&nd, &dt}) {
            d->opponent_rolls.resize(21);
            for (int o = 0; o < 21; ++o) {
                auto& od = d->opponent_rolls[o];
                od.die1 = ALL_ROLLS_21[o].d1;
                od.die2 = ALL_ROLLS_21[o].d2;
                od.post_move_board = flip(opp.rolls[o].chosen);
                od.probs = opp.rolls[o].probs;   // player's pre-roll POV
            }
        }
        opponent_roll_equities(nd, opp, 0, 1, cv, 2 * cv, cv, a1, a2, craw, is_money);
        opponent_roll_equities(dt, opp, 4, 5, 2 * cv, 4 * cv, cv, a1, a2, craw, is_money);
    }
    return cd;
}

// Cubeless N-ply evaluation through the cubeful evaluation engine
// (cube_eval.cpp): a single dead cube (cube_value=1, max_cube_value=1)
// makes cl2cf bypass Janowski everywhere and interior picks reduce to
// cubeless 1-ply equity, so the tree is exactly a cubeless N-ply
// evaluation of flip(board); its accumulated probs, inverted, are the
// post-move probs of `board`'s mover. Mirrors the dead-cube paths the
// rollout trial loop uses for cubeless move selection and truncation.
template<bool ShareBatches>
static std::array<float, NUM_OUTPUTS> cubeless_post_probs(
    const Board& board, const Board& pre_move_board, int plies, int n_threads,
    bool deep_prefilter, const Strategy& base, const BearoffDB* db,
    const Strategy* move_prefilter, CandidateBatchCache* batches, const std::function<void()>& check_budget = {})
{
    // Exact-DB short-circuit and 1-ply base case mirror the recursion's.
    if (db && db->is_bearoff(board)) {
        auto probs = db->lookup_probs(board, /*post_move=*/true);
        clamp_probs_to_board(probs, board);
        return probs;
    }
    if (plies <= 1) {
        auto probs = base.evaluate_probs(board, pre_move_board);
        clamp_probs_to_board(probs, board);
        return probs;
    }
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }

    static const CubeInfo dead = [] {
        CubeInfo c;
        c.cube_value = 1;
        c.max_cube_value = 1;
        return c;
    }();
    float cf_unused;
    std::array<float, NUM_OUTPUTS> tree_probs{};
    cubeful_equity_nply_multi_impl<ShareBatches>(
        flip(board), &dead, 1, base, plies, &cf_unused,
        MoveFilters::TINY, n_threads,
        move_prefilter,
        /*fTop=*/false, &tree_probs, deep_prefilter, &pre_move_board, batches, nullptr, check_budget);
    auto probs = invert_probs(tree_probs);
    clamp_probs_to_board(probs, board);
    return probs;
}

std::array<float, NUM_OUTPUTS> MultiPlyStrategy::cubeless_tree_probs(
    const Board& board, const Board& pre_move_board,
    int plies, int n_threads, bool deep_prefilter) const
{
    return cubeless_post_probs<false>(board, pre_move_board, plies, n_threads, deep_prefilter,
                                      *base_, bearoff_db_, move_prefilter_.get(), nullptr);
}

CubeDecision MultiPlyStrategy::cube_decision_with_cubeless_probs(
    const Board& board, const CubeInfo& cube, const Strategy* cube_move_filter,
    std::array<float, NUM_OUTPUTS>& pre_roll_probs, const std::function<void()>& check_budget) const
{
    const Board flipped = flip(board);
    if ((n_plies_ != 2 && n_plies_ != 3 && n_plies_ != 4) || full_depth_opponent_ || filter_strat_ ||
        (bearoff_db_ && bearoff_db_->is_bearoff(flipped)) ||
        check_game_over(flipped) != GameResult::NOT_OVER) {
        auto decision = cube_decision_nply(board, cube, *base_, n_plies_, move_filter_,
                                           parallel_threads_, cube_move_filter, check_budget);
        pre_roll_probs = invert_probs(evaluate_probs(flipped, flipped));
        return decision;
    }

#if defined(BGBOT_USE_ACCELERATE)
    // Accelerate cblas_sgemm chooses row-tiling kernels by batch size M, so
    // splitting a full candidate list into a saved subset + remainder can
    // perturb float32 accumulation in the last ULP. Full (unfiltered) roll
    // batches still share exact results; filtered rolls evaluate as one batch.
    const bool share_survivors = false;
#else
    const bool share_survivors = base_->supports_cube_candidate_reuse();
#endif
    CandidateBatchCache batches(n_plies_, share_survivors);
    CubeDecision decision{};
    cube_decision_nply_multi_impl<true>(board, &cube, 1, *base_, n_plies_, &decision,
        move_filter_, parallel_threads_, cube_move_filter, false, &batches, check_budget);

    batches.reading = true;
    const auto post_probs = cubeless_post_probs<true>(
        flipped, flipped, n_plies_, parallel_thread_count(n_plies_), false,
        *base_, bearoff_db_, move_prefilter_.get(), &batches, check_budget);
    pre_roll_probs = invert_probs(post_probs);
    return decision;
}

} // namespace bgbot
