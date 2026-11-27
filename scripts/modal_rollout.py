# SPDX-License-Identifier: MPL-2.0
"""
Roll out post-move positions on Modal (https://modal.com) CPU containers.

Input is a position file in the repo's training-data format: one post-move
board per line, 26 space-separated ints, the side that just moved positive
(what the harvest scripts write, e.g. ``data/s11-bg-containment-data``).
Output is the matching rollout file: the same 26 ints followed by the five
cubeless rolled-out probabilities to 4 decimals (W, Gw, Bw, Gl, Bl, from the
positive side's perspective) — the format ``run_s11_containment_sl.py`` and
``run_s11_backgame_sl.py`` read. A ``.meta.jsonl`` sidecar keeps equity,
standard errors and per-board seconds.

Default rollout convention (the one the S11 piles were rolled out with):
Stage 9 play, 1,296 trials, no truncation, 3-ply checker play throughout
(``ultra_late_threshold=9999``), VR on, PubEval 20/15 prefilter, bearoff DB.
Every knob is a flag.

Resume skips boards whose data and provenance records both match. Results are
appended as they arrive. Incomplete pairs and incompatible settings stop the
run before it appends further labels. ``--max-usd`` stops the run once the
estimated compute cost reaches the limit (the estimate counts rollout time only — container
start-up and idle time are extra; Modal's dashboard has the real bill).

The engine is built inside the Modal image from ``cpp/`` with the inherited
Linux x86-64 flags (AVX2/FMA, -ffast-math). Its values differ from the Apple
Accelerate build at the ~1e-3 equity level, well under a rollout's standard
error (~0.009 at 1,296 trials).

Usage (from the repo root, with ``pip install modal`` and ``modal setup`` done):

    # local sanity run on this Mac, same code path, no Modal
    python scripts/modal_rollout.py data/s11-bg-containment-data --local --limit 2

    # small cloud test: 20 boards, report measured cost per board
    python scripts/modal_rollout.py data/s11-bg-containment-data --limit 20 \\
        --out logs/modal_test-rollout

    # real run: 100 containers x 8 cores, stop at $300
    python scripts/modal_rollout.py data/s11-bg-snake-exit1-data \\
        --n-trials 648 --containers 100 --max-usd 300
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path
from typing import TextIO

try:
    import modal
except ImportError:
    modal = None

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Modal list prices (modal.com/pricing, checked 2026-11-18): CPU is billed per
# PHYSICAL core (2 vCPU) per second, memory per GiB per second, each on
# max(request, usage).
USD_PER_CORE_SEC = 0.0000131
USD_PER_GIB_SEC = 0.00000222

REMOTE_ROOT = "/root/neongammon"
ROLLOUT_OPTIONS = dict(enable_vr=True, ultra_late_threshold=9999)

# --------------------------------------------------------------------------
# Engine-side code (runs locally with --local, and inside the containers)
# --------------------------------------------------------------------------


def _engine_paths(root: str) -> None:
    for p in (os.path.join(root, "python"), os.path.join(root, "build")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _normalize_routing(routing: dict | None) -> dict | None:
    if routing is None:
        return None
    return {
        k: list(v) if isinstance(v, (tuple, list)) else v
        for k, v in sorted(routing.items())
    }


def _build_strategy(bgbot_cpp, w):
    from neongammon.weights import WeightConfigPair

    args = (w.strategy_type, w.weight_paths_list, w.hidden_sizes_list)
    if isinstance(w, WeightConfigPair) and w.routing is not None:
        routing = bgbot_cpp.RoutingManifest()
        for name, value in w.routing.items():
            setattr(routing, name, value)
        return bgbot_cpp.create_strategy(*args, routing=routing)
    return bgbot_cpp.create_strategy(*args)


def make_rollout(root: str, model: str, n_trials: int, checker_ply: int,
                 truncation_depth: int, n_threads: int, seed: int):
    """Build the rollout strategy. Returns (strategy, keepalive objects)."""
    _engine_paths(root)
    import bgbot_cpp
    from neongammon.weights import WeightConfig, bearoff_db_path

    bgbot_cpp.init_escape_tables()
    w = WeightConfig.from_model(model)
    w.validate()
    base = _build_strategy(bgbot_cpp, w)
    strat = bgbot_cpp.create_rollout_from_base(
        base,
        n_trials=n_trials,
        truncation_depth=truncation_depth,
        decision_ply=checker_ply,
        n_threads=n_threads,
        seed=seed,
        **ROLLOUT_OPTIONS,
    )
    pubeval = bgbot_cpp.PubEval()
    strat.set_move_filter(pubeval)
    keep = [pubeval]
    db_path = bearoff_db_path()
    if db_path:
        db = bgbot_cpp.BearoffDB()
        if not db.load(db_path):
            raise ValueError(f"Could not load bearoff database {db_path}")
        bgbot_cpp.rollout_set_bearoff_db(strat, db)
        keep.append(db)
    return strat, keep


def rollout_one(strat, board: list[int]) -> dict:
    t0 = time.perf_counter()
    r = strat.rollout_position(list(board))
    return {
        "board": list(board),
        "probs": [float(p) for p in r.mean_probs],
        "equity": float(r.equity),
        "se": float(r.std_error),
        "prob_se": [float(p) for p in r.prob_std_errors],
        "secs": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------
# Modal app
# --------------------------------------------------------------------------

def _weights_ignore(p: Path) -> bool:
    # Upload only the registry's weight files, not TD checkpoints or the
    # diagnostic subfolders.
    return not (p.name.startswith("sl_") and p.name.endswith(".weights.best"))


image = None
app = None
Roller = None

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential", "cmake", "ninja-build")
        .pip_install("numpy", "pybind11")
        .add_local_dir(_PROJECT_ROOT / "cpp", f"{REMOTE_ROOT}/cpp", copy=True,
                       ignore=["move_array_probe"])
        .run_commands(
            f"mkdir -p {REMOTE_ROOT}/build",
            f"cmake -S {REMOTE_ROOT}/cpp -B /tmp/bgbuild -G Ninja -DCMAKE_BUILD_TYPE=Release"
            " -DBGBOT_BUILD_APPS=OFF -Dpybind11_DIR=$(python -m pybind11 --cmakedir)",
            "ninja -C /tmp/bgbuild bgbot_cpp",
            f"cp /tmp/bgbuild/bgbot_cpp*.so {REMOTE_ROOT}/build/",
        )
        .add_local_dir(_PROJECT_ROOT / "python" / "neongammon", f"{REMOTE_ROOT}/python/neongammon",
                       ignore=["**/__pycache__"])
        .add_local_dir(_PROJECT_ROOT / "models", f"{REMOTE_ROOT}/models",
                       ignore=_weights_ignore)
        .add_local_file(_PROJECT_ROOT / "data" / "bearoff_1sided.db",
                        f"{REMOTE_ROOT}/data/bearoff_1sided.db")
    )

    app = modal.App("neongammon-rollout", image=image)

    @app.cls(cpu=8.0, memory=4096, timeout=3 * 3600, retries=1, max_containers=50)
    class Roller:
        model: str = modal.parameter(default="stage9")
        n_trials: int = modal.parameter(default=1296)
        checker_ply: int = modal.parameter(default=3)
        truncation_depth: int = modal.parameter(default=0)
        threads: int = modal.parameter(default=16)
        seed: int = modal.parameter(default=42)

        @modal.enter()
        def load(self):
            self.strat, self._keep = make_rollout(
                REMOTE_ROOT, self.model, self.n_trials, self.checker_ply,
                self.truncation_depth, self.threads, self.seed)

        @modal.method()
        def rollout(self, board: list[int]) -> dict:
            return rollout_one(self.strat, board)

        @modal.method()
        def rollout_many(self, boards: list[list[int]]) -> list[dict]:
            # One input per chunk: for sub-second boards (races) the per-input
            # overhead would otherwise rival the rollout itself.
            return [rollout_one(self.strat, b) for b in boards]


# --------------------------------------------------------------------------
# Local driver
# --------------------------------------------------------------------------


def _load_boards(path: Path) -> list[list[int]]:
    boards = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 26:
                continue
            boards.append([int(x) for x in parts[:26]])
    return boards


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpp_source_files(root: Path) -> list[Path]:
    """Return sorted C++ source files under cpp/, preferring git-tracked files."""
    cpp = root / "cpp"
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", "cpp"],
            check=True,
            capture_output=True,
        )
        tracked = [
            root / rel.decode("utf-8")
            for rel in proc.stdout.split(b"\0")
            if rel
        ]
        files = [
            path for path in tracked
            if path.is_file() and "move_array_probe" not in path.relative_to(cpp).parts
        ]
        if files:
            return sorted(files)
    except (OSError, subprocess.SubprocessError):
        pass
    return sorted(
        path for path in cpp.rglob("*")
        if path.is_file() and "move_array_probe" not in path.relative_to(cpp).parts
    )


def _provenance(config: dict, threads: int, local: bool) -> str:
    _engine_paths(str(_PROJECT_ROOT))
    from neongammon.weights import WeightConfig, bearoff_db_path

    weights = WeightConfig.from_model(config["model"])
    weights.validate()
    files = {path: _sha256_file(path) for path in dict.fromkeys(weights.weight_paths_list)}
    db = bearoff_db_path()
    if local:
        import bgbot_cpp

        engine = _sha256_file(bgbot_cpp.__file__)
    else:
        # Modal builds this uploaded C++ tree in its Linux x86-64 image.
        cpp = _PROJECT_ROOT / "cpp"
        engine = {
            path.relative_to(cpp).as_posix(): _sha256_file(path)
            for path in _cpp_source_files(_PROJECT_ROOT)
        }
    manifest = dict(
        version=1, config=config, options=ROLLOUT_OPTIONS, threads=threads,
        runner="local" if local else "modal", engine=engine,
        strategy=weights.strategy_type, hidden_sizes=weights.hidden_sizes_list,
        routing=_normalize_routing(getattr(weights, "routing", None)),
        weights=[files[path] for path in weights.weight_paths_list],
        bearoff=_sha256_file(db) if db else None,
    )
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _done_keys(output: TextIO, metadata: TextIO, provenance: str) -> set[tuple[int, ...]]:
    output.seek(0)
    metadata.seek(0)
    done = set()
    for number, (row, entry) in enumerate(zip_longest(output, metadata, fillvalue=""), 1):
        if not row.endswith("\n") or not entry.endswith("\n"):
            raise ValueError(f"incomplete label/metadata pair at row {number}")
        try:
            saved = json.loads(entry)
            fields = row.split()
            board = tuple(int(x) for x in fields[:26])
        except ValueError as error:
            raise ValueError(f"invalid rollout data at row {number}") from error
        if not isinstance(saved, dict) or saved.get("provenance") != provenance:
            raise ValueError(f"rollout provenance differs or is missing at row {number}")
        row_digest = hashlib.sha256(row[:-1].encode()).hexdigest()
        if (len(fields) != 31 or list(board) != saved.get("board")
                or saved.get("label_sha256") != row_digest or board in done):
            raise ValueError(f"label/metadata mismatch at row {number}")
        done.add(board)
    return done


@contextmanager
def _locked_output(out: Path, provenance: str, parser: argparse.ArgumentParser):
    meta = out.with_name(out.name + ".meta.jsonl")
    with out.open("a+", encoding="utf-8") as labels:
        # Serialize validation and appends for each output path.
        fcntl.flock(labels, fcntl.LOCK_EX)
        with meta.open("a+", encoding="utf-8") as metadata:
            try:
                done = _done_keys(labels, metadata, provenance)
            except ValueError as error:
                parser.error(
                    f"{out}: {error}. Use --out with a new file, or restore a verified matching pair."
                )
            yield labels, metadata, done


def _default_out(positions: Path) -> Path:
    name = positions.name
    if name.endswith("-data"):
        name = name[: -len("-data")]
    return positions.with_name(name + "-rollout")


def _format_row(res: dict) -> str:
    return (" ".join(str(x) for x in res["board"]) + " "
            + " ".join(f"{p:.4f}" for p in res["probs"]))


def _consume_remote_stream(
    stream,
    pending_count: int,
    chunk_size: int,
    record,
    progress,
    *,
    max_usd: float,
    usd_per_sec: float,
    get_counts,
) -> tuple[int, int, bool]:
    """Consume a Modal map() stream, recording results and counting failures.

    Returns ``(failed_boards, failed_chunks, stopped_early)``. When the stream
    runs to completion (``not stopped_early``), ``failed_boards`` is the exact
    count of pending boards that did not produce a result (avoiding tail-chunk
    overcounting when ``pending_count % chunk_size != 0``).
    """
    failed_chunks = 0
    stopped_early = False
    last = 0.0
    for item in stream:
        if isinstance(item, BaseException):
            failed_chunks += 1
            if chunk_size > 1:
                print(f"chunk failed (up to {chunk_size} board(s)): {item!r}", flush=True)
            else:
                print(f"input failed (1 board): {item!r}", flush=True)
            continue
        for res in (item if chunk_size > 1 else [item]):
            record(res)
        n_ok, busy_secs = get_counts()
        if time.time() - last > 30 or n_ok == pending_count:
            print(progress(), flush=True)
            last = time.time()
        if max_usd and busy_secs * usd_per_sec >= max_usd:
            print(f"--max-usd {max_usd} reached; stopping (re-run to resume)",
                  flush=True)
            stopped_early = True
            break
    n_ok, _ = get_counts()
    if failed_chunks == 0:
        n_fail = 0
    elif not stopped_early:
        n_fail = max(0, pending_count - n_ok)
    else:
        n_fail = min(failed_chunks * chunk_size, max(0, pending_count - n_ok))
    return n_fail, failed_chunks, stopped_early


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("positions", type=Path, help="position file (26 ints per line)")
    ap.add_argument("--out", type=Path, default=None,
                    help="rollout file to append to (default: <name>-rollout next to the input)")
    ap.add_argument("--model", default="stage9", help="model that plays the trials (default stage9)")
    ap.add_argument("--n-trials", type=int, default=1296)
    ap.add_argument("--checker-ply", type=int, default=3)
    ap.add_argument("--truncation-depth", type=int, default=0, help="0 = play to the end")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="roll out at most N pending boards")
    ap.add_argument("--cpu", type=float, default=8.0, help="physical cores per container")
    ap.add_argument("--memory", type=int, default=4096, help="MiB per container")
    ap.add_argument("--containers", type=int, default=50, help="max concurrent containers")
    ap.add_argument("--timeout", type=int, default=3 * 3600,
                    help="seconds per input (one board, or one --chunk)")
    ap.add_argument("--chunk", type=int, default=1,
                    help="boards per Modal input; raise it (e.g. 50) for sub-second boards")
    ap.add_argument("--max-usd", type=float, default=0.0,
                    help="stop once this run's estimated compute cost reaches it (0 = no limit)")
    ap.add_argument("--local", action="store_true", help="run in-process on this machine instead")
    ap.add_argument("--threads", type=int, default=0,
                    help="rollout threads (default: 2 x --cpu on Modal, all cores with --local)")
    args = ap.parse_args()

    positions = args.positions if args.positions.is_absolute() else Path.cwd() / args.positions
    out = args.out or _default_out(positions)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = dict(model=args.model, n_trials=args.n_trials, checker_ply=args.checker_ply,
                  truncation_depth=args.truncation_depth, seed=args.seed)
    default_threads = (os.cpu_count() or 1) if args.local else int(round(2 * args.cpu))
    threads = args.threads or default_threads
    provenance = _provenance(config, threads, args.local)
    with _locked_output(out, provenance, ap) as (labels, metadata, done):
        seen = set(done)

        pending = []
        for b in _load_boards(positions):
            k = tuple(b)
            if k in seen:
                continue
            seen.add(k)
            pending.append(b)
        if args.limit:
            pending = pending[: args.limit]
        print(f"{len(done)} boards already in {out}; {len(pending)} to roll out", flush=True)
        if not pending:
            return

        usd_per_sec = args.cpu * USD_PER_CORE_SEC + (args.memory / 1024) * USD_PER_GIB_SEC

        n_ok = n_fail = 0
        busy_secs = 0.0
        t_start = time.time()

        def record(res: dict) -> None:
            nonlocal n_ok, busy_secs
            row = _format_row(res)
            entry = {k: res[k] for k in ("board", "equity", "se", "prob_se", "secs")}
            entry.update(config, provenance=provenance, label_sha256=hashlib.sha256(row.encode()).hexdigest())
            labels.write(row + "\n")
            labels.flush()
            metadata.write(json.dumps(entry) + "\n")
            metadata.flush()
            n_ok += 1
            busy_secs += res["secs"]

        def progress() -> str:
            wall = time.time() - t_start
            est = busy_secs * usd_per_sec
            per = est / n_ok if n_ok else 0.0
            return (f"{n_ok}/{len(pending)} done, {n_fail} failed, {wall / 60:.1f} min wall, "
                    f"{busy_secs / max(n_ok, 1):.1f} s/board, est ${est:.2f} "
                    f"(${per:.4f}/board)")

        if args.local:
            strat, _keep = make_rollout(
                str(_PROJECT_ROOT), args.model, args.n_trials, args.checker_ply,
                args.truncation_depth, threads, args.seed)
            for b in pending:
                record(rollout_one(strat, b))
                print(progress(), flush=True)
            return

        if modal is None or app is None or Roller is None:
            raise SystemExit(
                "modal is required for remote rollouts; install modal or pass --local"
            )

        # A plain SIGTERM would kill the client without leaving app.run(), and
        # Modal keeps the ephemeral app's containers running (and billing)
        # until it notices; turn it into a KeyboardInterrupt so the app stops.
        def _term(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)

        roller_cls = Roller.with_options(cpu=args.cpu, memory=args.memory,
                                         timeout=args.timeout, max_containers=args.containers)
        with modal.enable_output(), app.run():
            roller = roller_cls(threads=threads, **config)
            if args.chunk > 1:
                chunks = [pending[i:i + args.chunk] for i in range(0, len(pending), args.chunk)]
                stream = roller.rollout_many.map(chunks, order_outputs=False,
                                                 return_exceptions=True)
            else:
                stream = roller.rollout.map(pending, order_outputs=False,
                                            return_exceptions=True)
            n_fail, failed_chunks, _ = _consume_remote_stream(
                stream,
                len(pending),
                args.chunk,
                record,
                progress,
                max_usd=args.max_usd,
                usd_per_sec=usd_per_sec,
                get_counts=lambda: (n_ok, busy_secs),
            )
        print(progress(), flush=True)
        if n_fail > 0 or failed_chunks > 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
