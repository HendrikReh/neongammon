# SPDX-License-Identifier: MPL-2.0
"""Profile the owned-money-cube fixture in a timing build of the 3.2.0 extension."""
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
import subprocess
import sys


def sample(extension):
    if extension:
        sys.path.insert(0, str(Path(extension).resolve()))
    import bgbot_cpp
    from importlib.metadata import version
    if version("neongammon") != "3.2.0":
        raise ValueError("Use the installed NeonGammon 3.2.0 wheel")
    if extension and Path(bgbot_cpp.__file__).resolve().parent != Path(extension).resolve():
        raise ValueError("The timing extension did not load")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmark_cube_action_decision import CASES, sample as time_public_call
    from neongammon import BgBotAnalyzer

    _, board, options = CASES[1]
    first, warm, result = time_public_call("cube_action", board, options)
    analyzer = BgBotAnalyzer(eval_level="3ply", parallel_threads=4)
    decision = asdict(analyzer.cube_decision(board, **options))
    independent = analyzer.cube_action(board, incl_2ply_details=True, **options)
    assert all(result[key] == value for key, value in decision.items())
    assert result["probs"] == asdict(independent.probs)
    assert result["cubeless_equity"] == independent.cubeless_equity
    print(json.dumps([first, warm, result]))


def profile(extension):
    first = []
    warm = []
    wheel_first, wheel_warm = [], []
    reference = None
    for pair in range(16):
        for instrumented in (False, True) if pair % 2 == 0 else (True, False):
            command = [sys.executable, "-I", __file__, "--sample"]
            if instrumented:
                command.append(extension)
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(process.stderr)
            public_first, public_warm, result = json.loads(process.stdout)
            if reference is None:
                reference = result
            if result != reference:
                raise ValueError("Installed wheel and diagnostic results differ")
            if instrumented:
                # Only cube_action's combined traversal emits these timings.
                phases = [tuple(int(value) / 1e6 for value in line.split()[1:])
                          for line in process.stderr.splitlines() if line.startswith("NG63 ")]
                if len(phases) != 1 + 3 + len(public_warm):
                    raise ValueError(f"Unexpected number of phase samples: {len(phases)}")
                first.append(phases[0])
                warm.extend(phases[-len(public_warm):])
            else:
                wheel_first.append(public_first)
                wheel_warm.extend(public_warm)
    for name, rows, wheel in (("first", first, wheel_first), ("warm", warm, wheel_warm)):
        cube, cubeless = zip(*rows)
        print(f"{name} ({len(rows)} calls): cube {median(cube):.3f} ms, "
              f"cubeless {median(cubeless):.3f} ms, "
              f"cubeless share {median(b/(a+b) for a,b in rows):.1%}, "
              f"wheel cube_action {median(wheel):.3f} ms")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--sample"]:
        sample(sys.argv[2] if len(sys.argv) > 2 else None)
    elif len(sys.argv) == 2:
        profile(sys.argv[1])
    else:
        sys.exit("usage: python -I measure_phases.py TIMING_EXTENSION_DIRECTORY")
