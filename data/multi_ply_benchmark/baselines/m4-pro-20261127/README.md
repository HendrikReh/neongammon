# Apple M4 Pro baseline

Measured on November 27, 2026 with an Apple M4 Pro (14 logical CPUs, 64 GB),
macOS 27.0.1 (26A434), and CPython 3.14.7. The arm64 Release
extension was built from [`8b70d18`](https://github.com/HendrikReh/neongammon/commit/8b70d1801e1b0a1d784230a91bcdf30747afd5cb)
using Apple Clang 21.0.0 and Accelerate. The run used Stage 11, the one-sided
bearoff database, four requested native threads and
`VECLIB_MAXIMUM_THREADS=1`.

The one-minute host load averaged 3.1 at the start and 7.9 at the end.
Each cell uses 100 API calls from one pass in fixed order. Timings measure
wall time in the public API, including any setup done by the first call.

## Latency in milliseconds

Each cell is median / 95th percentile.

| Level | Checker play | Cube action | Post-move analysis |
| --- | ---: | ---: | ---: |
| 1-ply | 0.039 / 0.145 | 0.011 / 0.017 | 0.007 / 0.010 |
| 2-ply | 1.001 / 2.262 | 0.291 / 0.568 | 0.138 / 0.206 |
| 3-ply | 24.245 / 109.097 | 3.348 / 7.995 | 1.745 / 2.836 |
| 4-ply | 334.746 / 1904.391 | 85.200 / 157.923 | 36.285 / 57.689 |
| 1T | 47.982 / 135.983 | 20.018 / 53.543 | 20.490 / 29.046 |
| 2T | 441.274 / 1298.344 | 215.096 / 301.194 | 217.873 / 297.057 |
| 3T | 1969.330 / 6674.207 | 1098.908 / 1911.537 | 1104.211 / 1903.768 |

## Comparison to frozen rollout values

Mean absolute equity errors are in mEq (one thousandth of an equity unit).
Checker errors use 378 matched rollout endpoints. Cube and post-move errors
each use 100 positions.

| Level | Checker cubeful MAE | Cube no-double MAE | Post-move cubeful MAE | Checker picks with rollout | Cube actions matching reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-ply | 28.86 | 64.49 | 74.88 | 99/100 | 88/100 |
| 2-ply | 23.21 | 54.95 | 64.90 | 100/100 | 88/100 |
| 3-ply | 19.43 | 42.95 | 70.28 | 100/100 | 95/100 |
| 4-ply | 16.07 | 34.59 | 59.86 | 100/100 | 98/100 |
| 1T | 17.00 | 27.30 | 28.37 | 100/100 | 98/100 |
| 2T | 11.88 | 28.85 | 30.87 | 100/100 | 96/100 |
| 3T | 10.71 | 17.76 | 18.91 | 100/100 | 98/100 |

[Machine-readable summary](summary.json) · [All 2,100 API results and timings](results.jsonl.gz) · [Reproduction command and reference coverage](../../README.md)
