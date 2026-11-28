# Primary Sources on NeonGammon Playing Strength

As of November 20, 2026.

## PureTD and directly trained cubeful play

Alexander Strehl trains checker and cube decisions via self-play. The money model learns equity per unit of the current cube value. It receives cube ownership and whether a cube or checker decision is pending. With exact Bellman updates, training averages over all 21 distinct dice rolls. For reproducing training from scratch, the preprint cites roughly 250 million games and roughly 70 hours on a Zen 5 machine with many CPU cores and a consumer GPU. [PureTD v3, Method](https://arxiv.org/html/2608.15146v3#S3), [Training recipe](https://arxiv.org/html/2608.15146v3#A6)

| Experiment in preprint | Result for cubeful money games |
| --- | --- |
| Head-to-head games | PureTD without lookahead scores +23.8 mEq per game against Open Sage Stage 11 with one move of lookahead and +27.2 against GNUbg with one move of lookahead. [Table 3](https://arxiv.org/html/2608.15146v3#S4.T3) |
| XGRoller++ PR | With zero, one, and two moves of lookahead, PureTD scores 0.80 / 0.40 / 0.29. The figures for Open Sage are 2.11 / 1.24 / 0.40. [Table 6](https://arxiv.org/html/2608.15146v3#S4.T6) |
| Direct comparison at equal network size | The directly cubeful-trained variant scores +14.6 mEq per game over the cubeless-trained model with the Janowski cube heuristic. [Table 7](https://arxiv.org/html/2608.15146v3#S4.T7) |

Open Sage calls the three search depths 1P, 2P, and 3P. The PR figures come from 1,000 self-play games per configuration. The head-to-head comparisons use Jacoby and cap each game's score contribution at ±128. The numbers refer to the September 7, 2026 version of Open Sage and to money games. [PureTD v3, Experiments](https://arxiv.org/html/2608.15146v3#S4), [Reproduction details](https://arxiv.org/html/2608.15146v3#A1)

The author publishes checkpoints, training, and test scripts. The current repository also provides a C API and newer model weights. For a reproducible comparison, pin the commit and weight file. The preprint uses `18b0a80d258e`. According to the author, the code and published models are under the MIT license, while third-party adapters and benchmark patches carry separate license notices in some cases. [PureTD repository and license notices](https://github.com/alexstrehl/puretd), [License at paper commit](https://raw.githubusercontent.com/alexstrehl/puretd/18b0a80d258e/LICENSE)

## GNU Backgammon and eXtreme Gammon

- **GNUbg** can be operated headlessly on Linux via `--tty`. Its move filters determine which candidates advance to the next search depth. Rollouts can use full games with an active cube, variance reduction, and stratified selection of early rolls. For reproducible comparisons, record search and rollout settings, match rules, and the equity scale used. GNUbg distinguishes, for example, Match Winning Chance (MWC) from equivalent money equity (EMG). [GNU Backgammon Manual](https://www.gnu.org/software/gnubg/manual/gnubg.html)
- **XG** documents Batch Analyze and Batch Rollout as well as saving analyzed games in the `.xg` format. XGRoller++ is a truncated rollout with 360 paths and variance reduction. Its documented truncation depth is five plies for checker decisions and seven for cube decisions. Full rollouts are therefore useful as an additional reference for selected disputed positions. [XG Manual](https://www.extremegammon.com/extremegammon2.pdf#page=82), [XG Support](https://www.extremegammon.com/support.aspx), [XG File Format](https://www.extremegammon.com/XGformat.aspx)
- **Usage rights.** XG documents machine-bound activations and publishes the file format specification. The public documentation does not contain an explicit grant for systematic model training on XG outputs or redistribution of such training data. GNUbg is GPL software and can be run as a separate process for comparisons, as demonstrated in the PureTD study using the unmodified binary. [XG Support](https://www.extremegammon.com/support.aspx), [GNU Backgammon Manual](https://www.gnu.org/software/gnubg/manual/gnubg.html), [PureTD v3, Reproduction details](https://arxiv.org/html/2608.15146v3#A1)

## Modal and rollout measurements

Modal distributes independent inputs across containers with `Function.map()`. Retries and restarts after preemption are documented. A reproducible work unit can pin the position, engine configuration, and dice seed and overwrite the same result on retry. This project already includes a CPU rollout entry point with local parity runs and an estimated cost cap. [Modal Functions](https://modal.com/docs/guide/function-invocation-methods), [Preemption](https://modal.com/docs/guide/preemption), [Retries](https://modal.com/docs/guide/retries), [Modal rollout script](https://github.com/HendrikReh/neongammon/blob/main/scripts/modal_rollout.py)

Tesauro and Galperin reported that Monte Carlo rollouts can improve on the moves of a fixed playing policy. The measured equity still depends on the policy used by both sides to play out the simulated games. This project's own comparison documents that dependence in practice: the ranking of Sage 3T and XGRoller++ reverses between the Sage and XG references. [Tesauro and Galperin, 1996](https://papers.nips.cc/paper_files/paper/1996/file/996009f2374006606f4c0b0fda878af1-Paper.pdf), [NeonGammon XG comparison](xg-comparison.md)
