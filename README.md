# Token Launch Radar r4

[![Token Radar](https://make.fun/badge/token-radar-01.svg)](https://make.fun/c/token-radar-01)
[![Rank](https://make.fun/badge/token-radar-01/firtualz/launch-radar-r4/rank.svg)](https://make.fun/c/token-radar-01)
[![Reward](https://make.fun/badge/token-radar-01/firtualz/launch-radar-r4/reward.svg)](https://make.fun/c/token-radar-01)

Rebuild of the token-radar-01 entry (make.fun). One ridge model, fitted on
the 81 traded comps in the frozen DefiLlama snapshot (2026-11-14), prices
every row of a 198-row board of pre-token protocols by implied value
(value-if-it-launches x token odds), each with a costed farm plan.

Run it:

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input ../input/snapshot.json
```

That writes `board.json`, `data.js`, `dashboard/` (with the generated
research notes `00-note.md`, `01-method.md`, `02-ledger.md`) and, once the
four screenshots are in `screenshots/`, `dossier.json` via `buildDossier`.

Screenshots are rendered from the same `out/dashboard` the program writes,
with a headless browser, one per view: board, detail, filtered, validation.
No number in them is typed by hand.

Model: ridge, lambda 30, features log10 TVL, log10 age, 7-day change
(clamped to [-60%, +120%]), log10 chain count, category dummies.
Validation is reported as a range (leave-one-out, true time-split,
walk-forward across three cutoffs), never a single split. Bands are
predictive intervals from the leave-one-out residuals, so they include
protocol-level noise. Farm plans are $1,000 standard plans with an EV
headline, net of gas, and a breakeven listing value; costs are assumptions
dated Nov 2026.
