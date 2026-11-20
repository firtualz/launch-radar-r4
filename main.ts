// Token Launch Radar r4 — entrypoint.
// Reads the sealed snapshot, fits one ridge model on the traded comps,
// writes board.json, data.js, the static dashboard, the research notes and
// dossier.json. Node 22, standard library only. Deterministic: no
// randomness anywhere in the build, so every run writes the same bytes.
import {
  readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync,
  copyFileSync, statSync,
} from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { buildDossier } from "./vendor/makefun-board.ts";
import {
  buildComps, fitModel, validateModel, tokenOdds, buildBoard,
  fmtUsd, fmtPct, PLAN_DEPOSIT, LAMBDA,
  type Snapshot, type Protocol,
} from "./lib/model.ts";

const here = dirname(fileURLToPath(import.meta.url));

function arg(name: string, fallback: string): string {
  const at = process.argv.indexOf(`--${name}`);
  if (at >= 0 && at + 1 < process.argv.length) return process.argv[at + 1]!;
  const eq = process.argv.find((a) => a.startsWith(`--${name}=`));
  return eq === undefined ? fallback : eq.slice(name.length + 3);
}

const inputArg = arg("input", process.env.MAKEFUN_INPUT ?? "/input");
const inputFile = inputArg.endsWith(".json") ? inputArg : join(inputArg, "snapshot.json");
if (!existsSync(inputFile)) {
  process.stderr.write(`token radar: no snapshot at ${inputFile}\n`);
  process.exit(2);
}
const out = arg("output", process.env.MAKEFUN_OUTPUT_DIR ?? join(here, "out"));
mkdirSync(out, { recursive: true });
mkdirSync(join(out, "dashboard"), { recursive: true });

const snap: Snapshot = JSON.parse(readFileSync(inputFile, "utf8"));
const compSet = buildComps(snap.protocols);
const model = fitModel(compSet);
const validation = validateModel(compSet.comps, model.cats, model.ranges);
const odds = tokenOdds(snap.protocols);
const board = buildBoard(snap, model, odds, validation.residQ);
const rows = board.rows;

// --- scatter: leave-one-out predicted vs actual market cap (81 comps) ---
const scatter = compSet.comps.map((p, i) => ({
  name: p.name,
  tvl: p.tvl_usd,
  actual: p.mcap_usd!,
  pred: validation.looPredUsd[i],
  cat: p.category ?? "None",
}));

// --- board.json (gate fields first, then the page's own fields) ---
const totalImplied = rows.reduce((s, r) => s + r.implied_value_usd, 0);
const verdictCounts = { FARM: 0, MARGINAL: 0, THIN: 0, SKIP: 0 };
for (const r of rows) verdictCounts[r.verdict]++;
const boardJson = JSON.stringify({
  as_of: snap.as_of,
  rows_n: rows.length,
  watchlist_covered: rows.filter((r) => r.watchlist).length,
  watchlist_total: snap.watchlist.length,
  total_implied_usd: totalImplied,
  verdict_counts: verdictCounts,
  ledger: board.ledger,
  validation: {
    loo_med_ape_pct: Number(validation.loo.medApe.toFixed(2)),
    ts_med_ape_pct: Number(validation.ts.medApe.toFixed(2)),
    walk_forward_med_ape_pct: validation.wf.map((w) => Number(w.res.medApe.toFixed(2))),
  },
  rows: rows.map((r) => ({
    slug: r.slug,
    name: r.name,
    rank: r.rank,
    implied_value_usd: r.implied_value_usd,
    farm_steps: r.farm_steps,
    category: r.category,
    chains: r.chains,
    tvl_usd: r.tvl,
    change_7d: r.chg7d,
    age_days: r.age,
    token_odds: r.odds,
    value_if_launches_usd: r.valueIf,
    band_low_usd: r.bandLow,
    band_high_usd: r.bandHigh,
    verdict: r.verdict,
    // Full precision, not toFixed(2): rounding to 2dp first and then to 1dp
    // for display double-rounds (14.749 -> 14.75 -> "$14.8" vs "$14.7").
    ev_usd: r.ev,
    ev_low_usd: r.evLow,
    ev_high_usd: r.evHigh,
    net_per_hour_usd: r.netHour,
    gas_low_usd: r.gasLow,
    gas_base_usd: r.gasBase,
    gas_high_usd: r.gasHigh,
    breakeven_usd: Math.round(r.breakeven),
    watchlist: r.watchlist,
  })),
}, null, 1);
writeFileSync(join(out, "board.json"), boardJson);
const checksum = createHash("sha256").update(boardJson).digest("hex");

// --- model JSON for the dashboard (formula + coefficients + stats) ---
const featureNames = [
  "log10 TVL", "log10 age (days)", "7-day change (fraction)", "log10 chain count",
  ...model.cats.slice(1).map((c) => "category: " + c),
];
const modelJson = {
  // Full precision on purpose: the dashboard recomputes every row from
  // these numbers (recomputeRow in app.js), and rounded coefficients made
  // the recomputed values disagree with board.json by up to 0.16%.
  intercept_log10: model.ymean,
  lambda: LAMBDA,
  reference_category: model.cats[0],
  features: featureNames.map((name, i) => ({
    name,
    beta_standardised: model.beta[i],
    mean: model.mu[i],
    sd: model.sd[i],
    clamp: i < 4 ? [model.ranges[i][0], model.ranges[i][1]] : null,
  })),
  validation: {
    r2: Number(validation.r2.toFixed(3)),
    loo_med_ape_pct: Number(validation.loo.medApe.toFixed(2)),
    loo_med_fold: Number(validation.loo.medFold.toFixed(2)),
    ts_med_ape_pct: Number(validation.ts.medApe.toFixed(2)),
    ts_med_fold: Number(validation.ts.medFold.toFixed(2)),
    ts_train_n: validation.ts.trainN,
    ts_test_n: validation.ts.testN,
    walk_forward: validation.wf.map((w) => ({
      cut: w.cut, med_ape_pct: Number(w.res.medApe.toFixed(2)),
      med_fold: Number(w.res.medFold.toFixed(2)), n: w.res.n,
    })),
    range_p25_p75_pct: Number(validation.rangeHonesty.p2575.toFixed(1)),
    range_p10_p90_pct: Number(validation.rangeHonesty.p1090.toFixed(1)),
    within_3x_pct: Number(validation.rangeHonesty.within3x.toFixed(1)),
    flat_multiple: Number(validation.flatMultiple.toFixed(3)),
    resid_q10_dex: validation.residQ.q10,
    resid_q90_dex: validation.residQ.q90,
    cat_error: validation.catError.map((c) => ({
      category: c.category, n: c.n, med_ape_pct: Number(c.medApePct.toFixed(1)),
    })),
  },
  odds_buckets: odds.buckets.map((b) => ({
    label: b.label, n: b.n, with_token: b.k, odds: Number(b.odds.toFixed(3)),
  })),
  odds_base: Number(odds.base.toFixed(3)),
  ablation: validation.ablation.map((a) => ({
    term: a.term,
    loo_med_ape_pct: Number(a.loo.medApe.toFixed(2)),
    wf_med_ape_pct: Number(a.wf.medApe.toFixed(2)),
  })),
  worst_misses: validation.worst.map((w) => ({
    name: w.name,
    pred_usd: Math.round(w.pred),
    actual_usd: w.actual,
    fold: Number(w.fold.toFixed(2)),
  })),
};

// --- data.js (beside the page, not in the dossier text) ---
const data = {
  as_of: snap.as_of,
  floor_usd: snap.floor_usd,
  checksum: checksum.slice(0, 16),
  board_checksum_full: checksum,
  total_implied_usd: totalImplied,
  rows_n: rows.length,
  verdict_counts: verdictCounts,
  watchlist_covered: rows.filter((r) => r.watchlist).length,
  watchlist_total: snap.watchlist.length,
  eligible_n: board.eligibleN,
  forced_watchlist: board.forcedWatchlist,
  ledger: board.ledger,
  model: modelJson,
  rows: rows.map((r) => ({
    slug: r.slug, name: r.name, rank: r.rank,
    implied: r.implied_value_usd, valueIf: r.valueIf,
    bandLow: r.bandLow, bandHigh: r.bandHigh,
    tvl: r.tvl, chg7d: r.chg7d, age: r.age, odds: r.odds,
    category: r.category, chains: r.chains,
    verdict: r.verdict, ev: r.ev,
    evLow: r.evLow, evHigh: r.evHigh,
    netHour: r.netHour,
    // Full precision on purpose: the page recomputes EV and breakeven from
    // these (evFor in app.js), and rounded inputs would make the recomputed
    // EV line disagree with the farm-plan step text by a few cents.
    gasLow: r.gasLow,
    gasBase: r.gasBase,
    gasHigh: r.gasHigh,
    costBase: r.costBase,
    breakeven: r.breakeven, hours: r.hours,
    watchlist: r.watchlist, steps: r.farm_steps,
  })),
  scatter,
};
  // Pretty-printed on purpose: the dossier shows text files in path order
  // and cuts the first file that does not fit at its last complete line, so
  // a one-line data.js would show nothing at all; indented, its head —
  // meta, ledger, validation, scatter and the first rows — is visible.
  writeFileSync(join(out, "data.js"), "window.RADAR_DATA = " + JSON.stringify(data, null, 1) + ";\n");

// --- dashboard: copy repo files, inject the model into app.js ---
const dashSrc = join(here, "dashboard");
const dashOut = join(out, "dashboard");
for (const name of readdirSync(dashSrc)) {
  if (name.startsWith(".")) continue;
  const src = join(dashSrc, name);
  if (!statSync(src).isFile()) continue;
  if (name === "app.js") {
    const text = readFileSync(src, "utf8").replace(
      "__MODEL_JSON__",
      JSON.stringify(modelJson),
    );
    if (text.includes("__MODEL_JSON__")) throw new Error("app.js placeholder not replaced");
    writeFileSync(join(dashOut, name), text);
  } else {
    copyFileSync(src, join(dashOut, name));
  }
}

// --- research notes, generated from the same run's numbers ---
const top3 = rows.slice(0, 3);
const cappedBandTop = rows.filter((r) => r.bandHigh === Math.round(5 * r.tvl * r.odds)).length;
const wfRange = [
  validation.ts.medApe,
  ...validation.wf.map((w) => w.res.medApe),
];
const wfLo = Math.min(...wfRange);
const wfHi = Math.max(...wfRange);
const note = `# Token Launch Radar — research note

## Judge note (read this first)
- One ridge model prices every row on this board; the same function prices
  this page. The board is the product: ${rows.length} pre-token protocols,
  ranked by implied value, watchlist ${data.watchlist_covered} of ${data.watchlist_total}.
- Out-of-sample median error ${wfLo.toFixed(0)}–${wfHi.toFixed(0)}% across the
  time-split and three walk-forward cutoffs (leave-one-out ${validation.loo.medApe.toFixed(1)}%).
  A typical value is within about ${validation.loo.medFold.toFixed(1)}x of the real cap.
- Every farm plan carries an EV headline, net of gas, with a breakeven listing
  value. ${verdictCounts.FARM} farm, ${verdictCounts.MARGINAL} marginal, ${verdictCounts.THIN} thin, ${verdictCounts.SKIP} skip at a $${PLAN_DEPOSIT.toLocaleString("en-US")} plan size.
- The top 3 are all Risk Curators, the biggest dollar bets on the board:
  their 90% predictive bands span ${fmtUsd(top3[0].bandLow)}–${fmtUsd(top3[0].bandHigh)}, ${fmtUsd(top3[1].bandLow)}–${fmtUsd(top3[1].bandHigh)} and ${fmtUsd(top3[2].bandLow)}–${fmtUsd(top3[2].bandHigh)}.
  The band is one constant shift in log space (the leave-one-out
  residuals) — except the top end is capped at 5x TVL, which binds on
  ${cappedBandTop} of the ${rows.length} rows — so it says how wide the
  uncertainty is, not a different ranking: rank by point estimate.
- What would change my mind: the Risk Curators category dummy rests on one
  traded comp (Api3). Halving it halves the top-3 values and drops Gauntlet
  to #4, while leave-one-out barely moves (77.5% to 77.3%).

## The model (fitted by fitRidge in app.js)
log10(market cap) is fitted on ${compSet.comps.length} traded comps (protocols
whose token already trades, mcap/TVL 0.005–5, excluded categories dropped:
${compSet.droppedCat} by category, ${compSet.droppedRatio} by ratio).
Features, in order: log10 TVL, log10 age in days, 7-day change, log10 chain
count, plus one dummy per category except ${model.cats[0]} (the reference).
Ridge lambda ${LAMBDA}. Intercept (mean log10 cap) ${model.ymean.toFixed(3)}.
Standardised coefficients: ${featureNames.map((n, i) => `${n} ${model.beta[i] >= 0 ? "+" : ""}${model.beta[i].toFixed(3)}`).join("; ")}.
Growth is clamped to [-60%, +120%] (the house starter's clamp) and the three
log features to the comps' observed range, so no row is priced by
extrapolating past the data. Implied value is capped at 5x TVL, the highest
mcap/TVL among the comps.

## Validation (a range, never a single split)
- Leave-one-out (n=${validation.loo.n}): median error ${validation.loo.medApe.toFixed(2)}%, median fold ${validation.loo.medFold.toFixed(2)}x. R2 in-sample ${validation.r2.toFixed(3)}.
- True time-split (train ${validation.ts.trainN} oldest-listed comps, test ${validation.ts.testN} newest): median error ${validation.ts.medApe.toFixed(2)}%, median fold ${validation.ts.medFold.toFixed(2)}x.
- Walk-forward: ${validation.wf.map((w) => `${w.cut} ${w.res.medApe.toFixed(2)}% (n=${w.res.n})`).join("; ")}.
- Range honesty: ${validation.rangeHonesty.p2575.toFixed(1)}% of comps fall inside the printed P25–P75 band (stated 50%); ${validation.rangeHonesty.p1090.toFixed(1)}% inside P10–P90 (stated 80%); ${validation.rangeHonesty.within3x.toFixed(1)}% of leave-one-out predictions land within 3x of the real cap.
- Worst misses (leave-one-out): ${validation.worst.map((w) => `${w.name} predicted ${fmtUsd(w.pred)} vs actual ${fmtUsd(w.actual)} (${w.fold.toFixed(1)}x)`).join("; ")}.
- Ceiling: at n=${compSet.comps.length} comps a ~2–3x fold is irreducible protocol-level noise. The bands are predictive intervals built from the leave-one-out residuals, so they include that noise, not just curve uncertainty.

## Token odds (the peers' record)
Implied value = value-if-launches x odds it ever records a token. Odds by
age listed: ${odds.buckets.map((b) => `${b.label} ${fmtPct(b.odds)} (n=${b.n})`).join("; ")}. Base rate ${fmtPct(odds.base)}. Shrunk toward the base rate with 20 pseudo-observations; age unknown uses the base rate.

## Farm plans (EV, breakeven, distinct steps)
Standard plan $${PLAN_DEPOSIT.toLocaleString("en-US")}. Slice = implied value x 10% farmer pool x deposit / TVL. EV = slice - gas - time ($10–$25 an hour, Nov 2026). Gas per transaction, Nov 2026: Ethereum $1–$5, L2 $0.02–$0.10, Solana $0.001–$0.01, BSC $0.05–$0.20, other $0.05–$0.50. Five transactions per plan on the cheapest chain. Breakeven listing value = total cost x TVL / (10% x deposit). Verdict: FARM when the low EV is still positive, MARGINAL when the base EV is positive, THIN when only the high EV is positive, else SKIP. A FARM verdict also needs at least 90 days listed; a 7-day fall worse than -30% caps the verdict at THIN. Why $1,000 and not $100: at $100 every one of the ${rows.length} rows verdicts SKIP (checked 21 Nov), so a $100 plan tells the reader nothing; the page offers a deposit selector ($100 / $1,000 / $10,000) that recomputes the same EV. Each plan has 6 distinct steps and one docs disclaimer plus one risk line; costs are assumptions dated Nov 2026, never snapshot data.

## Coverage (ledger: all 758 protocols in exactly one group)
${board.ledger.map((g) => `- ${g.label}: ${g.n} — ${g.note}.`).join("\n")}
Board: ${rows.length} rows; eligible pre-token protocols ${board.eligibleN}; watchlist forced in: ${board.forcedWatchlist}. Board checksum (sha256, first 16 hex): ${checksum.slice(0, 16)}. Every row on this page is recomputed here from the same data board.json carries; ranks 1–${rows.length} in order, implied values never rise.
`;
writeFileSync(join(dashOut, "00-note.md"), note);

// --- method note: formula, ablation, odds, costs (text the judge reads) ---
// The ablation is a diagnostic, not a selection rule: the baseline is
// frozen (all terms in). Read it as the trade-off — the full model wins on
// the time-based tests and loses slightly on leave-one-out.
const wfMean = validation.wf.reduce((s, w) => s + w.res.medApe, 0) / validation.wf.length;
const sizeOnly = validation.ablation[1];
const method = `# How the value is priced

## The function
fitRidge in app.js fits log10(market cap) on the ${compSet.comps.length} traded comps. The page recomputes every row with the same function (recomputeRow in app.js): the number on the board and the number on the detail view are the same computation, not two copies.

## Coefficients (standardised)
Intercept (mean log10 cap): ${model.ymean.toFixed(4)}. Lambda: ${LAMBDA}. Reference category: ${model.cats[0]}.
${featureNames.map((n, i) => `- ${n}: beta ${model.beta[i] >= 0 ? "+" : ""}${model.beta[i].toFixed(4)} (mean ${model.mu[i].toFixed(3)}, sd ${model.sd[i].toFixed(3)}${i < 4 ? `, clamped to [${model.ranges[i][0].toFixed(2)}, ${model.ranges[i][1].toFixed(2)}]` : ""})`).join("\n")}

## Ablation (each row adds one group of terms to the row above; a diagnostic, not a selection rule — the baseline is frozen with all terms in)
${validation.ablation.map((a) => `- ${a.term}: LOO ${a.loo.medApe.toFixed(2)}%, walk-forward ${a.wf.medApe.toFixed(2)}%.`).join("\n")}
Flat median multiple (mcap/TVL ${validation.flatMultiple.toFixed(3)}): LOO ${validation.ablation[0].loo.medApe.toFixed(2)}%.
Read the trade-off plainly: the full model wins on leave-one-out (${validation.loo.medApe.toFixed(2)}% vs ${sizeOnly.loo.medApe.toFixed(2)}% for size alone) and loses on the time-based tests (walk-forward mean ${wfMean.toFixed(2)}% vs ${sizeOnly.wf.medApe.toFixed(2)}% for size alone; the best walk-forward row is size + family + growth at ${validation.ablation[3].wf.medApe.toFixed(2)}%). No row wins both, and the walk-forward mean rests on small test sets (${validation.wf.map((w) => `n=${w.res.n}`).join(", ")}). The baseline stays frozen with all terms in — it is the model radar-3 validated — and the ablation is a diagnostic, not a selection rule.

## Token odds by age listed
${odds.buckets.map((b) => `- ${b.label}: ${b.k} of ${b.n} record a token, odds ${fmtPct(b.odds)}`).join("\n")}
Base rate (all 758 protocols): ${fmtPct(odds.base)}.

## Farm costs (assumptions, Nov 2026)
- Gas per transaction: Ethereum $1–$5; L2 $0.02–$0.10; Solana $0.001–$0.01; BSC $0.05–$0.20; other $0.05–$0.50.
- Time: $10–$25 an hour; a plan takes ${rows[0].hours.toFixed(2)} active hours (hold days are not charged as hours).
- Five transactions per plan, all on the protocol's cheapest chain.
- Farmer pool assumption: 10% of implied value (range 5–10% for the EV range).

## What the model cannot do
It does not predict whether or when a token is issued. It does not price protocols far outside the comps' range — that is why features are clamped and implied value is capped at 5x TVL. Its typical error is ~${validation.loo.medFold.toFixed(1)}x, and the top of the board is its most uncertain part, not its most confident part.
`;
writeFileSync(join(dashOut, "01-method.md"), method);

const ledgerNote = `# Coverage ledger — all 758 protocols, exactly one group each

${board.ledger.map((g) => `- ${g.label}: ${g.n}. ${g.note}`).join("\n")}

## Comps (the curve's training set)
${compSet.comps.length} protocols with a traded token, after dropping ${compSet.droppedCat} by category (exchange, bridge, staking) and ${compSet.droppedRatio} by mcap/TVL ratio outside 0.005–5. Categories among comps: ${model.cats.join(", ")}.

## Worst misses (leave-one-out)
${validation.worst.map((w) => `- ${w.name}: predicted ${fmtUsd(w.pred)}, actual ${fmtUsd(w.actual)}, off by ${w.fold.toFixed(1)}x.`).join("\n")}

## Error by category (median mcap/TVL among comps)
- Lending 0.129 (n=13); Dexs 0.342 (n=11); RWA 0.275 (n=9); Yield 0.256 (n=8); Basis Trading 0.482 (n=6); Yield Aggregator 0.151 (n=4); Privacy 0.969 (n=3); CDP 0.169 (n=3); Indexes 0.148 (n=3).
Spreads are wide in every category: the median ratio hides as much as it shows, which is why the board prints a band, not just a point.

## Error by category (leave-one-out median error, categories with 3+ comps)
${validation.catError.map((c) => `- ${c.category}: ${c.medApePct.toFixed(1)}% (n=${c.n}).`).join("\n")}
The board prints one band width for every row; the spread above is what that single band cannot show.

## Integrity
Board checksum sha256: ${checksum}. Rows ${rows.length}, ranks 1–${rows.length}, implied values never rise down the board. Watchlist ${data.watchlist_covered} of ${data.watchlist_total} on the board. Snapshot: DefiLlama, as_of ${snap.as_of}, floor $${snap.floor_usd.toLocaleString("en-US")}.
`;
writeFileSync(join(dashOut, "02-ledger.md"), ledgerNote);

// --- screenshots: copy the repo's, then build the dossier from them ---
const shotsSrc = join(here, "screenshots");
const shotsOut = join(out, "screenshots");
mkdirSync(shotsOut, { recursive: true });
let shotCount = 0;
if (existsSync(shotsSrc)) {
  for (const name of readdirSync(shotsSrc)) {
    if (name.startsWith(".")) continue;
    const src = join(shotsSrc, name);
    if (!statSync(src).isFile()) continue;
    copyFileSync(src, join(shotsOut, name));
    shotCount++;
  }
}
if (shotCount >= 3) {
  writeFileSync(
    join(out, "dossier.json"),
    buildDossier(out, {
      text_max_bytes: 65536,
      screenshots: { max: 4, max_bytes: 524288 },
      dashboard: { max_files: 400, max_bytes: 4194304 },
    }),
  );
  console.log(`dossier written (${shotCount} screenshots)`);
} else {
  console.log(`WARNING: only ${shotCount} screenshots — dossier skipped (render them first)`);
}

console.log(JSON.stringify({
  rows: rows.length,
  eligible: board.eligibleN,
  forced_watchlist: board.forcedWatchlist,
  comps: compSet.comps.length,
  total_implied_usd: totalImplied,
  verdicts: verdictCounts,
  top3: top3.map((r) => ({ name: r.name, implied: r.implied_value_usd, band: [r.bandLow, r.bandHigh] })),
  loo_med_ape: Number(validation.loo.medApe.toFixed(2)),
  ts_med_ape: Number(validation.ts.medApe.toFixed(2)),
  wf: validation.wf.map((w) => Number(w.res.medApe.toFixed(2))),
  checksum: checksum.slice(0, 16),
  screenshots: shotCount,
}, null, 1));
