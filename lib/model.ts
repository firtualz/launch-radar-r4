// Token Launch Radar r4 — model, validation, odds, farm plans.
// Written from scratch in this session from the frozen snapshot.
// One ridge model prices every row: the same function scores the board
// and the page (see recomputeRow in dashboard/app.js).
// All money figures below are the model's own claim, never make.fun's.

export interface Protocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: Protocol[];
  watchlist: string[];
}

export const REF_DATE = "2026-11-14";
export const LAMBDA = 30;
export const BOARD_N = 198;
export const PLAN_DEPOSIT = 1000; // dollars; see farm-plan note for why not $100

// Categories the house rule itself excludes from the launch race:
// an exchange, a bridge, or a staking product is not a launch candidate.
export const EXCL_CAT = new Set([
  "Bridge", "CEX", "Canonical Bridge", "Cross Chain Bridge",
  "Liquid Restaking", "Liquid Staking", "Restaked BTC",
]);

const CHG_LO = -60;
const CHG_HI = 120;

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function ageDays(p: Protocol): number | null {
  if (!p.listed_at) return null;
  const [y, m, d] = p.listed_at.split("-").map(Number);
  const listed = Date.UTC(y, m - 1, d);
  const ref = Date.UTC(2026, 10, 14);
  return Math.max(1, Math.round((ref - listed) / 86400000));
}

function isVersionedOrWrapped(name: string): boolean {
  return /\s(v2|V2|2\.0|3\.0)\s*$/.test(name) ||
    /\b(wrapped|staked|staking)\b/i.test(name);
}

/** A protocol is an eligible launch candidate when DefiLlama records no
 *  token for it, it is not an exchange/bridge/staking product, and its
 *  name is not a versioned or wrapped/staked variant. */
export function isEligible(p: Protocol): boolean {
  if (p.gecko_id !== null || p.symbol !== null) return false;
  if (p.category !== null && EXCL_CAT.has(p.category)) return false;
  if (isVersionedOrWrapped(p.name)) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Ridge regression (standardised features, intercept = mean of log target).
// Small dense linear algebra, written here so the entry needs no deps.
// ---------------------------------------------------------------------------

export interface RidgeModel {
  mu: number[];
  sd: number[];
  beta: number[];
  ymean: number;
  lam: number;
  cats: string[];
  ranges: [number, number][];
}

function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    // Pivot only among rows not yet used as a pivot (r >= col): swapping a
    // finished pivot row back in would reintroduce non-zeros it cleared.
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
    const div = M[col][col];
    if (Math.abs(div) < 1e-12) throw new Error("ridge: singular matrix");
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / div;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}

export function ridgeFit(X: number[][], y: number[], lam: number): {
  mu: number[]; sd: number[]; beta: number[]; ymean: number;
} {
  const n = X.length;
  const d = X[0].length;
  const mu = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const sd = Array.from({ length: d }, (_, j) => {
    const v = X.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / n;
    const s = Math.sqrt(v);
    return s === 0 ? 1 : s;
  });
  const ymean = y.reduce((s, v) => s + v, 0) / n;
  const A: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  const b: number[] = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    const xs = X[i].map((v, j) => (v - mu[j]) / sd[j]);
    const yc = y[i] - ymean;
    for (let a = 0; a < d; a++) {
      b[a] += xs[a] * yc;
      for (let c = 0; c < d; c++) A[a][c] += xs[a] * xs[c];
    }
  }
  for (let a = 0; a < d; a++) A[a][a] += lam;
  return { mu, sd, beta: solveLinear(A, b), ymean };
}

export function ridgePredict(
  fit: { mu: number[]; sd: number[]; beta: number[]; ymean: number },
  X: number[][],
): number[] {
  return X.map((row) => {
    let s = fit.ymean;
    for (let j = 0; j < row.length; j++) {
      s += ((row[j] - fit.mu[j]) / fit.sd[j]) * fit.beta[j];
    }
    return s;
  });
}

// ---------------------------------------------------------------------------
// Comps + features. A comp is a protocol whose token already trades, so its
// market cap is known: the curve is fitted on them, then read off for the
// pre-token rows. Categories with no usable cap and mcap/TVL ratios outside
// 0.005–5 are dropped (the ratio cut removes absurd multiples).
// ---------------------------------------------------------------------------

export interface CompSet {
  comps: Protocol[];
  droppedCat: number;
  droppedRatio: number;
  cats: string[];
}

export function buildComps(protocols: Protocol[]): CompSet {
  const withValue = protocols.filter((p) => p.mcap_usd !== null);
  const droppedCat = withValue.filter(
    (p) => p.category !== null && EXCL_CAT.has(p.category),
  ).length;
  const kept1 = withValue.filter(
    (p) => !(p.category !== null && EXCL_CAT.has(p.category)),
  );
  const droppedRatio = kept1.filter(
    (p) => !(p.mcap_usd! / p.tvl_usd >= 0.005 && p.mcap_usd! / p.tvl_usd <= 5),
  ).length;
  const comps = kept1.filter(
    (p) => p.mcap_usd! / p.tvl_usd >= 0.005 && p.mcap_usd! / p.tvl_usd <= 5,
  );
  const cats = [...new Set(comps.map((p) => p.category ?? "None"))].sort();
  return { comps, droppedCat, droppedRatio, cats };
}

/** Features, in order: log10 TVL, log10 age in days, 7-day change (fraction),
 *  log10 chain count, then one dummy per category except the first (reference).
 *  Growth is clamped to [-60%, +120%] (the house starter's own clamp) and the
 *  three log features are clamped to the comps' observed range, so no row is
 *  ever priced by extrapolating past the data that fitted the curve. */
export function featureRow(
  p: Protocol, cats: string[], ranges: [number, number][] | null,
): number[] {
  const age = ageDays(p) ?? 365;
  const chg = clamp(p.change_7d ?? 0, CHG_LO, CHG_HI) / 100;
  let vals = [
    Math.log10(Math.max(1, p.tvl_usd)),
    Math.log10(age),
    chg,
    Math.log10(Math.max(1, p.chains.length)),
  ];
  if (ranges) {
    vals = vals.map((v, i) => clamp(v, ranges[i][0], ranges[i][1]));
  }
  const cat = p.category ?? "None";
  for (const c of cats.slice(1)) vals.push(cat === c ? 1 : 0);
  return vals;
}

export function designMatrix(
  rows: Protocol[], cats: string[], ranges: [number, number][] | null,
): number[][] {
  return rows.map((p) => featureRow(p, cats, ranges));
}

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

export interface Apes { medApe: number; medFold: number; n: number }

export function apes(predUsd: number[], actualUsd: number[]): Apes {
  const a: number[] = [];
  const f: number[] = [];
  for (let i = 0; i < predUsd.length; i++) {
    a.push(Math.abs(predUsd[i] - actualUsd[i]) / actualUsd[i]);
    f.push(Math.max(predUsd[i], actualUsd[i]) / Math.min(predUsd[i], actualUsd[i]));
  }
  return { medApe: median(a) * 100, medFold: median(f), n: predUsd.length };
}

export interface Validation {
  r2: number;
  loo: Apes;
  looPredUsd: number[];
  ts: Apes & { trainN: number; testN: number };
  wf: { cut: string; res: Apes }[];
  ablation: { term: string; loo: Apes; wf: Apes; kept: boolean }[];
  worst: { name: string; pred: number; actual: number; fold: number }[];
  rangeHonesty: { p2575: number; p1090: number; within3x: number };
  flatMultiple: number;
  // Log10 leave-one-out residual quantiles (actual - LOO prediction): the
  // printed bands are each row's prediction plus these shifts.
  residQ: { q10: number; q25: number; q75: number; q90: number };
  // Leave-one-out median error by category (categories with 3+ comps).
  catError: { category: string; n: number; medApePct: number }[];
}

function looPredict(X: number[][], y: number[], lam: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < X.length; i++) {
    const Xtr = X.filter((_, j) => j !== i);
    const ytr = y.filter((_, j) => j !== i);
    const fit = ridgeFit(Xtr, ytr, lam);
    out.push(ridgePredict(fit, [X[i]])[0]);
  }
  return out;
}

export function validateModel(
  comps: Protocol[], cats: string[], ranges: [number, number][],
): Validation {
  const X = designMatrix(comps, cats, ranges);
  const y = comps.map((p) => Math.log10(p.mcap_usd!));
  const actual = comps.map((p) => p.mcap_usd!);
  const fit = ridgeFit(X, y, LAMBDA);
  const pred = ridgePredict(fit, X);
  const ssRes = pred.reduce((s, v, i) => s + (y[i] - v) ** 2, 0);
  const ssTot = y.reduce((s, v) => s + (v - y.reduce((a, b) => a + b, 0) / y.length) ** 2, 0);
  const r2 = 1 - ssRes / ssTot;

  const looLog = looPredict(X, y, LAMBDA);
  const loo = apes(looLog.map((v) => 10 ** v), actual);

  // True time-split: fit the comps listed longest ago, test the newest.
  const dated = comps
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => p.listed_at !== null)
    .sort((a, b) => (a.p.listed_at! < b.p.listed_at! ? -1 : 1));
  const trIdx = dated.slice(0, 42).map(({ i }) => i);
  const teIdx = dated.slice(-19).map(({ i }) => i);
  const fitTs = ridgeFit(trIdx.map((i) => X[i]), trIdx.map((i) => y[i]), LAMBDA);
  const tsPred = ridgePredict(fitTs, teIdx.map((i) => X[i])).map((v) => 10 ** v);
  const ts = {
    ...apes(tsPred, teIdx.map((i) => actual[i])),
    trainN: trIdx.length, testN: teIdx.length,
  };

  // Walk-forward across three cutoffs.
  const wf = ["2025-01-01", "2025-06-01", "2026-01-01"].map((cut) => {
    const tr = dated.filter(({ p }) => p.listed_at! < cut).map(({ i }) => i);
    const te = dated.filter(({ p }) => p.listed_at! >= cut).map(({ i }) => i);
    const f = ridgeFit(tr.map((i) => X[i]), tr.map((i) => y[i]), LAMBDA);
    const pr = ridgePredict(f, te.map((i) => X[i])).map((v) => 10 ** v);
    return { cut, res: apes(pr, te.map((i) => actual[i])) };
  });

  // Ablation: groups of terms added in order; a group is kept only when it
  // lowers BOTH the leave-one-out error and the walk-forward error (mean
  // of the three cutoffs). Feature indices: 0 log TVL, 1 log age, 2 7-day
  // change, 3 log chain count, 4.. the category dummies. Each row adds
  // exactly the group its label names — nothing more.
  const termNames = [
    "flat median multiple",
    "+ size curve (log TVL)",
    "+ family / category",
    "+ 7-day growth",
    "+ age (days listed)",
    "+ chain count",
  ];
  const dummyIdx = Array.from({ length: cats.length - 1 }, (_, j) => 4 + j);
  const groups: number[][] = [
    [],
    [0],
    [0, ...dummyIdx],
    [0, 2, ...dummyIdx],
    [0, 1, 2, ...dummyIdx],
    [0, 1, 2, 3, ...dummyIdx],
  ];
  const cuts = ["2025-01-01", "2025-06-01", "2026-01-01"];
  const evalSubset = (idxs: number[]): { loo: Apes; wf: Apes } => {
    const Xk = X.map((row) => idxs.map((j) => row[j]));
    const looLogK = looPredict(Xk, y, LAMBDA);
    const looK = apes(looLogK.map((v) => 10 ** v), actual);
    const perCut = cuts.map((cut) => {
      const tr = dated.filter(({ p }) => p.listed_at! < cut).map(({ i }) => i);
      const te = dated.filter(({ p }) => p.listed_at! >= cut).map(({ i }) => i);
      const f = ridgeFit(tr.map((i) => Xk[i]), tr.map((i) => y[i]), LAMBDA);
      const pr = ridgePredict(f, te.map((i) => Xk[i])).map((v) => 10 ** v);
      return apes(pr, te.map((i) => actual[i]));
    });
    const mean = (vs: number[]) => vs.reduce((s, v) => s + v, 0) / vs.length;
    return {
      loo: looK,
      wf: { medApe: mean(perCut.map((r) => r.medApe)), medFold: mean(perCut.map((r) => r.medFold)), n: y.length },
    };
  };
  const ablation: Validation["ablation"] = [];
  const flatMultiple = median(comps.map((p) => p.mcap_usd! / p.tvl_usd));
  let prevLoo = Infinity;
  let prevWf = Infinity;
  for (let k = 0; k <= 5; k++) {
    let looK: Apes;
    let wfK: Apes;
    if (k === 0) {
      const pr = comps.map((p) => flatMultiple * p.tvl_usd);
      looK = apes(pr, actual);
      // Walk-forward for the flat row, measured the same way as every
      // other row: fit the flat multiple on the train comps of each cut
      // only, then score the test comps. (It must NOT copy the LOO
      // number — that flattered the baseline by ~32 points.)
      const perCut = cuts.map((cut) => {
        const tr = dated.filter(({ p }) => p.listed_at! < cut);
        const te = dated.filter(({ p }) => p.listed_at! >= cut);
        const fm = median(tr.map(({ p }) => p.mcap_usd! / p.tvl_usd));
        return apes(te.map(({ p }) => fm * p.tvl_usd), te.map(({ p }) => p.mcap_usd!));
      });
      const mean = (vs: number[]) => vs.reduce((s, v) => s + v, 0) / vs.length;
      wfK = { medApe: mean(perCut.map((r) => r.medApe)), medFold: mean(perCut.map((r) => r.medFold)), n: y.length };
    } else {
      ({ loo: looK, wf: wfK } = evalSubset(groups[k]));
    }
    const kept = k === 0 ? true : looK.medApe < prevLoo && wfK.medApe < prevWf;
    ablation.push({ term: termNames[k], loo: looK, wf: wfK, kept });
    if (kept) { prevLoo = looK.medApe; prevWf = wfK.medApe; }
  }

  // Worst misses (leave-one-out, by fold error).
  const worst = looLog
    .map((v, i) => ({
      name: comps[i].name,
      pred: 10 ** v,
      actual: actual[i],
      fold: Math.max(10 ** v, actual[i]) / Math.min(10 ** v, actual[i]),
    }))
    .sort((a, b) => b.fold - a.fold)
    .slice(0, 3);

  // Range honesty: the printed bands are predictive intervals from the
  // leave-one-out residuals. For each comp, the interval is its LOO
  // prediction plus the residual quantiles; the quantiles for a comp
  // exclude that comp's own residual, so the measured shares below are a
  // genuine out-of-sample check of the printed bands.
  const res = y.map((v, i) => v - looLog[i]); // log10 actual - log10 LOO pred
  const sortedRes = [...res].sort((a, b) => a - b);
  const qAll = (p: number) => sortedRes[Math.min(sortedRes.length - 1, Math.floor(p * sortedRes.length))];
  const residQ = { q10: qAll(0.1), q25: qAll(0.25), q75: qAll(0.75), q90: qAll(0.9) };
  const qExcl = (exclude: number, p: number) => {
    const s = res.filter((_, i) => i !== exclude).sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(p * s.length))];
  };
  let in2575 = 0;
  let in1090 = 0;
  let in3x = 0;
  for (let i = 0; i < comps.length; i++) {
    const lo25 = looLog[i] + qExcl(i, 0.25);
    const hi25 = looLog[i] + qExcl(i, 0.75);
    const lo10 = looLog[i] + qExcl(i, 0.1);
    const hi10 = looLog[i] + qExcl(i, 0.9);
    if (y[i] >= lo25 && y[i] <= hi25) in2575++;
    if (y[i] >= lo10 && y[i] <= hi10) in1090++;
    const lp = 10 ** looLog[i];
    if (Math.max(lp, actual[i]) / Math.min(lp, actual[i]) <= 3) in3x++;
  }
  const rangeHonesty = {
    p2575: (in2575 / comps.length) * 100,
    p1090: (in1090 / comps.length) * 100,
    within3x: (in3x / comps.length) * 100,
  };
  // Error by category (leave-one-out, categories with 3+ comps): the
  // spread the board's single band cannot show.
  const byCat = new Map<string, number[]>();
  comps.forEach((p, i) => {
    const key = p.category ?? "None";
    const arr = byCat.get(key) ?? [];
    arr.push(Math.abs(10 ** looLog[i] - actual[i]) / actual[i]);
    byCat.set(key, arr);
  });
  const catError = [...byCat.entries()]
    .filter(([, v]) => v.length >= 3)
    .map(([category, v]) => ({ category, n: v.length, medApePct: median(v) * 100 }))
    .sort((a, b) => b.medApePct - a.medApePct);
  return { r2, loo, looPredUsd: looLog.map((v) => 10 ** v), ts, wf, ablation, worst, rangeHonesty, flatMultiple, residQ, catError };
}

// ---------------------------------------------------------------------------
// Token odds: the peers' record. For each age bucket, the share of protocols
// listed that long that record a token, shrunk toward the overall base rate
// (20 pseudo-observations). Age unknown: the base rate.
// ---------------------------------------------------------------------------

export interface OddsBucket {
  label: string; lo: number; hi: number; n: number; k: number; odds: number;
}

export function tokenOdds(protocols: Protocol[]): { buckets: OddsBucket[]; base: number } {
  const hasToken = (p: Protocol) => p.gecko_id !== null || p.symbol !== null;
  const base = protocols.filter(hasToken).length / protocols.length;
  const defs: [string, number, number][] = [
    ["under 6 months", 0, 180],
    ["6–12 months", 180, 365],
    ["1–2 years", 365, 730],
    ["2–3 years", 730, 1095],
    ["3–4 years", 1095, 1460],
    ["4+ years", 1460, 99999],
  ];
  const buckets: OddsBucket[] = defs.map(([label, lo, hi]) => {
    const sub = protocols.filter((p) => {
      const a = ageDays(p);
      return a !== null && a >= lo && a < hi;
    });
    const k = sub.filter(hasToken).length;
    const odds = (k + 20 * base) / (sub.length + 20);
    return { label, lo, hi, n: sub.length, k, odds };
  });
  return { buckets, base };
}

export function oddsFor(p: Protocol, odds: { buckets: OddsBucket[]; base: number }): number {
  const a = ageDays(p);
  if (a === null) return odds.base;
  const b = odds.buckets.find((bk) => a >= bk.lo && a < bk.hi);
  return b ? b.odds : odds.base;
}

// ---------------------------------------------------------------------------
// Farm economics. Assumptions, dated Nov 2026 — never snapshot data.
// A $1,000 plan: at $100 every one of the 198 rows verdicts SKIP (checked
// 21 Nov), so a $100 plan tells the reader nothing. The page also offers a
// deposit selector ($100 / $1,000 / $10,000) that recomputes the same EV.
// Slice = implied value x farmer pool (10%) x deposit / TVL: your share of
// the pool if the pool were split pro-rata to TVL. Time is charged at
// $10–$25 an hour; gas is priced per chain, Nov 2026.
// ---------------------------------------------------------------------------

export interface Gas { low: number; base: number; high: number }

const L2_CHAINS = new Set([
  "Arbitrum", "Optimism", "Base", "Ink", "Polygon", "Unichain", "Blast",
  "Linea", "Mode", "Corn", "Katana", "Robinhood Chain", "Arc", "Scroll",
  "Starknet", "zkSync", "Mantle", "Fraxtal",
]);

export function gasForChain(chain: string): Gas {
  const c = chain.toLowerCase();
  if (c.includes("solana")) return { low: 0.001, base: 0.005, high: 0.01 };
  if (c.includes("bsc") || c.includes("bnb") || c.includes("binance")) {
    return { low: 0.05, base: 0.1, high: 0.2 };
  }
  if (L2_CHAINS.has(chain)) return { low: 0.02, base: 0.05, high: 0.1 };
  if (c.includes("ethereum")) return { low: 1, base: 3, high: 5 };
  return { low: 0.05, base: 0.2, high: 0.5 };
}

export function cheapestChain(p: Protocol): string {
  return [...p.chains].sort((a, b) => gasForChain(a).base - gasForChain(b).base)[0] ??
    "Ethereum";
}

const ACTION_VERB: Record<string, string> = {
  "Risk Curators": "deposit into a curated vault",
  "Yield": "deposit into a yield vault",
  "Yield Aggregator": "deposit into a yield-aggregating vault",
  "Onchain Capital Allocator": "deposit into a capital-allocation vault",
  "Dexs": "swap half and add the rest as liquidity",
  "Lending": "supply as lending collateral",
  "CDP": "lock as collateral and mint against it",
  "RWA": "buy the tokenised asset",
  "Derivatives": "open a hedged derivatives position",
  "Chain": "stake with a validator",
  "Restaking": "restake with an operator",
  "Liquid Staking": "stake with a validator",
  "Privacy": "deposit into a shielded pool",
  "Indexes": "buy the index token",
  "Prediction Market": "take a position in an open market",
};

export function actionVerb(p: Protocol): string {
  return ACTION_VERB[p.category ?? ""] ?? "deposit into the protocol's main product";
}

function riskLine(p: Protocol): string {
  if (p.chains.length > 1) {
    return "Risk: activity on several chains from one wallet can trigger sybil filters; keep one wallet per chain.";
  }
  switch (p.category) {
    case "Lending": return "Risk: smart-contract risk plus liquidation risk if the collateral falls.";
    case "Dexs": return "Risk: impermanent loss plus smart-contract risk.";
    case "RWA": return "Risk: issuer and custody risk on top of smart-contract risk.";
    case "Derivatives": return "Risk: leverage can wipe the position; smart-contract risk on top.";
    case "CDP": return "Risk: liquidation risk if the collateral falls sharply.";
    default: return "Risk: smart-contract risk; the position can lose value while it is held.";
  }
}

export type Verdict = "FARM" | "MARGINAL" | "THIN" | "SKIP";

export interface FarmPlan {
  steps: string[];
  ev: number; evLow: number; evHigh: number;
  hours: number; gasLow: number; gasBase: number; gasHigh: number; costBase: number;
  breakeven: number; verdict: Verdict;
  deposit: number;
}

export function farmPlan(
  p: Protocol, implied: number, opts: { deposit?: number } = {},
): FarmPlan {
  const deposit = opts.deposit ?? PLAN_DEPOSIT;
  const chain = cheapestChain(p);
  const gas = gasForChain(chain);
  // Five transactions on the cheapest chain: fund, main action, exit test
  // out, exit test back in, claim.
  const gasLow = 5 * gas.low;
  const gasBase = 5 * gas.base;
  const gasHigh = 5 * gas.high;
  const minutes = 5 + 15 + 10 + 15 + 1; // steps 2,3,4,6 + verdict read
  const hours = minutes / 60;
  const timeLow = hours * 10;
  const timeBase = hours * 15;
  const timeHigh = hours * 25;
  const costLow = gasLow + timeLow;
  const costBase = gasBase + timeBase;
  const costHigh = gasHigh + timeHigh;
  const sliceBase = (implied * 0.1 * deposit) / p.tvl_usd;
  const sliceLow = (implied * 0.05 * deposit) / p.tvl_usd;
  const ev = sliceBase - costBase;
  const evLow = sliceLow - costHigh;
  const evHigh = sliceBase - costLow;
  const breakeven = (costBase * p.tvl_usd) / (0.1 * deposit);
  let verdict: Verdict =
    evLow > 0 ? "FARM" : ev > 0 ? "MARGINAL" : evHigh > 0 ? "THIN" : "SKIP";
  const age = ageDays(p);
  if (verdict === "FARM" && (age === null || age < 90)) verdict = "MARGINAL";
  if ((verdict === "FARM" || verdict === "MARGINAL") && (p.change_7d ?? 0) < -30) {
    verdict = "THIN";
  }
  const holdDays = age === null ? 45 : age < 365 ? 30 : age < 730 ? 45 : 60;
  const chg = p.change_7d === null ? "unreported" : `${p.change_7d.toFixed(1)}%`;
  // Same formatter as the dashboard's EV line (fmtSigned): one formatter
  // for one number everywhere, so the headline and the page can never
  // disagree by a trailing zero.
  const fmt$ = fmtSigned;
  const steps: string[] = [
    `EV ${fmt$(ev)} (range ${fmt$(evLow)} to ${fmt$(evHigh)}) per $${deposit} capital, ${hours.toFixed(2)} hours, net of ${fmt$(gasBase)} gas — pays for itself if the token lists at $${Math.round(breakeven).toLocaleString("en-US")} or more. Verdict: ${verdict}. Implied value $${Math.round(implied).toLocaleString("en-US")} on TVL $${Math.round(p.tvl_usd).toLocaleString("en-US")}, 7-day change ${chg}.`,
    `Use a fresh wallet for ${p.name}; keep the seed phrase offline. Fund it with $${deposit} plus about ${fmt$(gasBase)} of ${chain} gas (Nov 2026: ${fmt$(gas.low)} to ${fmt$(gas.high)} per transaction). Time: 5 minutes.`,
    `${cap1(actionVerb(p))} on ${chain} with the full $${deposit}. Snapshot data: ${p.name} holds $${Math.round(p.tvl_usd).toLocaleString("en-US")} TVL across ${p.chains.length} chain${p.chains.length > 1 ? "s" : ""}, listed ${age === null ? "on an unreported date" : age + " days ago"}, 7-day change ${chg}. Cost: about ${fmt$(gasBase)} of gas. Time: 15 minutes.`,
    `Two weeks later, test the exit path for ${p.name} on ${chain}: withdraw $50 of the position, then re-deposit it the same day and log both transaction hashes. The round trip costs about ${fmt$(2 * gasBase)} of gas on ${chain}; if the withdrawal fails or takes more than a day, stop here and do not add more capital. Time: 10 minutes.`,
    `Hold the position for at least ${holdDays} days. ${p.name} has run ${age === null ? "long enough" : age + " days"} with no token recorded, so ${holdDays} days is enough to register the activity. The $${deposit} stays in the position. Time: ${holdDays} days.`,
    `Claim any rewards only from the link ${p.name} itself publishes, then write the wallet, the dates and every transaction hash in a farm log. Costs priced Nov 2026; check the protocol's own docs first. ${riskLine(p)} Cost: about ${fmt$(gasBase)} of gas. Time: 15 minutes.`,
  ];
  return {
    steps, ev, evLow, evHigh, hours, gasLow, gasBase, gasHigh, costBase, breakeven, verdict, deposit,
  };
}

function cap1(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ---------------------------------------------------------------------------
// Board + ledger.
// ---------------------------------------------------------------------------

/** Fit the full model on the comps: feature ranges (for clamping), then ridge. */
export function fitModel(compSet: CompSet): RidgeModel {
  const raw = designMatrix(compSet.comps, compSet.cats, null);
  const ranges: [number, number][] = [0, 1, 2, 3].map((i) => {
    const col = raw.map((r) => r[i]);
    return [Math.min(...col), Math.max(...col)] as [number, number];
  });
  const X = designMatrix(compSet.comps, compSet.cats, ranges);
  const y = compSet.comps.map((p) => Math.log10(p.mcap_usd!));
  const fit = ridgeFit(X, y, LAMBDA);
  return { ...fit, lam: LAMBDA, cats: compSet.cats, ranges };
}

export interface BoardRow {
  slug: string;
  name: string;
  rank: number;
  implied_value_usd: number;
  farm_steps: string[];
  category: string;
  chains: string[];
  tvl: number;
  chg7d: number | null;
  age: number | null;
  odds: number;
  valueIf: number;
  bandLow: number;
  bandHigh: number;
  verdict: Verdict;
  ev: number; evLow: number; evHigh: number;
  netHour: number;
  gasLow: number; gasBase: number; gasHigh: number; costBase: number; breakeven: number;
  hours: number;
  watchlist: boolean;
}

export interface LedgerGroup { label: string; n: number; note: string }

export function buildBoard(
  snap: Snapshot, model: RidgeModel, odds: { buckets: OddsBucket[]; base: number },
  residQ: { q10: number; q25: number; q75: number; q90: number },
): { rows: BoardRow[]; ledger: LedgerGroup[]; eligibleN: number; forcedWatchlist: number } {
  const eligible = snap.protocols.filter(isEligible);
  const X = designMatrix(eligible, model.cats, model.ranges);
  const predLog = ridgePredict(
    { mu: model.mu, sd: model.sd, beta: model.beta, ymean: model.ymean }, X,
  );
  const cand: { p: Protocol; valueIf: number; odds: number; implied: number }[] =
    eligible.map((p, i) => {
      const valueIf = Math.min(10 ** predLog[i], 5 * p.tvl_usd);
      const o = oddsFor(p, odds);
      return { p, valueIf, odds: o, implied: Math.round(valueIf * o) };
    });
  cand.sort((a, b) => b.implied - a.implied || b.p.tvl_usd - a.p.tvl_usd);
  const watchlist = new Set(snap.watchlist);
  let rows = cand.slice(0, BOARD_N);
  const inBoard = new Set(rows.map((r) => r.p.slug));
  let forced = 0;
  for (const slug of snap.watchlist) {
    if (!inBoard.has(slug)) {
      const c = cand.find((r) => r.p.slug === slug);
      if (c) {
        rows[rows.length - 1] = c;
        inBoard.add(slug);
        forced++;
      }
    }
  }
  rows.sort((a, b) => b.implied - a.implied || b.p.tvl_usd - a.p.tvl_usd);

  // Predictive bands per board row: the row's log prediction plus the
  // leave-one-out residual quantiles (P10–P90), scaled by token odds and
  // capped at 5x TVL like the point estimate. One constant shift in log
  // space, so the band is the same log-width for every row — except the
  // top end is capped at 5x TVL, which shortens the band for rows near
  // the cap. It says how wide the uncertainty is, not a different ranking.
  const rowsWithBands = rows.map((r) => {
    const logValueIf = Math.log10(r.valueIf);
    const lo = Math.min(10 ** (logValueIf + residQ.q10), 5 * r.p.tvl_usd);
    const hi = Math.min(10 ** (logValueIf + residQ.q90), 5 * r.p.tvl_usd);
    return { ...r, bandLow: Math.round(lo * r.odds), bandHigh: Math.round(hi * r.odds) };
  });

  const boardRows: BoardRow[] = rowsWithBands.map((r, i) => {
    const plan = farmPlan(r.p, r.implied);
    return {
      slug: r.p.slug,
      name: r.p.name,
      rank: i + 1,
      implied_value_usd: r.implied,
      farm_steps: plan.steps,
      category: r.p.category ?? "None",
      chains: r.p.chains,
      tvl: r.p.tvl_usd,
      chg7d: r.p.change_7d,
      age: ageDays(r.p),
      odds: r.odds,
      valueIf: Math.round(r.valueIf),
      bandLow: r.bandLow,
      bandHigh: r.bandHigh,
      verdict: plan.verdict,
      ev: plan.ev, evLow: plan.evLow, evHigh: plan.evHigh,
      netHour: plan.ev / plan.hours,
      gasLow: plan.gasLow, gasBase: plan.gasBase, gasHigh: plan.gasHigh, costBase: plan.costBase,
      breakeven: plan.breakeven, hours: plan.hours,
      watchlist: watchlist.has(r.p.slug),
    };
  });

  // Ledger: every one of the 758 protocols in exactly one group.
  const boardSlugs = new Set(boardRows.map((r) => r.slug));
  const hasToken = (p: Protocol) => p.gecko_id !== null || p.symbol !== null;
  const compSlugs = new Set(
    buildComps(snap.protocols).comps.map((p) => p.slug),
  );
  const groups: LedgerGroup[] = [
    { label: "Ranked on the board", n: boardRows.length, note: "pre-token, ranked by implied value" },
    {
      label: "Eligible, not ranked", n: eligible.filter((p) => !boardSlugs.has(p.slug)).length,
      note: "pre-token but below the board cut",
    },
    {
      label: "Token trades, priced comp", n: [...compSlugs].filter((s) => !boardSlugs.has(s)).length,
      note: "their market caps fit the curve",
    },
    {
      label: "Token recorded, no usable cap",
      n: snap.protocols.filter((p) => hasToken(p) && p.mcap_usd === null).length,
      note: "a token is recorded but no market cap is listed",
    },
    {
      label: "Token trades, dropped from comps",
      n: snap.protocols.filter((p) => hasToken(p) && p.mcap_usd !== null && !compSlugs.has(p.slug)).length,
      note: "excluded category or mcap/TVL outside 0.005–5",
    },
    {
      label: "No token, excluded category",
      n: snap.protocols.filter((p) => !hasToken(p) && p.category !== null && EXCL_CAT.has(p.category)).length,
      note: "exchange, bridge or staking product",
    },
    {
      label: "No token, versioned or wrapped",
      n: snap.protocols.filter((p) => !hasToken(p) && !(p.category !== null && EXCL_CAT.has(p.category)) && isVersionedOrWrapped(p.name)).length,
      note: "a versioned release or a wrapped/staked variant",
    },
  ];
  const total = groups.reduce((s, g) => s + g.n, 0);
  if (total !== snap.protocols.length) {
    throw new Error(`ledger does not sum to 758: ${total}`);
  }
  return { rows: boardRows, ledger: groups, eligibleN: eligible.length, forcedWatchlist: forced };
}

// ---------------------------------------------------------------------------
// Formatting helpers shared by the note, the dashboard and the page.
// ---------------------------------------------------------------------------

export function fmtUsd(v: number): string {
  if (v >= 1e9) return "$" + (v / 1e9).toFixed(2) + "B";
  if (v >= 1e6) return "$" + (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return "$" + (v / 1e3).toFixed(1) + "K";
  return "$" + v.toFixed(0);
}

export function fmtSigned(v: number): string {
  const a = Math.abs(v);
  const s = a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : a.toFixed(2);
  return (v < 0 ? "-$" : "$") + s;
}

export function fmtPct(v: number, digits = 1): string {
  return (v * 100).toFixed(digits) + "%";
}
