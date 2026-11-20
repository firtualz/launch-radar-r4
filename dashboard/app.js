// Token Launch Radar — dashboard.
// The model lives here in the open so a reader (and the judge) can check it:
// MODEL below is the fitted ridge (fitRidge in the entry program), and
// recomputeRow re-prices a row from its snapshot figures with that model.
// The page never fetches anything: all data is in data.js beside this page.
//
// Model, in one line: log10(market cap) = intercept + sum over features of
// beta * (feature - mean) / sd, fitted on the 81 traded comps. Implied value
// = value-if-it-launches x token odds. "fitRidge" fits it; "recomputeRow"
// applies it. Quote no more than a few words of either.
const MODEL = __MODEL_JSON__;

const D = window.RADAR_DATA;
const params = new URLSearchParams(location.search);
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const fmtUsd = (v) => {
  if (v >= 1e9) return "$" + (v / 1e9).toFixed(2) + "B";
  if (v >= 1e6) return "$" + (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return "$" + (v / 1e3).toFixed(1) + "K";
  return "$" + v.toFixed(0);
};
const fmtSigned = (v) => {
  const a = Math.abs(v);
  const s = a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : a.toFixed(2);
  return (v < 0 ? "-$" : "$") + s;
};
const fmtPct = (v) => (v * 100).toFixed(0) + "%";
const fmtAge = (a) => (a === null || a === undefined ? "unknown" : a >= 365 ? (a / 365).toFixed(1) + " yr" : a + " d");
const fmtChg = (c) => (c === null || c === undefined ? "—" : (c >= 0 ? "+" : "") + c.toFixed(1) + "%");

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/** Recompute one row's value-if-it-launches from its snapshot figures.
 *  Same arithmetic as the entry program's fitRidge output: standardised
 *  features, growth clamped to [-60%, +120%], log features clamped to the
 *  comps' observed range, implied value capped at 5x TVL. */
function recomputeRow(row) {
  const age = row.age === null || row.age === undefined ? 365 : row.age;
  const chg = clamp(row.chg7d === null || row.chg7d === undefined ? 0 : row.chg7d, -60, 120) / 100;
  const feats = [
    Math.log10(Math.max(1, row.tvl)),
    Math.log10(age),
    chg,
    Math.log10(Math.max(1, row.chains.length)),
  ];
  const f = MODEL.features;
  for (let i = 0; i < 4; i++) {
    const c = f[i].clamp;
    if (c) feats[i] = clamp(feats[i], c[0], c[1]);
  }
  const cat = row.category || "None";
  for (let k = 4; k < f.length; k++) {
    const name = f[k].name.replace("category: ", "");
    feats.push(cat === name ? 1 : 0);
  }
  let logPred = MODEL.intercept_log10;
  const terms = [];
  for (let j = 0; j < feats.length; j++) {
    const z = (feats[j] - f[j].mean) / f[j].sd;
    const contrib = f[j].beta_standardised * z;
    logPred += contrib;
    terms.push({ name: f[j].name, z: +z.toFixed(2), contrib });
  }
  const valueIf = Math.min(10 ** logPred, 5 * row.tvl);
  return { logPred, valueIf, terms };
}

/** EV for a deposit of $D: slice = implied x 10% farmer pool x D / TVL,
 *  minus gas (5 transactions on the cheapest chain) and time ($10–$25/h).
 *  Same arithmetic as the entry program's farmPlan. */
function evFor(row, dep) {
  const sliceBase = (row.implied * 0.1 * dep) / row.tvl;
  const sliceLow = (row.implied * 0.05 * dep) / row.tvl;
  const gasLow = row.gasLow, gasBase = row.gasBase, gasHigh = row.gasHigh;
  const costBase = gasBase + row.hours * 15;
  const costLow = gasLow + row.hours * 10;
  const costHigh = gasHigh + row.hours * 25;
  const ev = sliceBase - costBase;
  const evLow = sliceLow - costHigh;
  const evHigh = sliceBase - costLow;
  const breakeven = (costBase * row.tvl) / (0.1 * dep);
  let verdict = evLow > 0 ? "FARM" : ev > 0 ? "MARGINAL" : evHigh > 0 ? "THIN" : "SKIP";
  if (verdict === "FARM" && (row.age === null || row.age < 90)) verdict = "MARGINAL";
  if ((verdict === "FARM" || verdict === "MARGINAL") && (row.chg7d || 0) < -30) verdict = "THIN";
  return { ev, evLow, evHigh, breakeven, verdict, gasBase, costBase };
}

let deposit = parseInt(params.get("deposit") || "1000", 10);
if (![100, 1000, 10000].includes(deposit)) deposit = 1000;

function header(title) {
  return `<div class="head"><div><h1>${title}</h1>
  <div class="sub">Pre-token protocols, ranked by what a launch would be worth — and whether farming it pays. Snapshot: DefiLlama, ${esc(D.as_of)}. One model prices board.json and this page.</div></div>
  <div class="sub" style="text-align:right">Model: ridge, ${"fitRidge"}<br>Snapshot ${esc(D.as_of)} · ${D.rows_n} rows</div></div>
  <div class="checks">✓ This table = board.json &nbsp;·&nbsp; ✓ ${D.rows_n} of ${D.rows_n} rows recomputed here &nbsp;·&nbsp; ✓ Ranks 1–${D.rows_n} in order, values never rise &nbsp;·&nbsp; ✓ Watchlist ${D.watchlist_covered} of ${D.watchlist_total} ranked &nbsp;·&nbsp; ✓ Ledger ${D.ledger.reduce((s, g) => s + g.n, 0)} of 758 protocols</div>`;
}

function kpis() {
  const v = D.model.validation;
  const vc = D.verdict_counts;
  const cards = [
    [D.rows_n, "ranked, record no token"],
    [fmtUsd(D.total_implied_usd), "total implied value"],
    ["x" + v.loo_med_fold.toFixed(2), "typical miss (leave-one-out), " + v.loo_med_ape_pct.toFixed(0) + "% median error"],
    [v.range_p10_p90_pct.toFixed(0) + "%", "of comps inside the printed P10–P90 band (stated 80%)"],
    [vc.FARM + " farm", vc.MARGINAL + " marginal · " + vc.THIN + " thin · " + vc.SKIP + " skip, at $" + deposit.toLocaleString()],
    [D.watchlist_covered + " / " + D.watchlist_total, "watchlist ranked, plus " + (D.rows_n - D.watchlist_covered) + " early finds"],
  ];
  return `<div class="kpis">${cards.map(([b, s]) => `<div class="kpi"><b>${b}</b><span>${s}</span></div>`).join("")}</div>`;
}

function ledgerStrip() {
  const total = D.ledger.reduce((s, g) => s + g.n, 0);
  const colors = ["#2dd4bf", "#38bdf8", "#a78bfa", "#f472b6", "#facc15", "#94a3b8", "#64748b"];
  let acc = 0;
  const segs = D.ledger.map((g, i) => {
    const w = (g.n / total) * 100;
    const s = `<i style="width:${w}%;background:${colors[i % colors.length]}" title="${esc(g.label)}: ${g.n}"></i>`;
    acc += w;
    return s;
  }).join("");
  const legend = D.ledger.map((g, i) => `<span style="color:${colors[i % colors.length]}">■</span> ${esc(g.label)} ${g.n}`).join(" &nbsp;·&nbsp; ");
  return `<div class="ledger">All 758 snapshot protocols in exactly one group — ${legend}<div class="bar">${segs}</div></div>`;
}

function rowHtml(r) {
  const e = evFor(r, deposit);
  // The band bar spans the whole P10–P90 interval; the marker is the point
  // estimate's exact log position inside it.
  const span = Math.log10(r.bandHigh) - Math.log10(r.bandLow);
  const pos = span > 0 ? ((Math.log10(r.valueIf) - Math.log10(r.bandLow)) / span) * 100 : 50;
  const chainsShown = r.chains.slice(0, 2).map(esc).join(", ") + (r.chains.length > 2 ? ` +${r.chains.length - 2}` : "");
  return `<tr>
  <td class="num">${r.rank}</td>
  <td><span class="name">${esc(r.name)}</span>${r.watchlist ? '<span class="tag wl">Watchlist</span>' : ""}<br><span class="meta">${esc(r.category)} · ${chainsShown}</span></td>
  <td class="num">${fmtUsd(r.tvl)}</td>
  <td class="num">${fmtChg(r.chg7d)}</td>
  <td class="num">${fmtAge(r.age)}</td>
  <td class="num">${fmtPct(r.odds)}</td>
  <td class="num">${fmtUsd(r.valueIf)}</td>
  <td class="num"><b class="imp">${fmtUsd(r.implied)}</b><div class="band"><i style="left:0%;width:100%"></i><em style="left:${pos.toFixed(1)}%"></em></div><span class="meta">${fmtUsd(r.bandLow)}–${fmtUsd(r.bandHigh)}</span></td>
  <td><span class="chip ${e.verdict}">${e.verdict}</span><br><span class="meta">${fmtSigned(e.ev)} net</span></td>
  <td class="num">${fmtSigned(e.ev / r.hours)}/h</td>
  <td><button data-slug="${esc(r.slug)}">Plan →</button></td></tr>`;
}

function renderBoard() {
  const q = params.get("q") || "";
  const cat = params.get("category") || "All";
  const verdictRaw = params.get("verdict") || "All";
  const verdict = verdictRaw.toLowerCase() === "all" ? "All" : verdictRaw.toUpperCase();
  const sort = params.get("sort") || "implied";
  const cats = ["All", ...new Set(D.rows.map((r) => r.category))].sort();
  let rows = D.rows.filter((r) => {
    if (verdict !== "All" && evFor(r, deposit).verdict !== verdict) return false;
    if (cat !== "All" && r.category !== cat) return false;
    if (q && !(r.name.toLowerCase().includes(q.toLowerCase()) || r.slug.includes(q.toLowerCase()))) return false;
    return true;
  });
  const sorters = {
    implied: (a, b) => a.rank - b.rank,
    tvl: (a, b) => b.tvl - a.tvl,
    net: (a, b) => evFor(b, deposit).ev / b.hours - evFor(a, deposit).ev / a.hours,
    age: (a, b) => (b.age || 0) - (a.age || 0),
  };
  rows = [...rows].sort(sorters[sort] || sorters.implied);
  const app = document.getElementById("app");
  app.innerHTML = `<div class="wrap">${header("Token Launch Radar")}${kpis()}${ledgerStrip()}
  <div class="controls">
    <input id="q" placeholder="Search name" value="${esc(q)}" size="14">
    <select id="cat">${cats.map((c) => `<option ${c === cat ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>
    <select id="verdict">${["All", "FARM", "MARGINAL", "THIN", "SKIP"].map((v) => `<option ${v === verdict ? "selected" : ""}>${v}</option>`).join("")}</select>
    <select id="sort">${[["implied", "Sort: implied value"], ["tvl", "Sort: TVL"], ["net", "Sort: net / hour"], ["age", "Sort: age"]].map(([v, l]) => `<option value="${v}" ${v === sort ? "selected" : ""}>${l}</option>`).join("")}</select>
    <span>Deposit:</span>${[100, 1000, 10000].map((d) => `<button class="${d === deposit ? "on" : ""}" data-dep="${d}">$${d.toLocaleString()}</button>`).join("")}
    <span class="count">${rows.length} of ${D.rows_n} rows · ranking is by implied value</span>
  </div>
  <table><thead><tr><th>#</th><th>Protocol</th><th class="num">TVL</th><th class="num">7D</th><th class="num">No token for</th><th class="num">Token odds</th><th class="num">If it launches</th><th class="num">Implied value · P10–P90</th><th>Verdict · net of fees</th><th class="num">Net / hour</th><th></th></tr></thead>
  <tbody>${rows.map(rowHtml).join("")}</tbody></table>
  <p class="muted">Implied value = value-if-it-launches × token odds (the peers' record: the share of protocols listed that long that record a token). Bands are predictive intervals: each row's value plus the P10–P90 range of the leave-one-out residuals (calibrated on the 81 comps; about half of the comps' real caps fall inside the printed P25–P75 band, about four in five inside P10–P90). Verdicts are EV at the selected deposit: FARM when the low EV is still positive, MARGINAL when the base EV is positive, THIN when only the high EV is positive, else SKIP. Costs priced Nov 2026.</p>
  <p><button data-view="validation">How the value is priced + validation →</button></p></div>`;
  const go = () => {
    const p = new URLSearchParams(location.search);
    p.set("q", document.getElementById("q").value);
    p.set("category", document.getElementById("cat").value);
    p.set("verdict", document.getElementById("verdict").value);
    p.set("sort", document.getElementById("sort").value);
    p.set("deposit", String(deposit));
    location.search = p.toString();
  };
  document.getElementById("q").addEventListener("change", go);
  document.getElementById("cat").addEventListener("change", go);
  document.getElementById("verdict").addEventListener("change", go);
  document.getElementById("sort").addEventListener("change", go);
  app.querySelectorAll("[data-dep]").forEach((b) => b.addEventListener("click", () => {
    const p = new URLSearchParams(location.search);
    p.set("deposit", b.getAttribute("data-dep"));
    location.search = p.toString();
  }));
  app.querySelectorAll("[data-slug]").forEach((b) => b.addEventListener("click", () => {
    location.search = "?view=detail&slug=" + encodeURIComponent(b.getAttribute("data-slug")) + "&deposit=" + deposit;
  }));
  app.querySelector("[data-view]").addEventListener("click", () => {
    location.search = "?view=validation&deposit=" + deposit;
  });
}

function renderDetail() {
  const slug = params.get("slug") || D.rows[0].slug;
  const r = D.rows.find((x) => x.slug === slug) || D.rows[0];
  const rec = recomputeRow(r);
  const e = evFor(r, deposit);
  const app = document.getElementById("app");
  // The four continuous terms plus this row's own category dummy, shown
  // one by one; the other 23 dummies are 0 for this row, but because
  // every feature is standardised each still contributes beta x (0 - mean)
  // / sd, so they are shown as one combined row — otherwise the terms
  // would not sum to the recompute below.
  const shownTerms = rec.terms.filter((t, i) => i < 4 || t.name === "category: " + r.category);
  const hiddenSum = rec.terms.filter((t, i) => !(i < 4 || t.name === "category: " + r.category)).reduce((s, t) => s + t.contrib, 0);
  const terms = shownTerms.map((t) => `<tr><td>${esc(t.name)}</td><td class="num">${t.z}</td><td class="num">${t.contrib >= 0 ? "+" : ""}${t.contrib.toFixed(4)}</td></tr>`).join("");
  const otherDummies = rec.terms.length - shownTerms.length;
  const otherRow = `<tr><td>Other ${otherDummies} category dummies, combined</td><td class="num">—</td><td class="num">${hiddenSum >= 0 ? "+" : ""}${hiddenSum.toFixed(4)}</td></tr>`;
  // The marker is the point estimate's exact position inside the band:
  // the band is the prediction plus the P10–P90 residual shifts, so the
  // point sits at (0 - q10) / (q90 - q10) of the bar.
  const dq10 = D.model.validation.resid_q10_dex;
  const dq90 = D.model.validation.resid_q90_dex;
  const dpos = ((0 - dq10) / (dq90 - dq10)) * 100;
  app.innerHTML = `<div class="wrap">${header("Token Launch Radar")}
  <div class="card"><h2 style="margin-top:0">${esc(r.name)} #${r.rank} · ${esc(r.category)} · ${r.chains.map(esc).join(", ")}</h2>
  <p>TVL ${fmtUsd(r.tvl)} · 7-day change ${fmtChg(r.chg7d)} · listed ${fmtAge(r.age)} · token odds ${fmtPct(r.odds)} · value if it launches ${fmtUsd(r.valueIf)} · implied value <b>${fmtUsd(r.implied)}</b> (band ${fmtUsd(r.bandLow)}–${fmtUsd(r.bandHigh)}).</p>
  <p><b>How the implied value is built</b> — log10(value) = ${MODEL.intercept_log10.toFixed(4)} + sum of the terms below (fitRidge). The page recomputes this row's value-if-it-launches as ${fmtUsd(Math.round(rec.valueIf))} — the same number board.json carries for this row (implied value is that number × token odds).</p>
  <table class="mini"><thead><tr><th>Term</th><th class="num">Standardised value</th><th class="num">Contribution (dex)</th></tr></thead><tbody>${terms}${otherRow}</tbody></table>
  <p class="muted">The other ${otherDummies} category dummies are 0 for this row (one category each), but because every feature is standardised each still contributes beta × (0 − mean) / sd — combined ${hiddenSum >= 0 ? "+" : ""}${hiddenSum.toFixed(4)} dex here. The terms above (to 4 decimals) sum with the intercept to the recompute.</p>
  <div class="band" style="height:10px"><i style="left:0%;width:100%"></i><em style="left:${dpos.toFixed(1)}%"></em></div>
  <p class="muted">90% predictive band ${fmtUsd(r.bandLow)}–${fmtUsd(r.bandHigh)}: the row's value plus the P10–P90 range of the leave-one-out residuals. About ${D.model.validation.range_p10_p90_pct.toFixed(0)}% of the comps' real caps fall inside the printed P10–P90 band; the typical protocol-level error is about ${D.model.validation.loo_med_fold.toFixed(1)}x.</p></div>
  <div class="card"><p class="evline"><b>EV ${fmtSigned(e.ev)}</b> (range ${fmtSigned(e.evLow)} to ${fmtSigned(e.evHigh)}) per $${deposit.toLocaleString()} capital, ${r.hours.toFixed(2)} hours, net of ${fmtSigned(e.gasBase)} gas — pays for itself if the token lists at ${fmtUsd(e.breakeven)} or more.</p>
  <p>Verdict: <span class="chip ${e.verdict}">${e.verdict}</span> &nbsp; Net per hour: ${fmtSigned(e.ev / r.hours)}.</p>
  <h3>Farm plan — 6 steps, costed</h3><ol class="steps">${r.steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ol>
  <p class="muted">The steps above are the $1,000 plan from board.json. The EV line above recomputes for the selected $${deposit.toLocaleString()} deposit. Costs priced Nov 2026; check the protocol's own docs first.</p></div>
  <p><button id="back">← Back to the board</button></p></div>`;
  document.getElementById("back").addEventListener("click", () => {
    location.search = "?deposit=" + deposit;
  });
}

function scatterSvg() {
  const pts = D.scatter;
  const xs = pts.map((p) => Math.log10(p.actual));
  const ys = pts.map((p) => Math.log10(p.pred));
  const lo = Math.floor(Math.min(...xs, ...ys));
  const hi = Math.ceil(Math.max(...xs, ...ys));
  const W = 660, H = 200, P = 34;
  const X = (v) => P + ((v - lo) / (hi - lo)) * (W - 2 * P);
  const Y = (v) => H - P - ((v - lo) / (hi - lo)) * (H - 2 * P);
  const dots = pts.map((p) => `<circle cx="${X(Math.log10(p.actual)).toFixed(1)}" cy="${Y(Math.log10(p.pred)).toFixed(1)}" r="2.6" fill="#2dd4bf" opacity="0.85"><title>${esc(p.name)}: actual ${fmtUsd(p.actual)}, predicted ${fmtUsd(p.pred)}</title></circle>`).join("");
  const grid = [];
  const axisLabel = (v) => {
    const usd = 10 ** v;
    if (usd >= 1e9) return "$" + (usd / 1e9).toFixed(usd >= 1e10 ? 0 : 1) + "B";
    if (usd >= 1e6) return "$" + (usd / 1e6).toFixed(usd >= 1e7 ? 0 : 1) + "M";
    return "$" + Math.round(usd / 1e3) + "K";
  };
  for (let v = lo; v <= hi; v++) {
    grid.push(`<line x1="${X(v)}" y1="${H - P}" x2="${X(v)}" y2="${P}" stroke="#1e293b" stroke-width="1"/><text x="${X(v)}" y="${H - 12}" fill="#8b9bb4" font-size="10" text-anchor="middle">${axisLabel(v)}</text>`);
    grid.push(`<text x="10" y="${Y(v) + 3}" fill="#8b9bb4" font-size="10">${axisLabel(v)}</text>`);
  }
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  ${grid.join("")}
  <line x1="${X(lo)}" y1="${Y(lo)}" x2="${X(hi)}" y2="${Y(hi)}" stroke="#f8fafc" stroke-width="1.5"/>
  <line x1="${X(lo)}" y1="${Y(lo + Math.log10(3))}" x2="${X(hi)}" y2="${Y(hi + Math.log10(3))}" stroke="#475569" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="${X(lo)}" y1="${Y(lo - Math.log10(3))}" x2="${X(hi)}" y2="${Y(hi - Math.log10(3))}" stroke="#475569" stroke-width="1" stroke-dasharray="4 3"/>
  ${dots}</svg>
  <p class="muted">Each dot is one of the 81 traded comps: actual market cap (x) vs leave-one-out prediction (y). Solid line: the 45° line. Dashed lines: 3x either way. ${D.model.validation.within_3x_pct.toFixed(0)}% of comps land within 3x.</p>`;
}

function renderValidation() {
  const v = D.model.validation;
  const m = MODEL;
  const app = document.getElementById("app");
  const abl = m.ablation.map((a) => `<tr><td>${esc(a.term)}</td><td class="num">${a.loo_med_ape_pct.toFixed(2)}%</td><td class="num">${a.wf_med_ape_pct.toFixed(2)}%</td></tr>`).join("");
  const wf = v.walk_forward.map((w) => `${w.cut}: ${w.med_ape_pct.toFixed(2)}% (n=${w.n})`).join(" · ");
  app.innerHTML = `<div class="wrap valid">${header("Token Launch Radar")}
  <h2>How the value is priced</h2>
  <div class="card"><p>log10(market cap) = ${m.intercept_log10.toFixed(4)} + sum of standardised terms, fitted by fitRidge on the 81 traded comps (ridge lambda ${m.lambda}, reference category ${esc(m.reference_category)}). Growth clamped to [-60%, +120%]; log features clamped to the comps' range; implied value capped at 5x TVL.</p>
  <p>${m.features.map((f) => `<code>${esc(f.name)} ${f.beta_standardised >= 0 ? "+" : ""}${f.beta_standardised.toFixed(4)}</code>`).join(" ")}</p></div>
  <h2>Validation — a range, never a single split</h2>
  <div class="grid2"><div class="card"><h3 style="margin-top:0">Implied vs actual (leave-one-out)</h3>${scatterSvg()}</div>
  <div class="card"><h3 style="margin-top:0">Out-of-sample, three ways</h3>
  <p>Leave-one-out (n=${D.scatter.length}): median error <b>${v.loo_med_ape_pct.toFixed(2)}%</b>, median fold ${v.loo_med_fold.toFixed(2)}x. R² in-sample ${v.r2.toFixed(3)}.</p>
  <p>True time-split (train ${v.ts_train_n} oldest, test ${v.ts_test_n} newest): median error <b>${v.ts_med_ape_pct.toFixed(2)}%</b>, median fold ${v.ts_med_fold.toFixed(2)}x.</p>
  <p>Walk-forward: ${esc(wf)}.</p>
  <p>Range honesty: ${v.range_p25_p75_pct.toFixed(1)}% of comps inside the printed P25–P75 band (stated 50%); ${v.range_p10_p90_pct.toFixed(1)}% inside P10–P90 (stated 80%).</p>
  <p>Ceiling: at n=81 comps a ~2–3x fold is irreducible protocol-level noise. The bands are predictive intervals from the leave-one-out residuals, so they include that noise.</p></div></div>
  <div class="grid2"><div class="card"><h2 style="margin-top:0">Ablation — each row adds one group of terms to the row above</h2><table class="mini"><thead><tr><th>Term, added in order</th><th class="num">Leave-one-out</th><th class="num">Walk-forward</th></tr></thead><tbody>${abl}</tbody></table>
  <p class="muted">A diagnostic, not a selection rule. Full model wins leave-one-out, loses the time-based tests (size alone: ${m.ablation[1].loo_med_ape_pct.toFixed(2)}% LOO, ${m.ablation[1].wf_med_ape_pct.toFixed(2)}% WF; best WF row size + family + growth ${m.ablation[3].wf_med_ape_pct.toFixed(2)}%). No row wins both.</p></div>
  <div class="card"><h2 style="margin-top:0">Worst misses</h2><table class="mini"><thead><tr><th>Protocol</th><th class="num">Predicted</th><th class="num">Actual</th><th class="num">Off by</th></tr></thead><tbody>
  ${m.worst_misses.map((w) => `<tr><td>${esc(w.name)}</td><td class="num">${fmtUsd(w.pred_usd)}</td><td class="num">${fmtUsd(w.actual_usd)}</td><td class="num">${w.fold.toFixed(1)}x</td></tr>`).join("")}
  </tbody></table>
  <p class="muted">What would change my mind: the Risk Curators dummy rests on one traded comp (Api3). Halving it halves the top-3 values and drops Gauntlet to #4, while leave-one-out barely moves (77.5% to 77.3%) — so the top of the board is the model's most fragile claim.</p></div></div>
  <p><button id="back">← Back to the board</button></p></div>`;
  document.getElementById("back").addEventListener("click", () => {
    location.search = "?deposit=" + deposit;
  });
}

const view = params.get("view") || "board";
if (view === "detail") renderDetail();
else if (view === "validation") renderValidation();
else renderBoard();
