/* The opening beats: one attempt · the branch · mapping every path · what we can afford · but they share states ·
   every rollout strengthens the map. Everything but the schematic tree comes from the loaded task's rollouts. */
(() => {
const W = 900, H = 440, PASS = "#2a9d4a", FAIL = "#d13b3b", INK = "#1c1c1c", MUTED = "#8a8a86", ACT = "#2b5fb3", OBS = "#8a5a00", ACT_BG = "#e8eefc", OBS_BG = "#f7ecd9", NOISE_STROKE = "#7b3fb3";
const srColor = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);
const clamp01 = (x) => Math.max(0, Math.min(1, x)), sub = (q, a, b) => clamp01((q - a) / (b - a)), lerp = (a, b, t) => a + (b - a) * t, ease = d3.easeCubicInOut;
const hash = (a, b, c) => ((Math.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453) % 1 + 1) % 1;
const TOOL_ICON = [[/find_user|get_user/, "id-card"], [/get_order/, "package"], [/get_product|list_all_product/, "tag"], [/return_/, "undo-2"], [/exchange_/, "repeat"], [/modify_.*address/, "map-pin"], [/modify_/, "pencil"], [/cancel_/, "ban"], [/transfer_to_human/, "hand-helping"], [/calculate/, "calculator"], [/certificate|refund/, "credit-card"]];
const toolCall = (a) => /^\s*([a-z_]+)\s*\(/.exec(a) || /"name"\s*:\s*"([a-z_]+)"/.exec(a);
const actionIcon = (a) => { const m = toolCall(a); if (!m) return "message-square"; for (const [re, ic] of TOOL_ICON) if (re.test(m[1])) return ic; return "wrench"; };
const obsIcon = (o) => /^\s*\[user\]/.test(o) ? "user" : /error|invalid|not found|failed/i.test(o.slice(0, 200)) ? "triangle-alert" : "server";
function icon(g, name, x, y, size, color = INK, op = 1) { const k = size / 24, el = g.append("g").attr("transform", `translate(${x - size / 2},${y - size / 2}) scale(${k})`).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("stroke-linecap", "round").attr("stroke-linejoin", "round").attr("opacity", op); el.html(window.LUCIDE[name] || window.LUCIDE["wrench"]); return el; }
function pair(g, x, y, ai, oi, size, op = 1) {
  const w = size * 1.5, h = size * 2.9, el = g.append("g").attr("transform", `translate(${x - w / 2},${y - h / 2})`).attr("opacity", op);
  el.append("rect").attr("width", w).attr("height", h / 2).attr("rx", w / 2).attr("fill", ACT_BG); el.append("rect").attr("y", h / 2).attr("width", w).attr("height", h / 2).attr("rx", w / 2).attr("fill", OBS_BG);
  icon(el, ai, w / 2, h / 4, size, ACT); icon(el, oi, w / 2, 3 * h / 4, size, OBS); return el;
}
const outcome = (g, x, y, pass, size = 16, op = 1) => icon(g, pass ? "circle-check" : "circle-x", x, y, size, pass ? PASS : FAIL, op);
const bot = (g, x, y, r = 11) => { const s = g.append("g").attr("transform", `translate(${x},${y})`); s.append("circle").attr("r", r).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2); icon(s, "bot", 0, 0, r * 1.15, INK); return s; };

let B = null, D = null;
// The global state graph over whole rollouts, for act 2's map, act 3, and the state-map panel (any task's bundle).
function buildGraph(rolls, pass, noise) {
  // the global state graph over whole rollouts. Clusters are formed over all steps of all rollouts, so one state
  // can occur at different step indices and the graph can loop. Nodes sit left to right by the mean step at which they
  // occur; the outlier bin (steps no cluster claims) sits apart at the bottom. Success rates count rollouts, not visits.
  const L = d3.max(rolls, r => r.n_steps), gstat = new Map(), gedges = new Map();
  rolls.forEach(r => { const p = pass(r), seenC = new Set(), seenE = new Set();
    for (let t = 0; t < r.n_steps; t++) { const c = r.steps[t].c; if (!gstat.has(c)) gstat.set(c, { c, n: 0, nr: 0, np: 0, ts: 0 }); const g0 = gstat.get(c); g0.n++; g0.ts += t;
      if (!seenC.has(c)) { seenC.add(c); g0.nr++; if (p) g0.np++; }
      if (t + 1 < r.n_steps) { const b = r.steps[t + 1].c, k = `${c}>${b}`; if (!gedges.has(k)) gedges.set(k, { a: c, b, n: 0, nr: 0, np: 0 }); const e = gedges.get(k); e.n++;
        if (!seenE.has(k)) { seenE.add(k); e.nr++; if (p) e.np++; } } } });
  gedges.forEach(e => { e.sr = e.np / e.nr; });
  const gnodes = [...gstat.values()].map(g0 => ({ ...g0, meanT: g0.ts / g0.n, sr: g0.np / g0.nr, r: 3.5 + Math.sqrt(g0.n) * 1.1, noise: !!noise.get(g0.c) }));
  const tMax = d3.max(gnodes, d => d.meanT), gx = d3.scaleLinear().domain([0, Math.max(1, tMax)]).range([70, 830]), CY = 222, NY = 396;
  // the trunk (states most rollouts visit) runs through the middle; rarer states sit further above or below it
  const nR = rolls.length, side = (nd) => (hash(nd.c, 1, 7) < 0.5 ? -1 : 1), yT = (nd) => nd.noise ? NY : CY + side(nd) * (1 - nd.nr / nR) * 130;
  gnodes.forEach(nd => { nd.x = gx(nd.meanT); nd.y = yT(nd); });
  const links = [...gedges.values()].filter(e => e.a !== e.b && !noise.get(e.a) && !noise.get(e.b)).map(e => ({ source: e.a, target: e.b, n: e.nr }));
  const sim = d3.forceSimulation(gnodes)
    .force("x", d3.forceX(d => gx(d.meanT)).strength(0.9))
    .force("y", d3.forceY(yT).strength(d => d.noise ? 1 : 0.12))
    .force("link", d3.forceLink(links).id(d => d.c).distance(50).strength(l => 0.01 + 0.01 * Math.sqrt(l.n)))
    .force("collide", d3.forceCollide(d => d.r + 13).iterations(3)).stop();
  for (let i = 0; i < 400; i++) sim.tick();
  gnodes.forEach(nd => { nd.x = Math.max(40, Math.min(860, nd.x)); nd.y = nd.noise ? NY : Math.max(84, Math.min(350, nd.y)); });
  const gnode = new Map(gnodes.map(nd => [nd.c, nd]));
  return { L, gnodes, gnode, gedges };
}

function prep(b) {
  const rolls = b.rollouts, S = 12, sr = new Map(b.clusters.map(c => [c.id, c.sr])), noise = new Map(b.clusters.map(c => [c.id, c.noise]));
  const pass = (r) => r.reward >= 0.5, ic = (r) => r.steps.map(s => [actionIcon(s.a), obsIcon(s.o)]);
  // the branch pair: a failing and a passing rollout with the longest shared state prefix
  // the branch must fall inside the S steps we draw, with room on both sides: prefer the longest shared prefix with 2 <= k <= S - 4
  let best = null, fallback = null;
  rolls.filter(r => !pass(r)).forEach(f => rolls.filter(r => pass(r)).forEach(p => { let k = 0; while (k < f.n_steps && k < p.n_steps && f.steps[k].c === p.steps[k].c) k++;
    const cand = { f, p, k }, better = (a, b) => !b || a.k > b.k || (a.k === b.k && a.f.n_steps + a.p.n_steps < b.f.n_steps + b.p.n_steps);
    if (k >= 2 && k <= S - 4 && p.n_steps > k + 2 && f.n_steps > k + 2) { if (better(cand, best)) best = cand; } else if (better(cand, fallback)) fallback = cand; }));
  if (!best) best = fallback;
  // the DAG over the first S steps: nodes (t, state) ordered by success rate, edges with counts and pass counts
  const cols = d3.range(S).map(t => { const m = new Map(); rolls.forEach(r => { if (t < r.n_steps) { const c = r.steps[t].c; if (!m.has(c)) m.set(c, { c, n: 0, np: 0 }); const e = m.get(c); e.n++; if (pass(r)) e.np++; } });
    const order = [...m.values()].sort((a, c) => (sr.get(c.c) - sr.get(a.c)) || (a.c - c.c)); const ys = d3.scalePoint().domain(order.map(o => o.c)).range([70, 400]).padding(0.6);
    order.forEach(o => { o.y = order.length === 1 ? 235 : ys(o.c); o.x = 70 + t * (780 / (S - 1)); }); return { t, nodes: order, byC: new Map(order.map(o => [o.c, o])) }; });
  const node = (t, c) => cols[t].byC.get(c);
  const edges = new Map(); rolls.forEach(r => { for (let t = 0; t + 1 < Math.min(S, r.n_steps); t++) { const k = `${t}:${r.steps[t].c}>${r.steps[t + 1].c}`; if (!edges.has(k)) edges.set(k, { t, a: r.steps[t].c, b: r.steps[t + 1].c, n: 0, np: 0 }); const e = edges.get(k); e.n++; if (pass(r)) e.np++; } });
  // branch points: nodes with at least two outgoing edges whose pass fractions differ strongly
  const branchPts = []; cols.forEach(col => col.nodes.forEach(nd => { const out = [...edges.values()].filter(e => e.t === col.t && e.a === nd.c && e.n >= 3); if (out.length < 2) return;
    const fr = out.map(e => e.np / e.n); if (d3.max(fr) - d3.min(fr) >= 0.5) branchPts.push({ t: col.t, c: nd.c, spread: d3.max(fr) - d3.min(fr) }); }));
  // act 2: the longest prefix of the failed rollout from which the shared-prefix completions both succeed and fail
  let prefix = null;
  for (let k = best.k; k >= 3 && !prefix; k--) { const sh = rolls.filter(r => r.n_steps > k && d3.range(k).every(t => r.steps[t].c === best.f.steps[t].c));
    if (sh.filter(pass).length >= 2 && sh.filter(r => !pass(r)).length >= 2 && sh.includes(best.p)) prefix = { k, rolls: sh }; }
  if (!prefix) prefix = { k: best.k, rolls: rolls.filter(r => r.n_steps > best.k && d3.range(best.k).every(t => r.steps[t].c === best.f.steps[t].c)) };
  const { L, gnodes, gnode, gedges } = buildGraph(rolls, pass, noise);
  // act 2: two steps in the same (non-outlier) state, from different rollouts and preferably different steps, saying the same thing in different words
  const wordsOf = (t) => new Set((t || "").toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(" ").filter(w => w.length > 2));
  let simPair = null;
  gnodes.filter(nd => !nd.noise).forEach(nd => { const occ = []; rolls.forEach(r => { for (let t = 0; t < r.n_steps; t++) if (r.steps[t].c === nd.c && !toolCall(r.steps[t].a) && r.steps[t].a.length < 260) occ.push({ r, t }); });
    const cand = occ.slice(0, 40);
    for (let i = 0; i < cand.length; i++) for (let j = i + 1; j < cand.length; j++) { const A0 = cand[i], B0 = cand[j]; if (A0.r === B0.r) continue;
      const A = wordsOf(A0.r.steps[A0.t].a), Bw = wordsOf(B0.r.steps[B0.t].a), inter = [...A].filter(w => Bw.has(w)).length, jac = inter / Math.max(1, A.size + Bw.size - inter);
      if (jac <= 0.15 || jac >= 0.6) continue; const score = nd.n * (1 - Math.abs(jac - 0.35)) * (Math.abs(A0.t - B0.t) >= 2 ? 3 : 1);
      if (!simPair || score > simPair.score) simPair = { c: nd.c, a: A0.t <= B0.t ? A0 : B0, b: A0.t <= B0.t ? B0 : A0, score, n: nd.n }; } });
  // act 3: a linear classifier over the states and transitions each rollout passes through (length-normalised counts,
  // L2 logistic regression), fit on two cross-validation thirds and scored on the third held out. All three folds are fit;
  // the page shows the fold whose held-out AUROC is closest to the three-fold mean, so the example is typical, not the best.
  const folds = (b.cv_folds || []).filter(fo => fo.length);
  const stepFeats = (r, t) => { const ks = [`s${r.steps[t].c}`]; if (t > 0) ks.push(`${r.steps[t - 1].c}>${r.steps[t].c}`); return ks; };
  const feats = (r) => { const v = new Map(); for (let t = 0; t < r.n_steps; t++) stepFeats(r, t).forEach(k => v.set(k, (v.get(k) || 0) + 1 / r.n_steps)); return v; };
  const aurocOf = (rs, sf) => { let num = 0, den = 0; rs.forEach(a => { if (!pass(a)) return; rs.forEach(c => { if (pass(c)) return; den++; const d = sf(a) - sf(c); num += d > 0 ? 1 : d === 0 ? 0.5 : 0; }); }); return den ? num / den : 0.5; };
  const fitFold = (testIds) => { const train = rolls.filter(r => !testIds.has(r.rid)), heldOut = rolls.filter(r => testIds.has(r.rid)), w = new Map(); let b0 = 0;
    const X = train.map(feats), y = train.map(r => pass(r) ? 1 : 0);
    for (let it = 0; it < 400; it++) { const gr = new Map(); let gb = 0;
      X.forEach((x, i) => { let z = b0; x.forEach((v, k) => z += (w.get(k) || 0) * v); const d = 1 / (1 + Math.exp(-z)) - y[i]; gb += d; x.forEach((v, k) => gr.set(k, (gr.get(k) || 0) + d * v)); });
      b0 -= 0.5 * gb / X.length; gr.forEach((gv, k) => w.set(k, (w.get(k) || 0) - 0.5 * (gv / X.length + 0.01 * (w.get(k) || 0)))); }
    // the score after reading u steps (fractional), so a marker can follow a rollout as it is read; u = n_steps gives the full score
    const partial = (r, u) => { let z = b0; const n = r.n_steps, whole = Math.min(n, Math.floor(u));
      for (let t = 0; t < whole; t++) stepFeats(r, t).forEach(k => z += (w.get(k) || 0) / n);
      if (whole < n && u > whole) stepFeats(r, whole).forEach(k => z += (u - whole) * (w.get(k) || 0) / n); return z; };
    const score = (r) => partial(r, r.n_steps);
    return { train, heldOut, w, b0, partial, score, auroc: aurocOf(heldOut, score) }; };
  const fits = folds.map(fo => fitFold(new Set(fo))), meanAuroc = d3.mean(fits, x => x.auroc);
  const fi = fits.length ? d3.minIndex(fits, x => Math.abs(x.auroc - meanAuroc)) : 0, fit = fits[fi] || fitFold(new Set(rolls.slice(0, 17).map(r => r.rid)));
  // presentation order: the success it scores highest, the failure it scores lowest, the median rollout, then the rest
  const byScore = [...fit.heldOut].sort((a, c) => fit.score(c) - fit.score(a)), ho = [];
  [byScore.find(pass), [...byScore].reverse().find(r => !pass(r)), byScore[Math.floor(byScore.length / 2)]].forEach(r => { if (r && !ho.includes(r)) ho.push(r); });
  fit.heldOut.forEach(r => { if (!ho.includes(r)) ho.push(r); });
  const sc = ho.map(fit.score), lo = d3.min(sc), hi = d3.max(sc), pad = (hi - lo) * 0.06 || 1, yS = d3.scaleLinear().domain([lo - pad, hi + pad]).range([372, 96]).clamp(true);
  const dots = ho.map((r, i) => ({ r, i, s: sc[i], x: 800, y: yS(sc[i]), ty: yS(sc[i]) }));
  const dsim = d3.forceSimulation(dots).force("x", d3.forceX(800).strength(0.12)).force("y", d3.forceY(d => d.ty).strength(1)).force("collide", d3.forceCollide(6.2).iterations(3)).stop();
  for (let i = 0; i < 200; i++) dsim.tick();
  // colour scale for the learned weights: the 90th percentile of |w|, so one large weight does not wash out the rest
  const clf = { fold: fi, foldAurocs: fits.map(x => x.auroc), meanAuroc, ...fit, heldOut: ho, yS, dots, refW: d3.quantile([...fit.w.values()].map(Math.abs).sort(d3.ascending), 0.9) || 1, wOf: (k) => fit.w.get(k) || 0 };
  return { rolls, S, L, sr, noise, pass, ic, pairFP: best, prefix, simPair, clf, gnodes, gnode, gedges, cols, node, edges, branchPts, laneY: (i) => 36 + i * (390 / (rolls.length - 1)), xStep: (t) => 70 + t * (780 / (S - 1)) };
}

// ---------- helpers for a lane of pills ----------
function lane(g, r, y, x0, x1, upto, size, D_, op = 1, withBot = true, frac = null) {
  const n = Math.min(D_.S, r.n_steps), xs = d3.scalePoint().domain(d3.range(n)).range([x0, x1]).padding(0.5), icons = D_.ic(r);
  g.append("line").attr("x1", x0).attr("x2", x1).attr("y1", y).attr("y2", y).attr("stroke", "#e3e3df").attr("opacity", op);
  const shown = Math.min(n, upto);
  for (let t = 0; t < shown; t++) pair(g, xs(t), y, icons[t][0], icons[t][1], size, op);
  if (withBot) { const bx = shown >= n ? x1 + 16 : (shown === 0 ? x0 : lerp(xs(shown - 1), shown < n ? xs(shown) : x1 + 16, frac ?? 0)); bot(g, bx, y, size * 1.6); }
  if (shown >= n) outcome(g, x1 + 46, y, D_.pass(r), size * 2.6, op);
  return xs;
}


// ---------- the attempt scene (act 1) ----------
const ACT_LABEL = { "message-square": "message to the user", "id-card": "look up the user", "package": "read an order", "tag": "read a product", "undo-2": "return items", "repeat": "exchange items", "map-pin": "change an address", "pencil": "modify an order", "ban": "cancel an order", "hand-helping": "hand off to a human", "calculator": "calculate", "credit-card": "payment", "wrench": "tool call" };
const OBS_LABEL = { user: "the user replies", server: "a tool returns", "triangle-alert": "a tool reports an error" };
const BASIS_LABEL = { DB: "final database state", COMMUNICATE: "required info given to the user", ACTION: "required actions taken", NL_ASSERTION: "behaviour checks", ENV_ASSERTION: "environment checks" };
const clean = (t) => (t || "").replace(/\s+/g, " ").trim();
function actText(a) { const m = toolCall(a); if (!m) return clean(a).replace(/\*\*/g, "");
  const i = a.indexOf("("), j = a.lastIndexOf(")"); try { const args = JSON.parse(a.slice(i + 1, j));
    return `${m[1]}(${Object.entries(args).map(([k, v]) => `${k}=${Array.isArray(v) ? "[" + v.join(", ") + "]" : v}`).join(", ")})`; } catch (e) { return clean(a); } }
function obsText(o) { const t = clean(o).replace(/^\[(user|tool)\]\s*/, "").replace(/###STOP###/g, "").trim(); if (!/^[\[{]/.test(t)) return t;
  const pri = ["status", "order_id", "name", "user_id", "product_id", "item_id", "total", "amount", "price"], pairs = [], seen = new Set(), re = /"([a-z_]+)"\s*:\s*("([^"]*)"|-?\d+(\.\d+)?)/g; let m;
  while ((m = re.exec(t)) && pairs.length < 60) if (!seen.has(m[1])) { seen.add(m[1]); pairs.push([m[1], m[3] ?? m[2]]); }
  pairs.sort((x, y) => ((pri.indexOf(x[0]) + 1) || 99) - ((pri.indexOf(y[0]) + 1) || 99));
  return pairs.slice(0, 4).map(([k, v]) => `${k}: ${v}`).join(" · ") || t; }
function wrapText(t, cpl) { const out = []; let cur = ""; t.split(" ").forEach(w => { if ((cur + " " + w).trim().length > cpl && cur) { out.push(cur); cur = w; } else cur = (cur + " " + w).trim(); }); if (cur) out.push(cur); return out; }
function textBlock(g, x, y, lines, fs, color, maxLines, mono) { lines.slice(0, maxLines).forEach((l, i) => { const last = i === maxLines - 1 && lines.length > maxLines;
  g.append("text").attr("x", x).attr("y", y + i * fs * 1.35).attr("font-size", fs).attr("fill", color).attr("font-family", mono ? "ui-monospace, Menlo, monospace" : null).text(last ? l.replace(/.{0,2}$/, "…") : l); }); }
function speech(g, x, y, text, op = 1, color = INK) { const w = text.length * 7.4 + 20, b = g.append("g").attr("opacity", op);
  b.append("rect").attr("x", x - w / 2).attr("y", y - 26).attr("width", w).attr("height", 24).attr("rx", 12).attr("fill", "#fff").attr("stroke", color).attr("stroke-width", 1.4);
  b.append("path").attr("d", `M${x - 5},${y - 2.5} L${x},${y + 5} L${x + 5},${y - 2.5}`).attr("fill", "#fff").attr("stroke", color).attr("stroke-width", 1.4);
  b.append("rect").attr("x", x - 6).attr("y", y - 4).attr("width", 12).attr("height", 3).attr("fill", "#fff");
  b.append("text").attr("x", x).attr("y", y - 10).attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 700).attr("fill", color).text(text); return b; }

function attemptScene(svg, f, opt = {}) {
  const keep = 1 - (opt.endFade || 0), g = svg.append("g"), r = D.pairFP.f, n = r.n_steps, steps = r.steps, grade = r.grader;
  const opening = r.opening || "", fIn = sub(f, 0, 0.008), typed = sub(f, 0.006, 0.052), agentIn = sub(f, 0.05, 0.062), shrink = ease(sub(f, 0.07, 0.11));
  const u = sub(f, 0.11, 0.74), p = n * Math.pow(u, 1.6);
  const doneT = sub(f, 0.75, 0.79), gradT = ease(sub(f, 0.80, 0.83)), scanT = sub(f, 0.83, 0.88), verT = sub(f, 0.88, 0.92), strikeT = ease(sub(f, 0.93, 0.96)), glowT = sub(f, 0.955, 0.99);
  // camera: zoomed in on the current step, pulling out as the attempt grows
  const zo = Math.pow(clamp01(p / n), 1.6), v = lerp(1.25, n + 1.0, zo), k = 860 / v;
  // hold on the step being written; glide to the next one only once it is complete
  const iCur = Math.min(n - 1, Math.floor(p)), xCam = p >= n ? n - 1 : iCur + ease(sub(p - Math.floor(p), 0.78, 1)) * (iCur < n - 1 ? 1 : 0);
  let left = xCam + 0.62 - v; if (left < -0.55) left = -0.55;
  const sx = (x) => 20 + (x - left) * k, cw = 0.84 * k, ch = Math.min(290, Math.max(26, cw * 0.6)), cy = lerp(262, 196, zo);
  const wFull = sub(cw, 210, 270), wPill = 1 - sub(cw, 64, 96), wMid = Math.max(0, 1 - wFull - wPill);
  // track
  if (p > 0 && wFull < 1) g.append("line").attr("x1", sx(-0.5)).attr("x2", sx(Math.min(p, n) - 0.5)).attr("y1", cy).attr("y2", cy).attr("stroke", "#d6d6d0").attr("stroke-width", Math.max(1.5, k * 0.01)).attr("opacity", 1 - wFull);
  const glow = glowT > 0 ? g.append("g").attr("opacity", keep) : null;
  for (let i = 0; i < n; i++) {
    if (i > p) break; const s = clamp01(p - i), aA = sub(s, 0, 0.22), aO = sub(s, 0.42, 0.66);
    const xc = sx(i); if (xc + cw / 2 < -10 || xc - cw / 2 > W + 10) continue;
    const ai = actionIcon(steps[i].a), oi = obsIcon(steps[i].o), x0 = xc - cw / 2, y0 = cy - ch / 2, half = ch / 2;
    if (wFull > 0) { const cg = g.append("g").attr("opacity", wFull), fs = Math.max(10.5, Math.min(13, cw * 0.022)), cpl = Math.floor((cw - 72) / (fs * 0.55)), ml = Math.max(1, Math.floor((half - 46) / (fs * 1.35)));
      const halfCard = (yy, op, bg, col, tag, label, ic, text, mono) => { const hg = cg.append("g").attr("opacity", op).attr("transform", `translate(0,${(1 - op) * -6})`);
        hg.append("rect").attr("x", x0).attr("y", yy).attr("width", cw).attr("height", half - 3).attr("rx", 10).attr("fill", bg).attr("stroke", col).attr("stroke-width", 1.2);
        hg.append("text").attr("x", x0 + 14).attr("y", yy + 19).attr("font-size", 10).attr("font-weight", 700).attr("letter-spacing", 1.2).attr("fill", col).text(tag);
        hg.append("text").attr("x", x0 + 14 + tag.length * 7.6).attr("y", yy + 19).attr("font-size", 10.5).attr("fill", col).text(`· ${label}`);
        icon(hg, ic, x0 + 30, yy + 19 + (half - 22) / 2 + 2, 26, col);
        textBlock(hg, x0 + 56, yy + 40, wrapText(text, cpl), fs, INK, ml, mono); };
      halfCard(y0, aA, ACT_BG, ACT, "ACTION", ACT_LABEL[ai] || "tool call", ai, actText(steps[i].a), !!toolCall(steps[i].a));
      halfCard(y0 + half + 3, aO, OBS_BG, OBS, "OBSERVATION", OBS_LABEL[oi] || "observation", oi, obsText(steps[i].o), false);
      cg.append("text").attr("x", x0 + cw - 10).attr("y", y0 - 6).attr("text-anchor", "end").attr("font-size", 10).attr("fill", MUTED).attr("opacity", aA).text(`step ${i + 1}`); }
    if (wMid > 0) { const mg = g.append("g").attr("opacity", wMid), is = Math.min(26, cw * 0.22), fs = 9.5, show = cw > 130;
      [[y0, aA, ACT_BG, ACT, ai, ACT_LABEL[ai]], [y0 + half + 2, aO, OBS_BG, OBS, oi, OBS_LABEL[oi]]].forEach(([yy, op, bg, col, ic, lab]) => { const hg = mg.append("g").attr("opacity", op);
        hg.append("rect").attr("x", x0).attr("y", yy).attr("width", cw).attr("height", half - 2).attr("rx", 8).attr("fill", bg).attr("stroke", col).attr("stroke-width", 1);
        icon(hg, ic, show ? x0 + 8 + is / 2 : xc, yy + (half - 2) / 2, is, col);
        if (show) hg.append("text").attr("x", x0 + 14 + is).attr("y", yy + (half - 2) / 2 + 3.5).attr("font-size", fs).attr("fill", col).text((lab || "").slice(0, Math.floor((cw - is - 20) / 5.3))); }); }
    if (wPill > 0) pair(g, xc, cy, ai, oi, Math.min(cw / 1.5, ch / 2.9) * 0.92, wPill * aA);
    if (glow) { const pulse = 0.5 + 0.5 * Math.sin((glowT * 3 + i * 0.08) * Math.PI); glow.append("rect").attr("x", x0 - 2).attr("y", cy - ch / 2 - 2).attr("width", cw + 4).attr("height", ch + 4).attr("rx", 6).attr("fill", ACT).attr("opacity", 0.05 + 0.1 * pulse * glowT); }
  }
  // the customer's request: a large card first, then a banner across the top
  { const big = g.append("g").attr("opacity", fIn * (1 - shrink));
    big.append("rect").attr("x", 130).attr("y", 70).attr("width", 640).attr("height", 136).attr("rx", 14).attr("fill", "#fff").attr("stroke", "#c9c9c3").attr("stroke-width", 1.4);
    icon(big, "clipboard-list", 164, 104, 28, INK); big.append("text").attr("x", 190).attr("y", 100).attr("font-size", 11).attr("letter-spacing", 1.4).attr("fill", MUTED).text("TASK · THE USER WRITES");
    const shown = opening.slice(0, Math.round(opening.length * typed)), caret = typed < 1 && Math.floor(f * 900) % 2 === 0 ? "▍" : "";
    textBlock(big, 160, 140, wrapText(`“${shown}${typed >= 1 ? "”" : caret}`, 56), 19, INK, 3, false);
    const ban = g.append("g").attr("opacity", shrink);
    ban.append("rect").attr("x", 20).attr("y", 10).attr("width", 860).attr("height", 38).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df");
    icon(ban, "clipboard-list", 40, 29, 18, INK); ban.append("text").attr("x", 58).attr("y", 33).attr("font-size", 12.5).attr("fill", INK).text(`Task · “${opening.length > 100 ? opening.slice(0, 99) + "…" : opening}”`); }
  // the agent: thinking under the request, then walking along the attempt
  const ar = Math.max(8, Math.min(16, k * 0.03));
  let axS = sx(Math.min(p, n) - 0.5), ayS = cy - ch / 2 - ar - 8;
  if (shrink < 1) { axS = lerp(450, axS, shrink); ayS = lerp(268, ayS, shrink); }
  if (agentIn > 0) { const sp = bot(g, axS, ayS, shrink < 1 ? lerp(18, ar, shrink) : ar); sp.attr("opacity", agentIn * keep);
    const sI = clamp01(p - Math.floor(p)), typing = (p > 0 && p < n && sI < 0.25) || (shrink < 1 && f > 0.062);
    if (typing) { const ph = Math.floor(f * 400) % 3; speech(g, axS + 26, ayS - 4, ["·  ", "·· ", "···"][ph], 1, MUTED); } }
  // done!
  if (doneT > 0 && keep > 0) { speech(g, axS, ayS - ar - 4, "Done!", doneT * keep, INK); }
  // the grader
  const tx0 = sx(-0.5), tx1 = sx(n - 0.5);
  if (gradT > 0) {
    const gg = g.append("g").attr("opacity", gradT * lerp(1, 0.35, strikeT) * keep).attr("transform", `translate(0,${(1 - gradT) * 30})`), gx = 250, gy = 248, gw = 400, gh = 150;
    gg.append("rect").attr("x", gx).attr("y", gy).attr("width", gw).attr("height", gh).attr("rx", 12).attr("fill", "#fff").attr("stroke", "#c9c9c3").attr("stroke-width", 1.4);
    icon(gg, "gavel", gx + 28, gy + 28, 26, INK);
    gg.append("text").attr("x", gx + 52).attr("y", gy + 26).attr("font-size", 15).attr("font-weight", 700).attr("fill", INK).text("Grader");
    gg.append("text").attr("x", gx + 52).attr("y", gy + 42).attr("font-size", 10.5).attr("fill", MUTED).text("checks the end state against the task's goal");
    if (scanT > 0 && verT === 0) gg.append("text").attr("x", gx + 22).attr("y", gy + 76).attr("font-size", 12).attr("fill", MUTED).text("checking" + ".".repeat(1 + (Math.floor(f * 300) % 3)));
    if (verT > 0) { const vg = gg.append("g").attr("opacity", verT); let yy = gy + 70;
      const basis = grade ? grade.basis : [];
      basis.forEach(b => { const ok = (grade.breakdown[b] ?? 0) >= 0.5; icon(vg, ok ? "circle-check" : "circle-x", gx + 30, yy - 4, 15, ok ? PASS : FAIL);
        vg.append("text").attr("x", gx + 46).attr("y", yy).attr("font-size", 12).attr("fill", INK).text(BASIS_LABEL[b] || b.toLowerCase()); yy += 19; });
      if (grade && grade.failed.length) { vg.append("text").attr("x", gx + 46).attr("y", yy).attr("font-size", 10.5).attr("font-style", "italic").attr("fill", MUTED).text(grade.failed[0].slice(0, 62)); yy += 8; }
      const res = vg.append("g").attr("transform", `translate(${gx + gw - 118},${gy + 22})`);
      res.append("rect").attr("width", 104).attr("height", 30).attr("rx", 15).attr("fill", D.pass(r) ? PASS : FAIL);
      res.append("text").attr("x", 52).attr("y", 20).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 800).attr("fill", "#fff").text(D.pass(r) ? "PASSED" : "FAILED"); }
    if (strikeT > 0) { const sg = g.append("g").attr("opacity", gradT * keep); const l1 = [[gx - 6, gy - 6], [gx + gw + 6, gy + gh + 6]], l2 = [[gx + gw + 6, gy - 6], [gx - 6, gy + gh + 6]];
      [l1, l2].forEach(([[x1, y1], [x2, y2]], i) => { const t = clamp01(strikeT * 2 - i); if (t > 0) sg.append("line").attr("x1", x1).attr("y1", y1).attr("x2", lerp(x1, x2, t)).attr("y2", lerp(y1, y2, t)).attr("stroke", INK).attr("stroke-width", 5).attr("stroke-linecap", "round"); });
      sg.append("text").attr("x", gx + gw / 2).attr("y", gy + gh + 26).attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-style", "italic").attr("fill", INK).attr("opacity", strikeT).text("without the grader"); }
  }
  // scanning band over the attempt
  if (scanT > 0 && scanT < 1) g.append("rect").attr("x", lerp(tx0, tx1, scanT) - 14).attr("y", cy - ch / 2 - 10).attr("width", 28).attr("height", ch + 20).attr("rx", 6).attr("fill", ACT).attr("opacity", 0.18);
  if (verT > 0) outcome(g, tx1 + 24, cy, D.pass(r), 18, verT);
  // only the actions and observations remain
  if (glowT > 0) { const gl = g.append("g").attr("opacity", glowT * keep);
    gl.append("rect").attr("x", tx0 - 10).attr("y", cy - ch / 2 - 10).attr("width", tx1 - tx0 + 20).attr("height", ch + 20).attr("rx", 10).attr("fill", "none").attr("stroke", ACT).attr("stroke-width", 2.5);
    gl.append("text").attr("x", tx0 - 6).attr("y", cy + ch / 2 + 28).attr("font-size", 13).attr("font-weight", 700).attr("fill", ACT).text("only the actions and observations"); }
}


// ---------- act 2: from one failed rollout to the map of behavior states ----------
// ---------- the state map, shared by acts 2 and 3 ----------
const NEU = "#b9b9b3", halo = (t) => t.attr("stroke", "#f7f7f5").attr("stroke-width", 4).attr("paint-order", "stroke");
// a transition between two states: forward edges bow alternately up or down, back edges arc over the top, self-loops sit on top of the node
const gEdgePath = (e, G = D) => { const a = G.gnode.get(e.a), b = G.gnode.get(e.b);
  if (e.a === e.b) { const r = a.r; return `M${a.x - r * 0.5},${a.y - r * 0.85} C${a.x - r * 1.4},${a.y - r * 2.5} ${a.x + r * 1.4},${a.y - r * 2.5} ${a.x + r * 0.5},${a.y - r * 0.85}`; }
  const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1, nx = -dy / len, ny = dx / len;
  const bend = dx >= 0 ? (hash(e.a, e.b, 3) < 0.5 ? 1 : -1) * Math.min(46, 0.14 * len) : Math.min(90, 0.3 * len);
  return `M${a.x},${a.y} Q${(a.x + b.x) / 2 + nx * bend},${(a.y + b.y) / 2 + ny * bend} ${b.x},${b.y}`; };
const isNoiseE = (e, G = D) => G.gnode.get(e.a).noise || G.gnode.get(e.b).noise, edgeW = (nr) => 0.5 + Math.sqrt(nr) * 0.7;
const drawEdge = (gg, e, col, op, nr = e.nr, w0 = 0, G = D) => gg.append("path").attr("d", gEdgePath(e, G)).attr("fill", "none").attr("stroke", col).attr("stroke-width", edgeW(nr) + w0)
  .attr("stroke-opacity", op * (0.22 + 0.5 * Math.min(1, nr / 20)) * (isNoiseE(e, G) ? 0.6 : 1)).attr("stroke-dasharray", isNoiseE(e, G) ? "3,3" : null);
const drawState = (gg, nd, rScale, op, col) => gg.append("circle").attr("cx", nd.x).attr("cy", nd.y).attr("r", nd.r * rScale).attr("fill", "#fff").attr("stroke", col).attr("stroke-width", 2.6).attr("stroke-dasharray", nd.noise ? "3,2" : null).attr("opacity", op);
const noiseLabel = (gg, op, G = D) => { const nd = G.gnodes.find(d => d.noise); if (nd && op > 0) halo(gg.append("text").attr("x", nd.x + nd.r + 7).attr("y", nd.y + 4).attr("font-size", 10).attr("fill", MUTED).attr("opacity", op)).text("unclustered steps"); };
const routeOf = (r) => d3.range(r.n_steps - 1).map(t => ({ a: r.steps[t].c, b: r.steps[t + 1].c }));

function drawBanner(g, opening, op = 1) { const ban = g.append("g").attr("opacity", op);
  ban.append("rect").attr("x", 20).attr("y", 10).attr("width", 860).attr("height", 38).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df");
  icon(ban, "clipboard-list", 40, 29, 18, INK); ban.append("text").attr("x", 58).attr("y", 33).attr("font-size", 12.5).attr("fill", INK).text(`Task · “${opening.length > 100 ? opening.slice(0, 99) + "…" : opening}”`); }
function branchScene(svg, f) {
  const g = svg.append("g"), F = D.pairFP.f, Pr = D.pairFP.p, n = F.n_steps, N = D.rolls.length, kR = D.prefix.k;
  const K = 860 / (n + 1), sxT = (i) => 20 + (i + 0.55) * K, cyT = 196, cwT = 0.84 * K, pillT = Math.min(cwT / 1.5, Math.max(26, cwT * 0.6) / 2.9) * 0.92;
  const laneY = (i) => 66 + i * (350 / (N - 1)), cap = (r) => Math.min(n, r.n_steps), opening = F.opening || "", icF = D.ic(F);
  const edgePath = (a, b) => `M${a.x},${a.y} C${(a.x + b.x) / 2},${a.y} ${(a.x + b.x) / 2},${b.y} ${b.x},${b.y}`, nodeR = (nd) => 5 + Math.sqrt(nd.n) * 2.2;
  const lanePills = (gg, r, y, size, op, upto) => { const c = cap(r), icons = D.ic(r);
    gg.append("line").attr("x1", sxT(0)).attr("x2", sxT(c - 1)).attr("y1", y).attr("y2", y).attr("stroke", "#e3e3df").attr("opacity", op * 0.8);
    for (let t = 0; t < Math.min(c, upto); t++) pair(gg, sxT(t), y, icons[t][0], icons[t][1], size, op); };
  // fifty attempts: the attempt from before is lane 1; lanes show every step as an action|observation chip
  const LN = D.L, L0 = 132, L1 = 700, xL = (t) => L0 + t * (L1 - L0) / (LN - 1), capL = (r) => r.n_steps;
  const order = [F, ...D.rolls.filter(r => r !== F)], laneOf = new Map(order.map((r, i) => [r, i])), gateX = 66, gateY = 241;
  const chip = (gg, x, y, op) => { if (op <= 0.01) return; gg.append("rect").attr("x", x - 5.6).attr("y", y - 2.5).attr("width", 5.2).attr("height", 5).attr("rx", 1).attr("fill", "#7d9fe0").attr("opacity", op);
    gg.append("rect").attr("x", x + 0.4).attr("y", y - 2.5).attr("width", 5.2).attr("height", 5).attr("rx", 1).attr("fill", "#e0a95a").attr("opacity", op); };
  const nodePos = (r, i, t) => { const nd = D.gnode.get(r.steps[t].c); return [nd.x + (hash(i, t, 5) - 0.5) * nd.r * 0.9, nd.y + (hash(i, t, 9) - 0.5) * nd.r * 0.9]; };
  const laneWin = (i) => i <= 3 ? [0.15 + (i - 1) * 0.1, 0.25 + (i - 1) * 0.1] : [0.45 + (i - 4) * 0.0035, 0.70 + (i - 4) * 0.0025];
  function lanes(gg, q, kitOp, laneOp, snap = 0) {
    const m = ease(sub(q, 0, 0.15)); let started = 0, ok = 0, bad = 0;
    if (m < 1) { for (let t = 0; t < n; t++) { const x = lerp(sxT(t), xL(t), m), y = lerp(cyT, laneY(0), m);
        pair(gg, x, y, icF[t][0], icF[t][1], lerp(pillT, 2.4, m), 1 - m); chip(gg, x, y, m); }
      outcome(gg, lerp(sxT(n - 1) + 44.5, xL(capL(F) - 1) + 14, m), lerp(cyT, laneY(0), m), D.pass(F), lerp(18, 7, m)); }
    order.forEach((r, i) => { const lp = i === 0 ? (m >= 1 ? 1 : 0) : sub(q, ...laneWin(i)); if (i > 0 && lp <= 0) return; started++;
      const c = capL(r), y = laneY(i), upto = lp * c, fanOp = 0.16 * kitOp * Math.min(1, (i === 0 ? m : lp) * 4);
      if (fanOp > 0) gg.append("line").attr("x1", gateX + 17).attr("y1", gateY).attr("x2", L0 - 12).attr("y2", y).attr("stroke", INK).attr("stroke-width", 0.8).attr("opacity", fanOp);
      if (i === 0 && m < 1) { if (!D.pass(r)) bad++; else ok++; return; }
      gg.append("line").attr("x1", xL(0)).attr("x2", xL(c - 1)).attr("y1", y).attr("y2", y).attr("stroke", "#e3e3df").attr("opacity", 0.8 * laneOp * (1 - snap));
      for (let t = 0; t < c; t++) { const a = i === 0 ? 1 : sub(upto, t, t + 0.5); if (a <= 0) continue;
        const [x, yy] = snap > 0 ? (() => { const [nx, ny] = nodePos(r, i, t); return [lerp(xL(t), nx, snap), lerp(y, ny, snap)]; })() : [xL(t), y];
        chip(gg, x, yy, a * laneOp * (1 - 0.85 * snap)); }
      if (i > 0 && lp < 1) { const tx = xL(Math.min(c - 1, upto)); if (i <= 3) bot(gg, tx + 9, y, 6).attr("opacity", laneOp); else gg.append("circle").attr("cx", tx + 7).attr("cy", y).attr("r", 1.8).attr("fill", INK).attr("opacity", laneOp); }
      if (lp >= 1) { outcome(gg, xL(c - 1) + 14, y, D.pass(r), 7, laneOp * (1 - snap)); if (D.pass(r)) ok++; else bad++; } });
    const k = kitOp * m; if (k <= 0) return;
    const kit = gg.append("g").attr("opacity", k);
    kit.append("circle").attr("cx", gateX).attr("cy", gateY).attr("r", 30).attr("fill", "#fff").attr("stroke", "#c9c9c3");
    icon(kit, "clipboard-list", gateX - 9, gateY - 8, 13, INK); icon(kit, "bot", gateX + 8, gateY - 8, 13, INK);
    kit.append("text").attr("x", gateX).attr("y", gateY + 14).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 800).attr("fill", INK).text("× 50");
    kit.append("text").attr("x", gateX).attr("y", gateY + 46).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).text("same task");
    kit.append("text").attr("x", gateX).attr("y", gateY + 59).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).text("same agent");
    kit.append("rect").attr("x", xL(0) - 9).attr("y", laneY(0) - 5).attr("width", xL(capL(F) - 1) - xL(0) + 32).attr("height", 10).attr("rx", 5).attr("fill", "none").attr("stroke", FAIL).attr("stroke-width", 1.2);
    halo(kit.append("text").attr("x", xL(capL(F) - 1) + 23).attr("y", laneY(0) - 8).attr("text-anchor", "end").attr("font-size", 9.5).attr("fill", FAIL)).text("attempt 1 · the one from before");
    const box = kit.append("g").attr("transform", "translate(730,92)");
    box.append("rect").attr("width", 150).attr("height", 70).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df");
    box.append("text").attr("x", 14).attr("y", 24).attr("font-size", 13).attr("font-weight", 800).attr("fill", INK).text(`attempt ${started} / ${N}`);
    icon(box, "circle-check", 22, 48, 14, PASS); box.append("text").attr("x", 34).attr("y", 52).attr("font-size", 13).attr("font-weight", 700).attr("fill", PASS).text(ok);
    icon(box, "circle-x", 76, 48, 14, FAIL); box.append("text").attr("x", 88).attr("y", 52).attr("font-size", 13).attr("font-weight", 700).attr("fill", FAIL).text(bad);
    kit.append("text").attr("x", 20).attr("y", 432).attr("font-size", 11).attr("fill", MUTED).attr("opacity", sub(q, 0.85, 1)).text(`${N} attempts · same task · same agent`);
  }
  // completions resampled from the same prefix: the successful rollout first, then the others that share the prefix
  const rowsY = [118, 274, 314, 354, 394, 157, 78];
  const comps = [Pr, ...D.prefix.rolls.filter(r => r !== F && r !== Pr)].slice(0, rowsY.length).map((r, i) => ({ r, y: rowsY[i] }));
  const drawComp = (gg, c, frac, op) => { const r = c.r, cp = cap(r), last = kR + (cp - kR) * frac, icons = D.ic(r);
    const pts = [[sxT(kR - 1), cyT]]; for (let t = kR; t <= Math.min(cp - 1, Math.floor(last)); t++) pts.push([sxT(t), c.y]);
    if (pts.length > 1) gg.append("path").attr("d", d3.line().curve(d3.curveMonotoneX)(pts)).attr("fill", "none").attr("stroke", D.pass(r) ? PASS : FAIL).attr("stroke-width", 2).attr("stroke-opacity", 0.55 * op);
    for (let t = kR; t < cp; t++) if (t <= last) pair(gg, sxT(t), c.y, icons[t][0], icons[t][1], pillT * 0.85, op * sub(last, t, t + 0.4));
    if (frac >= 1) outcome(gg, sxT(cp - 1) + 24, c.y, D.pass(r), 15, op);
    return last; };

  // ---- beats 1-4: the failed rollout; rewind and resample; mixed outcomes from one prefix; the cost of resampling everywhere
  if (f < 0.48) {
    attemptScene(g, 1, { endFade: sub(f, 0, 0.05) });
    const rw = sub(f, 0.05, 0.09), rs = sub(f, 0.09, 0.18), fan = sub(f, 0.18, 0.27), tally = sub(f, 0.25, 0.30), treeT = sub(f, 0.30, 0.45), clear = 1 - sub(f, 0.45, 0.48);
    if (rw > 0) { const pg = g.append("g").attr("opacity", clear);
      pg.append("rect").attr("x", sxT(0) - cwT / 2 - 3).attr("y", cyT - 18).attr("width", sxT(kR - 1) - sxT(0) + cwT + 6).attr("height", 36).attr("rx", 8).attr("fill", ACT).attr("opacity", 0.09 * sub(rw, 0.5, 1));
      halo(pg.append("text").attr("x", (sxT(0) + sxT(kR - 1)) / 2).attr("y", cyT + 34).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", ACT).attr("opacity", sub(rw, 0.5, 1))).text(`fixed prefix · first ${kR} states`);
      const x0 = sxT(n - 1), x1 = sxT(kR - 1), mid = (x0 + x1) / 2, arc = `M${x0},${cyT - 22} Q${mid},${cyT - 70} ${x1},${cyT - 22}`;
      pg.append("path").attr("d", arc).attr("fill", "none").attr("stroke", MUTED).attr("stroke-width", 1.5).attr("stroke-dasharray", "4,4").attr("opacity", rw < 1 ? 1 : 1 - sub(rs, 0, 0.3));
      halo(pg.append("text").attr("x", mid).attr("y", cyT - 52).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", MUTED).attr("opacity", rw * (1 - sub(rs, 0, 0.3)))).text("rewind");
      let bx = lerp(x0, x1, ease(rw)), by = cyT - 29;
      if (rs > 0) { const last = drawComp(pg, comps[0], rs, 1), cp = cap(comps[0].r); if (last < cp - 0.01) { bx = sxT(Math.max(kR - 1, Math.min(cp - 1, last))); by = (last < kR ? lerp(cyT, comps[0].y, last - (kR - 1)) : comps[0].y) - 22; } else { bx = sxT(cp - 1); by = comps[0].y - 22; } }
      if (treeT < 0.3) bot(pg, bx, by, 9).attr("opacity", 1 - sub(treeT, 0, 0.3));
      comps.slice(1).forEach((c, i) => { const fr = sub(fan, i * 0.09, i * 0.09 + 0.55); if (fr > 0) drawComp(pg, c, fr, 1); });
      if (tally > 0) { const shown = [F, ...comps.map(c => c.r)], np = shown.filter(D.pass).length;
        halo(pg.append("text").attr("x", 880).attr("y", 432).attr("text-anchor", "end").attr("font-size", 12.5).attr("font-weight", 700).attr("fill", INK).attr("opacity", tally)).text(`same prefix: ${np} succeed · ${shown.length - np} fail`); } }
    if (treeT > 0) { const tg = g.append("g").attr("opacity", clear), lf1 = sub(treeT, 0, 0.45), lf2 = sub(treeT, 0.4, 0.9);
      const branch = (i, j, y0, depth, seed) => { const pts = [[sxT(i), y0]]; let y = y0;
        for (let t = i + 1; t < n; t++) { y += (j ? 1 : -1) * (5 + 9 * hash(seed, t, j)) / (depth + 1) + (hash(seed, t, j + 3) - 0.5) * 10; y = Math.max(62, Math.min(424, y)); pts.push([sxT(t), y]); } return pts; };
      const drawPart = (pts, frac, w, op) => { const m = Math.max(2, Math.ceil(frac * (pts.length - 1)) + 1); tg.append("path").attr("d", d3.line().curve(d3.curveMonotoneX)(pts.slice(0, m))).attr("fill", "none").attr("stroke", "#8f8f8a").attr("stroke-width", w).attr("stroke-opacity", op); };
      for (let i = 0; i < n - 1; i++) for (let j = 0; j < 2; j++) { const b1 = branch(i, j, cyT, 0, i * 7 + j);
        if (lf1 > 0) drawPart(b1, lf1, 1.1, 0.4);
        if (lf2 > 0) for (let m = 2; m < b1.length - 1; m += 3) for (let jj = 0; jj < 2; jj++) drawPart(branch(i + m, jj, b1[m][1], 1, i * 31 + m * 7 + jj), lf2, 0.6, 0.25); }
      const count = Math.round(Math.pow(10, 1 + treeT * 6.4));
      const box = tg.append("g").attr("transform", "translate(20,58)").attr("opacity", sub(treeT, 0, 0.1));
      box.append("rect").attr("width", 200).attr("height", 62).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df");
      box.append("text").attr("x", 14).attr("y", 22).attr("font-size", 11).attr("fill", MUTED).text("rollouts sampled");
      box.append("text").attr("x", 14).attr("y", 49).attr("font-size", 22).attr("font-weight", 800).attr("fill", INK).text(d3.format(",")(count) + "…"); }
    return;
  }
  drawBanner(g, opening);
  // ---- beat 5: fifty attempts from the same gate: attempt 1 settles, attempts 2-4 run visibly, the rest fan out
  if (f < 0.60) { lanes(g, sub(f, 0.48, 0.60), 1, 1, 0); return; }
  // ---- beats 6-7: a pair of near-identical steps (same state, different steps) lifts out; then every step snaps into the global states
  const sp = D.simPair, q6 = sub(f, 0.60, 0.70), q7 = sub(f, 0.70, 0.84);
  const lift = f < 0.70 ? ease(sub(q6, 0.2, 0.6)) : 1 - ease(sub(q7, 0, 0.2)), mark = f < 0.70 ? sub(q6, 0, 0.2) : 1 - sub(q7, 0, 0.2);
  const cardsIn = f < 0.70 ? sub(q6, 0.45, 0.75) : 1 - sub(q7, 0, 0.12);
  const drawCard = (o, x0, y0, w, h, label) => { const t = o.t, r = o.r, px = xL(t), py = laneY(laneOf.get(r));
    const cg = g.append("g").attr("transform", `translate(${lerp(px, x0, lift)},${lerp(py, y0, lift)}) scale(${lerp(0.03, 1, lift)})`).attr("opacity", Math.min(1, lift * 1.5));
    cg.append("rect").attr("width", w).attr("height", h).attr("rx", 12).attr("fill", "#fff").attr("stroke", "#c9c9c3").attr("stroke-width", 1.4);
    cg.append("text").attr("x", 16).attr("y", 22).attr("font-size", 10.5).attr("letter-spacing", 1.2).attr("fill", MUTED).text(`${label} · STEP ${t + 1}`);
    const half = (h - 34) / 2, cpl = Math.floor((w - 40) / 6.3), sec = (yy, bg, col, tag, text, ml) => { cg.append("rect").attr("x", 10).attr("y", yy).attr("width", w - 20).attr("height", half - 6).attr("rx", 8).attr("fill", bg).attr("stroke", col).attr("stroke-width", 1);
      cg.append("text").attr("x", 22).attr("y", yy + 17).attr("font-size", 9.5).attr("font-weight", 700).attr("letter-spacing", 1.2).attr("fill", col).text(tag);
      textBlock(cg, 22, yy + 35, wrapText(text, cpl), 11.5, INK, ml, false); };
    sec(32, ACT_BG, ACT, "ACTION", actText(r.steps[t].a), 4); sec(32 + half, OBS_BG, OBS, "OBSERVATION", obsText(r.steps[t].o), 3); };
  if (f < 0.84) {
    const snap = f < 0.70 ? 0 : ease(sub(q7, 0.2, 0.62)), routes = f < 0.70 ? 0 : sub(q7, 0.58, 0.85), walk = f < 0.70 ? 0 : sub(q7, 0.7, 1), laneOp = 1 - 0.75 * lift, kitOp = f < 0.70 ? 1 - sub(q6, 0, 0.2) : 0;
    if (routes > 0) D.gedges.forEach(e => drawEdge(g, e, NEU, routes));
    if (sp && mark > 0) [sp.a, sp.b].forEach(o => g.append("rect").attr("x", xL(o.t) - 8).attr("y", 58).attr("width", 16).attr("height", 366).attr("rx", 7).attr("fill", ACT).attr("opacity", 0.07 * mark));
    lanes(g, 1, kitOp, laneOp, snap);
    if (sp && mark > 0) [sp.a, sp.b].forEach(o => g.append("circle").attr("cx", xL(o.t)).attr("cy", laneY(laneOf.get(o.r))).attr("r", 7).attr("fill", "none").attr("stroke", ACT).attr("stroke-width", 1.8).attr("opacity", mark));
    if (snap > 0) { D.gnodes.forEach(nd => drawState(g, nd, snap, snap, NEU)); noiseLabel(g, sub(snap, 0.6, 1)); }
    // one rollout (the attempt from before) traced as a path through the map
    if (walk > 0) { const rt = routeOf(F), upto = walk * rt.length; rt.forEach((e, t) => { if (t < upto) g.append("path").attr("d", gEdgePath(e)).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 2.2).attr("stroke-opacity", 0.85 * Math.min(1, upto - t)); });
      const tip = D.gnode.get(F.steps[Math.min(F.n_steps - 1, Math.floor(upto))].c); bot(g, tip.x, tip.y, 8); }
    if (sp && lift > 0) { drawCard(sp.a, 40, 70, 390, 270, "ROLLOUT A"); drawCard(sp.b, 470, 70, 390, 270, "ROLLOUT B");
      const label = sp.a.t !== sp.b.t ? `same state · step ${sp.a.t + 1} in one rollout, step ${sp.b.t + 1} in another` : `same state · step ${sp.a.t + 1}`, lw = label.length * 6.1 + 14;
      const bg = g.append("g").attr("opacity", cardsIn); bg.append("path").attr("d", `M235,66 V56 H${450 - lw / 2} M${450 + lw / 2},56 H665 V66`).attr("fill", "none").attr("stroke", ACT).attr("stroke-width", 1.6);
      bg.append("text").attr("x", 450).attr("y", 60).attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 700).attr("fill", ACT).text(label); }
    if (snap > 0) g.append("text").attr("x", 20).attr("y", 432).attr("font-size", 11).attr("fill", MUTED).attr("opacity", snap).text("each circle is one state: similar steps merged from any rollout, at any step · left to right: when it tends to occur");
    return; }
  // ---- beats 8-9: each rollout's outcome tints the states and transitions it passes through
  // the first few rollouts (the attempt from before, then the resampled success) walk the map one at a time; the rest stream through quickly
  const SLOW = 3, accOrder = [F, ...(Pr !== F ? [Pr] : []), ...order.filter(r => r !== F && r !== Pr)];
  const q8 = sub(f, 0.84, 0.95), pq = sub(q8, 0.05, 1), st = f >= 0.95 ? N : pq < 0.62 ? (pq / 0.62) * SLOW : SLOW + sub(pq, 0.62, 0.94) * (N - SLOW);
  const nIn = Math.min(N, Math.floor(st)), curF = st - nIn, hi = sub(f, 0.95, 1);
  const accE = new Map(), accS = new Map(), bump = (m, k, p) => { if (!m.has(k)) m.set(k, { nr: 0, np: 0 }); const o = m.get(k); o.nr++; if (p) o.np++; };
  for (let i = 0; i < nIn; i++) { const r = accOrder[i], p = D.pass(r); new Set(r.steps.map(s => s.c)).forEach(c => bump(accS, c, p)); new Set(routeOf(r).map(e => `${e.a}>${e.b}`)).forEach(k => bump(accE, k, p)); }
  // beat 9: everything that does not separate outcomes fades to grey; only clustered states, and transitions between them,
  // that at least four in five rollouts passing through end the same way keep their colour, deepened to pass or fail
  const pred = (o) => o.nr >= 5 && (o.sr >= 0.8 || o.sr <= 0.2), predE = (e) => !isNoiseE(e) && pred(e), keepOp = (yes) => yes ? 1 : 1 - 0.55 * hi;
  const recolor = (base, p, sr) => d3.interpolateRgb(base, p ? (sr >= 0.8 ? PASS : FAIL) : NEU)(hi);
  D.gedges.forEach(e => drawEdge(g, e, NEU, keepOp(false)));
  accE.forEach((a, k) => { const e = D.gedges.get(k), p = predE(e); if (p && hi > 0) drawEdge(g, e, "#fff", hi, a.nr, 4);
    drawEdge(g, e, recolor(srColor(a.np / a.nr), p, e.sr), p ? 1 + 0.8 * hi : keepOp(false), a.nr, p ? 1.5 * hi : 0); });
  D.gnodes.forEach(nd => { const a = accS.get(nd.c), p = !nd.noise && pred(nd);
    drawState(g, nd, 1, keepOp(p), recolor(a ? srColor(a.np / a.nr) : NEU, p, nd.sr)).attr("stroke-width", p ? 2.6 + 1.2 * hi : 2.6);
    if (p && hi > 0) g.append("circle").attr("cx", nd.x).attr("cy", nd.y).attr("r", Math.max(0, nd.r - 1.5)).attr("fill", nd.sr >= 0.8 ? PASS : FAIL).attr("opacity", 0.25 * hi); });
  noiseLabel(g, 1 - hi * 0.5);
  // the path traced at the end of beat 7 hands over to the stream
  const handOff = 1 - sub(q8, 0, 0.05);
  if (handOff > 0) { routeOf(F).forEach(e => g.append("path").attr("d", gEdgePath(e)).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 2.2).attr("stroke-opacity", 0.85 * handOff));
    const tip = D.gnode.get(F.steps[F.n_steps - 1].c); bot(g, tip.x, tip.y, 8).attr("opacity", handOff); }
  if (nIn < N && curF > 0 && f < 0.95) { const cur = accOrder[nIn], rt = routeOf(cur), slow = nIn < SLOW, upto = slow ? curF * (rt.length + 0.999) : rt.length;
    rt.forEach((e, t) => { if (t < upto) g.append("path").attr("d", gEdgePath(e)).attr("fill", "none").attr("stroke", D.pass(cur) ? PASS : FAIL).attr("stroke-width", slow ? 2.6 : 1.6).attr("stroke-opacity", (slow ? 0.9 : 0.5) * Math.min(1, upto - t)); });
    if (slow) { const tip = D.gnode.get(cur.steps[Math.min(cur.n_steps - 1, Math.floor(upto))].c); bot(g, tip.x, tip.y, 8); } }
  const boxOp = sub(q8, 0, 0.08) * (1 - hi);
  if (boxOp > 0) { const box = g.append("g").attr("transform", "translate(690,58)").attr("opacity", boxOp); box.append("rect").attr("width", 190).attr("height", 66).attr("rx", 10).attr("fill", "#fff").attr("fill-opacity", 0.92).attr("stroke", "#e3e3df");
    [[`rollouts   ${nIn} / ${N}`, 22, 700], [`states   ${accS.size}`, 40, 400], [`transitions   ${accE.size}`, 57, 400]].forEach(([t, y, w]) => box.append("text").attr("x", 14).attr("y", y).attr("font-size", 12).attr("font-weight", w).attr("fill", INK).text(t)); }
  g.append("text").attr("x", 20).attr("y", 432).attr("font-size", 11).attr("fill", MUTED).attr("opacity", 1 - hi).text("colour: of the rollouts passing through, how many succeed (red: none, green: all)");
  if (hi > 0) halo(g.append("text").attr("x", 20).attr("y", 432).attr("font-size", 12).attr("font-weight", 700).attr("fill", INK).attr("opacity", hi)).text("left in colour: states, and transitions between them, where at least four in five rollouts passing through end the same way");
}

// ---------- the finding (act 3) ----------
// Table 1 of the paper, for the closing chart: the best simple per-task classifier in each benchmark x model cell
let RES = null; fetch("results.json?v=5").then(r => r.json()).then(j => { RES = j; }).catch(() => {});
const BENCH_L = { tau2: "Tau2-Bench", bfclv3: "BFCLv3", agentclinic: "AgentClinic" }, MODEL_L = { deepseek: "DeepSeek-V3.2", qwen: "Qwen3-80B", gpt5mini: "GPT-5-mini" };
const METH_L = { NC: "nearest centroid", RBF: "RBF-SVM", FT1: "fastText, 1-gram", FT2: "fastText, 2-gram", FT5: "fastText, 5-gram" };
function findingScene(svg, f) {
  const g = svg.append("g"), C = D.clf, AX = 800, nH = C.heldOut.length, nT = C.train.length, opening = D.pairFP.f.opening || "";
  const q1 = sub(f, 0, 0.10), q2 = sub(f, 0.10, 0.25), q3 = sub(f, 0.25, 0.62), q4 = sub(f, 0.62, 0.74), q5 = sub(f, 0.74, 0.92);
  const x1 = sub(q1, 0, 0.3), sh = ease(sub(q1, 0.25, 0.65)), out = sub(q5, 0, 0.25), chartIn = sub(q5, 0.2, 0.45);
  const k = 1 - 0.2 * sh, toScr = (x, y) => [8 * sh + x * k, 22 * sh + y * k];
  // what the classifier learned: green leans toward success, red toward failure, grey carries little weight; it sweeps in left to right
  const wCol = (wv) => srColor(1 / (1 + Math.exp(-wv * Math.log(0.9 / 0.1) / C.refW)));
  const learnAmt = (wv, x) => { const s0 = clamp01((x - 40) / 820) * 0.6; return sub(q2, s0, s0 + 0.4) * Math.min(1, Math.abs(wv) / (0.6 * C.refW)); };
  drawBanner(g, opening, 1 - out);
  const mg = g.append("g").attr("opacity", x1 * (1 - out)).attr("transform", `translate(${8 * sh},${22 * sh}) scale(${k})`);
  D.gedges.forEach(e => { const wv = C.wOf(`${e.a}>${e.b}`), a = D.gnode.get(e.a), amt = learnAmt(wv, a.x);
    drawEdge(mg, e, d3.interpolateRgb(NEU, wCol(wv))(amt), 0.45 + 0.75 * amt, e.nr, 1.2 * amt); });
  D.gnodes.forEach(nd => { const wv = C.wOf(`s${nd.c}`), amt = learnAmt(wv, nd.x); drawState(mg, nd, 1, 1, d3.interpolateRgb(NEU, wCol(wv))(amt)); });
  noiseLabel(mg, 0.6);
  // act 2's last frame dissolves into this map
  if (x1 < 1) branchScene(g.append("g").attr("opacity", 1 - x1), 1);
  // the held-out third: a row of tokens whose outcomes are hidden
  const qy = 402, qx = (i) => 64 + i * 34, slowN = Math.min(3, nH);
  const win = (i) => i < slowN ? [0.04 + i * 0.24, 0.26 + i * 0.24] : [0.76 + (i - slowN) * 0.012, 0.79 + (i - slowN) * 0.012], local = (i) => sub(q3, ...win(i));
  const tg = g.append("g").attr("opacity", 1 - out);
  const qLab = sub(q1, 0.5, 0.7) * (1 - sub(q3, 0.85, 0.95)); if (qLab > 0) halo(tg.append("text").attr("x", qx(0) - 12).attr("y", qy - 20).attr("font-size", 10.5).attr("fill", MUTED).attr("opacity", qLab)).text(`held out: ${nH} rollouts, outcomes hidden`);
  C.heldOut.forEach((r, i) => { const app = sub(q1, 0.55 + i * 0.015, 0.7 + i * 0.015), gone = i < slowN ? sub(local(i), 0, 0.1) : sub(local(i), 0, 0.3); if (app <= 0 || gone >= 1) return;
    const tk = tg.append("g").attr("transform", `translate(${qx(i)},${qy})`).attr("opacity", app * (1 - gone));
    tk.append("rect").attr("x", -11).attr("y", -9).attr("width", 22).attr("height", 18).attr("rx", 9).attr("fill", "#fff").attr("stroke", "#b9b9b3").attr("stroke-width", 1.2);
    tk.append("text").attr("y", 4).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 700).attr("fill", MUTED).text("?"); });
  // the classifier's ranking: one slot per held-out rollout, the one it thinks most likely to succeed at the top
  const nS = C.heldOut.length, rankOf = new Map([...C.heldOut].sort((a, c) => C.score(c) - C.score(a)).map((r, i) => [r, i]));
  const RX = 806, PW = 46, PH = 13, slotY = (i) => 104 + i * ((372 - 104) / Math.max(1, nS - 1));
  const axIn = sub(q3, 0, 0.05) * (1 - out);
  if (axIn > 0) { const ag = g.append("g").attr("opacity", axIn);
    ag.append("text").attr("x", RX).attr("y", 72).attr("text-anchor", "middle").attr("font-size", 11.5).attr("font-weight", 700).attr("fill", INK).text("the classifier's ranking");
    ag.append("text").attr("x", RX).attr("y", 89).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).text("most likely to succeed");
    ag.append("text").attr("x", RX).attr("y", 392).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).text("most likely to fail");
    d3.range(nS).forEach(i => { ag.append("rect").attr("x", RX - PW / 2).attr("y", slotY(i) - PH / 2).attr("width", PW).attr("height", PH).attr("rx", PH / 2).attr("fill", "none").attr("stroke", "#dcdcd6").attr("stroke-dasharray", "2,2");
      ag.append("text").attr("x", RX - PW / 2 - 6).attr("y", slotY(i) + 3.5).attr("text-anchor", "end").attr("font-size", 9).attr("fill", MUTED).text(i + 1); }); }
  // a held-out rollout's token: "?" until its outcome is revealed, then a green check or a red cross
  const token = (gg, x, y, r, rv) => { const won = D.pass(r), fill = d3.interpolateRgb("#fff", won ? PASS : FAIL)(rv), t = gg.append("g").attr("transform", `translate(${x},${y})`);
    t.append("rect").attr("x", -PW / 2).attr("y", -PH / 2).attr("width", PW).attr("height", PH).attr("rx", PH / 2).attr("fill", fill).attr("stroke", rv > 0.5 ? fill : "#b9b9b3").attr("stroke-width", 1.2);
    t.append("text").attr("y", 3.6).attr("text-anchor", "middle").attr("font-size", 10).attr("font-weight", 700).attr("fill", rv > 0.5 ? "#fff" : MUTED).text(rv > 0.5 ? (won ? "✓" : "✗") : "?"); };
  // each held-out rollout is read along its path, then its token drops into its slot in the ranking
  const dg = g.append("g").attr("opacity", 1 - out);
  C.heldOut.forEach((r, i) => { const l = local(i); if (l <= 0 || l >= 1) return; const n = r.n_steps, rt = routeOf(r), sy = slotY(rankOf.get(r));
    if (i < slowN) { const u = sub(l, 0.1, 0.8) * n, settle = ease(sub(l, 0.8, 1)), routeOp = 1 - sub(l, 0.85, 1);
      rt.forEach((e, t) => { if (t + 1 < u) mg.append("path").attr("d", gEdgePath(e)).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 2.4).attr("stroke-opacity", 0.85 * routeOp * Math.min(1, u - t - 1)); });
      const nd = D.gnode.get(r.steps[Math.min(n - 1, Math.max(0, Math.floor(u)))].c);
      if (l < 0.12) { const [sx, sy0] = toScr(nd.x, nd.y), lt = ease(sub(l, 0, 0.1)); bot(g, lerp(qx(i), sx, lt), lerp(qy, sy0, lt), 8); }
      else if (l < 0.85) bot(mg, nd.x, nd.y, 10).attr("opacity", 1 - sub(l, 0.78, 0.85));
      if (l >= 0.8) { const [ex, ey] = toScr(nd.x, nd.y); token(dg, lerp(ex, RX, settle), lerp(ey, sy, settle), r, 0); } }
    else { rt.forEach(e => mg.append("path").attr("d", gEdgePath(e)).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 1.4).attr("stroke-opacity", 0.35 * (1 - l)));
      const lt = ease(sub(l, 0.3, 0.7)); if (l >= 0.3) token(dg, lerp(qx(i), RX, lt), lerp(qy, sy, lt), r, 0); } });
  // placed tokens; at the reveal they turn over from the top of the ranking to the bottom
  C.heldOut.forEach((r, i) => { if (local(i) < 1) return; const k = rankOf.get(r), s0 = (k / nS) * 0.55; token(dg, RX, slotY(k), r, sub(q4, s0, s0 + 0.3)); });
  // captions, one per beat
  const caps = [[sub(q1, 0.6, 0.9) * (1 - sub(q2, 0, 0.1)), `the classifier will learn from the other ${nT} rollouts only`],
    [sub(q2, 0, 0.1) * (1 - sub(q3, 0, 0.05)), `colour: what a linear classifier over states and transitions learned from those ${nT} · green leans toward success, red toward failure`],
    [sub(q3, 0, 0.05) * (1 - sub(q4, 0, 0.1)), "it scores each held-out rollout from the states and transitions along its path, then ranks them"],
    [sub(q4, 0, 0.1) * (1 - out), "each token is one held-out rollout, in the order the classifier ranked it · ✓ succeeded, ✗ failed"]];
  caps.forEach(([op, t]) => { if (op > 0) g.append("text").attr("x", 20).attr("y", 432).attr("font-size", 11).attr("fill", MUTED).attr("opacity", op).text(t); });
  // the paper's Table 1: the best simple per-task classifier, on held-out rollouts of held-out tasks
  if (chartIn > 0 && RES) { const hg = g.append("g").attr("opacity", chartIn), rows = RES.table1.rows;
    const bs = ["tau2", "bfclv3", "agentclinic"], ms = ["deepseek", "qwen", "gpt5mini"], x0 = 250, y0 = 104, cw = 186, ch = 84, gap = 8;
    const col = d3.scaleSequential(d3.interpolateBlues).domain([0.45, 1.02]), best = d3.max(rows, r => r.methods[r.best_baseline][0]);
    hg.append("text").attr("x", x0).attr("y", 60).attr("font-size", 13).attr("font-weight", 700).attr("fill", INK).text("Simple per-task classifiers, across benchmarks and models");
    hg.append("text").attr("x", x0).attr("y", 78).attr("font-size", 11).attr("fill", MUTED).text("AUROC on held-out rollouts · best of five classifiers per cell · averaged over each cell's test tasks");
    ms.forEach((m, j) => hg.append("text").attr("x", x0 + j * (cw + gap) + cw / 2).attr("y", y0 - 8 + 0).attr("dy", 0).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", INK).text(MODEL_L[m]));
    bs.forEach((bn, i) => hg.append("text").attr("x", x0 - 14).attr("y", y0 + 4 + i * (ch + gap) + ch / 2).attr("text-anchor", "end").attr("font-size", 12).attr("font-weight", 700).attr("fill", INK).text(BENCH_L[bn]));
    bs.forEach((bn, i) => ms.forEach((m, j) => { const row = rows.find(r => r.benchmark === bn && r.model === m); if (!row) return;
      const v = row.methods[row.best_baseline][0], cq = ease(sub(q5, 0.3 + (i * 3 + j) * 0.03, 0.5 + (i * 3 + j) * 0.03)), cv = lerp(0.5, v, cq);
      const cg = hg.append("g").attr("transform", `translate(${x0 + j * (cw + gap)},${y0 + 4 + i * (ch + gap)})`).attr("opacity", 0.3 + 0.7 * sub(cq, 0, 0.2));
      cg.append("rect").attr("width", cw).attr("height", ch).attr("rx", 8).attr("fill", col(cv));
      cg.append("text").attr("x", cw / 2).attr("y", ch / 2 + 4).attr("text-anchor", "middle").attr("font-size", 22).attr("font-weight", 800).attr("fill", cv > 0.7 ? "#fff" : INK).text(d3.format(".2f")(cv));
      cg.append("text").attr("x", cw / 2).attr("y", ch / 2 + 22).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", cv > 0.7 ? "#e8eefc" : MUTED).text(METH_L[row.best_baseline] || row.best_baseline);
      if (v === best) cg.append("rect").attr("x", -3).attr("y", -3).attr("width", cw + 6).attr("height", ch + 6).attr("rx", 10).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 2.5).attr("opacity", sub(q5, 0.8, 0.95)); }));
    const ly = y0 + 4 + 3 * (ch + gap) + 14, lw = 3 * cw + 2 * gap, lgd = hg.append("g").attr("transform", `translate(${x0},${ly})`), gid = "auroc-scale";
    const grad = lgd.append("defs").append("linearGradient").attr("id", gid); d3.range(0, 1.01, 0.1).forEach(t => grad.append("stop").attr("offset", `${t * 100}%`).attr("stop-color", col(0.5 + 0.5 * t)));
    lgd.append("rect").attr("width", lw).attr("height", 6).attr("rx", 3).attr("fill", `url(#${gid})`);
    lgd.append("text").attr("x", 0).attr("y", 20).attr("font-size", 10).attr("fill", MUTED).text("0.5 · chance");
    lgd.append("text").attr("x", lw).attr("y", 20).attr("text-anchor", "end").attr("font-size", 10).attr("fill", MUTED).text("1.0 · every success ranked above every failure"); }
}

// ---------- the obstacle (act 4) ----------
// a second task under the same agent, so act 4 can show that nothing learned on the story's task carries over
let D2 = null;
function loadSecond() { const P = window.PrecogData; if (!P || !B) return;
  const cand = P.index.tasks.filter(t => t.benchmark === B.benchmark && t.policy === B.policy && t.task !== B.task); if (!cand.length) { D2 = null; return; }
  const t = cand.find(x => x.held_out && x.domain === B.domain) || cand.find(x => x.held_out) || cand[0];   // same domain if possible: even a similar task has its own states if (D2 && D2.task === t.task && D2.policy === t.policy) return;
  fetch(P.url(t)).then(r => r.json()).then(b2 => { D2 = { b: b2, G: graphOf(b2), task: t.task, policy: t.policy, id: t.task_id }; }).catch(() => { D2 = null; }); }
document.addEventListener("precog:index", loadSecond);
// Figure 3 of the paper: the best per-task classifier at k labeled rollouts per class (max of NC, RBF-SVM, fastText 1-gram)
const FEW_K = [1, 2, 4, 8], FEW_B = [["tau2", "Tau2-Bench", "#2b5fb3"], ["bfclv3", "BFCLv3", "#5f86cf"], ["agentclinic", "AgentClinic", "#97b0e0"]];
function obstacleScene(svg, f) {
  const g = svg.append("g"), C = D.clf, opening = D.pairFP.f.opening || "";
  const q1 = sub(f, 0, 0.22), q2 = sub(f, 0.22, 0.58), q3 = sub(f, 0.58, 0.86);
  const x1 = sub(q1, 0, 0.3), chartIn = sub(q2, 0, 0.12) * (1 - sub(q3, 0, 0.2)), morph = ease(sub(q3, 0.1, 0.6));
  // the map sits where act 3 left it, steps aside for the chart, then returns as the second task's map
  const side = ease(sub(q2, 0, 0.15)) * (1 - ease(sub(q3, 0, 0.25)));
  const k = lerp(0.8, 0.5, side), tx = lerp(8, 0, side), ty = lerp(22, 88, side);
  const wCol = (wv) => srColor(1 / (1 + Math.exp(-wv * Math.log(0.9 / 0.1) / C.refW))), amtOf = (wv) => Math.min(1, Math.abs(wv) / (0.6 * C.refW));
  // the task banner: the old task's text fades out fully before the new one fades in, over a box that stays put
  if (D2 && morph > 0) g.append("rect").attr("x", 20).attr("y", 10).attr("width", 860).attr("height", 38).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df");
  drawBanner(g, opening, x1 * (D2 ? 1 - sub(morph, 0, 0.45) : 1 - morph));   // act 3 ends without the banner; it returns with the map
  if (D2 && morph > 0.55) drawBanner(g, D2.b.rollouts.find(r => r.opening)?.opening || D2.id, sub(morph, 0.55, 1));
  const mg = g.append("g").attr("opacity", x1 * (1 - morph)).attr("transform", `translate(${tx},${ty}) scale(${k})`);
  D.gedges.forEach(e => { const wv = C.wOf(`${e.a}>${e.b}`), a = amtOf(wv); drawEdge(mg, e, d3.interpolateRgb(NEU, wCol(wv))(a), 0.45 + 0.75 * a, e.nr, 1.2 * a); });
  D.gnodes.forEach(nd => { const wv = C.wOf(`s${nd.c}`); drawState(mg, nd, 1, 1, d3.interpolateRgb(NEU, wCol(wv))(amtOf(wv))); });
  noiseLabel(mg, 0.6);
  if (morph > 0) { const m2 = g.append("g").attr("opacity", morph).attr("transform", `translate(${tx},${ty}) scale(${k})`);
    if (D2) { D2.G.gedges.forEach(e => drawEdge(m2, e, NEU, 0.6, e.nr, 0, D2.G)); D2.G.gnodes.forEach(nd => drawState(m2, nd, 1, 1, NEU)); noiseLabel(m2, 0.6, D2.G); }
    else { D.gedges.forEach(e => drawEdge(m2, e, NEU, 0.3)); D.gnodes.forEach(nd => drawState(m2, nd, 1, 0.5, NEU)); } }
  // act 3's last frame dissolves into this one
  if (x1 < 1) findingScene(g.append("g").attr("opacity", 1 - x1), 1);
  // the labeled rollouts behind the classifier: all of them, then one batch's worth, then fewer, then none on the new task
  const lab = [...C.train].sort((a, c) => a.rid - c.rid), rnd = ((sd) => () => (sd = (sd * 1664525 + 1013904223) % 4294967296) / 4294967296)(1);
  const order = new Map(); { const P = lab.filter(D.pass), Fl = lab.filter(r => !D.pass(r)); [P, Fl].forEach(arr => { const a = [...arr]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } a.forEach((r, i) => order.set(r, i)); }); }
  const steps = [[0.2, 8], [0.4, 4], [0.6, 2], [0.8, 1]], stepAt = steps.filter(([t]) => q2 >= t).pop(), perClass = q3 > 0.15 ? 0 : stepAt ? stepAt[1] : Infinity;
  const nLab = perClass === Infinity ? lab.length : Math.min(lab.filter(D.pass).length, perClass) + Math.min(lab.filter(r => !D.pass(r)).length, perClass);
  const rowIn = sub(q1, 0.4, 0.7), ry = 404, rx = (i) => 64 + i * 13;
  if (rowIn > 0) { const rg = g.append("g").attr("opacity", rowIn);
    halo(rg.append("text").attr("x", 52).attr("y", ry - 16).attr("font-size", 10.5).attr("fill", MUTED)).text(perClass === 0 ? "labeled rollouts of this task: 0" : `labeled rollouts: ${nLab}${perClass !== Infinity ? ` (${perClass} per class)` : ""}`);
    lab.forEach((r, i) => { const keep = perClass === Infinity || order.get(r) < perClass; const fade = keep ? 1 : 0.12, gone = sub(q3, 0.1, 0.3);
      rg.append("circle").attr("cx", rx(i)).attr("cy", ry).attr("r", 4.6).attr("fill", D.pass(r) ? PASS : FAIL).attr("opacity", fade * (1 - gone)); });
    if (sub(q3, 0.2, 0.4) > 0) lab.forEach((r, i) => rg.append("circle").attr("cx", rx(i)).attr("cy", ry).attr("r", 4.2).attr("fill", "none").attr("stroke", "#c9c9c3").attr("stroke-dasharray", "2,2").attr("opacity", sub(q3, 0.2, 0.4))); }
  // the paper's Figure 3: fewer labels, lower AUROC, on every benchmark
  if (chartIn > 0 && RES && RES.fewshot) { const cg = g.append("g").attr("opacity", chartIn), X0 = 540, X1 = 790, Y0 = 350, Y1 = 110;
    const xs = d3.scaleLog().base(2).domain([1, 8]).range([X0, X1]), ys = d3.scaleLinear().domain([0.6, 0.95]).range([Y0, Y1]);
    cg.append("text").attr("x", X0 - 30).attr("y", 72).attr("font-size", 13).attr("font-weight", 700).attr("fill", INK).text("Fewer labels, worse predictions");
    cg.append("text").attr("x", X0 - 30).attr("y", 89).attr("font-size", 10.5).attr("fill", MUTED).text("best per-task classifier · AUROC on held-out tasks · paper, Figure 3");
    [0.6, 0.7, 0.8, 0.9].forEach(v => { cg.append("line").attr("x1", X0 - 8).attr("x2", X1).attr("y1", ys(v)).attr("y2", ys(v)).attr("stroke", "#ecece8");
      cg.append("text").attr("x", X0 - 14).attr("y", ys(v) + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", MUTED).text(v.toFixed(1)); });
    FEW_K.forEach(kk => cg.append("text").attr("x", xs(kk)).attr("y", Y0 + 18).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", INK).text(kk));
    cg.append("text").attr("x", (X0 + X1) / 2).attr("y", Y0 + 36).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", MUTED).text("labeled rollouts per class");
    // a GRPO batch: 16 rollouts, so at most 8 per class
    cg.append("line").attr("x1", xs(8)).attr("x2", xs(8)).attr("y1", Y1 - 4).attr("y2", Y0).attr("stroke", MUTED).attr("stroke-dasharray", "3,3");
    cg.append("text").attr("x", xs(8) - 6).attr("y", Y0 - 8).attr("text-anchor", "end").attr("font-size", 10).attr("fill", MUTED).text("one GRPO batch: 16 rollouts");
    const shownK = stepAt ? stepAt[1] : 16, upTo = FEW_K.filter(kk => kk >= shownK);
    if (stepAt) cg.append("line").attr("x1", xs(shownK)).attr("x2", xs(shownK)).attr("y1", Y1).attr("y2", Y0).attr("stroke", INK).attr("stroke-width", 1.2).attr("opacity", 0.35);
    FEW_B.forEach(([bn, label, col]) => { const byK = RES.fewshot[bn]; if (!byK) return; const best = (kk) => d3.max(["NC", "RBF", "FT1"], m => byK[String(kk)][m]);
      const pts = upTo.map(kk => [xs(kk), ys(best(kk))]);
      if (pts.length > 1) cg.append("path").attr("d", d3.line()(pts)).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2.5);
      pts.forEach(([x, y]) => cg.append("circle").attr("cx", x).attr("cy", y).attr("r", 4).attr("fill", col));
      if (pts.length) cg.append("text").attr("x", xs(8) + 8).attr("y", ys(best(8)) + 4).attr("font-size", 10.5).attr("fill", col).attr("font-weight", 700).text(label); }); }
  // captions, one per beat
  const caps = [[sub(q1, 0.6, 0.9) * (1 - sub(q2, 0, 0.1)), `the classifier from act 3: fit on ${lab.length} labeled rollouts of this task`],
    [sub(q2, 0, 0.1) * (1 - sub(q3, 0, 0.1)), "the classifiers here: nearest centroid, RBF-SVM and fastText, refit for each task; best of the three shown"],
    [sub(q3, 0.3, 0.5), D2 ? `a different task, ${D2.id}, same agent: ${D2.G.gnodes.filter(n => !n.noise).length} states of its own · nothing learned on ${B.task_id} applies` : "a different task: none of these states exist there, so nothing learned applies"]];
  caps.forEach(([op, t]) => { if (op > 0) g.append("text").attr("x", 20).attr("y", 432).attr("font-size", 11).attr("fill", MUTED).attr("opacity", op).text(t); });
}

// ---------- the idea (act 5): Precog-RM, run on act 4's new task ----------
// The model scene from pipeline.js (architecture tour, one rollout's point settling, sixteen labeled rollouts becoming two
// prototypes), drawn for the second task when it is loaded: a task the encoder never trained on. Act 4's last frame dissolves into it.
function modelScene(svg, f) {
  const g = svg.append("g"), P = window.PrecogPipeline; if (!P) return;
  const S = (D2 && (D2.S || (D2.S = P.buildScene(D2.b)))) || P.scene; if (!S) return;
  // act 4's last frame fades out first, then the model fades in, so the two never sit on top of each other
  const outA4 = sub(f, 0, 0.025), inM = sub(f, 0.02, 0.05), mg = g.append("g").attr("opacity", inM);
  P.renderScene(mg.append("g").node(), S, 0.38 + 0.40 * f, false, { noIntro: true });
  if (outA4 < 1) obstacleScene(g.append("g").attr("opacity", 1 - outA4), 1);
}

// ---------- the state-map panel (act 3's "look inside"): any task's global state graph, coloured by outcome ----------
const graphCache = new WeakMap();
function graphOf(b) { if (!graphCache.has(b)) graphCache.set(b, buildGraph(b.rollouts, (r) => r.reward >= 0.5, new Map(b.clusters.map(c => [c.id, c.noise])))); return graphCache.get(b); }
function mapPanel(node, b, opt = {}) {
  const G = graphOf(b), svg = d3.select(node); svg.selectAll("*").remove(); svg.attr("viewBox", `0 0 ${W} ${H}`);
  const sel = opt.selected ?? null, eg = svg.append("g"), ng = svg.append("g");
  const baseOp = (e) => (0.22 + 0.5 * Math.min(1, e.nr / 20)) * (isNoiseE(e, G) ? 0.6 : 1);
  const edges = [...G.gedges.values()].sort((a, c) => a.nr - c.nr).map(e => drawEdge(eg, e, srColor(e.sr), 1, e.nr, 0, G).datum(e));
  edges.forEach(p => p.append("title").text((e => `state ${e.a} to state ${e.b} · ${e.nr} rollouts · ${e.np} succeed`)(p.datum())));
  const nodes = G.gnodes.map(nd => { const c = drawState(ng, nd, 1, 1, srColor(nd.sr)).datum(nd).attr("stroke-width", nd.c === sel ? 4.5 : 2.6).style("cursor", "pointer");
    c.append("title").text(`state ${nd.c}${nd.noise ? " · unclustered steps" : ""} · ${nd.n} steps from ${nd.nr} rollouts · ${nd.np} of those rollouts succeed`);
    c.on("mouseenter", () => focus(nd.c)).on("mouseleave", () => focus(sel)).on("click", () => opt.onSelect && opt.onSelect(nd.c)); return c; });
  if (sel != null && G.gnode.get(sel)) { const nd = G.gnode.get(sel); ng.append("circle").attr("cx", nd.x).attr("cy", nd.y).attr("r", nd.r + 6).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 2).style("pointer-events", "none"); }
  // hovering (or selecting) a state keeps its transitions and neighbours bright and fades the rest
  function focus(c) { const near = (nd) => c == null || nd.c === c || G.gedges.has(`${c}>${nd.c}`) || G.gedges.has(`${nd.c}>${c}`), on = (e) => c == null || e.a === c || e.b === c;
    edges.forEach(p => { const e = p.datum(); p.attr("stroke-opacity", Math.min(1, baseOp(e) * (c == null ? 1 : on(e) ? 1.8 : 0.15))); if (c != null && on(e)) p.raise(); });
    nodes.forEach(n => n.attr("opacity", near(n.datum()) ? 1 : 0.3)); }
  focus(sel);
  noiseLabel(svg, 0.9, G);
  svg.append("text").attr("x", 20).attr("y", 432).attr("font-size", 11).attr("fill", MUTED).text("left to right: the step at which a state tends to occur · size: steps merged into it · colour: share of the rollouts through it that succeed");
  return G;
}

// ---------- beats ----------
const beats = {
  a1(svg, f) { attemptScene(svg, f); },
  a2(svg, f) { branchScene(svg, f); },
  a3(svg, f) { findingScene(svg, f); },
  a4(svg, f) { obstacleScene(svg, f); },
  a5(svg, f) { modelScene(svg, f); },
  // B1: one attempt, ending in failure
  o1(svg, f) { const g = svg.append("g"), r = D.pairFP.f, n = Math.min(D.S, r.n_steps), prog = sub(f, 0.05, 0.85) * n, upto = Math.floor(prog);
    g.append("text").attr("x", 60).attr("y", 60).attr("font-size", 13).attr("fill", MUTED).text("one attempt at the task");
    lane(g, r, 220, 90, 760, upto, 9, D, 1, true, prog - upto);
    if (f > 0.9) { const t = g.append("text").attr("x", 450).attr("y", 330).attr("text-anchor", "middle").attr("font-size", 15).attr("fill", INK).attr("opacity", sub(f, 0.9, 1)).text("Somewhere along the way, one decision sealed it."); } },
  // B2: rewind, same states up to step k, then a different choice that ends in success
  o2(svg, f) { const g = svg.append("g"), { f: F, p: P, k } = D.pairFP, nF = Math.min(D.S, F.n_steps), nP = Math.min(D.S, P.n_steps);
    const xs = d3.scalePoint().domain(d3.range(D.S)).range([90, 760]).padding(0.5), iF = D.ic(F), iP = D.ic(P);
    const rewind = sub(f, 0, 0.18), replay = sub(f, 0.22, 0.85) * nP, upto = Math.floor(replay);
    g.append("text").attr("x", 60).attr("y", 50).attr("font-size", 13).attr("fill", MUTED).text(rewind < 1 ? "rewind" : "replay, and choose differently at one step");
    // the failed attempt, greyed
    g.append("line").attr("x1", 90).attr("x2", 760).attr("y1", 260).attr("y2", 260).attr("stroke", "#e3e3df");
    for (let t = 0; t < nF; t++) pair(g, xs(t), 260, iF[t][0], iF[t][1], 9, t < k ? 0.9 : 0.35);
    outcome(g, 806, 260, false, 22, 0.5);
    // the replay: shared states ride the same lane, then branch up
    const branchY = 150, bx = (t) => xs(t), by = (t) => t < k ? 260 : branchY;
    if (upto > 0) g.append("path").attr("d", d3.line().curve(d3.curveMonotoneX)(d3.range(Math.min(upto, nP)).map(t => [bx(t), by(t)]))).attr("fill", "none").attr("stroke", PASS).attr("stroke-width", 2.5).attr("stroke-opacity", 0.8);
    for (let t = 0; t < Math.min(upto, nP); t++) if (t >= k) pair(g, bx(t), branchY, iP[t][0], iP[t][1], 9);
    const botX = upto >= nP ? 776 : lerp(upto === 0 ? 90 : bx(upto - 1), bx(Math.min(upto, nP - 1)), replay - upto), botY = upto >= nP ? branchY : lerp(by(Math.max(0, upto - 1)), by(Math.min(upto, nP - 1)), replay - upto);
    if (rewind >= 1) bot(g, botX, botY, 14);
    if (upto >= nP) outcome(g, 806, branchY, true, 22);
    // the branch point
    if (upto >= k && k > 0) { const t = k - 1, pulse = 1 + 0.15 * Math.sin(f * 20); g.append("circle").attr("cx", xs(t)).attr("cy", 260).attr("r", 22 * pulse).attr("fill", "none").attr("stroke", ACT).attr("stroke-width", 2.5);
      g.append("text").attr("x", xs(t)).attr("y", 320).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 700).attr("fill", ACT).text("critical branch point");
      g.append("text").attr("x", xs(t)).attr("y", 338).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).text(`same ${k} states, then a different action`); } },
  // B3: mapping every path by re-sampling at every step, schematic
  o3(svg, f) { const g = svg.append("g"), r = D.pairFP.f, T = 8, bf = 3, x0 = 80, dx = 92, y0 = 230;
    const grow = sub(f, 0, 0.8), levels = grow * 3.5;                    // trunk, then up to three generations of branches
    const draw = (x, y, depth, seedA, seedB, remaining, op) => { if (remaining <= 0 || depth > 3) return; const show = clamp01(levels - depth); if (show <= 0) return;
      const nb = depth === 0 ? 1 : bf - 1; for (let j = 0; j < (depth === 0 ? remaining : 1); j++) { }
      // one segment forward per step; from each step, (bf-1) hypothetical branches of the next generation
      let cx = x, cy = y; const pts = [[cx, cy]];
      for (let t = 0; t < remaining; t++) { cx += dx; cy += depth === 0 ? 0 : (hash(seedA, t, seedB) - 0.5) * 60; pts.push([cx, cy]); }
      const L = pts.length - 1, segShown = show * L;
      g.append("path").attr("d", d3.line().curve(d3.curveMonotoneX)(pts.slice(0, Math.ceil(segShown) + 1))).attr("fill", "none").attr("stroke", depth === 0 ? INK : "#9a9a9a").attr("stroke-width", depth === 0 ? 3 : Math.max(0.6, 2 - depth * 0.5)).attr("stroke-opacity", op * (depth === 0 ? 1 : 0.7 - depth * 0.15));
      if (show >= 1) outcome(g, cx + 16, cy, hash(seedA, seedB, 7) > 0.65, depth === 0 ? 18 : 10, op * (depth === 0 ? 1 : 0.6));
      if (show >= 1) for (let t = 1; t < remaining; t++) for (let b = 0; b < bf - 1; b++) draw(pts[t][0], pts[t][1], depth + 1, seedA * 7 + t * 13 + b * 3, seedB + 11 * b + t, remaining - t, op); };
    draw(x0, y0, 0, 1, 1, T, 1);
    const count = Math.round(Math.pow(bf, Math.min(T, grow * T * 1.25)));
    g.append("text").attr("x", 60).attr("y", 50).attr("font-size", 13).attr("fill", MUTED).text("re-sample at every step to find every branch point");
    const box = g.append("g").attr("transform", "translate(640,20)"); box.append("rect").attr("width", 230).attr("height", 64).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df");
    box.append("text").attr("x", 14).attr("y", 22).attr("font-size", 11).attr("fill", MUTED).text(`rollouts to map ${T} steps, ${bf} choices each`);
    box.append("text").attr("x", 14).attr("y", 50).attr("font-size", 22).attr("font-weight", 800).attr("fill", INK).text(d3.format(",")(count) + (grow >= 0.98 ? "" : "…"));
    if (f > 0.85) g.append("text").attr("x", 450).attr("y", 420).attr("text-anchor", "middle").attr("font-size", 15).attr("fill", INK).attr("opacity", sub(f, 0.85, 1)).text(`${bf}^${T} = ${d3.format(",")(Math.pow(bf, T))} rollouts for one task. Exponential.`); },
  // B4: fifty independent rollouts
  o4(svg, f) { const g = svg.append("g"), N = D.rolls.length, run = sub(f, 0.05, 0.85);
    g.append("text").attr("x", 60).attr("y", 24).attr("font-size", 13).attr("fill", MUTED).text(`${N} independent rollouts of the same task`);
    D.rolls.forEach((r, i) => { const y = D.laneY(i), n = Math.min(D.S, r.n_steps), li = clamp01(run * 1.3 - i * 0.006), shown = Math.floor(li * (n + 0.999)), xs = d3.scalePoint().domain(d3.range(D.S)).range([90, 760]).padding(0.5), icons = D.ic(r);
      g.append("line").attr("x1", 90).attr("x2", 760).attr("y1", y).attr("y2", y).attr("stroke", "#eceae5");
      for (let t = 0; t < Math.min(n, shown); t++) pair(g, xs(t), y, icons[t][0], icons[t][1], 2.6);
      if (shown < n && li > 0) bot(g, xs(Math.max(0, shown - 1)) + 10, y, 3.5); if (shown >= n) outcome(g, 790, y, D.pass(r), 8); });
    if (f > 0.9) g.append("text").attr("x", 450).attr("y", 432).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", INK).attr("opacity", sub(f, 0.9, 1)).text("No two share a branch point exactly."); },
  // B5: tuples snap into shared semantic states; lanes become routes through one DAG
  o5(svg, f) { const g = svg.append("g"), N = D.rolls.length, snap = ease(sub(f, 0.1, 0.6)), routes = sub(f, 0.55, 0.95);
    const xs = d3.scalePoint().domain(d3.range(D.S)).range([90, 760]).padding(0.5);
    g.append("text").attr("x", 60).attr("y", 24).attr("font-size", 13).attr("fill", MUTED).text(snap < 1 ? "similar actions and observations snap into states" : "fifty routes through one map");
    // edges (routes) drawn in once nodes have formed
    if (routes > 0) D.edges.forEach(e => { const a = D.node(e.t, e.a), b = D.node(e.t + 1, e.b); if (!a || !b) return;
      g.append("path").attr("d", `M${a.x},${a.y} C${(a.x + b.x) / 2},${a.y} ${(a.x + b.x) / 2},${b.y} ${b.x},${b.y}`).attr("fill", "none").attr("stroke", srColor(e.np / e.n)).attr("stroke-width", 0.8 + Math.sqrt(e.n) * 1.2).attr("stroke-opacity", 0.55 * routes); });
    D.rolls.forEach((r, i) => { const y = D.laneY(i), n = Math.min(D.S, r.n_steps), icons = D.ic(r);
      for (let t = 0; t < n; t++) { const nd = D.node(t, r.steps[t].c), k = hash(i, t, 5); const tx = lerp(xs(t), nd.x + (k - 0.5) * 6, snap), ty = lerp(y, nd.y + (hash(i, t, 9) - 0.5) * 6, snap);
        pair(g, tx, ty, icons[t][0], icons[t][1], 2.6, 1 - 0.85 * snap); } });
    if (snap > 0) D.cols.forEach(col => col.nodes.forEach(nd => { const r = 5 + Math.sqrt(nd.n) * 2.2;
      g.append("circle").attr("cx", nd.x).attr("cy", nd.y).attr("r", r * snap).attr("fill", "#fff").attr("stroke", D.noise.get(nd.c) ? NOISE_STROKE : srColor(D.sr.get(nd.c))).attr("stroke-width", 3).attr("stroke-dasharray", D.noise.get(nd.c) ? "3,2" : null).attr("opacity", snap); }));
    if (snap >= 1) d3.range(D.S).forEach(t => g.append("text").attr("x", D.xStep(t)).attr("y", 428).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).text(t + 1)); },
  // B6: the map accrues one rollout at a time; edges thicken and tint; branch points surface
  o6(svg, f) { const g = svg.append("g"), N = D.rolls.length, stream = sub(f, 0.04, 0.82) * N, nIn = Math.min(N, Math.floor(stream)), cur = D.rolls[Math.min(N - 1, nIn)], curF = stream - nIn;
    const cnt = new Map(), nodeSeen = new Map(); let nEdges = 0;
    for (let i = 0; i < nIn; i++) { const r = D.rolls[i]; for (let t = 0; t + 1 < Math.min(D.S, r.n_steps); t++) { const k = `${t}:${r.steps[t].c}>${r.steps[t + 1].c}`; if (!cnt.has(k)) { cnt.set(k, { n: 0, np: 0, t, a: r.steps[t].c, b: r.steps[t + 1].c }); nEdges++; } const e = cnt.get(k); e.n++; if (D.pass(r)) e.np++; }
      for (let t = 0; t < Math.min(D.S, r.n_steps); t++) nodeSeen.set(`${t}:${r.steps[t].c}`, (nodeSeen.get(`${t}:${r.steps[t].c}`) || 0) + 1); }
    g.append("text").attr("x", 60).attr("y", 24).attr("font-size", 13).attr("fill", MUTED).text("every rollout strengthens the map");
    cnt.forEach(e => { const a = D.node(e.t, e.a), b = D.node(e.t + 1, e.b); g.append("path").attr("d", `M${a.x},${a.y} C${(a.x + b.x) / 2},${a.y} ${(a.x + b.x) / 2},${b.y} ${b.x},${b.y}`).attr("fill", "none").attr("stroke", srColor(e.np / e.n)).attr("stroke-width", 0.8 + Math.sqrt(e.n) * 1.4).attr("stroke-opacity", 0.6); });
    D.cols.forEach(col => col.nodes.forEach(nd => { const seen = nodeSeen.get(`${col.t}:${nd.c}`) || 0; if (!seen) return; const r = 5 + Math.sqrt(seen) * 2.2;
      g.append("circle").attr("cx", nd.x).attr("cy", nd.y).attr("r", r).attr("fill", "#fff").attr("stroke", D.noise.get(nd.c) ? NOISE_STROKE : "#c9c9c3").attr("stroke-width", 2); }));
    // the rollout currently arriving: its route draws in step by step
    if (nIn < N && curF > 0) { const n = Math.min(D.S, cur.n_steps), upto = Math.min(n, Math.floor(curF * (n + 0.999))), pts = d3.range(upto).map(t => { const nd = D.node(t, cur.steps[t].c); return [nd.x, nd.y]; });
      if (pts.length > 1) g.append("path").attr("d", d3.line().curve(d3.curveMonotoneX)(pts)).attr("fill", "none").attr("stroke", D.pass(cur) ? PASS : FAIL).attr("stroke-width", 3).attr("stroke-opacity", 0.9);
      if (pts.length) bot(g, pts[pts.length - 1][0], pts[pts.length - 1][1], 9); }
    // counters
    const box = g.append("g").attr("transform", "translate(640,34)"); box.append("rect").attr("width", 230).attr("height", 70).attr("rx", 10).attr("fill", "#fff").attr("fill-opacity", 0.92).attr("stroke", "#e3e3df");
    [[`rollouts   ${nIn} / ${N}`, 22], [`states   ${nodeSeen.size}`, 42], [`edges   ${nEdges}`, 60]].forEach(([t, y]) => box.append("text").attr("x", 14).attr("y", y).attr("font-size", 12).attr("fill", INK).attr("font-weight", y === 22 ? 700 : 400).text(t));
    // the branch points, once the map is complete
    const reveal = sub(f, 0.86, 1);
    if (reveal > 0) D.branchPts.forEach((bp, i) => { const nd = D.node(bp.t, bp.c); g.append("circle").attr("cx", nd.x).attr("cy", nd.y).attr("r", 16 + 4 * Math.sin(f * 25 + i)).attr("fill", "none").attr("stroke", ACT).attr("stroke-width", 2.5).attr("opacity", reveal); });
    if (reveal > 0) g.append("text").attr("x", 450).attr("y", 432).attr("text-anchor", "middle").attr("font-size", 14).attr("fill", ACT).attr("font-weight", 700).attr("opacity", reveal).text(`${D.branchPts.length} branch points surface, from ${N} rollouts, without re-sampling`); },
};

function draw(kind, node, frac) {
  if (!D) return; const svg = d3.select(node); svg.selectAll("*").remove(); if (node.tagName === "svg") svg.attr("viewBox", `0 0 ${W} ${H}`);
  if (beats[kind]) beats[kind](svg, frac);
}
// token estimate for the focus rollout's first steps (chars / 4), for the hero counter
function tokensUpTo(r, upto) { let n = 0; for (let t = 0; t < Math.min(upto, r.n_steps); t++) n += (r.steps[t].a.length + r.steps[t].o.length) / 4; return Math.round(n); }
window.PrecogTokens = tokensUpTo;
function bindText() { const r = D.pairFP.f, gr = r.grader;
  document.querySelectorAll('[data-bind="n-steps"]').forEach(el => el.textContent = r.n_steps);
  const OK = { DB: "the database ended up right", COMMUNICATE: "the user got the information they needed", ACTION: "every required action was taken", NL_ASSERTION: "the behaviour checks passed", ENV_ASSERTION: "the environment checks passed" };
  const BAD = { DB: "the database ended up wrong", COMMUNICATE: "the user never got information they needed", ACTION: "a required action was missed", NL_ASSERTION: "a behaviour check failed", ENV_ASSERTION: "an environment check failed" };
  const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
  document.querySelectorAll('[data-bind="grader-summary"]').forEach(el => { if (!gr) { el.textContent = ""; return; }
    const ok = gr.basis.filter(b => (gr.breakdown[b] ?? 0) >= 0.5).map(b => OK[b] || `${b.toLowerCase()} passed`), bad = gr.basis.filter(b => (gr.breakdown[b] ?? 0) < 0.5).map(b => BAD[b] || `${b.toLowerCase()} failed`);
    el.textContent = cap(ok.length && bad.length ? `${ok.join(" and ")}, but ${bad.join(" and ")}.` : `${(bad.length ? bad : ok).join(" and ")}.`); }); }
function load(b) { B = b; D = prep(b); bindText(); loadSecond(); document.dispatchEvent(new CustomEvent("precog:opening")); }
document.addEventListener("precog:bundle", (e) => load(e.detail));
if (window.PRECOG_BUNDLE) load(window.PRECOG_BUNDLE);
window.PrecogOpening = { draw, mapPanel, get data() { return D; } };
})();
