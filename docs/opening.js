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
  return { rolls, S, sr, noise, pass, ic, pairFP: best, cols, node, edges, branchPts, laneY: (i) => 36 + i * (390 / (rolls.length - 1)), xStep: (t) => 70 + t * (780 / (S - 1)) };
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

// ---------- beats ----------
const beats = {
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
function load(b) { B = b; D = prep(b); document.dispatchEvent(new CustomEvent("precog:opening")); }
document.addEventListener("precog:bundle", (e) => load(e.detail));
if (window.PRECOG_BUNDLE) load(window.PRECOG_BUNDLE);
window.PrecogOpening = { draw, get data() { return D; } };
})();
