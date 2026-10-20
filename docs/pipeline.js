/* Opener: a near-wordless "video" of the pipeline, driven by a scrubber in [0,1].
   1 one task, one attempt (agent acts, environment answers) · 2 same task, sixteen attempts ·
   3 same moves become states, rollouts become routes · 4 encoder folds a route into a point ·
   5 prototypes from one labeled batch · 6 scoring a new rollout as it unfolds · 7 the gate.
   Geometry comes from the loaded task bundle (real states, latents, stop step); icons from the real tool calls. */
(() => {
const PASS = "#2a9d4a", FAIL = "#d13b3b", NOISE = "#b9b9b9", NOISE_STROKE = "#7b3fb3", INK = "#1c1c1c", MUTED = "#8a8a86";
const srColor = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);
const W = 900, H = 440, STEPS = 12, K_SUP = 8, INTRO_STEPS = 8, LANES = 16, DUR = 72000;
const BEATS = [
  { t0: 0.00, t1: 0.17, title: "One task, one attempt", text: "The agent acts, the environment answers. The outcome arrives only at the end." },
  { t0: 0.17, t1: 0.30, title: "Same task, sixteen attempts", text: "Different moves, different endings." },
  { t0: 0.30, t1: 0.42, title: "Same moves become states", text: "Each rollout is now a route through the state map. Red states lead to failure." },
  { t0: 0.42, t1: 0.54, title: "The encoder folds a route into a point", text: "Routes that end alike land near each other." },
  { t0: 0.54, t1: 0.64, title: "Two prototypes from one labeled batch", text: "No fitting: the mean of the passes and the mean of the failures." },
  { t0: 0.64, t1: 0.86, title: "Scoring a new rollout as it unfolds", text: "Its point moves after every step; distance to the prototypes gives P(pass)." },
  { t0: 0.86, t1: 1.00, title: "Confident enough: stop", text: "The predicted outcome becomes the reward. The rest is never generated." },
];
const TASK_GLOSS = { "retail/103": "A customer wants to return two items from a delivered order and fix a second order." };
const TOOL_ICON = [[/find_user|get_user/, "🪪"], [/get_order/, "📦"], [/get_product|list_all_product/, "🏷️"], [/return_/, "↩️"], [/exchange_/, "🔁"], [/modify_.*address/, "🏠"], [/modify_/, "✏️"], [/cancel_/, "🚫"], [/transfer_to_human/, "🙋"], [/calculate/, "🧮"], [/send_certificate|refund/, "💳"]];
const ICON_LABEL = { "💬": "message to the user", "🪪": "look up the user", "📦": "order details", "🏷️": "product details", "↩️": "return items", "🔁": "exchange items", "🏠": "change address", "✏️": "modify order", "🚫": "cancel order", "🙋": "hand off to a human", "🧮": "calculate", "💳": "payment", "🔧": "tool call" };
function actionIcon(a) {
  const m = /^\s*([a-z_]+)\s*\(/.exec(a) || /"name"\s*:\s*"([a-z_]+)"/.exec(a);
  if (!m) return "💬";
  for (const [re, ic] of TOOL_ICON) if (re.test(m[1])) return ic;
  return "🔧";
}
const obsIcon = (o) => /^\s*\[user\]/.test(o) ? "🧑" : /error|invalid|not found|failed/i.test(o.slice(0, 200)) ? "⚠️" : "🗄️";

let B = null, scene = null, playing = false, prog = 0, raf = null;
const $ = (s) => document.querySelector(s);
const lerp = (a, b, t) => a + (b - a) * t, ease = d3.easeCubicInOut, clamp01 = (x) => Math.max(0, Math.min(1, x));
const dot = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; };
const dist = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; } return Math.sqrt(s); };
const mean = (vs) => { const m = new Array(vs[0].length).fill(0); vs.forEach(v => v.forEach((x, k) => m[k] += x / vs.length)); return m; };

function buildScene(b) {
  const E = b.encoders.prefix, rolls = b.rollouts, idx = new Map(rolls.map((r, i) => [r.rid, i]));
  const pass = new Map(rolls.map(r => [r.rid, r.reward >= 0.5]));
  const zF = (rid) => { const z = E.z[idx.get(rid)]; return z[z.length - 1]; };
  const proj = (v) => { const p = E.pca; const c = v.map((x, k) => x - p.mean[k]); return [dot(c, p.components[0]), dot(c, p.components[1])]; };
  const held = new Set(b.cv_folds[0]), train = rolls.filter(r => !held.has(r.rid));
  const sup = [...train.filter(r => pass.get(r.rid)).slice(0, K_SUP), ...train.filter(r => !pass.get(r.rid)).slice(0, K_SUP)];
  // interleave so lanes alternate outcomes
  const trains = []; for (let i = 0; i < K_SUP; i++) { trains.push(sup[i]); trains.push(sup[K_SUP + i]); }
  const supIds = new Set(sup.map(r => r.rid));
  const cp = mean(sup.filter(r => pass.get(r.rid)).map(r => zF(r.rid))), cf = mean(sup.filter(r => !pass.get(r.rid)).map(r => zF(r.rid)));
  const score = (z) => 1 / (1 + Math.exp(dist(z, cp) - dist(z, cf)));
  const thr = b.stop_threshold;
  let best = null;
  rolls.filter(r => held.has(r.rid)).forEach(r => {
    const curve = E.z[idx.get(r.rid)].map(score);
    let stop = null; for (let t = 0; t < curve.length; t++) if (curve[t] > thr || 1 - curve[t] > thr) { stop = t; break; }
    if (stop == null || stop < 3) return;
    if ((curve[stop] > thr) !== pass.get(r.rid)) return;
    const frac = (stop + 1) / r.n_steps; if (!best || frac < best.frac) best = { r, curve, stop, frac };
  });
  if (!best) { const r = rolls.find(x => held.has(x.rid)); best = { r, curve: E.z[idx.get(r.rid)].map(score), stop: null, frac: 1 }; }
  // stations: per step, states ordered by success rate (success on top); station icon = most common action icon among its steps
  const sr = new Map(b.clusters.map(c => [c.id, c.sr])), noise = new Map(b.clusters.map(c => [c.id, c.noise]));
  const icons = new Map(trains.map(r => [r.rid, r.steps.map(s => actionIcon(s.a))]));
  const obs = new Map(trains.map(r => [r.rid, r.steps.map(s => obsIcon(s.o))]));
  const xs = d3.scalePoint().domain(d3.range(STEPS)).range([70, 560]).padding(0.4);
  const stations = [], stPos = new Map();
  for (let t = 0; t < STEPS; t++) {
    const present = new Map();
    trains.forEach(r => { if (t < r.n_steps) { const c = r.steps[t].c; if (!present.has(c)) present.set(c, []); present.get(c).push(icons.get(r.rid)[t]); } });
    const order = [...present.keys()].sort((a, c) => (sr.get(c) - sr.get(a)) || (a - c));
    const ys = d3.scalePoint().domain(order).range([70, 370]).padding(0.5);
    order.forEach(c => { const ic = present.get(c), top = [...d3.rollup(ic, v => v.length, d => d)].sort((p, q) => q[1] - p[1])[0][0];
      const s = { t, c, x: xs(t), y: order.length === 1 ? 220 : ys(c), n: ic.length, icon: top }; stations.push(s); stPos.set(`${t}:${c}`, s); });
  }
  const routes = trains.map((r, i) => { const off = ((i % 4) - 1.5) * 2.2;
    return { rid: r.rid, pass: pass.get(r.rid), pts: d3.range(Math.min(STEPS, r.n_steps)).map(t => { const s = stPos.get(`${t}:${r.steps[t].c}`); return [s.x, s.y + off]; }) }; });
  const finals = new Map(rolls.map(r => [r.rid, proj(zF(r.rid))]));
  const trail = E.z[idx.get(best.r.rid)].map(proj);
  const fx = [...finals.values()].map(p => p[0]), fy = [...finals.values()].map(p => p[1]);
  const padx = (d3.max(fx) - d3.min(fx)) * 0.1, pady = (d3.max(fy) - d3.min(fy)) * 0.1;
  const lx = d3.scaleLinear().domain([d3.min(fx) - padx, d3.max(fx) + padx]).range([150, 780]), ly = d3.scaleLinear().domain([d3.min(fy) - pady, d3.max(fy) + pady]).range([380, 70]);
  const L = (p) => [lx(p[0]), ly(p[1])];
  const latent = new Map([...finals].map(([rid, p]) => [rid, L(p)]));
  const hull = (ids) => { const pts = ids.map(rid => latent.get(rid)); return d3.polygonHull(pts) || pts; };
  const laneY = (i) => 46 + i * (344 / (LANES - 1));
  return { rolls, pass, supIds, sup, trains, stations, stPos, routes, latent, protoP: L(proj(cp)), protoF: L(proj(cf)),
           hullP: hull(sup.filter(r => pass.get(r.rid)).map(r => r.rid)), hullF: hull(sup.filter(r => !pass.get(r.rid)).map(r => r.rid)),
           query: best, trailL: trail.map(L), thr, sr, noise, xs, icons, obs, laneY, gloss: TASK_GLOSS[b.task_id] || `${b.benchmark} · task ${b.task_id}` };
}

function beatAt(p) { for (let i = 0; i < BEATS.length; i++) if (p < BEATS[i].t1 || i === BEATS.length - 1) return [i, clamp01((p - BEATS[i].t0) / (BEATS[i].t1 - BEATS[i].t0))]; }
const glyph = (g, x, y, ch, size, op = 1) => g.append("text").attr("x", x).attr("y", y).attr("text-anchor", "middle").attr("dominant-baseline", "central").attr("font-size", size).attr("opacity", op).text(ch);

function render(p) {
  if (!scene) return;
  const svg = d3.select("#pipeline"); svg.selectAll("*").remove(); svg.attr("viewBox", `0 0 ${W} ${H}`);
  const S = scene, [bi, q] = beatAt(p), g = svg.append("g");
  $("#pipeline-title").textContent = BEATS[bi].title; $("#pipeline-text").textContent = BEATS[bi].text;
  document.querySelectorAll("#pipeline-beats button").forEach((el, i) => el.classList.toggle("on", i === bi));
  const zoom = bi < 1 ? 0 : bi === 1 ? ease(clamp01(q / 0.35)) : 1;          // one lane -> sixteen lanes
  const group = bi < 2 ? 0 : bi === 2 ? ease(q) : 1;                           // icons -> stations
  const morph = bi < 3 ? 0 : bi === 3 ? ease(q) : 1;                           // routes -> points
  const district = bi < 4 ? 0 : bi === 4 ? ease(q) : 1;
  const travel = bi < 5 ? 0 : bi === 5 ? q : 1;
  const gate = bi < 6 ? 0 : ease(q);
  if (bi <= 2) drawIntro(g, S, bi, q, zoom, group);
  // stations (from beat 3, fade during the fold)
  if (bi >= 2 && morph < 1) {
    const st = g.append("g").attr("opacity", (bi === 2 ? clamp01((q - 0.25) / 0.5) : 1) * (1 - morph));
    const sg = st.selectAll("g").data(S.stations).join("g").attr("transform", d => `translate(${d.x},${d.y})`);
    sg.append("circle").attr("r", d => 9 + Math.sqrt(d.n) * 2.6).attr("fill", "#fff").attr("stroke", d => S.noise.get(d.c) ? NOISE_STROKE : srColor(S.sr.get(d.c))).attr("stroke-width", 3.5).attr("stroke-dasharray", d => S.noise.get(d.c) ? "3,2" : null);
    sg.append("text").attr("text-anchor", "middle").attr("dominant-baseline", "central").attr("font-size", d => 9 + Math.sqrt(d.n) * 1.6).text(d => d.icon);
    st.selectAll("text.ax").data(d3.range(STEPS)).join("text").attr("class", "ax").attr("x", t => S.xs(t)).attr("y", 405).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).text(t => t + 1);
  }
  // routes: appear in the second half of beat 3, fold into points in beat 4
  const routeOp = bi < 2 ? 0 : bi === 2 ? clamp01((q - 0.5) / 0.5) : 1;
  if (routeOp > 0) S.routes.forEach(r => {
    const target = S.latent.get(r.rid);
    const pts = r.pts.map(pt => [lerp(pt[0], target[0], morph), lerp(pt[1], target[1], morph)]);
    g.append("path").attr("d", d3.line().curve(d3.curveMonotoneX)(pts)).attr("fill", "none").attr("stroke", r.pass ? PASS : FAIL).attr("stroke-width", 2).attr("stroke-opacity", routeOp * (0.7 - 0.3 * morph));
    const head = pts[pts.length - 1];
    g.append("circle").attr("cx", head[0]).attr("cy", head[1]).attr("r", 4.5 + 1.5 * morph).attr("fill", r.pass ? PASS : FAIL).attr("stroke", "#fff").attr("stroke-width", 1.2).attr("opacity", routeOp);
  });
  // beat 5: other rollouts, districts, prototypes
  if (district > 0) {
    S.rolls.filter(r => !S.supIds.has(r.rid) && r.rid !== S.query.r.rid).forEach(r => { const [x, y] = S.latent.get(r.rid);
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 4).attr("fill", "#fff").attr("stroke", "#999").attr("stroke-width", 1.2).attr("opacity", district * 0.8); });
    [[S.hullP, PASS], [S.hullF, FAIL]].forEach(([h, col]) => { const c = d3.polygonCentroid(h);
      g.append("path").attr("d", d3.line().curve(d3.curveCatmullRomClosed.alpha(0.6))(h.map(pt => [lerp(c[0], pt[0], district), lerp(c[1], pt[1], district)]))).attr("fill", col).attr("fill-opacity", 0.10).attr("stroke", col).attr("stroke-opacity", 0.5).attr("stroke-dasharray", "4,3"); });
    [[S.protoP, PASS, "✓"], [S.protoF, FAIL, "✗"]].forEach(([pt, col, ch]) => {
      g.append("path").attr("d", d3.symbol(d3.symbolStar, 300)()).attr("transform", `translate(${pt[0]},${pt[1]}) scale(${district})`).attr("fill", col).attr("stroke", "#fff").attr("stroke-width", 1.5);
      glyph(g, pt[0] + 16, pt[1] - 12, ch, 13, district).attr("fill", col).attr("font-weight", 700); });
  }
  // beats 6-7: the new rollout
  if (travel > 0) {
    const Q = S.query, T = Q.r.n_steps, stop = Q.stop ?? T - 1;
    const tIdx = bi >= 6 ? stop : Math.min(stop, Math.floor(travel * (stop + 1.999)));
    const pathPts = S.trailL.slice(0, tIdx + 1), cur = pathPts[pathPts.length - 1];
    g.append("path").attr("d", d3.line().curve(d3.curveCatmullRom)(pathPts)).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 2).attr("stroke-opacity", 0.8);
    pathPts.slice(0, -1).forEach(pt => g.append("circle").attr("cx", pt[0]).attr("cy", pt[1]).attr("r", 2.2).attr("fill", INK).attr("opacity", 0.6));
    [[S.protoP, PASS], [S.protoF, FAIL]].forEach(([pt, col]) => g.append("line").attr("x1", cur[0]).attr("y1", cur[1]).attr("x2", pt[0]).attr("y2", pt[1]).attr("stroke", col).attr("stroke-width", 1.2).attr("stroke-dasharray", "3,3"));
    g.append("circle").attr("cx", cur[0]).attr("cy", cur[1]).attr("r", 11).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2.5);
    glyph(g, cur[0], cur[1], "🤖", 12);
    // meter
    const pNow = Q.curve[tIdx], mx = 24, my = 290, bar = d3.scaleLinear().domain([0, 1]).range([my + 84, my + 20]);
    g.append("rect").attr("x", mx).attr("y", my).attr("width", 96).attr("height", 96).attr("rx", 8).attr("fill", "#fff").attr("stroke", "#e3e3df");
    g.append("rect").attr("x", mx + 10).attr("y", my + 20).attr("width", 18).attr("height", 64).attr("fill", "#f1f1ee");
    g.append("rect").attr("x", mx + 10).attr("y", bar(pNow)).attr("width", 18).attr("height", my + 84 - bar(pNow)).attr("fill", pNow >= 0.5 ? PASS : FAIL);
    [S.thr, 1 - S.thr].forEach(v => g.append("line").attr("x1", mx + 8).attr("x2", mx + 30).attr("y1", bar(v)).attr("y2", bar(v)).attr("stroke", INK).attr("stroke-dasharray", "2,2"));
    g.append("text").attr("x", mx + 36).attr("y", my + 14).attr("font-size", 10).attr("fill", MUTED).text("P(pass)");
    g.append("text").attr("x", mx + 36).attr("y", my + 56).attr("font-size", 16).attr("font-weight", 700).attr("fill", INK).text(pNow.toFixed(2));
    g.append("text").attr("x", mx + 36).attr("y", bar(S.thr) + 4).attr("font-size", 9).attr("fill", MUTED).text("θ");
    // step lanes
    const tx = 150, ty = 412, tw = 630, cell = tw / T;
    const lane = (y, filled, ch, col) => { glyph(g, tx - 12, y + 5, ch, 11); d3.range(T).forEach(t => g.append("rect").attr("x", tx + t * cell + 1).attr("y", y).attr("width", cell - 2).attr("height", 9).attr("rx", 2).attr("fill", t < filled ? col : "#eceae5")); };
    const baseFill = bi >= 6 ? Math.round(stop + 1 + gate * (T - stop - 1)) : tIdx + 1;
    lane(ty - 13, tIdx + 1, "🤖", INK); lane(ty + 1, baseFill, "🐢", "#9a9a9a");
  }
  if (gate > 0) {
    const Q = S.query, T = Q.r.n_steps, stop = Q.stop ?? T - 1, cur = S.trailL[stop], predPass = Q.curve[stop] > S.thr, col = predPass ? PASS : FAIL;
    const gx = 150 + (stop + 1) * (630 / T);
    g.append("line").attr("x1", gx).attr("x2", gx).attr("y1", 412 - 20).attr("y2", 412 + 14).attr("stroke", INK).attr("stroke-width", 3 * gate);
    const s = 0.6 + 0.4 * gate, st = g.append("g").attr("transform", `translate(${W - 200},${30}) rotate(-4) scale(${s})`).attr("opacity", gate);
    st.append("rect").attr("width", 160).attr("height", 44).attr("rx", 8).attr("fill", "#fff").attr("stroke", col).attr("stroke-width", 3);
    st.append("text").attr("x", 80).attr("y", 20).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 800).attr("fill", col).text(`STOPPED · ${predPass ? "PASS ✓" : "FAIL ✗"}`);
    st.append("text").attr("x", 80).attr("y", 35).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).text(`reward ${predPass ? 1 : 0} → GRPO · step ${stop + 1} of ${T}`);
    g.append("line").attr("x1", cur[0]).attr("y1", cur[1]).attr("x2", W - 200 + 80 * s).attr("y2", 30 + 44 * s).attr("stroke", col).attr("stroke-opacity", 0.5 * gate);
    g.append("text").attr("x", 780).attr("y", 412 - 20).attr("text-anchor", "end").attr("font-size", 13).attr("font-weight", 700).attr("fill", INK).attr("opacity", gate).text(`${Math.round(100 * (1 - (stop + 1) / T))}% of this rollout never generated`);
  }
}

// ---------- beats 1-3 ----------
function drawIntro(g, S, bi, q, zoom, group) {
  const focus = S.trains[0], fi = S.icons.get(focus.rid), fo = S.obs.get(focus.rid), n = Math.min(INTRO_STEPS, focus.n_steps);
  const AX = (t) => 95 + t * 98, LANE = 232, ACT_Y = 160, OBS_Y = 304;
  // beat 1 progress: step k is "current" while q in [k/(n+1), (k+1)/(n+1)]
  const prog1 = bi === 0 ? q * (n + 1) : n + 1, cur = Math.min(n, Math.floor(prog1)), frac = clamp01(prog1 - Math.floor(prog1));
  const cardOp = 1 - zoom;
  // task card
  if (zoom < 1) {
    const tc = g.append("g").attr("opacity", cardOp);
    tc.append("rect").attr("x", 60).attr("y", 28).attr("width", 780).attr("height", 54).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df");
    glyph(tc, 88, 55, "📋", 22);
    tc.append("text").attr("x", 112).attr("y", 50).attr("font-size", 11).attr("fill", MUTED).text("TASK");
    tc.append("text").attr("x", 112).attr("y", 68).attr("font-size", 13).attr("fill", INK).text(S.gloss);
  }
  // the focus lane: big layout (beat 1) -> lane 0 of the grid (beat 2)
  const laneBig = LANE, lane0 = S.laneY(0);
  const lx0 = lerp(80, 70, zoom), lx1 = lerp(830, 580, zoom), ly = lerp(laneBig, lane0, zoom);
  g.append("line").attr("x1", lx0).attr("x2", lx1).attr("y1", ly).attr("y2", ly).attr("stroke", "#ddd").attr("stroke-width", lerp(3, 1, zoom));
  if (zoom < 1) for (let t = 0; t < n; t++) {
    const shown = t < cur, popping = t === cur && bi === 0, sc = shown ? 1 : popping ? ease(frac) : 0;
    if (sc === 0) continue;
    const x = lerp(AX(t), S.xs(t), zoom);
    // action bubble above, observation bubble below (big), collapsing to a single small icon on the lane
    const ay = lerp(ACT_Y, lane0, zoom), oy = lerp(OBS_Y, lane0, zoom), r = lerp(24, 7, zoom), fs = lerp(22, 11, zoom);
    const gA = g.append("g").attr("transform", `translate(${x},${ay}) scale(${sc})`);
    gA.append("circle").attr("r", r).attr("fill", "#e8eefc").attr("stroke", "#2b5fb3").attr("stroke-width", lerp(1.5, 0, zoom));
    glyph(gA, 0, 0, fi[t], fs);
    if (zoom < 1) { const gO = g.append("g").attr("transform", `translate(${x},${oy}) scale(${sc})`).attr("opacity", 1 - zoom);
      gO.append("circle").attr("r", r).attr("fill", "#f7ecd9").attr("stroke", "#8a5a00").attr("stroke-width", 1.5); glyph(gO, 0, 0, fo[t], fs);
      g.append("line").attr("x1", x).attr("x2", x).attr("y1", ACT_Y + 26).attr("y2", OBS_Y - 26).attr("stroke", "#ccc").attr("opacity", sc * (1 - zoom)); }
    if (zoom < 0.5 && (shown || popping)) g.append("text").attr("x", x).attr("y", ACT_Y - 34).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", MUTED).attr("opacity", sc * (1 - 2 * zoom)).text(ICON_LABEL[fi[t]] || "");
  }
  // agent sprite on the focus lane
  const sx = bi === 0 ? (cur >= n ? AX(n - 1) + 50 : lerp(cur === 0 ? 80 : AX(cur - 1), AX(cur), ease(frac))) : lerp(AX(n - 1) + 50, S.xs(Math.min(STEPS, focus.n_steps) - 1) + 14, zoom);
  const spr = g.append("g").attr("transform", `translate(${sx},${ly})`).attr("opacity", zoom < 1 ? 1 : 0);
  spr.append("circle").attr("r", lerp(15, 7, zoom)).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2); glyph(spr, 0, 0, "🤖", lerp(17, 8, zoom));
  // outcome: unknown until the end of beat 1
  if (bi === 0) { const done = cur >= n && frac > 0.4; glyph(g, 860, LANE, done ? (S.pass.get(focus.rid) ? "✅" : "❌") : "❓", 26, done ? 1 : 0.6); }
  // the other fifteen lanes (beat 2), with staggered agents running across
  if (zoom > 0) S.trains.forEach((r, i) => {
    const T = Math.min(STEPS, r.n_steps), y = S.laneY(i), ic = S.icons.get(r.rid);
    if (i === 0 && zoom < 1) return;
    const li = bi === 1 ? clamp01((q - 0.35 - i * 0.012) / 0.45) : 1;                 // lane progress in beat 2
    const shownSteps = i === 0 ? T : Math.floor(li * (T + 0.999));
    const op = (i === 0 ? 1 : zoom);
    g.append("line").attr("x1", 70).attr("x2", 580).attr("y1", y).attr("y2", y).attr("stroke", "#e4e4e0").attr("opacity", op * (1 - group));
    for (let t = 0; t < Math.min(T, shownSteps); t++) {
      let x = S.xs(t), yy = y;
      if (group > 0) { const st = S.stPos.get(`${t}:${r.steps[t].c}`); const k = S.trains.filter((rr, j) => j < i && t < rr.n_steps && rr.steps[t].c === r.steps[t].c).length;
        x = lerp(x, st.x + (k % 3 - 1) * 3, group); yy = lerp(y, st.y + (Math.floor(k / 3) - 1) * 3, group); }
      const gI = g.append("g").attr("transform", `translate(${x},${yy})`).attr("opacity", op * (1 - group));
      gI.append("circle").attr("r", 7).attr("fill", "#e8eefc"); glyph(gI, 0, 0, ic[t], 11);
    }
    if (i > 0 && bi === 1 && li < 1) { const ax = lerp(70, S.xs(Math.max(0, shownSteps - 1)), 1); const sp = g.append("g").attr("transform", `translate(${ax + 14},${y})`).attr("opacity", op);
      sp.append("circle").attr("r", 7).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 1.5); glyph(sp, 0, 0, "🤖", 8); }
    if (li >= 1 || i === 0) glyph(g, 606, y, S.pass.get(r.rid) ? "✅" : "❌", 12, op * (1 - group) * (bi === 1 && i > 0 ? clamp01((q - 0.8) / 0.2) : 1));
  });
}

// ---------- controls ----------
function setProg(p, fromUser) { prog = clamp01(p); $("#pipeline-scrub").value = Math.round(prog * 1000); render(prog); if (fromUser) stop(); }
function play() { if (playing) return; playing = true; $("#pipeline-play").textContent = "❚❚ pause"; let last = null;
  const tick = (ts) => { if (!playing) return; if (last != null) { prog += (ts - last) / DUR; if (prog >= 1) prog = 0; $("#pipeline-scrub").value = Math.round(prog * 1000); render(prog); } last = ts; raf = requestAnimationFrame(tick); };
  raf = requestAnimationFrame(tick); }
function stop() { playing = false; $("#pipeline-play").textContent = "▶ play"; if (raf) cancelAnimationFrame(raf); }
function initControls() {
  const beats = $("#pipeline-beats");
  BEATS.forEach((b, i) => { const el = document.createElement("button"); el.textContent = i + 1; el.title = b.title; el.onclick = () => setProg(b.t0 + 0.005, true); beats.appendChild(el); });
  $("#pipeline-scrub").oninput = (e) => setProg(+e.target.value / 1000, true);
  $("#pipeline-play").onclick = () => playing ? stop() : play();
}
initControls();
function load(b) { B = b; scene = buildScene(B); render(prog); }
document.addEventListener("precog:bundle", (e) => load(e.detail));
if (window.PRECOG_BUNDLE) load(window.PRECOG_BUNDLE);
const io = new IntersectionObserver((es) => { if (es.some(x => x.isIntersecting) && scene && !playing && prog === 0) play(); }, { threshold: 0.4 });
io.observe($("#pipeline-section"));
})();
