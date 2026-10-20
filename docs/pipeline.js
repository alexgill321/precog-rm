/* Opener: a near-wordless "video" of the pipeline, driven by a scrubber in [0,1].
   1 follow one agent through one attempt, then zoom out to the whole rollout · 2 the same task, sixteen attempts ·
   3 one step across attempts merges into states, then every step; rollouts become routes · 4 the encoder folds a route
   into a point · 5 prototypes from one labeled batch · 6 scoring a new rollout as it unfolds · 7 the gate.
   Geometry comes from the loaded task bundle (real states, latents, stop step); icons from the real tool calls (Lucide). */
(() => {
const PASS = "#2a9d4a", FAIL = "#d13b3b", NOISE = "#b9b9b9", NOISE_STROKE = "#7b3fb3", INK = "#1c1c1c", MUTED = "#8a8a86";
const ACT = "#2b5fb3", ACT_BG = "#e8eefc", OBS = "#8a5a00", OBS_BG = "#f7ecd9";
const srColor = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);
const W = 900, H = 440, STEPS = 12, K_SUP = 8, INTRO_STEPS = 8, LANES = 16, DUR = 90000;
const BEATS = [
  { t0: 0.00, t1: 0.21, title: "One task, one attempt", text: "The agent acts; the user or the system answers. The outcome is known only at the end." },
  { t0: 0.21, t1: 0.33, title: "Same task, sixteen attempts", text: "Different moves, different endings." },
  { t0: 0.33, t1: 0.54, title: "Similar steps merge into states", text: "One step across attempts first, then every step. Each rollout is a route through the states; red states lead to failure." },
  { t0: 0.54, t1: 0.63, title: "The encoder folds a route into a point", text: "Routes that end alike land near each other." },
  { t0: 0.63, t1: 0.71, title: "Two prototypes from one labeled batch", text: "No fitting: the mean of the passes and the mean of the failures." },
  { t0: 0.71, t1: 0.89, title: "Scoring a new rollout as it unfolds", text: "Its point moves after every step; distance to the prototypes gives P(pass)." },
  { t0: 0.89, t1: 1.00, title: "Confident enough: stop", text: "The predicted outcome becomes the reward. The rest is never generated." },
];
const TASK_GLOSS = { "retail/103": "Customer support: return two items from a delivered order, then sort out a second order." };
const TOOL_ICON = [[/find_user|get_user/, "id-card"], [/get_order/, "package"], [/get_product|list_all_product/, "tag"], [/return_/, "undo-2"], [/exchange_/, "repeat"], [/modify_.*address/, "map-pin"], [/modify_/, "pencil"], [/cancel_/, "ban"], [/transfer_to_human/, "hand-helping"], [/calculate/, "calculator"], [/certificate|refund/, "credit-card"]];
const ICON_LABEL = { "message-square": "message to the user", "id-card": "look up the user", "package": "order details", "tag": "product details", "undo-2": "return items", "repeat": "exchange items", "map-pin": "change address", "pencil": "modify order", "ban": "cancel order", "hand-helping": "hand off to a human", "calculator": "calculate", "credit-card": "payment", "wrench": "tool call", "user": "the user replies", "server": "the system replies", "triangle-alert": "the system reports an error" };
const toolCall = (a) => /^\s*([a-z_]+)\s*\(/.exec(a) || /"name"\s*:\s*"([a-z_]+)"/.exec(a);
function actionIcon(a) { const m = toolCall(a); if (!m) return "message-square"; for (const [re, ic] of TOOL_ICON) if (re.test(m[1])) return ic; return "wrench"; }
const obsIcon = (o) => /^\s*\[user\]/.test(o) ? "user" : /error|invalid|not found|failed/i.test(o.slice(0, 200)) ? "triangle-alert" : "server";
const clean = (t) => (t || "").replace(/\s+/g, " ").trim();
function actionText(a) { const m = toolCall(a); if (m) { const v = /"[a-z_]+"\s*:\s*"?([^",}\]]+)/.exec(a); return `${m[1]}(${v ? v[1].trim() : "…"})`; } return clean(a); }
const obsText = (o) => clean(o).replace(/^\[(user|tool)\]\s*/, "");

let B = null, scene = null, playing = false, prog = 0, raf = null;
const $ = (s) => document.querySelector(s);
const lerp = (a, b, t) => a + (b - a) * t, ease = d3.easeCubicInOut, clamp01 = (x) => Math.max(0, Math.min(1, x));
const dot = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; };
const dist = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; } return Math.sqrt(s); };
const mean = (vs) => { const m = new Array(vs[0].length).fill(0); vs.forEach(v => v.forEach((x, k) => m[k] += x / vs.length)); return m; };
// Lucide icon centred at (x,y), `size` px, stroke colour
function icon(g, name, x, y, size, color = INK, op = 1) {
  const k = size / 24, el = g.append("g").attr("transform", `translate(${x - size / 2},${y - size / 2}) scale(${k})`).attr("fill", "none").attr("stroke", color)
    .attr("stroke-width", 2).attr("stroke-linecap", "round").attr("stroke-linejoin", "round").attr("opacity", op);
  el.html(window.LUCIDE[name] || window.LUCIDE["wrench"]); return el;
}
// one step as an (action, observation) tuple: a pill, action on top (blue), observation below (tan)
function pair(g, x, y, ai, oi, size, op = 1, stroke = 0) {
  const w = size * 1.5, h = size * 2.9, el = g.append("g").attr("transform", `translate(${x - w / 2},${y - h / 2})`).attr("opacity", op);
  el.append("path").attr("d", `M${w / 2},0 h0 a${w / 2},${w / 2} 0 0 1 ${w / 2},${w / 2} v${h / 2 - w / 2} h${-w} v${-(h / 2 - w / 2)} a${w / 2},${w / 2} 0 0 1 ${w / 2},${-w / 2} z`).attr("fill", ACT_BG);
  el.append("path").attr("d", `M0,${h / 2} h${w} v${h / 2 - w / 2} a${w / 2},${w / 2} 0 0 1 ${-w / 2},${w / 2} h0 a${w / 2},${w / 2} 0 0 1 ${-w / 2},${-w / 2} z`).attr("fill", OBS_BG);
  if (stroke) el.append("rect").attr("width", w).attr("height", h).attr("rx", w / 2).attr("fill", "none").attr("stroke", "#c9c9c3").attr("stroke-width", stroke);
  icon(el, ai, w / 2, h / 4, size, ACT); icon(el, oi, w / 2, 3 * h / 4, size, OBS);
  return el;
}
function wrap(text, cpl) { const words = text.split(" "), lines = []; let cur = ""; words.forEach(w => { if ((cur + " " + w).trim().length > cpl && cur) { lines.push(cur); cur = w; } else cur = (cur + " " + w).trim(); }); if (cur) lines.push(cur); return lines; }

function buildScene(b) {
  const E = b.encoders.prefix, rolls = b.rollouts, idx = new Map(rolls.map((r, i) => [r.rid, i]));
  const pass = new Map(rolls.map(r => [r.rid, r.reward >= 0.5]));
  const zF = (rid) => { const z = E.z[idx.get(rid)]; return z[z.length - 1]; };
  const proj = (v) => { const p = E.pca; const c = v.map((x, k) => x - p.mean[k]); return [dot(c, p.components[0]), dot(c, p.components[1])]; };
  const held = new Set(b.cv_folds[0]), train = rolls.filter(r => !held.has(r.rid));
  const sup = [...train.filter(r => pass.get(r.rid)).slice(0, K_SUP), ...train.filter(r => !pass.get(r.rid)).slice(0, K_SUP)];
  const trains = []; for (let i = 0; i < K_SUP; i++) { trains.push(sup[i]); trains.push(sup[K_SUP + i]); }
  const supIds = new Set(sup.map(r => r.rid));
  const cp = mean(sup.filter(r => pass.get(r.rid)).map(r => zF(r.rid))), cf = mean(sup.filter(r => !pass.get(r.rid)).map(r => zF(r.rid)));
  const score = (z) => 1 / (1 + Math.exp(dist(z, cp) - dist(z, cf)));
  const thr = b.stop_threshold;
  let best = null;
  rolls.filter(r => held.has(r.rid)).forEach(r => {
    const curve = E.z[idx.get(r.rid)].map(score);
    let stop = null; for (let t = 0; t < curve.length; t++) if (curve[t] > thr || 1 - curve[t] > thr) { stop = t; break; }
    if (stop == null || stop < 3 || (curve[stop] > thr) !== pass.get(r.rid)) return;
    const frac = (stop + 1) / r.n_steps; if (!best || frac < best.frac) best = { r, curve, stop, frac };
  });
  if (!best) { const r = rolls.find(x => held.has(x.rid)); best = { r, curve: E.z[idx.get(r.rid)].map(score), stop: null, frac: 1 }; }
  const sr = new Map(b.clusters.map(c => [c.id, c.sr])), noise = new Map(b.clusters.map(c => [c.id, c.noise]));
  const icons = new Map(trains.map(r => [r.rid, r.steps.map(s => actionIcon(s.a))]));
  const obs = new Map(trains.map(r => [r.rid, r.steps.map(s => obsIcon(s.o))]));
  const xs = d3.scalePoint().domain(d3.range(STEPS)).range([70, 560]).padding(0.4);
  const stations = [], stPos = new Map(), colStates = [];
  for (let t = 0; t < STEPS; t++) {
    const present = new Map();
    trains.forEach(r => { if (t < r.n_steps) { const c = r.steps[t].c; if (!present.has(c)) present.set(c, []); present.get(c).push([icons.get(r.rid)[t], obs.get(r.rid)[t]]); } });
    const order = [...present.keys()].sort((a, c) => (sr.get(c) - sr.get(a)) || (a - c));
    colStates.push(order.length);
    const ys = d3.scalePoint().domain(order).range([70, 370]).padding(0.5);
    order.forEach(c => { const ic = present.get(c), mode = (k) => [...d3.rollup(ic, v => v.length, d => d[k])].sort((p, q) => q[1] - p[1])[0][0];
      const s = { t, c, x: xs(t), y: order.length === 1 ? 220 : ys(c), n: ic.length, icon: mode(0), oicon: mode(1) }; stations.push(s); stPos.set(`${t}:${c}`, s); });
  }
  // the column used to show merging first: the earliest step with 2-4 states
  const focusCol = colStates.findIndex((n, t) => t >= 2 && n >= 2 && n <= 4);
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
  const focus = trains[0];
  const focusSteps = focus.steps.slice(0, INTRO_STEPS).map(s => ({ ai: actionIcon(s.a), at: actionText(s.a), oi: obsIcon(s.o), ot: obsText(s.o) }));
  return { rolls, pass, supIds, sup, trains, stations, stPos, routes, latent, protoP: L(proj(cp)), protoF: L(proj(cf)),
           hullP: hull(sup.filter(r => pass.get(r.rid)).map(r => r.rid)), hullF: hull(sup.filter(r => !pass.get(r.rid)).map(r => r.rid)),
           query: best, trailL: trail.map(L), thr, sr, noise, xs, icons, obs, laneY, focus, focusSteps, focusCol: focusCol < 0 ? 3 : focusCol,
           gloss: TASK_GLOSS[b.task_id] || `${b.benchmark} · task ${b.task_id}` };
}

function beatAt(p) { for (let i = 0; i < BEATS.length; i++) if (p < BEATS[i].t1 || i === BEATS.length - 1) return [i, clamp01((p - BEATS[i].t0) / (BEATS[i].t1 - BEATS[i].t0))]; }
const sub = (q, a, b) => clamp01((q - a) / (b - a));

function render(p) {
  if (!scene) return;
  const svg = d3.select("#pipeline"); svg.selectAll("*").remove(); svg.attr("viewBox", `0 0 ${W} ${H}`);
  const S = scene, [bi, q] = beatAt(p), g = svg.append("g");
  $("#pipeline-title").textContent = BEATS[bi].title; $("#pipeline-text").textContent = BEATS[bi].text;
  document.querySelectorAll("#pipeline-beats button").forEach((el, i) => el.classList.toggle("on", i === bi));
  const zoom = bi < 1 ? 0 : bi === 1 ? ease(sub(q, 0, 0.35)) : 1;                       // whole rollout -> sixteen lanes
  const group1 = bi < 2 ? 0 : bi === 2 ? ease(sub(q, 0.05, 0.3)) : 1;                    // the focus column merges
  const group = bi < 2 ? 0 : bi === 2 ? ease(sub(q, 0.34, 0.54)) : 1;                     // every column merges
  const runOne = bi < 2 ? 0 : bi === 2 ? sub(q, 0.58, 0.76) : 1;                          // one rollout travels its route
  const runAll = bi < 2 ? 0 : bi === 2 ? sub(q, 0.8, 1) : 1;                              // the others follow
  const morph = bi < 3 ? 0 : bi === 3 ? ease(q) : 1;
  const district = bi < 4 ? 0 : bi === 4 ? ease(q) : 1;
  const travel = bi < 5 ? 0 : bi === 5 ? q : 1;
  const gate = bi < 6 ? 0 : ease(q);
  if (bi <= 2) drawIntro(g, S, bi, q, zoom, group1, group);
  if (bi >= 2 && morph < 1) {                                                              // stations
    const st = g.append("g").attr("opacity", 1 - morph);
    S.stations.forEach(d => { const op = bi === 2 ? (d.t === S.focusCol ? group1 : group) : 1; if (op <= 0) return;
      const r = 9 + Math.sqrt(d.n) * 2.6, sg = st.append("g").attr("transform", `translate(${d.x},${d.y})`).attr("opacity", op);
      sg.append("circle").attr("r", r).attr("fill", "#fff").attr("stroke", S.noise.get(d.c) ? NOISE_STROKE : srColor(S.sr.get(d.c))).attr("stroke-width", 3.5).attr("stroke-dasharray", S.noise.get(d.c) ? "3,2" : null);
      pair(sg, 0, 0, d.icon, d.oicon, Math.max(7, r * 0.58)); });
    if (bi > 2 || group > 0) d3.range(STEPS).forEach(t => st.append("text").attr("x", S.xs(t)).attr("y", 405).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).attr("opacity", group).text(t + 1));
  }
  if (runOne > 0) S.routes.forEach((r, i) => {
    // how far along its route this rollout has travelled: rollout 0 first, then the rest with a stagger
    const frac = bi > 2 ? 1 : i === 0 ? ease(runOne) : ease(sub(runAll, (i - 1) * 0.025, 0.65 + (i - 1) * 0.025));
    if (frac <= 0) return;
    const target = S.latent.get(r.rid), pts = r.pts.map(pt => [lerp(pt[0], target[0], morph), lerp(pt[1], target[1], morph)]);
    const path = g.append("path").attr("d", d3.line().curve(d3.curveMonotoneX)(pts)).attr("fill", "none").attr("stroke", r.pass ? PASS : FAIL).attr("stroke-width", bi === 2 && i === 0 && runAll < 1 ? 3 : 2).attr("stroke-opacity", 0.7 - 0.3 * morph);
    if (frac < 1) { const L = path.node().getTotalLength(); path.attr("stroke-dasharray", `${L * frac} ${L}`);
      const h = path.node().getPointAtLength(L * frac); const sp = g.append("g").attr("transform", `translate(${h.x},${h.y})`);
      sp.append("circle").attr("r", 9).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2); icon(sp, "bot", 0, 0, 11, INK); return; }
    const head = pts[pts.length - 1];
    g.append("circle").attr("cx", head[0]).attr("cy", head[1]).attr("r", 4.5 + 1.5 * morph).attr("fill", r.pass ? PASS : FAIL).attr("stroke", "#fff").attr("stroke-width", 1.2);
    icon(g, r.pass ? "circle-check" : "circle-x", head[0] + (bi === 2 ? 12 : 0), head[1], 11, r.pass ? PASS : FAIL, bi === 2 ? 1 : 0);
  });
  if (district > 0) {
    S.rolls.filter(r => !S.supIds.has(r.rid) && r.rid !== S.query.r.rid).forEach(r => { const [x, y] = S.latent.get(r.rid);
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 4).attr("fill", "#fff").attr("stroke", "#999").attr("stroke-width", 1.2).attr("opacity", district * 0.8); });
    [[S.hullP, PASS], [S.hullF, FAIL]].forEach(([h, col]) => { const c = d3.polygonCentroid(h);
      g.append("path").attr("d", d3.line().curve(d3.curveCatmullRomClosed.alpha(0.6))(h.map(pt => [lerp(c[0], pt[0], district), lerp(c[1], pt[1], district)]))).attr("fill", col).attr("fill-opacity", 0.10).attr("stroke", col).attr("stroke-opacity", 0.5).attr("stroke-dasharray", "4,3"); });
    [[S.protoP, PASS, "circle-check"], [S.protoF, FAIL, "circle-x"]].forEach(([pt, col, ic]) => {
      g.append("path").attr("d", d3.symbol(d3.symbolStar, 300)()).attr("transform", `translate(${pt[0]},${pt[1]}) scale(${district})`).attr("fill", col).attr("stroke", "#fff").attr("stroke-width", 1.5);
      icon(g, ic, pt[0] + 18, pt[1] - 14, 16, col, district); });
  }
  if (travel > 0) {
    const Q = S.query, T = Q.r.n_steps, stop = Q.stop ?? T - 1;
    const tIdx = bi >= 6 ? stop : Math.min(stop, Math.floor(travel * (stop + 1.999)));
    const pathPts = S.trailL.slice(0, tIdx + 1), cur = pathPts[pathPts.length - 1];
    g.append("path").attr("d", d3.line().curve(d3.curveCatmullRom)(pathPts)).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 2).attr("stroke-opacity", 0.8);
    pathPts.slice(0, -1).forEach(pt => g.append("circle").attr("cx", pt[0]).attr("cy", pt[1]).attr("r", 2.2).attr("fill", INK).attr("opacity", 0.6));
    [[S.protoP, PASS], [S.protoF, FAIL]].forEach(([pt, col]) => g.append("line").attr("x1", cur[0]).attr("y1", cur[1]).attr("x2", pt[0]).attr("y2", pt[1]).attr("stroke", col).attr("stroke-width", 1.2).attr("stroke-dasharray", "3,3"));
    g.append("circle").attr("cx", cur[0]).attr("cy", cur[1]).attr("r", 12).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2.5);
    icon(g, "bot", cur[0], cur[1], 14, INK);
    const pNow = Q.curve[tIdx], mx = 24, my = 290, bar = d3.scaleLinear().domain([0, 1]).range([my + 84, my + 20]);
    g.append("rect").attr("x", mx).attr("y", my).attr("width", 96).attr("height", 96).attr("rx", 8).attr("fill", "#fff").attr("stroke", "#e3e3df");
    g.append("rect").attr("x", mx + 10).attr("y", my + 20).attr("width", 18).attr("height", 64).attr("fill", "#f1f1ee");
    g.append("rect").attr("x", mx + 10).attr("y", bar(pNow)).attr("width", 18).attr("height", my + 84 - bar(pNow)).attr("fill", pNow >= 0.5 ? PASS : FAIL);
    [S.thr, 1 - S.thr].forEach(v => g.append("line").attr("x1", mx + 8).attr("x2", mx + 30).attr("y1", bar(v)).attr("y2", bar(v)).attr("stroke", INK).attr("stroke-dasharray", "2,2"));
    g.append("text").attr("x", mx + 36).attr("y", my + 14).attr("font-size", 10).attr("fill", MUTED).text("P(pass)");
    g.append("text").attr("x", mx + 36).attr("y", my + 56).attr("font-size", 16).attr("font-weight", 700).attr("fill", INK).text(pNow.toFixed(2));
    g.append("text").attr("x", mx + 36).attr("y", bar(S.thr) + 4).attr("font-size", 9).attr("fill", MUTED).text("θ");
    const tx = 150, ty = 412, tw = 630, cell = tw / T;
    const lane = (y, filled, ic, col) => { icon(g, ic, tx - 14, y + 4, 12, col); d3.range(T).forEach(t => g.append("rect").attr("x", tx + t * cell + 1).attr("y", y).attr("width", cell - 2).attr("height", 9).attr("rx", 2).attr("fill", t < filled ? col : "#eceae5")); };
    const baseFill = bi >= 6 ? Math.round(stop + 1 + gate * (T - stop - 1)) : tIdx + 1;
    lane(ty - 13, tIdx + 1, "bot", INK); lane(ty + 1, baseFill, "turtle", "#9a9a9a");
  }
  if (gate > 0) {
    const Q = S.query, T = Q.r.n_steps, stop = Q.stop ?? T - 1, cur = S.trailL[stop], predPass = Q.curve[stop] > S.thr, col = predPass ? PASS : FAIL;
    const gx = 150 + (stop + 1) * (630 / T);
    g.append("line").attr("x1", gx).attr("x2", gx).attr("y1", 412 - 20).attr("y2", 412 + 14).attr("stroke", INK).attr("stroke-width", 3 * gate);
    const s = 0.6 + 0.4 * gate, st = g.append("g").attr("transform", `translate(${W - 200},${30}) rotate(-4) scale(${s})`).attr("opacity", gate);
    st.append("rect").attr("width", 160).attr("height", 44).attr("rx", 8).attr("fill", "#fff").attr("stroke", col).attr("stroke-width", 3);
    st.append("text").attr("x", 90).attr("y", 20).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 800).attr("fill", col).text(`STOPPED · ${predPass ? "PASS" : "FAIL"}`);
    icon(st, predPass ? "circle-check" : "circle-x", 22, 16, 16, col);
    st.append("text").attr("x", 80).attr("y", 35).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).text(`reward ${predPass ? 1 : 0} → GRPO · step ${stop + 1} of ${T}`);
    g.append("line").attr("x1", cur[0]).attr("y1", cur[1]).attr("x2", W - 200 + 80 * s).attr("y2", 30 + 44 * s).attr("stroke", col).attr("stroke-opacity", 0.5 * gate);
    g.append("text").attr("x", 780).attr("y", 412 - 20).attr("text-anchor", "end").attr("font-size", 13).attr("font-weight", 700).attr("fill", INK).attr("opacity", gate).text(`${Math.round(100 * (1 - (stop + 1) / T))}% of this rollout never generated`);
  }
}

// ---------- beats 1-3 ----------
function card(g, x, y, w, h, bg, stroke, op) { return g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 10).attr("fill", bg).attr("stroke", stroke).attr("stroke-width", 1.5).attr("opacity", op); }
function drawIntro(g, S, bi, q, zoom, group1, group) {
  const F = S.focusSteps, n = F.length, focus = S.focus;
  const AX = (t) => 118 + t * 96, LANE = 232, ACT_Y = 158, OBS_Y = 306;      // the whole-rollout layout (end of beat 1)
  const follow = bi === 0 ? sub(q, 0.06, 0.78) : 1, out = bi === 0 ? ease(sub(q, 0.80, 0.97)) : 1;   // follow steps, then zoom out
  const k = Math.min(n, Math.floor(follow * n)), f = follow >= 1 ? 1 : follow * n - k;                 // current step, progress within it
  // task card (top), fades during the zoom to sixteen lanes
  if (zoom < 1) {
    const tc = g.append("g").attr("opacity", (bi === 0 ? ease(sub(q, 0, 0.06)) : 1) * (1 - zoom));
    card(tc, 60, 24, 780, 52, "#fff", "#e3e3df", 1); icon(tc, "clipboard-list", 88, 50, 24, INK);
    tc.append("text").attr("x", 110).attr("y", 44).attr("font-size", 10).attr("fill", MUTED).attr("letter-spacing", 1.2).text("TASK");
    tc.append("text").attr("x", 110).attr("y", 62).attr("font-size", 13).attr("fill", INK).text(S.gloss);
  }
  // ---- close-up: follow the agent one step at a time (beat 1, before the zoom-out)
  if (bi === 0 && out < 1 && follow > 0) {
    const cu = g.append("g").attr("opacity", 1 - out);
    const step = Math.min(k, n - 1), st = F[step], fin = k >= n;
    // agent at left
    cu.append("circle").attr("cx", 120).attr("cy", 232).attr("r", 30).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2.5);
    icon(cu, "bot", 120, 232, 34, INK); cu.append("text").attr("x", 120).attr("y", 280).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", MUTED).text(`agent · step ${Math.min(step + 1, n)} of ${focus.n_steps}`);
    const slideA = fin ? 1 : ease(sub(f, 0.0, 0.3)), slideO = fin ? 1 : ease(sub(f, 0.42, 0.72));
    // one card per step: the (action, observation) tuple
    const cx = lerp(240, 200, slideA), cd = cu.append("g").attr("opacity", slideA).attr("transform", `translate(${cx},84)`);
    card(cd, 0, 0, 620, 236, "#fff", "#c9c9c3", 1);
    cd.append("text").attr("x", 18).attr("y", 24).attr("font-size", 11).attr("fill", MUTED).attr("letter-spacing", 1.2).text(`STEP ${step + 1}`);
    cd.append("text").attr("x", 300).attr("y", 24).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", MUTED).text("( action , observation )");
    const row = (y, bg, col, label, ic, name, text, op, mono) => { const rw = cd.append("g").attr("opacity", op).attr("transform", `translate(14,${y})`);
      card(rw, 0, 0, 592, 92, bg, col, 1); rw.append("text").attr("x", 14).attr("y", 20).attr("font-size", 11).attr("fill", col).attr("font-weight", 700).attr("letter-spacing", 1.2).text(label);
      rw.append("text").attr("x", 14 + label.length * 8.2).attr("y", 20).attr("font-size", 11).attr("fill", col).text(`· ${name}`); icon(rw, ic, 36, 58, 34, col);
      wrap(text, 80).slice(0, 2).forEach((l, i) => rw.append("text").attr("x", 66).attr("y", 50 + i * 17).attr("font-size", 12.5).attr("fill", INK).attr("font-family", mono ? "ui-monospace, Menlo, monospace" : null).text(l)); };
    row(36, ACT_BG, ACT, "ACTION", st.ai, ICON_LABEL[st.ai], st.at, 1, /\(/.test(st.at) && st.ai !== "message-square");
    row(136, OBS_BG, OBS, "OBSERVATION", st.oi, ICON_LABEL[st.oi], st.ot, slideO, false);
    cu.append("path").attr("d", `M152,222 C180,200 180,190 ${cx - 4},185`).attr("fill", "none").attr("stroke", ACT).attr("stroke-width", 2).attr("opacity", slideA).attr("marker-end", "url(#arrowA)");
    cu.append("path").attr("d", `M${cx - 4},265 C180,270 180,255 152,244`).attr("fill", "none").attr("stroke", OBS).attr("stroke-width", 2).attr("opacity", slideO).attr("marker-end", "url(#arrowO)");
    // filmstrip of completed steps at the bottom
    const sx = (t) => 118 + t * 96;
    cu.append("line").attr("x1", 90).attr("x2", 118 + (n - 1) * 96 + 40).attr("y1", 400).attr("y2", 400).attr("stroke", "#ddd");
    for (let t = 0; t < Math.min(k + (f > 0.85 ? 1 : 0), n); t++) pair(cu, sx(t), 400, F[t].ai, F[t].oi, 11);
    const defs = g.append("defs");
    [["arrowA", ACT], ["arrowO", OBS]].forEach(([id, col]) => defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto").append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", col));
  }
  // ---- whole rollout (end of beat 1), then zoom into lane 0 of the grid (beat 2)
  if ((bi === 0 && out > 0) || (bi >= 1 && zoom < 1)) {
    const wr = g.append("g").attr("opacity", bi === 0 ? out : 1);
    const lane0 = S.laneY(0), ly = lerp(LANE, lane0, zoom);
    wr.append("line").attr("x1", lerp(90, 70, zoom)).attr("x2", lerp(118 + (n - 1) * 96 + 60, 580, zoom)).attr("y1", ly).attr("y2", ly).attr("stroke", "#ddd").attr("stroke-width", lerp(3, 1, zoom));
    if (zoom < 0.5) { wr.append("text").attr("x", 20).attr("y", LANE - 30).attr("font-size", 11).attr("fill", ACT).attr("opacity", 1 - 2 * zoom).text("action");
      wr.append("text").attr("x", 20).attr("y", LANE + 36).attr("font-size", 11).attr("fill", OBS).attr("opacity", 1 - 2 * zoom).text("observation"); }
    for (let t = 0; t < n; t++) pair(wr, lerp(AX(t), S.xs(t), zoom), ly, F[t].ai, F[t].oi, lerp(24, 7, zoom), 1, lerp(1, 0, zoom));
    // steps beyond the close-up appear as the rollout zooms out
    if (bi === 0 && focus.n_steps > n) wr.append("text").attr("x", 118 + (n - 1) * 96 + 44).attr("y", LANE + 4).attr("font-size", 12).attr("fill", MUTED).text(`… ${focus.n_steps - n} more steps`);
    const sprX = lerp(118 + (n - 1) * 96 + 60, S.xs(Math.min(STEPS, focus.n_steps) - 1) + 14, zoom);
    const spr = wr.append("g").attr("transform", `translate(${sprX},${ly})`).attr("opacity", 1 - zoom);
    spr.append("circle").attr("r", 15).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2); icon(spr, "bot", 0, 0, 17, INK);
    const outcome = bi === 0 ? sub(q, 0.97, 1) : 1;
    icon(wr, outcome > 0.5 ? (S.pass.get(focus.rid) ? "circle-check" : "circle-x") : "circle-help", lerp(856, 606, zoom), ly, lerp(30, 12, zoom), outcome > 0.5 ? (S.pass.get(focus.rid) ? PASS : FAIL) : MUTED, 1);
  }
  // ---- the sixteen lanes (beat 2) and the merge (beat 3)
  if (zoom > 0) S.trains.forEach((r, i) => {
    const T = Math.min(STEPS, r.n_steps), y = S.laneY(i), ic = S.icons.get(r.rid);
    if (i === 0 && zoom < 1) return;
    const li = bi === 1 ? sub(q, 0.38 + i * 0.012, 0.86 + i * 0.006) : 1, shownSteps = i === 0 ? T : Math.floor(li * (T + 0.999));
    const op = i === 0 ? 1 : zoom;
    g.append("line").attr("x1", 70).attr("x2", 580).attr("y1", y).attr("y2", y).attr("stroke", "#e4e4e0").attr("opacity", op * (1 - group));
    for (let t = 0; t < Math.min(T, shownSteps); t++) {
      const st = S.stPos.get(`${t}:${r.steps[t].c}`), kk = S.trains.filter((rr, j) => j < i && t < rr.n_steps && rr.steps[t].c === r.steps[t].c).length;
      const gg = t === S.focusCol ? group1 : group, dim = bi === 2 && t !== S.focusCol && group1 > 0 && group < 1 ? 1 - 0.7 * group1 * (1 - group) : 1;
      const x = lerp(S.xs(t), st.x + (kk % 3 - 1) * 3, gg), yy = lerp(y, st.y + (Math.floor(kk / 3) - 1) * 3, gg);
      const big = t === S.focusCol && bi === 2 ? 1 + 0.9 * Math.sin(Math.PI * Math.min(group1, 1 - group)) : 1;    // the focus column swells while it merges
      const gI = g.append("g").attr("transform", `translate(${x},${yy}) scale(${big})`).attr("opacity", op * (1 - gg) * dim);
      pair(gI, 0, 0, ic[t], S.obs.get(r.rid)[t], 7);
    }
    if (i > 0 && bi === 1 && li < 1 && zoom >= 1) { const ax = S.xs(Math.max(0, shownSteps - 1)); const sp = g.append("g").attr("transform", `translate(${ax + 14},${y})`).attr("opacity", op);
      sp.append("circle").attr("r", 7).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 1.5); icon(sp, "bot", 0, 0, 9, INK); }
    if (li >= 1 || i === 0) icon(g, S.pass.get(r.rid) ? "circle-check" : "circle-x", 606, y, 12, S.pass.get(r.rid) ? PASS : FAIL, op * (1 - group) * (bi === 1 && i > 0 ? sub(q, 0.9, 1) : 1));
  });
  // beat 3: a soft column highlight behind the step being merged first
  if (bi === 2 && group1 > 0 && group < 1) g.append("rect").attr("x", S.xs(S.focusCol) - 22).attr("y", 30).attr("width", 44).attr("height", 380).attr("rx", 12).attr("fill", ACT).attr("fill-opacity", 0.06 * (1 - group)).lower();
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
