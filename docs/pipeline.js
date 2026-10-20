/* Opener: a near-wordless "video" of the pipeline, driven by a scrubber in [0,1].
   1 follow one agent through one attempt, then zoom out to the whole rollout · 2 the same task, sixteen attempts ·
   3 one step across attempts merges into states, then every step; rollouts become routes · 4 the encoder folds a route
   into a point · 5 prototypes from one labeled batch · 6 scoring a new rollout as it unfolds · 7 the gate.
   Geometry comes from the loaded task bundle (real states, latents, stop step); icons from the real tool calls (Lucide). */
(() => {
const PASS = "#2a9d4a", FAIL = "#d13b3b", NOISE = "#b9b9b9", NOISE_STROKE = "#7b3fb3", INK = "#1c1c1c", MUTED = "#8a8a86";
const ACT = "#2b5fb3", ACT_BG = "#e8eefc", OBS = "#8a5a00", OBS_BG = "#f7ecd9";
const srColor = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);
const W = 900, H = 440, STEPS = 12, K_SUP = 8, INTRO_STEPS = 8, LANES = 16, DUR = 125000;
const BEATS = [
  { t0: 0.00, t1: 0.14, title: "One task, one attempt", text: "The agent acts; the user or the system answers. The outcome is known only at the end." },
  { t0: 0.14, t1: 0.22, title: "Same task, sixteen attempts", text: "Different moves, different endings." },
  { t0: 0.22, t1: 0.38, title: "Similar steps merge into states", text: "One step across attempts first, then every step. Each rollout is a route through the states; red states lead to failure." },
  { t0: 0.38, t1: 0.66, title: "Meet Precog-RM", text: "Each action and observation is embedded and projected; a small transformer reads the whole sequence; mean pooling gives one point z. The point settles as steps arrive. The label comes last." },
  { t0: 0.66, t1: 0.78, title: "Two prototypes from labeled rollouts", text: "Sixteen labeled rollouts become sixteen points. The mean of the passes and the mean of the failures are the prototypes." },
  { t0: 0.78, t1: 1.00, title: "A new rollout, step by step", text: "Re-encoded after every step, its point drifts toward the prototype it resembles. Distance gives P(pass)." },
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
  const trailOf = (r) => E.z[idx.get(r.rid)].map(proj);
  const shown = [...finals.values()].concat(trailOf(trains[0]), trailOf(best.r));
  const fx = shown.map(p => p[0]), fy = shown.map(p => p[1]);
  const padx = (d3.max(fx) - d3.min(fx)) * 0.08, pady = (d3.max(fy) - d3.min(fy)) * 0.08;
  // latent panel: below the architecture strip
  const lx = d3.scaleLinear().domain([d3.min(fx) - padx, d3.max(fx) + padx]).range([540, 870]), ly = d3.scaleLinear().domain([d3.min(fy) - pady, d3.max(fy) + pady]).range([425, 245]);
  const L = (p) => [lx(p[0]), ly(p[1])];
  const latent = new Map([...finals].map(([rid, p]) => [rid, L(p)]));
  const trails = new Map(rolls.map(r => [r.rid, E.z[idx.get(r.rid)].map(z => L(proj(z)))]));
  const hull = (ids) => { const pts = ids.map(rid => latent.get(rid)); return d3.polygonHull(pts) || pts; };
  const laneY = (i) => 46 + i * (344 / (LANES - 1));
  const focus = trains[0];
  const focusSteps = focus.steps.slice(0, INTRO_STEPS).map(s => ({ ai: actionIcon(s.a), at: actionText(s.a), oi: obsIcon(s.o), ot: obsText(s.o) }));
  return { rolls, pass, supIds, sup, trains, stations, stPos, routes, latent, protoP: L(proj(cp)), protoF: L(proj(cf)),
           hullP: hull(sup.filter(r => pass.get(r.rid)).map(r => r.rid)), hullF: hull(sup.filter(r => !pass.get(r.rid)).map(r => r.rid)),
           query: best, trails, thr, sr, noise, xs, icons, obs, laneY, focus, focusSteps, focusCol: focusCol < 0 ? 3 : focusCol, score,
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
  const fadeOut = bi < 3 ? 0 : bi === 3 ? ease(sub(q, 0, 0.12)) : 1;                       // station map -> model
  if (bi <= 2) drawIntro(g, S, bi, q, zoom, group1, group);
  if (bi >= 2 && fadeOut < 1) {                                                              // stations
    const st = g.append("g").attr("opacity", 1 - fadeOut);
    S.stations.forEach(d => { const op = bi === 2 ? (d.t === S.focusCol ? group1 : group) : 1; if (op <= 0) return;
      const r = 9 + Math.sqrt(d.n) * 2.6, sg = st.append("g").attr("transform", `translate(${d.x},${d.y})`).attr("opacity", op);
      sg.append("circle").attr("r", r).attr("fill", "#fff").attr("stroke", S.noise.get(d.c) ? NOISE_STROKE : srColor(S.sr.get(d.c))).attr("stroke-width", 3.5).attr("stroke-dasharray", S.noise.get(d.c) ? "3,2" : null);
      pair(sg, 0, 0, d.icon, d.oicon, Math.max(7, r * 0.58)); });
    if (bi > 2 || group > 0) d3.range(STEPS).forEach(t => st.append("text").attr("x", S.xs(t)).attr("y", 405).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).attr("opacity", group).text(t + 1));
  }
  if (runOne > 0 && fadeOut < 1) S.routes.forEach((r, i) => {
    const frac = bi > 2 ? 1 : i === 0 ? ease(runOne) : ease(sub(runAll, (i - 1) * 0.025, 0.65 + (i - 1) * 0.025));
    if (frac <= 0) return;
    const path = g.append("path").attr("d", d3.line().curve(d3.curveMonotoneX)(r.pts)).attr("fill", "none").attr("stroke", r.pass ? PASS : FAIL).attr("stroke-width", bi === 2 && i === 0 && runAll < 1 ? 3 : 2).attr("stroke-opacity", 0.7 * (1 - fadeOut));
    if (frac < 1) { const L = path.node().getTotalLength(); path.attr("stroke-dasharray", `${L * frac} ${L}`);
      const h = path.node().getPointAtLength(L * frac); const sp = g.append("g").attr("transform", `translate(${h.x},${h.y})`);
      sp.append("circle").attr("r", 9).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2); icon(sp, "bot", 0, 0, 11, INK); return; }
    const head = r.pts[r.pts.length - 1];
    g.append("circle").attr("cx", head[0]).attr("cy", head[1]).attr("r", 4.5).attr("fill", r.pass ? PASS : FAIL).attr("stroke", "#fff").attr("stroke-width", 1.2).attr("opacity", 1 - fadeOut);
    icon(g, r.pass ? "circle-check" : "circle-x", head[0] + 12, head[1], 11, r.pass ? PASS : FAIL, 1 - fadeOut);
  });
  if (bi >= 3) drawModel(g, S, bi, q, fadeOut);
}

// ---------- beats 4-6: the model ----------
const ST = { qX: 28, qW: 120, embX: 165, embW: 150, prjX: 345, prjW: 110, encX: 485, encW: 215, poolX: 730, poolW: 68, zX: 846, y: 44, h: 160 };
const YC = ST.y + ST.h / 2;
const hueOf = (i) => d3.hsl((360 * i / LANES + 200) % 360, 0.6, 0.5).formatHex();
const hash = (a, b, c) => ((Math.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453) % 1 + 1) % 1;
// an embedding vector as a row of cells in the rollout's hue (lightness varies per cell)
function vec(g, x, y, n, cw, ch, hue, seed, op = 1, kind = null, shown = n) {
  const el = g.append("g").attr("opacity", op), base = d3.hsl(hue);
  d3.range(n).forEach(k => { const c = d3.hsl(base.h, base.s, 0.3 + 0.5 * hash(seed, k, 1)); el.append("rect").attr("x", x + k * (cw + 1)).attr("y", y - ch / 2).attr("width", cw).attr("height", ch).attr("rx", 1).attr("fill", k < shown ? c.formatHex() : "#eceae5"); });
  if (kind) el.append("rect").attr("x", x - 1).attr("y", y - ch / 2 - 1).attr("width", n * (cw + 1) + 1).attr("height", ch + 2).attr("rx", 2).attr("fill", "none").attr("stroke", kind === "a" ? ACT : OBS).attr("stroke-width", 1);
  return el;
}
// a vector drawn as a vertical column of cells. scheme: "a" (blues), "o" (oranges) or "ao" (top half blue, bottom half orange)
const SCHEME = { a: 215, o: 30 };
function vcol(g, x, yTop, n, cw, ch, scheme, seed, op = 1, outline = true, shown = n, gap = 0.6) {
  const el = g.append("g").attr("opacity", op);
  d3.range(n).forEach(k => { const part = scheme === "ao" ? (k < n / 2 ? "a" : "o") : scheme; const c = d3.hsl(SCHEME[part], 0.65, 0.32 + 0.45 * hash(seed, k, part === "a" ? 1 : 2));
    el.append("rect").attr("x", x - cw / 2).attr("y", yTop + k * (ch + gap)).attr("width", cw).attr("height", ch).attr("fill", k < shown ? c.formatHex() : "#eceae5"); });
  if (outline) { if (scheme === "ao") { el.append("rect").attr("x", x - cw / 2 - 1).attr("y", yTop - 1).attr("width", cw + 2).attr("height", (n / 2) * (ch + gap) + 0.5).attr("rx", 1.5).attr("fill", "none").attr("stroke", ACT).attr("stroke-width", 0.8);
      el.append("rect").attr("x", x - cw / 2 - 1).attr("y", yTop + (n / 2) * (ch + gap) - 0.5).attr("width", cw + 2).attr("height", (n / 2) * (ch + gap) + 1).attr("rx", 1.5).attr("fill", "none").attr("stroke", OBS).attr("stroke-width", 0.8); }
    else el.append("rect").attr("x", x - cw / 2 - 1).attr("y", yTop - 1).attr("width", cw + 2).attr("height", n * (ch + gap) + 1).attr("rx", 1.5).attr("fill", "none").attr("stroke", scheme === "a" ? ACT : OBS).attr("stroke-width", 0.8); }
  return el;
}
function stage(g, x, w, label, sub, op, pulse = 0) {
  const b = g.append("g").attr("opacity", op);
  b.append("rect").attr("x", x).attr("y", ST.y).attr("width", w).attr("height", ST.h).attr("rx", 12).attr("fill", "#fff").attr("stroke", pulse > 0 ? ACT : "#c9c9c3").attr("stroke-width", 1.5 + 2 * pulse);
  b.append("text").attr("x", x + w / 2).attr("y", ST.y + 20).attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 700).attr("fill", INK).text(label);
  if (sub) b.append("text").attr("x", x + w / 2).attr("y", ST.y + 34).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", MUTED).text(sub);
  return b;
}
const arrowTo = (g, x1, x2, y, op) => g.append("line").attr("x1", x1).attr("x2", x2).attr("y1", y).attr("y2", y).attr("stroke", "#b5b5b0").attr("stroke-width", 1.5).attr("opacity", op).attr("marker-end", "url(#arrowG)");
// camera: zoom the drawing onto a rectangle (viewBox units)
const CAM = { over: [0, 0, W, H], emb: [150, 30, 190, 200], prj: [330, 30, 150, 200], enc: [470, 20, 250, 210], pool: [620, 20, 280, 260] };
function camera(rect) { const [x, y, w, h] = rect, k = Math.min(W / w, H / h); return { k, tx: W / 2 - k * (x + w / 2), ty: H / 2 - k * (y + h / 2) }; }
const camLerp = (a, b, t) => ({ k: lerp(a.k, b.k, t), tx: lerp(a.tx, b.tx, t), ty: lerp(a.ty, b.ty, t) });

function drawModel(gRoot, S, bi, q, fadeIn) {
  const Bd = window.PRECOG_BUNDLE, idxOf = (rid) => Bd.rollouts.findIndex(r => r.rid === rid);
  const defs = gRoot.append("defs");
  defs.append("marker").attr("id", "arrowG").attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto").append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "#b5b5b0");
  // ---- shots of beat 4
  let shot = "flow", cam = camera(CAM.over), sq = 0;
  if (bi === 3) {
    const shots = [["over", 0, 0.12], ["emb", 0.12, 0.30], ["prj", 0.30, 0.42], ["enc", 0.42, 0.62], ["pool", 0.62, 0.70], ["flow", 0.70, 1]];
    const i = shots.findIndex(([, a, b]) => q >= a && q < b), cur = shots[Math.max(0, i)];
    shot = cur[0]; sq = sub(q, cur[1], cur[2]);
    const target = camera(CAM[shot === "flow" ? "over" : shot]), prev = camera(CAM[i > 0 ? (shots[i - 1][0] === "flow" ? "over" : shots[i - 1][0]) : "over"]);
    cam = camLerp(prev, target, ease(sub(sq, 0, 0.18)));
  }
  const g = gRoot.append("g").attr("transform", `translate(${cam.tx},${cam.ty}) scale(${cam.k})`).attr("opacity", fadeIn);
  const detail = (name) => shot === name ? ease(sub(sq, 0.18, 0.4)) : 0;          // detail layers fade in once the camera arrives
  const anim = ["emb", "prj", "enc", "pool"].includes(shot) ? sub(sq, 0.3, 0.96) : null;   // the step being processed inside a zoom-in
  const focus = S.focus, Q = S.query.r, hueIdx = new Map(S.trains.map((r, i) => [r.rid, i]));
  // ---- the rollout flowing (beat 4 flow shot, beat 6) and its progress
  let rid = null, steps = 0, prog = 0, mode = bi === 4 ? "many" : bi === 5 ? "query" : "one";
  if (mode === "one") { rid = focus.rid; steps = focus.n_steps; const slow = Math.min(10, steps);
    prog = shot !== "flow" ? 0 : sq < 0.78 ? sub(sq, 0.06, 0.78) * slow : slow + sub(sq, 0.78, 0.9) * (steps - slow); }
  else if (mode === "query") { rid = Q.rid; steps = Q.n_steps; prog = sub(q, 0.06, 0.92) * steps; }
  const hue = mode === "query" ? "#8a8a86" : hueOf(hueIdx.get(rid) ?? 0);
  const t = Math.min(Math.max(steps - 1, 0), Math.floor(prog)), f = prog >= steps ? 1 : prog - Math.floor(prog);
  const seen = steps ? Math.min(t + (f >= 0.35 || prog >= steps ? 1 : 0), steps) : 0;
  // ---- architecture (always drawn)
  g.append("text").attr("x", ST.qX).attr("y", ST.y - 16).attr("font-size", 13).attr("font-weight", 700).attr("fill", INK).text("Precog-RM");
  const encPulse = mode !== "many" && prog > 0 && prog < steps ? Math.sin(Math.PI * f) : 0;
  stage(g, ST.embX, ST.embW, "embed", "frozen text encoder · 3072-d", 1);
  stage(g, ST.prjX, ST.prjW, "project", "linear · 128-d", 1);
  stage(g, ST.encX, ST.encW, "encode", "transformer, 2 layers", 1, encPulse);
  stage(g, ST.poolX, ST.poolW, "pool", "mean", 1);
  [[ST.qX + ST.qW, ST.embX], [ST.embX + ST.embW, ST.prjX], [ST.prjX + ST.prjW, ST.encX], [ST.encX + ST.encW, ST.poolX], [ST.poolX + ST.poolW, ST.zX - 12]].forEach(([a, b]) => arrowTo(g, a + 3, b - 3, YC, 1));
  g.append("circle").attr("cx", ST.zX).attr("cy", YC).attr("r", 10).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2);
  g.append("text").attr("x", ST.zX).attr("y", YC + 4).attr("text-anchor", "middle").attr("font-size", 12).attr("font-style", "italic").attr("fill", INK).text("z");
  // latent panel frame
  g.append("rect").attr("x", 520).attr("y", 232).attr("width", 365).attr("height", 203).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df");
  g.append("text").attr("x", 530).attr("y", 246).attr("font-size", 10).attr("fill", MUTED).text("latent space");
  g.append("path").attr("d", `M${ST.zX},${YC + 12} C${ST.zX},210 ${ST.zX},220 ${ST.zX},232`).attr("fill", "none").attr("stroke", "#b5b5b0").attr("stroke-dasharray", "3,3");
  // ---- input queue: the rollout's tuples
  const ic = rid != null ? (S.icons.get(rid) || Q.steps.map(x => actionIcon(x.a))) : null, oc = rid != null ? (S.obs.get(rid) || Q.steps.map(x => obsIcon(x.o))) : null;
  const slot = (k) => ST.qX + 8 + k * 11;
  if (mode !== "many" && rid != null) for (let k = 0; k < Math.min(steps, 11); k++) if (k > t || (k === t && f > 0.25) || prog === 0) pair(g, slot(k), YC, ic[k], oc[k], 4, k > t || prog === 0 ? 0.95 : 0.25);
  // ---- inside the stages: vectors as they are actually processed
  const F = S.focusSteps, fs = F[Math.min(t, F.length - 1)];
  const seed = (rid ?? 0) * 7 + t * 3;
  const HALF = 16, TALL = { n: HALF, cw: 9, ch: 2.6, gap: 0.6 }, CAT = { n: 2 * HALF, cw: 9, ch: 2.6, gap: 0.6 }, SHORT = { n: 8, cw: 9, ch: 4.6, gap: 1 };
  const halfH = HALF * (TALL.ch + TALL.gap), catH = 2 * halfH, shortH = SHORT.n * (SHORT.ch + SHORT.gap);
  const ex = (t_, n) => { t_ = clean(t_); return t_.length > n ? t_.slice(0, n - 1) + "…" : t_; };
  // -- embed: texts in; a (blue) and o (orange) vectors out, then stacked into one concatenated vector
  const inEmb = g.append("g").attr("transform", `translate(${ST.embX},${ST.y})`);
  const embAct = shot === "emb" && anim != null ? { arrive: ease(sub(anim, 0, 0.18)), fa: sub(anim, 0.2, 0.45), fo: sub(anim, 0.35, 0.6), stack: ease(sub(anim, 0.68, 0.92)) } : null;
  const embOp = embAct ? 1 : (shot === "flow" || mode !== "one") ? (prog > 0 && f < 0.55 ? 1 : 0.35) : 1;
  inEmb.append("text").attr("x", 10).attr("y", 50).attr("font-size", 6).attr("fill", ACT).attr("opacity", embOp).text(ex(fs ? fs.at : "", 22));
  inEmb.append("text").attr("x", 10).attr("y", 60).attr("font-size", 6).attr("fill", OBS).attr("opacity", embOp).text(ex(fs ? fs.ot : "", 22));
  inEmb.append("text").attr("x", 10).attr("y", 74).attr("font-size", 6).attr("fill", MUTED).attr("opacity", embOp).text("text → vector");
  arrowTo(inEmb, 60, 86, 98, embOp * 0.8);
  const catTop = 36, aTop = catTop, oTopSide = catTop, oTopStack = catTop + halfH, xA = 104, xO = 126, xCat = 115;
  const stack = embAct ? embAct.stack : 1;                                       // 1 = already concatenated (overview and flow)
  const xa = lerp(xA, xCat, stack), xo = lerp(xO, xCat, stack), oTop = lerp(oTopSide, oTopStack, stack);
  vcol(inEmb, xa, aTop, HALF, TALL.cw, TALL.ch, "a", seed, embOp, true, embAct ? Math.round(HALF * embAct.fa) : HALF, TALL.gap);
  vcol(inEmb, xo, oTop, HALF, TALL.cw, TALL.ch, "o", seed + 1, embOp, true, embAct ? Math.round(HALF * embAct.fo) : HALF, TALL.gap);
  inEmb.append("text").attr("x", xa).attr("y", aTop - 5).attr("text-anchor", "middle").attr("font-size", 7).attr("fill", ACT).attr("opacity", 1 - stack).text("a");
  inEmb.append("text").attr("x", xo).attr("y", oTopSide - 5).attr("text-anchor", "middle").attr("font-size", 7).attr("fill", OBS).attr("opacity", 1 - stack).text("o");
  inEmb.append("text").attr("x", xCat).attr("y", catTop + catH + 10).attr("text-anchor", "middle").attr("font-size", 6.5).attr("fill", MUTED).text(stack > 0.5 ? "[a ; o] · 2 × 3072-d" : "3072-d each");
  if (embAct && embAct.arrive < 1) pair(g, lerp(slot(0), ST.embX + 24, embAct.arrive), YC, fs ? fs.ai : "message-square", fs ? fs.oi : "user", 4 + embAct.arrive, 1);
  // -- project: the concatenated vector squeezed through W into a short one
  const inPrj = g.append("g").attr("transform", `translate(${ST.prjX},${ST.y})`);
  const prjAct = shot === "prj" && anim != null ? { sq: ease(sub(anim, 0.1, 0.6)), out: sub(anim, 0.45, 0.9) } : null;
  const prjOp = prjAct ? 1 : (shot === "flow" || mode !== "one") ? (prog > 0 && f >= 0.4 && f < 0.75 ? 1 : 0.35) : 1;
  const sqz = prjAct ? prjAct.sq : 0, tallTop = 36, shortTop = 36 + (catH - shortH) / 2;
  vcol(inPrj, lerp(18, 44, sqz), lerp(tallTop, shortTop + 6, sqz), CAT.n, lerp(7, 4, sqz), lerp(CAT.ch, 1.1, sqz), "ao", seed, prjOp * (1 - 0.5 * sqz), true, CAT.n, lerp(CAT.gap, 0.3, sqz));
  inPrj.append("rect").attr("x", 44).attr("y", shortTop - 6).attr("width", 24).attr("height", shortH + 12).attr("rx", 3).attr("fill", "#fff").attr("stroke", prjAct ? ACT : "#c9c9c3").attr("stroke-width", prjAct ? 1 + Math.sin(Math.PI * sqz) : 1);
  inPrj.append("text").attr("x", 56).attr("y", shortTop + shortH / 2 + 3).attr("text-anchor", "middle").attr("font-size", 8).attr("fill", INK).text("W");
  arrowTo(inPrj, 70, 80, shortTop + shortH / 2, prjOp);
  vcol(inPrj, 92, shortTop, SHORT.n, SHORT.cw, SHORT.ch, "ao", seed, prjOp, true, prjAct ? Math.round(SHORT.n * prjAct.out) : SHORT.n, SHORT.gap);
  inPrj.append("text").attr("x", 92).attr("y", shortTop + shortH + 10).attr("text-anchor", "middle").attr("font-size", 6.5).attr("fill", MUTED).text("128-d");
  inPrj.append("text").attr("x", 18).attr("y", tallTop + catH + 10).attr("text-anchor", "middle").attr("font-size", 6.5).attr("fill", MUTED).attr("opacity", 1 - sqz).text("6144-d");
  // -- encode: one token column per step spread across the block, under two full-width layers of nodes
  const inEnc = g.append("g").attr("transform", `translate(${ST.encX},${ST.y})`);
  const encAct = shot === "enc" && anim != null ? { arr: sub(anim, 0, 0.55) * 6, lines: sub(anim, 0.3, 0.7), lp: sub(anim, 0.6, 1) } : null;
  const nTokReal = mode === "many" ? 0 : Math.min(seen, 12);
  const ghosts = nTokReal === 0 && (shot === "enc" || shot === "over" || shot === "pool") ? (encAct ? Math.min(6, Math.ceil(encAct.arr)) : 6) : 0;
  const nTok = nTokReal || ghosts, tokW = 7, tokCell = 2.6, tokTop = ST.h - 36, L1Y = 92, L2Y = 58;
  const spanL = 16, spanR = ST.encW - 44, tokX = (j) => nTok <= 1 ? (spanL + spanR) / 2 : spanL + j * (spanR - spanL) / (nTok - 1), nodeX = tokX;
  [[L1Y, "layer 1"], [L2Y, "layer 2"]].forEach(([y, lab]) => { inEnc.append("rect").attr("x", 8).attr("y", y - 8).attr("width", ST.encW - 16).attr("height", 16).attr("rx", 8).attr("fill", "#f1f1ee");
    inEnc.append("text").attr("x", ST.encW - 10).attr("y", y + 3).attr("text-anchor", "end").attr("font-size", 6.5).attr("fill", MUTED).text(lab); });
  const linesOp = encAct ? 0.12 * encAct.lines : (shot === "over" || shot === "pool" || nTokReal > 0 ? 0.12 : 0);
  if (linesOp > 0 && nTok > 0) for (let j = 0; j < nTok; j++) for (let k = 0; k < nTok; k++) {
    inEnc.append("line").attr("x1", tokX(j)).attr("y1", tokTop).attr("x2", nodeX(k)).attr("y2", L1Y + 4).attr("stroke", INK).attr("stroke-opacity", linesOp);
    inEnc.append("line").attr("x1", nodeX(j)).attr("y1", L1Y - 4).attr("x2", nodeX(k)).attr("y2", L2Y + 4).attr("stroke", INK).attr("stroke-opacity", linesOp); }
  const hot = encPulse > 0 && f > 0.55 ? encPulse : (encAct ? Math.sin(Math.PI * clamp01(encAct.arr - Math.floor(encAct.arr))) * (encAct.arr < 6 ? 1 : 0) : 0);
  const hotJ = encAct ? Math.min(nTok - 1, Math.floor(encAct.arr)) : nTok - 1;
  if (hot > 0 && hotJ >= 0) for (let k = 0; k < nTok; k++) { inEnc.append("line").attr("x1", tokX(hotJ)).attr("y1", tokTop).attr("x2", nodeX(k)).attr("y2", L1Y + 4).attr("stroke", ACT).attr("stroke-width", 1).attr("stroke-opacity", 0.7 * hot);
    inEnc.append("line").attr("x1", nodeX(k)).attr("y1", L1Y - 4).attr("x2", nodeX(k)).attr("y2", L2Y + 4).attr("stroke", ACT).attr("stroke-width", 0.8).attr("stroke-opacity", 0.4 * hot); }
  for (let j = 0; j < nTok; j++) {
    const slide = encAct ? clamp01(encAct.arr - j) : 1, x = lerp(tokX(j) - 30, tokX(j), ease(slide));
    vcol(inEnc, x, tokTop, SHORT.n, tokW, tokCell, "ao", (rid ?? 0) * 7 + j * 3, (ghosts ? 0.6 : 1) * (0.3 + 0.7 * slide), true, SHORT.n, 0.4);
    [L1Y, L2Y].forEach(y => inEnc.append("circle").attr("cx", nodeX(j)).attr("cy", y).attr("r", 4).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 0.9).attr("opacity", 0.4 + 0.6 * slide));
  }
  inEnc.append("text").attr("x", spanL - 6).attr("y", tokTop + 14).attr("text-anchor", "end").attr("font-size", 6).attr("fill", MUTED).text(ghosts ? "steps" : "");
  if (ghosts) d3.range(ghosts).forEach(j => inEnc.append("text").attr("x", tokX(j)).attr("y", ST.h - 6).attr("text-anchor", "middle").attr("font-size", 5.5).attr("fill", MUTED).text(`t${j + 1}`));
  if (nTokReal) inEnc.append("text").attr("x", (spanL + spanR) / 2).attr("y", ST.h - 6).attr("text-anchor", "middle").attr("font-size", 6.5).attr("fill", MUTED).text(`${seen} step${seen > 1 ? "s" : ""} so far`);
  if (encAct) [L1Y, L2Y].forEach((y, li) => { const on = Math.sin(Math.PI * clamp01(encAct.lp * 2 - li)); if (on > 0) inEnc.append("rect").attr("x", 8).attr("y", y - 8).attr("width", ST.encW - 16).attr("height", 16).attr("rx", 8).attr("fill", "none").attr("stroke", ACT).attr("stroke-width", 1.2).attr("opacity", on); });
  // -- pool: the layer-2 outputs average into one 128-d column, z
  const inPool = g.append("g").attr("transform", `translate(${ST.poolX},${ST.y})`);
  const poolAct = shot === "pool" && anim != null ? { merge: ease(sub(anim, 0, 0.45)), drop: ease(sub(anim, 0.55, 1)) } : null;
  const zTop = 36 + (catH - shortH) / 2;
  if (poolAct) for (let j = 0; j < nTok; j++) { const x0 = ST.encX + nodeX(j), x1 = ST.poolX + ST.poolW / 2, y0 = ST.y + L2Y, y1 = ST.y + zTop + shortH / 2;
    g.append("circle").attr("cx", lerp(x0, x1, poolAct.merge)).attr("cy", lerp(y0, y1, poolAct.merge)).attr("r", 3.5).attr("fill", "#fff").attr("stroke", INK).attr("opacity", 0.9 * (1 - poolAct.merge)); }
  vcol(inPool, ST.poolW / 2, zTop, SHORT.n, SHORT.cw, SHORT.ch, "ao", (rid ?? 0) * 7 + 99, poolAct ? poolAct.merge : (nTokReal > 0 || shot === "over" ? 1 : 0.35), false, SHORT.n, SHORT.gap);
  inPool.append("rect").attr("x", ST.poolW / 2 - SHORT.cw / 2 - 2).attr("y", zTop - 2).attr("width", SHORT.cw + 4).attr("height", shortH + 3).attr("rx", 2).attr("fill", "none").attr("stroke", hue).attr("stroke-width", 1.2);
  inPool.append("text").attr("x", ST.poolW / 2).attr("y", zTop + shortH + 10).attr("text-anchor", "middle").attr("font-size", 6.5).attr("fill", MUTED).text("mean · z");
  if (poolAct && poolAct.drop > 0) { const dst = S.trails.get(focus.rid)[0]; g.append("circle").attr("cx", lerp(ST.zX, dst[0], poolAct.drop)).attr("cy", lerp(YC + 12, dst[1], poolAct.drop)).attr("r", 7).attr("fill", hue).attr("stroke", "#fff").attr("stroke-width", 1.5); }
  if (poolAct) g.append("circle").attr("cx", ST.zX).attr("cy", YC).attr("r", 10 + 4 * Math.sin(Math.PI * sub(anim, 0.45, 0.6))).attr("fill", "none").attr("stroke", ACT).attr("stroke-width", 1.5).attr("opacity", Math.sin(Math.PI * sub(anim, 0.45, 0.6)));
  // ---- the current tuple travelling between stages (flow shot, beat 6)
  if (mode !== "many" && prog > 0 && prog < steps) {
    const f1 = sub(f, 0, 0.25), f3 = sub(f, 0.55, 0.78);
    if (f1 < 1) pair(g, lerp(slot(t), ST.embX + 24, ease(f1)), YC, ic[t], oc[t], 4 + f1, 1);
    if (f >= 0.55 && f3 < 1) { const e3 = ease(f3), x = lerp(ST.prjX + 92, ST.encX + tokX(Math.min(nTok - 1, t)), e3), yT = lerp(ST.y + shortTop, ST.y + tokTop, e3);
      vcol(g, x, yT, SHORT.n, lerp(SHORT.cw, tokW, e3), lerp(SHORT.ch, tokCell, e3), "ao", seed, 1, true, SHORT.n, lerp(SHORT.gap, 0.4, e3)); }
  }
  // ---- latent panel contents
  const L = g.append("g");
  if (mode === "one") {
    if (seen > 0) { const tr = S.trails.get(rid), tz = Math.max(0, seen - 1), from = tr[Math.max(0, tz - 1)], to = tr[Math.min(tr.length - 1, tz)];
      const mv = prog >= steps ? 1 : ease(sub(f, 0.35, 0.97)), cur = [lerp(from[0], to[0], seen > 1 ? mv : 1), lerp(from[1], to[1], seen > 1 ? mv : 1)];
      L.append("path").attr("d", d3.line().curve(d3.curveCatmullRom)(tr.slice(0, Math.max(1, seen)).concat([cur]))).attr("fill", "none").attr("stroke", hue).attr("stroke-width", 1.5).attr("stroke-opacity", 0.6);
      const labelled = shot === "flow" ? sub(sq, 0.92, 0.98) : 0, pass = S.pass.get(rid);
      L.append("circle").attr("cx", cur[0]).attr("cy", cur[1]).attr("r", 8).attr("fill", d3.interpolateRgb(hue, pass ? PASS : FAIL)(labelled)).attr("stroke", "#fff").attr("stroke-width", 2);
      if (labelled > 0) icon(L, pass ? "circle-check" : "circle-x", cur[0] + 16, cur[1] - 12, 16, pass ? PASS : FAIL, labelled);
      g.append("text").attr("x", ST.qX).attr("y", ST.y + ST.h + 16).attr("font-size", 9).attr("fill", MUTED).text(`rollout 1 · step ${seen} of ${steps}`); }
  } else if (mode === "many") {
    const others = S.trains.slice(1), per = 0.55 / others.length, label = ease(sub(q, 0.6, 0.72)), proto = ease(sub(q, 0.76, 1));
    const p0 = S.latent.get(focus.rid);
    L.append("circle").attr("cx", p0[0]).attr("cy", p0[1]).attr("r", 6.5).attr("fill", S.pass.get(focus.rid) ? PASS : FAIL).attr("stroke", "#fff").attr("stroke-width", 1.5);
    let flowing = null;
    others.forEach((r, i) => { const fI = sub(q, 0.02 + i * per, 0.02 + (i + 1) * per); if (fI <= 0) return;
      const pt = S.latent.get(r.rid), pass = S.pass.get(r.rid), h = hueOf(hueIdx.get(r.rid));
      if (fI < 1) { flowing = { r, fI, h }; L.append("circle").attr("cx", pt[0]).attr("cy", pt[1]).attr("r", 6.5 * ease(fI)).attr("fill", h).attr("stroke", "#fff").attr("stroke-width", 1.5); }
      else { const li = clamp01(label * 1.4 - i * 0.025); L.append("circle").attr("cx", pt[0]).attr("cy", pt[1]).attr("r", 6.5).attr("fill", d3.interpolateRgb(h, pass ? PASS : FAIL)(li)).attr("stroke", "#fff").attr("stroke-width", 1.5); } });
    if (flowing) { const { r, fI, h } = flowing, icr = S.icons.get(r.rid), ocr = S.obs.get(r.rid);
      for (let k = 0; k < Math.min(11, r.n_steps); k++) pair(g, slot(k), YC, icr[k], ocr[k], 4, 0.9 * (1 - fI));
      const x = lerp(ST.embX + 115, ST.encX + 20, ease(fI)); vcol(g, x, ST.y + 36 + (catH - shortH) / 2, SHORT.n, SHORT.cw, SHORT.ch, "ao", r.rid * 7, 1, true, SHORT.n, SHORT.gap);
      const nT = Math.min(12, r.n_steps), tx_ = (j) => 16 + j * (ST.encW - 60) / Math.max(1, nT - 1);
      for (let j = 0; j < nT; j++) vcol(inEnc, tx_(j), ST.h - 36, SHORT.n, 7, 2.6, "ao", r.rid * 7 + j * 3, fI, true, SHORT.n, 0.4); }
    g.append("text").attr("x", ST.qX).attr("y", ST.y + ST.h + 16).attr("font-size", 9).attr("fill", MUTED).text(`${Math.min(others.length, Math.floor(sub(q, 0.02, 0.57) * others.length)) + 1} of ${S.trains.length} labeled rollouts`);
    if (proto > 0) drawPrototypes(L, S, proto, false);
  } else {
    drawPrototypes(L, S, 1, true);
    if (seen > 0) { const tr = S.trails.get(rid), tz = Math.max(0, seen - 1), from = tr[Math.max(0, tz - 1)], to = tr[Math.min(tr.length - 1, tz)];
      const mv = prog >= steps ? 1 : ease(sub(f, 0.35, 0.97)), cur = [lerp(from[0], to[0], seen > 1 ? mv : 1), lerp(from[1], to[1], seen > 1 ? mv : 1)];
      L.append("path").attr("d", d3.line().curve(d3.curveCatmullRom)(tr.slice(0, Math.max(1, seen)).concat([cur]))).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 1.5).attr("stroke-opacity", 0.6);
      [[S.protoP, PASS], [S.protoF, FAIL]].forEach(([pt, col]) => L.append("line").attr("x1", cur[0]).attr("y1", cur[1]).attr("x2", pt[0]).attr("y2", pt[1]).attr("stroke", col).attr("stroke-width", 1.2).attr("stroke-dasharray", "3,3"));
      L.append("circle").attr("cx", cur[0]).attr("cy", cur[1]).attr("r", 10).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2.5); icon(L, "bot", cur[0], cur[1], 12, INK);
      const pNow = S.score(Bd.encoders.prefix.z[idxOf(rid)][tz]);
      const mx = 30, my = 240, bar = d3.scaleLinear().domain([0, 1]).range([my + 150, my + 30]);
      g.append("rect").attr("x", mx).attr("y", my).attr("width", 130).attr("height", 170).attr("rx", 8).attr("fill", "#fff").attr("stroke", "#e3e3df");
      g.append("text").attr("x", mx + 12).attr("y", my + 18).attr("font-size", 11).attr("fill", MUTED).text("P(pass)");
      g.append("rect").attr("x", mx + 12).attr("y", my + 30).attr("width", 22).attr("height", 120).attr("fill", "#f1f1ee");
      g.append("rect").attr("x", mx + 12).attr("y", bar(pNow)).attr("width", 22).attr("height", my + 150 - bar(pNow)).attr("fill", pNow >= 0.5 ? PASS : FAIL);
      g.append("line").attr("x1", mx + 10).attr("x2", mx + 36).attr("y1", bar(0.5)).attr("y2", bar(0.5)).attr("stroke", "#999").attr("stroke-dasharray", "2,2");
      g.append("text").attr("x", mx + 46).attr("y", my + 80).attr("font-size", 22).attr("font-weight", 700).attr("fill", INK).text(pNow.toFixed(2));
      g.append("text").attr("x", mx + 46).attr("y", my + 100).attr("font-size", 10).attr("fill", MUTED).text(`step ${seen} of ${steps}`);
      const reveal = sub(q, 0.95, 1), pass = S.pass.get(rid);
      if (reveal > 0) icon(L, pass ? "circle-check" : "circle-x", cur[0] + 18, cur[1] - 14, 18, pass ? PASS : FAIL, reveal);
      g.append("text").attr("x", ST.qX).attr("y", ST.y + ST.h + 16).attr("font-size", 9).attr("fill", MUTED).text(`a new rollout · step ${seen} of ${steps}`); }
  }
}
function drawPrototypes(g, S, k, withPoints) {
  if (withPoints) S.sup.forEach(r => { const pt = S.latent.get(r.rid); g.append("circle").attr("cx", pt[0]).attr("cy", pt[1]).attr("r", 5.5).attr("fill", S.pass.get(r.rid) ? PASS : FAIL).attr("fill-opacity", 0.8).attr("stroke", "#fff").attr("stroke-width", 1.2); });
  [[S.hullP, PASS, S.protoP, "circle-check"], [S.hullF, FAIL, S.protoF, "circle-x"]].forEach(([h, col, pt, ic]) => {
    const c = d3.polygonCentroid(h);
    g.append("path").attr("d", d3.line().curve(d3.curveCatmullRomClosed.alpha(0.6))(h.map(p => [lerp(c[0], p[0], k), lerp(c[1], p[1], k)]))).attr("fill", col).attr("fill-opacity", 0.09).attr("stroke", col).attr("stroke-opacity", 0.45).attr("stroke-dasharray", "4,3");
    if (!withPoints) S.sup.filter(r => (S.pass.get(r.rid) ? PASS : FAIL) === col).forEach(r => { const p = S.latent.get(r.rid);
      g.append("line").attr("x1", p[0]).attr("y1", p[1]).attr("x2", lerp(p[0], pt[0], k)).attr("y2", lerp(p[1], pt[1], k)).attr("stroke", col).attr("stroke-opacity", 0.35 * (1 - k * 0.6)); });
    g.append("path").attr("d", d3.symbol(d3.symbolStar, 260)()).attr("transform", `translate(${pt[0]},${pt[1]}) scale(${k})`).attr("fill", col).attr("stroke", "#fff").attr("stroke-width", 1.5);
    icon(g, ic, pt[0] + 16, pt[1] - 12, 14, col, k);
  });
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
