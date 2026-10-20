/* Opener: a near-wordless "video" of the pipeline, driven by a scrubber in [0,1].
   1 follow one agent through one attempt, then zoom out to the whole rollout · 2 the same task, sixteen attempts ·
   3 one step across attempts merges into states, then every step; rollouts become routes · 4 the encoder folds a route
   into a point · 5 prototypes from one labeled batch · 6 scoring a new rollout as it unfolds · 7 the gate.
   Geometry comes from the loaded task bundle (real states, latents, stop step); icons from the real tool calls (Lucide). */
(() => {
const PASS = "#2a9d4a", FAIL = "#d13b3b", NOISE = "#b9b9b9", NOISE_STROKE = "#7b3fb3", INK = "#1c1c1c", MUTED = "#8a8a86";
const ACT = "#2b5fb3", ACT_BG = "#e8eefc", OBS = "#8a5a00", OBS_BG = "#f7ecd9";
const srColor = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);
const W = 900, H = 440, STEPS = 12, K_SUP = 8, INTRO_STEPS = 8, LANES = 16, DUR = 110000;
const BEATS = [
  { t0: 0.00, t1: 0.16, title: "One task, one attempt", text: "The agent acts; the user or the system answers. The outcome is known only at the end." },
  { t0: 0.16, t1: 0.26, title: "Same task, sixteen attempts", text: "Different moves, different endings." },
  { t0: 0.26, t1: 0.44, title: "Similar steps merge into states", text: "One step across attempts first, then every step. Each rollout is a route through the states; red states lead to failure." },
  { t0: 0.44, t1: 0.66, title: "Meet Precog-RM", text: "Each (action, observation) is embedded; the sequence runs through a small transformer; mean pooling gives one point z. The point settles as steps arrive. The label comes last." },
  { t0: 0.66, t1: 0.80, title: "Two prototypes from labeled rollouts", text: "Sixteen labeled rollouts become sixteen points. The mean of the passes and the mean of the failures are the prototypes." },
  { t0: 0.80, t1: 1.00, title: "A new rollout, step by step", text: "Re-encoded after every step, its point drifts toward the prototype it resembles. Distance gives P(pass)." },
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
  const fx = [...finals.values()].map(p => p[0]), fy = [...finals.values()].map(p => p[1]);
  const padx = (d3.max(fx) - d3.min(fx)) * 0.1, pady = (d3.max(fy) - d3.min(fy)) * 0.1;
  // latent panel: below the architecture strip
  const lx = d3.scaleLinear().domain([d3.min(fx) - padx, d3.max(fx) + padx]).range([200, 860]), ly = d3.scaleLinear().domain([d3.min(fy) - pady, d3.max(fy) + pady]).range([425, 215]);
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
const STAGE = { inX: 40, inW: 180, embX: 240, embW: 112, prjX: 410, prjW: 92, encX: 540, encW: 170, poolX: 740, poolW: 64, zX: 848, y: 68, h: 78 };
const cellColor = (rid, t, k) => d3.interpolateBlues(0.25 + 0.6 * ((Math.sin(rid * 12.9898 + t * 78.233 + k * 37.719) * 43758.5453) % 1 + 1) % 1);
function strip(g, x, y, rid, t, n, cw, ch, op = 1) { const el = g.append("g").attr("opacity", op); d3.range(n).forEach(k => el.append("rect").attr("x", x + k * (cw + 1)).attr("y", y - ch / 2).attr("width", cw).attr("height", ch).attr("rx", 1).attr("fill", cellColor(rid, t, k))); return el; }
function box(g, x, w, label, sub, op, pulse = 0) {
  const b = g.append("g").attr("opacity", op);
  b.append("rect").attr("x", x).attr("y", STAGE.y).attr("width", w).attr("height", STAGE.h).attr("rx", 10).attr("fill", "#fff").attr("stroke", pulse > 0 ? ACT : "#c9c9c3").attr("stroke-width", 1.5 + 2 * pulse);
  b.append("text").attr("x", x + w / 2).attr("y", STAGE.y + 22).attr("text-anchor", "middle").attr("font-size", 11.5).attr("font-weight", 600).attr("fill", INK).text(label);
  if (sub) b.append("text").attr("x", x + w / 2).attr("y", STAGE.y + 37).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", MUTED).text(sub);
  return b;
}
const arrow = (g, x1, x2, y, op) => g.append("line").attr("x1", x1).attr("x2", x2).attr("y1", y).attr("y2", y).attr("stroke", "#b5b5b0").attr("stroke-width", 1.5).attr("opacity", op).attr("marker-end", "url(#arrowG)");
function drawModel(g, S, bi, q, fadeIn) {
  const defs = g.append("defs");
  defs.append("marker").attr("id", "arrowG").attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto").append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "#b5b5b0");
  const op = fadeIn, yC = STAGE.y + STAGE.h / 2 + 10;
  // which rollout is flowing, and how far
  let rid, steps, prog, mode;                                   // mode: "one" (beat 4), "many" (beat 5), "query" (beat 6)
  const focus = S.focus, Q = S.query.r;
  if (bi === 3) { rid = focus.rid; steps = focus.n_steps; const slow = Math.min(10, steps);     // ten steps slowly, the rest fast-forwarded
    prog = q < 0.74 ? sub(q, 0.2, 0.74) * slow : slow + sub(q, 0.74, 0.86) * (steps - slow); mode = "one"; }
  else if (bi === 4) { mode = "many"; }
  else { rid = Q.rid; steps = Q.n_steps; prog = sub(q, 0.08, 0.92) * steps; mode = "query"; }
  // ---- architecture strip
  g.append("text").attr("x", STAGE.inX).attr("y", STAGE.y - 14).attr("font-size", 12).attr("font-weight", 700).attr("fill", INK).attr("opacity", op).text("Precog-RM");
  box(g, STAGE.embX, STAGE.embW, "text embedder", "frozen", op);
  box(g, STAGE.prjX, STAGE.prjW, "projection", "linear", op);
  const encPulse = mode !== "many" && prog > 0 && prog < steps ? Math.sin(Math.PI * (prog % 1)) : 0;
  box(g, STAGE.encX, STAGE.encW, "transformer encoder", "2 layers, the whole sequence", op, encPulse);
  box(g, STAGE.poolX, STAGE.poolW, "mean pool", null, op);
  [[STAGE.inX + STAGE.inW, STAGE.embX], [STAGE.embX + STAGE.embW, STAGE.prjX], [STAGE.prjX + STAGE.prjW, STAGE.encX], [STAGE.encX + STAGE.encW, STAGE.poolX], [STAGE.poolX + STAGE.poolW, STAGE.zX - 14]].forEach(([a, b]) => arrow(g, a + 2, b - 2, yC - 10, op));
  g.append("circle").attr("cx", STAGE.zX).attr("cy", yC - 10).attr("r", 9).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2).attr("opacity", op);
  g.append("text").attr("x", STAGE.zX).attr("y", yC - 6).attr("text-anchor", "middle").attr("font-size", 11).attr("font-style", "italic").attr("fill", INK).attr("opacity", op).text("z");
  g.append("text").attr("x", 200).attr("y", 440 - 6).attr("font-size", 10).attr("fill", MUTED).attr("opacity", op).text("latent space");
  // ---- the rollout flowing through (beats 4 and 6)
  if (mode !== "many") {
    const ic = S.icons.get(rid) || Q.steps.map(s => actionIcon(s.a)), oc = S.obs.get(rid) || Q.steps.map(s => obsIcon(s.o));
    const t = Math.min(steps - 1, Math.floor(prog)), f = prog >= steps ? 1 : prog - Math.floor(prog);
    const slot = (k) => STAGE.inX + 8 + k * 15;
    for (let k = 0; k < Math.min(steps, 12); k++) if (k > t || (k === t && f > 0.3)) pair(g, slot(k), yC - 10, ic[k], oc[k], 5, op * (k > t ? 0.9 : 0.25));
    // the current tuple travels: pill -> embedding strip -> projected strip -> into the encoder
    if (prog > 0 && prog < steps) {
      const f1 = sub(f, 0, 0.3), f2 = sub(f, 0.3, 0.55), f3 = sub(f, 0.55, 0.8);
      if (f1 < 1) pair(g, lerp(slot(t), STAGE.embX + 20, ease(f1)), yC - 10, ic[t], oc[t], 5 + 2 * f1, op);
      else if (f2 < 1) strip(g, lerp(STAGE.embX + STAGE.embW - 50, STAGE.prjX + 10, ease(f2)), yC - 10, rid, t, 10, 4, 14, op);
      else if (f3 < 1) strip(g, lerp(STAGE.prjX + STAGE.prjW - 36, STAGE.encX + 12, ease(f3)), yC - 10, rid, t, 6, 4, 14, op);
    }
    // sequence so far inside the encoder
    const seen = Math.min(t + (f >= 0.8 || prog >= steps ? 1 : 0), steps);
    d3.range(seen).forEach(k => strip(g, STAGE.encX + 10 + k * 15, STAGE.y + STAGE.h - 12, rid, k, 3, 3, 8, op));
    // the point in latent space: settles through the prefix latents; grey until labelled
    const tr = S.trails.get(rid), tz = Math.max(0, seen - 1), from = tr[Math.max(0, tz - 1)], to = tr[Math.min(tr.length - 1, tz)];
    const moveF = seen === 0 ? 0 : prog >= steps ? 1 : ease(sub(f, 0.8, 1));
    const cur = seen === 0 ? null : [lerp(from[0], to[0], seen > 1 ? moveF : 1), lerp(from[1], to[1], seen > 1 ? moveF : 1)];
    if (cur) {
      g.append("path").attr("d", d3.line().curve(d3.curveCatmullRom)(tr.slice(0, Math.max(1, seen)).concat([cur]))).attr("fill", "none").attr("stroke", "#777").attr("stroke-width", 1.5).attr("stroke-opacity", 0.6 * op);
      g.append("path").attr("d", `M${STAGE.zX},${yC} C${STAGE.zX},${yC + 60} ${cur[0]},${cur[1] - 60} ${cur[0]},${cur[1] - 12}`).attr("fill", "none").attr("stroke", "#b5b5b0").attr("stroke-dasharray", "3,3").attr("opacity", op * (seen < steps ? 1 : 0.4));
    }
    if (mode === "one") {
      // beat 4: the label arrives at the end and the point takes its colour
      const labelled = sub(q, 0.88, 0.95), pass = S.pass.get(rid);
      if (cur) { g.append("circle").attr("cx", cur[0]).attr("cy", cur[1]).attr("r", 9).attr("fill", labelled > 0 ? d3.interpolateRgb("#9a9a9a", pass ? PASS : FAIL)(labelled) : "#9a9a9a").attr("stroke", "#fff").attr("stroke-width", 2).attr("opacity", op);
        if (labelled > 0) icon(g, pass ? "circle-check" : "circle-x", cur[0] + 18, cur[1] - 14, 18, pass ? PASS : FAIL, labelled); }
      g.append("text").attr("x", STAGE.inX).attr("y", yC + 32).attr("font-size", 10).attr("fill", MUTED).attr("opacity", op).text(`rollout 1 · step ${Math.min(seen, steps)} of ${focus.n_steps}`);
    } else {
      // beat 6: the new rollout, with the prototypes in place
      drawPrototypes(g, S, 1, true);
      if (cur) {
        const z = S.trails.get(rid), tzi = Math.min(z.length - 1, Math.max(0, seen - 1));
        [[S.protoP, PASS], [S.protoF, FAIL]].forEach(([pt, col]) => g.append("line").attr("x1", cur[0]).attr("y1", cur[1]).attr("x2", pt[0]).attr("y2", pt[1]).attr("stroke", col).attr("stroke-width", 1.2).attr("stroke-dasharray", "3,3").attr("opacity", op));
        g.append("circle").attr("cx", cur[0]).attr("cy", cur[1]).attr("r", 11).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2.5); icon(g, "bot", cur[0], cur[1], 13, INK);
        const zq = window.PRECOG_BUNDLE.encoders.prefix.z[window.PRECOG_BUNDLE.rollouts.findIndex(r => r.rid === rid)];
        const pNow = S.score(zq[tzi]);
        const mx = 40, my = 250, bar = d3.scaleLinear().domain([0, 1]).range([my + 110, my + 24]);
        g.append("rect").attr("x", mx).attr("y", my).attr("width", 110).attr("height", 124).attr("rx", 8).attr("fill", "#fff").attr("stroke", "#e3e3df");
        g.append("text").attr("x", mx + 10).attr("y", my + 16).attr("font-size", 10).attr("fill", MUTED).text("P(pass)");
        g.append("rect").attr("x", mx + 10).attr("y", my + 24).attr("width", 20).attr("height", 86).attr("fill", "#f1f1ee");
        g.append("rect").attr("x", mx + 10).attr("y", bar(pNow)).attr("width", 20).attr("height", my + 110 - bar(pNow)).attr("fill", pNow >= 0.5 ? PASS : FAIL);
        g.append("line").attr("x1", mx + 8).attr("x2", mx + 32).attr("y1", bar(0.5)).attr("y2", bar(0.5)).attr("stroke", "#999").attr("stroke-dasharray", "2,2");
        g.append("text").attr("x", mx + 40).attr("y", my + 60).attr("font-size", 18).attr("font-weight", 700).attr("fill", INK).text(pNow.toFixed(2));
        g.append("text").attr("x", mx + 40).attr("y", my + 78).attr("font-size", 10).attr("fill", MUTED).text(`step ${Math.min(seen, steps)} of ${steps}`);
        const reveal = sub(q, 0.95, 1), pass = S.pass.get(rid);
        if (reveal > 0) icon(g, pass ? "circle-check" : "circle-x", cur[0] + 20, cur[1] - 16, 20, pass ? PASS : FAIL, reveal);
      }
      g.append("text").attr("x", STAGE.inX).attr("y", yC + 32).attr("font-size", 10).attr("fill", MUTED).attr("opacity", op).text(`a new rollout · step ${Math.min(seen, steps)} of ${steps}`);
    }
  } else {
    // ---- beat 5: the other fifteen labeled rollouts flow through quickly, get their colour, then the prototypes form
    const others = S.trains.slice(1), per = 0.55 / others.length;
    const label = ease(sub(q, 0.6, 0.72)), proto = ease(sub(q, 0.76, 1));
    // rollout 1 is already there and labelled
    const p0 = S.latent.get(focus.rid);
    g.append("circle").attr("cx", p0[0]).attr("cy", p0[1]).attr("r", 7).attr("fill", S.pass.get(focus.rid) ? PASS : FAIL).attr("stroke", "#fff").attr("stroke-width", 1.5);
    others.forEach((r, i) => {
      const f = sub(q, 0.02 + i * per, 0.02 + (i + 1) * per);
      if (f <= 0) return;
      const pt = S.latent.get(r.rid), pass = S.pass.get(r.rid);
      if (f < 1) {                                    // compressed flow: queued pills, a strip in the encoder, the point appearing
        const ic = S.icons.get(r.rid), oc = S.obs.get(r.rid);
        for (let k = 0; k < Math.min(12, r.n_steps); k++) pair(g, STAGE.inX + 8 + k * 15, yC - 10, ic[k], oc[k], 5, 0.9 * (1 - f));
        strip(g, lerp(STAGE.embX + 30, STAGE.encX + 20, ease(f)), yC - 10, r.rid, 0, 8, 4, 14, 1);
        g.append("circle").attr("cx", pt[0]).attr("cy", pt[1]).attr("r", 7 * ease(f)).attr("fill", "#9a9a9a").attr("stroke", "#fff").attr("stroke-width", 1.5);
      } else {
        const li = clamp01(label * 1.4 - i * 0.025);
        g.append("circle").attr("cx", pt[0]).attr("cy", pt[1]).attr("r", 7).attr("fill", d3.interpolateRgb("#9a9a9a", pass ? PASS : FAIL)(li)).attr("stroke", "#fff").attr("stroke-width", 1.5);
      }
    });
    g.append("text").attr("x", STAGE.inX).attr("y", yC + 32).attr("font-size", 10).attr("fill", MUTED).text(`${Math.min(others.length, Math.floor(sub(q, 0.02, 0.57) * others.length)) + 1} of ${S.trains.length} labeled rollouts`);
    if (proto > 0) drawPrototypes(g, S, proto, false);
  }
}
function drawPrototypes(g, S, k, withPoints) {
  if (withPoints) S.sup.forEach(r => { const pt = S.latent.get(r.rid); g.append("circle").attr("cx", pt[0]).attr("cy", pt[1]).attr("r", 6).attr("fill", S.pass.get(r.rid) ? PASS : FAIL).attr("fill-opacity", 0.8).attr("stroke", "#fff").attr("stroke-width", 1.2); });
  [[S.hullP, PASS, S.protoP, "circle-check"], [S.hullF, FAIL, S.protoF, "circle-x"]].forEach(([h, col, pt, ic]) => {
    const c = d3.polygonCentroid(h);
    g.append("path").attr("d", d3.line().curve(d3.curveCatmullRomClosed.alpha(0.6))(h.map(p => [lerp(c[0], p[0], k), lerp(c[1], p[1], k)]))).attr("fill", col).attr("fill-opacity", 0.09).attr("stroke", col).attr("stroke-opacity", 0.45).attr("stroke-dasharray", "4,3");
    if (!withPoints) S.sup.filter(r => (S.pass.get(r.rid) ? PASS : FAIL) === col).forEach(r => { const p = S.latent.get(r.rid);
      g.append("line").attr("x1", p[0]).attr("y1", p[1]).attr("x2", lerp(p[0], pt[0], k)).attr("y2", lerp(p[1], pt[1], k)).attr("stroke", col).attr("stroke-opacity", 0.35 * (1 - k * 0.6)); });
    g.append("path").attr("d", d3.symbol(d3.symbolStar, 320)()).attr("transform", `translate(${pt[0]},${pt[1]}) scale(${k})`).attr("fill", col).attr("stroke", "#fff").attr("stroke-width", 1.5);
    icon(g, ic, pt[0] + 18, pt[1] - 14, 16, col, k);
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
