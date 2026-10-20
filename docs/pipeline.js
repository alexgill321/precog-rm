/* "Subway map" pipeline animation: the behavior states of the loaded task are stations, rollouts are trains.
   Beat 1 trains branch across stations · 2 each route collapses into one point (the encoder) · 3 districts and
   prototypes form from labeled rollouts · 4 a new train is scored while it travels · 5 the gate stops it.
   Driven by a scrubber in [0,1]; listens for the "precog:bundle" event from app.js. */
(() => {
const PASS = "#2a9d4a", FAIL = "#d13b3b", NOISE = "#b9b9b9", NOISE_STROKE = "#7b3fb3", INK = "#1c1c1c", MUTED = "#6b6b6b";
const srColor = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);
const W = 900, H = 440, STEPS = 12, K_SUP = 8;
const BEATS = [
  { t0: 0.00, t1: 0.13, title: "1 · One rollout: actions and observations",
    text: "An agent works on a task one step at a time. Each step is an action it takes, a message or a tool call, and the observation that comes back from the user or the environment. The trajectory is that sequence; the outcome is only known at the end." },
  { t0: 0.13, t1: 0.22, title: "2 · The same task, sixteen attempts",
    text: "Sampling the task again gives a different rollout each time. Some succeed, some fail, and they share many of the same moves. Every step is embedded with a frozen text encoder, so each action–observation pair becomes a point in one shared space." },
  { t0: 0.22, t1: 0.34, title: "3 · Similar steps become behavior states",
    text: "Steps that say and do the same thing land close together; clustering them per task yields a handful of behavior states (stations). Each rollout is now a route through the station map, and a station's colour is how often rollouts passing through it end in success. The routes branch early, and some lines head into the red." },
  { t0: 0.34, t1: 0.48, title: "4 · The encoder folds each route into one point",
    text: "Precog-RM reads a route, the sequence of embedded action–observation steps, and maps it to a single point in a learned 128-d space (drawn here in 2-D). Routes that end alike land close together." },
  { t0: 0.48, t1: 0.60, title: "5 · A few labeled rollouts define two districts",
    text: "Eight finished successes and eight finished failures, one GRPO batch, are the support set. Their means are the pass and fail prototypes. No fitting happens: the prototypes are just averages, and everything else on the map is a rollout the model has not been told about." },
  { t0: 0.60, t1: 0.86, title: "6 · A new train is scored while it is still moving",
    text: "A new rollout is re-encoded after every step, so its point moves across the map as its history grows. Its predicted success is a softmax over the two distances; the meter fills as the point commits to a district." },
  { t0: 0.86, t1: 1.00, title: "7 · The gate closes and GRPO moves on",
    text: "When confidence crosses the threshold θ the controller terminates the rollout and hands GRPO the predicted outcome as its reward. The steps it did not generate are the saving; the baseline lane keeps running to the end." },
];
const INTRO_STEPS = 6, GRID_ROWS = 16;
let B = null, scene = null, playing = false, prog = 0, raf = null;
const $ = (s) => document.querySelector(s);
const lerp = (a, b, t) => a + (b - a) * t;
const ease = d3.easeCubicInOut;
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const dot = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; };
const dist = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; } return Math.sqrt(s); };
const mean = (vs) => { const m = new Array(vs[0].length).fill(0); vs.forEach(v => v.forEach((x, k) => m[k] += x / vs.length)); return m; };

function buildScene(b) {
  const E = b.encoders.prefix, rolls = b.rollouts, idx = new Map(rolls.map((r, i) => [r.rid, i]));
  const pass = new Map(rolls.map(r => [r.rid, r.reward >= 0.5]));
  const zF = (rid) => { const z = E.z[idx.get(rid)]; return z[z.length - 1]; };
  const proj = (v) => { const p = E.pca; const c = v.map((x, k) => x - p.mean[k]); return [dot(c, p.components[0]), dot(c, p.components[1])]; };
  // support: first K_SUP pass + K_SUP fail from the training folds (fold 0 held out); query from fold 0
  const held = new Set(b.cv_folds[0]);
  const train = rolls.filter(r => !held.has(r.rid));
  const sup = [...train.filter(r => pass.get(r.rid)).slice(0, K_SUP), ...train.filter(r => !pass.get(r.rid)).slice(0, K_SUP)];
  const supIds = new Set(sup.map(r => r.rid));
  const cp = mean(sup.filter(r => pass.get(r.rid)).map(r => zF(r.rid))), cf = mean(sup.filter(r => !pass.get(r.rid)).map(r => zF(r.rid)));
  const score = (z) => 1 / (1 + Math.exp(dist(z, cp) - dist(z, cf)));
  const thr = b.stop_threshold;
  // query: a held-out rollout the controller stops correctly, not absurdly early, earliest among those
  let best = null;
  rolls.filter(r => held.has(r.rid)).forEach(r => {
    const curve = E.z[idx.get(r.rid)].map(score);
    let stop = null; for (let t = 0; t < curve.length; t++) if (curve[t] > thr || 1 - curve[t] > thr) { stop = t; break; }
    if (stop == null || stop < 3) return;
    const correct = (curve[stop] > thr) === pass.get(r.rid);
    if (!correct) return;
    const frac = (stop + 1) / r.n_steps;
    if (!best || frac < best.frac) best = { r, curve, stop, frac };
  });
  if (!best) { const r = rolls.find(x => held.has(x.rid)); best = { r, curve: E.z[idx.get(r.rid)].map(score), stop: null, frac: 1 }; }
  // ---- beat 1 layout: stations per step (states ordered by success rate, success on top), 16 trains
  const sr = new Map(b.clusters.map(c => [c.id, c.sr])), noise = new Map(b.clusters.map(c => [c.id, c.noise]));
  const trains = sup;                                        // the 16 support rollouts are the beat-1 trains
  const xs = d3.scalePoint().domain(d3.range(STEPS)).range([70, 560]).padding(0.4);
  const stations = [], stPos = new Map();                    // key `${t}:${c}` -> {x,y}
  for (let t = 0; t < STEPS; t++) {
    const present = new Map();
    trains.forEach(r => { if (t < r.n_steps) { const c = r.steps[t].c; present.set(c, (present.get(c) || 0) + 1); } });
    const order = [...present.keys()].sort((a, c) => (sr.get(c) - sr.get(a)) || (a - c));
    const ys = d3.scalePoint().domain(order).range([70, 370]).padding(0.5);
    order.forEach(c => { const s = { t, c, x: xs(t), y: order.length === 1 ? 220 : ys(c), n: present.get(c) }; stations.push(s); stPos.set(`${t}:${c}`, s); });
  }
  // route vertices per train (with a small lane offset so parallel trains stay visible)
  const routes = trains.map((r, i) => {
    const off = ((i % 4) - 1.5) * 2.2;
    const pts = d3.range(Math.min(STEPS, r.n_steps)).map(t => { const s = stPos.get(`${t}:${r.steps[t].c}`); return [s.x, s.y + off]; });
    return { rid: r.rid, pass: pass.get(r.rid), pts };
  });
  // ---- latent layout (beats 2+): PCA of finals + the query's prefix trail, scaled into the right-hand map
  const finals = new Map(rolls.map(r => [r.rid, proj(zF(r.rid))]));
  const trail = E.z[idx.get(best.r.rid)].map(proj);
  const allx = [...finals.values()].map(p => p[0]).concat(trail.map(p => p[0])), ally = [...finals.values()].map(p => p[1]).concat(trail.map(p => p[1]));
  const padx = (d3.max(allx) - d3.min(allx)) * 0.08, pady = (d3.max(ally) - d3.min(ally)) * 0.08;
  const lx = d3.scaleLinear().domain([d3.min(allx) - padx, d3.max(allx) + padx]).range([130, 770]);
  const ly = d3.scaleLinear().domain([d3.min(ally) - pady, d3.max(ally) + pady]).range([380, 60]);
  const L = (p) => [lx(p[0]), ly(p[1])];
  const latent = new Map([...finals].map(([rid, p]) => [rid, L(p)]));
  const protoP = L(proj(cp)), protoF = L(proj(cf));
  const hull = (ids) => { const pts = ids.map(rid => latent.get(rid)); const h = d3.polygonHull(pts) || pts; return h; };
  // intro: the focus rollout's step texts, and the grid rows of the zoom-out
  const focus = trains[0];
  const excerpt = (t, n) => { t = (t || "").replace(/\s+/g, " ").trim(); return t.length > n ? t.slice(0, n - 1) + "…" : t; };
  const focusSteps = focus.steps.slice(0, INTRO_STEPS).map((st, t) => ({ t, a: excerpt(st.a, 175), o: excerpt(st.o, 175) }));
  const rowY = (i) => 48 + i * (338 / (GRID_ROWS - 1));
  return { rolls, pass, supIds, sup, trains, stations, routes, latent, protoP, protoF, focus, focusSteps, rowY, stPos, hullP: hull(sup.filter(r => pass.get(r.rid)).map(r => r.rid)), hullF: hull(sup.filter(r => !pass.get(r.rid)).map(r => r.rid)),
           query: best, trailL: trail.map(L), thr, sr, noise, xs };
}

function beatAt(p) { for (let i = 0; i < BEATS.length; i++) if (p < BEATS[i].t1 || i === BEATS.length - 1) return [i, clamp01((p - BEATS[i].t0) / (BEATS[i].t1 - BEATS[i].t0))]; }

function render(p) {
  if (!scene) return;
  const svg = d3.select("#pipeline"); svg.selectAll("*").remove();
  svg.attr("viewBox", `0 0 ${W} ${H}`);
  const S = scene, [bi, q] = beatAt(p);
  const g = svg.append("g");
  $("#pipeline-title").textContent = BEATS[bi].title; $("#pipeline-text").textContent = BEATS[bi].text;
  document.querySelectorAll("#pipeline-beats button").forEach((el, i) => el.classList.toggle("on", i === bi));
  // progress of each phase, 0..1 (beat indices: 0 one rollout, 1 many, 2 states, 3 encoder, 4 districts, 5 scoring, 6 gate)
  const intro = bi === 0 ? q : 1;
  const zoom = bi < 1 ? 0 : bi === 1 ? ease(q) : 1;                   // single rollout -> grid of 16
  const group = bi < 2 ? 0 : bi === 2 ? ease(q) : 1;                  // grid cards -> stations + routes
  const morph = bi < 3 ? 0 : bi === 3 ? ease(q) : 1;                  // routes -> points
  const district = bi < 4 ? 0 : bi === 4 ? ease(q) : 1;               // hulls + prototypes + other rollouts
  const travel = bi < 5 ? 0 : bi === 5 ? q : 1;                       // query along its trail
  const gate = bi < 6 ? 0 : ease(q);
  if (bi <= 2) drawIntro(g, S, bi, intro, zoom, group);
  // ---- stations (beat 1, fade during morph)
  if (bi >= 2 && morph < 1) {
    const st = g.append("g").attr("opacity", (bi === 2 ? group : 1) * (1 - morph));
    st.selectAll("line").data(d3.range(STEPS - 1)).join("line").attr("x1", t => S.xs(t)).attr("x2", t => S.xs(t + 1)).attr("y1", 220).attr("y2", 220).attr("stroke", "#eee");
    st.selectAll("circle").data(S.stations).join("circle").attr("cx", d => d.x).attr("cy", d => d.y).attr("r", d => 6 + Math.sqrt(d.n) * 2.2)
      .attr("fill", d => S.noise.get(d.c) ? NOISE : srColor(S.sr.get(d.c))).attr("stroke", d => S.noise.get(d.c) ? NOISE_STROKE : "#fff").attr("stroke-width", 1.5);
    st.selectAll("text").data(d3.range(STEPS)).join("text").attr("x", t => S.xs(t)).attr("y", 400).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", MUTED).text(t => t + 1);
    st.append("text").attr("x", S.xs(0)).attr("y", 420).attr("font-size", 11).attr("fill", MUTED).text("step →  (stations = behavior states, coloured by success rate)");
  }
  // ---- trains: beat 1 draw-on, beat 2 collapse to latent points
  const reveal = 1, routeOp = bi < 2 ? 0 : bi === 2 ? clamp01((q - 0.45) / 0.55) : 1;
  if (routeOp > 0) S.routes.forEach(r => {
    const target = S.latent.get(r.rid);
    const n = Math.max(2, Math.ceil(r.pts.length * reveal));
    const pts = r.pts.slice(0, n).map(pt => [lerp(pt[0], target[0], morph), lerp(pt[1], target[1], morph)]);
    g.append("path").attr("d", d3.line().curve(d3.curveMonotoneX)(pts)).attr("fill", "none").attr("stroke", r.pass ? PASS : FAIL).attr("stroke-width", 2).attr("stroke-opacity", routeOp * (0.75 - 0.35 * morph));
    // train head
    const head = pts[pts.length - 1];
    g.append("circle").attr("cx", head[0]).attr("cy", head[1]).attr("r", 4.5 + 1.5 * morph).attr("fill", r.pass ? PASS : FAIL).attr("stroke", "#fff").attr("stroke-width", 1.2).attr("opacity", routeOp);
  });
  // ---- beat 3: other rollouts, districts, prototypes
  if (district > 0) {
    S.rolls.filter(r => !S.supIds.has(r.rid) && r.rid !== S.query.r.rid).forEach(r => {
      const [x, y] = S.latent.get(r.rid);
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 4).attr("fill", "#fff").attr("stroke", "#999").attr("stroke-width", 1.2).attr("opacity", district * 0.8);
    });
    [[S.hullP, PASS], [S.hullF, FAIL]].forEach(([h, col]) => {
      const c = d3.polygonCentroid(h);
      const pts = h.map(pt => [lerp(c[0], pt[0], district), lerp(c[1], pt[1], district)]);
      g.append("path").attr("d", d3.line().curve(d3.curveCatmullRomClosed.alpha(0.6))(pts)).attr("fill", col).attr("fill-opacity", 0.10).attr("stroke", col).attr("stroke-opacity", 0.5).attr("stroke-dasharray", "4,3");
    });
    [[S.protoP, PASS, "pass prototype"], [S.protoF, FAIL, "fail prototype"]].forEach(([pt, col, label]) => {
      g.append("path").attr("d", d3.symbol(d3.symbolStar, 260)()).attr("transform", `translate(${pt[0]},${pt[1]}) scale(${district})`).attr("fill", col).attr("stroke", "#fff").attr("stroke-width", 1.5);
      g.append("text").attr("x", pt[0] + 12).attr("y", pt[1] + 4).attr("font-size", 11).attr("fill", col).attr("opacity", district).text(label);
    });
    g.append("text").attr("x", 130).attr("y", 420).attr("font-size", 11).attr("fill", MUTED).attr("opacity", district).text("learned latent space (2-D PCA view) · filled = support rollouts · hollow = rollouts the model has not seen labels for");
  }
  // ---- beat 4/5: the query train
  if (travel > 0) {
    const Q = S.query, T = Q.r.n_steps, stop = Q.stop ?? T - 1;
    const tEnd = bi >= 4 ? stop : Math.min(T - 1, Math.floor(travel * (stop + 1.999)));    // beat 4 travels up to the stop step
    const tIdx = Math.max(0, Math.min(tEnd, stop));
    const pathPts = S.trailL.slice(0, tIdx + 1);
    g.append("path").attr("d", d3.line().curve(d3.curveCatmullRom)(pathPts)).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 2).attr("stroke-opacity", 0.8);
    pathPts.slice(0, -1).forEach(pt => g.append("circle").attr("cx", pt[0]).attr("cy", pt[1]).attr("r", 2.2).attr("fill", INK).attr("opacity", 0.6));
    const cur = pathPts[pathPts.length - 1];
    [[S.protoP, PASS], [S.protoF, FAIL]].forEach(([pt, col]) => g.append("line").attr("x1", cur[0]).attr("y1", cur[1]).attr("x2", pt[0]).attr("y2", pt[1]).attr("stroke", col).attr("stroke-width", 1.2).attr("stroke-dasharray", "3,3"));
    g.append("circle").attr("cx", cur[0]).attr("cy", cur[1]).attr("r", 8).attr("fill", "#fff").attr("stroke", INK).attr("stroke-width", 2.5);
    g.append("text").attr("x", cur[0] + 12).attr("y", cur[1] - 10).attr("font-size", 11).attr("fill", INK).text(`new rollout · step ${tIdx + 1}`);
    // confidence meter + track (bottom-left panel)
    const pNow = Q.curve[tIdx];
    const mx = 20, my = 300, mw = 90;
    g.append("rect").attr("x", mx).attr("y", my).attr("width", mw).attr("height", 90).attr("rx", 8).attr("fill", "#fff").attr("stroke", "#e3e3df");
    g.append("text").attr("x", mx + 8).attr("y", my + 16).attr("font-size", 11).attr("fill", MUTED).text("P(pass)");
    const bar = d3.scaleLinear().domain([0, 1]).range([my + 80, my + 24]);
    g.append("rect").attr("x", mx + 8).attr("y", my + 24).attr("width", 16).attr("height", 56).attr("fill", "#f1f1ee");
    g.append("rect").attr("x", mx + 8).attr("y", bar(pNow)).attr("width", 16).attr("height", my + 80 - bar(pNow)).attr("fill", pNow >= 0.5 ? PASS : FAIL);
    [S.thr, 1 - S.thr].forEach(v => g.append("line").attr("x1", mx + 6).attr("x2", mx + 26).attr("y1", bar(v)).attr("y2", bar(v)).attr("stroke", INK).attr("stroke-dasharray", "2,2"));
    g.append("text").attr("x", mx + 30).attr("y", bar(S.thr) + 4).attr("font-size", 10).attr("fill", MUTED).text(`θ=${S.thr}`);
    g.append("text").attr("x", mx + 30).attr("y", bar(1 - S.thr) + 4).attr("font-size", 10).attr("fill", MUTED).text(`1−θ`);
    g.append("text").attr("x", mx + 30).attr("y", my + 56).attr("font-size", 13).attr("font-weight", 600).attr("fill", INK).text(pNow.toFixed(2));
    // tracks: controller lane vs baseline lane
    const tx = 130, ty = 408, tw = 640, cell = tw / T;
    const lane = (y, filled, label, col) => {
      g.append("text").attr("x", tx - 6).attr("y", y + 9).attr("text-anchor", "end").attr("font-size", 10).attr("fill", MUTED).text(label);
      d3.range(T).forEach(t => g.append("rect").attr("x", tx + t * cell + 1).attr("y", y).attr("width", cell - 2).attr("height", 10).attr("rx", 2)
        .attr("fill", t < filled ? col : "#eceae5"));
    };
    if (gate > 0 || bi === 3) {
      const baseFill = bi >= 4 ? Math.round(stop + 1 + gate * (T - stop - 1)) : tIdx + 1;
      lane(ty - 14, tIdx + 1, "controller", INK); lane(ty + 2, baseFill, "baseline", "#9a9a9a");
      g.append("text").attr("x", tx).attr("y", ty + 28).attr("font-size", 11).attr("fill", MUTED).text("generation steps");
    }
  }
  // ---- beat 5: gate + stamp + saving
  if (gate > 0) {
    const Q = S.query, T = Q.r.n_steps, stop = Q.stop ?? T - 1, cur = S.trailL[stop];
    const predPass = Q.curve[stop] > S.thr, col = predPass ? PASS : FAIL;
    const gx = 130 + (stop + 1) * (640 / T);
    g.append("line").attr("x1", gx).attr("x2", gx).attr("y1", 408 - 20).attr("y2", 408 + 16).attr("stroke", INK).attr("stroke-width", 3 * gate);
    const s = 0.6 + 0.4 * gate;
    const st = g.append("g").attr("transform", `translate(${W - 190},${34}) rotate(-4) scale(${s})`).attr("opacity", gate);
    g.append("line").attr("x1", cur[0]).attr("y1", cur[1]).attr("x2", W - 190 + 75 * s).attr("y2", 34 + 36 * s).attr("stroke", col).attr("stroke-width", 1).attr("stroke-opacity", 0.5 * gate);
    st.append("rect").attr("x", 0).attr("y", 0).attr("width", 150).attr("height", 36).attr("rx", 6).attr("fill", "#fff").attr("stroke", col).attr("stroke-width", 2.5);
    st.append("text").attr("x", 75).attr("y", 16).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 700).attr("fill", col).text(`STOPPED · predicted ${predPass ? "PASS" : "FAIL"}`);
    st.append("text").attr("x", 75).attr("y", 29).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", MUTED).text(`reward ${predPass ? 1 : 0} handed to GRPO · actual: ${S.pass.get(Q.r.rid) ? "pass" : "fail"}`);
    const saved = Math.round(100 * (1 - (stop + 1) / T));
    g.append("text").attr("x", 770).attr("y", 408 - 20).attr("text-anchor", "end").attr("font-size", 12).attr("font-weight", 600).attr("fill", INK).attr("opacity", gate)
      .text(`${T - stop - 1} of ${T} steps never generated (${saved}%)`);
  }
}

// ---------- intro beats: one rollout -> grid of rollouts -> stations ----------
const ACT_FILL = "#e8eefc", OBS_FILL = "#f7ecd9", ACT_INK = "#2b5fb3", OBS_INK = "#8a5a00";
function wrap(text, width, font) {
  const words = text.split(" "), lines = []; let cur = "";
  const cpl = Math.max(8, Math.floor(width / (font * 0.52)));
  words.forEach(w => { if ((cur + " " + w).trim().length > cpl && cur) { lines.push(cur); cur = w; } else cur = (cur + " " + w).trim(); });
  if (cur) lines.push(cur); return lines;
}
function drawIntro(g, S, bi, intro, zoom, group) {
  const n = S.focusSteps.length, bigW = 122, bigH = 250, bigX = (t) => 48 + t * 136, bigY = 92;
  const smallW = 30, smallH = 12;
  const stepsShown = bi === 0 ? Math.min(n, Math.floor(intro * (n + 0.999))) : n;
  // header line + agent sprite (beat 0)
  if (bi === 0) {
    g.append("text").attr("x", 48).attr("y", 40).attr("font-size", 12).attr("fill", INK).attr("font-weight", 600).text(`one rollout of ${S.focus.steps.length} steps · outcome: ${S.pass.get(S.focus.rid) ? "pass" : "fail"} (unknown until the end)`);
    g.append("line").attr("x1", 48).attr("x2", 48 + (n - 1) * 136 + bigW).attr("y1", 70).attr("y2", 70).attr("stroke", "#ddd");
    const ax = stepsShown > 0 ? bigX(Math.min(stepsShown, n) - 1) + bigW / 2 : bigX(0);
    const sp = g.append("g").attr("transform", `translate(${ax},70)`);
    sp.append("circle").attr("r", 9).attr("fill", INK); sp.append("text").attr("y", 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#fff").text("A");
    g.append("text").attr("x", 48).attr("y", 365).attr("font-size", 11).attr("fill", MUTED).text("step →");
  }
  // every train's cards: focus row is expanded in beat 0 and shrinks into row 0 of the grid in beat 1
  S.trains.forEach((r, i) => {
    const T = Math.min(STEPS, r.n_steps);
    for (let t = 0; t < T; t++) {
      const isFocus = i === 0, big = isFocus && t < n;
      if (!isFocus && zoom === 0) continue;
      if (big && bi === 0 && t >= stepsShown) continue;
      // positions: big card -> grid -> station pile
      const gx = S.xs(t) - smallW / 2, gy = S.rowY(i) - smallH / 2;
      let x, y, w, h;
      if (big) { x = lerp(bigX(t), gx, zoom); y = lerp(bigY, gy, zoom); w = lerp(bigW, smallW, zoom); h = lerp(bigH, smallH, zoom); }
      else { x = gx; y = gy; w = smallW; h = smallH; }
      if (group > 0) { const st = S.stPos.get(`${t}:${r.steps[t].c}`); const k = S.trains.filter((rr, j) => j < i && t < rr.n_steps && rr.steps[t].c === r.steps[t].c).length;
        x = lerp(x, st.x - smallW / 2, group); y = lerp(y, st.y - smallH / 2 + (k - 2) * 1.5, group); }
      const op = (isFocus ? 1 : zoom) * (1 - group);
      const card = g.append("g").attr("transform", `translate(${x},${y})`).attr("opacity", op);
      card.append("rect").attr("width", w).attr("height", h / 2).attr("fill", ACT_FILL).attr("stroke", "#fff").attr("stroke-width", 0.5);
      card.append("rect").attr("y", h / 2).attr("width", w).attr("height", h / 2).attr("fill", OBS_FILL).attr("stroke", "#fff").attr("stroke-width", 0.5);
      card.append("rect").attr("width", w).attr("height", h).attr("fill", "none").attr("stroke", "#c9c9c3").attr("rx", 2);
      if (big && zoom < 0.5) {
        const fs = 9.5, txt = S.focusSteps[t], tOp = 1 - zoom * 2;
        const put = (lines, y0, ink, label) => { const tx = card.append("text").attr("x", 5).attr("y", y0).attr("font-size", fs).attr("fill", ink).attr("opacity", tOp);
          tx.append("tspan").attr("x", 5).attr("font-weight", 700).attr("font-size", 7.5).text(label);
          lines.slice(0, 9).forEach(l => tx.append("tspan").attr("x", 5).attr("dy", fs + 2).text(l)); };
        put(wrap(txt.a, bigW - 10, fs), 10, ACT_INK, `step ${t + 1} · action`); put(wrap(txt.o, bigW - 10, fs), h / 2 + 10, OBS_INK, "observation");
      }
    }
    // outcome mark at the end of each grid row
    if (zoom > 0.6 && group < 1) g.append("circle").attr("cx", S.xs(STEPS - 1) + 28).attr("cy", S.rowY(i)).attr("r", 4).attr("fill", S.pass.get(r.rid) ? PASS : FAIL).attr("opacity", (zoom - 0.6) / 0.4 * (1 - group));
  });
  if (bi >= 1) {
    g.append("text").attr("x", 62).attr("y", 30).attr("font-size", 12).attr("fill", INK).attr("font-weight", 600).attr("opacity", zoom * (1 - group)).text(`${S.trains.length} rollouts of the same task · first ${STEPS} steps · dot = final outcome`);
    g.append("text").attr("x", 62).attr("y", 420).attr("font-size", 11).attr("fill", MUTED).attr("opacity", zoom * (1 - group)).text("each card is one embedded action (blue) and observation (tan) · step →");
  }
}

// ---------- controls ----------
function setProg(p, fromUser) { prog = clamp01(p); $("#pipeline-scrub").value = Math.round(prog * 1000); render(prog); if (fromUser) stop(); }
function play() { if (playing) return; playing = true; $("#pipeline-play").textContent = "❚❚ pause"; let last = null; const DUR = 40000;
  const tick = (ts) => { if (!playing) return; if (last != null) { prog += (ts - last) / DUR; if (prog >= 1) prog = 0; $("#pipeline-scrub").value = Math.round(prog * 1000); render(prog); } last = ts; raf = requestAnimationFrame(tick); };
  raf = requestAnimationFrame(tick); }
function stop() { playing = false; $("#pipeline-play").textContent = "▶ play"; if (raf) cancelAnimationFrame(raf); }
function initControls() {
  const beats = $("#pipeline-beats");
  BEATS.forEach((b, i) => { const el = document.createElement("button"); el.textContent = i + 1; el.title = b.title; el.onclick = () => setProg(b.t0 + 0.02, true); beats.appendChild(el); });
  $("#pipeline-scrub").oninput = (e) => setProg(+e.target.value / 1000, true);
  $("#pipeline-play").onclick = () => playing ? stop() : play();
}
initControls();
function load(b) { B = b; scene = buildScene(B); render(prog); }
document.addEventListener("precog:bundle", (e) => load(e.detail));
if (window.PRECOG_BUNDLE) load(window.PRECOG_BUNDLE);
// autoplay once the scene is on screen
const io = new IntersectionObserver((es) => { if (es.some(x => x.isIntersecting) && scene && !playing && prog === 0) play(); }, { threshold: 0.4 });
io.observe($("#pipeline-section"));
})();
