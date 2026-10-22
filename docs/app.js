/* Project page interactives. Data: docs/data/index.json + docs/data/<benchmark>/<policy>/<task>.json
   (built by scripts/build_web_bundles.py). No framework, D3 only. */
(() => {
const $ = (s) => document.querySelector(s);
const setStatus = (t) => { const el = document.querySelector("#load-status"); if (el) el.textContent = t; };
// Bundles live in the dataset repo (web/ part); a local build (scripts/build_web_bundles.py -> docs/data) is used when served from localhost.
const LOCAL = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
const DATA_BASE = LOCAL ? "data" : "https://huggingface.co/datasets/agill32/precog-rm/resolve/main/web";
const DATA_V = "7";   // bump when bundle contents change
const PASS = "#2a9d4a", FAIL = "#d13b3b", NOISE = "#b9b9b9", NOISE_STROKE = "#7b3fb3";
const srColor = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);
const fmt = d3.format(".2f"), fmt3 = d3.format(".3f");

let INDEX = null, B = null;                      // index, current bundle
let MB = null, mapSel = null, mapFollowsPage = true;   // the state-map panel's own bundle and selected state
const mapCache = new Map();
let S = { encoder: "prefix", preset: "cv0", support: new Set(), sel: null, t: 1 };

// ---------- loading ----------
async function init() {
  INDEX = await (await fetch(`${DATA_BASE}/index.json?v=${DATA_V}`)).json();
  const bms = [...new Set(INDEX.tasks.map(t => t.benchmark))];
  fill($("#sel-benchmark"), bms);
  $("#sel-benchmark").onchange = () => { refreshPolicies(); loadSelected(); };
  $("#sel-policy").onchange = () => { refreshTasks(); loadSelected(); };
  $("#sel-task").onchange = loadSelected;
  refreshPolicies();
  initMapSelectors();
  if ($("#n-bundled")) $("#n-bundled").textContent = INDEX.tasks.length;
  // the scenes fetch a second task (act 4) through this
  window.PrecogData = { index: INDEX, url: (t) => `${DATA_BASE}/${t.benchmark}/${t.policy}/${t.task}.json?v=${DATA_V}` };
  document.dispatchEvent(new CustomEvent("precog:index"));
  const q = new URLSearchParams(location.search);
  if (q.get("task")) {
    const t = INDEX.tasks.find(x => x.task === q.get("task"));
    if (t) { $("#sel-benchmark").value = t.benchmark; refreshPolicies(); $("#sel-policy").value = t.policy; refreshTasks(); $("#sel-task").value = t.task; }
  }
  await loadSelected();
}
function fill(sel, vals, labels) { sel.innerHTML = ""; vals.forEach((v, i) => { const o = document.createElement("option"); o.value = v; o.textContent = labels ? labels[i] : v; sel.appendChild(o); }); }
function refreshPolicies() {
  const bm = $("#sel-benchmark").value;
  fill($("#sel-policy"), [...new Set(INDEX.tasks.filter(t => t.benchmark === bm).map(t => t.policy))]);
  refreshTasks();
}
function refreshTasks() {
  const bm = $("#sel-benchmark").value, pol = $("#sel-policy").value;
  const ts = INDEX.tasks.filter(t => t.benchmark === bm && t.policy === pol);
  fill($("#sel-task"), ts.map(t => t.task), ts.map(t => `${t.task_id}  (success ${Math.round(t.success_rate * 100)}%)`));
}
async function loadSelected() {
  const bm = $("#sel-benchmark").value, pol = $("#sel-policy").value, task = $("#sel-task").value;
  setStatus(`Loading ${bm}/${pol}/${task}…`);
  B = await (await fetch(`${DATA_BASE}/${bm}/${pol}/${task}.json?v=${DATA_V}`)).json();
  setStatus("");
  history.replaceState(null, "", `?task=${task}`);
  const maxT = d3.max(B.rollouts, r => r.n_steps);
  $("#task-badges").innerHTML = [
    `<span class="badge">${B.task_id}</span>`, `<span class="badge">${B.n_rollouts} rollouts</span>`,
    `<span class="badge">success rate ${Math.round(B.success_rate * 100)}%</span>`,
    `<span class="badge">${B.clusters.length} behavior states</span>`, `<span class="badge">up to ${maxT} steps</span>`,
    B.held_out ? `<span class="badge">held-out task (not in any encoder's training pool)</span>` : `<span class="badge warn">task was in the encoders' training pool</span>`,
  ].join("");
  const setN = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(([v, l]) => `<div class="num"><b>${v}</b><span>${l}</span></div>`).join(""); };
  setN("#n1", [[B.n_rollouts, "rollouts of " + B.task_id], [B.clusters.length, "behavior states (incl. outlier bin)"], [`${Math.round(B.success_rate * 100)}%`, "success rate"], [maxT, "longest rollout, in steps"]]);
  const ENC_LABEL = { prefix: "Precog-RM, prefix-trained (deployed in GRPO)", full: "Precog-RM, generalist, full rollouts only", raw: "raw action–observation space, no learning (nearest centroid)" };
  const encSel = $("#sel-encoder"); encSel.innerHTML = ""; ["prefix", "full", "raw"].filter(k => B.encoders[k]).forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = ENC_LABEL[k]; encSel.appendChild(o); });
  if (!B.encoders[S.encoder]) S.encoder = Object.keys(B.encoders)[0]; encSel.value = S.encoder;
  S.thr = B.stop_threshold; $("#theta").value = S.thr; $("#theta-val").textContent = S.thr.toFixed(2);
  S.sel = null; applyPreset();
  drawScoring();
  window.PRECOG_BUNDLE = B;
  document.dispatchEvent(new CustomEvent("precog:bundle", { detail: B }));
  if (mapFollowsPage) { setMapTo(B.benchmark, B.policy, B.task); loadMap(); }
}
$("#sel-encoder").onchange = (e) => { S.encoder = e.target.value; drawScoring(); };
$("#sel-support").onchange = (e) => { S.preset = e.target.value; applyPreset(); drawScoring(); };
$("#prefix-t").oninput = (e) => { S.t = +e.target.value; drawScoring(false); };
if ($("#theta")) $("#theta").oninput = (e) => { S.thr = +e.target.value; $("#theta-val").textContent = S.thr.toFixed(2); drawScoring(false); };
document.addEventListener("precog:k", (e) => { S.preset = `k${e.detail}`; $("#sel-support").value = S.preset; applyPreset(); drawScoring(); });
window.addEventListener("resize", () => { if (B) drawScoring(false); });

// ---------- 1. the state map of one task (act 3's "look inside"): its own benchmark / policy / task selection ----------
const BENCH_L = { tau2: "Tau2-Bench", bfclv3: "BFCLv3", agentclinic: "AgentClinic" };
function initMapSelectors() {
  if (!$("#map-benchmark")) return;
  const bms = [...new Set(INDEX.tasks.map(t => t.benchmark))];
  fill($("#map-benchmark"), bms, bms.map(b => BENCH_L[b] || b));
  $("#map-benchmark").onchange = () => { mapFollowsPage = false; mapPolicies(); loadMap(); };
  $("#map-policy").onchange = () => { mapFollowsPage = false; mapTasks(); loadMap(); };
  $("#map-task").onchange = () => { mapFollowsPage = false; loadMap(); };
  mapPolicies();
}
function mapPolicies() { const bm = $("#map-benchmark").value; fill($("#map-policy"), [...new Set(INDEX.tasks.filter(t => t.benchmark === bm).map(t => t.policy))]); mapTasks(); }
function mapTasks() { const bm = $("#map-benchmark").value, pol = $("#map-policy").value, ts = INDEX.tasks.filter(t => t.benchmark === bm && t.policy === pol);
  fill($("#map-task"), ts.map(t => t.task), ts.map(t => `${t.task_id}  (success ${Math.round(t.success_rate * 100)}%)`)); }
function setMapTo(bm, pol, task) { if (!$("#map-benchmark")) return; $("#map-benchmark").value = bm; mapPolicies(); $("#map-policy").value = pol; mapTasks(); $("#map-task").value = task; }
async function loadMap() {
  if (!$("#map")) return;
  const bm = $("#map-benchmark").value, pol = $("#map-policy").value, task = $("#map-task").value; if (!task) return;
  const key = `${bm}/${pol}/${task}`;
  if (B && `${B.benchmark}/${B.policy}/${B.task}` === key) mapCache.set(key, Promise.resolve(B));
  if (!mapCache.has(key)) mapCache.set(key, fetch(`${DATA_BASE}/${key}.json?v=${DATA_V}`).then(r => r.json()));
  $("#map-badges").innerHTML = `<span class="badge">loading ${task}…</span>`;
  MB = await mapCache.get(key); mapSel = null;
  const maxT = d3.max(MB.rollouts, r => r.n_steps);
  $("#map-badges").innerHTML = [`<span class="badge">${MB.task_id}</span>`, `<span class="badge">${MB.n_rollouts} rollouts</span>`, `<span class="badge">success rate ${Math.round(MB.success_rate * 100)}%</span>`,
    `<span class="badge">${MB.clusters.filter(c => !c.noise).length} states</span>`, `<span class="badge">up to ${maxT} steps</span>`].join("");
  $("#cluster-panel").innerHTML = `<p class="muted">Hover a state to see where rollouts go from it; click it to read the steps it merges.</p>`;
  drawMap();
}
function drawMap() { if (!MB || !window.PrecogOpening || !window.PrecogOpening.mapPanel) return;
  window.PrecogOpening.mapPanel($("#map"), MB, { selected: mapSel, onSelect: (c) => { mapSel = mapSel === c ? null : c; drawMap(); showState(mapSel); } }); }
function showState(c) {
  if (c == null) { $("#cluster-panel").innerHTML = `<p class="muted">Hover a state to see where rollouts go from it; click it to read the steps it merges.</p>`; return; }
  const cl = MB.clusters.find(x => x.id === c), members = [];
  MB.rollouts.forEach(r => r.steps.forEach((s, i) => { if (s.c === c) members.push({ rid: r.rid, t: i, pass: r.reward >= 0.5, s }); }));
  members.sort((a, b) => a.t - b.t || a.rid - b.rid);
  const ts = members.map(m => m.t), shown = 40, med = cl.medoid || [];
  const card = (mm) => `<div class="pair ${mm.pass ? "pass" : "fail"}"><div class="who">rollout ${mm.rid}, step ${mm.t + 1}${mm.rid === med[0] && mm.t === med[1] ? " · medoid" : ""}</div><div class="a">${esc(mm.s.a)}</div><div class="o">${esc(mm.s.o)}</div></div>`;
  $("#cluster-panel").innerHTML = `<h3>State ${c}${cl.noise ? " · unclustered steps" : ""}</h3>
    <p class="muted small-text">${cl.n} steps from ${cl.n_rollouts} rollouts, at steps ${d3.min(ts) + 1} to ${d3.max(ts) + 1} · ${Math.round(cl.sr * cl.n_rollouts)} of those ${cl.n_rollouts} rollouts succeed</p>
    ${members.slice(0, shown).map(card).join("")}${members.length > shown ? `<p class="muted">… ${members.length - shown} more</p>` : ""}`;
}
const esc = (s) => s.replace(/[&<>]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));

// ---------- 2. latent-space scoring ----------
function applyPreset() {
  const rids = B.rollouts.map(r => r.rid);
  if (S.preset.startsWith("cv")) { const held = new Set(B.cv_folds[+S.preset[2]]); S.support = new Set(rids.filter(r => !held.has(r))); }
  else if (S.preset.startsWith("k")) { const k = +S.preset.slice(1), held = new Set(B.cv_folds[0]), pass = new Map(B.rollouts.map(r => [r.rid, r.reward >= 0.5]));
    const train = rids.filter(r => !held.has(r)); S.support = new Set([...train.filter(r => pass.get(r)).slice(0, k), ...train.filter(r => !pass.get(r)).slice(0, k)]); }
  // custom: keep as is
  if (S.sel == null || S.support.has(S.sel)) S.sel = rids.find(r => !S.support.has(r)) ?? rids[0];
}
function enc() { return B.encoders[S.encoder]; }
function zFinal(i) { const z = enc().z[i]; return z[z.length - 1]; }
function proj(v) { const p = enc().pca; const c = v.map((x, k) => x - p.mean[k]); return [dot(c, p.components[0]), dot(c, p.components[1])]; }
const dot = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; };
const dist = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; } return Math.sqrt(s); };
const mean = (vs) => { const m = new Array(vs[0].length).fill(0); vs.forEach(v => v.forEach((x, k) => m[k] += x / vs.length)); return m; };
function scoreFn(cp, cf) {     // softmax over [-d_fail, -d_pass] -> P(pass); temperature 1, no Platt (both released encoders)
  return (z) => { if (!cp) return 0; if (!cf) return 1; const dp = dist(z, cp), df = dist(z, cf); return 1 / (1 + Math.exp(dp - df)); };
}
function auroc(scores, labels) {
  const pos = scores.filter((s, i) => labels[i]), neg = scores.filter((s, i) => !labels[i]);
  if (!pos.length || !neg.length) return NaN;
  let s = 0; pos.forEach(p => neg.forEach(n => { s += p > n ? 1 : p === n ? 0.5 : 0; })); return s / (pos.length * neg.length);
}

function drawScoring(resetT = true) {
  const E = enc(), rolls = B.rollouts, idx = new Map(rolls.map((r, i) => [r.rid, i]));
  const pass = new Map(rolls.map(r => [r.rid, r.reward >= 0.5]));
  const supP = [...S.support].filter(r => pass.get(r)).map(r => zFinal(idx.get(r)));
  const supF = [...S.support].filter(r => !pass.get(r)).map(r => zFinal(idx.get(r)));
  const cp = supP.length ? mean(supP) : null, cf = supF.length ? mean(supF) : null;
  const score = scoreFn(cp, cf);
  const heldK = S.preset.startsWith("k") ? new Set(B.cv_folds[0]) : null;
  const queries = rolls.filter(r => !S.support.has(r.rid) && (!heldK || heldK.has(r.rid)));
  const qs = queries.map(r => score(zFinal(idx.get(r.rid)))), ql = queries.map(r => pass.get(r.rid));
  const au = auroc(qs, ql);
  const setN3 = (items) => { const el = $("#n3"); if (el) el.innerHTML = items.map(([v, l]) => `<div class="num"><b>${v}</b><span>${l}</span></div>`).join(""); };
  setN3([[S.encoder === "raw" ? `${E.z[0][0].length}-d` : "128-d", S.encoder === "raw" ? "raw pooled embedding (reduced for the browser)" : "latent per rollout or prefix"], ["2 prototypes", "means of the support passes and failures"], ["0 params", "fit at inference"], [isNaN(au) ? "n/a" : fmt3(au), `held-out AUROC on ${B.task_id} under this support set`]]);
  $("#score-badges").innerHTML = `<span class="badge">support: ${supP.length} pass + ${supF.length} fail</span><span class="badge">${queries.length} queries</span><span class="badge">AUROC ${isNaN(au) ? "n/a" : fmt3(au)}</span>${B.held_out ? "" : `<span class="badge warn">training-pool task</span>`}`;
  // selected rollout + prefix slider
  const sel = rolls[idx.get(S.sel)], zs = E.z[idx.get(S.sel)];
  const T = E.prefix ? sel.n_steps : 1;
  const sl = $("#prefix-t"); sl.max = T; sl.disabled = !E.prefix; if (resetT || S.t > T) S.t = T; sl.value = S.t;
  $("#prefix-t-val").textContent = E.prefix ? `${S.t} / ${sel.n_steps} steps` : "(full rollouts only for this encoder)";
  const curve = E.prefix ? zs.map(z => score(z)) : [score(zs[0])];
  const pNow = E.prefix ? curve[S.t - 1] : curve[0];
  const thr = S.thr;
  let stopAt = null; if (E.prefix) for (let t = 0; t < curve.length; t++) if (curve[t] > thr || 1 - curve[t] > thr) { stopAt = t; break; }
  $("#sel-readout").innerHTML = `rollout <b>${sel.rid}</b> (${pass.get(sel.rid) ? "passed" : "failed"}${S.support.has(sel.rid) ? ", in support" : ""}) · P(pass) = <b>${fmt3(pNow)}</b>` +
    (E.prefix ? (stopAt != null ? ` · controller would stop after step ${stopAt + 1} predicting <b>${curve[stopAt] > thr ? "pass" : "fail"}</b> (threshold ${thr})` : ` · never crosses the ${thr} threshold`) : "");
  drawLatent(E, rolls, idx, pass, cp, cf, score, zs);
  drawCurve(curve, thr, E.prefix, stopAt);
  drawStrip(queries, qs, ql);
  drawSteps(sel);
  // theta sweep on this task's queries (prefix encoders only)
  if (E.prefix) {
    let nStop = 0, nCorrect = 0, stepsSaved = 0, stepsAll = 0;
    queries.forEach(r => { const zq = E.z[idx.get(r.rid)]; stepsAll += r.n_steps; let st = null;
      for (let t = 0; t < zq.length; t++) { const p = score(zq[t]); if (p > thr || 1 - p > thr) { st = t; if ((p > thr) === pass.get(r.rid)) nCorrect++; break; } }
      if (st != null) { nStop++; stepsSaved += r.n_steps - st - 1; } });
    $("#theta-readout").innerHTML = `At θ=${thr.toFixed(2)} on this task's ${queries.length} queries: the controller stops <b>${nStop}</b> (${Math.round(100 * nStop / Math.max(1, queries.length))}%), is right on <b>${nStop ? Math.round(100 * nCorrect / nStop) : 0}%</b> of them, and skips <b>${Math.round(100 * stepsSaved / Math.max(1, stepsAll))}%</b> of their generation steps.`;
  } else $("#theta-readout").textContent = "";
  // AUROC of every encoder under the same support set
  const rows = Object.entries(B.encoders).map(([k, e]) => { const zf = (i) => e.z[i][e.z[i].length - 1];
    const sp = [...S.support].filter(r => pass.get(r)).map(r => zf(idx.get(r))), sf = [...S.support].filter(r => !pass.get(r)).map(r => zf(idx.get(r)));
    const sc = scoreFn(sp.length ? mean(sp) : null, sf.length ? mean(sf) : null);
    return [k, auroc(queries.map(r => sc(zf(idx.get(r.rid)))), ql)]; });
  const NAME = { prefix: "Precog-RM prefix-trained", full: "Precog-RM generalist", raw: "raw space, nearest centroid" };
  $("#auroc-table").innerHTML = `<table class="mini"><tr><th>encoder</th><th>AUROC</th></tr>${rows.map(([k, a]) => `<tr class="${k === S.encoder ? "on" : ""}"><td>${NAME[k] || k}</td><td>${isNaN(a) ? "n/a" : fmt3(a)}</td></tr>`).join("")}</table>`;
}

function drawLatent(E, rolls, idx, pass, cp, cf, score, zs) {
  const svg = d3.select("#latent"); svg.selectAll("*").remove();
  const W = svg.node().clientWidth || 600, H = svg.node().clientHeight || 460, m = 28;
  const pts = rolls.map(r => { const [px, py] = proj(zFinal(idx.get(r.rid))); return { rid: r.rid, x: px, y: py, pass: pass.get(r.rid), sup: S.support.has(r.rid) }; });
  const trail = E.prefix ? zs.map(z => proj(z)) : [];
  // fit the view to the full-rollout cloud; prefix trails may wander outside and are clipped
  const x = d3.scaleLinear().domain(pad(d3.extent(pts, p => p.x))).range([m, W - m]), y = d3.scaleLinear().domain(pad(d3.extent(pts, p => p.y))).range([H - m, m]);
  svg.append("clipPath").attr("id", "latent-clip").append("rect").attr("x", 0).attr("y", 0).attr("width", W).attr("height", H);
  svg.attr("clip-path", "url(#latent-clip)");
  svg.append("text").attr("x", m).attr("y", H - 8).attr("font-size", 11).attr("fill", "#6b6b6b").text((S.encoder === "raw" ? `PCA of the raw action–observation space (mean-pooled text-embedding-3-large, ${E.z[0][0].length} PCs)` : `PCA of the ${E.checkpoint.replace(".pt", "")} latent (128-d)`) + `, fitted on this task's ${rolls.length} full rollouts · explains ${Math.round((E.pca.explained[0] + E.pca.explained[1]) * 100)}% of variance`);
  // trail of the selected rollout's prefixes
  if (trail.length > 1) {
    svg.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1]))(trail.slice(0, S.t))).attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 1.5).attr("stroke-opacity", 0.6);
    svg.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1]))(trail.slice(S.t - 1))).attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 1).attr("stroke-dasharray", "2,3").attr("stroke-opacity", 0.35);
    svg.append("g").selectAll("circle").data(trail.slice(0, S.t - 1)).join("circle").attr("cx", d => x(d[0])).attr("cy", d => y(d[1])).attr("r", 2).attr("fill", "#333").attr("fill-opacity", 0.5);
  }
  // prototypes + distance lines from the current (prefix) point
  const cur = E.prefix ? proj(zs[S.t - 1]) : proj(zs[0]);
  const protos = [[cp, PASS, "pass prototype"], [cf, FAIL, "fail prototype"]].filter(p => p[0]);
  protos.forEach(([c, col, name]) => {
    const [px, py] = proj(c);
    svg.append("line").attr("x1", x(cur[0])).attr("y1", y(cur[1])).attr("x2", x(px)).attr("y2", y(py)).attr("stroke", col).attr("stroke-width", 1).attr("stroke-dasharray", "4,3");
    svg.append("path").attr("d", d3.symbol(d3.symbolStar, 220)()).attr("transform", `translate(${x(px)},${y(py)})`).attr("fill", col).attr("stroke", "#fff").attr("stroke-width", 1.5).append("title").text(name);
  });
  // rollouts
  const g = svg.append("g");
  g.selectAll("circle").data(pts).join("circle").attr("cx", d => x(d.x)).attr("cy", d => y(d.y)).attr("r", d => d.rid === S.sel ? 7 : 5)
    .attr("fill", d => d.sup ? (d.pass ? PASS : FAIL) : "#fff").attr("stroke", d => d.pass ? PASS : FAIL).attr("stroke-width", d => d.rid === S.sel ? 2.5 : 1.5)
    .style("cursor", "pointer")
    .on("click", (e, d) => { if (e.shiftKey) { toggleSupport(d.rid); } else selectRollout(d.rid); })
    .append("title").text(d => `rollout ${d.rid} · ${d.pass ? "pass" : "fail"} · ${d.sup ? "support" : `query, P(pass)=${fmt3(score(zFinal(idx.get(d.rid))))}`}\nclick: select · shift-click: toggle support`);
  // current prefix point of the selection
  svg.append("circle").attr("cx", x(cur[0])).attr("cy", y(cur[1])).attr("r", 7).attr("fill", "none").attr("stroke", "#000").attr("stroke-width", 2);
  // legend
  const lg = svg.append("g").attr("transform", `translate(${W - 190},${m - 10})`).attr("font-size", 11);
  [["filled = support", "#999", true], ["hollow = query", "#999", false]].forEach(([t, c, f], i) => { lg.append("circle").attr("cx", 6).attr("cy", i * 16 + 6).attr("r", 5).attr("fill", f ? c : "#fff").attr("stroke", c); lg.append("text").attr("x", 16).attr("y", i * 16 + 10).text(t); });
  lg.append("path").attr("d", d3.symbol(d3.symbolStar, 90)()).attr("transform", "translate(6,38)").attr("fill", "#999"); lg.append("text").attr("x", 16).attr("y", 42).text("prototype (mean of support)");
}
const pad = ([a, b]) => { const d = (b - a) || 1; return [a - 0.08 * d, b + 0.08 * d]; };

function drawCurve(curve, thr, isPrefix, stopAt) {
  const svg = d3.select("#curve"); svg.selectAll("*").remove();
  if (!isPrefix) { svg.append("text").attr("x", 10).attr("y", 24).attr("font-size", 12).attr("fill", "#6b6b6b").text("The generalist encoder was trained on complete rollouts only and is at chance on prefixes (Sec. 4.5); pick the prefix-trained encoder to see P(pass) evolve."); return; }
  const W = svg.node().clientWidth || 600, H = svg.node().clientHeight || 170, m = { l: 36, r: 12, t: 8, b: 24 };
  const x = d3.scaleLinear().domain([1, curve.length]).range([m.l, W - m.r]), y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
  svg.append("rect").attr("x", m.l).attr("y", y(1)).attr("width", W - m.l - m.r).attr("height", y(thr) - y(1)).attr("fill", PASS).attr("fill-opacity", 0.08);
  svg.append("rect").attr("x", m.l).attr("y", y(1 - thr)).attr("width", W - m.l - m.r).attr("height", y(0) - y(1 - thr)).attr("fill", FAIL).attr("fill-opacity", 0.08);
  [thr, 1 - thr].forEach(v => svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", "#999").attr("stroke-dasharray", "3,3"));
  svg.append("path").attr("d", d3.line().x((d, i) => x(i + 1)).y(d => y(d))(curve)).attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 1.8);
  svg.append("g").selectAll("circle").data(curve).join("circle").attr("cx", (d, i) => x(i + 1)).attr("cy", d => y(d)).attr("r", (d, i) => i === S.t - 1 ? 5 : 2.5).attr("fill", (d, i) => i === S.t - 1 ? "#000" : "#333");
  if (stopAt != null) svg.append("line").attr("x1", x(stopAt + 1)).attr("x2", x(stopAt + 1)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", "#000").attr("stroke-width", 1).attr("stroke-dasharray", "2,2");
  svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(Math.min(curve.length, 12)).tickFormat(d3.format("d"))).selectAll("text").style("font-size", "10px");
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3)).selectAll("text").style("font-size", "10px");
  svg.append("text").attr("x", W - m.r).attr("y", m.t + 10).attr("text-anchor", "end").attr("font-size", 11).attr("fill", "#6b6b6b").text(`P(pass | first t steps) · shaded = controller stops (threshold ${thr})`);
}

function drawStrip(queries, qs, ql) {
  const svg = d3.select("#strip"); svg.selectAll("*").remove();
  const W = svg.node().clientWidth || 320, H = svg.node().clientHeight || 150, m = { l: 10, r: 10, t: 10, b: 22 };
  const x = d3.scaleLinear().domain([0, 1]).range([m.l, W - m.r]);
  const nodes = queries.map((r, i) => ({ rid: r.rid, s: qs[i], pass: ql[i], x: x(qs[i]), y: H / 2 - 4 }));
  const sim = d3.forceSimulation(nodes).force("x", d3.forceX(d => x(d.s)).strength(1)).force("y", d3.forceY(H / 2 - 4).strength(0.1)).force("c", d3.forceCollide(5)).stop();
  for (let i = 0; i < 120; i++) sim.tick();
  svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5)).selectAll("text").style("font-size", "10px");
  svg.append("g").selectAll("circle").data(nodes).join("circle").attr("cx", d => d.x).attr("cy", d => Math.max(m.t + 4, Math.min(H - m.b - 6, d.y))).attr("r", d => d.rid === S.sel ? 6 : 4)
    .attr("fill", d => d.pass ? PASS : FAIL).attr("fill-opacity", 0.85).attr("stroke", d => d.rid === S.sel ? "#000" : "#fff").attr("stroke-width", d => d.rid === S.sel ? 2 : 0.8)
    .style("cursor", "pointer").on("click", (e, d) => selectRollout(d.rid)).append("title").text(d => `rollout ${d.rid} · ${d.pass ? "pass" : "fail"} · P(pass)=${fmt3(d.s)}`);
  svg.append("text").attr("x", m.l).attr("y", m.t).attr("font-size", 11).attr("fill", "#6b6b6b").text("P(pass) for full rollouts, coloured by true outcome");
}

function drawSteps(sel) {
  const el = $("#sel-steps");
  el.innerHTML = `<h3>Rollout ${sel.rid} · ${sel.n_steps} steps</h3>` + sel.steps.map((s, i) =>
    `<div class="step ${enc().prefix && i === S.t - 1 ? "cur" : ""}"><div class="t">step ${i + 1} · state ${s.c}</div><div class="pair"><div class="a">${esc(s.a.slice(0, 400))}${s.a.length > 400 ? "…" : ""}</div><div class="o">${esc(s.o.slice(0, 400))}${s.o.length > 400 ? "…" : ""}</div></div></div>`).join("");
  const cur = el.querySelector(".step.cur"), panel = el.closest(".panel");
  if (cur && panel) panel.scrollTop = Math.max(0, cur.offsetTop - panel.offsetTop - panel.clientHeight / 2);
}
function selectRollout(rid) { S.sel = rid; drawScoring(true); }
function toggleSupport(rid) {
  if (S.support.has(rid)) S.support.delete(rid); else S.support.add(rid);
  S.preset = "custom"; $("#sel-support").value = "custom";
  if (S.support.has(S.sel)) { /* selection may be in support: still fine to inspect */ }
  drawScoring(false);
}

init().catch(e => { setStatus("Failed to load data: " + e); console.error(e); });
})();
