/* Story page: autoplaying SVG loops cut from the opener's timeline (data-from/data-to), the GRPO loop,
   the scene rail, and redraws for the collapsed "look inside" panels. */
(() => {
const PASS = "#2a9d4a", FAIL = "#d13b3b", INK = "#1c1c1c", MUTED = "#8a8a86", ACT = "#2b5fb3";
let S = null;
const loops = [];
const icon = (g, name, x, y, size, color = INK, op = 1) => { const k = size / 24, el = g.append("g").attr("transform", `translate(${x - size / 2},${y - size / 2}) scale(${k})`).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("stroke-linecap", "round").attr("stroke-linejoin", "round").attr("opacity", op); el.html(window.LUCIDE[name] || ""); return el; };
const clamp01 = (x) => Math.max(0, Math.min(1, x)), sub = (q, a, b) => clamp01((q - a) / (b - a)), lerp = (a, b, t) => a + (b - a) * t;

// ---------- one GRPO group with the controller (scene 9) ----------
function drawGrpoLoop(svgNode, frac) {
  const W = 900, H = 440, svg = d3.select(svgNode); svg.selectAll("*").remove(); if (svgNode.tagName === "svg") svg.attr("viewBox", `0 0 ${W} ${H}`);
  const B = window.PRECOG_BUNDLE, E = B.encoders.prefix, idx = new Map(B.rollouts.map((r, i) => [r.rid, i]));
  const group = B.rollouts.filter(r => !S.supIds.has(r.rid)).slice(0, 16), thr = 0.8;
  const stops = group.map(r => { const z = E.z[idx.get(r.rid)]; for (let t = 3; t < z.length; t++) { const p = S.score(z[t]); if (p > thr || 1 - p > thr) return { t, pred: p > thr }; } return { t: null, pred: null }; });
  const Tmax = d3.max(group, r => r.n_steps), gen = sub(frac, 0, 0.72) * Tmax, hold = sub(frac, 0.78, 1);
  const x0 = 150, x1 = 640, laneY = (i) => 56 + i * 22, cellW = (x1 - x0) / Tmax;
  const g = svg.append("g");
  g.append("text").attr("x", 30).attr("y", 30).attr("font-size", 13).attr("font-weight", 700).attr("fill", INK).text("one GRPO group · 16 rollouts of one task");
  g.append("text").attr("x", x1).attr("y", 30).attr("text-anchor", "end").attr("font-size", 11).attr("fill", MUTED).text(`controller stops a rollout once P(pass) > ${thr} or < ${1 - thr}`);
  let genSteps = 0, fullSteps = 0, stopped = 0;
  group.forEach((r, i) => {
    const y = laneY(i), st = stops[i], runTo = st.t != null ? Math.min(st.t + 1, r.n_steps) : r.n_steps, shown = Math.min(gen, runTo);
    fullSteps += r.n_steps; genSteps += shown;
    icon(g, "bot", 118, y, 13, INK, 0.8); g.append("text").attr("x", 108).attr("y", y + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", MUTED).text(`rollout ${i + 1}`);
    g.append("rect").attr("x", x0).attr("y", y - 6).attr("width", r.n_steps * cellW).attr("height", 12).attr("rx", 3).attr("fill", "#eceae5");
    g.append("rect").attr("x", x0).attr("y", y - 6).attr("width", Math.max(0, shown * cellW)).attr("height", 12).attr("rx", 3).attr("fill", st.t != null && gen >= runTo ? (st.pred ? PASS : FAIL) : "#6b7280");
    if (st.t != null && gen >= runTo) { stopped++; const gx = x0 + runTo * cellW;
      g.append("line").attr("x1", gx).attr("x2", gx).attr("y1", y - 10).attr("y2", y + 10).attr("stroke", INK).attr("stroke-width", 2);
      icon(g, st.pred ? "circle-check" : "circle-x", gx + 12, y, 12, st.pred ? PASS : FAIL);
      g.append("text").attr("x", gx + 22).attr("y", y + 3.5).attr("font-size", 9).attr("fill", MUTED).text(`stopped · reward ${st.pred ? 1 : 0}`); }
    else if (gen >= r.n_steps) icon(g, r.reward >= 0.5 ? "circle-check" : "circle-x", x0 + r.n_steps * cellW + 12, y, 12, r.reward >= 0.5 ? PASS : FAIL, 0.9);
  });
  // compute meter
  const mx = 690, my = 56, saved = fullSteps ? 1 - genSteps / Math.max(1, Math.min(gen, Tmax) / Tmax * fullSteps) : 0;
  g.append("rect").attr("x", mx).attr("y", my).attr("width", 180).attr("height", 150).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df");
  g.append("text").attr("x", mx + 14).attr("y", my + 22).attr("font-size", 11).attr("fill", MUTED).text("generation so far");
  const bar = d3.scaleLinear().domain([0, 1]).range([0, 150]);
  g.append("rect").attr("x", mx + 14).attr("y", my + 34).attr("width", 150).attr("height", 14).attr("rx", 4).attr("fill", "#eceae5");
  g.append("rect").attr("x", mx + 14).attr("y", my + 34).attr("width", bar(fullSteps ? genSteps / fullSteps : 0)).attr("height", 14).attr("rx", 4).attr("fill", INK);
  g.append("rect").attr("x", mx + 14).attr("y", my + 58).attr("width", 150).attr("height", 14).attr("rx", 4).attr("fill", "#eceae5");
  g.append("rect").attr("x", mx + 14).attr("y", my + 58).attr("width", bar(Math.min(1, gen / Tmax))).attr("height", 14).attr("rx", 4).attr("fill", "#9a9a9a");
  g.append("text").attr("x", mx + 14).attr("y", my + 86).attr("font-size", 10).attr("fill", INK).text("with the controller");
  g.append("text").attr("x", mx + 14).attr("y", my + 100).attr("font-size", 10).attr("fill", MUTED).text("baseline GRPO");
  g.append("text").attr("x", mx + 14).attr("y", my + 132).attr("font-size", 12).attr("font-weight", 700).attr("fill", INK).text(`${stopped} of 16 stopped early`);
  if (hold > 0) { const t = g.append("g").attr("opacity", hold);
    t.append("text").attr("x", mx + 14).attr("y", my + 185).attr("font-size", 12).attr("fill", INK).attr("font-weight", 700).text("up to 47.3% fewer training FLOPs");
    t.append("text").attr("x", mx + 14).attr("y", my + 200).attr("font-size", 10).attr("fill", MUTED).text("pass@1 within one std of baseline (Table 3)"); }
  g.append("text").attr("x", x0).attr("y", 420).attr("font-size", 10).attr("fill", MUTED).text("steps generated →   grey = still running · green/red = stopped with the predicted reward");
}

// ---------- loop manager ----------
function makeLoop(svg) {
  const from = +svg.dataset.from, to = +svg.dataset.to, secs = +svg.dataset.seconds || 10, hold = +svg.dataset.hold || 1.5, kind = svg.dataset.kind || "scene";
  let start = null, raf = null, playing = false;
  const draw = (frac) => { if (kind.startsWith("o")) { if (window.PrecogOpening && window.PrecogOpening.data) window.PrecogOpening.draw(kind, svg, frac); return; }
    if (!S) return; if (kind === "grpo") drawGrpoLoop(svg, frac); else window.PrecogPipeline.renderScene(svg, S, from + (to - from) * frac); };
  const tick = (now) => { if (!playing) return; if (start == null) start = now; const e = ((now - start) / 1000) % (secs + hold); draw(Math.min(1, e / secs)); raf = requestAnimationFrame(tick); };
  const play = () => { if (playing) return; playing = true; start = null; raf = requestAnimationFrame(tick); };
  const stop = () => { playing = false; if (raf) cancelAnimationFrame(raf); };
  const io = new IntersectionObserver((es) => es.forEach(e => e.isIntersecting ? play() : stop()), { threshold: 0.25 });
  io.observe(svg);
  svg.addEventListener("click", () => playing ? stop() : play());
  return { draw: () => draw(0), svg };
}
document.querySelectorAll("svg.loop").forEach(svg => loops.push(makeLoop(svg)));
window.PrecogStory = { drawGrpoLoop, get scene() { return S; } };
document.addEventListener("precog:scene", (e) => { S = e.detail; loops.forEach(l => l.draw()); document.dispatchEvent(new CustomEvent("precog:story")); });
document.addEventListener("precog:opening", () => loops.forEach(l => l.draw()));
if (window.PrecogPipeline && window.PrecogPipeline.scene) { S = window.PrecogPipeline.scene; loops.forEach(l => l.draw()); }

// ---------- collapsed panels redraw when opened; scene rail ----------
document.querySelectorAll("details.inside").forEach(d => d.addEventListener("toggle", () => { if (d.open) setTimeout(() => window.dispatchEvent(new Event("resize")), 30); }));
const rail = document.querySelector(".rail");
if (rail) { const scenes = [...document.querySelectorAll("section.scene")];
  scenes.forEach((sc, i) => { const a = document.createElement("a"); a.href = "#" + sc.id; a.title = sc.dataset.title || sc.id; rail.appendChild(a); });
  const io = new IntersectionObserver((es) => es.forEach(e => { if (e.isIntersecting) rail.querySelectorAll("a").forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id)); }), { rootMargin: "-40% 0px -50% 0px" });
  scenes.forEach(sc => io.observe(sc)); }
})();
