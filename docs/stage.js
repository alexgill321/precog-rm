/* Full-page scrollytelling engine. A borderless canvas stays pinned behind the text; the text is a column of short
   beats. Each beat names a scene and a slice [f0, f1] of it: the slice plays, holds, and replays (crossfaded) until the
   reader moves on. Moving to another beat of the same scene scrubs the canvas from where it is to the new slice's start,
   so going forward fast-forwards and going back rewinds; a different scene crossfades in. */
(() => {
const $ = (s) => document.querySelector(s), $$ = (s) => [...document.querySelectorAll(s)];
const clamp01 = (x) => Math.max(0, Math.min(1, x)), sub = (q, a, b) => clamp01((q - a) / (b - a)), lerp = (a, b, t) => a + (b - a) * t;
const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const stageSvg = $("#stage"); if (!stageSvg) return;
const SW = 900, SH = 440;
const O = () => window.PrecogOpening, P = () => window.PrecogPipeline, ST = () => window.PrecogStory;
const seg = (f, a, b) => sub(f, a, b);
const SCENES = {
  a1: (g, f) => O().draw("a1", g, f),
  act2: (g, f) => { if (f < 0.2) O().draw("o2", g, seg(f, 0, 0.2)); else if (f < 0.37) O().draw("o3", g, seg(f, 0.2, 0.37)); else if (f < 0.52) O().draw("o4", g, seg(f, 0.37, 0.52)); else if (f < 0.72) O().draw("o5", g, seg(f, 0.52, 0.72)); else O().draw("o6", g, seg(f, 0.72, 1)); },
  act3: (g, f) => P().renderScene(g, ST().scene, 0.30 + 0.08 * f),
  act4: (g, f) => { O().draw("o4", g, f); if (f > 0.9) { const t = d3.select(g).append("g").attr("opacity", sub(f, 0.9, 1)).attr("transform", "translate(640,40)");
    t.append("rect").attr("width", 230).attr("height", 54).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df"); t.append("text").attr("x", 14).attr("y", 22).attr("font-size", 11).attr("fill", "#8a8a86").text("to fit one classifier for this task");
    t.append("text").attr("x", 14).attr("y", 44).attr("font-size", 15).attr("font-weight", 700).attr("fill", "#1c1c1c").text("≈ 33 labeled rollouts · reusable: 0"); } },
  act5: (g, f) => P().renderScene(g, ST().scene, 0.38 + 0.40 * f),
  act6: (g, f) => P().renderScene(g, ST().scene, 0.78 + 0.22 * f),
  act7: (g, f) => ST().drawGrpoLoop(g, f),
  act8: (g, f) => O().draw("o2", g, 1),
};
// ---------- canvas: the scene's 900x440 frame placed in the free area of the viewport; the text lane can widen smoothly ----------
const root = d3.select(stageSvg), layers = [root.append("g").attr("class", "layer"), root.append("g").attr("class", "layer").attr("opacity", 0)];
let cur = 0, laneCur = null, laneTarget = null;
const laneFor = (wide) => { const vw = window.innerWidth; return vw < 900 ? 0 : wide ? Math.min(600, vw * 0.44) : Math.min(440, vw * 0.34); };
function fit() {
  const vw = window.innerWidth, vh = window.innerHeight, small = vw < 900, lane = laneCur ?? laneFor(false);
  document.documentElement.style.setProperty("--lane", `${lane}px`);
  const top = 44, bottom = small ? vh * 0.42 : 24, right = small ? 12 : 28, leftPad = small ? 12 : lane;
  const aw = vw - leftPad - right, ah = vh - top - bottom, sc = Math.min(aw / SW, ah / SH);
  const ox = leftPad + (aw - SW * sc) / 2, oy = top + (ah - SH * sc) / 2;
  root.attr("viewBox", `${-ox / sc} ${-oy / sc} ${vw / sc} ${vh / sc}`).attr("preserveAspectRatio", "none");
}
window.addEventListener("resize", () => { laneCur = laneTarget = laneFor(beat && beat.el.classList.contains("wide")); fit(); }); fit();
// ---------- playback ----------
let scene = null, f = 0, beat = null, mode = "idle", t0 = 0, tw = null, landedOnce = new Set();
function crossfade() { const prev = cur; cur = 1 - cur; layers[prev].interrupt().transition().duration(450).attr("opacity", 0); layers[cur].interrupt().attr("opacity", 0).transition().duration(450).attr("opacity", 1); }
function draw() { try { layers[cur].selectAll("*").remove(); SCENES[scene](layers[cur].node(), f); } catch (e) { console.error(scene, e); } }
function activate(el) {
  if (!ready()) { pending = el; return; }
  const b = { el, scene: el.dataset.scene, f0: +el.dataset.f0, f1: +el.dataset.f1, secs: +(el.dataset.secs || 3), hold: +(el.dataset.hold || 1.6) };
  if (beat && beat.el === el) return;
  beat = b; $$(".beat").forEach(x => x.classList.toggle("active", x === el)); counters(el);
  laneTarget = laneFor(el.classList.contains("wide")); if (laneCur == null) { laneCur = laneTarget; fit(); }
  const now = performance.now();
  if (b.scene !== scene) { scene = b.scene; f = b.f0; crossfade(); mode = "play"; t0 = now; }
  else if (Math.abs(f - b.f0) < 1e-4) { mode = "play"; t0 = now; }
  else { mode = "tween"; tw = { from: f, to: b.f0, dur: Math.min(1.1, 0.35 + 2.2 * Math.abs(b.f0 - f)) }; t0 = now; }
  if (reduced) { f = b.f1; mode = "still"; draw(); el.classList.add("landed"); }
}
function tick(now) {
  if (laneTarget != null && laneCur != null && Math.abs(laneTarget - laneCur) > 0.5) { laneCur += (laneTarget - laneCur) * 0.12; fit(); }
  if (beat && scene && mode !== "still") {
    const e = (now - t0) / 1000;
    if (mode === "tween") { const q = clamp01(e / tw.dur); f = lerp(tw.from, tw.to, d3.easeCubicInOut(q)); if (q >= 1) { mode = "play"; t0 = now; } }
    else if (mode === "play") { f = beat.f1 === beat.f0 ? beat.f1 : lerp(beat.f0, beat.f1, clamp01(e / beat.secs));
      if (e >= beat.secs && !landedOnce.has(beat.el)) { landedOnce.add(beat.el); beat.el.classList.add("landed"); }
      if (e >= beat.secs + beat.hold && beat.f1 !== beat.f0) { crossfade(); f = beat.f0; t0 = now; } }
    draw();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
const ready = () => !!(O() && O().data && ST() && ST().scene && P());
let pending = null;
const io = new IntersectionObserver((es) => es.forEach(e => { if (e.isIntersecting) activate(e.target); }), { rootMargin: "-46% 0px -46% 0px", threshold: 0 });
$$(".beat").forEach(b => io.observe(b));
const tryStart = () => { if (!ready()) return; const mid = window.innerHeight / 2; const el = pending || $$(".beat").find(b => { const r = b.getBoundingClientRect(); return r.top <= mid && r.bottom >= mid; }) || $(".beat"); pending = null; activate(el); };
document.addEventListener("precog:opening", tryStart); document.addEventListener("precog:story", tryStart); setTimeout(tryStart, 60);
// ---------- counting numbers ----------
const counted = new Set();
function counters(el) {
  el.querySelectorAll(".num[data-n]").forEach(nm => { if (counted.has(nm)) return; counted.add(nm);
    const n = +nm.dataset.n, fmt = nm.dataset.fmt || "int", fm = (v) => fmt === "pct" ? d3.format(".1f")(v) + "%" : fmt === "dec" ? d3.format(".2f")(v) : d3.format(",")(Math.round(v));
    if (reduced) { nm.textContent = fm(n); return; }
    const s0 = performance.now(), dur = 1300, step = (now) => { const q = Math.min(1, (now - s0) / dur); nm.textContent = fm(n * d3.easeCubicOut(q)); if (q < 1) requestAnimationFrame(step); }; requestAnimationFrame(step); });
}
// ---------- progress lane ----------
const prog = $("#progress");
if (prog) { const W = 1000, svg = d3.select(prog).attr("viewBox", `0 0 ${W} 28`).attr("preserveAspectRatio", "none"), heads = $$(".beat.head");
  svg.append("line").attr("x1", 40).attr("x2", W - 40).attr("y1", 14).attr("y2", 14).attr("stroke", "#d6d6d0").attr("stroke-width", 2);
  const done = svg.append("line").attr("x1", 40).attr("x2", 40).attr("y1", 14).attr("y2", 14).attr("stroke", "#2b5fb3").attr("stroke-width", 2);
  const sprite = svg.append("g"); sprite.append("circle").attr("r", 9).attr("fill", "#fff").attr("stroke", "#1c1c1c").attr("stroke-width", 1.5);
  sprite.append("g").attr("transform", "translate(-6,-6) scale(0.5)").attr("fill", "none").attr("stroke", "#1c1c1c").attr("stroke-width", 2).attr("stroke-linecap", "round").attr("stroke-linejoin", "round").html(window.LUCIDE.bot);
  const check = svg.append("g").attr("transform", `translate(${W - 40},14)`).attr("opacity", 0); check.append("g").attr("transform", "translate(-8,-8) scale(0.67)").attr("fill", "none").attr("stroke", "#2a9d4a").attr("stroke-width", 2.5).html(window.LUCIDE["circle-check"]);
  const ticks = svg.append("g");
  const update = () => { const max = document.documentElement.scrollHeight - window.innerHeight, q = max > 0 ? clamp01(window.scrollY / max) : 0, x = 40 + (W - 80) * q;
    sprite.attr("transform", `translate(${x},14)`); done.attr("x2", x); check.attr("opacity", q > 0.985 ? 1 : 0);
    ticks.selectAll("*").remove(); heads.forEach(h => { const tq = max > 0 ? clamp01((h.offsetTop - window.innerHeight * 0.5) / max) : 0; ticks.append("circle").attr("cx", 40 + (W - 80) * tq).attr("cy", 14).attr("r", 3).attr("fill", tq <= q ? "#2b5fb3" : "#c9c9c3"); }); };
  window.addEventListener("scroll", update, { passive: true }); window.addEventListener("resize", update); setTimeout(update, 300); }
// ---------- drawer ----------
const drawer = $("#drawer");
if (drawer) { const open = (panel) => { drawer.classList.add("open"); $$(".drawer-panel").forEach(p => p.classList.toggle("on", p.dataset.panel === panel)); setTimeout(() => window.dispatchEvent(new Event("resize")), 60); };
  const close = () => drawer.classList.remove("open");
  $$("button.inside").forEach(b => b.addEventListener("click", () => open(b.dataset.panel)));
  $("#drawer-close").addEventListener("click", close); document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); }); }
window.PrecogStage = { activate, SCENES };
})();
