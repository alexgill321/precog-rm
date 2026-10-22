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
  a2: (g, f) => O().draw("a2", g, f),
  a3: (g, f) => O().draw("a3", g, f),
  a4: (g, f) => O().draw("a4", g, f),
  a5: (g, f) => O().draw("a5", g, f),
  a6: (g, f) => O().draw("a6", g, f),
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
// Each beat plays its slice of the scene once and holds the last frame. Moving forward fast-forwards to the new beat's start,
// then plays it; moving back rewinds the canvas to the earlier beat's last frame and holds there. Acts whose first frame
// equals the previous act's last frame (SEAMS) are crossed by playing or rewinding through the seam, so there is no fade;
// any other change of scene crossfades. A move is a plan of segments: tween (eased catch-up or rewind), play (the beat's
// clip at its own speed), swap (change scene, with or without a fade).
const BEAT_EL = $$(".beat"), ORDER = [...new Set(BEAT_EL.map(b => b.dataset.scene))];
const SEAMS = new Set(["a1>a2", "a2>a3", "a3>a4", "a4>a5", "a5>a6"]);
let scene = null, f = 0, beat = null, plan = [], cseg = null, t0 = 0, idleTick = 0;
const fwdDur = (d) => Math.min(1.1, 0.35 + 2.2 * Math.abs(d)), backDur = (d) => Math.min(1.4, 0.4 + 2.5 * Math.abs(d));
function crossfade() { const prev = cur; cur = 1 - cur; layers[prev].interrupt().transition().duration(450).attr("opacity", 0); layers[cur].interrupt().attr("opacity", 0).transition().duration(450).attr("opacity", 1); }
function draw() { try { layers[cur].selectAll("*").remove(); SCENES[scene](layers[cur].node(), f); } catch (e) { console.error(scene, e); } }
const tween = (to, durFn) => ({ kind: "tween", to, durFn }), play = (b) => ({ kind: "play", from: b.f0, to: b.f1, dur: b.f1 === b.f0 ? 0 : b.secs, el: b.el }), swap = (sc, at, fade) => ({ kind: "swap", sc, at, fade });
function planTo(b, forward, far) {
  const P = [], eps = 1e-4;
  // a long jump (a reload mid-page, a fast fling) cuts straight to the target instead of racing through everything between
  if (far) { if (b.scene !== scene) P.push(swap(b.scene, forward ? b.f0 : b.f1, true)); else P.push(tween(forward ? b.f0 : b.f1, fwdDur)); if (forward) P.push(play(b)); return P; }
  if (b.scene === scene) {
    if (forward) { if (Math.abs(f - b.f0) > eps) P.push(tween(b.f0, fwdDur)); P.push(play(b)); }
    else if (Math.abs(f - b.f1) > eps) P.push(tween(b.f1, backDur));
    return P; }
  const ci = ORDER.indexOf(scene), ti = ORDER.indexOf(b.scene);
  if (forward) {
    if (scene && ti === ci + 1 && SEAMS.has(`${scene}>${b.scene}`)) { if (f < 1 - eps) P.push(tween(1, fwdDur)); P.push(swap(b.scene, 0, false)); if (b.f0 > eps) P.push(tween(b.f0, fwdDur)); }
    else P.push(swap(b.scene, b.f0, true));
    P.push(play(b));
  } else {
    if (ti === ci - 1 && SEAMS.has(`${b.scene}>${scene}`)) { if (f > eps) P.push(tween(0, backDur)); P.push(swap(b.scene, 1, false)); if (b.f1 < 1 - eps) P.push(tween(b.f1, backDur)); }
    else P.push(swap(b.scene, b.f1, true));
  }
  return P; }
function nextSeg(now) {
  cseg = plan.shift() || null;
  while (cseg && cseg.kind === "swap") { scene = cseg.sc; f = cseg.at; if (cseg.fade) crossfade(); draw(); cseg = plan.shift() || null; }
  if (cseg && cseg.kind === "tween") { cseg.from = f; cseg.dur = cseg.durFn(cseg.to - f); }
  t0 = now; }
function activate(el) {
  if (!ready()) { pending = el; return; }
  if (beat && beat.el === el) return;
  const b = { el, scene: el.dataset.scene, f0: +el.dataset.f0, f1: +el.dataset.f1, secs: +(el.dataset.secs || 3) };
  const forward = !beat || BEAT_EL.indexOf(el) > BEAT_EL.indexOf(beat.el), far = !!beat && Math.abs(BEAT_EL.indexOf(el) - BEAT_EL.indexOf(beat.el)) > 3;
  beat = b; BEAT_EL.forEach(x => x.classList.toggle("active", x === el)); counters(el);
  laneTarget = laneFor(el.classList.contains("wide")); if (laneCur == null) { laneCur = laneTarget; fit(); }
  if (reduced) { if (b.scene !== scene) { scene = b.scene; crossfade(); } f = b.f1; plan = []; cseg = null; draw(); el.classList.add("landed"); return; }
  plan = planTo(b, forward, far); nextSeg(performance.now());
  if (!forward) el.classList.add("landed");   // going back lands on the beat's last frame straight away
}
function tick(now) {
  if (laneTarget != null && laneCur != null && Math.abs(laneTarget - laneCur) > 0.5) { laneCur += (laneTarget - laneCur) * 0.12; fit(); }
  if (scene && cseg) {
    const q = cseg.dur > 0 ? clamp01((now - t0) / 1000 / cseg.dur) : 1;
    f = lerp(cseg.from, cseg.to, cseg.kind === "tween" ? d3.easeCubicInOut(q) : q);
    draw();
    if (q >= 1) { if (cseg.kind === "play") cseg.el.classList.add("landed"); nextSeg(now); }
  } else if (scene && ++idleTick % 30 === 0) draw();   // holding: redraw now and then, in case data a scene uses arrived late
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
window.PrecogStage = { activate, SCENES, _tick: tick, get state() { return { scene, f, beat: beat && BEAT_EL.indexOf(beat.el), seg: cseg && cseg.kind, queued: plan.length }; } };
})();
