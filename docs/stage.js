/* The pinned stage: one canvas, two crossfading layers, acts composed from the loop drawers.
   Scroll selects the act (the card crossing the middle of the viewport); the stage plays it, holds, and the
   card's conclusion lands when the loop does. Also: counting numbers, the progress lane, the drawer. */
(() => {
const $ = (s) => document.querySelector(s), $$ = (s) => [...document.querySelectorAll(s)];
const clamp01 = (x) => Math.max(0, Math.min(1, x)), sub = (q, a, b) => clamp01((q - a) / (b - a));
const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const stageSvg = $("#stage"); if (!stageSvg) return;
const layers = [d3.select(stageSvg).append("g").attr("class", "layer"), d3.select(stageSvg).append("g").attr("class", "layer").attr("opacity", 0)];
let curLayer = 0;
const O = () => window.PrecogOpening, P = () => window.PrecogPipeline, ST = () => window.PrecogStory;
const seg = (f, a, b) => sub(f, a, b);
const ACTS = {
  hero: { secs: 12, hold: 3, draw(g, f) { O().draw("o1", g, f); const D = O().data, r = D.pairFP.f, n = Math.min(12, r.n_steps), upto = Math.floor(sub(f, 0.05, 0.85) * n);
    const tok = window.PrecogTokens(r, upto) * (r.n_steps / Math.max(1, n)); const t = d3.select(g).append("g").attr("transform", "translate(640,40)");
    t.append("text").attr("font-size", 11).attr("fill", "#8a8a86").text("tokens so far"); t.append("text").attr("y", 30).attr("font-size", 26).attr("font-weight", 800).attr("fill", "#1c1c1c").text(d3.format(",")(Math.round(tok))); } },
  act1: { secs: 24, hold: 3, draw(g, f) { if (f < 0.42) O().draw("o1", g, seg(f, 0, 0.42)); else O().draw("o2", g, seg(f, 0.42, 1)); } },
  act2: { secs: 48, hold: 4, draw(g, f) { if (f < 0.22) O().draw("o3", g, seg(f, 0, 0.22)); else if (f < 0.42) O().draw("o4", g, seg(f, 0.22, 0.42)); else if (f < 0.66) O().draw("o5", g, seg(f, 0.42, 0.66)); else O().draw("o6", g, seg(f, 0.66, 1)); } },
  act3: { secs: 9, hold: 3, draw(g, f) { P().renderScene(g, ST().scene, 0.30 + 0.08 * f); } },
  act4: { secs: 10, hold: 3, draw(g, f) { O().draw("o4", g, f); if (f > 0.9) { const t = d3.select(g).append("g").attr("opacity", sub(f, 0.9, 1)).attr("transform", "translate(640,40)");
    t.append("rect").attr("width", 230).attr("height", 54).attr("rx", 10).attr("fill", "#fff").attr("stroke", "#e3e3df"); t.append("text").attr("x", 14).attr("y", 22).attr("font-size", 11).attr("fill", "#8a8a86").text("to fit one classifier for this task");
    t.append("text").attr("x", 14).attr("y", 44).attr("font-size", 15).attr("font-weight", 700).attr("fill", "#1c1c1c").text("≈ 33 labeled rollouts · reusable: 0"); } } },
  act5: { secs: 40, hold: 3, draw(g, f) { P().renderScene(g, ST().scene, 0.38 + 0.40 * f); } },
  act6: { secs: 16, hold: 3, draw(g, f) { P().renderScene(g, ST().scene, 0.78 + 0.22 * f); } },
  act7: { secs: 11, hold: 4, draw(g, f) { ST().drawGrpoLoop(g, f); } },
  act8: { secs: 5, hold: 10, draw(g, f) { O().draw("o1", g, 1); } },
};
// ---------- playback ----------
let current = null, start = null, raf = null, landed = new Set();
function draw(id, f) { try { ACTS[id].draw(layers[curLayer].node(), f); } catch (e) { console.error(id, e); } }
function tick(now) {
  if (!current) return; if (start == null) start = now;
  const A = ACTS[current], e = (now - start) / 1000, cycle = A.secs + A.hold, f = Math.min(1, (e % cycle) / A.secs);
  draw(current, reduced ? 1 : f);
  if (!landed.has(current) && (f >= 1 || reduced)) { landed.add(current); const card = $(`.act[data-act="${current}"]`); if (card) card.classList.add("landed"); }
  raf = requestAnimationFrame(tick);
}
function show(id) {
  if (!ACTS[id] || id === current || !ready()) return;
  const prev = curLayer; curLayer = 1 - curLayer;
  layers[prev].interrupt().transition().duration(700).attr("opacity", 0);
  layers[curLayer].interrupt().attr("opacity", 0).transition().duration(700).attr("opacity", 1);
  current = id; start = null; if (raf) cancelAnimationFrame(raf); raf = requestAnimationFrame(tick);
  $$(".act").forEach(a => a.classList.toggle("active", a.dataset.act === id));
  counters(id);
}
const ready = () => !!(O() && O().data && ST() && ST().scene && P());
let pending = null;
const io = new IntersectionObserver((es) => { es.forEach(e => { if (e.isIntersecting) { pending = e.target.dataset.act; show(pending); } }); }, { rootMargin: "-42% 0px -42% 0px", threshold: 0 });
$$(".act").forEach(a => io.observe(a));
const tryStart = () => { if (ready()) show(pending || "hero"); };
document.addEventListener("precog:opening", tryStart); document.addEventListener("precog:story", tryStart); setTimeout(tryStart, 50);
// ---------- counting numbers ----------
const counted = new Set();
function counters(id) {
  $$(`.act[data-act="${id}"] .num[data-n]`).forEach(el => { if (counted.has(el)) return; counted.add(el);
    const n = +el.dataset.n, fmt = el.dataset.fmt || "int", f = (v) => fmt === "pct" ? d3.format(".1f")(v) + "%" : fmt === "dec" ? d3.format(".2f")(v) : fmt === "dec3" ? d3.format(".3f")(v) : d3.format(",")(Math.round(v));
    if (reduced) { el.textContent = f(n); return; }
    const t0 = performance.now(), dur = 1300; const step = (now) => { const q = Math.min(1, (now - t0) / dur), v = n * d3.easeCubicOut(q); el.textContent = f(v); if (q < 1) requestAnimationFrame(step); }; requestAnimationFrame(step); });
}
// ---------- progress lane ----------
const prog = $("#progress");
if (prog) { const acts = $$(".act"), svg = d3.select(prog), W = 1000, lane = svg.append("line").attr("x1", 40).attr("x2", W - 40).attr("y1", 14).attr("y2", 14).attr("stroke", "#d6d6d0").attr("stroke-width", 2);
  svg.attr("viewBox", `0 0 ${W} 28`).attr("preserveAspectRatio", "none");
  acts.forEach((a, i) => svg.append("circle").attr("cx", 40 + (W - 80) * i / Math.max(1, acts.length - 1)).attr("cy", 14).attr("r", 3).attr("fill", "#c9c9c3"));
  const done = svg.append("line").attr("x1", 40).attr("x2", 40).attr("y1", 14).attr("y2", 14).attr("stroke", "#2b5fb3").attr("stroke-width", 2);
  const sprite = svg.append("g"); sprite.append("circle").attr("r", 9).attr("fill", "#fff").attr("stroke", "#1c1c1c").attr("stroke-width", 1.5);
  const ic = sprite.append("g").attr("transform", "translate(-6,-6) scale(0.5)").attr("fill", "none").attr("stroke", "#1c1c1c").attr("stroke-width", 2).attr("stroke-linecap", "round").attr("stroke-linejoin", "round"); ic.html(window.LUCIDE.bot);
  const check = svg.append("g").attr("transform", `translate(${W - 40},14)`).attr("opacity", 0); const ck = check.append("g").attr("transform", "translate(-8,-8) scale(0.67)").attr("fill", "none").attr("stroke", "#2a9d4a").attr("stroke-width", 2.5); ck.html(window.LUCIDE["circle-check"]);
  const update = () => { const max = document.documentElement.scrollHeight - window.innerHeight, q = max > 0 ? clamp01(window.scrollY / max) : 0, x = 40 + (W - 80) * q; sprite.attr("transform", `translate(${x},14)`); done.attr("x2", x); check.attr("opacity", q > 0.985 ? 1 : 0); };
  window.addEventListener("scroll", update, { passive: true }); window.addEventListener("resize", update); update(); }
// ---------- drawer ----------
const drawer = $("#drawer");
if (drawer) { const open = (panel) => { drawer.classList.add("open"); $$(".drawer-panel").forEach(p => p.classList.toggle("on", p.dataset.panel === panel)); document.body.classList.add("drawer-open"); setTimeout(() => window.dispatchEvent(new Event("resize")), 60); };
  const close = () => { drawer.classList.remove("open"); document.body.classList.remove("drawer-open"); };
  $$("button.inside").forEach(b => b.addEventListener("click", () => open(b.dataset.panel)));
  $("#drawer-close").addEventListener("click", close); document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); }); }
window.PrecogStage = { show, ACTS };
})();
