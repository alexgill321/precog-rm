/* Result charts for chapters 2, 4, 5, 6, read from docs/results.json (scripts/build_web_results.py). */
(() => {
const $ = (s) => document.querySelector(s);
const BENCH = { tau2: "Tau2-Bench", bfclv3: "BFCLv3", agentclinic: "AgentClinic" };
const MODEL = { deepseek: "DeepSeek-V3.2", qwen: "Qwen3-80B", gpt5mini: "GPT-5-mini" };
const METH = { NC: "nearest centroid", RBF: "RBF", FT1: "fastText 1-gram", FT2: "fastText 2-gram", FT5: "fastText 5-gram", PN: "Precog-RM" };
const BCOL = { tau2: "#0072B2", bfclv3: "#E69F00", agentclinic: "#009E73" };
const VCOL = { baseline: "#6b7280", rolling: "#0072B2", permanent: "#E69F00" };
const fmt3 = d3.format(".3f"), fmt1 = d3.format(".1f");
let R = null, grpoOn = null;

async function init() {
  R = await (await fetch("results.json?v=5")).json();
  document.querySelectorAll("input[name=t1]").forEach(el => el.onchange = () => drawTable1());
  drawTable1(); drawFewshot(8); drawGen($("#gen-models"), R.generalization.models, "Cross-model: rows = policies in the training pool, columns = evaluated policy (Table 2)", m => ({ d: "DeepSeek", q: "Qwen", g: "GPT-5-mini" })[m] || m, c => MODEL[c]);
  drawGen($("#gen-bench"), R.generalization.benchmarks, "Cross-benchmark: rows = benchmarks in the training pool, columns = evaluated benchmark (Table 13)", r => r.replace(":g+q+d", "").replace(":all", "").split("+").map(b => BENCH[b]).join(" + "), c => BENCH[c]);
  drawGating(); drawGrpo();
  $("#k-slider").oninput = (e) => { const k = [1, 2, 4, 8][+e.target.value]; $("#k-val").textContent = k; drawFewshot(k); document.dispatchEvent(new CustomEvent("precog:k", { detail: k })); };
  numbers();
  window.addEventListener("resize", () => { drawTable1(); drawFewshot(+$("#k-val").textContent); drawGating(); drawGrpo(); });
}
const svgIn = (el, h) => { el.innerHTML = ""; const w = el.clientWidth || 600; return [d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("width", "100%"), w]; };
const title = (svg, w, t) => svg.append("text").attr("x", 0).attr("y", 12).attr("font-size", 12).attr("fill", "#444").text(t);

function drawTable1() {
  const mode = document.querySelector("input[name=t1]:checked").value;
  const [svg, w] = svgIn($("#table1"), 190);
  const bs = ["tau2", "bfclv3", "agentclinic"], ms = ["deepseek", "qwen", "gpt5mini"];
  const x = d3.scaleBand().domain(ms).range([130, w - 10]).padding(0.06), y = d3.scaleBand().domain(bs).range([30, 180]).padding(0.08);
  const col = d3.scaleSequential(d3.interpolateBlues).domain([0.7, 0.97]);
  title(svg, w, mode === "PN" ? "Precog-RM per-benchmark specialist: held-out-task AUROC, same folds as support, no per-task fitting" : "Best per-task discriminator: held-out-task AUROC under within-task 3-fold CV (about 33 labeled rollouts per fit)");
  R.table1.rows.forEach(r => {
    const meth = mode === "PN" ? "PN" : r.best_baseline, v = r.methods[meth];
    const g = svg.append("g").attr("transform", `translate(${x(r.model)},${y(r.benchmark)})`);
    g.append("rect").attr("width", x.bandwidth()).attr("height", y.bandwidth()).attr("rx", 6).attr("fill", col(v[0]));
    g.append("text").attr("x", x.bandwidth() / 2).attr("y", y.bandwidth() / 2 - 2).attr("text-anchor", "middle").attr("font-size", 16).attr("font-weight", 700).attr("fill", v[0] > 0.86 ? "#fff" : "#1c1c1c").text(fmt3(v[0]));
    g.append("text").attr("x", x.bandwidth() / 2).attr("y", y.bandwidth() / 2 + 14).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", v[0] > 0.86 ? "#e8eefc" : "#444").text(`±${fmt3(v[1])} · ${METH[meth]} · N=${r.n_tasks}`);
  });
  ms.forEach(m => svg.append("text").attr("x", x(m) + x.bandwidth() / 2).attr("y", 26).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#444").text(MODEL[m]));
  bs.forEach(b => svg.append("text").attr("x", 124).attr("y", y(b) + y.bandwidth() / 2 + 4).attr("text-anchor", "end").attr("font-size", 11).attr("fill", "#444").text(BENCH[b]));
}

function drawFewshot(k) {
  const [svg, w] = svgIn($("#fewshot"), 240);
  const m = { l: 40, r: 12, t: 24, b: 30 }, ks = [1, 2, 4, 8];
  const x = d3.scaleLog().domain([1, 8]).range([m.l, w - m.r]), y = d3.scaleLinear().domain([0.6, 0.95]).range([240 - m.b, m.t]);
  title(svg, w, "Few-shot AUROC vs. support rollouts per class (Figure 3): Precog-RM solid, best per-task baseline dashed");
  svg.append("g").attr("transform", `translate(0,${240 - m.b})`).call(d3.axisBottom(x).tickValues(ks).tickFormat(d3.format("d"))).selectAll("text").style("font-size", "10px");
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)).selectAll("text").style("font-size", "10px");
  svg.append("line").attr("x1", x(k)).attr("x2", x(k)).attr("y1", m.t).attr("y2", 240 - m.b).attr("stroke", "#999").attr("stroke-dasharray", "3,3");
  Object.entries(R.fewshot).forEach(([b, byK]) => {
    const pn = ks.map(kk => [kk, byK[String(kk)].ProtoNet]), base = ks.map(kk => [kk, d3.max(["NC", "RBF", "FT1"], mm => byK[String(kk)][mm])]);
    const line = d3.line().x(d => x(d[0])).y(d => y(d[1]));
    svg.append("path").attr("d", line(base)).attr("fill", "none").attr("stroke", BCOL[b]).attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3");
    svg.append("path").attr("d", line(pn)).attr("fill", "none").attr("stroke", BCOL[b]).attr("stroke-width", 2.5);
    svg.append("circle").attr("cx", x(k)).attr("cy", y(byK[String(k)].ProtoNet)).attr("r", 4).attr("fill", BCOL[b]);
    svg.append("text").attr("x", x(8) + 4).attr("y", y(byK["8"].ProtoNet) + 4).attr("font-size", 10).attr("fill", BCOL[b]).text(BENCH[b]);
  });
  svg.append("text").attr("x", w - m.r).attr("y", 240 - 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", "#6b6b6b").text("k per class (log)");
}

function drawGen(el, M, t, rowLabel, colLabel) {
  const [svg, w] = svgIn(el, 30 + 26 * M.rows.length + 10);
  const x = d3.scaleBand().domain(M.cols).range([210, w - 10]).padding(0.05), y = d3.scaleBand().domain(M.rows).range([28, 28 + 26 * M.rows.length]).padding(0.1);
  const col = d3.scaleSequential(d3.interpolateBlues).domain([0.6, 0.95]);
  title(svg, w, t);
  M.cols.forEach(c => svg.append("text").attr("x", x(c) + x.bandwidth() / 2).attr("y", 24).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#444").text(colLabel(c)));
  M.rows.forEach((r, i) => {
    svg.append("text").attr("x", 204).attr("y", y(r) + y.bandwidth() / 2 + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", "#444").text(rowLabel(r));
    M.cols.forEach((c, j) => {
      const v = M.matrix[i][j];
      const inPool = M === R.generalization.models ? r.split("+").includes({ deepseek: "d", qwen: "q", gpt5mini: "g" }[c]) : r.split(":")[0].split("+").includes(c);
      svg.append("rect").attr("x", x(c)).attr("y", y(r)).attr("width", x.bandwidth()).attr("height", y.bandwidth()).attr("rx", 4).attr("fill", col(v)).attr("stroke", inPool ? "none" : "#d13b3b").attr("stroke-width", inPool ? 0 : 1.5).attr("stroke-dasharray", inPool ? null : "3,2");
      svg.append("text").attr("x", x(c) + x.bandwidth() / 2).attr("y", y(r) + y.bandwidth() / 2 + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", v > 0.82 ? "#fff" : "#1c1c1c").text(fmt3(v));
    });
  });
  svg.append("text").attr("x", w - 10).attr("y", 28 + 26 * M.rows.length + 8).attr("text-anchor", "end").attr("font-size", 10).attr("fill", "#6b6b6b").text("dashed red = out of distribution (not in the training pool)");
}

function drawGating() {
  const [svg, w] = svgIn($("#gating"), 230);
  const m = { l: 40, r: 12, t: 24, b: 30 };
  const x = d3.scaleLinear().domain([0, 60]).range([m.l, w - m.r]), y = d3.scaleLinear().domain([0.5, 1]).range([230 - m.b, m.t]);
  title(svg, w, "Confidence-gated frontier on held-out tasks (Figure 4): accuracy of stopped rollouts vs. generation steps saved, θ annotated");
  svg.append("g").attr("transform", `translate(0,${230 - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(d => d + "%")).selectAll("text").style("font-size", "10px");
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)).selectAll("text").style("font-size", "10px");
  Object.entries(R.gating).forEach(([b, g]) => {
    const pts = g.shown.map(i => ({ x: g.steps_saved[i], y: g.accuracy[i], th: g.thetas[i] }));
    svg.append("path").attr("d", d3.line().x(d => x(d.x)).y(d => y(d.y))(pts)).attr("fill", "none").attr("stroke", BCOL[b]).attr("stroke-width", 2);
    svg.append("g").selectAll("circle").data(pts).join("circle").attr("cx", d => x(d.x)).attr("cy", d => y(d.y)).attr("r", 3.5).attr("fill", BCOL[b]).append("title").text(d => `${BENCH[b]} θ=${d.th}: ${d.x}% saved, acc ${d.y}`);
    pts.filter(d => [0.6, 0.8, 0.9].includes(d.th)).forEach(d => svg.append("text").attr("x", x(d.x) + 5).attr("y", y(d.y) - 5).attr("font-size", 9).attr("fill", BCOL[b]).text(`θ=${d.th}`));
    svg.append("text").attr("x", x(pts[0].x) - 4).attr("y", y(pts[0].y) + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", BCOL[b]).text(BENCH[b]);
  });
}

function drawGrpo() {
  const G = R.grpo, ids = G.table.map(t => t.id);
  if (!grpoOn) grpoOn = new Set(ids);
  const tg = $("#grpo-toggles"); tg.innerHTML = "";
  G.table.forEach(t => {
    const lab = document.createElement("label"); const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = grpoOn.has(t.id);
    cb.onchange = () => { if (cb.checked) grpoOn.add(t.id); else grpoOn.delete(t.id); drawGrpo(); };
    lab.appendChild(cb); const sw = document.createElement("i"); sw.className = "sw"; sw.style.background = VCOL[t.support]; sw.style.borderBottom = t.theta === 0.7 ? "2px dashed #333" : t.theta === 0.65 ? "2px dotted #333" : "none"; lab.appendChild(sw);
    lab.appendChild(document.createTextNode(t.id === "baseline" ? "GRPO baseline" : `${t.support} θ=${t.theta} m_c=${t.mc}`)); tg.appendChild(lab);
  });
  const style = (t) => ({ col: VCOL[t.support], dash: t.theta === 0.7 ? "6,3" : t.theta === 0.65 ? "2,3" : null, w: t.mc === 4 ? 2.6 : 1.6 });
  const panel = (el, key, ttl, ydom, yfmt, band) => {
    const [svg, w] = svgIn(el, 210); const m = { l: 40, r: 10, t: 22, b: 26 };
    const x = d3.scaleLinear().domain([0, G.matched_steps]).range([m.l, w - m.r]), y = d3.scaleLinear().domain(ydom).range([210 - m.b, m.t]);
    title(svg, w, ttl);
    svg.append("g").attr("transform", `translate(0,${210 - m.b})`).call(d3.axisBottom(x).ticks(5)).selectAll("text").style("font-size", "10px");
    svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(yfmt)).selectAll("text").style("font-size", "10px");
    if (band && G.curves.baseline) { const c = G.curves.baseline[key]; svg.append("path").attr("d", d3.area().x((d, i) => x(c.step[i])).y0((d, i) => y(c.lo[i])).y1((d, i) => y(c.hi[i]))(c.mean)).attr("fill", "#6b7280").attr("fill-opacity", 0.15); }
    if (key === "pass1" && G.base_pass1 != null) svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(G.base_pass1)).attr("y2", y(G.base_pass1)).attr("stroke", "#9ca3af").attr("stroke-dasharray", "2,2");
    G.table.forEach(t => { if (!grpoOn.has(t.id) || !G.curves[t.id]) return; const c = G.curves[t.id][key], s = style(t);
      svg.append("path").attr("d", d3.line().x((d, i) => x(c.step[i])).y(d => y(d)).defined(d => d != null)(c.mean)).attr("fill", "none").attr("stroke", s.col).attr("stroke-width", s.w).attr("stroke-dasharray", s.dash); });
  };
  panel($("#grpo-reward"), "reward", "(a) training reward · band = baseline seeds", [0.1, 0.45], d3.format(".2f"), true);
  panel($("#grpo-pass"), "pass1", "(b) test pass@1 · dashed = base model", [15, 45], d => d + "%", false);
  panel($("#grpo-flops"), "flops_pct", "(c) cumulative FLOPs, % of baseline", [30, 105], d => d + "%", false);
  // frontier: FLOPs saved vs pass@1
  const [svg, w] = svgIn($("#grpo-frontier"), 230); const m = { l: 40, r: 12, t: 24, b: 30 };
  const x = d3.scaleLinear().domain([0, 70]).range([m.l, w - m.r]), y = d3.scaleLinear().domain([24, 40]).range([230 - m.b, m.t]);
  title(svg, w, "Accuracy–compute frontier (Table 3): training FLOPs saved vs. final pass@1, mean ± std over 3 seeds; band = baseline ± 1 std");
  svg.append("g").attr("transform", `translate(0,${230 - m.b})`).call(d3.axisBottom(x).ticks(7).tickFormat(d => d + "%")).selectAll("text").style("font-size", "10px");
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d => d + "%")).selectAll("text").style("font-size", "10px");
  const base = G.table.find(t => t.id === "baseline");
  svg.append("rect").attr("x", m.l).attr("y", y(base.pass1[0] + base.pass1[1])).attr("width", w - m.l - m.r).attr("height", y(base.pass1[0] - base.pass1[1]) - y(base.pass1[0] + base.pass1[1])).attr("fill", "#6b7280").attr("fill-opacity", 0.12);
  svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(base.pass1[0])).attr("y2", y(base.pass1[0])).attr("stroke", "#6b7280").attr("stroke-dasharray", "3,3");
  svg.append("text").attr("x", m.l + 4).attr("y", y(base.pass1[0]) - 4).attr("font-size", 10).attr("fill", "#6b7280").text(`GRPO baseline ${base.pass1[0]}%`);
  G.table.filter(t => t.flops_saved).forEach(t => { const s = style(t), px = x(t.flops_saved[0]), py = y(t.pass1[0]);
    svg.append("line").attr("x1", px).attr("x2", px).attr("y1", y(t.pass1[0] - t.pass1[1])).attr("y2", y(t.pass1[0] + t.pass1[1])).attr("stroke", s.col).attr("stroke-opacity", 0.5);
    svg.append("line").attr("x1", x(t.flops_saved[0] - t.flops_saved[1])).attr("x2", x(t.flops_saved[0] + t.flops_saved[1])).attr("y1", py).attr("y2", py).attr("stroke", s.col).attr("stroke-opacity", 0.5);
    svg.append("circle").attr("cx", px).attr("cy", py).attr("r", t.mc === 4 ? 7 : 5).attr("fill", s.col).attr("stroke", "#fff").attr("stroke-width", 1.5).append("title").text(`${t.support} θ=${t.theta} m_c=${t.mc}: ${t.flops_saved[0]}% saved, pass@1 ${t.pass1[0]}±${t.pass1[1]}, retains ${Math.round(t.retained_gain * 100)}% of the gain`);
    svg.append("text").attr("x", px + 8).attr("y", py - 7).attr("font-size", 10).attr("fill", s.col).text(`θ=${t.theta} m_c=${t.mc}`); });
}

function numbers() {
  const t1 = R.table1.rows, best = t1.map(r => r.methods[r.best_baseline][0]);
  const set = (id, items) => { $(id).innerHTML = items.map(([v, l]) => `<div class="num"><b>${v}</b><span>${l}</span></div>`).join(""); };
  set("#n2", [[`${fmt3(d3.min(best))}–${fmt3(d3.max(best))}`, "best per-task AUROC across the 9 cells"], ["~33", "labeled rollouts per fit"], [`${t1.filter(r => (r.methods.PN || [0])[0] >= r.methods[r.best_baseline][0] - 0.0005).length} of 9`, "cells where Precog-RM matches or beats the best per-task fit"]]);
  const f = R.fewshot; const marg = Object.values(f).map(b => b["1"].ProtoNet - d3.max(["NC", "RBF", "FT1"], m => b["1"][m]));
  set("#n4", [[`+${fmt3(d3.min(marg))} to +${fmt3(d3.max(marg))}`, "AUROC margin over the best baseline at k=1"], ["k=2 ≈ baseline k=8", "four-fold fewer labeled rollouts"], [fmt3(R.generalization.models.matrix[6].reduce((a, b) => a + b, 0) / 3), "mean held-out AUROC, all-policy pool"]]);
  const g = R.gating.tau2, i9 = g.thetas.indexOf(0.9), i8 = g.thetas.indexOf(0.8);
  set("#n5", [[`${Math.round(g.stop_rate[i9] * 100)}%`, `Tau2 rollouts stopped at θ=0.9, at ${g.accuracy[i9]} accuracy`], [`${g.steps_saved[i8]}%`, "Tau2 generation steps saved at θ=0.8"], [`${Math.round(R.gating.agentclinic.stop_rate[i8] * 100)}%`, "AgentClinic rollouts stopped at θ=0.8 (abstains)"]]);
  const T = R.grpo.table, perm = T.find(t => t.id === "limit"), cons = T.find(t => t.id === "es08");
  set("#n6", [[`${perm.flops_saved[0]}%`, `training FLOPs saved, permanent θ=0.8, retaining ${Math.round(perm.retained_gain * 100)}% of the gain`], [`${cons.pass1[0]}% vs ${T[0].pass1[0]}%`, `pass@1, rolling θ=0.8 m_c=4 vs baseline, at ${cons.flops_saved[0]}% less compute`], [`${T.filter(t => t.flops_saved && Math.abs(t.pass1[0] - T[0].pass1[0]) <= T[0].pass1[1]).length} of 6`, "configurations within one baseline std"]]);
}
init().catch(e => console.error(e));
})();
// chapter nav highlight
(() => {
  const links = [...document.querySelectorAll(".toc a")];
  const ids = links.map(a => a.getAttribute("href").slice(1));
  const io = new IntersectionObserver((es) => { es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id)); }); }, { rootMargin: "-8% 0px -82% 0px" });
  ids.forEach(id => { const el = document.getElementById(id); if (el) io.observe(el); });
})();
