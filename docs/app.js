/* ===================== VitalAgent project page ===================== */
"use strict";

const TRACE_ROOT = "demo_traces/";

if ("scrollRestoration" in history) {
  history.scrollRestoration = "manual";
}
if (window.location.hash === "#demo") {
  history.replaceState(null, "", window.location.pathname + window.location.search);
  window.scrollTo(0, 0);
}

/* ---------- tiny DOM helper ---------- */
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
}
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* ---------- minimal, safe markdown (bold / code / lists / paragraphs) ---------- */
function miniMarkdown(src) {
  const lines = String(src).split("\n");
  let out = "", inList = false, para = [];
  const inline = (t) =>
    esc(t)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  const flushPara = () => { if (para.length) { out += "<p>" + para.join("<br>") + "</p>"; para = []; } };
  for (let raw of lines) {
    const line = raw.replace(/\s+$/g, "");
    if (/^\s*[-*]\s+/.test(line)) {
      flushPara();
      if (!inList) { out += "<ul>"; inList = true; }
      out += "<li>" + inline(line.replace(/^\s*[-*]\s+/, "")) + "</li>";
    } else if (line.trim() === "") {
      if (inList) { out += "</ul>"; inList = false; }
      flushPara();
    } else {
      if (inList) { out += "</ul>"; inList = false; }
      para.push(inline(line));
    }
  }
  if (inList) out += "</ul>";
  flushPara();
  return out;
}

/* ===================== reactive replay card ===================== */
function buildReplayCard(c) {
  const card = el("div", "replay");

  const playBtn = el("button", "play-btn", "▶ Play");

  const head = el("div", "replay-head");
  head.appendChild(el("span", "src-chip", esc(c.dataset)));
  head.appendChild(el("span", "replay-title", esc(c.title)));
  head.appendChild(playBtn);
  card.appendChild(head);

  const body = el("div", "replay-body");
  body.appendChild(el("div", "q-label", "User query"));
  body.appendChild(el("p", "q-line", esc(c.question)));
  const stage = el("div", "stage");
  body.appendChild(stage);
  card.appendChild(body);

  const routing = el("div", "routing-badge",
    `<span class="dot"></span> routed → ${esc(c.routing.actual_path)}`);
  stage.appendChild(routing);

  const tools = c.tool_chain.map((t, i) => {
    const step = el("div", "tool-step");
    step.appendChild(el("div", "num", String(i + 1)));
    const meta = el("div");
    meta.appendChild(el("div", "tool-name", esc(t.name)));
    meta.appendChild(el("div", "tool-label", esc(t.label)));
    step.appendChild(meta);
    stage.appendChild(step);
    return step;
  });

  const ans = el("div", "answer-box");
  ans.appendChild(el("div", "ans-label", "Grounded answer"));
  if (c.answer_rich) ans.appendChild(el("div", "answer-md", miniMarkdown(c.answer)));
  else ans.appendChild(el("div", "answer-badge", esc(c.answer)));
  stage.appendChild(ans);

  const foot = el("div", "replay-foot");
  const gold = Array.isArray(c.gold_answer) ? c.gold_answer.join(", ") : c.gold_answer;
  if (c.correct) {
    foot.appendChild(el("span", "verdict correct",
      `<span class="check">✓</span> matches gold &nbsp;<span class="foot-stat">(“${esc(gold)}”)</span>`));
  } else {
    foot.appendChild(el("span", "verdict", `gold: ${esc(gold)}`));
  }
  if (c.stats && c.stats.elapsed_sec != null) {
    foot.appendChild(el("span", "foot-stat foot-spacer",
      `${c.tool_chain.length} tool${c.tool_chain.length > 1 ? "s" : ""} · ${c.stats.elapsed_sec}s`));
  }

  const steps = [routing, ...tools, ans];
  let timer = null;
  function play() {
    steps.forEach((s) => s.classList.remove("reveal"));
    if (timer) clearTimeout(timer);
    playBtn.disabled = true;
    playBtn.innerHTML = "▶ Play";
    playBtn.classList.remove("replay-again");
    let i = 0;
    const tick = () => {
      if (i >= steps.length) {
        playBtn.disabled = false;
        playBtn.classList.add("replay-again");
        playBtn.innerHTML = "↻ Replay";
        return;
      }
      steps[i].classList.add("reveal");
      i++;
      timer = setTimeout(tick, 600);
    };
    timer = setTimeout(tick, 120);
  }
  playBtn.addEventListener("click", play);

  card.appendChild(foot);
  card._play = play;
  return card;
}

/* ---------- demo board: single frame + case selector ---------- */
function buildDemoBoard(cases) {
  const board = document.getElementById("demo-board");
  board.innerHTML = "";
  const tabs = el("div", "case-tabs");
  const stage = el("div", "case-stage");
  board.appendChild(tabs);
  board.appendChild(stage);

  let current = -1;
  function select(i) {
    if (i === current) return;
    current = i;
    [...tabs.children].forEach((t, j) => t.classList.toggle("active", j === i));
    stage.innerHTML = "";
    const card = buildReplayCard(cases[i]);
    stage.appendChild(card);
    setTimeout(() => card._play(), 60);
  }

  cases.forEach((c, i) => {
    const tab = el("button", "case-tab");
    tab.innerHTML =
      `<div class="ct-top"><span class="ct-idx">${i + 1}</span><span class="ct-ds">${esc(c.dataset)} · ${esc(c.modality)}</span></div>
       <div class="ct-title">${esc(c.title)}</div>`;
    tab.addEventListener("click", () => select(i));
    tabs.appendChild(tab);
  });

  select(0);
}

/* ===================== proactive monitoring timeline ===================== */
const RHYTHM_COLOR = { N: "#3fb27f", AF: "#e0533d", Other: "#b9c3cc", Unknown: "#e1e7ec" };

function buildProactive(d) {
  const host = document.getElementById("proactive");
  host.innerHTML = "";

  const top = el("div", "mon-top");
  const legend = el("div", "legend");
  [["N", "Normal sinus"], ["AF", "Atrial fibrillation"], ["Other", "Other / noise"]]
    .forEach(([k, lab]) => legend.appendChild(el("span", null, `<i style="background:${RHYTHM_COLOR[k]}"></i>${esc(lab)}`)));
  legend.appendChild(el("span", null, `<i style="background:#16202a;width:3px;height:13px;border-radius:1px"></i>VitalAgent detection`));
  top.appendChild(legend);
  const playBtn = el("button", "play-btn", "▶ Play monitor");
  top.appendChild(playBtn);
  host.appendChild(top);

  const W = 960, H = 178, padL = 10, padR = 10;
  const trackY = 86, trackH = 34, innerW = W - padL - padR;
  const n = d.windows.length, bw = innerW / n;
  const xOf = (i) => padL + i * bw;

  const svgWrap = el("div", "mon-svg-wrap");
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svgWrap.appendChild(svg);
  host.appendChild(svgWrap);

  const mk = (name, attrs) => {
    const e = document.createElementNS(ns, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };

  d.windows.forEach((w) => svg.appendChild(mk("rect", {
    x: xOf(w.i), y: trackY, width: Math.ceil(bw) + 0.5, height: trackH,
    fill: RHYTHM_COLOR[w.truth] || "#e1e7ec",
  })));
  svg.appendChild(mk("rect", { x: padL, y: trackY, width: innerW, height: trackH, fill: "none", stroke: "#cdd9e4", rx: 6 }));

  for (let i = 0; i <= n; i += 30) {
    const x = xOf(i);
    svg.appendChild(mk("line", { x1: x, y1: trackY + trackH, x2: x, y2: trackY + trackH + 5, stroke: "#aab6c1", "stroke-width": 1 }));
    svg.appendChild(mk("text", { x: x, y: trackY + trackH + 18, "text-anchor": "middle", "font-family": "Inter", "font-size": 11, fill: "#8b97a4" })).textContent = (i / 6).toFixed(0) + "m";
  }

  // detections: AF onsets get a prominent alert flag + latency; offsets get a quiet tick.
  let lastLabelX = -Infinity, lvl = 0;
  const annos = d.matched_transitions.map((m) => {
    const x = xOf(m.pred_window);
    const onset = m.pair.endsWith("AF");
    const g = mk("g", { opacity: 0 });
    g.style.transition = "opacity .4s ease";
    g.appendChild(mk("line", {
      x1: x, y1: trackY - 2, x2: x, y2: trackY + trackH + 2,
      stroke: onset ? "#16202a" : "#9aa7b3", "stroke-width": onset ? 1.6 : 1.2,
    }));
    g.appendChild(mk("circle", { cx: x, cy: trackY, r: onset ? 4 : 3, fill: onset ? "#e0533d" : "#9aa7b3" }));

    if (onset && x - lastLabelX >= 92) {
      const up = lvl % 2 === 0; lvl++; lastLabelX = x;
      const ly = up ? trackY - 40 : trackY - 22;
      const anchor = x > W - 90 ? "end" : (x < 70 ? "start" : "middle");
      const txt = mk("text", { x: x, y: ly, "text-anchor": anchor, "font-family": "Inter", "font-size": 12, "font-weight": 700, fill: "#c33f29" });
      txt.textContent = `⚠ AF +${m.latency_s}s`;
      if (up) g.appendChild(mk("line", { x1: x, y1: ly + 4, x2: x, y2: trackY - 2, stroke: "#e7b3a8", "stroke-width": 1 }));
      g.appendChild(txt);
    }
    svg.appendChild(g);
    return { x, g };
  });

  const head = mk("line", { x1: padL, y1: trackY - 48, x2: padL, y2: trackY + trackH + 8, stroke: "#0e8a8a", "stroke-width": 2 });
  head.style.opacity = 0;
  svg.appendChild(head);

  // onset-focused stats
  const s = d.summary;
  const gtOnsets = d.ground_truth_transitions.filter((t) => t.to === "AF").length;
  const onsetLat = d.matched_transitions.filter((m) => m.pair.endsWith("AF")).map((m) => m.latency_s).sort((a, b) => a - b);
  const transitionLat = (s.latencies_s || d.matched_transitions.map((m) => m.latency_s)).slice().sort((a, b) => a - b);
  const caught = onsetLat.length;
  const avgLat = transitionLat.length
    ? Math.round(transitionLat.reduce((sum, value) => sum + value, 0) / transitionLat.length)
    : 0;
  const afPct = ((s.truth_label_counts.AF || 0) / s.n_windows * 100).toFixed(0);

  const stats = el("div", "mon-stats");
  [
    [`${(s.duration_s / 60).toFixed(0)} min`, "ECG stream monitored"],
    [`${caught}/${gtOnsets}`, "AF episodes caught"],
    [`${afPct}%`, "of time spent in AF"],
    [`${avgLat}s`, "average transition delay"],
  ].forEach(([nv, lv]) => {
    const st = el("div", "mon-stat");
    st.appendChild(el("div", "n", nv));
    st.appendChild(el("div", "l", lv));
    stats.appendChild(st);
  });
  host.appendChild(stats);

  if (d.note) host.appendChild(el("div", "mon-note", esc(d.note)));

  let raf = null;
  function playMonitor() {
    if (raf) cancelAnimationFrame(raf);
    annos.forEach((a) => a.g.setAttribute("opacity", 0));
    head.style.opacity = 1;
    playBtn.disabled = true;
    const dur = 10000, t0 = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - t0) / dur);
      const x = padL + p * innerW;
      head.setAttribute("x1", x); head.setAttribute("x2", x);
      annos.forEach((a) => { if (x >= a.x) a.g.setAttribute("opacity", 1); });
      if (p < 1) raf = requestAnimationFrame(step);
      else {
        playBtn.disabled = false;
        playBtn.classList.add("replay-again");
        playBtn.innerHTML = "↻ Replay monitor";
        setTimeout(() => { head.style.opacity = 0; }, 600);
      }
    };
    raf = requestAnimationFrame(step);
  }
  playBtn.addEventListener("click", playMonitor);

  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        playMonitor();
        io.disconnect();
      }
    });
  }, { threshold: 0.4 });
  io.observe(svgWrap);
}

/* ---------- bibtex copy ---------- */
function wireCopy() {
  const btn = document.getElementById("copy-bib");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    const txt = document.getElementById("bibtex-block").textContent;
    try {
      await navigator.clipboard.writeText(txt);
      btn.textContent = "Copied!"; btn.classList.add("copied");
      setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1600);
    } catch (e) { /* ignore */ }
  });
}

function wireInternalScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (event) => {
      const hash = link.getAttribute("href");
      if (!hash || hash === "#") return;
      const target = document.querySelector(hash);
      if (!target) return;

      event.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      history.replaceState(null, "", window.location.pathname + window.location.search);
    });
  });
}

/* ---------- load + orchestrate ---------- */
async function loadCase(file) {
  const r = await fetch(TRACE_ROOT + file);
  if (!r.ok) throw new Error("fetch " + file + " → " + r.status);
  return r.json();
}

async function main() {
  wireInternalScroll();
  wireCopy();

  let index;
  try {
    index = await (await fetch(TRACE_ROOT + "index.json")).json();
  } catch (e) {
    document.getElementById("demo-board").innerHTML =
      `<div class="loading">Could not load demo traces. Serve this folder over HTTP (e.g. <code>python -m http.server</code>).</div>`;
    return;
  }

  const byId = {};
  await Promise.all(index.cases.map(async (m) => {
    try { byId[m.id] = await loadCase(m.file); } catch (e) { /* skip */ }
  }));

  const order = [
    "reactive_icentia11k_af",
    "reactive_af_ppg_ecg_af",
    "reactive_ppg_dalia_hr_trend",
    "reactive_wesad_stress",
  ];
  const cases = order.map((id) => byId[id]).filter(Boolean);
  if (cases.length) buildDemoBoard(cases);

  if (byId["proactive_af_onset"]) buildProactive(byId["proactive_af_onset"]);
}

document.addEventListener("DOMContentLoaded", main);
