"use strict";

/* ─── MEAM // ELASTIC RUNNER — frontend ─────────────────────────────
   Mission-control telemetry surface. Owns:
     - composition input + live sum bar
     - cell knobs (advanced)
     - LAUNCH → POST /api/predict
     - poll loop /api/jobs/<id>
     - live scope (pxx/pyy/pzz vs step) drawn into an SVG polyline triple
     - terminal log tail (auto-scroll)
     - readout dials + OVITO render slot
     - mission clock (T+MM:SS.f) while running
   ──────────────────────────────────────────────────────────────── */

const ELEMENTS = Array.from(document.querySelectorAll(".el__input"))
  .map((i) => i.name);
const KNOB_FIELDS = [
  "box_size_m", "temperature", "total_steps",
  "thermo_interval", "dump_interval",
];
const PRESETS = {
  pure_al:  { Al: 1.0 },
  aa7075:   { Al: 0.91, Mg: 0.029, Zn: 0.061 },
};

/* ─── DOM handles ─────────────────────────────────────────────────── */
const $        = (id) => document.getElementById(id);
const body     = document.body;
const stateVal = $("state-val");
const clock    = $("clock");
const jobIdEl  = $("job-id");
const sumVal   = $("sum-val");
const sumDelta = $("sum-delta");
const sumFill  = $("sum-fill");
const launchBtn= $("btn-launch");
const notice   = $("notice");
const statusStrip = $("status-strip");
const statusLbl   = $("status-strip-lbl");
const statusSub   = $("status-strip-sub");
const scopeWrap   = $("scope");
const scopeEmpty  = $("scope-empty");
const scopeCount  = $("scope-count");
const scopeStep   = $("scope-step");
const scopePxx    = $("scope-pxx");
const scopePyy    = $("scope-pyy");
const scopePzz    = $("scope-pzz");
const traceXX     = $("trace-xx");
const traceYY     = $("trace-yy");
const traceZZ     = $("trace-zz");
const term        = $("terminal-body");
const termCount   = $("log-count");
const btnWrap     = $("btn-wrap");
const readout     = $("readout");
const readoutSrc  = $("readout-src");
const rC11   = $("r-c11");
const rC12   = $("r-c12");
const rE     = $("r-e");
const rNu    = $("r-nu");
const rWarn  = $("r-warning");
const renderFig = $("render-fig");
const rRender   = $("r-render");

/* ─── State ───────────────────────────────────────────────────────── */
let mission = { active: false, startedAt: 0, raf: 0 };
let pollCtl = null;
let thermo  = [];         // accumulated thermo records {step,pxx,pyy,pzz}

/* ─── Status state machine ────────────────────────────────────────── */
function setState(s, sub) {
  body.dataset.state = s;
  statusStrip.dataset.state = s;
  const map = {
    standby: { lbl: "SYSTEM READY",  sub: "Awaiting composition." },
    queued:  { lbl: "QUEUED",        sub: "Worker is picking up the job…" },
    running: { lbl: "RUNNING",       sub: "LAMMPS streaming telemetry." },
    done:    { lbl: "OK",            sub: "Run complete. Result locked in." },
    error:   { lbl: "FAULT",         sub: "See log for traceback." },
  };
  const m = map[s] || map.standby;
  statusLbl.textContent = m.lbl;
  statusSub.textContent = sub || m.sub;
  stateVal.textContent = m.lbl;
}

/* ─── Mission clock ───────────────────────────────────────────────── */
function startClock() {
  mission.active = true;
  mission.startedAt = performance.now();
  const tick = () => {
    if (!mission.active) return;
    const t = (performance.now() - mission.startedAt) / 1000;
    const mm = Math.floor(t / 60).toString().padStart(2, "0");
    const ss = (t % 60).toFixed(1).padStart(4, "0");
    clock.textContent = `${mm}:${ss}`;
    mission.raf = requestAnimationFrame(tick);
  };
  tick();
}
function stopClock() {
  mission.active = false;
  if (mission.raf) cancelAnimationFrame(mission.raf);
  mission.raf = 0;
}

/* ─── Composition: live sum bar + per-cell fill ──────────────────── */
function recomputeSum() {
  let total = 0;
  for (const el of ELEMENTS) {
    const inp = $(`el-${el}`);
    if (!inp) continue;
    const raw = inp.value.trim();
    const f = raw === "" ? 0 : Number(raw);
    const v = Number.isFinite(f) && f > 0 ? f : 0;
    const lbl = inp.closest(".el");
    if (lbl) {
      const fill = lbl.querySelector(".el__fill");
      if (fill) fill.style.transform = `scaleX(${Math.min(v, 1)})`;
    }
    total += v;
  }
  sumVal.textContent = total.toFixed(4);
  const d = total - 1.0;
  sumDelta.textContent = `Δ ${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(4)}`;
  const ok = Math.abs(d) < 1e-3;
  sumDelta.classList.toggle("is-ok",   ok);
  sumDelta.classList.toggle("is-warn", !ok && total > 0);
  sumFill.style.width = `${Math.min(total, 1.0) * 100}%`;
  sumFill.classList.toggle("is-ok", ok);
  return total;
}
document.addEventListener("input", (ev) => {
  if (ev.target.matches(".el__input")) recomputeSum();
});

/* ─── Presets / clear ─────────────────────────────────────────────── */
function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  for (const el of ELEMENTS) $(`el-${el}`).value = "";
  for (const [k, v] of Object.entries(p)) {
    const inp = $(`el-${k}`);
    if (inp) inp.value = v;
  }
  recomputeSum();
}
$("btn-preset-al").addEventListener("click", () => applyPreset("pure_al"));
$("btn-preset-al7075").addEventListener("click", () => applyPreset("aa7075"));
$("btn-clear").addEventListener("click", () => {
  for (const el of ELEMENTS) $(`el-${el}`).value = "";
  recomputeSum();
});

/* ─── Terminal: append + auto-scroll ──────────────────────────────── */
let termLines = 0;
function termAppend(line) {
  if (termLines === 0) term.textContent = "";   // wipe the boot prompt
  // keep last ~400 lines in the DOM
  term.textContent += `${line}\n`;
  termLines++;
  termCount.textContent = termLines;
  // chunk-clip if it grows too long
  if (termLines > 400) {
    const chunk = term.textContent.split("\n");
    term.textContent = chunk.slice(chunk.length - 401).join("\n");
  }
  term.scrollTop = term.scrollHeight;
}
function termReset() {
  termLines = 0;
  termCount.textContent = 0;
  term.textContent = "> waiting for telemetry...\n";
}
btnWrap.addEventListener("click", () => {
  const on = term.dataset.wrap === "on";
  term.dataset.wrap = on ? "off" : "on";
  btnWrap.textContent = `wrap: ${on ? "off" : "on"}`;
});

/* ─── Scope: redraw whenever thermo grows ─────────────────────────── */
function redrawScope() {
  if (!thermo.length) {
    scopeEmpty.style.display = "grid";
    traceXX.setAttribute("points", "");
    traceYY.setAttribute("points", "");
    traceZZ.setAttribute("points", "");
    scopeCount.textContent = "0";
    return;
  }
  scopeEmpty.style.display = "none";
  scopeCount.textContent = thermo.length;
  // bounds
  let minStep = Infinity, maxStep = -Infinity;
  let absMax = 1;
  for (const t of thermo) {
    if (t.step < minStep) minStep = t.step;
    if (t.step > maxStep) maxStep = t.step;
    absMax = Math.max(absMax, Math.abs(t.pxx), Math.abs(t.pyy), Math.abs(t.pzz));
  }
  if (minStep === maxStep) maxStep = minStep + 1;
  const W = 600, H = 220, ZERO = H / 2;
  const xs = (s) => ((s - minStep) / (maxStep - minStep)) * W;
  const ys = (p) => ZERO - (p / absMax) * (H / 2 - 8);
  const ptsFor = (key) => thermo.map((t) => `${xs(t.step).toFixed(1)},${ys(t[key]).toFixed(1)}`).join(" ");
  traceXX.setAttribute("points", ptsFor("pxx"));
  traceYY.setAttribute("points", ptsFor("pyy"));
  traceZZ.setAttribute("points", ptsFor("pzz"));
  const last = thermo[thermo.length - 1];
  scopeStep.textContent = `step ${last.step}`;
  scopePxx.textContent = last.pxx.toFixed(2);
  scopePyy.textContent = last.pyy.toFixed(2);
  scopePzz.textContent = last.pzz.toFixed(2);
}

/* ─── Readout: write dials when result lands ──────────────────────── */
function renderResult(r, cached, cacheKey) {
  rC11.textContent = r.C11_GPa.toFixed(2);
  rC12.textContent = r.C12_GPa.toFixed(2);
  rE.textContent   = r.E_GPa.toFixed(2);
  rNu.textContent  = r.nu.toFixed(4);
  readout.hidden = false;
  readoutSrc.textContent = cached ? "cached" : "live";

  if (r.physical === false) {
    rWarn.textContent = `Non-physical: ${r.physical_reason}`;
    rWarn.hidden = false;
  } else {
    rWarn.hidden = true;
  }
}
function renderFigure(available, cacheKey) {
  if (available && cacheKey) {
    rRender.src = `/api/renders/${cacheKey}?t=${Date.now()}`;
    renderFig.hidden = false;
  } else {
    renderFig.hidden = true;
  }
}

/* ─── Payload collection ──────────────────────────────────────────── */
function collectPayload() {
  const composition = {};
  for (const el of ELEMENTS) {
    const raw = $(`el-${el}`).value.trim();
    if (raw === "") continue;
    const f = Number(raw);
    if (Number.isFinite(f) && f > 0) composition[el] = f;
  }
  const knobs = {};
  for (const k of KNOB_FIELDS) {
    const v = Number($(k).value);
    if (Number.isFinite(v)) knobs[k] = v;
  }
  const do_viz = $("do_viz").checked;
  return { composition, knobs, do_viz };
}

/* ─── Poll loop ───────────────────────────────────────────────────── */
async function pollJob(jobId, deadlineMs) {
  while (Date.now() < deadlineMs) {
    const resp = await fetch(`/api/jobs/${jobId}`);
    if (!resp.ok) {
      setState("error", `polling failed: HTTP ${resp.status}`);
      stopClock();
      launchBtn.disabled = false;
      launchBtn.dataset.busy = "";
      return;
    }
    const body = await resp.json();
    setState(body.status);

    // merge log lines (the server returns last 50 already)
    const lines = body.log_lines || [];
    const cur = termLines === 0 ? 0 : termLines;
    if (lines.length && cur < lines.length) {
      // simple replace-with-tail strategy: always re-render the last 50 we get
      termReset();
      for (const l of lines) termAppend(l);
    }

    // merge thermo records
    const tr = body.thermo || [];
    if (tr.length !== thermo.length) {
      thermo = tr.slice();
      redrawScope();
    }

    if (body.status === "done") {
      stopClock();
      renderResult(body.result, false, body.cache_key);
      renderFigure(body.render_available, body.cache_key);
      launchBtn.disabled = false;
      launchBtn.dataset.busy = "";
      return;
    }
    if (body.status === "error") {
      stopClock();
      setState("error", body.error ? body.error.message : "unknown");
      launchBtn.disabled = false;
      launchBtn.dataset.busy = "";
      return;
    }

    await new Promise((r) => setTimeout(r, 1000));
  }
  // timeout
  setState("error", "polling deadline reached (5 min)");
  stopClock();
  launchBtn.disabled = false;
  launchBtn.dataset.busy = "";
}

/* ─── Submit handler ──────────────────────────────────────────────── */
$("predict-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  notice.hidden = true; notice.textContent = "";

  // reset run state
  thermo = [];
  redrawScope();
  termReset();
  readout.hidden = true;
  renderFig.hidden = true;
  jobIdEl.textContent = "———————";
  clock.textContent = "00:00.0";

  const payload = collectPayload();
  if (!Object.keys(payload.composition).length) {
    notice.hidden = false;
    notice.textContent = "Composition is empty — enter at least one fraction.";
    return;
  }

  launchBtn.disabled = true;
  launchBtn.dataset.busy = "1";
  setState("queued", "Submitting payload…");

  let resp, body;
  try {
    resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    body = await resp.json();
  } catch (e) {
    setState("error", `network: ${e.message}`);
    launchBtn.disabled = false;
    launchBtn.dataset.busy = "";
    return;
  }
  if (!resp.ok) {
    setState("error", body.error || `HTTP ${resp.status}`);
    notice.hidden = false;
    notice.textContent = body.error || `HTTP ${resp.status}`;
    launchBtn.disabled = false;
    launchBtn.dataset.busy = "";
    return;
  }

  if (body.cached) {
    setState("done", "served from cache");
    renderResult(body.result, true, body.cache_key);
    renderFigure(body.render_available, body.cache_key);
    jobIdEl.textContent = (body.cache_key || "").slice(0, 7);
    launchBtn.disabled = false;
    launchBtn.dataset.busy = "";
    return;
  }

  jobIdEl.textContent = (body.job_id || "").slice(0, 7);
  startClock();
  await pollJob(body.job_id, Date.now() + 5 * 60 * 1000);
});

/* ─── Boot ────────────────────────────────────────────────────────── */
setState("standby");
recomputeSum();
termReset();
redrawScope();

/* ─── Theme toggle ───────────────────────────────────────────────────
   The initial data-theme attribute is set by an inline <head> script.
   This wires the rail button to flip and persist the choice. */
(function () {
  const btn = document.getElementById("theme-toggle");
  const lbl = document.getElementById("theme-toggle-lbl");
  if (!btn) return;
  const sync = () => {
    const cur = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
    if (lbl) lbl.textContent = cur === "light" ? "DARK" : "LIGHT";
  };
  sync();
  btn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
    const next = cur === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("meam-theme", next); } catch (e) {}
    sync();
  });
})();

