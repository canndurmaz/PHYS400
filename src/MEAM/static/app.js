"use strict";

const ELEMENTS = Array.from(document.querySelectorAll(".comp-table input"))
  .map((i) => i.name);
const KNOB_FIELDS = [
  "box_size_m", "temperature", "total_steps",
  "thermo_interval", "dump_interval",
];

function collectPayload() {
  const composition = {};
  for (const el of ELEMENTS) {
    const raw = document.getElementById(`el-${el}`).value.trim();
    if (raw === "") continue;
    const f = Number(raw);
    if (Number.isFinite(f) && f > 0) composition[el] = f;
  }
  const knobs = {};
  for (const k of KNOB_FIELDS) {
    const v = Number(document.getElementById(k).value);
    if (Number.isFinite(v)) knobs[k] = v;
  }
  const do_viz = document.getElementById("do_viz").checked;
  return { composition, knobs, do_viz };
}

function renderResult(body) {
  const sec = document.getElementById("result");
  sec.hidden = false;
  const r = body.result;
  document.getElementById("r-c11").textContent = r.C11_GPa.toFixed(2);
  document.getElementById("r-c12").textContent = r.C12_GPa.toFixed(2);
  document.getElementById("r-e").textContent = r.E_GPa.toFixed(2);
  document.getElementById("r-nu").textContent = r.nu.toFixed(4);

  const warn = document.getElementById("r-warning");
  if (r.physical === false) {
    warn.textContent = `Non-physical: ${r.physical_reason}`;
    warn.hidden = false;
  } else {
    warn.hidden = true;
  }

  document.getElementById("r-log").textContent =
    (body.log_lines || []).join("\n");
  document.getElementById("r-thermo").textContent =
    (body.thermo || []).slice(-50)
      .map((t) => `step=${t.step} pxx=${t.pxx.toFixed(2)} `
                + `pyy=${t.pyy.toFixed(2)} pzz=${t.pzz.toFixed(2)}`)
      .join("\n");

  const img = document.getElementById("r-render");
  if (body.render_available && body.cache_key) {
    img.src = `/api/renders/${body.cache_key}?t=${Date.now()}`;
    img.hidden = false;
  } else {
    img.hidden = true;
  }
}

async function pollJob(jobId, deadlineMs) {
  const status = document.getElementById("status");
  while (Date.now() < deadlineMs) {
    const resp = await fetch(`/api/jobs/${jobId}`);
    if (!resp.ok) {
      status.textContent = `polling failed: HTTP ${resp.status}`;
      return;
    }
    const body = await resp.json();
    const lastThermo = (body.thermo || []).slice(-1)[0];
    const stepInfo = lastThermo ? ` — last step ${lastThermo.step}` : "";
    status.textContent = `status: ${body.status}${stepInfo}`;
    if (body.status === "done") {
      status.textContent = "done";
      renderResult(body);
      return;
    }
    if (body.status === "error") {
      status.textContent = `error: ${body.error.message}`;
      return;
    }
    await new Promise((res) => setTimeout(res, 1000));
  }
  status.textContent = "still running — refresh to keep polling";
}

document.getElementById("predict-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  document.getElementById("result").hidden = true;
  const status = document.getElementById("status");
  status.textContent = "submitting…";
  const payload = collectPayload();
  const resp = await fetch("/api/predict", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await resp.json();
  if (!resp.ok) {
    status.textContent = `error: ${body.error || resp.status}`;
    return;
  }
  if (body.cached) {
    status.textContent = "cache hit";
    renderResult({ result: body.result, log_lines: [], thermo: [],
                   render_available: body.render_available,
                   cache_key: body.cache_key });
    return;
  }
  status.textContent = `queued (job ${body.job_id})`;
  await pollJob(body.job_id, Date.now() + 5 * 60 * 1000);
});
