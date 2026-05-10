// ─── Static lookups ─────────────────────────────────────────────────
const ATOMIC = {
  Al: 13, Co: 27, Cr: 24, Cu: 29, Fe: 26,
  Mg: 12, Mn: 25, Ni: 28, Ti: 22, Zn: 30,
};

// Periodic-table family per element (also encoded in CSS via [data-el]
// selectors; kept here so the stacked-bar segments can pick the matching
// hue without round-tripping through computed styles).
const FAMILY = {
  Mg: 'ae', Al: 'pt',
  Co: 'tm', Cr: 'tm', Cu: 'tm', Fe: 'tm',
  Mn: 'tm', Ni: 'tm', Ti: 'tm', Zn: 'tm',
};
const FAMILY_COLOR = { ae: '#d4c378', pt: '#8aabc5', tm: '#c98ba0' };

const PRESETS = {
  pure_al: { Al: 1.0 },
  aa7075:  { Al: 0.90, Zn: 0.057, Mg: 0.025, Cu: 0.016, Cr: 0.002 },
  al_cu:   { Al: 0.94, Cu: 0.06 },
  hea5:    { Al: 0.20, Cr: 0.20, Fe: 0.20, Ni: 0.20, Ti: 0.20 },
};

// Domain ranges for the axis markers. Picked to enclose the realistic span
// across the trained alloy basis (mean ± ~2σ from `nn_metrics.json`).
const RANGES = {
  E:   { min: 40,  max: 180 },
  nu:  { min: 0.0, max: 0.5 },
  C11: { min: 30,  max: 400 },
  C12: { min: 0,   max: 250 },
};

// Calibration block emitted by Flask into a <script type="application/json">
// tag. Used to draw the ±1σ band on each axis using the validation `Std`.
const CAL = (() => {
  try { return JSON.parse(document.getElementById('cal-data').textContent || '{}'); }
  catch { return {}; }
})();
const VAL_STATS = (CAL && CAL.val) || {};

// ─── DOM bootstrap ─────────────────────────────────────────────────
const form        = document.getElementById('predict-form');
const btn         = document.getElementById('btn-predict');
const btnClear    = document.getElementById('btn-clear');
const presetSel   = document.getElementById('preset-select');
const sumEl       = document.getElementById('sum-value');
const sumDeltaEl  = document.getElementById('sum-delta');
const stackSegs   = document.getElementById('stack-segments');
const sumWrap     = document.querySelector('.sum');
const noticeEl    = document.getElementById('notice');
const empty       = document.getElementById('empty-state');
const readout     = document.getElementById('readout');
const rawJsonEl   = document.getElementById('raw-json');

// Stamp atomic numbers into each cell label.
document.querySelectorAll('.cell').forEach(cell => {
  const el = cell.dataset.el;
  const numEl = cell.querySelector('[data-an]');
  if (numEl && ATOMIC[el] != null) numEl.textContent = ATOMIC[el];
});

const inputs = Array.from(form.querySelectorAll('.cell__input'));

// ─── Composition meter (per-cell fill + stacked bar) ────────────────
function recomputeSum() {
  // First pass: collect active fractions and update each cell's bottom fill.
  const active = [];   // [{el, frac}]
  let total = 0;
  inputs.forEach(inp => {
    const v = parseFloat(inp.value);
    const cell = inp.closest('.cell');
    const fill = cell.querySelector('.cell__fill');
    if (Number.isFinite(v) && v > 0) {
      total += v;
      active.push({ el: inp.name, frac: v });
      cell.classList.add('is-active');
      // Per-cell bar shows the fraction itself, 0–1 → 0–100%. Capped.
      fill.style.width = Math.min(100, v * 100).toFixed(1) + '%';
    } else {
      cell.classList.remove('is-active');
      fill.style.width = '0%';
    }
  });

  // Numeric Σ + Δ.
  sumEl.textContent = total.toFixed(4);
  const delta = total - 1.0;
  sumDeltaEl.textContent = `Δ ${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(4)}`;

  // Stacked bar. Segments occupy a 0..1 normalised space — i.e. each segment
  // takes (frac / max(1, total)) of the bar width. Sub-1.0 sums leave empty
  // headroom on the right (toward the 1.0 tick); over-1.0 sums saturate.
  const denom = Math.max(1.0, total);
  // Reuse existing segment nodes when possible to keep CSS transitions alive.
  const want = active.length;
  while (stackSegs.children.length < want) stackSegs.appendChild(document.createElement('span'));
  while (stackSegs.children.length > want) stackSegs.lastChild.remove();
  active.forEach(({ el, frac }, i) => {
    const seg = stackSegs.children[i];
    seg.className = 'seg';
    const fam = FAMILY[el] || 'tm';
    seg.style.setProperty('--seg-color', FAMILY_COLOR[fam]);
    seg.style.width = ((frac / denom) * 100).toFixed(2) + '%';
    seg.title = `${el}: ${(frac * 100).toFixed(2)}%`;
  });

  // Status modifier: green if Σ≈1, amber if within 0.5–1.5, red otherwise.
  sumWrap.classList.remove('is-ok', 'is-warn', 'is-err');
  if (total <= 0)                       { /* idle, no class */ }
  else if (Math.abs(delta) <= 1e-3)     sumWrap.classList.add('is-ok');
  else if (total >= 0.5 && total <= 1.5) sumWrap.classList.add('is-warn');
  else                                   sumWrap.classList.add('is-err');
}
inputs.forEach(inp => inp.addEventListener('input', recomputeSum));
recomputeSum();

// ─── Preset loader ──────────────────────────────────────────────────
presetSel.addEventListener('change', () => {
  const key = presetSel.value;
  if (!key) return;
  const comp = PRESETS[key] || {};
  inputs.forEach(inp => {
    const v = comp[inp.name];
    inp.value = (v != null && v > 0) ? v : '';
  });
  recomputeSum();
  hideNotice();
  presetSel.value = '';
});

// ─── Clear ──────────────────────────────────────────────────────────
btnClear.addEventListener('click', () => {
  inputs.forEach(inp => inp.value = '');
  recomputeSum();
  hideNotice();
  inputs[0]?.focus();
});

// ─── Notices ────────────────────────────────────────────────────────
function showNotice(message, kind = 'warn') {
  noticeEl.hidden = false;
  noticeEl.className = `notice is-${kind}`;
  noticeEl.textContent = message;
}
function hideNotice() {
  noticeEl.hidden = true;
  noticeEl.className = 'notice';
  noticeEl.textContent = '';
}

// ─── Submit / fetch ─────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideNotice();

  const composition = {};
  inputs.forEach(inp => {
    const v = parseFloat(inp.value);
    if (Number.isFinite(v) && v > 0) composition[inp.name] = v;
  });

  if (Object.keys(composition).length === 0) {
    showNotice('Enter at least one element with a positive fraction.', 'err');
    return;
  }

  btn.classList.add('is-loading');
  btn.disabled = true;

  try {
    const r = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ composition }),
    });
    const data = await r.json();
    if (!r.ok) {
      showNotice(data.error || `Server returned ${r.status}.`, 'err');
      return;
    }
    renderResult(data);
  } catch (err) {
    showNotice(`Network error: ${err.message}`, 'err');
  } finally {
    btn.classList.remove('is-loading');
    btn.disabled = false;
  }
});

// ─── Render ─────────────────────────────────────────────────────────
function fmt(v, digits = 2) {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(digits);
}

function setStat(key, value, digits) {
  const el = document.getElementById('val-' + key);
  if (!el) return;
  el.textContent = fmt(value, digits);

  const card = el.closest('.stat');
  const axis = card?.querySelector('.stat__axis');
  if (!axis || !RANGES[key]) return;

  const { min, max } = RANGES[key];
  const span = max - min;
  const clamped = Math.max(min, Math.min(max, value));
  const pct = ((clamped - min) / span) * 100;
  axis.style.setProperty('--mark-pos', pct + '%');

  // ±1σ confidence band, derived from the validation Std for this target.
  const std = VAL_STATS[key]?.Std;
  if (Number.isFinite(std)) {
    const bandW = Math.min(100, (2 * std / span) * 100);
    axis.style.setProperty('--band-w', bandW.toFixed(2) + '%');
  }
  card.classList.add('has-result');
}

function renderResult(data) {
  empty.hidden = true;
  readout.hidden = false;

  // Re-trigger the staggered animations on each update.
  readout.querySelectorAll('.stat').forEach(el => {
    el.style.animation = 'none';
    void el.offsetWidth;          // force reflow
    el.style.animation = '';
  });

  setStat('E',   data.E_GPa,   2);
  setStat('nu',  data.nu,      3);
  setStat('C11', data.C11_GPa, 2);
  setStat('C12', data.C12_GPa, 2);

  rawJsonEl.textContent = JSON.stringify(data, null, 2);

  // Surface non-fatal warnings.
  const issues = [];
  if (Array.isArray(data.unknown_elements) && data.unknown_elements.length) {
    issues.push(`Ignored unsupported element(s): ${data.unknown_elements.join(', ')}.`);
  }
  if (typeof data.input_sum === 'number' && Math.abs(data.input_sum - 1.0) > 0.05) {
    issues.push(`Input fractions summed to ${data.input_sum.toFixed(4)}; re-normalised to 1.0 before inference.`);
  }
  if (issues.length) showNotice(issues.join('  '), 'warn');
}

// Convenience: ⌘/Ctrl+Enter from any input submits the form.
form.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    form.requestSubmit();
  }
});
