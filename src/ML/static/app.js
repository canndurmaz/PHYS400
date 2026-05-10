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

// (E, ν) ground-truth cloud rendered as the Ashby backdrop. Decimated and
// emitted by Flask alongside the calibration block.
const CLOUD = (() => {
  try { return JSON.parse(document.getElementById('cloud-data').textContent || '[]'); }
  catch { return []; }
})();

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

// ─── Color helpers (theme-aware family blending) ────────────────────
function readFamilyRgb() {
  const cs = getComputedStyle(document.documentElement);
  const out = {};
  for (const fam of ['ae', 'pt', 'tm']) {
    const hex = cs.getPropertyValue(`--fam-${fam}`).trim();
    out[fam] = hexToRgb(hex);
  }
  return out;
}
function hexToRgb(hex) {
  let h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map(c => c + c).join('');
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rgbToHex([r, g, b]) {
  const c = (n) => Math.round(Math.max(0, Math.min(255, n))).toString(16).padStart(2, '0');
  return '#' + c(r) + c(g) + c(b);
}
function blendComposition(comp) {
  const rgb = readFamilyRgb();
  let R = 0, G = 0, B = 0, total = 0;
  for (const [el, frac] of Object.entries(comp)) {
    const fam = FAMILY[el];
    if (!fam || !(frac > 0)) continue;
    const [fr, fg, fb] = rgb[fam];
    R += fr * frac; G += fg * frac; B += fb * frac; total += frac;
  }
  if (total === 0) return null;
  return [R / total, G / total, B / total];
}
function lighten([r, g, b], amt) {
  return [r + (255 - r) * amt, g + (255 - g) * amt, b + (255 - b) * amt];
}
function darken([r, g, b], amt) {
  return [r * (1 - amt), g * (1 - amt), b * (1 - amt)];
}

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

  // Live alloy preview keeps the unit cell + legend in sync as the user types.
  const compNorm = total > 0
    ? Object.fromEntries(active.map(({el, frac}) => [el, frac / total]))
    : {};
  updateAlloyPreview(compNorm);
}

// ─── Alloy preview (composition panel) ──────────────────────────────
const previewWrap     = document.getElementById('alloy-preview');
const previewFormula  = document.getElementById('alloy-formula');
const previewLegend   = document.getElementById('alloy-legend');
const previewStop0    = document.getElementById('atom-blend-0');
const previewStop1    = document.getElementById('atom-blend-1');
const previewStop2    = document.getElementById('atom-blend-2');

function formatFormula(comp) {
  // Render entries in descending fraction with subscript fractions, e.g.
  // "Al₀.₉₀ Mg₀.₁₀" — capped to top 4 elements with an ellipsis tail.
  const sub = (s) => s.replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[+d]).replace(/\./g, '.');
  const sorted = Object.entries(comp).sort((a, b) => b[1] - a[1]);
  const top = sorted.slice(0, 4)
    .map(([el, f]) => `${el}${sub(f.toFixed(f >= 0.1 ? 2 : 3))}`)
    .join(' ');
  return sorted.length > 4 ? `${top} …` : top;
}

function updateAlloyPreview(comp) {
  const entries = Object.entries(comp).filter(([, f]) => f > 0);
  if (entries.length === 0) {
    previewWrap.classList.add('is-empty');
    previewFormula.textContent = 'awaiting input';
    previewLegend.innerHTML = '';
    // Reset gradient stops to a neutral grey so the unit cell still reads.
    if (previewStop0) {
      previewStop0.setAttribute('stop-color', getComputedStyle(document.documentElement).getPropertyValue('--ink-fade').trim());
      previewStop1.setAttribute('stop-color', getComputedStyle(document.documentElement).getPropertyValue('--ink-ghost').trim());
      previewStop2.setAttribute('stop-color', getComputedStyle(document.documentElement).getPropertyValue('--bg-deep').trim());
    }
    return;
  }
  previewWrap.classList.remove('is-empty');
  previewFormula.textContent = formatFormula(comp);

  // Compute the average atom color via family-weighted RGB blend, then derive
  // a 3-stop radial gradient (highlight → mid → shadow) for the SVG atoms.
  const mid = blendComposition(comp);
  if (mid && previewStop0) {
    previewStop0.setAttribute('stop-color', rgbToHex(lighten(mid, 0.35)));
    previewStop1.setAttribute('stop-color', rgbToHex(mid));
    previewStop2.setAttribute('stop-color', rgbToHex(darken(mid, 0.65)));
  }

  // Legend: top 5 by fraction, family-color swatch + percent.
  const cs = getComputedStyle(document.documentElement);
  const famColor = (fam) => cs.getPropertyValue(`--fam-${fam}`).trim();
  const top = entries.sort((a, b) => b[1] - a[1]).slice(0, 5);
  previewLegend.innerHTML = top.map(([el, f]) => {
    const fam = FAMILY[el] || 'tm';
    return `<li><span class="swatch" style="--swatch:${famColor(fam)}"></span>${el}<span class="pct">&nbsp;${(f * 100).toFixed(f >= 0.1 ? 1 : 2)}%</span></li>`;
  }).join('');
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
  readout.querySelectorAll('.stat, .tornado__row').forEach(el => {
    el.style.animation = 'none';
    void el.offsetWidth;          // force reflow
    el.style.animation = '';
  });

  setStat('E',   data.E_GPa,   2);
  setStat('nu',  data.nu,      3);
  setStat('C11', data.C11_GPa, 2);
  setStat('C12', data.C12_GPa, 2);

  renderAshby(data.E_GPa, data.nu);
  renderTornado(data.sensitivity || []);

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

// ─── Ashby (ν, E) plot ──────────────────────────────────────────────
const ASHBY = {
  vb: { w: 520, h: 280 },
  pad: { l: 44, r: 14, t: 14, b: 30 },
  // Domain ranges chosen to enclose the cloud with a little headroom.
  dom: { nuMin: 0.10, nuMax: 0.45, eMin: 0, eMax: 320 },
};
function ashbyX(nu) {
  const { l, r } = ASHBY.pad, w = ASHBY.vb.w;
  const { nuMin, nuMax } = ASHBY.dom;
  return l + (Math.max(nuMin, Math.min(nuMax, nu)) - nuMin) / (nuMax - nuMin) * (w - l - r);
}
function ashbyY(e) {
  const { t, b } = ASHBY.pad, h = ASHBY.vb.h;
  const { eMin, eMax } = ASHBY.dom;
  return h - b - (Math.max(eMin, Math.min(eMax, e)) - eMin) / (eMax - eMin) * (h - t - b);
}
function svgEl(name, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}
let _ashbyDrawn = false;
function drawAshbyAxes() {
  if (_ashbyDrawn) return;
  const grid = document.getElementById('ashby-grid');
  const axes = document.getElementById('ashby-axes');
  const cloud = document.getElementById('ashby-cloud');
  if (!grid || !axes || !cloud) return;

  // Vertical gridlines at ν = 0.15, 0.20, ..., 0.45
  for (let nu = 0.15; nu <= 0.45 + 1e-9; nu += 0.05) {
    const x = ashbyX(nu);
    const major = Math.abs(nu - 0.30) < 1e-6;
    grid.appendChild(svgEl('line', {
      x1: x, x2: x, y1: ASHBY.pad.t, y2: ASHBY.vb.h - ASHBY.pad.b,
      class: major ? 'major' : '',
    }));
    axes.appendChild(svgEl('text', {
      x: x, y: ASHBY.vb.h - ASHBY.pad.b + 14, 'text-anchor': 'middle',
    })).textContent = nu.toFixed(2);
  }
  // Horizontal gridlines at E = 0, 50, 100, ..., 300
  for (let e = 0; e <= 300; e += 50) {
    const y = ashbyY(e);
    grid.appendChild(svgEl('line', {
      x1: ASHBY.pad.l, x2: ASHBY.vb.w - ASHBY.pad.r, y1: y, y2: y,
      class: e === 100 ? 'major' : '',
    }));
    axes.appendChild(svgEl('text', {
      x: ASHBY.pad.l - 6, y: y + 3, 'text-anchor': 'end',
    })).textContent = e;
  }
  // Axis labels
  const xl = svgEl('text', {
    x: (ASHBY.pad.l + ASHBY.vb.w - ASHBY.pad.r) / 2,
    y: ASHBY.vb.h - 4, 'text-anchor': 'middle', class: 'axis-label',
  });
  xl.textContent = 'ν';
  axes.appendChild(xl);
  const yl = svgEl('text', {
    x: 12, y: (ASHBY.pad.t + ASHBY.vb.h - ASHBY.pad.b) / 2,
    'text-anchor': 'middle', class: 'axis-label',
    transform: `rotate(-90, 12, ${(ASHBY.pad.t + ASHBY.vb.h - ASHBY.pad.b) / 2})`,
  });
  yl.textContent = 'E (GPa)';
  axes.appendChild(yl);

  // Cloud — drawn once, doesn't change between predictions.
  const frag = document.createDocumentFragment();
  for (const p of CLOUD) {
    frag.appendChild(svgEl('circle', { cx: ashbyX(p.nu), cy: ashbyY(p.E), r: 1.3 }));
  }
  cloud.appendChild(frag);
  const nLbl = document.getElementById('ashby-n');
  if (nLbl) nLbl.textContent = CLOUD.length;
  _ashbyDrawn = true;
}
function renderAshby(E, nu) {
  drawAshbyAxes();
  const m = document.getElementById('ashby-marker');
  if (!m) return;
  const x = ashbyX(nu), y = ashbyY(E);
  // A ±1σ confidence ellipse around the marker, derived from validation Std.
  const stdE  = (VAL_STATS.E  && VAL_STATS.E.Std)  || 0;
  const stdNu = (VAL_STATS.nu && VAL_STATS.nu.Std) || 0;
  const rx = Math.abs(ashbyX(nu + stdNu) - x);
  const ry = Math.abs(y - ashbyY(E + stdE));
  m.innerHTML = '';
  if (rx > 0 && ry > 0) {
    m.appendChild(svgEl('ellipse', { cx: x, cy: y, rx, ry, class: 'ashby-band' }));
  }
  // Crosshair lines stretching across the panel for instant readability.
  m.appendChild(svgEl('line', {
    x1: ASHBY.pad.l, x2: ASHBY.vb.w - ASHBY.pad.r, y1: y, y2: y,
    class: 'ashby-cross', 'stroke-dasharray': '2 4',
  }));
  m.appendChild(svgEl('line', {
    x1: x, x2: x, y1: ASHBY.pad.t, y2: ASHBY.vb.h - ASHBY.pad.b,
    class: 'ashby-cross', 'stroke-dasharray': '2 4',
  }));
  m.appendChild(svgEl('circle', { cx: x, cy: y, r: 4, class: 'ashby-dot' }));
  const lbl = svgEl('text', {
    x: x + 8, y: y - 8,
  });
  lbl.textContent = `E ${E.toFixed(0)} · ν ${nu.toFixed(3)}`;
  m.appendChild(lbl);
}

// ─── Sensitivity tornado ────────────────────────────────────────────
function renderTornado(rows) {
  const t = document.getElementById('tornado');
  if (!t) return;
  if (!rows.length) {
    t.innerHTML = '<div class="tornado__empty">composition needs at least 2 elements for sensitivity analysis.</div>';
    return;
  }
  // Symmetric scale: pick the largest |ΔE| across all rows so bars are
  // comparable. A small floor avoids a divide-by-zero on flat sensitivity.
  const maxAbs = Math.max(0.5, ...rows.map(r => Math.max(Math.abs(r.dE_plus), Math.abs(r.dE_minus))));
  const cs = getComputedStyle(document.documentElement);
  const famColor = (fam) => cs.getPropertyValue(`--fam-${fam}`).trim();

  // For each element we draw a single horizontal range bar from min(ΔE) to
  // max(ΔE), centered on the zero-axis. If both perturbations push E in the
  // same direction, the bar sits entirely on one side; opposing pushes
  // produce a bar that straddles zero. Endpoint labels call out each
  // perturbation's signed ΔE.
  t.innerHTML = rows.map(r => {
    const fam = FAMILY[r.element] || 'tm';
    const seg = famColor(fam);
    const lo = Math.min(r.dE_plus, r.dE_minus);
    const hi = Math.max(r.dE_plus, r.dE_minus);
    const halfPct = 50;                                        // 50% = full half-width
    const left  = halfPct + (lo / maxAbs) * halfPct;           // % from container left
    const width = ((hi - lo) / maxAbs) * halfPct;
    const sign = (n) => (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(2);
    return `
      <div class="tornado__row">
        <span class="tornado__el">${r.element}<span class="frac">${(r.frac*100).toFixed(1)}%</span></span>
        <span class="tornado__bar">
          <span class="tornado__seg" style="left:${left.toFixed(2)}%; width:${width.toFixed(2)}%; --seg:${seg}"></span>
          <span class="tornado__cap" style="left:${left.toFixed(2)}%; --seg:${seg}"></span>
          <span class="tornado__cap" style="left:${(left + width).toFixed(2)}%; --seg:${seg}"></span>
        </span>
        <span class="tornado__delta">${sign(hi)}<span class="pm">${sign(lo)}&nbsp;GPa</span></span>
      </div>`;
  }).join('');
}

// Convenience: ⌘/Ctrl+Enter from any input submits the form.
form.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    form.requestSubmit();
  }
});

// ─── Theme toggle ───────────────────────────────────────────────────
// The initial data-theme attribute is set by an inline <head> script to
// avoid a flash. This handler just flips and persists.
const themeBtn = document.getElementById('theme-toggle');
if (themeBtn) {
  themeBtn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    const next = cur === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) {}
  });
}
