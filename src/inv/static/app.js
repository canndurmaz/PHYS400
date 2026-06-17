// ─── Static lookups ─────────────────────────────────────────────────
const FAMILY = {
  Mg: 'ae', Al: 'pt',
  Co: 'tm', Cr: 'tm', Cu: 'tm', Fe: 'tm',
  Mn: 'tm', Mo: 'tm', Ni: 'tm', Si: 'pt', Ti: 'tm', Zn: 'tm',
};

const TARGET_PRESETS = {
  al:        { E: 70,  nu: 0.33 },
  stiff:     { E: 110, nu: 0.30 },
  compliant: { E: 50,  nu: 0.36 },
};

// (E, ν) ground-truth cloud emitted by Flask.
const CLOUD = (() => {
  try { return JSON.parse(document.getElementById('cloud-data').textContent || '[]'); }
  catch { return []; }
})();

// ─── DOM ─────────────────────────────────────────────────────────────
const form      = document.getElementById('suggest-form');
const btn       = document.getElementById('btn-suggest');
const btnClear  = document.getElementById('btn-clear');
const presetSel = document.getElementById('preset-select');
const inE       = document.getElementById('in-E');
const inNu      = document.getElementById('in-nu');
const refineTog = document.getElementById('refine-toggle');
const alDomTog  = document.getElementById('al-dominant-toggle');
const alMinEl   = document.getElementById('al-min');
const maxElEl   = document.getElementById('max-el');
const noticeEl  = document.getElementById('notice');
const empty     = document.getElementById('empty-state');
const readout   = document.getElementById('readout');
const candsEl   = document.getElementById('cands');
const rawJsonEl = document.getElementById('raw-json');

// ─── Constraint cells: click cycles free → require → forbid → free ───
const CST_NEXT = { free: 'require', require: 'forbid', forbid: 'free' };
const CST_TAG  = { free: '', require: '✓ req', forbid: '✕ no' };
document.querySelectorAll('.cst').forEach(cell => {
  const tag = cell.querySelector('[data-tag]');
  cell.addEventListener('click', () => {
    const next = CST_NEXT[cell.dataset.state] || 'free';
    cell.dataset.state = next;
    tag.textContent = CST_TAG[next];
  });
});
function collectConstraints() {
  const forbid = [], require = [];
  document.querySelectorAll('.cst').forEach(cell => {
    if (cell.dataset.state === 'forbid') forbid.push(cell.dataset.el);
    else if (cell.dataset.state === 'require') require.push(cell.dataset.el);
  });
  return { forbid, require };
}

// ─── Presets ────────────────────────────────────────────────────────
presetSel.addEventListener('change', () => {
  const p = TARGET_PRESETS[presetSel.value];
  if (p) { inE.value = p.E; inNu.value = p.nu; }
  presetSel.value = '';
  hideNotice();
});

// ─── Clear ──────────────────────────────────────────────────────────
btnClear.addEventListener('click', () => {
  inE.value = ''; inNu.value = '';
  document.querySelectorAll('.cst').forEach(c => {
    c.dataset.state = 'free';
    c.querySelector('[data-tag]').textContent = '';
  });
  alDomTog.checked = false; alMinEl.value = 50; maxElEl.value = '';
  hideNotice();
  empty.hidden = false; readout.hidden = true;
  inE.focus();
});

// ─── Notices ────────────────────────────────────────────────────────
function showNotice(msg, kind = 'warn') {
  noticeEl.hidden = false;
  noticeEl.className = `notice is-${kind}`;
  noticeEl.textContent = msg;
}
function hideNotice() {
  noticeEl.hidden = true;
  noticeEl.className = 'notice';
  noticeEl.textContent = '';
}

// ─── Submit ─────────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideNotice();

  const E = parseFloat(inE.value), nu = parseFloat(inNu.value);
  if (!Number.isFinite(E) || E <= 0) { showNotice('Enter a positive E (GPa).', 'err'); return; }
  if (!Number.isFinite(nu) || nu <= 0 || nu >= 0.5) { showNotice('ν must be in (0, 0.5).', 'err'); return; }

  const { forbid, require } = collectConstraints();
  if (alDomTog.checked && forbid.includes('Al')) {
    showNotice('Al-dominant conflicts with forbidding Al.', 'err'); return;
  }
  const maxEl = parseInt(maxElEl.value, 10);
  const alMin = parseFloat(alMinEl.value);
  const payload = {
    E_GPa: E, nu, forbid, require, refine: refineTog.checked, k: 6,
    al_dominant: alDomTog.checked,
    al_min: Number.isFinite(alMin) ? alMin / 100 : 0.5,
  };
  if (Number.isFinite(maxEl) && maxEl >= 1) payload.max_elements = maxEl;

  btn.classList.add('is-loading'); btn.disabled = true;
  try {
    const r = await fetch('/api/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) { showNotice(data.error || `Server returned ${r.status}.`, 'err'); return; }
    renderResult(data);
  } catch (err) {
    showNotice(`Network error: ${err.message}`, 'err');
  } finally {
    btn.classList.remove('is-loading'); btn.disabled = false;
  }
});
form.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); form.requestSubmit(); }
});

// ─── Render ─────────────────────────────────────────────────────────
const ACH_COPY = {
  in:      { label: 'achievable',          hint: 'target inside training cloud' },
  edge:    { label: 'near the boundary',   hint: 'sparse region' },
  out:     { label: 'likely unachievable', hint: 'beyond training data' },
  unknown: { label: 'unknown',             hint: '' },
};

function fmtFormula(comp) {
  const sub = (s) => s.replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[+d]).replace(/\./g, '.');
  return Object.entries(comp).sort((a, b) => b[1] - a[1])
    .map(([el, f]) => `${el}${sub(f.toFixed(f >= 0.1 ? 2 : 3))}`).join(' ');
}

function renderResult(data) {
  empty.hidden = true; readout.hidden = false;

  // Achievability pill
  const a = data.achievability || {};
  const cls = ACH_COPY[a.class] ? a.class : 'unknown';
  const pill = document.getElementById('ach-pill');
  pill.dataset.class = cls;
  document.getElementById('ach-label').textContent = ACH_COPY[cls].label;
  const dHint = Number.isFinite(a.nearest_dist) ? `d=${a.nearest_dist}` : '';
  document.getElementById('ach-hint').textContent = [ACH_COPY[cls].hint, dHint].filter(Boolean).join(' · ');
  document.getElementById('n-eval').textContent = `${data.n_evaluated} candidates`;

  // Candidate cards
  const cands = data.candidates || [];
  if (!cands.length) {
    candsEl.innerHTML = '<div class="cand cand--empty">No candidate satisfied the constraints. Try relaxing them.</div>';
  } else {
    candsEl.innerHTML = cands.map((c, i) => {
      const fam = FAMILY[Object.entries(c.composition).sort((a, b) => b[1] - a[1])[0][0]] || 'tm';
      const sigma = (c.E_GPa_std > 0) ? ` <span class="cand__sig">±${c.E_GPa_std.toFixed(1)}</span>` : '';
      const bars = Object.entries(c.composition).sort((a, b) => b[1] - a[1]).map(([el, f]) =>
        `<span class="cand__seg" data-fam="${FAMILY[el] || 'tm'}" style="width:${(f * 100).toFixed(1)}%" title="${el}: ${(f * 100).toFixed(1)}%"></span>`
      ).join('');
      return `
        <article class="cand" data-fam="${fam}" style="--i:${i}">
          <div class="cand__rank">#${i + 1}</div>
          <div class="cand__main">
            <div class="cand__formula">${fmtFormula(c.composition)}</div>
            <div class="cand__bar">${bars}</div>
            <div class="cand__props">
              <span>E <b>${c.E_GPa.toFixed(1)}</b> GPa${sigma}
                <span class="cand__err ${c.e_err_pct <= 10 ? 'ok' : (c.e_err_pct <= 25 ? 'warn' : 'bad')}">${c.e_err_pct}%</span></span>
              <span>ν <b>${c.nu.toFixed(3)}</b>
                <span class="cand__err ${c.nu_err_pct <= 10 ? 'ok' : (c.nu_err_pct <= 25 ? 'warn' : 'bad')}">${c.nu_err_pct}%</span></span>
              <span class="cand__src cand__src--${c.source === 'retrieval' ? 'ret' : 'net'}">${c.source}</span>
            </div>
          </div>
        </article>`;
    }).join('');
  }

  renderAshby(data.target, cands);
  rawJsonEl.textContent = JSON.stringify(data, null, 2);

  if (cls === 'out') {
    showNotice('Target sits outside the trained alloy universe — suggestions are extrapolations; treat with caution.', 'warn');
  }
}

// ─── Ashby (ν, E) plot ──────────────────────────────────────────────
const ASHBY = {
  vb: { w: 520, h: 280 }, pad: { l: 44, r: 14, t: 14, b: 30 },
  dom: { nuMin: 0.10, nuMax: 0.45, eMin: 0, eMax: 320 },
};
function ashbyX(nu) {
  const { l, r } = ASHBY.pad, w = ASHBY.vb.w, { nuMin, nuMax } = ASHBY.dom;
  return l + (Math.max(nuMin, Math.min(nuMax, nu)) - nuMin) / (nuMax - nuMin) * (w - l - r);
}
function ashbyY(e) {
  const { t, b } = ASHBY.pad, h = ASHBY.vb.h, { eMin, eMax } = ASHBY.dom;
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
  for (let nu = 0.15; nu <= 0.45 + 1e-9; nu += 0.05) {
    const x = ashbyX(nu);
    grid.appendChild(svgEl('line', { x1: x, x2: x, y1: ASHBY.pad.t, y2: ASHBY.vb.h - ASHBY.pad.b,
      class: Math.abs(nu - 0.30) < 1e-6 ? 'major' : '' }));
    axes.appendChild(svgEl('text', { x, y: ASHBY.vb.h - ASHBY.pad.b + 14, 'text-anchor': 'middle' }))
      .textContent = nu.toFixed(2);
  }
  for (let e = 0; e <= 300; e += 50) {
    const y = ashbyY(e);
    grid.appendChild(svgEl('line', { x1: ASHBY.pad.l, x2: ASHBY.vb.w - ASHBY.pad.r, y1: y, y2: y,
      class: e === 100 ? 'major' : '' }));
    axes.appendChild(svgEl('text', { x: ASHBY.pad.l - 6, y: y + 3, 'text-anchor': 'end' })).textContent = e;
  }
  const xl = svgEl('text', { x: (ASHBY.pad.l + ASHBY.vb.w - ASHBY.pad.r) / 2, y: ASHBY.vb.h - 4,
    'text-anchor': 'middle', class: 'axis-label' });
  xl.textContent = 'ν'; axes.appendChild(xl);
  const yc = (ASHBY.pad.t + ASHBY.vb.h - ASHBY.pad.b) / 2;
  const yl = svgEl('text', { x: 12, y: yc, 'text-anchor': 'middle', class: 'axis-label',
    transform: `rotate(-90, 12, ${yc})` });
  yl.textContent = 'E (GPa)'; axes.appendChild(yl);
  const frag = document.createDocumentFragment();
  for (const p of CLOUD) frag.appendChild(svgEl('circle', { cx: ashbyX(p.nu), cy: ashbyY(p.E), r: 1.3 }));
  cloud.appendChild(frag);
  const nLbl = document.getElementById('ashby-n');
  if (nLbl) nLbl.textContent = CLOUD.length;
  _ashbyDrawn = true;
}
function renderAshby(target, cands) {
  drawAshbyAxes();
  const m = document.getElementById('ashby-marker');
  if (!m) return;
  m.innerHTML = '';
  const tx = ashbyX(target.nu), ty = ashbyY(target.E_GPa);
  // Candidate landings (where the forward model says each lands).
  for (const c of cands) {
    m.appendChild(svgEl('circle', { cx: ashbyX(c.nu), cy: ashbyY(c.E_GPa), r: 3, class: 'ashby-cand' }));
    m.appendChild(svgEl('line', { x1: tx, y1: ty, x2: ashbyX(c.nu), y2: ashbyY(c.E_GPa), class: 'ashby-link' }));
  }
  // Target crosshair + marker.
  m.appendChild(svgEl('line', { x1: ASHBY.pad.l, x2: ASHBY.vb.w - ASHBY.pad.r, y1: ty, y2: ty,
    class: 'ashby-cross', 'stroke-dasharray': '2 4' }));
  m.appendChild(svgEl('line', { x1: tx, x2: tx, y1: ASHBY.pad.t, y2: ASHBY.vb.h - ASHBY.pad.b,
    class: 'ashby-cross', 'stroke-dasharray': '2 4' }));
  m.appendChild(svgEl('path', {
    d: `M ${tx} ${ty - 6} L ${tx + 6} ${ty} L ${tx} ${ty + 6} L ${tx - 6} ${ty} Z`,
    class: 'ashby-target' }));
  const lbl = svgEl('text', { x: tx + 8, y: ty - 8 });
  lbl.textContent = `target E ${target.E_GPa.toFixed(0)} · ν ${target.nu.toFixed(3)}`;
  m.appendChild(lbl);
}

// ─── Theme toggle ───────────────────────────────────────────────────
const themeBtn = document.getElementById('theme-toggle');
if (themeBtn) {
  themeBtn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    const next = cur === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) {}
  });
}
