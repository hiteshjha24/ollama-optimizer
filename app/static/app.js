/* Ollama Optimizer frontend — vanilla JS, no build step, no CDN.
   Hash router + fetch + EventSource (with a polling fallback). */

'use strict';

// ---------------------------------------------------------------- helpers

const $ = (sel, root) => (root || document).querySelector(sel);
const view = () => $('#view');

function h(html) { return html; }

function esc(v) {
  if (v === null || v === undefined) return '';
  return String(v).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** Format a number, or the explicit unavailable marker — never a fake value. */
function num(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return 'N/A';
  return Number(v).toFixed(digits === undefined ? 2 : digits);
}
function stat(block, key, digits) {
  if (!block || block.n === 0 || block[key] === null || block[key] === undefined) return 'N/A';
  return num(block[key], digits);
}
function when(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString();
}
function pct(v, digits) {
  if (v === null || v === undefined) return 'N/A';
  return (v > 0 ? '+' : '') + num(v, digits === undefined ? 1 : digits) + '%';
}

async function api(path, options) {
  const res = await fetch(path, Object.assign({
    headers: { 'Content-Type': 'application/json' }
  }, options || {}));
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = { raw: text }; }
  if (!res.ok) {
    const err = (data && data.error) || {};
    throw new Error([err.message || ('HTTP ' + res.status), err.detail].filter(Boolean).join(' — '));
  }
  return data;
}

let toastTimer = null;
function toast(message, bad) {
  const existing = $('.toast');
  if (existing) existing.remove();
  const el = document.createElement('div');
  el.className = 'toast' + (bad ? ' bad' : '');
  el.innerHTML = esc(message);
  document.body.appendChild(el);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), 6500);
}

function badge(status) {
  const map = { completed: 'ok', running: 'running', queued: 'running',
                failed: 'failed', cancelled: 'cancelled' };
  return `<span class="badge ${map[status] || ''}">${esc(status || 'unknown')}</span>`;
}

// ---------------------------------------------------------------- state

const state = { health: null, models: [], prompts: [], optimizations: null, es: null, poll: null };

function stopStreams() {
  if (state.es) { state.es.close(); state.es = null; }
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
}

async function refreshStatus() {
  const box = $('#ollama-status');
  try {
    const data = await api('/api/health');
    state.health = data;
    $('#app-version').textContent = 'v' + data.app.version;
    const ok = data.ollama.connected;
    box.innerHTML = `<span class="dot ${ok ? 'on' : 'off'}"></span>` +
      (ok ? `Ollama connected<br><span class="faint">version ${esc(data.ollama.version || '?')}</span>`
          : `Ollama not reachable<br><span class="faint">${esc((data.ollama.error || {}).message || '')}</span>`);
  } catch (e) {
    box.innerHTML = `<span class="dot off"></span>API unreachable`;
  }
}

// ---------------------------------------------------------------- router

const routes = {
  dashboard: renderDashboard,
  models: renderModels,
  new: renderNew,
  experiments: renderExperiments,
  experiment: renderExperiment,
  reports: renderReports,
  settings: renderSettings,
};

function currentRoute() {
  const raw = (location.hash || '#/dashboard').replace(/^#\/?/, '');
  const parts = raw.split('/').filter(Boolean);
  return { name: parts[0] || 'dashboard', arg: parts[1] };
}

async function router() {
  stopStreams();
  const { name, arg } = currentRoute();
  document.querySelectorAll('#nav a').forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#/' + name);
  });
  const fn = routes[name] || renderDashboard;
  view().innerHTML = '<p class="muted"><span class="spin"></span>Loading…</p>';
  try {
    await fn(arg);
  } catch (e) {
    view().innerHTML = `<h2>Something went wrong</h2>
      <div class="notice bad">${esc(e.message)}</div>
      <button onclick="location.reload()">Reload</button>`;
  }
}

window.addEventListener('hashchange', router);
window.addEventListener('load', async () => { await refreshStatus(); router(); });

// ---------------------------------------------------------------- dashboard

async function renderDashboard() {
  const data = await api('/api/dashboard');
  const ok = data.ollama.connected;
  const rows = data.recent.map(e => `
    <tr>
      <td><a href="#/experiment/${esc(e.id)}" class="mono">${esc(e.id)}</a></td>
      <td>${esc(e.model)}</td>
      <td>${badge(e.status)}</td>
      <td class="muted">${esc(e.headline || e.error || '')}</td>
      <td class="num muted">${when(e.created_at)}</td>
    </tr>`).join('') ||
    '<tr><td colspan="5" class="muted">No experiments yet. Start one from “New Experiment”.</td></tr>';

  view().innerHTML = `
    <h2>Dashboard</h2>
    <p class="lede">Benchmark and optimization workbench for models served by your local Ollama
      instance. Every number shown anywhere in this app comes from a measured run.</p>

    ${ok ? '' : `<div class="notice bad"><strong>Ollama is not reachable.</strong>
        ${esc((data.ollama.error || {}).message || '')}<br>
        Start it with <code class="mono">ollama serve</code> and reload this page.
        You can change the endpoint under Settings.</div>`}

    <div class="grid cols-4">
      <div class="stat"><div class="k">Ollama</div>
        <div class="v small">${ok ? '● Connected' : '● Offline'}</div></div>
      <div class="stat"><div class="k">Models available</div>
        <div class="v">${data.models_available}</div></div>
      <div class="stat"><div class="k">Experiments</div>
        <div class="v">${data.counts.experiments || 0}</div></div>
      <div class="stat"><div class="k">Runs recorded</div>
        <div class="v">${data.counts.runs || 0}</div></div>
    </div>

    <h3>Recent experiments</h3>
    <div class="panel table-wrap">
      <table><thead><tr><th>ID</th><th>Model</th><th>Status</th><th>Result</th>
        <th class="num">Started</th></tr></thead><tbody>${rows}</tbody></table>
    </div>
    <div class="row">
      <a href="#/new"><button class="primary">New experiment</button></a>
      <a href="#/models"><button>Browse models</button></a>
    </div>`;
}

// ---------------------------------------------------------------- models

async function renderModels() {
  const data = await api('/api/models');
  state.models = data.models || [];
  if (!data.connected) {
    view().innerHTML = `<h2>Models</h2>
      <div class="notice bad"><strong>Ollama is not reachable.</strong>
      ${esc((data.error || {}).message || '')}<br>${esc(data.hint || '')}</div>`;
    return;
  }
  if (!state.models.length) {
    view().innerHTML = `<h2>Models</h2><div class="notice">Ollama is running but no models are
      installed. Pull one first, e.g. <code class="mono">ollama pull llama3.1:8b</code>.</div>`;
    return;
  }
  const cards = state.models.map(m => `
    <div class="panel">
      <h4 class="mono">${esc(m.name)}</h4>
      <dl class="kv">
        <dt>Size</dt><dd>${esc(m.size_human || 'N/A')}</dd>
        <dt>Parameters</dt><dd>${esc(m.parameter_size || 'N/A')}</dd>
        <dt>Quantization</dt><dd>${esc(m.quantization_level || 'N/A')}</dd>
        <dt>Family</dt><dd>${esc(m.family || 'N/A')}</dd>
        <dt>Format</dt><dd>${esc(m.format || 'N/A')}</dd>
        <dt>Modified</dt><dd>${esc(m.modified_at || 'N/A')}</dd>
        <dt>Digest</dt><dd class="faint">${esc((m.digest || '').slice(0, 20))}</dd>
      </dl>
      <div class="row" style="margin-top:10px">
        <button onclick="location.hash='#/new?model=' + encodeURIComponent('${esc(m.name)}')">Benchmark this model</button>
        <button class="link" data-info="${esc(m.name)}">Details</button>
      </div>
      <div class="model-details"></div>
    </div>`).join('');

  view().innerHTML = `<h2>Models</h2>
    <p class="lede">Discovered through the Ollama API (<code class="mono">/api/tags</code>).
      ${data.loaded && data.loaded.length ? 'Currently loaded: ' + esc(data.loaded.join(', ')) : ''}</p>
    <div class="grid cols-2">${cards}</div>`;

  view().querySelectorAll('[data-info]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const box = btn.closest('.panel').querySelector('.model-details');
      box.innerHTML = '<p class="muted"><span class="spin"></span>Reading /api/show…</p>';
      try {
        const info = await api('/api/models/' + encodeURIComponent(btn.dataset.info) + '/info');
        box.innerHTML = `<pre>${esc(JSON.stringify(info, null, 2))}</pre>`;
      } catch (e) { box.innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }
    });
  });
}

// ---------------------------------------------------------------- new experiment

async function renderNew() {
  const [models, prompts, opts, settings] = await Promise.all([
    api('/api/models'), api('/api/prompts'), api('/api/optimizations'), api('/api/settings')
  ]);
  state.prompts = prompts.prompts;
  const s = settings.settings;
  const preselect = decodeURIComponent((location.hash.split('?model=')[1] || ''));

  if (!models.connected || !models.models.length) {
    view().innerHTML = `<h2>New Experiment</h2>
      <div class="notice bad">${esc((models.error || {}).message || 'No models available.')}<br>
      ${esc(models.hint || 'Install a model with `ollama pull <name>`, then reload.')}</div>`;
    return;
  }

  const modelOptions = models.models.map(m =>
    `<option value="${esc(m.name)}" ${m.name === preselect ? 'selected' : ''}>${esc(m.name)}
      — ${esc(m.parameter_size || '?')} / ${esc(m.quantization_level || '?')} / ${esc(m.size_human || '?')}</option>`
  ).join('');

  const strategyOptions = opts.strategies.map(st =>
    `<option value="${esc(st.id)}" ${st.id === 'all' ? 'selected' : ''}>${esc(st.label)}</option>`).join('');

  const objectiveOptions = opts.objectives.map(o =>
    `<option value="${esc(o.id)}" ${o.id === s.default_objective ? 'selected' : ''}>${esc(o.id.replace('_', ' '))}</option>`).join('');

  const chips = prompts.prompts.map(p =>
    `<span class="chip" data-prompt="${esc(p.id)}" title="${esc(p.category)}">${esc(p.label)}</span>`).join('');

  view().innerHTML = `
    <h2>New Experiment</h2>
    <p class="lede">Pick a model, give it a prompt, and press start. Defaults are already sensible:
      all optimization families, ${s.default_runs_per_config} runs per configuration, balanced objective.</p>

    <div class="grid cols-2">
      <div class="panel">
        <label class="field"><span>Select model</span>
          <select id="f-model">${modelOptions}</select></label>
        <div id="model-meta"></div>

        <label class="field"><span>Test prompt</span>
          <textarea id="f-prompt" rows="9"
            placeholder="Enter the prompt you want to benchmark..."></textarea></label>
        <div>
          <span class="mono faint" style="font-size:11px;letter-spacing:.1em;">PREDEFINED BENCHMARK PROMPTS</span>
          <div class="chips">${chips}</div>
          <p class="faint"><small id="prompt-note">Predefined prompts carry checkable expectations,
            which makes the correctness and format scores stronger than for a free-text prompt.</small></p>
        </div>

        <label class="field"><span>System prompt (optional)</span>
          <textarea id="f-system" rows="2" placeholder="Leave empty to use the model's own default."></textarea></label>
      </div>

      <div class="panel">
        <label class="field"><span>Optimization strategy</span>
          <select id="f-strategy">${strategyOptions}</select></label>

        <label class="field"><span>Recommendation objective</span>
          <select id="f-objective">${objectiveOptions}<option value="custom">custom</option></select></label>
        <div id="custom-weights" style="display:none">
          <div class="grid cols-4">
            <label class="field"><span>Quality %</span><input type="number" id="w-quality" value="50" min="0" max="100"></label>
            <label class="field"><span>Speed %</span><input type="number" id="w-speed" value="30" min="0" max="100"></label>
            <label class="field"><span>Consistency %</span><input type="number" id="w-consistency" value="20" min="0" max="100"></label>
            <label class="field"><span>Efficiency %</span><input type="number" id="w-efficiency" value="0" min="0" max="100"></label>
          </div>
        </div>

        <div class="grid cols-3">
          <label class="field"><span>Runs per config</span>
            <input type="number" id="f-runs" min="1" max="25" value="${s.default_runs_per_config}"></label>
          <label class="field"><span>Max tokens</span>
            <input type="number" id="f-max-tokens" min="16" max="8192" value="${s.default_max_tokens}"></label>
          <label class="field"><span>Timeout (s)</span>
            <input type="number" id="f-timeout" min="10" max="1800" value="${s.default_timeout_seconds}"></label>
          <label class="field"><span>Concurrency</span>
            <input type="number" id="f-concurrency" min="1" max="8" value="${s.max_concurrency}"></label>
          <label class="field"><span>Seed</span>
            <input type="number" id="f-seed" value="42"></label>
          <label class="field"><span>Evaluation mode</span>
            <select id="f-evaluator">
              <option value="heuristic" ${s.evaluator_mode === 'heuristic' ? 'selected' : ''}>heuristic (deterministic)</option>
              <option value="heuristic+llm" ${s.evaluator_mode !== 'heuristic' ? 'selected' : ''}>heuristic + LLM judge (estimates)</option>
            </select></label>
        </div>
        <label class="check"><input type="checkbox" id="f-stream" ${s.default_stream ? 'checked' : ''}>
          Stream responses (required to measure time-to-first-token)</label>
        <label class="check"><input type="checkbox" id="f-pdf" checked>
          Generate the PDF report automatically</label>

        <div class="notice"><strong>Benchmark-only configurations.</strong> Every setting below is
          sent per request. No model, Modelfile, or Ollama setting on your machine is modified.</div>

        <div class="row">
          <button class="primary big" id="btn-start">Start Full Optimization</button>
        </div>
        <p class="faint"><small id="estimate"></small></p>
      </div>
    </div>

    <h3>What will be tested</h3>
    <div class="grid cols-2">
      ${opts.optimizations.map(o => `<div class="panel">
        <h4>${esc(o.name)} <span class="badge">${esc(o.category)}</span></h4>
        <p class="muted"><small>${esc(o.description)}</small></p>
      </div>`).join('')}
    </div>`;

  const showMeta = async () => {
    const name = $('#f-model').value;
    const m = models.models.find(x => x.name === name) || {};
    $('#model-meta').innerHTML = `<dl class="kv">
      <dt>Size</dt><dd>${esc(m.size_human || 'N/A')}</dd>
      <dt>Parameters</dt><dd>${esc(m.parameter_size || 'N/A')}</dd>
      <dt>Quantization</dt><dd>${esc(m.quantization_level || 'N/A')}</dd>
      <dt>Family</dt><dd>${esc(m.family || 'N/A')}</dd></dl>`;
  };
  $('#f-model').addEventListener('change', showMeta);
  showMeta();

  let promptId = null;
  view().querySelectorAll('[data-prompt]').forEach(chip => {
    chip.addEventListener('click', () => {
      const p = state.prompts.find(x => x.id === chip.dataset.prompt);
      view().querySelectorAll('[data-prompt]').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      promptId = p.id;
      $('#f-prompt').value = p.prompt;
      $('#prompt-note').textContent =
        `${p.label} — category: ${p.category}. Expectations are checked automatically.`;
    });
  });
  $('#f-prompt').addEventListener('input', () => {
    if (promptId) {
      const p = state.prompts.find(x => x.id === promptId);
      if (!p || $('#f-prompt').value !== p.prompt) {
        promptId = null;
        view().querySelectorAll('[data-prompt]').forEach(c => c.classList.remove('active'));
      }
    }
  });

  $('#f-objective').addEventListener('change', e => {
    $('#custom-weights').style.display = e.target.value === 'custom' ? 'block' : 'none';
  });

  const estimate = () => {
    const runs = parseInt($('#f-runs').value || '5', 10);
    const strategy = $('#f-strategy').value;
    const st = opts.strategies.find(x => x.id === strategy) || { optimizations: [] };
    const approx = { baseline: 1, generation: 17, prompt: 8, system_prompt: 5,
                     context: 5, runtime: 4, quantization: 2 };
    let configs = 1;
    st.optimizations.forEach(id => { configs += (approx[id] || approx[id.split('_')[0]] || 4); });
    $('#estimate').textContent =
      `Rough plan: up to ~${configs} configurations × ${runs} runs ≈ ${configs * runs} generations ` +
      `(the engine skips inapplicable and duplicate configurations, so the real number is usually lower).`;
  };
  ['#f-runs', '#f-strategy'].forEach(sel => $(sel).addEventListener('change', estimate));
  estimate();

  $('#btn-start').addEventListener('click', async () => {
    const prompt = $('#f-prompt').value.trim();
    if (!prompt) { toast('Enter a prompt, or pick one of the benchmark prompts.', true); return; }
    const objective = $('#f-objective').value;
    const body = {
      model: $('#f-model').value,
      prompt,
      prompt_id: promptId,
      system_prompt: $('#f-system').value.trim() || null,
      strategy: $('#f-strategy').value,
      objective: objective === 'custom' ? 'custom' : objective,
      runs_per_config: parseInt($('#f-runs').value, 10),
      max_tokens: parseInt($('#f-max-tokens').value, 10),
      timeout_seconds: parseFloat($('#f-timeout').value),
      concurrency: parseInt($('#f-concurrency').value, 10),
      seed: parseInt($('#f-seed').value, 10),
      evaluator_mode: $('#f-evaluator').value,
      stream: $('#f-stream').checked,
      generate_pdf: $('#f-pdf').checked,
    };
    if (objective === 'custom') {
      body.custom_weights = {
        quality: parseFloat($('#w-quality').value) / 100,
        speed: parseFloat($('#w-speed').value) / 100,
        consistency: parseFloat($('#w-consistency').value) / 100,
        efficiency: parseFloat($('#w-efficiency').value) / 100,
      };
    }
    const btn = $('#btn-start');
    btn.disabled = true;
    btn.innerHTML = '<span class="spin"></span>Starting…';
    try {
      const res = await api('/api/experiments', { method: 'POST', body: JSON.stringify(body) });
      location.hash = '#/experiment/' + res.experiment_id;
    } catch (e) {
      toast(e.message, true);
      btn.disabled = false;
      btn.textContent = 'Start Full Optimization';
    }
  });
}

// ---------------------------------------------------------------- experiments list

async function renderExperiments() {
  const data = await api('/api/experiments?limit=100');
  const rows = data.experiments.map(e => `
    <tr>
      <td><a class="mono" href="#/experiment/${esc(e.id)}">${esc(e.id)}</a></td>
      <td>${esc(e.model)}</td>
      <td>${badge(e.status)}</td>
      <td>${esc(e.strategy)} / ${esc(e.objective)}</td>
      <td class="muted"><small>${esc(e.headline || e.error || e.prompt_preview || '')}</small></td>
      <td class="num muted"><small>${when(e.created_at)}</small></td>
      <td><button class="link" data-del="${esc(e.id)}">delete</button></td>
    </tr>`).join('') || '<tr><td colspan="7" class="muted">No experiments recorded yet.</td></tr>';

  view().innerHTML = `<h2>Experiments</h2>
    <p class="lede">Every experiment is stored in SQLite with its full configuration, so it can be
      reopened and reproduced later.</p>
    <div class="panel table-wrap"><table>
      <thead><tr><th>ID</th><th>Model</th><th>Status</th><th>Strategy / objective</th>
        <th>Result</th><th class="num">Started</th><th></th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;

  view().querySelectorAll('[data-del]').forEach(b => b.addEventListener('click', async () => {
    if (!confirm('Delete this experiment and all of its runs? This cannot be undone.')) return;
    try {
      await api('/api/experiments/' + b.dataset.del, { method: 'DELETE' });
      renderExperiments();
    } catch (e) { toast(e.message, true); }
  }));
}

// ---------------------------------------------------------------- experiment detail

async function renderExperiment(expId) {
  if (!expId) { location.hash = '#/experiments'; return; }
  const data = await api('/api/experiments/' + expId);
  const exp = data.experiment;
  const live = exp.status === 'running' || exp.status === 'queued';

  view().innerHTML = `
    <h2>Experiment <span class="mono faint" style="font-size:15px">${esc(exp.id)}</span> ${badge(exp.status)}</h2>
    <p class="lede">${esc(exp.model)} · ${esc(exp.strategy)} strategy · ${esc(exp.runs_per_config)} runs per
      configuration · objective “${esc(exp.objective)}” · started ${when(exp.created_at)}</p>

    ${exp.error ? `<div class="notice bad">${esc(exp.error)}</div>` : ''}

    <div class="panel" id="progress-panel"></div>
    <div id="result-area"></div>

    <h3>Reproducibility record</h3>
    <div class="panel"><dl class="kv">
      <dt>Model</dt><dd>${esc(exp.model)}</dd>
      <dt>Ollama version</dt><dd>${esc(exp.ollama_version || 'N/A')}</dd>
      <dt>App version</dt><dd>${esc(exp.app_version || 'N/A')}</dd>
      <dt>Seed</dt><dd>${esc(exp.seed)}</dd>
      <dt>Runs / config</dt><dd>${esc(exp.runs_per_config)}</dd>
      <dt>Max tokens</dt><dd>${esc(exp.max_tokens)}</dd>
      <dt>Timeout</dt><dd>${esc(exp.timeout_seconds)}s</dd>
      <dt>Streaming</dt><dd>${exp.stream ? 'yes' : 'no'}</dd>
      <dt>Concurrency</dt><dd>${esc(exp.concurrency)}</dd>
      <dt>Evaluator</dt><dd>${esc(exp.evaluator_mode)}</dd>
      <dt>Host</dt><dd>${esc(JSON.stringify(exp.host_info || {}))}</dd>
    </dl>
    <h4>Prompt</h4><pre>${esc(exp.prompt)}</pre>
    ${exp.system_prompt ? `<h4>System prompt</h4><pre>${esc(exp.system_prompt)}</pre>` : ''}
    </div>`;

  paintProgress(exp);
  if (!live) {
    await paintResults(expId, data);
  } else {
    followExperiment(expId);
  }
}

function paintProgress(exp, snapshot) {
  const panel = $('#progress-panel');
  if (!panel) return;
  const p = snapshot || exp.progress || {};
  const steps = p.steps || [];
  const marks = { done: '✓', running: '●', failed: '✗', skipped: '–', pending: '○' };
  const live = exp.status === 'running' || exp.status === 'queued';
  const fraction = p.fraction || 0;

  panel.innerHTML = `
    <div class="row" style="justify-content:space-between">
      <h4 style="margin:0">Optimization progress</h4>
      ${live ? `<button class="danger" id="btn-cancel">Cancel experiment</button>` : ''}
    </div>
    <div class="bar"><i style="width:${Math.round(fraction * 100)}%"></i></div>
    <p class="faint mono"><small>
      ${p.completed_runs || 0}/${p.total_runs || 0} generations ·
      elapsed ${num(p.elapsed_seconds, 0)}s ·
      ${p.eta_seconds ? 'estimated ' + num(p.eta_seconds, 0) + 's remaining' : 'estimate pending'}
      ${p.current ? ' · ' + esc(p.current) : ''}</small></p>
    <ul class="checklist">
      ${steps.map(s => `<li class="${esc(s.status)}"><span class="mark">${marks[s.status] || '○'}</span>${esc(s.label)}
        ${s.detail ? `<span class="detail">${esc(s.detail)}</span>` : ''}</li>`).join('')
        || '<li class="pending"><span class="mark">○</span>Waiting for the worker to start…</li>'}
    </ul>`;

  const cancel = $('#btn-cancel');
  if (cancel) cancel.addEventListener('click', async () => {
    cancel.disabled = true;
    try {
      await api('/api/experiments/' + exp.id + '/cancel', { method: 'POST' });
      toast('Cancelling after the current generation finishes.');
    } catch (e) { toast(e.message, true); cancel.disabled = false; }
  });
}

function followExperiment(expId) {
  const finish = async () => {
    stopStreams();
    const fresh = await api('/api/experiments/' + expId);
    paintProgress(fresh.experiment);
    await paintResults(expId, fresh);
  };

  if (window.EventSource) {
    const es = new EventSource('/api/experiments/' + expId + '/events');
    state.es = es;
    es.onmessage = async ev => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.progress) {
        const exp = { id: expId, status: 'running', progress: msg.progress };
        paintProgress(exp, msg.progress);
      }
      if (['finished', 'failed', 'cancelled', 'closed'].includes(msg.type)) {
        await finish();
      }
    };
    es.onerror = () => { es.close(); state.es = null; startPolling(expId, finish); };
  } else {
    startPolling(expId, finish);
  }
}

function startPolling(expId, finish) {
  if (state.poll) return;
  state.poll = setInterval(async () => {
    try {
      const p = await api('/api/experiments/' + expId + '/progress');
      paintProgress({ id: expId, status: p.status, progress: p.progress }, p.progress);
      if (!['running', 'queued'].includes(p.status)) await finish();
    } catch (e) { /* keep polling */ }
  }, 2000);
}

async function paintResults(expId, data) {
  const area = $('#result-area');
  if (!area) return;
  const exp = data.experiment;
  const configs = (data.configurations || []).filter(c => c.status !== 'skipped');
  const skipped = (data.configurations || []).filter(c => c.status === 'skipped');
  const summary = exp.summary || {};
  const rec = summary.recommended;
  const cmp = summary.recommended_comparison || {};

  if (!configs.length) {
    area.innerHTML = `<div class="notice">No configuration produced results for this experiment.</div>`;
    return;
  }

  const bestId = rec ? rec.configuration_id : null;
  const rows = configs.map(c => {
    const a = c.aggregate || {};
    return `<tr class="${c.is_baseline ? 'is-baseline' : ''} ${c.id === bestId ? 'is-best' : ''}">
      <td>${esc(c.label)}${c.is_baseline ? ' <span class="badge">baseline</span>' : ''}
        ${c.id === bestId ? ' <span class="badge ok">recommended</span>' : ''}
        <div class="faint"><small>${esc(c.category)} · ${esc(c.phase)}</small></div></td>
      <td class="num">${stat(a.quality, 'mean')}</td>
      <td class="num">${stat(a.tokens_per_second, 'mean', 1)}</td>
      <td class="num">${stat(a.latency, 'median')}</td>
      <td class="num">${stat(a.time_to_first_token, 'mean')}</td>
      <td class="num">${a.consistency && a.consistency.score !== null && a.consistency.score !== undefined
          ? num(a.consistency.score, 2) : 'N/A'}</td>
      <td class="num">${a.successful_runs || 0}/${a.total_runs || 0}</td>
      <td><button class="link" data-runs="${esc(c.id)}">inspect</button></td>
    </tr>`;
  }).join('');

  area.innerHTML = `
    ${rec ? `<div class="notice good">
      <strong>Recommended configuration: ${esc(rec.label)}</strong><br>
      ${esc(summary.explanation || '')}
      ${summary.caveat ? `<br><span class="faint"><small>${esc(summary.caveat)}</small></span>` : ''}
    </div>
    <div class="grid cols-4">
      <div class="stat"><div class="k">Composite score</div><div class="v">${num(rec.composite_score, 2)}</div></div>
      <div class="stat"><div class="k">Quality vs baseline</div><div class="v small">${pct(cmp.quality_percent)}</div></div>
      <div class="stat"><div class="k">Tokens/s vs baseline</div><div class="v small">${pct(cmp.tokens_per_second_percent)}</div></div>
      <div class="stat"><div class="k">Median latency vs baseline</div><div class="v small">${pct(cmp.latency_percent)}</div></div>
    </div>` : `<div class="notice">${esc(summary.headline || 'No recommendation is available for this experiment.')}</div>`}

    <h3>Comparison — all configurations</h3>
    <div class="panel table-wrap">
      <table><thead><tr>
        <th>Configuration</th><th class="num">Quality /10</th><th class="num">Tokens/s</th>
        <th class="num">Median latency (s)</th><th class="num">TTFT (s)</th>
        <th class="num">Consistency</th><th class="num">Runs ok</th><th></th>
      </tr></thead><tbody>${rows}</tbody></table>
      <p class="faint"><small>Quality is the deterministic heuristic score (0–10). “N/A” means the
        metric was not reported by Ollama for these runs — nothing is estimated.</small></p>
    </div>
    <div id="runs-area"></div>

    ${skipped.length ? `<h3>Not tested</h3><div class="panel">
      ${skipped.map(c => `<p><strong>${esc(c.label)}</strong> — <span class="muted">${esc(c.skip_reason || 'skipped')}</span></p>`).join('')}
    </div>` : ''}
    ${(summary.not_tested || []).length ? `<h3>Explicitly not tested</h3><div class="panel">
      ${summary.not_tested.map(n => `<p><strong>${esc(n.label || n.id)}</strong> — <span class="muted">${esc(n.reason || '')}</span></p>`).join('')}
    </div>` : ''}

    <h3>Report</h3>
    <div class="panel">
      <div class="row">
        <button class="primary" id="btn-pdf">Download PDF Report</button>
        <button id="btn-md">Download Markdown</button>
        <button id="btn-regen">Regenerate report</button>
      </div>
      <p class="faint" id="report-note"><small>${
        (data.reports || []).map(r => `${esc(r.format)}: ${r.status === 'ok'
          ? esc(r.path) + (r.pages ? ` (${r.pages} pages)` : '')
          : 'failed — ' + esc(r.error || '')}`).join('<br>') || 'No report files generated yet.'}</small></p>
    </div>`;

  area.querySelectorAll('[data-runs]').forEach(btn => btn.addEventListener('click', async () => {
    const box = $('#runs-area');
    box.innerHTML = '<p class="muted"><span class="spin"></span>Loading runs…</p>';
    try {
      const cfg = configs.find(c => c.id === btn.dataset.runs);
      const res = await api('/api/experiments/' + expId + '/runs?configuration_id=' + btn.dataset.runs);
      box.innerHTML = renderRuns(cfg, res.runs);
    } catch (e) { box.innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }
  }));

  $('#btn-pdf').addEventListener('click', () => {
    window.location.href = '/api/experiments/' + expId + '/report.pdf';
  });
  $('#btn-md').addEventListener('click', () => {
    window.location.href = '/api/experiments/' + expId + '/report.md';
  });
  $('#btn-regen').addEventListener('click', async () => {
    const note = $('#report-note');
    note.innerHTML = '<span class="spin"></span>Rebuilding report and PDF…';
    try {
      const res = await api('/api/experiments/' + expId + '/report',
        { method: 'POST', body: JSON.stringify({ pdf: true }) });
      note.innerHTML = `<small>Markdown: ${esc(res.markdown)} (~${res.pages} pages)<br>` +
        (res.pdf ? `PDF: ${esc(res.pdf)} (${res.pdf_pages} pages)` :
          `PDF failed — ${esc(res.pdf_error || 'unknown error')}`) + '</small>';
    } catch (e) { note.innerHTML = `<span class="badge failed">${esc(e.message)}</span>`; }
  });
}

function renderRuns(cfg, runs) {
  const optionRows = Object.entries(cfg.options || {})
    .map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join('\n') || '(model defaults)';
  const items = runs.map(r => {
    const m = r.metrics || {};
    const ev = r.evaluation || {};
    const crit = Object.entries(ev.criteria || {}).map(([k, c]) =>
      `<tr><td>${esc(k.replace(/_/g, ' '))}</td><td class="num">${c.scorable === false ? 'N/A' : num(c.score, 1)}</td>
       <td class="muted"><small>${esc(c.basis || '')}</small></td></tr>`).join('');
    return `<details>
      <summary>Run #${r.run_index + 1} — ${esc(r.status)} ·
        ${m.wall_seconds !== undefined ? num(m.wall_seconds) + 's' : 'N/A'} ·
        quality ${ev.quality_score !== undefined && ev.quality_score !== null ? num(ev.quality_score, 2) : 'N/A'}</summary>
      ${r.error ? `<div class="notice bad">${esc(r.error)}</div>` : ''}
      <h4>Metrics</h4>
      <dl class="kv">
        <dt>Time to first token</dt><dd>${m.time_to_first_token !== undefined && m.time_to_first_token !== null ? num(m.time_to_first_token) + ' s' : 'N/A — not streamed or not reported'}</dd>
        <dt>Total duration</dt><dd>${m.total_duration !== undefined ? num(m.total_duration) + ' s' : 'N/A'}</dd>
        <dt>Prompt eval duration</dt><dd>${m.prompt_eval_duration !== undefined ? num(m.prompt_eval_duration) + ' s' : 'N/A'}</dd>
        <dt>Generation duration</dt><dd>${m.eval_duration !== undefined ? num(m.eval_duration) + ' s' : 'N/A'}</dd>
        <dt>Prompt tokens</dt><dd>${m.prompt_tokens !== undefined ? m.prompt_tokens : 'N/A'}</dd>
        <dt>Output tokens</dt><dd>${m.output_tokens !== undefined ? m.output_tokens : 'N/A'}</dd>
        <dt>Tokens / second</dt><dd>${m.tokens_per_second !== undefined && m.tokens_per_second !== null ? num(m.tokens_per_second, 1) : 'N/A'}</dd>
        <dt>Wall time</dt><dd>${m.wall_seconds !== undefined ? num(m.wall_seconds) + ' s' : 'N/A'}</dd>
      </dl>
      ${Object.keys(r.resources || {}).length ? `<h4>Resources</h4><pre>${esc(JSON.stringify(r.resources, null, 2))}</pre>` : ''}
      ${crit ? `<h4>Evaluation</h4><table><thead><tr><th>Criterion</th><th class="num">Score</th><th>Basis</th></tr></thead><tbody>${crit}</tbody></table>` : ''}
      <h4>Response</h4><pre>${esc(r.output || '(no output)')}</pre>
    </details>`;
  }).join('');

  return `<h3>Raw runs — ${esc(cfg.label)}</h3>
    <div class="panel">
      <p class="muted"><small>${esc(cfg.rationale || '')}</small></p>
      <h4>Configuration sent to Ollama</h4><pre>${esc(optionRows)}</pre>
      ${cfg.prompt_modified ? `<h4>Prompt used (modified)</h4><pre>${esc(cfg.prompt)}</pre>` : ''}
      ${cfg.system_prompt ? `<h4>System prompt</h4><pre>${esc(cfg.system_prompt)}</pre>` : ''}
      ${items || '<p class="muted">No runs recorded.</p>'}
    </div>`;
}

// ---------------------------------------------------------------- reports

async function renderReports() {
  const data = await api('/api/reports');
  const rows = data.reports.map(r => `<tr>
      <td><a class="mono" href="#/experiment/${esc(r.experiment_id)}">${esc(r.experiment_id)}</a></td>
      <td>${esc(r.model || '—')}</td>
      <td>${esc(r.format)}</td>
      <td class="num">${r.pages || '—'}</td>
      <td>${r.status === 'ok' ? '<span class="badge ok">ok</span>'
            : `<span class="badge failed">failed</span> <span class="muted"><small>${esc(r.error || '')}</small></span>`}</td>
      <td class="num muted"><small>${when(r.created_at)}</small></td>
      <td>${r.exists ? `<a href="/api/experiments/${esc(r.experiment_id)}/report.${r.format === 'pdf' ? 'pdf' : 'md'}">download</a>`
            : '<span class="faint">file missing</span>'}</td>
    </tr>`).join('') || '<tr><td colspan="7" class="muted">No reports generated yet.</td></tr>';

  view().innerHTML = `<h2>Reports</h2>
    <p class="lede">Markdown and PDF reports are written to the report directory configured in Settings.</p>
    <div class="panel table-wrap"><table><thead><tr>
      <th>Experiment</th><th>Model</th><th>Format</th><th class="num">Pages</th>
      <th>Status</th><th class="num">Created</th><th></th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
}

// ---------------------------------------------------------------- settings

async function renderSettings() {
  const data = await api('/api/settings');
  const s = data.settings;
  const field = (key, type, label, help) => `
    <label class="field"><span>${esc(label)}</span>
      ${type === 'checkbox'
        ? `<input type="checkbox" data-key="${key}" ${s[key] ? 'checked' : ''}>`
        : `<input type="${type}" data-key="${key}" value="${esc(s[key])}" step="any">`}
      ${help ? `<small class="faint">${esc(help)}</small>` : ''}</label>`;

  view().innerHTML = `<h2>Settings</h2>
    <p class="lede">Stored in <code class="mono">data/settings.json</code>. Environment variables and
      <code class="mono">.env</code> provide the defaults; values saved here override them.</p>
    <div class="grid cols-2">
      <div class="panel">
        <h4>Ollama</h4>
        ${field('ollama_url', 'text', 'Ollama URL', 'Default http://127.0.0.1:11434')}
        ${field('ollama_request_timeout', 'number', 'Request timeout (s)')}
        ${field('ollama_retries', 'number', 'Retries on transient errors')}
        ${field('ollama_keep_alive', 'text', 'Keep-alive', 'e.g. 5m — how long Ollama keeps the model loaded')}
        <h4>Evaluation</h4>
        ${field('evaluator_mode', 'text', 'Evaluator mode', 'heuristic | heuristic+llm')}
        ${field('evaluator_model', 'text', 'Evaluator model', 'Empty = reuse the benchmarked model. LLM scores are labelled estimates.')}
      </div>
      <div class="panel">
        <h4>Benchmark defaults</h4>
        ${field('default_runs_per_config', 'number', 'Runs per configuration')}
        ${field('default_max_tokens', 'number', 'Max generated tokens')}
        ${field('default_timeout_seconds', 'number', 'Timeout (s)')}
        ${field('max_concurrency', 'number', 'Maximum concurrency', 'Default 1 — local inference is resource bound')}
        ${field('default_strategy', 'text', 'Default strategy')}
        ${field('default_objective', 'text', 'Default objective')}
        ${field('default_stream', 'checkbox', 'Stream by default')}
        ${field('auto_generate_pdf', 'checkbox', 'Generate PDF automatically')}
        <h4>Storage</h4>
        ${field('database_path', 'text', 'Database path')}
        ${field('report_dir', 'text', 'Report directory')}
        ${field('log_level', 'text', 'Log level')}
      </div>
    </div>
    <div class="row"><button class="primary" id="btn-save">Save settings</button>
      <span id="save-note" class="muted"></span></div>`;

  $('#btn-save').addEventListener('click', async () => {
    const body = {};
    view().querySelectorAll('[data-key]').forEach(el => {
      const key = el.dataset.key;
      if (el.type === 'checkbox') body[key] = el.checked;
      else if (el.type === 'number') body[key] = parseFloat(el.value);
      else body[key] = el.value;
    });
    try {
      await api('/api/settings', { method: 'PUT', body: JSON.stringify(body) });
      $('#save-note').textContent = 'Saved.';
      refreshStatus();
    } catch (e) { toast(e.message, true); }
  });
}
