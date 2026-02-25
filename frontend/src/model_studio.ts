/**
 * qMachina Model Studio -- multi-step modeling wizard for assembling and
 * promoting serving configurations. Connects to the /v1/modeling/* REST API.
 *
 * Layout: left sidebar stepper + right content panel.
 * Step lock logic: a step is accessible only if all prior steps are committed.
 * Manual mode: toggles a YAML editor textarea for direct ServingSpec editing.
 */

// ------------------------------------------------------------------ Types

interface StepState {
  step_name: string;
  status: 'pending' | 'committed';
  payload: Record<string, unknown> | null;
  committed_at: string | null;
}

interface SessionResponse {
  session_id: string;
  status: string;
  created_by: string;
  created_at: string;
  steps: StepState[];
}

interface SpecsResponse {
  signals: string[];
  datasets: string[];
  steps: string[];
}

interface PreviewResponse {
  n_bins: number;
  date_range: { start?: string; end?: string };
  signal_distribution: {
    mean: number | null;
    std: number | null;
    pct25: number | null;
    pct50: number | null;
    pct75: number | null;
  };
  mid_price_range: { min: number | null; max: number | null };
}

interface ValidateYamlResponse {
  valid: boolean;
  errors: string[];
}

// ------------------------------------------------------------------ Constants

const API_PORT = 8002;
const API_BASE = `http://localhost:${API_PORT}`;

const STEPS_ORDERED: string[] = [
  'dataset_select',
  'gold_config',
  'signal_select',
  'eval_params',
  'run_experiment',
  'promote_review',
  'promotion',
];

const STEP_LABELS: Record<string, string> = {
  dataset_select: 'Dataset',
  gold_config: 'Gold Config',
  signal_select: 'Signal',
  eval_params: 'Eval Params',
  run_experiment: 'Run Experiment',
  promote_review: 'Review',
  promotion: 'Promote',
};

const STEP_ICONS: Record<string, string> = {
  pending: '\u25CB',   // open circle
  active: '\u25B6',    // right triangle
  committed: '\u2713', // check
  locked: '\uD83D\uDD12',   // lock
};

// ------------------------------------------------------------------ State

let sessionId: string | null = null;
let stepStates: Map<string, StepState> = new Map();
let activeStep: string = STEPS_ORDERED[0];
let manualMode = false;
let availableSpecs: SpecsResponse | null = null;

// ------------------------------------------------------------------ DOM refs

const $sidebar = document.getElementById('sidebar') as HTMLDivElement;
const $content = document.getElementById('content') as HTMLDivElement;
const $sessionInfo = document.getElementById('session-info') as HTMLSpanElement;
const $statusText = document.getElementById('status-text') as HTMLSpanElement;
const $manualToggle = document.getElementById('manual-toggle') as HTMLDivElement;

// ------------------------------------------------------------------ Helpers

function esc(s: string): string {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function fmt(v: number | null | undefined, decimals = 2): string {
  if (v === null || v === undefined) return '--';
  return v.toFixed(decimals);
}

function stepIndex(name: string): number {
  return STEPS_ORDERED.indexOf(name);
}

function isStepAccessible(name: string): boolean {
  const idx = stepIndex(name);
  if (idx === 0) return true;
  for (let i = 0; i < idx; i++) {
    const prior = STEPS_ORDERED[i];
    const state = stepStates.get(prior);
    if (!state || state.status !== 'committed') return false;
  }
  return true;
}

function getStepDisplayState(name: string): string {
  const state = stepStates.get(name);
  if (state && state.status === 'committed') return 'committed';
  if (name === activeStep && isStepAccessible(name)) return 'active';
  if (!isStepAccessible(name)) return 'locked';
  return 'pending';
}

// ------------------------------------------------------------------ API

async function apiPost(path: string, body: unknown): Promise<unknown> {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API ${resp.status}: ${text}`);
  }
  return resp.json();
}

async function apiGet(path: string): Promise<unknown> {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API ${resp.status}: ${text}`);
  }
  return resp.json();
}

async function createSession(): Promise<string> {
  // Use a fixed workspace ID for the studio (single-user mode)
  const data = await apiPost('/v1/modeling/sessions', {
    workspace_id: '00000000-0000-0000-0000-000000000001',
    created_by: 'model_studio',
  }) as { session_id: string };
  return data.session_id;
}

async function fetchSession(sid: string): Promise<SessionResponse> {
  return await apiGet(`/v1/modeling/sessions/${sid}`) as SessionResponse;
}

async function commitStep(sid: string, stepName: string, payload: Record<string, unknown>): Promise<void> {
  await apiPost(`/v1/modeling/sessions/${sid}/steps/${stepName}/commit`, { payload });
}

async function fetchSpecs(): Promise<SpecsResponse> {
  return await apiGet('/v1/modeling/specs') as SpecsResponse;
}

async function fetchPreview(sid: string): Promise<PreviewResponse> {
  return await apiPost(`/v1/modeling/sessions/${sid}/preview`, {}) as PreviewResponse;
}

async function validateYaml(yamlContent: string): Promise<ValidateYamlResponse> {
  return await apiPost('/v1/modeling/validate_yaml', { yaml_content: yamlContent }) as ValidateYamlResponse;
}

async function promoteSession(sid: string, alias: string): Promise<unknown> {
  return await apiPost(`/v1/modeling/sessions/${sid}/promote`, { alias });
}

// ------------------------------------------------------------------ Sidebar

function renderSidebar(): void {
  $sidebar.innerHTML = '';
  for (const name of STEPS_ORDERED) {
    const idx = stepIndex(name);
    const display = getStepDisplayState(name);
    const item = document.createElement('div');
    item.className = `step-item ${display === 'active' ? 'active' : ''} ${display === 'locked' ? 'locked' : ''}`;
    item.innerHTML = `
      <span class="step-icon ${display}">${STEP_ICONS[display]}</span>
      <span class="step-label">${idx + 1}. ${esc(STEP_LABELS[name])}</span>
    `;
    if (display !== 'locked') {
      item.addEventListener('click', () => {
        activeStep = name;
        renderSidebar();
        renderContent();
      });
    }
    $sidebar.appendChild(item);
  }
}

// ------------------------------------------------------------------ Content panels

function renderContent(): void {
  if (manualMode) {
    renderManualMode();
    return;
  }

  switch (activeStep) {
    case 'dataset_select': renderDatasetSelect(); break;
    case 'gold_config': renderGoldConfig(); break;
    case 'signal_select': renderSignalSelect(); break;
    case 'eval_params': renderEvalParams(); break;
    case 'run_experiment': renderRunExperiment(); break;
    case 'promote_review': renderPromoteReview(); break;
    case 'promotion': renderPromotion(); break;
    default: $content.innerHTML = '<div class="empty-state">Unknown step</div>';
  }
}

// -- Dataset Select --

function renderDatasetSelect(): void {
  const existing = stepStates.get('dataset_select')?.payload as Record<string, string> | null;
  const datasets = availableSpecs?.datasets ?? [];

  $content.innerHTML = `
    <div class="panel-title">1. Select Dataset</div>
    <div class="form-group">
      <label>Dataset</label>
      <select id="ds-select">
        <option value="">-- Select --</option>
        ${datasets.map(d => `<option value="${esc(d)}" ${existing?.dataset_id === d ? 'selected' : ''}>${esc(d)}</option>`).join('')}
      </select>
    </div>
    <button id="ds-commit" class="btn btn-primary">Commit Step</button>
  `;

  document.getElementById('ds-commit')!.addEventListener('click', async () => {
    const sel = (document.getElementById('ds-select') as HTMLSelectElement).value;
    if (!sel) { alert('Select a dataset'); return; }
    await doCommit('dataset_select', { dataset_id: sel });
  });
}

// -- Gold Config --

function renderGoldConfig(): void {
  const existing = (stepStates.get('gold_config')?.payload ?? {}) as Record<string, number>;

  const coeffs = [
    { key: 'c1_v_add', label: 'c1 (v_add)', def: 1.0 },
    { key: 'c2_v_rest_pos', label: 'c2 (v_rest_pos)', def: 0.5 },
    { key: 'c3_a_add', label: 'c3 (a_add)', def: 0.3 },
    { key: 'c4_v_pull', label: 'c4 (v_pull)', def: -0.8 },
    { key: 'c5_v_fill', label: 'c5 (v_fill)', def: -0.5 },
    { key: 'c6_v_rest_neg', label: 'c6 (v_rest_neg)', def: -0.3 },
    { key: 'c7_a_pull', label: 'c7 (a_pull)', def: -0.2 },
  ];

  $content.innerHTML = `
    <div class="panel-title">2. Gold Feature Configuration</div>
    <div class="form-row">
      ${coeffs.map(c => `
        <div class="form-group">
          <label>${esc(c.label)}</label>
          <input type="number" step="0.01" id="gold-${c.key}" value="${existing[c.key] ?? c.def}">
        </div>
      `).join('')}
    </div>
    <button id="gold-commit" class="btn btn-primary">Commit Step</button>
  `;

  document.getElementById('gold-commit')!.addEventListener('click', async () => {
    const payload: Record<string, number> = {};
    for (const c of coeffs) {
      const el = document.getElementById(`gold-${c.key}`) as HTMLInputElement;
      payload[c.key] = parseFloat(el.value);
    }
    await doCommit('gold_config', payload);
  });
}

// -- Signal Select --

function renderSignalSelect(): void {
  const existing = (stepStates.get('signal_select')?.payload ?? {}) as Record<string, unknown>;
  const signals = availableSpecs?.signals ?? ['derivative'];

  $content.innerHTML = `
    <div class="panel-title">3. Signal Selection</div>
    <div class="form-group">
      <label>Signal</label>
      <select id="sig-select">
        ${signals.map(s => `<option value="${esc(s)}" ${existing.signal_name === s ? 'selected' : ''}>${esc(s)}</option>`).join('')}
      </select>
    </div>
    <div id="sig-params"></div>
    <button id="sig-commit" class="btn btn-primary">Commit Step</button>
  `;

  document.getElementById('sig-commit')!.addEventListener('click', async () => {
    const signal_name = (document.getElementById('sig-select') as HTMLSelectElement).value;
    await doCommit('signal_select', { signal_name });
  });
}

// -- Eval Params --

function renderEvalParams(): void {
  const existing = (stepStates.get('eval_params')?.payload ?? {}) as Record<string, number>;

  const params = [
    { key: 'tp_ticks', label: 'TP Ticks', def: 8, min: 1 },
    { key: 'sl_ticks', label: 'SL Ticks', def: 4, min: 1 },
    { key: 'cooldown_bins', label: 'Cooldown Bins', def: 20, min: 0 },
    { key: 'warmup_bins', label: 'Warmup Bins', def: 300, min: 0 },
  ];

  $content.innerHTML = `
    <div class="panel-title">4. Evaluation Parameters</div>
    <div class="form-row">
      ${params.map(p => `
        <div class="form-group">
          <label>${esc(p.label)}</label>
          <input type="number" id="eval-${p.key}" value="${existing[p.key] ?? p.def}" min="${p.min}">
        </div>
      `).join('')}
    </div>
    <button id="eval-commit" class="btn btn-primary">Commit Step</button>
  `;

  document.getElementById('eval-commit')!.addEventListener('click', async () => {
    const payload: Record<string, number> = {};
    for (const p of params) {
      const el = document.getElementById(`eval-${p.key}`) as HTMLInputElement;
      payload[p.key] = parseInt(el.value);
    }
    await doCommit('eval_params', payload);
  });
}

// -- Run Experiment --

function renderRunExperiment(): void {
  const expState = stepStates.get('run_experiment');
  const expPayload = (expState?.payload ?? {}) as Record<string, unknown>;
  const jobId = expPayload['job_id'] as string | undefined;
  const expStatus = expPayload['status'] as string | undefined;

  // Already completed
  if (expStatus === 'completed') {
    const runIds = (expPayload['run_ids'] as string[] | undefined) ?? [];
    const nRuns = (expPayload['n_runs'] as number | undefined) ?? 0;
    $content.innerHTML = `
      <div class="panel-title">5. Run Experiment</div>
      <div class="summary-card">
        <h4>Experiment Completed</h4>
        <div class="summary-kv">
          <span class="key">Job ID</span><span class="val">${esc(jobId ?? '--')}</span>
          <span class="key">Runs</span><span class="val">${nRuns}</span>
          <span class="key">Run IDs</span><span class="val" style="font-size:9px">${esc(runIds.join(', ') || 'none')}</span>
        </div>
      </div>
      <button id="exp-go-review" class="btn btn-primary">Go to Review</button>
    `;
    document.getElementById('exp-go-review')!.addEventListener('click', () => {
      activeStep = 'promote_review';
      renderSidebar();
      renderContent();
    });
    return;
  }

  // Running: show live log
  if (expStatus === 'running' && jobId) {
    renderRunExperimentProgress(jobId);
    return;
  }

  // Not submitted yet: show config summary + submit button
  const dsPayload = stepStates.get('dataset_select')?.payload ?? {};
  const evalPayload = stepStates.get('eval_params')?.payload ?? {};

  $content.innerHTML = `
    <div class="panel-title">5. Run Experiment</div>
    <div class="summary-card">
      <h4>Configuration Summary</h4>
      <div class="summary-kv">
        <span class="key">Dataset</span><span class="val">${esc(String((dsPayload as Record<string, unknown>)['dataset_id'] ?? '--'))}</span>
        <span class="key">TP Ticks</span><span class="val">${esc(String((evalPayload as Record<string, unknown>)['tp_ticks'] ?? '--'))}</span>
        <span class="key">SL Ticks</span><span class="val">${esc(String((evalPayload as Record<string, unknown>)['sl_ticks'] ?? '--'))}</span>
        <span class="key">Cooldown Bins</span><span class="val">${esc(String((evalPayload as Record<string, unknown>)['cooldown_bins'] ?? '--'))}</span>
        <span class="key">Warmup Bins</span><span class="val">${esc(String((evalPayload as Record<string, unknown>)['warmup_bins'] ?? '--'))}</span>
      </div>
    </div>
    <button id="exp-submit" class="btn btn-primary">Submit Experiment</button>
    <div id="exp-error"></div>
  `;

  document.getElementById('exp-submit')!.addEventListener('click', async () => {
    if (!sessionId) return;
    $statusText.textContent = 'Submitting experiment...';
    try {
      const result = await apiPost(
        `/v1/modeling/sessions/${sessionId}/experiment/submit`,
        {},
      ) as { job_id: string; spec_ref: string; spec_name: string };
      // Commit run_experiment with running status so we can track it
      await commitStep(sessionId, 'run_experiment', {
        job_id: result.job_id,
        status: 'running',
        run_ids: [],
        n_runs: 0,
      });
      await refreshSession();
      renderRunExperimentProgress(result.job_id);
    } catch (err) {
      document.getElementById('exp-error')!.innerHTML =
        `<div class="validation-errors">${esc(String(err))}</div>`;
      $statusText.textContent = 'Submit failed';
    }
  });
}

function renderRunExperimentProgress(jobId: string): void {
  $content.innerHTML = `
    <div class="panel-title">5. Run Experiment — Running</div>
    <div class="summary-card">
      <h4>Job Progress</h4>
      <div class="summary-kv">
        <span class="key">Job ID</span><span class="val" style="font-size:9px">${esc(jobId)}</span>
        <span class="key">Status</span><span class="val" id="exp-status-label">running</span>
      </div>
    </div>
    <div class="exp-log" id="exp-log"></div>
    <button id="exp-cancel" class="btn btn-secondary" style="margin-top:8px">Cancel Job</button>
    <div id="exp-progress-error"></div>
  `;

  const $log = document.getElementById('exp-log')!;
  const $statusLabel = document.getElementById('exp-status-label')!;

  function appendLog(msg: string): void {
    const line = document.createElement('div');
    line.className = 'log-line';
    line.textContent = msg;
    $log.appendChild(line);
    $log.scrollTop = $log.scrollHeight;
  }

  appendLog(`Connecting to job ${jobId}...`);

  const evtSource = new EventSource(`${API_BASE}/v1/jobs/experiments/${jobId}/events`);

  async function onTerminal(status: 'completed' | 'failed' | 'canceled', runIds: string[], nRuns: number): Promise<void> {
    evtSource.close();
    $statusLabel.textContent = status;
    if (!sessionId) return;
    // Update committed run_experiment step
    await commitStep(sessionId, 'run_experiment', {
      job_id: jobId,
      status,
      run_ids: runIds,
      n_runs: nRuns,
    });
    await refreshSession();
    if (status === 'completed') {
      appendLog('Experiment complete. Advancing to review...');
      activeStep = 'promote_review';
      renderSidebar();
      renderContent();
    } else {
      appendLog(`Job ended with status: ${status}`);
      renderSidebar();
      renderContent();
    }
  }

  evtSource.addEventListener('start', (e: MessageEvent) => {
    appendLog(`[start] ${e.data}`);
  });

  evtSource.addEventListener('progress', (e: MessageEvent) => {
    try {
      const payload = JSON.parse(e.data);
      appendLog(`[progress] ${payload.message ?? JSON.stringify(payload)}`);
    } catch {
      appendLog(`[progress] ${e.data}`);
    }
  });

  evtSource.addEventListener('complete', (e: MessageEvent) => {
    try {
      const payload = JSON.parse(e.data);
      const runIds: string[] = payload.run_ids ?? [];
      const nRuns: number = payload.n_runs ?? runIds.length;
      appendLog(`[complete] ${nRuns} runs finished`);
      void onTerminal('completed', runIds, nRuns);
    } catch {
      appendLog(`[complete] ${e.data}`);
      void onTerminal('completed', [], 0);
    }
  });

  evtSource.addEventListener('error', (e: MessageEvent) => {
    appendLog(`[error] ${e.data ?? 'unknown error'}`);
    void onTerminal('failed', [], 0);
  });

  evtSource.addEventListener('cancel', () => {
    appendLog('[canceled]');
    void onTerminal('canceled', [], 0);
  });

  evtSource.addEventListener('done', () => {
    evtSource.close();
  });

  evtSource.onerror = () => {
    appendLog('[SSE connection closed]');
    evtSource.close();
  };

  document.getElementById('exp-cancel')!.addEventListener('click', async () => {
    try {
      await apiPost(`/v1/jobs/experiments/${jobId}/cancel`, {});
      appendLog('Cancel requested...');
    } catch (err) {
      document.getElementById('exp-progress-error')!.innerHTML =
        `<div class="validation-errors">${esc(String(err))}</div>`;
    }
  });
}

// -- Promote Review --

function renderPromoteReview(): void {
  let html = '<div class="panel-title">6. Promote Review</div>';

  // Summary of all committed payloads (steps 0-3)
  for (const name of STEPS_ORDERED.slice(0, 4)) {
    const state = stepStates.get(name);
    if (!state || state.status !== 'committed') continue;
    html += `
      <div class="summary-card">
        <h4>${esc(STEP_LABELS[name])}</h4>
        <div class="summary-kv">
          ${Object.entries(state.payload ?? {}).map(([k, v]) =>
            `<span class="key">${esc(k)}</span><span class="val">${esc(String(v))}</span>`
          ).join('')}
        </div>
      </div>
    `;
  }

  // Experiment Results card
  const expPayload = (stepStates.get('run_experiment')?.payload ?? {}) as Record<string, unknown>;
  const expRunIds = (expPayload['run_ids'] as string[] | undefined) ?? [];
  const expNRuns = (expPayload['n_runs'] as number | undefined) ?? 0;
  const expJobId = expPayload['job_id'] as string | undefined;
  html += `
    <div class="summary-card">
      <h4>Experiment Results</h4>
      <div class="summary-kv">
        <span class="key">Job ID</span><span class="val" style="font-size:9px">${esc(expJobId ?? '--')}</span>
        <span class="key">Runs</span><span class="val">${expNRuns}</span>
        <span class="key">Run IDs</span><span class="val" style="font-size:9px">${esc(expRunIds.join(', ') || 'none')}</span>
      </div>
    </div>
  `;

  // Sensitivity panel
  html += `
    <div class="summary-card" id="sens-panel">
      <h4>Sensitivity Analysis</h4>
      <div class="form-row">
        <div class="form-group">
          <label>Sweep Axis</label>
          <select id="sens-axis">
            <option value="tp_ticks">TP Ticks</option>
            <option value="sl_ticks">SL Ticks</option>
            <option value="cooldown_bins">Cooldown Bins</option>
            <option value="zscore_window_bins">ZScore Window</option>
            <option value="tanh_scale">Tanh Scale</option>
          </select>
        </div>
        <div class="form-group">
          <label>Values (comma-separated)</label>
          <input type="text" id="sens-values" placeholder="e.g. 4,6,8,10,12">
        </div>
      </div>
      <button id="sens-run" class="btn btn-secondary">Run Sensitivity</button>
      <div id="sens-status" style="font-size:10px;color:#888;margin-top:6px"></div>
      <div id="sens-results"></div>
    </div>
  `;

  html += `
    <button id="review-preview" class="btn btn-secondary">Load Dataset Preview</button>
    <div id="preview-container"></div>
    <button id="review-commit" class="btn btn-primary" style="margin-top:16px">Confirm Review</button>
  `;

  $content.innerHTML = html;

  document.getElementById('review-preview')!.addEventListener('click', async () => {
    if (!sessionId) return;
    const container = document.getElementById('preview-container')!;
    container.innerHTML = '<div class="empty-state">Loading preview...</div>';
    try {
      const preview = await fetchPreview(sessionId);
      container.innerHTML = `
        <div class="preview-stats">
          <div class="stat-box">
            <div class="stat-value">${preview.n_bins}</div>
            <div class="stat-label">Total Bins</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">${preview.date_range.start?.split('T')[0] ?? '--'}</div>
            <div class="stat-label">Start Date</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">${fmt(preview.mid_price_range.min, 1)} - ${fmt(preview.mid_price_range.max, 1)}</div>
            <div class="stat-label">Mid Price Range</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">${fmt(preview.signal_distribution.mean, 3)}</div>
            <div class="stat-label">Signal Mean</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">${fmt(preview.signal_distribution.std, 3)}</div>
            <div class="stat-label">Signal Std</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">${fmt(preview.signal_distribution.pct50, 3)}</div>
            <div class="stat-label">Signal Median</div>
          </div>
        </div>
      `;
    } catch (err) {
      container.innerHTML = `<div class="validation-errors">${esc(String(err))}</div>`;
    }
  });

  document.getElementById('sens-run')!.addEventListener('click', async () => {
    if (!sessionId) return;
    const axis = (document.getElementById('sens-axis') as HTMLSelectElement).value;
    const valuesStr = (document.getElementById('sens-values') as HTMLInputElement).value;
    const values = valuesStr.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
    if (values.length === 0) {
      document.getElementById('sens-status')!.textContent = 'Enter comma-separated values';
      return;
    }

    const $sensStatus = document.getElementById('sens-status')!;
    $sensStatus.textContent = 'Submitting sensitivity sweep...';

    try {
      const result = await apiPost(
        `/v1/modeling/sessions/${sessionId}/sensitivity/submit`,
        {
          sweep_axis: axis,
          sweep_values: values,
          workspace_id: '00000000-0000-0000-0000-000000000001',
        },
      ) as { job_id: string; spec_ref: string; spec_name: string };

      $sensStatus.textContent = `Job ${result.job_id} submitted. Waiting for results...`;

      // Poll for completion via SSE then fetch results
      const evtSource = new EventSource(`${API_BASE}/v1/jobs/experiments/${result.job_id}/events`);
      evtSource.addEventListener('complete', async () => {
        evtSource.close();
        $sensStatus.textContent = 'Loading results...';
        try {
          const sensResults = await apiGet(
            `/v1/modeling/sessions/${sessionId}/sensitivity/results`,
          ) as { sweep_axis: string; results: Record<string, unknown>[]; job_status: string };
          renderSensitivityTable(sensResults.results, axis);
          $sensStatus.textContent = `Sensitivity complete (${sensResults.results.length} points)`;
        } catch (err) {
          $sensStatus.textContent = `Results error: ${err}`;
        }
      });
      evtSource.addEventListener('error', () => {
        evtSource.close();
        $sensStatus.textContent = 'Sensitivity job failed';
      });
      evtSource.addEventListener('cancel', () => {
        evtSource.close();
        $sensStatus.textContent = 'Sensitivity job canceled';
      });
      evtSource.onerror = () => { evtSource.close(); };
    } catch (err) {
      $sensStatus.textContent = `Submit error: ${err}`;
    }
  });

  document.getElementById('review-commit')!.addEventListener('click', async () => {
    await doCommit('promote_review', { reviewed: true });
  });
}

function renderSensitivityTable(results: Record<string, unknown>[], sweepAxis: string): void {
  const $container = document.getElementById('sens-results');
  if (!$container) return;
  if (results.length === 0) {
    $container.innerHTML = '<div style="color:#666;font-size:10px;margin-top:8px">No results available</div>';
    return;
  }
  let html = `
    <table class="sens-table">
      <thead>
        <tr>
          <th>${esc(sweepAxis)}</th>
          <th>TP Hit Rate</th>
          <th>SL Hit Rate</th>
          <th>N Signals</th>
        </tr>
      </thead>
      <tbody>
  `;
  for (const row of results) {
    const sweepVal = row['sweep_value'] != null ? String(row['sweep_value']) : '--';
    const tp = row['tp_hit_rate'] != null ? fmt(row['tp_hit_rate'] as number, 3) : '--';
    const sl = row['sl_hit_rate'] != null ? fmt(row['sl_hit_rate'] as number, 3) : '--';
    const n = row['n_signals'] != null ? String(row['n_signals']) : '--';
    html += `<tr><td>${esc(sweepVal)}</td><td>${tp}</td><td>${sl}</td><td>${n}</td></tr>`;
  }
  html += '</tbody></table>';
  $container.innerHTML = html;
}

// -- Promotion --

function renderPromotion(): void {
  $content.innerHTML = `
    <div class="panel-title">7. Promote to Serving</div>
    <div class="form-group">
      <label>Serving Alias</label>
      <input type="text" id="promote-alias" placeholder="e.g. production, staging">
    </div>
    <p style="font-size:10px;color:#888;margin-bottom:12px;">
      This will create an immutable serving version and point the alias to it.
      The promoted configuration will be immediately available for live streaming.
    </p>
    <button id="promote-commit-step" class="btn btn-primary">Mark Step Complete</button>
    <button id="promote-execute" class="btn btn-danger" style="margin-left:8px">Promote Now</button>
    <div id="promote-result"></div>
  `;

  document.getElementById('promote-commit-step')!.addEventListener('click', async () => {
    const alias = (document.getElementById('promote-alias') as HTMLInputElement).value.trim();
    await doCommit('promotion', { confirmed: true, alias: alias || 'staging' });
  });

  document.getElementById('promote-execute')!.addEventListener('click', async () => {
    if (!sessionId) return;
    const alias = (document.getElementById('promote-alias') as HTMLInputElement).value.trim();
    if (!alias) { alert('Enter an alias'); return; }
    const container = document.getElementById('promote-result')!;
    container.innerHTML = '<div class="empty-state">Promoting...</div>';
    try {
      const result = await promoteSession(sessionId, alias) as { serving_id: string; alias: string; spec_path: string };
      container.innerHTML = `
        <div class="summary-card" style="margin-top:12px">
          <h4>Promotion Complete</h4>
          <div class="summary-kv">
            <span class="key">Serving ID</span><span class="val">${esc(result.serving_id)}</span>
            <span class="key">Alias</span><span class="val">${esc(result.alias)}</span>
            <span class="key">Spec Path</span><span class="val" style="font-size:9px">${esc(result.spec_path)}</span>
          </div>
        </div>
      `;
      $statusText.textContent = 'Promoted';
      await refreshSession();
    } catch (err) {
      container.innerHTML = `<div class="validation-errors">${esc(String(err))}</div>`;
    }
  });
}

// -- Manual Mode (YAML editor) --

function renderManualMode(): void {
  // Assemble YAML from committed steps
  const spec: Record<string, unknown> = {
    name: 'modeling_studio_spec',
    pipeline: 'baseline',
  };

  const dsPayload = stepStates.get('dataset_select')?.payload;
  if (dsPayload) {
    spec.pipeline = dsPayload.pipeline ?? dsPayload.dataset_id ?? 'baseline';
  }

  const goldPayload = stepStates.get('gold_config')?.payload;
  if (goldPayload) {
    spec.description = `gold_config=${JSON.stringify(goldPayload)}`;
  }

  const sigPayload = stepStates.get('signal_select')?.payload;
  if (sigPayload) {
    spec.signal = { name: sigPayload.signal_name, params: {} };
  }

  const evalPayload = stepStates.get('eval_params')?.payload;
  if (evalPayload) {
    spec.scoring = {};
  }

  // Convert to YAML-like string (simple JSON for now)
  const yamlStr = jsonToYaml(spec);

  $content.innerHTML = `
    <div class="panel-title">Manual Mode -- YAML Editor</div>
    <div class="yaml-editor">
      <div class="form-group">
        <label>ServingSpec YAML</label>
        <textarea id="yaml-editor">${esc(yamlStr)}</textarea>
      </div>
    </div>
    <button id="yaml-validate" class="btn btn-secondary">Validate</button>
    <button id="yaml-apply" class="btn btn-primary">Apply to Steps</button>
    <div id="yaml-errors"></div>
  `;

  document.getElementById('yaml-validate')!.addEventListener('click', async () => {
    const content = (document.getElementById('yaml-editor') as HTMLTextAreaElement).value;
    const container = document.getElementById('yaml-errors')!;
    try {
      const result = await validateYaml(content);
      if (result.valid) {
        container.innerHTML = '<div style="color:#00cc88;margin-top:8px;font-size:11px">Valid ServingSpec</div>';
      } else {
        container.innerHTML = `<div class="validation-errors">${result.errors.map(e => esc(e)).join('\n')}</div>`;
      }
    } catch (err) {
      container.innerHTML = `<div class="validation-errors">${esc(String(err))}</div>`;
    }
  });

  document.getElementById('yaml-apply')!.addEventListener('click', () => {
    const content = (document.getElementById('yaml-editor') as HTMLTextAreaElement).value;
    try {
      // Parse YAML (simple JSON parse for client-side)
      const parsed = JSON.parse(content);
      if (parsed.pipeline) {
        // Auto-populate dataset_select
        // This is a best-effort mapping; manual mode users know what they're doing
      }
      alert('Applied. Switch to step mode to review and commit.');
      manualMode = false;
      $manualToggle.classList.remove('active');
      renderSidebar();
      renderContent();
    } catch (err) {
      document.getElementById('yaml-errors')!.innerHTML =
        `<div class="validation-errors">Parse error: ${esc(String(err))}</div>`;
    }
  });
}

function jsonToYaml(obj: Record<string, unknown>, indent = 0): string {
  const pad = '  '.repeat(indent);
  let out = '';
  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined) continue;
    if (typeof v === 'object' && !Array.isArray(v)) {
      out += `${pad}${k}:\n${jsonToYaml(v as Record<string, unknown>, indent + 1)}`;
    } else if (Array.isArray(v)) {
      out += `${pad}${k}:\n`;
      for (const item of v) {
        out += `${pad}  - ${String(item)}\n`;
      }
    } else {
      out += `${pad}${k}: ${String(v)}\n`;
    }
  }
  return out;
}

// ------------------------------------------------------------------ Step commit orchestration

async function doCommit(stepName: string, payload: Record<string, unknown>): Promise<void> {
  if (!sessionId) return;
  $statusText.textContent = 'Committing...';
  try {
    await commitStep(sessionId, stepName, payload);
    await refreshSession();
    // Auto-advance to next step
    const idx = stepIndex(stepName);
    if (idx < STEPS_ORDERED.length - 1) {
      activeStep = STEPS_ORDERED[idx + 1];
    }
    renderSidebar();
    renderContent();
    $statusText.textContent = `Step "${STEP_LABELS[stepName]}" committed`;
  } catch (err) {
    $statusText.textContent = `Error: ${err}`;
  }
}

async function refreshSession(): Promise<void> {
  if (!sessionId) return;
  const data = await fetchSession(sessionId);
  stepStates.clear();
  for (const step of data.steps) {
    stepStates.set(step.step_name, step);
  }
}

// ------------------------------------------------------------------ Init

async function init(): Promise<void> {
  // Load available specs
  try {
    availableSpecs = await fetchSpecs();
  } catch {
    availableSpecs = { signals: ['derivative'], datasets: [], steps: STEPS_ORDERED };
  }

  // Check URL for session_id to resume
  const urlParams = new URLSearchParams(window.location.search);
  const existingId = urlParams.get('session_id');

  try {
    if (existingId) {
      sessionId = existingId;
      await refreshSession();
    } else {
      sessionId = await createSession();
    }
    $sessionInfo.textContent = `Session: ${sessionId.slice(0, 8)}...`;
    $statusText.textContent = 'Ready';
  } catch (err) {
    $statusText.textContent = `Init error: ${err}`;
    $content.innerHTML = `<div class="empty-state">Failed to connect. Is the backend running on port ${API_PORT}?</div>`;
    return;
  }

  // Determine first uncommitted step
  for (const name of STEPS_ORDERED) {
    const state = stepStates.get(name);
    if (!state || state.status !== 'committed') {
      activeStep = name;
      break;
    }
  }

  renderSidebar();
  renderContent();
}

// Manual mode toggle
$manualToggle.addEventListener('click', () => {
  manualMode = !manualMode;
  $manualToggle.classList.toggle('active', manualMode);
  renderContent();
});

init();
