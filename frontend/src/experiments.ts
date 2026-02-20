/**
 * Experiment Browser — fetches ranked experiment results from the backend
 * REST API and renders them in a sortable, filterable table. Clicking a run
 * opens its detail panel; "Launch Stream" opens the canonical streaming UI
 * for the run's dataset window (with derivative overrides when applicable).
 */

// ---------- Types ----------

interface RunSummary {
  run_id: string;
  signal_name: string;
  dataset_id: string;
  experiment_name: string;
  signal_params_json: string;
  threshold: number | null;
  cooldown_bins: number | null;
  tp_rate: number | null;
  sl_rate: number | null;
  timeout_rate: number | null;
  n_signals: number | null;
  mean_pnl_ticks: number | null;
  events_per_hour: number | null;
  eval_tp_ticks: number | null;
  eval_sl_ticks: number | null;
  timestamp_utc: string;
  streaming_url: string | null;
  can_stream: boolean;
}

interface RunsResponse {
  runs: RunSummary[];
  filters: {
    signals: string[];
    datasets: string[];
  };
}

interface ThresholdRow {
  threshold: number | null;
  cooldown_bins: number | null;
  n_signals: number | null;
  tp_rate: number | null;
  sl_rate: number | null;
  timeout_rate: number | null;
  mean_pnl_ticks: number | null;
  events_per_hour: number | null;
  median_time_to_outcome_ms: number | null;
  [key: string]: unknown;
}

interface RunDetail {
  meta: Record<string, unknown>;
  threshold_results: ThresholdRow[];
  streaming_url: string | null;
  can_stream: boolean;
  error?: string;
}

// ---------- DOM refs ----------

const $tbody = document.getElementById('runs-tbody') as HTMLTableSectionElement;
const $filterSignal = document.getElementById('filter-signal') as HTMLSelectElement;
const $filterDataset = document.getElementById('filter-dataset') as HTMLSelectElement;
const $filterSort = document.getElementById('filter-sort') as HTMLSelectElement;
const $filterMinSignals = document.getElementById('filter-min-signals') as HTMLInputElement;
const $statusText = document.getElementById('status-text') as HTMLSpanElement;
const $detailPanel = document.getElementById('detail-panel') as HTMLDivElement;
const $detailTitle = document.getElementById('detail-title') as HTMLHeadingElement;
const $detailMetrics = document.getElementById('detail-metrics') as HTMLDivElement;
const $detailParams = document.getElementById('detail-params') as HTMLDivElement;
const $detailThresholds = document.getElementById('detail-thresholds') as HTMLDivElement;
const $btnLaunch = document.getElementById('btn-launch') as HTMLButtonElement;
const $launchNote = document.getElementById('launch-note') as HTMLDivElement;

const API_PORT = 8002;

// ---------- State ----------

let currentRuns: RunSummary[] = [];
let selectedRunId: string | null = null;
let filtersPopulated = false;

// ---------- Helpers ----------

function fmt(v: number | null | undefined, decimals = 2): string {
  if (v === null || v === undefined) return '--';
  return v.toFixed(decimals);
}

function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined) return '--';
  return (v * 100).toFixed(1) + '%';
}

function tpClass(v: number | null | undefined): string {
  if (v === null || v === undefined) return '';
  if (v >= 0.4) return 'tp-high';
  if (v >= 0.33) return 'tp-mid';
  return 'tp-low';
}

// ---------- API ----------

async function fetchRuns(): Promise<RunsResponse> {
  const signal = $filterSignal.value || undefined;
  const dataset = $filterDataset.value || undefined;
  const sort = $filterSort.value;
  const minSignals = parseInt($filterMinSignals.value) || 5;

  const params = new URLSearchParams();
  if (signal) params.set('signal', signal);
  if (dataset) params.set('dataset_id', dataset);
  params.set('sort', sort);
  params.set('min_signals', String(minSignals));
  params.set('top_n', '100');

  const resp = await fetch(`http://localhost:${API_PORT}/v1/experiments/runs?${params}`);
  if (!resp.ok) throw new Error(`API error: ${resp.status}`);
  return resp.json();
}

async function fetchDetail(runId: string): Promise<RunDetail> {
  const resp = await fetch(`http://localhost:${API_PORT}/v1/experiments/runs/${runId}/detail`);
  if (!resp.ok) throw new Error(`API error: ${resp.status}`);
  return resp.json();
}

// ---------- Render table ----------

function renderTable(runs: RunSummary[]): void {
  $tbody.innerHTML = '';
  if (runs.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="13" style="text-align:center;color:#555;padding:40px;">No experiment runs found. Run experiments first via the harness CLI.</td>`;
    $tbody.appendChild(tr);
    return;
  }

  runs.forEach((run, i) => {
    const tr = document.createElement('tr');
    if (run.run_id === selectedRunId) tr.classList.add('selected');

    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${esc(run.signal_name)}</td>
      <td title="${esc(run.dataset_id)}">${esc(truncDs(run.dataset_id))}</td>
      <td class="${tpClass(run.tp_rate)}">${fmtPct(run.tp_rate)}</td>
      <td>${fmtPct(run.sl_rate)}</td>
      <td>${run.n_signals ?? '--'}</td>
      <td>${fmt(run.mean_pnl_ticks)}</td>
      <td>${fmt(run.events_per_hour, 1)}</td>
      <td>${run.eval_tp_ticks ?? '--'}</td>
      <td>${run.eval_sl_ticks ?? '--'}</td>
      <td>${fmt(run.threshold, 3)}</td>
      <td>${run.cooldown_bins ?? '--'}</td>
      <td class="${run.can_stream ? 'can-stream' : 'no-stream'}">${run.can_stream ? 'YES' : 'NO'}</td>
    `;

    tr.addEventListener('click', () => selectRun(run.run_id));
    $tbody.appendChild(tr);
  });
}

function esc(s: string): string {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function truncDs(ds: string): string {
  return ds.length > 28 ? ds.slice(0, 25) + '...' : ds;
}

// ---------- Populate filters ----------

function populateFilters(signals: string[], datasets: string[]): void {
  if (filtersPopulated) return;
  filtersPopulated = true;

  const prevSignal = $filterSignal.value;
  const prevDataset = $filterDataset.value;

  $filterSignal.innerHTML = '<option value="">All</option>';
  for (const s of signals) {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    $filterSignal.appendChild(opt);
  }
  if (prevSignal) $filterSignal.value = prevSignal;

  $filterDataset.innerHTML = '<option value="">All</option>';
  for (const d of datasets) {
    const opt = document.createElement('option');
    opt.value = d;
    opt.textContent = d;
    $filterDataset.appendChild(opt);
  }
  if (prevDataset) $filterDataset.value = prevDataset;
}

// ---------- Detail panel ----------

async function selectRun(runId: string): Promise<void> {
  selectedRunId = runId;

  // Highlight row
  const rows = $tbody.querySelectorAll('tr');
  rows.forEach(r => r.classList.remove('selected'));
  for (const r of rows) {
    if (r.querySelector('td')?.nextElementSibling?.textContent) {
      // Re-render simpler: just refresh table highlight
    }
  }
  renderTable(currentRuns);

  $detailPanel.classList.add('visible');
  $detailTitle.textContent = `Run ${runId.slice(0, 8)}...`;
  $detailMetrics.innerHTML = '<span style="color:#555">Loading...</span>';
  $detailParams.textContent = 'Loading...';
  $detailThresholds.innerHTML = '';
  $btnLaunch.disabled = true;
  $launchNote.textContent = '';

  try {
    const detail = await fetchDetail(runId);
    if (detail.error) {
      $detailMetrics.innerHTML = `<span style="color:#ff4444">${esc(detail.error)}</span>`;
      return;
    }

    const m = detail.meta;

    // Metrics
    $detailMetrics.innerHTML = `
      <span class="key">Signal</span><span class="val">${esc(String(m.signal_name ?? ''))}</span>
      <span class="key">Dataset</span><span class="val">${esc(String(m.dataset_id ?? ''))}</span>
      <span class="key">Experiment</span><span class="val">${esc(String(m.experiment_name ?? ''))}</span>
      <span class="key">TP ticks</span><span class="val">${m.eval_tp_ticks ?? '--'}</span>
      <span class="key">SL ticks</span><span class="val">${m.eval_sl_ticks ?? '--'}</span>
      <span class="key">Grid Variant</span><span class="val">${esc(String(m.grid_variant_id ?? 'immutable'))}</span>
      <span class="key">Config Hash</span><span class="val" style="font-size:9px">${esc(String(m.config_hash ?? '').slice(0, 16))}</span>
      <span class="key">Elapsed</span><span class="val">${fmt(m.elapsed_seconds as number | null, 1)}s</span>
      <span class="key">Timestamp</span><span class="val" style="font-size:9px">${esc(String(m.timestamp_utc ?? ''))}</span>
    `;

    // Params
    const params = m.signal_params ?? {};
    $detailParams.textContent = JSON.stringify(params, null, 2);

    // Threshold results
    if (detail.threshold_results.length > 0) {
      let html = `<table class="threshold-table">
        <thead><tr>
          <th>Threshold</th><th>Cooldown</th><th>N</th><th>TP%</th><th>SL%</th><th>PnL</th><th>Evt/hr</th>
        </tr></thead><tbody>`;
      for (const t of detail.threshold_results) {
        html += `<tr>
          <td>${fmt(t.threshold, 3)}</td>
          <td>${t.cooldown_bins ?? '--'}</td>
          <td>${t.n_signals ?? '--'}</td>
          <td class="${tpClass(t.tp_rate)}">${fmtPct(t.tp_rate)}</td>
          <td>${fmtPct(t.sl_rate)}</td>
          <td>${fmt(t.mean_pnl_ticks)}</td>
          <td>${fmt(t.events_per_hour, 1)}</td>
        </tr>`;
      }
      html += '</tbody></table>';
      $detailThresholds.innerHTML = html;
    } else {
      $detailThresholds.innerHTML = '<span style="color:#555">No threshold data</span>';
    }

    // Launch button
    if (detail.can_stream && detail.streaming_url) {
      $btnLaunch.disabled = false;
      $btnLaunch.onclick = () => {
        window.open(detail.streaming_url!, '_blank');
      };
      $launchNote.textContent = 'Opens streaming UI with this run\'s parameters';
    } else {
      $btnLaunch.disabled = true;
      $btnLaunch.onclick = null;
      $launchNote.textContent = 'Stream launch unavailable for this run (missing dataset metadata or unsupported params)';
    }

  } catch (err) {
    $detailMetrics.innerHTML = `<span style="color:#ff4444">Error: ${esc(String(err))}</span>`;
  }
}

// ---------- Load + refresh ----------

async function loadRuns(): Promise<void> {
  $statusText.textContent = 'Loading...';
  try {
    const data = await fetchRuns();
    currentRuns = data.runs;
    populateFilters(data.filters.signals, data.filters.datasets);
    renderTable(currentRuns);
    $statusText.textContent = `${currentRuns.length} runs`;
  } catch (err) {
    $statusText.textContent = `Error: ${err}`;
    $tbody.innerHTML = `<tr><td colspan="13" style="text-align:center;color:#ff4444;padding:40px;">
      Failed to load experiments. Is the backend running on port ${API_PORT}?
    </td></tr>`;
  }
}

// ---------- Event listeners ----------

$filterSignal.addEventListener('change', () => { filtersPopulated = false; loadRuns(); });
$filterDataset.addEventListener('change', () => { filtersPopulated = false; loadRuns(); });
$filterSort.addEventListener('change', () => loadRuns());
$filterMinSignals.addEventListener('change', () => loadRuns());

// Column header sorting
document.querySelectorAll('#runs-table th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = (th as HTMLElement).dataset.col!;
    if (col === 'rank') return;

    // Clear other sorted classes
    document.querySelectorAll('#runs-table th').forEach(h => {
      h.classList.remove('sorted-asc', 'sorted-desc');
    });

    // Client-side sort on current data
    const isDesc = !th.classList.contains('sorted-desc');
    th.classList.add(isDesc ? 'sorted-desc' : 'sorted-asc');

    currentRuns.sort((a, b) => {
      const av = (a as unknown as Record<string, unknown>)[col];
      const bv = (b as unknown as Record<string, unknown>)[col];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      const cmp = (av as number) < (bv as number) ? -1 : (av as number) > (bv as number) ? 1 : 0;
      return isDesc ? -cmp : cmp;
    });

    renderTable(currentRuns);
  });
});

// ---------- Init ----------

loadRuns();
