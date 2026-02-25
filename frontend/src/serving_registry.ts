/**
 * Serving Registry UI — manages immutable serving versions, mutable
 * aliases, activation history, and runtime snapshot diffs.
 *
 * Pure TypeScript, no external UI libraries. Matches the existing
 * frontend codebase style (experiments.ts).
 */

// ---------- Types ----------

interface VersionSummary {
  serving_id: string;
  description: string;
  source_experiment_name: string;
  source_run_id: string;
  source_config_hash: string;
  created_at_utc: string;
  spec_path: string;
}

interface VersionDetail extends VersionSummary {
  runtime_snapshot: Record<string, unknown>;
}

interface AliasSummary {
  alias: string;
  serving_id: string;
  updated_at_utc: string;
}

interface AliasDetail extends AliasSummary {
  history: ActivationEvent[];
}

interface ActivationEvent {
  event_id: number;
  alias: string;
  from_serving_id: string | null;
  to_serving_id: string;
  actor: string;
  timestamp_utc: string;
}

interface DiffResult {
  serving_id_a: string;
  serving_id_b: string;
  added: Record<string, unknown>;
  removed: Record<string, unknown>;
  changed: Record<string, { a: unknown; b: unknown }>;
  unchanged_count: number;
  summary: string;
}

interface ActivateResponse {
  alias: string;
  serving_id: string;
  from_serving_id: string | null;
  activated_at_utc: string;
  reason: string | null;
}

// ---------- Constants ----------

const API_PORT = 8002;
const REFRESH_INTERVAL_MS = 10_000;

// ---------- DOM refs ----------

const $statusText = document.getElementById('status-text') as HTMLSpanElement;
const $versionsTbody = document.getElementById('versions-tbody') as HTMLTableSectionElement;
const $aliasesTbody = document.getElementById('aliases-tbody') as HTMLTableSectionElement;
const $aliasDetailSection = document.getElementById('alias-detail-section') as HTMLDivElement;
const $diffSelectA = document.getElementById('diff-select-a') as HTMLSelectElement;
const $diffSelectB = document.getElementById('diff-select-b') as HTMLSelectElement;
const $btnDiff = document.getElementById('btn-diff') as HTMLButtonElement;
const $diffOutput = document.getElementById('diff-output') as HTMLDivElement;
const $confirmOverlay = document.getElementById('confirm-overlay') as HTMLDivElement;
const $confirmTitle = document.getElementById('confirm-title') as HTMLHeadingElement;
const $confirmMessage = document.getElementById('confirm-message') as HTMLParagraphElement;
const $confirmCancel = document.getElementById('confirm-cancel') as HTMLButtonElement;
const $confirmOk = document.getElementById('confirm-ok') as HTMLButtonElement;

// ---------- State ----------

let versions: VersionSummary[] = [];
let aliases: AliasSummary[] = [];
let expandedVersionId: string | null = null;
let selectedAlias: string | null = null;

// ---------- Helpers ----------

function esc(s: string): string {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 3) + '...' : s;
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

// ---------- Confirmation dialog ----------

function showConfirm(title: string, message: string): Promise<boolean> {
  return new Promise((resolve) => {
    $confirmTitle.textContent = title;
    $confirmMessage.textContent = message;
    $confirmOverlay.classList.add('visible');

    const cleanup = () => {
      $confirmOverlay.classList.remove('visible');
      $confirmCancel.onclick = null;
      $confirmOk.onclick = null;
    };

    $confirmCancel.onclick = () => { cleanup(); resolve(false); };
    $confirmOk.onclick = () => { cleanup(); resolve(true); };
  });
}

// ---------- API ----------

async function apiGet<T>(path: string): Promise<T> {
  const resp = await fetch(`http://localhost:${API_PORT}${path}`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status}: ${text}`);
  }
  return resp.json();
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`http://localhost:${API_PORT}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status}: ${text}`);
  }
  return resp.json();
}

async function fetchVersions(): Promise<VersionSummary[]> {
  return apiGet<VersionSummary[]>('/v1/serving/versions');
}

async function fetchVersionDetail(servingId: string): Promise<VersionDetail> {
  return apiGet<VersionDetail>(`/v1/serving/versions/${encodeURIComponent(servingId)}`);
}

async function fetchAliases(): Promise<AliasSummary[]> {
  return apiGet<AliasSummary[]>('/v1/serving/aliases');
}

async function fetchAliasDetail(alias: string): Promise<AliasDetail> {
  return apiGet<AliasDetail>(`/v1/serving/aliases/${encodeURIComponent(alias)}`);
}

async function activateAlias(alias: string, servingId: string, reason: string | null): Promise<ActivateResponse> {
  return apiPost<ActivateResponse>(
    `/v1/serving/aliases/${encodeURIComponent(alias)}/activate`,
    { serving_id: servingId, reason },
  );
}

async function fetchDiff(idA: string, idB: string): Promise<DiffResult> {
  return apiPost<DiffResult>('/v1/serving/diff', { serving_id_a: idA, serving_id_b: idB });
}

// ---------- Versions panel ----------

function renderVersions(): void {
  $versionsTbody.innerHTML = '';

  if (versions.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="6" style="text-align:center;color:#555;padding:40px;">No serving versions found. Promote an experiment first.</td>`;
    $versionsTbody.appendChild(tr);
    return;
  }

  for (const v of versions) {
    const tr = document.createElement('tr');
    const isExpanded = v.serving_id === expandedVersionId;
    if (isExpanded) tr.classList.add('selected');

    tr.innerHTML = `
      <td title="${esc(v.serving_id)}">${esc(truncate(v.serving_id, 40))}</td>
      <td>${esc(v.description || '--')}</td>
      <td>${esc(v.source_experiment_name)}</td>
      <td title="${esc(v.source_run_id)}">${esc(truncate(v.source_run_id, 16))}</td>
      <td>${esc(formatTime(v.created_at_utc))}</td>
      <td><button class="btn btn-secondary btn-set-active" data-sid="${esc(v.serving_id)}">Set as active</button></td>
    `;

    tr.addEventListener('click', (e) => {
      if ((e.target as HTMLElement).classList.contains('btn-set-active')) return;
      toggleVersionExpand(v.serving_id);
    });

    $versionsTbody.appendChild(tr);

    if (isExpanded) {
      const detailTr = document.createElement('tr');
      detailTr.classList.add('snapshot-row');
      detailTr.innerHTML = `<td colspan="6"><div class="snapshot-content" id="snapshot-${esc(v.serving_id)}">Loading snapshot...</div></td>`;
      $versionsTbody.appendChild(detailTr);
      loadSnapshot(v.serving_id);
    }
  }

  // Wire up "Set as active" buttons.
  document.querySelectorAll('.btn-set-active').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const sid = (btn as HTMLElement).dataset.sid!;
      await promptSetActive(sid);
    });
  });
}

async function toggleVersionExpand(servingId: string): Promise<void> {
  expandedVersionId = expandedVersionId === servingId ? null : servingId;
  renderVersions();
}

async function loadSnapshot(servingId: string): Promise<void> {
  const el = document.getElementById(`snapshot-${servingId}`);
  if (!el) return;
  try {
    const detail = await fetchVersionDetail(servingId);
    el.textContent = JSON.stringify(detail.runtime_snapshot, null, 2);
  } catch (err) {
    el.innerHTML = `<span class="msg-error">Failed to load: ${esc(String(err))}</span>`;
  }
}

async function promptSetActive(servingId: string): Promise<void> {
  // Ask which alias to set.
  const aliasName = prompt('Enter alias name to activate (e.g. "vp_main"):');
  if (!aliasName || !aliasName.trim()) return;
  const alias = aliasName.trim().toLowerCase();

  const confirmed = await showConfirm(
    'Activate Serving Version',
    `Point alias "${alias}" to serving version "${servingId}"?`,
  );
  if (!confirmed) return;

  try {
    await activateAlias(alias, servingId, 'Set via registry UI');
    $statusText.textContent = `Activated ${alias} -> ${truncate(servingId, 30)}`;
    await refreshAll();
  } catch (err) {
    $statusText.textContent = `Error: ${err}`;
  }
}

// ---------- Aliases panel ----------

function renderAliases(): void {
  $aliasesTbody.innerHTML = '';

  if (aliases.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="4" style="text-align:center;color:#555;padding:40px;">No aliases defined.</td>`;
    $aliasesTbody.appendChild(tr);
    return;
  }

  for (const a of aliases) {
    const tr = document.createElement('tr');
    if (a.alias === selectedAlias) tr.classList.add('selected');

    tr.innerHTML = `
      <td>${esc(a.alias)}</td>
      <td title="${esc(a.serving_id)}">${esc(truncate(a.serving_id, 40))}</td>
      <td>${esc(formatTime(a.updated_at_utc))}</td>
      <td>
        <button class="btn btn-secondary btn-alias-history" data-alias="${esc(a.alias)}">History</button>
        <button class="btn btn-danger btn-alias-rollback" data-alias="${esc(a.alias)}">Rollback</button>
      </td>
    `;

    $aliasesTbody.appendChild(tr);
  }

  // Wire history buttons.
  document.querySelectorAll('.btn-alias-history').forEach((btn) => {
    btn.addEventListener('click', () => {
      const alias = (btn as HTMLElement).dataset.alias!;
      selectAlias(alias);
    });
  });

  // Wire rollback buttons.
  document.querySelectorAll('.btn-alias-rollback').forEach((btn) => {
    btn.addEventListener('click', () => {
      const alias = (btn as HTMLElement).dataset.alias!;
      rollbackAlias(alias);
    });
  });
}

async function selectAlias(alias: string): Promise<void> {
  selectedAlias = alias;
  renderAliases();

  $aliasDetailSection.innerHTML = `<div class="alias-detail-aside"><h4>Loading history for "${esc(alias)}"...</h4></div>`;

  try {
    const detail = await fetchAliasDetail(alias);
    renderAliasDetail(detail);
  } catch (err) {
    $aliasDetailSection.innerHTML = `<div class="alias-detail-aside"><span class="msg-error">Failed: ${esc(String(err))}</span></div>`;
  }
}

function renderAliasDetail(detail: AliasDetail): void {
  let html = `<div class="alias-detail-aside">`;
  html += `<h4>Alias: ${esc(detail.alias)} -> ${esc(truncate(detail.serving_id, 40))}</h4>`;
  html += `<p style="color:#888;font-size:10px;margin-bottom:8px;">Updated: ${esc(formatTime(detail.updated_at_utc))}</p>`;

  if (detail.history.length === 0) {
    html += `<p class="msg-info">No activation history.</p>`;
  } else {
    html += `<ul class="history-list">`;
    for (const ev of detail.history) {
      html += `<li>`;
      html += `<span class="history-time">${esc(formatTime(ev.timestamp_utc))}</span>`;
      if (ev.from_serving_id) {
        html += `<span class="history-from">${esc(truncate(ev.from_serving_id, 30))}</span> `;
      } else {
        html += `<span style="color:#555">(none)</span> `;
      }
      html += `-> <span class="history-to">${esc(truncate(ev.to_serving_id, 30))}</span>`;
      html += ` <span class="history-actor">[${esc(ev.actor)}]</span>`;
      html += `</li>`;
    }
    html += `</ul>`;
  }

  html += `</div>`;
  $aliasDetailSection.innerHTML = html;
}

async function rollbackAlias(alias: string): Promise<void> {
  // Fetch history to find previous version.
  let detail: AliasDetail;
  try {
    detail = await fetchAliasDetail(alias);
  } catch (err) {
    $statusText.textContent = `Error loading alias: ${err}`;
    return;
  }

  if (detail.history.length < 2) {
    $statusText.textContent = `Cannot rollback: alias "${alias}" has no previous version.`;
    return;
  }

  // History is newest-first. The current activation is [0], previous is [1].
  const previousEvent = detail.history[1];
  const rollbackTo = previousEvent.to_serving_id;

  const confirmed = await showConfirm(
    'Rollback Alias',
    `Rollback alias "${alias}" from "${truncate(detail.serving_id, 30)}" to previous version "${truncate(rollbackTo, 30)}"?`,
  );
  if (!confirmed) return;

  try {
    await activateAlias(alias, rollbackTo, 'Rollback via registry UI');
    $statusText.textContent = `Rolled back ${alias} -> ${truncate(rollbackTo, 30)}`;
    await refreshAll();
    if (selectedAlias === alias) {
      await selectAlias(alias);
    }
  } catch (err) {
    $statusText.textContent = `Rollback error: ${err}`;
  }
}

// ---------- Diff panel ----------

function populateDiffSelects(): void {
  const prevA = $diffSelectA.value;
  const prevB = $diffSelectB.value;

  $diffSelectA.innerHTML = '<option value="">Select version...</option>';
  $diffSelectB.innerHTML = '<option value="">Select version...</option>';

  for (const v of versions) {
    const label = `${v.serving_id} (${v.source_experiment_name})`;
    const optA = document.createElement('option');
    optA.value = v.serving_id;
    optA.textContent = label;
    $diffSelectA.appendChild(optA);

    const optB = document.createElement('option');
    optB.value = v.serving_id;
    optB.textContent = label;
    $diffSelectB.appendChild(optB);
  }

  if (prevA) $diffSelectA.value = prevA;
  if (prevB) $diffSelectB.value = prevB;
}

async function runDiff(): Promise<void> {
  const idA = $diffSelectA.value;
  const idB = $diffSelectB.value;

  if (!idA || !idB) {
    $diffOutput.innerHTML = '<p class="msg-info">Select two versions to compare.</p>';
    return;
  }
  if (idA === idB) {
    $diffOutput.innerHTML = '<p class="msg-info">Select two different versions.</p>';
    return;
  }

  $diffOutput.innerHTML = '<p class="msg-info">Computing diff...</p>';

  try {
    const diff = await fetchDiff(idA, idB);
    renderDiff(diff);
  } catch (err) {
    $diffOutput.innerHTML = `<p class="msg-error">Diff failed: ${esc(String(err))}</p>`;
  }
}

function renderDiff(diff: DiffResult): void {
  let html = `<div class="diff-result">`;
  html += `<div class="diff-summary">${esc(diff.summary)}</div>`;

  const addedKeys = Object.keys(diff.added);
  const removedKeys = Object.keys(diff.removed);
  const changedKeys = Object.keys(diff.changed);

  if (addedKeys.length > 0) {
    html += `<div class="diff-section"><h4>Added (in B only)</h4>`;
    for (const k of addedKeys) {
      html += `<div class="diff-row"><span class="diff-key">${esc(k)}</span>: <span class="diff-added">${esc(JSON.stringify(diff.added[k]))}</span></div>`;
    }
    html += `</div>`;
  }

  if (removedKeys.length > 0) {
    html += `<div class="diff-section"><h4>Removed (in A only)</h4>`;
    for (const k of removedKeys) {
      html += `<div class="diff-row"><span class="diff-key">${esc(k)}</span>: <span class="diff-removed">${esc(JSON.stringify(diff.removed[k]))}</span></div>`;
    }
    html += `</div>`;
  }

  if (changedKeys.length > 0) {
    html += `<div class="diff-section"><h4>Changed</h4>`;
    for (const k of changedKeys) {
      const entry = diff.changed[k];
      html += `<div class="diff-row">`;
      html += `<span class="diff-key">${esc(k)}</span>: `;
      html += `<span class="diff-changed-a">${esc(JSON.stringify(entry.a))}</span>`;
      html += ` -> `;
      html += `<span class="diff-changed-b">${esc(JSON.stringify(entry.b))}</span>`;
      html += `</div>`;
    }
    html += `</div>`;
  }

  if (addedKeys.length === 0 && removedKeys.length === 0 && changedKeys.length === 0) {
    html += `<p class="msg-success">Snapshots are identical (${diff.unchanged_count} keys).</p>`;
  }

  html += `</div>`;
  $diffOutput.innerHTML = html;
}

// ---------- Tab switching ----------

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const tab = (btn as HTMLElement).dataset.tab!;

    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');

    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    const panel = document.getElementById(`panel-${tab}`);
    if (panel) panel.classList.add('active');
  });
});

// ---------- Load & refresh ----------

async function refreshAll(): Promise<void> {
  $statusText.textContent = 'Loading...';
  try {
    [versions, aliases] = await Promise.all([fetchVersions(), fetchAliases()]);
    renderVersions();
    renderAliases();
    populateDiffSelects();
    $statusText.textContent = `${versions.length} versions, ${aliases.length} aliases`;
  } catch (err) {
    $statusText.textContent = `Error: ${err}`;
  }
}

// ---------- Event listeners ----------

$btnDiff.addEventListener('click', () => runDiff());

// ---------- Init ----------

refreshAll();

// Auto-refresh alias status every 10s.
setInterval(async () => {
  try {
    aliases = await fetchAliases();
    renderAliases();
  } catch {
    // Silently ignore refresh errors.
  }
}, REFRESH_INTERVAL_MS);
