/**
 * Job Monitor -- displays experiment jobs, streams live events via SSE,
 * and provides submit/cancel controls. Auto-refreshes the job list
 * every 5 seconds.
 */

// ---------- Types ----------

interface JobSummary {
  job_id: string;
  workspace_id: string;
  spec_ref: string;
  status: string;
  progress: Record<string, unknown> | null;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

interface JobEvent {
  sequence: number;
  event_type: string;
  payload: Record<string, unknown> | null;
  created_at: string;
}

interface ArtifactRecord {
  artifact_id: string;
  artifact_type: string;
  uri: string;
  checksum: string;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

// ---------- DOM refs ----------

const $specSelect = document.getElementById('spec-select') as HTMLSelectElement;
const $workspaceInput = document.getElementById('workspace-input') as HTMLInputElement;
const $btnSubmit = document.getElementById('btn-submit') as HTMLButtonElement;
const $jobTbody = document.getElementById('job-tbody') as HTMLTableSectionElement;
const $statusText = document.getElementById('status-text') as HTMLSpanElement;
const $detailPanel = document.getElementById('detail-panel') as HTMLDivElement;
const $detailTitle = document.getElementById('detail-title') as HTMLHeadingElement;
const $detailStatus = document.getElementById('detail-status') as HTMLDivElement;
const $progressWrap = document.getElementById('progress-wrap') as HTMLDivElement;
const $progressFill = document.getElementById('progress-fill') as HTMLDivElement;
const $btnCancel = document.getElementById('btn-cancel') as HTMLButtonElement;
const $eventsLog = document.getElementById('events-log') as HTMLDivElement;
const $artifactsList = document.getElementById('artifacts-list') as HTMLDivElement;

const API_PORT = 8002;
const API_BASE = `http://localhost:${API_PORT}`;

// ---------- State ----------

let currentJobs: JobSummary[] = [];
let selectedJobId: string | null = null;
let activeSSE: EventSource | null = null;
let refreshTimer: number | null = null;

// ---------- Helpers ----------

function fmtTime(iso: string | null): string {
  if (!iso) return '--';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', { hour12: false });
  } catch {
    return iso;
  }
}

function badgeClass(status: string): string {
  switch (status) {
    case 'pending':   return 'badge-pending';
    case 'running':   return 'badge-running';
    case 'completed': return 'badge-completed';
    case 'failed':    return 'badge-failed';
    case 'canceled':  return 'badge-canceled';
    default:          return 'badge-pending';
  }
}

function shortId(id: string): string {
  return id.substring(0, 8);
}

// ---------- API calls ----------

async function fetchSpecs(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/v1/jobs/experiments/specs`);
    const data = await res.json();
    return data.specs || [];
  } catch {
    return [];
  }
}

async function fetchJobs(): Promise<JobSummary[]> {
  try {
    const res = await fetch(`${API_BASE}/v1/jobs/experiments`);
    const data = await res.json();
    return data.jobs || [];
  } catch {
    return [];
  }
}

async function fetchJobDetail(jobId: string): Promise<JobSummary | null> {
  try {
    const res = await fetch(`${API_BASE}/v1/jobs/experiments/${jobId}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function fetchArtifacts(jobId: string): Promise<ArtifactRecord[]> {
  try {
    const res = await fetch(`${API_BASE}/v1/jobs/experiments/${jobId}/artifacts`);
    const data = await res.json();
    return data.artifacts || [];
  } catch {
    return [];
  }
}

async function submitJob(specRef: string, workspaceId: string): Promise<string | null> {
  try {
    const res = await fetch(`${API_BASE}/v1/jobs/experiments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spec_ref: specRef, workspace_id: workspaceId }),
    });
    const data = await res.json();
    return data.job_id || null;
  } catch {
    return null;
  }
}

async function cancelJob(jobId: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/v1/jobs/experiments/${jobId}/cancel`, {
      method: 'POST',
    });
    return res.ok;
  } catch {
    return false;
  }
}

// ---------- Render ----------

function renderJobTable(): void {
  $jobTbody.innerHTML = '';
  for (const job of currentJobs) {
    const tr = document.createElement('tr');
    if (job.job_id === selectedJobId) tr.classList.add('selected');

    tr.innerHTML = `
      <td><span class="badge ${badgeClass(job.status)}">${job.status}</span></td>
      <td>${job.spec_ref}</td>
      <td>${fmtTime(job.created_at)}</td>
      <td>${fmtTime(job.started_at)}</td>
      <td>${fmtTime(job.completed_at)}</td>
      <td title="${job.job_id}">${shortId(job.job_id)}</td>
    `;
    tr.addEventListener('click', () => selectJob(job.job_id));
    $jobTbody.appendChild(tr);
  }
  $statusText.textContent = `${currentJobs.length} jobs`;
}

function renderDetailPanel(job: JobSummary): void {
  $detailTitle.textContent = `Job ${shortId(job.job_id)}`;
  $detailStatus.innerHTML = `
    <span class="key">Status</span><span class="val"><span class="badge ${badgeClass(job.status)}">${job.status}</span></span>
    <span class="key">Spec</span><span class="val">${job.spec_ref}</span>
    <span class="key">Created</span><span class="val">${fmtTime(job.created_at)}</span>
    <span class="key">Started</span><span class="val">${fmtTime(job.started_at)}</span>
    <span class="key">Completed</span><span class="val">${fmtTime(job.completed_at)}</span>
    ${job.error_message ? `<span class="key">Error</span><span class="val" style="color:#ff4444">${job.error_message}</span>` : ''}
  `;

  const isRunning = job.status === 'running' || job.status === 'pending';
  $progressWrap.style.display = isRunning ? 'block' : 'none';
  $btnCancel.disabled = !isRunning;
  $detailPanel.classList.add('visible');
}

function renderArtifacts(artifacts: ArtifactRecord[]): void {
  if (artifacts.length === 0) {
    $artifactsList.innerHTML = '<span style="color:#555;font-size:10px;">No artifacts</span>';
    return;
  }
  $artifactsList.innerHTML = artifacts.map(a => `
    <div class="artifact-row">
      <span class="art-type">${a.artifact_type}</span>
      <span class="art-uri">${a.uri}</span>
    </div>
  `).join('');
}

function appendEvent(evt: JobEvent): void {
  const line = document.createElement('div');
  line.className = 'event-line';
  const payloadStr = evt.payload ? JSON.stringify(evt.payload) : '';
  line.innerHTML = `
    <span class="event-time">${fmtTime(evt.created_at)}</span>
    <span class="event-type">${evt.event_type}</span>
    <span>${payloadStr}</span>
  `;
  $eventsLog.appendChild(line);
  $eventsLog.scrollTop = $eventsLog.scrollHeight;
}

// ---------- SSE ----------

function startSSE(jobId: string): void {
  stopSSE();
  $eventsLog.innerHTML = '';

  const url = `${API_BASE}/v1/jobs/experiments/${jobId}/events`;
  activeSSE = new EventSource(url);

  activeSSE.addEventListener('start', (e: MessageEvent) => {
    appendEvent(JSON.parse(e.data));
  });
  activeSSE.addEventListener('progress', (e: MessageEvent) => {
    const data = JSON.parse(e.data);
    appendEvent(data);
    // Update progress bar with a pulsing animation
    $progressFill.style.width = '100%';
    $progressFill.style.opacity = '0.6';
    setTimeout(() => { $progressFill.style.opacity = '1'; }, 500);
  });
  activeSSE.addEventListener('complete', (e: MessageEvent) => {
    appendEvent(JSON.parse(e.data));
    refreshAll();
  });
  activeSSE.addEventListener('error', (e: MessageEvent) => {
    if (e.data) appendEvent(JSON.parse(e.data));
    refreshAll();
  });
  activeSSE.addEventListener('cancel', (e: MessageEvent) => {
    appendEvent(JSON.parse(e.data));
    refreshAll();
  });
  activeSSE.addEventListener('done', (e: MessageEvent) => {
    appendEvent(JSON.parse(e.data));
    stopSSE();
    refreshAll();
  });
  activeSSE.onerror = () => {
    // Connection lost -- will auto-retry per SSE spec
  };
}

function stopSSE(): void {
  if (activeSSE) {
    activeSSE.close();
    activeSSE = null;
  }
}

// ---------- Actions ----------

async function selectJob(jobId: string): Promise<void> {
  selectedJobId = jobId;
  renderJobTable();

  const detail = await fetchJobDetail(jobId);
  if (!detail) return;
  renderDetailPanel(detail);

  const artifacts = await fetchArtifacts(jobId);
  renderArtifacts(artifacts);

  // Start SSE for non-terminal jobs
  if (detail.status === 'running' || detail.status === 'pending') {
    startSSE(jobId);
  } else {
    stopSSE();
    // Load historical events
    $eventsLog.innerHTML = '<span style="color:#555;font-size:10px;">Job completed. Events are not streamed for terminal jobs.</span>';
  }
}

async function refreshAll(): Promise<void> {
  currentJobs = await fetchJobs();
  renderJobTable();

  if (selectedJobId) {
    const detail = await fetchJobDetail(selectedJobId);
    if (detail) renderDetailPanel(detail);
  }
}

// ---------- Init ----------

async function init(): Promise<void> {
  // Load specs
  const specs = await fetchSpecs();
  $specSelect.innerHTML = specs.length > 0
    ? specs.map(s => `<option value="${s}">${s}</option>`).join('')
    : '<option value="">No specs found</option>';

  // Submit handler
  $btnSubmit.addEventListener('click', async () => {
    const specRef = $specSelect.value;
    const wsId = $workspaceInput.value.trim();
    if (!specRef || !wsId) return;

    $btnSubmit.disabled = true;
    const jobId = await submitJob(specRef, wsId);
    $btnSubmit.disabled = false;

    if (jobId) {
      await refreshAll();
      await selectJob(jobId);
    }
  });

  // Cancel handler
  $btnCancel.addEventListener('click', async () => {
    if (!selectedJobId) return;
    $btnCancel.disabled = true;
    await cancelJob(selectedJobId);
    await refreshAll();
  });

  // Initial load
  await refreshAll();

  // Auto-refresh every 5s
  refreshTimer = window.setInterval(refreshAll, 5000);
}

init();
