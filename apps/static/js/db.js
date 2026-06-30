'use strict';

// ── Tabs ───────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn, .tab-panel').forEach(el => el.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
  });
});

// ── Toast ──────────────────────────────────────────────────────────────
const toast = document.getElementById('toast');
let toastTimer;

function showToast(msg, isError = false) {
  clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.className   = 'show' + (isError ? ' error' : '');
  toastTimer = setTimeout(() => { toast.className = ''; }, 3000);
}

// ── Helpers ────────────────────────────────────────────────────────────
const TYPE_LABELS = {
  fire:                  'Fire',
  unauthorized_presence: 'Unauthorized Presence',
  unauthorized_touch:    'Unauthorized Touch',
};

function typePill(type) {
  const label = TYPE_LABELS[type] ?? type;
  return `<span class="type-pill ${type}">${label}</span>`;
}

function statusPill(dismissed) {
  return dismissed
    ? `<span class="status-pill dismissed">Dismissed</span>`
    : `<span class="status-pill active">Active</span>`;
}

function toLocalInputValue(isoStr) {
  if (!isoStr) return '';
  return isoStr.replace('Z', '').slice(0, 16);
}

// ── Alerts table ───────────────────────────────────────────────────────
async function loadAlerts() {
  const resp  = await fetch('/api/alerts');
  const rows  = await resp.json();
  const tbody = document.getElementById('alerts-body');

  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="7">No alerts in database.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => `
    <tr data-id="${r.alert_id}">
      <td><code>${r.alert_id}</code></td>
      <td>${typePill(r.alert_type)}</td>
      <td>${r.room_id}</td>
      <td><code>${r.timestamp}</code></td>
      <td>${statusPill(r.dismissed)}</td>
      <td><code>${r.dismissed_at ?? '—'}</code></td>
      <td>
        <div class="actions">
          <button class="btn btn-ghost btn-sm" onclick="openEditModal(${r.alert_id})">Edit</button>
          <button class="btn btn-danger btn-sm" onclick="deleteAlert(${r.alert_id})">Delete</button>
        </div>
      </td>
    </tr>
  `).join('');
}

async function deleteAlert(id) {
  if (!confirm(`Delete alert #${id}? This also removes its dismiss log entries.`)) return;
  await fetch(`/api/alerts/${id}`, { method: 'DELETE' });
  showToast(`Alert #${id} deleted.`);
  loadAlerts();
  loadDismissLog();
}

// ── Dismiss log table ──────────────────────────────────────────────────
async function loadDismissLog() {
  const resp  = await fetch('/api/dismiss-log');
  const rows  = await resp.json();
  const tbody = document.getElementById('log-body');

  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="6">No dismiss records.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><code>${r.log_id}</code></td>
      <td><code>${r.alert_id}</code></td>
      <td>${typePill(r.alert_type)}</td>
      <td>${r.room_id}</td>
      <td><code>${r.dismissed_at}</code></td>
      <td>
        <button class="btn btn-danger btn-sm" onclick="deleteDismissLog(${r.log_id})">Delete</button>
      </td>
    </tr>
  `).join('');
}

async function deleteDismissLog(id) {
  if (!confirm(`Delete dismiss log entry #${id}?`)) return;
  await fetch(`/api/dismiss-log/${id}`, { method: 'DELETE' });
  showToast(`Log entry #${id} deleted.`);
  loadDismissLog();
}

// ── Modal ──────────────────────────────────────────────────────────────
const modal        = document.getElementById('alert-modal');
const modalTitle   = document.getElementById('modal-title');
const fId          = document.getElementById('modal-alert-id');
const fType        = document.getElementById('f-type');
const fRoom        = document.getElementById('f-room');
const fTs          = document.getElementById('f-ts');
const fStatus      = document.getElementById('f-status');
const fDismissedAt = document.getElementById('f-dismissed-at');
const fDismissWrap = document.getElementById('f-dismissed-wrap');

fStatus.addEventListener('change', () => {
  fDismissWrap.style.display = fStatus.value === '1' ? 'flex' : 'none';
});

document.getElementById('add-alert-btn').addEventListener('click', () => {
  modalTitle.textContent = 'Add Alert';
  fId.value   = '';
  fType.value = 'fire';
  fRoom.value = '1';
  const now = new Date();
  fTs.value = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  fStatus.value      = '0';
  fDismissedAt.value = '';
  fDismissWrap.style.display = 'none';
  modal.classList.add('open');
});

async function openEditModal(id) {
  const resp = await fetch('/api/alerts');
  const rows = await resp.json();
  const row  = rows.find(r => r.alert_id === id);
  if (!row) return;

  modalTitle.textContent     = `Edit Alert #${id}`;
  fId.value                  = id;
  fType.value                = row.alert_type;
  fRoom.value                = row.room_id;
  fTs.value                  = toLocalInputValue(row.timestamp);
  fStatus.value              = String(row.dismissed);
  fDismissedAt.value         = toLocalInputValue(row.dismissed_at);
  fDismissWrap.style.display = row.dismissed ? 'flex' : 'none';
  modal.classList.add('open');
}

document.getElementById('modal-cancel').addEventListener('click', () => {
  modal.classList.remove('open');
});

modal.addEventListener('click', (e) => {
  if (e.target === modal) modal.classList.remove('open');
});

document.getElementById('modal-save').addEventListener('click', async () => {
  const id = fId.value;
  const payload = {
    alert_type:   fType.value,
    room_id:      parseInt(fRoom.value, 10),
    timestamp:    fTs.value ? new Date(fTs.value).toISOString() : '',
    dismissed:    parseInt(fStatus.value, 10),
    dismissed_at: fStatus.value === '1' && fDismissedAt.value
      ? new Date(fDismissedAt.value).toISOString()
      : null,
  };

  const resp = await fetch(id ? `/api/alerts/${id}` : '/api/alerts', {
    method:  id ? 'PUT' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(payload),
  });

  if (resp.ok) {
    showToast(id ? `Alert #${id} updated.` : 'Alert created.');
    modal.classList.remove('open');
    loadAlerts();
  } else {
    const err = await resp.json();
    showToast(err.error ?? 'Save failed.', true);
  }
});

// ── Init ───────────────────────────────────────────────────────────────
loadAlerts();
loadDismissLog();
