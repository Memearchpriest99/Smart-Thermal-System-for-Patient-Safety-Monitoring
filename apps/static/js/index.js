'use strict';

// ── Constants ──────────────────────────────────────────────────────────
const ROOM_TIMEOUT_MS = 3000;   // mark room Lost after this silence

const TYPE_LABELS = {
  fire:                  'Fire Detected',
  unauthorized_presence: 'Unauthorized Presence',
  unauthorized_touch:    'Unauthorized Touch',
};

// ── DOM references ─────────────────────────────────────────────────────
const alertList   = document.getElementById('alert-list');
const emptyState  = document.getElementById('empty-state');
const alertCount  = document.getElementById('alert-count');
const hbDot       = document.getElementById('hb-dot');
const hbLabel     = document.getElementById('hb-label');
const connBanner  = document.getElementById('conn-banner');
const soundBtn    = document.getElementById('sound-btn');
const roomGrid    = document.getElementById('room-grid');
const roomCount   = document.getElementById('room-count');

// ── Server connection (Socket.IO level) ────────────────────────────────
// Tracks whether the browser can reach the Flask server at all.
// Separate from room-level Pi heartbeats.

function setServerStatus(online) {
  if (online) {
    hbDot.className   = 'connected';
    hbLabel.className = 'connected';
    hbLabel.textContent = 'Server Online';
    connBanner.classList.remove('visible');
  } else {
    hbDot.className   = 'lost';
    hbLabel.className = 'lost';
    hbLabel.textContent = 'Server Offline';
    connBanner.classList.add('visible');
  }
}

// ── Per-room heartbeat tracking ────────────────────────────────────────
// roomStates: Map<room_id (number), { lastPing: number, lost: boolean }>

const roomStates = new Map();

function onRoomHeartbeat(room_id) {
  const isNew = !roomStates.has(room_id);
  roomStates.set(room_id, { lastPing: Date.now(), lost: false });
  if (isNew) {
    createRoomCard(room_id);
    refreshRoomCount();
  } else {
    // Transition back to Connected if it was Lost.
    updateRoomCard(room_id);
  }
}

// Poll every 500 ms: update each room's "last seen" age and Lost state.
setInterval(() => {
  for (const room_id of roomStates.keys()) {
    updateRoomCard(room_id);
  }
}, 500);

function updateRoomCard(room_id) {
  const card  = roomGrid.querySelector(`.room-card[data-room="${room_id}"]`);
  const state = roomStates.get(room_id);
  if (!card || !state) return;

  const elapsed = Date.now() - state.lastPing;
  const lost    = elapsed > ROOM_TIMEOUT_MS;

  if (lost !== state.lost) {
    state.lost = lost;
    card.classList.toggle('connected', !lost);
    card.classList.toggle('lost', lost);
    card.querySelector('.room-status').textContent = lost ? 'Lost' : 'Connected';
    refreshRoomCount();
  }

  card.querySelector('.room-age').textContent = formatAge(elapsed);
}

function createRoomCard(room_id) {
  // Remove placeholder text on first card.
  const placeholder = roomGrid.querySelector('.room-grid-empty');
  if (placeholder) placeholder.remove();

  const card = document.createElement('div');
  card.className   = 'room-card connected';
  card.dataset.room = room_id;
  card.innerHTML = `
    <div class="room-card-top">
      <span class="room-dot"></span>
      <span class="room-name">Room ${room_id}</span>
    </div>
    <div class="room-status">Connected</div>
    <div class="room-age">just now</div>`;

  // Insert in ascending room_id order.
  const existing = [...roomGrid.querySelectorAll('.room-card')];
  const after    = existing.find(c => parseInt(c.dataset.room, 10) > room_id);
  after ? roomGrid.insertBefore(card, after) : roomGrid.appendChild(card);
}

function refreshRoomCount() {
  const total = roomStates.size;
  const lost  = [...roomStates.values()].filter(s => s.lost).length;

  if (total === 0) {
    roomCount.textContent = '';
    roomCount.className   = '';
    return;
  }

  if (lost === 0) {
    roomCount.textContent = `${total} / ${total} online`;
    roomCount.className   = 'count-ok';
  } else {
    roomCount.textContent = `${total - lost} / ${total} online`;
    roomCount.className   = 'count-warn';
  }
}

function formatAge(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 5)  return 'just now';
  if (s < 60) return `${s}s ago`;
  return `${Math.floor(s / 60)}m ago`;
}

// ── Sound ──────────────────────────────────────────────────────────────
let audioCtx     = null;
let soundEnabled = false;
let beepTimer    = null;
let activeCount  = 0;

soundBtn.addEventListener('click', toggleSound);

function toggleSound() {
  if (soundEnabled) {
    soundEnabled = false;
    soundBtn.textContent = '🔇 Sound';
    soundBtn.classList.remove('sound-on');
    stopBeep();
  } else {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.resume();
    soundEnabled = true;
    soundBtn.textContent = '🔔 Sound';
    soundBtn.classList.add('sound-on');
    if (activeCount > 0) startBeep();
  }
}

function playTone(freq = 880, dur = 0.12) {
  if (!audioCtx || !soundEnabled) return;
  const osc  = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.type = 'square';
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
  osc.start(audioCtx.currentTime);
  osc.stop(audioCtx.currentTime + dur);
}

function startBeep() {
  if (beepTimer || !soundEnabled) return;
  playTone(880, 0.12);
  setTimeout(() => playTone(660, 0.12), 180);
  beepTimer = setInterval(() => {
    playTone(880, 0.12);
    setTimeout(() => playTone(660, 0.12), 180);
  }, 1800);
}

function stopBeep() {
  clearInterval(beepTimer);
  beepTimer = null;
}

function onAlertAdded() {
  activeCount++;
  updateAlertCount();
  if (soundEnabled) startBeep();
}

function onAlertRemoved() {
  activeCount = Math.max(0, activeCount - 1);
  updateAlertCount();
  if (activeCount === 0) stopBeep();
}

function updateAlertCount() {
  alertCount.textContent = activeCount;
  alertCount.style.display = activeCount > 0 ? 'inline-block' : 'none';
}

// ── Socket.IO ──────────────────────────────────────────────────────────
const socket = io({ transports: ['websocket', 'polling'] });

socket.on('connect',    () => setServerStatus(true));
socket.on('disconnect', () => setServerStatus(false));

socket.on('room_heartbeat', ({ room_id }) => onRoomHeartbeat(room_id));

socket.on('new_alert', (data) => addAlertCard(data));

socket.on('alert_dismissed', ({ alert_id }) => {
  const card = document.querySelector(`.alert-card[data-alert-id="${alert_id}"]`);
  if (card) removeCard(card.closest('li'));
});

// ── Alert rendering ────────────────────────────────────────────────────

function addAlertCard(data) {
  const { alert_id, alert_type, room_id, timestamp } = data;
  const label = TYPE_LABELS[alert_type] ?? alert_type.replace(/_/g, ' ');

  emptyState.style.display = 'none';

  const li = document.createElement('li');
  li.innerHTML = `
    <div class="alert-card" data-type="${alert_type}" data-alert-id="${alert_id}">
      <div class="card-meta">
        <span class="room-badge">Room ${room_id}</span>
        <span class="type-badge">${alert_type.replace(/_/g, ' ')}</span>
        <span class="id-badge">#${alert_id}</span>
      </div>
      <div class="card-body">
        <div class="card-title">${label}</div>
        <div class="card-time">${timestamp}</div>
      </div>
      <button class="dismiss-btn" type="button">Dismiss</button>
    </div>`;

  li.querySelector('.dismiss-btn').addEventListener('click', () => {
    socket.emit('dismiss_alert', { alert_id });
    removeCard(li);
  });

  alertList.prepend(li);
  onAlertAdded();
}

function removeCard(li) {
  const card = li.querySelector('.alert-card');
  if (!card || card.classList.contains('dismissing')) return;
  card.classList.add('dismissing');
  card.addEventListener('transitionend', () => {
    li.remove();
    onAlertRemoved();
    if (alertList.children.length === 0) emptyState.style.display = '';
  }, { once: true });
}
