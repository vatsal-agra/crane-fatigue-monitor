/* ==========================================================================
   Live supervisor dashboard.

   Update strategy: server-sent events carry live telemetry and alerts so a
   critical state reaches the screen in well under a second, while a slower
   poll of /api/live reconciles the device roster and catches anything missed
   while the tab was backgrounded. If SSE fails (an old proxy, a corporate
   filter) the poll alone still drives the whole page, just at 2 s instead.
   ========================================================================== */

import { LineChart, COLORS } from './charts.js';

const root = document.documentElement;
const LOW = parseFloat(root.dataset.low) || 30;
const HIGH = parseFloat(root.dataset.high) || 60;

const $ = (id) => document.getElementById(id);

const STATE_COLOR = {
  Normal: COLORS.green,
  Warning: COLORS.amber,
  Critical: COLORS.red,
  NoOperator: COLORS.magenta,
  Offline: COLORS.dim
};

const state = {
  devices: [],
  selected: null,
  latest: {},          // device_id -> most recent telemetry record
  seenEventIds: new Set(),
  rangeMinutes: 15,
  sseOk: false
};

/* ---------------------------------------------------------------- helpers */

const fmt = (v, digits = 1, dash = '—') =>
  (v === null || v === undefined || !isFinite(v)) ? dash : Number(v).toFixed(digits);

const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], {
  hour: '2-digit', minute: '2-digit', second: '2-digit'
});

function scoreColor(score) {
  if (score >= HIGH) return COLORS.red;
  if (score >= LOW) return COLORS.amber;
  return COLORS.green;
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + ' -> ' + res.status);
  return res.json();
}

/* ----------------------------------------------------------------- charts */

const scoreChart = new LineChart($('chart-score'), {
  height: 210,
  yMin: 0, yMax: 100,
  bands: [
    { from: 0, to: LOW, color: 'rgba(63,185,80,.07)' },
    { from: LOW, to: HIGH, color: 'rgba(210,153,34,.09)' },
    { from: HIGH, to: 100, color: 'rgba(248,81,73,.10)' }
  ],
  yFormat: (v) => v.toFixed(0)
});

const earChart = new LineChart($('chart-ear'), { height: 180, yMin: 0 });
const vitalsChart = new LineChart($('chart-vitals'), { height: 180 });

/* ------------------------------------------------------------ device list */

function renderDevices() {
  const host = $('devices');
  $('device-count').textContent = state.devices.length
    ? state.devices.length + ' unit' + (state.devices.length === 1 ? '' : 's')
    : '';

  if (!state.devices.length) {
    host.innerHTML = '<div class="empty">No cabin units reporting yet. '
      + 'Start an edge node with <code>python -m edge.node</code>.</div>';
    return;
  }

  host.innerHTML = state.devices.map(d => {
    const st = d.last_state || 'Offline';
    const live = state.latest[d.device_id];
    const score = d.online ? (live ? live.score : d.last_score) : null;
    return `
      <div class="device-card s-${st} ${d.device_id === state.selected ? 'selected' : ''}"
           data-device="${d.device_id}">
        <div class="row">
          <span class="name">${d.crane_id || d.device_id}</span>
          <span class="score" style="color:${STATE_COLOR[st] || COLORS.dim}">${fmt(score, 0, '--')}</span>
        </div>
        <div class="row">
          <span class="op">${d.operator_name || '—'} · ${d.operator_id || '—'}</span>
        </div>
        <div style="margin-top:8px">
          <span class="badge s-${st} ${st === 'Critical' ? 'pulse' : ''}">${st}</span>
        </div>
        <div class="meta">${d.online
          ? 'updated ' + d.seconds_since_seen.toFixed(0) + ' s ago'
          : 'no telemetry for ' + d.seconds_since_seen.toFixed(0) + ' s'}</div>
      </div>`;
  }).join('');

  host.querySelectorAll('.device-card').forEach(card => {
    card.addEventListener('click', () => selectDevice(card.dataset.device));
  });
}

function selectDevice(deviceId) {
  state.selected = deviceId;
  renderDevices();
  renderDetail();
  loadHistory();
}

/* ---------------------------------------------------------- detail panel */

function renderDetail() {
  const panel = $('detail-panel');
  const record = state.latest[state.selected];
  if (!state.selected || !record) { panel.hidden = true; return; }
  panel.hidden = false;

  const device = state.devices.find(d => d.device_id === state.selected);
  const offline = device && !device.online;
  const st = offline ? 'Offline' : (record.state || 'Normal');
  const score = offline ? 0 : (record.score || 0);

  $('d-operator').textContent =
    (record.operator_name || '—') + '  ·  ' + (record.operator_id || '—');
  $('d-device').textContent =
    (record.crane_id || '') + '  ·  ' + state.selected + '  ·  ' + (record.site || '');
  $('detail-updated').textContent = record.timestamp
    ? (offline ? 'last reported ' + clock(record.timestamp) : 'updated ' + clock(record.timestamp))
    : '';

  const badge = $('d-badge');
  badge.className = 'badge s-' + st + (st === 'Critical' ? ' pulse' : '');
  badge.textContent = st === 'NoOperator' ? 'NO OPERATOR' : st.toUpperCase();

  $('d-score').textContent = offline ? '--' : fmt(score, 0);
  $('d-score').style.color = offline ? COLORS.dim : scoreColor(score);

  const gauge = $('d-gauge');
  gauge.style.width = Math.max(0, Math.min(100, score)) + '%';
  gauge.style.background = offline ? COLORS.dim : scoreColor(score);

  $('d-reasons').innerHTML = offline
    ? '<span style="color:var(--dim)">No telemetry from this cabin unit — '
      + 'readings below are the last received, not current.</span>'
    : ((record.reasons || []).map(r => `<span>${r}</span>`).join('')
       || '<span>All indicators nominal</span>');

  const tiles = [
    ['EAR', fmt(record.ear, 3), 'eye openness'],
    ['MAR', fmt(record.mar, 3), 'mouth opening'],
    ['PERCLOS', fmt((record.perclos || 0) * 100, 0), '%'],
    ['Heart rate', fmt(record.heart_rate, 0), 'bpm'],
    ['HRV RMSSD', fmt(record.hrv_rmssd, 0), 'ms'],
    ['Head tilt', fmt(record.tilt, 1), 'deg'],
    ['Operator', record.face_visible ? 'in frame' : 'NOT VISIBLE', '']
  ];
  $('d-tiles').innerHTML = tiles.map(([k, v, unit]) => `
    <div class="tile">
      <div class="k">${k}</div>
      <div class="v">${v}${unit ? ` <small>${unit}</small>` : ''}</div>
    </div>`).join('');

  const simulated = record.sensor_source === 'simulated' || record.vision_backend === 'synthetic';
  $('d-provenance').innerHTML =
    `Vision backend: <b>${record.vision_backend || '—'}</b> · `
    + `Sensor hub: <b>${record.sensor_source || '—'}</b> · `
    + `Calibration: <b>${record.calibrated ? 'operator baseline' : 'fallback thresholds'}</b>`
    + (simulated ? ' — <span style="color:var(--blue)">simulated input, not a live measurement</span>' : '');

  renderIndicators(record);
}

const INDICATORS = [
  ['drowsiness', 'Drowsiness / microsleep', COLORS.green],
  ['yawn', 'Yawning', COLORS.blue],
  ['posture', 'Nod-off posture', COLORS.amber],
  ['vitals', 'Heart-rate anomaly', COLORS.red]
];

function renderIndicators(record) {
  const counters = record.counters || {};
  const subs = record.sub_scores || {};
  $('d-indicators').innerHTML = INDICATORS.map(([key, label, color]) => {
    const sub = subs[key] || 0;
    return `
      <div class="ind">
        <div class="lbl"><span>${label}</span>
          <span class="n">${counters[key] ?? 0} events · ${(sub * 100).toFixed(0)}%</span></div>
        <div class="track"><div class="fill"
             style="width:${Math.min(100, sub * 100)}%;background:${color}"></div></div>
      </div>`;
  }).join('');
}

/* -------------------------------------------------------------- history */

async function loadHistory() {
  if (!state.selected) return;
  try {
    const { points } = await getJSON(
      `/api/timeseries?device_id=${encodeURIComponent(state.selected)}&minutes=${state.rangeMinutes}`);

    scoreChart.setData([{
      key: 'score', color: COLORS.blue, width: 1.8,
      fill: 'rgba(88,166,255,.10)',
      points: points.map(p => [p.ts, p.score])
    }]);

    earChart.setData([
      { key: 'ear', color: COLORS.green, points: points.map(p => [p.ts, p.ear]) },
      { key: 'mar', color: COLORS.blue, points: points.map(p => [p.ts, p.mar]) }
    ]);

    vitalsChart.setData([
      { key: 'hr', color: COLORS.red, points: points.map(p => [p.ts, p.heart_rate]) },
      { key: 'perclos', color: COLORS.magenta,
        points: points.map(p => [p.ts, p.perclos === null ? null : p.perclos * 100]) }
    ]);
  } catch (err) {
    console.warn('history load failed', err);
  }
}

/* ------------------------------------------------------------ alert feed */

function feedItem(ev) {
  const acked = ev.acknowledged ? ' acked' : '';
  const ackBtn = ev.acknowledged
    ? `<span class="src">acknowledged by ${ev.acknowledged_by || 'supervisor'}</span>`
    : `<button class="btn small" data-ack="${ev.id}">Acknowledge</button>`;
  return `
    <div class="feed-item sev-${ev.severity}${acked}" data-event="${ev.id}">
      <span class="when">${clock(ev.ts)}</span>
      <div class="msg">
        <div><span class="kind">${ev.kind}</span> ${ev.message}</div>
        <div class="src">${ev.device_id} · operator ${ev.operator_id || '—'}</div>
        <div style="margin-top:6px">${ev.severity === 'info' ? '' : ackBtn}</div>
      </div>
    </div>`;
}

async function loadFeed() {
  try {
    const { events } = await getJSON('/api/events?limit=60&severity=escalated');
    const host = $('feed');
    $('alert-count').textContent = events.length ? events.length + ' recent' : '';
    host.innerHTML = events.length
      ? events.map(feedItem).join('')
      : '<div class="empty">No alerts yet.</div>';
    events.forEach(e => state.seenEventIds.add(e.id));

    host.querySelectorAll('[data-ack]').forEach(btn => {
      btn.addEventListener('click', async () => {
        btn.disabled = true;
        await fetch(`/api/events/${btn.dataset.ack}/acknowledge`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ by: 'supervisor' })
        });
        loadFeed();
      });
    });
  } catch (err) {
    console.warn('feed load failed', err);
  }
}

async function loadCommandLog() {
  try {
    const { commands } = await getJSON('/api/command_history');
    $('cmdlog').innerHTML = commands.length ? commands.map(c => `
      <div class="feed-item">
        <span class="when">${clock(c.issued_at)}</span>
        <div class="msg">
          <div><span class="kind">${c.action}</span> ${c.device_id}</div>
          <div class="src">by ${c.issued_by}${c.delivered ? ' · delivered' : ' · pending'}</div>
        </div>
      </div>`).join('') : '<div class="empty">No commands issued.</div>';
  } catch (err) { /* non-critical */ }
}

/* ------------------------------------------------------------- commands */

document.querySelectorAll('[data-cmd]').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!state.selected) {
      $('cmd-result').textContent = 'Select a cabin unit first.';
      return;
    }
    const action = btn.dataset.cmd;
    btn.disabled = true;
    try {
      const res = await fetch('/api/commands', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device_id: state.selected, action, by: 'supervisor' })
      });
      const data = await res.json();
      $('cmd-result').textContent = data.ok
        ? `Sent "${action}" to ${state.selected} — the cabin unit collects it within 2 s.`
        : `Failed: ${data.error}`;
      loadCommandLog();
    } finally {
      setTimeout(() => { btn.disabled = false; }, 600);
    }
  });
});

$('range-select').addEventListener('change', (e) => {
  state.rangeMinutes = parseFloat(e.target.value);
  loadHistory();
});

/* --------------------------------------------------------------- refresh */

async function refreshLive() {
  try {
    const data = await getJSON('/api/live');
    state.devices = data.devices;
    setConn(true, state.sseOk ? 'live (SSE)' : 'live (polling)');

    // Seed the live cache from the roster. Without this the operator panel
    // stays blank until the next SSE push - which never arrives at all for a
    // unit that has gone offline, so its last known readings would be
    // unreachable exactly when a supervisor wants to look at them.
    for (const d of data.devices) {
      if (!d.latest) continue;
      const cached = state.latest[d.device_id];
      if (!cached || (d.latest.timestamp || 0) >= (cached.timestamp || 0)) {
        state.latest[d.device_id] = d.latest;
      }
    }

    if (!state.selected && state.devices.length) {
      // Open on whatever needs attention most, not just the first unit.
      const worst = [...state.devices].sort(
        (a, b) => (b.last_score || 0) - (a.last_score || 0))[0];
      state.selected = worst.device_id;
      loadHistory();
    }
    renderDevices();
    renderDetail();
  } catch (err) {
    setConn(false, 'server unreachable');
  }
}

function setConn(ok, text) {
  $('conn-dot').className = 'dot ' + (ok ? 'live' : 'down');
  $('conn-text').textContent = text;
}

/* ------------------------------------------------------------------ SSE */

function connectStream() {
  let source;
  try {
    source = new EventSource('/api/stream');
  } catch (err) {
    return;
  }

  source.onopen = () => { state.sseOk = true; setConn(true, 'live (SSE)'); };

  source.onmessage = (message) => {
    let payload;
    try { payload = JSON.parse(message.data); } catch (err) { return; }

    if (payload.type === 'telemetry') {
      const record = payload.record;
      if (!record || !record.device_id) return;
      state.latest[record.device_id] = record;

      // Keep the roster fresh between /api/live polls.
      const device = state.devices.find(d => d.device_id === record.device_id);
      if (device) {
        device.last_state = record.state;
        device.last_score = record.score;
        device.last_seen = record.timestamp;
        device.online = true;
        device.seconds_since_seen = 0;
      }
      if (!state.selected) state.selected = record.device_id;
      renderDevices();
      if (record.device_id === state.selected) renderDetail();

    } else if (payload.type === 'event' || payload.type === 'alert') {
      loadFeed();
    } else if (payload.type === 'command') {
      loadCommandLog();
    } else if (payload.type === 'acknowledged') {
      loadFeed();
    }
  };

  source.onerror = () => {
    state.sseOk = false;
    setConn(false, 'stream lost — falling back to polling');
  };
}

/* ------------------------------------------------------------------ boot */

setInterval(() => {
  $('clock').textContent = new Date().toLocaleTimeString();
}, 1000);

refreshLive();
loadFeed();
loadCommandLog();
connectStream();

setInterval(refreshLive, 2000);
setInterval(loadHistory, 5000);
setInterval(loadFeed, 10000);
