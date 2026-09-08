/* ==========================================================================
   Post-shift report - algorithm step 16.

   The point of this page is not to look back at pretty numbers: it is to
   answer the two questions a site supervisor actually has at the end of a
   shift. Which hours are producing fatigue? And which operator needs a
   different rota?
   ========================================================================== */

import { BarChart, StackedBar, COLORS } from './charts.js';

const root = document.documentElement;
const LOW = parseFloat(root.dataset.low) || 30;
const HIGH = parseFloat(root.dataset.high) || 60;

const $ = (id) => document.getElementById(id);

const STATE_COLOR = {
  Normal: COLORS.green,
  Warning: COLORS.amber,
  Critical: COLORS.red,
  NoOperator: COLORS.magenta
};

const stateBar = new StackedBar($('chart-states'), { height: 56 });

const hourlyChart = new BarChart($('chart-hourly'), {
  height: 230,
  yMax: 100,
  colorFor: (v) => (v >= HIGH ? COLORS.red : v >= LOW ? COLORS.amber : COLORS.green),
  valueFormat: (v) => v.toFixed(0),
  thresholds: [
    { value: LOW, color: COLORS.amber },
    { value: HIGH, color: COLORS.red }
  ]
});

const fmtClock = (ts) => new Date(ts * 1000).toLocaleString([], {
  month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit'
});

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + ' -> ' + res.status);
  return res.json();
}

function tile(label, value, unit, color) {
  return `
    <div class="panel"><div class="body">
      <div class="k" style="font-size:10.5px;letter-spacing:.05em;
           text-transform:uppercase;color:var(--dim)">${label}</div>
      <div style="font-family:var(--mono);font-size:30px;font-weight:650;margin-top:4px;
           color:${color || 'var(--text)'}">${value}<small
           style="font-size:13px;color:var(--dim);font-weight:400"> ${unit || ''}</small></div>
    </div></div>`;
}

function renderSummary(report) {
  const hours = report.first_ts && report.last_ts
    ? ((report.last_ts - report.first_ts) / 3600) : 0;
  const peakScore = report.peak_score || 0;

  $('summary-tiles').innerHTML = [
    tile('Mean fatigue score', report.mean_score.toFixed(1), '/100',
         report.mean_score >= HIGH ? COLORS.red
           : report.mean_score >= LOW ? COLORS.amber : COLORS.green),
    tile('Peak fatigue score', peakScore.toFixed(1), '/100',
         peakScore >= HIGH ? COLORS.red : peakScore >= LOW ? COLORS.amber : COLORS.green),
    tile('Escalated alerts', report.total_alerts, ''),
    tile('Monitored time', hours.toFixed(1), 'h')
  ].join('');
}

// Severity order, so the strip always reads left-to-right from safe to unsafe
// instead of following whatever order the GROUP BY happened to return.
const STATE_ORDER = ['Normal', 'Warning', 'Critical', 'NoOperator'];

function renderStates(report) {
  const dist = report.state_distribution || {};
  const names = Object.keys(dist).sort((a, b) => {
    const ia = STATE_ORDER.indexOf(a), ib = STATE_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  stateBar.setData(names.map(name => ({
    label: name, value: dist[name], color: STATE_COLOR[name] || COLORS.dim
  })));
}

function renderHourly(report) {
  const rows = report.hourly || [];
  hourlyChart.setData(rows.map(r => ({
    label: String(r.hour).padStart(2, '0') + ':00',
    value: r.mean_score
  })));

  if (!rows.length) {
    $('hourly-insight').textContent = 'No telemetry in this window.';
    return;
  }
  const worst = rows.reduce((a, b) => (b.mean_score > a.mean_score ? b : a));
  const flagged = rows.filter(r => r.mean_score >= LOW);
  $('hourly-insight').innerHTML = flagged.length
    ? `Fatigue-prone hours: <b>${flagged.map(r =>
        String(r.hour).padStart(2, '0') + ':00').join(', ')}</b>. `
      + `Worst hour is <b>${String(worst.hour).padStart(2, '0')}:00</b> `
      + `at a mean score of <b>${worst.mean_score.toFixed(1)}</b> — `
      + `schedule a break or an operator rotation before this window.`
    : 'No hour in this window exceeded the warning threshold.';
}

function renderTables(report) {
  const events = report.event_counts || [];
  $('events-table').querySelector('tbody').innerHTML = events.length
    ? events.map(e => `
        <tr>
          <td><span class="kind">${e.kind}</span></td>
          <td><span class="badge ${e.severity === 'critical' ? 's-Critical'
              : e.severity === 'warning' ? 's-Warning' : 'info'}">${e.severity}</span></td>
          <td class="num">${e.n}</td>
        </tr>`).join('')
    : '<tr><td colspan="3" class="empty">No events recorded.</td></tr>';

  const ops = report.operators || [];
  $('operators-table').querySelector('tbody').innerHTML = ops.length
    ? ops.map(o => `
        <tr>
          <td>${o.operator_id}</td>
          <td class="num">${o.samples}</td>
          <td class="num" style="color:${o.mean_score >= HIGH ? COLORS.red
              : o.mean_score >= LOW ? COLORS.amber : COLORS.green}">${o.mean_score.toFixed(1)}</td>
          <td class="num">${o.peak_score.toFixed(1)}</td>
        </tr>`).join('')
    : '<tr><td colspan="4" class="empty">No operator data.</td></tr>';
}

async function renderLog(deviceId, hours) {
  const since = Date.now() / 1000 - hours * 3600;
  const params = new URLSearchParams({ limit: '200', severity: 'escalated', since: String(since) });
  if (deviceId) params.set('device_id', deviceId);
  const { events } = await getJSON('/api/events?' + params.toString());

  $('log-table').querySelector('tbody').innerHTML = events.length
    ? events.map(e => `
        <tr>
          <td style="font-family:var(--mono);white-space:nowrap">${fmtClock(e.ts)}</td>
          <td>${e.device_id}</td>
          <td>${e.operator_id || '—'}</td>
          <td><span class="kind">${e.kind}</span></td>
          <td style="color:var(--muted)">${e.message}</td>
          <td class="num">${e.score === null ? '—' : e.score.toFixed(0)}</td>
        </tr>`).join('')
    : '<tr><td colspan="6" class="empty">No alerts in this window.</td></tr>';
}

async function load() {
  const deviceId = $('device-select').value;
  const hours = parseFloat($('hours-select').value);

  const params = new URLSearchParams({ hours: String(hours) });
  if (deviceId) params.set('device_id', deviceId);

  const report = await getJSON('/api/report?' + params.toString());
  renderSummary(report);
  renderStates(report);
  renderHourly(report);
  renderTables(report);
  await renderLog(deviceId, hours);
}

async function loadDevices() {
  const { devices } = await getJSON('/api/live');
  const select = $('device-select');
  const current = select.value;
  select.innerHTML = '<option value="">All cabin units</option>'
    + devices.map(d => `<option value="${d.device_id}">${d.crane_id || d.device_id}</option>`).join('');
  select.value = current;
}

$('refresh').addEventListener('click', load);
$('device-select').addEventListener('change', load);
$('hours-select').addEventListener('change', load);
$('print').addEventListener('click', () => window.print());

loadDevices().then(load);
setInterval(load, 30000);
