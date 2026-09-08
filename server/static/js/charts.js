/* ==========================================================================
   Minimal canvas charting.

   Written by hand rather than pulled from a CDN for one practical reason: a
   construction site office often has no usable internet, and a supervisor
   dashboard that renders blank charts because cdn.jsdelivr.net is unreachable
   is worse than useless. Everything here is a few hundred lines of 2D canvas.

   Handles HiDPI scaling, so the charts stay sharp on a laptop panel.
   ========================================================================== */

const CSS = getComputedStyle(document.documentElement);
const C = (name, fallback) => (CSS.getPropertyValue(name) || fallback).trim();

export const COLORS = {
  green: C('--green', '#3fb950'),
  amber: C('--amber', '#d29922'),
  red: C('--red', '#f85149'),
  blue: C('--blue', '#58a6ff'),
  magenta: C('--magenta', '#bc8cff'),
  muted: C('--muted', '#8b949e'),
  dim: C('--dim', '#6e7681'),
  border: C('--border', '#262d38'),
  panel: C('--panel', '#161b22')
};

const MONO = '11px ui-monospace, Consolas, monospace';

function fitCanvas(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
  canvas.style.height = cssHeight + 'px';
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(cssHeight * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height: cssHeight };
}

function niceTicks(lo, hi, count) {
  if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

const clockLabel = (ts) => new Date(ts * 1000).toLocaleTimeString([], {
  hour: '2-digit', minute: '2-digit'
});

/* --------------------------------------------------------------------------
   Line chart with optional threshold bands.
   series: [{ key, label, color, points: [[x, y], ...], axis: 'left'|'right' }]
   -------------------------------------------------------------------------- */

export class LineChart {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = Object.assign({
      height: 200,
      padding: { top: 12, right: 46, bottom: 24, left: 44 },
      yMin: null, yMax: null,
      bands: [],           // [{ from, to, color }] shaded horizontal zones
      xFormat: clockLabel,
      yFormat: (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)),
      showPoints: false
    }, opts);
    this.series = [];
    this._onResize = () => this.draw();
    window.addEventListener('resize', this._onResize);
  }

  setData(series) { this.series = series || []; this.draw(); }

  destroy() { window.removeEventListener('resize', this._onResize); }

  draw() {
    const { ctx, width, height } = fitCanvas(this.canvas, this.opts.height);
    const p = this.opts.padding;
    const plotW = Math.max(10, width - p.left - p.right);
    const plotH = Math.max(10, height - p.top - p.bottom);

    ctx.clearRect(0, 0, width, height);

    const all = this.series.flatMap(s => s.points || []);
    if (all.length < 2) {
      ctx.fillStyle = COLORS.dim;
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('waiting for data…', width / 2, height / 2);
      return;
    }

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const [x, y] of all) {
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y === null || !isFinite(y)) continue;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    if (this.opts.yMin !== null) yMin = this.opts.yMin;
    if (this.opts.yMax !== null) yMax = this.opts.yMax;
    if (!isFinite(yMin) || !isFinite(yMax)) { yMin = 0; yMax = 1; }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    else { const pad = (yMax - yMin) * 0.08; yMin -= pad; yMax += pad; }
    if (xMin === xMax) xMax = xMin + 1;

    const X = (v) => p.left + (v - xMin) / (xMax - xMin) * plotW;
    const Y = (v) => p.top + plotH - (v - yMin) / (yMax - yMin) * plotH;

    // shaded threshold bands
    for (const band of this.opts.bands) {
      const y1 = Y(Math.min(band.to, yMax));
      const y2 = Y(Math.max(band.from, yMin));
      if (y2 <= y1) continue;
      ctx.fillStyle = band.color;
      ctx.fillRect(p.left, y1, plotW, y2 - y1);
    }

    // grid + y axis
    ctx.strokeStyle = COLORS.border;
    ctx.fillStyle = COLORS.dim;
    ctx.font = MONO;
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const tick of niceTicks(yMin, yMax, 4)) {
      const y = Math.round(Y(tick)) + 0.5;
      if (y < p.top - 1 || y > p.top + plotH + 1) continue;
      ctx.beginPath();
      ctx.moveTo(p.left, y);
      ctx.lineTo(p.left + plotW, y);
      ctx.stroke();
      ctx.fillText(this.opts.yFormat(tick), p.left - 7, y);
    }

    // x axis
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTickCount = Math.max(2, Math.min(6, Math.floor(plotW / 90)));
    for (let i = 0; i <= xTickCount; i++) {
      const v = xMin + (xMax - xMin) * i / xTickCount;
      ctx.fillStyle = COLORS.dim;
      ctx.fillText(this.opts.xFormat(v), X(v), p.top + plotH + 7);
    }

    // series
    for (const s of this.series) {
      const pts = s.points || [];
      if (pts.length < 2) continue;
      ctx.strokeStyle = s.color || COLORS.blue;
      ctx.lineWidth = s.width || 1.6;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let drawing = false;
      for (const [x, y] of pts) {
        if (y === null || !isFinite(y)) { drawing = false; continue; }
        const px = X(x), py = Y(y);
        if (!drawing) { ctx.moveTo(px, py); drawing = true; }
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      if (s.fill) {
        const last = pts[pts.length - 1];
        ctx.lineTo(X(last[0]), p.top + plotH);
        ctx.lineTo(X(pts[0][0]), p.top + plotH);
        ctx.closePath();
        ctx.fillStyle = s.fill;
        ctx.fill();
      }

      // current-value marker
      const valid = pts.filter(pt => pt[1] !== null && isFinite(pt[1]));
      if (valid.length && s.marker !== false) {
        const [lx, ly] = valid[valid.length - 1];
        ctx.fillStyle = s.color || COLORS.blue;
        ctx.beginPath();
        ctx.arc(X(lx), Y(ly), 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // frame
    ctx.strokeStyle = COLORS.border;
    ctx.strokeRect(p.left + 0.5, p.top + 0.5, plotW, plotH);
  }
}

/* --------------------------------------------------------------------------
   Vertical bar chart (used for fatigue-by-hour in the shift report).
   -------------------------------------------------------------------------- */

export class BarChart {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = Object.assign({
      height: 210,
      padding: { top: 14, right: 14, bottom: 30, left: 44 },
      yMax: null,
      colorFor: () => COLORS.blue,
      valueFormat: (v) => v.toFixed(0),
      thresholds: []      // [{ value, color, label }] horizontal marker lines
    }, opts);
    this.data = [];
    this._onResize = () => this.draw();
    window.addEventListener('resize', this._onResize);
  }

  /* data: [{ label, value, sublabel? }] */
  setData(data) { this.data = data || []; this.draw(); }

  destroy() { window.removeEventListener('resize', this._onResize); }

  draw() {
    const { ctx, width, height } = fitCanvas(this.canvas, this.opts.height);
    const p = this.opts.padding;
    const plotW = Math.max(10, width - p.left - p.right);
    const plotH = Math.max(10, height - p.top - p.bottom);

    ctx.clearRect(0, 0, width, height);
    if (!this.data.length) {
      ctx.fillStyle = COLORS.dim;
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('no data in this window', width / 2, height / 2);
      return;
    }

    const yMax = this.opts.yMax !== null
      ? this.opts.yMax
      : Math.max(1, ...this.data.map(d => d.value)) * 1.15;
    const Y = (v) => p.top + plotH - (v / yMax) * plotH;

    ctx.strokeStyle = COLORS.border;
    ctx.fillStyle = COLORS.dim;
    ctx.font = MONO;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const tick of niceTicks(0, yMax, 4)) {
      const y = Math.round(Y(tick)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(p.left, y);
      ctx.lineTo(p.left + plotW, y);
      ctx.stroke();
      ctx.fillText(this.opts.valueFormat(tick), p.left - 7, y);
    }

    for (const t of this.opts.thresholds) {
      const y = Math.round(Y(t.value)) + 0.5;
      ctx.save();
      ctx.strokeStyle = t.color;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(p.left, y);
      ctx.lineTo(p.left + plotW, y);
      ctx.stroke();
      ctx.restore();
    }

    const slot = plotW / this.data.length;
    const barW = Math.max(4, Math.min(46, slot * 0.62));
    ctx.textAlign = 'center';

    this.data.forEach((d, i) => {
      const cx = p.left + slot * (i + 0.5);
      const h = Math.max(1, plotH - (Y(d.value) - p.top));
      ctx.fillStyle = this.opts.colorFor(d.value, d);
      ctx.fillRect(cx - barW / 2, Y(d.value), barW, h);

      ctx.fillStyle = COLORS.dim;
      ctx.textBaseline = 'top';
      ctx.fillText(d.label, cx, p.top + plotH + 8);

      if (slot > 26) {
        ctx.fillStyle = COLORS.muted;
        ctx.textBaseline = 'bottom';
        ctx.fillText(this.opts.valueFormat(d.value), cx, Y(d.value) - 3);
      }
    });

    ctx.strokeStyle = COLORS.border;
    ctx.strokeRect(p.left + 0.5, p.top + 0.5, plotW, plotH);
  }
}

/* --------------------------------------------------------------------------
   Horizontal stacked bar - the state distribution strip.
   segments: [{ label, value, color }]
   -------------------------------------------------------------------------- */

export class StackedBar {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = Object.assign({ height: 54, barHeight: 26 }, opts);
    this.segments = [];
    this._onResize = () => this.draw();
    window.addEventListener('resize', this._onResize);
  }

  setData(segments) { this.segments = (segments || []).filter(s => s.value > 0); this.draw(); }

  destroy() { window.removeEventListener('resize', this._onResize); }

  draw() {
    const { ctx, width } = fitCanvas(this.canvas, this.opts.height);
    ctx.clearRect(0, 0, width, this.opts.height);

    const total = this.segments.reduce((a, s) => a + s.value, 0);
    if (!total) {
      ctx.fillStyle = COLORS.dim;
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('no data', 0, 20);
      return;
    }

    let x = 0;
    const h = this.opts.barHeight;
    ctx.font = MONO;
    ctx.textBaseline = 'middle';

    for (const seg of this.segments) {
      const w = (seg.value / total) * width;
      ctx.fillStyle = seg.color;
      ctx.fillRect(x, 0, Math.max(1, w - 1), h);
      if (w > 46) {
        ctx.fillStyle = '#0d1117';
        ctx.textAlign = 'center';
        ctx.fillText(seg.value.toFixed(0) + '%', x + w / 2, h / 2);
      }
      x += w;
    }

    x = 0;
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    for (const seg of this.segments) {
      const w = (seg.value / total) * width;
      if (w > 58) {
        ctx.fillStyle = COLORS.muted;
        ctx.fillText(seg.label, x + 1, h + 8);
      }
      x += w;
    }
  }
}
