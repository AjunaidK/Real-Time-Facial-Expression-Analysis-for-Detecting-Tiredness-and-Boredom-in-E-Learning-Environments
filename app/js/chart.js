// chart.js — minimal dependency-free line chart for the engagement time series.
export class MiniChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.series = { tiredness: [], boredom: [], engagement: [] };  // arrays of {t, v}
    this.maxPoints = 120;
    this._resize();
    window.addEventListener("resize", () => this._resize());
  }
  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = r.width * dpr;
    this.canvas.height = r.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = r.width; this.H = r.height;
  }
  push(t, tiredness, boredom, engagement) {
    if (tiredness != null) this.series.tiredness.push({ t, v: tiredness });
    if (boredom != null) this.series.boredom.push({ t, v: boredom });
    if (engagement != null) this.series.engagement.push({ t, v: engagement });
    for (const k of ["tiredness", "boredom", "engagement"])
      if (this.series[k].length > this.maxPoints) this.series[k].shift();
    this.draw();
  }
  setSeries(points) {
    this.series.tiredness = points.map(p => ({ t: p.t, v: p.tiredness }));
    this.series.boredom = points.map(p => ({ t: p.t, v: p.boredom }));
    this.series.engagement = points.map(p => ({ t: p.t, v: p.engagement }));
    this._resize(); this.draw();
  }
  reset() { this.series = { tiredness: [], boredom: [], engagement: [] }; this.draw(); }

  draw() {
    const { ctx, W, H } = this;
    ctx.clearRect(0, 0, W, H);
    const padL = 26, padB = 16, padT = 8, padR = 6;
    const w = W - padL - padR, h = H - padT - padB;

    // grid + y labels (0,0.5,1)
    ctx.strokeStyle = "#24374F"; ctx.fillStyle = "#64768C";
    ctx.font = "10px Inter, sans-serif"; ctx.lineWidth = 1;
    [0, 0.5, 1].forEach(g => {
      const y = padT + h * (1 - g);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.fillText(g.toFixed(1), 4, y + 3);
    });
    // disengaged threshold 0.7 (faint)
    const yth = padT + h * (1 - 0.7);
    ctx.strokeStyle = "#3A5060"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padL, yth); ctx.lineTo(W - padR, yth); ctx.stroke();
    ctx.setLineDash([]);

    const all = [...this.series.tiredness, ...this.series.boredom, ...this.series.engagement];
    if (!all.length) return;
    const times = all.map(p => p.t);
    const tMin = Math.min(...times), tMax = Math.max(Math.max(...times), tMin + 1);
    const X = t => padL + w * ((t - tMin) / (tMax - tMin || 1));
    const Y = v => padT + h * (1 - v);

    const line = (arr, color, fill) => {
      if (arr.length < 1) return;
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      arr.forEach((p, i) => i === 0 ? ctx.moveTo(X(p.t), Y(p.v)) : ctx.lineTo(X(p.t), Y(p.v)));
      ctx.stroke();
      if (fill) {
        ctx.lineTo(X(arr[arr.length - 1].t), padT + h); ctx.lineTo(X(arr[0].t), padT + h);
        ctx.closePath(); ctx.globalAlpha = 0.08; ctx.fillStyle = color; ctx.fill();
        ctx.globalAlpha = 1;
      }
    };
    line(this.series.engagement, "#5EEAD4", true);
    line(this.series.boredom, "#38BDF8", false);
    line(this.series.tiredness, "#E6A417", false);
  }
}
