// features.js — geometric + temporal feature extraction from MediaPipe landmarks.
// Mirrors the Python reference implementation (src/features/geometry.py &
// temporal.py) so client-side and server-side configurations are numerically
// comparable. See dissertation Ch 3.4 / 4.4.

// ---- MediaPipe Face Mesh landmark index groups ----
export const LEFT_EYE  = [33, 160, 158, 133, 153, 144];
export const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
export const MOUTH = { left: 61, right: 291, top: 13, bottom: 14, top2: 0, bottom2: 17 };
export const LEFT_IRIS = 468, RIGHT_IRIS = 473;

export const FRAME_FEATURE_ORDER  = ["ear", "mar", "pitch", "yaw", "roll", "gaze"];
export const WINDOW_FEATURE_ORDER = [
  "ear_mean", "ear_var", "perclos", "blink_rate",
  "mar_mean", "yawn_freq", "offscreen_prop", "yaw_var", "pitch_var", "noface_prop",
];

// Note: the model input vector uses WINDOW_FEATURE_ORDER above (10 features) so
// trained models stay compatible. The extra focus statistics below are exposed
// via windowFeaturesObject() for the heuristic detector and the UI, but are NOT
// appended to the model vector.

// thresholds (keep in sync with temporal.py)
export const TH = {
  EAR_BLINK: 0.21, EAR_CLOSED: 0.21, BLINK_MIN_FRAMES: 2,
  MAR_YAWN: 0.60, YAWN_MIN_FRAMES: 8, GAZE_OFFSCREEN: 0.35,
};

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const XY = (lm, i) => [lm[i].x, lm[i].y];

export function eyeAspectRatio(lm, idx) {
  const p = idx.map(i => XY(lm, i));
  const vert = dist(p[1], p[5]) + dist(p[2], p[4]);
  const horiz = 2 * dist(p[0], p[3]);
  return horiz > 1e-6 ? vert / horiz : 0;
}

export function mouthAspectRatio(lm) {
  const vert = (dist(XY(lm, MOUTH.top), XY(lm, MOUTH.bottom)) +
                dist(XY(lm, MOUTH.top2), XY(lm, MOUTH.bottom2))) / 2;
  const horiz = dist(XY(lm, MOUTH.left), XY(lm, MOUTH.right));
  return horiz > 1e-6 ? vert / horiz : 0;
}

// lightweight head-pose proxy (no solvePnP in-browser; Ch 4.6 trade-off note)
export function headPoseProxy(lm) {
  const nose = XY(lm, 1), le = XY(lm, 33), re = XY(lm, 263), chin = XY(lm, 152);
  const eyeMidX = (le[0] + re[0]) / 2, eyeSpan = Math.abs(re[0] - le[0]) || 1e-6;
  const yaw = ((nose[0] - eyeMidX) / eyeSpan) * 90;
  const eyeMidY = (le[1] + re[1]) / 2, faceH = Math.abs(chin[1] - eyeMidY) || 1e-6;
  const pitch = ((nose[1] - eyeMidY) / faceH - 0.5) * 90;
  const roll = Math.atan2(re[1] - le[1], re[0] - le[0]) * 180 / Math.PI;
  return { pitch, yaw, roll };
}

export function gazeOffset(lm) {
  if (lm.length <= RIGHT_IRIS) return 0;
  const ratio = (eye, iris) => {
    const lc = XY(lm, eye[0]), rc = XY(lm, eye[3]), ir = XY(lm, iris);
    const span = rc[0] - lc[0];
    return Math.abs(span) < 1e-6 ? 0 : ((ir[0] - lc[0]) / span - 0.5) * 2;
  };
  return (ratio(LEFT_EYE, LEFT_IRIS) + ratio(RIGHT_EYE, RIGHT_IRIS)) / 2;
}

export function perFrameFeatures(lm) {
  const ear = (eyeAspectRatio(lm, LEFT_EYE) + eyeAspectRatio(lm, RIGHT_EYE)) / 2;
  const mar = mouthAspectRatio(lm);
  const { pitch, yaw, roll } = headPoseProxy(lm);
  return { ear, mar, pitch, yaw, roll, gaze: gazeOffset(lm) };
}

// ---- temporal aggregator ----
export class TemporalAggregator {
  constructor(fps = 10, windowSeconds = 5) {
    this.fps = fps;
    this.windowLen = Math.max(1, Math.round(fps * windowSeconds));
    this.reset();
  }
  reset() {
    this.buffer = []; this.blinkEvents = []; this.yawnEvents = [];
    this.closedRun = 0; this.yawnRun = 0;
    this.totalBlinks = 0; this.totalYawns = 0;
    // blink-interval / focus tracking
    this.lastBlinkTime = null;      // ms timestamp of previous blink
    this.blinkIntervals = [];       // seconds between consecutive blinks
    this.lastIntervalS = null;      // most recent gap (seconds)
  }
  _trim(a) { while (a.length > this.windowLen) a.shift(); }

  update(feat, facePresent = true) {
    if (!facePresent) {
      const z = {}; FRAME_FEATURE_ORDER.forEach(k => z[k] = 0); z._noface = 1;
      this.buffer.push(z); this._trim(this.buffer);
      this.blinkEvents.push(0); this._trim(this.blinkEvents);
      this.yawnEvents.push(0); this._trim(this.yawnEvents);
      this.closedRun = 0; this.yawnRun = 0;
      return { blink: false, yawn: false };
    }
    this.buffer.push({ ...feat, _noface: 0 }); this._trim(this.buffer);

    let blink = 0;
    if (feat.ear < TH.EAR_BLINK) this.closedRun++;
    else {
      if (this.closedRun >= TH.BLINK_MIN_FRAMES) {
        blink = 1; this.totalBlinks++;
        // record the gap since the previous blink
        const now = (typeof performance !== "undefined") ? performance.now() : Date.now();
        if (this.lastBlinkTime != null) {
          const gapS = (now - this.lastBlinkTime) / 1000;
          this.lastIntervalS = gapS;
          this.blinkIntervals.push(gapS);
          if (this.blinkIntervals.length > 40) this.blinkIntervals.shift();
        }
        this.lastBlinkTime = now;
      }
      this.closedRun = 0;
    }
    this.blinkEvents.push(blink); this._trim(this.blinkEvents);

    let yawn = 0;
    if (feat.mar > TH.MAR_YAWN) this.yawnRun++;
    else { if (this.yawnRun >= TH.YAWN_MIN_FRAMES) { yawn = 1; this.totalYawns++; } this.yawnRun = 0; }
    this.yawnEvents.push(yawn); this._trim(this.yawnEvents);
    return { blink: !!blink, yawn: !!yawn };
  }

  ready() { return this.buffer.length >= this.windowLen; }
  _mean(a) { return a.reduce((s, v) => s + v, 0) / (a.length || 1); }
  _var(a) { const m = this._mean(a); return this._mean(a.map(v => (v - m) ** 2)); }

  windowFeatures() {
    const ear = this.buffer.map(f => f.ear), mar = this.buffer.map(f => f.mar);
    const yaw = this.buffer.map(f => f.yaw), pitch = this.buffer.map(f => f.pitch);
    const gaze = this.buffer.map(f => f.gaze), noface = this.buffer.map(f => f._noface || 0);
    const secs = this.buffer.length / this.fps || 1;
    const out = {
      ear_mean: this._mean(ear), ear_var: this._var(ear),
      perclos: this._mean(ear.map(v => v < TH.EAR_CLOSED ? 1 : 0)),
      blink_rate: this.blinkEvents.reduce((s, v) => s + v, 0) / secs,
      mar_mean: this._mean(mar),
      yawn_freq: this.yawnEvents.reduce((s, v) => s + v, 0) / secs,
      offscreen_prop: this._mean(gaze.map((g, i) =>
        (Math.abs(g) > TH.GAZE_OFFSCREEN || noface[i] > 0.5) ? 1 : 0)),
      yaw_var: this._var(yaw), pitch_var: this._var(pitch),
      noface_prop: this._mean(noface),
    };
    return WINDOW_FEATURE_ORDER.map(k => out[k]);
  }

  windowFeaturesObject() {
    const v = this.windowFeatures(), o = {};
    WINDOW_FEATURE_ORDER.forEach((k, i) => o[k] = v[i]);
    // --- extra focus / blink-interval statistics (not part of model vector) ---
    // Mean gap between recent blinks, in seconds. Long gaps => sustained focus.
    const gaps = this.blinkIntervals;
    o.mean_blink_gap = gaps.length ? gaps.reduce((s, g) => s + g, 0) / gaps.length : 0;
    // seconds since the last blink right now (rises while eyes stay open & steady)
    const now = (typeof performance !== "undefined") ? performance.now() : Date.now();
    o.since_last_blink = this.lastBlinkTime != null ? (now - this.lastBlinkTime) / 1000 : 0;
    // focus_index in 0..1: rewards long, regular blink gaps, penalises rapid blinking.
    // A relaxed, focused viewer blinks roughly every 4-8s+; rapid blinking (<2s gaps)
    // indicates fatigue/strain. Map mean gap through a soft curve.
    const g = o.mean_blink_gap;
    o.focus_index = g <= 0 ? 0.5 : Math.max(0, Math.min(1, (g - 1.5) / (8 - 1.5)));
    // rapid-blink flag feeds tiredness: high when blinks are close together
    o.rapid_blink = g <= 0 ? 0 : Math.max(0, Math.min(1, (2.5 - g) / 2.5));
    return o;
  }

  windowSequence() {
    const seq = this.buffer.map(f => FRAME_FEATURE_ORDER.map(k => f[k] || 0));
    while (seq.length < this.windowLen) seq.unshift(FRAME_FEATURE_ORDER.map(() => 0));
    return seq;
  }
}
