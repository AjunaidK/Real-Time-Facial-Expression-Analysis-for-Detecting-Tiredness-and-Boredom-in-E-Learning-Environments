// detector.js — produces tiredness & boredom scores from window features.
//
// Two modes, selected automatically:
//   (1) TRAINED: loads TensorFlow.js models from ./models/ if present.
//   (2) HEURISTIC: a transparent, calibrated rule-based detector grounded in the
//       drowsiness/boredom literature (PERCLOS, blink rate, yawns, gaze, head
//       pose). This makes the artefact fully functional for a live demo BEFORE
//       the deep models are trained, and serves as an interpretable baseline the
//       trained models are compared against (dissertation Ch 4.5 / Ch 5).
//
// Both modes expose the same async interface: predict(agg) -> {tiredness, boredom, source}

import { TH } from "./features.js";

const clamp01 = x => Math.max(0, Math.min(1, x));

// ------- HEURISTIC baseline (interpretable, literature-grounded) -------
// Per-person calibration: a short baseline captures the user's "alert & engaged"
// resting values so scores are relative to the individual (Ch 2.6 personalisation).
export class HeuristicDetector {
  constructor() {
    this.baseline = null;       // {ear, blinkRate, perclos}
    this.calibrating = false;
    this._calibSamples = [];
  }

  startCalibration() { this.calibrating = true; this._calibSamples = []; }
  addCalibrationSample(wf) { if (this.calibrating) this._calibSamples.push(wf); }
  finishCalibration() {
    this.calibrating = false;
    if (!this._calibSamples.length) return;
    const mean = k => this._calibSamples.reduce((s, w) => s + w[k], 0) / this._calibSamples.length;
    this.baseline = { ear: mean("ear_mean"), blink: mean("blink_rate"), perclos: mean("perclos") };
  }

  predict(wf) {
    // wf = window features object
    // --- TIREDNESS: PERCLOS dominant, then prolonged closure & yawns ---
    // PERCLOS is the most validated single drowsiness indicator (Ch 2.5).
    const perclos = wf.perclos;                       // 0..1 fraction eyes closed
    const yawn = clamp01(wf.yawn_freq / 0.15);        // ~1 yawn / window saturates
    const lowEar = this.baseline
      ? clamp01((this.baseline.ear - wf.ear_mean) / (this.baseline.ear * 0.35 + 1e-6))
      : clamp01((0.28 - wf.ear_mean) / 0.10);
    // Rapid, closely-spaced blinking is a fatigue/eye-strain cue; long gaps are not.
    // wf.rapid_blink is ~1 when blinks come <2.5s apart, ~0 when well spaced.
    const rapidBlink = wf.rapid_blink ?? 0;

    let tiredness = clamp01(
      0.50 * clamp01(perclos / 0.30) +   // eyes-closed fraction: dominant
      0.22 * yawn +                       // yawning
      0.14 * lowEar +                     // eyes droopier than personal normal
      0.14 * rapidBlink                   // rapid/closely-spaced blinking
    );

    // --- BOREDOM: gaze off-screen, head turned away, flat/low expressivity ---
    // Boredom is subtle & person-variable (Ch 2.6): built from attention proxies.
    const offscreen = wf.offscreen_prop;               // 0..1 gaze/no-face away
    const headAway = clamp01(Math.sqrt(wf.yaw_var) / 15);   // fidget / turn away
    const noface = wf.noface_prop;                     // left the screen
    const flatAffect = clamp01((0.02 - wf.ear_var) / 0.02) * 0.4; // very still eyes
    let boredom = clamp01(
      0.42 * offscreen +
      0.24 * headAway +
      0.18 * noface +
      0.10 * flatAffect +
      0.06 * (1 - (wf.focus_index ?? 0.5))   // low focus (rapid blinks) nudges boredom up
    );
    // Sustained focus (long, steady blink gaps + eyes on screen) relieves boredom:
    // a well-focused viewer shouldn't read as bored just for being still.
    const focus = wf.focus_index ?? 0.5;
    if (focus > 0.6 && offscreen < 0.2 && noface < 0.1) {
      boredom = clamp01(boredom * (1 - 0.35 * (focus - 0.6) / 0.4));
    }

    // engagement is the natural inverse composite (for the UI's engagement stat)
    const engagement = clamp01(1 - Math.max(boredom, 0.6 * tiredness));

    return { tiredness, boredom, engagement, source: "heuristic" };
  }
}

// ------- TRAINED model wrapper (TensorFlow.js) -------
export class TrainedDetector {
  constructor() { this.tired = null; this.bored = null; }

  async load(basePath = "./models") {
    // tfjs is loaded globally via <script>. Models are optional.
    if (typeof tf === "undefined") return false;
    try {
      this.tired = await tf.loadLayersModel(`${basePath}/tiredness/model.json`);
    } catch (e) { /* not present */ }
    try {
      this.bored = await tf.loadLayersModel(`${basePath}/boredom/model.json`);
    } catch (e) { /* not present */ }
    return !!(this.tired || this.bored);
  }

  available() { return !!(this.tired || this.bored); }

  async predict(agg, videoEl) {
    let tiredness = null, boredom = null;
    if (this.tired) {
      const seq = agg.windowSequence();
      const x = tf.tensor([seq]);
      const out = this.tired.predict(x);
      tiredness = (await out.data())[0];
      tf.dispose([x, out]);
    }
    if (this.bored && videoEl) {
      const img = tf.tidy(() =>
        tf.browser.fromPixels(videoEl).resizeBilinear([160, 160])
          .toFloat().div(127.5).sub(1).expandDims(0));
      const feats = tf.tensor([agg.windowFeatures()]);
      const out = this.bored.predict({ image: img, features: feats });
      boredom = (await out.data())[0];
      tf.dispose([img, feats, out]);
    }
    return { tiredness, boredom, source: "trained" };
  }
}

// ------- Unified detector: prefer trained, fall back to heuristic per-signal ----
export class EngagementDetector {
  constructor() {
    this.heuristic = new HeuristicDetector();
    this.trained = new TrainedDetector();
    this.mode = "heuristic";
  }

  async init(basePath = "./models") {
    const ok = await this.trained.load(basePath);
    this.mode = ok ? "hybrid" : "heuristic";
    return this.mode;
  }

  async predict(agg, videoEl) {
    const wf = agg.windowFeaturesObject();
    const h = this.heuristic.predict(wf);
    if (this.trained.available()) {
      const t = await this.trained.predict(agg, videoEl);
      // use trained score where available, else heuristic for that signal
      return {
        tiredness: t.tiredness != null ? t.tiredness : h.tiredness,
        boredom: t.boredom != null ? t.boredom : h.boredom,
        source: t.tiredness != null || t.boredom != null ? "trained" : "heuristic",
        features: wf,
      };
    }
    return { ...h, features: wf };
  }
}
