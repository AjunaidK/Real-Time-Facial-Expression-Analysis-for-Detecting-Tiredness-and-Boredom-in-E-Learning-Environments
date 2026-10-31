// engine.js — orchestrates the live pipeline and session state.
// Emits events the UI subscribes to; keeps no DOM logic itself.

import { perFrameFeatures, TemporalAggregator } from "./features.js";
import { EngagementDetector } from "./detector.js";

const FPS = 10;
const WINDOW_SECONDS = 5;
const INFER_INTERVAL = 1000 / FPS;
const CALIBRATION_MS = 5000;

export class Engine extends EventTarget {
  constructor() {
    super();
    this.agg = new TemporalAggregator(FPS, WINDOW_SECONDS);
    this.detector = new EngagementDetector();
    this.faceMesh = null;
    this.camera = null;
    this.videoEl = null;
    this.running = false;
    this.mode = "client";            // 'client' | 'server'
    this.serverUrl = "http://localhost:8000";
    this.sessionId = "sess-" + Math.random().toString(36).slice(2);
    this.consented = false;

    this._lastInfer = 0;
    this._frameStamps = [];
    this._latencies = [];
    this._calibrating = false;
    this._calibStart = 0;

    // session log for analytics + CSV export (evaluation study, Ch 3.7)
    this.log = [];               // {t, tiredness, boredom, engagement, ...}
    this.events = [];            // {t, type} e.g. blink, yawn, disengaged
    this.startTime = null;

    // focus / attention tracking
    this._focusStreakStart = null;
    this._longestFocusStreak = 0;
    this._focusedTime = 0;
    this._onTaskTime = 0;
    this._lastScoreT = null;
  }

  async init(videoEl) {
    this.videoEl = videoEl;
    const mode = await this.detector.init("./models");
    this.dispatchEvent(new CustomEvent("detector-mode", { detail: { mode } }));

    this.faceMesh = new FaceMesh({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
    });
    this.faceMesh.setOptions({
      maxNumFaces: 1, refineLandmarks: true,
      minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    this.faceMesh.onResults(r => this._onResults(r));
    return mode;
  }

  async start() {
    this.camera = new Camera(this.videoEl, {
      onFrame: async () => { await this.faceMesh.send({ image: this.videoEl }); },
      width: 640, height: 480,
    });
    await this.camera.start();
    this.running = true;
    this.startTime = performance.now();
    this.dispatchEvent(new CustomEvent("started"));
  }

  stop() {
    this.running = false;
    try { this.camera?.stop(); } catch {}
    this.dispatchEvent(new CustomEvent("stopped"));
  }

  setMode(mode) { this.mode = mode; }

  startCalibration() {
    this._calibrating = true;
    this._calibStart = performance.now();
    this.detector.heuristic.startCalibration();
    this.dispatchEvent(new CustomEvent("calibration-start"));
  }

  async _onResults(results) {
    const now = performance.now();
    this._frameStamps.push(now);
    this._frameStamps = this._frameStamps.filter(t => now - t < 1000);
    const fps = this._frameStamps.length;

    const faceT0 = performance.now();
    const hasFace = results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0;
    const lm = hasFace ? results.multiFaceLandmarks[0] : null;
    const faceMs = performance.now() - faceT0;

    let ev = { blink: false, yawn: false };
    let frameFeat = null;
    if (hasFace) {
      frameFeat = perFrameFeatures(lm);
      ev = this.agg.update(frameFeat, true);
    } else {
      this.agg.update({}, false);
    }
    if (ev.blink) this._pushEvent("blink");
    if (ev.yawn) this._pushEvent("yawn");

    this.dispatchEvent(new CustomEvent("frame", {
      detail: { landmarks: lm, hasFace, fps, frameFeat,
                blink: ev.blink, yawn: ev.yawn },
    }));

    // calibration window
    if (this._calibrating) {
      if (this.agg.ready()) this.detector.heuristic.addCalibrationSample(this.agg.windowFeaturesObject());
      if (now - this._calibStart > CALIBRATION_MS) {
        this._calibrating = false;
        this.detector.heuristic.finishCalibration();
        this.dispatchEvent(new CustomEvent("calibration-done"));
      }
    }

    // throttle inference
    if (now - this._lastInfer < INFER_INTERVAL) return;
    this._lastInfer = now;
    if (!this.agg.ready()) {
      this.dispatchEvent(new CustomEvent("warming", {
        detail: { pct: this.agg.buffer.length / this.agg.windowLen } }));
      return;
    }

    let result, inferMs, faceScore;
    const t0 = performance.now();
    if (this.mode === "server") {
      result = await this._inferServer();
      inferMs = performance.now() - t0;
    } else {
      result = await this.detector.predict(this.agg, this.videoEl);
      inferMs = performance.now() - t0;
    }
    this._latencies.push(inferMs);
    if (this._latencies.length > 100) this._latencies.shift();

    if (result) {
      const f = result.features || {};
      // engagement: prefer detector-provided, else derive from tired/bored
      const engagement = (result.engagement != null)
        ? result.engagement
        : Math.max(0, Math.min(1, 1 - Math.max(result.boredom || 0, 0.6 * (result.tiredness || 0))));
      const tSec = (now - this.startTime) / 1000;
      const rec = {
        t: tSec,
        tiredness: result.tiredness, boredom: result.boredom, engagement,
        focusIndex: f.focus_index ?? null,
        sinceBlink: f.since_last_blink ?? null,
        meanBlinkGap: f.mean_blink_gap ?? null,
        source: result.source, faceMs, inferMs, fps,
      };
      this.log.push(rec);

      const disengaged = (result.tiredness > 0.7) || (result.boredom > 0.7);
      if (disengaged) this._pushEvent("disengaged");

      // live "focused" streak: engagement high AND not disengaged
      const focused = engagement >= 0.6 && !disengaged;
      if (focused) {
        if (this._focusStreakStart == null) this._focusStreakStart = tSec;
        const streak = tSec - this._focusStreakStart;
        if (streak > this._longestFocusStreak) this._longestFocusStreak = streak;
      } else {
        this._focusStreakStart = null;
      }
      // running focused-time total (time-on-task)
      if (this._lastScoreT != null) {
        const dt = tSec - this._lastScoreT;
        if (focused) this._focusedTime += dt;
        this._onTaskTime += dt;
      }
      this._lastScoreT = tSec;

      this.dispatchEvent(new CustomEvent("score", {
        detail: { ...rec, features: result.features, disengaged, focused,
                  currentFocusStreak: this._focusStreakStart != null ? (tSec - this._focusStreakStart) : 0,
                  longestFocusStreak: this._longestFocusStreak,
                  focusedTime: this._focusedTime, onTaskTime: this._onTaskTime,
                  avgLatency: this._avg(this._latencies) } }));
    }
  }

  async _inferServer() {
    const b64 = this._frameToB64();
    try {
      const resp = await fetch(`${this.serverUrl}/infer`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: this.sessionId, image_b64: b64,
                               record: this.consented }),
      });
      const d = await resp.json();
      return { tiredness: d.tiredness, boredom: d.boredom,
               source: "server", features: this.agg.windowFeaturesObject() };
    } catch (e) {
      this.dispatchEvent(new CustomEvent("server-error"));
      return null;
    }
  }

  _frameToB64() {
    const c = document.createElement("canvas");
    c.width = 320; c.height = 240;
    c.getContext("2d").drawImage(this.videoEl, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.6);
  }

  _pushEvent(type) {
    this.events.push({ t: (performance.now() - this.startTime) / 1000, type });
    this.dispatchEvent(new CustomEvent("event", { detail: { type } }));
  }

  _avg(a) { return a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0; }

  // ---- analytics for the summary screen ----
  summary() {
    const scored = this.log.filter(r => r.tiredness != null || r.boredom != null);
    const avg = k => scored.length ? scored.reduce((s, r) => s + (r[k] || 0), 0) / scored.length : 0;
    const pctAbove = (k, th) => scored.length
      ? scored.filter(r => (r[k] || 0) > th).length / scored.length : 0;
    const durationS = this.log.length ? this.log[this.log.length - 1].t : 0;

    // blink rate over the whole session (blinks per minute)
    const blinkRatePerMin = durationS > 0 ? (this.agg.totalBlinks / durationS) * 60 : 0;
    // average attention span = mean gap between blinks (proxy for focus bursts)
    const gaps = this.agg.blinkIntervals;
    const attentionSpanS = gaps.length ? gaps.reduce((s, g) => s + g, 0) / gaps.length : 0;

    // per-minute breakdown
    const perMinute = [];
    if (scored.length) {
      const mins = Math.max(1, Math.ceil(durationS / 60));
      for (let mMin = 0; mMin < mins; mMin++) {
        const seg = scored.filter(r => r.t >= mMin * 60 && r.t < (mMin + 1) * 60);
        if (!seg.length) continue;
        const segAvg = k => seg.reduce((s, r) => s + (r[k] || 0), 0) / seg.length;
        perMinute.push({
          minute: mMin + 1,
          engagement: segAvg("engagement"),
          tiredness: segAvg("tiredness"),
          boredom: segAvg("boredom"),
        });
      }
    }

    return {
      durationS, samples: scored.length,
      avgTiredness: avg("tiredness"), avgBoredom: avg("boredom"),
      avgEngagement: avg("engagement"),
      pctTired: pctAbove("tiredness", 0.6), pctBored: pctAbove("boredom", 0.6),
      pctFocused: durationS > 0 ? this._focusedTime / Math.max(this._onTaskTime, 1e-6) : 0,
      blinks: this.agg.totalBlinks, yawns: this.agg.totalYawns,
      blinkRatePerMin, attentionSpanS,
      longestFocusStreakS: this._longestFocusStreak,
      focusedTimeS: this._focusedTime,
      disengagements: this.events.filter(e => e.type === "disengaged").length,
      avgLatency: this._avg(this._latencies),
      avgFps: this.log.length ? this.log.reduce((s, r) => s + r.fps, 0) / this.log.length : 0,
      series: scored.map(r => ({ t: r.t, tiredness: r.tiredness, boredom: r.boredom, engagement: r.engagement })),
      perMinute,
    };
  }

  exportCSV() {
    const head = "t_seconds,tiredness,boredom,engagement,focus_index,since_last_blink_s,mean_blink_gap_s,source,face_ms,infer_ms,fps";
    const rows = this.log.map(r =>
      [r.t.toFixed(2), r.tiredness?.toFixed(4) ?? "", r.boredom?.toFixed(4) ?? "",
       r.engagement?.toFixed(4) ?? "", r.focusIndex?.toFixed(4) ?? "",
       r.sinceBlink?.toFixed(2) ?? "", r.meanBlinkGap?.toFixed(2) ?? "",
       r.source, r.faceMs?.toFixed(2) ?? "", r.inferMs?.toFixed(2) ?? "", r.fps].join(","));
    return [head, ...rows].join("\n");
  }
}
