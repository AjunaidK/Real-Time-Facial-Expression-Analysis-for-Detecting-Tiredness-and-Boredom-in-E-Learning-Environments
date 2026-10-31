// ui.js — wires the Engine to the DOM. No detection logic lives here.
import { Engine } from "./engine.js";
import { Overlay } from "./overlay.js";
import { MiniChart } from "./chart.js";

const $ = id => document.getElementById(id);
const ARC = 352;                       // 2πr for r=56

const engine = new Engine();
const video = $("video");
const overlay = new Overlay($("overlay"), video);
const chart = new MiniChart($("chart"));
let sumChart = null;
let bytesSent = 0;

function setPill(el, cls, text) {
  el.className = "pill " + cls;
  el.innerHTML = `<span class="led"></span> ${text}`;
}

// ---------- init ----------
(async function () {
  const mode = await engine.init(video);
  if (mode === "hybrid") setPill($("modePill"), "trained", "trained models loaded");
  else setPill($("modePill"), "heuristic", "heuristic detector (no trained model)");
})();

// ---------- controls ----------
$("startBtn").addEventListener("click", async () => {
  engine.consented = true;
  $("startBtn").disabled = true;
  try {
    await engine.start();
    $("curtain").classList.add("hidden");
    $("stopBtn").disabled = false;
    $("calibBtn").disabled = false;
    setPill($("livePill"), "live", "live");
    setTimeout(() => { overlay.resize(); engine.startCalibration(); }, 400);
  } catch (e) {
    $("startBtn").disabled = false;
    alert("Camera access is required for the demo. Please allow it and try again.");
  }
});

$("calibBtn").addEventListener("click", () => engine.startCalibration());

$("stopBtn").addEventListener("click", () => {
  engine.stop();
  showSummary();
});

$("modeToggle").addEventListener("click", () => {
  const on = $("modeToggle").classList.toggle("on");
  engine.setMode(on ? "server" : "client");
  $("perfMode").textContent = on ? "server-side" : "client-side";
  setPill($("modePill"), on ? "" : (engine.detector.trained.available() ? "trained" : "heuristic"),
          on ? "server-side inference" : (engine.detector.trained.available() ? "trained models loaded" : "heuristic detector"));
});

// ---------- engine events ----------
engine.addEventListener("calibration-start", () => {
  $("warming").classList.remove("hidden");
  $("warming").textContent = "calibrating your alert baseline… hold still & look at the screen";
});
engine.addEventListener("calibration-done", () => {
  $("warming").textContent = "calibration complete ✓";
  setTimeout(() => $("warming").classList.add("hidden"), 1500);
});

engine.addEventListener("detector-mode", e => { /* handled in init */ });

engine.addEventListener("frame", e => {
  const d = e.detail;
  overlay.resize();
  overlay.draw(d.landmarks, { hasFace: d.hasFace, blink: d.blink, yawn: d.yawn,
                              disengaged: window.__disengaged });
  $("fpsTag").textContent = `${d.fps} fps`;
  $("pFps").textContent = d.fps;
});

engine.addEventListener("warming", e => {
  if (engine._calibrating) return;
  $("warming").classList.remove("hidden");
  $("warming").textContent = `building analysis window… ${Math.round(e.detail.pct * 100)}%`;
  if (e.detail.pct >= 1) $("warming").classList.add("hidden");
});

engine.addEventListener("score", e => {
  const d = e.detail;
  if (engine._calibrating) return;
  $("warming").classList.add("hidden");

  const tired = d.tiredness ?? 0, bored = d.boredom ?? 0;
  const eng = d.engagement ?? Math.max(0, 1 - Math.max(bored, 0.6 * tired));
  updateGauge("tired", tired);
  updateGauge("bored", bored);
  updateGauge("eng", eng);
  $("srcTag").textContent = d.source;

  // features
  const f = d.features;
  setMetric("perclos", f.perclos, `${Math.round(f.perclos * 100)}%`, f.perclos.toFixed(2), f.perclos / 0.4);
  const blinkPerMin = f.blink_rate * 60;
  setMetric("blink", null, "", Math.round(blinkPerMin), blinkPerMin / 40);
  $("blinkV").textContent = Math.round(blinkPerMin);
  $("yawnV").textContent = engine.agg.totalYawns;
  $("yawnBar") && ($("yawnBar").style.width = Math.min(100, engine.agg.totalYawns * 20) + "%");
  $("offV").textContent = `${Math.round(f.offscreen_prop * 100)}%`;
  $("offBar").style.width = Math.min(100, f.offscreen_prop * 100) + "%";
  $("perclosPct").textContent = `${Math.round(f.perclos * 100)}%`;

  // attention & focus tiles
  $("streakV").textContent = fmtSecs(d.currentFocusStreak || 0);
  $("streakBar").style.width = Math.min(100, (d.currentFocusStreak || 0) / 60 * 100) + "%";
  $("longStreakV").textContent = fmtSecs(d.longestFocusStreak || 0);
  $("spanV").textContent = (f.mean_blink_gap ? f.mean_blink_gap.toFixed(1) : "0") + "s";
  const onTaskPct = d.onTaskTime > 0 ? (d.focusedTime / d.onTaskTime) : 0;
  $("onTaskV").textContent = Math.round(onTaskPct * 100) + "%";
  $("onTaskBar").style.width = Math.round(onTaskPct * 100) + "%";
  $("sinceBlinkV").textContent = (f.since_last_blink ? f.since_last_blink.toFixed(1) : "0") + "s";
  $("watchedV").textContent = fmtSecs(getVideoElapsed());

  // chart + perf
  chart.push(d.t, tired, bored, eng);
  $("latTag").textContent = `${Math.round(d.avgLatency)} ms`;
  $("pLat").textContent = Math.round(d.avgLatency);
  if (engine.mode === "server") { bytesSent += 320 * 240 * 0.15; $("pBw").textContent = Math.round(bytesSent / 1024); }

  // disengagement prompt
  window.__disengaged = d.disengaged;
  $("prompt").classList.toggle("show", d.disengaged);
});

engine.addEventListener("server-error", () => {
  setPill($("livePill"), "heuristic", "server unreachable — start uvicorn");
});

// ---------- helpers ----------
function updateGauge(which, v) {
  const pct = Math.max(0, Math.min(1, v));
  $(which + "Arc").style.strokeDashoffset = ARC * (1 - pct);
  $(which + "Val").textContent = Math.round(pct * 100);
  let state, color;
  if (which === "tired") {
    state = pct > 0.7 ? "Very tired" : pct > 0.4 ? "Getting tired" : "Alert";
    color = pct > 0.7 ? "#F5D98B" : pct > 0.4 ? "#93A6BE" : "#5EEAD4";
  } else if (which === "bored") {
    state = pct > 0.7 ? "Disengaged" : pct > 0.4 ? "Drifting" : "Engaged";
    color = pct > 0.7 ? "#F5D98B" : pct > 0.4 ? "#93A6BE" : "#5EEAD4";
  } else { // engagement — high is good, so colours invert
    state = pct > 0.66 ? "Focused" : pct > 0.4 ? "Partial" : "Low";
    color = pct > 0.66 ? "#5EEAD4" : pct > 0.4 ? "#93A6BE" : "#F5D98B";
  }
  $(which + "State").textContent = state;
  $(which + "State").style.color = color;
}

function setMetric(id, _v, _pct, val, barFrac) {
  if (val !== undefined && $(id + "V")) $(id + "V").textContent = val;
  if ($(id + "Bar")) $(id + "Bar").style.width = Math.min(100, (barFrac || 0) * 100) + "%";
}

function fmtSecs(s) {
  s = Math.round(s || 0);
  if (s < 60) return s + "s";
  return Math.floor(s / 60) + "m " + (s % 60) + "s";
}

// ---------- summary ----------
function showSummary() {
  const s = engine.summary();
  const g = $("sumGrid");
  const stat = (b, l) => `<div class="sum-stat"><b>${b}</b><span>${l}</span></div>`;
  g.innerHTML =
    stat(fmtTime(s.durationS), "duration") +
    stat(Math.round(s.avgEngagement * 100), "avg engagement") +
    stat(Math.round(s.avgTiredness * 100), "avg tiredness") +
    stat(Math.round(s.avgBoredom * 100), "avg boredom") +
    stat(Math.round(s.pctFocused * 100) + "%", "time on task") +
    stat(fmtSecs(s.longestFocusStreakS), "longest focus") +
    stat(s.attentionSpanS.toFixed(1) + "s", "attention span") +
    stat(Math.round(s.blinkRatePerMin), "blinks/min") +
    stat(s.yawns, "yawns") +
    stat(s.disengagements, "disengagements") +
    stat(Math.round(s.avgLatency) + "ms", "avg latency") +
    stat(Math.round(s.avgFps), "avg fps");

  // per-minute breakdown table
  renderPerMinute(s.perMinute);
  $("modalBg").classList.add("show");
  if (!sumChart) sumChart = new MiniChart($("sumChart"));
  sumChart.setSeries(s.series);
}

$("closeModal").addEventListener("click", () => $("modalBg").classList.remove("show"));
$("restartBtn").addEventListener("click", () => location.reload());
$("exportBtn").addEventListener("click", () => {
  const blob = new Blob([engine.exportCSV()], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `engagesense_session_${Date.now()}.csv`;
  a.click();
});

function fmtTime(s) {
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
}

// ---------- per-minute breakdown ----------
function renderPerMinute(perMinute) {
  const wrap = $("perMinuteWrap");
  if (!wrap) return;
  if (!perMinute || !perMinute.length) { wrap.innerHTML = ""; return; }
  let html = '<div class="pm-title">Per-minute breakdown (engagement / tiredness / boredom)</div>';
  perMinute.forEach(m => {
    const e = Math.round(m.engagement * 100);
    const t = Math.round(m.tiredness * 100);
    const b = Math.round(m.boredom * 100);
    html +=
      '<div class="pm-row"><div class="lab">min ' + m.minute + '</div>' +
      '<div class="pm-bars">' +
        '<div class="seg" style="width:' + e + '%;background:#5EEAD4" title="engagement ' + e + '%"></div>' +
        '<div class="seg" style="width:' + Math.max(0, t) / 2 + '%;background:#E6A417" title="tiredness ' + t + '%"></div>' +
        '<div class="seg" style="width:' + Math.max(0, b) / 2 + '%;background:#38BDF8" title="boredom ' + b + '%"></div>' +
      '</div></div>';
  });
  wrap.innerHTML = html;
}

// ---------- video panel: YouTube link OR upload OR direct URL ----------
const lessonVideo = $("lesson");
const ytFrame = $("ytFrame");
let ytStartTime = null;   // for YouTube we can't read currentTime, so track wall-clock

function parseYouTubeId(url) {
  // handles youtu.be/ID, youtube.com/watch?v=ID, /embed/ID, /shorts/ID
  const patterns = [
    /(?:youtube\.com\/watch\?v=)([\w-]{11})/,
    /(?:youtu\.be\/)([\w-]{11})/,
    /(?:youtube\.com\/embed\/)([\w-]{11})/,
    /(?:youtube\.com\/shorts\/)([\w-]{11})/,
  ];
  for (const p of patterns) { const m = url.match(p); if (m) return m[1]; }
  return null;
}

function showYouTube(id) {
  lessonVideo.pause?.();
  lessonVideo.classList.add("hidden");
  ytFrame.classList.remove("hidden");
  ytFrame.src = `https://www.youtube.com/embed/${id}?autoplay=1&rel=0`;
  ytStartTime = performance.now();
  $("lessonTitle").textContent = "YouTube lesson";
  $("lessonSub").textContent = "Loaded from link · monitoring your engagement";
}

function showLocalVideo(src, title) {
  ytFrame.classList.add("hidden");
  ytFrame.src = "";
  ytStartTime = null;
  lessonVideo.classList.remove("hidden");
  if (src) { lessonVideo.src = src; lessonVideo.load(); lessonVideo.play?.().catch(() => {}); }
  $("lessonTitle").textContent = title || "Uploaded video";
  $("lessonSub").textContent = "Playing locally · monitoring your engagement";
}

function getVideoElapsed() {
  // YouTube: wall-clock since load (best we can do behind the iframe wall)
  if (ytFrame && !ytFrame.classList.contains("hidden") && ytStartTime != null)
    return (performance.now() - ytStartTime) / 1000;
  // local video: real playback position
  if (lessonVideo && !lessonVideo.classList.contains("hidden"))
    return lessonVideo.currentTime || 0;
  return 0;
}

$("loadUrlBtn")?.addEventListener("click", () => {
  const url = ($("videoUrl").value || "").trim();
  if (!url) return;
  const ytId = parseYouTubeId(url);
  if (ytId) showYouTube(ytId);
  else showLocalVideo(url, "Video from URL");   // treat as a direct video link
});

$("videoUrl")?.addEventListener("keydown", e => { if (e.key === "Enter") $("loadUrlBtn").click(); });

$("videoFile")?.addEventListener("change", e => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const objUrl = URL.createObjectURL(file);
  showLocalVideo(objUrl, file.name);
});
