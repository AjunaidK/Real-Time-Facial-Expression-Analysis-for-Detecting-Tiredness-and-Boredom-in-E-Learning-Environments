// overlay.js — draws the face-mesh overlay and live status on the video canvas.
import { LEFT_EYE, RIGHT_EYE, MOUTH } from "./features.js";

export class Overlay {
  constructor(canvas, video) {
    this.canvas = canvas;
    this.video = video;
    this.ctx = canvas.getContext("2d");
  }

  resize() {
    this.canvas.width = this.video.videoWidth || 640;
    this.canvas.height = this.video.videoHeight || 480;
  }

  draw(landmarks, { hasFace, blink, yawn, disengaged }) {
    const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (!hasFace) {
      ctx.fillStyle = "rgba(230,164,23,0.14)";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#E6A417";
      ctx.font = "600 20px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No face detected", W / 2, H / 2);
      return;
    }

    // full mesh — subtle dots
    ctx.fillStyle = disengaged ? "rgba(230,164,23,0.55)" : "rgba(120,200,220,0.45)";
    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * W, p.y * H, 0.9, 0, Math.PI * 2);
      ctx.fill();
    }

    // highlight tracked features
    this._trace(LEFT_EYE, landmarks, W, H, blink ? "#E6A417" : "#38BDF8");
    this._trace(RIGHT_EYE, landmarks, W, H, blink ? "#E6A417" : "#38BDF8");
    const mouth = [MOUTH.left, MOUTH.top, MOUTH.right, MOUTH.bottom];
    this._trace(mouth, landmarks, W, H, yawn ? "#E6A417" : "#5EEAD4");

    // event badges
    let bx = 12;
    if (blink) bx = this._badge(ctx, bx, "BLINK", "#38BDF8");
    if (yawn)  bx = this._badge(ctx, bx, "YAWN", "#5EEAD4");
    if (disengaged) this._badge(ctx, bx, "DISENGAGED", "#E6A417");
  }

  _trace(idx, lm, W, H, color) {
    const ctx = this.ctx;
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    idx.forEach((i, k) => {
      const x = lm[i].x * W, y = lm[i].y * H;
      k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.closePath(); ctx.stroke();
  }

  _badge(ctx, x, text, color) {
    ctx.font = "700 12px Inter, system-ui, sans-serif";
    const w = ctx.measureText(text).width + 16;
    ctx.fillStyle = color;
    this._roundRect(ctx, x, 12, w, 22, 6); ctx.fill();
    ctx.fillStyle = "#0B1220"; ctx.textAlign = "left";
    ctx.fillText(text, x + 8, 27);
    return x + w + 8;
  }

  _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
}
