"""The interactive picture for the bee labeller (a Streamlit custom component, v2).

Click on a bee -> a box appears.  Click on a box -> it is selected (drag it to move it).
Mouse wheel -> zoom.  Drag the picture -> move around.  Keys: Del delete, 1-9 type, +/- size,
Enter save & next, 0 fit picture, Esc deselect.
The edited boxes are sent back to Python only when a toolbar button is pressed.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

HTML = """
<div class="bl-root" tabindex="0">
  <div class="bl-bar">
    <span class="bl-classes"></span>
    <span class="bl-sep"></span>
    <button class="bl-btn" data-act="smaller" title="Make the selected box smaller (-)">&minus; size</button>
    <button class="bl-btn" data-act="bigger" title="Make the selected box bigger (+)">+ size</button>
    <button class="bl-btn" data-act="delete" title="Delete the selected box (Del)">Delete box</button>
    <button class="bl-btn" data-act="clear" title="Remove all boxes: no insect in this picture">No insect here</button>
    <button class="bl-btn" data-act="fit" title="Show the whole picture (0)">Whole picture</button>
  </div>
  <div class="bl-stage"><canvas></canvas></div>
  <div class="bl-bottom">
    <button class="bl-btn" data-act="back" title="Previous picture">&#9664; Back</button>
    <span class="bl-status"></span>
    <button class="bl-btn" data-act="skip" title="Next picture without saving">Skip</button>
    <button class="bl-btn bl-primary" data-act="save" title="Save and go to the next picture (Enter)">Save &amp; next &#9654;</button>
  </div>
  <div class="bl-help">Click a bee to put a box on it &middot; click a box to select it (drag to move) &middot;
    scroll to zoom &middot; drag the picture to move around &middot; keys: Del delete, 1-9 type, +/- size, Enter save</div>
</div>
"""

CSS = """
.bl-root { font-family: var(--st-font, sans-serif); color: var(--st-text-color); outline: none; }
.bl-bar, .bl-bottom { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0; }
.bl-sep { flex: 1; }
.bl-btn { border: 1px solid var(--st-border-color, #ccc); background: var(--st-secondary-background-color, #f0f2f6);
  color: var(--st-text-color); border-radius: 8px; padding: 6px 12px; cursor: pointer; font-size: 14px; }
.bl-btn:hover { border-color: var(--st-primary-color, #ff4b4b); }
.bl-btn.bl-active { outline: 3px solid var(--st-primary-color, #ff4b4b); }
.bl-primary { background: var(--st-primary-color, #ff4b4b); color: #fff; border-color: transparent; font-weight: 600; }
.bl-status { flex: 1; text-align: center; font-size: 14px; opacity: 0.85; }
.bl-stage { width: 100%; background: #111; border-radius: 8px; overflow: hidden; touch-action: none; }
.bl-stage canvas { display: block; width: 100%; cursor: crosshair; }
.bl-help { font-size: 12px; opacity: 0.7; margin-top: 4px; }
"""

JS = """
const STATE = new WeakMap();

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector(".bl-root");
  const canvas = parentElement.querySelector("canvas");
  const stage = parentElement.querySelector(".bl-stage");
  const status = parentElement.querySelector(".bl-status");
  const classBar = parentElement.querySelector(".bl-classes");
  if (!root || !canvas || !data) return;
  const ctx = canvas.getContext("2d");
  const classes = data.classes || ["bee"];
  const colors = data.colors || ["#ffd400"];

  let s = STATE.get(parentElement);
  if (!s) {
    s = { img: null, src: null, token: null, boxes: [], sel: -1, cls: 0, k: 1, kfit: 1, vx: 0, vy: 0,
          press: null, moved: false };
    STATE.set(parentElement, s);
  }
  if (s.token !== data.token) {           // a new picture (or "reset"): take the boxes from Python
    s.token = data.token;
    s.boxes = (data.boxes || []).map(b => ({ ...b }));
    s.sel = -1;
    s.needFit = true;
  }
  if (s.src !== data.image) {
    s.src = data.image;
    const im = new Image();
    im.onload = () => { s.img = im; s.needFit = true; layout(); };
    im.src = data.image;
  }

  // ------------------------------------------------------------------ class buttons
  classBar.innerHTML = "";
  classes.forEach((name, i) => {
    const b = document.createElement("button");
    b.className = "bl-btn" + (i === s.cls ? " bl-active" : "");
    b.textContent = (i + 1) + " " + name;
    b.style.borderLeft = "6px solid " + colors[i % colors.length];
    b.title = "Type of new boxes and of the selected box (key " + (i + 1) + ")";
    b.onclick = () => setClass(i);
    classBar.appendChild(b);
  });

  function setClass(i) {
    s.cls = i;
    if (s.sel >= 0) s.boxes[s.sel].cls = i;
    classBar.querySelectorAll(".bl-btn").forEach((b, j) => b.classList.toggle("bl-active", j === i));
    draw();
  }

  // ------------------------------------------------------------------ geometry
  function layout() {
    if (!s.img) return;
    const W = s.img.naturalWidth, H = s.img.naturalHeight;
    const cw = stage.clientWidth || 800;
    const ch = Math.min(Math.round(cw * H / W), Math.round(window.innerHeight * 0.72));
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
    canvas.style.height = ch + "px";
    const newFit = Math.min(cw / W, ch / H);
    if (!s.needFit && s.cw && (s.cw !== cw || s.ch !== ch)) {   // resized: keep relative zoom and centre
      const zoom = s.k / s.kfit, cx = s.vx + s.cw / (2 * s.k), cy = s.vy + s.ch / (2 * s.k);
      s.k = zoom * newFit; s.vx = cx - cw / (2 * s.k); s.vy = cy - ch / (2 * s.k);
    }
    s.cw = cw; s.ch = ch; s.dpr = dpr; s.kfit = newFit;
    if (s.needFit) fit(); else clampView();
    draw();
  }
  function fit() {
    const W = s.img.naturalWidth, H = s.img.naturalHeight;
    s.k = s.kfit; s.vx = (W - s.cw / s.k) / 2; s.vy = (H - s.ch / s.k) / 2; s.needFit = false;
  }
  const toImg = (px, py) => [s.vx + px / s.k, s.vy + py / s.k];
  const toScr = (x, y) => [(x - s.vx) * s.k, (y - s.vy) * s.k];
  function clampView() {
    const W = s.img.naturalWidth, H = s.img.naturalHeight, vw = s.cw / s.k, vh = s.ch / s.k;
    s.vx = vw >= W ? (W - vw) / 2 : Math.min(Math.max(s.vx, 0), W - vw);
    s.vy = vh >= H ? (H - vh) / 2 : Math.min(Math.max(s.vy, 0), H - vh);
  }

  // ------------------------------------------------------------------ drawing
  function draw() {
    if (!s.img || !s.cw) return;
    ctx.setTransform(s.dpr, 0, 0, s.dpr, 0, 0);
    ctx.fillStyle = "#111"; ctx.fillRect(0, 0, s.cw, s.ch);
    ctx.imageSmoothingEnabled = s.k < 2;
    const [ix, iy] = toScr(0, 0);
    ctx.drawImage(s.img, ix, iy, s.img.naturalWidth * s.k, s.img.naturalHeight * s.k);
    s.boxes.forEach((b, i) => {
      const [x1, y1] = toScr(b.x1, b.y1), [x2, y2] = toScr(b.x2, b.y2);
      const col = colors[b.cls % colors.length];
      const pad = 3;
      ctx.lineWidth = i === s.sel ? 3 : 2;
      ctx.strokeStyle = i === s.sel ? "#ffffff" : col;
      ctx.setLineDash(i === s.sel ? [6, 3] : []);
      ctx.strokeRect(x1 - pad, y1 - pad, x2 - x1 + 2 * pad, y2 - y1 + 2 * pad);
      ctx.setLineDash([]);
      ctx.fillStyle = col;
      ctx.font = "bold 13px sans-serif";
      ctx.font = b.label ? "bold 15px sans-serif" : "bold 13px sans-serif";
      ctx.fillText(b.label ? b.label : String(b.cls + 1), x1 - pad, y1 - pad - 3);
    });
    status.textContent = s.boxes.length + (s.boxes.length === 1 ? " box" : " boxes") +
      "  ·  zoom " + (s.k / s.kfit).toFixed(1) + "x" + (s.sel >= 0 ? "  ·  box selected" : "");
  }

  function hit(x, y) {
    const tol = 6 / s.k;
    for (let i = s.boxes.length - 1; i >= 0; i--) {
      const b = s.boxes[i];
      if (x >= b.x1 - tol && x <= b.x2 + tol && y >= b.y1 - tol && y <= b.y2 + tol) return i;
    }
    return -1;
  }

  function resize(f) {
    if (s.sel < 0) return;
    const b = s.boxes[s.sel], cx = (b.x1 + b.x2) / 2, cy = (b.y1 + b.y2) / 2;
    const w = Math.max(4, (b.x2 - b.x1) * f), h = Math.max(4, (b.y2 - b.y1) * f);
    Object.assign(b, { x1: cx - w / 2, x2: cx + w / 2, y1: cy - h / 2, y2: cy + h / 2 });
    draw();
  }

  function send(type) {
    setTriggerValue("action", { type, token: s.token, boxes: s.boxes.map(b => ({ ...b })) });
  }

  // ------------------------------------------------------------------ mouse
  canvas.onpointerdown = (e) => {
    root.focus({ preventScroll: true });
    canvas.setPointerCapture(e.pointerId);
    const r = canvas.getBoundingClientRect();
    const [x, y] = toImg(e.clientX - r.left, e.clientY - r.top);
    const i = hit(x, y);
    s.press = { sx: e.clientX, sy: e.clientY, vx: s.vx, vy: s.vy, box: i, x, y,
                orig: i >= 0 ? { ...s.boxes[i] } : null };
    s.moved = false;
  };
  canvas.onpointermove = (e) => {
    if (!s.press) return;
    const dx = e.clientX - s.press.sx, dy = e.clientY - s.press.sy;
    if (!s.moved && Math.hypot(dx, dy) < 4) return;
    s.moved = true;
    if (s.press.box >= 0 && s.press.box === s.sel) {      // drag the selected box
      const o = s.press.orig, b = s.boxes[s.sel];
      Object.assign(b, { x1: o.x1 + dx / s.k, x2: o.x2 + dx / s.k, y1: o.y1 + dy / s.k, y2: o.y2 + dy / s.k });
    } else {                                              // move the picture
      s.vx = s.press.vx - dx / s.k; s.vy = s.press.vy - dy / s.k; clampView();
    }
    draw();
  };
  canvas.onpointerup = (e) => {
    const p = s.press; s.press = null;
    if (!p || s.moved) return;
    if (p.box >= 0) { s.sel = p.box; setClass(s.boxes[p.box].cls); return; }
    const size = data.box_size || 40;                     // click on empty picture: new box
    s.boxes.push({ x1: p.x - size / 2, y1: p.y - size * 0.4, x2: p.x + size / 2, y2: p.y + size * 0.4, cls: s.cls });
    s.sel = s.boxes.length - 1;
    draw();
  };
  canvas.onwheel = (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const [x, y] = toImg(px, py);
    const k = Math.min(Math.max(s.k * (e.deltaY < 0 ? 1.25 : 0.8), s.kfit), s.kfit * 16);
    s.k = k; s.vx = x - px / k; s.vy = y - py / k; clampView(); draw();
  };

  // ------------------------------------------------------------------ buttons and keys
  parentElement.querySelectorAll("[data-act]").forEach(btn => {
    btn.onclick = () => {
      const act = btn.dataset.act;
      if (act === "smaller") resize(0.85);
      else if (act === "bigger") resize(1.18);
      else if (act === "delete") { if (s.sel >= 0) { s.boxes.splice(s.sel, 1); s.sel = -1; draw(); } }
      else if (act === "clear") { s.boxes = []; s.sel = -1; draw(); }
      else if (act === "fit") { fit(); draw(); }
      else send(act);                                     // back / skip / save -> Python
    };
  });
  root.onkeydown = (e) => {
    if (e.key === "Delete" || e.key === "Backspace") { if (s.sel >= 0) { s.boxes.splice(s.sel, 1); s.sel = -1; draw(); } }
    else if (e.key === "Enter") send("save");
    else if (e.key === "Escape") { s.sel = -1; draw(); }
    else if (e.key === "+" || e.key === "=") resize(1.18);
    else if (e.key === "-") resize(0.85);
    else if (e.key === "0") { fit(); draw(); }
    else if (/^[1-9]$/.test(e.key) && Number(e.key) <= classes.length) setClass(Number(e.key) - 1);
    else return;
    e.preventDefault();
  };

  s.layout = layout;                       // the resize listener is added once and calls the latest layout
  if (!s.onResize) {
    s.onResize = () => s.layout && s.layout();
    window.addEventListener("resize", s.onResize);
  }
  layout();
  return () => { window.removeEventListener("resize", s.onResize); s.onResize = null; };
}
"""

_COMPONENT = st.components.v2.component("bee_labeller_canvas", html=HTML, css=CSS, js=JS)

COLORS = ["#ffd400", "#ff5e3a", "#3ad1ff", "#7dff6a", "#d66bff", "#ff9ad5", "#ffffff", "#ffa500", "#00e0b0"]


def image_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def bee_canvas(image: str | Path, boxes: list[dict], classes: list[str], token: str, box_size: float = 40,
               default_class: int = 0, key: str = "bee_canvas"):
    """Show one picture with editable boxes. Returns the toolbar action (or None) of this run:
    ``{"type": "save"|"skip"|"back", "token": ..., "boxes": [{x1, y1, x2, y2, cls}, ...]}`` in pixels."""
    result = _COMPONENT(
        key=key,
        data={"image": image_data_url(image), "boxes": boxes, "classes": classes, "colors": COLORS,
              "token": token, "box_size": float(box_size), "default_class": int(default_class)},
        on_action_change=lambda: None,
    )
    return result.action
