#!/usr/bin/env python3
"""Browse a COCO split with raw | annotated side by side, arrow keys to move.

    python tools/view_tiled.py /scratch/ad158/gollum-v4-tiled --split train

Then open the printed URL. Same controls as view_annotations.py, but the left
canvas always shows the untouched tile and the right one shows it with masks
so cut-off annotations at tile edges are easy to spot.
"""

import argparse
import json
import os
import socket
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer

# Reuse the dataset-agnostic pieces (works on any COCO split, tiled or not).
from view_annotations import PALETTE, build, routable_ip

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>tiled annotations</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0e13; color:#e6e8eb;
         font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
         display:flex; height:100vh; overflow:hidden; }
  #stage { flex:1; display:flex; align-items:center; justify-content:center;
           gap:14px; position:relative; padding:14px; box-sizing:border-box; }
  .pane { display:flex; flex-direction:column; align-items:center; gap:6px;
          min-width:0; min-height:0; flex:1; height:100%; }
  .pane h2 { margin:0; font-size:11px; letter-spacing:.08em; text-transform:uppercase;
             color:#6f7c8e; }
  canvas { max-width:100%; max-height:calc(100% - 20px); image-rendering:auto;
           border:1px solid #1f2733; }
  #side { width:290px; border-left:1px solid #1f2733; padding:16px; overflow-y:auto;
          background:#0e1219; display:flex; flex-direction:column; gap:14px; flex:none; }
  h1 { font-size:13px; margin:0; letter-spacing:.08em; text-transform:uppercase; color:#8b97a8; }
  .row { display:flex; justify-content:space-between; gap:10px; }
  .muted { color:#8b97a8; }
  .file { font-size:11px; word-break:break-all; color:#6f7c8e; }
  label { display:flex; align-items:center; gap:8px; cursor:pointer; padding:3px 0; }
  .sw { width:11px; height:11px; border-radius:3px; flex:none; }
  kbd { background:#1a212c; border:1px solid #2a3441; border-radius:4px;
        padding:1px 6px; font:11px ui-monospace,monospace; }
  #bar { position:absolute; left:0; right:0; bottom:0; height:3px; background:#161c25; }
  #fill { height:100%; background:#22d3ee; width:0; }
  input[type=number] { width:80px; background:#161c25; color:#e6e8eb;
        border:1px solid #2a3441; border-radius:5px; padding:3px 6px; }
  hr { border:0; border-top:1px solid #1f2733; margin:2px 0; }
</style>
<div id="stage">
  <div class="pane"><h2>raw</h2><canvas id="cvRaw"></canvas></div>
  <div class="pane"><h2>annotated</h2><canvas id="cvAnn"></canvas></div>
  <div id="bar"><div id="fill"></div></div>
</div>
<div id="side">
  <h1>Tiled Annotations</h1>
  <div class="row"><span class="muted">split</span><b id="split"></b></div>
  <div class="row"><span class="muted">image</span><b id="idx"></b></div>
  <div class="row"><span class="muted">instances</span><b id="ninst"></b></div>
  <div class="file" id="fname"></div>
  <hr>
  <h1>Classes</h1>
  <div id="classes"></div>
  <hr>
  <h1>Show only</h1>
  <div id="only"></div>
  <hr>
  <h1>View</h1>
  <label><input type="checkbox" id="showMask" checked> filled masks</label>
  <label><input type="checkbox" id="showBox"> bounding boxes</label>
  <label><input type="checkbox" id="showLbl" checked> labels</label>
  <label><input type="checkbox" id="zoom"> 2x zoom</label>
  <hr>
  <div class="row"><span class="muted">jump to</span>
    <input type="number" id="jump" min="1"></div>
  <hr>
  <h1>Keys</h1>
  <div class="muted">
    <kbd>&larr;</kbd> <kbd>&rarr;</kbd> prev / next<br>
    <kbd>&uarr;</kbd> <kbd>&darr;</kbd> jump 25<br>
    <kbd>m</kbd> masks &nbsp; <kbd>b</kbd> boxes &nbsp; <kbd>l</kbd> labels<br>
    <kbd>z</kbd> zoom &nbsp; <kbd>e</kbd> next empty
  </div>
</div>
<script>
const cvRaw = document.getElementById('cvRaw'), ctxRaw = cvRaw.getContext('2d');
const cvAnn = document.getElementById('cvAnn'), ctxAnn = cvAnn.getContext('2d');
let D = null, i = 0, view = [], img = new Image();
const el = id => document.getElementById(id);

fetch('data.json').then(r => r.json()).then(d => {
  D = d;
  el('split').textContent = d.split;
  d.classes.forEach((c, k) => {
    el('classes').insertAdjacentHTML('beforeend',
      '<label><span class="sw" style="background:' + d.colors[k] + '"></span>' +
      '<input type="checkbox" data-c="' + k + '" class="vis" checked>' + c +
      ' <span class="muted">(' + d.counts[k] + ')</span></label>');
    el('only').insertAdjacentHTML('beforeend',
      '<label><input type="radio" name="only" data-o="' + k + '">' + c + '</label>');
  });
  el('only').insertAdjacentHTML('afterbegin',
    '<label><input type="radio" name="only" data-o="-1" checked>all images</label>');
  document.querySelectorAll('.vis').forEach(x => x.onchange = draw);
  document.querySelectorAll('[data-o]').forEach(x => x.onchange = e => {
    filter(+e.target.dataset.o); });
  ['showMask','showBox','showLbl','zoom'].forEach(id => el(id).onchange = () => {
    id === 'zoom' ? show() : draw(); });
  el('jump').onchange = e => { i = Math.min(view.length - 1,
    Math.max(0, (+e.target.value || 1) - 1)); show(); };
  filter(-1);
});

function filter(c) {
  view = D.images.map((_, k) => k).filter(k =>
    c < 0 || D.images[k].a.some(a => a.c === c));
  el('jump').max = view.length;
  i = 0; show();
}

function show() {
  if (!view.length) return;
  const im = D.images[view[i]];
  el('idx').textContent = (i + 1) + ' / ' + view.length;
  el('ninst').textContent = im.a.length;
  el('fname').textContent = im.f;
  el('jump').value = i + 1;
  el('fill').style.width = (100 * (i + 1) / view.length) + '%';
  img.onload = draw;
  img.src = 'img/' + encodeURIComponent(im.f);
}

function draw() {
  if (!img.complete || !img.naturalWidth) return;
  const im = D.images[view[i]], s = el('zoom').checked ? 2 : 1;
  for (const cv of [cvRaw, cvAnn]) { cv.width = im.w * s; cv.height = im.h * s; }

  ctxRaw.setTransform(1, 0, 0, 1, 0, 0);
  ctxRaw.drawImage(img, 0, 0, cvRaw.width, cvRaw.height);

  ctxAnn.setTransform(1, 0, 0, 1, 0, 0);
  ctxAnn.drawImage(img, 0, 0, cvAnn.width, cvAnn.height);
  ctxAnn.scale(s, s);
  const on = new Set([...document.querySelectorAll('.vis:checked')]
    .map(x => +x.dataset.c));
  for (const a of im.a) {
    if (!on.has(a.c)) continue;
    const col = D.colors[a.c];
    ctxAnn.strokeStyle = col; ctxAnn.lineWidth = 1.5 / s;
    ctxAnn.fillStyle = col + '55';
    for (const p of a.p) {
      ctxAnn.beginPath();
      ctxAnn.moveTo(p[0], p[1]);
      for (let k = 2; k < p.length; k += 2) ctxAnn.lineTo(p[k], p[k + 1]);
      ctxAnn.closePath();
      if (el('showMask').checked) ctxAnn.fill();
      ctxAnn.stroke();
    }
    if (el('showBox').checked) {
      ctxAnn.setLineDash([3, 3]);
      ctxAnn.strokeRect(a.b[0], a.b[1], a.b[2], a.b[3]);
      ctxAnn.setLineDash([]);
    }
    if (el('showLbl').checked) {
      const t = D.classes[a.c].split(' -')[0] + ' ' + Math.round(a.r) + 'px';
      ctxAnn.font = (10 / s) + 'px ui-monospace,monospace';
      const w = ctxAnn.measureText(t).width;
      ctxAnn.fillStyle = col;
      ctxAnn.fillRect(a.b[0], a.b[1] - 12 / s, w + 5 / s, 12 / s);
      ctxAnn.fillStyle = '#04070c';
      ctxAnn.fillText(t, a.b[0] + 2.5 / s, a.b[1] - 3 / s);
    }
  }
}

addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const n = view.length;
  if (e.key === 'ArrowRight') i = (i + 1) % n;
  else if (e.key === 'ArrowLeft') i = (i - 1 + n) % n;
  else if (e.key === 'ArrowDown') i = Math.min(n - 1, i + 25);
  else if (e.key === 'ArrowUp') i = Math.max(0, i - 25);
  else if (e.key === 'e') {
    for (let k = 1; k <= n; k++) {
      const j = (i + k) % n;
      if (!D.images[view[j]].a.length) { i = j; break; }
    }
  }
  else if ('mblz'.includes(e.key)) {
    const id = { m: 'showMask', b: 'showBox', l: 'showLbl', z: 'zoom' }[e.key];
    el(id).checked = !el(id).checked;
    return e.preventDefault(), (e.key === 'z' ? show() : draw());
  } else return;
  e.preventDefault(); show();
});
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *a, payload=None, img_dir=None, **kw):
        self.payload = payload
        self.img_dir = img_dir
        super().__init__(*a, **kw)

    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import unquote

        path = unquote(self.path.split("?")[0])
        if path in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if path == "/data.json":
            return self._send(json.dumps(self.payload).encode(), "application/json")
        if path.startswith("/img/"):
            name = os.path.basename(path[5:])
            fp = os.path.join(self.img_dir, name)
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    return self._send(f.read(), "image/jpeg")
        self.send_error(404)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--split", default="train")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--host", default="0.0.0.0",
                   help="0.0.0.0 lets you reach it without an ssh tunnel")
    args = p.parse_args()

    payload = build(args.root, args.split)
    n_ann = sum(len(im["a"]) for im in payload["images"])
    print(f"{args.split}: {len(payload['images'])} images, {n_ann} instances")
    for name, c in zip(payload["classes"], payload["counts"]):
        print(f"  {name[:34]:<36}{c:>6}")

    handler = partial(Handler, payload=payload,
                      img_dir=os.path.join(args.root, args.split))
    srv = HTTPServer((args.host, args.port), handler)

    ip, user = routable_ip(), os.environ.get("USER", "user")
    print(f"\nserving {socket.getfqdn()} on {args.host}:{args.port}")
    if args.host != "127.0.0.1":
        print(f"  direct:  http://{ip}:{args.port}")
    print(f"  tunnel:  ssh -N -L {args.port}:localhost:{args.port} {user}@{ip}")
    print(f"           then http://localhost:{args.port}")
    print("(ctrl-c to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
