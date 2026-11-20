"""FastAPI app: MJPEG video stream + detection JSON API + snapshots.

Run:  python app.py        (or)   uvicorn app:app --host 0.0.0.0 --port 8000
Open: http://localhost:8000
"""
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline import DetectionPipeline

pipeline = None

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RTSP Vehicle Detection</title>
  <style>
    body { margin: 0; background: #0e1117; color: #e6e6e6;
           font-family: system-ui, Segoe UI, Roboto, sans-serif; }
    header { padding: 12px 18px; background: #161b22; border-bottom: 1px solid #30363d; }
    header h1 { margin: 0; font-size: 16px; font-weight: 600; }
    .wrap { display: flex; gap: 16px; padding: 16px; flex-wrap: wrap; }
    .stream { flex: 3 1 640px; }
    .stream img { width: 100%; border-radius: 10px; border: 1px solid #30363d; background: #000; }
    .panel { flex: 1 1 300px; background: #161b22; border: 1px solid #30363d;
             border-radius: 10px; padding: 14px; }
    .stat { display: flex; justify-content: space-between; padding: 6px 0;
            border-bottom: 1px dashed #30363d; font-size: 14px; }
    .stat span:last-child { color: #58a6ff; font-weight: 600; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
    th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #30363d; }
    th { color: #8b949e; font-weight: 500; }
  </style>
</head>
<body>
  <header><h1>RTSP Vehicle Detection Pipeline</h1></header>
  <div class="wrap">
    <div class="stream">
      <img src="/video_feed" alt="live stream" />
    </div>
    <div class="panel">
      <div class="stat"><span>Camera</span><span id="cam">-</span></div>
      <div class="stat"><span>RTSP</span><span id="rtsp">-</span></div>
      <div class="stat"><span>FPS</span><span id="fps">-</span></div>
      <div class="stat"><span>Vehicles in frame</span><span id="veh">-</span></div>
      <div class="stat"><span>Records saved</span><span id="rec">-</span></div>
      <div class="stat"><span>OCR</span><span id="ocr">-</span></div>
      <h3 style="font-size:13px;margin:14px 0 0;">Latest detections</h3>
      <table>
        <thead><tr><th>Time</th><th>Vehicle</th><th>Plate</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </div>
<script>
async function tick() {
  try {
    const s = await (await fetch('/api/stats')).json();
    document.getElementById('cam').textContent = s.camera_number;
    document.getElementById('rtsp').textContent = s.rtsp_connected ? 'connected' : 'connecting...';
    document.getElementById('fps').textContent = s.fps;
    document.getElementById('veh').textContent = s.vehicles_in_frame;
    document.getElementById('rec').textContent = s.records;
    document.getElementById('ocr').textContent = s.ocr_available ? 'on' : 'off';

    const rows = await (await fetch('/api/detections?limit=12')).json();
    document.getElementById('rows').innerHTML = rows.slice().reverse().map(r =>
      `<tr><td>${r.time}</td><td>${r.vehicle_type}</td><td>${r.plate_text || '-'}</td></tr>`
    ).join('');
  } catch (e) { /* ignore transient errors */ }
}
tick();
setInterval(tick, 1500);
</script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app):
    global pipeline
    print("[app] starting pipeline ...")
    pipeline = DetectionPipeline(config).start()
    try:
        yield
    finally:
        print("[app] stopping pipeline ...")
        pipeline.stop()


app = FastAPI(title="RTSP Vehicle Detection Pipeline", lifespan=lifespan)
app.mount("/media", StaticFiles(directory=str(config.OUTPUT_DIR)), name="media")


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/video_feed")
def video_feed():
    def generate():
        last_id = -1
        while True:
            publish_id, jpeg = pipeline.wait_jpeg(last_id, timeout=2.0)
            if jpeg is None or publish_id == last_id:
                time.sleep(0.01)  # no new frame yet; avoid a busy loop
                continue
            last_id = publish_id
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/detections")
def detections(limit: int = 100):
    return JSONResponse(pipeline.store.all(limit))


@app.get("/api/stats")
def stats():
    return JSONResponse(pipeline.stats())


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
