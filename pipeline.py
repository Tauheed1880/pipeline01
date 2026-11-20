"""Pipeline orchestration.

Thread model (built for low latency so the browser stream stays smooth):

  rtsp-reader thread  -> always keeps the newest frame (no buffering delay)
  pipeline thread     -> detect -> track -> annotate -> encode JPEG -> publish
  recorder thread     -> OCR + snapshot saving + JSON write, off the hot path

Heavy work (OCR, disk IO) runs in the recorder thread through a queue, so the
stream never blocks waiting for it.
"""
import queue
import threading
import time
from datetime import datetime

import cv2

import config as default_config
from capture import RTSPStream
from detector import VehiclePlateDetector
from ocr import PlateOCR
from storage import DetectionStore
from tracker import IouTracker


class DetectionPipeline:
    def __init__(self, settings=default_config):
        self.cfg = settings
        self.stream = RTSPStream(settings.RTSP_URL, settings.RECONNECT_DELAY).start()
        self.detector = VehiclePlateDetector(
            settings.VEHICLE_MODEL,
            settings.PLATE_MODEL,
            imgsz=settings.IMG_SIZE,
            vehicle_conf=settings.VEHICLE_CONF,
            plate_conf=settings.PLATE_CONF,
            vehicle_class_ids=settings.VEHICLE_CLASS_IDS,
            device=settings.DEVICE,
        )
        self.ocr = PlateOCR()
        self.tracker = IouTracker(settings.IOU_THRESHOLD, settings.MAX_MISSED)
        self.store = DetectionStore(
            settings.RECORDS_FILE, settings.SNAPSHOT_DIR, settings.PLATE_DIR
        )

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = f"{settings.CAMERA_NUMBER.replace(' ', '')}_{run_stamp}"

        self._latest_jpeg = None
        self._publish_id = 0
        self._jpeg_cond = threading.Condition()

        self._record_queue = queue.Queue(maxsize=200)
        self._running = False
        self._thread = None
        self._recorder = None
        self._fps = 0.0
        self._vehicle_count = 0

    # ------------------------------------------------------------------ life
    def start(self):
        self._running = True
        self._recorder = threading.Thread(target=self._recorder_loop, name="recorder", daemon=True)
        self._recorder.start()
        self._thread = threading.Thread(target=self._loop, name="pipeline", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._recorder is not None:
            self._recorder.join(timeout=5.0)  # drains any pending records
        self.stream.stop()

    # ------------------------------------------------------------ main loop
    def _loop(self):
        last = time.time()
        while self._running:
            item = self.stream.read()
            if item is None:
                time.sleep(0.005)
                continue
            frame, _ = item
            try:
                self._process(frame)
            except Exception as exc:  # never let one bad frame kill the stream
                print(f"[pipeline] frame processing error: {exc}")

            now = time.time()
            dt = now - last
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            last = now

    def _process(self, frame):
        now = datetime.now()
        vehicles = self.detector.detect_vehicles(frame)
        plates = self.detector.detect_plates(frame, [v["box"] for v in vehicles])
        tracks = self.tracker.update(
            [(v["box"], v["label"], v["conf"]) for v in vehicles]
        )
        self._vehicle_count = len(vehicles)

        annotated = frame.copy()
        for vehicle in vehicles:
            self._draw_box(annotated, vehicle["box"],
                           f'{vehicle["label"]} {vehicle["conf"]:.2f}', (0, 255, 0))
        for plate in plates:
            self._draw_box(annotated, plate["box"],
                           f'plate {plate["conf"]:.2f}', (0, 215, 255))

        for track in tracks:
            plate = self._match_plate(track.box, plates)
            if plate is not None:
                track.plate_box = plate["box"]
                track.plate_conf = plate["conf"]
            ready = track.plate_box is not None or track.age >= self.cfg.RECORD_GRACE_FRAMES
            if not track.recorded and ready:
                self._enqueue_record(frame, track, now)

        self._draw_hud(annotated, now)
        self._publish(annotated)

    # ---------------------------------------------------------------- records
    def _enqueue_record(self, frame, track, now):
        track.recorded = True
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.detector._clamp_frame(track.box, width, height)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return
        plate_bbox, plate_crop = None, None
        if track.plate_box is not None:
            px1, py1, px2, py2 = self.detector._clamp_frame(track.plate_box, width, height)
            if px2 - px1 >= 4 and py2 - py1 >= 4:
                plate_bbox = [px1, py1, px2, py2]
                plate_crop = frame[py1:py2, px1:px2].copy()

        job = {
            "record_id": f"{self._run_id}_t{track.id:04d}",
            "now": now,
            "vehicle_type": track.label,
            "vehicle_conf": track.conf,
            "vehicle_bbox": [x1, y1, x2, y2],
            "plate_bbox": plate_bbox,
            "vehicle_crop": frame[y1:y2, x1:x2].copy(),
            "plate_crop": plate_crop,
        }
        try:
            self._record_queue.put_nowait(job)
        except queue.Full:
            print("[pipeline] record queue full, dropping one record")

    def _recorder_loop(self):
        while self._running or not self._record_queue.empty():
            try:
                job = self._record_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write_record(job)
            except Exception as exc:
                print(f"[pipeline] record write error: {exc}")
            finally:
                self._record_queue.task_done()

    def _write_record(self, job):
        now = job["now"]
        plate_text, plate_conf = "", 0.0
        if job["plate_crop"] is not None:
            plate_text, plate_conf = self.ocr.read(job["plate_crop"])

        vehicle_path = self.store.save_image(job["vehicle_crop"], "vehicle", job["record_id"])
        plate_path = self.store.save_image(job["plate_crop"], "plate", job["record_id"])

        record = {
            "id": job["record_id"],
            "camera_number": self.cfg.CAMERA_NUMBER,
            "date": now.strftime("%d/%m/%Y"),
            "time": now.strftime("%H:%M:%S"),
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "vehicle_type": job["vehicle_type"],
            "vehicle_confidence": round(job["vehicle_conf"], 3),
            "plate_text": plate_text,
            "plate_confidence": round(plate_conf, 3),
            "vehicle_bbox": job["vehicle_bbox"],
            "plate_bbox": job["plate_bbox"],
            "vehicle_snapshot": vehicle_path,
            "plate_snapshot": plate_path,
            "timestamp": now.isoformat(timespec="seconds"),
        }
        self.store.add(record)

    # ------------------------------------------------------------- rendering
    @staticmethod
    def _center_inside(inner, outer):
        cx = (inner[0] + inner[2]) / 2.0
        cy = (inner[1] + inner[3]) / 2.0
        return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

    def _match_plate(self, vehicle_box, plates):
        best, best_conf = None, -1.0
        for plate in plates:
            if self._center_inside(plate["box"], vehicle_box) and plate["conf"] > best_conf:
                best, best_conf = plate, plate["conf"]
        return best

    @staticmethod
    def _draw_box(image, box, label, color):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_hud(self, image, now):
        lines = [
            f"CAM: {self.cfg.CAMERA_NUMBER}",
            f"DATE: {now.strftime('%d/%m/%Y')}",
            f"TIME: {now.strftime('%H:%M:%S')}",
            f"FPS: {self._fps:.1f}",
            f"VEHICLES: {self._vehicle_count}",
            f"RECORDS: {self.store.count()}",
        ]
        y = 24
        for line in lines:
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
            y += 26

    def _publish(self, image):
        ok, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.STREAM_JPEG_QUALITY]
        )
        if not ok:
            return
        with self._jpeg_cond:
            self._latest_jpeg = buffer.tobytes()
            self._publish_id += 1
            self._jpeg_cond.notify_all()

    # ------------------------------------------------------------------- api
    def wait_jpeg(self, last_id, timeout=2.0):
        with self._jpeg_cond:
            if self._publish_id == last_id and self._running:
                self._jpeg_cond.wait(timeout)
            return self._publish_id, self._latest_jpeg

    def stats(self):
        return {
            "camera_number": self.cfg.CAMERA_NUMBER,
            "rtsp_connected": self.stream.connected,
            "fps": round(self._fps, 1),
            "vehicles_in_frame": self._vehicle_count,
            "records": self.store.count(),
            "ocr_available": self.ocr.available,
            "models": {
                "vehicle": self.cfg.VEHICLE_MODEL,
                "plate": self.cfg.PLATE_MODEL,
            },
        }
