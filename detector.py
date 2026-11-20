"""Two-stage YOLO detector.

Stage 1: detect vehicles (car / bus / truck / motorcycle) on the full frame.
Stage 2: detect the number plate inside each vehicle ROI, then map the plate
boxes back to full-frame coordinates.

Both models are expected as .pt files inside the models/ folder.
"""
import os

from ultralytics import YOLO


class VehiclePlateDetector:
    def __init__(self, vehicle_model_path, plate_model_path=None, imgsz=640,
                 vehicle_conf=0.4, plate_conf=0.3, vehicle_class_ids=None, device=None):
        self.imgsz = imgsz
        self.vehicle_conf = vehicle_conf
        self.plate_conf = plate_conf
        self.vehicle_class_ids = vehicle_class_ids
        self.device = device

        self.vehicle_model = self._load(vehicle_model_path, required=True)
        self.plate_model = self._load(plate_model_path, required=False)

    @staticmethod
    def _load(path, required):
        if not path or not os.path.exists(path):
            message = f"model file not found: {path}"
            if required:
                raise FileNotFoundError(message)
            print(f"[detector] WARNING: {message} -> plate detection disabled")
            return None
        return YOLO(path)

    @staticmethod
    def _clamp_frame(box, width, height):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        return x1, y1, x2, y2

    def detect_vehicles(self, frame):
        """Return [{'box': [x1,y1,x2,y2], 'label': str, 'conf': float}, ...]."""
        results = self.vehicle_model.predict(
            frame, imgsz=self.imgsz, conf=self.vehicle_conf,
            device=self.device, verbose=False,
        )
        result = results[0]
        names = result.names
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if self.vehicle_class_ids is not None and cls_id not in self.vehicle_class_ids:
                continue
            detections.append({
                "box": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                "label": names.get(cls_id, str(cls_id)),
                "conf": float(box.conf[0]),
            })
        return detections

    def detect_plates(self, frame, vehicle_boxes):
        """Run the plate model on each vehicle crop; return full-frame plate boxes."""
        if self.plate_model is None or not vehicle_boxes:
            return []
        height, width = frame.shape[:2]
        crops, offsets = [], []
        for box in vehicle_boxes:
            x1, y1, x2, y2 = self._clamp_frame(box, width, height)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crops.append(frame[y1:y2, x1:x2])
            offsets.append((x1, y1))
        if not crops:
            return []

        results = self.plate_model.predict(
            crops, imgsz=self.imgsz, conf=self.plate_conf,
            device=self.device, verbose=False,
        )
        plates = []
        for result, (ox, oy) in zip(results, offsets):
            for box in result.boxes:
                px1, py1, px2, py2 = box.xyxy[0].tolist()
                plates.append({
                    "box": [round(px1 + ox, 1), round(py1 + oy, 1),
                            round(px2 + ox, 1), round(py2 + oy, 1)],
                    "conf": float(box.conf[0]),
                })
        return plates
