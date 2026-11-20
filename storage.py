"""Persistence for detections.

- All records live in a single JSON file (a JSON array) so it can be imported
  straight into a database later.
- Each vehicle snapshot ("vehicle" + "plate" crops) is written next to it and
  its path is stored inside the record, so the saved image is the exact source
  of the recorded attributes.
- Records are de-duplicated by their unique id (camera + run + track id), so
  re-processing never writes the same vehicle twice.
"""
import json
import threading
from pathlib import Path

import cv2


class DetectionStore:
    def __init__(self, records_file, snapshot_dir, plate_dir):
        self.records_file = Path(records_file)
        self.snapshot_dir = Path(snapshot_dir)
        self.plate_dir = Path(plate_dir)
        self.records_file.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.plate_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._records = self._load()
        self._ids = {r.get("id") for r in self._records}

    def _load(self):
        if self.records_file.exists():
            try:
                with open(self.records_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _flush(self):
        tmp = self.records_file.with_suffix(self.records_file.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self._records, handle, ensure_ascii=False, indent=2)
        tmp.replace(self.records_file)

    def save_image(self, image, kind, record_id):
        if image is None or image.size == 0:
            return None
        folder = self.snapshot_dir if kind == "vehicle" else self.plate_dir
        path = folder / f"{record_id}.jpg"
        cv2.imwrite(str(path), image)
        return str(path)

    def add(self, record):
        """Append a record and flush to disk immediately. Idempotent by id."""
        with self._lock:
            if record.get("id") in self._ids:
                return False
            self._records.append(record)
            self._ids.add(record["id"])
            self._flush()
            return True

    def all(self, limit=None):
        with self._lock:
            if limit:
                return list(self._records[-limit:])
            return list(self._records)

    def count(self):
        with self._lock:
            return len(self._records)
