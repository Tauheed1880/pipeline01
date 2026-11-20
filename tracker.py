"""Minimal IoU tracker.

Assigns a stable integer track id to each vehicle so that a record (and its
snapshots) is written exactly once per vehicle -> idempotent JSON output.
"""


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, track_id, box, label, conf):
        self.id = track_id
        self.box = box
        self.label = label
        self.conf = conf
        self.age = 0
        self.misses = 0
        self.matched = True
        self.recorded = False
        self.plate_box = None
        self.plate_conf = 0.0


class IouTracker:
    def __init__(self, iou_threshold=0.3, max_missed=30):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks = {}
        self._next_id = 1

    def update(self, detections):
        """detections: list of (box, label, conf). Returns the active tracks."""
        for tr in self._tracks.values():
            tr.matched = False

        for box, label, conf in detections:
            best, best_iou = None, self.iou_threshold
            for tr in self._tracks.values():
                if tr.matched:
                    continue
                score = _iou(tr.box, box)
                if score >= best_iou:
                    best, best_iou = tr, score
            if best is not None:
                best.box = box
                best.label = label
                best.conf = conf
                best.misses = 0
                best.matched = True
            else:
                tr = Track(self._next_id, box, label, conf)
                self._tracks[self._next_id] = tr
                self._next_id += 1

        for tid in list(self._tracks):
            tr = self._tracks[tid]
            if not tr.matched:
                tr.misses += 1
                if tr.misses > self.max_missed:
                    del self._tracks[tid]

        for tr in self._tracks.values():
            tr.age += 1
        return list(self._tracks.values())
