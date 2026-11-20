"""Low-latency RTSP frame reader.

A dedicated background thread reads the RTSP stream as fast as it can and keeps
ONLY the most recent frame. Consumers always get the latest frame, so OpenCV's
internal buffer can never build up latency. If the connection drops it is
re-opened automatically.
"""
import os
import threading
import time

import cv2


def _configure_ffmpeg(url):
    """Set FFmpeg capture options that match the URL scheme.

    RTSP gets transport/low-delay flags. HTTP(S)/HLS gets NO extra flags: the
    low-delay flags make this HLS server fail to load the first segment, and
    HLS latency is already handled by the buffer-size + latest-frame reader.
    """
    if url.lower().startswith("rtsp"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
        )
    else:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)


class RTSPStream:
    def __init__(self, url, reconnect_delay=2.0):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._cap = None
        self._connected = False

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, name="rtsp-reader", daemon=True)
        self._thread.start()
        return self

    def _open(self):
        _configure_ffmpeg(self.url)
        backend = cv2.CAP_ANY
        if self.url.lower().startswith(("rtsp", "http://", "https://")):
            backend = cv2.CAP_FFMPEG
        cap = cv2.VideoCapture(self.url, backend)
        # Keep the transport buffer tiny so we always get the newest frame.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _reader_loop(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._connected = False
                self._cap = self._open()
                if not self._cap.isOpened():
                    time.sleep(self.reconnect_delay)
                    continue
                self._connected = True

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._cap.release()
                self._cap = None
                self._connected = False
                time.sleep(self.reconnect_delay)
                continue

            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self):
        """Return (frame, frame_id) for the latest frame, or None."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy(), self._frame_id

    @property
    def connected(self):
        return self._connected

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
