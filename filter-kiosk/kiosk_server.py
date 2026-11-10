"""Filter Station: the hallway kiosk.

This program runs on the Raspberry Pi. It:
  1. reads the webcam (one camera loop for all four filters),
  2. runs the filter that's switched on (see filter_adapters.py) on every frame,
  3. sends the finished video to the kiosk page in Chromium, and
  4. switches filters when someone taps or clicks a card on the page.

Privacy, built in:
  - It only listens on 127.0.0.1, which means only this Pi can open the page or the video.
    Nothing leaves the Pi. It never uses the internet.
  - No photo or video is ever written to disk. Each frame lives in memory until the next one
    replaces it. The only file anything writes is the team's 5-foot calibration (3 numbers).
  - If a filter ever crashes on a frame, that frame is shown fully blurred instead.

Run:   python3 kiosk_server.py                  (then open http://127.0.0.1:8000)
       python3 kiosk_server.py --video demo.mp4 (try it with a video file instead of a camera)
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import filter_adapters as fa
import previews

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
FRAME_W, FRAME_H = 640, 480      # every filter was built and calibrated at this size
JPEG_QUALITY = 80
START_FILTER = "goldilocks"


def to_640x480(frame):
    """Cameras that won't give 640x480 get cropped to the same 4:3 shape, then resized,
    so faces keep their real proportions and the calibration still fits."""
    h, w = frame.shape[:2]
    if (w, h) == (FRAME_W, FRAME_H):
        return frame
    if w * 3 > h * 4:                              # too wide: crop the sides
        nw = h * 4 // 3
        frame = frame[:, (w - nw) // 2:(w - nw) // 2 + nw]
    elif w * 3 < h * 4:                            # too tall: crop top and bottom
        nh = w * 3 // 4
        frame = frame[(h - nh) // 2:(h - nh) // 2 + nh]
    return cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)


class CameraSource:
    """Reads the webcam in its own thread and keeps only the newest frame, so the screen never
    falls behind. If the camera is unplugged, it keeps trying to reopen it."""

    def __init__(self, index=0):
        self.index = index
        self.frame, self.seq = None, 0
        self.cond = threading.Condition()
        self.cap = None
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        if sys.platform.startswith("linux"):
            backend = cv2.CAP_V4L2
        elif sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION
        elif sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        fails = 0
        while True:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open()
                if not self.cap.isOpened():
                    print("Camera not found; trying again in 2 seconds.")
                    time.sleep(2)
                    continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 30:                     # about 1 second of failures: reopen
                    print("Camera stopped sending frames; reopening it.")
                    self.cap.release()
                    self.cap = None
                    fails = 0
                    time.sleep(1)
                continue
            fails = 0
            frame = to_640x480(frame)
            with self.cond:
                self.frame, self.seq = frame, self.seq + 1
                self.cond.notify_all()

    def wait_frame(self, last_seq, timeout=1.0):
        """Wait for a frame newer than last_seq. Returns (seq, frame), or (last_seq, None)."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq != last_seq, timeout):
                return last_seq, None
            return self.seq, self.frame


class VideoFileSource(CameraSource):
    """Plays a video file on a loop instead of the camera (for trying the kiosk on a computer).
    The file is only read, never changed."""

    def __init__(self, path):
        self.path = path
        super().__init__(index=None)

    def _run(self):
        while True:
            cap = cv2.VideoCapture(self.path)
            delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = to_640x480(frame)
                with self.cond:
                    self.frame, self.seq = frame, self.seq + 1
                    self.cond.notify_all()
                time.sleep(delay)
            cap.release()


def privacy_fallback(frame):
    """Used only if a filter crashes: show the whole picture heavily blurred, never unfiltered."""
    small = cv2.resize(cv2.flip(frame, 1), (FRAME_W // 8, FRAME_H // 8), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)


class Kiosk:
    """One camera loop that runs whichever filter is switched on."""

    def __init__(self, source, start_filter=START_FILTER):
        fa.check_model()
        self.source = source
        self.calibration = fa.make_shared_calibration()
        self.adapters, self.errors = {}, {}
        for f in fa.FILTERS:
            try:
                self.adapters[f["id"]] = fa.ADAPTERS[f["id"]](self.calibration)
                print(f"Loaded {f['name']}.")
            except Exception as e:                 # one broken filter shouldn't stop the kiosk
                self.errors[f["id"]] = f"{type(e).__name__}: {e}"
                print(f"Could not load {f['name']}: {self.errors[f['id']]}")
        if not self.adapters:
            raise RuntimeError("No filters could be loaded.")
        self.active = start_filter if start_filter in self.adapters else next(iter(self.adapters))
        # Card pictures: a drawn mannequin wearing each filter's real artwork (made once, here)
        self.previews = previews.make_previews(self.adapters)
        self.lock = threading.Lock()
        self.requested = self.request_time = None
        self.calibrate_requested = False
        self.new_jpeg = threading.Condition()
        self.jpeg, self.jpeg_seq, self.shown = None, 0, None
        self.last_switch_ms = None
        self.engaged = False
        self.fps = 0.0
        self._last_step = None
        self.last_frame_time = 0.0
        self.crashes = 0
        self.running = True

    # ---- called by the web page (through the HTTP handler) ----
    def request_filter(self, filter_id):
        if filter_id not in self.adapters:
            return False
        with self.lock:                            # if taps come faster than frames, the last one wins
            self.requested, self.request_time = filter_id, time.monotonic()
        return True

    def request_calibration(self):
        with self.lock:
            self.calibrate_requested = True

    def state(self):
        cal = self.calibration
        message = cal.message.replace("then press C", "then tap Start calibration") \
                             .replace("press C to retry", "tap Start calibration to retry") \
                             .replace("and press C", "and tap Start calibration") \
                             .replace(" C: recalibrate", "").replace("  Q: quit", "")
        return dict(
            shown=self.shown, active=self.active, frame=self.jpeg_seq, fps=round(self.fps, 1),
            engaged=self.engaged, camera_ok=time.monotonic() - self.last_frame_time < 2.0,
            calibrated=cal.reference is not None and not cal.active, calibrating=cal.active,
            calibration_message=message, last_switch_ms=self.last_switch_ms, crashes=self.crashes,
            filters=[dict(id=f["id"], name=f["name"], tagline=f["tagline"],
                          available=f["id"] in self.adapters, error=self.errors.get(f["id"]))
                     for f in fa.FILTERS])

    # ---- the camera loop ----
    def run(self):
        seq = 0
        while self.running:
            seq, frame = self.source.wait_frame(seq, timeout=1.0)
            if frame is not None:
                self.step(frame)

    def step(self, frame):
        """Everything the kiosk does to one camera frame."""
        with self.lock:
            req, t_req = self.requested, self.request_time
            self.requested = None
            calibrate, self.calibrate_requested = self.calibrate_requested, False
        if calibrate:
            self.calibration.begin(time.monotonic())
            for adapter in self.adapters.values():
                adapter.reset()
        if req is not None and req != self.active:
            self.adapters[req].reset()             # start fresh: no tracking carried over
            self.active = req
        adapter = self.adapters[self.active]
        try:
            out = adapter.process(frame)
            engaged = adapter.engaged
        except Exception:
            traceback.print_exc()
            self.crashes += 1
            out, engaged = privacy_fallback(frame), False
            adapter.reset()
        ok, data = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        now = time.perf_counter()
        if self._last_step is not None:
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._last_step, 1e-3)
        self._last_step = now
        with self.new_jpeg:
            self.jpeg, self.jpeg_seq = data.tobytes(), self.jpeg_seq + 1
            self.shown, self.engaged = self.active, engaged
            self.last_frame_time = time.monotonic()
            if req is not None:
                self.last_switch_ms = round((time.monotonic() - t_req) * 1000)
            self.new_jpeg.notify_all()
        return out

    def wait_jpeg(self, last_seq, timeout=2.0):
        with self.new_jpeg:
            self.new_jpeg.wait_for(lambda: self.jpeg_seq != last_seq, timeout)
            return self.jpeg_seq, self.jpeg


TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
         ".ttf": "font/ttf", ".txt": "text/plain; charset=utf-8"}


def make_handler(kiosk):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):              # keep the console quiet
            pass

        def _send(self, code, body, ctype="application/json", cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                path = "/index.html"
            if path == "/stream.mjpg":
                return self._stream()
            if path == "/api/state":
                return self._json(200, kiosk.state())
            if path.startswith("/preview/") and path.endswith(".png"):
                data = kiosk.previews.get(path[len("/preview/"):-len(".png")])
                if data is None:
                    return self._send(404, b"Not found", "text/plain")
                return self._send(200, data, "image/png")
            # files from the web folder only (never anything outside it)
            full = os.path.realpath(os.path.join(WEB_DIR, path.lstrip("/")))
            if not full.startswith(os.path.realpath(WEB_DIR) + os.sep) or not os.path.isfile(full):
                return self._send(404, b"Not found", "text/plain")
            with open(full, "rb") as f:
                body = f.read()
            ext = os.path.splitext(full)[1]
            self._send(200, body, TYPES.get(ext, "application/octet-stream"), cache=ext == ".ttf")

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 10_000)
                data = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json(400, dict(error="bad request"))
            if path == "/api/filter":
                if kiosk.request_filter(str(data.get("id"))):
                    return self._json(202, dict(requested=data.get("id")))
                return self._json(404, dict(error="no such filter, or it isn't available"))
            if path == "/api/calibrate":
                kiosk.request_calibration()
                return self._json(202, dict(calibrating=True))
            return self._json(404, dict(error="not found"))

        def _stream(self):
            """Motion JPEG: one picture after another over one open connection."""
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            try:
                while kiosk.running:
                    seq, data = kiosk.wait_jpeg(seq)
                    if data is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Filter Station hallway kiosk")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--camera", type=int, default=0, help="camera number (0 = the first webcam)")
    ap.add_argument("--video", help="play this video file instead of using the camera")
    ap.add_argument("--start", default=START_FILTER, help="filter that's on when the kiosk starts")
    args = ap.parse_args()

    source = VideoFileSource(args.video) if args.video else CameraSource(args.camera)
    kiosk = Kiosk(source, args.start)
    threading.Thread(target=kiosk.run, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(kiosk))
    server.daemon_threads = True
    print(f"Filter Station is running. Open http://127.0.0.1:{args.port} on this computer.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        kiosk.running = False


if __name__ == "__main__":
    main()
