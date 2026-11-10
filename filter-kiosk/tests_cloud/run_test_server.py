# Starts the REAL kiosk program, but with a fake camera that plays the hallway scene on a loop.
# Usage: python run_test_server.py PORT [calibrated|uncalibrated] [crash]
import sys, threading, time
sys.path.insert(0, "/home/claude/tests")
import kiosk_common as kc

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
mode = sys.argv[2] if len(sys.argv) > 2 else "calibrated"
K = kc.scratch_kiosk()
if mode == "calibrated":
    kc.write_calibration(K)
import kiosk_server as ks
from http.server import ThreadingHTTPServer

FRAMES = [kc.raw_scene(k) for k in range(int(7.5 * kc.pt.FPS))]


class FakeCamera(ks.CameraSource):
    """Plays the scene at 15 frames per second, like a slow webcam."""
    def __init__(self):
        super().__init__(index=None)

    def _run(self):
        k = 0
        while True:
            frame = FRAMES[k % len(FRAMES)]
            with self.cond:
                self.frame, self.seq = frame, self.seq + 1
                self.cond.notify_all()
            k += 1
            time.sleep(1 / 15)


kiosk = ks.Kiosk(FakeCamera())
if "crash" in sys.argv:            # make one filter throw an error on every 10th frame
    real = kiosk.adapters["smile"].process
    count = [0]
    def flaky(frame):
        count[0] += 1
        if count[0] % 10 == 0:
            raise RuntimeError("test crash")
        return real(frame)
    kiosk.adapters["smile"].process = flaky
threading.Thread(target=kiosk.run, daemon=True).start()
server = ThreadingHTTPServer(("127.0.0.1", port), ks.make_handler(kiosk))
server.daemon_threads = True
print("TEST SERVER READY", port, flush=True)
server.serve_forever()
