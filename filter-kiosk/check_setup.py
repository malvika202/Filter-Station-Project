"""Checks the kiosk is ready, in plain words. Run on the Pi:  .venv/bin/python check_setup.py
Add --no-camera to skip the camera check. Changes nothing; it only looks."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
problems = []


def ok(msg):
    print("  OK       " + msg)


def bad(msg):
    print("  PROBLEM  " + msg)
    problems.append(msg)


def note(msg):
    print("  NOTE     " + msg)


print("Filter Station setup check")
v = sys.version_info
(ok if (3, 9) <= (v.major, v.minor) <= (3, 12) else bad)(f"Python {v.major}.{v.minor} (needs 3.9 to 3.12)")

try:
    import numpy
    import cv2
    major, minor = (int(p) for p in cv2.__version__.split(".")[:2])
    (ok if (major, minor) >= (4, 8) else bad)(f"OpenCV {cv2.__version__} (needs 4.8 or newer), numpy {numpy.__version__}")
except Exception as e:
    bad(f"OpenCV or numpy won't load: {e}")
    cv2 = None

try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        ok(f"mediapipe {mp.__version__} has FaceMesh (Adoggy needs it)")
    else:
        bad(f"mediapipe {mp.__version__} has no FaceMesh. Run: .venv/bin/python -m pip install mediapipe==0.10.18")
except Exception as e:
    bad(f"mediapipe won't load, so Adoggy will be off: {e}")

import filter_adapters as fa
try:
    fa.check_model()
    ok("face model is in the filters folder")
except RuntimeError as e:
    bad(str(e))

for name in ["smile.png", "clown_nose.png", "dog visor.png"]:
    path = os.path.join(fa.FILTERS_DIR, name)
    if not os.path.exists(path):
        bad(f"{name} is missing from the filters folder")
        continue
    with open(path, "rb") as f:
        standin = b"standin" in f.read()
    (note if standin else ok)(f"{name}" + (" is still the STAND-IN picture. Copy the real one in with the same name." if standin else " is the real picture"))

for f in ["index.html", "kiosk.css", "kiosk.js"]:
    if not os.path.exists(os.path.join(HERE, "web", f)):
        bad(f"the page file web/{f} is missing")

if cv2 is not None:
    try:
        cal = fa.make_shared_calibration()
        for fl in fa.FILTERS:
            try:
                fa.ADAPTERS[fl["id"]](cal)
                ok(f"{fl['name']} loads")
            except Exception as e:
                bad(f"{fl['name']} won't load: {e}")
        if cal.reference is None:
            note("not calibrated yet. Every face stays blurred until a staff member calibrates (see README).")
        else:
            ok("5-foot calibration is saved")
    except Exception as e:
        bad(f"the filters won't load: {e}")

if cv2 is not None and "--no-camera" not in sys.argv:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    got, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    if got:
        h, w = frame.shape[:2]
        (ok if (w, h) == (640, 480) else note)(f"camera works ({w}x{h})" + ("" if (w, h) == (640, 480) else ", the kiosk crops and resizes it to 640x480"))
    else:
        bad("no picture from camera 0. Check the USB cable, or close anything else using the camera.")

print()
print("All good." if not problems else f"{len(problems)} problem(s) to fix (see above).")
sys.exit(1 if problems else 0)
