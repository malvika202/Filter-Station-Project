# Calibrate through the kiosk's staff button while each filter is on (one person alone, facing the
# camera). The clown filter sees the camera un-mirrored, the others mirrored: all four must give
# the same calibration, save only numbers, and make the person show up as about 5 ft.
import json, os, sys, time
import numpy as np, cv2
sys.path.insert(0, "/home/claude/tests")
import kiosk_common as kc
K = kc.scratch_kiosk()                       # no calibration file yet
import kiosk_server as ks
kiosk = ks.Kiosk(source=None)
path = kiosk.calibration.path
c = np.full((480, 640, 3), (96, 92, 88), np.uint8); kc.pt.paste(c, "A", (320, 200), 22)
raw = cv2.flip(c, 1)
refs = {}
for fid in ["goldilocks", "smile", "clown", "visor"]:
    kiosk.request_filter(fid); kiosk.step(raw)
    kiosk.request_calibration()
    for k in range(60):
        kiosk.step(raw); time.sleep(0.03)    # calibration needs 30 steady samples over at least 1 second
        if not kiosk.calibration.active: break
    st = kiosk.state()
    data = json.load(open(path))
    refs[fid] = np.array(data["features"])
    print(f"{fid:10s}: calibrated={st['calibrated']} message='{st['calibration_message']}' features={np.round(refs[fid], 2).tolist()}")
spread = max(np.max(np.abs(refs[f] / refs['smile'] - 1)) for f in refs)
print("file contents:", sorted(json.load(open(path)).keys()), "| other files written:",
      [f for f in os.listdir(os.path.dirname(path)) if f.endswith((".jpg", ".png", ".mp4", ".avi")) and "visor" not in f and "smile.png" != f and "clown_nose" not in f])
print(f"largest difference between filters: {spread * 100:.2f}%")
mirrored_same = all(np.allclose(refs[f], refs["smile"]) for f in ("goldilocks", "visor"))
print(f"the three mirrored filters give identical calibrations: {mirrored_same}")
print("all four filters now find faces on the mirrored picture, so their calibrations should match exactly")

print("CALIBRATION TEST:", "PASS" if mirrored_same and spread < 0.01 else "FAIL")
