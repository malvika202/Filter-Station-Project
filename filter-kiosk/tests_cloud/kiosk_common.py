# Shared test helpers: a scratch copy of the kiosk with a test calibration, and the hallway scene.
import json, os, shutil, sys, tempfile, math
import numpy as np, cv2

sys.path.insert(0, "/home/claude/hairtests")
import picktest as pt   # the kit's hallway scene (public test photos): A on the spot, B, a poster

W = 640


def scratch_kiosk():
    """Copy the kiosk to a scratch folder so test files never end up in the real one."""
    d = tempfile.mkdtemp(prefix="kiosk_test_")
    shutil.copytree("/home/claude/kiosk", os.path.join(d, "kiosk"),
                    ignore=shutil.ignore_patterns("__pycache__", "smile_distance_calibration.json", "web"))
    k = os.path.join(d, "kiosk")
    os.symlink("/home/claude/kiosk/web", os.path.join(k, "web"))   # page edits show up without a restart
    sys.path.insert(0, k)
    return k


def raw_scene(k):
    """Frame k as the camera would see it (NOT mirrored)."""
    c = np.full((480, 640, 3), (96, 92, 88), np.uint8)
    for lab, xy, g in pt.scene(k):
        pt.paste(c, lab, xy, g)
    # the kit's scene is drawn as the mirrored view, so flip it to get the raw camera frame
    return cv2.flip(c, 1)


def who(k):
    """Where each person is in the MIRRORED (on-screen) view: label -> (x, y, eye gap)."""
    return {lab: (xy[0], xy[1], g) for lab, xy, g in pt.scene(k)}


def write_calibration(kiosk_dir, feet_for_A=4.0):
    """Calibrate as if person A (eye gap 22 px) stands at feet_for_A feet."""
    import filter_adapters as fa
    sm = fa.load_module(fa.file_for("smile"))
    c = np.full((480, 640, 3), (96, 92, 88), np.uint8); pt.paste(c, "A", (320, 190), 22)
    det = cv2.FaceDetectorYN.create(sm.MODEL_PATH, "", (640, 480), score_threshold=0.7)
    _, f = det.detect(c)
    yaw = sm.estimate_yaw(f[0], c.shape)
    feat = sm.distance_features(f[0], yaw, c.shape) * (feet_for_A / 5.0)
    path = os.path.join(kiosk_dir, "filters", "smile_distance_calibration.json")
    json.dump(dict(version=1, reference_feet=5.0, frame_size=[640, 480], features=feat.tolist()), open(path, "w"))
    return path


def sharpness(img, cx, cy, r):
    x0, y0 = int(max(cx - r, 0)), int(max(cy - r, 0))
    patch = img[y0:int(cy + r), x0:int(cx + r)]
    return cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
