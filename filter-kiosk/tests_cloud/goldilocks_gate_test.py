# Tests the team rules inside Goldilocks v3 with a calibration, using the kit's hallway scene.
# A stands on the spot (about 4 ft), B walks past far away then steps in close, a poster hangs
# far away. Checks: hair only on faces that passed the team checks that frame, far faces blurred,
# and a simulated head turn past 50 degrees removes the hair and blurs the face.
import sys, json, math, os, tempfile
import numpy as np, cv2
sys.path.insert(0, "/home/claude/hair"); sys.path.insert(0, "/home/claude/hairtests")
import hair_filter as hf
hf.MODEL_PATH = "/home/claude/halo/face_detection_yunet_2023mar.onnx"
import picktest as pt

W = 640
def mirrored_scene(k):
    c = np.full((480, 640, 3), (96, 92, 88), np.uint8)
    for lab, xy, g in pt.scene(k): pt.paste(c, lab, xy, g)
    return cv2.flip(c, 1)

det = hf.make_detector()
# Calibration: measure A alone in the mirrored frame, then say A stands at 4 ft.
c = np.full((480, 640, 3), (96, 92, 88), np.uint8); pt.paste(c, "A", (320, 190), 22); c = cv2.flip(c, 1)
det.setInputSize((W, 480)); _, f = det.detect(c)
yaw = hf.estimate_yaw(f[0], c.shape); feat = hf.distance_features(f[0], yaw, c.shape)
tmp = tempfile.mkdtemp(); cal_path = os.path.join(tmp, "smile_distance_calibration.json")
json.dump(dict(version=1, reference_feet=5.0, frame_size=[640, 480], features=(feat * 0.8).tolist()), open(cal_path, "w"))
cal = hf.DistanceCalibration(path=cal_path)
assert cal.reference is not None, "calibration did not load"

g = hf.GoldilocksFilter(calibration=cal, detector=det)
renders = []
orig_render = hf.HairState.render
def spy(self, frame, pose, alpha, *a, **k):
    renders.append((K, pose["x"], pose["y"]))
    return orig_render(self, frame, pose, alpha, *a, **k)
hf.HairState.render = spy

TURN = range(15, 26)            # frames where A's head is "turned 70 degrees"
orig_yaw = hf.estimate_yaw
def turned_yaw(face, shape):
    y = orig_yaw(face, shape)
    nose_x = face[8]
    if K in TURN and y is not None and abs(nose_x - (W - 1 - 320)) < 30:
        return 70.0
    return y
hf.estimate_yaw = turned_yaw

def sharpness(img, cx, cy, r):
    x0, y0 = int(max(cx - r, 0)), int(max(cy - r, 0))
    patch = img[y0:int(cy + r), x0:int(cx + r)]
    return cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

LABELS = []; problems = []; hair_frames = {"A": 0, "B": 0}; poster_sharp = []; poster_orig = []; turn_blur = []
N = int(7.5 * pt.FPS)
for K in range(N):
    frame = mirrored_scene(K); before = frame.copy()
    out = g.process(frame, show_status=False)
    who = {lab: (W - 1 - xy[0], xy[1], gap) for lab, xy, gap in pt.scene(K)}
    now_tracks = [t for t in g.criteria.tracks.values()]
    for (k, x, y) in [r for r in renders if r[0] == K]:
        # who is under the hair? pose x,y is the point between the eyes
        lab = min(who, key=lambda L: math.hypot(who[L][0] - x, who[L][1] - who[L][2] * 0.52 - y))
        if lab == "P" or (lab == "B" and who["B"][2] < 20):
            problems.append(f"frame {K}: hair on far face {lab}")
        if K in TURN and lab == "A":
            problems.append(f"frame {K}: hair on A while turned past 50 degrees")
        T = g.picker.target
        tr = g.criteria.tracks.get(T.criteria_id) if T is not None else None
        if tr is None or not tr["valid"]:
            problems.append(f"frame {K}: hair drawn but the face failed the team checks")
        hair_frames[lab] = hair_frames.get(lab, 0) + 1
        LABELS.append((K, lab))
    px, py, pg = who["P"]
    poster_sharp.append(sharpness(out, px, py, 2 * pg)); poster_orig.append(sharpness(before, px, py, 2 * pg))
    if K in TURN and "A" in who:
        ax, ay, ag = who["A"]
        turn_blur.append(sharpness(out, ax, ay, 1.5 * ag) / max(sharpness(before, ax, ay, 1.5 * ag), 1e-6))

ratio = np.median(np.array(poster_sharp) / np.maximum(poster_orig, 1e-6))
print(f"frames: {N}   hair frames on A: {hair_frames['A']}   on B (after stepping close): {hair_frames['B']}")
print(f"poster sharpness after/before (lower = more blurred): {ratio:.2f}")
print(f"A's face sharpness while turned past 50 deg, after/before: {np.median(turn_blur):.2f}")
first_back = min((k for k, lab in LABELS if k > TURN[-1] and lab == "A"), default=None)
print(f"hair back on A at frame {first_back} (turn ended at frame {TURN[-1]})")
print("problems:", len(problems)); [print("  ", p) for p in problems[:10]]
ok = (not problems and hair_frames["A"] > 30 and hair_frames["B"] > 15 and ratio < 0.5
      and np.median(turn_blur) < 0.5 and first_back is not None and first_back - TURN[-1] <= 6)
print("GATE TEST:", "PASS" if ok else "FAIL")
