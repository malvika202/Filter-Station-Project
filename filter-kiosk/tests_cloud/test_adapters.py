# Runs the hallway scene through each of the four filters inside the kiosk, with a calibration.
# Checks: the person on the spot gets the filter; the far poster is blurred in every frame, and
# so is every far face the detector finds; output is a mirrored 640x480 frame; speed per frame.
import time, sys
import numpy as np, cv2
sys.path.insert(0, "/home/claude/tests")
import kiosk_common as kc

K = kc.scratch_kiosk()
kc.write_calibration(K)
import filter_adapters as fa

cal = fa.make_shared_calibration()
assert cal.reference is not None
sheet, ok_all = [], True
for f in fa.FILTERS:
    a = fa.ADAPTERS[f["id"]](cal)
    a.reset()
    engaged, times, poster, far_b, shape_ok = 0, [], [], [], True
    thr = {"goldilocks": 0.8, "clown": 0.8}.get(f["id"], 0.7)      # each filter's own detector setting
    det = cv2.FaceDetectorYN.create(fa.load_module(fa.file_for("smile")).MODEL_PATH, "", (640, 480), score_threshold=thr)
    
    for k in range(int(7.5 * kc.pt.FPS)):
        raw = kc.raw_scene(k)
        t = time.perf_counter(); out = a.process(raw.copy()); times.append((time.perf_counter() - t) * 1000)
        shape_ok &= out.shape == (480, 640, 3) and out.dtype == np.uint8
        engaged += a.engaged
        w = kc.who(k); mirrored_in = cv2.flip(raw, 1)
        px, py, pg = w["P"]
        poster.append(kc.sharpness(out, px, py, 2 * pg) / max(kc.sharpness(mirrored_in, px, py, 2 * pg), 1e-6))
        # Every FAR face the detector finds (true eye gap under 16 px, about 5.5 ft or more) must come out
        # blurred: measure the sharpness inside that face's own box, before vs after.
        _, faces = det.detect(mirrored_in)
        for r in (faces if faces is not None else []):
            # who is this? the scene says where everyone really is and how far (their true eye gap)
            lab = min(w, key=lambda L: np.hypot(w[L][0] - r[8], w[L][1] - r[9]))
            if np.hypot(w[lab][0] - r[8], w[lab][1] - r[9]) > 1.5 * w[lab][2] or w[lab][2] >= 16:
                continue        # not clearly a far person
            x, y, bw, bh = r[:4]
            cx, cy, rad = x + bw / 2, y + bh / 2, 0.4 * min(bw, bh)
            far_b.append(kc.sharpness(out, cx, cy, rad) / max(kc.sharpness(mirrored_in, cx, cy, rad), 1e-6))
        if k == 30:
            sheet.append(cv2.putText(out.copy(), f["name"], (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2))
    worst_poster = max(poster); worst_b = max(far_b) if far_b else 1
    ok = shape_ok and engaged > 40 and worst_poster < 0.6 and worst_b < 0.6
    ok_all &= ok
    print(f"{f['name']:11s} filter on someone in {engaged:3d}/112 frames | poster blur worst {worst_poster:.2f} "
          f"| far faces found {len(far_b)}, least blurred {worst_b:.2f} | {np.median(times):5.1f} ms/frame (median) | {'PASS' if ok else 'FAIL'}")
cv2.imwrite("/home/claude/results/adapters_sheet.png", np.vstack([np.hstack(sheet[:2]), np.hstack(sheet[2:])]))
print("ADAPTER TEST:", "PASS" if ok_all else "FAIL")
