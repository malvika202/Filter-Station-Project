# Frame-exact switching test on the real Kiosk class (fake camera = the hallway scene).
# AC5: many switches, including bursts of fast taps and tapping the filter that's already on:
#      the next frame always uses the LAST filter tapped, and nothing crashes.
# AC7: in EVERY frame, during and after switches, every far face the detector finds is blurred.
# Also: a filter that crashes produces a fully blurred frame, never an unfiltered one.
import random, sys, time
import numpy as np, cv2
sys.path.insert(0, "/home/claude/tests")
import kiosk_common as kc

K = kc.scratch_kiosk()
kc.write_calibration(K)
import kiosk_server as ks
import filter_adapters as fa

kiosk = ks.Kiosk(source=None)
ids = [f["id"] for f in fa.FILTERS]
det = cv2.FaceDetectorYN.create(fa.load_module(fa.file_for("smile")).MODEL_PATH, "", (640, 480), score_threshold=0.7)
rng = random.Random(4)
N = 300
wrong_filter = 0; far_checked = 0; far_sharp = []; switches = 0; burst_ok = 0; bursts = 0
for k in range(N):
    raw = kc.raw_scene(k % 112)
    expect = None
    if k % 4 == 0:                                   # a tap every 4 frames
        if rng.random() < 0.25:                      # a burst: three fast taps before the next frame
            taps = [rng.choice(ids) for _ in range(3)]; bursts += 1
        elif rng.random() < 0.2:
            taps = [kiosk.active]                    # tapping the card that's already on
        else:
            taps = [rng.choice(ids)]
        for t in taps:
            kiosk.request_filter(t)
        expect = taps[-1]; switches += len(taps)
    out = kiosk.step(raw)
    if expect is not None:
        ok = kiosk.shown == expect
        wrong_filter += not ok
        if len(taps) == 3:
            burst_ok += ok
    # privacy: every far face the detector finds must be blurred in this frame
    mirrored = cv2.flip(raw, 1); w = kc.who(k % 112)
    _, faces = det.detect(mirrored)
    for r in (faces if faces is not None else []):
        lab = min(w, key=lambda L: np.hypot(w[L][0] - r[8], w[L][1] - r[9]))
        if np.hypot(w[lab][0] - r[8], w[lab][1] - r[9]) > 1.5 * w[lab][2] or w[lab][2] >= 16:
            continue
        x, y, bw, bh = r[:4]; cx, cy, rad = x + bw / 2, y + bh / 2, 0.4 * min(bw, bh)
        far_sharp.append(kc.sharpness(out, cx, cy, rad) / max(kc.sharpness(mirrored, cx, cy, rad), 1e-6))
        far_checked += 1

print(f"taps sent: {switches} ({bursts} bursts of 3 fast taps) over {N} frames")
print(f"frames where the filter on screen wasn't the last one tapped: {wrong_filter}")
print(f"bursts that ended on the last tap: {burst_ok}/{bursts}")
print(f"far faces checked: {far_checked}, least blurred (sharpness after/before): {max(far_sharp):.2f}")
print(f"crashes: {kiosk.crashes}")

# Crash fallback: make the smile filter crash, then check that frame is fully blurred
kiosk.request_filter("smile"); kiosk.step(kc.raw_scene(30))
real = kiosk.adapters["smile"].process
kiosk.adapters["smile"].process = lambda f: (_ for _ in ()).throw(RuntimeError("test crash"))
raw = kc.raw_scene(30); out = kiosk.step(raw)
kiosk.adapters["smile"].process = real
whole = kc.sharpness(out, 320, 240, 230) / kc.sharpness(cv2.flip(raw, 1), 320, 240, 230)
print(f"crash frame: whole-picture sharpness after/before {whole:.3f}, crashes counted {kiosk.crashes}")
after = kiosk.step(kc.raw_scene(31))
print(f"next frame after the crash works normally: {after.shape == (480, 640, 3) and kiosk.shown == 'smile'}")

ok = wrong_filter == 0 and burst_ok == bursts and max(far_sharp) < 0.6 and whole < 0.1 and kiosk.crashes == 1
print("SWITCHING TEST:", "PASS" if ok else "FAIL")
