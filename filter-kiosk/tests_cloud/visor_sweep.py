# Same face at different distances: does The Visor show? v1 (Kate's original) vs v2 (zoom-in fix).
import sys, numpy as np, cv2, time
sys.path.insert(0, "/home/claude/tests")
import kiosk_common as kc
K = kc.scratch_kiosk(); kc.write_calibration(K)      # face A with a 22 px eye gap = 4 ft
import filter_adapters as fa
cal = fa.make_shared_calibration()
v2 = fa.VisorAdapter(cal)
v1m = fa.load_module("dog_visor_update.py")            # Kate's original, run like its main() loop
import mediapipe as mp
fm1 = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
det1 = cv2.FaceDetectorYN.create(v1m.MODEL_PATH, "", (640, 480), score_threshold=0.7)
def v1_engaged(frame, crit):
    frame = cv2.flip(frame, 1); _, faces = det1.detect(frame); now = time.monotonic()
    crit.update(faces, frame.shape, now)
    meshes = fm1.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks or []
    detected = [(k, t['face'], t['valid']) for k, t in crit.tracks.items() if t['time'] == now]
    m = v1m.match_meshes(meshes, detected, frame.shape)
    return any(x is not None and x[2] for x in m.values())
print("eye gap (about feet) | v1 visor shown | v2 visor shown | v2 ms/frame")
for person in ("A", "B"):
    for gap in (19, 20, 22, 24, 26, 30, 40):
        c = np.full((480, 640, 3), (96, 92, 88), np.uint8); kc.pt.paste(c, person, (320, 200), gap); raw = cv2.flip(c, 1)
        crit = v1m.FaceCriteria(cal); v2.reset(); e1 = e2 = 0; ms = []
        for k in range(8):
            e1 += v1_engaged(raw.copy(), crit)
            t = time.perf_counter(); v2.process(raw.copy()); ms.append((time.perf_counter() - t) * 1000); e2 += v2.engaged
        print(f"  {person} {gap}px (~{4 * 22 / gap:.1f} ft) | {e1}/8 | {e2}/8 | {np.median(ms):.0f}")
