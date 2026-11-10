"""
Goldilocks: a face filter that gives you long, wavy, pale-gold hair with a light glow and drifting sparkles, drawn on
the person standing on the opt-in spot. Uses OpenCV's YuNet face detector.

How it works, in short:
  1. YuNet finds faces and five points on each one (two eyes, nose tip, two mouth corners).
  2. The team rules come first (the same code as the team's other filters): every face is
     measured against the saved 5-foot calibration. Faces farther than about 5 ft, or that can't
     be measured, are blurred. Only a face inside the range and turned less than 50 degrees can
     get hair, and only on frames where it passes those checks.
     Then only the opt-in person gets the filter: the biggest of those faces that is inside the
     middle part of the screen.
  3. From the five points we work out the head's 3D pose: where it is, how big it is, and how
     it is turned (left/right), tipped (up/down) and tilted (sideways).
  4. The hair is a 3D model of about a thousand strands grouped into locks. The locks start at
     the part, lie close to the head, and hang in soft waves. Every frame we turn the model to
     match the head, let the loose ends swing like real hair, and light every strand.
  5. The strands are drawn into see-through layers at double size and then shrunk, so each
     strand ends up thinner than a pixel and the hair builds up the way real hair looks on
     camera. A solid cap in a shadowed blonde sits under the strands, so the person's own hair
     never shows through. Things further from the camera are drawn smaller, like a real camera
     sees them (this matters a lot up close). The back layer goes behind the face, the front
     layer frames it, and the front hair casts a soft shadow on the face.
  6. The hair follows the room's brightness and color, gets a little camera grain so it doesn't
     look pasted on, and a light glow and floating gold sparkles go on top.
  7. The eyes, nose and mouth are always kept clear of hair, glow and sparkles, using where the
     detector actually found them (so it still works on a turned head or a profile).

Needs in the same folder as this script:
  - the YuNet model file (downloaded automatically the first time)
  - smile_distance_calibration.json, the team's shared 5-foot calibration (press C to make it).
    It holds three face-size measurements, not photos. Until it exists, every face is blurred.

Keys:
  q  quit
  d  toggle the debug view (face points, head pose, and who the filter is locked on)
  c  recalibrate alone, facing forward with your face 5 feet from the camera
"""
import json
import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

FILTER_NAME = "Goldilocks"
WINDOW_NAME = FILTER_NAME       # the title on the display window
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")
# media.githubusercontent.com serves the real file. The plain github.com "raw" link can hand
# back a 378-byte placeholder instead of the model.
MODEL_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")

# =====================================  TEAM RULES  =====================================
# Copied word for word from the team's shared code (the same in every filter), so all four
# filters measure distance, blur, and check head turn the same way.
CALIBRATION_PATH = os.path.join(SCRIPT_DIR, "smile_distance_calibration.json")

RANGE_FEET = 5.0
ENTER_CLEAR_FEET = 4.7  # Re-enter slightly closer to prevent boundary flicker.
CALIBRATION_SAMPLES = 30
DISTANCE_SMOOTH_SECONDS = 0.18
BLUR_PADDING = 0.18


MAX_YAW_DEGREES = 50.0
# Optional calibration for the UNMIRRORED webcam at its actual capture resolution.
# Set both from cv2.calibrateCamera: a 3x3 camera matrix and distortion vector.
# If unavailable, use approximate intrinsics; the cutoff is then approximate,
# not a measurement guaranteed to be accurate to a physical degree.
CAMERA_MATRIX = None
DIST_COEFFS = None

# Generic face in millimetres: screen-left eye, screen-right eye, nose,
# screen-left mouth corner, screen-right mouth corner in a frontal view.
# Camera convention: x right, y down, z away; nose projects toward camera.
# Individual facial proportions introduce uncertainty even with calibration.
POSE_POINTS = np.array([
    [-32.0, -32.0, 0.0], [32.0, -32.0, 0.0],
    [0.0, 0.0, -30.0],
    [-25.0, 32.0, -5.0], [25.0, 32.0, -5.0],
], dtype=np.float64)


def estimate_yaw(face, frame_shape):
    """Return yaw in degrees, or None for a failed/unreliable five-point fit.

    YuNet runs on the mirrored display frame. Undo the reflection and exchange
    each left/right pair so real camera calibration remains valid. Do not sort
    by x: landmark identity must survive head roll.
    """
    h, w = frame_shape[:2]
    points = np.asarray(face[4:14], dtype=np.float64).reshape(5, 2).copy()
    points[:, 0] = w - 1 - points[:, 0]
    points = np.ascontiguousarray(points[[1, 0, 2, 4, 3]])
    if not np.isfinite(points).all():
        return None
    if ((points < 0).any() or (points[:, 0] >= w).any()
            or (points[:, 1] >= h).any()):
        return None
    eye_span = np.linalg.norm(points[1] - points[0])
    if eye_span < 12.0:  # Too few pixels for a useful pose estimate.
        return None
    if CAMERA_MATRIX is None:
        focal = float(max(w, h))
        camera = np.array([[focal, 0, (w - 1) / 2],
                           [0, focal, (h - 1) / 2], [0, 0, 1]],
                          dtype=np.float64)
        distortion = np.zeros((5, 1), dtype=np.float64)
    else:
        camera = np.asarray(CAMERA_MATRIX, dtype=np.float64)
        distortion = (np.zeros((5, 1), dtype=np.float64) if DIST_COEFFS is None
                      else np.asarray(DIST_COEFFS, dtype=np.float64))
    try:
        ok, rvec, tvec = cv2.solvePnP(
            POSE_POINTS, points, camera, distortion,
            flags=cv2.SOLVEPNP_SQPNP)
        if not ok or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
            return None
        rvec, tvec = cv2.solvePnPRefineLM(
            POSE_POINTS, points, camera, distortion, rvec, tvec)
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = (rotation @ POSE_POINTS.T + tvec.reshape(3, 1)).T
        if not np.isfinite(camera_points).all() or (camera_points[:, 2] <= 0).any():
            return None
        projected, _ = cv2.projectPoints(
            POSE_POINTS, rvec, tvec, camera, distortion)
        errors = np.linalg.norm(projected.reshape(5, 2) - points, axis=1)
        # A generic five-point face cannot fit every person's proportions exactly.
        # Use both RMS and worst-point residuals rather than rejecting a whole
        # face for a two-pixel discrepancy, especially at smaller image sizes.
        if (not np.isfinite(errors).all()
                or np.sqrt(np.mean(errors ** 2)) > max(3.0, eye_span * 0.12)
                or errors.max() > max(5.0, eye_span * 0.22)):
            return None
        # Forward face normal is -z. atan2 retains turns beyond 90 degrees;
        # unlike eye-line tilt, this measures left/right yaw, not head roll.
        yaw = math.degrees(math.atan2(rotation[0, 2], rotation[2, 2]))
        return yaw if math.isfinite(yaw) else None
    except cv2.error:
        return None


def distance_features(face, yaw, frame_shape):
    """Eye spacing, face height and face width, in pixels.

    Cosine correction approximates yaw foreshortening. Bounding boxes are not
    anatomical measurements, so height/width get lower weights. Large roll,
    clipped faces and uncertain pose cannot provide dependable size estimates.
    """
    if yaw is None or not math.isfinite(yaw) or abs(yaw) > MAX_YAW_DEGREES:
        return None
    face = np.asarray(face, dtype=np.float64)
    if not np.isfinite(face).all():
        return None
    h, w = frame_shape[:2]
    x, y, fw, fh = face[:4]
    if min(fw, fh) < 20 or x < 0 or y < 0 or x + fw > w or y + fh > h:
        return None
    dx, dy = face[6:8] - face[4:6]
    eye_span = math.hypot(dx, dy)
    roll = math.degrees(math.atan2(abs(dy), abs(dx)))
    if eye_span < 12 or roll > 20:
        return None
    cosine = math.cos(math.radians(yaw))
    return np.array([eye_span / cosine, fh, fw / cosine], dtype=np.float64)


class DistanceCalibration:
    """Reference face size at 5 feet. No physical distance is inferred unsupervised."""

    def __init__(self, path=CALIBRATION_PATH):
        self.path = path
        self.reference = None
        self.frame_size = None
        self.active = False
        self.samples = []
        self.started = 0.0
        self.first_sample = None
        self.message = 'Stand alone at 5 ft, face forward, then press C.'
        try:
            with open(path, encoding='utf-8') as file:
                data = json.load(file)
            values = np.asarray(data['features'], dtype=np.float64)
            size = data['frame_size']
            if (data['version'] != 1 or data['reference_feet'] != RANGE_FEET
                    or values.shape != (3,) or not np.isfinite(values).all()
                    or (values <= 0).any() or len(size) != 2
                    or any(type(n) is not int or n <= 0 for n in size)):
                raise ValueError('Invalid calibration')
            self.reference, self.frame_size = values, tuple(size)
            self.message = 'Calibration loaded. C: recalibrate  Q: quit'
        except (OSError, ValueError, TypeError, KeyError):
            pass

    def begin(self, now):
        self.reference = None
        self.active = True
        self.samples = []
        self.started = now
        self.first_sample = None
        self.message = 'At 5 ft: one face only; face forward and hold still.'

    def update(self, faces, frame_shape, now):
        size = (int(frame_shape[1]), int(frame_shape[0]))
        if self.frame_size is not None and self.frame_size != size:
            self.reference = None
            self.message = 'Resolution changed. Stand at 5 ft and press C.'
        self.frame_size = size
        if not self.active:
            return
        if now - self.started > 20:
            self.active = False
            self.message = 'Calibration timed out. At 5 ft, press C to retry.'
            return
        features = None
        if faces is not None and len(faces) == 1:
            face = faces[0]
            yaw = estimate_yaw(face, frame_shape)
            if yaw is not None and abs(yaw) <= 12 and face[14] >= 0.8:
                features = distance_features(face, yaw, frame_shape)
        if features is None:
            self.samples = []
            self.first_sample = None
            self.message = 'At 5 ft: one face only; face forward and hold still.'
            return
        if self.samples:
            reference = np.median(self.samples, axis=0)
            if np.max(np.abs(features / reference - 1)) > 0.08:
                self.samples = []
                self.first_sample = None
        if not self.samples:
            self.first_sample = now
        self.samples.append(features)
        self.message = f'Calibrating at 5 ft: {min(len(self.samples), CALIBRATION_SAMPLES)}/{CALIBRATION_SAMPLES}'
        if len(self.samples) < CALIBRATION_SAMPLES or now - self.first_sample < 1.0:
            return
        self.reference = np.median(self.samples, axis=0)
        self.active = False
        self.message = 'Calibrated. Step closer than 4.7 ft to enter. C: recalibrate'
        data = dict(version=1, reference_feet=RANGE_FEET,
                    frame_size=list(size), features=self.reference.tolist())
        try:
            with open(self.path + '.tmp', 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2)
            os.replace(self.path + '.tmp', self.path)
        except OSError:
            self.message = 'Calibrated for this session; could not save calibration.'

    def estimate(self, face, yaw, frame_shape):
        if self.active or self.reference is None:
            return None
        if self.frame_size != (frame_shape[1], frame_shape[0]):
            return None
        values = distance_features(face, yaw, frame_shape)
        if values is None:
            return None
        estimates = RANGE_FEET * self.reference / values
        median = float(np.median(estimates))
        # Measurements that disagree suggest turning, bad landmarks, clipping,
        # or a face shape that this calibration does not describe well.
        if np.max(np.abs(estimates / median - 1)) > 0.25:
            return None
        return float(np.dot(estimates, [0.50, 0.35, 0.15]))


def blur_faces(frame, boxes):
    """Blur padded face rectangles only, including short detection gaps."""
    h, w = frame.shape[:2]
    for box in boxes:
        x, y, fw, fh = np.asarray(box, dtype=float)
        if not np.isfinite([x, y, fw, fh]).all() or min(fw, fh) <= 0:
            continue
        x1 = max(0, min(w, int(math.floor(x - fw * BLUR_PADDING))))
        y1 = max(0, min(h, int(math.floor(y - fh * BLUR_PADDING))))
        x2 = max(0, min(w, int(math.ceil(x + fw * (1 + BLUR_PADDING)))))
        y2 = max(0, min(h, int(math.ceil(y + fh * (1 + BLUR_PADDING)))))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        sigma = max(8.0, min(roi.shape[:2]) * 0.16)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma)
    return frame


class FaceCriteria:
    """Associate faces geometrically; never share smoothing across detections.

    Missing detections have no overlay; blur the last box for at most 0.25 seconds
    for reacquisition. Ambiguous matches start fresh to avoid carrying an overlay
    from one person to another. No identity recognition is performed.
    """

    def __init__(self, calibration):
        self.tracks = {}
        self.next_id = 0
        self.calibration = calibration
        self.blur_boxes = []

    @staticmethod
    def match_cost(box, old_box):
        a, b = np.asarray(box), np.asarray(old_box)
        span = max(float(np.linalg.norm(a[2:])), float(np.linalg.norm(b[2:])), 1.0)
        distance = np.linalg.norm(a[:2] + a[2:] / 2 - b[:2] - b[2:] / 2) / span
        ratio = np.maximum(a[2:] / b[2:], b[2:] / a[2:]).max()
        return float(distance) if distance < 0.45 and ratio < 1.8 else float('inf')

    def update(self, faces, frame_shape, now=None):
        now = time.monotonic() if now is None else now
        self.tracks = {key: value for key, value in self.tracks.items()
                       if 0 <= now - value['time'] <= 0.25}
        faces = [] if faces is None else [np.asarray(f, dtype=np.float64) for f in faces
                if np.isfinite(f[:4]).all() and f[2] > 0 and f[3] > 0]
        keys = list(self.tracks)
        costs = np.array([[self.match_cost(f[:4], self.tracks[k]['box'])
                           for k in keys] for f in faces], dtype=float)
        assignments = {}
        # Accept only mutual, unambiguous nearest neighbours. This is independent
        # of detector row order and enforces one detection per track.
        if faces and keys:
            for i in range(len(faces)):
                j = int(np.argmin(costs[i]))
                best = costs[i, j]
                if not np.isfinite(best) or int(np.argmin(costs[:, j])) != i:
                    continue
                row = np.sort(costs[i])
                col = np.sort(costs[:, j])
                if ((len(row) > 1 and row[1] - best < 0.08)
                        or (len(col) > 1 and col[1] - best < 0.08)):
                    continue
                assignments[i] = keys[j]
        overlays = []
        self.blur_boxes = []
        seen = set()
        for i, face in enumerate(faces):
            key = assignments.get(i)
            previous = self.tracks.get(key)
            if key is None:
                key = self.next_id
                self.next_id += 1
            seen.add(key)
            # Gate the CURRENT measurement: smoothing cannot leave the overlay on
            # after a measured turn beyond 50 degrees, or after a failed fit.
            yaw = estimate_yaw(face, frame_shape)
            raw_distance = self.calibration.estimate(face, yaw, frame_shape)
            distance = raw_distance
            was_clear = previous is not None and previous['clear']
            if (raw_distance is not None and previous is not None
                    and previous['distance'] is not None
                    and now - previous['time'] <= 0.10):
                dt = max(now - previous['time'], 0.001)
                alpha = 1.0 - math.exp(-dt / DISTANCE_SMOOTH_SECONDS)
                distance = alpha * raw_distance + (1 - alpha) * previous['distance']
            # A buffer on the INSIDE of 5 ft prevents boundary flicker. Current
            # measurements above 5 ft hide immediately; no stale near estimate
            # may keep a background face clear while the smoothing catches up.
            limit = RANGE_FEET if was_clear else ENTER_CLEAR_FEET
            clear = (raw_distance is not None and raw_distance <= limit
                     and distance <= limit)
            valid = clear and yaw is not None and abs(yaw) <= MAX_YAW_DEGREES
            if not clear:
                self.blur_boxes.append(face[:4].copy())
            self.tracks[key] = dict(box=face[:4].copy(), face=face.copy(), time=now, valid=valid,
                                    clear=clear, distance=distance)
            if valid:
                overlays.append((key, face.copy()))
        for key, track in self.tracks.items():
            if key not in seen:
                self.blur_boxes.append(track['box'])
                # Reacquired faces must qualify for the inner entry threshold.
                track['clear'] = False
                track['valid'] = False
                track['distance'] = None
        return overlays


def draw_calibration_status(frame, calibration):
    """Display setup instructions only; no labels after successful calibration."""
    if calibration.reference is None or calibration.active:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, max(0, h - 48)), (w, h), (25, 25, 25), -1)
        cv2.putText(frame, calibration.message, (8, h - 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'C: calibrate at 5 ft | Q: quit', (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

# =====================================  TUNING KNOBS  =====================================
# ---- The look ----
LOCKS_PER_SIDE = 34       # locks of hair on each side of the part
STRANDS_PER_LOCK = 12     # strands in each lock (lower both if the Pi feels slow)
HAIR_VOLUME = 1.0         # 1.0 = natural; bigger = the hair stands further off the head
HAIR_LENGTH = 1.0         # 1.0 = past the shoulders onto the chest
PART_SHIFT = 8            # where the part is, in mm from the middle of the head: 0 = dead center,
                          #   about 30 = a side part, minus numbers move it to the other side
GLOW = 0.4                # light glow around the shiny parts of the hair (0 = off)
SPARKLES = 30             # gold sparkles floating around the hair (0 = off)
SWAY = 0.7                # 0 = stiff, 1 = loose and swingy
BREEZE = 1.0              # gentle idle movement so the hair never looks frozen (0 = off)
FACE_SHADOW = 0.35        # how dark the soft shadow is that the hair casts on the face (0 = off)
MATCH_LIGHT = 0.8         # 0 = always the same bright gold; 1 = fully follow the room's light and color
GRAIN = 2.0               # camera-like grain on the hair so it doesn't look pasted on (0 = off)
FAR_SIDE_TURN = math.radians(20)   # past this turn, the far side's front hair tucks behind the cheek
CLEAR_FACE = 1.0          # keeps hair and sparkles off the eyes, nose and mouth (0 = off)
PERSPECTIVE = 1.0         # 1 = real camera depth: parts further from the camera look smaller (0 = off)
FOCAL_PX = 425            # the webcam's focal length in pixels, for a 640-wide picture (about 74 degree view)
GRAVITY = 0.75            # how much the long hair keeps hanging down when the head tilts
SEED = 7                  # change for a different random layout of the locks

# ---- Speed ----
TARGET_FPS = 15           # if it runs slower than this, the hair automatically thins itself out
MIN_DETAIL = 0.3          # ...but never below this much of the full detail
SUPERSAMPLE = 2           # the hair is drawn this many times bigger, then shrunk (finer, smoother strands)

# ---- Who gets the filter ----
SCORE_THRESHOLD = 0.8     # how sure YuNet must be that something is a face
MIN_EYE_GAP_PX = 11       # smaller faces are too far away to be on the opt-in spot
ZONE_LEFT, ZONE_RIGHT = 0.12, 0.88   # the opt-in face must be inside this part of the width
ACQUIRE_FRAMES = 4        # a face must show up this many frames in a row before hair appears
SWITCH_RATIO = 1.35       # another face must be this much bigger...
SWITCH_FRAMES = 12        # ...for this many frames in a row to take the filter over
MAX_MISSES = 12           # keep the hair (fading out) this many frames after losing the face
TRACK_RADIUS = 1.5        # a face is the same person if its nose moved less than this many eye gaps

# ---- Head pose ----
YAW_LIMIT = math.radians(70)     # turning left/right is capped here (the detector guesses past it)
PITCH_LIMIT = math.radians(30)   # tipping up/down is capped here
SHAPE_JUMP = 0.12                # a one-frame change in the face shape this big (in eye gaps) is a glitch,
SHAPE_JUMP_MIN_PX = 4.0          # but never count less than this many pixels (normal wobble on small faces)
FIT_ERROR_LIMIT = 0.16           # if the 5 points don't fit a real head this well, don't trust them...
REBASE_FRAMES = 8                # ...unless it lasts this long, which means it's a real pose
# ===========================================================================================

# ---------------------------------------------------------------------------------------
# The 3D face. Units are millimetres. The origin is the point between the eyes, x points to
# the right of the screen, y points down, z points toward the camera. These numbers were
# measured from where YuNet actually puts its five points on real frontal faces.
# ---------------------------------------------------------------------------------------
EYE_X = 31.5
NOSE_Y, NOSE_Z = 33.0, 20.0
MOUTH_X, MOUTH_Y, MOUTH_Z = 26.5, 61.0, -5.0
FACE_MODEL = np.array([[-EYE_X, 0, 0], [EYE_X, 0, 0], [0, NOSE_Y, NOSE_Z],
                       [-MOUTH_X, MOUTH_Y, MOUTH_Z], [MOUTH_X, MOUTH_Y, MOUTH_Z]], float)
HEAD_CENTER = np.array([0.0, -12.0, -88.0])    # middle of the skull
# The surface the hair lies on: an egg shape around HEAD_CENTER (half-width, half-height,
# half-depth in mm). It's the skull plus the person's own short hair, so ours sits on top of it.
SCALP = np.array([93.0, 123.0, 112.0])


def load_model() -> None:
    """Download the YuNet model the first time the script runs."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    print("Downloading YuNet face model (one time only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if os.path.getsize(MODEL_PATH) < 100_000:
        os.remove(MODEL_PATH)
        sys.exit("Error: the model download came back empty. Download it by hand from\n"
                 f"{MODEL_URL}\nand put it next to this script.")
    print("Done.")


def rotate(P, yaw, pitch, roll):
    """Turn 3D points by the head pose. Order: turn (yaw), then tip (pitch), then tilt (roll).
    The angles can be single numbers or arrays (one angle per point)."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    cy, sy = np.cos(yaw), np.sin(yaw)
    x1, z1 = x * cy + z * sy, z * cy - x * sy
    cp, sp = np.cos(pitch), np.sin(pitch)
    y2, z2 = y * cp - z1 * sp, y * sp + z1 * cp
    cr, sr = np.cos(roll), np.sin(roll)
    return x1 * cr - y2 * sr, x1 * sr + y2 * cr, z2


def depth_scale(Z, pose):
    """Real cameras make things further away look smaller. The face points (z about 0) are
    where the pose's size s was measured; a point Z mm closer to the camera looks bigger by
    this factor, one further away (the back of the head, hair behind the shoulders) smaller.
    It matters most up close: at 40 cm the back of the head looks about 30% smaller."""
    if PERSPECTIVE <= 0:
        return np.ones_like(np.asarray(Z, float))
    zc = pose.get("f", FOCAL_PX) / pose["s"]           # how far the face is from the camera, in mm
    k = zc / np.maximum(zc - np.asarray(Z, float), 0.5 * zc)
    return 1.0 + PERSPECTIVE * (k - 1.0)


def project(P, pose, ox=0.0, oy=0.0, keep=1.0):
    """3D points around the head (mm) -> screen pixels, turned with the head pose. keep < 1
    only follows part of the tip and tilt (for hair and the neck, which hang)."""
    X, Y, Z = rotate(np.asarray(P, float), pose["yaw"], pose["pitch"] * keep, pose["roll"] * keep)
    k = pose["s"] * depth_scale(Z, pose)
    return np.stack([pose["x"] + k * X - ox, pose["y"] + k * Y - oy], -1)


# =====================================  HEAD POSE  =====================================
def face_points(row):
    """YuNet row -> the 5 points as a (5, 2) array: L eye, R eye, nose, L mouth, R mouth."""
    p = row[4:14].reshape(5, 2).astype(float)
    eyes = p[:2][np.argsort(p[:2, 0])]          # left = smaller x on screen
    mouth = p[3:5][np.argsort(p[3:5, 0])]
    return np.vstack([eyes, p[2:3], mouth])


def _project_face(params):
    x0, y0, log_s, yaw, pitch, roll = params
    X, Y, _ = rotate(FACE_MODEL, yaw, pitch, roll)
    s = math.exp(log_s)
    return np.stack([x0 + s * X, y0 + s * Y], 1)


def estimate_pose(pts):
    """Work out the head pose from the 5 points.

    Returns a dict: x, y (the point between the eyes on screen), s (pixels per millimetre),
    yaw, pitch, roll (radians), err (how badly the points fit a real head) and gap (eye gap px).
    """
    le, re, nose, ml, mr = pts
    ev, mv = re - le, mr - ml
    gap, mouth_w = math.hypot(*ev), math.hypot(*mv)
    if gap < 1 or mouth_w < 1:
        return None

    # 1. A quick first guess. Rotate the points so the eyes are level...
    roll = math.atan2(ev[1] + 0.5 * mv[1], ev[0] + 0.5 * mv[0])
    c, s = math.cos(-roll), math.sin(-roll)
    E = (le + re) / 2
    up = lambda q: np.array([c * (q[0] - E[0]) - s * (q[1] - E[1]), s * (q[0] - E[0]) + c * (q[1] - E[1])])
    M, N = (up(ml) + up(mr)) / 2, up(nose)
    if M[1] < 1:
        return None
    r = N[1] / M[1]                                   # how far down the nose sits, eyes -> mouth
    depth = NOSE_Z - r * MOUTH_Z                      # how far the nose sticks out
    # ...turning sideways slides the nose off the eye-mouth center line and squeezes the eye gap
    yaw = math.atan2((N[0] - r * M[0]) / depth, (gap + mouth_w) / (2 * EYE_X + 2 * MOUTH_X))
    # ...tipping up or down slides the nose up or down between the eyes and the mouth
    pitch = math.atan((NOSE_Y - r * MOUTH_Y) / (depth * max(math.cos(yaw), 0.3)))
    s_side = math.hypot((gap + mouth_w) / (2 * EYE_X + 2 * MOUTH_X), (N[0] - r * M[0]) / depth)
    s_vert = M[1] / max(MOUTH_Y * math.cos(pitch) - MOUTH_Z * math.sin(pitch), 1.0)
    params = np.array([E[0], E[1], math.log(math.sqrt(s_side * s_vert)), yaw, pitch, roll])

    # 2. Polish the guess: nudge all six numbers until the 3D face lands on the 5 points.
    eps = np.array([0.05, 0.05, 1e-3, 1e-3, 1e-3, 1e-3])
    for _ in range(5):
        res = (_project_face(params) - pts).ravel()
        J = np.empty((10, 6))
        for k in range(6):
            p2 = params.copy()
            p2[k] += eps[k]
            J[:, k] = ((_project_face(p2) - pts).ravel() - res) / eps[k]
        A = J.T @ J
        A += np.diag(np.diag(A)) * 0.01 + np.eye(6) * 1e-9
        params -= np.linalg.solve(A, J.T @ res)

    err = math.sqrt(np.mean(np.sum((_project_face(params) - pts) ** 2, 1))) / gap
    x0, y0, log_s, yaw, pitch, roll = params
    return dict(x=x0, y=y0, s=math.exp(log_s), yaw=float(np.clip(yaw, -YAW_LIMIT, YAW_LIMIT)),
                pitch=float(np.clip(pitch, -PITCH_LIMIT, PITCH_LIMIT)), roll=roll,
                err=err, gap=gap, nose=pts[2].copy(), pts=pts)


class OneEuro:
    """Smooths a jumpy number. When it's barely moving it smooths hard (no shaking); when it's
    moving fast it smooths less (no lag). This is the standard trick face filters use."""

    def __init__(self, min_cutoff, beta):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))

    def __call__(self, x, dt):
        if self.x is None:
            self.x, self.dx = x, 0.0
            return x
        dx = (x - self.x) / dt
        self.dx += self._alpha(1.0, dt) * (dx - self.dx)
        self.x += self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt) * (x - self.x)
        return self.x


MEDIAN_FRAMES = 1     # median of this many frames before smoothing (1 = off; 3 added lag)
# (min_cutoff in Hz, beta) for each pose number. Picked by a sweep that scored shimmer when
# standing still against lag when turning, stepping, sitting and walking.
FILTER_SETTINGS = dict(x=(2.5, 0.04), y=(2.5, 0.04), log_s=(0.4, 1.5),
                       yaw=(0.4, 0.3), pitch=(0.4, 0.3), roll=(0.6, 0.3))


class Target:
    """One person the filter is drawn on: their smoothed head pose, fade, and hair state."""

    def __init__(self, raw, model):
        self.filters = {k: OneEuro(*v) for k, v in FILTER_SETTINGS.items()}
        self.history = []            # the last 3 raw poses, for throwing out one-frame glitches
        self.pose = None             # smoothed pose we draw with
        self.trusted_raw = None
        self.nose = raw["nose"]
        self.gap = raw["gap"]
        self.criteria_id = raw.get("criteria_id")   # the team rules' id for this face
        self.seen = 0                # frames in a row we've seen this face
        self.misses = 0              # frames in a row we haven't
        self.untrusted_run = 0
        self.shape = self.pending = None   # the 5-point shape we trust, and a suspicious new one
        self.alpha = 0.0             # 0 = invisible, 1 = fully drawn
        self.feat = None             # where the eyes, nose and mouth really are (kept clear of hair)
        self.hair = HairState(model)
        self.update(raw, 1 / 15)

    @property
    def acquired(self):
        return self.seen >= ACQUIRE_FRAMES or self.alpha > 0

    def update(self, raw, dt):
        self.seen += 1
        self.misses = 0
        nose_move = raw["nose"] - self.nose
        self.nose, self.gap = raw["nose"], raw["gap"]

        # Glitch check. On a real head the 5 points keep almost the same shape from one frame
        # to the next (measured from the nose, in eye gaps). If the shape suddenly jumps, it's
        # either a detector glitch (it snaps back next frame) or a real change (it stays). We
        # hold for one frame to find out. Points that don't fit a real head at all (a hand over
        # the face) are also held, until it has lasted REBASE_FRAMES, which means it's real.
        shape = (raw["pts"] - raw["nose"]) / raw["gap"]     # in eye gaps, so stepping closer doesn't count
        limit = max(SHAPE_JUMP, SHAPE_JUMP_MIN_PX / raw["gap"])    # tiny faces wobble more
        jump = self.shape is not None and np.abs(shape - self.shape).max() > limit
        if jump and self.pending is not None and np.abs(shape - self.pending).max() < limit / 2:
            jump = False                                   # it stayed: real, accept it
        bad = jump or raw["err"] > FIT_ERROR_LIMIT
        if bad and self.trusted_raw is not None and self.untrusted_run < REBASE_FRAMES:
            self.untrusted_run += 1
            self.pending = shape
            raw = dict(self.trusted_raw, x=self.trusted_raw["x"] + nose_move[0],
                       y=self.trusted_raw["y"] + nose_move[1])
            self.trusted_raw = raw
            self.feat = self.feat + nose_move if self.feat is not None else None
        else:
            self.untrusted_run = 0
            # remember a smoothed shape, so one noisy frame isn't the yardstick for the next
            self.shape = shape if self.shape is None else 0.5 * self.shape + 0.5 * shape
            self.pending = None
            self.trusted_raw = raw
            self.feat = np.array(raw["pts"], float)

        # Median of the last 3 frames throws out one-frame glitches, then the smoother.
        vals = dict(x=raw["x"], y=raw["y"], log_s=math.log(raw["s"]),
                    yaw=raw["yaw"], pitch=raw["pitch"], roll=raw["roll"])
        self.history = (self.history + [vals])[-MEDIAN_FRAMES:]
        out = {}
        for k in vals:
            v = float(np.median([h[k] for h in self.history]))
            out[k] = self.filters[k](v, dt)
        self.pose = dict(x=out["x"], y=out["y"], s=math.exp(out["log_s"]),
                         yaw=out["yaw"], pitch=out["pitch"], roll=out["roll"], feat=self.feat)
        if self.acquired:
            self.alpha = min(1.0, self.alpha + dt / 0.35)      # fade in over 0.35 s

    def miss(self, dt):
        self.seen = 0
        self.misses += 1
        if self.misses > 2:                                     # a 1-2 frame detector blink
            self.alpha = max(0.0, self.alpha - dt / 0.3)        # doesn't fade; longer ones do, fast

    @property
    def gone(self):
        return self.misses > MAX_MISSES or (self.misses > 2 and self.alpha <= 0)


# =====================================  HAIR MODEL  =====================================
def _spline(ctrl, n):
    """Smooth curve through the control points, resampled to n evenly spaced points."""
    P = np.vstack([2 * ctrl[0] - ctrl[1], ctrl, 2 * ctrl[-1] - ctrl[-2]])
    t = np.linspace(0, 1, 10, endpoint=False)[:, None]
    dense = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        dense.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                            + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    dense = np.vstack(dense + [ctrl[-1:]])
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    want = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(want, cum, dense[:, k]) for k in range(3)], 1), cum


def _resample(curve, d):
    """Points along a curve (made of evenly spaced points) at fractions d of its length."""
    x = np.clip(d, 0, 1) * (len(curve) - 1)
    i = np.minimum(x.astype(int), len(curve) - 2)
    f = (x - i)[:, None]
    return curve[i] * (1 - f) + curve[i + 1] * f


def to_scalp(p, lift):
    """Slide points straight in or out from the middle of the head until they sit on the scalp,
    then lift them that many mm off it. This is what keeps the hair hugging the head."""
    v = np.asarray(p, float) - HEAD_CENTER
    on = v / np.sqrt(((v / SCALP) ** 2).sum(-1, keepdims=True))
    out = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return HEAD_CENTER + on + out * np.asarray(lift, float)[..., None]


# Where the hair lies on the head, for the right side (the left side is a mirror image).
# Each guide runs from the part, over the head, to where the hair leaves the head and hangs.
# u = 0 is the front of the part, 0.5 the top of the crown, 1 the back of the crown.
# The guide points only give the direction; to_scalp() decides how far out they sit.
LAY_GUIDES = [
    (0.0, [(0, -53, 14), (27, -52, 11), (48, -39, 1), (63, -17, -16), (78, 8, -30)]),
    (0.5, [(0, -130, -70), (50, -122, -72), (85, -80, -72), (95, -30, -72), (95, 22, -70)]),
    (1.0, [(0, -110, -168), (48, -102, -168), (84, -66, -156), (95, -14, -144), (95, 40, -130)]),
]
LAY_LIFT = np.array([4.0, 7.5, 11.0, 15.0, 19.0])   # mm above the scalp: flat at the part, fuller lower down
# Framing locks start at the front of the part, fall over the corners of the forehead in front
# of the face, and run down along the cheeks. They soften the hairline and hide the temples.
FRAME_GUIDE = [(0, -53, 14), (18, -51, 13), (37, -41, 6), (50, -24, -6), (57, 0, -14)]
FRAME_LIFT = np.array([2.0, 4.0, 6.0, 8.0, 9.0])
FRAME_LOCKS = 4       # per side
UNDER_LOCKS = 10      # per side: darker hair underneath, filling in beside the neck
FLYAWAYS = 8          # per side: long loose hairs with lazy curves
BABY_HAIRS = 10       # per side: short fine hairs along the hairline (soft hairline)
PART_CROSSERS = 3     # per side: short hairs that stray across the part
FRIZZ = 22            # per side: little arcs of hair that break up the outline
N_POINTS = 33         # points along each strand
N_DENSE = 129         # points along each lock's center line
N_SECTIONS = 8        # each strand is drawn in 8 pieces so its shading can change along it
CAP_U, CAP_T = 13, 7  # the solid "cap" under the hair: rows front to back, points from the part down
CAP_LIFT = 1.5        # mm above the scalp (just under the strands)
CAP_FRONT_U = 0.42    # the front part of the cap is drawn with the front hair
TUCK_BELOW_Y = -30    # on a turned head, far-side front hair lower than this (mm) goes behind the cheek


def _lay_guide(u):
    for (u0, g0), (u1, g1) in zip(LAY_GUIDES, LAY_GUIDES[1:]):
        if u <= u1:
            f = (u - u0) / (u1 - u0)
            return (1 - f) * np.array(g0, float) + f * np.array(g1, float)
    return np.array(LAY_GUIDES[-1][1], float)


def _part_wander(u):
    """The part isn't a ruler-straight line: it wanders a little from front to back."""
    return 1.3 * math.sin(23 * u + 1.7) + 0.7 * math.sin(61 * u)


class HairModel:
    """Builds the hairstyle once at startup, in 3D, in millimetres around the head.

    The hair is made of locks. Each lock has a center curve that starts at the part, lies close
    over the head, leaves the head beside the face, and hangs down in big soft waves. Its strands
    spread around that center, bunch into a few smaller clumps, end at different lengths, and
    gather toward the tips, which is what makes hair look like hair instead of spaghetti.
    """

    def __init__(self, seed=SEED):
        rng = np.random.default_rng(seed)
        S = {k: [] for k in ("P", "d", "hs", "front", "side", "kind", "fam", "ao", "bias", "spec", "rank")}

        def add(P, d, hs, front, side, kind, fam, ao, bias, spec, rank):
            for k, v in zip(S, (P, d, hs, front, side, kind, fam, ao, bias, spec, rank)):
                S[k].append(v)

        def add_lock(side, kind, u=0.0, n=STRANDS_PER_LOCK, width=12.0, thick=3.5, depth=1.0, rank_step=1.0,
                     body=True):
            front = kind in ("front", "frame")
            lock = self._lock(side, kind, u, rng)
            pts, hs, N, B, stack = lock
            fam = int(rng.choice(3, p=FAMILY_MIX))
            bias = rng.uniform(0.86, 1.1)
            spec = rng.normal(0, 0.06)
            ao = depth * (0.88 + 0.12 * stack)            # locks lying on top get more light
            # The lock's center line doubles as a thick, darker "body" under its strands: the
            # shadow you see between strands, and it keeps the real hair from showing through.
            if body:
                d = np.linspace(0.015, 0.9, N_POINTS)
                add(_resample(pts, d), d, hs, front, side, 2, 0, ao, bias, spec, 0)
            # A lock bunches into a few smaller clumps that wave a little on their own
            subs = [(rng.normal(0, 0.45) * width, rng.normal(0, 0.4) * thick, rng.normal(0, 0.15) * width,
                     rng.uniform(3, 6), rng.uniform(0, 2 * np.pi)) for _ in range(3)]
            spread = 0.45 if kind in ("back", "under") else 0.3     # back tips thin out more
            for j in range(n):
                P, d, inner = self._strand(lock, subs[rng.integers(3)], rng, width, thick, spread)
                f = fam if rng.random() < 0.8 else int(rng.choice(3, p=FAMILY_MIX))
                add(P, d, hs, front, side, 0, f, ao * (0.8 + 0.2 * inner),
                    bias * rng.uniform(0.85, 1.12), spec + rng.normal(0, 0.18), 1 + j * rank_step)
            return lock

        for side in (-1, 1):
            tops = []
            for i in range(LOCKS_PER_SIDE):
                u = (i + rng.uniform(0.15, 0.85)) / LOCKS_PER_SIDE
                kind = "front" if u < 0.42 + rng.uniform(-0.08, 0.08) else "back"
                tops.append((add_lock(side, kind, u), kind))
            for i in range(FRAME_LOCKS):
                add_lock(side, "frame", 0.0, n=max(4, STRANDS_PER_LOCK * 2 // 3), width=6.0, thick=2.5, body=False)
            for i in range(UNDER_LOCKS):
                add_lock(side, "under", n=max(4, STRANDS_PER_LOCK * 2 // 3), width=14.0, thick=5.0,
                         depth=0.72, rank_step=1.5)

            # Loose flyaway hairs: thin, light, with lazy curves
            for _ in range(FLYAWAYS):
                u = rng.uniform(0.0, 0.8)
                kind = "front" if u < 0.4 else "back"
                lock = self._lock(side, kind, u, rng, wild=True)
                P, d, _ = self._strand(lock, (0, 0, 0, 1, 0), rng, 20.0, 6.0, spread=0.2)
                add(P, d, lock[1], kind == "front", side, 1, 1, 0.95, rng.uniform(0.95, 1.08), rng.normal(0, 0.1), 5)

            # Baby hairs: short, fine hairs lying along the hairline, so it isn't a hard edge
            for _ in range(BABY_HAIRS):
                pts, hs, N, B, _ = self._lock(side, "frame", 0.0, rng)
                d0 = rng.uniform(0.01, 0.1)
                d = np.linspace(d0, d0 + rng.uniform(0.03, 0.055), N_POINTS)
                tau = np.linspace(0, 1, N_POINTS)[:, None]
                # lie along the hairline, sitting a few mm below the rest of the hair
                P = _resample(pts, d) + [0, 1, 0] * (rng.uniform(2, 5) + tau * rng.uniform(0, 3))
                add(to_scalp(P, 1.5), d + 0.07, hs, True, side, 1, 1, 0.95, rng.uniform(1.0, 1.15), rng.normal(0, 0.1), 2)

            # A few short hairs that stray across the part
            for _ in range(PART_CROSSERS):
                u = rng.uniform(0.0, 0.5)
                g = _lay_guide(u)[0]
                x0 = PART_SHIFT + _part_wander(u)
                ctrl = np.array([[x0 + side * rng.uniform(2, 6), g[1], g[2]],
                                 [x0 - side * rng.uniform(3, 6), g[1] - 2, g[2] + rng.normal(0, 3)],
                                 [x0 - side * rng.uniform(12, 22), g[1] + rng.normal(0, 3), g[2] + rng.normal(0, 5)]])
                P, _ = _spline(ctrl, N_POINTS)
                tau = np.linspace(0, 1, N_POINTS)
                d = np.linspace(0.01, 0.05, N_POINTS)
                add(to_scalp(P, 3 + 2 * np.sin(np.pi * tau)), d, 0.5, u < 0.42, side, 1, 1, 0.9,
                    rng.uniform(0.95, 1.1), rng.normal(0, 0.1), 2)

            # Frizz: little arcs that lift off the top locks and break up the outline
            for _ in range(FRIZZ):
                (pts, hs, N, B, _), kind = tops[rng.integers(len(tops))]
                i0 = int(rng.uniform(0.05, min(hs, 0.4)) * (N_DENSE - 1))
                T = pts[min(i0 + 1, N_DENSE - 1)] - pts[max(i0 - 1, 0)]
                T = T / (np.linalg.norm(T) + 1e-9)
                tau = np.linspace(0, 1, N_POINTS)[:, None]
                L = rng.uniform(18, 40)
                P = (pts[i0] + T * L * tau + N[i0] * rng.uniform(4, 12) * np.sin(np.pi * tau ** 0.8)
                     + B[i0] * rng.uniform(-6, 6) * np.sin(2 * np.pi * tau + rng.uniform(0, 6)))
                d0 = i0 / (N_DENSE - 1)
                add(P, np.linspace(d0, d0 + L / 600, N_POINTS), hs, kind == "front", side, 1,
                    int(rng.choice(3, p=FAMILY_MIX)), 1.0, rng.uniform(0.95, 1.1), rng.normal(0, 0.1), 4)

        # Sort the strands by importance (lock bodies first, then the 1st strand of every lock,
        # then the 2nd...). Drawing only the first K strands then thins the hair out evenly.
        order = np.lexsort((rng.random(len(S["rank"])), np.array(S["rank"])))
        A = {k: np.array(v)[order] for k, v in S.items()}
        self.base = A["P"]                                   # (strands, N_POINTS, 3)
        self.d = A["d"]                                      # how far along its lock each point is (0..1)
        self.front = A["front"].astype(bool)
        self.side = A["side"]
        self.kind = A["kind"]                                # 0 strand, 1 fine hair, 2 lock body
        self.n_bodies = int((self.kind == 2).sum())          # they come first in the order
        self.family = A["fam"]                               # which blonde tone
        self.ao = A["ao"]                                    # how much light reaches it (deeper = darker)
        self.bias = A["bias"]                                # each strand a little lighter or darker
        self.spec_shift = A["spec"]                          # moves each strand's shine a little
        # Which pieces of each strand lie low enough to go behind the cheek on a turned head
        # (the part on top of the forehead stays in front, so it keeps covering the hairline).
        mid = slice((N_POINTS - 1) // N_SECTIONS // 2, N_POINTS - 1, (N_POINTS - 1) // N_SECTIONS)
        self.low = self.base[:, mid, 1] > TUCK_BELOW_Y
        # Depth band: strands deeper in the hair (less light) are drawn first, the outer ones last
        self.depth_band = np.minimum((np.argsort(np.argsort(self.ao)) * DEPTH_BANDS) // len(self.ao),
                                     DEPTH_BANDS - 1)
        n = len(self.base)

        # How much of each strand is hanging free. Hanging parts swing and keep pointing down
        # when you tilt your head.
        hs = A["hs"][:, None]
        self.hang = np.clip((self.d - hs) / np.maximum(1 - hs, 1e-3), 0, 1)
        k_tip = 0.55 - 0.4 * SWAY
        self.stiff = 1.0 - (1.0 - k_tip) * self.hang ** 1.2
        # Breeze: a slow side-to-side wave, mostly at the tips, different for each lock
        self.breeze_amp = BREEZE * 5.0 * self.hang ** 1.5 * rng.uniform(0.6, 1.4, (n, 1))
        self.breeze_phase = rng.uniform(0, 2 * np.pi, (n, 1)) * 0.3 + self.d * 2.2
        self.breeze_speed = rng.uniform(1.0, 1.7, (n, 1))
        # The part line. A root-colored strip is drawn under it, so the person's own hair never
        # shows through the gap between the two sides.
        u = np.linspace(0, 1, N_POINTS)
        self.part_line = to_scalp([[PART_SHIFT + _part_wander(v), *_lay_guide(v)[0][1:]] for v in u], 1.0)
        # A solid cap in a shadowed hair color, under all the strands. Wherever the strands leave
        # a gap (at the part, between locks, at the hairline), you see more blonde hair, not the
        # person's own hair.
        cap = []
        for side in (-1, 1):
            rows = []
            for v in np.linspace(0, 1, CAP_U):
                g = _lay_guide(v)
                g[:, 0] *= side
                g[0, 0] = PART_SHIFT + _part_wander(v) + side * 0.3
                g[1:, 0] += PART_SHIFT * np.array([0.6, 0.25, 0.0, 0.0])
                rows.append(to_scalp(_spline(g, CAP_T)[0], CAP_LIFT))
            cap.append(rows)
        self.cap = np.array(cap)                             # (2 sides, CAP_U, CAP_T, 3)
        self.cap_front_rows = int(CAP_FRONT_U * (CAP_U - 1))   # quads in front of this row go up front
        self.cap_high_cols = max(1, int((self.cap[1, 0, :, 1] <= TUCK_BELOW_Y).sum()) - 1)   # ...the rest tucks
        # Points to place sparkles near: the outside of the hair
        self.sparkle_anchors = self.base[self.kind == 0][:, ::3].reshape(-1, 3)

    def _lock(self, side, kind, u, rng, wild=False):
        """One lock's center line (N_DENSE points), the fraction of it where it starts hanging,
        two directions across it (N: away from the head, B: sideways along the head), and how
        high it is stacked on the other locks (0..1)."""
        if kind == "under":
            # Rooted lower on the side and back of the head, underneath the top locks
            th, p0 = math.radians(rng.uniform(85, 165)), math.radians(rng.uniform(66, 84))
            ps = np.linspace(p0, math.radians(104), 5)
            g = HEAD_CENTER + np.stack([side * np.sin(ps) * math.sin(th), -np.cos(ps),
                                        np.sin(ps) * math.cos(th)], 1) * SCALP
            lift, stack = np.array([2.0, 4.0, 6.0, 8.0, 10.0]) * HAIR_VOLUME, 0.0
        else:
            if kind == "frame":
                g, lift = np.array(FRAME_GUIDE, float), FRAME_LIFT.copy()
                g[2:, 0] += rng.uniform(-2, 7)          # each framing lock takes its own path
            else:
                g, lift = _lay_guide(u), LAY_LIFT.copy()
            g[:, 0] *= side
            g = g + rng.normal(0, 2.5, (5, 3))
            # Every root starts right next to the part (on this lock's side of it)
            g[0, 0] = PART_SHIFT + _part_wander(u) + side * rng.uniform(0.5, 2.0)
            g[1:, 0] += PART_SHIFT * np.array([0.6, 0.25, 0.0, 0.0])   # the shift fades out down the side
            stack = rng.uniform(0, 1)
            lift[1:] = lift[1:] * HAIR_VOLUME + 4 * stack            # locks lie on top of each other
        lay = to_scalp(g, lift)

        Lx, Ly, Lz = lay[-1]
        # Lengths vary from lock to lock. The back is cut more unevenly, so it never ends in
        # one flat line (you see that edge when the head is turned to the side).
        L = rng.uniform(*((0.74, 1.16) if kind in ("back", "under") else (0.88, 1.08))) * HAIR_LENGTH
        V = HAIR_VOLUME
        if kind == "front":     # falls over the front of the shoulder, onto the chest
            ys, xo, zs = [Ly + 70, 170, 255, 340, 420], [6, 20, 30, 34, 30], [-30, -36, -34, -30, -26]
        elif kind == "frame":   # runs down along the cheek, then over the chest
            ys, xo, zs = [Ly + 60, 150, 240, 330, 410], [2, 9, 18, 24, 22], [-10, -24, -32, -30, -26]
        elif kind == "back":    # falls down the back, ending behind the shoulders
            ys, xo, zs = [Ly + 65, 150, 215, 255], [14, 30, 42, 48], Lz + np.array([-8, -14, -18, -20])
        else:                   # under: hangs straight down beside the neck
            ys, xo, zs = [Ly + 50, 130, 195, 235], [4, 8, 12, 14], Lz + np.array([-4, -8, -10, -12])
        ys = Ly + (np.array(ys, float) - Ly) * L
        xs = Lx + side * np.array(xo, float) * V + rng.normal(0, 3, len(ys))
        hang = np.stack([xs, ys, np.array(zs, float)], 1)
        pts, _ = _spline(np.vstack([lay, hang]), N_DENSE)
        # Where along the lock it leaves the head (as a fraction of its length)
        lay_len = np.linalg.norm(np.diff(lay, axis=0), axis=1).sum()
        total = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        hang_start = min(lay_len / total, 0.6)
        # Big soft waves in the hanging part, growing toward the tips
        d = np.linspace(0, 1, N_DENSE)
        grow = np.clip((d - hang_start) / 0.22, 0, 1)
        amp = rng.uniform(10, 17) * (1.4 if wild else 1.0) * V
        wl, ph = rng.uniform(150, 200) / total, rng.uniform(0, 2 * np.pi)      # waves 15-20 cm long
        pts[:, 0] += amp * grow * np.sin(2 * np.pi * d / wl + ph)
        pts[:, 2] += 0.45 * amp * grow * np.cos(2 * np.pi * d / wl + ph)

        # Directions across the lock: N points away from the head (on the head) or away from
        # the body (hanging); B points sideways. Strands are spread out along these.
        T = np.gradient(pts, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        h = np.clip((d - hang_start) / 0.12, 0, 1)[:, None]
        n1 = pts - HEAD_CENTER
        n2 = pts * [1, 0, 1] - [0, 0, HEAD_CENTER[2]]
        n = (n1 / np.linalg.norm(n1, axis=1, keepdims=True) * (1 - h)
             + n2 / (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-9) * h)
        N = n - (n * T).sum(1, keepdims=True) * T
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
        return pts, hang_start, N, np.cross(T, N), stack

    @staticmethod
    def _strand(lock, sub, rng, width, thick, spread=0.3):
        """One strand of a lock. Returns its points, how far along the lock each point is,
        and how far to the outside of the lock it lies (0 = inside, in shadow; 1 = outside)."""
        pts, hs, N, B, _ = lock
        f_len = 1 - spread * rng.random() ** 1.6                     # strands end at different lengths
        d = np.linspace(rng.uniform(0, 0.012), f_len, N_POINTS)      # roots spread out from the part
        C, Nv, Bv = _resample(pts, d), _resample(N, d), _resample(B, d)
        hang = np.clip((d - hs) / max(1 - hs, 1e-3), 0, 1)
        gather = 1 - 0.75 * np.clip((d - 0.55) / 0.45, 0, 1) ** 1.2  # tips gather into points
        widen = 1 + 0.35 * np.sin(np.pi * hang)                      # locks are fullest halfway down
        sa, sb, wamp, wk, wph = sub
        oa, ob = rng.normal(0, 0.35) * width, rng.normal(0, 0.35) * thick
        a = ((sa + oa) * widen * gather + wamp * np.sin(wk * np.pi * d + wph) * hang
             + rng.normal(0, 1.0) * np.sin(rng.uniform(4, 9) * np.pi * d + rng.uniform(0, 6)) * hang)
        b = np.maximum((sb + ob) * (1 + 1.8 * hang) * gather, -2.0 - 6 * hang)
        P = C + a[:, None] * Bv + b[:, None] * Nv
        inner = float(np.clip(0.5 + 0.5 * (sb + ob) / max(thick, 1e-3), 0, 1))
        return P, d, inner


# Three blonde tones, each running from dark (deep shadow inside the hair) to light (shine), in
# BGR. Every strand gets one, so the hair has the mix of shades that real blonde hair has.
# Kept pale and only lightly golden: a webcam, a monitor and a phone all push gold toward orange,
# so these start paler than the gold people should see. Like real blonde hair, the shadows are a
# warm light brown and the shine is a creamy yellow (the hue moves from about 32 to 45 degrees).
PALETTE_STOPS = [0.0, 0.2, 0.42, 0.64, 0.83, 1.0]
PALETTES = np.array([
    [[30, 50, 69], [64, 99, 128], [106, 154, 184], [146, 197, 222], [186, 228, 245], [223, 246, 254]],   # honey
    [[34, 53, 69], [70, 104, 128], [116, 160, 184], [155, 202, 222], [193, 232, 245], [228, 248, 254]],   # pale gold
    [[43, 57, 69], [85, 111, 128], [134, 167, 184], [175, 209, 222], [208, 236, 245], [233, 249, 254]],   # champagne
], float)
FAMILY_MIX = [0.4, 0.45, 0.15]    # how often each tone is picked
N_SHADES = 24
COLORS = [[tuple(int(round(np.interp(i / (N_SHADES - 1), PALETTE_STOPS, pal[:, c]))) for c in range(3)) + (255,)
           for i in range(N_SHADES)] for pal in PALETTES]
BODY_MM = 7.0           # how wide the dark lock bodies are drawn (mm)
BODY_DARKER = 2         # lock bodies are this many shades darker than their strands
BODY_ALPHA = 0.85       # ...and almost solid, so the hair reads as a full mass up close
BODY_COLORS = [tuple(int(round(v * BODY_ALPHA)) for v in c) for c in COLORS[0]]
BODY_WIDTH_CLASS = np.array([0, 0, 0, 0, 0, 1, 2, -1])   # bodies get thinner toward the tips, then stop
SOFTEN = 1.2            # up close, blur the hair very slightly, like the camera blurs real hair
                        #   (stops the "wires" look; 0 = off). Starts at s = 0.55, full at s = 1.2
PART_MM = 7.0           # width of the root-colored strip under the part (mm)
PART_COLOR = COLORS[1][10]  # a little darker than the hair, like natural roots
CAP_COLOR = COLORS[1][9]    # the solid cap under the hair: shadowed blonde
REF_BRIGHTNESS = 105    # average camera brightness where the hair is drawn at its normal brightness
# The order the groups of strand pieces are drawn in, for each layer: lock bodies first, then
# strands from the deepest depth band to the outermost, and inside each band darkest to brightest.
DEPTH_BANDS = 3
DRAW_ORDER = [[((((layer * DEPTH_BANDS + band) * 5 + what) * 3 + f) * N_SHADES + b, what, f, b)
               for band, what, f, b in [(0, w_, 0, b_) for w_ in (0, 1, 2) for b_ in range(N_SHADES)]
               + [(d_, w_, f_, b_) for d_ in range(DEPTH_BANDS) for b_ in range(N_SHADES)
                  for f_ in range(3) for w_ in (3, 4)]]
              for layer in (0, 1)]
LIGHT = np.array([-0.05, -0.75, 0.65]) / np.linalg.norm([-0.05, -0.75, 0.65])   # ceiling lights, a bit in front
HALF = (LIGHT + [0, 0, 1]) / np.linalg.norm(LIGHT + [0, 0, 1])
GLOW_TINT = np.array([0.80, 0.93, 1.0])   # soft, pale gold glow (BGR)
GLOW_SCALE = tuple(float(v) for v in GLOW_TINT * GLOW * 2.0) + (0.0,)


# The eyes + eyebrows, nose and mouth area on the 3D face (mm), kept clear of hair and sparkles.
CLEAR_MODEL = np.array([[-31.5, -27, 5], [31.5, -27, 5], [-50, -17, -6], [50, -17, -6],     # brows
                        [-50, 8, -10], [50, 8, -10], [-30, 19, 0], [30, 19, 0],               # eye corners, under-eyes
                        [-14, 34, 9], [14, 34, 9], [0, 46, 15],                               # nose
                        [-37, 61, -10], [37, 61, -10], [-24, 77, -8], [24, 77, -8], [0, 80, -4]], float)   # mouth


def clear_zone(pose):
    """The eyes, nose and mouth area, in screen pixels: hair and sparkles stay off it.

    The area is laid on the 3D face, so it gets narrower on a turned head like the face does.
    Then it's moved to where the detector actually found the eyes, nose and mouth
    (pose["feat"]), and stretched to take in the nose tip and mouth corners, so it's still
    right when the head turns past what the 3D model follows (a profile)."""
    zone = project(CLEAR_MODEL, pose)
    feat = pose.get("feat")
    if feat is None:
        return cv2.convexHull(np.float32(zone)).reshape(-1, 2)
    feat = np.asarray(feat, float)
    zone = zone + (feat - project(FACE_MODEL, pose)).mean(0)
    u = pose["s"] * 2 * EYE_X                                     # the real eye gap in pixels
    ring = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
    extra = [feat[2] + ring * 0.13 * u, feat[3:] + 0.08 * u * np.array([[-1, 0], [1, 0]])]
    return cv2.convexHull(np.float32(np.vstack([zone] + extra))).reshape(-1, 2)


def _over(img, layer):
    """Lay a see-through layer (BGRA, color already multiplied by its opacity) over an image."""
    inv = cv2.bitwise_not(cv2.extractChannel(layer, 3))
    return cv2.add(cv2.multiply(img, cv2.merge([inv, inv, inv]), scale=1 / 255),
                   cv2.cvtColor(layer, cv2.COLOR_BGRA2BGR))


class Sparkles:
    """Gold specks that drift slowly upward around the hair, twinkle, and fade out."""

    def __init__(self, model, n, rng):
        self.model, self.rng, self.n = model, rng, n
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.age = np.zeros(n)
        self.life = np.ones(n)
        self.size = np.ones(n)
        self.tw = np.zeros(n)
        for i in range(n):
            self._spawn(i)
            self.age[i] = rng.uniform(0, self.life[i])     # don't all start at once

    def _spawn(self, i):
        rng = self.rng
        for _ in range(20):
            a = self.model.sparkle_anchors[rng.integers(len(self.model.sparkle_anchors))]
            out = a - HEAD_CENTER
            out[1] = min(out[1], 0) if a[1] < 60 else out[1] * 0.2
            out = out / (np.linalg.norm(out) + 1e-6)
            p = a + out * rng.uniform(12, 45) + rng.normal(0, 8, 3)
            if not (abs(p[0]) < 78 and -85 < p[1] < 130 and p[2] > -60):    # keep off the face
                break
        self.pos[i] = p
        self.vel[i] = [rng.normal(0, 4), -rng.uniform(4, 12), rng.normal(0, 3)]
        self.age[i] = 0
        self.life[i] = rng.uniform(1.6, 3.6)
        self.size[i] = rng.uniform(2.2, 4.5)
        self.tw[i] = rng.uniform(5, 11)

    def step(self, dt):
        self.age += dt
        self.pos += self.vel * dt
        for i in np.nonzero(self.age > self.life)[0]:
            self._spawn(i)

    def draw(self, layer, pose, ox, oy, t):
        """Draw onto a black layer (later added on top with a glow)."""
        X, Y, Z = rotate(self.pos, pose["yaw"], pose["pitch"], pose["roll"])
        s = pose["s"] * depth_scale(Z, pose)
        px, py = pose["x"] + s * X - ox, pose["y"] + s * Y - oy
        env = np.sin(np.pi * np.clip(self.age / self.life, 0, 1))
        bright = env * (0.55 + 0.45 * np.sin(self.tw * self.age + self.tw))
        on_face = (np.abs(self.pos[:, 0]) < 72) & (self.pos[:, 1] > -80) & (self.pos[:, 1] < 125) & (self.pos[:, 2] > -60)
        for i in range(self.n):
            b = bright[i]
            if b < 0.05 or on_face[i]:
                continue
            c = (int(185 * b), int(235 * b), int(255 * b))
            r = max(1.0, self.size[i] * s[i] * (0.6 + 0.4 * b))
            x, y = int(px[i] * 4), int(py[i] * 4)          # 4x sub-pixel precision
            R = int(r * 4 * 1.8)
            cv2.line(layer, (x - R, y), (x + R, y), c, 1, cv2.LINE_AA, 2)
            cv2.line(layer, (x, y - R), (x, y + R), c, 1, cv2.LINE_AA, 2)
            cv2.circle(layer, (x, y), max(4, int(r * 4 * 0.45)), (int(200 * b), int(245 * b), int(255 * b)),
                       -1, cv2.LINE_AA, 2)


class HairState:
    """The moving parts for one person's hair: where every strand point is right now
    (for the swing), the room-light match, and their sparkles."""

    def __init__(self, model):
        self.model = model
        self.pos = None
        self.prev = None
        self.light = None            # how much to brighten/darken and tint the hair (B, G, R)
        self.sparkles = Sparkles(model, SPARKLES, np.random.default_rng()) if SPARKLES else None

    def _targets(self, pose, t, K):
        m = self.model
        P = m.base[:K].copy()
        P[..., 0] += m.breeze_amp[:K] * np.sin(m.breeze_speed[:K] * t + m.breeze_phase[:K])
        # Long hair hangs: the hanging part only follows part of the head's tip and tilt.
        keep = 1 - GRAVITY * m.hang[:K]
        X, Y, Z = rotate(P, pose["yaw"], pose["pitch"] * keep, pose["roll"] * keep)
        k = pose["s"] * depth_scale(Z, pose)
        return np.stack([pose["x"] + k * X, pose["y"] + k * Y], -1), np.stack([X, Y, Z], -1)

    def _swing(self, target, dt, s):
        """Each point is pulled toward where the hairstyle says it should be. Roots are pulled
        all the way; tips only part way, so they trail behind and swing when you move."""
        K = len(target)
        if self.pos is None or len(self.pos) != K or np.abs(self.pos[:, 0] - target[:, 0]).max() > 120 * s:
            if self.pos is not None and len(self.pos) != K:     # detail level changed:
                keep = min(K, len(self.pos))                    # keep the strands we had
                pos, prev = target.copy(), target.copy()
                pos[:keep], prev[:keep] = self.pos[:keep], self.prev[:keep]
                self.pos, self.prev = pos, prev
            else:
                self.pos, self.prev = target.copy(), target.copy()
                return
        f = dt * 30.0                                        # same feel at any frame rate
        k = (1 - (1 - self.model.stiff[:K]) ** f)[..., None]
        damp = 0.82 ** f
        vel = (self.pos - self.prev) * damp
        self.prev = self.pos
        self.pos = self.pos + vel + k * (target - self.pos - vel)
        # never let a tip fall too far behind (stops wild swings at low frame rates)
        lag = self.pos - target
        n = np.linalg.norm(lag, axis=-1, keepdims=True)
        cap = 40 * s
        self.pos = np.where(n > cap, target + lag * cap / np.maximum(n, 1e-6), self.pos)

    def _shade(self, P3, K):
        """How lit each piece of each strand is (0 = deep shadow, 1 = bright shine).
        Worked out once per piece (N_SECTIONS per strand), since each piece gets one color."""
        m = self.model
        step = (N_POINTS - 1) // N_SECTIONS
        ends = P3[:, ::step]                                       # where each piece starts and ends
        T = np.diff(ends, axis=1)
        T = T / (np.linalg.norm(T, axis=-1, keepdims=True) + 1e-6)
        n = (ends[:, 1:] + ends[:, :-1]) / 2 - self._center
        n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)
        diffuse = np.sqrt(np.clip(1 - (T @ LIGHT) ** 2, 0, 1))
        # Hair has two shines: a bright white one, and a weaker golden one a bit further down.
        # Each strand's shine is nudged a little differently, so it breaks up like real hair
        # instead of making one smooth stripe.
        sh = m.spec_shift[:K, None, None]
        T1 = T + sh * n
        T1 /= np.linalg.norm(T1, axis=-1, keepdims=True) + 1e-6
        T2 = T - (0.15 - 0.5 * sh) * n
        T2 /= np.linalg.norm(T2, axis=-1, keepdims=True) + 1e-6
        spec1 = np.sqrt(np.clip(1 - (T1 @ HALF) ** 2, 0, 1)) ** 80
        spec2 = np.sqrt(np.clip(1 - (T2 @ HALF) ** 2, 0, 1)) ** 16
        facing = np.clip(n @ LIGHT, 0, 1)
        mid = slice(step // 2, N_POINTS - 1, step)                # the middle point of each piece
        hang, d = m.hang[:K, mid], m.d[:K, mid]
        lit = (0.3 + 0.36 * facing) * (1 - hang) + 0.78 * hang
        # Shadow: hair deeper inside the hair, the back layer, and the roots get less light
        ao = m.ao[:K, None] * np.where(m.front[:K, None], 1.0, 0.8) * (0.72 + 0.28 * np.clip(d / 0.06, 0, 1))
        shade = ao * (m.bias[:K, None] * (0.27 + 0.72 * diffuse * lit) + 0.3 * spec1 + 0.16 * spec2)
        return shade + 0.06 * d ** 2                                # sun-lightened ends

    def _match_light(self, frame, dt):
        """Follow the room: in a dim room the hair gets darker, and it takes on a little of the
        room's color cast, like everything else the camera sees. Changes slowly, so no flicker."""
        if MATCH_LIGHT <= 0:
            self.light = np.ones(3)
            return
        mean = np.array(cv2.mean(frame[::8, ::8])[:3])            # every 8th pixel is plenty
        luma = float(mean @ [0.114, 0.587, 0.299])
        gain = np.clip((luma / REF_BRIGHTNESS) ** 0.6, 0.55, 1.05)
        tint = np.clip((mean / max(mean.mean(), 1.0)) ** 0.3, 0.94, 1.06)
        want = 1 + MATCH_LIGHT * (gain * tint - 1)
        self.light = want if self.light is None else self.light + min(1.0, dt / 0.6) * (want - self.light)

    def _draw_layers(self, P3, K, pose, x0, y0, w, h):
        """Draw every strand into two see-through layers (back and front), SUPERSAMPLE times
        bigger than the screen, then shrink them. Returns the two layers as BGRA images."""
        m, s, SS = self.model, pose["s"], SUPERSAMPLE
        shade = self._shade(P3, K)
        q = np.round((self.pos - [x0, y0]) * (SS * 4)).astype(np.int32)     # 4x sub-pixel precision
        step = (N_POINTS - 1) // N_SECTIONS
        idx = np.arange(N_SECTIONS)[:, None] * step + np.arange(step + 1)[None, :]
        sec_pts = q[:, idx].reshape(-1, step + 1, 2)                          # every piece of every strand
        bucket = np.clip((shade * (N_SHADES - 1) + 0.5).astype(int), 0, N_SHADES - 1)

        # Sort every piece into a group: (layer, depth band, what it is, color tone, shade). Each group is
        # then drawn with ONE OpenCV call, which is much faster than one call per strand.
        kind = m.kind[:K, None]
        bucket = np.where(kind == 2, np.maximum(bucket - BODY_DARKER, 0), bucket)
        what = np.where(kind == 2, BODY_WIDTH_CLASS[None, :], np.where(kind == 1, 4, 3))
        fam = np.where(kind == 2, 0, m.family[:K, None])
        # When the head turns, the front hair on the far side goes behind the cheek, so it
        # moves to the back layer (the face is drawn back over it). Only the part below the
        # forehead moves: the hair on top of the forehead still covers the hairline.
        far = (m.side * pose["yaw"] < -FAR_SIDE_TURN)[:K, None]
        front_now = (m.front[:K, None] & ~(far & m.low[:K])).astype(int)
        band = np.where(kind == 2, 0, m.depth_band[:K, None])
        group = (((front_now * DEPTH_BANDS + band) * 5 + what) * 3 + fam) * N_SHADES + bucket
        group = np.where(what < 0, -1, group).ravel()
        order = np.argsort(group, kind="stable")
        bounds = np.searchsorted(group[order], np.arange(2 * DEPTH_BANDS * 5 * 3 * N_SHADES + 1))
        bw = max(2, int(round(BODY_MM * HAIR_VOLUME * s * SS)))
        # strands stay 1 pixel wide on the big canvas: OpenCV draws those much faster than wider
        # smooth lines (up close, SOFTEN fills them out instead)
        widths = [bw, max(1, round(bw * 0.6)), max(1, round(bw * 0.35)), 1, 1]

        used = set(np.nonzero(np.diff(bounds))[0].tolist())     # groups that have pieces in them

        # The solid cap and the part strip go under everything. The front of the cap goes with
        # the front hair, except on the far side of a turned head (that hair is behind the cheek).
        cq = np.round(project(m.cap, pose, x0, y0) * (SS * 4)).astype(np.int32)       # (2, U, T, 2)
        quads = np.stack([cq[:, :-1, :-1], cq[:, 1:, :-1], cq[:, 1:, 1:], cq[:, :-1, 1:]], 3)
        far = [side * pose["yaw"] < -FAR_SIDE_TURN for side in (-1, 1)]
        pl = np.round(project(m.part_line, pose, x0, y0) * (SS * 4)).astype(np.int32)
        part_w = max(2, int(round(PART_MM * s * SS)))
        n_front = m.cap_front_rows

        layers = []
        for layer in (0, 1):
            canvas = np.zeros((h * SS, w * SS, 4), np.uint8)
            for si in (0, 1):
                rows = CAP_U - 1 if layer == 0 else n_front
                cols = CAP_T - 1 if layer == 0 or not far[si] else m.cap_high_cols
                for quad in quads[si, :rows, :cols].reshape(-1, 4, 2):
                    cv2.fillConvexPoly(canvas, quad, CAP_COLOR, cv2.LINE_8, 2)
            line = pl if layer == 0 else pl[:int(CAP_FRONT_U * (len(pl) - 1)) + 1]
            cv2.polylines(canvas, [line], False, PART_COLOR, part_w, cv2.LINE_AA, 2)
            # Lock bodies first (the dark inside of the hair), then strands from darkest to
            # brightest, so the shiny ones end up on top.
            for g, what_, f, b in DRAW_ORDER[layer]:
                if g in used:
                    # lock bodies sit under the strands, so they don't need smooth edges (faster)
                    cv2.polylines(canvas, list(sec_pts[order[bounds[g]:bounds[g + 1]]]), False,
                                  (BODY_COLORS if what_ < 3 else COLORS[f])[b], widths[what_],
                                  cv2.LINE_8 if what_ < 3 else cv2.LINE_AA, 2)
            small = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
            sigma = SOFTEN * min(max(s - 0.55, 0.0), 0.65)
            if sigma > 0.15:
                small = cv2.GaussianBlur(small, (0, 0), sigma)
            layers.append(small)
        return layers

    def render(self, frame, pose, alpha, t, dt, debug=False, detail=1.0):
        if alpha <= 0.01:
            return frame
        m = self.model
        s = pose["s"]
        if "f" not in pose:                     # the camera's focal length, for real-depth drawing
            pose = dict(pose, f=FOCAL_PX * frame.shape[1] / 640)
        # Level of detail: a small, far-away head gets fewer strands (they'd land on the same
        # pixels anyway), and "detail" (set by the speed auto-adjust) thins it further.
        K = int(len(m.base) * np.clip(s / 0.45, 0.75, 1.0) * detail)
        K = max(K, m.n_bodies)
        target, P3 = self._targets(pose, t, K)
        self._center = np.array(rotate(HEAD_CENTER, pose["yaw"], pose["pitch"], pose["roll"]))
        self._swing(target, dt, s)
        pts = self.pos
        if self.sparkles is not None:
            self.sparkles.step(dt)
        self._match_light(frame, dt)

        # Work only inside the box the hair covers (much faster than the whole frame)
        H, W = frame.shape[:2]
        pad = int(35 * s) + 6
        x0 = max(int(pts[..., 0].min()) - pad, 0)
        y0 = max(int(pts[..., 1].min()) - pad, 0)
        x1 = min(int(pts[..., 0].max()) + pad, W)
        y1 = min(int(pts[..., 1].max()) + pad, H)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return frame
        orig = frame[y0:y1, x0:x1].copy()
        roi = orig.copy()
        h, w = roi.shape[:2]

        back, front = self._draw_layers(P3, K, pose, x0, y0, w, h)
        gain = tuple(float(v) for v in self.light) + (1.0,)
        back, front = cv2.multiply(back, gain), cv2.multiply(front, gain)

        # 1. back layer
        roi = _over(roi, back)
        # 2. put the face and neck back on top of the back layer (soft edges)
        mask = np.zeros(roi.shape[:2], np.uint8)
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ca, sa = np.cos(a), np.sin(a)
        # a "squarish oval" for the face, a little narrower at the temples (people have hair there)
        sq = np.where(sa < 0, 0.7, 0.85)
        half_w = np.where(sa < 0, 60, 68)
        oval = np.stack([half_w * np.sign(ca) * np.abs(ca) ** sq, 28 + np.where(sa < 0, 80, 90) * np.sign(sa) * np.abs(sa) ** sq,
                         -5 - 30 * ca ** 2], 1)
        neck = np.array([[-52, 90, -40], [52, 90, -40], [58, 300, -60], [-58, 300, -60]], float)
        for shape, keep in ((oval, 1.0), (neck, 1 - GRAVITY)):
            poly = project(shape, pose, x0, y0, keep)
            cv2.fillPoly(mask, [np.round(poly * 4).astype(np.int32)], 255, cv2.LINE_AA, 2)
        k = int(9 * s) * 2 + 1
        bx, by, bw, bh = cv2.boundingRect(cv2.dilate(mask, None, iterations=k // 2 + 1))
        if bw > 0 and bh > 0:                                   # blend only inside the face's box
            sub = (slice(by, by + bh), slice(bx, bx + bw))
            soft = cv2.GaussianBlur(mask[sub], (k, k), 0).astype(np.float32) / 255
            roi[sub] = cv2.blendLinear(orig[sub], roi[sub], soft, 1 - soft)
        # 3. the soft shadow the front hair casts on the face, the neck and the hair behind it
        a_front = cv2.extractChannel(front, 3)
        if FACE_SHADOW > 0 and h > 8:
            small = cv2.resize(a_front, ((w + 3) // 4, (h + 3) // 4), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), max(0.8, s))       # about 4 mm soft
            shadow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            dy = max(1, int(round(4 * s)))                             # light from above: shadow falls down
            shadow = np.vstack([np.zeros((dy, w), np.uint8), shadow[:-dy]])
            keep = cv2.bitwise_not(cv2.convertScaleAbs(shadow, alpha=FACE_SHADOW))
            roi = cv2.multiply(roi, cv2.merge([keep, keep, keep]), scale=1 / 255)
        # 4. front layer
        roi = _over(roi, front)
        # 5. camera grain on the hair, so it has the same fine noise as the rest of the picture
        if GRAIN > 0:
            noise = np.empty((h, w), np.float32)
            cv2.randn(noise, 0, GRAIN)
            noise *= cv2.max(a_front, cv2.extractChannel(back, 3)).astype(np.float32) / 255
            roi = cv2.add(roi.astype(np.float32), cv2.merge([noise, noise, noise])).clip(0, 255).astype(np.uint8)
        # 6. a light glow: blur a small copy of just the shiny parts and add it back ("screen" blend)
        if GLOW > 0:                     # (all done on a quarter-size copy, then scaled back up)
            size = (w // 4 + 1, h // 4 + 1)
            shiny = cv2.max(cv2.cvtColor(front, cv2.COLOR_BGRA2BGR), cv2.cvtColor(back, cv2.COLOR_BGRA2BGR))
            small = cv2.resize(shiny, size, interpolation=cv2.INTER_AREA)
            small = cv2.subtract(small, (85, 85, 85, 0))            # only the bright parts glow
            small = cv2.GaussianBlur(small, (0, 0), max(1.5, 8 * s / 4))
            small = cv2.multiply(small, GLOW_SCALE)                 # warm tint and strength
            glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            # "screen" blend: brightens, never goes past white
            roi = cv2.add(roi, cv2.multiply(glow, cv2.bitwise_not(roi), scale=1 / 255))

        # 7. sparkles, with their own little glow
        if self.sparkles is not None:
            sp = np.zeros_like(roi)
            self.sparkles.draw(sp, pose, x0, y0, t)
            sp = cv2.multiply(sp, cv2.merge([cv2.bitwise_not(mask)] * 3), scale=1 / 255)   # never on the face
            halo = cv2.GaussianBlur(sp, (0, 0), max(1.2, 3 * s))
            sp = cv2.add(sp, cv2.addWeighted(halo, 2.2, halo, 0, 0))
            roi = cv2.add(roi, sp)

        # 8. keep the eyes, nose and mouth clear: no hair, glow or sparkles over them
        if CLEAR_FACE > 0:
            zone = clear_zone(pose) - [x0, y0]
            cmask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(cmask, np.round(zone * 4).astype(np.int32), 255, cv2.LINE_AA, 2)
            sig = max(1.0, 0.07 * s * 2 * EYE_X)                      # soft edge, about 4 mm
            pad = int(3 * sig) + 1
            bx, by, bw, bh = cv2.boundingRect(cmask)
            bx, by = max(bx - pad, 0), max(by - pad, 0)
            bw, bh = min(bw + 2 * pad, w - bx), min(bh + 2 * pad, h - by)
            if bw > 0 and bh > 0:
                sub = (slice(by, by + bh), slice(bx, bx + bw))
                soft = cv2.GaussianBlur(cmask[sub], (0, 0), sig).astype(np.float32) * (CLEAR_FACE / 255)
                roi[sub] = cv2.blendLinear(orig[sub], roi[sub], soft, 1 - soft)

        # 9. fade in / out
        if alpha < 1:
            roi = cv2.addWeighted(roi, alpha, orig, 1 - alpha, 0)
        frame[y0:y1, x0:x1] = roi
        if debug:
            cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (80, 80, 80), 1)
        return frame


# =====================================  WHO GETS IT  =====================================
class Picker:
    """Decides who the opt-in person is, and keeps the filter locked on them.

    - Faces that are too small (too far away) or near the edges of the screen are ignored.
    - A new face must show up for a few frames in a row before hair appears (so a one-frame
      false detection never flashes hair on a poster).
    - Once locked on, the filter stays with that person. Someone else only takes it over if they
      are clearly closer (bigger) for a while, or the person leaves.
    - Team rule: hair is only drawn on a frame where the person's face passed the team checks
      (inside 5 ft, turned less than 50 degrees). There is no fade-out, so hair never lingers
      on someone who just failed a check or walked off.
    """

    def __init__(self, model):
        self.model = model
        self.target = None
        self.challenger_nose = None
        self.challenge = 0

    def eligible(self, raw, w):
        return raw["gap"] >= MIN_EYE_GAP_PX and ZONE_LEFT * w <= raw["nose"][0] <= ZONE_RIGHT * w

    def update(self, raws, w, dt):
        cands = [r for r in raws if self.eligible(r, w)]
        if self.target is not None:
            T = self.target
            best, best_d = None, float("inf")
            for r in cands:
                if r.get("criteria_id") != T.criteria_id:   # must be the same face the team rules tracked
                    continue
                d = np.hypot(*(r["nose"] - T.nose))
                if d < best_d:
                    best, best_d = r, d
            if best is not None and best_d <= TRACK_RADIUS * max(T.gap, best["gap"]):
                T.update(best, dt)
                cands = [r for r in cands if r is not best]
            else:
                # The face didn't pass the team checks this frame, or the detector blinked.
                # No hair this frame (draw() skips it). A 1-2 frame blink keeps the smoothing,
                # so the hair comes straight back; any longer and the target is dropped below.
                T.miss(dt)
            # Is somebody else clearly closer, for a while?
            big = max(cands, key=lambda r: r["gap"], default=None)
            if big is not None and big["gap"] > SWITCH_RATIO * T.gap and (
                    self.challenger_nose is None or
                    np.hypot(*(big["nose"] - self.challenger_nose)) < TRACK_RADIUS * big["gap"]):
                self.challenge += 1
                self.challenger_nose = big["nose"]
            else:
                self.challenge, self.challenger_nose = 0, None
            if self.challenge >= SWITCH_FRAMES:
                self.target = Target(big, self.model)
                self.target.seen = ACQUIRE_FRAMES
                self.challenge, self.challenger_nose = 0, None
            elif T.misses > 2:                   # gone for more than a blink: drop, no fade-out
                self.target = None
        if self.target is None and cands:
            self.target = Target(max(cands, key=lambda r: r["gap"]), self.model)

    def draw(self, frame, t, dt, debug=False, detail=1.0):
        # Team rule: only a face that passed this frame's checks gets hair (misses == 0).
        T = self.target
        if T is not None and T.acquired and T.misses == 0:
            T.hair.render(frame, T.pose, T.alpha, t, dt, debug, detail)
        return frame


def draw_debug(frame, raws, picker, fps, detail):
    h, w = frame.shape[:2]
    for xz in (ZONE_LEFT, ZONE_RIGHT):
        cv2.line(frame, (int(xz * w), 0), (int(xz * w), h), (90, 90, 90), 1)
    T = picker.target
    for r in raws:
        mine = T is not None and np.hypot(*(r["nose"] - T.nose)) < 1
        ok = picker.eligible(r, w)
        color = (0, 255, 0) if mine else ((0, 200, 255) if ok else (0, 0, 255))
        for (x, y) in r["pts"]:
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)
        label = "TARGET" if mine else ("ok" if ok else "ignored")
        label += f" gap {r['gap']:.0f}px yaw {math.degrees(r['yaw']):+.0f} pitch {math.degrees(r['pitch']):+.0f}"
        label += "" if r["err"] <= FIT_ERROR_LIMIT else " (bad fit)"
        cv2.putText(frame, label, (int(r["nose"][0]) - 70, int(r["nose"][1]) + int(r["gap"] * 2.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    cv2.putText(frame, f"{fps:.0f} fps   hair detail {detail * 100:.0f}%", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def make_detector():
    """Load YuNet. Newer OpenCV (5.x) prints a harmless "setPreferableTarget ... not supported by
    the new graph engine" warning while loading it, so warnings are muted just for this step."""
    log = getattr(getattr(cv2, "utils", None), "logging", None)
    old = None
    try:
        old = log.getLogLevel()
        log.setLogLevel(log.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    try:
        return cv2.FaceDetectorYN.create(MODEL_PATH, "", (640, 480), score_threshold=SCORE_THRESHOLD)
    finally:
        if old is not None:
            log.setLogLevel(old)


class GoldilocksFilter:
    """Everything Goldilocks does to one camera frame, in one place. main() below calls it for
    every webcam frame, and the hallway kiosk calls the very same process() method, so the
    filter behaves the same in both."""

    def __init__(self, calibration=None, detector=None):
        load_model()
        self.detector = detector if detector is not None else make_detector()
        print(f"{FILTER_NAME}: building the hair model...")
        self.model = HairModel(SEED)
        self.picker = Picker(self.model)
        # The kiosk passes in the one calibration all four filters share.
        self.calibration = calibration if calibration is not None else DistanceCalibration()
        self.criteria = FaceCriteria(self.calibration)
        self.show_debug = False
        self.t0 = self.last = None
        self.fps = float(TARGET_FPS)
        self.detail, self.hair_ms = 1.0, 0.0

    def recalibrate(self, now):
        """Start a new 5-foot calibration and forget everyone being tracked."""
        self.calibration.begin(now)
        self.reset()

    def reset(self):
        """Forget everyone being tracked (used when switching to this filter in the kiosk)."""
        self.criteria = FaceCriteria(self.calibration)
        self.picker = Picker(self.model)
        self.last = None

    def process(self, frame, show_status=True):
        """frame: a MIRRORED 640x480 camera frame (like a selfie). Draws on it and returns it."""
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        dt = 1 / TARGET_FPS if self.last is None else min(max(now - self.last, 1 / 60), 0.25)
        self.last = now
        self.fps = 0.9 * self.fps + 0.1 / dt

        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        mono = time.monotonic()
        # Team rules: update the calibration, then find which faces may get hair this frame.
        # Every other detected face is added to the blur list.
        self.calibration.update(faces, frame.shape, mono)
        allowed = self.criteria.update(faces, frame.shape, mono)
        raws = []
        for track_id, row in allowed:
            pts = face_points(row)
            if not (0 <= pts[2][0] < w and 0 <= pts[2][1] < h):   # nose off screen: skip
                continue
            try:
                raw = estimate_pose(pts)
            except (ValueError, OverflowError, np.linalg.LinAlgError):
                raw = None
            if raw is not None and np.isfinite([raw[k] for k in ("x", "y", "s", "yaw", "pitch", "roll", "err")]).all():
                raw["pts"] = pts
                raw["criteria_id"] = track_id
                raws.append(raw)
            else:
                self.criteria.blur_boxes.append(row[:4].copy())   # can't measure it: blur it

        self.picker.update(raws, w, dt)
        t_hair = time.perf_counter()
        self.picker.draw(frame, now - self.t0, dt, self.show_debug, self.detail)
        self.hair_ms = 0.9 * self.hair_ms + 0.1 * (time.perf_counter() - t_hair) * 1000

        # Speed auto-adjust: too slow, and the hair is a real part of the work? Thin it a little.
        # Plenty fast? Bring detail back a little. (Small steps, so you never see it jump.)
        if self.fps < TARGET_FPS and self.hair_ms > 0.25 * 1000 / max(self.fps, 1):
            self.detail = max(MIN_DETAIL, self.detail * 0.96)
        elif self.fps > TARGET_FPS * 1.3:
            self.detail = min(1.0, self.detail * 1.02)

        # Blur last, so no hair or sparkle can cover up a face that should be blurred.
        frame = blur_faces(frame, self.criteria.blur_boxes)
        if self.show_debug:
            draw_debug(frame, raws, self.picker, self.fps, self.detail)
        if show_status:
            draw_calibration_status(frame, self.calibration)
        return frame


def main():
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    goldilocks = GoldilocksFilter()
    print("Running. Q: quit. D: debug. C: calibrate at 5 feet. Saved calibration loads automatically.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)                      # mirror view, like a selfie camera
        frame = goldilocks.process(frame)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key in (ord('d'), ord('D')):
            goldilocks.show_debug = not goldilocks.show_debug
        if key in (ord('c'), ord('C')):
            goldilocks.recalibrate(time.monotonic())

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
