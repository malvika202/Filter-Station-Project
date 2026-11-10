"""Clown nose filter with +/-50 degree yaw and calibrated 5-foot range checks.

Keep clown_nose.png and smile_distance_calibration.json beside this script.
The saved calibration is compatible with the smile filter at the same camera,
resolution and zoom. C recalibrates; Q quits. No distance labels are displayed
once calibrated. Detected far-away or uncertain faces are blurred; only the
largest eligible face receives the clown nose, as in the original filter.

Single-webcam distance and generic-face pose estimates are approximate.
Undetected faces cannot be reliably blurred; this is not guaranteed anonymity.
"""
import math
import json
import time
import os
import sys
import urllib.request

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "face_detection_yunet_2023mar.onnx"
)

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

NOSE_PATH = os.path.join(SCRIPT_DIR, "clown_nose.png")

NOSE_SCALE = 0.9
MIN_MODEL_BYTES = 100000


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
# screen-left anchors corner, screen-right anchors corner in a frontal view.
# Camera convention: x right, y down, z away; nose projects toward camera.
# Individual facial proportions introduce uncertainty even with calibration.
POSE_POINTS = np.array([
    [-32.0, -32.0, 0.0], [32.0, -32.0, 0.0],
    [0.0, 0.0, -30.0],
    [-25.0, 32.0, -5.0], [25.0, 32.0, -5.0],
], dtype=np.float64)


def estimate_yaw(face, frame_shape):
    """Return yaw in degrees, or None for a failed/unreliable five-point fit.

    YuNet runs on the original, unmirrored camera frame. Use its landmark
    identities directly; do not sort by x, since identity must survive roll.
    """
    h, w = frame_shape[:2]
    points = np.asarray(face[4:14], dtype=np.float64).reshape(5, 2).copy()
    points = np.ascontiguousarray(points)
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


class NoseTracker:
    """Associate faces geometrically; never share smoothing across detections.

    Missing detections have no nose; blur the last box for at most 0.25 seconds
    for reacquisition. Ambiguous matches start fresh to avoid carrying a anchors
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
            # Gate the CURRENT measurement: smoothing cannot leave the nose on
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
            anchors = face[4:10].reshape(3, 2).copy()
            if (valid and previous is not None and previous['valid']
                    and now - previous['time'] <= 0.10):
                dt = max(now - previous['time'], 0.001)
                alpha = 1.0 - math.exp(-dt / 0.055)
                # Follow sudden real movement immediately; smooth small jitter.
                jump = np.linalg.norm(anchors.mean(axis=0) - previous['anchors'].mean(axis=0))
                if jump < 0.12 * face[2]:
                    anchors = alpha * anchors + (1.0 - alpha) * previous['anchors']
            self.tracks[key] = dict(box=face[:4].copy(), anchors=anchors,
                                    time=now, valid=valid, clear=clear, distance=distance)
            if valid:
                eye_span = float(np.linalg.norm(anchors[1] - anchors[0]))
                overlays.append((anchors[2].copy(), eye_span, face[:4].copy()))
        for key, track in self.tracks.items():
            if key not in seen:
                self.blur_boxes.append(track['box'])
                # Reacquired faces must qualify for the inner entry threshold.
                track['clear'] = False
                track['valid'] = False
                track['distance'] = None
        return overlays


def model_ok() -> bool:
    return (
        os.path.exists(MODEL_PATH)
        and os.path.getsize(MODEL_PATH) > MIN_MODEL_BYTES
    )


def load_model() -> bool:
    if model_ok():
        return True

    print("Downloading YuNet face model (one time only)...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    except Exception as e:
        print(f"Error: download failed: {e}")

        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            print(
                "Fix (Mac): open Applications > Python 3.x and "
                "double-click 'Install Certificates.command', "
                "then run this again."
            )

        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        return False

    if not model_ok():
        print("Error: downloaded model file is too small.")

        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        return False

    print("Done.")
    return True


def overlay_nose(
    frame: np.ndarray,
    nose_img: np.ndarray,
    nose_center: tuple,
    eye_dist: float
) -> np.ndarray:

    n_h, n_w = nose_img.shape[:2]

    target_w = int(eye_dist * NOSE_SCALE)

    if target_w < 2:
        return frame

    target_h = max(
        1,
        int(n_h * target_w / n_w)
    )

    interp = (
        cv2.INTER_AREA
        if target_w < n_w
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        nose_img,
        (target_w, target_h),
        interpolation=interp
    )

    x = int(nose_center[0] - target_w / 2)
    y = int(nose_center[1] - target_h / 2)

    fh, fw = frame.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)

    x2 = min(x + target_w, fw)
    y2 = min(y + target_h, fh)

    if x1 >= x2 or y1 >= y2:
        return frame

    nose_crop = resized[
        y1 - y:y2 - y,
        x1 - x:x2 - x
    ]

    roi = frame[y1:y2, x1:x2].astype(np.float32)

    alpha = nose_crop[:, :, 3:4].astype(np.float32) / 255.0

    blended = (
        roi * (1 - alpha)
        + nose_crop[:, :, :3].astype(np.float32) * alpha
    )

    frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    return frame


def main():

    major, minor = (
        int(p)
        for p in cv2.__version__.split(".")[:2]
    )

    if (major, minor) < (4, 8):
        print(
            f"Error: OpenCV {cv2.__version__} is too old. "
            "Run: pip install --upgrade opencv-python"
        )
        return

    if not load_model():
        return

    nose_img = cv2.imread(
        NOSE_PATH,
        cv2.IMREAD_UNCHANGED
    )

    if nose_img is None:
        print(f"Error: Could not load {NOSE_PATH}")
        return

    if nose_img.ndim != 3 or nose_img.shape[2] != 4:
        print(
            "Error: clown_nose.png must be a PNG "
            "with transparency (4 channels)."
        )
        return

    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    elif sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (640, 480),
        score_threshold=0.8
    )

    print("Running. Saved 5-foot calibration loads automatically.")
    print("C: calibrate alone at 5 feet. Q: quit. Distances are approximate.")
    calibration = DistanceCalibration()
    tracker = NoseTracker(calibration)

    failed_reads = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            failed_reads += 1

            if failed_reads > 30:
                print("Error: Failed to grab frames from the webcam.")
                break

            continue

        failed_reads = 0

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))

        _, faces = detector.detect(frame)

        now = time.monotonic()
        calibration.update(faces, frame.shape, now)
        overlays = tracker.update(faces, frame.shape, now)
        frame = blur_faces(frame, tracker.blur_boxes)

        # Preserve the original single-nose behavior: largest eligible face.
        if overlays:
            nose_center, eye_dist, box = max(overlays, key=lambda item: item[2][2] * item[2][3])
            frame = overlay_nose(frame, nose_img, nose_center, eye_dist)
            x, y, fw, fh = map(int, box)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        # Keep the calibrated view free of distance labels and status banners.
        if calibration.reference is None or calibration.active:
            cv2.rectangle(frame, (0, max(0, h - 48)), (w, h), (25, 25, 25), -1)
            cv2.putText(frame, calibration.message, (8, h - 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'C: calibrate at 5 ft | Q: quit', (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Clown Face Filter", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("c"), ord("C")):
            calibration.begin(time.monotonic())
            tracker = NoseTracker(calibration)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
