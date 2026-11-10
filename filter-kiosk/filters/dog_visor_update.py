"""The Visor filter - beginner-friendly prototype.

Put this file, dog visor.png, and smile_distance_calibration.json together.
Uses saved 5-foot calibration and +/-50 degree yaw. Far/uncertain faces blur.
C recalibrates; Q quits. No distance labels are shown once calibrated.
Distances are approximate; undetected faces cannot be guaranteed blurred.
Install dependencies if needed:
    pip install opencv-python mediapipe numpy
"""
import cv2
import numpy as np
import math
import os
import sys
import time
import json
import urllib.request
from collections import deque
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The supplied asset is named dog visor.png; retain support for the old name.
VISOR_FILE = Path(__file__).with_name("dog visor.png")
if not VISOR_FILE.exists():
    VISOR_FILE = Path(__file__).with_name("a_clean_transparent_background_png_style_image_w.png")
CAMERA_NUMBER = 0
SMOOTHING_FRAMES = 5
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")
MODEL_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")

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


def load_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    print("Downloading YuNet face model (one time only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if os.path.getsize(MODEL_PATH) < 100_000:
        raise RuntimeError("YuNet download was incomplete. Put the model beside this script.")


def load_visor_image(path):
    """Accept transparent artwork or the supplied white-background dog image.

    For RGB artwork, key only near-white pixels connected to the image border.
    This happens in memory; the original asset file is never changed.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Keep dog visor.png beside this script.")
    if image.shape[2] == 4:
        return image
    white = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
    padded = cv2.copyMakeBorder(white, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    cv2.floodFill(padded, None, (0, 0), 128)
    alpha = np.where(padded[1:-1, 1:-1] == 128, 0, 255).astype(np.uint8)
    return np.dstack([image, alpha])


# Store a few recent eye positions so the visor does not jump as much.
recent_eye_centers = {}
face_geometry = {}  # Stores the last stable eye-to-nose distances and visor shape.


def smooth_point(face_id, point):
    """Return the average of the latest eye-center positions for this face."""
    if face_id not in recent_eye_centers:
        recent_eye_centers[face_id] = deque(maxlen=SMOOTHING_FRAMES)
    recent_eye_centers[face_id].append(point)
    x = int(sum(p[0] for p in recent_eye_centers[face_id]) / len(recent_eye_centers[face_id]))
    y = int(sum(p[1] for p in recent_eye_centers[face_id]) / len(recent_eye_centers[face_id]))
    return x, y


def overlay_transparent(background, foreground, center_x, center_y):
    """Place an RGBA image over a BGR camera frame, clipping at the edges."""
    fh, fw = foreground.shape[:2]
    x1 = center_x - fw // 2
    y1 = center_y - fh // 2
    x2 = x1 + fw
    y2 = y1 + fh

    # Clip the overlay to the camera frame so array indexes stay valid.
    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(background.shape[1], x2), min(background.shape[0], y2)
    if bx1 >= bx2 or by1 >= by2:
        return background

    fx1, fy1 = bx1 - x1, by1 - y1
    fx2, fy2 = fx1 + (bx2 - bx1), fy1 + (by2 - by1)
    overlay = foreground[fy1:fy2, fx1:fx2]
    roi = background[by1:by2, bx1:bx2]
    alpha = overlay[:, :, 3:4].astype(float) / 255.0
    roi[:] = (alpha * overlay[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    return background


def rotate_rgba(image, angle_degrees):
    """Rotate the transparent visor without cropping its corners."""
    height, width = image.shape[:2]
    new_width = int(abs(height * math.sin(math.radians(angle_degrees))) + abs(width * math.cos(math.radians(angle_degrees))))
    new_height = int(abs(height * math.cos(math.radians(angle_degrees))) + abs(width * math.sin(math.radians(angle_degrees))))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, 1.0)
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    return cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))




def render_visor(frame, face, face_id, visor_image):
    frame_height, frame_width = frame.shape[:2]
    # Face Mesh points 33 and 263 are the outside corners of the eyes.
    left = face.landmark[33]
    right = face.landmark[263]
    nose = face.landmark[1]
    lx, ly = int(left.x * frame_width), int(left.y * frame_height)
    rx, ry = int(right.x * frame_width), int(right.y * frame_height)
    nx, ny = int(nose.x * frame_width), int(nose.y * frame_height)

    # Do not guess if either eye is outside the visible frame.
    if not (0 <= lx < frame_width and 0 <= rx < frame_width and
            0 <= ly < frame_height and 0 <= ry < frame_height):
        return frame

    eye_distance = math.hypot(rx - lx, ry - ly)
    # Range is checked by calibrated YuNet measurements; this is only a
    # minimum usable MediaPipe landmark spacing.
    if eye_distance < 12:
        return frame

    center_x = (lx + rx) // 2
    center_y = (ly + ry) // 2
    center_x, center_y = smooth_point(face_id, (center_x, center_y))
    angle = math.degrees(math.atan2(ry - ly, rx - lx))

    # Check each eye against the nose, not just against the other eye.
    left_nose_distance = math.hypot(lx - nx, ly - ny)
    right_nose_distance = math.hypot(rx - nx, ry - ny)
    current_distances = (left_nose_distance, right_nose_distance)
    previous = face_geometry.get(face_id)
    drift_detected = False
    if previous is None:
        face_geometry[face_id] = {
            "distances": current_distances,
            "width": int(eye_distance * 2.8),
            "angle": angle,
            "eye_distance": eye_distance,
        }
    else:
        old_left, old_right = previous["distances"]
        left_change = abs(left_nose_distance - old_left) / max(old_left, 1)
        right_change = abs(right_nose_distance - old_right) / max(old_right, 1)
        # If both eye-to-nose distances shift a lot, trust the nose
        # for position and keep the last stable width and angle.
        if left_change > 0.25 and right_change > 0.25:
            drift_detected = True
            center_x, center_y = nx, ny
            eye_distance = previous["eye_distance"]
            angle = previous["angle"]
        else:
            face_geometry[face_id] = {
                "distances": current_distances,
                "width": int(eye_distance * 2.8),
                "angle": angle,
                "eye_distance": eye_distance,
            }

    # The asset is wide, so its width is set from the eye distance.
    target_width = int(eye_distance * 2.8)
    target_height = max(1, int(visor_image.shape[0] * target_width / visor_image.shape[1]))
    if drift_detected:
        target_width = face_geometry[face_id]["width"]
        target_height = max(1, int(visor_image.shape[0] * target_width / visor_image.shape[1]))
    resized_visor = cv2.resize(visor_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    rotated_visor = rotate_rgba(resized_visor, angle)

    # Normally align with the eyes; if both eyes drift, center near the nose.
    visor_y = center_y - int(eye_distance * 0.12) if not drift_detected else center_y - int(eye_distance * 0.65)
    frame = overlay_transparent(frame, rotated_visor, center_x, visor_y)
    return frame


def mesh_box(face, frame_shape):
    h, w = frame_shape[:2]
    points = np.array([(p.x * w, p.y * h) for p in face.landmark], dtype=float)
    if not np.isfinite(points).all():
        return None
    lo, hi = points.min(axis=0), points.max(axis=0)
    return np.r_[lo, hi - lo]


def match_meshes(meshes, detected, frame_shape):
    """Mutual one-to-one matches; ambiguous faces never inherit a clear state."""
    costs = np.full((len(meshes), len(detected)), np.inf)
    h, w = frame_shape[:2]
    for i, face in enumerate(meshes):
        if mesh_box(face, frame_shape) is None:
            continue
        nose = face.landmark[1]
        point = np.array([nose.x * w, nose.y * h])
        for j, (_, row, _) in enumerate(detected):
            x, y, fw, fh = row[:4]
            if x <= point[0] <= x + fw and y <= point[1] <= y + fh:
                cost = np.linalg.norm(point - row[8:10]) / max(np.hypot(fw, fh), 1)
                if np.isfinite(cost) and cost < 0.25:
                    costs[i, j] = cost
    result = {}
    if not meshes or not detected:
        return result
    for i in range(len(meshes)):
        j = int(np.argmin(costs[i]))
        best = costs[i, j]
        if not np.isfinite(best) or int(np.argmin(costs[:, j])) != i:
            continue
        row, col = np.sort(costs[i]), np.sort(costs[:, j])
        if ((len(row) > 1 and row[1] - best < 0.06)
                or (len(col) > 1 and col[1] - best < 0.06)):
            continue
        result[i] = detected[j]
    return result


def main():
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh  # Same FaceMesh dependency as the original.
    visor_image = load_visor_image(VISOR_FILE)
    load_model()
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (640, 480), score_threshold=0.7)
    calibration = DistanceCalibration()
    criteria = FaceCriteria(calibration)
    camera = cv2.VideoCapture(CAMERA_NUMBER)
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera. Check camera permissions.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Q: quit. C: calibrate alone at 5 feet. Saved calibration loads automatically.")
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=3, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                success, frame = camera.read()
                if not success:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)
                now = time.monotonic()
                calibration.update(faces, frame.shape, now)
                criteria.update(faces, frame.shape, now)
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                meshes = results.multi_face_landmarks or []
                # Match using actual YuNet landmarks, including denied faces.
                detected = [(key, track['face'], track['valid'])
                            for key, track in criteria.tracks.items() if track['time'] == now]
                matches = match_meshes(meshes, detected, frame.shape)
                active = set()
                for i, mesh in enumerate(meshes):
                    match = matches.get(i)
                    if match is not None and match[2]:
                        track_id = match[0]
                        active.add(track_id)
                        frame = render_visor(frame, mesh, track_id, visor_image)
                    else:
                        box = mesh_box(mesh, frame.shape)
                        if box is not None:
                            criteria.blur_boxes.append(box)
                for storage in (recent_eye_centers, face_geometry):
                    for key in list(storage):
                        if key not in active:
                            del storage[key]
                # Blur last; no overlay can restore background face pixels.
                frame = blur_faces(frame, criteria.blur_boxes)
                draw_calibration_status(frame, calibration)
                cv2.imshow("The Visor - press Q to quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    break
                if key in (ord('c'), ord('C')):
                    calibration.begin(now)
                    criteria = FaceCriteria(calibration)
                    recent_eye_centers.clear()
                    face_geometry.clear()
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
