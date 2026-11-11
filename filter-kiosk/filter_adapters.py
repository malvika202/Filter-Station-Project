"""Adapters: run each team member's filter one frame at a time, inside the kiosk.

Every filter file has its own main() with its own camera loop and window, so the kiosk can't just
call main(). Each adapter below does exactly what that filter's main() loop does for ONE frame,
using the filter's own functions and classes, in the same order. The Smile file (Malvika's) is never
changed: it's loaded from the filters folder as it is. Goldilocks v3, Clown Nose v2 and Adoggy v2
(changed with the team's permission) each have a process() method that their own main() uses too,
so their adapters just call it.

Every adapter has the same three parts:
    process(frame)  takes a raw (not mirrored) 640x480 camera frame and returns the finished,
                    mirrored frame for the screen
    reset()         forgets everyone being tracked (called when someone switches to this filter)
    engaged         True if the filter was drawn on someone in the last frame

All four share ONE DistanceCalibration (the team's 5-foot calibration), so calibrating once sets
the range for every filter.
"""
import importlib.util
import os
import sys
import time

import cv2

FILTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filters")
MODEL_FILE = "face_detection_yunet_2023mar.onnx"
CALIBRATION_FILE = "smile_distance_calibration.json"

# The four filters, in the order the cards appear. "file" is the filter's Python file in the
# filters folder. If a teammate sends a newer file with a different name, change it here.
FILTERS = [
    dict(id="goldilocks", file="Goldilocks_hair_filter_v3.py", name="Goldilocks",
         tagline="Long, wavy golden hair with sparkles"),
    dict(id="smile", file="smile_filter_update4.py", name="Smile",
         tagline="A big cartoon smile over your mouth"),
    dict(id="clown", file="clown_face_filter_update_v2.py", name="Clown Nose",
         tagline="A shiny red clown nose"),
    dict(id="visor", file="dog_visor_update_v2.py", name="Adoggy",
         tagline="Dog ears, a nose and a tongue"),
]


def file_for(filter_id):
    return next(f["file"] for f in FILTERS if f["id"] == filter_id)


def load_module(filename):
    """Load a filter file by its file name (works even if the name isn't a normal module name)."""
    path = os.path.join(FILTERS_DIR, filename)
    name = os.path.splitext(filename)[0].replace(" ", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_model():
    """The kiosk is offline, so the face model must already be in the filters folder. (The
    filters' own load_model() would try to download it; the kiosk never lets that happen.)"""
    path = os.path.join(FILTERS_DIR, MODEL_FILE)
    if not os.path.exists(path) or os.path.getsize(path) < 100_000:
        raise RuntimeError(f"The face model {MODEL_FILE} is missing from {FILTERS_DIR}. "
                           "Copy it there (it's in Max's Downloads).")


def load_rgba(path, what):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not load {what}: {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError(f"{what} needs a transparent background (4 channels): {path}")
    return image


def make_shared_calibration():
    """The one 5-foot calibration all four filters use. The class comes from the smile filter;
    it's the same team code in every filter file."""
    smile = load_module(file_for("smile"))
    return smile.DistanceCalibration(path=os.path.join(FILTERS_DIR, CALIBRATION_FILE))


class SmileAdapter:
    """smile_filter_update4.py, one frame of its main() loop."""

    def __init__(self, calibration):
        self.m = load_module(file_for("smile"))
        self.calibration = calibration
        self.smile = load_rgba(self.m.SMILE_PATH, "smile.png")
        self.detector = cv2.FaceDetectorYN.create(self.m.MODEL_PATH, "", (640, 480), score_threshold=0.7)
        self.engaged = False
        self.reset()

    def reset(self):
        self.tracker = self.m.SmileTracker(self.calibration)
        self.engaged = False

    def process(self, frame):
        m = self.m
        frame = cv2.flip(frame, 1)   # mirror view, like a selfie camera
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        now = time.monotonic()
        self.calibration.update(faces, frame.shape, now)
        overlays = self.tracker.update(faces, frame.shape, now)
        frame = m.blur_faces(frame, self.tracker.blur_boxes)
        for left_mouth, right_mouth in overlays:
            frame = m.overlay_smile(frame, self.smile, left_mouth, right_mouth)
        self.engaged = len(overlays) > 0
        return frame


class ClownAdapter:
    """clown_face_filter_update_v2.py. Its main() and the kiosk both call the same
    ClownFilter.process(), so there's nothing to copy here."""

    def __init__(self, calibration):
        self.m = load_module(file_for("clown"))
        self.c = self.m.ClownFilter(calibration=calibration)
        self.engaged = False

    def reset(self):
        self.c.reset()
        self.engaged = False

    def process(self, frame):
        frame = cv2.flip(frame, 1)   # mirror view first, like the other filters
        frame = self.c.process(frame, show_status=False)
        self.engaged = self.c.engaged
        return frame


class VisorAdapter:
    """dog_visor_update_v2.py (Kate's filter, version 2). Like Goldilocks, its main() and the
    kiosk both call the same VisorFilter.process(), so there's nothing to copy here."""

    def __init__(self, calibration):
        import mediapipe as mp       # checked here, so a missing mediapipe only turns off this one
        if not hasattr(mp, "solutions"):   # filter, not the whole kiosk
            raise RuntimeError(f"mediapipe {mp.__version__} has no FaceMesh. Install mediapipe==0.10.18.")
        self.m = load_module(file_for("visor"))
        self.v = self.m.VisorFilter(calibration=calibration)
        self.engaged = False

    def reset(self):
        self.v.reset()
        self.engaged = False

    def process(self, frame):
        frame = cv2.flip(frame, 1)
        frame = self.v.process(frame, show_status=False)
        self.engaged = self.v.engaged
        return frame


class GoldilocksAdapter:
    """Goldilocks_hair_filter_v3.py. Its main() and the kiosk both call the same
    GoldilocksFilter.process(), so there's nothing to copy here."""

    def __init__(self, calibration):
        self.m = load_module(file_for("goldilocks"))
        self.g = self.m.GoldilocksFilter(calibration=calibration)
        self.engaged = False

    def reset(self):
        self.g.reset()
        self.engaged = False

    def process(self, frame):
        frame = cv2.flip(frame, 1)
        frame = self.g.process(frame, show_status=False)
        T = self.g.picker.target
        self.engaged = T is not None and T.acquired and T.misses == 0
        return frame


ADAPTERS = dict(goldilocks=GoldilocksAdapter, smile=SmileAdapter, clown=ClownAdapter, visor=VisorAdapter)
