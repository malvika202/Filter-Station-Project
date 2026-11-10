"""Card pictures: a simple drawn mannequin head wearing each filter's REAL artwork.

The kiosk makes these when it starts, using each filter's own drawing code and its own picture
files (the Goldilocks hair, smile.png, clown_nose.png, "dog visor.png"). So when the real pictures
replace the stand-ins, the cards update by themselves. No photos of anyone are used.
"""
import math

import cv2
import numpy as np

SIZE = 320                          # each picture is SIZE x SIZE pixels
CX, EYE_Y, GAP = 160, 150, 60       # the face: center line, eye height, gap between the eyes
# soft background color for each card picture (a pale version of the filter's sky color)
TINTS = {"goldilocks": (148, 214, 246), "smile": (214, 205, 248), "clown": (190, 196, 248), "visor": (222, 236, 196)}

# The 5 face points a face detector would find on the drawn head (eyes, nose, mouth corners)
LEFT_EYE, RIGHT_EYE = (CX - GAP / 2, EYE_Y), (CX + GAP / 2, EYE_Y)
NOSE = (CX, EYE_Y + 0.55 * GAP)
LEFT_MOUTH, RIGHT_MOUTH = (CX - 0.38 * GAP, EYE_Y + 1.05 * GAP), (CX + 0.38 * GAP, EYE_Y + 1.05 * GAP)


def mannequin(tint, mouth=True):
    """The plain drawn head and shoulders on a soft colored background."""
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:] = tint
    top = np.array(tint, float) * 1.04
    for y in range(SIZE):                               # gentle light-to-darker background
        img[y] = np.clip(top * (1 - 0.12 * y / SIZE), 0, 255)
    body = (118, 104, 92)                               # slate blue-grey sweater
    cv2.ellipse(img, (CX, SIZE + 40), (128, 112), 0, 180, 360, body, -1, cv2.LINE_AA)
    skin = (214, 220, 228)                              # neutral mannequin color
    cv2.rectangle(img, (CX - 26, 220), (CX + 26, 262), (196, 203, 212), -1, cv2.LINE_AA)
    cv2.ellipse(img, (CX, 262), (40, 14), 0, 0, 180, (196, 203, 212), -1, cv2.LINE_AA)
    cv2.ellipse(img, (CX, 162), (70, 88), 0, 0, 360, skin, -1, cv2.LINE_AA)
    for ex, ey in (LEFT_EYE, RIGHT_EYE):
        cv2.ellipse(img, (int(ex), int(ey)), (6, 7), 0, 0, 360, (70, 56, 43), -1, cv2.LINE_AA)
    cv2.ellipse(img, (CX, int(NOSE[1]) - 4), (6, 4), 0, 0, 180, (188, 194, 204), 2, cv2.LINE_AA)
    if mouth:                                           # (left off under the smile filter)
        cv2.ellipse(img, (CX, int(LEFT_MOUTH[1]) - 6), (16, 7), 0, 10, 170, (150, 150, 196), 3, cv2.LINE_AA)
    return img


def to_round_png(img):
    """Cut the picture into a circle with a smooth edge, and return PNG bytes."""
    big = 4
    mask = np.zeros((SIZE * big, SIZE * big), np.uint8)
    cv2.circle(mask, (SIZE * big // 2, SIZE * big // 2), SIZE * big // 2 - big, 255, -1, cv2.LINE_AA)
    mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".png", np.dstack([img, mask]))
    return data.tobytes()


def goldilocks_preview(adapter):
    m = adapter.m
    img = mannequin(TINTS["goldilocks"])
    pts = np.array([LEFT_EYE, RIGHT_EYE, NOSE, LEFT_MOUTH, RIGHT_MOUTH], float)
    raw = m.estimate_pose(pts)
    pose = dict(x=raw["x"], y=raw["y"], s=raw["s"], yaw=raw["yaw"], pitch=raw["pitch"],
                roll=raw["roll"], feat=pts)
    hair = m.HairState(adapter.g.model)
    for k in range(20):                     # a few frames so the hair settles into place
        out = img.copy()
        hair.render(out, pose, 1.0, 1.0 + k / 15, 1 / 15)
    return out


def smile_preview(adapter):
    img = mannequin(TINTS["smile"], mouth=False)
    return adapter.m.overlay_smile(img, adapter.smile, LEFT_MOUTH, RIGHT_MOUTH)


def clown_preview(adapter):
    img = mannequin(TINTS["clown"])
    return adapter.m.overlay_nose(img, adapter.c.nose_img, NOSE, GAP)


def visor_preview(adapter):
    m = adapter.m
    img = mannequin(TINTS["visor"])
    # FaceMesh points the visor uses: 33 and 263 are the outer eye corners, 1 is the nose tip
    points = [m._Point(CX / SIZE, EYE_Y / SIZE) for _ in range(478)]
    corner = GAP / 2 + 0.22 * GAP      # outer eye corners: about 1.45 x the eye gap apart
    points[33] = m._Point((CX - corner) / SIZE, EYE_Y / SIZE)
    points[263] = m._Point((CX + corner) / SIZE, EYE_Y / SIZE)
    points[1] = m._Point(NOSE[0] / SIZE, NOSE[1] / SIZE)
    out = m.render_visor(img, m._Mesh(points), "card-preview", adapter.v.visor_image)
    m.recent_eye_centers.pop("card-preview", None)
    m.face_geometry.pop("card-preview", None)
    return out


MAKERS = dict(goldilocks=goldilocks_preview, smile=smile_preview, clown=clown_preview, visor=visor_preview)


def make_previews(adapters):
    """{filter id: PNG bytes} for every filter that loaded. A preview that fails is skipped
    (the card then shows a plain colored circle)."""
    out = {}
    for fid, adapter in adapters.items():
        try:
            out[fid] = to_round_png(MAKERS[fid](adapter))
        except Exception as e:
            print(f"Could not make the {fid} card picture: {type(e).__name__}: {e}")
    return out
