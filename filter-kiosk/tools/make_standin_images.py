"""Makes the three STAND-IN filter images, so the kiosk can run before Malvika sends the real ones.

They use the exact file names the filters expect. To use the real art, copy the real files into
the filters folder with the same names (smile.png, clown_nose.png, "dog visor.png") and restart.
Run:  python3 tools/make_standin_images.py   (only overwrites files that are still stand-ins)
"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILTERS = os.path.join(HERE, "..", "filters")
SS = 4  # draw 4x bigger, then shrink, for smooth edges
MARK = b"standin"  # written into each stand-in PNG, so this script never overwrites real art


def finish(big, path):
    small = cv2.resize(big, (big.shape[1] // SS, big.shape[0] // SS), interpolation=cv2.INTER_AREA)
    if os.path.exists(path):
        with open(path, "rb") as f:
            if MARK not in f.read():
                print(f"kept the real image: {os.path.basename(path)}")
                return
    ok, data = cv2.imencode(".png", small)
    with open(path, "wb") as f:
        f.write(data.tobytes() + MARK)   # bytes after the PNG's end are ignored by image readers
    print(f"wrote stand-in: {os.path.basename(path)} {small.shape[1]}x{small.shape[0]}")


def smile():
    # Same size and mouth-corner spots as the real smile.png (495 x 404, corners at
    # (60, 108) and (428, 126)), because overlay_smile() lines the art up using those spots.
    w, h = 495 * SS, 404 * SS
    img = np.zeros((h, w, 4), np.uint8)
    L, R = np.array([60, 108]) * SS, np.array([428, 126]) * SS
    t = np.linspace(0, 1, 80)[:, None]
    top = L + (R - L) * t + np.c_[np.zeros(80), 40 * SS * np.sin(np.pi * t[:, 0])]
    bot = L + (R - L) * t + np.c_[np.zeros(80), 230 * SS * np.sin(np.pi * t[:, 0])]
    mouth = np.vstack([top, bot[::-1]]).astype(np.int32)
    cv2.fillPoly(img, [mouth], (40, 20, 150, 255), cv2.LINE_AA)                 # mouth (dark red)
    teeth = np.vstack([top, (top + (bot - top) * 0.32)[::-1]]).astype(np.int32)
    cv2.fillPoly(img, [teeth], (245, 245, 250, 255), cv2.LINE_AA)               # teeth
    tongue_c = (int((L[0] + R[0]) / 2), int(L[1] + 205 * SS))
    tongue = np.zeros_like(img)
    cv2.ellipse(tongue, tongue_c, (95 * SS, 50 * SS), 0, 180, 360, (120, 110, 235, 255), -1, cv2.LINE_AA)
    inside = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(inside, [mouth], 255, cv2.LINE_AA)
    a = (tongue[..., 3:4].astype(np.float32) / 255) * (inside[..., None].astype(np.float32) / 255)
    img[:] = (img * (1 - a) + tongue * a).astype(np.uint8)
    cv2.polylines(img, [mouth], True, (20, 10, 40, 255), 9 * SS, cv2.LINE_AA)   # outline
    for c in (L, R):                                                             # dimples
        cv2.ellipse(img, (int(c[0]), int(c[1])), (22 * SS, 30 * SS), 0, 0, 360, (20, 10, 40, 255), 7 * SS, cv2.LINE_AA)
    img[mouth[:, 1].max() + 30 * SS:] = 0
    finish(img, os.path.join(FILTERS, "smile.png"))


def clown_nose():
    n = 240 * SS
    img = np.zeros((n, n, 4), np.uint8)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    c, r = n / 2, n * 0.46
    d = np.hypot(xx - c, yy - c) / r
    inside = d <= 1
    light = np.clip(1.0 - np.hypot(xx - c * 0.72, yy - c * 0.68) / (r * 1.6), 0, 1)
    img[..., 2] = np.where(inside, 120 + 135 * light, 0).astype(np.uint8)        # red
    img[..., 1] = np.where(inside, 10 + 40 * light ** 3, 0).astype(np.uint8)
    img[..., 0] = np.where(inside, 20 + 50 * light ** 3, 0).astype(np.uint8)
    img[..., 3] = np.where(inside, 255, 0).astype(np.uint8)
    cv2.ellipse(img, (int(c * 0.7), int(c * 0.66)), (int(r * 0.22), int(r * 0.13)), -35, 0, 360,
                (235, 235, 255, 255), -1, cv2.LINE_AA)                            # shine
    finish(img, os.path.join(FILTERS, "clown_nose.png"))


def dog_visor():
    w, h = 640 * SS, 260 * SS
    img = np.zeros((h, w, 4), np.uint8)
    # floppy ears hanging at each end
    for side in (-1, 1):
        cx = w / 2 + side * 262 * SS
        ear = np.array([[cx - side * 10 * SS, 60 * SS], [cx + side * 40 * SS, 80 * SS],
                        [cx + side * 45 * SS, 230 * SS], [cx - side * 5 * SS, 250 * SS],
                        [cx - side * 40 * SS, 150 * SS]], np.int32)
        cv2.fillPoly(img, [ear], (40, 80, 140, 255), cv2.LINE_AA)
        cv2.polylines(img, [ear], True, (20, 40, 70, 255), 6 * SS, cv2.LINE_AA)
    # the visor lens across the eyes, tinted and see-through
    lens = cv2.ellipse2Poly((w // 2, 110 * SS), (285 * SS, 70 * SS), 0, 0, 360, 4)
    cv2.fillPoly(img, [lens], (110, 70, 30, 175), cv2.LINE_AA)
    cv2.polylines(img, [lens], True, (75, 170, 215, 255), 10 * SS, cv2.LINE_AA)    # brass rim
    cv2.ellipse(img, (w // 2 - 120 * SS, 85 * SS), (90 * SS, 18 * SS), -8, 0, 360, (230, 220, 210, 120), -1, cv2.LINE_AA)
    # a little nose button in the middle of the rim
    cv2.circle(img, (w // 2, 172 * SS), 22 * SS, (30, 30, 35, 255), -1, cv2.LINE_AA)
    finish(img, os.path.join(FILTERS, "dog visor.png"))


if __name__ == "__main__":
    smile()
    clown_nose()
    dog_visor()
