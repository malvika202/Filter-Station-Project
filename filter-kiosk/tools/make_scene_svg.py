"""Draws the winter background (mountains and pines) as SVG shapes, and puts it into
web/index.html between the <!-- scene --> markers. Run: python3 tools/make_scene_svg.py

It draws the same scene four times, once in each filter's sky colors. The page fades between
them when a different filter is picked (a fade is cheap for the Pi's graphics chip).
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1600, 900
rng = random.Random(11)


def ridge(y_min, y_max, step_min, step_max, rough):
    """A mountain ridge from left to right: peaks and valleys with small rocky bumps between."""
    pts, x, up = [(-40, y_max)], -40, True
    while x < W + 40:
        x += rng.uniform(step_min, step_max)
        y = rng.uniform(y_min, (y_min + y_max) / 2) if up else rng.uniform((y_min + y_max) / 2, y_max)
        px, py = pts[-1]
        for t in (0.33, 0.66):           # rocky bumps along each slope
            pts.append((px + (x - px) * t, py + (y - py) * t + rng.uniform(-rough, rough)))
        pts.append((x, y))
        up = not up
    d = "M" + " L".join(f"{x:.0f},{y:.0f}" for x, y in pts)
    return d + f" L{W + 40},{H} L-40,{H} Z"


def pine(x, base, h, w):
    """One pine tree: four tiers of branches getting narrower toward the top, and a trunk."""
    tiers, pts_l, pts_r = 4, [], []
    for i in range(tiers):
        f = i / tiers
        y_bottom = base - h * 0.12 - (h * 0.88) * f
        half = w / 2 * (1 - f * 0.78)
        y_top_next = base - h * 0.12 - (h * 0.88) * (i + 1) / tiers
        pts_l += [(x - half, y_bottom), (x - half * 0.42, y_top_next + h * 0.08)]
        pts_r += [(x + half, y_bottom), (x + half * 0.42, y_top_next + h * 0.08)]
    top = (x, base - h)
    trunk = [(x + w * 0.05, base - h * 0.12), (x + w * 0.05, base), (x - w * 0.05, base), (x - w * 0.05, base - h * 0.12)]
    poly = trunk[:1] + pts_r + [top] + pts_l[::-1] + [trunk[3], trunk[2], trunk[1]]
    return "M" + " L".join(f"{px:.0f},{py:.0f}" for px, py in poly) + " Z"


def tree_row(y_base, h_min, h_max, gap_min, gap_max):
    parts, x = [], -30
    while x < W + 30:
        h = rng.uniform(h_min, h_max)
        parts.append(pine(x, y_base + rng.uniform(-6, 10), h, h * rng.uniform(0.34, 0.42)))
        x += rng.uniform(gap_min, gap_max)
    return " ".join(parts)


# Each filter's sky: top, middle, and the glow at the horizon. Gold, pink, coral, teal.
SKIES = {
    "goldilocks": ("#13223d", "#3c4a6e", "#f4c06a"),
    "smile": ("#191d3d", "#4f4673", "#f5a3bf"),
    "clown": ("#1c1934", "#563f63", "#f27a63"),
    "visor": ("#0b2233", "#1f4f5e", "#6fdccb"),
}


def mix(a, b, t):
    """Blend color a toward color b by t (0 to 1)."""
    a, b = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


# The shapes are made once and used for all four skies, so only the colors differ.
STARS = " ".join(f'<circle cx="{rng.uniform(0, W):.0f}" cy="{rng.uniform(10, 380):.0f}" r="{rng.choice([0.8, 1, 1.2, 1.6]):.1f}"/>'
                 for _ in range(46))
FAR, MID, HILL = ridge(400, 600, 110, 220, 14), ridge(490, 690, 90, 180, 12), ridge(660, 760, 120, 260, 6)
PINES_BACK, PINES_FRONT = tree_row(830, 70, 125, 22, 46), tree_row(905, 120, 200, 34, 78)


def scene(name, top, mid, glow):
    """One complete picture in one filter's colors. The peaks and rock take a little of the
    sky's color, like real dusk light."""
    snow_far, rock_far = mix("#e6edf5", glow, 0.28), mix("#4c6584", mid, 0.3)
    snow_mid, rock_mid = mix("#d5e0ec", glow, 0.2), mix("#2c4462", top, 0.2)
    hill = mix("#1d3149", top, 0.15)
    n = name
    return f'''<svg class="scene-art" data-sky="{n}" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMax slice" aria-hidden="true" focusable="false">
  <defs>
    <linearGradient id="sky-{n}" x1="0" y1="0" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{top}"/><stop offset="0.48" stop-color="{mid}"/><stop offset="0.74" stop-color="{glow}"/>
    </linearGradient>
    <radialGradient id="glow-{n}" cx="960" cy="560" r="620" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{glow}" stop-opacity="0.55"/><stop offset="1" stop-color="{glow}" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="far-{n}" x1="0" y1="430" x2="0" y2="700" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{snow_far}"/><stop offset="0.16" stop-color="{snow_far}"/><stop offset="0.62" stop-color="{rock_far}"/>
    </linearGradient>
    <linearGradient id="mid-{n}" x1="0" y1="500" x2="0" y2="760" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{snow_mid}"/><stop offset="0.12" stop-color="{snow_mid}"/><stop offset="0.55" stop-color="{rock_mid}"/>
    </linearGradient>
  </defs>
  <rect width="{W}" height="{H}" fill="url(#sky-{n})"/>
  <g fill="#ffffff" opacity="0.45">{STARS}</g>
  <rect width="{W}" height="{H}" fill="url(#glow-{n})"/>
  <path fill="url(#far-{n})" d="{FAR}"/>
  <path fill="url(#mid-{n})" d="{MID}"/>
  <path fill="{hill}" d="{HILL}"/>
  <path fill="#132739" d="{PINES_BACK}"/>
  <path fill="#0c1c2b" d="{PINES_FRONT}"/>
</svg>'''


svg = "\n".join(scene(n, *c) for n, c in SKIES.items())


page = os.path.join(HERE, "..", "web", "index.html")
html = open(page).read()
a, b = html.index("<!-- scene -->"), html.index("<!-- /scene -->")
open(page, "w").write(html[:a] + "<!-- scene -->\n" + svg + "\n" + html[b:])
print(f"4 scenes written into web/index.html ({len(svg) // 1024} KB)")
