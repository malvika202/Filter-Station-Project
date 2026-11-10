# Test report: Filter Station kiosk (version 4), Goldilocks v3, Clown Nose v2 and The Visor v2

**Where these tests ran:** in the cloud, before the kiosk was on the Pi. They used Python 3.11 with
the exact library versions in `requirements.txt` (the same ones the Pi will use), a fake camera
built from the kit's public test photos (no photos of classmates), and Google Chrome for the page.
The test computer had only 2 processor cores and no graphics chip, so speed on the Pi will be
different. These tests show the code works, but they **don't replace testing on the real Pi** with
a real camera, so don't check acceptance-criteria boxes from this report alone.

## 1. Goldilocks v3 (v2 plus the team rules)

v3 adds the team's shared rules, copied word for word from Malvika's version: the 5-foot
calibration, blurring far or uncertain faces, and hair only on a face turned less than 50 degrees.
The hair itself is v2's. All six kit tests, unchanged:

| Kit test | v2 | v3 |
|---|---|---|
| Shimmer standing still (near / tips) | 0.67 / 0.52 px | 0.67 / 0.52 px (same) |
| Worst lag (turn / step / sit / walk) | 0.14 / 0.24 / 0.26 / 0.17 | same |
| Normal frames wrongly held back | 2 and 14 of 264 | same |
| Edge cases | 1152/1152 | 1152/1152 |
| Hair on the poster or a fake face | 0 | 0 |
| Hair left floating after someone walked off | 6 frames | 2 frames, and none of them are drawn* |
| Speed (opt-in spot / one step closer) | 18.9 / 33.4 ms | 17.0 / 34.0 ms (same, within noise) |
| End to end (real `main()`, fake camera) | 112 frames | 112 frames |

\*The kit test counts frames where the filter is still "locked on" during a 1-2 frame detector
blink. v3 never draws hair on those frames (team rule), which the next test checks pixel by pixel.

**Team-rules test** (`goldilocks_gate_test.py`), with a calibration: hair drawn on a face that
failed the checks: **0 frames**. Far poster blurred (sharpness 0.29 of the original). A simulated
head turn past 50 degrees: hair gone and the face blurred (0.32), hair back 4 frames after
turning back. Also passes with the Pi's library versions.

## 2. The Visor v2 (Kate's filter, fixed with her permission)

**The problem:** FaceMesh looked at the whole 640x480 picture and couldn't find a face that small,
so the visor never showed farther than about 3.5 feet. **The fix:** for each face that passes the
team checks but that FaceMesh missed, cut out a square around the face, enlarge it, run FaceMesh on
just that square, and move the points back. Everything else (team rules, blur, drawing) is the same.

Same face, different distances, 8 frames each (`visor_sweep.py`):

| Distance (eye gap) | 4.6 ft (19 px) | 4.4 ft (20 px) | 4.0 ft (22 px) | 3.7 ft (24 px) | 3.4 ft and closer |
|---|---|---|---|---|---|
| v1, person A | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 |
| v2, person A | **8/8** | **8/8** | **8/8** | **8/8** | 8/8 |
| v1, person B | 0/8 | 0/8 | 0/8 | 8/8 | 7/8 to 8/8 |
| v2, person B | **8/8** | **8/8** | **8/8** | **8/8** | 8/8 |

In the hallway scene, The Visor now shows on the person on the spot in **90 of 112 frames** (v1: 47),
the same as Smile and Clown Nose. It takes about 22 to 34 ms per frame here.

## 3. Clown Nose v2 (fixed with the team's permission)

**The problems:** (1) it found faces on the un-mirrored picture while the other three use the
mirrored one, and the face detector measures a flipped face 4 to 8% differently, so its 5-foot
line sat a few inches away from everyone else's; (2) it drew a green box around the face, which
looked like a debug box; (3) its face detector was stricter (0.8) than Smile and The Visor (0.7),
so it found, and blurred, fewer faces. **The fixes:** it now mirrors first and uses the team's
mirrored head-turn code (copied from the smile filter), the green box is gone, and the detector
uses 0.7. The nose drawing and tracking are the same.

**Result:** calibrating under each filter now gives **exactly the same** numbers for all four
(difference 0.00%, before 7.5%). In the hallway scene the nose is on the person on the spot in
90 of 112 frames, and every far face is blurred.

## 4. The kiosk

| Test | Result |
|---|---|
| All four filters run inside the kiosk (hallway scene, 112 frames each) | Pass. Filter on the person on the spot in 82 to 90 frames. Every far face the detector found was blurred: **143 of 143** in each filter. |
| Speed per frame at 640x480, measured here | Goldilocks 30 ms, Smile 21 ms, Clown Nose 14 ms, The Visor 23 ms. **The Pi will be slower: check it there** (Staff settings shows frames per second). Goldilocks thins its hair automatically if it runs slow. |
| Frame-by-frame switching (300 frames, 127 taps, 26 bursts of 3 fast taps, repeat taps) | Pass. The filter on screen was always the last one tapped (0 wrong). **393 far faces** checked during switches, all blurred. |
| A filter crashes | Pass. That frame is fully blurred (sharpness 0.001), the next frame works normally. |
| Calibration through Staff settings, under each filter | Pass. All four give exactly the same calibration. Saves only the 3 numbers plus settings; no picture. |
| Start script started twice / stop script | One browser and one kiosk program; stop shuts both down. |
| What the code can write or send | Only the calibration file is ever written. The page loads nothing from the internet. The server listens on 127.0.0.1 only. |
| Fresh copy from the zip | Setup check passes, all four filters load, switching works, nothing extra written. |

### Acceptance criteria, in a real browser (touch taps and mouse clicks)

| | Result |
|---|---|
| AC1 four cards with name, picture, description | Pass. Each picture is made by the kiosk from the filter's own art and loads on all four cards. |
| AC2 / AC3 tap / click switches within 1 second | Pass. Tap to new picture: worst about 0.26 s. Tap to lit card: median about 0.26 s, worst 0.77 s with the frosted-glass look (3 runs on the slow test computer). |
| AC4 exactly one card lit, matching the video | Pass at every check. The sky also changes to the tapped filter's color. |
| AC5 50 switches (taps + clicks, fast doubles, repeats) | Pass, 50/50 in every run. Video still moving after. Fast double taps: 0 wrong out of 140. |
| AC6 cards at least 70 x 250 px | Pass on 1920x1080 (576x131), 1280x720 (384x72), 1024x600 (379x76), 1080x1920 (499x140), 720x1280 (330x102). Text fits on all. |
| AC7 privacy during switches | Pass (switching test above) |
| AC8 starts on Goldilocks | Pass |

## 5. Bugs the tests found, and what changed

1. **Fast double taps could end on the wrong filter** (4 out of 40). Two taps sent at the same
   moment could reach the kiosk program in the wrong order. Fix: the page now sends taps one at a
   time, in order. After: **0 out of 80**, then 0 out of 60 on the new design.
2. **The Visor didn't work at the spot** (see section 2). Fixed in The Visor v2.
   **The clown filter's 5-foot line didn't match the others** (see section 3). Fixed in Clown Nose v2.
3. **The new design slowed the page down.** Falling snow and a whole-page color fade made the card
   take up to 1.4 seconds to light up on the test computer. Fixes: the four skies are drawn once and
   cross-faded (cheap), the card color changes instantly, falling snow is off by default (it can be
   turned on, see README), and the page checks every 0.1 s right after a tap. After: worst 0.68 s.
4. **Card text spilled out of the cards on small screens.** Fix: each card sizes its icon and text
   to its own height. After: text fits on all 5 screen sizes.
5. **A privacy sentence wasn't strictly true.** "Nothing is saved" ignored the calibration file
   and a log file the start script kept. Fix: no log file any more, and the words now say exactly
   what's saved. (Any false claim scores 0.)
6. **Staff couldn't leave the kiosk.** Chromium reopened itself 2 seconds after closing. Fix:
   Alt+F4 now closes it and shows the desktop.
7. Some of my own tests were wrong at first and were fixed: one measured a face that was hidden
   behind another person, and one stopped waiting before a double tap had finished.

## 6. Limits the team should know about

1. **A face the detector misses can't be blurred.** The tests saw this with a half-visible face
   at the edge of a photo. The sign says so.
2. **mediapipe must stay at version 0.10.18.** Newer versions removed FaceMesh, and 0.10.18 is the
   newest that installs on a Pi 5.
3. The real filter pictures from Malvika are in. With them, The Visor's dog face was moved so its
   nose sits on the person's nose (it used to be centered on the eyes).
4. **Speed on the Pi is unknown until it's measured there.** Check Staff settings while switching.
