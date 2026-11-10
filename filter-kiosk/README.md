# Filter Station: the hallway kiosk

The kiosk's own screen for the team's four filters (Goldilocks, Smile, Clown Nose, The Visor).
It runs on the Raspberry Pi 5 with no internet. People tap a card on the touchscreen (or click it
with a mouse) to pick a filter, and the privacy information is always on screen.

**How it works, in short:** `kiosk_server.py` reads the webcam, runs the filter that's switched
on, and sends the finished video to a web page (`web/`) shown full screen in Chromium. Both run on
the Pi, and the server only listens on the Pi itself (127.0.0.1). `filter_adapters.py` runs each
filter a frame at a time, doing exactly what its own `main()` loop does.

**The look:** a winter forest at dusk, with snowy mountains and pines. The sky's color follows the
filter that's on (gold for Goldilocks, pink for Smile, coral for Clown Nose, teal for The Visor) and
fades when someone picks a different one. The cards and the privacy panel are frosted glass, and the
filter that's on is a solid white card. Each card's picture is a drawn mannequin wearing that filter's
real artwork, made by the kiosk when it starts (`previews.py`), so real pictures show up by themselves.

## What's in the folder

| | |
|---|---|
| `kiosk_server.py` | The kiosk program: camera loop, switching, video stream to the page |
| `filter_adapters.py` | Runs each filter one frame at a time. The list of filters and file names is at the top. |
| `previews.py` | Makes the card pictures from each filter's own art when the kiosk starts |
| `filters/` | The filter files, the face model, the three pictures, and the calibration once made. Smile is Malvika's file, unchanged (it must stay that way). Clown Nose is `clown_face_filter_update_v2.py` and The Visor is `dog_visor_update_v2.py`, both fixed with the team's permission; the originals are kept beside them for reference. |
| `web/` | The page: `index.html`, `kiosk.css`, `kiosk.js`. Plain system font, no internet needed. |
| `setup_pi.sh` | One-time setup on the Pi |
| `start_kiosk.sh`, `run_server_loop.sh`, `stop_kiosk.sh` | Start, keep running, stop |
| `check_setup.py` | Checks everything is ready, in plain words |
| `sign/` | The printable privacy sign (PDF) |
| `tools/` | `make_standin_images.py` made the stand-in pictures; `make_scene_svg.py` draws the mountains and pines into the page |
| `USER_STORY.md`, `PRIVACY_CHECK.md`, `TEST_REPORT.md` | The switching user story, why every privacy claim is true, and the test results |

## Before you start

- Raspberry Pi 5 with **Raspberry Pi OS (64-bit) with desktop, Bookworm**. (Newer OS versions work
  too: the setup gets the right Python version itself.)
- The USB webcam, and a touchscreen or a monitor with a mouse.
- **Internet once**, for setup only (school Wi-Fi, or a phone hotspot).
- The real `smile.png`, `clown_nose.png` and `dog visor.png` are already in `filters/` (from
  Malvika). If she made `smile_distance_calibration.json` **with this same webcam at 640x480**,
  add it to `filters/` too. If not, you'll calibrate on the Pi (step 6).

## Set up the Pi (one time)

1. Copy the whole `filter-kiosk` folder onto a USB stick, plug it into the Pi, and drag the folder
   into the Pi's home folder (so it's at `/home/<your user>/filter-kiosk`).
2. If Malvika has a calibration file from this webcam, copy it into `filter-kiosk/filters/`.
3. Open **Terminal** on the Pi and paste:
   ```
   cd ~/filter-kiosk && bash setup_pi.sh
   ```
   It takes a few minutes. At the end it runs the setup check. "PROBLEM" lines say what to fix;
   "NOTE" lines are fine (for example, "not calibrated yet").
4. Still in Terminal, paste `sudo raspi-config`, then:
   - **System Options > Boot / Auto Login > Desktop Autologin**
   - **Display Options > Screen Blanking > No**
   - **Finish**, and say **Yes** to reboot.
5. The Pi restarts straight into the kiosk, full screen.
6. **Calibrate** (needed once, and again if the camera, its zoom or its position changes):
   hold the **Filter Station** title for 3 seconds to open Staff settings. Have everyone else step
   out of view, stand alone with your face **exactly 5 feet** from the camera (use a tape measure),
   face it, tap **Start calibration** and hold still until it says Calibrated. Until then, every
   face is blurred and no filter shows.
7. Optional, makes the privacy claim even stronger: turn off Wi-Fi (network icon, top right,
   "Turn Off Wireless LAN"). The kiosk doesn't need it.
8. Put the printed sign (`sign/Filter_Station_privacy_sign.pdf`) next to the screen, and mark the
   spot on the floor about 4 feet from the camera.

## Every day

Turn the Pi on. That's it. To get out of the kiosk, plug in a keyboard and press **Alt+F4**
(the desktop shows). To bring it back, restart the Pi. To stop everything, open Terminal and run
`bash ~/filter-kiosk/stop_kiosk.sh`.

Keyboard shortcuts, if a keyboard is plugged in: **1 to 4** pick a filter, **C** opens Staff settings.

## Changing things

- **Real pictures:** copy them into `filters/` with the same names and restart the Pi. Run
  `.venv/bin/python check_setup.py` to confirm they're not stand-ins any more.
- **A teammate sends a newer filter file:** copy it into `filters/`. If its name changed, change
  that filter's `file=` line at the top of `filter_adapters.py`. If it changes how its `main()`
  loop works, its adapter in `filter_adapters.py` has to match it. Then run the setup check, test
  it on the Pi, and re-check `PRIVACY_CHECK.md`.
- **Card names and descriptions:** the `FILTERS` list at the top of `filter_adapters.py`.
- **Frosted glass:** if the page feels slow on the Pi, delete the two `backdrop-filter` lines under
  `.look` in `web/kiosk.css` (the cards stay dark, just not blurred).
- **Falling snow:** off by default, because animation over the whole screen can slow the page on a
  busy computer. To try it, open `start_kiosk.sh` and change `http://127.0.0.1:8000/` to
  `http://127.0.0.1:8000/?snow=1`, restart, and watch the speed in Staff settings while switching
  filters. If it stays smooth, keep it.
- **Privacy words:** change them in `web/index.html` **and** `sign/privacy_sign.html` (keep them
  the same), and only if they're still true (see `PRIVACY_CHECK.md`).

## Troubleshooting

| What you see | What to do |
|---|---|
| "The camera isn't sending video" | Check the USB cable. Unplug and replug the camera; the kiosk reopens it by itself. |
| "Setup needed. Staff: hold the title..." | Calibrate (step 6). |
| A card says "Not available right now" | Open Staff settings: the Filters line says why. Usually a missing picture or mediapipe. |
| Blank screen at startup | Wait 30 seconds (the filters take a moment to load). If it stays blank, run `bash ~/filter-kiosk/stop_kiosk.sh`, then `cd ~/filter-kiosk && .venv/bin/python kiosk_server.py` in Terminal to see the messages. |
| It's slow | Staff settings shows frames per second. Goldilocks thins its hair by itself when slow. |

## Trying it on a computer (no Pi)

On a Mac or PC with Python 3.11: `python3.11 -m venv .venv`, then
`.venv/bin/python -m pip install -r requirements.txt`, then `.venv/bin/python kiosk_server.py`
and open http://127.0.0.1:8000. The privacy text says "this kiosk", so only show it to people on
the Pi.
