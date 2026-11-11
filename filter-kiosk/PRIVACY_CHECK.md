# Privacy check: every claim on the screen and the sign, and why it's true

The rubric gives 0 for privacy if any claim is false, so each sentence the kiosk shows is listed
here with the reason it's true, where that happens in the code, and the test that checked it.
The screen (`web/index.html`) and the sign (`sign/privacy_sign.html`) use the exact same words.

**Re-check this list whenever anyone changes a filter file, the kiosk code, or the wording.**

| Claim | Why it's true | Where | Checked by |
|---|---|---|---|
| It shows a live, mirror-image video on this screen | Every filter's finished frame is mirrored before it's shown. | `filter_adapters.py` (each `process()`), `kiosk_server.py` stream | adapter test, browser screenshots |
| and finds where faces are, so it can add a filter | YuNet finds faces and 5 points on each; Adoggy also uses mediapipe FaceMesh. Adoggy v2 only zooms in on faces that already passed the team checks. | all four filter files | adapter test |
| It doesn't know who you are. | Nothing in any file does face recognition. Faces are only matched from one frame to the next by where their box is ("No identity recognition is performed" in the team's tracker code). | the trackers in each filter | code read |
| No photos or video are saved | No code writes an image or video. Searching every file for anything that writes a file finds only the calibration save. Chromium runs in incognito mode, and the start script keeps no log. | `kiosk_server.py`, `start_kiosk.sh` | file-write search (TEST_REPORT.md) |
| and nothing leaves this kiosk | The kiosk program only listens on 127.0.0.1, so only the Pi itself can open the page or video. The page loads only local files (no internet addresses in it). The filters' own "download the model" code never runs in the kiosk: it checks that the model file is already there instead. | `kiosk_server.py` (`127.0.0.1`), `filter_adapters.check_model()` | address search, server test |
| Each frame is shown here, then replaced by the next. | The kiosk keeps only the newest frame in memory and overwrites it with each new one. | `CameraSource`, `Kiosk.step()` | code read |
| The only thing the filters save is a staff setup measurement: three numbers about face size, not a photo. | The calibration file holds `features` (3 face-size numbers) plus its settings (version, 5 ft, 640x480). It's written only when staff tap Start calibration. | `DistanceCalibration.update()` in the team code | calibration test (file contents printed) |
| Only someone within about 5 feet who is facing the camera [gets a filter] | Team rules in every filter: a face only gets a filter if its measured distance is inside 5 ft (4.7 ft to come back in) and it's turned less than 50 degrees. Goldilocks also needs the face in the middle part of the screen. "About" matters: it's an estimate from face size. All four filters now use the exact same 5-foot line. | `FaceCriteria` / trackers, Goldilocks `Picker` | Goldilocks team-rules test, adapter test, switching test |
| Faces farther away, turned away, or hard to measure are blurred. | Any detected face that doesn't pass the distance check is added to the blur list. Turned past 50 degrees or can't be measured means no distance, so it's blurred. Before calibration, every detected face is blurred. If a filter crashes, the whole frame is blurred. | `blur_faces()`, trackers, `privacy_fallback()` | adapter test (143 far faces), switching test (393), crash test |
| Blurring only works on faces the camera finds, so it can miss one. | True and honest: a face the detector misses (very small, half hidden, at the edge) can't be blurred. The tests saw this happen. | | adapter test, screenshots |
| Stay out of the camera's view. Anyone in view shows up on this screen, even far back, though far faces are blurred. | The whole camera picture is shown; only faces are blurred, not bodies. | | screenshots |

## Things that would make a claim false (don't do these)

- Adding any screenshot, photo or video saving (including a "debug" save).
- Changing the server to listen on `0.0.0.0` or opening the page from another device.
- Turning on a log of video frames, or adding a CDN font or script to the page.
- Removing a filter's blur code, or adding a filter that doesn't use the team rules.
- Changing the 5 ft range or the 50 degree limit without changing the words.

To make "nothing leaves this kiosk" even stronger, turn off the Pi's Wi-Fi after setup
(top-right network icon, then "Turn Off Wireless LAN"). The kiosk doesn't need it.
