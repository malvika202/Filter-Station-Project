# User story: switching filters on the kiosk

**As** a student walking down the IT hallway,
**I want** to tap a filter card on the kiosk screen (or click it with the mouse),
**so that** I can try each of the team's four filters on myself without anyone helping me.

## Acceptance criteria

Each one can be checked with a real test. Only check a box after you have actually done the test.

- [ ] **AC1. All four filters are on the screen.** The screen always shows four cards beside the live view: Goldilocks, Smile, Clown Nose and Adoggy. Each card has the filter's name, a picture and a one-line description.
- [ ] **AC2. Tapping works.** Tapping a card on the touchscreen switches the live view to that filter within 1 second.
- [ ] **AC3. Clicking works.** Clicking a card with the mouse switches the live view to that filter within 1 second.
- [ ] **AC4. The screen shows which filter is on.** Exactly one card is lit up at a time, and it is always the filter you can see in the live view.
- [ ] **AC5. It works every time.** 50 switches in a row (taps and clicks mixed, including tapping fast and tapping the card that is already on) all end on the card that was tapped last, with no crash and no frozen video.
- [ ] **AC6. Cards are easy to hit.** Each card is at least 70 pixels tall and 250 pixels wide on the kiosk screen, well above the usual 48-pixel minimum for a finger tap.
- [ ] **AC7. Switching never breaks the privacy rules.** In every frame during and after a switch, faces farther than about 5 feet are still blurred, and the new filter only goes on a person within about 5 feet who is facing the camera.
- [ ] **AC8. It starts ready.** When the kiosk turns on, one filter (Goldilocks) is already on and its card is lit, with no tapping needed.

## How each one gets tested

| Criterion | Test |
|---|---|
| AC1 | Look at the screen after it starts. Count the cards and read each one. |
| AC2 | On the Pi, tap each card 5 times while standing on the spot. Time it with a phone stopwatch, or read the "last switch" time on the staff panel (hold the title for 3 seconds). |
| AC3 | Same as AC2, with the mouse. |
| AC4 | After each tap in AC2 and AC3, check that the lit card matches the filter on your face. |
| AC5 | Do 50 switches in a row, mixing taps and clicks, then check the video is still moving and the last card tapped is the one that's on. |
| AC6 | Open the staff panel (hold the title for 3 seconds) and read "Smallest card". |
| AC7 | Have a second person stand far back (about 8 feet) while you switch through all four filters. Their face must stay blurred the whole time. |
| AC8 | Restart the Pi and watch the screen come up. |

The cloud tests Claude ran before the Pi exists (fake camera frames, simulated taps and clicks in a real browser) are in `TEST_REPORT.md`. They don't replace the real test on the Pi, so don't check a box from those alone.
