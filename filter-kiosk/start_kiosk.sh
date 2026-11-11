#!/bin/bash
# Filter Station: starts the kiosk program (if it isn't running), waits for it, then opens
# Chromium full screen. Runs automatically at login (setup_pi.sh sets that up).
# To leave the kiosk: plug in a keyboard and press Alt+F4. To open it again: restart the Pi,
# or run this file again (bash ~/filter-kiosk/start_kiosk.sh).
cd "$(dirname "$0")"

# 1. The kiosk program, in the background. run_server_loop.sh restarts it if it ever stops,
#    and makes sure only one copy runs.
nohup bash run_server_loop.sh > /dev/null 2>&1 &

# 2. Wait until the page answers (loading the four filters takes a few seconds)
for i in $(seq 1 90); do
  curl -s -o /dev/null http://127.0.0.1:8000/api/state && break
  sleep 1
done

# 3. Chromium in kiosk mode: full screen, no pinch zoom, no swipe-back, no pop-ups, incognito
#    (nothing kept after it closes). The lock stops a second window opening.
BROWSER="$(command -v chromium-browser || command -v chromium)"
exec flock -n /tmp/filter-station-browser.lock "$BROWSER" --kiosk --incognito --password-store=basic --noerrdialogs \
  --disable-infobars --no-first-run --disable-pinch --overscroll-history-navigation=0 \
  --disable-features=Translate --check-for-update-interval=31536000 --ozone-platform-hint=auto \
  http://127.0.0.1:8000/
