#!/bin/bash
# Filter Station: one-time setup on the Raspberry Pi. Needs the internet ONCE (to download the
# Python libraries). After this, the kiosk runs with no internet at all.
#
# Run it from the filter-kiosk folder:   bash setup_pi.sh
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"
echo "== Filter Station setup in $HERE"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "This needs 64-bit Raspberry Pi OS (uname says $(uname -m)). Reinstall with the 64-bit version."
  exit 1
fi

# 1. Python 3.11 (mediapipe 0.10.18 needs 3.9 to 3.12). Bookworm has 3.11 built in.
if command -v python3.11 >/dev/null 2>&1; then
  echo "== Making the Python environment with the Pi's Python 3.11"
  python3.11 -m venv .venv
else
  echo "== This Pi doesn't have Python 3.11 (newer OS). Getting it with uv."
  if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  "$UV" venv --python 3.11 --seed .venv
fi

# 2. The libraries, from PyPI, at the exact tested versions
echo "== Installing mediapipe, OpenCV and numpy (this takes a few minutes)"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt

# 3. Check everything (the camera check runs too, if one is plugged in)
echo "== Checking the setup"
.venv/bin/python check_setup.py || true

# 4. Start the kiosk automatically when the Pi logs in to the desktop
chmod +x start_kiosk.sh run_server_loop.sh stop_kiosk.sh
mkdir -p "$HOME/.config/labwc" "$HOME/.config/autostart"
LINE="$HERE/start_kiosk.sh &"
touch "$HOME/.config/labwc/autostart"
grep -qF "$HERE/start_kiosk.sh" "$HOME/.config/labwc/autostart" || echo "$LINE" >> "$HOME/.config/labwc/autostart"
cat > "$HOME/.config/autostart/filter-station.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Filter Station
Exec=$HERE/start_kiosk.sh
X-GNOME-Autostart-enabled=true
EOF
echo "== Autostart is set (labwc and desktop autostart; start_kiosk.sh makes sure only one copy runs)."
echo
echo "Setup finished. Next: turn on desktop auto-login and turn off screen blanking (see README),"
echo "then restart the Pi:  sudo reboot"
