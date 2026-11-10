#!/bin/bash
# Stops the kiosk: closes Chromium and the kiosk program. Restart the Pi to start it again.
pkill -f run_server_loop.sh
pkill -f kiosk_server.py
pkill -f "127.0.0.1:8000"
echo "Filter Station stopped."
