#!/bin/bash
# Keeps the kiosk program running: if it ever stops, it starts again 2 seconds later.
# Only one copy runs at a time. Its messages are thrown away (no log file is kept, so the
# privacy sign stays true). To see the messages, stop the kiosk and run kiosk_server.py by hand
# (see Troubleshooting in README.md).
cd "$(dirname "$0")"
exec 9>/tmp/filter-station-server.lock
flock -n 9 || exit 0
while true; do
  .venv/bin/python kiosk_server.py > /dev/null 2>&1
  sleep 2
done
