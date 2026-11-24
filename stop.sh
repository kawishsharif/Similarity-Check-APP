#!/usr/bin/env bash
# Stop the app started by ./run.sh   (./stop.sh docker stops the Docker stack)
set -uo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "docker" ]; then
  docker compose down
  exit 0
fi

PID_FILE=".run/app.pid"
if [ ! -f "$PID_FILE" ]; then echo "Not running (no pid file)."; exit 0; fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$PID" 2>/dev/null && { echo "Did not stop gracefully; forcing."; kill -9 "$PID"; }
  echo "Stopped (pid $PID)."
else
  echo "Process $PID was not running."
fi
rm -f "$PID_FILE"
