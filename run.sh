#!/usr/bin/env bash
# Start the whole app locally (web + background checks) or with Docker.
#   ./run.sh            local: virtualenv + SQLite + in-process queue, http://localhost:8000
#   ./run.sh docker     full stack with Docker Compose (needs .env, see .env.example)
# Settings come from .env (created on first run). Stop with ./stop.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "docker" ]; then
  [ -f .env ] || { echo "Create .env first: cp .env.example .env  (then edit it)"; exit 1; }
  docker compose up -d --build
  echo "Started. Open https://\$DOMAIN (see .env). Stop with: ./stop.sh docker"
  exit 0
fi

PORT="${PORT:-8000}"
VENV_DIR="${VENV_DIR:-.venv}"
RUN_DIR=".run"; mkdir -p "$RUN_DIR"
PID_FILE="$RUN_DIR/app.pid"; LOG_FILE="$RUN_DIR/app.log"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Already running (pid $(cat "$PID_FILE")) at http://localhost:$PORT. Use ./stop.sh first to restart."
  exit 0
fi

# First run: create .env with an admin login.
if [ ! -f .env ]; then
  PASS="$(python3 -c 'import secrets;print(secrets.token_urlsafe(12))')"
  cat > .env <<EOF
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=$PASS
SIGNUP_OPEN=true
BASE_URL=http://localhost:$PORT
EOF
  chmod 600 .env
  echo "Created .env with admin login: admin@example.com / $PASS  (change it in .env)"
fi
set -a; . ./.env; set +a

# Python environment
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
[ -d "$VENV_DIR" ] || python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"
pip install -q --disable-pip-version-check -r requirements.txt

# Optional system tools: warn, don't fail.
command -v soffice >/dev/null || command -v libreoffice >/dev/null || echo "warning: LibreOffice not found - DOCX uploads will be refused (PDF and TXT work)."
command -v tesseract >/dev/null || echo "warning: Tesseract not found - scanned (image-only) pages cannot be read."
python - <<'PY' 2>/dev/null || echo "warning: WeasyPrint cannot load Pango - install libpango (apt install libpango-1.0-0 libpangoft2-1.0-0) or reports will fail."
import weasyprint
PY

nohup python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --proxy-headers >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

for _ in $(seq 1 40); do
  if curl -fsS "http://localhost:$PORT/healthz" >/dev/null 2>&1; then
    echo "Running at http://localhost:$PORT  (pid $(cat "$PID_FILE"), log: $LOG_FILE)"
    echo "Sign in as ${ADMIN_EMAIL:-the admin}, open Settings to upload your report logo."
    exit 0
  fi
  kill -0 "$(cat "$PID_FILE")" 2>/dev/null || { echo "The app exited. Last log lines:"; tail -20 "$LOG_FILE"; rm -f "$PID_FILE"; exit 1; }
  sleep 0.5
done
echo "Started but not answering yet; check $LOG_FILE"
