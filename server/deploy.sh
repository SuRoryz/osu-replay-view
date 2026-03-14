#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="${OSU_SERVER_NAME:-osu-replay-social-server}"
ENV_FILE="$SCRIPT_DIR/.env"
STORAGE_DIR="$SCRIPT_DIR/storage/replays"
LOG_DIR="$SCRIPT_DIR/logs"

COMPOSE_CMD=()

detect_compose() {
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(docker compose)
        return
    fi
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(docker-compose)
        return
    fi
    echo "Docker Compose was not found." >&2
    echo "Install Docker with the compose plugin, then rerun deploy.sh" >&2
    exit 1
}

compose() {
    "${COMPOSE_CMD[@]}" "$@"
}

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker was not found in PATH." >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "Docker daemon is not running or is not accessible." >&2
        exit 1
    fi
}

write_default_env() {
    if [[ -f "$ENV_FILE" ]]; then
        return
    fi

    cat >"$ENV_FILE" <<'EOF'
OSU_SERVER_PORT=8000
POSTGRES_DB=osu_replay_v2
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
OSU_SERVER_CORS=*
EOF
    echo "[deploy] Created default .env file at $ENV_FILE"
}

load_env() {
    if [[ -f "$ENV_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$ENV_FILE"
    fi
}

wait_for_service() {
    local service_name="$1"
    local label="$2"
    local retries=45

    while [[ "$retries" -gt 0 ]]; do
        local container_id
        container_id="$(compose ps -q "$service_name" 2>/dev/null || true)"
        if [[ -n "$container_id" ]]; then
            local status
            status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id" 2>/dev/null || true)"
            if [[ "$status" == "healthy" || "$status" == "running" ]]; then
                echo "[deploy] $label is $status"
                return 0
            fi
        fi
        retries=$((retries - 1))
        sleep 2
    done

    echo "[deploy] $label did not become ready in time." >&2
    compose ps || true
    compose logs --tail=100 "$service_name" || true
    exit 1
}

detect_compose
require_docker
write_default_env
load_env

mkdir -p "$STORAGE_DIR" "$LOG_DIR"

echo "[deploy] Building and starting Docker services"
compose up -d --build --remove-orphans

wait_for_service db "PostgreSQL"
wait_for_service app "$APP_NAME"

echo "Done"
