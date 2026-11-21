#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
docker compose up -d --build --wait
printf 'ReelTone: http://127.0.0.1:%s/\n' "${REELTONE_PORT:-19955}"
