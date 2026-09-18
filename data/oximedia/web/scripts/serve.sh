#!/usr/bin/env bash
# Serves web/ over plain HTTP (no COOP/COEP headers needed — oximedia-web
# never uses SharedArrayBuffer/threads). Demo lives at /demo/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT" && exec python3 -m http.server "${1:-8080}"
