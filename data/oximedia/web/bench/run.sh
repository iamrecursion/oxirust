#!/usr/bin/env bash
# Runs the oximedia-web benchmark harness end-to-end in headless Chrome and
# prints/saves the measured results. See ./README.md for methodology and
# how to read the numbers, and bench.js's module doc for the completion
# protocol this script depends on.
#
# Usage:
#   ./run.sh
#
# Env overrides (all optional):
#   OXIBENCH_CHROME_BIN    path to a Chrome/Chromium binary
#   OXIBENCH_CHROME_FLAGS  extra flags appended to the Chrome invocation
#                          (space-separated, e.g. "--no-sandbox" for some
#                          containerized/CI environments)
#   OXIBENCH_TIMEOUT       seconds to wait for the benchmark to complete
#                          (default 180)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)" # web/
BENCH_DIR="$ROOT/bench"
RESULTS_DIR="$BENCH_DIR/results"
RESULT_JSON="$RESULTS_DIR/local-latest.json"

CHROME_BIN="${OXIBENCH_CHROME_BIN:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
TIMEOUT_SECS="${OXIBENCH_TIMEOUT:-180}"

RESULT_MARKER="OXIBENCH_RESULT:"
ERROR_MARKER="OXIBENCH_ERROR:"

mkdir -p "$RESULTS_DIR"

if [ ! -x "$CHROME_BIN" ]; then
    cat >&2 <<EOF
error: no Chrome binary found at:
  $CHROME_BIN

This harness needs a real browser (WebCodecs + wasm SIMD aren't available
in Node), so it can't fall back to a headless-less run. To run it manually
instead:

  1. ./scripts/build.sh            # if dist/ is missing or stale
  2. ./scripts/serve.sh 8080
  3. open http://127.0.0.1:8080/bench/index.html in a browser (or append
     ?auto=1 to run automatically) and use the "Run benchmarks" button.

Set OXIBENCH_CHROME_BIN to point this script at a different Chrome/Chromium
binary if you have one installed elsewhere.
EOF
    exit 0
fi

if [ ! -d "$ROOT/dist/wasm" ]; then
    echo "==> dist/ missing or incomplete, building..." >&2
    "$ROOT/scripts/build.sh" >&2
fi

PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
BASE_URL="http://127.0.0.1:$PORT"

SERVER_LOG="$(mktemp -t oxibench-server)"
CHROME_LOG="$(mktemp -t oxibench-chrome)"
SERVER_PID=""
CHROME_PID=""
KEEP_CHROME_LOG=0

cleanup() {
    if [ -n "$CHROME_PID" ]; then
        kill "$CHROME_PID" >/dev/null 2>&1 || true
        wait "$CHROME_PID" 2>/dev/null || true
    fi
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$SERVER_LOG"
    if [ "$KEEP_CHROME_LOG" -ne 1 ]; then
        rm -f "$CHROME_LOG"
    fi
}
trap cleanup EXIT

(cd "$ROOT" && exec python3 -m http.server "$PORT") >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "==> waiting for dev server on $BASE_URL ..." >&2
server_ready=0
for _ in $(seq 1 50); do
    if curl -s -o /dev/null -w '%{http_code}' "$BASE_URL/bench/index.html" 2>/dev/null | grep -q '^200$'; then
        server_ready=1
        break
    fi
    sleep 0.2
done
if [ "$server_ready" -ne 1 ]; then
    echo "error: dev server did not come up on $BASE_URL within 10s" >&2
    echo "--- server log ---" >&2
    cat "$SERVER_LOG" >&2
    exit 1
fi

CHROME_ARGS=(--headless=new --disable-gpu --enable-logging=stderr)
if [ -n "${OXIBENCH_CHROME_FLAGS:-}" ]; then
    # Intentional word-splitting: OXIBENCH_CHROME_FLAGS is a space-separated
    # list of flags the caller opted into via an env var, same convention
    # every other "extra flags" env var uses.
    # shellcheck disable=SC2206
    CHROME_ARGS+=($OXIBENCH_CHROME_FLAGS)
fi
CHROME_ARGS+=("$BASE_URL/bench/index.html?auto=1")

echo "==> launching headless Chrome (timeout ${TIMEOUT_SECS}s): $CHROME_BIN ${CHROME_ARGS[*]}" >&2
"$CHROME_BIN" "${CHROME_ARGS[@]}" >/dev/null 2>"$CHROME_LOG" &
CHROME_PID=$!

# bench.js reports completion (success or failure) by console.log-ing a
# marker line, which shows up in Chrome's --enable-logging=stderr output as
# an [INFO:CONSOLE:N] line. Poll the log file for either marker rather than
# waiting on the process to exit: headless Chrome given a plain URL (no
# --dump-dom/--screenshot/--print-to-pdf) does not exit on its own once the
# page has loaded, it just keeps running.
poll_ticks=$((TIMEOUT_SECS * 5))
outcome="timeout"
for _ in $(seq 1 "$poll_ticks"); do
    if grep -aq "$RESULT_MARKER" "$CHROME_LOG" 2>/dev/null; then
        outcome="result"
        break
    fi
    if grep -aq "$ERROR_MARKER" "$CHROME_LOG" 2>/dev/null; then
        outcome="error"
        break
    fi
    if ! kill -0 "$CHROME_PID" 2>/dev/null; then
        outcome="crashed"
        break
    fi
    sleep 0.2
done

if [ "$outcome" != "result" ]; then
    KEEP_CHROME_LOG=1
    case "$outcome" in
    timeout)
        echo "error: benchmark did not complete within ${TIMEOUT_SECS}s (no $RESULT_MARKER seen)" >&2
        ;;
    error)
        echo "error: the benchmark page reported a failure:" >&2
        grep -a "$ERROR_MARKER" "$CHROME_LOG" | tail -1 >&2
        ;;
    crashed)
        echo "error: headless Chrome exited before reporting a result" >&2
        ;;
    esac
    echo "  full Chrome log kept at: $CHROME_LOG" >&2
    exit 1
fi

python3 - "$CHROME_LOG" "$RESULT_JSON" "$RESULT_MARKER" <<'PY'
import json
import sys

log_path, out_path, marker = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(log_path, encoding="utf-8", errors="replace").read()

# Chrome's console-log sink wraps the message in a plain pair of double
# quotes without escaping quotes that are already part of the message
# (JSON.stringify's own \" escaping inside e.g. a user-agent string is
# preserved verbatim) — so the reliable terminator is the fixed literal
# suffix Chrome always appends, not a naive "next quote" search. Use the
# *last* marker occurrence in case Chrome logged a truncated duplicate
# during page navigation retries.
idx = text.rfind(marker)
if idx < 0:
    print(f"error: marker {marker!r} not found in chrome log", file=sys.stderr)
    sys.exit(1)
start = idx + len(marker)
end = text.find('", source:', start)
if end < 0:
    print("error: could not find the end of the console message", file=sys.stderr)
    sys.exit(1)
payload = text[start:end]

try:
    data = json.loads(payload)
except json.JSONDecodeError as exc:
    print(f"error: extracted payload is not valid JSON: {exc}", file=sys.stderr)
    sys.exit(1)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

results = data.get("results", [])
if not results:
    print("warning: results array is empty", file=sys.stderr)

name_w = max([len("suite")] + [len(r["name"]) for r in results])
header = f"{'suite'.ljust(name_w)}  {'n':>4}  {'median_ms':>10}  {'p95_ms':>10}  {'min_ms':>10}"
print()
print(header)
print("-" * len(header))
for r in results:
    print(
        f"{r['name'].ljust(name_w)}  {r['n']:>4}  "
        f"{r['median_ms']:>10.3f}  {r['p95_ms']:>10.3f}  {r['min_ms']:>10.3f}"
    )
print()
print(f"wrote {out_path}")
PY
