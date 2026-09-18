#!/usr/bin/env bash
# Fetch JPL DE441 ephemeris data and IERS Earth-orientation data into data/
# (gitignored, never committed). All sources are public domain (JPL/US Gov)
# or published standards (IERS). Re-runnable; resumes partial downloads.
#
# Style mirrors scripts/fetch_de440.sh, but prefers aria2c (parallel,
# resumable segmented downloads -- useful for the ~2.7 GB DE441 binary) and
# falls back to curl when aria2c is unavailable.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="$root/data"
mkdir -p "$data/de441" "$data/iers"

# Fetches url into dir/name, resuming/parallelizing with aria2c when present,
# falling back to a plain resumable curl GET otherwise. Idempotent: a
# complete previous download (no leftover "$name.aria2" control file) is
# left untouched.
fetch() { # url dir name
    local url="$1" dir="$2" name="$3"
    local dest="$dir/$name"
    local control="$dest.aria2"
    if [[ -f "$dest" && ! -f "$control" ]]; then
        echo "have $dest"
        return 0
    fi
    if command -v aria2c >/dev/null 2>&1; then
        echo "fetching (aria2c) $url"
        aria2c -x8 -s8 -c --console-log-level=warn --summary-interval=0 \
            -d "$dir" -o "$name" "$url"
    else
        echo "fetching (curl) $url"
        curl -fsSL --retry 5 --retry-delay 2 -C - -o "$dest.part" "$url"
        mv "$dest.part" "$dest"
    fi
}

base=https://ssd.jpl.nasa.gov/ftp/eph/planets

# Required: DE441 golden test data + classic binary ephemeris (large: the
# Linux export covers -13200 to +17191, ~2.7 GB).
fetch "$base/Linux/de441/testpo.441"               "$data/de441" "testpo.441"
fetch "$base/Linux/de441/linux_m13000p17000.441"   "$data/de441" "linux_m13000p17000.441"

# IERS finals2000A.all: measured UT1-UTC (and polar motion), the preferred
# 1962+ source for Delta-T = TT - UT1 (see
# crates/oxiephemeris-core/src/time/delta_t.rs doc comments); the
# Espenak-Meeus polynomial in that module is the historical/predictive
# fallback for dates this file does not cover.
fetch "https://datacenter.iers.org/data/9/finals2000A.all" "$data/iers" "finals2000A.all" \
    || echo "optional: finals2000A.all unavailable"

touch "$data/.complete"
echo "downloads complete"
