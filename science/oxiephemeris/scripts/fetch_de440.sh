#!/usr/bin/env bash
# Fetch JPL DE440 ephemeris data and public-domain reference materials into
# data/ (gitignored, never committed). All sources are public domain (JPL/US
# Gov) or published standards (IERS). Re-runnable; resumes partial downloads.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="$root/data"
mkdir -p "$data/de440" "$data/de440t" "$data/iers" "$data/ref"

fetch() { # url dest
    local url="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        echo "have $dest"
        return 0
    fi
    echo "fetching $url"
    curl -fsSL --retry 5 --retry-delay 2 -C - -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

base=https://ssd.jpl.nasa.gov/ftp/eph/planets

# Required: DE440 golden test data + classic binary ephemeris
fetch "$base/Linux/de440/testpo.440"           "$data/de440/testpo.440"
fetch "$base/Linux/de440/linux_p1550p2650.440" "$data/de440/linux_p1550p2650.440"

# Reference materials (public-domain JPL Fortran defines the classic binary
# format authoritatively; ASCII header for constants cross-check)
fetch "$base/fortran/testeph.f"      "$data/ref/testeph.f"  || echo "optional: testeph.f unavailable"
fetch "$base/fortran/asc2eph.f"      "$data/ref/asc2eph.f"  || echo "optional: asc2eph.f unavailable"
fetch "$base/ascii/de440/header.440" "$data/ref/header.440" || echo "optional: header.440 unavailable"

# Optional: DE440t (carries the integrated TT-TDB series -> oracle for the
# Fairhead-Bretagnon TDB implementation)
fetch "$base/Linux/de440t/linux_p1550p2650.440t" "$data/de440t/linux_p1550p2650.440t" || echo "optional: de440t unavailable"
fetch "$base/Linux/de440t/testpo.440t"           "$data/de440t/testpo.440t"           || echo "optional: testpo.440t unavailable"

# Optional: IERS Conventions 2010 IAU 2000A_R06 nutation series, longitude
# (tab5.3a) AND obliquity (tab5.3b) — BOTH are required by the oracle test
# for the truncated nutation implementation
# (crates/oxiephemeris-bodies/tests/nutation_oracle.rs)
iers=https://iers-conventions.obspm.fr/content/chapter5/additional_info
fetch "$iers/tab5.3a.txt" "$data/iers/tab5.3a.txt" || echo "optional: tab5.3a.txt unavailable"
fetch "$iers/tab5.3b.txt" "$data/iers/tab5.3b.txt" || echo "optional: tab5.3b.txt unavailable"

touch "$data/.complete"
echo "downloads complete"
