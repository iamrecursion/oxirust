#!/usr/bin/env bash
# Fetch the analytic-theory source data into data/ (gitignored, never
# committed): VSOP87E (CDS VI/81, Bretagnon & Francou 1988) and
# ELP2000-82B (CDS VI/79, Chapront-Touzé & Chapront 1988), plus each
# catalog's own notice/reference files. These feed the xtask generators
# (`gen-vsop87`, `gen-elp`) and the analytic crate's oracle tests.
# Re-runnable; skips files already present.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="$root/data"
mkdir -p "$data/vsop87" "$data/elp2000"

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

fetch_gz() { # url dest (url is the gzipped file; dest the decompressed one)
    local url="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        echo "have $dest"
        return 0
    fi
    echo "fetching $url"
    curl -fsSL --retry 5 --retry-delay 2 -o "$dest.gz.part" "$url"
    mv "$dest.gz.part" "$dest.gz"
    gzip -d "$dest.gz"
}

vsop=https://cdsarc.cds.unistra.fr/ftp/VI/81
elp=https://cdsarc.cds.unistra.fr/ftp/VI/79

# VSOP87E: barycentric rectangular J2000 series (9 bodies), the format
# notice, and the BDL check values used by tests/vsop87_chk.rs.
for suffix in sun mer ven ear mar jup sat ura nep; do
    fetch "$vsop/VSOP87E.$suffix" "$data/vsop87/VSOP87E.$suffix"
done
fetch "$vsop/vsop87.txt" "$data/vsop87/vsop87.txt"
fetch "$vsop/vsop87.chk" "$data/vsop87/vsop87.chk"

# ELP2000-82B: the 36 series files, the catalog ReadMe, and the BDL
# reference implementation + notice (record formats and constants).
for n in $(seq 1 36); do
    fetch "$elp/ELP$n" "$data/elp2000/ELP$n"
done
fetch "$elp/ReadMe" "$data/elp2000/ReadMe"
fetch_gz "$elp/elp82b.f.gz" "$data/elp2000/elp82b.f"
fetch_gz "$elp/example.f.gz" "$data/elp2000/example.f"
fetch_gz "$elp/elp82b.ps.gz" "$data/elp2000/elp82b.ps"

echo "analytic-theory data ready under $data/{vsop87,elp2000}"
