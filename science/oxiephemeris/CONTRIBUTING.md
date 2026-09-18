# Contributing to OxiEphemeris

## Clean-Room Provenance Statement (binding for all contributors)

This project is developed **without reference to the Swiss Ephemeris source
code or the SOFA source code**. Nobody working on this project may read,
consult, or copy from either codebase. All algorithms are implemented from
public standards and papers:

- IERS Conventions (2010)
- IAU 2006 precession (Capitaine et al. 2003, P03)
- IAU 2000A nutation (Mathews, Herring & Buffett 2002)
- Fairhead, Bretagnon & Lestrade (1988, IAU Symp. 128) / Fairhead &
  Bretagnon (1990) for TT<->TDB
- Espenak & Meeus ΔT polynomial expressions
- JPL DE format documentation and public-domain `testpo` files

Behavioral compatibility with Swiss Ephemeris is verified **only** against
published documentation and program *outputs* (committed fixture files).
Consulting SE documentation for API shape and flag semantics is permitted;
transcribing algorithm descriptions is not.

Every algorithm implementation MUST cite its source equation/paper in doc
comments.

## COOLJAPAN POLICY

Pure Rust / No Warnings / No Unwrap / No Unsafe / Workspace deps /
Latest crates / files < 2000 lines (use `splitrs`) / Apache-2.0.

Pre-commit:
```
cargo fmt --all
cargo clippy --all-features --all-targets -- -D warnings
cargo nextest run --all-features
```

## Data setup

The full test suite (and the CLI's DE-backed `pos`/`chart` commands) needs
several public-domain/published data files. None of these are committed to
the repository (see `.gitignore`'s `data/` entry) — they're large,
externally reproducible, and keeping them out preserves the clean-room
provenance story (only code and cited-paper-derived tables are checked in).

Three fetch scripts under `scripts/` retrieve most of what's needed, into
`data/` (all re-runnable; they skip files already present):

- `scripts/fetch_de440.sh` — the DE440 classic binary ephemeris
  (`linux_p1550p2650.440`), its `testpo.440` golden test-case file, DE440's
  Fortran/ASCII reference materials, the optional DE440t variant (integrated
  TT-TDB series), and the IERS Conventions (2010) truncated-nutation tables
  `tab5.3a`/`tab5.3b`.
- `scripts/fetch_de441.sh` — the DE441 classic binary ephemeris
  (`linux_m13000p17000.441`, ~2.7 GB) and its `testpo.441` golden test-case
  file, plus the IERS `finals2000A.all` measured UT1-UTC/polar-motion table.
  Prefers `aria2c` for parallel, resumable segmented downloads of the large
  DE441 binary, and falls back to a plain resumable `curl` when `aria2c`
  isn't installed.
- `scripts/fetch_analytic.sh` — the VSOP87E (CDS VI/81) and ELP2000-82B
  (CDS VI/79) source tables that feed the `xtask` coefficient generators and
  the `oxiephemeris-analytic` crate's oracle tests, plus each catalog's
  notice/reference files.

Two data files used by the test suite have **no fetch script yet** and must
be obtained manually today:

- The SPK/DAF `de440.bsp` kernel (exercised by `oxiephemeris-de`'s SPK
  reader tests).
- The Hipparcos `hip_main.dat` catalog (exercised by
  `oxiephemeris-astro`'s fixed-star tests).

This is a known, accurate gap, not an oversight to work around silently —
contributions adding fetch scripts for these two are welcome.
