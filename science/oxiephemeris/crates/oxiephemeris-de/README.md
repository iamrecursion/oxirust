# oxiephemeris-de

JPL DE binary ephemeris kernel reader — the classic export format and the
SPK/DAF (`.bsp`) format — with Chebyshev interpolation.

This is the data layer of the OxiEphemeris workspace: `oxiephemeris-bodies`
evaluates its apparent-place pipeline on top of the states this crate
returns, and `oxiephemeris-astro` calls into it directly for the one
ephemeris-dependent quantity in the astrology layer (the GM constant behind
the true/osculating lunar node). Unlike every other layered crate here,
`oxiephemeris-de` has **no runtime dependency on `oxiephemeris-core`**:
`DeFile`'s own API takes epochs as a raw two-part `(f64, f64)` Julian Date
tuple (`hi + lo`, TDB), not `oxiephemeris_core::time::JulianDate` — the two
happen to share the same two-field shape, so a `JulianDate`'s `.hi`/`.lo`
slot in directly if you have one, but nothing forces it (`oxiephemeris-core`
appears only as a dev-dependency, for this crate's own tests). `no_std` by
default; the caller supplies the file bytes as `&[u8]` and owns how they got
there (`std::fs::read`, a memory-mapped file, bytes embedded with
`include_bytes!`, a `Vec<u8>` fetched over the network, ...).

The classic binary layout (`DeFile`) is implemented from the public-domain
JPL Fortran export utilities and papers only — `asc2eph.f` and `testeph.f`
(header/record layout, `INTERP`/`STATE`/`PLEPH`), Standish's DE405/LE405
export-format IOM, and Park et al.'s DE440/DE441 paper — verified against
the public-domain JPL `testpo.440`/`testpo.441` golden test files (see the
workspace root README for the measured error figures). The SPK/DAF reader
(`spk` module) covers segment Types 2 and 3 (Chebyshev) and Type 13 (Hermite
on doubled nodes).

## What's here

- `DeFile::parse` — parses a classic binary DE file (e.g.
  `linux_p1550p2650.440`) from a byte slice, auto-detecting endianness via
  an `NCON` sanity check.
- `DeFile::series_state` — evaluates one raw [`Series`] (Mercury..Pluto,
  Moon, Sun, nutation, libration, lunar Euler-angle rates, TT-TDB) at a
  two-part Julian Date via Clenshaw-recurrence Chebyshev interpolation,
  returning both value and rate.
- `DeFile::state_km` — SSB-centered `[x, y, z, dx, dy, dz]` state (km,
  km/day) for a [`Body`], deriving Earth and Moon from the stored
  Earth-Moon-barycenter and geocentric-Moon series (`EMRAT`).
- `DeFile::testpo_value` — reproduces the `PLEPH` target/center/coordinate
  semantics of the JPL `testpo` golden files, for oracle-style testing.
- `DeFile::span`, `de_number`, `au_km`, `emrat`, `constant`/`constants`,
  `title`, `pointer_triplet` — header introspection.
- `spk::SpkFile::parse` — parses a DAF container (little- or big-endian),
  walking the summary/name record chains; `segment_for`/`state`/
  `state_at_et` locate and evaluate a segment for a `(target, center)` pair
  at an ephemeris-seconds-past-J2000 epoch.
- `read_de_file` (needs the `std` feature) — a one-line `std::fs::read`
  convenience wrapper.

## Usage

```rust,no_run
use oxiephemeris_de::{Body, DeFile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("data/de440/linux_p1550p2650.440")?;
    let de = DeFile::parse(&bytes)?;
    assert_eq!(de.de_number(), 440);

    // DeFile's own API: a raw two-part JD tuple, TDB. This is the Unix
    // epoch's JD (see oxiephemeris-core's README); using it verbatim here
    // is illustrative, not a rigorous UTC/TDB conversion.
    let jd: (f64, f64) = (2_440_587.5, 0.0);
    let sun = de.state_km(Body::Sun, jd)?;
    println!(
        "Sun (SSB-centered): x={:.1} y={:.1} z={:.1} km",
        sun[0], sun[1], sun[2]
    );
    // -> Sun (SSB-centered): x=644175.8 y=255784.2 z=101011.4 km
    Ok(())
}
```

## Feature flags

- `std` (off by default): enables `read_de_file`, a convenience
  `std::fs::read` loader. Without it the crate is `#![no_std]`.

See the [main OxiEphemeris README](../../README.md) for the full workspace
overview.

Licensed under Apache-2.0 (see workspace [LICENSE](../../LICENSE)).
