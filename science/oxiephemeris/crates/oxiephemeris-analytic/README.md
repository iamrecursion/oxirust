# oxiephemeris-analytic

A data-file-free analytic fallback ephemeris: truncated VSOP87E (Sun and
the eight planets) and ELP2000-82B (Moon) — no JPL DE/SPK kernel required.

Two committed, truncated series of the published analytic theories:

- **VSOP87E** (Bretagnon & Francou 1988, A&A 202, 309): barycentric
  rectangular coordinates of the Sun and the eight planets, dynamical
  ecliptic and equinox J2000. There is no Pluto entry — no VSOP87 variant
  models Pluto; a DE or SPK file is required for it.
- **ELP2000-82B** (Chapront-Touzé & Chapront 1983, A&A 124, 50; 1988, A&A
  190, 342): geocentric rectangular coordinates of the Moon, mean dynamical
  ecliptic and inertial equinox of J2000.

Both theories are fit to JPL DE200; against DE440 this crate lands at an
*arcsecond-class* accuracy (measured in `tests/`), not a replacement for
`oxiephemeris-de`. It is unconditionally `#![no_std]` — there is no `std`
feature at all — and its only dependency is `libm`; unlike `-de`,
`-bodies`, and `-astro`, it doesn't depend on `oxiephemeris-core` either.
**No other crate in this workspace currently depends on
`oxiephemeris-analytic`**: it is not wired into the CLI, the chart facade,
or the Python/WASM bindings, all of which compute against a real JPL DE
file via `oxiephemeris-de`/`oxiephemeris-bodies`. Depend on this crate
directly if you want approximate ephemerides without shipping or fetching a
DE/SPK kernel.

## What's here

- `vsop87::VsopBody` — `Sun`, `Mercury`, `Venus`, `Earth`, `Mars`,
  `Jupiter`, `Saturn`, `Uranus`, `Neptune` (no Pluto).
- `vsop87::state_ecliptic_au` — barycentric `[x, y, z, dx, dy, dz]` state
  (AU, AU/day) in the theory's native frame (dynamical ecliptic and
  equinox J2000).
- `vsop87::state_au` — the same state rotated to FK5 equatorial J2000 axes
  (FK5 agrees with the ICRS to ≈ 0.02″, via `ECL_J2000_TO_FK5`).
- `elp2000::geocentric_moon_ecliptic_km` /
  `elp2000::geocentric_moon_km` — geocentric lunar state (km, km/day) in
  the theory's native frame, and rotated to FK5 equatorial J2000.
- `ECL_J2000_TO_FK5` — the dynamical-ecliptic-J2000 → FK5-equatorial-J2000
  rotation matrix used by both `*_au`/`*_km` functions, public so callers
  of the `*_ecliptic_*` variants can apply (or invert) it themselves.

All position/velocity functions take `jd_tdb: (f64, f64)`, a plain
two-part Julian Date — the same shape `oxiephemeris-de` uses, but this
crate has no dependency on it or on `oxiephemeris-core`. TT is fine in
place of TDB at these accuracies (`|TT-TDB|` stays below 2 ms, i.e.
sub-meter even for Mercury).

## Usage

```rust
use oxiephemeris_analytic::{geocentric_moon_km, state_au, VsopBody};

fn main() {
    // 1970-01-01T00:00, a plain two-part JD tuple -- no oxiephemeris-core
    // dependency needed.
    let jd: (f64, f64) = (2_440_587.5, 0.0);

    let sun = state_au(VsopBody::Sun, jd);
    let earth = state_au(VsopBody::Earth, jd);
    let geo_sun = [sun[0] - earth[0], sun[1] - earth[1], sun[2] - earth[2]];
    let r_au = (geo_sun[0] * geo_sun[0] + geo_sun[1] * geo_sun[1] + geo_sun[2] * geo_sun[2]).sqrt();
    println!("Sun (Earth->Sun, FK5 equatorial J2000), r = {r_au:.5} au");
    // -> Sun (Earth->Sun, FK5 equatorial J2000), r = 0.98331 au

    let moon = geocentric_moon_km(jd);
    let r_km = (moon[0] * moon[0] + moon[1] * moon[1] + moon[2] * moon[2]).sqrt();
    println!("Moon (geocentric, FK5 equatorial J2000), r = {r_km:.1} km");
    // -> Moon (geocentric, FK5 equatorial J2000), r = 392028.2 km
}
```

## Measured accuracy

Barycentric states carry an inherent ≈ 1e-5 au common-mode offset (the
solar-system barycenter itself shifted between DE200, which both theories
are fit to, and DE440). Geocentric differences — the way a fallback
ephemeris is actually consumed — cancel that common mode: cross-checked
against JPL DE440 over 1900–2100, the geocentric Sun is good to
≈ 1.9e-7 au (≈ 0.04″) and the Moon to ≈ 2.7 km / ≈ 0.64 km/day; see
`tests/de440_oracle.rs` for the full per-body, per-epoch-span gates
(including the wider 1650–2400 / 1700–2200 long-span figures).

See the [main OxiEphemeris README](../../README.md) for the full workspace
overview.

Licensed under Apache-2.0 (see workspace [LICENSE](../../LICENSE)).
