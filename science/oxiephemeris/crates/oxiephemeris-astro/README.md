# oxiephemeris-astro

The astrology layer: chart angles, house systems, aspects, lunar
nodes/apogee, essential dignities, Arabic Parts, midpoints, declinations,
synastry, and the sidereal zodiac.

Pure spherical trigonometry and published mean-element polynomials on top
of `oxiephemeris-bodies` — no data files for almost all of it. This is the
top of the workspace's astronomy stack: `oxiephemeris-chart` (the
CLI/Python/WASM-shared chart facade), `oxiephemeris-cli`, and
`oxiephemeris-compat` all build natal charts, synastry, and transits out of
this crate's functions. `no_std` by design (enable `std` for `std`-only
conveniences); all float math in library paths goes through `libm`.

Almost every function here takes its geometry as plain angles — `theta`
(local apparent sidereal time / RAMC), `phi` (geographic latitude), `eps`
(obliquity), and ecliptic longitudes — rather than an epoch or a DE file, so
most of the crate (angles, houses, aspects, ayanamsha, zodiac, motion,
dignities, parts, midpoints, declination, synastry) has no ephemeris
dependency at all: callers resolve an epoch to those angles themselves
(typically via `oxiephemeris-bodies`' `sidereal`/`frames` modules, as in the
example below). The one exception is `nodes::true_node_of_date`, which
needs a `&DeFile` to read `GM_EarthMoon` from the header for the osculating
lunar orbit; the mean node/apogee/perigee in the same module are closed-form
polynomials and need no file.

## What's here

- `angles` — Ascendant, MC, Vertex, and East Point from `(theta, phi, eps)`.
- `houses` — house cusps (`cusps`) for 7 systems: Placidus, Koch, Whole
  Sign, Equal, Porphyry, Regiomontanus, Campanus (`HouseSystem`).
- `aspects` — aspect detection (`find_aspect`) with a configurable
  `OrbPolicy` and applying/separating classification from daily speeds.
- `nodes` — mean lunar node/apogee/perigee (polynomials of `t`) and the
  true (osculating) node/apogee (`true_node_of_date`, the one function
  here that needs a `&DeFile`).
- `ayanamsha` — sidereal-zodiac offsets (`Ayanamsha`: Fagan/Bradley,
  Lahiri, Krishnamurti, Raman, a J2000-zero convention, or a
  caller-supplied custom anchor), propagated from cited anchor epochs via
  the IAU 2006 general-precession polynomial.
- `zodiac` — the twelve tropical signs, their element/modality, and
  longitude → sign/degree/DMS decomposition (`Sign`, `SignPosition`).
- `motion` — direct/retrograde/stationary classification from a daily
  longitude speed (`MotionState`).
- `dignities` — essential dignities (domicile/exaltation/triplicity/term/
  face rulers, detriment/fall/peregrine) and element/modality
  `Distribution`.
- `parts` — day/night `Sect` and the Arabic Parts (`part_of_fortune`,
  `part_of_spirit`, and the generic `lot`).
- `midpoints` — near/far ecliptic midpoints, plus a batch helper for
  composite-chart construction.
- `declination` — ecliptic→equatorial `declination`, `antiscia`/
  `contra_antiscia`, and parallel/contraparallel/out-of-bounds
  `DeclinationAspect`.
- `synastry` — cross-aspects between two body sets
  (`for_each_cross_aspect`, `cross_aspects_into`), feeding synastry,
  transit, and progression comparisons.
- `stars` — a committed 177-star Hipparcos (V≤3) subset feeding
  `oxiephemeris_bodies::star`'s apparent-star pipeline.

## Usage

```rust
use oxiephemeris_astro::angles::local_sidereal_time;
use oxiephemeris_astro::houses::{cusps, HouseSystem};
use oxiephemeris_astro::{ascendant, mc, SignPosition};
use oxiephemeris_bodies::frames::{mean_obliquity_iau2006, nutation_iau2000a};
use oxiephemeris_bodies::sidereal::gast_iau2006;
use oxiephemeris_core::time::{delta_t_seconds, julday, utc_to_tt, Calendar, J2000_JD};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The Unix epoch (1970-01-01T00:00:00Z) at the Royal Observatory,
    // Greenwich -- the workspace's canonical reference chart.
    let (lat_deg, lon_deg) = (51.4779_f64, 0.0_f64);

    let jd_utc = julday(Calendar::Gregorian, 1970, 1, 1, 0.0).expect("valid calendar date");
    let jd_tt = utc_to_tt(jd_utc).unwrap_or_else(|_| jd_utc.add_seconds(delta_t_seconds(1970.0)));
    let t_tt = ((jd_tt.hi - J2000_JD) + jd_tt.lo) / 36_525.0;

    // No DE file needed: sidereal time and nutation are both closed-form
    // series in `t_tt`.
    let gast_rad = gast_iau2006(jd_utc, t_tt); // dut1 = 0 (UT1 ~ UTC)
    let theta = local_sidereal_time(gast_rad, lon_deg.to_radians());
    let nut = nutation_iau2000a(t_tt);
    let eps = mean_obliquity_iau2006(t_tt) + nut.deps_rad; // true obliquity of date

    let phi = lat_deg.to_radians();
    let asc = ascendant(theta, phi, eps)?;
    let mc_val = mc(theta, eps);
    let cusps = cusps(HouseSystem::Placidus, theta, phi, eps)?;

    let sp = SignPosition::of(asc);
    println!("Ascendant: {} {:.2} deg", sp.sign.name(), sp.degrees_in_sign);
    println!("MC: cusp[10] = {:.2} deg absolute", cusps[9].to_degrees());
    assert_eq!(cusps[9].to_bits(), mc_val.to_bits());
    // -> Ascendant: Libra 7.22 deg
    // -> MC: cusp[10] = 99.40 deg absolute
    Ok(())
}
```

## Feature flags

- `std` (off by default): forwards to `oxiephemeris-core/std`,
  `oxiephemeris-bodies/std`, and `oxiephemeris-de/std`. Without it the
  crate is `#![no_std]`.

See the [main OxiEphemeris README](../../README.md) for the full workspace
overview.

Licensed under Apache-2.0 (see workspace [LICENSE](../../LICENSE)).
