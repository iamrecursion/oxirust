# oxiephemeris-bodies

Reference-frame mathematics and the apparent-place pipeline for
solar-system bodies and fixed stars: IAU 2006 precession, full IAU 2000A
nutation, sidereal time, and topocentric observers, evaluated on top of a
JPL DE ephemeris.

This is the middle layer of the OxiEphemeris workspace: it consumes the raw
Chebyshev states from `oxiephemeris-de` and turns them into an apparent
place (light-time, aberration, gravitational deflection, frame rotation),
which `oxiephemeris-astro` in turn consumes for chart angles, houses, and
aspects. `no_std` by default (enable `std` for `std`-only conveniences);
every algorithm in library paths goes through [`libm`], never `std`'s
floating-point intrinsics, so the crate builds identically with or without
the feature.

Everything here is implemented from published standards and papers —
IERS Conventions (2010) TN36 chapter 5, Capitaine/Wallace/Chapront (2003)
for the P03/IAU 2006 precession solution, Hilton et al. (2006), McCarthy &
Luzum (2003) for the IAU 2000B abridged-nutation criterion, Simon et al.
(1994) for the fundamental arguments, and Kaplan et al. (1989, AJ 97, 1197)
/ USNO Circular 179 for the apparent-place pipeline itself — never from
SOFA, ERFA, or Swiss Ephemeris source.

## What's here

- `math` — `Vec3`/`Mat3`, the IERS `R1`/`R2`/`R3` rotation convention, and
  spherical ↔ Cartesian helpers, used throughout the rest of the workspace.
- `frames` — frame bias + IAU 2006 precession (`fw_angles_iau2006`,
  `precession_bias_matrix`), the full IAU `2000A_R06` nutation series
  (`nutation_iau2000a`, the default) plus a ≈15×-cheaper IAU-2000B-accuracy
  truncation (`nutation_iau2000a_truncated`, selected via `NutationModel`),
  and IAU 2006 mean obliquity (`mean_obliquity_iau2006`).
- `sidereal` — Earth Rotation Angle (`era`), GMST and GAST
  (`gmst_iau2006`/`gast_iau2006`, with the equation of the equinoxes and
  its complementary terms) — no DE file needed, only a UT1 Julian Date and
  `t`, Julian centuries TT.
- `apparent` — `apparent`/`apparent_topocentric`: light-time iteration,
  relativistic annual aberration, optional solar gravitational light
  deflection, and frame rotation to one of five output frames (`Frame`),
  driven by an `Options` value (`center`, `frame`, `aberration`,
  `deflection`, `light_time`, `with_speed`, `nutation`). `Options` is
  `#[non_exhaustive]` but every field is `pub`, so build one with
  `Options::default()` and assign the fields you want to change (or use
  `Options::new` to set all seven explicitly).
- `topocentric` — WGS84 geodetic site → ITRS (`Observer`), the polar-motion
  matrix `W(t)` with the TIO locator `s'`, an `Eop` (`dut1_s`, `xp_arcsec`,
  `yp_arcsec`) struct, and station GCRS state (diurnal aberration +
  parallax) feeding `apparent_topocentric`.
- `star` — `apparent_star`/`apparent_star_topocentric`: the same pipeline
  applied to a `CatalogStar` (position, proper motion, parallax, radial
  velocity) instead of a DE body, via the classical Kaplan (1989) method.

## Usage

```rust,no_run
use oxiephemeris_bodies::apparent::{apparent, BodiesError, Frame, Options, Target};
use oxiephemeris_core::time::{delta_t_seconds, julday, utc_to_tt, Calendar};
use oxiephemeris_de::DeFile;

fn main() -> Result<(), BodiesError> {
    let bytes = std::fs::read("data/de440/linux_p1550p2650.440").expect("read de440 file");
    let de = DeFile::parse(&bytes)?;

    // The Unix epoch (1970-01-01T00:00:00Z); it predates the 1972
    // leap-second table, so fall back to the Espenak-Meeus Delta T
    // polynomial (the same fallback oxiephemeris-chart uses internally).
    let jd_utc = julday(Calendar::Gregorian, 1970, 1, 1, 0.0).expect("valid calendar date");
    let jd_tt = utc_to_tt(jd_utc).unwrap_or_else(|_| jd_utc.add_seconds(delta_t_seconds(1970.0)));

    let mut opts = Options::default();
    opts.frame = Frame::EclipticTrueOfDate; // SE-style apparent longitude
    opts.with_speed = true;
    let sun = apparent(&de, Target::Sun, jd_tt, opts)?;
    println!("Sun: lon={:.3} deg, r={:.5} au", sun.lon_rad.to_degrees(), sun.r_au);
    // -> Sun: lon=280.156 deg, r=0.98331 au
    // (280.156 deg = 10.16 deg into Capricorn, matching the natal-chart
    // example in the root and oxiephemeris-py READMEs.)
    Ok(())
}
```

## Feature flags

- `std` (off by default): forwards to `oxiephemeris-core/std` and
  `oxiephemeris-de/std`. Without it the crate is `#![no_std]`; all float
  math already goes through `libm` regardless of this feature.

See the [main OxiEphemeris README](../../README.md) for the full workspace
overview.

Licensed under Apache-2.0 (see workspace [LICENSE](../../LICENSE)).
