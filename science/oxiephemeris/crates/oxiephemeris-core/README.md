# oxiephemeris-core

Two-part Julian Date/calendar arithmetic, TAI/TT/TDB time scales, and IERS
leap seconds.

This is the foundational time-scale layer used by every other crate in the
OxiEphemeris workspace: `oxiephemeris-de` evaluates its Chebyshev series at
two-part Julian dates built here, and every higher layer (`-bodies`,
`-astro`) takes its epochs as a `time::JulianDate` or the equivalent
`(hi, lo)` tuple. `no_std` by default (enable `std` for
`std::error::Error` on [`CoreError`]); every algorithm — the proleptic
Gregorian/Julian calendar conversions, the embedded UTC leap-second table,
the Fairhead–Bretagnon TDB−TT series — is implemented from published
standards (IERS Conventions 2010, the *Explanatory Supplement to the
Astronomical Almanac*), never from SOFA/ERFA source.

## What's here

- `time::JulianDate` — an extended-precision two-part Julian Date
  (`hi + lo`, canonicalized with an error-free `TwoSum`): a single `f64`
  JD cannot represent contemporary dates below ≈ 40 µs, far too coarse for
  this workspace's microarcsecond-class targets.
- `time::julday` / `time::revjul` — proleptic Gregorian *and* Julian
  calendar ↔ Julian Date conversions, BCE sign-correct (astronomical year
  numbering: year 0 = 1 BCE).
- `time::{utc_to_tai, tai_to_tt, utc_to_tt, ...}` and the embedded
  `time::LEAP_SECONDS` table (1972-01-01 onward; earlier UTC epochs are
  out of scope and return `CoreError::DateOutOfRange`).
- `time::tdb_minus_tt_seconds` — the Fairhead–Bretagnon TDB−TT analytical
  series.
- `time::delta_t_seconds` — the Espenak–Meeus ΔT (TT−UT1) polynomial, for
  epochs where a measured UT1−UTC isn't available.
- `angle` — `DEG2RAD`/`AS2R`/... unit constants and the
  `normalize_0_two_pi`/`normalize_pm_pi` wrapping helpers used throughout
  the workspace.

## Usage

```rust
use oxiephemeris_core::time::{julday, revjul, utc_to_tt, Calendar, SECONDS_PER_DAY};
use oxiephemeris_core::CoreError;

fn main() -> Result<(), CoreError> {
    // The Unix epoch: proleptic Gregorian calendar -> two-part Julian Date.
    let jd_unix = julday(Calendar::Gregorian, 1970, 1, 1, 0.0)?;
    assert_eq!(jd_unix.value(), 2_440_587.5);

    // UTC -> TT through the embedded leap-second table (valid from
    // 1972-01-01 onward).
    let jd_utc = julday(Calendar::Gregorian, 2020, 1, 1, 0.0)?;
    let jd_tt = utc_to_tt(jd_utc)?;
    let offset_s = jd_tt.diff_days(jd_utc) * SECONDS_PER_DAY;
    assert!((offset_s - 69.184).abs() < 1e-6); // 37 leap seconds + 32.184s TT-TAI

    let (date, hours) = revjul(jd_tt, Calendar::Gregorian)?;
    println!("{}-{:02}-{:02} {hours:.6}h TT", date.year, date.month, date.day);
    Ok(())
}
```

## Feature flags

- `std` (off by default): implements `std::error::Error` for `CoreError`.
  Without it the crate is `#![no_std]`.

See the [main OxiEphemeris README](../../README.md) for the full workspace
overview.

Licensed under Apache-2.0 (see workspace [LICENSE](../../LICENSE)).
