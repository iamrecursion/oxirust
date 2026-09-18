//! Analytic fallback ephemerides: no data files required.
//!
//! Truncated, committed series of the two primary published analytic
//! theories:
//!
//! * **VSOP87E** (Bretagnon & Francou 1988, A&A 202, 309): barycentric
//!   rectangular coordinates of the Sun and the eight planets, dynamical
//!   ecliptic and equinox J2000.
//! * **ELP2000-82B** (Chapront-Touzé & Chapront 1983, A&A 124, 50;
//!   1988, A&A 190, 342): geocentric rectangular coordinates of the
//!   Moon, mean dynamical ecliptic and inertial equinox of J2000.
//!
//! Both theories are fitted to JPL DE200; against DE440 this crate is an
//! *arcsecond-class* fallback (measured gates in `tests/`), not a
//! replacement for the DE/SPK readers. **Pluto is not available** from
//! these theories — Pluto requires a DE or SPK file.
//!
//! Time arguments are TDB Julian dates as two-part `(hi, lo)` sums, as
//! in `oxiephemeris-de` (TT is fine at these accuracies: |TT−TDB| stays
//! below 2 ms, i.e. sub-meter even for Mercury).

#![no_std]

pub mod elp2000;
pub mod vsop87;

mod elp2000_tables;
mod vsop87_tables;

pub use elp2000::{geocentric_moon_ecliptic_km, geocentric_moon_km};
pub use vsop87::{state_au, state_ecliptic_au, VsopBody};

/// J2000.0 epoch as a Julian date (TDB).
pub(crate) const J2000_JD: f64 = 2_451_545.0;

/// The rotation from the dynamical ecliptic frame J2000 of the BDL
/// theories to the FK5 equatorial frame J2000, from the CDS `vsop87.txt`
/// notice (§ REFERENCE SYSTEM). FK5 agrees with the ICRS to ~0.02″ —
/// below both theories' own accuracy against modern JPL ephemerides.
/// Public so callers of the `*_ecliptic_*` functions can apply (or
/// invert) the exact same frame tie themselves.
pub const ECL_J2000_TO_FK5: [[f64; 3]; 3] = [
    [1.000_000_000_000, 0.000_000_440_360, -0.000_000_190_919],
    [-0.000_000_479_966, 0.917_482_137_087, -0.397_776_982_902],
    [0.0, 0.397_776_982_902, 0.917_482_137_087],
];

/// Applies [`ECL_J2000_TO_FK5`] to a position + velocity 6-vector.
pub(crate) fn rotate_to_fk5(state: [f64; 6]) -> [f64; 6] {
    let m = &ECL_J2000_TO_FK5;
    let mut out = [0.0; 6];
    for (i, row) in m.iter().enumerate() {
        out[i] = row[0] * state[0] + row[1] * state[1] + row[2] * state[2];
        out[i + 3] = row[0] * state[3] + row[1] * state[4] + row[2] * state[5];
    }
    out
}
