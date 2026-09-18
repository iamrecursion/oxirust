//! VSOP87E evaluation: barycentric rectangular states of the Sun and
//! the eight planets from the truncated committed series.
//!
//! Bretagnon P., Francou G. 1988, A&A 202, 309, "Planetary theories in
//! rectangular and spherical variables. VSOP87 solutions"; series files
//! and evaluation rules from the CDS VI/81 notice (`vsop87.txt`): each
//! coordinate is `Σ_alpha T^alpha Σ_j A_j cos(B_j + C_j·T)` with `T` in
//! thousands of Julian years (TDB) from J2000. Velocities here are the
//! exact analytic derivative of the truncated series.

use crate::{rotate_to_fk5, vsop87_tables, J2000_JD};

/// Days per thousand Julian years (the unit of the VSOP87 time `T`).
const DAYS_PER_TJY: f64 = 365_250.0;

/// One body's series: `[coordinate][alpha]` slices of `(A, B, C)`.
type BodyTables = [[&'static [(f64, f64, f64)]; 6]; 3];

/// Exact small-integer alphas for the Poisson product rule.
const ALPHA: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

/// A body carried by the VSOP87E solution.
///
/// There is no Pluto entry: no VSOP87 variant models Pluto (a DE or SPK
/// file is required for it), and the Moon lives in [`crate::elp2000`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VsopBody {
    /// The Sun's barycentric wobble (the reflex of all the planets).
    Sun,
    /// Mercury's barycentric rectangular position.
    Mercury,
    /// Venus's barycentric rectangular position.
    Venus,
    /// Earth's barycentric rectangular position.
    Earth,
    /// Mars's barycentric rectangular position.
    Mars,
    /// Jupiter's barycentric rectangular position.
    Jupiter,
    /// Saturn's barycentric rectangular position.
    Saturn,
    /// Uranus's barycentric rectangular position.
    Uranus,
    /// Neptune's barycentric rectangular position.
    Neptune,
}

impl VsopBody {
    fn tables(self) -> &'static BodyTables {
        match self {
            Self::Sun => &vsop87_tables::sun::TABLES,
            Self::Mercury => &vsop87_tables::mercury::TABLES,
            Self::Venus => &vsop87_tables::venus::TABLES,
            Self::Earth => &vsop87_tables::earth::TABLES,
            Self::Mars => &vsop87_tables::mars::TABLES,
            Self::Jupiter => &vsop87_tables::jupiter::TABLES,
            Self::Saturn => &vsop87_tables::saturn::TABLES,
            Self::Uranus => &vsop87_tables::uranus::TABLES,
            Self::Neptune => &vsop87_tables::neptune::TABLES,
        }
    }
}

/// Barycentric state `[x, y, z, dx, dy, dz]` in au and au/day,
/// dynamical ecliptic and equinox J2000 (the theory's native frame).
///
/// `jd_tdb` is a two-part TDB Julian date (`hi + lo`); the split is
/// only a convenience — the series argument is a plain `f64`, whose
/// ~1e-9-day quantization is far below the theory's accuracy.
#[must_use]
pub fn state_ecliptic_au(body: VsopBody, jd_tdb: (f64, f64)) -> [f64; 6] {
    let t = ((jd_tdb.0 - J2000_JD) + jd_tdb.1) / DAYS_PER_TJY;
    let tables = body.tables();
    let mut state = [0.0; 6];
    for (coord, series) in tables.iter().enumerate() {
        let mut pos = 0.0;
        let mut rate_per_tjy = 0.0;
        // T^alpha and T^(alpha-1), advanced together so the Poisson
        // product rule d(T^alpha·S)/dT = alpha·T^(alpha-1)·S + T^alpha·S'
        // needs no branch except at alpha = 0 (where the first summand
        // vanishes identically).
        let mut t_pow = 1.0;
        let mut t_pow_prev = 0.0;
        for (alpha, terms) in series.iter().enumerate() {
            let mut sum = 0.0;
            let mut dsum = 0.0;
            for &(a, b, c) in *terms {
                let phase = b + c * t;
                sum += a * libm::cos(phase);
                dsum -= a * c * libm::sin(phase);
            }
            pos += t_pow * sum;
            rate_per_tjy += t_pow * dsum;
            // ALPHA[0]·t_pow_prev is the vanishing first summand of
            // the product rule.
            rate_per_tjy += ALPHA[alpha] * t_pow_prev * sum;
            t_pow_prev = t_pow;
            t_pow *= t;
        }
        state[coord] = pos;
        state[coord + 3] = rate_per_tjy / DAYS_PER_TJY;
    }
    state
}

/// Barycentric state `[x, y, z, dx, dy, dz]` in au and au/day, FK5
/// equatorial J2000 axes (≈ ICRS to ~0.02″; see
/// [`crate::ECL_J2000_TO_FK5`]).
#[must_use]
pub fn state_au(body: VsopBody, jd_tdb: (f64, f64)) -> [f64; 6] {
    rotate_to_fk5(state_ecliptic_au(body, jd_tdb))
}
