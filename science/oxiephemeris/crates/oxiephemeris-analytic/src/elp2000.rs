//! ELP2000-82B evaluation: the geocentric lunar state from the
//! truncated committed series.
//!
//! Chapront-Touzé M., Chapront J. 1983, A&A 124, 50 ("The lunar
//! ephemeris ELP 2000") and 1988, A&A 190, 342 (ELP 2000-85); series
//! files, argument polynomials, DE200/LE200 fit constants and the
//! change of coordinates from the BDL reference subroutine `ELP82B`
//! (CDS VI/79). The series produce spherical coordinates on the mean
//! ecliptic and equinox of date; Laskar's ecliptic-precession
//! polynomials (`P`, `Q` below) then rotate the rectangular vector to
//! the **mean dynamical ecliptic and inertial equinox of J2000** — the
//! same J2000 ecliptic frame as VSOP87A/E, so the same
//! [`crate::ECL_J2000_TO_FK5`] frame tie applies unchanged.
//!
//! Velocities are five-point central differences of the rectangular
//! position (step [`SPEED_STEP_DAYS`]): with the Moon's ~0.23 rad/day
//! motion the stencil's `h⁴` term is ≲ 1e-4 km/day, far below the
//! series' own accuracy against a modern JPL ephemeris.

use core::f64::consts::PI;

use crate::{elp2000_tables as tables, rotate_to_fk5, J2000_JD};

/// Days per Julian century (the unit of the ELP time argument).
const DAYS_PER_CENTURY: f64 = 36_525.0;

/// Arcseconds per radian.
const RAD_AS: f64 = 648_000.0 / PI;

/// Half-step of the five-point velocity stencil, days.
pub const SPEED_STEP_DAYS: f64 = 0.02;

/// `ELP82B`'s `ath`: the distance unit of the series, km.
const ATH_KM: f64 = 384_747.980_674_316_5;

/// `ELP82B`'s `a0`: the fitted mean semi-major axis, km — series
/// distances are rescaled by `a0/ath`.
const A0_KM: f64 = 384_747.980_644_895_4;

/// Accumulated general precession in longitude, rad/century (the `ζ`
/// argument's rate on top of `W1`).
const PRECES: f64 = 5_029.096_6 / RAD_AS;

/// Laskar's ecliptic-precession polynomial `P` (rad, factors of t¹..t⁵).
const LASKAR_P: [f64; 5] = [
    0.101_803_91e-4,
    0.470_204_39e-6,
    -0.541_736_7e-9,
    -0.250_794_8e-11,
    0.463_486e-14,
];

/// Laskar's ecliptic-precession polynomial `Q` (rad, factors of t¹..t⁵).
const LASKAR_Q: [f64; 5] = [
    -0.113_469_002e-3,
    0.123_726_74e-6,
    0.126_541_7e-8,
    -0.137_180_8e-11,
    -0.320_334e-14,
];

/// Degrees/arcminutes/arcseconds → radians.
fn dms(d: f64, m: f64, s: f64) -> f64 {
    (d + m / 60.0 + s / 3600.0) * PI / 180.0
}

/// Degree-4 polynomial in `t` (coefficients per power of t, radians).
fn poly(c: &[f64; 5], tp: &[f64; 5]) -> f64 {
    c[0] * tp[0] + c[1] * tp[1] + c[2] * tp[2] + c[3] * tp[3] + c[4] * tp[4]
}

/// The fundamental arguments of `ELP82B`, as polynomial coefficient
/// arrays over t (TDB Julian centuries from J2000): radians, rad/cy,
/// rad/cy², ...
struct Arguments {
    /// W₁, the Moon's mean longitude.
    w1: [f64; 5],
    /// Delaunay D, l′, l, F (in the record order of every table).
    del: [[f64; 5]; 4],
    /// ζ = W₁ + accumulated precession (linear).
    zeta: [f64; 2],
    /// Mean longitudes of Mercury..Neptune (linear).
    p: [[f64; 2]; 8],
}

/// Builds the argument polynomials (constants verbatim from `ELP82B`).
#[allow(clippy::unreadable_literal)] // verbatim theory constants
fn arguments() -> Arguments {
    let w1 = [
        dms(218.0, 18.0, 59.95571),
        1732559343.73604 / RAD_AS,
        -5.8883 / RAD_AS,
        0.006604 / RAD_AS,
        -0.00003169 / RAD_AS,
    ];
    let w2 = [
        dms(83.0, 21.0, 11.67475),
        14643420.2632 / RAD_AS,
        -38.2776 / RAD_AS,
        -0.045047 / RAD_AS,
        0.00021301 / RAD_AS,
    ];
    let w3 = [
        dms(125.0, 2.0, 40.39816),
        -6967919.3622 / RAD_AS,
        6.3622 / RAD_AS,
        0.007625 / RAD_AS,
        -0.00003586 / RAD_AS,
    ];
    let eart = [
        dms(100.0, 27.0, 59.22059),
        129597742.2758 / RAD_AS,
        -0.0202 / RAD_AS,
        0.000009 / RAD_AS,
        0.00000015 / RAD_AS,
    ];
    let peri = [
        dms(102.0, 56.0, 14.42753),
        1161.2283 / RAD_AS,
        0.5327 / RAD_AS,
        -0.000138 / RAD_AS,
        0.0,
    ];
    let mut del = [[0.0; 5]; 4];
    for k in 0..5 {
        del[0][k] = w1[k] - eart[k]; // D
        del[1][k] = eart[k] - peri[k]; // l'
        del[2][k] = w1[k] - w2[k]; // l
        del[3][k] = w1[k] - w3[k]; // F
    }
    del[0][0] += PI;
    let zeta = [w1[0], w1[1] + PRECES];
    let p = [
        [dms(252.0, 15.0, 3.25986), 538101628.68898 / RAD_AS],
        [dms(181.0, 58.0, 47.28305), 210664136.43355 / RAD_AS],
        [eart[0], eart[1]],
        [dms(355.0, 25.0, 59.78866), 68905077.59284 / RAD_AS],
        [dms(34.0, 21.0, 5.34212), 10925660.42861 / RAD_AS],
        [dms(50.0, 4.0, 38.89694), 4399609.65932 / RAD_AS],
        [dms(314.0, 3.0, 18.01841), 1542481.19393 / RAD_AS],
        [dms(304.0, 20.0, 55.19575), 786550.32074 / RAD_AS],
    ];
    Arguments { w1, del, zeta, p }
}

/// Sums one main-problem table (`sin` series; the distance table's
/// cosine convention enters through `phase0 = π/2`).
fn sum_main(terms: &[(i8, i8, i8, i8, f64)], args: &Arguments, tp: &[f64; 5], phase0: f64) -> f64 {
    let mut sum = 0.0;
    for &(i1, i2, i3, i4, a) in terms {
        let mut y = phase0;
        let ilu = [f64::from(i1), f64::from(i2), f64::from(i3), f64::from(i4)];
        for (mult, del) in ilu.iter().zip(&args.del) {
            y += mult * poly(del, tp);
        }
        sum += a * libm::sin(y);
    }
    sum
}

/// Sums one ζ-group table: argument `phase + iz·ζ + Σ ilu·Delaunay`
/// (linear polynomials), amplitude scaled by `t_scale`.
fn sum_zeta(
    terms: &[(i8, i8, i8, i8, i8, f64, f64)],
    args: &Arguments,
    t: f64,
    t_scale: f64,
) -> f64 {
    let mut sum = 0.0;
    for &(iz, i1, i2, i3, i4, phase, a) in terms {
        let mut y = phase + f64::from(iz) * (args.zeta[0] + args.zeta[1] * t);
        let ilu = [f64::from(i1), f64::from(i2), f64::from(i3), f64::from(i4)];
        for (mult, del) in ilu.iter().zip(&args.del) {
            y += mult * (del[0] + del[1] * t);
        }
        sum += a * t_scale * libm::sin(y);
    }
    sum
}

/// Sums one planetary table. Table 1 (`second_set = false`): 8
/// planetary longitudes then (D, l, F). Table 2 (`second_set = true`):
/// 7 planetary longitudes then (D, l′, l, F). All arguments linear.
fn sum_planetary(
    terms: &[([i8; 11], f64, f64)],
    args: &Arguments,
    t: f64,
    t_scale: f64,
    second_set: bool,
) -> f64 {
    let mut sum = 0.0;
    for &(ipla, phase, a) in terms {
        let mut y = phase;
        if second_set {
            for (i, p) in args.p.iter().take(7).enumerate() {
                y += f64::from(ipla[i]) * (p[0] + p[1] * t);
            }
            for (i, del) in args.del.iter().enumerate() {
                y += f64::from(ipla[7 + i]) * (del[0] + del[1] * t);
            }
        } else {
            for (i, p) in args.p.iter().enumerate() {
                y += f64::from(ipla[i]) * (p[0] + p[1] * t);
            }
            // (D, l, F) — the main problem's arguments 1, 3 and 4.
            for (i, del_index) in [0usize, 2, 3].iter().enumerate() {
                y += f64::from(ipla[8 + i])
                    * (args.del[*del_index][0] + args.del[*del_index][1] * t);
            }
        }
        sum += a * t_scale * libm::sin(y);
    }
    sum
}

/// Geocentric rectangular position (km) on the mean dynamical ecliptic
/// and inertial equinox of J2000.
fn position_km(jd_tdb: (f64, f64)) -> [f64; 3] {
    let t = ((jd_tdb.0 - J2000_JD) + jd_tdb.1) / DAYS_PER_CENTURY;
    let tp = [1.0, t, t * t, t * t * t, t * t * t * t];
    let args = arguments();

    let pis2 = PI / 2.0;
    let mut lon_as = sum_main(tables::MAIN_LON, &args, &tp, 0.0);
    let mut lat_as = sum_main(tables::MAIN_LAT, &args, &tp, 0.0);
    let mut dist_km = sum_main(tables::MAIN_DIST, &args, &tp, pis2);

    lon_as += sum_zeta(tables::ZETA_LON_T0, &args, t, 1.0)
        + sum_zeta(tables::ZETA_LON_T1, &args, t, t)
        + sum_zeta(tables::ZETA_LON_T2, &args, t, t * t);
    lat_as += sum_zeta(tables::ZETA_LAT_T0, &args, t, 1.0)
        + sum_zeta(tables::ZETA_LAT_T1, &args, t, t)
        + sum_zeta(tables::ZETA_LAT_T2, &args, t, t * t);
    dist_km += sum_zeta(tables::ZETA_DIST_T0, &args, t, 1.0)
        + sum_zeta(tables::ZETA_DIST_T1, &args, t, t)
        + sum_zeta(tables::ZETA_DIST_T2, &args, t, t * t);

    lon_as += sum_planetary(tables::PLAN1_LON_T0, &args, t, 1.0, false)
        + sum_planetary(tables::PLAN1_LON_T1, &args, t, t, false)
        + sum_planetary(tables::PLAN2_LON_T0, &args, t, 1.0, true)
        + sum_planetary(tables::PLAN2_LON_T1, &args, t, t, true);
    lat_as += sum_planetary(tables::PLAN1_LAT_T0, &args, t, 1.0, false)
        + sum_planetary(tables::PLAN1_LAT_T1, &args, t, t, false)
        + sum_planetary(tables::PLAN2_LAT_T0, &args, t, 1.0, true)
        + sum_planetary(tables::PLAN2_LAT_T1, &args, t, t, true);
    dist_km += sum_planetary(tables::PLAN1_DIST_T0, &args, t, 1.0, false)
        + sum_planetary(tables::PLAN1_DIST_T1, &args, t, t, false)
        + sum_planetary(tables::PLAN2_DIST_T0, &args, t, 1.0, true)
        + sum_planetary(tables::PLAN2_DIST_T1, &args, t, t, true);

    // Spherical on the mean ecliptic and equinox of date: the series
    // longitude rides on top of W₁, the distance on a0/ath.
    let lon = lon_as / RAD_AS + poly(&args.w1, &tp);
    let lat = lat_as / RAD_AS;
    let r = dist_km * (A0_KM / ATH_KM);

    let x1 = r * libm::cos(lat) * libm::cos(lon);
    let x2 = r * libm::cos(lat) * libm::sin(lon);
    let x3 = r * libm::sin(lat);

    // Laskar's ecliptic precession, of date → J2000 (verbatim ELP82B).
    let pw = (LASKAR_P[0]
        + LASKAR_P[1] * t
        + LASKAR_P[2] * tp[2]
        + LASKAR_P[3] * tp[3]
        + LASKAR_P[4] * tp[4])
        * t;
    let qw = (LASKAR_Q[0]
        + LASKAR_Q[1] * t
        + LASKAR_Q[2] * tp[2]
        + LASKAR_Q[3] * tp[3]
        + LASKAR_Q[4] * tp[4])
        * t;
    let ra = 2.0 * libm::sqrt(1.0 - pw * pw - qw * qw);
    let pwqw = 2.0 * pw * qw;
    let pw2 = 1.0 - 2.0 * pw * pw;
    let qw2 = 1.0 - 2.0 * qw * qw;
    let pw = pw * ra;
    let qw = qw * ra;
    [
        pw2 * x1 + pwqw * x2 + pw * x3,
        pwqw * x1 + qw2 * x2 - qw * x3,
        -pw * x1 + qw * x2 + (pw2 + qw2 - 1.0) * x3,
    ]
}

/// Geocentric lunar state `[x, y, z, dx, dy, dz]` in km and km/day,
/// mean dynamical ecliptic and inertial equinox of J2000 (the theory's
/// native frame).
///
/// `jd_tdb` is a two-part TDB Julian date (`hi + lo`).
#[must_use]
pub fn geocentric_moon_ecliptic_km(jd_tdb: (f64, f64)) -> [f64; 6] {
    let position = position_km(jd_tdb);
    let h = SPEED_STEP_DAYS;
    let at = |dt: f64| position_km((jd_tdb.0, jd_tdb.1 + dt));
    let (m2, m1, p1, p2) = (at(-2.0 * h), at(-h), at(h), at(2.0 * h));
    let mut state = [0.0; 6];
    for i in 0..3 {
        state[i] = position[i];
        state[i + 3] = (m2[i] - 8.0 * m1[i] + 8.0 * p1[i] - p2[i]) / (12.0 * h);
    }
    state
}

/// Geocentric lunar state `[x, y, z, dx, dy, dz]` in km and km/day,
/// FK5 equatorial J2000 axes (≈ ICRS to ~0.02″; see
/// [`crate::ECL_J2000_TO_FK5`]).
#[must_use]
pub fn geocentric_moon_km(jd_tdb: (f64, f64)) -> [f64; 6] {
    rotate_to_fk5(geocentric_moon_ecliptic_km(jd_tdb))
}
