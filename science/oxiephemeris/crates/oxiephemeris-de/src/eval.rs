//! Epoch normalization, record/granule lookup, series interpolation, and
//! the barycentric / `testpo` state combinations.
//!
//! The algorithms replicate the public-domain JPL `testeph.f` subroutines:
//! `SPLIT` (two-part epoch normalization), `STATE` (record lookup and the
//! within-record fraction `T(1)`), `INTERP` (granule selection, normalized
//! Chebyshev argument `TC`, velocity scale `VFAC = 2*NA/T(2)`), and `PLEPH`
//! (target/center numbering, the Earth/Moon derivation from the Earth-Moon
//! barycenter with `EMRAT`, and the AU conversion used by `testpo` files).

use crate::cheby::{chebyshev_rate, chebyshev_value, f64_to_usize_trunc, usize_to_f64};
use crate::parse::{f64_at, DeFile, MAX_NCF};
use crate::{Body, DeError, Series, SeriesState};

/// Splits `t` into a whole part (truncated toward negative infinity for
/// negative inputs) and a fraction in `[0, 1)`; `SPLIT` of `testeph.f`.
fn split(t: f64) -> (f64, f64) {
    let ipart = libm::trunc(t);
    let fpart = t - ipart;
    if fpart < 0.0 {
        (ipart - 1.0, fpart + 1.0)
    } else {
        (ipart, fpart)
    }
}

/// Normalizes a two-part JD `(hi, lo)` into `(whole, frac)` where `whole`
/// is aligned to a half-integer JD (record boundaries fall on `*.5`) and
/// `frac` is in `[0, 1)`.  Mirrors the `PJD` computation in `STATE` of
/// `testeph.f`: the big-magnitude parts combine exactly, preserving the
/// caller's extended precision in `frac`.
fn normalize_epoch(jd: (f64, f64)) -> (f64, f64) {
    let (hi_whole, hi_frac) = split(jd.0 - 0.5);
    let (lo_whole, lo_frac) = split(jd.1);
    let mut whole = hi_whole + lo_whole + 0.5;
    let (carry, frac) = split(hi_frac + lo_frac);
    whole += carry;
    (whole, frac)
}

/// Picks coordinate `coord` (1..=6) out of a position+velocity 6-vector.
fn component6(state: [f64; 6], coord: u32) -> Result<f64, DeError> {
    match coord {
        1 => Ok(state[0]),
        2 => Ok(state[1]),
        3 => Ok(state[2]),
        4 => Ok(state[3]),
        5 => Ok(state[4]),
        6 => Ok(state[5]),
        _ => Err(DeError::BadArgument),
    }
}

/// Maps a `testpo`/`PLEPH` body number (1..=13) to a [`Body`].
fn testpo_body(number: u32) -> Result<Body, DeError> {
    match number {
        1 => Ok(Body::Mercury),
        2 => Ok(Body::Venus),
        3 => Ok(Body::Earth),
        4 => Ok(Body::Mars),
        5 => Ok(Body::Jupiter),
        6 => Ok(Body::Saturn),
        7 => Ok(Body::Uranus),
        8 => Ok(Body::Neptune),
        9 => Ok(Body::Pluto),
        10 => Ok(Body::Moon),
        11 => Ok(Body::Sun),
        12 => Ok(Body::Ssb),
        13 => Ok(Body::Emb),
        _ => Err(DeError::BadArgument),
    }
}

impl DeFile<'_> {
    /// Evaluates one raw Chebyshev series at the two-part JD `jd`
    /// (TDB; `jd.0 + jd.1`, split for extended precision).
    ///
    /// Units: km and km/day for bodies, radians and radians/day for
    /// nutation/libration/Euler rates, seconds and seconds/day for TT-TDB.
    ///
    /// # Errors
    /// * [`DeError::SeriesUnavailable`] — the series has no coefficients.
    /// * [`DeError::EpochOutOfRange`] — `jd` is outside the file span.
    /// * [`DeError::Corrupt`] — the located record does not bracket `jd`.
    /// * [`DeError::Truncated`] — the byte slice ends inside a record.
    pub fn series_state(&self, series: Series, jd: (f64, f64)) -> Result<SeriesState, DeError> {
        let lay = self.layout(series)?;
        let (whole, frac) = normalize_epoch(jd);
        let epoch = whole + frac;
        if epoch < self.ss[0] || epoch > self.ss[1] {
            return Err(DeError::EpochOutOfRange);
        }

        // Record index: NR = IDINT((PJD(1)-SS(1))/SS(3)), clamped so that
        // jd == stop JD lands in the last record (STATE of testeph.f).
        let idx_f = libm::trunc((whole - self.ss[0]) / self.ss[2]);
        let mut irec = f64_to_usize_trunc(idx_f);
        if irec >= self.nrec {
            irec = self.nrec - 1;
        }
        let rec = self.record_bytes(irec)?;

        // Defensive bracket check: every data record repeats its own span
        // in its first two doubles (GROUP 1070 of the ASCII format).
        let rec_t0 = f64_at(rec, 0, self.endian)?;
        let rec_t1 = f64_at(rec, 8, self.endian)?;
        if epoch < rec_t0 - 1e-9 || epoch > rec_t1 + 1e-9 {
            return Err(DeError::Corrupt);
        }

        // Fraction of the record interval, computed so the large-magnitude
        // subtraction happens first and is exact (both operands are
        // half-integer JDs), then the sub-day fraction is added.
        let rec_start = usize_to_f64(irec) * self.ss[2] + self.ss[0];
        let t1 = ((whole - rec_start) + frac) / self.ss[2];

        // Granule selection and normalized argument (INTERP of testeph.f):
        // TC = 2*(DMOD(NA*T1, 1) + DINT(T1)) - 1, in [-1, 1].
        let whole_t1 = libm::trunc(t1); // 1 only when jd == stop JD
        let scaled = lay.na_f * t1;
        let mut granule = f64_to_usize_trunc(scaled - whole_t1);
        if granule >= lay.na {
            granule = lay.na - 1;
        }
        let tc = 2.0 * (libm::fmod(scaled, 1.0) + whole_t1) - 1.0;

        // d(tc)/dt = 2*NA/span: VFAC of INTERP, in 1/day.
        let vfac = (lay.na_f + lay.na_f) / self.ss[2];

        let mut state = SeriesState {
            value: [0.0; 3],
            rate: [0.0; 3],
            ncomp: lay.ncomp,
        };
        let base = lay.start + granule * lay.ncf * lay.ncomp;
        let mut buf = [0.0_f64; MAX_NCF];
        for comp in 0..lay.ncomp {
            for (i, slot) in buf.iter_mut().enumerate().take(lay.ncf) {
                *slot = f64_at(rec, 8 * (base + comp * lay.ncf + i), self.endian)?;
            }
            let coeffs = buf.get(..lay.ncf).ok_or(DeError::Corrupt)?;
            let (value_slot, rate_slot) = state
                .value
                .iter_mut()
                .zip(state.rate.iter_mut())
                .nth(comp)
                .ok_or(DeError::Corrupt)?;
            *value_slot = chebyshev_value(coeffs, tc);
            *rate_slot = chebyshev_rate(coeffs, tc) * vfac;
        }
        Ok(state)
    }

    /// Evaluates a series into a `[x, y, z, dx, dy, dz]` 6-vector
    /// (unused components zero).
    fn series6(&self, series: Series, jd: (f64, f64)) -> Result<[f64; 6], DeError> {
        let s = self.series_state(series, jd)?;
        Ok([
            s.value[0], s.value[1], s.value[2], s.rate[0], s.rate[1], s.rate[2],
        ])
    }

    /// SSB-centered state `[x, y, z, dx, dy, dz]` in km and km/day.
    ///
    /// Earth and Moon are derived from the stored Earth-Moon-barycenter and
    /// geocentric-Moon series exactly as in `PLEPH` of `testeph.f`:
    /// `Earth = EMB - Moon_geo/(1 + EMRAT)`, `Moon = Earth + Moon_geo`.
    ///
    /// # Errors
    /// Same conditions as [`Self::series_state`].
    pub fn state_km(&self, body: Body, jd: (f64, f64)) -> Result<[f64; 6], DeError> {
        match body {
            Body::Ssb => Ok([0.0; 6]),
            Body::Mercury => self.series6(Series::Mercury, jd),
            Body::Venus => self.series6(Series::Venus, jd),
            Body::Mars => self.series6(Series::Mars, jd),
            Body::Jupiter => self.series6(Series::Jupiter, jd),
            Body::Saturn => self.series6(Series::Saturn, jd),
            Body::Uranus => self.series6(Series::Uranus, jd),
            Body::Neptune => self.series6(Series::Neptune, jd),
            Body::Pluto => self.series6(Series::Pluto, jd),
            Body::Sun => self.series6(Series::Sun, jd),
            Body::Emb => self.series6(Series::EarthMoonBarycenter, jd),
            Body::Earth => {
                let emb = self.series6(Series::EarthMoonBarycenter, jd)?;
                let moon = self.series6(Series::Moon, jd)?;
                let factor = 1.0 + self.emrat;
                let mut earth = [0.0; 6];
                for ((e, &b), &m) in earth.iter_mut().zip(&emb).zip(&moon) {
                    *e = b - m / factor;
                }
                Ok(earth)
            }
            Body::Moon => {
                let earth = self.state_km(Body::Earth, jd)?;
                let moon_geo = self.series6(Series::Moon, jd)?;
                let mut moon = [0.0; 6];
                for ((s, &e), &m) in moon.iter_mut().zip(&earth).zip(&moon_geo) {
                    *s = e + m;
                }
                Ok(moon)
            }
        }
    }

    /// SSB-centered state of a `testpo` body number in AU and AU/day
    /// (every component divided by the header `AU`, as in `STATE` with
    /// `KM = .FALSE.`).
    fn testpo_state_au(&self, number: u32, jd: (f64, f64)) -> Result<[f64; 6], DeError> {
        let mut state = self.state_km(testpo_body(number)?, jd)?;
        for c in &mut state {
            *c /= self.au_km;
        }
        Ok(state)
    }

    /// One `testpo` test-case value: coordinate `coord` (1..=6) of the state
    /// of `target` relative to `center`, in the units of the JPL `testpo`
    /// files (`PLEPH` of `testeph.f` with `KM = .FALSE.`).
    ///
    /// Numbering: 1..=9 Mercury..Pluto, 10 Moon, 11 Sun, 12 SSB, 13 EMB
    /// (AU, AU/day); 14 nutation (coord 1..=2 angles in radians, 3..=4 rates
    /// in radians/day, 5..=6 zero, `center` ignored); 15 librations and
    /// 16 lunar Euler-angle rates (radians, radians/day); 17 TT-TDB
    /// (coord 1 seconds, coord 2 seconds/day).
    ///
    /// The Earth/Moon pair (3, 10) is served directly from the geocentric
    /// Moon series to avoid the barycentric round trip, as `PLEPH` does.
    ///
    /// # Errors
    /// * [`DeError::BadArgument`] — unknown target/coordinate combination.
    /// * Otherwise the same conditions as [`Self::series_state`].
    pub fn testpo_value(
        &self,
        target: u32,
        center: u32,
        coord: u32,
        jd: (f64, f64),
    ) -> Result<f64, DeError> {
        match target {
            14 => {
                let s = self.series_state(Series::Nutation, jd)?;
                match coord {
                    1 => Ok(s.value[0]),
                    2 => Ok(s.value[1]),
                    3 => Ok(s.rate[0]),
                    4 => Ok(s.rate[1]),
                    // PLEPH sets RRD(5) = RRD(6) = 0 for nutation calls.
                    5 | 6 => Ok(0.0),
                    _ => Err(DeError::BadArgument),
                }
            }
            15 => component6(self.series6(Series::Libration, jd)?, coord),
            16 => component6(self.series6(Series::LunarEulerRates, jd)?, coord),
            17 => {
                let s = self.series_state(Series::TtTdb, jd)?;
                match coord {
                    1 => Ok(s.value[0]),
                    2 => Ok(s.rate[0]),
                    _ => Err(DeError::BadArgument),
                }
            }
            1..=13 => {
                if !(1..=6).contains(&coord) {
                    return Err(DeError::BadArgument);
                }
                if target == center {
                    return Ok(0.0);
                }
                // Earth/Moon special pair: NTARG*NCENT == 30 && sum == 13.
                if (target == 10 && center == 3) || (target == 3 && center == 10) {
                    let mut moon_geo = self.series6(Series::Moon, jd)?;
                    for c in &mut moon_geo {
                        *c /= self.au_km;
                    }
                    let sign = if target == 10 { 1.0 } else { -1.0 };
                    return Ok(sign * component6(moon_geo, coord)?);
                }
                let t_state = self.testpo_state_au(target, jd)?;
                let c_state = self.testpo_state_au(center, jd)?;
                Ok(component6(t_state, coord)? - component6(c_state, coord)?)
            }
            _ => Err(DeError::BadArgument),
        }
    }
}
