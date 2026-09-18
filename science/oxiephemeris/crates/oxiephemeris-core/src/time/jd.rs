//! Two-part Julian Date arithmetic.
//!
//! # Why a two-part Julian Date is mandatory
//!
//! Contemporary Julian Dates are ≈ 2.46 × 10⁶ days, i.e. in the binade
//! `[2²¹, 2²²)`, where one ulp of an `f64` is 2⁻³¹ day ≈ 4.7 × 10⁻¹⁰ day
//! ≈ 40 µs. One microsecond is 1.157 × 10⁻¹¹ day, so a single `f64` JD
//! cannot even represent time to the microsecond — far too coarse for
//! microarcsecond-grade work (Earth's orbital motion is ≈ 0.04 mas per
//! millisecond of time). Splitting the JD into an exact large part (`hi`)
//! plus a small remainder (`lo`) restores full double precision in the
//! remainder: with `hi` on a half-day boundary, `lo < 1` carries ulp
//! ≈ 2⁻⁵³ day ≈ 10 ps.
//!
//! # Convention
//!
//! `hi + lo` is the Julian Date. `hi` carries the large part (typically the
//! integer-plus-half day boundary), `lo` the small remainder (typically the
//! fraction of a day). The split is *not* unique: any `(hi, lo)` pair with
//! the same exact sum denotes the same instant. [`JulianDate::new`]
//! canonicalizes a pair with an error-free two-sum so that
//! `hi = fl(hi + lo)` and `lo` is the exact rounding remainder;
//! [`crate::time::calendar::julday`] instead intentionally returns the
//! day-boundary split described in its documentation.
//!
//! # References
//!
//! - Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed., §4.2.2,
//!   Theorem B (error-free transformation of a floating-point sum).
//! - Dekker (1971), *A floating-point technique for extending the available
//!   precision*, Numer. Math. 18, 224–242.

/// Julian Date of the standard epoch J2000.0 (2000 January 1.5 TT).
///
/// Value from IAU 1994 Resolution C7 (also USNO Circular 179, §2.3).
pub const J2000_JD: f64 = 2_451_545.0;

/// Seconds per day (exact, by definition of the SI-based day).
pub const SECONDS_PER_DAY: f64 = 86_400.0;

/// Error-free transformation of a sum: returns `(s, e)` with
/// `s = fl(a + b)` and `s + e == a + b` exactly.
///
/// Knuth's `TwoSum` (TAOCP vol. 2, §4.2.2, Theorem B); branch-free variant,
/// valid for any pair of finite doubles.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// Two-part Julian Date for extended precision.
///
/// `hi + lo` is the Julian Date; `hi` carries the large part, `lo` the
/// small remainder (see the module documentation for the convention and
/// the rationale).
///
/// Note: the derived `PartialEq` compares the *representation* `(hi, lo)`,
/// not the instant; two different splits of the same instant compare
/// unequal. Use [`JulianDate::diff_days`] to compare instants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JulianDate {
    /// High-order part (typically the epoch/day part).
    pub hi: f64,
    /// Low-order part (typically the fraction of day).
    pub lo: f64,
}

impl JulianDate {
    /// Construct from a `(hi, lo)` pair, canonicalizing with an error-free
    /// two-sum: the result satisfies `hi = fl(hi_in + lo_in)` and
    /// `hi + lo == hi_in + lo_in` exactly.
    #[must_use]
    pub fn new(hi: f64, lo: f64) -> Self {
        let (s, e) = two_sum(hi, lo);
        Self { hi: s, lo: e }
    }

    /// Construct from a single `f64` JD (reduced precision: ≈ 40 µs
    /// granularity for contemporary dates; see the module documentation).
    #[must_use]
    pub fn from_f64(jd: f64) -> Self {
        Self { hi: jd, lo: 0.0 }
    }

    /// Total value as a single `f64` (loses the two-part precision
    /// advantage).
    #[must_use]
    pub fn value(self) -> f64 {
        self.hi + self.lo
    }

    /// Add a number of days, preserving two-part precision.
    #[must_use]
    pub fn add_days(self, days: f64) -> Self {
        let (s, e) = two_sum(self.hi, days);
        Self::new(s, e + self.lo)
    }

    /// Add a number of seconds (converted to days with one rounding),
    /// preserving two-part precision.
    #[must_use]
    pub fn add_seconds(self, seconds: f64) -> Self {
        self.add_days(seconds / SECONDS_PER_DAY)
    }

    /// Difference `self − other` in days, evaluated so that the large parts
    /// cancel before the small parts are added (full precision for nearby
    /// epochs).
    #[must_use]
    pub fn diff_days(self, other: Self) -> f64 {
        (self.hi - other.hi) + (self.lo - other.lo)
    }
}

#[cfg(test)]
mod tests {
    use super::{two_sum, JulianDate, J2000_JD, SECONDS_PER_DAY};

    #[test]
    fn two_sum_is_error_free_for_representative_pairs() {
        // (a, b) chosen so that a + b rounds: b is far below one ulp of a
        // (ulp(J2000_JD) = 2⁻³¹ day), so fl(a + b) == a and the error term
        // must recover b exactly.
        let a = J2000_JD;
        let b = 1e-12;
        let (s, e) = two_sum(a, b);
        assert!(s.to_bits() == a.to_bits());
        assert!(e.to_bits() == b.to_bits());
        // Re-splitting an already canonical pair is the identity.
        let (s2, e2) = two_sum(s, e);
        assert!(s2.to_bits() == s.to_bits());
        assert!(e2.to_bits() == e.to_bits());
    }

    #[test]
    fn new_normalizes() {
        let jd = JulianDate::new(J2000_JD, 0.75);
        assert!(jd.hi.to_bits() == (J2000_JD + 0.75).to_bits());
        assert!(jd.lo.to_bits() == 0.0f64.to_bits());
    }

    #[test]
    fn add_seconds_keeps_microsecond_precision() {
        let t0 = JulianDate::from_f64(J2000_JD);
        let t1 = t0.add_seconds(1e-6);
        let dt = t1.diff_days(t0) * SECONDS_PER_DAY;
        // 1 µs survives the round trip to well below a picosecond.
        assert!((dt - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn add_days_accumulates_exactly_for_half_days() {
        let mut t = JulianDate::from_f64(0.0);
        for _ in 0..4 {
            t = t.add_days(0.5);
        }
        assert!((t.value() - 2.0).abs() < 1e-15);
    }
}
