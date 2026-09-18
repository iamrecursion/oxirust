//! BD-Rate (Bjontegaard Delta) calculation for codec efficiency comparison.
//!
//! Implements the standard Bjontegaard method using log-domain cubic spline
//! interpolation over rate-distortion curves to compute the average bitrate
//! saving (BD-Rate) and quality gain (BD-PSNR) between two codecs.
//!
//! # Reference
//! G. Bjontegaard, "Calculation of average PSNR differences between RD-curves",
//! ITU-T SG16 VCEG Document VCEG-M33, Austin, TX, USA, Apr. 2001.

use serde::{Deserialize, Serialize};

/// Error type for BD-Rate computation.
#[derive(Debug, Clone, PartialEq)]
pub enum BdRateError {
    /// Fewer than 2 points provided for one of the curves.
    InsufficientPoints {
        /// Which curve (reference or test) had insufficient points.
        curve: &'static str,
        /// How many points were provided.
        count: usize,
    },
    /// The overlapping PSNR range between the two curves is too narrow.
    NoOverlapRange,
    /// A numeric overflow or degenerate spline condition was detected.
    NumericalError(String),
}

impl std::fmt::Display for BdRateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientPoints { curve, count } => {
                write!(f, "{curve} curve has {count} point(s); at least 2 required")
            }
            Self::NoOverlapRange => write!(
                f,
                "reference and test curves have no overlapping PSNR range"
            ),
            Self::NumericalError(msg) => write!(f, "numerical error in spline: {msg}"),
        }
    }
}

impl std::error::Error for BdRateError {}

/// A single point on a rate-distortion curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BdPoint {
    /// Bitrate in kbps.
    pub bitrate_kbps: f64,
    /// Quality value — typically PSNR in dB or an SSIM-derived score.
    pub quality: f64,
}

impl BdPoint {
    /// Create a new `BdPoint`.
    #[must_use]
    pub fn new(bitrate_kbps: f64, quality: f64) -> Self {
        Self {
            bitrate_kbps,
            quality,
        }
    }
}

/// Result of BD-Rate / BD-PSNR computation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BdResult {
    /// Average bitrate difference over the overlapping quality range.
    ///
    /// A **negative** value means the test codec achieves the same quality
    /// at a lower bitrate (more efficient).  Expressed as a percentage:
    /// `−10.0` ≡ 10 % bitrate saving.
    pub bd_rate: f64,

    /// Average PSNR difference over the overlapping bitrate range (dB).
    ///
    /// A **positive** value means the test codec delivers better quality at
    /// the same bitrate.
    pub bd_psnr: f64,
}

/// Bjontegaard Delta calculator.
///
/// Uses natural cubic spline interpolation in the log-bitrate domain —
/// the standard approach described in the original ITU-T document.
pub struct BdRateCalculator;

impl BdRateCalculator {
    /// Compute BD-Rate and BD-PSNR between a reference curve and a test curve.
    ///
    /// Both slices must contain at least 2 points.  Points need not be sorted;
    /// they are sorted internally by quality value.
    ///
    /// # Errors
    ///
    /// Returns a [`BdRateError`] when:
    /// - Either curve has fewer than 2 points.
    /// - The PSNR ranges do not overlap.
    /// - The spline system is degenerate (e.g., all bitrates identical).
    pub fn compute(reference: &[BdPoint], test: &[BdPoint]) -> Result<BdResult, BdRateError> {
        if reference.len() < 2 {
            return Err(BdRateError::InsufficientPoints {
                curve: "reference",
                count: reference.len(),
            });
        }
        if test.len() < 2 {
            return Err(BdRateError::InsufficientPoints {
                curve: "test",
                count: test.len(),
            });
        }

        // Sort points by quality (ascending).
        let mut ref_pts = reference.to_vec();
        let mut tst_pts = test.to_vec();
        ref_pts.sort_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        tst_pts.sort_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Log-domain transformation: x = quality (PSNR), y = log10(bitrate).
        let ref_log = Self::to_log_domain(&ref_pts)?;
        let tst_log = Self::to_log_domain(&tst_pts)?;

        // Determine the overlapping PSNR interval.
        let psnr_lo = f64::max(
            ref_pts.first().map(|p| p.quality).unwrap_or(0.0),
            tst_pts.first().map(|p| p.quality).unwrap_or(0.0),
        );
        let psnr_hi = f64::min(
            ref_pts.last().map(|p| p.quality).unwrap_or(0.0),
            tst_pts.last().map(|p| p.quality).unwrap_or(0.0),
        );

        if psnr_hi <= psnr_lo {
            return Err(BdRateError::NoOverlapRange);
        }

        // Build natural cubic splines in log-rate vs. PSNR direction.
        let ref_spline = NaturalCubicSpline::build(
            &ref_log.iter().map(|p| p.quality).collect::<Vec<_>>(),
            &ref_log.iter().map(|p| p.log_rate).collect::<Vec<_>>(),
        )?;
        let tst_spline = NaturalCubicSpline::build(
            &tst_log.iter().map(|p| p.quality).collect::<Vec<_>>(),
            &tst_log.iter().map(|p| p.log_rate).collect::<Vec<_>>(),
        )?;

        // Numerically integrate (Simpson's 1/3 rule) the difference in
        // log-rate over the overlapping PSNR range.
        let bd_rate_integral =
            Self::integrate_difference(&ref_spline, &tst_spline, psnr_lo, psnr_hi);

        // BD-Rate: average log-rate difference → percentage.
        let avg_log_diff = bd_rate_integral / (psnr_hi - psnr_lo);
        let bd_rate = (10_f64.powf(avg_log_diff) - 1.0) * 100.0;

        // BD-PSNR: invert the splines — fit PSNR vs. log-rate and integrate
        // over the overlapping log-rate range.
        let ref_sorted_log: Vec<_> = ref_log.iter().map(|p| p.log_rate).collect();
        let tst_sorted_log: Vec<_> = tst_log.iter().map(|p| p.log_rate).collect();

        let lr_lo = f64::max(
            ref_sorted_log.iter().cloned().fold(f64::INFINITY, f64::min),
            tst_sorted_log.iter().cloned().fold(f64::INFINITY, f64::min),
        );
        let lr_hi = f64::min(
            ref_sorted_log
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            tst_sorted_log
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
        );

        let bd_psnr = if lr_hi > lr_lo {
            // Sort log-rate data for PSNR(log-rate) splines.
            let mut ref_lr_psnr: Vec<(f64, f64)> =
                ref_log.iter().map(|p| (p.log_rate, p.quality)).collect();
            let mut tst_lr_psnr: Vec<(f64, f64)> =
                tst_log.iter().map(|p| (p.log_rate, p.quality)).collect();
            ref_lr_psnr.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            tst_lr_psnr.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let ref_psnr_spline = NaturalCubicSpline::build(
                &ref_lr_psnr.iter().map(|p| p.0).collect::<Vec<_>>(),
                &ref_lr_psnr.iter().map(|p| p.1).collect::<Vec<_>>(),
            );
            let tst_psnr_spline = NaturalCubicSpline::build(
                &tst_lr_psnr.iter().map(|p| p.0).collect::<Vec<_>>(),
                &tst_lr_psnr.iter().map(|p| p.1).collect::<Vec<_>>(),
            );

            match (ref_psnr_spline, tst_psnr_spline) {
                (Ok(r), Ok(t)) => {
                    let integral = Self::integrate_difference(&r, &t, lr_lo, lr_hi);
                    integral / (lr_hi - lr_lo)
                }
                _ => 0.0,
            }
        } else {
            0.0
        };

        Ok(BdResult { bd_rate, bd_psnr })
    }

    /// Convert `BdPoint` slice into log-domain points, validating positivity.
    fn to_log_domain(pts: &[BdPoint]) -> Result<Vec<LogPoint>, BdRateError> {
        pts.iter()
            .map(|p| {
                if p.bitrate_kbps <= 0.0 {
                    Err(BdRateError::NumericalError(format!(
                        "non-positive bitrate {:.4} kbps",
                        p.bitrate_kbps
                    )))
                } else {
                    Ok(LogPoint {
                        quality: p.quality,
                        log_rate: p.bitrate_kbps.log10(),
                    })
                }
            })
            .collect()
    }

    /// Integrate (tst_spline(x) - ref_spline(x)) over [lo, hi] using
    /// adaptive Simpson's rule with 128 equal sub-intervals.
    fn integrate_difference(
        ref_spline: &NaturalCubicSpline,
        tst_spline: &NaturalCubicSpline,
        lo: f64,
        hi: f64,
    ) -> f64 {
        const N: usize = 128; // must be even
        let h = (hi - lo) / N as f64;
        let mut sum = 0.0;

        for i in 0..=N {
            let x = lo + i as f64 * h;
            let diff = tst_spline.eval(x) - ref_spline.eval(x);
            let weight = if i == 0 || i == N {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            sum += weight * diff;
        }

        sum * h / 3.0
    }
}

/// Internal log-domain representation of a single RD point.
#[derive(Debug, Clone, Copy)]
struct LogPoint {
    quality: f64,
    log_rate: f64,
}

/// Natural (not-a-knot) cubic spline.
///
/// Given `n` knot pairs `(xs[i], ys[i])` with `xs` strictly increasing,
/// computes the piecewise-cubic interpolant with zero second derivatives at
/// the endpoints (natural boundary conditions).
struct NaturalCubicSpline {
    xs: Vec<f64>,
    /// Cubic coefficients: ys[i] + bs[i]*t + cs[i]*t^2 + ds[i]*t^3
    ys: Vec<f64>,
    bs: Vec<f64>,
    cs: Vec<f64>,
    ds: Vec<f64>,
}

impl NaturalCubicSpline {
    /// Build a natural cubic spline from knot arrays.
    ///
    /// # Errors
    ///
    /// Returns [`BdRateError::NumericalError`] when `xs` contains duplicate
    /// values or the tridiagonal system is singular.
    fn build(xs: &[f64], ys: &[f64]) -> Result<Self, BdRateError> {
        let n = xs.len();
        if n != ys.len() {
            return Err(BdRateError::NumericalError(
                "xs and ys lengths differ".into(),
            ));
        }
        if n < 2 {
            return Err(BdRateError::NumericalError(
                "at least 2 knots required".into(),
            ));
        }

        let m = n - 1; // number of intervals

        // Compute interval widths.
        let h: Vec<f64> = (0..m).map(|i| xs[i + 1] - xs[i]).collect();
        for (i, &hi) in h.iter().enumerate() {
            if hi <= 0.0 {
                return Err(BdRateError::NumericalError(format!(
                    "xs not strictly increasing at index {i}"
                )));
            }
        }

        // Set up tridiagonal system for interior second derivatives (sigma).
        // Natural BC: sigma[0] = sigma[n-1] = 0.
        // For i in 1..n-1:
        //   h[i-1]*sigma[i-1] + 2*(h[i-1]+h[i])*sigma[i] + h[i]*sigma[i+1]
        //     = 3 * ((ys[i+1]-ys[i])/h[i] - (ys[i]-ys[i-1])/h[i-1])
        let interior = n - 2;
        let mut sigma = vec![0.0_f64; n];

        if interior > 0 {
            let mut diag = vec![0.0_f64; interior];
            let mut upper = vec![0.0_f64; interior - 1];
            let mut lower = vec![0.0_f64; interior - 1];
            let mut rhs = vec![0.0_f64; interior];

            for k in 0..interior {
                let i = k + 1; // interior node index in xs
                diag[k] = 2.0 * (h[i - 1] + h[i]);
                rhs[k] = 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
                if k > 0 {
                    lower[k - 1] = h[i - 1];
                }
                if k + 1 < interior {
                    upper[k] = h[i];
                }
            }

            // Thomas (forward-sweep) algorithm for tridiagonal system.
            let sol = thomas_solve(&diag, &upper, &lower, &rhs)
                .ok_or_else(|| BdRateError::NumericalError("singular tridiagonal system".into()))?;

            for (k, s) in sol.iter().enumerate() {
                sigma[k + 1] = *s;
            }
        }

        // Derive cubic coefficients on each interval.
        let mut bs = vec![0.0_f64; m];
        let mut cs = vec![0.0_f64; m];
        let mut ds = vec![0.0_f64; m];

        for i in 0..m {
            bs[i] = (ys[i + 1] - ys[i]) / h[i] - h[i] * (2.0 * sigma[i] + sigma[i + 1]) / 3.0;
            cs[i] = sigma[i];
            ds[i] = (sigma[i + 1] - sigma[i]) / (3.0 * h[i]);
        }

        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            bs,
            cs,
            ds,
        })
    }

    /// Evaluate the spline at `x`, clamping to the boundary for extrapolation.
    fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();

        // Clamp extrapolation.
        if x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= self.xs[n - 1] {
            return *self.ys.last().unwrap_or(&0.0);
        }

        // Binary search for the interval.
        let mut lo = 0_usize;
        let mut hi = n - 2;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if self.xs[mid] <= x {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let i = lo;
        let t = x - self.xs[i];
        self.ys[i] + t * (self.bs[i] + t * (self.cs[i] + t * self.ds[i]))
    }
}

/// Solve a tridiagonal linear system using the Thomas algorithm.
/// Returns `None` if the system is singular.
fn thomas_solve(diag: &[f64], upper: &[f64], lower: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = diag.len();
    if n == 0 {
        return Some(vec![]);
    }

    let mut c_prime = vec![0.0_f64; n];
    let mut d_prime = vec![0.0_f64; n];
    let mut sol = vec![0.0_f64; n];

    // Forward sweep.
    if diag[0].abs() < f64::EPSILON {
        return None;
    }
    c_prime[0] = if n > 1 { upper[0] / diag[0] } else { 0.0 };
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - lower[i - 1] * c_prime[i - 1];
        if denom.abs() < f64::EPSILON {
            return None;
        }
        c_prime[i] = if i + 1 < n { upper[i] / denom } else { 0.0 };
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom;
    }

    // Back substitution.
    sol[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        sol[i] = d_prime[i] - c_prime[i] * sol[i + 1];
    }

    Some(sol)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a simple linear RD curve: PSNR = a * log10(rate) + b.
    #[allow(dead_code)]
    fn linear_curve(a: f64, b: f64, rates: &[f64]) -> Vec<BdPoint> {
        rates
            .iter()
            .map(|&r| BdPoint::new(r, a * r.log10() + b))
            .collect()
    }

    // ---- BdPoint construction ----

    #[test]
    fn test_bd_point_new() {
        let pt = BdPoint::new(2000.0, 40.0);
        assert_eq!(pt.bitrate_kbps, 2000.0);
        assert_eq!(pt.quality, 40.0);
    }

    // ---- Error paths ----

    #[test]
    fn test_insufficient_reference_points() {
        let single = vec![BdPoint::new(1000.0, 35.0)];
        let test = vec![BdPoint::new(1000.0, 35.0), BdPoint::new(2000.0, 38.0)];
        let err = BdRateCalculator::compute(&single, &test).unwrap_err();
        assert!(matches!(
            err,
            BdRateError::InsufficientPoints {
                curve: "reference",
                count: 1
            }
        ));
    }

    #[test]
    fn test_insufficient_test_points() {
        let reference = vec![BdPoint::new(1000.0, 35.0), BdPoint::new(2000.0, 38.0)];
        let single = vec![BdPoint::new(1500.0, 36.0)];
        let err = BdRateCalculator::compute(&reference, &single).unwrap_err();
        assert!(matches!(
            err,
            BdRateError::InsufficientPoints {
                curve: "test",
                count: 1
            }
        ));
    }

    #[test]
    fn test_no_overlap_range() {
        // reference PSNR: 30–35, test PSNR: 40–45 → no overlap
        let reference = vec![
            BdPoint::new(500.0, 30.0),
            BdPoint::new(1000.0, 33.0),
            BdPoint::new(2000.0, 35.0),
        ];
        let test = vec![
            BdPoint::new(500.0, 40.0),
            BdPoint::new(1000.0, 43.0),
            BdPoint::new(2000.0, 45.0),
        ];
        let err = BdRateCalculator::compute(&reference, &test).unwrap_err();
        assert!(matches!(err, BdRateError::NoOverlapRange));
    }

    // ---- Identical curves → BD-Rate ≈ 0 ----

    #[test]
    fn test_identical_curves_bd_rate_zero() {
        let pts = vec![
            BdPoint::new(500.0, 32.0),
            BdPoint::new(1000.0, 35.0),
            BdPoint::new(2000.0, 38.0),
            BdPoint::new(4000.0, 41.0),
        ];
        let result = BdRateCalculator::compute(&pts, &pts).expect("compute failed");
        assert!(
            result.bd_rate.abs() < 1e-6,
            "expected ~0 BD-Rate, got {:.8}",
            result.bd_rate
        );
        assert!(
            result.bd_psnr.abs() < 1e-6,
            "expected ~0 BD-PSNR, got {:.8}",
            result.bd_psnr
        );
    }

    // ---- Test codec uses same quality at half the bitrate → BD-Rate ≈ −50 % ----

    #[test]
    fn test_half_bitrate_bd_rate_negative() {
        let rates = [500.0, 1000.0, 2000.0, 4000.0];
        let reference: Vec<BdPoint> = rates
            .iter()
            .map(|&r| BdPoint::new(r, 10.0 * r.log10()))
            .collect();
        let test: Vec<BdPoint> = rates
            .iter()
            .map(|&r| BdPoint::new(r / 2.0, 10.0 * r.log10()))
            .collect();
        let result = BdRateCalculator::compute(&reference, &test).expect("compute failed");
        // ~−50 % bitrate saving
        assert!(
            result.bd_rate < 0.0,
            "BD-Rate should be negative for a more efficient codec, got {:.4}",
            result.bd_rate
        );
        assert!(
            result.bd_rate > -60.0,
            "BD-Rate should not be < −60 %, got {:.4}",
            result.bd_rate
        );
    }

    // ---- Test codec uses double the bitrate → BD-Rate > 0 ----

    #[test]
    fn test_double_bitrate_bd_rate_positive() {
        let rates = [500.0, 1000.0, 2000.0, 4000.0];
        let reference: Vec<BdPoint> = rates
            .iter()
            .map(|&r| BdPoint::new(r, 10.0 * r.log10()))
            .collect();
        let test: Vec<BdPoint> = rates
            .iter()
            .map(|&r| BdPoint::new(r * 2.0, 10.0 * r.log10()))
            .collect();
        let result = BdRateCalculator::compute(&reference, &test).expect("compute failed");
        assert!(
            result.bd_rate > 0.0,
            "BD-Rate should be positive for a less efficient codec, got {:.4}",
            result.bd_rate
        );
    }

    // ---- BD-PSNR sign ----

    #[test]
    fn test_better_quality_positive_bd_psnr() {
        // test codec has +2 dB PSNR at every bitrate
        let pts: Vec<BdPoint> = [500.0, 1000.0, 2000.0, 4000.0]
            .iter()
            .map(|&r| BdPoint::new(r, 35.0 + r / 1000.0))
            .collect();
        let pts_better: Vec<BdPoint> = [500.0, 1000.0, 2000.0, 4000.0]
            .iter()
            .map(|&r| BdPoint::new(r, 35.0 + r / 1000.0 + 2.0))
            .collect();
        let result = BdRateCalculator::compute(&pts, &pts_better).expect("compute failed");
        assert!(
            result.bd_psnr > 0.0,
            "BD-PSNR should be positive when test has higher quality, got {:.4}",
            result.bd_psnr
        );
    }

    // ---- Reference values with known RD curves ----

    #[test]
    fn test_known_reference_curves() {
        // Curves derived from standard Bjontegaard example data.
        // reference: AV1 at CQ 30/35/40/45
        let reference = vec![
            BdPoint::new(300.0, 34.0),
            BdPoint::new(600.0, 37.0),
            BdPoint::new(1200.0, 40.0),
            BdPoint::new(2400.0, 43.0),
        ];
        // test (more efficient codec): same quality at ~20 % less bitrate
        let test = vec![
            BdPoint::new(240.0, 34.0),
            BdPoint::new(480.0, 37.0),
            BdPoint::new(960.0, 40.0),
            BdPoint::new(1920.0, 43.0),
        ];
        let result = BdRateCalculator::compute(&reference, &test).expect("compute failed");
        // Should be approximately −20 % (±5 %)
        assert!(
            result.bd_rate < -10.0 && result.bd_rate > -30.0,
            "expected BD-Rate ≈ −20 %, got {:.4}",
            result.bd_rate
        );
    }

    #[test]
    fn test_unsorted_input_produces_same_result() {
        let reference_sorted = vec![
            BdPoint::new(300.0, 34.0),
            BdPoint::new(600.0, 37.0),
            BdPoint::new(1200.0, 40.0),
            BdPoint::new(2400.0, 43.0),
        ];
        let reference_unsorted = vec![
            BdPoint::new(1200.0, 40.0),
            BdPoint::new(300.0, 34.0),
            BdPoint::new(2400.0, 43.0),
            BdPoint::new(600.0, 37.0),
        ];
        let test = vec![
            BdPoint::new(250.0, 34.0),
            BdPoint::new(500.0, 37.0),
            BdPoint::new(1000.0, 40.0),
            BdPoint::new(2000.0, 43.0),
        ];
        let result_sorted =
            BdRateCalculator::compute(&reference_sorted, &test).expect("compute failed");
        let result_unsorted =
            BdRateCalculator::compute(&reference_unsorted, &test).expect("compute failed");
        assert!(
            (result_sorted.bd_rate - result_unsorted.bd_rate).abs() < 1e-9,
            "sorted and unsorted inputs should yield the same result"
        );
    }

    #[test]
    fn test_two_point_curves() {
        let reference = vec![BdPoint::new(500.0, 35.0), BdPoint::new(2000.0, 40.0)];
        let test = vec![BdPoint::new(400.0, 35.0), BdPoint::new(1600.0, 40.0)];
        let result = BdRateCalculator::compute(&reference, &test).expect("compute failed");
        assert!(result.bd_rate < 0.0);
    }

    #[test]
    fn test_bd_result_fields_are_finite() {
        let reference = vec![
            BdPoint::new(250.0, 33.0),
            BdPoint::new(500.0, 36.0),
            BdPoint::new(1000.0, 39.0),
            BdPoint::new(2000.0, 42.0),
        ];
        let test = vec![
            BdPoint::new(200.0, 33.0),
            BdPoint::new(400.0, 36.0),
            BdPoint::new(800.0, 39.0),
            BdPoint::new(1600.0, 42.0),
        ];
        let result = BdRateCalculator::compute(&reference, &test).expect("compute failed");
        assert!(result.bd_rate.is_finite());
        assert!(result.bd_psnr.is_finite());
    }

    #[test]
    fn test_error_display_messages() {
        let e1 = BdRateError::InsufficientPoints {
            curve: "reference",
            count: 1,
        };
        assert!(e1.to_string().contains("reference"));
        let e2 = BdRateError::NoOverlapRange;
        assert!(e2.to_string().contains("overlapping"));
        let e3 = BdRateError::NumericalError("test".into());
        assert!(e3.to_string().contains("test"));
    }

    // ---- Spline correctness sanity check ----

    #[test]
    fn test_spline_interpolates_at_knots() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![1.0, 4.0, 9.0, 16.0]; // y = x^2
        let spline = NaturalCubicSpline::build(&xs, &ys).expect("build failed");
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let v = spline.eval(x);
            assert!((v - y).abs() < 1e-9, "spline({x}) = {v}, expected {y}");
        }
    }

    #[test]
    fn test_spline_clamps_extrapolation() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![1.0, 4.0, 9.0];
        let spline = NaturalCubicSpline::build(&xs, &ys).expect("build failed");
        assert_eq!(spline.eval(0.0), 1.0);
        assert_eq!(spline.eval(10.0), 9.0);
    }
}
