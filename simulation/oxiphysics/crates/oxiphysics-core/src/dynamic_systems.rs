// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Dynamical systems — maps, flows, bifurcations, synchronisation.
//!
//! Covers discrete maps (logistic, Hénon, Chirikov), strange attractor analysis,
//! normal forms for bifurcations, Poincaré sections, flow maps and the
//! fundamental solution matrix, and Pecora–Carroll / phase synchronisation.

// ─────────────────────────────────────────────────────────────────────────────
// DiscreteMap
// ─────────────────────────────────────────────────────────────────────────────

/// A discrete map that can be iterated.
pub trait DiscreteMapIterate {
    /// State dimension.
    fn dim(&self) -> usize;
    /// Advance the state by one step.  `state` must have length `dim()`.
    fn step(&self, state: &[f64]) -> Vec<f64>;

    /// Compute an orbit of length `n` starting from `init`.
    fn orbit(&self, init: &[f64], n: usize) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(n + 1);
        out.push(init.to_vec());
        for _ in 0..n {
            let next = self.step(out.last().expect("orbit is non-empty"));
            out.push(next);
        }
        out
    }
}

// ── Logistic Map ─────────────────────────────────────────────────────────────

/// The logistic map  `x_{n+1} = r · xₙ (1 − xₙ)`.
///
/// Classic example of period-doubling route to chaos with `r ∈ [0, 4]`.
#[derive(Debug, Clone)]
pub struct LogisticMap {
    /// Growth parameter `r`.
    pub r: f64,
}

impl LogisticMap {
    /// Create a logistic map with growth parameter `r`.
    pub fn new(r: f64) -> Self {
        Self { r }
    }

    /// One application of the map.
    pub fn apply(&self, x: f64) -> f64 {
        self.r * x * (1.0 - x)
    }

    /// Compute an orbit starting at `x0` for `n` steps.
    pub fn orbit_1d(&self, x0: f64, n: usize) -> Vec<f64> {
        let mut orbit = Vec::with_capacity(n + 1);
        orbit.push(x0);
        for _ in 0..n {
            let last = *orbit.last().expect("orbit is non-empty");
            orbit.push(self.apply(last));
        }
        orbit
    }

    /// Generate a bifurcation diagram by sweeping `r` in `[r_min, r_max]`
    /// with `n_r` steps.  For each `r` value the system is iterated
    /// `transient` steps (discarded) then `keep` steps are recorded.
    ///
    /// Returns a list of `(r, x)` pairs.
    pub fn bifurcation_diagram(
        r_min: f64,
        r_max: f64,
        n_r: usize,
        transient: usize,
        keep: usize,
    ) -> Vec<(f64, f64)> {
        let mut result = Vec::with_capacity(n_r * keep);
        for k in 0..n_r {
            let r = r_min + (r_max - r_min) * k as f64 / n_r.saturating_sub(1).max(1) as f64;
            let map = LogisticMap::new(r);
            let mut x = 0.5;
            for _ in 0..transient {
                x = map.apply(x);
            }
            for _ in 0..keep {
                x = map.apply(x);
                result.push((r, x));
            }
        }
        result
    }

    /// Largest Lyapunov exponent estimated over `n` iterations.
    ///
    /// Returns `None` if the trajectory passes through 0 or 1 (derivative undefined).
    pub fn lyapunov_exponent(&self, x0: f64, n: usize) -> Option<f64> {
        let mut x = x0;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let deriv = (self.r * (1.0 - 2.0 * x)).abs();
            if deriv < 1e-300 {
                return None;
            }
            sum += deriv.ln();
            x = self.apply(x);
        }
        Some(sum / n as f64)
    }
}

impl DiscreteMapIterate for LogisticMap {
    fn dim(&self) -> usize {
        1
    }
    fn step(&self, state: &[f64]) -> Vec<f64> {
        vec![self.apply(state[0])]
    }
}

// ── Hénon Map ─────────────────────────────────────────────────────────────────

/// The Hénon map:
///
/// ```text
/// x_{n+1} = 1 − a xₙ² + yₙ
/// y_{n+1} = b xₙ
/// ```
///
/// Classical parameter values are `a = 1.4`, `b = 0.3`.
#[derive(Debug, Clone)]
pub struct HenonMap {
    /// Parameter `a`.
    pub a: f64,
    /// Parameter `b`.
    pub b: f64,
}

impl HenonMap {
    /// Create a Hénon map with given parameters.
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    /// Apply one step starting at `(x, y)`.  Returns `(x_new, y_new)`.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (1.0 - self.a * x * x + y, self.b * x)
    }

    /// Compute an orbit for `n` steps starting at `(x0, y0)`.
    pub fn orbit_2d(&self, x0: f64, y0: f64, n: usize) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(n + 1);
        out.push((x0, y0));
        for _ in 0..n {
            let &(x, y) = out.last().expect("orbit is non-empty");
            out.push(self.apply(x, y));
        }
        out
    }

    /// Jacobian determinant (constant for the Hénon map): `−b`.
    pub fn jacobian_det(&self) -> f64 {
        -self.b
    }

    /// Jacobian matrix at `(x, y)`:
    /// ```text
    /// [[-2ax,  1],
    ///  [b,     0]]
    /// ```
    pub fn jacobian(&self, x: f64, _y: f64) -> [[f64; 2]; 2] {
        [[-2.0 * self.a * x, 1.0], [self.b, 0.0]]
    }
}

impl DiscreteMapIterate for HenonMap {
    fn dim(&self) -> usize {
        2
    }
    fn step(&self, state: &[f64]) -> Vec<f64> {
        let (xn, yn) = self.apply(state[0], state[1]);
        vec![xn, yn]
    }
}

// ── Chirikov Standard Map ─────────────────────────────────────────────────────

/// The Chirikov–Taylor standard map (area-preserving):
///
/// ```text
/// p_{n+1} = p_n + K sin(θ_n)   (mod 2π)
/// θ_{n+1} = θ_n + p_{n+1}      (mod 2π)
/// ```
///
/// `K` is the stochasticity parameter; chaos sets in for `K ≳ 1`.
#[derive(Debug, Clone)]
pub struct ChirikovStandardMap {
    /// Stochasticity parameter `K`.
    pub k: f64,
}

impl ChirikovStandardMap {
    /// Create a Chirikov standard map with parameter `K`.
    pub fn new(k: f64) -> Self {
        Self { k }
    }

    /// Apply one step.  Both `theta` and `p` are taken modulo `2π`.
    /// Returns `(theta_new, p_new)`.
    pub fn apply(&self, theta: f64, p: f64) -> (f64, f64) {
        let p_new = (p + self.k * theta.sin()).rem_euclid(std::f64::consts::TAU);
        let theta_new = (theta + p_new).rem_euclid(std::f64::consts::TAU);
        (theta_new, p_new)
    }

    /// Compute an orbit for `n` steps.
    pub fn orbit_2d(&self, theta0: f64, p0: f64, n: usize) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(n + 1);
        out.push((theta0, p0));
        for _ in 0..n {
            let &(th, p) = out.last().expect("orbit is non-empty");
            out.push(self.apply(th, p));
        }
        out
    }

    /// Estimate the Lyapunov exponent via the method of Benettin et al.
    pub fn lyapunov_exponent(&self, theta0: f64, p0: f64, n: usize) -> f64 {
        let mut theta = theta0;
        let mut p = p0;
        // tangent vector
        let mut dt = 1.0f64;
        let mut dp = 0.0f64;
        let mut sum = 0.0f64;

        for _ in 0..n {
            let cos_th = theta.cos();
            // Jacobian: [[1, 0], [K cos θ, 1]] * tangent
            let dp_new = dp + self.k * cos_th * dt;
            let dt_new = dt + dp_new;
            let norm = (dt_new * dt_new + dp_new * dp_new).sqrt().max(1e-300);
            sum += norm.ln();
            dt = dt_new / norm;
            dp = dp_new / norm;
            let (th_new, p_new) = self.apply(theta, p);
            theta = th_new;
            p = p_new;
        }
        sum / n as f64
    }
}

impl DiscreteMapIterate for ChirikovStandardMap {
    fn dim(&self) -> usize {
        2
    }
    fn step(&self, state: &[f64]) -> Vec<f64> {
        let (th, p) = self.apply(state[0], state[1]);
        vec![th, p]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AttractorAnalysis
// ─────────────────────────────────────────────────────────────────────────────

/// Tools for characterising attractors from time-series data.
#[derive(Debug, Clone)]
pub struct AttractorAnalysis {
    /// The embedded time-series (each inner `Vec` is one point in phase space).
    pub data: Vec<Vec<f64>>,
}

impl AttractorAnalysis {
    /// Create from a time-series.
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        Self { data }
    }

    /// Find approximate fixed points: points where `|f(x) − x| < tol`.
    ///
    /// `f` is the map applied once.
    pub fn find_fixed_points<F>(&self, f: F, tol: f64) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        self.data
            .iter()
            .filter(|x| {
                let fx = f(x.as_slice());
                let dist: f64 = x
                    .iter()
                    .zip(fx.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                dist < tol
            })
            .cloned()
            .collect()
    }

    /// Detect periodic orbits of period `p`: points where `|f^p(x) − x| < tol`.
    pub fn find_periodic_orbits<F>(&self, f: &F, period: usize, tol: f64) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        self.data
            .iter()
            .filter(|x| {
                let mut state = x.to_vec();
                for _ in 0..period {
                    state = f(&state);
                }
                let dist: f64 = x
                    .iter()
                    .zip(state.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                dist < tol
            })
            .cloned()
            .collect()
    }

    /// Estimate the correlation dimension via the Grassberger–Procaccia algorithm.
    ///
    /// Computes the correlation sum `C(r)` — the fraction of point pairs at
    /// distance `≤ r` — for `n_radii` radii spaced logarithmically between
    /// `1.1 · d_min` and `0.9 · d_max` (`d_min` is the smallest *positive* and
    /// `d_max` the largest pairwise distance; coincident points carry no scale
    /// information) and returns the `(ln r, ln C(r))` pairs with `C(r) > 0`,
    /// from which the slope gives the correlation dimension.
    ///
    /// The distances are never stored: one pass finds `d_min` and `d_max`, a
    /// second pass bins every squared distance into the squared radii by
    /// binary search, and a cumulative sum yields `C(r_k)`.  Memory use is
    /// `O(n_radii)`, time `O(n²)`, and no square root or sort is needed per
    /// pair.  Pairs with a NaN distance are ignored.  Returns an empty vector
    /// for fewer than two points, for `n_radii == 0`, when all points
    /// coincide, or when `1.1 · d_min >= 0.9 · d_max` leaves no range of radii.
    pub fn correlation_dimension(&self, n_radii: usize) -> Vec<(f64, f64)> {
        self.correlation_dimension_theiler(n_radii, 0)
    }

    /// [`Self::correlation_dimension`] for time-series data with a Theiler
    /// window: pairs `(i, j)` with `|i − j| < theiler_window` are skipped, so
    /// that temporally adjacent samples of one orbit segment do not inflate
    /// `C(r)` at small radii (Theiler 1986).  `0` and `1` both mean "only
    /// exclude `i = j`", i.e. every distinct pair is used.
    pub fn correlation_dimension_theiler(
        &self,
        n_radii: usize,
        theiler_window: usize,
    ) -> Vec<(f64, f64)> {
        let n = self.data.len();
        let min_sep = theiler_window.max(1);
        if n_radii == 0 || n <= min_sep {
            return vec![];
        }

        // Pass 1: extreme pairwise distances (squared).
        let mut d2_min = f64::INFINITY;
        let mut d2_max = 0.0f64;
        for (i, point) in self.data.iter().enumerate() {
            for other in &self.data[(i + min_sep).min(n)..] {
                let d2 = squared_distance(point, other);
                if d2 > 0.0 && d2 < d2_min {
                    d2_min = d2;
                }
                if d2 > d2_max {
                    d2_max = d2;
                }
            }
        }
        if !d2_min.is_finite() {
            return vec![];
        }
        let r_min = d2_min.sqrt() * 1.1;
        let r_max = d2_max.sqrt() * 0.9;
        if r_min >= r_max {
            return vec![];
        }

        let denom = n_radii.saturating_sub(1).max(1) as f64;
        let radii: Vec<f64> = (0..n_radii)
            .map(|k| r_min * (r_max / r_min).powf(k as f64 / denom))
            .collect();
        let thresholds: Vec<f64> = radii.iter().map(|r| r * r).collect();

        // Pass 2: bin each pair at the first radius that contains it.
        let mut bins = vec![0u64; n_radii];
        let mut total_pairs = 0u64;
        for (i, point) in self.data.iter().enumerate() {
            for other in &self.data[(i + min_sep).min(n)..] {
                let d2 = squared_distance(point, other);
                if d2.is_nan() {
                    continue;
                }
                total_pairs += 1;
                let first_containing = thresholds.partition_point(|&t| t < d2);
                if let Some(bin) = bins.get_mut(first_containing) {
                    *bin += 1;
                }
            }
        }

        let mut result = Vec::with_capacity(n_radii);
        let mut within = 0u64;
        for (&r, &bin) in radii.iter().zip(bins.iter()) {
            within += bin;
            if within > 0 {
                result.push((r.ln(), (within as f64 / total_pairs as f64).ln()));
            }
        }
        result
    }

    /// Estimate the largest Lyapunov exponent from the data via the Rosenstein method.
    ///
    /// For every point the nearest neighbour *outside the Theiler window* is
    /// found (`|i − j| >= theiler_window`, compared by squared distance), the
    /// distance of the pair is followed for `horizon` samples, and the mean of
    /// `ln d_i(t)` over all points is returned as a function of `t`.  The
    /// slope of the initial linear part of this curve, divided by the sampling
    /// interval, estimates the largest Lyapunov exponent.  The logarithm is
    /// taken only of the distances of the chosen pairs.
    ///
    /// Rosenstein et al. (1993) require the neighbour to be temporally
    /// separated from the point by more than the mean period of the orbit.
    /// On a sampled flow the nearest neighbour of a point is otherwise its own
    /// predecessor or successor on the same orbit, and the curve then tracks
    /// the trajectory increments instead of the divergence of nearby orbits.
    /// [`Self::default_theiler_window`] gives the usual choice and
    /// [`Self::rosenstein_lyapunov_auto`] applies it; `0` and `1` both mean
    /// "only exclude the point itself".
    ///
    /// The curve has `min(horizon, n / 2)` entries.  Only points that can be
    /// followed for the full horizon, together with neighbours that can be,
    /// take part, so that every entry averages over the same set of pairs.  It
    /// is empty when no point has an admissible neighbour; an entry is `NaN`
    /// if every pair had zero distance at that time.  The neighbour search is
    /// brute force, `O(n²)` in the number of points.
    pub fn rosenstein_lyapunov(&self, horizon: usize, theiler_window: usize) -> Vec<f64> {
        let n = self.data.len();
        if n < 2 || horizon == 0 {
            return vec![];
        }
        let steps = horizon.min(n / 2);
        let min_sep = theiler_window.max(1);
        // Index range of points that can be followed for `steps` samples.
        let followable = n - steps + 1;
        let mut sum_log = vec![0.0f64; steps];
        let mut counts = vec![0usize; steps];
        let mut pairs = 0usize;

        for (i, point) in self.data.iter().enumerate().take(followable) {
            let mut nearest: Option<(usize, f64)> = None;
            for (j, other) in self.data.iter().enumerate().take(followable) {
                if i.abs_diff(j) < min_sep {
                    continue;
                }
                let d2 = squared_distance(point, other);
                let closer = match nearest {
                    Some((_, best)) => d2 < best,
                    None => !d2.is_nan(),
                };
                if closer {
                    nearest = Some((j, d2));
                }
            }
            let Some((nn_idx, _)) = nearest else {
                continue;
            };
            pairs += 1;
            for (t, (sum, count)) in sum_log.iter_mut().zip(counts.iter_mut()).enumerate() {
                let d = squared_distance(&self.data[i + t], &self.data[nn_idx + t]).sqrt();
                if d > 1e-300 {
                    *sum += d.ln();
                    *count += 1;
                }
            }
        }
        if pairs == 0 {
            return vec![];
        }

        sum_log
            .iter()
            .zip(counts.iter())
            .map(|(&s, &c)| if c > 0 { s / c as f64 } else { f64::NAN })
            .collect()
    }

    /// [`Self::rosenstein_lyapunov`] with the Theiler window of
    /// [`Self::default_theiler_window`].
    pub fn rosenstein_lyapunov_auto(&self, horizon: usize) -> Vec<f64> {
        self.rosenstein_lyapunov(horizon, self.default_theiler_window())
    }

    /// Default Theiler window (in samples) for [`Self::rosenstein_lyapunov`]:
    /// the first lag at which the autocorrelation of coordinate 0 of the data
    /// crosses zero (a quarter of the dominant period for an oscillation).
    ///
    /// If the autocorrelation does not cross zero at any lag below `n / 2`
    /// (a trend, or a constant signal) the series has no resolvable
    /// decorrelation time and `n / 2`, the longest lag that can be estimated,
    /// is returned.
    pub fn default_theiler_window(&self) -> usize {
        let n = self.data.len();
        let coordinate: Vec<f64> = self
            .data
            .iter()
            .map(|p| p.first().copied().unwrap_or(0.0))
            .collect();
        first_autocorrelation_zero_crossing(&coordinate).unwrap_or(n / 2)
    }

    /// Check whether the attractor looks strange by testing if the
    /// correlation dimension estimate is fractional (> 1.5).
    pub fn is_strange_attractor(&self) -> bool {
        let pairs = self.correlation_dimension(20);
        if pairs.len() < 4 {
            return false;
        }
        // Linear regression slope on the log-log pairs
        let slope = linear_slope(&pairs);
        slope > 1.5 && slope < 4.0
    }
}

/// Squared Euclidean distance between two points (extra coordinates of the
/// longer one are ignored).
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// First lag `k >= 1` at which the autocorrelation of `x` is not positive, or
/// `None` if it stays positive for all `k < n / 2` (or `x` is constant).
fn first_autocorrelation_zero_crossing(x: &[f64]) -> Option<usize> {
    let n = x.len();
    if n < 2 {
        return None;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let variance: f64 = centered.iter().map(|v| v * v).sum();
    if variance <= 0.0 {
        return None;
    }
    (1..n / 2).find(|&lag| {
        let covariance: f64 = centered
            .iter()
            .zip(centered[lag..].iter())
            .map(|(a, b)| a * b)
            .sum();
        covariance <= 0.0
    })
}

/// Helper: linear regression slope for `(x, y)` pairs.
fn linear_slope(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    let sx: f64 = pairs.iter().map(|(x, _)| x).sum();
    let sy: f64 = pairs.iter().map(|(_, y)| y).sum();
    let sxy: f64 = pairs.iter().map(|(x, y)| x * y).sum();
    let sxx: f64 = pairs.iter().map(|(x, _)| x * x).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (n * sxy - sx * sy) / denom
}

// ─────────────────────────────────────────────────────────────────────────────
// DynamicalNormalForm
// ─────────────────────────────────────────────────────────────────────────────

/// Normal forms for codimension-1 bifurcations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BifurcationType {
    /// Saddle-node (fold): `ẋ = μ + x²`.
    SaddleNode,
    /// Transcritical: `ẋ = μx − x²`.
    Transcritical,
    /// Supercritical pitchfork: `ẋ = μx − x³`.
    PitchforkSuper,
    /// Subcritical pitchfork: `ẋ = μx + x³`.
    PitchforkSub,
    /// Supercritical Hopf (2D): `ṙ = μr − r³`.
    HopfSuper,
    /// Subcritical Hopf (2D): `ṙ = μr + r³`.
    HopfSub,
}

/// Normal-form vector field evaluator.
#[derive(Debug, Clone)]
pub struct DynamicalNormalForm {
    /// Bifurcation type.
    pub bif_type: BifurcationType,
    /// Bifurcation parameter `μ`.
    pub mu: f64,
}

impl DynamicalNormalForm {
    /// Create a new normal form.
    pub fn new(bif_type: BifurcationType, mu: f64) -> Self {
        Self { bif_type, mu }
    }

    /// Evaluate the vector field at state `x` (scalar for 1D, `(r, θ)` for Hopf).
    ///
    /// For 1D normal forms `state[0] = x`.  For Hopf `state[0] = r`, `state[1] = θ`.
    pub fn rhs(&self, state: &[f64]) -> Vec<f64> {
        match self.bif_type {
            BifurcationType::SaddleNode => {
                let x = state[0];
                vec![self.mu + x * x]
            }
            BifurcationType::Transcritical => {
                let x = state[0];
                vec![self.mu * x - x * x]
            }
            BifurcationType::PitchforkSuper => {
                let x = state[0];
                vec![self.mu * x - x * x * x]
            }
            BifurcationType::PitchforkSub => {
                let x = state[0];
                vec![self.mu * x + x * x * x]
            }
            BifurcationType::HopfSuper => {
                let r = state[0];
                let theta = if state.len() > 1 { state[1] } else { 0.0 };
                vec![self.mu * r - r * r * r, theta + 1.0]
            }
            BifurcationType::HopfSub => {
                let r = state[0];
                let theta = if state.len() > 1 { state[1] } else { 0.0 };
                vec![self.mu * r + r * r * r, theta + 1.0]
            }
        }
    }

    /// Fixed points of the 1D normal form.
    ///
    /// Returns up to 3 fixed-point values.  Empty vec if none exist for the
    /// given `μ`.
    pub fn fixed_points_1d(&self) -> Vec<f64> {
        match self.bif_type {
            BifurcationType::SaddleNode => {
                // ẋ = μ + x² = 0  ⟹  x = ±√(−μ)
                if self.mu <= 0.0 {
                    let s = (-self.mu).sqrt();
                    if s < 1e-12 { vec![0.0] } else { vec![-s, s] }
                } else {
                    vec![]
                }
            }
            BifurcationType::Transcritical => {
                // ẋ = μx − x² = x(μ − x) = 0  ⟹  x=0 or x=μ
                vec![0.0, self.mu]
            }
            BifurcationType::PitchforkSuper | BifurcationType::PitchforkSub => {
                // ẋ = μx ± x³ = x(μ ± x²) = 0  ⟹  x=0 and (for super) x=±√μ if μ>0
                let mut fps = vec![0.0];
                if self.bif_type == BifurcationType::PitchforkSuper && self.mu > 0.0 {
                    let s = self.mu.sqrt();
                    fps.push(-s);
                    fps.push(s);
                } else if self.bif_type == BifurcationType::PitchforkSub && self.mu < 0.0 {
                    let s = (-self.mu).sqrt();
                    fps.push(-s);
                    fps.push(s);
                }
                fps
            }
            BifurcationType::HopfSuper | BifurcationType::HopfSub => {
                // ṙ = 0  ⟹  r=0, and for super/sub: r=√(±μ)
                let mut fps = vec![0.0];
                if self.bif_type == BifurcationType::HopfSuper && self.mu > 0.0 {
                    fps.push(self.mu.sqrt());
                } else if self.bif_type == BifurcationType::HopfSub && self.mu < 0.0 {
                    fps.push((-self.mu).sqrt());
                }
                fps
            }
        }
    }

    /// Euler-integrate the normal form for `n` steps with step size `dt`.
    pub fn integrate_euler(&self, init: &[f64], dt: f64, n: usize) -> Vec<Vec<f64>> {
        let mut state = init.to_vec();
        let mut traj = Vec::with_capacity(n + 1);
        traj.push(state.clone());
        for _ in 0..n {
            let rhs = self.rhs(&state);
            for (s, r) in state.iter_mut().zip(rhs.iter()) {
                *s += dt * r;
            }
            traj.push(state.clone());
        }
        traj
    }

    /// Classify the stability of a 1D fixed point `x*` via the sign of the
    /// linearisation `f'(x*)`.
    pub fn stability_1d(&self, x_star: f64) -> Stability {
        let eps = 1e-7;
        let rhs_plus = self.rhs(&[x_star + eps])[0];
        let rhs_minus = self.rhs(&[x_star - eps])[0];
        let deriv = (rhs_plus - rhs_minus) / (2.0 * eps);
        if deriv < -1e-10 {
            Stability::Stable
        } else if deriv > 1e-10 {
            Stability::Unstable
        } else {
            Stability::Marginal
        }
    }
}

/// Stability classification of a fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stability {
    /// Asymptotically stable (attracting).
    Stable,
    /// Unstable (repelling).
    Unstable,
    /// Marginally stable (centre or degenerate).
    Marginal,
}

// ─────────────────────────────────────────────────────────────────────────────
// PoincareSection
// ─────────────────────────────────────────────────────────────────────────────

/// Poincaré section analysis for flows or maps.
///
/// Records intersections of a trajectory with a hyperplane defined by
/// `normal · (x − origin) = 0`.
#[derive(Debug, Clone)]
pub struct PoincareSection {
    /// Normal vector of the section hyperplane.
    pub normal: Vec<f64>,
    /// A point on the section hyperplane (origin).
    pub origin: Vec<f64>,
    /// Collected intersection points.
    pub intersections: Vec<Vec<f64>>,
}

impl PoincareSection {
    /// Create a new Poincaré section.
    pub fn new(normal: Vec<f64>, origin: Vec<f64>) -> Self {
        Self {
            normal,
            origin,
            intersections: Vec::new(),
        }
    }

    /// Signed distance of point `p` from the section.
    pub fn signed_distance(&self, p: &[f64]) -> f64 {
        p.iter()
            .zip(self.normal.iter())
            .zip(self.origin.iter())
            .map(|((pi, ni), oi)| (pi - oi) * ni)
            .sum()
    }

    /// Process a trajectory (list of states) and record all up-crossings of
    /// the section using linear interpolation.
    pub fn process_trajectory(&mut self, traj: &[Vec<f64>]) {
        for w in traj.windows(2) {
            let d0 = self.signed_distance(&w[0]);
            let d1 = self.signed_distance(&w[1]);
            // Up-crossing: d0 < 0 and d1 >= 0
            if d0 < 0.0 && d1 >= 0.0 {
                let t = d0.abs() / (d0.abs() + d1.abs() + 1e-300);
                let interp: Vec<f64> = w[0]
                    .iter()
                    .zip(w[1].iter())
                    .map(|(a, b)| a + t * (b - a))
                    .collect();
                self.intersections.push(interp);
            }
        }
    }

    /// Compute the return map: the sequence of successive intersection
    /// coordinates along axis `idx`.
    pub fn return_map_1d(&self, idx: usize) -> Vec<(f64, f64)> {
        let coords: Vec<f64> = self
            .intersections
            .iter()
            .filter_map(|p| p.get(idx).copied())
            .collect();
        coords.windows(2).map(|w| (w[0], w[1])).collect()
    }

    /// Rotation number: average angular advance between successive crossings.
    ///
    /// Assumes the section is in a 2D projection and computes the angle of
    /// each crossing relative to the section origin.
    pub fn rotation_number(&self, angle_idx_x: usize, angle_idx_y: usize) -> f64 {
        if self.intersections.len() < 2 {
            return 0.0;
        }
        let angles: Vec<f64> = self
            .intersections
            .iter()
            .filter_map(|p| {
                let x = p.get(angle_idx_x).copied()? - self.origin.get(angle_idx_x).copied()?;
                let y = p.get(angle_idx_y).copied()? - self.origin.get(angle_idx_y).copied()?;
                Some(y.atan2(x))
            })
            .collect();

        let mut total = 0.0f64;
        for w in angles.windows(2) {
            let mut diff = w[1] - w[0];
            // Wrap to [-π, π]
            while diff > std::f64::consts::PI {
                diff -= std::f64::consts::TAU;
            }
            while diff < -std::f64::consts::PI {
                diff += std::f64::consts::TAU;
            }
            total += diff;
        }
        total / (std::f64::consts::TAU * (angles.len() - 1) as f64)
    }

    /// Number of recorded intersections.
    pub fn count(&self) -> usize {
        self.intersections.len()
    }

    /// Clear all recorded intersections.
    pub fn clear(&mut self) {
        self.intersections.clear();
    }
}

/// Relative length of the central-difference perturbation used for the
/// Jacobian–vector products of the variational equation, `ε^(1/3)` for the
/// machine epsilon `ε`: it balances truncation (`h²`) against rounding (`ε/h`).
const JVP_STEP: f64 = 6.0e-6;

/// `base + h · direction`, componentwise.
fn shifted(base: &[f64], h: f64, direction: &[f64]) -> Vec<f64> {
    base.iter().zip(direction).map(|(b, v)| b + h * v).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// FlowMap
// ─────────────────────────────────────────────────────────────────────────────

/// Numerical flow integration for autonomous ODEs `ẋ = f(x)`.
///
/// Uses a fixed-step 4th-order Runge–Kutta integrator.
#[derive(Debug, Clone)]
pub struct FlowMap {
    /// Dimension of the state space.
    pub dim: usize,
    /// Integration step size.
    pub dt: f64,
}

impl FlowMap {
    /// Create a flow map integrator.
    pub fn new(dim: usize, dt: f64) -> Self {
        Self { dim, dt }
    }

    /// Advance the state by `n_steps` using RK4 with the vector field `f`.
    pub fn advance<F>(&self, init: &[f64], n_steps: usize, f: &F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let mut state = init.to_vec();
        for _ in 0..n_steps {
            state = self.rk4_step(&state, f);
        }
        state
    }

    /// Generate a trajectory (list of states) of length `n_steps + 1`.
    pub fn trajectory<F>(&self, init: &[f64], n_steps: usize, f: &F) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let mut traj = Vec::with_capacity(n_steps + 1);
        traj.push(init.to_vec());
        for _ in 0..n_steps {
            let next = self.rk4_step(traj.last().expect("trajectory is non-empty"), f);
            traj.push(next);
        }
        traj
    }

    /// Stroboscopic map: sample the trajectory at every `period` steps.
    pub fn stroboscopic_map<F>(
        &self,
        init: &[f64],
        n_periods: usize,
        steps_per_period: usize,
        f: &F,
    ) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let mut state = init.to_vec();
        let mut samples = Vec::with_capacity(n_periods + 1);
        samples.push(state.clone());
        for _ in 0..n_periods {
            state = self.advance(&state, steps_per_period, f);
            samples.push(state.clone());
        }
        samples
    }

    /// Fundamental solution matrix `Φ(T) = ∂x(T)/∂x(0)` of the flow after
    /// `n_steps` steps, i.e. the solution at `t = n_steps · dt` of the
    /// variational equation `Φ' = J(x(t)) Φ`, `Φ(0) = I`, along the trajectory
    /// starting at `init`.
    ///
    /// The trajectory and the `dim` columns of `Φ` are advanced together with the
    /// same four classical RK4 stages as [`advance`](Self::advance), with the
    /// Jacobian–vector products `J(x) φ` of every stage taken as central
    /// differences `(f(x + h φ) − f(x − h φ)) / 2h` of `f`.  `Φ` is therefore the
    /// derivative of the discrete flow map (up to the finite-difference error,
    /// about 1e-10) and no analytic Jacobian is needed.
    ///
    /// Returns the `dim × dim` matrix in row-major order with
    /// `Φ[i][j] = ∂x_i(T)/∂x_j(0)`: column `j` is the unit vector `e_j`
    /// transported along the flow.  For `n_steps = 0` this is the identity.
    ///
    /// Returns an empty matrix if `init.len() != dim` or `f` does not map
    /// `dim`-vectors to `dim`-vectors.
    pub fn fundamental_solution_matrix<F>(
        &self,
        init: &[f64],
        n_steps: usize,
        f: &F,
    ) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        self.try_fundamental_solution_matrix(init, n_steps, f)
            .unwrap_or_default()
    }

    /// [`fundamental_solution_matrix`](Self::fundamental_solution_matrix) with
    /// `None` standing for inconsistent dimensions.
    fn try_fundamental_solution_matrix<F>(
        &self,
        init: &[f64],
        n_steps: usize,
        f: &F,
    ) -> Option<Vec<Vec<f64>>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let d = self.dim;
        if init.len() != d {
            return None;
        }
        let dt = self.dt;
        let field = |x: &[f64]| -> Option<Vec<f64>> {
            let value = f(x);
            (value.len() == d).then_some(value)
        };
        // J(x) φ by a central difference whose perturbation has the length
        // `JVP_STEP · max(1, |x|∞)`.
        let jvp = |x: &[f64], phi: &[f64]| -> Option<Vec<f64>> {
            let norm = phi.iter().map(|a| a * a).sum::<f64>().sqrt();
            if norm == 0.0 {
                return Some(vec![0.0; d]);
            }
            let scale = x.iter().fold(1.0_f64, |m, a| m.max(a.abs()));
            let h = JVP_STEP * scale / norm;
            let plus = shifted(x, h, phi);
            let minus = shifted(x, -h, phi);
            let f_plus = field(&plus)?;
            let f_minus = field(&minus)?;
            Some(
                f_plus
                    .iter()
                    .zip(&f_minus)
                    .map(|(a, b)| (a - b) / (2.0 * h))
                    .collect(),
            )
        };

        let mut state = init.to_vec();
        // columns[c] is the transported unit vector e_c.
        let mut columns: Vec<Vec<f64>> = (0..d)
            .map(|c| {
                let mut e = vec![0.0; d];
                e[c] = 1.0;
                e
            })
            .collect();
        for _ in 0..n_steps {
            let k1 = field(&state)?;
            let s2 = shifted(&state, 0.5 * dt, &k1);
            let k2 = field(&s2)?;
            let s3 = shifted(&state, 0.5 * dt, &k2);
            let k3 = field(&s3)?;
            let s4 = shifted(&state, dt, &k3);
            let k4 = field(&s4)?;
            for phi in &mut columns {
                let v1 = jvp(&state, phi)?;
                let v2 = jvp(&s2, &shifted(phi, 0.5 * dt, &v1))?;
                let v3 = jvp(&s3, &shifted(phi, 0.5 * dt, &v2))?;
                let v4 = jvp(&s4, &shifted(phi, dt, &v3))?;
                for (i, p) in phi.iter_mut().enumerate() {
                    *p += dt / 6.0 * (v1[i] + 2.0 * v2[i] + 2.0 * v3[i] + v4[i]);
                }
            }
            for (i, x) in state.iter_mut().enumerate() {
                *x += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
        }
        Some(
            (0..d)
                .map(|i| columns.iter().map(|column| column[i]).collect())
                .collect(),
        )
    }

    fn rk4_step<F>(&self, state: &[f64], f: &F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = f(state);
        let s2: Vec<f64> = state
            .iter()
            .zip(k1.iter())
            .map(|(s, k)| s + 0.5 * self.dt * k)
            .collect();
        let k2 = f(&s2);
        let s3: Vec<f64> = state
            .iter()
            .zip(k2.iter())
            .map(|(s, k)| s + 0.5 * self.dt * k)
            .collect();
        let k3 = f(&s3);
        let s4: Vec<f64> = state
            .iter()
            .zip(k3.iter())
            .map(|(s, k)| s + self.dt * k)
            .collect();
        let k4 = f(&s4);
        state
            .iter()
            .enumerate()
            .map(|(i, s)| s + self.dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SynchronizationAnalysis
// ─────────────────────────────────────────────────────────────────────────────

/// Synchronisation analysis for coupled oscillators / chaotic systems.
#[derive(Debug, Clone)]
pub struct SynchronizationAnalysis;

impl SynchronizationAnalysis {
    /// Pecora–Carroll synchronisation check.
    ///
    /// Drives a response system (with the same structure as the drive) by
    /// replacing one of its variables with the drive's corresponding variable.
    ///
    /// Returns `true` if the synchronisation error decays below `tol`.
    ///
    /// `drive_orbit` and `response_orbit` are trajectories of the same length;
    /// `sync_idx` is the variable index that is replaced in the response.
    pub fn pecora_carroll_sync(
        drive_orbit: &[Vec<f64>],
        response_orbit: &[Vec<f64>],
        sync_idx: usize,
        tol: f64,
    ) -> bool {
        if drive_orbit.is_empty() || response_orbit.is_empty() {
            return false;
        }
        // Check that the final error (excluding sync_idx variable) is small
        let last_drive = drive_orbit.last().expect("drive orbit is non-empty");
        let last_response = response_orbit.last().expect("response orbit is non-empty");
        let err: f64 = last_drive
            .iter()
            .zip(last_response.iter())
            .enumerate()
            .filter(|(i, _)| *i != sync_idx)
            .map(|(_, (d, r))| (d - r).powi(2))
            .sum::<f64>()
            .sqrt();
        err < tol
    }

    /// Phase synchronisation index (mean resultant length of phase differences).
    ///
    /// Given two phase time-series `phi1` and `phi2` (in radians), computes
    ///
    /// `R = |⟨exp(i(φ₁ − φ₂))⟩|`
    ///
    /// `R = 1` means perfect phase synchronisation; `R ≈ 0` means asynchronous.
    pub fn phase_synchronization_index(phi1: &[f64], phi2: &[f64]) -> f64 {
        let n = phi1.len().min(phi2.len());
        if n == 0 {
            return 0.0;
        }
        let sum_cos: f64 = phi1
            .iter()
            .zip(phi2.iter())
            .map(|(a, b)| (a - b).cos())
            .sum();
        let sum_sin: f64 = phi1
            .iter()
            .zip(phi2.iter())
            .map(|(a, b)| (a - b).sin())
            .sum();
        let n_f = n as f64;
        ((sum_cos / n_f).powi(2) + (sum_sin / n_f).powi(2)).sqrt()
    }

    /// Generalised synchronisation check via the auxiliary system method.
    ///
    /// Two copies of the response system starting at different initial conditions
    /// should converge to the same trajectory if GS holds.  Returns `true` if
    /// the distance between `resp1` and `resp2` at the end is below `tol`.
    pub fn generalised_sync_check(resp1: &[Vec<f64>], resp2: &[Vec<f64>], tol: f64) -> bool {
        let last1 = match resp1.last() {
            Some(v) => v,
            None => return false,
        };
        let last2 = match resp2.last() {
            Some(v) => v,
            None => return false,
        };
        let dist: f64 = last1
            .iter()
            .zip(last2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        dist < tol
    }

    /// Mutual information estimate between two scalar signals using histogram
    /// binning with `n_bins` bins.
    pub fn mutual_information(x: &[f64], y: &[f64], n_bins: usize) -> f64 {
        let n = x.len().min(y.len());
        if n == 0 || n_bins == 0 {
            return 0.0;
        }
        let x_min = x[..n].iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x[..n].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = y[..n].iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y[..n].iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let x_range = (x_max - x_min).max(1e-15);
        let y_range = (y_max - y_min).max(1e-15);

        let mut joint = vec![vec![0u64; n_bins]; n_bins];
        let mut hist_x = vec![0u64; n_bins];
        let mut hist_y = vec![0u64; n_bins];

        for i in 0..n {
            let ix = ((x[i] - x_min) / x_range * n_bins as f64)
                .floor()
                .clamp(0.0, (n_bins - 1) as f64) as usize;
            let iy = ((y[i] - y_min) / y_range * n_bins as f64)
                .floor()
                .clamp(0.0, (n_bins - 1) as f64) as usize;
            joint[ix][iy] += 1;
            hist_x[ix] += 1;
            hist_y[iy] += 1;
        }

        let n_f = n as f64;
        let mut mi = 0.0f64;
        for (joint_row, &px_count) in joint.iter().zip(hist_x.iter()) {
            let p_x = px_count as f64 / n_f;
            for (jval, &py_count) in joint_row.iter().zip(hist_y.iter()) {
                let p_xy = *jval as f64 / n_f;
                let p_y = py_count as f64 / n_f;
                if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                    mi += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }
        mi
    }

    /// Instantaneous phase of a signal from its Hilbert-transform analytic
    /// signal.
    ///
    /// The mean is removed, the analytic signal `z = x + i·H[x]` is formed by
    /// [`crate::signal_processing::analytic_signal`] (FFT based; the signal is
    /// zero-padded to the next power of two) and the phase `atan2(Im z, Re z)`
    /// is unwrapped by removing the `2π` jumps between consecutive samples, so
    /// the result is continuous.  A monochromatic signal of frequency `f`
    /// sampled at `f_s` gives a straight line of slope `2π f / f_s` per sample.
    ///
    /// Signals with fewer than two samples, and signals that are constant
    /// (nothing is left after the mean is removed), return zeros: without an
    /// oscillation there is no phase.
    pub fn instantaneous_phase(signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        if n < 2 {
            return vec![0.0; n];
        }
        let mean = signal.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = signal.iter().map(|s| s - mean).collect();
        if centered.iter().all(|&c| c == 0.0) {
            return vec![0.0; n];
        }
        let analytic = crate::signal_processing::analytic_signal(&centered);

        let tau = std::f64::consts::TAU;
        let mut phase = Vec::with_capacity(n);
        let mut previous_wrapped: Option<f64> = None;
        let mut unwrap_offset = 0.0f64;
        for z in &analytic {
            let wrapped = z.im.atan2(z.re);
            if let Some(previous) = previous_wrapped {
                unwrap_offset -= tau * ((wrapped - previous) / tau).round();
            }
            phase.push(wrapped + unwrap_offset);
            previous_wrapped = Some(wrapped);
        }
        phase
    }

    /// Cross-correlation between two signals at lag 0 (normalised).
    pub fn cross_correlation_zero_lag(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n == 0 {
            return 0.0;
        }
        let mx = x.iter().sum::<f64>() / n as f64;
        let my = y.iter().sum::<f64>() / n as f64;
        let num: f64 = x[..n]
            .iter()
            .zip(y[..n].iter())
            .map(|(a, b)| (a - mx) * (b - my))
            .sum();
        let sx: f64 = x[..n].iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
        let sy: f64 = y[..n].iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();
        if sx < 1e-15 || sy < 1e-15 {
            return 0.0;
        }
        num / (sx * sy)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenz system helper (used in tests)
// ─────────────────────────────────────────────────────────────────────────────

/// Lorenz system parameters.
#[derive(Debug, Clone)]
pub struct LorenzSystem {
    /// σ parameter.
    pub sigma: f64,
    /// ρ parameter.
    pub rho: f64,
    /// β parameter.
    pub beta: f64,
}

impl LorenzSystem {
    /// Create with standard chaotic parameters `σ=10, ρ=28, β=8/3`.
    pub fn standard() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        }
    }

    /// Evaluate the Lorenz vector field at `(x, y, z)`.
    pub fn rhs(&self, state: &[f64]) -> Vec<f64> {
        let (x, y, z) = (state[0], state[1], state[2]);
        vec![
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;

#[cfg(test)]
mod regression_tests;
