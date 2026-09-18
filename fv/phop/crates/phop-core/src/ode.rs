//! Governing-equation discovery: recover the right-hand side of an autonomous ODE
//! `dx/dt = f(x)` from a sampled trajectory, using phop as the function learner.
//!
//! This is the "phop for dynamical systems" research direction: estimate the derivative `dx/dt`
//! from a uniformly-sampled scalar series by central differences, then run the standard discovery
//! pipeline on the regression `x ↦ dx/dt`. The recovered law *is* the governing equation. (It is a
//! lightweight, self-contained cousin of SINDy — phop supplies the nonlinear function class instead
//! of a fixed sparse library; coupling to `oxieml::symreg::{sindy, pde}` for multi-variable systems
//! is future work.)

use crate::config::Config;
use crate::dataset::DataSet;
use crate::discoverer::Discoverer;
use crate::error::{PhopError, Result};
use crate::pareto::ParetoFront;
use scirs2_core::ndarray::{Array1, Array2};

/// Discover the right-hand side `f` of an autonomous scalar ODE `dx/dt = f(x)` from a
/// uniformly-sampled trajectory `series` with timestep `dt`.
///
/// Derivatives at interior samples are estimated with the central difference
/// `(x[i+1] − x[i−1]) / (2·dt)`; the resulting `(x, dx/dt)` pairs are handed to [`Discoverer::fit`].
/// The returned Pareto front is over candidate `f`.
///
/// # Errors
/// Returns [`PhopError::ShapeMismatch`] if the series is too short (fewer than 3 samples) or `dt`
/// is non-positive, or propagates any discovery error.
pub fn discover_ode(series: &[f64], dt: f64, cfg: &Config) -> Result<ParetoFront> {
    if series.len() < 3 {
        return Err(PhopError::ShapeMismatch(
            "need at least 3 samples to estimate a derivative".to_string(),
        ));
    }
    if dt <= 0.0 || dt.is_nan() {
        return Err(PhopError::ShapeMismatch("dt must be positive".to_string()));
    }
    let mut xs = Vec::with_capacity(series.len() - 2);
    let mut dxdt = Vec::with_capacity(series.len() - 2);
    for i in 1..series.len() - 1 {
        xs.push(series[i]);
        dxdt.push((series[i + 1] - series[i - 1]) / (2.0 * dt));
    }
    let x = Array2::from_shape_vec((xs.len(), 1), xs)
        .map_err(|e| PhopError::ShapeMismatch(e.to_string()))?;
    let ds = DataSet::from_arrays(x, Array1::from(dxdt))?;
    Discoverer::new(cfg.clone()).fit(&ds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_exponential_growth_ode() {
        // x(t) = e^t solves dx/dt = x. Sampling e^t and recovering f should give f(x) = x.
        let dt = 0.02;
        let series: Vec<f64> = (0..200).map(|i| (i as f64 * dt).exp()).collect();
        let cfg = Config::default().max_depth(2).max_epochs(200).seed(0);
        let front = discover_ode(&series, dt, &cfg).expect("ode discovery");
        let best = front.best().expect("non-empty front");
        // dx/dt = x is the identity in x; central-difference error is O(dt^2), so the relative fit
        // is tight. Use a relative-MSE check against the variance of dx/dt (= variance of x).
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let var = series.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / series.len() as f64;
        assert!(
            best.mse < var * 1e-3,
            "ODE rhs not recovered: mse {} vs var {} ({})",
            best.mse,
            var,
            best.pretty()
        );
    }
}
