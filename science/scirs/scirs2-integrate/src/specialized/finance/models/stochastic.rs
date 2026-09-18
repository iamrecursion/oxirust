//! Stochastic process models for asset price dynamics

use std::fmt::Debug;

/// Advanced stochastic processes
#[derive(Debug, Clone)]
pub enum StochasticProcess {
    /// Geometric Brownian Motion (Black-Scholes)
    GeometricBrownian {
        /// Drift parameter
        mu: f64,
        /// Diffusion parameter
        sigma: f64,
    },
    /// Variance Gamma process
    VarianceGamma {
        /// Drift of the Brownian motion with random time
        theta: f64,
        /// Volatility of the Brownian motion with random time
        sigma: f64,
        /// Variance rate of the time change
        nu: f64,
    },
    /// Normal Inverse Gaussian process
    NormalInverseGaussian {
        /// Asymmetry parameter
        alpha: f64,
        /// Tail heaviness parameter
        beta: f64,
        /// Scale parameter
        delta: f64,
        /// Location parameter
        mu: f64,
    },
    /// CGMY/KoBoL process
    CGMY {
        /// Fine structure of price jumps near zero
        c: f64,
        /// Positive jump activity
        g: f64,
        /// Negative jump activity
        m: f64,
        /// Blowup rate of jump activity near zero
        y: f64,
    },
    /// Merton jump diffusion
    MertonJumpDiffusion {
        /// GBM parameters
        mu: f64,
        sigma: f64,
        /// Jump parameters
        lambda: f64,
        mu_jump: f64,
        sigma_jump: f64,
    },
}

/// Jump process specification
pub enum JumpProcess {
    /// Simple Poisson jump process
    Poisson {
        /// Jump intensity (average number of jumps per year)
        lambda: f64,
        /// Mean jump size
        mu_jump: f64,
        /// Jump size standard deviation
        sigma_jump: f64,
    },
    /// Double exponential jump process (Kou model)
    DoubleExponential {
        /// Jump intensity
        lambda: f64,
        /// Probability of upward jump
        p: f64,
        /// Rate parameter for positive jumps
        eta_up: f64,
        /// Rate parameter for negative jumps
        eta_down: f64,
    },
    /// Compound Poisson with normal jumps
    CompoundPoissonNormal {
        /// Jump intensity
        lambda: f64,
        /// Jump size mean
        mu: f64,
        /// Jump size variance
        sigma_squared: f64,
    },
    /// Jump process with time-varying intensity
    TimeVaryingIntensity {
        /// Intensity function
        intensity_fn: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        /// Jump size distribution parameters
        mu_jump: f64,
        sigma_jump: f64,
    },
}

impl Debug for JumpProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JumpProcess::Poisson {
                lambda,
                mu_jump,
                sigma_jump,
            } => f
                .debug_struct("Poisson")
                .field("lambda", lambda)
                .field("mu_jump", mu_jump)
                .field("sigma_jump", sigma_jump)
                .finish(),
            JumpProcess::DoubleExponential {
                lambda,
                p,
                eta_up,
                eta_down,
            } => f
                .debug_struct("DoubleExponential")
                .field("lambda", lambda)
                .field("p", p)
                .field("eta_up", eta_up)
                .field("eta_down", eta_down)
                .finish(),
            JumpProcess::CompoundPoissonNormal {
                lambda,
                mu,
                sigma_squared,
            } => f
                .debug_struct("CompoundPoissonNormal")
                .field("lambda", lambda)
                .field("mu", mu)
                .field("sigma_squared", sigma_squared)
                .finish(),
            JumpProcess::TimeVaryingIntensity {
                mu_jump,
                sigma_jump,
                ..
            } => f
                .debug_struct("TimeVaryingIntensity")
                .field("intensity_fn", &"<function>")
                .field("mu_jump", mu_jump)
                .field("sigma_jump", sigma_jump)
                .finish(),
        }
    }
}

impl Clone for JumpProcess {
    fn clone(&self) -> Self {
        match self {
            JumpProcess::Poisson {
                lambda,
                mu_jump,
                sigma_jump,
            } => JumpProcess::Poisson {
                lambda: *lambda,
                mu_jump: *mu_jump,
                sigma_jump: *sigma_jump,
            },
            JumpProcess::DoubleExponential {
                lambda,
                p,
                eta_up,
                eta_down,
            } => JumpProcess::DoubleExponential {
                lambda: *lambda,
                p: *p,
                eta_up: *eta_up,
                eta_down: *eta_down,
            },
            JumpProcess::CompoundPoissonNormal {
                lambda,
                mu,
                sigma_squared,
            } => JumpProcess::CompoundPoissonNormal {
                lambda: *lambda,
                mu: *mu,
                sigma_squared: *sigma_squared,
            },
            JumpProcess::TimeVaryingIntensity {
                intensity_fn: _,
                mu_jump,
                sigma_jump,
            } => {
                // Note: Function pointers cannot be cloned in general
                // For now, we'll create a simple constant intensity
                let constant_intensity = 0.1; // Default intensity
                JumpProcess::Poisson {
                    lambda: constant_intensity,
                    mu_jump: *mu_jump,
                    sigma_jump: *sigma_jump,
                }
            }
        }
    }
}
