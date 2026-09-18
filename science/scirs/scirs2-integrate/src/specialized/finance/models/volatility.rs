//! Volatility models for financial derivatives pricing

/// Volatility model
pub enum VolatilityModel {
    /// Constant volatility (Black-Scholes)
    Constant(f64),
    /// Heston stochastic volatility model
    Heston {
        /// Initial volatility
        v0: f64,
        /// Long-term variance
        theta: f64,
        /// Mean reversion speed
        kappa: f64,
        /// Volatility of volatility
        sigma: f64,
        /// Correlation between asset and volatility
        rho: f64,
    },
    /// SABR (Stochastic Alpha Beta Rho) model
    SABR {
        /// Initial volatility
        alpha: f64,
        /// CEV exponent
        beta: f64,
        /// Volatility of volatility
        nu: f64,
        /// Correlation
        rho: f64,
    },
    /// Local volatility surface
    LocalVolatility(Box<dyn Fn(f64, f64) -> f64 + Send + Sync>),
    /// Bates model (Heston + jumps in volatility)
    Bates {
        /// Heston parameters
        v0: f64,
        theta: f64,
        kappa: f64,
        sigma: f64,
        rho: f64,
        /// Jump parameters for volatility
        lambda_v: f64,
        mu_v: f64,
        sigma_v: f64,
    },
    /// Hull-White stochastic volatility
    HullWhite {
        /// Initial volatility
        v0: f64,
        /// Volatility drift
        alpha: f64,
        /// Volatility of volatility
        beta: f64,
        /// Correlation
        rho: f64,
    },
    /// 3/2 stochastic volatility model
    ThreeHalves {
        /// Initial variance
        v0: f64,
        /// Long-term variance
        theta: f64,
        /// Mean reversion speed
        kappa: f64,
        /// Volatility of volatility
        sigma: f64,
        /// Correlation
        rho: f64,
    },
}

/// Heston model parameters
#[derive(Debug, Clone)]
pub struct HestonModelParams {
    /// Initial volatility
    pub v0: f64,
    /// Long-term variance
    pub theta: f64,
    /// Mean reversion speed
    pub kappa: f64,
    /// Volatility of volatility
    pub sigma: f64,
    /// Correlation between asset and volatility
    pub rho: f64,
    /// Time to maturity
    pub maturity: f64,
    /// Initial asset price
    pub initial_price: f64,
    /// Initial variance (for backwards compatibility)
    pub initial_variance: f64,
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Correlation alias for backwards compatibility
    pub correlation: f64,
    /// Volatility of volatility alias for backwards compatibility
    pub vol_of_vol: f64,
}
