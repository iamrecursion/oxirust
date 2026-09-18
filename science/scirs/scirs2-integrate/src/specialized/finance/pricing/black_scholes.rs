//! Black-Scholes model implementations

use crate::error::IntegrateResult;
use crate::specialized::finance::types::{FinancialOption, OptionType};

/// Calculate the cumulative normal distribution
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

/// Black-Scholes formula for European options
pub fn black_scholes_price(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    volatility: f64,
    time: f64,
    option_type: OptionType,
) -> f64 {
    let d1 = ((spot / strike).ln() + (rate - dividend + 0.5 * volatility * volatility) * time)
        / (volatility * time.sqrt());
    let d2 = d1 - volatility * time.sqrt();

    match option_type {
        OptionType::Call => {
            spot * (-(dividend * time)).exp() * normal_cdf(d1)
                - strike * (-(rate * time)).exp() * normal_cdf(d2)
        }
        OptionType::Put => {
            strike * (-(rate * time)).exp() * normal_cdf(-d2)
                - spot * (-(dividend * time)).exp() * normal_cdf(-d1)
        }
    }
}
