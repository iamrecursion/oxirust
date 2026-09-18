//! Greeks calculations for option risk management

use crate::specialized::finance::pricing::black_scholes::normal_cdf;
use crate::specialized::finance::types::{FinancialOption, OptionType};

/// Greeks - risk sensitivities for options
#[derive(Debug, Clone)]
pub struct Greeks {
    /// Delta - rate of change of option price with respect to underlying asset price
    pub delta: f64,
    /// Gamma - rate of change of delta with respect to underlying asset price
    pub gamma: f64,
    /// Vega - sensitivity to volatility
    pub vega: f64,
    /// Theta - time decay
    pub theta: f64,
    /// Rho - sensitivity to interest rate
    pub rho: f64,
}

impl Greeks {
    /// Calculate Greeks for a European option using Black-Scholes
    pub fn black_scholes(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: OptionType,
    ) -> Self {
        let sqrt_time = time.sqrt();
        let d1 = ((spot / strike).ln() + (rate - dividend + 0.5 * volatility * volatility) * time)
            / (volatility * sqrt_time);
        let d2 = d1 - volatility * sqrt_time;

        let nd1 = normal_cdf(d1);
        let nd2 = normal_cdf(d2);
        let nprime_d1 = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * d1 * d1).exp();

        let delta = match option_type {
            OptionType::Call => (-(dividend * time)).exp() * nd1,
            OptionType::Put => -(-(dividend * time)).exp() * normal_cdf(-d1),
        };

        let gamma = (-(dividend * time)).exp() * nprime_d1 / (spot * volatility * sqrt_time);

        let vega = spot * (-(dividend * time)).exp() * nprime_d1 * sqrt_time / 100.0;

        let theta = match option_type {
            OptionType::Call => {
                (-spot * nprime_d1 * volatility * (-(dividend * time)).exp() / (2.0 * sqrt_time)
                    - rate * strike * (-(rate * time)).exp() * nd2
                    + dividend * spot * (-(dividend * time)).exp() * nd1)
                    / 365.0
            }
            OptionType::Put => {
                (-spot * nprime_d1 * volatility * (-(dividend * time)).exp() / (2.0 * sqrt_time)
                    + rate * strike * (-(rate * time)).exp() * normal_cdf(-d2)
                    - dividend * spot * (-(dividend * time)).exp() * normal_cdf(-d1))
                    / 365.0
            }
        };

        let rho = match option_type {
            OptionType::Call => strike * time * (-(rate * time)).exp() * nd2 / 100.0,
            OptionType::Put => -strike * time * (-(rate * time)).exp() * normal_cdf(-d2) / 100.0,
        };

        Greeks {
            delta,
            gamma,
            vega,
            theta,
            rho,
        }
    }
}
