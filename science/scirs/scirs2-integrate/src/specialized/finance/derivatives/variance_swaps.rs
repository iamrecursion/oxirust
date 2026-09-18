//! Variance and volatility derivatives
//!
//! This module implements pricing and hedging strategies for variance swaps, volatility swaps,
//! and other volatility-based derivatives using model-free implied variance.

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::finance::pricing::black_scholes::black_scholes_price;
use crate::specialized::finance::types::OptionType;

/// Result of variance swap valuation
#[derive(Debug, Clone)]
pub struct VarianceSwapResult {
    /// Fair variance strike
    pub fair_strike: f64,
    /// Realized variance (if historical data provided)
    pub realized_variance: Option<f64>,
    /// Payoff given realized variance
    pub payoff: Option<f64>,
    /// Vega notional (sensitivity to variance)
    pub vega_notional: f64,
}

/// Variance swap contract
#[derive(Debug, Clone)]
pub struct VarianceSwap {
    /// Variance strike (fixed leg)
    pub strike: f64,
    /// Notional amount (in variance units)
    pub notional: f64,
    /// Time to maturity in years
    pub maturity: f64,
    /// Number of observations per year
    pub observations_per_year: usize,
    /// Risk-free rate
    pub risk_free_rate: f64,
}

impl VarianceSwap {
    /// Create a new variance swap
    pub fn new(
        strike: f64,
        notional: f64,
        maturity: f64,
        observations_per_year: usize,
        risk_free_rate: f64,
    ) -> IntegrateResult<Self> {
        if strike < 0.0 {
            return Err(IntegrateError::ValueError(
                "Variance strike must be non-negative".to_string(),
            ));
        }

        if notional <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Notional must be positive".to_string(),
            ));
        }

        if maturity <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Maturity must be positive".to_string(),
            ));
        }

        Ok(Self {
            strike,
            notional,
            maturity,
            observations_per_year,
            risk_free_rate,
        })
    }

    /// Calculate payoff given realized variance
    pub fn payoff(&self, realized_variance: f64) -> f64 {
        self.notional * (realized_variance - self.strike)
    }

    /// Calculate fair variance strike from option prices using log-contract replication
    ///
    /// Uses model-free implied variance formula:
    /// Var = (2/T) × ∫[e^(-rT) × (option_price(K) / K²) dK]
    pub fn fair_strike_from_options(
        spot: f64,
        maturity: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        strikes: &[f64],
        call_prices: &[f64],
        put_prices: &[f64],
    ) -> IntegrateResult<f64> {
        if strikes.len() != call_prices.len() || strikes.len() != put_prices.len() {
            return Err(IntegrateError::ValueError(
                "Strike and price arrays must have same length".to_string(),
            ));
        }

        if strikes.len() < 2 {
            return Err(IntegrateError::ValueError(
                "Need at least 2 strikes for integration".to_string(),
            ));
        }

        let discount = (-risk_free_rate * maturity).exp();
        let forward = spot * ((risk_free_rate - dividend_yield) * maturity).exp();

        let mut integral = 0.0;

        // Find closest strike to forward (ATM)
        let atm_idx = strikes
            .iter()
            .enumerate()
            .min_by(|(_, &k1), (_, &k2)| {
                (k1 - forward)
                    .abs()
                    .partial_cmp(&(k2 - forward).abs())
                    .expect("Operation failed")
            })
            .map(|(i, _)| i)
            .expect("Operation failed");

        // OTM puts: K < F
        for i in 0..atm_idx {
            if i + 1 < strikes.len() {
                let k1 = strikes[i];
                let k2 = strikes[i + 1];
                let dk = k2 - k1;
                let mid_k = (k1 + k2) / 2.0;
                let put_price = (put_prices[i] + put_prices[i + 1]) / 2.0;

                integral += (dk / (mid_k * mid_k)) * put_price;
            }
        }

        // ATM options
        let atm_strike = strikes[atm_idx];
        let atm_call = call_prices[atm_idx];
        let atm_put = put_prices[atm_idx];

        if atm_idx > 0 && atm_idx < strikes.len() - 1 {
            let dk = strikes[atm_idx + 1] - strikes[atm_idx - 1];
            integral += (dk / (atm_strike * atm_strike)) * ((atm_call + atm_put) / 2.0);
        }

        // OTM calls: K > F
        for i in atm_idx..strikes.len() - 1 {
            let k1 = strikes[i];
            let k2 = strikes[i + 1];
            let dk = k2 - k1;
            let mid_k = (k1 + k2) / 2.0;
            let call_price = (call_prices[i] + call_prices[i + 1]) / 2.0;

            integral += (dk / (mid_k * mid_k)) * call_price;
        }

        // Model-free implied variance
        let variance = (2.0 * discount / maturity) * integral
            - ((forward / spot).ln() - (risk_free_rate - dividend_yield) * maturity).powi(2)
                / maturity;

        Ok(variance.max(0.0))
    }

    /// Calculate realized variance from historical returns
    ///
    /// Uses the standard estimator: Var = (252/n) × Σ(r_i²)
    /// where r_i are log returns
    pub fn realized_variance(returns: &[f64], annualization_factor: f64) -> IntegrateResult<f64> {
        if returns.is_empty() {
            return Err(IntegrateError::ValueError(
                "Returns array cannot be empty".to_string(),
            ));
        }

        let sum_squared: f64 = returns.iter().map(|r| r * r).sum();
        let variance = (annualization_factor / returns.len() as f64) * sum_squared;

        Ok(variance)
    }
}

/// Volatility swap contract (pays on volatility, not variance)
#[derive(Debug, Clone)]
pub struct VolatilitySwap {
    /// Volatility strike (fixed leg)
    pub strike: f64,
    /// Notional amount (in volatility units)
    pub notional: f64,
    /// Time to maturity
    pub maturity: f64,
    /// Variance notional for hedging
    pub variance_notional: f64,
}

impl VolatilitySwap {
    /// Create a new volatility swap
    pub fn new(strike: f64, notional: f64, maturity: f64) -> IntegrateResult<Self> {
        if strike < 0.0 {
            return Err(IntegrateError::ValueError(
                "Volatility strike must be non-negative".to_string(),
            ));
        }

        if notional <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Notional must be positive".to_string(),
            ));
        }

        // For hedging, convert vol notional to variance notional
        let variance_notional = notional / (2.0 * strike);

        Ok(Self {
            strike,
            notional,
            maturity,
            variance_notional,
        })
    }

    /// Calculate payoff given realized volatility
    pub fn payoff(&self, realized_volatility: f64) -> f64 {
        self.notional * (realized_volatility - self.strike)
    }

    /// Calculate fair volatility strike from variance strike with convexity adjustment
    ///
    /// Uses approximation: Vol ≈ √Var - (Vvol / 8) × Var^(-3/2)
    /// where Vvol is the variance of variance
    pub fn fair_strike_from_variance(
        variance_strike: f64,
        variance_of_variance: f64,
    ) -> IntegrateResult<f64> {
        if variance_strike < 0.0 {
            return Err(IntegrateError::ValueError(
                "Variance strike must be non-negative".to_string(),
            ));
        }

        let vol = variance_strike.sqrt();

        // Convexity adjustment (Jensen's inequality correction)
        let convexity_adjustment = if variance_strike > 1e-10 {
            (variance_of_variance / 8.0) * variance_strike.powf(-1.5)
        } else {
            0.0
        };

        Ok(vol - convexity_adjustment)
    }

    /// Calculate realized volatility from returns
    pub fn realized_volatility(returns: &[f64], annualization_factor: f64) -> IntegrateResult<f64> {
        let variance = VarianceSwap::realized_variance(returns, annualization_factor)?;
        Ok(variance.sqrt())
    }
}

/// Option chain for variance swap pricing
#[derive(Debug, Clone)]
pub struct OptionChain {
    /// Underlying spot price
    pub spot: f64,
    /// Time to maturity
    pub maturity: f64,
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Dividend yield
    pub dividend_yield: f64,
    /// Strike prices
    pub strikes: Vec<f64>,
    /// Call option prices
    pub call_prices: Vec<f64>,
    /// Put option prices
    pub put_prices: Vec<f64>,
}

impl OptionChain {
    /// Create option chain from Black-Scholes model with constant volatility
    pub fn from_black_scholes(
        spot: f64,
        maturity: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        volatility: f64,
        strike_range: (f64, f64),
        n_strikes: usize,
    ) -> IntegrateResult<Self> {
        if n_strikes < 2 {
            return Err(IntegrateError::ValueError(
                "Need at least 2 strikes".to_string(),
            ));
        }

        let (k_min, k_max) = strike_range;
        let dk = (k_max - k_min) / (n_strikes - 1) as f64;

        let mut strikes = Vec::with_capacity(n_strikes);
        let mut call_prices = Vec::with_capacity(n_strikes);
        let mut put_prices = Vec::with_capacity(n_strikes);

        for i in 0..n_strikes {
            let strike = k_min + i as f64 * dk;
            strikes.push(strike);

            let call = black_scholes_price(
                spot,
                strike,
                risk_free_rate,
                dividend_yield,
                volatility,
                maturity,
                OptionType::Call,
            );
            call_prices.push(call);

            let put = black_scholes_price(
                spot,
                strike,
                risk_free_rate,
                dividend_yield,
                volatility,
                maturity,
                OptionType::Put,
            );
            put_prices.push(put);
        }

        Ok(Self {
            spot,
            maturity,
            risk_free_rate,
            dividend_yield,
            strikes,
            call_prices,
            put_prices,
        })
    }

    /// Calculate fair variance strike from this option chain
    pub fn fair_variance_strike(&self) -> IntegrateResult<f64> {
        VarianceSwap::fair_strike_from_options(
            self.spot,
            self.maturity,
            self.risk_free_rate,
            self.dividend_yield,
            &self.strikes,
            &self.call_prices,
            &self.put_prices,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_swap_creation() {
        let swap = VarianceSwap::new(0.04, 100000.0, 1.0, 252, 0.05).expect("Operation failed");

        assert_eq!(swap.strike, 0.04);
        assert_eq!(swap.notional, 100000.0);
        assert_eq!(swap.maturity, 1.0);
    }

    #[test]
    fn test_variance_swap_payoff() {
        let swap = VarianceSwap::new(0.04, 100000.0, 1.0, 252, 0.05).expect("Operation failed");

        // Realized variance = 0.05 (higher than strike)
        let payoff = swap.payoff(0.05);
        assert!((payoff - 1000.0).abs() < 1.0, "Payoff: {}", payoff);

        // Realized variance = 0.03 (lower than strike)
        let payoff = swap.payoff(0.03);
        assert!((payoff - (-1000.0)).abs() < 1.0, "Payoff: {}", payoff);
    }

    #[test]
    fn test_realized_variance() {
        // Generate sample daily returns (1% daily vol)
        let returns = vec![0.01, -0.01, 0.005, -0.005, 0.015, -0.01, 0.01, -0.015];

        let realized_var =
            VarianceSwap::realized_variance(&returns, 252.0).expect("Operation failed");

        // With these returns, annualized variance should be around 0.01-0.02
        assert!(
            realized_var > 0.01 && realized_var < 0.03,
            "Realized variance: {}",
            realized_var
        );
    }

    #[test]
    fn test_fair_variance_strike_black_scholes() {
        let chain = OptionChain::from_black_scholes(
            100.0, // spot
            1.0,   // maturity
            0.05,  // rate
            0.0,   // dividend
            0.2,   // volatility (implies variance = 0.04)
            (50.0, 150.0),
            50,
        )
        .expect("Operation failed");

        let fair_strike = chain.fair_variance_strike().expect("Operation failed");

        // Should be close to 0.04 (0.2²)
        assert!(
            (fair_strike - 0.04).abs() < 0.005,
            "Fair variance strike: {}, expected ~0.04",
            fair_strike
        );
    }

    #[test]
    fn test_volatility_swap_creation() {
        let swap = VolatilitySwap::new(0.2, 100000.0, 1.0).expect("Operation failed");

        assert_eq!(swap.strike, 0.2);
        assert_eq!(swap.notional, 100000.0);

        // Variance notional should be notional / (2 * strike)
        assert!((swap.variance_notional - 250000.0).abs() < 1.0);
    }

    #[test]
    fn test_volatility_swap_payoff() {
        let swap = VolatilitySwap::new(0.2, 100000.0, 1.0).expect("Operation failed");

        // Realized vol = 0.25
        let payoff = swap.payoff(0.25);
        assert!((payoff - 5000.0).abs() < 1.0, "Payoff: {}", payoff);

        // Realized vol = 0.15
        let payoff = swap.payoff(0.15);
        assert!((payoff - (-5000.0)).abs() < 1.0, "Payoff: {}", payoff);
    }

    #[test]
    fn test_volatility_from_variance_convexity() {
        let variance_strike = 0.04; // 20% vol
        let variance_of_variance = 0.001;

        let vol_strike =
            VolatilitySwap::fair_strike_from_variance(variance_strike, variance_of_variance)
                .expect("Operation failed");

        // Convexity adjustment formula: (Vvol / 8) * Var^(-3/2)
        // = 0.001 / 8 * 0.04^(-1.5) ≈ 0.0156
        // So vol_strike ≈ 0.2 - 0.0156 ≈ 0.184
        assert!(
            vol_strike < 0.2 && vol_strike > 0.18,
            "Vol strike: {}",
            vol_strike
        );
    }

    #[test]
    fn test_realized_volatility() {
        let returns = vec![0.01, -0.01, 0.005, -0.005, 0.015, -0.01];

        let realized_vol =
            VolatilitySwap::realized_volatility(&returns, 252.0).expect("Operation failed");

        // Should be positive and reasonable
        assert!(
            realized_vol > 0.1 && realized_vol < 0.3,
            "Realized vol: {}",
            realized_vol
        );
    }

    #[test]
    fn test_option_chain_consistency() {
        let chain = OptionChain::from_black_scholes(100.0, 1.0, 0.05, 0.0, 0.2, (80.0, 120.0), 20)
            .expect("Operation failed");

        // Check put-call parity at ATM
        let mid_idx = chain.strikes.len() / 2;
        let strike = chain.strikes[mid_idx];
        let call = chain.call_prices[mid_idx];
        let put = chain.put_prices[mid_idx];

        // Put-Call Parity: C - P = S - K*exp(-rT)
        let lhs = call - put;
        let rhs = chain.spot - strike * (-chain.risk_free_rate * chain.maturity).exp();

        assert!(
            (lhs - rhs).abs() < 1e-10,
            "Put-call parity violated: {} vs {}",
            lhs,
            rhs
        );
    }

    #[test]
    fn test_variance_swap_fair_value() {
        // Create option chain
        let chain = OptionChain::from_black_scholes(100.0, 1.0, 0.05, 0.0, 0.2, (70.0, 130.0), 30)
            .expect("Operation failed");

        let fair_strike = chain.fair_variance_strike().expect("Operation failed");

        // Create swap at fair strike
        let swap =
            VarianceSwap::new(fair_strike, 100000.0, 1.0, 252, 0.05).expect("Operation failed");

        // At inception with fair strike, swap value should be approximately zero
        // (we can't test this exactly without realized variance, but we verify the strike is reasonable)
        assert!(
            fair_strike > 0.035 && fair_strike < 0.045,
            "Fair strike should be near 0.04, got: {}",
            fair_strike
        );
    }
}
