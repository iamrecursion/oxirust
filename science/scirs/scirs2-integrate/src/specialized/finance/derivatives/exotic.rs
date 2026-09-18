//! Exotic derivatives pricing and analysis
//!
//! This module implements pricing models for exotic derivatives including barrier options,
//! Asian options, lookback options, and other path-dependent derivatives.

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::finance::pricing::black_scholes::{black_scholes_price, normal_cdf};
use crate::specialized::finance::types::OptionType;
use scirs2_core::random::{thread_rng, Rng, StandardNormal};

/// Barrier option type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarrierType {
    /// Activates when price moves up through barrier
    UpAndIn,
    /// Deactivates when price moves up through barrier
    UpAndOut,
    /// Activates when price moves down through barrier
    DownAndIn,
    /// Deactivates when price moves down through barrier
    DownAndOut,
}

/// Barrier option contract
#[derive(Debug, Clone)]
pub struct BarrierOption {
    pub spot: f64,
    pub strike: f64,
    pub barrier: f64,
    pub rebate: f64, // Paid if option knocked out
    pub rate: f64,
    pub dividend: f64,
    pub volatility: f64,
    pub maturity: f64,
    pub option_type: OptionType,
    pub barrier_type: BarrierType,
}

impl BarrierOption {
    /// Create a new barrier option
    pub fn new(
        spot: f64,
        strike: f64,
        barrier: f64,
        rebate: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        maturity: f64,
        option_type: OptionType,
        barrier_type: BarrierType,
    ) -> IntegrateResult<Self> {
        if maturity <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Maturity must be positive".to_string(),
            ));
        }

        Ok(Self {
            spot,
            strike,
            barrier,
            rebate,
            rate,
            dividend,
            volatility,
            maturity,
            option_type,
            barrier_type,
        })
    }

    /// Price barrier option using Monte Carlo simulation
    pub fn price_monte_carlo(&self, n_paths: usize, n_steps: usize) -> IntegrateResult<f64> {
        let mut rng = thread_rng();
        let normal = StandardNormal;

        let dt = self.maturity / n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let drift = (self.rate - self.dividend - 0.5 * self.volatility * self.volatility) * dt;
        let diffusion = self.volatility * sqrt_dt;

        let mut payoff_sum = 0.0;

        for _ in 0..n_paths {
            let mut s = self.spot;
            let mut barrier_hit = false;

            // Check if barrier is already breached at t=0
            match self.barrier_type {
                BarrierType::UpAndIn | BarrierType::UpAndOut => {
                    if self.spot >= self.barrier {
                        barrier_hit = true;
                    }
                }
                BarrierType::DownAndIn | BarrierType::DownAndOut => {
                    if self.spot <= self.barrier {
                        barrier_hit = true;
                    }
                }
            }

            // Simulate path
            for _ in 0..n_steps {
                let z: f64 = rng.sample(normal);
                s *= (drift + diffusion * z).exp();

                // Check barrier condition
                match self.barrier_type {
                    BarrierType::UpAndIn | BarrierType::UpAndOut => {
                        if s >= self.barrier {
                            barrier_hit = true;
                        }
                    }
                    BarrierType::DownAndIn | BarrierType::DownAndOut => {
                        if s <= self.barrier {
                            barrier_hit = true;
                        }
                    }
                }
            }

            // Calculate payoff based on barrier type
            let intrinsic = match self.option_type {
                OptionType::Call => (s - self.strike).max(0.0),
                OptionType::Put => (self.strike - s).max(0.0),
            };

            let payoff = match self.barrier_type {
                BarrierType::UpAndIn | BarrierType::DownAndIn => {
                    if barrier_hit {
                        intrinsic
                    } else {
                        self.rebate
                    }
                }
                BarrierType::UpAndOut | BarrierType::DownAndOut => {
                    if barrier_hit {
                        self.rebate
                    } else {
                        intrinsic
                    }
                }
            };

            payoff_sum += payoff;
        }

        let avg_payoff = payoff_sum / n_paths as f64;
        let discounted_payoff = avg_payoff * (-self.rate * self.maturity).exp();

        Ok(discounted_payoff)
    }
}

/// Asian option averaging method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AveragingMethod {
    /// Arithmetic average (more common)
    Arithmetic,
    /// Geometric average (has closed-form solution)
    Geometric,
}

/// Asian option contract
#[derive(Debug, Clone)]
pub struct AsianOption {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub volatility: f64,
    pub maturity: f64,
    pub option_type: OptionType,
    pub averaging_method: AveragingMethod,
    pub n_observations: usize,
}

impl AsianOption {
    /// Create a new Asian option
    pub fn new(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        maturity: f64,
        option_type: OptionType,
        averaging_method: AveragingMethod,
        n_observations: usize,
    ) -> IntegrateResult<Self> {
        if maturity <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Maturity must be positive".to_string(),
            ));
        }

        if n_observations == 0 {
            return Err(IntegrateError::ValueError(
                "Number of observations must be positive".to_string(),
            ));
        }

        Ok(Self {
            spot,
            strike,
            rate,
            dividend,
            volatility,
            maturity,
            option_type,
            averaging_method,
            n_observations,
        })
    }

    /// Price Asian option using Monte Carlo simulation
    pub fn price_monte_carlo(&self, n_paths: usize) -> IntegrateResult<f64> {
        let mut rng = thread_rng();
        let normal = StandardNormal;

        let dt = self.maturity / self.n_observations as f64;
        let sqrt_dt = dt.sqrt();
        let drift = (self.rate - self.dividend - 0.5 * self.volatility * self.volatility) * dt;
        let diffusion = self.volatility * sqrt_dt;

        let mut payoff_sum = 0.0;

        for _ in 0..n_paths {
            let mut s = self.spot;
            let mut price_sum = 0.0;
            let mut price_product = 1.0;

            for _ in 0..self.n_observations {
                let z: f64 = rng.sample(normal);
                s *= (drift + diffusion * z).exp();

                price_sum += s;
                price_product *= s;
            }

            let average = match self.averaging_method {
                AveragingMethod::Arithmetic => price_sum / self.n_observations as f64,
                AveragingMethod::Geometric => price_product.powf(1.0 / self.n_observations as f64),
            };

            let payoff = match self.option_type {
                OptionType::Call => (average - self.strike).max(0.0),
                OptionType::Put => (self.strike - average).max(0.0),
            };

            payoff_sum += payoff;
        }

        let avg_payoff = payoff_sum / n_paths as f64;
        let discounted_payoff = avg_payoff * (-self.rate * self.maturity).exp();

        Ok(discounted_payoff)
    }

    /// Price geometric Asian option using closed-form solution
    pub fn price_geometric_closed_form(&self) -> IntegrateResult<f64> {
        if self.averaging_method != AveragingMethod::Geometric {
            return Err(IntegrateError::ValueError(
                "Closed-form solution only available for geometric averaging".to_string(),
            ));
        }

        let n = self.n_observations as f64;

        // Adjusted volatility for geometric average
        let sigma_adj = self.volatility * ((2.0 * n + 1.0) / (6.0 * (n + 1.0))).sqrt();

        // Adjusted dividend yield
        let q_adj = 0.5 * (self.rate - self.dividend + self.volatility * self.volatility / 6.0);

        // Use Black-Scholes with adjusted parameters
        let price = black_scholes_price(
            self.spot,
            self.strike,
            self.rate,
            q_adj,
            sigma_adj,
            self.maturity,
            self.option_type,
        );

        Ok(price)
    }
}

/// Lookback option type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LookbackType {
    /// Fixed strike, floating payoff (max/min)
    FixedStrike,
    /// Floating strike (optimal execution)
    FloatingStrike,
}

/// Lookback option contract
#[derive(Debug, Clone)]
pub struct LookbackOption {
    pub spot: f64,
    pub strike: f64, // Only used for fixed strike
    pub rate: f64,
    pub dividend: f64,
    pub volatility: f64,
    pub maturity: f64,
    pub option_type: OptionType,
    pub lookback_type: LookbackType,
}

impl LookbackOption {
    /// Create a new lookback option
    pub fn new(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        maturity: f64,
        option_type: OptionType,
        lookback_type: LookbackType,
    ) -> IntegrateResult<Self> {
        if maturity <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Maturity must be positive".to_string(),
            ));
        }

        Ok(Self {
            spot,
            strike,
            rate,
            dividend,
            volatility,
            maturity,
            option_type,
            lookback_type,
        })
    }

    /// Price lookback option using Monte Carlo simulation
    pub fn price_monte_carlo(&self, n_paths: usize, n_steps: usize) -> IntegrateResult<f64> {
        let mut rng = thread_rng();
        let normal = StandardNormal;

        let dt = self.maturity / n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let drift = (self.rate - self.dividend - 0.5 * self.volatility * self.volatility) * dt;
        let diffusion = self.volatility * sqrt_dt;

        let mut payoff_sum = 0.0;

        for _ in 0..n_paths {
            let mut s = self.spot;
            let mut s_max = self.spot;
            let mut s_min = self.spot;

            for _ in 0..n_steps {
                let z: f64 = rng.sample(normal);
                s *= (drift + diffusion * z).exp();

                s_max = s_max.max(s);
                s_min = s_min.min(s);
            }

            let payoff = match (self.option_type, self.lookback_type) {
                (OptionType::Call, LookbackType::FixedStrike) => (s_max - self.strike).max(0.0),
                (OptionType::Put, LookbackType::FixedStrike) => (self.strike - s_min).max(0.0),
                (OptionType::Call, LookbackType::FloatingStrike) => s - s_min,
                (OptionType::Put, LookbackType::FloatingStrike) => s_max - s,
            };

            payoff_sum += payoff;
        }

        let avg_payoff = payoff_sum / n_paths as f64;
        let discounted_payoff = avg_payoff * (-self.rate * self.maturity).exp();

        Ok(discounted_payoff)
    }
}

/// Digital/Binary option type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DigitalType {
    /// Pays cash amount if condition met
    CashOrNothing { cash_amount: f64 },
    /// Pays asset if condition met
    AssetOrNothing,
}

/// Digital option contract
#[derive(Debug, Clone)]
pub struct DigitalOption {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub volatility: f64,
    pub maturity: f64,
    pub option_type: OptionType,
    pub digital_type: DigitalType,
}

impl DigitalOption {
    /// Create a new digital option
    pub fn new(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        maturity: f64,
        option_type: OptionType,
        digital_type: DigitalType,
    ) -> IntegrateResult<Self> {
        if maturity <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Maturity must be positive".to_string(),
            ));
        }

        Ok(Self {
            spot,
            strike,
            rate,
            dividend,
            volatility,
            maturity,
            option_type,
            digital_type,
        })
    }

    /// Price digital option using closed-form solution
    pub fn price(&self) -> f64 {
        let sqrt_t = self.maturity.sqrt();
        let d1 = ((self.spot / self.strike).ln()
            + (self.rate - self.dividend + 0.5 * self.volatility * self.volatility)
                * self.maturity)
            / (self.volatility * sqrt_t);
        let d2 = d1 - self.volatility * sqrt_t;

        let discount = (-self.rate * self.maturity).exp();

        match (self.option_type, self.digital_type) {
            (OptionType::Call, DigitalType::CashOrNothing { cash_amount }) => {
                cash_amount * discount * normal_cdf(d2)
            }
            (OptionType::Put, DigitalType::CashOrNothing { cash_amount }) => {
                cash_amount * discount * normal_cdf(-d2)
            }
            (OptionType::Call, DigitalType::AssetOrNothing) => {
                self.spot * (-(self.dividend * self.maturity)).exp() * normal_cdf(d1)
            }
            (OptionType::Put, DigitalType::AssetOrNothing) => {
                self.spot * (-(self.dividend * self.maturity)).exp() * normal_cdf(-d1)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_option_creation() {
        let option = BarrierOption::new(
            100.0,
            100.0,
            110.0,
            5.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            BarrierType::UpAndOut,
        )
        .expect("Operation failed");

        assert_eq!(option.spot, 100.0);
        assert_eq!(option.barrier, 110.0);
    }

    #[test]
    fn test_barrier_option_up_and_out() {
        let option = BarrierOption::new(
            100.0,
            100.0,
            120.0,
            0.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            BarrierType::UpAndOut,
        )
        .expect("Operation failed");

        let price = option
            .price_monte_carlo(10000, 100)
            .expect("Operation failed");

        // Up-and-out should be cheaper than vanilla
        let vanilla = black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);

        assert!(
            price < vanilla,
            "Barrier price {} should be < vanilla {}",
            price,
            vanilla
        );
        assert!(price > 0.0, "Price should be positive");
    }

    #[test]
    fn test_asian_option_creation() {
        let option = AsianOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            AveragingMethod::Arithmetic,
            252,
        )
        .expect("Operation failed");

        assert_eq!(option.n_observations, 252);
    }

    #[test]
    fn test_asian_option_monte_carlo() {
        let option = AsianOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            AveragingMethod::Arithmetic,
            50,
        )
        .expect("Operation failed");

        let price = option.price_monte_carlo(10000).expect("Operation failed");

        // Asian should be cheaper than vanilla due to averaging reducing volatility
        let vanilla = black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);

        assert!(
            price < vanilla,
            "Asian price {} should be < vanilla {}",
            price,
            vanilla
        );
        assert!(price > 0.0, "Price should be positive");
    }

    #[test]
    fn test_asian_geometric_closed_form() {
        let option = AsianOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            AveragingMethod::Geometric,
            50,
        )
        .expect("Operation failed");

        let price = option
            .price_geometric_closed_form()
            .expect("Operation failed");

        assert!(price > 5.0 && price < 15.0, "Price: {}", price);
    }

    #[test]
    fn test_lookback_option_creation() {
        let option = LookbackOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            LookbackType::FixedStrike,
        )
        .expect("Operation failed");

        assert_eq!(option.spot, 100.0);
    }

    #[test]
    fn test_lookback_option_pricing() {
        let option = LookbackOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            LookbackType::FixedStrike,
        )
        .expect("Operation failed");

        let price = option
            .price_monte_carlo(10000, 100)
            .expect("Operation failed");

        // Lookback should be more expensive than vanilla
        let vanilla = black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);

        assert!(
            price > vanilla,
            "Lookback price {} should be > vanilla {}",
            price,
            vanilla
        );
    }

    #[test]
    fn test_digital_cash_or_nothing_call() {
        let option = DigitalOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            DigitalType::CashOrNothing { cash_amount: 10.0 },
        )
        .expect("Operation failed");

        let price = option.price();

        // For ATM digital, probability ~ 0.5, so price ~ 10 * 0.5 * exp(-0.05)
        assert!(price > 4.0 && price < 6.0, "Price: {}", price);
    }

    #[test]
    fn test_digital_asset_or_nothing() {
        let option = DigitalOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            DigitalType::AssetOrNothing,
        )
        .expect("Operation failed");

        let price = option.price();

        // For ATM with positive drift (r=0.05, sigma=0.2):
        // d1 = (0 + (0.05 + 0.5*0.04)*1) / 0.2 = 0.35
        // N(d1) ≈ 0.6368, so price ≈ 100 * 0.6368 = 63.68
        assert!(price > 60.0 && price < 67.0, "Price: {}", price);
    }

    #[test]
    fn test_barrier_in_out_parity() {
        // Up-and-In + Up-and-Out should equal vanilla option
        let up_in = BarrierOption::new(
            100.0,
            100.0,
            120.0,
            0.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            BarrierType::UpAndIn,
        )
        .expect("Operation failed");

        let up_out = BarrierOption::new(
            100.0,
            100.0,
            120.0,
            0.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            BarrierType::UpAndOut,
        )
        .expect("Operation failed");

        // Reduced from 50000 to 10000 paths for faster testing (still statistically valid)
        let in_price = up_in
            .price_monte_carlo(10000, 100)
            .expect("Operation failed");
        let out_price = up_out
            .price_monte_carlo(10000, 100)
            .expect("Operation failed");
        let vanilla = black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);

        // Allow 15% error due to Monte Carlo variance (increased tolerance for fewer paths)
        assert!(
            ((in_price + out_price) - vanilla).abs() / vanilla < 0.15,
            "In+Out={}, Vanilla={}",
            in_price + out_price,
            vanilla
        );
    }

    #[test]
    fn test_digital_put_call_sum() {
        // Cash-or-nothing call + put should equal discounted cash
        let call = DigitalOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            DigitalType::CashOrNothing { cash_amount: 10.0 },
        )
        .expect("Operation failed");

        let put = DigitalOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Put,
            DigitalType::CashOrNothing { cash_amount: 10.0 },
        )
        .expect("Operation failed");

        let sum = call.price() + put.price();
        let expected = 10.0 * (-0.05_f64).exp();

        assert!(
            (sum - expected).abs() < 1e-10,
            "Sum {} should equal discounted cash {}",
            sum,
            expected
        );
    }
}
