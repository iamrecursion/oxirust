//! Structured products pricing and risk management
//!
//! This module implements pricing and analysis tools for structured products including
//! autocallable notes, principal-protected notes, and multi-asset derivatives.

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::finance::pricing::black_scholes::black_scholes_price;
use crate::specialized::finance::types::OptionType;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Rng, StandardNormal};

/// Autocallable note with barrier monitoring
#[derive(Debug, Clone)]
pub struct AutocallableNote {
    /// Initial spot prices for each underlying
    pub spots: Vec<f64>,
    /// Autocall barrier levels (as % of initial spot)
    pub autocall_barriers: Vec<f64>,
    /// Observation dates (in years)
    pub observation_dates: Vec<f64>,
    /// Coupon rates at each observation
    pub coupon_rates: Vec<f64>,
    /// Downside barrier (knock-in level)
    pub knock_in_barrier: f64,
    /// Principal amount
    pub principal: f64,
    /// Risk-free rate
    pub rate: f64,
    /// Volatilities for each underlying
    pub volatilities: Vec<f64>,
    /// Correlation matrix
    pub correlation: Array2<f64>,
}

impl AutocallableNote {
    /// Create a new autocallable note
    pub fn new(
        spots: Vec<f64>,
        autocall_barriers: Vec<f64>,
        observation_dates: Vec<f64>,
        coupon_rates: Vec<f64>,
        knock_in_barrier: f64,
        principal: f64,
        rate: f64,
        volatilities: Vec<f64>,
        correlation: Array2<f64>,
    ) -> IntegrateResult<Self> {
        if spots.len() != volatilities.len() {
            return Err(IntegrateError::ValueError(
                "Number of spots and volatilities must match".to_string(),
            ));
        }

        if observation_dates.len() != coupon_rates.len() {
            return Err(IntegrateError::ValueError(
                "Number of observations and coupons must match".to_string(),
            ));
        }

        let n = spots.len();
        if correlation.nrows() != n || correlation.ncols() != n {
            return Err(IntegrateError::ValueError(
                "Correlation matrix dimensions must match number of underlyings".to_string(),
            ));
        }

        Ok(Self {
            spots,
            autocall_barriers,
            observation_dates,
            coupon_rates,
            knock_in_barrier,
            principal,
            rate,
            volatilities,
            correlation,
        })
    }

    /// Price autocallable note using Monte Carlo simulation
    pub fn price_monte_carlo(&self, n_paths: usize) -> IntegrateResult<f64> {
        let n_assets = self.spots.len();
        let n_obs = self.observation_dates.len();

        // Cholesky decomposition for correlation
        let chol = cholesky_decomposition(&self.correlation)?;

        let mut rng = thread_rng();
        let normal = StandardNormal;

        let mut total_payoff = 0.0;

        for _ in 0..n_paths {
            let mut prices = self.spots.clone();
            let mut autocalled = false;
            let mut knock_in_hit = false;
            let mut payoff = 0.0;
            let mut prev_time = 0.0;

            for (obs_idx, &obs_time) in self.observation_dates.iter().enumerate() {
                let dt = obs_time - prev_time;

                // Generate correlated random numbers
                let mut z = vec![0.0; n_assets];
                for i in 0..n_assets {
                    z[i] = rng.sample(normal);
                }

                let corr_z = apply_cholesky(&chol, &z);

                // Update prices
                for i in 0..n_assets {
                    let drift = (self.rate - 0.5 * self.volatilities[i].powi(2)) * dt;
                    let diffusion = self.volatilities[i] * dt.sqrt() * corr_z[i];
                    prices[i] *= (drift + diffusion).exp();

                    // Check knock-in barrier
                    if prices[i] / self.spots[i] <= self.knock_in_barrier {
                        knock_in_hit = true;
                    }
                }

                // Check autocall condition (all assets above barrier)
                let all_above_barrier = prices
                    .iter()
                    .zip(&self.spots)
                    .all(|(&p, &s)| p / s >= self.autocall_barriers[obs_idx]);

                if all_above_barrier {
                    // Autocall triggered
                    let cumulative_coupons: f64 = self.coupon_rates[..=obs_idx].iter().sum();
                    payoff = self.principal * (1.0 + cumulative_coupons);
                    payoff *= (-self.rate * obs_time).exp();
                    autocalled = true;
                    break;
                }

                prev_time = obs_time;
            }

            // If not autocalled, calculate maturity payoff
            if !autocalled {
                let final_time = self.observation_dates.last().expect("Operation failed");

                if knock_in_hit {
                    // Worst-of performance
                    let worst_performance = prices
                        .iter()
                        .zip(&self.spots)
                        .map(|(&p, &s)| p / s)
                        .fold(f64::INFINITY, f64::min);

                    payoff = self.principal * worst_performance;
                } else {
                    // Return principal plus all coupons
                    let total_coupons: f64 = self.coupon_rates.iter().sum();
                    payoff = self.principal * (1.0 + total_coupons);
                }

                payoff *= (-self.rate * final_time).exp();
            }

            total_payoff += payoff;
        }

        Ok(total_payoff / n_paths as f64)
    }
}

/// Principal-protected note with participation in upside
#[derive(Debug, Clone)]
pub struct PrincipalProtectedNote {
    /// Principal amount (guaranteed)
    pub principal: f64,
    /// Underlying spot price
    pub spot: f64,
    /// Strike price
    pub strike: f64,
    /// Time to maturity
    pub maturity: f64,
    /// Risk-free rate
    pub rate: f64,
    /// Volatility
    pub volatility: f64,
    /// Participation rate in upside
    pub participation_rate: f64,
    /// Cap on returns (optional)
    pub cap: Option<f64>,
}

impl PrincipalProtectedNote {
    /// Create a new principal-protected note
    pub fn new(
        principal: f64,
        spot: f64,
        strike: f64,
        maturity: f64,
        rate: f64,
        volatility: f64,
        participation_rate: f64,
        cap: Option<f64>,
    ) -> IntegrateResult<Self> {
        if principal <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Principal must be positive".to_string(),
            ));
        }

        if participation_rate < 0.0 {
            return Err(IntegrateError::ValueError(
                "Participation rate must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            principal,
            spot,
            strike,
            maturity,
            rate,
            volatility,
            participation_rate,
            cap,
        })
    }

    /// Price the principal-protected note
    pub fn price(&self) -> f64 {
        // Zero-coupon bond for principal protection
        let bond_value = self.principal * (-self.rate * self.maturity).exp();

        // Call option for upside participation
        let call_value = black_scholes_price(
            self.spot,
            self.strike,
            self.rate,
            0.0,
            self.volatility,
            self.maturity,
            OptionType::Call,
        );

        let upside_value = if let Some(cap_level) = self.cap {
            // If capped, subtract value of call at cap level
            let cap_call = black_scholes_price(
                self.spot,
                cap_level,
                self.rate,
                0.0,
                self.volatility,
                self.maturity,
                OptionType::Call,
            );
            self.participation_rate * (call_value - cap_call)
        } else {
            self.participation_rate * call_value
        };

        bond_value + upside_value
    }

    /// Calculate fair participation rate given note price
    pub fn fair_participation_rate(&self, target_price: f64) -> IntegrateResult<f64> {
        let bond_value = self.principal * (-self.rate * self.maturity).exp();

        if target_price < bond_value {
            return Err(IntegrateError::ValueError(
                "Target price must be at least the bond value".to_string(),
            ));
        }

        let call_value = black_scholes_price(
            self.spot,
            self.strike,
            self.rate,
            0.0,
            self.volatility,
            self.maturity,
            OptionType::Call,
        );

        let available_for_upside = target_price - bond_value;

        if call_value < 1e-10 {
            return Err(IntegrateError::ValueError(
                "Call option has negligible value".to_string(),
            ));
        }

        Ok(available_for_upside / call_value)
    }
}

/// Basket option on multiple underlyings
#[derive(Debug, Clone)]
pub struct BasketOption {
    /// Spot prices of underlyings
    pub spots: Vec<f64>,
    /// Weights for basket composition
    pub weights: Vec<f64>,
    /// Strike price
    pub strike: f64,
    /// Time to maturity
    pub maturity: f64,
    /// Risk-free rate
    pub rate: f64,
    /// Volatilities
    pub volatilities: Vec<f64>,
    /// Correlation matrix
    pub correlation: Array2<f64>,
    /// Option type
    pub option_type: OptionType,
}

impl BasketOption {
    /// Create a new basket option
    pub fn new(
        spots: Vec<f64>,
        weights: Vec<f64>,
        strike: f64,
        maturity: f64,
        rate: f64,
        volatilities: Vec<f64>,
        correlation: Array2<f64>,
        option_type: OptionType,
    ) -> IntegrateResult<Self> {
        let n = spots.len();

        if weights.len() != n || volatilities.len() != n {
            return Err(IntegrateError::ValueError(
                "Spots, weights, and volatilities must have same length".to_string(),
            ));
        }

        if correlation.nrows() != n || correlation.ncols() != n {
            return Err(IntegrateError::ValueError(
                "Correlation matrix dimensions must match number of assets".to_string(),
            ));
        }

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        Ok(Self {
            spots,
            weights: normalized_weights,
            strike,
            maturity,
            rate,
            volatilities,
            correlation,
            option_type,
        })
    }

    /// Price basket option using Monte Carlo
    pub fn price_monte_carlo(&self, n_paths: usize) -> IntegrateResult<f64> {
        let chol = cholesky_decomposition(&self.correlation)?;

        let mut rng = thread_rng();
        let normal = StandardNormal;

        let mut payoff_sum = 0.0;

        for _ in 0..n_paths {
            // Generate correlated random numbers
            let mut z = vec![0.0; self.spots.len()];
            for i in 0..self.spots.len() {
                z[i] = rng.sample(normal);
            }

            let corr_z = apply_cholesky(&chol, &z);

            // Compute terminal basket value
            let mut basket_value = 0.0;
            for i in 0..self.spots.len() {
                let drift = (self.rate - 0.5 * self.volatilities[i].powi(2)) * self.maturity;
                let diffusion = self.volatilities[i] * self.maturity.sqrt() * corr_z[i];
                let terminal_price = self.spots[i] * (drift + diffusion).exp();
                basket_value += self.weights[i] * terminal_price;
            }

            // Calculate payoff
            let payoff = match self.option_type {
                OptionType::Call => (basket_value - self.strike).max(0.0),
                OptionType::Put => (self.strike - basket_value).max(0.0),
            };

            payoff_sum += payoff;
        }

        let avg_payoff = payoff_sum / n_paths as f64;
        Ok(avg_payoff * (-self.rate * self.maturity).exp())
    }
}

/// Range accrual note - accrues interest when underlying stays in range
#[derive(Debug, Clone)]
pub struct RangeAccrualNote {
    /// Principal amount
    pub principal: f64,
    /// Underlying spot price
    pub spot: f64,
    /// Lower barrier
    pub lower_barrier: f64,
    /// Upper barrier
    pub upper_barrier: f64,
    /// Accrual rate per day (annualized)
    pub accrual_rate: f64,
    /// Time to maturity
    pub maturity: f64,
    /// Risk-free rate
    pub rate: f64,
    /// Volatility
    pub volatility: f64,
    /// Number of observation days
    pub n_observations: usize,
}

impl RangeAccrualNote {
    /// Create a new range accrual note
    pub fn new(
        principal: f64,
        spot: f64,
        lower_barrier: f64,
        upper_barrier: f64,
        accrual_rate: f64,
        maturity: f64,
        rate: f64,
        volatility: f64,
        n_observations: usize,
    ) -> IntegrateResult<Self> {
        if lower_barrier >= upper_barrier {
            return Err(IntegrateError::ValueError(
                "Lower barrier must be less than upper barrier".to_string(),
            ));
        }

        if n_observations == 0 {
            return Err(IntegrateError::ValueError(
                "Number of observations must be positive".to_string(),
            ));
        }

        Ok(Self {
            principal,
            spot,
            lower_barrier,
            upper_barrier,
            accrual_rate,
            maturity,
            rate,
            volatility,
            n_observations,
        })
    }

    /// Price range accrual note using Monte Carlo
    pub fn price_monte_carlo(&self, n_paths: usize) -> IntegrateResult<f64> {
        let mut rng = thread_rng();
        let normal = StandardNormal;

        let dt = self.maturity / self.n_observations as f64;
        let sqrt_dt = dt.sqrt();
        let drift = (self.rate - 0.5 * self.volatility.powi(2)) * dt;
        let diffusion = self.volatility * sqrt_dt;

        let mut total_payoff = 0.0;

        for _ in 0..n_paths {
            let mut s = self.spot;
            let mut days_in_range = 0;

            for _ in 0..self.n_observations {
                let z: f64 = rng.sample(normal);
                s *= (drift + diffusion * z).exp();

                if s >= self.lower_barrier && s <= self.upper_barrier {
                    days_in_range += 1;
                }
            }

            let accrual_fraction = days_in_range as f64 / self.n_observations as f64;
            let interest = self.principal * self.accrual_rate * self.maturity * accrual_fraction;
            let payoff = self.principal + interest;

            total_payoff += payoff;
        }

        let avg_payoff = total_payoff / n_paths as f64;
        Ok(avg_payoff * (-self.rate * self.maturity).exp())
    }
}

/// Cholesky decomposition for correlation matrix
fn cholesky_decomposition(corr: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
    let n = corr.nrows();
    let mut l = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }

            if i == j {
                let diag = corr[[i, i]] - sum;
                if diag < 0.0 {
                    return Err(IntegrateError::ValueError(
                        "Correlation matrix is not positive definite".to_string(),
                    ));
                }
                l[[i, j]] = diag.sqrt();
            } else {
                if l[[j, j]].abs() < 1e-10 {
                    return Err(IntegrateError::ValueError(
                        "Cholesky decomposition failed: zero diagonal".to_string(),
                    ));
                }
                l[[i, j]] = (corr[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Apply Cholesky matrix to generate correlated random numbers
fn apply_cholesky(chol: &Array2<f64>, z: &[f64]) -> Vec<f64> {
    let n = chol.nrows();
    let mut result = vec![0.0; n];

    for i in 0..n {
        for j in 0..=i {
            result[i] += chol[[i, j]] * z[j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_autocallable_creation() {
        let corr = arr2(&[[1.0, 0.5], [0.5, 1.0]]);

        let note = AutocallableNote::new(
            vec![100.0, 100.0],
            vec![1.0, 0.95, 0.90],
            vec![0.5, 1.0, 1.5],
            vec![0.05, 0.05, 0.05],
            0.7,
            100.0,
            0.03,
            vec![0.2, 0.25],
            corr,
        )
        .expect("Operation failed");

        assert_eq!(note.spots.len(), 2);
        assert_eq!(note.observation_dates.len(), 3);
    }

    #[test]
    fn test_autocallable_pricing() {
        let corr = arr2(&[[1.0, 0.6], [0.6, 1.0]]);

        let note = AutocallableNote::new(
            vec![100.0, 100.0],
            vec![1.05, 1.05, 1.05],
            vec![0.5, 1.0, 1.5],
            vec![0.06, 0.06, 0.06],
            0.65,
            100.0,
            0.03,
            vec![0.2, 0.2],
            corr,
        )
        .expect("Operation failed");

        let price = note.price_monte_carlo(10000).expect("Operation failed");

        // Should be close to principal plus some coupon value
        assert!(price > 95.0 && price < 110.0, "Price: {}", price);
    }

    #[test]
    fn test_principal_protected_note() {
        let ppn = PrincipalProtectedNote::new(100.0, 100.0, 100.0, 1.0, 0.05, 0.2, 0.8, None)
            .expect("Operation failed");

        let price = ppn.price();

        // Should be at least the discounted principal
        let bond_value = 100.0 * (-0.05_f64).exp();
        assert!(
            price >= bond_value,
            "Price {} < bond value {}",
            price,
            bond_value
        );
        assert!(price < 110.0, "Price too high: {}", price);
    }

    #[test]
    fn test_ppn_fair_participation() {
        let ppn = PrincipalProtectedNote::new(100.0, 100.0, 100.0, 1.0, 0.05, 0.2, 0.0, None)
            .expect("Operation failed");

        let target_price = 100.0;
        let fair_part = ppn
            .fair_participation_rate(target_price)
            .expect("Operation failed");

        assert!(
            fair_part > 0.0 && fair_part < 2.0,
            "Fair participation: {}",
            fair_part
        );
    }

    #[test]
    fn test_basket_option_creation() {
        let corr = arr2(&[[1.0, 0.3], [0.3, 1.0]]);

        let basket = BasketOption::new(
            vec![100.0, 100.0],
            vec![0.5, 0.5],
            100.0,
            1.0,
            0.05,
            vec![0.2, 0.25],
            corr,
            OptionType::Call,
        )
        .expect("Operation failed");

        assert_eq!(basket.spots.len(), 2);
        assert!((basket.weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_basket_option_pricing() {
        let corr = arr2(&[[1.0, 0.5], [0.5, 1.0]]);

        let basket = BasketOption::new(
            vec![100.0, 100.0],
            vec![0.5, 0.5],
            100.0,
            1.0,
            0.05,
            vec![0.2, 0.2],
            corr,
            OptionType::Call,
        )
        .expect("Operation failed");

        let price = basket.price_monte_carlo(10000).expect("Operation failed");

        assert!(price > 5.0 && price < 15.0, "Price: {}", price);
    }

    #[test]
    fn test_range_accrual_note() {
        let note = RangeAccrualNote::new(100.0, 100.0, 90.0, 110.0, 0.05, 1.0, 0.03, 0.15, 252)
            .expect("Operation failed");

        let price = note.price_monte_carlo(5000).expect("Operation failed");

        // With low volatility and reasonable range, should get most of interest
        let min_price = 100.0 * (-0.03_f64).exp(); // Just principal discounted
        let max_price = (100.0 + 100.0 * 0.05) * (-0.03_f64).exp(); // Principal + full interest

        assert!(price > min_price && price < max_price, "Price: {}", price);
    }

    #[test]
    fn test_cholesky_decomposition() {
        let corr = arr2(&[[1.0, 0.5], [0.5, 1.0]]);

        let chol = cholesky_decomposition(&corr).expect("Operation failed");

        // Verify L * L^T = corr
        let mut reconstructed = Array2::<f64>::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    reconstructed[[i, j]] += chol[[i, k]] * chol[[j, k]];
                }
            }
        }

        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - corr[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_apply_cholesky() {
        let chol = arr2(&[[1.0, 0.0], [0.5, 0.866025]]);
        let z = vec![1.0, 1.0];

        let result = apply_cholesky(&chol, &z);

        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 1.366025).abs() < 1e-5);
    }

    #[test]
    fn test_ppn_with_cap() {
        let ppn =
            PrincipalProtectedNote::new(100.0, 100.0, 100.0, 1.0, 0.05, 0.2, 0.8, Some(120.0))
                .expect("Operation failed");

        let price_capped = ppn.price();

        let ppn_uncapped =
            PrincipalProtectedNote::new(100.0, 100.0, 100.0, 1.0, 0.05, 0.2, 0.8, None)
                .expect("Operation failed");

        let price_uncapped = ppn_uncapped.price();

        // Capped should be cheaper
        assert!(
            price_capped < price_uncapped,
            "Capped {} should be < uncapped {}",
            price_capped,
            price_uncapped
        );
    }
}
