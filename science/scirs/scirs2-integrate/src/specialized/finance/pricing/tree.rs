//! Tree-based methods for option pricing (Binomial and Trinomial)

use crate::error::IntegrateResult;
use crate::specialized::finance::models::VolatilityModel;
use crate::specialized::finance::solvers::StochasticPDESolver;
use crate::specialized::finance::types::{FinancialOption, OptionStyle, OptionType};

/// Tree-based pricing implementation (Cox-Ross-Rubinstein binomial tree)
pub fn price_tree(
    solver: &StochasticPDESolver,
    option: &FinancialOption,
    n_steps: usize,
) -> IntegrateResult<f64> {
    match &solver.volatility_model {
        VolatilityModel::Constant(sigma) => binomial_tree_black_scholes(option, *sigma, n_steps),
        _ => Err(crate::error::IntegrateError::ValueError(
            "Tree pricing currently only supports constant volatility (Black-Scholes)".to_string(),
        )),
    }
}

/// Cox-Ross-Rubinstein binomial tree for Black-Scholes model
fn binomial_tree_black_scholes(
    option: &FinancialOption,
    sigma: f64,
    n_steps: usize,
) -> IntegrateResult<f64> {
    let dt = option.maturity / n_steps as f64;
    let discount = (-option.risk_free_rate * dt).exp();

    // CRR parameters
    let u = (sigma * dt.sqrt()).exp(); // Up factor
    let d = 1.0 / u; // Down factor
    let q = ((option.risk_free_rate - option.dividend_yield) * dt).exp(); // Growth factor
    let p = (q - d) / (u - d); // Risk-neutral probability

    // Validate probability
    if !(0.0..=1.0).contains(&p) {
        return Err(crate::error::IntegrateError::ValueError(format!(
            "Invalid risk-neutral probability: {}. Check parameters.",
            p
        )));
    }

    // Build terminal node prices and values
    let mut values = vec![0.0; n_steps + 1];

    // Terminal stock prices at maturity
    for i in 0..=n_steps {
        let n_up = i;
        let n_down = n_steps - i;
        let s_t = option.spot * u.powi(n_up as i32) * d.powi(n_down as i32);

        // Terminal payoff
        values[i] = match option.option_type {
            OptionType::Call => (s_t - option.strike).max(0.0),
            OptionType::Put => (option.strike - s_t).max(0.0),
        };
    }

    // Backward induction
    for step in (0..n_steps).rev() {
        for i in 0..=step {
            let n_up = i;
            let n_down = step - i;
            let s = option.spot * u.powi(n_up as i32) * d.powi(n_down as i32);

            // Expected continuation value
            let continuation_value = discount * (p * values[i + 1] + (1.0 - p) * values[i]);

            // Intrinsic value (for early exercise)
            let intrinsic_value = match option.option_type {
                OptionType::Call => (s - option.strike).max(0.0),
                OptionType::Put => (option.strike - s).max(0.0),
            };

            // Apply early exercise for American options
            values[i] = match option.option_style {
                OptionStyle::American => continuation_value.max(intrinsic_value),
                OptionStyle::European => continuation_value,
                _ => {
                    return Err(crate::error::IntegrateError::ValueError(
                        "Tree pricing only supports European and American options".to_string(),
                    ))
                }
            };
        }
    }

    Ok(values[0])
}

/// Trinomial tree pricing (Boyle 1988)
#[allow(dead_code)]
fn trinomial_tree_black_scholes(
    option: &FinancialOption,
    sigma: f64,
    n_steps: usize,
) -> IntegrateResult<f64> {
    let dt = option.maturity / n_steps as f64;
    let discount = (-option.risk_free_rate * dt).exp();

    // Trinomial parameters
    let dx = sigma * (3.0 * dt).sqrt();
    let nu = option.risk_free_rate - option.dividend_yield - 0.5 * sigma * sigma;

    // Probabilities (Boyle method)
    let p_u = 0.5 * ((sigma * sigma * dt + nu * nu * dt * dt) / (dx * dx) + nu * dt / dx);
    let p_d = 0.5 * ((sigma * sigma * dt + nu * nu * dt * dt) / (dx * dx) - nu * dt / dx);
    let p_m = 1.0 - p_u - p_d;

    // Validate probabilities
    if p_u < 0.0 || p_d < 0.0 || p_m < 0.0 || p_u > 1.0 || p_d > 1.0 || p_m > 1.0 {
        return Err(crate::error::IntegrateError::ValueError(
            "Invalid trinomial probabilities. Consider increasing n_steps.".to_string(),
        ));
    }

    // Initialize value grid: each time step has (2*step + 1) nodes
    let max_nodes = 2 * n_steps + 1;
    let mut values = vec![0.0; max_nodes];

    // Terminal values at maturity
    for i in 0..max_nodes {
        let j = i as i32 - n_steps as i32; // Node index: -n_steps to +n_steps
        let s_t = option.spot * (j as f64 * dx).exp();

        values[i] = match option.option_type {
            OptionType::Call => (s_t - option.strike).max(0.0),
            OptionType::Put => (option.strike - s_t).max(0.0),
        };
    }

    // Temporary storage for next step values
    let mut next_values = vec![0.0; max_nodes];

    // Backward induction
    for step in (0..n_steps).rev() {
        let n_nodes = 2 * step + 1;

        for i in 0..n_nodes {
            let j = i as i32 - step as i32;
            let s = option.spot * (j as f64 * dx).exp();

            // Map to indices in value array
            let idx_center = (j + n_steps as i32) as usize;
            let idx_up = idx_center + 1;
            let idx_down = idx_center.saturating_sub(1);

            // Expected continuation value
            let continuation_value = discount
                * (p_u * values[idx_up] + p_m * values[idx_center] + p_d * values[idx_down]);

            // Intrinsic value
            let intrinsic_value = match option.option_type {
                OptionType::Call => (s - option.strike).max(0.0),
                OptionType::Put => (option.strike - s).max(0.0),
            };

            // Apply early exercise for American options
            next_values[idx_center] = match option.option_style {
                OptionStyle::American => continuation_value.max(intrinsic_value),
                OptionStyle::European => continuation_value,
                _ => {
                    return Err(crate::error::IntegrateError::ValueError(
                        "Trinomial tree only supports European and American options".to_string(),
                    ))
                }
            };
        }

        // Swap buffers
        std::mem::swap(&mut values, &mut next_values);
    }

    Ok(values[n_steps]) // Center node at t=0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialized::finance::models::VolatilityModel;
    use crate::specialized::finance::types::{FinanceMethod, OptionType};

    #[test]
    fn test_binomial_european_call() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        let solver = StochasticPDESolver::new(
            100,
            50,
            VolatilityModel::Constant(0.2),
            FinanceMethod::Tree { n_steps: 100 },
        );

        let price = price_tree(&solver, &option, 100).expect("Operation failed");

        // Black-Scholes reference: ~10.45
        assert!(price > 10.0 && price < 11.0, "Price: {}", price);
    }

    #[test]
    fn test_binomial_european_put() {
        let option = FinancialOption {
            option_type: OptionType::Put,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        let solver = StochasticPDESolver::new(
            100,
            50,
            VolatilityModel::Constant(0.2),
            FinanceMethod::Tree { n_steps: 100 },
        );

        let price = price_tree(&solver, &option, 100).expect("Operation failed");

        // Black-Scholes reference: ~5.57
        assert!(price > 5.2 && price < 6.0, "Price: {}", price);
    }

    #[test]
    fn test_binomial_american_put() {
        let option = FinancialOption {
            option_type: OptionType::Put,
            option_style: OptionStyle::American,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        let solver = StochasticPDESolver::new(
            100,
            50,
            VolatilityModel::Constant(0.2),
            FinanceMethod::Tree { n_steps: 100 },
        );

        let price = price_tree(&solver, &option, 100).expect("Operation failed");

        // American put should be more valuable than European (early exercise premium)
        assert!(price > 5.8 && price < 7.0, "Price: {}", price);
    }

    #[test]
    fn test_binomial_american_call_no_dividend() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::American,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        let solver = StochasticPDESolver::new(
            100,
            50,
            VolatilityModel::Constant(0.2),
            FinanceMethod::Tree { n_steps: 100 },
        );

        let price = price_tree(&solver, &option, 100).expect("Operation failed");

        // American call with no dividends should equal European call
        // (never optimal to exercise early)
        assert!(price > 10.0 && price < 11.0, "Price: {}", price);
    }

    #[test]
    fn test_trinomial_european_call() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        let price = trinomial_tree_black_scholes(&option, 0.2, 50).expect("Operation failed");

        // Should be close to binomial and Black-Scholes
        assert!(price > 9.5 && price < 11.5, "Price: {}", price);
    }
}
