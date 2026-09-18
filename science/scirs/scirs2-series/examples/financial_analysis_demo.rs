//! Financial Time Series Analysis - Minimal Working Demo
//!
//! This example demonstrates the current function-based API for financial analysis.
//! Note: Full examples will be updated in v0.1.0 after API stabilization.

use scirs2_core::ndarray::{compat::ArrayStatCompat, Array1};
use scirs2_series::financial::{
    models::{Distribution, GarchConfig, GarchModel, MeanModel},
    risk::{expected_shortfall, max_drawdown, sharpe_ratio, var_historical},
    technical_indicators::{ema, rsi, sma},
    volatility::{ewma_volatility, realized_volatility},
};
use statrs::statistics::Statistics;

fn main() {
    println!("=== Financial Time Series Analysis Demo ===\n");

    // Generate synthetic financial data
    let (prices, returns) = generate_financial_data();
    println!("Generated {} price observations\n", prices.len());

    // Demo 1: Technical Indicators
    println!("1. Technical Indicators");
    technical_indicators_demo(&prices);

    // Demo 2: Volatility Estimation
    println!("\n2. Volatility Estimation");
    volatility_demo(&returns);

    // Demo 3: GARCH Modeling
    println!("\n3. GARCH Volatility Modeling");
    garch_demo(&returns);

    // Demo 4: Risk Metrics
    println!("\n4. Risk Metrics");
    risk_metrics_demo(&prices, &returns);

    println!("\n=== Financial Analysis Complete ===");
}

fn generate_financial_data() -> (Array1<f64>, Array1<f64>) {
    let n = 500;
    let mut prices = Array1::zeros(n);
    let mut returns = Array1::zeros(n - 1);

    prices[0] = 100.0;
    let mut volatility = 0.02;

    for i in 1..n {
        let t = i as f64;

        // Time-varying volatility
        let vol_innovation = 0.001 * (t * 0.01).sin();
        volatility = 0.9 * volatility + 0.1 * 0.02 + vol_innovation.abs();

        // Generate return with pseudo-random noise
        let seed = (i * 7919 + 1013) % 10000;
        let z = (seed as f64 / 5000.0) - 1.0; // Normalize to [-1, 1]

        let return_val = volatility * z;
        let trend = -0.0001 * (prices[i - 1] - 100.0) / 100.0;
        let market_trend = 0.0002 * (1.0 + (t / 200.0).sin());

        let total_return = return_val + trend + market_trend;
        returns[i - 1] = total_return;
        prices[i] = prices[i - 1] * (1.0 + total_return);
    }

    (prices, returns)
}

fn technical_indicators_demo(prices: &Array1<f64>) {
    // Simple Moving Average
    if let Ok(sma_result) = sma(prices, 20) {
        println!(
            "  SMA(20): last value = {:.2}",
            sma_result[sma_result.len() - 1]
        );
    }

    // Exponential Moving Average
    if let Ok(ema_result) = ema(prices, 0.1) {
        println!(
            "  EMA(α=0.1): last value = {:.2}",
            ema_result[ema_result.len() - 1]
        );
    }

    // Relative Strength Index
    if let Ok(rsi_result) = rsi(prices, 14) {
        let last_rsi = rsi_result[rsi_result.len() - 1];
        let signal = if last_rsi > 70.0 {
            "Overbought"
        } else if last_rsi < 30.0 {
            "Oversold"
        } else {
            "Neutral"
        };
        println!("  RSI(14): {:.2} [{}]", last_rsi, signal);
    }

    // Bollinger Bands (using basic version which returns a tuple)
    if let Ok((middle, upper, lower)) =
        scirs2_series::financial::technical_indicators::basic::bollinger_bands(prices, 20, 2.0)
    {
        let idx = middle.len() - 1;
        println!(
            "  Bollinger Bands(20, 2σ): middle={:.2}, upper={:.2}, lower={:.2}",
            middle[idx], upper[idx], lower[idx]
        );
    }
}

fn volatility_demo(returns: &Array1<f64>) {
    // Realized volatility
    let realized_vol = realized_volatility(returns);
    let annualized_vol = realized_vol * (252.0_f64).sqrt();
    println!(
        "  Realized volatility: {:.4} ({:.2}% annualized)",
        realized_vol,
        annualized_vol * 100.0
    );

    // EWMA volatility
    if let Ok(ewma_vol) = ewma_volatility(returns, 0.94) {
        let annualized_ewma = ewma_vol.clone() * (252.0_f64).sqrt();
        println!(
            "  EWMA volatility (λ=0.94): {:.4} ({:.2}% annualized)",
            ewma_vol,
            annualized_ewma * 100.0
        );
    }
}

fn garch_demo(returns: &Array1<f64>) {
    let config = GarchConfig {
        p: 1,
        q: 1,
        mean_model: MeanModel::Constant,
        distribution: Distribution::Normal,
        max_iterations: 500,
        tolerance: 1e-6,
        use_numerical_derivatives: false,
    };

    let mut garch_model = GarchModel::new(config);
    match garch_model.fit(returns) {
        Ok(result) => {
            println!("  GARCH(1,1) fitted successfully");
            println!("    Log-likelihood: {:.2}", result.log_likelihood);
            println!("    AIC: {:.2}, BIC: {:.2}", result.aic, result.bic);

            let cond_var = &result.conditional_variance;
            if !cond_var.is_empty() {
                let last_var = cond_var[cond_var.len() - 1];
                let last_vol = last_var.sqrt();
                let annualized = last_vol * (252.0_f64).sqrt();
                println!(
                    "    Last conditional volatility: {:.4} ({:.2}% annualized)",
                    last_vol,
                    annualized * 100.0
                );
            }
        }
        Err(e) => println!("  GARCH fitting failed: {}", e),
    }
}

fn risk_metrics_demo(prices: &Array1<f64>, returns: &Array1<f64>) {
    let risk_free_rate = 0.02; // 2% annual
    let periods_per_year = 252;

    // Value at Risk
    if let Ok(var_95) = var_historical(returns, 0.95) {
        println!("  VaR (95%): {:.4} ({:.2}%)", var_95, var_95 * 100.0);
    }

    // Expected Shortfall
    if let Ok(es_95) = expected_shortfall(returns, 0.95) {
        println!(
            "  Expected Shortfall (95%): {:.4} ({:.2}%)",
            es_95,
            es_95 * 100.0
        );
    }

    // Sharpe Ratio
    if let Ok(sharpe) = sharpe_ratio(returns, risk_free_rate, periods_per_year) {
        println!("  Sharpe Ratio: {:.3}", sharpe);
    }

    // Maximum Drawdown
    if let Ok(max_dd) = max_drawdown(prices) {
        println!("  Maximum Drawdown: {:.2}%", max_dd * 100.0);
    }

    // Basic statistics
    let mean_return = returns.mean_or(0.0);
    let volatility = returns.std_dev();
    let annualized_return = mean_return * periods_per_year as f64;
    let annualized_volatility = volatility * (periods_per_year as f64).sqrt();

    println!("\n  Performance Summary:");
    println!(
        "    Mean return: {:.4}% (daily), {:.2}% (annual)",
        mean_return * 100.0,
        annualized_return * 100.0
    );
    println!(
        "    Volatility: {:.4}% (daily), {:.2}% (annual)",
        volatility * 100.0,
        annualized_volatility * 100.0
    );
}
