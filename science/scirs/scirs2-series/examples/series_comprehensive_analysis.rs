//! Comprehensive Time Series Analysis - Minimal Working Demo
//!
//! This example demonstrates the current function-based API for time series analysis.
//! Note: Full examples will be updated in v0.1.0 after API stabilization.

use scirs2_core::ndarray::{array, s, Array1};
use scirs2_series::{
    anomaly::{detect_anomalies, AnomalyMethod, AnomalyOptions},
    arima_models::ArimaModel,
    change_point::{detect_change_points, ChangePointMethod, ChangePointOptions, CostFunction},
    clustering::TimeSeriesClusterer,
    decomposition::stl::{stl_decomposition, STLOptions},
};

fn main() {
    println!("=== Comprehensive Time Series Analysis Demo ===\n");

    // Generate synthetic time series data
    let data = generate_test_series(500, 0.01, 12, 1.0);
    println!("Generated {} observations\n", data.len());

    // Demo 1: STL Decomposition
    println!("1. STL Decomposition");
    stl_demo(&data);

    // Demo 2: Anomaly Detection
    println!("\n2. Anomaly Detection");
    anomaly_demo(&data);

    // Demo 3: Change Point Detection
    println!("\n3. Change Point Detection");
    change_point_demo();

    // Demo 4: ARIMA Modeling
    println!("\n4. ARIMA Modeling");
    arima_demo(&data);

    // Demo 5: Time Series Clustering
    println!("\n5. Time Series Clustering");
    clustering_demo();

    println!("\n=== Analysis Complete ===");
}

fn generate_test_series(
    length: usize,
    trend: f64,
    seasonal_period: usize,
    noise_std: f64,
) -> Array1<f64> {
    let mut series = Array1::zeros(length);
    let mut rng_state = 42u64;

    for i in 0..length {
        // Deterministic pseudo-random noise
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = ((rng_state % 20000) as f64 / 20000.0 - 0.5) * noise_std;

        let trend_component = (i as f64) * trend;
        let seasonal_component = (2.0 * std::f64::consts::PI * (i % seasonal_period) as f64
            / seasonal_period as f64)
            .sin()
            * 5.0;

        series[i] = 50.0 + trend_component + seasonal_component + noise;
    }

    series
}

fn stl_demo(data: &Array1<f64>) {
    let period = 12;
    let options = STLOptions::default();

    match stl_decomposition(data, period, &options) {
        Ok(result) => {
            println!("  STL decomposition successful");
            println!(
                "    Trend range: {:.2} to {:.2}",
                result.trend.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                result
                    .trend
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            );
            println!(
                "    Seasonal range: {:.2} to {:.2}",
                result.seasonal.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                result
                    .seasonal
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            );

            let residual_std = (result.residual.iter().map(|&x| x * x).sum::<f64>()
                / result.residual.len() as f64)
                .sqrt();
            println!("    Residual std dev: {:.4}", residual_std);
        }
        Err(e) => println!("  STL decomposition failed: {}", e),
    }
}

fn anomaly_demo(data: &Array1<f64>) {
    let options = AnomalyOptions {
        method: AnomalyMethod::ZScore,
        threshold: Some(3.0),
        ..Default::default()
    };

    match detect_anomalies(data, &options) {
        Ok(result) => {
            let n_anomalies = result.is_anomaly.iter().filter(|&&x| x).count();
            println!(
                "  Detected {} anomalies using Z-score method (threshold=3.0)",
                n_anomalies
            );

            if n_anomalies > 0 {
                if let Some(first_idx) = result.is_anomaly.iter().position(|&x| x) {
                    println!(
                        "    First anomaly at index {}: value = {:.2}",
                        first_idx, data[first_idx]
                    );
                }
            }
        }
        Err(e) => println!("  Anomaly detection failed: {}", e),
    }

    // Try IQR method
    let iqr_options = AnomalyOptions {
        method: AnomalyMethod::InterquartileRange,
        threshold: Some(1.5),
        ..Default::default()
    };

    match detect_anomalies(data, &iqr_options) {
        Ok(result) => {
            let n_iqr_anomalies = result.is_anomaly.iter().filter(|&&x| x).count();
            println!("  IQR method detected {} anomalies", n_iqr_anomalies);
        }
        Err(e) => println!("  IQR anomaly detection failed: {}", e),
    }
}

fn change_point_demo() {
    // Generate data with known change points
    let mut series = Array1::zeros(300);
    let changepoints = [100, 200];
    let mut current_mean = 10.0;
    let mut next_change_idx = 0;

    for i in 0..300 {
        if next_change_idx < changepoints.len() && i == changepoints[next_change_idx] {
            current_mean += 15.0;
            next_change_idx += 1;
        }
        let noise = ((i % 17) as f64 - 8.0) * 0.3;
        series[i] = current_mean + noise;
    }

    let options = ChangePointOptions {
        method: ChangePointMethod::PELT,
        cost_function: CostFunction::Normal,
        penalty: 3.0,
        ..Default::default()
    };

    match detect_change_points(&series, &options) {
        Ok(result) => {
            println!(
                "  PELT detected {} change points",
                result.change_points.len()
            );
            if !result.change_points.is_empty() {
                println!("    Change points at: {:?}", result.change_points);
            }
        }
        Err(e) => println!("  Change point detection failed: {}", e),
    }
}

fn arima_demo(data: &Array1<f64>) {
    // Take a subset for faster fitting
    let subset = data.slice(s![..200]).to_owned();

    match ArimaModel::new(1, 1, 1) {
        Ok(mut arima) => {
            match arima.fit(&subset) {
                Ok(_) => {
                    println!("  ARIMA(1,1,1) fitted successfully");

                    // Make predictions (forecast requires the data as second parameter)
                    match arima.forecast(10, &subset) {
                        Ok(forecast) => {
                            println!("    Forecast next 10 steps:");
                            for (i, &val) in forecast.iter().enumerate() {
                                println!("      Step {}: {:.2}", i + 1, val);
                            }
                        }
                        Err(e) => println!("    Forecast failed: {}", e),
                    }
                }
                Err(e) => println!("  ARIMA fitting failed: {}", e),
            }
        }
        Err(e) => println!("  ARIMA model creation failed: {}", e),
    }
}

fn clustering_demo() {
    // Create sample time series for clustering
    let series1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let series2 = array![1.1, 2.1, 3.1, 4.1, 5.1];
    let series3 = array![10.0, 11.0, 12.0, 13.0, 14.0];

    println!("  Created 3 sample time series");
    println!("    Series 1 & 2: similar pattern (values 1-5)");
    println!("    Series 3: different pattern (values 10-14)");
    println!("  Note: Clustering API requires further implementation");
}
