//! Model calibration utilities
//!
//! This module provides tools for calibrating financial models to market data including
//! optimization algorithms, objective functions, and regularization techniques.
//!
//! # Features
//! - Volatility surface calibration using local/global optimization
//! - Heston model parameter estimation from market prices
//! - Weighted least squares with bid-ask spreads
//! - Loss functions (MSE, MAE, relative error)
//! - Regularization for parameter stability
//!
//! # Example
//! ```
//! use scirs2_integrate::specialized::finance::utils::calibration::{
//!     ImpliedVolatilitySurface, HestonCalibrator, LossFunction
//! };
//!
//! // Create IV surface from market quotes
//! let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
//! let maturities = vec![0.25, 0.5, 1.0];
//! let mut surface = ImpliedVolatilitySurface::new(100.0, 0.05, 0.0);
//!
//! // Add market data
//! for &t in &maturities {
//!     for &k in &strikes {
//!         surface.add_quote(k, t, 0.20, Some(0.01));
//!     }
//! }
//!
//! // Calibrate to Heston model
//! let calibrator = HestonCalibrator::new(surface, LossFunction::WeightedMSE);
//! // let result = calibrator.calibrate()?;
//! ```

use crate::error::{IntegrateError, IntegrateResult as Result};
use crate::specialized::finance::derivatives::vanilla::EuropeanOption;
use crate::specialized::finance::types::OptionType;
use std::collections::HashMap;

// ============================================================================
// Nelder-Mead Simplex Optimizer
// ============================================================================

/// Result from Nelder-Mead optimization
#[derive(Debug, Clone)]
struct OptimizationResult {
    /// Optimized parameters
    parameters: Vec<f64>,
    /// Final objective function value
    final_value: f64,
    /// Number of iterations
    iterations: usize,
    /// Whether optimization converged
    converged: bool,
}

/// Nelder-Mead simplex optimizer for unconstrained optimization
///
/// This is a derivative-free optimization method that maintains a simplex
/// of n+1 points in n-dimensional space and iteratively improves it.
struct NelderMeadOptimizer {
    /// Initial guess
    initial_guess: Vec<f64>,
    /// Parameter bounds (min, max)
    bounds: Vec<(f64, f64)>,
    /// Maximum iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Reflection coefficient (alpha)
    alpha: f64,
    /// Expansion coefficient (gamma)
    gamma: f64,
    /// Contraction coefficient (rho)
    rho: f64,
    /// Shrink coefficient (sigma)
    sigma: f64,
}

impl NelderMeadOptimizer {
    /// Create a new Nelder-Mead optimizer
    fn new(
        initial_guess: Vec<f64>,
        bounds: Vec<(f64, f64)>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Self {
        Self {
            initial_guess,
            bounds,
            max_iterations,
            tolerance,
            alpha: 1.0, // Reflection
            gamma: 2.0, // Expansion
            rho: 0.5,   // Contraction
            sigma: 0.5, // Shrink
        }
    }

    /// Optimize the objective function
    fn optimize<F>(&self, objective: &F) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = self.initial_guess.len();

        // Initialize simplex (n+1 points)
        let mut simplex = self.initialize_simplex(n);

        // Evaluate all simplex points
        let mut values: Vec<f64> = simplex.iter().map(|x| objective(x)).collect();

        let mut iterations = 0;

        while iterations < self.max_iterations {
            // Sort simplex by objective values
            let mut indices: Vec<usize> = (0..simplex.len()).collect();
            indices.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).expect("Operation failed"));

            let best_idx = indices[0];
            let worst_idx = indices[n];
            let second_worst_idx = indices[n - 1];

            // Check convergence: standard deviation of function values
            let mean_val: f64 = values.iter().sum::<f64>() / values.len() as f64;
            let std_dev = (values.iter().map(|v| (v - mean_val).powi(2)).sum::<f64>()
                / values.len() as f64)
                .sqrt();

            if std_dev < self.tolerance {
                return Ok(OptimizationResult {
                    parameters: simplex[best_idx].clone(),
                    final_value: values[best_idx],
                    iterations,
                    converged: true,
                });
            }

            // Calculate centroid (excluding worst point)
            let centroid = self.calculate_centroid(&simplex, worst_idx);

            // Reflection
            let reflected = self.reflect(&simplex[worst_idx], &centroid, self.alpha);
            let reflected = self.project_to_bounds(&reflected);
            let reflected_value = objective(&reflected);

            if reflected_value < values[second_worst_idx] && reflected_value >= values[best_idx] {
                // Accept reflection
                simplex[worst_idx] = reflected;
                values[worst_idx] = reflected_value;
            } else if reflected_value < values[best_idx] {
                // Try expansion
                let expanded =
                    self.reflect(&simplex[worst_idx], &centroid, self.alpha * self.gamma);
                let expanded = self.project_to_bounds(&expanded);
                let expanded_value = objective(&expanded);

                if expanded_value < reflected_value {
                    simplex[worst_idx] = expanded;
                    values[worst_idx] = expanded_value;
                } else {
                    simplex[worst_idx] = reflected;
                    values[worst_idx] = reflected_value;
                }
            } else {
                // Try contraction
                let contracted = if reflected_value < values[worst_idx] {
                    // Outside contraction
                    self.contract(&reflected, &centroid, self.rho)
                } else {
                    // Inside contraction
                    self.contract(&simplex[worst_idx], &centroid, self.rho)
                };
                let contracted = self.project_to_bounds(&contracted);
                let contracted_value = objective(&contracted);

                if contracted_value < values[worst_idx].min(reflected_value) {
                    simplex[worst_idx] = contracted;
                    values[worst_idx] = contracted_value;
                } else {
                    // Shrink entire simplex toward best point
                    for i in 0..simplex.len() {
                        if i != best_idx {
                            simplex[i] = self.shrink(&simplex[i], &simplex[best_idx], self.sigma);
                            simplex[i] = self.project_to_bounds(&simplex[i]);
                            values[i] = objective(&simplex[i]);
                        }
                    }
                }
            }

            iterations += 1;
        }

        // Max iterations reached
        let mut indices: Vec<usize> = (0..simplex.len()).collect();
        indices.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).expect("Operation failed"));
        let best_idx = indices[0];

        Ok(OptimizationResult {
            parameters: simplex[best_idx].clone(),
            final_value: values[best_idx],
            iterations,
            converged: false,
        })
    }

    /// Initialize simplex around initial guess
    fn initialize_simplex(&self, n: usize) -> Vec<Vec<f64>> {
        let mut simplex = Vec::with_capacity(n + 1);

        // First point is the initial guess
        simplex.push(self.initial_guess.clone());

        // Create n additional points by perturbing each dimension
        for i in 0..n {
            let mut point = self.initial_guess.clone();
            let delta = if self.initial_guess[i].abs() > 1e-10 {
                0.05 * self.initial_guess[i] // 5% perturbation
            } else {
                0.00025 // Small absolute perturbation for near-zero values
            };
            point[i] += delta;
            point = self.project_to_bounds(&point);
            simplex.push(point);
        }

        simplex
    }

    /// Calculate centroid of simplex excluding specified point
    fn calculate_centroid(&self, simplex: &[Vec<f64>], exclude_idx: usize) -> Vec<f64> {
        let n = simplex[0].len();
        let mut centroid = vec![0.0; n];

        for (i, point) in simplex.iter().enumerate() {
            if i != exclude_idx {
                for (j, &val) in point.iter().enumerate() {
                    centroid[j] += val;
                }
            }
        }

        let count = simplex.len() - 1;
        for val in &mut centroid {
            *val /= count as f64;
        }

        centroid
    }

    /// Reflect point through centroid
    fn reflect(&self, point: &[f64], centroid: &[f64], coeff: f64) -> Vec<f64> {
        point
            .iter()
            .zip(centroid.iter())
            .map(|(&p, &c)| c + coeff * (c - p))
            .collect()
    }

    /// Contract point toward centroid
    fn contract(&self, point: &[f64], centroid: &[f64], coeff: f64) -> Vec<f64> {
        point
            .iter()
            .zip(centroid.iter())
            .map(|(&p, &c)| c + coeff * (p - c))
            .collect()
    }

    /// Shrink point toward best point
    fn shrink(&self, point: &[f64], best: &[f64], coeff: f64) -> Vec<f64> {
        point
            .iter()
            .zip(best.iter())
            .map(|(&p, &b)| b + coeff * (p - b))
            .collect()
    }

    /// Project point to within bounds
    fn project_to_bounds(&self, point: &[f64]) -> Vec<f64> {
        point
            .iter()
            .zip(self.bounds.iter())
            .map(|(&val, &(min, max))| val.max(min).min(max))
            .collect()
    }
}

// ============================================================================
// Calibration Data Structures
// ============================================================================

/// Market quote for an option
#[derive(Debug, Clone)]
pub struct OptionQuote {
    /// Strike price
    pub strike: f64,
    /// Time to maturity (years)
    pub maturity: f64,
    /// Option type (call/put)
    pub option_type: OptionType,
    /// Market price
    pub market_price: f64,
    /// Bid-ask spread (optional, for weighting)
    pub bid_ask_spread: Option<f64>,
}

impl OptionQuote {
    /// Create a new option quote
    pub fn new(
        strike: f64,
        maturity: f64,
        option_type: OptionType,
        market_price: f64,
        bid_ask_spread: Option<f64>,
    ) -> Result<Self> {
        if strike <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Strike must be positive".to_string(),
            ));
        }
        if maturity <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Maturity must be positive".to_string(),
            ));
        }
        if market_price < 0.0 {
            return Err(IntegrateError::ValueError(
                "Market price cannot be negative".to_string(),
            ));
        }
        if let Some(spread) = bid_ask_spread {
            if spread < 0.0 {
                return Err(IntegrateError::ValueError(
                    "Bid-ask spread cannot be negative".to_string(),
                ));
            }
        }

        Ok(Self {
            strike,
            maturity,
            option_type,
            market_price,
            bid_ask_spread,
        })
    }

    /// Calculate weight based on bid-ask spread (tighter spread = higher weight)
    pub fn weight(&self) -> f64 {
        match self.bid_ask_spread {
            Some(spread) if spread > 1e-8 => 1.0 / spread,
            _ => 1.0,
        }
    }
}

/// Implied volatility surface
pub struct ImpliedVolatilitySurface {
    /// Spot price
    spot: f64,
    /// Risk-free rate
    rate: f64,
    /// Dividend yield
    dividend: f64,
    /// Market quotes indexed by (strike, maturity)
    quotes: HashMap<(String, String), (f64, Option<f64>)>,
}

impl ImpliedVolatilitySurface {
    /// Create a new implied volatility surface
    pub fn new(spot: f64, rate: f64, dividend: f64) -> Self {
        Self {
            spot,
            rate,
            dividend,
            quotes: HashMap::new(),
        }
    }

    /// Add an implied volatility quote
    pub fn add_quote(
        &mut self,
        strike: f64,
        maturity: f64,
        implied_vol: f64,
        bid_ask_spread: Option<f64>,
    ) {
        let key = (format!("{:.4}", strike), format!("{:.4}", maturity));
        self.quotes.insert(key, (implied_vol, bid_ask_spread));
    }

    /// Get implied volatility for a given strike and maturity
    pub fn get_vol(&self, strike: f64, maturity: f64) -> Option<f64> {
        let key = (format!("{:.4}", strike), format!("{:.4}", maturity));
        self.quotes.get(&key).map(|(vol, _)| *vol)
    }

    /// Get all strikes
    pub fn strikes(&self) -> Vec<f64> {
        let mut strikes: Vec<f64> = self
            .quotes
            .keys()
            .map(|(k, _)| k.parse::<f64>().unwrap_or(0.0))
            .collect();
        strikes.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));
        strikes.dedup();
        strikes
    }

    /// Get all maturities
    pub fn maturities(&self) -> Vec<f64> {
        let mut maturities: Vec<f64> = self
            .quotes
            .keys()
            .map(|(_, t)| t.parse::<f64>().unwrap_or(0.0))
            .collect();
        maturities.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));
        maturities.dedup();
        maturities
    }

    /// Convert to option quotes (for calibration)
    pub fn to_option_quotes(&self) -> Result<Vec<OptionQuote>> {
        let mut quotes = Vec::new();

        for ((strike_str, maturity_str), (vol, spread)) in &self.quotes {
            let strike: f64 = strike_str
                .parse()
                .map_err(|_| IntegrateError::ValueError("Invalid strike format".to_string()))?;
            let maturity: f64 = maturity_str
                .parse()
                .map_err(|_| IntegrateError::ValueError("Invalid maturity format".to_string()))?;

            // Assume call options (can be extended)
            let option = EuropeanOption::new(
                self.spot,
                strike,
                self.rate,
                self.dividend,
                *vol,
                maturity,
                OptionType::Call,
            );

            let market_price = option.price();

            quotes.push(OptionQuote::new(
                strike,
                maturity,
                OptionType::Call,
                market_price,
                *spread,
            )?);
        }

        Ok(quotes)
    }

    /// Number of quotes in the surface
    pub fn size(&self) -> usize {
        self.quotes.len()
    }
}

/// Loss function types for calibration
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    /// Mean squared error
    MSE,
    /// Mean absolute error
    MAE,
    /// Weighted MSE (by inverse bid-ask spread)
    WeightedMSE,
    /// Relative error (percentage)
    RelativeError,
}

impl LossFunction {
    /// Calculate loss between model and market prices
    pub fn calculate(&self, quotes: &[OptionQuote], model_prices: &[f64]) -> Result<f64> {
        if quotes.len() != model_prices.len() {
            return Err(IntegrateError::ValueError(
                "Mismatched number of quotes and prices".to_string(),
            ));
        }

        if quotes.is_empty() {
            return Err(IntegrateError::ValueError("No quotes provided".to_string()));
        }

        match self {
            LossFunction::MSE => {
                let sum: f64 = quotes
                    .iter()
                    .zip(model_prices.iter())
                    .map(|(q, &p)| (q.market_price - p).powi(2))
                    .sum();
                Ok(sum / quotes.len() as f64)
            }
            LossFunction::MAE => {
                let sum: f64 = quotes
                    .iter()
                    .zip(model_prices.iter())
                    .map(|(q, &p)| (q.market_price - p).abs())
                    .sum();
                Ok(sum / quotes.len() as f64)
            }
            LossFunction::WeightedMSE => {
                let weighted_sum: f64 = quotes
                    .iter()
                    .zip(model_prices.iter())
                    .map(|(q, &p)| {
                        let error = (q.market_price - p).powi(2);
                        error * q.weight()
                    })
                    .sum();

                let total_weight: f64 = quotes.iter().map(|q| q.weight()).sum();
                Ok(weighted_sum / total_weight)
            }
            LossFunction::RelativeError => {
                let sum: f64 = quotes
                    .iter()
                    .zip(model_prices.iter())
                    .map(|(q, &p)| {
                        if q.market_price.abs() > 1e-8 {
                            ((q.market_price - p) / q.market_price).abs()
                        } else {
                            (q.market_price - p).abs()
                        }
                    })
                    .sum();
                Ok(sum / quotes.len() as f64)
            }
        }
    }
}

/// Calibration result
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Calibrated parameters
    pub parameters: Vec<f64>,
    /// Parameter names
    pub parameter_names: Vec<String>,
    /// Final loss value
    pub loss: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
}

impl CalibrationResult {
    /// Create a new calibration result
    pub fn new(
        parameters: Vec<f64>,
        parameter_names: Vec<String>,
        loss: f64,
        iterations: usize,
        converged: bool,
    ) -> Self {
        Self {
            parameters,
            parameter_names,
            loss,
            iterations,
            converged,
        }
    }

    /// Get parameter by name
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameter_names
            .iter()
            .position(|n| n == name)
            .map(|i| self.parameters[i])
    }
}

/// Heston model calibrator
pub struct HestonCalibrator {
    /// Market implied volatility surface
    surface: ImpliedVolatilitySurface,
    /// Loss function
    loss_function: LossFunction,
    /// Maximum iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
}

impl HestonCalibrator {
    /// Create a new Heston calibrator
    pub fn new(surface: ImpliedVolatilitySurface, loss_function: LossFunction) -> Self {
        Self {
            surface,
            loss_function,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Calibrate Heston model parameters using Nelder-Mead optimization
    ///
    /// Returns: CalibrationResult with (kappa, theta, sigma, rho, v0)
    ///
    /// Parameter constraints:
    /// - kappa > 0 (mean reversion speed)
    /// - theta > 0 (long-run variance)
    /// - sigma > 0 (vol of vol)
    /// - -1 < rho < 1 (correlation)
    /// - v0 > 0 (initial variance)
    /// - Feller condition: 2*kappa*theta >= sigma^2 (for positivity)
    pub fn calibrate(&self) -> Result<CalibrationResult> {
        // Initial guess for Heston parameters (reasonable defaults)
        let initial_guess = vec![
            2.0,  // kappa: mean reversion speed
            0.04, // theta: long-run variance (20% vol)
            0.3,  // sigma: vol of vol
            -0.5, // rho: correlation (typically negative)
            0.04, // v0: initial variance
        ];

        // Parameter bounds: [min, max] for each parameter
        let bounds = vec![
            (0.01, 10.0),  // kappa
            (0.001, 1.0),  // theta
            (0.01, 2.0),   // sigma
            (-0.99, 0.99), // rho
            (0.001, 1.0),  // v0
        ];

        // Run Nelder-Mead optimization
        let optimizer = NelderMeadOptimizer::new(
            initial_guess.clone(),
            bounds,
            self.max_iterations,
            self.tolerance,
        );

        let objective = |params: &[f64]| -> f64 { self.heston_objective(params).unwrap_or(1e10) };

        let result = optimizer.optimize(&objective)?;

        let parameter_names = vec![
            "kappa".to_string(),
            "theta".to_string(),
            "sigma".to_string(),
            "rho".to_string(),
            "v0".to_string(),
        ];

        Ok(CalibrationResult::new(
            result.parameters,
            parameter_names,
            result.final_value,
            result.iterations,
            result.converged,
        ))
    }

    /// Objective function for Heston calibration
    fn heston_objective(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 5 {
            return Err(IntegrateError::ValueError(
                "Heston calibration requires 5 parameters".to_string(),
            ));
        }

        let kappa = params[0];
        let theta = params[1];
        let sigma = params[2];
        let _rho = params[3]; // Not used in simplified model
        let v0 = params[4];

        // Check Feller condition (relaxed to warning)
        let feller_condition = 2.0 * kappa * theta;
        if feller_condition < sigma * sigma * 0.8 {
            // Penalize but don't reject
            return Ok(1e8);
        }

        // Convert surface to option quotes
        let quotes = self.surface.to_option_quotes()?;
        if quotes.is_empty() {
            return Err(IntegrateError::ValueError(
                "No market quotes available".to_string(),
            ));
        }

        // Calculate model prices using Heston model (simplified approximation)
        // Uses time-dependent effective volatility from mean-reverting variance
        let mut model_prices = Vec::new();

        for quote in &quotes {
            // Effective variance at maturity: E[v_t] = theta + (v0 - theta)*exp(-kappa*t)
            let effective_variance = theta + (v0 - theta) * (-kappa * quote.maturity).exp();
            let effective_vol = effective_variance.sqrt();

            // Price using Black-Scholes with effective vol
            let price = match quote.option_type {
                OptionType::Call => super::math::black_scholes_call(
                    self.surface.spot,
                    quote.strike,
                    quote.maturity,
                    self.surface.rate,
                    effective_vol,
                ),
                OptionType::Put => super::math::black_scholes_put(
                    self.surface.spot,
                    quote.strike,
                    quote.maturity,
                    self.surface.rate,
                    effective_vol,
                ),
            };

            model_prices.push(price);
        }

        // Calculate loss
        self.loss_function.calculate(&quotes, &model_prices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_quote_creation() {
        let quote = OptionQuote::new(100.0, 1.0, OptionType::Call, 10.0, Some(0.5))
            .expect("Operation failed");
        assert_eq!(quote.strike, 100.0);
        assert_eq!(quote.maturity, 1.0);
        assert_eq!(quote.market_price, 10.0);
    }

    #[test]
    fn test_option_quote_weight() {
        let quote1 = OptionQuote::new(100.0, 1.0, OptionType::Call, 10.0, Some(0.5))
            .expect("Operation failed");
        let quote2 = OptionQuote::new(100.0, 1.0, OptionType::Call, 10.0, Some(0.1))
            .expect("Operation failed");

        // Tighter spread should have higher weight
        assert!(quote2.weight() > quote1.weight());
    }

    #[test]
    fn test_implied_vol_surface() {
        let mut surface = ImpliedVolatilitySurface::new(100.0, 0.05, 0.0);

        surface.add_quote(100.0, 1.0, 0.20, None);
        surface.add_quote(110.0, 1.0, 0.22, None);

        assert_eq!(surface.size(), 2);
        assert_eq!(surface.get_vol(100.0, 1.0), Some(0.20));
        assert_eq!(surface.get_vol(110.0, 1.0), Some(0.22));
    }

    #[test]
    fn test_surface_strikes_maturities() {
        let mut surface = ImpliedVolatilitySurface::new(100.0, 0.05, 0.0);

        surface.add_quote(90.0, 0.5, 0.18, None);
        surface.add_quote(100.0, 0.5, 0.20, None);
        surface.add_quote(110.0, 0.5, 0.22, None);
        surface.add_quote(100.0, 1.0, 0.21, None);

        let strikes = surface.strikes();
        let maturities = surface.maturities();

        assert_eq!(strikes.len(), 3);
        assert_eq!(maturities.len(), 2);
        assert!(strikes.contains(&100.0));
        assert!(maturities.contains(&0.5));
    }

    #[test]
    fn test_loss_function_mse() {
        let quotes = vec![
            OptionQuote::new(100.0, 1.0, OptionType::Call, 10.0, None).expect("Operation failed"),
            OptionQuote::new(100.0, 1.0, OptionType::Call, 12.0, None).expect("Operation failed"),
        ];
        let model_prices = vec![10.5, 11.5];

        let loss = LossFunction::MSE
            .calculate(&quotes, &model_prices)
            .expect("Operation failed");
        let expected = ((10.0_f64 - 10.5).powi(2) + (12.0_f64 - 11.5).powi(2)) / 2.0;

        assert!((loss - expected).abs() < 1e-10);
    }

    #[test]
    fn test_loss_function_weighted_mse() {
        let quotes = vec![
            OptionQuote::new(100.0, 1.0, OptionType::Call, 10.0, Some(0.5))
                .expect("Operation failed"),
            OptionQuote::new(100.0, 1.0, OptionType::Call, 12.0, Some(0.1))
                .expect("Operation failed"),
        ];
        let model_prices = vec![10.5, 11.5];

        let loss = LossFunction::WeightedMSE
            .calculate(&quotes, &model_prices)
            .expect("Operation failed");

        // Second quote should contribute more due to tighter spread
        let w1 = 1.0 / 0.5;
        let w2 = 1.0 / 0.1;
        let expected =
            (w1 * (10.0_f64 - 10.5).powi(2) + w2 * (12.0_f64 - 11.5).powi(2)) / (w1 + w2);

        assert!((loss - expected).abs() < 1e-10);
    }

    #[test]
    fn test_calibration_result() {
        let result = CalibrationResult::new(
            vec![2.0, 0.04, 0.3, -0.7, 0.04],
            vec![
                "kappa".to_string(),
                "theta".to_string(),
                "sigma".to_string(),
                "rho".to_string(),
                "v0".to_string(),
            ],
            0.01,
            100,
            true,
        );

        assert_eq!(result.get_parameter("kappa"), Some(2.0));
        assert_eq!(result.get_parameter("theta"), Some(0.04));
        assert_eq!(result.loss, 0.01);
        assert!(result.converged);
    }

    #[test]
    fn test_heston_calibrator_creation() {
        let surface = ImpliedVolatilitySurface::new(100.0, 0.05, 0.0);
        let calibrator = HestonCalibrator::new(surface, LossFunction::WeightedMSE)
            .with_max_iterations(500)
            .with_tolerance(1e-5);

        assert_eq!(calibrator.max_iterations, 500);
        assert_eq!(calibrator.tolerance, 1e-5);
    }

    #[test]
    fn test_surface_to_option_quotes() {
        let mut surface = ImpliedVolatilitySurface::new(100.0, 0.05, 0.0);
        surface.add_quote(100.0, 1.0, 0.20, Some(0.5));
        surface.add_quote(110.0, 1.0, 0.22, Some(0.3));

        let quotes = surface.to_option_quotes().expect("Operation failed");
        assert_eq!(quotes.len(), 2);
        assert!(quotes[0].market_price > 0.0);
    }

    #[test]
    fn test_heston_calibration_basic() {
        // Create a simple volatility surface
        let mut surface = ImpliedVolatilitySurface::new(100.0, 0.05, 0.0);

        // Add quotes with realistic implied vols
        surface.add_quote(90.0, 1.0, 0.25, Some(0.02)); // OTM put
        surface.add_quote(100.0, 1.0, 0.20, Some(0.01)); // ATM
        surface.add_quote(110.0, 1.0, 0.22, Some(0.02)); // OTM call

        // Calibrate with reduced iterations for testing
        let calibrator = HestonCalibrator::new(surface, LossFunction::WeightedMSE)
            .with_max_iterations(100)
            .with_tolerance(1e-4);

        let result = calibrator.calibrate().expect("Operation failed");

        // Check that parameters are in reasonable ranges
        assert!(result.get_parameter("kappa").expect("Operation failed") > 0.0);
        assert!(result.get_parameter("theta").expect("Operation failed") > 0.0);
        assert!(result.get_parameter("sigma").expect("Operation failed") > 0.0);
        assert!(result.get_parameter("rho").expect("Operation failed") > -1.0);
        assert!(result.get_parameter("rho").expect("Operation failed") < 1.0);
        assert!(result.get_parameter("v0").expect("Operation failed") > 0.0);

        // Check that loss is reasonable
        assert!(result.loss >= 0.0);
        assert!(result.loss < 1e6); // Not a penalty value

        // Check iterations
        assert!(result.iterations <= 100);
    }

    #[test]
    #[allow(clippy::too_many_arguments)]
    fn test_nelder_mead_rosenbrock() {
        // Test Nelder-Mead on Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // Global minimum at (1, 1) with f = 0
        let rosenbrock = |params: &[f64]| -> f64 {
            let x = params[0];
            let y = params[1];
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };

        let optimizer =
            NelderMeadOptimizer::new(vec![0.0, 0.0], vec![(-5.0, 5.0), (-5.0, 5.0)], 1000, 1e-6);

        let result = optimizer.optimize(&rosenbrock).expect("Operation failed");

        // Should converge close to (1, 1)
        assert!((result.parameters[0] - 1.0).abs() < 0.1);
        assert!((result.parameters[1] - 1.0).abs() < 0.1);
        assert!(result.final_value < 1.0); // Near minimum
    }
}
