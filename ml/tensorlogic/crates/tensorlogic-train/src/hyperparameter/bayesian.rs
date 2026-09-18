//! Bayesian Optimization for hyperparameter tuning.

use crate::TrainResult;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{RngExt, SeedableRng, StdRng};
use std::collections::HashMap;

use super::acquisition::AcquisitionFunction;
use super::gp::GaussianProcess;
use super::kernel::GpKernel;
use super::space::HyperparamSpace;
use super::value::{HyperparamConfig, HyperparamResult, HyperparamValue};

/// Bayesian Optimization for hyperparameter tuning.
///
/// Uses Gaussian Processes to model the objective function and acquisition
/// functions to intelligently select the next hyperparameters to evaluate.
///
/// # Algorithm
/// 1. Initialize with random samples
/// 2. Fit Gaussian Process to observed data
/// 3. Optimize acquisition function to find next point
/// 4. Evaluate objective at new point
/// 5. Repeat steps 2-4 until budget exhausted
///
/// # Example
/// ```
/// use tensorlogic_train::*;
/// use std::collections::HashMap;
///
/// let mut param_space = HashMap::new();
/// param_space.insert(
///     "lr".to_string(),
///     HyperparamSpace::log_uniform(1e-4, 1e-1).expect("unwrap"),
/// );
///
/// let mut bayes_opt = BayesianOptimization::new(
///     param_space,
///     10,  // n_iterations
///     5,   // n_initial_points
///     42,  // seed
/// );
///
/// // In practice, you would evaluate your model here
/// // bayes_opt.add_result(result);
/// ```
#[derive(Debug)]
pub struct BayesianOptimization {
    /// Parameter space definition.
    param_space: HashMap<String, HyperparamSpace>,
    /// Number of optimization iterations.
    n_iterations: usize,
    /// Number of random initial points.
    n_initial_points: usize,
    /// Acquisition function.
    acquisition_fn: AcquisitionFunction,
    /// Gaussian Process kernel.
    kernel: GpKernel,
    /// Observation noise.
    noise_variance: f64,
    /// Random number generator.
    rng: StdRng,
    /// Results from all evaluations.
    results: Vec<HyperparamResult>,
    /// Bounds for normalization [min, max] per dimension.
    bounds: Vec<(f64, f64)>,
    /// Parameter names in order.
    param_names: Vec<String>,
}

impl BayesianOptimization {
    /// Create a new Bayesian Optimization instance.
    ///
    /// # Arguments
    /// * `param_space` - Hyperparameter space definition
    /// * `n_iterations` - Number of optimization iterations
    /// * `n_initial_points` - Number of random initialization points
    /// * `seed` - Random seed for reproducibility
    pub fn new(
        param_space: HashMap<String, HyperparamSpace>,
        n_iterations: usize,
        n_initial_points: usize,
        seed: u64,
    ) -> Self {
        let mut param_names: Vec<String> = param_space.keys().cloned().collect();
        param_names.sort();
        let bounds = Self::extract_bounds(&param_space, &param_names);
        Self {
            param_space,
            n_iterations,
            n_initial_points,
            acquisition_fn: AcquisitionFunction::default(),
            kernel: GpKernel::default(),
            noise_variance: 1e-6,
            rng: StdRng::seed_from_u64(seed),
            results: Vec::new(),
            bounds,
            param_names,
        }
    }

    /// Set acquisition function.
    pub fn with_acquisition(mut self, acquisition_fn: AcquisitionFunction) -> Self {
        self.acquisition_fn = acquisition_fn;
        self
    }

    /// Set kernel.
    pub fn with_kernel(mut self, kernel: GpKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set noise variance.
    pub fn with_noise(mut self, noise_variance: f64) -> Self {
        self.noise_variance = noise_variance;
        self
    }

    /// Extract bounds from parameter space.
    fn extract_bounds(
        param_space: &HashMap<String, HyperparamSpace>,
        param_names: &[String],
    ) -> Vec<(f64, f64)> {
        param_names
            .iter()
            .map(|name| match &param_space[name] {
                HyperparamSpace::Continuous { min, max } => (*min, *max),
                HyperparamSpace::LogUniform { min, max } => (min.ln(), max.ln()),
                HyperparamSpace::IntRange { min, max } => (*min as f64, *max as f64),
                HyperparamSpace::Discrete(values) => (0.0, (values.len() - 1) as f64),
            })
            .collect()
    }

    /// Suggest next hyperparameter configuration to evaluate.
    pub fn suggest(&mut self) -> TrainResult<HyperparamConfig> {
        if self.results.len() < self.n_initial_points {
            return Ok(self.random_sample());
        }
        let (x_observed, y_observed) = self.get_observations();
        let mut gp = GaussianProcess::new(self.kernel, self.noise_variance);
        gp.fit(x_observed, y_observed)?;
        let best_x = self.optimize_acquisition(&gp)?;
        self.vector_to_config(&best_x)
    }

    /// Get observations as (X, y) matrices.
    fn get_observations(&self) -> (Array2<f64>, Array1<f64>) {
        let n_samples = self.results.len();
        let n_dims = self.param_names.len();
        let mut x = Array2::zeros((n_samples, n_dims));
        let mut y = Array1::zeros(n_samples);
        for (i, result) in self.results.iter().enumerate() {
            let x_vec = self.config_to_vector(&result.config);
            for (j, &val) in x_vec.iter().enumerate() {
                x[[i, j]] = val;
            }
            y[i] = result.score;
        }
        (x, y)
    }

    /// Optimize acquisition function to find next point.
    fn optimize_acquisition(&mut self, gp: &GaussianProcess) -> TrainResult<Array1<f64>> {
        let n_dims = self.param_names.len();
        let n_candidates = 1000;
        let n_restarts = 10;
        let mut best_acq_value = f64::NEG_INFINITY;
        let mut best_x = Array1::zeros(n_dims);
        for _ in 0..n_restarts {
            for _ in 0..(n_candidates / n_restarts) {
                let mut x_candidate = Array1::zeros(n_dims);
                for (i, (min, max)) in self.bounds.iter().enumerate() {
                    x_candidate[i] = min + (max - min) * self.rng.random::<f64>();
                }
                let acq_value = self.evaluate_acquisition(gp, &x_candidate)?;
                if acq_value > best_acq_value {
                    best_acq_value = acq_value;
                    best_x = x_candidate;
                }
            }
        }
        Ok(best_x)
    }

    /// Evaluate acquisition function at a point.
    fn evaluate_acquisition(&self, gp: &GaussianProcess, x: &Array1<f64>) -> TrainResult<f64> {
        let x_mat = x
            .clone()
            .into_shape_with_order((1, x.len()))
            .expect("shape and data length match");
        let (mean, std) = gp.predict(&x_mat)?;
        let mu = mean[0];
        let sigma = std[0];
        if sigma < 1e-10 {
            return Ok(0.0);
        }
        let f_best = self
            .results
            .iter()
            .map(|r| r.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let acq = match self.acquisition_fn {
            AcquisitionFunction::ExpectedImprovement { xi } => {
                let z = (mu - f_best - xi) / sigma;
                let phi = Self::normal_cdf(z);
                let pdf = Self::normal_pdf(z);
                (mu - f_best - xi) * phi + sigma * pdf
            }
            AcquisitionFunction::UpperConfidenceBound { kappa } => mu + kappa * sigma,
            AcquisitionFunction::ProbabilityOfImprovement { xi } => {
                let z = (mu - f_best - xi) / sigma;
                Self::normal_cdf(z)
            }
        };
        Ok(acq)
    }

    /// Standard normal CDF (cumulative distribution function).
    pub(super) fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
    }

    /// Standard normal PDF (probability density function).
    pub(super) fn normal_pdf(x: f64) -> f64 {
        (-0.5 * x.powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    /// Error function approximation.
    pub(super) fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        sign * y
    }

    /// Convert configuration to normalized vector [0, 1]^d.
    fn config_to_vector(&self, config: &HyperparamConfig) -> Array1<f64> {
        let n_dims = self.param_names.len();
        let mut x = Array1::zeros(n_dims);
        for (i, name) in self.param_names.iter().enumerate() {
            let value = &config[name];
            let (min, max) = self.bounds[i];
            x[i] = match &self.param_space[name] {
                HyperparamSpace::Continuous { .. } => {
                    let v = value
                        .as_float()
                        .expect("Continuous space requires float value");
                    (v - min) / (max - min)
                }
                HyperparamSpace::LogUniform { .. } => {
                    let v = value
                        .as_float()
                        .expect("LogUniform space requires float value");
                    let log_v = v.ln();
                    (log_v - min) / (max - min)
                }
                HyperparamSpace::IntRange { .. } => {
                    let v = value.as_int().expect("IntRange space requires int value") as f64;
                    (v - min) / (max - min)
                }
                HyperparamSpace::Discrete(values) => {
                    let idx = values.iter().position(|v| v == value).unwrap_or(0);
                    (idx as f64 - min) / (max - min)
                }
            };
        }
        x
    }

    /// Convert normalized vector to configuration.
    fn vector_to_config(&self, x: &Array1<f64>) -> TrainResult<HyperparamConfig> {
        let mut config = HashMap::new();
        for (i, name) in self.param_names.iter().enumerate() {
            let normalized = x[i].clamp(0.0, 1.0);
            let (min, max) = self.bounds[i];
            let value_raw = min + normalized * (max - min);
            let value = match &self.param_space[name] {
                HyperparamSpace::Continuous { .. } => HyperparamValue::Float(value_raw),
                HyperparamSpace::LogUniform { .. } => HyperparamValue::Float(value_raw.exp()),
                HyperparamSpace::IntRange { .. } => HyperparamValue::Int(value_raw.round() as i64),
                HyperparamSpace::Discrete(values) => {
                    let idx = value_raw.round() as usize;
                    values[idx.min(values.len() - 1)].clone()
                }
            };
            config.insert(name.clone(), value);
        }
        Ok(config)
    }

    /// Generate a random sample from parameter space.
    fn random_sample(&mut self) -> HyperparamConfig {
        let mut config = HashMap::new();
        for (name, space) in &self.param_space {
            let value = space.sample(&mut self.rng);
            config.insert(name.clone(), value);
        }
        config
    }

    /// Add a result from evaluating a configuration.
    pub fn add_result(&mut self, result: HyperparamResult) {
        self.results.push(result);
    }

    /// Get the best result found so far.
    pub fn best_result(&self) -> Option<&HyperparamResult> {
        self.results.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get all results sorted by score (descending).
    pub fn sorted_results(&self) -> Vec<&HyperparamResult> {
        let mut results: Vec<&HyperparamResult> = self.results.iter().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Get all results.
    pub fn results(&self) -> &[HyperparamResult] {
        &self.results
    }

    /// Check if optimization is complete.
    pub fn is_complete(&self) -> bool {
        self.results.len() >= self.n_iterations + self.n_initial_points
    }

    /// Get current iteration number.
    pub fn current_iteration(&self) -> usize {
        self.results.len()
    }

    /// Get total budget (initial + iterations).
    pub fn total_budget(&self) -> usize {
        self.n_iterations + self.n_initial_points
    }
}
