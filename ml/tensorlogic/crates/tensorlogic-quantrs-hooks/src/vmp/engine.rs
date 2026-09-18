//! Variational Message Passing engine.
//!
//! Implements Winn & Bishop (2005) VMP for conjugate-exponential models built on
//! three families: Gaussian (mean-unknown / precision-known), Categorical, and
//! Dirichlet. The engine consumes a structural description via [`VmpConfig`] and
//! runs coordinate-ascent natural-parameter updates until the ELBO / L∞ residual
//! converges.
//!
//! The algorithm is independent of the discrete factor-potential tables carried
//! by the `FactorGraph` type: VMP operates purely in continuous natural-parameter
//! space. The user is therefore required to annotate each variable with its
//! family and each factor with its conjugate role (see [`VmpFactor`]).
//!
//! # High-level flow
//!
//! 1. Initialise each variable's variational distribution `q(v)` from its prior.
//! 2. For each iteration:
//!    - For every variable `v` in a deterministic order:
//!      - Accumulate contributions from every adjacent factor (natural-parameter
//!        deltas).
//!      - Replace `q(v)`'s natural parameters by `prior_nat + Σ Δ`.
//!    - Compute the ELBO.
//! 3. Stop when |ΔELBO| < ε or the maximum natural-parameter residual < ε.
//!
//! Divergence (ELBO decreasing by more than a small tolerance) is detected and
//! surfaced as a `ConvergenceFailure` error so the caller does not silently
//! consume a broken result.

use std::collections::HashMap;

use crate::error::{PgmError, Result};
use crate::graph::FactorGraph;

use super::distributions::{
    categorical_kl, dirichlet_kl, gaussian_kl, CategoricalNP, DirichletNP, GaussianNP,
};
use super::exponential_family::ExponentialFamily;

// ---------------------------------------------------------------------------
// Family tagging
// ---------------------------------------------------------------------------

/// Variational family assigned to a variable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// Univariate Gaussian with known precision (only mean is random).
    Gaussian,
    /// Categorical over `k` categories.
    Categorical,
    /// Dirichlet over `k` components (the conjugate prior for Categorical).
    Dirichlet,
}

/// Variational state carried for a single variable.
#[derive(Clone, Debug)]
pub enum VariationalState {
    /// Gaussian variable (mean unknown, precision fixed).
    Gaussian { q: GaussianNP, prior: GaussianNP },
    /// Categorical variable with a Dirichlet-prior parent.
    Categorical {
        q: CategoricalNP,
        prior: CategoricalNP,
    },
    /// Dirichlet variable.
    Dirichlet { q: DirichletNP, prior: DirichletNP },
}

impl VariationalState {
    /// Family tag.
    pub fn family(&self) -> Family {
        match self {
            Self::Gaussian { .. } => Family::Gaussian,
            Self::Categorical { .. } => Family::Categorical,
            Self::Dirichlet { .. } => Family::Dirichlet,
        }
    }

    /// Current natural parameter vector.
    pub fn natural_params(&self) -> Vec<f64> {
        match self {
            Self::Gaussian { q, .. } => q.natural_params(),
            Self::Categorical { q, .. } => q.natural_params(),
            Self::Dirichlet { q, .. } => q.natural_params(),
        }
    }

    /// Current entropy `H(q)`.
    pub fn entropy(&self) -> Result<f64> {
        match self {
            Self::Gaussian { q, .. } => q.entropy(),
            Self::Categorical { q, .. } => q.entropy(),
            Self::Dirichlet { q, .. } => q.entropy(),
        }
    }

    /// KL from the current variational posterior to its prior.
    pub fn kl_to_prior(&self) -> Result<f64> {
        match self {
            Self::Gaussian { q, prior } => gaussian_kl(q, prior),
            Self::Categorical { q, prior } => categorical_kl(q, prior),
            Self::Dirichlet { q, prior } => dirichlet_kl(q, prior),
        }
    }
}

// ---------------------------------------------------------------------------
// Factor tagging
// ---------------------------------------------------------------------------

/// Conjugate relationship represented by a single factor.
#[derive(Clone, Debug)]
pub enum VmpFactor {
    /// Gaussian observation with known precision, centred on another Gaussian
    /// variable. Produces a natural-parameter delta of
    /// `η_delta = [τ_obs · y]` with posterior precision contribution `τ_obs`.
    GaussianObservation {
        /// Variable whose mean is being inferred.
        target: String,
        /// Observed value y.
        observation: f64,
        /// Known observation precision τ_obs.
        precision: f64,
    },
    /// Gaussian `x_child ~ N(x_parent, 1/τ)` — a "Gaussian step" between two
    /// unknown means sharing a known precision. Both endpoints receive a
    /// symmetric natural-parameter delta driven by the other's expected mean.
    GaussianStep {
        /// Endpoint 1 variable name.
        lhs: String,
        /// Endpoint 2 variable name.
        rhs: String,
        /// Known precision τ.
        precision: f64,
    },
    /// `x ~ Categorical(π)` where π is itself a Dirichlet variable. Updates the
    /// Dirichlet concentration by `E_q[u(x)] = softmax(η_x)` and contributes
    /// `E_q[log π]` back to the Categorical natural parameters.
    DirichletCategorical {
        /// Dirichlet-distributed variable π.
        dirichlet: String,
        /// Categorical-distributed variable x.
        categorical: String,
    },
    /// Observed categorical value (evidence). Only contributes counts to its
    /// Dirichlet parent; the categorical itself is pinned.
    CategoricalObservation {
        /// Dirichlet-distributed variable π.
        dirichlet: String,
        /// Observed category index.
        observation: usize,
        /// Number of categories.
        num_categories: usize,
    },
}

// ---------------------------------------------------------------------------
// Engine configuration
// ---------------------------------------------------------------------------

/// User-facing configuration describing a VMP problem.
#[derive(Clone, Debug, Default)]
pub struct VmpConfig {
    /// Per-variable variational state.
    pub states: HashMap<String, VariationalState>,
    /// Factors (conjugate relationships).
    pub factors: Vec<VmpFactor>,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on both ELBO change and the max L∞ residual.
    pub tolerance: f64,
    /// Maximum allowed ELBO decrease before the engine bails out with a
    /// `ConvergenceFailure` error (guards against numerical divergence).
    pub divergence_tolerance: f64,
}

impl VmpConfig {
    /// Build an empty configuration with sensible defaults.
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            factors: Vec::new(),
            max_iterations: 100,
            tolerance: 1e-6,
            divergence_tolerance: 1e-4,
        }
    }

    /// Register a Gaussian variable with a prior `N(prior_mean, 1/precision)`.
    pub fn with_gaussian(mut self, name: &str, prior_mean: f64, precision: f64) -> Result<Self> {
        let prior = GaussianNP::new(prior_mean, precision)?;
        let q = prior.clone();
        self.states
            .insert(name.to_string(), VariationalState::Gaussian { q, prior });
        Ok(self)
    }

    /// Register a Categorical variable with a flat prior over `k` categories.
    pub fn with_categorical(mut self, name: &str, num_categories: usize) -> Result<Self> {
        if num_categories == 0 {
            return Err(PgmError::InvalidDistribution(
                "Categorical needs at least one category".to_string(),
            ));
        }
        let probs = vec![1.0 / num_categories as f64; num_categories];
        let prior = CategoricalNP::from_probs(&probs)?;
        let q = prior.clone();
        self.states
            .insert(name.to_string(), VariationalState::Categorical { q, prior });
        Ok(self)
    }

    /// Register a Dirichlet variable with prior concentration α.
    pub fn with_dirichlet(mut self, name: &str, concentration: Vec<f64>) -> Result<Self> {
        let prior = DirichletNP::new(concentration)?;
        let q = prior.clone();
        self.states
            .insert(name.to_string(), VariationalState::Dirichlet { q, prior });
        Ok(self)
    }

    /// Append a VMP factor.
    pub fn with_factor(mut self, factor: VmpFactor) -> Self {
        self.factors.push(factor);
        self
    }

    /// Override the max iterations / tolerance pair.
    pub fn with_limits(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }

    /// Ensure every variable appearing in a factor is registered with a family
    /// and that its family matches the factor's expectation.
    fn validate(&self) -> Result<()> {
        for f in &self.factors {
            match f {
                VmpFactor::GaussianObservation { target, .. } => {
                    let state = self
                        .states
                        .get(target)
                        .ok_or_else(|| PgmError::VariableNotFound(target.clone()))?;
                    if !matches!(state, VariationalState::Gaussian { .. }) {
                        return Err(PgmError::InvalidGraph(format!(
                            "GaussianObservation on non-Gaussian variable '{}'",
                            target
                        )));
                    }
                }
                VmpFactor::GaussianStep { lhs, rhs, .. } => {
                    for v in [lhs, rhs] {
                        let state = self
                            .states
                            .get(v)
                            .ok_or_else(|| PgmError::VariableNotFound(v.clone()))?;
                        if !matches!(state, VariationalState::Gaussian { .. }) {
                            return Err(PgmError::InvalidGraph(format!(
                                "GaussianStep on non-Gaussian variable '{}'",
                                v
                            )));
                        }
                    }
                }
                VmpFactor::DirichletCategorical {
                    dirichlet,
                    categorical,
                } => {
                    let d = self
                        .states
                        .get(dirichlet)
                        .ok_or_else(|| PgmError::VariableNotFound(dirichlet.clone()))?;
                    let c = self
                        .states
                        .get(categorical)
                        .ok_or_else(|| PgmError::VariableNotFound(categorical.clone()))?;
                    if !matches!(d, VariationalState::Dirichlet { .. }) {
                        return Err(PgmError::InvalidGraph(format!(
                            "DirichletCategorical: '{}' is not a Dirichlet variable",
                            dirichlet
                        )));
                    }
                    if !matches!(c, VariationalState::Categorical { .. }) {
                        return Err(PgmError::InvalidGraph(format!(
                            "DirichletCategorical: '{}' is not a Categorical variable",
                            categorical
                        )));
                    }
                }
                VmpFactor::CategoricalObservation {
                    dirichlet,
                    num_categories,
                    observation,
                } => {
                    let d = self
                        .states
                        .get(dirichlet)
                        .ok_or_else(|| PgmError::VariableNotFound(dirichlet.clone()))?;
                    match d {
                        VariationalState::Dirichlet { q, .. } => {
                            if q.concentration.len() != *num_categories {
                                return Err(PgmError::DimensionMismatch {
                                    expected: vec![*num_categories],
                                    got: vec![q.concentration.len()],
                                });
                            }
                            if observation >= num_categories {
                                return Err(PgmError::InvalidDistribution(format!(
                                    "observation {} out of range for {} categories",
                                    observation, num_categories
                                )));
                            }
                        }
                        _ => {
                            return Err(PgmError::InvalidGraph(format!(
                                "CategoricalObservation: '{}' is not a Dirichlet variable",
                                dirichlet
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Summary of a VMP run.
#[derive(Clone, Debug)]
pub struct VmpResult {
    /// Final variational distributions, keyed by variable name.
    pub states: HashMap<String, VariationalState>,
    /// ELBO value at each iteration (length = iterations run + 1).
    pub elbo_history: Vec<f64>,
    /// Iterations actually run.
    pub iterations: usize,
    /// Whether the run met the tolerance criterion.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// The VMP coordinate-ascent engine.
///
/// The engine itself is a thin state machine that holds the [`VmpConfig`] and
/// drives the per-variable update rules defined by each `VmpFactor` variant.
pub struct VariationalMessagePassing {
    config: VmpConfig,
    update_order: Vec<String>,
}

impl VariationalMessagePassing {
    /// Build an engine from a validated configuration.
    pub fn new(config: VmpConfig) -> Result<Self> {
        config.validate()?;
        let mut keys: Vec<String> = config.states.keys().cloned().collect();
        keys.sort(); // deterministic update order
        Ok(Self {
            config,
            update_order: keys,
        })
    }

    /// Construct an engine from an already-existing `FactorGraph` (structure
    /// only) plus the VMP annotations. The graph is consulted to validate that
    /// every factor references variables that the user *also* registered with
    /// the factor-graph API, which is useful when VMP is layered on top of the
    /// generic PGM pipeline.
    pub fn with_graph(graph: &FactorGraph, config: VmpConfig) -> Result<Self> {
        for v in config.states.keys() {
            if graph.get_variable(v).is_none() {
                return Err(PgmError::VariableNotFound(format!(
                    "'{}' declared in VmpConfig but missing from FactorGraph",
                    v
                )));
            }
        }
        Self::new(config)
    }

    /// Run the coordinate-ascent loop.
    pub fn run(&mut self) -> Result<VmpResult> {
        let elbo0 = self.compute_elbo()?;
        let mut elbo_history = vec![elbo0];
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            let snapshot = self.snapshot_natural_params();
            self.coordinate_sweep()?;
            let elbo_new = self.compute_elbo()?;
            let prev = *elbo_history
                .last()
                .ok_or_else(|| PgmError::ConvergenceFailure("elbo history is empty".into()))?;

            // Divergence check: the ELBO is guaranteed to be non-decreasing for
            // exact conjugate VMP, so a drop larger than the divergence
            // tolerance is a red flag (numerical breakdown or ill-posed model).
            if elbo_new < prev - self.config.divergence_tolerance {
                return Err(PgmError::ConvergenceFailure(format!(
                    "VMP ELBO decreased from {} to {} at iteration {}",
                    prev, elbo_new, iter
                )));
            }

            elbo_history.push(elbo_new);
            iterations = iter + 1;

            let linf = self.linf_from_snapshot(&snapshot);
            let elbo_delta = (elbo_new - prev).abs();
            if elbo_delta < self.config.tolerance || linf < self.config.tolerance {
                converged = true;
                break;
            }
        }

        Ok(VmpResult {
            states: self.config.states.clone(),
            elbo_history,
            iterations,
            converged,
        })
    }

    /// Single coordinate sweep across all variables in deterministic order.
    fn coordinate_sweep(&mut self) -> Result<()> {
        // We iterate by name over a cloned order so that the engine can mutate
        // `self.config.states` freely inside the loop.
        let order = self.update_order.clone();
        for var in order {
            self.update_variable(&var)?;
        }
        Ok(())
    }

    /// Compute the natural-parameter update for one variable by aggregating the
    /// contributions of every adjacent factor.
    fn update_variable(&mut self, var: &str) -> Result<()> {
        let state = self
            .config
            .states
            .get(var)
            .cloned()
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;
        match state.family() {
            Family::Gaussian => self.update_gaussian(var),
            Family::Categorical => self.update_categorical(var),
            Family::Dirichlet => self.update_dirichlet(var),
        }
    }

    fn update_gaussian(&mut self, var: &str) -> Result<()> {
        // Pull the prior natural parameters.
        let (mut posterior_precision, mut posterior_natural_mean) = match self
            .config
            .states
            .get(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?
        {
            VariationalState::Gaussian { prior, .. } => {
                // prior contributes η_prior = τ_prior · μ_prior and precision τ_prior.
                (prior.precision, prior.precision * prior.mean)
            }
            _ => unreachable!("non-Gaussian state reached update_gaussian"),
        };

        for factor in &self.config.factors {
            match factor {
                VmpFactor::GaussianObservation {
                    target,
                    observation,
                    precision,
                } if target == var => {
                    posterior_precision += precision;
                    posterior_natural_mean += precision * observation;
                }
                VmpFactor::GaussianStep {
                    lhs,
                    rhs,
                    precision,
                } => {
                    // Symmetric Gaussian step: each endpoint observes E[q(other)]
                    // with the given precision τ.
                    let (other, is_self) = if lhs == var {
                        (rhs, true)
                    } else if rhs == var {
                        (lhs, true)
                    } else {
                        (lhs, false)
                    };
                    if is_self {
                        let other_mean = match self
                            .config
                            .states
                            .get(other)
                            .ok_or_else(|| PgmError::VariableNotFound(other.clone()))?
                        {
                            VariationalState::Gaussian { q, .. } => q.mean,
                            _ => {
                                return Err(PgmError::InvalidGraph(format!(
                                    "GaussianStep neighbour '{}' is not Gaussian",
                                    other
                                )));
                            }
                        };
                        posterior_precision += precision;
                        posterior_natural_mean += precision * other_mean;
                    }
                }
                _ => {}
            }
        }

        if posterior_precision <= 0.0 || !posterior_precision.is_finite() {
            return Err(PgmError::InvalidDistribution(format!(
                "Gaussian posterior precision must be positive (got {})",
                posterior_precision
            )));
        }

        let new_mean = posterior_natural_mean / posterior_precision;
        let state = self
            .config
            .states
            .get_mut(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;
        if let VariationalState::Gaussian { q, .. } = state {
            // The stored q carries the *effective* posterior precision derived
            // from the prior plus every adjacent observation / step factor; the
            // prior's precision is preserved separately, which is what keeps
            // the KL to the prior well defined (see `VariationalState::kl_to_prior`).
            q.precision = posterior_precision;
            q.mean = new_mean;
        }
        Ok(())
    }

    fn update_categorical(&mut self, var: &str) -> Result<()> {
        let num_categories = match self
            .config
            .states
            .get(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?
        {
            VariationalState::Categorical { q, .. } => q.num_categories(),
            _ => unreachable!(),
        };

        // Start from the prior natural parameters.
        let mut natural = match self
            .config
            .states
            .get(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?
        {
            VariationalState::Categorical { prior, .. } => prior.natural_params(),
            _ => unreachable!(),
        };

        for factor in &self.config.factors {
            if let VmpFactor::DirichletCategorical {
                dirichlet,
                categorical,
            } = factor
            {
                if categorical == var {
                    let dir_state = self
                        .config
                        .states
                        .get(dirichlet)
                        .ok_or_else(|| PgmError::VariableNotFound(dirichlet.clone()))?;
                    if let VariationalState::Dirichlet { q, .. } = dir_state {
                        let e_log_pi = q.expected_sufficient_statistics();
                        if e_log_pi.len() != num_categories {
                            return Err(PgmError::DimensionMismatch {
                                expected: vec![num_categories],
                                got: vec![e_log_pi.len()],
                            });
                        }
                        for (a, b) in natural.iter_mut().zip(e_log_pi.iter()) {
                            *a += *b;
                        }
                    }
                }
            }
        }

        let state = self
            .config
            .states
            .get_mut(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;
        if let VariationalState::Categorical { q, .. } = state {
            q.set_natural(&natural)?;
        }
        Ok(())
    }

    fn update_dirichlet(&mut self, var: &str) -> Result<()> {
        // Start from prior natural parameters.
        let mut natural = match self
            .config
            .states
            .get(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?
        {
            VariationalState::Dirichlet { prior, .. } => prior.natural_params(),
            _ => unreachable!(),
        };

        let num_components = natural.len();
        for factor in &self.config.factors {
            match factor {
                VmpFactor::DirichletCategorical {
                    dirichlet,
                    categorical,
                } if dirichlet == var => {
                    let cat_state = self
                        .config
                        .states
                        .get(categorical)
                        .ok_or_else(|| PgmError::VariableNotFound(categorical.clone()))?;
                    if let VariationalState::Categorical { q, .. } = cat_state {
                        let expected_counts = q.expected_sufficient_statistics();
                        if expected_counts.len() != num_components {
                            return Err(PgmError::DimensionMismatch {
                                expected: vec![num_components],
                                got: vec![expected_counts.len()],
                            });
                        }
                        for (a, b) in natural.iter_mut().zip(expected_counts.iter()) {
                            *a += *b;
                        }
                    }
                }
                VmpFactor::CategoricalObservation {
                    dirichlet,
                    observation,
                    num_categories,
                } if dirichlet == var => {
                    if *num_categories != num_components {
                        return Err(PgmError::DimensionMismatch {
                            expected: vec![num_components],
                            got: vec![*num_categories],
                        });
                    }
                    if let Some(slot) = natural.get_mut(*observation) {
                        *slot += 1.0;
                    } else {
                        return Err(PgmError::InvalidDistribution(format!(
                            "observation {} out of range for {} categories",
                            observation, num_categories
                        )));
                    }
                }
                _ => {}
            }
        }

        let state = self
            .config
            .states
            .get_mut(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;
        if let VariationalState::Dirichlet { q, .. } = state {
            q.set_natural(&natural)?;
        }
        Ok(())
    }

    // ---------------------------------------------------------------------
    // ELBO
    // ---------------------------------------------------------------------

    /// Evidence Lower Bound `L(q) = E_q[log p(x, z)] − E_q[log q(z)]`.
    ///
    /// For the three conjugate relationships shipped in v0.2.0 the ELBO
    /// decomposes as `Σ E_q[log p(factor)] − Σ KL(q(v) || prior(v))` because
    /// each prior cancels with the log p(z) term.
    pub fn compute_elbo(&self) -> Result<f64> {
        let mut elbo = 0.0;
        // Likelihood contributions from each factor.
        for factor in &self.config.factors {
            elbo += self.factor_expected_log_joint(factor)?;
        }
        // − KL(q(v) || prior(v)) for every variable.
        for state in self.config.states.values() {
            elbo -= state.kl_to_prior()?;
        }
        Ok(elbo)
    }

    fn factor_expected_log_joint(&self, factor: &VmpFactor) -> Result<f64> {
        match factor {
            VmpFactor::GaussianObservation {
                target,
                observation,
                precision,
            } => {
                let state = self
                    .config
                    .states
                    .get(target)
                    .ok_or_else(|| PgmError::VariableNotFound(target.clone()))?;
                if let VariationalState::Gaussian { q, .. } = state {
                    // E_q[log N(y | μ, 1/τ)] = ½ log(τ / 2π) − (τ/2) (E[μ²] + y² − 2 y E[μ]).
                    // For q with precision τ_q (posterior effective), E[μ] = μ_q, Var[μ] = 1/τ_q.
                    let e_mu = q.mean;
                    let e_mu2 = q.mean * q.mean + 1.0 / q.precision;
                    let y = *observation;
                    let p = *precision;
                    let coef = 0.5 * p;
                    let log_norm = 0.5 * (p / (2.0 * std::f64::consts::PI)).ln();
                    Ok(log_norm - coef * (e_mu2 + y * y - 2.0 * y * e_mu))
                } else {
                    Err(PgmError::InvalidGraph(format!(
                        "GaussianObservation target '{}' is not Gaussian",
                        target
                    )))
                }
            }
            VmpFactor::GaussianStep {
                lhs,
                rhs,
                precision,
            } => {
                let lq = match self
                    .config
                    .states
                    .get(lhs)
                    .ok_or_else(|| PgmError::VariableNotFound(lhs.clone()))?
                {
                    VariationalState::Gaussian { q, .. } => q,
                    _ => {
                        return Err(PgmError::InvalidGraph(format!(
                            "GaussianStep endpoint '{}' is not Gaussian",
                            lhs
                        )));
                    }
                };
                let rq = match self
                    .config
                    .states
                    .get(rhs)
                    .ok_or_else(|| PgmError::VariableNotFound(rhs.clone()))?
                {
                    VariationalState::Gaussian { q, .. } => q,
                    _ => {
                        return Err(PgmError::InvalidGraph(format!(
                            "GaussianStep endpoint '{}' is not Gaussian",
                            rhs
                        )));
                    }
                };
                // E_q[log N(lhs | rhs, 1/τ)]
                let e_l = lq.mean;
                let e_l2 = lq.mean * lq.mean + 1.0 / lq.precision;
                let e_r = rq.mean;
                let e_r2 = rq.mean * rq.mean + 1.0 / rq.precision;
                let log_norm = 0.5 * (precision / (2.0 * std::f64::consts::PI)).ln();
                let coef = 0.5 * precision;
                Ok(log_norm - coef * (e_l2 - 2.0 * e_l * e_r + e_r2))
            }
            VmpFactor::DirichletCategorical {
                dirichlet,
                categorical,
            } => {
                let d = match self
                    .config
                    .states
                    .get(dirichlet)
                    .ok_or_else(|| PgmError::VariableNotFound(dirichlet.clone()))?
                {
                    VariationalState::Dirichlet { q, .. } => q,
                    _ => {
                        return Err(PgmError::InvalidGraph(format!(
                            "DirichletCategorical: '{}' not Dirichlet",
                            dirichlet
                        )));
                    }
                };
                let c = match self
                    .config
                    .states
                    .get(categorical)
                    .ok_or_else(|| PgmError::VariableNotFound(categorical.clone()))?
                {
                    VariationalState::Categorical { q, .. } => q,
                    _ => {
                        return Err(PgmError::InvalidGraph(format!(
                            "DirichletCategorical: '{}' not Categorical",
                            categorical
                        )));
                    }
                };
                let e_log_pi = d.expected_sufficient_statistics();
                let probs = c.probs();
                if e_log_pi.len() != probs.len() {
                    return Err(PgmError::DimensionMismatch {
                        expected: vec![probs.len()],
                        got: vec![e_log_pi.len()],
                    });
                }
                Ok(e_log_pi.iter().zip(probs.iter()).map(|(l, p)| l * p).sum())
            }
            VmpFactor::CategoricalObservation {
                dirichlet,
                observation,
                ..
            } => {
                let d = match self
                    .config
                    .states
                    .get(dirichlet)
                    .ok_or_else(|| PgmError::VariableNotFound(dirichlet.clone()))?
                {
                    VariationalState::Dirichlet { q, .. } => q,
                    _ => {
                        return Err(PgmError::InvalidGraph(format!(
                            "CategoricalObservation: '{}' not Dirichlet",
                            dirichlet
                        )));
                    }
                };
                let e_log_pi = d.expected_sufficient_statistics();
                e_log_pi.get(*observation).cloned().ok_or_else(|| {
                    PgmError::InvalidDistribution(format!(
                        "observation {} out of range for Dirichlet with {} components",
                        observation,
                        e_log_pi.len()
                    ))
                })
            }
        }
    }

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    fn snapshot_natural_params(&self) -> HashMap<String, Vec<f64>> {
        self.config
            .states
            .iter()
            .map(|(k, v)| (k.clone(), v.natural_params()))
            .collect()
    }

    fn linf_from_snapshot(&self, snapshot: &HashMap<String, Vec<f64>>) -> f64 {
        let mut max = 0.0f64;
        for (k, v) in &self.config.states {
            let before = match snapshot.get(k) {
                Some(vec) => vec,
                None => continue,
            };
            for (a, b) in v.natural_params().iter().zip(before.iter()) {
                max = max.max((a - b).abs());
            }
        }
        max
    }

    /// Read-only access to the current states.
    pub fn states(&self) -> &HashMap<String, VariationalState> {
        &self.config.states
    }
}
