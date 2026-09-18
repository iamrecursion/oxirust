//! Observational data container and result types for causal inference.
//!
//! Defines [`ObservationalData`] plus the small result/spec types
//! [`Intervention`], [`TreatmentEffect`], and [`BackdoorAdjustment`].

use super::error::CausalError;

// ---------------------------------------------------------------------------
// ObservationalData
// ---------------------------------------------------------------------------

/// Container for observational (non-interventional) data.
///
/// Data is stored as a matrix: `samples[i]` is the i-th observation,
/// with one entry per variable in the same order as `variables`.
#[derive(Debug, Clone)]
pub struct ObservationalData {
    variables: Vec<String>,
    samples: Vec<Vec<f64>>,
}

impl ObservationalData {
    /// Create an empty dataset with the given variable names.
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            variables,
            samples: Vec::new(),
        }
    }

    /// Add a single observation. Returns an error if the dimension does not match.
    pub fn add_sample(&mut self, sample: Vec<f64>) -> Result<(), CausalError> {
        if sample.len() != self.variables.len() {
            return Err(CausalError::DimensionMismatch);
        }
        self.samples.push(sample);
        Ok(())
    }

    /// Number of observations.
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Number of variables.
    pub fn n_variables(&self) -> usize {
        self.variables.len()
    }

    /// Return the column index for a variable name.
    pub(super) fn var_index(&self, var: &str) -> Option<usize> {
        self.variables.iter().position(|v| v == var)
    }

    /// Extract all values for a single variable.
    pub fn column(&self, var: &str) -> Option<Vec<f64>> {
        let idx = self.var_index(var)?;
        Some(self.samples.iter().map(|s| s[idx]).collect())
    }

    /// Compute the marginal mean of a variable.
    pub fn mean(&self, var: &str) -> Option<f64> {
        let col = self.column(var)?;
        if col.is_empty() {
            return None;
        }
        Some(col.iter().sum::<f64>() / col.len() as f64)
    }

    /// Compute the mean of `outcome` conditioned on `condition_var == condition_val`.
    ///
    /// Equality is checked with a small tolerance (1e-9) to handle floating-point values.
    pub fn conditional_mean(
        &self,
        outcome: &str,
        condition_var: &str,
        condition_val: f64,
    ) -> Option<f64> {
        let out_idx = self.var_index(outcome)?;
        let cond_idx = self.var_index(condition_var)?;
        let filtered: Vec<f64> = self
            .samples
            .iter()
            .filter(|s| (s[cond_idx] - condition_val).abs() < 1e-9)
            .map(|s| s[out_idx])
            .collect();
        if filtered.is_empty() {
            return None;
        }
        Some(filtered.iter().sum::<f64>() / filtered.len() as f64)
    }

    /// Return a reference to the variable names.
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Return all samples as a slice of rows.
    pub fn samples(&self) -> &[Vec<f64>] {
        &self.samples
    }
}

// ---------------------------------------------------------------------------
// Intervention / TreatmentEffect / BackdoorAdjustment
// ---------------------------------------------------------------------------

/// A do-calculus intervention: fix variable `variable` to `value`.
#[derive(Debug, Clone)]
pub struct Intervention {
    /// The name of the intervened-upon variable.
    pub variable: String,
    /// The value to which the variable is set.
    pub value: f64,
}

/// Result of an average treatment effect estimation.
#[derive(Debug, Clone)]
pub struct TreatmentEffect {
    /// Average treatment effect: E[Y | do(T=1)] − E[Y | do(T=0)].
    pub ate: f64,
    /// Average treatment effect on the treated subgroup (ATT).
    pub ate_treated: f64,
    /// Average treatment effect on the control subgroup (ATC).
    pub ate_control: f64,
    /// Estimation method used: `"backdoor"`, `"frontdoor"`, or `"iv"`.
    pub estimator: String,
    /// Number of samples used.
    pub n_samples: usize,
    /// Bootstrap 95% confidence interval, if computed.
    pub confidence_interval: Option<(f64, f64)>,
}

/// Outcome of a backdoor adjustment set search.
#[derive(Debug, Clone)]
pub struct BackdoorAdjustment {
    /// The chosen adjustment set (variable names).
    pub adjustment_set: Vec<String>,
    /// Whether the set satisfies the backdoor criterion.
    pub valid: bool,
}
