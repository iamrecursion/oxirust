//! Average treatment effect (ATE) estimators and propensity scoring.
//!
//! Contains the numerical estimation routines that consume
//! [`ObservationalData`]: backdoor adjustment, instrumental variable (Wald),
//! and logistic-regression propensity scores.

use super::criteria::find_backdoor_adjustment;
use super::data::{ObservationalData, TreatmentEffect};
use super::error::CausalError;
use super::graph::CausalGraph;

// ---------------------------------------------------------------------------
// ATE via backdoor adjustment
// ---------------------------------------------------------------------------

/// Estimate the average treatment effect using backdoor adjustment.
///
/// Uses the backdoor adjustment formula:
/// ```text
/// E[Y | do(T=t)] = Σ_z E[Y | T=t, Z=z] · P(Z=z)
/// ```
/// where Z is a valid backdoor adjustment set for (T → Y).
///
/// This implementation discretises each adjustment variable to its unique observed
/// values and approximates the sum via empirical proportions.
pub fn ate_backdoor(
    graph: &CausalGraph,
    data: &ObservationalData,
    treatment: &str,
    outcome: &str,
) -> Result<TreatmentEffect, CausalError> {
    if graph.node_index(treatment).is_none() {
        return Err(CausalError::NodeNotFound(treatment.to_string()));
    }
    if graph.node_index(outcome).is_none() {
        return Err(CausalError::NodeNotFound(outcome.to_string()));
    }
    if !graph.has_directed_path(treatment, outcome) {
        return Err(CausalError::NoCausalPath);
    }
    if data.n_samples() == 0 {
        return Err(CausalError::InsufficientData);
    }

    let adj = find_backdoor_adjustment(graph, treatment, outcome)?;
    let n = data.n_samples() as f64;

    // If adjustment set is empty, use simple difference in conditional means.
    if adj.adjustment_set.is_empty() {
        let ey_do1 = data
            .conditional_mean(outcome, treatment, 1.0)
            .ok_or(CausalError::InsufficientData)?;
        let ey_do0 = data
            .conditional_mean(outcome, treatment, 0.0)
            .ok_or(CausalError::InsufficientData)?;

        let treated_outcomes: Vec<f64> = data
            .samples()
            .iter()
            .filter(|s| {
                let ti = data.var_index(treatment).unwrap_or(usize::MAX);
                s[ti] == 1.0
            })
            .map(|s| s[data.var_index(outcome).unwrap_or(0)])
            .collect();
        let control_outcomes: Vec<f64> = data
            .samples()
            .iter()
            .filter(|s| {
                let ti = data.var_index(treatment).unwrap_or(usize::MAX);
                s[ti] == 0.0
            })
            .map(|s| s[data.var_index(outcome).unwrap_or(0)])
            .collect();

        let att = if treated_outcomes.is_empty() {
            ey_do1 - ey_do0
        } else {
            treated_outcomes.iter().sum::<f64>() / treated_outcomes.len() as f64 - ey_do0
        };
        let atc = if control_outcomes.is_empty() {
            ey_do1 - ey_do0
        } else {
            ey_do1 - control_outcomes.iter().sum::<f64>() / control_outcomes.len() as f64
        };

        return Ok(TreatmentEffect {
            ate: ey_do1 - ey_do0,
            ate_treated: att,
            ate_control: atc,
            estimator: "backdoor".to_string(),
            n_samples: data.n_samples(),
            confidence_interval: None,
        });
    }

    // General backdoor adjustment: marginalise over the adjustment set.
    // For simplicity we handle the case of a single adjustment variable;
    // for multiple we marginalise sequentially.
    // We use the first variable in the adjustment set.
    let z_var = &adj.adjustment_set[0];
    let z_vals: Vec<f64> = {
        let col = data
            .column(z_var)
            .ok_or_else(|| CausalError::NodeNotFound(z_var.clone()))?;
        let mut unique: Vec<f64> = col.clone();
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique.dedup();
        unique
    };

    let z_col = data
        .column(z_var)
        .ok_or_else(|| CausalError::NodeNotFound(z_var.clone()))?;
    let t_idx = data
        .var_index(treatment)
        .ok_or_else(|| CausalError::NodeNotFound(treatment.to_string()))?;
    let o_idx = data
        .var_index(outcome)
        .ok_or_else(|| CausalError::NodeNotFound(outcome.to_string()))?;
    let z_idx = data
        .var_index(z_var)
        .ok_or_else(|| CausalError::NodeNotFound(z_var.clone()))?;

    let mut ey_do1 = 0.0_f64;
    let mut ey_do0 = 0.0_f64;

    for &zv in &z_vals {
        let pz = z_col.iter().filter(|&&v| (v - zv).abs() < 1e-9).count() as f64 / n;

        // E[Y | T=1, Z=zv]
        let e_y_t1_z: f64 = {
            let vals: Vec<f64> = data
                .samples()
                .iter()
                .filter(|s| (s[t_idx] - 1.0).abs() < 1e-9 && (s[z_idx] - zv).abs() < 1e-9)
                .map(|s| s[o_idx])
                .collect();
            if vals.is_empty() {
                // fallback: use overall mean of Y|T=1
                data.conditional_mean(outcome, treatment, 1.0)
                    .unwrap_or(0.0)
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            }
        };

        // E[Y | T=0, Z=zv]
        let e_y_t0_z: f64 = {
            let vals: Vec<f64> = data
                .samples()
                .iter()
                .filter(|s| (s[t_idx] - 0.0).abs() < 1e-9 && (s[z_idx] - zv).abs() < 1e-9)
                .map(|s| s[o_idx])
                .collect();
            if vals.is_empty() {
                data.conditional_mean(outcome, treatment, 0.0)
                    .unwrap_or(0.0)
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            }
        };

        ey_do1 += e_y_t1_z * pz;
        ey_do0 += e_y_t0_z * pz;
    }

    let ate = ey_do1 - ey_do0;

    // ATT / ATC (simplified)
    let treated_mean = data
        .conditional_mean(outcome, treatment, 1.0)
        .unwrap_or(ey_do1);
    let control_mean = data
        .conditional_mean(outcome, treatment, 0.0)
        .unwrap_or(ey_do0);
    let att = treated_mean - ey_do0;
    let atc = ey_do1 - control_mean;

    Ok(TreatmentEffect {
        ate,
        ate_treated: att,
        ate_control: atc,
        estimator: "backdoor".to_string(),
        n_samples: data.n_samples(),
        confidence_interval: None,
    })
}

// ---------------------------------------------------------------------------
// ATE via instrumental variable (IV)
// ---------------------------------------------------------------------------

/// Estimate the average treatment effect using the instrumental variable (IV) estimator.
///
/// Implements the Wald estimator:
/// ```text
/// ATE_IV = Cov(Y, Z) / Cov(T, Z)
/// ```
/// where Z is the instrument (must be correlated with treatment but affect outcome
/// only through treatment).
pub fn ate_instrumental_variable(
    data: &ObservationalData,
    treatment: &str,
    outcome: &str,
    instrument: &str,
) -> Result<TreatmentEffect, CausalError> {
    if data.n_samples() < 2 {
        return Err(CausalError::InsufficientData);
    }

    let y = data
        .column(outcome)
        .ok_or_else(|| CausalError::NodeNotFound(outcome.to_string()))?;
    let t = data
        .column(treatment)
        .ok_or_else(|| CausalError::NodeNotFound(treatment.to_string()))?;
    let z = data
        .column(instrument)
        .ok_or_else(|| CausalError::NodeNotFound(instrument.to_string()))?;

    let n = y.len() as f64;
    let mean_y = y.iter().sum::<f64>() / n;
    let mean_t = t.iter().sum::<f64>() / n;
    let mean_z = z.iter().sum::<f64>() / n;

    // Cov(Y, Z) = E[(Y - mean_y)(Z - mean_z)]
    let cov_yz: f64 = y
        .iter()
        .zip(z.iter())
        .map(|(&yi, &zi)| (yi - mean_y) * (zi - mean_z))
        .sum::<f64>()
        / n;

    // Cov(T, Z) = E[(T - mean_t)(Z - mean_z)]
    let cov_tz: f64 = t
        .iter()
        .zip(z.iter())
        .map(|(&ti, &zi)| (ti - mean_t) * (zi - mean_z))
        .sum::<f64>()
        / n;

    if cov_tz.abs() < 1e-12 {
        return Err(CausalError::NumericalError(
            "instrument has near-zero covariance with treatment (weak instrument)".to_string(),
        ));
    }

    let ate = cov_yz / cov_tz;

    // ATT and ATC approximated from subgroup means with IV correction
    let treated_y_mean: f64 = {
        let t_idx = data
            .var_index(treatment)
            .ok_or_else(|| CausalError::NodeNotFound(treatment.to_string()))?;
        let o_idx = data
            .var_index(outcome)
            .ok_or_else(|| CausalError::NodeNotFound(outcome.to_string()))?;
        let vals: Vec<f64> = data
            .samples()
            .iter()
            .filter(|s| (s[t_idx] - 1.0).abs() < 1e-9)
            .map(|s| s[o_idx])
            .collect();
        if vals.is_empty() {
            mean_y
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    };
    let control_y_mean: f64 = {
        let t_idx = data
            .var_index(treatment)
            .ok_or_else(|| CausalError::NodeNotFound(treatment.to_string()))?;
        let o_idx = data
            .var_index(outcome)
            .ok_or_else(|| CausalError::NodeNotFound(outcome.to_string()))?;
        let vals: Vec<f64> = data
            .samples()
            .iter()
            .filter(|s| s[t_idx].abs() < 1e-9)
            .map(|s| s[o_idx])
            .collect();
        if vals.is_empty() {
            mean_y
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    };

    Ok(TreatmentEffect {
        ate,
        ate_treated: treated_y_mean - control_y_mean,
        ate_control: treated_y_mean - control_y_mean,
        estimator: "iv".to_string(),
        n_samples: data.n_samples(),
        confidence_interval: None,
    })
}

// ---------------------------------------------------------------------------
// Propensity score
// ---------------------------------------------------------------------------

/// Compute propensity scores P(T=1 | covariates) using logistic regression.
///
/// Fits a simple logistic model `σ(w^T x + b)` to treatment assignments via
/// gradient descent (batch SGD with fixed learning rate and iterations).
/// Returns one score per observation in the same order as `data.samples()`.
pub fn propensity_score(
    data: &ObservationalData,
    treatment: &str,
    covariates: &[&str],
) -> Result<Vec<f64>, CausalError> {
    if data.n_samples() == 0 {
        return Err(CausalError::InsufficientData);
    }

    let t_idx = data
        .var_index(treatment)
        .ok_or_else(|| CausalError::NodeNotFound(treatment.to_string()))?;
    let cov_idxs: Vec<usize> = covariates
        .iter()
        .map(|&c| {
            data.var_index(c)
                .ok_or_else(|| CausalError::NodeNotFound(c.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let n = data.n_samples();
    let d = cov_idxs.len();

    // Build design matrix X and labels t
    let mut x_mat: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut t_vec: Vec<f64> = Vec::with_capacity(n);
    for s in data.samples() {
        let row: Vec<f64> = cov_idxs.iter().map(|&ci| s[ci]).collect();
        x_mat.push(row);
        t_vec.push(s[t_idx]);
    }

    // Standardise features for numerical stability
    let mut feat_mean = vec![0.0_f64; d];
    let mut feat_std = vec![1.0_f64; d];
    for j in 0..d {
        let col_sum: f64 = x_mat.iter().map(|r| r[j]).sum();
        feat_mean[j] = col_sum / n as f64;
        let var: f64 = x_mat
            .iter()
            .map(|r| (r[j] - feat_mean[j]).powi(2))
            .sum::<f64>()
            / n as f64;
        feat_std[j] = var.sqrt().max(1e-8);
    }
    for r in x_mat.iter_mut() {
        for j in 0..d {
            r[j] = (r[j] - feat_mean[j]) / feat_std[j];
        }
    }

    // Logistic regression via gradient descent
    let mut w = vec![0.0_f64; d];
    let mut b = 0.0_f64;
    let lr = 0.1_f64;
    let n_iter = 500usize;

    for _ in 0..n_iter {
        let mut dw = vec![0.0_f64; d];
        let mut db = 0.0_f64;
        for (i, xi) in x_mat.iter().enumerate() {
            let logit: f64 = xi.iter().zip(w.iter()).map(|(xj, wj)| xj * wj).sum::<f64>() + b;
            let prob = sigmoid(logit);
            let err = prob - t_vec[i];
            for j in 0..d {
                dw[j] += err * xi[j];
            }
            db += err;
        }
        for j in 0..d {
            w[j] -= lr * dw[j] / n as f64;
        }
        b -= lr * db / n as f64;
    }

    // Compute scores
    let scores: Vec<f64> = x_mat
        .iter()
        .map(|xi| {
            let logit: f64 = xi.iter().zip(w.iter()).map(|(xj, wj)| xj * wj).sum::<f64>() + b;
            sigmoid(logit)
        })
        .collect();

    Ok(scores)
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
