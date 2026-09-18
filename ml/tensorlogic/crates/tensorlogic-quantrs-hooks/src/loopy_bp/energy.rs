//! Bethe free energy calculation for converged LBP beliefs.

use scirs2_core::ndarray::{Array1, ArrayD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::graph::FactorGraph;

/// Bethe free energy components.
///
/// The Bethe approximation decomposes the global free energy into single-variable
/// and factor contributions.  At the fixed point of LBP, the Bethe free energy
/// equals the variational free energy under the Bethe approximation to the
/// belief propagation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BetheFreeEnergy {
    /// Factor-node energy term: ∑_a ∑_{x_a} b_a(x_a) ln[φ_a(x_a) / b_a(x_a)]
    pub factor_energy: f64,
    /// Variable-node entropy term: (1 - d_i) ∑_i ∑_{x_i} b_i(x_i) ln b_i(x_i)
    pub variable_entropy: f64,
    /// Total Bethe free energy = factor_energy + variable_entropy.
    pub total: f64,
    /// Approximate log-partition function: -F_Bethe.
    pub log_z: f64,
}

/// Compute the Bethe free energy from converged LBP beliefs.
///
/// `beliefs_var` is a map from variable name → marginal belief vector.
/// `beliefs_fac` is a map from factor id → joint belief tensor (over factor scope).
pub fn bethe_free_energy(
    graph: &FactorGraph,
    beliefs_var: &HashMap<String, Array1<f64>>,
    beliefs_fac: &HashMap<String, ArrayD<f64>>,
) -> BetheFreeEnergy {
    let eps = 1e-300_f64;

    // Factor-node contribution: ∑_a ∑_{x_a} b_a(x_a) [ln φ_a(x_a) - ln b_a(x_a)]
    let mut factor_energy = 0.0_f64;
    for (fac_id, fac_belief) in beliefs_fac {
        if let Some(factor) = graph.get_factor(fac_id) {
            for (b, phi) in fac_belief.iter().zip(factor.values.iter()) {
                if *b > eps {
                    let log_phi = if *phi > eps { phi.ln() } else { -700.0 };
                    factor_energy += b * (log_phi - b.ln());
                }
            }
        }
    }

    // Variable-node contribution: ∑_i (1 - d_i) ∑_{x_i} b_i(x_i) ln b_i(x_i)
    let mut variable_entropy = 0.0_f64;
    for (var_name, belief) in beliefs_var {
        // degree d_i = number of factors containing variable i
        let degree = graph
            .get_adjacent_factors(var_name)
            .map(|v| v.len())
            .unwrap_or(0) as f64;
        let entropy_i: f64 = belief
            .iter()
            .filter(|&&b| b > eps)
            .map(|&b| b * b.ln())
            .sum::<f64>();
        // variable term = -(1 - d_i) * H_i  where H_i = -∑ b ln b
        // ↔ (1 - d_i) * ∑ b ln b
        variable_entropy += (1.0 - degree) * entropy_i;
    }

    let total = -(factor_energy + variable_entropy);
    BetheFreeEnergy {
        factor_energy,
        variable_entropy,
        total,
        log_z: -total,
    }
}
