//! Natural-parameter messages for Variational Message Passing.
//!
//! In VMP every message carries a vector of *natural parameters* that can be
//! summed element-wise: the product of two exponential-family densities that
//! share the same sufficient statistics is another density in the same family
//! with natural parameters equal to the sum of the two input vectors. That is
//! the basic arithmetic this module exposes — it deliberately does not carry
//! probabilities, because everything in VMP happens in log / natural space.
//!
//! A message is tagged with a direction so that the engine can distinguish
//! factor→variable messages (information flowing into a variable's posterior)
//! from variable→factor messages (sufficient statistics needed by the factor
//! to emit its own factor→variable updates).

use crate::error::{PgmError, Result};

/// Direction of a VMP message.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MessageDirection {
    /// A factor sending an update to one of its variable neighbours.
    FactorToVariable,
    /// A variable sending its expected sufficient statistics to an adjacent factor.
    VariableToFactor,
}

/// A VMP message.
///
/// The `natural_params` vector has the dimensionality of the *receiving
/// variable's* exponential family in the factor→variable direction and the
/// dimensionality of the *sender's* sufficient statistics in the opposite
/// direction. The engine is responsible for maintaining this invariant.
#[derive(Clone, Debug)]
pub struct VmpMessage {
    /// Natural-parameter vector η (or its analogue for variable→factor messages,
    /// which carry expected sufficient statistics).
    pub natural_params: Vec<f64>,
    /// Sender identifier (factor id or variable name).
    pub from: String,
    /// Receiver identifier.
    pub to: String,
    /// Direction (factor→variable or variable→factor).
    pub direction: MessageDirection,
}

impl VmpMessage {
    /// Zero-message in the given direction and dimensionality.
    pub fn zeros(from: String, to: String, direction: MessageDirection, dim: usize) -> Self {
        Self {
            natural_params: vec![0.0; dim],
            from,
            to,
            direction,
        }
    }

    /// Dimensionality of the message.
    pub fn dim(&self) -> usize {
        self.natural_params.len()
    }

    /// Sum two messages element-wise, producing a third (natural parameters
    /// add under the product-of-densities rule).
    ///
    /// Requires identical direction / endpoints / dimensionality — otherwise
    /// returns a dimension-mismatch error.
    pub fn product(a: &Self, b: &Self) -> Result<Self> {
        if a.natural_params.len() != b.natural_params.len() {
            return Err(PgmError::DimensionMismatch {
                expected: vec![a.natural_params.len()],
                got: vec![b.natural_params.len()],
            });
        }
        let summed = a
            .natural_params
            .iter()
            .zip(b.natural_params.iter())
            .map(|(x, y)| x + y)
            .collect();
        Ok(Self {
            natural_params: summed,
            from: a.from.clone(),
            to: a.to.clone(),
            direction: a.direction,
        })
    }

    /// Add `rhs` into `self` in place, producing the natural-parameter sum.
    pub fn accumulate(&mut self, rhs: &Self) -> Result<()> {
        if self.natural_params.len() != rhs.natural_params.len() {
            return Err(PgmError::DimensionMismatch {
                expected: vec![self.natural_params.len()],
                got: vec![rhs.natural_params.len()],
            });
        }
        for (lhs, r) in self
            .natural_params
            .iter_mut()
            .zip(rhs.natural_params.iter())
        {
            *lhs += *r;
        }
        Ok(())
    }

    /// L∞ residual between two messages of identical shape. Useful for
    /// convergence monitoring in the engine.
    pub fn linf_residual(a: &Self, b: &Self) -> Result<f64> {
        if a.natural_params.len() != b.natural_params.len() {
            return Err(PgmError::DimensionMismatch {
                expected: vec![a.natural_params.len()],
                got: vec![b.natural_params.len()],
            });
        }
        let mut max = 0.0_f64;
        for (x, y) in a.natural_params.iter().zip(b.natural_params.iter()) {
            max = max.max((x - y).abs());
        }
        Ok(max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_requested_dim() {
        let m = VmpMessage::zeros(
            "f".to_string(),
            "v".to_string(),
            MessageDirection::FactorToVariable,
            3,
        );
        assert_eq!(m.dim(), 3);
        assert!(m.natural_params.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn product_is_element_wise_sum() {
        let mut a = VmpMessage::zeros(
            "f1".into(),
            "v".into(),
            MessageDirection::FactorToVariable,
            3,
        );
        a.natural_params = vec![1.0, 2.0, 3.0];
        let mut b = a.clone();
        b.natural_params = vec![0.5, -1.0, 4.0];
        let p = VmpMessage::product(&a, &b).expect("product");
        assert_eq!(p.natural_params, vec![1.5, 1.0, 7.0]);
    }

    #[test]
    fn accumulate_matches_product() {
        let mut a = VmpMessage::zeros(
            "f".into(),
            "v".into(),
            MessageDirection::FactorToVariable,
            2,
        );
        a.natural_params = vec![1.0, -1.0];
        let mut b = a.clone();
        b.natural_params = vec![2.5, 0.5];
        a.accumulate(&b).expect("accum");
        assert_eq!(a.natural_params, vec![3.5, -0.5]);
    }

    #[test]
    fn linf_residual_is_max_abs() {
        let mut a = VmpMessage::zeros(
            "f".into(),
            "v".into(),
            MessageDirection::FactorToVariable,
            3,
        );
        a.natural_params = vec![1.0, 2.0, 3.0];
        let mut b = a.clone();
        b.natural_params = vec![1.1, 1.5, 5.0];
        let r = VmpMessage::linf_residual(&a, &b).expect("residual");
        assert!((r - 2.0).abs() < 1e-12);
    }

    #[test]
    fn dimension_mismatch_is_error() {
        let a = VmpMessage::zeros(
            "f".into(),
            "v".into(),
            MessageDirection::FactorToVariable,
            2,
        );
        let b = VmpMessage::zeros(
            "f".into(),
            "v".into(),
            MessageDirection::FactorToVariable,
            3,
        );
        assert!(VmpMessage::product(&a, &b).is_err());
        assert!(VmpMessage::linf_residual(&a, &b).is_err());
    }
}
