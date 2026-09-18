// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Error type for fallible articulated-body computations.

/// Errors returned by fallible articulated-body algorithms and constructors
/// (operational-space inertia in [`crate::osc`], [`crate::aba::try_aba`],
/// [`crate::ik::solve_ik_dls`], [`crate::model::ArticulatedModel::try_add_body`],
/// the joints' `try_new` constructors, …).
#[derive(Debug, Clone, PartialEq)]
pub enum ArticulatedError {
    /// The joint-space mass matrix was not positive-definite: the Cholesky
    /// factorisation met a non-positive pivot.
    NotPositiveDefinite {
        /// Zero-based pivot index where factorisation failed.
        index: usize,
    },
    /// The task-space inertia inverse `J·M⁻¹·Jᵀ` is singular (rank-deficient),
    /// e.g. a 6-row task Jacobian on a robot with fewer than 6 DoF or at a
    /// kinematic singularity. Use a damped variant to obtain a regularised
    /// result.
    SingularTaskInertia,
    /// An input had the wrong length.
    DimensionMismatch {
        /// Which input was malformed (e.g. `"jacobian rows"`).
        what: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length supplied.
        got: usize,
    },
    /// A numeric parameter was outside its valid domain.
    InvalidParameter {
        /// Name of the offending parameter.
        what: &'static str,
        /// The supplied value.
        value: f64,
    },
    /// The joint-space articulated inertia `D = Sᵀ·I_A·S` of a body's joint is
    /// singular (e.g. a massless, inertia-free link driven by a revolute
    /// joint), so the forward-dynamics joint acceleration is undefined.
    SingularJointInertia {
        /// Index of the body whose joint inertia is singular.
        body: usize,
    },
    /// A joint axis was zero (or numerically zero) or had non-finite
    /// components, so it cannot be normalised to a unit direction.
    ZeroAxis {
        /// The rejected axis.
        axis: [f64; 3],
    },
    /// A body's parent index does not precede it in topological order
    /// (`parent < body` is required), e.g. a self-parent or forward reference.
    InvalidParent {
        /// Index the new body would receive.
        body: usize,
        /// The rejected parent index.
        parent: usize,
    },
    /// A body index was out of range for the model.
    InvalidBodyIndex {
        /// The supplied body index.
        index: usize,
        /// Number of bodies in the model.
        num_bodies: usize,
    },
}

impl std::fmt::Display for ArticulatedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPositiveDefinite { index } => write!(
                f,
                "mass matrix is not positive-definite at pivot index {index}"
            ),
            Self::SingularTaskInertia => write!(
                f,
                "task-space inertia J·M⁻¹·Jᵀ is singular (rank-deficient task Jacobian)"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch for {what}: expected {expected}, got {got}"
            ),
            Self::InvalidParameter { what, value } => {
                write!(f, "invalid parameter {what} = {value}")
            }
            Self::SingularJointInertia { body } => write!(
                f,
                "joint-space articulated inertia of body {body} is singular"
            ),
            Self::ZeroAxis { axis } => write!(
                f,
                "joint axis {axis:?} is zero or non-finite and cannot be normalised"
            ),
            Self::InvalidParent { body, parent } => write!(
                f,
                "invalid parent {parent} for body {body}: parents must precede their children"
            ),
            Self::InvalidBodyIndex { index, num_bodies } => write!(
                f,
                "body index {index} out of range for a model with {num_bodies} bodies"
            ),
        }
    }
}

impl std::error::Error for ArticulatedError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_name_the_problem() {
        let e = ArticulatedError::NotPositiveDefinite { index: 3 };
        assert!(e.to_string().contains("pivot index 3"));
        let e = ArticulatedError::DimensionMismatch {
            what: "jacobian rows",
            expected: 6,
            got: 5,
        };
        let msg = e.to_string();
        assert!(msg.contains("jacobian rows") && msg.contains('6') && msg.contains('5'));
        assert!(
            ArticulatedError::SingularTaskInertia
                .to_string()
                .contains("singular")
        );
        let msg = ArticulatedError::SingularJointInertia { body: 4 }.to_string();
        assert!(msg.contains("body 4") && msg.contains("singular"));
        let msg = ArticulatedError::ZeroAxis {
            axis: [0.0, 0.0, 0.0],
        }
        .to_string();
        assert!(msg.contains("axis"));
        let msg = ArticulatedError::InvalidParent { body: 2, parent: 5 }.to_string();
        assert!(msg.contains("parent 5") && msg.contains("body 2"));
        let msg = ArticulatedError::InvalidBodyIndex {
            index: 9,
            num_bodies: 3,
        }
        .to_string();
        assert!(msg.contains('9') && msg.contains('3'));
    }
}
