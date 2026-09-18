// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Error type of the [`MultibodyChain`](super::MultibodyChain) dynamics.

use thiserror::Error;

/// Reasons why a query on a [`MultibodyChain`](super::MultibodyChain) cannot be
/// answered.
///
/// The dynamics routines take caller-supplied slices (joint torques, joint
/// accelerations) and indices (the link of a Jacobian) that cannot be checked
/// at compile time.  They report a bad argument through this type instead of
/// indexing out of bounds, and they report a degenerate chain instead of
/// returning made-up accelerations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MultibodyError {
    /// A slice or table does not hold exactly one entry per joint.
    ///
    /// This covers the argument slices of the dynamics routines as well as the
    /// per-link tables of the chain itself (`parent`, `joint_angles`,
    /// `joint_velocities`), which are public fields and can fall out of step
    /// with `links`.
    #[error("expected {expected} entries (one per joint), got {got}")]
    DimensionMismatch {
        /// Number of joints of the chain.
        expected: usize,
        /// Number of entries that were supplied.
        got: usize,
    },

    /// A link index does not refer to a link of the chain.
    #[error("link index {0} is out of range")]
    LinkOutOfRange(usize),

    /// A link names a parent that is not stored before it.
    ///
    /// The kinematic and dynamic recursions walk the links in storage order, so
    /// every parent index must be smaller than the index of its child.  A link
    /// that is its own ancestor would also make the recursions run forever.
    #[error("link {link} has parent {parent}, which is not stored before it")]
    InvalidParent {
        /// Index of the link with the bad parent.
        link: usize,
        /// The offending parent index.
        parent: usize,
    },

    /// The joint accelerations are undetermined: the joint-space mass matrix is
    /// singular or the equations of motion are not finite.
    ///
    /// The mass matrix is singular when some joint has no inertia about its
    /// axis, for example a massless link or a point mass on the axis.  A NaN or
    /// infinite joint state, torque or gravity also leaves no accelerations to
    /// report.  In both cases the torques do not determine `q̈`, and no
    /// placeholder values are returned.
    #[error(
        "the joint accelerations are undetermined: the mass matrix is singular or the state is not finite"
    )]
    SingularMassMatrix,
}
