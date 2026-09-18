// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Error type shared by every QM engine behind [`QmRegion::run`](super::QmRegion::run).

use thiserror::Error;

/// Failure of a QM (or QM/MM) calculation.
///
/// Every engine (PM3, SCC-DFTB, HF/STO-3G, KS-DFT) reports its failures through
/// this type instead of silently leaving stale energies and forces behind.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum QmError {
    /// An atom has an atomic number for which the selected method has no parameters
    /// or basis set.
    #[error("element Z={0} is not supported by the selected QM method")]
    UnsupportedElement(u32),
    /// Open-shell systems (odd electron count or multiplicity ≠ 1) are not supported
    /// by the closed-shell engines.
    #[error("open-shell systems (odd electron count or multiplicity != 1) are not supported")]
    OpenShellNotSupported,
    /// The generalised symmetric eigensolver failed (overlap matrix not positive definite).
    #[error("generalised eigensolver failed (overlap matrix not positive definite)")]
    EigensolverFailed,
    /// The SCF / SCC cycle did not converge within the maximum number of iterations.
    #[error("SCF did not converge")]
    ScfNotConverged,
    /// The QM region contains no atoms.
    #[error("QM region contains no atoms")]
    EmptyRegion,
    /// The electron count implied by the atoms and the total charge is negative or does
    /// not fit into the available basis functions.
    #[error("invalid electron count {electrons} for {n_basis} basis functions")]
    InvalidElectronCount {
        /// Number of electrons implied by the valence counts and the total charge.
        electrons: i64,
        /// Number of basis functions available to hold them.
        n_basis: usize,
    },
    /// Two parallel input arrays have inconsistent lengths.
    #[error("dimension mismatch for {what}: expected {expected}, found {found}")]
    DimensionMismatch {
        /// Which quantity was inconsistent.
        what: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        found: usize,
    },
    /// An external (MM) point charge has a non-finite position or charge, or sits on
    /// top of a QM nucleus, so the embedding potential is singular.
    #[error("external point charge {index} is non-finite or coincides with a QM atom")]
    InvalidExternalCharge {
        /// Index of the offending charge in `QmRegion::external_charges`.
        index: usize,
    },
    /// A failure reported by the underlying quantum-chemistry solver.
    #[error("quantum-chemistry solver: {0}")]
    QuantumChemistry(#[from] crate::quantum_chemistry::QuantumChemistryError),
    /// An atom index stored in a link atom does not exist in its region.
    #[error("link atom references {region} atom {index}, but the region has {len} atoms")]
    InvalidLinkAtom {
        /// `"QM"` or `"MM"`.
        region: &'static str,
        /// Referenced index.
        index: usize,
        /// Number of atoms in the referenced region.
        len: usize,
    },
    /// Two QM atoms occupy (numerically) the same position, so the interatomic
    /// distance vanishes and no overlap or two-centre integral is defined.  This is
    /// also what an unpopulated [`QmRegion`](super::QmRegion) looks like: every atom
    /// a hydrogen at the origin.
    #[error("QM atoms {first} and {second} coincide")]
    CoincidentAtoms {
        /// Index of the first atom.
        first: usize,
        /// Index of the second atom.
        second: usize,
    },
}

/// Failure of the classical energy evaluation of an [`MmRegion`](super::MmRegion).
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum MmError {
    /// A per-atom array does not have one entry per MM atom position.
    #[error("MM region array {what} has {found} entries, expected {expected}")]
    LengthMismatch {
        /// Name of the offending array.
        what: &'static str,
        /// Expected length (the number of positions).
        expected: usize,
        /// Actual length.
        found: usize,
    },
    /// A position, charge or Lennard-Jones parameter is not finite, or a
    /// Lennard-Jones well depth or diameter is negative.
    #[error("MM atom {index} has an invalid {what}")]
    InvalidParameter {
        /// Name of the offending quantity.
        what: &'static str,
        /// Index of the offending atom.
        index: usize,
    },
    /// Two MM atoms occupy (numerically) the same position, so their Coulomb and
    /// Lennard-Jones interaction is singular.
    #[error("MM atoms {first} and {second} coincide")]
    CoincidentAtoms {
        /// Index of the first atom.
        first: usize,
        /// Index of the second atom.
        second: usize,
    },
}
