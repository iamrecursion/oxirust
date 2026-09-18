//! Checksum generation and verification for digital preservation
//!
//! This module provides multi-algorithm checksum support including:
//! - MD5 (legacy compatibility)
//! - SHA-256 (recommended)
//! - SHA-512 (high security)
//! - xxHash (fast verification)
//! - BLAKE3 (modern, fast, secure)

pub mod generate;
pub mod tree;
pub mod verify;

pub use generate::{ChecksumGenerator, FileChecksum};
pub use tree::{MerkleNode, MerkleProofStep, MerkleTree, ProofDirection};
pub use verify::{
    AlgoVerificationResult, ChecksumVerifier, ConcurrentVerificationReport, FileVerificationReport,
    VerificationReport, VerificationResult,
};

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported checksum algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    /// MD5 (128-bit, legacy)
    Md5,
    /// SHA-256 (256-bit, recommended)
    Sha256,
    /// SHA-512 (512-bit, high security)
    Sha512,
    /// xxHash64 (64-bit, fast)
    XxHash64,
    /// BLAKE3 (256-bit, modern)
    Blake3,
}

impl ChecksumAlgorithm {
    /// Returns the name of the algorithm
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Md5 => "MD5",
            Self::Sha256 => "SHA-256",
            Self::Sha512 => "SHA-512",
            Self::XxHash64 => "xxHash64",
            Self::Blake3 => "BLAKE3",
        }
    }

    /// Returns the recommended algorithm for preservation
    #[must_use]
    pub const fn recommended() -> Self {
        Self::Sha256
    }

    /// Returns all supported algorithms
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Md5,
            Self::Sha256,
            Self::Sha512,
            Self::XxHash64,
            Self::Blake3,
        ]
    }

    /// Returns whether this algorithm is cryptographically secure
    #[must_use]
    pub const fn is_cryptographic(&self) -> bool {
        matches!(self, Self::Sha256 | Self::Sha512 | Self::Blake3)
    }
}

impl fmt::Display for ChecksumAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_names() {
        assert_eq!(ChecksumAlgorithm::Md5.name(), "MD5");
        assert_eq!(ChecksumAlgorithm::Sha256.name(), "SHA-256");
        assert_eq!(ChecksumAlgorithm::Blake3.name(), "BLAKE3");
    }

    #[test]
    fn test_recommended_algorithm() {
        assert_eq!(ChecksumAlgorithm::recommended(), ChecksumAlgorithm::Sha256);
    }

    #[test]
    fn test_cryptographic_algorithms() {
        assert!(!ChecksumAlgorithm::Md5.is_cryptographic());
        assert!(ChecksumAlgorithm::Sha256.is_cryptographic());
        assert!(ChecksumAlgorithm::Blake3.is_cryptographic());
        assert!(!ChecksumAlgorithm::XxHash64.is_cryptographic());
    }
}
