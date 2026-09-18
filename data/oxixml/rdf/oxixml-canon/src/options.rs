//! Tunable parameters for the canonicalization algorithm: the hash function and the work budget.

use oxixml_support::hash::{sha256, sha384};

/// The hash function parameter of [RDFC-1.0](https://www.w3.org/TR/rdf-canon/#dfn-hash-algorithm).
///
/// The specification is parameterized on a cryptographic hash function; SHA-256 is the default and
/// SHA-384 is the standard alternative (used, for example, by the suite's `SHA384` test cases).
/// Both are provided by [`oxixml_support::hash`] and require no external dependencies.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
#[non_exhaustive]
pub enum HashAlgorithm {
    /// SHA-256 — the RDFC-1.0 default.
    #[default]
    Sha256,
    /// SHA-384.
    Sha384,
}

impl HashAlgorithm {
    /// Hashes `input` and returns the digest as a lowercase hexadecimal string, exactly as
    /// required by the RDFC-1.0 [hash steps](https://www.w3.org/TR/rdf-canon/#hash-1d-quads).
    #[must_use]
    pub(crate) fn hex_digest(self, input: &str) -> String {
        match self {
            Self::Sha256 => to_hex(&sha256(input.as_bytes())),
            Self::Sha384 => to_hex(&sha384(input.as_bytes())),
        }
    }
}

/// The default work budget, in abstract work units: 262 144. See [`WorkBudget`].
///
/// The heaviest non-adversarial dataset in the W3C `rdf-canon` test suite — including the
/// "poison – evil" cases that are explicitly *computable given defined limits* — settles in under
/// 4 096 work units, so this default leaves roughly two orders of magnitude of headroom while
/// still rejecting the deliberately intractable inputs (such as the suite's fully connected
/// 10-node blank-node clique) in a fraction of a second.
pub const DEFAULT_WORK_BUDGET: u64 = 1 << 18;

/// A guard against adversarial ("poison") datasets, expressed as a maximum number of abstract
/// work units the [`hash-n-degree-quads`](https://www.w3.org/TR/rdf-canon/#hash-nd-quads)
/// recursion may consume.
///
/// One work unit is charged for every gossip-path permutation examined and for every recursive
/// `hash-n-degree-quads` invocation. When the budget is exhausted, canonicalization fails with
/// [`CanonicalizationError::WorkBudgetExceeded`](crate::CanonicalizationError::WorkBudgetExceeded)
/// instead of running for an unbounded (up to factorial) amount of time, as recommended by the
/// RDFC-1.0 [security considerations](https://www.w3.org/TR/rdf-canon/#security-considerations).
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct WorkBudget {
    max_work_units: u64,
}

impl WorkBudget {
    /// Creates a budget of at most `max_work_units` work units.
    #[must_use]
    pub const fn new(max_work_units: u64) -> Self {
        Self { max_work_units }
    }

    /// Creates an effectively unlimited budget.
    ///
    /// <div class="warning">Only use this on trusted input: an adversarial dataset can make
    /// canonicalization take super-polynomial time.</div>
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            max_work_units: u64::MAX,
        }
    }

    /// Returns the maximum number of work units permitted.
    #[must_use]
    pub const fn max_work_units(self) -> u64 {
        self.max_work_units
    }
}

impl Default for WorkBudget {
    #[inline]
    fn default() -> Self {
        Self::new(DEFAULT_WORK_BUDGET)
    }
}

/// The full set of parameters for a canonicalization run.
///
/// Build one with [`CanonicalizationOptions::default`] and the `with_*` builders:
///
/// ```
/// use oxixml_canon::{CanonicalizationOptions, HashAlgorithm, WorkBudget};
///
/// let options = CanonicalizationOptions::default()
///     .with_hash_algorithm(HashAlgorithm::Sha384)
///     .with_work_budget(WorkBudget::new(1_000_000));
/// assert_eq!(options.hash_algorithm(), HashAlgorithm::Sha384);
/// ```
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub struct CanonicalizationOptions {
    hash_algorithm: HashAlgorithm,
    work_budget: WorkBudget,
}

impl CanonicalizationOptions {
    /// Returns a copy of these options with the given [`HashAlgorithm`].
    #[must_use]
    pub const fn with_hash_algorithm(mut self, hash_algorithm: HashAlgorithm) -> Self {
        self.hash_algorithm = hash_algorithm;
        self
    }

    /// Returns a copy of these options with the given [`WorkBudget`].
    #[must_use]
    pub const fn with_work_budget(mut self, work_budget: WorkBudget) -> Self {
        self.work_budget = work_budget;
        self
    }

    /// Returns the configured [`HashAlgorithm`].
    #[must_use]
    pub const fn hash_algorithm(self) -> HashAlgorithm {
        self.hash_algorithm
    }

    /// Returns the configured [`WorkBudget`].
    #[must_use]
    pub const fn work_budget(self) -> WorkBudget {
        self.work_budget
    }
}

/// Renders `bytes` as a lowercase hexadecimal string.
fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0F)] as char);
    }
    out
}
