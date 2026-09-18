//! Budgets that bound the cost of entity expansion.
//!
//! Entity expansion is the classic denial-of-service surface of an XML
//! processor. Two shapes of attack matter:
//!
//! * **Exponential blow-up** ("billion laughs"): ten entities, each referencing
//!   the previous one ten times, expand a few hundred bytes of declarations
//!   into a gigabyte of text.
//! * **Quadratic blow-up**: one large entity referenced a few thousand times.
//!
//! [`ExpansionLimits`] caps both, plus the nesting depth (which additionally
//! turns an accidentally self-referential entity into a clean error rather than
//! a stack overflow). Every expansion path in this crate — parameter entities
//! during DTD parsing, general entities in attribute values and in content —
//! draws from the same [`Budget`], so a document cannot dodge the cap by
//! splitting an attack across several places.

use crate::error::{DtdError, DtdErrorKind, ExpansionLimit};

/// Caps on how much work entity expansion may cost.
///
/// The defaults are deliberately generous enough for real documents (a 10 MiB
/// expanded budget, 10 000 individual expansions, 40 levels of nesting) and far
/// too small for an expansion bomb.
///
/// # Examples
///
/// ```
/// use oxixml_dtd::ExpansionLimits;
///
/// let limits = ExpansionLimits::default();
/// assert_eq!(limits.max_depth, 40);
///
/// // Tighten every knob at once, for untrusted input.
/// let strict = ExpansionLimits {
///     max_depth: 8,
///     max_expansions: 128,
///     max_expanded_size: 64 * 1024,
/// };
/// assert!(strict.max_expanded_size < limits.max_expanded_size);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpansionLimits {
    /// The greatest depth of nested entity references. Default `40`.
    pub max_depth: usize,
    /// The greatest number of individual entity expansions performed while
    /// reading one DTD or validating one document. Default `10_000`.
    pub max_expansions: usize,
    /// The greatest total number of characters that entity expansion may
    /// produce. Default `10_485_760` (10 MiB).
    pub max_expanded_size: usize,
}

impl ExpansionLimits {
    /// The default limits: 40 levels deep, 10 000 expansions, 10 MiB of text.
    pub const DEFAULT: Self = Self {
        max_depth: 40,
        max_expansions: 10_000,
        max_expanded_size: 10 * 1024 * 1024,
    };

    /// Limits suitable for hostile input: 8 levels deep, 128 expansions, 64 KiB.
    pub const STRICT: Self = Self {
        max_depth: 8,
        max_expansions: 128,
        max_expanded_size: 64 * 1024,
    };

    /// Limits high enough that only a genuine bomb hits them: 256 levels,
    /// one million expansions, 1 GiB.
    ///
    /// Even this is bounded on purpose — there is no "unlimited" setting,
    /// because an unbounded expander cannot be made safe.
    pub const RELAXED: Self = Self {
        max_depth: 256,
        max_expansions: 1_000_000,
        max_expanded_size: 1024 * 1024 * 1024,
    };

    /// Starts a fresh [`Budget`] against these limits.
    #[must_use]
    pub const fn budget(self) -> Budget {
        Budget {
            limits: self,
            expansions: 0,
            expanded_size: 0,
        }
    }
}

impl Default for ExpansionLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The running tally of entity-expansion work done so far.
///
/// A `Budget` is created from [`ExpansionLimits::budget`] and threaded through
/// every expansion. It is deliberately *not* reset between declarations: the
/// whole point is that the totals accumulate across a document.
///
/// # Examples
///
/// ```
/// use oxixml_dtd::{ExpansionLimits, ExpansionLimit, DtdErrorKind};
///
/// let mut budget = ExpansionLimits { max_depth: 4, max_expansions: 2, max_expanded_size: 100 }
///     .budget();
///
/// assert!(budget.charge_expansion(10).is_ok());
/// assert!(budget.charge_expansion(10).is_ok());
/// let error = budget.charge_expansion(10).expect_err("the third expansion is over budget");
/// assert!(matches!(
///     error.kind(),
///     DtdErrorKind::EntityExpansionLimit(ExpansionLimit::Count(2)),
/// ));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    limits: ExpansionLimits,
    expansions: usize,
    expanded_size: usize,
}

impl Budget {
    /// The limits this budget is measured against.
    #[must_use]
    pub const fn limits(&self) -> ExpansionLimits {
        self.limits
    }

    /// How many entity expansions have been charged so far.
    #[must_use]
    pub const fn expansions(&self) -> usize {
        self.expansions
    }

    /// How many characters of expanded text have been charged so far.
    #[must_use]
    pub const fn expanded_size(&self) -> usize {
        self.expanded_size
    }

    /// Charges one entity expansion producing `chars` characters.
    ///
    /// # Errors
    ///
    /// Returns [`DtdErrorKind::EntityExpansionLimit`] when either the expansion
    /// count or the accumulated expanded size would exceed its limit.
    pub fn charge_expansion(&mut self, chars: usize) -> Result<(), DtdError> {
        self.expansions = self.expansions.saturating_add(1);
        if self.expansions > self.limits.max_expansions {
            return Err(DtdError::detached(DtdErrorKind::EntityExpansionLimit(
                ExpansionLimit::Count(self.limits.max_expansions),
            )));
        }
        self.expanded_size = self.expanded_size.saturating_add(chars);
        if self.expanded_size > self.limits.max_expanded_size {
            return Err(DtdError::detached(DtdErrorKind::EntityExpansionLimit(
                ExpansionLimit::TotalSize(self.limits.max_expanded_size),
            )));
        }
        Ok(())
    }

    /// Checks a nesting depth against [`ExpansionLimits::max_depth`].
    ///
    /// # Errors
    ///
    /// Returns [`DtdErrorKind::EntityExpansionLimit`] when `depth` exceeds the
    /// configured maximum.
    pub fn check_depth(&self, depth: usize) -> Result<(), DtdError> {
        if depth > self.limits.max_depth {
            return Err(DtdError::detached(DtdErrorKind::EntityExpansionLimit(
                ExpansionLimit::Depth(self.limits.max_depth),
            )));
        }
        Ok(())
    }

    /// Clears the running totals, keeping the limits.
    pub fn reset(&mut self) {
        self.expansions = 0;
        self.expanded_size = 0;
    }
}

impl Default for Budget {
    fn default() -> Self {
        ExpansionLimits::DEFAULT.budget()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_limit_trips_before_count_limit_for_one_huge_entity() {
        let mut budget = ExpansionLimits {
            max_depth: 4,
            max_expansions: 100,
            max_expanded_size: 16,
        }
        .budget();
        let error = budget
            .charge_expansion(17)
            .expect_err("over the size limit");
        assert!(matches!(
            error.kind(),
            DtdErrorKind::EntityExpansionLimit(ExpansionLimit::TotalSize(16))
        ));
    }

    #[test]
    fn depth_limit_is_inclusive() {
        let budget = ExpansionLimits {
            max_depth: 3,
            max_expansions: 10,
            max_expanded_size: 100,
        }
        .budget();
        assert!(budget.check_depth(3).is_ok());
        assert!(budget.check_depth(4).is_err());
    }

    #[test]
    fn reset_clears_totals_but_keeps_limits() {
        let mut budget = ExpansionLimits::STRICT.budget();
        budget.charge_expansion(10).expect("within budget");
        assert_eq!(budget.expansions(), 1);
        budget.reset();
        assert_eq!(budget.expansions(), 0);
        assert_eq!(budget.expanded_size(), 0);
        assert_eq!(budget.limits(), ExpansionLimits::STRICT);
    }
}
