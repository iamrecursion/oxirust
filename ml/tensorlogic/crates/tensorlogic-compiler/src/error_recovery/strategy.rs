//! Recovery strategy used by the tolerant compiler.
//!
//! A [`RecoveryStrategy`] selects how the tolerant driver reacts to a
//! blocking diagnostic — i.e. a [`crate::error_recovery::Severity::Error`]
//! or [`crate::error_recovery::Severity::Fatal`].
//!
//! | Strategy         | Warning  | Error                 | Fatal                 |
//! |------------------|----------|-----------------------|-----------------------|
//! | `SkipOnError`    | continue | skip this expr only   | skip this expr only   |
//! | `SkipOnFatal`    | continue | skip this expr only   | abort whole program   |
//! | `AbortOnAny`     | continue | abort whole program   | abort whole program   |
//!
//! "Skip this expr only" means the offending expression's slot becomes
//! `None` in the result while the remaining expressions are still compiled.
//! "Abort whole program" means *all* later expressions are left as `None`.
//!
//! The default strategy is [`RecoveryStrategy::SkipOnError`] — the most
//! tolerant mode, which matches the intent of partial error recovery.

use super::diagnostic::Severity;

/// Configurable error-recovery policy for the tolerant compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RecoveryStrategy {
    /// Skip only the offending expression on *any* blocking diagnostic
    /// (Error or Fatal). Every well-formed expression is still compiled.
    ///
    /// This is the most tolerant mode and the default for partial error
    /// recovery.
    #[default]
    SkipOnError,

    /// Skip the offending expression on [`Severity::Error`] but abort the
    /// entire program on [`Severity::Fatal`].
    SkipOnFatal,

    /// Abort on the first blocking diagnostic (Error or Fatal). Warnings and
    /// Infos are still collected but never cause an abort.
    ///
    /// This is functionally equivalent to the pre-existing strict
    /// compilation mode but also returns the partial diagnostics collected
    /// so far.
    AbortOnAny,
}

impl RecoveryStrategy {
    /// Decide what the driver should do for a diagnostic of this severity.
    pub fn decide(self, severity: Severity) -> RecoveryAction {
        match (self, severity) {
            // Non-blocking severities never alter control flow.
            (_, Severity::Info) | (_, Severity::Warning) => RecoveryAction::Continue,

            // SkipOnError: never abort — always skip just this expression.
            (RecoveryStrategy::SkipOnError, Severity::Error) => RecoveryAction::SkipExpression,
            (RecoveryStrategy::SkipOnError, Severity::Fatal) => RecoveryAction::SkipExpression,

            // SkipOnFatal: skip on Error, abort on Fatal.
            (RecoveryStrategy::SkipOnFatal, Severity::Error) => RecoveryAction::SkipExpression,
            (RecoveryStrategy::SkipOnFatal, Severity::Fatal) => RecoveryAction::AbortProgram,

            // AbortOnAny: abort on any blocking severity.
            (RecoveryStrategy::AbortOnAny, Severity::Error) => RecoveryAction::AbortProgram,
            (RecoveryStrategy::AbortOnAny, Severity::Fatal) => RecoveryAction::AbortProgram,
        }
    }
}

/// Action the tolerant driver takes after reporting a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecoveryAction {
    /// Keep compiling the current expression and then move on.
    Continue,
    /// Drop the current expression (slot becomes `None`) and proceed to the
    /// next expression.
    SkipExpression,
    /// Stop compilation immediately; remaining expressions become `None`.
    AbortProgram,
}

impl RecoveryAction {
    /// Returns `true` when this action aborts the whole program.
    pub fn is_abort(self) -> bool {
        matches!(self, RecoveryAction::AbortProgram)
    }

    /// Returns `true` when this action drops the current expression slot.
    pub fn drops_expression(self) -> bool {
        matches!(
            self,
            RecoveryAction::SkipExpression | RecoveryAction::AbortProgram
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_skip_on_error() {
        assert_eq!(RecoveryStrategy::default(), RecoveryStrategy::SkipOnError);
    }

    #[test]
    fn skip_on_error_never_aborts() {
        let s = RecoveryStrategy::SkipOnError;
        assert_eq!(s.decide(Severity::Info), RecoveryAction::Continue);
        assert_eq!(s.decide(Severity::Warning), RecoveryAction::Continue);
        assert_eq!(s.decide(Severity::Error), RecoveryAction::SkipExpression);
        assert_eq!(s.decide(Severity::Fatal), RecoveryAction::SkipExpression);
    }

    #[test]
    fn skip_on_fatal_aborts_only_on_fatal() {
        let s = RecoveryStrategy::SkipOnFatal;
        assert_eq!(s.decide(Severity::Warning), RecoveryAction::Continue);
        assert_eq!(s.decide(Severity::Error), RecoveryAction::SkipExpression);
        assert_eq!(s.decide(Severity::Fatal), RecoveryAction::AbortProgram);
    }

    #[test]
    fn abort_on_any_aborts_on_both_error_and_fatal() {
        let s = RecoveryStrategy::AbortOnAny;
        assert_eq!(s.decide(Severity::Info), RecoveryAction::Continue);
        assert_eq!(s.decide(Severity::Warning), RecoveryAction::Continue);
        assert_eq!(s.decide(Severity::Error), RecoveryAction::AbortProgram);
        assert_eq!(s.decide(Severity::Fatal), RecoveryAction::AbortProgram);
    }

    #[test]
    fn action_classifiers() {
        assert!(RecoveryAction::AbortProgram.is_abort());
        assert!(!RecoveryAction::SkipExpression.is_abort());
        assert!(!RecoveryAction::Continue.is_abort());

        assert!(!RecoveryAction::Continue.drops_expression());
        assert!(RecoveryAction::SkipExpression.drops_expression());
        assert!(RecoveryAction::AbortProgram.drops_expression());
    }
}
