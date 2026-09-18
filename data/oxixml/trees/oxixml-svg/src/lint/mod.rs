//! SVG-aware document validation and linting.

pub mod diagnostic;
pub mod rules;

// Convenience re-exports of the main lint types.
pub use diagnostic::{Diagnostic, LintOptions, Severity, codes};
pub use rules::Linter;
