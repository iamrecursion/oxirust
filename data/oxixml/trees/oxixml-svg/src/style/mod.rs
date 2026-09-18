//! The `style` attribute, `<style>` stylesheets, and presentation properties.

pub mod attr;
pub mod css;
pub mod property;

// Convenience re-exports of the main style-engine types.
pub use attr::Declaration;
pub use css::{CompoundSelector, ElementFacts, Selector, Specificity, StyleRule, Stylesheet};
pub use property::PropertyId;
