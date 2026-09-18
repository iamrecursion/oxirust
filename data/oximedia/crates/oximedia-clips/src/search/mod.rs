//! Search and filtering system.

pub mod engine;
pub mod filter;
pub mod fuzzy;

pub use engine::SearchEngine;
pub use filter::{ClipFilter, FilterCriteria};
pub use fuzzy::FuzzyMatcher;
