//! Full-text search implementation.

pub mod fuzzy;
#[cfg(feature = "search-engine")]
pub mod search;
pub mod stemmer;
pub mod tokenizer;

pub use fuzzy::FuzzyMatcher;
#[cfg(feature = "search-engine")]
pub use search::TextSearchIndex;
pub use stemmer::Stemmer;
pub use tokenizer::Tokenizer;
