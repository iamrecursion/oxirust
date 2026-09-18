//! Matrix decomposition techniques
//!
//! This module provides various matrix decomposition algorithms that can be used
//! for feature extraction, data compression, and interpretable representations.

mod dictionary_learning;
mod nmf;

pub use self::dictionary_learning::DictionaryLearning;
pub use self::nmf::NMF;
