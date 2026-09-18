//! # SparqlGenConfig - Trait Implementations
//!
//! This module contains trait implementations for `SparqlGenConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SparqlGenConfig;

impl Default for SparqlGenConfig {
    fn default() -> Self {
        SparqlGenConfig {
            base_prefix: "http://tensorlogic.org/ont#".to_owned(),
            variable_prefix: "v_".to_owned(),
            use_optional_for_exists: false,
            indent: "  ".to_owned(),
            max_depth: 50,
        }
    }
}
