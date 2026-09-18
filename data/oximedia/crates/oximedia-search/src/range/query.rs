//! Range query builder.

use serde::{Deserialize, Serialize};

/// Range query builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeQueryBuilder {
    /// Field name
    pub field: String,
    /// Minimum value
    pub min: Option<i64>,
    /// Maximum value
    pub max: Option<i64>,
}

impl RangeQueryBuilder {
    /// Create a new range query
    #[must_use]
    pub fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            min: None,
            max: None,
        }
    }

    /// Set minimum value
    #[must_use]
    pub const fn min(mut self, value: i64) -> Self {
        self.min = Some(value);
        self
    }

    /// Set maximum value
    #[must_use]
    pub const fn max(mut self, value: i64) -> Self {
        self.max = Some(value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_query() {
        let query = RangeQueryBuilder::new("duration_ms").min(1000).max(5000);

        assert_eq!(query.min, Some(1000));
        assert_eq!(query.max, Some(5000));
    }
}
