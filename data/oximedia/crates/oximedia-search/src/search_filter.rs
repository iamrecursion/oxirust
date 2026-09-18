#![allow(dead_code)]
//! Advanced post-retrieval filtering engine for search results.
//!
//! Provides composable filter predicates for narrowing search results
//! after initial retrieval, including numeric ranges, string matching,
//! date ranges, set membership, and boolean combinations.

use std::collections::HashSet;

/// A comparison operator for numeric and date filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Equal to.
    Eq,
    /// Not equal to.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// A value that can be filtered on.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    /// Integer value.
    Int(i64),
    /// Floating point value.
    Float(f64),
    /// String value.
    Str(String),
    /// Boolean value.
    Bool(bool),
    /// Null / missing value.
    Null,
}

impl FilterValue {
    /// Try to extract as i64.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as f64.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to extract as string slice.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to extract as boolean.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Check if this value is null.
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

/// A filterable document providing named field access.
pub trait Filterable {
    /// Get a field value by name.
    fn field(&self, name: &str) -> FilterValue;
}

/// A single filter condition.
#[derive(Debug, Clone)]
pub enum FilterCondition {
    /// Compare a field with a numeric value.
    NumericCompare {
        /// Field name.
        field: String,
        /// Comparison operator.
        op: CompareOp,
        /// Comparison value.
        value: f64,
    },
    /// Check if field is within a numeric range (inclusive).
    NumericRange {
        /// Field name.
        field: String,
        /// Minimum value (inclusive).
        min: f64,
        /// Maximum value (inclusive).
        max: f64,
    },
    /// Match field against a string pattern (case-insensitive contains).
    StringContains {
        /// Field name.
        field: String,
        /// Substring to find.
        pattern: String,
    },
    /// Exact string match.
    StringExact {
        /// Field name.
        field: String,
        /// Exact value.
        value: String,
    },
    /// Match field against a prefix.
    StringPrefix {
        /// Field name.
        field: String,
        /// Prefix string.
        prefix: String,
    },
    /// Check if field value is in a set.
    InSet {
        /// Field name.
        field: String,
        /// Set of allowed string values.
        values: HashSet<String>,
    },
    /// Check if field is null/missing.
    IsNull {
        /// Field name.
        field: String,
    },
    /// Check if field is not null.
    IsNotNull {
        /// Field name.
        field: String,
    },
    /// Boolean AND of sub-conditions.
    And(Vec<FilterCondition>),
    /// Boolean OR of sub-conditions.
    Or(Vec<FilterCondition>),
    /// Negation.
    Not(Box<FilterCondition>),
}

impl FilterCondition {
    /// Create a numeric comparison filter.
    #[must_use]
    pub fn numeric_compare(field: &str, op: CompareOp, value: f64) -> Self {
        Self::NumericCompare {
            field: field.to_string(),
            op,
            value,
        }
    }

    /// Create a numeric range filter.
    #[must_use]
    pub fn numeric_range(field: &str, min: f64, max: f64) -> Self {
        Self::NumericRange {
            field: field.to_string(),
            min,
            max,
        }
    }

    /// Create a string contains filter.
    #[must_use]
    pub fn string_contains(field: &str, pattern: &str) -> Self {
        Self::StringContains {
            field: field.to_string(),
            pattern: pattern.to_lowercase(),
        }
    }

    /// Create an exact string match filter.
    #[must_use]
    pub fn string_exact(field: &str, value: &str) -> Self {
        Self::StringExact {
            field: field.to_string(),
            value: value.to_string(),
        }
    }

    /// Create a prefix match filter.
    #[must_use]
    pub fn string_prefix(field: &str, prefix: &str) -> Self {
        Self::StringPrefix {
            field: field.to_string(),
            prefix: prefix.to_lowercase(),
        }
    }

    /// Create a set membership filter.
    #[must_use]
    pub fn in_set(field: &str, values: &[&str]) -> Self {
        Self::InSet {
            field: field.to_string(),
            values: values.iter().map(|v| (*v).to_string()).collect(),
        }
    }

    /// Create a null check filter.
    #[must_use]
    pub fn is_null(field: &str) -> Self {
        Self::IsNull {
            field: field.to_string(),
        }
    }

    /// Create a not-null check filter.
    #[must_use]
    pub fn is_not_null(field: &str) -> Self {
        Self::IsNotNull {
            field: field.to_string(),
        }
    }

    /// Create an AND combination.
    #[must_use]
    pub fn and(conditions: Vec<Self>) -> Self {
        Self::And(conditions)
    }

    /// Create an OR combination.
    #[must_use]
    pub fn or(conditions: Vec<Self>) -> Self {
        Self::Or(conditions)
    }

    /// Create a NOT wrapper.
    #[must_use]
    pub fn not(condition: Self) -> Self {
        Self::Not(Box::new(condition))
    }

    /// Evaluate this condition against a filterable item.
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate<T: Filterable>(&self, item: &T) -> bool {
        match self {
            Self::NumericCompare { field, op, value } => {
                let fv = item.field(field);
                if let Some(v) = fv.as_float() {
                    match op {
                        CompareOp::Eq => (v - value).abs() < f64::EPSILON,
                        CompareOp::Ne => (v - value).abs() >= f64::EPSILON,
                        CompareOp::Lt => v < *value,
                        CompareOp::Le => v <= *value,
                        CompareOp::Gt => v > *value,
                        CompareOp::Ge => v >= *value,
                    }
                } else {
                    false
                }
            }
            Self::NumericRange { field, min, max } => {
                let fv = item.field(field);
                if let Some(v) = fv.as_float() {
                    v >= *min && v <= *max
                } else {
                    false
                }
            }
            Self::StringContains { field, pattern } => {
                let fv = item.field(field);
                if let Some(s) = fv.as_str() {
                    s.to_lowercase().contains(pattern)
                } else {
                    false
                }
            }
            Self::StringExact { field, value } => {
                let fv = item.field(field);
                if let Some(s) = fv.as_str() {
                    s == value
                } else {
                    false
                }
            }
            Self::StringPrefix { field, prefix } => {
                let fv = item.field(field);
                if let Some(s) = fv.as_str() {
                    s.to_lowercase().starts_with(prefix)
                } else {
                    false
                }
            }
            Self::InSet { field, values } => {
                let fv = item.field(field);
                if let Some(s) = fv.as_str() {
                    values.contains(s)
                } else {
                    false
                }
            }
            Self::IsNull { field } => item.field(field).is_null(),
            Self::IsNotNull { field } => !item.field(field).is_null(),
            Self::And(conditions) => conditions.iter().all(|c| c.evaluate(item)),
            Self::Or(conditions) => conditions.iter().any(|c| c.evaluate(item)),
            Self::Not(condition) => !condition.evaluate(item),
        }
    }
}

/// A composable filter pipeline that applies multiple conditions.
#[derive(Debug, Clone)]
pub struct FilterPipeline {
    /// Root condition (all conditions AND-ed together).
    conditions: Vec<FilterCondition>,
}

impl FilterPipeline {
    /// Create an empty filter pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Add a condition to the pipeline.
    pub fn add(&mut self, condition: FilterCondition) {
        self.conditions.push(condition);
    }

    /// Build a chained condition builder.
    #[must_use]
    pub fn with(mut self, condition: FilterCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Return the number of conditions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    /// Check if the pipeline has no conditions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Evaluate all conditions against an item (all must pass).
    pub fn evaluate<T: Filterable>(&self, item: &T) -> bool {
        self.conditions.iter().all(|c| c.evaluate(item))
    }

    /// Filter a collection, returning only matching items.
    #[must_use]
    pub fn apply<T: Filterable>(&self, items: Vec<T>) -> Vec<T> {
        items
            .into_iter()
            .filter(|item| self.evaluate(item))
            .collect()
    }

    /// Count how many items match.
    pub fn count_matches<T: Filterable>(&self, items: &[T]) -> usize {
        items.iter().filter(|item| self.evaluate(*item)).count()
    }
}

impl Default for FilterPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple test document for filter evaluation.
    struct TestDoc {
        title: String,
        score: f64,
        duration: i64,
        format: String,
        has_audio: bool,
    }

    impl TestDoc {
        fn new(title: &str, score: f64, duration: i64, format: &str, has_audio: bool) -> Self {
            Self {
                title: title.to_string(),
                score,
                duration,
                format: format.to_string(),
                has_audio,
            }
        }
    }

    impl Filterable for TestDoc {
        #[allow(clippy::cast_precision_loss)]
        fn field(&self, name: &str) -> FilterValue {
            match name {
                "title" => FilterValue::Str(self.title.clone()),
                "score" => FilterValue::Float(self.score),
                "duration" => FilterValue::Int(self.duration),
                "format" => FilterValue::Str(self.format.clone()),
                "has_audio" => FilterValue::Bool(self.has_audio),
                _ => FilterValue::Null,
            }
        }
    }

    #[test]
    fn test_numeric_compare_gt() {
        let doc = TestDoc::new("test", 0.8, 100, "mp4", true);
        let cond = FilterCondition::numeric_compare("score", CompareOp::Gt, 0.5);
        assert!(cond.evaluate(&doc));
    }

    #[test]
    fn test_numeric_compare_lt() {
        let doc = TestDoc::new("test", 0.3, 100, "mp4", true);
        let cond = FilterCondition::numeric_compare("score", CompareOp::Lt, 0.5);
        assert!(cond.evaluate(&doc));
    }

    #[test]
    fn test_numeric_range() {
        let doc = TestDoc::new("test", 0.5, 120, "mp4", true);
        let cond = FilterCondition::numeric_range("duration", 60.0, 180.0);
        assert!(cond.evaluate(&doc));
    }

    #[test]
    fn test_numeric_range_outside() {
        let doc = TestDoc::new("test", 0.5, 200, "mp4", true);
        let cond = FilterCondition::numeric_range("duration", 60.0, 180.0);
        assert!(!cond.evaluate(&doc));
    }

    #[test]
    fn test_string_contains() {
        let doc = TestDoc::new("My Great Video", 0.5, 100, "mp4", true);
        let cond = FilterCondition::string_contains("title", "great");
        assert!(cond.evaluate(&doc));
    }

    #[test]
    fn test_string_exact() {
        let doc = TestDoc::new("test", 0.5, 100, "mp4", true);
        let cond = FilterCondition::string_exact("format", "mp4");
        assert!(cond.evaluate(&doc));
        let cond2 = FilterCondition::string_exact("format", "avi");
        assert!(!cond2.evaluate(&doc));
    }

    #[test]
    fn test_string_prefix() {
        let doc = TestDoc::new("Sunset Timelapse", 0.5, 100, "mp4", true);
        let cond = FilterCondition::string_prefix("title", "sun");
        assert!(cond.evaluate(&doc));
    }

    #[test]
    fn test_in_set() {
        let doc = TestDoc::new("test", 0.5, 100, "mp4", true);
        let cond = FilterCondition::in_set("format", &["mp4", "mkv", "avi"]);
        assert!(cond.evaluate(&doc));
        let cond2 = FilterCondition::in_set("format", &["mkv", "avi"]);
        assert!(!cond2.evaluate(&doc));
    }

    #[test]
    fn test_is_null() {
        let doc = TestDoc::new("test", 0.5, 100, "mp4", true);
        let cond = FilterCondition::is_null("missing_field");
        assert!(cond.evaluate(&doc));
        let cond2 = FilterCondition::is_not_null("title");
        assert!(cond2.evaluate(&doc));
    }

    #[test]
    fn test_and_combination() {
        let doc = TestDoc::new("test", 0.8, 100, "mp4", true);
        let cond = FilterCondition::and(vec![
            FilterCondition::numeric_compare("score", CompareOp::Gt, 0.5),
            FilterCondition::string_exact("format", "mp4"),
        ]);
        assert!(cond.evaluate(&doc));
    }

    #[test]
    fn test_or_combination() {
        let doc = TestDoc::new("test", 0.3, 100, "avi", true);
        let cond = FilterCondition::or(vec![
            FilterCondition::string_exact("format", "mp4"),
            FilterCondition::string_exact("format", "avi"),
        ]);
        assert!(cond.evaluate(&doc));
    }

    #[test]
    fn test_not_condition() {
        let doc = TestDoc::new("test", 0.3, 100, "avi", true);
        let cond = FilterCondition::not(FilterCondition::string_exact("format", "mp4"));
        assert!(cond.evaluate(&doc));
    }

    #[test]
    fn test_filter_pipeline() {
        let items = vec![
            TestDoc::new("A", 0.9, 120, "mp4", true),
            TestDoc::new("B", 0.3, 60, "avi", false),
            TestDoc::new("C", 0.7, 90, "mp4", true),
        ];

        let pipeline = FilterPipeline::new()
            .with(FilterCondition::numeric_compare(
                "score",
                CompareOp::Gt,
                0.5,
            ))
            .with(FilterCondition::string_exact("format", "mp4"));

        let filtered = pipeline.apply(items);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_pipeline_count_matches() {
        let items = vec![
            TestDoc::new("A", 0.9, 120, "mp4", true),
            TestDoc::new("B", 0.3, 60, "avi", false),
            TestDoc::new("C", 0.7, 90, "mp4", true),
        ];

        let pipeline = FilterPipeline::new().with(FilterCondition::string_exact("format", "mp4"));

        assert_eq!(pipeline.count_matches(&items), 2);
    }

    #[test]
    fn test_empty_pipeline_passes_all() {
        let doc = TestDoc::new("test", 0.5, 100, "mp4", true);
        let pipeline = FilterPipeline::new();
        assert!(pipeline.is_empty());
        assert!(pipeline.evaluate(&doc));
    }

    #[test]
    fn test_filter_value_conversions() {
        let v = FilterValue::Int(42);
        assert_eq!(v.as_int(), Some(42));
        assert!((v.as_float().expect("should succeed in test") - 42.0).abs() < f64::EPSILON);
        assert!(v.as_str().is_none());
        assert!(v.as_bool().is_none());
        assert!(!v.is_null());

        let v2 = FilterValue::Null;
        assert!(v2.is_null());
    }
}
