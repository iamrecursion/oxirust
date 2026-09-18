#![allow(dead_code)]
//! Metadata field validation utilities.
//!
//! Validates individual metadata field values against type rules,
//! collecting errors into a structured report.

use std::collections::HashMap;

/// The logical type expected for a metadata field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldType {
    /// Plain text; may optionally be required to be non-empty.
    Text { required: bool },
    /// A numeric value stored as text.
    Number,
    /// A date/time string (ISO 8601 subset).
    DateTime,
    /// A URL string.
    Url,
    /// A free-form tag that must match an identifier pattern.
    Tag,
    /// An integer stored as text.
    Integer,
}

impl FieldType {
    /// Returns `true` if an empty string is acceptable for this type.
    pub fn allows_empty(self) -> bool {
        match self {
            Self::Text { required } => !required,
            Self::Number | Self::Integer | Self::DateTime | Self::Url | Self::Tag => false,
        }
    }

    /// A human-readable label for the type.
    pub fn label(self) -> &'static str {
        match self {
            Self::Text { .. } => "text",
            Self::Number => "number",
            Self::DateTime => "datetime",
            Self::Url => "url",
            Self::Tag => "tag",
            Self::Integer => "integer",
        }
    }
}

/// Describes a single validation failure.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    field: String,
    message: String,
}

impl ValidationError {
    /// Create a new validation error.
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Returns the name of the field that failed validation.
    pub fn field_name(&self) -> &str {
        &self.field
    }

    /// Returns the human-readable error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.field, self.message)
    }
}

/// Validates metadata field values against declared types.
pub struct FieldValidator {
    rules: HashMap<String, FieldType>,
}

impl FieldValidator {
    /// Create an empty validator with no rules.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Register a field with its expected type.
    pub fn add_rule(&mut self, field: impl Into<String>, field_type: FieldType) {
        self.rules.insert(field.into(), field_type);
    }

    /// Validate a string field value.
    ///
    /// Returns `Ok(())` if valid, or a `ValidationError` describing the problem.
    pub fn validate_string(&self, field: &str, value: &str) -> Result<(), ValidationError> {
        let ft = match self.rules.get(field) {
            Some(t) => *t,
            None => return Ok(()), // unknown field; skip
        };

        if value.is_empty() && !ft.allows_empty() {
            return Err(ValidationError::new(
                field,
                format!("{} field must not be empty", ft.label()),
            ));
        }

        match ft {
            FieldType::Number => {
                if value.parse::<f64>().is_err() {
                    return Err(ValidationError::new(
                        field,
                        format!("expected a numeric value, got '{value}'"),
                    ));
                }
            }
            FieldType::Integer => {
                if value.parse::<i64>().is_err() {
                    return Err(ValidationError::new(
                        field,
                        format!("expected an integer value, got '{value}'"),
                    ));
                }
            }
            FieldType::DateTime => {
                // Very lightweight ISO 8601 date check: YYYY-MM-DD prefix.
                if value.len() < 10 || value.chars().nth(4) != Some('-') {
                    return Err(ValidationError::new(
                        field,
                        format!("expected ISO 8601 date, got '{value}'"),
                    ));
                }
            }
            FieldType::Url => {
                if !value.starts_with("http://") && !value.starts_with("https://") {
                    return Err(ValidationError::new(
                        field,
                        format!("expected http/https URL, got '{value}'"),
                    ));
                }
            }
            FieldType::Tag => {
                let ok = value
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-');
                if !ok {
                    return Err(ValidationError::new(
                        field,
                        format!("tag must be alphanumeric/dash/underscore, got '{value}'"),
                    ));
                }
            }
            FieldType::Text { .. } => {}
        }

        Ok(())
    }

    /// Validate a numeric value using the configured rule.
    ///
    /// Checks that `value` satisfies the numeric type's constraints (non-NaN, finite for f64).
    pub fn validate_number(&self, field: &str, value: f64) -> Result<(), ValidationError> {
        if !value.is_finite() {
            return Err(ValidationError::new(
                field,
                "numeric value must be finite (not NaN or infinite)",
            ));
        }
        Ok(())
    }

    /// Returns a reference to the registered rules.
    pub fn rules(&self) -> &HashMap<String, FieldType> {
        &self.rules
    }
}

impl Default for FieldValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregates validation errors across multiple fields.
#[derive(Debug, Default)]
pub struct FieldValidationReport {
    errors: Vec<ValidationError>,
}

impl FieldValidationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Add a validation error to the report.
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    /// Run the validator against all fields in a map, populating this report.
    pub fn validate_all(&mut self, validator: &FieldValidator, fields: &HashMap<String, String>) {
        for (field, value) in fields {
            if let Err(e) = validator.validate_string(field, value) {
                self.add_error(e);
            }
        }
    }

    /// Returns `true` if there are any errors in the report.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Returns all collected errors.
    pub fn errors(&self) -> &[ValidationError] {
        &self.errors
    }

    /// Returns errors for a specific field.
    pub fn errors_for(&self, field: &str) -> Vec<&ValidationError> {
        self.errors
            .iter()
            .filter(|e| e.field_name() == field)
            .collect()
    }

    /// Returns the total number of errors.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validator_with_rules() -> FieldValidator {
        let mut v = FieldValidator::new();
        v.add_rule("title", FieldType::Text { required: true });
        v.add_rule("year", FieldType::Integer);
        v.add_rule("rating", FieldType::Number);
        v.add_rule("released", FieldType::DateTime);
        v.add_rule("source_url", FieldType::Url);
        v.add_rule("genre_tag", FieldType::Tag);
        v
    }

    #[test]
    fn test_field_type_allows_empty_optional_text() {
        assert!(FieldType::Text { required: false }.allows_empty());
    }

    #[test]
    fn test_field_type_required_text_no_empty() {
        assert!(!FieldType::Text { required: true }.allows_empty());
    }

    #[test]
    fn test_field_type_number_no_empty() {
        assert!(!FieldType::Number.allows_empty());
    }

    #[test]
    fn test_validate_string_valid_title() {
        let v = validator_with_rules();
        assert!(v.validate_string("title", "My Song").is_ok());
    }

    #[test]
    fn test_validate_string_empty_required_text_fails() {
        let v = validator_with_rules();
        assert!(v.validate_string("title", "").is_err());
    }

    #[test]
    fn test_validate_string_valid_integer() {
        let v = validator_with_rules();
        assert!(v.validate_string("year", "2024").is_ok());
    }

    #[test]
    fn test_validate_string_invalid_integer() {
        let v = validator_with_rules();
        let err = v.validate_string("year", "not_a_number").unwrap_err();
        assert_eq!(err.field_name(), "year");
    }

    #[test]
    fn test_validate_string_valid_number() {
        let v = validator_with_rules();
        assert!(v.validate_string("rating", "9.5").is_ok());
    }

    #[test]
    fn test_validate_string_invalid_number() {
        let v = validator_with_rules();
        assert!(v.validate_string("rating", "abc").is_err());
    }

    #[test]
    fn test_validate_string_valid_datetime() {
        let v = validator_with_rules();
        assert!(v.validate_string("released", "2024-06-01").is_ok());
    }

    #[test]
    fn test_validate_string_invalid_datetime() {
        let v = validator_with_rules();
        assert!(v.validate_string("released", "June 2024").is_err());
    }

    #[test]
    fn test_validate_string_valid_url() {
        let v = validator_with_rules();
        assert!(v
            .validate_string("source_url", "https://example.com")
            .is_ok());
    }

    #[test]
    fn test_validate_string_invalid_url() {
        let v = validator_with_rules();
        assert!(v.validate_string("source_url", "ftp://old.net").is_err());
    }

    #[test]
    fn test_validate_string_valid_tag() {
        let v = validator_with_rules();
        assert!(v.validate_string("genre_tag", "rock-pop").is_ok());
    }

    #[test]
    fn test_validate_string_invalid_tag() {
        let v = validator_with_rules();
        assert!(v.validate_string("genre_tag", "rock pop!").is_err());
    }

    #[test]
    fn test_validate_number_finite() {
        let v = validator_with_rules();
        assert!(v.validate_number("rating", 7.5).is_ok());
    }

    #[test]
    fn test_validate_number_nan_fails() {
        let v = validator_with_rules();
        assert!(v.validate_number("rating", f64::NAN).is_err());
    }

    #[test]
    fn test_validation_report_no_errors() {
        let v = validator_with_rules();
        let mut report = FieldValidationReport::new();
        let fields: HashMap<String, String> = [
            ("title".to_string(), "Song".to_string()),
            ("year".to_string(), "2023".to_string()),
        ]
        .into();
        report.validate_all(&v, &fields);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_validation_report_has_errors() {
        let v = validator_with_rules();
        let mut report = FieldValidationReport::new();
        let fields: HashMap<String, String> = [("year".to_string(), "bad".to_string())].into();
        report.validate_all(&v, &fields);
        assert!(report.has_errors());
        assert_eq!(report.errors_for("year").len(), 1);
    }

    #[test]
    fn test_validation_error_display() {
        let e = ValidationError::new("field", "bad value");
        let s = e.to_string();
        assert!(s.contains("field"));
        assert!(s.contains("bad value"));
    }
}
