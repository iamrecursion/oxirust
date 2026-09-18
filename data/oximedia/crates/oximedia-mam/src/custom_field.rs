//! User-defined metadata fields with type validation.
//!
//! Enables operators to extend the base asset schema with arbitrary custom
//! fields scoped to a "field set" (e.g. a project or asset type).  Each field
//! has a declared type, optional constraints (required, min/max, regex,
//! allowed values), and produces validated `FieldValue` instances.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Field type
// ---------------------------------------------------------------------------

/// The scalar type of a custom field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// UTF-8 text.
    Text,
    /// Long-form text (multi-line).
    TextArea,
    /// 64-bit signed integer.
    Integer,
    /// 64-bit IEEE-754 float.
    Float,
    /// Boolean flag.
    Boolean,
    /// ISO-8601 date string (YYYY-MM-DD).
    Date,
    /// ISO-8601 datetime string.
    DateTime,
    /// One value from a predefined list.
    Select(Vec<String>),
    /// Zero or more values from a predefined list.
    MultiSelect(Vec<String>),
    /// Hyperlink URL.
    Url,
    /// Email address.
    Email,
    /// Positive integer representing a duration in seconds.
    DurationSecs,
    /// File size in bytes (u64).
    FileSizeBytes,
}

impl FieldType {
    /// Human-readable type label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Text => "text",
            Self::TextArea => "textarea",
            Self::Integer => "integer",
            Self::Float => "float",
            Self::Boolean => "boolean",
            Self::Date => "date",
            Self::DateTime => "datetime",
            Self::Select(_) => "select",
            Self::MultiSelect(_) => "multiselect",
            Self::Url => "url",
            Self::Email => "email",
            Self::DurationSecs => "duration_secs",
            Self::FileSizeBytes => "file_size_bytes",
        }
    }
}

// ---------------------------------------------------------------------------
// Field definition
// ---------------------------------------------------------------------------

/// Constraints on a custom field value.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FieldConstraints {
    /// Whether a value is required (must be present and non-empty).
    pub required: bool,
    /// Minimum length for text fields.
    pub min_length: Option<usize>,
    /// Maximum length for text fields.
    pub max_length: Option<usize>,
    /// Minimum value for numeric fields.
    pub min_value: Option<f64>,
    /// Maximum value for numeric fields.
    pub max_value: Option<f64>,
    /// ECMAScript-compatible regex pattern for text validation.
    pub regex_pattern: Option<String>,
    /// Human-readable description of the constraint.
    pub description: Option<String>,
}

/// A custom field definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomFieldDef {
    /// Unique field id.
    pub id: Uuid,
    /// Machine-readable name (snake_case, unique within a field set).
    pub name: String,
    /// Human-readable label.
    pub label: String,
    /// Scalar type of this field.
    pub field_type: FieldType,
    /// Validation constraints.
    pub constraints: FieldConstraints,
    /// Default value (JSON).
    pub default_value: Option<serde_json::Value>,
    /// Display order within the field set.
    pub order: u32,
    /// Whether the field is currently active.
    pub active: bool,
    /// Optional help text shown in the UI.
    pub help_text: Option<String>,
}

impl CustomFieldDef {
    /// Create a new active field definition.
    #[must_use]
    pub fn new(name: impl Into<String>, label: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            label: label.into(),
            field_type,
            constraints: FieldConstraints::default(),
            default_value: None,
            order: 0,
            active: true,
            help_text: None,
        }
    }

    /// Builder: mark as required.
    #[must_use]
    pub fn required(mut self) -> Self {
        self.constraints.required = true;
        self
    }

    /// Builder: set text length limits.
    #[must_use]
    pub fn with_length_limits(mut self, min: Option<usize>, max: Option<usize>) -> Self {
        self.constraints.min_length = min;
        self.constraints.max_length = max;
        self
    }

    /// Builder: set numeric value limits.
    #[must_use]
    pub fn with_value_limits(mut self, min: Option<f64>, max: Option<f64>) -> Self {
        self.constraints.min_value = min;
        self.constraints.max_value = max;
        self
    }

    /// Builder: set a regex pattern.
    #[must_use]
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.constraints.regex_pattern = Some(pattern.into());
        self
    }

    /// Builder: set a default value.
    #[must_use]
    pub fn with_default(mut self, value: serde_json::Value) -> Self {
        self.default_value = Some(value);
        self
    }

    /// Builder: set display order.
    #[must_use]
    pub fn with_order(mut self, order: u32) -> Self {
        self.order = order;
        self
    }

    /// Builder: set help text.
    #[must_use]
    pub fn with_help(mut self, text: impl Into<String>) -> Self {
        self.help_text = Some(text.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Field value
// ---------------------------------------------------------------------------

/// A validated value for a custom field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    Text(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Date(String),
    DateTime(String),
    Select(String),
    MultiSelect(Vec<String>),
    Url(String),
    Email(String),
    DurationSecs(u64),
    FileSizeBytes(u64),
    /// Explicit null / not set.
    Null,
}

impl FieldValue {
    /// Returns `true` if this is the null variant.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Convert to JSON value.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Text(s)
            | Self::Date(s)
            | Self::DateTime(s)
            | Self::Select(s)
            | Self::Url(s)
            | Self::Email(s) => serde_json::json!(s),
            Self::Integer(i) => serde_json::json!(i),
            Self::Float(f) => serde_json::json!(f),
            Self::Boolean(b) => serde_json::json!(b),
            Self::MultiSelect(v) => serde_json::json!(v),
            Self::DurationSecs(s) => serde_json::json!(s),
            Self::FileSizeBytes(b) => serde_json::json!(b),
            Self::Null => serde_json::Value::Null,
        }
    }
}

// ---------------------------------------------------------------------------
// Validation error
// ---------------------------------------------------------------------------

/// A validation error produced when a value fails a constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationError {
    /// Field name.
    pub field: String,
    /// Human-readable description of the error.
    pub message: String,
}

impl ValidationError {
    fn new(field: &str, msg: impl Into<String>) -> Self {
        Self {
            field: field.to_string(),
            message: msg.into(),
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.field, self.message)
    }
}

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// Validates a `FieldValue` against a `CustomFieldDef`.
pub struct FieldValidator;

impl FieldValidator {
    /// Validate a value against its field definition.
    ///
    /// Returns `Ok(())` on success or `Err(Vec<ValidationError>)` listing all
    /// problems found.
    ///
    /// # Errors
    ///
    /// Returns a non-empty error vector when the value violates one or more
    /// constraints defined on the field.
    pub fn validate(def: &CustomFieldDef, value: &FieldValue) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        let name = &def.name;

        // Required check
        if def.constraints.required && value.is_null() {
            errors.push(ValidationError::new(name, "Field is required"));
            return Err(errors); // No point continuing on null
        }

        if value.is_null() {
            // Non-required null is always valid
            return Ok(());
        }

        // Type check + constraint checks
        match (&def.field_type, value) {
            (
                FieldType::Text | FieldType::TextArea | FieldType::Url | FieldType::Email,
                FieldValue::Text(s) | FieldValue::Url(s) | FieldValue::Email(s),
            ) => {
                Self::check_text(name, s, &def.constraints, &mut errors);
            }
            (FieldType::Integer, FieldValue::Integer(i)) => {
                let v = *i as f64;
                Self::check_numeric(name, v, &def.constraints, &mut errors);
            }
            (
                FieldType::Float | FieldType::DurationSecs | FieldType::FileSizeBytes,
                FieldValue::Float(f),
            ) => {
                Self::check_numeric(name, *f, &def.constraints, &mut errors);
            }
            (FieldType::DurationSecs, FieldValue::DurationSecs(d)) => {
                let v = *d as f64;
                Self::check_numeric(name, v, &def.constraints, &mut errors);
            }
            (FieldType::FileSizeBytes, FieldValue::FileSizeBytes(b)) => {
                let v = *b as f64;
                Self::check_numeric(name, v, &def.constraints, &mut errors);
            }
            (FieldType::Boolean, FieldValue::Boolean(_)) => {}
            (FieldType::Date, FieldValue::Date(d)) => {
                Self::validate_date_format(name, d, &mut errors);
            }
            (FieldType::DateTime, FieldValue::DateTime(dt)) => {
                // Very loose check — must contain 'T'
                if !dt.contains('T') {
                    errors.push(ValidationError::new(
                        name,
                        "Invalid datetime format (expected ISO-8601)",
                    ));
                }
            }
            (FieldType::Select(options), FieldValue::Select(v)) => {
                if !options.contains(v) {
                    errors.push(ValidationError::new(
                        name,
                        format!("Value '{v}' is not in the allowed options"),
                    ));
                }
            }
            (FieldType::MultiSelect(options), FieldValue::MultiSelect(vals)) => {
                for v in vals {
                    if !options.contains(v) {
                        errors.push(ValidationError::new(
                            name,
                            format!("Value '{v}' is not in the allowed options"),
                        ));
                    }
                }
            }
            _ => {
                errors.push(ValidationError::new(
                    name,
                    format!(
                        "Type mismatch: field expects '{}' but got incompatible value",
                        def.field_type.label()
                    ),
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn check_text(name: &str, s: &str, c: &FieldConstraints, errors: &mut Vec<ValidationError>) {
        if let Some(min) = c.min_length {
            if s.len() < min {
                errors.push(ValidationError::new(
                    name,
                    format!("Minimum length is {min}"),
                ));
            }
        }
        if let Some(max) = c.max_length {
            if s.len() > max {
                errors.push(ValidationError::new(
                    name,
                    format!("Maximum length is {max}"),
                ));
            }
        }
        if let Some(ref pattern) = c.regex_pattern {
            // Lightweight hand-rolled check: we only support anchored prefix/suffix patterns
            // using simple '*' wildcard for this implementation.
            if !simple_pattern_match(pattern, s) {
                errors.push(ValidationError::new(
                    name,
                    format!("Value does not match required pattern '{pattern}'"),
                ));
            }
        }
    }

    fn check_numeric(name: &str, v: f64, c: &FieldConstraints, errors: &mut Vec<ValidationError>) {
        if let Some(min) = c.min_value {
            if v < min {
                errors.push(ValidationError::new(
                    name,
                    format!("Minimum value is {min}"),
                ));
            }
        }
        if let Some(max) = c.max_value {
            if v > max {
                errors.push(ValidationError::new(
                    name,
                    format!("Maximum value is {max}"),
                ));
            }
        }
    }

    fn validate_date_format(name: &str, d: &str, errors: &mut Vec<ValidationError>) {
        // Expect YYYY-MM-DD
        let parts: Vec<&str> = d.split('-').collect();
        if parts.len() != 3
            || parts[0].len() != 4
            || parts[1].len() != 2
            || parts[2].len() != 2
            || parts.iter().any(|p| !p.chars().all(|c| c.is_ascii_digit()))
        {
            errors.push(ValidationError::new(
                name,
                "Invalid date format (expected YYYY-MM-DD)",
            ));
        }
    }
}

/// Simplified pattern matcher supporting `*` as "any sequence of characters".
fn simple_pattern_match(pattern: &str, s: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if !pattern.contains('*') {
        return s == pattern;
    }
    // Split on * and require each segment to appear in order
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut remaining = s;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            // First segment must be a prefix
            if !remaining.starts_with(part) {
                return false;
            }
            remaining = &remaining[part.len()..];
        } else if i == parts.len() - 1 && !pattern.ends_with('*') {
            // Last segment must be a suffix
            if !remaining.ends_with(part) {
                return false;
            }
        } else if let Some(pos) = remaining.find(part) {
            remaining = &remaining[pos + part.len()..];
        } else {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Field set
// ---------------------------------------------------------------------------

/// A named collection of custom field definitions scoped to a project or type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSet {
    pub id: Uuid,
    /// Logical name for the field set (e.g. "broadcast_metadata").
    pub name: String,
    /// Human-readable label.
    pub label: String,
    /// Ordered list of field definitions (sorted by `order`).
    pub fields: Vec<CustomFieldDef>,
}

impl FieldSet {
    /// Create a new empty field set.
    #[must_use]
    pub fn new(name: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            label: label.into(),
            fields: Vec::new(),
        }
    }

    /// Add a field definition to the set.
    pub fn add_field(&mut self, mut def: CustomFieldDef) {
        def.order = self.fields.len() as u32;
        self.fields.push(def);
        self.fields.sort_by_key(|f| f.order);
    }

    /// Look up a field by name.
    #[must_use]
    pub fn field_by_name(&self, name: &str) -> Option<&CustomFieldDef> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Active fields only.
    #[must_use]
    pub fn active_fields(&self) -> Vec<&CustomFieldDef> {
        self.fields.iter().filter(|f| f.active).collect()
    }

    /// Validate a map of field values against this field set.
    ///
    /// Returns a map of field name → list of errors.  An empty map means all
    /// values are valid.  Fields in the set that are required but absent from
    /// the map are reported as errors.
    #[must_use]
    pub fn validate_values(
        &self,
        values: &HashMap<String, FieldValue>,
    ) -> HashMap<String, Vec<ValidationError>> {
        let mut report: HashMap<String, Vec<ValidationError>> = HashMap::new();

        for def in self.active_fields() {
            let value = values.get(&def.name).cloned().unwrap_or(FieldValue::Null);
            if let Err(errs) = FieldValidator::validate(def, &value) {
                report.insert(def.name.clone(), errs);
            }
        }

        report
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- FieldType ---

    #[test]
    fn test_field_type_labels() {
        assert_eq!(FieldType::Text.label(), "text");
        assert_eq!(FieldType::Integer.label(), "integer");
        assert_eq!(FieldType::Select(vec![]).label(), "select");
        assert_eq!(FieldType::MultiSelect(vec![]).label(), "multiselect");
    }

    // --- CustomFieldDef builder ---

    #[test]
    fn test_field_def_builder() {
        let def = CustomFieldDef::new("budget", "Budget (USD)", FieldType::Float)
            .required()
            .with_value_limits(Some(0.0), Some(1_000_000.0))
            .with_help("Enter the production budget in USD");
        assert!(def.constraints.required);
        assert_eq!(def.constraints.min_value, Some(0.0));
        assert_eq!(def.constraints.max_value, Some(1_000_000.0));
        assert!(def.help_text.is_some());
    }

    // --- FieldValue ---

    #[test]
    fn test_field_value_is_null() {
        assert!(FieldValue::Null.is_null());
        assert!(!FieldValue::Text("hi".to_string()).is_null());
    }

    #[test]
    fn test_field_value_to_json() {
        assert_eq!(FieldValue::Integer(42).to_json(), serde_json::json!(42));
        assert_eq!(FieldValue::Boolean(true).to_json(), serde_json::json!(true));
        assert_eq!(FieldValue::Null.to_json(), serde_json::Value::Null);
        assert_eq!(
            FieldValue::MultiSelect(vec!["a".into(), "b".into()]).to_json(),
            serde_json::json!(["a", "b"])
        );
    }

    // --- FieldValidator ---

    #[test]
    fn test_validate_required_null() {
        let def = CustomFieldDef::new("title", "Title", FieldType::Text).required();
        assert!(FieldValidator::validate(&def, &FieldValue::Null).is_err());
    }

    #[test]
    fn test_validate_optional_null_ok() {
        let def = CustomFieldDef::new("note", "Note", FieldType::Text);
        assert!(FieldValidator::validate(&def, &FieldValue::Null).is_ok());
    }

    #[test]
    fn test_validate_text_min_length() {
        let def =
            CustomFieldDef::new("slug", "Slug", FieldType::Text).with_length_limits(Some(3), None);
        assert!(FieldValidator::validate(&def, &FieldValue::Text("ab".to_string())).is_err());
        assert!(FieldValidator::validate(&def, &FieldValue::Text("abc".to_string())).is_ok());
    }

    #[test]
    fn test_validate_text_max_length() {
        let def =
            CustomFieldDef::new("tag", "Tag", FieldType::Text).with_length_limits(None, Some(5));
        assert!(FieldValidator::validate(&def, &FieldValue::Text("toolong".to_string())).is_err());
        assert!(FieldValidator::validate(&def, &FieldValue::Text("ok".to_string())).is_ok());
    }

    #[test]
    fn test_validate_integer_range() {
        let def = CustomFieldDef::new("rating", "Rating", FieldType::Integer)
            .with_value_limits(Some(1.0), Some(5.0));
        assert!(FieldValidator::validate(&def, &FieldValue::Integer(0)).is_err());
        assert!(FieldValidator::validate(&def, &FieldValue::Integer(6)).is_err());
        assert!(FieldValidator::validate(&def, &FieldValue::Integer(3)).is_ok());
    }

    #[test]
    fn test_validate_select_valid() {
        let def = CustomFieldDef::new(
            "status",
            "Status",
            FieldType::Select(vec!["draft".to_string(), "final".to_string()]),
        );
        assert!(FieldValidator::validate(&def, &FieldValue::Select("draft".to_string())).is_ok());
        assert!(
            FieldValidator::validate(&def, &FieldValue::Select("unknown".to_string())).is_err()
        );
    }

    #[test]
    fn test_validate_multiselect() {
        let def = CustomFieldDef::new(
            "tags",
            "Tags",
            FieldType::MultiSelect(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        );
        assert!(FieldValidator::validate(
            &def,
            &FieldValue::MultiSelect(vec!["a".to_string(), "b".to_string()])
        )
        .is_ok());
        assert!(FieldValidator::validate(
            &def,
            &FieldValue::MultiSelect(vec!["a".to_string(), "z".to_string()])
        )
        .is_err());
    }

    #[test]
    fn test_validate_date_format_ok() {
        let def = CustomFieldDef::new("release", "Release Date", FieldType::Date);
        assert!(
            FieldValidator::validate(&def, &FieldValue::Date("2024-03-15".to_string())).is_ok()
        );
    }

    #[test]
    fn test_validate_date_format_bad() {
        let def = CustomFieldDef::new("release", "Release Date", FieldType::Date);
        assert!(
            FieldValidator::validate(&def, &FieldValue::Date("15/03/2024".to_string())).is_err()
        );
        assert!(FieldValidator::validate(&def, &FieldValue::Date("2024-3-5".to_string())).is_err());
    }

    #[test]
    fn test_validate_datetime_ok() {
        let def = CustomFieldDef::new("ts", "Timestamp", FieldType::DateTime);
        assert!(FieldValidator::validate(
            &def,
            &FieldValue::DateTime("2024-03-15T12:00:00Z".to_string())
        )
        .is_ok());
    }

    #[test]
    fn test_validate_datetime_bad() {
        let def = CustomFieldDef::new("ts", "Timestamp", FieldType::DateTime);
        assert!(
            FieldValidator::validate(&def, &FieldValue::DateTime("2024-03-15".to_string()))
                .is_err()
        );
    }

    #[test]
    fn test_validate_type_mismatch() {
        let def = CustomFieldDef::new("count", "Count", FieldType::Integer);
        // Providing a Float value for an Integer field
        assert!(FieldValidator::validate(&def, &FieldValue::Float(3.14)).is_err());
    }

    #[test]
    fn test_validate_boolean() {
        let def = CustomFieldDef::new("active", "Active", FieldType::Boolean);
        assert!(FieldValidator::validate(&def, &FieldValue::Boolean(false)).is_ok());
        assert!(FieldValidator::validate(&def, &FieldValue::Boolean(true)).is_ok());
    }

    #[test]
    fn test_validate_duration_secs() {
        let def = CustomFieldDef::new("length", "Length", FieldType::DurationSecs)
            .with_value_limits(Some(1.0), Some(7200.0));
        assert!(FieldValidator::validate(&def, &FieldValue::DurationSecs(3600)).is_ok());
        assert!(FieldValidator::validate(&def, &FieldValue::DurationSecs(0)).is_err());
    }

    // --- FieldSet ---

    #[test]
    fn test_field_set_add_and_lookup() {
        let mut fs = FieldSet::new("broadcast", "Broadcast Metadata");
        fs.add_field(CustomFieldDef::new("title", "Title", FieldType::Text).required());
        fs.add_field(CustomFieldDef::new(
            "episode",
            "Episode",
            FieldType::Integer,
        ));

        assert_eq!(fs.active_fields().len(), 2);
        assert!(fs.field_by_name("title").is_some());
        assert!(fs.field_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_field_set_validate_values_all_ok() {
        let mut fs = FieldSet::new("meta", "Meta");
        fs.add_field(CustomFieldDef::new("title", "Title", FieldType::Text).required());
        fs.add_field(
            CustomFieldDef::new("rating", "Rating", FieldType::Integer)
                .with_value_limits(Some(1.0), Some(10.0)),
        );

        let mut values = HashMap::new();
        values.insert(
            "title".to_string(),
            FieldValue::Text("My Movie".to_string()),
        );
        values.insert("rating".to_string(), FieldValue::Integer(8));

        let errors = fs.validate_values(&values);
        assert!(errors.is_empty(), "Expected no errors, got: {errors:?}");
    }

    #[test]
    fn test_field_set_validate_values_missing_required() {
        let mut fs = FieldSet::new("meta", "Meta");
        fs.add_field(CustomFieldDef::new("title", "Title", FieldType::Text).required());

        let errors = fs.validate_values(&HashMap::new());
        assert!(errors.contains_key("title"));
    }

    #[test]
    fn test_field_set_validate_values_bad_range() {
        let mut fs = FieldSet::new("meta", "Meta");
        fs.add_field(
            CustomFieldDef::new("rating", "Rating", FieldType::Integer)
                .with_value_limits(Some(1.0), Some(5.0)),
        );

        let mut values = HashMap::new();
        values.insert("rating".to_string(), FieldValue::Integer(99));

        let errors = fs.validate_values(&values);
        assert!(errors.contains_key("rating"));
    }

    #[test]
    fn test_simple_pattern_wildcard_all() {
        assert!(simple_pattern_match("*", "anything"));
        assert!(simple_pattern_match("*", ""));
    }

    #[test]
    fn test_simple_pattern_prefix() {
        assert!(simple_pattern_match("hello*", "hello world"));
        assert!(!simple_pattern_match("hello*", "world"));
    }

    #[test]
    fn test_simple_pattern_exact() {
        assert!(simple_pattern_match("exact", "exact"));
        assert!(!simple_pattern_match("exact", "not exact"));
    }

    #[test]
    fn test_field_value_text_pattern_constraint() {
        // Simulate a prefix pattern check
        let def = CustomFieldDef::new("code", "Code", FieldType::Text).with_pattern("OX*");
        // "OX123" should match "OX*"
        assert!(FieldValidator::validate(&def, &FieldValue::Text("OX123".to_string())).is_ok());
        // "AB123" should not match
        assert!(FieldValidator::validate(&def, &FieldValue::Text("AB123".to_string())).is_err());
    }
}
