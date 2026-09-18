//! Metadata schema validation.
//!
//! Provides schema definitions and validation for metadata fields,
//! including pre-built schemas for Dublin Core and XMP Basic.

use std::collections::HashMap;

/// The data type of a schema field.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    /// Free-form text.
    Text,
    /// Integer number.
    Integer,
    /// Floating-point number.
    Float,
    /// ISO 8601 date string.
    Date,
    /// Boolean true/false.
    Boolean,
    /// Comma-separated list of values.
    List,
    /// URL / URI.
    Url,
}

impl FieldType {
    /// Return a human-readable name for the field type.
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "Text",
            Self::Integer => "Integer",
            Self::Float => "Float",
            Self::Date => "Date",
            Self::Boolean => "Boolean",
            Self::List => "List",
            Self::Url => "Url",
        }
    }

    /// Check whether the supplied string value is valid for this type.
    fn validate_value(&self, value: &str) -> bool {
        match self {
            Self::Text | Self::List => true,
            Self::Integer => value.parse::<i64>().is_ok(),
            Self::Float => value.parse::<f64>().is_ok(),
            Self::Date => {
                // Accept YYYY, YYYY-MM, YYYY-MM-DD, or YYYY-MM-DDTHH:MM:SS patterns
                value.len() >= 4 && value.chars().next().is_some_and(|c| c.is_ascii_digit())
            }
            Self::Boolean => matches!(value.to_lowercase().as_str(), "true" | "false" | "1" | "0"),
            Self::Url => value.starts_with("http://") || value.starts_with("https://"),
        }
    }
}

/// Schema definition for a single metadata field.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FieldSchema {
    /// Field name / key.
    pub name: String,
    /// Expected data type.
    pub field_type: FieldType,
    /// Whether this field must be present.
    pub required: bool,
    /// Optional maximum string length.
    pub max_length: Option<usize>,
    /// If non-empty, the value must be one of these strings.
    pub allowed_values: Vec<String>,
}

impl FieldSchema {
    /// Create a new field schema.
    #[allow(dead_code)]
    pub fn new(name: &str, field_type: FieldType, required: bool) -> Self {
        Self {
            name: name.to_string(),
            field_type,
            required,
            max_length: None,
            allowed_values: Vec::new(),
        }
    }

    /// Set the maximum allowed length.
    #[allow(dead_code)]
    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = Some(max);
        self
    }

    /// Restrict to a set of allowed string values.
    #[allow(dead_code)]
    pub fn with_allowed_values(mut self, values: Vec<String>) -> Self {
        self.allowed_values = values;
        self
    }

    /// Validate a single value against this field's constraints.
    /// Returns a list of error messages (empty means valid).
    fn validate_value(&self, value: &str) -> Vec<String> {
        let mut errors = Vec::new();

        if !self.field_type.validate_value(value) {
            errors.push(format!(
                "Field '{}': value '{}' is not a valid {}",
                self.name,
                value,
                self.field_type.as_str()
            ));
        }

        if let Some(max) = self.max_length {
            if value.len() > max {
                errors.push(format!(
                    "Field '{}': value length {} exceeds maximum {}",
                    self.name,
                    value.len(),
                    max
                ));
            }
        }

        if !self.allowed_values.is_empty() && !self.allowed_values.iter().any(|a| a == value) {
            errors.push(format!(
                "Field '{}': value '{}' is not in the allowed set {:?}",
                self.name, value, self.allowed_values
            ));
        }

        errors
    }
}

/// A named schema that groups multiple field definitions.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MetadataSchema {
    /// Schema name (e.g., "Dublin Core", "XMP Basic").
    pub name: String,
    /// Field definitions.
    pub fields: Vec<FieldSchema>,
}

impl MetadataSchema {
    /// Create a new, empty metadata schema.
    #[allow(dead_code)]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            fields: Vec::new(),
        }
    }

    /// Add a field definition to this schema.
    #[allow(dead_code)]
    pub fn add_field(&mut self, field: FieldSchema) {
        self.fields.push(field);
    }

    /// Validate a metadata map against this schema.
    /// Returns a list of validation error messages.
    #[allow(dead_code)]
    pub fn validate(&self, data: &HashMap<String, String>) -> Vec<String> {
        let mut errors = Vec::new();

        // Check required fields are present
        for field in &self.fields {
            if field.required && !data.contains_key(&field.name) {
                errors.push(format!("Required field '{}' is missing", field.name));
            }
        }

        // Validate present values
        let field_map: HashMap<&str, &FieldSchema> =
            self.fields.iter().map(|f| (f.name.as_str(), f)).collect();

        for (key, value) in data {
            if let Some(field) = field_map.get(key.as_str()) {
                errors.extend(field.validate_value(value));
            }
            // Unknown fields are allowed by default (permissive schema)
        }

        errors
    }

    /// Return all required fields.
    #[allow(dead_code)]
    pub fn required_fields(&self) -> Vec<&FieldSchema> {
        self.fields.iter().filter(|f| f.required).collect()
    }
}

/// Build the standard Dublin Core metadata schema.
#[allow(dead_code)]
pub fn dublin_core_schema() -> MetadataSchema {
    let mut schema = MetadataSchema::new("Dublin Core");

    // DC elements
    schema.add_field(FieldSchema::new("title", FieldType::Text, false).with_max_length(1024));
    schema.add_field(FieldSchema::new("creator", FieldType::Text, false).with_max_length(512));
    schema.add_field(FieldSchema::new("subject", FieldType::List, false));
    schema.add_field(FieldSchema::new("description", FieldType::Text, false).with_max_length(4096));
    schema.add_field(FieldSchema::new("publisher", FieldType::Text, false).with_max_length(512));
    schema.add_field(FieldSchema::new("contributor", FieldType::Text, false));
    schema.add_field(FieldSchema::new("date", FieldType::Date, false));
    schema.add_field(
        FieldSchema::new("type", FieldType::Text, false).with_allowed_values(vec![
            "Collection".to_string(),
            "Dataset".to_string(),
            "Event".to_string(),
            "Image".to_string(),
            "InteractiveResource".to_string(),
            "MovingImage".to_string(),
            "PhysicalObject".to_string(),
            "Service".to_string(),
            "Software".to_string(),
            "Sound".to_string(),
            "StillImage".to_string(),
            "Text".to_string(),
        ]),
    );
    schema.add_field(FieldSchema::new("format", FieldType::Text, false).with_max_length(256));
    schema.add_field(FieldSchema::new("identifier", FieldType::Text, false));
    schema.add_field(FieldSchema::new("source", FieldType::Text, false));
    schema.add_field(FieldSchema::new("language", FieldType::Text, false).with_max_length(64));
    schema.add_field(FieldSchema::new("relation", FieldType::Text, false));
    schema.add_field(FieldSchema::new("coverage", FieldType::Text, false));
    schema.add_field(FieldSchema::new("rights", FieldType::Text, false).with_max_length(2048));

    schema
}

/// Build the XMP Basic namespace schema.
#[allow(dead_code)]
pub fn xmp_basic_schema() -> MetadataSchema {
    let mut schema = MetadataSchema::new("XMP Basic");

    schema.add_field(FieldSchema::new("xmp:CreateDate", FieldType::Date, false));
    schema.add_field(FieldSchema::new("xmp:ModifyDate", FieldType::Date, false));
    schema.add_field(FieldSchema::new("xmp:MetadataDate", FieldType::Date, false));
    schema.add_field(
        FieldSchema::new("xmp:CreatorTool", FieldType::Text, false).with_max_length(512),
    );
    schema.add_field(FieldSchema::new("xmp:Identifier", FieldType::Text, false));
    schema.add_field(FieldSchema::new("xmp:Label", FieldType::Text, false).with_max_length(64));
    schema.add_field(FieldSchema::new("xmp:Rating", FieldType::Integer, false));
    schema.add_field(FieldSchema::new("xmp:Nickname", FieldType::Text, false).with_max_length(256));
    schema.add_field(FieldSchema::new("xmp:Thumbnails", FieldType::Text, false));

    schema
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_field_type_as_str() {
        assert_eq!(FieldType::Text.as_str(), "Text");
        assert_eq!(FieldType::Integer.as_str(), "Integer");
        assert_eq!(FieldType::Float.as_str(), "Float");
        assert_eq!(FieldType::Date.as_str(), "Date");
        assert_eq!(FieldType::Boolean.as_str(), "Boolean");
        assert_eq!(FieldType::List.as_str(), "List");
        assert_eq!(FieldType::Url.as_str(), "Url");
    }

    #[test]
    fn test_integer_validation() {
        assert!(FieldType::Integer.validate_value("42"));
        assert!(FieldType::Integer.validate_value("-100"));
        assert!(!FieldType::Integer.validate_value("3.14"));
        assert!(!FieldType::Integer.validate_value("abc"));
    }

    #[test]
    fn test_float_validation() {
        assert!(FieldType::Float.validate_value("3.14"));
        assert!(FieldType::Float.validate_value("100"));
        assert!(!FieldType::Float.validate_value("abc"));
    }

    #[test]
    fn test_boolean_validation() {
        assert!(FieldType::Boolean.validate_value("true"));
        assert!(FieldType::Boolean.validate_value("false"));
        assert!(FieldType::Boolean.validate_value("1"));
        assert!(FieldType::Boolean.validate_value("0"));
        assert!(!FieldType::Boolean.validate_value("yes"));
    }

    #[test]
    fn test_url_validation() {
        assert!(FieldType::Url.validate_value("https://example.com"));
        assert!(FieldType::Url.validate_value("http://example.com/path"));
        assert!(!FieldType::Url.validate_value("ftp://example.com"));
        assert!(!FieldType::Url.validate_value("just-text"));
    }

    #[test]
    fn test_date_validation() {
        assert!(FieldType::Date.validate_value("2024-01-15"));
        assert!(FieldType::Date.validate_value("2024"));
        assert!(!FieldType::Date.validate_value("abc"));
        assert!(!FieldType::Date.validate_value(""));
    }

    #[test]
    fn test_schema_validate_required_missing() {
        let mut schema = MetadataSchema::new("Test");
        schema.add_field(FieldSchema::new("title", FieldType::Text, true));
        let data = make_data(&[]);
        let errors = schema.validate(&data);
        assert!(errors.iter().any(|e| e.contains("title")));
    }

    #[test]
    fn test_schema_validate_max_length() {
        let mut schema = MetadataSchema::new("Test");
        schema.add_field(FieldSchema::new("title", FieldType::Text, false).with_max_length(5));
        let data = make_data(&[("title", "This is too long")]);
        let errors = schema.validate(&data);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_schema_validate_allowed_values() {
        let mut schema = MetadataSchema::new("Test");
        schema.add_field(
            FieldSchema::new("status", FieldType::Text, false)
                .with_allowed_values(vec!["active".to_string(), "inactive".to_string()]),
        );
        let data_ok = make_data(&[("status", "active")]);
        assert!(schema.validate(&data_ok).is_empty());
        let data_bad = make_data(&[("status", "pending")]);
        assert!(!schema.validate(&data_bad).is_empty());
    }

    #[test]
    fn test_required_fields() {
        let mut schema = MetadataSchema::new("Test");
        schema.add_field(FieldSchema::new("a", FieldType::Text, true));
        schema.add_field(FieldSchema::new("b", FieldType::Text, false));
        schema.add_field(FieldSchema::new("c", FieldType::Text, true));
        let required: Vec<&str> = schema
            .required_fields()
            .iter()
            .map(|f| f.name.as_str())
            .collect();
        assert_eq!(required.len(), 2);
        assert!(required.contains(&"a"));
        assert!(required.contains(&"c"));
    }

    #[test]
    fn test_dublin_core_schema() {
        let schema = dublin_core_schema();
        assert_eq!(schema.name, "Dublin Core");
        assert!(!schema.fields.is_empty());
        // None of the DC elements are required
        assert!(schema.required_fields().is_empty());
    }

    #[test]
    fn test_xmp_basic_schema() {
        let schema = xmp_basic_schema();
        assert_eq!(schema.name, "XMP Basic");
        assert!(!schema.fields.is_empty());
    }

    #[test]
    fn test_dublin_core_type_validation() {
        let schema = dublin_core_schema();
        let data_ok = make_data(&[("type", "MovingImage")]);
        assert!(schema.validate(&data_ok).is_empty());
        let data_bad = make_data(&[("type", "Unknown")]);
        assert!(!schema.validate(&data_bad).is_empty());
    }

    #[test]
    fn test_valid_data_produces_no_errors() {
        let mut schema = MetadataSchema::new("Minimal");
        schema.add_field(FieldSchema::new("title", FieldType::Text, true));
        schema.add_field(FieldSchema::new("year", FieldType::Integer, false));
        let data = make_data(&[("title", "My Film"), ("year", "2024")]);
        assert!(schema.validate(&data).is_empty());
    }
}
