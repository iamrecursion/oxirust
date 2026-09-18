//! Metadata template system for MAM.
//!
//! Allows users to define named templates composed of typed fields with
//! optional default values.  Templates can be applied to assets to
//! pre-populate metadata during ingest workflows.

#![allow(dead_code)]

use std::collections::HashMap;

/// Supported data types for template fields.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FieldType {
    /// Free-form text.
    Text,
    /// Integer number.
    Integer,
    /// Decimal number (stored as string to avoid fp issues in templates).
    Decimal,
    /// Boolean (`true` / `false`).
    Boolean,
    /// Date string (ISO 8601: `YYYY-MM-DD`).
    Date,
    /// Enumerated choice – the allowed values are stored in `allowed_values`.
    Enum,
    /// URL / URI.
    Url,
}

/// A single field definition within a template.
#[derive(Clone, Debug)]
pub struct TemplateField {
    /// Field identifier / key.
    pub name: String,
    /// Human-readable label.
    pub label: String,
    /// Data type.
    pub field_type: FieldType,
    /// Default value (empty string = no default).
    pub default_value: String,
    /// Whether this field must be filled before applying the template.
    pub required: bool,
    /// Allowed values for [`FieldType::Enum`] fields.
    pub allowed_values: Vec<String>,
    /// Freeform description shown in UI.
    pub description: String,
}

impl TemplateField {
    /// Create a simple text field.
    #[must_use]
    pub fn text(name: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            field_type: FieldType::Text,
            default_value: String::new(),
            required: false,
            allowed_values: Vec::new(),
            description: String::new(),
        }
    }

    /// Create a required text field.
    #[must_use]
    pub fn required_text(name: impl Into<String>, label: impl Into<String>) -> Self {
        let mut f = Self::text(name, label);
        f.required = true;
        f
    }

    /// Create an enum field with allowed values.
    #[must_use]
    pub fn enum_field(
        name: impl Into<String>,
        label: impl Into<String>,
        values: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            field_type: FieldType::Enum,
            default_value: String::new(),
            required: false,
            allowed_values: values,
            description: String::new(),
        }
    }

    /// Validate that a proposed value is acceptable for this field.
    ///
    /// Returns `Ok(())` on success or an error message on failure.
    pub fn validate(&self, value: &str) -> Result<(), String> {
        if self.required && value.trim().is_empty() {
            return Err(format!("Field '{}' is required", self.name));
        }
        if self.field_type == FieldType::Enum && !self.allowed_values.is_empty() {
            if !value.is_empty() && !self.allowed_values.iter().any(|v| v == value) {
                return Err(format!(
                    "Value '{}' not in allowed values for field '{}'",
                    value, self.name
                ));
            }
        }
        if self.field_type == FieldType::Integer && !value.is_empty() {
            value
                .parse::<i64>()
                .map_err(|_| format!("Field '{}' must be an integer", self.name))?;
        }
        if self.field_type == FieldType::Boolean && !value.is_empty() {
            match value {
                "true" | "false" | "1" | "0" | "yes" | "no" => {}
                _ => {
                    return Err(format!(
                        "Field '{}' must be a boolean (true/false)",
                        self.name
                    ))
                }
            }
        }
        Ok(())
    }
}

/// A named metadata template consisting of ordered field definitions.
#[derive(Clone, Debug)]
pub struct MetadataTemplate {
    /// Template name (unique within a library).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Ordered list of fields.
    pub fields: Vec<TemplateField>,
    /// Template version.
    pub version: u32,
}

impl MetadataTemplate {
    /// Create a new empty template.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            fields: Vec::new(),
            version: 1,
        }
    }

    /// Add a field to the template.
    pub fn add_field(&mut self, field: TemplateField) {
        self.fields.push(field);
    }

    /// Return the field with the given name, if present.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&TemplateField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Apply defaults from this template to a metadata map.
    ///
    /// Only inserts fields that are not already present in `map`.
    pub fn apply_defaults(&self, map: &mut HashMap<String, String>) {
        for field in &self.fields {
            if !field.default_value.is_empty() && !map.contains_key(&field.name) {
                map.insert(field.name.clone(), field.default_value.clone());
            }
        }
    }

    /// Validate a complete metadata map against this template.
    ///
    /// Returns a list of validation errors (empty = valid).
    #[must_use]
    pub fn validate(&self, map: &HashMap<String, String>) -> Vec<String> {
        let mut errors = Vec::new();
        for field in &self.fields {
            let value = map.get(&field.name).map(String::as_str).unwrap_or("");
            if let Err(e) = field.validate(value) {
                errors.push(e);
            }
        }
        errors
    }

    /// Return the count of required fields.
    #[must_use]
    pub fn required_field_count(&self) -> usize {
        self.fields.iter().filter(|f| f.required).count()
    }
}

/// A named collection of [`MetadataTemplate`] instances.
///
/// # Example
/// ```
/// use oximedia_mam::metadata_template::{TemplateField, MetadataTemplate, TemplateLibrary};
///
/// let mut lib = TemplateLibrary::new();
/// let mut t = MetadataTemplate::new("news_clip");
/// t.add_field(TemplateField::required_text("title", "Title"));
/// lib.add(t);
/// assert!(lib.get("news_clip").is_some());
/// ```
#[derive(Default)]
pub struct TemplateLibrary {
    templates: HashMap<String, MetadataTemplate>,
}

impl TemplateLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace a template.
    pub fn add(&mut self, template: MetadataTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Remove a template by name.  Returns `true` if it existed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.templates.remove(name).is_some()
    }

    /// Look up a template by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&MetadataTemplate> {
        self.templates.get(name)
    }

    /// Return the number of templates in the library.
    #[must_use]
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Return `true` when the library is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Iterate over all templates.
    pub fn iter(&self) -> impl Iterator<Item = &MetadataTemplate> {
        self.templates.values()
    }

    /// Return template names sorted alphabetically.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.templates.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn news_template() -> MetadataTemplate {
        let mut t = MetadataTemplate::new("news");
        t.description = "Standard news clip template".to_string();
        t.add_field(TemplateField::required_text("title", "Title"));
        t.add_field(TemplateField::text("reporter", "Reporter"));
        let mut status_field = TemplateField::enum_field(
            "status",
            "Status",
            vec![
                "draft".to_string(),
                "approved".to_string(),
                "archived".to_string(),
            ],
        );
        status_field.default_value = "draft".to_string();
        t.add_field(status_field);
        t
    }

    #[test]
    fn test_template_field_count() {
        let t = news_template();
        assert_eq!(t.fields.len(), 3);
    }

    #[test]
    fn test_required_field_count() {
        let t = news_template();
        assert_eq!(t.required_field_count(), 1);
    }

    #[test]
    fn test_field_lookup() {
        let t = news_template();
        assert!(t.field("title").is_some());
        assert!(t.field("nonexistent").is_none());
    }

    #[test]
    fn test_apply_defaults_fills_missing() {
        let t = news_template();
        let mut map = HashMap::new();
        map.insert("title".to_string(), "Breaking News".to_string());
        t.apply_defaults(&mut map);
        assert_eq!(map.get("status").map(String::as_str), Some("draft"));
    }

    #[test]
    fn test_apply_defaults_does_not_overwrite() {
        let t = news_template();
        let mut map = HashMap::new();
        map.insert("status".to_string(), "approved".to_string());
        t.apply_defaults(&mut map);
        assert_eq!(map.get("status").map(String::as_str), Some("approved"));
    }

    #[test]
    fn test_validate_required_field_missing() {
        let t = news_template();
        let map = HashMap::new(); // title missing
        let errors = t.validate(&map);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("title"));
    }

    #[test]
    fn test_validate_valid_map() {
        let t = news_template();
        let mut map = HashMap::new();
        map.insert("title".to_string(), "Story".to_string());
        map.insert("status".to_string(), "approved".to_string());
        assert!(t.validate(&map).is_empty());
    }

    #[test]
    fn test_validate_invalid_enum_value() {
        let t = news_template();
        let mut map = HashMap::new();
        map.insert("title".to_string(), "Story".to_string());
        map.insert("status".to_string(), "pending".to_string()); // not in allowed
        let errors = t.validate(&map);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_field_validate_integer() {
        let mut f = TemplateField::text("count", "Count");
        f.field_type = FieldType::Integer;
        assert!(f.validate("42").is_ok());
        assert!(f.validate("not_a_number").is_err());
    }

    #[test]
    fn test_field_validate_boolean() {
        let mut f = TemplateField::text("flag", "Flag");
        f.field_type = FieldType::Boolean;
        assert!(f.validate("true").is_ok());
        assert!(f.validate("false").is_ok());
        assert!(f.validate("maybe").is_err());
    }

    #[test]
    fn test_library_add_and_get() {
        let mut lib = TemplateLibrary::new();
        lib.add(news_template());
        assert!(lib.get("news").is_some());
        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn test_library_remove() {
        let mut lib = TemplateLibrary::new();
        lib.add(news_template());
        assert!(lib.remove("news"));
        assert!(lib.is_empty());
    }

    #[test]
    fn test_library_names_sorted() {
        let mut lib = TemplateLibrary::new();
        lib.add(MetadataTemplate::new("zebra"));
        lib.add(MetadataTemplate::new("alpha"));
        let names = lib.names();
        assert_eq!(names, vec!["alpha", "zebra"]);
    }

    #[test]
    fn test_library_remove_nonexistent() {
        let mut lib = TemplateLibrary::new();
        assert!(!lib.remove("ghost"));
    }

    #[test]
    fn test_template_version_default() {
        let t = MetadataTemplate::new("v_test");
        assert_eq!(t.version, 1);
    }

    #[test]
    fn test_field_type_enum_empty_value_passes() {
        let f =
            TemplateField::enum_field("cat", "Category", vec!["a".to_string(), "b".to_string()]);
        // Empty string is allowed (field is not required).
        assert!(f.validate("").is_ok());
    }
}
