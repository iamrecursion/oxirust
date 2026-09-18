#![allow(dead_code)]
//! Metadata template system for common workflows.
//!
//! Provides reusable metadata presets (e.g., broadcast, podcast, music release)
//! that can be applied to media files for consistent tagging.

use std::collections::HashMap;
use std::fmt;

/// A named, reusable metadata template.
#[derive(Debug, Clone)]
pub struct MetadataTemplate {
    /// Unique name of this template.
    name: String,
    /// Human-readable description.
    description: String,
    /// Category this template belongs to.
    category: TemplateCategory,
    /// Field definitions in the template.
    fields: Vec<TemplateField>,
    /// Whether fields not listed in the template should be stripped.
    strip_unlisted: bool,
}

/// Category of a metadata template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplateCategory {
    /// Broadcast / TV delivery.
    Broadcast,
    /// Podcast production.
    Podcast,
    /// Music release (album, single).
    Music,
    /// Film / cinema delivery.
    Film,
    /// Social media export.
    Social,
    /// Archival / preservation.
    Archive,
    /// Custom user-defined category.
    Custom,
}

impl fmt::Display for TemplateCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Broadcast => write!(f, "Broadcast"),
            Self::Podcast => write!(f, "Podcast"),
            Self::Music => write!(f, "Music"),
            Self::Film => write!(f, "Film"),
            Self::Social => write!(f, "Social"),
            Self::Archive => write!(f, "Archive"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// A single field definition inside a template.
#[derive(Debug, Clone)]
pub struct TemplateField {
    /// Key / tag name.
    pub key: String,
    /// Whether this field is required.
    pub required: bool,
    /// Default value (applied when no user value is provided).
    pub default_value: Option<String>,
    /// Validation rule for the field value.
    pub validation: FieldValidation,
}

/// Validation rules for template fields.
#[derive(Debug, Clone)]
pub enum FieldValidation {
    /// No validation.
    None,
    /// Value must not be empty.
    NonEmpty,
    /// Value must match a regex-like pattern (simplified glob).
    Pattern(String),
    /// Value must be one of the given options.
    OneOf(Vec<String>),
    /// Value length must be within a range (min, max inclusive).
    LengthRange(usize, usize),
}

impl MetadataTemplate {
    /// Create a new empty template.
    pub fn new(name: impl Into<String>, category: TemplateCategory) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            category,
            fields: Vec::new(),
            strip_unlisted: false,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Enable stripping of unlisted fields.
    pub fn with_strip_unlisted(mut self, strip: bool) -> Self {
        self.strip_unlisted = strip;
        self
    }

    /// Add a field definition.
    pub fn add_field(&mut self, field: TemplateField) {
        self.fields.push(field);
    }

    /// Get the template name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the template description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the template category.
    pub fn category(&self) -> TemplateCategory {
        self.category
    }

    /// Get all field definitions.
    pub fn fields(&self) -> &[TemplateField] {
        &self.fields
    }

    /// Whether unlisted fields should be stripped.
    pub fn strip_unlisted(&self) -> bool {
        self.strip_unlisted
    }

    /// Validate a set of key-value pairs against this template.
    ///
    /// Returns a list of validation error messages (empty if all OK).
    pub fn validate(&self, values: &HashMap<String, String>) -> Vec<String> {
        let mut errors = Vec::new();
        for field in &self.fields {
            match values.get(&field.key) {
                Some(val) => {
                    if let Some(err) = validate_field_value(&field.key, val, &field.validation) {
                        errors.push(err);
                    }
                }
                None => {
                    if field.required && field.default_value.is_none() {
                        errors.push(format!("Missing required field: {}", field.key));
                    }
                }
            }
        }
        errors
    }

    /// Apply template defaults to a mutable map, filling in missing values.
    pub fn apply_defaults(&self, values: &mut HashMap<String, String>) {
        for field in &self.fields {
            if !values.contains_key(&field.key) {
                if let Some(ref default) = field.default_value {
                    values.insert(field.key.clone(), default.clone());
                }
            }
        }
    }

    /// Return the set of required field keys.
    pub fn required_keys(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| f.required)
            .map(|f| f.key.as_str())
            .collect()
    }

    /// Return the number of fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

/// A registry of named templates.
#[derive(Debug, Clone, Default)]
pub struct TemplateRegistry {
    /// Templates indexed by name.
    templates: HashMap<String, MetadataTemplate>,
}

impl TemplateRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template. Overwrites any existing template with the same name.
    pub fn register(&mut self, template: MetadataTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Look up a template by name.
    pub fn get(&self, name: &str) -> Option<&MetadataTemplate> {
        self.templates.get(name)
    }

    /// Remove a template by name.
    pub fn remove(&mut self, name: &str) -> Option<MetadataTemplate> {
        self.templates.remove(name)
    }

    /// List all template names.
    pub fn names(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }

    /// Filter templates by category.
    pub fn by_category(&self, category: TemplateCategory) -> Vec<&MetadataTemplate> {
        self.templates
            .values()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Number of registered templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

/// Validate a single field value against a rule. Returns `Some(error)` on failure.
fn validate_field_value(key: &str, value: &str, rule: &FieldValidation) -> Option<String> {
    match rule {
        FieldValidation::None => None,
        FieldValidation::NonEmpty => {
            if value.is_empty() {
                Some(format!("Field '{}' must not be empty", key))
            } else {
                None
            }
        }
        FieldValidation::Pattern(pat) => {
            // Simplified: check if value contains the pattern substring
            if value.contains(pat.as_str()) {
                None
            } else {
                Some(format!("Field '{}' does not match pattern '{}'", key, pat))
            }
        }
        FieldValidation::OneOf(options) => {
            if options.iter().any(|o| o == value) {
                None
            } else {
                Some(format!(
                    "Field '{}' must be one of: {}",
                    key,
                    options.join(", ")
                ))
            }
        }
        FieldValidation::LengthRange(min, max) => {
            let len = value.len();
            if len < *min || len > *max {
                Some(format!(
                    "Field '{}' length {} not in range [{}, {}]",
                    key, len, min, max
                ))
            } else {
                None
            }
        }
    }
}

/// Build a standard broadcast metadata template.
pub fn broadcast_template() -> MetadataTemplate {
    let mut t = MetadataTemplate::new("broadcast", TemplateCategory::Broadcast)
        .with_description("Standard broadcast delivery metadata");
    t.add_field(TemplateField {
        key: "title".into(),
        required: true,
        default_value: None,
        validation: FieldValidation::NonEmpty,
    });
    t.add_field(TemplateField {
        key: "program_id".into(),
        required: true,
        default_value: None,
        validation: FieldValidation::NonEmpty,
    });
    t.add_field(TemplateField {
        key: "broadcast_date".into(),
        required: false,
        default_value: None,
        validation: FieldValidation::None,
    });
    t.add_field(TemplateField {
        key: "rating".into(),
        required: false,
        default_value: Some("TV-G".into()),
        validation: FieldValidation::OneOf(vec![
            "TV-G".into(),
            "TV-PG".into(),
            "TV-14".into(),
            "TV-MA".into(),
        ]),
    });
    t
}

/// Build a standard podcast metadata template.
pub fn podcast_template() -> MetadataTemplate {
    let mut t = MetadataTemplate::new("podcast", TemplateCategory::Podcast)
        .with_description("Podcast episode metadata");
    t.add_field(TemplateField {
        key: "title".into(),
        required: true,
        default_value: None,
        validation: FieldValidation::NonEmpty,
    });
    t.add_field(TemplateField {
        key: "author".into(),
        required: true,
        default_value: None,
        validation: FieldValidation::NonEmpty,
    });
    t.add_field(TemplateField {
        key: "episode_number".into(),
        required: false,
        default_value: None,
        validation: FieldValidation::None,
    });
    t.add_field(TemplateField {
        key: "explicit".into(),
        required: false,
        default_value: Some("false".into()),
        validation: FieldValidation::OneOf(vec!["true".into(), "false".into()]),
    });
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_creation() {
        let t = MetadataTemplate::new("test", TemplateCategory::Music);
        assert_eq!(t.name(), "test");
        assert_eq!(t.category(), TemplateCategory::Music);
        assert_eq!(t.field_count(), 0);
    }

    #[test]
    fn test_template_with_description() {
        let t =
            MetadataTemplate::new("t", TemplateCategory::Film).with_description("A film template");
        assert_eq!(t.description(), "A film template");
    }

    #[test]
    fn test_add_field() {
        let mut t = MetadataTemplate::new("t", TemplateCategory::Custom);
        t.add_field(TemplateField {
            key: "title".into(),
            required: true,
            default_value: None,
            validation: FieldValidation::NonEmpty,
        });
        assert_eq!(t.field_count(), 1);
        assert_eq!(t.fields()[0].key, "title");
    }

    #[test]
    fn test_validate_required_missing() {
        let mut t = MetadataTemplate::new("t", TemplateCategory::Custom);
        t.add_field(TemplateField {
            key: "title".into(),
            required: true,
            default_value: None,
            validation: FieldValidation::NonEmpty,
        });
        let vals = HashMap::new();
        let errors = t.validate(&vals);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("Missing required field"));
    }

    #[test]
    fn test_validate_non_empty_fail() {
        let mut t = MetadataTemplate::new("t", TemplateCategory::Custom);
        t.add_field(TemplateField {
            key: "title".into(),
            required: true,
            default_value: None,
            validation: FieldValidation::NonEmpty,
        });
        let mut vals = HashMap::new();
        vals.insert("title".into(), String::new());
        let errors = t.validate(&vals);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("must not be empty"));
    }

    #[test]
    fn test_validate_one_of_pass() {
        let mut t = MetadataTemplate::new("t", TemplateCategory::Custom);
        t.add_field(TemplateField {
            key: "rating".into(),
            required: false,
            default_value: None,
            validation: FieldValidation::OneOf(vec!["G".into(), "PG".into()]),
        });
        let mut vals = HashMap::new();
        vals.insert("rating".into(), "G".into());
        let errors = t.validate(&vals);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_one_of_fail() {
        let mut t = MetadataTemplate::new("t", TemplateCategory::Custom);
        t.add_field(TemplateField {
            key: "rating".into(),
            required: false,
            default_value: None,
            validation: FieldValidation::OneOf(vec!["G".into(), "PG".into()]),
        });
        let mut vals = HashMap::new();
        vals.insert("rating".into(), "R".into());
        let errors = t.validate(&vals);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_validate_length_range() {
        let mut t = MetadataTemplate::new("t", TemplateCategory::Custom);
        t.add_field(TemplateField {
            key: "code".into(),
            required: false,
            default_value: None,
            validation: FieldValidation::LengthRange(2, 5),
        });
        let mut vals = HashMap::new();
        vals.insert("code".into(), "AB".into());
        assert!(t.validate(&vals).is_empty());

        vals.insert("code".into(), "A".into());
        assert_eq!(t.validate(&vals).len(), 1);

        vals.insert("code".into(), "ABCDEF".into());
        assert_eq!(t.validate(&vals).len(), 1);
    }

    #[test]
    fn test_apply_defaults() {
        let mut t = MetadataTemplate::new("t", TemplateCategory::Custom);
        t.add_field(TemplateField {
            key: "explicit".into(),
            required: false,
            default_value: Some("false".into()),
            validation: FieldValidation::None,
        });
        t.add_field(TemplateField {
            key: "title".into(),
            required: true,
            default_value: None,
            validation: FieldValidation::NonEmpty,
        });
        let mut vals = HashMap::new();
        vals.insert("title".into(), "My Song".into());
        t.apply_defaults(&mut vals);
        assert_eq!(
            vals.get("explicit").expect("should succeed in test"),
            "false"
        );
        // title should remain as set by user
        assert_eq!(
            vals.get("title").expect("should succeed in test"),
            "My Song"
        );
    }

    #[test]
    fn test_required_keys() {
        let mut t = MetadataTemplate::new("t", TemplateCategory::Custom);
        t.add_field(TemplateField {
            key: "a".into(),
            required: true,
            default_value: None,
            validation: FieldValidation::None,
        });
        t.add_field(TemplateField {
            key: "b".into(),
            required: false,
            default_value: None,
            validation: FieldValidation::None,
        });
        let keys = t.required_keys();
        assert_eq!(keys, vec!["a"]);
    }

    #[test]
    fn test_registry_basic() {
        let mut reg = TemplateRegistry::new();
        assert!(reg.is_empty());
        reg.register(MetadataTemplate::new("one", TemplateCategory::Music));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("one").is_some());
        assert!(reg.get("two").is_none());
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = TemplateRegistry::new();
        reg.register(MetadataTemplate::new("rm", TemplateCategory::Film));
        assert!(reg.remove("rm").is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_by_category() {
        let mut reg = TemplateRegistry::new();
        reg.register(MetadataTemplate::new("a", TemplateCategory::Music));
        reg.register(MetadataTemplate::new("b", TemplateCategory::Film));
        reg.register(MetadataTemplate::new("c", TemplateCategory::Music));
        let music = reg.by_category(TemplateCategory::Music);
        assert_eq!(music.len(), 2);
    }

    #[test]
    fn test_broadcast_template_preset() {
        let t = broadcast_template();
        assert_eq!(t.name(), "broadcast");
        assert_eq!(t.category(), TemplateCategory::Broadcast);
        assert!(t.field_count() >= 3);
    }

    #[test]
    fn test_podcast_template_preset() {
        let t = podcast_template();
        assert_eq!(t.name(), "podcast");
        assert_eq!(t.category(), TemplateCategory::Podcast);
        let req = t.required_keys();
        assert!(req.contains(&"title"));
        assert!(req.contains(&"author"));
    }

    #[test]
    fn test_template_category_display() {
        assert_eq!(TemplateCategory::Broadcast.to_string(), "Broadcast");
        assert_eq!(TemplateCategory::Archive.to_string(), "Archive");
        assert_eq!(TemplateCategory::Custom.to_string(), "Custom");
    }

    #[test]
    fn test_strip_unlisted_flag() {
        let t = MetadataTemplate::new("t", TemplateCategory::Custom).with_strip_unlisted(true);
        assert!(t.strip_unlisted());
    }

    #[test]
    fn test_validate_pattern_pass() {
        let mut t = MetadataTemplate::new("t", TemplateCategory::Custom);
        t.add_field(TemplateField {
            key: "id".into(),
            required: false,
            default_value: None,
            validation: FieldValidation::Pattern("ID-".into()),
        });
        let mut vals = HashMap::new();
        vals.insert("id".into(), "ID-12345".into());
        assert!(t.validate(&vals).is_empty());
    }

    #[test]
    fn test_validate_pattern_fail() {
        let mut t = MetadataTemplate::new("t", TemplateCategory::Custom);
        t.add_field(TemplateField {
            key: "id".into(),
            required: false,
            default_value: None,
            validation: FieldValidation::Pattern("ID-".into()),
        });
        let mut vals = HashMap::new();
        vals.insert("id".into(), "12345".into());
        assert_eq!(t.validate(&vals).len(), 1);
    }
}
