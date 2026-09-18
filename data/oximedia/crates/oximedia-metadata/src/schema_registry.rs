//! Metadata schema registry: schema versioning, field definitions, and type constraints.

#![allow(dead_code)]

use std::collections::HashMap;

/// Data type for a metadata field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    /// A UTF-8 text string.
    Text,
    /// An integer value.
    Integer,
    /// A floating-point value.
    Float,
    /// A boolean value.
    Boolean,
    /// A binary blob.
    Binary,
    /// A date/time in ISO 8601 format.
    DateTime,
    /// A list of text strings.
    TextList,
}

/// Cardinality constraint for a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cardinality {
    /// Exactly one value required.
    Required,
    /// Zero or one values.
    Optional,
    /// One or more values.
    OneOrMore,
    /// Zero or more values.
    Many,
}

/// Definition of a single metadata field.
#[derive(Debug, Clone)]
pub struct FieldDefinition {
    /// Field identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Data type.
    pub field_type: FieldType,
    /// Cardinality constraint.
    pub cardinality: Cardinality,
    /// Optional description.
    pub description: Option<String>,
    /// Optional set of allowed values (for enumerated fields).
    pub allowed_values: Option<Vec<String>>,
    /// Optional maximum text length.
    pub max_length: Option<usize>,
}

impl FieldDefinition {
    /// Create a new required text field.
    #[must_use]
    pub fn required_text(id: &str, label: &str) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            field_type: FieldType::Text,
            cardinality: Cardinality::Required,
            description: None,
            allowed_values: None,
            max_length: None,
        }
    }

    /// Create a new optional text field.
    #[must_use]
    pub fn optional_text(id: &str, label: &str) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            field_type: FieldType::Text,
            cardinality: Cardinality::Optional,
            description: None,
            allowed_values: None,
            max_length: None,
        }
    }

    /// Create a new optional integer field.
    #[must_use]
    pub fn optional_integer(id: &str, label: &str) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            field_type: FieldType::Integer,
            cardinality: Cardinality::Optional,
            description: None,
            allowed_values: None,
            max_length: None,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set allowed values for enumerated fields.
    #[must_use]
    pub fn with_allowed_values(mut self, values: Vec<String>) -> Self {
        self.allowed_values = Some(values);
        self
    }

    /// Set maximum text length.
    #[must_use]
    pub fn with_max_length(mut self, len: usize) -> Self {
        self.max_length = Some(len);
        self
    }

    /// Validate a string value against this field definition.
    #[must_use]
    pub fn validate_value(&self, value: &str) -> bool {
        if let Some(max) = self.max_length {
            if value.len() > max {
                return false;
            }
        }
        if let Some(ref allowed) = self.allowed_values {
            return allowed.iter().any(|v| v == value);
        }
        true
    }
}

/// Version of a metadata schema.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SchemaVersion {
    /// Major version number.
    pub major: u32,
    /// Minor version number.
    pub minor: u32,
    /// Patch version number.
    pub patch: u32,
}

impl SchemaVersion {
    /// Create a new schema version.
    #[must_use]
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is backward-compatible with `other`.
    /// A version is compatible when its major number matches and its minor is >= other's minor.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}

impl std::fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// A metadata schema: a versioned collection of field definitions.
#[derive(Debug, Clone)]
pub struct MetadataSchema {
    /// Unique schema identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Schema version.
    pub version: SchemaVersion,
    /// Map from field ID to its definition.
    fields: HashMap<String, FieldDefinition>,
}

impl MetadataSchema {
    /// Create a new empty schema.
    #[must_use]
    pub fn new(id: &str, name: &str, version: SchemaVersion) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            version,
            fields: HashMap::new(),
        }
    }

    /// Add a field definition.
    pub fn add_field(&mut self, def: FieldDefinition) {
        self.fields.insert(def.id.clone(), def);
    }

    /// Get a field definition by ID.
    #[must_use]
    pub fn get_field(&self, id: &str) -> Option<&FieldDefinition> {
        self.fields.get(id)
    }

    /// Check whether a field is defined in this schema.
    #[must_use]
    pub fn has_field(&self, id: &str) -> bool {
        self.fields.contains_key(id)
    }

    /// Return all required field IDs.
    #[must_use]
    pub fn required_fields(&self) -> Vec<&str> {
        self.fields
            .values()
            .filter(|f| f.cardinality == Cardinality::Required)
            .map(|f| f.id.as_str())
            .collect()
    }

    /// Validate a set of key/value pairs against the schema.
    /// Returns a list of validation errors (empty = valid).
    #[must_use]
    pub fn validate(&self, data: &HashMap<String, String>) -> Vec<String> {
        let mut errors = Vec::new();

        // Check required fields are present
        for field in self.fields.values() {
            if field.cardinality == Cardinality::Required && !data.contains_key(&field.id) {
                errors.push(format!("Required field '{}' is missing", field.id));
            }
        }

        // Validate provided values
        for (key, value) in data {
            if let Some(field) = self.fields.get(key) {
                if !field.validate_value(value) {
                    errors.push(format!("Field '{}' has invalid value: '{}'", key, value));
                }
            }
        }

        errors
    }

    /// Return the number of fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

/// Registry that manages multiple named schemas with versioning.
#[derive(Debug, Default)]
pub struct SchemaRegistry {
    /// Map from schema ID to ordered list of versions (oldest first).
    schemas: HashMap<String, Vec<MetadataSchema>>,
}

impl SchemaRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }

    /// Register a schema. Schemas with the same ID are stored as separate versions.
    pub fn register(&mut self, schema: MetadataSchema) {
        self.schemas
            .entry(schema.id.clone())
            .or_default()
            .push(schema);
    }

    /// Retrieve the latest version of a schema by ID.
    #[must_use]
    pub fn get_latest(&self, id: &str) -> Option<&MetadataSchema> {
        self.schemas.get(id)?.last()
    }

    /// Retrieve a specific version of a schema.
    #[must_use]
    pub fn get_version(&self, id: &str, version: &SchemaVersion) -> Option<&MetadataSchema> {
        self.schemas.get(id)?.iter().find(|s| &s.version == version)
    }

    /// List all registered schema IDs.
    #[must_use]
    pub fn schema_ids(&self) -> Vec<&str> {
        self.schemas.keys().map(String::as_str).collect()
    }

    /// Return the number of registered schemas (unique IDs).
    #[must_use]
    pub fn count(&self) -> usize {
        self.schemas.len()
    }

    /// Return the total number of schema versions registered.
    #[must_use]
    pub fn total_versions(&self) -> usize {
        self.schemas.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_version(major: u32, minor: u32, patch: u32) -> SchemaVersion {
        SchemaVersion::new(major, minor, patch)
    }

    fn make_schema(id: &str, major: u32, minor: u32) -> MetadataSchema {
        let mut schema = MetadataSchema::new(id, id, make_version(major, minor, 0));
        schema.add_field(FieldDefinition::required_text("title", "Title"));
        schema.add_field(FieldDefinition::optional_text("artist", "Artist"));
        schema
    }

    #[test]
    fn test_schema_version_display() {
        let v = SchemaVersion::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_schema_version_compatibility_same_major() {
        let v1 = make_version(1, 0, 0);
        let v2 = make_version(1, 1, 0);
        assert!(v2.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v2));
    }

    #[test]
    fn test_schema_version_compatibility_different_major() {
        let v1 = make_version(1, 5, 0);
        let v2 = make_version(2, 0, 0);
        assert!(!v1.is_compatible_with(&v2));
        assert!(!v2.is_compatible_with(&v1));
    }

    #[test]
    fn test_field_definition_required_text() {
        let f = FieldDefinition::required_text("title", "Title");
        assert_eq!(f.id, "title");
        assert_eq!(f.field_type, FieldType::Text);
        assert_eq!(f.cardinality, Cardinality::Required);
    }

    #[test]
    fn test_field_definition_validate_max_length() {
        let f = FieldDefinition::optional_text("tag", "Tag").with_max_length(5);
        assert!(f.validate_value("abc"));
        assert!(!f.validate_value("toolongstring"));
    }

    #[test]
    fn test_field_definition_validate_allowed_values() {
        let f = FieldDefinition::optional_text("status", "Status")
            .with_allowed_values(vec!["active".to_string(), "archived".to_string()]);
        assert!(f.validate_value("active"));
        assert!(!f.validate_value("deleted"));
    }

    #[test]
    fn test_schema_has_field() {
        let schema = make_schema("audio", 1, 0);
        assert!(schema.has_field("title"));
        assert!(schema.has_field("artist"));
        assert!(!schema.has_field("missing"));
    }

    #[test]
    fn test_schema_required_fields() {
        let schema = make_schema("audio", 1, 0);
        let required = schema.required_fields();
        assert!(required.contains(&"title"));
        assert!(!required.contains(&"artist"));
    }

    #[test]
    fn test_schema_validate_missing_required() {
        let schema = make_schema("audio", 1, 0);
        let data = HashMap::new();
        let errors = schema.validate(&data);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("title")));
    }

    #[test]
    fn test_schema_validate_ok() {
        let schema = make_schema("audio", 1, 0);
        let mut data = HashMap::new();
        data.insert("title".to_string(), "My Song".to_string());
        let errors = schema.validate(&data);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_schema_field_count() {
        let schema = make_schema("audio", 1, 0);
        assert_eq!(schema.field_count(), 2);
    }

    #[test]
    fn test_registry_register_and_get_latest() {
        let mut registry = SchemaRegistry::new();
        registry.register(make_schema("music", 1, 0));
        registry.register(make_schema("music", 1, 1));
        let latest = registry
            .get_latest("music")
            .expect("should succeed in test");
        assert_eq!(latest.version, make_version(1, 1, 0));
    }

    #[test]
    fn test_registry_get_specific_version() {
        let mut registry = SchemaRegistry::new();
        registry.register(make_schema("music", 1, 0));
        registry.register(make_schema("music", 2, 0));
        let v1 = registry
            .get_version("music", &make_version(1, 0, 0))
            .expect("should succeed in test");
        assert_eq!(v1.version.major, 1);
    }

    #[test]
    fn test_registry_count() {
        let mut registry = SchemaRegistry::new();
        registry.register(make_schema("schema_a", 1, 0));
        registry.register(make_schema("schema_b", 1, 0));
        assert_eq!(registry.count(), 2);
    }

    #[test]
    fn test_registry_total_versions() {
        let mut registry = SchemaRegistry::new();
        registry.register(make_schema("x", 1, 0));
        registry.register(make_schema("x", 1, 1));
        registry.register(make_schema("y", 1, 0));
        assert_eq!(registry.total_versions(), 3);
    }

    #[test]
    fn test_registry_schema_ids() {
        let mut registry = SchemaRegistry::new();
        registry.register(make_schema("alpha", 1, 0));
        registry.register(make_schema("beta", 1, 0));
        let ids = registry.schema_ids();
        assert!(ids.contains(&"alpha"));
        assert!(ids.contains(&"beta"));
    }

    #[test]
    fn test_field_definition_with_description() {
        let f = FieldDefinition::required_text("title", "Title")
            .with_description("The title of the media");
        assert!(f.description.is_some());
    }
}
