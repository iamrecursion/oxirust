//! Lazy deserialization for large JSON request bodies.
//!
//! Provides on-demand field extraction from JSON payloads without
//! deserializing the entire body upfront. Useful for large media
//! metadata requests where only a subset of fields may be needed.

#![allow(dead_code)]

use std::collections::HashMap;

/// A lazily-deserialized JSON value.
///
/// Holds the raw JSON string and extracts fields on demand.
#[derive(Debug, Clone)]
pub struct LazyJson {
    /// Raw JSON string.
    raw: String,
    /// Pre-parsed top-level field offsets (field_name -> value_start..value_end).
    field_index: HashMap<String, (usize, usize)>,
    /// Whether the index has been built.
    indexed: bool,
}

impl LazyJson {
    /// Creates a new lazy JSON wrapper.
    pub fn new(raw: impl Into<String>) -> Self {
        Self {
            raw: raw.into(),
            field_index: HashMap::new(),
            indexed: false,
        }
    }

    /// Returns the raw JSON string.
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// Returns the byte length of the raw JSON.
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    /// Returns true if the raw JSON is empty.
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    /// Builds the top-level field index.
    pub fn build_index(&mut self) {
        if self.indexed {
            return;
        }

        let bytes = self.raw.as_bytes();
        let len = bytes.len();
        let mut pos = 0;

        // Skip to opening brace
        while pos < len && bytes[pos] != b'{' {
            pos += 1;
        }
        if pos >= len {
            self.indexed = true;
            return;
        }
        pos += 1; // skip '{'

        loop {
            // Skip whitespace
            pos = skip_whitespace(bytes, pos);
            if pos >= len || bytes[pos] == b'}' {
                break;
            }

            // Expect a key
            if bytes[pos] != b'"' {
                break;
            }

            let key_start = pos + 1;
            pos += 1;
            // Find end of key
            while pos < len && bytes[pos] != b'"' {
                if bytes[pos] == b'\\' {
                    pos += 1; // skip escaped char
                }
                pos += 1;
            }
            let key_end = pos;
            pos += 1; // skip closing quote

            let key = String::from_utf8_lossy(&bytes[key_start..key_end]).to_string();

            // Skip colon
            pos = skip_whitespace(bytes, pos);
            if pos >= len || bytes[pos] != b':' {
                break;
            }
            pos += 1;
            pos = skip_whitespace(bytes, pos);

            // Find value extent
            let value_start = pos;
            pos = skip_json_value(bytes, pos);
            let value_end = pos;

            self.field_index.insert(key, (value_start, value_end));

            // Skip comma
            pos = skip_whitespace(bytes, pos);
            if pos < len && bytes[pos] == b',' {
                pos += 1;
            }
        }

        self.indexed = true;
    }

    /// Returns the raw JSON value for a top-level field.
    pub fn get_raw_field(&mut self, field: &str) -> Option<&str> {
        self.build_index();
        self.field_index
            .get(field)
            .map(|&(start, end)| &self.raw[start..end])
    }

    /// Extracts a string field (removes surrounding quotes).
    pub fn get_string(&mut self, field: &str) -> Option<String> {
        let raw = self.get_raw_field(field)?;
        let trimmed = raw.trim();
        if trimmed.starts_with('"') && trimmed.ends_with('"') {
            Some(
                trimmed[1..trimmed.len() - 1]
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\")
                    .replace("\\n", "\n")
                    .replace("\\t", "\t"),
            )
        } else {
            None
        }
    }

    /// Extracts an integer field.
    pub fn get_i64(&mut self, field: &str) -> Option<i64> {
        let raw = self.get_raw_field(field)?;
        raw.trim().parse().ok()
    }

    /// Extracts a u64 field.
    pub fn get_u64(&mut self, field: &str) -> Option<u64> {
        let raw = self.get_raw_field(field)?;
        raw.trim().parse().ok()
    }

    /// Extracts a float field.
    pub fn get_f64(&mut self, field: &str) -> Option<f64> {
        let raw = self.get_raw_field(field)?;
        raw.trim().parse().ok()
    }

    /// Extracts a boolean field.
    pub fn get_bool(&mut self, field: &str) -> Option<bool> {
        let raw = self.get_raw_field(field)?;
        match raw.trim() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    }

    /// Checks if a field is null.
    pub fn is_null(&mut self, field: &str) -> bool {
        self.get_raw_field(field)
            .map(|v| v.trim() == "null")
            .unwrap_or(false)
    }

    /// Returns the list of top-level field names.
    pub fn field_names(&mut self) -> Vec<String> {
        self.build_index();
        self.field_index.keys().cloned().collect()
    }

    /// Returns the number of top-level fields.
    pub fn field_count(&mut self) -> usize {
        self.build_index();
        self.field_index.len()
    }

    /// Whether the JSON has a specific top-level field.
    pub fn has_field(&mut self, field: &str) -> bool {
        self.build_index();
        self.field_index.contains_key(field)
    }
}

/// Skips whitespace in a byte slice.
fn skip_whitespace(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len()
        && (bytes[pos] == b' ' || bytes[pos] == b'\n' || bytes[pos] == b'\r' || bytes[pos] == b'\t')
    {
        pos += 1;
    }
    pos
}

/// Skips a complete JSON value (string, number, object, array, true, false, null).
fn skip_json_value(bytes: &[u8], mut pos: usize) -> usize {
    if pos >= bytes.len() {
        return pos;
    }

    match bytes[pos] {
        b'"' => {
            // String
            pos += 1;
            while pos < bytes.len() {
                if bytes[pos] == b'\\' {
                    pos += 2; // skip escape sequence
                } else if bytes[pos] == b'"' {
                    pos += 1;
                    return pos;
                } else {
                    pos += 1;
                }
            }
            pos
        }
        b'{' => {
            // Object
            let mut depth = 1;
            pos += 1;
            while pos < bytes.len() && depth > 0 {
                match bytes[pos] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    b'"' => {
                        pos += 1;
                        while pos < bytes.len() && bytes[pos] != b'"' {
                            if bytes[pos] == b'\\' {
                                pos += 1;
                            }
                            pos += 1;
                        }
                    }
                    _ => {}
                }
                pos += 1;
            }
            pos
        }
        b'[' => {
            // Array
            let mut depth = 1;
            pos += 1;
            while pos < bytes.len() && depth > 0 {
                match bytes[pos] {
                    b'[' => depth += 1,
                    b']' => depth -= 1,
                    b'"' => {
                        pos += 1;
                        while pos < bytes.len() && bytes[pos] != b'"' {
                            if bytes[pos] == b'\\' {
                                pos += 1;
                            }
                            pos += 1;
                        }
                    }
                    _ => {}
                }
                pos += 1;
            }
            pos
        }
        _ => {
            // Number, true, false, null
            while pos < bytes.len()
                && bytes[pos] != b','
                && bytes[pos] != b'}'
                && bytes[pos] != b']'
                && bytes[pos] != b' '
                && bytes[pos] != b'\n'
                && bytes[pos] != b'\r'
                && bytes[pos] != b'\t'
            {
                pos += 1;
            }
            pos
        }
    }
}

/// Validates that a JSON string has matching braces/brackets.
pub fn validate_json_structure(json: &str) -> bool {
    let bytes = json.as_bytes();
    let mut stack: Vec<u8> = Vec::new();
    let mut in_string = false;
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if b == b'\\' {
                i += 1; // skip escaped char
            } else if b == b'"' {
                in_string = false;
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' => stack.push(b'}'),
                b'[' => stack.push(b']'),
                b'}' | b']' => {
                    if stack.pop() != Some(b) {
                        return false;
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }

    stack.is_empty() && !in_string
}

/// Estimates memory savings from lazy deserialization.
pub struct LazySavingsEstimate {
    /// Total JSON size.
    pub total_size: usize,
    /// Number of fields accessed.
    pub fields_accessed: usize,
    /// Total fields available.
    pub total_fields: usize,
    /// Bytes actually read.
    pub bytes_read: usize,
}

impl LazySavingsEstimate {
    /// Savings ratio (0.0 to 1.0).
    pub fn savings_ratio(&self) -> f64 {
        if self.total_size == 0 {
            return 0.0;
        }
        1.0 - (self.bytes_read as f64 / self.total_size as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_JSON: &str = r#"{"name": "test media", "duration": 120.5, "codec": "av1", "width": 1920, "height": 1080, "active": true, "tags": null}"#;

    // LazyJson

    #[test]
    fn test_lazy_json_creation() {
        let lj = LazyJson::new(SAMPLE_JSON);
        assert!(!lj.is_empty());
        assert_eq!(lj.raw(), SAMPLE_JSON);
    }

    #[test]
    fn test_get_string() {
        let mut lj = LazyJson::new(SAMPLE_JSON);
        assert_eq!(lj.get_string("name"), Some("test media".to_string()));
        assert_eq!(lj.get_string("codec"), Some("av1".to_string()));
    }

    #[test]
    fn test_get_i64() {
        let mut lj = LazyJson::new(SAMPLE_JSON);
        assert_eq!(lj.get_i64("width"), Some(1920));
        assert_eq!(lj.get_i64("height"), Some(1080));
    }

    #[test]
    fn test_get_u64() {
        let mut lj = LazyJson::new(SAMPLE_JSON);
        assert_eq!(lj.get_u64("width"), Some(1920));
    }

    #[test]
    fn test_get_f64() {
        let mut lj = LazyJson::new(SAMPLE_JSON);
        let d = lj.get_f64("duration");
        assert!(d.is_some());
        assert!((d.expect("should succeed") - 120.5).abs() < 1e-9);
    }

    #[test]
    fn test_get_bool() {
        let mut lj = LazyJson::new(SAMPLE_JSON);
        assert_eq!(lj.get_bool("active"), Some(true));
    }

    #[test]
    fn test_is_null() {
        let mut lj = LazyJson::new(SAMPLE_JSON);
        assert!(lj.is_null("tags"));
        assert!(!lj.is_null("name"));
    }

    #[test]
    fn test_missing_field() {
        let mut lj = LazyJson::new(SAMPLE_JSON);
        assert!(lj.get_string("nonexistent").is_none());
        assert!(lj.get_i64("nonexistent").is_none());
    }

    #[test]
    fn test_field_names() {
        let mut lj = LazyJson::new(SAMPLE_JSON);
        let names = lj.field_names();
        assert!(names.contains(&"name".to_string()));
        assert!(names.contains(&"duration".to_string()));
        assert!(names.contains(&"codec".to_string()));
    }

    #[test]
    fn test_field_count() {
        let mut lj = LazyJson::new(SAMPLE_JSON);
        assert_eq!(lj.field_count(), 7);
    }

    #[test]
    fn test_has_field() {
        let mut lj = LazyJson::new(SAMPLE_JSON);
        assert!(lj.has_field("name"));
        assert!(!lj.has_field("missing"));
    }

    #[test]
    fn test_nested_object() {
        let json = r#"{"meta": {"title": "test"}, "id": 42}"#;
        let mut lj = LazyJson::new(json);
        let raw = lj.get_raw_field("meta");
        assert!(raw.is_some());
        assert!(raw.expect("should succeed").contains("title"));
        assert_eq!(lj.get_i64("id"), Some(42));
    }

    #[test]
    fn test_array_value() {
        let json = r#"{"items": [1, 2, 3], "count": 3}"#;
        let mut lj = LazyJson::new(json);
        let raw = lj.get_raw_field("items");
        assert!(raw.is_some());
        assert!(raw.expect("should succeed").contains("[1, 2, 3]"));
    }

    #[test]
    fn test_escaped_string() {
        let json = r#"{"msg": "hello \"world\"", "n": 1}"#;
        let mut lj = LazyJson::new(json);
        let s = lj.get_string("msg");
        assert_eq!(s, Some("hello \"world\"".to_string()));
    }

    #[test]
    fn test_empty_json() {
        let mut lj = LazyJson::new("{}");
        assert_eq!(lj.field_count(), 0);
    }

    #[test]
    fn test_empty_string() {
        let lj = LazyJson::new("");
        assert!(lj.is_empty());
    }

    // validate_json_structure

    #[test]
    fn test_validate_valid_json() {
        assert!(validate_json_structure(SAMPLE_JSON));
        assert!(validate_json_structure("{}"));
        assert!(validate_json_structure("[]"));
        assert!(validate_json_structure(r#"{"a": [1, {"b": 2}]}"#));
    }

    #[test]
    fn test_validate_invalid_json() {
        assert!(!validate_json_structure("{"));
        assert!(!validate_json_structure("[}"));
        assert!(!validate_json_structure(r#"{"unclosed": "string}"#));
    }

    #[test]
    fn test_validate_string_with_braces() {
        assert!(validate_json_structure(r#"{"a": "value with { and }"}"#));
    }

    // LazySavingsEstimate

    #[test]
    fn test_savings_estimate() {
        let est = LazySavingsEstimate {
            total_size: 10000,
            fields_accessed: 2,
            total_fields: 50,
            bytes_read: 200,
        };
        assert!((est.savings_ratio() - 0.98).abs() < 1e-9);
    }

    #[test]
    fn test_savings_estimate_zero() {
        let est = LazySavingsEstimate {
            total_size: 0,
            fields_accessed: 0,
            total_fields: 0,
            bytes_read: 0,
        };
        assert!((est.savings_ratio()).abs() < 1e-9);
    }

    // Large JSON

    #[test]
    fn test_large_json_partial_access() {
        let mut fields = Vec::new();
        for i in 0..100 {
            fields.push(format!("\"field_{}\": \"value_{}\"", i, i));
        }
        let json = format!("{{{}}}", fields.join(", "));

        let mut lj = LazyJson::new(&json);
        // Only access 2 fields
        assert_eq!(lj.get_string("field_0"), Some("value_0".to_string()));
        assert_eq!(lj.get_string("field_99"), Some("value_99".to_string()));
        assert_eq!(lj.field_count(), 100);
    }

    #[test]
    fn test_whitespace_handling() {
        let json = r#"{
            "name"  :  "spaced out" ,
            "value" :  42
        }"#;
        let mut lj = LazyJson::new(json);
        assert_eq!(lj.get_string("name"), Some("spaced out".to_string()));
        assert_eq!(lj.get_i64("value"), Some(42));
    }

    #[test]
    fn test_bool_false() {
        let json = r#"{"active": false}"#;
        let mut lj = LazyJson::new(json);
        assert_eq!(lj.get_bool("active"), Some(false));
    }

    #[test]
    fn test_negative_number() {
        let json = r#"{"offset": -42}"#;
        let mut lj = LazyJson::new(json);
        assert_eq!(lj.get_i64("offset"), Some(-42));
    }
}
