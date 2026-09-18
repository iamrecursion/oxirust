//! Metadata repair functionality.
//!
//! This module provides functions to repair corrupt metadata in media files.

use crate::Result;
use std::collections::HashMap;

/// Metadata field.
#[derive(Debug, Clone)]
pub struct MetadataField {
    /// Field name.
    pub name: String,
    /// Field value.
    pub value: String,
    /// Is this field corrupt.
    pub corrupt: bool,
}

/// Repair corrupt metadata fields.
pub fn repair_metadata(fields: &mut [MetadataField]) -> Result<usize> {
    let mut repaired = 0;

    for field in fields.iter_mut() {
        if field.corrupt {
            if let Some(fixed_value) = attempt_repair(&field.name, &field.value) {
                field.value = fixed_value;
                field.corrupt = false;
                repaired += 1;
            }
        }
    }

    Ok(repaired)
}

/// Attempt to repair a metadata field.
fn attempt_repair(name: &str, value: &str) -> Option<String> {
    match name {
        "duration" => repair_duration(value),
        "date" => repair_date(value),
        "title" => repair_title(value),
        _ => Some(sanitize_string(value)),
    }
}

/// Repair duration field.
fn repair_duration(value: &str) -> Option<String> {
    // Try to extract numbers from corrupt duration
    let digits: String = value.chars().filter(|c| c.is_ascii_digit()).collect();

    if !digits.is_empty() {
        Some(digits)
    } else {
        Some("0".to_string())
    }
}

/// Repair date field.
fn repair_date(value: &str) -> Option<String> {
    // Try to extract valid date components
    let digits: String = value
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '-')
        .collect();

    if digits.len() >= 8 {
        Some(digits)
    } else {
        Some("1970-01-01".to_string())
    }
}

/// Repair title field.
fn repair_title(value: &str) -> Option<String> {
    Some(sanitize_string(value))
}

/// Sanitize string by removing invalid characters.
fn sanitize_string(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || "-_.,!?".contains(*c))
        .collect()
}

/// Validate metadata fields.
pub fn validate_metadata(fields: &[MetadataField]) -> Vec<String> {
    let mut issues = Vec::new();

    for field in fields {
        if field.value.is_empty() {
            issues.push(format!("Empty value for field: {}", field.name));
        }

        if field.corrupt {
            issues.push(format!("Corrupt field: {}", field.name));
        }
    }

    issues
}

/// Merge metadata from multiple sources.
pub fn merge_metadata(sources: Vec<HashMap<String, String>>) -> HashMap<String, String> {
    let mut merged = HashMap::new();

    for source in sources {
        for (key, value) in source {
            merged.entry(key).or_insert(value);
        }
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_string() {
        let input = "Hello\x00World\x01!";
        let output = sanitize_string(input);
        assert_eq!(output, "HelloWorld!");
    }

    #[test]
    fn test_repair_duration() {
        let result = repair_duration("12abc34");
        assert_eq!(result, Some("1234".to_string()));
    }

    #[test]
    fn test_repair_date() {
        let result = repair_date("2024-01-01xxx");
        assert_eq!(result, Some("2024-01-01".to_string()));
    }
}
