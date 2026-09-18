// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Template variable substitution.

use std::collections::HashMap;

/// Variables for template substitution.
#[derive(Debug, Clone, Default)]
pub struct TemplateVariables {
    variables: HashMap<String, String>,
}

impl TemplateVariables {
    /// Create a new template variables collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Add a variable.
    pub fn set<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.variables.insert(key.into(), value.into());
    }

    /// Get a variable value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&String> {
        self.variables.get(key)
    }

    /// Check if a variable exists.
    #[must_use]
    pub fn has(&self, key: &str) -> bool {
        self.variables.contains_key(key)
    }

    /// Remove a variable.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.variables.remove(key)
    }

    /// Substitute variables in a string.
    #[must_use]
    pub fn substitute(&self, template: &str) -> String {
        let mut result = template.to_string();

        for (key, value) in &self.variables {
            let placeholder = format!("{{{key}}}");
            result = result.replace(&placeholder, value);
        }

        result
    }

    /// Add common file variables from a path.
    pub fn add_file_variables(&mut self, path: &std::path::Path) {
        if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
            self.set("name", name);
            self.set("filename", name);
        }

        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            self.set("ext", ext);
            self.set("extension", ext);
        }

        if let Some(parent) = path.parent().and_then(|p| p.to_str()) {
            self.set("dir", parent);
            self.set("directory", parent);
        }
    }

    /// Add timestamp variables.
    pub fn add_timestamp_variables(&mut self) {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.set("timestamp", timestamp.to_string());

        // Add formatted date/time
        // In a real implementation, this would use chrono or similar
        self.set("date", format!("{timestamp}"));
        self.set("time", format!("{timestamp}"));
    }

    /// Add index variable.
    pub fn add_index(&mut self, index: usize) {
        self.set("index", index.to_string());
        self.set("i", index.to_string());
    }

    /// Get all variables.
    #[must_use]
    pub fn all(&self) -> &HashMap<String, String> {
        &self.variables
    }

    /// Clear all variables.
    pub fn clear(&mut self) {
        self.variables.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_variables_creation() {
        let vars = TemplateVariables::new();
        assert!(vars.variables.is_empty());
    }

    #[test]
    fn test_set_get() {
        let mut vars = TemplateVariables::new();
        vars.set("key", "value");

        assert_eq!(vars.get("key"), Some(&"value".to_string()));
        assert!(vars.has("key"));
    }

    #[test]
    fn test_substitute() {
        let mut vars = TemplateVariables::new();
        vars.set("name", "video");
        vars.set("ext", "mp4");

        let result = vars.substitute("{name}.{ext}");
        assert_eq!(result, "video.mp4");

        let result = vars.substitute("output_{name}_final.{ext}");
        assert_eq!(result, "output_video_final.mp4");
    }

    #[test]
    fn test_file_variables() {
        let mut vars = TemplateVariables::new();
        let path = Path::new("/path/to/video.mp4");

        vars.add_file_variables(path);

        assert_eq!(vars.get("name"), Some(&"video".to_string()));
        assert_eq!(vars.get("ext"), Some(&"mp4".to_string()));
    }

    #[test]
    fn test_index_variable() {
        let mut vars = TemplateVariables::new();
        vars.add_index(5);

        assert_eq!(vars.get("index"), Some(&"5".to_string()));
        assert_eq!(vars.get("i"), Some(&"5".to_string()));
    }

    #[test]
    fn test_timestamp_variables() {
        let mut vars = TemplateVariables::new();
        vars.add_timestamp_variables();

        assert!(vars.has("timestamp"));
        assert!(vars.has("date"));
        assert!(vars.has("time"));
    }

    #[test]
    fn test_remove() {
        let mut vars = TemplateVariables::new();
        vars.set("key", "value");

        let removed = vars.remove("key");
        assert_eq!(removed, Some("value".to_string()));
        assert!(!vars.has("key"));
    }

    #[test]
    fn test_clear() {
        let mut vars = TemplateVariables::new();
        vars.set("key1", "value1");
        vars.set("key2", "value2");

        vars.clear();
        assert!(vars.variables.is_empty());
    }
}
