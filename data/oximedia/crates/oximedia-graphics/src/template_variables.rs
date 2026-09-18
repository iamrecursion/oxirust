//! Template variable hot-reload from JSON.
//!
//! Provides a `TemplateVariableStore` that holds typed key-value variables
//! and can reload them from a JSON source (file path or JSON string). The
//! store tracks which variables have changed since the last check so that
//! consumers can react only to updates rather than polling every frame.

use serde_json::Value;
use std::collections::HashMap;

/// Supported variable value types.
#[derive(Debug, Clone, PartialEq)]
pub enum VarValue {
    /// String value.
    Text(String),
    /// Numeric (floating-point) value.
    Number(f64),
    /// Boolean value.
    Bool(bool),
    /// RGBA color encoded as `[r, g, b, a]` in 0..=255.
    Color([u8; 4]),
    /// Null / unset value.
    Null,
}

impl VarValue {
    /// Try to interpret the value as a string slice.
    pub fn as_str(&self) -> Option<&str> {
        if let Self::Text(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    /// Try to interpret the value as a number.
    pub fn as_f64(&self) -> Option<f64> {
        if let Self::Number(n) = self {
            Some(*n)
        } else {
            None
        }
    }

    /// Try to interpret the value as a bool.
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    /// Try to interpret the value as a color.
    pub fn as_color(&self) -> Option<[u8; 4]> {
        if let Self::Color(c) = self {
            Some(*c)
        } else {
            None
        }
    }

    /// Returns `true` if the value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

impl From<&str> for VarValue {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<String> for VarValue {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<f64> for VarValue {
    fn from(n: f64) -> Self {
        Self::Number(n)
    }
}

impl From<f32> for VarValue {
    fn from(n: f32) -> Self {
        Self::Number(n as f64)
    }
}

impl From<bool> for VarValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

/// Convert a `serde_json::Value` into a `VarValue`.
fn json_to_var_value(val: &Value) -> VarValue {
    match val {
        Value::String(s) => VarValue::Text(s.clone()),
        Value::Number(n) => VarValue::Number(n.as_f64().unwrap_or(0.0)),
        Value::Bool(b) => VarValue::Bool(*b),
        Value::Array(arr) if arr.len() == 4 => {
            // Try to interpret a 4-element array as RGBA.
            let extract = |v: &Value| -> Option<u8> { v.as_u64().map(|n| n.min(255) as u8) };
            if let (Some(r), Some(g), Some(b), Some(a)) = (
                extract(&arr[0]),
                extract(&arr[1]),
                extract(&arr[2]),
                extract(&arr[3]),
            ) {
                VarValue::Color([r, g, b, a])
            } else {
                VarValue::Null
            }
        }
        Value::Null => VarValue::Null,
        _ => VarValue::Null,
    }
}

/// A variable store that supports hot-reload from JSON.
///
/// Variables are identified by string keys. The store tracks which variables
/// have been modified since the last call to `clear_dirty()`.
#[derive(Debug, Default)]
pub struct TemplateVariableStore {
    /// Current variable values.
    variables: HashMap<String, VarValue>,
    /// Keys that have changed since the last `clear_dirty()`.
    dirty: Vec<String>,
}

impl TemplateVariableStore {
    /// Create an empty variable store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a variable by key and value.
    ///
    /// Marks the key as dirty if the value changed.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<VarValue>) {
        let k = key.into();
        let v = value.into();
        let changed = self.variables.get(&k) != Some(&v);
        self.variables.insert(k.clone(), v);
        if changed {
            if !self.dirty.contains(&k) {
                self.dirty.push(k);
            }
        }
    }

    /// Get a variable value by key.
    pub fn get(&self, key: &str) -> Option<&VarValue> {
        self.variables.get(key)
    }

    /// Remove a variable.
    pub fn remove(&mut self, key: &str) -> Option<VarValue> {
        let removed = self.variables.remove(key);
        if removed.is_some() && !self.dirty.contains(&key.to_string()) {
            self.dirty.push(key.to_string());
        }
        removed
    }

    /// Number of variables stored.
    pub fn len(&self) -> usize {
        self.variables.len()
    }

    /// Returns `true` if there are no variables.
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }

    /// Returns the list of dirty (changed) keys since the last `clear_dirty()`.
    pub fn dirty_keys(&self) -> &[String] {
        &self.dirty
    }

    /// Returns `true` if any variable has changed since the last `clear_dirty()`.
    pub fn has_changes(&self) -> bool {
        !self.dirty.is_empty()
    }

    /// Clear the dirty set (acknowledge all changes).
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// Load / merge variables from a JSON string.
    ///
    /// The JSON must be an object `{ "key": value, ... }`.
    /// Existing variables not present in the JSON are left unchanged.
    ///
    /// Returns the number of variables that changed.
    pub fn load_json_str(&mut self, json: &str) -> Result<usize, serde_json::Error> {
        let root: Value = serde_json::from_str(json)?;
        let mut changed = 0;
        if let Value::Object(map) = root {
            for (k, v) in map {
                let new_val = json_to_var_value(&v);
                let is_new = self
                    .variables
                    .get(&k)
                    .map(|old| old != &new_val)
                    .unwrap_or(true);
                if is_new {
                    self.variables.insert(k.clone(), new_val);
                    if !self.dirty.contains(&k) {
                        self.dirty.push(k);
                    }
                    changed += 1;
                }
            }
        }
        Ok(changed)
    }

    /// Load variables from a JSON file at `path`.
    ///
    /// Returns the number of variables that changed.
    pub fn load_json_file(&mut self, path: &std::path::Path) -> std::io::Result<usize> {
        let content = std::fs::read_to_string(path)?;
        self.load_json_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Serialize the current variable store to a JSON string.
    pub fn to_json_str(&self) -> String {
        let mut map = serde_json::Map::new();
        for (k, v) in &self.variables {
            let json_val = match v {
                VarValue::Text(s) => Value::String(s.clone()),
                VarValue::Number(n) => Value::Number(
                    serde_json::Number::from_f64(*n).unwrap_or(serde_json::Number::from(0)),
                ),
                VarValue::Bool(b) => Value::Bool(*b),
                VarValue::Color([r, g, b, a]) => Value::Array(vec![
                    Value::Number((*r).into()),
                    Value::Number((*g).into()),
                    Value::Number((*b).into()),
                    Value::Number((*a).into()),
                ]),
                VarValue::Null => Value::Null,
            };
            map.insert(k.clone(), json_val);
        }
        serde_json::to_string(&Value::Object(map)).unwrap_or_default()
    }

    /// Iterate over all current variables.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &VarValue)> {
        self.variables.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    // -----------------------------------------------------------------------
    // VarValue tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_var_value_text() {
        let v = VarValue::Text("hello".to_string());
        assert_eq!(v.as_str(), Some("hello"));
        assert!(v.as_f64().is_none());
        assert!(!v.is_null());
    }

    #[test]
    fn test_var_value_number() {
        let v = VarValue::Number(42.5);
        assert_eq!(v.as_f64(), Some(42.5));
        assert!(v.as_str().is_none());
    }

    #[test]
    fn test_var_value_bool() {
        let v = VarValue::Bool(true);
        assert_eq!(v.as_bool(), Some(true));
    }

    #[test]
    fn test_var_value_color() {
        let v = VarValue::Color([255, 128, 0, 255]);
        assert_eq!(v.as_color(), Some([255, 128, 0, 255]));
    }

    #[test]
    fn test_var_value_null() {
        let v = VarValue::Null;
        assert!(v.is_null());
        assert!(v.as_str().is_none());
        assert!(v.as_f64().is_none());
    }

    #[test]
    fn test_var_value_from_str() {
        let v: VarValue = "world".into();
        assert_eq!(v.as_str(), Some("world"));
    }

    #[test]
    fn test_var_value_from_f64() {
        let v: VarValue = 3.14_f64.into();
        assert!((v.as_f64().expect("should be f64") - 3.14).abs() < 1e-9);
    }

    #[test]
    fn test_var_value_from_bool() {
        let v: VarValue = false.into();
        assert_eq!(v.as_bool(), Some(false));
    }

    // -----------------------------------------------------------------------
    // TemplateVariableStore tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_set_get() {
        let mut store = TemplateVariableStore::new();
        store.set("name", "Alice");
        assert_eq!(store.get("name").and_then(VarValue::as_str), Some("Alice"));
    }

    #[test]
    fn test_store_set_numeric() {
        let mut store = TemplateVariableStore::new();
        store.set("score", 100.0_f64);
        assert_eq!(store.get("score").and_then(VarValue::as_f64), Some(100.0));
    }

    #[test]
    fn test_store_set_bool() {
        let mut store = TemplateVariableStore::new();
        store.set("visible", true);
        assert_eq!(store.get("visible").and_then(VarValue::as_bool), Some(true));
    }

    #[test]
    fn test_store_len() {
        let mut store = TemplateVariableStore::new();
        assert!(store.is_empty());
        store.set("a", "1");
        store.set("b", "2");
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_store_remove() {
        let mut store = TemplateVariableStore::new();
        store.set("x", "val");
        let removed = store.remove("x");
        assert!(removed.is_some());
        assert!(store.get("x").is_none());
    }

    #[test]
    fn test_store_dirty_tracking() {
        let mut store = TemplateVariableStore::new();
        store.set("a", "1");
        assert!(store.has_changes());
        assert!(store.dirty_keys().contains(&"a".to_string()));
        store.clear_dirty();
        assert!(!store.has_changes());
    }

    #[test]
    fn test_store_dirty_only_on_change() {
        let mut store = TemplateVariableStore::new();
        store.set("key", "value");
        store.clear_dirty();
        store.set("key", "value"); // Same value — should not be dirty.
        assert!(!store.has_changes());
    }

    #[test]
    fn test_store_dirty_on_different_value() {
        let mut store = TemplateVariableStore::new();
        store.set("key", "old");
        store.clear_dirty();
        store.set("key", "new");
        assert!(store.has_changes());
    }

    #[test]
    fn test_store_load_json_str_text() {
        let mut store = TemplateVariableStore::new();
        let changed = store
            .load_json_str(r#"{"title":"Breaking News","score":42.0}"#)
            .expect("should parse");
        assert_eq!(changed, 2);
        assert_eq!(
            store.get("title").and_then(VarValue::as_str),
            Some("Breaking News")
        );
        assert_eq!(store.get("score").and_then(VarValue::as_f64), Some(42.0));
    }

    #[test]
    fn test_store_load_json_str_bool() {
        let mut store = TemplateVariableStore::new();
        store.load_json_str(r#"{"visible":true}"#).expect("parse");
        assert_eq!(store.get("visible").and_then(VarValue::as_bool), Some(true));
    }

    #[test]
    fn test_store_load_json_str_color() {
        let mut store = TemplateVariableStore::new();
        store
            .load_json_str(r#"{"accent":[255,128,0,255]}"#)
            .expect("parse");
        assert_eq!(
            store.get("accent").and_then(VarValue::as_color),
            Some([255, 128, 0, 255])
        );
    }

    #[test]
    fn test_store_load_json_str_no_change_on_same() {
        let mut store = TemplateVariableStore::new();
        store.load_json_str(r#"{"x":"same"}"#).expect("parse");
        store.clear_dirty();
        let changed = store.load_json_str(r#"{"x":"same"}"#).expect("parse");
        assert_eq!(changed, 0);
    }

    #[test]
    fn test_store_load_json_str_invalid() {
        let mut store = TemplateVariableStore::new();
        let result = store.load_json_str("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_store_to_json_roundtrip() {
        let mut store = TemplateVariableStore::new();
        store.set("name", "Test");
        store.set("count", 5.0_f64);
        let json = store.to_json_str();
        let mut store2 = TemplateVariableStore::new();
        store2.load_json_str(&json).expect("roundtrip parse");
        assert_eq!(store2.get("name").and_then(VarValue::as_str), Some("Test"));
        assert_eq!(store2.get("count").and_then(VarValue::as_f64), Some(5.0));
    }

    #[test]
    fn test_store_load_json_file() {
        let dir = env::temp_dir();
        let path = dir.join("oximedia_graphics_template_vars_test.json");
        fs::write(&path, r#"{"lower_name":"Jane","score":99.0}"#).expect("write temp file");

        let mut store = TemplateVariableStore::new();
        let changed = store.load_json_file(&path).expect("load from file");
        assert_eq!(changed, 2);
        assert_eq!(
            store.get("lower_name").and_then(VarValue::as_str),
            Some("Jane")
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_store_iter() {
        let mut store = TemplateVariableStore::new();
        store.set("a", "1");
        store.set("b", "2");
        let keys: Vec<&String> = store.iter().map(|(k, _)| k).collect();
        assert!(keys.contains(&&"a".to_string()));
        assert!(keys.contains(&&"b".to_string()));
    }
}
