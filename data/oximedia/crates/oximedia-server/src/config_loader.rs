//! Hierarchical server configuration loader.
//!
//! Provides `ConfigKey`, `ConfigValue`, `ServerConfig`, and `ConfigLoader`
//! for loading and querying typed server configuration.

#![allow(dead_code)]

use std::collections::HashMap;

// ── ConfigKey ─────────────────────────────────────────────────────────────────

/// Well-known configuration keys for the media server.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConfigKey {
    /// Bind address (e.g., "0.0.0.0:8080").
    BindAddress,
    /// Database connection URL.
    DatabaseUrl,
    /// Maximum concurrent connections.
    MaxConnections,
    /// Request timeout in seconds.
    RequestTimeoutSecs,
    /// Enable HTTPS (TLS).
    TlsEnabled,
    /// Path to TLS certificate file.
    TlsCertPath,
    /// Path to TLS private key file.
    TlsKeyPath,
    /// Log level string (e.g., "info", "debug").
    LogLevel,
    /// Maximum upload size in bytes.
    MaxUploadBytes,
    /// Custom string key not covered by the above variants.
    Custom(String),
}

impl ConfigKey {
    /// Returns `true` when this key must be present in a valid configuration.
    pub fn required(&self) -> bool {
        matches!(self, Self::BindAddress | Self::DatabaseUrl)
    }

    /// String representation used in config files / env vars.
    pub fn as_env_key(&self) -> String {
        match self {
            Self::BindAddress => "SERVER_BIND_ADDRESS".into(),
            Self::DatabaseUrl => "SERVER_DATABASE_URL".into(),
            Self::MaxConnections => "SERVER_MAX_CONNECTIONS".into(),
            Self::RequestTimeoutSecs => "SERVER_REQUEST_TIMEOUT_SECS".into(),
            Self::TlsEnabled => "SERVER_TLS_ENABLED".into(),
            Self::TlsCertPath => "SERVER_TLS_CERT_PATH".into(),
            Self::TlsKeyPath => "SERVER_TLS_KEY_PATH".into(),
            Self::LogLevel => "SERVER_LOG_LEVEL".into(),
            Self::MaxUploadBytes => "SERVER_MAX_UPLOAD_BYTES".into(),
            Self::Custom(s) => s.clone(),
        }
    }
}

// ── ConfigValue ───────────────────────────────────────────────────────────────

/// A typed configuration value.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    /// Plain string.
    Str(String),
    /// Integer value.
    Int(i64),
    /// Boolean value.
    Bool(bool),
    /// Floating-point value.
    Float(f64),
}

impl ConfigValue {
    /// Try to interpret as an `i64`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Str(s) => s.parse().ok(),
            Self::Bool(b) => Some(i64::from(*b)),
            Self::Float(f) => Some(*f as i64),
        }
    }

    /// Try to interpret as a `bool`.
    ///
    /// Strings `"true"`, `"1"`, `"yes"` resolve to `true`;
    /// `"false"`, `"0"`, `"no"` to `false`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            Self::Int(i) => Some(*i != 0),
            Self::Str(s) => match s.trim().to_lowercase().as_str() {
                "true" | "1" | "yes" => Some(true),
                "false" | "0" | "no" => Some(false),
                _ => None,
            },
            Self::Float(f) => Some(*f != 0.0),
        }
    }

    /// Try to interpret as a string slice.
    pub fn as_str(&self) -> Option<&str> {
        if let Self::Str(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    /// Try to interpret as an `f64`.
    #[allow(clippy::cast_precision_loss)]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(i) => Some(*i as f64),
            Self::Str(s) => s.parse().ok(),
            Self::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        }
    }
}

impl From<&str> for ConfigValue {
    fn from(s: &str) -> Self {
        Self::Str(s.to_string())
    }
}

impl From<String> for ConfigValue {
    fn from(s: String) -> Self {
        Self::Str(s)
    }
}

impl From<i64> for ConfigValue {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

impl From<bool> for ConfigValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<f64> for ConfigValue {
    fn from(f: f64) -> Self {
        Self::Float(f)
    }
}

// ── ServerConfig ──────────────────────────────────────────────────────────────

/// A flat key-value server configuration map.
pub struct ServerConfig {
    values: HashMap<String, ConfigValue>,
}

impl ServerConfig {
    /// Create an empty configuration.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Insert or overwrite a value.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<ConfigValue>) {
        self.values.insert(key.into(), value.into());
    }

    /// Retrieve a value by key string.
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.values.get(key)
    }

    /// Returns `true` when the key is present.
    pub fn has_key(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// Number of stored keys.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` when no keys are stored.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterate over all key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ConfigValue)> {
        self.values.iter()
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── ConfigLoader ──────────────────────────────────────────────────────────────

/// Loads and merges server configuration from multiple sources.
pub struct ConfigLoader {
    config: ServerConfig,
}

impl ConfigLoader {
    /// Create a loader with an empty configuration.
    pub fn new() -> Self {
        Self {
            config: ServerConfig::new(),
        }
    }

    /// Populate well-known defaults so the server can start without a config file.
    pub fn load_defaults(&mut self) -> &mut Self {
        self.config.set(
            "SERVER_BIND_ADDRESS",
            ConfigValue::Str("0.0.0.0:8080".into()),
        );
        self.config.set(
            "SERVER_DATABASE_URL",
            ConfigValue::Str("sqlite://:memory:".into()),
        );
        self.config
            .set("SERVER_MAX_CONNECTIONS", ConfigValue::Int(100));
        self.config
            .set("SERVER_REQUEST_TIMEOUT_SECS", ConfigValue::Int(30));
        self.config
            .set("SERVER_TLS_ENABLED", ConfigValue::Bool(false));
        self.config
            .set("SERVER_LOG_LEVEL", ConfigValue::Str("info".into()));
        self.config.set(
            "SERVER_MAX_UPLOAD_BYTES",
            ConfigValue::Int(5 * 1024 * 1024 * 1024),
        );
        self
    }

    /// Merge values from a string map (e.g., parsed from a TOML/env source).
    pub fn merge_map(&mut self, map: HashMap<String, String>) -> &mut Self {
        for (k, v) in map {
            self.config.set(k, ConfigValue::Str(v));
        }
        self
    }

    /// Override a single key.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<ConfigValue>) -> &mut Self {
        self.config.set(key, value);
        self
    }

    /// Consume the loader and return the built `ServerConfig`.
    pub fn build(self) -> ServerConfig {
        self.config
    }

    /// Validate that all required keys are present.
    ///
    /// Returns a list of missing required env-key strings.
    pub fn validate_required(&self) -> Vec<String> {
        let required_keys = [ConfigKey::BindAddress, ConfigKey::DatabaseUrl];
        required_keys
            .iter()
            .filter(|k| k.required() && !self.config.has_key(&k.as_env_key()))
            .map(|k| k.as_env_key())
            .collect()
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ConfigKey

    #[test]
    fn config_key_required() {
        assert!(ConfigKey::BindAddress.required());
        assert!(ConfigKey::DatabaseUrl.required());
        assert!(!ConfigKey::LogLevel.required());
        assert!(!ConfigKey::TlsEnabled.required());
    }

    #[test]
    fn config_key_env_key() {
        assert_eq!(ConfigKey::BindAddress.as_env_key(), "SERVER_BIND_ADDRESS");
        assert_eq!(
            ConfigKey::MaxConnections.as_env_key(),
            "SERVER_MAX_CONNECTIONS"
        );
    }

    #[test]
    fn config_key_custom() {
        let k = ConfigKey::Custom("MY_KEY".into());
        assert_eq!(k.as_env_key(), "MY_KEY");
        assert!(!k.required());
    }

    // ConfigValue

    #[test]
    fn config_value_as_int_from_int() {
        assert_eq!(ConfigValue::Int(42).as_int(), Some(42));
    }

    #[test]
    fn config_value_as_int_from_str() {
        assert_eq!(ConfigValue::Str("99".into()).as_int(), Some(99));
        assert_eq!(ConfigValue::Str("bad".into()).as_int(), None);
    }

    #[test]
    fn config_value_as_bool_from_bool() {
        assert_eq!(ConfigValue::Bool(true).as_bool(), Some(true));
        assert_eq!(ConfigValue::Bool(false).as_bool(), Some(false));
    }

    #[test]
    fn config_value_as_bool_from_str() {
        assert_eq!(ConfigValue::Str("true".into()).as_bool(), Some(true));
        assert_eq!(ConfigValue::Str("yes".into()).as_bool(), Some(true));
        assert_eq!(ConfigValue::Str("0".into()).as_bool(), Some(false));
        assert_eq!(ConfigValue::Str("no".into()).as_bool(), Some(false));
        assert_eq!(ConfigValue::Str("maybe".into()).as_bool(), None);
    }

    #[test]
    fn config_value_as_bool_from_int() {
        assert_eq!(ConfigValue::Int(1).as_bool(), Some(true));
        assert_eq!(ConfigValue::Int(0).as_bool(), Some(false));
    }

    #[test]
    fn config_value_as_str() {
        assert_eq!(ConfigValue::Str("hello".into()).as_str(), Some("hello"));
        assert_eq!(ConfigValue::Int(1).as_str(), None);
    }

    #[test]
    fn config_value_as_float_from_int() {
        let v = ConfigValue::Int(3)
            .as_float()
            .expect("should succeed in test");
        assert!((v - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn config_value_from_conversions() {
        let _: ConfigValue = "hello".into();
        let _: ConfigValue = 42_i64.into();
        let _: ConfigValue = true.into();
        let _: ConfigValue = 3.14_f64.into();
    }

    // ServerConfig

    #[test]
    fn server_config_set_get() {
        let mut cfg = ServerConfig::new();
        cfg.set("key1", ConfigValue::Int(7));
        assert_eq!(cfg.get("key1"), Some(&ConfigValue::Int(7)));
        assert!(cfg.has_key("key1"));
        assert!(!cfg.has_key("key2"));
    }

    #[test]
    fn server_config_len_and_empty() {
        let mut cfg = ServerConfig::new();
        assert!(cfg.is_empty());
        cfg.set("k", ConfigValue::Bool(true));
        assert_eq!(cfg.len(), 1);
    }

    // ConfigLoader

    #[test]
    fn loader_load_defaults_provides_bind_address() {
        let mut loader = ConfigLoader::new();
        loader.load_defaults();
        let cfg = loader.build();
        assert!(cfg.has_key("SERVER_BIND_ADDRESS"));
        assert_eq!(
            cfg.get("SERVER_BIND_ADDRESS").and_then(|v| v.as_str()),
            Some("0.0.0.0:8080")
        );
    }

    #[test]
    fn loader_load_defaults_max_connections() {
        let mut loader = ConfigLoader::new();
        loader.load_defaults();
        let cfg = loader.build();
        assert_eq!(
            cfg.get("SERVER_MAX_CONNECTIONS").and_then(|v| v.as_int()),
            Some(100)
        );
    }

    #[test]
    fn loader_validate_required_missing() {
        let loader = ConfigLoader::new(); // no defaults loaded
        let missing = loader.validate_required();
        assert!(missing.contains(&"SERVER_BIND_ADDRESS".to_string()));
        assert!(missing.contains(&"SERVER_DATABASE_URL".to_string()));
    }

    #[test]
    fn loader_validate_required_satisfied() {
        let mut loader = ConfigLoader::new();
        loader.load_defaults();
        let missing = loader.validate_required();
        assert!(missing.is_empty(), "unexpected missing: {missing:?}");
    }

    #[test]
    fn loader_merge_map_overrides() {
        let mut loader = ConfigLoader::new();
        loader.load_defaults();
        let mut overrides = HashMap::new();
        overrides.insert("SERVER_BIND_ADDRESS".into(), "127.0.0.1:9090".into());
        loader.merge_map(overrides);
        let cfg = loader.build();
        assert_eq!(
            cfg.get("SERVER_BIND_ADDRESS").and_then(|v| v.as_str()),
            Some("127.0.0.1:9090")
        );
    }

    #[test]
    fn loader_tls_disabled_by_default() {
        let mut loader = ConfigLoader::new();
        loader.load_defaults();
        let cfg = loader.build();
        assert_eq!(
            cfg.get("SERVER_TLS_ENABLED").and_then(|v| v.as_bool()),
            Some(false)
        );
    }
}
