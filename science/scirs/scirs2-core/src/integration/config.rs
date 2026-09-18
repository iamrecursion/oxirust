//! Unified Configuration System for SciRS2 Ecosystem
//!
//! This module provides a comprehensive configuration system for managing
//! ecosystem-wide settings across all SciRS2 modules.
//!
//! # Features
//!
//! - **Hierarchical Configuration**: Module-specific configs with inheritance
//! - **Type-Safe Settings**: Strongly typed configuration values
//! - **Runtime Flexibility**: Dynamic configuration changes with validation
//! - **Cross-Module Consistency**: Unified settings propagation

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

/// Precision level for numerical computations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Precision {
    /// Half precision (16-bit floating point)
    Half,
    /// Single precision (32-bit floating point)
    Single,
    /// Double precision (64-bit floating point)
    #[default]
    Double,
    /// Extended precision (80-bit or higher)
    Extended,
    /// Automatic selection based on operation
    Auto,
}

impl Precision {
    /// Get the number of bits for this precision level
    #[must_use]
    pub const fn bits(&self) -> usize {
        match self {
            Precision::Half => 16,
            Precision::Single => 32,
            Precision::Double => 64,
            Precision::Extended => 80,
            Precision::Auto => 64, // Default to double
        }
    }

    /// Get the epsilon value for this precision
    #[must_use]
    pub fn epsilon(&self) -> f64 {
        match self {
            Precision::Half => 9.77e-4,      // ~2^-10
            Precision::Single => 1.19e-7,    // ~2^-23
            Precision::Double => 2.22e-16,   // ~2^-52
            Precision::Extended => 1.08e-19, // ~2^-63
            Precision::Auto => f64::EPSILON,
        }
    }
}

/// Logging level for diagnostics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub enum LogLevel {
    /// No logging
    Off,
    /// Critical errors only
    Error,
    /// Warnings and errors
    Warn,
    /// Informational messages
    #[default]
    Info,
    /// Detailed debugging information
    Debug,
    /// Very detailed tracing information
    Trace,
}

impl LogLevel {
    /// Check if this level should log at the given level
    #[must_use]
    pub const fn should_log(&self, level: LogLevel) -> bool {
        (*self as u8) >= (level as u8)
    }
}

/// Precision-related configuration
#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionConfig {
    /// Default precision level
    pub default_precision: Precision,
    /// Epsilon for floating-point comparisons
    pub epsilon: f64,
    /// Enable strict precision mode
    pub strict_mode: bool,
    /// Allow precision loss in conversions
    pub allow_precision_loss: bool,
    /// Maximum relative error allowed
    pub max_relative_error: f64,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            default_precision: Precision::Double,
            epsilon: 1e-10,
            strict_mode: false,
            allow_precision_loss: true,
            max_relative_error: 1e-6,
        }
    }
}

/// Parallel processing configuration
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelConfig {
    /// Enable parallel processing
    pub enabled: bool,
    /// Number of threads to use (None = auto-detect)
    pub num_threads: Option<usize>,
    /// Minimum chunk size for parallel operations
    pub min_chunk_size: usize,
    /// Maximum number of parallel tasks
    pub max_tasks: usize,
    /// Enable work stealing
    pub work_stealing: bool,
    /// Thread priority (0-100)
    pub thread_priority: u8,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_threads: None,
            min_chunk_size: 1024,
            max_tasks: 256,
            work_stealing: true,
            thread_priority: 50,
        }
    }
}

impl ParallelConfig {
    /// Get the effective number of threads
    #[must_use]
    pub fn effective_threads(&self) -> usize {
        self.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        })
    }
}

/// Memory management configuration
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryConfig {
    /// Maximum memory limit in bytes (None = unlimited)
    pub max_memory: Option<usize>,
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Pool block size
    pub pool_block_size: usize,
    /// Enable memory-mapped files for large data
    pub enable_mmap: bool,
    /// Threshold for using mmap (in bytes)
    pub mmap_threshold: usize,
    /// Enable memory usage tracking
    pub track_usage: bool,
    /// Cache size limit
    pub cache_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory: None,
            enable_pooling: true,
            pool_block_size: 4096,
            enable_mmap: true,
            mmap_threshold: 64 * 1024 * 1024, // 64 MB
            track_usage: false,
            cache_size: 256 * 1024 * 1024, // 256 MB
        }
    }
}

/// Numeric computation configuration
#[derive(Debug, Clone, PartialEq)]
pub struct NumericConfig {
    /// Maximum iterations for iterative algorithms
    pub max_iterations: usize,
    /// Default tolerance for convergence
    pub tolerance: f64,
    /// Enable overflow checking
    pub check_overflow: bool,
    /// Enable NaN/Inf detection
    pub check_special_values: bool,
    /// Default seed for random operations (None = random)
    pub random_seed: Option<u64>,
    /// Enable deterministic mode
    pub deterministic: bool,
}

impl Default for NumericConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            check_overflow: true,
            check_special_values: true,
            random_seed: None,
            deterministic: false,
        }
    }
}

/// Diagnostics and debugging configuration
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticsConfig {
    /// Logging level
    pub log_level: LogLevel,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Enable tracing
    pub enable_tracing: bool,
    /// Collect performance metrics
    pub collect_metrics: bool,
    /// Enable debug assertions
    pub debug_assertions: bool,
    /// Output format for diagnostics
    pub output_format: DiagnosticsFormat,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            enable_profiling: false,
            enable_tracing: false,
            collect_metrics: false,
            debug_assertions: cfg!(debug_assertions),
            output_format: DiagnosticsFormat::Text,
        }
    }
}

/// Output format for diagnostics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DiagnosticsFormat {
    /// Plain text output
    #[default]
    Text,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Binary format
    Binary,
}

/// Module-specific configuration container
#[derive(Debug, Clone, Default)]
pub struct ModuleConfig {
    /// Module name
    pub name: String,
    /// Custom key-value settings
    pub settings: HashMap<String, ConfigValue>,
    /// Whether this module is enabled
    pub enabled: bool,
}

impl ModuleConfig {
    /// Create a new module configuration
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            settings: HashMap::new(),
            enabled: true,
        }
    }

    /// Set a configuration value
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<ConfigValue>) -> &mut Self {
        self.settings.insert(key.into(), value.into());
        self
    }

    /// Get a configuration value
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.settings.get(key)
    }

    /// Get a typed configuration value
    pub fn get_typed<T: TryFrom<ConfigValue, Error = CoreError>>(
        &self,
        key: &str,
    ) -> CoreResult<T> {
        match self.settings.get(key) {
            Some(value) => T::try_from(value.clone()),
            None => Err(CoreError::ConfigError(
                ErrorContext::new(format!(
                    "Key '{key}' not found in module '{name}'",
                    name = self.name
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }

    /// Enable the module
    pub fn enable(&mut self) -> &mut Self {
        self.enabled = true;
        self
    }

    /// Disable the module
    pub fn disable(&mut self) -> &mut Self {
        self.enabled = false;
        self
    }
}

/// Configuration value type
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Unsigned integer value
    UInt(u64),
    /// Floating-point value
    Float(f64),
    /// String value
    String(String),
    /// List of values
    List(Vec<ConfigValue>),
    /// Map of values
    Map(HashMap<String, ConfigValue>),
}

impl From<bool> for ConfigValue {
    fn from(v: bool) -> Self {
        ConfigValue::Bool(v)
    }
}

impl From<i64> for ConfigValue {
    fn from(v: i64) -> Self {
        ConfigValue::Int(v)
    }
}

impl From<i32> for ConfigValue {
    fn from(v: i32) -> Self {
        ConfigValue::Int(v as i64)
    }
}

impl From<u64> for ConfigValue {
    fn from(v: u64) -> Self {
        ConfigValue::UInt(v)
    }
}

impl From<usize> for ConfigValue {
    fn from(v: usize) -> Self {
        ConfigValue::UInt(v as u64)
    }
}

impl From<f64> for ConfigValue {
    fn from(v: f64) -> Self {
        ConfigValue::Float(v)
    }
}

impl From<f32> for ConfigValue {
    fn from(v: f32) -> Self {
        ConfigValue::Float(v as f64)
    }
}

impl From<String> for ConfigValue {
    fn from(v: String) -> Self {
        ConfigValue::String(v)
    }
}

impl From<&str> for ConfigValue {
    fn from(v: &str) -> Self {
        ConfigValue::String(v.to_string())
    }
}

impl TryFrom<ConfigValue> for bool {
    type Error = CoreError;

    fn try_from(value: ConfigValue) -> Result<Self, Self::Error> {
        match value {
            ConfigValue::Bool(v) => Ok(v),
            _ => Err(CoreError::ConfigError(
                ErrorContext::new(format!("Expected bool, got {:?}", value))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }
}

impl TryFrom<ConfigValue> for i64 {
    type Error = CoreError;

    fn try_from(value: ConfigValue) -> Result<Self, Self::Error> {
        match value {
            ConfigValue::Int(v) => Ok(v),
            ConfigValue::UInt(v) if v <= i64::MAX as u64 => Ok(v as i64),
            _ => Err(CoreError::ConfigError(
                ErrorContext::new(format!("Expected int, got {:?}", value))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }
}

impl TryFrom<ConfigValue> for u64 {
    type Error = CoreError;

    fn try_from(value: ConfigValue) -> Result<Self, Self::Error> {
        match value {
            ConfigValue::UInt(v) => Ok(v),
            ConfigValue::Int(v) if v >= 0 => Ok(v as u64),
            _ => Err(CoreError::ConfigError(
                ErrorContext::new(format!("Expected uint, got {:?}", value))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }
}

impl TryFrom<ConfigValue> for f64 {
    type Error = CoreError;

    fn try_from(value: ConfigValue) -> Result<Self, Self::Error> {
        match value {
            ConfigValue::Float(v) => Ok(v),
            ConfigValue::Int(v) => Ok(v as f64),
            ConfigValue::UInt(v) => Ok(v as f64),
            _ => Err(CoreError::ConfigError(
                ErrorContext::new(format!("Expected float, got {:?}", value))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }
}

impl TryFrom<ConfigValue> for String {
    type Error = CoreError;

    fn try_from(value: ConfigValue) -> Result<Self, Self::Error> {
        match value {
            ConfigValue::String(v) => Ok(v),
            _ => Err(CoreError::ConfigError(
                ErrorContext::new(format!("Expected string, got {:?}", value))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }
}

/// Ecosystem-wide configuration
#[derive(Debug, Clone)]
pub struct EcosystemConfig {
    /// Precision settings
    pub precision: PrecisionConfig,
    /// Parallel processing settings
    pub parallel: ParallelConfig,
    /// Memory management settings
    pub memory: MemoryConfig,
    /// Numeric computation settings
    pub numeric: NumericConfig,
    /// Diagnostics settings
    pub diagnostics: DiagnosticsConfig,
    /// Module-specific configurations
    pub modules: HashMap<String, ModuleConfig>,
    /// Global custom settings
    pub custom: HashMap<String, ConfigValue>,
    /// Configuration version
    pub version: u32,
}

impl Default for EcosystemConfig {
    fn default() -> Self {
        Self {
            precision: PrecisionConfig::default(),
            parallel: ParallelConfig::default(),
            memory: MemoryConfig::default(),
            numeric: NumericConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
            modules: HashMap::new(),
            custom: HashMap::new(),
            version: 1,
        }
    }
}

impl EcosystemConfig {
    /// Create a new default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for configuration
    #[must_use]
    pub fn builder() -> EcosystemConfigBuilder {
        EcosystemConfigBuilder::new()
    }

    /// Get module configuration
    #[must_use]
    pub fn module(&self, name: &str) -> Option<&ModuleConfig> {
        self.modules.get(name)
    }

    /// Get mutable module configuration
    pub fn module_mut(&mut self, name: &str) -> Option<&mut ModuleConfig> {
        self.modules.get_mut(name)
    }

    /// Register a module configuration
    pub fn register_module(&mut self, config: ModuleConfig) -> &mut Self {
        self.modules.insert(config.name.clone(), config);
        self
    }

    /// Set a custom configuration value
    pub fn set_custom(
        &mut self,
        key: impl Into<String>,
        value: impl Into<ConfigValue>,
    ) -> &mut Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Get a custom configuration value
    #[must_use]
    pub fn get_custom(&self, key: &str) -> Option<&ConfigValue> {
        self.custom.get(key)
    }

    /// Validate the configuration
    pub fn validate(&self) -> CoreResult<()> {
        // Validate precision settings
        if self.precision.epsilon <= 0.0 {
            return Err(CoreError::ConfigError(
                ErrorContext::new("Epsilon must be positive")
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        if self.precision.max_relative_error <= 0.0 {
            return Err(CoreError::ConfigError(
                ErrorContext::new("Max relative error must be positive")
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Validate parallel settings
        if self.parallel.min_chunk_size == 0 {
            return Err(CoreError::ConfigError(
                ErrorContext::new("Minimum chunk size must be positive")
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Validate numeric settings
        if self.numeric.max_iterations == 0 {
            return Err(CoreError::ConfigError(
                ErrorContext::new("Maximum iterations must be positive")
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        if self.numeric.tolerance <= 0.0 {
            return Err(CoreError::ConfigError(
                ErrorContext::new("Tolerance must be positive")
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        Ok(())
    }

    /// Merge with another configuration (other takes precedence)
    pub fn merge(&mut self, other: &EcosystemConfig) {
        // Merge modules
        for (name, config) in &other.modules {
            if let Some(existing) = self.modules.get_mut(name) {
                existing.settings.extend(config.settings.clone());
                existing.enabled = config.enabled;
            } else {
                self.modules.insert(name.clone(), config.clone());
            }
        }

        // Merge custom settings
        self.custom.extend(other.custom.clone());
    }
}

/// Builder for EcosystemConfig
#[derive(Debug, Default)]
pub struct EcosystemConfigBuilder {
    config: EcosystemConfig,
}

impl EcosystemConfigBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set precision level
    #[must_use]
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.config.precision.default_precision = precision;
        self.config.precision.epsilon = precision.epsilon();
        self
    }

    /// Set epsilon for comparisons
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.config.precision.epsilon = epsilon;
        self
    }

    /// Enable/disable parallel processing
    #[must_use]
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.config.parallel.enabled = enabled;
        self
    }

    /// Set number of threads
    #[must_use]
    pub fn with_num_threads(mut self, threads: usize) -> Self {
        self.config.parallel.num_threads = Some(threads);
        self
    }

    /// Set maximum memory limit
    #[must_use]
    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.config.memory.max_memory = Some(bytes);
        self
    }

    /// Enable/disable memory pooling
    #[must_use]
    pub fn with_memory_pooling(mut self, enabled: bool) -> Self {
        self.config.memory.enable_pooling = enabled;
        self
    }

    /// Set maximum iterations
    #[must_use]
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.config.numeric.max_iterations = max;
        self
    }

    /// Set tolerance
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.config.numeric.tolerance = tolerance;
        self
    }

    /// Set random seed for deterministic behavior
    #[must_use]
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.config.numeric.random_seed = Some(seed);
        self.config.numeric.deterministic = true;
        self
    }

    /// Enable deterministic mode
    #[must_use]
    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.config.numeric.deterministic = deterministic;
        self
    }

    /// Set log level
    #[must_use]
    pub fn with_log_level(mut self, level: LogLevel) -> Self {
        self.config.diagnostics.log_level = level;
        self
    }

    /// Enable/disable profiling
    #[must_use]
    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.config.diagnostics.enable_profiling = enabled;
        self
    }

    /// Enable/disable tracing
    #[must_use]
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.config.diagnostics.enable_tracing = enabled;
        self
    }

    /// Add a module configuration
    #[must_use]
    pub fn with_module(mut self, config: ModuleConfig) -> Self {
        self.config.modules.insert(config.name.clone(), config);
        self
    }

    /// Add a custom setting
    #[must_use]
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<ConfigValue>) -> Self {
        self.config.custom.insert(key.into(), value.into());
        self
    }

    /// Build the configuration
    ///
    /// # Errors
    ///
    /// Returns error if validation fails
    pub fn build(self) -> CoreResult<EcosystemConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation
    #[must_use]
    pub fn build_unchecked(self) -> EcosystemConfig {
        self.config
    }
}

/// Global ecosystem configuration singleton
static GLOBAL_ECOSYSTEM_CONFIG: std::sync::LazyLock<RwLock<Arc<EcosystemConfig>>> =
    std::sync::LazyLock::new(|| RwLock::new(Arc::new(EcosystemConfig::default())));

/// Get the global ecosystem configuration
#[must_use]
pub fn global_config() -> Arc<EcosystemConfig> {
    GLOBAL_ECOSYSTEM_CONFIG
        .read()
        .map(|guard| Arc::clone(&*guard))
        .unwrap_or_else(|_| Arc::new(EcosystemConfig::default()))
}

/// Set the global ecosystem configuration
pub fn set_global_config(config: EcosystemConfig) -> CoreResult<()> {
    config.validate()?;
    let mut guard = GLOBAL_ECOSYSTEM_CONFIG.write().map_err(|e| {
        CoreError::ConfigError(
            ErrorContext::new(format!("Failed to acquire write lock: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;
    *guard = Arc::new(config);
    Ok(())
}

/// Update the global configuration with a function
pub fn update_global_config<F>(f: F) -> CoreResult<()>
where
    F: FnOnce(&mut EcosystemConfig),
{
    let mut guard = GLOBAL_ECOSYSTEM_CONFIG.write().map_err(|e| {
        CoreError::ConfigError(
            ErrorContext::new(format!("Failed to acquire write lock: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    let mut config = (**guard).clone();
    f(&mut config);
    config.validate()?;
    *guard = Arc::new(config);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_levels() {
        assert_eq!(Precision::Half.bits(), 16);
        assert_eq!(Precision::Single.bits(), 32);
        assert_eq!(Precision::Double.bits(), 64);
        assert_eq!(Precision::Extended.bits(), 80);
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Error < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Trace);
    }

    #[test]
    fn test_config_builder() {
        let config = EcosystemConfig::builder()
            .with_precision(Precision::Single)
            .with_parallel(true)
            .with_num_threads(4)
            .with_max_iterations(500)
            .with_tolerance(1e-6)
            .build()
            .expect("Config should be valid");

        assert_eq!(config.precision.default_precision, Precision::Single);
        assert!(config.parallel.enabled);
        assert_eq!(config.parallel.num_threads, Some(4));
        assert_eq!(config.numeric.max_iterations, 500);
        assert_eq!(config.numeric.tolerance, 1e-6);
    }

    #[test]
    fn test_module_config() {
        let mut module = ModuleConfig::new("test_module");
        module
            .set("key1", true)
            .set("key2", 42i64)
            .set("key3", "value");

        assert!(module.get_typed::<bool>("key1").expect("Should work"));
        assert_eq!(module.get_typed::<i64>("key2").expect("Should work"), 42);
        assert_eq!(
            module.get_typed::<String>("key3").expect("Should work"),
            "value"
        );
    }

    #[test]
    fn test_config_validation() {
        let config = EcosystemConfig::builder()
            .with_epsilon(-1.0)
            .build_unchecked();

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_value_conversions() {
        let bool_val = ConfigValue::from(true);
        assert!(bool::try_from(bool_val).expect("Should work"));

        let int_val = ConfigValue::from(42i64);
        assert_eq!(i64::try_from(int_val).expect("Should work"), 42);

        let float_val = ConfigValue::from(3.5f64);
        let f = f64::try_from(float_val).expect("Should work");
        assert!((f - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig {
            num_threads: Some(8),
            ..Default::default()
        };
        assert_eq!(config.effective_threads(), 8);

        let auto_config = ParallelConfig::default();
        assert!(auto_config.effective_threads() >= 1);
    }

    #[test]
    fn test_ecosystem_config_merge() {
        let mut config1 = EcosystemConfig::new();
        config1.set_custom("key1", "value1");
        config1.register_module(ModuleConfig::new("module1"));

        let mut config2 = EcosystemConfig::new();
        config2.set_custom("key2", "value2");
        config2.register_module(ModuleConfig::new("module2"));

        config1.merge(&config2);

        assert!(config1.get_custom("key1").is_some());
        assert!(config1.get_custom("key2").is_some());
        assert!(config1.module("module1").is_some());
        assert!(config1.module("module2").is_some());
    }
}
