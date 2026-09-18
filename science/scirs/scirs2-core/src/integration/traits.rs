//! Common Interface Traits for Module Interoperability
//!
//! This module defines standard traits that SciRS2 modules can implement
//! to enable seamless interoperability across the ecosystem.
//!
//! # Core Traits
//!
//! - **Configurable**: Modules that accept configuration
//! - **DataProvider/DataConsumer**: Data flow interfaces
//! - **ModuleCapability**: Capability discovery
//! - **Serializable**: Cross-module serialization
//! - **ResourceAware**: Resource management
//! - **Diagnosable**: Diagnostics and debugging

use std::any::Any;
use std::collections::HashMap;
use std::fmt;

use super::config::{EcosystemConfig, ModuleConfig};
use super::zero_copy::{SharedArrayView, SharedArrayViewMut};
use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

/// Trait for identifiable components
pub trait Identifiable {
    /// Get the unique identifier for this component
    fn id(&self) -> &str;

    /// Get a human-readable name
    fn name(&self) -> &str;

    /// Get the component type
    fn component_type(&self) -> &str;

    /// Get additional metadata
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Trait for configurable modules
pub trait Configurable {
    /// Apply ecosystem-wide configuration
    fn configure(&mut self, config: &EcosystemConfig) -> CoreResult<()>;

    /// Apply module-specific configuration
    fn configure_module(&mut self, config: &ModuleConfig) -> CoreResult<()>;

    /// Get current configuration as key-value pairs
    fn get_config(&self) -> HashMap<String, String>;

    /// Reset to default configuration
    fn reset_config(&mut self);

    /// Validate current configuration
    fn validate_config(&self) -> CoreResult<()>;
}

/// Trait for components that provide data
pub trait DataProvider<T> {
    /// Get the number of items available
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a read-only view of the data
    fn view(&self) -> CoreResult<SharedArrayView<'_, T>>;

    /// Get data shape (for multi-dimensional providers)
    fn shape(&self) -> &[usize];

    /// Get data type name
    fn dtype(&self) -> &str;

    /// Check if data is contiguous in memory
    fn is_contiguous(&self) -> bool;
}

/// Trait for components that consume data
pub trait DataConsumer<T> {
    /// Consume data from a provider
    fn consume<P: DataProvider<T>>(&mut self, provider: &P) -> CoreResult<()>;

    /// Consume data from a slice
    fn consume_slice(&mut self, data: &[T]) -> CoreResult<()>;

    /// Consume data from a view
    fn consume_view(&mut self, view: SharedArrayView<'_, T>) -> CoreResult<()>;

    /// Get expected input shape
    fn expected_shape(&self) -> Option<&[usize]>;

    /// Check if consumer can accept the given shape
    fn can_accept_shape(&self, shape: &[usize]) -> bool;
}

/// Trait for components that can provide mutable data access
pub trait MutableDataProvider<T>: DataProvider<T> {
    /// Get a mutable view of the data
    fn view_mut(&mut self) -> CoreResult<SharedArrayViewMut<'_, T>>;

    /// Apply an operation to all elements
    fn apply<F>(&mut self, f: F) -> CoreResult<()>
    where
        F: Fn(&mut T);

    /// Clear all data
    fn clear(&mut self);
}

/// Module capabilities enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Can process in parallel
    Parallel,
    /// Can use GPU acceleration
    GpuAcceleration,
    /// Supports SIMD operations
    Simd,
    /// Supports streaming/incremental processing
    Streaming,
    /// Supports distributed processing
    Distributed,
    /// Can serialize/deserialize state
    Serializable,
    /// Supports checkpointing
    Checkpointing,
    /// Thread-safe operations
    ThreadSafe,
    /// Async operations supported
    Async,
    /// Memory-mapped file support
    MemoryMapped,
    /// Zero-copy data sharing
    ZeroCopy,
    /// Supports batched operations
    Batched,
    /// Custom capability
    Custom(&'static str),
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Capability::Parallel => write!(f, "parallel"),
            Capability::GpuAcceleration => write!(f, "gpu"),
            Capability::Simd => write!(f, "simd"),
            Capability::Streaming => write!(f, "streaming"),
            Capability::Distributed => write!(f, "distributed"),
            Capability::Serializable => write!(f, "serializable"),
            Capability::Checkpointing => write!(f, "checkpointing"),
            Capability::ThreadSafe => write!(f, "thread-safe"),
            Capability::Async => write!(f, "async"),
            Capability::MemoryMapped => write!(f, "mmap"),
            Capability::ZeroCopy => write!(f, "zero-copy"),
            Capability::Batched => write!(f, "batched"),
            Capability::Custom(name) => write!(f, "custom:{name}"),
        }
    }
}

/// Trait for capability discovery
pub trait ModuleCapability {
    /// Get list of supported capabilities
    fn capabilities(&self) -> Vec<Capability>;

    /// Check if a specific capability is supported
    fn has_capability(&self, cap: Capability) -> bool {
        self.capabilities().contains(&cap)
    }

    /// Get capability requirements for a specific operation
    fn required_capabilities(&self, operation: &str) -> Vec<Capability>;

    /// Check if all required capabilities for an operation are available
    fn can_perform(&self, operation: &str) -> bool {
        let required = self.required_capabilities(operation);
        required.iter().all(|cap| self.has_capability(*cap))
    }
}

/// Serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SerializationFormat {
    /// JSON format
    Json,
    /// MessagePack format
    MessagePack,
    /// Binary format (custom)
    Binary,
    /// CBOR format
    Cbor,
    /// Protocol Buffers
    Protobuf,
}

/// Trait for serializable components
pub trait Serializable {
    /// Supported serialization formats
    fn supported_formats(&self) -> Vec<SerializationFormat>;

    /// Serialize to bytes
    fn serialize(&self, format: SerializationFormat) -> CoreResult<Vec<u8>>;

    /// Deserialize from bytes
    fn deserialize(&mut self, data: &[u8], format: SerializationFormat) -> CoreResult<()>;

    /// Get serialization size estimate
    fn estimated_size(&self, format: SerializationFormat) -> usize;
}

/// Trait for cross-module operators
pub trait CrossModuleOperator<Input, Output> {
    /// Apply the operation
    fn apply(&self, input: &Input) -> CoreResult<Output>;

    /// Apply in-place if possible
    fn apply_inplace(&self, data: &mut Input) -> CoreResult<()>
    where
        Input: From<Output>;

    /// Get the operator name
    fn operator_name(&self) -> &str;

    /// Get input type information
    fn input_info(&self) -> &str;

    /// Get output type information
    fn output_info(&self) -> &str;

    /// Check if operator is deterministic
    fn is_deterministic(&self) -> bool;
}

/// API version information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ApiVersion {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch version
    pub patch: u32,
}

impl ApiVersion {
    /// Create a new version
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with another
    #[must_use]
    pub const fn is_compatible(&self, other: &Self) -> bool {
        self.major == other.major && self.minor >= other.minor
    }

    /// Check if this version is newer than another
    #[must_use]
    pub const fn is_newer_than(&self, other: &Self) -> bool {
        if self.major != other.major {
            return self.major > other.major;
        }
        if self.minor != other.minor {
            return self.minor > other.minor;
        }
        self.patch > other.patch
    }
}

impl fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Trait for versioned interfaces
pub trait VersionedInterface {
    /// Get the interface version
    fn version(&self) -> ApiVersion;

    /// Check compatibility with another version
    fn is_compatible_with(&self, version: &ApiVersion) -> bool {
        self.version().is_compatible(version)
    }

    /// Get minimum required version
    fn minimum_version(&self) -> ApiVersion;

    /// Get deprecated features
    fn deprecated_features(&self) -> Vec<String>;
}

/// Module interface descriptor
pub trait ModuleInterface:
    Identifiable + Configurable + ModuleCapability + VersionedInterface
{
    /// Initialize the module
    fn initialize(&mut self) -> CoreResult<()>;

    /// Shutdown the module
    fn shutdown(&mut self) -> CoreResult<()>;

    /// Get module health status
    fn health_check(&self) -> CoreResult<HealthStatus>;

    /// Get module statistics
    fn statistics(&self) -> HashMap<String, f64>;

    /// Reset module state
    fn reset(&mut self) -> CoreResult<()>;
}

/// Health status for modules
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    /// Module is healthy and operational
    Healthy,
    /// Module is degraded but functional
    Degraded(String),
    /// Module is unhealthy
    Unhealthy(String),
    /// Health status unknown
    Unknown,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded(msg) => write!(f, "degraded: {msg}"),
            HealthStatus::Unhealthy(msg) => write!(f, "unhealthy: {msg}"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Resource usage information
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// CPU usage percentage (0-100)
    pub cpu_percent: f32,
    /// GPU memory usage in bytes
    pub gpu_memory_bytes: usize,
    /// Number of threads in use
    pub thread_count: usize,
    /// Number of file handles open
    pub file_handles: usize,
    /// Custom resource metrics
    pub custom: HashMap<String, f64>,
}

/// Trait for resource-aware components
pub trait ResourceAware {
    /// Get current resource usage
    fn resource_usage(&self) -> ResourceUsage;

    /// Get resource limits
    fn resource_limits(&self) -> ResourceUsage;

    /// Set resource limits
    fn set_resource_limits(&mut self, limits: ResourceUsage) -> CoreResult<()>;

    /// Check if resource usage is within limits
    fn within_limits(&self) -> bool {
        let usage = self.resource_usage();
        let limits = self.resource_limits();

        if limits.memory_bytes > 0 && usage.memory_bytes > limits.memory_bytes {
            return false;
        }
        if limits.cpu_percent > 0.0 && usage.cpu_percent > limits.cpu_percent {
            return false;
        }
        if limits.gpu_memory_bytes > 0 && usage.gpu_memory_bytes > limits.gpu_memory_bytes {
            return false;
        }
        if limits.thread_count > 0 && usage.thread_count > limits.thread_count {
            return false;
        }
        true
    }

    /// Release unused resources
    fn release_resources(&mut self) -> CoreResult<usize>;

    /// Estimate resources needed for an operation
    fn estimate_resources(&self, operation: &str, input_size: usize) -> ResourceUsage;
}

/// Diagnostic information
#[derive(Debug, Clone)]
pub struct DiagnosticInfo {
    /// Component name
    pub component: String,
    /// Diagnostic level
    pub level: DiagnosticLevel,
    /// Message
    pub message: String,
    /// Additional context
    pub context: HashMap<String, String>,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Diagnostic level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticLevel {
    /// Trace-level diagnostics
    Trace,
    /// Debug-level diagnostics
    Debug,
    /// Informational diagnostics
    Info,
    /// Warning diagnostics
    Warning,
    /// Error diagnostics
    Error,
}

impl fmt::Display for DiagnosticLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticLevel::Trace => write!(f, "TRACE"),
            DiagnosticLevel::Debug => write!(f, "DEBUG"),
            DiagnosticLevel::Info => write!(f, "INFO"),
            DiagnosticLevel::Warning => write!(f, "WARN"),
            DiagnosticLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// Trait for diagnosable components
pub trait Diagnosable {
    /// Get diagnostics
    fn diagnostics(&self) -> Vec<DiagnosticInfo>;

    /// Get diagnostics at or above a level
    fn diagnostics_at_level(&self, min_level: DiagnosticLevel) -> Vec<DiagnosticInfo> {
        self.diagnostics()
            .into_iter()
            .filter(|d| d.level >= min_level)
            .collect()
    }

    /// Clear diagnostics
    fn clear_diagnostics(&mut self);

    /// Enable diagnostic collection
    fn enable_diagnostics(&mut self, enabled: bool);

    /// Check if diagnostics are enabled
    fn diagnostics_enabled(&self) -> bool;

    /// Get diagnostic summary
    fn diagnostic_summary(&self) -> String {
        let diags = self.diagnostics();
        let errors = diags
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .count();
        let warnings = diags
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Warning)
            .count();
        format!(
            "{} diagnostics ({} errors, {} warnings)",
            diags.len(),
            errors,
            warnings
        )
    }
}

/// Trait for composable components
pub trait Composable {
    /// Compose with another component
    fn compose<Other: Composable>(&self, other: &Other) -> CoreResult<Box<dyn Any>>;

    /// Check if composition is possible
    fn can_compose_with<Other: Composable>(&self, other: &Other) -> bool;

    /// Get composition requirements
    fn composition_requirements(&self) -> Vec<String>;

    /// Get composition outputs
    fn composition_outputs(&self) -> Vec<String>;
}

/// Pipeline stage trait
pub trait PipelineStage<Input, Output> {
    /// Process input and produce output
    fn process(&self, input: Input) -> CoreResult<Output>;

    /// Get stage name
    fn stage_name(&self) -> &str;

    /// Check if stage is ready
    fn is_ready(&self) -> bool;

    /// Get estimated processing time in milliseconds
    fn estimated_time_ms(&self, input_size: usize) -> f64;
}

/// Trait for chainable operations
pub trait Chainable<T> {
    /// Chain with another operation
    fn chain<F, U>(self, f: F) -> Chain<Self, F>
    where
        Self: Sized,
        F: Fn(T) -> U;

    /// Map over values
    fn map<F, U>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
        F: Fn(T) -> U;

    /// Filter values
    fn filter<F>(self, f: F) -> Filter<Self, F>
    where
        Self: Sized,
        F: Fn(&T) -> bool;
}

/// Chain combinator
#[derive(Debug)]
pub struct Chain<A, F> {
    inner: A,
    f: F,
}

impl<A, F> Chain<A, F> {
    /// Create a new chain
    #[must_use]
    pub const fn new(inner: A, f: F) -> Self {
        Self { inner, f }
    }

    /// Get the inner value
    #[must_use]
    pub const fn inner(&self) -> &A {
        &self.inner
    }
}

/// Map combinator
#[derive(Debug)]
pub struct Map<A, F> {
    inner: A,
    f: F,
}

impl<A, F> Map<A, F> {
    /// Create a new map
    #[must_use]
    pub const fn new(inner: A, f: F) -> Self {
        Self { inner, f }
    }
}

/// Filter combinator
#[derive(Debug)]
pub struct Filter<A, F> {
    inner: A,
    predicate: F,
}

impl<A, F> Filter<A, F> {
    /// Create a new filter
    #[must_use]
    pub const fn new(inner: A, predicate: F) -> Self {
        Self { inner, predicate }
    }
}

/// Observer trait for reactive patterns
pub trait Observer<T> {
    /// Called when a new value is available
    fn on_next(&mut self, value: T);

    /// Called when an error occurs
    fn on_error(&mut self, error: CoreError);

    /// Called when the stream completes
    fn on_complete(&mut self);
}

/// Observable trait for reactive patterns
pub trait Observable<T> {
    /// Subscribe an observer
    fn subscribe(&mut self, observer: Box<dyn Observer<T> + Send>);

    /// Unsubscribe an observer
    fn unsubscribe(&mut self, observer_id: usize);

    /// Get number of subscribers
    fn subscriber_count(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_version() {
        let v1 = ApiVersion::new(1, 2, 3);
        let v2 = ApiVersion::new(1, 3, 0);
        let v3 = ApiVersion::new(2, 0, 0);

        assert!(v2.is_compatible(&v1));
        assert!(!v1.is_compatible(&v2));
        assert!(!v3.is_compatible(&v1));

        assert!(v2.is_newer_than(&v1));
        assert!(v3.is_newer_than(&v2));
    }

    #[test]
    fn test_version_display() {
        let v = ApiVersion::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_capability_display() {
        assert_eq!(Capability::Parallel.to_string(), "parallel");
        assert_eq!(Capability::GpuAcceleration.to_string(), "gpu");
        assert_eq!(Capability::Custom("test").to_string(), "custom:test");
    }

    #[test]
    fn test_health_status() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert!(HealthStatus::Degraded("slow".into())
            .to_string()
            .contains("slow"));
        assert!(HealthStatus::Unhealthy("failed".into())
            .to_string()
            .contains("failed"));
    }

    #[test]
    fn test_resource_usage() {
        let usage = ResourceUsage {
            memory_bytes: 1024,
            cpu_percent: 50.0,
            thread_count: 4,
            ..Default::default()
        };

        assert_eq!(usage.memory_bytes, 1024);
        assert_eq!(usage.cpu_percent, 50.0);
        assert_eq!(usage.thread_count, 4);
    }

    #[test]
    fn test_diagnostic_level_ordering() {
        assert!(DiagnosticLevel::Error > DiagnosticLevel::Warning);
        assert!(DiagnosticLevel::Warning > DiagnosticLevel::Info);
        assert!(DiagnosticLevel::Info > DiagnosticLevel::Debug);
        assert!(DiagnosticLevel::Debug > DiagnosticLevel::Trace);
    }

    #[test]
    fn test_diagnostic_info() {
        let info = DiagnosticInfo {
            component: "test".into(),
            level: DiagnosticLevel::Info,
            message: "test message".into(),
            context: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
        };

        assert_eq!(info.component, "test");
        assert_eq!(info.level, DiagnosticLevel::Info);
    }

    #[test]
    fn test_chain_combinator() {
        let chain = Chain::new(42, |x: i32| x * 2);
        assert_eq!(*chain.inner(), 42);
    }

    #[test]
    fn test_serialization_format() {
        let format = SerializationFormat::Json;
        assert_eq!(format, SerializationFormat::Json);
    }
}
