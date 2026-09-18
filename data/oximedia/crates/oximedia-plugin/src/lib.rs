//! Plugin system for OxiMedia.
//!
//! Enables dynamic loading of external codec implementations,
//! allowing third-party or patent-encumbered codecs to be used
//! without bundling them in the core library.
//!
//! # Architecture
//!
//! Plugins implement the [`CodecPlugin`] trait and are loaded from
//! shared libraries (.so/.dylib/.dll) at runtime. Each plugin
//! declares its capabilities (which codecs it provides, whether
//! it can decode/encode), and the central [`PluginRegistry`] manages
//! discovery, loading, and codec lookup.
//!
//! # Feature Gates
//!
//! - `dynamic-loading`: Enables loading plugins from shared libraries
//!   (requires libloading). Without this feature, only static plugin
//!   registration is available.
//!
//! # Static Plugins
//!
//! Even without dynamic loading, you can register plugins statically
//! using [`StaticPlugin`] and the builder pattern:
//!
//! ```rust
//! use oximedia_plugin::{StaticPlugin, CodecPluginInfo, PluginCapability, PluginRegistry};
//! use std::sync::Arc;
//! use std::collections::HashMap;
//!
//! let info = CodecPluginInfo {
//!     name: "my-plugin".to_string(),
//!     version: "1.0.0".to_string(),
//!     author: "Test".to_string(),
//!     description: "A test plugin".to_string(),
//!     api_version: oximedia_plugin::PLUGIN_API_VERSION,
//!     license: "MIT".to_string(),
//!     patent_encumbered: false,
//! };
//!
//! let plugin = StaticPlugin::new(info)
//!     .add_capability(PluginCapability {
//!         codec_name: "test-codec".to_string(),
//!         can_decode: true,
//!         can_encode: false,
//!         pixel_formats: vec!["yuv420p".to_string()],
//!         properties: HashMap::new(),
//!     });
//!
//! let registry = PluginRegistry::new();
//! registry.register(Arc::new(plugin)).expect("registration should succeed");
//! assert_eq!(registry.plugin_count(), 1);
//! ```
//!
//! # Dynamic Plugins (feature = "dynamic-loading")
//!
//! With the `dynamic-loading` feature, plugins can be loaded from
//! shared libraries. The shared library must export two symbols:
//!
//! - `oximedia_plugin_api_version() -> u32`
//! - `oximedia_plugin_create() -> *mut dyn CodecPlugin`
//!
//! Use the [`declare_plugin!`] macro to generate these exports.
//!
//! Loading a shared library executes arbitrary code with no sandboxing,
//! so `loader::LoadedPlugin::load_with_digest` (and
//! [`PluginRegistry::load_plugin_with_digest`]) should be preferred over
//! the unchecked `load`/`load_plugin` entry points whenever the expected
//! SHA-256 digest of the plugin file is known ahead of time. See the
//! `loader` module documentation (available with the `dynamic-loading`
//! feature) for the full trust model.

pub mod capability;
pub mod config_persist;
pub mod config_persistence;
pub mod error;
pub mod filter_plugin;
pub mod graceful_reload;
pub mod harness;
pub mod health;
pub mod health_check;
pub mod health_monitor;
pub mod hot_reload;
pub mod lazy;
pub mod lazy_init;
pub mod manifest;
pub mod plugin_config;
pub mod plugin_telemetry;
pub mod pool;
pub mod priority;
pub mod registry;
pub mod resources;
pub mod sandbox;
pub mod static_plugin;
pub mod traits;
pub mod version_resolver;

#[cfg(feature = "dynamic-loading")]
pub mod loader;

pub use error::{PluginError, PluginResult};
pub use hot_reload::{
    compute_hash, GracefulReload, HotReloadManager, PluginLifecycle, PluginVersion, ReloadPolicy,
    WatchEntry,
};
pub use manifest::{
    resolve_dependencies, DependencyResolution, ManifestCodec, PluginManifest, SemVer, SemVerOp,
    SemVerReq,
};
pub use registry::PluginRegistry;
pub use sandbox::{
    PermissionSet, PluginSandbox, SandboxConfig, SandboxContext, SandboxError, PERM_AUDIO,
    PERM_FILESYSTEM, PERM_GPU, PERM_MEMORY_LARGE, PERM_NETWORK, PERM_VIDEO,
};
pub use static_plugin::StaticPlugin;
pub use traits::{CodecPlugin, CodecPluginInfo, PluginCapability, PLUGIN_API_VERSION};
pub use version_resolver::{
    DependencyResolver, PluginDependency, ResolveError, SemVer as ResolverSemVer, VersionConstraint,
};
