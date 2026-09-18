//! Lazy plugin initialization — defer codec creation until first use.
//!
//! Creating a codec plugin can be expensive: it may load large lookup tables,
//! allocate GPU resources, or warm up a neural network.  With lazy
//! initialisation the host pays that cost only when the first decode/encode
//! request arrives, not at registration time.
//!
//! # Design
//!
//! [`LazyPlugin`] wraps a *factory closure* that produces the real plugin on
//! first access.  The inner plugin is stored behind an `RwLock<Option<…>>`.
//!
//! - First call to any delegating method acquires a write lock, invokes the
//!   factory, stores the result, then falls through to the real implementation.
//! - Subsequent calls acquire only a read lock — concurrent access has minimal
//!   contention once initialised.
//!
//! A [`LazyPlugin`] also exposes [`LazyPlugin::is_initialised`] so callers can
//! check initialisation state without triggering it.

use crate::error::{PluginError, PluginResult};
use crate::traits::{CodecPlugin, CodecPluginInfo, PluginCapability};
use oximedia_codec::{CodecResult, EncoderConfig, VideoDecoder, VideoEncoder};
use std::sync::{Arc, RwLock};

// ── LazyPlugin ────────────────────────────────────────────────────────────────

/// Shared factory type: `Arc<dyn Fn() -> PluginResult<Arc<dyn CodecPlugin>> + Send + Sync>`.
pub type PluginFactory = Arc<dyn Fn() -> PluginResult<Arc<dyn CodecPlugin>> + Send + Sync>;

/// A lazily-initialised codec plugin.
///
/// The wrapped factory is invoked at most once (on first use) and the result
/// is cached.  All [`CodecPlugin`] methods are forwarded to the inner plugin;
/// calling them before or during initialisation is safe.
///
/// # Example
///
/// ```rust
/// use oximedia_plugin::lazy_init::LazyPlugin;
/// use oximedia_plugin::{StaticPlugin, CodecPluginInfo, PLUGIN_API_VERSION};
/// use std::sync::Arc;
///
/// let lazy = LazyPlugin::new(
///     "example-plugin".to_string(),
///     "1.0.0".to_string(),
///     Arc::new(|| {
///         let info = CodecPluginInfo {
///             name: "example-plugin".to_string(),
///             version: "1.0.0".to_string(),
///             author: "Test".to_string(),
///             description: "Expensive plugin".to_string(),
///             api_version: PLUGIN_API_VERSION,
///             license: "MIT".to_string(),
///             patent_encumbered: false,
///         };
///         Ok(Arc::new(StaticPlugin::new(info)))
///     }),
/// );
/// assert!(!lazy.is_initialised());
/// ```
pub struct LazyPlugin {
    /// Pre-declared plugin name used for [`CodecPlugin::info`] *before* init.
    name: String,
    /// Pre-declared version used for [`CodecPlugin::info`] before init.
    version: String,
    /// Factory that creates the real plugin on first use.
    factory: PluginFactory,
    /// The real plugin, populated on first use.
    inner: RwLock<Option<Arc<dyn CodecPlugin>>>,
}

impl LazyPlugin {
    /// Create a new `LazyPlugin`.
    ///
    /// `name` and `version` are returned from [`CodecPlugin::info`] before
    /// the inner plugin is initialised; once initialised the inner plugin's
    /// `info()` takes over.
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        factory: PluginFactory,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            factory,
            inner: RwLock::new(None),
        }
    }

    /// Return `true` if the inner plugin has already been initialised.
    pub fn is_initialised(&self) -> bool {
        match self.inner.read() {
            Ok(guard) => guard.is_some(),
            Err(_) => false,
        }
    }

    /// Ensure the inner plugin is initialised, triggering the factory if not.
    ///
    /// Returns an `Arc` to the inner plugin.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by the factory, or
    /// [`PluginError::InitFailed`] on lock poisoning.
    pub fn ensure_init(&self) -> PluginResult<Arc<dyn CodecPlugin>> {
        // Fast path: already initialised.
        {
            let guard = self
                .inner
                .read()
                .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
            if let Some(ref p) = *guard {
                return Ok(Arc::clone(p));
            }
        }

        // Slow path: initialise under write lock.
        let mut guard = self
            .inner
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        // Double-check after acquiring write lock.
        if let Some(ref p) = *guard {
            return Ok(Arc::clone(p));
        }

        let plugin = (self.factory)()?;
        *guard = Some(Arc::clone(&plugin));
        Ok(plugin)
    }

    /// Force a re-initialisation, replacing the existing inner plugin.
    ///
    /// Useful after a hot-reload signal.
    ///
    /// # Errors
    ///
    /// Propagates factory errors or lock-poisoning.
    pub fn reinitialise(&self) -> PluginResult<()> {
        let plugin = (self.factory)()?;
        let mut guard = self
            .inner
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        *guard = Some(plugin);
        Ok(())
    }
}

impl CodecPlugin for LazyPlugin {
    fn info(&self) -> CodecPluginInfo {
        // Return the inner plugin's info if available; otherwise a stub.
        if let Ok(guard) = self.inner.read() {
            if let Some(ref p) = *guard {
                return p.info();
            }
        }
        CodecPluginInfo {
            name: self.name.clone(),
            version: self.version.clone(),
            author: "lazy (not yet initialised)".to_string(),
            description: "Plugin not yet initialised".to_string(),
            api_version: crate::traits::PLUGIN_API_VERSION,
            license: "unknown".to_string(),
            patent_encumbered: false,
        }
    }

    fn capabilities(&self) -> Vec<PluginCapability> {
        match self.ensure_init() {
            Ok(p) => p.capabilities(),
            Err(_) => Vec::new(),
        }
    }

    fn create_decoder(&self, codec_name: &str) -> CodecResult<Box<dyn VideoDecoder>> {
        self.ensure_init()
            .map_err(|e| oximedia_codec::CodecError::Internal(e.to_string()))
            .and_then(|p| p.create_decoder(codec_name))
    }

    fn create_encoder(
        &self,
        codec_name: &str,
        config: EncoderConfig,
    ) -> CodecResult<Box<dyn VideoEncoder>> {
        self.ensure_init()
            .map_err(|e| oximedia_codec::CodecError::Internal(e.to_string()))
            .and_then(|p| p.create_encoder(codec_name, config))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_plugin::StaticPlugin;
    use crate::traits::PLUGIN_API_VERSION;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn make_inner(name: &str) -> Arc<dyn CodecPlugin> {
        let info = CodecPluginInfo {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            author: "Test".to_string(),
            description: "Inner plugin".to_string(),
            api_version: PLUGIN_API_VERSION,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };
        Arc::new(StaticPlugin::new(info))
    }

    // 1. is_initialised starts as false.
    #[test]
    fn test_not_initialised_at_start() {
        let lazy = LazyPlugin::new("p", "1.0.0", Arc::new(|| Ok(make_inner("p"))));
        assert!(!lazy.is_initialised());
    }

    // 2. ensure_init triggers the factory once.
    #[test]
    fn test_ensure_init_triggers_factory() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&call_count);
        let lazy = LazyPlugin::new(
            "p",
            "1.0.0",
            Arc::new(move || {
                count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(make_inner("p"))
            }),
        );
        lazy.ensure_init().expect("init");
        lazy.ensure_init().expect("init again");
        assert_eq!(call_count.load(Ordering::SeqCst), 1); // factory called once
    }

    // 3. After ensure_init, is_initialised is true.
    #[test]
    fn test_is_initialised_after_ensure() {
        let lazy = LazyPlugin::new("p", "1.0.0", Arc::new(|| Ok(make_inner("p"))));
        lazy.ensure_init().expect("init");
        assert!(lazy.is_initialised());
    }

    // 4. info() before init returns stub.
    #[test]
    fn test_info_before_init_is_stub() {
        let lazy = LazyPlugin::new("stub-name", "0.0.1", Arc::new(|| Ok(make_inner("real"))));
        let info = lazy.info();
        assert_eq!(info.name, "stub-name");
        assert_eq!(info.version, "0.0.1");
    }

    // 5. info() after init returns inner plugin's info.
    #[test]
    fn test_info_after_init_is_inner() {
        let lazy = LazyPlugin::new("stub", "1.0.0", Arc::new(|| Ok(make_inner("inner-name"))));
        lazy.ensure_init().expect("init");
        assert_eq!(lazy.info().name, "inner-name");
    }

    // 6. Factory error propagates.
    #[test]
    fn test_factory_error_propagates() {
        let lazy = LazyPlugin::new(
            "p",
            "1.0.0",
            Arc::new(|| Err(PluginError::InitFailed("factory failed".to_string()))),
        );
        let err = lazy.ensure_init();
        assert!(err.is_err());
        assert!(!lazy.is_initialised());
    }

    // 7. reinitialise replaces the inner plugin.
    #[test]
    fn test_reinitialise() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);
        let lazy = LazyPlugin::new(
            "p",
            "1.0.0",
            Arc::new(move || {
                cc.fetch_add(1, Ordering::SeqCst);
                Ok(make_inner("p"))
            }),
        );
        lazy.ensure_init().expect("first init");
        lazy.reinitialise().expect("reinit");
        assert_eq!(call_count.load(Ordering::SeqCst), 2); // factory called twice
    }

    // 8. capabilities() triggers init.
    #[test]
    fn test_capabilities_triggers_init() {
        let lazy = LazyPlugin::new("p", "1.0.0", Arc::new(|| Ok(make_inner("p"))));
        let _caps = lazy.capabilities();
        assert!(lazy.is_initialised());
    }

    // 9. create_decoder triggers init and fails gracefully (no factory).
    #[test]
    fn test_create_decoder_triggers_init() {
        let lazy = LazyPlugin::new("p", "1.0.0", Arc::new(|| Ok(make_inner("p"))));
        let result = lazy.create_decoder("h264");
        assert!(lazy.is_initialised());
        // StaticPlugin with no decoder factory returns an error — that's expected.
        assert!(result.is_err());
    }
}
