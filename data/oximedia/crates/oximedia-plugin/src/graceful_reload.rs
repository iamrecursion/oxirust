//! Graceful plugin reload with old-plugin fallback serving.
//!
//! During a plugin reload, there is a window between "new plugin is being
//! initialised" and "new plugin is ready".  Naive approaches drop the old
//! plugin immediately, causing in-flight requests to fail.
//!
//! This module extends the basic [`crate::hot_reload::GracefulReload`] pattern from
//! `hot_reload` with a *double-buffered* swap strategy:
//!
//! 1. The **old** plugin remains active and continues handling requests.
//! 2. The **new** plugin is initialised in parallel (or sequentially with a
//!    configurable init timeout).
//! 3. Only when the new plugin signals readiness (via [`InitState`]) is the
//!    old plugin retired.
//!
//! The public API is intentionally synchronous / pure-Rust so it can be
//! tested without any OS threading primitives.

use crate::error::{PluginError, PluginResult};
use crate::traits::CodecPlugin;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// ── InitState ────────────────────────────────────────────────────────────────

/// Lifecycle state of a new plugin during initialisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitState {
    /// Initialisation has not yet started.
    Pending,
    /// Initialisation is in progress.
    InProgress,
    /// Initialisation completed successfully; the plugin is ready to serve.
    Ready,
    /// Initialisation failed; the old plugin should continue serving.
    Failed,
}

// ── SwappablePlugin ──────────────────────────────────────────────────────────

/// An atomically-swappable plugin slot that keeps a *fallback* plugin active
/// while a new plugin is being prepared.
///
/// All codec dispatch goes through [`SwappablePlugin::current`], which always
/// returns the best-available plugin:
///
/// - When no new plugin is initialising: returns the primary plugin.
/// - While a new plugin is [`InitState::InProgress`]: returns the **old**
///   (fallback) plugin so service continues uninterrupted.
/// - Once the new plugin reaches [`InitState::Ready`]: atomically promotes it
///   to primary and retires the old plugin.
pub struct SwappablePlugin {
    /// Currently active plugin (serves all requests).
    primary: RwLock<Arc<dyn CodecPlugin>>,
    /// Pending replacement plugin (may be `None` if no reload is in flight).
    pending: RwLock<Option<Arc<dyn CodecPlugin>>>,
    /// State of the pending plugin's initialisation.
    pending_state: RwLock<InitState>,
    /// When the pending initialisation began.
    init_started_at: RwLock<Option<Instant>>,
    /// Maximum time allowed for the new plugin to reach `Ready`.
    init_timeout: Duration,
}

impl SwappablePlugin {
    /// Create a new swappable slot backed by `initial`.
    pub fn new(initial: Arc<dyn CodecPlugin>, init_timeout: Duration) -> Self {
        Self {
            primary: RwLock::new(initial),
            pending: RwLock::new(None),
            pending_state: RwLock::new(InitState::Pending),
            init_started_at: RwLock::new(None),
            init_timeout,
        }
    }

    /// Return the plugin that should service the next request.
    ///
    /// If a new plugin is being initialised and its timeout has not elapsed,
    /// the current primary (old) plugin is returned.  Once init completes
    /// (state set to [`InitState::Ready`]) via [`Self::complete_init`], the
    /// new plugin is promoted automatically.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::InitFailed`] if the lock is poisoned.
    pub fn current(&self) -> PluginResult<Arc<dyn CodecPlugin>> {
        // Check whether init timed out and roll back.
        self.maybe_timeout_pending()?;

        let state = self
            .pending_state
            .read()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        match *state {
            InitState::Ready => {
                // Pending is ready — perform the atomic swap now.
                drop(state);
                self.promote_pending()?;
            }
            InitState::InProgress | InitState::Pending | InitState::Failed => {
                // Serve from the primary (old) plugin.
            }
        }

        let primary = self
            .primary
            .read()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        Ok(Arc::clone(&*primary))
    }

    /// Begin initialising a new plugin candidate.
    ///
    /// The `pending` slot is filled with the candidate and the state is set to
    /// [`InitState::InProgress`].  The **primary plugin remains active** and
    /// will continue serving requests until [`Self::complete_init`] is called.
    ///
    /// Returns an error if another init is already in progress.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::InitFailed`] if an init is already in progress
    /// or if a lock is poisoned.
    pub fn begin_init(&self, candidate: Arc<dyn CodecPlugin>) -> PluginResult<()> {
        let state = self
            .pending_state
            .read()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        if *state == InitState::InProgress {
            return Err(PluginError::InitFailed(
                "A plugin initialisation is already in progress".to_string(),
            ));
        }
        drop(state);

        let mut pending = self
            .pending
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        let mut pstate = self
            .pending_state
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        let mut started = self
            .init_started_at
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        *pending = Some(candidate);
        *pstate = InitState::InProgress;
        *started = Some(Instant::now());
        Ok(())
    }

    /// Signal that the pending plugin has finished initialising.
    ///
    /// - On success (`ok = true`) the state is set to [`InitState::Ready`].
    ///   The next call to [`Self::current`] will atomically promote the pending
    ///   plugin to primary.
    /// - On failure (`ok = false`) the state is set to [`InitState::Failed`]
    ///   and the pending slot is cleared; the old primary continues serving.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::InitFailed`] if no init is in progress, or if
    /// a lock is poisoned.
    pub fn complete_init(&self, ok: bool) -> PluginResult<()> {
        let mut pstate = self
            .pending_state
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        if *pstate != InitState::InProgress {
            return Err(PluginError::InitFailed(
                "complete_init called but no init is in progress".to_string(),
            ));
        }

        if ok {
            *pstate = InitState::Ready;
        } else {
            *pstate = InitState::Failed;
            drop(pstate);
            self.clear_pending()?;
        }
        Ok(())
    }

    /// Force-promote the pending plugin to primary regardless of state.
    ///
    /// Useful in tests or emergency overrides.  Clears the pending slot.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::NotFound`] if there is no pending plugin.
    pub fn force_promote(&self) -> PluginResult<()> {
        let mut pending = self
            .pending
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        let candidate = pending
            .take()
            .ok_or_else(|| PluginError::NotFound("No pending plugin to promote".to_string()))?;

        let mut primary = self
            .primary
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        *primary = candidate;

        let mut pstate = self
            .pending_state
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        *pstate = InitState::Pending;

        let mut started = self
            .init_started_at
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        *started = None;
        Ok(())
    }

    /// Return the current [`InitState`] of the pending plugin.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    pub fn pending_state(&self) -> PluginResult<InitState> {
        self.pending_state
            .read()
            .map(|g| *g)
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))
    }

    /// Atomically swap the pending plugin into the primary slot.
    fn promote_pending(&self) -> PluginResult<()> {
        let mut pending = self
            .pending
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        if let Some(candidate) = pending.take() {
            let mut primary = self
                .primary
                .write()
                .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
            *primary = candidate;
        }

        let mut pstate = self
            .pending_state
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        *pstate = InitState::Pending;

        let mut started = self
            .init_started_at
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        *started = None;
        Ok(())
    }

    /// If the init deadline has elapsed, transition to `Failed` and clear.
    fn maybe_timeout_pending(&self) -> PluginResult<()> {
        let state = self
            .pending_state
            .read()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        if *state != InitState::InProgress {
            return Ok(());
        }
        drop(state);

        let started = self
            .init_started_at
            .read()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        if let Some(start) = *started {
            if start.elapsed() > self.init_timeout {
                drop(started);
                let mut pstate = self
                    .pending_state
                    .write()
                    .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
                *pstate = InitState::Failed;
                drop(pstate);
                self.clear_pending()?;
                tracing::warn!(
                    "Plugin initialisation timed out after {:?}; retaining old plugin",
                    self.init_timeout
                );
            }
        }
        Ok(())
    }

    /// Clear the pending slot.
    fn clear_pending(&self) -> PluginResult<()> {
        let mut pending = self
            .pending
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        *pending = None;

        let mut started = self
            .init_started_at
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
        *started = None;
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_plugin::StaticPlugin;
    use crate::traits::{CodecPluginInfo, PLUGIN_API_VERSION};

    fn make_plugin(name: &str) -> Arc<dyn CodecPlugin> {
        let info = CodecPluginInfo {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            author: "Test".to_string(),
            description: "Test plugin".to_string(),
            api_version: PLUGIN_API_VERSION,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };
        Arc::new(StaticPlugin::new(info))
    }

    // 1. current() returns primary when no init is in flight.
    #[test]
    fn test_current_returns_primary() {
        let old = make_plugin("old");
        let slot = SwappablePlugin::new(Arc::clone(&old), Duration::from_secs(5));
        let current = slot.current().expect("current");
        assert_eq!(current.info().name, "old");
    }

    // 2. begin_init sets state to InProgress.
    #[test]
    fn test_begin_init_sets_in_progress() {
        let old = make_plugin("old");
        let slot = SwappablePlugin::new(Arc::clone(&old), Duration::from_secs(5));
        slot.begin_init(make_plugin("new")).expect("begin_init");
        assert_eq!(slot.pending_state().expect("state"), InitState::InProgress);
    }

    // 3. While InProgress, current() still returns old primary.
    #[test]
    fn test_in_progress_serves_old() {
        let old = make_plugin("old");
        let slot = SwappablePlugin::new(Arc::clone(&old), Duration::from_secs(5));
        slot.begin_init(make_plugin("new")).expect("begin_init");
        let current = slot.current().expect("current");
        assert_eq!(current.info().name, "old");
    }

    // 4. After complete_init(true), current() promotes new plugin.
    #[test]
    fn test_complete_init_ok_promotes() {
        let old = make_plugin("old");
        let slot = SwappablePlugin::new(Arc::clone(&old), Duration::from_secs(5));
        slot.begin_init(make_plugin("new")).expect("begin_init");
        slot.complete_init(true).expect("complete");
        let current = slot.current().expect("current");
        assert_eq!(current.info().name, "new");
    }

    // 5. After complete_init(false), primary stays old.
    #[test]
    fn test_complete_init_failure_keeps_old() {
        let old = make_plugin("old");
        let slot = SwappablePlugin::new(Arc::clone(&old), Duration::from_secs(5));
        slot.begin_init(make_plugin("new")).expect("begin_init");
        slot.complete_init(false).expect("complete");
        let current = slot.current().expect("current");
        assert_eq!(current.info().name, "old");
    }

    // 6. begin_init while InProgress returns an error.
    #[test]
    fn test_double_begin_init_rejected() {
        let old = make_plugin("old");
        let slot = SwappablePlugin::new(Arc::clone(&old), Duration::from_secs(5));
        slot.begin_init(make_plugin("new-a")).expect("first begin");
        let err = slot.begin_init(make_plugin("new-b"));
        assert!(err.is_err());
    }

    // 7. complete_init without begin_init returns an error.
    #[test]
    fn test_complete_without_begin_fails() {
        let old = make_plugin("old");
        let slot = SwappablePlugin::new(Arc::clone(&old), Duration::from_secs(5));
        assert!(slot.complete_init(true).is_err());
    }

    // 8. force_promote swaps immediately.
    #[test]
    fn test_force_promote() {
        let old = make_plugin("old");
        let slot = SwappablePlugin::new(Arc::clone(&old), Duration::from_secs(5));
        slot.begin_init(make_plugin("new")).expect("begin_init");
        slot.force_promote().expect("force");
        assert_eq!(slot.current().expect("current").info().name, "new");
    }

    // 9. force_promote without pending plugin returns NotFound.
    #[test]
    fn test_force_promote_no_pending_fails() {
        let old = make_plugin("old");
        let slot = SwappablePlugin::new(Arc::clone(&old), Duration::from_secs(5));
        assert!(matches!(
            slot.force_promote(),
            Err(PluginError::NotFound(_))
        ));
    }

    // 10. Timeout: init that exceeds the deadline falls back to old.
    #[test]
    fn test_init_timeout_falls_back() {
        let old = make_plugin("old");
        // Use a very short timeout.
        let slot = SwappablePlugin::new(Arc::clone(&old), Duration::from_millis(1));
        slot.begin_init(make_plugin("new")).expect("begin_init");
        // Wait long enough for timeout to trigger.
        std::thread::sleep(Duration::from_millis(5));
        let current = slot.current().expect("current after timeout");
        assert_eq!(current.info().name, "old");
        assert_eq!(slot.pending_state().expect("state"), InitState::Failed);
    }
}
