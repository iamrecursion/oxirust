//! Bandwidth-aware transcoding adaptation controller.
//!
//! Wires a [`BandwidthTrigger`] to a user-supplied callback so that callers
//! can react to quality changes without polling the trigger themselves.
//!
//! ## Design
//!
//! ```text
//!  external source ──► BandwidthAdaptationController::update(obs)
//!                              │
//!                     BandwidthTrigger::evaluate()
//!                              │
//!                     ┌────────┴────────────────┐
//!                     │ Hold → no callback       │
//!                     │ Downgrade/Upgrade → cb() │
//!                     └──────────────────────────┘
//! ```
//!
//! The callback is `Box<dyn Fn(TriggerAction) + Send>` so it can be moved
//! across thread boundaries.  The actual codec API call (e.g. changing bitrate)
//! is the caller's responsibility.

use crate::bandwidth_trigger::{
    BandwidthObservation, BandwidthTrigger, TriggerAction, TriggerConfig,
};
use crate::error::NetResult;

/// Wires a [`BandwidthTrigger`] to a callback that receives [`TriggerAction`]
/// events on quality changes.
///
/// Feed bandwidth observations via [`update`](Self::update).  After each
/// observation the trigger is evaluated; if the recommended action is not
/// [`TriggerAction::Hold`] the callback is invoked with the action.
///
/// # Thread Safety
///
/// The callback must be `Send` (required by the bound) but the controller
/// itself is not `Sync`.  Wrap in `Mutex` / `RwLock` for concurrent access.
///
/// # Example
///
/// ```rust
/// use oximedia_net::bandwidth_adaptation::BandwidthAdaptationController;
/// use oximedia_net::bandwidth_trigger::{BandwidthObservation, TriggerAction, TriggerConfig};
/// use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
///
/// let downgrades = Arc::new(AtomicUsize::new(0));
/// let downgrades_clone = Arc::clone(&downgrades);
///
/// let config = TriggerConfig::default();
/// let mut ctrl = BandwidthAdaptationController::new(config, move |action| {
///     if matches!(action, TriggerAction::Downgrade { .. }) {
///         downgrades_clone.fetch_add(1, Ordering::Relaxed);
///     }
/// }).expect("valid config");
///
/// ctrl.update(BandwidthObservation::new(100_000.0)); // very low bandwidth
/// ```
pub struct BandwidthAdaptationController {
    trigger: BandwidthTrigger,
    callback: Box<dyn Fn(TriggerAction) + Send>,
}

impl BandwidthAdaptationController {
    /// Create a new controller.
    ///
    /// # Arguments
    ///
    /// * `config`   - [`TriggerConfig`] that drives the underlying
    ///                [`BandwidthTrigger`].  Must pass
    ///                [`TriggerConfig::validate`](TriggerConfig::validate).
    /// * `callback` - Closure invoked whenever a [`TriggerAction::Downgrade`]
    ///                or [`TriggerAction::Upgrade`] is emitted.
    ///                [`TriggerAction::Hold`] never triggers the callback.
    ///
    /// # Errors
    ///
    /// Returns [`NetError`](crate::error::NetError) if `config.validate()`
    /// fails (e.g. empty tier list, zero bitrate, invalid safety factor).
    pub fn new(
        config: TriggerConfig,
        callback: impl Fn(TriggerAction) + Send + 'static,
    ) -> NetResult<Self> {
        let trigger = BandwidthTrigger::new(config)?;
        Ok(Self {
            trigger,
            callback: Box::new(callback),
        })
    }

    /// Feed a bandwidth observation and fire the callback if the action is not
    /// [`TriggerAction::Hold`].
    ///
    /// This method:
    /// 1. Forwards `obs` to the underlying [`BandwidthTrigger`].
    /// 2. Evaluates the trigger.
    /// 3. Invokes the callback for `Downgrade` or `Upgrade` actions.
    pub fn update(&mut self, obs: BandwidthObservation) {
        self.trigger.add_observation(obs);
        let action = self.trigger.evaluate();
        match &action {
            TriggerAction::Hold => {} // no callback on Hold
            _ => (self.callback)(action),
        }
    }

    /// Returns the current EMA bandwidth estimate in bps.
    #[must_use]
    pub fn ema_bps(&self) -> f64 {
        self.trigger.ema_bps()
    }

    /// Returns the current quality tier index.
    #[must_use]
    pub fn current_tier(&self) -> usize {
        self.trigger.current_tier()
    }

    /// Resets the underlying trigger state (EMA, tier, history, timers).
    pub fn reset(&mut self) {
        self.trigger.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    fn make_low_bandwidth_config() -> TriggerConfig {
        // Use a zero upgrade_hold and tight safety_factor so the trigger reacts
        // quickly in tests.
        TriggerConfig {
            upgrade_hold: std::time::Duration::ZERO,
            ..TriggerConfig::default()
        }
    }

    // Feed `n` identical observations and return total callback invocations.
    fn feed_bps_and_count(ctrl: &mut BandwidthAdaptationController, bps: f64, n: usize) -> usize {
        let counter = Arc::new(AtomicUsize::new(0));
        // We can't easily swap the closure here; instead we use the separate
        // test-specific constructor.  This test uses the Arc pattern inline.
        for _ in 0..n {
            ctrl.update(BandwidthObservation::new(bps));
        }
        // The counter is embedded in the controller's callback; use
        // trigger history length as a proxy.
        counter.load(Ordering::Relaxed)
    }

    /// Callback fires with `Downgrade` when bandwidth drops below the current
    /// tier's threshold.
    #[test]
    fn test_controller_fires_on_downgrade() {
        let fired = Arc::new(AtomicUsize::new(0));
        let fired_clone = Arc::clone(&fired);

        let config = TriggerConfig::default();
        let mut ctrl = BandwidthAdaptationController::new(config, move |action| {
            if matches!(action, TriggerAction::Downgrade { .. }) {
                fired_clone.fetch_add(1, Ordering::Relaxed);
            }
        })
        .expect("valid config");

        // Start at the top tier (1080p, 5 Mbps) then drop bandwidth to 100 kbps.
        ctrl.trigger.force_tier(3).expect("tier 3 exists");

        for _ in 0..15 {
            ctrl.update(BandwidthObservation::new(100_000.0));
        }

        assert!(
            fired.load(Ordering::Relaxed) > 0,
            "callback should have fired at least once with Downgrade"
        );
    }

    /// Callback is NOT invoked on `Hold` — stable bandwidth keeps it quiet.
    #[test]
    fn test_controller_no_callback_on_hold() {
        let fired = Arc::new(AtomicUsize::new(0));
        let fired_clone = Arc::clone(&fired);

        // 100 s upgrade_hold → upgrade will never fire during the test window.
        let config = TriggerConfig {
            upgrade_hold: std::time::Duration::from_secs(100),
            ..TriggerConfig::default()
        };

        let mut ctrl = BandwidthAdaptationController::new(config, move |_action| {
            fired_clone.fetch_add(1, Ordering::Relaxed);
        })
        .expect("valid config");

        // Tier 0 (240p, 400 kbps); feed bandwidth just above its threshold so
        // there's no downgrade (already at the floor) and no upgrade (hold not met).
        // 240p * 1.25 = 500 kbps; feed 501 kbps — just above upgrade threshold
        // but hold period is 100 s, so no upgrade either.
        for _ in 0..10 {
            ctrl.update(BandwidthObservation::new(501_000.0));
        }

        assert_eq!(
            fired.load(Ordering::Relaxed),
            0,
            "callback must not fire on Hold"
        );
    }

    /// Callback fires with `Upgrade` after high-bandwidth samples accumulate for
    /// the configured hold period (here: `Duration::ZERO`).
    #[test]
    fn test_controller_fires_on_upgrade() {
        let upgrades = Arc::new(AtomicUsize::new(0));
        let upgrades_clone = Arc::clone(&upgrades);

        let config = make_low_bandwidth_config(); // upgrade_hold = ZERO
        let mut ctrl = BandwidthAdaptationController::new(config, move |action| {
            if matches!(action, TriggerAction::Upgrade { .. }) {
                upgrades_clone.fetch_add(1, Ordering::Relaxed);
            }
        })
        .expect("valid config");

        // Feed massive bandwidth — well above every tier * 1.25.
        for _ in 0..20 {
            ctrl.update(BandwidthObservation::new(50_000_000.0));
        }

        assert!(
            upgrades.load(Ordering::Relaxed) > 0,
            "callback should have fired at least once with Upgrade"
        );
    }

    /// `BandwidthAdaptationController::new` returns an error for an invalid config.
    #[test]
    fn test_controller_invalid_config_returns_error() {
        let mut config = TriggerConfig::default();
        config.tiers.clear(); // no tiers → invalid
        let result = BandwidthAdaptationController::new(config, |_| {});
        assert!(result.is_err(), "empty tier list should be rejected");
    }

    /// `reset()` clears the trigger state so subsequent updates start fresh.
    #[test]
    fn test_controller_reset_clears_state() {
        let fired = Arc::new(AtomicUsize::new(0));
        let fired_clone = Arc::clone(&fired);

        let config = make_low_bandwidth_config();
        let mut ctrl = BandwidthAdaptationController::new(config, move |_| {
            fired_clone.fetch_add(1, Ordering::Relaxed);
        })
        .expect("valid config");

        // Drive to a non-zero EMA.
        ctrl.update(BandwidthObservation::new(5_000_000.0));
        assert!(ctrl.ema_bps() > 0.0);

        ctrl.reset();
        assert_eq!(ctrl.ema_bps(), 0.0);
        assert_eq!(ctrl.current_tier(), 0);
    }
}
