//! Automated CG/lower-third graphics overlay insertion during playout.
//!
//! This module manages a timed queue of graphics overlay events that the
//! automation system injects at frame-accurate positions during playout.
//! Overlays are described by an [`OverlayDescriptor`] which carries:
//!
//! - A unique event identifier and template name.
//! - The trigger time (milliseconds from the start of the current item).
//! - An optional hold duration (how long the overlay stays on-air before
//!   auto-clearing).
//! - A flat map of substitution variables for the template renderer.
//!
//! The [`GraphicsOverlayScheduler`] maintains a priority queue ordered by
//! trigger time and exposes a simple `due_now` method that playout engines
//! call on every tick to retrieve overlays that should be inserted.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use tracing::{debug, info};

// ─────────────────────────────────────────────────────────────────────────────
// Overlay descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// Category of a CG overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverlayKind {
    /// Lower-third name/title card.
    LowerThird,
    /// Full-screen graphic or slide.
    FullScreen,
    /// Bug / logo insert (typically persistent).
    Bug,
    /// Score/statistics overlay.
    Score,
    /// Clock / countdown timer.
    Clock,
    /// Ticker / crawl text at the bottom of the frame.
    Ticker,
}

impl OverlayKind {
    /// Returns a human-readable label for the overlay kind.
    pub fn label(self) -> &'static str {
        match self {
            Self::LowerThird => "lower-third",
            Self::FullScreen => "full-screen",
            Self::Bug => "bug",
            Self::Score => "score",
            Self::Clock => "clock",
            Self::Ticker => "ticker",
        }
    }
}

/// Description of a single CG overlay event to be inserted during playout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlayDescriptor {
    /// Unique identifier for this overlay event.
    pub id: String,
    /// Kind of overlay.
    pub kind: OverlayKind,
    /// Template name registered in the CG system (e.g. `"lower_third_v2"`).
    pub template: String,
    /// Trigger offset in milliseconds from the start of the current item.
    pub trigger_ms: u64,
    /// How long (ms) to hold the overlay on-air before auto-clearing.
    /// `None` means the overlay stays until explicitly removed.
    pub hold_ms: Option<u64>,
    /// Substitution variables for the template renderer.
    pub variables: HashMap<String, String>,
    /// Channel ID this overlay is bound to.
    pub channel_id: usize,
}

impl OverlayDescriptor {
    /// Create a new overlay descriptor.
    pub fn new(
        id: impl Into<String>,
        kind: OverlayKind,
        template: impl Into<String>,
        trigger_ms: u64,
        channel_id: usize,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            template: template.into(),
            trigger_ms,
            hold_ms: None,
            variables: HashMap::new(),
            channel_id,
        }
    }

    /// Set a hold duration in milliseconds.
    pub fn with_hold(mut self, hold_ms: u64) -> Self {
        self.hold_ms = Some(hold_ms);
        self
    }

    /// Add a template variable.
    pub fn with_variable(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.variables.insert(key.into(), value.into());
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Overlay state
// ─────────────────────────────────────────────────────────────────────────────

/// State of an overlay that has been inserted on-air.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveOverlay {
    /// The descriptor that produced this overlay.
    pub descriptor: OverlayDescriptor,
    /// Absolute time (ms from item start) when the overlay went on-air.
    pub inserted_at_ms: u64,
    /// Absolute time (ms from item start) when the overlay should be cleared.
    /// `None` if no auto-clear is configured.
    pub clear_at_ms: Option<u64>,
}

impl ActiveOverlay {
    /// Returns `true` if the overlay should be auto-cleared at `now_ms`.
    pub fn should_clear(&self, now_ms: u64) -> bool {
        match self.clear_at_ms {
            Some(clear_at) => now_ms >= clear_at,
            None => false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Scheduler for automated CG/lower-third overlay insertion.
///
/// Call [`Self::schedule`] to register overlays and [`Self::due_now`] on every playout
/// tick to drain overlays that have reached their trigger time.
#[derive(Debug, Default)]
pub struct GraphicsOverlayScheduler {
    /// Pending overlays stored in a `BTreeMap` keyed by `(trigger_ms, id)` for
    /// deterministic chronological iteration.
    pending: BTreeMap<(u64, String), OverlayDescriptor>,
    /// Currently active (on-air) overlays.
    active: Vec<ActiveOverlay>,
    /// All-time history of inserted overlays (id → descriptor).
    history: Vec<String>,
}

impl GraphicsOverlayScheduler {
    /// Create a new, empty scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Schedule an overlay event.
    ///
    /// If an overlay with the same `id` already exists in the pending queue it
    /// is replaced.
    pub fn schedule(&mut self, descriptor: OverlayDescriptor) {
        info!(
            "Scheduled {} overlay '{}' at {}ms on channel {}",
            descriptor.kind.label(),
            descriptor.id,
            descriptor.trigger_ms,
            descriptor.channel_id
        );
        let key = (descriptor.trigger_ms, descriptor.id.clone());
        self.pending.insert(key, descriptor);
    }

    /// Remove a pending overlay by ID before it fires.
    pub fn cancel(&mut self, id: &str) -> bool {
        let key_to_remove = self.pending.keys().find(|(_, k)| k == id).cloned();
        if let Some(key) = key_to_remove {
            self.pending.remove(&key);
            debug!("Cancelled overlay '{}'", id);
            true
        } else {
            false
        }
    }

    /// Return all overlays whose trigger time is ≤ `now_ms`, removing them
    /// from the pending queue and adding them to the active list.
    pub fn due_now(&mut self, now_ms: u64) -> Vec<OverlayDescriptor> {
        // Collect keys that are due.
        let due_keys: Vec<(u64, String)> = self
            .pending
            .range(..=(now_ms, "\u{10FFFF}".to_string()))
            .map(|(k, _)| k.clone())
            .collect();

        let mut fired = Vec::with_capacity(due_keys.len());
        for key in due_keys {
            if let Some(desc) = self.pending.remove(&key) {
                let clear_at_ms = desc.hold_ms.map(|h| now_ms + h);
                self.active.push(ActiveOverlay {
                    descriptor: desc.clone(),
                    inserted_at_ms: now_ms,
                    clear_at_ms,
                });
                self.history.push(desc.id.clone());
                fired.push(desc);
            }
        }
        fired
    }

    /// Tick the scheduler at `now_ms`, returning overlays that should be
    /// auto-cleared this tick.  Cleared overlays are removed from the active
    /// list.
    pub fn tick_clear(&mut self, now_ms: u64) -> Vec<ActiveOverlay> {
        let mut cleared = Vec::new();
        self.active.retain(|overlay| {
            if overlay.should_clear(now_ms) {
                cleared.push(overlay.clone());
                false
            } else {
                true
            }
        });
        cleared
    }

    /// Return a slice of currently active overlays.
    pub fn active_overlays(&self) -> &[ActiveOverlay] {
        &self.active
    }

    /// Forcibly clear a specific overlay by ID, returning `true` if found.
    pub fn clear_overlay(&mut self, id: &str) -> bool {
        let before = self.active.len();
        self.active.retain(|o| o.descriptor.id != id);
        self.active.len() < before
    }

    /// Return the number of pending (not yet fired) overlays.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Return the number of overlays ever inserted (history length).
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Clear all pending and active state (e.g. on item change).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.active.clear();
        info!("Graphics overlay scheduler reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lower_third(id: &str, trigger_ms: u64) -> OverlayDescriptor {
        OverlayDescriptor::new(id, OverlayKind::LowerThird, "lt_basic", trigger_ms, 0)
            .with_variable("name", "Alice Smith")
            .with_variable("title", "Reporter")
    }

    #[test]
    fn test_schedule_and_due_now() {
        let mut sched = GraphicsOverlayScheduler::new();
        sched.schedule(make_lower_third("lt1", 1000));
        sched.schedule(make_lower_third("lt2", 2000));

        assert_eq!(sched.pending_count(), 2);

        let due = sched.due_now(1000);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, "lt1");
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn test_due_now_returns_multiple_at_same_time() {
        let mut sched = GraphicsOverlayScheduler::new();
        sched.schedule(make_lower_third("a", 500));
        sched.schedule(make_lower_third("b", 500));

        let due = sched.due_now(500);
        assert_eq!(due.len(), 2);
    }

    #[test]
    fn test_cancel_removes_pending() {
        let mut sched = GraphicsOverlayScheduler::new();
        sched.schedule(make_lower_third("x", 999));
        assert!(sched.cancel("x"));
        assert_eq!(sched.pending_count(), 0);
        assert!(!sched.cancel("x"), "double cancel should return false");
    }

    #[test]
    fn test_active_overlays_populated() {
        let mut sched = GraphicsOverlayScheduler::new();
        sched.schedule(make_lower_third("lt_active", 100));
        sched.due_now(100);
        assert_eq!(sched.active_overlays().len(), 1);
    }

    #[test]
    fn test_tick_clear_removes_expired() {
        let mut sched = GraphicsOverlayScheduler::new();
        let desc = make_lower_third("timed", 0).with_hold(3000);
        sched.schedule(desc);
        sched.due_now(0); // fires the overlay (hold_ms = 3000, clear_at = 3000)
        assert_eq!(sched.active_overlays().len(), 1);

        let cleared = sched.tick_clear(3000);
        assert_eq!(cleared.len(), 1);
        assert_eq!(sched.active_overlays().len(), 0);
    }

    #[test]
    fn test_clear_overlay_by_id() {
        let mut sched = GraphicsOverlayScheduler::new();
        sched.schedule(make_lower_third("remove_me", 0));
        sched.due_now(0);
        assert!(sched.clear_overlay("remove_me"));
        assert_eq!(sched.active_overlays().len(), 0);
    }

    #[test]
    fn test_reset_clears_all() {
        let mut sched = GraphicsOverlayScheduler::new();
        sched.schedule(make_lower_third("r1", 0));
        sched.schedule(make_lower_third("r2", 100));
        sched.due_now(0);
        sched.reset();
        assert_eq!(sched.pending_count(), 0);
        assert_eq!(sched.active_overlays().len(), 0);
    }

    #[test]
    fn test_overlay_kind_label() {
        assert_eq!(OverlayKind::LowerThird.label(), "lower-third");
        assert_eq!(OverlayKind::Bug.label(), "bug");
        assert_eq!(OverlayKind::Ticker.label(), "ticker");
    }

    #[test]
    fn test_history_grows_on_fire() {
        let mut sched = GraphicsOverlayScheduler::new();
        sched.schedule(make_lower_third("h1", 0));
        sched.schedule(make_lower_third("h2", 0));
        sched.due_now(0);
        assert_eq!(sched.history_len(), 2);
    }
}
