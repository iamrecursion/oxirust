//! Tally light control system for broadcast playout.
//!
//! Manages programme/preview tally state for every source connected to the
//! playout server, and propagates tally updates to downstream controllers.

#![allow(dead_code)]

use std::collections::HashMap;

/// The tally state of a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TallyState {
    /// Source is on-air (programme output).
    Programme,
    /// Source is in preview (next to go on-air).
    Preview,
    /// Source is neither on programme nor preview.
    Idle,
}

impl TallyState {
    /// Returns `true` if any tally is active (programme or preview).
    #[must_use]
    pub fn is_active(self) -> bool {
        matches!(self, Self::Programme | Self::Preview)
    }

    /// Returns `true` if the source is live on the programme bus.
    #[must_use]
    pub fn is_programme(self) -> bool {
        self == Self::Programme
    }

    /// Short human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Programme => "PGM",
            Self::Preview => "PVW",
            Self::Idle => "IDLE",
        }
    }
}

/// A tally update event for a single source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TallyUpdate {
    /// Source identifier (e.g. "CAM1", "VT2").
    pub source_id: String,
    /// New tally state.
    pub state: TallyState,
}

impl TallyUpdate {
    /// Create a new tally update.
    #[must_use]
    pub fn new(source_id: impl Into<String>, state: TallyState) -> Self {
        Self {
            source_id: source_id.into(),
            state,
        }
    }
}

/// Central tally registry for the playout server.
#[derive(Debug, Default)]
pub struct TallySystem {
    /// Current state of every registered source.
    states: HashMap<String, TallyState>,
}

impl TallySystem {
    /// Create a new, empty tally system.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a source with an initial idle state.  If the source is already
    /// registered its state is left unchanged.
    pub fn register_source(&mut self, source_id: impl Into<String>) {
        self.states
            .entry(source_id.into())
            .or_insert(TallyState::Idle);
    }

    /// Apply a tally update.  If the source is not yet registered it is added
    /// automatically.
    pub fn apply_update(&mut self, update: &TallyUpdate) {
        self.states.insert(update.source_id.clone(), update.state);
    }

    /// Get the current tally state of a source.
    #[must_use]
    pub fn state_of(&self, source_id: &str) -> TallyState {
        self.states
            .get(source_id)
            .copied()
            .unwrap_or(TallyState::Idle)
    }

    /// Set the programme source.  Any previous programme source is moved to
    /// idle.  Returns the ID of the previous programme source (if any).
    pub fn set_programme(&mut self, source_id: impl Into<String>) -> Option<String> {
        let new_pgm = source_id.into();
        let prev = self
            .states
            .iter()
            .find(|(_, &v)| v == TallyState::Programme)
            .map(|(k, _)| k.clone());

        if let Some(ref prev_id) = prev {
            if prev_id != &new_pgm {
                self.states.insert(prev_id.clone(), TallyState::Idle);
            }
        }

        self.states.insert(new_pgm, TallyState::Programme);
        prev
    }

    /// Set the preview source.  Any previous preview source is moved to idle.
    pub fn set_preview(&mut self, source_id: impl Into<String>) -> Option<String> {
        let new_pvw = source_id.into();
        let prev = self
            .states
            .iter()
            .find(|(_, &v)| v == TallyState::Preview)
            .map(|(k, _)| k.clone());

        if let Some(ref prev_id) = prev {
            if prev_id != &new_pvw {
                self.states.insert(prev_id.clone(), TallyState::Idle);
            }
        }

        self.states.insert(new_pvw, TallyState::Preview);
        prev
    }

    /// Collect all sources currently in programme state.
    #[must_use]
    pub fn programme_sources(&self) -> Vec<&str> {
        self.states
            .iter()
            .filter(|(_, &v)| v == TallyState::Programme)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Collect all sources currently in preview state.
    #[must_use]
    pub fn preview_sources(&self) -> Vec<&str> {
        self.states
            .iter()
            .filter(|(_, &v)| v == TallyState::Preview)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Number of registered sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.states.len()
    }

    /// Build a list of all current updates (snapshot of the whole system).
    #[must_use]
    pub fn snapshot(&self) -> Vec<TallyUpdate> {
        self.states
            .iter()
            .map(|(id, &state)| TallyUpdate::new(id.clone(), state))
            .collect()
    }

    /// Reset all sources to idle.
    pub fn clear_all(&mut self) {
        for v in self.states.values_mut() {
            *v = TallyState::Idle;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_state_is_active() {
        assert!(TallyState::Programme.is_active());
        assert!(TallyState::Preview.is_active());
        assert!(!TallyState::Idle.is_active());
    }

    #[test]
    fn tally_state_is_programme() {
        assert!(TallyState::Programme.is_programme());
        assert!(!TallyState::Preview.is_programme());
    }

    #[test]
    fn tally_state_labels() {
        assert_eq!(TallyState::Programme.label(), "PGM");
        assert_eq!(TallyState::Preview.label(), "PVW");
        assert_eq!(TallyState::Idle.label(), "IDLE");
    }

    #[test]
    fn register_source_idle_by_default() {
        let mut ts = TallySystem::new();
        ts.register_source("CAM1");
        assert_eq!(ts.state_of("CAM1"), TallyState::Idle);
    }

    #[test]
    fn unregistered_source_returns_idle() {
        let ts = TallySystem::new();
        assert_eq!(ts.state_of("UNKNOWN"), TallyState::Idle);
    }

    #[test]
    fn apply_update_changes_state() {
        let mut ts = TallySystem::new();
        ts.apply_update(&TallyUpdate::new("VT1", TallyState::Programme));
        assert_eq!(ts.state_of("VT1"), TallyState::Programme);
    }

    #[test]
    fn set_programme_moves_old_to_idle() {
        let mut ts = TallySystem::new();
        ts.apply_update(&TallyUpdate::new("CAM1", TallyState::Programme));
        let prev = ts.set_programme("CAM2");
        assert_eq!(prev.as_deref(), Some("CAM1"));
        assert_eq!(ts.state_of("CAM1"), TallyState::Idle);
        assert_eq!(ts.state_of("CAM2"), TallyState::Programme);
    }

    #[test]
    fn set_preview_moves_old_to_idle() {
        let mut ts = TallySystem::new();
        ts.apply_update(&TallyUpdate::new("CAM3", TallyState::Preview));
        ts.set_preview("CAM4");
        assert_eq!(ts.state_of("CAM3"), TallyState::Idle);
        assert_eq!(ts.state_of("CAM4"), TallyState::Preview);
    }

    #[test]
    fn programme_sources_list() {
        let mut ts = TallySystem::new();
        ts.set_programme("CAM1");
        let pgm = ts.programme_sources();
        assert_eq!(pgm.len(), 1);
        assert!(pgm.contains(&"CAM1"));
    }

    #[test]
    fn preview_sources_list() {
        let mut ts = TallySystem::new();
        ts.set_preview("CAM2");
        let pvw = ts.preview_sources();
        assert!(pvw.contains(&"CAM2"));
    }

    #[test]
    fn source_count() {
        let mut ts = TallySystem::new();
        ts.register_source("A");
        ts.register_source("B");
        assert_eq!(ts.source_count(), 2);
    }

    #[test]
    fn snapshot_contains_all_sources() {
        let mut ts = TallySystem::new();
        ts.apply_update(&TallyUpdate::new("X", TallyState::Programme));
        ts.apply_update(&TallyUpdate::new("Y", TallyState::Idle));
        let snap = ts.snapshot();
        assert_eq!(snap.len(), 2);
    }

    #[test]
    fn clear_all_sets_idle() {
        let mut ts = TallySystem::new();
        ts.set_programme("CAM1");
        ts.set_preview("CAM2");
        ts.clear_all();
        assert_eq!(ts.state_of("CAM1"), TallyState::Idle);
        assert_eq!(ts.state_of("CAM2"), TallyState::Idle);
    }

    #[test]
    fn tally_update_new() {
        let u = TallyUpdate::new("SRC", TallyState::Preview);
        assert_eq!(u.source_id, "SRC");
        assert_eq!(u.state, TallyState::Preview);
    }
}
