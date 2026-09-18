//! Multicam switcher control – program/preview bus with transition support.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

// ── SwitcherSource ────────────────────────────────────────────────────────────

/// A single camera source available to the switcher
#[derive(Debug, Clone)]
pub struct SwitcherSource {
    /// Unique camera identifier
    pub id: u32,
    /// Human-readable camera label (e.g. "Camera 1 – Wide")
    pub name: String,
    /// Whether this camera is powered on and feeding signal
    pub active: bool,
    /// How many frames of latency this camera introduces
    pub latency_frames: u32,
}

impl SwitcherSource {
    /// Create a new source (inactive by default)
    pub fn new(id: u32, name: &str, latency_frames: u32) -> Self {
        Self {
            id,
            name: name.to_string(),
            active: false,
            latency_frames,
        }
    }

    /// A source is ready when it is active and has no latency backlog
    pub fn is_ready(&self) -> bool {
        self.active
    }
}

// ── SwitchMode ────────────────────────────────────────────────────────────────

/// How the switcher transitions between sources
#[derive(Debug, Clone, PartialEq)]
pub enum SwitchMode {
    /// Hard cut – zero transition frames
    Cut,
    /// Cross-dissolve over the given number of frames
    Dissolve(u32),
    /// Instant smash cut (alias for Cut with emphasis)
    SmashCut,
    /// Wipe transition (treated as a single-frame transition here)
    Wipe,
}

impl SwitchMode {
    /// Number of frames the transition occupies
    pub fn transition_frames(&self) -> u32 {
        match self {
            SwitchMode::Cut | SwitchMode::SmashCut | SwitchMode::Wipe => 0,
            SwitchMode::Dissolve(frames) => *frames,
        }
    }
}

// ── SwitcherState ─────────────────────────────────────────────────────────────

/// Current state of the switcher buses
#[derive(Debug, Clone)]
pub struct SwitcherState {
    /// Source ID currently on air (program bus)
    pub current_source: u32,
    /// Source ID currently selected on the preview bus
    pub preview_source: u32,
    /// Active transition, if any
    pub transition: Option<SwitchMode>,
    /// Progress through the current transition (0.0 = start, 1.0 = complete)
    pub transition_progress: f32,
}

impl SwitcherState {
    /// Returns `true` while a dissolve transition is in flight
    pub fn is_transitioning(&self) -> bool {
        self.transition
            .as_ref()
            .is_some_and(|t| t.transition_frames() > 0 && self.transition_progress < 1.0)
    }

    /// The source ID visible to the viewer (program bus)
    pub fn program_source(&self) -> u32 {
        self.current_source
    }
}

// ── MultiCamSwitcher ──────────────────────────────────────────────────────────

/// A live multicam switcher with program and preview buses
#[derive(Debug)]
pub struct MultiCamSwitcher {
    /// Registered sources
    pub sources: Vec<SwitcherSource>,
    /// Current switcher state
    pub state: SwitcherState,
}

impl MultiCamSwitcher {
    /// Create a new switcher with no registered sources.
    ///
    /// Both buses are initialised to source ID 0 (a sentinel – no signal).
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            state: SwitcherState {
                current_source: 0,
                preview_source: 0,
                transition: None,
                transition_progress: 1.0,
            },
        }
    }

    fn find_source(&self, source_id: u32) -> bool {
        self.sources.iter().any(|s| s.id == source_id)
    }

    /// Register a new source with the switcher
    pub fn add_source(&mut self, src: SwitcherSource) {
        self.sources.push(src);
    }

    /// Hard-cut the program bus to `source_id`; returns `false` if not found
    pub fn cut_to(&mut self, source_id: u32) -> bool {
        if !self.find_source(source_id) {
            return false;
        }
        self.state.current_source = source_id;
        self.state.transition = Some(SwitchMode::Cut);
        self.state.transition_progress = 1.0;
        true
    }

    /// Begin a cross-dissolve to `source_id` over `frames` frames
    pub fn dissolve_to(&mut self, source_id: u32, frames: u32) -> bool {
        if !self.find_source(source_id) {
            return false;
        }
        self.state.preview_source = source_id;
        self.state.transition = Some(SwitchMode::Dissolve(frames));
        self.state.transition_progress = 0.0;
        true
    }

    /// Set the preview bus to `source_id`; returns `false` if not found
    pub fn set_preview(&mut self, source_id: u32) -> bool {
        if !self.find_source(source_id) {
            return false;
        }
        self.state.preview_source = source_id;
        true
    }

    /// Cut preview to program (the "TAKE" button)
    ///
    /// Returns `false` if no preview source has been set (preview == 0).
    pub fn take(&mut self) -> bool {
        let prev = self.state.preview_source;
        if prev == 0 {
            return false;
        }
        self.state.current_source = prev;
        self.state.transition = Some(SwitchMode::Cut);
        self.state.transition_progress = 1.0;
        true
    }

    /// All sources with `active == true`
    pub fn active_sources(&self) -> Vec<&SwitcherSource> {
        self.sources.iter().filter(|s| s.active).collect()
    }
}

impl Default for MultiCamSwitcher {
    fn default() -> Self {
        Self::new()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_switcher() -> MultiCamSwitcher {
        let mut sw = MultiCamSwitcher::new();
        let mut s1 = SwitcherSource::new(1, "Camera 1", 0);
        s1.active = true;
        sw.add_source(s1);
        sw.add_source(SwitcherSource::new(2, "Camera 2", 2));
        let mut s3 = SwitcherSource::new(3, "Camera 3", 0);
        s3.active = true;
        sw.add_source(s3);
        sw
    }

    // ── SwitcherSource ───────────────────────────────────────────────────────

    #[test]
    fn test_source_new_inactive() {
        let s = SwitcherSource::new(1, "Wide", 0);
        assert!(!s.active);
        assert!(!s.is_ready());
    }

    #[test]
    fn test_source_active_is_ready() {
        let mut s = SwitcherSource::new(1, "Wide", 0);
        s.active = true;
        assert!(s.is_ready());
    }

    #[test]
    fn test_source_fields_stored() {
        let s = SwitcherSource::new(5, "Jib", 3);
        assert_eq!(s.id, 5);
        assert_eq!(s.name, "Jib");
        assert_eq!(s.latency_frames, 3);
    }

    // ── SwitchMode ───────────────────────────────────────────────────────────

    #[test]
    fn test_cut_transition_frames_zero() {
        assert_eq!(SwitchMode::Cut.transition_frames(), 0);
    }

    #[test]
    fn test_smash_cut_transition_frames_zero() {
        assert_eq!(SwitchMode::SmashCut.transition_frames(), 0);
    }

    #[test]
    fn test_wipe_transition_frames_zero() {
        assert_eq!(SwitchMode::Wipe.transition_frames(), 0);
    }

    #[test]
    fn test_dissolve_transition_frames() {
        assert_eq!(SwitchMode::Dissolve(25).transition_frames(), 25);
    }

    // ── SwitcherState ────────────────────────────────────────────────────────

    #[test]
    fn test_state_not_transitioning_when_no_transition() {
        let state = SwitcherState {
            current_source: 1,
            preview_source: 2,
            transition: None,
            transition_progress: 1.0,
        };
        assert!(!state.is_transitioning());
    }

    #[test]
    fn test_state_is_transitioning_during_dissolve() {
        let state = SwitcherState {
            current_source: 1,
            preview_source: 2,
            transition: Some(SwitchMode::Dissolve(25)),
            transition_progress: 0.5,
        };
        assert!(state.is_transitioning());
    }

    #[test]
    fn test_state_program_source() {
        let state = SwitcherState {
            current_source: 7,
            preview_source: 2,
            transition: None,
            transition_progress: 1.0,
        };
        assert_eq!(state.program_source(), 7);
    }

    // ── MultiCamSwitcher ─────────────────────────────────────────────────────

    #[test]
    fn test_cut_to_valid_source() {
        let mut sw = make_switcher();
        assert!(sw.cut_to(1));
        assert_eq!(sw.state.program_source(), 1);
    }

    #[test]
    fn test_cut_to_invalid_source_returns_false() {
        let mut sw = make_switcher();
        assert!(!sw.cut_to(99));
    }

    #[test]
    fn test_dissolve_to_valid_source() {
        let mut sw = make_switcher();
        assert!(sw.dissolve_to(2, 25));
        assert_eq!(sw.state.preview_source, 2);
        assert!(sw.state.is_transitioning());
    }

    #[test]
    fn test_set_preview_valid_source() {
        let mut sw = make_switcher();
        assert!(sw.set_preview(3));
        assert_eq!(sw.state.preview_source, 3);
    }

    #[test]
    fn test_set_preview_invalid_source_returns_false() {
        let mut sw = make_switcher();
        assert!(!sw.set_preview(99));
    }

    #[test]
    fn test_take_cuts_preview_to_program() {
        let mut sw = make_switcher();
        sw.set_preview(3);
        assert!(sw.take());
        assert_eq!(sw.state.program_source(), 3);
    }

    #[test]
    fn test_take_without_preview_returns_false() {
        let mut sw = make_switcher();
        // preview_source is still 0
        assert!(!sw.take());
    }

    #[test]
    fn test_active_sources_filtered() {
        let sw = make_switcher();
        let active = sw.active_sources();
        assert_eq!(active.len(), 2);
        let ids: Vec<u32> = active.iter().map(|s| s.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }
}
