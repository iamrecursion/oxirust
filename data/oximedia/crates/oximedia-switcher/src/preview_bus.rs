#![allow(dead_code)]
//! Preview bus management for live production switchers.
//!
//! Manages the preview (PVW) bus for each M/E row, supporting source
//! selection, clean feed, overlay composition, and safe-area display.
//! The preview bus shows what will go on-air after the next transition.

use std::collections::HashMap;
use std::fmt;

/// Preview bus overlay mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewOverlay {
    /// No overlay on preview.
    None,
    /// Show safe area markers on preview.
    SafeArea,
    /// Show title/action safe area guides.
    TitleActionSafe,
    /// Show center cross marker.
    CenterCross,
    /// Show grid overlay.
    Grid,
    /// Show all markers.
    All,
}

impl fmt::Display for PreviewOverlay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::SafeArea => write!(f, "Safe Area"),
            Self::TitleActionSafe => write!(f, "Title/Action Safe"),
            Self::CenterCross => write!(f, "Center Cross"),
            Self::Grid => write!(f, "Grid"),
            Self::All => write!(f, "All Markers"),
        }
    }
}

/// Clean feed mode for the preview bus output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanFeedMode {
    /// Normal preview (shows keyers and graphics).
    Normal,
    /// Clean feed: strip downstream keyers.
    CleanNoDownstreamKeyers,
    /// Ultra clean: strip all keyers.
    CleanNoKeyers,
    /// Source only: no effects at all.
    SourceOnly,
}

impl fmt::Display for CleanFeedMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::CleanNoDownstreamKeyers => write!(f, "Clean (no DSK)"),
            Self::CleanNoKeyers => write!(f, "Clean (no keyers)"),
            Self::SourceOnly => write!(f, "Source Only"),
        }
    }
}

/// Preview transition display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewTransitionMode {
    /// Show the preview source directly.
    SourceOnly,
    /// Show a preview of the transition effect.
    TransitionPreview,
    /// Show a split-screen comparison (program / preview).
    SplitScreen,
}

impl fmt::Display for PreviewTransitionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceOnly => write!(f, "Source Only"),
            Self::TransitionPreview => write!(f, "Transition Preview"),
            Self::SplitScreen => write!(f, "Split Screen"),
        }
    }
}

/// Configuration for a single preview bus.
#[derive(Debug, Clone, PartialEq)]
pub struct PreviewBusConfig {
    /// M/E row this preview bus belongs to.
    pub me_row: usize,
    /// Overlay mode.
    pub overlay: PreviewOverlay,
    /// Clean feed mode.
    pub clean_feed: CleanFeedMode,
    /// Transition preview mode.
    pub transition_mode: PreviewTransitionMode,
    /// Whether to display tally indicator on the preview output.
    pub show_tally: bool,
    /// Whether to display input label on the preview output.
    pub show_label: bool,
    /// Border color for active preview (RGB packed as u32).
    pub border_color: u32,
    /// Border width in pixels.
    pub border_width: u32,
}

impl PreviewBusConfig {
    /// Create a new preview bus configuration.
    pub fn new(me_row: usize) -> Self {
        Self {
            me_row,
            overlay: PreviewOverlay::None,
            clean_feed: CleanFeedMode::Normal,
            transition_mode: PreviewTransitionMode::SourceOnly,
            show_tally: true,
            show_label: true,
            border_color: 0x00_FF_00, // green
            border_width: 2,
        }
    }

    /// Set overlay mode.
    pub fn with_overlay(mut self, overlay: PreviewOverlay) -> Self {
        self.overlay = overlay;
        self
    }

    /// Set clean feed mode.
    pub fn with_clean_feed(mut self, mode: CleanFeedMode) -> Self {
        self.clean_feed = mode;
        self
    }

    /// Set transition preview mode.
    pub fn with_transition_mode(mut self, mode: PreviewTransitionMode) -> Self {
        self.transition_mode = mode;
        self
    }
}

/// State of a single preview bus.
#[derive(Debug, Clone, PartialEq)]
pub struct PreviewBusState {
    /// M/E row index.
    pub me_row: usize,
    /// Currently selected source input ID.
    pub source_input: Option<usize>,
    /// Previous source input ID (before last change).
    pub previous_input: Option<usize>,
    /// Whether a transition is active on this M/E.
    pub transition_active: bool,
    /// Transition progress (0.0 to 1.0).
    pub transition_progress: f64,
    /// Configuration.
    pub config: PreviewBusConfig,
    /// Label for the current source.
    pub source_label: String,
}

impl PreviewBusState {
    /// Create a new preview bus state.
    pub fn new(me_row: usize) -> Self {
        Self {
            me_row,
            source_input: None,
            previous_input: None,
            transition_active: false,
            transition_progress: 0.0,
            config: PreviewBusConfig::new(me_row),
            source_label: String::new(),
        }
    }

    /// Select a source input.
    pub fn select_source(&mut self, input_id: usize, label: &str) {
        self.previous_input = self.source_input;
        self.source_input = Some(input_id);
        self.source_label = label.to_string();
    }

    /// Deselect the current source.
    pub fn deselect(&mut self) {
        self.previous_input = self.source_input;
        self.source_input = None;
        self.source_label.clear();
    }

    /// Whether a source is currently selected.
    pub fn has_source(&self) -> bool {
        self.source_input.is_some()
    }

    /// Set transition progress.
    pub fn set_transition_progress(&mut self, progress: f64) {
        self.transition_progress = progress.clamp(0.0, 1.0);
        self.transition_active = progress > 0.0 && progress < 1.0;
    }

    /// Reset transition state.
    pub fn reset_transition(&mut self) {
        self.transition_active = false;
        self.transition_progress = 0.0;
    }
}

impl fmt::Display for PreviewBusState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PVW M/E{}: ", self.me_row + 1)?;
        match self.source_input {
            Some(id) => write!(f, "Input {} ({})", id, self.source_label)?,
            None => write!(f, "No Source")?,
        }
        if self.transition_active {
            write!(f, " [TRANS {:.0}%]", self.transition_progress * 100.0)?;
        }
        Ok(())
    }
}

/// Manager for all preview buses in a switcher.
#[derive(Debug, Clone)]
pub struct PreviewBusManager {
    /// States for each M/E row's preview bus.
    buses: HashMap<usize, PreviewBusState>,
    /// Number of M/E rows.
    me_rows: usize,
    /// Input labels.
    input_labels: HashMap<usize, String>,
}

impl PreviewBusManager {
    /// Create a new preview bus manager.
    pub fn new(me_rows: usize) -> Self {
        let mut buses = HashMap::new();
        for row in 0..me_rows {
            buses.insert(row, PreviewBusState::new(row));
        }
        Self {
            buses,
            me_rows,
            input_labels: HashMap::new(),
        }
    }

    /// Set an input label.
    pub fn set_input_label(&mut self, input_id: usize, label: &str) {
        self.input_labels.insert(input_id, label.to_string());
    }

    /// Get the label for an input.
    pub fn get_input_label(&self, input_id: usize) -> &str {
        self.input_labels
            .get(&input_id)
            .map_or("Unknown", std::string::String::as_str)
    }

    /// Select a source for a preview bus.
    pub fn select(&mut self, me_row: usize, input_id: usize) -> Result<(), PreviewBusError> {
        let label = self.get_input_label(input_id).to_string();
        let bus = self
            .buses
            .get_mut(&me_row)
            .ok_or(PreviewBusError::InvalidMeRow(me_row))?;
        bus.select_source(input_id, &label);
        Ok(())
    }

    /// Deselect a preview bus.
    pub fn deselect(&mut self, me_row: usize) -> Result<(), PreviewBusError> {
        let bus = self
            .buses
            .get_mut(&me_row)
            .ok_or(PreviewBusError::InvalidMeRow(me_row))?;
        bus.deselect();
        Ok(())
    }

    /// Get the selected source for a preview bus.
    pub fn get_source(&self, me_row: usize) -> Result<Option<usize>, PreviewBusError> {
        let bus = self
            .buses
            .get(&me_row)
            .ok_or(PreviewBusError::InvalidMeRow(me_row))?;
        Ok(bus.source_input)
    }

    /// Get the state of a preview bus.
    pub fn get_state(&self, me_row: usize) -> Result<&PreviewBusState, PreviewBusError> {
        self.buses
            .get(&me_row)
            .ok_or(PreviewBusError::InvalidMeRow(me_row))
    }

    /// Get mutable state of a preview bus.
    pub fn get_state_mut(
        &mut self,
        me_row: usize,
    ) -> Result<&mut PreviewBusState, PreviewBusError> {
        self.buses
            .get_mut(&me_row)
            .ok_or(PreviewBusError::InvalidMeRow(me_row))
    }

    /// Set the configuration for a preview bus.
    pub fn set_config(
        &mut self,
        me_row: usize,
        config: PreviewBusConfig,
    ) -> Result<(), PreviewBusError> {
        let bus = self
            .buses
            .get_mut(&me_row)
            .ok_or(PreviewBusError::InvalidMeRow(me_row))?;
        bus.config = config;
        Ok(())
    }

    /// Get all preview bus states.
    pub fn get_all_states(&self) -> Vec<&PreviewBusState> {
        let mut states: Vec<_> = self.buses.values().collect();
        states.sort_by_key(|s| s.me_row);
        states
    }

    /// Number of M/E rows.
    pub fn me_rows(&self) -> usize {
        self.me_rows
    }

    /// Reset all preview buses.
    pub fn reset_all(&mut self) {
        for bus in self.buses.values_mut() {
            bus.deselect();
            bus.reset_transition();
        }
    }
}

/// Errors from preview bus operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreviewBusError {
    /// Invalid M/E row index.
    InvalidMeRow(usize),
    /// No source selected.
    NoSourceSelected(usize),
}

impl fmt::Display for PreviewBusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMeRow(row) => write!(f, "Invalid M/E row: {row}"),
            Self::NoSourceSelected(row) => write!(f, "No source selected on M/E row {row}"),
        }
    }
}

impl std::error::Error for PreviewBusError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preview_overlay_display() {
        assert_eq!(format!("{}", PreviewOverlay::SafeArea), "Safe Area");
        assert_eq!(format!("{}", PreviewOverlay::None), "None");
        assert_eq!(format!("{}", PreviewOverlay::All), "All Markers");
    }

    #[test]
    fn test_clean_feed_display() {
        assert_eq!(format!("{}", CleanFeedMode::Normal), "Normal");
        assert_eq!(
            format!("{}", CleanFeedMode::CleanNoDownstreamKeyers),
            "Clean (no DSK)"
        );
    }

    #[test]
    fn test_preview_bus_config_default() {
        let config = PreviewBusConfig::new(0);
        assert_eq!(config.me_row, 0);
        assert_eq!(config.overlay, PreviewOverlay::None);
        assert_eq!(config.clean_feed, CleanFeedMode::Normal);
        assert!(config.show_tally);
    }

    #[test]
    fn test_preview_bus_config_builder() {
        let config = PreviewBusConfig::new(1)
            .with_overlay(PreviewOverlay::Grid)
            .with_clean_feed(CleanFeedMode::SourceOnly)
            .with_transition_mode(PreviewTransitionMode::SplitScreen);
        assert_eq!(config.overlay, PreviewOverlay::Grid);
        assert_eq!(config.clean_feed, CleanFeedMode::SourceOnly);
        assert_eq!(config.transition_mode, PreviewTransitionMode::SplitScreen);
    }

    #[test]
    fn test_preview_bus_state_creation() {
        let state = PreviewBusState::new(0);
        assert_eq!(state.me_row, 0);
        assert!(!state.has_source());
        assert!(state.source_input.is_none());
    }

    #[test]
    fn test_preview_bus_select_source() {
        let mut state = PreviewBusState::new(0);
        state.select_source(3, "Camera 3");
        assert!(state.has_source());
        assert_eq!(state.source_input, Some(3));
        assert_eq!(state.source_label, "Camera 3");
    }

    #[test]
    fn test_preview_bus_deselect() {
        let mut state = PreviewBusState::new(0);
        state.select_source(3, "Camera 3");
        state.deselect();
        assert!(!state.has_source());
        assert_eq!(state.previous_input, Some(3));
    }

    #[test]
    fn test_preview_bus_transition_progress() {
        let mut state = PreviewBusState::new(0);
        state.set_transition_progress(0.5);
        assert!(state.transition_active);
        assert!((state.transition_progress - 0.5).abs() < f64::EPSILON);
        state.set_transition_progress(1.0);
        assert!(!state.transition_active);
    }

    #[test]
    fn test_preview_bus_state_display() {
        let mut state = PreviewBusState::new(0);
        state.select_source(1, "Camera 1");
        let s = format!("{state}");
        assert!(s.contains("PVW M/E1"));
        assert!(s.contains("Input 1"));
        assert!(s.contains("Camera 1"));
    }

    #[test]
    fn test_manager_creation() {
        let manager = PreviewBusManager::new(2);
        assert_eq!(manager.me_rows(), 2);
    }

    #[test]
    fn test_manager_select() {
        let mut manager = PreviewBusManager::new(2);
        manager.set_input_label(1, "Camera 1");
        manager.select(0, 1).expect("should succeed in test");
        assert_eq!(
            manager.get_source(0).expect("should succeed in test"),
            Some(1)
        );
    }

    #[test]
    fn test_manager_invalid_me_row() {
        let mut manager = PreviewBusManager::new(1);
        let result = manager.select(5, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_deselect() {
        let mut manager = PreviewBusManager::new(2);
        manager.select(0, 1).expect("should succeed in test");
        manager.deselect(0).expect("should succeed in test");
        assert_eq!(manager.get_source(0).expect("should succeed in test"), None);
    }

    #[test]
    fn test_manager_get_all_states() {
        let mut manager = PreviewBusManager::new(3);
        manager.select(0, 1).expect("should succeed in test");
        manager.select(2, 5).expect("should succeed in test");
        let states = manager.get_all_states();
        assert_eq!(states.len(), 3);
        assert_eq!(states[0].me_row, 0);
        assert_eq!(states[1].me_row, 1);
        assert_eq!(states[2].me_row, 2);
    }

    #[test]
    fn test_manager_reset_all() {
        let mut manager = PreviewBusManager::new(2);
        manager.select(0, 1).expect("should succeed in test");
        manager.select(1, 2).expect("should succeed in test");
        manager.reset_all();
        assert_eq!(manager.get_source(0).expect("should succeed in test"), None);
        assert_eq!(manager.get_source(1).expect("should succeed in test"), None);
    }

    #[test]
    fn test_preview_bus_error_display() {
        let err = PreviewBusError::InvalidMeRow(5);
        assert_eq!(format!("{err}"), "Invalid M/E row: 5");
    }
}
