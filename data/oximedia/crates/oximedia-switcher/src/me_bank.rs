//! Mix/Effect bank management for video switchers.
//!
//! A Mix/Effect (M/E) bank is the core processing unit of a professional video switcher.
//! Each M/E bank has its own program and preview buses, keyers, and transition engine.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Transition style for an M/E bank.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TransitionStyle {
    /// Mix/Dissolve (crossfade)
    Mix,
    /// Dip to color
    Dip,
    /// Wipe with pattern
    Wipe,
    /// Sting (graphic wipe)
    Sting,
    /// Push (DVE slide)
    Push,
}

impl TransitionStyle {
    /// Returns true if this is a simple (non-DVE) transition.
    #[must_use]
    pub fn is_simple(&self) -> bool {
        matches!(self, Self::Mix | Self::Dip | Self::Wipe)
    }
}

/// Key state for an M/E bank upstream keyer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyState {
    /// Keyer ID
    pub key_id: u32,
    /// Whether the keyer is on-air
    pub on_air: bool,
    /// Mix level (0.0 = fully transparent, 1.0 = fully opaque)
    pub mix_level: f32,
    /// Pattern index for pattern keyers
    pub pattern: u32,
}

impl KeyState {
    /// Create a new key state.
    #[must_use]
    pub fn new(key_id: u32) -> Self {
        Self {
            key_id,
            on_air: false,
            mix_level: 1.0,
            pattern: 0,
        }
    }

    /// Set on-air state.
    pub fn set_on_air(&mut self, on_air: bool) {
        self.on_air = on_air;
    }

    /// Set mix level clamped to [0.0, 1.0].
    pub fn set_mix_level(&mut self, level: f32) {
        self.mix_level = level.clamp(0.0, 1.0);
    }
}

/// Transition state for an M/E bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionState {
    /// Whether a transition is in progress
    pub in_progress: bool,
    /// Transition position (0.0 = start, 1.0 = complete)
    pub position: f32,
    /// Transition style
    pub style: TransitionStyle,
    /// Duration in frames
    pub rate_frames: u32,
}

impl TransitionState {
    /// Create a new transition state with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            in_progress: false,
            position: 0.0,
            style: TransitionStyle::Mix,
            rate_frames: 25,
        }
    }

    /// Create with a specific style and rate.
    #[must_use]
    pub fn with_style(style: TransitionStyle, rate_frames: u32) -> Self {
        Self {
            in_progress: false,
            position: 0.0,
            style,
            rate_frames,
        }
    }
}

impl Default for TransitionState {
    fn default() -> Self {
        Self::new()
    }
}

/// Mix/Effect bank — core processing unit of a switcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeBank {
    /// Bank ID
    pub id: u32,
    /// Current program input
    pub program_input: u32,
    /// Current preview input
    pub preview_input: u32,
    /// Key states for each upstream keyer
    pub key_states: Vec<KeyState>,
    /// Active transition state
    pub transition: TransitionState,
}

impl MeBank {
    /// Create a new M/E bank with the given number of keyers.
    #[must_use]
    pub fn new(id: u32, key_count: u32) -> Self {
        let key_states = (0..key_count).map(KeyState::new).collect();
        Self {
            id,
            program_input: 0,
            preview_input: 1,
            key_states,
            transition: TransitionState::new(),
        }
    }

    /// Get the bank ID.
    #[must_use]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Get the current program input.
    #[must_use]
    pub fn program_input(&self) -> u32 {
        self.program_input
    }

    /// Set the program input.
    pub fn set_program_input(&mut self, input: u32) {
        self.program_input = input;
    }

    /// Get the current preview input.
    #[must_use]
    pub fn preview_input(&self) -> u32 {
        self.preview_input
    }

    /// Set the preview input.
    pub fn set_preview_input(&mut self, input: u32) {
        self.preview_input = input;
    }

    /// Begin an auto transition from preview to program.
    ///
    /// Starts the transition engine with the given total frame count.
    pub fn auto_transition(&mut self, total_frames: u32) {
        self.transition.in_progress = true;
        self.transition.position = 0.0;
        self.transition.rate_frames = total_frames.max(1);
    }

    /// Advance the transition by one frame.
    ///
    /// Returns `true` if the transition completed this frame.
    pub fn advance_transition(&mut self) -> bool {
        if !self.transition.in_progress {
            return false;
        }

        let step = 1.0 / self.transition.rate_frames as f32;
        self.transition.position = (self.transition.position + step).min(1.0);

        if self.transition.position >= 1.0 {
            // Transition complete: take preview to program
            self.program_input = self.preview_input;
            self.transition.in_progress = false;
            self.transition.position = 0.0;
            true
        } else {
            false
        }
    }

    /// Perform an instant cut — preview immediately becomes program.
    pub fn cut(&mut self) {
        std::mem::swap(&mut self.program_input, &mut self.preview_input);
        // Cancel any in-progress transition
        self.transition.in_progress = false;
        self.transition.position = 0.0;
    }

    /// Get a reference to a key state by index.
    #[must_use]
    pub fn key_state(&self, key_id: u32) -> Option<&KeyState> {
        self.key_states.iter().find(|k| k.key_id == key_id)
    }

    /// Get a mutable reference to a key state by index.
    pub fn key_state_mut(&mut self, key_id: u32) -> Option<&mut KeyState> {
        self.key_states.iter_mut().find(|k| k.key_id == key_id)
    }

    /// Returns `true` if a transition is currently in progress.
    #[must_use]
    pub fn is_transitioning(&self) -> bool {
        self.transition.in_progress
    }

    /// Get the current transition position (0.0–1.0).
    #[must_use]
    pub fn transition_position(&self) -> f32 {
        self.transition.position
    }

    /// Set the transition style.
    pub fn set_transition_style(&mut self, style: TransitionStyle) {
        self.transition.style = style;
    }

    /// Set the transition rate in frames.
    pub fn set_transition_rate(&mut self, rate_frames: u32) {
        self.transition.rate_frames = rate_frames.max(1);
    }
}

/// Configuration for creating an M/E bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeBankConfig {
    /// Human-readable name
    pub name: String,
    /// Number of inputs available to this bank
    pub input_count: u32,
    /// Number of upstream keyers
    pub key_count: u32,
    /// Whether downstream keyers (DSK) are supported
    pub supports_dsk: bool,
}

impl MeBankConfig {
    /// Create a standard M/E bank configuration.
    #[must_use]
    pub fn new(name: impl Into<String>, input_count: u32, key_count: u32) -> Self {
        Self {
            name: name.into(),
            input_count,
            key_count,
            supports_dsk: false,
        }
    }

    /// Create with downstream keyer support.
    #[must_use]
    pub fn with_dsk(mut self) -> Self {
        self.supports_dsk = true;
        self
    }
}

impl Default for MeBankConfig {
    fn default() -> Self {
        Self::new("M/E 1", 8, 4)
    }
}

/// Manager for multiple M/E banks.
pub struct MeManager {
    banks: HashMap<u32, MeBank>,
    configs: HashMap<u32, MeBankConfig>,
    next_id: u32,
}

impl MeManager {
    /// Create a new M/E manager with no banks.
    #[must_use]
    pub fn new() -> Self {
        Self {
            banks: HashMap::new(),
            configs: HashMap::new(),
            next_id: 0,
        }
    }

    /// Create an M/E bank from a configuration and return its assigned ID.
    pub fn create_bank(&mut self, config: MeBankConfig) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let bank = MeBank::new(id, config.key_count);
        self.banks.insert(id, bank);
        self.configs.insert(id, config);
        id
    }

    /// Get a mutable reference to a bank by ID.
    pub fn bank(&mut self, id: u32) -> Option<&mut MeBank> {
        self.banks.get_mut(&id)
    }

    /// Get an immutable reference to a bank by ID.
    pub fn bank_ref(&self, id: u32) -> Option<&MeBank> {
        self.banks.get(&id)
    }

    /// Get the configuration for a bank.
    pub fn bank_config(&self, id: u32) -> Option<&MeBankConfig> {
        self.configs.get(&id)
    }

    /// Get the total number of banks.
    #[must_use]
    pub fn bank_count(&self) -> u32 {
        self.banks.len() as u32
    }

    /// Remove a bank.
    pub fn remove_bank(&mut self, id: u32) -> bool {
        let removed = self.banks.remove(&id).is_some();
        self.configs.remove(&id);
        removed
    }

    /// Get all bank IDs.
    #[must_use]
    pub fn bank_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.banks.keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}

impl Default for MeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_style_is_simple() {
        assert!(TransitionStyle::Mix.is_simple());
        assert!(TransitionStyle::Dip.is_simple());
        assert!(TransitionStyle::Wipe.is_simple());
        assert!(!TransitionStyle::Sting.is_simple());
        assert!(!TransitionStyle::Push.is_simple());
    }

    #[test]
    fn test_key_state_creation() {
        let ks = KeyState::new(0);
        assert_eq!(ks.key_id, 0);
        assert!(!ks.on_air);
        assert_eq!(ks.mix_level, 1.0);
        assert_eq!(ks.pattern, 0);
    }

    #[test]
    fn test_key_state_set_mix_level_clamped() {
        let mut ks = KeyState::new(0);
        ks.set_mix_level(1.5);
        assert_eq!(ks.mix_level, 1.0);
        ks.set_mix_level(-0.5);
        assert_eq!(ks.mix_level, 0.0);
    }

    #[test]
    fn test_me_bank_creation() {
        let bank = MeBank::new(0, 4);
        assert_eq!(bank.id(), 0);
        assert_eq!(bank.program_input(), 0);
        assert_eq!(bank.preview_input(), 1);
        assert_eq!(bank.key_states.len(), 4);
        assert!(!bank.is_transitioning());
    }

    #[test]
    fn test_me_bank_cut() {
        let mut bank = MeBank::new(0, 2);
        bank.set_program_input(1);
        bank.set_preview_input(2);
        bank.cut();
        assert_eq!(bank.program_input(), 2);
        assert_eq!(bank.preview_input(), 1);
    }

    #[test]
    fn test_me_bank_cut_cancels_transition() {
        let mut bank = MeBank::new(0, 2);
        bank.auto_transition(25);
        assert!(bank.is_transitioning());
        bank.cut();
        assert!(!bank.is_transitioning());
    }

    #[test]
    fn test_me_bank_auto_transition_begins() {
        let mut bank = MeBank::new(0, 2);
        bank.set_program_input(1);
        bank.set_preview_input(2);
        bank.auto_transition(10);
        assert!(bank.is_transitioning());
        assert_eq!(bank.transition_position(), 0.0);
    }

    #[test]
    fn test_me_bank_advance_transition_completes() {
        let mut bank = MeBank::new(0, 2);
        bank.set_program_input(1);
        bank.set_preview_input(2);
        bank.auto_transition(1);

        let done = bank.advance_transition();
        assert!(done);
        assert!(!bank.is_transitioning());
        // After completion program should equal what was preview
        assert_eq!(bank.program_input(), 2);
    }

    #[test]
    fn test_me_bank_advance_transition_partial() {
        let mut bank = MeBank::new(0, 2);
        bank.set_program_input(1);
        bank.set_preview_input(2);
        bank.auto_transition(10);

        // Advance 5 frames — should not be complete yet
        for _ in 0..5 {
            let done = bank.advance_transition();
            assert!(!done);
        }
        assert!(bank.is_transitioning());
        assert!(bank.transition_position() > 0.0 && bank.transition_position() < 1.0);
    }

    #[test]
    fn test_me_bank_advance_transition_full() {
        let mut bank = MeBank::new(0, 2);
        bank.set_program_input(1);
        bank.set_preview_input(3);
        bank.auto_transition(5);

        let mut completed = false;
        for _ in 0..5 {
            completed = bank.advance_transition();
        }
        assert!(completed);
        assert!(!bank.is_transitioning());
    }

    #[test]
    fn test_me_bank_advance_when_not_transitioning() {
        let mut bank = MeBank::new(0, 2);
        let done = bank.advance_transition();
        assert!(!done);
    }

    #[test]
    fn test_me_bank_key_state_access() {
        let mut bank = MeBank::new(0, 4);
        {
            let ks = bank.key_state_mut(2).expect("key 2 should exist");
            ks.set_on_air(true);
        }
        assert!(bank.key_state(2).expect("should succeed in test").on_air);
        assert!(!bank.key_state(0).expect("should succeed in test").on_air);
    }

    #[test]
    fn test_me_bank_set_transition_style() {
        let mut bank = MeBank::new(0, 2);
        bank.set_transition_style(TransitionStyle::Push);
        assert_eq!(bank.transition.style, TransitionStyle::Push);
    }

    #[test]
    fn test_me_bank_config_default() {
        let config = MeBankConfig::default();
        assert_eq!(config.input_count, 8);
        assert_eq!(config.key_count, 4);
        assert!(!config.supports_dsk);
    }

    #[test]
    fn test_me_bank_config_with_dsk() {
        let config = MeBankConfig::new("M/E 1", 16, 4).with_dsk();
        assert!(config.supports_dsk);
    }

    #[test]
    fn test_me_manager_create_bank() {
        let mut manager = MeManager::new();
        let id = manager.create_bank(MeBankConfig::default());
        assert_eq!(id, 0);
        assert_eq!(manager.bank_count(), 1);
    }

    #[test]
    fn test_me_manager_multiple_banks() {
        let mut manager = MeManager::new();
        let id0 = manager.create_bank(MeBankConfig::new("M/E 1", 8, 4));
        let id1 = manager.create_bank(MeBankConfig::new("M/E 2", 8, 4));
        assert_ne!(id0, id1);
        assert_eq!(manager.bank_count(), 2);
    }

    #[test]
    fn test_me_manager_bank_access() {
        let mut manager = MeManager::new();
        let id = manager.create_bank(MeBankConfig::new("Main", 8, 4));

        {
            let bank = manager.bank(id).expect("should succeed in test");
            bank.set_program_input(5);
        }

        assert_eq!(
            manager
                .bank_ref(id)
                .expect("should succeed in test")
                .program_input(),
            5
        );
    }

    #[test]
    fn test_me_manager_remove_bank() {
        let mut manager = MeManager::new();
        let id = manager.create_bank(MeBankConfig::default());
        assert!(manager.remove_bank(id));
        assert_eq!(manager.bank_count(), 0);
        assert!(manager.bank(id).is_none());
    }

    #[test]
    fn test_me_manager_bank_ids_sorted() {
        let mut manager = MeManager::new();
        manager.create_bank(MeBankConfig::default());
        manager.create_bank(MeBankConfig::default());
        manager.create_bank(MeBankConfig::default());
        let ids = manager.bank_ids();
        assert_eq!(ids, vec![0, 1, 2]);
    }
}
