#![allow(dead_code)]
//! Preset save and recall system for live production switchers.
//!
//! Provides a facility to snapshot the entire switcher state (bus
//! assignments, transition settings, keyer parameters, audio levels)
//! into named presets that can be recalled instantly during a live
//! production.

use std::collections::HashMap;
use std::fmt;
use std::time::SystemTime;

/// Category tag for organizing presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresetCategory {
    /// Full switcher state snapshot.
    FullState,
    /// Bus assignments only.
    BusOnly,
    /// Keyer parameters only.
    KeyerOnly,
    /// Transition settings only.
    TransitionOnly,
    /// Audio mix settings only.
    AudioOnly,
    /// Graphics / CG settings only.
    GraphicsOnly,
    /// User-defined category.
    Custom,
}

impl fmt::Display for PresetCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullState => write!(f, "Full State"),
            Self::BusOnly => write!(f, "Bus Only"),
            Self::KeyerOnly => write!(f, "Keyer Only"),
            Self::TransitionOnly => write!(f, "Transition Only"),
            Self::AudioOnly => write!(f, "Audio Only"),
            Self::GraphicsOnly => write!(f, "Graphics Only"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Recall mode controlling how a preset is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallMode {
    /// Instant recall (hard cut to preset state).
    Instant,
    /// Smooth transition to preset state over N frames.
    Crossfade(u32),
    /// Recall only specified layers.
    Selective,
}

impl fmt::Display for RecallMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Instant => write!(f, "Instant"),
            Self::Crossfade(frames) => write!(f, "Crossfade ({frames} frames)"),
            Self::Selective => write!(f, "Selective"),
        }
    }
}

/// Bus assignment snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BusSnapshot {
    /// M/E row index.
    pub me_row: usize,
    /// Program source input ID.
    pub program_input: Option<usize>,
    /// Preview source input ID.
    pub preview_input: Option<usize>,
}

impl BusSnapshot {
    /// Create a new bus snapshot.
    pub fn new(me_row: usize, program: Option<usize>, preview: Option<usize>) -> Self {
        Self {
            me_row,
            program_input: program,
            preview_input: preview,
        }
    }
}

/// Keyer parameter snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyerSnapshot {
    /// Keyer index.
    pub keyer_id: usize,
    /// Whether upstream keyer.
    pub is_upstream: bool,
    /// Whether on-air.
    pub on_air: bool,
    /// Fill source input.
    pub fill_source: Option<usize>,
    /// Key source input.
    pub key_source: Option<usize>,
    /// Clip level (0.0 to 1.0).
    pub clip: f64,
    /// Gain level (0.0 to 1.0).
    pub gain: f64,
    /// Mask enabled.
    pub mask_enabled: bool,
}

impl KeyerSnapshot {
    /// Create a new keyer snapshot.
    pub fn new(keyer_id: usize, is_upstream: bool) -> Self {
        Self {
            keyer_id,
            is_upstream,
            on_air: false,
            fill_source: None,
            key_source: None,
            clip: 0.5,
            gain: 1.0,
            mask_enabled: false,
        }
    }
}

/// Transition parameter snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionSnapshot {
    /// M/E row index.
    pub me_row: usize,
    /// Transition type name.
    pub transition_type: String,
    /// Duration in frames.
    pub duration_frames: u32,
    /// Wipe pattern index (if wipe).
    pub wipe_pattern: Option<u32>,
    /// Whether reverse direction.
    pub reverse: bool,
}

impl TransitionSnapshot {
    /// Create a new transition snapshot.
    pub fn new(me_row: usize, transition_type: &str, duration_frames: u32) -> Self {
        Self {
            me_row,
            transition_type: transition_type.to_string(),
            duration_frames,
            wipe_pattern: None,
            reverse: false,
        }
    }
}

/// Audio level snapshot for one channel.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioLevelSnapshot {
    /// Channel / input ID.
    pub channel_id: usize,
    /// Fader level (0.0 to 1.0).
    pub fader: f64,
    /// Whether muted.
    pub muted: bool,
    /// AFV (Audio Follow Video) enabled.
    pub afv: bool,
    /// Pan position (-1.0 left to 1.0 right).
    pub pan: f64,
}

impl AudioLevelSnapshot {
    /// Create a new audio level snapshot.
    pub fn new(channel_id: usize, fader: f64) -> Self {
        Self {
            channel_id,
            fader: fader.clamp(0.0, 1.0),
            muted: false,
            afv: false,
            pan: 0.0,
        }
    }
}

/// Aux bus assignment snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuxSnapshot {
    /// Aux bus index.
    pub aux_id: usize,
    /// Assigned input ID.
    pub input_id: Option<usize>,
}

impl AuxSnapshot {
    /// Create a new aux snapshot.
    pub fn new(aux_id: usize, input_id: Option<usize>) -> Self {
        Self { aux_id, input_id }
    }
}

/// A complete switcher preset snapshot.
#[derive(Debug, Clone)]
pub struct SwitcherPreset {
    /// Unique preset ID.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Category.
    pub category: PresetCategory,
    /// Description / notes.
    pub description: String,
    /// Bus assignments.
    pub buses: Vec<BusSnapshot>,
    /// Keyer parameters.
    pub keyers: Vec<KeyerSnapshot>,
    /// Transition settings.
    pub transitions: Vec<TransitionSnapshot>,
    /// Audio levels.
    pub audio_levels: Vec<AudioLevelSnapshot>,
    /// Aux assignments.
    pub aux_assignments: Vec<AuxSnapshot>,
    /// When the preset was created.
    pub created_at: SystemTime,
    /// When the preset was last recalled.
    pub last_recalled: Option<SystemTime>,
    /// Number of times recalled.
    pub recall_count: u32,
    /// Custom tags.
    pub tags: Vec<String>,
    /// Whether this preset is locked (cannot be overwritten).
    pub locked: bool,
}

impl SwitcherPreset {
    /// Create a new empty preset.
    pub fn new(id: u32, name: &str, category: PresetCategory) -> Self {
        Self {
            id,
            name: name.to_string(),
            category,
            description: String::new(),
            buses: Vec::new(),
            keyers: Vec::new(),
            transitions: Vec::new(),
            audio_levels: Vec::new(),
            aux_assignments: Vec::new(),
            created_at: SystemTime::now(),
            last_recalled: None,
            recall_count: 0,
            tags: Vec::new(),
            locked: false,
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a bus snapshot.
    pub fn add_bus(&mut self, bus: BusSnapshot) {
        self.buses.push(bus);
    }

    /// Add a keyer snapshot.
    pub fn add_keyer(&mut self, keyer: KeyerSnapshot) {
        self.keyers.push(keyer);
    }

    /// Add a transition snapshot.
    pub fn add_transition(&mut self, transition: TransitionSnapshot) {
        self.transitions.push(transition);
    }

    /// Add an audio level snapshot.
    pub fn add_audio_level(&mut self, level: AudioLevelSnapshot) {
        self.audio_levels.push(level);
    }

    /// Add an aux snapshot.
    pub fn add_aux(&mut self, aux: AuxSnapshot) {
        self.aux_assignments.push(aux);
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: &str) {
        if !self.tags.contains(&tag.to_string()) {
            self.tags.push(tag.to_string());
        }
    }

    /// Lock the preset.
    pub fn lock(&mut self) {
        self.locked = true;
    }

    /// Unlock the preset.
    pub fn unlock(&mut self) {
        self.locked = false;
    }

    /// Mark as recalled.
    pub fn mark_recalled(&mut self) {
        self.last_recalled = Some(SystemTime::now());
        self.recall_count += 1;
    }

    /// Whether this preset has bus data.
    pub fn has_buses(&self) -> bool {
        !self.buses.is_empty()
    }

    /// Whether this preset has keyer data.
    pub fn has_keyers(&self) -> bool {
        !self.keyers.is_empty()
    }

    /// Whether this preset has audio data.
    pub fn has_audio(&self) -> bool {
        !self.audio_levels.is_empty()
    }

    /// Total number of snapshots in this preset.
    pub fn component_count(&self) -> usize {
        self.buses.len()
            + self.keyers.len()
            + self.transitions.len()
            + self.audio_levels.len()
            + self.aux_assignments.len()
    }
}

impl fmt::Display for SwitcherPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:03}] {} ({}) - {} components",
            self.id,
            self.name,
            self.category,
            self.component_count()
        )
    }
}

/// Preset storage and recall manager.
#[derive(Debug, Clone)]
pub struct PresetManager {
    /// Stored presets indexed by ID.
    presets: HashMap<u32, SwitcherPreset>,
    /// Next available ID.
    next_id: u32,
    /// Maximum number of presets.
    max_presets: usize,
}

impl PresetManager {
    /// Create a new preset manager.
    pub fn new(max_presets: usize) -> Self {
        Self {
            presets: HashMap::new(),
            next_id: 1,
            max_presets,
        }
    }

    /// Store a preset. Returns the assigned ID.
    pub fn store(&mut self, mut preset: SwitcherPreset) -> Result<u32, PresetError> {
        if self.presets.len() >= self.max_presets {
            return Err(PresetError::StorageFull(self.max_presets));
        }
        let id = self.next_id;
        preset.id = id;
        self.next_id += 1;
        self.presets.insert(id, preset);
        Ok(id)
    }

    /// Store a preset at a specific ID slot.
    pub fn store_at(&mut self, id: u32, mut preset: SwitcherPreset) -> Result<(), PresetError> {
        if let Some(existing) = self.presets.get(&id) {
            if existing.locked {
                return Err(PresetError::PresetLocked(id));
            }
        }
        preset.id = id;
        self.presets.insert(id, preset);
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        Ok(())
    }

    /// Recall a preset by ID.
    pub fn recall(&mut self, id: u32) -> Result<&SwitcherPreset, PresetError> {
        let preset = self.presets.get_mut(&id).ok_or(PresetError::NotFound(id))?;
        preset.mark_recalled();
        Ok(preset)
    }

    /// Get a preset by ID without marking as recalled.
    pub fn get(&self, id: u32) -> Option<&SwitcherPreset> {
        self.presets.get(&id)
    }

    /// Delete a preset.
    pub fn delete(&mut self, id: u32) -> Result<(), PresetError> {
        if let Some(preset) = self.presets.get(&id) {
            if preset.locked {
                return Err(PresetError::PresetLocked(id));
            }
        }
        self.presets
            .remove(&id)
            .map(|_| ())
            .ok_or(PresetError::NotFound(id))
    }

    /// List all presets.
    pub fn list(&self) -> Vec<&SwitcherPreset> {
        let mut presets: Vec<_> = self.presets.values().collect();
        presets.sort_by_key(|p| p.id);
        presets
    }

    /// List presets by category.
    pub fn list_by_category(&self, category: PresetCategory) -> Vec<&SwitcherPreset> {
        let mut presets: Vec<_> = self
            .presets
            .values()
            .filter(|p| p.category == category)
            .collect();
        presets.sort_by_key(|p| p.id);
        presets
    }

    /// Search presets by name substring.
    pub fn search(&self, query: &str) -> Vec<&SwitcherPreset> {
        let lower_query = query.to_lowercase();
        self.presets
            .values()
            .filter(|p| p.name.to_lowercase().contains(&lower_query))
            .collect()
    }

    /// Number of stored presets.
    pub fn count(&self) -> usize {
        self.presets.len()
    }

    /// Whether the storage is full.
    pub fn is_full(&self) -> bool {
        self.presets.len() >= self.max_presets
    }

    /// Clear all presets (except locked ones).
    pub fn clear_unlocked(&mut self) {
        self.presets.retain(|_, p| p.locked);
    }
}

/// Errors from preset operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresetError {
    /// Preset not found.
    NotFound(u32),
    /// Preset is locked.
    PresetLocked(u32),
    /// Preset storage is full.
    StorageFull(usize),
}

impl fmt::Display for PresetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Preset {id} not found"),
            Self::PresetLocked(id) => write!(f, "Preset {id} is locked"),
            Self::StorageFull(max) => write!(f, "Preset storage full (max {max})"),
        }
    }
}

impl std::error::Error for PresetError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_category_display() {
        assert_eq!(format!("{}", PresetCategory::FullState), "Full State");
        assert_eq!(format!("{}", PresetCategory::AudioOnly), "Audio Only");
        assert_eq!(format!("{}", PresetCategory::Custom), "Custom");
    }

    #[test]
    fn test_recall_mode_display() {
        assert_eq!(format!("{}", RecallMode::Instant), "Instant");
        assert_eq!(
            format!("{}", RecallMode::Crossfade(30)),
            "Crossfade (30 frames)"
        );
    }

    #[test]
    fn test_preset_creation() {
        let preset = SwitcherPreset::new(1, "Opening", PresetCategory::FullState);
        assert_eq!(preset.id, 1);
        assert_eq!(preset.name, "Opening");
        assert_eq!(preset.category, PresetCategory::FullState);
        assert!(!preset.locked);
        assert_eq!(preset.recall_count, 0);
    }

    #[test]
    fn test_preset_add_bus() {
        let mut preset = SwitcherPreset::new(1, "Test", PresetCategory::BusOnly);
        preset.add_bus(BusSnapshot::new(0, Some(1), Some(2)));
        assert!(preset.has_buses());
        assert_eq!(preset.buses.len(), 1);
        assert_eq!(preset.buses[0].program_input, Some(1));
    }

    #[test]
    fn test_preset_add_keyer() {
        let mut preset = SwitcherPreset::new(1, "Test", PresetCategory::KeyerOnly);
        let mut keyer = KeyerSnapshot::new(0, true);
        keyer.on_air = true;
        keyer.fill_source = Some(5);
        preset.add_keyer(keyer);
        assert!(preset.has_keyers());
        assert!(preset.keyers[0].on_air);
    }

    #[test]
    fn test_preset_add_audio() {
        let mut preset = SwitcherPreset::new(1, "Test", PresetCategory::AudioOnly);
        preset.add_audio_level(AudioLevelSnapshot::new(0, 0.75));
        assert!(preset.has_audio());
        assert!((preset.audio_levels[0].fader - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_preset_component_count() {
        let mut preset = SwitcherPreset::new(1, "Test", PresetCategory::FullState);
        preset.add_bus(BusSnapshot::new(0, Some(1), Some(2)));
        preset.add_keyer(KeyerSnapshot::new(0, true));
        preset.add_transition(TransitionSnapshot::new(0, "mix", 30));
        preset.add_audio_level(AudioLevelSnapshot::new(0, 1.0));
        preset.add_aux(AuxSnapshot::new(0, Some(3)));
        assert_eq!(preset.component_count(), 5);
    }

    #[test]
    fn test_preset_lock_unlock() {
        let mut preset = SwitcherPreset::new(1, "Test", PresetCategory::FullState);
        assert!(!preset.locked);
        preset.lock();
        assert!(preset.locked);
        preset.unlock();
        assert!(!preset.locked);
    }

    #[test]
    fn test_preset_tags() {
        let mut preset = SwitcherPreset::new(1, "Test", PresetCategory::FullState);
        preset.add_tag("live");
        preset.add_tag("news");
        preset.add_tag("live"); // duplicate
        assert_eq!(preset.tags.len(), 2);
    }

    #[test]
    fn test_preset_display() {
        let mut preset = SwitcherPreset::new(1, "Opening", PresetCategory::FullState);
        preset.add_bus(BusSnapshot::new(0, Some(1), Some(2)));
        let s = format!("{preset}");
        assert!(s.contains("001"));
        assert!(s.contains("Opening"));
        assert!(s.contains("Full State"));
    }

    #[test]
    fn test_manager_store_and_get() {
        let mut manager = PresetManager::new(100);
        let preset = SwitcherPreset::new(0, "Test", PresetCategory::FullState);
        let id = manager.store(preset).expect("should succeed in test");
        assert!(manager.get(id).is_some());
        assert_eq!(manager.count(), 1);
    }

    #[test]
    fn test_manager_recall() {
        let mut manager = PresetManager::new(100);
        let preset = SwitcherPreset::new(0, "Test", PresetCategory::FullState);
        let id = manager.store(preset).expect("should succeed in test");
        let recalled = manager.recall(id).expect("should succeed in test");
        assert_eq!(recalled.recall_count, 1);
    }

    #[test]
    fn test_manager_delete() {
        let mut manager = PresetManager::new(100);
        let preset = SwitcherPreset::new(0, "Test", PresetCategory::FullState);
        let id = manager.store(preset).expect("should succeed in test");
        manager.delete(id).expect("should succeed in test");
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_manager_delete_locked() {
        let mut manager = PresetManager::new(100);
        let mut preset = SwitcherPreset::new(0, "Test", PresetCategory::FullState);
        preset.lock();
        let id = manager.store(preset).expect("should succeed in test");
        let result = manager.delete(id);
        assert_eq!(result, Err(PresetError::PresetLocked(id)));
    }

    #[test]
    fn test_manager_storage_full() {
        let mut manager = PresetManager::new(1);
        let p1 = SwitcherPreset::new(0, "A", PresetCategory::FullState);
        manager.store(p1).expect("should succeed in test");
        let p2 = SwitcherPreset::new(0, "B", PresetCategory::FullState);
        let result = manager.store(p2);
        assert_eq!(result, Err(PresetError::StorageFull(1)));
    }

    #[test]
    fn test_manager_list_by_category() {
        let mut manager = PresetManager::new(100);
        manager
            .store(SwitcherPreset::new(0, "A", PresetCategory::FullState))
            .expect("should succeed in test");
        manager
            .store(SwitcherPreset::new(0, "B", PresetCategory::AudioOnly))
            .expect("should succeed in test");
        manager
            .store(SwitcherPreset::new(0, "C", PresetCategory::FullState))
            .expect("should succeed in test");
        let full = manager.list_by_category(PresetCategory::FullState);
        assert_eq!(full.len(), 2);
    }

    #[test]
    fn test_manager_search() {
        let mut manager = PresetManager::new(100);
        manager
            .store(SwitcherPreset::new(
                0,
                "Opening Shot",
                PresetCategory::FullState,
            ))
            .expect("should succeed in test");
        manager
            .store(SwitcherPreset::new(
                0,
                "Interview",
                PresetCategory::FullState,
            ))
            .expect("should succeed in test");
        let found = manager.search("open");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "Opening Shot");
    }

    #[test]
    fn test_manager_clear_unlocked() {
        let mut manager = PresetManager::new(100);
        let mut locked = SwitcherPreset::new(0, "Locked", PresetCategory::FullState);
        locked.lock();
        manager.store(locked).expect("should succeed in test");
        manager
            .store(SwitcherPreset::new(
                0,
                "Unlocked",
                PresetCategory::FullState,
            ))
            .expect("should succeed in test");
        manager.clear_unlocked();
        assert_eq!(manager.count(), 1);
    }

    #[test]
    fn test_preset_error_display() {
        assert_eq!(
            format!("{}", PresetError::NotFound(5)),
            "Preset 5 not found"
        );
        assert_eq!(
            format!("{}", PresetError::PresetLocked(3)),
            "Preset 3 is locked"
        );
    }

    #[test]
    fn test_audio_level_clamp() {
        let level = AudioLevelSnapshot::new(0, 1.5);
        assert!((level.fader - 1.0).abs() < f64::EPSILON);
        let level2 = AudioLevelSnapshot::new(0, -0.5);
        assert!((level2.fader - 0.0).abs() < f64::EPSILON);
    }
}
