//! FX strip and effect slot management for clips.
//!
//! Provides a slot-based effects chain per clip with enable/disable,
//! preset snapshots, and active-count queries.

#![allow(dead_code)]

/// A single slot in an effects strip.
#[derive(Debug, Clone)]
pub struct FxSlot {
    /// Display name of the effect in this slot.
    pub name: String,
    /// Whether this slot is currently enabled.
    pub enabled: bool,
    /// Arbitrary parameter values for the effect.
    pub params: Vec<f32>,
}

impl FxSlot {
    /// Create a new `FxSlot` with the given name, enabled by default.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            params: Vec::new(),
        }
    }

    /// Returns `true` if this slot is currently enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable this slot.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable this slot.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

/// An ordered strip of [`FxSlot`]s applied to a clip.
#[derive(Debug, Clone, Default)]
pub struct FxStrip {
    slots: Vec<FxSlot>,
}

impl FxStrip {
    /// Create an empty `FxStrip`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an effect slot to the end of the strip.
    ///
    /// Returns the index at which the slot was inserted.
    pub fn add_effect(&mut self, slot: FxSlot) -> usize {
        let idx = self.slots.len();
        self.slots.push(slot);
        idx
    }

    /// Enable the slot at `index`.
    ///
    /// Returns `false` if `index` is out of bounds.
    pub fn enable(&mut self, index: usize) -> bool {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.enable();
            true
        } else {
            false
        }
    }

    /// Disable the slot at `index`.
    ///
    /// Returns `false` if `index` is out of bounds.
    pub fn disable(&mut self, index: usize) -> bool {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.disable();
            true
        } else {
            false
        }
    }

    /// Returns the number of currently active (enabled) slots.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_enabled()).count()
    }

    /// Returns a slice of all slots in this strip.
    #[must_use]
    pub fn slots(&self) -> &[FxSlot] {
        &self.slots
    }

    /// Returns the total number of slots (enabled or disabled).
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Returns `true` if there are no slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// A named preset that snapshots an [`FxStrip`] configuration.
#[derive(Debug, Clone)]
pub struct FxChainPreset {
    preset_name: String,
    slots: Vec<FxSlot>,
}

impl FxChainPreset {
    /// Create an `FxChainPreset` by capturing the current state of a strip.
    #[must_use]
    pub fn from_strip(name: impl Into<String>, strip: &FxStrip) -> Self {
        Self {
            preset_name: name.into(),
            slots: strip.slots().to_vec(),
        }
    }

    /// Returns the name of this preset.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.preset_name
    }

    /// Returns the number of effects captured in this preset.
    #[must_use]
    pub fn effect_count(&self) -> usize {
        self.slots.len()
    }

    /// Restore this preset into a new `FxStrip`.
    #[must_use]
    pub fn to_strip(&self) -> FxStrip {
        FxStrip {
            slots: self.slots.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fx_slot_is_enabled_default() {
        let slot = FxSlot::new("Blur");
        assert!(slot.is_enabled());
    }

    #[test]
    fn test_fx_slot_disable_enable() {
        let mut slot = FxSlot::new("Blur");
        slot.disable();
        assert!(!slot.is_enabled());
        slot.enable();
        assert!(slot.is_enabled());
    }

    #[test]
    fn test_fx_strip_add_effect() {
        let mut strip = FxStrip::new();
        let idx = strip.add_effect(FxSlot::new("Sharpen"));
        assert_eq!(idx, 0);
        let idx2 = strip.add_effect(FxSlot::new("Color"));
        assert_eq!(idx2, 1);
    }

    #[test]
    fn test_fx_strip_active_count_all_enabled() {
        let mut strip = FxStrip::new();
        strip.add_effect(FxSlot::new("A"));
        strip.add_effect(FxSlot::new("B"));
        strip.add_effect(FxSlot::new("C"));
        assert_eq!(strip.active_count(), 3);
    }

    #[test]
    fn test_fx_strip_active_count_after_disable() {
        let mut strip = FxStrip::new();
        strip.add_effect(FxSlot::new("A"));
        strip.add_effect(FxSlot::new("B"));
        strip.disable(0);
        assert_eq!(strip.active_count(), 1);
    }

    #[test]
    fn test_fx_strip_enable_disable_returns_false_oob() {
        let mut strip = FxStrip::new();
        assert!(!strip.enable(0));
        assert!(!strip.disable(0));
    }

    #[test]
    fn test_fx_strip_enable_returns_true() {
        let mut strip = FxStrip::new();
        strip.add_effect(FxSlot::new("X"));
        strip.disable(0);
        assert!(strip.enable(0));
        assert!(strip.slots()[0].is_enabled());
    }

    #[test]
    fn test_fx_strip_len_and_is_empty() {
        let mut strip = FxStrip::new();
        assert!(strip.is_empty());
        strip.add_effect(FxSlot::new("X"));
        assert!(!strip.is_empty());
        assert_eq!(strip.len(), 1);
    }

    #[test]
    fn test_fx_chain_preset_name() {
        let strip = FxStrip::new();
        let preset = FxChainPreset::from_strip("MyPreset", &strip);
        assert_eq!(preset.name(), "MyPreset");
    }

    #[test]
    fn test_fx_chain_preset_effect_count() {
        let mut strip = FxStrip::new();
        strip.add_effect(FxSlot::new("A"));
        strip.add_effect(FxSlot::new("B"));
        let preset = FxChainPreset::from_strip("Two", &strip);
        assert_eq!(preset.effect_count(), 2);
    }

    #[test]
    fn test_fx_chain_preset_to_strip() {
        let mut strip = FxStrip::new();
        strip.add_effect(FxSlot::new("Blur"));
        let preset = FxChainPreset::from_strip("P", &strip);
        let restored = preset.to_strip();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.slots()[0].name, "Blur");
    }

    #[test]
    fn test_fx_strip_disable_all_active_zero() {
        let mut strip = FxStrip::new();
        strip.add_effect(FxSlot::new("A"));
        strip.add_effect(FxSlot::new("B"));
        strip.disable(0);
        strip.disable(1);
        assert_eq!(strip.active_count(), 0);
    }
}
