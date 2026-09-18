#![allow(dead_code)]
//! Input source labeling and naming system for the video switcher.
//!
//! Manages human-readable labels for switcher inputs, supporting
//! both short (4-character) labels for tally displays and long labels
//! for multiviewer overlays. Supports label presets and bulk operations.

use std::collections::HashMap;

/// Maximum length for a short label (fits tally display).
const SHORT_LABEL_MAX: usize = 4;

/// Maximum length for a long label.
const LONG_LABEL_MAX: usize = 64;

/// A source label with short and long variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLabel {
    /// Short label (up to 4 characters, for tally/button displays).
    short: String,
    /// Long label (up to 64 characters, for multiviewer/UI).
    long: String,
    /// Optional color coding (RGB hex string).
    color: Option<String>,
    /// Whether this label has been customized by the user.
    custom: bool,
}

impl SourceLabel {
    /// Create a new source label.
    pub fn new(short: &str, long: &str) -> Self {
        let short = if short.len() > SHORT_LABEL_MAX {
            short[..SHORT_LABEL_MAX].to_string()
        } else {
            short.to_string()
        };
        let long = if long.len() > LONG_LABEL_MAX {
            long[..LONG_LABEL_MAX].to_string()
        } else {
            long.to_string()
        };
        Self {
            short,
            long,
            color: None,
            custom: true,
        }
    }

    /// Create a default label for an input number.
    pub fn default_for_input(input_id: usize) -> Self {
        let short = format!("In{}", input_id + 1);
        let short = if short.len() > SHORT_LABEL_MAX {
            short[..SHORT_LABEL_MAX].to_string()
        } else {
            short
        };
        let long = format!("Input {}", input_id + 1);
        Self {
            short,
            long,
            color: None,
            custom: false,
        }
    }

    /// Get the short label.
    pub fn short(&self) -> &str {
        &self.short
    }

    /// Get the long label.
    pub fn long(&self) -> &str {
        &self.long
    }

    /// Set the short label (truncated to 4 characters).
    pub fn set_short(&mut self, label: &str) {
        self.short = if label.len() > SHORT_LABEL_MAX {
            label[..SHORT_LABEL_MAX].to_string()
        } else {
            label.to_string()
        };
        self.custom = true;
    }

    /// Set the long label (truncated to 64 characters).
    pub fn set_long(&mut self, label: &str) {
        self.long = if label.len() > LONG_LABEL_MAX {
            label[..LONG_LABEL_MAX].to_string()
        } else {
            label.to_string()
        };
        self.custom = true;
    }

    /// Get the color.
    pub fn color(&self) -> Option<&str> {
        self.color.as_deref()
    }

    /// Set the color (RGB hex string, e.g., "#FF0000").
    pub fn set_color(&mut self, color: &str) {
        self.color = Some(color.to_string());
    }

    /// Clear the color.
    pub fn clear_color(&mut self) {
        self.color = None;
    }

    /// Whether this label was customized by the user.
    pub fn is_custom(&self) -> bool {
        self.custom
    }

    /// Reset to default for the given input id.
    pub fn reset_to_default(&mut self, input_id: usize) {
        let default = Self::default_for_input(input_id);
        self.short = default.short;
        self.long = default.long;
        self.color = None;
        self.custom = false;
    }
}

/// Label preset for common production scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelPreset {
    /// Generic numbered inputs.
    Generic,
    /// News production layout.
    News,
    /// Sports production layout.
    Sports,
    /// Music concert production layout.
    Concert,
    /// Corporate event layout.
    Corporate,
}

/// Source label manager for all switcher inputs.
#[derive(Debug)]
pub struct SourceLabelManager {
    /// Labels indexed by input ID.
    labels: HashMap<usize, SourceLabel>,
    /// Total number of inputs.
    num_inputs: usize,
    /// Currently active preset (if any).
    active_preset: Option<LabelPreset>,
}

impl SourceLabelManager {
    /// Create a new label manager with default labels.
    pub fn new(num_inputs: usize) -> Self {
        let mut labels = HashMap::new();
        for i in 0..num_inputs {
            labels.insert(i, SourceLabel::default_for_input(i));
        }
        Self {
            labels,
            num_inputs,
            active_preset: None,
        }
    }

    /// Get the label for an input.
    pub fn get(&self, input_id: usize) -> Option<&SourceLabel> {
        self.labels.get(&input_id)
    }

    /// Get a mutable label for an input.
    pub fn get_mut(&mut self, input_id: usize) -> Option<&mut SourceLabel> {
        self.labels.get_mut(&input_id)
    }

    /// Set the label for an input.
    pub fn set(&mut self, input_id: usize, label: SourceLabel) {
        if input_id < self.num_inputs {
            self.labels.insert(input_id, label);
            self.active_preset = None;
        }
    }

    /// Get the short label for an input (convenience method).
    pub fn short_label(&self, input_id: usize) -> &str {
        self.labels
            .get(&input_id)
            .map(|l| l.short())
            .unwrap_or("????")
    }

    /// Get the long label for an input (convenience method).
    pub fn long_label(&self, input_id: usize) -> &str {
        self.labels
            .get(&input_id)
            .map(|l| l.long())
            .unwrap_or("Unknown Input")
    }

    /// Get the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Apply a label preset.
    pub fn apply_preset(&mut self, preset: LabelPreset) {
        let preset_labels = generate_preset_labels(preset, self.num_inputs);
        for (id, label) in preset_labels {
            self.labels.insert(id, label);
        }
        self.active_preset = Some(preset);
    }

    /// Get the currently active preset.
    pub fn active_preset(&self) -> Option<LabelPreset> {
        self.active_preset
    }

    /// Reset all labels to defaults.
    pub fn reset_all(&mut self) {
        for i in 0..self.num_inputs {
            self.labels.insert(i, SourceLabel::default_for_input(i));
        }
        self.active_preset = None;
    }

    /// Get all labels as a sorted vec of (input_id, label) pairs.
    pub fn all_labels(&self) -> Vec<(usize, &SourceLabel)> {
        let mut result: Vec<_> = self.labels.iter().map(|(&k, v)| (k, v)).collect();
        result.sort_by_key(|(k, _)| *k);
        result
    }

    /// Count how many labels have been customized.
    pub fn custom_count(&self) -> usize {
        self.labels.values().filter(|l| l.is_custom()).count()
    }
}

/// Generate preset labels for a given scenario.
fn generate_preset_labels(preset: LabelPreset, num_inputs: usize) -> Vec<(usize, SourceLabel)> {
    let templates: Vec<(&str, &str)> = match preset {
        LabelPreset::Generic => (0..num_inputs)
            .map(|i| {
                // These are static strings we'll handle below
                let _ = i;
                ("", "")
            })
            .collect::<Vec<_>>(), // Will be overridden
        LabelPreset::News => vec![
            ("Anc1", "Anchor Camera 1"),
            ("Anc2", "Anchor Camera 2"),
            ("Wide", "Wide Shot"),
            ("Rem1", "Remote 1"),
            ("Rem2", "Remote 2"),
            ("GFX", "Graphics"),
            ("VTR1", "VTR Playback 1"),
            ("VTR2", "VTR Playback 2"),
        ],
        LabelPreset::Sports => vec![
            ("Cam1", "Camera 1 - Main"),
            ("Cam2", "Camera 2 - Reverse"),
            ("Cam3", "Camera 3 - Beauty"),
            ("Cam4", "Camera 4 - Handheld"),
            ("Rply", "Replay System"),
            ("GFX", "Graphics / Scoreboard"),
            ("Sky", "Skycam"),
            ("VTR", "VTR Playback"),
        ],
        LabelPreset::Concert => vec![
            ("Wide", "Wide Stage Shot"),
            ("Ctr", "Center Stage"),
            ("Lft", "Stage Left"),
            ("Rgt", "Stage Right"),
            ("Crwd", "Crowd Camera"),
            ("Jib", "Jib Camera"),
            ("GFX", "Graphics / Lyrics"),
            ("VTR", "VTR Playback"),
        ],
        LabelPreset::Corporate => vec![
            ("Pres", "Presenter Camera"),
            ("Slid", "Slides / Presentation"),
            ("Aud", "Audience Camera"),
            ("Rem", "Remote Feed"),
            ("Logo", "Logo / Holding Slide"),
            ("GFX", "Graphics"),
            ("VTR", "VTR Playback"),
            ("Cam2", "Camera 2"),
        ],
    };

    let mut result = Vec::new();

    if matches!(preset, LabelPreset::Generic) {
        for i in 0..num_inputs {
            result.push((i, SourceLabel::default_for_input(i)));
        }
    } else {
        for (i, (short, long)) in templates.iter().enumerate() {
            if i >= num_inputs {
                break;
            }
            let mut label = SourceLabel::new(short, long);
            label.custom = false; // Preset labels aren't "custom"
            result.push((i, label));
        }
        // Fill remaining with defaults
        for i in templates.len()..num_inputs {
            result.push((i, SourceLabel::default_for_input(i)));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_label_creation() {
        let label = SourceLabel::new("CAM1", "Camera 1 - Main Stage");
        assert_eq!(label.short(), "CAM1");
        assert_eq!(label.long(), "Camera 1 - Main Stage");
        assert!(label.is_custom());
    }

    #[test]
    fn test_source_label_truncation() {
        let label = SourceLabel::new("TOOLONG", "This label is fine");
        assert_eq!(label.short(), "TOOL");
        assert_eq!(label.long(), "This label is fine");
    }

    #[test]
    fn test_source_label_default() {
        let label = SourceLabel::default_for_input(0);
        assert_eq!(label.short(), "In1");
        assert_eq!(label.long(), "Input 1");
        assert!(!label.is_custom());
    }

    #[test]
    fn test_source_label_default_double_digit() {
        let label = SourceLabel::default_for_input(9);
        assert_eq!(label.short(), "In10");
        assert_eq!(label.long(), "Input 10");
    }

    #[test]
    fn test_source_label_color() {
        let mut label = SourceLabel::new("CAM1", "Camera 1");
        assert!(label.color().is_none());
        label.set_color("#FF0000");
        assert_eq!(label.color(), Some("#FF0000"));
        label.clear_color();
        assert!(label.color().is_none());
    }

    #[test]
    fn test_source_label_set_short() {
        let mut label = SourceLabel::new("OLD", "Old Label");
        label.set_short("NEW");
        assert_eq!(label.short(), "NEW");
    }

    #[test]
    fn test_source_label_set_long() {
        let mut label = SourceLabel::new("CAM", "Old");
        label.set_long("New Long Label");
        assert_eq!(label.long(), "New Long Label");
    }

    #[test]
    fn test_source_label_reset() {
        let mut label = SourceLabel::new("CAM1", "Camera 1");
        label.reset_to_default(2);
        assert_eq!(label.short(), "In3");
        assert_eq!(label.long(), "Input 3");
        assert!(!label.is_custom());
    }

    #[test]
    fn test_label_manager_creation() {
        let mgr = SourceLabelManager::new(8);
        assert_eq!(mgr.num_inputs(), 8);
        assert_eq!(mgr.short_label(0), "In1");
        assert_eq!(mgr.long_label(0), "Input 1");
    }

    #[test]
    fn test_label_manager_set() {
        let mut mgr = SourceLabelManager::new(4);
        mgr.set(0, SourceLabel::new("CAM1", "Camera 1"));
        assert_eq!(mgr.short_label(0), "CAM1");
    }

    #[test]
    fn test_label_manager_unknown_input() {
        let mgr = SourceLabelManager::new(4);
        assert_eq!(mgr.short_label(999), "????");
        assert_eq!(mgr.long_label(999), "Unknown Input");
    }

    #[test]
    fn test_label_manager_apply_news_preset() {
        let mut mgr = SourceLabelManager::new(8);
        mgr.apply_preset(LabelPreset::News);
        assert_eq!(mgr.short_label(0), "Anc1");
        assert_eq!(mgr.long_label(0), "Anchor Camera 1");
        assert_eq!(mgr.active_preset(), Some(LabelPreset::News));
    }

    #[test]
    fn test_label_manager_apply_sports_preset() {
        let mut mgr = SourceLabelManager::new(4);
        mgr.apply_preset(LabelPreset::Sports);
        assert_eq!(mgr.short_label(0), "Cam1");
        // Only 4 inputs, so last 4 of template are skipped
        assert_eq!(mgr.short_label(3), "Cam4");
    }

    #[test]
    fn test_label_manager_reset_all() {
        let mut mgr = SourceLabelManager::new(4);
        mgr.set(0, SourceLabel::new("X", "Custom"));
        mgr.reset_all();
        assert_eq!(mgr.short_label(0), "In1");
        assert!(mgr.active_preset().is_none());
    }

    #[test]
    fn test_label_manager_all_labels() {
        let mgr = SourceLabelManager::new(3);
        let all = mgr.all_labels();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].0, 0);
        assert_eq!(all[1].0, 1);
        assert_eq!(all[2].0, 2);
    }

    #[test]
    fn test_label_manager_custom_count() {
        let mut mgr = SourceLabelManager::new(4);
        assert_eq!(mgr.custom_count(), 0);
        mgr.set(0, SourceLabel::new("CAM", "Camera"));
        assert_eq!(mgr.custom_count(), 1);
    }
}
