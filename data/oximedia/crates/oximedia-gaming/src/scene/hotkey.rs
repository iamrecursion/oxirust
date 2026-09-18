//! Hotkey binding for scene switching.

use crate::GamingResult;

/// Hotkey manager.
pub struct HotkeyManager {
    hotkeys: Vec<Hotkey>,
}

/// Hotkey definition.
#[derive(Debug, Clone)]
pub struct Hotkey {
    /// Key combination
    pub key: String,
    /// Action
    pub action: HotkeyAction,
}

/// Hotkey action.
#[derive(Debug, Clone)]
pub enum HotkeyAction {
    /// Switch to scene
    SwitchScene(String),
    /// Start streaming
    StartStream,
    /// Stop streaming
    StopStream,
    /// Start recording
    StartRecording,
    /// Stop recording
    StopRecording,
    /// Save replay
    SaveReplay,
}

impl HotkeyManager {
    /// Create a new hotkey manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hotkeys: Vec::new(),
        }
    }

    /// Register a hotkey.
    pub fn register(&mut self, hotkey: Hotkey) {
        self.hotkeys.push(hotkey);
    }

    /// Unregister a hotkey.
    pub fn unregister(&mut self, key: &str) {
        self.hotkeys.retain(|h| h.key != key);
    }

    /// Process key press.
    pub fn process_key(&self, _key: &str) -> GamingResult<Option<HotkeyAction>> {
        Ok(None)
    }

    /// Get hotkey count.
    #[must_use]
    pub fn hotkey_count(&self) -> usize {
        self.hotkeys.len()
    }
}

impl Default for HotkeyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hotkey_manager_creation() {
        let manager = HotkeyManager::new();
        assert_eq!(manager.hotkey_count(), 0);
    }

    #[test]
    fn test_register_hotkey() {
        let mut manager = HotkeyManager::new();
        manager.register(Hotkey {
            key: "F1".to_string(),
            action: HotkeyAction::SwitchScene("Gameplay".to_string()),
        });
        assert_eq!(manager.hotkey_count(), 1);
    }
}
