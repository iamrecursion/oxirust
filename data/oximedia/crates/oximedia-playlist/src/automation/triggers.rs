//! Event triggers for automation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Type of trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerType {
    /// Time-based trigger.
    Time {
        /// Scheduled time.
        time: DateTime<Utc>,
    },

    /// GPI (General Purpose Input) trigger.
    Gpi {
        /// GPI port number.
        port: u32,
        /// Expected signal state.
        state: bool,
    },

    /// Manual trigger.
    Manual,

    /// Timecode trigger.
    Timecode {
        /// Timecode value.
        timecode: String,
    },

    /// Item completion trigger.
    ItemComplete {
        /// Item index.
        item_index: usize,
    },

    /// Custom trigger with a name.
    Custom {
        /// Custom trigger name.
        name: String,
    },
}

/// Action to perform when trigger activates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerAction {
    /// Start playlist playback.
    StartPlayback,

    /// Stop playlist playback.
    StopPlayback,

    /// Pause playlist playback.
    PausePlayback,

    /// Skip to next item.
    SkipToNext,

    /// Skip to previous item.
    SkipToPrevious,

    /// Jump to specific item.
    JumpToItem {
        /// Item index to jump to.
        index: usize,
    },

    /// Insert live content.
    InsertLive {
        /// Live source ID.
        source_id: String,
    },

    /// Show graphics overlay.
    ShowGraphics {
        /// Graphics overlay ID.
        graphics_id: String,
    },

    /// Hide graphics overlay.
    HideGraphics {
        /// Graphics overlay ID.
        graphics_id: String,
    },

    /// Execute custom command.
    CustomCommand {
        /// Command string.
        command: String,
    },
}

/// Event trigger configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trigger {
    /// Unique identifier.
    pub id: String,

    /// Trigger name.
    pub name: String,

    /// Type of trigger.
    pub trigger_type: TriggerType,

    /// Action to perform.
    pub action: TriggerAction,

    /// Whether the trigger is enabled.
    pub enabled: bool,

    /// Whether the trigger is one-shot (fires once then disables).
    pub one_shot: bool,
}

impl Trigger {
    /// Creates a new trigger.
    #[must_use]
    pub fn new<S: Into<String>>(name: S, trigger_type: TriggerType, action: TriggerAction) -> Self {
        Self {
            id: generate_id(),
            name: name.into(),
            trigger_type,
            action,
            enabled: true,
            one_shot: false,
        }
    }

    /// Makes this trigger one-shot.
    #[must_use]
    pub const fn as_one_shot(mut self) -> Self {
        self.one_shot = true;
        self
    }

    /// Disables this trigger.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enables this trigger.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

/// Manager for automation triggers.
#[derive(Debug, Default)]
pub struct TriggerManager {
    triggers: Vec<Trigger>,
}

impl TriggerManager {
    /// Creates a new trigger manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a trigger.
    pub fn add_trigger(&mut self, trigger: Trigger) {
        self.triggers.push(trigger);
    }

    /// Removes a trigger by ID.
    pub fn remove_trigger(&mut self, trigger_id: &str) {
        self.triggers.retain(|t| t.id != trigger_id);
    }

    /// Gets all enabled triggers.
    #[must_use]
    pub fn get_enabled_triggers(&self) -> Vec<&Trigger> {
        self.triggers.iter().filter(|t| t.enabled).collect()
    }

    /// Evaluates time-based triggers at a specific time.
    #[must_use]
    pub fn evaluate_time_triggers(&self, now: DateTime<Utc>) -> Vec<&Trigger> {
        self.triggers
            .iter()
            .filter(|t| {
                t.enabled
                    && matches!(
                        &t.trigger_type,
                        TriggerType::Time { time } if *time <= now
                    )
            })
            .collect()
    }

    /// Fires a trigger and handles one-shot behavior.
    pub fn fire_trigger(&mut self, trigger_id: &str) {
        if let Some(trigger) = self.triggers.iter_mut().find(|t| t.id == trigger_id) {
            if trigger.one_shot {
                trigger.enabled = false;
            }
        }
    }

    /// Returns the number of triggers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triggers.len()
    }

    /// Returns true if there are no triggers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triggers.is_empty()
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("trigger_{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_creation() {
        let trigger = Trigger::new(
            "start_at_9am",
            TriggerType::Time { time: Utc::now() },
            TriggerAction::StartPlayback,
        );

        assert_eq!(trigger.name, "start_at_9am");
        assert!(trigger.enabled);
    }

    #[test]
    fn test_trigger_manager() {
        let mut manager = TriggerManager::new();
        let trigger = Trigger::new("test", TriggerType::Manual, TriggerAction::StartPlayback);

        manager.add_trigger(trigger);
        assert_eq!(manager.len(), 1);

        let enabled = manager.get_enabled_triggers();
        assert_eq!(enabled.len(), 1);
    }

    #[test]
    fn test_one_shot_trigger() {
        let mut manager = TriggerManager::new();
        let trigger = Trigger::new(
            "one_shot",
            TriggerType::Manual,
            TriggerAction::StartPlayback,
        )
        .as_one_shot();

        let trigger_id = trigger.id.clone();
        manager.add_trigger(trigger);

        manager.fire_trigger(&trigger_id);
        let enabled = manager.get_enabled_triggers();
        assert_eq!(enabled.len(), 0);
    }
}
