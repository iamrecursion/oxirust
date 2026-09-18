#![allow(dead_code)]
//! Safety interlock system for broadcast automation.
//!
//! Interlocks prevent dangerous or invalid automation actions from executing.
//! They enforce rules such as "never go to black air", "never cut audio during
//! a live broadcast", and "always have a valid backup source ready".
//! This module provides a rule-based interlock engine that gates automation
//! actions and can veto or modify them before execution.

use std::collections::HashMap;
use std::fmt;

/// Result of an interlock check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterlockVerdict {
    /// Action is allowed to proceed.
    Allow,
    /// Action is blocked with a reason.
    Block {
        /// Reason the action was blocked.
        reason: String,
    },
    /// Action is allowed but with a warning.
    Warn {
        /// Warning message.
        message: String,
    },
    /// Action should be modified before proceeding.
    Modify {
        /// Description of the required modification.
        modification: String,
    },
}

impl InterlockVerdict {
    /// Check if the verdict allows the action to proceed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow | Self::Warn { .. } | Self::Modify { .. })
    }

    /// Check if the verdict blocks the action.
    pub fn is_blocked(&self) -> bool {
        matches!(self, Self::Block { .. })
    }
}

impl fmt::Display for InterlockVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allow => write!(f, "ALLOW"),
            Self::Block { reason } => write!(f, "BLOCK: {reason}"),
            Self::Warn { message } => write!(f, "WARN: {message}"),
            Self::Modify { modification } => write!(f, "MODIFY: {modification}"),
        }
    }
}

/// Category of interlock rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterlockCategory {
    /// Prevents black/silence on air.
    AirSafety,
    /// Protects against audio issues (clip, silence, phase).
    AudioSafety,
    /// Ensures valid video output at all times.
    VideoSafety,
    /// Device protection (prevent conflicting commands).
    DeviceProtection,
    /// Timing constraints (minimum segment duration, etc.).
    TimingConstraint,
    /// Regulatory compliance (EAS, content ratings).
    Compliance,
    /// Operator safety (confirmation for destructive ops).
    OperatorSafety,
}

impl fmt::Display for InterlockCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AirSafety => write!(f, "Air Safety"),
            Self::AudioSafety => write!(f, "Audio Safety"),
            Self::VideoSafety => write!(f, "Video Safety"),
            Self::DeviceProtection => write!(f, "Device Protection"),
            Self::TimingConstraint => write!(f, "Timing Constraint"),
            Self::Compliance => write!(f, "Compliance"),
            Self::OperatorSafety => write!(f, "Operator Safety"),
        }
    }
}

/// Severity of an interlock rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InterlockSeverity {
    /// Advisory only - log a warning but allow.
    Advisory,
    /// Cautionary - require explicit override.
    Cautionary,
    /// Mandatory - cannot be overridden.
    Mandatory,
}

/// An automation action to be checked against interlocks.
#[derive(Debug, Clone)]
pub struct AutomationAction {
    /// Type of action.
    pub action_type: ActionType,
    /// Channel this action targets.
    pub channel_id: String,
    /// Source involved (if any).
    pub source_id: Option<String>,
    /// Destination involved (if any).
    pub destination_id: Option<String>,
    /// Duration in frames (if applicable).
    pub duration_frames: Option<u64>,
    /// Whether the operator has provided an override.
    pub operator_override: bool,
}

/// Types of automation actions that can be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    /// Switch video source.
    SourceSwitch,
    /// Start playout.
    PlayoutStart,
    /// Stop playout.
    PlayoutStop,
    /// Mute audio.
    AudioMute,
    /// Un-mute audio.
    AudioUnmute,
    /// Insert graphic overlay.
    GraphicInsert,
    /// Remove graphic overlay.
    GraphicRemove,
    /// Start ad break.
    AdBreakStart,
    /// End ad break.
    AdBreakEnd,
    /// Emergency alert insertion.
    EmergencyAlert,
    /// System shutdown.
    SystemShutdown,
}

/// A single interlock rule.
#[derive(Debug, Clone)]
pub struct InterlockRule {
    /// Unique rule identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Category of this rule.
    pub category: InterlockCategory,
    /// Severity level.
    pub severity: InterlockSeverity,
    /// Action types this rule applies to.
    pub applies_to: Vec<ActionType>,
    /// Whether this rule is currently enabled.
    pub enabled: bool,
    /// Description of what the rule checks.
    pub description: String,
}

impl InterlockRule {
    /// Create a new interlock rule.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        category: InterlockCategory,
        severity: InterlockSeverity,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            category,
            severity,
            applies_to: Vec::new(),
            enabled: true,
            description: String::new(),
        }
    }

    /// Add action types this rule applies to.
    pub fn with_applies_to(mut self, actions: Vec<ActionType>) -> Self {
        self.applies_to = actions;
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Check if this rule applies to the given action type.
    pub fn applies_to_action(&self, action_type: ActionType) -> bool {
        self.applies_to.is_empty() || self.applies_to.contains(&action_type)
    }
}

/// Channel state used for interlock evaluation.
#[derive(Debug, Clone)]
pub struct ChannelState {
    /// Current active source.
    pub active_source: Option<String>,
    /// Whether audio is currently live.
    pub audio_live: bool,
    /// Whether video is currently live.
    pub video_live: bool,
    /// Whether an ad break is in progress.
    pub ad_break_active: bool,
    /// Whether an emergency alert is active.
    pub emergency_active: bool,
    /// Backup source available.
    pub backup_source_available: bool,
    /// Current playout state.
    pub playout_active: bool,
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            active_source: None,
            audio_live: false,
            video_live: false,
            ad_break_active: false,
            emergency_active: false,
            backup_source_available: true,
            playout_active: false,
        }
    }
}

/// The interlock engine that evaluates rules against actions.
#[derive(Debug)]
pub struct InterlockEngine {
    /// Registered interlock rules.
    rules: HashMap<String, InterlockRule>,
    /// Channel states for evaluation.
    channel_states: HashMap<String, ChannelState>,
    /// Number of blocks issued.
    block_count: u64,
    /// Number of warnings issued.
    warn_count: u64,
    /// Whether interlocks can be globally bypassed (emergency mode).
    bypass_mode: bool,
}

impl InterlockEngine {
    /// Create a new interlock engine.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            channel_states: HashMap::new(),
            block_count: 0,
            warn_count: 0,
            bypass_mode: false,
        }
    }

    /// Register an interlock rule.
    pub fn add_rule(&mut self, rule: InterlockRule) {
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Remove an interlock rule by ID.
    pub fn remove_rule(&mut self, id: &str) -> Option<InterlockRule> {
        self.rules.remove(id)
    }

    /// Update or insert a channel state.
    pub fn set_channel_state(&mut self, channel_id: impl Into<String>, state: ChannelState) {
        self.channel_states.insert(channel_id.into(), state);
    }

    /// Get channel state.
    pub fn get_channel_state(&self, channel_id: &str) -> Option<&ChannelState> {
        self.channel_states.get(channel_id)
    }

    /// Enable or disable bypass mode.
    pub fn set_bypass_mode(&mut self, bypass: bool) {
        self.bypass_mode = bypass;
    }

    /// Check if bypass mode is active.
    pub fn is_bypass_mode(&self) -> bool {
        self.bypass_mode
    }

    /// Evaluate an action against all applicable interlock rules.
    pub fn evaluate(&mut self, action: &AutomationAction) -> Vec<(String, InterlockVerdict)> {
        if self.bypass_mode {
            return vec![("BYPASS".to_string(), InterlockVerdict::Allow)];
        }

        let channel_state = self.channel_states.get(&action.channel_id).cloned();
        let mut verdicts = Vec::new();

        for rule in self.rules.values() {
            if !rule.enabled || !rule.applies_to_action(action.action_type) {
                continue;
            }

            let verdict = self.evaluate_rule(rule, action, channel_state.as_ref());
            match &verdict {
                InterlockVerdict::Block { .. } => self.block_count += 1,
                InterlockVerdict::Warn { .. } => self.warn_count += 1,
                _ => {}
            }
            verdicts.push((rule.id.clone(), verdict));
        }

        if verdicts.is_empty() {
            verdicts.push(("DEFAULT".to_string(), InterlockVerdict::Allow));
        }
        verdicts
    }

    /// Evaluate a single rule against an action.
    fn evaluate_rule(
        &self,
        rule: &InterlockRule,
        action: &AutomationAction,
        channel_state: Option<&ChannelState>,
    ) -> InterlockVerdict {
        // If operator override is set and severity is not mandatory, allow
        if action.operator_override && rule.severity != InterlockSeverity::Mandatory {
            return InterlockVerdict::Warn {
                message: format!("Rule '{}' overridden by operator", rule.name),
            };
        }

        match rule.category {
            InterlockCategory::AirSafety => self.check_air_safety(action, channel_state),
            InterlockCategory::AudioSafety => self.check_audio_safety(action, channel_state),
            InterlockCategory::TimingConstraint => self.check_timing(action),
            _ => InterlockVerdict::Allow,
        }
    }

    /// Check air safety rules.
    fn check_air_safety(
        &self,
        action: &AutomationAction,
        state: Option<&ChannelState>,
    ) -> InterlockVerdict {
        let Some(state) = state else {
            return InterlockVerdict::Allow;
        };

        // Don't allow stopping playout if no backup and we're live
        if action.action_type == ActionType::PlayoutStop
            && state.video_live
            && !state.backup_source_available
        {
            return InterlockVerdict::Block {
                reason: "Cannot stop playout while on-air with no backup source".to_string(),
            };
        }

        // Don't allow switching to no source
        if action.action_type == ActionType::SourceSwitch
            && action.source_id.is_none()
            && state.video_live
        {
            return InterlockVerdict::Block {
                reason: "Cannot switch to null source while on-air".to_string(),
            };
        }

        InterlockVerdict::Allow
    }

    /// Check audio safety rules.
    fn check_audio_safety(
        &self,
        action: &AutomationAction,
        state: Option<&ChannelState>,
    ) -> InterlockVerdict {
        let Some(state) = state else {
            return InterlockVerdict::Allow;
        };

        if action.action_type == ActionType::AudioMute && state.audio_live && state.emergency_active
        {
            return InterlockVerdict::Block {
                reason: "Cannot mute audio during emergency alert".to_string(),
            };
        }

        InterlockVerdict::Allow
    }

    /// Check timing constraints.
    fn check_timing(&self, action: &AutomationAction) -> InterlockVerdict {
        if let Some(dur) = action.duration_frames {
            if dur == 0 {
                return InterlockVerdict::Block {
                    reason: "Action duration cannot be zero frames".to_string(),
                };
            }
        }
        InterlockVerdict::Allow
    }

    /// Check whether an action is fully allowed (no blocks).
    pub fn is_allowed(&mut self, action: &AutomationAction) -> bool {
        let verdicts = self.evaluate(action);
        verdicts.iter().all(|(_, v)| v.is_allowed())
    }

    /// Get total block count.
    pub fn block_count(&self) -> u64 {
        self.block_count
    }

    /// Get total warning count.
    pub fn warn_count(&self) -> u64 {
        self.warn_count
    }

    /// Get the number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Get all enabled rules.
    pub fn enabled_rules(&self) -> Vec<&InterlockRule> {
        self.rules.values().filter(|r| r.enabled).collect()
    }
}

impl Default for InterlockEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_action(action_type: ActionType, channel_id: &str) -> AutomationAction {
        AutomationAction {
            action_type,
            channel_id: channel_id.to_string(),
            source_id: Some("SRC1".to_string()),
            destination_id: None,
            duration_frames: Some(100),
            operator_override: false,
        }
    }

    #[test]
    fn test_interlock_verdict_display() {
        assert_eq!(InterlockVerdict::Allow.to_string(), "ALLOW");
        let block = InterlockVerdict::Block {
            reason: "test".to_string(),
        };
        assert_eq!(block.to_string(), "BLOCK: test");
    }

    #[test]
    fn test_interlock_verdict_is_allowed() {
        assert!(InterlockVerdict::Allow.is_allowed());
        assert!(InterlockVerdict::Warn {
            message: "x".to_string()
        }
        .is_allowed());
        assert!(!InterlockVerdict::Block {
            reason: "x".to_string()
        }
        .is_allowed());
    }

    #[test]
    fn test_interlock_verdict_is_blocked() {
        assert!(!InterlockVerdict::Allow.is_blocked());
        assert!(InterlockVerdict::Block {
            reason: "x".to_string()
        }
        .is_blocked());
    }

    #[test]
    fn test_interlock_category_display() {
        assert_eq!(InterlockCategory::AirSafety.to_string(), "Air Safety");
        assert_eq!(InterlockCategory::AudioSafety.to_string(), "Audio Safety");
        assert_eq!(InterlockCategory::Compliance.to_string(), "Compliance");
    }

    #[test]
    fn test_interlock_rule_creation() {
        let rule = InterlockRule::new(
            "r1",
            "Air Safety Check",
            InterlockCategory::AirSafety,
            InterlockSeverity::Mandatory,
        )
        .with_applies_to(vec![ActionType::PlayoutStop])
        .with_description("Prevents stopping playout while on-air");
        assert_eq!(rule.id, "r1");
        assert!(rule.enabled);
        assert!(rule.applies_to_action(ActionType::PlayoutStop));
        assert!(!rule.applies_to_action(ActionType::AudioMute));
    }

    #[test]
    fn test_interlock_rule_empty_applies_to() {
        let rule = InterlockRule::new(
            "r2",
            "Global Rule",
            InterlockCategory::OperatorSafety,
            InterlockSeverity::Advisory,
        );
        // Empty applies_to means applies to all actions
        assert!(rule.applies_to_action(ActionType::PlayoutStop));
        assert!(rule.applies_to_action(ActionType::AudioMute));
    }

    #[test]
    fn test_engine_add_remove_rules() {
        let mut engine = InterlockEngine::new();
        let rule = InterlockRule::new(
            "r1",
            "Rule 1",
            InterlockCategory::AirSafety,
            InterlockSeverity::Mandatory,
        );
        engine.add_rule(rule);
        assert_eq!(engine.rule_count(), 1);
        assert!(engine.remove_rule("r1").is_some());
        assert_eq!(engine.rule_count(), 0);
    }

    #[test]
    fn test_engine_air_safety_block() {
        let mut engine = InterlockEngine::new();
        let rule = InterlockRule::new(
            "air1",
            "No Black Air",
            InterlockCategory::AirSafety,
            InterlockSeverity::Mandatory,
        )
        .with_applies_to(vec![ActionType::PlayoutStop]);
        engine.add_rule(rule);

        let mut state = ChannelState::default();
        state.video_live = true;
        state.backup_source_available = false;
        engine.set_channel_state("CH1", state);

        let action = make_action(ActionType::PlayoutStop, "CH1");
        assert!(!engine.is_allowed(&action));
        assert_eq!(engine.block_count(), 1);
    }

    #[test]
    fn test_engine_air_safety_allow_with_backup() {
        let mut engine = InterlockEngine::new();
        let rule = InterlockRule::new(
            "air1",
            "No Black Air",
            InterlockCategory::AirSafety,
            InterlockSeverity::Mandatory,
        )
        .with_applies_to(vec![ActionType::PlayoutStop]);
        engine.add_rule(rule);

        let mut state = ChannelState::default();
        state.video_live = true;
        state.backup_source_available = true;
        engine.set_channel_state("CH1", state);

        let action = make_action(ActionType::PlayoutStop, "CH1");
        assert!(engine.is_allowed(&action));
    }

    #[test]
    fn test_engine_audio_safety_block_during_emergency() {
        let mut engine = InterlockEngine::new();
        let rule = InterlockRule::new(
            "aud1",
            "No Mute During EAS",
            InterlockCategory::AudioSafety,
            InterlockSeverity::Mandatory,
        )
        .with_applies_to(vec![ActionType::AudioMute]);
        engine.add_rule(rule);

        let mut state = ChannelState::default();
        state.audio_live = true;
        state.emergency_active = true;
        engine.set_channel_state("CH1", state);

        let action = make_action(ActionType::AudioMute, "CH1");
        assert!(!engine.is_allowed(&action));
    }

    #[test]
    fn test_engine_bypass_mode() {
        let mut engine = InterlockEngine::new();
        let rule = InterlockRule::new(
            "r1",
            "Block Everything",
            InterlockCategory::AirSafety,
            InterlockSeverity::Mandatory,
        )
        .with_applies_to(vec![ActionType::PlayoutStop]);
        engine.add_rule(rule);

        let mut state = ChannelState::default();
        state.video_live = true;
        state.backup_source_available = false;
        engine.set_channel_state("CH1", state);

        engine.set_bypass_mode(true);
        assert!(engine.is_bypass_mode());
        let action = make_action(ActionType::PlayoutStop, "CH1");
        assert!(engine.is_allowed(&action));
    }

    #[test]
    fn test_engine_timing_constraint_zero_duration() {
        let mut engine = InterlockEngine::new();
        let rule = InterlockRule::new(
            "t1",
            "No Zero Duration",
            InterlockCategory::TimingConstraint,
            InterlockSeverity::Mandatory,
        )
        .with_applies_to(vec![ActionType::AdBreakStart]);
        engine.add_rule(rule);

        let mut action = make_action(ActionType::AdBreakStart, "CH1");
        action.duration_frames = Some(0);
        assert!(!engine.is_allowed(&action));
    }

    #[test]
    fn test_engine_no_rules_allows() {
        let mut engine = InterlockEngine::new();
        let action = make_action(ActionType::SourceSwitch, "CH1");
        assert!(engine.is_allowed(&action));
    }

    #[test]
    fn test_engine_enabled_rules() {
        let mut engine = InterlockEngine::new();
        let mut r1 = InterlockRule::new(
            "r1",
            "Enabled",
            InterlockCategory::AirSafety,
            InterlockSeverity::Mandatory,
        );
        r1.enabled = true;
        let mut r2 = InterlockRule::new(
            "r2",
            "Disabled",
            InterlockCategory::AirSafety,
            InterlockSeverity::Mandatory,
        );
        r2.enabled = false;
        engine.add_rule(r1);
        engine.add_rule(r2);
        assert_eq!(engine.enabled_rules().len(), 1);
    }
}
