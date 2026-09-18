//! Event-driven workflow trigger system for the MAM.
//!
//! Allows rules to be registered that fire workflow actions
//! automatically when asset events occur.

#![allow(dead_code)]

/// Types of events that can occur on a MAM asset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssetEvent {
    /// A new asset was ingested.
    Ingested,
    /// An asset moved to Active state.
    Activated,
    /// An asset was archived.
    Archived,
    /// An asset's metadata was updated.
    MetadataUpdated,
    /// An asset failed quality control.
    QcFailed,
    /// An asset passed quality control.
    QcPassed,
    /// An asset was deleted.
    Deleted,
}

impl AssetEvent {
    /// Returns a short label for the event.
    pub fn label(&self) -> &'static str {
        match self {
            AssetEvent::Ingested => "ingested",
            AssetEvent::Activated => "activated",
            AssetEvent::Archived => "archived",
            AssetEvent::MetadataUpdated => "metadata_updated",
            AssetEvent::QcFailed => "qc_failed",
            AssetEvent::QcPassed => "qc_passed",
            AssetEvent::Deleted => "deleted",
        }
    }

    /// Returns `true` for state-change events (as opposed to metadata events).
    pub fn is_state_change(&self) -> bool {
        !matches!(self, AssetEvent::MetadataUpdated)
    }
}

/// A condition that must be satisfied for a trigger rule to fire.
#[derive(Debug, Clone)]
pub enum TriggerCondition {
    /// Always satisfied.
    Always,
    /// Satisfied when the asset's media type equals the given string.
    MediaTypeIs(String),
    /// Satisfied when the asset ID matches the given prefix.
    AssetIdPrefix(String),
}

impl TriggerCondition {
    /// Evaluates the condition against the provided asset context.
    pub fn evaluate(&self, ctx: &TriggerContext) -> bool {
        match self {
            TriggerCondition::Always => true,
            TriggerCondition::MediaTypeIs(mt) => ctx.media_type.eq_ignore_ascii_case(mt),
            TriggerCondition::AssetIdPrefix(prefix) => ctx.asset_id.starts_with(prefix.as_str()),
        }
    }
}

/// Context provided when an event fires.
#[derive(Debug, Clone)]
pub struct TriggerContext {
    /// The asset that generated the event.
    pub asset_id: String,
    /// Media type label (e.g. "video", "audio").
    pub media_type: String,
    /// Timestamp (ms since epoch) of the event.
    pub timestamp_ms: u64,
    /// Optional actor that caused the event.
    pub actor: Option<String>,
}

impl TriggerContext {
    /// Creates a new trigger context.
    pub fn new(
        asset_id: impl Into<String>,
        media_type: impl Into<String>,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            media_type: media_type.into(),
            timestamp_ms,
            actor: None,
        }
    }

    /// Attaches an actor to the context.
    pub fn with_actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = Some(actor.into());
        self
    }
}

/// An action to execute when a trigger rule fires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerAction {
    /// Start a named workflow.
    StartWorkflow(String),
    /// Send a notification to the named recipient.
    SendNotification(String),
    /// Apply a tag to the asset.
    ApplyTag(String),
    /// Log a message to the audit trail.
    AuditLog(String),
}

impl TriggerAction {
    /// Returns a human-readable description of the action.
    pub fn description(&self) -> String {
        match self {
            TriggerAction::StartWorkflow(n) => format!("Start workflow '{n}'"),
            TriggerAction::SendNotification(r) => format!("Notify '{r}'"),
            TriggerAction::ApplyTag(t) => format!("Apply tag '{t}'"),
            TriggerAction::AuditLog(m) => format!("Audit: {m}"),
        }
    }
}

/// A single trigger rule: event + condition → list of actions.
#[derive(Debug, Clone)]
pub struct TriggerRule {
    /// Unique rule identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Event that activates this rule.
    pub on_event: AssetEvent,
    /// Condition that must be satisfied.
    pub condition: TriggerCondition,
    /// Actions to execute when the rule fires.
    pub actions: Vec<TriggerAction>,
    /// Whether the rule is currently active.
    pub enabled: bool,
}

impl TriggerRule {
    /// Creates a new enabled trigger rule.
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        on_event: AssetEvent,
        condition: TriggerCondition,
        actions: Vec<TriggerAction>,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            on_event,
            condition,
            actions,
            enabled: true,
        }
    }

    /// Disables the rule so it won't fire even if its event and condition match.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Re-enables the rule.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Returns `true` when this rule should fire for the given event and context.
    pub fn should_fire(&self, event: &AssetEvent, ctx: &TriggerContext) -> bool {
        self.enabled && self.on_event == *event && self.condition.evaluate(ctx)
    }
}

/// A record of a rule firing.
#[derive(Debug, Clone)]
pub struct TriggerFiring {
    /// ID of the rule that fired.
    pub rule_id: String,
    /// The event that caused the firing.
    pub event: AssetEvent,
    /// Asset context at the time of firing.
    pub context: TriggerContext,
    /// Actions that were executed.
    pub actions_executed: Vec<TriggerAction>,
}

/// Registry of trigger rules with an in-memory firing log.
#[derive(Debug, Default)]
pub struct TriggerRegistry {
    rules: Vec<TriggerRule>,
    /// Log of all firings.
    firing_log: Vec<TriggerFiring>,
}

impl TriggerRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new rule.
    pub fn register(&mut self, rule: TriggerRule) {
        self.rules.push(rule);
    }

    /// Removes a rule by ID. Returns `true` if found and removed.
    pub fn remove_rule(&mut self, rule_id: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.id != rule_id);
        self.rules.len() < before
    }

    /// Disables the rule with the given ID.
    pub fn disable_rule(&mut self, rule_id: &str) -> bool {
        if let Some(r) = self.rules.iter_mut().find(|r| r.id == rule_id) {
            r.disable();
            true
        } else {
            false
        }
    }

    /// Enables the rule with the given ID.
    pub fn enable_rule(&mut self, rule_id: &str) -> bool {
        if let Some(r) = self.rules.iter_mut().find(|r| r.id == rule_id) {
            r.enable();
            true
        } else {
            false
        }
    }

    /// Fires all matching rules for the given event and context.
    ///
    /// Returns the list of firing records.
    pub fn fire(&mut self, event: AssetEvent, ctx: TriggerContext) -> Vec<TriggerFiring> {
        let mut firings = Vec::new();
        for rule in &self.rules {
            if rule.should_fire(&event, &ctx) {
                let firing = TriggerFiring {
                    rule_id: rule.id.clone(),
                    event: event.clone(),
                    context: ctx.clone(),
                    actions_executed: rule.actions.clone(),
                };
                firings.push(firing);
            }
        }
        self.firing_log.extend(firings.clone());
        firings
    }

    /// Returns the number of rules registered.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Returns the number of enabled rules.
    pub fn enabled_rule_count(&self) -> usize {
        self.rules.iter().filter(|r| r.enabled).count()
    }

    /// Returns all firings in the log.
    pub fn firing_log(&self) -> &[TriggerFiring] {
        &self.firing_log
    }

    /// Returns firings for a specific rule ID.
    pub fn firings_for_rule(&self, rule_id: &str) -> Vec<&TriggerFiring> {
        self.firing_log
            .iter()
            .filter(|f| f.rule_id == rule_id)
            .collect()
    }

    /// Returns all rules that listen for the given event.
    pub fn rules_for_event(&self, event: &AssetEvent) -> Vec<&TriggerRule> {
        self.rules.iter().filter(|r| r.on_event == *event).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ingest_ctx(asset_id: &str) -> TriggerContext {
        TriggerContext::new(asset_id, "video", 1_000)
    }

    fn simple_rule(id: &str, event: AssetEvent) -> TriggerRule {
        TriggerRule::new(
            id,
            "test rule",
            event,
            TriggerCondition::Always,
            vec![TriggerAction::AuditLog(format!("Rule {id} fired"))],
        )
    }

    // --- AssetEvent ---

    #[test]
    fn test_event_label() {
        assert_eq!(AssetEvent::Ingested.label(), "ingested");
        assert_eq!(AssetEvent::Deleted.label(), "deleted");
    }

    #[test]
    fn test_event_is_state_change() {
        assert!(AssetEvent::Ingested.is_state_change());
        assert!(!AssetEvent::MetadataUpdated.is_state_change());
    }

    // --- TriggerCondition ---

    #[test]
    fn test_condition_always() {
        let ctx = ingest_ctx("a1");
        assert!(TriggerCondition::Always.evaluate(&ctx));
    }

    #[test]
    fn test_condition_media_type_match() {
        let ctx = ingest_ctx("a1");
        assert!(TriggerCondition::MediaTypeIs("video".to_string()).evaluate(&ctx));
    }

    #[test]
    fn test_condition_media_type_no_match() {
        let ctx = ingest_ctx("a1");
        assert!(!TriggerCondition::MediaTypeIs("audio".to_string()).evaluate(&ctx));
    }

    #[test]
    fn test_condition_asset_id_prefix_match() {
        let ctx = ingest_ctx("news-001");
        assert!(TriggerCondition::AssetIdPrefix("news-".to_string()).evaluate(&ctx));
    }

    #[test]
    fn test_condition_asset_id_prefix_no_match() {
        let ctx = ingest_ctx("sports-001");
        assert!(!TriggerCondition::AssetIdPrefix("news-".to_string()).evaluate(&ctx));
    }

    // --- TriggerAction ---

    #[test]
    fn test_action_description_workflow() {
        let a = TriggerAction::StartWorkflow("transcode".to_string());
        assert!(a.description().contains("transcode"));
    }

    #[test]
    fn test_action_description_notify() {
        let a = TriggerAction::SendNotification("editor@example.com".to_string());
        assert!(a.description().contains("editor@example.com"));
    }

    // --- TriggerRule ---

    #[test]
    fn test_rule_fires_on_matching_event() {
        let rule = simple_rule("r1", AssetEvent::Ingested);
        let ctx = ingest_ctx("a1");
        assert!(rule.should_fire(&AssetEvent::Ingested, &ctx));
    }

    #[test]
    fn test_rule_does_not_fire_on_wrong_event() {
        let rule = simple_rule("r1", AssetEvent::Ingested);
        let ctx = ingest_ctx("a1");
        assert!(!rule.should_fire(&AssetEvent::Deleted, &ctx));
    }

    #[test]
    fn test_rule_disabled_does_not_fire() {
        let mut rule = simple_rule("r1", AssetEvent::Ingested);
        rule.disable();
        let ctx = ingest_ctx("a1");
        assert!(!rule.should_fire(&AssetEvent::Ingested, &ctx));
    }

    #[test]
    fn test_rule_re_enable() {
        let mut rule = simple_rule("r1", AssetEvent::Ingested);
        rule.disable();
        rule.enable();
        let ctx = ingest_ctx("a1");
        assert!(rule.should_fire(&AssetEvent::Ingested, &ctx));
    }

    // --- TriggerRegistry ---

    #[test]
    fn test_register_and_rule_count() {
        let mut reg = TriggerRegistry::new();
        reg.register(simple_rule("r1", AssetEvent::Ingested));
        reg.register(simple_rule("r2", AssetEvent::Activated));
        assert_eq!(reg.rule_count(), 2);
    }

    #[test]
    fn test_fire_returns_matching_firings() {
        let mut reg = TriggerRegistry::new();
        reg.register(simple_rule("r1", AssetEvent::Ingested));
        reg.register(simple_rule("r2", AssetEvent::Deleted));
        let firings = reg.fire(AssetEvent::Ingested, ingest_ctx("a1"));
        assert_eq!(firings.len(), 1);
        assert_eq!(firings[0].rule_id, "r1");
    }

    #[test]
    fn test_fire_appends_to_log() {
        let mut reg = TriggerRegistry::new();
        reg.register(simple_rule("r1", AssetEvent::Ingested));
        reg.fire(AssetEvent::Ingested, ingest_ctx("a1"));
        reg.fire(AssetEvent::Ingested, ingest_ctx("a2"));
        assert_eq!(reg.firing_log().len(), 2);
    }

    #[test]
    fn test_firings_for_rule() {
        let mut reg = TriggerRegistry::new();
        reg.register(simple_rule("r1", AssetEvent::Ingested));
        reg.fire(AssetEvent::Ingested, ingest_ctx("a1"));
        reg.fire(AssetEvent::Ingested, ingest_ctx("a2"));
        assert_eq!(reg.firings_for_rule("r1").len(), 2);
        assert!(reg.firings_for_rule("ghost").is_empty());
    }

    #[test]
    fn test_remove_rule() {
        let mut reg = TriggerRegistry::new();
        reg.register(simple_rule("r1", AssetEvent::Ingested));
        assert!(reg.remove_rule("r1"));
        assert_eq!(reg.rule_count(), 0);
    }

    #[test]
    fn test_remove_missing_rule() {
        let mut reg = TriggerRegistry::new();
        assert!(!reg.remove_rule("ghost"));
    }

    #[test]
    fn test_disable_and_enable_rule() {
        let mut reg = TriggerRegistry::new();
        reg.register(simple_rule("r1", AssetEvent::Ingested));
        assert!(reg.disable_rule("r1"));
        assert_eq!(reg.enabled_rule_count(), 0);
        assert!(reg.enable_rule("r1"));
        assert_eq!(reg.enabled_rule_count(), 1);
    }

    #[test]
    fn test_rules_for_event() {
        let mut reg = TriggerRegistry::new();
        reg.register(simple_rule("r1", AssetEvent::Ingested));
        reg.register(simple_rule("r2", AssetEvent::Ingested));
        reg.register(simple_rule("r3", AssetEvent::Deleted));
        let rules = reg.rules_for_event(&AssetEvent::Ingested);
        assert_eq!(rules.len(), 2);
    }
}
