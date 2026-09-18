//! Automatic and manual asset tagging with taxonomy support.
//!
//! Provides `TagType`, `AssetTag`, and `TaggingEngine` for attaching
//! structured tags to media assets and running auto-tagging rules.

#![allow(dead_code)]

use std::collections::HashMap;

/// The semantic category of a tag.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TagType {
    /// Subject matter tag (e.g. "sports", "news").
    Subject,
    /// Descriptive keyword (e.g. "outdoor", "night").
    Keyword,
    /// Person name tag.
    Person,
    /// Location-based tag.
    Location,
    /// Mood or tone tag (e.g. "dramatic", "upbeat").
    Mood,
    /// Custom / user-defined category.
    Custom(String),
}

impl TagType {
    /// Return a human-readable label for this tag type.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Subject => "subject",
            Self::Keyword => "keyword",
            Self::Person => "person",
            Self::Location => "location",
            Self::Mood => "mood",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// A single tag attached to an asset.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetTag {
    /// Asset identifier this tag belongs to.
    pub asset_id: u64,
    /// Tag value / text.
    pub value: String,
    /// Semantic type of this tag.
    pub tag_type: TagType,
    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// True when the tag was added by a human reviewer.
    pub manual: bool,
}

impl AssetTag {
    /// Create a new manually assigned tag with full confidence.
    #[must_use]
    pub fn manual(asset_id: u64, value: impl Into<String>, tag_type: TagType) -> Self {
        Self {
            asset_id,
            value: value.into(),
            tag_type,
            confidence: 1.0,
            manual: true,
        }
    }

    /// Create a new automatically generated tag with a given confidence.
    #[must_use]
    pub fn auto(
        asset_id: u64,
        value: impl Into<String>,
        tag_type: TagType,
        confidence: f32,
    ) -> Self {
        Self {
            asset_id,
            value: value.into(),
            tag_type,
            confidence: confidence.clamp(0.0, 1.0),
            manual: false,
        }
    }

    /// Return `true` if the tag confidence meets the given threshold.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// A simple tagging rule applied during auto-tagging.
#[derive(Debug, Clone)]
pub struct TagRule {
    /// Keyword to match in the asset filename / metadata.
    pub keyword: String,
    /// Tag value to emit when the rule matches.
    pub tag_value: String,
    /// Tag type assigned to the emitted tag.
    pub tag_type: TagType,
    /// Confidence of the emitted tag.
    pub confidence: f32,
}

impl TagRule {
    /// Create a new tagging rule.
    #[must_use]
    pub fn new(
        keyword: impl Into<String>,
        tag_value: impl Into<String>,
        tag_type: TagType,
        confidence: f32,
    ) -> Self {
        Self {
            keyword: keyword.into(),
            tag_value: tag_value.into(),
            tag_type,
            confidence,
        }
    }
}

/// Engine that manages tag rules and applies them to assets.
#[derive(Debug, Default)]
pub struct TaggingEngine {
    rules: Vec<TagRule>,
    /// In-memory tag store keyed by asset_id.
    tags: HashMap<u64, Vec<AssetTag>>,
}

impl TaggingEngine {
    /// Create a new, empty tagging engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tagging rule.
    pub fn add_rule(&mut self, rule: TagRule) {
        self.rules.push(rule);
    }

    /// Manually attach a tag to an asset (always stored, regardless of rules).
    pub fn attach_tag(&mut self, tag: AssetTag) {
        self.tags.entry(tag.asset_id).or_default().push(tag);
    }

    /// Run all rules against `hint` (filename / metadata string) for `asset_id`.
    ///
    /// Returns the list of auto-generated tags that were added.
    pub fn auto_tag(&mut self, asset_id: u64, hint: &str) -> Vec<AssetTag> {
        let hint_lower = hint.to_lowercase();
        let mut generated = Vec::new();

        for rule in &self.rules {
            if hint_lower.contains(&rule.keyword.to_lowercase()) {
                let tag = AssetTag::auto(
                    asset_id,
                    rule.tag_value.clone(),
                    rule.tag_type.clone(),
                    rule.confidence,
                );
                generated.push(tag);
            }
        }

        for tag in &generated {
            self.tags.entry(asset_id).or_default().push(tag.clone());
        }

        generated
    }

    /// Return all tags for the given asset.
    #[must_use]
    pub fn tags_for(&self, asset_id: u64) -> &[AssetTag] {
        self.tags.get(&asset_id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Return only tags that meet the confidence threshold.
    #[must_use]
    pub fn confident_tags(&self, asset_id: u64, threshold: f32) -> Vec<&AssetTag> {
        self.tags_for(asset_id)
            .iter()
            .filter(|t| t.is_confident(threshold))
            .collect()
    }

    /// Remove all tags for the given asset.
    pub fn clear_tags(&mut self, asset_id: u64) {
        self.tags.remove(&asset_id);
    }

    /// Return the total number of tags stored across all assets.
    #[must_use]
    pub fn total_tag_count(&self) -> usize {
        self.tags.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_with_rules() -> TaggingEngine {
        let mut e = TaggingEngine::new();
        e.add_rule(TagRule::new("sports", "sports", TagType::Subject, 0.9));
        e.add_rule(TagRule::new("outdoor", "outdoor", TagType::Keyword, 0.8));
        e.add_rule(TagRule::new("london", "London", TagType::Location, 0.95));
        e
    }

    #[test]
    fn test_tag_type_label() {
        assert_eq!(TagType::Subject.label(), "subject");
        assert_eq!(TagType::Keyword.label(), "keyword");
        assert_eq!(TagType::Person.label(), "person");
        assert_eq!(TagType::Location.label(), "location");
        assert_eq!(TagType::Mood.label(), "mood");
        assert_eq!(TagType::Custom("genre".into()).label(), "genre");
    }

    #[test]
    fn test_manual_tag_full_confidence() {
        let tag = AssetTag::manual(1, "news", TagType::Subject);
        assert!(tag.manual);
        assert!((tag.confidence - 1.0).abs() < f32::EPSILON);
        assert_eq!(tag.value, "news");
    }

    #[test]
    fn test_auto_tag_confidence_clamped() {
        let tag = AssetTag::auto(2, "drama", TagType::Mood, 1.5);
        assert!((tag.confidence - 1.0).abs() < f32::EPSILON);
        let tag2 = AssetTag::auto(2, "drama", TagType::Mood, -0.5);
        assert!((tag2.confidence - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_is_confident() {
        let tag = AssetTag::auto(1, "x", TagType::Keyword, 0.7);
        assert!(tag.is_confident(0.5));
        assert!(tag.is_confident(0.7));
        assert!(!tag.is_confident(0.8));
    }

    #[test]
    fn test_auto_tag_matches_keyword() {
        let mut engine = engine_with_rules();
        let tags = engine.auto_tag(10, "sports_highlight_2024.mp4");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].value, "sports");
    }

    #[test]
    fn test_auto_tag_multiple_matches() {
        let mut engine = engine_with_rules();
        let tags = engine.auto_tag(11, "outdoor_sports_event.mov");
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn test_auto_tag_case_insensitive() {
        let mut engine = engine_with_rules();
        let tags = engine.auto_tag(12, "LONDON_TOUR.mp4");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].value, "London");
    }

    #[test]
    fn test_auto_tag_no_match() {
        let mut engine = engine_with_rules();
        let tags = engine.auto_tag(13, "random_video.avi");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_attach_manual_tag() {
        let mut engine = TaggingEngine::new();
        let tag = AssetTag::manual(20, "interview", TagType::Subject);
        engine.attach_tag(tag);
        assert_eq!(engine.tags_for(20).len(), 1);
        assert!(engine.tags_for(20)[0].manual);
    }

    #[test]
    fn test_tags_for_unknown_asset() {
        let engine = TaggingEngine::new();
        assert!(engine.tags_for(999).is_empty());
    }

    #[test]
    fn test_confident_tags_filter() {
        let mut engine = engine_with_rules();
        engine.auto_tag(30, "outdoor sports london");
        let high = engine.confident_tags(30, 0.9);
        // sports=0.9, london=0.95 → both pass; outdoor=0.8 → filtered out
        assert_eq!(high.len(), 2);
    }

    #[test]
    fn test_clear_tags() {
        let mut engine = engine_with_rules();
        engine.auto_tag(40, "sports outdoor");
        assert!(!engine.tags_for(40).is_empty());
        engine.clear_tags(40);
        assert!(engine.tags_for(40).is_empty());
    }

    #[test]
    fn test_total_tag_count() {
        let mut engine = engine_with_rules();
        engine.auto_tag(50, "sports");
        engine.auto_tag(51, "outdoor london");
        assert_eq!(engine.total_tag_count(), 3);
    }

    #[test]
    fn test_tag_rule_new() {
        let rule = TagRule::new("news", "breaking-news", TagType::Subject, 0.85);
        assert_eq!(rule.keyword, "news");
        assert_eq!(rule.tag_value, "breaking-news");
        assert!((rule.confidence - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_engine_default_is_empty() {
        let engine = TaggingEngine::default();
        assert_eq!(engine.total_tag_count(), 0);
    }
}
