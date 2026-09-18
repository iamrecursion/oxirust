//! Temporal expression extraction and processing

use super::entities::{Entity, EntityType};
use crate::error::Result;
use regex::Regex;

/// Advanced temporal expression extractor
pub struct TemporalExtractor {
    patterns: Vec<(String, Regex)>,
}

impl Default for TemporalExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalExtractor {
    /// Create new temporal extractor with predefined patterns
    pub fn new() -> Self {
        let patterns = vec![
            // Relative dates
            (
                "relative_date".to_string(),
                Regex::new(r"(?i)\b(?:yesterday|today|tomorrow|last|next|this)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b").expect("Operation failed")
            ),

            // Time ranges
            (
                "time_range".to_string(),
                Regex::new(
                    r"(?i)\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9]\s*-\s*(?:[01]?[0-9]|2[0-3]):[0-5][0-9]\b",
                )
                .expect("Operation failed"),
            ),

            // Durations
            (
                "duration".to_string(),
                Regex::new(
                    r"(?i)\b(?:\d+)\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
                )
                .expect("Operation failed"),
            ),

            // Seasons and holidays
            (
                "seasonal".to_string(),
                Regex::new(r"(?i)\b(?:spring|summer|fall|autumn|winter|christmas|thanksgiving|easter|halloween|new year)\b").expect("Operation failed")
            ),
        ];

        Self { patterns }
    }

    /// Extract temporal expressions from text
    pub fn extract(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        for (pattern_type, pattern) in &self.patterns {
            for mat in pattern.find_iter(text) {
                entities.push(Entity {
                    text: mat.as_str().to_string(),
                    entity_type: EntityType::Custom(format!("temporal_{pattern_type}")),
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.85,
                });
            }
        }

        Ok(entities)
    }
}
