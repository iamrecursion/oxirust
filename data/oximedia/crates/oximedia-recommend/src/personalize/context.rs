//! Context-aware recommendations.

use serde::{Deserialize, Serialize};

/// User context for personalization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserContext {
    /// Time of day (0-23)
    pub hour: Option<u8>,
    /// Day of week (0-6)
    pub day: Option<u8>,
    /// Device type
    pub device: Option<String>,
    /// Location
    pub location: Option<String>,
    /// User mood/intent
    pub mood: Option<UserMood>,
    /// Social context
    pub social: SocialContext,
}

/// User mood categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UserMood {
    /// Relaxed, leisure time
    Relaxed,
    /// Energetic, active
    Energetic,
    /// Focused, learning
    Focused,
    /// Social, sharing
    Social,
}

/// Social context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SocialContext {
    /// Watching alone or with others
    pub watching_alone: bool,
    /// Number of people watching together
    pub group_size: usize,
}

/// Context processor
pub struct ContextProcessor {
    /// Time-based weights
    time_weights: TimeWeights,
}

/// Time-based recommendation weights
#[derive(Debug, Clone)]
pub struct TimeWeights {
    /// Morning preferences (6-12)
    pub morning: f32,
    /// Afternoon preferences (12-18)
    pub afternoon: f32,
    /// Evening preferences (18-22)
    pub evening: f32,
    /// Night preferences (22-6)
    pub night: f32,
}

impl Default for TimeWeights {
    fn default() -> Self {
        Self {
            morning: 1.0,
            afternoon: 1.0,
            evening: 1.2, // Boost evening viewing
            night: 0.8,
        }
    }
}

impl ContextProcessor {
    /// Create a new context processor
    #[must_use]
    pub fn new() -> Self {
        Self {
            time_weights: TimeWeights::default(),
        }
    }

    /// Calculate context boost for recommendation
    #[must_use]
    pub fn calculate_context_boost(&self, context: &UserContext) -> f32 {
        let mut boost = 1.0;

        // Time-based boost
        if let Some(hour) = context.hour {
            boost *= self.get_time_weight(hour);
        }

        // Device-based boost
        if let Some(ref device) = context.device {
            boost *= self.get_device_weight(device);
        }

        // Social context boost
        if !context.social.watching_alone && context.social.group_size > 1 {
            boost *= 1.1; // Boost content suitable for groups
        }

        boost
    }

    /// Get time-based weight
    fn get_time_weight(&self, hour: u8) -> f32 {
        match hour {
            6..=11 => self.time_weights.morning,
            12..=17 => self.time_weights.afternoon,
            18..=21 => self.time_weights.evening,
            _ => self.time_weights.night,
        }
    }

    /// Get device-based weight
    fn get_device_weight(&self, device: &str) -> f32 {
        match device {
            "mobile" => 0.9, // Prefer shorter content
            "tablet" => 1.0,
            "tv" => 1.2, // Prefer longer, high-quality content
            "desktop" => 1.1,
            _ => 1.0,
        }
    }

    /// Detect user intent from context
    #[must_use]
    pub fn detect_intent(&self, context: &UserContext) -> UserIntent {
        // Simplified intent detection
        if let Some(hour) = context.hour {
            if (6..12).contains(&hour) {
                return UserIntent::QuickView;
            } else if (22..24).contains(&hour) || hour < 6 {
                return UserIntent::Binge;
            }
        }

        if !context.social.watching_alone {
            return UserIntent::Social;
        }

        UserIntent::Browse
    }
}

impl Default for ContextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// User intent categories
#[derive(Debug, Clone, Copy)]
pub enum UserIntent {
    /// Quick viewing session
    QuickView,
    /// Binge watching
    Binge,
    /// Social viewing
    Social,
    /// Casual browsing
    Browse,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_processor() {
        let processor = ContextProcessor::new();
        let mut context = UserContext::default();
        context.hour = Some(20); // Evening

        let boost = processor.calculate_context_boost(&context);
        assert!(boost > 1.0); // Evening should be boosted
    }

    #[test]
    fn test_detect_intent() {
        let processor = ContextProcessor::new();
        let mut context = UserContext::default();
        context.hour = Some(8); // Morning

        let intent = processor.detect_intent(&context);
        assert!(matches!(intent, UserIntent::QuickView));
    }
}
