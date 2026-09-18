//! Ad/interstitial insertion management for broadcast playlists.
//!
//! This module provides tools for planning and filling interstitial slots
//! (advertisements, bumpers, promos, etc.) within a broadcast schedule.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Type of interstitial content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InterstitialType {
    /// Paid advertisement.
    Advertisement,
    /// Short bumper (e.g., "Back in a moment").
    Bumper,
    /// Promotional spot for upcoming content.
    Promo,
    /// Station identification clip.
    Ident,
    /// Placeholder slate / holding frame.
    Slate,
    /// Station ID (e.g., legal ID segment).
    StationId,
}

impl InterstitialType {
    /// Return the typical duration in seconds for this interstitial type.
    #[must_use]
    pub const fn typical_duration_secs(&self) -> u32 {
        match self {
            Self::Advertisement => 30,
            Self::Bumper => 5,
            Self::Promo => 20,
            Self::Ident => 10,
            Self::Slate => 15,
            Self::StationId => 8,
        }
    }
}

/// A single interstitial slot within a playlist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterstitialSlot {
    /// Position within the content, in milliseconds from the start.
    pub position_ms: u64,
    /// The type of interstitial for this slot.
    pub slot_type: InterstitialType,
    /// Maximum allowed duration for content placed in this slot, in seconds.
    pub max_duration_secs: u32,
    /// Whether this slot has been filled with content.
    pub filled: bool,
}

impl InterstitialSlot {
    /// Create a new unfilled interstitial slot.
    #[must_use]
    pub fn new(position_ms: u64, slot_type: InterstitialType, max_duration_secs: u32) -> Self {
        Self {
            position_ms,
            slot_type,
            max_duration_secs,
            filled: false,
        }
    }
}

/// Policy governing how interstitial slots are created for a piece of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterstitialPolicy {
    /// Insert a pre-roll slot at the beginning of the content.
    pub pre_roll: bool,
    /// Interval between mid-roll slots, in seconds. `0` disables mid-rolls.
    pub mid_roll_interval_secs: u32,
    /// Insert a post-roll slot at the end of the content.
    pub post_roll: bool,
    /// Maximum number of interstitial slots allowed per hour of content.
    pub max_per_hour: u32,
}

impl InterstitialPolicy {
    /// Create a default ad-supported policy.
    #[must_use]
    pub fn default_ad_supported() -> Self {
        Self {
            pre_roll: true,
            mid_roll_interval_secs: 900, // every 15 minutes
            post_roll: false,
            max_per_hour: 6,
        }
    }

    /// Create a policy with no interstitials.
    #[must_use]
    pub fn ad_free() -> Self {
        Self {
            pre_roll: false,
            mid_roll_interval_secs: 0,
            post_roll: false,
            max_per_hour: 0,
        }
    }
}

impl Default for InterstitialPolicy {
    fn default() -> Self {
        Self::default_ad_supported()
    }
}

/// Plans interstitial slots for a given piece of content.
pub struct InterstitialScheduler;

impl InterstitialScheduler {
    /// Plan interstitial slots for content of the given duration.
    ///
    /// Slots are distributed according to `policy`. The number of mid-roll
    /// slots is capped at `policy.max_per_hour` (prorated for the content
    /// duration) minus any pre/post-roll slots.
    #[must_use]
    pub fn plan(content_duration_ms: u64, policy: &InterstitialPolicy) -> Vec<InterstitialSlot> {
        let mut slots: Vec<InterstitialSlot> = Vec::new();

        // Pre-roll
        if policy.pre_roll {
            slots.push(InterstitialSlot::new(
                0,
                InterstitialType::Advertisement,
                30,
            ));
        }

        // Mid-rolls
        if policy.mid_roll_interval_secs > 0 && content_duration_ms > 0 {
            let interval_ms = (policy.mid_roll_interval_secs as u64) * 1_000;
            let content_hours = content_duration_ms as f64 / 3_600_000.0;
            let max_midrolls = ((policy.max_per_hour as f64 * content_hours) as u32)
                .saturating_sub(u32::from(policy.pre_roll))
                .saturating_sub(u32::from(policy.post_roll));

            let mut pos_ms = interval_ms;
            let mut count = 0u32;
            while pos_ms < content_duration_ms && count < max_midrolls {
                slots.push(InterstitialSlot::new(
                    pos_ms,
                    InterstitialType::Advertisement,
                    30,
                ));
                pos_ms += interval_ms;
                count += 1;
            }
        }

        // Post-roll
        if policy.post_roll {
            slots.push(InterstitialSlot::new(
                content_duration_ms,
                InterstitialType::Advertisement,
                30,
            ));
        }

        // Sort by position
        slots.sort_by_key(|s| s.position_ms);
        slots
    }
}

/// Fills interstitial slots with advertisement content.
pub struct InterstitialFiller;

impl InterstitialFiller {
    /// Attempt to fill a slot with content of the given duration.
    ///
    /// Returns `true` if the slot was successfully filled, `false` if the
    /// provided duration exceeds the slot's maximum allowed duration.
    pub fn fill_slot(slot: &mut InterstitialSlot, ad_duration_secs: u32) -> bool {
        if ad_duration_secs <= slot.max_duration_secs {
            slot.filled = true;
            true
        } else {
            false
        }
    }
}

/// An ad break containing one or more advertisement assets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdBreak {
    /// The interstitial slot this break occupies.
    pub slot: InterstitialSlot,
    /// Ordered list of ad asset identifiers (e.g. file paths or IDs).
    pub ads: Vec<String>,
    /// Total duration of all ads combined, in seconds.
    pub total_duration_secs: u32,
}

impl AdBreak {
    /// Create a new empty ad break for a slot.
    #[must_use]
    pub fn new(slot: InterstitialSlot) -> Self {
        Self {
            slot,
            ads: Vec::new(),
            total_duration_secs: 0,
        }
    }

    /// Add an advertisement to this break.
    pub fn add_ad(&mut self, ad_id: String, duration_secs: u32) {
        self.ads.push(ad_id);
        self.total_duration_secs += duration_secs;
    }

    /// Returns `true` if the total ad duration exceeds the slot's maximum.
    #[must_use]
    pub fn is_overfull(&self) -> bool {
        self.total_duration_secs > self.slot.max_duration_secs
    }

    /// Returns `true` if this break has no ads.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ads.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interstitial_type_typical_duration() {
        assert_eq!(InterstitialType::Advertisement.typical_duration_secs(), 30);
        assert_eq!(InterstitialType::Bumper.typical_duration_secs(), 5);
        assert_eq!(InterstitialType::Promo.typical_duration_secs(), 20);
        assert_eq!(InterstitialType::Ident.typical_duration_secs(), 10);
        assert_eq!(InterstitialType::Slate.typical_duration_secs(), 15);
        assert_eq!(InterstitialType::StationId.typical_duration_secs(), 8);
    }

    #[test]
    fn test_interstitial_slot_new() {
        let slot = InterstitialSlot::new(60_000, InterstitialType::Advertisement, 30);
        assert_eq!(slot.position_ms, 60_000);
        assert_eq!(slot.max_duration_secs, 30);
        assert!(!slot.filled);
    }

    #[test]
    fn test_policy_ad_free() {
        let policy = InterstitialPolicy::ad_free();
        assert!(!policy.pre_roll);
        assert!(!policy.post_roll);
        assert_eq!(policy.mid_roll_interval_secs, 0);
        assert_eq!(policy.max_per_hour, 0);
    }

    #[test]
    fn test_scheduler_plan_pre_post_roll() {
        let policy = InterstitialPolicy {
            pre_roll: true,
            mid_roll_interval_secs: 0,
            post_roll: true,
            max_per_hour: 4,
        };
        let content_ms = 3_600_000u64; // 1 hour
        let slots = InterstitialScheduler::plan(content_ms, &policy);
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].position_ms, 0);
        assert_eq!(slots[1].position_ms, content_ms);
    }

    #[test]
    fn test_scheduler_plan_no_interstitials() {
        let policy = InterstitialPolicy::ad_free();
        let slots = InterstitialScheduler::plan(3_600_000, &policy);
        assert!(slots.is_empty());
    }

    #[test]
    fn test_scheduler_plan_midrolls_evenly_distributed() {
        let policy = InterstitialPolicy {
            pre_roll: false,
            mid_roll_interval_secs: 900, // 15 minutes
            post_roll: false,
            max_per_hour: 4,
        };
        // 1 hour content
        let slots = InterstitialScheduler::plan(3_600_000, &policy);
        // Expect mid-rolls at 15m, 30m, 45m (3 slots, since max_per_hour=4 and
        // positions 15m,30m,45m are < 3_600_000)
        assert!(!slots.is_empty());
        // First mid-roll at 15 minutes
        assert_eq!(slots[0].position_ms, 900_000);
    }

    #[test]
    fn test_scheduler_plan_sorted_by_position() {
        let policy = InterstitialPolicy {
            pre_roll: true,
            mid_roll_interval_secs: 600,
            post_roll: true,
            max_per_hour: 10,
        };
        let slots = InterstitialScheduler::plan(3_600_000, &policy);
        for i in 1..slots.len() {
            assert!(slots[i].position_ms >= slots[i - 1].position_ms);
        }
    }

    #[test]
    fn test_filler_fill_slot_success() {
        let mut slot = InterstitialSlot::new(0, InterstitialType::Advertisement, 30);
        let result = InterstitialFiller::fill_slot(&mut slot, 30);
        assert!(result);
        assert!(slot.filled);
    }

    #[test]
    fn test_filler_fill_slot_failure_overfull() {
        let mut slot = InterstitialSlot::new(0, InterstitialType::Advertisement, 30);
        let result = InterstitialFiller::fill_slot(&mut slot, 60);
        assert!(!result);
        assert!(!slot.filled);
    }

    #[test]
    fn test_ad_break_is_overfull() {
        let slot = InterstitialSlot::new(0, InterstitialType::Advertisement, 30);
        let mut ad_break = AdBreak::new(slot);
        ad_break.add_ad("ad_001.mp4".to_string(), 15);
        assert!(!ad_break.is_overfull());
        ad_break.add_ad("ad_002.mp4".to_string(), 20);
        assert!(ad_break.is_overfull());
    }

    #[test]
    fn test_ad_break_is_empty() {
        let slot = InterstitialSlot::new(0, InterstitialType::Advertisement, 30);
        let ad_break = AdBreak::new(slot);
        assert!(ad_break.is_empty());
    }

    #[test]
    fn test_ad_break_add_multiple_ads() {
        let slot = InterstitialSlot::new(0, InterstitialType::Advertisement, 90);
        let mut ad_break = AdBreak::new(slot);
        ad_break.add_ad("ad_001.mp4".to_string(), 30);
        ad_break.add_ad("ad_002.mp4".to_string(), 30);
        assert_eq!(ad_break.ads.len(), 2);
        assert_eq!(ad_break.total_duration_secs, 60);
        assert!(!ad_break.is_overfull());
    }

    #[test]
    fn test_interstitial_slot_bumper() {
        let slot = InterstitialSlot::new(1800_000, InterstitialType::Bumper, 10);
        assert_eq!(slot.slot_type, InterstitialType::Bumper);
        assert_eq!(slot.max_duration_secs, 10);
    }
}
