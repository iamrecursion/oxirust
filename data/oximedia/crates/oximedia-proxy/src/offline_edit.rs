//! Offline editing proxy management.
//!
//! This module provides tools for managing offline editing workflows, including
//! proxy configuration for various NLE software and relinking proxies to
//! high-resolution master files after the offline edit is complete.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Configuration for offline editing proxy generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineEditConfig {
    /// Codec to use for the proxy (e.g. "dnxhd", "prores_proxy", "h264").
    pub proxy_codec: String,
    /// Proxy resolution (width, height).
    pub resolution: (u32, u32),
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Number of audio channels.
    pub audio_channels: u8,
}

impl OfflineEditConfig {
    /// Avid DNxHD proxy preset: 1920×1080, 36 Mbps, stereo.
    #[must_use]
    pub fn avid_dnxhd_proxy() -> Self {
        Self {
            proxy_codec: "dnxhd".to_string(),
            resolution: (1920, 1080),
            bitrate_kbps: 36_000,
            audio_channels: 2,
        }
    }

    /// Apple ProRes Proxy preset: 1920×1080, 45 Mbps, stereo.
    #[must_use]
    pub fn prores_proxy() -> Self {
        Self {
            proxy_codec: "prores_proxy".to_string(),
            resolution: (1920, 1080),
            bitrate_kbps: 45_000,
            audio_channels: 2,
        }
    }

    /// H.264 offline proxy preset: 1280×720, 8 Mbps, stereo.
    #[must_use]
    pub fn h264_proxy() -> Self {
        Self {
            proxy_codec: "h264".to_string(),
            resolution: (1280, 720),
            bitrate_kbps: 8_000,
            audio_channels: 2,
        }
    }
}

/// Represents the relink relationship between a proxy clip and its master file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineRelink {
    /// ID of the proxy clip.
    pub proxy_id: String,
    /// ID of the matched master file.
    pub master_id: String,
    /// Frame offset between proxy and master (positive = master ahead).
    pub offset_frames: i64,
    /// Confidence score for the match (0.0–1.0).
    pub confidence: f32,
}

/// Strategy used when relinking proxies to masters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelinkStrategy {
    /// Require an exact ID/timecode match.
    ExactMatch,
    /// Allow fuzzy matching based on metadata similarity.
    FuzzyMatch,
    /// Require manual approval for every link.
    ManualApproval,
}

impl RelinkStrategy {
    /// Minimum confidence score required to accept a match automatically.
    #[must_use]
    pub fn min_confidence(self) -> f32 {
        match self {
            Self::ExactMatch => 1.0,
            Self::FuzzyMatch => 0.75,
            Self::ManualApproval => 0.0,
        }
    }
}

/// An edit event from the offline edit (a clip usage in the timeline).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditEvent {
    /// Clip identifier (should match a proxy ID).
    pub clip_id: String,
    /// In-point (inclusive) in frames.
    pub in_point: u64,
    /// Out-point (exclusive) in frames.
    pub out_point: u64,
}

/// A master (high-resolution) media file available for relinking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterFile {
    /// Unique identifier.
    pub id: String,
    /// File system path.
    pub path: String,
    /// Total duration in frames.
    pub duration_frames: u64,
}

/// Relinks a proxy edit to master files.
pub struct OfflineEditRelinker {
    /// Strategy to apply when matching.
    pub strategy: RelinkStrategy,
}

impl OfflineEditRelinker {
    /// Create a new relinker with the given strategy.
    #[must_use]
    pub fn new(strategy: RelinkStrategy) -> Self {
        Self { strategy }
    }

    /// Create a relinker with the default `FuzzyMatch` strategy.
    #[must_use]
    pub fn default_fuzzy() -> Self {
        Self::new(RelinkStrategy::FuzzyMatch)
    }

    /// Attempt to relink every edit event to a master file.
    ///
    /// Each `EditEvent.clip_id` is compared against `MasterFile.id`.
    /// - Exact match: clip_id == master_id → confidence 1.0
    /// - Fuzzy match: clip_id starts with master_id prefix → confidence 0.85
    /// - Otherwise: confidence 0.0 (skipped if strategy requires higher confidence)
    #[must_use]
    pub fn relink(
        &self,
        proxy_edit: &[EditEvent],
        master_files: &[MasterFile],
    ) -> Vec<OnlineRelink> {
        let min_conf = self.strategy.min_confidence();
        let mut results = Vec::new();

        for event in proxy_edit {
            // Try exact match first
            if let Some(master) = master_files.iter().find(|m| m.id == event.clip_id) {
                let link = OnlineRelink {
                    proxy_id: event.clip_id.clone(),
                    master_id: master.id.clone(),
                    offset_frames: 0,
                    confidence: 1.0,
                };
                if link.confidence >= min_conf {
                    results.push(link);
                    continue;
                }
            }

            // Fuzzy match: strip suffix digits / common prefix comparison
            let best = master_files.iter().find_map(|m| {
                let conf = compute_fuzzy_confidence(&event.clip_id, &m.id);
                if conf >= min_conf {
                    Some((m, conf))
                } else {
                    None
                }
            });

            if let Some((master, conf)) = best {
                results.push(OnlineRelink {
                    proxy_id: event.clip_id.clone(),
                    master_id: master.id.clone(),
                    offset_frames: 0,
                    confidence: conf,
                });
            }
        }

        results
    }
}

/// Compute a simple fuzzy confidence score between two IDs.
fn compute_fuzzy_confidence(proxy_id: &str, master_id: &str) -> f32 {
    // Both empty → no useful match information
    if proxy_id.is_empty() && master_id.is_empty() {
        return 0.0;
    }
    if proxy_id == master_id {
        return 1.0;
    }
    // Prefix-based fuzzy match: common prefix relative to shorter string,
    // boosted by squaring the ratio so IDs sharing a long base name
    // (e.g. "clip_001_proxy" vs "clip_001_master") get a high score.
    let common = proxy_id
        .chars()
        .zip(master_id.chars())
        .take_while(|(a, b)| a == b)
        .count();
    let min_len = proxy_id.len().min(master_id.len());
    if min_len == 0 {
        return 0.0;
    }
    let ratio = common as f32 / min_len as f32;
    // Use sqrt to boost partial prefix matches, cap at 0.95
    (ratio.sqrt() * 0.95).min(0.95)
}

/// Summary report of a relink operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelinkReport {
    /// Total number of clips processed.
    pub total_clips: u32,
    /// Number of clips successfully relinked.
    pub relinked: u32,
    /// Number of clips that could not be relinked.
    pub failed: u32,
    /// Average confidence across relinked clips.
    pub confidence_avg: f32,
}

impl RelinkReport {
    /// Build a report from a set of edit events and their relink results.
    #[must_use]
    pub fn from_results(events: &[EditEvent], relinks: &[OnlineRelink]) -> Self {
        let total_clips = events.len() as u32;
        let relinked = relinks.len() as u32;
        let failed = total_clips.saturating_sub(relinked);
        let confidence_avg = if relinked == 0 {
            0.0
        } else {
            relinks.iter().map(|r| r.confidence).sum::<f32>() / relinked as f32
        };
        Self {
            total_clips,
            relinked,
            failed,
            confidence_avg,
        }
    }

    /// Fraction of clips that were successfully relinked (0.0–1.0).
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        if self.total_clips == 0 {
            return 1.0;
        }
        self.relinked as f32 / self.total_clips as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avid_dnxhd_proxy_preset() {
        let cfg = OfflineEditConfig::avid_dnxhd_proxy();
        assert_eq!(cfg.proxy_codec, "dnxhd");
        assert_eq!(cfg.resolution, (1920, 1080));
        assert_eq!(cfg.bitrate_kbps, 36_000);
        assert_eq!(cfg.audio_channels, 2);
    }

    #[test]
    fn test_prores_proxy_preset() {
        let cfg = OfflineEditConfig::prores_proxy();
        assert_eq!(cfg.proxy_codec, "prores_proxy");
        assert_eq!(cfg.resolution, (1920, 1080));
        assert_eq!(cfg.bitrate_kbps, 45_000);
    }

    #[test]
    fn test_h264_proxy_preset() {
        let cfg = OfflineEditConfig::h264_proxy();
        assert_eq!(cfg.proxy_codec, "h264");
        assert_eq!(cfg.resolution, (1280, 720));
        assert_eq!(cfg.bitrate_kbps, 8_000);
    }

    #[test]
    fn test_relink_strategy_min_confidence() {
        assert!((RelinkStrategy::ExactMatch.min_confidence() - 1.0).abs() < f32::EPSILON);
        assert!(RelinkStrategy::FuzzyMatch.min_confidence() < 1.0);
        assert!((RelinkStrategy::ManualApproval.min_confidence() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_exact_relink() {
        let relinker = OfflineEditRelinker::new(RelinkStrategy::ExactMatch);
        let events = vec![EditEvent {
            clip_id: "clip_001".to_string(),
            in_point: 0,
            out_point: 100,
        }];
        let masters = vec![MasterFile {
            id: "clip_001".to_string(),
            path: "/media/clip_001.mov".to_string(),
            duration_frames: 200,
        }];
        let relinks = relinker.relink(&events, &masters);
        assert_eq!(relinks.len(), 1);
        assert!((relinks[0].confidence - 1.0).abs() < f32::EPSILON);
        assert_eq!(relinks[0].master_id, "clip_001");
    }

    #[test]
    fn test_exact_relink_no_match() {
        let relinker = OfflineEditRelinker::new(RelinkStrategy::ExactMatch);
        let events = vec![EditEvent {
            clip_id: "clip_999".to_string(),
            in_point: 0,
            out_point: 50,
        }];
        let masters = vec![MasterFile {
            id: "clip_001".to_string(),
            path: "/media/clip_001.mov".to_string(),
            duration_frames: 200,
        }];
        let relinks = relinker.relink(&events, &masters);
        // No exact match, and fuzzy confidence < 1.0 so not accepted
        assert_eq!(relinks.len(), 0);
    }

    #[test]
    fn test_fuzzy_relink_partial_match() {
        let relinker = OfflineEditRelinker::new(RelinkStrategy::FuzzyMatch);
        let events = vec![EditEvent {
            clip_id: "clip_001_proxy".to_string(),
            in_point: 0,
            out_point: 100,
        }];
        let masters = vec![MasterFile {
            id: "clip_001_master".to_string(),
            path: "/media/clip_001_master.mov".to_string(),
            duration_frames: 200,
        }];
        let relinks = relinker.relink(&events, &masters);
        // "clip_001_" prefix is shared; should get a reasonable fuzzy score
        assert_eq!(relinks.len(), 1);
        assert!(relinks[0].confidence >= 0.75);
    }

    #[test]
    fn test_manual_approval_strategy_accepts_all() {
        let relinker = OfflineEditRelinker::new(RelinkStrategy::ManualApproval);
        let events = vec![EditEvent {
            clip_id: "xyz".to_string(),
            in_point: 0,
            out_point: 10,
        }];
        let masters = vec![MasterFile {
            id: "abc".to_string(),
            path: "/media/abc.mov".to_string(),
            duration_frames: 100,
        }];
        // With min_confidence=0, even low-confidence fuzzy matches are accepted
        let relinks = relinker.relink(&events, &masters);
        // "xyz" vs "abc" share no prefix → confidence 0.0, which still >= 0.0
        assert_eq!(relinks.len(), 1);
    }

    #[test]
    fn test_relink_multiple_events() {
        let relinker = OfflineEditRelinker::new(RelinkStrategy::ExactMatch);
        let events = vec![
            EditEvent {
                clip_id: "a".to_string(),
                in_point: 0,
                out_point: 10,
            },
            EditEvent {
                clip_id: "b".to_string(),
                in_point: 10,
                out_point: 20,
            },
            EditEvent {
                clip_id: "c".to_string(),
                in_point: 20,
                out_point: 30,
            },
        ];
        let masters = vec![
            MasterFile {
                id: "a".to_string(),
                path: "/a.mov".to_string(),
                duration_frames: 50,
            },
            MasterFile {
                id: "b".to_string(),
                path: "/b.mov".to_string(),
                duration_frames: 50,
            },
        ];
        let relinks = relinker.relink(&events, &masters);
        assert_eq!(relinks.len(), 2); // "c" has no master
    }

    #[test]
    fn test_relink_report_success_rate() {
        let events = vec![
            EditEvent {
                clip_id: "a".to_string(),
                in_point: 0,
                out_point: 10,
            },
            EditEvent {
                clip_id: "b".to_string(),
                in_point: 10,
                out_point: 20,
            },
        ];
        let relinks = vec![OnlineRelink {
            proxy_id: "a".to_string(),
            master_id: "a".to_string(),
            offset_frames: 0,
            confidence: 1.0,
        }];
        let report = RelinkReport::from_results(&events, &relinks);
        assert_eq!(report.total_clips, 2);
        assert_eq!(report.relinked, 1);
        assert_eq!(report.failed, 1);
        assert!((report.success_rate() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_relink_report_empty() {
        let report = RelinkReport::from_results(&[], &[]);
        assert_eq!(report.total_clips, 0);
        assert!((report.success_rate() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_relink_report_confidence_avg() {
        let events = vec![
            EditEvent {
                clip_id: "a".to_string(),
                in_point: 0,
                out_point: 5,
            },
            EditEvent {
                clip_id: "b".to_string(),
                in_point: 5,
                out_point: 10,
            },
        ];
        let relinks = vec![
            OnlineRelink {
                proxy_id: "a".to_string(),
                master_id: "a".to_string(),
                offset_frames: 0,
                confidence: 1.0,
            },
            OnlineRelink {
                proxy_id: "b".to_string(),
                master_id: "b".to_string(),
                offset_frames: 0,
                confidence: 0.5,
            },
        ];
        let report = RelinkReport::from_results(&events, &relinks);
        assert!((report.confidence_avg - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_fuzzy_confidence_identical() {
        assert!((compute_fuzzy_confidence("abc", "abc") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_fuzzy_confidence_partial() {
        let c = compute_fuzzy_confidence("abcXXX", "abcYYY");
        // 3 common out of 6 max → 3/6 * 0.95 ≈ 0.475
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_compute_fuzzy_confidence_empty() {
        assert!((compute_fuzzy_confidence("", "") - 0.0).abs() < f32::EPSILON);
    }
}
