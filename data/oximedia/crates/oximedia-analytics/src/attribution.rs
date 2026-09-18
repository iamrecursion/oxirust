//! Watch time attribution: allocate total engagement credit to content segments.
//!
//! Attribution models answer the question: "which parts of a piece of content
//! are most responsible for viewers watching all the way through?"
//!
//! Three models are provided:
//!
//! * **Uniform** — every watched second contributes equally.
//! * **Position-weighted** — later segments receive proportionally more credit
//!   (reflects the idea that retaining a viewer to the end is harder).
//! * **Engagement-weighted** — segments where re-watch or re-visit behaviour
//!   is detected receive a bonus.
//!
//! All credit values are normalised so that the total across all segments sums
//! to 1.0 (within floating-point precision).

use crate::error::AnalyticsError;
use crate::retention::ContentSegment;
use crate::session::{build_playback_map, ViewerSession};

// ─── Attribution model ────────────────────────────────────────────────────────

/// How to weight watched seconds when computing attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributionModel {
    /// Every watched second counts equally.
    Uniform,
    /// Seconds watched later in the content receive more credit.
    /// Credit for second s = (s + 1) / total_seconds.
    PositionWeighted,
    /// Seconds that are re-watched (watched by proportionally more viewers
    /// than the preceding second) receive a bonus multiplier of 2×.
    EngagementWeighted,
}

/// Attribution credit assigned to one content segment.
#[derive(Debug, Clone)]
pub struct SegmentAttribution {
    pub segment_name: String,
    pub start_ms: u64,
    pub end_ms: u64,
    /// Total raw credit before normalisation.
    pub raw_credit: f64,
    /// Normalised credit in [0.0, 1.0]; sums to 1.0 across all segments.
    pub normalised_credit: f64,
    /// Fraction of sessions that watched any part of this segment.
    pub reach_pct: f32,
}

// ─── Core function ────────────────────────────────────────────────────────────

/// Compute watch-time attribution for each content segment.
///
/// # Arguments
///
/// * `sessions`           — viewer sessions to analyse.
/// * `segments`           — ordered segments (chapters/sections) of the content.
/// * `content_duration_ms`— total content duration; sets the playback map size.
/// * `model`              — the attribution model to apply.
///
/// Returns one `SegmentAttribution` per segment, with `normalised_credit`
/// values summing to ≈ 1.0.
///
/// Returns an error if `sessions` or `segments` is empty.
pub fn compute_attribution(
    sessions: &[ViewerSession],
    segments: &[ContentSegment],
    content_duration_ms: u64,
    model: AttributionModel,
) -> Result<Vec<SegmentAttribution>, AnalyticsError> {
    if sessions.is_empty() {
        return Err(AnalyticsError::InsufficientData(
            "attribution requires at least one session".to_string(),
        ));
    }
    if segments.is_empty() {
        return Err(AnalyticsError::ConfigError(
            "attribution requires at least one segment".to_string(),
        ));
    }
    if content_duration_ms == 0 {
        return Err(AnalyticsError::ConfigError(
            "content_duration_ms must be non-zero".to_string(),
        ));
    }

    let total_sec = ((content_duration_ms + 999) / 1000) as usize;

    // Build all playback maps.
    let maps: Vec<_> = sessions
        .iter()
        .map(|s| build_playback_map(s, content_duration_ms))
        .collect();

    // Compute per-second watch counts.
    let mut sec_watch_count = vec![0u32; total_sec];
    for map in &maps {
        for (s, &watched) in map.positions_watched.iter().enumerate() {
            if watched && s < sec_watch_count.len() {
                sec_watch_count[s] += 1;
            }
        }
    }

    // Compute position weights.
    let position_weights: Vec<f64> = (0..total_sec)
        .map(|s| match model {
            AttributionModel::Uniform => 1.0,
            AttributionModel::PositionWeighted => (s + 1) as f64 / total_sec as f64,
            AttributionModel::EngagementWeighted => {
                // Re-watch bonus: if this second has more views than the
                // previous second, it's a "re-watch hotspot".
                let prev = if s > 0 { sec_watch_count[s - 1] } else { 0 };
                let curr = sec_watch_count[s];
                if curr > prev && prev > 0 {
                    2.0 // bonus multiplier
                } else {
                    1.0
                }
            }
        })
        .collect();

    let n_sessions = sessions.len() as f32;
    let mut attributions: Vec<SegmentAttribution> = segments
        .iter()
        .map(|seg| {
            let start_sec = (seg.start_ms / 1000) as usize;
            let end_sec = ((seg.end_ms + 999) / 1000).min(total_sec as u64) as usize;

            let mut raw_credit = 0.0f64;
            let mut reach_viewers = 0u32;

            for s in start_sec..end_sec {
                if s >= sec_watch_count.len() {
                    break;
                }
                let count = sec_watch_count[s] as f64;
                let weight = position_weights.get(s).copied().unwrap_or(1.0);
                raw_credit += count * weight;
                if count > 0.0 && s == start_sec {
                    reach_viewers = sec_watch_count[s];
                }
            }

            let reach_pct = if n_sessions > 0.0 {
                reach_viewers as f32 / n_sessions * 100.0
            } else {
                0.0
            };

            SegmentAttribution {
                segment_name: seg.name.clone(),
                start_ms: seg.start_ms,
                end_ms: seg.end_ms,
                raw_credit,
                normalised_credit: raw_credit, // filled in after normalisation
                reach_pct,
            }
        })
        .collect();

    // Normalise.
    let total_credit: f64 = attributions.iter().map(|a| a.raw_credit).sum();
    if total_credit > 0.0 {
        for a in &mut attributions {
            a.normalised_credit = a.raw_credit / total_credit;
        }
    }

    Ok(attributions)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{PlaybackEvent, ViewerSession};

    fn full_session(id: &str, duration_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "c1".to_string(),
            started_at_ms: 0,
            events: vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::End {
                    position_ms: duration_ms,
                    watch_duration_ms: duration_ms,
                },
            ],
        }
    }

    fn two_equal_segments(duration_ms: u64) -> Vec<ContentSegment> {
        let mid = duration_ms / 2;
        vec![
            ContentSegment {
                name: "first_half".to_string(),
                start_ms: 0,
                end_ms: mid,
            },
            ContentSegment {
                name: "second_half".to_string(),
                start_ms: mid,
                end_ms: duration_ms,
            },
        ]
    }

    // ── uniform model ────────────────────────────────────────────────────────

    #[test]
    fn uniform_equal_segments_equal_credit() {
        let sessions: Vec<_> = (0..5)
            .map(|i| full_session(&format!("s{i}"), 10_000))
            .collect();
        let segs = two_equal_segments(10_000);
        let attrs = compute_attribution(&sessions, &segs, 10_000, AttributionModel::Uniform)
            .expect("compute attribution should succeed");
        assert_eq!(attrs.len(), 2);
        // Both halves watched equally → equal normalised credit.
        let diff = (attrs[0].normalised_credit - attrs[1].normalised_credit).abs();
        assert!(diff < 0.05, "uniform: credits differ by {diff}");
    }

    #[test]
    fn normalised_credit_sums_to_one() {
        let sessions: Vec<_> = (0..3)
            .map(|i| full_session(&format!("s{i}"), 12_000))
            .collect();
        let segs = vec![
            ContentSegment {
                name: "a".to_string(),
                start_ms: 0,
                end_ms: 4_000,
            },
            ContentSegment {
                name: "b".to_string(),
                start_ms: 4_000,
                end_ms: 8_000,
            },
            ContentSegment {
                name: "c".to_string(),
                start_ms: 8_000,
                end_ms: 12_000,
            },
        ];
        for model in [
            AttributionModel::Uniform,
            AttributionModel::PositionWeighted,
            AttributionModel::EngagementWeighted,
        ] {
            let attrs = compute_attribution(&sessions, &segs, 12_000, model)
                .expect("compute attribution should succeed");
            let total: f64 = attrs.iter().map(|a| a.normalised_credit).sum();
            assert!((total - 1.0).abs() < 1e-9, "{model:?}: sum={total}");
        }
    }

    // ── position-weighted model ──────────────────────────────────────────────

    #[test]
    fn position_weighted_second_half_gets_more_credit() {
        let sessions: Vec<_> = (0..5)
            .map(|i| full_session(&format!("s{i}"), 10_000))
            .collect();
        let segs = two_equal_segments(10_000);
        let attrs =
            compute_attribution(&sessions, &segs, 10_000, AttributionModel::PositionWeighted)
                .expect("value should be present should succeed");
        assert!(
            attrs[1].normalised_credit > attrs[0].normalised_credit,
            "second half should get more credit in position-weighted model"
        );
    }

    // ── error handling ───────────────────────────────────────────────────────

    #[test]
    fn attribution_empty_sessions_error() {
        let segs = two_equal_segments(10_000);
        assert!(compute_attribution(&[], &segs, 10_000, AttributionModel::Uniform).is_err());
    }

    #[test]
    fn attribution_empty_segments_error() {
        let sessions = vec![full_session("s1", 10_000)];
        assert!(compute_attribution(&sessions, &[], 10_000, AttributionModel::Uniform).is_err());
    }

    #[test]
    fn attribution_zero_duration_error() {
        let sessions = vec![full_session("s1", 10_000)];
        let segs = two_equal_segments(10_000);
        assert!(compute_attribution(&sessions, &segs, 0, AttributionModel::Uniform).is_err());
    }
}
