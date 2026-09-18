//! Analysis and statistics for conform sessions.

use crate::exporters::report::MatchReport;
use crate::timeline::Timeline;
use crate::types::{ClipMatch, FrameRate, MediaFile};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Timeline statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineStatistics {
    /// Total duration in seconds.
    pub duration_seconds: f64,
    /// Total number of clips.
    pub clip_count: usize,
    /// Number of video clips.
    pub video_clip_count: usize,
    /// Number of audio clips.
    pub audio_clip_count: usize,
    /// Number of tracks.
    pub track_count: usize,
    /// Average clip duration in seconds.
    pub avg_clip_duration: f64,
    /// Shortest clip duration in seconds.
    pub min_clip_duration: f64,
    /// Longest clip duration in seconds.
    pub max_clip_duration: f64,
    /// Total gaps in seconds.
    pub total_gaps: f64,
    /// Number of transitions.
    pub transition_count: usize,
}

/// Match quality statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchQualityStats {
    /// Average match score.
    pub avg_score: f64,
    /// Minimum match score.
    pub min_score: f64,
    /// Maximum match score.
    pub max_score: f64,
    /// Number of high-quality matches (>0.9).
    pub high_quality_count: usize,
    /// Number of medium-quality matches (0.7-0.9).
    pub medium_quality_count: usize,
    /// Number of low-quality matches (<0.7).
    pub low_quality_count: usize,
    /// Match method distribution.
    pub method_distribution: HashMap<String, usize>,
}

/// Media statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaStatistics {
    /// Total number of media files.
    pub total_files: usize,
    /// Total size in bytes.
    pub total_size: u64,
    /// Average file size in bytes.
    pub avg_size: u64,
    /// Format distribution.
    pub format_distribution: HashMap<String, usize>,
    /// Resolution distribution.
    pub resolution_distribution: HashMap<String, usize>,
    /// Frame rate distribution.
    pub fps_distribution: HashMap<String, usize>,
}

/// Timeline analyzer.
pub struct TimelineAnalyzer;

impl TimelineAnalyzer {
    /// Analyze a timeline.
    #[must_use]
    pub fn analyze(timeline: &Timeline) -> TimelineStatistics {
        let duration_seconds = timeline.duration_seconds();

        let mut all_clips = Vec::new();
        for track in &timeline.video_tracks {
            all_clips.extend(track.clips.iter());
        }
        for track in &timeline.audio_tracks {
            all_clips.extend(track.clips.iter());
        }

        let clip_count = all_clips.len();
        let video_clip_count = timeline.video_tracks.iter().map(|t| t.clips.len()).sum();
        let audio_clip_count = timeline.audio_tracks.iter().map(|t| t.clips.len()).sum();

        let track_count = timeline.track_count();

        let clip_durations: Vec<f64> = all_clips.iter().map(|c| c.duration_seconds()).collect();

        let avg_clip_duration = if clip_durations.is_empty() {
            0.0
        } else {
            clip_durations.iter().sum::<f64>() / clip_durations.len() as f64
        };

        let min_clip_duration = clip_durations.iter().copied().fold(f64::INFINITY, f64::min);
        let max_clip_duration = clip_durations
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let total_gaps = Self::calculate_total_gaps(timeline);

        let transition_count = timeline
            .video_tracks
            .iter()
            .chain(timeline.audio_tracks.iter())
            .map(|t| t.transitions.len())
            .sum();

        TimelineStatistics {
            duration_seconds,
            clip_count,
            video_clip_count,
            audio_clip_count,
            track_count,
            avg_clip_duration,
            min_clip_duration: if min_clip_duration.is_finite() {
                min_clip_duration
            } else {
                0.0
            },
            max_clip_duration: if max_clip_duration.is_finite() {
                max_clip_duration
            } else {
                0.0
            },
            total_gaps,
            transition_count,
        }
    }

    /// Calculate total gaps in timeline.
    fn calculate_total_gaps(timeline: &Timeline) -> f64 {
        let mut total_gaps = 0.0;

        for track in &timeline.video_tracks {
            total_gaps += Self::calculate_track_gaps(track, timeline.fps);
        }

        for track in &timeline.audio_tracks {
            total_gaps += Self::calculate_track_gaps(track, timeline.fps);
        }

        total_gaps
    }

    /// Calculate gaps in a track.
    fn calculate_track_gaps(track: &crate::timeline::Track, fps: FrameRate) -> f64 {
        let mut gaps = 0.0;

        for i in 0..track.clips.len() - 1 {
            let current = &track.clips[i];
            let next = &track.clips[i + 1];

            let current_out = current.timeline_out.to_frames(fps);
            let next_in = next.timeline_in.to_frames(fps);

            if next_in > current_out {
                let gap_frames = next_in - current_out;
                gaps += gap_frames as f64 / fps.as_f64();
            }
        }

        gaps
    }
}

/// Match analyzer.
pub struct MatchAnalyzer;

impl MatchAnalyzer {
    /// Analyze match quality.
    #[must_use]
    pub fn analyze_quality(matches: &[ClipMatch]) -> MatchQualityStats {
        if matches.is_empty() {
            return MatchQualityStats {
                avg_score: 0.0,
                min_score: 0.0,
                max_score: 0.0,
                high_quality_count: 0,
                medium_quality_count: 0,
                low_quality_count: 0,
                method_distribution: HashMap::new(),
            };
        }

        let scores: Vec<f64> = matches.iter().map(|m| m.score).collect();
        let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let min_score = scores.iter().copied().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let high_quality_count = scores.iter().filter(|&&s| s > 0.9).count();
        let medium_quality_count = scores.iter().filter(|&&s| (0.7..=0.9).contains(&s)).count();
        let low_quality_count = scores.iter().filter(|&&s| s < 0.7).count();

        let mut method_distribution = HashMap::new();
        for clip_match in matches {
            *method_distribution
                .entry(clip_match.method.to_string())
                .or_insert(0) += 1;
        }

        MatchQualityStats {
            avg_score,
            min_score,
            max_score,
            high_quality_count,
            medium_quality_count,
            low_quality_count,
            method_distribution,
        }
    }

    /// Analyze match report.
    #[must_use]
    pub fn analyze_report(report: &MatchReport) -> MatchReportAnalysis {
        let quality_stats = Self::analyze_quality(&report.matched);

        MatchReportAnalysis {
            total_clips: report.stats.total_clips,
            matched_count: report.stats.matched_count,
            missing_count: report.stats.missing_count,
            ambiguous_count: report.stats.ambiguous_count,
            conform_rate: report.stats.conform_rate,
            quality_stats,
        }
    }
}

/// Match report analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchReportAnalysis {
    /// Total clips.
    pub total_clips: usize,
    /// Matched clips.
    pub matched_count: usize,
    /// Missing clips.
    pub missing_count: usize,
    /// Ambiguous clips.
    pub ambiguous_count: usize,
    /// Conform rate.
    pub conform_rate: f64,
    /// Quality statistics.
    pub quality_stats: MatchQualityStats,
}

/// Media analyzer.
pub struct MediaAnalyzer;

impl MediaAnalyzer {
    /// Analyze media files.
    #[must_use]
    pub fn analyze(media_files: &[MediaFile]) -> MediaStatistics {
        let total_files = media_files.len();
        let total_size: u64 = media_files.iter().filter_map(|m| m.size).sum();
        let avg_size = if total_files > 0 {
            total_size / total_files as u64
        } else {
            0
        };

        let mut format_distribution = HashMap::new();
        let mut resolution_distribution = HashMap::new();
        let mut fps_distribution = HashMap::new();

        for media in media_files {
            // Format distribution
            if let Some(ext) = media.path.extension().and_then(|e| e.to_str()) {
                *format_distribution.entry(ext.to_lowercase()).or_insert(0) += 1;
            }

            // Resolution distribution
            if let (Some(width), Some(height)) = (media.width, media.height) {
                let res_key = format!("{width}x{height}");
                *resolution_distribution.entry(res_key).or_insert(0) += 1;
            }

            // FPS distribution
            if let Some(fps) = media.fps {
                let fps_key = format!("{:.2}", fps.as_f64());
                *fps_distribution.entry(fps_key).or_insert(0) += 1;
            }
        }

        MediaStatistics {
            total_files,
            total_size,
            avg_size,
            format_distribution,
            resolution_distribution,
            fps_distribution,
        }
    }

    /// Find duplicate media files by checksum.
    #[must_use]
    pub fn find_duplicates(media_files: &[MediaFile]) -> Vec<Vec<MediaFile>> {
        let mut checksum_map: HashMap<String, Vec<MediaFile>> = HashMap::new();

        for media in media_files {
            if let Some(ref md5) = media.md5 {
                checksum_map
                    .entry(md5.clone())
                    .or_default()
                    .push(media.clone());
            }
        }

        checksum_map
            .into_iter()
            .filter(|(_, v)| v.len() > 1)
            .map(|(_, v)| v)
            .collect()
    }

    /// Find missing media files (referenced but not found).
    #[must_use]
    pub fn find_missing(referenced: &[String], available: &[MediaFile]) -> Vec<String> {
        let available_names: std::collections::HashSet<_> =
            available.iter().map(|m| &m.filename).collect();

        referenced
            .iter()
            .filter(|name| !available_names.contains(name))
            .cloned()
            .collect()
    }
}

/// Comprehensive session analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAnalysis {
    /// Timeline statistics.
    pub timeline_stats: Option<TimelineStatistics>,
    /// Match report analysis.
    pub match_analysis: MatchReportAnalysis,
    /// Media statistics.
    pub media_stats: MediaStatistics,
    /// Processing time in seconds.
    pub processing_time: f64,
    /// Issues detected.
    pub issues: Vec<String>,
}

/// Session analyzer.
pub struct SessionAnalyzer;

impl SessionAnalyzer {
    /// Perform comprehensive session analysis.
    #[must_use]
    pub fn analyze(
        timeline: Option<&Timeline>,
        report: &MatchReport,
        media_files: &[MediaFile],
        processing_time: f64,
    ) -> SessionAnalysis {
        let timeline_stats = timeline.map(TimelineAnalyzer::analyze);
        let match_analysis = MatchAnalyzer::analyze_report(report);
        let media_stats = MediaAnalyzer::analyze(media_files);

        let mut issues = Vec::new();

        // Detect issues
        if match_analysis.conform_rate < 0.9 {
            issues.push(format!(
                "Low conform rate: {:.1}%",
                match_analysis.conform_rate * 100.0
            ));
        }

        if match_analysis.quality_stats.low_quality_count > 0 {
            issues.push(format!(
                "{} low-quality matches detected",
                match_analysis.quality_stats.low_quality_count
            ));
        }

        if let Some(ref stats) = timeline_stats {
            if stats.total_gaps > 0.0 {
                issues.push(format!("Total gaps: {:.2}s", stats.total_gaps));
            }
        }

        SessionAnalysis {
            timeline_stats,
            match_analysis,
            media_stats,
            processing_time,
            issues,
        }
    }

    /// Generate a text report.
    #[must_use]
    pub fn generate_text_report(analysis: &SessionAnalysis) -> String {
        let mut report = String::new();

        report.push_str("=== SESSION ANALYSIS ===\n\n");

        if let Some(ref stats) = analysis.timeline_stats {
            report.push_str("Timeline Statistics:\n");
            report.push_str(&format!("  Duration: {:.2}s\n", stats.duration_seconds));
            report.push_str(&format!("  Total Clips: {}\n", stats.clip_count));
            report.push_str(&format!("  Video Clips: {}\n", stats.video_clip_count));
            report.push_str(&format!("  Audio Clips: {}\n", stats.audio_clip_count));
            report.push_str(&format!("  Tracks: {}\n", stats.track_count));
            report.push_str(&format!(
                "  Avg Clip Duration: {:.2}s\n",
                stats.avg_clip_duration
            ));
            report.push_str(&format!("  Total Gaps: {:.2}s\n", stats.total_gaps));
            report.push_str(&format!("  Transitions: {}\n\n", stats.transition_count));
        }

        report.push_str("Match Analysis:\n");
        report.push_str(&format!(
            "  Total Clips: {}\n",
            analysis.match_analysis.total_clips
        ));
        report.push_str(&format!(
            "  Matched: {}\n",
            analysis.match_analysis.matched_count
        ));
        report.push_str(&format!(
            "  Missing: {}\n",
            analysis.match_analysis.missing_count
        ));
        report.push_str(&format!(
            "  Ambiguous: {}\n",
            analysis.match_analysis.ambiguous_count
        ));
        report.push_str(&format!(
            "  Conform Rate: {:.1}%\n",
            analysis.match_analysis.conform_rate * 100.0
        ));
        report.push_str(&format!(
            "  Avg Score: {:.3}\n\n",
            analysis.match_analysis.quality_stats.avg_score
        ));

        report.push_str("Media Statistics:\n");
        report.push_str(&format!(
            "  Total Files: {}\n",
            analysis.media_stats.total_files
        ));
        report.push_str(&format!(
            "  Total Size: {}\n",
            crate::utils::format_file_size(analysis.media_stats.total_size)
        ));
        report.push_str(&format!(
            "  Avg Size: {}\n\n",
            crate::utils::format_file_size(analysis.media_stats.avg_size)
        ));

        if !analysis.issues.is_empty() {
            report.push_str("Issues Detected:\n");
            for issue in &analysis.issues {
                report.push_str(&format!("  - {issue}\n"));
            }
            report.push('\n');
        }

        report.push_str(&format!(
            "Processing Time: {:.2}s\n",
            analysis.processing_time
        ));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_analyzer_empty() {
        let timeline = Timeline::new("Test".to_string(), FrameRate::Fps25);
        let stats = TimelineAnalyzer::analyze(&timeline);
        assert_eq!(stats.clip_count, 0);
    }

    #[test]
    fn test_match_analyzer_empty() {
        let stats = MatchAnalyzer::analyze_quality(&[]);
        assert!((stats.avg_score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_media_analyzer_empty() {
        let stats = MediaAnalyzer::analyze(&[]);
        assert_eq!(stats.total_files, 0);
    }
}
