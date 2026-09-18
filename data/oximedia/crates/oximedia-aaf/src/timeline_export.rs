//! AAF timeline export configuration and event model.
//!
//! Provides `ExportTarget`, `AafTimelineExport`, and `AafExportConfig`
//! for controlling how an AAF timeline is exported to downstream formats.

#![allow(dead_code)]

use std::collections::HashMap;

/// The target format/system for an AAF timeline export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportTarget {
    /// CMX 3600 EDL.
    Cmx3600Edl,
    /// OpenTimelineIO JSON.
    OpenTimelineIo,
    /// Avid project interchange.
    AvidInterchange,
    /// Final Cut Pro XML.
    FcpXml,
    /// DaVinci Resolve project.
    DaVinciResolve,
    /// Raw AAF (re-export without changes).
    RawAaf,
}

impl ExportTarget {
    /// Returns `true` if this target format can carry marker/cue-point data.
    #[must_use]
    pub fn supports_markers(self) -> bool {
        matches!(
            self,
            Self::OpenTimelineIo | Self::FcpXml | Self::DaVinciResolve | Self::RawAaf
        )
    }

    /// Returns `true` if this target is a text-based format.
    #[must_use]
    pub fn is_text_format(self) -> bool {
        matches!(self, Self::Cmx3600Edl | Self::OpenTimelineIo | Self::FcpXml)
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cmx3600Edl => "CMX 3600 EDL",
            Self::OpenTimelineIo => "OpenTimelineIO",
            Self::AvidInterchange => "Avid Interchange",
            Self::FcpXml => "FCP XML",
            Self::DaVinciResolve => "DaVinci Resolve",
            Self::RawAaf => "Raw AAF",
        }
    }
}

/// A single export event (clip, transition, or marker) in the timeline.
#[derive(Debug, Clone)]
pub struct ExportEvent {
    /// Sequential event number (1-based).
    pub number: u32,
    /// Source reel / mob name.
    pub reel_name: String,
    /// Start frame offset on the source.
    pub source_in: i64,
    /// End frame offset on the source.
    pub source_out: i64,
    /// Record-in position on the timeline.
    pub record_in: i64,
    /// Record-out position on the timeline.
    pub record_out: i64,
    /// Optional comment or marker text.
    pub comment: Option<String>,
}

impl ExportEvent {
    /// Create a new `ExportEvent`.
    #[must_use]
    pub fn new(
        number: u32,
        reel_name: impl Into<String>,
        source_in: i64,
        source_out: i64,
        record_in: i64,
        record_out: i64,
    ) -> Self {
        Self {
            number,
            reel_name: reel_name.into(),
            source_in,
            source_out,
            record_in,
            record_out,
            comment: None,
        }
    }

    /// Duration in frames on the record timeline.
    #[must_use]
    pub fn duration_frames(&self) -> i64 {
        (self.record_out - self.record_in).max(0)
    }
}

/// Accumulates `ExportEvent` entries and provides summary queries.
#[derive(Debug, Clone, Default)]
pub struct AafTimelineExport {
    events: Vec<ExportEvent>,
}

impl AafTimelineExport {
    /// Create an empty export.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event.
    pub fn add_event(&mut self, event: ExportEvent) {
        self.events.push(event);
    }

    /// Total number of events added.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Return all events as a slice.
    #[must_use]
    pub fn events(&self) -> &[ExportEvent] {
        &self.events
    }

    /// Total timeline duration in frames across all events.
    #[must_use]
    pub fn total_duration_frames(&self) -> i64 {
        self.events.iter().map(|e| e.duration_frames()).sum()
    }

    /// Return a sorted list of unique reel names referenced by all events.
    #[must_use]
    pub fn referenced_reels(&self) -> Vec<&str> {
        let mut seen: HashMap<&str, ()> = HashMap::new();
        for e in &self.events {
            seen.insert(e.reel_name.as_str(), ());
        }
        let mut reels: Vec<&str> = seen.into_keys().collect();
        reels.sort_unstable();
        reels
    }
}

/// High-level configuration for an AAF timeline export operation.
#[derive(Debug, Clone)]
pub struct AafExportConfig {
    /// The export target format.
    pub target: ExportTarget,
    /// Whether to embed original AAF metadata in the output.
    pub embed_metadata: bool,
    /// Whether to use high-fidelity timecode (sub-frame precision).
    pub high_fidelity: bool,
    /// Whether to include audio track exports.
    pub include_audio: bool,
    /// Whether to include video track exports.
    pub include_video: bool,
    /// Maximum reel name length (None = unlimited).
    pub max_reel_name_len: Option<usize>,
}

impl AafExportConfig {
    /// Create a default config for the given target.
    #[must_use]
    pub fn for_target(target: ExportTarget) -> Self {
        Self {
            target,
            embed_metadata: true,
            high_fidelity: false,
            include_audio: true,
            include_video: true,
            max_reel_name_len: None,
        }
    }

    /// Returns `true` if high-fidelity timecode is requested.
    #[must_use]
    pub fn is_high_fidelity(&self) -> bool {
        self.high_fidelity
    }

    /// Enable high-fidelity mode.
    pub fn enable_high_fidelity(&mut self) {
        self.high_fidelity = true;
    }

    /// Disable metadata embedding.
    pub fn disable_metadata(&mut self) {
        self.embed_metadata = false;
    }
}

impl Default for AafExportConfig {
    fn default() -> Self {
        Self::for_target(ExportTarget::Cmx3600Edl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ExportTarget tests ---

    #[test]
    fn test_cmx3600_does_not_support_markers() {
        assert!(!ExportTarget::Cmx3600Edl.supports_markers());
    }

    #[test]
    fn test_otio_supports_markers() {
        assert!(ExportTarget::OpenTimelineIo.supports_markers());
    }

    #[test]
    fn test_raw_aaf_supports_markers() {
        assert!(ExportTarget::RawAaf.supports_markers());
    }

    #[test]
    fn test_fcpxml_is_text_format() {
        assert!(ExportTarget::FcpXml.is_text_format());
    }

    #[test]
    fn test_avid_not_text_format() {
        assert!(!ExportTarget::AvidInterchange.is_text_format());
    }

    #[test]
    fn test_target_name() {
        assert_eq!(ExportTarget::Cmx3600Edl.name(), "CMX 3600 EDL");
        assert_eq!(ExportTarget::DaVinciResolve.name(), "DaVinci Resolve");
    }

    // --- ExportEvent tests ---

    #[test]
    fn test_event_duration() {
        let ev = ExportEvent::new(1, "REEL_A", 0, 100, 0, 100);
        assert_eq!(ev.duration_frames(), 100);
    }

    #[test]
    fn test_event_zero_duration_clamped() {
        let ev = ExportEvent::new(1, "REEL_A", 0, 0, 50, 40);
        assert_eq!(ev.duration_frames(), 0);
    }

    // --- AafTimelineExport tests ---

    #[test]
    fn test_export_add_event_count() {
        let mut export = AafTimelineExport::new();
        assert_eq!(export.event_count(), 0);
        export.add_event(ExportEvent::new(1, "R1", 0, 50, 0, 50));
        export.add_event(ExportEvent::new(2, "R2", 50, 100, 50, 100));
        assert_eq!(export.event_count(), 2);
    }

    #[test]
    fn test_export_total_duration() {
        let mut export = AafTimelineExport::new();
        export.add_event(ExportEvent::new(1, "R1", 0, 25, 0, 25));
        export.add_event(ExportEvent::new(2, "R1", 25, 75, 25, 75));
        assert_eq!(export.total_duration_frames(), 75);
    }

    #[test]
    fn test_export_referenced_reels_unique_sorted() {
        let mut export = AafTimelineExport::new();
        export.add_event(ExportEvent::new(1, "REEL_B", 0, 10, 0, 10));
        export.add_event(ExportEvent::new(2, "REEL_A", 0, 10, 10, 20));
        export.add_event(ExportEvent::new(3, "REEL_B", 0, 5, 20, 25));
        let reels = export.referenced_reels();
        assert_eq!(reels, vec!["REEL_A", "REEL_B"]);
    }

    #[test]
    fn test_export_empty_reels() {
        let export = AafTimelineExport::new();
        assert!(export.referenced_reels().is_empty());
    }

    // --- AafExportConfig tests ---

    #[test]
    fn test_config_default_not_high_fidelity() {
        let cfg = AafExportConfig::for_target(ExportTarget::OpenTimelineIo);
        assert!(!cfg.is_high_fidelity());
    }

    #[test]
    fn test_config_enable_high_fidelity() {
        let mut cfg = AafExportConfig::for_target(ExportTarget::OpenTimelineIo);
        cfg.enable_high_fidelity();
        assert!(cfg.is_high_fidelity());
    }

    #[test]
    fn test_config_default_embed_metadata() {
        let cfg = AafExportConfig::default();
        assert!(cfg.embed_metadata);
    }

    #[test]
    fn test_config_disable_metadata() {
        let mut cfg = AafExportConfig::default();
        cfg.disable_metadata();
        assert!(!cfg.embed_metadata);
    }
}
