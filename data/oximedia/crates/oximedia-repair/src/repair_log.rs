//! Structured logging of repair actions taken on a media file.
//!
//! `RepairLog` records every action applied during a repair session and
//! provides aggregate statistics such as overall success rate.
//!
//! Logs can be exported as structured JSON for machine-parseable repair history.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Category of action taken during a repair pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairAction {
    /// A corrupted header field was overwritten with a valid value.
    HeaderRewrite {
        /// Name of the field that was repaired.
        field: String,
    },
    /// A missing or invalid index was reconstructed.
    IndexRebuild,
    /// Timestamps were corrected to form a monotonic sequence.
    TimestampCorrection {
        /// Number of timestamps that were adjusted.
        count: u32,
    },
    /// A missing packet was synthesised from context.
    PacketConcealment {
        /// Sequence number of the concealed packet.
        seq: u32,
    },
    /// A damaged frame was repaired using inpainting or interpolation.
    FrameRepair {
        /// Frame index in the video stream.
        frame_index: u64,
    },
    /// Audio and video streams were re-synchronised.
    AVResync {
        /// Offset applied (in milliseconds, positive = video delayed).
        offset_ms: i32,
    },
    /// Metadata was reconstructed from side-channel information.
    MetadataReconstruction {
        /// Key of the metadata field that was restored.
        key: String,
    },
    /// A custom, application-specific repair action.
    Custom {
        /// Short label describing the action.
        label: String,
    },
}

/// Outcome of a single repair action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionOutcome {
    /// The action succeeded.
    Success,
    /// The action was attempted but produced only a partial fix.
    Partial,
    /// The action failed and the issue was not repaired.
    Failed,
    /// The action was skipped (e.g. not applicable to this file).
    Skipped,
}

/// A single entry in the repair log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairLogEntry {
    /// The repair action that was taken.
    pub action: RepairAction,
    /// Outcome of the action.
    pub outcome: ActionOutcome,
    /// Optional human-readable note.
    pub note: Option<String>,
    /// Wall-clock time at which the action was recorded.
    pub elapsed: Duration,
}

/// Structured log of all repair actions for a session.
#[derive(Debug)]
pub struct RepairLog {
    entries: Vec<RepairLogEntry>,
    start: Instant,
}

impl RepairLog {
    /// Create a new, empty repair log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            start: Instant::now(),
        }
    }

    /// Append an entry to the log.
    pub fn record(&mut self, action: RepairAction, outcome: ActionOutcome, note: Option<String>) {
        self.entries.push(RepairLogEntry {
            action,
            outcome,
            note,
            elapsed: self.start.elapsed(),
        });
    }

    /// Shorthand: record a successful action.
    pub fn record_ok(&mut self, action: RepairAction) {
        self.record(action, ActionOutcome::Success, None);
    }

    /// Shorthand: record a failed action with a reason.
    pub fn record_fail(&mut self, action: RepairAction, reason: &str) {
        self.record(action, ActionOutcome::Failed, Some(reason.to_owned()));
    }

    /// Calculate the fraction of actions that succeeded.
    ///
    /// Returns a value in `[0.0, 1.0]`. Returns `0.0` if the log is empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let successes = self
            .entries
            .iter()
            .filter(|e| e.outcome == ActionOutcome::Success)
            .count();
        successes as f64 / self.entries.len() as f64
    }

    /// Count entries with a specific outcome.
    pub fn count_outcome(&self, outcome: ActionOutcome) -> usize {
        self.entries.iter().filter(|e| e.outcome == outcome).count()
    }

    /// All log entries.
    pub fn entries(&self) -> &[RepairLogEntry] {
        &self.entries
    }

    /// Total number of actions recorded.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no actions have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Generate a plain-text summary report.
    pub fn summary(&self) -> String {
        let success = self.count_outcome(ActionOutcome::Success);
        let partial = self.count_outcome(ActionOutcome::Partial);
        let failed = self.count_outcome(ActionOutcome::Failed);
        let skipped = self.count_outcome(ActionOutcome::Skipped);
        format!(
            "Repair log: {} total | {} ok | {} partial | {} failed | {} skipped | {:.1}% success rate",
            self.entries.len(),
            success,
            partial,
            failed,
            skipped,
            self.success_rate() * 100.0,
        )
    }

    /// Elapsed time since the log was created.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Default for RepairLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_log_success_rate() {
        let log = RepairLog::new();
        assert_eq!(log.success_rate(), 0.0);
    }

    #[test]
    fn test_all_success() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::IndexRebuild);
        log.record_ok(RepairAction::IndexRebuild);
        assert!((log.success_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mixed_outcomes() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::IndexRebuild);
        log.record_fail(RepairAction::IndexRebuild, "disk full");
        // 1 success out of 2
        assert!((log.success_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_count_outcome() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::IndexRebuild);
        log.record(RepairAction::IndexRebuild, ActionOutcome::Partial, None);
        log.record(RepairAction::IndexRebuild, ActionOutcome::Skipped, None);
        assert_eq!(log.count_outcome(ActionOutcome::Success), 1);
        assert_eq!(log.count_outcome(ActionOutcome::Partial), 1);
        assert_eq!(log.count_outcome(ActionOutcome::Skipped), 1);
        assert_eq!(log.count_outcome(ActionOutcome::Failed), 0);
    }

    #[test]
    fn test_is_empty() {
        let log = RepairLog::new();
        assert!(log.is_empty());
    }

    #[test]
    fn test_len() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::IndexRebuild);
        log.record_ok(RepairAction::IndexRebuild);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_header_rewrite_action() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::HeaderRewrite {
            field: "moov.mvhd.duration".to_string(),
        });
        assert_eq!(log.len(), 1);
        if let RepairAction::HeaderRewrite { field } = &log.entries()[0].action {
            assert_eq!(field, "moov.mvhd.duration");
        } else {
            panic!("wrong action variant");
        }
    }

    #[test]
    fn test_timestamp_correction_action() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::TimestampCorrection { count: 42 });
        if let RepairAction::TimestampCorrection { count } = log.entries()[0].action {
            assert_eq!(count, 42);
        }
    }

    #[test]
    fn test_packet_concealment_action() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::PacketConcealment { seq: 1234 });
        if let RepairAction::PacketConcealment { seq } = log.entries()[0].action {
            assert_eq!(seq, 1234);
        }
    }

    #[test]
    fn test_frame_repair_action() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::FrameRepair { frame_index: 500 });
        if let RepairAction::FrameRepair { frame_index } = log.entries()[0].action {
            assert_eq!(frame_index, 500);
        }
    }

    #[test]
    fn test_av_resync_action() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::AVResync { offset_ms: -40 });
        if let RepairAction::AVResync { offset_ms } = log.entries()[0].action {
            assert_eq!(offset_ms, -40);
        }
    }

    #[test]
    fn test_summary_contains_total() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::IndexRebuild);
        log.record_fail(RepairAction::IndexRebuild, "err");
        let s = log.summary();
        assert!(s.contains("2 total"), "summary: {s}");
    }

    #[test]
    fn test_note_preserved() {
        let mut log = RepairLog::new();
        log.record(
            RepairAction::IndexRebuild,
            ActionOutcome::Failed,
            Some("checksum mismatch".to_string()),
        );
        assert_eq!(log.entries()[0].note.as_deref(), Some("checksum mismatch"));
    }

    #[test]
    fn test_custom_action() {
        let mut log = RepairLog::new();
        log.record_ok(RepairAction::Custom {
            label: "deinterlace_fix".to_string(),
        });
        if let RepairAction::Custom { label } = &log.entries()[0].action {
            assert_eq!(label, "deinterlace_fix");
        }
    }

    #[test]
    fn test_elapsed_nonnegative() {
        let log = RepairLog::new();
        // Elapsed should be a very small but non-negative duration.
        let e = log.elapsed();
        assert!(e.as_nanos() < 5_000_000_000, "elapsed seems too large");
    }
}
