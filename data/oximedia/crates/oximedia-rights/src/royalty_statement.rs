//! Royalty statement generation, CSV/JSON export, and period-based aggregation.
//!
//! Provides a complete pipeline from raw usage records to auditable royalty
//! statements with per-ISRC line items, period boundaries, and export
//! capabilities.

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]

use crate::{Result, RightsError};
use serde::{Deserialize, Serialize};

// ── Data types ────────────────────────────────────────────────────────────────

/// A single raw usage event for a sound recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    /// International Standard Recording Code of the recording.
    pub isrc: String,
    /// Number of streaming plays in this record.
    pub streams: u64,
    /// Number of download transactions in this record.
    pub downloads: u64,
    /// Per-stream royalty rate (currency units per stream).
    pub stream_rate: f64,
    /// Per-download royalty rate (currency units per download).
    pub download_rate: f64,
}

impl UsageRecord {
    /// Create a new usage record.
    pub fn new(
        isrc: impl Into<String>,
        streams: u64,
        downloads: u64,
        stream_rate: f64,
        download_rate: f64,
    ) -> Self {
        Self {
            isrc: isrc.into(),
            streams,
            downloads,
            stream_rate,
            download_rate,
        }
    }
}

/// A single line item in a royalty statement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatementLine {
    /// ISRC of the recording.
    pub isrc: String,
    /// Total streaming plays for the period.
    pub streams: u64,
    /// Total download transactions for the period.
    pub downloads: u64,
    /// Blended effective rate used to compute the amount.
    /// `amount / (streams + downloads)` if combined, or the primary rate when
    /// only one usage type is present.
    pub rate: f64,
    /// Total royalty amount for this line: `streams * stream_rate + downloads * download_rate`.
    pub amount: f64,
}

/// A complete royalty statement covering a specific reporting period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoyaltyStatement {
    /// Unix timestamp (seconds) for the start of the reporting period.
    pub period_start: u64,
    /// Unix timestamp (seconds) for the end of the reporting period.
    pub period_end: u64,
    /// Individual line items, one per unique ISRC.
    pub lines: Vec<StatementLine>,
}

impl RoyaltyStatement {
    /// Sum of all `amount` values across all line items.
    pub fn total_amount(&self) -> f64 {
        self.lines.iter().map(|l| l.amount).sum()
    }

    /// Export the statement as a CSV string.
    ///
    /// The CSV header is:
    /// `isrc,streams,downloads,rate,amount`
    pub fn to_csv(&self) -> String {
        let mut out = String::from("isrc,streams,downloads,rate,amount\n");
        for line in &self.lines {
            out.push_str(&format!(
                "{},{},{},{:.6},{:.6}\n",
                csv_escape(&line.isrc),
                line.streams,
                line.downloads,
                line.rate,
                line.amount,
            ));
        }
        out
    }

    /// Export the statement as a JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| RightsError::Serialization(e.to_string()))
    }

    /// Return a new statement containing only lines whose `amount >= min_amount`.
    pub fn filter_min_amount(&self, min_amount: f64) -> RoyaltyStatement {
        RoyaltyStatement {
            period_start: self.period_start,
            period_end: self.period_end,
            lines: self
                .lines
                .iter()
                .filter(|l| l.amount >= min_amount)
                .cloned()
                .collect(),
        }
    }

    /// Return a new statement with lines sorted by `amount` descending.
    pub fn sort_by_amount_desc(&self) -> RoyaltyStatement {
        let mut lines = self.lines.clone();
        lines.sort_by(|a, b| {
            b.amount
                .partial_cmp(&a.amount)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        RoyaltyStatement {
            period_start: self.period_start,
            period_end: self.period_end,
            lines,
        }
    }
}

/// Escape a string value for safe embedding inside a CSV field.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// ── Generator ─────────────────────────────────────────────────────────────────

/// Generates royalty statements from collections of [`UsageRecord`]s.
#[derive(Debug, Clone)]
pub struct RoyaltyStatementGenerator {
    /// Fallback per-stream rate used when a record's `stream_rate` is zero.
    pub default_stream_rate: f64,
    /// Fallback per-download rate used when a record's `download_rate` is zero.
    pub default_download_rate: f64,
}

impl RoyaltyStatementGenerator {
    /// Create a generator with explicit default rates.
    pub fn new(default_stream_rate: f64, default_download_rate: f64) -> Self {
        Self {
            default_stream_rate,
            default_download_rate,
        }
    }

    /// Create a generator using industry-typical defaults:
    /// $0.004 per stream, $0.70 per download.
    pub fn with_typical_defaults() -> Self {
        Self::new(0.004, 0.70)
    }

    /// Generate a [`RoyaltyStatement`] from a slice of usage records.
    ///
    /// Records are aggregated by ISRC.  Each record's own `stream_rate` /
    /// `download_rate` is used if non-zero; otherwise the generator's
    /// defaults are applied.
    pub fn generate(
        &self,
        period_start: u64,
        period_end: u64,
        usages: &[UsageRecord],
    ) -> RoyaltyStatement {
        use std::collections::HashMap;

        // Accumulator: isrc -> (total_streams, total_downloads, total_amount, weighted_rate_num, weighted_rate_den)
        struct Acc {
            streams: u64,
            downloads: u64,
            amount: f64,
        }

        let mut map: HashMap<String, Acc> = HashMap::new();

        for record in usages {
            let s_rate = if record.stream_rate > 0.0 {
                record.stream_rate
            } else {
                self.default_stream_rate
            };
            let d_rate = if record.download_rate > 0.0 {
                record.download_rate
            } else {
                self.default_download_rate
            };

            let record_amount =
                (record.streams as f64) * s_rate + (record.downloads as f64) * d_rate;

            let acc = map.entry(record.isrc.clone()).or_insert(Acc {
                streams: 0,
                downloads: 0,
                amount: 0.0,
            });
            acc.streams = acc.streams.saturating_add(record.streams);
            acc.downloads = acc.downloads.saturating_add(record.downloads);
            acc.amount += record_amount;
        }

        // Build sorted lines (sorted by ISRC for deterministic output).
        let mut isrcs: Vec<String> = map.keys().cloned().collect();
        isrcs.sort();

        let lines = isrcs
            .into_iter()
            .map(|isrc| {
                let acc = &map[&isrc];
                let total_units = acc.streams + acc.downloads;
                let rate = if total_units > 0 {
                    acc.amount / total_units as f64
                } else {
                    0.0
                };
                StatementLine {
                    isrc,
                    streams: acc.streams,
                    downloads: acc.downloads,
                    rate,
                    amount: acc.amount,
                }
            })
            .collect();

        RoyaltyStatement {
            period_start,
            period_end,
            lines,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gen() -> RoyaltyStatementGenerator {
        RoyaltyStatementGenerator::new(0.004, 0.70)
    }

    fn make_record(isrc: &str, streams: u64, downloads: u64) -> UsageRecord {
        UsageRecord::new(isrc, streams, downloads, 0.004, 0.70)
    }

    // ── generation tests ──────────────────────────────────────────────────────

    #[test]
    fn test_generate_single_record() {
        let records = vec![make_record("USAAA2400001", 1_000, 10)];
        let stmt = gen().generate(0, 1_000, &records);
        assert_eq!(stmt.lines.len(), 1);
        let line = &stmt.lines[0];
        assert_eq!(line.isrc, "USAAA2400001");
        assert_eq!(line.streams, 1_000);
        assert_eq!(line.downloads, 10);
        // 1000 * 0.004 + 10 * 0.70 = 4.0 + 7.0 = 11.0
        assert!((line.amount - 11.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_aggregates_same_isrc() {
        let records = vec![
            make_record("USAAA2400001", 500, 5),
            make_record("USAAA2400001", 500, 5),
        ];
        let stmt = gen().generate(0, 1_000, &records);
        assert_eq!(stmt.lines.len(), 1);
        assert_eq!(stmt.lines[0].streams, 1_000);
        assert_eq!(stmt.lines[0].downloads, 10);
    }

    #[test]
    fn test_generate_multiple_isrcs() {
        let records = vec![
            make_record("USAAA2400001", 1_000, 0),
            make_record("USAAA2400002", 500, 0),
        ];
        let stmt = gen().generate(0, 1_000, &records);
        assert_eq!(stmt.lines.len(), 2);
    }

    #[test]
    fn test_total_amount() {
        let records = vec![
            make_record("AA0000000001", 1_000, 0),
            make_record("AA0000000002", 0, 2),
        ];
        let stmt = gen().generate(0, 1_000, &records);
        // 1000 * 0.004 = 4.0; 2 * 0.70 = 1.40; total = 5.40
        assert!((stmt.total_amount() - 5.40).abs() < 1e-9);
    }

    #[test]
    fn test_generate_empty_usages() {
        let stmt = gen().generate(0, 1_000, &[]);
        assert!(stmt.lines.is_empty());
        assert_eq!(stmt.total_amount(), 0.0);
    }

    #[test]
    fn test_default_rate_fallback() {
        let record = UsageRecord::new("ISRC001", 100, 0, 0.0, 0.0); // rates are 0 -> use defaults
        let stmt = gen().generate(0, 100, &[record]);
        // 100 * 0.004 = 0.4
        assert!((stmt.lines[0].amount - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_period_timestamps_preserved() {
        let stmt = gen().generate(1_700_000_000, 1_700_086_400, &[]);
        assert_eq!(stmt.period_start, 1_700_000_000);
        assert_eq!(stmt.period_end, 1_700_086_400);
    }

    // ── CSV export tests ──────────────────────────────────────────────────────

    #[test]
    fn test_to_csv_header() {
        let stmt = gen().generate(0, 1, &[]);
        let csv = stmt.to_csv();
        assert!(csv.starts_with("isrc,streams,downloads,rate,amount\n"));
    }

    #[test]
    fn test_to_csv_data_row() {
        let records = vec![make_record("USABC2400001", 1_000, 10)];
        let stmt = gen().generate(0, 1_000, &records);
        let csv = stmt.to_csv();
        assert!(csv.contains("USABC2400001,1000,10,"));
    }

    #[test]
    fn test_to_csv_empty_is_header_only() {
        let stmt = gen().generate(0, 1, &[]);
        let csv = stmt.to_csv();
        let lines: Vec<&str> = csv.trim_end().lines().collect();
        assert_eq!(lines.len(), 1); // only header
    }

    // ── JSON export tests ─────────────────────────────────────────────────────

    #[test]
    fn test_to_json_valid() {
        let records = vec![make_record("USJJJ2400001", 200, 3)];
        let stmt = gen().generate(0, 86_400, &records);
        let json = stmt.to_json().expect("json export ok");
        assert!(json.contains("USJJJ2400001"));
        assert!(json.contains("period_start"));
        assert!(json.contains("period_end"));
    }

    #[test]
    fn test_to_json_roundtrip() {
        let records = vec![make_record("USZZZ2400001", 500, 7)];
        let stmt = gen().generate(1_000, 2_000, &records);
        let json = stmt.to_json().expect("json ok");
        let decoded: RoyaltyStatement = serde_json::from_str(&json).expect("decode ok");
        assert_eq!(decoded.period_start, 1_000);
        assert_eq!(decoded.period_end, 2_000);
        assert_eq!(decoded.lines[0].isrc, "USZZZ2400001");
    }

    // ── filter / sort ─────────────────────────────────────────────────────────

    #[test]
    fn test_filter_min_amount() {
        let records = vec![
            make_record("AA0000000001", 1_000, 0), // 4.0
            make_record("AA0000000002", 1, 0),     // 0.004
        ];
        let stmt = gen().generate(0, 1_000, &records);
        let filtered = stmt.filter_min_amount(1.0);
        assert_eq!(filtered.lines.len(), 1);
        assert_eq!(filtered.lines[0].isrc, "AA0000000001");
    }

    #[test]
    fn test_sort_by_amount_desc() {
        let records = vec![
            make_record("AA0000000001", 100, 0),   // 0.4
            make_record("AA0000000002", 1_000, 0), // 4.0
        ];
        let stmt = gen().generate(0, 1_000, &records);
        let sorted = stmt.sort_by_amount_desc();
        assert_eq!(sorted.lines[0].isrc, "AA0000000002");
    }

    #[test]
    fn test_with_typical_defaults() {
        let g = RoyaltyStatementGenerator::with_typical_defaults();
        let records = vec![UsageRecord::new("X001", 1_000, 1, 0.0, 0.0)];
        let stmt = g.generate(0, 1_000, &records);
        // 1000 * 0.004 + 1 * 0.70 = 4.0 + 0.70 = 4.70
        assert!((stmt.total_amount() - 4.70).abs() < 1e-9);
    }
}
