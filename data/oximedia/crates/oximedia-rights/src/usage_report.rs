//! Usage reporting for media assets under rights management.
//!
//! Tracks how, when, and where a piece of content was used so that
//! rights holders can be properly compensated and compliance can be
//! demonstrated to licensors and collecting societies.

#![allow(dead_code)]

use std::collections::HashMap;

/// The time granularity that a usage report covers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UsagePeriod {
    /// A single calendar day (`YYYY-MM-DD`).
    Daily(String),
    /// A calendar month (`YYYY-MM`).
    Monthly(String),
    /// A calendar quarter (`YYYY-Q{1-4}`).
    Quarterly(String),
    /// A full calendar year (`YYYY`).
    Annual(String),
    /// An arbitrary custom range described by start/end strings.
    Custom {
        /// Start of the custom range (e.g. ISO 8601 date string).
        start: String,
        /// End of the custom range (e.g. ISO 8601 date string).
        end: String,
    },
}

impl UsagePeriod {
    /// Return a human-readable label for the period.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Daily(d) => format!("Day:{d}"),
            Self::Monthly(m) => format!("Month:{m}"),
            Self::Quarterly(q) => format!("Quarter:{q}"),
            Self::Annual(y) => format!("Year:{y}"),
            Self::Custom { start, end } => format!("{start}..{end}"),
        }
    }
}

/// The type of usage event that was recorded.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UsageType {
    /// Content was streamed online.
    Streaming,
    /// Content was broadcast on linear television or radio.
    Broadcast,
    /// Content was downloaded by a user.
    Download,
    /// Content was synchronised with audio-visual media (sync licence).
    Synchronisation,
    /// Content was used in a public performance.
    PublicPerformance,
    /// Content was reproduced in print or digital media.
    Reproduction,
    /// Any other type of usage.
    Other(String),
}

impl UsageType {
    /// Return a short code for the usage type, suitable for CSV export.
    #[must_use]
    pub fn code(&self) -> &str {
        match self {
            Self::Streaming => "STREAM",
            Self::Broadcast => "BCAST",
            Self::Download => "DL",
            Self::Synchronisation => "SYNC",
            Self::PublicPerformance => "PP",
            Self::Reproduction => "REPRO",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// A single usage event for a specific asset.
///
/// # Example
///
/// ```
/// use oximedia_rights::usage_report::{UsageRecord, UsageType};
///
/// let record = UsageRecord::new("asset-001", UsageType::Streaming, 150_000)
///     .with_territory("US")
///     .with_platform("OTT-X");
/// assert_eq!(record.play_count, 150_000);
/// ```
#[derive(Debug, Clone)]
pub struct UsageRecord {
    /// Identifier of the asset (track, clip, etc.) that was used.
    pub asset_id: String,
    /// How the asset was used.
    pub usage_type: UsageType,
    /// Number of times (plays, streams, downloads, etc.) the usage occurred.
    pub play_count: u64,
    /// ISO 3166-1 alpha-2 territory code where the usage occurred.
    pub territory: Option<String>,
    /// Name of the distribution platform or broadcaster.
    pub platform: Option<String>,
    /// Calculated revenue attributed to this usage, in minor currency units.
    pub revenue_minor: Option<i64>,
}

impl UsageRecord {
    /// Create a new `UsageRecord`.
    #[must_use]
    pub fn new(asset_id: impl Into<String>, usage_type: UsageType, play_count: u64) -> Self {
        Self {
            asset_id: asset_id.into(),
            usage_type,
            play_count,
            territory: None,
            platform: None,
            revenue_minor: None,
        }
    }

    /// Set the territory.
    #[must_use]
    pub fn with_territory(mut self, territory: impl Into<String>) -> Self {
        self.territory = Some(territory.into());
        self
    }

    /// Set the platform name.
    #[must_use]
    pub fn with_platform(mut self, platform: impl Into<String>) -> Self {
        self.platform = Some(platform.into());
        self
    }

    /// Set the revenue attributed to this record (in minor currency units).
    #[must_use]
    pub fn with_revenue(mut self, revenue_minor: i64) -> Self {
        self.revenue_minor = Some(revenue_minor);
        self
    }
}

/// An aggregated usage report covering a defined period.
///
/// # Example
///
/// ```
/// use oximedia_rights::usage_report::{UsagePeriod, UsageRecord, UsageReport, UsageType};
///
/// let period = UsagePeriod::Monthly("2025-06".to_string());
/// let mut report = UsageReport::new(period);
/// report.add(UsageRecord::new("a-1", UsageType::Streaming, 1_000));
/// report.add(UsageRecord::new("a-2", UsageType::Broadcast, 500));
/// report.add(UsageRecord::new("a-1", UsageType::Streaming, 2_000));
///
/// assert_eq!(report.total_plays(), 3_500);
/// ```
#[derive(Debug)]
pub struct UsageReport {
    /// The time period this report covers.
    pub period: UsagePeriod,
    records: Vec<UsageRecord>,
}

impl UsageReport {
    /// Create an empty `UsageReport` for the given period.
    #[must_use]
    pub fn new(period: UsagePeriod) -> Self {
        Self {
            period,
            records: Vec::new(),
        }
    }

    /// Append a usage record to the report.
    pub fn add(&mut self, record: UsageRecord) {
        self.records.push(record);
    }

    /// Return the total number of records.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Return the sum of all play counts across all records.
    #[must_use]
    pub fn total_plays(&self) -> u64 {
        self.records.iter().map(|r| r.play_count).sum()
    }

    /// Return the total revenue (in minor units) across all records that
    /// have revenue set.
    #[must_use]
    pub fn total_revenue(&self) -> i64 {
        self.records.iter().filter_map(|r| r.revenue_minor).sum()
    }

    /// Return a map from `UsageType` to total play count for that type.
    #[must_use]
    pub fn total_by_type(&self) -> HashMap<String, u64> {
        let mut map: HashMap<String, u64> = HashMap::new();
        for r in &self.records {
            *map.entry(r.usage_type.code().to_string()).or_insert(0) += r.play_count;
        }
        map
    }

    /// Return all records for a specific asset id.
    #[must_use]
    pub fn records_for_asset(&self, asset_id: &str) -> Vec<&UsageRecord> {
        self.records
            .iter()
            .filter(|r| r.asset_id == asset_id)
            .collect()
    }

    /// Return total play count for a specific asset id.
    #[must_use]
    pub fn plays_for_asset(&self, asset_id: &str) -> u64 {
        self.records_for_asset(asset_id)
            .iter()
            .map(|r| r.play_count)
            .sum()
    }

    /// Return a sorted list of (territory, total_plays) pairs, descending.
    #[must_use]
    pub fn plays_by_territory(&self) -> Vec<(String, u64)> {
        let mut map: HashMap<String, u64> = HashMap::new();
        for r in &self.records {
            if let Some(t) = &r.territory {
                *map.entry(t.clone()).or_insert(0) += r.play_count;
            }
        }
        let mut pairs: Vec<(String, u64)> = map.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs
    }

    /// Return `true` if the report contains no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_report() -> UsageReport {
        let period = UsagePeriod::Monthly("2025-06".to_string());
        let mut report = UsageReport::new(period);
        report.add(
            UsageRecord::new("a-1", UsageType::Streaming, 1_000)
                .with_territory("US")
                .with_platform("OTT-X")
                .with_revenue(5_000),
        );
        report.add(
            UsageRecord::new("a-2", UsageType::Broadcast, 500)
                .with_territory("GB")
                .with_revenue(2_500),
        );
        report.add(
            UsageRecord::new("a-1", UsageType::Streaming, 2_000)
                .with_territory("DE")
                .with_revenue(8_000),
        );
        report.add(UsageRecord::new("a-3", UsageType::Download, 200));
        report
    }

    #[test]
    fn test_record_count() {
        assert_eq!(make_report().record_count(), 4);
    }

    #[test]
    fn test_total_plays() {
        assert_eq!(make_report().total_plays(), 3_700);
    }

    #[test]
    fn test_total_revenue() {
        assert_eq!(make_report().total_revenue(), 15_500);
    }

    #[test]
    fn test_total_by_type_streaming() {
        let map = make_report().total_by_type();
        assert_eq!(map.get("STREAM").copied().unwrap_or(0), 3_000);
    }

    #[test]
    fn test_total_by_type_broadcast() {
        let map = make_report().total_by_type();
        assert_eq!(map.get("BCAST").copied().unwrap_or(0), 500);
    }

    #[test]
    fn test_records_for_asset() {
        let report = make_report();
        assert_eq!(report.records_for_asset("a-1").len(), 2);
    }

    #[test]
    fn test_plays_for_asset() {
        assert_eq!(make_report().plays_for_asset("a-1"), 3_000);
    }

    #[test]
    fn test_plays_by_territory_order() {
        let pairs = make_report().plays_by_territory();
        // DE has 2000, US has 1000, GB has 500 — descending
        assert_eq!(pairs[0].0, "DE");
        assert_eq!(pairs[0].1, 2_000);
    }

    #[test]
    fn test_usage_period_label() {
        let p = UsagePeriod::Monthly("2025-06".to_string());
        assert_eq!(p.label(), "Month:2025-06");
    }

    #[test]
    fn test_usage_period_custom_label() {
        let p = UsagePeriod::Custom {
            start: "2025-01-01".to_string(),
            end: "2025-03-31".to_string(),
        };
        assert_eq!(p.label(), "2025-01-01..2025-03-31");
    }

    #[test]
    fn test_usage_type_code() {
        assert_eq!(UsageType::Streaming.code(), "STREAM");
        assert_eq!(UsageType::Broadcast.code(), "BCAST");
        assert_eq!(UsageType::Download.code(), "DL");
        assert_eq!(UsageType::Synchronisation.code(), "SYNC");
        assert_eq!(UsageType::PublicPerformance.code(), "PP");
        assert_eq!(UsageType::Reproduction.code(), "REPRO");
    }

    #[test]
    fn test_is_empty() {
        let report = UsageReport::new(UsagePeriod::Annual("2025".to_string()));
        assert!(report.is_empty());
    }

    #[test]
    fn test_total_revenue_no_revenue_set() {
        let period = UsagePeriod::Daily("2025-06-01".to_string());
        let mut report = UsageReport::new(period);
        report.add(UsageRecord::new("a-1", UsageType::Download, 100));
        assert_eq!(report.total_revenue(), 0);
    }

    #[test]
    fn test_plays_by_territory_excludes_no_territory() {
        let report = make_report();
        let pairs = report.plays_by_territory();
        // "a-3" Download has no territory, so should not appear
        assert!(pairs.iter().all(|(t, _)| !t.is_empty()));
    }
}
