//! Temporal forensics: detect timestamp anomalies in media metadata.
//!
//! Validates creation times, modification times, and in-stream timestamps
//! for consistency and plausibility, flagging discrepancies that may
//! indicate post-capture editing or fabrication.

#![allow(dead_code)]

// ── TimestampSource ───────────────────────────────────────────────────────────

/// Origin of a timestamp value in a media asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimestampSource {
    /// Filesystem creation time (OS metadata).
    FilesystemCreation,
    /// Filesystem modification time (OS metadata).
    FilesystemModified,
    /// EXIF / XMP embedded metadata.
    EmbeddedMetadata,
    /// Container-level timestamp (e.g. MP4 `mvhd` atom).
    ContainerHeader,
    /// GPS-synchronized timestamp from the capture device.
    GpsSynchronized,
    /// Network Time Protocol reference.
    Ntp,
}

impl TimestampSource {
    /// Returns a reliability score (0.0 – 1.0) for this source.
    ///
    /// Higher values indicate more trustworthy timestamps.
    #[must_use]
    pub fn reliability(&self) -> f64 {
        match self {
            Self::GpsSynchronized => 0.95,
            Self::Ntp => 0.90,
            Self::ContainerHeader => 0.70,
            Self::EmbeddedMetadata => 0.65,
            Self::FilesystemCreation => 0.50,
            Self::FilesystemModified => 0.30,
        }
    }

    /// Returns a human-readable label for this source.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::FilesystemCreation => "Filesystem Creation",
            Self::FilesystemModified => "Filesystem Modified",
            Self::EmbeddedMetadata => "Embedded Metadata",
            Self::ContainerHeader => "Container Header",
            Self::GpsSynchronized => "GPS Synchronized",
            Self::Ntp => "NTP",
        }
    }
}

// ── TimestampAnomaly ──────────────────────────────────────────────────────────

/// A timestamp discrepancy detected during temporal analysis.
#[derive(Debug, Clone)]
pub struct TimestampAnomaly {
    /// The timestamp sources that disagree.
    pub source_a: TimestampSource,
    pub source_b: TimestampSource,
    /// Difference between the two timestamps in seconds.
    pub delta_seconds: f64,
    /// Brief description of why this is suspicious.
    pub reason: String,
}

impl TimestampAnomaly {
    /// Create a new anomaly.
    #[must_use]
    pub fn new(
        source_a: TimestampSource,
        source_b: TimestampSource,
        delta_seconds: f64,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            source_a,
            source_b,
            delta_seconds: delta_seconds.abs(),
            reason: reason.into(),
        }
    }

    /// Returns `true` if the discrepancy is suspicious.
    ///
    /// A discrepancy is suspicious when the delta exceeds a threshold that
    /// depends on the reliability of the sources involved:
    /// - High-reliability pair (both >= 0.8): threshold 60 s
    /// - Medium pair (at least one >= 0.5): threshold 3600 s
    /// - Low pair: threshold 86 400 s (1 day)
    #[must_use]
    pub fn is_suspicious(&self) -> bool {
        let rel_a = self.source_a.reliability();
        let rel_b = self.source_b.reliability();
        let min_rel = rel_a.min(rel_b);

        let threshold = if min_rel >= 0.8 {
            60.0
        } else if min_rel >= 0.5 {
            3_600.0
        } else {
            86_400.0
        };

        self.delta_seconds > threshold
    }

    /// Returns the absolute delta in seconds.
    #[must_use]
    pub fn delta_seconds(&self) -> f64 {
        self.delta_seconds
    }
}

// ── TimeForensics ─────────────────────────────────────────────────────────────

/// A recorded timestamp observation: source + Unix epoch seconds.
#[derive(Debug, Clone)]
pub struct TimestampObservation {
    /// Where this timestamp came from.
    pub source: TimestampSource,
    /// Unix timestamp (seconds since 1970-01-01T00:00:00Z).
    pub unix_seconds: f64,
}

impl TimestampObservation {
    /// Create a new observation.
    #[must_use]
    pub fn new(source: TimestampSource, unix_seconds: f64) -> Self {
        Self {
            source,
            unix_seconds,
        }
    }
}

/// Analyzes a collection of timestamp observations for anomalies.
#[derive(Debug, Default)]
pub struct TimeForensics {
    /// Collected timestamp observations.
    observations: Vec<TimestampObservation>,
}

impl TimeForensics {
    /// Create an empty analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a timestamp observation.
    pub fn add_observation(&mut self, obs: TimestampObservation) {
        self.observations.push(obs);
    }

    /// Convenience method to add source + value directly.
    pub fn add(&mut self, source: TimestampSource, unix_seconds: f64) {
        self.observations.push(TimestampObservation {
            source,
            unix_seconds,
        });
    }

    /// Analyze all pairs of observations for anomalies and return a report.
    #[must_use]
    pub fn analyze_timestamps(&self) -> TimeAnomalyReport {
        let mut anomalies = Vec::new();

        for i in 0..self.observations.len() {
            for j in (i + 1)..self.observations.len() {
                let a = &self.observations[i];
                let b = &self.observations[j];
                let delta = (a.unix_seconds - b.unix_seconds).abs();

                let anomaly = TimestampAnomaly::new(
                    a.source,
                    b.source,
                    delta,
                    format!(
                        "Δ {delta:.1} s between {} and {}",
                        a.source.label(),
                        b.source.label()
                    ),
                );

                if anomaly.is_suspicious() {
                    anomalies.push(anomaly);
                }
            }
        }

        TimeAnomalyReport { anomalies }
    }

    /// Returns the number of observations.
    #[must_use]
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }
}

// ── TimeAnomalyReport ─────────────────────────────────────────────────────────

/// Report produced by timestamp analysis.
#[derive(Debug, Clone, Default)]
pub struct TimeAnomalyReport {
    /// All suspicious timestamp anomalies found.
    pub anomalies: Vec<TimestampAnomaly>,
}

impl TimeAnomalyReport {
    /// Create an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of suspicious timestamp anomalies.
    #[must_use]
    pub fn suspicious_count(&self) -> usize {
        self.anomalies.len()
    }

    /// Returns `true` if any anomaly was found.
    #[must_use]
    pub fn has_anomalies(&self) -> bool {
        !self.anomalies.is_empty()
    }

    /// Returns the largest delta observed (in seconds), or 0.0.
    #[must_use]
    pub fn max_delta_seconds(&self) -> f64 {
        self.anomalies
            .iter()
            .map(TimestampAnomaly::delta_seconds)
            .fold(0.0_f64, f64::max)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const UNIX_BASE: f64 = 1_700_000_000.0; // 2023-11-14T22:13:20Z

    #[test]
    fn test_timestamp_source_reliability_ordering() {
        assert!(
            TimestampSource::GpsSynchronized.reliability()
                > TimestampSource::FilesystemModified.reliability()
        );
        assert!(
            TimestampSource::Ntp.reliability() > TimestampSource::FilesystemCreation.reliability()
        );
    }

    #[test]
    fn test_timestamp_source_gps_reliability() {
        assert!((TimestampSource::GpsSynchronized.reliability() - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_timestamp_source_label() {
        assert_eq!(TimestampSource::Ntp.label(), "NTP");
        assert_eq!(
            TimestampSource::FilesystemModified.label(),
            "Filesystem Modified"
        );
    }

    #[test]
    fn test_timestamp_anomaly_is_suspicious_high_rel_small_delta() {
        // GPS vs NTP – both >= 0.8 → threshold 60 s
        let a = TimestampAnomaly::new(
            TimestampSource::GpsSynchronized,
            TimestampSource::Ntp,
            30.0, // within threshold
            "test",
        );
        assert!(!a.is_suspicious());
    }

    #[test]
    fn test_timestamp_anomaly_is_suspicious_high_rel_large_delta() {
        let a = TimestampAnomaly::new(
            TimestampSource::GpsSynchronized,
            TimestampSource::Ntp,
            120.0, // exceeds 60 s threshold
            "test",
        );
        assert!(a.is_suspicious());
    }

    #[test]
    fn test_timestamp_anomaly_not_suspicious_low_rel() {
        // FilesystemCreation vs FilesystemModified – both < 0.5 → threshold 86 400 s
        let a = TimestampAnomaly::new(
            TimestampSource::FilesystemCreation,
            TimestampSource::FilesystemModified,
            500.0, // well within 86 400 s
            "test",
        );
        assert!(!a.is_suspicious());
    }

    #[test]
    fn test_timestamp_anomaly_delta_seconds() {
        let a = TimestampAnomaly::new(
            TimestampSource::ContainerHeader,
            TimestampSource::EmbeddedMetadata,
            -500.0, // negative → abs value stored
            "test",
        );
        assert!((a.delta_seconds() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_time_forensics_no_observations() {
        let tf = TimeForensics::new();
        let report = tf.analyze_timestamps();
        assert_eq!(report.suspicious_count(), 0);
    }

    #[test]
    fn test_time_forensics_consistent_timestamps() {
        let mut tf = TimeForensics::new();
        // GPS and NTP differ by only 5 s — not suspicious
        tf.add(TimestampSource::GpsSynchronized, UNIX_BASE);
        tf.add(TimestampSource::Ntp, UNIX_BASE + 5.0);
        let report = tf.analyze_timestamps();
        assert!(!report.has_anomalies());
    }

    #[test]
    fn test_time_forensics_suspicious_pair() {
        let mut tf = TimeForensics::new();
        // GPS and NTP differ by 300 s — suspicious (threshold 60 s)
        tf.add(TimestampSource::GpsSynchronized, UNIX_BASE);
        tf.add(TimestampSource::Ntp, UNIX_BASE + 300.0);
        let report = tf.analyze_timestamps();
        assert!(report.has_anomalies());
        assert_eq!(report.suspicious_count(), 1);
    }

    #[test]
    fn test_time_forensics_observation_count() {
        let mut tf = TimeForensics::new();
        tf.add(TimestampSource::ContainerHeader, UNIX_BASE);
        tf.add(TimestampSource::EmbeddedMetadata, UNIX_BASE + 10.0);
        tf.add(TimestampSource::FilesystemCreation, UNIX_BASE + 20.0);
        assert_eq!(tf.observation_count(), 3);
    }

    #[test]
    fn test_time_anomaly_report_max_delta() {
        let mut tf = TimeForensics::new();
        // GPS vs NTP: 200 s gap (suspicious)
        tf.add(TimestampSource::GpsSynchronized, UNIX_BASE);
        tf.add(TimestampSource::Ntp, UNIX_BASE + 200.0);
        // Container vs Embedded: 90 s gap (suspicious if both medium-rel)
        tf.add(TimestampSource::ContainerHeader, UNIX_BASE + 50.0);
        tf.add(TimestampSource::EmbeddedMetadata, UNIX_BASE + 140.0);
        let report = tf.analyze_timestamps();
        assert!(report.max_delta_seconds() >= 200.0);
    }

    #[test]
    fn test_time_anomaly_report_empty() {
        let r = TimeAnomalyReport::new();
        assert_eq!(r.suspicious_count(), 0);
        assert!(!r.has_anomalies());
        assert_eq!(r.max_delta_seconds(), 0.0);
    }

    #[test]
    fn test_time_forensics_add_observation() {
        let mut tf = TimeForensics::new();
        tf.add_observation(TimestampObservation::new(
            TimestampSource::GpsSynchronized,
            UNIX_BASE,
        ));
        assert_eq!(tf.observation_count(), 1);
    }
}
