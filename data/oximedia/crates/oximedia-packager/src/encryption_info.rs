#![allow(dead_code)]
//! Encryption metadata management for adaptive streaming packages.
//!
//! Tracks per-segment and per-key-period encryption state, including
//! key IDs, IVs, key rotation schedules, and PSSH box generation hints.
//! This module is purely informational -- it does not perform actual
//! cryptographic operations but organises the metadata needed by
//! manifest generators and DRM licence servers.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Supported DRM system identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrmSystem {
    /// Widevine (Google).
    Widevine,
    /// `PlayReady` (Microsoft).
    PlayReady,
    /// `FairPlay` (Apple).
    FairPlay,
    /// `ClearKey` (W3C).
    ClearKey,
}

impl fmt::Display for DrmSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Widevine => "Widevine",
            Self::PlayReady => "PlayReady",
            Self::FairPlay => "FairPlay",
            Self::ClearKey => "ClearKey",
        };
        write!(f, "{label}")
    }
}

impl DrmSystem {
    /// Return the standard DASH system ID (UUID) for this DRM system.
    #[must_use]
    pub fn system_id(&self) -> &'static str {
        match self {
            Self::Widevine => "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed",
            Self::PlayReady => "9a04f079-9840-4286-ab92-e65be0885f95",
            Self::FairPlay => "94ce86fb-07ff-4f43-adb8-93d2fa968ca2",
            Self::ClearKey => "e2719d58-a985-b3c9-781a-b030af78d30e",
        }
    }
}

/// A key period describes a time range covered by a single content key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyPeriod {
    /// Unique identifier for this key period.
    pub id: String,
    /// Key ID (KID) as a 16-byte array formatted in hex.
    pub key_id: String,
    /// Start time of this key period.
    pub start: Duration,
    /// End time of this key period (`None` = open-ended / last period).
    pub end: Option<Duration>,
    /// Initialisation vector template (hex string).
    pub iv_template: Option<String>,
}

impl KeyPeriod {
    /// Create a new key period.
    #[must_use]
    pub fn new(id: impl Into<String>, key_id: impl Into<String>, start: Duration) -> Self {
        Self {
            id: id.into(),
            key_id: key_id.into(),
            start,
            end: None,
            iv_template: None,
        }
    }

    /// Set the end time.
    #[must_use]
    pub fn with_end(mut self, end: Duration) -> Self {
        self.end = Some(end);
        self
    }

    /// Set the IV template.
    #[must_use]
    pub fn with_iv_template(mut self, iv: impl Into<String>) -> Self {
        self.iv_template = Some(iv.into());
        self
    }

    /// Duration of this key period.
    #[must_use]
    pub fn duration(&self) -> Option<Duration> {
        self.end.map(|e| e.saturating_sub(self.start))
    }

    /// Check whether a given time falls within this period.
    #[must_use]
    pub fn contains_time(&self, t: Duration) -> bool {
        if t < self.start {
            return false;
        }
        match self.end {
            Some(end) => t < end,
            None => true,
        }
    }
}

/// Per-segment encryption information.
#[derive(Debug, Clone)]
pub struct SegmentEncryptionInfo {
    /// Segment number.
    pub segment_number: u64,
    /// Key period ID that covers this segment.
    pub key_period_id: String,
    /// The IV used for this specific segment (hex string).
    pub iv: String,
    /// Whether this segment required a key change (first segment of a new period).
    pub key_change: bool,
}

/// A key rotation schedule that assigns key periods to time ranges.
#[derive(Debug, Clone)]
pub struct KeyRotationSchedule {
    /// Ordered list of key periods.
    periods: Vec<KeyPeriod>,
    /// Target rotation interval.
    pub rotation_interval: Duration,
}

impl KeyRotationSchedule {
    /// Create a new rotation schedule with the given interval.
    #[must_use]
    pub fn new(rotation_interval: Duration) -> Self {
        Self {
            periods: Vec::new(),
            rotation_interval,
        }
    }

    /// Add a key period.
    pub fn add_period(&mut self, period: KeyPeriod) {
        self.periods.push(period);
    }

    /// Return the number of key periods.
    #[must_use]
    pub fn period_count(&self) -> usize {
        self.periods.len()
    }

    /// Find the key period covering the given time.
    #[must_use]
    pub fn period_for_time(&self, t: Duration) -> Option<&KeyPeriod> {
        self.periods.iter().find(|p| p.contains_time(t))
    }

    /// Return all key periods.
    #[must_use]
    pub fn periods(&self) -> &[KeyPeriod] {
        &self.periods
    }
}

/// Aggregated encryption information for an entire packaged output.
#[derive(Debug, Clone)]
pub struct EncryptionInfo {
    /// Active DRM systems.
    pub drm_systems: Vec<DrmSystem>,
    /// Key rotation schedule.
    pub rotation: KeyRotationSchedule,
    /// Per-segment encryption details (segment number -> info).
    pub segment_info: HashMap<u64, SegmentEncryptionInfo>,
    /// PSSH data per DRM system (hex-encoded).
    pub pssh_data: HashMap<DrmSystem, String>,
}

impl EncryptionInfo {
    /// Create empty encryption info with a given rotation interval.
    #[must_use]
    pub fn new(rotation_interval: Duration) -> Self {
        Self {
            drm_systems: Vec::new(),
            rotation: KeyRotationSchedule::new(rotation_interval),
            segment_info: HashMap::new(),
            pssh_data: HashMap::new(),
        }
    }

    /// Register a DRM system.
    pub fn add_drm_system(&mut self, system: DrmSystem) {
        if !self.drm_systems.contains(&system) {
            self.drm_systems.push(system);
        }
    }

    /// Set PSSH data for a DRM system.
    pub fn set_pssh(&mut self, system: DrmSystem, data: impl Into<String>) {
        self.pssh_data.insert(system, data.into());
    }

    /// Record per-segment encryption info.
    pub fn record_segment(&mut self, info: SegmentEncryptionInfo) {
        self.segment_info.insert(info.segment_number, info);
    }

    /// Number of distinct key changes.
    #[must_use]
    pub fn key_change_count(&self) -> usize {
        self.segment_info.values().filter(|s| s.key_change).count()
    }

    /// Check whether any DRM systems are configured.
    #[must_use]
    pub fn has_drm(&self) -> bool {
        !self.drm_systems.is_empty()
    }
}

/// Compute an IV for a given segment number using a simple counter scheme.
///
/// Returns a 32-character hex string (16 bytes).
#[must_use]
pub fn compute_segment_iv(segment_number: u64) -> String {
    format!("{segment_number:032x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_drm_system_display() {
        assert_eq!(DrmSystem::Widevine.to_string(), "Widevine");
        assert_eq!(DrmSystem::FairPlay.to_string(), "FairPlay");
    }

    #[test]
    fn test_drm_system_id() {
        assert!(DrmSystem::Widevine.system_id().contains("edef8ba9"));
        assert!(DrmSystem::PlayReady.system_id().contains("9a04f079"));
    }

    #[test]
    fn test_key_period_creation() {
        let kp = KeyPeriod::new("kp1", "aabbccdd", Duration::from_secs(0));
        assert_eq!(kp.id, "kp1");
        assert_eq!(kp.key_id, "aabbccdd");
        assert!(kp.end.is_none());
    }

    #[test]
    fn test_key_period_with_end() {
        let kp =
            KeyPeriod::new("kp1", "aa", Duration::from_secs(0)).with_end(Duration::from_secs(60));
        assert_eq!(kp.duration(), Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_key_period_contains_time() {
        let kp =
            KeyPeriod::new("kp1", "aa", Duration::from_secs(10)).with_end(Duration::from_secs(20));
        assert!(!kp.contains_time(Duration::from_secs(5)));
        assert!(kp.contains_time(Duration::from_secs(10)));
        assert!(kp.contains_time(Duration::from_secs(15)));
        assert!(!kp.contains_time(Duration::from_secs(20)));
    }

    #[test]
    fn test_key_period_open_ended() {
        let kp = KeyPeriod::new("kp1", "aa", Duration::from_secs(10));
        assert!(kp.contains_time(Duration::from_secs(1000)));
        assert!(kp.duration().is_none());
    }

    #[test]
    fn test_rotation_schedule() {
        let mut sched = KeyRotationSchedule::new(Duration::from_secs(60));
        sched.add_period(
            KeyPeriod::new("kp1", "aa", Duration::ZERO).with_end(Duration::from_secs(60)),
        );
        sched.add_period(KeyPeriod::new("kp2", "bb", Duration::from_secs(60)));
        assert_eq!(sched.period_count(), 2);
        assert_eq!(
            sched
                .period_for_time(Duration::from_secs(30))
                .expect("should succeed in test")
                .id,
            "kp1"
        );
        assert_eq!(
            sched
                .period_for_time(Duration::from_secs(90))
                .expect("should succeed in test")
                .id,
            "kp2"
        );
    }

    #[test]
    fn test_encryption_info_add_drm() {
        let mut info = EncryptionInfo::new(Duration::from_secs(60));
        info.add_drm_system(DrmSystem::Widevine);
        info.add_drm_system(DrmSystem::Widevine); // duplicate
        assert_eq!(info.drm_systems.len(), 1);
        assert!(info.has_drm());
    }

    #[test]
    fn test_encryption_info_pssh() {
        let mut info = EncryptionInfo::new(Duration::from_secs(60));
        info.set_pssh(DrmSystem::Widevine, "deadbeef");
        assert_eq!(info.pssh_data[&DrmSystem::Widevine], "deadbeef");
    }

    #[test]
    fn test_record_segment() {
        let mut info = EncryptionInfo::new(Duration::from_secs(60));
        info.record_segment(SegmentEncryptionInfo {
            segment_number: 0,
            key_period_id: "kp1".to_string(),
            iv: compute_segment_iv(0),
            key_change: true,
        });
        info.record_segment(SegmentEncryptionInfo {
            segment_number: 1,
            key_period_id: "kp1".to_string(),
            iv: compute_segment_iv(1),
            key_change: false,
        });
        assert_eq!(info.key_change_count(), 1);
    }

    #[test]
    fn test_compute_segment_iv() {
        let iv = compute_segment_iv(0);
        assert_eq!(iv.len(), 32);
        assert_eq!(iv, "00000000000000000000000000000000");
        let iv1 = compute_segment_iv(1);
        assert_eq!(iv1, "00000000000000000000000000000001");
    }

    #[test]
    fn test_encryption_info_no_drm() {
        let info = EncryptionInfo::new(Duration::from_secs(60));
        assert!(!info.has_drm());
    }

    #[test]
    fn test_iv_template() {
        let kp = KeyPeriod::new("kp1", "aa", Duration::ZERO)
            .with_iv_template("00112233445566778899aabbccddeeff");
        assert_eq!(
            kp.iv_template.as_deref(),
            Some("00112233445566778899aabbccddeeff")
        );
    }
}
