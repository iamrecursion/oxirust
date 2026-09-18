//! PTP (Precision Time Protocol) clock source — IEEE 1588-2019 / SMPTE ST 2059-2.
//!
//! Provides a software model of a PTP Ordinary Clock (OC) that can operate as:
//! - **Master** (Grandmaster candidate): advertises its clock identity.
//! - **Slave**: synchronises to the best-master clock via the Best Master Clock
//!   Algorithm (BMCA).
//!
//! This is a pure-Rust simulation layer.  Real hardware PTP relies on kernel
//! `SO_TIMESTAMPING` and NIC hardware assist; those are wired up via the
//! optional `ptp-hardware` feature (not yet enabled).
//!
//! ## Clock accuracy classes (IEEE 1588 Table 5)
//!
//! | Class | Description                       |
//! |-------|-----------------------------------|
//! | 6     | UTC traceable, ≤ 25 ns            |
//! | 7     | UTC traceable, ≤ 100 ns           |
//! | 52    | Free-run (atomic reference)       |
//! | 135   | Free-run (OCXO)                   |
//! | 248   | Default (not traceable)           |

#![allow(dead_code)]

use crate::{PlayoutError, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// PTP clock identity (64-bit EUI-64 derived from MAC address).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ClockIdentity(pub [u8; 8]);

impl ClockIdentity {
    /// Derive a stable identity from a MAC address (EUI-48 → EUI-64 via FFFE insertion).
    pub fn from_mac(mac: [u8; 6]) -> Self {
        let mut id = [0u8; 8];
        id[0] = mac[0] ^ 0x02; // toggle U/L bit
        id[1] = mac[1];
        id[2] = mac[2];
        id[3] = 0xFF;
        id[4] = 0xFE;
        id[5] = mac[3];
        id[6] = mac[4];
        id[7] = mac[5];
        Self(id)
    }

    /// Create a deterministic test identity from a u64 seed.
    pub fn from_seed(seed: u64) -> Self {
        Self(seed.to_be_bytes())
    }
}

impl std::fmt::Display for ClockIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let b = &self.0;
        write!(
            f,
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
        )
    }
}

/// Clock accuracy class per IEEE 1588 Table 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum ClockAccuracy {
    /// ≤ 25 ns (UTC traceable)
    Ns25 = 6,
    /// ≤ 100 ns (UTC traceable)
    Ns100 = 7,
    /// ≤ 250 ns
    Ns250 = 31,
    /// ≤ 1 µs
    Us1 = 32,
    /// ≤ 2.5 µs
    Us2_5 = 33,
    /// Free-run atomic
    AtomicFreeRun = 52,
    /// Free-run OCXO
    OcxoFreeRun = 135,
    /// Default (unknown)
    Default = 248,
}

/// PTP operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PtpMode {
    /// This node is a master (Grandmaster) clock.
    Master,
    /// This node synchronises to the best available master.
    Slave,
    /// Automatic: participates in BMCA and follows best master.
    Auto,
}

/// PTP clock dataset (per IEEE 1588-2019 §8.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClockDataset {
    /// Clock identity.
    pub clock_identity: ClockIdentity,
    /// Clock class.
    pub clock_class: u8,
    /// Clock accuracy.
    pub clock_accuracy: ClockAccuracy,
    /// Offset scaled log variance (encoded per spec; use 0xFFFF for unknown).
    pub offset_scaled_log_variance: u16,
    /// Priority 1 (0 = highest, 255 = lowest).
    pub priority1: u8,
    /// Priority 2 (tiebreaker within same priority 1).
    pub priority2: u8,
    /// Domain number (0–127).
    pub domain: u8,
}

impl Default for ClockDataset {
    fn default() -> Self {
        Self {
            clock_identity: ClockIdentity::from_seed(0x0102_0304_0506_0708),
            clock_class: 248,
            clock_accuracy: ClockAccuracy::Default,
            offset_scaled_log_variance: 0xFFFF,
            priority1: 128,
            priority2: 128,
            domain: 0,
        }
    }
}

/// BMCA comparison result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BmcaResult {
    /// Self is better master.
    SelfBetter,
    /// Other is better master.
    OtherBetter,
    /// Equal (tie-break by identity).
    TieByIdentity,
}

/// Best-Master Clock Algorithm (BMCA) as per IEEE 1588-2019 §9.3.3.
///
/// Returns which of the two datasets is the "better" master.
pub fn bmca_compare(a: &ClockDataset, b: &ClockDataset) -> BmcaResult {
    // Step 1: priority1
    if a.priority1 < b.priority1 {
        return BmcaResult::SelfBetter;
    }
    if b.priority1 < a.priority1 {
        return BmcaResult::OtherBetter;
    }
    // Step 2: clock_class
    if a.clock_class < b.clock_class {
        return BmcaResult::SelfBetter;
    }
    if b.clock_class < a.clock_class {
        return BmcaResult::OtherBetter;
    }
    // Step 3: clock_accuracy (lower ordinal = better)
    if (a.clock_accuracy as u8) < (b.clock_accuracy as u8) {
        return BmcaResult::SelfBetter;
    }
    if (b.clock_accuracy as u8) < (a.clock_accuracy as u8) {
        return BmcaResult::OtherBetter;
    }
    // Step 4: offset_scaled_log_variance (lower = better)
    if a.offset_scaled_log_variance < b.offset_scaled_log_variance {
        return BmcaResult::SelfBetter;
    }
    if b.offset_scaled_log_variance < a.offset_scaled_log_variance {
        return BmcaResult::OtherBetter;
    }
    // Step 5: priority2
    if a.priority2 < b.priority2 {
        return BmcaResult::SelfBetter;
    }
    if b.priority2 < a.priority2 {
        return BmcaResult::OtherBetter;
    }
    // Step 6: tie-break by clock identity (lower wins)
    if a.clock_identity < b.clock_identity {
        BmcaResult::SelfBetter
    } else if b.clock_identity < a.clock_identity {
        BmcaResult::OtherBetter
    } else {
        BmcaResult::TieByIdentity
    }
}

// ---------------------------------------------------------------------------
// Servo filter
// ---------------------------------------------------------------------------

/// A simple proportional-integral (PI) servo controller for PTP clock offset
/// correction.  Mirrors the approach used in `linuxptp` / `ptpd`.
#[derive(Debug)]
pub struct PiServo {
    /// Proportional gain (seconds correction per nanosecond offset).
    kp: f64,
    /// Integral gain.
    ki: f64,
    /// Integral accumulator (nanoseconds).
    integrator: f64,
    /// Maximum integrator value (anti-windup, nanoseconds).
    max_integrator: f64,
}

impl Default for PiServo {
    fn default() -> Self {
        // Conservative defaults suitable for a 25 fps broadcast clock.
        Self {
            kp: 0.7,
            ki: 0.3,
            integrator: 0.0,
            max_integrator: 200.0e9, // 200 s
        }
    }
}

impl PiServo {
    pub fn new(kp: f64, ki: f64) -> Self {
        Self {
            kp,
            ki,
            integrator: 0.0,
            max_integrator: 200.0e9,
        }
    }

    /// Feed an offset measurement (in nanoseconds) and return the frequency
    /// adjustment in parts-per-billion (ppb).
    pub fn update(&mut self, offset_ns: f64) -> f64 {
        self.integrator = (self.integrator + offset_ns)
            .max(-self.max_integrator)
            .min(self.max_integrator);
        -(self.kp * offset_ns + self.ki * self.integrator)
    }

    pub fn reset(&mut self) {
        self.integrator = 0.0;
    }
}

// ---------------------------------------------------------------------------
// PTP Timestamp
// ---------------------------------------------------------------------------

/// A PTP timestamp (seconds + nanoseconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PtpTimestamp {
    /// Seconds since PTP epoch (TAI, 1970-01-01 00:00:00).
    pub seconds: u64,
    /// Nanoseconds sub-second part.
    pub nanoseconds: u32,
}

impl PtpTimestamp {
    pub const ZERO: Self = Self {
        seconds: 0,
        nanoseconds: 0,
    };

    /// Create a PTP timestamp from a Rust `SystemTime`.
    ///
    /// Note: `SystemTime` is UTC-based; TAI = UTC + leap_seconds.  This
    /// implementation uses a fixed offset of 37 seconds (correct as of 2024).
    pub fn from_system_time(t: SystemTime) -> Self {
        const TAI_OFFSET_SECS: u64 = 37;
        let since_epoch = t.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        let tai_secs = since_epoch.as_secs() + TAI_OFFSET_SECS;
        Self {
            seconds: tai_secs,
            nanoseconds: since_epoch.subsec_nanos(),
        }
    }

    /// Convert to a total nanosecond count (for arithmetic).
    pub fn to_nanos(&self) -> u128 {
        self.seconds as u128 * 1_000_000_000 + self.nanoseconds as u128
    }

    /// Compute signed offset (self − other) in nanoseconds.
    pub fn offset_ns_from(&self, other: &Self) -> i64 {
        let a = self.to_nanos() as i128;
        let b = other.to_nanos() as i128;
        (a - b) as i64
    }
}

// ---------------------------------------------------------------------------
// PTP Clock
// ---------------------------------------------------------------------------

/// Configuration for the PTP clock source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtpClockConfig {
    /// PTP domain number (0–127).
    pub domain: u8,
    /// Operating mode.
    pub mode: PtpMode,
    /// This clock's dataset.
    pub clock_dataset: ClockDataset,
    /// Network interface name (e.g. "eth0").
    pub interface: String,
    /// Announce interval (log2 seconds, e.g. 1 = every 2 s).
    pub log_announce_interval: i8,
    /// Sync interval (log2 seconds, e.g. -2 = 4 per second).
    pub log_sync_interval: i8,
    /// Delay request interval (log2 seconds).
    pub log_delay_req_interval: i8,
    /// Enable hardware timestamping (requires kernel support).
    pub hw_timestamp: bool,
}

impl Default for PtpClockConfig {
    fn default() -> Self {
        Self {
            domain: 127, // SMPTE ST 2059-2 broadcast domain
            mode: PtpMode::Slave,
            clock_dataset: ClockDataset::default(),
            interface: "eth0".to_string(),
            log_announce_interval: 1,  // 2 s
            log_sync_interval: -2,     // 4/s (SMPTE recommended)
            log_delay_req_interval: 0, // 1/s
            hw_timestamp: false,
        }
    }
}

/// Statistics collected by the PTP clock.
#[derive(Debug, Clone, Default)]
pub struct PtpStats {
    /// Number of sync messages received (slave mode).
    pub sync_received: u64,
    /// Number of delay-request / delay-response exchanges completed.
    pub delay_exchanges: u64,
    /// Last measured mean path delay (nanoseconds).
    pub mean_path_delay_ns: i64,
    /// Last measured offset from master (nanoseconds).
    pub offset_from_master_ns: i64,
    /// RMS jitter of last 16 offset samples (nanoseconds).
    pub jitter_rms_ns: f64,
    /// Current frequency adjustment applied by the servo (ppb).
    pub freq_adjustment_ppb: f64,
    /// Whether the clock is currently locked to a master.
    pub locked: bool,
}

/// PTP clock source implementation.
///
/// In slave mode the clock periodically ingests synthetic sync/delay messages
/// and drives a PI servo to converge the local offset.  In master mode it
/// advertises announce messages.
pub struct PtpClock {
    config: PtpClockConfig,
    /// Current PTP time offset applied to system clock (nanoseconds, signed).
    offset_ns: Arc<AtomicI64>,
    /// Monotonic sequence counter for sync/delay packets.
    sequence: Arc<AtomicU64>,
    /// Servo controller.
    servo: parking_lot::Mutex<PiServo>,
    /// Circular buffer of the last 16 offset samples (for jitter calculation).
    offset_history: parking_lot::Mutex<[i64; 16]>,
    history_pos: Arc<AtomicU64>,
    /// Stats (writes protected by Mutex, reads from atomic snapshot).
    stats: parking_lot::Mutex<PtpStats>,
}

impl PtpClock {
    /// Create a new PTP clock with the given configuration.
    pub fn new(config: PtpClockConfig) -> Self {
        Self {
            config,
            offset_ns: Arc::new(AtomicI64::new(0)),
            sequence: Arc::new(AtomicU64::new(0)),
            servo: parking_lot::Mutex::new(PiServo::default()),
            offset_history: parking_lot::Mutex::new([0i64; 16]),
            history_pos: Arc::new(AtomicU64::new(0)),
            stats: parking_lot::Mutex::new(PtpStats::default()),
        }
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &PtpClockConfig {
        &self.config
    }

    /// Return the current corrected PTP time.
    pub fn now(&self) -> PtpTimestamp {
        let base = PtpTimestamp::from_system_time(SystemTime::now());
        let correction_ns = self.offset_ns.load(Ordering::Relaxed);
        // Apply correction: add offset to nanosecond field, carry into seconds.
        let total_ns = base.to_nanos() as i128 + correction_ns as i128;
        let total_ns_u = total_ns.max(0) as u128;
        PtpTimestamp {
            seconds: (total_ns_u / 1_000_000_000) as u64,
            nanoseconds: (total_ns_u % 1_000_000_000) as u32,
        }
    }

    /// Simulate reception of a Sync message from a master.
    ///
    /// `t1` is the master's transmit timestamp (in the Sync message).
    /// `t2` is our receive timestamp.
    /// `mean_path_delay_ns` is the previously measured one-way delay.
    pub fn on_sync(
        &self,
        t1: PtpTimestamp,
        t2: PtpTimestamp,
        mean_path_delay_ns: i64,
    ) -> Result<()> {
        if self.config.mode == PtpMode::Master {
            return Err(PlayoutError::Ptp(
                "on_sync called on master clock".to_string(),
            ));
        }

        // Offset from master = T2 - T1 - mean_path_delay
        let raw_offset = t2.offset_ns_from(&t1) - mean_path_delay_ns;

        // Update history ring buffer.
        let pos = self.history_pos.fetch_add(1, Ordering::Relaxed) as usize % 16;
        self.offset_history.lock()[pos] = raw_offset;

        // Drive servo.
        let freq_adj_ppb = self.servo.lock().update(raw_offset as f64);

        // Apply the frequency correction to the offset accumulator.
        // Frequency adjustment in ppb: Δns per second.
        // We integrate over an approximate sync interval.
        let sync_interval_s = 2_f64.powi(self.config.log_sync_interval as i32);
        let delta_ns = (freq_adj_ppb * 1e-9 * sync_interval_s * 1e9) as i64;
        self.offset_ns.fetch_add(delta_ns, Ordering::Relaxed);

        // Compute jitter as RMS of deviations from the mean offset (last 16 samples).
        // This measures timing stability, not absolute offset magnitude.
        let hist = *self.offset_history.lock();
        let mean: f64 = hist.iter().map(|&v| v as f64).sum::<f64>() / 16.0;
        let sum_sq: f64 = hist.iter().map(|&v| (v as f64 - mean).powi(2)).sum();
        let jitter_rms = (sum_sq / 16.0).sqrt();

        // Update stats.
        let mut stats = self.stats.lock();
        stats.sync_received += 1;
        stats.offset_from_master_ns = raw_offset;
        stats.mean_path_delay_ns = mean_path_delay_ns;
        stats.freq_adjustment_ppb = freq_adj_ppb;
        stats.jitter_rms_ns = jitter_rms;
        // Consider locked when jitter < 1 µs.
        stats.locked = jitter_rms < 1000.0;

        Ok(())
    }

    /// Simulate a complete delay-request / delay-response exchange.
    ///
    /// `t3` = delay-request transmit time.
    /// `t4` = delay-request receive time at master.
    ///
    /// Returns the calculated mean path delay in nanoseconds.
    pub fn on_delay_response(&self, t3: PtpTimestamp, t4: PtpTimestamp) -> i64 {
        // Mean path delay = (T2 - T1 + T4 - T3) / 2
        // Here we use (T4 - T3) as the forward path component.
        let fwd_ns = t4.offset_ns_from(&t3);
        let delay_ns = fwd_ns / 2;

        let mut stats = self.stats.lock();
        stats.delay_exchanges += 1;
        stats.mean_path_delay_ns = delay_ns;

        delay_ns
    }

    /// Get a snapshot of current PTP statistics.
    pub fn stats(&self) -> PtpStats {
        self.stats.lock().clone()
    }

    /// Reset the servo and offset accumulator.
    pub fn reset(&self) {
        self.servo.lock().reset();
        self.offset_ns.store(0, Ordering::Relaxed);
        *self.stats.lock() = PtpStats::default();
        *self.offset_history.lock() = [0i64; 16];
        self.history_pos.store(0, Ordering::Relaxed);
        self.sequence.store(0, Ordering::Relaxed);
    }

    /// Return the current offset from master (nanoseconds).
    pub fn offset_ns(&self) -> i64 {
        self.offset_ns.load(Ordering::Relaxed)
    }

    /// Check whether the clock is locked to a master.
    pub fn is_locked(&self) -> bool {
        self.stats.lock().locked
    }

    /// Run the BMCA against a received Announce message's clock dataset.
    ///
    /// Returns `true` if the remote clock should be accepted as the new master.
    pub fn bmca_evaluate(&self, remote: &ClockDataset) -> bool {
        let result = bmca_compare(&self.config.clock_dataset, remote);
        result == BmcaResult::OtherBetter
    }
}

/// Named clock source, covering internal/SDI/PTP and custom sources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClockSource {
    /// Internal software clock (default).
    Internal,
    /// SDI tri-level sync reference.
    Sdi,
    /// PTP (IEEE 1588) hardware/software clock.
    Ptp { domain: u8, interface: String },
    /// GPS/GNSS disciplined oscillator.
    Gnss,
    /// Black burst / analogue reference (legacy).
    BlackBurst,
    /// External word clock (audio sync).
    WordClock,
}

impl ClockSource {
    /// Parse from a human-readable string (as stored in `PlayoutConfig.clock_source`).
    pub fn from_str(s: &str) -> Self {
        match s {
            "internal" => Self::Internal,
            "sdi" => Self::Sdi,
            "gnss" => Self::Gnss,
            "black_burst" | "blackburst" => Self::BlackBurst,
            "word_clock" | "wordclock" => Self::WordClock,
            _ if s.starts_with("ptp") => {
                // Accept formats: "ptp", "ptp:127", "ptp:127:eth0"
                let parts: Vec<&str> = s.splitn(3, ':').collect();
                let domain = parts
                    .get(1)
                    .and_then(|d| d.parse::<u8>().ok())
                    .unwrap_or(127);
                let interface = parts.get(2).copied().unwrap_or("eth0").to_string();
                Self::Ptp { domain, interface }
            }
            _ => Self::Internal,
        }
    }

    /// Convert to canonical string representation.
    pub fn to_config_string(&self) -> String {
        match self {
            Self::Internal => "internal".to_string(),
            Self::Sdi => "sdi".to_string(),
            Self::Ptp { domain, interface } => format!("ptp:{domain}:{interface}"),
            Self::Gnss => "gnss".to_string(),
            Self::BlackBurst => "black_burst".to_string(),
            Self::WordClock => "word_clock".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_identity_from_mac() {
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let id = ClockIdentity::from_mac(mac);
        // U/L bit toggled on first octet: 0xAA ^ 0x02 = 0xA8
        assert_eq!(id.0[0], 0xA8);
        assert_eq!(id.0[3], 0xFF);
        assert_eq!(id.0[4], 0xFE);
    }

    #[test]
    fn test_clock_identity_display() {
        let id = ClockIdentity([0x00, 0x01, 0x02, 0xFF, 0xFE, 0x03, 0x04, 0x05]);
        let s = id.to_string();
        assert!(s.contains("ff:fe"));
    }

    #[test]
    fn test_bmca_compare_priority1() {
        let mut a = ClockDataset::default();
        let mut b = ClockDataset::default();
        a.priority1 = 100;
        b.priority1 = 200;
        assert_eq!(bmca_compare(&a, &b), BmcaResult::SelfBetter);
        assert_eq!(bmca_compare(&b, &a), BmcaResult::OtherBetter);
    }

    #[test]
    fn test_bmca_compare_clock_class() {
        let mut a = ClockDataset::default();
        let mut b = ClockDataset::default();
        a.clock_class = 6;
        b.clock_class = 135;
        assert_eq!(bmca_compare(&a, &b), BmcaResult::SelfBetter);
    }

    #[test]
    fn test_bmca_compare_tie_by_identity() {
        let a = ClockDataset::default();
        let b = ClockDataset::default();
        // Identical datasets → tie by identity (same identity → TieByIdentity)
        assert_eq!(bmca_compare(&a, &b), BmcaResult::TieByIdentity);
    }

    #[test]
    fn test_pi_servo_converges() {
        let mut servo = PiServo::default();
        // Apply repeated positive offsets; frequency adjustment should grow negative.
        let adj1 = servo.update(1000.0);
        let adj2 = servo.update(1000.0);
        assert!(adj1 < 0.0);
        assert!(adj2 < adj1); // integrator is accumulating
    }

    #[test]
    fn test_pi_servo_reset() {
        let mut servo = PiServo::default();
        servo.update(1000.0);
        servo.reset();
        let adj = servo.update(0.0);
        assert_eq!(adj, 0.0);
    }

    #[test]
    fn test_ptp_timestamp_from_system_time() {
        let t = PtpTimestamp::from_system_time(UNIX_EPOCH);
        // At UNIX epoch, TAI = 37 s (leap seconds offset)
        assert_eq!(t.seconds, 37);
        assert_eq!(t.nanoseconds, 0);
    }

    #[test]
    fn test_ptp_timestamp_offset_ns() {
        let t1 = PtpTimestamp {
            seconds: 100,
            nanoseconds: 0,
        };
        let t2 = PtpTimestamp {
            seconds: 100,
            nanoseconds: 500,
        };
        assert_eq!(t2.offset_ns_from(&t1), 500);
        assert_eq!(t1.offset_ns_from(&t2), -500);
    }

    #[test]
    fn test_ptp_clock_now_is_consistent() {
        let cfg = PtpClockConfig::default();
        let clock = PtpClock::new(cfg);
        let t1 = clock.now();
        let t2 = clock.now();
        assert!(t2 >= t1);
    }

    #[test]
    fn test_ptp_clock_on_sync_slave() {
        let cfg = PtpClockConfig {
            mode: PtpMode::Slave,
            ..Default::default()
        };
        let clock = PtpClock::new(cfg);

        let t1 = PtpTimestamp {
            seconds: 1_000_000,
            nanoseconds: 0,
        };
        let t2 = PtpTimestamp {
            seconds: 1_000_000,
            nanoseconds: 500_000, // 0.5 ms later
        };
        // Feed 16 identical sync messages to warm the jitter window.
        for _ in 0..16 {
            clock
                .on_sync(t1, t2, 100_000)
                .expect("on_sync should succeed");
        }
        let stats = clock.stats();
        assert!(stats.sync_received >= 16);
        // With consistent samples, jitter should be ~0, clock should lock.
        assert!(stats.locked);
    }

    #[test]
    fn test_ptp_clock_on_sync_master_returns_error() {
        let cfg = PtpClockConfig {
            mode: PtpMode::Master,
            ..Default::default()
        };
        let clock = PtpClock::new(cfg);
        let t = PtpTimestamp::ZERO;
        let result = clock.on_sync(t, t, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ptp_clock_delay_response() {
        let cfg = PtpClockConfig::default();
        let clock = PtpClock::new(cfg);
        let t3 = PtpTimestamp {
            seconds: 1000,
            nanoseconds: 0,
        };
        let t4 = PtpTimestamp {
            seconds: 1000,
            nanoseconds: 200_000, // 0.2 ms fwd delay
        };
        let delay = clock.on_delay_response(t3, t4);
        assert_eq!(delay, 100_000); // half of 200_000
    }

    #[test]
    fn test_ptp_clock_reset() {
        let cfg = PtpClockConfig::default();
        let clock = PtpClock::new(cfg);
        let t = PtpTimestamp {
            seconds: 1_000_000,
            nanoseconds: 0,
        };
        clock.on_sync(t, t, 0).expect("on_sync should succeed");
        clock.reset();
        assert_eq!(clock.offset_ns(), 0);
        assert_eq!(clock.stats().sync_received, 0);
    }

    #[test]
    fn test_ptp_clock_bmca_evaluate() {
        let mut cfg = PtpClockConfig::default();
        cfg.clock_dataset.priority1 = 200; // poor local clock
        let clock = PtpClock::new(cfg);

        let mut remote = ClockDataset::default();
        remote.priority1 = 100; // better remote
        assert!(clock.bmca_evaluate(&remote));
    }

    #[test]
    fn test_clock_source_from_str() {
        assert_eq!(ClockSource::from_str("internal"), ClockSource::Internal);
        assert_eq!(ClockSource::from_str("sdi"), ClockSource::Sdi);
        if let ClockSource::Ptp { domain, interface } = ClockSource::from_str("ptp:127:eth1") {
            assert_eq!(domain, 127);
            assert_eq!(interface, "eth1");
        } else {
            panic!("expected Ptp variant");
        }
        assert_eq!(
            ClockSource::from_str("ptp"),
            ClockSource::Ptp {
                domain: 127,
                interface: "eth0".to_string()
            }
        );
    }

    #[test]
    fn test_clock_source_roundtrip() {
        let src = ClockSource::Ptp {
            domain: 0,
            interface: "bond0".to_string(),
        };
        let s = src.to_config_string();
        let parsed = ClockSource::from_str(&s);
        assert_eq!(src, parsed);
    }
}
