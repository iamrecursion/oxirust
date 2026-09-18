//! RIST (Reliable Internet Stream Transport) protocol support.
//!
//! Implements RIST Simple, Main, and Advanced profiles for reliable
//! media transport over lossy IP networks using ARQ retransmission.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// RIST profile level.
///
/// Each profile adds capabilities over the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RistProfile {
    /// Simple Profile – single-path, no bonding, no encryption.
    Simple,
    /// Main Profile – bonding support, optional encryption.
    Main,
    /// Advanced Profile – full bonding, encryption, and tunnelling.
    Advanced,
}

impl RistProfile {
    /// Returns `true` if this profile supports link bonding (multi-path).
    #[must_use]
    pub fn supports_bonding(self) -> bool {
        matches!(self, Self::Main | Self::Advanced)
    }

    /// Returns `true` if this profile supports AES encryption.
    #[must_use]
    pub fn supports_encryption(self) -> bool {
        matches!(self, Self::Main | Self::Advanced)
    }

    /// Returns a human-readable name for the profile.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Simple => "RIST Simple",
            Self::Main => "RIST Main",
            Self::Advanced => "RIST Advanced",
        }
    }
}

/// Configuration for a RIST sender or receiver.
#[derive(Debug, Clone)]
pub struct RistConfig {
    /// RIST profile in use.
    pub profile: RistProfile,
    /// Re-order / jitter buffer in milliseconds.
    pub buffer_ms: u32,
    /// Maximum expected bitrate in kilobits per second.
    pub max_bitrate_kbps: u32,
    /// Acceptable packet loss percentage before alerting (0.0–100.0).
    pub loss_threshold_pct: f32,
}

impl RistConfig {
    /// Creates a new `RistConfig`.
    #[must_use]
    pub fn new(
        profile: RistProfile,
        buffer_ms: u32,
        max_bitrate_kbps: u32,
        loss_threshold_pct: f32,
    ) -> Self {
        Self {
            profile,
            buffer_ms,
            max_bitrate_kbps,
            loss_threshold_pct,
        }
    }

    /// Returns `true` if the buffer size targets low-latency (< 500 ms).
    #[must_use]
    pub fn is_low_latency(&self) -> bool {
        self.buffer_ms < 500
    }

    /// Returns the maximum payload bandwidth in bytes per second.
    #[must_use]
    pub fn max_bytes_per_second(&self) -> u64 {
        u64::from(self.max_bitrate_kbps) * 1_000 / 8
    }

    /// Returns a default low-latency Simple profile configuration.
    #[must_use]
    pub fn default_simple() -> Self {
        Self::new(RistProfile::Simple, 100, 20_000, 5.0)
    }

    /// Returns a default Main profile configuration suitable for WAN.
    #[must_use]
    pub fn default_main() -> Self {
        Self::new(RistProfile::Main, 1_000, 50_000, 10.0)
    }
}

impl Default for RistConfig {
    fn default() -> Self {
        Self::default_simple()
    }
}

/// Runtime statistics for a RIST session.
#[derive(Debug, Clone, Default)]
pub struct RistStats {
    /// Total packets sent (including retransmissions).
    pub packets_sent: u64,
    /// Total packets received (after ARQ recovery).
    pub packets_received: u64,
    /// Number of retransmission requests issued or fulfilled.
    pub retransmissions: u64,
    /// Raw loss percentage before retransmission (0.0–100.0).
    pub loss_pct: f32,
    /// Round-trip time in milliseconds.
    pub rtt_ms: f32,
}

impl RistStats {
    /// Creates a new `RistStats` instance.
    #[must_use]
    pub fn new(
        packets_sent: u64,
        packets_received: u64,
        retransmissions: u64,
        loss_pct: f32,
        rtt_ms: f32,
    ) -> Self {
        Self {
            packets_sent,
            packets_received,
            retransmissions,
            loss_pct,
            rtt_ms,
        }
    }

    /// Returns the effective packet loss after ARQ retransmissions.
    ///
    /// Computed as the fraction of sent packets never recovered.
    /// The result is clamped to `[0.0, 100.0]`.
    #[must_use]
    pub fn effective_loss_pct(&self) -> f32 {
        if self.packets_sent == 0 {
            return 0.0;
        }
        let recovered = self.retransmissions.min(self.packets_sent);
        let lost_before_recovery = self.packets_sent.saturating_sub(self.packets_received);
        let still_lost = lost_before_recovery.saturating_sub(recovered);
        let pct = still_lost as f32 / self.packets_sent as f32 * 100.0;
        pct.clamp(0.0, 100.0)
    }

    /// Returns a composite quality score in `[0.0, 1.0]`.
    ///
    /// Penalises both residual loss and high RTT (> 200 ms).
    #[must_use]
    pub fn quality_score(&self) -> f32 {
        let loss_factor = 1.0 - (self.effective_loss_pct() / 100.0).clamp(0.0, 1.0);
        let rtt_penalty = (self.rtt_ms / 200.0).clamp(0.0, 1.0);
        let rtt_factor = 1.0 - rtt_penalty * 0.5;
        (loss_factor * rtt_factor).clamp(0.0, 1.0)
    }
}

/// A bonded RIST connection consisting of multiple network paths.
///
/// Bonding (link aggregation) is only available in Main and Advanced profiles.
#[derive(Debug, Clone, Default)]
pub struct RistBond {
    /// List of `(ip_address, port)` pairs representing individual paths.
    bonds: Vec<(String, u16)>,
}

impl RistBond {
    /// Creates an empty `RistBond`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a bonded path.
    pub fn add_bond(&mut self, ip: impl Into<String>, port: u16) {
        self.bonds.push((ip.into(), port));
    }

    /// Returns the number of bonded paths.
    #[must_use]
    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    /// Returns a reference to the primary (first) bond, if any.
    #[must_use]
    pub fn primary(&self) -> Option<&(String, u16)> {
        self.bonds.first()
    }

    /// Returns a slice of all bonded paths.
    #[must_use]
    pub fn all_bonds(&self) -> &[(String, u16)] {
        &self.bonds
    }

    /// Returns `true` if there are no bonded paths.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bonds.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. RistProfile::supports_bonding
    #[test]
    fn test_simple_no_bonding() {
        assert!(!RistProfile::Simple.supports_bonding());
    }

    #[test]
    fn test_main_supports_bonding() {
        assert!(RistProfile::Main.supports_bonding());
    }

    #[test]
    fn test_advanced_supports_bonding() {
        assert!(RistProfile::Advanced.supports_bonding());
    }

    // 2. RistProfile::supports_encryption
    #[test]
    fn test_simple_no_encryption() {
        assert!(!RistProfile::Simple.supports_encryption());
    }

    #[test]
    fn test_main_supports_encryption() {
        assert!(RistProfile::Main.supports_encryption());
    }

    // 3. RistProfile::name
    #[test]
    fn test_profile_names() {
        assert_eq!(RistProfile::Simple.name(), "RIST Simple");
        assert_eq!(RistProfile::Main.name(), "RIST Main");
        assert_eq!(RistProfile::Advanced.name(), "RIST Advanced");
    }

    // 4. RistConfig::is_low_latency
    #[test]
    fn test_low_latency_true() {
        let cfg = RistConfig::new(RistProfile::Simple, 200, 10_000, 5.0);
        assert!(cfg.is_low_latency());
    }

    #[test]
    fn test_low_latency_false() {
        let cfg = RistConfig::new(RistProfile::Simple, 500, 10_000, 5.0);
        assert!(!cfg.is_low_latency());
    }

    // 5. RistConfig::max_bytes_per_second
    #[test]
    fn test_max_bytes_per_second() {
        let cfg = RistConfig::new(RistProfile::Main, 500, 8_000, 5.0);
        assert_eq!(cfg.max_bytes_per_second(), 1_000_000);
    }

    // 6. RistStats::effective_loss_pct – zero packets
    #[test]
    fn test_effective_loss_zero_sent() {
        let stats = RistStats::default();
        assert_eq!(stats.effective_loss_pct(), 0.0);
    }

    // 7. RistStats::effective_loss_pct – all recovered
    #[test]
    fn test_effective_loss_fully_recovered() {
        // Sent 100, received 90 (10 lost), retransmitted 10 → 0 still lost
        let stats = RistStats::new(100, 90, 10, 10.0, 50.0);
        assert_eq!(stats.effective_loss_pct(), 0.0);
    }

    // 8. RistStats::effective_loss_pct – partial recovery
    #[test]
    fn test_effective_loss_partial() {
        // Sent 100, received 80 (20 lost), retransmitted 10 → 10 still lost → 10%
        let stats = RistStats::new(100, 80, 10, 20.0, 50.0);
        assert!((stats.effective_loss_pct() - 10.0).abs() < 1e-3);
    }

    // 9. RistStats::quality_score – perfect
    #[test]
    fn test_quality_score_perfect() {
        let stats = RistStats::new(100, 100, 0, 0.0, 0.0);
        assert!((stats.quality_score() - 1.0).abs() < 1e-3);
    }

    // 10. RistStats::quality_score – high RTT degrades score
    #[test]
    fn test_quality_score_high_rtt() {
        let stats = RistStats::new(100, 100, 0, 0.0, 200.0);
        assert!(stats.quality_score() < 1.0);
        assert!(stats.quality_score() > 0.0);
    }

    // 11. RistBond::add_bond and bond_count
    #[test]
    fn test_bond_add_and_count() {
        let mut bond = RistBond::new();
        assert!(bond.is_empty());
        bond.add_bond("192.168.1.1", 5000);
        bond.add_bond("10.0.0.1", 5000);
        assert_eq!(bond.bond_count(), 2);
    }

    // 12. RistBond::primary
    #[test]
    fn test_bond_primary() {
        let mut bond = RistBond::new();
        assert!(bond.primary().is_none());
        bond.add_bond("192.168.1.1", 5000);
        bond.add_bond("10.0.0.1", 5001);
        let primary = bond.primary().expect("should succeed in test");
        assert_eq!(primary.0, "192.168.1.1");
        assert_eq!(primary.1, 5000);
    }

    // 13. RistBond::all_bonds
    #[test]
    fn test_bond_all_bonds() {
        let mut bond = RistBond::new();
        bond.add_bond("1.2.3.4", 7000);
        bond.add_bond("5.6.7.8", 7001);
        let all = bond.all_bonds();
        assert_eq!(all.len(), 2);
        assert_eq!(all[1].1, 7001);
    }
}
