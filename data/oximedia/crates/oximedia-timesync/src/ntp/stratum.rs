//! NTP stratum hierarchy.

use std::fmt;

/// NTP stratum level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Stratum(u8);

impl Stratum {
    /// Primary reference (e.g., GPS, atomic clock)
    pub const PRIMARY_REFERENCE: Self = Self(1);

    /// Secondary reference (synced via NTP) - minimum
    pub const SECONDARY_MIN: Self = Self(2);
    /// Secondary reference (synced via NTP) - maximum
    pub const SECONDARY_MAX: Self = Self(15);

    /// Unsynchronized
    pub const UNSYNCHRONIZED: Self = Self(16);

    /// Reserved
    pub const RESERVED: Self = Self(0);

    /// Create from u8.
    #[must_use]
    pub fn from_u8(value: u8) -> Self {
        Self(value)
    }

    /// Convert to u8.
    #[must_use]
    pub fn to_u8(self) -> u8 {
        self.0
    }

    /// Check if this is a primary reference.
    #[must_use]
    pub fn is_primary(&self) -> bool {
        self.0 == 1
    }

    /// Check if this is a secondary reference.
    #[must_use]
    pub fn is_secondary(&self) -> bool {
        self.0 >= 2 && self.0 <= 15
    }

    /// Check if synchronized.
    #[must_use]
    pub fn is_synchronized(&self) -> bool {
        self.0 >= 1 && self.0 <= 15
    }

    /// Check if unsynchronized.
    #[must_use]
    pub fn is_unsynchronized(&self) -> bool {
        self.0 == 0 || self.0 == 16
    }

    /// Get description of this stratum.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self.0 {
            0 => "Reserved/Unspecified",
            1 => "Primary Reference (GPS, Atomic Clock, etc.)",
            2..=15 => "Secondary Reference (NTP)",
            16 => "Unsynchronized",
            _ => "Reserved",
        }
    }
}

impl fmt::Display for Stratum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stratum {} ({})", self.0, self.description())
    }
}

impl From<u8> for Stratum {
    fn from(value: u8) -> Self {
        Self::from_u8(value)
    }
}

impl From<Stratum> for u8 {
    fn from(stratum: Stratum) -> Self {
        stratum.to_u8()
    }
}

// ---------------------------------------------------------------------------
// NtpServerInfo and NtpServerSelector
// ---------------------------------------------------------------------------

/// Information about a single NTP server candidate used for server selection.
#[derive(Debug, Clone)]
pub struct NtpServerInfo {
    /// Server address as a string (e.g. `"pool.ntp.org:123"`).
    pub address: String,
    /// NTP stratum reported by this server.
    pub stratum: Stratum,
    /// Round-trip time (RTT) to the server in seconds.
    ///
    /// This is used as a secondary sort key when stratum values are equal.
    pub rtt_secs: f64,
}

impl NtpServerInfo {
    /// Creates a new server descriptor.
    #[must_use]
    pub fn new(address: impl Into<String>, stratum: Stratum, rtt_secs: f64) -> Self {
        Self {
            address: address.into(),
            stratum,
            rtt_secs,
        }
    }
}

/// Stateless NTP server selector.
///
/// Implements the server-selection policy used by NTP clients:
/// 1. Prefer the server with the **lowest stratum** (closest to primary ref).
/// 2. Among servers with equal stratum, prefer the one with the **lowest RTT**.
/// 3. Servers that are unsynchronised (stratum 0 or 16) are excluded.
pub struct NtpServerSelector;

impl NtpServerSelector {
    /// Returns a reference to the best server from `servers`, or `None` if
    /// `servers` is empty or all servers are unsynchronised.
    ///
    /// Selection criteria (in order of precedence):
    /// 1. Lowest stratum (excludes stratum 0 and 16).
    /// 2. Lowest RTT as tiebreaker.
    #[must_use]
    pub fn best_server<'a>(servers: &'a [NtpServerInfo]) -> Option<&'a NtpServerInfo> {
        servers
            .iter()
            .filter(|s| s.stratum.is_synchronized())
            .min_by(|a, b| {
                // Primary sort: stratum (lower is better).
                a.stratum
                    .cmp(&b.stratum)
                    // Secondary sort: RTT (lower is better).
                    .then_with(|| {
                        a.rtt_secs
                            .partial_cmp(&b.rtt_secs)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratum_creation() {
        let s1 = Stratum::from_u8(1);
        assert!(s1.is_primary());
        assert!(s1.is_synchronized());

        let s2 = Stratum::from_u8(2);
        assert!(s2.is_secondary());
        assert!(s2.is_synchronized());

        let s16 = Stratum::from_u8(16);
        assert!(s16.is_unsynchronized());
        assert!(!s16.is_synchronized());
    }

    #[test]
    fn test_stratum_ordering() {
        let s1 = Stratum::from_u8(1);
        let s2 = Stratum::from_u8(2);
        let s3 = Stratum::from_u8(3);

        assert!(s1 < s2);
        assert!(s2 < s3);
        assert!(s1 < s3);
    }

    #[test]
    fn test_stratum_display() {
        let s1 = Stratum::PRIMARY_REFERENCE;
        let display = format!("{}", s1);
        assert!(display.contains("Primary Reference"));
    }

    // -----------------------------------------------------------------------
    // NtpServerSelector tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_server_selector_empty() {
        assert!(NtpServerSelector::best_server(&[]).is_none());
    }

    #[test]
    fn test_server_selector_all_unsync() {
        let servers = vec![
            NtpServerInfo::new("a:123", Stratum::UNSYNCHRONIZED, 0.01),
            NtpServerInfo::new("b:123", Stratum::from_u8(0), 0.02),
        ];
        assert!(
            NtpServerSelector::best_server(&servers).is_none(),
            "all unsynchronised servers → no best server"
        );
    }

    #[test]
    fn test_server_selector_lowest_stratum_wins() {
        let servers = vec![
            NtpServerInfo::new("s2a:123", Stratum::from_u8(2), 0.010),
            NtpServerInfo::new("s1:123", Stratum::from_u8(1), 0.020), // lowest stratum
            NtpServerInfo::new("s3:123", Stratum::from_u8(3), 0.005),
        ];
        let best = NtpServerSelector::best_server(&servers).expect("should find best server");
        assert_eq!(
            best.stratum.to_u8(),
            1,
            "server with stratum 1 should be selected"
        );
        assert_eq!(best.address, "s1:123");
    }

    #[test]
    fn test_server_selector_rtt_tiebreaker() {
        // Two stratum-2 servers; lower RTT should win.
        let servers = vec![
            NtpServerInfo::new("s2-slow:123", Stratum::from_u8(2), 0.050),
            NtpServerInfo::new("s2-fast:123", Stratum::from_u8(2), 0.005),
        ];
        let best = NtpServerSelector::best_server(&servers).expect("should find best server");
        assert_eq!(
            best.address, "s2-fast:123",
            "server with lower RTT should win tie"
        );
    }

    #[test]
    fn test_server_selector_skips_unsync_prefers_higher_stratum_sync() {
        // Stratum-16 is unsync; a stratum-5 server should still win.
        let servers = vec![
            NtpServerInfo::new("bad:123", Stratum::UNSYNCHRONIZED, 0.001),
            NtpServerInfo::new("ok:123", Stratum::from_u8(5), 0.050),
        ];
        let best = NtpServerSelector::best_server(&servers).expect("should find ok server");
        assert_eq!(best.address, "ok:123");
    }

    #[test]
    fn test_server_selector_multiple_stratums() {
        let servers = vec![
            NtpServerInfo::new("pool-a:123", Stratum::from_u8(3), 0.015),
            NtpServerInfo::new("pool-b:123", Stratum::from_u8(2), 0.025),
            NtpServerInfo::new("pool-c:123", Stratum::from_u8(4), 0.005),
            NtpServerInfo::new("gps:123", Stratum::from_u8(1), 0.100), // best stratum
        ];
        let best = NtpServerSelector::best_server(&servers).expect("should find best");
        assert_eq!(best.stratum.to_u8(), 1, "GPS stratum-1 should be selected");
    }
}
