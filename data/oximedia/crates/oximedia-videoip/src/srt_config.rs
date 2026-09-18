#![allow(dead_code)]
//! SRT (Secure Reliable Transport) configuration for video-over-IP.
//!
//! Provides configuration types and validation for SRT connections,
//! including caller/listener/rendezvous modes, encryption, latency
//! settings, and bandwidth overhead. SRT is widely used in broadcast
//! contribution and distribution links over the public internet.

use std::fmt;
use std::net::SocketAddr;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// SRT connection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SrtMode {
    /// Caller initiates the connection to a listener.
    Caller,
    /// Listener waits for incoming caller connections.
    Listener,
    /// Both sides attempt simultaneous connection (firewall traversal).
    Rendezvous,
}

impl fmt::Display for SrtMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Caller => write!(f, "caller"),
            Self::Listener => write!(f, "listener"),
            Self::Rendezvous => write!(f, "rendezvous"),
        }
    }
}

/// SRT encryption key length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SrtKeyLength {
    /// No encryption.
    None,
    /// AES-128 encryption.
    Aes128,
    /// AES-192 encryption.
    Aes192,
    /// AES-256 encryption.
    Aes256,
}

impl SrtKeyLength {
    /// Return the key length in bits.
    #[must_use]
    pub fn bits(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Aes128 => 128,
            Self::Aes192 => 192,
            Self::Aes256 => 256,
        }
    }
}

/// Congestion control algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CongestionControl {
    /// Live mode (low-latency, UDP-based pacing).
    Live,
    /// File mode (high-throughput, TCP-like congestion avoidance).
    File,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// SRT connection configuration.
#[derive(Debug, Clone)]
pub struct SrtConfig {
    /// Connection mode.
    pub mode: SrtMode,
    /// Local bind address (for listener or rendezvous).
    pub local_addr: SocketAddr,
    /// Remote address (for caller or rendezvous).
    pub remote_addr: Option<SocketAddr>,
    /// Latency (the receive buffer duration). Typical: 120..4000 ms.
    pub latency: Duration,
    /// Peer latency (override for the remote side).
    pub peer_latency: Option<Duration>,
    /// Maximum bandwidth in bits/sec (0 = unlimited).
    pub max_bandwidth_bps: u64,
    /// Overhead bandwidth percentage (5..100).
    pub overhead_percent: u8,
    /// Encryption key length.
    pub encryption: SrtKeyLength,
    /// Passphrase for encryption (10..79 characters).
    pub passphrase: Option<String>,
    /// Stream ID (application-level routing).
    pub stream_id: Option<String>,
    /// Congestion control mode.
    pub congestion_control: CongestionControl,
    /// Maximum segment size (payload bytes per UDP packet).
    pub mss: u16,
    /// Flight flag size (send buffer in packets).
    pub flight_flag_size: u32,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Enable periodic NAK reports.
    pub nak_report: bool,
    /// Time-to-live for packets (hops).
    pub ttl: u8,
}

impl Default for SrtConfig {
    fn default() -> Self {
        Self {
            mode: SrtMode::Caller,
            local_addr: "0.0.0.0:0".parse().expect("valid default addr"),
            remote_addr: None,
            latency: Duration::from_millis(120),
            peer_latency: None,
            max_bandwidth_bps: 0,
            overhead_percent: 25,
            encryption: SrtKeyLength::None,
            passphrase: None,
            stream_id: None,
            congestion_control: CongestionControl::Live,
            mss: 1500,
            flight_flag_size: 25600,
            connect_timeout: Duration::from_secs(3),
            nak_report: true,
            ttl: 64,
        }
    }
}

/// Validation error for SRT configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtConfigError {
    /// Description of the validation failure.
    pub message: String,
}

impl fmt::Display for SrtConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SRT config error: {}", self.message)
    }
}

impl std::error::Error for SrtConfigError {}

impl SrtConfig {
    /// Create a caller configuration connecting to the given remote.
    #[must_use]
    pub fn caller(remote: SocketAddr) -> Self {
        Self {
            mode: SrtMode::Caller,
            remote_addr: Some(remote),
            ..Default::default()
        }
    }

    /// Create a listener configuration bound to the given address.
    #[must_use]
    pub fn listener(bind: SocketAddr) -> Self {
        Self {
            mode: SrtMode::Listener,
            local_addr: bind,
            ..Default::default()
        }
    }

    /// Create a rendezvous configuration.
    #[must_use]
    pub fn rendezvous(local: SocketAddr, remote: SocketAddr) -> Self {
        Self {
            mode: SrtMode::Rendezvous,
            local_addr: local,
            remote_addr: Some(remote),
            ..Default::default()
        }
    }

    /// Set latency.
    #[must_use]
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = latency;
        self
    }

    /// Set encryption.
    #[must_use]
    pub fn with_encryption(mut self, key_len: SrtKeyLength, passphrase: &str) -> Self {
        self.encryption = key_len;
        self.passphrase = Some(passphrase.to_owned());
        self
    }

    /// Set stream ID.
    #[must_use]
    pub fn with_stream_id(mut self, id: &str) -> Self {
        self.stream_id = Some(id.to_owned());
        self
    }

    /// Set maximum bandwidth.
    #[must_use]
    pub fn with_max_bandwidth(mut self, bps: u64) -> Self {
        self.max_bandwidth_bps = bps;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), SrtConfigError> {
        // Caller and rendezvous require a remote address
        if self.mode != SrtMode::Listener && self.remote_addr.is_none() {
            return Err(SrtConfigError {
                message: format!("{} mode requires a remote address", self.mode),
            });
        }

        // Passphrase length
        if let Some(ref pp) = self.passphrase {
            if pp.len() < 10 || pp.len() > 79 {
                return Err(SrtConfigError {
                    message: "Passphrase must be 10..79 characters".to_owned(),
                });
            }
        }

        // Encryption requires passphrase
        if self.encryption != SrtKeyLength::None && self.passphrase.is_none() {
            return Err(SrtConfigError {
                message: "Encryption requires a passphrase".to_owned(),
            });
        }

        // Overhead
        if self.overhead_percent < 5 || self.overhead_percent > 100 {
            return Err(SrtConfigError {
                message: "Overhead percentage must be 5..100".to_owned(),
            });
        }

        // MSS
        if self.mss < 76 {
            return Err(SrtConfigError {
                message: "MSS must be >= 76".to_owned(),
            });
        }

        // Latency sanity
        if self.latency.as_millis() > 30_000 {
            return Err(SrtConfigError {
                message: "Latency must be <= 30000 ms".to_owned(),
            });
        }

        Ok(())
    }

    /// Build an SRT URI string (srt://host:port?key=val&...).
    #[must_use]
    pub fn to_uri(&self) -> String {
        let addr = self.remote_addr.unwrap_or(self.local_addr);
        let mut params = Vec::new();
        params.push(format!("mode={}", self.mode));
        params.push(format!("latency={}", self.latency.as_millis()));
        if self.encryption != SrtKeyLength::None {
            params.push(format!("pbkeylen={}", self.encryption.bits()));
        }
        if let Some(ref pp) = self.passphrase {
            params.push(format!("passphrase={pp}"));
        }
        if let Some(ref sid) = self.stream_id {
            params.push(format!("streamid={sid}"));
        }
        format!("srt://{}?{}", addr, params.join("&"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn remote() -> SocketAddr {
        "192.168.1.100:9000"
            .parse()
            .expect("should succeed in test")
    }

    fn local() -> SocketAddr {
        "0.0.0.0:9000".parse().expect("should succeed in test")
    }

    #[test]
    fn test_default_config() {
        let cfg = SrtConfig::default();
        assert_eq!(cfg.mode, SrtMode::Caller);
        assert_eq!(cfg.latency, Duration::from_millis(120));
        assert_eq!(cfg.encryption, SrtKeyLength::None);
        assert!(cfg.passphrase.is_none());
    }

    #[test]
    fn test_caller_factory() {
        let cfg = SrtConfig::caller(remote());
        assert_eq!(cfg.mode, SrtMode::Caller);
        assert_eq!(cfg.remote_addr, Some(remote()));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_listener_factory() {
        let cfg = SrtConfig::listener(local());
        assert_eq!(cfg.mode, SrtMode::Listener);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_rendezvous_factory() {
        let cfg = SrtConfig::rendezvous(local(), remote());
        assert_eq!(cfg.mode, SrtMode::Rendezvous);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_caller_without_remote_fails() {
        let cfg = SrtConfig::default(); // caller but no remote
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_encryption_without_passphrase_fails() {
        let cfg = SrtConfig {
            encryption: SrtKeyLength::Aes256,
            passphrase: None,
            remote_addr: Some(remote()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_passphrase_too_short() {
        let cfg = SrtConfig {
            encryption: SrtKeyLength::Aes128,
            passphrase: Some("short".to_owned()),
            remote_addr: Some(remote()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_valid_encryption() {
        let cfg = SrtConfig::caller(remote())
            .with_encryption(SrtKeyLength::Aes256, "mySecretPassphrase123");
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_overhead_out_of_range() {
        let cfg = SrtConfig {
            overhead_percent: 3,
            remote_addr: Some(remote()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_mss_too_small() {
        let cfg = SrtConfig {
            mss: 50,
            remote_addr: Some(remote()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_latency_too_large() {
        let cfg = SrtConfig {
            latency: Duration::from_secs(60),
            remote_addr: Some(remote()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_key_length_bits() {
        assert_eq!(SrtKeyLength::None.bits(), 0);
        assert_eq!(SrtKeyLength::Aes128.bits(), 128);
        assert_eq!(SrtKeyLength::Aes192.bits(), 192);
        assert_eq!(SrtKeyLength::Aes256.bits(), 256);
    }

    #[test]
    fn test_to_uri() {
        let cfg = SrtConfig::caller(remote())
            .with_latency(Duration::from_millis(200))
            .with_stream_id("camera1");
        let uri = cfg.to_uri();
        assert!(uri.starts_with("srt://"));
        assert!(uri.contains("latency=200"));
        assert!(uri.contains("streamid=camera1"));
    }

    #[test]
    fn test_mode_display() {
        assert_eq!(SrtMode::Caller.to_string(), "caller");
        assert_eq!(SrtMode::Listener.to_string(), "listener");
        assert_eq!(SrtMode::Rendezvous.to_string(), "rendezvous");
    }

    #[test]
    fn test_builder_chain() {
        let cfg = SrtConfig::caller(remote())
            .with_latency(Duration::from_millis(500))
            .with_max_bandwidth(10_000_000)
            .with_stream_id("test");
        assert_eq!(cfg.latency, Duration::from_millis(500));
        assert_eq!(cfg.max_bandwidth_bps, 10_000_000);
        assert_eq!(cfg.stream_id.as_deref(), Some("test"));
    }
}
