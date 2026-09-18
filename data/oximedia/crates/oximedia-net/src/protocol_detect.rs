#![allow(dead_code)]
//! Network protocol auto-detection from URIs and byte patterns.
//!
//! Provides [`ProtocolType`] enumeration and helper utilities that inspect a URI
//! scheme or peek at the first few bytes of an incoming connection to determine
//! the streaming protocol in use.

use std::fmt;

/// Supported network streaming protocol families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolType {
    /// HTTP Live Streaming (HLS).
    Hls,
    /// MPEG-DASH.
    Dash,
    /// Real-Time Messaging Protocol.
    Rtmp,
    /// Secure Reliable Transport.
    Srt,
    /// WebRTC (typically `webrtc:` or `stun:`/`turn:` related).
    WebRtc,
    /// SMPTE ST 2110 (professional media over managed IP).
    Smpte2110,
    /// Real-time Transport Protocol (RTP unicast/multicast).
    Rtp,
    /// Unrecognised protocol.
    Unknown,
}

impl fmt::Display for ProtocolType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Hls => "HLS",
            Self::Dash => "DASH",
            Self::Rtmp => "RTMP",
            Self::Srt => "SRT",
            Self::WebRtc => "WebRTC",
            Self::Smpte2110 => "SMPTE2110",
            Self::Rtp => "RTP",
            Self::Unknown => "Unknown",
        };
        f.write_str(label)
    }
}

/// Result of a protocol detection attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionResult {
    /// The detected protocol type.
    pub protocol: ProtocolType,
    /// Confidence level in 0.0..=1.0.
    pub confidence: f64,
    /// Human-readable reason for the detection.
    pub reason: String,
}

impl DetectionResult {
    /// Creates a new detection result.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(protocol: ProtocolType, confidence: f64, reason: impl Into<String>) -> Self {
        Self {
            protocol,
            confidence: confidence.clamp(0.0, 1.0),
            reason: reason.into(),
        }
    }

    /// Returns `true` if confidence exceeds the given threshold.
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Detects the protocol from a URI string by inspecting its scheme and path.
///
/// Returns a [`DetectionResult`] with confidence in `0.0..=1.0`.
pub fn detect_from_uri(uri: &str) -> DetectionResult {
    let lower = uri.to_ascii_lowercase();

    // Check scheme first
    if lower.starts_with("rtmp://") || lower.starts_with("rtmps://") {
        return DetectionResult::new(ProtocolType::Rtmp, 1.0, "URI scheme rtmp(s)");
    }
    if lower.starts_with("srt://") {
        return DetectionResult::new(ProtocolType::Srt, 1.0, "URI scheme srt");
    }
    if lower.starts_with("rtp://") || lower.starts_with("rtsp://") {
        return DetectionResult::new(ProtocolType::Rtp, 0.9, "URI scheme rtp/rtsp");
    }

    // For HTTP(S) URIs inspect the path / extension
    if lower.starts_with("http://") || lower.starts_with("https://") {
        if lower.ends_with(".m3u8") || lower.contains("/hls/") {
            return DetectionResult::new(
                ProtocolType::Hls,
                0.95,
                "HTTP URI with .m3u8 or /hls/ path",
            );
        }
        if lower.ends_with(".mpd") || lower.contains("/dash/") {
            return DetectionResult::new(
                ProtocolType::Dash,
                0.95,
                "HTTP URI with .mpd or /dash/ path",
            );
        }
        // Generic HTTP — could be either HLS or DASH
        return DetectionResult::new(
            ProtocolType::Unknown,
            0.1,
            "HTTP URI with no clear indicator",
        );
    }

    DetectionResult::new(ProtocolType::Unknown, 0.0, "unrecognised URI scheme")
}

/// RTMP handshake magic byte (version 3).
const RTMP_MAGIC: u8 = 0x03;

/// Minimum header bytes needed for byte-level protocol detection.
pub const MIN_PROBE_BYTES: usize = 4;

/// Attempts to identify the protocol by peeking at the first bytes of a stream.
///
/// `data` should contain at least [`MIN_PROBE_BYTES`] bytes from the connection.
pub fn detect_from_bytes(data: &[u8]) -> DetectionResult {
    if data.is_empty() {
        return DetectionResult::new(ProtocolType::Unknown, 0.0, "empty data");
    }

    // RTMP C0 byte
    if data[0] == RTMP_MAGIC && data.len() >= 4 {
        return DetectionResult::new(ProtocolType::Rtmp, 0.85, "C0 magic byte 0x03");
    }

    // STUN binding request starts with 0x00 0x01
    if data.len() >= 4 && data[0] == 0x00 && data[1] == 0x01 {
        return DetectionResult::new(ProtocolType::WebRtc, 0.7, "STUN binding request header");
    }

    // RTP version 2 => (data[0] >> 6) == 2
    if data[0] >> 6 == 2 && data.len() >= 4 {
        // Could be RTP or SRTP
        return DetectionResult::new(ProtocolType::Rtp, 0.6, "RTP version 2 bit pattern");
    }

    DetectionResult::new(ProtocolType::Unknown, 0.0, "no matching byte signature")
}

/// Per-protocol default port numbers (informational).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefaultPort {
    /// Protocol this port applies to.
    pub protocol: ProtocolType,
    /// The well-known port number.
    pub port: u16,
    /// Whether TLS is implied.
    pub tls: bool,
}

/// Returns the well-known default ports for each known protocol.
pub fn default_ports() -> Vec<DefaultPort> {
    vec![
        DefaultPort {
            protocol: ProtocolType::Rtmp,
            port: 1935,
            tls: false,
        },
        DefaultPort {
            protocol: ProtocolType::Srt,
            port: 9000,
            tls: false,
        },
        DefaultPort {
            protocol: ProtocolType::Hls,
            port: 443,
            tls: true,
        },
        DefaultPort {
            protocol: ProtocolType::Dash,
            port: 443,
            tls: true,
        },
        DefaultPort {
            protocol: ProtocolType::Rtp,
            port: 5004,
            tls: false,
        },
        DefaultPort {
            protocol: ProtocolType::WebRtc,
            port: 3478,
            tls: false,
        },
    ]
}

/// Guesses the protocol from a port number alone.
///
/// This is low-confidence since many services share ports.
pub fn detect_from_port(port: u16) -> DetectionResult {
    match port {
        1935 => DetectionResult::new(ProtocolType::Rtmp, 0.8, "port 1935 (RTMP)"),
        9000 => DetectionResult::new(ProtocolType::Srt, 0.5, "port 9000 (SRT common)"),
        5004 => DetectionResult::new(ProtocolType::Rtp, 0.5, "port 5004 (RTP default)"),
        3478 | 5349 => DetectionResult::new(ProtocolType::WebRtc, 0.5, "STUN/TURN port"),
        80 | 8080 => {
            DetectionResult::new(ProtocolType::Unknown, 0.1, "HTTP port, protocol ambiguous")
        }
        443 | 8443 => {
            DetectionResult::new(ProtocolType::Unknown, 0.1, "HTTPS port, protocol ambiguous")
        }
        _ => DetectionResult::new(ProtocolType::Unknown, 0.0, "unknown port"),
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // 1. detect_from_uri — RTMP scheme
    #[test]
    fn test_detect_uri_rtmp() {
        let r = detect_from_uri("rtmp://live.example.com/app/stream");
        assert_eq!(r.protocol, ProtocolType::Rtmp);
        assert!((r.confidence - 1.0).abs() < f64::EPSILON);
    }

    // 2. detect_from_uri — SRT scheme
    #[test]
    fn test_detect_uri_srt() {
        let r = detect_from_uri("srt://host:9000?mode=caller");
        assert_eq!(r.protocol, ProtocolType::Srt);
    }

    // 3. detect_from_uri — HLS via .m3u8 extension
    #[test]
    fn test_detect_uri_hls_m3u8() {
        let r = detect_from_uri("https://cdn.example.com/live/index.m3u8");
        assert_eq!(r.protocol, ProtocolType::Hls);
        assert!(r.confidence > 0.9);
    }

    // 4. detect_from_uri — DASH via .mpd extension
    #[test]
    fn test_detect_uri_dash_mpd() {
        let r = detect_from_uri("https://cdn.example.com/vod/manifest.mpd");
        assert_eq!(r.protocol, ProtocolType::Dash);
    }

    // 5. detect_from_uri — HLS via /hls/ path segment
    #[test]
    fn test_detect_uri_hls_path() {
        let r = detect_from_uri("https://cdn.example.com/hls/live");
        assert_eq!(r.protocol, ProtocolType::Hls);
    }

    // 6. detect_from_uri — DASH via /dash/ path segment
    #[test]
    fn test_detect_uri_dash_path() {
        let r = detect_from_uri("https://cdn.example.com/dash/live");
        assert_eq!(r.protocol, ProtocolType::Dash);
    }

    // 7. detect_from_uri — unknown scheme
    #[test]
    fn test_detect_uri_unknown() {
        let r = detect_from_uri("ftp://files.example.com/video.mp4");
        assert_eq!(r.protocol, ProtocolType::Unknown);
    }

    // 8. detect_from_bytes — RTMP magic
    #[test]
    fn test_detect_bytes_rtmp() {
        let data = [0x03, 0x00, 0x00, 0x00];
        let r = detect_from_bytes(&data);
        assert_eq!(r.protocol, ProtocolType::Rtmp);
    }

    // 9. detect_from_bytes — STUN binding request
    #[test]
    fn test_detect_bytes_stun() {
        let data = [0x00, 0x01, 0x00, 0x58];
        let r = detect_from_bytes(&data);
        assert_eq!(r.protocol, ProtocolType::WebRtc);
    }

    // 10. detect_from_bytes — RTP v2
    #[test]
    fn test_detect_bytes_rtp() {
        // version=2 in top 2 bits => 0x80
        let data = [0x80, 0x60, 0x00, 0x01];
        let r = detect_from_bytes(&data);
        assert_eq!(r.protocol, ProtocolType::Rtp);
    }

    // 11. detect_from_bytes — empty
    #[test]
    fn test_detect_bytes_empty() {
        let r = detect_from_bytes(&[]);
        assert_eq!(r.protocol, ProtocolType::Unknown);
    }

    // 12. detect_from_port — RTMP default
    #[test]
    fn test_detect_port_rtmp() {
        let r = detect_from_port(1935);
        assert_eq!(r.protocol, ProtocolType::Rtmp);
        assert!(r.confidence > 0.5);
    }

    // 13. DetectionResult confidence clamped
    #[test]
    fn test_confidence_clamped() {
        let r = DetectionResult::new(ProtocolType::Hls, 1.5, "over");
        assert!((r.confidence - 1.0).abs() < f64::EPSILON);
        let r2 = DetectionResult::new(ProtocolType::Hls, -0.5, "under");
        assert!((r2.confidence).abs() < f64::EPSILON);
    }

    // 14. ProtocolType display
    #[test]
    fn test_protocol_display() {
        assert_eq!(format!("{}", ProtocolType::Hls), "HLS");
        assert_eq!(format!("{}", ProtocolType::Unknown), "Unknown");
    }

    // 15. default_ports returns expected count
    #[test]
    fn test_default_ports_count() {
        let ports = default_ports();
        assert!(ports.len() >= 5);
    }

    // 16. is_confident helper
    #[test]
    fn test_is_confident() {
        let r = DetectionResult::new(ProtocolType::Srt, 0.75, "test");
        assert!(r.is_confident(0.5));
        assert!(!r.is_confident(0.9));
    }
}
