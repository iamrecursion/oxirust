//! SDP (Session Description Protocol) generation for video-over-IP streams.
//!
//! Provides lightweight SDP building and parsing utilities tailored to
//! professional IP media (SMPTE ST 2110, RTP/AVP).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// SDP media type for a session line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SdpMediaType {
    /// Video stream (ST 2110-20).
    Video,
    /// Audio stream (ST 2110-30/31).
    Audio,
    /// Application / ancillary data (ST 2110-40).
    Application,
}

impl SdpMediaType {
    /// Returns the SDP keyword for this media type.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Application => "application",
        }
    }
}

/// A single SDP attribute (`a=<key>:<value>` or `a=<flag>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdpAttribute {
    /// Attribute key.
    pub key: String,
    /// Attribute value (empty for flag attributes).
    pub value: String,
}

impl SdpAttribute {
    /// Creates a key-value attribute.
    #[must_use]
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }

    /// Creates a flag attribute (no value).
    #[must_use]
    pub fn flag(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: String::new(),
        }
    }

    /// Serialises the attribute to `a=<key>` or `a=<key>:<value>`.
    #[must_use]
    pub fn to_sdp_line(&self) -> String {
        if self.value.is_empty() {
            format!("a={}", self.key)
        } else {
            format!("a={}:{}", self.key, self.value)
        }
    }
}

/// An SDP media section (`m=` + connection + attributes).
#[derive(Debug, Clone)]
pub struct SdpMediaSection {
    /// Media type.
    pub media_type: SdpMediaType,
    /// UDP port the media is sent to.
    pub port: u16,
    /// RTP payload type.
    pub payload_type: u8,
    /// Destination address (unicast IP or multicast group).
    pub connection_addr: String,
    /// List of `a=` attributes for this media section.
    pub attributes: Vec<SdpAttribute>,
}

impl SdpMediaSection {
    /// Creates a new media section.
    #[must_use]
    pub fn new(
        media_type: SdpMediaType,
        port: u16,
        payload_type: u8,
        connection_addr: impl Into<String>,
    ) -> Self {
        Self {
            media_type,
            port,
            payload_type,
            connection_addr: connection_addr.into(),
            attributes: Vec::new(),
        }
    }

    /// Adds an attribute to this media section.
    pub fn add_attribute(&mut self, attr: SdpAttribute) {
        self.attributes.push(attr);
    }

    /// Adds a key-value attribute.
    pub fn add_kv(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.push(SdpAttribute::new(key, value));
    }

    /// Returns `true` if the connection address is an IPv4 multicast address.
    #[must_use]
    pub fn is_multicast(&self) -> bool {
        if let Ok(ip) = self.connection_addr.parse::<std::net::Ipv4Addr>() {
            return ip.is_multicast();
        }
        false
    }

    /// Serialises this media section to SDP lines.
    #[must_use]
    pub fn to_sdp_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!(
            "m={} {} RTP/AVP {}",
            self.media_type.as_str(),
            self.port,
            self.payload_type
        ));
        lines.push(format!("c=IN IP4 {}", self.connection_addr));
        for attr in &self.attributes {
            lines.push(attr.to_sdp_line());
        }
        lines
    }
}

/// A complete SDP session description.
#[derive(Debug, Clone)]
pub struct SdpSession {
    /// Session name (`s=` line).
    pub session_name: String,
    /// Originator info (`o=` line, simplified to a string).
    pub originator: String,
    /// Session-level attributes.
    pub attributes: Vec<SdpAttribute>,
    /// Media sections.
    pub media: Vec<SdpMediaSection>,
}

impl SdpSession {
    /// Creates a new SDP session.
    #[must_use]
    pub fn new(session_name: impl Into<String>, originator: impl Into<String>) -> Self {
        Self {
            session_name: session_name.into(),
            originator: originator.into(),
            attributes: Vec::new(),
            media: Vec::new(),
        }
    }

    /// Adds a session-level attribute.
    pub fn add_attribute(&mut self, attr: SdpAttribute) {
        self.attributes.push(attr);
    }

    /// Adds a media section.
    pub fn add_media(&mut self, section: SdpMediaSection) {
        self.media.push(section);
    }

    /// Returns the number of media sections.
    #[must_use]
    pub fn media_count(&self) -> usize {
        self.media.len()
    }

    /// Returns media sections of the given type.
    #[must_use]
    pub fn media_of_type(&self, t: SdpMediaType) -> Vec<&SdpMediaSection> {
        self.media.iter().filter(|m| m.media_type == t).collect()
    }

    /// Serialises the full SDP session to a string.
    #[must_use]
    pub fn to_sdp_string(&self) -> String {
        let mut lines = Vec::new();
        lines.push("v=0".to_string());
        lines.push(format!("o={}", self.originator));
        lines.push(format!("s={}", self.session_name));
        lines.push("t=0 0".to_string());
        for attr in &self.attributes {
            lines.push(attr.to_sdp_line());
        }
        for section in &self.media {
            lines.extend(section.to_sdp_lines());
        }
        lines.join("\r\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. SdpMediaType::as_str
    #[test]
    fn test_media_type_as_str() {
        assert_eq!(SdpMediaType::Video.as_str(), "video");
        assert_eq!(SdpMediaType::Audio.as_str(), "audio");
        assert_eq!(SdpMediaType::Application.as_str(), "application");
    }

    // 2. SdpAttribute::to_sdp_line – key-value
    #[test]
    fn test_attribute_kv_line() {
        let attr = SdpAttribute::new("rtpmap", "96 raw/90000");
        assert_eq!(attr.to_sdp_line(), "a=rtpmap:96 raw/90000");
    }

    // 3. SdpAttribute::to_sdp_line – flag
    #[test]
    fn test_attribute_flag_line() {
        let attr = SdpAttribute::flag("recvonly");
        assert_eq!(attr.to_sdp_line(), "a=recvonly");
    }

    // 4. SdpMediaSection::is_multicast – true
    #[test]
    fn test_is_multicast_true() {
        let sec = SdpMediaSection::new(SdpMediaType::Video, 5004, 96, "239.100.0.1");
        assert!(sec.is_multicast());
    }

    // 5. SdpMediaSection::is_multicast – false (unicast)
    #[test]
    fn test_is_multicast_false() {
        let sec = SdpMediaSection::new(SdpMediaType::Video, 5004, 96, "192.168.1.10");
        assert!(!sec.is_multicast());
    }

    // 6. SdpMediaSection::to_sdp_lines – m= line present
    #[test]
    fn test_media_section_m_line() {
        let sec = SdpMediaSection::new(SdpMediaType::Audio, 5006, 97, "239.100.0.2");
        let lines = sec.to_sdp_lines();
        assert!(lines[0].starts_with("m=audio 5006 RTP/AVP 97"));
    }

    // 7. SdpMediaSection::to_sdp_lines – c= line present
    #[test]
    fn test_media_section_c_line() {
        let sec = SdpMediaSection::new(SdpMediaType::Video, 5004, 96, "239.100.0.1");
        let lines = sec.to_sdp_lines();
        assert!(lines[1].contains("239.100.0.1"));
    }

    // 8. SdpMediaSection::add_attribute
    #[test]
    fn test_add_attribute_to_section() {
        let mut sec = SdpMediaSection::new(SdpMediaType::Video, 5004, 96, "239.100.0.1");
        sec.add_kv("rtpmap", "96 raw/90000");
        let lines = sec.to_sdp_lines();
        assert!(lines.iter().any(|l| l.contains("rtpmap")));
    }

    // 9. SdpSession::media_count
    #[test]
    fn test_session_media_count() {
        let mut sess = SdpSession::new("Test", "- 0 0 IN IP4 127.0.0.1");
        assert_eq!(sess.media_count(), 0);
        sess.add_media(SdpMediaSection::new(
            SdpMediaType::Video,
            5004,
            96,
            "239.100.0.1",
        ));
        assert_eq!(sess.media_count(), 1);
    }

    // 10. SdpSession::media_of_type
    #[test]
    fn test_session_media_of_type() {
        let mut sess = SdpSession::new("S", "- 0 0 IN IP4 127.0.0.1");
        sess.add_media(SdpMediaSection::new(
            SdpMediaType::Video,
            5004,
            96,
            "239.0.0.1",
        ));
        sess.add_media(SdpMediaSection::new(
            SdpMediaType::Audio,
            5006,
            97,
            "239.0.0.2",
        ));
        sess.add_media(SdpMediaSection::new(
            SdpMediaType::Video,
            5008,
            98,
            "239.0.0.3",
        ));
        assert_eq!(sess.media_of_type(SdpMediaType::Video).len(), 2);
        assert_eq!(sess.media_of_type(SdpMediaType::Audio).len(), 1);
        assert_eq!(sess.media_of_type(SdpMediaType::Application).len(), 0);
    }

    // 11. SdpSession::to_sdp_string – v=0 present
    #[test]
    fn test_sdp_string_starts_with_version() {
        let sess = SdpSession::new("Test Session", "- 0 0 IN IP4 127.0.0.1");
        let sdp = sess.to_sdp_string();
        assert!(sdp.starts_with("v=0"));
    }

    // 12. SdpSession::to_sdp_string – session name present
    #[test]
    fn test_sdp_string_has_session_name() {
        let sess = SdpSession::new("MyStream", "- 0 0 IN IP4 10.0.0.1");
        let sdp = sess.to_sdp_string();
        assert!(sdp.contains("s=MyStream"));
    }

    // 13. SdpSession::to_sdp_string – session-level attribute serialised
    #[test]
    fn test_sdp_string_session_attribute() {
        let mut sess = SdpSession::new("S", "- 0 0 IN IP4 127.0.0.1");
        sess.add_attribute(SdpAttribute::flag("sendonly"));
        let sdp = sess.to_sdp_string();
        assert!(sdp.contains("a=sendonly"));
    }
}
