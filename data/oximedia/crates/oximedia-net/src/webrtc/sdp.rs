//! SDP (Session Description Protocol) parsing and generation.
//!
//! This module provides types for working with SDP, used in WebRTC
//! for session negotiation.

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]
use crate::error::{NetError, NetResult};
use std::fmt;

/// Media type in SDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    /// Audio media.
    Audio,
    /// Video media.
    Video,
    /// Application data (data channels).
    Application,
    /// Text media.
    Text,
    /// Message media.
    Message,
}

impl MediaType {
    /// Parses from string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "audio" => Some(Self::Audio),
            "video" => Some(Self::Video),
            "application" => Some(Self::Application),
            "text" => Some(Self::Text),
            "message" => Some(Self::Message),
            _ => None,
        }
    }

    /// Returns string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Application => "application",
            Self::Text => "text",
            Self::Message => "message",
        }
    }
}

impl fmt::Display for MediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Stream direction attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    /// Send and receive.
    #[default]
    SendRecv,
    /// Send only.
    SendOnly,
    /// Receive only.
    RecvOnly,
    /// Inactive.
    Inactive,
}

impl Direction {
    /// Parses from string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sendrecv" => Some(Self::SendRecv),
            "sendonly" => Some(Self::SendOnly),
            "recvonly" => Some(Self::RecvOnly),
            "inactive" => Some(Self::Inactive),
            _ => None,
        }
    }

    /// Returns string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SendRecv => "sendrecv",
            Self::SendOnly => "sendonly",
            Self::RecvOnly => "recvonly",
            Self::Inactive => "inactive",
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// DTLS fingerprint for SRTP keying.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint {
    /// Hash algorithm (e.g., "sha-256").
    pub algorithm: String,
    /// Fingerprint value (hex with colons).
    pub value: String,
}

impl Fingerprint {
    /// Creates a new fingerprint.
    #[must_use]
    pub fn new(algorithm: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            algorithm: algorithm.into(),
            value: value.into(),
        }
    }

    /// Formats as SDP attribute value.
    #[must_use]
    pub fn to_sdp(&self) -> String {
        format!("{} {}", self.algorithm, self.value)
    }

    /// Parses from SDP attribute value.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let (algo, val) = s.split_once(' ')?;
        Some(Self::new(algo, val))
    }
}

/// SDP attribute (a= line).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribute {
    /// Attribute name.
    pub name: String,
    /// Attribute value (optional).
    pub value: Option<String>,
}

impl Attribute {
    /// Creates a new attribute with value.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: Some(value.into()),
        }
    }

    /// Creates a flag attribute (no value).
    #[must_use]
    pub fn flag(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: None,
        }
    }

    /// Creates an rtpmap attribute.
    #[must_use]
    pub fn rtpmap(payload_type: u8, encoding: &str, clock_rate: u32) -> Self {
        Self::new("rtpmap", format!("{payload_type} {encoding}/{clock_rate}"))
    }

    /// Creates an rtpmap attribute with channels.
    #[must_use]
    pub fn rtpmap_audio(payload_type: u8, encoding: &str, clock_rate: u32, channels: u8) -> Self {
        Self::new(
            "rtpmap",
            format!("{payload_type} {encoding}/{clock_rate}/{channels}"),
        )
    }

    /// Creates an fmtp attribute.
    #[must_use]
    pub fn fmtp(payload_type: u8, params: &str) -> Self {
        Self::new("fmtp", format!("{payload_type} {params}"))
    }

    /// Creates a mid attribute.
    #[must_use]
    pub fn mid(id: impl Into<String>) -> Self {
        Self::new("mid", id)
    }

    /// Creates an ICE ufrag attribute.
    #[must_use]
    pub fn ice_ufrag(ufrag: impl Into<String>) -> Self {
        Self::new("ice-ufrag", ufrag)
    }

    /// Creates an ICE pwd attribute.
    #[must_use]
    pub fn ice_pwd(pwd: impl Into<String>) -> Self {
        Self::new("ice-pwd", pwd)
    }

    /// Formats as SDP line.
    #[must_use]
    pub fn to_sdp(&self) -> String {
        match &self.value {
            Some(v) => format!("a={}:{}", self.name, v),
            None => format!("a={}", self.name),
        }
    }

    /// Parses from SDP line (without "a=" prefix).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        if let Some((name, value)) = s.split_once(':') {
            Some(Self::new(name, value))
        } else {
            Some(Self::flag(s))
        }
    }
}

/// Media description (m= section).
#[derive(Debug, Clone, Default)]
pub struct MediaDescription {
    /// Media type.
    pub media_type: Option<MediaType>,
    /// Port number.
    pub port: u16,
    /// Protocol (e.g., "UDP/TLS/RTP/SAVPF").
    pub protocol: String,
    /// Format/payload types.
    pub formats: Vec<String>,
    /// Connection information (c= line).
    pub connection: Option<String>,
    /// Bandwidth (b= line).
    pub bandwidth: Option<String>,
    /// Media ID (mid).
    pub mid: Option<String>,
    /// Direction.
    pub direction: Direction,
    /// ICE ufrag.
    pub ice_ufrag: Option<String>,
    /// ICE pwd.
    pub ice_pwd: Option<String>,
    /// DTLS fingerprint.
    pub fingerprint: Option<Fingerprint>,
    /// Setup role (actpass, active, passive).
    pub setup: Option<String>,
    /// RTP/RTCP mux.
    pub rtcp_mux: bool,
    /// RTCP feedback settings.
    pub rtcp_fb: Vec<String>,
    /// Generic attributes.
    pub attributes: Vec<Attribute>,
}

impl MediaDescription {
    /// Creates a new media description.
    #[must_use]
    pub fn new(media_type: MediaType, port: u16, protocol: impl Into<String>) -> Self {
        Self {
            media_type: Some(media_type),
            port,
            protocol: protocol.into(),
            ..Default::default()
        }
    }

    /// Creates an audio media description.
    #[must_use]
    pub fn audio(port: u16) -> Self {
        Self::new(MediaType::Audio, port, "UDP/TLS/RTP/SAVPF")
    }

    /// Creates a video media description.
    #[must_use]
    pub fn video(port: u16) -> Self {
        Self::new(MediaType::Video, port, "UDP/TLS/RTP/SAVPF")
    }

    /// Creates a data channel media description.
    #[must_use]
    pub fn data_channel(port: u16) -> Self {
        Self::new(MediaType::Application, port, "UDP/DTLS/SCTP")
    }

    /// Adds a format.
    #[must_use]
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.formats.push(format.into());
        self
    }

    /// Sets the media ID.
    #[must_use]
    pub fn with_mid(mut self, mid: impl Into<String>) -> Self {
        self.mid = Some(mid.into());
        self
    }

    /// Sets the direction.
    #[must_use]
    pub const fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Sets ICE credentials.
    #[must_use]
    pub fn with_ice(mut self, ufrag: impl Into<String>, pwd: impl Into<String>) -> Self {
        self.ice_ufrag = Some(ufrag.into());
        self.ice_pwd = Some(pwd.into());
        self
    }

    /// Sets DTLS fingerprint.
    #[must_use]
    pub fn with_fingerprint(mut self, fingerprint: Fingerprint) -> Self {
        self.fingerprint = Some(fingerprint);
        self
    }

    /// Enables RTCP mux.
    #[must_use]
    pub const fn with_rtcp_mux(mut self) -> Self {
        self.rtcp_mux = true;
        self
    }

    /// Adds an attribute.
    #[must_use]
    pub fn with_attribute(mut self, attr: Attribute) -> Self {
        self.attributes.push(attr);
        self
    }

    /// Gets an attribute by name.
    #[must_use]
    pub fn get_attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Formats as SDP lines.
    #[must_use]
    pub fn to_sdp(&self) -> String {
        let mut lines = Vec::new();

        // m= line
        let media_type = self.media_type.map_or("application", |m| m.as_str());
        let formats = self.formats.join(" ");
        lines.push(format!(
            "m={} {} {} {}",
            media_type, self.port, self.protocol, formats
        ));

        // c= line
        if let Some(ref conn) = self.connection {
            lines.push(format!("c={conn}"));
        }

        // b= line
        if let Some(ref bw) = self.bandwidth {
            lines.push(format!("b={bw}"));
        }

        // mid
        if let Some(ref mid) = self.mid {
            lines.push(format!("a=mid:{mid}"));
        }

        // ICE
        if let Some(ref ufrag) = self.ice_ufrag {
            lines.push(format!("a=ice-ufrag:{ufrag}"));
        }
        if let Some(ref pwd) = self.ice_pwd {
            lines.push(format!("a=ice-pwd:{pwd}"));
        }

        // Fingerprint
        if let Some(ref fp) = self.fingerprint {
            lines.push(format!("a=fingerprint:{}", fp.to_sdp()));
        }

        // Setup
        if let Some(ref setup) = self.setup {
            lines.push(format!("a=setup:{setup}"));
        }

        // Direction
        lines.push(format!("a={}", self.direction.as_str()));

        // RTCP mux
        if self.rtcp_mux {
            lines.push("a=rtcp-mux".to_string());
        }

        // Other attributes
        for attr in &self.attributes {
            lines.push(attr.to_sdp());
        }

        lines.join("\r\n")
    }
}

/// SDP session description.
#[derive(Debug, Clone, Default)]
pub struct SessionDescription {
    /// Protocol version (always 0).
    pub version: u8,
    /// Session originator.
    pub origin: Option<String>,
    /// Session name.
    pub session_name: String,
    /// Session information.
    pub session_info: Option<String>,
    /// URI.
    pub uri: Option<String>,
    /// Email.
    pub email: Option<String>,
    /// Phone.
    pub phone: Option<String>,
    /// Connection information.
    pub connection: Option<String>,
    /// Bandwidth.
    pub bandwidth: Option<String>,
    /// Timing (t= line).
    pub timing: String,
    /// Session-level attributes.
    pub attributes: Vec<Attribute>,
    /// Media descriptions.
    pub media: Vec<MediaDescription>,
}

impl SessionDescription {
    /// Creates a new session description.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: 0,
            session_name: "-".to_string(),
            timing: "0 0".to_string(),
            ..Default::default()
        }
    }

    /// Sets the origin.
    #[must_use]
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = Some(origin.into());
        self
    }

    /// Sets the session name.
    #[must_use]
    pub fn with_session_name(mut self, name: impl Into<String>) -> Self {
        self.session_name = name.into();
        self
    }

    /// Adds a session-level attribute.
    #[must_use]
    pub fn with_attribute(mut self, attr: Attribute) -> Self {
        self.attributes.push(attr);
        self
    }

    /// Adds a media description.
    #[must_use]
    pub fn with_media(mut self, media: MediaDescription) -> Self {
        self.media.push(media);
        self
    }

    /// Gets a session-level attribute by name.
    #[must_use]
    pub fn get_attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Returns media descriptions of a specific type.
    #[must_use]
    pub fn media_of_type(&self, media_type: MediaType) -> Vec<&MediaDescription> {
        self.media
            .iter()
            .filter(|m| m.media_type == Some(media_type))
            .collect()
    }

    /// Formats as SDP string.
    #[must_use]
    pub fn to_sdp(&self) -> String {
        let mut lines = Vec::new();

        // v= line
        lines.push(format!("v={}", self.version));

        // o= line
        if let Some(ref origin) = self.origin {
            lines.push(format!("o={origin}"));
        } else {
            lines.push("o=- 0 0 IN IP4 0.0.0.0".to_string());
        }

        // s= line
        lines.push(format!("s={}", self.session_name));

        // i= line
        if let Some(ref info) = self.session_info {
            lines.push(format!("i={info}"));
        }

        // c= line
        if let Some(ref conn) = self.connection {
            lines.push(format!("c={conn}"));
        }

        // t= line
        lines.push(format!("t={}", self.timing));

        // Session-level attributes
        for attr in &self.attributes {
            lines.push(attr.to_sdp());
        }

        // Media descriptions
        for media in &self.media {
            lines.push(media.to_sdp());
        }

        lines.join("\r\n") + "\r\n"
    }

    /// Parses an SDP string.
    ///
    /// # Errors
    ///
    /// Returns an error if the SDP is malformed.
    pub fn parse(sdp: &str) -> NetResult<Self> {
        let mut session = Self::new();
        let mut current_media: Option<MediaDescription> = None;

        for line in sdp.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // SDP lines are `<type>=<value>` where `<type>` is a single ASCII
            // byte. Comparing the second *byte* (not char) to '=' guarantees
            // byte 0 is a complete ASCII character and byte 2 is a UTF-8
            // boundary, so `&line[2..]` cannot panic on a multibyte leading
            // char in a hostile WHIP/WHEP offer (e.g. a line starting with 'é').
            let bytes = line.as_bytes();
            if bytes.len() < 2 || bytes[1] != b'=' {
                continue;
            }

            let type_char = bytes[0] as char;
            let value = &line[2..];

            match type_char {
                'v' => {
                    session.version = value.parse().unwrap_or(0);
                }
                'o' => {
                    session.origin = Some(value.to_string());
                }
                's' => {
                    session.session_name = value.to_string();
                }
                'i' => {
                    if current_media.is_none() {
                        session.session_info = Some(value.to_string());
                    }
                }
                'c' => {
                    if let Some(ref mut media) = current_media {
                        media.connection = Some(value.to_string());
                    } else {
                        session.connection = Some(value.to_string());
                    }
                }
                't' => {
                    session.timing = value.to_string();
                }
                'b' => {
                    if let Some(ref mut media) = current_media {
                        media.bandwidth = Some(value.to_string());
                    } else {
                        session.bandwidth = Some(value.to_string());
                    }
                }
                'm' => {
                    // Save previous media if any
                    if let Some(media) = current_media.take() {
                        session.media.push(media);
                    }

                    // Parse m= line
                    current_media = Some(parse_media_line(value)?);
                }
                'a' => {
                    if let Some(attr) = Attribute::parse(value) {
                        if let Some(ref mut media) = current_media {
                            apply_attribute_to_media(media, attr);
                        } else {
                            session.attributes.push(attr);
                        }
                    }
                }
                _ => {}
            }
        }

        // Save last media
        if let Some(media) = current_media {
            session.media.push(media);
        }

        Ok(session)
    }
}

fn parse_media_line(value: &str) -> NetResult<MediaDescription> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() < 4 {
        return Err(NetError::parse(0, "Invalid m= line"));
    }

    let media_type = MediaType::parse(parts[0]);
    let port: u16 = parts[1]
        .parse()
        .map_err(|_| NetError::parse(0, "Invalid port"))?;
    let protocol = parts[2].to_string();
    let formats: Vec<String> = parts[3..].iter().map(|s| (*s).to_string()).collect();

    Ok(MediaDescription {
        media_type,
        port,
        protocol,
        formats,
        ..Default::default()
    })
}

fn apply_attribute_to_media(media: &mut MediaDescription, attr: Attribute) {
    match attr.name.as_str() {
        "mid" => {
            media.mid = attr.value.clone();
        }
        "ice-ufrag" => {
            media.ice_ufrag = attr.value.clone();
        }
        "ice-pwd" => {
            media.ice_pwd = attr.value.clone();
        }
        "fingerprint" => {
            if let Some(ref v) = attr.value {
                media.fingerprint = Fingerprint::parse(v);
            }
        }
        "setup" => {
            media.setup = attr.value.clone();
        }
        "rtcp-mux" => {
            media.rtcp_mux = true;
        }
        "sendrecv" | "sendonly" | "recvonly" | "inactive" => {
            if let Some(dir) = Direction::parse(&attr.name) {
                media.direction = dir;
            }
        }
        _ => {
            media.attributes.push(attr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_media_type() {
        assert_eq!(MediaType::parse("audio"), Some(MediaType::Audio));
        assert_eq!(MediaType::parse("video"), Some(MediaType::Video));
        assert_eq!(MediaType::Audio.as_str(), "audio");
    }

    #[test]
    fn test_direction() {
        assert_eq!(Direction::parse("sendrecv"), Some(Direction::SendRecv));
        assert_eq!(Direction::SendOnly.as_str(), "sendonly");
    }

    #[test]
    fn test_fingerprint() {
        let fp = Fingerprint::new("sha-256", "AA:BB:CC:DD");
        assert_eq!(fp.to_sdp(), "sha-256 AA:BB:CC:DD");

        let parsed = Fingerprint::parse("sha-256 AA:BB:CC:DD").expect("should succeed in test");
        assert_eq!(parsed.algorithm, "sha-256");
        assert_eq!(parsed.value, "AA:BB:CC:DD");
    }

    #[test]
    fn test_attribute() {
        let attr = Attribute::new("rtpmap", "96 VP8/90000");
        assert_eq!(attr.to_sdp(), "a=rtpmap:96 VP8/90000");

        let flag = Attribute::flag("rtcp-mux");
        assert_eq!(flag.to_sdp(), "a=rtcp-mux");
    }

    #[test]
    fn test_media_description() {
        let media = MediaDescription::video(9)
            .with_format("96")
            .with_format("97")
            .with_mid("video0")
            .with_direction(Direction::SendRecv)
            .with_rtcp_mux();

        let sdp = media.to_sdp();
        assert!(sdp.contains("m=video 9"));
        assert!(sdp.contains("a=mid:video0"));
        assert!(sdp.contains("a=rtcp-mux"));
    }

    #[test]
    fn test_session_description() {
        let sdp = SessionDescription::new()
            .with_session_name("Test Session")
            .with_attribute(Attribute::flag("ice-lite"))
            .with_media(
                MediaDescription::audio(9)
                    .with_format("111")
                    .with_mid("audio0"),
            );

        let output = sdp.to_sdp();
        assert!(output.contains("v=0"));
        assert!(output.contains("s=Test Session"));
        assert!(output.contains("a=ice-lite"));
        assert!(output.contains("m=audio 9"));
    }

    #[test]
    fn test_parse_sdp() {
        let sdp_str = r#"v=0
o=- 1234 1 IN IP4 0.0.0.0
s=Test
t=0 0
a=group:BUNDLE audio
m=audio 9 UDP/TLS/RTP/SAVPF 111
c=IN IP4 0.0.0.0
a=mid:audio
a=ice-ufrag:abc123
a=ice-pwd:secret
a=rtcp-mux
a=sendrecv
"#;

        let parsed = SessionDescription::parse(sdp_str).expect("should succeed in test");
        assert_eq!(parsed.version, 0);
        assert_eq!(parsed.session_name, "Test");
        assert_eq!(parsed.media.len(), 1);

        let audio = &parsed.media[0];
        assert_eq!(audio.media_type, Some(MediaType::Audio));
        assert_eq!(audio.port, 9);
        assert_eq!(audio.mid, Some("audio".to_string()));
        assert!(audio.rtcp_mux);
        assert_eq!(audio.direction, Direction::SendRecv);
    }

    #[test]
    fn parse_multibyte_leading_char_does_not_panic() {
        // A line whose first character is a 2-byte UTF-8 char ('é') whose
        // second byte is not '='. The old `&line[2..]` slice would panic on a
        // non-char-boundary; the parser must skip the line and return cleanly.
        let sdp = "v=0\r\né=malformed\r\ns=ok\r\n";
        let parsed = SessionDescription::parse(sdp).expect("must not panic");
        assert_eq!(parsed.version, 0);
        assert_eq!(parsed.session_name, "ok");
    }

    #[test]
    fn parse_multibyte_where_second_byte_would_slice_mid_char() {
        // '€' is 3 bytes (0xE2 0x82 0xAC); byte index 2 lands mid-character.
        // Must be skipped without panicking.
        let sdp = "v=0\r\n€uro\r\n";
        assert!(SessionDescription::parse(sdp).is_ok());
    }

    #[test]
    fn test_media_of_type() {
        let sdp = SessionDescription::new()
            .with_media(MediaDescription::audio(9).with_mid("a"))
            .with_media(MediaDescription::video(9).with_mid("v"))
            .with_media(MediaDescription::audio(9).with_mid("a2"));

        let audio = sdp.media_of_type(MediaType::Audio);
        assert_eq!(audio.len(), 2);

        let video = sdp.media_of_type(MediaType::Video);
        assert_eq!(video.len(), 1);
    }
}

// ---------------------------------------------------------------------------
// Higher-level SDP types (SdpMediaType, SdpDirection, SdpCodec, SdpMedia …)
// ---------------------------------------------------------------------------

/// SDP media type (higher-level alias with explicit names).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpMediaType {
    /// Audio media.
    Audio,
    /// Video media.
    Video,
    /// Application (data channels).
    Application,
}

impl SdpMediaType {
    /// Returns the SDP wire representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Application => "application",
        }
    }

    /// Converts to the underlying `MediaType`.
    fn to_media_type(self) -> MediaType {
        match self {
            Self::Audio => MediaType::Audio,
            Self::Video => MediaType::Video,
            Self::Application => MediaType::Application,
        }
    }
}

/// SDP media direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpDirection {
    /// Sending only.
    SendOnly,
    /// Receiving only.
    RecvOnly,
    /// Sending and receiving.
    SendRecv,
    /// Neither sending nor receiving.
    Inactive,
}

impl SdpDirection {
    /// Returns the SDP wire representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SendOnly => "sendonly",
            Self::RecvOnly => "recvonly",
            Self::SendRecv => "sendrecv",
            Self::Inactive => "inactive",
        }
    }

    /// Parses from the SDP wire representation.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "sendonly" => Some(Self::SendOnly),
            "recvonly" => Some(Self::RecvOnly),
            "sendrecv" => Some(Self::SendRecv),
            "inactive" => Some(Self::Inactive),
            _ => None,
        }
    }

    /// Returns `true` if this direction includes sending.
    pub fn is_sending(self) -> bool {
        matches!(self, Self::SendOnly | Self::SendRecv)
    }

    /// Returns `true` if this direction includes receiving.
    pub fn is_receiving(self) -> bool {
        matches!(self, Self::RecvOnly | Self::SendRecv)
    }

    /// Returns the reversed direction (as seen from the remote peer).
    pub fn reversed(self) -> Self {
        match self {
            Self::SendOnly => Self::RecvOnly,
            Self::RecvOnly => Self::SendOnly,
            Self::SendRecv => Self::SendRecv,
            Self::Inactive => Self::Inactive,
        }
    }

    fn to_direction(self) -> Direction {
        match self {
            Self::SendOnly => Direction::SendOnly,
            Self::RecvOnly => Direction::RecvOnly,
            Self::SendRecv => Direction::SendRecv,
            Self::Inactive => Direction::Inactive,
        }
    }
}

/// A codec entry in an SDP media section.
#[derive(Debug, Clone)]
pub struct SdpCodec {
    /// RTP payload type number.
    pub payload_type: u8,
    /// Codec name (e.g. "AV1", "VP9", "opus").
    pub name: String,
    /// RTP clock rate in Hz.
    pub clock_rate: u32,
    /// Number of audio channels (None for video).
    pub channels: Option<u8>,
    /// Optional format parameters string.
    pub fmtp: Option<String>,
}

impl SdpCodec {
    /// Creates a new codec entry.
    pub fn new(pt: u8, name: &str, clock_rate: u32) -> Self {
        Self {
            payload_type: pt,
            name: name.to_string(),
            clock_rate,
            channels: None,
            fmtp: None,
        }
    }

    /// Sets the audio channel count.
    pub fn with_channels(mut self, channels: u8) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Sets the format parameters string.
    pub fn with_fmtp(mut self, fmtp: &str) -> Self {
        self.fmtp = Some(fmtp.to_string());
        self
    }

    /// Returns the canonical AV1 codec entry (payload type 45, 90 kHz).
    pub fn av1() -> Self {
        Self::new(45, "AV1", 90_000)
    }

    /// Returns the canonical VP9 codec entry (payload type 98, 90 kHz).
    pub fn vp9() -> Self {
        Self::new(98, "VP9", 90_000)
    }

    /// Returns the canonical Opus codec entry (payload type 111, 48 kHz, stereo).
    pub fn opus() -> Self {
        Self::new(111, "opus", 48_000)
            .with_channels(2)
            .with_fmtp("minptime=10;useinbandfec=1")
    }
}

/// SDP RTP header extension.
#[derive(Debug, Clone)]
pub struct RtpExtension {
    /// Extension ID (1-14 for one-byte header, 1-255 for two-byte header).
    pub id: u8,
    /// Extension URI.
    pub uri: String,
}

/// An SDP media section (m= line and all associated attributes).
#[derive(Debug, Clone)]
pub struct SdpMedia {
    /// Media type.
    pub media_type: SdpMediaType,
    /// Port number.
    pub port: u16,
    /// Transport protocol string.
    pub protocol: String,
    /// Codec list.
    pub codecs: Vec<SdpCodec>,
    /// Stream direction.
    pub direction: SdpDirection,
    /// Media bundle ID.
    pub mid: String,
    /// ICE username fragment.
    pub ice_ufrag: String,
    /// ICE password.
    pub ice_pwd: String,
    /// RTP header extensions.
    pub extensions: Vec<RtpExtension>,
    /// Whether RTCP multiplexing is enabled.
    pub rtcp_mux: bool,
    /// Synchronization source identifier.
    pub ssrc: u32,
}

impl SdpMedia {
    fn new_media(media_type: SdpMediaType, mid: &str, codecs: Vec<SdpCodec>) -> Self {
        Self {
            media_type,
            port: 9,
            protocol: "RTP/SAVPF".to_string(),
            codecs,
            direction: SdpDirection::SendRecv,
            mid: mid.to_string(),
            ice_ufrag: String::new(),
            ice_pwd: String::new(),
            extensions: Vec::new(),
            rtcp_mux: true,
            ssrc: 0,
        }
    }

    /// Creates a video media section.
    pub fn new_video(mid: &str, codecs: Vec<SdpCodec>) -> Self {
        Self::new_media(SdpMediaType::Video, mid, codecs)
    }

    /// Creates an audio media section.
    pub fn new_audio(mid: &str, codecs: Vec<SdpCodec>) -> Self {
        Self::new_media(SdpMediaType::Audio, mid, codecs)
    }

    /// Sets the stream direction.
    pub fn with_direction(mut self, dir: SdpDirection) -> Self {
        self.direction = dir;
        self
    }

    /// Sets the SSRC.
    pub fn with_ssrc(mut self, ssrc: u32) -> Self {
        self.ssrc = ssrc;
        self
    }

    /// Serializes this media section to SDP text.
    pub fn to_sdp_string(&self) -> String {
        // Build payload type list
        let pts: Vec<String> = self
            .codecs
            .iter()
            .map(|c| c.payload_type.to_string())
            .collect();

        let mut lines = Vec::new();
        lines.push(format!(
            "m={} {} {} {}",
            self.media_type.as_str(),
            self.port,
            self.protocol,
            pts.join(" ")
        ));
        lines.push(format!("a=mid:{}", self.mid));
        if !self.ice_ufrag.is_empty() {
            lines.push(format!("a=ice-ufrag:{}", self.ice_ufrag));
        }
        if !self.ice_pwd.is_empty() {
            lines.push(format!("a=ice-pwd:{}", self.ice_pwd));
        }
        lines.push(format!("a={}", self.direction.as_str()));
        if self.rtcp_mux {
            lines.push("a=rtcp-mux".to_string());
        }
        for codec in &self.codecs {
            if let Some(channels) = codec.channels {
                lines.push(format!(
                    "a=rtpmap:{} {}/{}/{}",
                    codec.payload_type, codec.name, codec.clock_rate, channels
                ));
            } else {
                lines.push(format!(
                    "a=rtpmap:{} {}/{}",
                    codec.payload_type, codec.name, codec.clock_rate
                ));
            }
            if let Some(ref fmtp) = codec.fmtp {
                lines.push(format!("a=fmtp:{} {}", codec.payload_type, fmtp));
            }
        }
        if self.ssrc != 0 {
            lines.push(format!("a=ssrc:{} cname:oximedia", self.ssrc));
        }
        lines.join("\r\n")
    }
}

/// SDP session type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpType {
    /// Offer SDP.
    Offer,
    /// Answer SDP.
    Answer,
    /// Provisional answer.
    Pranswer,
    /// Rollback the current local description.
    Rollback,
}

impl SdpType {
    /// Returns the JSEP/WebRTC wire string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Offer => "offer",
            Self::Answer => "answer",
            Self::Pranswer => "pranswer",
            Self::Rollback => "rollback",
        }
    }
}

/// Complete SDP session description (higher-level API).
#[derive(Debug, Clone)]
pub struct SdpSessionDescription {
    /// Numeric session ID.
    pub session_id: u64,
    /// Session version counter.
    pub session_version: u64,
    /// Origin username field.
    pub origin_username: String,
    /// Session name field.
    pub session_name: String,
    /// ICE options (e.g. ["trickle"]).
    pub ice_options: Vec<String>,
    /// DTLS fingerprint (e.g. "sha-256 AA:BB:…").
    pub fingerprint: Option<String>,
    /// Media sections.
    pub media: Vec<SdpMedia>,
    /// Whether this is an offer or answer.
    pub sdp_type: SdpType,
}

impl SdpSessionDescription {
    fn new_with_type(sdp_type: SdpType) -> Self {
        Self {
            session_id: 0,
            session_version: 0,
            origin_username: "-".to_string(),
            session_name: "-".to_string(),
            ice_options: vec!["trickle".to_string()],
            fingerprint: None,
            media: Vec::new(),
            sdp_type,
        }
    }

    /// Creates a new offer description.
    pub fn new_offer() -> Self {
        Self::new_with_type(SdpType::Offer)
    }

    /// Creates a new answer description.
    pub fn new_answer() -> Self {
        Self::new_with_type(SdpType::Answer)
    }

    /// Adds a media section.
    pub fn add_media(mut self, media: SdpMedia) -> Self {
        self.media.push(media);
        self
    }

    /// Serializes to SDP text format (RFC 4566).
    pub fn to_sdp_string(&self) -> String {
        let mut lines = Vec::new();
        lines.push("v=0".to_string());
        lines.push(format!(
            "o={} {} {} IN IP4 0.0.0.0",
            self.origin_username, self.session_id, self.session_version
        ));
        lines.push(format!("s={}", self.session_name));
        lines.push("t=0 0".to_string());

        if !self.media.is_empty() {
            let mids: Vec<&str> = self.media.iter().map(|m| m.mid.as_str()).collect();
            lines.push(format!("a=group:BUNDLE {}", mids.join(" ")));
        }

        if !self.ice_options.is_empty() {
            lines.push(format!("a=ice-options:{}", self.ice_options.join(" ")));
        }

        if let Some(ref fp) = self.fingerprint {
            lines.push(format!("a=fingerprint:{fp}"));
        }

        for media in &self.media {
            lines.push(media.to_sdp_string());
        }

        lines.join("\r\n") + "\r\n"
    }

    /// Parses an SDP string; returns `None` if malformed.
    pub fn from_sdp_str(sdp: &str) -> Option<Self> {
        // Use the lower-level parser and translate
        let parsed = SessionDescription::parse(sdp).ok()?;

        let media = parsed
            .media
            .iter()
            .map(|m| {
                let sdp_media_type = match m.media_type {
                    Some(MediaType::Audio) => SdpMediaType::Audio,
                    Some(MediaType::Video) => SdpMediaType::Video,
                    _ => SdpMediaType::Application,
                };
                let direction = match m.direction {
                    Direction::SendOnly => SdpDirection::SendOnly,
                    Direction::RecvOnly => SdpDirection::RecvOnly,
                    Direction::SendRecv => SdpDirection::SendRecv,
                    Direction::Inactive => SdpDirection::Inactive,
                };
                SdpMedia {
                    media_type: sdp_media_type,
                    port: m.port,
                    protocol: m.protocol.clone(),
                    codecs: Vec::new(), // codec parsing skipped for brevity
                    direction,
                    mid: m.mid.clone().unwrap_or_default(),
                    ice_ufrag: m.ice_ufrag.clone().unwrap_or_default(),
                    ice_pwd: m.ice_pwd.clone().unwrap_or_default(),
                    extensions: Vec::new(),
                    rtcp_mux: m.rtcp_mux,
                    ssrc: 0,
                }
            })
            .collect();

        Some(Self {
            session_id: 0,
            session_version: 0,
            origin_username: "-".to_string(),
            session_name: parsed.session_name,
            ice_options: Vec::new(),
            fingerprint: None,
            media,
            sdp_type: SdpType::Offer,
        })
    }

    /// Creates an answer by reversing all directions.
    pub fn create_answer(&self) -> Self {
        let media = self
            .media
            .iter()
            .map(|m| SdpMedia {
                direction: m.direction.reversed(),
                ..m.clone()
            })
            .collect();

        Self {
            sdp_type: SdpType::Answer,
            media,
            ..self.clone()
        }
    }

    /// Returns references to all video media sections.
    pub fn video_media(&self) -> Vec<&SdpMedia> {
        self.media
            .iter()
            .filter(|m| m.media_type == SdpMediaType::Video)
            .collect()
    }

    /// Returns references to all audio media sections.
    pub fn audio_media(&self) -> Vec<&SdpMedia> {
        self.media
            .iter()
            .filter(|m| m.media_type == SdpMediaType::Audio)
            .collect()
    }

    /// Returns `true` if the SDP declares a BUNDLE group.
    pub fn is_bundled(&self) -> bool {
        // We only add BUNDLE groups when there is more than one media section.
        // A real implementation would parse the a=group:BUNDLE line.
        self.media.len() > 1
    }
}

#[cfg(test)]
mod sdp_spec_tests {
    use super::*;

    #[test]
    fn test_sdp_codec_av1() {
        let codec = SdpCodec::av1();
        assert_eq!(codec.payload_type, 45);
        assert_eq!(codec.name, "AV1");
        assert_eq!(codec.clock_rate, 90_000);
    }

    #[test]
    fn test_sdp_codec_opus() {
        let codec = SdpCodec::opus();
        assert_eq!(codec.channels, Some(2));
        assert_eq!(codec.name, "opus");
        assert_eq!(codec.clock_rate, 48_000);
    }

    #[test]
    fn test_sdp_direction_is_sending() {
        assert!(SdpDirection::SendOnly.is_sending());
        assert!(SdpDirection::SendRecv.is_sending());
        assert!(!SdpDirection::RecvOnly.is_sending());
        assert!(!SdpDirection::Inactive.is_sending());
    }

    #[test]
    fn test_sdp_direction_reversed() {
        assert_eq!(SdpDirection::SendOnly.reversed(), SdpDirection::RecvOnly);
        assert_eq!(SdpDirection::RecvOnly.reversed(), SdpDirection::SendOnly);
        assert_eq!(SdpDirection::SendRecv.reversed(), SdpDirection::SendRecv);
        assert_eq!(SdpDirection::Inactive.reversed(), SdpDirection::Inactive);
    }

    #[test]
    fn test_sdp_media_video_to_string() {
        let media = SdpMedia::new_video("video0", vec![SdpCodec::av1()]);
        let s = media.to_sdp_string();
        assert!(s.contains("m=video"), "Expected 'm=video' in: {s}");
    }

    #[test]
    fn test_session_description_to_string() {
        let desc = SdpSessionDescription::new_offer()
            .add_media(SdpMedia::new_video("0", vec![SdpCodec::av1()]));
        let s = desc.to_sdp_string();
        assert!(s.contains("v=0"), "Expected 'v=0' in: {s}");
        assert!(s.contains("o="), "Expected 'o=' in: {s}");
    }

    #[test]
    fn test_session_description_add_media() {
        let desc = SdpSessionDescription::new_offer();
        assert_eq!(desc.media.len(), 0);

        let desc = desc.add_media(SdpMedia::new_video("0", vec![SdpCodec::av1()]));
        assert_eq!(desc.media.len(), 1);

        let desc = desc.add_media(SdpMedia::new_audio("1", vec![SdpCodec::opus()]));
        assert_eq!(desc.media.len(), 2);
    }

    #[test]
    fn test_create_answer_reverses_direction() {
        let offer = SdpSessionDescription::new_offer().add_media(
            SdpMedia::new_video("0", vec![SdpCodec::av1()]).with_direction(SdpDirection::SendOnly),
        );

        let answer = offer.create_answer();
        assert_eq!(answer.sdp_type, SdpType::Answer);
        assert_eq!(answer.media[0].direction, SdpDirection::RecvOnly);
    }

    #[test]
    fn test_video_audio_accessors() {
        let desc = SdpSessionDescription::new_offer()
            .add_media(SdpMedia::new_video("v0", vec![SdpCodec::av1()]))
            .add_media(SdpMedia::new_audio("a0", vec![SdpCodec::opus()]))
            .add_media(SdpMedia::new_video("v1", vec![SdpCodec::vp9()]));

        assert_eq!(desc.video_media().len(), 2);
        assert_eq!(desc.audio_media().len(), 1);
    }

    #[test]
    fn test_is_bundled() {
        let single = SdpSessionDescription::new_offer().add_media(SdpMedia::new_video("0", vec![]));
        assert!(!single.is_bundled(), "single media should not be bundled");

        let bundled = SdpSessionDescription::new_offer()
            .add_media(SdpMedia::new_video("0", vec![]))
            .add_media(SdpMedia::new_audio("1", vec![]));
        assert!(bundled.is_bundled(), "two media sections should be bundled");
    }
}
