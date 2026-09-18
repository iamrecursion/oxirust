//! DDEX ERN 4.1 message parsing.
//!
//! Implements a manual XML parser for DDEX Electronic Release Notification (ERN)
//! 4.1 messages without any external XML crate dependency.  All parsing is done
//! via byte-level string operations on the raw XML text.

#![allow(missing_docs)]

use crate::{Result, RightsError};

// ── Public types ─────────────────────────────────────────────────────────────

/// Top-level DDEX ERN 4.1 message.
#[derive(Debug, Clone, PartialEq)]
pub struct DdexErnMessage {
    /// Message sender party identifier.
    pub message_sender: String,
    /// Message recipient party identifier.
    pub message_recipient: String,
    /// ISO 8601 date/time the message was created.
    pub message_created: String,
    /// Releases contained in this ERN.
    pub releases: Vec<DdexRelease>,
}

/// A release (e.g. an album or single) within a DDEX ERN message.
#[derive(Debug, Clone, PartialEq)]
pub struct DdexRelease {
    /// Release identifier (e.g. a UPC or proprietary ID).
    pub release_id: String,
    /// Display title of the release.
    pub title: String,
    /// Names of the principal artists for this release.
    pub artists: Vec<String>,
    /// Sound recordings contained in this release.
    pub recordings: Vec<DdexRecording>,
}

/// A sound recording reference within a DDEX ERN release.
#[derive(Debug, Clone, PartialEq)]
pub struct DdexRecording {
    /// International Standard Recording Code.
    pub isrc: String,
    /// Display title of the recording.
    pub title: String,
    /// Duration of the recording in milliseconds.
    pub duration_ms: u64,
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Stateless DDEX ERN parser.
pub struct DdexParser;

impl DdexParser {
    /// Parse a DDEX ERN 4.1 XML string into a [`DdexErnMessage`].
    ///
    /// Returns [`RightsError::Serialization`] on any structural or format error.
    pub fn parse(xml: &str) -> Result<DdexErnMessage> {
        let sender = extract_tag_content(xml, "MessageSender")
            .unwrap_or_default()
            .to_string();
        let recipient = extract_tag_content(xml, "MessageRecipient")
            .unwrap_or_default()
            .to_string();
        let created = extract_tag_content(xml, "MessageCreatedDateTime")
            .unwrap_or_default()
            .to_string();

        let releases = parse_releases(xml)?;

        Ok(DdexErnMessage {
            message_sender: sender,
            message_recipient: recipient,
            message_created: created,
            releases,
        })
    }
}

// ── Internal parsing helpers ──────────────────────────────────────────────────

/// Extract the text content of the *first* occurrence of `<tag>…</tag>` in `src`.
///
/// Returns `None` if the tag is not present.
fn extract_tag_content<'a>(src: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);

    let start_tag = src.find(open.as_str())?;
    // Skip to the `>` that ends the opening tag (handles attributes).
    let gt = src[start_tag..].find('>')?;
    let content_start = start_tag + gt + 1;

    let content_end = src[content_start..].find(close.as_str())?;
    Some(src[content_start..content_start + content_end].trim())
}

/// Extract the text content of `<tag>…</tag>` within a sub-slice `src`,
/// searching *only* within that slice.
fn extract_tag_in<'a>(src: &'a str, tag: &str) -> Option<&'a str> {
    extract_tag_content(src, tag)
}

/// Return all sub-strings that are enclosed by `<tag …>…</tag>` pairs in `src`.
fn extract_all_blocks<'a>(src: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let mut result = Vec::new();
    let mut search_from = 0usize;

    while search_from < src.len() {
        let remaining = &src[search_from..];
        let tag_start = match remaining.find(open.as_str()) {
            Some(p) => p,
            None => break,
        };
        // Find the `>` that closes the opening tag.
        let gt_rel = match remaining[tag_start..].find('>') {
            Some(p) => p,
            None => break,
        };
        let block_content_start = search_from + tag_start + gt_rel + 1;

        // Find closing tag starting from block_content_start.
        let after_open = &src[block_content_start..];
        let close_pos = match after_open.find(close.as_str()) {
            Some(p) => p,
            None => break,
        };
        result.push(&src[block_content_start..block_content_start + close_pos]);

        search_from = block_content_start + close_pos + close.len();
    }

    result
}

/// Parse ISO 8601 duration string (PT…M…S) into milliseconds.
///
/// Supports `PT<minutes>M<seconds>S` and `PT<seconds>S` and `PT<hours>H<minutes>M<seconds>S`.
/// Fractional seconds are supported (e.g. `PT3M45.500S`).
fn parse_iso_duration_ms(duration: &str) -> Result<u64> {
    // Strip leading `PT` or `P`
    let s = duration.trim();
    let s = if let Some(rest) = s.strip_prefix("PT") {
        rest
    } else if let Some(rest) = s.strip_prefix('P') {
        rest
    } else {
        return Err(RightsError::Serialization(format!(
            "Cannot parse ISO duration: {duration}"
        )));
    };

    let mut total_ms: f64 = 0.0;
    let mut remaining = s;

    // Hours
    if let Some(h_pos) = remaining.find('H') {
        let h_str = &remaining[..h_pos];
        let hours: f64 = h_str.parse().map_err(|_| {
            RightsError::Serialization(format!("Invalid hours in duration: {duration}"))
        })?;
        total_ms += hours * 3_600_000.0;
        remaining = &remaining[h_pos + 1..];
    }

    // Minutes
    if let Some(m_pos) = remaining.find('M') {
        let m_str = &remaining[..m_pos];
        let minutes: f64 = m_str.parse().map_err(|_| {
            RightsError::Serialization(format!("Invalid minutes in duration: {duration}"))
        })?;
        total_ms += minutes * 60_000.0;
        remaining = &remaining[m_pos + 1..];
    }

    // Seconds (possibly fractional)
    if let Some(s_pos) = remaining.find('S') {
        let sec_str = &remaining[..s_pos];
        let seconds: f64 = sec_str.parse().map_err(|_| {
            RightsError::Serialization(format!("Invalid seconds in duration: {duration}"))
        })?;
        total_ms += seconds * 1_000.0;
    }

    Ok(total_ms as u64)
}

/// Parse all `<Release>` blocks from the ERN XML.
fn parse_releases(xml: &str) -> Result<Vec<DdexRelease>> {
    let release_blocks = extract_all_blocks(xml, "Release");
    let mut releases = Vec::new();

    for block in release_blocks {
        let release_id = extract_release_id(block);
        let title = extract_tag_in(block, "TitleText").unwrap_or("").to_string();
        let artists = extract_artists(block);
        let recordings = parse_recordings(block)?;

        releases.push(DdexRelease {
            release_id,
            title,
            artists,
            recordings,
        });
    }

    Ok(releases)
}

/// Extract the release ID — tries `<ReleaseId>`, `<ProprietaryId>`, `<GRid>`, `<ICPN>` in order.
fn extract_release_id(block: &str) -> String {
    // Try nested <ReleaseId> block first, then fall back to top-level identifiers.
    if let Some(id_block) = extract_all_blocks(block, "ReleaseId").into_iter().next() {
        for tag in &["GRid", "ICPN", "ProprietaryId", "CatalogNumber"] {
            if let Some(v) = extract_tag_in(id_block, tag) {
                if !v.is_empty() {
                    return v.to_string();
                }
            }
        }
    }
    // Fall back to a direct `<ReleaseReference>` or any id-like tag.
    extract_tag_in(block, "ReleaseReference")
        .unwrap_or("")
        .to_string()
}

/// Collect all artist display names from a release block.
fn extract_artists(block: &str) -> Vec<String> {
    let mut artists = Vec::new();

    // Artists live inside <DisplayArtist> or <MainArtist> blocks.
    for container_tag in &["DisplayArtist", "MainArtist", "Artist"] {
        for artist_block in extract_all_blocks(block, container_tag) {
            // Try <FullName> then <ArtistName>/<FullName>
            let name = extract_tag_in(artist_block, "FullName")
                .or_else(|| extract_tag_in(artist_block, "ArtistName"))
                .unwrap_or("")
                .trim()
                .to_string();
            if !name.is_empty() && !artists.contains(&name) {
                artists.push(name);
            }
        }
    }

    artists
}

/// Parse all `<SoundRecording>` blocks within a release block.
fn parse_recordings(release_block: &str) -> Result<Vec<DdexRecording>> {
    let recording_blocks = extract_all_blocks(release_block, "SoundRecording");
    let mut recordings = Vec::new();

    for rec_block in recording_blocks {
        let isrc = extract_isrc(rec_block);
        let title = extract_tag_in(rec_block, "TitleText")
            .unwrap_or("")
            .to_string();
        let duration_str = extract_tag_in(rec_block, "Duration").unwrap_or("PT0S");
        let duration_ms = parse_iso_duration_ms(duration_str)?;

        recordings.push(DdexRecording {
            isrc,
            title,
            duration_ms,
        });
    }

    Ok(recordings)
}

/// Extract the ISRC from a recording block.
fn extract_isrc(block: &str) -> String {
    if let Some(id_block) = extract_all_blocks(block, "SoundRecordingId")
        .into_iter()
        .next()
    {
        if let Some(v) = extract_tag_in(id_block, "ISRC") {
            return v.to_string();
        }
    }
    extract_tag_in(block, "ISRC").unwrap_or("").to_string()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── minimal ERN skeleton ──────────────────────────────────────────────────

    fn minimal_ern() -> &'static str {
        r#"<?xml version="1.0" encoding="UTF-8"?>
<ernm:NewReleaseMessage xmlns:ernm="http://ddex.net/xml/ern/41">
  <MessageHeader>
    <MessageSender>SENDER001</MessageSender>
    <MessageRecipient>RECIPIENT002</MessageRecipient>
    <MessageCreatedDateTime>2024-01-15T10:30:00</MessageCreatedDateTime>
  </MessageHeader>
  <ReleaseList>
    <Release>
      <ReleaseId>
        <ICPN>0123456789012</ICPN>
      </ReleaseId>
      <ReferenceTitle>
        <TitleText>My Album</TitleText>
      </ReferenceTitle>
      <DisplayArtist>
        <FullName>Artist One</FullName>
      </DisplayArtist>
      <SoundRecording>
        <SoundRecordingId>
          <ISRC>USABC2400001</ISRC>
        </SoundRecordingId>
        <ReferenceTitle>
          <TitleText>Track One</TitleText>
        </ReferenceTitle>
        <Duration>PT3M45S</Duration>
      </SoundRecording>
    </Release>
  </ReleaseList>
</ernm:NewReleaseMessage>"#
    }

    // ── basic parse tests ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_message_sender() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        assert_eq!(msg.message_sender, "SENDER001");
    }

    #[test]
    fn test_parse_message_recipient() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        assert_eq!(msg.message_recipient, "RECIPIENT002");
    }

    #[test]
    fn test_parse_message_created() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        assert_eq!(msg.message_created, "2024-01-15T10:30:00");
    }

    #[test]
    fn test_parse_release_count() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        assert_eq!(msg.releases.len(), 1);
    }

    #[test]
    fn test_parse_release_id() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        assert_eq!(msg.releases[0].release_id, "0123456789012");
    }

    #[test]
    fn test_parse_release_title() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        assert_eq!(msg.releases[0].title, "My Album");
    }

    #[test]
    fn test_parse_release_artists() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        assert_eq!(msg.releases[0].artists, vec!["Artist One".to_string()]);
    }

    #[test]
    fn test_parse_recording_count() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        assert_eq!(msg.releases[0].recordings.len(), 1);
    }

    #[test]
    fn test_parse_recording_isrc() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        assert_eq!(msg.releases[0].recordings[0].isrc, "USABC2400001");
    }

    #[test]
    fn test_parse_recording_title() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        assert_eq!(msg.releases[0].recordings[0].title, "Track One");
    }

    #[test]
    fn test_parse_recording_duration_ms() {
        let msg = DdexParser::parse(minimal_ern()).expect("parse should succeed");
        // PT3M45S = 3*60_000 + 45_000 = 225_000 ms
        assert_eq!(msg.releases[0].recordings[0].duration_ms, 225_000);
    }

    // ── ISO duration parsing ──────────────────────────────────────────────────

    #[test]
    fn test_parse_iso_duration_seconds_only() {
        assert_eq!(parse_iso_duration_ms("PT45S").expect("parse ok"), 45_000);
    }

    #[test]
    fn test_parse_iso_duration_minutes_only() {
        assert_eq!(parse_iso_duration_ms("PT3M").expect("parse ok"), 180_000);
    }

    #[test]
    fn test_parse_iso_duration_hours_minutes_seconds() {
        // PT1H2M3S = 3600_000 + 120_000 + 3_000 = 3_723_000
        assert_eq!(
            parse_iso_duration_ms("PT1H2M3S").expect("parse ok"),
            3_723_000
        );
    }

    #[test]
    fn test_parse_iso_duration_fractional_seconds() {
        // PT1M30.500S = 60_000 + 30_500 = 90_500
        assert_eq!(
            parse_iso_duration_ms("PT1M30.500S").expect("parse ok"),
            90_500
        );
    }

    #[test]
    fn test_parse_iso_duration_zero() {
        assert_eq!(parse_iso_duration_ms("PT0S").expect("parse ok"), 0);
    }

    // ── multi-release & multi-recording ──────────────────────────────────────

    #[test]
    fn test_parse_multiple_releases() {
        let xml = r#"<Root>
  <MessageSender>S1</MessageSender>
  <MessageRecipient>R1</MessageRecipient>
  <MessageCreatedDateTime>2024-06-01</MessageCreatedDateTime>
  <Release>
    <ReleaseId><ICPN>ICPN001</ICPN></ReleaseId>
    <TitleText>Album Alpha</TitleText>
  </Release>
  <Release>
    <ReleaseId><ICPN>ICPN002</ICPN></ReleaseId>
    <TitleText>Album Beta</TitleText>
  </Release>
</Root>"#;
        let msg = DdexParser::parse(xml).expect("parse ok");
        assert_eq!(msg.releases.len(), 2);
        assert_eq!(msg.releases[0].release_id, "ICPN001");
        assert_eq!(msg.releases[1].release_id, "ICPN002");
    }

    #[test]
    fn test_parse_multiple_recordings() {
        let xml = r#"<Root>
  <MessageSender>S</MessageSender>
  <MessageRecipient>R</MessageRecipient>
  <MessageCreatedDateTime>2024-01-01</MessageCreatedDateTime>
  <Release>
    <ReleaseId><ICPN>X</ICPN></ReleaseId>
    <TitleText>EP</TitleText>
    <SoundRecording>
      <SoundRecordingId><ISRC>AA0000000001</ISRC></SoundRecordingId>
      <TitleText>Song A</TitleText>
      <Duration>PT2M10S</Duration>
    </SoundRecording>
    <SoundRecording>
      <SoundRecordingId><ISRC>AA0000000002</ISRC></SoundRecordingId>
      <TitleText>Song B</TitleText>
      <Duration>PT3M20S</Duration>
    </SoundRecording>
  </Release>
</Root>"#;
        let msg = DdexParser::parse(xml).expect("parse ok");
        assert_eq!(msg.releases[0].recordings.len(), 2);
        assert_eq!(msg.releases[0].recordings[0].isrc, "AA0000000001");
        assert_eq!(msg.releases[0].recordings[1].isrc, "AA0000000002");
        // PT2M10S = 130_000 ms
        assert_eq!(msg.releases[0].recordings[0].duration_ms, 130_000);
        // PT3M20S = 200_000 ms
        assert_eq!(msg.releases[0].recordings[1].duration_ms, 200_000);
    }

    #[test]
    fn test_parse_multiple_artists() {
        let xml = r#"<Root>
  <MessageSender>S</MessageSender>
  <MessageRecipient>R</MessageRecipient>
  <MessageCreatedDateTime>2024-01-01</MessageCreatedDateTime>
  <Release>
    <ReleaseId><ICPN>X</ICPN></ReleaseId>
    <TitleText>Collab</TitleText>
    <DisplayArtist><FullName>Artist X</FullName></DisplayArtist>
    <DisplayArtist><FullName>Artist Y</FullName></DisplayArtist>
  </Release>
</Root>"#;
        let msg = DdexParser::parse(xml).expect("parse ok");
        assert!(msg.releases[0].artists.contains(&"Artist X".to_string()));
        assert!(msg.releases[0].artists.contains(&"Artist Y".to_string()));
    }

    #[test]
    fn test_parse_empty_xml() {
        let msg = DdexParser::parse("<Root></Root>").expect("parse ok (empty)");
        assert!(msg.releases.is_empty());
        assert!(msg.message_sender.is_empty());
    }

    #[test]
    fn test_invalid_duration_returns_error() {
        let err = parse_iso_duration_ms("INVALID");
        assert!(err.is_err());
    }

    #[test]
    fn test_extract_tag_content_missing() {
        let result = extract_tag_content("<Root></Root>", "Missing");
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_all_blocks_multiple() {
        let xml = "<R><Item>A</Item><Item>B</Item><Item>C</Item></R>";
        let blocks = extract_all_blocks(xml, "Item");
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0], "A");
        assert_eq!(blocks[2], "C");
    }
}
