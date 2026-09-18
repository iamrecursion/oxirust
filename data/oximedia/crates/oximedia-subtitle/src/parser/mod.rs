//! Subtitle format parsers.
//!
//! This module provides parsers for various subtitle formats:
//!
//! - [`srt`] - SubRip (`.srt`) files
//! - [`webvtt`] - WebVTT (`.vtt`) files
//! - [`ssa`] - SubStation Alpha/Advanced SubStation Alpha (`.ssa`/`.ass`) files
//! - [`ttml`] - TTML (Timed Text Markup Language) files
//! - [`cea608_decoder`] - CEA-608 closed caption decoder
//! - [`cea708_decoder`] - CEA-708 closed caption decoder
//! - [`dvb`] - DVB (Digital Video Broadcasting) subtitles
//! - [`pgs`] - PGS (Presentation Graphic Stream) Blu-ray subtitles

pub mod cea608_decoder;
pub mod cea708_decoder;
pub mod dvb;
// Stateful real-time CEA-608 decoder for live streams (`RealtimeCea608Decoder`).
// Note: crate::eia608_realtime (top-level) is a separate module with a different
// API (`Eia608Decoder`).  No symbol collision — different module paths.
pub mod eia608_realtime;
pub mod pgs;
pub mod srt;
pub mod ssa;
pub mod ttml;
pub mod webvtt;

#[cfg(feature = "closed-captions")]
pub mod cea608;

use crate::{SubtitleError, SubtitleResult};

/// Auto-detect subtitle format from file extension or content.
///
/// # Errors
///
/// Returns error if format cannot be detected.
pub fn detect_format(data: &[u8], filename: Option<&str>) -> SubtitleResult<SubtitleFormat> {
    // Try filename first
    if let Some(name) = filename {
        let lower = name.to_lowercase();
        if lower.ends_with(".srt") {
            return Ok(SubtitleFormat::Srt);
        }
        if lower.ends_with(".vtt") || lower.ends_with(".webvtt") {
            return Ok(SubtitleFormat::WebVtt);
        }
        if lower.ends_with(".ssa") {
            return Ok(SubtitleFormat::Ssa);
        }
        if lower.ends_with(".ass") {
            return Ok(SubtitleFormat::Ass);
        }
        if lower.ends_with(".ttml") || lower.ends_with(".xml") {
            return Ok(SubtitleFormat::Ttml);
        }
        if lower.ends_with(".sup") {
            return Ok(SubtitleFormat::Pgs);
        }
    }

    // Try content detection
    let text = String::from_utf8_lossy(data);

    if text.starts_with("WEBVTT") {
        return Ok(SubtitleFormat::WebVtt);
    }

    if text.contains("[Script Info]") {
        if text.contains("ScriptType: v4.00+") {
            return Ok(SubtitleFormat::Ass);
        }
        return Ok(SubtitleFormat::Ssa);
    }

    // Check for TTML (XML-based)
    if text.trim_start().starts_with("<?xml") && text.contains("<tt") {
        return Ok(SubtitleFormat::Ttml);
    }

    // Check for PGS magic bytes
    if data.len() >= 2 && &data[0..2] == b"PG" {
        return Ok(SubtitleFormat::Pgs);
    }

    // Check for DVB subtitle sync byte
    if !data.is_empty() && data[0] == 0x0F {
        return Ok(SubtitleFormat::Dvb);
    }

    // SRT is detected by pattern matching
    if srt::is_srt_format(&text) {
        return Ok(SubtitleFormat::Srt);
    }

    Err(SubtitleError::InvalidFormat(
        "Cannot detect subtitle format".to_string(),
    ))
}

/// Supported subtitle formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubtitleFormat {
    /// SubRip (.srt).
    Srt,
    /// WebVTT (.vtt).
    WebVtt,
    /// SubStation Alpha (.ssa).
    Ssa,
    /// Advanced SubStation Alpha (.ass).
    Ass,
    /// TTML (.ttml).
    Ttml,
    /// CEA-608 closed captions.
    #[cfg(feature = "closed-captions")]
    Cea608,
    /// CEA-708 closed captions.
    #[cfg(feature = "closed-captions")]
    Cea708,
    /// DVB subtitles.
    Dvb,
    /// PGS (Blu-ray) subtitles.
    Pgs,
}

impl SubtitleFormat {
    /// Get the typical file extension for this format.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::WebVtt => "vtt",
            Self::Ssa => "ssa",
            Self::Ass => "ass",
            Self::Ttml => "ttml",
            #[cfg(feature = "closed-captions")]
            Self::Cea608 => "608",
            #[cfg(feature = "closed-captions")]
            Self::Cea708 => "708",
            Self::Dvb => "dvb",
            Self::Pgs => "sup",
        }
    }

    /// Get a human-readable name for this format.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Srt => "SubRip",
            Self::WebVtt => "WebVTT",
            Self::Ssa => "SubStation Alpha",
            Self::Ass => "Advanced SubStation Alpha",
            Self::Ttml => "Timed Text Markup Language",
            #[cfg(feature = "closed-captions")]
            Self::Cea608 => "CEA-608",
            #[cfg(feature = "closed-captions")]
            Self::Cea708 => "CEA-708",
            Self::Dvb => "DVB Subtitles",
            Self::Pgs => "PGS (Blu-ray)",
        }
    }
}
