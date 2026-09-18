//! Format parsers and writers for various caption formats

pub mod arib;
pub mod ass;
pub mod cea608;
pub mod cea708;
pub mod dfxp;
pub mod dvb;
pub mod itt;
pub mod pgs;
pub mod scc;
pub mod srt;
pub mod ssa;
pub mod stl;
pub mod teletext;
pub mod ttml;
pub mod vobsub;
pub mod webvtt;

use crate::error::Result;
use crate::types::CaptionTrack;
use crate::CaptionFormat;

/// Format parser trait
pub trait FormatParser {
    /// Parse caption data
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack>;
}

/// Format writer trait
pub trait FormatWriter {
    /// Write caption track to bytes
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>>;
}

/// Get parser for a format
#[must_use]
pub fn get_parser(format: CaptionFormat) -> Option<Box<dyn FormatParser>> {
    match format {
        CaptionFormat::Srt => Some(Box::new(srt::SrtParser)),
        CaptionFormat::WebVtt => Some(Box::new(webvtt::WebVttParser)),
        CaptionFormat::Ass => Some(Box::new(ass::AssParser)),
        CaptionFormat::Ssa => Some(Box::new(ssa::SsaParser)),
        CaptionFormat::Ttml | CaptionFormat::Dfxp => Some(Box::new(ttml::TtmlParser)),
        CaptionFormat::Scc => Some(Box::new(scc::SccParser)),
        CaptionFormat::EbuStl => Some(Box::new(stl::EbuStlParser)),
        CaptionFormat::SpruceStl => Some(Box::new(stl::SpruceStlParser)),
        CaptionFormat::ITt => Some(Box::new(itt::IttParser)),
        #[cfg(feature = "cea")]
        CaptionFormat::Cea608 => Some(Box::new(cea608::Cea608Parser)),
        #[cfg(feature = "cea")]
        CaptionFormat::Cea708 => Some(Box::new(cea708::Cea708Parser)),
        #[cfg(feature = "broadcast")]
        CaptionFormat::Teletext => Some(Box::new(teletext::TeletextParser)),
        #[cfg(feature = "broadcast")]
        CaptionFormat::Arib => Some(Box::new(arib::AribParser)),
        #[cfg(feature = "broadcast")]
        CaptionFormat::Dvb => Some(Box::new(dvb::DvbParser)),
        _ => None,
    }
}

/// Get writer for a format
#[must_use]
pub fn get_writer(format: CaptionFormat) -> Option<Box<dyn FormatWriter>> {
    match format {
        CaptionFormat::Srt => Some(Box::new(srt::SrtWriter)),
        CaptionFormat::WebVtt => Some(Box::new(webvtt::WebVttWriter)),
        CaptionFormat::Ass => Some(Box::new(ass::AssWriter)),
        CaptionFormat::Ssa => Some(Box::new(ssa::SsaWriter)),
        CaptionFormat::Ttml | CaptionFormat::Dfxp => Some(Box::new(ttml::TtmlWriter)),
        CaptionFormat::Scc => Some(Box::new(scc::SccWriter)),
        CaptionFormat::EbuStl => Some(Box::new(stl::EbuStlWriter)),
        CaptionFormat::ITt => Some(Box::new(itt::IttWriter)),
        #[cfg(feature = "cea")]
        CaptionFormat::Cea608 => Some(Box::new(cea608::Cea608Writer)),
        #[cfg(feature = "cea")]
        CaptionFormat::Cea708 => Some(Box::new(cea708::Cea708Writer)),
        _ => None,
    }
}

/// Auto-detect caption format from file content
#[must_use]
pub fn detect_format(data: &[u8]) -> Option<CaptionFormat> {
    let text = std::str::from_utf8(data).ok()?;
    let trimmed = text.trim_start();

    // Check for format signatures
    if trimmed.starts_with("WEBVTT") {
        Some(CaptionFormat::WebVtt)
    } else if trimmed.starts_with("[Script Info]") {
        Some(CaptionFormat::Ass)
    } else if trimmed.starts_with("[Script") {
        Some(CaptionFormat::Ssa)
    } else if trimmed.starts_with("<?xml") && text.contains("tt:tt") {
        Some(CaptionFormat::Ttml)
    } else if trimmed.starts_with("Scenarist_SCC") {
        Some(CaptionFormat::Scc)
    } else if trimmed.starts_with("<?xml") && text.contains("tt xmlns") {
        Some(CaptionFormat::ITt)
    } else if is_srt_format(trimmed) {
        Some(CaptionFormat::Srt)
    } else {
        None
    }
}

fn is_srt_format(text: &str) -> bool {
    // SRT format starts with a number
    let lines: Vec<&str> = text.lines().take(3).collect();
    if lines.len() >= 3 {
        // First line should be a number
        if lines[0].trim().parse::<u32>().is_ok() {
            // Second line should contain timestamp
            return lines[1].contains("-->");
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_srt() {
        let srt = b"1\n00:00:01,000 --> 00:00:03,000\nTest\n\n";
        assert_eq!(detect_format(srt), Some(CaptionFormat::Srt));
    }

    #[test]
    fn test_detect_webvtt() {
        let vtt = b"WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nTest\n\n";
        assert_eq!(detect_format(vtt), Some(CaptionFormat::WebVtt));
    }

    #[test]
    fn test_detect_ass() {
        let ass = b"[Script Info]\nTitle: Test\n";
        assert_eq!(detect_format(ass), Some(CaptionFormat::Ass));
    }
}
