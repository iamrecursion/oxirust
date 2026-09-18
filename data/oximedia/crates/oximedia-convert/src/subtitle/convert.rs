// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Subtitle format conversion using the oximedia-subtitle parsing pipeline.

use crate::{ConversionError, Result};
use std::path::Path;

/// Converter for subtitle formats.
#[derive(Debug, Clone)]
pub struct SubtitleConverter {
    encoding: String,
}

impl SubtitleConverter {
    /// Create a new subtitle converter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoding: "UTF-8".to_string(),
        }
    }

    /// Set the text encoding (informational; UTF-8 is always used for output).
    pub fn with_encoding<S: Into<String>>(mut self, encoding: S) -> Self {
        self.encoding = encoding.into();
        self
    }

    /// Convert subtitle format.
    ///
    /// Reads the input file, parses it according to the detected source format,
    /// and serialises the result in `target_format`. Returns the number of
    /// subtitle events written.
    ///
    /// # Errors
    ///
    /// Returns [`ConversionError::InvalidInput`] when the input file is absent,
    /// [`ConversionError::UnsupportedFormat`] for formats without a parser, and
    /// [`ConversionError::Io`] on read/write failures.
    pub async fn convert<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        target_format: SubtitleFormat,
    ) -> Result<usize> {
        let input = input.as_ref();
        let output = output.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        let source_format = self.detect_format(input)?;

        // Fast path: identical format — copy bytes unchanged.
        if source_format == target_format {
            std::fs::copy(input, output).map_err(ConversionError::Io)?;
            // Count events without re-parsing.
            let text = std::fs::read_to_string(input).map_err(ConversionError::Io)?;
            let count = count_srt_events_approx(&text);
            return Ok(count);
        }

        let text = std::fs::read_to_string(input).map_err(ConversionError::Io)?;

        // Parse to canonical `Vec<Subtitle>` via the appropriate high-level parser.
        use oximedia_subtitle::{AssParser, SrtParser, WebVttParser};

        let subtitles: Vec<oximedia_subtitle::Subtitle> = match source_format {
            SubtitleFormat::Srt | SubtitleFormat::SubRip => SrtParser::parse(&text)
                .map_err(|e| ConversionError::UnsupportedFormat(e.to_string()))?,
            SubtitleFormat::WebVtt => WebVttParser::parse(&text)
                .map_err(|e| ConversionError::UnsupportedFormat(e.to_string()))?,
            SubtitleFormat::Ass => AssParser::parse(&text)
                .map_err(|e| ConversionError::UnsupportedFormat(e.to_string()))?,
            SubtitleFormat::Ttml => {
                // Use the TTML v2 parser available in oximedia-subtitle.
                use oximedia_subtitle::{SubtitleEntry as TtmlEntry, TtmlParser};
                let ttml_entries: Vec<TtmlEntry> = TtmlParser::parse_v2(&text)
                    .map_err(|e| ConversionError::UnsupportedFormat(e.to_string()))?;
                // Convert TTML entries → generic Subtitle.
                ttml_entries
                    .into_iter()
                    .map(|e| oximedia_subtitle::Subtitle::new(e.start_ms, e.end_ms, e.text.clone()))
                    .collect()
            }
            SubtitleFormat::Sbv => {
                return Err(ConversionError::UnsupportedFormat(
                    "SBV source format is not supported for conversion".to_string(),
                ));
            }
        };

        let event_count = subtitles.len();

        // Serialise to target format.
        let serialised = match target_format {
            SubtitleFormat::Srt | SubtitleFormat::SubRip => {
                use oximedia_subtitle::format_convert::{
                    SrtSerializer, SubtitleEntry as ConvEntry,
                };
                let entries: Vec<ConvEntry> = subtitles
                    .iter()
                    .map(|s| {
                        ConvEntry::new(
                            s.start_time.max(0) as u64,
                            s.end_time.max(0) as u64,
                            s.text.clone(),
                        )
                    })
                    .collect();
                SrtSerializer::to_srt(&entries)
            }
            SubtitleFormat::WebVtt => {
                use oximedia_subtitle::format_convert::{
                    SubtitleEntry as ConvEntry, VttSerializer,
                };
                let entries: Vec<ConvEntry> = subtitles
                    .iter()
                    .map(|s| {
                        ConvEntry::new(
                            s.start_time.max(0) as u64,
                            s.end_time.max(0) as u64,
                            s.text.clone(),
                        )
                    })
                    .collect();
                VttSerializer::to_vtt(&entries)
            }
            SubtitleFormat::Ass => {
                // Minimal ASS output — header + events.
                let mut out = String::from(
                    "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
                );
                for s in &subtitles {
                    out.push_str(&format!(
                        "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
                        ms_to_ass_time(s.start_time.max(0) as u64),
                        ms_to_ass_time(s.end_time.max(0) as u64),
                        s.text.replace('\n', "\\N"),
                    ));
                }
                out
            }
            SubtitleFormat::Sbv => {
                return Err(ConversionError::UnsupportedFormat(
                    "SBV target format is not supported for output".to_string(),
                ));
            }
            SubtitleFormat::Ttml => {
                // Minimal TTML output.
                let mut out = String::from(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<tt xml:lang=\"\" xmlns=\"http://www.w3.org/ns/ttml\">\n  <body>\n    <div>\n",
                );
                for s in &subtitles {
                    out.push_str(&format!(
                        "      <p begin=\"{}\" end=\"{}\">{}</p>\n",
                        ms_to_ttml_time(s.start_time.max(0) as u64),
                        ms_to_ttml_time(s.end_time.max(0) as u64),
                        xml_escape(&s.text),
                    ));
                }
                out.push_str("    </div>\n  </body>\n</tt>\n");
                out
            }
        };

        std::fs::write(output, serialised.as_bytes()).map_err(ConversionError::Io)?;
        Ok(event_count)
    }

    /// Detect subtitle format from file extension.
    pub fn detect_format<P: AsRef<Path>>(&self, path: P) -> Result<SubtitleFormat> {
        let path = path.as_ref();

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| ConversionError::FormatDetection("No file extension".to_string()))?;

        match ext.to_lowercase().as_str() {
            "srt" => Ok(SubtitleFormat::Srt),
            "vtt" => Ok(SubtitleFormat::WebVtt),
            "ass" | "ssa" => Ok(SubtitleFormat::Ass),
            "sub" => Ok(SubtitleFormat::SubRip),
            "sbv" => Ok(SubtitleFormat::Sbv),
            "ttml" => Ok(SubtitleFormat::Ttml),
            _ => Err(ConversionError::UnsupportedCodec(format!(
                "Unknown subtitle format: {ext}"
            ))),
        }
    }

    /// Convert SRT to `WebVTT`.
    pub async fn srt_to_webvtt<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<usize> {
        self.convert(input, output, SubtitleFormat::WebVtt).await
    }

    /// Convert `WebVTT` to SRT.
    pub async fn webvtt_to_srt<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<usize> {
        self.convert(input, output, SubtitleFormat::Srt).await
    }

    /// Convert any format to SRT.
    pub async fn to_srt<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<usize> {
        self.convert(input, output, SubtitleFormat::Srt).await
    }

    /// Convert any format to `WebVTT`.
    pub async fn to_webvtt<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<usize> {
        self.convert(input, output, SubtitleFormat::WebVtt).await
    }
}

impl Default for SubtitleConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Format helpers ──────────────────────────────────────────────────────────

/// Format milliseconds as ASS time string `H:MM:SS.cc`.
fn ms_to_ass_time(ms: u64) -> String {
    let centisecs = (ms % 1000) / 10;
    let total_secs = ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;
    format!("{hours}:{mins:02}:{secs:02}.{centisecs:02}")
}

/// Format milliseconds as TTML time string `HH:MM:SS.mmm`.
fn ms_to_ttml_time(ms: u64) -> String {
    let millis = ms % 1000;
    let total_secs = ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;
    format!("{hours:02}:{mins:02}:{secs:02}.{millis:03}")
}

/// Escape XML special characters in subtitle text.
fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Quick approximate event counter for already-parsed SRT text (avoids a
/// second full parse when source == target on the fast-copy path).
fn count_srt_events_approx(text: &str) -> usize {
    // Count non-empty blocks separated by blank lines.
    text.split("\n\n")
        .map(str::trim)
        .filter(|b| !b.is_empty())
        .count()
}

/// Supported subtitle formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    /// `SubRip` (.srt)
    Srt,
    /// `WebVTT` (.vtt)
    WebVtt,
    /// Advanced `SubStation` Alpha (.ass / .ssa)
    Ass,
    /// `SubRip` (.sub) — treated the same as SRT
    SubRip,
    /// `YouTube` SBV (.sbv)
    Sbv,
    /// Timed Text Markup Language (.ttml)
    Ttml,
}

impl SubtitleFormat {
    /// Get the file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::WebVtt => "vtt",
            Self::Ass => "ass",
            Self::SubRip => "sub",
            Self::Sbv => "sbv",
            Self::Ttml => "ttml",
        }
    }

    /// Get the MIME type for this format.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Srt => "text/srt",
            Self::WebVtt => "text/vtt",
            Self::Ass => "text/x-ssa",
            Self::SubRip => "text/plain",
            Self::Sbv => "text/sbv",
            Self::Ttml => "application/ttml+xml",
        }
    }

    /// Get the format name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Srt => "SubRip",
            Self::WebVtt => "WebVTT",
            Self::Ass => "Advanced SubStation Alpha",
            Self::SubRip => "SubRip",
            Self::Sbv => "YouTube SBV",
            Self::Ttml => "TTML",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = SubtitleConverter::new();
        assert_eq!(converter.encoding, "UTF-8");
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(SubtitleFormat::Srt.extension(), "srt");
        assert_eq!(SubtitleFormat::WebVtt.extension(), "vtt");
        assert_eq!(SubtitleFormat::Ass.extension(), "ass");
    }

    #[test]
    fn test_format_mime_type() {
        assert_eq!(SubtitleFormat::Srt.mime_type(), "text/srt");
        assert_eq!(SubtitleFormat::WebVtt.mime_type(), "text/vtt");
    }

    #[test]
    fn test_format_name() {
        assert_eq!(SubtitleFormat::Srt.name(), "SubRip");
        assert_eq!(SubtitleFormat::WebVtt.name(), "WebVTT");
    }

    #[test]
    fn test_detect_format() {
        let converter = SubtitleConverter::new();

        let path = Path::new("test.srt");
        assert_eq!(converter.detect_format(path).unwrap(), SubtitleFormat::Srt);

        let path = Path::new("test.vtt");
        assert_eq!(
            converter.detect_format(path).unwrap(),
            SubtitleFormat::WebVtt
        );

        let path = Path::new("test.ass");
        assert_eq!(converter.detect_format(path).unwrap(), SubtitleFormat::Ass);
    }

    #[test]
    fn test_ms_to_ass_time() {
        // 1h 2m 3s 456ms → 1:02:03.45
        let ms = 3600_000 + 2 * 60_000 + 3_000 + 456;
        let s = ms_to_ass_time(ms);
        assert_eq!(s, "1:02:03.45");
    }

    #[test]
    fn test_ms_to_ttml_time() {
        let ms = 3600_000 + 2 * 60_000 + 3_000 + 789;
        assert_eq!(ms_to_ttml_time(ms), "01:02:03.789");
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("Hello & <World>"), "Hello &amp; &lt;World&gt;");
    }

    #[test]
    fn test_count_srt_events_approx() {
        let srt = "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n2\n00:00:05,000 --> 00:00:08,000\nWorld\n\n";
        assert_eq!(count_srt_events_approx(srt), 2);
    }

    #[tokio::test]
    async fn test_convert_missing_input_errors() {
        let converter = SubtitleConverter::new();
        let input = std::env::temp_dir().join("__oximedia_nonexistent_subtitle__.srt");
        let output = std::env::temp_dir().join("__oximedia_nonexistent_out__.vtt");
        let result = converter
            .convert(&input, &output, SubtitleFormat::WebVtt)
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_convert_srt_to_vtt_roundtrip() {
        use std::io::Write;
        let tmp_dir = std::env::temp_dir();
        let srt_path = tmp_dir.join("oximedia_convert_test_sub.srt");
        let vtt_path = tmp_dir.join("oximedia_convert_test_sub.vtt");

        let srt_content = "1\n00:00:01,000 --> 00:00:04,000\nHello, world!\n\n2\n00:00:05,000 --> 00:00:08,000\nSecond line.\n\n";
        {
            let mut f = std::fs::File::create(&srt_path).expect("create tmp srt");
            f.write_all(srt_content.as_bytes()).expect("write srt");
        }

        let converter = SubtitleConverter::new();
        let count = converter
            .convert(&srt_path, &vtt_path, SubtitleFormat::WebVtt)
            .await
            .expect("srt->vtt conversion should succeed");

        assert_eq!(count, 2, "expected 2 subtitle events");
        let vtt_text = std::fs::read_to_string(&vtt_path).expect("read vtt");
        assert!(vtt_text.starts_with("WEBVTT"), "VTT must start with WEBVTT");
        assert!(vtt_text.contains("Hello, world!"));
        assert!(vtt_text.contains("Second line."));

        // Clean up
        let _ = std::fs::remove_file(&srt_path);
        let _ = std::fs::remove_file(&vtt_path);
    }

    #[tokio::test]
    async fn test_convert_srt_to_ass() {
        use std::io::Write;
        let tmp_dir = std::env::temp_dir();
        let srt_path = tmp_dir.join("oximedia_convert_test_ass_in.srt");
        let ass_path = tmp_dir.join("oximedia_convert_test_ass_out.ass");

        let srt_content = "1\n00:00:01,000 --> 00:00:04,000\nHello ASS!\n\n";
        {
            let mut f = std::fs::File::create(&srt_path).expect("create tmp srt");
            f.write_all(srt_content.as_bytes()).expect("write srt");
        }

        let converter = SubtitleConverter::new();
        let count = converter
            .convert(&srt_path, &ass_path, SubtitleFormat::Ass)
            .await
            .expect("srt->ass conversion should succeed");

        assert_eq!(count, 1);
        let ass_text = std::fs::read_to_string(&ass_path).expect("read ass");
        assert!(ass_text.contains("[Script Info]"));
        assert!(ass_text.contains("Hello ASS!"));

        let _ = std::fs::remove_file(&srt_path);
        let _ = std::fs::remove_file(&ass_path);
    }
}
