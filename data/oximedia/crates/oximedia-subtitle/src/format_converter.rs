//! Multi-format subtitle converter with rich styling and document model.
//!
//! Supports parsing and serializing SRT, VTT, ASS/SSA, TTML, SBV, LRC formats,
//! with a unified `SubtitleDocument` intermediate representation.

use std::collections::HashMap;

// ── Public Types ─────────────────────────────────────────────────────────────

/// Identifies the on-disk / in-memory subtitle format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubtitleFormat {
    /// SubRip `.srt`
    Srt,
    /// WebVTT `.vtt`
    Vtt,
    /// Advanced SubStation Alpha `.ass`
    Ass,
    /// SubStation Alpha `.ssa`
    Ssa,
    /// Timed Text Markup Language `.ttml` / `.xml`
    Ttml,
    /// Distribution Format Exchange Profile (TTML subset) `.dfxp`
    Dfxp,
    /// Scenarist Closed Captions `.scc`
    Scc,
    /// SubViewer / YouTube `.sbv`
    Sbv,
    /// LRC lyrics format `.lrc`
    Lrc,
}

impl SubtitleFormat {
    /// Returns the canonical file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Ass => "ass",
            Self::Ssa => "ssa",
            Self::Ttml => "ttml",
            Self::Dfxp => "dfxp",
            Self::Scc => "scc",
            Self::Sbv => "sbv",
            Self::Lrc => "lrc",
        }
    }

    /// Returns true if the format natively supports rich styling.
    #[must_use]
    pub fn supports_styling(&self) -> bool {
        matches!(
            self,
            Self::Ass | Self::Ssa | Self::Vtt | Self::Ttml | Self::Dfxp
        )
    }
}

/// Screen position / alignment of a subtitle cue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitlePosition {
    /// Top-left corner.
    TopLeft,
    /// Top-center.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Middle-left.
    MiddleLeft,
    /// Middle-center.
    MiddleCenter,
    /// Middle-right.
    MiddleRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-center (default for most subtitle formats).
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
}

impl SubtitlePosition {
    /// Map from an ASS numpad alignment value (1-9) to a `SubtitlePosition`.
    #[must_use]
    pub fn from_ass_alignment(n: u8) -> Self {
        match n {
            1 => Self::BottomLeft,
            2 => Self::BottomCenter,
            3 => Self::BottomRight,
            4 => Self::MiddleLeft,
            5 => Self::MiddleCenter,
            6 => Self::MiddleRight,
            7 => Self::TopLeft,
            8 => Self::TopCenter,
            9 => Self::TopRight,
            _ => Self::BottomCenter,
        }
    }

    /// Return the ASS numpad alignment value (1-9) for this position.
    #[must_use]
    pub fn to_ass_alignment(self) -> u8 {
        match self {
            Self::BottomLeft => 1,
            Self::BottomCenter => 2,
            Self::BottomRight => 3,
            Self::MiddleLeft => 4,
            Self::MiddleCenter => 5,
            Self::MiddleRight => 6,
            Self::TopLeft => 7,
            Self::TopCenter => 8,
            Self::TopRight => 9,
        }
    }
}

/// Per-cue inline styling metadata.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SubtitleStyle {
    /// Italic text.
    pub italic: bool,
    /// Bold text.
    pub bold: bool,
    /// Underline text.
    pub underline: bool,
    /// Foreground colour (R, G, B).
    pub color: Option<(u8, u8, u8)>,
    /// Cue position override.
    pub position: Option<SubtitlePosition>,
    /// Font size in points.
    pub font_size: Option<f32>,
}

/// A single subtitle cue with timing, text and optional per-cue styling.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleEntry {
    /// Sequence index (1-based in SRT; 0 = unset).
    pub index: u32,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Plain text (markup stripped).
    pub text: String,
    /// Optional per-cue style information.
    pub style: Option<SubtitleStyle>,
}

impl SubtitleEntry {
    /// Create a bare entry without styling.
    #[must_use]
    pub fn new(index: u32, start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self {
            index,
            start_ms,
            end_ms,
            text: text.into(),
            style: None,
        }
    }

    /// Duration of this cue in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` if `pts_ms` falls within `[start_ms, end_ms)`.
    #[must_use]
    pub fn is_active_at(&self, pts_ms: u64) -> bool {
        pts_ms >= self.start_ms && pts_ms < self.end_ms
    }
}

/// An ASS/SSA style definition.
#[derive(Debug, Clone, PartialEq)]
pub struct AssStyle {
    /// Style name.
    pub name: String,
    /// Font family.
    pub fontname: String,
    /// Font size in points.
    pub fontsize: f32,
    /// Primary colour (R, G, B, A).
    pub primary_color: (u8, u8, u8, u8),
    /// Bold flag.
    pub bold: bool,
    /// Italic flag.
    pub italic: bool,
    /// Numpad alignment (1-9).
    pub alignment: u8,
    /// Left margin in pixels.
    pub margin_l: u32,
    /// Right margin in pixels.
    pub margin_r: u32,
    /// Vertical margin in pixels.
    pub margin_v: u32,
}

impl Default for AssStyle {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            fontname: "Arial".to_string(),
            fontsize: 48.0,
            primary_color: (255, 255, 255, 255),
            bold: false,
            italic: false,
            alignment: 2,
            margin_l: 10,
            margin_r: 10,
            margin_v: 10,
        }
    }
}

/// A parsed subtitle document: a list of cues plus format metadata.
#[derive(Debug, Clone)]
pub struct SubtitleDocument {
    /// The format this document was originally parsed from.
    pub format: SubtitleFormat,
    /// All subtitle cues in presentation order.
    pub entries: Vec<SubtitleEntry>,
    /// Key-value metadata (e.g. `ScriptType`, `PlayResX`…).
    pub metadata: HashMap<String, String>,
    /// ASS style definitions (empty for non-ASS formats).
    pub styles: Vec<AssStyle>,
}

impl SubtitleDocument {
    /// Create an empty document with the given format.
    #[must_use]
    pub fn empty(format: SubtitleFormat) -> Self {
        Self {
            format,
            entries: Vec::new(),
            metadata: HashMap::new(),
            styles: Vec::new(),
        }
    }

    // ── Parsers ───────────────────────────────────────────────────────────

    /// Parse a SubRip (SRT) document.
    ///
    /// # Errors
    ///
    /// Returns a `String` describing the first parse error encountered.
    pub fn parse_srt(text: &str) -> Result<Self, String> {
        let mut entries = Vec::new();
        let blocks: Vec<&str> = text
            .split("\n\n")
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .collect();

        for block in &blocks {
            let lines: Vec<&str> = block.lines().collect();
            if lines.len() < 2 {
                continue;
            }

            // Sequence number
            let index: u32 = lines[0]
                .trim()
                .parse()
                .map_err(|_| format!("Expected sequence number, got {:?}", lines[0]))?;

            // Timing line
            let timing = lines[1];
            let parts: Vec<&str> = timing.splitn(2, "-->").collect();
            if parts.len() != 2 {
                return Err(format!("Invalid timing line: {timing:?}"));
            }
            let start_ms = parse_srt_timestamp(parts[0].trim())
                .ok_or_else(|| format!("Bad start timestamp: {:?}", parts[0]))?;
            let end_ms = parse_srt_timestamp(parts[1].trim())
                .ok_or_else(|| format!("Bad end timestamp: {:?}", parts[1]))?;

            // Text lines — may contain inline HTML tags
            let raw_text = lines[2..].join("\n");
            let (plain, style) = strip_html_tags(&raw_text);

            entries.push(SubtitleEntry {
                index,
                start_ms,
                end_ms,
                text: plain,
                style: if style == SubtitleStyle::default() {
                    None
                } else {
                    Some(style)
                },
            });
        }

        Ok(Self {
            format: SubtitleFormat::Srt,
            entries,
            metadata: HashMap::new(),
            styles: Vec::new(),
        })
    }

    /// Parse a WebVTT document.
    ///
    /// # Errors
    ///
    /// Returns a `String` describing the first parse error encountered.
    pub fn parse_vtt(text: &str) -> Result<Self, String> {
        // Must start with WEBVTT
        if !text.trim_start().starts_with("WEBVTT") {
            return Err("Missing WEBVTT header".to_string());
        }

        let mut entries = Vec::new();
        let mut index: u32 = 1;
        let mut metadata = HashMap::new();

        // Split on double newlines
        let blocks: Vec<&str> = text
            .split("\n\n")
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .collect();

        for block in &blocks {
            // Skip the WEBVTT header block
            if block.trim_start().starts_with("WEBVTT") {
                // Parse header metadata: "Key: Value" lines after the first line
                for line in block.lines().skip(1) {
                    if let Some((k, v)) = line.split_once(':') {
                        metadata.insert(k.trim().to_string(), v.trim().to_string());
                    }
                }
                continue;
            }

            // Skip NOTE blocks
            if block.starts_with("NOTE") {
                continue;
            }

            let lines: Vec<&str> = block.lines().collect();
            if lines.is_empty() {
                continue;
            }

            // Optional cue identifier: first line if it doesn't contain "-->"
            let (timing_line_idx, _cue_id) = if !lines[0].contains("-->") {
                (1, Some(lines[0]))
            } else {
                (0, None)
            };

            if timing_line_idx >= lines.len() {
                continue;
            }

            let timing_full = lines[timing_line_idx];
            // Cue settings follow the timestamp pair after whitespace
            let timing_and_settings: Vec<&str> = timing_full.splitn(2, "-->").collect();
            if timing_and_settings.len() != 2 {
                continue;
            }
            let start_ms = parse_vtt_timestamp(timing_and_settings[0].trim())
                .ok_or_else(|| format!("Bad VTT start: {:?}", timing_and_settings[0]))?;

            // End time may be followed by cue settings (position:xx% etc.)
            let end_part = timing_and_settings[1].trim();
            let end_str = end_part.split_whitespace().next().unwrap_or(end_part);
            let end_ms =
                parse_vtt_timestamp(end_str).ok_or_else(|| format!("Bad VTT end: {end_str:?}"))?;

            // Parse cue settings for position
            let position = parse_vtt_position(end_part);

            let raw_text = lines[timing_line_idx + 1..].join("\n");
            let (plain, mut style) = strip_html_tags(&raw_text);
            if position.is_some() {
                style.position = position;
            }

            entries.push(SubtitleEntry {
                index,
                start_ms,
                end_ms,
                text: plain,
                style: if style == SubtitleStyle::default() {
                    None
                } else {
                    Some(style)
                },
            });
            index += 1;
        }

        Ok(Self {
            format: SubtitleFormat::Vtt,
            entries,
            metadata,
            styles: Vec::new(),
        })
    }

    /// Parse an ASS / SSA document.
    ///
    /// # Errors
    ///
    /// Returns a `String` describing the first parse error encountered.
    pub fn parse_ass(text: &str) -> Result<Self, String> {
        let mut metadata = HashMap::new();
        let mut styles: Vec<AssStyle> = Vec::new();
        let mut entries = Vec::new();

        let mut section = "";
        let mut style_format: Vec<&str> = Vec::new();
        let mut event_format: Vec<&str> = Vec::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(';') {
                continue;
            }

            if line.starts_with('[') && line.ends_with(']') {
                section = line;
                continue;
            }

            match section {
                "[Script Info]" => {
                    if let Some((k, v)) = line.split_once(':') {
                        metadata.insert(k.trim().to_string(), v.trim().to_string());
                    }
                }
                "[V4 Styles]" | "[V4+ Styles]" => {
                    if line.starts_with("Format:") {
                        let fmt_str = line.trim_start_matches("Format:").trim();
                        style_format = fmt_str.split(',').map(str::trim).collect();
                    } else if line.starts_with("Style:") {
                        let data = line.trim_start_matches("Style:").trim();
                        if let Some(s) = parse_ass_style(data, &style_format) {
                            styles.push(s);
                        }
                    }
                }
                "[Events]" => {
                    if line.starts_with("Format:") {
                        let fmt_str = line.trim_start_matches("Format:").trim();
                        event_format = fmt_str.split(',').map(str::trim).collect();
                    } else if line.starts_with("Dialogue:") {
                        let data = line.trim_start_matches("Dialogue:").trim();
                        if let Some(entry) =
                            parse_ass_dialogue(data, &event_format, entries.len() as u32 + 1)
                        {
                            entries.push(entry);
                        }
                    }
                }
                _ => {}
            }
        }

        // Sort by start time
        entries.sort_by_key(|e| e.start_ms);

        // Re-index
        for (i, e) in entries.iter_mut().enumerate() {
            e.index = i as u32 + 1;
        }

        Ok(Self {
            format: SubtitleFormat::Ass,
            entries,
            metadata,
            styles,
        })
    }

    /// Parse a SubViewer / YouTube `.sbv` document.
    ///
    /// # Errors
    ///
    /// Returns a `String` describing the first parse error encountered.
    pub fn parse_sbv(text: &str) -> Result<Self, String> {
        let mut entries = Vec::new();
        let blocks: Vec<&str> = text
            .split("\n\n")
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .collect();

        for block in &blocks {
            let lines: Vec<&str> = block.lines().collect();
            if lines.is_empty() {
                continue;
            }

            // First line: H:MM:SS.mmm,H:MM:SS.mmm
            let timing = lines[0];
            let parts: Vec<&str> = timing.splitn(2, ',').collect();
            if parts.len() != 2 {
                return Err(format!("Invalid SBV timing: {timing:?}"));
            }
            let start_ms = parse_sbv_timestamp(parts[0].trim())
                .ok_or_else(|| format!("Bad SBV start: {:?}", parts[0]))?;
            let end_ms = parse_sbv_timestamp(parts[1].trim())
                .ok_or_else(|| format!("Bad SBV end: {:?}", parts[1]))?;

            let text_content = lines[1..].join("\n");

            entries.push(SubtitleEntry {
                index: entries.len() as u32 + 1,
                start_ms,
                end_ms,
                text: text_content,
                style: None,
            });
        }

        Ok(Self {
            format: SubtitleFormat::Sbv,
            entries,
            metadata: HashMap::new(),
            styles: Vec::new(),
        })
    }

    /// Parse an LRC lyrics document.
    ///
    /// # Errors
    ///
    /// Returns a `String` describing the first parse error encountered.
    pub fn parse_lrc(text: &str) -> Result<Self, String> {
        let mut entries = Vec::new();
        let mut metadata = HashMap::new();
        let mut timed_lines: Vec<(u64, String)> = Vec::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Tags: [MM:SS.xx]text  or  [key:value]
            if line.starts_with('[') {
                if let Some(end) = line.find(']') {
                    let tag_content = &line[1..end];
                    let after = &line[end + 1..];

                    if let Some(ts) = parse_lrc_timestamp(tag_content) {
                        timed_lines.push((ts, after.to_string()));
                    } else if let Some((k, v)) = tag_content.split_once(':') {
                        metadata.insert(k.trim().to_string(), v.trim().to_string());
                    }
                }
            }
        }

        // Sort by timestamp
        timed_lines.sort_by_key(|(ts, _)| *ts);

        // Build entries: each line lasts until the next one (or +5s for the last)
        let n = timed_lines.len();
        for i in 0..n {
            let (start_ms, ref lyric) = timed_lines[i];
            let end_ms = if i + 1 < n {
                timed_lines[i + 1].0
            } else {
                start_ms + 5000
            };

            entries.push(SubtitleEntry {
                index: i as u32 + 1,
                start_ms,
                end_ms,
                text: lyric.clone(),
                style: None,
            });
        }

        Ok(Self {
            format: SubtitleFormat::Lrc,
            entries,
            metadata,
            styles: Vec::new(),
        })
    }

    // ── Serializers ───────────────────────────────────────────────────────

    /// Serialize to SubRip (SRT) format.
    #[must_use]
    pub fn to_srt(&self) -> String {
        let mut out = String::new();
        for (i, entry) in self.entries.iter().enumerate() {
            let n = i + 1;
            let start = format_ms_srt(entry.start_ms);
            let end = format_ms_srt(entry.end_ms);
            let text = apply_srt_tags(entry);
            out.push_str(&format!("{n}\n{start} --> {end}\n{text}\n\n"));
        }
        out
    }

    /// Serialize to WebVTT format.
    #[must_use]
    pub fn to_vtt(&self) -> String {
        let mut out = String::from("WEBVTT\n\n");
        for (i, entry) in self.entries.iter().enumerate() {
            let n = i + 1;
            let start = format_ms_vtt(entry.start_ms);
            let end = format_ms_vtt(entry.end_ms);
            let position_cue = entry
                .style
                .as_ref()
                .and_then(|s| s.position)
                .map(|p| format!(" align:{}", vtt_position_label(p)))
                .unwrap_or_default();
            out.push_str(&format!(
                "{n}\n{start} --> {end}{position_cue}\n{}\n\n",
                entry.text
            ));
        }
        out
    }

    /// Serialize to ASS format.
    #[must_use]
    pub fn to_ass(&self) -> String {
        let default_style = self.styles.first().cloned().unwrap_or_default();
        let mut out = String::new();

        out.push_str("[Script Info]\n");
        out.push_str("ScriptType: v4.00+\n");
        out.push_str("PlayResX: 1920\n");
        out.push_str("PlayResY: 1080\n");
        for (k, v) in &self.metadata {
            if k != "ScriptType" && k != "PlayResX" && k != "PlayResY" {
                out.push_str(&format!("{k}: {v}\n"));
            }
        }
        out.push('\n');

        out.push_str("[V4+ Styles]\n");
        out.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n");

        let styles_to_write = if self.styles.is_empty() {
            vec![AssStyle::default()]
        } else {
            self.styles.clone()
        };

        for s in &styles_to_write {
            let (r, g, b, a) = s.primary_color;
            let ass_color = format!("&H{:02X}{:02X}{:02X}{:02X}", 255 - a, b, g, r);
            out.push_str(&format!(
                "Style: {},{},{:.0},{},&H000000FF,&H00000000,&H00000000,{},{},0,0,100,100,0,0,1,2,2,{},{},{},{},1\n",
                s.name, s.fontname, s.fontsize, ass_color,
                if s.bold { "-1" } else { "0" },
                if s.italic { "-1" } else { "0" },
                s.alignment, s.margin_l, s.margin_r, s.margin_v
            ));
        }
        out.push('\n');

        out.push_str("[Events]\n");
        out.push_str(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
        );
        for entry in &self.entries {
            let start = format_ms_ass(entry.start_ms);
            let end = format_ms_ass(entry.end_ms);
            let style_name = entry
                .style
                .as_ref()
                .map(|_| default_style.name.as_str())
                .unwrap_or("Default");
            let text = apply_ass_override_tags(entry);
            out.push_str(&format!(
                "Dialogue: 0,{start},{end},{style_name},,0,0,0,,{text}\n"
            ));
        }

        out
    }

    /// Serialize to TTML (Timed Text Markup Language) format.
    #[must_use]
    pub fn to_ttml(&self) -> String {
        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<tt xml:lang=\"\" xmlns=\"http://www.w3.org/ns/ttml\">\n");
        out.push_str("  <body>\n");
        out.push_str("    <div>\n");

        for entry in &self.entries {
            let begin = format_ms_ttml(entry.start_ms);
            let end = format_ms_ttml(entry.end_ms);
            // Escape XML special chars
            let text = xml_escape(&entry.text);
            out.push_str(&format!(
                "      <p begin=\"{begin}\" end=\"{end}\">{text}</p>\n"
            ));
        }

        out.push_str("    </div>\n");
        out.push_str("  </body>\n");
        out.push_str("</tt>\n");
        out
    }

    /// Serialize to SubViewer / YouTube `.sbv` format.
    #[must_use]
    pub fn to_sbv(&self) -> String {
        let mut out = String::new();
        for entry in &self.entries {
            let start = format_ms_sbv(entry.start_ms);
            let end = format_ms_sbv(entry.end_ms);
            out.push_str(&format!("{start},{end}\n{}\n\n", entry.text));
        }
        out
    }
}

// ── SubtitleConverter ────────────────────────────────────────────────────────

/// High-level converter that transforms between subtitle formats and adjusts timing.
pub struct SubtitleConverter;

impl SubtitleConverter {
    /// Convert a document to the target format.
    ///
    /// The returned document has `format` set to `target` and entries with
    /// styling stripped when the target format does not support it.
    #[must_use]
    pub fn convert(doc: &SubtitleDocument, target: SubtitleFormat) -> SubtitleDocument {
        let mut new_entries: Vec<SubtitleEntry> = doc
            .entries
            .iter()
            .map(|e| {
                let style = if target.supports_styling() {
                    e.style.clone()
                } else {
                    None
                };
                SubtitleEntry {
                    index: e.index,
                    start_ms: e.start_ms,
                    end_ms: e.end_ms,
                    text: e.text.clone(),
                    style,
                }
            })
            .collect();

        // Re-number
        for (i, e) in new_entries.iter_mut().enumerate() {
            e.index = i as u32 + 1;
        }

        SubtitleDocument {
            format: target,
            entries: new_entries,
            metadata: doc.metadata.clone(),
            styles: doc.styles.clone(),
        }
    }

    /// Shift all timestamps by `offset_ms` milliseconds.
    ///
    /// Timestamps that would go below zero are clamped to zero.
    pub fn shift_timing(doc: &mut SubtitleDocument, offset_ms: i64) {
        for entry in &mut doc.entries {
            entry.start_ms = shift_ts(entry.start_ms, offset_ms);
            entry.end_ms = shift_ts(entry.end_ms, offset_ms);
        }
    }

    /// Scale all timestamps by `factor` (e.g. 23.976/25.0 for PAL→NTSC).
    pub fn scale_timing(doc: &mut SubtitleDocument, factor: f64) {
        for entry in &mut doc.entries {
            entry.start_ms = (entry.start_ms as f64 * factor).round() as u64;
            entry.end_ms = (entry.end_ms as f64 * factor).round() as u64;
        }
    }

    /// Merge two subtitle documents into one, sorted by start time.
    ///
    /// Overlapping entries are kept as-is (no re-timing); the caller should
    /// use `shift_timing` beforehand if the tracks have different origins.
    /// The resulting document adopts the format of `a`.
    #[must_use]
    pub fn merge(a: &SubtitleDocument, b: &SubtitleDocument) -> SubtitleDocument {
        let mut entries: Vec<SubtitleEntry> =
            a.entries.iter().chain(b.entries.iter()).cloned().collect();

        entries.sort_by(|x, y| x.start_ms.cmp(&y.start_ms).then(x.end_ms.cmp(&y.end_ms)));

        // Resolve exact-same-time overlaps by appending to previous entry text
        let mut resolved: Vec<SubtitleEntry> = Vec::with_capacity(entries.len());
        for entry in entries {
            if let Some(prev) = resolved.last_mut() {
                if prev.start_ms == entry.start_ms && prev.end_ms == entry.end_ms {
                    prev.text.push('\n');
                    prev.text.push_str(&entry.text);
                    continue;
                }
            }
            resolved.push(entry);
        }

        // Re-index
        for (i, e) in resolved.iter_mut().enumerate() {
            e.index = i as u32 + 1;
        }

        let mut metadata = a.metadata.clone();
        for (k, v) in &b.metadata {
            metadata.entry(k.clone()).or_insert_with(|| v.clone());
        }

        let mut styles = a.styles.clone();
        for s in &b.styles {
            if !styles.iter().any(|existing| existing.name == s.name) {
                styles.push(s.clone());
            }
        }

        SubtitleDocument {
            format: a.format,
            entries: resolved,
            metadata,
            styles,
        }
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Apply offset to a u64 timestamp, clamping to 0.
fn shift_ts(ts: u64, offset_ms: i64) -> u64 {
    let result = ts as i64 + offset_ms;
    result.max(0) as u64
}

/// Parse `HH:MM:SS,mmm` SRT timestamp → ms.
fn parse_srt_timestamp(s: &str) -> Option<u64> {
    let s = s.trim();
    let (time_part, millis_str) = s.split_once(',')?;
    let millis: u64 = millis_str.trim().parse().ok()?;
    let parts: Vec<&str> = time_part.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: u64 = parts[0].trim().parse().ok()?;
    let m: u64 = parts[1].trim().parse().ok()?;
    let sec: u64 = parts[2].trim().parse().ok()?;
    Some((h * 3600 + m * 60 + sec) * 1000 + millis)
}

/// Parse `HH:MM:SS.mmm` or `MM:SS.mmm` VTT timestamp → ms.
fn parse_vtt_timestamp(s: &str) -> Option<u64> {
    let s = s.trim();
    let (time_part, millis_str) = s.split_once('.')?;
    let millis: u64 = millis_str
        .trim_end_matches(|c: char| !c.is_ascii_digit())
        .parse()
        .ok()?;
    let parts: Vec<&str> = time_part.split(':').collect();
    let (h, m, sec) = match parts.len() {
        3 => {
            let h: u64 = parts[0].trim().parse().ok()?;
            let m: u64 = parts[1].trim().parse().ok()?;
            let sec: u64 = parts[2].trim().parse().ok()?;
            (h, m, sec)
        }
        2 => {
            let m: u64 = parts[0].trim().parse().ok()?;
            let sec: u64 = parts[1].trim().parse().ok()?;
            (0, m, sec)
        }
        _ => return None,
    };
    Some((h * 3600 + m * 60 + sec) * 1000 + millis)
}

/// Parse ASS timestamp `H:MM:SS.cc` (centiseconds) → ms.
fn parse_ass_timestamp(s: &str) -> Option<u64> {
    let s = s.trim();
    let (time_part, cs_str) = s.split_once('.')?;
    let cs: u64 = cs_str.parse().ok()?;
    let parts: Vec<&str> = time_part.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: u64 = parts[0].parse().ok()?;
    let m: u64 = parts[1].parse().ok()?;
    let sec: u64 = parts[2].parse().ok()?;
    Some((h * 3600 + m * 60 + sec) * 1000 + cs * 10)
}

/// Parse SBV timestamp `H:MM:SS.mmm` → ms.
fn parse_sbv_timestamp(s: &str) -> Option<u64> {
    parse_vtt_timestamp(s)
}

/// Parse LRC timestamp `MM:SS.xx` → ms.
fn parse_lrc_timestamp(s: &str) -> Option<u64> {
    // Format: MM:SS.xx  (xx = hundredths)
    let (min_str, rest) = s.split_once(':')?;
    let mins: u64 = min_str.trim().parse().ok()?;
    let (sec_str, frac_str) = rest.split_once('.')?;
    let secs: u64 = sec_str.trim().parse().ok()?;
    let frac: u64 = frac_str.trim().parse().ok()?;
    // frac is typically 2 digits = hundredths → *10 for ms
    Some((mins * 60 + secs) * 1000 + frac * 10)
}

/// Parse a VTT cue-settings string and extract a `SubtitlePosition` if present.
fn parse_vtt_position(settings: &str) -> Option<SubtitlePosition> {
    // Look for "line:" or "align:" cue settings
    for part in settings.split_whitespace() {
        if let Some(val) = part.strip_prefix("line:") {
            // Negative line values → top; positive → bottom
            if val.starts_with('-') {
                return Some(SubtitlePosition::TopCenter);
            }
        }
        if let Some(val) = part.strip_prefix("align:") {
            return match val {
                "left" => Some(SubtitlePosition::BottomLeft),
                "right" => Some(SubtitlePosition::BottomRight),
                "center" | "middle" => Some(SubtitlePosition::BottomCenter),
                "start" => Some(SubtitlePosition::BottomLeft),
                "end" => Some(SubtitlePosition::BottomRight),
                _ => None,
            };
        }
    }
    None
}

/// Return a VTT cue align label for a position.
fn vtt_position_label(pos: SubtitlePosition) -> &'static str {
    match pos {
        SubtitlePosition::BottomLeft | SubtitlePosition::MiddleLeft | SubtitlePosition::TopLeft => {
            "left"
        }
        SubtitlePosition::BottomRight
        | SubtitlePosition::MiddleRight
        | SubtitlePosition::TopRight => "right",
        _ => "center",
    }
}

/// Strip simple HTML tags (`<i>`, `<b>`, `<u>`, `<font color=...>`) from text,
/// returning the plain text and inferred `SubtitleStyle`.
fn strip_html_tags(text: &str) -> (String, SubtitleStyle) {
    let mut style = SubtitleStyle::default();
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch != '<' {
            out.push(ch);
            continue;
        }

        // Collect tag content
        let mut tag = String::new();
        for inner in chars.by_ref() {
            if inner == '>' {
                break;
            }
            tag.push(inner);
        }

        let tag_lower = tag.to_lowercase();
        let tag_trimmed = tag_lower.trim();

        if tag_trimmed == "i" {
            style.italic = true;
        } else if tag_trimmed == "b" {
            style.bold = true;
        } else if tag_trimmed == "u" {
            style.underline = true;
        } else if tag_trimmed.starts_with("font") {
            // Extract color attribute: color="#rrggbb" or color="name"
            if let Some(color) = extract_font_color(&tag) {
                style.color = Some(color);
            }
        }
        // Closing tags (/i, /b, /u, /font) are silently consumed
    }

    (out, style)
}

/// Extract RGB colour from a `<font color="...">` tag string.
fn extract_font_color(tag: &str) -> Option<(u8, u8, u8)> {
    let lower = tag.to_lowercase();
    let start = lower.find("color")? + 5;
    let after_key = lower[start..].trim_start_matches(['=', ' ', '"', '\'']);
    let hex_str = after_key.trim_start_matches('#');
    let hex_str = &hex_str[..hex_str
        .find(|c: char| !c.is_ascii_hexdigit())
        .unwrap_or(hex_str.len())];
    if hex_str.len() == 6 {
        let r = u8::from_str_radix(&hex_str[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex_str[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex_str[4..6], 16).ok()?;
        return Some((r, g, b));
    }
    None
}

/// Parse an ASS Style: line using the given field format.
fn parse_ass_style(data: &str, format: &[&str]) -> Option<AssStyle> {
    let values: Vec<&str> = data.splitn(format.len(), ',').collect();
    let field = |name: &str| -> Option<&str> {
        format
            .iter()
            .position(|&f| f.eq_ignore_ascii_case(name))
            .and_then(|i| values.get(i).copied())
    };

    let name = field("Name")?.trim().to_string();
    let fontname = field("Fontname").unwrap_or("Arial").trim().to_string();
    let fontsize: f32 = field("Fontsize")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(48.0);
    let bold = field("Bold").map(|s| s.trim() == "-1").unwrap_or(false);
    let italic = field("Italic").map(|s| s.trim() == "-1").unwrap_or(false);
    let alignment: u8 = field("Alignment")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(2);
    let margin_l: u32 = field("MarginL")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(10);
    let margin_r: u32 = field("MarginR")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(10);
    let margin_v: u32 = field("MarginV")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(10);

    let primary_color = field("PrimaryColour")
        .and_then(|s| parse_ass_color(s.trim()))
        .unwrap_or((255, 255, 255, 255));

    Some(AssStyle {
        name,
        fontname,
        fontsize,
        primary_color,
        bold,
        italic,
        alignment,
        margin_l,
        margin_r,
        margin_v,
    })
}

/// Parse ASS `&HAABBGGRR` colour → (R, G, B, A).
fn parse_ass_color(s: &str) -> Option<(u8, u8, u8, u8)> {
    let s = s.trim_start_matches("&H").trim_start_matches("0x");
    if s.len() < 8 {
        return None;
    }
    let aa = u8::from_str_radix(&s[0..2], 16).ok()?;
    let bb = u8::from_str_radix(&s[2..4], 16).ok()?;
    let gg = u8::from_str_radix(&s[4..6], 16).ok()?;
    let rr = u8::from_str_radix(&s[6..8], 16).ok()?;
    Some((rr, gg, bb, 255 - aa))
}

/// Parse an ASS Dialogue line into a `SubtitleEntry`.
fn parse_ass_dialogue(data: &str, format: &[&str], index: u32) -> Option<SubtitleEntry> {
    // Split into exactly format.len() fields; the last field (Text) may contain commas
    let mut values: Vec<&str> = data.splitn(format.len(), ',').collect();

    // Pad if fewer values than format fields
    while values.len() < format.len() {
        values.push("");
    }

    let field = |name: &str| -> Option<&str> {
        format
            .iter()
            .position(|&f| f.eq_ignore_ascii_case(name))
            .and_then(|i| values.get(i).copied())
    };

    let start_str = field("Start")?;
    let end_str = field("End")?;
    let raw_text = field("Text").unwrap_or("").trim();

    let start_ms = parse_ass_timestamp(start_str.trim())?;
    let end_ms = parse_ass_timestamp(end_str.trim())?;

    // Strip ASS override tags: {…}
    let text = strip_ass_overrides(raw_text);

    Some(SubtitleEntry {
        index,
        start_ms,
        end_ms,
        text,
        style: None,
    })
}

/// Remove ASS override tag blocks `{...}` from text.
fn strip_ass_overrides(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0usize;
    for ch in text.chars() {
        match ch {
            '{' => depth += 1,
            '}' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            _ => {
                if depth == 0 {
                    out.push(ch);
                }
            }
        }
    }
    // Normalise \N and \n line separators
    out.replace("\\N", "\n").replace("\\n", "\n")
}

/// Apply SRT inline tags for bold/italic/underline based on style.
fn apply_srt_tags(entry: &SubtitleEntry) -> String {
    let Some(style) = &entry.style else {
        return entry.text.clone();
    };

    let mut text = entry.text.clone();
    if style.underline {
        text = format!("<u>{text}</u>");
    }
    if style.bold {
        text = format!("<b>{text}</b>");
    }
    if style.italic {
        text = format!("<i>{text}</i>");
    }
    if let Some((r, g, b)) = style.color {
        text = format!("<font color=\"#{r:02X}{g:02X}{b:02X}\">{text}</font>");
    }
    text
}

/// Apply ASS inline override tags for style.
fn apply_ass_override_tags(entry: &SubtitleEntry) -> String {
    let Some(style) = &entry.style else {
        return entry.text.clone();
    };

    let mut tags = String::new();
    if style.bold {
        tags.push_str("\\b1");
    }
    if style.italic {
        tags.push_str("\\i1");
    }
    if style.underline {
        tags.push_str("\\u1");
    }
    if let Some((r, g, b)) = style.color {
        tags.push_str(&format!("\\c&H{b:02X}{g:02X}{r:02X}&"));
    }

    if tags.is_empty() {
        entry.text.clone()
    } else {
        format!("{{{tags}}}{}", entry.text)
    }
}

// ── Time formatters ───────────────────────────────────────────────────────────

/// Format ms → `HH:MM:SS,mmm` (SRT).
fn format_ms_srt(ms: u64) -> String {
    let millis = ms % 1000;
    let total_secs = ms / 1000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3600;
    format!("{hours:02}:{mins:02}:{secs:02},{millis:03}")
}

/// Format ms → `HH:MM:SS.mmm` (VTT / SBV).
fn format_ms_vtt(ms: u64) -> String {
    let millis = ms % 1000;
    let total_secs = ms / 1000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3600;
    format!("{hours:02}:{mins:02}:{secs:02}.{millis:03}")
}

/// Format ms → `H:MM:SS.cc` (ASS centiseconds).
fn format_ms_ass(ms: u64) -> String {
    let cs = (ms % 1000) / 10;
    let total_secs = ms / 1000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3600;
    format!("{hours}:{mins:02}:{secs:02}.{cs:02}")
}

/// Format ms → `HH:MM:SS.mmm` (TTML).
fn format_ms_ttml(ms: u64) -> String {
    format_ms_vtt(ms)
}

/// Format ms → `H:MM:SS.mmm` (SBV, no leading zero on hours).
fn format_ms_sbv(ms: u64) -> String {
    let millis = ms % 1000;
    let total_secs = ms / 1000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3600;
    format!("{hours}:{mins:02}:{secs:02}.{millis:03}")
}

/// Escape XML special characters.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SRT: &str = "\
1\n\
00:00:01,000 --> 00:00:04,000\n\
Hello, world!\n\
\n\
2\n\
00:00:05,000 --> 00:00:08,500\n\
<i>Italic text</i>\n\
\n";

    const SAMPLE_VTT: &str = "\
WEBVTT\n\
\n\
1\n\
00:00:01.000 --> 00:00:04.000\n\
Hello VTT\n\
\n\
2\n\
00:00:05.000 --> 00:00:08.000\n\
Second cue\n\
\n";

    const SAMPLE_ASS: &str = "\
[Script Info]\n\
ScriptType: v4.00+\n\
\n\
[V4+ Styles]\n\
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\
\n\
[Events]\n\
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello ASS!\n\
Dialogue: 0,0:00:05.00,0:00:08.00,Default,,0,0,0,,{\\i1}Italic ASS{\\i0}\n\
\n";

    const SAMPLE_SBV: &str = "\
0:00:01.000,0:00:04.000\n\
Hello SBV\n\
\n\
0:00:05.000,0:00:08.000\n\
Second SBV line\n\
\n";

    const SAMPLE_LRC: &str = "\
[ti:Test Song]\n\
[ar:Test Artist]\n\
[00:01.00]First lyric line\n\
[00:05.00]Second lyric line\n\
";

    // ── Format parsers ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_srt_basic() {
        let doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        assert_eq!(doc.entries.len(), 2);
        assert_eq!(doc.entries[0].text, "Hello, world!");
        assert_eq!(doc.entries[0].start_ms, 1_000);
        assert_eq!(doc.entries[0].end_ms, 4_000);
    }

    #[test]
    fn test_parse_srt_italic_tag() {
        let doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        let entry = &doc.entries[1];
        assert_eq!(entry.text, "Italic text");
        assert!(entry.style.as_ref().map(|s| s.italic).unwrap_or(false));
    }

    #[test]
    fn test_parse_srt_index() {
        let doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        assert_eq!(doc.entries[0].index, 1);
        assert_eq!(doc.entries[1].index, 2);
    }

    #[test]
    fn test_parse_vtt_basic() {
        let doc = SubtitleDocument::parse_vtt(SAMPLE_VTT).expect("parse_vtt should succeed");
        assert_eq!(doc.entries.len(), 2);
        assert_eq!(doc.entries[0].text, "Hello VTT");
    }

    #[test]
    fn test_parse_vtt_timing() {
        let doc = SubtitleDocument::parse_vtt(SAMPLE_VTT).expect("parse_vtt should succeed");
        assert_eq!(doc.entries[1].start_ms, 5_000);
        assert_eq!(doc.entries[1].end_ms, 8_000);
    }

    #[test]
    fn test_parse_vtt_missing_header() {
        let result = SubtitleDocument::parse_vtt("This is not VTT");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ass_basic() {
        let doc = SubtitleDocument::parse_ass(SAMPLE_ASS).expect("parse_ass should succeed");
        assert!(!doc.entries.is_empty());
        assert_eq!(doc.entries[0].text, "Hello ASS!");
    }

    #[test]
    fn test_parse_ass_styles() {
        let doc = SubtitleDocument::parse_ass(SAMPLE_ASS).expect("parse_ass should succeed");
        assert!(!doc.styles.is_empty());
        assert_eq!(doc.styles[0].name, "Default");
    }

    #[test]
    fn test_parse_sbv_basic() {
        let doc = SubtitleDocument::parse_sbv(SAMPLE_SBV).expect("parse_sbv should succeed");
        assert_eq!(doc.entries.len(), 2);
        assert_eq!(doc.entries[0].text, "Hello SBV");
        assert_eq!(doc.entries[0].start_ms, 1_000);
    }

    #[test]
    fn test_parse_lrc_basic() {
        let doc = SubtitleDocument::parse_lrc(SAMPLE_LRC).expect("parse_lrc should succeed");
        assert_eq!(doc.entries.len(), 2);
        assert_eq!(doc.entries[0].text, "First lyric line");
    }

    #[test]
    fn test_parse_lrc_metadata() {
        let doc = SubtitleDocument::parse_lrc(SAMPLE_LRC).expect("parse_lrc should succeed");
        assert_eq!(doc.metadata.get("ti"), Some(&"Test Song".to_string()));
        assert_eq!(doc.metadata.get("ar"), Some(&"Test Artist".to_string()));
    }

    // ── Serializers ────────────────────────────────────────────────────────

    #[test]
    fn test_to_srt_roundtrip() {
        let doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        let srt = doc.to_srt();
        let doc2 = SubtitleDocument::parse_srt(&srt).expect("re-parse should succeed");
        assert_eq!(doc.entries.len(), doc2.entries.len());
        assert_eq!(doc.entries[0].start_ms, doc2.entries[0].start_ms);
    }

    #[test]
    fn test_to_vtt_has_header() {
        let doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        let vtt = doc.to_vtt();
        assert!(vtt.starts_with("WEBVTT"));
    }

    #[test]
    fn test_to_ass_has_sections() {
        let doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        let ass = doc.to_ass();
        assert!(ass.contains("[Script Info]"));
        assert!(ass.contains("[V4+ Styles]"));
        assert!(ass.contains("[Events]"));
    }

    #[test]
    fn test_to_ttml_valid_xml() {
        let doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        let ttml = doc.to_ttml();
        assert!(ttml.starts_with("<?xml"));
        assert!(ttml.contains("<tt"));
        assert!(ttml.contains("<p begin="));
    }

    #[test]
    fn test_to_sbv_format() {
        let doc = SubtitleDocument::parse_sbv(SAMPLE_SBV).expect("parse_sbv should succeed");
        let sbv = doc.to_sbv();
        assert!(sbv.contains("0:00:01.000,0:00:04.000"));
        assert!(sbv.contains("Hello SBV"));
    }

    // ── Converter ─────────────────────────────────────────────────────────

    #[test]
    fn test_convert_srt_to_vtt() {
        let doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        let vtt_doc = SubtitleConverter::convert(&doc, SubtitleFormat::Vtt);
        assert_eq!(vtt_doc.format, SubtitleFormat::Vtt);
        assert_eq!(vtt_doc.entries.len(), doc.entries.len());
    }

    #[test]
    fn test_shift_timing() {
        let mut doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        SubtitleConverter::shift_timing(&mut doc, 1000);
        assert_eq!(doc.entries[0].start_ms, 2_000);
        assert_eq!(doc.entries[0].end_ms, 5_000);
    }

    #[test]
    fn test_shift_timing_negative_clamps() {
        let mut doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        SubtitleConverter::shift_timing(&mut doc, -5000);
        // start_ms was 1000; 1000 - 5000 < 0 → clamped to 0
        assert_eq!(doc.entries[0].start_ms, 0);
    }

    #[test]
    fn test_scale_timing() {
        let mut doc = SubtitleDocument::parse_srt(SAMPLE_SRT).expect("parse_srt should succeed");
        // Double all timestamps
        SubtitleConverter::scale_timing(&mut doc, 2.0);
        assert_eq!(doc.entries[0].start_ms, 2_000);
        assert_eq!(doc.entries[0].end_ms, 8_000);
    }

    #[test]
    fn test_merge_two_documents() {
        // Use docs with distinct timings so no entries collapse
        let mut doc_a = SubtitleDocument::empty(SubtitleFormat::Srt);
        doc_a
            .entries
            .push(SubtitleEntry::new(1, 1_000, 3_000, "A1"));
        doc_a
            .entries
            .push(SubtitleEntry::new(2, 4_000, 6_000, "A2"));
        let mut doc_b = SubtitleDocument::empty(SubtitleFormat::Srt);
        doc_b
            .entries
            .push(SubtitleEntry::new(1, 7_000, 9_000, "B1"));
        doc_b
            .entries
            .push(SubtitleEntry::new(2, 10_000, 12_000, "B2"));
        let merged = SubtitleConverter::merge(&doc_a, &doc_b);
        assert_eq!(merged.entries.len(), 4);
    }

    #[test]
    fn test_merge_sorted_by_start_time() {
        let mut doc_a = SubtitleDocument::empty(SubtitleFormat::Srt);
        doc_a
            .entries
            .push(SubtitleEntry::new(1, 5_000, 8_000, "Late"));
        let mut doc_b = SubtitleDocument::empty(SubtitleFormat::Srt);
        doc_b
            .entries
            .push(SubtitleEntry::new(1, 1_000, 4_000, "Early"));
        let merged = SubtitleConverter::merge(&doc_a, &doc_b);
        assert!(merged.entries[0].start_ms <= merged.entries[1].start_ms);
    }

    #[test]
    fn test_format_supports_styling() {
        assert!(SubtitleFormat::Ass.supports_styling());
        assert!(SubtitleFormat::Vtt.supports_styling());
        assert!(!SubtitleFormat::Srt.supports_styling());
        assert!(!SubtitleFormat::Sbv.supports_styling());
    }
}

// ============================================================================
// Parallel batch parsing using rayon
// ============================================================================

use rayon::prelude::*;

/// Result of a single file parse in a batch operation.
#[derive(Debug)]
pub struct BatchParseResult {
    /// The original input index.
    pub index: usize,
    /// Parsed document on success.
    pub document: Result<SubtitleDocument, String>,
}

/// Parse a batch of `(format, text)` inputs in parallel using rayon.
///
/// Each element of `inputs` is `(SubtitleFormat, &str)`.  Returns a vector of
/// [`BatchParseResult`] in the **same order** as the inputs (sorted by index).
///
/// # Example
///
/// ```rust
/// use oximedia_subtitle::format_converter::{parallel_parse_batch, SubtitleFormat};
///
/// let inputs = vec![
///     (SubtitleFormat::Srt, "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n"),
///     (SubtitleFormat::Vtt, "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nWorld\n\n"),
/// ];
/// let results = parallel_parse_batch(&inputs);
/// assert_eq!(results.len(), 2);
/// ```
#[must_use]
pub fn parallel_parse_batch(inputs: &[(SubtitleFormat, &str)]) -> Vec<BatchParseResult> {
    let mut results: Vec<BatchParseResult> = inputs
        .par_iter()
        .enumerate()
        .map(|(index, (format, text))| {
            let document = parse_one(*format, text);
            BatchParseResult { index, document }
        })
        .collect();

    // Restore original order (rayon may reorder).
    results.sort_by_key(|r| r.index);
    results
}

/// Parse a single subtitle document from text in the specified format.
fn parse_one(format: SubtitleFormat, text: &str) -> Result<SubtitleDocument, String> {
    match format {
        SubtitleFormat::Srt => SubtitleDocument::parse_srt(text),
        SubtitleFormat::Vtt => SubtitleDocument::parse_vtt(text),
        SubtitleFormat::Ass | SubtitleFormat::Ssa => SubtitleDocument::parse_ass(text),
        SubtitleFormat::Ttml | SubtitleFormat::Dfxp => SubtitleDocument::parse_vtt(text),
        SubtitleFormat::Sbv => SubtitleDocument::parse_sbv(text),
        SubtitleFormat::Lrc => SubtitleDocument::parse_lrc(text),
        SubtitleFormat::Scc => Err("SCC batch parsing not yet supported".to_string()),
    }
}

/// Parse a batch of `(format, owned_text)` inputs in parallel, where inputs
/// are owned strings (useful when texts come from async I/O or allocation).
///
/// Returns results in the same order as inputs.
#[must_use]
pub fn parallel_parse_batch_owned(inputs: &[(SubtitleFormat, String)]) -> Vec<BatchParseResult> {
    let mut results: Vec<BatchParseResult> = inputs
        .par_iter()
        .enumerate()
        .map(|(index, (format, text))| {
            let document = parse_one(*format, text);
            BatchParseResult { index, document }
        })
        .collect();

    results.sort_by_key(|r| r.index);
    results
}

#[cfg(test)]
mod parallel_tests {
    use super::*;

    const SAMPLE_SRT_BATCH: &str =
        "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n2\n00:00:05,000 --> 00:00:08,000\nWorld\n\n";
    const SAMPLE_VTT_BATCH: &str = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nHello VTT\n\n";

    #[test]
    fn test_parallel_batch_srt() {
        let inputs = vec![(SubtitleFormat::Srt, SAMPLE_SRT_BATCH)];
        let results = parallel_parse_batch(&inputs);
        assert_eq!(results.len(), 1);
        assert!(results[0].document.is_ok());
        let doc = results[0].document.as_ref().expect("doc");
        assert_eq!(doc.entries.len(), 2);
    }

    #[test]
    fn test_parallel_batch_vtt() {
        let inputs = vec![(SubtitleFormat::Vtt, SAMPLE_VTT_BATCH)];
        let results = parallel_parse_batch(&inputs);
        assert_eq!(results.len(), 1);
        assert!(results[0].document.is_ok());
    }

    #[test]
    fn test_parallel_batch_mixed_formats() {
        let inputs = vec![
            (SubtitleFormat::Srt, SAMPLE_SRT_BATCH),
            (SubtitleFormat::Vtt, SAMPLE_VTT_BATCH),
        ];
        let results = parallel_parse_batch(&inputs);
        assert_eq!(results.len(), 2);
        // Verify order is preserved
        assert_eq!(results[0].index, 0);
        assert_eq!(results[1].index, 1);
        // Both should succeed
        assert!(results[0].document.is_ok());
        assert!(results[1].document.is_ok());
    }

    #[test]
    fn test_parallel_batch_order_preserved() {
        // Use many inputs to exercise parallel re-ordering
        let texts: Vec<String> = (1u32..=20)
            .map(|i| {
                format!(
                    "{i}\n00:00:{i:02},000 --> 00:00:{:02},000\nCue {i}\n\n",
                    i + 1
                )
            })
            .collect();
        let inputs: Vec<(SubtitleFormat, &str)> = texts
            .iter()
            .map(|t| (SubtitleFormat::Srt, t.as_str()))
            .collect();
        let results = parallel_parse_batch(&inputs);
        assert_eq!(results.len(), 20);
        for (expected_idx, result) in results.iter().enumerate() {
            assert_eq!(
                result.index, expected_idx,
                "Order violated at index {expected_idx}"
            );
        }
    }

    #[test]
    fn test_parallel_batch_empty() {
        let results = parallel_parse_batch(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parallel_batch_error_propagation() {
        let inputs = vec![(SubtitleFormat::Srt, "not valid srt at all\n\n")];
        let results = parallel_parse_batch(&inputs);
        assert_eq!(results.len(), 1);
        // Invalid SRT — either error or empty doc; the important thing is it doesn't panic
        let _ = &results[0].document;
    }

    #[test]
    fn test_parallel_batch_owned() {
        let inputs = vec![
            (SubtitleFormat::Srt, SAMPLE_SRT_BATCH.to_string()),
            (SubtitleFormat::Vtt, SAMPLE_VTT_BATCH.to_string()),
        ];
        let results = parallel_parse_batch_owned(&inputs);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 0);
        assert_eq!(results[1].index, 1);
    }

    #[test]
    fn test_parse_one_scc_unsupported() {
        let result = parse_one(SubtitleFormat::Scc, "some scc data");
        assert!(result.is_err());
    }
}
