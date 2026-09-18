//! SSML prosody control for TTS with full markup support.
//!
//! Provides comprehensive SSML (Speech Synthesis Markup Language) generation
//! including prosody (rate, pitch, volume), emphasis, breaks, say-as,
//! phoneme, sub, and mark elements for fine-grained speech control.

use crate::error::{AccessError, AccessResult};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Prosody configuration for speech synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyConfig {
    /// Speech rate multiplier (0.5 to 2.0).
    pub rate: f32,
    /// Pitch shift in semitones (-12 to 12).
    pub pitch: f32,
    /// Volume level (0.0 to 1.0).
    pub volume: f32,
    /// Emphasis level (0.0 to 1.0).
    pub emphasis: f32,
}

impl Default for ProsodyConfig {
    fn default() -> Self {
        Self {
            rate: 1.0,
            pitch: 0.0,
            volume: 0.8,
            emphasis: 0.5,
        }
    }
}

/// SSML emphasis level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmphasisLevel {
    /// Reduced emphasis.
    Reduced,
    /// No additional emphasis.
    None,
    /// Moderate emphasis.
    Moderate,
    /// Strong emphasis.
    Strong,
}

impl fmt::Display for EmphasisLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reduced => write!(f, "reduced"),
            Self::None => write!(f, "none"),
            Self::Moderate => write!(f, "moderate"),
            Self::Strong => write!(f, "strong"),
        }
    }
}

/// SSML break strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BreakStrength {
    /// No break.
    None,
    /// Extra-weak break (within word).
    ExtraWeak,
    /// Weak break (comma-like).
    Weak,
    /// Medium break (sentence boundary).
    Medium,
    /// Strong break (paragraph boundary).
    Strong,
    /// Extra-strong break (section boundary).
    ExtraStrong,
}

impl fmt::Display for BreakStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::ExtraWeak => write!(f, "x-weak"),
            Self::Weak => write!(f, "weak"),
            Self::Medium => write!(f, "medium"),
            Self::Strong => write!(f, "strong"),
            Self::ExtraStrong => write!(f, "x-strong"),
        }
    }
}

/// SSML say-as interpret type for pronunciation guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SayAsInterpret {
    /// Spell out characters (e.g., "ABC" -> "A B C").
    Characters,
    /// Cardinal number.
    Cardinal,
    /// Ordinal number (e.g., "1st").
    Ordinal,
    /// Fraction.
    Fraction,
    /// Telephone number.
    Telephone,
    /// Date (with optional format attribute).
    Date,
    /// Time value.
    Time,
    /// Postal/ZIP code.
    Address,
    /// Currency value.
    Currency,
    /// Unit of measure.
    Unit,
    /// Verbatim spelling.
    Verbatim,
}

impl fmt::Display for SayAsInterpret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Characters => write!(f, "characters"),
            Self::Cardinal => write!(f, "cardinal"),
            Self::Ordinal => write!(f, "ordinal"),
            Self::Fraction => write!(f, "fraction"),
            Self::Telephone => write!(f, "telephone"),
            Self::Date => write!(f, "date"),
            Self::Time => write!(f, "time"),
            Self::Address => write!(f, "address"),
            Self::Currency => write!(f, "currency"),
            Self::Unit => write!(f, "unit"),
            Self::Verbatim => write!(f, "verbatim"),
        }
    }
}

/// An SSML element in the markup tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SsmlElement {
    /// Plain text content.
    Text(String),
    /// Prosody wrapper (rate, pitch, volume).
    Prosody {
        /// Rate multiplier.
        rate: Option<f32>,
        /// Pitch in semitones.
        pitch: Option<f32>,
        /// Volume percentage (0-100).
        volume: Option<f32>,
        /// Child elements.
        children: Vec<SsmlElement>,
    },
    /// Emphasis wrapper.
    Emphasis {
        /// Emphasis level.
        level: EmphasisLevel,
        /// Child elements.
        children: Vec<SsmlElement>,
    },
    /// Break (pause) element.
    Break {
        /// Break strength.
        strength: Option<BreakStrength>,
        /// Break time in milliseconds (overrides strength if set).
        time_ms: Option<u32>,
    },
    /// Say-as element for pronunciation control.
    SayAs {
        /// Interpretation type.
        interpret_as: SayAsInterpret,
        /// Optional format (e.g., "mdy" for dates).
        format: Option<String>,
        /// The text to interpret.
        text: String,
    },
    /// Phoneme element for precise pronunciation.
    Phoneme {
        /// Phonetic alphabet ("ipa" or "x-sampa").
        alphabet: String,
        /// Phonetic transcription.
        ph: String,
        /// The original text.
        text: String,
    },
    /// Substitution element (speak one thing, display another).
    Sub {
        /// The text to actually speak.
        alias: String,
        /// The original/display text.
        text: String,
    },
    /// Mark element for synchronization bookmarks.
    Mark {
        /// Mark name identifier.
        name: String,
    },
    /// Sentence element.
    Sentence {
        /// Child elements.
        children: Vec<SsmlElement>,
    },
    /// Paragraph element.
    Paragraph {
        /// Child elements.
        children: Vec<SsmlElement>,
    },
}

impl SsmlElement {
    /// Render this element to an SSML string.
    #[must_use]
    pub fn to_ssml(&self) -> String {
        match self {
            Self::Text(t) => escape_xml(t),
            Self::Prosody {
                rate,
                pitch,
                volume,
                children,
            } => {
                let mut attrs = Vec::new();
                if let Some(r) = rate {
                    attrs.push(format!("rate=\"{r}\""));
                }
                if let Some(p) = pitch {
                    attrs.push(format!("pitch=\"{p}st\""));
                }
                if let Some(v) = volume {
                    attrs.push(format!("volume=\"{v}\""));
                }
                let attr_str = if attrs.is_empty() {
                    String::new()
                } else {
                    format!(" {}", attrs.join(" "))
                };
                let inner: String = children.iter().map(|c| c.to_ssml()).collect();
                format!("<prosody{attr_str}>{inner}</prosody>")
            }
            Self::Emphasis { level, children } => {
                let inner: String = children.iter().map(|c| c.to_ssml()).collect();
                format!("<emphasis level=\"{level}\">{inner}</emphasis>")
            }
            Self::Break { strength, time_ms } => {
                if let Some(ms) = time_ms {
                    format!("<break time=\"{ms}ms\"/>")
                } else if let Some(s) = strength {
                    format!("<break strength=\"{s}\"/>")
                } else {
                    "<break/>".to_string()
                }
            }
            Self::SayAs {
                interpret_as,
                format,
                text,
            } => {
                let fmt_attr = format
                    .as_ref()
                    .map(|f| format!(" format=\"{f}\""))
                    .unwrap_or_default();
                format!(
                    "<say-as interpret-as=\"{interpret_as}\"{fmt_attr}>{}</say-as>",
                    escape_xml(text)
                )
            }
            Self::Phoneme { alphabet, ph, text } => {
                format!(
                    "<phoneme alphabet=\"{alphabet}\" ph=\"{ph}\">{}</phoneme>",
                    escape_xml(text)
                )
            }
            Self::Sub { alias, text } => {
                format!(
                    "<sub alias=\"{}\">{}</sub>",
                    escape_xml(alias),
                    escape_xml(text)
                )
            }
            Self::Mark { name } => {
                format!("<mark name=\"{name}\"/>")
            }
            Self::Sentence { children } => {
                let inner: String = children.iter().map(|c| c.to_ssml()).collect();
                format!("<s>{inner}</s>")
            }
            Self::Paragraph { children } => {
                let inner: String = children.iter().map(|c| c.to_ssml()).collect();
                format!("<p>{inner}</p>")
            }
        }
    }
}

/// Escape XML special characters.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Builder for constructing complex SSML documents.
///
/// Provides a fluent API for building SSML markup with nested elements,
/// prosody control, emphasis, breaks, and pronunciation guidance.
#[derive(Debug, Clone)]
pub struct SsmlBuilder {
    /// Language attribute for the speak element.
    language: String,
    /// Root-level elements.
    elements: Vec<SsmlElement>,
}

impl SsmlBuilder {
    /// Create a new SSML builder with the given language.
    #[must_use]
    pub fn new(language: &str) -> Self {
        Self {
            language: language.to_string(),
            elements: Vec::new(),
        }
    }

    /// Add plain text.
    #[must_use]
    pub fn text(mut self, text: &str) -> Self {
        self.elements.push(SsmlElement::Text(text.to_string()));
        self
    }

    /// Add a break/pause by strength.
    #[must_use]
    pub fn break_strength(mut self, strength: BreakStrength) -> Self {
        self.elements.push(SsmlElement::Break {
            strength: Some(strength),
            time_ms: None,
        });
        self
    }

    /// Add a break/pause by time in milliseconds.
    #[must_use]
    pub fn break_time(mut self, time_ms: u32) -> Self {
        self.elements.push(SsmlElement::Break {
            strength: None,
            time_ms: Some(time_ms),
        });
        self
    }

    /// Add emphasized text.
    #[must_use]
    pub fn emphasis(mut self, level: EmphasisLevel, text: &str) -> Self {
        self.elements.push(SsmlElement::Emphasis {
            level,
            children: vec![SsmlElement::Text(text.to_string())],
        });
        self
    }

    /// Add text with prosody control.
    #[must_use]
    pub fn prosody(
        mut self,
        rate: Option<f32>,
        pitch: Option<f32>,
        volume: Option<f32>,
        text: &str,
    ) -> Self {
        self.elements.push(SsmlElement::Prosody {
            rate,
            pitch,
            volume,
            children: vec![SsmlElement::Text(text.to_string())],
        });
        self
    }

    /// Add a say-as element for pronunciation guidance.
    #[must_use]
    pub fn say_as(mut self, interpret_as: SayAsInterpret, text: &str) -> Self {
        self.elements.push(SsmlElement::SayAs {
            interpret_as,
            format: None,
            text: text.to_string(),
        });
        self
    }

    /// Add a say-as element with format attribute.
    #[must_use]
    pub fn say_as_with_format(
        mut self,
        interpret_as: SayAsInterpret,
        format: &str,
        text: &str,
    ) -> Self {
        self.elements.push(SsmlElement::SayAs {
            interpret_as,
            format: Some(format.to_string()),
            text: text.to_string(),
        });
        self
    }

    /// Add a phoneme element for precise pronunciation.
    #[must_use]
    pub fn phoneme(mut self, alphabet: &str, ph: &str, text: &str) -> Self {
        self.elements.push(SsmlElement::Phoneme {
            alphabet: alphabet.to_string(),
            ph: ph.to_string(),
            text: text.to_string(),
        });
        self
    }

    /// Add a substitution (alias) element.
    #[must_use]
    pub fn sub(mut self, alias: &str, text: &str) -> Self {
        self.elements.push(SsmlElement::Sub {
            alias: alias.to_string(),
            text: text.to_string(),
        });
        self
    }

    /// Add a synchronization mark.
    #[must_use]
    pub fn mark(mut self, name: &str) -> Self {
        self.elements.push(SsmlElement::Mark {
            name: name.to_string(),
        });
        self
    }

    /// Wrap elements in a sentence.
    #[must_use]
    pub fn sentence(mut self, children: Vec<SsmlElement>) -> Self {
        self.elements.push(SsmlElement::Sentence { children });
        self
    }

    /// Wrap elements in a paragraph.
    #[must_use]
    pub fn paragraph(mut self, children: Vec<SsmlElement>) -> Self {
        self.elements.push(SsmlElement::Paragraph { children });
        self
    }

    /// Build the complete SSML document.
    #[must_use]
    pub fn build(&self) -> String {
        let inner: String = self.elements.iter().map(|e| e.to_ssml()).collect();
        format!(
            "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" \
             xml:lang=\"{}\">{inner}</speak>",
            self.language
        )
    }

    /// Validate the SSML document structure.
    pub fn validate(&self) -> AccessResult<()> {
        if self.elements.is_empty() {
            return Err(AccessError::TtsFailed(
                "SSML document has no elements".to_string(),
            ));
        }
        if self.language.is_empty() {
            return Err(AccessError::TtsFailed(
                "SSML language must not be empty".to_string(),
            ));
        }
        for element in &self.elements {
            validate_element(element)?;
        }
        Ok(())
    }

    /// Get element count.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Get the language.
    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }
}

/// Validate a single SSML element recursively.
fn validate_element(element: &SsmlElement) -> AccessResult<()> {
    match element {
        SsmlElement::Prosody {
            rate,
            pitch,
            children,
            ..
        } => {
            if let Some(r) = rate {
                if *r < 0.1 || *r > 10.0 {
                    return Err(AccessError::TtsFailed(format!(
                        "SSML prosody rate {r} out of range [0.1, 10.0]"
                    )));
                }
            }
            if let Some(p) = pitch {
                if *p < -24.0 || *p > 24.0 {
                    return Err(AccessError::TtsFailed(format!(
                        "SSML prosody pitch {p}st out of range [-24, 24]"
                    )));
                }
            }
            for child in children {
                validate_element(child)?;
            }
        }
        SsmlElement::Emphasis { children, .. } => {
            for child in children {
                validate_element(child)?;
            }
        }
        SsmlElement::Sentence { children } | SsmlElement::Paragraph { children } => {
            for child in children {
                validate_element(child)?;
            }
        }
        SsmlElement::Phoneme { alphabet, .. } => {
            if alphabet != "ipa" && alphabet != "x-sampa" {
                return Err(AccessError::TtsFailed(format!(
                    "Unsupported phonetic alphabet: {alphabet}"
                )));
            }
        }
        _ => {}
    }
    Ok(())
}

/// Controls prosody (pitch, rate, volume) of synthesized speech.
pub struct ProsodyControl {
    config: ProsodyConfig,
}

impl ProsodyControl {
    /// Create a new prosody controller.
    #[must_use]
    pub const fn new(config: ProsodyConfig) -> Self {
        Self { config }
    }

    /// Set speech rate.
    pub fn set_rate(&mut self, rate: f32) {
        self.config.rate = rate.clamp(0.5, 2.0);
    }

    /// Set pitch shift.
    pub fn set_pitch(&mut self, pitch: f32) {
        self.config.pitch = pitch.clamp(-12.0, 12.0);
    }

    /// Set volume.
    pub fn set_volume(&mut self, volume: f32) {
        self.config.volume = volume.clamp(0.0, 1.0);
    }

    /// Generate SSML markup for prosody.
    #[must_use]
    pub fn to_ssml(&self, text: &str) -> String {
        format!(
            "<prosody rate=\"{}\" pitch=\"{}st\" volume=\"{}\">{}</prosody>",
            self.config.rate,
            self.config.pitch,
            self.config.volume * 100.0,
            text
        )
    }

    /// Generate full SSML document using the builder.
    #[must_use]
    pub fn to_ssml_document(&self, text: &str, language: &str) -> String {
        SsmlBuilder::new(language)
            .prosody(
                Some(self.config.rate),
                Some(self.config.pitch),
                Some(self.config.volume * 100.0),
                text,
            )
            .build()
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &ProsodyConfig {
        &self.config
    }
}

impl Default for ProsodyControl {
    fn default() -> Self {
        Self::new(ProsodyConfig::default())
    }
}

/// Convert annotated text to SSML using simple markup conventions.
///
/// Supported markers:
/// - `*word*` -> emphasis (strong)
/// - `_word_` -> emphasis (moderate)
/// - `{pause:500}` -> break of 500ms
/// - `{spell:ABC}` -> say-as characters
/// - `{num:42}` -> say-as cardinal
/// - `{date:01/15/2024}` -> say-as date
/// - `{sub:alias|text}` -> substitution
pub fn annotated_to_ssml(text: &str, language: &str) -> AccessResult<String> {
    let mut builder = SsmlBuilder::new(language);
    let mut remaining = text;

    while !remaining.is_empty() {
        // Look for the next special marker
        let next_marker = find_next_marker(remaining);

        match next_marker {
            Some((before, marker_type, content, after)) => {
                if !before.is_empty() {
                    builder = builder.text(before);
                }
                match marker_type {
                    MarkerType::StrongEmphasis => {
                        builder = builder.emphasis(EmphasisLevel::Strong, content);
                    }
                    MarkerType::ModerateEmphasis => {
                        builder = builder.emphasis(EmphasisLevel::Moderate, content);
                    }
                    MarkerType::Pause(ms) => {
                        builder = builder.break_time(ms);
                    }
                    MarkerType::Spell => {
                        builder = builder.say_as(SayAsInterpret::Characters, content);
                    }
                    MarkerType::Number => {
                        builder = builder.say_as(SayAsInterpret::Cardinal, content);
                    }
                    MarkerType::Date => {
                        builder = builder.say_as_with_format(SayAsInterpret::Date, "mdy", content);
                    }
                    MarkerType::Sub(alias) => {
                        builder = builder.sub(&alias, content);
                    }
                }
                remaining = after;
            }
            None => {
                builder = builder.text(remaining);
                remaining = "";
            }
        }
    }

    builder.validate()?;
    Ok(builder.build())
}

/// Internal marker type for annotated text parsing.
enum MarkerType {
    StrongEmphasis,
    ModerateEmphasis,
    Pause(u32),
    Spell,
    Number,
    Date,
    Sub(String),
}

/// Find the next marker in annotated text.
/// Returns (before_text, marker_type, content, after_text) or None.
fn find_next_marker(text: &str) -> Option<(&str, MarkerType, &str, &str)> {
    let mut best_pos = text.len();
    let mut best_result: Option<(MarkerType, &str, &str)> = None;

    // Check for *strong emphasis*
    if let Some(star_pos) = text.find('*') {
        if let Some(end_pos) = text[star_pos + 1..].find('*') {
            let content_start = star_pos + 1;
            let content_end = star_pos + 1 + end_pos;
            if star_pos < best_pos && content_start < content_end {
                best_pos = star_pos;
                best_result = Some((
                    MarkerType::StrongEmphasis,
                    &text[content_start..content_end],
                    &text[content_end + 1..],
                ));
            }
        }
    }

    // Check for _moderate emphasis_
    if let Some(under_pos) = text.find('_') {
        if let Some(end_pos) = text[under_pos + 1..].find('_') {
            let content_start = under_pos + 1;
            let content_end = under_pos + 1 + end_pos;
            if under_pos < best_pos && content_start < content_end {
                best_pos = under_pos;
                best_result = Some((
                    MarkerType::ModerateEmphasis,
                    &text[content_start..content_end],
                    &text[content_end + 1..],
                ));
            }
        }
    }

    // Check for {command:value} patterns
    if let Some(brace_pos) = text.find('{') {
        if let Some(close_pos) = text[brace_pos..].find('}') {
            let inner = &text[brace_pos + 1..brace_pos + close_pos];
            if brace_pos < best_pos {
                if let Some(parsed) = parse_brace_command(inner) {
                    let after = &text[brace_pos + close_pos + 1..];
                    best_pos = brace_pos;
                    best_result = Some((parsed.0, parsed.1, after));
                }
            }
        }
    }

    best_result
        .map(|(marker_type, content, after)| (&text[..best_pos], marker_type, content, after))
}

/// Parse a brace command like "pause:500" or "spell:ABC".
fn parse_brace_command(inner: &str) -> Option<(MarkerType, &str)> {
    let (cmd, value) = inner.split_once(':')?;
    match cmd {
        "pause" => {
            let ms: u32 = value.parse().ok()?;
            // Return empty content for pause since content isn't used
            Some((MarkerType::Pause(ms), ""))
        }
        "spell" => Some((MarkerType::Spell, value)),
        "num" => Some((MarkerType::Number, value)),
        "date" => Some((MarkerType::Date, value)),
        "sub" => {
            let (alias, display) = value.split_once('|')?;
            // We need to return a MarkerType::Sub with the alias
            // and display text as content
            Some((MarkerType::Sub(alias.to_string()), display))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prosody_creation() {
        let prosody = ProsodyControl::default();
        assert!((prosody.config().rate - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_rate() {
        let mut prosody = ProsodyControl::default();
        prosody.set_rate(1.5);
        assert!((prosody.config().rate - 1.5).abs() < f32::EPSILON);

        prosody.set_rate(5.0);
        assert!((prosody.config().rate - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ssml_generation() {
        let prosody = ProsodyControl::default();
        let ssml = prosody.to_ssml("Hello world");
        assert!(ssml.contains("prosody"));
        assert!(ssml.contains("Hello world"));
    }

    // ============================================================
    // SSML builder tests
    // ============================================================

    #[test]
    fn test_ssml_builder_basic() {
        let ssml = SsmlBuilder::new("en-US").text("Hello world.").build();
        assert!(ssml.contains("<speak"));
        assert!(ssml.contains("xml:lang=\"en-US\""));
        assert!(ssml.contains("Hello world."));
        assert!(ssml.contains("</speak>"));
    }

    #[test]
    fn test_ssml_builder_emphasis() {
        let ssml = SsmlBuilder::new("en-US")
            .text("This is ")
            .emphasis(EmphasisLevel::Strong, "important")
            .text(" information.")
            .build();
        assert!(ssml.contains("<emphasis level=\"strong\">important</emphasis>"));
    }

    #[test]
    fn test_ssml_builder_break_strength() {
        let ssml = SsmlBuilder::new("en-US")
            .text("First part.")
            .break_strength(BreakStrength::Strong)
            .text("Second part.")
            .build();
        assert!(ssml.contains("<break strength=\"strong\"/>"));
    }

    #[test]
    fn test_ssml_builder_break_time() {
        let ssml = SsmlBuilder::new("en-US")
            .text("Wait.")
            .break_time(500)
            .text("Continue.")
            .build();
        assert!(ssml.contains("<break time=\"500ms\"/>"));
    }

    #[test]
    fn test_ssml_builder_prosody() {
        let ssml = SsmlBuilder::new("en-US")
            .prosody(Some(0.8), Some(2.0), None, "Slow and high")
            .build();
        assert!(ssml.contains("rate=\"0.8\""));
        assert!(ssml.contains("pitch=\"2st\""));
        assert!(!ssml.contains("volume"));
    }

    #[test]
    fn test_ssml_builder_say_as() {
        let ssml = SsmlBuilder::new("en-US")
            .say_as(SayAsInterpret::Characters, "NATO")
            .build();
        assert!(ssml.contains("<say-as interpret-as=\"characters\">NATO</say-as>"));
    }

    #[test]
    fn test_ssml_builder_say_as_with_format() {
        let ssml = SsmlBuilder::new("en-US")
            .say_as_with_format(SayAsInterpret::Date, "mdy", "01/15/2024")
            .build();
        assert!(ssml.contains("<say-as interpret-as=\"date\" format=\"mdy\">01/15/2024</say-as>"));
    }

    #[test]
    fn test_ssml_builder_phoneme() {
        let ssml = SsmlBuilder::new("en-US")
            .phoneme("ipa", "t\u{0259}me\u{026a}to\u{028a}", "tomato")
            .build();
        assert!(ssml.contains("<phoneme alphabet=\"ipa\""));
        assert!(ssml.contains(">tomato</phoneme>"));
    }

    #[test]
    fn test_ssml_builder_sub() {
        let ssml = SsmlBuilder::new("en-US")
            .sub("World Wide Web Consortium", "W3C")
            .build();
        assert!(ssml.contains("<sub alias=\"World Wide Web Consortium\">W3C</sub>"));
    }

    #[test]
    fn test_ssml_builder_mark() {
        let ssml = SsmlBuilder::new("en-US")
            .text("Part one.")
            .mark("section2")
            .text("Part two.")
            .build();
        assert!(ssml.contains("<mark name=\"section2\"/>"));
    }

    #[test]
    fn test_ssml_builder_sentence_and_paragraph() {
        let ssml = SsmlBuilder::new("en-US")
            .paragraph(vec![
                SsmlElement::Sentence {
                    children: vec![SsmlElement::Text("First sentence.".to_string())],
                },
                SsmlElement::Sentence {
                    children: vec![SsmlElement::Text("Second sentence.".to_string())],
                },
            ])
            .build();
        assert!(ssml.contains("<p>"));
        assert!(ssml.contains("<s>First sentence.</s>"));
        assert!(ssml.contains("<s>Second sentence.</s>"));
        assert!(ssml.contains("</p>"));
    }

    #[test]
    fn test_ssml_builder_validate_empty() {
        let builder = SsmlBuilder::new("en-US");
        assert!(builder.validate().is_err());
    }

    #[test]
    fn test_ssml_builder_validate_empty_language() {
        let builder = SsmlBuilder::new("").text("Hello");
        assert!(builder.validate().is_err());
    }

    #[test]
    fn test_ssml_builder_validate_bad_rate() {
        let builder = SsmlBuilder::new("en-US").prosody(Some(100.0), None, None, "Too fast");
        assert!(builder.validate().is_err());
    }

    #[test]
    fn test_ssml_builder_validate_bad_phoneme_alphabet() {
        let builder = SsmlBuilder::new("en-US").phoneme("invalid-alphabet", "test", "test");
        assert!(builder.validate().is_err());
    }

    #[test]
    fn test_ssml_builder_validate_success() {
        let builder = SsmlBuilder::new("en-US")
            .text("Hello")
            .emphasis(EmphasisLevel::Strong, "world")
            .break_time(200);
        assert!(builder.validate().is_ok());
    }

    #[test]
    fn test_ssml_builder_element_count() {
        let builder = SsmlBuilder::new("en-US")
            .text("A")
            .break_time(100)
            .text("B");
        assert_eq!(builder.element_count(), 3);
    }

    #[test]
    fn test_ssml_builder_language() {
        let builder = SsmlBuilder::new("ja-JP");
        assert_eq!(builder.language(), "ja-JP");
    }

    #[test]
    fn test_xml_escape() {
        let ssml = SsmlBuilder::new("en-US").text("A < B & C > D").build();
        assert!(ssml.contains("A &lt; B &amp; C &gt; D"));
    }

    #[test]
    fn test_prosody_control_to_ssml_document() {
        let prosody = ProsodyControl::new(ProsodyConfig {
            rate: 0.9,
            pitch: 1.0,
            volume: 0.7,
            emphasis: 0.5,
        });
        let doc = prosody.to_ssml_document("Hello", "en-US");
        assert!(doc.contains("<speak"));
        assert!(doc.contains("prosody"));
        assert!(doc.contains("Hello"));
    }

    // ============================================================
    // Annotated text to SSML tests
    // ============================================================

    #[test]
    fn test_annotated_strong_emphasis() {
        let result = annotated_to_ssml("This is *important* text.", "en-US");
        let ssml = result.expect("should succeed");
        assert!(ssml.contains("<emphasis level=\"strong\">important</emphasis>"));
        assert!(ssml.contains("This is "));
        assert!(ssml.contains(" text."));
    }

    #[test]
    fn test_annotated_moderate_emphasis() {
        let result = annotated_to_ssml("This is _somewhat_ relevant.", "en-US");
        let ssml = result.expect("should succeed");
        assert!(ssml.contains("<emphasis level=\"moderate\">somewhat</emphasis>"));
    }

    #[test]
    fn test_annotated_pause() {
        let result = annotated_to_ssml("Wait.{pause:500}Continue.", "en-US");
        let ssml = result.expect("should succeed");
        assert!(ssml.contains("<break time=\"500ms\"/>"));
    }

    #[test]
    fn test_annotated_spell() {
        let result = annotated_to_ssml("The code is {spell:NASA}.", "en-US");
        let ssml = result.expect("should succeed");
        assert!(ssml.contains("<say-as interpret-as=\"characters\">NASA</say-as>"));
    }

    #[test]
    fn test_annotated_number() {
        let result = annotated_to_ssml("There are {num:42} items.", "en-US");
        let ssml = result.expect("should succeed");
        assert!(ssml.contains("<say-as interpret-as=\"cardinal\">42</say-as>"));
    }

    #[test]
    fn test_annotated_date() {
        let result = annotated_to_ssml("Born on {date:01/15/2024}.", "en-US");
        let ssml = result.expect("should succeed");
        assert!(ssml.contains("interpret-as=\"date\""));
        assert!(ssml.contains("format=\"mdy\""));
    }

    #[test]
    fn test_annotated_sub() {
        let result = annotated_to_ssml("Visit {sub:World Wide Web Consortium|W3C}.", "en-US");
        let ssml = result.expect("should succeed");
        assert!(ssml.contains("<sub alias=\"World Wide Web Consortium\">W3C</sub>"));
    }

    #[test]
    fn test_annotated_plain_text() {
        let result = annotated_to_ssml("Just plain text.", "en-US");
        let ssml = result.expect("should succeed");
        assert!(ssml.contains("Just plain text."));
        assert!(!ssml.contains("<emphasis"));
        assert!(!ssml.contains("<break"));
    }

    #[test]
    fn test_emphasis_level_display() {
        assert_eq!(EmphasisLevel::Strong.to_string(), "strong");
        assert_eq!(EmphasisLevel::Moderate.to_string(), "moderate");
        assert_eq!(EmphasisLevel::Reduced.to_string(), "reduced");
        assert_eq!(EmphasisLevel::None.to_string(), "none");
    }

    #[test]
    fn test_break_strength_display() {
        assert_eq!(BreakStrength::ExtraWeak.to_string(), "x-weak");
        assert_eq!(BreakStrength::Weak.to_string(), "weak");
        assert_eq!(BreakStrength::Medium.to_string(), "medium");
        assert_eq!(BreakStrength::Strong.to_string(), "strong");
        assert_eq!(BreakStrength::ExtraStrong.to_string(), "x-strong");
        assert_eq!(BreakStrength::None.to_string(), "none");
    }

    #[test]
    fn test_say_as_interpret_display() {
        assert_eq!(SayAsInterpret::Characters.to_string(), "characters");
        assert_eq!(SayAsInterpret::Cardinal.to_string(), "cardinal");
        assert_eq!(SayAsInterpret::Ordinal.to_string(), "ordinal");
        assert_eq!(SayAsInterpret::Telephone.to_string(), "telephone");
        assert_eq!(SayAsInterpret::Date.to_string(), "date");
        assert_eq!(SayAsInterpret::Time.to_string(), "time");
        assert_eq!(SayAsInterpret::Currency.to_string(), "currency");
        assert_eq!(SayAsInterpret::Verbatim.to_string(), "verbatim");
    }

    #[test]
    fn test_complex_ssml_document() {
        let ssml = SsmlBuilder::new("en-US")
            .paragraph(vec![SsmlElement::Sentence {
                children: vec![
                    SsmlElement::Text("Welcome to ".to_string()),
                    SsmlElement::Emphasis {
                        level: EmphasisLevel::Strong,
                        children: vec![SsmlElement::Text("OxiMedia".to_string())],
                    },
                    SsmlElement::Text(".".to_string()),
                ],
            }])
            .break_time(300)
            .prosody(Some(0.9), None, None, "Let me explain the features.")
            .build();

        assert!(ssml.contains("<p>"));
        assert!(ssml.contains("<emphasis level=\"strong\">OxiMedia</emphasis>"));
        assert!(ssml.contains("<break time=\"300ms\"/>"));
        assert!(ssml.contains("rate=\"0.9\""));
    }
}
