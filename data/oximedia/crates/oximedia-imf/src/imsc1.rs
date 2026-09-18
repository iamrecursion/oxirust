//! IMSC1 (Internet Media Subtitles and Captions) handling
//!
//! Implements reading and writing of TTML-based subtitles as defined by
//! W3C TTML and the IMSC1 profile (W3C TTML Profiles for Internet Media Subtitles
//! and Captions 1.0.1).

/// A rectangular region in the IMSC1 document, expressed as normalized (0.0–1.0) coordinates
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct ImscRegion {
    /// Unique region identifier
    pub id: String,
    /// Width and height as fractions of the root container (0.0–1.0)
    pub extent: (f32, f32),
    /// X and Y offset from the top-left as fractions of the root container (0.0–1.0)
    pub origin: (f32, f32),
}

impl ImscRegion {
    /// Create a new `ImscRegion`
    #[must_use]
    pub fn new(id: String, extent: (f32, f32), origin: (f32, f32)) -> Self {
        Self { id, extent, origin }
    }
}

/// A span of text with optional styling and timing
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct ImscSpan {
    /// The visible text
    pub text: String,
    /// Optional reference to a style in the document
    pub style_id: Option<String>,
    /// Presentation begin time in milliseconds
    pub begin_ms: u64,
    /// Presentation end time in milliseconds
    pub end_ms: u64,
}

impl ImscSpan {
    /// Create a new `ImscSpan`
    #[must_use]
    pub fn new(text: String, style_id: Option<String>, begin_ms: u64, end_ms: u64) -> Self {
        Self {
            text,
            style_id,
            begin_ms,
            end_ms,
        }
    }
}

/// A paragraph element containing one or more spans
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct ImscParagraph {
    /// Unique paragraph identifier
    pub id: String,
    /// Reference to the region where this paragraph is displayed
    pub region_id: String,
    /// The spans contained in this paragraph
    pub spans: Vec<ImscSpan>,
    /// Overall begin time in milliseconds
    pub begin_ms: u64,
    /// Overall end time in milliseconds
    pub end_ms: u64,
}

impl ImscParagraph {
    /// Create a new `ImscParagraph`
    #[must_use]
    pub fn new(id: String, region_id: String, begin_ms: u64, end_ms: u64) -> Self {
        Self {
            id,
            region_id,
            spans: Vec::new(),
            begin_ms,
            end_ms,
        }
    }

    /// Add a span to this paragraph
    pub fn add_span(&mut self, span: ImscSpan) {
        self.spans.push(span);
    }
}

/// A complete IMSC1 document
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ImscDocument {
    /// Title of the document
    pub title: String,
    /// Frame rate used for timecode calculations
    pub frame_rate: u32,
    /// Regions defined in this document
    pub regions: Vec<ImscRegion>,
    /// Paragraphs (subtitle cues) in this document
    pub paragraphs: Vec<ImscParagraph>,
}

impl ImscDocument {
    /// Create a new empty `ImscDocument`
    #[must_use]
    pub fn new(title: String, frame_rate: u32) -> Self {
        Self {
            title,
            frame_rate,
            regions: Vec::new(),
            paragraphs: Vec::new(),
        }
    }

    /// Add a region to this document
    pub fn add_region(&mut self, region: ImscRegion) {
        self.regions.push(region);
    }

    /// Add a paragraph to this document
    pub fn add_paragraph(&mut self, paragraph: ImscParagraph) {
        self.paragraphs.push(paragraph);
    }

    /// Return all paragraphs that are active (visible) at the given time in milliseconds
    #[must_use]
    pub fn active_at(&self, time_ms: u64) -> Vec<&ImscParagraph> {
        self.paragraphs
            .iter()
            .filter(|p| p.begin_ms <= time_ms && time_ms < p.end_ms)
            .collect()
    }
}

/// Serializes an `ImscDocument` to TTML XML
#[allow(dead_code)]
pub struct ImscSerializer;

impl ImscSerializer {
    /// Generate a TTML XML string from the given document
    #[must_use]
    pub fn to_ttml(doc: &ImscDocument) -> String {
        let mut out = String::new();

        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<tt xmlns=\"http://www.w3.org/ns/ttml\"\n");
        out.push_str("    xmlns:tts=\"http://www.w3.org/ns/ttml#styling\"\n");
        out.push_str(&format!(
            "    xml:lang=\"en\" frameRate=\"{}\">\n",
            doc.frame_rate
        ));

        // Head
        out.push_str("  <head>\n");
        out.push_str("    <metadata>\n");
        out.push_str(&format!(
            "      <ttm:title>{}</ttm:title>\n",
            escape_xml(&doc.title)
        ));
        out.push_str("    </metadata>\n");
        out.push_str("    <layout>\n");
        for region in &doc.regions {
            out.push_str(&format!(
                "      <region xml:id=\"{}\" tts:origin=\"{:.4}% {:.4}%\" tts:extent=\"{:.4}% {:.4}%\"/>\n",
                region.id,
                region.origin.0 * 100.0,
                region.origin.1 * 100.0,
                region.extent.0 * 100.0,
                region.extent.1 * 100.0,
            ));
        }
        out.push_str("    </layout>\n");
        out.push_str("  </head>\n");

        // Body
        out.push_str("  <body>\n");
        out.push_str("    <div>\n");
        for para in &doc.paragraphs {
            out.push_str(&format!(
                "      <p xml:id=\"{}\" region=\"{}\" begin=\"{}\" end=\"{}\">\n",
                para.id,
                para.region_id,
                ms_to_ttml_time(para.begin_ms),
                ms_to_ttml_time(para.end_ms),
            ));
            for span in &para.spans {
                if let Some(ref style) = span.style_id {
                    out.push_str(&format!(
                        "        <span style=\"{}\">{}</span>\n",
                        style,
                        escape_xml(&span.text)
                    ));
                } else {
                    out.push_str(&format!(
                        "        <span>{}</span>\n",
                        escape_xml(&span.text)
                    ));
                }
            }
            out.push_str("      </p>\n");
        }
        out.push_str("    </div>\n");
        out.push_str("  </body>\n");
        out.push_str("</tt>\n");

        out
    }
}

/// Parses TTML XML into an `ImscDocument`
#[allow(dead_code)]
pub struct ImscParser;

impl ImscParser {
    /// Parse a TTML XML string into an `ImscDocument`.
    ///
    /// Returns `Err(String)` with a description if parsing fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the XML is malformed or missing required elements.
    pub fn from_ttml(xml: &str) -> Result<ImscDocument, String> {
        // Extract title
        let title = extract_between(xml, "<ttm:title>", "</ttm:title>")
            .unwrap_or_default()
            .to_string();

        // Extract frameRate from tt element attribute
        let frame_rate = extract_attribute(xml, "frameRate")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(25);

        let mut doc = ImscDocument::new(title, frame_rate);

        // Parse regions
        for region_xml in find_elements(xml, "region") {
            let id = extract_attribute(&region_xml, "xml:id")
                .unwrap_or_default()
                .to_string();
            if id.is_empty() {
                continue;
            }
            let origin_str = extract_attribute(&region_xml, "tts:origin").unwrap_or("0% 0%");
            let extent_str = extract_attribute(&region_xml, "tts:extent").unwrap_or("100% 100%");
            let origin = parse_percent_pair(origin_str);
            let extent = parse_percent_pair(extent_str);
            doc.add_region(ImscRegion::new(id, extent, origin));
        }

        // Parse paragraphs
        for (idx, para_xml) in find_elements(xml, "p").iter().enumerate() {
            let id = extract_attribute(para_xml, "xml:id")
                .map(String::from)
                .unwrap_or_else(|| format!("p{idx}"));
            let region_id = extract_attribute(para_xml, "region")
                .unwrap_or_default()
                .to_string();
            let begin_ms = extract_attribute(para_xml, "begin")
                .map(ttml_time_to_ms)
                .unwrap_or(0);
            let end_ms = extract_attribute(para_xml, "end")
                .map(ttml_time_to_ms)
                .unwrap_or(0);

            let mut para = ImscParagraph::new(id, region_id, begin_ms, end_ms);

            // Parse spans inside paragraph
            for span_xml in find_elements(para_xml, "span") {
                let style_id = extract_attribute(&span_xml, "style").map(String::from);
                let text = strip_tags(&span_xml);
                para.add_span(ImscSpan::new(text, style_id, begin_ms, end_ms));
            }

            doc.add_paragraph(para);
        }

        Ok(doc)
    }
}

// ---- Helper functions -------------------------------------------------------

/// Convert milliseconds to TTML time format (HH:MM:SS.mmm)
fn ms_to_ttml_time(ms: u64) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let ms_rem = ms % 1_000;
    format!("{h:02}:{m:02}:{s:02}.{ms_rem:03}")
}

/// Convert TTML time string to milliseconds.
/// Supports formats: HH:MM:SS.mmm and HH:MM:SS:FF (drop-frame ignored)
fn ttml_time_to_ms(s: &str) -> u64 {
    let parts: Vec<&str> = s.splitn(4, ':').collect();
    if parts.len() < 3 {
        return 0;
    }
    let h: u64 = parts[0].parse().unwrap_or(0);
    let m: u64 = parts[1].parse().unwrap_or(0);
    let s_and_ms: Vec<&str> = parts[2].splitn(2, '.').collect();
    let sec: u64 = s_and_ms[0].parse().unwrap_or(0);
    let ms_frac: u64 = if s_and_ms.len() > 1 {
        let frac_str = s_and_ms[1];
        // Normalise to milliseconds (3 digits)
        let padded = format!("{frac_str:0<3}");
        padded[..3.min(padded.len())].parse().unwrap_or(0)
    } else {
        0
    };
    h * 3_600_000 + m * 60_000 + sec * 1_000 + ms_frac
}

/// Escape XML special characters
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Extract text between two literal delimiters (first occurrence)
fn extract_between<'a>(haystack: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let start_pos = haystack.find(start)?;
    let after_start = start_pos + start.len();
    let end_pos = haystack[after_start..].find(end)?;
    Some(&haystack[after_start..after_start + end_pos])
}

/// Extract the value of a named attribute from an XML element string
fn extract_attribute<'a>(xml: &'a str, attr: &str) -> Option<&'a str> {
    let search = format!("{attr}=\"");
    let start = xml.find(search.as_str())? + search.len();
    let end = xml[start..].find('"')?;
    Some(&xml[start..start + end])
}

/// Find all occurrences of an element by tag name (simple, non-nested extraction)
fn find_elements(xml: &str, tag: &str) -> Vec<String> {
    let mut results = Vec::new();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut pos = 0;

    while let Some(start) = xml[pos..].find(open.as_str()) {
        let abs_start = pos + start;
        // Find the end of this element
        if let Some(end_rel) = xml[abs_start..].find(close.as_str()) {
            let abs_end = abs_start + end_rel + close.len();
            results.push(xml[abs_start..abs_end].to_string());
            pos = abs_end;
        } else if let Some(self_close) = xml[abs_start..].find("/>") {
            let abs_end = abs_start + self_close + 2;
            results.push(xml[abs_start..abs_end].to_string());
            pos = abs_end;
        } else {
            break;
        }
    }

    results
}

/// Parse a pair like "10.00% 20.00%" into (0.1, 0.2)
fn parse_percent_pair(s: &str) -> (f32, f32) {
    let parts: Vec<&str> = s.split_whitespace().collect();
    let parse_pct = |p: &str| p.trim_end_matches('%').parse::<f32>().unwrap_or(0.0) / 100.0;
    let x = parts.first().map_or(0.0, |p| parse_pct(p));
    let y = parts.get(1).map_or(0.0, |p| parse_pct(p));
    (x, y)
}

/// Strip XML tags, returning inner text content
fn strip_tags(xml: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for ch in xml.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => result.push(c),
            _ => {}
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imsc_region_creation() {
        let region = ImscRegion::new("r1".to_string(), (0.8, 0.1), (0.1, 0.8));
        assert_eq!(region.id, "r1");
        assert!((region.extent.0 - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_imsc_span_creation() {
        let span = ImscSpan::new("Hello".to_string(), None, 1000, 3000);
        assert_eq!(span.text, "Hello");
        assert_eq!(span.begin_ms, 1000);
        assert_eq!(span.end_ms, 3000);
    }

    #[test]
    fn test_imsc_paragraph_creation() {
        let mut para = ImscParagraph::new("p1".to_string(), "r1".to_string(), 0, 5000);
        para.add_span(ImscSpan::new("Hello World".to_string(), None, 0, 5000));
        assert_eq!(para.spans.len(), 1);
        assert_eq!(para.begin_ms, 0);
        assert_eq!(para.end_ms, 5000);
    }

    #[test]
    fn test_document_active_at() {
        let mut doc = ImscDocument::new("Test".to_string(), 25);
        doc.add_paragraph(ImscParagraph::new(
            "p1".to_string(),
            "r1".to_string(),
            1000,
            4000,
        ));
        doc.add_paragraph(ImscParagraph::new(
            "p2".to_string(),
            "r1".to_string(),
            5000,
            8000,
        ));

        let active = doc.active_at(2000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "p1");

        let active2 = doc.active_at(6000);
        assert_eq!(active2.len(), 1);
        assert_eq!(active2[0].id, "p2");

        let active3 = doc.active_at(4500);
        assert!(active3.is_empty());
    }

    #[test]
    fn test_document_active_at_boundary() {
        let mut doc = ImscDocument::new("Test".to_string(), 25);
        doc.add_paragraph(ImscParagraph::new(
            "p1".to_string(),
            "r1".to_string(),
            1000,
            4000,
        ));

        // begin is inclusive, end is exclusive
        assert_eq!(doc.active_at(1000).len(), 1);
        assert_eq!(doc.active_at(3999).len(), 1);
        assert_eq!(doc.active_at(4000).len(), 0);
    }

    #[test]
    fn test_ms_to_ttml_time() {
        assert_eq!(ms_to_ttml_time(0), "00:00:00.000");
        assert_eq!(ms_to_ttml_time(1000), "00:00:01.000");
        assert_eq!(ms_to_ttml_time(61_500), "00:01:01.500");
        assert_eq!(ms_to_ttml_time(3_661_200), "01:01:01.200");
    }

    #[test]
    fn test_ttml_time_to_ms() {
        assert_eq!(ttml_time_to_ms("00:00:01.000"), 1000);
        assert_eq!(ttml_time_to_ms("00:01:01.500"), 61_500);
        assert_eq!(ttml_time_to_ms("01:01:01.200"), 3_661_200);
    }

    #[test]
    fn test_serializer_output_structure() {
        let mut doc = ImscDocument::new("My Subs".to_string(), 25);
        doc.add_region(ImscRegion::new("r1".to_string(), (0.8, 0.1), (0.1, 0.8)));
        let mut para = ImscParagraph::new("p1".to_string(), "r1".to_string(), 1000, 3000);
        para.add_span(ImscSpan::new("Hello".to_string(), None, 1000, 3000));
        doc.add_paragraph(para);

        let ttml = ImscSerializer::to_ttml(&doc);

        assert!(ttml.contains("<?xml"));
        assert!(ttml.contains("<tt"));
        assert!(ttml.contains("My Subs"));
        assert!(ttml.contains("frameRate=\"25\""));
        assert!(ttml.contains("xml:id=\"r1\""));
        assert!(ttml.contains("xml:id=\"p1\""));
        assert!(ttml.contains("Hello"));
        assert!(ttml.contains("00:00:01.000"));
    }

    #[test]
    fn test_parser_round_trip() {
        let mut doc = ImscDocument::new("RoundTrip".to_string(), 30);
        doc.add_region(ImscRegion::new("r1".to_string(), (1.0, 0.2), (0.0, 0.8)));
        let mut para = ImscParagraph::new("p1".to_string(), "r1".to_string(), 2000, 5000);
        para.add_span(ImscSpan::new("Test subtitle".to_string(), None, 2000, 5000));
        doc.add_paragraph(para);

        let ttml = ImscSerializer::to_ttml(&doc);
        let parsed = ImscParser::from_ttml(&ttml).expect("Failed to parse TTML");

        assert_eq!(parsed.title, "RoundTrip");
        assert_eq!(parsed.frame_rate, 30);
        assert!(!parsed.regions.is_empty());
        assert_eq!(parsed.paragraphs.len(), 1);
        assert_eq!(parsed.paragraphs[0].begin_ms, 2000);
        assert_eq!(parsed.paragraphs[0].end_ms, 5000);
    }

    #[test]
    fn test_xml_escaping() {
        let escaped = escape_xml("<Hello & \"World\">");
        assert!(!escaped.contains('<'));
        assert!(!escaped.contains('>'));
        assert!(escaped.contains("&lt;"));
        assert!(escaped.contains("&amp;"));
        assert!(escaped.contains("&gt;"));
    }
}
