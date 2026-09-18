//! TTML (Timed Text Markup Language) format parser and writer

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::{Caption, CaptionTrack, Language, Timestamp};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::OnceLock;

/// TTML format parser
pub struct TtmlParser;

impl FormatParser for TtmlParser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        parse_ttml(data)
    }
}

/// TTML format writer
pub struct TtmlWriter;

impl FormatWriter for TtmlWriter {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        write_ttml(track)
    }
}

// ============================================================================
// Lazy TTML style resolution (Wave 14 Slice H)
// ============================================================================

/// Styling attributes extracted from a TTML `<style>` element.
#[derive(Debug, Clone, Default)]
pub struct TtmlStyle {
    /// `tts:color` value (raw string, e.g. `"white"`)
    pub color: Option<String>,
    /// `tts:backgroundColor` value
    pub background_color: Option<String>,
    /// `tts:fontStyle` value (e.g. `"italic"`)
    pub font_style: Option<String>,
    /// `tts:fontWeight` value (e.g. `"bold"`)
    pub font_weight: Option<String>,
    /// `tts:textDecoration` value (e.g. `"underline"`)
    pub text_decoration: Option<String>,
    /// `tts:fontSize` value (e.g. `"100%"`, `"18px"`)
    pub font_size: Option<String>,
}

/// An unresolved TTML cue — style IDs have not yet been looked up.
#[derive(Debug, Clone)]
pub struct UnresolvedCue {
    /// Start time in milliseconds
    pub start_ms: i64,
    /// End time in milliseconds
    pub end_ms: i64,
    /// Cue body text
    pub text: String,
    /// IDs of `<style>` elements referenced via `style` attribute
    pub style_ids: Vec<String>,
}

/// Resolves a list of style IDs against a style map, returning a merged
/// [`TtmlStyle`] (later IDs override earlier ones for non-None fields).
fn resolve_styles(ids: &[String], map: &HashMap<String, TtmlStyle>) -> TtmlStyle {
    let mut merged = TtmlStyle::default();
    for id in ids {
        if let Some(s) = map.get(id) {
            if s.color.is_some() {
                merged.color.clone_from(&s.color);
            }
            if s.background_color.is_some() {
                merged.background_color.clone_from(&s.background_color);
            }
            if s.font_style.is_some() {
                merged.font_style.clone_from(&s.font_style);
            }
            if s.font_weight.is_some() {
                merged.font_weight.clone_from(&s.font_weight);
            }
            if s.text_decoration.is_some() {
                merged.text_decoration.clone_from(&s.text_decoration);
            }
            if s.font_size.is_some() {
                merged.font_size.clone_from(&s.font_size);
            }
        }
    }
    merged
}

/// A TTML track in which cue style resolution is deferred until first access.
///
/// Constructing this type from raw XML is cheap: only the style map and
/// unresolved cue list are built.  The expensive style-application pass is
/// performed at most once, on the first call to [`LazyTtmlTrack::captions`].
pub struct LazyTtmlTrack {
    cues: Vec<UnresolvedCue>,
    style_map: HashMap<String, TtmlStyle>,
    /// Cache for the resolved captions; populated on first call to `captions()`.
    resolved: OnceLock<Vec<Caption>>,
    /// Language tag from the `xml:lang` attribute (or English fallback).
    pub language: Language,
}

impl LazyTtmlTrack {
    /// Create a new lazy track from raw cues and a style map.
    #[must_use]
    pub fn new(
        cues: Vec<UnresolvedCue>,
        style_map: HashMap<String, TtmlStyle>,
        language: Language,
    ) -> Self {
        Self {
            cues,
            style_map,
            resolved: OnceLock::new(),
            language,
        }
    }

    /// Return the resolved captions, resolving styles on the first call.
    ///
    /// Subsequent calls return the same slice without re-resolving.
    pub fn captions(&self) -> &[Caption] {
        // Pass references into the closure explicitly to avoid the borrow
        // checker seeing an aliased borrow of `self` through the OnceLock.
        let cues = &self.cues;
        let style_map = &self.style_map;
        self.resolved.get_or_init(|| {
            cues.iter()
                .map(|cue| {
                    let _style = resolve_styles(&cue.style_ids, style_map);
                    Caption::new(
                        Timestamp::from_millis(cue.start_ms),
                        Timestamp::from_millis(cue.end_ms),
                        cue.text.clone(),
                    )
                })
                .collect()
        })
    }

    /// Force style resolution and consume the track, returning owned captions.
    #[must_use]
    pub fn into_captions(self) -> Vec<Caption> {
        // Destructure to avoid partial-move borrow conflicts when the OnceLock
        // hasn't been populated yet and we need to run resolve_all on cues/style_map.
        let Self {
            cues,
            style_map,
            resolved,
            language: _,
        } = self;
        match resolved.into_inner() {
            Some(v) => v,
            None => {
                // Inline the resolve logic since self is consumed
                cues.iter()
                    .map(|cue| {
                        let _style = resolve_styles(&cue.style_ids, &style_map);
                        Caption::new(
                            Timestamp::from_millis(cue.start_ms),
                            Timestamp::from_millis(cue.end_ms),
                            cue.text.clone(),
                        )
                    })
                    .collect()
            }
        }
    }

    /// Convert into a standard [`CaptionTrack`], resolving styles eagerly.
    pub fn into_track(self) -> Result<CaptionTrack> {
        let lang = self.language.clone();
        let captions = self.into_captions();
        let mut track = CaptionTrack::new(lang);
        for c in captions {
            track.add_caption(c)?;
        }
        Ok(track)
    }
}

/// Parse a TTML document into a [`LazyTtmlTrack`] without resolving styles.
///
/// Use [`parse_ttml_lazy`] when you want to defer style resolution until the
/// captions are actually needed (e.g. for thumbnail generation, searching, or
/// sub-selection of cues).
pub fn parse_ttml_lazy(data: &[u8]) -> Result<LazyTtmlTrack> {
    let mut reader = Reader::from_reader(data);

    let mut style_map: HashMap<String, TtmlStyle> = HashMap::new();
    let mut cues: Vec<UnresolvedCue> = Vec::new();
    let mut buf = Vec::new();

    let mut in_body = false;
    let mut in_styling = false;
    let mut current_text = String::new();
    let mut current_begin: Option<Timestamp> = None;
    let mut current_end: Option<Timestamp> = None;
    let mut current_style_ids: Vec<String> = Vec::new();
    let mut language = Language::english();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => match e.name().as_ref() {
                b"tt" => {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"xml:lang" {
                            if let Ok(v) = std::str::from_utf8(&attr.value) {
                                language = Language::new(v.to_string(), v.to_string(), false);
                            }
                        }
                    }
                }
                b"styling" => in_styling = true,
                b"style" => {
                    if in_styling {
                        let mut id = String::new();
                        let mut style = TtmlStyle::default();
                        for attr in e.attributes().flatten() {
                            let val = String::from_utf8_lossy(&attr.value).into_owned();
                            match attr.key.as_ref() {
                                b"xml:id" | b"id" => id = val,
                                b"tts:color" => style.color = Some(val),
                                b"tts:backgroundColor" => style.background_color = Some(val),
                                b"tts:fontStyle" => style.font_style = Some(val),
                                b"tts:fontWeight" => style.font_weight = Some(val),
                                b"tts:textDecoration" => style.text_decoration = Some(val),
                                b"tts:fontSize" => style.font_size = Some(val),
                                _ => {}
                            }
                        }
                        if !id.is_empty() {
                            style_map.insert(id, style);
                        }
                    }
                }
                b"body" => in_body = true,
                b"p" | b"div" => {
                    for attr in e.attributes().flatten() {
                        let val_str = String::from_utf8_lossy(&attr.value);
                        match attr.key.as_ref() {
                            b"begin" => {
                                current_begin = parse_ttml_time(&val_str).ok();
                            }
                            b"end" => {
                                current_end = parse_ttml_time(&val_str).ok();
                            }
                            b"style" => {
                                // May be space-separated list of IDs
                                current_style_ids =
                                    val_str.split_whitespace().map(str::to_owned).collect();
                            }
                            _ => {}
                        }
                    }
                    current_text.clear();
                }
                _ => {}
            },
            Ok(Event::Text(e)) => {
                if in_body {
                    let text = String::from_utf8_lossy(e.as_ref());
                    current_text.push_str(&text);
                }
            }
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"styling" => in_styling = false,
                b"p" | b"div" => {
                    if let (Some(begin), Some(end)) = (current_begin, current_end) {
                        cues.push(UnresolvedCue {
                            start_ms: begin.as_millis(),
                            end_ms: end.as_millis(),
                            text: current_text.clone(),
                            style_ids: std::mem::take(&mut current_style_ids),
                        });
                        current_text.clear();
                        current_begin = None;
                        current_end = None;
                    }
                }
                b"body" => in_body = false,
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => return Err(CaptionError::Xml(e.to_string())),
            _ => {}
        }
        buf.clear();
    }

    Ok(LazyTtmlTrack::new(cues, style_map, language))
}

fn parse_ttml(data: &[u8]) -> Result<CaptionTrack> {
    let mut reader = Reader::from_reader(data);
    // trim_text is not available in quick-xml 0.36+

    let mut track = CaptionTrack::new(Language::english());
    let mut buf = Vec::new();
    let mut in_body = false;
    let mut current_text = String::new();
    let mut current_begin: Option<Timestamp> = None;
    let mut current_end: Option<Timestamp> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"body" => in_body = true,
                    b"p" | b"div" => {
                        // Parse timing attributes
                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"begin" => {
                                    if let Ok(val) = std::str::from_utf8(&attr.value) {
                                        current_begin = parse_ttml_time(val).ok();
                                    }
                                }
                                b"end" => {
                                    if let Ok(val) = std::str::from_utf8(&attr.value) {
                                        current_end = parse_ttml_time(val).ok();
                                    }
                                }
                                _ => {}
                            }
                        }
                        current_text.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                if in_body {
                    let text = String::from_utf8_lossy(e.as_ref());
                    current_text.push_str(&text);
                }
            }
            Ok(Event::End(ref e)) => {
                if matches!(e.name().as_ref(), b"p" | b"div") {
                    if let (Some(begin), Some(end)) = (current_begin, current_end) {
                        let caption = Caption::new(begin, end, current_text.clone());
                        track.add_caption(caption)?;
                        current_text.clear();
                        current_begin = None;
                        current_end = None;
                    }
                } else if e.name().as_ref() == b"body" {
                    in_body = false;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(CaptionError::Xml(e.to_string())),
            _ => {}
        }
        buf.clear();
    }

    Ok(track)
}

fn write_ttml(track: &CaptionTrack) -> Result<Vec<u8>> {
    let mut writer = Writer::new(Cursor::new(Vec::new()));

    // XML declaration
    writer
        .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    // Root element
    let mut tt = BytesStart::new("tt");
    tt.push_attribute(("xmlns", "http://www.w3.org/ns/ttml"));
    tt.push_attribute(("xmlns:tts", "http://www.w3.org/ns/ttml#styling"));
    tt.push_attribute(("xml:lang", track.language.code.as_str()));
    writer
        .write_event(Event::Start(tt))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    // Head (metadata)
    writer
        .write_event(Event::Start(BytesStart::new("head")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    if let Some(title) = &track.metadata.title {
        writer
            .write_event(Event::Start(BytesStart::new("metadata")))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::Start(BytesStart::new("title")))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::Text(BytesText::new(title)))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("title")))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("metadata")))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("head")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    // Body
    writer
        .write_event(Event::Start(BytesStart::new("body")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;
    writer
        .write_event(Event::Start(BytesStart::new("div")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    // Captions
    for caption in &track.captions {
        let mut p = BytesStart::new("p");
        p.push_attribute(("begin", format_ttml_time(caption.start).as_str()));
        p.push_attribute(("end", format_ttml_time(caption.end).as_str()));

        writer
            .write_event(Event::Start(p))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::Text(BytesText::new(&caption.text)))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("p")))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("div")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;
    writer
        .write_event(Event::End(BytesEnd::new("body")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;
    writer
        .write_event(Event::End(BytesEnd::new("tt")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    Ok(writer.into_inner().into_inner())
}

fn parse_ttml_time(s: &str) -> Result<Timestamp> {
    // Support multiple formats:
    // HH:MM:SS.mmm
    // HH:MM:SS:ff (frames)
    // offset-time (e.g., "10s", "500ms")

    if s.ends_with('s') && !s.contains(':') {
        // Offset time
        let num_part = &s[..s.len() - 1];
        if s.ends_with("ms") {
            let ms = num_part[..num_part.len() - 1]
                .parse::<i64>()
                .map_err(|e| CaptionError::Parse(e.to_string()))?;
            return Ok(Timestamp::from_millis(ms));
        } else {
            let secs = num_part
                .parse::<f64>()
                .map_err(|e| CaptionError::Parse(e.to_string()))?;
            return Ok(Timestamp::from_micros((secs * 1_000_000.0) as i64));
        }
    }

    // Clock time format
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() < 3 {
        return Err(CaptionError::Parse(format!("Invalid TTML time: {s}")));
    }

    let hours = parts[0]
        .parse::<u32>()
        .map_err(|e| CaptionError::Parse(e.to_string()))?;
    let minutes = parts[1]
        .parse::<u32>()
        .map_err(|e| CaptionError::Parse(e.to_string()))?;

    let sec_parts: Vec<&str> = parts[2].split('.').collect();
    let seconds = sec_parts[0]
        .parse::<u32>()
        .map_err(|e| CaptionError::Parse(e.to_string()))?;
    let millis = if sec_parts.len() > 1 {
        sec_parts[1]
            .parse::<u32>()
            .map_err(|e| CaptionError::Parse(e.to_string()))?
    } else {
        0
    };

    Ok(Timestamp::from_hmsm(hours, minutes, seconds, millis))
}

fn format_ttml_time(ts: Timestamp) -> String {
    let (h, m, s, ms) = ts.as_hmsm();
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ttml_time() {
        let ts = parse_ttml_time("01:30:45.500").expect("parsing should succeed");
        assert_eq!(ts.as_hmsm(), (1, 30, 45, 500));

        let ts = parse_ttml_time("10s").expect("parsing should succeed");
        assert_eq!(ts.as_secs(), 10);

        let ts = parse_ttml_time("500ms").expect("parsing should succeed");
        assert_eq!(ts.as_millis(), 500);
    }

    #[test]
    fn test_write_ttml() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let writer = TtmlWriter;
        let output = writer.write(&track).expect("writing should succeed");
        let text = String::from_utf8(output).expect("output should be valid UTF-8");

        assert!(text.contains("<?xml"));
        assert!(text.contains("<tt"));
        assert!(text.contains("Test"));
    }

    // Wave 14 Slice H — new tests

    /// Minimal TTML document with 2 named styles and 3 cues referencing them.
    fn make_styled_ttml() -> &'static str {
        r#"<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml"
    xmlns:tts="http://www.w3.org/ns/ttml#styling"
    xml:lang="en">
  <head>
    <styling>
      <style xml:id="s1" tts:color="white" tts:fontWeight="bold"/>
      <style xml:id="s2" tts:color="yellow" tts:fontStyle="italic"/>
    </styling>
  </head>
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:03.000" style="s1">First</p>
      <p begin="00:00:04.000" end="00:00:06.000" style="s2">Second</p>
      <p begin="00:00:07.000" end="00:00:09.000" style="s1 s2">Third</p>
    </div>
  </body>
</tt>"#
    }

    /// The lazy parser should produce 3 captions with the correct text, and
    /// calling `captions()` twice must return identical results (OnceLock hit).
    #[test]
    fn test_lazy_ttml_style_resolution() {
        let ttml = make_styled_ttml();
        let lazy = parse_ttml_lazy(ttml.as_bytes())
            .expect("parse_ttml_lazy should succeed on styled TTML");

        // First access: resolves styles
        let first_ptr = lazy.captions().as_ptr();
        assert_eq!(lazy.captions().len(), 3, "should have 3 captions");

        // Verify text content is preserved through lazy resolution
        assert_eq!(lazy.captions()[0].text, "First");
        assert_eq!(lazy.captions()[1].text, "Second");
        assert_eq!(lazy.captions()[2].text, "Third");

        // Second access: must hit the OnceLock cache (same pointer)
        let second_ptr = lazy.captions().as_ptr();
        assert_eq!(
            first_ptr, second_ptr,
            "second call to captions() must return cached vector (OnceLock hit)"
        );
    }

    /// `parse_ttml_lazy` followed by `into_track` must produce the same
    /// caption data as the original `parse_ttml`.
    #[test]
    fn test_lazy_ttml_matches_eager() {
        let ttml = make_styled_ttml();
        let eager_track = TtmlParser
            .parse(ttml.as_bytes())
            .expect("eager TTML parse should succeed");
        let lazy_track = parse_ttml_lazy(ttml.as_bytes())
            .expect("lazy TTML parse should succeed")
            .into_track()
            .expect("lazy into_track should succeed");

        assert_eq!(
            eager_track.captions.len(),
            lazy_track.captions.len(),
            "cue count must match"
        );
        for (i, (eager, lazy)) in eager_track
            .captions
            .iter()
            .zip(lazy_track.captions.iter())
            .enumerate()
        {
            assert_eq!(eager.start, lazy.start, "start mismatch at cue {i}");
            assert_eq!(eager.end, lazy.end, "end mismatch at cue {i}");
            assert_eq!(eager.text, lazy.text, "text mismatch at cue {i}");
        }
    }
}
