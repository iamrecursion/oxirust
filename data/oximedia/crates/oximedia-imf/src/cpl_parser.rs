//! CPL (Composition Playlist) structural parsing helpers.
//!
//! Provides lightweight, allocation-friendly types for representing the
//! segment/sequence hierarchy of an IMF Composition Playlist (SMPTE ST 2067-3)
//! without requiring a full XML round-trip.

/// A single resource reference within a CPL sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CplResource {
    /// IETF RFC 4122 UUID identifying the track-file asset.
    pub track_file_id: String,
    /// Edit-unit offset within the referenced track file.
    pub source_duration: u64,
    /// Number of edit units from this resource to include.
    pub entry_point: u64,
    /// Intrinsic duration of the resource in edit units.
    pub intrinsic_duration: u64,
    /// Repeat count (usually 1).
    pub repeat_count: u32,
}

impl CplResource {
    /// Create a minimal [`CplResource`] pointing at `track_file_id` for
    /// `duration` edit units starting from the beginning.
    #[must_use]
    pub fn simple(track_file_id: impl Into<String>, duration: u64) -> Self {
        Self {
            track_file_id: track_file_id.into(),
            source_duration: duration,
            entry_point: 0,
            intrinsic_duration: duration,
            repeat_count: 1,
        }
    }

    /// Effective duration contributed by this resource.
    #[must_use]
    pub fn effective_duration(&self) -> u64 {
        self.source_duration * u64::from(self.repeat_count)
    }
}

/// A CPL sequence groups resources of the same type (video, audio, subtitle …).
#[derive(Debug, Clone)]
pub struct CplSequence {
    /// UUID of this sequence.
    pub id: String,
    /// UUID of the virtual track to which this sequence belongs.
    pub track_id: String,
    /// Ordered list of resource references.
    pub resources: Vec<CplResource>,
}

impl CplSequence {
    /// Create a new empty [`CplSequence`].
    #[must_use]
    pub fn new(id: impl Into<String>, track_id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            track_id: track_id.into(),
            resources: Vec::new(),
        }
    }

    /// Total edit-unit duration of this sequence.
    #[must_use]
    pub fn total_duration(&self) -> u64 {
        self.resources
            .iter()
            .map(CplResource::effective_duration)
            .sum()
    }

    /// Append a resource to the sequence.
    pub fn add_resource(&mut self, resource: CplResource) {
        self.resources.push(resource);
    }
}

/// A CPL segment groups simultaneously playing sequences.
///
/// Each segment corresponds to one `<Segment>` element in the CPL XML.
#[derive(Debug, Clone)]
pub struct CplSegment {
    /// UUID of this segment.
    pub id: String,
    /// Human-readable annotation label (optional in SMPTE ST 2067-3).
    pub annotation: Option<String>,
    /// All sequences within this segment.
    pub sequences: Vec<CplSequence>,
}

impl CplSegment {
    /// Create a new [`CplSegment`] with no sequences.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            annotation: None,
            sequences: Vec::new(),
        }
    }

    /// Append a sequence to this segment.
    pub fn add_sequence(&mut self, seq: CplSequence) {
        self.sequences.push(seq);
    }

    /// Maximum edit-unit duration across all sequences in this segment.
    ///
    /// Per SMPTE ST 2067-3 all sequences in a segment must have the same
    /// duration; this method returns the maximum as a guard against malformed
    /// data.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.sequences
            .iter()
            .map(CplSequence::total_duration)
            .max()
            .unwrap_or(0)
    }
}

/// In-memory representation of a Composition Playlist.
///
/// This is a pure Rust data type suitable for building CPLs programmatically
/// or as an intermediate representation after XML parsing.
#[derive(Debug, Clone)]
pub struct CompositionPlaylist {
    /// CPL UUID.
    pub id: String,
    /// Human-readable title.
    pub content_title: String,
    /// Edit rate as a `(numerator, denominator)` fraction.
    pub edit_rate: (u32, u32),
    /// Ordered list of segments.
    segments: Vec<CplSegment>,
}

impl CompositionPlaylist {
    /// Create a new empty [`CompositionPlaylist`].
    ///
    /// # Arguments
    /// * `id`            – CPL UUID string.
    /// * `content_title` – Human-readable title.
    /// * `edit_rate`     – `(numerator, denominator)` e.g. `(24, 1)`.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        content_title: impl Into<String>,
        edit_rate: (u32, u32),
    ) -> Self {
        Self {
            id: id.into(),
            content_title: content_title.into(),
            edit_rate,
            segments: Vec::new(),
        }
    }

    /// Append a segment to the composition.
    pub fn add_segment(&mut self, segment: CplSegment) {
        self.segments.push(segment);
    }

    /// Ordered slice of segments in this composition.
    #[must_use]
    pub fn segments(&self) -> &[CplSegment] {
        &self.segments
    }

    /// Total edit-unit duration (sum of all segment durations).
    #[must_use]
    pub fn total_duration(&self) -> u64 {
        self.segments.iter().map(CplSegment::duration).sum()
    }

    /// Total duration in seconds.
    #[must_use]
    pub fn total_duration_secs(&self) -> f64 {
        let (num, den) = self.edit_rate;
        if num == 0 {
            return 0.0;
        }
        self.total_duration() as f64 * den as f64 / num as f64
    }

    /// Number of segments in the composition.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns `true` when the composition has no segments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

// ── Lazy / section-targeted parsing ──────────────────────────────────────────

/// Header metadata extracted from a CPL without parsing the full segment tree.
#[derive(Debug, Clone, PartialEq)]
pub struct CplHeader {
    /// CPL UUID (without `urn:uuid:` prefix).
    pub id: String,
    /// Human-readable content title.
    pub content_title: String,
    /// Edit rate as `(numerator, denominator)`.
    pub edit_rate: (u32, u32),
}

/// A lightweight reel descriptor used by [`parse_reel_list`].
///
/// In the simplified CPL schema used by this crate the term "reel" maps
/// directly to a [`CplSegment`] (each `<Segment>` element).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reel {
    /// Segment UUID.
    pub id: String,
    /// Optional annotation / label.
    pub annotation: Option<String>,
}

/// Parse only the CPL header section (Id, ContentTitle, EditRate) using
/// `quick-xml` event iteration, skipping the SegmentList subtree entirely.
///
/// This is significantly faster than [`CompositionPlaylist::from_xml`] when
/// only header metadata is needed.
pub fn parse_cpl_header(xml: &str) -> Result<CplHeader, crate::ImfError> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();

    let mut id = String::new();
    let mut content_title = String::new();
    let mut edit_rate: (u32, u32) = (24, 1);

    // Track which top-level text element we are inside.
    #[derive(PartialEq)]
    enum Ctx {
        None,
        Id,
        ContentTitle,
        EditRate,
    }
    let mut ctx = Ctx::None;
    // Depth inside SegmentList — when > 0 we skip everything.
    let mut segment_depth: usize = 0;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if segment_depth == 0 => {
                let local = local_name(e.name().as_ref());
                match local.as_str() {
                    "SegmentList" => segment_depth += 1,
                    "Id" => ctx = Ctx::Id,
                    "ContentTitle" => ctx = Ctx::ContentTitle,
                    "EditRate" => ctx = Ctx::EditRate,
                    _ => ctx = Ctx::None,
                }
            }
            Ok(Event::Start(_)) => {
                // Inside SegmentList — count nesting but don't parse
                segment_depth += 1;
            }
            Ok(Event::End(ref e)) if segment_depth > 0 => {
                let local = local_name(e.name().as_ref());
                if local == "SegmentList" || segment_depth > 1 {
                    segment_depth = segment_depth.saturating_sub(1);
                } else {
                    segment_depth = 0;
                }
            }
            Ok(Event::End(_)) => {
                ctx = Ctx::None;
            }
            Ok(Event::Text(ref e)) if segment_depth == 0 => {
                let text = e
                    .decode()
                    .map_err(|err| crate::ImfError::XmlError(err.to_string()))?
                    .into_owned();
                match ctx {
                    Ctx::Id => {
                        id = text.trim_start_matches("urn:uuid:").to_string();
                        ctx = Ctx::None;
                    }
                    Ctx::ContentTitle => {
                        content_title = text;
                        ctx = Ctx::None;
                    }
                    Ctx::EditRate => {
                        let parts: Vec<&str> = text.split_whitespace().collect();
                        if parts.len() == 2 {
                            edit_rate = (
                                parts[0].parse().unwrap_or(24),
                                parts[1].parse().unwrap_or(1),
                            );
                        }
                        ctx = Ctx::None;
                    }
                    Ctx::None => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(crate::ImfError::XmlError(e.to_string()));
            }
            _ => {}
        }
        buf.clear();
    }

    if id.is_empty() {
        return Err(crate::ImfError::MissingElement("Id".to_string()));
    }

    Ok(CplHeader {
        id,
        content_title,
        edit_rate,
    })
}

/// Parse only the reel list (top-level `<Segment>` elements) from a CPL.
///
/// Segment content (SequenceList) is skipped; only `<Id>` and `<Annotation>`
/// are extracted per segment.
pub fn parse_reel_list(xml: &str) -> Result<Vec<Reel>, crate::ImfError> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();

    let mut reels: Vec<Reel> = Vec::new();
    let mut cur_reel: Option<Reel> = None;

    // Depth tracking:
    // 0 = outside SegmentList
    // 1 = inside SegmentList but outside any Segment
    // 2 = directly inside a Segment
    // 3+ = inside Segment children (SequenceList etc.) — skip
    let mut in_segment_list = false;
    let mut segment_depth: usize = 0;

    #[derive(PartialEq)]
    enum Ctx {
        None,
        Id,
        Annotation,
    }
    let mut ctx = Ctx::None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let local = local_name(e.name().as_ref());
                if !in_segment_list {
                    if local == "SegmentList" {
                        in_segment_list = true;
                    }
                    // skip everything outside SegmentList
                } else if segment_depth == 0 {
                    if local == "Segment" {
                        segment_depth = 1;
                        cur_reel = Some(Reel {
                            id: String::new(),
                            annotation: None,
                        });
                        ctx = Ctx::None;
                    }
                } else if segment_depth == 1 {
                    match local.as_str() {
                        "Id" => ctx = Ctx::Id,
                        "Annotation" => ctx = Ctx::Annotation,
                        _ => {
                            // SequenceList or other container — skip its subtree
                            segment_depth += 1;
                            ctx = Ctx::None;
                        }
                    }
                } else {
                    segment_depth += 1;
                    ctx = Ctx::None;
                }
            }
            Ok(Event::End(ref e)) if in_segment_list => {
                let local = local_name(e.name().as_ref());
                if local == "SegmentList" {
                    break;
                } else if local == "Segment" && segment_depth == 1 {
                    if let Some(reel) = cur_reel.take() {
                        reels.push(reel);
                    }
                    segment_depth = 0;
                    ctx = Ctx::None;
                } else if segment_depth > 1 {
                    segment_depth -= 1;
                    ctx = Ctx::None;
                } else {
                    ctx = Ctx::None;
                }
            }
            Ok(Event::Text(ref e)) if in_segment_list && segment_depth == 1 => {
                let text = e
                    .decode()
                    .map_err(|err| crate::ImfError::XmlError(err.to_string()))?
                    .into_owned();
                if let Some(ref mut reel) = cur_reel {
                    match ctx {
                        Ctx::Id => {
                            reel.id = text.trim_start_matches("urn:uuid:").to_string();
                        }
                        Ctx::Annotation => {
                            reel.annotation = Some(text);
                        }
                        Ctx::None => {}
                    }
                }
                ctx = Ctx::None;
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(crate::ImfError::XmlError(e.to_string()));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(reels)
}

/// Extract the local name from a qualified XML name (strip namespace prefix).
fn local_name(name: &[u8]) -> String {
    let s = std::str::from_utf8(name).unwrap_or("");
    // "ns:LocalName" → "LocalName"
    match s.rfind(':') {
        Some(pos) => s[pos + 1..].to_string(),
        None => s.to_string(),
    }
}

// ── XML serialisation ─────────────────────────────────────────────────────────

impl CompositionPlaylist {
    /// Serialize the CPL to a minimal XML string.
    ///
    /// The output conforms to a simplified subset of SMPTE ST 2067-3 sufficient
    /// for round-trip testing: Id, ContentTitle, EditRate, and all Segments
    /// with their Sequences and Resources are serialised.
    ///
    /// Namespace URIs are abbreviated (`cpl:` prefix) for readability.
    #[must_use]
    pub fn to_xml(&self) -> String {
        let (rate_num, rate_den) = self.edit_rate;
        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str(
            "<CompositionPlaylist xmlns=\"http://www.smpte-ra.org/schemas/2067-3/2016\">\n",
        );
        out.push_str(&format!("  <Id>urn:uuid:{}</Id>\n", self.id));
        out.push_str(&format!(
            "  <ContentTitle>{}</ContentTitle>\n",
            escape_xml_text(&self.content_title)
        ));
        out.push_str(&format!("  <EditRate>{rate_num} {rate_den}</EditRate>\n"));
        out.push_str("  <SegmentList>\n");
        for seg in &self.segments {
            out.push_str("    <Segment>\n");
            out.push_str(&format!("      <Id>urn:uuid:{}</Id>\n", seg.id));
            if let Some(ref ann) = seg.annotation {
                out.push_str(&format!(
                    "      <Annotation>{}</Annotation>\n",
                    escape_xml_text(ann)
                ));
            }
            out.push_str("      <SequenceList>\n");
            for seq in &seg.sequences {
                out.push_str("        <Sequence>\n");
                out.push_str(&format!("          <Id>urn:uuid:{}</Id>\n", seq.id));
                out.push_str(&format!(
                    "          <TrackId>urn:uuid:{}</TrackId>\n",
                    seq.track_id
                ));
                out.push_str("          <ResourceList>\n");
                for res in &seq.resources {
                    out.push_str("            <Resource>\n");
                    out.push_str(&format!(
                        "              <TrackFileId>urn:uuid:{}</TrackFileId>\n",
                        res.track_file_id
                    ));
                    out.push_str(&format!(
                        "              <SourceDuration>{}</SourceDuration>\n",
                        res.source_duration
                    ));
                    out.push_str(&format!(
                        "              <EntryPoint>{}</EntryPoint>\n",
                        res.entry_point
                    ));
                    out.push_str(&format!(
                        "              <IntrinsicDuration>{}</IntrinsicDuration>\n",
                        res.intrinsic_duration
                    ));
                    out.push_str(&format!(
                        "              <RepeatCount>{}</RepeatCount>\n",
                        res.repeat_count
                    ));
                    out.push_str("            </Resource>\n");
                }
                out.push_str("          </ResourceList>\n");
                out.push_str("        </Sequence>\n");
            }
            out.push_str("      </SequenceList>\n");
            out.push_str("    </Segment>\n");
        }
        out.push_str("  </SegmentList>\n");
        out.push_str("</CompositionPlaylist>\n");
        out
    }

    /// Parse a CPL from its XML representation produced by [`Self::to_xml`].
    ///
    /// This is a targeted parser designed to round-trip the output of `to_xml`
    /// and should not be used as a general SMPTE ST 2067-3 parser.
    pub fn from_xml(xml: &str) -> Result<Self, String> {
        use std::collections::VecDeque;

        // Very lightweight recursive-descent over the XML element tree.
        // We rely on the fact that `to_xml` produces well-indented, one-tag-
        // per-line output, so we can parse by walking tag/text pairs.
        // For a general implementation see `cpl_parser` module or quick-xml.

        let mut id = String::new();
        let mut content_title = String::new();
        let mut edit_rate = (24u32, 1u32);
        let mut segments: Vec<CplSegment> = Vec::new();

        // State machine: we walk line by line and track context via a tag stack.
        let mut tag_stack: VecDeque<String> = VecDeque::new();

        // Current objects being built.
        let mut cur_seg: Option<CplSegment> = None;
        let mut cur_seq: Option<CplSequence> = None;
        let mut cur_res: Option<CplResource> = None;

        for raw_line in xml.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with("<?") {
                continue;
            }

            // Detect closing tag first (before open tag, in case self-close)
            if line.starts_with("</") {
                let tag = line
                    .trim_start_matches("</")
                    .trim_end_matches('>')
                    .to_string();
                // Strip namespace prefix if any (e.g. "cpl:Resource" -> "Resource")
                let tag = tag.split(':').last().unwrap_or(&tag).to_string();

                match tag.as_str() {
                    "Segment" => {
                        if let Some(seg) = cur_seg.take() {
                            segments.push(seg);
                        }
                    }
                    "Sequence" => {
                        if let (Some(seq), Some(seg)) = (cur_seq.take(), cur_seg.as_mut()) {
                            seg.sequences.push(seq);
                        }
                    }
                    "Resource" => {
                        if let (Some(res), Some(seq)) = (cur_res.take(), cur_seq.as_mut()) {
                            seq.resources.push(res);
                        }
                    }
                    _ => {}
                }
                tag_stack.pop_back();
                continue;
            }

            // Opening tag (possibly with text content on the same line)
            if line.starts_with('<') {
                // Extract tag name (up to first space or '>')
                let inner = line.trim_start_matches('<');
                let tag_end = inner.find(['>', ' ']).unwrap_or(inner.len());
                let tag_raw = &inner[..tag_end];
                // Strip namespace prefix
                let tag = tag_raw.split(':').last().unwrap_or(tag_raw).to_string();

                // Extract text between > … </ on the same line
                let text = if let (Some(open), Some(close)) = (line.find('>'), line.rfind("</")) {
                    let t = &line[open + 1..close];
                    unescape_xml_text(t)
                } else {
                    String::new()
                };

                match tag.as_str() {
                    "CompositionPlaylist" | "SegmentList" | "SequenceList" | "ResourceList" => {
                        // container — no value
                    }
                    "Id" => {
                        let val = text.trim_start_matches("urn:uuid:").to_string();
                        match tag_stack.back().map(String::as_str) {
                            Some("Segment") | None => {
                                // Segment ID
                                if let Some(seg) = cur_seg.as_mut() {
                                    seg.id = val;
                                } else {
                                    id = val;
                                }
                            }
                            Some("Sequence") => {
                                if let Some(seq) = cur_seq.as_mut() {
                                    seq.id = val;
                                }
                            }
                            Some("Resource") => {
                                // Resource has no plain Id in our schema
                            }
                            _ => {
                                id = val;
                            }
                        }
                    }
                    "ContentTitle" => content_title = text,
                    "EditRate" => {
                        let parts: Vec<&str> = text.split_whitespace().collect();
                        if parts.len() == 2 {
                            edit_rate = (
                                parts[0].parse().unwrap_or(24),
                                parts[1].parse().unwrap_or(1),
                            );
                        }
                    }
                    "Segment" => {
                        cur_seg = Some(CplSegment::new(""));
                    }
                    "Sequence" => {
                        cur_seq = Some(CplSequence::new("", ""));
                    }
                    "Resource" => {
                        cur_res = Some(CplResource {
                            track_file_id: String::new(),
                            source_duration: 0,
                            entry_point: 0,
                            intrinsic_duration: 0,
                            repeat_count: 1,
                        });
                    }
                    "TrackFileId" => {
                        let val = text.trim_start_matches("urn:uuid:").to_string();
                        if let Some(res) = cur_res.as_mut() {
                            res.track_file_id = val;
                        }
                    }
                    "SourceDuration" => {
                        if let Some(res) = cur_res.as_mut() {
                            res.source_duration = text.parse().unwrap_or(0);
                        }
                    }
                    "EntryPoint" => {
                        if let Some(res) = cur_res.as_mut() {
                            res.entry_point = text.parse().unwrap_or(0);
                        }
                    }
                    "IntrinsicDuration" => {
                        if let Some(res) = cur_res.as_mut() {
                            res.intrinsic_duration = text.parse().unwrap_or(0);
                        }
                    }
                    "RepeatCount" => {
                        if let Some(res) = cur_res.as_mut() {
                            res.repeat_count = text.parse().unwrap_or(1);
                        }
                    }
                    "TrackId" => {
                        let val = text.trim_start_matches("urn:uuid:").to_string();
                        if let Some(seq) = cur_seq.as_mut() {
                            seq.track_id = val;
                        }
                    }
                    "Annotation" => {
                        if let Some(seg) = cur_seg.as_mut() {
                            seg.annotation = Some(text);
                        }
                    }
                    _ => {}
                }

                // Push the opening tag onto the stack for context (unless self-closing)
                if !line.ends_with("/>") && !line.contains("</") {
                    tag_stack.push_back(tag);
                }

                continue;
            }
        }

        // Resolve segment IDs: the parser above picks up Id inside Segment context
        // via tag_stack but we need the top-level Id captured first.
        // Re-parse just Id and ContentTitle with a simple regex-free scan.
        let mut found_top_id = false;
        for raw_line in xml.lines() {
            let line = raw_line.trim();
            if line.starts_with("<Id>") && !found_top_id {
                let val = line
                    .trim_start_matches("<Id>")
                    .trim_end_matches("</Id>")
                    .trim_start_matches("urn:uuid:")
                    .to_string();
                id = val;
                found_top_id = true;
            }
        }

        let mut cpl = CompositionPlaylist::new(id, content_title, edit_rate);
        for seg in segments {
            cpl.add_segment(seg);
        }
        Ok(cpl)
    }
}

/// Escape XML text content (not attribute values).
fn escape_xml_text(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Reverse XML text escaping.
fn unescape_xml_text(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_resource(dur: u64) -> CplResource {
        CplResource::simple("urn:uuid:test-tf-id", dur)
    }

    fn make_sequence(dur: u64) -> CplSequence {
        let mut seq = CplSequence::new("seq-1", "track-1");
        seq.add_resource(make_resource(dur));
        seq
    }

    fn make_segment(dur: u64) -> CplSegment {
        let mut seg = CplSegment::new("seg-1");
        seg.add_sequence(make_sequence(dur));
        seg
    }

    // ── CplResource ───────────────────────────────────────────────────────

    #[test]
    fn test_resource_simple_construction() {
        let r = make_resource(100);
        assert_eq!(r.intrinsic_duration, 100);
        assert_eq!(r.entry_point, 0);
        assert_eq!(r.repeat_count, 1);
    }

    #[test]
    fn test_resource_effective_duration_single() {
        let r = make_resource(50);
        assert_eq!(r.effective_duration(), 50);
    }

    #[test]
    fn test_resource_effective_duration_repeat() {
        let mut r = make_resource(50);
        r.repeat_count = 3;
        assert_eq!(r.effective_duration(), 150);
    }

    // ── CplSequence ───────────────────────────────────────────────────────

    #[test]
    fn test_sequence_empty_duration() {
        let seq = CplSequence::new("id", "track");
        assert_eq!(seq.total_duration(), 0);
    }

    #[test]
    fn test_sequence_single_resource() {
        let seq = make_sequence(240);
        assert_eq!(seq.total_duration(), 240);
    }

    #[test]
    fn test_sequence_multiple_resources() {
        let mut seq = CplSequence::new("s", "t");
        seq.add_resource(make_resource(100));
        seq.add_resource(make_resource(200));
        assert_eq!(seq.total_duration(), 300);
    }

    // ── CplSegment ────────────────────────────────────────────────────────

    #[test]
    fn test_segment_empty_duration() {
        let seg = CplSegment::new("seg");
        assert_eq!(seg.duration(), 0);
    }

    #[test]
    fn test_segment_duration() {
        let seg = make_segment(480);
        assert_eq!(seg.duration(), 480);
    }

    #[test]
    fn test_segment_annotation_optional() {
        let mut seg = CplSegment::new("seg");
        assert!(seg.annotation.is_none());
        seg.annotation = Some("Act 1".to_string());
        assert_eq!(seg.annotation.as_deref(), Some("Act 1"));
    }

    // ── CompositionPlaylist ───────────────────────────────────────────────

    #[test]
    fn test_cpl_empty() {
        let cpl = CompositionPlaylist::new("cpl-id", "My Film", (24, 1));
        assert!(cpl.is_empty());
        assert_eq!(cpl.total_duration(), 0);
        assert_eq!(cpl.segment_count(), 0);
    }

    #[test]
    fn test_cpl_add_segment() {
        let mut cpl = CompositionPlaylist::new("cpl-id", "My Film", (24, 1));
        cpl.add_segment(make_segment(2400)); // 100 s at 24 fps
        assert_eq!(cpl.segment_count(), 1);
        assert_eq!(cpl.total_duration(), 2400);
    }

    #[test]
    fn test_cpl_total_duration_secs() {
        let mut cpl = CompositionPlaylist::new("id", "Title", (24, 1));
        cpl.add_segment(make_segment(2400));
        let secs = cpl.total_duration_secs();
        assert!((secs - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_cpl_segments_slice() {
        let mut cpl = CompositionPlaylist::new("id", "Title", (24, 1));
        cpl.add_segment(make_segment(100));
        cpl.add_segment(make_segment(200));
        assert_eq!(cpl.segments().len(), 2);
    }

    #[test]
    fn test_cpl_zero_edit_rate_denominator() {
        // Edge case: zero numerator should not panic.
        let cpl = CompositionPlaylist::new("id", "Title", (0, 1));
        assert_eq!(cpl.total_duration_secs(), 0.0);
    }

    #[test]
    fn test_cpl_content_title() {
        let cpl = CompositionPlaylist::new("id", "Feature Film 2025", (25, 1));
        assert_eq!(cpl.content_title, "Feature Film 2025");
    }

    #[test]
    fn test_cpl_edit_rate_stored() {
        let cpl = CompositionPlaylist::new("id", "Title", (30000, 1001));
        assert_eq!(cpl.edit_rate, (30000, 1001));
    }

    // ── Round-trip tests ──────────────────────────────────────────────────

    /// Build a minimal CPL, serialise to XML, parse back, and verify the
    /// structural invariants are preserved.
    #[test]
    fn test_cpl_roundtrip() {
        // Build a CPL with two segments.
        let cpl_id = "550e8400-e29b-41d4-a716-446655440000";
        let mut original = CompositionPlaylist::new(cpl_id, "Round-Trip Test Film", (24, 1));

        // Segment 1: one video sequence with one resource.
        let mut seg1 = CplSegment::new("seg-id-0001");
        let mut seq1 = CplSequence::new("seq-id-0001", "track-id-0001");
        seq1.add_resource(CplResource::simple("tf-id-video-001", 2400));
        seg1.add_sequence(seq1);
        original.add_segment(seg1);

        // Segment 2: two sequences (video + audio).
        let mut seg2 = CplSegment::new("seg-id-0002");
        seg2.annotation = Some("Act II".to_string());
        let mut seq2v = CplSequence::new("seq-id-0002v", "track-id-0001");
        seq2v.add_resource(CplResource::simple("tf-id-video-001", 4800));
        let mut seq2a = CplSequence::new("seq-id-0002a", "track-id-0002");
        seq2a.add_resource(CplResource::simple("tf-id-audio-001", 4800));
        seg2.add_sequence(seq2v);
        seg2.add_sequence(seq2a);
        original.add_segment(seg2);

        // Serialise to XML.
        let xml = original.to_xml();

        // Verify it is valid XML: must contain the CPL namespace.
        assert!(xml.contains("CompositionPlaylist"));
        assert!(xml.contains("Round-Trip Test Film"));
        assert!(xml.contains("24 1"));

        // Parse back.
        let parsed = CompositionPlaylist::from_xml(&xml).expect("round-trip parse must succeed");

        // Verify CPL ID is preserved.
        assert_eq!(parsed.id, cpl_id, "CPL id must survive round-trip");

        // Verify content title.
        assert_eq!(
            parsed.content_title, "Round-Trip Test Film",
            "content title must survive round-trip"
        );

        // Verify edit rate.
        assert_eq!(
            parsed.edit_rate,
            (24, 1),
            "edit rate must survive round-trip"
        );

        // Verify segment count.
        assert_eq!(
            parsed.segment_count(),
            original.segment_count(),
            "segment count must match after round-trip"
        );

        // Verify total duration.
        assert_eq!(
            parsed.total_duration(),
            original.total_duration(),
            "total duration must match after round-trip"
        );
    }

    #[test]
    fn test_cpl_roundtrip_edit_rate_25() {
        let mut cpl = CompositionPlaylist::new("cpl-pal-001", "PAL Broadcast", (25, 1));
        let mut seg = CplSegment::new("seg-pal-001");
        let mut seq = CplSequence::new("seq-pal-001", "track-pal-001");
        seq.add_resource(CplResource::simple("tf-pal-001", 1500));
        seg.add_sequence(seq);
        cpl.add_segment(seg);

        let xml = cpl.to_xml();
        let parsed = CompositionPlaylist::from_xml(&xml).expect("parse");

        assert_eq!(parsed.edit_rate, (25, 1));
        assert_eq!(parsed.segment_count(), 1);
        assert_eq!(parsed.total_duration(), 1500);
    }

    #[test]
    fn test_cpl_roundtrip_empty_segments() {
        let cpl = CompositionPlaylist::new("cpl-empty", "Empty CPL", (24, 1));
        let xml = cpl.to_xml();
        let parsed = CompositionPlaylist::from_xml(&xml).expect("parse");
        assert_eq!(parsed.segment_count(), 0);
        assert_eq!(parsed.total_duration(), 0);
        assert_eq!(parsed.edit_rate, (24, 1));
    }

    #[test]
    fn test_cpl_to_xml_contains_segment_ids() {
        let mut cpl = CompositionPlaylist::new("cpl-001", "Test", (24, 1));
        cpl.add_segment(CplSegment::new("my-seg-id-001"));
        let xml = cpl.to_xml();
        assert!(xml.contains("my-seg-id-001"), "XML must embed segment IDs");
    }

    #[test]
    fn test_cpl_roundtrip_resource_fields() {
        let mut cpl = CompositionPlaylist::new("cpl-res", "Resources Test", (24, 1));
        let mut seg = CplSegment::new("seg-001");
        let mut seq = CplSequence::new("seq-001", "track-001");
        let mut res = CplResource::simple("tf-001", 960);
        res.entry_point = 24;
        res.repeat_count = 2;
        seq.add_resource(res);
        seg.add_sequence(seq);
        cpl.add_segment(seg);

        let xml = cpl.to_xml();
        let parsed = CompositionPlaylist::from_xml(&xml).expect("parse");

        let parsed_res = &parsed.segments()[0].sequences[0].resources[0];
        assert_eq!(parsed_res.source_duration, 960);
        assert_eq!(parsed_res.entry_point, 24);
        assert_eq!(parsed_res.repeat_count, 2);
    }

    // ── Lazy parsing tests ────────────────────────────────────────────────

    /// Build a synthetic CPL with two segments and return the XML string.
    fn synthetic_cpl_xml() -> (String, CompositionPlaylist) {
        let cpl_id = "aa550e84-e29b-41d4-a716-446655440000";
        let mut cpl = CompositionPlaylist::new(cpl_id, "Lazy Test Film", (24, 1));

        let mut seg1 = CplSegment::new("seg-lazy-0001");
        seg1.annotation = Some("Act I".to_string());
        let mut seq1 = CplSequence::new("seq-lazy-0001", "track-lazy-0001");
        seq1.add_resource(CplResource::simple("tf-lazy-video-001", 2400));
        seg1.add_sequence(seq1);
        cpl.add_segment(seg1);

        let mut seg2 = CplSegment::new("seg-lazy-0002");
        let mut seq2 = CplSequence::new("seq-lazy-0002", "track-lazy-0001");
        seq2.add_resource(CplResource::simple("tf-lazy-video-002", 4800));
        seg2.add_sequence(seq2);
        cpl.add_segment(seg2);

        (cpl.to_xml(), cpl)
    }

    #[test]
    fn test_lazy_header_parse_matches_eager() {
        let (xml, eager_cpl) = synthetic_cpl_xml();

        let header = parse_cpl_header(&xml).expect("lazy header parse must succeed");

        assert_eq!(header.id, eager_cpl.id, "lazy id must match eager id");
        assert_eq!(
            header.content_title, eager_cpl.content_title,
            "lazy content_title must match eager"
        );
        assert_eq!(
            header.edit_rate, eager_cpl.edit_rate,
            "lazy edit_rate must match eager"
        );
    }

    #[test]
    fn test_lazy_reel_list_skips_segment_data() {
        let (xml, eager_cpl) = synthetic_cpl_xml();

        let reels = parse_reel_list(&xml).expect("lazy reel list parse must succeed");

        // We should get one Reel per Segment
        assert_eq!(
            reels.len(),
            eager_cpl.segment_count(),
            "reel count must match segment count"
        );

        // Verify IDs match segment IDs
        let seg_ids: Vec<&str> = eager_cpl.segments().iter().map(|s| s.id.as_str()).collect();
        for (reel, expected_id) in reels.iter().zip(seg_ids.iter()) {
            assert_eq!(&reel.id, expected_id, "reel id must match segment id");
        }

        // First segment has an annotation
        assert_eq!(
            reels[0].annotation.as_deref(),
            Some("Act I"),
            "annotation must be preserved"
        );
        // Second segment has no annotation
        assert!(
            reels[1].annotation.is_none(),
            "missing annotation must be None"
        );
    }

    #[test]
    fn test_lazy_header_missing_id_returns_error() {
        let xml = r#"<?xml version="1.0"?><CompositionPlaylist xmlns="http://example.com"><ContentTitle>No ID</ContentTitle></CompositionPlaylist>"#;
        let result = parse_cpl_header(xml);
        assert!(result.is_err(), "missing Id must yield an error");
    }
}
