// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! DASH segment timeline and template generation.
//!
//! Provides structures for describing DASH `<SegmentTemplate>` elements with
//! optional `<SegmentTimeline>` children, segment URL template resolution,
//! and ISOBMFF `sidx` (Segment Index) box serialisation.

// ---------------------------------------------------------------------------
// SegmentTimelineEntry
// ---------------------------------------------------------------------------

/// A single `<S>` element inside a `<SegmentTimeline>`.
///
/// Corresponds to `S@t`, `S@d`, and `S@r` in the DASH MPD schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentTimelineEntry {
    /// Absolute presentation time at segment start (in timescale ticks).
    /// `None` means "inherit from the preceding entry".
    pub t: Option<u64>,
    /// Segment duration in timescale ticks.
    pub d: u64,
    /// Repeat count.  `None` / `Some(0)` = no repeat (segment appears once).
    /// `Some(r)` means the segment appears `r + 1` times in total.
    pub r: Option<i32>,
}

impl SegmentTimelineEntry {
    /// Create a new entry with an explicit start time.
    #[must_use]
    pub fn with_time(t: u64, d: u64) -> Self {
        Self {
            t: Some(t),
            d,
            r: None,
        }
    }

    /// Create a new entry without an explicit start time (inherits from previous).
    #[must_use]
    pub fn without_time(d: u64) -> Self {
        Self {
            t: None,
            d,
            r: None,
        }
    }

    /// Set the repeat count on this entry.
    #[must_use]
    pub fn with_repeat(mut self, r: i32) -> Self {
        self.r = Some(r);
        self
    }

    /// Return the total count of segments represented by this entry.
    /// An entry with `r = None` or `r = Some(0)` represents one segment;
    /// `r = Some(n)` represents `n + 1` segments.
    #[must_use]
    pub fn segment_count(&self) -> u64 {
        match self.r {
            None | Some(0) => 1,
            Some(r) if r > 0 => (r as u64) + 1,
            _ => 1, // negative repeat is not standard; treat as 1
        }
    }

    /// Serialise to `<S .../>` XML string.
    #[must_use]
    pub fn to_xml(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(t) = self.t {
            parts.push(format!("t=\"{t}\""));
        }
        parts.push(format!("d=\"{}\"", self.d));
        if let Some(r) = self.r {
            if r != 0 {
                parts.push(format!("r=\"{r}\""));
            }
        }
        format!("<S {}/>", parts.join(" "))
    }
}

// ---------------------------------------------------------------------------
// SegmentTimeline
// ---------------------------------------------------------------------------

/// A DASH `<SegmentTimeline>` containing a list of `<S>` entries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SegmentTimeline {
    /// Ordered list of timeline entries.
    pub entries: Vec<SegmentTimelineEntry>,
}

impl SegmentTimeline {
    /// Create an empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a timeline from a flat list of per-segment durations (in ticks).
    ///
    /// Consecutive equal durations are merged using the `r` repeat field to
    /// produce compact MPD output.
    #[must_use]
    pub fn from_durations(durations: &[u64], _timescale: u32) -> Self {
        let mut entries: Vec<SegmentTimelineEntry> = Vec::new();
        let mut current_time: u64 = 0;

        let mut i = 0;
        while i < durations.len() {
            let d = durations[i];
            // Count how many consecutive segments share this duration
            let mut repeat_count: i32 = 0;
            while i + 1 + (repeat_count as usize) < durations.len()
                && durations[i + 1 + repeat_count as usize] == d
            {
                repeat_count += 1;
            }

            let entry = SegmentTimelineEntry {
                t: if i == 0 { Some(current_time) } else { None },
                d,
                r: if repeat_count > 0 {
                    Some(repeat_count)
                } else {
                    None
                },
            };
            current_time += d * (entry.segment_count());
            i += entry.segment_count() as usize;
            entries.push(entry);
        }

        Self { entries }
    }

    /// Return the total duration in timescale ticks covered by this timeline.
    #[must_use]
    pub fn total_duration_ticks(&self) -> u64 {
        self.entries.iter().map(|e| e.d * e.segment_count()).sum()
    }

    /// Serialise to the `<SegmentTimeline>…</SegmentTimeline>` XML element.
    #[must_use]
    pub fn to_xml(&self) -> String {
        let mut xml = String::from("<SegmentTimeline>\n");
        for entry in &self.entries {
            xml.push_str("  ");
            xml.push_str(&entry.to_xml());
            xml.push('\n');
        }
        xml.push_str("</SegmentTimeline>");
        xml
    }
}

// ---------------------------------------------------------------------------
// SegmentTemplate
// ---------------------------------------------------------------------------

/// A DASH `<SegmentTemplate>` element descriptor.
#[derive(Debug, Clone)]
pub struct SegmentTemplate {
    /// Media segment URL template (may contain `$Number$`, `$Time$`,
    /// `$Bandwidth$`).
    pub media_template: String,
    /// Initialisation segment URL (relative to `<BaseURL>`).
    pub initialization: String,
    /// Timescale (ticks per second) for all time values in this template.
    pub timescale: u32,
    /// Sequence number of the first segment.
    pub start_number: u32,
    /// Optional segment timeline.
    pub timeline: Option<SegmentTimeline>,
}

impl SegmentTemplate {
    /// Create a new template with sensible defaults.
    #[must_use]
    pub fn new(
        media_template: impl Into<String>,
        initialization: impl Into<String>,
        timescale: u32,
    ) -> Self {
        Self {
            media_template: media_template.into(),
            initialization: initialization.into(),
            timescale,
            start_number: 1,
            timeline: None,
        }
    }

    /// Set the start number.
    #[must_use]
    pub fn with_start_number(mut self, n: u32) -> Self {
        self.start_number = n;
        self
    }

    /// Attach a segment timeline.
    #[must_use]
    pub fn with_timeline(mut self, tl: SegmentTimeline) -> Self {
        self.timeline = Some(tl);
        self
    }

    /// Serialise the `<SegmentTemplate>` XML element (and its children).
    #[must_use]
    pub fn to_mpd_xml(&self) -> String {
        let timeline_xml = self
            .timeline
            .as_ref()
            .map(|tl| format!("\n  {}", tl.to_xml().replace('\n', "\n  ")))
            .unwrap_or_default();

        if timeline_xml.is_empty() {
            format!(
                "<SegmentTemplate timescale=\"{}\" media=\"{}\" initialization=\"{}\" \
                 startNumber=\"{}\"/>",
                self.timescale, self.media_template, self.initialization, self.start_number,
            )
        } else {
            format!(
                "<SegmentTemplate timescale=\"{}\" media=\"{}\" initialization=\"{}\" \
                 startNumber=\"{}\">{}\n</SegmentTemplate>",
                self.timescale,
                self.media_template,
                self.initialization,
                self.start_number,
                timeline_xml,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// SegmentNaming
// ---------------------------------------------------------------------------

/// URL template resolver for DASH segment naming.
///
/// Supports `$Number$`, `$Time$`, and `$Bandwidth$` substitution tokens as
/// defined in DASH-IF IOP.
#[derive(Debug, Clone)]
pub struct SegmentNaming {
    /// The template string (e.g. `"seg-$Number%05d$.cmfv"`).
    pub template: String,
}

impl SegmentNaming {
    /// Create a new naming helper from a template string.
    #[must_use]
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }

    /// Resolve the template for the given segment number, presentation time,
    /// and bandwidth.
    ///
    /// Supports `$Number$`, `$Time$`, and `$Bandwidth$` tokens.
    #[must_use]
    pub fn resolve(&self, number: u32, time: u64, bandwidth: u32) -> String {
        self.template
            .replace("$Number$", &number.to_string())
            .replace("$Time$", &time.to_string())
            .replace("$Bandwidth$", &bandwidth.to_string())
    }
}

// ---------------------------------------------------------------------------
// SidxEntry / SegmentIndex
// ---------------------------------------------------------------------------

/// A single entry inside an ISOBMFF `sidx` box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidxEntry {
    /// Reference type: 0 = media, 1 = sidx.
    pub reference_type: u8,
    /// Size of the referenced material in bytes.
    pub referenced_size: u32,
    /// Duration of the sub-segment in timescale ticks.
    pub subsegment_duration: u32,
    /// Whether this sub-segment starts with a SAP.
    pub starts_with_sap: bool,
    /// SAP type (0–7).
    pub sap_type: u8,
}

impl SidxEntry {
    /// Construct a new media entry.
    #[must_use]
    pub fn media(referenced_size: u32, subsegment_duration: u32, starts_with_sap: bool) -> Self {
        Self {
            reference_type: 0,
            referenced_size,
            subsegment_duration,
            starts_with_sap,
            sap_type: if starts_with_sap { 1 } else { 0 },
        }
    }
}

/// A complete DASH `sidx` (Segment Index) descriptor.
#[derive(Debug, Clone)]
pub struct SegmentIndex {
    /// Earliest presentation time for the referenced media, in timescale ticks.
    pub earliest_presentation_time: u64,
    /// Segment index entries.
    pub entries: Vec<SidxEntry>,
}

impl SegmentIndex {
    /// Construct a new segment index.
    #[must_use]
    pub fn new(earliest_presentation_time: u64) -> Self {
        Self {
            earliest_presentation_time,
            entries: Vec::new(),
        }
    }

    /// Append an entry.
    pub fn add_entry(&mut self, entry: SidxEntry) {
        self.entries.push(entry);
    }

    /// Total number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if there are no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// sidx box writer
// ---------------------------------------------------------------------------

/// Serialise a `SegmentIndex` into an ISOBMFF `sidx` box (version 1, 64-bit
/// times).
///
/// Box layout (ISO/IEC 14496-12 §8.16.3, version 1):
/// ```text
/// 4   size
/// 4   "sidx"
/// 1   version = 1
/// 3   flags = 0
/// 4   reference_id = 1
/// 4   timescale
/// 8   earliest_presentation_time   (version 1)
/// 8   first_offset = 0             (version 1)
/// 2   reserved = 0
/// 2   reference_count
/// Per entry (12 bytes):
///   1 bit  reference_type
///  31 bits referenced_size
///  32 bits subsegment_duration
///   1 bit  starts_with_SAP
///   3 bits SAP_type
///  28 bits SAP_delta_time = 0
/// ```
#[must_use]
pub fn write_sidx_box(index: &SegmentIndex, timescale: u32) -> Vec<u8> {
    use crate::isobmff_writer::BoxWriter;

    let mut out: Vec<u8> = Vec::new();

    BoxWriter::write_box(&mut out, b"sidx", |w| {
        // FullBox header: version 1, flags 0
        BoxWriter::write_full_box_header(w, 1, 0);

        w.write_u32(1); // reference_id
        w.write_u32(timescale);
        w.write_u64(index.earliest_presentation_time);
        w.write_u64(0); // first_offset

        w.write_u16(0); // reserved
        w.write_u16(index.entries.len() as u16); // reference_count

        for entry in &index.entries {
            // First u32: reference_type (1 bit) | referenced_size (31 bits)
            let ref_type_bit: u32 = if entry.reference_type != 0 {
                0x8000_0000
            } else {
                0
            };
            let first_word = ref_type_bit | (entry.referenced_size & 0x7FFF_FFFF);
            w.write_u32(first_word);

            // Second u32: subsegment_duration
            w.write_u32(entry.subsegment_duration);

            // Third u32: starts_with_SAP (1 bit) | SAP_type (3 bits) | SAP_delta_time (28 bits)
            let sap_bit: u32 = if entry.starts_with_sap {
                0x8000_0000
            } else {
                0
            };
            let sap_type_bits: u32 = ((entry.sap_type as u32) & 0x07) << 28;
            w.write_u32(sap_bit | sap_type_bits);
        }
    });

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SegmentTimelineEntry -----------------------------------------------

    #[test]
    fn test_entry_with_time() {
        let e = SegmentTimelineEntry::with_time(1000, 3000);
        assert_eq!(e.t, Some(1000));
        assert_eq!(e.d, 3000);
        assert!(e.r.is_none());
    }

    #[test]
    fn test_entry_without_time() {
        let e = SegmentTimelineEntry::without_time(3000);
        assert!(e.t.is_none());
    }

    #[test]
    fn test_entry_segment_count_no_repeat() {
        let e = SegmentTimelineEntry::without_time(3000);
        assert_eq!(e.segment_count(), 1);
    }

    #[test]
    fn test_entry_segment_count_with_repeat() {
        let e = SegmentTimelineEntry::without_time(3000).with_repeat(4);
        assert_eq!(e.segment_count(), 5);
    }

    #[test]
    fn test_entry_xml_no_repeat_no_t() {
        let e = SegmentTimelineEntry::without_time(3000);
        let xml = e.to_xml();
        assert!(xml.contains("d=\"3000\""));
        assert!(!xml.contains('t'));
        assert!(!xml.contains('r'));
    }

    #[test]
    fn test_entry_xml_with_t_and_r() {
        let e = SegmentTimelineEntry::with_time(0, 4000).with_repeat(2);
        let xml = e.to_xml();
        assert!(xml.contains("t=\"0\""));
        assert!(xml.contains("d=\"4000\""));
        assert!(xml.contains("r=\"2\""));
    }

    // --- SegmentTimeline ----------------------------------------------------

    #[test]
    fn test_from_durations_single_duration() {
        let tl = SegmentTimeline::from_durations(&[3000, 3000, 3000], 90_000);
        // All three identical → one entry with r=2
        assert_eq!(tl.entries.len(), 1);
        assert_eq!(tl.entries[0].d, 3000);
        assert_eq!(tl.entries[0].r, Some(2));
    }

    #[test]
    fn test_from_durations_mixed() {
        let durations = vec![3000, 3000, 4000, 4000, 3000];
        let tl = SegmentTimeline::from_durations(&durations, 90_000);
        // Expect 3 groups: [3000 r=1], [4000 r=1], [3000]
        assert_eq!(tl.entries.len(), 3);
        assert_eq!(tl.entries[1].d, 4000);
    }

    #[test]
    fn test_from_durations_empty() {
        let tl = SegmentTimeline::from_durations(&[], 90_000);
        assert!(tl.entries.is_empty());
    }

    #[test]
    fn test_total_duration_ticks() {
        let tl = SegmentTimeline::from_durations(&[3000; 10], 90_000);
        assert_eq!(tl.total_duration_ticks(), 30_000);
    }

    #[test]
    fn test_timeline_xml_contains_s_elements() {
        let tl = SegmentTimeline::from_durations(&[3000, 3000], 90_000);
        let xml = tl.to_xml();
        assert!(xml.contains("<SegmentTimeline>"));
        assert!(xml.contains("<S "));
        assert!(xml.contains("</SegmentTimeline>"));
    }

    // --- SegmentTemplate ----------------------------------------------------

    #[test]
    fn test_segment_template_to_mpd_xml_no_timeline() {
        let tmpl = SegmentTemplate::new("seg-$Number$.cmfv", "init.cmfv", 90_000);
        let xml = tmpl.to_mpd_xml();
        assert!(xml.contains("timescale=\"90000\""));
        assert!(xml.contains("seg-$Number$.cmfv"));
        assert!(xml.contains("init.cmfv"));
        assert!(xml.contains("startNumber=\"1\""));
    }

    #[test]
    fn test_segment_template_with_timeline_xml() {
        let tl = SegmentTimeline::from_durations(&[3000, 3000], 90_000);
        let tmpl = SegmentTemplate::new("seg-$Number$.cmfv", "init.cmfv", 90_000).with_timeline(tl);
        let xml = tmpl.to_mpd_xml();
        assert!(xml.contains("<SegmentTimeline>"));
        assert!(xml.contains("</SegmentTemplate>"));
    }

    #[test]
    fn test_segment_template_start_number() {
        let tmpl = SegmentTemplate::new("s$Number$.m4s", "init.mp4", 1000).with_start_number(100);
        let xml = tmpl.to_mpd_xml();
        assert!(xml.contains("startNumber=\"100\""));
    }

    // --- SegmentNaming ------------------------------------------------------

    #[test]
    fn test_segment_naming_number_substitution() {
        let naming = SegmentNaming::new("seg-$Number$.cmfv");
        let result = naming.resolve(5, 0, 1_000_000);
        assert_eq!(result, "seg-5.cmfv");
    }

    #[test]
    fn test_segment_naming_time_substitution() {
        let naming = SegmentNaming::new("seg-$Time$.cmfv");
        let result = naming.resolve(1, 90_000, 0);
        assert_eq!(result, "seg-90000.cmfv");
    }

    #[test]
    fn test_segment_naming_bandwidth_substitution() {
        let naming = SegmentNaming::new("$Bandwidth$/seg-$Number$.cmfv");
        let result = naming.resolve(3, 0, 5_000_000);
        assert_eq!(result, "5000000/seg-3.cmfv");
    }

    #[test]
    fn test_segment_naming_all_tokens() {
        let naming = SegmentNaming::new("$Bandwidth$/$Number$-$Time$.m4s");
        let result = naming.resolve(10, 270_000, 3_000_000);
        assert_eq!(result, "3000000/10-270000.m4s");
    }

    // --- SidxEntry / SegmentIndex -------------------------------------------

    #[test]
    fn test_sidx_entry_media() {
        let e = SidxEntry::media(65536, 90_000, true);
        assert_eq!(e.reference_type, 0);
        assert_eq!(e.referenced_size, 65536);
        assert_eq!(e.subsegment_duration, 90_000);
        assert!(e.starts_with_sap);
    }

    #[test]
    fn test_segment_index_empty() {
        let idx = SegmentIndex::new(0);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_segment_index_add_entries() {
        let mut idx = SegmentIndex::new(0);
        idx.add_entry(SidxEntry::media(1000, 3000, true));
        idx.add_entry(SidxEntry::media(2000, 3000, false));
        assert_eq!(idx.len(), 2);
    }

    // --- sidx box -----------------------------------------------------------

    #[test]
    fn test_write_sidx_box_fourcc() {
        let idx = SegmentIndex::new(0);
        let out = write_sidx_box(&idx, 90_000);
        assert_eq!(&out[4..8], b"sidx");
    }

    #[test]
    fn test_write_sidx_box_size_correct() {
        let mut idx = SegmentIndex::new(0);
        idx.add_entry(SidxEntry::media(65536, 90_000, true));
        let out = write_sidx_box(&idx, 90_000);
        let size = u32::from_be_bytes(out[0..4].try_into().expect("4 bytes")) as usize;
        assert_eq!(size, out.len());
    }

    #[test]
    fn test_write_sidx_box_entry_count() {
        let mut idx = SegmentIndex::new(0);
        idx.add_entry(SidxEntry::media(1000, 3000, true));
        idx.add_entry(SidxEntry::media(2000, 3000, false));
        let out = write_sidx_box(&idx, 90_000);
        // reference_count is at offset: 8(fullbox) + 4(ref_id) + 4(ts) + 8(ept) + 8(first_off) + 2(reserved) = 34
        // But we need to account for the 8-byte box header, so +8 from out[0]
        let base = 8; // box header size
        let fullbox = 4;
        let ref_id = 4;
        let ts = 4;
        let ept = 8;
        let first_off = 8;
        let reserved = 2;
        let count_offset = base + fullbox + ref_id + ts + ept + first_off + reserved;
        let count = u16::from_be_bytes(
            out[count_offset..count_offset + 2]
                .try_into()
                .expect("2 bytes"),
        );
        assert_eq!(count, 2);
    }

    #[test]
    fn test_write_sidx_box_version_is_1() {
        let idx = SegmentIndex::new(0);
        let out = write_sidx_box(&idx, 90_000);
        // version byte is at offset 8 (after box header)
        assert_eq!(out[8], 1);
    }
}
