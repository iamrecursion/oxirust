//! Final Cut Pro 7 XML (`<xmeml>`) export for multi-camera timelines.
//!
//! Generates a Final Cut Pro 7–compatible XML interchange document from a
//! `MultiCamTimeline` so the edit can be imported into FCP 7, DaVinci
//! Resolve, or any NLE that supports the `xmeml` schema.
//!
//! # Format overview
//!
//! ```xml
//! <?xml version="1.0" encoding="UTF-8"?>
//! <!DOCTYPE xmeml>
//! <xmeml version="5">
//!   <sequence>
//!     <name>…</name>
//!     <rate><timebase>…</timebase><ntsc>FALSE</ntsc></rate>
//!     <media>
//!       <video>
//!         <!-- one <track> per angle, containing <clipitem>s -->
//!       </video>
//!       <audio/>
//!     </media>
//!   </sequence>
//! </xmeml>
//! ```
//!
//! Each camera angle becomes a separate `<track>`.  Edit decisions on the
//! primary sequence track are represented as `<clipitem>` elements.

use crate::edit::timeline::MultiCamTimeline;

// ── MultiCamXmlExporter ───────────────────────────────────────────────────────

/// Exports a [`crate::edit::MultiCamTimeline`] as Final Cut Pro 7 XML.
///
/// # Example
///
/// ```
/// use oximedia_multicam::edit::MultiCamTimeline;
/// use oximedia_multicam::fcp_xml::{MultiCamXmlExporter, XmlExportConfig};
///
/// let mut timeline = MultiCamTimeline::new(3);
/// timeline.set_duration(240);
/// timeline.add_cut(0, 0).ok();
/// timeline.add_cut(48, 1).ok();
/// timeline.add_cut(96, 2).ok();
///
/// let exporter = MultiCamXmlExporter::new(XmlExportConfig {
///     sequence_name: "Interview".into(),
///     frame_rate: 24,
///     ntsc: false,
///     clip_name_prefix: "CAM".into(),
/// });
///
/// let xml = exporter.export(&timeline);
/// assert!(xml.contains("<xmeml"));
/// assert!(xml.contains("<timebase>24</timebase>"));
/// ```
#[derive(Debug, Clone)]
pub struct MultiCamXmlExporter {
    config: XmlExportConfig,
}

/// Configuration for FCP XML export.
#[derive(Debug, Clone)]
pub struct XmlExportConfig {
    /// Name of the exported sequence.
    pub sequence_name: String,
    /// Integer frame rate (e.g. 24, 25, 30, 50, 60).
    pub frame_rate: u32,
    /// Whether the frame rate is NTSC drop-frame (29.97, 59.94, etc.).
    pub ntsc: bool,
    /// Prefix used for clip names in the XML (e.g. "CAM" → "CAM_1", "CAM_2").
    pub clip_name_prefix: String,
}

impl Default for XmlExportConfig {
    fn default() -> Self {
        Self {
            sequence_name: "MultiCam Sequence".into(),
            frame_rate: 25,
            ntsc: false,
            clip_name_prefix: "Angle".into(),
        }
    }
}

impl MultiCamXmlExporter {
    /// Create a new exporter with the given configuration.
    #[must_use]
    pub fn new(config: XmlExportConfig) -> Self {
        Self { config }
    }

    /// Create an exporter with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(XmlExportConfig::default())
    }

    /// Export `timeline` as a Final Cut Pro 7 XML string.
    ///
    /// The output is a well-formed XML document and can be written directly to
    /// a `.xml` file for import into an NLE.
    #[must_use]
    pub fn export(&self, timeline: &MultiCamTimeline) -> String {
        let mut out = String::with_capacity(4096);
        let cfg = &self.config;

        // ── XML declaration ──────────────────────────────────────────────────
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<!DOCTYPE xmeml>\n");
        out.push_str("<xmeml version=\"5\">\n");

        // ── <sequence> ───────────────────────────────────────────────────────
        out.push_str("  <sequence>\n");

        let seq_name = xml_escape(&cfg.sequence_name);
        out.push_str(&format!("    <name>{seq_name}</name>\n"));

        // Rate block.
        let ntsc_str = if cfg.ntsc { "TRUE" } else { "FALSE" };
        out.push_str("    <rate>\n");
        out.push_str(&format!("      <timebase>{}</timebase>\n", cfg.frame_rate));
        out.push_str(&format!("      <ntsc>{ntsc_str}</ntsc>\n"));
        out.push_str("    </rate>\n");

        // Duration of the sequence in frames.
        out.push_str(&format!(
            "    <duration>{}</duration>\n",
            timeline.duration()
        ));

        // ── <media> ──────────────────────────────────────────────────────────
        out.push_str("    <media>\n");
        out.push_str("      <video>\n");

        self.write_video_tracks(&mut out, timeline);

        out.push_str("      </video>\n");
        out.push_str("      <audio/>\n");
        out.push_str("    </media>\n");

        out.push_str("  </sequence>\n");
        out.push_str("</xmeml>\n");

        out
    }

    /// Write one `<track>` per camera angle into `out`.
    fn write_video_tracks(&self, out: &mut String, timeline: &MultiCamTimeline) {
        let cfg = &self.config;
        let angle_count = timeline.angle_count();
        let duration = timeline.duration();
        let decisions = timeline.edit_decisions();

        for angle in 0..angle_count {
            out.push_str("        <track>\n");

            // Collect clip items for this angle.
            // A clip item is emitted for each contiguous run of this angle on
            // the primary cut sequence.

            // Build (start_frame, end_frame) pairs for this angle on the
            // primary output track.
            let intervals = Self::angle_intervals(angle, decisions, duration);

            for (clip_idx, (start, end)) in intervals.iter().enumerate() {
                let clip_name = xml_escape(&format!(
                    "{prefix}_{angle_num}_{clip}",
                    prefix = cfg.clip_name_prefix,
                    angle_num = angle + 1,
                    clip = clip_idx + 1,
                ));

                let in_frame = start;
                let out_frame = end;
                let seq_in = start;
                let seq_out = end;

                out.push_str("          <clipitem>\n");
                out.push_str(&format!("            <name>{clip_name}</name>\n"));
                out.push_str("            <rate>\n");
                out.push_str(&format!(
                    "              <timebase>{}</timebase>\n",
                    cfg.frame_rate
                ));
                out.push_str(&format!(
                    "              <ntsc>{}</ntsc>\n",
                    if cfg.ntsc { "TRUE" } else { "FALSE" }
                ));
                out.push_str("            </rate>\n");
                out.push_str(&format!("            <in>{in_frame}</in>\n"));
                out.push_str(&format!("            <out>{out_frame}</out>\n"));
                out.push_str(&format!("            <start>{seq_in}</start>\n"));
                out.push_str(&format!("            <end>{seq_out}</end>\n"));
                out.push_str("          </clipitem>\n");
            }

            out.push_str("        </track>\n");
        }
    }

    /// Compute the intervals of frame ranges where `angle` is the active angle
    /// on the primary (sequenced) output.
    ///
    /// Returns a list of `(start_frame, end_frame)` tuples (inclusive).
    fn angle_intervals(
        angle: usize,
        decisions: &[crate::edit::EditDecision],
        duration: u64,
    ) -> Vec<(u64, u64)> {
        if duration == 0 {
            return Vec::new();
        }

        let mut intervals = Vec::new();

        // Reconstruct the active angle at each edit point.
        // decisions are ordered by frame.
        let mut events: Vec<(u64, usize)> = decisions.iter().map(|d| (d.frame, d.angle)).collect();
        // Prepend a virtual event at frame 0 with angle 0 if none exists.
        if events.is_empty() || events[0].0 != 0 {
            events.insert(0, (0, 0));
        }

        let n = events.len();
        for i in 0..n {
            let (start, active_angle) = events[i];
            let end = if i + 1 < n {
                events[i + 1].0.saturating_sub(1)
            } else {
                duration.saturating_sub(1)
            };

            if active_angle == angle && start <= end {
                intervals.push((start, end));
            }
        }

        intervals
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Escape `<`, `>`, `&`, `"`, and `'` for XML content/attributes.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::MultiCamTimeline;

    fn make_timeline() -> MultiCamTimeline {
        let mut t = MultiCamTimeline::new(3);
        t.set_duration(120);
        t.add_cut(0, 0).expect("valid");
        t.add_cut(40, 1).expect("valid");
        t.add_cut(80, 2).expect("valid");
        t
    }

    // ── structural tests ─────────────────────────────────────────────────────

    #[test]
    fn test_export_contains_xmeml_root() {
        let exporter = MultiCamXmlExporter::default_config();
        let xml = exporter.export(&make_timeline());
        assert!(xml.contains("<xmeml version=\"5\">"), "Missing <xmeml>");
    }

    #[test]
    fn test_export_contains_closing_xmeml() {
        let exporter = MultiCamXmlExporter::default_config();
        let xml = exporter.export(&make_timeline());
        assert!(xml.contains("</xmeml>"), "Missing </xmeml>");
    }

    #[test]
    fn test_export_contains_sequence_name() {
        let exporter = MultiCamXmlExporter::new(XmlExportConfig {
            sequence_name: "SportsFinal".into(),
            ..XmlExportConfig::default()
        });
        let xml = exporter.export(&make_timeline());
        assert!(xml.contains("<name>SportsFinal</name>"), "Missing name");
    }

    #[test]
    fn test_export_contains_frame_rate() {
        let exporter = MultiCamXmlExporter::new(XmlExportConfig {
            frame_rate: 24,
            ..XmlExportConfig::default()
        });
        let xml = exporter.export(&make_timeline());
        assert!(
            xml.contains("<timebase>24</timebase>"),
            "Missing timebase 24"
        );
    }

    #[test]
    fn test_export_ntsc_true() {
        let exporter = MultiCamXmlExporter::new(XmlExportConfig {
            ntsc: true,
            frame_rate: 30,
            ..XmlExportConfig::default()
        });
        let xml = exporter.export(&make_timeline());
        assert!(xml.contains("<ntsc>TRUE</ntsc>"), "Expected NTSC=TRUE");
    }

    #[test]
    fn test_export_ntsc_false() {
        let exporter = MultiCamXmlExporter::default_config();
        let xml = exporter.export(&make_timeline());
        assert!(xml.contains("<ntsc>FALSE</ntsc>"), "Expected NTSC=FALSE");
    }

    #[test]
    fn test_export_contains_tracks_for_each_angle() {
        let exporter = MultiCamXmlExporter::default_config();
        let xml = exporter.export(&make_timeline());
        let track_count = xml.matches("<track>").count();
        assert_eq!(track_count, 3, "Expected 3 tracks, got {track_count}");
    }

    #[test]
    fn test_export_contains_clip_items() {
        let exporter = MultiCamXmlExporter::default_config();
        let xml = exporter.export(&make_timeline());
        assert!(xml.contains("<clipitem>"), "Missing <clipitem>");
    }

    #[test]
    fn test_export_xml_escape_in_name() {
        let exporter = MultiCamXmlExporter::new(XmlExportConfig {
            sequence_name: "Test & <Demo>".into(),
            ..XmlExportConfig::default()
        });
        let xml = exporter.export(&make_timeline());
        assert!(
            xml.contains("Test &amp; &lt;Demo&gt;"),
            "XML escaping failed"
        );
    }

    #[test]
    fn test_export_empty_timeline_no_crash() {
        let exporter = MultiCamXmlExporter::default_config();
        let empty = MultiCamTimeline::new(2);
        let xml = exporter.export(&empty);
        assert!(xml.contains("<xmeml"));
    }

    // ── angle_intervals ──────────────────────────────────────────────────────

    #[test]
    fn test_angle_intervals_angle0() {
        let decisions = vec![
            crate::edit::EditDecision::cut(0, 0),
            crate::edit::EditDecision::cut(40, 1),
            crate::edit::EditDecision::cut(80, 2),
        ];
        let intervals = MultiCamXmlExporter::angle_intervals(0, &decisions, 120);
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0], (0, 39));
    }

    #[test]
    fn test_angle_intervals_last_angle() {
        let decisions = vec![
            crate::edit::EditDecision::cut(0, 0),
            crate::edit::EditDecision::cut(60, 2),
        ];
        let intervals = MultiCamXmlExporter::angle_intervals(2, &decisions, 120);
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0], (60, 119));
    }
}
