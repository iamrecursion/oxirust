//! Final Cut Pro XML (FCP 7 / FCPXML) export for clip lists.
//!
//! Generates well-formed FCP 7–style XML that can be imported into Final Cut
//! Pro 7, DaVinci Resolve, Premiere Pro and other NLEs that support the CMX
//! interchange format.  The produced output contains a single `<sequence>`
//! element containing one `<clipitem>` per input `Clip`.
//!
//! The exporter deliberately targets the FCP 7 XML dialect (not FCPXML 1.x)
//! because it is the lowest-common-denominator that every major NLE supports.

use crate::clip::Clip;

/// Converts a frame count to a FCP 7 `<timebase>` rational string `"N/M"`.
fn frames_to_rational(frame: i64, num: i64, den: i64) -> String {
    // FCP 7 XML uses rational values expressed as integer-frames when the
    // time base is exact.  We simply write the frame-offset multiplied by the
    // denominator over the numerator.
    if num == 0 {
        return "0/1".to_string();
    }
    format!("{}/{}", frame * den, num)
}

/// A single FCP 7 `<clipitem>` record.
struct ClipItem<'a> {
    clip: &'a Clip,
    id: usize,
    fps_num: i64,
    fps_den: i64,
}

impl ClipItem<'_> {
    fn render(&self) -> String {
        let in_pt = self.clip.in_point.unwrap_or(0);
        let out_pt = self.clip.out_point.or(self.clip.duration).unwrap_or(0);
        let duration = out_pt - in_pt;

        let src_in = frames_to_rational(in_pt, self.fps_num, self.fps_den);
        let src_out = frames_to_rational(out_pt, self.fps_num, self.fps_den);
        let rec_in = frames_to_rational(0, self.fps_num, self.fps_den);
        let rec_out = frames_to_rational(duration, self.fps_num, self.fps_den);
        let dur_str = frames_to_rational(duration, self.fps_num, self.fps_den);

        let file_path = self.clip.file_path.to_string_lossy();
        let name = xml_escape(&self.clip.name);

        let mut s = String::new();
        s.push_str(&format!("      <clipitem id=\"clipitem-{}\">\n", self.id));
        s.push_str(&format!("        <name>{name}</name>\n"));
        s.push_str(&format!("        <duration>{dur_str}</duration>\n"));
        s.push_str(&format!("        <start>{rec_in}</start>\n"));
        s.push_str(&format!("        <end>{rec_out}</end>\n"));
        s.push_str(&format!("        <in>{src_in}</in>\n"));
        s.push_str(&format!("        <out>{src_out}</out>\n"));
        s.push_str("        <file>\n");
        s.push_str(&format!(
            "          <pathurl>{}</pathurl>\n",
            xml_escape(&file_path)
        ));
        s.push_str("        </file>\n");
        if !self.clip.keywords.is_empty() {
            s.push_str("        <comments>\n");
            let kws = self
                .clip
                .keywords
                .iter()
                .map(|k| xml_escape(k))
                .collect::<Vec<_>>()
                .join(", ");
            s.push_str(&format!(
                "          <mastercomment1>{kws}</mastercomment1>\n"
            ));
            s.push_str("        </comments>\n");
        }
        s.push_str("      </clipitem>\n");
        s
    }
}

/// Escapes a string for use in an XML text node.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Exports a slice of `Clip`s to FCP 7–compatible XML.
///
/// # Frame-rate configuration
///
/// The exporter accepts an optional `(numerator, denominator)` pair for the
/// project frame rate.  If `None` the first clip's `frame_rate` field is
/// consulted; failing that, `24/1` is used as a safe default.
#[derive(Debug, Clone)]
pub struct FcpXmlClipExporter {
    /// Project frame rate numerator.
    fps_num: i64,
    /// Project frame rate denominator.
    fps_den: i64,
    /// Project name used in the `<name>` element of the `<sequence>`.
    project_name: String,
}

impl FcpXmlClipExporter {
    /// Creates an exporter with an explicit frame rate.
    #[must_use]
    pub fn new(fps_num: i64, fps_den: i64, project_name: impl Into<String>) -> Self {
        Self {
            fps_num,
            fps_den,
            project_name: project_name.into(),
        }
    }

    /// Creates an exporter with 24 fps and a default project name.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(24, 1, "Exported Sequence")
    }

    /// Generates and returns the FCP 7 XML string.
    ///
    /// The output is a complete, standalone XML document.
    #[must_use]
    pub fn export(&self, clips: &[Clip]) -> String {
        let total_duration: i64 = clips
            .iter()
            .map(|c| {
                let in_pt = c.in_point.unwrap_or(0);
                let out_pt = c.out_point.or(c.duration).unwrap_or(0);
                (out_pt - in_pt).max(0)
            })
            .sum();

        let total_dur_str = frames_to_rational(total_duration, self.fps_num, self.fps_den);
        let timebase = if self.fps_den == 1 {
            self.fps_num.to_string()
        } else {
            format!("{}/{}", self.fps_num, self.fps_den)
        };

        let project_name = xml_escape(&self.project_name);

        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<!DOCTYPE xmeml>\n");
        xml.push_str("<xmeml version=\"5\">\n");
        xml.push_str("  <sequence>\n");
        xml.push_str(&format!("    <name>{project_name}</name>\n"));
        xml.push_str(&format!("    <duration>{total_dur_str}</duration>\n"));
        xml.push_str("    <rate>\n");
        xml.push_str(&format!("      <timebase>{timebase}</timebase>\n"));
        xml.push_str("      <ntsc>FALSE</ntsc>\n");
        xml.push_str("    </rate>\n");
        xml.push_str("    <media>\n");
        xml.push_str("      <video>\n");
        xml.push_str("        <track>\n");

        for (idx, clip) in clips.iter().enumerate() {
            let item = ClipItem {
                clip,
                id: idx + 1,
                fps_num: self.fps_num,
                fps_den: self.fps_den,
            };
            xml.push_str(&item.render());
        }

        xml.push_str("        </track>\n");
        xml.push_str("      </video>\n");
        xml.push_str("    </media>\n");
        xml.push_str("  </sequence>\n");
        xml.push_str("</xmeml>\n");
        xml
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_clip(name: &str, duration: i64, in_pt: i64, out_pt: i64) -> Clip {
        let mut c = Clip::new(PathBuf::from(format!("/media/{name}.mov")));
        c.set_name(name);
        c.set_duration(duration);
        c.set_in_point(in_pt);
        c.set_out_point(out_pt);
        c
    }

    #[test]
    fn test_export_produces_xml_declaration() {
        let exporter = FcpXmlClipExporter::with_defaults();
        let xml = exporter.export(&[]);
        assert!(xml.starts_with("<?xml version=\"1.0\""));
    }

    #[test]
    fn test_export_contains_xmeml_root() {
        let exporter = FcpXmlClipExporter::with_defaults();
        let xml = exporter.export(&[]);
        assert!(xml.contains("<xmeml version=\"5\">"));
        assert!(xml.contains("</xmeml>"));
    }

    #[test]
    fn test_export_contains_sequence() {
        let exporter = FcpXmlClipExporter::with_defaults();
        let xml = exporter.export(&[]);
        assert!(xml.contains("<sequence>"));
        assert!(xml.contains("</sequence>"));
    }

    #[test]
    fn test_export_contains_project_name() {
        let exporter = FcpXmlClipExporter::new(25, 1, "My Docfilm");
        let xml = exporter.export(&[]);
        assert!(xml.contains("<name>My Docfilm</name>"));
    }

    #[test]
    fn test_export_timebase() {
        let exporter = FcpXmlClipExporter::new(25, 1, "Seq");
        let xml = exporter.export(&[]);
        assert!(xml.contains("<timebase>25</timebase>"));
    }

    #[test]
    fn test_export_single_clip_item() {
        let clip = make_clip("Interview_001", 1000, 0, 240);
        let exporter = FcpXmlClipExporter::new(24, 1, "Test");
        let xml = exporter.export(&[clip]);
        assert!(xml.contains("clipitem-1"));
        assert!(xml.contains("<name>Interview_001</name>"));
    }

    #[test]
    fn test_export_multiple_clips() {
        let clips = vec![
            make_clip("Clip_A", 500, 0, 120),
            make_clip("Clip_B", 500, 0, 120),
        ];
        let exporter = FcpXmlClipExporter::with_defaults();
        let xml = exporter.export(&clips);
        assert!(xml.contains("clipitem-1"));
        assert!(xml.contains("clipitem-2"));
        assert!(xml.contains("Clip_A"));
        assert!(xml.contains("Clip_B"));
    }

    #[test]
    fn test_export_clip_with_keywords_in_comments() {
        let mut clip = Clip::new(PathBuf::from("/media/shot.mov"));
        clip.set_name("Shot");
        clip.set_duration(100);
        clip.set_in_point(0);
        clip.set_out_point(100);
        clip.add_keyword("outdoor");
        clip.add_keyword("sunny");

        let exporter = FcpXmlClipExporter::with_defaults();
        let xml = exporter.export(&[clip]);
        assert!(xml.contains("<comments>"));
        assert!(xml.contains("outdoor"));
        assert!(xml.contains("sunny"));
    }

    #[test]
    fn test_xml_escape_special_chars() {
        let mut clip = Clip::new(PathBuf::from("/media/test.mov"));
        clip.set_name("Scene <Ext> & \"Drama\"");
        clip.set_duration(100);
        clip.set_in_point(0);
        clip.set_out_point(100);

        let exporter = FcpXmlClipExporter::with_defaults();
        let xml = exporter.export(&[clip]);
        assert!(xml.contains("&lt;Ext&gt;"));
        assert!(xml.contains("&amp;"));
        assert!(xml.contains("&quot;Drama&quot;"));
    }

    #[test]
    fn test_export_empty_clips() {
        let exporter = FcpXmlClipExporter::with_defaults();
        let xml = exporter.export(&[]);
        // Should still produce valid XML without clipitems
        assert!(xml.contains("<track>"));
        assert!(!xml.contains("clipitem"));
    }
}
