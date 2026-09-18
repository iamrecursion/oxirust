//! Final Cut Pro 7 XML (FCPXML v5 / xmeml) export
//!
//! Exports an AAF `CompositionMob` to FCP 7 XML (`<xmeml version="5">`),
//! mapping AAF tracks to FCP video/audio tracks and source clips to
//! `<clipitem>` elements with timecode attributes.
//!
//! Reference: Apple Final Cut Pro XML Interchange Format (version 5, FCP 7)

use crate::composition::{CompositionMob, Sequence, SequenceComponent, SourceClip, Track};
use crate::object_model::Header;
use crate::timeline::EditRate;
use crate::{AafError, Result};
use std::fmt::Write as FmtWrite;

/// FCP 7 XML exporter
pub struct FcpXmlExporter {
    /// Whether to include audio tracks (default: true)
    include_audio: bool,
    /// Whether to include video tracks (default: true)
    include_video: bool,
}

impl FcpXmlExporter {
    /// Create a new FCP XML exporter with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            include_audio: true,
            include_video: true,
        }
    }

    /// Set whether to include audio tracks
    #[must_use]
    pub fn with_audio(mut self, include: bool) -> Self {
        self.include_audio = include;
        self
    }

    /// Set whether to include video tracks
    #[must_use]
    pub fn with_video(mut self, include: bool) -> Self {
        self.include_video = include;
        self
    }

    /// Export a composition mob to FCP 7 XML string
    ///
    /// # Errors
    ///
    /// Returns `AafError::ConversionError` if XML string formatting fails.
    pub fn export(&self, composition: &CompositionMob, header: &Header) -> Result<String> {
        let mut xml = String::new();

        // XML declaration
        writeln!(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "<xmeml version=\"5\">")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        // Sequence element
        self.write_sequence_element(composition, header, &mut xml)?;

        writeln!(xml, "</xmeml>").map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(xml)
    }

    fn write_sequence_element(
        &self,
        composition: &CompositionMob,
        _header: &Header,
        xml: &mut String,
    ) -> Result<()> {
        let name = xml_escape(composition.name());
        let duration = composition.duration().unwrap_or(0);
        let edit_rate = composition.edit_rate().unwrap_or(EditRate::PAL_25);

        writeln!(xml, "  <sequence>").map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "    <name>{name}</name>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "    <duration>{duration}</duration>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(
            xml,
            "    <rate><timebase>{}</timebase><ntsc>{}</ntsc></rate>",
            edit_rate_timebase(edit_rate),
            if edit_rate.is_ntsc() { "TRUE" } else { "FALSE" }
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "    <uuid>{}</uuid>", composition.mob_id())
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        writeln!(xml, "    <media>").map_err(|e| AafError::ConversionError(e.to_string()))?;

        // Separate video and audio tracks
        let video_tracks: Vec<Track> = composition
            .tracks()
            .into_iter()
            .filter(|t| t.is_picture())
            .collect();
        let audio_tracks: Vec<Track> = composition
            .tracks()
            .into_iter()
            .filter(|t| t.is_sound())
            .collect();

        if self.include_video {
            writeln!(xml, "      <video>").map_err(|e| AafError::ConversionError(e.to_string()))?;
            for track in &video_tracks {
                self.write_track(track, edit_rate, xml)?;
            }
            writeln!(xml, "      </video>")
                .map_err(|e| AafError::ConversionError(e.to_string()))?;
        }

        if self.include_audio {
            writeln!(xml, "      <audio>").map_err(|e| AafError::ConversionError(e.to_string()))?;
            for track in &audio_tracks {
                self.write_track(track, edit_rate, xml)?;
            }
            writeln!(xml, "      </audio>")
                .map_err(|e| AafError::ConversionError(e.to_string()))?;
        }

        writeln!(xml, "    </media>").map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "  </sequence>").map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(())
    }

    fn write_track(&self, track: &Track, edit_rate: EditRate, xml: &mut String) -> Result<()> {
        let track_name = xml_escape(&track.name);
        writeln!(xml, "        <track>").map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "          <name>{track_name}</name>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        if let Some(ref sequence) = track.sequence {
            self.write_clipitems(sequence, track, edit_rate, xml)?;
        }

        writeln!(xml, "        </track>").map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(())
    }

    fn write_clipitems(
        &self,
        sequence: &Sequence,
        track: &Track,
        edit_rate: EditRate,
        xml: &mut String,
    ) -> Result<()> {
        let mut timeline_pos = 0i64;
        let mut clip_index = 1u32;

        for component in &sequence.components {
            match component {
                SequenceComponent::SourceClip(clip) => {
                    self.write_clipitem(clip, track, clip_index, timeline_pos, edit_rate, xml)?;
                    clip_index += 1;
                    timeline_pos += clip.length;
                }
                SequenceComponent::Filler(filler) => {
                    timeline_pos += filler.length;
                }
                SequenceComponent::Transition(_trans) => {
                    // Transitions do not advance timeline position in FCP XML
                }
                SequenceComponent::Effect(_effect) => {
                    // Effects are ignored in basic FCP export
                }
            }
        }

        Ok(())
    }

    fn write_clipitem(
        &self,
        clip: &SourceClip,
        track: &Track,
        index: u32,
        timeline_in: i64,
        edit_rate: EditRate,
        xml: &mut String,
    ) -> Result<()> {
        let timeline_out = timeline_in + clip.length;
        let source_in = clip.start_time.0;
        let source_out = source_in + clip.length;

        let name = xml_escape(&format!(
            "Clip_{index}_{}",
            &clip.source_mob_id.to_string()[..8]
        ));
        let timebase = edit_rate_timebase(edit_rate);
        let ntsc_str = if edit_rate.is_ntsc() { "TRUE" } else { "FALSE" };
        let tc_in = frames_to_timecode(timeline_in, edit_rate);
        let tc_out = frames_to_timecode(timeline_out, edit_rate);
        let src_tc_in = frames_to_timecode(source_in, edit_rate);
        let src_tc_out = frames_to_timecode(source_out, edit_rate);

        writeln!(xml, "          <clipitem id=\"clipitem-{index}\">")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "            <name>{name}</name>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "            <duration>{}</duration>", clip.length)
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(
            xml,
            "            <rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>"
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "            <in>{tc_in}</in>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "            <out>{tc_out}</out>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "            <start>{tc_in}</start>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "            <end>{tc_out}</end>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(
            xml,
            "            <sourcetrack><mediatype>{}</mediatype></sourcetrack>",
            if track.is_picture() { "video" } else { "audio" }
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "            <file id=\"file-{}\">", clip.source_mob_id)
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(
            xml,
            "              <name>mob_{}</name>",
            &clip.source_mob_id.to_string()[..8]
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(
            xml,
            "              <timecode><string>{src_tc_in}</string><frame>{source_in}</frame></timecode>"
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "              <out>{src_tc_out}</out>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "            </file>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "          </clipitem>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(())
    }
}

impl Default for FcpXmlExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape special XML characters
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Derive FCP timebase integer from EditRate
fn edit_rate_timebase(rate: EditRate) -> i32 {
    let fps = rate.to_float();
    if (fps - 23.976).abs() < 0.01 {
        24
    } else if (fps - 29.97).abs() < 0.01 {
        30
    } else if (fps - 59.94).abs() < 0.01 {
        60
    } else {
        fps.round() as i32
    }
}

/// Convert a frame count to HH:MM:SS:FF timecode string
fn frames_to_timecode(frames: i64, rate: EditRate) -> String {
    let fps = edit_rate_timebase(rate) as i64;
    if fps == 0 {
        return "00:00:00:00".to_string();
    }
    let abs_frames = frames.unsigned_abs();
    let ff = (abs_frames % fps as u64) as i32;
    let total_secs = abs_frames / fps as u64;
    let ss = (total_secs % 60) as i32;
    let total_min = total_secs / 60;
    let mm = (total_min % 60) as i32;
    let hh = (total_min / 60) as i32;
    let sign = if frames < 0 { "-" } else { "" };
    format!("{sign}{hh:02}:{mm:02}:{ss:02}:{ff:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        CompositionMob, Filler, Sequence, SequenceComponent, SourceClip, Track, TrackType,
    };
    use crate::dictionary::Auid;
    use crate::object_model::Header;
    use crate::timeline::{EditRate, Position};
    use uuid::Uuid;

    fn make_test_composition() -> CompositionMob {
        let source1 = Uuid::new_v4();
        let source2 = Uuid::new_v4();

        let mut comp = CompositionMob::new(Uuid::new_v4(), "My Edit");

        // Video track
        let mut vid_seq = Sequence::new(Auid::PICTURE);
        vid_seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            100,
            Position::zero(),
            source1,
            1,
        )));
        vid_seq.add_component(SequenceComponent::Filler(Filler::new(10)));
        vid_seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            50,
            Position::new(100),
            source2,
            1,
        )));
        let mut vid_track = Track::new(1, "V1", EditRate::PAL_25, TrackType::Picture);
        vid_track.set_sequence(vid_seq);
        comp.add_track(vid_track);

        // Audio track
        let mut aud_seq = Sequence::new(Auid::SOUND);
        aud_seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            150,
            Position::zero(),
            source1,
            2,
        )));
        let mut aud_track = Track::new(2, "A1", EditRate::PAL_25, TrackType::Sound);
        aud_track.set_sequence(aud_seq);
        comp.add_track(aud_track);

        comp
    }

    #[test]
    fn test_fcp_xml_root_element() {
        let comp = make_test_composition();
        let header = Header::new();
        let exporter = FcpXmlExporter::new();
        let xml = exporter
            .export(&comp, &header)
            .expect("export should succeed");
        assert!(xml.contains("<xmeml version=\"5\">"));
        assert!(xml.contains("</xmeml>"));
    }

    #[test]
    fn test_fcp_xml_sequence_name() {
        let comp = make_test_composition();
        let header = Header::new();
        let exporter = FcpXmlExporter::new();
        let xml = exporter
            .export(&comp, &header)
            .expect("export should succeed");
        assert!(xml.contains("<name>My Edit</name>"));
    }

    #[test]
    fn test_fcp_xml_has_video_track() {
        let comp = make_test_composition();
        let header = Header::new();
        let exporter = FcpXmlExporter::new();
        let xml = exporter
            .export(&comp, &header)
            .expect("export should succeed");
        assert!(xml.contains("<video>"));
        assert!(xml.contains("</video>"));
    }

    #[test]
    fn test_fcp_xml_has_audio_track() {
        let comp = make_test_composition();
        let header = Header::new();
        let exporter = FcpXmlExporter::new();
        let xml = exporter
            .export(&comp, &header)
            .expect("export should succeed");
        assert!(xml.contains("<audio>"));
        assert!(xml.contains("</audio>"));
    }

    #[test]
    fn test_fcp_xml_clipitem_present() {
        let comp = make_test_composition();
        let header = Header::new();
        let exporter = FcpXmlExporter::new();
        let xml = exporter
            .export(&comp, &header)
            .expect("export should succeed");
        assert!(xml.contains("<clipitem"));
        assert!(xml.contains("</clipitem>"));
    }

    #[test]
    fn test_fcp_xml_video_only() {
        let comp = make_test_composition();
        let header = Header::new();
        let exporter = FcpXmlExporter::new().with_audio(false);
        let xml = exporter
            .export(&comp, &header)
            .expect("export should succeed");
        assert!(!xml.contains("<audio>"));
        assert!(xml.contains("<video>"));
    }

    #[test]
    fn test_fcp_xml_audio_only() {
        let comp = make_test_composition();
        let header = Header::new();
        let exporter = FcpXmlExporter::new().with_video(false);
        let xml = exporter
            .export(&comp, &header)
            .expect("export should succeed");
        assert!(xml.contains("<audio>"));
        assert!(!xml.contains("<video>"));
    }

    #[test]
    fn test_fcp_xml_rate_element() {
        let comp = make_test_composition();
        let header = Header::new();
        let exporter = FcpXmlExporter::new();
        let xml = exporter
            .export(&comp, &header)
            .expect("export should succeed");
        assert!(xml.contains("<timebase>25</timebase>"));
        assert!(xml.contains("<ntsc>FALSE</ntsc>"));
    }

    #[test]
    fn test_fcp_xml_duration_element() {
        let comp = make_test_composition();
        let header = Header::new();
        let exporter = FcpXmlExporter::new();
        let xml = exporter
            .export(&comp, &header)
            .expect("export should succeed");
        // duration of video track: 100 + 10 (filler) + 50 = 160
        assert!(xml.contains("<duration>160</duration>"));
    }

    #[test]
    fn test_fcp_xml_uuid_element() {
        let comp = make_test_composition();
        let mob_id = comp.mob_id();
        let header = Header::new();
        let exporter = FcpXmlExporter::new();
        let xml = exporter
            .export(&comp, &header)
            .expect("export should succeed");
        assert!(xml.contains(&format!("<uuid>{mob_id}</uuid>")));
    }

    #[test]
    fn test_frames_to_timecode_pal() {
        let rate = EditRate::PAL_25;
        assert_eq!(frames_to_timecode(0, rate), "00:00:00:00");
        assert_eq!(frames_to_timecode(25, rate), "00:00:01:00");
        assert_eq!(frames_to_timecode(75, rate), "00:00:03:00");
        assert_eq!(frames_to_timecode(26, rate), "00:00:01:01");
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("a & b"), "a &amp; b");
        assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
        assert_eq!(xml_escape("\"quote\""), "&quot;quote&quot;");
    }
}
