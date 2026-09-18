//! XMLTV format export for EPG data.

use super::generate::ProgramEntry;
use crate::Result;
use std::io::Write;

/// XMLTV exporter for EPG data.
pub struct XmltvExporter;

impl XmltvExporter {
    /// Exports programs to XMLTV format.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn export<W: Write>(programs: &[ProgramEntry], writer: &mut W) -> Result<()> {
        // Write XML header
        writeln!(writer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
        writeln!(writer, "<!DOCTYPE tv SYSTEM \"xmltv.dtd\">")?;
        writeln!(writer, "<tv>")?;

        // Group programs by channel and write channel info
        let mut channels: Vec<String> = programs.iter().map(|p| p.channel_id.clone()).collect();
        channels.sort();
        channels.dedup();

        for channel_id in &channels {
            Self::write_channel(writer, channel_id)?;
        }

        // Write program entries
        for program in programs {
            Self::write_program(writer, program)?;
        }

        writeln!(writer, "</tv>")?;
        Ok(())
    }

    fn write_channel<W: Write>(writer: &mut W, channel_id: &str) -> Result<()> {
        writeln!(
            writer,
            "  <channel id=\"{}\">",
            Self::escape_xml(channel_id)
        )?;
        writeln!(
            writer,
            "    <display-name>{}</display-name>",
            Self::escape_xml(channel_id)
        )?;
        writeln!(writer, "  </channel>")?;
        Ok(())
    }

    fn write_program<W: Write>(writer: &mut W, program: &ProgramEntry) -> Result<()> {
        let start = program.start_time.format("%Y%m%d%H%M%S %z");
        let end = program.end_time.format("%Y%m%d%H%M%S %z");

        writeln!(
            writer,
            "  <programme start=\"{}\" stop=\"{}\" channel=\"{}\">",
            start,
            end,
            Self::escape_xml(&program.channel_id)
        )?;

        // Titles — one element per language, with lang attribute per XMLTV spec.
        for (lang, text) in &program.title {
            writeln!(
                writer,
                "    <title lang=\"{}\">{}</title>",
                Self::escape_xml(lang),
                Self::escape_xml(text)
            )?;
        }

        // Descriptions — one element per language.
        for (lang, text) in &program.description {
            writeln!(
                writer,
                "    <desc lang=\"{}\">{}</desc>",
                Self::escape_xml(lang),
                Self::escape_xml(text)
            )?;
        }

        // Episode info
        if let (Some(season), Some(episode)) = (program.season, program.episode) {
            writeln!(
                writer,
                "    <episode-num system=\"onscreen\">S{season:02}E{episode:02}</episode-num>"
            )?;
            writeln!(
                writer,
                "    <episode-num system=\"xmltv_ns\">{}.{}.0/1</episode-num>",
                season - 1,
                episode - 1
            )?;
        }

        // Rating
        if let Some(rating) = &program.rating {
            writeln!(writer, "    <rating>")?;
            writeln!(writer, "      <value>{}</value>", Self::escape_xml(rating))?;
            writeln!(writer, "    </rating>")?;
        }

        // Genres
        for genre in &program.genres {
            writeln!(
                writer,
                "    <category>{}</category>",
                Self::escape_xml(genre)
            )?;
        }

        // Flags
        if program.is_live {
            writeln!(writer, "    <live />")?;
        }
        if program.is_premiere {
            writeln!(writer, "    <premiere />")?;
        }
        if program.is_repeat {
            writeln!(writer, "    <previously-shown />")?;
        }

        writeln!(writer, "  </programme>")?;
        Ok(())
    }

    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::time::Duration;

    #[test]
    fn test_xmltv_export() {
        let program = ProgramEntry::new(
            "Test Show",
            "channel1",
            Utc::now(),
            Duration::from_secs(3600),
        )
        .with_description("A test show")
        .with_episode(1, 5)
        .with_rating("TV-PG")
        .with_genre("Drama")
        .as_premiere();

        let programs = vec![program];
        let mut output = Vec::new();

        XmltvExporter::export(&programs, &mut output).expect("should succeed in test");

        let xml = String::from_utf8(output).expect("should succeed in test");
        assert!(xml.contains("<?xml version"));
        assert!(xml.contains("<tv>"));
        assert!(xml.contains("Test Show"));
        assert!(xml.contains("Drama"));
        assert!(xml.contains("<premiere />"));
        // Lang attribute must be present on the title element.
        assert!(xml.contains("lang=\"en\""), "expected lang=en in: {xml}");
    }

    #[test]
    fn test_epg_multilang_xmltv_output() {
        let program =
            ProgramEntry::new("Morning News", "ch1", Utc::now(), Duration::from_secs(1800))
                .with_title("ja", "モーニングニュース")
                .with_description_lang("ja", "朝のニュース番組");

        let programs = vec![program];
        let mut output = Vec::new();
        XmltvExporter::export(&programs, &mut output).expect("export should succeed");

        let xml = String::from_utf8(output).expect("should be valid utf-8");
        assert!(xml.contains("lang=\"en\""), "missing en title: {xml}");
        assert!(xml.contains("lang=\"ja\""), "missing ja title: {xml}");
        assert!(xml.contains("Morning News"), "missing en title text: {xml}");
        assert!(
            xml.contains("モーニングニュース"),
            "missing ja title text: {xml}"
        );
    }

    #[test]
    fn test_xml_escaping() {
        assert_eq!(
            XmltvExporter::escape_xml("Test & <Show>"),
            "Test &amp; &lt;Show&gt;"
        );
    }
}
