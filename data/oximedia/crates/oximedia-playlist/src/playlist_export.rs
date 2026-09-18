#![allow(dead_code)]
//! Playlist export to standard formats (M3U, M3U8, XSPF, PLS).

/// Supported playlist export formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportFormat {
    /// Legacy M3U format.
    M3u,
    /// Extended M3U / HLS format.
    M3u8,
    /// XML Shareable Playlist Format.
    Xspf,
    /// Winamp PLS format.
    Pls,
}

impl ExportFormat {
    /// Returns the MIME type string for the format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            ExportFormat::M3u => "audio/x-mpegurl",
            ExportFormat::M3u8 => "application/vnd.apple.mpegurl",
            ExportFormat::Xspf => "application/xspf+xml",
            ExportFormat::Pls => "audio/x-scpls",
        }
    }

    /// Returns the file extension for the format.
    pub fn file_extension(&self) -> &'static str {
        match self {
            ExportFormat::M3u => "m3u",
            ExportFormat::M3u8 => "m3u8",
            ExportFormat::Xspf => "xspf",
            ExportFormat::Pls => "pls",
        }
    }

    /// Returns true if the format supports extended metadata tags.
    pub fn supports_extended_tags(&self) -> bool {
        matches!(self, ExportFormat::M3u8 | ExportFormat::Xspf)
    }
}

/// A single entry in an exportable playlist.
#[derive(Debug, Clone)]
pub struct ExportEntry {
    /// URI or path to the media file.
    pub uri: String,
    /// Optional display title.
    pub title: Option<String>,
    /// Optional duration in seconds.
    pub duration_secs: Option<f64>,
    /// Optional artist name.
    pub artist: Option<String>,
}

impl ExportEntry {
    /// Creates a new export entry with the given URI.
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            title: None,
            duration_secs: None,
            artist: None,
        }
    }

    /// Attaches a title to the entry.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Attaches a duration to the entry.
    pub fn with_duration(mut self, secs: f64) -> Self {
        self.duration_secs = Some(secs);
        self
    }

    /// Attaches an artist name to the entry.
    pub fn with_artist(mut self, artist: impl Into<String>) -> Self {
        self.artist = Some(artist.into());
        self
    }
}

/// Exports playlists to various standard formats.
#[derive(Debug, Default)]
pub struct PlaylistExporter {
    entries: Vec<ExportEntry>,
}

impl PlaylistExporter {
    /// Creates a new empty exporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an entry to the export list.
    pub fn add_entry(&mut self, entry: ExportEntry) {
        self.entries.push(entry);
    }

    /// Returns the number of entries queued for export.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Exports the playlist to the given format, returning the serialised string.
    pub fn export(&self, format: &ExportFormat) -> String {
        match format {
            ExportFormat::M3u => self.export_m3u(),
            ExportFormat::M3u8 => self.export_m3u8(),
            ExportFormat::Xspf => self.export_xspf(),
            ExportFormat::Pls => self.export_pls(),
        }
    }

    fn export_m3u(&self) -> String {
        let mut out = String::from("#EXTM3U\n");
        for e in &self.entries {
            let dur = e.duration_secs.unwrap_or(-1.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let dur_int = dur as i64;
            let label = e.title.clone().unwrap_or_else(|| e.uri.clone());
            out.push_str(&format!("#EXTINF:{},{}\n{}\n", dur_int, label, e.uri));
        }
        out
    }

    fn export_m3u8(&self) -> String {
        // M3U8 shares format with M3U but uses UTF-8 header comment
        let mut out = String::from("#EXTM3U\n# Encoding: UTF-8\n");
        for e in &self.entries {
            let dur = e.duration_secs.unwrap_or(-1.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let dur_int = dur as i64;
            let label = e.title.clone().unwrap_or_else(|| e.uri.clone());
            if let Some(ref artist) = e.artist {
                out.push_str(&format!(
                    "#EXTINF:{},artist=\"{}\" tvg-name=\"{}\"\n{}\n",
                    dur_int, artist, label, e.uri
                ));
            } else {
                out.push_str(&format!("#EXTINF:{},{}\n{}\n", dur_int, label, e.uri));
            }
        }
        out
    }

    fn export_xspf(&self) -> String {
        let mut out = String::from(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <playlist version=\"1\" xmlns=\"http://xspf.org/ns/0/\">\n  <trackList>\n",
        );
        for e in &self.entries {
            out.push_str("    <track>\n");
            out.push_str(&format!("      <location>{}</location>\n", e.uri));
            if let Some(ref t) = e.title {
                out.push_str(&format!("      <title>{t}</title>\n"));
            }
            if let Some(ref a) = e.artist {
                out.push_str(&format!("      <creator>{a}</creator>\n"));
            }
            if let Some(dur) = e.duration_secs {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let dur_ms = (dur * 1000.0) as u64;
                out.push_str(&format!("      <duration>{dur_ms}</duration>\n"));
            }
            out.push_str("    </track>\n");
        }
        out.push_str("  </trackList>\n</playlist>\n");
        out
    }

    fn export_pls(&self) -> String {
        let mut out = String::from("[playlist]\n");
        for (i, e) in self.entries.iter().enumerate() {
            let n = i + 1;
            out.push_str(&format!("File{}={}\n", n, e.uri));
            if let Some(ref t) = e.title {
                out.push_str(&format!("Title{n}={t}\n"));
            }
            if let Some(dur) = e.duration_secs {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let dur_int = dur as i64;
                out.push_str(&format!("Length{n}={dur_int}\n"));
            }
        }
        out.push_str(&format!("NumberOfEntries={}\n", self.entries.len()));
        out.push_str("Version=2\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mime_type_m3u() {
        assert_eq!(ExportFormat::M3u.mime_type(), "audio/x-mpegurl");
    }

    #[test]
    fn test_mime_type_m3u8() {
        assert_eq!(
            ExportFormat::M3u8.mime_type(),
            "application/vnd.apple.mpegurl"
        );
    }

    #[test]
    fn test_mime_type_xspf() {
        assert_eq!(ExportFormat::Xspf.mime_type(), "application/xspf+xml");
    }

    #[test]
    fn test_mime_type_pls() {
        assert_eq!(ExportFormat::Pls.mime_type(), "audio/x-scpls");
    }

    #[test]
    fn test_file_extension() {
        assert_eq!(ExportFormat::M3u.file_extension(), "m3u");
        assert_eq!(ExportFormat::M3u8.file_extension(), "m3u8");
        assert_eq!(ExportFormat::Xspf.file_extension(), "xspf");
        assert_eq!(ExportFormat::Pls.file_extension(), "pls");
    }

    #[test]
    fn test_supports_extended_tags() {
        assert!(!ExportFormat::M3u.supports_extended_tags());
        assert!(ExportFormat::M3u8.supports_extended_tags());
        assert!(ExportFormat::Xspf.supports_extended_tags());
        assert!(!ExportFormat::Pls.supports_extended_tags());
    }

    #[test]
    fn test_entry_count_empty() {
        let exp = PlaylistExporter::new();
        assert_eq!(exp.entry_count(), 0);
    }

    #[test]
    fn test_entry_count_after_add() {
        let mut exp = PlaylistExporter::new();
        exp.add_entry(ExportEntry::new("file1.mp3"));
        exp.add_entry(ExportEntry::new("file2.mp3"));
        assert_eq!(exp.entry_count(), 2);
    }

    #[test]
    fn test_export_m3u_contains_extinf() {
        let mut exp = PlaylistExporter::new();
        exp.add_entry(
            ExportEntry::new("track.mp3")
                .with_title("My Track")
                .with_duration(120.0),
        );
        let out = exp.export(&ExportFormat::M3u);
        assert!(out.contains("#EXTM3U"));
        assert!(out.contains("#EXTINF:120,My Track"));
        assert!(out.contains("track.mp3"));
    }

    #[test]
    fn test_export_m3u8_utf8_header() {
        let mut exp = PlaylistExporter::new();
        exp.add_entry(
            ExportEntry::new("v.mp4")
                .with_title("Video")
                .with_duration(60.0),
        );
        let out = exp.export(&ExportFormat::M3u8);
        assert!(out.contains("UTF-8"));
        assert!(out.contains("#EXTINF:60,Video"));
    }

    #[test]
    fn test_export_xspf_xml_structure() {
        let mut exp = PlaylistExporter::new();
        exp.add_entry(
            ExportEntry::new("song.flac")
                .with_title("Song")
                .with_artist("Artist")
                .with_duration(200.0),
        );
        let out = exp.export(&ExportFormat::Xspf);
        assert!(out.contains("<playlist"));
        assert!(out.contains("<trackList>"));
        assert!(out.contains("<location>song.flac</location>"));
        assert!(out.contains("<title>Song</title>"));
        assert!(out.contains("<creator>Artist</creator>"));
        assert!(out.contains("<duration>200000</duration>"));
    }

    #[test]
    fn test_export_pls_structure() {
        let mut exp = PlaylistExporter::new();
        exp.add_entry(
            ExportEntry::new("a.mp3")
                .with_title("Track A")
                .with_duration(90.0),
        );
        let out = exp.export(&ExportFormat::Pls);
        assert!(out.contains("[playlist]"));
        assert!(out.contains("File1=a.mp3"));
        assert!(out.contains("Title1=Track A"));
        assert!(out.contains("Length1=90"));
        assert!(out.contains("NumberOfEntries=1"));
        assert!(out.contains("Version=2"));
    }

    #[test]
    fn test_export_multiple_entries_pls() {
        let mut exp = PlaylistExporter::new();
        exp.add_entry(ExportEntry::new("a.mp3"));
        exp.add_entry(ExportEntry::new("b.mp3"));
        exp.add_entry(ExportEntry::new("c.mp3"));
        let out = exp.export(&ExportFormat::Pls);
        assert!(out.contains("NumberOfEntries=3"));
        assert!(out.contains("File3=c.mp3"));
    }
}
