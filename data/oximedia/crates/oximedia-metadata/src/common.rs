//! Common metadata fields that are supported across all formats.
//!
//! This module provides a unified interface for working with common metadata fields
//! like title, artist, album, etc., regardless of the underlying format.

use crate::{Metadata, MetadataFormat, MetadataValue, Picture};

/// Common metadata fields that are supported across all formats.
#[derive(Debug, Clone, Default)]
pub struct CommonFields {
    /// Track title
    pub title: Option<String>,
    /// Artist name
    pub artist: Option<String>,
    /// Album title
    pub album: Option<String>,
    /// Album artist
    pub album_artist: Option<String>,
    /// Track number
    pub track_number: Option<u32>,
    /// Total tracks
    pub total_tracks: Option<u32>,
    /// Disc number
    pub disc_number: Option<u32>,
    /// Total discs
    pub total_discs: Option<u32>,
    /// Release year
    pub year: Option<u32>,
    /// Release date (ISO 8601 format)
    pub date: Option<String>,
    /// Genre
    pub genre: Option<String>,
    /// Comment
    pub comment: Option<String>,
    /// Composer
    pub composer: Option<String>,
    /// Conductor
    pub conductor: Option<String>,
    /// Lyricist
    pub lyricist: Option<String>,
    /// Copyright
    pub copyright: Option<String>,
    /// Publisher
    pub publisher: Option<String>,
    /// ISRC (International Standard Recording Code)
    pub isrc: Option<String>,
    /// Barcode (UPC/EAN)
    pub barcode: Option<String>,
    /// BPM (Beats Per Minute)
    pub bpm: Option<u32>,
    /// Musical key
    pub key: Option<String>,
    /// Mood
    pub mood: Option<String>,
    /// Rating (0-100)
    pub rating: Option<u8>,
    /// Lyrics (unsynchronized)
    pub lyrics: Option<String>,
    /// Synchronized lyrics
    pub synced_lyrics: Option<String>,
    /// Cover art
    pub cover_art: Option<Picture>,
    /// Additional pictures
    pub pictures: Vec<Picture>,
    /// Episode title (for podcasts)
    pub episode_title: Option<String>,
    /// Show name (for podcasts)
    pub show_name: Option<String>,
    /// Episode number (for podcasts)
    pub episode_number: Option<u32>,
    /// Season number (for podcasts)
    pub season_number: Option<u32>,
    /// Encoder
    pub encoder: Option<String>,
    /// Encoded by
    pub encoded_by: Option<String>,
    /// Language (ISO 639-2 code)
    pub language: Option<String>,
    /// Original artist
    pub original_artist: Option<String>,
    /// Original album
    pub original_album: Option<String>,
    /// Original year
    pub original_year: Option<u32>,
    /// Remixer
    pub remixer: Option<String>,
    /// Compilation flag
    pub compilation: Option<bool>,
    /// Grouping
    pub grouping: Option<String>,
    /// Work name
    pub work: Option<String>,
    /// Movement name
    pub movement: Option<String>,
    /// Movement number
    pub movement_number: Option<u32>,
    /// Movement count
    pub movement_count: Option<u32>,
}

impl CommonFields {
    /// Create a new empty common fields struct.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract common fields from metadata.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn from_metadata(metadata: &Metadata) -> Self {
        let mut fields = Self::new();

        // Extract based on format
        match metadata.format() {
            MetadataFormat::Id3v2 => {
                fields.extract_id3v2(metadata);
            }
            MetadataFormat::VorbisComments => {
                fields.extract_vorbis(metadata);
            }
            MetadataFormat::Apev2 => {
                fields.extract_ape(metadata);
            }
            MetadataFormat::iTunes => {
                fields.extract_itunes(metadata);
            }
            MetadataFormat::Xmp => {
                fields.extract_xmp(metadata);
            }
            MetadataFormat::Exif => {
                fields.extract_exif(metadata);
            }
            MetadataFormat::Iptc => {
                fields.extract_iptc(metadata);
            }
            MetadataFormat::QuickTime => {
                fields.extract_quicktime(metadata);
            }
            MetadataFormat::Matroska => {
                fields.extract_matroska(metadata);
            }
        }

        fields
    }

    /// Apply common fields to metadata.
    #[allow(clippy::too_many_lines)]
    pub fn apply_to_metadata(&self, metadata: &mut Metadata) {
        match metadata.format() {
            MetadataFormat::Id3v2 => {
                self.apply_id3v2(metadata);
            }
            MetadataFormat::VorbisComments => {
                self.apply_vorbis(metadata);
            }
            MetadataFormat::Apev2 => {
                self.apply_ape(metadata);
            }
            MetadataFormat::iTunes => {
                self.apply_itunes(metadata);
            }
            MetadataFormat::Xmp => {
                self.apply_xmp(metadata);
            }
            MetadataFormat::Exif => {
                self.apply_exif(metadata);
            }
            MetadataFormat::Iptc => {
                self.apply_iptc(metadata);
            }
            MetadataFormat::QuickTime => {
                self.apply_quicktime(metadata);
            }
            MetadataFormat::Matroska => {
                self.apply_matroska(metadata);
            }
        }
    }

    // Extract methods for each format
    fn extract_id3v2(&mut self, metadata: &Metadata) {
        self.title = metadata
            .get("TIT2")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.artist = metadata
            .get("TPE1")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.album = metadata
            .get("TALB")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.album_artist = metadata
            .get("TPE2")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.genre = metadata
            .get("TCON")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.year = metadata
            .get("TYER")
            .and_then(|v| v.as_text())
            .and_then(|s| s.parse().ok());
        self.date = metadata
            .get("TDRC")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.comment = metadata
            .get("COMM")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.composer = metadata
            .get("TCOM")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.conductor = metadata
            .get("TPE3")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.lyricist = metadata
            .get("TEXT")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.copyright = metadata
            .get("TCOP")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.publisher = metadata
            .get("TPUB")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.isrc = metadata
            .get("TSRC")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.bpm = metadata
            .get("TBPM")
            .and_then(|v| v.as_text())
            .and_then(|s| s.parse().ok());
        self.lyrics = metadata
            .get("USLT")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.encoder = metadata
            .get("TSSE")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.language = metadata
            .get("TLAN")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.grouping = metadata
            .get("TIT1")
            .and_then(|v| v.as_text())
            .map(String::from);

        // Track number (TRCK format: "track/total")
        if let Some(trck) = metadata.get("TRCK").and_then(|v| v.as_text()) {
            let parts: Vec<&str> = trck.split('/').collect();
            self.track_number = parts.first().and_then(|s| s.parse().ok());
            self.total_tracks = parts.get(1).and_then(|s| s.parse().ok());
        }

        // Disc number (TPOS format: "disc/total")
        if let Some(tpos) = metadata.get("TPOS").and_then(|v| v.as_text()) {
            let parts: Vec<&str> = tpos.split('/').collect();
            self.disc_number = parts.first().and_then(|s| s.parse().ok());
            self.total_discs = parts.get(1).and_then(|s| s.parse().ok());
        }

        // Cover art (APIC frame)
        if let Some(pic) = metadata.get("APIC").and_then(|v| v.as_picture()) {
            self.cover_art = Some(pic.clone());
        }
    }

    fn extract_vorbis(&mut self, metadata: &Metadata) {
        self.title = metadata
            .get("TITLE")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.artist = metadata
            .get("ARTIST")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.album = metadata
            .get("ALBUM")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.album_artist = metadata
            .get("ALBUMARTIST")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.genre = metadata
            .get("GENRE")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.date = metadata
            .get("DATE")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.comment = metadata
            .get("COMMENT")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.composer = metadata
            .get("COMPOSER")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.conductor = metadata
            .get("CONDUCTOR")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.lyricist = metadata
            .get("LYRICIST")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.copyright = metadata
            .get("COPYRIGHT")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.publisher = metadata
            .get("PUBLISHER")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.isrc = metadata
            .get("ISRC")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.track_number = metadata
            .get("TRACKNUMBER")
            .and_then(|v| v.as_text())
            .and_then(|s| s.parse().ok());
        self.total_tracks = metadata
            .get("TOTALTRACKS")
            .and_then(|v| v.as_text())
            .and_then(|s| s.parse().ok());
        self.disc_number = metadata
            .get("DISCNUMBER")
            .and_then(|v| v.as_text())
            .and_then(|s| s.parse().ok());
        self.total_discs = metadata
            .get("TOTALDISCS")
            .and_then(|v| v.as_text())
            .and_then(|s| s.parse().ok());
        self.encoder = metadata
            .get("ENCODER")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.language = metadata
            .get("LANGUAGE")
            .and_then(|v| v.as_text())
            .map(String::from);
    }

    fn extract_ape(&mut self, metadata: &Metadata) {
        self.title = metadata
            .get("Title")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.artist = metadata
            .get("Artist")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.album = metadata
            .get("Album")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.album_artist = metadata
            .get("Album Artist")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.genre = metadata
            .get("Genre")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.year = metadata
            .get("Year")
            .and_then(|v| v.as_text())
            .and_then(|s| s.parse().ok());
        self.comment = metadata
            .get("Comment")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.composer = metadata
            .get("Composer")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.track_number = metadata
            .get("Track")
            .and_then(|v| v.as_text())
            .and_then(|s| s.parse().ok());
    }

    fn extract_itunes(&mut self, metadata: &Metadata) {
        self.title = metadata
            .get("©nam")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.artist = metadata
            .get("©ART")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.album = metadata
            .get("©alb")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.album_artist = metadata
            .get("aART")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.genre = metadata
            .get("©gen")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.date = metadata
            .get("©day")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.comment = metadata
            .get("©cmt")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.composer = metadata
            .get("©wrt")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.copyright = metadata
            .get("cprt")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.encoder = metadata
            .get("©too")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.grouping = metadata
            .get("©grp")
            .and_then(|v| v.as_text())
            .map(String::from);
    }

    fn extract_xmp(&mut self, metadata: &Metadata) {
        self.title = metadata
            .get("dc:title")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.artist = metadata
            .get("dc:creator")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.copyright = metadata
            .get("dc:rights")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.date = metadata
            .get("xmp:CreateDate")
            .and_then(|v| v.as_text())
            .map(String::from);
    }

    fn extract_exif(&mut self, metadata: &Metadata) {
        self.artist = metadata
            .get("Artist")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.copyright = metadata
            .get("Copyright")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.date = metadata
            .get("DateTime")
            .and_then(|v| v.as_text())
            .map(String::from);
    }

    fn extract_iptc(&mut self, metadata: &Metadata) {
        self.title = metadata
            .get("ObjectName")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.artist = metadata
            .get("By-line")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.copyright = metadata
            .get("CopyrightNotice")
            .and_then(|v| v.as_text())
            .map(String::from);
    }

    fn extract_quicktime(&mut self, metadata: &Metadata) {
        self.title = metadata
            .get("©nam")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.artist = metadata
            .get("©ART")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.album = metadata
            .get("©alb")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.comment = metadata
            .get("©cmt")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.copyright = metadata
            .get("©cpy")
            .and_then(|v| v.as_text())
            .map(String::from);
    }

    fn extract_matroska(&mut self, metadata: &Metadata) {
        self.title = metadata
            .get("TITLE")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.artist = metadata
            .get("ARTIST")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.album = metadata
            .get("ALBUM")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.date = metadata
            .get("DATE_RELEASED")
            .and_then(|v| v.as_text())
            .map(String::from);
        self.comment = metadata
            .get("COMMENT")
            .and_then(|v| v.as_text())
            .map(String::from);
    }

    // Apply methods for each format
    fn apply_id3v2(&self, metadata: &mut Metadata) {
        if let Some(ref title) = self.title {
            metadata.insert("TIT2".to_string(), MetadataValue::Text(title.clone()));
        }
        if let Some(ref artist) = self.artist {
            metadata.insert("TPE1".to_string(), MetadataValue::Text(artist.clone()));
        }
        if let Some(ref album) = self.album {
            metadata.insert("TALB".to_string(), MetadataValue::Text(album.clone()));
        }
        if let Some(ref album_artist) = self.album_artist {
            metadata.insert(
                "TPE2".to_string(),
                MetadataValue::Text(album_artist.clone()),
            );
        }
        if let Some(ref genre) = self.genre {
            metadata.insert("TCON".to_string(), MetadataValue::Text(genre.clone()));
        }
        if let Some(ref date) = self.date {
            metadata.insert("TDRC".to_string(), MetadataValue::Text(date.clone()));
        }
        if let Some(ref comment) = self.comment {
            metadata.insert("COMM".to_string(), MetadataValue::Text(comment.clone()));
        }
        if let Some(ref composer) = self.composer {
            metadata.insert("TCOM".to_string(), MetadataValue::Text(composer.clone()));
        }
        if let Some(ref cover_art) = self.cover_art {
            metadata.insert(
                "APIC".to_string(),
                MetadataValue::Picture(cover_art.clone()),
            );
        }

        // Track number
        if let Some(track) = self.track_number {
            let trck = if let Some(total) = self.total_tracks {
                format!("{track}/{total}")
            } else {
                track.to_string()
            };
            metadata.insert("TRCK".to_string(), MetadataValue::Text(trck));
        }

        // Disc number
        if let Some(disc) = self.disc_number {
            let tpos = if let Some(total) = self.total_discs {
                format!("{disc}/{total}")
            } else {
                disc.to_string()
            };
            metadata.insert("TPOS".to_string(), MetadataValue::Text(tpos));
        }
    }

    fn apply_vorbis(&self, metadata: &mut Metadata) {
        if let Some(ref title) = self.title {
            metadata.insert("TITLE".to_string(), MetadataValue::Text(title.clone()));
        }
        if let Some(ref artist) = self.artist {
            metadata.insert("ARTIST".to_string(), MetadataValue::Text(artist.clone()));
        }
        if let Some(ref album) = self.album {
            metadata.insert("ALBUM".to_string(), MetadataValue::Text(album.clone()));
        }
        if let Some(ref album_artist) = self.album_artist {
            metadata.insert(
                "ALBUMARTIST".to_string(),
                MetadataValue::Text(album_artist.clone()),
            );
        }
        if let Some(ref genre) = self.genre {
            metadata.insert("GENRE".to_string(), MetadataValue::Text(genre.clone()));
        }
        if let Some(ref date) = self.date {
            metadata.insert("DATE".to_string(), MetadataValue::Text(date.clone()));
        }
        if let Some(track) = self.track_number {
            metadata.insert(
                "TRACKNUMBER".to_string(),
                MetadataValue::Text(track.to_string()),
            );
        }
        if let Some(total) = self.total_tracks {
            metadata.insert(
                "TOTALTRACKS".to_string(),
                MetadataValue::Text(total.to_string()),
            );
        }
    }

    fn apply_ape(&self, metadata: &mut Metadata) {
        if let Some(ref title) = self.title {
            metadata.insert("Title".to_string(), MetadataValue::Text(title.clone()));
        }
        if let Some(ref artist) = self.artist {
            metadata.insert("Artist".to_string(), MetadataValue::Text(artist.clone()));
        }
        if let Some(ref album) = self.album {
            metadata.insert("Album".to_string(), MetadataValue::Text(album.clone()));
        }
    }

    fn apply_itunes(&self, metadata: &mut Metadata) {
        if let Some(ref title) = self.title {
            metadata.insert("©nam".to_string(), MetadataValue::Text(title.clone()));
        }
        if let Some(ref artist) = self.artist {
            metadata.insert("©ART".to_string(), MetadataValue::Text(artist.clone()));
        }
        if let Some(ref album) = self.album {
            metadata.insert("©alb".to_string(), MetadataValue::Text(album.clone()));
        }
    }

    fn apply_xmp(&self, metadata: &mut Metadata) {
        if let Some(ref title) = self.title {
            metadata.insert("dc:title".to_string(), MetadataValue::Text(title.clone()));
        }
        if let Some(ref artist) = self.artist {
            metadata.insert(
                "dc:creator".to_string(),
                MetadataValue::Text(artist.clone()),
            );
        }
    }

    fn apply_exif(&self, metadata: &mut Metadata) {
        if let Some(ref artist) = self.artist {
            metadata.insert("Artist".to_string(), MetadataValue::Text(artist.clone()));
        }
        if let Some(ref copyright) = self.copyright {
            metadata.insert(
                "Copyright".to_string(),
                MetadataValue::Text(copyright.clone()),
            );
        }
    }

    fn apply_iptc(&self, metadata: &mut Metadata) {
        if let Some(ref title) = self.title {
            metadata.insert("ObjectName".to_string(), MetadataValue::Text(title.clone()));
        }
        if let Some(ref artist) = self.artist {
            metadata.insert("By-line".to_string(), MetadataValue::Text(artist.clone()));
        }
    }

    fn apply_quicktime(&self, metadata: &mut Metadata) {
        if let Some(ref title) = self.title {
            metadata.insert("©nam".to_string(), MetadataValue::Text(title.clone()));
        }
        if let Some(ref artist) = self.artist {
            metadata.insert("©ART".to_string(), MetadataValue::Text(artist.clone()));
        }
    }

    fn apply_matroska(&self, metadata: &mut Metadata) {
        if let Some(ref title) = self.title {
            metadata.insert("TITLE".to_string(), MetadataValue::Text(title.clone()));
        }
        if let Some(ref artist) = self.artist {
            metadata.insert("ARTIST".to_string(), MetadataValue::Text(artist.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetadataFormat;

    #[test]
    fn test_common_fields_new() {
        let fields = CommonFields::new();
        assert!(fields.title.is_none());
        assert!(fields.artist.is_none());
    }

    #[test]
    fn test_common_fields_id3v2_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::Id3v2);

        let mut fields = CommonFields::new();
        fields.title = Some("Test Title".to_string());
        fields.artist = Some("Test Artist".to_string());
        fields.album = Some("Test Album".to_string());
        fields.track_number = Some(5);
        fields.total_tracks = Some(10);

        fields.apply_to_metadata(&mut metadata);

        assert_eq!(
            metadata.get("TIT2").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            metadata.get("TPE1").and_then(|v| v.as_text()),
            Some("Test Artist")
        );
        assert_eq!(metadata.get("TRCK").and_then(|v| v.as_text()), Some("5/10"));

        let extracted = CommonFields::from_metadata(&metadata);
        assert_eq!(extracted.title, Some("Test Title".to_string()));
        assert_eq!(extracted.artist, Some("Test Artist".to_string()));
        assert_eq!(extracted.track_number, Some(5));
        assert_eq!(extracted.total_tracks, Some(10));
    }

    #[test]
    fn test_common_fields_vorbis_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::VorbisComments);

        let mut fields = CommonFields::new();
        fields.title = Some("Test Title".to_string());
        fields.artist = Some("Test Artist".to_string());
        fields.track_number = Some(5);

        fields.apply_to_metadata(&mut metadata);

        assert_eq!(
            metadata.get("TITLE").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            metadata.get("TRACKNUMBER").and_then(|v| v.as_text()),
            Some("5")
        );

        let extracted = CommonFields::from_metadata(&metadata);
        assert_eq!(extracted.title, Some("Test Title".to_string()));
        assert_eq!(extracted.track_number, Some(5));
    }
}
