//! Music licensing and cue sheet management.
//!
//! Provides types for tracking music cues in a production, generating cue sheets,
//! and formatting submissions for performing rights organizations (PROs).

#![allow(dead_code)]

/// Type of music cue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CueType {
    /// Main theme or opening/closing title music.
    Theme,
    /// Background underscore music.
    Background,
    /// Source music (music heard by characters, e.g. from a radio).
    Source,
    /// Music used in a montage sequence.
    Montage,
    /// Short musical punctuation (sting).
    Sting,
    /// Short musical bridge or transition.
    Bumper,
}

/// How the music is being used in the production.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicUsage {
    /// Feature film theatrical release.
    FeatureFilm,
    /// Television broadcast.
    Tv,
    /// Documentary.
    Documentary,
    /// Commercial / advertisement.
    Commercial,
    /// Online / streaming distribution.
    Online,
    /// Public performance (concert, venue, etc.).
    PublicPerformance,
}

/// A single music cue in a production.
#[derive(Debug, Clone)]
pub struct MusicCue {
    /// Song title.
    pub title: String,
    /// Composer name(s).
    pub composer: String,
    /// Music publisher name.
    pub publisher: String,
    /// International Standard Recording Code.
    pub isrc: Option<String>,
    /// Duration of the cue in seconds as used in the production.
    pub duration_secs: f64,
    /// How the cue is used (background, theme, etc.).
    pub cue_type: CueType,
    /// Distribution usage type.
    pub usage: MusicUsage,
}

impl MusicCue {
    /// Create a new music cue.
    #[must_use]
    pub fn new(
        title: impl Into<String>,
        composer: impl Into<String>,
        publisher: impl Into<String>,
        duration_secs: f64,
        cue_type: CueType,
        usage: MusicUsage,
    ) -> Self {
        Self {
            title: title.into(),
            composer: composer.into(),
            publisher: publisher.into(),
            isrc: None,
            duration_secs,
            cue_type,
            usage,
        }
    }

    /// Set ISRC code.
    #[must_use]
    pub fn with_isrc(mut self, isrc: impl Into<String>) -> Self {
        self.isrc = Some(isrc.into());
        self
    }

    /// Duration formatted as MM:SS.
    #[must_use]
    pub fn duration_formatted(&self) -> String {
        let total = self.duration_secs as u64;
        format!("{:02}:{:02}", total / 60, total % 60)
    }
}

/// A complete cue sheet for a production.
#[derive(Debug, Clone)]
pub struct CueSheet {
    /// Title of the production.
    pub title: String,
    /// Production/episode identifier.
    pub production_id: String,
    /// Music cues in order of appearance.
    pub cues: Vec<MusicCue>,
    /// Total duration of the production in seconds.
    pub total_duration_secs: f64,
}

impl CueSheet {
    /// Create a new empty cue sheet.
    #[must_use]
    pub fn new(
        title: impl Into<String>,
        production_id: impl Into<String>,
        total_duration_secs: f64,
    ) -> Self {
        Self {
            title: title.into(),
            production_id: production_id.into(),
            cues: Vec::new(),
            total_duration_secs,
        }
    }

    /// Add a music cue to the sheet.
    pub fn add_cue(&mut self, cue: MusicCue) {
        self.cues.push(cue);
    }

    /// Total duration of all music cues in seconds.
    #[must_use]
    pub fn total_music_duration(&self) -> f64 {
        self.cues.iter().map(|c| c.duration_secs).sum()
    }

    /// Number of cues in the sheet.
    #[must_use]
    pub fn cue_count(&self) -> usize {
        self.cues.len()
    }

    /// Music as percentage of total production duration.
    #[must_use]
    pub fn music_percentage(&self) -> f64 {
        if self.total_duration_secs <= 0.0 {
            return 0.0;
        }
        (self.total_music_duration() / self.total_duration_secs * 100.0).min(100.0)
    }

    /// Generate CSV format for the cue sheet.
    ///
    /// Columns: Cue #, Title, Composer, Publisher, ISRC, Duration, Type, Usage
    #[must_use]
    pub fn to_csv_format(&self) -> String {
        let mut lines = Vec::new();

        // Header
        lines.push(format!(
            "Production: {} ({})",
            self.title, self.production_id
        ));
        lines.push("Cue #,Title,Composer,Publisher,ISRC,Duration,Type,Usage".to_string());

        for (i, cue) in self.cues.iter().enumerate() {
            lines.push(format!(
                "{},{},{},{},{},{},{:?},{:?}",
                i + 1,
                csv_escape(&cue.title),
                csv_escape(&cue.composer),
                csv_escape(&cue.publisher),
                cue.isrc.as_deref().unwrap_or(""),
                cue.duration_formatted(),
                cue.cue_type,
                cue.usage,
            ));
        }

        lines.join("\n")
    }
}

/// Escape a CSV field by quoting it if it contains commas or quotes.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// A Performing Rights Organization (PRO).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformingRightsOrg {
    /// American Society of Composers, Authors and Publishers (US).
    ASCAP,
    /// Broadcast Music, Inc. (US).
    BMI,
    /// Society of European Stage Authors and Composers (US).
    SESAC,
    /// Society of Composers, Authors and Music Publishers of Canada.
    SOCAN,
    /// Performing Right Society (UK).
    PRS,
    /// Australasian Performing Right Association (Australia/NZ).
    APRA,
}

impl PerformingRightsOrg {
    /// Get the expected submission format for this PRO.
    #[must_use]
    pub fn submission_format(&self) -> &str {
        match self {
            Self::ASCAP => "ASCAP Online Clearance System (CSV)",
            Self::BMI => "BMI Royalty Connect (XML)",
            Self::SESAC => "SESAC Portal (Excel/CSV)",
            Self::SOCAN => "SOCAN Cue Sheet Portal (CSV)",
            Self::PRS => "PRS Online (Excel)",
            Self::APRA => "APRA AMCOS Online (CSV)",
        }
    }

    /// Get the PRO's full name.
    #[must_use]
    pub fn full_name(&self) -> &str {
        match self {
            Self::ASCAP => "American Society of Composers, Authors and Publishers",
            Self::BMI => "Broadcast Music, Inc.",
            Self::SESAC => "Society of European Stage Authors and Composers",
            Self::SOCAN => "Society of Composers, Authors and Music Publishers of Canada",
            Self::PRS => "Performing Right Society",
            Self::APRA => "Australasian Performing Right Association",
        }
    }
}

/// Handles formatting cue sheets for PRO submission.
pub struct CueSheetSubmission;

impl CueSheetSubmission {
    /// Format a cue sheet for submission to a specific PRO.
    #[must_use]
    pub fn format_for_pro(sheet: &CueSheet, org: PerformingRightsOrg) -> String {
        let header = format!(
            "=== {} Cue Sheet Submission ===\nProduction: {}\nID: {}\nFormat: {}\nTotal Music: {:.1}s / {:.1}s ({:.1}%)\n\n",
            org.full_name(),
            sheet.title,
            sheet.production_id,
            org.submission_format(),
            sheet.total_music_duration(),
            sheet.total_duration_secs,
            sheet.music_percentage(),
        );

        let cue_lines: Vec<String> = sheet
            .cues
            .iter()
            .enumerate()
            .map(|(i, cue)| {
                format!(
                    "{}. \"{}\" by {} | Published by {} | ISRC: {} | Duration: {} | Type: {:?} | Usage: {:?}",
                    i + 1,
                    cue.title,
                    cue.composer,
                    cue.publisher,
                    cue.isrc.as_deref().unwrap_or("N/A"),
                    cue.duration_formatted(),
                    cue.cue_type,
                    cue.usage,
                )
            })
            .collect();

        format!("{}{}", header, cue_lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cue_sheet() -> CueSheet {
        let mut sheet = CueSheet::new("The Great Film", "PROD-2024-001", 5400.0); // 90 min
        sheet.add_cue(
            MusicCue::new(
                "Opening Theme",
                "John Williams",
                "Hal Leonard",
                120.0,
                CueType::Theme,
                MusicUsage::FeatureFilm,
            )
            .with_isrc("USRC12345678"),
        );
        sheet.add_cue(MusicCue::new(
            "Chase Scene",
            "Hans Zimmer",
            "Universal Music",
            90.0,
            CueType::Background,
            MusicUsage::FeatureFilm,
        ));
        sheet.add_cue(MusicCue::new(
            "Radio Song",
            "Various",
            "Sony Music",
            30.0,
            CueType::Source,
            MusicUsage::Tv,
        ));
        sheet
    }

    #[test]
    fn test_cue_sheet_creation() {
        let sheet = make_cue_sheet();
        assert_eq!(sheet.title, "The Great Film");
        assert_eq!(sheet.cue_count(), 3);
    }

    #[test]
    fn test_total_music_duration() {
        let sheet = make_cue_sheet();
        assert_eq!(sheet.total_music_duration(), 240.0); // 120 + 90 + 30
    }

    #[test]
    fn test_music_percentage() {
        let sheet = make_cue_sheet(); // 240s music / 5400s total
        let pct = sheet.music_percentage();
        assert!((pct - 4.44).abs() < 0.01);
    }

    #[test]
    fn test_duration_formatted() {
        let cue = MusicCue::new(
            "Title",
            "Composer",
            "Publisher",
            90.0,
            CueType::Sting,
            MusicUsage::Commercial,
        );
        assert_eq!(cue.duration_formatted(), "01:30");
    }

    #[test]
    fn test_duration_formatted_hours() {
        let cue = MusicCue::new("T", "C", "P", 3725.0, CueType::Background, MusicUsage::Tv);
        // 62:05 (3725 / 60 = 62 min, 5 sec)
        assert_eq!(cue.duration_formatted(), "62:05");
    }

    #[test]
    fn test_to_csv_format_header() {
        let sheet = make_cue_sheet();
        let csv = sheet.to_csv_format();
        assert!(csv.contains("Production: The Great Film"));
        assert!(csv.contains("Cue #,Title,Composer"));
    }

    #[test]
    fn test_to_csv_format_cue_rows() {
        let sheet = make_cue_sheet();
        let csv = sheet.to_csv_format();
        assert!(csv.contains("John Williams"));
        assert!(csv.contains("USRC12345678"));
        assert!(csv.contains("01:30"));
    }

    #[test]
    fn test_pro_submission_format() {
        let org = PerformingRightsOrg::BMI;
        let fmt = org.submission_format();
        assert!(fmt.contains("BMI"));
    }

    #[test]
    fn test_ascap_full_name() {
        let org = PerformingRightsOrg::ASCAP;
        assert!(org.full_name().contains("American Society"));
    }

    #[test]
    fn test_format_for_pro_ascap() {
        let sheet = make_cue_sheet();
        let output = CueSheetSubmission::format_for_pro(&sheet, PerformingRightsOrg::ASCAP);
        assert!(output.contains("American Society of Composers"));
        assert!(output.contains("Opening Theme"));
        assert!(output.contains("The Great Film"));
    }

    #[test]
    fn test_format_for_pro_bmi() {
        let sheet = make_cue_sheet();
        let output = CueSheetSubmission::format_for_pro(&sheet, PerformingRightsOrg::BMI);
        assert!(output.contains("Broadcast Music"));
        assert!(output.contains("Chase Scene"));
    }

    #[test]
    fn test_music_cue_with_isrc() {
        let cue = MusicCue::new(
            "Track",
            "Artist",
            "Publisher",
            60.0,
            CueType::Bumper,
            MusicUsage::Online,
        )
        .with_isrc("GB-ACE-19-12345");
        assert_eq!(cue.isrc, Some("GB-ACE-19-12345".to_string()));
    }

    #[test]
    fn test_empty_cue_sheet() {
        let sheet = CueSheet::new("Empty", "ID-001", 3600.0);
        assert_eq!(sheet.total_music_duration(), 0.0);
        assert_eq!(sheet.music_percentage(), 0.0);
        assert_eq!(sheet.cue_count(), 0);
    }

    #[test]
    fn test_csv_escape_with_comma() {
        let escaped = csv_escape("Smith, John");
        assert_eq!(escaped, "\"Smith, John\"");
    }

    #[test]
    fn test_all_pro_variants() {
        let pros = vec![
            PerformingRightsOrg::ASCAP,
            PerformingRightsOrg::BMI,
            PerformingRightsOrg::SESAC,
            PerformingRightsOrg::SOCAN,
            PerformingRightsOrg::PRS,
            PerformingRightsOrg::APRA,
        ];
        for org in pros {
            assert!(!org.submission_format().is_empty());
            assert!(!org.full_name().is_empty());
        }
    }
}
