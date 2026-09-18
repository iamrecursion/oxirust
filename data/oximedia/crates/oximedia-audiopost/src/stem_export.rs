//! Audio stem export: stem grouping, stem naming conventions, and format options.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Audio file format for stem export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportFormat {
    /// Broadcast WAV (BWF).
    BroadcastWav,
    /// Standard AIFF.
    Aiff,
    /// FLAC (lossless compressed).
    Flac,
    /// MP3 (lossy compressed).
    Mp3,
    /// AAC (lossy compressed).
    Aac,
}

impl ExportFormat {
    /// Return the file extension.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::BroadcastWav => "wav",
            Self::Aiff => "aiff",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
        }
    }

    /// Whether the format is lossless.
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::BroadcastWav | Self::Aiff | Self::Flac)
    }
}

/// Bit depth for PCM formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    /// 16-bit integer.
    Bit16,
    /// 24-bit integer.
    Bit24,
    /// 32-bit float.
    Bit32Float,
}

impl BitDepth {
    /// Return the bit depth as a number.
    pub fn bits(&self) -> u32 {
        match self {
            Self::Bit16 => 16,
            Self::Bit24 => 24,
            Self::Bit32Float => 32,
        }
    }
}

/// Standard stem group categories used in professional audio delivery.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StemGroup {
    /// Dialogue (all spoken word tracks).
    Dialogue,
    /// Music (score and source music).
    Music,
    /// Sound effects.
    Effects,
    /// Background ambience.
    Ambience,
    /// Foley tracks.
    Foley,
    /// M&E (Music and Effects composite).
    MusicAndEffects,
    /// Combined full mix.
    FullMix,
    /// Custom stem.
    Custom(String),
}

impl StemGroup {
    /// Return the standard abbreviation used in file naming.
    pub fn abbreviation(&self) -> String {
        match self {
            Self::Dialogue => "DX".to_string(),
            Self::Music => "MX".to_string(),
            Self::Effects => "FX".to_string(),
            Self::Ambience => "AMB".to_string(),
            Self::Foley => "FOL".to_string(),
            Self::MusicAndEffects => "ME".to_string(),
            Self::FullMix => "FULL".to_string(),
            Self::Custom(s) => s.to_uppercase(),
        }
    }

    /// Return a display name.
    pub fn display_name(&self) -> String {
        match self {
            Self::Dialogue => "Dialogue".to_string(),
            Self::Music => "Music".to_string(),
            Self::Effects => "Sound Effects".to_string(),
            Self::Ambience => "Ambience".to_string(),
            Self::Foley => "Foley".to_string(),
            Self::MusicAndEffects => "M&E".to_string(),
            Self::FullMix => "Full Mix".to_string(),
            Self::Custom(s) => s.clone(),
        }
    }
}

/// Naming convention for exported stem files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamingConvention {
    /// `{project}_{reel}_{stem}.{ext}` — standard broadcast format.
    BroadcastStandard,
    /// `{stem}_{reel}.{ext}` — abbreviated format.
    Abbreviated,
    /// `{project}_{stem}_{version}.{ext}` — versioned format.
    Versioned,
    /// Custom pattern with `{project}`, `{reel}`, `{stem}`, `{ext}` placeholders.
    Custom(String),
}

impl NamingConvention {
    /// Generate a filename from template fields.
    pub fn generate_filename(
        &self,
        project: &str,
        reel: &str,
        stem: &StemGroup,
        version: u32,
        format: &ExportFormat,
    ) -> String {
        let ext = format.extension();
        let stem_abbr = stem.abbreviation();
        match self {
            Self::BroadcastStandard => {
                format!("{project}_{reel}_{stem_abbr}.{ext}")
            }
            Self::Abbreviated => {
                format!("{stem_abbr}_{reel}.{ext}")
            }
            Self::Versioned => {
                format!("{project}_{stem_abbr}_v{version:03}.{ext}")
            }
            Self::Custom(pattern) => pattern
                .replace("{project}", project)
                .replace("{reel}", reel)
                .replace("{stem}", &stem_abbr)
                .replace("{ext}", ext),
        }
    }
}

/// Configuration for a single stem in an export job.
#[derive(Debug, Clone)]
pub struct StemConfig {
    /// Stem group.
    pub group: StemGroup,
    /// Track IDs included in this stem.
    pub track_ids: Vec<String>,
    /// Whether to include in this export.
    pub enabled: bool,
    /// Override file name (if None, naming convention is used).
    pub filename_override: Option<String>,
}

impl StemConfig {
    /// Create a new stem config.
    pub fn new(group: StemGroup, track_ids: Vec<String>) -> Self {
        Self {
            group,
            track_ids,
            enabled: true,
            filename_override: None,
        }
    }

    /// Disable this stem.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Set a filename override.
    pub fn set_filename_override(&mut self, name: &str) {
        self.filename_override = Some(name.to_string());
    }
}

/// Full stem export job configuration.
#[derive(Debug, Clone)]
pub struct StemExportJob {
    /// Project identifier.
    pub project: String,
    /// Reel or episode identifier.
    pub reel: String,
    /// Export format.
    pub format: ExportFormat,
    /// Bit depth (for PCM formats).
    pub bit_depth: BitDepth,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Naming convention.
    pub naming: NamingConvention,
    /// Version number for versioned exports.
    pub version: u32,
    /// Output directory.
    pub output_dir: String,
    /// Stem configurations.
    pub stems: Vec<StemConfig>,
}

impl StemExportJob {
    /// Create a new stem export job.
    pub fn new(project: &str, reel: &str, output_dir: &str) -> Self {
        Self {
            project: project.to_string(),
            reel: reel.to_string(),
            format: ExportFormat::BroadcastWav,
            bit_depth: BitDepth::Bit24,
            sample_rate: 48000,
            naming: NamingConvention::BroadcastStandard,
            version: 1,
            output_dir: output_dir.to_string(),
            stems: Vec::new(),
        }
    }

    /// Add a stem config.
    pub fn add_stem(&mut self, stem: StemConfig) {
        self.stems.push(stem);
    }

    /// Set the export format.
    pub fn set_format(&mut self, format: ExportFormat) {
        self.format = format;
    }

    /// Set the naming convention.
    pub fn set_naming(&mut self, naming: NamingConvention) {
        self.naming = naming;
    }

    /// Count enabled stems.
    pub fn enabled_stem_count(&self) -> usize {
        self.stems.iter().filter(|s| s.enabled).count()
    }

    /// Generate output file paths for all enabled stems.
    pub fn output_file_paths(&self) -> HashMap<String, String> {
        let mut paths = HashMap::new();
        for stem in self.stems.iter().filter(|s| s.enabled) {
            let filename = if let Some(ref ov) = stem.filename_override {
                ov.clone()
            } else {
                self.naming.generate_filename(
                    &self.project,
                    &self.reel,
                    &stem.group,
                    self.version,
                    &self.format,
                )
            };
            let path = format!("{}/{}", self.output_dir, filename);
            paths.insert(stem.group.abbreviation(), path);
        }
        paths
    }

    /// Validate the job configuration.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.project.is_empty() {
            errors.push("Project name must not be empty".to_string());
        }
        if self.reel.is_empty() {
            errors.push("Reel identifier must not be empty".to_string());
        }
        if self.output_dir.is_empty() {
            errors.push("Output directory must not be empty".to_string());
        }
        if self.stems.is_empty() {
            errors.push("At least one stem must be configured".to_string());
        }
        if self.sample_rate == 0 {
            errors.push("Sample rate must be greater than zero".to_string());
        }
        errors
    }
}

/// Build a standard broadcast stem export job with common stems.
pub fn build_standard_broadcast_job(project: &str, reel: &str, output_dir: &str) -> StemExportJob {
    let mut job = StemExportJob::new(project, reel, output_dir);
    job.add_stem(StemConfig::new(
        StemGroup::Dialogue,
        vec!["DX_1".to_string(), "DX_2".to_string()],
    ));
    job.add_stem(StemConfig::new(StemGroup::Music, vec!["MX_1".to_string()]));
    job.add_stem(StemConfig::new(
        StemGroup::Effects,
        vec!["FX_1".to_string(), "FX_2".to_string()],
    ));
    job.add_stem(StemConfig::new(
        StemGroup::MusicAndEffects,
        vec!["MX_1".to_string(), "FX_1".to_string(), "FX_2".to_string()],
    ));
    job.add_stem(StemConfig::new(
        StemGroup::FullMix,
        vec!["DX_1".to_string(), "MX_1".to_string(), "FX_1".to_string()],
    ));
    job
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::BroadcastWav.extension(), "wav");
        assert_eq!(ExportFormat::Aiff.extension(), "aiff");
        assert_eq!(ExportFormat::Flac.extension(), "flac");
        assert_eq!(ExportFormat::Mp3.extension(), "mp3");
        assert_eq!(ExportFormat::Aac.extension(), "aac");
    }

    #[test]
    fn test_export_format_lossless() {
        assert!(ExportFormat::BroadcastWav.is_lossless());
        assert!(ExportFormat::Aiff.is_lossless());
        assert!(ExportFormat::Flac.is_lossless());
        assert!(!ExportFormat::Mp3.is_lossless());
        assert!(!ExportFormat::Aac.is_lossless());
    }

    #[test]
    fn test_bit_depth_bits() {
        assert_eq!(BitDepth::Bit16.bits(), 16);
        assert_eq!(BitDepth::Bit24.bits(), 24);
        assert_eq!(BitDepth::Bit32Float.bits(), 32);
    }

    #[test]
    fn test_stem_group_abbreviation() {
        assert_eq!(StemGroup::Dialogue.abbreviation(), "DX");
        assert_eq!(StemGroup::Music.abbreviation(), "MX");
        assert_eq!(StemGroup::Effects.abbreviation(), "FX");
        assert_eq!(StemGroup::Ambience.abbreviation(), "AMB");
        assert_eq!(StemGroup::Foley.abbreviation(), "FOL");
        assert_eq!(StemGroup::MusicAndEffects.abbreviation(), "ME");
        assert_eq!(StemGroup::FullMix.abbreviation(), "FULL");
        assert_eq!(StemGroup::Custom("VFX".to_string()).abbreviation(), "VFX");
    }

    #[test]
    fn test_stem_group_display_name() {
        assert_eq!(StemGroup::Dialogue.display_name(), "Dialogue");
        assert_eq!(StemGroup::MusicAndEffects.display_name(), "M&E");
    }

    #[test]
    fn test_naming_broadcast_standard() {
        let naming = NamingConvention::BroadcastStandard;
        let name = naming.generate_filename(
            "MyFilm",
            "R1",
            &StemGroup::Dialogue,
            1,
            &ExportFormat::BroadcastWav,
        );
        assert_eq!(name, "MyFilm_R1_DX.wav");
    }

    #[test]
    fn test_naming_abbreviated() {
        let naming = NamingConvention::Abbreviated;
        let name =
            naming.generate_filename("MyFilm", "R1", &StemGroup::Music, 1, &ExportFormat::Aiff);
        assert_eq!(name, "MX_R1.aiff");
    }

    #[test]
    fn test_naming_versioned() {
        let naming = NamingConvention::Versioned;
        let name =
            naming.generate_filename("MyFilm", "R1", &StemGroup::Effects, 3, &ExportFormat::Flac);
        assert_eq!(name, "MyFilm_FX_v003.flac");
    }

    #[test]
    fn test_naming_custom() {
        let naming = NamingConvention::Custom("{stem}_{project}.{ext}".to_string());
        let name = naming.generate_filename(
            "Proj",
            "R1",
            &StemGroup::FullMix,
            1,
            &ExportFormat::BroadcastWav,
        );
        assert_eq!(name, "FULL_Proj.wav");
    }

    #[test]
    fn test_stem_config_new() {
        let cfg = StemConfig::new(StemGroup::Dialogue, vec!["DX_1".to_string()]);
        assert!(cfg.enabled);
        assert_eq!(cfg.track_ids.len(), 1);
    }

    #[test]
    fn test_stem_config_disable() {
        let mut cfg = StemConfig::new(StemGroup::Dialogue, vec![]);
        cfg.disable();
        assert!(!cfg.enabled);
    }

    #[test]
    fn test_stem_config_filename_override() {
        let mut cfg = StemConfig::new(StemGroup::Music, vec![]);
        cfg.set_filename_override("custom_music.wav");
        assert_eq!(cfg.filename_override.as_deref(), Some("custom_music.wav"));
    }

    #[test]
    fn test_stem_export_job_new_defaults() {
        let job = StemExportJob::new("Film", "R1", "/out");
        assert_eq!(job.format, ExportFormat::BroadcastWav);
        assert_eq!(job.bit_depth, BitDepth::Bit24);
        assert_eq!(job.sample_rate, 48000);
        assert_eq!(job.version, 1);
    }

    #[test]
    fn test_stem_export_job_enabled_stem_count() {
        let mut job = StemExportJob::new("Film", "R1", "/out");
        job.add_stem(StemConfig::new(StemGroup::Dialogue, vec![]));
        let mut s2 = StemConfig::new(StemGroup::Music, vec![]);
        s2.disable();
        job.add_stem(s2);
        assert_eq!(job.enabled_stem_count(), 1);
    }

    #[test]
    fn test_stem_export_job_output_file_paths() {
        let mut job = StemExportJob::new("Film", "R1", "/out");
        job.add_stem(StemConfig::new(StemGroup::Dialogue, vec![]));
        let paths = job.output_file_paths();
        assert!(paths.contains_key("DX"));
        assert!(paths["DX"].ends_with("Film_R1_DX.wav"));
    }

    #[test]
    fn test_stem_export_job_validate_valid() {
        let mut job = StemExportJob::new("Film", "R1", "/out");
        job.add_stem(StemConfig::new(StemGroup::FullMix, vec![]));
        assert!(job.validate().is_empty());
    }

    #[test]
    fn test_stem_export_job_validate_missing_project() {
        let mut job = StemExportJob::new("", "R1", "/out");
        job.add_stem(StemConfig::new(StemGroup::FullMix, vec![]));
        let errors = job.validate();
        assert!(errors.iter().any(|e| e.contains("Project")));
    }

    #[test]
    fn test_stem_export_job_validate_no_stems() {
        let job = StemExportJob::new("Film", "R1", "/out");
        let errors = job.validate();
        assert!(errors.iter().any(|e| e.contains("stem")));
    }

    #[test]
    fn test_build_standard_broadcast_job() {
        let job = build_standard_broadcast_job("Film", "R1", "/out");
        assert_eq!(job.stems.len(), 5);
        assert_eq!(job.enabled_stem_count(), 5);
    }

    #[test]
    fn test_output_path_with_filename_override() {
        let mut job = StemExportJob::new("Film", "R1", "/out");
        let mut cfg = StemConfig::new(StemGroup::Dialogue, vec![]);
        cfg.set_filename_override("my_dialogue_override.wav");
        job.add_stem(cfg);
        let paths = job.output_file_paths();
        assert_eq!(paths["DX"], "/out/my_dialogue_override.wav");
    }
}
