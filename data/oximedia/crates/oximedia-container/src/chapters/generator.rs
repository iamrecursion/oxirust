//! Chapter generation from timecodes.
//!
//! Automatically generates chapters from various sources.

#![forbid(unsafe_code)]

use super::{matroska::MatroskaChapter, mp4::Mp4Chapter};

/// Configuration for chapter generation.
#[derive(Debug, Clone)]
pub struct ChapterGeneratorConfig {
    /// Interval between chapters in seconds.
    pub interval_secs: f64,
    /// Prefix for chapter titles.
    pub title_prefix: String,
    /// Whether to number chapters.
    pub number_chapters: bool,
    /// Language code for chapter titles.
    pub language: String,
}

impl Default for ChapterGeneratorConfig {
    fn default() -> Self {
        Self {
            interval_secs: 300.0, // 5 minutes
            title_prefix: "Chapter".into(),
            number_chapters: true,
            language: "eng".into(),
        }
    }
}

impl ChapterGeneratorConfig {
    /// Creates a new configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the interval.
    #[must_use]
    pub fn with_interval(mut self, interval_secs: f64) -> Self {
        self.interval_secs = interval_secs;
        self
    }

    /// Sets the title prefix.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.title_prefix = prefix.into();
        self
    }
}

/// Generator for creating chapters.
pub struct ChapterGenerator {
    config: ChapterGeneratorConfig,
}

impl ChapterGenerator {
    /// Creates a new chapter generator.
    #[must_use]
    pub fn new(config: ChapterGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generates chapters at regular intervals.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn generate_interval_chapters(&self, duration_secs: f64) -> Vec<MatroskaChapter> {
        let mut chapters = Vec::new();
        let mut time = 0.0;
        let mut chapter_num = 1;

        while time < duration_secs {
            let title = if self.config.number_chapters {
                format!("{} {}", self.config.title_prefix, chapter_num)
            } else {
                self.config.title_prefix.clone()
            };

            let chapter = MatroskaChapter::new(chapter_num, (time * 1_000_000_000.0) as u64)
                .with_display(&self.config.language, title);

            chapters.push(chapter);

            time += self.config.interval_secs;
            chapter_num += 1;
        }

        chapters
    }

    /// Generates chapters from explicit timestamps.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn generate_from_timestamps(
        &self,
        timestamps_secs: &[f64],
        titles: Option<&[String]>,
    ) -> Vec<MatroskaChapter> {
        timestamps_secs
            .iter()
            .enumerate()
            .map(|(i, &time)| {
                let title = titles
                    .and_then(|t| t.get(i))
                    .cloned()
                    .unwrap_or_else(|| format!("{} {}", self.config.title_prefix, i + 1));

                MatroskaChapter::new((i + 1) as u64, (time * 1_000_000_000.0) as u64)
                    .with_display(&self.config.language, title)
            })
            .collect()
    }

    /// Generates MP4-style chapters.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn generate_mp4_chapters(&self, duration_secs: f64) -> Vec<Mp4Chapter> {
        let mut chapters = Vec::new();
        let mut time = 0.0;
        let mut chapter_num = 1;

        while time < duration_secs {
            let title = if self.config.number_chapters {
                format!("{} {}", self.config.title_prefix, chapter_num)
            } else {
                self.config.title_prefix.clone()
            };

            chapters.push(Mp4Chapter::new((time * 1000.0) as u64, title));

            time += self.config.interval_secs;
            chapter_num += 1;
        }

        chapters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chapter_generator_config() {
        let config = ChapterGeneratorConfig::new()
            .with_interval(60.0)
            .with_prefix("Part");

        assert_eq!(config.interval_secs, 60.0);
        assert_eq!(config.title_prefix, "Part");
    }

    #[test]
    fn test_generate_interval_chapters() {
        let config = ChapterGeneratorConfig::new().with_interval(10.0);
        let generator = ChapterGenerator::new(config);

        let chapters = generator.generate_interval_chapters(30.0);
        assert_eq!(chapters.len(), 3);
        assert_eq!(chapters[0].start_time_ns, 0);
        assert_eq!(chapters[1].start_time_ns, 10_000_000_000);
    }

    #[test]
    fn test_generate_from_timestamps() {
        let config = ChapterGeneratorConfig::new();
        let generator = ChapterGenerator::new(config);

        let timestamps = vec![0.0, 5.0, 10.0];
        let titles = vec!["Intro".to_string(), "Middle".to_string(), "End".to_string()];

        let chapters = generator.generate_from_timestamps(&timestamps, Some(&titles));
        assert_eq!(chapters.len(), 3);
        assert_eq!(chapters[0].default_title(), Some("Intro"));
    }

    #[test]
    fn test_generate_mp4_chapters() {
        let config = ChapterGeneratorConfig::new().with_interval(5.0);
        let generator = ChapterGenerator::new(config);

        let chapters = generator.generate_mp4_chapters(15.0);
        assert_eq!(chapters.len(), 3);
        assert_eq!(chapters[0].start_time_ms, 0);
        assert_eq!(chapters[1].start_time_ms, 5000);
    }
}
