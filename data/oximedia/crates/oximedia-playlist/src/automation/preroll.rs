//! Pre-roll and post-roll management.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Pre-roll content played before main content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreRoll {
    /// Path to pre-roll media file.
    pub path: PathBuf,

    /// Duration of the pre-roll.
    pub duration: Duration,

    /// Whether the pre-roll is mandatory.
    pub mandatory: bool,

    /// Audio level adjustment (dB).
    pub audio_level: f64,
}

impl PreRoll {
    /// Creates a new pre-roll.
    #[must_use]
    pub fn new<P: Into<PathBuf>>(path: P, duration: Duration) -> Self {
        Self {
            path: path.into(),
            duration,
            mandatory: false,
            audio_level: 0.0,
        }
    }

    /// Makes this pre-roll mandatory.
    #[must_use]
    pub const fn as_mandatory(mut self) -> Self {
        self.mandatory = true;
        self
    }

    /// Sets the audio level.
    #[must_use]
    pub const fn with_audio_level(mut self, db: f64) -> Self {
        self.audio_level = db;
        self
    }
}

/// Post-roll content played after main content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostRoll {
    /// Path to post-roll media file.
    pub path: PathBuf,

    /// Duration of the post-roll.
    pub duration: Duration,

    /// Whether the post-roll is mandatory.
    pub mandatory: bool,

    /// Audio level adjustment (dB).
    pub audio_level: f64,
}

impl PostRoll {
    /// Creates a new post-roll.
    #[must_use]
    pub fn new<P: Into<PathBuf>>(path: P, duration: Duration) -> Self {
        Self {
            path: path.into(),
            duration,
            mandatory: false,
            audio_level: 0.0,
        }
    }

    /// Makes this post-roll mandatory.
    #[must_use]
    pub const fn as_mandatory(mut self) -> Self {
        self.mandatory = true;
        self
    }

    /// Sets the audio level.
    #[must_use]
    pub const fn with_audio_level(mut self, db: f64) -> Self {
        self.audio_level = db;
        self
    }
}

/// Manager for pre-roll and post-roll content.
#[derive(Debug, Clone, Default)]
pub struct RollManager {
    /// Pre-roll items.
    pub prerolls: Vec<PreRoll>,

    /// Post-roll items.
    pub postrolls: Vec<PostRoll>,

    /// Whether pre-rolls are enabled globally.
    pub preroll_enabled: bool,

    /// Whether post-rolls are enabled globally.
    pub postroll_enabled: bool,
}

impl RollManager {
    /// Creates a new roll manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a pre-roll.
    pub fn add_preroll(&mut self, preroll: PreRoll) {
        self.prerolls.push(preroll);
    }

    /// Adds a post-roll.
    pub fn add_postroll(&mut self, postroll: PostRoll) {
        self.postrolls.push(postroll);
    }

    /// Enables pre-rolls.
    pub fn enable_prerolls(&mut self) {
        self.preroll_enabled = true;
    }

    /// Disables pre-rolls.
    pub fn disable_prerolls(&mut self) {
        self.preroll_enabled = false;
    }

    /// Enables post-rolls.
    pub fn enable_postrolls(&mut self) {
        self.postroll_enabled = true;
    }

    /// Disables post-rolls.
    pub fn disable_postrolls(&mut self) {
        self.postroll_enabled = false;
    }

    /// Gets the total pre-roll duration.
    #[must_use]
    pub fn total_preroll_duration(&self) -> Duration {
        if self.preroll_enabled {
            self.prerolls.iter().map(|p| p.duration).sum()
        } else {
            Duration::ZERO
        }
    }

    /// Gets the total post-roll duration.
    #[must_use]
    pub fn total_postroll_duration(&self) -> Duration {
        if self.postroll_enabled {
            self.postrolls.iter().map(|p| p.duration).sum()
        } else {
            Duration::ZERO
        }
    }

    /// Clears all pre-rolls.
    pub fn clear_prerolls(&mut self) {
        self.prerolls.clear();
    }

    /// Clears all post-rolls.
    pub fn clear_postrolls(&mut self) {
        self.postrolls.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preroll() {
        let preroll = PreRoll::new("intro.mxf", Duration::from_secs(5))
            .as_mandatory()
            .with_audio_level(-3.0);

        assert!(preroll.mandatory);
        assert_eq!(preroll.audio_level, -3.0);
    }

    #[test]
    fn test_roll_manager() {
        let mut manager = RollManager::new();
        manager.add_preroll(PreRoll::new("intro.mxf", Duration::from_secs(5)));
        manager.add_postroll(PostRoll::new("outro.mxf", Duration::from_secs(3)));

        manager.enable_prerolls();
        manager.enable_postrolls();

        assert_eq!(manager.total_preroll_duration(), Duration::from_secs(5));
        assert_eq!(manager.total_postroll_duration(), Duration::from_secs(3));
    }
}
