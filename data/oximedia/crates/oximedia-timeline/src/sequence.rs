//! Timeline sequence management.
//!
//! A `Sequence` represents a top-level editing project with its own frame
//! rate, resolution, audio sample rate, and a set of track identifiers.

static SEQUENCE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_sequence_id() -> u64 {
    SEQUENCE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Advanced settings that can be applied to a sequence.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SequenceSettings {
    /// Use SMPTE drop-frame timecode.
    pub use_drop_frame: bool,
    /// Colour bit depth (e.g. 8, 10, 12, 16).
    pub bit_depth: u8,
    /// Colour space identifier (e.g. "Rec. 709", "DCI-P3").
    pub color_space: String,
    /// Whether HDR mode is active.
    pub hdr_mode: bool,
}

impl Default for SequenceSettings {
    fn default() -> Self {
        Self {
            use_drop_frame: false,
            bit_depth: 8,
            color_space: "Rec. 709".to_string(),
            hdr_mode: false,
        }
    }
}

/// A top-level editing sequence.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Sequence {
    /// Unique sequence identifier.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Frames per second.
    pub frame_rate: f64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Ordered list of track identifiers belonging to this sequence.
    pub tracks: Vec<u64>,
    /// Advanced sequence settings.
    settings: SequenceSettings,
    /// Total duration of the sequence in frames (derived from clips on tracks).
    duration_frames: u64,
}

impl Sequence {
    /// Create a new sequence with default settings.
    #[must_use]
    pub fn new(name: &str, fps: f64, width: u32, height: u32) -> Self {
        Self {
            id: next_sequence_id(),
            name: name.to_string(),
            frame_rate: fps.max(f64::EPSILON),
            width,
            height,
            sample_rate: 48_000,
            tracks: Vec::new(),
            settings: SequenceSettings::default(),
            duration_frames: 0,
        }
    }

    /// Return the current duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.duration_frames
    }

    /// Set the sequence duration in frames.
    pub fn set_duration_frames(&mut self, frames: u64) {
        self.duration_frames = frames;
    }

    /// Convert a frame number to a presentation timestamp in milliseconds.
    #[must_use]
    pub fn frame_to_ms(&self, frame: u64) -> f64 {
        if self.frame_rate <= 0.0 {
            return 0.0;
        }
        (frame as f64 / self.frame_rate) * 1000.0
    }

    /// Convert a timestamp in milliseconds to the nearest frame number.
    #[must_use]
    pub fn ms_to_frame(&self, ms: f64) -> u64 {
        if self.frame_rate <= 0.0 || ms < 0.0 {
            return 0;
        }
        ((ms / 1000.0) * self.frame_rate).round() as u64
    }

    /// Add a track identifier to this sequence.
    pub fn add_track(&mut self, track_id: u64) {
        if !self.tracks.contains(&track_id) {
            self.tracks.push(track_id);
        }
    }

    /// Remove a track identifier from this sequence.
    ///
    /// Returns `true` if the track was present and removed.
    pub fn remove_track(&mut self, track_id: u64) -> bool {
        if let Some(pos) = self.tracks.iter().position(|&id| id == track_id) {
            self.tracks.remove(pos);
            true
        } else {
            false
        }
    }

    /// Apply advanced settings to this sequence.
    pub fn apply_settings(&mut self, settings: SequenceSettings) {
        self.settings = settings;
    }

    /// Return a reference to the current settings.
    #[must_use]
    pub fn settings(&self) -> &SequenceSettings {
        &self.settings
    }

    /// Return the aspect ratio as `(width, height)`.
    #[must_use]
    pub fn aspect_ratio(&self) -> (u32, u32) {
        let d = gcd(self.width, self.height);
        if d == 0 {
            return (self.width, self.height);
        }
        (self.width / d, self.height / d)
    }

    /// Return the number of tracks in the sequence.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Return the total duration of the sequence in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        if self.frame_rate <= 0.0 {
            return 0.0;
        }
        self.duration_frames as f64 / self.frame_rate
    }
}

/// Compute the greatest common divisor of two unsigned integers.
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq() -> Sequence {
        Sequence::new("Test Seq", 24.0, 1920, 1080)
    }

    #[test]
    fn test_sequence_creation() {
        let s = seq();
        assert_eq!(s.name, "Test Seq");
        assert!((s.frame_rate - 24.0).abs() < f64::EPSILON);
        assert_eq!(s.width, 1920);
        assert_eq!(s.height, 1080);
    }

    #[test]
    fn test_sequence_unique_ids() {
        let s1 = Sequence::new("A", 24.0, 1920, 1080);
        let s2 = Sequence::new("B", 24.0, 1920, 1080);
        assert_ne!(s1.id, s2.id);
    }

    #[test]
    fn test_frame_to_ms_24fps() {
        let s = seq();
        let ms = s.frame_to_ms(24);
        assert!((ms - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_ms_to_frame_24fps() {
        let s = seq();
        let frame = s.ms_to_frame(1000.0);
        assert_eq!(frame, 24);
    }

    #[test]
    fn test_ms_to_frame_negative() {
        let s = seq();
        assert_eq!(s.ms_to_frame(-100.0), 0);
    }

    #[test]
    fn test_add_track() {
        let mut s = seq();
        s.add_track(10);
        s.add_track(20);
        assert_eq!(s.track_count(), 2);
    }

    #[test]
    fn test_add_track_duplicate_ignored() {
        let mut s = seq();
        s.add_track(10);
        s.add_track(10);
        assert_eq!(s.track_count(), 1);
    }

    #[test]
    fn test_remove_track_existing() {
        let mut s = seq();
        s.add_track(10);
        assert!(s.remove_track(10));
        assert_eq!(s.track_count(), 0);
    }

    #[test]
    fn test_remove_track_nonexistent() {
        let mut s = seq();
        assert!(!s.remove_track(99));
    }

    #[test]
    fn test_apply_settings() {
        let mut s = seq();
        let settings = SequenceSettings {
            use_drop_frame: true,
            bit_depth: 10,
            color_space: "DCI-P3".to_string(),
            hdr_mode: true,
        };
        s.apply_settings(settings);
        assert!(s.settings().use_drop_frame);
        assert_eq!(s.settings().bit_depth, 10);
        assert_eq!(s.settings().color_space, "DCI-P3");
        assert!(s.settings().hdr_mode);
    }

    #[test]
    fn test_aspect_ratio_1080p() {
        let s = seq();
        assert_eq!(s.aspect_ratio(), (16, 9));
    }

    #[test]
    fn test_set_duration_frames() {
        let mut s = seq();
        s.set_duration_frames(240);
        assert_eq!(s.duration_frames(), 240);
        assert!((s.duration_seconds() - 10.0).abs() < 0.001);
    }
}
