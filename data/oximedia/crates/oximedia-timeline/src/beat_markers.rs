//! Beat and music markers on a timeline.
//!
//! Provides beat grid alignment, tempo mapping, and snap-to-beat
//! functionality for music-synchronized timeline editing.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single beat marker on the timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct BeatMarker {
    /// Position on the timeline in frames.
    pub frame: u64,
    /// Beat number within the current bar (1-based).
    pub beat: u32,
    /// Bar number (1-based).
    pub bar: u32,
    /// Optional label (e.g., "Chorus", "Drop").
    pub label: Option<String>,
}

impl BeatMarker {
    /// Creates a new beat marker.
    #[must_use]
    pub fn new(frame: u64, beat: u32, bar: u32) -> Self {
        Self {
            frame,
            beat,
            bar,
            label: None,
        }
    }

    /// Adds a label to this marker.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns a string representation like "3|2" (bar 3, beat 2).
    #[must_use]
    pub fn position_string(&self) -> String {
        format!("{}|{}", self.bar, self.beat)
    }
}

/// A tempo change event in a tempo map.
#[derive(Debug, Clone, PartialEq)]
pub struct TempoEvent {
    /// Frame where this tempo starts.
    pub frame: u64,
    /// Beats per minute at this point.
    pub bpm: f64,
    /// Beats per bar (time signature numerator).
    pub beats_per_bar: u32,
    /// Note value of one beat (4 = quarter note).
    pub beat_unit: u32,
}

impl TempoEvent {
    /// Creates a new tempo event.
    #[must_use]
    pub fn new(frame: u64, bpm: f64, beats_per_bar: u32, beat_unit: u32) -> Self {
        Self {
            frame,
            bpm,
            beats_per_bar,
            beat_unit,
        }
    }

    /// Returns seconds per beat at this tempo.
    #[must_use]
    pub fn seconds_per_beat(&self) -> f64 {
        60.0 / self.bpm
    }

    /// Returns seconds per bar at this tempo.
    #[must_use]
    pub fn seconds_per_bar(&self) -> f64 {
        self.seconds_per_beat() * self.beats_per_bar as f64
    }
}

/// A tempo map holding all tempo and time-signature changes for a timeline.
#[derive(Debug, Clone)]
pub struct TempoMap {
    /// All tempo events, sorted by frame.
    events: Vec<TempoEvent>,
    /// Timeline frame rate (frames per second).
    fps: f64,
}

impl TempoMap {
    /// Creates a new tempo map with a constant tempo.
    #[must_use]
    pub fn new(fps: f64, initial_bpm: f64) -> Self {
        Self {
            events: vec![TempoEvent::new(0, initial_bpm, 4, 4)],
            fps,
        }
    }

    /// Adds a tempo change event.
    pub fn add_event(&mut self, event: TempoEvent) {
        self.events.push(event);
        self.events.sort_by_key(|e| e.frame);
    }

    /// Returns the active tempo event at the given frame.
    #[must_use]
    pub fn event_at(&self, frame: u64) -> &TempoEvent {
        self.events
            .iter()
            .rev()
            .find(|e| e.frame <= frame)
            .unwrap_or(&self.events[0])
    }

    /// Computes the frame position of a given bar and beat.
    #[must_use]
    pub fn frame_of(&self, bar: u32, beat: u32) -> u64 {
        // Simplified: assume constant tempo from start
        let event = &self.events[0];
        let beats_from_start = (bar as u64 - 1) * event.beats_per_bar as u64 + (beat as u64 - 1);
        let seconds = beats_from_start as f64 * event.seconds_per_beat();
        (seconds * self.fps).round() as u64
    }

    /// Generates beat markers for a range of frames.
    #[must_use]
    pub fn generate_markers(&self, start_frame: u64, end_frame: u64) -> Vec<BeatMarker> {
        let mut markers = Vec::new();
        let event = self.event_at(start_frame);
        let frames_per_beat = (event.seconds_per_beat() * self.fps).round() as u64;
        if frames_per_beat == 0 {
            return markers;
        }
        let mut frame = start_frame;
        let mut total_beats = 0u64;
        while frame <= end_frame {
            let bar = (total_beats / event.beats_per_bar as u64) as u32 + 1;
            let beat = (total_beats % event.beats_per_bar as u64) as u32 + 1;
            markers.push(BeatMarker::new(frame, beat, bar));
            frame += frames_per_beat;
            total_beats += 1;
        }
        markers
    }

    /// Snaps a frame to the nearest beat.
    #[must_use]
    pub fn snap_to_beat(&self, frame: u64) -> u64 {
        let event = self.event_at(frame);
        let frames_per_beat = (event.seconds_per_beat() * self.fps).round() as u64;
        if frames_per_beat == 0 {
            return frame;
        }
        let remainder = frame % frames_per_beat;
        if remainder <= frames_per_beat / 2 {
            frame - remainder
        } else {
            frame + (frames_per_beat - remainder)
        }
    }

    /// Snaps a frame to the nearest bar.
    #[must_use]
    pub fn snap_to_bar(&self, frame: u64) -> u64 {
        let event = self.event_at(frame);
        let frames_per_bar = (event.seconds_per_bar() * self.fps).round() as u64;
        if frames_per_bar == 0 {
            return frame;
        }
        let remainder = frame % frames_per_bar;
        if remainder <= frames_per_bar / 2 {
            frame - remainder
        } else {
            frame + (frames_per_bar - remainder)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beat_marker_position_string() {
        let m = BeatMarker::new(100, 2, 3);
        assert_eq!(m.position_string(), "3|2");
    }

    #[test]
    fn test_beat_marker_with_label() {
        let m = BeatMarker::new(100, 1, 1).with_label("Intro");
        assert_eq!(m.label.as_deref(), Some("Intro"));
    }

    #[test]
    fn test_tempo_event_seconds_per_beat_120bpm() {
        let e = TempoEvent::new(0, 120.0, 4, 4);
        assert!((e.seconds_per_beat() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_tempo_event_seconds_per_bar_120bpm_4_4() {
        let e = TempoEvent::new(0, 120.0, 4, 4);
        assert!((e.seconds_per_bar() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_tempo_map_snap_to_beat_on_beat() {
        // 120 bpm, 30fps => frames_per_beat = 15
        let map = TempoMap::new(30.0, 120.0);
        assert_eq!(map.snap_to_beat(30), 30);
    }

    #[test]
    fn test_tempo_map_snap_to_beat_before_mid() {
        let map = TempoMap::new(30.0, 120.0);
        // frames_per_beat = 15; frame 32 is 2 frames past beat 2 (30) => snaps to 30
        assert_eq!(map.snap_to_beat(32), 30);
    }

    #[test]
    fn test_tempo_map_snap_to_beat_after_mid() {
        let map = TempoMap::new(30.0, 120.0);
        // frames_per_beat = 15; frame 38 is 8 frames past beat 2 (30) => snaps to 45
        assert_eq!(map.snap_to_beat(38), 45);
    }

    #[test]
    fn test_tempo_map_snap_to_bar() {
        // 120 bpm, 4/4, 30fps => frames_per_bar = 60
        let map = TempoMap::new(30.0, 120.0);
        assert_eq!(map.snap_to_bar(25), 0);
        assert_eq!(map.snap_to_bar(35), 60);
    }

    #[test]
    fn test_generate_markers_count() {
        // 120 bpm, 30fps => beat every 15 frames, 5 beats in frames 0..=60
        let map = TempoMap::new(30.0, 120.0);
        let markers = map.generate_markers(0, 60);
        assert_eq!(markers.len(), 5); // frames 0, 15, 30, 45, 60
    }

    #[test]
    fn test_generate_markers_beat_numbering() {
        let map = TempoMap::new(30.0, 120.0);
        let markers = map.generate_markers(0, 60);
        assert_eq!(markers[0].bar, 1);
        assert_eq!(markers[0].beat, 1);
        assert_eq!(markers[4].bar, 2);
        assert_eq!(markers[4].beat, 1);
    }

    #[test]
    fn test_frame_of_bar1_beat1() {
        let map = TempoMap::new(30.0, 120.0);
        assert_eq!(map.frame_of(1, 1), 0);
    }

    #[test]
    fn test_frame_of_bar2_beat1() {
        // 120 bpm, 30fps => 15 frames/beat, 4 beats/bar => 60 frames/bar
        let map = TempoMap::new(30.0, 120.0);
        assert_eq!(map.frame_of(2, 1), 60);
    }

    #[test]
    fn test_tempo_map_add_event() {
        let mut map = TempoMap::new(30.0, 120.0);
        map.add_event(TempoEvent::new(300, 140.0, 4, 4));
        assert_eq!(map.events.len(), 2);
        let e = map.event_at(300);
        assert!((e.bpm - 140.0).abs() < 1e-9);
    }

    #[test]
    fn test_event_at_before_first() {
        let map = TempoMap::new(30.0, 120.0);
        let e = map.event_at(0);
        assert!((e.bpm - 120.0).abs() < 1e-9);
    }
}
