//! General event detection.

use crate::common::Confidence;
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Type of video event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// Scene change.
    SceneChange,
    /// Camera motion (pan, tilt, zoom).
    CameraMotion,
    /// Flash or sudden brightness change.
    Flash,
    /// Applause or crowd reaction.
    Applause,
    /// Explosion or impact.
    Explosion,
    /// Unknown event.
    Unknown,
}

impl EventType {
    /// Get human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::SceneChange => "Scene Change",
            Self::CameraMotion => "Camera Motion",
            Self::Flash => "Flash",
            Self::Applause => "Applause",
            Self::Explosion => "Explosion",
            Self::Unknown => "Unknown",
        }
    }
}

/// Detected video event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoEvent {
    /// Event type.
    pub event_type: EventType,
    /// Frame number.
    pub frame_number: usize,
    /// Detection confidence.
    pub confidence: Confidence,
    /// Event duration (frames).
    pub duration: usize,
}

/// Event detector.
pub struct EventDetector {
    min_frames: usize,
}

impl EventDetector {
    /// Create a new event detector.
    #[must_use]
    pub fn new() -> Self {
        Self { min_frames: 3 }
    }

    /// Detect events in frame sequence.
    ///
    /// # Errors
    ///
    /// Returns error if detection fails.
    pub fn detect(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<VideoEvent>> {
        if frames.len() < self.min_frames {
            return Err(SceneError::InsufficientData(format!(
                "Need at least {} frames",
                self.min_frames
            )));
        }

        let mut events = Vec::new();

        // Detect scene changes
        for i in 1..frames.len() {
            let diff = self.frame_difference(frames[i - 1], frames[i], width, height);

            if diff > 0.5 {
                events.push(VideoEvent {
                    event_type: EventType::SceneChange,
                    frame_number: i,
                    confidence: Confidence::new(diff.min(1.0)),
                    duration: 1,
                });
            }
        }

        // Detect camera motion
        let camera_events = self.detect_camera_motion(frames, width, height)?;
        events.extend(camera_events);

        // Detect flashes
        let flash_events = self.detect_flashes(frames, width, height)?;
        events.extend(flash_events);

        Ok(events)
    }

    fn frame_difference(&self, frame1: &[u8], frame2: &[u8], _width: usize, _height: usize) -> f32 {
        let mut diff_sum = 0u64;
        let mut count = 0;

        for i in (0..frame1.len().min(frame2.len())).step_by(3) {
            for c in 0..3 {
                diff_sum += (frame1[i + c] as i32 - frame2[i + c] as i32).unsigned_abs() as u64;
            }
            count += 3;
        }

        if count > 0 {
            (diff_sum as f32 / count as f32 / 255.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn detect_camera_motion(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<VideoEvent>> {
        let mut events = Vec::new();

        for i in 1..frames.len() {
            let motion = self.estimate_global_motion(frames[i - 1], frames[i], width, height);

            if motion > 0.3 {
                events.push(VideoEvent {
                    event_type: EventType::CameraMotion,
                    frame_number: i,
                    confidence: Confidence::new(motion.min(1.0)),
                    duration: 1,
                });
            }
        }

        Ok(events)
    }

    fn estimate_global_motion(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        height: usize,
    ) -> f32 {
        let block_size = 32;
        let mut motion_sum = 0.0;
        let mut count = 0;

        for y in (0..height - block_size).step_by(block_size) {
            for x in (0..width - block_size).step_by(block_size) {
                let mut min_diff = f32::MAX;

                // Simple block matching
                for dy in -8..=8 {
                    for dx in -8..=8 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0
                            && ny >= 0
                            && (nx as usize + block_size) < width
                            && (ny as usize + block_size) < height
                        {
                            let diff = self.block_diff(
                                frame1,
                                frame2,
                                width,
                                x,
                                y,
                                nx as usize,
                                ny as usize,
                                block_size,
                            );
                            min_diff = min_diff.min(diff);
                        }
                    }
                }

                motion_sum += min_diff;
                count += 1;
            }
        }

        if count > 0 {
            (motion_sum / count as f32 / 255.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn block_diff(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        size: usize,
    ) -> f32 {
        let mut diff = 0.0;

        for dy in 0..size {
            for dx in 0..size {
                let idx1 = ((y1 + dy) * width + (x1 + dx)) * 3;
                let idx2 = ((y2 + dy) * width + (x2 + dx)) * 3;

                if idx1 + 2 < frame1.len() && idx2 + 2 < frame2.len() {
                    for c in 0..3 {
                        diff += (frame1[idx1 + c] as i32 - frame2[idx2 + c] as i32).unsigned_abs()
                            as f32;
                    }
                }
            }
        }

        diff / (size * size * 3) as f32
    }

    fn detect_flashes(
        &self,
        frames: &[&[u8]],
        _width: usize,
        _height: usize,
    ) -> SceneResult<Vec<VideoEvent>> {
        let mut events = Vec::new();

        for i in 1..frames.len() {
            let brightness1 = self.avg_brightness(frames[i - 1]);
            let brightness2 = self.avg_brightness(frames[i]);

            let change = (brightness2 - brightness1).abs() / 255.0;

            if change > 0.7 {
                events.push(VideoEvent {
                    event_type: EventType::Flash,
                    frame_number: i,
                    confidence: Confidence::new(change.min(1.0)),
                    duration: 1,
                });
            }
        }

        Ok(events)
    }

    fn avg_brightness(&self, frame: &[u8]) -> f32 {
        let mut sum = 0.0;
        for i in (0..frame.len()).step_by(3) {
            sum += (frame[i] as f32 + frame[i + 1] as f32 + frame[i + 2] as f32) / 3.0;
        }
        sum / (frame.len() / 3) as f32
    }
}

impl Default for EventDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_detector() {
        let detector = EventDetector::new();
        // Use small frames to keep block-matching O(N) tractable.
        // 64×48: (64/32)*(48/32) = 2*1 = 2 blocks with 17*17 candidates each.
        let width = 64;
        let height = 48;
        let frame = vec![128u8; width * height * 3];
        let frames: Vec<&[u8]> = (0..10).map(|_| &frame[..]).collect();

        let result = detector.detect(&frames, width, height);
        assert!(result.is_ok());
    }

    #[test]
    fn test_event_type_name() {
        assert_eq!(EventType::SceneChange.name(), "Scene Change");
        assert_eq!(EventType::Flash.name(), "Flash");
    }
}
