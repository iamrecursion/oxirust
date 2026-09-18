//! Sprite sheet management and animation.
//!
//! Provides types for describing sprite sheets -- grid-based image atlases
//! where each cell is an animation frame -- and a simple playback controller
//! that advances through frames at a given framerate.

#![allow(dead_code)]

/// A single rectangular region within a sprite sheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpriteFrame {
    /// X-offset in pixels from the top-left of the atlas.
    pub x: u32,
    /// Y-offset in pixels from the top-left of the atlas.
    pub y: u32,
    /// Width of the frame in pixels.
    pub width: u32,
    /// Height of the frame in pixels.
    pub height: u32,
}

impl SpriteFrame {
    /// Create a new [`SpriteFrame`].
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Area of the frame in pixels.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns `true` when the frame has non-zero dimensions.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0
    }
}

/// A sprite sheet composed of a uniform grid of frames.
#[derive(Debug, Clone)]
pub struct SpriteSheet {
    /// Total atlas width in pixels.
    pub atlas_width: u32,
    /// Total atlas height in pixels.
    pub atlas_height: u32,
    /// Ordered list of frames.
    pub frames: Vec<SpriteFrame>,
    /// Frame width (uniform).
    pub frame_width: u32,
    /// Frame height (uniform).
    pub frame_height: u32,
}

impl SpriteSheet {
    /// Build a [`SpriteSheet`] from atlas dimensions and a uniform cell size.
    ///
    /// Frames are enumerated left-to-right, top-to-bottom.
    #[must_use]
    pub fn from_grid(
        atlas_width: u32,
        atlas_height: u32,
        frame_width: u32,
        frame_height: u32,
    ) -> Self {
        let cols = atlas_width.checked_div(frame_width).unwrap_or(0);
        let rows = atlas_height.checked_div(frame_height).unwrap_or(0);
        let mut frames = Vec::with_capacity((cols * rows) as usize);

        for row in 0..rows {
            for col in 0..cols {
                frames.push(SpriteFrame::new(
                    col * frame_width,
                    row * frame_height,
                    frame_width,
                    frame_height,
                ));
            }
        }

        Self {
            atlas_width,
            atlas_height,
            frames,
            frame_width,
            frame_height,
        }
    }

    /// Number of frames in the sheet.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Retrieve a frame by zero-based index.
    #[must_use]
    pub fn get_frame(&self, index: usize) -> Option<&SpriteFrame> {
        self.frames.get(index)
    }

    /// Number of columns in the grid.
    #[must_use]
    pub fn columns(&self) -> u32 {
        self.atlas_width.checked_div(self.frame_width).unwrap_or(0)
    }

    /// Number of rows in the grid.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.atlas_height
            .checked_div(self.frame_height)
            .unwrap_or(0)
    }
}

/// Playback mode for a sprite animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackMode {
    /// Play once and stop on the last frame.
    Once,
    /// Loop forever from start after the last frame.
    Loop,
    /// Ping-pong: play forward then backward, repeating.
    PingPong,
}

/// Animation controller for a [`SpriteSheet`].
#[derive(Debug, Clone)]
pub struct SpriteAnimation {
    /// Number of frames in the animation.
    total_frames: usize,
    /// Frames per second.
    fps: f32,
    /// Playback mode.
    mode: PlaybackMode,
    /// Accumulated time in seconds.
    elapsed: f32,
}

impl SpriteAnimation {
    /// Create a new [`SpriteAnimation`].
    ///
    /// # Arguments
    /// * `total_frames` -- Total frames available.
    /// * `fps`          -- Playback framerate.
    /// * `mode`         -- Playback mode.
    #[must_use]
    pub fn new(total_frames: usize, fps: f32, mode: PlaybackMode) -> Self {
        Self {
            total_frames,
            fps: fps.max(0.001),
            mode,
            elapsed: 0.0,
        }
    }

    /// Advance the animation clock by `dt` seconds.
    pub fn advance(&mut self, dt: f32) {
        self.elapsed += dt;
    }

    /// Reset the animation to the beginning.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
    }

    /// Compute the current frame index based on elapsed time.
    #[must_use]
    pub fn current_frame(&self) -> usize {
        if self.total_frames == 0 {
            return 0;
        }

        let raw = (self.elapsed * self.fps) as usize;

        match self.mode {
            PlaybackMode::Once => raw.min(self.total_frames - 1),
            PlaybackMode::Loop => raw % self.total_frames,
            PlaybackMode::PingPong => {
                let cycle = if self.total_frames > 1 {
                    (self.total_frames - 1) * 2
                } else {
                    1
                };
                let pos = raw % cycle;
                if pos < self.total_frames {
                    pos
                } else {
                    cycle - pos
                }
            }
        }
    }

    /// Returns `true` when the one-shot animation has finished.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        if self.total_frames == 0 {
            return true;
        }
        if self.mode != PlaybackMode::Once {
            return false;
        }
        let raw = (self.elapsed * self.fps) as usize;
        raw >= self.total_frames - 1
    }

    /// Total duration of one cycle in seconds.
    #[must_use]
    pub fn cycle_duration(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.total_frames as f32 / self.fps
    }

    /// Current playback mode.
    #[must_use]
    pub fn mode(&self) -> PlaybackMode {
        self.mode
    }
}

// -- unit tests --

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprite_frame_new() {
        let f = SpriteFrame::new(10, 20, 64, 64);
        assert_eq!(f.x, 10);
        assert_eq!(f.y, 20);
        assert_eq!(f.width, 64);
        assert_eq!(f.height, 64);
    }

    #[test]
    fn test_sprite_frame_area() {
        let f = SpriteFrame::new(0, 0, 100, 50);
        assert_eq!(f.area(), 5000);
    }

    #[test]
    fn test_sprite_frame_validity() {
        assert!(SpriteFrame::new(0, 0, 10, 10).is_valid());
        assert!(!SpriteFrame::new(0, 0, 0, 10).is_valid());
    }

    #[test]
    fn test_sprite_sheet_from_grid() {
        let sheet = SpriteSheet::from_grid(256, 128, 64, 64);
        assert_eq!(sheet.columns(), 4);
        assert_eq!(sheet.rows(), 2);
        assert_eq!(sheet.frame_count(), 8);
    }

    #[test]
    fn test_sprite_sheet_frame_positions() {
        let sheet = SpriteSheet::from_grid(128, 64, 32, 32);
        // 4 columns, 2 rows = 8 frames
        assert_eq!(sheet.get_frame(0), Some(&SpriteFrame::new(0, 0, 32, 32)));
        assert_eq!(sheet.get_frame(4), Some(&SpriteFrame::new(0, 32, 32, 32)));
    }

    #[test]
    fn test_sprite_sheet_zero_cell_size() {
        let sheet = SpriteSheet::from_grid(256, 256, 0, 0);
        assert_eq!(sheet.frame_count(), 0);
        assert_eq!(sheet.columns(), 0);
        assert_eq!(sheet.rows(), 0);
    }

    #[test]
    fn test_animation_once() {
        let mut anim = SpriteAnimation::new(4, 10.0, PlaybackMode::Once);
        assert_eq!(anim.current_frame(), 0);
        anim.advance(0.15); // 1.5 frames -> frame 1
        assert_eq!(anim.current_frame(), 1);
        anim.advance(10.0); // way past end
        assert_eq!(anim.current_frame(), 3);
        assert!(anim.is_finished());
    }

    #[test]
    fn test_animation_loop() {
        let mut anim = SpriteAnimation::new(4, 10.0, PlaybackMode::Loop);
        anim.advance(0.5); // 5 frames -> 5 % 4 = 1
        assert_eq!(anim.current_frame(), 1);
        assert!(!anim.is_finished());
    }

    #[test]
    fn test_animation_pingpong() {
        let mut anim = SpriteAnimation::new(4, 10.0, PlaybackMode::PingPong);
        // cycle = (4-1)*2 = 6
        // frame 0->1->2->3->2->1->0->...
        anim.advance(0.4); // raw frame 4 => pos=4%6=4 => cycle-pos=6-4=2
        assert_eq!(anim.current_frame(), 2);
    }

    #[test]
    fn test_animation_reset() {
        let mut anim = SpriteAnimation::new(8, 30.0, PlaybackMode::Loop);
        anim.advance(1.0);
        assert_ne!(anim.current_frame(), 0);
        anim.reset();
        assert_eq!(anim.current_frame(), 0);
    }

    #[test]
    fn test_cycle_duration() {
        let anim = SpriteAnimation::new(10, 25.0, PlaybackMode::Loop);
        assert!((anim.cycle_duration() - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_animation_zero_frames() {
        let anim = SpriteAnimation::new(0, 30.0, PlaybackMode::Once);
        assert_eq!(anim.current_frame(), 0);
        assert!(anim.is_finished());
        assert_eq!(anim.cycle_duration(), 0.0);
    }

    #[test]
    fn test_animation_mode_getter() {
        let anim = SpriteAnimation::new(4, 10.0, PlaybackMode::PingPong);
        assert_eq!(anim.mode(), PlaybackMode::PingPong);
    }

    #[test]
    fn test_get_frame_out_of_bounds() {
        let sheet = SpriteSheet::from_grid(64, 64, 32, 32);
        assert!(sheet.get_frame(100).is_none());
    }

    #[test]
    fn test_animation_single_frame() {
        let mut anim = SpriteAnimation::new(1, 10.0, PlaybackMode::PingPong);
        anim.advance(5.0);
        assert_eq!(anim.current_frame(), 0);
    }
}
