//! NDI audio/video frame types and a dual-queue synchronisation buffer
//!
//! This module defines rich `NdiAudioFrame` and `NdiVideoFrame` types and a
//! `FrameSyncQueue` that manages bounded queues for both streams, enabling
//! detection of audio starvation.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]

/// The broad class of an NDI frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// A compressed or raw video frame.
    Video,
    /// A block of PCM audio samples.
    Audio,
    /// An XML metadata frame.
    Metadata,
    /// A status-change notification (e.g. connection lost).
    StatusChange,
}

impl FrameType {
    /// Returns `true` when the frame carries audio or video essence.
    pub fn is_av(&self) -> bool {
        matches!(self, Self::Video | Self::Audio)
    }
}

/// Nominal audio sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSampleRate {
    /// 44 100 Hz
    Hz44100,
    /// 48 000 Hz
    Hz48000,
    /// 96 000 Hz
    Hz96000,
}

impl AudioSampleRate {
    /// Return the sample rate in Hz.
    pub fn hz(&self) -> u32 {
        match self {
            Self::Hz44100 => 44_100,
            Self::Hz48000 => 48_000,
            Self::Hz96000 => 96_000,
        }
    }
}

/// A block of interleaved PCM audio samples received over NDI.
#[derive(Debug, Clone)]
pub struct NdiAudioFrame {
    /// Sample rate of the audio.
    pub sample_rate: AudioSampleRate,
    /// Number of audio channels.
    pub channels: u32,
    /// Interleaved f32 PCM samples (all channels).
    pub samples: Vec<f32>,
    /// 100-nanosecond timecode (NDI wire format).
    pub timecode: i64,
}

impl NdiAudioFrame {
    /// Create a new `NdiAudioFrame`.
    pub fn new(
        sample_rate: AudioSampleRate,
        channels: u32,
        samples: Vec<f32>,
        timecode: i64,
    ) -> Self {
        Self {
            sample_rate,
            channels,
            samples,
            timecode,
        }
    }

    /// Duration of this frame in milliseconds.
    pub fn duration_ms(&self) -> f32 {
        let samples_per_ch = self.samples_per_channel();
        samples_per_ch as f32 / self.sample_rate.hz() as f32 * 1000.0
    }

    /// Number of samples per channel (total samples ÷ channels).
    pub fn samples_per_channel(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }
}

/// A raw video frame descriptor received over NDI.
#[derive(Debug, Clone)]
pub struct NdiVideoFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame-rate numerator.
    pub frame_rate_n: u32,
    /// Frame-rate denominator (must be > 0).
    pub frame_rate_d: u32,
    /// 100-nanosecond timecode (NDI wire format).
    pub timecode: i64,
    /// Bytes per line (stride).
    pub line_stride_bytes: u32,
    /// Total data size in bytes.
    pub data_size_bytes: u32,
}

impl NdiVideoFrame {
    /// Create a new `NdiVideoFrame`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: u32,
        height: u32,
        frame_rate_n: u32,
        frame_rate_d: u32,
        timecode: i64,
        line_stride_bytes: u32,
        data_size_bytes: u32,
    ) -> Self {
        Self {
            width,
            height,
            frame_rate_n,
            frame_rate_d,
            timecode,
            line_stride_bytes,
            data_size_bytes,
        }
    }

    /// Frame rate as `f32`.
    pub fn frame_rate(&self) -> f32 {
        self.frame_rate_n as f32 / self.frame_rate_d as f32
    }

    /// Returns `true` when `line_stride_bytes` does **not** equal
    /// `width × 4` (packed RGBA), implying an interlaced or special stride.
    ///
    /// A simple heuristic: if the stride is exactly twice the expected packed
    /// stride, we treat it as interlaced.
    pub fn is_interlaced(&self) -> bool {
        let packed = self.width * 4;
        packed > 0 && self.line_stride_bytes == packed * 2
    }
}

/// Bounded dual-queue buffer for NDI audio and video frames.
#[derive(Debug)]
pub struct FrameSyncQueue {
    /// Maximum number of video frames to buffer.
    pub max_video_frames: usize,
    /// Maximum number of audio frames to buffer.
    pub max_audio_frames: usize,
    /// Buffered video frames (oldest first).
    pub video_queue: Vec<NdiVideoFrame>,
    /// Buffered audio frames (oldest first).
    pub audio_queue: Vec<NdiAudioFrame>,
}

impl FrameSyncQueue {
    /// Create a new `FrameSyncQueue` with the given capacities.
    pub fn new(max_video_frames: usize, max_audio_frames: usize) -> Self {
        Self {
            max_video_frames,
            max_audio_frames,
            video_queue: Vec::new(),
            audio_queue: Vec::new(),
        }
    }

    /// Add a video frame.  If the queue is full the oldest frame is dropped.
    pub fn add_video(&mut self, frame: NdiVideoFrame) {
        if self.video_queue.len() >= self.max_video_frames && self.max_video_frames > 0 {
            self.video_queue.remove(0);
        }
        self.video_queue.push(frame);
    }

    /// Add an audio frame.  If the queue is full the oldest frame is dropped.
    pub fn add_audio(&mut self, frame: NdiAudioFrame) {
        if self.audio_queue.len() >= self.max_audio_frames && self.max_audio_frames > 0 {
            self.audio_queue.remove(0);
        }
        self.audio_queue.push(frame);
    }

    /// Remove and return the oldest video frame.
    pub fn pop_video(&mut self) -> Option<NdiVideoFrame> {
        if self.video_queue.is_empty() {
            None
        } else {
            Some(self.video_queue.remove(0))
        }
    }

    /// Remove and return the oldest audio frame.
    pub fn pop_audio(&mut self) -> Option<NdiAudioFrame> {
        if self.audio_queue.is_empty() {
            None
        } else {
            Some(self.audio_queue.remove(0))
        }
    }

    /// Returns `true` when the audio queue is empty (starved).
    pub fn is_audio_starved(&self) -> bool {
        self.audio_queue.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_audio(rate: AudioSampleRate, channels: u32, sample_count: usize) -> NdiAudioFrame {
        NdiAudioFrame::new(rate, channels, vec![0.0_f32; sample_count], 0)
    }

    fn make_video(w: u32, h: u32, fps_n: u32, fps_d: u32) -> NdiVideoFrame {
        NdiVideoFrame::new(w, h, fps_n, fps_d, 0, w * 4, w * h * 4)
    }

    // --- FrameType ---

    #[test]
    fn test_frame_type_is_av_video() {
        assert!(FrameType::Video.is_av());
    }

    #[test]
    fn test_frame_type_is_av_audio() {
        assert!(FrameType::Audio.is_av());
    }

    #[test]
    fn test_frame_type_is_av_metadata_false() {
        assert!(!FrameType::Metadata.is_av());
    }

    #[test]
    fn test_frame_type_is_av_status_change_false() {
        assert!(!FrameType::StatusChange.is_av());
    }

    // --- AudioSampleRate ---

    #[test]
    fn test_sample_rate_hz_44100() {
        assert_eq!(AudioSampleRate::Hz44100.hz(), 44_100);
    }

    #[test]
    fn test_sample_rate_hz_48000() {
        assert_eq!(AudioSampleRate::Hz48000.hz(), 48_000);
    }

    #[test]
    fn test_sample_rate_hz_96000() {
        assert_eq!(AudioSampleRate::Hz96000.hz(), 96_000);
    }

    // --- NdiAudioFrame ---

    #[test]
    fn test_samples_per_channel() {
        let f = make_audio(AudioSampleRate::Hz48000, 2, 480 * 2);
        assert_eq!(f.samples_per_channel(), 480);
    }

    #[test]
    fn test_duration_ms_48k() {
        // 480 samples @ 48 kHz = 10 ms
        let f = make_audio(AudioSampleRate::Hz48000, 2, 480 * 2);
        assert!((f.duration_ms() - 10.0_f32).abs() < 0.01);
    }

    #[test]
    fn test_samples_per_channel_zero_channels() {
        let f = NdiAudioFrame::new(AudioSampleRate::Hz48000, 0, vec![], 0);
        assert_eq!(f.samples_per_channel(), 0);
    }

    // --- NdiVideoFrame ---

    #[test]
    fn test_frame_rate_30() {
        let f = make_video(1920, 1080, 30, 1);
        assert!((f.frame_rate() - 30.0_f32).abs() < 0.001);
    }

    #[test]
    fn test_frame_rate_29_97() {
        let f = make_video(1920, 1080, 30_000, 1001);
        let expected = 30_000_f32 / 1001_f32;
        assert!((f.frame_rate() - expected).abs() < 0.001);
    }

    #[test]
    fn test_is_interlaced_false_packed() {
        // stride == width * 4  → progressive
        let f = make_video(1920, 1080, 25, 1);
        assert!(!f.is_interlaced());
    }

    #[test]
    fn test_is_interlaced_true_double_stride() {
        let mut f = make_video(1920, 1080, 25, 1);
        f.line_stride_bytes = 1920 * 4 * 2; // double stride
        assert!(f.is_interlaced());
    }

    // --- FrameSyncQueue ---

    #[test]
    fn test_add_and_pop_video() {
        let mut q = FrameSyncQueue::new(4, 4);
        q.add_video(make_video(1920, 1080, 25, 1));
        assert!(q.pop_video().is_some());
        assert!(q.pop_video().is_none());
    }

    #[test]
    fn test_add_and_pop_audio() {
        let mut q = FrameSyncQueue::new(4, 4);
        q.add_audio(make_audio(AudioSampleRate::Hz48000, 2, 960));
        assert!(q.pop_audio().is_some());
    }

    #[test]
    fn test_is_audio_starved_initially() {
        let q = FrameSyncQueue::new(4, 4);
        assert!(q.is_audio_starved());
    }

    #[test]
    fn test_is_audio_starved_false_after_add() {
        let mut q = FrameSyncQueue::new(4, 4);
        q.add_audio(make_audio(AudioSampleRate::Hz48000, 2, 960));
        assert!(!q.is_audio_starved());
    }

    #[test]
    fn test_video_queue_bounded() {
        let mut q = FrameSyncQueue::new(2, 4);
        q.add_video(make_video(1920, 1080, 25, 1));
        q.add_video(make_video(1920, 1080, 25, 1));
        q.add_video(make_video(1920, 1080, 25, 1)); // should evict oldest
        assert_eq!(q.video_queue.len(), 2);
    }
}
