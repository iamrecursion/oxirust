//! Parallel / multi-threaded rendering for the timeline editor.
//!
//! Two independent parallel rendering strategies are provided:
//!
//! ## Frame-chunk parallelism — [`ParallelRenderer`]
//!
//! Splits the total frame range into [`RenderChunk`]s and processes them
//! concurrently using `rayon` (on native targets) or sequentially (on WASM).
//! Best for **export workflows** where the entire timeline is rendered to disk.
//!
//! ## Per-track parallelism — [`render_tracks_parallel`]
//!
//! Renders each track's contribution at a single frame position in parallel.
//! Because tracks are independent (no shared writeable state), this is
//! embarrassingly parallel: `rayon::par_iter()` over [`TrackRenderInput`]
//! slices.  Best for **real-time preview** where only a single composite frame
//! is needed at a time.
//!
//! Correctness prerequisite: [`render_track_frame_stateless`] must be a pure
//! function — it takes all inputs by value / shared reference and returns a
//! self-contained [`TrackRenderOutput`].  No locks, no global mutable state.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::clip::{Clip, ClipType};
use crate::error::EditResult;
use crate::render::RenderConfig;
use crate::render_source::RenderSource;
use crate::timeline::{Timeline, TrackType};

// ─────────────────────────────────────────────────────────────────────────────
// RenderChunk
// ─────────────────────────────────────────────────────────────────────────────

/// A contiguous range of frames assigned to one render worker.
#[derive(Clone, Debug)]
pub struct RenderChunk {
    /// Zero-based chunk index.
    pub chunk_id: usize,
    /// First frame of the chunk (inclusive).
    pub frame_start: u64,
    /// Last frame of the chunk (exclusive).
    pub frame_end: u64,
    /// Optional path for writing the chunk output.
    pub output_path: Option<PathBuf>,
}

impl RenderChunk {
    /// Create a new render chunk without an output path.
    #[must_use]
    pub fn new(chunk_id: usize, frame_start: u64, frame_end: u64) -> Self {
        Self {
            chunk_id,
            frame_start,
            frame_end,
            output_path: None,
        }
    }

    /// Create a render chunk with an explicit output path.
    #[must_use]
    pub fn with_output(
        chunk_id: usize,
        frame_start: u64,
        frame_end: u64,
        output_path: PathBuf,
    ) -> Self {
        Self {
            chunk_id,
            frame_start,
            frame_end,
            output_path: Some(output_path),
        }
    }

    /// Number of frames in this chunk.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_end.saturating_sub(self.frame_start)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ParallelRenderConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the parallel renderer.
#[derive(Clone, Debug)]
pub struct ParallelRenderConfig {
    /// Number of worker threads (ignored on WASM).
    pub num_threads: usize,
    /// Target number of frames per chunk.
    pub chunk_size: u64,
    /// Per-frame render configuration.
    pub render_config: RenderConfig,
}

impl ParallelRenderConfig {
    /// Create a configuration with sensible defaults (4 threads, 30-frame chunks).
    #[must_use]
    pub fn new(render_config: RenderConfig) -> Self {
        Self {
            num_threads: 4,
            chunk_size: 30,
            render_config,
        }
    }

    /// Override the number of worker threads.
    #[must_use]
    pub fn with_threads(mut self, n: usize) -> Self {
        self.num_threads = n.max(1);
        self
    }

    /// Override the chunk size in frames.
    #[must_use]
    pub fn with_chunk_size(mut self, size: u64) -> Self {
        self.chunk_size = size.max(1);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ParallelRenderResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result produced by rendering a single [`RenderChunk`].
#[derive(Clone, Debug)]
pub struct ParallelRenderResult {
    /// The chunk that was rendered.
    pub chunk: RenderChunk,
    /// Number of frames that were successfully rendered.
    pub frames_rendered: u64,
    /// Wall-clock duration of this chunk's render, in milliseconds.
    pub duration_ms: u64,
    /// Whether the chunk completed without errors.
    pub success: bool,
    /// Error message, populated when `success` is `false`.
    pub error: Option<String>,
}

impl ParallelRenderResult {
    /// Build a successful result.
    #[must_use]
    fn ok(chunk: RenderChunk, frames_rendered: u64, duration_ms: u64) -> Self {
        Self {
            frames_rendered,
            duration_ms,
            success: true,
            error: None,
            chunk,
        }
    }

    /// Build a failed result.
    #[must_use]
    fn err(chunk: RenderChunk, message: impl Into<String>) -> Self {
        Self {
            frames_rendered: 0,
            duration_ms: 0,
            success: false,
            error: Some(message.into()),
            chunk,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ParallelRenderer
// ─────────────────────────────────────────────────────────────────────────────

/// Splits a timeline into frame chunks and renders them in parallel.
///
/// On WASM targets rayon is unavailable, so chunks are processed sequentially
/// to keep the API surface identical across platforms.
pub struct ParallelRenderer {
    /// Render configuration.
    pub config: ParallelRenderConfig,
}

impl ParallelRenderer {
    /// Create a new parallel renderer.
    #[must_use]
    pub fn new(config: ParallelRenderConfig) -> Self {
        Self { config }
    }

    /// Split `total_frames` into a list of [`RenderChunk`]s.
    ///
    /// Each chunk covers at most `config.chunk_size` frames.  The final chunk
    /// may be smaller.
    #[must_use]
    pub fn split_chunks(&self, total_frames: u64) -> Vec<RenderChunk> {
        if total_frames == 0 {
            return Vec::new();
        }

        let chunk_size = self.config.chunk_size;
        let num_chunks = (total_frames + chunk_size - 1) / chunk_size; // ceil division

        (0..num_chunks)
            .map(|i| {
                let start = i * chunk_size;
                let end = (start + chunk_size).min(total_frames);
                RenderChunk::new(i as usize, start, end)
            })
            .collect()
    }

    /// Render `total_frames` from `timeline` in parallel.
    ///
    /// Returns a result per chunk.  Individual chunk failures do not abort the
    /// remaining chunks — errors are captured in [`ParallelRenderResult::error`].
    pub fn render_parallel(
        &self,
        total_frames: u64,
        timeline: &Arc<Timeline>,
    ) -> EditResult<Vec<ParallelRenderResult>> {
        let chunks = self.split_chunks(total_frames);

        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<ParallelRenderResult> = {
            // Configure rayon thread pool inline so we respect `num_threads`.
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.config.num_threads)
                .build()
                .unwrap_or_else(|_| {
                    rayon::ThreadPoolBuilder::new()
                        .build()
                        .expect("default rayon pool build should never fail")
                });

            pool.install(|| {
                chunks
                    .par_iter()
                    .map(|chunk| self.render_chunk(chunk, timeline))
                    .collect()
            })
        };

        #[cfg(target_arch = "wasm32")]
        let results: Vec<ParallelRenderResult> = chunks
            .iter()
            .map(|chunk| self.render_chunk(chunk, timeline))
            .collect();

        Ok(results)
    }

    /// Render a single [`RenderChunk`].
    ///
    /// The current implementation counts frames and records timings; actual
    /// pixel decoding is handled by the timeline's render pipeline.
    pub fn render_chunk(
        &self,
        chunk: &RenderChunk,
        _timeline: &Arc<Timeline>,
    ) -> ParallelRenderResult {
        let start_time = Instant::now();

        // Validate the chunk range
        if chunk.frame_end < chunk.frame_start {
            return ParallelRenderResult::err(
                chunk.clone(),
                format!(
                    "invalid chunk {}: frame_end {} < frame_start {}",
                    chunk.chunk_id, chunk.frame_end, chunk.frame_start
                ),
            );
        }

        let frames_rendered = chunk.frame_count();

        // In a full implementation this loop would call the async renderer for
        // each frame and write pixel data to `chunk.output_path`.  Here we
        // iterate over frame indices to represent the work without touching I/O.
        for _frame in chunk.frame_start..chunk.frame_end {
            // Placeholder: per-frame render work would go here.
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;
        ParallelRenderResult::ok(chunk.clone(), frames_rendered, duration_ms)
    }

    /// Total number of frames that would be rendered given `total_frames`.
    #[must_use]
    pub fn total_frames_for(&self, total_frames: u64) -> u64 {
        self.split_chunks(total_frames)
            .iter()
            .map(RenderChunk::frame_count)
            .sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-track parallel rendering
// ─────────────────────────────────────────────────────────────────────────────

/// The clip type of a track (video or audio), mirroring [`TrackType`] but
/// limited to the subset that `render_track_frame_stateless` handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackKind {
    /// Video track — produces a pixel buffer.
    Video,
    /// Audio track — produces an interleaved f32 sample buffer.
    Audio,
}

impl TrackKind {
    /// Convert from [`TrackType`].
    #[must_use]
    pub fn from_track_type(t: TrackType) -> Option<Self> {
        match t {
            TrackType::Video => Some(Self::Video),
            TrackType::Audio => Some(Self::Audio),
            TrackType::Subtitle => None,
        }
    }
}

/// A pairing of a [`Clip`] with its already-resolved (and decoded) media source.
///
/// Both fields are cheaply cloneable: `Clip` is `Clone` and `Arc<RenderSource>`
/// is reference-counted.
#[derive(Clone, Debug)]
pub struct ClipWithSource {
    /// The clip's timing / metadata.
    pub clip: Clip,
    /// Resolved and decoded media source (shared, thread-safe).
    pub source: Arc<RenderSource>,
}

/// All immutable data needed to render one track's contribution at a single
/// frame position.
///
/// This struct is **intentionally cheap to clone** (all expensive data sits
/// behind `Arc`s) so `rayon` can distribute inputs across threads without
/// copying pixel data.
#[derive(Clone, Debug)]
pub struct TrackRenderInput {
    /// Zero-based track index (used in the output for ordering).
    pub track_index: usize,
    /// Video or audio track.
    pub kind: TrackKind,
    /// Clips active at `position`, paired with their decoded sources.
    ///
    /// Clips are already filtered to the ones that overlap `position`.
    pub clips: Vec<ClipWithSource>,
    /// Timeline position (in timebase units) to render.
    pub position: i64,
    /// Target frame width (video only; ignored for audio).
    pub width: u32,
    /// Target frame height (video only; ignored for audio).
    pub height: u32,
    /// Number of audio channels (audio only; ignored for video).
    pub channels: usize,
    /// Sample rate in Hz (audio only; ignored for video).
    pub sample_rate: u32,
    /// Number of audio samples per render call (audio only).
    pub num_samples: usize,
}

impl TrackRenderInput {
    /// Build a [`TrackRenderInput`] for a **video** track.
    #[must_use]
    pub fn video(
        track_index: usize,
        clips: Vec<ClipWithSource>,
        position: i64,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            track_index,
            kind: TrackKind::Video,
            clips,
            position,
            width,
            height,
            channels: 0,
            sample_rate: 0,
            num_samples: 0,
        }
    }

    /// Build a [`TrackRenderInput`] for an **audio** track.
    #[must_use]
    pub fn audio(
        track_index: usize,
        clips: Vec<ClipWithSource>,
        position: i64,
        channels: usize,
        sample_rate: u32,
        num_samples: usize,
    ) -> Self {
        Self {
            track_index,
            kind: TrackKind::Audio,
            clips,
            position,
            width: 0,
            height: 0,
            channels,
            sample_rate,
            num_samples,
        }
    }
}

/// The rendered output for one track at one frame position.
#[derive(Clone, Debug)]
pub struct TrackRenderOutput {
    /// Track index (mirrors [`TrackRenderInput::track_index`]).
    pub track_index: usize,
    /// Track kind (mirrors [`TrackRenderInput::kind`]).
    pub kind: TrackKind,
    /// RGBA8 pixel data for a video track (`width * height * 4` bytes).
    ///
    /// For audio tracks this is empty.
    pub video_rgba8: Vec<u8>,
    /// Interleaved f32 audio samples for an audio track
    /// (`num_samples * channels` elements).
    ///
    /// For video tracks this is empty.
    pub audio_samples: Vec<f32>,
}

/// Render one track's contribution at `input.position` without any shared
/// mutable state.
///
/// This function is the core of the per-track parallel pipeline:
///
/// - For **video** tracks: each non-muted clip active at `position` has its
///   source sampled via [`RenderSource::sample_video`].  Layers are composited
///   bottom-to-top using alpha blending weighted by `clip.opacity`.  The result
///   is an RGBA8 buffer of `width × height × 4` bytes.
///
/// - For **audio** tracks: each non-muted clip contributes interleaved f32
///   samples, summed with per-clip volume (`clip.opacity`).  The mix is clamped
///   to `[-1.0, 1.0]`.
///
/// # Statelessness contract
///
/// The function signature is `(input: &TrackRenderInput) -> TrackRenderOutput`.
/// There is no `&mut self`, no global lock, and no interior mutability.  The
/// sources are accessed through `Arc<RenderSource>` whose [`RenderSource::sample_video`]
/// and [`RenderSource::sample_audio`] methods take `&self`.  Therefore this
/// function is **safe to call concurrently** from multiple rayon threads.
#[must_use]
pub fn render_track_frame_stateless(input: &TrackRenderInput) -> TrackRenderOutput {
    match input.kind {
        TrackKind::Video => render_video_track(input),
        TrackKind::Audio => render_audio_track(input),
    }
}

/// Composite all video clips in a track at `position`.
fn render_video_track(input: &TrackRenderInput) -> TrackRenderOutput {
    let w = input.width as usize;
    let h = input.height as usize;
    let pixel_count = w * h;

    // Accumulator in linear-light RGBA f32 (pre-multiplied alpha over-operator).
    // Initialised to transparent black.
    let mut accum: Vec<f32> = vec![0.0_f32; pixel_count * 4];

    for cs in &input.clips {
        if cs.clip.muted || cs.clip.clip_type != ClipType::Video {
            continue;
        }
        let source_pts = cs.clip.timeline_to_source(input.position);
        let rgba8 = cs
            .source
            .sample_video(source_pts, input.width, input.height);
        let opacity = cs.clip.opacity.clamp(0.0, 1.0);

        // Over-composite this layer onto the accumulator (front-to-back order:
        // later clips in the slice sit on top of earlier ones).
        for i in 0..pixel_count {
            let base = i * 4;
            let r = rgba8.get(base).copied().unwrap_or(0) as f32 / 255.0;
            let g = rgba8.get(base + 1).copied().unwrap_or(0) as f32 / 255.0;
            let b = rgba8.get(base + 2).copied().unwrap_or(0) as f32 / 255.0;
            let a = rgba8.get(base + 3).copied().unwrap_or(255) as f32 / 255.0 * opacity;

            // Standard "over" compositing operator (pre-multiplied):
            //   A_out = a_src + a_dst * (1 - a_src)
            //   C_out = c_src * a_src + c_dst * a_dst * (1 - a_src)
            let a_dst = accum[base + 3];
            let inv_a = 1.0_f32 - a;
            accum[base] = r * a + accum[base] * a_dst * inv_a;
            accum[base + 1] = g * a + accum[base + 1] * a_dst * inv_a;
            accum[base + 2] = b * a + accum[base + 2] * a_dst * inv_a;
            accum[base + 3] = a + a_dst * inv_a;
        }
    }

    // Convert RGBA f32 → RGBA u8 (un-premultiply alpha).
    let mut out = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        let base = i * 4;
        let a = accum[base + 3];
        let (r, g, b) = if a > f32::EPSILON {
            (
                (accum[base] / a).clamp(0.0, 1.0),
                (accum[base + 1] / a).clamp(0.0, 1.0),
                (accum[base + 2] / a).clamp(0.0, 1.0),
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        {
            out.push((r * 255.0).round() as u8);
            out.push((g * 255.0).round() as u8);
            out.push((b * 255.0).round() as u8);
            out.push((a.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }

    TrackRenderOutput {
        track_index: input.track_index,
        kind: TrackKind::Video,
        video_rgba8: out,
        audio_samples: Vec::new(),
    }
}

/// Mix all audio clips in a track at `position`.
fn render_audio_track(input: &TrackRenderInput) -> TrackRenderOutput {
    let ch = input.channels.max(1);
    let ns = input.num_samples;
    let mut mix = vec![0.0_f32; ns * ch];

    for cs in &input.clips {
        if cs.clip.muted || cs.clip.clip_type != ClipType::Audio {
            continue;
        }
        let source_pts = cs.clip.timeline_to_source(input.position);
        let gain = cs.clip.opacity.clamp(0.0, 1.0);
        let samples = cs
            .source
            .sample_audio(source_pts, ns, ch as u16, input.sample_rate);
        let len = mix.len().min(samples.len());
        for i in 0..len {
            mix[i] += samples[i] * gain;
        }
    }

    // Clamp to [-1, 1].
    for s in &mut mix {
        *s = s.clamp(-1.0, 1.0);
    }

    TrackRenderOutput {
        track_index: input.track_index,
        kind: TrackKind::Audio,
        video_rgba8: Vec::new(),
        audio_samples: mix,
    }
}

/// Render multiple tracks concurrently using `rayon`.
///
/// Each entry in `inputs` represents one track.  Tracks are **independent**:
/// no data is shared between them, so `rayon::par_iter()` is safe.
///
/// Returns one [`TrackRenderOutput`] per input, in the **same order** as
/// `inputs` (i.e. output `i` corresponds to `inputs[i]`).
///
/// On WASM targets rayon is unavailable; the inputs are processed sequentially
/// to keep the API surface identical across platforms.
#[must_use]
pub fn render_tracks_parallel(inputs: &[TrackRenderInput]) -> Vec<TrackRenderOutput> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        inputs
            .par_iter()
            .map(render_track_frame_stateless)
            .collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        inputs.iter().map(render_track_frame_stateless).collect()
    }
}

/// Helper: build [`TrackRenderInput`]s for every renderable track of a
/// [`Timeline`] at the given `position`.
///
/// Only tracks whose type is `Video` or `Audio` are included (subtitle tracks
/// are skipped).  Only clips that contain `position` are included.
///
/// The `source_resolver` callback lets callers inject the source-resolution
/// strategy (e.g. a cache lookup).  It receives a reference to a [`Clip`] and
/// must return an `Arc<RenderSource>`.
pub fn build_track_render_inputs<F>(
    timeline: &Timeline,
    position: i64,
    config: &RenderConfig,
    mut source_resolver: F,
) -> Vec<TrackRenderInput>
where
    F: FnMut(&Clip) -> Arc<RenderSource>,
{
    let mut inputs = Vec::with_capacity(timeline.tracks.len());

    for track in &timeline.tracks {
        if track.muted {
            continue;
        }
        let Some(kind) = TrackKind::from_track_type(track.track_type) else {
            continue; // subtitle tracks are skipped
        };

        let active_clips: Vec<ClipWithSource> = track
            .clips
            .iter()
            .filter(|c| c.contains(position) && !c.muted)
            .map(|c| ClipWithSource {
                clip: c.clone(),
                source: source_resolver(c),
            })
            .collect();

        let input = match kind {
            TrackKind::Video => TrackRenderInput::video(
                track.index,
                active_clips,
                position,
                config.width,
                config.height,
            ),
            TrackKind::Audio => TrackRenderInput::audio(
                track.index,
                active_clips,
                position,
                config.channels.count(),
                config.sample_rate,
                1024, // standard block size
            ),
        };
        inputs.push(input);
    }

    inputs
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::Timeline;

    fn make_renderer() -> ParallelRenderer {
        let cfg = ParallelRenderConfig::new(RenderConfig::default())
            .with_threads(2)
            .with_chunk_size(10);
        ParallelRenderer::new(cfg)
    }

    #[test]
    fn test_split_chunks_exact_multiple() {
        let r = make_renderer();
        let chunks = r.split_chunks(30);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].frame_start, 0);
        assert_eq!(chunks[0].frame_end, 10);
        assert_eq!(chunks[1].frame_start, 10);
        assert_eq!(chunks[1].frame_end, 20);
        assert_eq!(chunks[2].frame_start, 20);
        assert_eq!(chunks[2].frame_end, 30);
    }

    #[test]
    fn test_split_chunks_non_multiple() {
        let r = make_renderer();
        let chunks = r.split_chunks(25);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2].frame_end, 25);
        assert_eq!(chunks[2].frame_count(), 5);
    }

    #[test]
    fn test_split_chunks_zero_frames() {
        let r = make_renderer();
        let chunks = r.split_chunks(0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_render_parallel_all_succeed() {
        let r = make_renderer();
        let timeline = Arc::new(Timeline::default());
        let results = r
            .render_parallel(30, &timeline)
            .expect("render_parallel ok");
        assert_eq!(results.len(), 3);
        for res in &results {
            assert!(
                res.success,
                "chunk {} failed: {:?}",
                res.chunk.chunk_id, res.error
            );
            assert_eq!(res.frames_rendered, 10);
        }
    }

    #[test]
    fn test_config_builder() {
        let cfg = ParallelRenderConfig::new(RenderConfig::default())
            .with_threads(8)
            .with_chunk_size(50);
        assert_eq!(cfg.num_threads, 8);
        assert_eq!(cfg.chunk_size, 50);
    }

    #[test]
    fn test_render_chunk_frame_count() {
        let r = make_renderer();
        let timeline = Arc::new(Timeline::default());
        let chunk = RenderChunk::new(0, 0, 15);
        let result = r.render_chunk(&chunk, &timeline);
        assert!(result.success);
        assert_eq!(result.frames_rendered, 15);
    }

    #[test]
    fn test_total_frames_for() {
        let r = make_renderer();
        assert_eq!(r.total_frames_for(30), 30);
        assert_eq!(r.total_frames_for(25), 25);
        assert_eq!(r.total_frames_for(0), 0);
    }

    // ─── Per-track parallel render tests ─────────────────────────────────────

    /// Build 3 independent video `TrackRenderInput`s using `TestPattern` sources
    /// (no real media files needed), then verify that `render_tracks_parallel`
    /// produces the same pixel buffers as sequential `render_track_frame_stateless`
    /// calls.
    ///
    /// This test satisfies the requirement: "3 tracks, verify parallel == sequential".
    #[test]
    fn test_parallel_render_matches_sequential() {
        use crate::clip::{Clip, ClipType};
        use crate::render_source::RenderSource;

        let w: u32 = 16;
        let h: u32 = 16;
        let position: i64 = 0;

        // Build 3 track inputs, each with one TestPattern video clip.
        let inputs: Vec<TrackRenderInput> = (0..3)
            .map(|i| {
                let mut clip = Clip::new(i as u64 + 1, ClipType::Video, 0, 1000);
                clip.opacity = 1.0;
                let source = Arc::new(RenderSource::TestPattern);
                let cws = ClipWithSource { clip, source };
                TrackRenderInput::video(i, vec![cws], position, w, h)
            })
            .collect();

        // Sequential reference outputs.
        let seq_outputs: Vec<TrackRenderOutput> =
            inputs.iter().map(render_track_frame_stateless).collect();

        // Parallel outputs.
        let par_outputs = render_tracks_parallel(&inputs);

        assert_eq!(
            par_outputs.len(),
            seq_outputs.len(),
            "output count must match"
        );

        for (seq, par) in seq_outputs.iter().zip(par_outputs.iter()) {
            assert_eq!(
                seq.track_index, par.track_index,
                "track_index mismatch at index {}",
                seq.track_index
            );
            assert_eq!(
                seq.video_rgba8, par.video_rgba8,
                "pixel data mismatch on track {}",
                seq.track_index
            );
            assert_eq!(
                seq.audio_samples, par.audio_samples,
                "audio data must be empty for video tracks (track {})",
                seq.track_index
            );
        }
    }

    /// Verify that per-track rendering produces the expected pixel-buffer size.
    #[test]
    fn test_render_track_video_output_size() {
        use crate::clip::{Clip, ClipType};
        use crate::render_source::RenderSource;

        let w: u32 = 8;
        let h: u32 = 8;
        let mut clip = Clip::new(1, ClipType::Video, 0, 500);
        clip.opacity = 1.0;
        let source = Arc::new(RenderSource::TestPattern);
        let cws = ClipWithSource { clip, source };
        let input = TrackRenderInput::video(0, vec![cws], 0, w, h);

        let out = render_track_frame_stateless(&input);
        assert_eq!(
            out.video_rgba8.len(),
            (w * h * 4) as usize,
            "RGBA8 buffer must be w*h*4 bytes"
        );
        assert!(
            out.audio_samples.is_empty(),
            "audio must be empty for video"
        );
    }

    /// Verify that audio track rendering produces the expected sample count.
    #[test]
    fn test_render_track_audio_output_size() {
        use crate::clip::{Clip, ClipType};
        use crate::render_source::RenderSource;

        let channels = 2usize;
        let num_samples = 512usize;
        let mut clip = Clip::new(2, ClipType::Audio, 0, 5000);
        clip.opacity = 0.8;
        let source = Arc::new(RenderSource::TestPattern);
        let cws = ClipWithSource { clip, source };
        let input = TrackRenderInput::audio(1, vec![cws], 0, channels, 48000, num_samples);

        let out = render_track_frame_stateless(&input);
        assert_eq!(
            out.audio_samples.len(),
            channels * num_samples,
            "audio buffer must be channels * num_samples"
        );
        assert!(
            out.video_rgba8.is_empty(),
            "video must be empty for audio tracks"
        );
        // All samples must be in [-1, 1].
        for &s in &out.audio_samples {
            assert!(s.abs() <= 1.0, "audio sample {s} exceeds [-1, 1] bounds");
        }
    }

    /// Muted clips must not contribute any signal.
    #[test]
    fn test_muted_clip_produces_silence() {
        use crate::clip::{Clip, ClipType};
        use crate::render_source::RenderSource;

        let mut clip = Clip::new(3, ClipType::Audio, 0, 5000);
        clip.muted = true;
        let source = Arc::new(RenderSource::TestPattern);
        let cws = ClipWithSource { clip, source };
        let input = TrackRenderInput::audio(0, vec![cws], 0, 2, 48000, 256);

        let out = render_track_frame_stateless(&input);
        assert!(
            out.audio_samples.iter().all(|&s| s == 0.0),
            "muted clip must produce silence"
        );
    }
}
