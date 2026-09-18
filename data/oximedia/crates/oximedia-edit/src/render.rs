//! Timeline rendering.
//!
//! Renders the timeline to a stream of video and audio frames.

use bytes::Bytes;
use oximedia_audio::{AudioBuffer, AudioFrame, ChannelLayout};
use oximedia_codec::VideoFrame;
use oximedia_core::{PixelFormat, Rational, SampleFormat, Timestamp};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc;

use crate::clip::Clip;
use crate::error::EditResult;
use crate::frame_prefetch::{PrefetchConfig, PrefetchEngine};
use crate::incremental_render::DirtyRegion;
use crate::parallel_render::{
    render_tracks_parallel, ClipWithSource, TrackKind, TrackRenderInput, TrackRenderOutput,
};
use crate::render_source::RenderSource;
use crate::timeline::{Timeline, TimelineConfig, TrackType};
use crate::transition::Transition;

/// Timeline renderer.
pub struct TimelineRenderer {
    /// Timeline to render.
    timeline: Arc<Timeline>,
    /// Render configuration.
    config: RenderConfig,
    /// Frame cache.
    cache: FrameCache,
    /// Raw byte-buffer cache for source frames, keyed by `(clip_id, source_pts)`.
    raw_frame_cache: RawFrameCache,
    /// Per-path decoded source cache.  Shared via `Arc` so clips pointing at the
    /// same file decode it only once.
    source_cache: HashMap<PathBuf, Arc<RenderSource>>,
    /// Dirty regions: only frames in these ranges need re-rendering.
    /// An empty list means everything is clean (or no incremental tracking active).
    dirty_regions: Vec<DirtyRegion>,
    /// Predictive prefetch engine.  Advances after each render call to warm the
    /// cache for upcoming frames.
    prefetch: PrefetchEngine,
    /// When `true` multi-track decode is routed through `render_tracks_parallel`
    /// instead of the sequential compositor loop.
    use_parallel: bool,
}

impl TimelineRenderer {
    /// Create a new timeline renderer.
    #[must_use]
    pub fn new(timeline: Arc<Timeline>, config: RenderConfig) -> Self {
        let cache_size = config.cache_size;
        let max_pos = timeline.duration.max(0) as i64;
        let prefetch_config = PrefetchConfig::for_playback(30.0, 1.0);
        Self {
            timeline,
            config,
            cache: FrameCache::new(cache_size),
            raw_frame_cache: RawFrameCache::new(RAW_FRAME_CACHE_CAPACITY),
            source_cache: HashMap::new(),
            dirty_regions: Vec::new(),
            prefetch: PrefetchEngine::new(prefetch_config, max_pos),
            use_parallel: false,
        }
    }

    // ─── Dirty-region tracking API ────────────────────────────────────────────

    /// Mark a frame range `[start_frame, end_frame)` as dirty (needs re-render).
    ///
    /// Overlapping or adjacent regions are coalesced automatically.
    pub fn mark_dirty(&mut self, start_frame: u64, end_frame: u64) {
        self.dirty_regions
            .push(DirtyRegion::new(start_frame, end_frame));
        self.coalesce_dirty();
    }

    /// Clear all dirty regions (mark entire timeline as clean).
    pub fn clear_dirty(&mut self) {
        self.dirty_regions.clear();
    }

    /// Mark the entire timeline as dirty, forcing a full re-render.
    pub fn force_full_redraw(&mut self) {
        let total = self.timeline.duration.unsigned_abs();
        self.dirty_regions = vec![DirtyRegion::new(0, total.max(1))];
    }

    /// Returns `true` when the given frame position overlaps any dirty region.
    ///
    /// If no dirty regions are tracked (empty list) the frame is always
    /// considered dirty so the renderer behaves as if all regions need updating.
    #[must_use]
    pub fn is_position_dirty(&self, position: i64) -> bool {
        if self.dirty_regions.is_empty() {
            return true;
        }
        let frame = position.unsigned_abs();
        self.dirty_regions.iter().any(|r| r.contains(frame))
    }

    /// Expose the prefetch engine's last known playhead position for tests.
    #[must_use]
    pub fn prefetch_playhead(&self) -> i64 {
        self.prefetch.playhead()
    }

    /// Enable or disable parallel multi-track rendering.
    pub fn set_use_parallel(&mut self, enabled: bool) {
        self.use_parallel = enabled;
    }

    /// Return whether parallel rendering is active.
    #[must_use]
    pub fn use_parallel(&self) -> bool {
        self.use_parallel
    }

    // ── Internal dirty-region coalesce ────────────────────────────────────────

    fn coalesce_dirty(&mut self) {
        if self.dirty_regions.len() <= 1 {
            return;
        }
        self.dirty_regions.sort_by_key(|r| r.start_frame);
        let mut merged: Vec<DirtyRegion> = Vec::with_capacity(self.dirty_regions.len());
        for region in &self.dirty_regions {
            if let Some(last) = merged.last_mut() {
                if last.end_frame >= region.start_frame {
                    last.end_frame = last.end_frame.max(region.end_frame);
                    continue;
                }
            }
            merged.push(*region);
        }
        self.dirty_regions = merged;
    }

    /// Render a frame at a specific timeline position.
    pub async fn render_frame_at(&mut self, position: i64) -> EditResult<RenderFrame> {
        // Check cache first
        if let Some(frame) = self.cache.get(position) {
            return Ok(frame);
        }

        // Collect active clips — clone so we don't hold a borrow on `self.timeline`
        // across the mutable render calls below.
        let clips_at_pos: Vec<(usize, Clip)> = self
            .timeline
            .get_clips_at(position)
            .into_iter()
            .map(|(ti, c)| (ti, c.clone()))
            .collect();
        let clips_refs: Vec<(usize, &Clip)> = clips_at_pos.iter().map(|(ti, c)| (*ti, c)).collect();

        // Render video layers
        let video_frame = if self.config.render_video {
            self.render_video_at(position, &clips_refs).await?
        } else {
            None
        };

        // Render audio
        let audio_frame = if self.config.render_audio {
            self.render_audio_at(position, &clips_refs).await?
        } else {
            None
        };

        let frame = RenderFrame {
            position,
            timestamp: Timestamp::new(position, self.timeline.timebase),
            video: video_frame,
            audio: audio_frame,
        };

        // Cache the frame
        self.cache.put(position, frame.clone());

        // Advance the prefetch engine so it warms the cache for upcoming frames.
        let _ = self.prefetch.update(position);

        Ok(frame)
    }

    /// Render video frame at position.
    async fn render_video_at(
        &mut self,
        position: i64,
        clips: &[(usize, &Clip)],
    ) -> EditResult<Option<VideoFrame>> {
        let video_clips: Vec<(usize, Clip)> = clips
            .iter()
            .filter(|(_, clip)| clip.is_video())
            .map(|(ti, c)| (*ti, (*c).clone()))
            .collect();

        if video_clips.is_empty() {
            return Ok(None);
        }

        // ── Incremental skip: skip entirely if no dirty region overlaps this position ──
        if !self.is_position_dirty(position) {
            return Ok(None);
        }

        let w = self.config.width;
        let h = self.config.height;

        // ── Parallel multi-track path ──────────────────────────────────────────
        if self.use_parallel {
            return self.render_video_at_parallel(position, w, h);
        }

        // Collect active transitions for this position (clone so we don't hold
        // a borrow on `self.timeline` while calling `&mut self` methods later).
        let active_transitions: Vec<(usize, Transition)> = video_clips
            .iter()
            .flat_map(|(track_idx, _)| {
                self.timeline
                    .transitions
                    .get_active_at(*track_idx, position)
                    .into_iter()
                    .map(|t| (*track_idx, t.clone()))
            })
            .collect();

        // Build the set of clip IDs involved in active transitions so we can
        // replace them with a single blended layer.
        // map: clip_a_id → (frame_a, frame_b, transition, progress)
        let mut in_transition_pair: HashMap<u64, (VideoFrame, VideoFrame, Transition, f64)> =
            HashMap::new();
        let mut transitioned_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();

        for (_, transition) in &active_transitions {
            let clip_a_id = transition.clip_a;
            let clip_b_id = transition.clip_b;

            let clip_a = video_clips
                .iter()
                .find(|(_, c)| c.id == clip_a_id)
                .map(|(_, c)| c.clone());
            let clip_b = video_clips
                .iter()
                .find(|(_, c)| c.id == clip_b_id)
                .map(|(_, c)| c.clone());

            if let (Some(ca), Some(cb)) = (clip_a, clip_b) {
                let pos_a = ca.timeline_to_source(position);
                let pos_b = cb.timeline_to_source(position);
                let frame_a = self.get_source_frame(&ca, pos_a)?;
                let frame_b = self.get_source_frame(&cb, pos_b)?;
                let progress = transition.progress_at(position);
                transitioned_ids.insert(clip_a_id);
                transitioned_ids.insert(clip_b_id);
                // Only store once per transition (first time we see clip_a).
                in_transition_pair.entry(clip_a_id).or_insert((
                    frame_a,
                    frame_b,
                    transition.clone(),
                    progress,
                ));
            }
        }

        // Compositor: bottom-to-top (rev() order mirrors layer stack).
        use oximedia_graphics::hdr_composite::HdrCompositor;

        let mut compositor = HdrCompositor::new(w, h, 1000.0);

        // Iterate bottom-to-top.
        for (_, clip) in video_clips.iter().rev() {
            if clip.muted {
                continue;
            }

            let source_pos = clip.timeline_to_source(position);

            if transitioned_ids.contains(&clip.id) {
                // This clip is part of a transition pair.
                if let Some((fa, fb, trans, progress)) = in_transition_pair.remove(&clip.id) {
                    let blended = TransitionRenderer::blend_video(&fa, &fb, &trans, progress);
                    let layer = video_frame_to_hdr_layer(&blended, clip.opacity);
                    compositor.add_layer(layer);
                }
                // Clip B's entry was already consumed with clip A; skip.
                continue;
            }

            let source_frame = self.get_source_frame(clip, source_pos)?;
            let layer = video_frame_to_hdr_layer(&source_frame, clip.opacity);
            compositor.add_layer(layer);
        }

        // Flatten compositor to RGBA f32 → pack into output VideoFrame.
        let rgba_f32 = compositor.composite();
        let mut output = VideoFrame::new(self.config.pixel_format, w, h);
        output.allocate();
        output.timestamp = Timestamp::new(position, self.timeline.timebase);

        // Write RGBA f32 → luma plane (BT.709 coefficients).
        fill_output_frame_from_rgba_f32(&mut output, &rgba_f32, w, h);

        Ok(Some(output))
    }

    /// Parallel variant of `render_video_at`.
    ///
    /// Builds one [`TrackRenderInput`] per video track, fans out to
    /// `render_tracks_parallel`, then composites the RGBA8 layers
    /// bottom-to-top into the output `VideoFrame`.
    fn render_video_at_parallel(
        &mut self,
        position: i64,
        w: u32,
        h: u32,
    ) -> EditResult<Option<VideoFrame>> {
        // Build per-track inputs for every Video track.
        let config_clone = self.config.clone();
        let timeline_clone = self.timeline.clone();

        let inputs: Vec<TrackRenderInput> = timeline_clone
            .tracks
            .iter()
            .filter(|t| !t.muted && matches!(t.track_type, TrackType::Video))
            .map(|track| {
                let active_clips: Vec<ClipWithSource> = track
                    .clips
                    .iter()
                    .filter(|c| c.contains(position) && !c.muted)
                    .map(|c| {
                        let source = self.resolve_source(c);
                        ClipWithSource {
                            clip: c.clone(),
                            source,
                        }
                    })
                    .collect();
                TrackRenderInput::video(track.index, active_clips, position, w, h)
            })
            .collect();

        if inputs.is_empty() {
            return Ok(None);
        }

        // Fan-out: all tracks rendered in parallel.
        let outputs: Vec<TrackRenderOutput> = render_tracks_parallel(&inputs);

        // Composite bottom-to-top: output with the *lowest* track index is rendered
        // first (bottom layer); higher indices sit on top.  `render_tracks_parallel`
        // preserves input order, so outputs[0] corresponds to inputs[0].
        use oximedia_graphics::hdr_composite::HdrCompositor;

        let mut compositor = HdrCompositor::new(w, h, 1000.0);
        for out in outputs.iter().rev() {
            if out.kind != TrackKind::Video || out.video_rgba8.is_empty() {
                continue;
            }
            // Convert RGBA8 → HdrLayer (opacity = 1.0 per track; per-clip opacity
            // was already applied inside render_track_frame_stateless).
            use oximedia_graphics::hdr_composite::HdrLayer;
            let pixel_count = (w as usize) * (h as usize);
            let mut layer = HdrLayer::new(w, h);
            layer.opacity = 1.0;
            for i in 0..pixel_count {
                let base = i * 4;
                if base + 3 < out.video_rgba8.len() {
                    layer.pixels[base] = out.video_rgba8[base] as f32 / 255.0;
                    layer.pixels[base + 1] = out.video_rgba8[base + 1] as f32 / 255.0;
                    layer.pixels[base + 2] = out.video_rgba8[base + 2] as f32 / 255.0;
                    layer.pixels[base + 3] = out.video_rgba8[base + 3] as f32 / 255.0;
                }
            }
            compositor.add_layer(layer);
        }

        let rgba_f32 = compositor.composite();
        let mut output = VideoFrame::new(config_clone.pixel_format, w, h);
        output.allocate();
        output.timestamp = Timestamp::new(position, self.timeline.timebase);
        fill_output_frame_from_rgba_f32(&mut output, &rgba_f32, w, h);

        Ok(Some(output))
    }

    /// Render audio frame at position.
    async fn render_audio_at(
        &mut self,
        position: i64,
        clips: &[(usize, &Clip)],
    ) -> EditResult<Option<AudioFrame>> {
        let audio_clips: Vec<(usize, Clip)> = clips
            .iter()
            .filter(|(_, clip)| clip.is_audio())
            .map(|(ti, c)| (*ti, (*c).clone()))
            .collect();

        if audio_clips.is_empty() {
            return Ok(None);
        }

        let ch_count = self.config.channels.count();
        // Produce 1024 samples per render call (approx. 21 ms at 48 kHz).
        let num_samples: usize = 1024;
        let mut mix_buf = vec![0.0_f32; num_samples * ch_count];

        // Collect active CrossFade transitions (clone so we don't hold a borrow
        // on `self.timeline` while calling `&mut self` decode methods).
        use crate::transition::TransitionType;

        let crossfade_transitions: Vec<(usize, Transition)> = audio_clips
            .iter()
            .flat_map(|(track_idx, _)| {
                self.timeline
                    .transitions
                    .get_active_at(*track_idx, position)
                    .into_iter()
                    .filter(|t| matches!(t.transition_type, TransitionType::CrossFade))
                    .map(|t| (*track_idx, t.clone()))
            })
            .collect();

        // Identify audio clips involved in CrossFade transitions.
        let mut crossfade_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut crossfade_pairs: Vec<(AudioFrame, AudioFrame, Transition, f64)> = Vec::new();

        for (_, transition) in &crossfade_transitions {
            let ca_id = transition.clip_a;
            let cb_id = transition.clip_b;

            let ca = audio_clips
                .iter()
                .find(|(_, c)| c.id == ca_id)
                .map(|(_, c)| (*c).clone());
            let cb = audio_clips
                .iter()
                .find(|(_, c)| c.id == cb_id)
                .map(|(_, c)| (*c).clone());
            if let (Some(ca), Some(cb)) = (ca, cb) {
                let fa = self.get_source_audio_frame(&ca, ca.timeline_to_source(position))?;
                let fb = self.get_source_audio_frame(&cb, cb.timeline_to_source(position))?;
                let progress = transition.progress_at(position);
                crossfade_ids.insert(ca_id);
                crossfade_ids.insert(cb_id);
                crossfade_pairs.push((fa, fb, transition.clone(), progress));
            }
        }

        // Mix crossfade pairs.
        for (fa, fb, trans, progress) in &crossfade_pairs {
            let blended = TransitionRenderer::mix_audio(fa, fb, trans, *progress);
            accumulate_audio_frame_into(&mut mix_buf, &blended, 1.0, ch_count, num_samples);
        }

        // Mix non-transitioned clips.
        for (_, clip) in audio_clips.iter() {
            if clip.muted || crossfade_ids.contains(&clip.id) {
                continue;
            }
            let source_pos = clip.timeline_to_source(position);
            let src_frame = self.get_source_audio_frame(clip, source_pos)?;
            let gain = clip.opacity; // opacity == volume for audio clips
            accumulate_audio_frame_into(&mut mix_buf, &src_frame, gain, ch_count, num_samples);
        }

        // Clamp mix to [-1, 1].
        for s in &mut mix_buf {
            *s = s.clamp(-1.0, 1.0);
        }

        // Pack into output AudioFrame.
        let bytes: Vec<u8> = mix_buf.iter().flat_map(|s| s.to_ne_bytes()).collect();

        let mut output = AudioFrame {
            format: self.config.sample_format,
            sample_rate: self.config.sample_rate,
            channels: self.config.channels.clone(),
            samples: oximedia_audio::AudioBuffer::Interleaved(Bytes::from(bytes)),
            timestamp: Timestamp::new(position, self.timeline.timebase),
        };
        output.timestamp = Timestamp::new(position, self.timeline.timebase);

        Ok(Some(output))
    }

    /// Resolve the `RenderSource` for a clip, using the source cache.
    fn resolve_source(&mut self, clip: &Clip) -> Arc<RenderSource> {
        match &clip.source {
            None => Arc::new(RenderSource::TestPattern),
            Some(path) => {
                if let Some(cached) = self.source_cache.get(path) {
                    return cached.clone();
                }
                let resolved = RenderSource::from_path(path)
                    .unwrap_or_else(|_| Arc::new(RenderSource::TestPattern));
                self.source_cache.insert(path.clone(), resolved.clone());
                resolved
            }
        }
    }

    /// Get a decoded video frame for `clip` at `source_pts`, using the raw
    /// frame byte cache to avoid redundant decoding.
    fn get_source_frame(&mut self, clip: &Clip, source_pts: i64) -> EditResult<VideoFrame> {
        let w = self.config.width;
        let h = self.config.height;

        // Compound cache key: high 32 bits = clip id (truncated), low 32 bits = pts hash.
        // Using wrapping arithmetic to avoid overflow; collisions are acceptable for a
        // preview cache (a cache miss just re-decodes).
        let key = (clip.id.wrapping_mul(0x9e3779b9)).wrapping_add(source_pts.unsigned_abs())
            ^ (source_pts.signum() as u64);

        let source = self.resolve_source(clip);

        // Borrow-checker dance: we need to call get_or_render, which needs
        // `source` to be captured in the closure but also needs `&mut self.raw_frame_cache`.
        // We clone the Arc so the closure doesn't borrow `self`.
        let source_clone = source.clone();
        let rgba8 = self
            .raw_frame_cache
            .get_or_render(key, || source_clone.sample_video(source_pts, w, h))
            .to_vec();

        let mut frame = VideoFrame::new(self.config.pixel_format, w, h);
        frame.allocate();
        fill_output_frame_from_rgba8(&mut frame, &rgba8, w, h);
        Ok(frame)
    }

    /// Get decoded audio samples for `clip` at `source_pts`.
    fn get_source_audio_frame(&mut self, clip: &Clip, source_pts: i64) -> EditResult<AudioFrame> {
        let ch_count = self.config.channels.count() as u16;
        let sample_rate = self.config.sample_rate;
        let num_samples: usize = 1024;

        let source = self.resolve_source(clip);
        let samples = source.sample_audio(source_pts, num_samples, ch_count, sample_rate);

        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_ne_bytes()).collect();

        Ok(AudioFrame {
            format: self.config.sample_format,
            sample_rate,
            channels: self.config.channels.clone(),
            samples: oximedia_audio::AudioBuffer::Interleaved(Bytes::from(bytes)),
            timestamp: Timestamp::new(source_pts, self.timeline.timebase),
        })
    }

    /// Start background rendering.
    pub fn start_background_render(&mut self) -> BackgroundRenderer {
        BackgroundRenderer::new(self.timeline.clone(), self.config.clone())
    }

    /// Clear frame cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

// ─── Render pipeline helper functions ─────────────────────────────────────────

/// Convert a decoded `VideoFrame` (any format, already allocated) into an
/// [`HdrLayer`] for the HDR compositor.
///
/// The frame data is interpreted as RGBA8 when all planes are concatenated.
/// For planar YUV formats the luma plane is broadcast to R, G, B channels.
fn video_frame_to_hdr_layer(
    frame: &VideoFrame,
    opacity: f32,
) -> oximedia_graphics::hdr_composite::HdrLayer {
    use oximedia_graphics::hdr_composite::HdrLayer;

    let w = frame.width;
    let h = frame.height;
    let pixel_count = (w as usize) * (h as usize);
    let mut layer = HdrLayer::new(w, h);
    layer.opacity = opacity.clamp(0.0, 1.0);

    // Best-effort pixel extraction: use the luma plane (plane 0) for all
    // colour channels when no better interleaved data is available.
    if let Some(plane) = frame.planes.first() {
        for i in 0..pixel_count {
            let idx = i * 4;
            let luma = if i < plane.data.len() {
                plane.data[i] as f32 / 255.0
            } else {
                0.0_f32
            };
            layer.pixels[idx] = luma;
            layer.pixels[idx + 1] = luma;
            layer.pixels[idx + 2] = luma;
            layer.pixels[idx + 3] = 1.0;
        }
    }

    layer
}

/// Write RGBA f32 linear-light compositor output into the first plane of a
/// `VideoFrame`.  Values are tone-mapped to `[0, 255]` (divide by peak_nits,
/// clamp, scale).
fn fill_output_frame_from_rgba_f32(frame: &mut VideoFrame, rgba: &[f32], w: u32, h: u32) {
    let pixel_count = (w as usize) * (h as usize);
    if let Some(plane) = frame.planes.first_mut() {
        let out_len = plane.data.len().min(pixel_count);
        for i in 0..out_len {
            let base = i * 4;
            if base + 2 < rgba.len() {
                // Luma: 0.2126*R + 0.7152*G + 0.0722*B  (BT.709)
                let luma =
                    (0.2126 * rgba[base] + 0.7152 * rgba[base + 1] + 0.0722 * rgba[base + 2])
                        .clamp(0.0, 1.0);
                #[allow(clippy::cast_possible_truncation)]
                #[allow(clippy::cast_sign_loss)]
                let y = (luma * 255.0).round() as u8;
                plane.data[i] = y;
            }
        }
    }
}

/// Write RGBA8 pixel bytes into the first plane of a `VideoFrame` as luma.
fn fill_output_frame_from_rgba8(frame: &mut VideoFrame, rgba8: &[u8], w: u32, h: u32) {
    let pixel_count = (w as usize) * (h as usize);
    if let Some(plane) = frame.planes.first_mut() {
        let out_len = plane.data.len().min(pixel_count);
        for i in 0..out_len {
            let base = i * 4;
            if base + 2 < rgba8.len() {
                // Luma (integer approximation of BT.601).
                let r = rgba8[base] as u32;
                let g = rgba8[base + 1] as u32;
                let b = rgba8[base + 2] as u32;
                let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                plane.data[i] = y.min(255) as u8;
            }
        }
    }
}

/// Accumulate samples from an `AudioFrame` (F32 interleaved) into `mix_buf`
/// with the given `gain`.
fn accumulate_audio_frame_into(
    mix_buf: &mut [f32],
    frame: &AudioFrame,
    gain: f32,
    ch_count: usize,
    num_samples: usize,
) {
    use oximedia_mixer::simd_audio::mix_and_gain_simd;

    let expected = num_samples * ch_count;
    let src_samples: Vec<f32> = match &frame.samples {
        oximedia_audio::AudioBuffer::Interleaved(bytes) => bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .take(expected)
            .collect(),
        oximedia_audio::AudioBuffer::Planar(planes) => {
            // Interleave planes.
            let frames_per_plane = if planes.is_empty() {
                0
            } else {
                (planes[0].len() / 4).min(num_samples)
            };
            let mut interleaved = vec![0.0_f32; frames_per_plane * ch_count];
            for (c, plane) in planes.iter().enumerate().take(ch_count) {
                for f in 0..frames_per_plane {
                    let base = f * 4;
                    if base + 3 < plane.len() {
                        interleaved[f * ch_count + c] = f32::from_ne_bytes([
                            plane[base],
                            plane[base + 1],
                            plane[base + 2],
                            plane[base + 3],
                        ]);
                    }
                }
            }
            interleaved
        }
    };

    if src_samples.is_empty() {
        return;
    }

    let dst_len = mix_buf.len().min(expected);
    let src_len = src_samples.len().min(dst_len);
    mix_and_gain_simd(&mut mix_buf[..dst_len], &src_samples[..src_len], gain);
}

/// Rendered frame containing video and audio.
#[derive(Clone, Debug)]
pub struct RenderFrame {
    /// Timeline position.
    pub position: i64,
    /// Timestamp.
    pub timestamp: Timestamp,
    /// Video frame.
    pub video: Option<VideoFrame>,
    /// Audio frame.
    pub audio: Option<AudioFrame>,
}

impl RenderFrame {
    /// Check if frame has video.
    #[must_use]
    pub fn has_video(&self) -> bool {
        self.video.is_some()
    }

    /// Check if frame has audio.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }
}

/// Render configuration.
#[derive(Clone, Debug)]
pub struct RenderConfig {
    /// Render video.
    pub render_video: bool,
    /// Render audio.
    pub render_audio: bool,
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
    /// Pixel format.
    pub pixel_format: PixelFormat,
    /// Sample rate.
    pub sample_rate: u32,
    /// Sample format.
    pub sample_format: SampleFormat,
    /// Audio channels.
    pub channels: ChannelLayout,
    /// Frame cache size.
    pub cache_size: usize,
    /// Number of render threads.
    pub num_threads: usize,
    /// Quality preset.
    pub quality: RenderQuality,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            render_video: true,
            render_audio: true,
            width: 1920,
            height: 1080,
            pixel_format: PixelFormat::Yuv420p,
            sample_rate: 48000,
            sample_format: SampleFormat::F32,
            channels: ChannelLayout::Stereo,
            cache_size: 30,
            num_threads: 4,
            quality: RenderQuality::High,
        }
    }
}

impl RenderConfig {
    /// Create config from timeline config.
    #[must_use]
    pub fn from_timeline_config(config: &TimelineConfig) -> Self {
        Self {
            width: config.width,
            height: config.height,
            sample_rate: config.sample_rate,
            channels: ChannelLayout::from_count(config.channels as usize),
            ..Default::default()
        }
    }
}

/// Render quality preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderQuality {
    /// Draft quality (fast, low quality).
    Draft,
    /// Preview quality (balanced).
    Preview,
    /// High quality (slow, high quality).
    High,
    /// Maximum quality (very slow, maximum quality).
    Maximum,
}

impl RenderQuality {
    /// Get quality factor (0.0 to 1.0).
    #[must_use]
    pub fn factor(&self) -> f32 {
        match self {
            Self::Draft => 0.25,
            Self::Preview => 0.5,
            Self::High => 0.75,
            Self::Maximum => 1.0,
        }
    }
}

/// Frame cache for rendered frames.
#[derive(Debug)]
struct FrameCache {
    /// Cache storage.
    frames: VecDeque<(i64, RenderFrame)>,
    /// Maximum cache size.
    capacity: usize,
}

impl FrameCache {
    /// Create a new frame cache.
    fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Get a frame from cache.
    fn get(&self, position: i64) -> Option<RenderFrame> {
        self.frames
            .iter()
            .find(|(pos, _)| *pos == position)
            .map(|(_, frame)| frame.clone())
    }

    /// Put a frame into cache.
    fn put(&mut self, position: i64, frame: RenderFrame) {
        // Remove oldest if at capacity
        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back((position, frame));
    }

    /// Clear the cache.
    fn clear(&mut self) {
        self.frames.clear();
    }
}

/// Background renderer for non-blocking rendering.
#[cfg(not(target_arch = "wasm32"))]
pub struct BackgroundRenderer {
    /// Timeline to render.
    timeline: Arc<Timeline>,
    /// Render configuration.
    config: RenderConfig,
    /// Render task handle.
    handle: Option<tokio::task::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl BackgroundRenderer {
    /// Create a new background renderer.
    #[must_use]
    pub fn new(timeline: Arc<Timeline>, config: RenderConfig) -> Self {
        Self {
            timeline,
            config,
            handle: None,
        }
    }

    /// Start rendering in the background.
    pub fn start(&mut self, start: i64, end: i64) -> mpsc::Receiver<RenderFrame> {
        let (tx, rx) = mpsc::channel(100);
        let timeline = self.timeline.clone();
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            let mut renderer = TimelineRenderer::new(timeline, config);

            for position in start..end {
                match renderer.render_frame_at(position).await {
                    Ok(frame) => {
                        if tx.send(frame).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        self.handle = Some(handle);
        rx
    }

    /// Stop background rendering.
    pub async fn stop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
            let _ = handle.await;
        }
    }

    /// Check if rendering is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.handle
            .as_ref()
            .map_or(true, tokio::task::JoinHandle::is_finished)
    }
}

/// Real-time preview renderer.
pub struct PreviewRenderer {
    /// Timeline renderer.
    renderer: TimelineRenderer,
    /// Target frame rate.
    frame_rate: Rational,
    /// Current position.
    position: i64,
    /// Playing state.
    playing: bool,
}

impl PreviewRenderer {
    /// Create a new preview renderer.
    #[must_use]
    pub fn new(timeline: Arc<Timeline>, config: RenderConfig) -> Self {
        let frame_rate = timeline.frame_rate;
        Self {
            renderer: TimelineRenderer::new(timeline, config),
            frame_rate,
            position: 0,
            playing: false,
        }
    }

    /// Start playback.
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stop playback and reset.
    pub fn stop(&mut self) {
        self.playing = false;
        self.position = 0;
    }

    /// Get next preview frame.
    pub async fn next_frame(&mut self) -> EditResult<Option<RenderFrame>> {
        if !self.playing {
            return Ok(None);
        }

        let frame = self.renderer.render_frame_at(self.position).await?;

        // Advance position
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_precision_loss)]
        let frame_duration = (1000.0 / self.frame_rate.to_f64()) as i64;
        self.position += frame_duration;

        // Check if we've reached the end
        if self.position >= self.renderer.timeline.duration {
            self.stop();
        }

        Ok(Some(frame))
    }

    /// Seek to position.
    pub fn seek(&mut self, position: i64) {
        self.position = position.clamp(0, self.renderer.timeline.duration);
    }

    /// Get current position.
    #[must_use]
    pub fn position(&self) -> i64 {
        self.position
    }

    /// Check if playing.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.playing
    }
}

/// Export renderer for final output.
pub struct ExportRenderer {
    /// Timeline renderer.
    renderer: TimelineRenderer,
    /// Export settings.
    settings: ExportSettings,
}

impl ExportRenderer {
    /// Create a new export renderer.
    #[must_use]
    pub fn new(timeline: Arc<Timeline>, settings: ExportSettings) -> Self {
        let config = RenderConfig {
            render_video: settings.video_enabled,
            render_audio: settings.audio_enabled,
            width: settings.width,
            height: settings.height,
            pixel_format: settings.pixel_format,
            sample_rate: settings.sample_rate,
            sample_format: settings.sample_format,
            channels: settings.channels.clone(),
            quality: settings.quality,
            ..Default::default()
        };

        Self {
            renderer: TimelineRenderer::new(timeline, config),
            settings,
        }
    }

    /// Export timeline to frames.
    pub async fn export(&mut self) -> EditResult<Vec<RenderFrame>> {
        let mut frames = Vec::new();
        let start = self.settings.start.unwrap_or(0);
        let end = self.settings.end.unwrap_or(self.renderer.timeline.duration);

        for position in start..end {
            let frame = self.renderer.render_frame_at(position).await?;
            frames.push(frame);
        }

        Ok(frames)
    }

    /// Export timeline as a stream.
    pub fn export_stream(&mut self) -> ExportStream {
        let start = self.settings.start.unwrap_or(0);
        let end = self.settings.end.unwrap_or(self.renderer.timeline.duration);

        ExportStream {
            renderer: self.renderer.clone_for_stream(),
            current: start,
            end,
        }
    }
}

/// Export settings.
#[derive(Clone, Debug)]
pub struct ExportSettings {
    /// Export video.
    pub video_enabled: bool,
    /// Export audio.
    pub audio_enabled: bool,
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
    /// Pixel format.
    pub pixel_format: PixelFormat,
    /// Sample rate.
    pub sample_rate: u32,
    /// Sample format.
    pub sample_format: SampleFormat,
    /// Audio channels.
    pub channels: ChannelLayout,
    /// Quality preset.
    pub quality: RenderQuality,
    /// Start position (None = beginning).
    pub start: Option<i64>,
    /// End position (None = end of timeline).
    pub end: Option<i64>,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            video_enabled: true,
            audio_enabled: true,
            width: 1920,
            height: 1080,
            pixel_format: PixelFormat::Yuv420p,
            sample_rate: 48000,
            sample_format: SampleFormat::F32,
            channels: ChannelLayout::Stereo,
            quality: RenderQuality::High,
            start: None,
            end: None,
        }
    }
}

/// Stream of exported frames.
pub struct ExportStream {
    renderer: TimelineRenderer,
    current: i64,
    end: i64,
}

// Note: This is a stub implementation. The actual Stream trait requires proper
// async support with a stored future, not creating a new future in poll_next.
// Consider using `tokio::stream::StreamExt` or `futures::stream::unfold` for proper implementation.
#[allow(dead_code)]
impl ExportStream {
    /// Create an async stream from the export stream.
    /// This should be used instead of directly implementing Stream.
    pub fn into_stream(self) -> impl futures::stream::Stream<Item = EditResult<RenderFrame>> {
        futures::stream::unfold(self, |mut state| async move {
            if state.current >= state.end {
                return None;
            }
            let position = state.current;
            state.current += 1;
            let result = state.renderer.render_frame_at(position).await;
            Some((result, state))
        })
    }
}

impl TimelineRenderer {
    /// Clone renderer for streaming.
    fn clone_for_stream(&self) -> Self {
        let max_pos = self.timeline.duration.max(0) as i64;
        let prefetch_config = PrefetchConfig::for_playback(30.0, 1.0);
        Self {
            timeline: self.timeline.clone(),
            config: self.config.clone(),
            cache: FrameCache::new(self.config.cache_size),
            raw_frame_cache: RawFrameCache::new(RAW_FRAME_CACHE_CAPACITY),
            source_cache: HashMap::new(),
            dirty_regions: Vec::new(),
            prefetch: PrefetchEngine::new(prefetch_config, max_pos),
            use_parallel: self.use_parallel,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RawFrameCache — byte-buffer frame cache with LRU eviction
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of raw frames kept in [`RawFrameCache`] before LRU eviction.
pub const RAW_FRAME_CACHE_CAPACITY: usize = 32;

/// A cache that stores raw pixel byte buffers (e.g. decoded video frames) keyed
/// by frame number.
///
/// When the cache reaches [`RAW_FRAME_CACHE_CAPACITY`] entries the least-recently
/// used frame is evicted before inserting the new one.
///
/// # Example
///
/// ```
/// use oximedia_edit::render::RawFrameCache;
///
/// let mut cache = RawFrameCache::new(4);
/// let data = cache.get_or_render(0, || vec![0u8; 1024]);
/// assert_eq!(data.len(), 1024);
/// ```
pub struct RawFrameCache {
    /// Frame data keyed by frame number.
    store: HashMap<u64, Vec<u8>>,
    /// Insertion-order tracking for LRU eviction (front = oldest).
    order: VecDeque<u64>,
    /// Maximum number of frames to retain.
    capacity: usize,
}

impl RawFrameCache {
    /// Create a new cache with the given capacity (clamped to at least 1).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            store: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Return a reference to the cached bytes for `frame_num`, rendering and
    /// inserting them via `render_fn` if not already present.
    ///
    /// The rendered bytes are stored in the cache; on subsequent calls the same
    /// reference is returned without invoking `render_fn`.
    ///
    /// When the cache is full the **oldest** frame is evicted first (LRU by
    /// insertion order).
    pub fn get_or_render(&mut self, frame_num: u64, render_fn: impl FnOnce() -> Vec<u8>) -> &[u8] {
        if !self.store.contains_key(&frame_num) {
            // Evict oldest entry if at capacity.
            if self.store.len() >= self.capacity {
                if let Some(oldest) = self.order.pop_front() {
                    self.store.remove(&oldest);
                }
            }

            let data = render_fn();
            self.store.insert(frame_num, data);
            self.order.push_back(frame_num);
        }

        // Safety: key is guaranteed to be present after the block above.
        self.store.get(&frame_num).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Return a reference to the cached bytes for `frame_num` without rendering.
    ///
    /// Returns `None` if the frame is not in the cache.
    #[must_use]
    pub fn get(&self, frame_num: u64) -> Option<&[u8]> {
        self.store.get(&frame_num).map(Vec::as_slice)
    }

    /// Explicitly insert pre-rendered bytes for `frame_num`.
    ///
    /// If the frame already exists it is replaced.  If the cache is full the
    /// oldest frame is evicted.
    pub fn insert(&mut self, frame_num: u64, data: Vec<u8>) {
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.store.entry(frame_num) {
            e.insert(data);
            return;
        }
        if self.store.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.store.remove(&oldest);
            }
        }
        self.store.insert(frame_num, data);
        self.order.push_back(frame_num);
    }

    /// Invalidate (remove) the cache entry for `frame_num`, if present.
    pub fn invalidate(&mut self, frame_num: u64) {
        if self.store.remove(&frame_num).is_some() {
            self.order.retain(|&f| f != frame_num);
        }
    }

    /// Clear all cached frames.
    pub fn clear(&mut self) {
        self.store.clear();
        self.order.clear();
    }

    /// Return the number of frames currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Return `true` if the cache contains no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Return the maximum number of frames this cache holds.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return `true` if a frame for `frame_num` is present in the cache.
    #[must_use]
    pub fn contains(&self, frame_num: u64) -> bool {
        self.store.contains_key(&frame_num)
    }
}

impl std::fmt::Debug for RawFrameCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawFrameCache")
            .field("len", &self.store.len())
            .field("capacity", &self.capacity)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests for RawFrameCache
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod raw_cache_tests {
    use super::{RawFrameCache, RAW_FRAME_CACHE_CAPACITY};

    #[test]
    fn test_raw_frame_cache_basic_get_or_render() {
        let mut cache = RawFrameCache::new(4);
        let mut render_count = 0usize;

        // First call: renders.
        let data = cache.get_or_render(0, || {
            render_count += 1;
            vec![1u8, 2, 3]
        });
        assert_eq!(data, &[1u8, 2, 3]);
        assert_eq!(render_count, 1);

        // Second call: cached — render_fn must not be invoked.
        let data2 = cache.get_or_render(0, || {
            render_count += 1;
            vec![99u8]
        });
        assert_eq!(data2, &[1u8, 2, 3]);
        assert_eq!(render_count, 1, "render_fn should not be called twice");
    }

    #[test]
    fn test_raw_frame_cache_lru_eviction() {
        let mut cache = RawFrameCache::new(4);

        // Fill cache to capacity.
        for i in 0u64..4 {
            cache.get_or_render(i, || vec![i as u8]);
        }
        assert_eq!(cache.len(), 4);

        // Insert one more frame — oldest (frame 0) should be evicted.
        cache.get_or_render(4, || vec![4u8]);
        assert_eq!(cache.len(), 4, "cache must not exceed capacity");
        assert!(
            !cache.contains(0),
            "oldest frame (0) should have been evicted"
        );
        assert!(
            cache.contains(4),
            "newly inserted frame (4) should be present"
        );
    }

    #[test]
    fn test_raw_frame_cache_capacity_32() {
        let cache = RawFrameCache::new(RAW_FRAME_CACHE_CAPACITY);
        assert_eq!(cache.capacity(), 32);
    }

    #[test]
    fn test_raw_frame_cache_get_missing() {
        let cache = RawFrameCache::new(4);
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_raw_frame_cache_insert_and_get() {
        let mut cache = RawFrameCache::new(4);
        cache.insert(7, vec![10, 20, 30]);
        assert_eq!(cache.get(7), Some(&[10u8, 20, 30][..]));
    }

    #[test]
    fn test_raw_frame_cache_invalidate() {
        let mut cache = RawFrameCache::new(4);
        cache.insert(1, vec![1, 2]);
        cache.invalidate(1);
        assert!(!cache.contains(1));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_raw_frame_cache_clear() {
        let mut cache = RawFrameCache::new(4);
        for i in 0u64..4 {
            cache.insert(i, vec![i as u8]);
        }
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_raw_frame_cache_eviction_order() {
        let mut cache = RawFrameCache::new(3);
        cache.insert(10, vec![10]);
        cache.insert(20, vec![20]);
        cache.insert(30, vec![30]);

        // Frame 40 → evicts frame 10 (oldest).
        cache.insert(40, vec![40]);
        assert!(!cache.contains(10));
        assert!(cache.contains(20));
        assert!(cache.contains(30));
        assert!(cache.contains(40));

        // Frame 50 → evicts frame 20.
        cache.insert(50, vec![50]);
        assert!(!cache.contains(20));
        assert!(cache.contains(30));
        assert!(cache.contains(40));
        assert!(cache.contains(50));
    }

    #[test]
    fn test_raw_frame_cache_capacity_clamped_to_one() {
        let cache = RawFrameCache::new(0);
        assert_eq!(cache.capacity(), 1);
    }

    #[test]
    fn test_raw_frame_cache_is_empty_initially() {
        let cache = RawFrameCache::new(8);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_raw_frame_cache_debug_format() {
        let cache = RawFrameCache::new(4);
        let debug = format!("{cache:?}");
        assert!(debug.contains("RawFrameCache"), "debug output: {debug}");
    }
}

// Note: Unit tests for dirty-region tracking, prefetch wiring, and parallel
// rendering are in tests/incremental_render.rs (integration tests) and
// tests/renderer_unit.rs (synchronous unit tests).

/// Transition renderer helper.
pub struct TransitionRenderer;

impl TransitionRenderer {
    /// Blend two video frames based on transition progress.
    ///
    /// `progress` ranges from `0.0` (fully `frame_a`) to `1.0` (fully `frame_b`).
    /// When the two frames have different dimensions the larger frame is returned
    /// unchanged.  When formats differ `frame_a` is returned unchanged.
    #[must_use]
    pub fn blend_video(
        frame_a: &VideoFrame,
        frame_b: &VideoFrame,
        transition: &Transition,
        progress: f64,
    ) -> VideoFrame {
        use crate::transition::TransitionType;

        // Dimension / format mismatch — return the larger (or frame_a) unblended.
        if frame_a.format != frame_b.format {
            return frame_a.clone();
        }
        let a_pixels = frame_a.width as u64 * frame_a.height as u64;
        let b_pixels = frame_b.width as u64 * frame_b.height as u64;
        if a_pixels != b_pixels {
            return if b_pixels > a_pixels {
                frame_b.clone()
            } else {
                frame_a.clone()
            };
        }

        let progress_f32 = progress as f32;

        match &transition.transition_type {
            TransitionType::Dissolve => Self::dissolve_video(frame_a, frame_b, progress_f32),
            TransitionType::WipeLeft => {
                Self::wipe_video(frame_a, frame_b, progress_f32, WipeDirection::Left)
            }
            TransitionType::WipeRight => {
                Self::wipe_video(frame_a, frame_b, progress_f32, WipeDirection::Right)
            }
            TransitionType::WipeDown => {
                Self::wipe_video(frame_a, frame_b, progress_f32, WipeDirection::Down)
            }
            TransitionType::WipeUp => {
                Self::wipe_video(frame_a, frame_b, progress_f32, WipeDirection::Up)
            }
            // CrossFade is audio-only; all remaining video variants and Cut-like
            // behaviour: switch at mid-point.
            _ => {
                if progress_f32 >= 0.5 {
                    frame_b.clone()
                } else {
                    frame_a.clone()
                }
            }
        }
    }

    /// Mix two audio frames based on transition progress (cross-fade).
    ///
    /// `progress` ranges from `0.0` (fully `frame_a`) to `1.0` (fully `frame_b`).
    /// `F32` interleaved and `F32p` planar audio are blended; all other formats
    /// fall back to returning `frame_a` unchanged.  When sample formats differ,
    /// `frame_a` is returned unchanged.
    #[must_use]
    pub fn mix_audio(
        frame_a: &AudioFrame,
        frame_b: &AudioFrame,
        _transition: &Transition,
        progress: f64,
    ) -> AudioFrame {
        // Format mismatch — return frame_a unblended.
        if frame_a.format != frame_b.format {
            return frame_a.clone();
        }

        let alpha = progress as f32;
        let inv_alpha = 1.0_f32 - alpha;

        match (&frame_a.samples, &frame_b.samples) {
            (AudioBuffer::Interleaved(a_bytes), AudioBuffer::Interleaved(b_bytes))
                if frame_a.format == SampleFormat::F32 =>
            {
                let len_samples = (a_bytes.len() / 4).min(b_bytes.len() / 4);
                let mut out_bytes = Vec::with_capacity(len_samples * 4);

                for i in 0..len_samples {
                    let base = i * 4;
                    let a_val = f32::from_ne_bytes([
                        a_bytes[base],
                        a_bytes[base + 1],
                        a_bytes[base + 2],
                        a_bytes[base + 3],
                    ]);
                    let b_val = f32::from_ne_bytes([
                        b_bytes[base],
                        b_bytes[base + 1],
                        b_bytes[base + 2],
                        b_bytes[base + 3],
                    ]);
                    let blended = (a_val * inv_alpha + b_val * alpha).clamp(-1.0, 1.0);
                    out_bytes.extend_from_slice(&blended.to_ne_bytes());
                }

                AudioFrame {
                    format: frame_a.format,
                    sample_rate: frame_a.sample_rate,
                    channels: frame_a.channels.clone(),
                    samples: AudioBuffer::Interleaved(Bytes::from(out_bytes)),
                    timestamp: frame_a.timestamp,
                }
            }
            (AudioBuffer::Planar(a_planes), AudioBuffer::Planar(b_planes))
                if frame_a.format == SampleFormat::F32p =>
            {
                let plane_count = a_planes.len().min(b_planes.len());
                let mut out_planes = Vec::with_capacity(plane_count);

                for p in 0..plane_count {
                    let a_plane = &a_planes[p];
                    let b_plane = &b_planes[p];
                    let len_samples = (a_plane.len() / 4).min(b_plane.len() / 4);
                    let mut plane_bytes = Vec::with_capacity(len_samples * 4);

                    for i in 0..len_samples {
                        let base = i * 4;
                        let a_val = f32::from_ne_bytes([
                            a_plane[base],
                            a_plane[base + 1],
                            a_plane[base + 2],
                            a_plane[base + 3],
                        ]);
                        let b_val = f32::from_ne_bytes([
                            b_plane[base],
                            b_plane[base + 1],
                            b_plane[base + 2],
                            b_plane[base + 3],
                        ]);
                        let blended = (a_val * inv_alpha + b_val * alpha).clamp(-1.0, 1.0);
                        plane_bytes.extend_from_slice(&blended.to_ne_bytes());
                    }

                    out_planes.push(Bytes::from(plane_bytes));
                }

                AudioFrame {
                    format: frame_a.format,
                    sample_rate: frame_a.sample_rate,
                    channels: frame_a.channels.clone(),
                    samples: AudioBuffer::Planar(out_planes),
                    timestamp: frame_a.timestamp,
                }
            }
            // Unsupported format combination — return frame_a unchanged.
            _ => frame_a.clone(),
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Linear dissolve blend over all planes.
    fn dissolve_video(frame_a: &VideoFrame, frame_b: &VideoFrame, progress: f32) -> VideoFrame {
        use oximedia_codec::frame::Plane;

        let inv = 1.0_f32 - progress;
        let mut output = frame_a.clone();

        for (out_plane, b_plane) in output.planes.iter_mut().zip(frame_b.planes.iter()) {
            let len = out_plane.data.len().min(b_plane.data.len());
            let blended: Vec<u8> = (0..len)
                .map(|i| {
                    #[allow(clippy::cast_possible_truncation)]
                    #[allow(clippy::cast_sign_loss)]
                    let v = (out_plane.data[i] as f32 * inv + b_plane.data[i] as f32 * progress)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    v
                })
                .collect();

            // Rebuild the plane so that stride/width/height are preserved and
            // only the pixel data is replaced.
            let new_plane = Plane::with_dimensions(
                blended,
                out_plane.stride,
                out_plane.width,
                out_plane.height,
            );
            *out_plane = new_plane;
        }

        output
    }

    /// Wipe transition — one side uses `frame_b`, the other `frame_a`.
    fn wipe_video(
        frame_a: &VideoFrame,
        frame_b: &VideoFrame,
        progress: f32,
        direction: WipeDirection,
    ) -> VideoFrame {
        use oximedia_codec::frame::Plane;

        let mut output = frame_a.clone();

        // Process plane by plane so that chroma subsampling is handled correctly.
        for (out_plane, b_plane) in output.planes.iter_mut().zip(frame_b.planes.iter()) {
            let pw = out_plane.width as usize;
            let ph = out_plane.height as usize;
            // Chroma planes have reduced spatial size; compute the wipe boundary
            // in plane-local coordinates by using the plane's own dimensions.

            let mut new_data = out_plane.data.clone();

            match direction {
                WipeDirection::Left | WipeDirection::Right => {
                    // Number of columns (in this plane) that show frame_b.
                    #[allow(clippy::cast_possible_truncation)]
                    #[allow(clippy::cast_sign_loss)]
                    let boundary = (progress * pw as f32).round() as usize;
                    for y in 0..ph {
                        for x in 0..pw {
                            let use_b = match direction {
                                WipeDirection::Left => x < boundary,
                                WipeDirection::Right => x >= pw.saturating_sub(boundary),
                                _ => false,
                            };
                            if use_b {
                                let src_idx = y * b_plane.stride + x;
                                let dst_idx = y * out_plane.stride + x;
                                if src_idx < b_plane.data.len() && dst_idx < new_data.len() {
                                    new_data[dst_idx] = b_plane.data[src_idx];
                                }
                            }
                        }
                    }
                }
                WipeDirection::Down | WipeDirection::Up => {
                    #[allow(clippy::cast_possible_truncation)]
                    #[allow(clippy::cast_sign_loss)]
                    let boundary = (progress * ph as f32).round() as usize;
                    for y in 0..ph {
                        let use_b = match direction {
                            WipeDirection::Down => y < boundary,
                            WipeDirection::Up => y >= ph.saturating_sub(boundary),
                            _ => false,
                        };
                        if use_b {
                            for x in 0..pw {
                                let src_idx = y * b_plane.stride + x;
                                let dst_idx = y * out_plane.stride + x;
                                if src_idx < b_plane.data.len() && dst_idx < new_data.len() {
                                    new_data[dst_idx] = b_plane.data[src_idx];
                                }
                            }
                        }
                    }
                }
            }

            let new_plane = Plane::with_dimensions(
                new_data,
                out_plane.stride,
                out_plane.width,
                out_plane.height,
            );
            *out_plane = new_plane;
        }

        output
    }
}

/// Direction for wipe transitions.
#[derive(Clone, Copy)]
enum WipeDirection {
    Left,
    Right,
    Down,
    Up,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests for TransitionRenderer
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod transition_renderer_tests {
    use super::*;
    use crate::transition::{Transition, TransitionType};
    use bytes::Bytes;
    use oximedia_audio::{AudioBuffer, AudioFrame, ChannelLayout};
    use oximedia_codec::VideoFrame;
    use oximedia_core::{PixelFormat, SampleFormat};

    /// Build a solid-colour YUV420p video frame (all planes set to `value`).
    fn make_video_frame(width: u32, height: u32, value: u8) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();
        for plane in &mut frame.planes {
            for b in &mut plane.data {
                *b = value;
            }
        }
        frame
    }

    /// Build an F32 interleaved audio frame with all samples set to `value`.
    fn make_audio_frame(num_samples: usize, value: f32) -> AudioFrame {
        let bytes: Vec<u8> = (0..num_samples).flat_map(|_| value.to_ne_bytes()).collect();
        AudioFrame {
            format: SampleFormat::F32,
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            samples: AudioBuffer::Interleaved(Bytes::from(bytes)),
            timestamp: oximedia_core::Timestamp::new(0, oximedia_core::Rational::new(1, 48_000)),
        }
    }

    /// Make a minimal `Transition` with the given type (no real clips needed).
    fn make_transition(tt: TransitionType) -> Transition {
        Transition::new(0, tt, 0, 0, 1000, 0, 1)
    }

    // ── blend_video tests ────────────────────────────────────────────────────

    /// Dissolve at progress=0.5 should produce pixels near (100+200)/2 = 150.
    #[test]
    fn test_blend_video_dissolve_mid() {
        let frame_a = make_video_frame(8, 4, 100);
        let frame_b = make_video_frame(8, 4, 200);
        let t = make_transition(TransitionType::Dissolve);

        let out = TransitionRenderer::blend_video(&frame_a, &frame_b, &t, 0.5);

        for plane in &out.planes {
            for &b in &plane.data {
                assert!((i32::from(b) - 150).abs() <= 1, "expected ~150, got {b}");
            }
        }
    }

    /// Dissolve at progress=0.0 should reproduce frame_a exactly.
    #[test]
    fn test_blend_video_dissolve_zero() {
        let frame_a = make_video_frame(8, 4, 80);
        let frame_b = make_video_frame(8, 4, 200);
        let t = make_transition(TransitionType::Dissolve);

        let out = TransitionRenderer::blend_video(&frame_a, &frame_b, &t, 0.0);

        for plane in &out.planes {
            for &b in &plane.data {
                assert_eq!(b, 80);
            }
        }
    }

    /// Dissolve at progress=1.0 should reproduce frame_b exactly.
    #[test]
    fn test_blend_video_dissolve_one() {
        let frame_a = make_video_frame(8, 4, 80);
        let frame_b = make_video_frame(8, 4, 200);
        let t = make_transition(TransitionType::Dissolve);

        let out = TransitionRenderer::blend_video(&frame_a, &frame_b, &t, 1.0);

        for plane in &out.planes {
            for &b in &plane.data {
                assert_eq!(b, 200);
            }
        }
    }

    /// Dimension mismatch must not panic and must return the larger frame.
    #[test]
    fn test_blend_video_dimension_mismatch_no_panic() {
        let frame_a = make_video_frame(8, 4, 100);
        let frame_b = make_video_frame(16, 8, 200);
        let t = make_transition(TransitionType::Dissolve);

        // Must not panic.
        let out = TransitionRenderer::blend_video(&frame_a, &frame_b, &t, 0.5);
        // The larger frame (frame_b, 16×8) is returned.
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 8);
    }

    /// Dimension mismatch: same format but frame_a is larger.
    #[test]
    fn test_blend_video_dimension_mismatch_a_larger() {
        let frame_a = make_video_frame(16, 8, 100);
        let frame_b = make_video_frame(8, 4, 200);
        let t = make_transition(TransitionType::Dissolve);

        let out = TransitionRenderer::blend_video(&frame_a, &frame_b, &t, 0.5);
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 8);
    }

    /// WipeLeft at 0.5 — left half of output should equal frame_b pixels.
    #[test]
    fn test_blend_video_wipe_left() {
        let frame_a = make_video_frame(8, 4, 10);
        let frame_b = make_video_frame(8, 4, 250);
        let t = make_transition(TransitionType::WipeLeft);

        let out = TransitionRenderer::blend_video(&frame_a, &frame_b, &t, 0.5);

        // Y-plane: columns 0-3 should be frame_b (250), columns 4-7 should be frame_a (10).
        let y_plane = &out.planes[0];
        for y in 0..4usize {
            for x in 0..4usize {
                assert_eq!(y_plane.data[y * y_plane.stride + x], 250, "x={x},y={y}");
            }
            for x in 4..8usize {
                assert_eq!(y_plane.data[y * y_plane.stride + x], 10, "x={x},y={y}");
            }
        }
    }

    // ── mix_audio tests ──────────────────────────────────────────────────────

    /// Cross-fade at 0.5: 0.5*a + 0.5*b; with a=0.5 and b=-0.5 → 0.0.
    #[test]
    fn test_mix_audio_f32_mid() {
        let frame_a = make_audio_frame(64, 0.5_f32);
        let frame_b = make_audio_frame(64, -0.5_f32);
        let t = make_transition(TransitionType::CrossFade);

        let out = TransitionRenderer::mix_audio(&frame_a, &frame_b, &t, 0.5);

        if let AudioBuffer::Interleaved(bytes) = &out.samples {
            for chunk in bytes.chunks_exact(4) {
                let v = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                assert!(v.abs() < 1e-5, "expected ~0.0, got {v}");
            }
        } else {
            panic!("expected interleaved buffer");
        }
    }

    /// Cross-fade at 0.0 should preserve frame_a samples.
    #[test]
    fn test_mix_audio_f32_zero() {
        let frame_a = make_audio_frame(32, 0.8_f32);
        let frame_b = make_audio_frame(32, -0.8_f32);
        let t = make_transition(TransitionType::CrossFade);

        let out = TransitionRenderer::mix_audio(&frame_a, &frame_b, &t, 0.0);

        if let AudioBuffer::Interleaved(bytes) = &out.samples {
            for chunk in bytes.chunks_exact(4) {
                let v = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                assert!((v - 0.8).abs() < 1e-5, "expected 0.8, got {v}");
            }
        } else {
            panic!("expected interleaved buffer");
        }
    }

    /// Cross-fade at 1.0 should reproduce frame_b samples.
    #[test]
    fn test_mix_audio_f32_one() {
        let frame_a = make_audio_frame(32, 0.3_f32);
        let frame_b = make_audio_frame(32, 0.9_f32);
        let t = make_transition(TransitionType::CrossFade);

        let out = TransitionRenderer::mix_audio(&frame_a, &frame_b, &t, 1.0);

        if let AudioBuffer::Interleaved(bytes) = &out.samples {
            for chunk in bytes.chunks_exact(4) {
                let v = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                assert!((v - 0.9).abs() < 1e-5, "expected 0.9, got {v}");
            }
        } else {
            panic!("expected interleaved buffer");
        }
    }

    /// Format mismatch on audio returns frame_a unchanged.
    #[test]
    fn test_mix_audio_format_mismatch() {
        let frame_a = make_audio_frame(16, 0.5_f32);
        let mut frame_b = make_audio_frame(16, -0.5_f32);
        frame_b.format = SampleFormat::S16; // intentional mismatch

        let t = make_transition(TransitionType::CrossFade);
        let out = TransitionRenderer::mix_audio(&frame_a, &frame_b, &t, 0.5);

        // Should return frame_a unchanged.
        assert_eq!(out.format, SampleFormat::F32);
        assert_eq!(out.samples.size(), frame_a.samples.size());
    }
}
