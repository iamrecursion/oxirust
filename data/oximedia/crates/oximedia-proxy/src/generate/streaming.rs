//! Streaming (progressive) proxy generation — start editing before the proxy
//! has finished generating.
//!
//! The batch [`crate::ProxyGenerator`] produces a proxy *all at once*: callers
//! must `await` the whole transcode before any of the output is usable. This
//! module adds the complementary *progressive* path. A
//! [`StreamingProxyGenerator`] produces a proxy **segment by segment** along a
//! timeline and exposes a monotonic "ready-up-to" cursor so a downstream editor
//! or consumer can open the proxy and seek/edit within the already-generated
//! range while later segments are still being produced.
//!
//! It also pairs naturally with the streaming *delivery* layer
//! ([`crate::proxy_streaming`]): every [`ProxySegment`] carries a
//! [`ProxySegment::byte_range`] that can be handed straight to the delivery
//! server once the segment is ready.
//!
//! # Architecture
//!
//! * [`SegmentPlan`] divides a total timeline into fixed-duration segments
//!   (the final segment may be shorter).
//! * [`ProgressiveProxy`] is the synchronous readiness state machine. It owns
//!   the monotonic cursor ([`ProgressiveProxy::ready_until`]), the ordered list
//!   of completed segments, and a **bounded buffer** that provides
//!   back-pressure: the generator may run at most `buffer_capacity` segments
//!   ahead of the consumer before it must wait for the consumer to drain.
//! * [`StreamingProxyGenerator`] is the `tokio`-based async ergonomic wrapper.
//!   Producer and consumer share one [`ProgressiveProxy`] behind a mutex and
//!   coordinate through [`tokio::sync::Notify`]; a consumer can `await` the next
//!   ready segment or `await` a specific timeline position becoming ready.
//!
//! # What is real and what is modelled
//!
//! The readiness/streaming machinery — the monotonic cursor, in-order segment
//! completion, bounded back-pressure buffering, and finalization signalling —
//! is fully real. The *per-segment encode* is delegated to an injected closure
//! ([`StreamingProxyGenerator::run_with`]). The built-in default
//! ([`ConstantBitrateModel`]) models segment byte sizes deterministically from
//! the configured bitrate and segment duration; it does **not** invoke a real
//! video codec, matching the rest of this crate's simulated encode path. To
//! drive a real encoder, pass your own segment-producer closure that performs
//! the actual range transcode and returns each segment's real byte length.
//!
//! # Examples
//!
//! Synchronous core — produce and inspect every segment:
//!
//! ```
//! use std::time::Duration;
//! use oximedia_proxy::generate::streaming::{
//!     ConstantBitrateModel, ProgressiveProxy, SegmentPlan,
//! };
//!
//! // 250 ms timeline cut into 100 ms segments -> 3 segments (100, 100, 50).
//! let plan = SegmentPlan::new(Duration::from_millis(250), Duration::from_millis(100))
//!     .expect("segment duration is non-zero");
//! let mut proxy = ProgressiveProxy::new(plan, 0); // 0 = unbounded buffer
//! let model = ConstantBitrateModel::new(8_000_000); // 8 Mbps
//!
//! let segments = proxy
//!     .generate_to_completion(|spec| Ok(model.segment_bytes(spec)))
//!     .expect("synthetic encode never fails");
//!
//! assert_eq!(segments.len(), 3);
//! assert!(proxy.is_finished());
//! assert_eq!(proxy.ready_until(), Duration::from_millis(250));
//! ```
//!
//! Asynchronous progressive generation — edit while generating:
//!
//! ```no_run
//! use std::time::Duration;
//! use oximedia_proxy::generate::streaming::{StreamingProxyConfig, StreamingProxyGenerator};
//!
//! # async fn example() -> oximedia_proxy::Result<()> {
//! let config = StreamingProxyConfig::new(
//!     Duration::from_secs(60),      // 60 s proxy
//!     Duration::from_secs(2),       // 2 s segments
//!     6_000_000,                    // 6 Mbps size model
//!     4,                            // at most 4 segments ahead of the consumer
//! )?;
//! let generator = StreamingProxyGenerator::new(config);
//!
//! // Drive generation in the background.
//! let producer = generator.clone();
//! let handle = tokio::spawn(async move { producer.run_modeled().await });
//!
//! // The editor consumes segments as they become ready.
//! while let Some(segment) = generator.next_ready_segment().await {
//!     // open/seek/edit `segment` on the proxy timeline...
//!     let _ = segment.byte_range();
//! }
//! handle.await.expect("join")?;
//! # Ok(())
//! # }
//! ```

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use tokio::sync::Notify;

use super::settings::ProxyGenerationSettings;
use crate::proxy_streaming::ByteRange;
use crate::{ProxyError, Result};

const NANOS_PER_SEC: u128 = 1_000_000_000;
const BITS_PER_BYTE: u128 = 8;

/// Reconstruct a [`Duration`] from a nanosecond count, saturating on overflow.
#[allow(clippy::cast_possible_truncation)]
fn duration_from_nanos(nanos: u128) -> Duration {
    let secs = (nanos / NANOS_PER_SEC).min(u128::from(u64::MAX)) as u64;
    let sub = (nanos % NANOS_PER_SEC) as u32;
    Duration::new(secs, sub)
}

// ---------------------------------------------------------------------------
// Segment plan
// ---------------------------------------------------------------------------

/// A timeline divided into fixed-duration segments.
///
/// The final segment is shortened so the segments exactly tile
/// `[0, total_duration)`. A zero-length timeline contains zero segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentPlan {
    total_duration: Duration,
    segment_duration: Duration,
}

impl SegmentPlan {
    /// Create a plan that cuts `total_duration` into `segment_duration` pieces.
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError::InvalidInput`] when `segment_duration` is zero.
    pub fn new(total_duration: Duration, segment_duration: Duration) -> Result<Self> {
        if segment_duration.is_zero() {
            return Err(ProxyError::InvalidInput(
                "segment_duration must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            total_duration,
            segment_duration,
        })
    }

    /// Total timeline duration.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    /// Nominal duration of each (non-final) segment.
    #[must_use]
    pub fn segment_duration(&self) -> Duration {
        self.segment_duration
    }

    /// Number of segments that tile the timeline (`ceil(total / segment)`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn segment_count(&self) -> u64 {
        let total = self.total_duration.as_nanos();
        let seg = self.segment_duration.as_nanos();
        if total == 0 || seg == 0 {
            return 0;
        }
        ((total + seg - 1) / seg) as u64
    }

    /// Start offset and duration of segment `index`, or `None` if out of range.
    #[must_use]
    pub fn segment_at(&self, index: u64) -> Option<(Duration, Duration)> {
        if index >= self.segment_count() {
            return None;
        }
        let seg = self.segment_duration.as_nanos();
        let total = self.total_duration.as_nanos();
        let start = seg.saturating_mul(u128::from(index));
        if start >= total {
            return None;
        }
        let dur = (total - start).min(seg);
        Some((duration_from_nanos(start), duration_from_nanos(dur)))
    }

    /// Index of the segment that contains timeline position `t`, or `None` if
    /// `t` lies outside `[0, total_duration)`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn index_for_time(&self, t: Duration) -> Option<u64> {
        let total = self.total_duration.as_nanos();
        let tn = t.as_nanos();
        if total == 0 || tn >= total {
            return None;
        }
        let seg = self.segment_duration.as_nanos();
        Some((tn / seg) as u64)
    }
}

// ---------------------------------------------------------------------------
// Segment spec / completed segment
// ---------------------------------------------------------------------------

/// A planned (not-yet-encoded) segment handed to the per-segment encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentSpec {
    /// Zero-based index of the segment within the timeline.
    pub index: u64,
    /// Start position of the segment on the proxy timeline.
    pub start: Duration,
    /// Duration of the segment.
    pub duration: Duration,
}

impl SegmentSpec {
    /// End position of the segment (`start + duration`).
    #[must_use]
    pub fn end(&self) -> Duration {
        self.start.saturating_add(self.duration)
    }
}

/// A completed proxy segment placed on the proxy file's byte layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProxySegment {
    /// Zero-based index of the segment within the timeline.
    pub index: u64,
    /// Start position of the segment on the proxy timeline.
    pub start: Duration,
    /// Duration of the segment.
    pub duration: Duration,
    /// Byte offset of this segment's data within the proxy file.
    pub byte_offset: u64,
    /// Number of bytes this segment occupies in the proxy file.
    pub byte_len: u64,
}

impl ProxySegment {
    /// End position of the segment (`start + duration`).
    #[must_use]
    pub fn end(&self) -> Duration {
        self.start.saturating_add(self.duration)
    }

    /// Whether timeline position `t` falls within `[start, end)`.
    #[must_use]
    pub fn contains(&self, t: Duration) -> bool {
        t >= self.start && t < self.end()
    }

    /// Byte range this segment occupies within the proxy file, suitable for the
    /// streaming delivery layer.
    #[must_use]
    pub fn byte_range(&self) -> ByteRange {
        ByteRange::new(
            self.byte_offset,
            self.byte_offset.saturating_add(self.byte_len),
        )
    }
}

// ---------------------------------------------------------------------------
// Completion outcome
// ---------------------------------------------------------------------------

/// Outcome of attempting to complete the next segment in [`ProgressiveProxy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompleteOutcome {
    /// The segment was completed and is now ready.
    Completed(ProxySegment),
    /// The bounded buffer is full; the consumer must drain a segment before the
    /// generator may run further ahead.
    BufferFull,
    /// Every segment in the plan has already been completed.
    AllSegmentsDone,
}

// ---------------------------------------------------------------------------
// Generation progress snapshot
// ---------------------------------------------------------------------------

/// Immutable snapshot of progressive-generation progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationProgress {
    /// Monotonic readiness cursor: all of `[0, ready_until)` is generated.
    pub ready_until: Duration,
    /// Total timeline duration once complete.
    pub total_duration: Duration,
    /// Number of segments completed (the cursor reflects these).
    pub segments_ready: u64,
    /// Number of completed segments already drained by a consumer.
    pub segments_consumed: u64,
    /// Total number of segments in the plan.
    pub total_segments: u64,
    /// Whether generation has been finalized (all segments produced).
    pub finished: bool,
}

impl GenerationProgress {
    /// Fraction of segments completed, in `[0.0, 1.0]`.
    ///
    /// An empty plan reports `1.0` once finished and `0.0` beforehand.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn ratio(&self) -> f64 {
        if self.total_segments == 0 {
            return if self.finished { 1.0 } else { 0.0 };
        }
        self.segments_ready as f64 / self.total_segments as f64
    }

    /// Number of completed-but-unconsumed segments currently buffered.
    #[must_use]
    pub fn pending(&self) -> u64 {
        self.segments_ready.saturating_sub(self.segments_consumed)
    }
}

// ---------------------------------------------------------------------------
// Constant-bitrate size model (default synthetic encoder)
// ---------------------------------------------------------------------------

/// Deterministic per-segment byte-size model derived from a constant bitrate.
///
/// This is the default stand-in for a real codec: it returns the number of
/// bytes a CBR encode of the given duration would occupy
/// (`bitrate_bps * seconds / 8`). It produces **sizes only** — never fabricated
/// encoded video data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantBitrateModel {
    bitrate_bps: u64,
}

impl ConstantBitrateModel {
    /// Create a model for the given bitrate in bits per second.
    #[must_use]
    pub fn new(bitrate_bps: u64) -> Self {
        Self { bitrate_bps }
    }

    /// Configured bitrate in bits per second.
    #[must_use]
    pub fn bitrate_bps(&self) -> u64 {
        self.bitrate_bps
    }

    /// Modelled byte length of a segment of the given spec.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn segment_bytes(&self, spec: &SegmentSpec) -> u64 {
        let bits = u128::from(self.bitrate_bps) * spec.duration.as_nanos();
        (bits / (BITS_PER_BYTE * NANOS_PER_SEC)) as u64
    }
}

// ---------------------------------------------------------------------------
// Progressive proxy — synchronous readiness state machine
// ---------------------------------------------------------------------------

/// Synchronous progressive-proxy state machine.
///
/// Holds the monotonic readiness cursor, the ordered list of completed
/// segments, and a bounded buffer that bounds how far the generator may run
/// ahead of the consumer (back-pressure). This type performs no I/O and no
/// async work; it is the deterministic core that [`StreamingProxyGenerator`]
/// drives.
#[derive(Debug)]
pub struct ProgressiveProxy {
    plan: SegmentPlan,
    buffer_capacity: usize,
    segments: Vec<ProxySegment>,
    completed: u64,
    consumed: u64,
    ready_until: Duration,
    next_byte_offset: u64,
    finished: bool,
}

impl ProgressiveProxy {
    /// Create a progressive proxy for `plan` with the given back-pressure
    /// buffer capacity.
    ///
    /// `buffer_capacity == 0` means the buffer is unbounded — the generator
    /// never blocks waiting for a consumer. A positive capacity bounds the
    /// number of completed-but-unconsumed segments; once that many are pending,
    /// [`ProgressiveProxy::try_place`] returns [`CompleteOutcome::BufferFull`]
    /// until a segment is drained with [`ProgressiveProxy::consume_next`].
    #[must_use]
    pub fn new(plan: SegmentPlan, buffer_capacity: usize) -> Self {
        let finished = plan.segment_count() == 0;
        Self {
            plan,
            buffer_capacity,
            segments: Vec::new(),
            completed: 0,
            consumed: 0,
            ready_until: Duration::ZERO,
            next_byte_offset: 0,
            finished,
        }
    }

    /// The plan backing this proxy.
    #[must_use]
    pub fn plan(&self) -> &SegmentPlan {
        &self.plan
    }

    /// Back-pressure buffer capacity (`0` = unbounded).
    #[must_use]
    pub fn buffer_capacity(&self) -> usize {
        self.buffer_capacity
    }

    /// Spec of the next segment awaiting completion, or `None` if all segments
    /// have already been completed.
    #[must_use]
    pub fn next_spec(&self) -> Option<SegmentSpec> {
        let index = self.completed;
        let (start, duration) = self.plan.segment_at(index)?;
        Some(SegmentSpec {
            index,
            start,
            duration,
        })
    }

    /// Number of completed-but-unconsumed segments currently buffered.
    #[must_use]
    pub fn pending(&self) -> u64 {
        self.completed - self.consumed
    }

    /// Whether the bounded buffer is full (always `false` when unbounded).
    #[must_use]
    pub fn is_buffer_full(&self) -> bool {
        self.buffer_capacity != 0 && self.pending() >= self.buffer_capacity as u64
    }

    /// Attempt to complete the next segment, recording `byte_len` bytes for it.
    ///
    /// Advances the readiness cursor and appends the segment on success. Returns
    /// [`CompleteOutcome::BufferFull`] (without advancing) when the bounded
    /// buffer is full, or [`CompleteOutcome::AllSegmentsDone`] when the plan is
    /// exhausted.
    pub fn try_place(&mut self, byte_len: u64) -> CompleteOutcome {
        if self.completed >= self.plan.segment_count() {
            return CompleteOutcome::AllSegmentsDone;
        }
        if self.is_buffer_full() {
            return CompleteOutcome::BufferFull;
        }
        let index = self.completed;
        let Some((start, duration)) = self.plan.segment_at(index) else {
            return CompleteOutcome::AllSegmentsDone;
        };
        let segment = ProxySegment {
            index,
            start,
            duration,
            byte_offset: self.next_byte_offset,
            byte_len,
        };
        self.next_byte_offset = self.next_byte_offset.saturating_add(byte_len);
        self.ready_until = segment.end();
        self.completed += 1;
        self.segments.push(segment);
        CompleteOutcome::Completed(segment)
    }

    /// Drain the oldest completed-but-unconsumed segment, freeing a buffer slot.
    ///
    /// Returns `None` when no completed segment is awaiting consumption.
    pub fn consume_next(&mut self) -> Option<ProxySegment> {
        if self.consumed >= self.completed {
            return None;
        }
        let idx = self.consumed as usize;
        let segment = *self.segments.get(idx)?;
        self.consumed += 1;
        Some(segment)
    }

    /// Finalize generation. Valid only once every segment has been completed.
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError::GenerationError`] if called before all segments are
    /// completed.
    pub fn finish(&mut self) -> Result<()> {
        let count = self.plan.segment_count();
        if self.completed < count {
            return Err(ProxyError::GenerationError(format!(
                "cannot finalize streaming proxy: {}/{count} segments generated",
                self.completed
            )));
        }
        self.finished = true;
        Ok(())
    }

    /// Drive every remaining segment to completion synchronously, draining the
    /// bounded buffer as needed so a positive capacity never deadlocks, then
    /// finalize. Returns every segment produced, in order.
    ///
    /// This is the segment-aware synchronous analogue of the batch generator:
    /// it does not stream to an external consumer.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by `encode`, and any error from
    /// finalization.
    pub fn generate_to_completion<F>(&mut self, mut encode: F) -> Result<Vec<ProxySegment>>
    where
        F: FnMut(&SegmentSpec) -> Result<u64>,
    {
        let mut produced = Vec::new();
        while let Some(spec) = self.next_spec() {
            let byte_len = encode(&spec)?;
            match self.try_place(byte_len) {
                CompleteOutcome::Completed(segment) => produced.push(segment),
                CompleteOutcome::BufferFull => {
                    // Self-drain to make room, then retry the same segment.
                    let _ = self.consume_next();
                }
                CompleteOutcome::AllSegmentsDone => break,
            }
        }
        self.finish()?;
        Ok(produced)
    }

    /// Monotonic readiness cursor: all of `[0, ready_until)` is generated.
    #[must_use]
    pub fn ready_until(&self) -> Duration {
        self.ready_until
    }

    /// Whether timeline position `t` is ready (lies within a completed segment,
    /// or equals the end of a finished proxy).
    #[must_use]
    pub fn is_ready(&self, t: Duration) -> bool {
        if t < self.ready_until {
            return true;
        }
        self.finished && t <= self.plan.total_duration()
    }

    /// Whether generation has been finalized.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Whether every completed segment has been consumed.
    #[must_use]
    pub fn fully_consumed(&self) -> bool {
        self.consumed >= self.completed
    }

    /// Number of segments completed so far.
    #[must_use]
    pub fn ready_segment_count(&self) -> usize {
        self.segments.len()
    }

    /// All completed segments, in timeline order.
    #[must_use]
    pub fn ready_segments(&self) -> &[ProxySegment] {
        &self.segments
    }

    /// The completed segment that contains timeline position `t`, if ready.
    #[must_use]
    pub fn ready_segment_for_time(&self, t: Duration) -> Option<ProxySegment> {
        use std::cmp::Ordering;
        match self.segments.binary_search_by(|s| {
            if t < s.start {
                Ordering::Greater
            } else if t >= s.end() {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        }) {
            Ok(i) => self.segments.get(i).copied(),
            Err(_) => None,
        }
    }

    /// Snapshot the current progress.
    #[must_use]
    pub fn progress(&self) -> GenerationProgress {
        GenerationProgress {
            ready_until: self.ready_until,
            total_duration: self.plan.total_duration(),
            segments_ready: self.completed,
            segments_consumed: self.consumed,
            total_segments: self.plan.segment_count(),
            finished: self.finished,
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming proxy configuration
// ---------------------------------------------------------------------------

/// Configuration for a [`StreamingProxyGenerator`].
#[derive(Debug, Clone)]
pub struct StreamingProxyConfig {
    plan: SegmentPlan,
    bitrate_bps: u64,
    buffer_capacity: usize,
}

impl StreamingProxyConfig {
    /// Build a configuration from explicit timeline parameters.
    ///
    /// `bitrate_bps` feeds the default [`ConstantBitrateModel`]; `buffer_capacity`
    /// bounds how far the generator runs ahead of the consumer (`0` = unbounded).
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError::InvalidInput`] when `segment_duration` is zero.
    pub fn new(
        total_duration: Duration,
        segment_duration: Duration,
        bitrate_bps: u64,
        buffer_capacity: usize,
    ) -> Result<Self> {
        let plan = SegmentPlan::new(total_duration, segment_duration)?;
        Ok(Self {
            plan,
            bitrate_bps,
            buffer_capacity,
        })
    }

    /// Build a configuration whose size model is derived from
    /// [`ProxyGenerationSettings`] (video + audio bitrate).
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError::InvalidInput`] when `segment_duration` is zero.
    pub fn from_settings(
        settings: &ProxyGenerationSettings,
        total_duration: Duration,
        segment_duration: Duration,
        buffer_capacity: usize,
    ) -> Result<Self> {
        let bitrate_bps = settings.bitrate.saturating_add(settings.audio_bitrate);
        Self::new(
            total_duration,
            segment_duration,
            bitrate_bps,
            buffer_capacity,
        )
    }

    /// The segment plan.
    #[must_use]
    pub fn plan(&self) -> &SegmentPlan {
        &self.plan
    }

    /// Bitrate (bps) feeding the default size model.
    #[must_use]
    pub fn bitrate_bps(&self) -> u64 {
        self.bitrate_bps
    }

    /// Back-pressure buffer capacity (`0` = unbounded).
    #[must_use]
    pub fn buffer_capacity(&self) -> usize {
        self.buffer_capacity
    }

    /// The default constant-bitrate size model for this configuration.
    #[must_use]
    pub fn model(&self) -> ConstantBitrateModel {
        ConstantBitrateModel::new(self.bitrate_bps)
    }
}

// ---------------------------------------------------------------------------
// Async streaming proxy generator
// ---------------------------------------------------------------------------

/// Local decision used by [`StreamingProxyGenerator::next_ready_segment`].
enum ConsumeStep {
    Yield(ProxySegment),
    Done,
    Wait,
}

/// State shared between the producer and consumer sides of a generator.
#[derive(Debug)]
struct Shared {
    core: Mutex<ProgressiveProxy>,
    model: ConstantBitrateModel,
    /// Signalled by the producer when a segment is completed or finalized.
    produced: Notify,
    /// Signalled by the consumer when a buffer slot is freed.
    consumed: Notify,
}

/// `tokio`-based progressive proxy generator.
///
/// Cheaply cloneable (`Arc` inside): clone it to move a producer into a spawned
/// task while keeping a consumer handle. The producer drives segment production
/// with [`StreamingProxyGenerator::run_modeled`] or
/// [`StreamingProxyGenerator::run_with`]; the consumer awaits ready segments
/// with [`StreamingProxyGenerator::next_ready_segment`] or waits for a timeline
/// position with [`StreamingProxyGenerator::wait_ready`].
#[derive(Debug, Clone)]
pub struct StreamingProxyGenerator {
    shared: Arc<Shared>,
}

impl StreamingProxyGenerator {
    /// Create a generator from a configuration.
    #[must_use]
    pub fn new(config: StreamingProxyConfig) -> Self {
        let core = ProgressiveProxy::new(config.plan().clone(), config.buffer_capacity());
        let model = config.model();
        Self {
            shared: Arc::new(Shared {
                core: Mutex::new(core),
                model,
                produced: Notify::new(),
                consumed: Notify::new(),
            }),
        }
    }

    /// Create a generator whose size model is derived from
    /// [`ProxyGenerationSettings`].
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError::InvalidInput`] when `segment_duration` is zero.
    pub fn from_settings(
        settings: &ProxyGenerationSettings,
        total_duration: Duration,
        segment_duration: Duration,
        buffer_capacity: usize,
    ) -> Result<Self> {
        let config = StreamingProxyConfig::from_settings(
            settings,
            total_duration,
            segment_duration,
            buffer_capacity,
        )?;
        Ok(Self::new(config))
    }

    fn lock_core(&self) -> MutexGuard<'_, ProgressiveProxy> {
        self.shared
            .core
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Drive generation to completion using the default constant-bitrate size
    /// model.
    ///
    /// # Errors
    ///
    /// Propagates any error from finalization (none under normal flow).
    pub async fn run_modeled(&self) -> Result<GenerationProgress> {
        let model = self.shared.model;
        self.run_with(move |spec| Ok(model.segment_bytes(spec)))
            .await
    }

    /// Drive generation to completion using a custom per-segment encoder.
    ///
    /// `encode` is called exactly once per segment (in order) and returns the
    /// segment's byte length. Real encoders perform the actual range transcode
    /// here. The driver respects back-pressure: it never runs more than
    /// `buffer_capacity` segments ahead of the consumer, awaiting a freed buffer
    /// slot when the bound is reached.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by `encode` and any finalization error.
    pub async fn run_with<F>(&self, mut encode: F) -> Result<GenerationProgress>
    where
        F: FnMut(&SegmentSpec) -> Result<u64>,
    {
        loop {
            let spec = self.lock_core().next_spec();
            let Some(spec) = spec else { break };
            let byte_len = encode(&spec)?;

            // Place the segment, waiting for buffer room under back-pressure.
            loop {
                let outcome = self.lock_core().try_place(byte_len);
                match outcome {
                    CompleteOutcome::Completed(_) => {
                        self.shared.produced.notify_waiters();
                        break;
                    }
                    CompleteOutcome::AllSegmentsDone => break,
                    CompleteOutcome::BufferFull => {
                        let notified = self.shared.consumed.notified();
                        tokio::pin!(notified);
                        // Register before re-checking so a concurrent drain is
                        // never missed.
                        notified.as_mut().enable();
                        if self.lock_core().is_buffer_full() {
                            notified.await;
                        }
                    }
                }
            }
        }

        self.lock_core().finish()?;
        // Wake any consumer parked on an empty-but-now-finished proxy.
        self.shared.produced.notify_waiters();
        Ok(self.progress())
    }

    /// Await and drain the next ready segment in timeline order.
    ///
    /// Returns `None` once generation has finished and every segment has been
    /// consumed. Draining a segment frees a back-pressure buffer slot.
    pub async fn next_ready_segment(&self) -> Option<ProxySegment> {
        loop {
            let notified = self.shared.produced.notified();
            tokio::pin!(notified);
            // Register before inspecting state so a concurrent completion is
            // never missed.
            notified.as_mut().enable();

            let step = {
                let mut core = self.lock_core();
                if let Some(segment) = core.consume_next() {
                    ConsumeStep::Yield(segment)
                } else if core.is_finished() && core.fully_consumed() {
                    ConsumeStep::Done
                } else {
                    ConsumeStep::Wait
                }
            };

            match step {
                ConsumeStep::Yield(segment) => {
                    self.shared.consumed.notify_waiters();
                    return Some(segment);
                }
                ConsumeStep::Done => return None,
                ConsumeStep::Wait => notified.await,
            }
        }
    }

    /// Await until timeline position `t` becomes ready.
    ///
    /// Returns `true` once `t` is ready, or `false` if generation finishes
    /// without `t` ever becoming ready (e.g. `t` lies beyond the total
    /// duration).
    pub async fn wait_ready(&self, t: Duration) -> bool {
        loop {
            let notified = self.shared.produced.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            {
                let core = self.lock_core();
                if core.is_ready(t) {
                    return true;
                }
                if core.is_finished() {
                    return false;
                }
            }

            notified.await;
        }
    }

    /// Snapshot the current progress.
    #[must_use]
    pub fn progress(&self) -> GenerationProgress {
        self.lock_core().progress()
    }

    /// Current readiness cursor.
    #[must_use]
    pub fn ready_until(&self) -> Duration {
        self.lock_core().ready_until()
    }

    /// Whether timeline position `t` is ready right now (non-blocking).
    #[must_use]
    pub fn is_ready(&self, t: Duration) -> bool {
        self.lock_core().is_ready(t)
    }

    /// Whether generation has finished.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.lock_core().is_finished()
    }

    /// Number of segments completed so far.
    #[must_use]
    pub fn ready_segment_count(&self) -> usize {
        self.lock_core().ready_segment_count()
    }

    /// Snapshot of all completed segments, in timeline order.
    #[must_use]
    pub fn ready_segments(&self) -> Vec<ProxySegment> {
        self.lock_core().ready_segments().to_vec()
    }

    /// The completed segment containing timeline position `t`, if ready.
    #[must_use]
    pub fn ready_segment_for_time(&self, t: Duration) -> Option<ProxySegment> {
        self.lock_core().ready_segment_for_time(t)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_ms(total: u64, seg: u64) -> SegmentPlan {
        SegmentPlan::new(Duration::from_millis(total), Duration::from_millis(seg))
            .expect("segment duration non-zero")
    }

    // ----- SegmentPlan -----------------------------------------------------

    #[test]
    fn test_plan_rejects_zero_segment() {
        let result = SegmentPlan::new(Duration::from_secs(10), Duration::ZERO);
        assert!(result.is_err());
    }

    #[test]
    fn test_plan_segment_count_even() {
        let plan = plan_ms(1000, 100);
        assert_eq!(plan.segment_count(), 10);
    }

    #[test]
    fn test_plan_segment_count_ceil() {
        // 250 / 100 -> 3 segments (last is partial).
        let plan = plan_ms(250, 100);
        assert_eq!(plan.segment_count(), 3);
    }

    #[test]
    fn test_plan_segment_at_partial_last() {
        let plan = plan_ms(250, 100);
        assert_eq!(
            plan.segment_at(0),
            Some((Duration::from_millis(0), Duration::from_millis(100)))
        );
        assert_eq!(
            plan.segment_at(2),
            Some((Duration::from_millis(200), Duration::from_millis(50)))
        );
        assert_eq!(plan.segment_at(3), None);
    }

    #[test]
    fn test_plan_empty_timeline() {
        let plan = plan_ms(0, 100);
        assert_eq!(plan.segment_count(), 0);
        assert_eq!(plan.segment_at(0), None);
    }

    #[test]
    fn test_plan_single_segment_shorter_than_segment_duration() {
        let plan = plan_ms(50, 100);
        assert_eq!(plan.segment_count(), 1);
        assert_eq!(
            plan.segment_at(0),
            Some((Duration::from_millis(0), Duration::from_millis(50)))
        );
    }

    #[test]
    fn test_plan_index_for_time() {
        let plan = plan_ms(1000, 100);
        assert_eq!(plan.index_for_time(Duration::from_millis(0)), Some(0));
        assert_eq!(plan.index_for_time(Duration::from_millis(150)), Some(1));
        assert_eq!(plan.index_for_time(Duration::from_millis(999)), Some(9));
        assert_eq!(plan.index_for_time(Duration::from_secs(1)), None);
    }

    // ----- ProxySegment ----------------------------------------------------

    #[test]
    fn test_segment_contains_and_byte_range() {
        let seg = ProxySegment {
            index: 1,
            start: Duration::from_millis(100),
            duration: Duration::from_millis(100),
            byte_offset: 4096,
            byte_len: 1024,
        };
        assert_eq!(seg.end(), Duration::from_millis(200));
        assert!(seg.contains(Duration::from_millis(150)));
        assert!(!seg.contains(Duration::from_millis(200)));
        assert!(!seg.contains(Duration::from_millis(99)));
        let range = seg.byte_range();
        assert_eq!(range.start, 4096);
        assert_eq!(range.end, 5120);
        assert_eq!(range.len(), 1024);
    }

    // ----- ConstantBitrateModel -------------------------------------------

    #[test]
    fn test_cbr_model_size() {
        let model = ConstantBitrateModel::new(8_000_000); // 8 Mbps
        let spec = SegmentSpec {
            index: 0,
            start: Duration::ZERO,
            duration: Duration::from_secs(1),
        };
        // 8_000_000 bits/s * 1 s / 8 = 1_000_000 bytes.
        assert_eq!(model.segment_bytes(&spec), 1_000_000);
    }

    #[test]
    fn test_cbr_model_partial_second() {
        let model = ConstantBitrateModel::new(8_000_000);
        let spec = SegmentSpec {
            index: 0,
            start: Duration::ZERO,
            duration: Duration::from_millis(500),
        };
        assert_eq!(model.segment_bytes(&spec), 500_000);
    }

    // ----- ProgressiveProxy: ordering + monotonic cursor -------------------

    #[test]
    fn test_segments_complete_in_order_and_cursor_monotonic() {
        let mut proxy = ProgressiveProxy::new(plan_ms(500, 100), 0);
        let model = ConstantBitrateModel::new(8_000_000);

        let mut prev_cursor = Duration::ZERO;
        let mut expected_index = 0_u64;
        let mut expected_offset = 0_u64;

        while let Some(spec) = proxy.next_spec() {
            let bytes = model.segment_bytes(&spec);
            match proxy.try_place(bytes) {
                CompleteOutcome::Completed(seg) => {
                    // In-order indices.
                    assert_eq!(seg.index, expected_index);
                    expected_index += 1;
                    // Contiguous byte layout.
                    assert_eq!(seg.byte_offset, expected_offset);
                    expected_offset += seg.byte_len;
                    // Monotonically non-decreasing cursor that equals segment end.
                    assert!(proxy.ready_until() >= prev_cursor);
                    assert_eq!(proxy.ready_until(), seg.end());
                    prev_cursor = proxy.ready_until();
                }
                other => panic!("unexpected outcome with unbounded buffer: {other:?}"),
            }
        }

        proxy.finish().expect("all segments completed");
        assert_eq!(proxy.ready_segment_count(), 5);
        assert_eq!(proxy.ready_until(), Duration::from_millis(500));
        assert!(proxy.is_finished());
    }

    #[test]
    fn test_generate_to_completion_helper() {
        let mut proxy = ProgressiveProxy::new(plan_ms(250, 100), 0);
        let model = ConstantBitrateModel::new(4_000_000);
        let segments = proxy
            .generate_to_completion(|spec| Ok(model.segment_bytes(spec)))
            .expect("synthetic encode never fails");
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].index, 0);
        assert_eq!(segments[2].index, 2);
        assert!(proxy.is_finished());
        let progress = proxy.progress();
        assert!((progress.ratio() - 1.0).abs() < f64::EPSILON);
    }

    // ----- ProgressiveProxy: readiness queries -----------------------------

    #[test]
    fn test_is_ready_before_and_after() {
        let mut proxy = ProgressiveProxy::new(plan_ms(300, 100), 0);
        let model = ConstantBitrateModel::new(8_000_000);

        // Nothing ready yet.
        assert!(!proxy.is_ready(Duration::from_millis(0)));
        assert!(!proxy.is_ready(Duration::from_millis(50)));

        // Complete first segment (0..100ms).
        let spec = proxy.next_spec().expect("segment 0 spec");
        let bytes = model.segment_bytes(&spec);
        assert!(matches!(
            proxy.try_place(bytes),
            CompleteOutcome::Completed(_)
        ));

        assert!(proxy.is_ready(Duration::from_millis(50)));
        assert!(proxy.is_ready(Duration::from_millis(99)));
        // The boundary instant (start of the not-yet-ready segment) is not ready.
        assert!(!proxy.is_ready(Duration::from_millis(100)));
        assert!(!proxy.is_ready(Duration::from_millis(150)));
    }

    #[test]
    fn test_ready_segment_for_time_seek() {
        let mut proxy = ProgressiveProxy::new(plan_ms(300, 100), 0);
        let model = ConstantBitrateModel::new(8_000_000);
        let _ = proxy
            .generate_to_completion(|spec| Ok(model.segment_bytes(spec)))
            .expect("ok");

        let seg = proxy
            .ready_segment_for_time(Duration::from_millis(150))
            .expect("time within range is ready");
        assert_eq!(seg.index, 1);
        assert!(proxy
            .ready_segment_for_time(Duration::from_millis(500))
            .is_none());
    }

    #[test]
    fn test_finished_endpoint_is_ready() {
        let mut proxy = ProgressiveProxy::new(plan_ms(200, 100), 0);
        let model = ConstantBitrateModel::new(8_000_000);
        let _ = proxy
            .generate_to_completion(|spec| Ok(model.segment_bytes(spec)))
            .expect("ok");
        // Endpoint of a finished proxy is ready; beyond the end is not.
        assert!(proxy.is_ready(Duration::from_millis(200)));
        assert!(!proxy.is_ready(Duration::from_millis(201)));
    }

    // ----- ProgressiveProxy: finalization ----------------------------------

    #[test]
    fn test_finish_before_complete_errors() {
        let mut proxy = ProgressiveProxy::new(plan_ms(300, 100), 0);
        // No segments placed yet.
        assert!(proxy.finish().is_err());
    }

    #[test]
    fn test_empty_timeline_finished_immediately() {
        let proxy = ProgressiveProxy::new(plan_ms(0, 100), 0);
        assert!(proxy.is_finished());
        assert_eq!(proxy.ready_until(), Duration::ZERO);
        assert_eq!(proxy.ready_segment_count(), 0);
        // The single boundary point t == 0 of an empty finished proxy is ready.
        assert!(proxy.is_ready(Duration::ZERO));
        assert!(!proxy.is_ready(Duration::from_millis(1)));
        assert!((proxy.progress().ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_segment_edge_case() {
        let mut proxy = ProgressiveProxy::new(plan_ms(50, 100), 0);
        let model = ConstantBitrateModel::new(8_000_000);
        let segments = proxy
            .generate_to_completion(|spec| Ok(model.segment_bytes(spec)))
            .expect("ok");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].duration, Duration::from_millis(50));
        assert_eq!(proxy.ready_until(), Duration::from_millis(50));
        assert!(proxy.is_finished());
    }

    // ----- ProgressiveProxy: bounded back-pressure -------------------------

    #[test]
    fn test_back_pressure_blocks_when_buffer_full() {
        let mut proxy = ProgressiveProxy::new(plan_ms(500, 100), 2); // capacity 2
        let model = ConstantBitrateModel::new(8_000_000);

        let bytes = |proxy: &ProgressiveProxy| {
            let spec = proxy.next_spec().expect("spec available");
            model.segment_bytes(&spec)
        };

        // First two placements succeed.
        let b = bytes(&proxy);
        assert!(matches!(proxy.try_place(b), CompleteOutcome::Completed(_)));
        let b = bytes(&proxy);
        assert!(matches!(proxy.try_place(b), CompleteOutcome::Completed(_)));
        assert_eq!(proxy.pending(), 2);
        assert!(proxy.is_buffer_full());

        // Third placement is blocked by back-pressure (cursor stays put).
        let cursor_before = proxy.ready_until();
        let b = bytes(&proxy);
        assert!(matches!(proxy.try_place(b), CompleteOutcome::BufferFull));
        assert_eq!(proxy.ready_until(), cursor_before);
        assert_eq!(proxy.pending(), 2);

        // Consumer drains one -> a slot frees and placement resumes.
        let drained = proxy.consume_next().expect("a segment is buffered");
        assert_eq!(drained.index, 0);
        assert_eq!(proxy.pending(), 1);
        assert!(!proxy.is_buffer_full());
        let b = bytes(&proxy);
        assert!(matches!(proxy.try_place(b), CompleteOutcome::Completed(_)));
        assert_eq!(proxy.pending(), 2);
    }

    #[test]
    fn test_consume_next_in_order() {
        let mut proxy = ProgressiveProxy::new(plan_ms(300, 100), 0);
        let model = ConstantBitrateModel::new(8_000_000);
        let _ = proxy
            .generate_to_completion(|spec| Ok(model.segment_bytes(spec)))
            .expect("ok");
        // generate_to_completion self-drains only on BufferFull; with an
        // unbounded buffer nothing was drained, so all three remain consumable.
        assert_eq!(proxy.consume_next().expect("seg 0").index, 0);
        assert_eq!(proxy.consume_next().expect("seg 1").index, 1);
        assert_eq!(proxy.consume_next().expect("seg 2").index, 2);
        assert!(proxy.consume_next().is_none());
    }

    // ----- StreamingProxyGenerator (async) ---------------------------------

    fn config_ms(total: u64, seg: u64, bitrate: u64, cap: usize) -> StreamingProxyConfig {
        StreamingProxyConfig::new(
            Duration::from_millis(total),
            Duration::from_millis(seg),
            bitrate,
            cap,
        )
        .expect("valid config")
    }

    #[tokio::test]
    async fn test_async_run_modeled_completes() {
        let gen = StreamingProxyGenerator::new(config_ms(1000, 100, 8_000_000, 0));
        let progress = gen.run_modeled().await.expect("generation succeeds");
        assert!(progress.finished);
        assert_eq!(progress.total_segments, 10);
        assert_eq!(progress.segments_ready, 10);
        assert_eq!(progress.ready_until, Duration::from_secs(1));
        assert_eq!(progress.total_duration, Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_async_segments_arrive_in_order() {
        let gen = StreamingProxyGenerator::new(config_ms(500, 100, 8_000_000, 2));
        let producer = gen.clone();
        let handle = tokio::spawn(async move { producer.run_modeled().await });

        let mut indices = Vec::new();
        while let Some(seg) = gen.next_ready_segment().await {
            indices.push(seg.index);
        }
        handle.await.expect("join").expect("generation ok");

        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn test_async_consumer_reads_before_finished() {
        // Small capacity so the producer cannot finish without the consumer.
        let gen = StreamingProxyGenerator::new(config_ms(1000, 100, 8_000_000, 1));
        let producer = gen.clone();
        let handle = tokio::spawn(async move { producer.run_modeled().await });

        // The first segment must be readable while generation is still ongoing.
        let first = gen
            .next_ready_segment()
            .await
            .expect("first segment available");
        assert_eq!(first.index, 0);
        assert!(gen.ready_until() >= Duration::from_millis(100));

        // Drain the rest.
        let mut count = 1;
        while let Some(_seg) = gen.next_ready_segment().await {
            count += 1;
        }
        handle.await.expect("join").expect("generation ok");
        assert_eq!(count, 10);
        assert!(gen.is_finished());
    }

    #[tokio::test]
    async fn test_async_back_pressure_caps_lookahead() {
        // 10 segments, capacity 2: with no consumer draining, the generator can
        // never complete more than `capacity` segments, and can never finish.
        let gen = StreamingProxyGenerator::new(config_ms(1000, 100, 8_000_000, 2));
        let producer = gen.clone();
        let handle = tokio::spawn(async move { producer.run_modeled().await });

        // Let the producer run until it is back-pressure-bound.
        for _ in 0..1000 {
            if gen.progress().segments_ready > 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }

        let progress = gen.progress();
        assert!(
            progress.segments_ready <= 2,
            "back-pressure must cap look-ahead at capacity, got {}",
            progress.segments_ready
        );
        assert!(progress.segments_ready >= 1, "the producer should have run");
        assert!(
            !progress.finished,
            "cannot finish while back-pressure-bound"
        );

        // Draining unblocks the producer and lets it finish (liveness).
        let mut count = 0;
        while let Some(_seg) = gen.next_ready_segment().await {
            count += 1;
        }
        handle.await.expect("join").expect("generation ok");
        assert_eq!(count, 10);
        assert!(gen.is_finished());
    }

    #[tokio::test]
    async fn test_async_wait_ready_true_for_in_range() {
        let gen = StreamingProxyGenerator::new(config_ms(1000, 100, 8_000_000, 0));
        let producer = gen.clone();
        let handle = tokio::spawn(async move { producer.run_modeled().await });

        // A mid-timeline position eventually becomes ready.
        assert!(gen.wait_ready(Duration::from_millis(500)).await);
        handle.await.expect("join").expect("generation ok");
    }

    #[tokio::test]
    async fn test_async_wait_ready_false_beyond_total() {
        let gen = StreamingProxyGenerator::new(config_ms(300, 100, 8_000_000, 0));
        gen.run_modeled().await.expect("generation ok");
        // A position beyond the total duration is never ready, even when done.
        assert!(!gen.wait_ready(Duration::from_millis(400)).await);
    }

    #[tokio::test]
    async fn test_async_empty_timeline() {
        let gen = StreamingProxyGenerator::new(config_ms(0, 100, 8_000_000, 4));
        let producer = gen.clone();
        let handle = tokio::spawn(async move { producer.run_modeled().await });

        // No segments to read; the consumer immediately observes completion.
        assert!(gen.next_ready_segment().await.is_none());
        let progress = handle.await.expect("join").expect("generation ok");
        assert!(progress.finished);
        assert_eq!(progress.total_segments, 0);
        assert_eq!(progress.ready_until, Duration::ZERO);
    }

    #[tokio::test]
    async fn test_async_single_segment() {
        let gen = StreamingProxyGenerator::new(config_ms(50, 100, 8_000_000, 4));
        let producer = gen.clone();
        let handle = tokio::spawn(async move { producer.run_modeled().await });

        let seg = gen.next_ready_segment().await.expect("one segment");
        assert_eq!(seg.index, 0);
        assert_eq!(seg.duration, Duration::from_millis(50));
        assert!(gen.next_ready_segment().await.is_none());
        handle.await.expect("join").expect("generation ok");
    }

    #[tokio::test]
    async fn test_async_run_with_custom_encoder() {
        let gen = StreamingProxyGenerator::new(config_ms(300, 100, 0, 0));
        // Custom encoder assigns a fixed byte length per segment.
        let progress = gen.run_with(|_spec| Ok(2048)).await.expect("generation ok");
        assert!(progress.finished);
        let segments = gen.ready_segments();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].byte_offset, 0);
        assert_eq!(segments[1].byte_offset, 2048);
        assert_eq!(segments[2].byte_offset, 4096);
    }

    #[tokio::test]
    async fn test_async_run_with_encoder_error_propagates() {
        let gen = StreamingProxyGenerator::new(config_ms(300, 100, 0, 0));
        let result = gen
            .run_with(|spec| {
                if spec.index == 1 {
                    Err(ProxyError::GenerationError("boom".to_string()))
                } else {
                    Ok(1024)
                }
            })
            .await;
        assert!(result.is_err());
        // Only the first segment was completed before the error.
        assert_eq!(gen.progress().segments_ready, 1);
        assert!(!gen.is_finished());
    }

    #[test]
    fn test_config_from_settings() {
        let settings = ProxyGenerationSettings::default(); // 5 Mbps + 128 kbps audio
        let config = StreamingProxyConfig::from_settings(
            &settings,
            Duration::from_secs(8),
            Duration::from_secs(1),
            4,
        )
        .expect("valid");
        assert_eq!(config.bitrate_bps(), 5_000_000 + 128_000);
        assert_eq!(config.plan().segment_count(), 8);
    }

    #[test]
    fn test_config_rejects_zero_segment() {
        let result =
            StreamingProxyConfig::new(Duration::from_secs(8), Duration::ZERO, 5_000_000, 4);
        assert!(result.is_err());
    }
}
