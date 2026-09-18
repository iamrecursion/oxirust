//! Auto-caption pipeline: ASR → diarization → caption blocks → line breaking
//! → WCAG validation, all in one orchestrator.
//!
//! The pipeline is generic over an [`AsrEngine`] (the speech-to-text
//! component) and accepts an optional [`DiarizationEngine`] for speaker
//! attribution.  All stages operate on `f32` PCM mono audio at a known
//! sample rate; callers wanting integration with `oximedia-core`'s
//! `AudioFrame` should down-mix and convert to `f32` before calling
//! [`AutoCaptionPipeline::process_audio`].
//!
//! # Example
//!
//! ```rust
//! use oximedia_caption_gen::auto_pipeline::{
//!     AsrEngine, AutoCaptionPipeline, LineBreakStrategy, PipelineConfig,
//! };
//! use oximedia_caption_gen::alignment::WordTimestamp;
//! use oximedia_caption_gen::CaptionGenError;
//!
//! struct FixedAsr;
//! impl AsrEngine for FixedAsr {
//!     fn transcribe(
//!         &self,
//!         _audio: &[f32],
//!         _sample_rate: u32,
//!     ) -> Result<Vec<WordTimestamp>, CaptionGenError> {
//!         Ok(vec![
//!             WordTimestamp {
//!                 word: "hello".into(),
//!                 start_ms: 0,
//!                 end_ms: 600,
//!                 confidence: 0.95,
//!                 word_confidence: 0.95,
//!             },
//!             WordTimestamp {
//!                 word: "world".into(),
//!                 start_ms: 700,
//!                 end_ms: 1300,
//!                 confidence: 0.92,
//!                 word_confidence: 0.92,
//!             },
//!         ])
//!     }
//! }
//!
//! let config = PipelineConfig {
//!     max_line_length: 32,
//!     max_lines_per_block: 2,
//!     min_block_duration_ms: 1_000,
//!     max_block_duration_ms: 6_000,
//!     min_gap_ms: 80,
//!     language: Some("en".into()),
//!     enable_diarization: false,
//!     line_break_strategy: LineBreakStrategy::OptimalSmawk,
//! };
//! let pipeline = AutoCaptionPipeline::new(FixedAsr, config);
//! let track = pipeline.process_audio(&vec![0.0_f32; 16_000], 16_000)
//!     .expect("pipeline runs to completion");
//! assert!(!track.blocks.is_empty());
//! ```

use crate::alignment::{
    build_caption_blocks, CaptionBlock, CaptionPosition, TranscriptSegment, WordTimestamp,
};
use crate::diarization::{assign_speakers_to_blocks, DiarizationResult};
use crate::language_detect::{LanguageCode, LanguageDetector};
use crate::line_breaking::{greedy_break, optimal_break, optimal_break_smawk};
use crate::wcag::{run_all_checks, WcagLevel, WcagViolation};
use crate::CaptionGenError;

// ─── Engines ─────────────────────────────────────────────────────────────────

/// A pluggable automatic speech recogniser.
///
/// Implementations consume mono `f32` PCM samples at a known sample
/// rate and return word-level timestamps.  Errors are surfaced as
/// [`CaptionGenError`] so they integrate cleanly with the rest of the
/// crate.
pub trait AsrEngine: Send + Sync {
    /// Transcribe `audio` and return word-level timestamps in temporal
    /// order.
    fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<WordTimestamp>, CaptionGenError>;
}

/// A pluggable diarisation engine.
///
/// Implementations consume the same `f32` PCM input as the ASR engine
/// and produce a [`DiarizationResult`] enumerating speaker turns.
pub trait DiarizationEngine: Send + Sync {
    /// Diarise `audio` into speaker turns.
    fn diarize(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<DiarizationResult, CaptionGenError>;
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Strategy used when wrapping a caption block's text onto multiple lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineBreakStrategy {
    /// Greedy first-fit; fastest but least balanced.
    Greedy,
    /// Knuth-Plass DP with squared-slack penalty; `O(n^2)`.
    OptimalDp,
    /// Knuth-Plass DP accelerated with SMAWK; amortised `O(n)` on
    /// bounded-width inputs.
    OptimalSmawk,
}

/// Configuration for an auto-caption pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum characters per caption line.  WCAG 2.1 strongly
    /// recommends ≤ 32 characters for legibility on 720p+ displays.
    pub max_line_length: usize,
    /// Maximum number of lines per caption block.  WCAG 2.1 strongly
    /// recommends ≤ 2 lines so the caption never occludes more than a
    /// quarter of the frame.
    pub max_lines_per_block: usize,
    /// Minimum on-screen duration (ms) for any caption block.
    pub min_block_duration_ms: u32,
    /// Maximum on-screen duration (ms); blocks are split if longer.
    pub max_block_duration_ms: u32,
    /// Minimum gap (ms) between consecutive blocks.  Words separated
    /// by more than this gap start a new block.
    pub min_gap_ms: u32,
    /// ISO 639-1 language code (`Some("en")`, `Some("ja")`, ...).
    /// When `None`, the language is auto-detected from the ASR output
    /// via [`LanguageDetector`].
    pub language: Option<String>,
    /// Toggle the diarisation stage; when `true` and a
    /// [`DiarizationEngine`] is attached, speaker tags propagate to
    /// the output blocks.
    pub enable_diarization: bool,
    /// Which line-breaking algorithm to use.
    pub line_break_strategy: LineBreakStrategy,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_line_length: 32,
            max_lines_per_block: 2,
            min_block_duration_ms: 1_000,
            max_block_duration_ms: 7_000,
            min_gap_ms: 80,
            language: None,
            enable_diarization: false,
            line_break_strategy: LineBreakStrategy::OptimalSmawk,
        }
    }
}

// ─── Caption track ───────────────────────────────────────────────────────────

/// A complete caption track emitted by the pipeline.
#[derive(Debug, Clone)]
pub struct CaptionTrack {
    /// Ordered caption blocks.
    pub blocks: Vec<CaptionBlock>,
    /// Detected (or supplied) language for the track.
    pub language: LanguageCode,
    /// WCAG violations found in the produced blocks.  An empty vec
    /// means the track passed the requested conformance level.
    pub wcag_violations: Vec<WcagViolation>,
    /// Speakers attributed to blocks, sorted by appearance.  Empty
    /// when diarisation was disabled.
    pub speakers: Vec<u8>,
    /// Total audio duration the pipeline considered (ms).
    pub total_duration_ms: u64,
}

impl Default for CaptionTrack {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            language: LanguageCode::unknown(),
            wcag_violations: Vec::new(),
            speakers: Vec::new(),
            total_duration_ms: 0,
        }
    }
}

impl CaptionTrack {
    /// True if the track has no caption blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

// ─── Pipeline ────────────────────────────────────────────────────────────────

/// End-to-end auto-caption pipeline.
///
/// The pipeline owns an [`AsrEngine`] and optional
/// [`DiarizationEngine`].  Construction is via [`Self::new`]; attach
/// a diariser with [`Self::with_diarizer`] (builder style).
pub struct AutoCaptionPipeline<A: AsrEngine> {
    asr: A,
    diarizer: Option<Box<dyn DiarizationEngine>>,
    language_detector: LanguageDetector,
    config: PipelineConfig,
    wcag_level: WcagLevel,
}

impl<A: AsrEngine> AutoCaptionPipeline<A> {
    /// Create a new pipeline using the given ASR engine and config.
    ///
    /// WCAG level defaults to [`WcagLevel::AA`].
    pub fn new(asr: A, config: PipelineConfig) -> Self {
        Self {
            asr,
            diarizer: None,
            language_detector: LanguageDetector::new(),
            config,
            wcag_level: WcagLevel::AA,
        }
    }

    /// Attach a [`DiarizationEngine`] to the pipeline.
    ///
    /// The pipeline still requires `config.enable_diarization` to be
    /// `true` for the diariser to run.
    pub fn with_diarizer<D: DiarizationEngine + 'static>(mut self, d: D) -> Self {
        self.diarizer = Some(Box::new(d));
        self
    }

    /// Override the WCAG conformance level used for the final
    /// validation stage.  Defaults to [`WcagLevel::AA`].
    pub fn with_wcag_level(mut self, level: WcagLevel) -> Self {
        self.wcag_level = level;
        self
    }

    /// Process a slab of audio and emit a [`CaptionTrack`].
    ///
    /// Handles all six stages:
    /// 1. Optional language auto-detection
    /// 2. ASR transcription
    /// 3. Optional diarisation
    /// 4. Word → caption-block grouping
    /// 5. Per-block line breaking
    /// 6. WCAG 2.1 validation
    pub fn process_audio(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<CaptionTrack, CaptionGenError> {
        if sample_rate == 0 {
            return Err(CaptionGenError::InvalidParameter(
                "sample_rate must be non-zero".to_string(),
            ));
        }

        // Total duration is `samples * 1000 / sample_rate` ms.
        let total_duration_ms = (samples.len() as u64)
            .saturating_mul(1000)
            .checked_div(u64::from(sample_rate))
            .unwrap_or(0);

        // Empty audio → empty track (no error, no panic).
        if samples.is_empty() {
            return Ok(CaptionTrack {
                blocks: Vec::new(),
                language: self
                    .config
                    .language
                    .as_deref()
                    .map(|c| LanguageCode(c.to_string()))
                    .unwrap_or_else(LanguageCode::unknown),
                wcag_violations: Vec::new(),
                speakers: Vec::new(),
                total_duration_ms: 0,
            });
        }

        // ── Stage 2: ASR ────────────────────────────────────────────
        let words = self.asr.transcribe(samples, sample_rate)?;

        if words.is_empty() {
            return Ok(CaptionTrack {
                blocks: Vec::new(),
                language: self
                    .config
                    .language
                    .as_deref()
                    .map(|c| LanguageCode(c.to_string()))
                    .unwrap_or_else(LanguageCode::unknown),
                wcag_violations: Vec::new(),
                speakers: Vec::new(),
                total_duration_ms,
            });
        }

        // ── Stage 1 (deferred to here so we have transcript text): language detect ──
        let language = self.resolve_language(&words);

        // ── Stages 4a + 4b: group words → segments → blocks ─────────
        let segments = self.group_words_to_segments(&words);
        let mut blocks = build_caption_blocks(
            &segments,
            self.config.max_lines_per_block.min(u8::MAX as usize) as u8,
            self.config.max_line_length.min(u8::MAX as usize) as u8,
        );

        // Re-flow lines with the selected strategy.
        self.rewrap_blocks(&mut blocks, &segments);

        // ── Stage 3: optional diarisation + speaker attribution ─────
        let mut speakers_seen: Vec<u8> = Vec::new();
        if self.config.enable_diarization {
            if let Some(diarizer) = self.diarizer.as_ref() {
                let diar_result = diarizer.diarize(samples, sample_rate)?;
                assign_speakers_to_blocks(&mut blocks, &diar_result);
                for b in &blocks {
                    if let Some(sid) = b.speaker_id {
                        if !speakers_seen.contains(&sid) {
                            speakers_seen.push(sid);
                        }
                    }
                }
            }
        }

        // ── Stage 6: WCAG validation ────────────────────────────────
        let wcag_violations = run_all_checks(&blocks, total_duration_ms, self.wcag_level.clone());

        Ok(CaptionTrack {
            blocks,
            language,
            wcag_violations,
            speakers: speakers_seen,
            total_duration_ms,
        })
    }

    // ── Helpers ─────────────────────────────────────────────────────

    /// Resolve the working language: configured override or
    /// auto-detection from the transcript text.
    fn resolve_language(&self, words: &[WordTimestamp]) -> LanguageCode {
        if let Some(code) = self.config.language.as_deref() {
            return LanguageCode(code.to_string());
        }
        let combined: String = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        self.language_detector.detect(&combined).language
    }

    /// Group word timestamps into caption-block-sized segments by
    /// breaking on gaps `> config.min_gap_ms` and on cumulative
    /// duration exceeding `config.max_block_duration_ms`.
    fn group_words_to_segments(&self, words: &[WordTimestamp]) -> Vec<TranscriptSegment> {
        if words.is_empty() {
            return Vec::new();
        }
        let min_gap = u64::from(self.config.min_gap_ms);
        let max_dur = u64::from(self.config.max_block_duration_ms);
        let min_dur = u64::from(self.config.min_block_duration_ms);

        let mut out: Vec<TranscriptSegment> = Vec::new();
        let mut bucket: Vec<WordTimestamp> = Vec::new();

        let flush = |bucket: &mut Vec<WordTimestamp>, out: &mut Vec<TranscriptSegment>| {
            if bucket.is_empty() {
                return;
            }
            let start_ms = bucket.first().map(|w| w.start_ms).unwrap_or(0);
            let end_ms = bucket.iter().map(|w| w.end_ms).max().unwrap_or(start_ms);
            let text = bucket
                .iter()
                .map(|w| w.word.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let words = std::mem::take(bucket);
            out.push(TranscriptSegment {
                text,
                start_ms,
                end_ms,
                speaker_id: None,
                words,
            });
        };

        for w in words {
            let next_start = w.start_ms;
            if let Some(last) = bucket.last() {
                let gap = next_start.saturating_sub(last.end_ms);
                let seg_start = bucket.first().map(|w0| w0.start_ms).unwrap_or(next_start);
                let span = w.end_ms.saturating_sub(seg_start);
                if gap > min_gap || span > max_dur {
                    flush(&mut bucket, &mut out);
                }
            }
            bucket.push(w.clone());
        }
        flush(&mut bucket, &mut out);

        // Enforce min_block_duration_ms by extending undersized blocks
        // to at least min_dur (capped by the next block's start when
        // possible).
        if min_dur > 0 {
            for i in 0..out.len() {
                let span = out[i].end_ms.saturating_sub(out[i].start_ms);
                if span < min_dur {
                    let next_start = out.get(i + 1).map(|s| s.start_ms);
                    let cap = next_start.unwrap_or(out[i].start_ms + min_dur);
                    let extended_end = (out[i].start_ms + min_dur).min(cap);
                    if extended_end > out[i].end_ms {
                        out[i].end_ms = extended_end;
                    }
                }
            }
        }

        out
    }

    /// Re-wrap every block's text using the configured line-break
    /// strategy.
    fn rewrap_blocks(&self, blocks: &mut [CaptionBlock], _segments: &[TranscriptSegment]) {
        let max_len = self.config.max_line_length.min(u8::MAX as usize) as u8;
        for block in blocks.iter_mut() {
            let combined = block.lines.join(" ");
            let mut new_lines = match self.config.line_break_strategy {
                LineBreakStrategy::Greedy => greedy_break(&combined, max_len),
                LineBreakStrategy::OptimalDp => optimal_break(&combined, max_len),
                LineBreakStrategy::OptimalSmawk => optimal_break_smawk(&combined, max_len),
            };
            if new_lines.len() > self.config.max_lines_per_block
                && self.config.max_lines_per_block > 0
            {
                let keep = self.config.max_lines_per_block - 1;
                let mut truncated: Vec<String> = new_lines.drain(..keep).collect();
                let overflow = new_lines.join(" ");
                truncated.push(overflow);
                new_lines = truncated;
            }
            block.lines = new_lines;
            block.position = CaptionPosition::Bottom;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct CannedAsr(Vec<WordTimestamp>);
    impl AsrEngine for CannedAsr {
        fn transcribe(
            &self,
            _audio: &[f32],
            _sample_rate: u32,
        ) -> Result<Vec<WordTimestamp>, CaptionGenError> {
            Ok(self.0.clone())
        }
    }

    fn mk_word(w: &str, s: u64, e: u64) -> WordTimestamp {
        WordTimestamp {
            word: w.to_string(),
            start_ms: s,
            end_ms: e,
            confidence: 1.0,
            word_confidence: 1.0,
        }
    }

    #[test]
    fn pipeline_default_config_runs() {
        let asr = CannedAsr(vec![
            mk_word("hello", 0, 500),
            mk_word("there", 500, 1000),
            mk_word("world", 1100, 1800),
        ]);
        let cfg = PipelineConfig {
            language: Some("en".to_string()),
            ..PipelineConfig::default()
        };
        let pipe = AutoCaptionPipeline::new(asr, cfg);
        let track = pipe
            .process_audio(&vec![0.0_f32; 32_000], 16_000)
            .expect("pipeline runs");
        assert!(!track.blocks.is_empty());
    }

    #[test]
    fn pipeline_empty_audio_returns_empty_track() {
        let asr = CannedAsr(vec![]);
        let cfg = PipelineConfig::default();
        let pipe = AutoCaptionPipeline::new(asr, cfg);
        let track = pipe.process_audio(&[], 16_000).expect("ok");
        assert!(track.is_empty());
        assert_eq!(track.total_duration_ms, 0);
    }

    #[test]
    fn pipeline_rejects_zero_sample_rate() {
        let asr = CannedAsr(vec![]);
        let cfg = PipelineConfig::default();
        let pipe = AutoCaptionPipeline::new(asr, cfg);
        assert!(pipe.process_audio(&[0.0_f32; 100], 0).is_err());
    }

    #[test]
    fn group_words_breaks_on_gap() {
        let asr = CannedAsr(vec![]);
        let pipe = AutoCaptionPipeline::new(
            asr,
            PipelineConfig {
                min_gap_ms: 500,
                max_block_duration_ms: 10_000,
                min_block_duration_ms: 0,
                ..PipelineConfig::default()
            },
        );
        let words = vec![
            mk_word("a", 0, 100),
            mk_word("b", 200, 300),   // small gap → same group
            mk_word("c", 1000, 1100), // gap = 700ms > 500ms → split
        ];
        let segs = pipe.group_words_to_segments(&words);
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn group_words_breaks_on_max_duration() {
        let asr = CannedAsr(vec![]);
        let pipe = AutoCaptionPipeline::new(
            asr,
            PipelineConfig {
                min_gap_ms: 10_000,
                max_block_duration_ms: 1_500,
                min_block_duration_ms: 0,
                ..PipelineConfig::default()
            },
        );
        let words = vec![
            mk_word("a", 0, 500),
            mk_word("b", 600, 1100),
            mk_word("c", 1200, 1700),
            mk_word("d", 1800, 2300),
        ];
        let segs = pipe.group_words_to_segments(&words);
        assert!(segs.len() >= 2);
    }
}
