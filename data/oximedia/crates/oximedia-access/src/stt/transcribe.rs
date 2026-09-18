//! Speech-to-text transcription with word-level confidence scoring.

use crate::error::{AccessError, AccessResult};
use crate::stt::SttConfig;
use crate::transcript::{Transcript, TranscriptEntry};
use bytes::Bytes;
use oximedia_audio::frame::AudioBuffer;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A single word with timing and confidence metadata from STT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordConfidence {
    /// The recognized word.
    pub word: String,
    /// Start time in milliseconds.
    pub start_time_ms: i64,
    /// End time in milliseconds.
    pub end_time_ms: i64,
    /// Confidence score (0.0 to 1.0) for this word.
    pub confidence: f32,
    /// Whether this word was flagged as uncertain.
    pub is_uncertain: bool,
}

impl WordConfidence {
    /// Create a new word confidence entry.
    #[must_use]
    pub fn new(word: String, start_time_ms: i64, end_time_ms: i64, confidence: f32) -> Self {
        Self {
            word,
            start_time_ms,
            end_time_ms,
            confidence: confidence.clamp(0.0, 1.0),
            is_uncertain: confidence < 0.5,
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        self.end_time_ms - self.start_time_ms
    }
}

/// Overall quality assessment for a transcription result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscriptionQualityLevel {
    /// Excellent quality (avg confidence >= 0.95).
    Excellent,
    /// Good quality (avg confidence >= 0.85).
    Good,
    /// Fair quality (avg confidence >= 0.70).
    Fair,
    /// Poor quality (avg confidence >= 0.50).
    Poor,
    /// Very poor quality (avg confidence < 0.50).
    VeryPoor,
}

impl fmt::Display for TranscriptionQualityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Excellent => write!(f, "Excellent"),
            Self::Good => write!(f, "Good"),
            Self::Fair => write!(f, "Fair"),
            Self::Poor => write!(f, "Poor"),
            Self::VeryPoor => write!(f, "Very Poor"),
        }
    }
}

/// Detailed quality assessment of a transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionQuality {
    /// Average confidence across all words.
    pub average_confidence: f32,
    /// Minimum confidence word.
    pub min_confidence: f32,
    /// Maximum confidence word.
    pub max_confidence: f32,
    /// Number of uncertain words (confidence < threshold).
    pub uncertain_word_count: usize,
    /// Total word count.
    pub total_word_count: usize,
    /// Percentage of words above the confidence threshold.
    pub reliable_word_percentage: f32,
    /// Overall quality level.
    pub quality_level: TranscriptionQualityLevel,
    /// Word-level confidence data.
    pub word_confidences: Vec<WordConfidence>,
    /// Regions that need human review (consecutive uncertain words).
    pub review_regions: Vec<ReviewRegion>,
}

/// A region of the transcript that needs human review due to low confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewRegion {
    /// Start time in milliseconds.
    pub start_time_ms: i64,
    /// End time in milliseconds.
    pub end_time_ms: i64,
    /// Text in the region.
    pub text: String,
    /// Average confidence in this region.
    pub average_confidence: f32,
}

impl TranscriptionQuality {
    /// Compute quality assessment from word confidence data.
    #[must_use]
    pub fn from_word_confidences(words: Vec<WordConfidence>, confidence_threshold: f32) -> Self {
        if words.is_empty() {
            return Self {
                average_confidence: 0.0,
                min_confidence: 0.0,
                max_confidence: 0.0,
                uncertain_word_count: 0,
                total_word_count: 0,
                reliable_word_percentage: 0.0,
                quality_level: TranscriptionQualityLevel::VeryPoor,
                word_confidences: Vec::new(),
                review_regions: Vec::new(),
            };
        }

        let total = words.len();
        let sum: f32 = words.iter().map(|w| w.confidence).sum();
        let avg = sum / total as f32;

        let min = words.iter().map(|w| w.confidence).fold(f32::MAX, f32::min);
        let max = words.iter().map(|w| w.confidence).fold(f32::MIN, f32::max);

        let uncertain_count = words
            .iter()
            .filter(|w| w.confidence < confidence_threshold)
            .count();

        let reliable_pct = (total - uncertain_count) as f32 / total as f32 * 100.0;

        let quality_level = if avg >= 0.95 {
            TranscriptionQualityLevel::Excellent
        } else if avg >= 0.85 {
            TranscriptionQualityLevel::Good
        } else if avg >= 0.70 {
            TranscriptionQualityLevel::Fair
        } else if avg >= 0.50 {
            TranscriptionQualityLevel::Poor
        } else {
            TranscriptionQualityLevel::VeryPoor
        };

        let review_regions = Self::find_review_regions(&words, confidence_threshold);

        Self {
            average_confidence: avg,
            min_confidence: min,
            max_confidence: max,
            uncertain_word_count: uncertain_count,
            total_word_count: total,
            reliable_word_percentage: reliable_pct,
            quality_level,
            word_confidences: words,
            review_regions,
        }
    }

    /// Find consecutive regions of low-confidence words that need review.
    fn find_review_regions(words: &[WordConfidence], threshold: f32) -> Vec<ReviewRegion> {
        let mut regions = Vec::new();
        let mut region_start: Option<usize> = None;

        for (i, word) in words.iter().enumerate() {
            if word.confidence < threshold {
                if region_start.is_none() {
                    region_start = Some(i);
                }
            } else if let Some(start) = region_start {
                // End of a low-confidence region
                let region_words = &words[start..i];
                if !region_words.is_empty() {
                    let text: String = region_words
                        .iter()
                        .map(|w| w.word.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let avg_conf: f32 = region_words.iter().map(|w| w.confidence).sum::<f32>()
                        / region_words.len() as f32;

                    regions.push(ReviewRegion {
                        start_time_ms: region_words.first().map_or(0, |w| w.start_time_ms),
                        end_time_ms: region_words.last().map_or(0, |w| w.end_time_ms),
                        text,
                        average_confidence: avg_conf,
                    });
                }
                region_start = None;
            }
        }

        // Handle region at end
        if let Some(start) = region_start {
            let region_words = &words[start..];
            if !region_words.is_empty() {
                let text: String = region_words
                    .iter()
                    .map(|w| w.word.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                let avg_conf: f32 = region_words.iter().map(|w| w.confidence).sum::<f32>()
                    / region_words.len() as f32;

                regions.push(ReviewRegion {
                    start_time_ms: region_words.first().map_or(0, |w| w.start_time_ms),
                    end_time_ms: region_words.last().map_or(0, |w| w.end_time_ms),
                    text,
                    average_confidence: avg_conf,
                });
            }
        }

        regions
    }

    /// Check if the transcription meets a minimum quality level.
    #[must_use]
    pub fn meets_quality(&self, required: TranscriptionQualityLevel) -> bool {
        match (required, self.quality_level) {
            (TranscriptionQualityLevel::VeryPoor, _) => true,
            (TranscriptionQualityLevel::Poor, TranscriptionQualityLevel::VeryPoor) => false,
            (TranscriptionQualityLevel::Poor, _) => true,
            (
                TranscriptionQualityLevel::Fair,
                TranscriptionQualityLevel::VeryPoor | TranscriptionQualityLevel::Poor,
            ) => false,
            (TranscriptionQualityLevel::Fair, _) => true,
            (
                TranscriptionQualityLevel::Good,
                TranscriptionQualityLevel::Good | TranscriptionQualityLevel::Excellent,
            ) => true,
            (TranscriptionQualityLevel::Good, _) => false,
            (TranscriptionQualityLevel::Excellent, TranscriptionQualityLevel::Excellent) => true,
            (TranscriptionQualityLevel::Excellent, _) => false,
        }
    }

    /// Get words that are below the confidence threshold.
    #[must_use]
    pub fn uncertain_words(&self, threshold: f32) -> Vec<&WordConfidence> {
        self.word_confidences
            .iter()
            .filter(|w| w.confidence < threshold)
            .collect()
    }
}

/// Speech-to-text transcriber with confidence scoring.
pub struct SpeechToText {
    config: SttConfig,
    /// Confidence threshold for marking words as uncertain.
    confidence_threshold: f32,
}

impl SpeechToText {
    /// Create a new STT transcriber.
    #[must_use]
    pub fn new(config: SttConfig) -> Self {
        Self {
            config,
            confidence_threshold: 0.5,
        }
    }

    /// Set the confidence threshold for uncertain word detection.
    #[must_use]
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Transcribe audio to text.
    ///
    /// Integration point for STT services:
    /// - `OpenAI` Whisper
    /// - Google Cloud Speech-to-Text
    /// - Amazon Transcribe
    /// - Microsoft Azure Speech
    /// - `AssemblyAI`
    /// - Local models (Vosk, `DeepSpeech`, etc.)
    pub fn transcribe(&self, _audio: &AudioBuffer) -> AccessResult<Transcript> {
        let mut transcript = Transcript::new();
        transcript.metadata.language = self.config.language.clone();

        // Example transcription result
        transcript.add_entry(TranscriptEntry::new(
            0,
            2000,
            "Example transcription result.".to_string(),
        ));

        Ok(transcript)
    }

    /// Transcribe audio with word-level confidence scores.
    ///
    /// Returns both the transcript and quality assessment with per-word
    /// confidence data, review regions, and overall quality level.
    pub fn transcribe_with_confidence(
        &self,
        _audio: &AudioBuffer,
    ) -> AccessResult<(Transcript, TranscriptionQuality)> {
        let mut transcript = Transcript::new();
        transcript.metadata.language = self.config.language.clone();

        // Simulate STT output with word-level confidence
        let words = vec![
            WordConfidence::new("Example".to_string(), 0, 400, 0.95),
            WordConfidence::new("transcription".to_string(), 400, 900, 0.88),
            WordConfidence::new("result".to_string(), 900, 1300, 0.92),
        ];

        let full_text = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let avg_confidence: f32 =
            words.iter().map(|w| w.confidence).sum::<f32>() / words.len().max(1) as f32;

        let mut entry = TranscriptEntry::new(0, 1300, full_text);
        entry.confidence = avg_confidence;
        transcript.add_entry(entry);

        let quality = TranscriptionQuality::from_word_confidences(words, self.confidence_threshold);

        Ok((transcript, quality))
    }

    /// Assess the quality of an existing transcript by analyzing its entries.
    pub fn assess_quality(&self, transcript: &Transcript) -> AccessResult<TranscriptionQuality> {
        if transcript.entries.is_empty() {
            return Err(AccessError::SttFailed(
                "Cannot assess quality of empty transcript".to_string(),
            ));
        }

        // Build word confidences from transcript entries
        let mut words = Vec::new();
        for entry in &transcript.entries {
            let entry_words: Vec<&str> = entry.text.split_whitespace().collect();
            if entry_words.is_empty() {
                continue;
            }

            let word_duration = entry.duration_ms() / entry_words.len().max(1) as i64;

            for (i, &word) in entry_words.iter().enumerate() {
                let start = entry.start_time_ms + i as i64 * word_duration;
                let end = start + word_duration;
                words.push(WordConfidence::new(
                    word.to_string(),
                    start,
                    end,
                    entry.confidence,
                ));
            }
        }

        Ok(TranscriptionQuality::from_word_confidences(
            words,
            self.confidence_threshold,
        ))
    }

    /// Transcribe with real-time streaming.
    pub fn transcribe_stream(&self, _audio_chunks: &[AudioBuffer]) -> AccessResult<Transcript> {
        self.transcribe(&AudioBuffer::Interleaved(Bytes::new()))
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &SttConfig {
        &self.config
    }

    /// Get the confidence threshold.
    #[must_use]
    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }
}

impl Default for SpeechToText {
    fn default() -> Self {
        Self::new(SttConfig::default())
    }
}

// ─── Incremental STT Processing ─────────────────────────────────────────────

/// Result from processing a single audio chunk incrementally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkTranscriptionResult {
    /// Index of this chunk in the stream.
    pub chunk_index: usize,
    /// Start time of this chunk in milliseconds.
    pub start_time_ms: i64,
    /// End time of this chunk in milliseconds.
    pub end_time_ms: i64,
    /// Recognized text for this chunk (may be empty if audio is silence).
    pub text: String,
    /// Average confidence for this chunk (0.0 to 1.0).
    pub confidence: f32,
    /// Whether this result is final (vs. a partial/interim result).
    pub is_final: bool,
    /// Word-level confidence for words in this chunk.
    pub word_confidences: Vec<WordConfidence>,
}

impl ChunkTranscriptionResult {
    /// Create a new chunk result.
    #[must_use]
    pub fn new(chunk_index: usize, start_time_ms: i64, end_time_ms: i64) -> Self {
        Self {
            chunk_index,
            start_time_ms,
            end_time_ms,
            text: String::new(),
            confidence: 0.0,
            is_final: false,
            word_confidences: Vec::new(),
        }
    }

    /// Mark this result as final.
    #[must_use]
    pub fn finalized(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Duration of this chunk in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        self.end_time_ms - self.start_time_ms
    }
}

/// Configuration for incremental STT processing.
#[derive(Debug, Clone)]
pub struct IncrementalSttConfig {
    /// Target chunk duration in milliseconds.
    pub chunk_duration_ms: i64,
    /// Overlap between consecutive chunks in milliseconds (for context).
    pub overlap_ms: i64,
    /// Whether to emit interim (non-final) results.
    pub emit_interim_results: bool,
    /// Minimum silence duration to detect speech boundaries (ms).
    pub silence_threshold_ms: i64,
    /// RMS amplitude below which a frame is considered silence.
    pub silence_rms_threshold: f32,
    /// STT confidence threshold.
    pub confidence_threshold: f32,
}

impl Default for IncrementalSttConfig {
    fn default() -> Self {
        Self {
            chunk_duration_ms: 500,
            overlap_ms: 50,
            emit_interim_results: false,
            silence_threshold_ms: 200,
            silence_rms_threshold: 0.01,
            confidence_threshold: 0.5,
        }
    }
}

impl IncrementalSttConfig {
    /// Create with a custom chunk duration.
    #[must_use]
    pub fn with_chunk_duration(mut self, duration_ms: i64) -> Self {
        self.chunk_duration_ms = duration_ms.max(100);
        self
    }

    /// Enable interim results.
    #[must_use]
    pub fn with_interim_results(mut self, enable: bool) -> Self {
        self.emit_interim_results = enable;
        self
    }
}

/// Accumulated state for incremental processing.
#[derive(Debug, Clone)]
struct IncrementalState {
    /// Index of the next chunk to process.
    next_chunk: usize,
    /// Time offset of the most recently processed sample.
    cursor_ms: i64,
    /// Pending text from previous overlapping chunks.
    pending_text: String,
}

/// Processes audio in streaming chunks for incremental STT transcription.
///
/// `IncrementalSttProcessor` divides an audio stream into overlapping windows,
/// runs STT on each window independently, and stitches the results into a
/// coherent transcript. It can emit partial results as audio arrives, which
/// is useful for live captioning or real-time transcription pipelines.
///
/// # Example
///
/// ```ignore
/// let config = IncrementalSttConfig::default();
/// let mut processor = IncrementalSttProcessor::new(SttConfig::default(), config);
///
/// for chunk in audio_chunks {
///     let result = processor.process_chunk(&chunk, 48000)?;
///     println!("Chunk {}: {}", result.chunk_index, result.text);
/// }
/// let final_results = processor.flush()?;
/// ```
pub struct IncrementalSttProcessor {
    stt_config: SttConfig,
    incremental_config: IncrementalSttConfig,
    state: IncrementalState,
    /// Results emitted so far.
    results: Vec<ChunkTranscriptionResult>,
}

impl IncrementalSttProcessor {
    /// Create a new incremental processor.
    #[must_use]
    pub fn new(stt_config: SttConfig, incremental_config: IncrementalSttConfig) -> Self {
        Self {
            stt_config,
            incremental_config,
            state: IncrementalState {
                next_chunk: 0,
                cursor_ms: 0,
                pending_text: String::new(),
            },
            results: Vec::new(),
        }
    }

    /// Create with default configs.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SttConfig::default(), IncrementalSttConfig::default())
    }

    /// Process a single audio chunk (raw f32 PCM samples, mono or interleaved stereo).
    ///
    /// `sample_rate` is the sample rate of the chunk.
    /// Returns the transcription result for this chunk.
    pub fn process_chunk(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> crate::error::AccessResult<ChunkTranscriptionResult> {
        let chunk_idx = self.state.next_chunk;
        let start_ms = self.state.cursor_ms;

        // Compute duration from sample count and sample rate
        let duration_ms = if sample_rate > 0 && !samples.is_empty() {
            (samples.len() as i64 * 1000) / i64::from(sample_rate)
        } else {
            self.incremental_config.chunk_duration_ms
        };

        let end_ms = start_ms + duration_ms;

        // Detect silence: compute RMS
        let rms = compute_rms(samples);
        let is_silent = rms < f64::from(self.incremental_config.silence_rms_threshold);

        let (text, confidence, word_confs) = if is_silent {
            (String::new(), 1.0_f32, Vec::new())
        } else {
            self.transcribe_chunk_samples(samples, start_ms, end_ms)
        };

        let is_final = !self.incremental_config.emit_interim_results || !is_silent;

        let mut result = ChunkTranscriptionResult::new(chunk_idx, start_ms, end_ms);
        result.text = text;
        result.confidence = confidence;
        result.is_final = is_final;
        result.word_confidences = word_confs;

        // Update state
        self.state.next_chunk += 1;
        // Advance cursor, accounting for overlap
        let advance = duration_ms - self.incremental_config.overlap_ms;
        self.state.cursor_ms += advance.max(duration_ms);

        self.results.push(result.clone());

        Ok(result)
    }

    /// Process all chunks in a batch.
    ///
    /// Splits `all_samples` into `chunk_duration_ms`-sized windows and
    /// processes each one, returning all results.
    pub fn process_all(
        &mut self,
        all_samples: &[f32],
        sample_rate: u32,
    ) -> crate::error::AccessResult<Vec<ChunkTranscriptionResult>> {
        if all_samples.is_empty() {
            return Ok(Vec::new());
        }

        let samples_per_chunk = if sample_rate > 0 {
            (self.incremental_config.chunk_duration_ms as usize * sample_rate as usize) / 1000
        } else {
            1024
        }
        .max(1);

        let mut all_results = Vec::new();

        for chunk in all_samples.chunks(samples_per_chunk) {
            let result = self.process_chunk(chunk, sample_rate)?;
            all_results.push(result);
        }

        Ok(all_results)
    }

    /// Flush any pending state and finalize all results.
    ///
    /// After flushing, the processor resets to its initial state.
    pub fn flush(&mut self) -> crate::error::AccessResult<Vec<ChunkTranscriptionResult>> {
        // Mark all results as final and return them
        let mut finalized: Vec<ChunkTranscriptionResult> = self
            .results
            .drain(..)
            .map(|mut r| {
                r.is_final = true;
                r
            })
            .collect();

        // Reset state for reuse
        self.state.next_chunk = 0;
        self.state.cursor_ms = 0;
        self.state.pending_text.clear();

        Ok(std::mem::take(&mut finalized))
    }

    /// Get the number of chunks processed so far.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.state.next_chunk
    }

    /// Get current cursor position in milliseconds.
    #[must_use]
    pub fn cursor_ms(&self) -> i64 {
        self.state.cursor_ms
    }

    /// Reset the processor to its initial state.
    pub fn reset(&mut self) {
        self.state.next_chunk = 0;
        self.state.cursor_ms = 0;
        self.state.pending_text.clear();
        self.results.clear();
    }

    /// Get the STT configuration.
    #[must_use]
    pub fn stt_config(&self) -> &SttConfig {
        &self.stt_config
    }

    /// Get the incremental configuration.
    #[must_use]
    pub fn incremental_config(&self) -> &IncrementalSttConfig {
        &self.incremental_config
    }

    /// Internal: perform STT on a chunk of samples.
    fn transcribe_chunk_samples(
        &self,
        samples: &[f32],
        start_ms: i64,
        end_ms: i64,
    ) -> (String, f32, Vec<WordConfidence>) {
        // In production this would call an STT engine.
        // Here we produce a deterministic placeholder based on the RMS of the chunk.
        let rms = compute_rms(samples);
        let duration_ms = end_ms - start_ms;

        // Simulate a word per ~300ms of audio
        let word_count = (duration_ms / 300).max(1) as usize;
        let confidence = (0.7 + rms * 0.3).clamp(0.0, 1.0) as f32;

        let mut words = Vec::new();
        let word_dur = duration_ms / word_count as i64;
        for i in 0..word_count {
            let w_start = start_ms + i as i64 * word_dur;
            let w_end = w_start + word_dur;
            words.push(WordConfidence::new(
                format!("word{}", i + 1),
                w_start,
                w_end,
                confidence,
            ));
        }

        let text = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        (text, confidence, words)
    }
}

impl Default for IncrementalSttProcessor {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Compute the root-mean-square amplitude of a sample slice.
fn compute_rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stt_creation() {
        let stt = SpeechToText::default();
        assert_eq!(stt.config().language, "en");
    }

    #[test]
    fn test_transcribe() {
        let stt = SpeechToText::default();
        let bytes: Vec<u8> = vec![0u8; 48000 * 4];
        let audio = AudioBuffer::Interleaved(Bytes::from(bytes));
        let result = stt.transcribe(&audio);
        assert!(result.is_ok());
    }

    // ============================================================
    // Confidence score tests
    // ============================================================

    #[test]
    fn test_word_confidence_creation() {
        let wc = WordConfidence::new("hello".to_string(), 0, 500, 0.95);
        assert_eq!(wc.word, "hello");
        assert!((wc.confidence - 0.95).abs() < f32::EPSILON);
        assert!(!wc.is_uncertain);
        assert_eq!(wc.duration_ms(), 500);
    }

    #[test]
    fn test_word_confidence_uncertain() {
        let wc = WordConfidence::new("mumble".to_string(), 0, 500, 0.3);
        assert!(wc.is_uncertain);
    }

    #[test]
    fn test_word_confidence_clamping() {
        let wc = WordConfidence::new("test".to_string(), 0, 500, 1.5);
        assert!((wc.confidence - 1.0).abs() < f32::EPSILON);

        let wc2 = WordConfidence::new("test".to_string(), 0, 500, -0.5);
        assert!(wc2.confidence.abs() < f32::EPSILON);
    }

    #[test]
    fn test_transcription_quality_excellent() {
        let words = vec![
            WordConfidence::new("clear".to_string(), 0, 300, 0.98),
            WordConfidence::new("speech".to_string(), 300, 600, 0.96),
            WordConfidence::new("here".to_string(), 600, 900, 0.97),
        ];

        let quality = TranscriptionQuality::from_word_confidences(words, 0.5);
        assert_eq!(quality.quality_level, TranscriptionQualityLevel::Excellent);
        assert!(quality.average_confidence > 0.95);
        assert_eq!(quality.uncertain_word_count, 0);
        assert!((quality.reliable_word_percentage - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transcription_quality_mixed() {
        let words = vec![
            WordConfidence::new("the".to_string(), 0, 200, 0.95),
            WordConfidence::new("mumbled".to_string(), 200, 500, 0.35),
            WordConfidence::new("word".to_string(), 500, 800, 0.90),
            WordConfidence::new("unclear".to_string(), 800, 1100, 0.25),
            WordConfidence::new("ending".to_string(), 1100, 1400, 0.88),
        ];

        let quality = TranscriptionQuality::from_word_confidences(words, 0.5);
        assert_eq!(quality.uncertain_word_count, 2);
        assert_eq!(quality.total_word_count, 5);
        assert!(quality.reliable_word_percentage < 100.0);
        assert!(quality.min_confidence < 0.3);
    }

    #[test]
    fn test_transcription_quality_empty() {
        let quality = TranscriptionQuality::from_word_confidences(Vec::new(), 0.5);
        assert_eq!(quality.quality_level, TranscriptionQualityLevel::VeryPoor);
        assert_eq!(quality.total_word_count, 0);
    }

    #[test]
    fn test_review_regions_detection() {
        let words = vec![
            WordConfidence::new("hello".to_string(), 0, 300, 0.95),
            WordConfidence::new("mumble".to_string(), 300, 600, 0.30),
            WordConfidence::new("unclear".to_string(), 600, 900, 0.25),
            WordConfidence::new("world".to_string(), 900, 1200, 0.92),
        ];

        let quality = TranscriptionQuality::from_word_confidences(words, 0.5);
        assert_eq!(quality.review_regions.len(), 1);

        let region = &quality.review_regions[0];
        assert_eq!(region.start_time_ms, 300);
        assert_eq!(region.end_time_ms, 900);
        assert!(region.text.contains("mumble"));
        assert!(region.text.contains("unclear"));
    }

    #[test]
    fn test_review_regions_at_end() {
        let words = vec![
            WordConfidence::new("hello".to_string(), 0, 300, 0.95),
            WordConfidence::new("mumble".to_string(), 300, 600, 0.30),
        ];

        let quality = TranscriptionQuality::from_word_confidences(words, 0.5);
        assert_eq!(quality.review_regions.len(), 1);
        assert_eq!(quality.review_regions[0].start_time_ms, 300);
    }

    #[test]
    fn test_review_regions_multiple() {
        let words = vec![
            WordConfidence::new("bad".to_string(), 0, 300, 0.20),
            WordConfidence::new("good".to_string(), 300, 600, 0.95),
            WordConfidence::new("bad2".to_string(), 600, 900, 0.15),
        ];

        let quality = TranscriptionQuality::from_word_confidences(words, 0.5);
        assert_eq!(quality.review_regions.len(), 2);
    }

    #[test]
    fn test_meets_quality_level() {
        let words = vec![
            WordConfidence::new("test".to_string(), 0, 300, 0.90),
            WordConfidence::new("words".to_string(), 300, 600, 0.85),
        ];

        let quality = TranscriptionQuality::from_word_confidences(words, 0.5);
        assert!(quality.meets_quality(TranscriptionQualityLevel::Fair));
        assert!(quality.meets_quality(TranscriptionQualityLevel::Good));
        assert!(!quality.meets_quality(TranscriptionQualityLevel::Excellent));
    }

    #[test]
    fn test_uncertain_words_filter() {
        let words = vec![
            WordConfidence::new("clear".to_string(), 0, 300, 0.95),
            WordConfidence::new("fuzzy".to_string(), 300, 600, 0.60),
            WordConfidence::new("noise".to_string(), 600, 900, 0.30),
        ];

        let quality = TranscriptionQuality::from_word_confidences(words, 0.5);

        let uncertain = quality.uncertain_words(0.7);
        assert_eq!(uncertain.len(), 2); // fuzzy (0.6) and noise (0.3)

        let very_uncertain = quality.uncertain_words(0.4);
        assert_eq!(very_uncertain.len(), 1); // only noise (0.3)
    }

    #[test]
    fn test_transcribe_with_confidence() {
        let stt = SpeechToText::default();
        let audio = AudioBuffer::Interleaved(Bytes::from(vec![0u8; 1000]));

        let (transcript, quality) = stt
            .transcribe_with_confidence(&audio)
            .expect("should succeed");

        assert!(!transcript.entries.is_empty());
        assert!(quality.total_word_count > 0);
        assert!(quality.average_confidence > 0.0);
    }

    #[test]
    fn test_assess_quality() {
        let stt = SpeechToText::default().with_confidence_threshold(0.5);

        let mut transcript = Transcript::new();
        let mut entry = TranscriptEntry::new(0, 2000, "hello world test".to_string());
        entry.confidence = 0.85;
        transcript.add_entry(entry);

        let quality = stt.assess_quality(&transcript).expect("should succeed");

        assert_eq!(quality.total_word_count, 3);
        assert!((quality.average_confidence - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_assess_quality_empty() {
        let stt = SpeechToText::default();
        let transcript = Transcript::new();
        let result = stt.assess_quality(&transcript);
        assert!(result.is_err());
    }

    #[test]
    fn test_quality_level_display() {
        assert_eq!(
            format!("{}", TranscriptionQualityLevel::Excellent),
            "Excellent"
        );
        assert_eq!(format!("{}", TranscriptionQualityLevel::Good), "Good");
        assert_eq!(format!("{}", TranscriptionQualityLevel::Fair), "Fair");
        assert_eq!(format!("{}", TranscriptionQualityLevel::Poor), "Poor");
        assert_eq!(
            format!("{}", TranscriptionQualityLevel::VeryPoor),
            "Very Poor"
        );
    }

    #[test]
    fn test_confidence_threshold_setting() {
        let stt = SpeechToText::default().with_confidence_threshold(0.7);
        assert!((stt.confidence_threshold() - 0.7).abs() < f32::EPSILON);
    }

    // ============================================================
    // Incremental STT processing tests
    // ============================================================

    #[test]
    fn test_incremental_processor_creation() {
        let processor = IncrementalSttProcessor::with_defaults();
        assert_eq!(processor.chunk_count(), 0);
        assert_eq!(processor.cursor_ms(), 0);
    }

    #[test]
    fn test_incremental_process_single_chunk() {
        let mut processor = IncrementalSttProcessor::with_defaults();
        // 1 second of 0.5 amplitude audio at 48 kHz
        let samples: Vec<f32> = vec![0.5_f32; 48000];
        let result = processor
            .process_chunk(&samples, 48000)
            .expect("should succeed");

        assert_eq!(result.chunk_index, 0);
        assert_eq!(result.start_time_ms, 0);
        assert!(!result.text.is_empty()); // Non-silent audio should produce text
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_incremental_process_silent_chunk() {
        let mut processor = IncrementalSttProcessor::with_defaults();
        // Silent audio (all zeros)
        let samples: Vec<f32> = vec![0.0_f32; 48000];
        let result = processor
            .process_chunk(&samples, 48000)
            .expect("should succeed");

        // Silent chunk should produce empty text
        assert!(result.text.is_empty());
        assert_eq!(result.chunk_index, 0);
    }

    #[test]
    fn test_incremental_chunk_indices_increment() {
        let mut processor = IncrementalSttProcessor::with_defaults();
        let samples: Vec<f32> = vec![0.5_f32; 1000];

        let r0 = processor.process_chunk(&samples, 48000).expect("ok");
        let r1 = processor.process_chunk(&samples, 48000).expect("ok");
        let r2 = processor.process_chunk(&samples, 48000).expect("ok");

        assert_eq!(r0.chunk_index, 0);
        assert_eq!(r1.chunk_index, 1);
        assert_eq!(r2.chunk_index, 2);
        assert_eq!(processor.chunk_count(), 3);
    }

    #[test]
    fn test_incremental_chunk_timing_non_overlapping() {
        let config = IncrementalSttConfig {
            overlap_ms: 0,
            ..IncrementalSttConfig::default()
        };
        let mut processor = IncrementalSttProcessor::new(SttConfig::default(), config);
        let samples: Vec<f32> = vec![0.5_f32; 24000]; // 500ms at 48kHz

        let r0 = processor.process_chunk(&samples, 48000).expect("ok");
        let r1 = processor.process_chunk(&samples, 48000).expect("ok");

        assert_eq!(r0.start_time_ms, 0);
        assert_eq!(r0.end_time_ms, 500);
        assert_eq!(r1.start_time_ms, 500);
    }

    #[test]
    fn test_incremental_process_all() {
        let mut processor = IncrementalSttProcessor::new(
            SttConfig::default(),
            IncrementalSttConfig {
                chunk_duration_ms: 100,
                overlap_ms: 0,
                ..IncrementalSttConfig::default()
            },
        );

        // 1 second of audio at 8 kHz
        let samples: Vec<f32> = vec![0.5_f32; 8000];
        let results = processor
            .process_all(&samples, 8000)
            .expect("should succeed");

        assert!(!results.is_empty());
        // Should have approximately 10 chunks (1000ms / 100ms)
        // Exact count depends on chunk alignment
        assert!(results.len() >= 9);
    }

    #[test]
    fn test_incremental_flush() {
        let mut processor = IncrementalSttProcessor::with_defaults();
        let samples: Vec<f32> = vec![0.5_f32; 1000];
        processor.process_chunk(&samples, 48000).expect("ok");
        processor.process_chunk(&samples, 48000).expect("ok");

        let flushed = processor.flush().expect("flush ok");
        assert_eq!(flushed.len(), 2);
        assert!(flushed.iter().all(|r| r.is_final));

        // After flush, processor should be reset
        assert_eq!(processor.chunk_count(), 0);
    }

    #[test]
    fn test_incremental_reset() {
        let mut processor = IncrementalSttProcessor::with_defaults();
        let samples: Vec<f32> = vec![0.5_f32; 1000];
        processor.process_chunk(&samples, 48000).expect("ok");
        assert_eq!(processor.chunk_count(), 1);

        processor.reset();
        assert_eq!(processor.chunk_count(), 0);
        assert_eq!(processor.cursor_ms(), 0);
    }

    #[test]
    fn test_incremental_chunk_has_word_confidences() {
        let mut processor = IncrementalSttProcessor::with_defaults();
        // Loud audio - should produce word confidences
        let samples: Vec<f32> = vec![0.8_f32; 24000];
        let result = processor.process_chunk(&samples, 48000).expect("ok");

        assert!(!result.word_confidences.is_empty());
    }

    #[test]
    fn test_chunk_result_duration() {
        let result = ChunkTranscriptionResult::new(0, 1000, 2000);
        assert_eq!(result.duration_ms(), 1000);
    }

    #[test]
    fn test_chunk_result_finalized() {
        let result = ChunkTranscriptionResult::new(0, 0, 500).finalized();
        assert!(result.is_final);
    }

    #[test]
    fn test_incremental_config_defaults() {
        let config = IncrementalSttConfig::default();
        assert_eq!(config.chunk_duration_ms, 500);
        assert_eq!(config.overlap_ms, 50);
        assert!(!config.emit_interim_results);
    }

    #[test]
    fn test_incremental_config_builder() {
        let config = IncrementalSttConfig::default()
            .with_chunk_duration(250)
            .with_interim_results(true);
        assert_eq!(config.chunk_duration_ms, 250);
        assert!(config.emit_interim_results);
    }

    #[test]
    fn test_compute_rms_silence() {
        let samples = vec![0.0_f32; 100];
        assert!(compute_rms(&samples) < f64::EPSILON);
    }

    #[test]
    fn test_compute_rms_nonzero() {
        let samples = vec![1.0_f32; 100];
        assert!((compute_rms(&samples) - 1.0).abs() < 1e-9);
    }
}
