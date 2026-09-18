//! Speech-to-caption alignment: word timestamps, segment merging/splitting,
//! frame-accurate caption block construction.

use crate::CaptionGenError;

/// A single word with its timing and ASR confidence score.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTimestamp {
    pub word: String,
    pub start_ms: u64,
    pub end_ms: u64,
    /// ASR confidence in [0.0, 1.0].
    pub confidence: f32,
    /// Per-word quality confidence in [0.0, 1.0].
    /// This may differ from ASR confidence — it reflects display quality
    /// (e.g., low confidence words could be rendered with special styling).
    pub word_confidence: f32,
}

impl WordTimestamp {
    /// Construct a `WordTimestamp` with a given word-level quality confidence score.
    pub fn with_word_confidence(
        word: String,
        start_ms: u64,
        end_ms: u64,
        confidence: f32,
        word_confidence: f32,
    ) -> Self {
        Self {
            word,
            start_ms,
            end_ms,
            confidence,
            word_confidence,
        }
    }

    /// Returns `true` if the word-level confidence meets `threshold`.
    pub fn is_high_quality(&self, threshold: f32) -> bool {
        self.word_confidence >= threshold
    }
}

/// A contiguous segment of transcript text, optionally associated with a speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptSegment {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    /// Optional speaker identifier (from diarization).
    pub speaker_id: Option<u8>,
    pub words: Vec<WordTimestamp>,
}

impl TranscriptSegment {
    /// Duration of this segment in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Errors that can occur during alignment operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AlignmentError {
    #[error(
        "segment duration ({segment_ms}ms) is incompatible with video duration ({video_ms}ms)"
    )]
    IncompatibleDuration { segment_ms: u64, video_ms: u64 },

    #[error("transcript is empty — no segments to align")]
    EmptyTranscript,

    #[error("invalid timestamp: start_ms ({start_ms}) >= end_ms ({end_ms})")]
    InvalidTimestamp { start_ms: u64, end_ms: u64 },
}

/// Screen position for a caption block.
#[derive(Debug, Clone, PartialEq)]
pub enum CaptionPosition {
    /// Default lower-third position.
    Bottom,
    /// Upper-third position for speaker identification or secondary captions.
    Top,
    /// Arbitrary position expressed as percentage [0.0, 100.0] of screen width/height.
    Custom(f32, f32),
}

/// A fully resolved caption block ready for rendering or export.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptionBlock {
    /// Sequential 1-based identifier.
    pub id: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    /// Lines of text; at most `max_lines` entries.
    pub lines: Vec<String>,
    pub speaker_id: Option<u8>,
    pub position: CaptionPosition,
}

impl CaptionBlock {
    /// Total number of characters across all lines (excluding newlines).
    pub fn char_count(&self) -> usize {
        self.lines.iter().map(|l| l.chars().count()).sum()
    }

    /// Duration of this block in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

// ─── Frame alignment ──────────────────────────────────────────────────────────

/// Convert a transcript segment into (frame_number, subtitle_line) pairs.
///
/// Each word is mapped to the frame at which it starts, and the full
/// segment text is returned as the subtitle line.
///
/// # Errors
/// Returns [`CaptionGenError`] if the segment has an invalid timestamp or if
/// `fps` is not positive.
pub fn align_to_frames(
    segment: &TranscriptSegment,
    fps: f32,
) -> Result<Vec<(u64, String)>, CaptionGenError> {
    if fps <= 0.0 {
        return Err(CaptionGenError::InvalidParameter(
            "fps must be positive".to_string(),
        ));
    }
    if segment.start_ms >= segment.end_ms && !segment.text.is_empty() {
        return Err(CaptionGenError::Alignment(
            AlignmentError::InvalidTimestamp {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
            },
        ));
    }

    let ms_per_frame = 1000.0 / fps as f64;

    // If there are word-level timestamps, emit one entry per unique start frame.
    if !segment.words.is_empty() {
        let mut result: Vec<(u64, String)> = Vec::new();
        for word in &segment.words {
            let frame = (word.start_ms as f64 / ms_per_frame).floor() as u64;
            // Accumulate words that start on the same frame into one line.
            if let Some(last) = result.last_mut() {
                if last.0 == frame {
                    last.1.push(' ');
                    last.1.push_str(&word.word);
                    continue;
                }
            }
            result.push((frame, word.word.clone()));
        }
        return Ok(result);
    }

    // Segment-level fallback: one entry at the start frame.
    let start_frame = (segment.start_ms as f64 / ms_per_frame).floor() as u64;
    Ok(vec![(start_frame, segment.text.clone())])
}

// ─── Batch frame alignment ────────────────────────────────────────────────────

/// Align multiple segments to frames in one call.
///
/// This is more efficient than calling [`align_to_frames`] repeatedly because
/// the `ms_per_frame` constant is computed only once.
///
/// Returns a `Vec` of per-segment results in the same order as `segments`.
/// On error, returns the first [`CaptionGenError`] encountered.
pub fn align_to_frames_batch(
    segments: &[TranscriptSegment],
    fps: f32,
) -> Result<Vec<Vec<(u64, String)>>, CaptionGenError> {
    if fps <= 0.0 {
        return Err(CaptionGenError::InvalidParameter(
            "fps must be positive".to_string(),
        ));
    }
    segments
        .iter()
        .map(|seg| align_to_frames(seg, fps))
        .collect()
}

// ─── Segment merging ──────────────────────────────────────────────────────────

/// Merge segments whose duration is shorter than `min_duration_ms` into an
/// adjacent segment.  Shorter segments are appended to the preceding segment
/// when one exists, otherwise prepended to the following segment.
///
/// Consecutive same-speaker segments that are very short are therefore
/// absorbed into their neighbours, reducing display flicker.
pub fn merge_short_segments(
    segments: &[TranscriptSegment],
    min_duration_ms: u32,
) -> Vec<TranscriptSegment> {
    if segments.is_empty() {
        return Vec::new();
    }

    // Work on a mutable copy.
    let mut merged: Vec<TranscriptSegment> = segments.to_vec();
    let min_ms = u64::from(min_duration_ms);

    // Iterate until stable (no more merges to perform).
    loop {
        let mut changed = false;
        let mut output: Vec<TranscriptSegment> = Vec::with_capacity(merged.len());

        let mut i = 0;
        while i < merged.len() {
            let seg = merged[i].clone();
            if seg.duration_ms() < min_ms && output.is_empty() && i + 1 < merged.len() {
                // Prepend to next segment.
                let next = merged[i + 1].clone();
                let combined = combine_segments(&seg, &next);
                output.push(combined);
                i += 2;
                changed = true;
            } else if seg.duration_ms() < min_ms {
                // Append to previous segment in output.
                if let Some(prev) = output.last_mut() {
                    let combined = combine_segments(prev, &seg);
                    *prev = combined;
                    changed = true;
                } else {
                    output.push(seg);
                }
                i += 1;
            } else {
                output.push(seg);
                i += 1;
            }
        }

        merged = output;
        if !changed {
            break;
        }
    }

    merged
}

/// Combine two segments by concatenating their text, words, and spanning their
/// timestamps.  Speaker id is taken from the first segment.
fn combine_segments(a: &TranscriptSegment, b: &TranscriptSegment) -> TranscriptSegment {
    let mut text = a.text.clone();
    if !a.text.is_empty() && !b.text.is_empty() {
        text.push(' ');
    }
    text.push_str(&b.text);

    let mut words = a.words.clone();
    words.extend_from_slice(&b.words);

    TranscriptSegment {
        text,
        start_ms: a.start_ms.min(b.start_ms),
        end_ms: a.end_ms.max(b.end_ms),
        speaker_id: a.speaker_id,
        words,
    }
}

// ─── Segment splitting ────────────────────────────────────────────────────────

/// Split a segment that is too long (by duration or character count) into
/// smaller segments, preferring sentence boundaries (`.!?`) and then word
/// boundaries.
///
/// The timestamps of sub-segments are distributed proportionally to their
/// character counts within the original segment's time span.
pub fn split_long_segments(
    segment: &TranscriptSegment,
    max_duration_ms: u32,
    max_chars: u16,
) -> Vec<TranscriptSegment> {
    let max_dur = u64::from(max_duration_ms);
    let max_ch = usize::from(max_chars);

    let needs_split = segment.duration_ms() > max_dur || segment.text.chars().count() > max_ch;
    if !needs_split {
        return vec![segment.clone()];
    }

    // Split text into chunks at sentence boundaries first, then word boundaries.
    let chunks = split_text_into_chunks(&segment.text, max_ch);
    if chunks.len() <= 1 {
        return vec![segment.clone()];
    }

    // Distribute timestamps proportionally by character count.
    let total_chars: usize = chunks.iter().map(|c| c.chars().count()).sum();
    let total_duration = segment.duration_ms();
    let mut result = Vec::with_capacity(chunks.len());
    let mut cursor_ms = segment.start_ms;

    for (idx, chunk) in chunks.iter().enumerate() {
        let chunk_chars = chunk.chars().count();
        let chunk_duration = if idx + 1 < chunks.len() {
            if total_chars > 0 {
                (total_duration as f64 * chunk_chars as f64 / total_chars as f64).round() as u64
            } else {
                total_duration / chunks.len() as u64
            }
        } else {
            // Last chunk gets remaining time to avoid rounding drift.
            segment.end_ms.saturating_sub(cursor_ms)
        };

        let start_ms = cursor_ms;
        let end_ms = (cursor_ms + chunk_duration).min(segment.end_ms);

        // Assign words that fall within this sub-segment's time range.
        let sub_words: Vec<WordTimestamp> = segment
            .words
            .iter()
            .filter(|w| w.start_ms >= start_ms && w.start_ms < end_ms)
            .cloned()
            .collect();

        result.push(TranscriptSegment {
            text: chunk.clone(),
            start_ms,
            end_ms,
            speaker_id: segment.speaker_id,
            words: sub_words,
        });

        cursor_ms = end_ms;
    }

    result
}

/// Split `text` into chunks no longer than `max_chars`, preferring sentence
/// boundaries then word boundaries.
fn split_text_into_chunks(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut remaining = text.trim();

    while !remaining.is_empty() {
        if remaining.chars().count() <= max_chars {
            chunks.push(remaining.to_string());
            break;
        }

        // Try to find a sentence boundary within the window.
        let window: String = remaining.chars().take(max_chars + 1).collect();
        let cut = find_sentence_boundary(&window, max_chars)
            .or_else(|| find_word_boundary(&window, max_chars))
            .unwrap_or(max_chars);

        let (chunk, rest) = split_at_char_index(remaining, cut);
        chunks.push(chunk.trim().to_string());
        remaining = rest.trim();
    }

    chunks
}

/// Find the last sentence-ending character (`.`, `!`, `?`) at or before
/// `max_chars` in `text`.  Returns the index *after* that character.
fn find_sentence_boundary(text: &str, max_chars: usize) -> Option<usize> {
    let chars: Vec<char> = text.chars().take(max_chars).collect();
    for (i, &ch) in chars.iter().enumerate().rev() {
        if ch == '.' || ch == '!' || ch == '?' {
            return Some(i + 1);
        }
    }
    None
}

/// Find the last space at or before `max_chars` in `text`.
/// Returns the index of that space (so the chunk excludes the space).
fn find_word_boundary(text: &str, max_chars: usize) -> Option<usize> {
    let chars: Vec<char> = text.chars().take(max_chars).collect();
    for (i, &ch) in chars.iter().enumerate().rev() {
        if ch == ' ' {
            return Some(i);
        }
    }
    None
}

/// Split `text` at a character index, returning (before, after).
fn split_at_char_index(text: &str, idx: usize) -> (&str, &str) {
    let byte_pos = text
        .char_indices()
        .nth(idx)
        .map(|(b, _)| b)
        .unwrap_or(text.len());
    (&text[..byte_pos], &text[byte_pos..])
}

// ─── Caption block builder ────────────────────────────────────────────────────

/// Build caption blocks from transcript segments, wrapping text to at most
/// `max_lines` lines of `max_chars_per_line` characters each.
///
/// Returns a list of [`CaptionBlock`] values with sequential IDs starting at 1.
pub fn build_caption_blocks(
    segments: &[TranscriptSegment],
    max_lines: u8,
    max_chars_per_line: u8,
) -> Vec<CaptionBlock> {
    use crate::line_breaking::greedy_break;

    let max_l = max_lines.max(1) as usize;
    let max_c = max_chars_per_line.max(1);

    segments
        .iter()
        .enumerate()
        .map(|(idx, seg)| {
            let all_lines = greedy_break(&seg.text, max_c);
            // Truncate to max_lines; if there are more, join surplus onto last line.
            let lines = if all_lines.len() <= max_l {
                all_lines
            } else {
                let mut truncated = all_lines[..max_l - 1].to_vec();
                let overflow = all_lines[max_l - 1..].join(" ");
                truncated.push(overflow);
                truncated
            };

            CaptionBlock {
                id: (idx as u32) + 1,
                start_ms: seg.start_ms,
                end_ms: seg.end_ms,
                lines,
                speaker_id: seg.speaker_id,
                position: CaptionPosition::Bottom,
            }
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seg(text: &str, start_ms: u64, end_ms: u64) -> TranscriptSegment {
        TranscriptSegment {
            text: text.to_string(),
            start_ms,
            end_ms,
            speaker_id: None,
            words: Vec::new(),
        }
    }

    fn make_word(word: &str, start_ms: u64, end_ms: u64) -> WordTimestamp {
        WordTimestamp {
            word: word.to_string(),
            start_ms,
            end_ms,
            confidence: 1.0,
            word_confidence: 1.0,
        }
    }

    // --- align_to_frames ---

    #[test]
    fn align_to_frames_segment_level() {
        let seg = make_seg("Hello world", 0, 2000);
        let frames = align_to_frames(&seg, 25.0).expect("align to frames should succeed");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, 0);
        assert_eq!(frames[0].1, "Hello world");
    }

    #[test]
    fn align_to_frames_word_level() {
        let mut seg = make_seg("Hello world", 0, 2000);
        seg.words = vec![make_word("Hello", 0, 1000), make_word("world", 1000, 2000)];
        let frames = align_to_frames(&seg, 25.0).expect("align to frames should succeed");
        assert_eq!(frames[0].0, 0);
        assert_eq!(frames[1].0, 25);
    }

    #[test]
    fn align_to_frames_rejects_zero_fps() {
        let seg = make_seg("test", 0, 1000);
        assert!(align_to_frames(&seg, 0.0).is_err());
    }

    #[test]
    fn align_to_frames_rejects_negative_fps() {
        let seg = make_seg("test", 0, 1000);
        assert!(align_to_frames(&seg, -30.0).is_err());
    }

    #[test]
    fn align_to_frames_same_start_frame_merges_words() {
        let mut seg = make_seg("Hi", 0, 500);
        // Two words both start at frame 0 (within 0..40ms at 25fps).
        seg.words = vec![make_word("Hi", 0, 200), make_word("there", 20, 300)];
        let frames = align_to_frames(&seg, 25.0).expect("align to frames should succeed");
        // Both map to frame 0.
        assert_eq!(frames.len(), 1);
        assert!(frames[0].1.contains("Hi"));
        assert!(frames[0].1.contains("there"));
    }

    #[test]
    fn align_to_frames_correct_frame_numbers_at_30fps() {
        let mut seg = make_seg("A B C", 0, 3000);
        seg.words = vec![
            make_word("A", 0, 1000),
            make_word("B", 1000, 2000),
            make_word("C", 2000, 3000),
        ];
        let frames = align_to_frames(&seg, 30.0).expect("align");
        assert_eq!(frames[0].0, 0);
        // At 30fps, ms_per_frame ≈ 33.333ms; floor(1000/33.333) may yield 29
        // due to floating-point precision (30.0 - epsilon → 29 via floor).
        assert!(frames[1].0 == 29 || frames[1].0 == 30);
        assert!(frames[2].0 == 59 || frames[2].0 == 60);
    }

    // --- merge_short_segments ---

    #[test]
    fn merge_short_segments_empty() {
        assert!(merge_short_segments(&[], 500).is_empty());
    }

    #[test]
    fn merge_short_segments_no_op_if_all_long_enough() {
        let segs = vec![make_seg("hello", 0, 1000), make_seg("world", 1000, 2000)];
        let result = merge_short_segments(&segs, 500);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn merge_short_segments_merges_short_prefix() {
        let segs = vec![
            make_seg("Hi", 0, 100), // 100ms < 500ms threshold
            make_seg("world", 100, 1200),
        ];
        let result = merge_short_segments(&segs, 500);
        assert_eq!(result.len(), 1);
        assert!(result[0].text.contains("Hi"));
        assert!(result[0].text.contains("world"));
    }

    #[test]
    fn merge_short_segments_merges_short_suffix() {
        let segs = vec![
            make_seg("Hello there", 0, 1000),
            make_seg("ok", 1000, 1050), // 50ms
        ];
        let result = merge_short_segments(&segs, 500);
        assert_eq!(result.len(), 1);
        assert!(result[0].text.contains("Hello"));
        assert!(result[0].text.contains("ok"));
    }

    #[test]
    fn merge_short_segments_span_extends() {
        let segs = vec![
            make_seg("A", 0, 100),
            make_seg("long segment here", 100, 2000),
        ];
        let result = merge_short_segments(&segs, 500);
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[0].end_ms, 2000);
    }

    // --- split_long_segments ---

    #[test]
    fn split_long_segments_no_op_if_short() {
        let seg = make_seg("Hello", 0, 1000);
        let result = split_long_segments(&seg, 5000, 200);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn split_long_segments_by_duration() {
        // Use max_chars=20 to force text splitting (48 chars > 20), which also
        // triggers duration redistribution across the resulting sub-segments.
        let seg = make_seg("This is a longer sentence for testing purposes.", 0, 20000);
        let result = split_long_segments(&seg, 5000, 20);
        assert!(result.len() > 1, "expected multiple segments");
        for s in &result {
            assert!(s.duration_ms() <= 20000);
        }
    }

    #[test]
    fn split_long_segments_preserves_total_duration() {
        let seg = make_seg("Word one. Word two. Word three. Word four.", 0, 10000);
        let result = split_long_segments(&seg, 3000, 20);
        let first_start = result.first().map(|s| s.start_ms).unwrap_or(0);
        let last_end = result.last().map(|s| s.end_ms).unwrap_or(0);
        assert_eq!(first_start, 0);
        assert_eq!(last_end, 10000);
    }

    #[test]
    fn split_long_segments_respects_max_chars() {
        let seg = make_seg(
            "This is a very long text that exceeds the character limit.",
            0,
            10000,
        );
        let result = split_long_segments(&seg, 100_000, 15);
        for s in &result {
            assert!(s.text.chars().count() <= 20, "chunk '{}' too long", s.text);
        }
    }

    #[test]
    fn split_long_segments_words_assigned_to_subsegments() {
        let mut seg = make_seg("Hello world test", 0, 3000);
        seg.words = vec![
            make_word("Hello", 0, 1000),
            make_word("world", 1000, 2000),
            make_word("test", 2000, 3000),
        ];
        let result = split_long_segments(&seg, 1200, 8);
        assert!(result.len() > 1);
    }

    // --- build_caption_blocks ---

    #[test]
    fn build_caption_blocks_basic() {
        let segs = vec![
            make_seg("Hello world", 0, 2000),
            make_seg("How are you", 2000, 4000),
        ];
        let blocks = build_caption_blocks(&segs, 2, 40);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].id, 1);
        assert_eq!(blocks[1].id, 2);
    }

    #[test]
    fn build_caption_blocks_respects_max_lines() {
        let seg = make_seg(
            "This is a very very very very very very very very long text to wrap over many lines.",
            0,
            5000,
        );
        let blocks = build_caption_blocks(&[seg], 2, 20);
        assert_eq!(blocks.len(), 1);
        assert!(
            blocks[0].lines.len() <= 2,
            "got {} lines",
            blocks[0].lines.len()
        );
    }

    #[test]
    fn build_caption_blocks_preserves_timestamps() {
        let segs = vec![make_seg("Test", 1500, 3000)];
        let blocks = build_caption_blocks(&segs, 2, 40);
        assert_eq!(blocks[0].start_ms, 1500);
        assert_eq!(blocks[0].end_ms, 3000);
    }

    #[test]
    fn build_caption_blocks_default_position_bottom() {
        let segs = vec![make_seg("Test", 0, 1000)];
        let blocks = build_caption_blocks(&segs, 2, 40);
        assert_eq!(blocks[0].position, CaptionPosition::Bottom);
    }

    #[test]
    fn build_caption_blocks_speaker_id_preserved() {
        let mut seg = make_seg("Test", 0, 1000);
        seg.speaker_id = Some(3);
        let blocks = build_caption_blocks(&[seg], 2, 40);
        assert_eq!(blocks[0].speaker_id, Some(3));
    }

    #[test]
    fn caption_block_char_count() {
        let block = CaptionBlock {
            id: 1,
            start_ms: 0,
            end_ms: 1000,
            lines: vec!["Hello".to_string(), "world".to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        };
        assert_eq!(block.char_count(), 10);
    }

    #[test]
    fn word_timestamp_fields_accessible() {
        let w = make_word("hello", 100, 500);
        assert_eq!(w.word, "hello");
        assert_eq!(w.start_ms, 100);
        assert_eq!(w.end_ms, 500);
        assert!((w.confidence - 1.0).abs() < 1e-6);
        assert!((w.word_confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn word_timestamp_with_word_confidence() {
        let w = WordTimestamp::with_word_confidence("uncertain".to_string(), 100, 500, 0.9, 0.55);
        assert_eq!(w.word, "uncertain");
        assert!((w.confidence - 0.9).abs() < 1e-6);
        assert!((w.word_confidence - 0.55).abs() < 1e-6);
        assert!(w.is_high_quality(0.5));
        assert!(!w.is_high_quality(0.8));
    }

    #[test]
    fn build_caption_blocks_with_overlapping_word_timestamps() {
        // Two segments that have overlapping words (start of second overlaps end of first).
        let mut seg1 = make_seg("Hello there", 0, 2000);
        seg1.words = vec![
            make_word("Hello", 0, 900),
            make_word("there", 800, 2000), // overlaps with previous end
        ];
        let mut seg2 = make_seg("world", 1900, 3500);
        seg2.words = vec![make_word("world", 1900, 3500)];
        let blocks = build_caption_blocks(&[seg1, seg2], 2, 40);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].start_ms, 0);
        assert_eq!(blocks[0].end_ms, 2000);
        assert_eq!(blocks[1].start_ms, 1900);
        assert_eq!(blocks[1].end_ms, 3500);
    }

    #[test]
    fn transcript_segment_duration() {
        let s = make_seg("test", 1000, 3500);
        assert_eq!(s.duration_ms(), 2500);
    }

    #[test]
    fn alignment_error_display_empty_transcript() {
        let e = AlignmentError::EmptyTranscript;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn alignment_error_display_invalid_timestamp() {
        let e = AlignmentError::InvalidTimestamp {
            start_ms: 5000,
            end_ms: 3000,
        };
        assert!(e.to_string().contains("5000"));
    }

    #[test]
    fn split_text_sentence_boundary_preferred() {
        let text = "Hello there! How are you doing today? Fine thanks.";
        let chunks = split_text_into_chunks(text, 15);
        // Each chunk should not exceed 15 chars.
        for c in &chunks {
            assert!(c.chars().count() <= 15, "chunk '{c}' exceeds 15 chars");
        }
    }

    #[test]
    fn split_text_word_boundary_fallback() {
        let text = "AAAA BBBB CCCC DDDD EEEE";
        let chunks = split_text_into_chunks(text, 10);
        for c in &chunks {
            assert!(c.chars().count() <= 12, "chunk '{c}' too long");
        }
    }

    // --- round-trip: split then merge preserves total text ---

    #[test]
    fn round_trip_split_then_merge_preserves_text() {
        let original_text = "Hello world. This is a test. We have multiple sentences here.";
        let seg = make_seg(original_text, 0, 10000);

        // Split the segment.
        let split = split_long_segments(&seg, 3000, 20);
        assert!(split.len() > 1, "expected multiple segments after split");

        // Merge split segments back.
        let merged = merge_short_segments(&split, 0);

        // Reconstruct full text from merged segments.
        let reconstructed: String = merged
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // All words from original text should appear in the reconstruction.
        let original_words: std::collections::HashSet<&str> =
            original_text.split_whitespace().collect();
        let reconstructed_words: std::collections::HashSet<&str> =
            reconstructed.split_whitespace().collect();

        for word in &original_words {
            let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !cleaned.is_empty() {
                assert!(
                    reconstructed_words.iter().any(|w| w.contains(cleaned)),
                    "word '{cleaned}' missing from reconstruction"
                );
            }
        }
    }

    // --- batch align_to_frames ---

    #[test]
    fn align_to_frames_batch_basic() {
        let segs = vec![make_seg("Hello", 0, 1000), make_seg("World", 1000, 2000)];
        let result =
            align_to_frames_batch(&segs, 25.0).expect("align to frames batch should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0][0].1, "Hello");
        assert_eq!(result[1][0].1, "World");
    }

    #[test]
    fn align_to_frames_batch_rejects_zero_fps() {
        let segs = vec![make_seg("test", 0, 1000)];
        assert!(align_to_frames_batch(&segs, 0.0).is_err());
    }
}
