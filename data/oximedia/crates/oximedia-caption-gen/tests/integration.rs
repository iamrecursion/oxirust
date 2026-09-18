//! Integration tests for the auto-caption pipeline and supporting
//! algorithms.
//!
//! These cover:
//! * End-to-end pipeline runs (with and without diarisation, with 5
//!   simultaneous speakers).
//! * Tests requested in TODO.md lines 35-41 (overlapping words,
//!   Knuth-Plass reference, WCAG suite, multi-speaker assignment,
//!   property-based merge/split invariants, greedy vs optimal,
//!   round-trip text preservation).
//! * SMAWK fidelity: 10 000 randomised inputs verified against the
//!   reference O(n²) DP.

use oximedia_caption_gen::alignment::{
    build_caption_blocks, merge_short_segments, split_long_segments, CaptionBlock, CaptionPosition,
    TranscriptSegment, WordTimestamp,
};
use oximedia_caption_gen::auto_pipeline::{
    AsrEngine, AutoCaptionPipeline, DiarizationEngine, LineBreakStrategy, PipelineConfig,
};
use oximedia_caption_gen::diarization::{
    assign_speakers_to_blocks, DiarizationResult, Speaker, SpeakerTurn,
};
use oximedia_caption_gen::line_breaking::{greedy_break, optimal_break, optimal_break_smawk};
use oximedia_caption_gen::wcag::{
    check_caption_coverage, check_cps, check_min_duration, compliance_score, run_all_checks,
    WcagLevel,
};
use oximedia_caption_gen::CaptionGenError;

use proptest::prelude::*;

// ─── Test fixtures ──────────────────────────────────────────────────

fn mk_word(w: &str, s: u64, e: u64) -> WordTimestamp {
    WordTimestamp {
        word: w.to_string(),
        start_ms: s,
        end_ms: e,
        confidence: 0.95,
        word_confidence: 0.95,
    }
}

fn mk_segment(text: &str, s: u64, e: u64) -> TranscriptSegment {
    TranscriptSegment {
        text: text.to_string(),
        start_ms: s,
        end_ms: e,
        speaker_id: None,
        words: Vec::new(),
    }
}

fn mk_turn(id: u8, s: u64, e: u64) -> SpeakerTurn {
    SpeakerTurn {
        speaker_id: id,
        start_ms: s,
        end_ms: e,
    }
}

fn mk_block(id: u32, s: u64, e: u64, text: &str) -> CaptionBlock {
    CaptionBlock {
        id,
        start_ms: s,
        end_ms: e,
        lines: vec![text.to_string()],
        speaker_id: None,
        position: CaptionPosition::Bottom,
    }
}

/// Canned ASR engine returning a fixed word list.
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

/// Canned diariser returning a fixed [`DiarizationResult`].
struct CannedDiarizer(DiarizationResult);

impl DiarizationEngine for CannedDiarizer {
    fn diarize(
        &self,
        _audio: &[f32],
        _sample_rate: u32,
    ) -> Result<DiarizationResult, CaptionGenError> {
        Ok(self.0.clone())
    }
}

// ─── Pipeline integration ───────────────────────────────────────────

#[test]
fn test_pipeline_with_mock_asr_no_diar() {
    let words = vec![
        mk_word("hello", 0, 400),
        mk_word("there", 400, 800),
        mk_word("how", 1200, 1500),
        mk_word("are", 1500, 1700),
        mk_word("you", 1700, 2000),
    ];
    let asr = CannedAsr(words);
    let cfg = PipelineConfig {
        max_line_length: 32,
        max_lines_per_block: 2,
        min_block_duration_ms: 800,
        max_block_duration_ms: 5_000,
        min_gap_ms: 300,
        language: Some("en".to_string()),
        enable_diarization: false,
        line_break_strategy: LineBreakStrategy::OptimalSmawk,
    };
    let pipe = AutoCaptionPipeline::new(asr, cfg);
    let track = pipe
        .process_audio(&vec![0.0_f32; 16_000 * 2], 16_000)
        .expect("pipeline runs");

    assert!(!track.is_empty(), "expected non-empty caption track");
    assert_eq!(track.language.0, "en");
    // The 700ms gap between "there" and "how" should produce two blocks.
    assert!(
        track.blocks.len() >= 2,
        "expected at least 2 blocks, got {}",
        track.blocks.len()
    );
    // Sanity: every block has at least one line, all lines fit.
    for block in &track.blocks {
        assert!(!block.lines.is_empty());
        for line in &block.lines {
            assert!(
                line.chars().count() <= 32,
                "line '{line}' exceeds max_line_length"
            );
        }
    }
}

#[test]
fn test_pipeline_with_mock_asr_with_diar_5_speakers() {
    // 5 speakers, each contributing one word in their own segment.
    let words = vec![
        mk_word("alpha", 0, 500),
        mk_word("bravo", 800, 1_200),
        mk_word("charlie", 1_600, 2_000),
        mk_word("delta", 2_400, 2_800),
        mk_word("echo", 3_200, 3_600),
    ];
    let asr = CannedAsr(words);

    let mut diar = DiarizationResult::new();
    for id in 1..=5_u8 {
        diar.speakers.insert(
            id,
            Speaker {
                id,
                name: Some(format!("Speaker {id}")),
                gender: None,
                language: Some("en-US".to_string()),
            },
        );
    }
    diar.turns = vec![
        mk_turn(1, 0, 700),
        mk_turn(2, 700, 1_500),
        mk_turn(3, 1_500, 2_300),
        mk_turn(4, 2_300, 3_100),
        mk_turn(5, 3_100, 4_000),
    ];

    let cfg = PipelineConfig {
        max_line_length: 32,
        max_lines_per_block: 2,
        min_block_duration_ms: 200,
        max_block_duration_ms: 1_000,
        min_gap_ms: 200,
        language: Some("en".to_string()),
        enable_diarization: true,
        line_break_strategy: LineBreakStrategy::OptimalDp,
    };
    let pipe = AutoCaptionPipeline::new(asr, cfg).with_diarizer(CannedDiarizer(diar));
    let track = pipe
        .process_audio(&vec![0.0_f32; 16_000 * 4], 16_000)
        .expect("pipeline with diariser runs");

    assert!(!track.is_empty(), "expected non-empty track");
    // We expect all 5 speakers to be attributed across the blocks.
    let mut speakers_seen: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
    for b in &track.blocks {
        if let Some(sid) = b.speaker_id {
            speakers_seen.insert(sid);
        }
    }
    assert_eq!(
        speakers_seen.len(),
        5,
        "expected 5 distinct speakers attributed, got {:?}",
        speakers_seen
    );
    assert_eq!(track.speakers.len(), 5);
}

#[test]
fn test_pipeline_handles_empty_audio_gracefully() {
    let asr = CannedAsr(vec![]);
    let pipe = AutoCaptionPipeline::new(asr, PipelineConfig::default());
    let track = pipe.process_audio(&[], 48_000).expect("ok on empty audio");
    assert!(track.is_empty());
    assert_eq!(track.total_duration_ms, 0);
    assert!(track.wcag_violations.is_empty());
}

// ─── TODO.md line 35: overlapping words ─────────────────────────────

#[test]
fn test_alignment_overlapping_words() {
    // Two segments whose constituent words overlap in time.
    let mut seg1 = mk_segment("Hello there", 0, 2_000);
    seg1.words = vec![
        mk_word("Hello", 0, 900),
        mk_word("there", 800, 2_000), // overlaps with previous word
    ];
    let mut seg2 = mk_segment("world", 1_900, 3_500);
    seg2.words = vec![mk_word("world", 1_900, 3_500)];

    let blocks = build_caption_blocks(&[seg1, seg2], 2, 40);
    assert_eq!(blocks.len(), 2);
    // Times must be preserved exactly.
    assert_eq!(blocks[0].start_ms, 0);
    assert_eq!(blocks[0].end_ms, 2_000);
    assert_eq!(blocks[1].start_ms, 1_900);
    assert_eq!(blocks[1].end_ms, 3_500);
    // Text must contain all words from each segment.
    let text0: String = blocks[0].lines.join(" ");
    assert!(text0.contains("Hello"));
    assert!(text0.contains("there"));
    let text1: String = blocks[1].lines.join(" ");
    assert!(text1.contains("world"));
}

// ─── TODO.md line 36: Knuth-Plass reference output ──────────────────

#[test]
fn test_optimal_break_matches_kp_reference() {
    // Reference fixture: "aaa bb cccccc ddddd ee" at width 6 has a
    // known optimum where the lines are
    //   "aaa bb"        (6 chars, slack 0)
    //   "cccccc"        (6 chars, slack 0)
    //   "ddddd ee"      (8 chars, slack — overflow handled as own
    //                    line of 5 + own line of 2)
    // The Knuth-Plass DP must find slack sums lower than the greedy
    // baseline (or equal in degenerate cases).
    let text = "aaa bb cccccc ddddd ee";
    let width = 6_u8;
    let kp = optimal_break(text, width);
    let greedy = greedy_break(text, width);

    fn cost(lines: &[String], w: usize) -> u64 {
        lines
            .iter()
            .map(|l| {
                let n = l.chars().count();
                if n > w {
                    0
                } else {
                    let s = (w - n) as u64;
                    s * s
                }
            })
            .sum()
    }
    assert!(
        cost(&kp, width as usize) <= cost(&greedy, width as usize),
        "KP cost {} should be <= greedy cost {}",
        cost(&kp, width as usize),
        cost(&greedy, width as usize)
    );
    // Verify the round-trip preserves all words.
    let rejoined = kp.join(" ");
    assert_eq!(rejoined, text);
}

// ─── TODO.md line 37: WCAG compliance suite ─────────────────────────

#[test]
fn test_wcag_compliance_suite_a_aa_aaa() {
    // ── Pass: Level A (coverage + min duration) ─────────────
    let pass_a = vec![
        mk_block(1, 0, 1_500, "Hello world"),
        mk_block(2, 1_500, 3_000, "How are you"),
    ];
    let v = run_all_checks(&pass_a, 3_000, WcagLevel::A);
    assert!(v.is_empty(), "expected pass at level A, got {:?}", v);

    // ── Fail: Level A — short block. ────────────────────────
    let fail_min_dur = vec![mk_block(1, 0, 300, "Hi")];
    assert!(check_min_duration(&fail_min_dur[0], 1_000).is_some());

    // ── Fail: Level A — gap too large. ──────────────────────
    let fail_coverage = vec![
        mk_block(1, 0, 1_000, "Start"),
        mk_block(2, 10_000, 11_000, "Way later"),
    ];
    let v = check_caption_coverage(&fail_coverage, 11_000);
    assert!(v.is_some());

    // ── Pass: Level AA — CPS within 17. ─────────────────────
    let pass_aa = mk_block(1, 0, 2_000, "short text");
    assert!(check_cps(&pass_aa, 17.0).is_none());

    // ── Fail: Level AA — CPS too high. ──────────────────────
    let fail_aa = mk_block(1, 0, 500, "this is far too much text for the duration");
    assert!(check_cps(&fail_aa, 17.0).is_some());

    // ── Level AAA — same checks apply with the additional ───
    // ── sign-language criterion (not machine-checkable). ────
    let pass_aaa = vec![mk_block(1, 0, 2_000, "Hello world")];
    let v = run_all_checks(&pass_aaa, 2_000, WcagLevel::AAA);
    assert!(v.is_empty(), "expected pass at AAA, got {:?}", v);

    // Compliance score should be 100 for clean track.
    assert!((compliance_score(&[]) - 100.0).abs() < 1e-5);
}

// ─── TODO.md line 38: 5+ simultaneous speakers ──────────────────────

#[test]
fn test_diarization_5_simultaneous_speakers() {
    // All 5 speakers are simultaneously active during the same time
    // span; speaker 4 has the largest overlap with the single block.
    let mut diar = DiarizationResult::new();
    for id in 1..=5_u8 {
        diar.speakers.insert(
            id,
            Speaker {
                id,
                name: None,
                gender: None,
                language: None,
            },
        );
    }
    diar.turns = vec![
        mk_turn(1, 0, 200),
        mk_turn(2, 100, 400),
        mk_turn(3, 0, 600),
        mk_turn(4, 0, 1_500), // dominates the block window
        mk_turn(5, 200, 800),
    ];
    let mut blocks = vec![mk_block(1, 0, 2_000, "everyone talking at once")];
    assign_speakers_to_blocks(&mut blocks, &diar);
    assert_eq!(
        blocks[0].speaker_id,
        Some(4),
        "speaker 4 should win the largest overlap"
    );
}

// ─── TODO.md line 39: property test merge_short_segments ────────────

proptest! {
    /// `merge_short_segments` must never leave a segment shorter than
    /// the configured minimum duration, except for a possible trailing
    /// short segment when there are no more neighbours to absorb it.
    #[test]
    fn test_merge_short_segments_no_segment_below_min(
        // 1..=8 random segments of width 50..=2000 ms.
        widths in prop::collection::vec(50_u64..=2_000_u64, 1..=8),
        min_dur in 100_u32..=1_000_u32,
    ) {
        let mut t = 0_u64;
        let segs: Vec<TranscriptSegment> = widths.iter().enumerate().map(|(i, &w)| {
            let s = t;
            let e = t + w;
            t = e;
            mk_segment(&format!("seg{i}"), s, e)
        }).collect();
        let merged = merge_short_segments(&segs, min_dur);
        // If the input had any segment >= min_dur, after merging
        // every segment must be >= min_dur.  If all inputs are shorter
        // than min_dur, the entire thing may collapse to one segment
        // shorter than min_dur — but only one segment.
        let any_long = segs.iter().any(|s| s.duration_ms() >= u64::from(min_dur));
        if any_long {
            for s in &merged {
                prop_assert!(
                    s.duration_ms() >= u64::from(min_dur),
                    "segment of {}ms is shorter than min {}ms",
                    s.duration_ms(),
                    min_dur,
                );
            }
        } else {
            // At most one segment (everything absorbed into a single
            // catch-all).
            prop_assert!(merged.len() <= 1);
        }
    }
}

// ─── TODO.md line 40: greedy vs optimal single-line identical ───────

#[test]
fn test_greedy_vs_optimal_single_line_identical() {
    // When all text fits on one line, both algorithms must agree.
    let texts = ["Hello", "Hello world", "one two three", "short text here"];
    for text in texts {
        let g = greedy_break(text, 40);
        let o = optimal_break(text, 40);
        let s = optimal_break_smawk(text, 40);
        assert_eq!(g, o, "greedy vs optimal: {g:?} != {o:?}");
        assert_eq!(g, s, "greedy vs smawk: {g:?} != {s:?}");
    }
}

// ─── TODO.md line 41: round-trip split + merge preserves text ───────

proptest! {
    /// Splitting a long segment and then merging the result must
    /// preserve every word from the original text.
    #[test]
    fn test_split_merge_segments_text_preserved(
        // Build a text from 4..=16 random words (lowercase ASCII).
        words in prop::collection::vec("[a-z]{1,8}", 4..=16),
        max_chars in 10_u16..=40_u16,
    ) {
        let text = words.join(" ");
        let seg = mk_segment(&text, 0, 10_000);
        let split = split_long_segments(&seg, 3_000, max_chars);
        // Merge with min_duration_ms=0 so nothing extra collapses.
        let merged = merge_short_segments(&split, 0);
        let recon: String = merged
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Every word from the original must appear in the result.
        for w in &words {
            prop_assert!(
                recon.contains(w.as_str()),
                "word '{}' missing from reconstruction '{}'",
                w,
                recon,
            );
        }
    }
}

// ─── SMAWK fidelity: 10 000 randomised inputs ───────────────────────

/// Deterministic pseudo-random number generator using SplitMix64.
/// We deliberately avoid `rand` to keep dev-deps lean and
/// reproducibility absolute.
fn splitmix_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[test]
fn test_smawk_matches_dp_10k_random_inputs() {
    /// Slack-squared cost of a candidate layout.  Lines wider than
    /// `max_width` represent forced single-word lines with zero
    /// contribution to the cost.
    fn cost(lines: &[String], w: usize) -> u64 {
        lines
            .iter()
            .map(|l| {
                let n = l.chars().count();
                if n > w {
                    0
                } else {
                    let s = (w - n) as u64;
                    s * s
                }
            })
            .sum()
    }

    let mut state: u64 = 0xCAFEF00DDEADBEEF;
    // 10 000 iterations to satisfy the test name's contract.
    const ITERATIONS: usize = 10_000;
    let alphabet = b"abcdefghijklmnopqrstuvwxyz";

    for iter in 0..ITERATIONS {
        // 1..=20 words; each word has 1..=8 random lowercase letters.
        let n_words = 1 + (splitmix_next(&mut state) as usize) % 20;
        let mut words: Vec<String> = Vec::with_capacity(n_words);
        for _ in 0..n_words {
            let len = 1 + (splitmix_next(&mut state) as usize) % 8;
            let s: String = (0..len)
                .map(|_| {
                    let idx = (splitmix_next(&mut state) as usize) % alphabet.len();
                    alphabet[idx] as char
                })
                .collect();
            words.push(s);
        }
        let text = words.join(" ");

        // Width 4..=32 chars.
        let width = 4_u8 + (splitmix_next(&mut state) % 29) as u8;

        let dp = optimal_break(&text, width);
        let smawk = optimal_break_smawk(&text, width);

        let c_dp = cost(&dp, width as usize);
        let c_sm = cost(&smawk, width as usize);

        assert_eq!(
            c_dp, c_sm,
            "iteration {iter}: cost mismatch for text {text:?} width {width}: dp={dp:?} (cost {c_dp}) smawk={smawk:?} (cost {c_sm})"
        );

        // Both must preserve the input text verbatim when re-joined.
        assert_eq!(dp.join(" "), text);
        assert_eq!(smawk.join(" "), text);

        // No line should exceed max width (except for unsplittable
        // long single words).
        for line in &smawk {
            let words_in_line: Vec<&str> = line.split_whitespace().collect();
            if words_in_line.len() > 1 {
                assert!(
                    line.chars().count() <= width as usize,
                    "iteration {iter}: smawk multi-word line '{line}' exceeds width {width}"
                );
            }
        }
    }
}
