//! WCAG 2.1 accessibility compliance checks for caption blocks.
//!
//! This module implements machine-checkable WCAG 2.1 success criteria that
//! apply to caption and subtitle tracks in multimedia content.  Each check
//! function corresponds to one or more guidelines; the table below maps
//! functions to their rule identifiers, WCAG success criteria, and conformance
//! level.
//!
//! ## Conformance Level Mapping
//!
//! | Function | rule_id | WCAG SC | Level | Notes |
//! |---|---|---|---|---|
//! | [`check_caption_coverage`] | `"1.2.2"` | 1.2.2 Captions (Prerecorded) | A | Fails when gap > 2 000 ms |
//! | [`check_live_latency`] | `"1.2.4"` | 1.2.4 Captions (Live) | AA | Fails when latency > 3 000 ms |
//! | [`check_sign_language`] | — | 1.2.6 Sign Language | AAA | Always `None`; not machine-checkable |
//! | [`check_cps`] | `"CPS"` | Reading speed (BBC/Netflix 17 cps) | AA | Per-block check |
//! | [`check_min_duration`] | `"MIN_DUR"` | Min display time 1 000 ms | A | Per-block check |
//! | [`check_gap_duration`] | `"GAP"` | Gap between blocks | A | Returns all violations |
//! | [`run_all_checks`] | (multi) | Aggregate | varies | Runs coverage + min\_dur + CPS (≥ AA) + gap |
//! | [`check_max_simultaneous_captions`] | `"MAX_SIMULTANEOUS"` | EBU STL / SMPTE 2052 | AA | ≤ 2 simultaneous blocks |
//! | [`check_cps_for_audience`] | `"CPS_AUDIENCE"` | Per-audience cps limit | AA | Children vs. adult profiles |
//! | [`compliance_score`] | — | Scoring helper | — | 100 − penalty (10 / AA-AAA, 2 / A) |
//!
//! ## Not Implemented
//!
//! **1.2.5 Audio Description (Prerecorded)** and **1.2.7 Extended Audio
//! Description (Prerecorded)** are _not_ implemented here.  These success
//! criteria require a producer to supply a separate audio track describing
//! visual information; they are not derivable from caption text alone and
//! therefore cannot be machine-checked by this module.
//!
//! ## Usage Example
//!
//! ```rust
//! use oximedia_caption_gen::wcag::{
//!     WcagChecker, WcagLevel, run_all_checks, compliance_score,
//! };
//! use oximedia_caption_gen::alignment::{CaptionBlock, CaptionPosition};
//!
//! // Build a minimal set of caption blocks for illustration.
//! let blocks = vec![
//!     CaptionBlock {
//!         id: 1,
//!         start_ms: 0,
//!         end_ms: 3_000,
//!         lines: vec!["Hello, world.".to_string()],
//!         speaker_id: None,
//!         position: CaptionPosition::Bottom,
//!     },
//!     CaptionBlock {
//!         id: 2,
//!         start_ms: 3_500,
//!         end_ms: 7_000,
//!         lines: vec!["This caption meets the minimum display-time requirement.".to_string()],
//!         speaker_id: None,
//!         position: CaptionPosition::Bottom,
//!     },
//! ];
//!
//! // Create a checker targeting WCAG Level AA.
//! let checker = WcagChecker::new(WcagLevel::AA);
//! let content_duration_ms: u64 = 7_000;
//!
//! // Run all checks appropriate for the configured level.
//! let violations = run_all_checks(&blocks, content_duration_ms, checker.level.clone());
//!
//! // Compute a numeric compliance score.
//! let score = compliance_score(&violations);
//! assert!(score >= 0.0 && score <= 100.0);
//! println!("WCAG compliance score: {score:.1} ({} violation(s))", violations.len());
//! ```

use crate::{alignment::CaptionBlock, line_breaking::compute_cps};

/// WCAG conformance level.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum WcagLevel {
    /// The minimum level; fundamental accessibility.
    A,
    /// Enhanced accessibility for broader audiences.
    AA,
    /// Highest level of accessibility.
    AAA,
}

impl std::fmt::Display for WcagLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WcagLevel::A => write!(f, "A"),
            WcagLevel::AA => write!(f, "AA"),
            WcagLevel::AAA => write!(f, "AAA"),
        }
    }
}

/// A single WCAG violation found during compliance checking.
#[derive(Debug, Clone, PartialEq)]
pub struct WcagViolation {
    /// Short identifier for the violated rule (e.g., `"1.2.2"`).
    pub rule_id: String,
    /// Human-readable description of the violation.
    pub message: String,
    /// WCAG conformance level at which this rule applies.
    pub severity: WcagLevel,
    /// Optional timestamp (in ms) where the violation occurs.
    pub timestamp_ms: Option<u64>,
}

impl std::fmt::Display for WcagViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[WCAG {}] {}: {}",
            self.severity, self.rule_id, self.message
        )
    }
}

/// Performs WCAG compliance checks at a given conformance level.
#[derive(Debug, Clone)]
pub struct WcagChecker {
    /// Target conformance level; checks for levels above this are skipped.
    pub level: WcagLevel,
}

impl WcagChecker {
    /// Create a checker targeting the given WCAG level.
    pub fn new(level: WcagLevel) -> Self {
        Self { level }
    }
}

// ─── Individual checks ────────────────────────────────────────────────────────

/// **WCAG 1.2.2** — Check that caption coverage gaps do not exceed 2 seconds.
///
/// If any gap between successive caption blocks exceeds 2000 ms (a gap in
/// coverage), a violation is returned.
///
/// Returns `None` when no violations are found.
pub fn check_caption_coverage(
    blocks: &[CaptionBlock],
    content_duration_ms: u64,
) -> Option<WcagViolation> {
    const MAX_GAP_MS: u64 = 2000;

    if blocks.is_empty() && content_duration_ms > MAX_GAP_MS {
        return Some(WcagViolation {
            rule_id: "1.2.2".to_string(),
            message: format!(
                "No captions provided for content of {content_duration_ms}ms duration (max gap 2000ms)"
            ),
            severity: WcagLevel::A,
            timestamp_ms: Some(0),
        });
    }

    let mut sorted = blocks.to_vec();
    sorted.sort_by_key(|b| b.start_ms);

    // Gap before first block.
    if let Some(first) = sorted.first() {
        if first.start_ms > MAX_GAP_MS {
            return Some(WcagViolation {
                rule_id: "1.2.2".to_string(),
                message: format!(
                    "Gap before first caption: {}ms exceeds 2000ms",
                    first.start_ms
                ),
                severity: WcagLevel::A,
                timestamp_ms: Some(0),
            });
        }
    }

    // Gaps between blocks.
    for pair in sorted.windows(2) {
        let gap = pair[1].start_ms.saturating_sub(pair[0].end_ms);
        if gap > MAX_GAP_MS {
            return Some(WcagViolation {
                rule_id: "1.2.2".to_string(),
                message: format!(
                    "Caption gap of {}ms at ~{}ms exceeds 2000ms",
                    gap, pair[0].end_ms
                ),
                severity: WcagLevel::A,
                timestamp_ms: Some(pair[0].end_ms),
            });
        }
    }

    // Gap after last block.
    if let Some(last) = sorted.last() {
        let trailing_gap = content_duration_ms.saturating_sub(last.end_ms);
        if trailing_gap > MAX_GAP_MS {
            return Some(WcagViolation {
                rule_id: "1.2.2".to_string(),
                message: format!(
                    "Caption gap of {}ms after last block at {}ms exceeds 2000ms",
                    trailing_gap, last.end_ms
                ),
                severity: WcagLevel::A,
                timestamp_ms: Some(last.end_ms),
            });
        }
    }

    None
}

/// **WCAG 1.2.4** — Check live caption latency.
///
/// Returns a violation if `latency_ms` exceeds 3000 ms.
pub fn check_live_latency(latency_ms: u32) -> Option<WcagViolation> {
    const MAX_LATENCY_MS: u32 = 3000;
    if latency_ms > MAX_LATENCY_MS {
        Some(WcagViolation {
            rule_id: "1.2.4".to_string(),
            message: format!("Live caption latency {latency_ms}ms exceeds maximum 3000ms"),
            severity: WcagLevel::AA,
            timestamp_ms: None,
        })
    } else {
        None
    }
}

/// **WCAG 1.2.6** — Sign Language (Level AAA).
///
/// Not machine-checkable; always returns `None`.
pub fn check_sign_language() -> Option<WcagViolation> {
    None
}

/// Check that the reading speed of a caption block does not exceed `max_cps`.
///
/// The BBC/Netflix guideline is 17 chars/sec. Returns a violation if exceeded.
pub fn check_cps(block: &CaptionBlock, max_cps: f32) -> Option<WcagViolation> {
    let text: String = block.lines.join(" ");
    let duration_ms = block.duration_ms();
    if duration_ms == 0 {
        return None;
    }
    let cps = compute_cps(&text, duration_ms);
    if cps > max_cps {
        Some(WcagViolation {
            rule_id: "CPS".to_string(),
            message: format!(
                "Block {} reading speed {:.1} chars/sec exceeds maximum {:.1} chars/sec",
                block.id, cps, max_cps
            ),
            severity: WcagLevel::AA,
            timestamp_ms: Some(block.start_ms),
        })
    } else {
        None
    }
}

/// Check that a caption block is displayed for at least `min_ms` milliseconds.
///
/// WCAG recommends a minimum of 1000 ms.
pub fn check_min_duration(block: &CaptionBlock, min_ms: u32) -> Option<WcagViolation> {
    let dur = block.duration_ms();
    if dur < u64::from(min_ms) {
        Some(WcagViolation {
            rule_id: "MIN_DUR".to_string(),
            message: format!(
                "Block {} duration {}ms is shorter than minimum {}ms",
                block.id, dur, min_ms
            ),
            severity: WcagLevel::A,
            timestamp_ms: Some(block.start_ms),
        })
    } else {
        None
    }
}

/// Check that no gap between caption blocks exceeds `max_gap_ms`.
///
/// Returns a `Vec` of all violations (one per offending gap).
pub fn check_gap_duration(blocks: &[CaptionBlock], max_gap_ms: u32) -> Vec<WcagViolation> {
    let mut violations: Vec<WcagViolation> = Vec::new();
    let max_gap = u64::from(max_gap_ms);

    let mut sorted = blocks.to_vec();
    sorted.sort_by_key(|b| b.start_ms);

    for pair in sorted.windows(2) {
        let gap = pair[1].start_ms.saturating_sub(pair[0].end_ms);
        if gap > max_gap {
            violations.push(WcagViolation {
                rule_id: "GAP".to_string(),
                message: format!(
                    "Gap of {}ms between block {} and block {} exceeds {}ms",
                    gap, pair[0].id, pair[1].id, max_gap_ms
                ),
                severity: WcagLevel::A,
                timestamp_ms: Some(pair[0].end_ms),
            });
        }
    }
    violations
}

/// Run all checks appropriate for the given `level` against `blocks`.
///
/// Checks run:
/// - 1.2.2 caption coverage (Level A and above)
/// - CPS check (Level AA and above)
/// - Minimum duration (Level A and above)
/// - Gap duration (Level A and above)
pub fn run_all_checks(
    blocks: &[CaptionBlock],
    content_duration_ms: u64,
    level: WcagLevel,
) -> Vec<WcagViolation> {
    let mut violations: Vec<WcagViolation> = Vec::new();

    // 1.2.2 — Level A.
    if level >= WcagLevel::A {
        if let Some(v) = check_caption_coverage(blocks, content_duration_ms) {
            violations.push(v);
        }
    }

    for block in blocks {
        // Minimum duration — Level A.
        if level >= WcagLevel::A {
            if let Some(v) = check_min_duration(block, 1000) {
                violations.push(v);
            }
        }
        // CPS — Level AA.
        if level >= WcagLevel::AA {
            if let Some(v) = check_cps(block, 17.0) {
                violations.push(v);
            }
        }
    }

    // Gap duration — Level A.
    if level >= WcagLevel::A {
        violations.extend(check_gap_duration(blocks, 2000));
    }

    violations
}

/// Check that the maximum number of simultaneously-visible caption blocks does
/// not exceed `max_simultaneous`.
///
/// Most broadcast standards (e.g., EBU STL, SMPTE 2052) recommend no more
/// than 2 simultaneous blocks to avoid cognitive overload.
///
/// Returns one [`WcagViolation`] for each timestamp where the count is exceeded.
pub fn check_max_simultaneous_captions(
    blocks: &[CaptionBlock],
    max_simultaneous: usize,
) -> Vec<WcagViolation> {
    if blocks.is_empty() || max_simultaneous == 0 {
        return Vec::new();
    }

    // Collect all events: (time_ms, +1 = start, -1 = end, block_id).
    let mut events: Vec<(u64, i32, u32)> = Vec::with_capacity(blocks.len() * 2);
    for block in blocks {
        events.push((block.start_ms, 1, block.id));
        events.push((block.end_ms, -1, block.id));
    }
    // Sort: earlier time first; ends before starts at same time.
    events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut count: i64 = 0;
    let mut violations: Vec<WcagViolation> = Vec::new();

    for (time_ms, delta, block_id) in events {
        count += i64::from(delta);
        if count > max_simultaneous as i64 {
            violations.push(WcagViolation {
                rule_id: "MAX_SIMULTANEOUS".to_string(),
                message: format!(
                    "At {}ms: {} simultaneous caption blocks exceeds maximum of {} (block {} \
                     just started)",
                    time_ms, count, max_simultaneous, block_id
                ),
                severity: WcagLevel::AA,
                timestamp_ms: Some(time_ms),
            });
        }
    }

    violations
}

/// Check reading speed validation for a specific audience profile.
///
/// Uses per-audience CPS limits rather than the generic 17 CPS limit.
/// Children's content in particular requires much slower reading speeds.
pub fn check_cps_for_audience(
    block: &CaptionBlock,
    audience: crate::line_breaking::AudienceProfile,
) -> Option<WcagViolation> {
    let text: String = block.lines.join(" ");
    let duration_ms = block.duration_ms();
    if duration_ms == 0 {
        return None;
    }
    let max_cps = audience.max_cps();
    let cps = compute_cps(&text, duration_ms);
    if cps > max_cps {
        Some(WcagViolation {
            rule_id: "CPS_AUDIENCE".to_string(),
            message: format!(
                "Block {} reading speed {:.1} chars/sec exceeds maximum {:.1} chars/sec for \
                 audience {:?}",
                block.id, cps, max_cps, audience
            ),
            severity: WcagLevel::AA,
            timestamp_ms: Some(block.start_ms),
        })
    } else {
        None
    }
}

/// Compute a compliance score in [0.0, 100.0].
///
/// Starts at 100, subtracts 10 per AA/AAA error and 2 per A warning.
pub fn compliance_score(violations: &[WcagViolation]) -> f32 {
    let penalty: f32 = violations
        .iter()
        .map(|v| match v.severity {
            WcagLevel::A => 2.0,
            WcagLevel::AA => 10.0,
            WcagLevel::AAA => 10.0,
        })
        .sum();
    (100.0 - penalty).max(0.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::{CaptionBlock, CaptionPosition};

    fn make_block(id: u32, start_ms: u64, end_ms: u64, text: &str) -> CaptionBlock {
        CaptionBlock {
            id,
            start_ms,
            end_ms,
            lines: vec![text.to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }

    // --- check_caption_coverage ---

    #[test]
    fn coverage_passes_with_no_gaps() {
        let blocks = vec![
            make_block(1, 0, 2000, "Hello"),
            make_block(2, 2000, 4000, "World"),
        ];
        assert!(check_caption_coverage(&blocks, 4000).is_none());
    }

    #[test]
    fn coverage_fails_on_large_interior_gap() {
        let blocks = vec![make_block(1, 0, 1000, "A"), make_block(2, 5000, 6000, "B")];
        let v = check_caption_coverage(&blocks, 6000);
        assert!(v.is_some());
        let v = v.expect("value should be present should succeed");
        assert_eq!(v.rule_id, "1.2.2");
        assert_eq!(v.severity, WcagLevel::A);
    }

    #[test]
    fn coverage_fails_on_empty_blocks_with_long_content() {
        let v = check_caption_coverage(&[], 10000);
        assert!(v.is_some());
    }

    #[test]
    fn coverage_passes_small_trailing_gap() {
        let blocks = vec![make_block(1, 0, 3500, "Text")];
        // trailing gap = 4000 - 3500 = 500ms < 2000ms
        assert!(check_caption_coverage(&blocks, 4000).is_none());
    }

    #[test]
    fn coverage_fails_large_leading_gap() {
        let blocks = vec![make_block(1, 5000, 7000, "Late start")];
        let v = check_caption_coverage(&blocks, 7000);
        assert!(v.is_some());
    }

    // --- check_live_latency ---

    #[test]
    fn live_latency_passes_under_limit() {
        assert!(check_live_latency(2999).is_none());
    }

    #[test]
    fn live_latency_passes_at_limit() {
        assert!(check_live_latency(3000).is_none());
    }

    #[test]
    fn live_latency_fails_over_limit() {
        let v = check_live_latency(3001);
        assert!(v.is_some());
        let v = v.expect("value should be present should succeed");
        assert_eq!(v.rule_id, "1.2.4");
        assert_eq!(v.severity, WcagLevel::AA);
    }

    // --- check_sign_language ---

    #[test]
    fn sign_language_always_none() {
        assert!(check_sign_language().is_none());
    }

    // --- check_cps ---

    #[test]
    fn cps_passes_slow_text() {
        // 5 chars over 2000ms = 2.5 cps < 17.
        let block = make_block(1, 0, 2000, "Hello");
        assert!(check_cps(&block, 17.0).is_none());
    }

    #[test]
    fn cps_fails_fast_text() {
        // 100 chars over 1000ms = 100 cps > 17.
        let text = "A".repeat(100);
        let block = make_block(1, 0, 1000, &text);
        let v = check_cps(&block, 17.0);
        assert!(v.is_some());
        assert_eq!(
            v.expect("value should be present should succeed").rule_id,
            "CPS"
        );
    }

    #[test]
    fn cps_zero_duration_passes() {
        let block = make_block(1, 1000, 1000, "Hello");
        assert!(check_cps(&block, 17.0).is_none());
    }

    // --- check_min_duration ---

    #[test]
    fn min_duration_passes() {
        let block = make_block(1, 0, 1500, "OK");
        assert!(check_min_duration(&block, 1000).is_none());
    }

    #[test]
    fn min_duration_fails_short_block() {
        let block = make_block(1, 0, 500, "Too short");
        let v = check_min_duration(&block, 1000);
        assert!(v.is_some());
        assert_eq!(
            v.expect("value should be present should succeed").rule_id,
            "MIN_DUR"
        );
    }

    #[test]
    fn min_duration_passes_at_exactly_min() {
        let block = make_block(1, 0, 1000, "Exactly 1s");
        assert!(check_min_duration(&block, 1000).is_none());
    }

    // --- check_gap_duration ---

    #[test]
    fn gap_duration_no_violations_when_close() {
        let blocks = vec![make_block(1, 0, 1000, "A"), make_block(2, 1500, 2500, "B")];
        let v = check_gap_duration(&blocks, 2000);
        assert!(v.is_empty());
    }

    #[test]
    fn gap_duration_detects_violation() {
        let blocks = vec![make_block(1, 0, 1000, "A"), make_block(2, 5000, 6000, "B")];
        let v = check_gap_duration(&blocks, 2000);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].rule_id, "GAP");
    }

    #[test]
    fn gap_duration_multiple_violations() {
        let blocks = vec![
            make_block(1, 0, 100, "A"),
            make_block(2, 5000, 5100, "B"),
            make_block(3, 10000, 10100, "C"),
        ];
        let v = check_gap_duration(&blocks, 500);
        assert_eq!(v.len(), 2);
    }

    // --- run_all_checks ---

    #[test]
    fn run_all_checks_clean_content() {
        let blocks = vec![
            make_block(1, 0, 2000, "Hello world here"),
            make_block(2, 2000, 4000, "How are you today"),
        ];
        let v = run_all_checks(&blocks, 4000, WcagLevel::AA);
        assert!(v.is_empty(), "unexpected violations: {:?}", v);
    }

    #[test]
    fn run_all_checks_detects_short_block() {
        let blocks = vec![make_block(1, 0, 200, "Hi")];
        let v = run_all_checks(&blocks, 200, WcagLevel::A);
        assert!(!v.is_empty());
    }

    // --- compliance_score ---

    #[test]
    fn compliance_score_no_violations() {
        assert!((compliance_score(&[]) - 100.0).abs() < 1e-5);
    }

    #[test]
    fn compliance_score_one_a_violation() {
        let v = vec![WcagViolation {
            rule_id: "1.2.2".to_string(),
            message: "test".to_string(),
            severity: WcagLevel::A,
            timestamp_ms: None,
        }];
        assert!((compliance_score(&v) - 98.0).abs() < 1e-5);
    }

    #[test]
    fn compliance_score_one_aa_violation() {
        let v = vec![WcagViolation {
            rule_id: "CPS".to_string(),
            message: "test".to_string(),
            severity: WcagLevel::AA,
            timestamp_ms: None,
        }];
        assert!((compliance_score(&v) - 90.0).abs() < 1e-5);
    }

    #[test]
    fn compliance_score_never_below_zero() {
        let violations: Vec<WcagViolation> = (0..20)
            .map(|i| WcagViolation {
                rule_id: format!("R{i}"),
                message: "test".to_string(),
                severity: WcagLevel::AA,
                timestamp_ms: None,
            })
            .collect();
        assert_eq!(compliance_score(&violations), 0.0);
    }

    #[test]
    fn wcag_level_ordering() {
        assert!(WcagLevel::A < WcagLevel::AA);
        assert!(WcagLevel::AA < WcagLevel::AAA);
    }

    #[test]
    fn wcag_violation_display() {
        let v = WcagViolation {
            rule_id: "1.2.2".to_string(),
            message: "test violation".to_string(),
            severity: WcagLevel::A,
            timestamp_ms: Some(1000),
        };
        let s = v.to_string();
        assert!(s.contains("1.2.2"));
        assert!(s.contains("test violation"));
    }
}
