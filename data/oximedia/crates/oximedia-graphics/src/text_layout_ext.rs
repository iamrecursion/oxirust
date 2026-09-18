//! Extended text layout capabilities for broadcast graphics.
//!
//! Provides multi-line text wrapping with word-break and hyphenation,
//! text justification modes (left, center, right, full-justify with
//! Knuth-Plass-style penalty optimization), vertical text layout for
//! CJK characters, and text fitting (auto-size text to fit bounding box).

use crate::text_layout::{TextLayoutConfig, TextLayoutEngine, TextLayoutResult};

// ============================================================================
// Word-break / hyphenation
// ============================================================================

/// Word-break strategy controlling how long words are split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WordBreakStrategy {
    /// Never break inside a word (overflow allowed).
    #[default]
    Normal,
    /// Break at any character boundary when a word exceeds the line width.
    BreakAll,
    /// Insert a soft hyphen and break at syllable-like boundaries.
    Hyphenate,
}

/// Minimal syllable-boundary hyphenation.
///
/// Uses a simple vowel/consonant heuristic: break after a vowel followed by
/// a consonant cluster of length >= 1 when the remaining fragment is >= 2
/// characters.  This is intentionally simplified — a full Liang-Knuth
/// hyphenation algorithm would require dictionary data.
pub fn find_hyphenation_points(word: &str) -> Vec<usize> {
    let chars: Vec<char> = word.chars().collect();
    let len = chars.len();
    if len < 4 {
        return Vec::new();
    }

    fn is_vowel(c: char) -> bool {
        matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
    }

    let mut points = Vec::new();
    let mut i = 1;
    while i < len.saturating_sub(2) {
        // Pattern: vowel followed by consonant → potential break after the vowel
        if is_vowel(chars[i]) && !is_vowel(chars[i + 1]) {
            // Ensure at least 2 chars on each side of the break
            if i >= 1 && (len - i - 1) >= 2 {
                points.push(i + 1);
                i += 2; // skip ahead to avoid adjacent break points
                continue;
            }
        }
        i += 1;
    }
    points
}

/// Break a long word according to the given strategy so that each fragment
/// fits within `max_width_px`.
///
/// `char_width_fn` maps a character to its estimated pixel advance.
/// Returns a vector of `(fragment, was_hyphenated)` pairs.
pub fn break_word<F>(
    word: &str,
    max_width_px: f32,
    strategy: WordBreakStrategy,
    char_width_fn: F,
) -> Vec<(String, bool)>
where
    F: Fn(char) -> f32,
{
    let total_width: f32 = word.chars().map(&char_width_fn).sum();
    if total_width <= max_width_px {
        return vec![(word.to_string(), false)];
    }

    match strategy {
        WordBreakStrategy::Normal => vec![(word.to_string(), false)],
        WordBreakStrategy::BreakAll => {
            let mut fragments = Vec::new();
            let mut current = String::new();
            let mut width = 0.0_f32;
            for ch in word.chars() {
                let cw = char_width_fn(ch);
                if width + cw > max_width_px && !current.is_empty() {
                    fragments.push((current, false));
                    current = String::new();
                    width = 0.0;
                }
                current.push(ch);
                width += cw;
            }
            if !current.is_empty() {
                fragments.push((current, false));
            }
            fragments
        }
        WordBreakStrategy::Hyphenate => {
            let hyp_points = find_hyphenation_points(word);
            if hyp_points.is_empty() {
                // Fall back to break-all
                return break_word(
                    word,
                    max_width_px,
                    WordBreakStrategy::BreakAll,
                    char_width_fn,
                );
            }

            let chars: Vec<char> = word.chars().collect();
            let hyphen_width = char_width_fn('-');
            let mut fragments = Vec::new();
            let mut start = 0;

            for &bp in &hyp_points {
                // Measure fragment [start..bp]
                let frag_width: f32 = chars[start..bp].iter().map(|&c| char_width_fn(c)).sum();
                if frag_width + hyphen_width > max_width_px && start < bp {
                    // This fragment itself is too wide; emit what we have
                    let frag: String = chars[start..bp].iter().collect();
                    fragments.push((frag, true));
                    start = bp;
                }
            }
            // Emit remaining
            if start < chars.len() {
                let remaining_width: f32 = chars[start..].iter().map(|&c| char_width_fn(c)).sum();
                if remaining_width > max_width_px && !hyp_points.is_empty() {
                    // Try to break at each hyphenation point within [start..]
                    let mut sub_start = start;
                    for &bp in &hyp_points {
                        if bp <= sub_start {
                            continue;
                        }
                        let sub_width: f32 =
                            chars[sub_start..bp].iter().map(|&c| char_width_fn(c)).sum();
                        if sub_width + hyphen_width > max_width_px && sub_start < bp {
                            let frag: String = chars[sub_start..bp].iter().collect();
                            fragments.push((frag, true));
                            sub_start = bp;
                        }
                    }
                    let tail: String = chars[sub_start..].iter().collect();
                    fragments.push((tail, false));
                } else {
                    let frag: String = chars[start..].iter().collect();
                    fragments.push((frag, false));
                }
            }

            if fragments.is_empty() {
                fragments.push((word.to_string(), false));
            }
            fragments
        }
    }
}

// ============================================================================
// Knuth-Plass style justification
// ============================================================================

/// A line-break candidate for Knuth-Plass optimization.
#[derive(Debug, Clone)]
struct KpBreakpoint {
    /// Index into the word list where this break occurs (break *before* this word).
    word_index: usize,
    /// Cumulative demerits up to this breakpoint.
    demerits: f64,
    /// Index of the previous breakpoint in the `breakpoints` vec.
    prev: Option<usize>,
}

/// Compute Knuth-Plass-style optimal line breaks for a sequence of words.
///
/// Returns a list of word indices at which lines should break (each value is
/// the index of the first word on a new line).
///
/// `word_widths` gives the pixel width of each word.
/// `space_width` is the natural width of an inter-word space.
/// `line_width` is the target line width.
pub fn knuth_plass_breaks(word_widths: &[f32], space_width: f32, line_width: f32) -> Vec<usize> {
    let n = word_widths.len();
    if n == 0 {
        return Vec::new();
    }

    // Dynamic programming: find optimal breakpoints minimizing total demerits.
    let mut breakpoints: Vec<KpBreakpoint> = vec![KpBreakpoint {
        word_index: 0,
        demerits: 0.0,
        prev: None,
    }];

    for i in 0..n {
        let mut best_demerits = f64::MAX;
        let mut best_prev: Option<usize> = None;

        // Try each active breakpoint as the start of this line.
        for (bp_idx, bp) in breakpoints.iter().enumerate() {
            let start = bp.word_index;
            if start > i {
                continue;
            }

            // Compute line width from word[start] through word[i].
            let content_width: f32 = word_widths[start..=i].iter().sum::<f32>()
                + space_width * (i.saturating_sub(start)) as f32;

            if content_width > line_width * 1.15 && start < i {
                // Line is too wide — skip
                continue;
            }

            let slack = line_width - content_width;
            // Penalty: quadratic badness for loose lines, cubic for overfull
            let badness = if slack >= 0.0 {
                (slack as f64 / line_width as f64).powi(2) * 100.0
            } else {
                (-slack as f64 / line_width as f64).powi(3) * 1000.0
            };

            let total = bp.demerits + badness;
            if total < best_demerits {
                best_demerits = total;
                best_prev = Some(bp_idx);
            }
        }

        // If this could be the end of a line, record a breakpoint for the next word.
        if best_prev.is_some() && i < n {
            breakpoints.push(KpBreakpoint {
                word_index: i + 1,
                demerits: best_demerits,
                prev: best_prev,
            });
        }
    }

    // Find the best breakpoint that covers all words.
    let final_bp = breakpoints
        .iter()
        .enumerate()
        .filter(|(_, bp)| bp.word_index == n)
        .min_by(|(_, a), (_, b)| {
            a.demerits
                .partial_cmp(&b.demerits)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    let mut line_starts = Vec::new();
    if let Some((mut idx, _)) = final_bp {
        while let Some(prev_idx) = breakpoints[idx].prev {
            line_starts.push(breakpoints[idx].word_index);
            idx = prev_idx;
        }
        line_starts.push(0);
        line_starts.reverse();
        // Remove trailing entry if it equals n (end sentinel).
        if line_starts.last() == Some(&n) {
            line_starts.pop();
        }
    } else {
        // Fallback: all on one line
        line_starts.push(0);
    }

    line_starts
}

// ============================================================================
// Vertical (CJK) text layout
// ============================================================================

/// Orientation for text layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextOrientation {
    /// Horizontal layout (default, left-to-right lines stacked top-to-bottom).
    #[default]
    Horizontal,
    /// Vertical layout: characters stacked top-to-bottom, columns flow
    /// right-to-left (traditional CJK).
    VerticalRtl,
    /// Vertical layout: characters stacked top-to-bottom, columns flow
    /// left-to-right.
    VerticalLtr,
}

/// Check whether a character is a CJK ideograph or fullwidth character that
/// naturally renders in vertical orientation.
pub fn is_cjk_char(ch: char) -> bool {
    let cp = ch as u32;
    // CJK Unified Ideographs
    (0x4E00..=0x9FFF).contains(&cp)
        // CJK Extension A
        || (0x3400..=0x4DBF).contains(&cp)
        // CJK Extension B
        || (0x20000..=0x2A6DF).contains(&cp)
        // CJK Compatibility Ideographs
        || (0xF900..=0xFAFF).contains(&cp)
        // Hiragana
        || (0x3040..=0x309F).contains(&cp)
        // Katakana
        || (0x30A0..=0x30FF).contains(&cp)
        // Hangul Syllables
        || (0xAC00..=0xD7AF).contains(&cp)
        // CJK Symbols and Punctuation
        || (0x3000..=0x303F).contains(&cp)
        // Fullwidth Forms
        || (0xFF01..=0xFF60).contains(&cp)
        || (0xFFE0..=0xFFE6).contains(&cp)
}

/// A positioned glyph in a vertical layout.
#[derive(Clone, Debug)]
pub struct VerticalGlyph {
    /// Character.
    pub character: char,
    /// X position (column).
    pub x: f32,
    /// Y position (row within column).
    pub y: f32,
    /// Advance height (typically equal to font size for CJK).
    pub advance_height: f32,
    /// Column index (0 = first column).
    pub column_index: usize,
    /// Index in original text.
    pub char_index: usize,
}

/// A vertical column of text.
#[derive(Clone, Debug)]
pub struct VerticalColumn {
    /// Glyphs in this column, ordered top-to-bottom.
    pub glyphs: Vec<VerticalGlyph>,
    /// X offset of this column.
    pub x_offset: f32,
    /// Column width (typically font_size for CJK).
    pub width: f32,
    /// Total content height.
    pub height: f32,
}

/// Result of a vertical text layout.
#[derive(Clone, Debug)]
pub struct VerticalLayoutResult {
    /// Columns of glyphs.
    pub columns: Vec<VerticalColumn>,
    /// Total width across all columns.
    pub total_width: f32,
    /// Total height (max column height).
    pub total_height: f32,
    /// Number of characters laid out.
    pub chars_fitted: usize,
}

/// Configuration for vertical text layout.
#[derive(Clone, Debug)]
pub struct VerticalLayoutConfig {
    /// Font size in pixels (used for both width and height of CJK glyphs).
    pub font_size: f32,
    /// Maximum column height in pixels (0 = unlimited).
    pub max_height: f32,
    /// Maximum number of columns (0 = unlimited).
    pub max_columns: usize,
    /// Column spacing in pixels.
    pub column_spacing: f32,
    /// Orientation (VerticalRtl or VerticalLtr).
    pub orientation: TextOrientation,
    /// Inter-character spacing in pixels.
    pub char_spacing: f32,
}

impl Default for VerticalLayoutConfig {
    fn default() -> Self {
        Self {
            font_size: 24.0,
            max_height: 800.0,
            max_columns: 0,
            column_spacing: 8.0,
            orientation: TextOrientation::VerticalRtl,
            char_spacing: 2.0,
        }
    }
}

/// Lay out text vertically (top-to-bottom, columns flowing right-to-left or
/// left-to-right).
pub fn layout_vertical(text: &str, config: &VerticalLayoutConfig) -> VerticalLayoutResult {
    let cell_h = config.font_size + config.char_spacing;
    let col_width = config.font_size + config.column_spacing;

    let mut columns: Vec<VerticalColumn> = Vec::new();
    let mut current_col = VerticalColumn {
        glyphs: Vec::new(),
        x_offset: 0.0,
        width: config.font_size,
        height: 0.0,
    };
    let mut y = 0.0_f32;
    let mut col_idx = 0usize;
    let mut chars_fitted = 0usize;

    for (i, ch) in text.chars().enumerate() {
        if ch == '\n' {
            // Force new column
            current_col.height = y;
            columns.push(std::mem::replace(
                &mut current_col,
                VerticalColumn {
                    glyphs: Vec::new(),
                    x_offset: (col_idx + 1) as f32 * col_width,
                    width: config.font_size,
                    height: 0.0,
                },
            ));
            col_idx += 1;
            if config.max_columns > 0 && col_idx >= config.max_columns {
                chars_fitted = i + 1;
                break;
            }
            y = 0.0;
            chars_fitted = i + 1;
            continue;
        }

        // Check if we need a new column (height exceeded)
        if config.max_height > 0.0 && y + cell_h > config.max_height {
            current_col.height = y;
            columns.push(std::mem::replace(
                &mut current_col,
                VerticalColumn {
                    glyphs: Vec::new(),
                    x_offset: (col_idx + 1) as f32 * col_width,
                    width: config.font_size,
                    height: 0.0,
                },
            ));
            col_idx += 1;
            if config.max_columns > 0 && col_idx >= config.max_columns {
                chars_fitted = i;
                break;
            }
            y = 0.0;
        }

        current_col.glyphs.push(VerticalGlyph {
            character: ch,
            x: col_idx as f32 * col_width,
            y,
            advance_height: cell_h,
            column_index: col_idx,
            char_index: i,
        });
        y += cell_h;
        chars_fitted = i + 1;
    }

    // Push the last column if it has content
    if !current_col.glyphs.is_empty() {
        current_col.height = y;
        columns.push(current_col);
    }

    // If orientation is RTL, reverse column order and mirror x positions
    if config.orientation == TextOrientation::VerticalRtl && !columns.is_empty() {
        let total_cols = columns.len();
        for (new_idx, col) in columns.iter_mut().enumerate() {
            let mirrored_idx = total_cols - 1 - new_idx;
            let new_x = mirrored_idx as f32 * col_width;
            col.x_offset = new_x;
            for glyph in &mut col.glyphs {
                glyph.x = new_x;
            }
        }
        columns.reverse();
    }

    let total_width = if columns.is_empty() {
        0.0
    } else {
        columns.len() as f32 * col_width - config.column_spacing
    };
    let total_height = columns.iter().map(|c| c.height).fold(0.0_f32, f32::max);

    VerticalLayoutResult {
        columns,
        total_width,
        total_height,
        chars_fitted,
    }
}

// ============================================================================
// Text fitting (auto-size to bounding box)
// ============================================================================

/// Strategy for fitting text into a bounding box.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextFitStrategy {
    /// Reduce font size uniformly until text fits.
    #[default]
    ShrinkFont,
    /// Reduce letter spacing first, then shrink font.
    ReduceSpacingFirst,
    /// Increase font size to fill the box (never exceed box).
    GrowToFill,
}

/// Configuration for text fitting.
#[derive(Clone, Debug)]
pub struct TextFitConfig {
    /// Bounding box width in pixels.
    pub box_width: f32,
    /// Bounding box height in pixels.
    pub box_height: f32,
    /// Minimum font size (pixels) — never shrink below this.
    pub min_font_size: f32,
    /// Maximum font size (pixels) — never grow above this.
    pub max_font_size: f32,
    /// Fitting strategy.
    pub strategy: TextFitStrategy,
    /// Base text layout config (font_size field will be adjusted).
    pub layout_config: TextLayoutConfig,
    /// Convergence tolerance for binary search (pixels).
    pub tolerance: f32,
}

impl Default for TextFitConfig {
    fn default() -> Self {
        Self {
            box_width: 800.0,
            box_height: 600.0,
            min_font_size: 6.0,
            max_font_size: 200.0,
            strategy: TextFitStrategy::ShrinkFont,
            layout_config: TextLayoutConfig::default(),
            tolerance: 0.5,
        }
    }
}

/// Result of a text fitting operation.
#[derive(Clone, Debug)]
pub struct TextFitResult {
    /// The font size that was determined to fit.
    pub fitted_font_size: f32,
    /// Adjusted letter spacing (if `ReduceSpacingFirst` was used).
    pub letter_spacing: f32,
    /// The layout result at the fitted size.
    pub layout: TextLayoutResult,
    /// Whether the text was successfully fitted within bounds.
    pub fits: bool,
}

/// Find the optimal font size so that `text` fits within the bounding box
/// defined by `config`.
///
/// Uses binary search to converge on the largest font size that keeps the
/// layout within bounds.
pub fn fit_text(text: &str, config: &TextFitConfig) -> TextFitResult {
    if text.is_empty() {
        let mut lc = config.layout_config.clone();
        lc.max_width = config.box_width;
        lc.max_height = config.box_height;
        let engine = TextLayoutEngine::new(lc);
        let layout = engine.layout(text);
        return TextFitResult {
            fitted_font_size: config.layout_config.font_size,
            letter_spacing: config.layout_config.letter_spacing,
            layout,
            fits: true,
        };
    }

    match config.strategy {
        TextFitStrategy::ShrinkFont | TextFitStrategy::GrowToFill => {
            binary_search_font_size(text, config)
        }
        TextFitStrategy::ReduceSpacingFirst => {
            // First try reducing letter spacing from current down to -2.0
            let mut lc = config.layout_config.clone();
            lc.max_width = config.box_width;
            lc.max_height = config.box_height;

            let mut spacing = lc.letter_spacing;
            let min_spacing = -2.0_f32;
            let spacing_step = 0.5_f32;

            while spacing >= min_spacing {
                lc.letter_spacing = spacing;
                let engine = TextLayoutEngine::new(lc.clone());
                let layout = engine.layout(text);
                if !layout.truncated
                    && layout.total_width <= config.box_width
                    && layout.total_height <= config.box_height
                {
                    return TextFitResult {
                        fitted_font_size: lc.font_size,
                        letter_spacing: spacing,
                        layout,
                        fits: true,
                    };
                }
                spacing -= spacing_step;
            }

            // Spacing reduction didn't help, fall back to font shrink
            binary_search_font_size(text, config)
        }
    }
}

fn binary_search_font_size(text: &str, config: &TextFitConfig) -> TextFitResult {
    let mut lo = config.min_font_size;
    let mut hi = config.max_font_size;
    let mut best_layout = None;
    let mut best_size = lo;

    // Maximum iterations to prevent infinite loop
    let max_iter = 40;
    let mut iter = 0;

    while (hi - lo) > config.tolerance && iter < max_iter {
        let mid = (lo + hi) / 2.0;
        let mut lc = config.layout_config.clone();
        lc.font_size = mid;
        lc.max_width = config.box_width;
        lc.max_height = config.box_height;

        let engine = TextLayoutEngine::new(lc);
        let layout = engine.layout(text);

        let fits = !layout.truncated
            && layout.total_width <= config.box_width + 0.1
            && layout.total_height <= config.box_height + 0.1;

        if fits {
            best_size = mid;
            best_layout = Some(layout);
            lo = mid;
        } else {
            hi = mid;
        }
        iter += 1;
    }

    // Final layout at best size
    let layout = best_layout.unwrap_or_else(|| {
        let mut lc = config.layout_config.clone();
        lc.font_size = best_size;
        lc.max_width = config.box_width;
        lc.max_height = config.box_height;
        TextLayoutEngine::new(lc).layout(text)
    });

    let fits = !layout.truncated
        && layout.total_width <= config.box_width + 0.1
        && layout.total_height <= config.box_height + 0.1;

    TextFitResult {
        fitted_font_size: best_size,
        letter_spacing: config.layout_config.letter_spacing,
        layout,
        fits,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Hyphenation ─────────────────────────────────────────────────────────

    #[test]
    fn test_find_hyphenation_points_short_word() {
        // Words shorter than 4 chars should yield no hyphenation points
        assert!(find_hyphenation_points("cat").is_empty());
        assert!(find_hyphenation_points("go").is_empty());
    }

    #[test]
    fn test_find_hyphenation_points_longer_word() {
        let pts = find_hyphenation_points("beautiful");
        // Should find at least one break point
        assert!(
            !pts.is_empty(),
            "Should find hyphenation points in 'beautiful'"
        );
        // All points should be within valid range
        for &p in &pts {
            assert!(p > 0 && p < "beautiful".len());
        }
    }

    #[test]
    fn test_find_hyphenation_points_consonant_only() {
        // All consonants — fewer break opportunities
        let pts = find_hyphenation_points("rhythm");
        // Even with a tricky word, the function should not panic
        assert!(pts.len() <= 3);
    }

    // ── Word breaking ───────────────────────────────────────────────────────

    #[test]
    fn test_break_word_fits_returns_single() {
        let frags = break_word("hello", 100.0, WordBreakStrategy::Normal, |_| 10.0);
        assert_eq!(frags.len(), 1);
        assert_eq!(frags[0].0, "hello");
        assert!(!frags[0].1);
    }

    #[test]
    fn test_break_word_break_all() {
        let frags = break_word("abcdef", 25.0, WordBreakStrategy::BreakAll, |_| 10.0);
        // 6 chars at 10px each = 60px, max 25px → should split into ~3 fragments
        assert!(frags.len() >= 2, "Should break into multiple fragments");
        let total: String = frags.iter().map(|(f, _)| f.as_str()).collect();
        assert_eq!(total, "abcdef");
    }

    #[test]
    fn test_break_word_hyphenate() {
        let frags = break_word("information", 50.0, WordBreakStrategy::Hyphenate, |_| 8.0);
        // "information" is 11 chars * 8px = 88px, max 50px
        assert!(frags.len() >= 1);
        let total: String = frags.iter().map(|(f, _)| f.as_str()).collect();
        assert_eq!(total, "information");
    }

    #[test]
    fn test_break_word_normal_does_not_split() {
        let frags = break_word("toolong", 20.0, WordBreakStrategy::Normal, |_| 10.0);
        assert_eq!(frags.len(), 1);
        assert_eq!(frags[0].0, "toolong");
    }

    // ── Knuth-Plass line breaking ───────────────────────────────────────────

    #[test]
    fn test_knuth_plass_single_word() {
        let breaks = knuth_plass_breaks(&[50.0], 10.0, 200.0);
        assert_eq!(breaks, vec![0]);
    }

    #[test]
    fn test_knuth_plass_fits_one_line() {
        let widths = vec![30.0, 40.0, 50.0]; // total with spaces = 30+10+40+10+50 = 140
        let breaks = knuth_plass_breaks(&widths, 10.0, 200.0);
        assert_eq!(breaks, vec![0]); // all on one line
    }

    #[test]
    fn test_knuth_plass_forces_two_lines() {
        let widths = vec![50.0, 50.0, 50.0, 50.0]; // 50+10+50+10+50+10+50 = 230
        let breaks = knuth_plass_breaks(&widths, 10.0, 120.0);
        assert!(
            breaks.len() >= 2,
            "Should break into at least 2 lines: {:?}",
            breaks
        );
    }

    #[test]
    fn test_knuth_plass_empty() {
        let breaks = knuth_plass_breaks(&[], 10.0, 200.0);
        assert!(breaks.is_empty());
    }

    // ── CJK detection ───────────────────────────────────────────────────────

    #[test]
    fn test_is_cjk_char_chinese() {
        assert!(is_cjk_char('\u{4E2D}')); // 中
        assert!(is_cjk_char('\u{6587}')); // 文
    }

    #[test]
    fn test_is_cjk_char_hiragana() {
        assert!(is_cjk_char('\u{3042}')); // あ
    }

    #[test]
    fn test_is_cjk_char_katakana() {
        assert!(is_cjk_char('\u{30A2}')); // ア
    }

    #[test]
    fn test_is_cjk_char_latin_not_cjk() {
        assert!(!is_cjk_char('A'));
        assert!(!is_cjk_char('z'));
        assert!(!is_cjk_char('5'));
    }

    #[test]
    fn test_is_cjk_char_hangul() {
        assert!(is_cjk_char('\u{AC00}')); // 가
    }

    // ── Vertical layout ─────────────────────────────────────────────────────

    #[test]
    fn test_vertical_layout_basic() {
        let config = VerticalLayoutConfig::default();
        let result = layout_vertical("ABCD", &config);
        assert_eq!(result.columns.len(), 1);
        assert_eq!(result.chars_fitted, 4);
    }

    #[test]
    fn test_vertical_layout_wraps_to_columns() {
        let config = VerticalLayoutConfig {
            font_size: 20.0,
            max_height: 50.0, // fits ~2 chars per column (20+2=22 per char)
            char_spacing: 2.0,
            column_spacing: 4.0,
            orientation: TextOrientation::VerticalLtr,
            ..Default::default()
        };
        let result = layout_vertical("ABCDEF", &config);
        assert!(result.columns.len() >= 2, "Should use multiple columns");
        assert_eq!(result.chars_fitted, 6);
    }

    #[test]
    fn test_vertical_layout_newline_creates_column() {
        let config = VerticalLayoutConfig {
            max_height: 1000.0,
            ..Default::default()
        };
        let result = layout_vertical("AB\nCD", &config);
        assert_eq!(result.columns.len(), 2);
    }

    #[test]
    fn test_vertical_layout_empty_text() {
        let config = VerticalLayoutConfig::default();
        let result = layout_vertical("", &config);
        assert!(result.columns.is_empty());
        assert_eq!(result.chars_fitted, 0);
    }

    #[test]
    fn test_vertical_layout_rtl_column_order() {
        let config = VerticalLayoutConfig {
            max_height: 50.0,
            font_size: 20.0,
            char_spacing: 2.0,
            orientation: TextOrientation::VerticalRtl,
            ..Default::default()
        };
        let result = layout_vertical("ABCDEF", &config);
        // In RTL vertical, the first logical column (containing 'A','B')
        // should appear at the rightmost position. After our reversal,
        // column ordering in the vec goes right-to-left (highest x first).
        if result.columns.len() >= 2 {
            // The first glyph 'A' should be in a column with the highest x.
            let first_char_col = result
                .columns
                .iter()
                .find(|c| c.glyphs.iter().any(|g| g.character == 'A'));
            let last_char_col = result.columns.iter().find(|c| {
                c.glyphs
                    .iter()
                    .any(|g| g.char_index == result.chars_fitted - 1)
            });
            if let (Some(fc), Some(lc)) = (first_char_col, last_char_col) {
                assert!(
                    fc.x_offset >= lc.x_offset,
                    "RTL: first-char column x={} should be >= last-char column x={}",
                    fc.x_offset,
                    lc.x_offset
                );
            }
        }
    }

    // ── Text fitting ────────────────────────────────────────────────────────

    #[test]
    fn test_fit_text_empty() {
        let config = TextFitConfig::default();
        let result = fit_text("", &config);
        assert!(result.fits);
    }

    #[test]
    fn test_fit_text_short_text_fits() {
        let config = TextFitConfig {
            box_width: 800.0,
            box_height: 600.0,
            max_font_size: 100.0,
            ..Default::default()
        };
        let result = fit_text("Hi", &config);
        assert!(result.fits);
        assert!(result.fitted_font_size > 0.0);
    }

    #[test]
    fn test_fit_text_shrinks_for_long_text() {
        let config = TextFitConfig {
            box_width: 200.0,
            box_height: 50.0,
            min_font_size: 6.0,
            max_font_size: 100.0,
            layout_config: TextLayoutConfig {
                font_size: 48.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = fit_text(
            "This is a moderately long text that should be shrunk",
            &config,
        );
        // Font size should have been reduced from the max
        assert!(result.fitted_font_size < 100.0);
    }

    #[test]
    fn test_fit_text_grow_to_fill() {
        let config = TextFitConfig {
            box_width: 800.0,
            box_height: 400.0,
            min_font_size: 6.0,
            max_font_size: 200.0,
            strategy: TextFitStrategy::GrowToFill,
            ..Default::default()
        };
        let result = fit_text("Hi", &config);
        assert!(result.fits);
        // For a short word in a big box, font should grow large
        assert!(result.fitted_font_size > 20.0);
    }

    #[test]
    fn test_fit_text_reduce_spacing_first() {
        let config = TextFitConfig {
            box_width: 200.0,
            box_height: 100.0,
            strategy: TextFitStrategy::ReduceSpacingFirst,
            layout_config: TextLayoutConfig {
                font_size: 16.0,
                letter_spacing: 5.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = fit_text("Some text here", &config);
        // Should at least attempt to adjust
        assert!(result.fitted_font_size > 0.0);
    }

    // ── TextOrientation enum ────────────────────────────────────────────────

    #[test]
    fn test_text_orientation_default() {
        assert_eq!(TextOrientation::default(), TextOrientation::Horizontal);
    }

    #[test]
    fn test_word_break_strategy_default() {
        assert_eq!(WordBreakStrategy::default(), WordBreakStrategy::Normal);
    }

    #[test]
    fn test_text_fit_strategy_default() {
        assert_eq!(TextFitStrategy::default(), TextFitStrategy::ShrinkFont);
    }

    #[test]
    fn test_vertical_layout_max_columns_limit() {
        let config = VerticalLayoutConfig {
            max_height: 30.0,
            font_size: 20.0,
            char_spacing: 2.0,
            max_columns: 2,
            ..Default::default()
        };
        // 10 chars, ~1 per column at 30px height → wants many columns, limited to 2
        let result = layout_vertical("ABCDEFGHIJ", &config);
        assert!(result.columns.len() <= 2);
    }
}
