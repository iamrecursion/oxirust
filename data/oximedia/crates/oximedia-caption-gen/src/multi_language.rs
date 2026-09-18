//! Multi-language / bilingual caption layout.
//!
//! Bilingual captions display two languages simultaneously — the original
//! audio language and a translation — on screen.  This module provides:
//!
//! - [`BilingualBlock`]: a caption block that carries two language tracks.
//! - [`BilingualLayout`]: layout strategies for positioning the two tracks.
//! - [`merge_bilingual`]: merges two aligned [`CaptionBlock`] slices into a
//!   `Vec<BilingualBlock>` using timestamp overlap matching.
//! - [`split_bilingual`]: extracts a single language track from a bilingual
//!   track.
//! - [`validate_bilingual_sync`]: checks that the two tracks are well-aligned.
//!
//! ## Layout strategies
//!
//! | Strategy | Description |
//! |----------|-------------|
//! | `PrimaryBottom` | Primary at bottom, secondary just above (most common). |
//! | `PrimaryTop` | Primary at top, secondary below. |
//! | `SideBySide` | Both languages in the same block, separated by `\|`. |
//! | `InterleavedLines` | Primary and secondary lines interleaved. |

use crate::alignment::{CaptionBlock, CaptionPosition};

// ─── Layout strategy ─────────────────────────────────────────────────────────

/// Describes how two language tracks are positioned on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BilingualLayout {
    /// Primary language at the bottom, secondary just above.
    PrimaryBottom,
    /// Primary language at the top, secondary just below.
    PrimaryTop,
    /// Both languages merged into one block with `|` separator.
    SideBySide,
    /// Lines from each language interleaved (primary line, secondary line, …).
    InterleavedLines,
}

// ─── Bilingual block ─────────────────────────────────────────────────────────

/// A caption block that carries text in two languages simultaneously.
#[derive(Debug, Clone, PartialEq)]
pub struct BilingualBlock {
    /// Sequential 1-based identifier.
    pub id: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    /// Lines of text in the primary (original) language.
    pub primary_lines: Vec<String>,
    /// Lines of text in the secondary (translated) language.
    pub secondary_lines: Vec<String>,
    /// Language code for the primary track (BCP-47, e.g. `"en"`).
    pub primary_lang: String,
    /// Language code for the secondary track.
    pub secondary_lang: String,
    /// Layout strategy to use when rendering.
    pub layout: BilingualLayout,
}

impl BilingualBlock {
    /// Total display duration of this block.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Render the bilingual block into a `Vec<String>` according to `layout`.
    ///
    /// The returned lines can be passed directly to a caption renderer or
    /// export function.
    pub fn render(&self) -> Vec<String> {
        match self.layout {
            BilingualLayout::PrimaryBottom => {
                let mut lines = self.secondary_lines.clone();
                lines.extend_from_slice(&self.primary_lines);
                lines
            }
            BilingualLayout::PrimaryTop => {
                let mut lines = self.primary_lines.clone();
                lines.extend_from_slice(&self.secondary_lines);
                lines
            }
            BilingualLayout::SideBySide => {
                // Zip primary and secondary lines together with " | " separator.
                // If lengths differ, pad shorter side with empty strings.
                let max_len = self.primary_lines.len().max(self.secondary_lines.len());
                (0..max_len)
                    .map(|i| {
                        let p = self.primary_lines.get(i).map(|s| s.as_str()).unwrap_or("");
                        let s = self
                            .secondary_lines
                            .get(i)
                            .map(|s| s.as_str())
                            .unwrap_or("");
                        format!("{p} | {s}")
                    })
                    .collect()
            }
            BilingualLayout::InterleavedLines => {
                let max_len = self.primary_lines.len().max(self.secondary_lines.len());
                let mut rendered = Vec::with_capacity(max_len * 2);
                for i in 0..max_len {
                    if let Some(p) = self.primary_lines.get(i) {
                        rendered.push(p.clone());
                    }
                    if let Some(s) = self.secondary_lines.get(i) {
                        rendered.push(s.clone());
                    }
                }
                rendered
            }
        }
    }

    /// Convert this bilingual block to a [`CaptionBlock`] using the
    /// `render()` output as the lines, positioned at the `Bottom`.
    pub fn to_caption_block(&self) -> CaptionBlock {
        CaptionBlock {
            id: self.id,
            start_ms: self.start_ms,
            end_ms: self.end_ms,
            lines: self.render(),
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }
}

// ─── Merge two tracks ─────────────────────────────────────────────────────────

/// Configuration for merging two tracks into bilingual blocks.
#[derive(Debug, Clone)]
pub struct BilingualMergeConfig {
    /// Language code of the primary (original) track.
    pub primary_lang: String,
    /// Language code of the secondary (translated) track.
    pub secondary_lang: String,
    /// Layout strategy to use.
    pub layout: BilingualLayout,
    /// Minimum overlap (in milliseconds) between a primary and secondary block
    /// for them to be paired.  Pairs with less overlap are handled as
    /// unmatched blocks.
    pub min_overlap_ms: u64,
}

impl BilingualMergeConfig {
    /// Create a default config for English + Spanish.
    pub fn en_es() -> Self {
        Self {
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryBottom,
            min_overlap_ms: 100,
        }
    }
}

/// Merge two aligned caption tracks into a `Vec<BilingualBlock>`.
///
/// For each block in `primary`, the secondary block that has the greatest
/// temporal overlap (≥ `config.min_overlap_ms`) is paired with it.  If no
/// secondary block meets the threshold, the bilingual block has empty
/// `secondary_lines`.
///
/// The returned list is sorted by `start_ms`.
pub fn merge_bilingual(
    primary: &[CaptionBlock],
    secondary: &[CaptionBlock],
    config: &BilingualMergeConfig,
) -> Vec<BilingualBlock> {
    let mut result: Vec<BilingualBlock> = Vec::with_capacity(primary.len());

    for (idx, pblock) in primary.iter().enumerate() {
        // Find best-overlapping secondary block.
        let best_secondary = secondary
            .iter()
            .filter_map(|sblock| {
                let overlap_start = pblock.start_ms.max(sblock.start_ms);
                let overlap_end = pblock.end_ms.min(sblock.end_ms);
                if overlap_end > overlap_start
                    && overlap_end - overlap_start >= config.min_overlap_ms
                {
                    Some((sblock, overlap_end - overlap_start))
                } else {
                    None
                }
            })
            .max_by_key(|(_, overlap)| *overlap)
            .map(|(sblock, _)| sblock);

        let secondary_lines = best_secondary.map(|b| b.lines.clone()).unwrap_or_default();

        result.push(BilingualBlock {
            id: (idx as u32) + 1,
            start_ms: pblock.start_ms,
            end_ms: pblock.end_ms,
            primary_lines: pblock.lines.clone(),
            secondary_lines,
            primary_lang: config.primary_lang.clone(),
            secondary_lang: config.secondary_lang.clone(),
            layout: config.layout,
        });
    }

    result.sort_by_key(|b| b.start_ms);
    result
}

// ─── Split bilingual track ────────────────────────────────────────────────────

/// Which language track to extract from a bilingual track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BilingualTrackSide {
    Primary,
    Secondary,
}

/// Extract a single language track from a bilingual track.
///
/// Returns a `Vec<CaptionBlock>` containing only the requested language's
/// lines, with the same timestamps as the original bilingual blocks.
/// Blocks with no lines for the requested side are omitted.
pub fn split_bilingual(blocks: &[BilingualBlock], side: BilingualTrackSide) -> Vec<CaptionBlock> {
    blocks
        .iter()
        .filter_map(|b| {
            let lines = match side {
                BilingualTrackSide::Primary => b.primary_lines.clone(),
                BilingualTrackSide::Secondary => b.secondary_lines.clone(),
            };
            if lines.is_empty() {
                None
            } else {
                Some(CaptionBlock {
                    id: b.id,
                    start_ms: b.start_ms,
                    end_ms: b.end_ms,
                    lines,
                    speaker_id: None,
                    position: CaptionPosition::Bottom,
                })
            }
        })
        .collect()
}

// ─── Synchronisation validation ───────────────────────────────────────────────

/// A synchronisation warning for a bilingual track.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncWarning {
    /// 1-based index of the primary block with the issue.
    pub primary_block_id: u32,
    /// Human-readable description of the synchronisation problem.
    pub message: String,
    /// Timing offset at which the problem occurs.
    pub timestamp_ms: u64,
}

/// Validate that bilingual blocks are well-synchronised.
///
/// Checks performed:
/// - Blocks with no secondary text are flagged.
/// - Blocks where the secondary text is much longer than the primary are
///   flagged (would require very fast reading speed).
pub fn validate_bilingual_sync(blocks: &[BilingualBlock]) -> Vec<SyncWarning> {
    let mut warnings: Vec<SyncWarning> = Vec::new();

    for block in blocks {
        if block.secondary_lines.is_empty() {
            warnings.push(SyncWarning {
                primary_block_id: block.id,
                message: format!(
                    "Block {} has no secondary-language text ({}→{})",
                    block.id, block.primary_lang, block.secondary_lang
                ),
                timestamp_ms: block.start_ms,
            });
            continue;
        }

        // Check relative text length: secondary > 1.5× primary may be too fast.
        let primary_chars: usize = block.primary_lines.iter().map(|l| l.chars().count()).sum();
        let secondary_chars: usize = block
            .secondary_lines
            .iter()
            .map(|l| l.chars().count())
            .sum();

        if secondary_chars > 0 && primary_chars > 0 {
            let ratio = secondary_chars as f32 / primary_chars as f32;
            if ratio > 1.5 {
                warnings.push(SyncWarning {
                    primary_block_id: block.id,
                    message: format!(
                        "Block {}: secondary text ({secondary_chars} chars) is {:.1}× longer than \
                         primary ({primary_chars} chars); may be too fast to read",
                        block.id, ratio
                    ),
                    timestamp_ms: block.start_ms,
                });
            }
        }
    }

    warnings
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_caption(id: u32, start_ms: u64, end_ms: u64, text: &str) -> CaptionBlock {
        CaptionBlock {
            id,
            start_ms,
            end_ms,
            lines: vec![text.to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }

    fn default_config() -> BilingualMergeConfig {
        BilingualMergeConfig::en_es()
    }

    // ─── BilingualBlock::render ───────────────────────────────────────────────

    #[test]
    fn render_primary_bottom_secondary_on_top() {
        let block = BilingualBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            primary_lines: vec!["Hello".to_string()],
            secondary_lines: vec!["Hola".to_string()],
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryBottom,
        };
        let rendered = block.render();
        // Secondary at top, primary at bottom.
        assert_eq!(rendered[0], "Hola");
        assert_eq!(rendered[1], "Hello");
    }

    #[test]
    fn render_primary_top() {
        let block = BilingualBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            primary_lines: vec!["Hello".to_string()],
            secondary_lines: vec!["Hola".to_string()],
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryTop,
        };
        let rendered = block.render();
        assert_eq!(rendered[0], "Hello");
        assert_eq!(rendered[1], "Hola");
    }

    #[test]
    fn render_side_by_side() {
        let block = BilingualBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            primary_lines: vec!["Hello".to_string()],
            secondary_lines: vec!["Hola".to_string()],
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::SideBySide,
        };
        let rendered = block.render();
        assert_eq!(rendered.len(), 1);
        assert!(rendered[0].contains("Hello"));
        assert!(rendered[0].contains("Hola"));
        assert!(rendered[0].contains(" | "));
    }

    #[test]
    fn render_interleaved() {
        let block = BilingualBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            primary_lines: vec!["Hello".to_string(), "World".to_string()],
            secondary_lines: vec!["Hola".to_string(), "Mundo".to_string()],
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::InterleavedLines,
        };
        let rendered = block.render();
        assert_eq!(rendered.len(), 4);
        assert_eq!(rendered[0], "Hello");
        assert_eq!(rendered[1], "Hola");
        assert_eq!(rendered[2], "World");
        assert_eq!(rendered[3], "Mundo");
    }

    // ─── merge_bilingual ──────────────────────────────────────────────────────

    #[test]
    fn merge_pairs_overlapping_blocks() {
        let primary = vec![make_caption(1, 0, 2000, "Hello")];
        let secondary = vec![make_caption(1, 500, 2500, "Hola")];
        let result = merge_bilingual(&primary, &secondary, &default_config());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].primary_lines, vec!["Hello"]);
        assert_eq!(result[0].secondary_lines, vec!["Hola"]);
    }

    #[test]
    fn merge_empty_secondary_when_no_overlap() {
        let primary = vec![make_caption(1, 0, 1000, "Hello")];
        let secondary = vec![make_caption(1, 5000, 7000, "Hola")]; // far away
        let result = merge_bilingual(&primary, &secondary, &default_config());
        assert_eq!(result.len(), 1);
        assert!(result[0].secondary_lines.is_empty());
    }

    #[test]
    fn merge_picks_best_overlapping_secondary() {
        let primary = vec![make_caption(1, 0, 3000, "Hello")];
        let secondary = vec![
            make_caption(1, 0, 500, "Short"), // 500ms overlap
            make_caption(2, 0, 2000, "Long"), // 2000ms overlap — should win
        ];
        let result = merge_bilingual(&primary, &secondary, &default_config());
        assert_eq!(result[0].secondary_lines, vec!["Long"]);
    }

    #[test]
    fn merge_empty_primary_returns_empty() {
        let secondary = vec![make_caption(1, 0, 1000, "Hola")];
        let result = merge_bilingual(&[], &secondary, &default_config());
        assert!(result.is_empty());
    }

    #[test]
    fn merge_preserves_timestamps() {
        let primary = vec![make_caption(1, 1000, 3000, "Hello")];
        let secondary = vec![make_caption(1, 1200, 3200, "Hola")];
        let result = merge_bilingual(&primary, &secondary, &default_config());
        assert_eq!(result[0].start_ms, 1000);
        assert_eq!(result[0].end_ms, 3000);
    }

    // ─── split_bilingual ──────────────────────────────────────────────────────

    #[test]
    fn split_primary_side() {
        let blocks = vec![BilingualBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            primary_lines: vec!["Hello".to_string()],
            secondary_lines: vec!["Hola".to_string()],
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryBottom,
        }];
        let extracted = split_bilingual(&blocks, BilingualTrackSide::Primary);
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].lines, vec!["Hello"]);
    }

    #[test]
    fn split_secondary_side() {
        let blocks = vec![BilingualBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            primary_lines: vec!["Hello".to_string()],
            secondary_lines: vec!["Hola".to_string()],
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryBottom,
        }];
        let extracted = split_bilingual(&blocks, BilingualTrackSide::Secondary);
        assert_eq!(extracted[0].lines, vec!["Hola"]);
    }

    #[test]
    fn split_omits_blocks_with_empty_side() {
        let blocks = vec![BilingualBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            primary_lines: vec!["Hello".to_string()],
            secondary_lines: vec![], // empty
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryBottom,
        }];
        let extracted = split_bilingual(&blocks, BilingualTrackSide::Secondary);
        assert!(extracted.is_empty());
    }

    // ─── validate_bilingual_sync ──────────────────────────────────────────────

    #[test]
    fn sync_warn_empty_secondary() {
        let blocks = vec![BilingualBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            primary_lines: vec!["Hello".to_string()],
            secondary_lines: vec![],
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryBottom,
        }];
        let warnings = validate_bilingual_sync(&blocks);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("no secondary-language text"));
    }

    #[test]
    fn sync_warn_secondary_too_long() {
        let blocks = vec![BilingualBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            primary_lines: vec!["Hi".to_string()], // 2 chars
            secondary_lines: vec!["Esta es una frase muy larga".to_string()], // 27 chars → 13.5x
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryBottom,
        }];
        let warnings = validate_bilingual_sync(&blocks);
        assert!(!warnings.is_empty());
        assert!(warnings[0].message.contains("too fast to read"));
    }

    #[test]
    fn sync_no_warnings_for_balanced_blocks() {
        let blocks = vec![BilingualBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            primary_lines: vec!["Hello world".to_string()],
            secondary_lines: vec!["Hola mundo".to_string()],
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryBottom,
        }];
        let warnings = validate_bilingual_sync(&blocks);
        assert!(warnings.is_empty());
    }

    // ─── BilingualBlock::to_caption_block ─────────────────────────────────────

    #[test]
    fn to_caption_block_preserves_id_and_timestamps() {
        let block = BilingualBlock {
            id: 5,
            start_ms: 1000,
            end_ms: 3000,
            primary_lines: vec!["Hello".to_string()],
            secondary_lines: vec!["Hola".to_string()],
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryBottom,
        };
        let caption = block.to_caption_block();
        assert_eq!(caption.id, 5);
        assert_eq!(caption.start_ms, 1000);
        assert_eq!(caption.end_ms, 3000);
    }

    // ─── duration_ms ──────────────────────────────────────────────────────────

    #[test]
    fn bilingual_block_duration() {
        let block = BilingualBlock {
            id: 1,
            start_ms: 1000,
            end_ms: 4000,
            primary_lines: vec!["Hello".to_string()],
            secondary_lines: vec!["Hola".to_string()],
            primary_lang: "en".to_string(),
            secondary_lang: "es".to_string(),
            layout: BilingualLayout::PrimaryBottom,
        };
        assert_eq!(block.duration_ms(), 3000);
    }
}
