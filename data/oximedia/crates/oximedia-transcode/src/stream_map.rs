//! FFmpeg-style `--map` stream selection for the packet-level pipeline.
//!
//! This module parses the single-input subset of FFmpeg's `-map` selector
//! grammar and resolves a set of selectors against the streams probed from
//! a demuxer:
//!
//! ```text
//! selector := ['-'] 0 [':' part [':' index]]
//! part     := 'v' | 'a' | 's' | <stream index>
//! ```
//!
//! Examples: `0` (all streams), `0:v` (all video), `0:a:1` (second audio
//! stream), `0:1` (stream index 1), `-0:a` (exclude all audio).
//!
//! # Resolution semantics
//!
//! * An empty selector list keeps every stream (today's default behavior).
//! * Positive selectors union; negative selectors subtract.
//! * When *only* negative selectors are given, the subtraction starts from
//!   the full stream set (so `--map -0:a` means "everything except audio").
//!   Strict FFmpeg would start from an empty mapping and thus always end
//!   empty — an error under the rules below — so the only useful reading
//!   is adopted instead.
//! * A positive selector that matches nothing is an error, as is a final
//!   selection that is empty while the selector list was not.
//!
//! The resolved result is the ordered list of **original** demuxer stream
//! indices to keep. Callers that feed positional muxers (Matroska, Ogg)
//! must additionally remap each surviving packet's `stream_index` to its
//! position in the filtered list — see [`build_index_remap`].
//!
//! This is deliberately a fresh parser rather than a reuse of
//! `oximedia-compat-ffmpeg`'s private `parse_map_spec`: that crate depends
//! on this one, so the reverse edge would create a dependency cycle.

use crate::{Result, TranscodeError};
use oximedia_container::StreamInfo;
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// Human-readable summary of the accepted selector grammar, embedded in
/// every parse/resolution error so failures are directly actionable.
const VALID_SYNTAX: &str = "valid --map selectors: '0' (all streams), \
     '0:v' / '0:a' / '0:s' (all video/audio/subtitle streams), \
     '0:N' (stream with index N), \
     '0:v:N' / '0:a:N' / '0:s:N' (N-th stream of that kind, 0-based); \
     prefix with '-' to exclude instead (e.g. '-0:a')";

// ─── StreamKind ───────────────────────────────────────────────────────────────

/// Media kind addressed by a kind-based `--map` selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamKind {
    /// Video streams (`v`).
    Video,
    /// Audio streams (`a`).
    Audio,
    /// Subtitle streams (`s`).
    Subtitle,
}

impl StreamKind {
    /// Returns `true` when `stream` is of this kind.
    #[must_use]
    pub fn matches(&self, stream: &StreamInfo) -> bool {
        match self {
            Self::Video => stream.is_video(),
            Self::Audio => stream.is_audio(),
            Self::Subtitle => stream.is_subtitle(),
        }
    }

    /// The single-character FFmpeg stream-specifier for this kind.
    #[must_use]
    pub const fn spec_char(&self) -> char {
        match self {
            Self::Video => 'v',
            Self::Audio => 'a',
            Self::Subtitle => 's',
        }
    }

    /// Parses a single-character kind specifier (`v`, `a`, `s`).
    fn from_spec(part: &str) -> Option<Self> {
        match part {
            "v" => Some(Self::Video),
            "a" => Some(Self::Audio),
            "s" => Some(Self::Subtitle),
            _ => None,
        }
    }
}

// ─── StreamMapSelector ────────────────────────────────────────────────────────

/// What a single `--map` selector addresses within the (single) input file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamMapSelector {
    /// Every stream (`0`).
    All,
    /// One stream by its demuxer index (`0:N`).
    Index(usize),
    /// Every stream of a kind (`0:v`, `0:a`, `0:s`).
    Kind(StreamKind),
    /// The N-th stream of a kind, 0-based (`0:a:1`).
    KindIndex(StreamKind, usize),
}

impl fmt::Display for StreamMapSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => write!(f, "0"),
            Self::Index(n) => write!(f, "0:{n}"),
            Self::Kind(k) => write!(f, "0:{}", k.spec_char()),
            Self::KindIndex(k, n) => write!(f, "0:{}:{n}", k.spec_char()),
        }
    }
}

// ─── StreamMap ────────────────────────────────────────────────────────────────

/// One parsed `--map` argument: a selector plus its include/exclude polarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamMap {
    /// `true` for exclusion selectors (leading `-`, e.g. `-0:a`).
    pub negative: bool,
    /// The stream(s) this map addresses.
    pub selector: StreamMapSelector,
}

impl fmt::Display for StreamMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negative {
            write!(f, "-")?;
        }
        self.selector.fmt(f)
    }
}

impl StreamMap {
    /// Parses one FFmpeg-style `--map` selector string.
    ///
    /// Accepted forms (single input, so the file index must be `0`):
    /// `0`, `0:v`, `0:a`, `0:s`, `0:N`, `0:v:N`, `0:a:N`, `0:s:N`, each
    /// optionally prefixed with `-` for exclusion.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] — with the full accepted
    /// grammar in the message — for anything outside that subset, including
    /// non-zero input-file indices and filter-graph labels.
    pub fn parse(s: &str) -> Result<Self> {
        let trimmed = s.trim();
        let invalid = |detail: &str| {
            TranscodeError::InvalidInput(format!(
                "invalid --map selector '{s}': {detail}; {VALID_SYNTAX}"
            ))
        };

        let (negative, body) = match trimmed.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, trimmed),
        };
        if body.is_empty() {
            return Err(invalid("empty selector"));
        }

        let parts: Vec<&str> = body.split(':').collect();
        if parts.len() > 3 {
            return Err(invalid("too many ':'-separated components (max 3)"));
        }

        // Component 1: input-file index. Only a single input is supported,
        // so anything but `0` cannot ever match.
        match parts[0].parse::<usize>() {
            Ok(0) => {}
            Ok(n) => {
                return Err(invalid(&format!(
                    "input file index {n} is out of range (single-input pipeline; only 0 exists)"
                )));
            }
            Err(_) => {
                return Err(invalid(&format!(
                    "'{}' is not a valid input file index",
                    parts[0]
                )));
            }
        }

        // Component 2 (optional): stream kind or absolute stream index.
        let selector = match parts.len() {
            1 => StreamMapSelector::All,
            2 => {
                if let Some(kind) = StreamKind::from_spec(parts[1]) {
                    StreamMapSelector::Kind(kind)
                } else if let Ok(index) = parts[1].parse::<usize>() {
                    StreamMapSelector::Index(index)
                } else {
                    return Err(invalid(&format!(
                        "'{}' is neither a stream kind (v/a/s) nor a stream index",
                        parts[1]
                    )));
                }
            }
            _ => {
                // Component 3 requires component 2 to be a kind.
                let Some(kind) = StreamKind::from_spec(parts[1]) else {
                    return Err(invalid(&format!(
                        "'{}' must be a stream kind (v/a/s) when a per-kind index follows",
                        parts[1]
                    )));
                };
                let Ok(kind_index) = parts[2].parse::<usize>() else {
                    return Err(invalid(&format!(
                        "'{}' is not a valid per-kind stream index",
                        parts[2]
                    )));
                };
                StreamMapSelector::KindIndex(kind, kind_index)
            }
        };

        Ok(Self { negative, selector })
    }
}

// ─── Resolution ───────────────────────────────────────────────────────────────

/// Collects the original stream indices matched by `selector`.
fn select_matching(streams: &[StreamInfo], selector: &StreamMapSelector) -> Vec<usize> {
    match selector {
        StreamMapSelector::All => streams.iter().map(|s| s.index).collect(),
        StreamMapSelector::Index(n) => streams
            .iter()
            .filter(|s| s.index == *n)
            .map(|s| s.index)
            .collect(),
        StreamMapSelector::Kind(kind) => streams
            .iter()
            .filter(|s| kind.matches(s))
            .map(|s| s.index)
            .collect(),
        StreamMapSelector::KindIndex(kind, nth) => streams
            .iter()
            .filter(|s| kind.matches(s))
            .nth(*nth)
            .map(|s| s.index)
            .into_iter()
            .collect(),
    }
}

/// Formats the available streams as `index:kind` pairs for error messages.
fn describe_streams(streams: &[StreamInfo]) -> String {
    let entries: Vec<String> = streams
        .iter()
        .map(|s| {
            let kind = if s.is_video() {
                'v'
            } else if s.is_audio() {
                'a'
            } else if s.is_subtitle() {
                's'
            } else {
                '?'
            };
            format!("{}:{}", s.index, kind)
        })
        .collect();
    format!("{} stream(s): [{}]", streams.len(), entries.join(", "))
}

/// Resolves a `--map` selector list against the probed input streams.
///
/// Returns the **original** demuxer stream indices to keep, in ascending
/// order. See the module docs for the union/subtract semantics.
///
/// # Errors
///
/// Returns [`TranscodeError::InvalidInput`] when a positive selector
/// matches no stream, or when the final selection is empty while `maps`
/// was non-empty. Both messages include the input's stream inventory and
/// the accepted selector grammar.
pub fn resolve_stream_selection(streams: &[StreamInfo], maps: &[StreamMap]) -> Result<Vec<usize>> {
    if maps.is_empty() {
        return Ok(streams.iter().map(|s| s.index).collect());
    }

    let has_positive = maps.iter().any(|m| !m.negative);

    // Purely-negative selector lists subtract from the full stream set;
    // otherwise start empty and union the positive matches in.
    let mut kept: BTreeSet<usize> = if has_positive {
        BTreeSet::new()
    } else {
        streams.iter().map(|s| s.index).collect()
    };

    // FFmpeg processing order: create mappings first, then apply negatives.
    for map in maps.iter().filter(|m| !m.negative) {
        let matched = select_matching(streams, &map.selector);
        if matched.is_empty() {
            return Err(TranscodeError::InvalidInput(format!(
                "--map selector '{map}' matched no streams; input has {}; {VALID_SYNTAX}",
                describe_streams(streams)
            )));
        }
        kept.extend(matched);
    }
    for map in maps.iter().filter(|m| m.negative) {
        for index in select_matching(streams, &map.selector) {
            kept.remove(&index);
        }
    }

    if kept.is_empty() {
        return Err(TranscodeError::InvalidInput(format!(
            "--map selection removed every stream; input has {}; {VALID_SYNTAX}",
            describe_streams(streams)
        )));
    }

    Ok(kept.into_iter().collect())
}

/// Builds the original-index → sequential-output-index remap table from the
/// kept indices returned by [`resolve_stream_selection`].
///
/// Both the Matroska and Ogg muxers route `write_packet` by the packet's
/// **position in the muxer's own stream list**, not by original stream
/// identity, so after filtering, every surviving packet's `stream_index`
/// must be rewritten through this table (and packets from unselected
/// streams dropped) before muxing.
#[must_use]
pub fn build_index_remap(kept_indices: &[usize]) -> HashMap<usize, usize> {
    kept_indices
        .iter()
        .enumerate()
        .map(|(new_index, &original_index)| (original_index, new_index))
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::{CodecId, MediaType, Rational};

    /// Builds a `StreamInfo` with the given index and media type
    /// (`StreamInfo::new` derives `media_type` from the codec).
    fn stream(index: usize, media_type: MediaType) -> StreamInfo {
        let codec = match media_type {
            MediaType::Video => CodecId::Vp9,
            MediaType::Audio => CodecId::Opus,
            _ => CodecId::Srt,
        };
        StreamInfo::new(index, codec, Rational::new(1, 1000))
    }

    /// Standard 4-stream fixture: video, audio, audio, subtitle.
    fn vaas() -> Vec<StreamInfo> {
        vec![
            stream(0, MediaType::Video),
            stream(1, MediaType::Audio),
            stream(2, MediaType::Audio),
            stream(3, MediaType::Subtitle),
        ]
    }

    fn parse(s: &str) -> StreamMap {
        StreamMap::parse(s).unwrap_or_else(|e| panic!("'{s}' should parse: {e}"))
    }

    // ── parse: accepted grammar ────────────────────────────────────────────

    #[test]
    fn parse_all_streams() {
        let map = parse("0");
        assert!(!map.negative);
        assert_eq!(map.selector, StreamMapSelector::All);
    }

    #[test]
    fn parse_absolute_index() {
        assert_eq!(parse("0:0").selector, StreamMapSelector::Index(0));
        assert_eq!(parse("0:7").selector, StreamMapSelector::Index(7));
    }

    #[test]
    fn parse_kinds() {
        assert_eq!(
            parse("0:v").selector,
            StreamMapSelector::Kind(StreamKind::Video)
        );
        assert_eq!(
            parse("0:a").selector,
            StreamMapSelector::Kind(StreamKind::Audio)
        );
        assert_eq!(
            parse("0:s").selector,
            StreamMapSelector::Kind(StreamKind::Subtitle)
        );
    }

    #[test]
    fn parse_kind_index() {
        assert_eq!(
            parse("0:a:1").selector,
            StreamMapSelector::KindIndex(StreamKind::Audio, 1)
        );
        assert_eq!(
            parse("0:v:0").selector,
            StreamMapSelector::KindIndex(StreamKind::Video, 0)
        );
    }

    #[test]
    fn parse_negative() {
        let map = parse("-0:a");
        assert!(map.negative);
        assert_eq!(map.selector, StreamMapSelector::Kind(StreamKind::Audio));
    }

    #[test]
    fn parse_trims_whitespace() {
        assert_eq!(
            parse("  0:v ").selector,
            StreamMapSelector::Kind(StreamKind::Video)
        );
    }

    #[test]
    fn display_round_trips() {
        for s in ["0", "0:3", "0:v", "0:a:1", "-0:a", "-0:v:2"] {
            assert_eq!(parse(s).to_string(), s, "Display must round-trip '{s}'");
        }
    }

    // ── parse: rejections carry the grammar in the message ────────────────

    #[test]
    fn parse_rejections() {
        for bad in [
            "", " ", "-", "1", "1:v", "0:x", "0:", "0:v:", "0:v:x", "0:v:1:2", "abc", "0:-1",
            "0:V", "[v0]", "0:a:-1",
        ] {
            let err = StreamMap::parse(bad).expect_err(&format!("'{bad}' must be rejected"));
            let msg = err.to_string();
            assert!(
                msg.contains("valid --map selectors"),
                "error for '{bad}' must list the valid syntax, got: {msg}"
            );
        }
    }

    #[test]
    fn parse_nonzero_file_index_names_the_problem() {
        let msg = StreamMap::parse("1:v")
            .expect_err("file index 1 must be rejected")
            .to_string();
        assert!(
            msg.contains("single-input"),
            "must explain the single-input constraint, got: {msg}"
        );
    }

    // ── resolve: semantics ─────────────────────────────────────────────────

    #[test]
    fn resolve_empty_maps_keeps_all() {
        let kept = resolve_stream_selection(&vaas(), &[]).expect("empty maps keep every stream");
        assert_eq!(kept, vec![0, 1, 2, 3]);
    }

    #[test]
    fn resolve_kind_selects_all_of_kind() {
        let kept = resolve_stream_selection(&vaas(), &[parse("0:a")])
            .expect("audio selector should resolve");
        assert_eq!(kept, vec![1, 2]);
    }

    #[test]
    fn resolve_positive_union() {
        let kept = resolve_stream_selection(&vaas(), &[parse("0:v"), parse("0:a:1")])
            .expect("union should resolve");
        assert_eq!(kept, vec![0, 2]);
    }

    #[test]
    fn resolve_index_selector() {
        let kept = resolve_stream_selection(&vaas(), &[parse("0:3")])
            .expect("index selector should resolve");
        assert_eq!(kept, vec![3]);
    }

    #[test]
    fn resolve_negative_subtracts_from_positive() {
        let kept = resolve_stream_selection(&vaas(), &[parse("0"), parse("-0:s")])
            .expect("all-minus-subtitle should resolve");
        assert_eq!(kept, vec![0, 1, 2]);
    }

    #[test]
    fn resolve_pure_negative_starts_from_all() {
        let kept = resolve_stream_selection(&vaas(), &[parse("-0:a")])
            .expect("pure-negative subtracts from the full set");
        assert_eq!(kept, vec![0, 3]);
    }

    #[test]
    fn resolve_negative_matching_nothing_is_noop() {
        // No subtitles matched by -0:s:5 — nothing to remove, not an error.
        let kept = resolve_stream_selection(&vaas(), &[parse("0:v"), parse("-0:s:5")])
            .expect("non-matching negatives are no-ops");
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn resolve_positive_no_match_errors_with_inventory() {
        let streams = vec![stream(0, MediaType::Audio)];
        let msg = resolve_stream_selection(&streams, &[parse("0:v")])
            .expect_err("video selector on audio-only input must fail")
            .to_string();
        assert!(
            msg.contains("matched no streams") && msg.contains("0:a"),
            "error must name the miss and list available streams, got: {msg}"
        );
    }

    #[test]
    fn resolve_kind_index_out_of_range_errors() {
        assert!(
            resolve_stream_selection(&vaas(), &[parse("0:a:2")]).is_err(),
            "only two audio streams exist; 0:a:2 must fail"
        );
    }

    #[test]
    fn resolve_everything_removed_errors() {
        let msg = resolve_stream_selection(&vaas(), &[parse("0:v"), parse("-0:v")])
            .expect_err("selecting then excluding everything must fail")
            .to_string();
        assert!(
            msg.contains("removed every stream"),
            "error must state the empty result, got: {msg}"
        );
    }

    #[test]
    fn resolve_duplicate_selectors_do_not_duplicate_streams() {
        let kept = resolve_stream_selection(&vaas(), &[parse("0:a"), parse("0:1")])
            .expect("overlapping selectors should resolve");
        assert_eq!(kept, vec![1, 2], "stream 1 must appear exactly once");
    }

    // ── remap table ────────────────────────────────────────────────────────

    #[test]
    fn build_index_remap_is_sequential() {
        let remap = build_index_remap(&[1, 3]);
        assert_eq!(remap.get(&1), Some(&0));
        assert_eq!(remap.get(&3), Some(&1));
        assert_eq!(remap.get(&0), None, "filtered streams must not remap");
        assert_eq!(remap.len(), 2);
    }
}
