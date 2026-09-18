//! `SubRip` (SRT) format parser and writer

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::{Caption, CaptionTrack, Language, Timestamp};
use nom::{
    bytes::complete::tag,
    character::complete::{char, digit1, space0},
    combinator::map_res,
    sequence::separated_pair,
    IResult, Parser,
};

/// SRT format parser
pub struct SrtParser;

impl FormatParser for SrtParser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        let text = std::str::from_utf8(data).map_err(|e| CaptionError::Encoding(e.to_string()))?;

        parse_srt(text)
    }
}

/// SRT format writer
pub struct SrtWriter;

impl FormatWriter for SrtWriter {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        let mut output = String::new();

        for (index, caption) in track.captions.iter().enumerate() {
            // Caption number
            output.push_str(&format!("{}\n", index + 1));

            // Timestamp
            let start = format_timestamp(caption.start);
            let end = format_timestamp(caption.end);
            output.push_str(&format!("{start} --> {end}\n"));

            // Text
            output.push_str(&caption.text);
            output.push_str("\n\n");
        }

        Ok(output.into_bytes())
    }
}

fn parse_srt(text: &str) -> Result<CaptionTrack> {
    let mut track = CaptionTrack::new(Language::english());
    let blocks: Vec<&str> = text
        .split("\n\n")
        .filter(|s| !s.trim().is_empty())
        .collect();

    for block in blocks {
        if let Ok(caption) = parse_srt_block(block) {
            track.add_caption(caption)?;
        }
    }

    Ok(track)
}

fn parse_srt_block(block: &str) -> Result<Caption> {
    let lines: Vec<&str> = block.lines().collect();
    if lines.len() < 3 {
        return Err(CaptionError::Parse("Invalid SRT block".to_string()));
    }

    // Parse timestamp line
    let (start, end) = parse_timestamp_line(lines[1])?;

    // Combine remaining lines as text
    let text = lines[2..].join("\n");

    Ok(Caption::new(start, end, text))
}

fn parse_timestamp_line(line: &str) -> Result<(Timestamp, Timestamp)> {
    match srt_timestamp_line(line) {
        Ok((_, (start, end))) => Ok((start, end)),
        Err(e) => Err(CaptionError::Parse(format!("Invalid timestamp: {e}"))),
    }
}

fn srt_timestamp_line(input: &str) -> IResult<&str, (Timestamp, Timestamp)> {
    separated_pair(srt_timestamp, (space0, tag("-->"), space0), srt_timestamp).parse(input)
}

fn srt_timestamp(input: &str) -> IResult<&str, Timestamp> {
    let (input, hours) = map_res(digit1, |s: &str| s.parse::<u32>()).parse(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, minutes) = map_res(digit1, |s: &str| s.parse::<u32>()).parse(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, seconds) = map_res(digit1, |s: &str| s.parse::<u32>()).parse(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, millis) = map_res(digit1, |s: &str| s.parse::<u32>()).parse(input)?;

    Ok((input, Timestamp::from_hmsm(hours, minutes, seconds, millis)))
}

fn format_timestamp(ts: Timestamp) -> String {
    let (h, m, s, ms) = ts.as_hmsm();
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

// ============================================================================
// Zero-copy nom SRT parser (Wave 14 Slice H)
// ============================================================================

/// A parsed SRT cue that borrows from the input string — zero allocation per
/// entry.  The `text` field is a direct slice of the original input.
#[derive(Debug, Clone, PartialEq)]
pub struct SrtCueRef<'a> {
    /// Sequence index (1-based as found in the file; 0 if not present)
    pub index: u32,
    /// Start time in milliseconds
    pub start_ms: u64,
    /// End time in milliseconds
    pub end_ms: u64,
    /// Cue body — a slice of the original `&str` input, no allocation
    pub text: &'a str,
}

/// Parse an SRT timestamp (`HH:MM:SS,mmm`) into milliseconds.
fn nom_srt_timestamp(input: &str) -> IResult<&str, u64> {
    let (input, hours) = map_res(digit1, |s: &str| s.parse::<u64>()).parse(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, minutes) = map_res(digit1, |s: &str| s.parse::<u64>()).parse(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, seconds) = map_res(digit1, |s: &str| s.parse::<u64>()).parse(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, millis) = map_res(digit1, |s: &str| s.parse::<u64>()).parse(input)?;
    Ok((
        input,
        hours * 3_600_000 + minutes * 60_000 + seconds * 1_000 + millis,
    ))
}

/// Parse `HH:MM:SS,mmm --> HH:MM:SS,mmm` into `(start_ms, end_ms)`.
fn nom_srt_timestamp_line(input: &str) -> IResult<&str, (u64, u64)> {
    separated_pair(
        nom_srt_timestamp,
        (space0, tag("-->"), space0),
        nom_srt_timestamp,
    )
    .parse(input)
}

/// Parse a single SRT block, returning a zero-copy [`SrtCueRef`].
///
/// Accepts the raw block text (text between two blank lines).  `original_input`
/// is the full file string and is used only to compute the exact subslice for
/// the cue body — no copying is performed.
fn parse_srt_cue_ref(block: &str) -> Option<SrtCueRef<'_>> {
    let mut lines = block.splitn(4, '\n');

    // Line 0: sequence index (optional; we skip it on parse failure)
    let index_line = lines.next()?.trim();
    let index: u32 = index_line.parse().unwrap_or(0);

    // Line 1: timestamp
    let ts_line = lines.next()?.trim();
    let (start_ms, end_ms) = match nom_srt_timestamp_line(ts_line) {
        Ok((_, pair)) => pair,
        Err(_) => return None,
    };

    // Lines 2+: cue body (the `splitn(4, '\n')` remainder is a single &str)
    let text_start = lines.next()?;
    // Collect any further lines that were not consumed by splitn (the 4th capture
    // already includes everything after the 3rd newline as a single slice).
    let rest = lines.next().unwrap_or("");
    // Reconstruct the body slice from the block's interior; we want the portion
    // starting at text_start up to the end of the block.
    let text = if rest.is_empty() {
        text_start.trim_end_matches('\r')
    } else {
        // The block contains more than 3 lines; take from text_start to end of block.
        // Since splitn gives us overlapping borrows we need pointer arithmetic.
        let block_start = block.as_ptr() as usize;
        let text_ptr = text_start.as_ptr() as usize;
        let offset = text_ptr - block_start;
        block[offset..]
            .trim_end_matches('\r')
            .trim_end_matches('\n')
            .trim_end_matches('\r')
    };

    Some(SrtCueRef {
        index,
        start_ms,
        end_ms,
        text,
    })
}

/// Zero-copy SRT parse returning `Vec<SrtCueRef<'a>>` that borrow from
/// `input`.  Avoids all intermediate `String` allocations for callers that
/// only need to inspect the parsed data without retaining owned values.
///
/// For callers that need owned [`Caption`] values use [`fast_parse_srt`]
/// instead.
pub fn parse_srt_nom(input: &str) -> Result<Vec<SrtCueRef<'_>>> {
    let mut cues = Vec::new();
    for block in input.split("\n\n") {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(cue) = parse_srt_cue_ref(trimmed) {
            cues.push(cue);
        }
    }
    Ok(cues)
}

/// Fast SRT parser that uses nom timestamp parsing internally but still
/// produces an owned [`CaptionTrack`].  The intermediate
/// `.lines().collect::<Vec<&str>>()` allocation present in the original
/// `parse_srt` is replaced by a zero-alloc nom pass for the timestamp line,
/// and block splitting avoids an extra owned-string copy.
pub fn fast_parse_srt(text: &str) -> Result<CaptionTrack> {
    let mut track = CaptionTrack::new(Language::english());
    for block in text.split("\n\n") {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(cue) = parse_srt_cue_ref(trimmed) {
            let start = Timestamp::from_millis(cue.start_ms as i64);
            let end = Timestamp::from_millis(cue.end_ms as i64);
            let caption = Caption::new(start, end, cue.text.to_owned());
            track.add_caption(caption)?;
        }
    }
    Ok(track)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt() {
        let srt = b"1\n00:00:01,000 --> 00:00:03,000\nFirst caption\n\n2\n00:00:05,000 --> 00:00:07,500\nSecond caption\n\n";
        let parser = SrtParser;
        let track = parser.parse(srt).expect("parsing should succeed");

        assert_eq!(track.captions.len(), 2);
        assert_eq!(track.captions[0].text, "First caption");
        assert_eq!(track.captions[1].text, "Second caption");
    }

    #[test]
    fn test_write_srt() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Test caption".to_string(),
            ))
            .expect("operation should succeed in test");

        let writer = SrtWriter;
        let output = writer.write(&track).expect("writing should succeed");
        let text = String::from_utf8(output).expect("output should be valid UTF-8");

        assert!(text.contains("1\n"));
        assert!(text.contains("Test caption"));
        assert!(text.contains("-->"));
    }

    #[test]
    fn test_timestamp_parsing() {
        let (_, ts) = srt_timestamp("00:01:30,500").expect("timestamp parsing should succeed");
        assert_eq!(ts.as_hmsm(), (0, 1, 30, 500));
    }

    /// Round-trip test: write an SRT track then parse it back and verify the
    /// key fields are preserved.
    #[test]
    fn test_srt_roundtrip() {
        let mut original = CaptionTrack::new(Language::english());
        original
            .add_caption(Caption::new(
                Timestamp::from_hmsm(0, 0, 1, 0),
                Timestamp::from_hmsm(0, 0, 3, 500),
                "Hello, world!".to_string(),
            ))
            .expect("add_caption should succeed");
        original
            .add_caption(Caption::new(
                Timestamp::from_hmsm(0, 0, 5, 250),
                Timestamp::from_hmsm(0, 0, 8, 0),
                "Second caption line".to_string(),
            ))
            .expect("add_caption should succeed");

        // Serialize to SRT bytes
        let bytes = SrtWriter.write(&original).expect("write should succeed");

        // Deserialize back to a track
        let parsed = SrtParser.parse(&bytes).expect("parse should succeed");

        assert_eq!(
            parsed.captions.len(),
            original.captions.len(),
            "caption count must match after round-trip"
        );

        for (orig, rt) in original.captions.iter().zip(parsed.captions.iter()) {
            assert_eq!(orig.start, rt.start, "start timestamp mismatch");
            assert_eq!(orig.end, rt.end, "end timestamp mismatch");
            assert_eq!(orig.text, rt.text, "text content mismatch");
        }
    }

    // Wave 14 Slice H — new tests

    /// Build an SRT string with `n` cues starting at second `0` and spaced 5 s apart.
    fn make_srt(n: usize) -> String {
        let mut out = String::with_capacity(n * 60);
        for i in 0..n {
            let start_s = i * 5;
            let end_s = start_s + 3;
            let sh = start_s / 3600;
            let sm = (start_s % 3600) / 60;
            let ss = start_s % 60;
            let eh = end_s / 3600;
            let em = (end_s % 3600) / 60;
            let es = end_s % 60;
            out.push_str(&format!(
                "{}\n{:02}:{:02}:{:02},000 --> {:02}:{:02}:{:02},000\nCaption text {}\n\n",
                i + 1,
                sh,
                sm,
                ss,
                eh,
                em,
                es,
                i + 1
            ));
        }
        out
    }

    /// `fast_parse_srt` and the original `parse_srt` must agree on every cue.
    #[test]
    fn test_nom_srt_matches_original() {
        let input = make_srt(50);
        let original_track =
            parse_srt(&input).expect("original parse_srt should succeed on 50-cue input");
        let fast_track =
            fast_parse_srt(&input).expect("fast_parse_srt should succeed on 50-cue input");

        assert_eq!(
            original_track.captions.len(),
            fast_track.captions.len(),
            "cue count must match"
        );
        for (i, (orig, fast)) in original_track
            .captions
            .iter()
            .zip(fast_track.captions.iter())
            .enumerate()
        {
            assert_eq!(orig.start, fast.start, "start_ms mismatch at cue {i}");
            assert_eq!(orig.end, fast.end, "end_ms mismatch at cue {i}");
            assert_eq!(orig.text, fast.text, "text mismatch at cue {i}");
        }
    }

    /// The zero-copy `parse_srt_nom` must also agree with the original parser.
    #[test]
    fn test_parse_srt_nom_matches_original() {
        let input = make_srt(20);
        let original_track = parse_srt(&input).expect("original parse should succeed");
        let cues = parse_srt_nom(&input).expect("nom parse should succeed");

        assert_eq!(
            original_track.captions.len(),
            cues.len(),
            "cue count must match between original and nom parsers"
        );
        for (i, (orig, cue)) in original_track.captions.iter().zip(cues.iter()).enumerate() {
            let orig_start_ms = orig.start.as_millis() as u64;
            let orig_end_ms = orig.end.as_millis() as u64;
            assert_eq!(orig_start_ms, cue.start_ms, "start_ms mismatch at cue {i}");
            assert_eq!(orig_end_ms, cue.end_ms, "end_ms mismatch at cue {i}");
            assert_eq!(orig.text, cue.text.trim_end(), "text mismatch at cue {i}");
        }
    }

    /// Parse a 10 000-cue SRT string via the nom zero-copy parser and assert
    /// 10 000 cues are returned without allocation error.
    #[test]
    fn test_srt_large_file_parse() {
        let input = make_srt(10_000);
        let cues = parse_srt_nom(&input).expect("nom parse of 10 000-cue SRT should succeed");
        assert_eq!(cues.len(), 10_000, "expected 10 000 cues from large SRT");
    }
}
