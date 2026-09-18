//! DASH (Dynamic Adaptive Streaming over HTTP) MPD parser fuzzer.
//!
//! This fuzzer tests the DASH MPD (Media Presentation Description) parser for:
//! - XML parsing of MPD manifests
//! - MPD root element attributes:
//!   - type (static/dynamic)
//!   - mediaPresentationDuration
//!   - minBufferTime
//!   - profiles
//! - Period parsing (multiple periods, dynamic insertion)
//! - AdaptationSet parsing:
//!   - mimeType, codecs
//!   - contentType (audio, video, text)
//!   - segmentAlignment
//! - Representation parsing:
//!   - bandwidth, width, height
//!   - frameRate, sampleRate
//! - Segment template parsing:
//!   - initialization
//!   - media (with $Number$, $Time$, $Bandwidth$ variables)
//!   - startNumber, timescale
//! - SegmentTimeline parsing (S elements with d, r, t attributes)
//! - SegmentList and SegmentBase parsing
//! - BaseURL resolution
//! - ContentProtection (DRM) parsing
//! - Role and Accessibility descriptors
//! - ISO 8601 duration parsing
//! - Malformed XML handling
//! - Missing required attributes
//! - Invalid attribute values
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_net::dash::Mpd;

fuzz_target!(|data: &[u8]| {
    // Convert data to UTF-8 string
    // Invalid UTF-8 sequences will be replaced with the replacement character
    let text = String::from_utf8_lossy(data);

    // Test MPD parsing
    // This should never panic even on completely invalid XML
    let _ = Mpd::parse(&text);

    // Test parsing with various modifications to explore edge cases
    if !text.is_empty() {
        // Test with added XML declaration if missing
        let with_declaration = if !text.starts_with("<?xml") {
            format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n{}", text)
        } else {
            text.to_string()
        };

        let _ = Mpd::parse(&with_declaration);

        // Test with added MPD root element if missing
        let with_mpd = if !text.contains("<MPD") {
            format!(
                "<?xml version=\"1.0\"?>\n<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\">{}</MPD>",
                text
            )
        } else {
            text.to_string()
        };

        let _ = Mpd::parse(&with_mpd);

        // Test with duplicated content (stress test)
        if text.len() < 1000 {
            let duplicated = text.repeat(2);
            let _ = Mpd::parse(&duplicated);
        }

        // Test with truncated input
        if text.len() > 20 {
            let truncated = &text[..text.len() / 2];
            let _ = Mpd::parse(truncated);
        }

        // Test with extra whitespace
        let with_spaces = text.replace('>', ">  \n  ");
        let _ = Mpd::parse(&with_spaces);

        // Test with attribute injection
        let with_attrs = text.replace("<MPD", "<MPD type=\"static\" profiles=\"urn:mpeg:dash:profile:isoff-main:2011\"");
        let _ = Mpd::parse(&with_attrs);
    }

    // Test specific XML patterns that might cause issues
    let test_patterns = [
        "<?xml version=\"1.0\"?><MPD></MPD>",
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\"></MPD>",
        "<MPD><Period></Period></MPD>",
        "<MPD><Period><AdaptationSet></AdaptationSet></Period></MPD>",
        "<MPD><Period><AdaptationSet><Representation></Representation></AdaptationSet></Period></MPD>",
        "<MPD type=\"static\"><Period><AdaptationSet><Representation bandwidth=\"\"></Representation></AdaptationSet></Period></MPD>",
        "<MPD><Period><AdaptationSet><Representation><SegmentTemplate></SegmentTemplate></Representation></AdaptationSet></Period></MPD>",
        "<MPD><Period><AdaptationSet><Representation><SegmentTemplate><SegmentTimeline></SegmentTimeline></SegmentTemplate></Representation></AdaptationSet></Period></MPD>",
    ];

    for pattern in &test_patterns {
        let combined = format!("{}{}", pattern, text);
        let _ = Mpd::parse(&combined);
    }

    // Test ISO 8601 duration parsing via the MPD attribute path
    // (the internal duration parser runs on these attribute values)
    let duration_patterns = [
        "PT1H30M",
        "PT30S",
        "P1DT12H",
        "PT0.5S",
        "P1Y2M3DT4H5M6S",
        "PT",
        "P",
        "",
    ];

    for pattern in &duration_patterns {
        let combined = format!(
            "<MPD mediaPresentationDuration=\"{}{}\" minBufferTime=\"{}\"></MPD>",
            pattern,
            text.chars().take(64).collect::<String>().replace('"', ""),
            pattern
        );
        let _ = Mpd::parse(&combined);
    }
});
