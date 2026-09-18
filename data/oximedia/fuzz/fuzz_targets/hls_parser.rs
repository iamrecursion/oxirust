//! HLS (HTTP Live Streaming) playlist parser fuzzer.
//!
//! This fuzzer tests the HLS playlist parser for:
//! - M3U8 header parsing (#EXTM3U)
//! - Master playlist parsing (multi-bitrate)
//! - Media playlist parsing (segments)
//! - Tag parsing:
//!   - EXT-X-VERSION
//!   - EXT-X-STREAM-INF (bandwidth, resolution, codecs)
//!   - EXT-X-TARGETDURATION
//!   - EXTINF (segment duration, title)
//!   - EXT-X-BYTERANGE
//!   - EXT-X-DISCONTINUITY
//!   - EXT-X-KEY (encryption)
//!   - EXT-X-MEDIA (alternative renditions)
//!   - EXT-X-ENDLIST
//!   - EXT-X-PLAYLIST-TYPE
//! - Variant stream parsing
//! - Segment URL parsing
//! - Attribute parsing
//! - Line continuation handling
//! - Malformed tag handling
//! - Missing required fields
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_net::hls::{MasterPlaylist, MediaPlaylist};

fuzz_target!(|data: &[u8]| {
    // Convert data to UTF-8 string
    // Invalid UTF-8 sequences will be replaced with the replacement character
    let text = String::from_utf8_lossy(data);

    // Test master playlist parsing
    // This should never panic even on completely invalid input
    let _ = MasterPlaylist::parse(&text);

    // Test media playlist parsing
    // This should never panic even on completely invalid input
    let _ = MediaPlaylist::parse(&text);

    // Test parsing with various modifications to explore edge cases
    if !text.is_empty() {
        // Test with added EXTM3U header if missing
        let with_header = if !text.starts_with("#EXTM3U") {
            format!("#EXTM3U\n{}", text)
        } else {
            text.to_string()
        };

        let _ = MasterPlaylist::parse(&with_header);
        let _ = MediaPlaylist::parse(&with_header);

        // Test with duplicated content (stress test)
        if text.len() < 1000 {
            let duplicated = text.repeat(3);
            let _ = MasterPlaylist::parse(&duplicated);
            let _ = MediaPlaylist::parse(&duplicated);
        }

        // Test with truncated input
        if text.len() > 10 {
            let truncated = &text[..text.len() / 2];
            let _ = MasterPlaylist::parse(truncated);
            let _ = MediaPlaylist::parse(truncated);
        }

        // Test with line endings variations
        let with_crlf = text.replace('\n', "\r\n");
        let _ = MasterPlaylist::parse(&with_crlf);
        let _ = MediaPlaylist::parse(&with_crlf);

        // Test with extra whitespace
        let with_spaces = text.replace('\n', "\n  ");
        let _ = MasterPlaylist::parse(&with_spaces);
        let _ = MediaPlaylist::parse(&with_spaces);
    }

    // Test specific malformed patterns that might cause issues
    let test_patterns = [
        "#EXTM3U\n",
        "#EXTM3U\n#EXT-X-VERSION:",
        "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=",
        "#EXTM3U\n#EXTINF:",
        "#EXTM3U\n#EXT-X-BYTERANGE:",
        "#EXTM3U\n#EXT-X-KEY:METHOD=",
        "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1000000\nhttp://example.com/stream.m3u8",
        "#EXTM3U\n#EXTINF:10.0,\nsegment.ts",
    ];

    for pattern in &test_patterns {
        let combined = format!("{}{}", pattern, text);
        let _ = MasterPlaylist::parse(&combined);
        let _ = MediaPlaylist::parse(&combined);
    }
});
