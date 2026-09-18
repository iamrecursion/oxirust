// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: Vorbis comment edge cases.
//!
//! Verifies the `oximedia_container::metadata::vorbis` parser/encoder against:
//!
//! - Japanese ("こんにちは"), Arabic ("مرحبا"), and emoji ("🎵") tag values.
//! - Empty tag values ("KEY=").
//! - Repeated keys with distinct values (artist=A, artist=B).
//! - Vendor strings containing non-ASCII bytes.
//!
//! For multi-tag instances the HashMap-backed `TagMap` does NOT preserve
//! iteration order, so byte-for-byte equality is enforced only on hand-built
//! single-tag payloads. Multi-tag round-trips assert the tag-value set is
//! preserved instead.
//!
//! TODO.md item: line 64 — "Test metadata/vorbis.rs Vorbis comment handling
//! with edge cases (empty tags, unicode)".

use oximedia_container::metadata::vorbis::{VorbisComments, VorbisCommentsBuilder};

/// Builds a Vorbis comment block by hand. Lets us assert byte-equality after
/// parse → re-encode for the single-tag case.
fn build_raw_block(vendor: &str, comments: &[(&str, &str)]) -> Vec<u8> {
    let mut out = Vec::new();
    let vendor_bytes = vendor.as_bytes();
    out.extend_from_slice(&(vendor_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(vendor_bytes);
    out.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for (k, v) in comments {
        let entry = format!("{k}={v}");
        let bytes = entry.as_bytes();
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

#[test]
fn test_vorbis_unicode_japanese_byte_preservation() {
    let title = "こんにちは"; // 15 UTF-8 bytes
    let raw = build_raw_block("OxiMedia/0.1", &[("TITLE", title)]);

    let parsed = VorbisComments::parse(&raw).expect("japanese parse");
    assert_eq!(parsed.vendor, "OxiMedia/0.1");
    assert_eq!(parsed.tags.get_text("TITLE"), Some(title));

    // Single-tag block: encoding order is deterministic → byte equality.
    let re_encoded = parsed.encode();
    assert_eq!(re_encoded, raw, "byte-level round-trip must be lossless");
}

#[test]
fn test_vorbis_unicode_arabic_byte_preservation() {
    let artist = "مرحبا"; // 10 UTF-8 bytes, RTL script
    let raw = build_raw_block("OxiMedia", &[("ARTIST", artist)]);

    let parsed = VorbisComments::parse(&raw).expect("arabic parse");
    assert_eq!(parsed.tags.get_text("ARTIST"), Some(artist));
    let re_encoded = parsed.encode();
    assert_eq!(re_encoded, raw, "arabic round-trip must be lossless");
}

#[test]
fn test_vorbis_unicode_emoji_byte_preservation() {
    let comment = "🎵 music 🎶"; // 4-byte emoji codepoints
    let raw = build_raw_block("OxiMedia", &[("COMMENT", comment)]);

    let parsed = VorbisComments::parse(&raw).expect("emoji parse");
    assert_eq!(parsed.tags.get_text("COMMENT"), Some(comment));
    let re_encoded = parsed.encode();
    assert_eq!(re_encoded, raw, "emoji round-trip must be lossless");
}

#[test]
fn test_vorbis_empty_value_and_unicode_vendor() {
    let vendor = "OxiMedia/日本語 0.1.7"; // non-ASCII vendor
    let raw = build_raw_block(vendor, &[("TITLE", "")]);

    let parsed = VorbisComments::parse(&raw).expect("empty value parse");
    assert_eq!(parsed.vendor, vendor);
    assert_eq!(
        parsed.tags.get_text("TITLE"),
        Some(""),
        "value must be empty string, not missing"
    );
    assert_eq!(parsed.len(), 1, "empty value is still a present tag");

    let re_encoded = parsed.encode();
    assert_eq!(re_encoded, raw, "empty-value round-trip must be lossless");
}

#[test]
fn test_vorbis_repeated_keys_and_mixed_scripts() {
    let raw = build_raw_block(
        "OxiMedia",
        &[
            ("ARTIST", "Alice"),
            ("ARTIST", "Bob"),
            ("TITLE", "テスト"),
            ("COMMENT", "🎵"),
            ("ARTIST", "Charlie"),
        ],
    );

    let parsed = VorbisComments::parse(&raw).expect("repeated keys parse");
    assert_eq!(parsed.vendor, "OxiMedia");

    let artists = parsed.tags.get_all("ARTIST");
    assert_eq!(artists.len(), 3, "three artist entries");
    let mut artist_strings: Vec<&str> = artists.iter().filter_map(|v| v.as_text()).collect();
    artist_strings.sort_unstable();
    assert_eq!(artist_strings, vec!["Alice", "Bob", "Charlie"]);

    assert_eq!(parsed.tags.get_text("TITLE"), Some("テスト"));
    assert_eq!(parsed.tags.get_text("COMMENT"), Some("🎵"));

    // For multi-tag instances HashMap iteration order is non-deterministic, so
    // assert value-set preservation by feeding the encoded bytes back through
    // the parser and comparing tag values.
    let re_encoded = parsed.encode();
    let reparsed = VorbisComments::parse(&re_encoded).expect("reparse re-encoded");
    let mut re_artists: Vec<&str> = reparsed
        .tags
        .get_all("ARTIST")
        .iter()
        .filter_map(|v| v.as_text())
        .collect();
    re_artists.sort_unstable();
    assert_eq!(re_artists, vec!["Alice", "Bob", "Charlie"]);
    assert_eq!(reparsed.tags.get_text("TITLE"), Some("テスト"));
    assert_eq!(reparsed.tags.get_text("COMMENT"), Some("🎵"));
}

#[test]
fn test_vorbis_builder_unicode() {
    // Exercise the builder path with non-ASCII content and verify the encoded
    // payload parses back into the same single value (single-tag → byte equal).
    let built = VorbisCommentsBuilder::new()
        .vendor("OxiMedia/0.1.7")
        .tag("TITLE", "🎵 こんにちは مرحبا")
        .build();

    let encoded = built.encode();
    let reparsed = VorbisComments::parse(&encoded).expect("builder parse");
    assert_eq!(reparsed.vendor, "OxiMedia/0.1.7");
    assert_eq!(reparsed.tags.get_text("TITLE"), Some("🎵 こんにちは مرحبا"));

    // Re-encoding once more must match the first encoding (single-tag is order-stable).
    let re_encoded = reparsed.encode();
    assert_eq!(re_encoded, encoded, "builder single-tag round-trip");
}
