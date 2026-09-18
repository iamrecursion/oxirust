//! Smoke tests for orphan modules wired in 0.1.8 Wave 6 Slice F.
//!
//! Verifies that newly registered modules compile and expose the expected
//! public API surface.  Tests are intentionally lightweight — full
//! functionality is covered by in-module unit tests.

// ============================================================================
// glyph_atlas
// ============================================================================

#[test]
fn glyph_atlas_allocate_and_insert() {
    use oximedia_subtitle::glyph_atlas::{AtlasConfig, GlyphAtlas, GlyphKey};

    let config = AtlasConfig {
        width: 256,
        height: 256,
        padding: 1,
    };
    let mut atlas = GlyphAtlas::new(config);

    // Allocate space for an 8x8 glyph
    let slot = atlas.allocate(8, 8).expect("should allocate successfully");
    assert_eq!(slot.width, 8);
    assert_eq!(slot.height, 8);

    // Insert the glyph
    let key = GlyphKey::new('A', 16.0, 0);
    let bitmap = vec![255u8; 8 * 8]; // white 8x8 alpha glyph
    atlas
        .insert(key, slot, &bitmap)
        .expect("insert should succeed");
    assert_eq!(atlas.cached_count(), 1);
    assert_eq!(atlas.allocation_count(), 1);
}

#[test]
fn glyph_atlas_default_size_is_1024() {
    use oximedia_subtitle::glyph_atlas::GlyphAtlas;

    let atlas = GlyphAtlas::default_size();
    assert_eq!(atlas.config.width, 1024);
    assert_eq!(atlas.config.height, 1024);
    // RGBA: 4 bytes per pixel
    assert_eq!(atlas.pixels().len(), 1024 * 1024 * 4);
}

#[test]
fn glyph_atlas_zero_remaining_after_fill() {
    use oximedia_subtitle::glyph_atlas::{AtlasConfig, GlyphAtlas};

    // Small atlas: 16x16, padding=0
    let config = AtlasConfig {
        width: 16,
        height: 16,
        padding: 0,
    };
    let mut atlas = GlyphAtlas::new(config);
    assert!(atlas.remaining_height() > 0);
    // Filling the atlas eventually consumes remaining height
    let _ = atlas.allocate(16, 16).expect("first allocation fits");
    assert_eq!(atlas.remaining_height(), 0);
}

// ============================================================================
// karaoke_engine
// ============================================================================

#[test]
fn karaoke_engine_syllable_state_at_timestamp() {
    use oximedia_subtitle::karaoke_engine::{KaraokeSyllable, KaraokeTrack, SyllableState};

    let mut track = KaraokeTrack::new();
    track.add_syllable(KaraokeSyllable::new("Ka-", 0, 300));
    track.add_syllable(KaraokeSyllable::new("ra-", 300, 300));
    track.add_syllable(KaraokeSyllable::new("oke", 600, 400));

    // At t=0: first syllable active, rest pending
    let states = track.syllable_states(0);
    assert!(
        matches!(states[0], SyllableState::Active { .. }),
        "first syllable should be active at t=0"
    );
    assert_eq!(states[1], SyllableState::Pending);
    assert_eq!(states[2], SyllableState::Pending);

    // At t=350: second syllable is active, first is completed
    let states = track.syllable_states(350);
    assert_eq!(states[0], SyllableState::Completed);
    assert!(matches!(states[1], SyllableState::Active { .. }));
}

#[test]
fn karaoke_engine_full_text() {
    use oximedia_subtitle::karaoke_engine::{KaraokeSyllable, KaraokeTrack};

    let track = KaraokeTrack::from_syllables(vec![
        KaraokeSyllable::new("Hel-", 0, 200),
        KaraokeSyllable::new("lo", 200, 200),
    ]);
    assert_eq!(track.full_text(), "Hel-lo");
    assert_eq!(track.total_duration_ms(), 400);
}

// ============================================================================
// wcag_validator
// ============================================================================

#[test]
fn wcag_validator_white_on_black_passes_aaa() {
    use oximedia_subtitle::wcag_validator::{WcagLevel, WcagValidator};

    let validator = WcagValidator::new();
    let result = validator.check_contrast((255, 255, 255), (0, 0, 0), WcagLevel::Aaa);
    assert!(result.passes, "white on black must pass WCAG AAA");
    assert!(result.ratio > 20.9, "expected contrast ~21:1");
}

#[test]
fn wcag_validator_low_contrast_fails_aa() {
    use oximedia_subtitle::wcag_validator::{WcagLevel, WcagValidator};

    let validator = WcagValidator::new();
    // Very similar gray tones should fail AA
    let result = validator.check_contrast((128, 128, 128), (160, 160, 160), WcagLevel::Aa);
    assert!(!result.passes, "similar grays should fail AA");
}

// ============================================================================
// teletext
// ============================================================================

#[test]
fn teletext_parser_creation() {
    use oximedia_subtitle::teletext::TeletextParser;

    let parser = TeletextParser::default_subtitle_parser();
    assert!(parser.target_pages.contains(&0x888));
    assert!(!parser.has_completed_pages());
}

#[test]
fn teletext_parser_drains_empty_cleanly() {
    use oximedia_subtitle::teletext::TeletextParser;

    let mut parser = TeletextParser::new(vec![888]);
    let pages = parser.drain_completed();
    assert!(pages.is_empty());
}

// ============================================================================
// subtitle_ocr
// ============================================================================

#[test]
fn subtitle_ocr_blank_image_returns_empty() {
    use oximedia_subtitle::subtitle_ocr::{BitmapImage, BitmapOcr, GlyphAtlas};

    let atlas = GlyphAtlas::new();
    let ocr = BitmapOcr::new(atlas);

    // A completely black (blank) image should produce empty text
    let image = BitmapImage::new(32, 16);
    let text = ocr.extract_text(&image, 0.5);
    assert!(
        text.trim().is_empty(),
        "blank image should yield empty text"
    );
}

#[test]
fn subtitle_ocr_binarise_threshold() {
    use oximedia_subtitle::subtitle_ocr::BitmapImage;

    let mut image = BitmapImage::new(4, 4);
    image.set_pixel(1, 1, 200); // bright pixel above threshold 128
    image.set_pixel(2, 2, 50); // dark pixel below threshold

    let binary = image.binarise(128);
    assert_eq!(binary.pixel(1, 1), 255, "bright pixel should become 255");
    assert_eq!(binary.pixel(2, 2), 0, "dark pixel should become 0");
}
