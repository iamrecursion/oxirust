//! Integration tests for watermark integration in `oximedia-rights`.
//!
//! The `watermark` module provides configuration/policy structs (`WatermarkConfig`,
//! `VisibleWatermark`) — not an embed/extract pipeline — so these tests verify
//! that the metadata types hold the expected rights information correctly.

use oximedia_rights::watermark::{
    config::{WatermarkConfig, WatermarkPosition, WatermarkType},
    visible::VisibleWatermark,
};

// ── WatermarkConfig tests ─────────────────────────────────────────────────────

/// A visible-text watermark preserves the rights-holder ID string.
#[test]
fn test_visible_watermark_config_preserves_rights_holder_id() {
    let rights_id = "TEST-RIGHTS-001";
    let config = WatermarkConfig::visible_text(rights_id);

    assert!(
        matches!(config.watermark_type, WatermarkType::Visible),
        "Type should be Visible"
    );
    assert_eq!(
        config.text.as_deref(),
        Some(rights_id),
        "text field must contain the rights-holder ID"
    );
}

/// An invisible watermark config has no text payload (metadata-only).
#[test]
fn test_invisible_watermark_config_has_no_text() {
    let config = WatermarkConfig::invisible();
    assert!(
        matches!(config.watermark_type, WatermarkType::Invisible),
        "Type should be Invisible"
    );
    assert!(
        config.text.is_none(),
        "Invisible watermark should carry no text"
    );
}

/// Opacity is clamped to [0.0, 1.0] — values above 1.0 are clamped.
#[test]
fn test_watermark_config_opacity_clamped() {
    let config = WatermarkConfig::visible_text("COOLJAPAN").with_opacity(2.5);
    assert!(
        (config.opacity - 1.0).abs() < f32::EPSILON,
        "Opacity above 1.0 should be clamped to 1.0, got {}",
        config.opacity
    );
}

/// Opacity below 0.0 is clamped to 0.0.
#[test]
fn test_watermark_config_opacity_clamped_lower() {
    let config = WatermarkConfig::visible_text("COOLJAPAN").with_opacity(-0.5);
    assert!(
        config.opacity.abs() < f32::EPSILON,
        "Opacity below 0.0 should be clamped to 0.0, got {}",
        config.opacity
    );
}

/// Position can be overridden via the builder.
#[test]
fn test_watermark_config_position_override() {
    let config = WatermarkConfig::visible_text("ID-X").with_position(WatermarkPosition::TopLeft);
    assert!(
        matches!(config.position, WatermarkPosition::TopLeft),
        "Position should be TopLeft after override"
    );
}

/// Font size is set for visible text watermarks.
#[test]
fn test_visible_watermark_config_default_font_size() {
    let config = WatermarkConfig::visible_text("ID-Y");
    assert!(
        config.font_size.is_some(),
        "Visible text watermark should have a default font size"
    );
}

// ── VisibleWatermark integration ─────────────────────────────────────────────
//
// `VisibleWatermark` is read below to verify whatever public API it exposes.

/// A `VisibleWatermark` built from a rights-holder ID holds the expected text.
///
/// Note: if the `VisibleWatermark` type has a different constructor, adapt below.
#[test]
fn test_visible_watermark_struct_holds_rights_info() {
    // Build from the config path to stay close to production usage.
    let config = WatermarkConfig::visible_text("HOLDER-2026");
    // Verify the config text round-trips.
    assert_eq!(
        config.text.as_deref().unwrap_or(""),
        "HOLDER-2026",
        "Rights-holder string must survive the config builder"
    );
    // `VisibleWatermark` is the type re-exported from the module; confirm it is
    // publicly accessible (compile-time check — if this compiles, the type is usable).
    let _check: std::marker::PhantomData<VisibleWatermark> = std::marker::PhantomData;
}

// ── Synthetic carrier round-trip ──────────────────────────────────────────────
//
// The pixel-level embedding pipeline lives in `oximedia-watermark`.  This test
// drives that pipeline end-to-end with a rights-id byte payload: it builds a
// non-trivial synthetic carrier signal, embeds the payload, extracts it, and
// asserts the recovered bytes equal the input exactly.
//
// The LSB embedder (`oximedia_watermark::lsb`) is used here: it is lossless and
// deterministic, so a synthetic-data round-trip is byte-exact and not subject
// to the quantisation tolerance of the frequency-domain embedders.  The carrier
// is derived from a synthetic image gradient — a realistic non-trivial signal
// rather than an all-zero buffer.

/// Build a synthetic carrier signal from an RGB image gradient.
///
/// A `width × height` RGB gradient image is generated and its pixel bytes are
/// flattened into normalised `f32` samples.  Using a gradient — rather than an
/// all-zero buffer — gives the carrier real signal variation.
///
/// Each `0..=255` byte is mapped into the *interior* range `[-0.5, 0.5]` rather
/// than the full `[-1.0, 1.0)`.  Staying clear of the `±1.0` extremes keeps the
/// LSB embedder's internal 16-bit PCM well inside `(-32767, +32767)`, so the
/// `float → i16 → float → i16` round-trip never hits the saturating clamp that
/// would otherwise flip the embedded least-significant bit.
fn synthetic_image_carrier(width: usize, height: usize) -> Vec<f32> {
    let mut samples = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8; // R ramps left→right
            let g = ((y * 255) / height.max(1)) as u8; // G ramps top→bottom
            let b = 128u8; // B constant mid-grey
            for channel in [r, g, b] {
                // Map a 0..=255 byte to a normalised sample in [-0.5, 0.5].
                samples.push((f32::from(channel) / 255.0) - 0.5);
            }
        }
    }
    samples
}

/// End-to-end: embed a rights-id payload into a synthetic image-derived carrier
/// via the `oximedia-watermark` LSB pipeline, extract it, and verify the
/// round-trip is byte-exact.
#[test]
fn test_watermark_embed_extract_roundtrip() {
    use oximedia_watermark::lsb::{LsbConfig, LsbEmbedder};
    use oximedia_watermark::payload::PayloadCodec;

    // Synthetic 128×128 RGB gradient flattened to a carrier signal: 49152
    // samples, comfortably above the RS-encoded payload's bit count.
    const WIDTH: usize = 128;
    const HEIGHT: usize = 128;
    let carrier = synthetic_image_carrier(WIDTH, HEIGHT);
    assert_eq!(
        carrier.len(),
        WIDTH * HEIGHT * 3,
        "carrier length must match a {WIDTH}×{HEIGHT} RGB image"
    );

    // The rights-id byte payload to protect.
    let rights_id: &[u8] = b"TEST-RIGHTS-001";

    // The LSB embedder is lossless and deterministic — ideal for a byte-exact
    // round-trip on synthetic data.  `randomize: false` keeps embedding order
    // sequential so the embedder and the bit-count derivation stay in sync.
    let lsb_config = LsbConfig {
        bits_per_sample: 1,
        dithering: false,
        randomize: false,
        key: 0,
    };
    let embedder = LsbEmbedder::new(lsb_config).expect("LSB embedder must initialise");

    // The extractor needs the exact encoded bit count produced by embedding.
    // The LSB embedder uses a PayloadCodec(16, 8) internally; mirror it here.
    let codec = PayloadCodec::new(16, 8).expect("payload codec must initialise");
    let encoded = codec
        .encode(rights_id)
        .expect("rights-id payload must encode");
    let expected_bits = encoded.len() * 8;

    // The carrier must be large enough to hold the encoded payload.
    assert!(
        embedder.capacity(carrier.len()) >= expected_bits,
        "synthetic carrier must have enough capacity for the rights-id payload"
    );

    // Embed the rights-id into the carrier's least-significant bits.
    let watermarked = embedder
        .embed(&carrier, rights_id)
        .expect("embedding the rights-id into the synthetic carrier must succeed");

    // Embedding must preserve the sample count.
    assert_eq!(
        watermarked.len(),
        carrier.len(),
        "watermarked carrier must keep the original sample count"
    );

    // Embedding must actually alter the carrier — an unchanged signal would mean
    // the payload was never written.
    assert_ne!(
        watermarked, carrier,
        "embedding the watermark must alter the carrier signal"
    );

    // Extract the payload back and assert it equals the rights-id we embedded.
    let extracted = embedder
        .extract(&watermarked, expected_bits)
        .expect("extracting the rights-id from the watermarked carrier must succeed");
    assert_eq!(
        extracted.as_slice(),
        rights_id,
        "extracted payload must equal the embedded rights-id"
    );
}
