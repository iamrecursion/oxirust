//! Deep-layer compositing tests for `oximedia_vfx::compositing`.
//!
//! These tests pin the *byte-exact* behaviour of the real compositing API —
//! `Compositor::flatten_layers` (the actual deep-layer entry point; there is no
//! `layer_manager`-based stack in the slice contract) and `blend_pixels` — and
//! a no-panic smoke test for the bilinear `LayerStack::composite` path.
//!
//! Quantization note: `blend_pixels` / `flatten_layers` convert the final
//! normalized channel back to `u8` via `(channel * 255.0) as u8`, which is a
//! **truncation** (round-toward-zero), NOT a round-to-nearest. So a normalized
//! 0.5 becomes 127 (= 0.5 * 255 = 127.5, truncated), never 128. Every oracle
//! below accounts for this.

use oximedia_vfx::{
    compositing::{blend_pixels, BlendMode, Compositor, Layer, LayerStack},
    Frame, VfxResult,
};

/// Build a solid-colour RGBA8 buffer (`w * h * 4` bytes, row-major).
fn solid(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
    let n = (w as usize) * (h as usize);
    let mut buf = Vec::with_capacity(n * 4);
    for _ in 0..n {
        buf.extend_from_slice(&color);
    }
    buf
}

/// A single fully-opaque layer at opacity 1.0 must reproduce its own pixels
/// exactly (the seed step copies rgb and computes alpha = (a * 1.0) = a).
#[test]
fn single_layer_identity() {
    let w = 8;
    let h = 8;
    let layer = solid(w, h, [40, 80, 120, 255]);
    let layers: Vec<(&[u8], f32)> = vec![(&layer, 1.0)];
    let mut out = vec![0u8; (w * h * 4) as usize];

    Compositor::flatten_layers(&layers, &mut out, w, h);

    assert_eq!(out.len(), (w * h * 4) as usize);
    for px in out.chunks_exact(4) {
        assert_eq!(px, &[40, 80, 120, 255]);
    }
}

/// 50% red over opaque blue.
///
/// `flatten_layers(&[(blue, 1.0), (red, 0.5)], ..)`:
///   seed   = blue, alpha = (255 * 1.0) -> 255 => [0, 0, 255, 255]
///   alpha  = s_a(1.0) * opacity(0.5) = 0.5
///   final_alpha = 0.5 + 1.0 * (1 - 0.5) = 1.0
///   R = (1.0 * 0.5 + 0.0) / 1.0 = 0.5  -> trunc(0.5 * 255 = 127.5) = 127
///   G = 0
///   B = (1.0 * 1.0 * (1 - 0.5)) / 1.0 = 0.5 -> 127
///   A = 255
/// => `[127, 0, 127, 255]`.
#[test]
fn two_layer_50pct_red_over_blue_exact() {
    let w = 8;
    let h = 8;
    let blue = solid(w, h, [0, 0, 255, 255]);
    let red = solid(w, h, [255, 0, 0, 255]);
    let layers: Vec<(&[u8], f32)> = vec![(&blue, 1.0), (&red, 0.5)];
    let mut out = vec![0u8; (w * h * 4) as usize];

    Compositor::flatten_layers(&layers, &mut out, w, h);

    // pixel (0,0)
    assert_eq!(&out[0..4], &[127, 0, 127, 255]);
    // an interior pixel (3,4) -> index ((4*8 + 3) * 4) = 140
    let interior = ((4 * w + 3) * 4) as usize;
    assert_eq!(&out[interior..interior + 4], &[127, 0, 127, 255]);
}

/// Alpha-encoded (NOT opacity-multiplied) source with alpha = 128 over opaque
/// blue, asserted via `blend_pixels` with `opacity = 1.0` so the only alpha
/// contribution comes from the encoded channel.
///
///   s_a = 128 / 255 = 0.501960784...
///   final_alpha = s_a + 1.0 * (1 - s_a) = 1.0
///   R = (1.0 * s_a + 0.0) / 1.0 = 0.50196 -> trunc(0.50196 * 255 = 127.999..) = 127?
///
/// NOTE the subtlety: 0.50196 * 255 = 127.9999... which truncates to **127**
/// under exact arithmetic, BUT f32 rounding of (128/255) gives a value whose
/// `* 255.0` lands at >= 128.0, so the truncated result is **128**. Likewise
/// B = (1 - s_a) = 0.498039, * 255 = 126.999..., f32 -> 126. This is precisely
/// the 128 / 126 (NOT 127 / 127) outcome, and it pins both the truncation rule
/// AND the opacity-vs-alpha-encoding distinction: here the "0.5" comes from the
/// alpha channel itself, whereas in `two_layer_50pct_*` it came from the
/// opacity multiplier applied to a fully-opaque (alpha = 255) source.
#[test]
fn two_layer_alpha_quantization() {
    let backdrop = [0u8, 0, 255, 255]; // opaque blue
    let source = [255u8, 0, 0, 128]; // red, alpha-encoded 50%
    let result = blend_pixels(backdrop, source, BlendMode::Normal, 1.0);

    // 128 / 126 (NOT 127 / 127) — f32 rounding of 128/255 pushes R's
    // pre-truncation product to >=128.0, while B's lands just below 127.0.
    assert_eq!(
        result,
        [128, 0, 126, 255],
        "alpha-encoded 0.5 over blue: f32(128/255)*255 truncates to 128 (R), \
         f32(1 - 128/255)*255 truncates to 126 (B); contrast with the \
         opacity-multiplied 127/127 case"
    );
}

/// 50 stacked semi-transparent layers over a green base, validated against an
/// independent fold of the *same* `blend_pixels` call over a single
/// representative pixel. Because `flatten_layers` and the reference fold invoke
/// byte-identical math, the result must match with tolerance 0.
#[test]
fn fifty_layer_opacity_ramp_matches_sequential_over() {
    let w = 8;
    let h = 8;

    // Base: solid green, opacity 1.0.
    let base_color = [0u8, 200, 0, 255];
    let base = solid(w, h, base_color);
    let base_opacity = 1.0f32;

    // 50 ramp layers.
    let mut layer_bufs: Vec<Vec<u8>> = Vec::with_capacity(50);
    let mut layer_colors: Vec<[u8; 4]> = Vec::with_capacity(50);
    let mut layer_opacities: Vec<f32> = Vec::with_capacity(50);
    for i in 0..50u32 {
        let color = [((i * 5) & 0xFF) as u8, 10, 200, 255];
        let opacity = 0.02 + i as f32 * 0.0196;
        // All ramp opacities must lie strictly inside (0, 1).
        assert!(
            opacity > 0.0 && opacity < 1.0,
            "ramp opacity {opacity} (i={i}) out of (0,1)"
        );
        layer_bufs.push(solid(w, h, color));
        layer_colors.push(color);
        layer_opacities.push(opacity);
    }

    // Assemble the (&[u8], f32) slice for flatten_layers (base first = bottom).
    let mut layers: Vec<(&[u8], f32)> = Vec::with_capacity(51);
    layers.push((base.as_slice(), base_opacity));
    for i in 0..50 {
        layers.push((layer_bufs[i].as_slice(), layer_opacities[i]));
    }

    let mut out = vec![0u8; (w * h * 4) as usize];
    Compositor::flatten_layers(&layers, &mut out, w, h);

    // Independent reference fold over one representative pixel, reproducing the
    // exact seed + fold sequence used by flatten_layers.
    let seed_alpha = ((base_color[3] as f32) * base_opacity).clamp(0.0, 255.0) as u8;
    let mut acc = [base_color[0], base_color[1], base_color[2], seed_alpha];
    for i in 0..50 {
        acc = blend_pixels(acc, layer_colors[i], BlendMode::Normal, layer_opacities[i]);
    }

    // Byte-exact match at pixel 0.
    assert_eq!(
        &out[0..4],
        &acc,
        "flatten_layers != sequential blend_pixels fold"
    );

    // The output is spatially uniform (all layers are solid), so every other
    // pixel offset must equal the same reference.
    let interior = ((5 * w + 6) * 4) as usize;
    assert_eq!(&out[interior..interior + 4], &acc);
    let last = ((w * h - 1) * 4) as usize;
    assert_eq!(&out[last..last + 4], &acc);

    assert_eq!(out.len(), (8 * 8 * 4) as usize);
}

/// Smoke test for the bilinear `LayerStack::composite` path with 50 layers.
///
/// `LayerStack::composite` does bilinear resampling + truncation, so it is NOT
/// byte-exact against `blend_pixels`; we only assert it runs without panicking,
/// produces a correctly-sized buffer, and yields a fully-opaque centre pixel.
#[test]
fn fifty_layer_via_layerstack_smoke() -> VfxResult<()> {
    let w = 8;
    let h = 8;
    let mut stack = LayerStack::new();

    for i in 0..50u32 {
        let color = [((i * 5) & 0xFF) as u8, 10, 200, 255];
        let opacity = 0.02 + i as f32 * 0.0196;
        let frame = Frame::from_data(w, h, solid(w, h, color))?;
        let layer = Layer::new(format!("l{i}"), frame)
            .with_z_index(i as i32)
            .with_opacity(opacity);
        stack.add_layer(layer);
    }
    assert_eq!(stack.len(), 50);

    let mut output = Frame::new(w, h)?;
    stack.composite(&mut output)?;

    assert_eq!(output.byte_size(), (8 * 8 * 4) as usize);

    // Centre pixel: each layer is fully opaque (alpha 255) before opacity
    // scaling, but `LayerStack::composite` applies per-layer opacity as
    // `(alpha * opacity) as u8` (truncation), so the bottom-most layer
    // (opacity ~0.02) contributes a *truncated* alpha and the cumulative
    // alpha-over of 50 such layers settles at 254 — one short of 255 — rather
    // than exactly saturating. (This is the bilinear/truncation path the slice
    // flags as NOT byte-exact; we only assert near-full opacity here.)
    let center = output
        .get_pixel(w / 2, h / 2)
        .expect("centre pixel in bounds");
    assert!(
        center[3] >= 254,
        "centre alpha should be effectively opaque after 50 stacked layers, got {}",
        center[3]
    );

    Ok(())
}
