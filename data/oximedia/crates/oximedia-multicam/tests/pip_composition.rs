//! Integration tests for picture-in-picture composition geometry.
//!
//! Wave 29 / Slice 8 — PURE test-hardening of `composite::pip`.
//!
//! Verifies PIP output dimensions and placement for all corner positions,
//! padding overrides, custom positions, the containment invariant, and a
//! second (1280×720, scale 0.5) geometry. Oracles are derived directly from
//! the source math in `src/composite/pip.rs`.

#![allow(clippy::float_cmp)]

use oximedia_multicam::composite::pip::{PictureInPicture, PipPosition};
use oximedia_multicam::composite::Compositor;

/// Default construction: 1080p, scale 0.25, padding 20, BottomRight.
#[test]
fn test_pip_defaults_1080p() {
    let pip = PictureInPicture::new(1920, 1080);
    assert_eq!(pip.dimensions(), (1920, 1080));
    assert_eq!(pip.scale(), 0.25);
    assert_eq!(pip.padding(), 20);
    assert_eq!(pip.position(), PipPosition::BottomRight);
    // 1920*0.25 = 480, 1080*0.25 = 270
    assert_eq!(pip.calculate_pip_dimensions(), (480, 270));
}

/// TopLeft corner sits at exactly (padding, padding).
#[test]
fn test_pip_top_left_1080p() {
    let mut pip = PictureInPicture::new(1920, 1080);
    pip.set_position(PipPosition::TopLeft);

    assert_eq!(pip.calculate_pip_dimensions(), (480, 270));
    assert_eq!(pip.calculate_pip_position(), (20, 20));
    assert_eq!(pip.pip_region(), (20, 20, 480, 270));
    // Main region is always the full frame.
    assert_eq!(pip.main_region(), (0, 0, 1920, 1080));
}

/// TopRight corner: x = width - pip_w - padding = 1920 - 480 - 20 = 1420.
#[test]
fn test_pip_top_right_1080p() {
    let mut pip = PictureInPicture::new(1920, 1080);
    pip.set_position(PipPosition::TopRight);

    assert_eq!(pip.calculate_pip_dimensions(), (480, 270));
    assert_eq!(pip.calculate_pip_position(), (1420, 20));
    assert_eq!(pip.pip_region(), (1420, 20, 480, 270));
}

/// BottomLeft corner: y = height - pip_h - padding = 1080 - 270 - 20 = 790.
#[test]
fn test_pip_bottom_left_1080p() {
    let mut pip = PictureInPicture::new(1920, 1080);
    pip.set_position(PipPosition::BottomLeft);

    assert_eq!(pip.calculate_pip_dimensions(), (480, 270));
    assert_eq!(pip.calculate_pip_position(), (20, 790));
    assert_eq!(pip.pip_region(), (20, 790, 480, 270));
}

/// BottomRight corner: (1420, 790).
#[test]
fn test_pip_bottom_right_1080p() {
    let mut pip = PictureInPicture::new(1920, 1080);
    pip.set_position(PipPosition::BottomRight);

    assert_eq!(pip.calculate_pip_dimensions(), (480, 270));
    assert_eq!(pip.calculate_pip_position(), (1420, 790));
    assert_eq!(pip.pip_region(), (1420, 790, 480, 270));
}

/// Custom positions are passed through verbatim, ignoring padding.
#[test]
fn test_pip_custom_position() {
    let mut pip = PictureInPicture::new(1920, 1080);
    pip.set_position(PipPosition::Custom(100, 50));

    assert_eq!(pip.calculate_pip_position(), (100, 50));
    assert_eq!(pip.pip_region(), (100, 50, 480, 270));
}

/// Overriding padding shifts the corner inset accordingly.
/// padding 16 + BottomRight → x = 1920-480-16 = 1424, y = 1080-270-16 = 794.
#[test]
fn test_pip_padding_override_bottom_right() {
    let mut pip = PictureInPicture::new(1920, 1080);
    pip.set_padding(16);
    pip.set_position(PipPosition::BottomRight);

    assert_eq!(pip.padding(), 16);
    assert_eq!(pip.calculate_pip_dimensions(), (480, 270));
    assert_eq!(pip.calculate_pip_position(), (1424, 794));
    assert_eq!(pip.pip_region(), (1424, 794, 480, 270));
}

/// Containment invariant for every corner: the PIP rect must fit inside the frame.
#[test]
fn test_pip_containment_all_corners() {
    let corners = [
        PipPosition::TopLeft,
        PipPosition::TopRight,
        PipPosition::BottomLeft,
        PipPosition::BottomRight,
    ];
    let (frame_w, frame_h) = (1920u32, 1080u32);

    for corner in corners {
        let mut pip = PictureInPicture::new(frame_w, frame_h);
        pip.set_position(corner);
        let (x, y, w, h) = pip.pip_region();
        assert_eq!((w, h), (480, 270), "dims for {corner:?}");
        assert!(
            x + w <= frame_w,
            "x+w out of bounds for {corner:?}: {x}+{w} > {frame_w}"
        );
        assert!(
            y + h <= frame_h,
            "y+h out of bounds for {corner:?}: {y}+{h} > {frame_h}"
        );
    }
}

/// `set_scale` is clamped to the inclusive range 0.1..=0.5.
#[test]
fn test_pip_scale_clamped() {
    let mut pip = PictureInPicture::new(1920, 1080);

    pip.set_scale(0.9); // above max → 0.5
    assert_eq!(pip.scale(), 0.5);

    pip.set_scale(0.01); // below min → 0.1
    assert_eq!(pip.scale(), 0.1);

    pip.set_scale(0.5); // boundary is inclusive
    assert_eq!(pip.scale(), 0.5);

    pip.set_scale(0.1); // boundary is inclusive
    assert_eq!(pip.scale(), 0.1);

    pip.set_scale(0.25); // in-range passes through
    assert_eq!(pip.scale(), 0.25);
}

/// Second geometry: 1280×720, scale clamped to 0.5, zero padding, BottomRight.
/// pip dims = (640, 360); BottomRight pos = (1280-640-0, 720-360-0) = (640, 360).
#[test]
fn test_pip_720p_half_scale_zero_padding() {
    let mut pip = PictureInPicture::new(1280, 720);
    pip.set_scale(0.5);
    pip.set_padding(0);
    pip.set_position(PipPosition::BottomRight);

    assert_eq!(pip.dimensions(), (1280, 720));
    assert_eq!(pip.scale(), 0.5);
    assert_eq!(pip.padding(), 0);
    assert_eq!(pip.calculate_pip_dimensions(), (640, 360));
    assert_eq!(pip.calculate_pip_position(), (640, 360));
    assert_eq!(pip.pip_region(), (640, 360, 640, 360));

    // Containment: at scale 0.5 the inset abuts the frame edges exactly.
    let (x, y, w, h) = pip.pip_region();
    assert!(x + w <= 1280);
    assert!(y + h <= 720);
    assert_eq!(x + w, 1280);
    assert_eq!(y + h, 720);
}

/// 720p half-scale TopLeft with zero padding hugs the origin.
#[test]
fn test_pip_720p_top_left_zero_padding() {
    let mut pip = PictureInPicture::new(1280, 720);
    pip.set_scale(0.5);
    pip.set_padding(0);
    pip.set_position(PipPosition::TopLeft);

    assert_eq!(pip.calculate_pip_position(), (0, 0));
    assert_eq!(pip.pip_region(), (0, 0, 640, 360));
}
