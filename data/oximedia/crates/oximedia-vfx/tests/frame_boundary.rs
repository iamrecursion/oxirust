//! Boundary tests for `oximedia_vfx::Frame` at maximum (8K) and minimum (1x1)
//! resolutions, plus the zero / overflow dimension error paths.
//!
//! `Frame` stores RGBA8, 4 bytes/pixel, row-major; the pixel byte index is
//! `((y * width + x) * 4)`. `Frame::new` allocates `width * height * 4` zeroed
//! bytes, returning `Err(VfxError::InvalidDimensions { .. })` when either
//! dimension is 0 or when `width * height * 4` overflows `usize`.
//!
//! Memory note: the 8K frame is ~127 MiB; only one is alive at any time across
//! this file (each test allocates and drops its own).

use oximedia_vfx::{
    compositing::{BlendMode, Compositor},
    Frame, VfxError,
};

/// Full 8K (7680 x 4320) allocation, clear, and corner-pixel addressing.
#[test]
fn frame_8k_alloc_and_clear() {
    let mut frame = Frame::new(7680, 4320).expect("8K frame should allocate");

    // 7680 * 4320 * 4 = 132_710_400 bytes.
    assert_eq!(frame.data.len(), 132_710_400);
    assert_eq!(frame.byte_size(), 132_710_400);

    frame.clear([10, 20, 30, 255]);

    // First pixel.
    assert_eq!(frame.get_pixel(0, 0), Some([10, 20, 30, 255]));
    // Last in-bounds pixel (width-1, height-1).
    assert_eq!(frame.get_pixel(7679, 4319), Some([10, 20, 30, 255]));
    // One past the bottom-right corner -> out of bounds.
    assert_eq!(frame.get_pixel(7680, 4320), None);
}

/// 1x1 frame: smallest valid frame, set/get round-trip, OOB neighbours, clear.
#[test]
fn frame_1x1_ops() {
    let mut frame = Frame::new(1, 1).expect("1x1 frame should allocate");
    assert_eq!(frame.data.len(), 4);
    assert_eq!(frame.byte_size(), 4);

    frame.set_pixel(0, 0, [1, 2, 3, 4]);
    assert_eq!(frame.get_pixel(0, 0), Some([1, 2, 3, 4]));

    // Immediate neighbours are out of bounds for a 1x1 frame.
    assert_eq!(frame.get_pixel(1, 0), None);
    assert_eq!(frame.get_pixel(0, 1), None);

    // clear overwrites the single pixel.
    frame.clear([9, 8, 7, 6]);
    assert_eq!(frame.get_pixel(0, 0), Some([9, 8, 7, 6]));
}

/// A 1x1 `Compositor::composite` must not panic and, for `Normal` blend at full
/// opacity over an opaque backdrop, returns the (opaque) top pixel unchanged.
#[test]
fn frame_1x1_composite_no_panic() {
    let bottom = [255u8, 0, 0, 255]; // red
    let top = [0u8, 255, 0, 255]; // green, opaque
    let mut out = [0u8; 4];
    Compositor::composite(&bottom, &top, &mut out, 1, 1, BlendMode::Normal, 1.0);
    assert_eq!(out, [0, 255, 0, 255]);
}

/// Zero width is rejected.
#[test]
fn frame_zero_width_err() {
    assert!(matches!(
        Frame::new(0, 100),
        Err(VfxError::InvalidDimensions {
            width: 0,
            height: 100
        })
    ));
}

/// Zero height is rejected.
#[test]
fn frame_zero_height_err() {
    assert!(matches!(
        Frame::new(100, 0),
        Err(VfxError::InvalidDimensions {
            width: 100,
            height: 0
        })
    ));
}

/// Zero-by-zero is rejected.
#[test]
fn frame_zero_by_zero_err() {
    assert!(matches!(
        Frame::new(0, 0),
        Err(VfxError::InvalidDimensions {
            width: 0,
            height: 0
        })
    ));
}

/// `u32::MAX x u32::MAX` overflows `width * height * 4` and is rejected via the
/// checked-multiply path (NOT a panic, NOT an allocation).
#[test]
fn frame_overflow_dims_err() {
    assert!(matches!(
        Frame::new(u32::MAX, u32::MAX),
        Err(VfxError::InvalidDimensions { .. })
    ));
}
