//! VP8L pixel arithmetic helpers.
//!
//! Pure helper functions for ARGB pixel manipulation used by the VP8L
//! lossless decoder's spatial prediction transforms.

/// Add two ARGB pixels component-wise (mod 256 per channel).
#[inline]
pub(crate) fn add_pixels(a: u32, b: u32) -> u32 {
    let a0 = (a >> 24) & 0xFF;
    let a1 = (a >> 16) & 0xFF;
    let a2 = (a >> 8) & 0xFF;
    let a3 = a & 0xFF;

    let b0 = (b >> 24) & 0xFF;
    let b1 = (b >> 16) & 0xFF;
    let b2 = (b >> 8) & 0xFF;
    let b3 = b & 0xFF;

    (((a0 + b0) & 0xFF) << 24)
        | (((a1 + b1) & 0xFF) << 16)
        | (((a2 + b2) & 0xFF) << 8)
        | ((a3 + b3) & 0xFF)
}

/// Average two bytes (component-wise).
#[inline]
fn average2_byte(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b)) / 2) as u8
}

/// Average two ARGB pixels component-wise.
#[inline]
pub(crate) fn average2(a: u32, b: u32) -> u32 {
    let a_ch = pixel_channels(a);
    let b_ch = pixel_channels(b);
    channels_to_pixel([
        average2_byte(a_ch[0], b_ch[0]),
        average2_byte(a_ch[1], b_ch[1]),
        average2_byte(a_ch[2], b_ch[2]),
        average2_byte(a_ch[3], b_ch[3]),
    ])
}

/// Extract ARGB channels as [A, R, G, B].
#[inline]
pub(crate) fn pixel_channels(p: u32) -> [u8; 4] {
    [
        ((p >> 24) & 0xFF) as u8,
        ((p >> 16) & 0xFF) as u8,
        ((p >> 8) & 0xFF) as u8,
        (p & 0xFF) as u8,
    ]
}

/// Pack [A, R, G, B] channels into a pixel.
#[inline]
pub(crate) fn channels_to_pixel(ch: [u8; 4]) -> u32 {
    (u32::from(ch[0]) << 24) | (u32::from(ch[1]) << 16) | (u32::from(ch[2]) << 8) | u32::from(ch[3])
}

/// Clamp a value to [0, 255].
#[inline]
fn clamp_byte(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Select predictor: choose L or T based on Manhattan distance to TL.
#[inline]
pub(crate) fn select(left: u32, top: u32, top_left: u32) -> u32 {
    let l = pixel_channels(left);
    let t = pixel_channels(top);
    let tl = pixel_channels(top_left);

    let predict_l: i32 = (0..4)
        .map(|i| (i32::from(t[i]) - i32::from(tl[i])).abs())
        .sum();
    let predict_t: i32 = (0..4)
        .map(|i| (i32::from(l[i]) - i32::from(tl[i])).abs())
        .sum();

    if predict_l < predict_t {
        left
    } else {
        top
    }
}

/// ClampAddSubtractFull: L + T - TL, clamped per channel.
#[inline]
pub(crate) fn clamp_add_subtract_full(left: u32, top: u32, top_left: u32) -> u32 {
    let l = pixel_channels(left);
    let t = pixel_channels(top);
    let tl = pixel_channels(top_left);
    channels_to_pixel([
        clamp_byte(i32::from(l[0]) + i32::from(t[0]) - i32::from(tl[0])),
        clamp_byte(i32::from(l[1]) + i32::from(t[1]) - i32::from(tl[1])),
        clamp_byte(i32::from(l[2]) + i32::from(t[2]) - i32::from(tl[2])),
        clamp_byte(i32::from(l[3]) + i32::from(t[3]) - i32::from(tl[3])),
    ])
}

/// ClampAddSubtractHalf: avg + (avg - other) / 2, clamped.
#[inline]
pub(crate) fn clamp_add_subtract_half(avg: u32, other: u32) -> u32 {
    let a = pixel_channels(avg);
    let o = pixel_channels(other);
    channels_to_pixel([
        clamp_byte(i32::from(a[0]) + (i32::from(a[0]) - i32::from(o[0])) / 2),
        clamp_byte(i32::from(a[1]) + (i32::from(a[1]) - i32::from(o[1])) / 2),
        clamp_byte(i32::from(a[2]) + (i32::from(a[2]) - i32::from(o[2])) / 2),
        clamp_byte(i32::from(a[3]) + (i32::from(a[3]) - i32::from(o[3])) / 2),
    ])
}

/// Color transform delta.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn color_transform_delta(multiplier: i32, channel: i32) -> i32 {
    // The spec defines: (multiplier * channel) >> 5
    // but the multiplier is a signed byte interpreted as i8.
    ((multiplier as i8 as i32) * (channel as i8 as i32)) >> 5
}

/// Predict a pixel using one of the 14 predictor modes.
pub(crate) fn predict(mode: u8, left: u32, top: u32, top_left: u32, top_right: u32) -> u32 {
    match mode {
        0 => 0xFF00_0000, // black with opaque alpha
        1 => left,
        2 => top,
        3 => top_right,
        4 => top_left,
        5 => average2(average2(left, top_right), top),
        6 => average2(left, top_left),
        7 => average2(left, top),
        8 => average2(top_left, top),
        9 => average2(top, top_right),
        10 => average2(average2(left, top_left), average2(top, top_right)),
        11 => select(left, top, top_left),
        12 => clamp_add_subtract_full(left, top, top_left),
        13 => clamp_add_subtract_half(average2(left, top), top_left),
        _ => 0xFF00_0000, // fallback
    }
}

/// Integer division rounding up.
#[inline]
pub(crate) fn div_round_up(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
