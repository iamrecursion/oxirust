//! Alpha compositing operations.

use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

/// Alpha mode for compositing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlphaMode {
    /// Straight alpha (unassociated).
    Straight,
    /// Premultiplied alpha (associated).
    Premultiplied,
}

/// Porter-Duff compositing operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositeOp {
    /// A over B (normal compositing).
    Over,
    /// A under B.
    Under,
    /// A atop B.
    Atop,
    /// A xor B.
    Xor,
    /// Plus (additive).
    Plus,
    /// In (intersection).
    In,
    /// Out (exclusion).
    Out,
}

/// Composite source over backdrop (Porter-Duff over operator).
pub fn composite_over(
    backdrop: &Frame,
    source: &Frame,
    output: &mut Frame,
    alpha_mode: AlphaMode,
) -> VfxResult<()> {
    for y in 0..output.height {
        for x in 0..output.width {
            let b_pixel = backdrop.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            let s_pixel = source.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);

            let result = match alpha_mode {
                AlphaMode::Straight => composite_over_straight(b_pixel, s_pixel),
                AlphaMode::Premultiplied => composite_over_premult(b_pixel, s_pixel),
            };

            output.set_pixel(x, y, result);
        }
    }
    Ok(())
}

/// Composite source under backdrop (Porter-Duff under operator).
pub fn composite_under(
    backdrop: &Frame,
    source: &Frame,
    output: &mut Frame,
    alpha_mode: AlphaMode,
) -> VfxResult<()> {
    // Under is just swapped Over
    composite_over(source, backdrop, output, alpha_mode)
}

/// Composite source atop backdrop (Porter-Duff atop operator).
pub fn composite_atop(
    backdrop: &Frame,
    source: &Frame,
    output: &mut Frame,
    alpha_mode: AlphaMode,
) -> VfxResult<()> {
    for y in 0..output.height {
        for x in 0..output.width {
            let b_pixel = backdrop.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            let s_pixel = source.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);

            let result = match alpha_mode {
                AlphaMode::Straight => composite_atop_straight(b_pixel, s_pixel),
                AlphaMode::Premultiplied => composite_atop_premult(b_pixel, s_pixel),
            };

            output.set_pixel(x, y, result);
        }
    }
    Ok(())
}

/// Composite with XOR operation.
pub fn composite_xor(
    backdrop: &Frame,
    source: &Frame,
    output: &mut Frame,
    alpha_mode: AlphaMode,
) -> VfxResult<()> {
    for y in 0..output.height {
        for x in 0..output.width {
            let b_pixel = backdrop.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            let s_pixel = source.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);

            let result = match alpha_mode {
                AlphaMode::Straight => composite_xor_straight(b_pixel, s_pixel),
                AlphaMode::Premultiplied => composite_xor_premult(b_pixel, s_pixel),
            };

            output.set_pixel(x, y, result);
        }
    }
    Ok(())
}

/// Composite with Plus (additive) operation.
pub fn composite_plus(backdrop: &Frame, source: &Frame, output: &mut Frame) -> VfxResult<()> {
    for y in 0..output.height {
        for x in 0..output.width {
            let b_pixel = backdrop.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            let s_pixel = source.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);

            let result = [
                b_pixel[0].saturating_add(s_pixel[0]),
                b_pixel[1].saturating_add(s_pixel[1]),
                b_pixel[2].saturating_add(s_pixel[2]),
                b_pixel[3].saturating_add(s_pixel[3]),
            ];

            output.set_pixel(x, y, result);
        }
    }
    Ok(())
}

fn composite_over_straight(backdrop: [u8; 4], source: [u8; 4]) -> [u8; 4] {
    let b_a = f32::from(backdrop[3]) / 255.0;
    let s_a = f32::from(source[3]) / 255.0;

    let out_a = s_a + b_a * (1.0 - s_a);

    if out_a == 0.0 {
        return [0, 0, 0, 0];
    }

    let b_r = f32::from(backdrop[0]) / 255.0;
    let b_g = f32::from(backdrop[1]) / 255.0;
    let b_b = f32::from(backdrop[2]) / 255.0;

    let s_r = f32::from(source[0]) / 255.0;
    let s_g = f32::from(source[1]) / 255.0;
    let s_b = f32::from(source[2]) / 255.0;

    let out_r = (s_r * s_a + b_r * b_a * (1.0 - s_a)) / out_a;
    let out_g = (s_g * s_a + b_g * b_a * (1.0 - s_a)) / out_a;
    let out_b = (s_b * s_a + b_b * b_a * (1.0 - s_a)) / out_a;

    [
        (out_r * 255.0) as u8,
        (out_g * 255.0) as u8,
        (out_b * 255.0) as u8,
        (out_a * 255.0) as u8,
    ]
}

fn composite_over_premult(backdrop: [u8; 4], source: [u8; 4]) -> [u8; 4] {
    let b_a = f32::from(backdrop[3]) / 255.0;
    let s_a = f32::from(source[3]) / 255.0;

    let b_r = f32::from(backdrop[0]) / 255.0;
    let b_g = f32::from(backdrop[1]) / 255.0;
    let b_b = f32::from(backdrop[2]) / 255.0;

    let s_r = f32::from(source[0]) / 255.0;
    let s_g = f32::from(source[1]) / 255.0;
    let s_b = f32::from(source[2]) / 255.0;

    let out_r = s_r + b_r * (1.0 - s_a);
    let out_g = s_g + b_g * (1.0 - s_a);
    let out_b = s_b + b_b * (1.0 - s_a);
    let out_a = s_a + b_a * (1.0 - s_a);

    [
        (out_r * 255.0) as u8,
        (out_g * 255.0) as u8,
        (out_b * 255.0) as u8,
        (out_a * 255.0) as u8,
    ]
}

fn composite_atop_straight(backdrop: [u8; 4], source: [u8; 4]) -> [u8; 4] {
    let b_a = f32::from(backdrop[3]) / 255.0;
    let s_a = f32::from(source[3]) / 255.0;

    let out_a = b_a;

    if out_a == 0.0 {
        return [0, 0, 0, 0];
    }

    let b_r = f32::from(backdrop[0]) / 255.0;
    let b_g = f32::from(backdrop[1]) / 255.0;
    let b_b = f32::from(backdrop[2]) / 255.0;

    let s_r = f32::from(source[0]) / 255.0;
    let s_g = f32::from(source[1]) / 255.0;
    let s_b = f32::from(source[2]) / 255.0;

    let out_r = (s_r * s_a * b_a + b_r * b_a * (1.0 - s_a)) / out_a;
    let out_g = (s_g * s_a * b_a + b_g * b_a * (1.0 - s_a)) / out_a;
    let out_b = (s_b * s_a * b_a + b_b * b_a * (1.0 - s_a)) / out_a;

    [
        (out_r * 255.0) as u8,
        (out_g * 255.0) as u8,
        (out_b * 255.0) as u8,
        (out_a * 255.0) as u8,
    ]
}

fn composite_atop_premult(backdrop: [u8; 4], source: [u8; 4]) -> [u8; 4] {
    let b_a = f32::from(backdrop[3]) / 255.0;
    let s_a = f32::from(source[3]) / 255.0;

    let b_r = f32::from(backdrop[0]) / 255.0;
    let b_g = f32::from(backdrop[1]) / 255.0;
    let b_b = f32::from(backdrop[2]) / 255.0;

    let s_r = f32::from(source[0]) / 255.0;
    let s_g = f32::from(source[1]) / 255.0;
    let s_b = f32::from(source[2]) / 255.0;

    let out_r = s_r * b_a + b_r * (1.0 - s_a);
    let out_g = s_g * b_a + b_g * (1.0 - s_a);
    let out_b = s_b * b_a + b_b * (1.0 - s_a);
    let out_a = b_a;

    [
        (out_r * 255.0) as u8,
        (out_g * 255.0) as u8,
        (out_b * 255.0) as u8,
        (out_a * 255.0) as u8,
    ]
}

fn composite_xor_straight(backdrop: [u8; 4], source: [u8; 4]) -> [u8; 4] {
    let b_a = f32::from(backdrop[3]) / 255.0;
    let s_a = f32::from(source[3]) / 255.0;

    let out_a = s_a * (1.0 - b_a) + b_a * (1.0 - s_a);

    if out_a == 0.0 {
        return [0, 0, 0, 0];
    }

    let b_r = f32::from(backdrop[0]) / 255.0;
    let b_g = f32::from(backdrop[1]) / 255.0;
    let b_b = f32::from(backdrop[2]) / 255.0;

    let s_r = f32::from(source[0]) / 255.0;
    let s_g = f32::from(source[1]) / 255.0;
    let s_b = f32::from(source[2]) / 255.0;

    let out_r = (s_r * s_a * (1.0 - b_a) + b_r * b_a * (1.0 - s_a)) / out_a;
    let out_g = (s_g * s_a * (1.0 - b_a) + b_g * b_a * (1.0 - s_a)) / out_a;
    let out_b = (s_b * s_a * (1.0 - b_a) + b_b * b_a * (1.0 - s_a)) / out_a;

    [
        (out_r * 255.0) as u8,
        (out_g * 255.0) as u8,
        (out_b * 255.0) as u8,
        (out_a * 255.0) as u8,
    ]
}

fn composite_xor_premult(backdrop: [u8; 4], source: [u8; 4]) -> [u8; 4] {
    let b_a = f32::from(backdrop[3]) / 255.0;
    let s_a = f32::from(source[3]) / 255.0;

    let b_r = f32::from(backdrop[0]) / 255.0;
    let b_g = f32::from(backdrop[1]) / 255.0;
    let b_b = f32::from(backdrop[2]) / 255.0;

    let s_r = f32::from(source[0]) / 255.0;
    let s_g = f32::from(source[1]) / 255.0;
    let s_b = f32::from(source[2]) / 255.0;

    let out_r = s_r * (1.0 - b_a) + b_r * (1.0 - s_a);
    let out_g = s_g * (1.0 - b_a) + b_g * (1.0 - s_a);
    let out_b = s_b * (1.0 - b_a) + b_b * (1.0 - s_a);
    let out_a = s_a * (1.0 - b_a) + b_a * (1.0 - s_a);

    [
        (out_r * 255.0) as u8,
        (out_g * 255.0) as u8,
        (out_b * 255.0) as u8,
        (out_a * 255.0) as u8,
    ]
}

/// Convert straight alpha to premultiplied alpha.
pub fn straight_to_premult(pixel: [u8; 4]) -> [u8; 4] {
    let alpha = f32::from(pixel[3]) / 255.0;
    [
        (f32::from(pixel[0]) * alpha) as u8,
        (f32::from(pixel[1]) * alpha) as u8,
        (f32::from(pixel[2]) * alpha) as u8,
        pixel[3],
    ]
}

/// Convert premultiplied alpha to straight alpha.
pub fn premult_to_straight(pixel: [u8; 4]) -> [u8; 4] {
    if pixel[3] == 0 {
        return [0, 0, 0, 0];
    }
    let alpha = f32::from(pixel[3]) / 255.0;
    [
        (f32::from(pixel[0]) / alpha).min(255.0) as u8,
        (f32::from(pixel[1]) / alpha).min(255.0) as u8,
        (f32::from(pixel[2]) / alpha).min(255.0) as u8,
        pixel[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_over_opaque() {
        let backdrop = [100, 100, 100, 255];
        let source = [200, 200, 200, 255];
        let result = composite_over_straight(backdrop, source);
        assert_eq!(result, [200, 200, 200, 255]);
    }

    #[test]
    fn test_composite_over_transparent() {
        let backdrop = [100, 100, 100, 255];
        let source = [200, 200, 200, 0];
        let result = composite_over_straight(backdrop, source);
        assert_eq!(result, [100, 100, 100, 255]);
    }

    #[test]
    fn test_composite_over_semitransparent() {
        let backdrop = [0, 0, 0, 255];
        let source = [255, 255, 255, 128];
        let result = composite_over_straight(backdrop, source);
        assert!(result[0] > 100 && result[0] < 200);
    }

    #[test]
    fn test_alpha_conversion() {
        let straight = [128, 128, 128, 128];
        let premult = straight_to_premult(straight);
        assert!(premult[0] < straight[0]);
        let back = premult_to_straight(premult);
        assert!((back[0] as i16 - straight[0] as i16).abs() <= 1);
    }
}
