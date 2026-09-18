//! Matte operations for masking and hold-outs.

use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

/// Type of matte operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatteType {
    /// Use luminance as matte.
    Luma,
    /// Use alpha channel as matte.
    Alpha,
    /// Inverted luma matte.
    InvertedLuma,
    /// Inverted alpha matte.
    InvertedAlpha,
}

/// Apply matte to a frame using matte source.
pub fn apply_matte(
    input: &Frame,
    matte: &Frame,
    output: &mut Frame,
    matte_type: MatteType,
) -> VfxResult<()> {
    for y in 0..output.height {
        for x in 0..output.width {
            let input_pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            let matte_pixel = matte.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);

            let matte_value = match matte_type {
                MatteType::Luma => calculate_luma(matte_pixel),
                MatteType::Alpha => matte_pixel[3],
                MatteType::InvertedLuma => 255 - calculate_luma(matte_pixel),
                MatteType::InvertedAlpha => 255 - matte_pixel[3],
            };

            let output_pixel = [
                input_pixel[0],
                input_pixel[1],
                input_pixel[2],
                ((f32::from(input_pixel[3]) * f32::from(matte_value)) / 255.0) as u8,
            ];

            output.set_pixel(x, y, output_pixel);
        }
    }
    Ok(())
}

/// Apply hold-out matte (removes foreground where matte is white).
pub fn apply_holdout(
    input: &Frame,
    matte: &Frame,
    output: &mut Frame,
    matte_type: MatteType,
) -> VfxResult<()> {
    for y in 0..output.height {
        for x in 0..output.width {
            let input_pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            let matte_pixel = matte.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);

            let matte_value = match matte_type {
                MatteType::Luma => calculate_luma(matte_pixel),
                MatteType::Alpha => matte_pixel[3],
                MatteType::InvertedLuma => 255 - calculate_luma(matte_pixel),
                MatteType::InvertedAlpha => 255 - matte_pixel[3],
            };

            // Invert matte for hold-out
            let inv_matte = 255 - matte_value;

            let output_pixel = [
                input_pixel[0],
                input_pixel[1],
                input_pixel[2],
                ((f32::from(input_pixel[3]) * f32::from(inv_matte)) / 255.0) as u8,
            ];

            output.set_pixel(x, y, output_pixel);
        }
    }
    Ok(())
}

/// Choke (erode) matte edges.
pub fn choke_matte(input: &Frame, output: &mut Frame, amount: i32) -> VfxResult<()> {
    let amount = amount.max(0).min(10);

    for y in 0..output.height {
        for x in 0..output.width {
            let mut min_alpha = 255u8;

            // Find minimum alpha in neighborhood
            for dy in -amount..=amount {
                for dx in -amount..=amount {
                    let nx = (x as i32 + dx).max(0).min((input.width - 1) as i32) as u32;
                    let ny = (y as i32 + dy).max(0).min((input.height - 1) as i32) as u32;

                    if let Some(pixel) = input.get_pixel(nx, ny) {
                        min_alpha = min_alpha.min(pixel[3]);
                    }
                }
            }

            let input_pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            let output_pixel = [input_pixel[0], input_pixel[1], input_pixel[2], min_alpha];
            output.set_pixel(x, y, output_pixel);
        }
    }
    Ok(())
}

/// Spread (dilate) matte edges.
pub fn spread_matte(input: &Frame, output: &mut Frame, amount: i32) -> VfxResult<()> {
    let amount = amount.max(0).min(10);

    for y in 0..output.height {
        for x in 0..output.width {
            let mut max_alpha = 0u8;

            // Find maximum alpha in neighborhood
            for dy in -amount..=amount {
                for dx in -amount..=amount {
                    let nx = (x as i32 + dx).max(0).min((input.width - 1) as i32) as u32;
                    let ny = (y as i32 + dy).max(0).min((input.height - 1) as i32) as u32;

                    if let Some(pixel) = input.get_pixel(nx, ny) {
                        max_alpha = max_alpha.max(pixel[3]);
                    }
                }
            }

            let input_pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            let output_pixel = [input_pixel[0], input_pixel[1], input_pixel[2], max_alpha];
            output.set_pixel(x, y, output_pixel);
        }
    }
    Ok(())
}

/// Feather (blur) matte edges.
pub fn feather_matte(input: &Frame, output: &mut Frame, amount: i32) -> VfxResult<()> {
    let amount = amount.max(0).min(10);
    let _kernel_size = (amount * 2 + 1) as usize;

    for y in 0..output.height {
        for x in 0..output.width {
            let mut alpha_sum = 0u32;
            let mut count = 0u32;

            // Box blur
            for dy in -amount..=amount {
                for dx in -amount..=amount {
                    let nx = (x as i32 + dx).max(0).min((input.width - 1) as i32) as u32;
                    let ny = (y as i32 + dy).max(0).min((input.height - 1) as i32) as u32;

                    if let Some(pixel) = input.get_pixel(nx, ny) {
                        alpha_sum += u32::from(pixel[3]);
                        count += 1;
                    }
                }
            }

            let avg_alpha = (alpha_sum.checked_div(count).unwrap_or(0)) as u8;

            let input_pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
            let output_pixel = [input_pixel[0], input_pixel[1], input_pixel[2], avg_alpha];
            output.set_pixel(x, y, output_pixel);
        }
    }
    Ok(())
}

/// Calculate luminance from RGB pixel.
fn calculate_luma(pixel: [u8; 4]) -> u8 {
    // ITU-R BT.709 luma coefficients
    let r = f32::from(pixel[0]) * 0.2126;
    let g = f32::from(pixel[1]) * 0.7152;
    let b = f32::from(pixel[2]) * 0.0722;
    (r + g + b) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luma_calculation() {
        assert_eq!(calculate_luma([255, 255, 255, 255]), 255);
        assert_eq!(calculate_luma([0, 0, 0, 255]), 0);
        let gray = calculate_luma([128, 128, 128, 255]);
        assert!(gray > 100 && gray < 150);
    }

    #[test]
    fn test_apply_luma_matte() -> VfxResult<()> {
        let mut input = Frame::new(10, 10)?;
        let mut matte = Frame::new(10, 10)?;
        let mut output = Frame::new(10, 10)?;

        // Set input to white
        input.clear([255, 255, 255, 255]);
        // Set matte to 50% gray
        matte.clear([128, 128, 128, 255]);

        apply_matte(&input, &matte, &mut output, MatteType::Luma)?;

        let pixel = output.get_pixel(5, 5).expect("should succeed in test");
        assert!(pixel[3] < 255); // Alpha should be reduced
        Ok(())
    }

    #[test]
    fn test_choke_matte() -> VfxResult<()> {
        let mut input = Frame::new(10, 10)?;
        let mut output = Frame::new(10, 10)?;

        // Create a gradient
        for y in 0..10 {
            for x in 0..10 {
                let alpha = ((x + y) * 255 / 18) as u8;
                input.set_pixel(x, y, [255, 255, 255, alpha]);
            }
        }

        choke_matte(&input, &mut output, 1)?;

        // Choked matte should have lower alpha values
        let input_center = input.get_pixel(5, 5).expect("should succeed in test");
        let output_center = output.get_pixel(5, 5).expect("should succeed in test");
        assert!(output_center[3] <= input_center[3]);
        Ok(())
    }

    #[test]
    fn test_spread_matte() -> VfxResult<()> {
        let mut input = Frame::new(10, 10)?;
        let mut output = Frame::new(10, 10)?;

        // Create a gradient
        for y in 0..10 {
            for x in 0..10 {
                let alpha = ((x + y) * 255 / 18) as u8;
                input.set_pixel(x, y, [255, 255, 255, alpha]);
            }
        }

        spread_matte(&input, &mut output, 1)?;

        // Spread matte should have higher alpha values
        let input_center = input.get_pixel(5, 5).expect("should succeed in test");
        let output_center = output.get_pixel(5, 5).expect("should succeed in test");
        assert!(output_center[3] >= input_center[3]);
        Ok(())
    }
}
