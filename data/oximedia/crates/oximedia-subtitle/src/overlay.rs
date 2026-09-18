//! Video frame overlay for subtitle rendering.

use crate::style::Color;
use crate::text::{PositionedGlyph, TextLayout};
use crate::{SubtitleError, SubtitleResult};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

/// Overlay subtitle text layout onto a video frame.
///
/// # Errors
///
/// Returns error if the frame format is not supported.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn overlay_subtitle(
    frame: &mut VideoFrame,
    layout: &TextLayout,
    x: i32,
    y: i32,
    color: Color,
    outline_color: Option<Color>,
    outline_width: f32,
) -> SubtitleResult<()> {
    match frame.format {
        PixelFormat::Rgb24 | PixelFormat::Rgba32 => {
            overlay_rgb(frame, layout, x, y, color, outline_color, outline_width)
        }
        PixelFormat::Yuv420p => {
            overlay_yuv420p(frame, layout, x, y, color, outline_color, outline_width)
        }
        _ => Err(SubtitleError::InvalidFrameFormat(format!(
            "Unsupported pixel format for subtitle overlay: {:?}",
            frame.format
        ))),
    }
}

/// Overlay onto RGB/RGBA frame.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::too_many_arguments)]
fn overlay_rgb(
    frame: &mut VideoFrame,
    layout: &TextLayout,
    base_x: i32,
    base_y: i32,
    color: Color,
    outline_color: Option<Color>,
    outline_width: f32,
) -> SubtitleResult<()> {
    if frame.planes.is_empty() {
        return Err(SubtitleError::InvalidFrameFormat(
            "Frame has no planes".to_string(),
        ));
    }

    let bytes_per_pixel = if frame.format == PixelFormat::Rgb24 {
        3
    } else {
        4
    };

    let width = frame.width as usize;
    let height = frame.height as usize;
    let stride = frame.planes[0].stride;

    // Get mutable access to plane data
    let plane_data = &frame.planes[0].data;
    let mut output = plane_data.to_vec();

    // Render each line
    for line in &layout.lines {
        for glyph in &line.glyphs {
            let glyph_x = base_x + glyph.x as i32;
            let glyph_y = base_y + glyph.y as i32;

            // Render outline first if present
            if let Some(outline) = outline_color {
                render_glyph_outline(
                    &mut output,
                    width,
                    height,
                    stride,
                    bytes_per_pixel,
                    glyph,
                    glyph_x,
                    glyph_y,
                    outline,
                    outline_width,
                );
            }

            // Render main glyph
            render_glyph_rgba(
                &mut output,
                width,
                height,
                stride,
                bytes_per_pixel,
                glyph,
                glyph_x,
                glyph_y,
                color,
            );
        }
    }

    // Update frame data
    frame.planes[0].data = output;

    Ok(())
}

/// Render a single glyph onto RGBA buffer.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn render_glyph_rgba(
    buffer: &mut [u8],
    frame_width: usize,
    frame_height: usize,
    stride: usize,
    bytes_per_pixel: usize,
    glyph: &PositionedGlyph,
    x: i32,
    y: i32,
    color: Color,
) {
    for gy in 0..glyph.height {
        for gx in 0..glyph.width {
            let px = x + gx as i32;
            let py = y + gy as i32;

            // Bounds check
            if px < 0 || py < 0 || px >= frame_width as i32 || py >= frame_height as i32 {
                continue;
            }

            let glyph_idx = gy * glyph.width + gx;
            let alpha = glyph.bitmap[glyph_idx];

            if alpha == 0 {
                continue;
            }

            let pixel_idx = py as usize * stride + px as usize * bytes_per_pixel;

            if pixel_idx + bytes_per_pixel <= buffer.len() {
                // Blend glyph onto frame
                let alpha_f = f32::from(alpha) / 255.0 * f32::from(color.a) / 255.0;
                let inv_alpha = 1.0 - alpha_f;

                buffer[pixel_idx] =
                    (f32::from(color.r) * alpha_f + f32::from(buffer[pixel_idx]) * inv_alpha) as u8;
                buffer[pixel_idx + 1] = (f32::from(color.g) * alpha_f
                    + f32::from(buffer[pixel_idx + 1]) * inv_alpha)
                    as u8;
                buffer[pixel_idx + 2] = (f32::from(color.b) * alpha_f
                    + f32::from(buffer[pixel_idx + 2]) * inv_alpha)
                    as u8;

                // Keep alpha channel if RGBA
                if bytes_per_pixel == 4 {
                    buffer[pixel_idx + 3] = buffer[pixel_idx + 3].saturating_add(
                        ((255.0 - f32::from(buffer[pixel_idx + 3])) * alpha_f) as u8,
                    );
                }
            }
        }
    }
}

/// Render glyph outline.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn render_glyph_outline(
    buffer: &mut [u8],
    frame_width: usize,
    frame_height: usize,
    stride: usize,
    bytes_per_pixel: usize,
    glyph: &PositionedGlyph,
    x: i32,
    y: i32,
    color: Color,
    width: f32,
) {
    let outline_radius = width.ceil() as i32;

    for gy in 0..glyph.height {
        for gx in 0..glyph.width {
            let glyph_idx = gy * glyph.width + gx;
            let alpha = glyph.bitmap[glyph_idx];

            if alpha == 0 {
                continue;
            }

            // Draw outline pixels around glyph
            for dy in -outline_radius..=outline_radius {
                for dx in -outline_radius..=outline_radius {
                    // Check if within outline radius
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq as f32 > width * width {
                        continue;
                    }

                    let px = x + gx as i32 + dx;
                    let py = y + gy as i32 + dy;

                    if px < 0 || py < 0 || px >= frame_width as i32 || py >= frame_height as i32 {
                        continue;
                    }

                    let pixel_idx = py as usize * stride + px as usize * bytes_per_pixel;

                    if pixel_idx + bytes_per_pixel <= buffer.len() {
                        let alpha_f = f32::from(alpha) / 255.0 * f32::from(color.a) / 255.0 * 0.5;
                        let inv_alpha = 1.0 - alpha_f;

                        buffer[pixel_idx] = (f32::from(color.r) * alpha_f
                            + f32::from(buffer[pixel_idx]) * inv_alpha)
                            as u8;
                        buffer[pixel_idx + 1] = (f32::from(color.g) * alpha_f
                            + f32::from(buffer[pixel_idx + 1]) * inv_alpha)
                            as u8;
                        buffer[pixel_idx + 2] = (f32::from(color.b) * alpha_f
                            + f32::from(buffer[pixel_idx + 2]) * inv_alpha)
                            as u8;
                    }
                }
            }
        }
    }
}

/// Overlay onto YUV420p frame.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::too_many_arguments)]
fn overlay_yuv420p(
    frame: &mut VideoFrame,
    layout: &TextLayout,
    base_x: i32,
    base_y: i32,
    color: Color,
    outline_color: Option<Color>,
    outline_width: f32,
) -> SubtitleResult<()> {
    if frame.planes.len() != 3 {
        return Err(SubtitleError::InvalidFrameFormat(
            "YUV420p requires 3 planes".to_string(),
        ));
    }

    // Convert RGB color to YUV
    let (y_val, u_val, v_val) = rgb_to_yuv(color.r, color.g, color.b);
    let yuv_color = (y_val, u_val, v_val, color.a);

    let outline_yuv = outline_color.map(|c| {
        let (y, u, v) = rgb_to_yuv(c.r, c.g, c.b);
        (y, u, v, c.a)
    });

    let width = frame.width as usize;
    let height = frame.height as usize;

    // Get mutable copies of plane data
    let mut y_plane = frame.planes[0].data.to_vec();
    let mut u_plane = frame.planes[1].data.to_vec();
    let mut v_plane = frame.planes[2].data.to_vec();

    let y_stride = frame.planes[0].stride;
    let uv_stride = frame.planes[1].stride;
    let uv_width = width / 2;
    let uv_height = height / 2;

    // Render each line
    for line in &layout.lines {
        for glyph in &line.glyphs {
            let glyph_x = base_x + glyph.x as i32;
            let glyph_y = base_y + glyph.y as i32;

            // Render outline first if present
            if let Some(outline) = outline_yuv {
                render_glyph_yuv_outline(
                    &mut y_plane,
                    &mut u_plane,
                    &mut v_plane,
                    width,
                    height,
                    y_stride,
                    uv_stride,
                    uv_width,
                    uv_height,
                    glyph,
                    glyph_x,
                    glyph_y,
                    outline,
                    outline_width,
                );
            }

            // Render main glyph
            render_glyph_yuv(
                &mut y_plane,
                &mut u_plane,
                &mut v_plane,
                width,
                height,
                y_stride,
                uv_stride,
                uv_width,
                uv_height,
                glyph,
                glyph_x,
                glyph_y,
                yuv_color,
            );
        }
    }

    // Update frame data
    frame.planes[0].data = y_plane;
    frame.planes[1].data = u_plane;
    frame.planes[2].data = v_plane;

    Ok(())
}

/// Render glyph on YUV420p planes.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn render_glyph_yuv(
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
    uv_width: usize,
    uv_height: usize,
    glyph: &PositionedGlyph,
    x: i32,
    y: i32,
    color: (u8, u8, u8, u8),
) {
    let (y_col, u_col, v_col, alpha_col) = color;

    for gy in 0..glyph.height {
        for gx in 0..glyph.width {
            let px = x + gx as i32;
            let py = y + gy as i32;

            if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                continue;
            }

            let glyph_idx = gy * glyph.width + gx;
            let alpha = glyph.bitmap[glyph_idx];

            if alpha == 0 {
                continue;
            }

            // Blend onto Y plane
            let y_idx = py as usize * y_stride + px as usize;
            if y_idx < y_plane.len() {
                let alpha_f = f32::from(alpha) / 255.0 * f32::from(alpha_col) / 255.0;
                let inv_alpha = 1.0 - alpha_f;
                y_plane[y_idx] =
                    (f32::from(y_col) * alpha_f + f32::from(y_plane[y_idx]) * inv_alpha) as u8;
            }

            // Blend onto UV planes (subsampled)
            let uv_x = px as usize / 2;
            let uv_y = py as usize / 2;

            if uv_x < uv_width && uv_y < uv_height {
                let uv_idx = uv_y * uv_stride + uv_x;

                if uv_idx < u_plane.len() {
                    let alpha_f = f32::from(alpha) / 255.0 * f32::from(alpha_col) / 255.0 * 0.25;
                    let inv_alpha = 1.0 - alpha_f;

                    u_plane[uv_idx] =
                        (f32::from(u_col) * alpha_f + f32::from(u_plane[uv_idx]) * inv_alpha) as u8;
                    v_plane[uv_idx] =
                        (f32::from(v_col) * alpha_f + f32::from(v_plane[uv_idx]) * inv_alpha) as u8;
                }
            }
        }
    }
}

/// Render glyph outline on YUV420p planes.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn render_glyph_yuv_outline(
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
    uv_width: usize,
    uv_height: usize,
    glyph: &PositionedGlyph,
    x: i32,
    y: i32,
    color: (u8, u8, u8, u8),
    outline_width: f32,
) {
    let (y_col, u_col, v_col, alpha_col) = color;
    let outline_radius = outline_width.ceil() as i32;

    for gy in 0..glyph.height {
        for gx in 0..glyph.width {
            let glyph_idx = gy * glyph.width + gx;
            let alpha = glyph.bitmap[glyph_idx];

            if alpha == 0 {
                continue;
            }

            for dy in -outline_radius..=outline_radius {
                for dx in -outline_radius..=outline_radius {
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq as f32 > outline_width * outline_width {
                        continue;
                    }

                    let px = x + gx as i32 + dx;
                    let py = y + gy as i32 + dy;

                    if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                        continue;
                    }

                    let y_idx = py as usize * y_stride + px as usize;
                    if y_idx < y_plane.len() {
                        let alpha_f = f32::from(alpha) / 255.0 * f32::from(alpha_col) / 255.0 * 0.5;
                        let inv_alpha = 1.0 - alpha_f;
                        y_plane[y_idx] = (f32::from(y_col) * alpha_f
                            + f32::from(y_plane[y_idx]) * inv_alpha)
                            as u8;
                    }

                    let uv_x = px as usize / 2;
                    let uv_y = py as usize / 2;

                    if uv_x < uv_width && uv_y < uv_height {
                        let uv_idx = uv_y * uv_stride + uv_x;
                        if uv_idx < u_plane.len() {
                            let alpha_f =
                                f32::from(alpha) / 255.0 * f32::from(alpha_col) / 255.0 * 0.25;
                            let inv_alpha = 1.0 - alpha_f;

                            u_plane[uv_idx] = (f32::from(u_col) * alpha_f
                                + f32::from(u_plane[uv_idx]) * inv_alpha)
                                as u8;
                            v_plane[uv_idx] = (f32::from(v_col) * alpha_f
                                + f32::from(v_plane[uv_idx]) * inv_alpha)
                                as u8;
                        }
                    }
                }
            }
        }
    }
}

/// Convert RGB to YUV (BT.709).
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn rgb_to_yuv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = f32::from(r);
    let g = f32::from(g);
    let b = f32::from(b);

    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let u = (b - y) / 1.8556 + 128.0;
    let v = (r - y) / 1.5748 + 128.0;

    (
        y.clamp(0.0, 255.0) as u8,
        u.clamp(0.0, 255.0) as u8,
        v.clamp(0.0, 255.0) as u8,
    )
}
