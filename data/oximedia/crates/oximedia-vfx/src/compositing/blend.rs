//! Blending modes for layer compositing.

use serde::{Deserialize, Serialize};

/// Blending modes for layer compositing (30+ professional modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum BlendMode {
    // Normal modes
    Normal,
    Dissolve,

    // Darken modes
    Darken,
    Multiply,
    ColorBurn,
    LinearBurn,
    DarkerColor,

    // Lighten modes
    Lighten,
    Screen,
    ColorDodge,
    LinearDodge,
    LighterColor,

    // Contrast modes
    Overlay,
    SoftLight,
    HardLight,
    VividLight,
    LinearLight,
    PinLight,
    HardMix,

    // Inversion modes
    Difference,
    Exclusion,
    Subtract,
    Divide,

    // Component modes
    Hue,
    Saturation,
    Color,
    Luminosity,

    // Math modes
    Add,
    Average,
    Negation,
    Phoenix,

    // Advanced modes
    Reflect,
    Glow,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// Blend two pixels using the specified blend mode.
#[allow(clippy::too_many_lines)]
pub fn blend_pixels(backdrop: [u8; 4], source: [u8; 4], mode: BlendMode, opacity: f32) -> [u8; 4] {
    let opacity = opacity.clamp(0.0, 1.0);

    // Convert to normalized floats
    let b = [
        f32::from(backdrop[0]) / 255.0,
        f32::from(backdrop[1]) / 255.0,
        f32::from(backdrop[2]) / 255.0,
        f32::from(backdrop[3]) / 255.0,
    ];

    let s = [
        f32::from(source[0]) / 255.0,
        f32::from(source[1]) / 255.0,
        f32::from(source[2]) / 255.0,
        f32::from(source[3]) / 255.0,
    ];

    let alpha = s[3] * opacity;

    // Apply blend mode
    let result = match mode {
        BlendMode::Normal => s,
        BlendMode::Dissolve => dissolve_blend(b, s, alpha),

        BlendMode::Darken => darken_blend(b, s),
        BlendMode::Multiply => multiply_blend(b, s),
        BlendMode::ColorBurn => color_burn_blend(b, s),
        BlendMode::LinearBurn => linear_burn_blend(b, s),
        BlendMode::DarkerColor => darker_color_blend(b, s),

        BlendMode::Lighten => lighten_blend(b, s),
        BlendMode::Screen => screen_blend(b, s),
        BlendMode::ColorDodge => color_dodge_blend(b, s),
        BlendMode::LinearDodge => linear_dodge_blend(b, s),
        BlendMode::LighterColor => lighter_color_blend(b, s),

        BlendMode::Overlay => overlay_blend(b, s),
        BlendMode::SoftLight => soft_light_blend(b, s),
        BlendMode::HardLight => hard_light_blend(b, s),
        BlendMode::VividLight => vivid_light_blend(b, s),
        BlendMode::LinearLight => linear_light_blend(b, s),
        BlendMode::PinLight => pin_light_blend(b, s),
        BlendMode::HardMix => hard_mix_blend(b, s),

        BlendMode::Difference => difference_blend(b, s),
        BlendMode::Exclusion => exclusion_blend(b, s),
        BlendMode::Subtract => subtract_blend(b, s),
        BlendMode::Divide => divide_blend(b, s),

        BlendMode::Hue => hue_blend(b, s),
        BlendMode::Saturation => saturation_blend(b, s),
        BlendMode::Color => color_blend(b, s),
        BlendMode::Luminosity => luminosity_blend(b, s),

        BlendMode::Add => add_blend(b, s),
        BlendMode::Average => average_blend(b, s),
        BlendMode::Negation => negation_blend(b, s),
        BlendMode::Phoenix => phoenix_blend(b, s),

        BlendMode::Reflect => reflect_blend(b, s),
        BlendMode::Glow => glow_blend(b, s),
    };

    // Composite with alpha
    let final_alpha = alpha + b[3] * (1.0 - alpha);
    let composite = if final_alpha > 0.0 {
        [
            (result[0] * alpha + b[0] * b[3] * (1.0 - alpha)) / final_alpha,
            (result[1] * alpha + b[1] * b[3] * (1.0 - alpha)) / final_alpha,
            (result[2] * alpha + b[2] * b[3] * (1.0 - alpha)) / final_alpha,
            final_alpha,
        ]
    } else {
        [0.0, 0.0, 0.0, 0.0]
    };

    // Convert back to u8
    [
        (composite[0] * 255.0).clamp(0.0, 255.0) as u8,
        (composite[1] * 255.0).clamp(0.0, 255.0) as u8,
        (composite[2] * 255.0).clamp(0.0, 255.0) as u8,
        (composite[3] * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

fn dissolve_blend(_b: [f32; 4], s: [f32; 4], _alpha: f32) -> [f32; 4] {
    // Simplified dissolve - would use random threshold in real implementation
    s
}

fn darken_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [b[0].min(s[0]), b[1].min(s[1]), b[2].min(s[2]), s[3]]
}

fn multiply_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [b[0] * s[0], b[1] * s[1], b[2] * s[2], s[3]]
}

fn color_burn_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        if s[0] > 0.0 {
            1.0 - ((1.0 - b[0]) / s[0]).min(1.0)
        } else {
            0.0
        },
        if s[1] > 0.0 {
            1.0 - ((1.0 - b[1]) / s[1]).min(1.0)
        } else {
            0.0
        },
        if s[2] > 0.0 {
            1.0 - ((1.0 - b[2]) / s[2]).min(1.0)
        } else {
            0.0
        },
        s[3],
    ]
}

fn linear_burn_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        (b[0] + s[0] - 1.0).max(0.0),
        (b[1] + s[1] - 1.0).max(0.0),
        (b[2] + s[2] - 1.0).max(0.0),
        s[3],
    ]
}

fn darker_color_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let b_lum = b[0] * 0.299 + b[1] * 0.587 + b[2] * 0.114;
    let s_lum = s[0] * 0.299 + s[1] * 0.587 + s[2] * 0.114;
    if s_lum < b_lum {
        s
    } else {
        b
    }
}

fn lighten_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [b[0].max(s[0]), b[1].max(s[1]), b[2].max(s[2]), s[3]]
}

fn screen_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        1.0 - (1.0 - b[0]) * (1.0 - s[0]),
        1.0 - (1.0 - b[1]) * (1.0 - s[1]),
        1.0 - (1.0 - b[2]) * (1.0 - s[2]),
        s[3],
    ]
}

fn color_dodge_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        if s[0] < 1.0 {
            (b[0] / (1.0 - s[0])).min(1.0)
        } else {
            1.0
        },
        if s[1] < 1.0 {
            (b[1] / (1.0 - s[1])).min(1.0)
        } else {
            1.0
        },
        if s[2] < 1.0 {
            (b[2] / (1.0 - s[2])).min(1.0)
        } else {
            1.0
        },
        s[3],
    ]
}

fn linear_dodge_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        (b[0] + s[0]).min(1.0),
        (b[1] + s[1]).min(1.0),
        (b[2] + s[2]).min(1.0),
        s[3],
    ]
}

fn lighter_color_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let b_lum = b[0] * 0.299 + b[1] * 0.587 + b[2] * 0.114;
    let s_lum = s[0] * 0.299 + s[1] * 0.587 + s[2] * 0.114;
    if s_lum > b_lum {
        s
    } else {
        b
    }
}

fn overlay_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let blend_channel = |backdrop: f32, source: f32| {
        if backdrop < 0.5 {
            2.0 * backdrop * source
        } else {
            1.0 - 2.0 * (1.0 - backdrop) * (1.0 - source)
        }
    };
    [
        blend_channel(b[0], s[0]),
        blend_channel(b[1], s[1]),
        blend_channel(b[2], s[2]),
        s[3],
    ]
}

fn soft_light_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let blend_channel = |backdrop: f32, source: f32| {
        if source < 0.5 {
            backdrop - (1.0 - 2.0 * source) * backdrop * (1.0 - backdrop)
        } else {
            backdrop + (2.0 * source - 1.0) * (d_function(backdrop) - backdrop)
        }
    };
    [
        blend_channel(b[0], s[0]),
        blend_channel(b[1], s[1]),
        blend_channel(b[2], s[2]),
        s[3],
    ]
}

fn d_function(x: f32) -> f32 {
    if x <= 0.25 {
        ((16.0 * x - 12.0) * x + 4.0) * x
    } else {
        x.sqrt()
    }
}

fn hard_light_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let blend_channel = |backdrop: f32, source: f32| {
        if source < 0.5 {
            2.0 * backdrop * source
        } else {
            1.0 - 2.0 * (1.0 - backdrop) * (1.0 - source)
        }
    };
    [
        blend_channel(b[0], s[0]),
        blend_channel(b[1], s[1]),
        blend_channel(b[2], s[2]),
        s[3],
    ]
}

fn vivid_light_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let blend_channel = |backdrop: f32, source: f32| {
        if source < 0.5 {
            if source > 0.0 {
                1.0 - ((1.0 - backdrop) / (2.0 * source)).min(1.0)
            } else {
                0.0
            }
        } else if source < 1.0 {
            (backdrop / (2.0 * (1.0 - source))).min(1.0)
        } else {
            1.0
        }
    };
    [
        blend_channel(b[0], s[0]),
        blend_channel(b[1], s[1]),
        blend_channel(b[2], s[2]),
        s[3],
    ]
}

fn linear_light_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        (b[0] + 2.0 * s[0] - 1.0).clamp(0.0, 1.0),
        (b[1] + 2.0 * s[1] - 1.0).clamp(0.0, 1.0),
        (b[2] + 2.0 * s[2] - 1.0).clamp(0.0, 1.0),
        s[3],
    ]
}

fn pin_light_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let blend_channel = |backdrop: f32, source: f32| {
        if source < 0.5 {
            backdrop.min(2.0 * source)
        } else {
            backdrop.max(2.0 * source - 1.0)
        }
    };
    [
        blend_channel(b[0], s[0]),
        blend_channel(b[1], s[1]),
        blend_channel(b[2], s[2]),
        s[3],
    ]
}

fn hard_mix_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let blend_channel = |backdrop: f32, source: f32| {
        if backdrop + source < 1.0 {
            0.0
        } else {
            1.0
        }
    };
    [
        blend_channel(b[0], s[0]),
        blend_channel(b[1], s[1]),
        blend_channel(b[2], s[2]),
        s[3],
    ]
}

fn difference_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        (b[0] - s[0]).abs(),
        (b[1] - s[1]).abs(),
        (b[2] - s[2]).abs(),
        s[3],
    ]
}

fn exclusion_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        b[0] + s[0] - 2.0 * b[0] * s[0],
        b[1] + s[1] - 2.0 * b[1] * s[1],
        b[2] + s[2] - 2.0 * b[2] * s[2],
        s[3],
    ]
}

fn subtract_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        (b[0] - s[0]).max(0.0),
        (b[1] - s[1]).max(0.0),
        (b[2] - s[2]).max(0.0),
        s[3],
    ]
}

fn divide_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        if s[0] > 0.0 {
            (b[0] / s[0]).min(1.0)
        } else {
            1.0
        },
        if s[1] > 0.0 {
            (b[1] / s[1]).min(1.0)
        } else {
            1.0
        },
        if s[2] > 0.0 {
            (b[2] / s[2]).min(1.0)
        } else {
            1.0
        },
        s[3],
    ]
}

fn hue_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let b_hsl = rgb_to_hsl([b[0], b[1], b[2]]);
    let s_hsl = rgb_to_hsl([s[0], s[1], s[2]]);
    let result_rgb = hsl_to_rgb([s_hsl[0], b_hsl[1], b_hsl[2]]);
    [result_rgb[0], result_rgb[1], result_rgb[2], s[3]]
}

fn saturation_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let b_hsl = rgb_to_hsl([b[0], b[1], b[2]]);
    let s_hsl = rgb_to_hsl([s[0], s[1], s[2]]);
    let result_rgb = hsl_to_rgb([b_hsl[0], s_hsl[1], b_hsl[2]]);
    [result_rgb[0], result_rgb[1], result_rgb[2], s[3]]
}

fn color_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let b_hsl = rgb_to_hsl([b[0], b[1], b[2]]);
    let s_hsl = rgb_to_hsl([s[0], s[1], s[2]]);
    let result_rgb = hsl_to_rgb([s_hsl[0], s_hsl[1], b_hsl[2]]);
    [result_rgb[0], result_rgb[1], result_rgb[2], s[3]]
}

fn luminosity_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    let b_hsl = rgb_to_hsl([b[0], b[1], b[2]]);
    let s_hsl = rgb_to_hsl([s[0], s[1], s[2]]);
    let result_rgb = hsl_to_rgb([b_hsl[0], b_hsl[1], s_hsl[2]]);
    [result_rgb[0], result_rgb[1], result_rgb[2], s[3]]
}

fn add_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        (b[0] + s[0]).min(1.0),
        (b[1] + s[1]).min(1.0),
        (b[2] + s[2]).min(1.0),
        s[3],
    ]
}

fn average_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        (b[0] + s[0]) / 2.0,
        (b[1] + s[1]) / 2.0,
        (b[2] + s[2]) / 2.0,
        s[3],
    ]
}

fn negation_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        1.0 - (1.0 - b[0] - s[0]).abs(),
        1.0 - (1.0 - b[1] - s[1]).abs(),
        1.0 - (1.0 - b[2] - s[2]).abs(),
        s[3],
    ]
}

fn phoenix_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        b[0].min(s[0]) - b[0].max(s[0]) + 1.0,
        b[1].min(s[1]) - b[1].max(s[1]) + 1.0,
        b[2].min(s[2]) - b[2].max(s[2]) + 1.0,
        s[3],
    ]
}

fn reflect_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        if s[0] < 1.0 {
            (b[0] * b[0] / (1.0 - s[0])).min(1.0)
        } else {
            1.0
        },
        if s[1] < 1.0 {
            (b[1] * b[1] / (1.0 - s[1])).min(1.0)
        } else {
            1.0
        },
        if s[2] < 1.0 {
            (b[2] * b[2] / (1.0 - s[2])).min(1.0)
        } else {
            1.0
        },
        s[3],
    ]
}

fn glow_blend(b: [f32; 4], s: [f32; 4]) -> [f32; 4] {
    [
        if b[0] < 1.0 {
            (s[0] * s[0] / (1.0 - b[0])).min(1.0)
        } else {
            1.0
        },
        if b[1] < 1.0 {
            (s[1] * s[1] / (1.0 - b[1])).min(1.0)
        } else {
            1.0
        },
        if b[2] < 1.0 {
            (s[2] * s[2] / (1.0 - b[2])).min(1.0)
        } else {
            1.0
        },
        s[3],
    ]
}

fn rgb_to_hsl(rgb: [f32; 3]) -> [f32; 3] {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let delta = max - min;

    let l = (max + min) / 2.0;

    if delta == 0.0 {
        return [0.0, 0.0, l];
    }

    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    let h = if max == rgb[0] {
        ((rgb[1] - rgb[2]) / delta + if rgb[1] < rgb[2] { 6.0 } else { 0.0 }) / 6.0
    } else if max == rgb[1] {
        ((rgb[2] - rgb[0]) / delta + 2.0) / 6.0
    } else {
        ((rgb[0] - rgb[1]) / delta + 4.0) / 6.0
    };

    [h, s, l]
}

fn hsl_to_rgb(hsl: [f32; 3]) -> [f32; 3] {
    let h = hsl[0];
    let s = hsl[1];
    let l = hsl[2];

    if s == 0.0 {
        return [l, l, l];
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;

    let hue_to_rgb = |p: f32, q: f32, t: f32| {
        let mut t = t;
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            return p + (q - p) * 6.0 * t;
        }
        if t < 1.0 / 2.0 {
            return q;
        }
        if t < 2.0 / 3.0 {
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
        }
        p
    };

    [
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_blend() {
        let backdrop = [128, 128, 128, 255];
        let source = [255, 0, 0, 255];
        let result = blend_pixels(backdrop, source, BlendMode::Normal, 1.0);
        assert_eq!(result, [255, 0, 0, 255]);
    }

    #[test]
    fn test_multiply_blend() {
        let backdrop = [128, 128, 128, 255];
        let source = [255, 255, 255, 255];
        let result = blend_pixels(backdrop, source, BlendMode::Multiply, 1.0);
        assert!(result[0] > 0 && result[0] < 255);
    }

    #[test]
    fn test_screen_blend() {
        let backdrop = [128, 128, 128, 255];
        let source = [128, 128, 128, 255];
        let result = blend_pixels(backdrop, source, BlendMode::Screen, 1.0);
        assert!(result[0] >= 128);
    }

    #[test]
    fn test_blend_with_opacity() {
        let backdrop = [0, 0, 0, 255];
        let source = [255, 255, 255, 255];
        let result = blend_pixels(backdrop, source, BlendMode::Normal, 0.5);
        assert!(result[0] > 100 && result[0] < 200);
    }

    #[test]
    fn test_rgb_hsl_conversion() {
        let rgb = [1.0, 0.0, 0.0];
        let hsl = rgb_to_hsl(rgb);
        let rgb2 = hsl_to_rgb(hsl);
        assert!((rgb[0] - rgb2[0]).abs() < 0.01);
        assert!((rgb[1] - rgb2[1]).abs() < 0.01);
        assert!((rgb[2] - rgb2[2]).abs() < 0.01);
    }
}
