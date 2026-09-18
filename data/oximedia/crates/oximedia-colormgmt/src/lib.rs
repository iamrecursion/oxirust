#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_arguments
)]
//! Professional color management system for `OxiMedia`.
//!
//! This crate provides comprehensive color management capabilities including:
//!
//! - **Standard Color Spaces**: sRGB, Adobe RGB, `ProPhoto` RGB, Display P3, Rec.709, Rec.2020, DCI-P3
//! - **ACES Support**: Full ACES workflow (IDT, RRT, ODT, LMT) with AP0 and AP1 primaries
//! - **ICC Profile Support**: Parse, validate, and apply ICC v2/v4 profiles
//! - **HDR Processing**: PQ/HLG transfer functions, tone mapping operators
//! - **Gamut Mapping**: Advanced gamut compression and expansion algorithms
//! - **Color Transforms**: Matrix-based, LUT-based, and parametric transforms
//! - **Professional Accuracy**: ΔE < 1 for standard conversions, proper linear-light processing
//!
//! # Examples
//!
//! ## Basic Color Space Conversion
//!
//! ```no_run
//! use oximedia_colormgmt::{colorspaces::ColorSpace, transforms::rgb_to_rgb};
//!
//! let srgb = ColorSpace::srgb().expect("srgb");
//! let rec2020 = ColorSpace::rec2020().expect("rec2020");
//!
//! let rgb = [0.5, 0.3, 0.2];
//! let converted = rgb_to_rgb(&rgb, &srgb, &rec2020);
//! ```
//!
//! ## ACES Workflow
//!
//! ```
//! use oximedia_colormgmt::aces::{AcesColorSpace, AcesTransform};
//!
//! // Convert from ACEScg to ACES2065-1
//! let acescg = AcesColorSpace::ACEScg;
//! let aces2065 = AcesColorSpace::ACES2065_1;
//!
//! let transform = AcesTransform::new(acescg, aces2065);
//! let converted = transform.apply([0.5, 0.3, 0.2]);
//! ```
//!
//! ## Color Pipeline
//!
//! ```no_run
//! use oximedia_colormgmt::pipeline::{ColorPipeline, ColorTransform};
//! use oximedia_colormgmt::colorspaces::ColorSpace;
//!
//! let srgb = ColorSpace::srgb().expect("srgb");
//! let mut pipeline = ColorPipeline::new();
//! pipeline.add_transform(ColorTransform::Linearize(srgb));
//! pipeline.add_transform(ColorTransform::Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]));
//!
//! let result = pipeline.transform_pixel([0.5, 0.3, 0.2]);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod aces;
pub mod aces_config;
pub mod aces_gamut;
pub mod aces_output_transform;
pub mod aces_pipeline;
pub mod chromatic_adapt;
pub mod chromatic_adaptation;
pub mod ciecam02;
pub mod color_appearance;
pub mod color_blindness;
pub mod color_checker;
pub mod color_convert;
pub mod color_diff;
pub mod color_diff_batch;
pub mod color_difference;
pub mod color_harmony;
pub mod color_palette;
pub mod color_pipeline;
pub mod color_quantize;
pub mod color_temperature;
pub mod colorspaces;
pub mod ctl_interpreter;
pub mod ctl_lexer;
pub mod ctl_parser;
pub mod curves;
pub mod cusp_gamut;
pub mod cusp_gamut_map;
pub mod display_profile;
pub mod error;
pub mod gamut;
pub mod gamut_clip;
pub mod gamut_compression;
pub mod gamut_mapping;
pub mod gamut_ops;
pub mod gamuts;
pub mod grading;
pub mod hdr;
pub mod hdr_color;
pub mod hdr_gamut;
pub mod icc;
pub mod icc_profile;
pub mod icc_reader;
pub mod icc_v5;
pub mod ictcp;
pub mod illuminant_tests;
pub mod jzazbz;
pub mod lab_color;
pub mod look_table;
pub mod lut_chain_opt;
pub mod lut_interp;
pub mod match_color;
pub mod math;
pub mod matrix_coeff;
pub mod multi_illuminant;
pub mod ocio;
pub mod ocio_config;
pub mod ocio_parser;
pub mod oklab;
pub mod perceptual_uniformity;
pub mod pipeline;
pub mod rendering_intent;
pub mod soft_clip_gamut;
pub mod spectral_data;
pub mod spectral_locus;
pub mod spectral_upsampling;
pub mod tone_map;
pub mod transfer_function;
pub mod transforms;
pub mod utils;
pub mod white_balance;
pub mod white_point;
pub mod xyz;

pub use color_convert::{ColorSpaceId, ColorTransformUtil, TransferFunctionId};

pub use error::{ColorError, Result};

/// Delta E (color difference) calculations.
pub mod delta_e {
    //! Color difference metrics (ΔE) for perceptual color comparison.
    //!
    //! Supports CIE76 (1976) and CIEDE2000 with configurable parametric weighting
    //! factors (k_L, k_C, k_H) per CIE 142-2001.

    use crate::xyz::Lab;

    /// Configurable weighting factors for CIEDE2000.
    ///
    /// Per CIE 142-2001, the weighting factors k_L, k_C, k_H allow the formula
    /// to be tuned for different application domains:
    ///
    /// - **Reference conditions**: k_L = k_C = k_H = 1.0 (default)
    /// - **Textiles**: k_L = 2.0, k_C = k_H = 1.0 (the SL weighting reduces
    ///   lightness sensitivity for fabric samples)
    /// - **Graphic arts**: k_L = 1.0, k_C = 1.0, k_H = 1.0
    ///
    /// The weighting factors are clamped to a minimum of 0.01 to prevent division
    /// by zero or numerical instability.
    #[derive(Debug, Clone)]
    pub struct CieDe2000Weights {
        /// Lightness weighting factor (default 1.0, 2.0 for textiles).
        pub k_l: f64,
        /// Chroma weighting factor (default 1.0).
        pub k_c: f64,
        /// Hue weighting factor (default 1.0).
        pub k_h: f64,
    }

    impl CieDe2000Weights {
        /// Reference (standard) conditions: k_L = k_C = k_H = 1.0.
        #[must_use]
        pub fn reference() -> Self {
            Self {
                k_l: 1.0,
                k_c: 1.0,
                k_h: 1.0,
            }
        }

        /// Textile industry conditions: k_L = 2.0, k_C = k_H = 1.0.
        ///
        /// Per CIE 142-2001, the larger k_L reduces the contribution of lightness
        /// differences, which is appropriate for textile color matching where
        /// observers are more tolerant of lightness variation.
        #[must_use]
        pub fn textiles() -> Self {
            Self {
                k_l: 2.0,
                k_c: 1.0,
                k_h: 1.0,
            }
        }

        /// Custom weights.
        ///
        /// Values are clamped to a minimum of 0.01.
        #[must_use]
        pub fn custom(k_l: f64, k_c: f64, k_h: f64) -> Self {
            Self {
                k_l: k_l.max(0.01),
                k_c: k_c.max(0.01),
                k_h: k_h.max(0.01),
            }
        }
    }

    impl Default for CieDe2000Weights {
        fn default() -> Self {
            Self::reference()
        }
    }

    /// Calculates ΔE 1976 (CIE76) - simple Euclidean distance in Lab space.
    ///
    /// # Arguments
    ///
    /// * `lab1` - First color in Lab
    /// * `lab2` - Second color in Lab
    ///
    /// # Returns
    ///
    /// Color difference value (ΔE). Values < 1 are imperceptible, < 2.3 are acceptable.
    #[must_use]
    pub fn delta_e_1976(lab1: &Lab, lab2: &Lab) -> f64 {
        let dl = lab1.l - lab2.l;
        let da = lab1.a - lab2.a;
        let db = lab1.b - lab2.b;
        (dl * dl + da * da + db * db).sqrt()
    }

    /// Calculates ΔE 2000 (CIEDE2000) with explicit scalar weighting factors.
    ///
    /// This is a convenience wrapper over [`delta_e_2000_weighted`] that accepts
    /// plain `[f64; 3]` Lab arrays and explicit k_L / k_C / k_H scalars so that
    /// callers do not need to construct a [`Lab`] or [`CieDe2000Weights`] struct.
    ///
    /// # Arguments
    ///
    /// * `lab1` - First color as `[L, a, b]`
    /// * `lab2` - Second color as `[L, a, b]`
    /// * `k_l`  - Lightness weighting factor (1.0 = standard, 2.0 = textiles)
    /// * `k_c`  - Chroma weighting factor (default 1.0)
    /// * `k_h`  - Hue weighting factor (default 1.0)
    ///
    /// # Returns
    ///
    /// ΔE 2000 with the given weights applied.
    #[must_use]
    pub fn delta_e_2000_weighted_arr(
        lab1: [f64; 3],
        lab2: [f64; 3],
        k_l: f64,
        k_c: f64,
        k_h: f64,
    ) -> f64 {
        let l1 = Lab::new(lab1[0], lab1[1], lab1[2]);
        let l2 = Lab::new(lab2[0], lab2[1], lab2[2]);
        delta_e_2000_weighted(&l1, &l2, &CieDe2000Weights::custom(k_l, k_c, k_h))
    }

    /// Calculates ΔE 2000 (CIEDE2000) with default reference weights (k_L=k_C=k_H=1).
    ///
    /// More accurate than ΔE 1976, accounting for perceptual non-uniformities.
    ///
    /// # Arguments
    ///
    /// * `lab1` - First color in Lab
    /// * `lab2` - Second color in Lab
    ///
    /// # Returns
    ///
    /// Color difference value (ΔE 2000). Values < 1 are imperceptible.
    #[must_use]
    pub fn delta_e_2000(lab1: &Lab, lab2: &Lab) -> f64 {
        delta_e_2000_weighted(lab1, lab2, &CieDe2000Weights::reference())
    }

    /// Calculates ΔE 2000 (CIEDE2000) with configurable weighting parameters.
    ///
    /// The parametric factors k_L, k_C, and k_H allow tuning the formula for
    /// different application domains (textiles, graphic arts, etc.).
    ///
    /// # Arguments
    ///
    /// * `lab1` - First color in Lab
    /// * `lab2` - Second color in Lab
    /// * `weights` - Weighting factors for lightness, chroma, and hue
    ///
    /// # Returns
    ///
    /// Color difference value (ΔE 2000) with the specified weights applied.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn delta_e_2000_weighted(lab1: &Lab, lab2: &Lab, weights: &CieDe2000Weights) -> f64 {
        use std::f64::consts::PI;

        let l1 = lab1.l;
        let a1 = lab1.a;
        let b1 = lab1.b;
        let l2 = lab2.l;
        let a2 = lab2.a;
        let b2 = lab2.b;

        let k_l = weights.k_l.max(0.01);
        let k_c = weights.k_c.max(0.01);
        let k_h = weights.k_h.max(0.01);

        // Calculate C1 and C2
        let c1 = (a1 * a1 + b1 * b1).sqrt();
        let c2 = (a2 * a2 + b2 * b2).sqrt();
        let c_bar = (c1 + c2) / 2.0;

        // Calculate G
        let g = 0.5 * (1.0 - ((c_bar.powi(7)) / (c_bar.powi(7) + 25.0_f64.powi(7))).sqrt());

        // Calculate a'
        let a1_prime = a1 * (1.0 + g);
        let a2_prime = a2 * (1.0 + g);

        // Calculate C' and h'
        let c1_prime = (a1_prime * a1_prime + b1 * b1).sqrt();
        let c2_prime = (a2_prime * a2_prime + b2 * b2).sqrt();

        let h1_prime = if b1 == 0.0 && a1_prime == 0.0 {
            0.0
        } else {
            let mut h = b1.atan2(a1_prime).to_degrees();
            if h < 0.0 {
                h += 360.0;
            }
            h
        };

        let h2_prime = if b2 == 0.0 && a2_prime == 0.0 {
            0.0
        } else {
            let mut h = b2.atan2(a2_prime).to_degrees();
            if h < 0.0 {
                h += 360.0;
            }
            h
        };

        // Calculate ΔL', ΔC', ΔH'
        let delta_l_prime = l2 - l1;
        let delta_c_prime = c2_prime - c1_prime;

        let delta_h_prime = if c1_prime * c2_prime == 0.0 {
            0.0
        } else if (h2_prime - h1_prime).abs() <= 180.0 {
            h2_prime - h1_prime
        } else if h2_prime - h1_prime > 180.0 {
            h2_prime - h1_prime - 360.0
        } else {
            h2_prime - h1_prime + 360.0
        };

        let delta_big_h_prime =
            2.0 * (c1_prime * c2_prime).sqrt() * ((delta_h_prime / 2.0) * PI / 180.0).sin();

        // Calculate L', C', H' bar
        let l_bar_prime = (l1 + l2) / 2.0;
        let c_bar_prime = (c1_prime + c2_prime) / 2.0;

        let h_bar_prime = if c1_prime * c2_prime == 0.0 {
            h1_prime + h2_prime
        } else if (h1_prime - h2_prime).abs() <= 180.0 {
            (h1_prime + h2_prime) / 2.0
        } else if h1_prime + h2_prime < 360.0 {
            (h1_prime + h2_prime + 360.0) / 2.0
        } else {
            (h1_prime + h2_prime - 360.0) / 2.0
        };

        // Calculate T
        let t = 1.0 - 0.17 * ((h_bar_prime - 30.0) * PI / 180.0).cos()
            + 0.24 * ((2.0 * h_bar_prime) * PI / 180.0).cos()
            + 0.32 * ((3.0 * h_bar_prime + 6.0) * PI / 180.0).cos()
            - 0.20 * ((4.0 * h_bar_prime - 63.0) * PI / 180.0).cos();

        // Calculate S_L, S_C, S_H
        let s_l = 1.0
            + ((0.015 * (l_bar_prime - 50.0).powi(2))
                / (20.0 + (l_bar_prime - 50.0).powi(2)).sqrt());
        let s_c = 1.0 + 0.045 * c_bar_prime;
        let s_h = 1.0 + 0.015 * c_bar_prime * t;

        // Calculate R_T
        let delta_theta = 30.0 * (-(((h_bar_prime - 275.0) / 25.0).powi(2))).exp();
        let r_c = 2.0 * ((c_bar_prime.powi(7)) / (c_bar_prime.powi(7) + 25.0_f64.powi(7))).sqrt();
        let r_t = -r_c * (2.0 * delta_theta * PI / 180.0).sin();

        // Calculate ΔE 2000 with configurable weights
        ((delta_l_prime / (k_l * s_l)).powi(2)
            + (delta_c_prime / (k_c * s_c)).powi(2)
            + (delta_big_h_prime / (k_h * s_h)).powi(2)
            + r_t * (delta_c_prime / (k_c * s_c)) * (delta_big_h_prime / (k_h * s_h)))
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::delta_e::*;
    use crate::xyz::Lab;

    // ── Required integration tests ────────────────────────────────────────────

    #[test]
    fn test_oklab_roundtrip() {
        use crate::oklab::{oklab_to_srgb, srgb_to_oklab};
        let colors: &[[f64; 3]] = &[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.2, 0.6, 0.8],
        ];
        for &rgb in colors {
            let lab = srgb_to_oklab(rgb);
            let rgb2 = oklab_to_srgb(lab);
            for i in 0..3 {
                assert!(
                    (rgb2[i] - rgb[i]).abs() < 1e-5,
                    "Oklab roundtrip channel {i}: {} vs {} for input {:?}",
                    rgb2[i],
                    rgb[i],
                    rgb
                );
            }
        }
    }

    #[test]
    fn test_jzazbz_roundtrip() {
        use crate::jzazbz::{jzazbz_to_xyz, xyz_to_jzazbz};
        let colors: &[[f64; 3]] = &[
            [95.047, 100.0, 108.883],
            [41.24, 21.26, 1.93],
            [18.05, 7.22, 95.05],
            [50.0, 50.0, 50.0],
        ];
        for &xyz in colors {
            let jab = xyz_to_jzazbz(xyz);
            let xyz2 = jzazbz_to_xyz(jab);
            let tol = (xyz[0].abs().max(xyz[1].abs()).max(xyz[2].abs())) * 0.01 + 0.1;
            for i in 0..3 {
                assert!(
                    (xyz2[i] - xyz[i]).abs() < tol,
                    "Jzazbz roundtrip channel {i}: {} vs {} (tol={tol}) for input {:?}",
                    xyz2[i],
                    xyz[i],
                    xyz
                );
            }
        }
    }

    #[test]
    fn test_delta_e2000_weighted_default_is_standard() {
        let lab1 = Lab::new(50.0, 25.0, 10.0);
        let lab2 = Lab::new(55.0, 20.0, 15.0);
        let de_standard = delta_e_2000(&lab1, &lab2);
        let de_arr = delta_e_2000_weighted_arr(
            [lab1.l, lab1.a, lab1.b],
            [lab2.l, lab2.a, lab2.b],
            1.0,
            1.0,
            1.0,
        );
        assert!(
            (de_standard - de_arr).abs() < 1e-10,
            "Array weighted with k=1 should match standard: {} vs {}",
            de_standard,
            de_arr
        );
    }

    #[test]
    fn test_crate_version() {
        assert!(!env!("CARGO_PKG_VERSION").is_empty());
    }

    // ── CIEDE2000 weighting tests ────────────────────────────────────────────

    #[test]
    fn test_ciede2000_default_weights_match_unweighted() {
        let lab1 = Lab::new(50.0, 25.0, 10.0);
        let lab2 = Lab::new(55.0, 20.0, 15.0);
        let de_default = delta_e_2000(&lab1, &lab2);
        let de_weighted = delta_e_2000_weighted(&lab1, &lab2, &CieDe2000Weights::reference());
        assert!(
            (de_default - de_weighted).abs() < 1e-10,
            "Default and reference weights should match: {} vs {}",
            de_default,
            de_weighted
        );
    }

    #[test]
    fn test_ciede2000_textile_weights() {
        let lab1 = Lab::new(50.0, 25.0, 10.0);
        let lab2 = Lab::new(60.0, 25.0, 10.0);
        let de_ref = delta_e_2000(&lab1, &lab2);
        let de_textile = delta_e_2000_weighted(&lab1, &lab2, &CieDe2000Weights::textiles());
        // With k_L=2.0, lightness-only difference should be halved
        assert!(
            de_textile < de_ref,
            "Textile dE ({}) should be less than reference ({}) for lightness-only diff",
            de_textile,
            de_ref
        );
    }

    #[test]
    fn test_ciede2000_larger_kl_reduces_lightness_sensitivity() {
        let lab1 = Lab::new(50.0, 0.0, 0.0);
        let lab2 = Lab::new(70.0, 0.0, 0.0);
        let de_kl1 = delta_e_2000_weighted(&lab1, &lab2, &CieDe2000Weights::custom(1.0, 1.0, 1.0));
        let de_kl2 = delta_e_2000_weighted(&lab1, &lab2, &CieDe2000Weights::custom(2.0, 1.0, 1.0));
        let de_kl3 = delta_e_2000_weighted(&lab1, &lab2, &CieDe2000Weights::custom(3.0, 1.0, 1.0));
        assert!(de_kl2 < de_kl1, "k_L=2 should give smaller dE than k_L=1");
        assert!(de_kl3 < de_kl2, "k_L=3 should give smaller dE than k_L=2");
    }

    #[test]
    fn test_ciede2000_larger_kc_reduces_chroma_sensitivity() {
        let lab1 = Lab::new(50.0, 0.0, 0.0);
        let lab2 = Lab::new(50.0, 30.0, 0.0);
        let de_kc1 = delta_e_2000_weighted(&lab1, &lab2, &CieDe2000Weights::custom(1.0, 1.0, 1.0));
        let de_kc2 = delta_e_2000_weighted(&lab1, &lab2, &CieDe2000Weights::custom(1.0, 2.0, 1.0));
        assert!(de_kc2 < de_kc1, "k_C=2 should give smaller dE than k_C=1");
    }

    #[test]
    fn test_ciede2000_larger_kh_reduces_hue_sensitivity() {
        let lab1 = Lab::new(50.0, 30.0, 0.0);
        let lab2 = Lab::new(50.0, 0.0, 30.0);
        let de_kh1 = delta_e_2000_weighted(&lab1, &lab2, &CieDe2000Weights::custom(1.0, 1.0, 1.0));
        let de_kh2 = delta_e_2000_weighted(&lab1, &lab2, &CieDe2000Weights::custom(1.0, 1.0, 2.0));
        assert!(de_kh2 < de_kh1, "k_H=2 should give smaller dE than k_H=1");
    }

    #[test]
    fn test_ciede2000_custom_weights_clamp() {
        let w = CieDe2000Weights::custom(-1.0, 0.0, -5.0);
        assert!(w.k_l >= 0.01);
        assert!(w.k_c >= 0.01);
        assert!(w.k_h >= 0.01);
    }

    #[test]
    fn test_ciede2000_identical_colors() {
        let lab = Lab::new(50.0, 25.0, 10.0);
        let de = delta_e_2000(&lab, &lab);
        assert!(
            de.abs() < 1e-10,
            "Identical colors should have dE=0, got {}",
            de
        );
    }

    #[test]
    fn test_ciede2000_symmetry() {
        let lab1 = Lab::new(50.0, 25.0, 10.0);
        let lab2 = Lab::new(73.0, 15.0, -22.0);
        let de12 = delta_e_2000(&lab1, &lab2);
        let de21 = delta_e_2000(&lab2, &lab1);
        assert!(
            (de12 - de21).abs() < 1e-10,
            "CIEDE2000 should be symmetric: {} vs {}",
            de12,
            de21
        );
    }

    #[test]
    fn test_ciede2000_weighted_symmetry() {
        let lab1 = Lab::new(50.0, 25.0, 10.0);
        let lab2 = Lab::new(73.0, 15.0, -22.0);
        let w = CieDe2000Weights::textiles();
        let de12 = delta_e_2000_weighted(&lab1, &lab2, &w);
        let de21 = delta_e_2000_weighted(&lab2, &lab1, &w);
        assert!(
            (de12 - de21).abs() < 1e-10,
            "Weighted CIEDE2000 should be symmetric"
        );
    }

    #[test]
    fn test_delta_e_1976_known_value() {
        let lab1 = Lab::new(50.0, 0.0, 0.0);
        let lab2 = Lab::new(50.0, 30.0, 40.0);
        let de = delta_e_1976(&lab1, &lab2);
        assert!((de - 50.0).abs() < 1e-10, "dE76 should be 50.0, got {}", de);
    }
}
