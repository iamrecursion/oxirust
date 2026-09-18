//! Integration tests for the least-squares 3×3 colour-correction matrix.
//!
//! Validates [`CameraCalibrator::fit_color_matrix`]:
//!   * recovery of a planted linear transform,
//!   * identity recovery when measured == reference,
//!   * graceful error on a degenerate (singular) patch set,
//!   * perceptual ΔE2000 accuracy of the calibrated output, and
//!   * that the fit strictly beats the identity baseline.

use oximedia_calibrate::camera::calibrate::CameraCalibrator;
use oximedia_calibrate::camera::color_space::srgb_to_lab;
use oximedia_calibrate::camera::{ColorChecker, ColorCheckerType, PatchColor};
use oximedia_calibrate::delta_e::delta_e_2000;
use oximedia_calibrate::Matrix3x3;

// ── helpers ────────────────────────────────────────────────────────────────

/// Apply a row-major 3×3 matrix to a column vector.
fn mat_vec(m: &Matrix3x3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Build a `PatchColor` from measured/reference RGB. Lab/XYZ fields are filled
/// from the reference for completeness but are not used by the fit itself.
fn patch(index: usize, measured: [f64; 3], reference: [f64; 3]) -> PatchColor {
    PatchColor {
        index,
        measured_rgb: measured,
        reference_rgb: reference,
        reference_lab: srgb_to_lab(reference),
        reference_xyz: [0.0, 0.0, 0.0],
        name: format!("patch{index}"),
    }
}

/// Wrap a patch vector into a Classic24-typed `ColorChecker`.
fn checker(patches: Vec<PatchColor>) -> ColorChecker {
    ColorChecker {
        checker_type: ColorCheckerType::Classic24,
        patches,
        bounding_box: None,
        confidence: 1.0,
    }
}

/// Small deterministic pseudo-noise in [-amp, amp] from an integer seed.
fn jitter(seed: usize, amp: f64) -> f64 {
    // Cheap LCG-style hash → fractional value, mapped to [-amp, amp].
    let h = (seed.wrapping_mul(2_654_435_761) ^ 0x9E37_79B9) as u32;
    let frac = f64::from(h % 10_000) / 10_000.0; // [0,1)
    (frac * 2.0 - 1.0) * amp
}

// ── tests ──────────────────────────────────────────────────────────────────

/// A known mild, non-singular linear transform K applied to measured values
/// must be recovered by the least-squares fit (reference = K·measured).
#[test]
fn recovers_planted_linear_transform() {
    // Well-conditioned, asymmetric, non-singular K (channel mixing + scale).
    let k: Matrix3x3 = [
        [1.10, -0.05, 0.02],
        [0.03, 0.95, -0.01],
        [-0.02, 0.06, 1.08],
    ];

    // A diverse, high-energy set of measured colours that strongly spans R³ so
    // the smallest eigenvalue of A = Σ meas·measᵀ is well above 1.0. This keeps
    // the λ=1e-6 Tikhonov bias (≈ λ/σ_min(A)) below the 1e-6 recovery tolerance.
    let measured_set = [
        [0.95, 0.05, 0.05],
        [0.05, 0.95, 0.05],
        [0.05, 0.05, 0.95],
        [0.90, 0.90, 0.10],
        [0.90, 0.10, 0.90],
        [0.10, 0.90, 0.90],
        [0.95, 0.95, 0.95],
        [0.80, 0.20, 0.50],
        [0.20, 0.80, 0.50],
        [0.50, 0.20, 0.80],
        [0.70, 0.60, 0.30],
        [0.30, 0.70, 0.60],
        [0.60, 0.30, 0.70],
        [0.85, 0.45, 0.15],
        [0.15, 0.55, 0.85],
        [0.55, 0.85, 0.25],
    ];

    let patches: Vec<PatchColor> = measured_set
        .iter()
        .enumerate()
        .map(|(i, &meas)| patch(i, meas, mat_vec(&k, meas)))
        .collect();

    let calibrator = CameraCalibrator::default_calibrator();
    let m = calibrator
        .fit_color_matrix(&checker(patches))
        .expect("fit must succeed for a well-conditioned planted transform");

    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (m[i][j] - k[i][j]).abs() < 1e-6,
                "M[{i}][{j}]={} should recover K[{i}][{j}]={}",
                m[i][j],
                k[i][j]
            );
        }
    }
}

/// When measured == reference (classic24), the fitted matrix must be ~identity.
#[test]
fn identity_when_measured_equals_reference() {
    let calibrator = CameraCalibrator::default_calibrator();
    let cc = checker(ColorChecker::classic24_reference());

    let m = calibrator
        .fit_color_matrix(&cc)
        .expect("fit must succeed for classic24");

    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (m[i][j] - expected).abs() < 1e-4,
                "M[{i}][{j}]={} should be ~{expected}",
                m[i][j]
            );
        }
    }
}

/// A patch set where every patch is identical is rank-1 → the normal equations
/// are singular even after Tikhonov regularisation; the fit must return an
/// error (never panic).
#[test]
fn singular_patch_set_is_graceful_error() {
    let constant = [0.5, 0.5, 0.5];
    let patches: Vec<PatchColor> = (0..6)
        .map(|i| patch(i, constant, [0.6, 0.4, 0.5]))
        .collect();

    let calibrator = CameraCalibrator::default_calibrator();
    let result = calibrator.fit_color_matrix(&checker(patches));
    assert!(
        result.is_err(),
        "an all-identical (rank-1) patch set must yield an error, not a panic"
    );
}

/// With a realistic camera distortion + small per-patch noise, the calibrated
/// output (M·measured) must be perceptually close to the reference: mean
/// ΔE2000 in CIE Lab under 2.0.
#[test]
fn calibration_delta_e_under_2() {
    let (cc, _g) = synthetic_distorted_checker();

    let calibrator = CameraCalibrator::default_calibrator();
    let m = calibrator
        .fit_color_matrix(&cc)
        .expect("fit must succeed for the synthetic checker");

    let mut total = 0.0;
    for p in &cc.patches {
        let corrected = mat_vec(&m, p.measured_rgb);
        let corrected_clamped = [
            corrected[0].clamp(0.0, 1.0),
            corrected[1].clamp(0.0, 1.0),
            corrected[2].clamp(0.0, 1.0),
        ];
        let de = delta_e_2000(srgb_to_lab(corrected_clamped), srgb_to_lab(p.reference_rgb));
        total += de;
    }
    let mean_de = total / cc.patches.len() as f64;

    assert!(
        mean_de < 2.0,
        "mean ΔE2000 after calibration must be < 2.0, got {mean_de:.4}"
    );
}

/// The fitted matrix must strictly reduce perceptual error versus doing nothing
/// (identity). ΔE(after fit) < ΔE(identity).
#[test]
fn calibration_beats_identity_baseline() {
    let (cc, _g) = synthetic_distorted_checker();

    let calibrator = CameraCalibrator::default_calibrator();
    let m = calibrator
        .fit_color_matrix(&cc)
        .expect("fit must succeed for the synthetic checker");

    let mean_de = |transform: &dyn Fn([f64; 3]) -> [f64; 3]| -> f64 {
        let mut total = 0.0;
        for p in &cc.patches {
            let out = transform(p.measured_rgb);
            let out_c = [
                out[0].clamp(0.0, 1.0),
                out[1].clamp(0.0, 1.0),
                out[2].clamp(0.0, 1.0),
            ];
            total += delta_e_2000(srgb_to_lab(out_c), srgb_to_lab(p.reference_rgb));
        }
        total / cc.patches.len() as f64
    };

    let de_identity = mean_de(&|rgb| rgb);
    let de_fit = mean_de(&|rgb| mat_vec(&m, rgb));

    assert!(
        de_fit < de_identity,
        "calibrated ΔE ({de_fit:.4}) must beat identity ΔE ({de_identity:.4})"
    );
    // Sanity: the planted distortion is large enough that identity is clearly bad.
    assert!(
        de_identity > 2.0,
        "identity baseline should be visibly distorted (ΔE={de_identity:.4})"
    );
}

/// Build a synthetic ColorChecker whose `measured_rgb` is a realistic linear
/// camera distortion of the classic24 reference plus small per-patch noise.
/// Returns the checker and the distortion matrix G (measured = G·reference + ε).
fn synthetic_distorted_checker() -> (ColorChecker, Matrix3x3) {
    // Mild but clearly visible colour cast + channel cross-talk.
    let g: Matrix3x3 = [[0.92, 0.07, 0.04], [0.05, 0.88, 0.06], [0.03, 0.09, 0.85]];

    let patches: Vec<PatchColor> = ColorChecker::classic24_reference()
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let reference = p.reference_rgb;
            let distorted = mat_vec(&g, reference);
            let measured = [
                (distorted[0] + jitter(i * 3, 0.003)).clamp(0.0, 1.0),
                (distorted[1] + jitter(i * 3 + 1, 0.003)).clamp(0.0, 1.0),
                (distorted[2] + jitter(i * 3 + 2, 0.003)).clamp(0.0, 1.0),
            ];
            patch(i, measured, reference)
        })
        .collect();

    (checker(patches), g)
}
