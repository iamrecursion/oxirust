//! Wave 15 Slice B tests: ICC tile parallelism + chromatic adaptation cache.

use oximedia_calibrate::{
    chromatic::adapt::{ChromaticAdaptation, ChromaticAdaptationMethod},
    delta_e::delta_e_2000,
    icc::apply::IccProfileApplicator,
    icc::parse::IccProfile,
    Illuminant,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build an identity ICC applicator (1:1 RGB transform).
fn identity_applicator() -> IccProfileApplicator {
    let identity = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    IccProfileApplicator::new(IccProfile::new(
        "Identity".to_string(),
        identity,
        Illuminant::D65,
    ))
}

/// Serial (scalar) implementation of `apply_to_image` for bit-exact comparison.
fn apply_serial(applicator: &IccProfileApplicator, image_data: &[u8]) -> Vec<u8> {
    image_data
        .chunks_exact(3)
        .flat_map(|chunk| {
            let r = f64::from(chunk[0]) / 255.0;
            let g = f64::from(chunk[1]) / 255.0;
            let b = f64::from(chunk[2]) / 255.0;
            let t = applicator.apply_to_color(&[r, g, b]);
            [
                (t[0] * 255.0).clamp(0.0, 255.0) as u8,
                (t[1] * 255.0).clamp(0.0, 255.0) as u8,
                (t[2] * 255.0).clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}

// ── Test 1: parallel output is bit-exact vs serial ────────────────────────────

#[test]
fn test_icc_parallel_matches_serial() {
    let applicator = identity_applicator();

    // Synthetic 100×100 RGB image — deterministic pseudo-random pixels.
    let width = 100usize;
    let height = 100usize;
    let mut image: Vec<u8> = Vec::with_capacity(width * height * 3);
    for i in 0..(width * height) {
        image.push((i % 256) as u8);
        image.push(((i * 7) % 256) as u8);
        image.push(((i * 13) % 256) as u8);
    }

    let serial_out = apply_serial(&applicator, &image);
    let parallel_out = applicator
        .apply_to_image(&image, width, height)
        .expect("parallel apply must succeed");

    assert_eq!(
        serial_out.len(),
        parallel_out.len(),
        "output lengths must match"
    );
    for (idx, (s, p)) in serial_out.iter().zip(parallel_out.iter()).enumerate() {
        assert_eq!(s, p, "byte {idx} differs: serial={s} parallel={p}");
    }
}

// ── Test 2: large-image parallel apply ───────────────────────────────────────

#[test]
fn test_icc_parallel_large_image() {
    let applicator = identity_applicator();

    let width = 1920usize;
    let height = 1080usize;
    let total_pixels = width * height;
    let mut image: Vec<u8> = Vec::with_capacity(total_pixels * 3);
    for i in 0..total_pixels {
        image.push(((i * 3) % 256) as u8);
        image.push(((i * 5) % 256) as u8);
        image.push(((i * 11) % 256) as u8);
    }

    let out = applicator
        .apply_to_image(&image, width, height)
        .expect("large-image apply must not panic");

    assert_eq!(out.len(), image.len(), "output size must match input size");

    // Ensure no pixel channel is NaN or inf — we roundtrip u8 so just check
    // that every byte is a valid u8 (always true), and that the identity
    // transform does not discard information.
    for (idx, (&inp, &out_b)) in image.iter().zip(out.iter()).enumerate() {
        // With the identity matrix the output should equal the input exactly
        // (within the u8→f64→u8 roundtrip that is lossless for this matrix).
        assert_eq!(
            inp, out_b,
            "identity ICC should preserve pixel byte at position {idx}: inp={inp} out={out_b}"
        );
    }
}

// ── Test 3: chromatic adaptation cache produces identical matrices ────────────

#[test]
fn test_chromatic_cache_matches_fresh() {
    let a = ChromaticAdaptation::new(
        ChromaticAdaptationMethod::Bradford,
        Illuminant::D50,
        Illuminant::D65,
    )
    .expect("first construction must succeed");

    let b = ChromaticAdaptation::new(
        ChromaticAdaptationMethod::Bradford,
        Illuminant::D50,
        Illuminant::D65,
    )
    .expect("second construction (cache hit) must succeed");

    // Both must carry identical 3×3 matrices.
    let ma = a.adapt_xyz(&[0.96422, 1.0, 0.82521]);
    let mb = b.adapt_xyz(&[0.96422, 1.0, 0.82521]);

    for i in 0..3 {
        assert!(
            (ma[i] - mb[i]).abs() < 1e-15,
            "element {i}: first={} second={} differ (cache must return same matrix)",
            ma[i],
            mb[i]
        );
    }
}

// ── Test 4: different keys yield different (non-trivially equal) matrices ─────

#[test]
fn test_chromatic_cache_different_keys() {
    // Bradford D50→D65 vs Von Kries D50→D65: different method, different matrix.
    let bradford = ChromaticAdaptation::new(
        ChromaticAdaptationMethod::Bradford,
        Illuminant::D50,
        Illuminant::D65,
    )
    .expect("Bradford must succeed");

    let von_kries = ChromaticAdaptation::new(
        ChromaticAdaptationMethod::VonKries,
        Illuminant::D50,
        Illuminant::D65,
    )
    .expect("VonKries must succeed");

    // The two methods produce different matrices — test input where they diverge.
    let test_xyz = [0.5, 0.4, 0.3];
    let out_b = bradford.adapt_xyz(&test_xyz);
    let out_v = von_kries.adapt_xyz(&test_xyz);

    let max_diff = out_b
        .iter()
        .zip(out_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff > 1e-6,
        "Bradford and VonKries must produce different results for D50→D65 (max_diff={max_diff})"
    );

    // Also verify D50→D65 vs D65→D50 (Bradford) differ.
    let inverse = ChromaticAdaptation::new(
        ChromaticAdaptationMethod::Bradford,
        Illuminant::D65,
        Illuminant::D50,
    )
    .expect("Bradford D65→D50 must succeed");

    let out_inv = inverse.adapt_xyz(&test_xyz);
    let max_diff2 = out_b
        .iter()
        .zip(out_inv.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff2 > 1e-6,
        "Forward and inverse Bradford transforms must differ (max_diff={max_diff2})"
    );
}

// ── Test 5: ICC profile write→parse→compare TRCs (matrix round-trip) ─────────

#[test]
fn test_icc_roundtrip() {
    // Synthesize a simple sRGB-to-XYZ matrix (the standard sRGB primaries).
    let srgb_to_xyz: [[f64; 3]; 3] = [
        [0.4124_f64, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ];

    let original = IccProfile::new("sRGB Wave15".to_string(), srgb_to_xyz, Illuminant::D65);

    // Serialize to binary ICC bytes.
    let bytes = original.to_bytes().expect("to_bytes must succeed");
    assert!(bytes.len() >= 128, "ICC binary must be at least 128 bytes");

    // Parse back.
    let parsed = IccProfile::from_bytes(&bytes).expect("from_bytes must succeed on our output");

    assert_eq!(parsed.description, original.description);
    assert_eq!(parsed.white_point, original.white_point);

    // TRC / matrix tolerance: s15Fixed16 quantization = 1/65536 ≈ 1.5e-5; use 1e-4.
    for row in 0..3 {
        for col in 0..3 {
            let orig = original.to_xyz_matrix[row][col];
            let got = parsed.to_xyz_matrix[row][col];
            assert!(
                (orig - got).abs() < 1e-4,
                "to_xyz_matrix[{row}][{col}]: original={orig:.6} parsed={got:.6} diff={}",
                (orig - got).abs()
            );
        }
    }
}

// ── Test 6: ΔE2000 < 2 for identity-calibrated ColorChecker patch ─────────────

/// Linear RGB → CIE L*a*b* (D65 reference white).
fn rgb_to_lab(rgb: [f64; 3]) -> [f64; 3] {
    // sRGB linearise (approximate, already linear for our synthesized input)
    let lin = rgb;

    // sRGB to XYZ (D65) matrix
    let x = lin[0] * 0.4124 + lin[1] * 0.3576 + lin[2] * 0.1805;
    let y = lin[0] * 0.2126 + lin[1] * 0.7152 + lin[2] * 0.0722;
    let z = lin[0] * 0.0193 + lin[1] * 0.1192 + lin[2] * 0.9505;

    // D65 reference white (Y=1)
    let xn = 0.95047_f64;
    let yn = 1.00000_f64;
    let zn = 1.08883_f64;

    let fx = lab_f(x / xn);
    let fy = lab_f(y / yn);
    let fz = lab_f(z / zn);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_star = 200.0 * (fy - fz);

    [l, a, b_star]
}

#[inline]
fn lab_f(t: f64) -> f64 {
    let delta = 6.0 / 29.0;
    if t > delta * delta * delta {
        t.cbrt()
    } else {
        t / (3.0 * delta * delta) + 4.0 / 29.0
    }
}

#[test]
fn test_delta_e_calibrated_under_2() {
    // Synthesize a near-neutral ColorChecker patch #19 (moderate olive):
    // approximate linearised sRGB values from published CIE Lab reference.
    // Reference CIE Lab: L*=55.26, a*=-38.34, b*=31.37 (X-Rite CC Classic patch 19).
    // We use a synthesized sRGB triplet that produces a known Lab and apply an
    // identity calibration, so measured ≈ reference → ΔE2000 ≈ 0.

    // Synthesized patch (normalized linear RGB, ~Olive-yellow swatch).
    let patch_rgb = [0.34_f64, 0.42_f64, 0.12_f64];

    // Reference Lab computed from the same formula (ground truth).
    let reference_lab = rgb_to_lab(patch_rgb);

    // Apply identity calibration: ICC with identity matrix → XYZ → back to RGB
    // using the inverse of the same identity = no change.
    let identity = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let applicator = IccProfileApplicator::new(IccProfile::new(
        "Identity Calibration".to_string(),
        identity,
        Illuminant::D65,
    ));

    // Encode the patch as an 8-bit pixel, apply, decode back.
    let r8 = (patch_rgb[0] * 255.0).clamp(0.0, 255.0) as u8;
    let g8 = (patch_rgb[1] * 255.0).clamp(0.0, 255.0) as u8;
    let b8 = (patch_rgb[2] * 255.0).clamp(0.0, 255.0) as u8;

    let input = vec![r8, g8, b8];
    let output = applicator
        .apply_to_image(&input, 1, 1)
        .expect("identity calibration must succeed");

    let out_rgb = [
        f64::from(output[0]) / 255.0,
        f64::from(output[1]) / 255.0,
        f64::from(output[2]) / 255.0,
    ];
    let measured_lab = rgb_to_lab(out_rgb);

    let de = delta_e_2000(reference_lab, measured_lab);

    assert!(
        de < 2.0,
        "ΔE2000 after identity calibration must be < 2.0, got {de:.4}\n  reference_lab={reference_lab:?}\n  measured_lab={measured_lab:?}"
    );
}
