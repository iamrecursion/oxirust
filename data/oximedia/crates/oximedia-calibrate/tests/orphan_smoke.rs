//! Smoke tests for newly-wired orphan modules in oximedia-calibrate.

#[test]
fn test_aces_calibration_constants_accessible() {
    use oximedia_calibrate::aces_calibration::{AP1_TO_XYZ, XYZ_TO_AP1};
    // Verify matrix constants are non-zero
    assert!(AP1_TO_XYZ[0][0] != 0.0);
    assert!(XYZ_TO_AP1[0][0] != 0.0);
}

#[test]
fn test_ambient_compensation_measurement_construct() {
    use oximedia_calibrate::ambient_compensation::AmbientMeasurement;
    let m = AmbientMeasurement::from_cct(6500.0, 500.0);
    assert!(m.is_ok(), "D65 measurement should succeed");
}

#[test]
fn test_batch_calibrate_camera_measurement_empty_fails() {
    use oximedia_calibrate::batch_calibrate::CameraMeasurement;
    // empty patches should return error
    let result = CameraMeasurement::new("cam01", vec![], vec![]);
    assert!(result.is_err(), "Empty patches should fail");
}

#[test]
fn test_calibration_extras_icc_builder() {
    use oximedia_calibrate::calibration_extras::{
        CalibrationLut, IccIlluminant, IccProfileBuilder, Primaries,
    };
    let builder = IccProfileBuilder::new(Primaries::Srgb, 2.2, IccIlluminant::D65);
    let profile = builder.build();
    // A minimal ICC profile should be non-empty
    assert!(!profile.is_empty());
    let _ = CalibrationLut;
}

#[test]
fn test_calibration_schedule_empty() {
    use oximedia_calibrate::calibration_schedule::CalibrationSchedule;
    let sched = CalibrationSchedule::new();
    assert_eq!(sched.device_count(), 0);
}

#[test]
fn test_hdr_calibration_pq_roundtrip() {
    use oximedia_calibrate::hdr_calibration::{pq_decode, pq_encode};
    let nits = 1000.0_f64;
    let code = pq_encode(nits);
    let back = pq_decode(code);
    assert!(
        (back - nits).abs() < 1.0,
        "PQ round-trip error: {}",
        (back - nits).abs()
    );
}

#[test]
fn test_hdr_calibration_hlg_encode_range() {
    use oximedia_calibrate::hdr_calibration::hlg_encode;
    let code = hlg_encode(0.5);
    assert!(
        code > 0.0 && code <= 1.0,
        "HLG code for e=0.5 should be in (0,1]: {code}"
    );
}

#[test]
fn test_spectral_reconstruction_spd_dominant() {
    use oximedia_calibrate::spectral_reconstruction::SpectralPowerDistribution;
    use oximedia_calibrate::spectral_reconstruction::NUM_BANDS;
    let mut values = [0.0f64; NUM_BANDS];
    values[10] = 1.0; // Peak at band 10
    let spd = SpectralPowerDistribution { values };
    assert_eq!(spd.dominant_band(), 10);
}

#[test]
fn test_printer_calibration_ink_limit() {
    use oximedia_calibrate::printer_calibration::enforce_ink_limit;
    let cmyk = [0.5, 0.5, 0.5, 0.5]; // TAC = 2.0
    let limited = enforce_ink_limit(cmyk, 3.0); // max TAC = 3.0, no change needed
    let tac: f64 = limited.iter().sum();
    assert!(
        tac <= 3.0 + 1e-9,
        "TAC should be <= max_tac after enforcement"
    );
}
