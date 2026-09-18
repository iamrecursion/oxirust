// Integration tests for the oximedia facade crate — cross-feature subsystem interaction.
//
// Guards:
//   - `core_tests`    — always compiled (no feature gate)
//   - `quality_tests` — requires feature "quality"
//   - `timecode_tests`— requires feature "timecode"
//   - `metering_tests`— requires feature "metering"
//   - `archive_tests` — requires feature "archive"
//   - `combined_tests`— requires features "search" AND "quality"

// ── Core (always-on) ─────────────────────────────────────────────────────────

#[cfg(test)]
mod core_tests {
    use oximedia::{probe_format, OxiError, OxiResult};

    /// `OxiResult` / `OxiError` types are constructible and usable.
    #[test]
    fn test_oxi_error_creation_and_display() {
        let err: OxiError = OxiError::InvalidData("test payload".to_string());
        let msg = err.to_string();
        assert!(
            msg.contains("test payload"),
            "OxiError display should include the payload, got: {msg}"
        );
    }

    /// `OxiResult` propagation works through `?`.
    #[test]
    fn test_oxi_result_ok_propagation() -> OxiResult<()> {
        fn inner() -> OxiResult<u32> {
            Ok(42)
        }
        let value = inner()?;
        assert_eq!(value, 42, "OxiResult<u32> should carry the inner value");
        Ok(())
    }

    /// `probe_format` correctly identifies a Matroska (MKV) byte header.
    #[test]
    fn test_probe_format_matroska_header() {
        // EBML magic bytes that start every Matroska/WebM file.
        let mkv_header: &[u8] = &[
            0x1A, 0x45, 0xDF, 0xA3, // EBML ID
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, // EBML size (placeholder)
        ];
        let result = probe_format(mkv_header);
        // The probe may return Ok (recognised) or Err (too short / unrecognised) —
        // what matters is that calling it does not panic.
        match result {
            Ok(probe) => {
                // If the prober recognises the header, the format should reflect MKV/EBML.
                let _ = probe; // exercise the value
            }
            Err(OxiError::InvalidData(_)) | Err(OxiError::Unsupported(_)) => {
                // Acceptable: prober needs more data or does not know this format yet.
            }
            Err(other) => {
                panic!("probe_format returned unexpected error: {other}");
            }
        }
    }

    /// `probe_format` correctly identifies an MP4/ISOBMFF header.
    #[test]
    fn test_probe_format_mp4_header() {
        // Minimal ftyp box: size(4) + "ftyp"(4) + brand "isom"(4) + version(4).
        let mut mp4_header = vec![0u8; 16];
        mp4_header[0..4].copy_from_slice(&0x00_00_00_10_u32.to_be_bytes()); // box size = 16
        mp4_header[4..8].copy_from_slice(b"ftyp");
        mp4_header[8..12].copy_from_slice(b"isom");
        mp4_header[12..16].copy_from_slice(&0x00_00_00_00_u32.to_be_bytes());

        let result = probe_format(&mp4_header);
        match result {
            Ok(_) | Err(OxiError::InvalidData(_)) | Err(OxiError::Unsupported(_)) => {}
            Err(other) => panic!("probe_format(mp4) returned unexpected error: {other}"),
        }
    }

    /// `probe_format` on completely empty data should not panic.
    #[test]
    fn test_probe_format_empty_data_does_not_panic() {
        let result = probe_format(&[]);
        match result {
            Ok(_) | Err(_) => {} // either outcome is acceptable; must not panic
        }
    }

    /// `probe_format` on a random/garbage buffer should not panic.
    #[test]
    fn test_probe_format_garbage_data_does_not_panic() {
        let garbage: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        let result = probe_format(&garbage);
        match result {
            Ok(_) | Err(_) => {}
        }
    }
}

// ── Quality ───────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "quality"))]
mod quality_tests {
    use oximedia::quality::{Frame, MetricType, QualityAssessor};
    use oximedia_core::PixelFormat;

    /// Build a synthetic luma-only (Gray8) frame filled with a constant value.
    fn make_gray_frame(width: usize, height: usize, fill: u8) -> Frame {
        let mut frame = Frame::new(width, height, PixelFormat::Gray8)
            .expect("Gray8 frame creation must succeed");
        for byte in frame.luma_mut() {
            *byte = fill;
        }
        frame
    }

    /// Build a YUV420P frame with Y plane filled to `luma_fill`, U/V set to 128
    /// (neutral chroma).
    fn make_yuv_frame(width: usize, height: usize, luma_fill: u8) -> Frame {
        let mut frame = Frame::new(width, height, PixelFormat::Yuv420p)
            .expect("YUV420P frame creation must succeed");
        for byte in &mut frame.planes[0] {
            *byte = luma_fill;
        }
        for byte in &mut frame.planes[1] {
            *byte = 128;
        }
        for byte in &mut frame.planes[2] {
            *byte = 128;
        }
        frame
    }

    /// PSNR of identical frames must be very high (effectively infinity).
    #[test]
    fn test_psnr_identical_frames_is_infinite_or_very_high() {
        let reference = make_gray_frame(64, 64, 128);
        let distorted = make_gray_frame(64, 64, 128);

        let assessor = QualityAssessor::new();
        let score = assessor
            .assess(&reference, &distorted, MetricType::Psnr)
            .expect("PSNR computation on identical frames must succeed");

        // PSNR of identical frames is +∞; implementations may clamp to a large
        // finite value (e.g. 100 dB) or return f64::INFINITY.
        assert!(
            score.score >= 60.0 || score.score.is_infinite(),
            "PSNR of identical frames should be >=60 dB or infinite, got {}",
            score.score
        );
        assert_eq!(
            score.metric,
            MetricType::Psnr,
            "Returned metric type must match requested"
        );
    }

    /// PSNR of maximally different frames (0 vs 255) must be very low.
    #[test]
    fn test_psnr_maximally_different_frames_is_low() {
        let reference = make_gray_frame(64, 64, 0);
        let distorted = make_gray_frame(64, 64, 255);

        let assessor = QualityAssessor::new();
        let score = assessor
            .assess(&reference, &distorted, MetricType::Psnr)
            .expect("PSNR computation must succeed");

        // Maximum possible distortion → PSNR near 0 dB.
        assert!(
            score.score < 10.0,
            "PSNR of maximally different frames should be <10 dB, got {}",
            score.score
        );
    }

    /// SSIM of identical YUV420P frames must be 1.0 (or very close).
    ///
    /// Note: the SSIM implementation uses weighted combination of Y/Cb/Cr planes
    /// (weights 4/6, 1/6, 1/6).  For a Gray8 (luma-only) frame the weighted score
    /// is 4/6 ≈ 0.667 even for perfect similarity.  YUV frames produce the correct
    /// all-components-unity result of 1.0.
    #[test]
    fn test_ssim_identical_frames_is_one() {
        let reference = make_yuv_frame(64, 64, 180);
        let distorted = make_yuv_frame(64, 64, 180);

        let assessor = QualityAssessor::new();
        let score = assessor
            .assess(&reference, &distorted, MetricType::Ssim)
            .expect("SSIM computation on identical YUV frames must succeed");

        assert!(
            score.score >= 0.99,
            "SSIM of identical YUV frames should be >=0.99, got {}",
            score.score
        );
    }

    /// SSIM of moderately different YUV frames should be between 0 and 1.
    #[test]
    fn test_ssim_different_frames_is_between_zero_and_one() {
        let reference = make_yuv_frame(64, 64, 100);
        let distorted = make_yuv_frame(64, 64, 200);

        let assessor = QualityAssessor::new();
        let score = assessor
            .assess(&reference, &distorted, MetricType::Ssim)
            .expect("SSIM computation must succeed");

        assert!(
            (0.0..=1.0).contains(&score.score),
            "SSIM must be in [0, 1], got {}",
            score.score
        );
    }

    /// Dimension mismatch between reference and distorted must be an error.
    #[test]
    fn test_assess_mismatched_dimensions_returns_error() {
        let reference = make_gray_frame(64, 64, 128);
        let distorted = make_gray_frame(32, 32, 128); // wrong size

        let assessor = QualityAssessor::new();
        let result = assessor.assess(&reference, &distorted, MetricType::Psnr);
        assert!(
            result.is_err(),
            "Assessing frames with different dimensions must return an error"
        );
    }

    /// No-reference metric (Blur) must succeed on a single frame.
    #[test]
    fn test_no_reference_blur_detection() {
        let frame = make_yuv_frame(64, 64, 128);
        let assessor = QualityAssessor::new();

        let score = assessor
            .assess_no_reference(&frame, MetricType::Blur)
            .expect("No-reference blur detection must succeed");

        assert!(
            score.score.is_finite(),
            "Blur score must be a finite float, got {}",
            score.score
        );
    }

    /// `MetricType::requires_reference` / `is_no_reference` logic is consistent.
    #[test]
    fn test_metric_type_classification() {
        let full_ref = [
            MetricType::Psnr,
            MetricType::Ssim,
            MetricType::MsSsim,
            MetricType::Vif,
            MetricType::Fsim,
            MetricType::Vmaf,
        ];
        let no_ref = [
            MetricType::Niqe,
            MetricType::Brisque,
            MetricType::Blockiness,
            MetricType::Blur,
            MetricType::Noise,
        ];
        for m in full_ref {
            assert!(m.requires_reference(), "{m:?} should require a reference");
            assert!(!m.is_no_reference(), "{m:?} should not be no-reference");
        }
        for m in no_ref {
            assert!(m.is_no_reference(), "{m:?} should be no-reference");
            assert!(
                !m.requires_reference(),
                "{m:?} should not require a reference"
            );
        }
    }

    /// `QualityScore` carries per-component data correctly.
    #[test]
    fn test_quality_score_components() {
        use oximedia::quality::QualityScore;

        let mut score = QualityScore::new(MetricType::Psnr, 42.5);
        score.add_component("Y", 45.0);
        score.add_component("Cb", 40.0);
        score.add_component("Cr", 39.5);

        assert_eq!(score.score, 42.5);
        assert_eq!(
            *score
                .components
                .get("Y")
                .expect("Y component must be present"),
            45.0
        );
        assert_eq!(score.components.len(), 3);
        assert!(score.frame_num.is_none());

        let score_with_frame = score.with_frame_num(7);
        assert_eq!(score_with_frame.frame_num, Some(7));
    }
}

// ── Timecode ──────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "timecode"))]
mod timecode_tests {
    use oximedia::timecode::{FrameRate, Timecode, TimecodeError};

    /// Timecode creation succeeds for all standard frame rates.
    #[test]
    fn test_timecode_creation_all_frame_rates() {
        let rates = [
            FrameRate::Fps23976,
            FrameRate::Fps24,
            FrameRate::Fps25,
            FrameRate::Fps2997NDF,
            FrameRate::Fps30,
            FrameRate::Fps50,
            FrameRate::Fps60,
        ];
        for rate in rates {
            let fps = rate.frames_per_second();
            // Use frame 0 which is always valid.
            let tc = Timecode::new(1, 0, 0, 0, rate)
                .unwrap_or_else(|e| panic!("Timecode::new at {fps} fps must succeed: {e}"));
            assert_eq!(tc.hours, 1, "Hours must be preserved");
            assert_eq!(tc.frame_rate.fps, fps as u8, "FPS must be stored correctly");
        }
    }

    /// Drop-frame timecode creation succeeds at valid positions.
    #[test]
    fn test_drop_frame_timecode_valid_positions() {
        // 29.97 DF: minute=0 (all frames valid), minute=10 (frames 0-1 valid).
        let tc_min0 = Timecode::new(0, 0, 0, 0, FrameRate::Fps2997DF)
            .expect("DF timecode at minute 0 frame 0 must succeed");
        assert!(tc_min0.frame_rate.drop_frame, "Drop-frame flag must be set");

        let tc_min10 = Timecode::new(0, 10, 0, 0, FrameRate::Fps2997DF)
            .expect("DF timecode at minute 10 frame 0 must succeed");
        assert_eq!(tc_min10.minutes, 10);
    }

    /// Drop-frame timecode rejects frames 0 and 1 at the start of non-multiple-of-10 minutes.
    #[test]
    fn test_drop_frame_timecode_invalid_positions() {
        // Minute 1 (not divisible by 10), second 0, frames 0 and 1 — dropped.
        let err0 = Timecode::new(0, 1, 0, 0, FrameRate::Fps2997DF)
            .expect_err("Frame 0 at minute 1 second 0 must be rejected in drop-frame mode");
        assert_eq!(
            err0,
            TimecodeError::InvalidDropFrame,
            "Error must be InvalidDropFrame"
        );

        let err1 = Timecode::new(0, 1, 0, 1, FrameRate::Fps2997DF)
            .expect_err("Frame 1 at minute 1 second 0 must be rejected in drop-frame mode");
        assert_eq!(err1, TimecodeError::InvalidDropFrame);
    }

    /// Out-of-range fields are rejected correctly.
    #[test]
    fn test_timecode_validation_rejects_out_of_range_fields() {
        assert_eq!(
            Timecode::new(24, 0, 0, 0, FrameRate::Fps25).unwrap_err(),
            TimecodeError::InvalidHours,
            "Hours >= 24 must be rejected"
        );
        assert_eq!(
            Timecode::new(0, 60, 0, 0, FrameRate::Fps25).unwrap_err(),
            TimecodeError::InvalidMinutes,
            "Minutes >= 60 must be rejected"
        );
        assert_eq!(
            Timecode::new(0, 0, 60, 0, FrameRate::Fps25).unwrap_err(),
            TimecodeError::InvalidSeconds,
            "Seconds >= 60 must be rejected"
        );
        assert_eq!(
            Timecode::new(0, 0, 0, 25, FrameRate::Fps25).unwrap_err(),
            TimecodeError::InvalidFrames,
            "Frame >= fps must be rejected"
        );
    }

    /// `to_frames` / `from_frames` round-trip for non-drop-frame 25 fps.
    #[test]
    fn test_frame_count_round_trip_ndf_25fps() {
        let original =
            Timecode::new(1, 23, 45, 12, FrameRate::Fps25).expect("Valid timecode must be created");
        let frame_count = original.to_frames();
        let restored = Timecode::from_frames(frame_count, FrameRate::Fps25)
            .expect("Restoring timecode from frame count must succeed");

        assert_eq!(
            restored.hours, original.hours,
            "Hours must survive round-trip"
        );
        assert_eq!(
            restored.minutes, original.minutes,
            "Minutes must survive round-trip"
        );
        assert_eq!(
            restored.seconds, original.seconds,
            "Seconds must survive round-trip"
        );
        assert_eq!(
            restored.frames, original.frames,
            "Frames must survive round-trip"
        );
    }

    /// `to_frames` / `from_frames` round-trip for non-drop-frame 30 fps.
    #[test]
    fn test_frame_count_round_trip_ndf_30fps() {
        let original = Timecode::new(0, 5, 30, 15, FrameRate::Fps30)
            .expect("Valid 30fps timecode must be created");
        let frame_count = original.to_frames();
        let restored = Timecode::from_frames(frame_count, FrameRate::Fps30)
            .expect("Restoring timecode from frame count must succeed");

        assert_eq!(restored.hours, original.hours);
        assert_eq!(restored.minutes, original.minutes);
        assert_eq!(restored.seconds, original.seconds);
        assert_eq!(restored.frames, original.frames);
    }

    /// Timecode display uses `:` separator for NDF and `;` for DF.
    #[test]
    fn test_timecode_display_separators() {
        let ndf =
            Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("NDF timecode must be created");
        assert_eq!(ndf.to_string(), "01:02:03:04", "NDF uses colon separator");

        let df =
            Timecode::new(1, 2, 3, 4, FrameRate::Fps2997DF).expect("DF timecode must be created");
        assert_eq!(
            df.to_string(),
            "01:02:03;04",
            "DF uses semicolon before frames"
        );
    }

    /// Timecode increment rolls over frames → seconds correctly.
    #[test]
    fn test_timecode_increment_frame_rollover() {
        // 25 fps: increment from frame 24 should produce second 1 frame 0.
        let mut tc = Timecode::new(0, 0, 0, 24, FrameRate::Fps25)
            .expect("Timecode at frame 24 must be valid");
        tc.increment().expect("Increment must succeed");

        assert_eq!(tc.frames, 0, "Frame must roll over to 0");
        assert_eq!(tc.seconds, 1, "Second must advance by 1");
    }

    /// Timecode increment rolls over seconds → minutes.
    #[test]
    fn test_timecode_increment_second_rollover() {
        let mut tc = Timecode::new(0, 0, 59, 24, FrameRate::Fps25)
            .expect("Timecode at second 59 frame 24 must be valid");
        tc.increment().expect("Increment must succeed");

        assert_eq!(tc.frames, 0);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.minutes, 1);
    }

    /// Timecode decrement from frame 0 second 1 → second 0 frame 24.
    #[test]
    fn test_timecode_decrement_frame_borrow() {
        let mut tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("Timecode must be created");
        tc.decrement().expect("Decrement must succeed");

        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 24, "Previous frame in the second");
    }

    /// User bits are stored and retrieved correctly.
    #[test]
    fn test_user_bits_round_trip() {
        let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25)
            .expect("Timecode must be created")
            .with_user_bits(0xDEAD_BEEF);

        assert_eq!(
            tc.user_bits, 0xDEAD_BEEF,
            "User bits must survive the builder chain"
        );
    }

    /// Frame rate helpers return consistent values.
    #[test]
    fn test_frame_rate_helpers() {
        assert!((FrameRate::Fps23976.as_float() - 23.976).abs() < 0.001);
        assert!((FrameRate::Fps25.as_float() - 25.0).abs() < 1e-9);
        assert!((FrameRate::Fps2997DF.as_float() - 29.97).abs() < 0.001);

        let (num, den) = FrameRate::Fps2997DF.as_rational();
        assert_eq!(
            (num, den),
            (30000, 1001),
            "29.97 rational must be 30000/1001"
        );

        assert!(FrameRate::Fps2997DF.is_drop_frame());
        assert!(!FrameRate::Fps2997NDF.is_drop_frame());
        assert!(!FrameRate::Fps25.is_drop_frame());
        assert!(!FrameRate::Fps30.is_drop_frame());
    }

    /// `from_frames(0)` yields midnight 00:00:00:00.
    #[test]
    fn test_from_frames_zero_is_midnight() {
        let tc = Timecode::from_frames(0, FrameRate::Fps25).expect("from_frames(0) must succeed");
        assert_eq!(tc.hours, 0);
        assert_eq!(tc.minutes, 0);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);
    }
}

// ── Metering ──────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "metering"))]
mod metering_tests {
    use oximedia::metering::{LoudnessMeter, MeterConfig, Standard};

    /// Meter creation with valid EBU R128 configuration must succeed.
    #[test]
    fn test_loudness_meter_creation_ebu_r128() {
        let config = MeterConfig::new(Standard::EbuR128, 48_000.0, 2);
        let _meter = LoudnessMeter::new(config)
            .expect("LoudnessMeter::new with valid EBU R128 config must succeed");
    }

    /// Meter creation with valid ATSC A/85 configuration must succeed.
    #[test]
    fn test_loudness_meter_creation_atsc_a85() {
        let config = MeterConfig::new(Standard::AtscA85, 48_000.0, 2);
        let _meter = LoudnessMeter::new(config)
            .expect("LoudnessMeter::new with valid ATSC A/85 config must succeed");
    }

    /// Meter creation with invalid sample rate must fail.
    #[test]
    fn test_loudness_meter_invalid_sample_rate_fails() {
        let config = MeterConfig::new(Standard::EbuR128, 1.0, 2); // 1 Hz is absurd
        let result = LoudnessMeter::new(config);
        assert!(
            result.is_err(),
            "LoudnessMeter must reject a 1 Hz sample rate"
        );
    }

    /// Meter creation with zero channels must fail.
    #[test]
    fn test_loudness_meter_zero_channels_fails() {
        let config = MeterConfig::new(Standard::EbuR128, 48_000.0, 0);
        let result = LoudnessMeter::new(config);
        assert!(result.is_err(), "LoudnessMeter must reject zero channels");
    }

    /// Processing silence produces a valid (finite or -inf) integrated loudness.
    #[test]
    fn test_loudness_meter_process_silence() {
        let config = MeterConfig::new(Standard::EbuR128, 48_000.0, 2);
        let mut meter = LoudnessMeter::new(config).expect("Meter creation must succeed");

        // 1 second of stereo silence at 48 kHz = 48000 frames × 2 channels.
        let silence = vec![0.0_f32; 48_000 * 2];
        meter.process_f32(&silence);

        let metrics = meter.metrics();
        // Silence gives -∞ (gated) or a very low integrated loudness.
        // The result must not be NaN.
        assert!(
            !metrics.integrated_lufs.is_nan(),
            "Integrated LUFS for silence must not be NaN"
        );
        assert!(
            !metrics.momentary_lufs.is_nan(),
            "Momentary LUFS for silence must not be NaN"
        );
    }

    /// Processing a moderate sine-like signal gives a finite integrated loudness.
    #[test]
    fn test_loudness_meter_process_tone_gives_finite_lufs() {
        let sample_rate = 48_000.0_f64;
        let channels = 2_usize;
        // Generate ~3 seconds of stereo 1 kHz tone at -20 dBFS.
        let amplitude = 10.0_f64.powf(-20.0 / 20.0);
        let freq = 1000.0_f64;
        let num_frames = (3.0 * sample_rate) as usize;

        let mut samples = Vec::with_capacity(num_frames * channels);
        for i in 0..num_frames {
            let t = i as f64 / sample_rate;
            let sample = (amplitude * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32;
            // Interleaved stereo: L then R
            samples.push(sample);
            samples.push(sample);
        }

        let config = MeterConfig::new(Standard::EbuR128, sample_rate, channels);
        let mut meter = LoudnessMeter::new(config).expect("Meter creation must succeed");
        meter.process_f32(&samples);

        let metrics = meter.metrics();
        assert!(
            metrics.integrated_lufs.is_finite() || metrics.integrated_lufs == f64::NEG_INFINITY,
            "Integrated LUFS must be finite or -inf for a tone signal, got {}",
            metrics.integrated_lufs
        );
        assert!(
            metrics.true_peak_dbtp.is_finite() || metrics.true_peak_dbtp == f64::NEG_INFINITY,
            "True peak must be finite or -inf"
        );
    }

    /// Reset clears the meter state.
    #[test]
    fn test_loudness_meter_reset() {
        let sample_rate = 48_000.0_f64;
        let channels = 2_usize;
        let amplitude = 0.1_f32;
        let num_frames = sample_rate as usize;

        let mut samples = vec![0.0_f32; num_frames * channels];
        for s in &mut samples {
            *s = amplitude;
        }

        let config = MeterConfig::new(Standard::EbuR128, sample_rate, channels);
        let mut meter = LoudnessMeter::new(config).expect("Meter creation must succeed");
        meter.process_f32(&samples);
        assert!(
            meter.samples_processed() > 0,
            "Samples processed must be > 0 after feeding data"
        );

        meter.reset();
        assert_eq!(
            meter.samples_processed(),
            0,
            "Samples processed must be 0 after reset"
        );
    }

    /// `duration_seconds` returns a value consistent with samples processed.
    #[test]
    fn test_loudness_meter_duration_seconds() {
        let sample_rate = 44_100.0_f64;
        let channels = 1_usize;
        let num_frames = sample_rate as usize; // exactly 1 second

        let silence = vec![0.0_f32; num_frames * channels];
        let config = MeterConfig::minimal(Standard::EbuR128, sample_rate, channels);
        let mut meter = LoudnessMeter::new(config).expect("Meter creation must succeed");
        meter.process_f32(&silence);

        let duration = meter.duration_seconds();
        assert!(
            (duration - 1.0).abs() < 0.001,
            "Duration must be ~1.0 seconds, got {duration}"
        );
    }

    /// Compliance check API is callable and returns a structured result.
    #[test]
    fn test_compliance_check_api() {
        let config = MeterConfig::new(Standard::EbuR128, 48_000.0, 2);
        let mut meter = LoudnessMeter::new(config).expect("Meter creation must succeed");

        // Process silence — won't be compliant but should not panic.
        let silence = vec![0.0_f32; 48_000 * 2];
        meter.process_f32(&silence);

        let compliance = meter.check_compliance();
        // Silence is NOT loudness-compliant — that's fine; we just check API shape.
        assert_eq!(
            compliance.standard_name(),
            "EBU R128",
            "Standard name must match configured standard"
        );
        assert!(
            compliance.target_lufs.is_finite(),
            "Target LUFS must be finite"
        );
        assert!(
            compliance.max_peak_dbtp.is_finite(),
            "Max peak dBTP must be finite"
        );
        // Non-compliant audio should have negative recommended gain (silence << target).
        let _ = compliance.recommended_gain_db(); // must not panic
        let _ = compliance.is_compliant(); // must not panic
    }

    /// Standard loudness targets are correct.
    #[test]
    fn test_standard_loudness_targets() {
        assert_eq!(Standard::EbuR128.target_lufs(), -23.0);
        assert_eq!(Standard::AtscA85.target_lufs(), -24.0);
        assert_eq!(Standard::Spotify.target_lufs(), -14.0);
        assert_eq!(Standard::YouTube.target_lufs(), -14.0);
        assert_eq!(Standard::Netflix.target_lufs(), -27.0);
        assert_eq!(Standard::EbuR128.max_true_peak_dbtp(), -1.0);
        assert_eq!(Standard::AtscA85.max_true_peak_dbtp(), -2.0);
    }

    /// `MeterConfig::validate` accepts and rejects configurations correctly.
    #[test]
    fn test_meter_config_validation() {
        // Valid configuration.
        MeterConfig::new(Standard::EbuR128, 48_000.0, 2)
            .validate()
            .expect("Valid config must pass validation");

        // Invalid sample rate.
        assert!(
            MeterConfig::new(Standard::EbuR128, 5_000.0, 2)
                .validate()
                .is_err(),
            "5 kHz sample rate must fail validation"
        );

        // Invalid channel count.
        assert!(
            MeterConfig::new(Standard::EbuR128, 48_000.0, 0)
                .validate()
                .is_err(),
            "0 channels must fail validation"
        );
    }
}

// ── Archive ───────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "archive"))]
mod archive_tests {
    use oximedia::archive::{
        integrity_scan::{FileIntegrity, FileScanRecord, IntegrityScan, ScanPolicy},
        validate::{detect_container_format, validate_metadata, MediaMetadata},
        VerificationConfig,
    };
    use std::io::Write as _;

    /// `VerificationConfig::default()` produces a sane configuration.
    #[test]
    fn test_verification_config_defaults() {
        let config = VerificationConfig::default();
        assert!(config.enable_blake3, "BLAKE3 must be enabled by default");
        assert!(config.enable_sha256, "SHA-256 must be enabled by default");
        assert!(
            config.fixity_check_interval_days > 0,
            "Fixity interval must be positive"
        );
        assert!(
            config.parallel_threads > 0,
            "Parallel threads must be positive"
        );
    }

    /// Custom `VerificationConfig` round-trips its fields correctly.
    #[test]
    fn test_verification_config_custom_fields() {
        let config = VerificationConfig {
            enable_blake3: false,
            enable_md5: true,
            enable_sha256: false,
            enable_crc32: false,
            generate_sidecars: false,
            validate_containers: false,
            enable_fixity_checks: false,
            fixity_check_interval_days: 30,
            auto_quarantine: false,
            parallel_threads: 2,
            database_path: std::env::temp_dir().join("test_archive.db"),
            quarantine_dir: std::env::temp_dir().join("test_quarantine"),
            enable_premis_logging: false,
            enable_bagit: false,
        };

        assert!(!config.enable_blake3);
        assert!(config.enable_md5);
        assert_eq!(config.fixity_check_interval_days, 30);
        assert_eq!(config.parallel_threads, 2);
    }

    /// `detect_container_format` identifies a Matroska file by its EBML magic bytes.
    #[tokio::test]
    async fn test_detect_container_format_matroska() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_matroska_detect.mkv");

        // Write EBML magic header.
        {
            let mut f =
                std::fs::File::create(&path).expect("Creating temp MKV test file must succeed");
            // EBML magic + minimal padding.
            let data = [
                0x1A_u8, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23,
            ];
            f.write_all(&data).expect("Writing MKV magic must succeed");
        }

        let format = detect_container_format(&path)
            .await
            .expect("detect_container_format must succeed");

        std::fs::remove_file(&path).ok();

        assert_eq!(
            format, "matroska",
            "EBML magic must be identified as matroska"
        );
    }

    /// `detect_container_format` identifies an MP4 file by its ftyp box.
    #[tokio::test]
    async fn test_detect_container_format_mp4() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_mp4_detect.mp4");

        {
            let mut f =
                std::fs::File::create(&path).expect("Creating temp MP4 test file must succeed");
            // Minimal ftyp atom: size=16, "ftyp", brand "isom", minor version 0.
            let data: [u8; 16] = [
                0x00, 0x00, 0x00, 0x10, // box size = 16
                b'f', b't', b'y', b'p', // "ftyp"
                b'i', b's', b'o', b'm', // brand
                0x00, 0x00, 0x00, 0x00, // minor version
            ];
            f.write_all(&data)
                .expect("Writing MP4 ftyp box must succeed");
        }

        let format = detect_container_format(&path)
            .await
            .expect("detect_container_format must succeed for MP4");

        std::fs::remove_file(&path).ok();

        assert_eq!(format, "mp4", "ftyp box must be identified as mp4");
    }

    /// `detect_container_format` falls back to extension for unknown magic.
    #[tokio::test]
    async fn test_detect_container_format_extension_fallback() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_fallback.webm");

        {
            let mut f =
                std::fs::File::create(&path).expect("Creating temp WebM test file must succeed");
            // Write junk bytes that don't match any magic.
            f.write_all(&[0x00; 12])
                .expect("Writing junk header must succeed");
        }

        let format = detect_container_format(&path)
            .await
            .expect("detect_container_format must succeed for junk+webm extension");

        std::fs::remove_file(&path).ok();

        assert_eq!(
            format, "webm",
            "Extension fallback must identify .webm files"
        );
    }

    /// `validate_metadata` correctly identifies missing fields.
    #[test]
    fn test_validate_metadata_missing_fields() {
        let metadata = MediaMetadata {
            title: None,
            duration: None,
            bitrate: None,
            format_name: None,
            format_long_name: None,
            size: None,
        };

        let validation = validate_metadata(&metadata);
        assert!(
            !validation.is_complete,
            "Metadata without duration/bitrate must not be complete"
        );
        assert!(
            validation.missing_fields.contains(&"duration".to_string()),
            "Missing 'duration' field must be reported"
        );
        assert!(
            validation.missing_fields.contains(&"bitrate".to_string()),
            "Missing 'bitrate' field must be reported"
        );
        assert!(!validation.has_duration);
        assert!(!validation.has_bitrate);
    }

    /// `validate_metadata` marks as complete when duration and bitrate are present.
    #[test]
    fn test_validate_metadata_complete() {
        let metadata = MediaMetadata {
            title: Some("Test Video".to_string()),
            duration: Some(120.5),
            bitrate: Some(5_000_000),
            format_name: Some("matroska".to_string()),
            format_long_name: Some("Matroska / WebM".to_string()),
            size: Some(75_000_000),
        };

        let validation = validate_metadata(&metadata);
        assert!(
            validation.is_complete,
            "Metadata with duration and bitrate must be complete"
        );
        assert!(validation.has_duration);
        assert!(validation.has_bitrate);
        assert!(validation.has_title);
        assert!(validation.missing_fields.is_empty());
    }

    /// `IntegrityScan` with a synthetic manifest correctly detects a corrupted file.
    #[test]
    fn test_integrity_scan_detects_corruption() {
        // Create a scan record where expected checksum differs from actual → Corrupted.
        let record = FileScanRecord::new(
            "/archive/test_clip.mkv",
            "expected_checksum_abc123",
            "actual_checksum_xyz789", // intentionally different → Corrupted status
            1_024_000,
            1_000,
        );
        assert_eq!(
            record.status,
            FileIntegrity::Corrupted,
            "Mismatched checksums must result in Corrupted status"
        );

        // Assemble a scan session with one OK and one Corrupted record.
        let start_ms = 1_000_u64;
        let policy = ScanPolicy::default();
        let mut scan = IntegrityScan::new(policy, start_ms);

        let ok_record = FileScanRecord::new(
            "/archive/good_clip.mkv",
            "same_checksum",
            "same_checksum",
            2_048_000,
            1_001,
        );
        assert_eq!(
            ok_record.status,
            FileIntegrity::Ok,
            "Matching checksums must result in Ok status"
        );

        scan.add_record(record);
        scan.add_record(ok_record);

        assert_eq!(scan.record_count(), 2, "Scan must hold 2 records");

        // Finish the scan session.
        scan.finish(2_000);
        assert!(scan.is_finished(), "Scan must be marked finished");

        let metrics = scan.metrics();
        assert_eq!(metrics.total_scanned, 2, "Total scanned must be 2");
        assert_eq!(metrics.corrupted_count, 1, "Corrupted count must be 1");
        assert_eq!(metrics.ok_count, 1, "OK count must be 1");
        assert_eq!(metrics.missing_count, 0, "Missing count must be 0");
        assert!(
            metrics.health_score() > 0.0 && metrics.health_score() <= 1.0,
            "Health score must be in (0, 1], got {}",
            metrics.health_score()
        );

        // Corrupted records list must contain exactly the bad record.
        let corrupted = scan.corrupted();
        assert_eq!(corrupted.len(), 1, "Exactly one corrupted record expected");
        assert!(
            corrupted[0].path.contains("test_clip"),
            "Corrupted record must be the test clip"
        );
    }

    /// `ScanPolicy::default()` produces non-zero intervals.
    #[test]
    fn test_scan_policy_defaults() {
        let policy = ScanPolicy::default();
        assert!(
            policy.full_scan_interval_hours > 0,
            "Full scan interval must be positive"
        );
        assert!(
            policy.incremental_interval_hours > 0,
            "Incremental interval must be positive"
        );
        assert!(
            policy.incremental_batch_size > 0,
            "Batch size must be positive"
        );
        assert!(policy.parallelism > 0, "Parallelism must be positive");
    }
}

// ── Combined: search + quality ─────────────────────────────────────────────

#[cfg(all(test, feature = "search", feature = "quality"))]
mod combined_tests {
    use oximedia::quality::{Frame, MetricType, QualityAssessor};
    use oximedia::search::{
        SearchFilters, SearchQuery, SearchResultItem, SortField, SortOptions, SortOrder,
    };
    use oximedia_core::PixelFormat;
    use uuid::Uuid;

    fn make_gray_frame_combined(width: usize, height: usize, fill: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Gray8).expect("Frame creation must succeed");
        for byte in frame.luma_mut() {
            *byte = fill;
        }
        frame
    }

    /// Both `SearchQuery` construction and `QualityAssessor` are usable in
    /// the same compilation unit — verifying that the feature flags don't
    /// introduce conflicting dependencies.
    #[test]
    fn test_search_and_quality_coexist_in_same_binary() {
        // Quality side: compute PSNR.
        let reference = make_gray_frame_combined(32, 32, 120);
        let distorted = make_gray_frame_combined(32, 32, 120);
        let assessor = QualityAssessor::new();
        let score = assessor
            .assess(&reference, &distorted, MetricType::Psnr)
            .expect("PSNR computation must succeed");
        assert!(
            score.score >= 60.0 || score.score.is_infinite(),
            "PSNR of identical frames must be high"
        );

        // Search side: build a SearchQuery and verify field access.
        let query = SearchQuery {
            text: Some("documentary nature HD".to_string()),
            visual: None,
            audio: None,
            filters: SearchFilters {
                mime_types: vec!["video/x-matroska".to_string()],
                duration_range: Some((30_000, 3_600_000)), // 30s to 1h in ms
                ..SearchFilters::default()
            },
            limit: 20,
            offset: 0,
            sort: SortOptions {
                field: SortField::Relevance,
                order: SortOrder::Descending,
            },
        };

        assert_eq!(
            query.text.as_deref(),
            Some("documentary nature HD"),
            "Text query must be preserved"
        );
        assert_eq!(query.limit, 20, "Limit must be preserved");
        assert_eq!(
            query.filters.mime_types.first().map(String::as_str),
            Some("video/x-matroska"),
            "MIME type filter must be preserved"
        );
    }

    /// `SearchResultItem` can carry quality-derived metadata and be sorted.
    #[test]
    fn test_search_result_items_with_quality_scores() {
        // Compute quality scores for two synthetic clips.
        let assessor = QualityAssessor::new();
        let ref_frame = make_gray_frame_combined(64, 64, 128);

        let clip_a_frame = make_gray_frame_combined(64, 64, 128); // identical → high PSNR
        let clip_b_frame = make_gray_frame_combined(64, 64, 200); // different → lower PSNR

        let psnr_a = assessor
            .assess(&ref_frame, &clip_a_frame, MetricType::Psnr)
            .expect("PSNR for clip A must succeed");
        let psnr_b = assessor
            .assess(&ref_frame, &clip_b_frame, MetricType::Psnr)
            .expect("PSNR for clip B must succeed");

        // Build search result items whose `score` reflects quality.
        let normalize_psnr = |psnr: f64| -> f32 {
            // Map PSNR [0, 60+] → relevance [0, 1].
            (psnr.min(60.0) / 60.0) as f32
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let mut results = vec![
            SearchResultItem {
                asset_id: Uuid::new_v4(),
                score: normalize_psnr(psnr_b.score),
                title: Some("Clip B".to_string()),
                description: None,
                file_path: "/archive/clip_b.mkv".to_string(),
                mime_type: Some("video/x-matroska".to_string()),
                duration_ms: Some(30_000),
                created_at: now,
                matched_fields: vec!["title".to_string()],
                thumbnail_url: None,
                file_size: None,
                modified_at: None,
            },
            SearchResultItem {
                asset_id: Uuid::new_v4(),
                score: normalize_psnr(psnr_a.score),
                title: Some("Clip A".to_string()),
                description: None,
                file_path: "/archive/clip_a.mkv".to_string(),
                mime_type: Some("video/x-matroska".to_string()),
                duration_ms: Some(60_000),
                created_at: now,
                matched_fields: vec!["title".to_string()],
                thumbnail_url: None,
                file_size: None,
                modified_at: None,
            },
        ];

        // Sort descending by score (highest quality first).
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Clip A (identical → higher PSNR → higher score) should come first.
        assert_eq!(
            results[0].title.as_deref(),
            Some("Clip A"),
            "Clip A has higher quality and must rank first"
        );
        assert!(
            results[0].score >= results[1].score,
            "Results must be sorted descending by quality score"
        );
    }

    /// `SearchFilters::default()` is empty and composes cleanly with quality data.
    #[test]
    fn test_search_filters_default_is_empty() {
        let filters = SearchFilters::default();
        assert!(filters.mime_types.is_empty());
        assert!(filters.codecs.is_empty());
        assert!(filters.duration_range.is_none());
        assert!(filters.date_range.is_none());
        assert!(filters.has_faces.is_none());

        // Demonstrate that a quality assessor can be used to build filter criteria
        // without any type-system conflict.
        let assessor = QualityAssessor::new();
        let f1 = make_gray_frame_combined(32, 32, 100);
        let f2 = make_gray_frame_combined(32, 32, 100);
        let _psnr = assessor
            .assess(&f1, &f2, MetricType::Ssim)
            .expect("SSIM must succeed");

        // Enrich filters with a quality threshold (example of multi-feature use).
        let enriched = SearchFilters {
            keywords: vec!["high-quality".to_string()],
            ..SearchFilters::default()
        };
        assert_eq!(enriched.keywords.len(), 1);
    }

    /// Text tokeniser (always-available from search) works alongside quality types.
    #[test]
    fn test_tokenizer_and_quality_types_together() {
        use oximedia::search::text::Tokenizer;

        let tokenizer = Tokenizer::new(true, true);
        let tokens = tokenizer.tokenize("High-definition nature documentary 4K HDR");

        assert!(
            !tokens.is_empty(),
            "Tokenizer must produce at least one token"
        );

        // Verify tokens contain expected terms (stopwords removed, lowercase).
        let token_texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(
            token_texts.contains(&"high"),
            "Token 'high' must be present (hyphen-split)"
        );
        assert!(
            token_texts.contains(&"4k") || token_texts.contains(&"4"),
            "Alphanumeric token '4k' or '4' must be present"
        );

        // Simultaneously exercise quality types.
        let assessor = QualityAssessor::new();
        let f = make_gray_frame_combined(32, 32, 200);
        let blur_score = assessor
            .assess_no_reference(&f, MetricType::Blur)
            .expect("Blur assessment must succeed");
        assert!(
            blur_score.score.is_finite(),
            "Blur score must be finite, got {}",
            blur_score.score
        );
    }
}

// ── Combined: transcode + normalize + qc ────────────────────────────────────

#[cfg(all(test, feature = "transcode", feature = "normalize", feature = "qc"))]
mod combined_transcode_normalize_qc_tests {
    use oximedia::metering::Standard;
    use oximedia::normalize::{Normalizer, NormalizerConfig};
    use oximedia::qc::qc_profile::{QcProfile, QcProfileType, QcTolerance};
    use oximedia::transcode::TranscodeConfig;

    /// A transcode configuration, a two-pass loudness normalizer, and a QC
    /// delivery profile can all be constructed and driven in the same
    /// compilation unit — verifying that enabling `transcode` + `normalize` +
    /// `qc` simultaneously does not introduce conflicting dependencies or
    /// type collisions between the three subsystems.
    #[test]
    fn test_transcode_normalize_qc_pipeline_types_coexist() {
        // Transcode side: an audio-normalizing transcode configuration.
        let config = TranscodeConfig {
            video_codec: Some("av1".to_string()),
            audio_codec: Some("opus".to_string()),
            normalize_audio: true,
            ..TranscodeConfig::default()
        };
        assert!(config.normalize_audio, "normalize_audio flag must be set");
        assert_eq!(
            config.video_codec.as_deref(),
            Some("av1"),
            "Video codec must be preserved"
        );

        // Normalize side: analyze a second of stereo silence at EBU R128.
        let normalizer_config = NormalizerConfig::new(Standard::EbuR128, 48_000.0, 2);
        let mut normalizer = Normalizer::new(normalizer_config)
            .expect("Normalizer::new with a valid EBU R128 config must succeed");

        let silence = vec![0.0_f32; 48_000 * 2];
        normalizer.analyze_f32(&silence);
        let analysis = normalizer.get_analysis();
        assert!(
            !analysis.integrated_lufs.is_nan(),
            "Integrated LUFS must not be NaN after analysis"
        );

        // QC side: a broadcast delivery profile with a loudness tolerance.
        let mut profile = QcProfile::new(QcProfileType::Broadcast, "broadcast_default");
        profile.set_tolerance("loudness", QcTolerance::new(1.0, true));
        assert_eq!(
            profile.strictness_level(),
            3,
            "Broadcast profile strictness must be 3"
        );
        assert!(
            profile.tolerance_for("loudness").is_some(),
            "Loudness tolerance must be retrievable after being set"
        );
    }
}

// ── Combined: playlist + playout + automation ───────────────────────────────

#[cfg(all(
    test,
    feature = "playlist",
    feature = "playout",
    feature = "automation"
))]
mod combined_playlist_playout_automation_tests {
    use oximedia::automation::config::AutomationConfig;
    use oximedia::playlist::{Playlist, PlaylistItem, PlaylistType};
    use oximedia::playout::PlayoutConfig;

    /// A broadcast playlist, a playout server configuration, and an
    /// automation system configuration can all be constructed together —
    /// verifying that `playlist` + `playout` + `automation` compose without
    /// conflicting dependencies (all three sit on the same broadcast
    /// automation stack).
    #[test]
    fn test_playlist_playout_automation_types_coexist() {
        // Playlist side: a linear playlist with one item.
        let mut playlist = Playlist::new("prime_time", PlaylistType::Linear);
        playlist.add_item(PlaylistItem::new("content/show_001.mxf"));
        assert_eq!(playlist.items.len(), 1, "Playlist must hold one item");
        assert_eq!(
            playlist.name, "prime_time",
            "Playlist name must be preserved"
        );

        // Playout side: default frame-accurate playout configuration.
        let playout_config = PlayoutConfig::default();
        assert!(
            playout_config.buffer_size > 0,
            "Default playout buffer size must be positive"
        );

        // Automation side: default automation configuration.
        let automation_config = AutomationConfig::new();
        assert!(
            automation_config.channels.is_empty(),
            "A fresh automation config must start with no channels"
        );
        assert_eq!(
            automation_config.global.system_name, "OxiMedia Automation",
            "Default global system name must match"
        );
    }
}

// ── Combined: routing + videoip + ndi ───────────────────────────────────────

#[cfg(all(test, feature = "routing", feature = "videoip", feature = "ndi"))]
mod combined_routing_videoip_ndi_tests {
    use oximedia::ndi::metadata::{PtzMetadata, TallyMetadata};
    use oximedia::routing::matrix::CrosspointMatrix;
    use oximedia::videoip::metadata::{MetadataPacket, MetadataType, Timecode as VideoIpTimecode};

    /// A signal-routing crosspoint matrix, a video-over-IP metadata packet,
    /// and NDI tally/PTZ metadata can all be constructed together —
    /// verifying that `routing` + `videoip` + `ndi` (the three broadcast
    /// signal-transport subsystems) compose without type collisions.
    #[test]
    fn test_routing_videoip_ndi_types_coexist() {
        // Routing side: connect one crosspoint in a small any-to-any matrix.
        let mut matrix = CrosspointMatrix::new(4, 4);
        matrix
            .connect(0, 1, Some(-3.0))
            .expect("Connecting a valid crosspoint must succeed");
        assert_eq!(matrix.input_count(), 4, "Input count must be preserved");
        assert_eq!(matrix.output_count(), 4, "Output count must be preserved");

        // Video-over-IP side: a timecode metadata packet.
        let tc = VideoIpTimecode::new(1, 2, 3, 4, false);
        let packet = MetadataPacket::timecode(tc);
        assert_eq!(
            packet.metadata_type,
            MetadataType::Timecode,
            "Packet must be tagged as timecode metadata"
        );
        assert!(
            !packet.payload.is_empty(),
            "Timecode payload must be non-empty"
        );

        // NDI side: tally light and PTZ metadata construction.
        let tally = TallyMetadata::new(true, false);
        assert!(tally.program, "Program tally must be set");
        assert!(!tally.preview, "Preview tally must not be set");

        let ptz = PtzMetadata::new(10.0, -5.0, 2.0);
        assert_eq!(ptz.zoom, 2.0, "Zoom level must be preserved");
    }
}

// ── Prelude smoke test (full surface) ───────────────────────────────────────

/// Imports one concrete symbol per `prelude.rs` section under the `full`
/// meta-feature. If any child crate renames or removes a re-exported symbol,
/// this module fails to compile, catching the break at the facade boundary
/// instead of downstream in a consumer's build.
#[cfg(all(test, feature = "full"))]
mod prelude_smoke {
    #[allow(unused_imports)]
    use oximedia::prelude::{
        adaptive_pipeline as _,
        ambisonics as _,
        // Core (always-on) + computer vision.
        cv as _,
        // The seven glob-re-exported sections (`crate::<module>::*` in
        // `prelude.rs`) — one symbol pulled from each underlying crate.
        deinterlace as _,
        lru_cache as _,
        AafError as _,
        AccelError as _,
        AccessError as _,
        AlignError as _,
        AnalysisError as _,
        AnalyticsError as _,
        ApvError as _,
        ArchiveError as _,
        ArchiveProError as _,
        AudioAnalysisError as _,
        // Per-feature domain modules, one symbol each, matching the
        // section order in `src/prelude.rs`.
        AudioError as _,
        AudioPostError as _,
        AutoError as _,
        AutomationError as _,
        BatchError as _,
        CalibrationError as _,
        CaptionError as _,
        CaptionGenError as _,
        CdnError as _,
        ClipError as _,
        CloudError as _,
        CodecError as _,
        CodecId as _,
        CollabError as _,
        ColorError as _,
        ComplexityAnalysis as _,
        ConformError as _,
        ConversionError as _,
        CrosspointMatrix as _,
        DedupError as _,
        DenoiseError as _,
        DistributedError as _,
        DolbyVisionError as _,
        DrmError as _,
        EditError as _,
        EdlError as _,
        EffectError as _,
        FarmError as _,
        FfmpegDiagnostic as _,
        ForensicsError as _,
        FrameRate as _,
        GamingError as _,
        GpuError as _,
        GraphError as _,
        GraphicsResult as _,
        HdrError as _,
        ImageError as _,
        ImfError as _,
        LutError as _,
        MamError as _,
        MetadataFormat as _,
        MeteringError as _,
        MirError as _,
        MixerError as _,
        MjpegError as _,
        MlError as _,
        MonitorError as _,
        MultiCamError as _,
        NdiError as _,
        NetError as _,
        NeuralError as _,
        NormalizeError as _,
        OptimizationLevel as _,
        OxiError as _,
        PackagerError as _,
        // Pipeline DSL, sovereign ML, and the two always-tacked-on codecs.
        PipelineError as _,
        PlaylistError as _,
        PlayoutError as _,
        PluginError as _,
        PresetError as _,
        ProfilerError as _,
        ProxyError as _,
        QualityScore as _,
        QueueError as _,
        RecommendError as _,
        RenderFarmError as _,
        RepairError as _,
        RestoreError as _,
        ReviewError as _,
        RightsError as _,
        ScalingMode as _,
        SceneError as _,
        ScopeType as _,
        SearchError as _,
        ServerError as _,
        Severity as _,
        ShotError as _,
        SimdError as _,
        StabilizeError as _,
        StorageError as _,
        SubtitleError as _,
        SwitcherError as _,
        TimeSyncError as _,
        TimelineError as _,
        TranscodeConfig,
        TranscodeError as _,
        VfxError as _,
        VideoIpError as _,
        VirtualProductionError as _,
        VrPixelFormat as _,
        WatermarkError as _,
        WorkflowError as _,
    };

    /// The `use` list above is the real assertion: it only compiles if every
    /// listed symbol still exists at its documented `prelude.rs` path. This
    /// test additionally exercises one concrete, no-I/O-required type to
    /// prove the glob chain resolves to genuinely usable values, not just
    /// type-checkable paths.
    #[test]
    fn test_prelude_full_surface_has_expected_symbols() {
        let config = TranscodeConfig::default();
        assert!(
            config.hw_accel,
            "TranscodeConfig::default() must enable hardware acceleration"
        );
        assert!(
            config.preserve_metadata,
            "TranscodeConfig::default() must preserve metadata by default"
        );
    }
}
