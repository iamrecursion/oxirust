//! Batch normalization convergence / consistency tests.
//!
//! ## Why tests 1-3 target `batch_normalizer::BatchNormalizer`, not `batch::BatchProcessor`
//!
//! `batch_normalizer::BatchNormalizer` and `batch::BatchProcessor` are two distinct,
//! legitimate batch engines with different shapes:
//!
//! * `BatchNormalizer` operates on in-memory sample buffers the caller already holds
//!   (or on pre-computed measurements via `register_measurement`) and exposes an
//!   explicit two-pass measure → `schedule_gains` → `apply_to_item` flow with several
//!   gain modes (`Independent` / `Album` / `AlbumAverage`). Tests 1-3 exercise its
//!   closed-form gain-scheduling math directly.
//! * `BatchProcessor` operates on WAV files on disk end-to-end (decode → analyze →
//!   gain → encode); see `batch_processor_normalizes_real_wav_files_on_disk` below.
//!
//! (Historical note: `BatchProcessor::process_file`/`process_directory` were once
//! placeholder stubs that never ingested audio — see git history for
//! `batch_processor_is_a_placeholder_stub` — and have since been given a real,
//! WAV-backed implementation in `src/batch.rs`.)
//!
//! Key engine property exploited below: the scheduler does **not** re-measure after applying
//! gain. In `Independent` mode with `true_peak_ceiling_dbtp: None` and a wide gain window,
//! `gain_db == target_lufs - measured_lufs` *exactly*, so the modeled achieved loudness
//! `measured_lufs + gain_db == target_lufs` to f64 epsilon.

// Mirrors src/batch_normalizer.rs:45-47 — energy/dB math involves f64↔f32 casts in tests.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use oximedia_audio::wav::{WavReader, WavSpec, WavWriter};
use oximedia_metering::Standard;
use oximedia_normalize::batch::{BatchConfig, BatchProcessor};
use oximedia_normalize::batch_normalizer::{BatchNormalizer, BatchNormalizerConfig, GainMode};
use std::io::{BufReader, BufWriter};

/// Build a wide-window, peak-ceiling-free config that makes `gain_db` an exact function of
/// the measured loudness (no clamping, no peak back-off) for the given target.
fn exact_config(target_lufs: f64, mode: GainMode) -> BatchNormalizerConfig {
    BatchNormalizerConfig {
        target_lufs,
        max_gain_db: 40.0,
        min_gain_db: -40.0,
        mode,
        true_peak_ceiling_dbtp: None,
        clamp_gain: true,
    }
}

/// Synthesize a mono 1 kHz sine at a given amplitude expressed in dBFS.
///
/// `amp_dbfs` is the linear-amplitude level in dBFS (e.g. -6.0 ⇒ amplitude 10^(-6/20)).
fn sine(amp_dbfs: f64, sample_rate: f64, secs: f64) -> Vec<f32> {
    let amplitude = 10.0_f64.powf(amp_dbfs / 20.0) as f32;
    let n = (sample_rate * secs) as usize;
    let sr = sample_rate as f32;
    (0..n)
        .map(|i| amplitude * (std::f32::consts::TAU * 1000.0 * i as f32 / sr).sin())
        .collect()
}

/// 1. Independent mode drives **every** item exactly to the absolute target loudness.
///
/// Uses pre-computed measurements so the convergence is checked against the scheduler's
/// closed-form `gain_db = target - measured` (no measurement noise). The achieved loudness
/// model `measured + gain_db` must equal the target to f64 epsilon for all items, and the
/// per-item achieved loudness must agree across outputs.
#[test]
fn independent_mode_converges_every_item_to_exact_target() {
    let target = -16.0;
    let mut bn = BatchNormalizer::new(exact_config(target, GainMode::Independent))
        .expect("create normalizer");

    // (measured_lufs, true_peak_dbtp) triples spanning a 24 LU range.
    let items = [(-6.0, -3.0), (-18.0, -9.0), (-30.0, -15.0)];
    let mut ids = Vec::new();
    for (idx, &(measured, peak)) in items.iter().enumerate() {
        let id = bn
            .register_measurement(format!("item{idx}"), measured, peak, 48_000.0, 2)
            .expect("register measurement");
        ids.push(id);
    }

    let schedule = bn.schedule_gains().expect("schedule gains");

    // Nothing should be clamped given the wide ±40 dB window and no peak ceiling.
    assert_eq!(
        schedule.clamped_count, 0,
        "no item should be clamped with a ±40 dB window and no peak ceiling"
    );
    assert!(
        (schedule.effective_target_lufs - target).abs() < 1e-12,
        "Independent-mode effective target must equal config target: {} vs {target}",
        schedule.effective_target_lufs
    );

    let mut achieved = Vec::new();
    for (&id, &(measured, _)) in ids.iter().zip(items.iter()) {
        let entry = schedule.entry(id).expect("entry present");

        // Modeled achieved loudness lands exactly on target.
        let achieved_lufs = entry.measured_lufs + entry.gain_db;
        assert!(
            (achieved_lufs - target).abs() < 1e-9,
            "item {id}: achieved {achieved_lufs} != target {target}"
        );
        // Closed-form gain identity: gain_db == target - measured.
        assert!(
            (entry.gain_db - (target - measured)).abs() < 1e-9,
            "item {id}: gain_db {} != target-measured {}",
            entry.gain_db,
            target - measured
        );
        // Round-tripped measured value preserved exactly.
        assert!(
            (entry.measured_lufs - measured).abs() < 1e-12,
            "item {id}: measured_lufs {} != registered {measured}",
            entry.measured_lufs
        );
        assert!(
            !entry.gain_clamped,
            "item {id}: gain unexpectedly flagged as clamped"
        );
        achieved.push(achieved_lufs);
    }

    // Cross-output consistency: all items achieve the same loudness.
    for w in achieved.windows(2) {
        assert!(
            (w[0] - w[1]).abs() < 1e-9,
            "achieved loudness diverges across outputs: {} vs {}",
            w[0],
            w[1]
        );
    }
}

/// 2. Distinct *measured* signals at different loudness all schedule to the target within
///    EBU R128 tolerance.
///
/// Synthesizes three 1 kHz sines 12 dB apart, measures each through the real
/// `measure(&[f32], ...)` path, proves the measurements are distinct (> 1 LU apart), then
/// asserts each scheduled output lands on the target within R128 tolerance (±0.5 LU; in
/// practice it is exact because the scheduler does not re-measure).
#[test]
fn measured_signals_at_different_loudness_all_schedule_to_target_within_tolerance() {
    let target = -16.0;
    let sr = 48_000.0;
    let secs = 2.0;

    let mut bn = BatchNormalizer::new(exact_config(target, GainMode::Independent))
        .expect("create normalizer");

    let quiet = sine(-30.0, sr, secs);
    let mid = sine(-18.0, sr, secs);
    let loud = sine(-6.0, sr, secs);

    let id_q = bn.measure("quiet", &quiet, sr, 1).expect("measure quiet");
    let id_m = bn.measure("mid", &mid, sr, 1).expect("measure mid");
    let id_l = bn.measure("loud", &loud, sr, 1).expect("measure loud");

    // The three measurements must be genuinely distinct (energy 12 dB apart ⇒ ~12 LU apart),
    // otherwise the convergence assertion would be trivially satisfiable.
    let m = bn.measurements();
    let lufs: Vec<f64> = m.iter().map(|x| x.integrated_lufs).collect();
    for x in &lufs {
        assert!(x.is_finite(), "measured loudness must be finite, got {x}");
    }
    let mut sorted = lufs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite ordering"));
    for w in sorted.windows(2) {
        assert!(
            (w[1] - w[0]) > 1.0,
            "measured loudness levels not distinct (>1 LU apart): {sorted:?}"
        );
    }

    let schedule = bn.schedule_gains().expect("schedule gains");

    for id in [id_q, id_m, id_l] {
        let entry = schedule.entry(id).expect("entry present");
        let achieved = entry.measured_lufs + entry.gain_db;
        assert!(
            (achieved - target).abs() <= 0.5,
            "measured-path item {id}: achieved {achieved} outside ±0.5 LU of target {target}"
        );
    }
}

/// 3. Album mode shares a single gain across all items.
///
/// Documents that "consistent target loudness across outputs" is an *Independent*-mode
/// property: in `Album` mode the relative loudness relationships are preserved (one shared
/// gain), and only the loudest item is brought exactly to the target.
#[test]
fn album_mode_shares_single_gain() {
    let target = -16.0;
    let mut bn =
        BatchNormalizer::new(exact_config(target, GainMode::Album)).expect("create normalizer");

    // Same triples as test 1; loudest measured is -6 LUFS.
    let items = [(-6.0, -3.0), (-18.0, -9.0), (-30.0, -15.0)];
    let mut ids = Vec::new();
    for (idx, &(measured, peak)) in items.iter().enumerate() {
        let id = bn
            .register_measurement(format!("trk{idx}"), measured, peak, 48_000.0, 2)
            .expect("register measurement");
        ids.push(id);
    }

    let schedule = bn.schedule_gains().expect("schedule gains");

    // All entries share one gain.
    let gains: Vec<f64> = ids
        .iter()
        .map(|&id| schedule.entry(id).expect("entry present").gain_db)
        .collect();
    for w in gains.windows(2) {
        assert!(
            (w[0] - w[1]).abs() < 1e-9,
            "album mode must apply a single shared gain: {} vs {}",
            w[0],
            w[1]
        );
    }

    // The loudest item (-6 LUFS) is brought exactly to the target; quieter items are NOT,
    // by design — relative loudness is preserved.
    let loudest_measured = -6.0;
    let loudest_id = ids[0];
    let loudest = schedule.entry(loudest_id).expect("entry present");
    let loudest_achieved = loudest.measured_lufs + loudest.gain_db;
    assert!(
        (loudest_achieved - target).abs() < 1e-9,
        "album mode: loudest item achieved {loudest_achieved} != target {target}"
    );
    assert!(
        (loudest.gain_db - (target - loudest_measured)).abs() < 1e-9,
        "album mode: shared gain {} != target-loudest {}",
        loudest.gain_db,
        target - loudest_measured
    );

    // A quieter item, under the same shared gain, lands ABOVE the target (relative
    // loudness preserved) — i.e. it is NOT independently converged.
    let quiet = schedule.entry(ids[2]).expect("entry present");
    let quiet_achieved = quiet.measured_lufs + quiet.gain_db;
    assert!(
        quiet_achieved < target - 1.0,
        "album mode should preserve relative loudness; quiet item achieved {quiet_achieved} \
         was expected to remain well below target {target}"
    );
}

/// 4. `batch::BatchProcessor` now has a real, WAV-backed implementation: verify it
/// actually ingests audio from disk and converges loudness toward the configured
/// target, using real files under `std::env::temp_dir()`.
///
/// This supersedes the former `batch_processor_is_a_placeholder_stub` pin (see git
/// history) now that `process_file` / `process_directory` decode real audio, apply
/// a real gain, and write a real WAV output instead of operating on an empty meter.
#[test]
fn batch_processor_normalizes_real_wav_files_on_disk() {
    let sample_rate = 48_000u32;
    let channels = 1u16;
    let target = Standard::Spotify;

    let config = BatchConfig {
        standard: target,
        enable_limiter: false,
        enable_drc: false,
        write_metadata: false,
        write_replaygain: true,
        output_format: None,
        max_gain_db: 40.0,
        overwrite: true,
        parallel: false,
    };
    let processor = BatchProcessor::new(config);

    let dir = std::env::temp_dir().join("oximedia_batch_processor_convergence_test");
    let dir_out = dir.join("dir_out");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let input_path = dir.join("quiet_input.wav");
    let output_path = dir.join("normalized_output.wav");

    // Synthesize a quiet 1 kHz sine (well below the -14 LUFS Spotify target) as a
    // real 16-bit PCM WAV file — no fixtures, no shortcuts.
    let wav_spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        float: false,
    };
    let amplitude = 0.05_f32;
    let n = (f64::from(sample_rate) * 3.0) as usize;
    let input_samples: Vec<f32> = (0..n)
        .map(|i| amplitude * (std::f32::consts::TAU * 1000.0 * i as f32 / sample_rate as f32).sin())
        .collect();
    {
        let file = std::fs::File::create(&input_path).expect("create input wav");
        let mut writer = WavWriter::new(BufWriter::new(file), wav_spec);
        writer
            .write_samples_f32(&input_samples)
            .expect("write input samples");
        writer.finalize().expect("finalize input wav");
    }

    // -- process_file: real ingest, real gain, real output --------------------
    let result = processor
        .process_file(&input_path, &output_path, f64::from(sample_rate), 1)
        .expect("process_file should now really ingest audio and succeed");

    assert!(
        result.analysis.integrated_lufs.is_finite(),
        "STUB REGRESSION: integrated_lufs is not finite — process_file is not decoding \
         real audio"
    );
    assert!(
        result.applied_gain_db > 1.0,
        "STUB REGRESSION: applied_gain_db ({}) should be a substantial positive value for \
         a quiet input targeting {:?}",
        result.applied_gain_db,
        target
    );
    assert!(result.success);
    let rg = result
        .replay_gain
        .expect("write_replaygain=true should populate a real ReplayGain result");
    assert!(rg.track_gain.is_finite());

    // The output must exist and be measurably louder than the input — i.e. this is
    // not a silent / unmodified copy.
    let out_file = std::fs::File::open(&output_path).expect("open output wav");
    let mut out_reader = WavReader::new(BufReader::new(out_file)).expect("parse output wav header");
    let output_samples = out_reader
        .read_samples_f32()
        .expect("decode output samples");
    assert_eq!(output_samples.len(), input_samples.len());

    let rms = |s: &[f32]| -> f64 {
        (s.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / s.len() as f64).sqrt()
    };
    let rms_in = rms(&input_samples);
    let rms_out = rms(&output_samples);
    assert!(
        rms_out > rms_in * 1.5,
        "output should be measurably louder than input after positive-gain normalization: \
         rms_in={rms_in}, rms_out={rms_out}"
    );

    // -- process_directory: real enumeration, per-file results ------------------
    let results = processor
        .process_directory(&dir, &dir_out)
        .expect("process_directory should succeed");
    assert!(
        !results.is_empty(),
        "STUB REGRESSION: process_directory returned an empty Vec for a non-empty directory"
    );
    assert!(
        results.iter().any(|r| r.success),
        "process_directory should successfully process at least the WAV file present"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
