//! Integration tests for `-hwaccel` argument wiring in `parse_and_translate`.

use oximedia_compat_ffmpeg::{parse_and_translate, HwBackend};

fn sv(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// `-hwaccel cuda` should produce a `TranscodeJob` with an `hwaccel` field
/// whose backend is `HwBackend::Cuda` and whose GPU flag is set.
#[test]
fn test_hwaccel_cuda_wired_into_job() {
    let args = sv(&[
        "-hwaccel",
        "cuda",
        "-i",
        "input.mkv",
        "-c:v",
        "av1",
        "output.webm",
    ]);
    let result = parse_and_translate(&args);
    assert!(
        !result.has_errors(),
        "unexpected errors: {:?}",
        result.diagnostics
    );
    assert_eq!(result.jobs.len(), 1);

    let hw = result.jobs[0]
        .hwaccel
        .as_ref()
        .expect("hwaccel should be populated from -hwaccel cuda");
    assert_eq!(hw.backend, HwBackend::Cuda);
    assert!(
        hw.is_gpu_enabled(),
        "Cuda backend should report GPU enabled"
    );
}

/// `-hwaccel vaapi` should produce a VAAPI config in the job.
#[test]
fn test_hwaccel_vaapi_wired_into_job() {
    let args = sv(&[
        "-hwaccel",
        "vaapi",
        "-i",
        "input.mp4",
        "-c:v",
        "vp9",
        "output.webm",
    ]);
    let result = parse_and_translate(&args);
    assert!(
        !result.has_errors(),
        "unexpected errors: {:?}",
        result.diagnostics
    );

    let hw = result.jobs[0]
        .hwaccel
        .as_ref()
        .expect("hwaccel should be populated from -hwaccel vaapi");
    assert_eq!(hw.backend, HwBackend::Vaapi);
    assert!(hw.is_gpu_enabled());
}

/// When `-hwaccel` is absent the `hwaccel` field in the job should be `None`.
#[test]
fn test_no_hwaccel_produces_none() {
    let args = sv(&["-i", "input.mkv", "-c:v", "av1", "output.webm"]);
    let result = parse_and_translate(&args);
    assert!(!result.has_errors());
    assert!(
        result.jobs[0].hwaccel.is_none(),
        "no -hwaccel flag should leave hwaccel as None"
    );
}

/// `-hwaccel none` / `software` should produce a Software backend.
#[test]
fn test_hwaccel_software_fallback() {
    let args = sv(&[
        "-hwaccel",
        "none",
        "-i",
        "input.mkv",
        "-c:a",
        "opus",
        "output.ogg",
    ]);
    let result = parse_and_translate(&args);
    assert!(!result.has_errors());
    let hw = result.jobs[0]
        .hwaccel
        .as_ref()
        .expect("hwaccel field populated");
    assert_eq!(hw.backend, HwBackend::Software);
    assert!(!hw.is_gpu_enabled());
    assert!(hw.is_software_only());
}
