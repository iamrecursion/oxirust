//! Integration tests verifying that `-stats`, `-progress`, `-g`, and
//! `-keyint_min` are properly handled by `parse_and_translate` and do NOT
//! produce `UnknownOptionIgnored` diagnostics.

use oximedia_compat_ffmpeg::parse_and_translate;

fn sv(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// Helper: collect the text of all diagnostics that indicate an unknown option.
fn unknown_option_diagnostics(result: &oximedia_compat_ffmpeg::TranslateResult) -> Vec<String> {
    result
        .diagnostics
        .iter()
        .filter(|d| {
            let msg = format!("{:?}", d);
            msg.contains("unknown") || msg.contains("Unknown") || msg.contains("UnknownOption")
        })
        .map(|d| format!("{:?}", d))
        .collect()
}

// ── -stats ────────────────────────────────────────────────────────────────────

/// `-stats` (no-value flag) should be silently accepted — zero diagnostics.
#[test]
fn test_stats_flag_no_diagnostics() {
    let args = sv(&["-stats", "-i", "input.mkv", "-c:a", "opus", "output.ogg"]);
    let result = parse_and_translate(&args);
    assert!(
        !result.has_errors(),
        "unexpected errors: {:?}",
        result.diagnostics
    );
    let unknowns = unknown_option_diagnostics(&result);
    assert!(
        unknowns.is_empty(),
        "-stats must not produce UnknownOption diagnostics; got: {:?}",
        unknowns
    );
}

// ── -progress ────────────────────────────────────────────────────────────────

/// `-progress URL` should consume the URL and produce no unknown-option
/// diagnostics.
#[test]
fn test_progress_flag_no_diagnostics() {
    let args = sv(&[
        "-progress",
        "pipe:1",
        "-i",
        "input.mkv",
        "-c:a",
        "opus",
        "output.ogg",
    ]);
    let result = parse_and_translate(&args);
    assert!(
        !result.has_errors(),
        "unexpected errors: {:?}",
        result.diagnostics
    );
    let unknowns = unknown_option_diagnostics(&result);
    assert!(
        unknowns.is_empty(),
        "-progress must not produce UnknownOption diagnostics; got: {:?}",
        unknowns
    );
}

// ── -g / -keyint_min ─────────────────────────────────────────────────────────

/// `-g N` should parse into `TranscodeJob::gop_size` with no unknown-option
/// diagnostics.
#[test]
fn test_gop_size_wired_into_job() {
    let args = sv(&["-i", "input.mkv", "-c:v", "av1", "-g", "250", "output.webm"]);
    let result = parse_and_translate(&args);
    assert!(
        !result.has_errors(),
        "unexpected errors: {:?}",
        result.diagnostics
    );
    assert_eq!(result.jobs.len(), 1);

    let gop = result.jobs[0].gop_size;
    assert_eq!(gop, Some(250), "expected gop_size 250, got {:?}", gop);

    let unknowns = unknown_option_diagnostics(&result);
    assert!(
        unknowns.is_empty(),
        "-g must not produce UnknownOption diagnostics; got: {:?}",
        unknowns
    );
}

/// `-keyint_min N` should parse into `TranscodeJob::keyint_min` with no
/// unknown-option diagnostics.
#[test]
fn test_keyint_min_wired_into_job() {
    let args = sv(&[
        "-i",
        "input.mkv",
        "-c:v",
        "vp9",
        "-keyint_min",
        "25",
        "output.webm",
    ]);
    let result = parse_and_translate(&args);
    assert!(
        !result.has_errors(),
        "unexpected errors: {:?}",
        result.diagnostics
    );
    assert_eq!(result.jobs.len(), 1);

    let kmin = result.jobs[0].keyint_min;
    assert_eq!(kmin, Some(25), "expected keyint_min 25, got {:?}", kmin);

    let unknowns = unknown_option_diagnostics(&result);
    assert!(
        unknowns.is_empty(),
        "-keyint_min must not produce UnknownOption diagnostics; got: {:?}",
        unknowns
    );
}

/// Combining `-g` and `-keyint_min` in the same command should work cleanly.
#[test]
fn test_gop_and_keyint_combined() {
    let args = sv(&[
        "-i",
        "source.mkv",
        "-c:v",
        "av1",
        "-g",
        "120",
        "-keyint_min",
        "12",
        "-c:a",
        "opus",
        "output.webm",
    ]);
    let result = parse_and_translate(&args);
    assert!(
        !result.has_errors(),
        "unexpected errors: {:?}",
        result.diagnostics
    );

    let job = &result.jobs[0];
    assert_eq!(job.gop_size, Some(120));
    assert_eq!(job.keyint_min, Some(12));

    let unknowns = unknown_option_diagnostics(&result);
    assert!(
        unknowns.is_empty(),
        "combined -g -keyint_min must produce no UnknownOption diagnostics; got: {:?}",
        unknowns
    );
}

/// All four flags together in one command should be clean.
#[test]
fn test_all_four_flags_together() {
    let args = sv(&[
        "-stats",
        "-progress",
        "pipe:1",
        "-i",
        "input.mkv",
        "-c:v",
        "av1",
        "-g",
        "60",
        "-keyint_min",
        "6",
        "output.webm",
    ]);
    let result = parse_and_translate(&args);
    assert!(
        !result.has_errors(),
        "unexpected errors: {:?}",
        result.diagnostics
    );

    let job = &result.jobs[0];
    assert_eq!(job.gop_size, Some(60));
    assert_eq!(job.keyint_min, Some(6));

    let unknowns = unknown_option_diagnostics(&result);
    assert!(
        unknowns.is_empty(),
        "all four flags together must produce no UnknownOption diagnostics; got: {:?}",
        unknowns
    );
}
