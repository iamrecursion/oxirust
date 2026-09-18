//! Golden translation tests for `oximedia-ff`.
//!
//! Each fixture in `tests/ff_golden/*.json` declares an FFmpeg-style invocation
//! and the expected translation output.  The runner shells out to the
//! `oximedia-ff` binary with `--json` (a structured-output flag added in 0.1.7)
//! and asserts on diagnostic kinds, number of jobs, codec choices, filter
//! counts, and other field-level expectations defined by [`Expect`].
//!
//! Adding a new fixture: drop a JSON file into `tests/ff_golden/` matching the
//! schema of [`Fixture`].  Keep expectations *narrow* — only assert on fields
//! you can verify by running the binary; partial-match is intentional.
//!
//! The fixtures cover ≥50 distinct FFmpeg-style invocations spanning patent
//! substitutions, quality flags, filter chains, filter_complex graphs, stream
//! mapping, seeking, metadata, two-pass encoding, hardware acceleration, and
//! loglevel/banner flags.

#![allow(clippy::expect_used)] // Tests intentionally panic on fixture I/O failure.

use assert_cmd::Command;
use serde_json::Value;
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Fixture {
    name: String,
    args: Vec<String>,
    #[serde(default)]
    expect: Expect,
}

#[derive(serde::Deserialize, Default)]
struct Expect {
    /// Optional — expect the exit status of the command.
    /// Defaults to `Some(true)` (success).
    #[serde(default)]
    success: Option<bool>,
    /// Optional — assert the diagnostics array length is exactly this.
    diagnostic_count: Option<usize>,
    /// Optional — every named kind must appear at least once in the
    /// `diagnostics[].kind` field of the JSON output.
    contains_diagnostic_kinds: Option<Vec<String>>,
    /// Optional — substring matching against `diagnostics[].message`.
    contains_diagnostic_messages: Option<Vec<String>>,
    /// Optional — exact number of jobs in the output.
    jobs: Option<usize>,
    /// Optional — first job's `video_codec` substring.
    video_codec: Option<String>,
    /// Optional — first job's `audio_codec` substring.
    audio_codec: Option<String>,
    /// Optional — first job's `crf` field (exact f64 within 0.001).
    crf: Option<f64>,
    /// Optional — first job's `video_bitrate` field.
    video_bitrate: Option<String>,
    /// Optional — first job's `audio_bitrate` field.
    audio_bitrate: Option<String>,
    /// Optional — first job's `preset` field.
    preset: Option<String>,
    /// Optional — first job's `tune` field.
    tune: Option<String>,
    /// Optional — first job's `profile` field.
    profile: Option<String>,
    /// Optional — first job's `seek` field (exact match).
    seek: Option<String>,
    /// Optional — first job's `duration` field.
    duration: Option<String>,
    /// Optional — first job's `format` field.
    format: Option<String>,
    /// Optional — first job's `video_filters` count.
    video_filters: Option<usize>,
    /// Optional — first job's `audio_filters` count.
    audio_filters: Option<usize>,
    /// Optional — first job's `overwrite` field.
    overwrite: Option<bool>,
    /// Optional — first job's `no_video` field.
    no_video: Option<bool>,
    /// Optional — first job's `no_audio` field.
    no_audio: Option<bool>,
    /// Optional — first job's `map` count.
    map: Option<usize>,
    /// Optional — first job's `map_metadata` count.
    map_metadata: Option<usize>,
    /// Optional — first job's `pass` field.
    pass: Option<u8>,
    /// Optional — first job's `hwaccel` substring (Debug name e.g. `"Cuda"`).
    hwaccel: Option<String>,
    /// Optional — first job's `muxer_actions` must contain this label.
    contains_muxer_actions: Option<Vec<String>>,
    /// Optional — first job's `metadata` map must include these key/value pairs.
    contains_metadata: Option<Vec<(String, String)>>,
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/ff_golden")
}

fn load_fixtures() -> Vec<(PathBuf, Fixture)> {
    let dir = fixtures_dir();
    let mut fixtures = Vec::new();
    let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("dir entry readable");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let body = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let fixture: Fixture = serde_json::from_str(&body)
            .unwrap_or_else(|e| panic!("{}: parse fixture: {e}", path.display()));
        fixtures.push((path, fixture));
    }
    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

#[test]
fn fixture_corpus_is_complete() {
    let fixtures = load_fixtures();
    assert!(
        fixtures.len() >= 50,
        "expected at least 50 ff_golden fixtures, found {} (in {})",
        fixtures.len(),
        fixtures_dir().display(),
    );
}

#[test]
fn fixture_names_unique() {
    let fixtures = load_fixtures();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (path, fx) in &fixtures {
        assert!(
            seen.insert(fx.name.clone()),
            "duplicate fixture name {:?} (in {})",
            fx.name,
            path.display(),
        );
    }
}

#[test]
fn all_golden_fixtures_pass() {
    let fixtures = load_fixtures();
    assert!(!fixtures.is_empty(), "no fixtures discovered");

    let mut failures: Vec<String> = Vec::new();

    for (path, fx) in &fixtures {
        let mut cmd =
            Command::cargo_bin("oximedia-ff").expect("cargo-built oximedia-ff binary present");
        cmd.arg("--json");
        for a in &fx.args {
            cmd.arg(a);
        }
        let assertion = cmd.assert();
        let expected_success = fx.expect.success.unwrap_or(true);
        let assertion = if expected_success {
            assertion.success()
        } else {
            assertion.failure()
        };
        let output = assertion.get_output().clone();

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let actual: Value = match serde_json::from_str(&stdout) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!(
                    "{} ({}): JSON parse error: {e}\n--- stdout ---\n{stdout}",
                    fx.name,
                    path.display()
                ));
                continue;
            }
        };

        if let Err(msg) = check_fixture(&fx.expect, &actual) {
            failures.push(format!(
                "{} ({}): {msg}\n--- args ---\n{}\n--- stdout ---\n{}",
                fx.name,
                path.display(),
                fx.args.join(" "),
                stdout,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} golden fixture(s) failed:\n\n{}",
        failures.len(),
        fixtures.len(),
        failures.join("\n\n========================================\n\n"),
    );
}

fn check_fixture(expect: &Expect, actual: &Value) -> Result<(), String> {
    let diags = actual
        .get("diagnostics")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "missing 'diagnostics' array".to_string())?;
    let jobs = actual
        .get("jobs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "missing 'jobs' array".to_string())?;

    if let Some(expected_count) = expect.diagnostic_count {
        if diags.len() != expected_count {
            return Err(format!(
                "diagnostics: expected {expected_count} got {}",
                diags.len()
            ));
        }
    }
    if let Some(kinds) = &expect.contains_diagnostic_kinds {
        for k in kinds {
            let found = diags.iter().any(|d| {
                d.get("kind")
                    .and_then(|v| v.as_str())
                    .map(|s| s == k)
                    .unwrap_or(false)
            });
            if !found {
                let actual_kinds: Vec<String> = diags
                    .iter()
                    .filter_map(|d| d.get("kind").and_then(|v| v.as_str()).map(str::to_string))
                    .collect();
                return Err(format!(
                    "missing diagnostic kind {k:?}; saw {actual_kinds:?}"
                ));
            }
        }
    }
    if let Some(msgs) = &expect.contains_diagnostic_messages {
        for needle in msgs {
            let found = diags.iter().any(|d| {
                d.get("message")
                    .and_then(|v| v.as_str())
                    .map(|s| s.contains(needle))
                    .unwrap_or(false)
            });
            if !found {
                return Err(format!("missing diagnostic message containing {needle:?}"));
            }
        }
    }
    if let Some(jc) = expect.jobs {
        if jobs.len() != jc {
            return Err(format!("jobs: expected {jc} got {}", jobs.len()));
        }
    }

    let Some(j0) = jobs.first() else {
        // No first job — only further checks possible if expect had no
        // first-job assertions, in which case we're fine.
        return if has_first_job_assertions(expect) {
            Err("no first job in output but fixture has first-job assertions".to_string())
        } else {
            Ok(())
        };
    };

    if let Some(vc) = &expect.video_codec {
        let actual_vc = j0.get("video_codec").and_then(|v| v.as_str()).unwrap_or("");
        if !actual_vc.contains(vc) {
            return Err(format!(
                "video_codec: expected to contain {vc:?}, got {actual_vc:?}"
            ));
        }
    }
    if let Some(ac) = &expect.audio_codec {
        let actual_ac = j0.get("audio_codec").and_then(|v| v.as_str()).unwrap_or("");
        if !actual_ac.contains(ac) {
            return Err(format!(
                "audio_codec: expected to contain {ac:?}, got {actual_ac:?}"
            ));
        }
    }
    if let Some(crf) = expect.crf {
        let actual_crf = j0.get("crf").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
        if !(actual_crf.is_finite() && (actual_crf - crf).abs() < 0.001) {
            return Err(format!("crf: expected {crf} got {actual_crf}"));
        }
    }
    if let Some(vb) = &expect.video_bitrate {
        let actual_vb = j0
            .get("video_bitrate")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if actual_vb != vb {
            return Err(format!("video_bitrate: expected {vb:?} got {actual_vb:?}"));
        }
    }
    if let Some(ab) = &expect.audio_bitrate {
        let actual_ab = j0
            .get("audio_bitrate")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if actual_ab != ab {
            return Err(format!("audio_bitrate: expected {ab:?} got {actual_ab:?}"));
        }
    }
    if let Some(p) = &expect.preset {
        let actual_p = j0.get("preset").and_then(|v| v.as_str()).unwrap_or("");
        if actual_p != p {
            return Err(format!("preset: expected {p:?} got {actual_p:?}"));
        }
    }
    if let Some(t) = &expect.tune {
        let actual_t = j0.get("tune").and_then(|v| v.as_str()).unwrap_or("");
        if actual_t != t {
            return Err(format!("tune: expected {t:?} got {actual_t:?}"));
        }
    }
    if let Some(pf) = &expect.profile {
        let actual_pf = j0.get("profile").and_then(|v| v.as_str()).unwrap_or("");
        if actual_pf != pf {
            return Err(format!("profile: expected {pf:?} got {actual_pf:?}"));
        }
    }
    if let Some(s) = &expect.seek {
        let actual_s = j0.get("seek").and_then(|v| v.as_str()).unwrap_or("");
        if actual_s != s {
            return Err(format!("seek: expected {s:?} got {actual_s:?}"));
        }
    }
    if let Some(d) = &expect.duration {
        let actual_d = j0.get("duration").and_then(|v| v.as_str()).unwrap_or("");
        if actual_d != d {
            return Err(format!("duration: expected {d:?} got {actual_d:?}"));
        }
    }
    if let Some(f) = &expect.format {
        let actual_f = j0.get("format").and_then(|v| v.as_str()).unwrap_or("");
        if actual_f != f {
            return Err(format!("format: expected {f:?} got {actual_f:?}"));
        }
    }
    if let Some(vf) = expect.video_filters {
        let actual_vf = j0
            .get("video_filters")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        if actual_vf != vf {
            return Err(format!("video_filters: expected {vf} got {actual_vf}"));
        }
    }
    if let Some(af) = expect.audio_filters {
        let actual_af = j0
            .get("audio_filters")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        if actual_af != af {
            return Err(format!("audio_filters: expected {af} got {actual_af}"));
        }
    }
    if let Some(o) = expect.overwrite {
        let actual_o = j0
            .get("overwrite")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if actual_o != o {
            return Err(format!("overwrite: expected {o} got {actual_o}"));
        }
    }
    if let Some(nv) = expect.no_video {
        let actual_nv = j0
            .get("no_video")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if actual_nv != nv {
            return Err(format!("no_video: expected {nv} got {actual_nv}"));
        }
    }
    if let Some(na) = expect.no_audio {
        let actual_na = j0
            .get("no_audio")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if actual_na != na {
            return Err(format!("no_audio: expected {na} got {actual_na}"));
        }
    }
    if let Some(m) = expect.map {
        let actual_m = j0.get("map").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if actual_m != m {
            return Err(format!("map: expected {m} got {actual_m}"));
        }
    }
    if let Some(mm) = expect.map_metadata {
        let actual_mm = j0.get("map_metadata").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if actual_mm != mm {
            return Err(format!("map_metadata: expected {mm} got {actual_mm}"));
        }
    }
    if let Some(p) = expect.pass {
        let actual_p = j0.get("pass").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
        if actual_p != p {
            return Err(format!("pass: expected {p} got {actual_p}"));
        }
    }
    if let Some(hw) = &expect.hwaccel {
        let actual_hw = j0.get("hwaccel").and_then(|v| v.as_str()).unwrap_or("");
        if !actual_hw.contains(hw) {
            return Err(format!(
                "hwaccel: expected to contain {hw:?} got {actual_hw:?}"
            ));
        }
    }
    if let Some(actions) = &expect.contains_muxer_actions {
        let actual_actions = j0
            .get("muxer_actions")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for action in actions {
            if !actual_actions.iter().any(|a| a == action) {
                return Err(format!(
                    "missing muxer action {action:?}; saw {actual_actions:?}"
                ));
            }
        }
    }
    if let Some(meta_pairs) = &expect.contains_metadata {
        let actual_meta = j0
            .get("metadata")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        for (k, v) in meta_pairs {
            let got = actual_meta
                .get(k)
                .and_then(|val| val.as_str())
                .unwrap_or("");
            if got != v {
                return Err(format!(
                    "metadata[{k:?}]: expected {v:?} got {got:?} (full map: {actual_meta:?})"
                ));
            }
        }
    }

    Ok(())
}

fn has_first_job_assertions(expect: &Expect) -> bool {
    expect.video_codec.is_some()
        || expect.audio_codec.is_some()
        || expect.crf.is_some()
        || expect.video_bitrate.is_some()
        || expect.audio_bitrate.is_some()
        || expect.preset.is_some()
        || expect.tune.is_some()
        || expect.profile.is_some()
        || expect.seek.is_some()
        || expect.duration.is_some()
        || expect.format.is_some()
        || expect.video_filters.is_some()
        || expect.audio_filters.is_some()
        || expect.overwrite.is_some()
        || expect.no_video.is_some()
        || expect.no_audio.is_some()
        || expect.map.is_some()
        || expect.map_metadata.is_some()
        || expect.pass.is_some()
        || expect.hwaccel.is_some()
        || expect.contains_muxer_actions.is_some()
        || expect.contains_metadata.is_some()
}
