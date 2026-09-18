//! Integration tests for `-map_metadata` argument parsing in `parse_and_translate`.

use oximedia_compat_ffmpeg::{parse_and_translate, MapMetadataDirective};

fn sv(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// `-map_metadata 0` should produce a `FromInput(0)` directive.
#[test]
fn test_map_metadata_from_input_zero() {
    let args = sv(&[
        "-i",
        "input.mkv",
        "-map_metadata",
        "0",
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

    let dirs = &result.jobs[0].map_metadata;
    assert_eq!(dirs.len(), 1);
    assert_eq!(dirs[0], MapMetadataDirective::FromInput(0));
}

/// `-map_metadata -1` should strip all metadata (`StripAll`).
#[test]
fn test_map_metadata_strip_all() {
    let args = sv(&["-i", "input.mkv", "-map_metadata", "-1", "output.webm"]);
    let result = parse_and_translate(&args);
    assert!(
        !result.has_errors(),
        "unexpected errors: {:?}",
        result.diagnostics
    );

    let dirs = &result.jobs[0].map_metadata;
    assert_eq!(dirs.len(), 1);
    assert_eq!(dirs[0], MapMetadataDirective::StripAll);
}

/// `-map_metadata 0:s:0` should produce a `FromStream` directive targeting
/// the first video (or generic) stream of input 0.
#[test]
fn test_map_metadata_from_stream() {
    let args = sv(&["-i", "input.mkv", "-map_metadata", "0:s:0", "output.webm"]);
    let result = parse_and_translate(&args);
    assert!(
        !result.has_errors(),
        "unexpected errors: {:?}",
        result.diagnostics
    );

    let dirs = &result.jobs[0].map_metadata;
    assert_eq!(dirs.len(), 1);
    assert!(matches!(
        dirs[0],
        MapMetadataDirective::FromStream {
            file_idx: 0,
            stream_type: 's',
            stream_idx: 0
        }
    ));
}

/// Multiple `-map_metadata` flags should all be collected.
#[test]
fn test_multiple_map_metadata_directives() {
    let args = sv(&[
        "-i",
        "input.mkv",
        "-map_metadata",
        "0",
        "-map_metadata",
        "-1",
        "output.webm",
    ]);
    let result = parse_and_translate(&args);
    assert!(!result.has_errors());

    let dirs = &result.jobs[0].map_metadata;
    assert_eq!(dirs.len(), 2);
    assert_eq!(dirs[0], MapMetadataDirective::FromInput(0));
    assert_eq!(dirs[1], MapMetadataDirective::StripAll);
}

/// When `-map_metadata` is absent the directive list should be empty.
#[test]
fn test_no_map_metadata_produces_empty_vec() {
    let args = sv(&["-i", "input.mkv", "-c:a", "opus", "output.ogg"]);
    let result = parse_and_translate(&args);
    assert!(!result.has_errors());
    assert!(result.jobs[0].map_metadata.is_empty());
}
