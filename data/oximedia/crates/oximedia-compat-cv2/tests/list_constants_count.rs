//! Sanity coverage for the build-time-generated `LIST_CONSTANTS` table.
//!
//! `build.rs` syn-parses `src/constants.rs` and emits a static slice of
//! `(category, name, type, value-as-string)` tuples. These tests verify
//! the table is non-trivial, well-formed, and covers the canonical
//! cv2 constants that the `oximedia-cv2` CLI relies on.

use oximedia_compat_cv2::constants_list::LIST_CONSTANTS;

#[test]
fn count_at_least_130() {
    assert!(
        LIST_CONSTANTS.len() >= 130,
        "expected at least 130 cv2 constants, got {}",
        LIST_CONSTANTS.len()
    );
}

#[test]
fn includes_imread_and_color_constants() {
    let names: Vec<&str> = LIST_CONSTANTS.iter().map(|(_, n, _, _)| *n).collect();
    let has = |s: &str| names.iter().any(|n| *n == s);
    assert!(has("IMREAD_COLOR"), "missing IMREAD_COLOR");
    assert!(has("IMREAD_GRAYSCALE"), "missing IMREAD_GRAYSCALE");
    assert!(has("COLOR_BGR2RGB"), "missing COLOR_BGR2RGB");
    assert!(has("COLOR_BGR2GRAY"), "missing COLOR_BGR2GRAY");
}

#[test]
fn entries_have_non_empty_fields() {
    for (category, name, ty, val) in LIST_CONSTANTS.iter() {
        // Top-level constants legitimately have an empty category string.
        let _ = category;
        assert!(!name.is_empty(), "empty name");
        assert!(!ty.is_empty(), "empty type for {name}");
        assert!(!val.is_empty(), "empty value for {name}");
    }
}

#[test]
fn category_is_known_module_or_empty() {
    // Categories must be either "" (file-root constants) or one of the
    // sub-modules declared in src/constants.rs. This guards against an
    // accidental rename or new module sneaking in unnoticed.
    let allowed: &[&str] = &[
        "",
        "adaptive_thresh",
        "border",
        "cap_prop",
        "chain_approx",
        "color",
        "compare",
        "contour_retr",
        "data_type",
        "dist_type",
        "draw_matches_flags",
        "feature_flags",
        "font",
        "hough",
        "imread",
        "interpolation",
        "line_type",
        "marker_type",
        "morph_op",
        "morph_shape",
        "norm_type",
        "optical_flow_flags",
        "rotate",
        "template_match",
        "threshold",
        "warp_flags",
    ];
    for (category, name, _, _) in LIST_CONSTANTS.iter() {
        assert!(
            allowed.contains(category),
            "unexpected category {category:?} for {name}; \
             update the allow-list or rename the sub-module"
        );
    }
}

#[test]
fn values_match_canonical_int_literals() {
    // Spot-check a handful of well-known cv2 integer values to confirm the
    // `value` column round-trips cleanly via `parse::<i32>()`.
    let expect: &[(&str, i32)] = &[
        ("IMREAD_COLOR", 1),
        ("IMREAD_GRAYSCALE", 0),
        ("IMREAD_UNCHANGED", -1),
        ("INTER_LINEAR", 1),
        ("THRESH_BINARY", 0),
        ("THRESH_OTSU", 8),
        ("MORPH_RECT", 0),
        ("CV_8U", 0),
        ("CV_32F", 5),
        ("FILLED", -1),
    ];
    for (want_name, want_val) in expect {
        let got = LIST_CONSTANTS
            .iter()
            .find(|(_, n, _, _)| n == want_name)
            .map(|(_, _, _, v)| v.parse::<i32>().ok());
        assert_eq!(got, Some(Some(*want_val)), "value mismatch for {want_name}");
    }
}
