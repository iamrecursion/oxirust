//! Structural and self-consistency validation for HLS M3U8, DASH MPD, and
//! bitrate-ladder output.
//!
//! # Scope
//!
//! These tests parse the text that our own generators produce and assert
//! structural invariants. They are NOT conformance tests against external
//! tools:
//!
//! - HLS: not run through Apple's `mediastreamvalidator`
//! - DASH: not validated against the ISO 23009-1 MPD XML Schema (XSD)
//! - Ladder: not verified against a standards-body conformance suite
//!
//! They provide fast, deterministic, in-repo confidence that the generators
//! emit structurally correct and internally self-consistent documents.

use oximedia_packager::dash::{
    AdaptationSet, DashProfile, MpdBuilder, MpdType, Period, Representation, SegmentTemplate,
};
use oximedia_packager::hls::VariantStream as HlsVariantStream;
use oximedia_packager::hls::{MasterPlaylistBuilder, MediaPlaylistBuilder};
use oximedia_packager::ladder::{BitrateLadderGenerator, LadderPresets, SourceAnalysis};
use oximedia_packager::manifest_update::{ManifestSegmentEntry, ManifestType, ManifestUpdater};
use oximedia_packager::segment::SegmentInfo;
use std::collections::HashSet;
use std::time::Duration;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

fn make_segment(index: u64, duration_ms: u64, path: &str) -> SegmentInfo {
    SegmentInfo {
        index,
        duration: Duration::from_millis(duration_ms),
        size: 512_000,
        path: path.to_string(),
        keyframe: true,
        timestamp: Duration::from_millis(index * duration_ms),
    }
}

/// Parse the integer value after a tag prefix, e.g.
/// `parse_tag_integer(playlist, "#EXT-X-TARGETDURATION:") → Some(7)`.
fn parse_tag_integer(text: &str, prefix: &str) -> Option<u64> {
    text.lines()
        .find(|l| l.starts_with(prefix))
        .and_then(|l| l.trim_start_matches(prefix).trim().parse().ok())
}

/// Collect all `#EXTINF:X.XXX,` durations from a media playlist.
fn collect_extinf_secs(playlist: &str) -> Vec<f64> {
    playlist
        .lines()
        .filter_map(|l| {
            let rest = l.strip_prefix("#EXTINF:")?;
            rest.split(',').next()?.parse().ok()
        })
        .collect()
}

/// Return the 0-based line index where a line starts with `prefix`,
/// or `None` if not found.
fn line_pos(text: &str, prefix: &str) -> Option<usize> {
    text.lines().position(|l| l.starts_with(prefix))
}

// ══════════════════════════════════════════════════════════════════════════════
// HLS M3U8 structural validation
// ══════════════════════════════════════════════════════════════════════════════

/// The mandatory HLS header tags must appear in the order required by the spec:
///   line 0  → `#EXTM3U`
///   before target-duration → `#EXT-X-VERSION`
///   before media-sequence  → `#EXT-X-TARGETDURATION`
///   before first segment   → `#EXT-X-MEDIA-SEQUENCE`
///
/// NOTE: structural/self-consistency check, not mediastreamvalidator output.
#[test]
fn test_hls_media_playlist_required_tag_order() {
    let mut builder = MediaPlaylistBuilder::new(Duration::from_secs(6));
    builder.add_segment(make_segment(0, 6_000, "seg0.m4s"));
    builder.add_segment(make_segment(1, 6_000, "seg1.m4s"));
    let playlist = builder
        .build()
        .expect("MediaPlaylistBuilder::build must succeed");

    let i_extm3u = line_pos(&playlist, "#EXTM3U").expect("#EXTM3U must be present");
    let i_version = line_pos(&playlist, "#EXT-X-VERSION").expect("#EXT-X-VERSION must be present");
    let i_target = line_pos(&playlist, "#EXT-X-TARGETDURATION")
        .expect("#EXT-X-TARGETDURATION must be present");
    let i_seq = line_pos(&playlist, "#EXT-X-MEDIA-SEQUENCE")
        .expect("#EXT-X-MEDIA-SEQUENCE must be present");
    let i_seg0 = playlist
        .lines()
        .position(|l| l == "seg0.m4s")
        .expect("segment URI seg0.m4s must appear in playlist");

    assert_eq!(i_extm3u, 0, "#EXTM3U must be the first line (line 0)");
    assert!(i_extm3u < i_version, "#EXTM3U must precede #EXT-X-VERSION");
    assert!(
        i_version < i_target,
        "#EXT-X-VERSION must precede #EXT-X-TARGETDURATION"
    );
    assert!(
        i_target < i_seq,
        "#EXT-X-TARGETDURATION must precede #EXT-X-MEDIA-SEQUENCE"
    );
    assert!(
        i_seq < i_seg0,
        "#EXT-X-MEDIA-SEQUENCE must precede the first segment URI"
    );
}

/// `#EXT-X-TARGETDURATION` must be ≥ every `#EXTINF` duration in the playlist
/// (HLS spec §4.4.3.1 — TARGETDURATION is a ceiling, not a floor).
///
/// NOTE: structural/self-consistency check, not mediastreamvalidator output.
#[test]
fn test_hls_targetduration_gte_max_extinf() {
    // Segments: 5.500 s, 6.000 s, 5.750 s — max EXTINF = 6.000 s.
    let mut builder = MediaPlaylistBuilder::new(Duration::from_secs(6));
    builder.add_segment(make_segment(0, 5_500, "a.m4s"));
    builder.add_segment(make_segment(1, 6_000, "b.m4s"));
    builder.add_segment(make_segment(2, 5_750, "c.m4s"));
    let playlist = builder.build().expect("build must succeed");

    let target_dur = parse_tag_integer(&playlist, "#EXT-X-TARGETDURATION:")
        .expect("#EXT-X-TARGETDURATION must be present");

    let extinf = collect_extinf_secs(&playlist);
    assert!(!extinf.is_empty(), "playlist must contain EXTINF lines");

    let max_extinf = extinf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    assert!(
        target_dur as f64 >= max_extinf,
        "#EXT-X-TARGETDURATION ({target_dur}) must be >= max EXTINF ({max_extinf:.3})"
    );
}

/// A VOD media playlist built with `.with_end_list()` must end with
/// `#EXT-X-ENDLIST` as the last non-empty line.
///
/// NOTE: structural/self-consistency check.
#[test]
fn test_hls_vod_playlist_last_tag_is_endlist() {
    let mut builder = MediaPlaylistBuilder::new(Duration::from_secs(6)).with_end_list();
    builder.add_segment(make_segment(0, 6_000, "s0.m4s"));
    builder.add_segment(make_segment(1, 6_000, "s1.m4s"));
    let playlist = builder.build().expect("build must succeed");

    let last_non_empty = playlist
        .lines()
        .filter(|l| !l.is_empty())
        .last()
        .expect("playlist must be non-empty");

    assert_eq!(
        last_non_empty, "#EXT-X-ENDLIST",
        "last non-empty line of a VOD playlist must be #EXT-X-ENDLIST, got: {last_non_empty:?}"
    );
}

/// A live media playlist (no `.with_end_list()`) must NOT carry
/// `#EXT-X-ENDLIST`.
#[test]
fn test_hls_live_playlist_has_no_endlist() {
    let mut builder = MediaPlaylistBuilder::new(Duration::from_secs(6));
    builder.add_segment(make_segment(0, 6_000, "s0.m4s"));
    let playlist = builder.build().expect("build must succeed");

    assert!(
        !playlist.contains("#EXT-X-ENDLIST"),
        "live playlist must not carry #EXT-X-ENDLIST"
    );
}

/// Every `#EXT-X-STREAM-INF` line in a master playlist must include
/// `BANDWIDTH=`, `CODECS=`, and (for video tracks) `RESOLUTION=`.
///
/// NOTE: structural/self-consistency check, not mediastreamvalidator output.
#[test]
fn test_hls_master_stream_inf_required_attributes() {
    let mut builder = MasterPlaylistBuilder::new();
    builder.add_variant(
        HlsVariantStream::new(
            5_000_000,
            "av01.0.09M.08".to_string(),
            "1080p.m3u8".to_string(),
        )
        .with_resolution(1920, 1080)
        .with_frame_rate(25.0),
    );
    builder.add_variant(
        HlsVariantStream::new(
            2_500_000,
            "av01.0.05M.08".to_string(),
            "720p.m3u8".to_string(),
        )
        .with_resolution(1280, 720)
        .with_frame_rate(25.0),
    );
    let playlist = builder.build().expect("build must succeed");

    let mut stream_inf_count = 0usize;
    for line in playlist.lines() {
        if line.starts_with("#EXT-X-STREAM-INF:") {
            stream_inf_count += 1;
            assert!(
                line.contains("BANDWIDTH="),
                "#EXT-X-STREAM-INF must carry BANDWIDTH=: {line}"
            );
            assert!(
                line.contains("CODECS="),
                "#EXT-X-STREAM-INF must carry CODECS=: {line}"
            );
            assert!(
                line.contains("RESOLUTION="),
                "#EXT-X-STREAM-INF for a video variant must carry RESOLUTION=: {line}"
            );
        }
    }
    assert!(
        stream_inf_count >= 2,
        "master playlist must contain at least 2 #EXT-X-STREAM-INF lines (found {stream_inf_count})"
    );
}

/// Every `#EXT-X-STREAM-INF` tag must be immediately followed by a URI line
/// (non-empty, not a tag).  HLS spec RFC 8216 §4.3.4.2.
///
/// NOTE: structural check using MasterPlaylistBuilder (the existing
/// manifest_builder.rs tests cover ManifestBuilder; this covers the richer
/// builder).
#[test]
fn test_hls_stream_inf_immediately_followed_by_uri() {
    let mut builder = MasterPlaylistBuilder::new();
    builder.add_variant(
        HlsVariantStream::new(3_000_000, "av01".to_string(), "v720.m3u8".to_string())
            .with_resolution(1280, 720),
    );
    builder.add_variant(
        HlsVariantStream::new(1_500_000, "av01".to_string(), "v480.m3u8".to_string())
            .with_resolution(854, 480),
    );
    let playlist = builder.build().expect("build must succeed");
    let lines: Vec<&str> = playlist.lines().collect();

    for (i, &line) in lines.iter().enumerate() {
        if line.starts_with("#EXT-X-STREAM-INF:") {
            let next = lines.get(i + 1).copied().unwrap_or("");
            assert!(
                !next.is_empty(),
                "#EXT-X-STREAM-INF at line {i} must not be followed by an empty line"
            );
            assert!(
                !next.starts_with('#'),
                "#EXT-X-STREAM-INF at line {i} must be followed by a URI, not a tag: got {next:?}"
            );
        }
    }
}

/// Simulate 20 continuous live segment arrivals and verify that the
/// `#EXT-X-MEDIA-SEQUENCE` value in the rendered playlist is
/// monotonically non-decreasing throughout.
///
/// NOTE: structural/self-consistency check.
#[test]
fn test_hls_media_sequence_monotonically_non_decreasing() {
    let mut updater = ManifestUpdater::new(ManifestType::HlsMedia, 5);
    let mut prev_seq = 0u64;

    for i in 0..20u64 {
        updater.add_segment(ManifestSegmentEntry::new(
            i,
            Duration::from_secs(6),
            format!("seg{i:04}.m4s"),
        ));
        let playlist = updater.render_hls_media_playlist();
        let seq = parse_tag_integer(&playlist, "#EXT-X-MEDIA-SEQUENCE:")
            .expect("#EXT-X-MEDIA-SEQUENCE must be present in live playlist");

        assert!(
            seq >= prev_seq,
            "#EXT-X-MEDIA-SEQUENCE must be non-decreasing: {prev_seq} → {seq} at step {i}"
        );
        prev_seq = seq;
    }
    // After 20 segments with a 5-segment window, 15 segments were evicted,
    // so the final sequence must be 15.
    assert_eq!(
        prev_seq, 15,
        "after evicting 15 segments the media sequence must be 15"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// DASH MPD structural validation
// ══════════════════════════════════════════════════════════════════════════════

fn video_repr(id: &str, bw: u32, w: u32, h: u32) -> Representation {
    Representation::new(id.to_string(), bw, "av01.0.05M.08".to_string())
        .with_dimensions(w, h)
        .with_frame_rate(25.0)
}

fn audio_repr(id: &str, bw: u32) -> Representation {
    Representation::new(id.to_string(), bw, "opus".to_string())
}

/// The MPD root element must carry a `profiles=` attribute referencing a
/// recognised `urn:mpeg:dash:profile:` URN.
///
/// NOTE: structural check; not validated against the ISO 23009-1 XSD.
#[test]
fn test_dash_mpd_has_profiles_attribute() {
    let mpd = MpdBuilder::new(MpdType::Static, DashProfile::OnDemand)
        .build()
        .expect("MpdBuilder::build must succeed");

    assert!(
        mpd.contains("profiles="),
        "MPD root element must carry a 'profiles=' attribute"
    );
    assert!(
        mpd.contains("urn:mpeg:dash:profile:"),
        "profiles must reference a urn:mpeg:dash:profile: URN; got: {mpd}"
    );
}

/// The generated MPD must exhibit the mandatory hierarchy:
/// `<MPD>` → `<Period>` → `<AdaptationSet>` → `<Representation>`.
/// Every opened element must have a matching closing tag.
///
/// NOTE: structural/self-consistency check, not ISO 23009-1 XSD validation.
#[test]
fn test_dash_mpd_required_hierarchy_and_well_formedness() {
    let mut builder = MpdBuilder::new(MpdType::Static, DashProfile::OnDemand)
        .with_duration(Duration::from_secs(120));

    let mut period = Period::new("p0".to_string()).with_duration(Duration::from_secs(120));
    let mut video_adapt = AdaptationSet::new(0, "video".to_string(), "video/mp4".to_string());
    video_adapt.add_representation(video_repr("v1", 5_000_000, 1920, 1080));
    video_adapt.add_representation(video_repr("v2", 2_500_000, 1280, 720));
    let mut audio_adapt = AdaptationSet::new(1, "audio".to_string(), "audio/mp4".to_string());
    audio_adapt.add_representation(audio_repr("a1", 128_000));
    period.add_adaptation_set(video_adapt);
    period.add_adaptation_set(audio_adapt);
    builder.add_period(period);

    let mpd = builder.build().expect("build must succeed");

    // Required elements present.
    for tag in ["<MPD", "<Period", "<AdaptationSet", "<Representation"] {
        assert!(mpd.contains(tag), "MPD must contain {tag}");
    }

    // Closing tags must balance opening tags (well-formedness check).
    for (open, close) in [
        ("<MPD", "</MPD>"),
        ("<Period", "</Period>"),
        ("<AdaptationSet", "</AdaptationSet>"),
        ("<Representation", "</Representation>"),
    ] {
        let n_open = mpd.matches(open).count();
        let n_close = mpd.matches(close).count();
        assert_eq!(
            n_open, n_close,
            "{open} count ({n_open}) must equal {close} count ({n_close})"
        );
    }
}

/// Every `<Representation>` element must carry `id=` and `bandwidth=`
/// attributes with the values that were configured.
///
/// NOTE: structural/self-consistency check, not ISO 23009-1 XSD validation.
#[test]
fn test_dash_mpd_representation_has_id_and_bandwidth() {
    let mut builder = MpdBuilder::new(MpdType::Static, DashProfile::OnDemand);
    let mut period = Period::new("p0".to_string());
    let mut adapt = AdaptationSet::new(0, "video".to_string(), "video/mp4".to_string());
    adapt.add_representation(video_repr("video_1080p", 5_000_000, 1920, 1080));
    adapt.add_representation(video_repr("video_720p", 2_500_000, 1280, 720));
    period.add_adaptation_set(adapt);
    builder.add_period(period);

    let mpd = builder.build().expect("build must succeed");

    // Parse out the Representation opening-tag fragments.
    let repr_frags: Vec<&str> = mpd.split("<Representation").skip(1).collect();
    assert_eq!(
        repr_frags.len(),
        2,
        "expected exactly 2 Representation elements in the MPD"
    );

    for frag in &repr_frags {
        // The attribute list ends at the first '>' or '/>'.
        let attrs = frag.split_once('>').map(|(a, _)| a).unwrap_or(frag);
        assert!(
            attrs.contains("id="),
            "Representation must carry 'id=' attribute in: {attrs}"
        );
        assert!(
            attrs.contains("bandwidth="),
            "Representation must carry 'bandwidth=' attribute in: {attrs}"
        );
    }

    // Spot-check specific values.
    assert!(
        mpd.contains("bandwidth=\"5000000\""),
        "5 Mbps bandwidth must appear"
    );
    assert!(
        mpd.contains("bandwidth=\"2500000\""),
        "2.5 Mbps bandwidth must appear"
    );
    assert!(
        mpd.contains("id=\"video_1080p\""),
        "video_1080p id must appear"
    );
    assert!(
        mpd.contains("id=\"video_720p\""),
        "video_720p id must appear"
    );
}

/// A `<SegmentTemplate>` element must carry a `timescale=` attribute.
///
/// NOTE: structural check; not ISO 23009-1 XSD validation.
#[test]
fn test_dash_mpd_segment_template_timescale_present() {
    let mut builder = MpdBuilder::new(MpdType::Static, DashProfile::Live);
    let mut period = Period::new("p0".to_string());
    let mut adapt = AdaptationSet::new(0, "video".to_string(), "video/mp4".to_string());

    let template = SegmentTemplate::new(
        "init-$RepresentationID$.mp4".to_string(),
        "$RepresentationID$-seg$Number$.m4s".to_string(),
        240_000, // duration in timescale units (= 6 s at 40 000 Hz)
        40_000,  // timescale: 40 000 ticks per second
    );
    let repr = video_repr("v1", 3_000_000, 1280, 720).with_segment_template(template);
    adapt.add_representation(repr);
    period.add_adaptation_set(adapt);
    builder.add_period(period);

    let mpd = builder.build().expect("build must succeed");

    assert!(
        mpd.contains("SegmentTemplate"),
        "MPD must include a <SegmentTemplate> element"
    );
    assert!(
        mpd.contains("timescale="),
        "SegmentTemplate must carry a 'timescale=' attribute"
    );
    assert!(
        mpd.contains("timescale=\"40000\""),
        "SegmentTemplate timescale value must be 40000; MPD:\n{mpd}"
    );
}

/// `mediaPresentationDuration` on a static MPD must be formatted as an
/// ISO 8601 duration (`PT...S`).
///
/// NOTE: structural/format check; not ISO 23009-1 XSD validation.
#[test]
fn test_dash_mpd_duration_format_is_iso8601() {
    let mut builder = MpdBuilder::new(MpdType::Static, DashProfile::OnDemand)
        .with_duration(Duration::from_secs(3601)); // 1 h + 1 s
    let period = Period::new("p0".to_string());
    builder.add_period(period);

    let mpd = builder.build().expect("build must succeed");

    assert!(
        mpd.contains("mediaPresentationDuration="),
        "static MPD must carry mediaPresentationDuration attribute"
    );

    // Extract the quoted duration value.
    let dur_val = mpd
        .split("mediaPresentationDuration=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .expect("mediaPresentationDuration must have a quoted value");

    assert!(
        dur_val.starts_with("PT"),
        "mediaPresentationDuration must start with 'PT' (ISO 8601): got {dur_val:?}"
    );
    assert!(
        dur_val.ends_with('S'),
        "mediaPresentationDuration must end with 'S' (ISO 8601): got {dur_val:?}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Bitrate-ladder ordering assertions
// ══════════════════════════════════════════════════════════════════════════════

/// `BitrateLadderGenerator` must produce rungs in STRICTLY DESCENDING bitrate
/// order (highest-resolution rung first) for every supported source resolution.
///
/// NOTE: structural/self-consistency check.
#[test]
fn test_ladder_generator_rungs_strictly_descending_bitrate() {
    for (w, h, label) in [
        (3840u32, 2160u32, "4K"),
        (1920, 1080, "1080p"),
        (1280, 720, "720p"),
    ] {
        let rungs = BitrateLadderGenerator::new(SourceAnalysis::new(w, h, 0.5))
            .generate()
            .expect("generate must succeed");

        let bps: Vec<u64> = rungs.iter().map(|r| r.target_bitrate_bps).collect();
        for pair in bps.windows(2) {
            assert!(
                pair[0] > pair[1],
                "{label} ladder must be in strictly descending bitrate order: {bps:?}"
            );
        }
    }
}

/// `BitrateLadderGenerator` must not produce two rungs with identical
/// `target_bitrate_bps` values.
///
/// NOTE: structural/self-consistency check.
#[test]
fn test_ladder_generator_no_duplicate_bitrates() {
    for (w, h, label) in [(3840u32, 2160u32, "4K"), (1920, 1080, "1080p")] {
        let rungs = BitrateLadderGenerator::new(SourceAnalysis::new(w, h, 0.5))
            .generate()
            .expect("generate must succeed");

        let bps: Vec<u64> = rungs.iter().map(|r| r.target_bitrate_bps).collect();
        let unique: HashSet<u64> = bps.iter().copied().collect();
        assert_eq!(
            bps.len(),
            unique.len(),
            "{label} ladder must not contain duplicate bitrates: {bps:?}"
        );
    }
}

/// For every pair of adjacent rungs, the higher-bitrate rung must have a
/// pixel count at least as large as the lower-bitrate rung.  Resolution
/// must be monotonically non-increasing as bitrate decreases.
///
/// NOTE: structural/self-consistency check.
#[test]
fn test_ladder_generator_resolution_monotonic_with_bitrate() {
    let rungs = BitrateLadderGenerator::new(SourceAnalysis::new(1920, 1080, 0.5))
        .generate()
        .expect("generate must succeed");

    // Rungs are highest-first, so pixel counts must also be non-increasing.
    let pixels: Vec<u64> = rungs.iter().map(|r| r.pixels()).collect();
    for pair in pixels.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "pixel count must be non-increasing across adjacent rungs: {pixels:?}"
        );
    }
}

/// `LadderPresets::hls_1080p()` must be in strictly descending bitrate order
/// and have no duplicates.
///
/// NOTE: structural/self-consistency check.
#[test]
fn test_preset_hls_1080p_descending_no_duplicates() {
    let ladder = LadderPresets::hls_1080p();
    let bps: Vec<u32> = ladder.entries.iter().map(|e| e.bitrate).collect();
    for pair in bps.windows(2) {
        assert!(
            pair[0] > pair[1],
            "hls_1080p preset must be in strictly descending bitrate order: {bps:?}"
        );
    }
    let unique: HashSet<u32> = bps.iter().copied().collect();
    assert_eq!(
        bps.len(),
        unique.len(),
        "hls_1080p preset must have no duplicate bitrates"
    );
}

/// `LadderPresets::dash_4k()` must be in strictly descending bitrate order
/// and have no duplicates.
///
/// NOTE: structural/self-consistency check.
#[test]
fn test_preset_dash_4k_descending_no_duplicates() {
    let ladder = LadderPresets::dash_4k();
    let bps: Vec<u32> = ladder.entries.iter().map(|e| e.bitrate).collect();
    for pair in bps.windows(2) {
        assert!(
            pair[0] > pair[1],
            "dash_4k preset must be in strictly descending bitrate order: {bps:?}"
        );
    }
    let unique: HashSet<u32> = bps.iter().copied().collect();
    assert_eq!(
        bps.len(),
        unique.len(),
        "dash_4k preset must have no duplicate bitrates"
    );
}

/// `LadderPresets::mobile_optimized()` must be in strictly descending bitrate
/// order and have no duplicates.
///
/// NOTE: structural/self-consistency check.
#[test]
fn test_preset_mobile_optimized_descending_no_duplicates() {
    let ladder = LadderPresets::mobile_optimized();
    let bps: Vec<u32> = ladder.entries.iter().map(|e| e.bitrate).collect();
    for pair in bps.windows(2) {
        assert!(
            pair[0] > pair[1],
            "mobile_optimized preset must be in strictly descending bitrate order: {bps:?}"
        );
    }
    let unique: HashSet<u32> = bps.iter().copied().collect();
    assert_eq!(
        bps.len(),
        unique.len(),
        "mobile_optimized preset must have no duplicate bitrates"
    );
}

/// Resolution (pixels) must be non-increasing as bitrate decreases for every
/// preset ladder.
///
/// NOTE: structural/self-consistency check.
#[test]
fn test_presets_resolution_monotonic_with_bitrate() {
    for (name, ladder) in [
        ("hls_1080p", LadderPresets::hls_1080p()),
        ("dash_4k", LadderPresets::dash_4k()),
        ("mobile_optimized", LadderPresets::mobile_optimized()),
    ] {
        let pixels: Vec<u64> = ladder
            .entries
            .iter()
            .map(|e| u64::from(e.width) * u64::from(e.height))
            .collect();
        for pair in pixels.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "preset '{name}' pixel count must be non-increasing with bitrate: {pixels:?}"
            );
        }
    }
}

/// `MasterPlaylistBuilder` automatically sorts variants by bandwidth
/// (ascending).  Verify that regardless of insertion order the rendered
/// playlist always has BANDWIDTH values in strictly ascending order.
///
/// NOTE: structural/self-consistency check.
#[test]
fn test_master_playlist_builder_sorts_ascending_by_bandwidth() {
    let mut builder = MasterPlaylistBuilder::new();
    // Insert in descending order; the builder should sort to ascending.
    builder.add_variant(
        HlsVariantStream::new(5_000_000, "av01".to_string(), "1080p.m3u8".to_string())
            .with_resolution(1920, 1080),
    );
    builder.add_variant(
        HlsVariantStream::new(2_500_000, "av01".to_string(), "720p.m3u8".to_string())
            .with_resolution(1280, 720),
    );
    builder.add_variant(
        HlsVariantStream::new(1_200_000, "av01".to_string(), "480p.m3u8".to_string())
            .with_resolution(854, 480),
    );
    let playlist = builder.build().expect("build must succeed");

    // Collect all BANDWIDTH= values in document order.
    let bandwidths: Vec<u32> = playlist
        .lines()
        .filter(|l| l.starts_with("#EXT-X-STREAM-INF:"))
        .filter_map(|l| {
            l.split("BANDWIDTH=")
                .nth(1)
                .and_then(|s| s.split([',', '\n', '\r']).next())
                .and_then(|s| s.parse().ok())
        })
        .collect();

    assert_eq!(
        bandwidths.len(),
        3,
        "expected 3 BANDWIDTH values in master playlist"
    );
    for pair in bandwidths.windows(2) {
        assert!(
            pair[0] < pair[1],
            "MasterPlaylistBuilder must output variants in ascending BANDWIDTH order: {bandwidths:?}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Live packaging rolling-window integration test
// ══════════════════════════════════════════════════════════════════════════════

/// Simulate 20 continuous segment arrivals into a live manifest updater with
/// a 5-segment sliding window.  Assert all of the following on every step:
///
/// 1. The window size stays ≤ 5.
/// 2. `#EXT-X-MEDIA-SEQUENCE` is monotonically non-decreasing.
/// 3. `#EXT-X-TARGETDURATION` is monotonically non-decreasing.
/// 4. The rendered playlist contains exactly the URIs in the current window —
///    no evicted segments appear in the rendered text.
/// 5. When a longer segment (8 s) arrives at step 10, TARGETDURATION rises
///    to reflect it.
///
/// NOTE: structural/self-consistency check; not mediastreamvalidator output.
#[test]
fn test_live_packaging_rolling_window_all_invariants() {
    const WINDOW: usize = 5;
    let mut updater = ManifestUpdater::new(ManifestType::HlsMedia, WINDOW);
    let mut prev_seq = 0u64;
    let mut prev_target_dur = 0u64;

    for step in 0..20u64 {
        // Inject an extra-long segment at step 10 to exercise TARGETDURATION
        // update.
        let dur_ms = if step == 10 { 8_000 } else { 6_000 };
        updater.add_segment(ManifestSegmentEntry::new(
            step,
            Duration::from_millis(dur_ms),
            format!("live/seg{step:04}.m4s"),
        ));

        let playlist = updater.render_hls_media_playlist();

        // 1. Window size bounded.
        assert!(
            updater.segment_count() <= WINDOW,
            "window must stay ≤ {WINDOW} at step {step} (got {})",
            updater.segment_count()
        );

        // 2. Media sequence non-decreasing.
        let seq = parse_tag_integer(&playlist, "#EXT-X-MEDIA-SEQUENCE:")
            .expect("#EXT-X-MEDIA-SEQUENCE must be present");
        assert!(
            seq >= prev_seq,
            "#EXT-X-MEDIA-SEQUENCE must be non-decreasing: {prev_seq} → {seq} at step {step}"
        );
        prev_seq = seq;

        // 3. TARGETDURATION non-decreasing.
        let td = parse_tag_integer(&playlist, "#EXT-X-TARGETDURATION:")
            .expect("#EXT-X-TARGETDURATION must be present");
        assert!(
            td >= prev_target_dur,
            "#EXT-X-TARGETDURATION must be non-decreasing: {prev_target_dur} → {td} at step {step}"
        );
        prev_target_dur = td;

        // After the 8-second segment arrives, TARGETDURATION must cover it.
        if step >= 10 {
            assert!(
                td >= 8,
                "#EXT-X-TARGETDURATION must be ≥ 8 after the long segment (got {td})"
            );
        }

        // 4. Current-window URIs must appear in the rendered playlist.
        for entry in updater.segments() {
            assert!(
                playlist.contains(&entry.uri),
                "current-window URI {} must appear in playlist at step {step}",
                entry.uri
            );
        }

        // 5. Evicted URIs must NOT appear in the rendered playlist.
        if step >= WINDOW as u64 {
            let evicted_until = step + 1 - WINDOW as u64;
            for evicted_step in 0..evicted_until {
                let uri = format!("live/seg{evicted_step:04}.m4s");
                assert!(
                    !playlist.contains(&uri),
                    "evicted URI {uri} must not appear in playlist at step {step}"
                );
            }
        }
    }

    // After 20 segments with window=5: 15 segments were evicted.
    assert_eq!(
        updater.media_sequence(),
        15,
        "after 20 segments with window=5, media_sequence must be 15"
    );
    assert_eq!(
        updater.segment_count(),
        WINDOW,
        "after filling the window, segment_count must equal window size {WINDOW}"
    );
}
