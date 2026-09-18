//! Integration tests for the HLS and DASH manifest validator.

use oximedia_stream::manifest_builder::{
    build_dash_mpd, build_master_playlist, build_media_playlist, DashMpd, DashRepresentation,
    HlsManifest, HlsSegment, StreamVariant,
};
use oximedia_stream::validator::{validate_dash, validate_hls, ValidationError};

// ─── Helper constructors ──────────────────────────────────────────────────────

fn make_variants() -> Vec<StreamVariant> {
    vec![
        StreamVariant {
            bandwidth: 800_000,
            resolution: Some((1280, 720)),
            codecs: "av01.0.05M.08".to_string(),
            uri: "720p/playlist.m3u8".to_string(),
            frame_rate: Some(25.0),
        },
        StreamVariant {
            bandwidth: 400_000,
            resolution: Some((640, 360)),
            codecs: "av01.0.04M.08".to_string(),
            uri: "360p/playlist.m3u8".to_string(),
            frame_rate: Some(25.0),
        },
    ]
}

fn make_media_manifest() -> HlsManifest {
    HlsManifest {
        target_duration: 6,
        media_sequence: 0,
        segments: vec![
            HlsSegment {
                duration: 6.0,
                uri: "seg000.m4s".to_string(),
                byte_range: None,
                discontinuity: false,
                program_date_time: None,
                date_range: None,
            },
            HlsSegment {
                duration: 6.0,
                uri: "seg001.m4s".to_string(),
                byte_range: None,
                discontinuity: false,
                program_date_time: None,
                date_range: None,
            },
        ],
        is_endlist: false,
        allow_cache: false,
        skip: None,
    }
}

fn make_dash_mpd() -> DashMpd {
    DashMpd {
        min_buffer_time_ms: 2000,
        representations: vec![
            DashRepresentation {
                id: "720p".to_string(),
                bandwidth: 800_000,
                width: 1280,
                height: 720,
                codec: "av01.0.05M.08".to_string(),
                base_url: "720p/".to_string(),
                segment_template: None,
            },
            DashRepresentation {
                id: "360p".to_string(),
                bandwidth: 400_000,
                width: 640,
                height: 360,
                codec: "av01.0.04M.08".to_string(),
                base_url: "360p/".to_string(),
                segment_template: None,
            },
        ],
    }
}

// ─── HLS tests ────────────────────────────────────────────────────────────────

#[test]
fn test_hls_master_playlist_valid() {
    let variants = make_variants();
    let playlist = build_master_playlist(&variants);
    validate_hls(&playlist).expect("master playlist built by build_master_playlist must be valid");
}

#[test]
fn test_hls_media_playlist_valid() {
    let manifest = make_media_manifest();
    let playlist = build_media_playlist(&manifest);
    validate_hls(&playlist).expect("media playlist built by build_media_playlist must be valid");
}

#[test]
fn test_hls_missing_extm3u_rejected() {
    let err = validate_hls("GARBAGE\n#EXT-X-TARGETDURATION:6\n")
        .expect_err("playlist without #EXTM3U must be rejected");
    assert_eq!(
        err,
        ValidationError::HlsMissingExtm3u,
        "expected HlsMissingExtm3u, got {:?}",
        err
    );
}

#[test]
fn test_hls_skip_without_version9_rejected() {
    // Build a playlist string with #EXT-X-SKIP but #EXT-X-VERSION:3 (not 9)
    let playlist = concat!(
        "#EXTM3U\n",
        "#EXT-X-VERSION:3\n",
        "#EXT-X-TARGETDURATION:6\n",
        "#EXT-X-MEDIA-SEQUENCE:0\n",
        "#EXT-X-SKIP:SKIPPED-SEGMENTS=3\n",
        "#EXTINF:6.000000,\n",
        "seg003.m4s\n",
    );
    let err = validate_hls(playlist)
        .expect_err("playlist with #EXT-X-SKIP and VERSION:3 must be rejected");
    assert_eq!(
        err,
        ValidationError::HlsSkipRequiresVersion9,
        "expected HlsSkipRequiresVersion9, got {:?}",
        err
    );
}

// ─── DASH tests ───────────────────────────────────────────────────────────────

#[test]
fn test_dash_mpd_valid() {
    let mpd = make_dash_mpd();
    let xml = build_dash_mpd(&mpd);
    validate_dash(&xml).expect("MPD built by build_dash_mpd must be valid");
}

#[test]
fn test_dash_missing_adaptationset_rejected() {
    // Minimal MPD XML with Period but no AdaptationSet
    let xml = concat!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"dynamic\">\n",
        "  <Period>\n",
        "  </Period>\n",
        "</MPD>\n",
    );
    let err = validate_dash(xml).expect_err("MPD without AdaptationSet must be rejected");
    assert_eq!(
        err,
        ValidationError::DashMissingAdaptationSet,
        "expected DashMissingAdaptationSet, got {:?}",
        err
    );
}
