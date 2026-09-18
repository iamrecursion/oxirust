//! End-to-end live streaming pipeline walkthrough.
//!
//! This example demonstrates how the major components of `oximedia-stream` fit
//! together in a typical live-streaming workflow:
//!
//! 1. Quality ladder selection via [`AdaptivePipeline`]
//! 2. Segment lifecycle management via [`SegmentManager`]
//! 3. HLS master and media manifest generation via the manifest builder
//! 4. Stream health tracking via [`StreamHealthMonitor`]
//!
//! No actual network I/O or real media files are used; all data is synthetic.

use oximedia_stream::adaptive_pipeline::{AdaptivePipeline, QualityLadder, SwitchReason};
use oximedia_stream::manifest_builder::{
    build_dash_mpd, build_master_playlist, build_media_playlist, DashMpd, DashRepresentation,
    HlsManifest, HlsSegment, StreamVariant,
};
use oximedia_stream::segment_manager::SegmentManager;
use oximedia_stream::stream_health::StreamHealthMonitor;

fn main() {
    println!("=== OxiMedia Live Streaming Pipeline Demo ===\n");

    // ── 1. Adaptive pipeline ──────────────────────────────────────────────────
    let ladder = QualityLadder::default_ladder();
    println!(
        "Quality ladder has {} tiers (lowest → highest bitrate):",
        ladder.tiers.len()
    );
    for (i, tier) in ladder.tiers.iter().enumerate() {
        println!(
            "  [{i}] {name} — {vbr}+{abr} kbps  (min bw {min} kbps)",
            name = tier.name,
            vbr = tier.video_bitrate_kbps,
            abr = tier.audio_bitrate_kbps,
            min = tier.min_bandwidth_kbps,
        );
    }

    let mut pipeline = AdaptivePipeline::new(ladder);
    // Simulate measuring a 3 000 kbps download
    pipeline.record_download(3_000_000, 8.0); // 3 MB in 8 s → 3 000 kbps
    pipeline.update_buffer(20.0); // buffer at 20 s
                                  // Bypass cooldown for demo purposes
    pipeline.upgrade_cooldown_secs = 0.0;
    pipeline.downgrade_cooldown_secs = 0.0;

    match pipeline.evaluate_switch() {
        Some(sw) => println!(
            "\nABR switch: tier {} → {} (reason: {:?})",
            sw.from_tier, sw.to_tier, sw.reason
        ),
        None => println!("\nABR: no switch needed at current conditions"),
    }

    // Force a known tier for demo segment creation
    pipeline
        .force_tier(2, SwitchReason::UserRequested)
        .expect("tier 2 exists in default ladder");
    let current = pipeline.ladder.current();
    println!(
        "Active tier: {} ({}×{} @ {}+{} kbps)",
        current.name,
        current.width,
        current.height,
        current.video_bitrate_kbps,
        current.audio_bitrate_kbps,
    );

    // ── 2. Segment manager ────────────────────────────────────────────────────
    println!("\n--- Segment Manager ---");
    let mut seg_mgr = SegmentManager::new(30, 3);

    // Create 5 synthetic 6-second segments
    let tier_name = current.name.clone();
    let mut seg_ids = Vec::new();
    for i in 0..5u64 {
        let id = seg_mgr.create_segment(
            i * 6000, // pts_ms
            6000,     // duration_ms
            750_000,  // ~750 KB
            &tier_name,
        );
        seg_ids.push(id);
    }

    // Simulate download: transition Pending → Downloading → Available
    for id in &seg_ids {
        seg_mgr.mark_downloading(id);
        seg_mgr.mark_available(id, 3_000.0); // 3 000 kbps
    }

    let avail = seg_mgr.available_segments();
    println!("Available segments: {}", avail.len());
    for seg in &avail {
        println!(
            "  seq={} pts={}ms dur={}ms tier={}",
            seg.sequence_number, seg.pts_start_ms, seg.duration_ms, seg.tier_name,
        );
    }

    // ── 3. Manifest generation ────────────────────────────────────────────────
    println!("\n--- HLS Master Playlist (first 20 lines) ---");
    let variants = vec![
        StreamVariant {
            bandwidth: 3_200_000,
            resolution: Some((1280, 720)),
            codecs: "av01.0.05M.08".to_string(),
            uri: "720p/playlist.m3u8".to_string(),
            frame_rate: Some(25.0),
        },
        StreamVariant {
            bandwidth: 1_100_000,
            resolution: Some((854, 480)),
            codecs: "av01.0.04M.08".to_string(),
            uri: "480p/playlist.m3u8".to_string(),
            frame_rate: Some(25.0),
        },
        StreamVariant {
            bandwidth: 500_000,
            resolution: Some((640, 360)),
            codecs: "av01.0.04M.08".to_string(),
            uri: "360p/playlist.m3u8".to_string(),
            frame_rate: Some(25.0),
        },
    ];
    let master = build_master_playlist(&variants);
    for line in master.lines().take(20) {
        println!("{line}");
    }

    println!("\n--- HLS Media Playlist (first 20 lines) ---");
    let media_manifest = HlsManifest {
        target_duration: 6,
        media_sequence: 0,
        segments: avail
            .iter()
            .map(|s| HlsSegment {
                duration: s.duration_ms as f64 / 1000.0,
                uri: format!("seg{:03}.m4s", s.sequence_number),
                byte_range: None,
                discontinuity: false,
                program_date_time: None,
                date_range: None,
            })
            .collect(),
        is_endlist: false,
        allow_cache: false,
        skip: None,
    };
    let media = build_media_playlist(&media_manifest);
    for line in media.lines().take(20) {
        println!("{line}");
    }

    println!("\n--- DASH MPD (first 20 lines) ---");
    let dash_mpd = DashMpd {
        min_buffer_time_ms: 2000,
        representations: variants
            .iter()
            .map(|v| DashRepresentation {
                id: v.uri.split('/').next().unwrap_or("rep").to_string(),
                bandwidth: v.bandwidth,
                width: v.resolution.map(|(w, _)| w).unwrap_or(0),
                height: v.resolution.map(|(_, h)| h).unwrap_or(0),
                codec: v.codecs.clone(),
                base_url: v.uri.clone(),
                segment_template: None,
            })
            .collect(),
    };
    let mpd_xml = build_dash_mpd(&dash_mpd);
    for line in mpd_xml.lines().take(20) {
        println!("{line}");
    }

    // ── 4. Stream health monitoring ───────────────────────────────────────────
    println!("\n--- Stream Health Monitor ---");
    let mut health = StreamHealthMonitor::new(100);
    health.start_session();
    health.first_frame();

    // Simulate a stable 30-second session with minor hiccups
    health.record_playback_duration(30_000);
    health.record_rebuffer_with_duration(200); // 200 ms stall
    health.record_quality_switch();
    health.record_dropped_frames(5);

    let report = health.report(
        current.video_bitrate_kbps as f64 + current.audio_bitrate_kbps as f64,
        20.0,
    );
    println!("Overall QoE score:  {:.1}/100", report.overall_score);
    println!("Rebuffer events:    {}", report.rebuffer_events);
    println!("Quality switches:   {}", report.quality_switches);
    println!("Dropped frames:     {}", report.dropped_frames);
    println!(
        "Rebuffer ratio:     {:.2}%",
        health.rebuffer_ratio() * 100.0
    );

    if report.issues.is_empty() {
        println!("Health issues:      none");
    } else {
        println!("Health issues:");
        for issue in &report.issues {
            println!("  - {:?}", issue);
        }
    }

    println!("\nPipeline demo complete.");
}
