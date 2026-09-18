//! Integration tests for Slice STREAM-ASYNC-PACKAGER.
//!
//! Covers:
//! 1. Async HLS manifest equivalence to sync output.
//! 2. Async LL-HLS playlist serialization.
//! 3. CMAF fMP4 backend dispatch when `enable_cmaf == true`.
//! 4. Raw concatenation regression when `enable_cmaf == false`.
//! 5. `MediaUnitPool` reuse bound under sequential acquire.
//! 6. `MediaUnitPool` concurrency safety under contended acquire/release.
//! 7. `fmp4::build_sidx` re-export linkability.

#[cfg(feature = "async")]
use oximedia_stream::manifest_builder::{
    build_master_playlist, build_media_playlist, HlsManifest, HlsSegment, StreamVariant,
};
use oximedia_stream::media_unit_pool::MediaUnitPool;
use oximedia_stream::stream_packager::{MediaUnit, PackagerConfig, SegmentPackager, StreamType};

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[cfg(feature = "async")]
fn make_variants() -> Vec<StreamVariant> {
    vec![
        StreamVariant {
            bandwidth: 1_500_000,
            resolution: Some((1280, 720)),
            codecs: "av01.0.04M.08".to_string(),
            uri: "stream_1500000.m3u8".to_string(),
            frame_rate: Some(30.0),
        },
        StreamVariant {
            bandwidth: 3_000_000,
            resolution: Some((1920, 1080)),
            codecs: "av01.0.08M.08".to_string(),
            uri: "stream_3000000.m3u8".to_string(),
            frame_rate: Some(30.0),
        },
    ]
}

#[cfg(feature = "async")]
fn make_media_manifest() -> HlsManifest {
    HlsManifest {
        target_duration: 6,
        media_sequence: 7,
        segments: (0..3)
            .map(|i| HlsSegment {
                duration: 6.0,
                uri: format!("seg{i:04}.m4s"),
                byte_range: None,
                discontinuity: false,
                program_date_time: None,
                date_range: None,
            })
            .collect(),
        is_endlist: false,
        allow_cache: false,
        skip: None,
    }
}

fn make_video_keyframe(pts_ms: i64) -> MediaUnit {
    MediaUnit {
        pts_ms,
        dts_ms: pts_ms,
        data: vec![0xAA, 0xBB, 0xCC, 0xDD],
        is_keyframe: true,
        stream_type: StreamType::Video,
    }
}

fn make_video_delta(pts_ms: i64) -> MediaUnit {
    MediaUnit {
        pts_ms,
        dts_ms: pts_ms,
        data: vec![0x11, 0x22, 0x33],
        is_keyframe: false,
        stream_type: StreamType::Video,
    }
}

// ─── 1. HLS manifest async == sync ───────────────────────────────────────────

#[cfg(feature = "async")]
#[test]
fn test_hls_manifest_write_async_equivalent_to_sync() {
    let variants = make_variants();
    let sync_bytes = build_master_playlist(&variants);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()
        .expect("tokio runtime");

    let async_bytes = rt.block_on(async {
        let mut buf: Vec<u8> = Vec::new();
        oximedia_stream::manifest_builder::write_master_playlist_async(&variants, &mut buf)
            .await
            .expect("async write master");
        buf
    });

    assert_eq!(
        sync_bytes.as_bytes(),
        async_bytes.as_slice(),
        "async master playlist bytes must equal sync output"
    );

    // Also verify media playlist parity.
    let manifest = make_media_manifest();
    let sync_media = build_media_playlist(&manifest);
    let async_media = rt.block_on(async {
        let mut buf: Vec<u8> = Vec::new();
        oximedia_stream::manifest_builder::write_media_playlist_async(&manifest, &mut buf)
            .await
            .expect("async write media");
        buf
    });
    assert_eq!(sync_media.as_bytes(), async_media.as_slice());
}

// ─── 2. LL-HLS write_async ───────────────────────────────────────────────────

#[cfg(feature = "async")]
#[test]
fn test_ll_hls_manifest_write_async() {
    use oximedia_stream::ll_hls::{LlHlsConfig, LlHlsPlaylist};

    let config = LlHlsConfig::default();
    let playlist = LlHlsPlaylist::new(config);

    let sync_text = playlist.generate_media_playlist();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()
        .expect("tokio runtime");
    let async_bytes = rt.block_on(async {
        let mut buf: Vec<u8> = Vec::new();
        playlist
            .write_async(&mut buf)
            .await
            .expect("ll-hls async write");
        buf
    });

    assert_eq!(
        sync_text.as_bytes(),
        async_bytes.as_slice(),
        "ll-hls async output must match sync"
    );
    // Spot-check that the output actually contains the expected header.
    assert!(sync_text.starts_with("#EXTM3U"));
}

// ─── 3. CMAF dispatch: enable_cmaf=true emits moof markers ───────────────────

#[test]
fn test_packager_emits_fmp4_when_cmaf_enabled() {
    let cfg = PackagerConfig {
        enable_cmaf: true,
        ..PackagerConfig::default()
    };
    let mut p = SegmentPackager::new(cfg);

    // Feed 3 units into the same in-progress segment.
    p.push(make_video_keyframe(0));
    p.push(make_video_delta(100));
    p.push(make_video_delta(200));

    let seg = p.flush().expect("segment produced on flush");
    assert!(!seg.data.is_empty(), "CMAF segment data must be non-empty");

    // Search for the `moof` box marker.
    let needle = b"moof";
    let found = seg.data.windows(4).any(|w| w == needle);
    assert!(
        found,
        "CMAF-mode segment must contain a moof box marker; data starts with {:02X?}",
        &seg.data.iter().take(16).collect::<Vec<_>>()
    );

    // Should also contain `ftyp` (CMAF segment header).
    let ftyp_found = seg.data.windows(4).any(|w| w == b"ftyp");
    assert!(ftyp_found, "CMAF segment must contain ftyp box marker");
}

// ─── 4. enable_cmaf=false: raw concatenation regression ──────────────────────

#[test]
fn test_packager_emits_ts_when_cmaf_disabled() {
    let cfg = PackagerConfig::default();
    assert!(!cfg.enable_cmaf, "default packager should not enable cmaf");

    let mut p = SegmentPackager::new(cfg);
    p.push(make_video_keyframe(0));
    p.push(make_video_delta(100));

    let seg = p.flush().expect("segment produced on flush");
    // In non-CMAF mode the data is the raw concatenation of unit payloads.
    // Total bytes: 4 (keyframe data) + 3 (delta data) = 7.
    assert_eq!(seg.data.len(), 7);
    assert_eq!(&seg.data[..4], &[0xAA, 0xBB, 0xCC, 0xDD]);
    assert_eq!(&seg.data[4..], &[0x11, 0x22, 0x33]);

    // No CMAF markers should appear.
    let has_moof = seg.data.windows(4).any(|w| w == b"moof");
    assert!(!has_moof, "non-CMAF segment must not contain moof markers");
}

// ─── 5. MediaUnitPool reuses units ───────────────────────────────────────────

#[test]
fn test_media_unit_pool_reuses_units() {
    let pool = MediaUnitPool::new(10);

    // Acquire 100 units sequentially. Each one is dropped immediately,
    // so the pool's free list never exceeds capacity (10).
    for i in 0..100 {
        let mut g = pool.acquire();
        g.unit_mut().pts_ms = i as i64;
        g.unit_mut().data.extend_from_slice(&[i as u8; 32]);
        // drop here recycles back into the pool
    }

    // After 100 acquire/release cycles, pool.len() must be bounded by capacity.
    assert!(
        pool.len() <= 10,
        "pool length {} exceeded capacity 10",
        pool.len()
    );
    // And at least one unit should have been recycled.
    assert!(
        pool.len() > 0,
        "pool should retain at least one recycled unit"
    );
}

// ─── 6. MediaUnitPool concurrent acquire/release ─────────────────────────────

#[test]
fn test_media_unit_pool_concurrent_acquire_release() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    let pool = Arc::new(MediaUnitPool::new(16));
    let success_count = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::with_capacity(8);
    for tid in 0..8usize {
        let p = Arc::clone(&pool);
        let sc = Arc::clone(&success_count);
        handles.push(thread::spawn(move || {
            for i in 0..50 {
                let mut g = p.acquire();
                let u = g.unit_mut();
                u.pts_ms = (tid * 1_000 + i) as i64;
                u.data.extend_from_slice(&[(tid as u8); 8]);
                // Implicit drop returns to pool.
            }
            sc.fetch_add(1, Ordering::SeqCst);
        }));
    }

    // No deadlock allowed; if join hangs the test runner times out.
    for h in handles {
        h.join().expect("thread joined cleanly");
    }
    assert_eq!(
        success_count.load(Ordering::SeqCst),
        8,
        "all 8 threads must have completed"
    );
    // Pool length must remain bounded by capacity even after heavy contention.
    assert!(pool.len() <= 16);
}

// ─── 7. SIDX re-export compiles and produces a sidx box ──────────────────────

#[test]
fn test_sidx_reexport_compiles() {
    let bytes = oximedia_stream::fmp4::build_sidx(
        /* reference_id        */ 1, /* timescale           */ 90_000,
        /* earliest_pts        */ 0, /* referenced_size     */ 2_048,
        /* subsegment_duration */ 90_000, /* is_sap              */ true,
    );

    // Output must be a full ISO BMFF box: 4-byte size + b"sidx" + body.
    assert!(
        bytes.len() >= 8,
        "sidx must include at least 8 bytes of header"
    );
    assert_eq!(&bytes[4..8], b"sidx", "fourcc must be sidx");
    // Size field should equal total length.
    let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(
        size as usize,
        bytes.len(),
        "size field must match total length"
    );
}
