//! Smoke tests for newly-wired orphan modules in `oximedia-io`.

// ── async_reader ──────────────────────────────────────────────────────────────
#[test]
fn async_reader_open_missing_file_errors() {
    use oximedia_io::async_reader::BufferedAsyncReader;
    let result = BufferedAsyncReader::new("/dev/null/no_such_file.mp4", 4096);
    // Missing file must return an error rather than panic.
    assert!(result.is_err());
}

// ── progress ──────────────────────────────────────────────────────────────────
#[test]
fn progress_percent_advance() {
    use oximedia_io::progress::FileProgress;
    let mut p = FileProgress::new(1_000_000);
    p.advance(250_000);
    let pct = p.percent();
    assert!((pct - 25.0_f32).abs() < 0.01, "expected 25%, got {pct}");
    assert!(!p.is_complete());
    p.advance(750_000);
    assert!(p.is_complete());
    assert_eq!(p.remaining_bytes(), 0);
}

#[test]
fn progress_eta_unavailable_when_no_bytes() {
    use oximedia_io::progress::FileProgress;
    let p = FileProgress::new(1_000_000);
    assert!(p.eta_secs(1.0).is_none());
}

// ── crc_stream ────────────────────────────────────────────────────────────────
#[test]
fn crc_stream_known_crc32() {
    use oximedia_io::crc_stream::{Crc32Stream, Crc32Table};
    let _table = Crc32Table::new();
    let mut stream = Crc32Stream::new();
    let data = b"hello world";
    stream.update(data);
    assert_eq!(stream.bytes_processed(), data.len() as u64);
    // IEEE CRC-32 of "hello world" = 0x0D4A_1185
    assert_eq!(stream.value(), 0x0D4A_1185u32);
}

#[test]
fn crc_stream_reset_clears_state() {
    use oximedia_io::crc_stream::Crc32Stream;
    let mut stream = Crc32Stream::new();
    stream.update(b"abc");
    stream.reset();
    assert_eq!(stream.bytes_processed(), 0);
    assert_eq!(stream.value(), 0);
}

// ── watcher ──────────────────────────────────────────────────────────────────
#[test]
fn watcher_nonexistent_file_does_not_exist() {
    use oximedia_io::watcher::FileWatcher;
    let tmp_path = std::env::temp_dir().join("oximedia_orphan_smoke_nonexistent_99999.mp4");
    let w = FileWatcher::new(&tmp_path.to_string_lossy());
    assert!(!w.exists());
    let _result = w.poll(0);
}

// ── dedup_writer ─────────────────────────────────────────────────────────────
#[test]
fn dedup_writer_stats_default_zero() {
    use oximedia_io::dedup_writer::DedupStats;
    let stats = DedupStats::default();
    assert_eq!(stats.bytes_saved(), 0);
}

// ── io_metrics ───────────────────────────────────────────────────────────────
#[test]
fn io_metrics_initial_zero() {
    use oximedia_io::io_metrics::IoMetrics;
    let m = IoMetrics::new();
    assert_eq!(m.read_bytes, 0);
    assert_eq!(m.write_bytes, 0);
    assert_eq!(m.total_errors(), 0);
    assert_eq!(m.total_bytes(), 0);
}

// ── sparse_file ───────────────────────────────────────────────────────────────
#[test]
fn sparse_map_empty_zero_regions() {
    use oximedia_io::sparse_file::SparseMap;
    let map = SparseMap::new(0, vec![]);
    assert_eq!(map.region_count(), 0);
    assert_eq!(map.file_size(), 0);
}

#[test]
fn sparse_region_data_vs_hole() {
    use oximedia_io::sparse_file::SparseRegion;
    let data_region = SparseRegion::Data {
        offset: 0,
        length: 1024,
    };
    let hole_region = SparseRegion::Hole {
        offset: 1024,
        length: 4096,
    };
    assert!(data_region.is_data());
    assert!(!data_region.is_hole());
    assert!(hole_region.is_hole());
    assert!(!hole_region.is_data());
    assert_eq!(data_region.length(), 1024);
    assert_eq!(hole_region.offset(), 1024);
    assert_eq!(hole_region.end(), 5120);
}

#[test]
fn sparse_map_dense_single_region() {
    use oximedia_io::sparse_file::SparseMap;
    let map = SparseMap::dense(4096);
    assert_eq!(map.region_count(), 1);
    assert_eq!(map.data_bytes(), 4096);
    assert_eq!(map.hole_bytes(), 0);
}

// ── parallel_copy ─────────────────────────────────────────────────────────────
#[test]
fn parallel_copy_config_default_valid() {
    use oximedia_io::parallel_copy::ParallelCopyConfig;
    let cfg = ParallelCopyConfig::default();
    assert!(cfg.validate().is_ok());
}

#[test]
fn copy_progress_fraction() {
    use oximedia_io::parallel_copy::CopyProgress;
    let p = CopyProgress {
        total_bytes: 1000,
        copied_bytes: 500,
        chunks_completed: 1,
        total_chunks: 2,
        finished: false,
    };
    let frac = p.fraction();
    assert!((frac - 0.5).abs() < 1e-9);
    assert_eq!(p.percentage(), 50);
}

// ── prefetch ─────────────────────────────────────────────────────────────────
#[test]
fn prefetcher_initial_state() {
    use oximedia_io::prefetch::Prefetcher;
    let p = Prefetcher::with_defaults();
    assert_eq!(p.sequential_count(), 0);
    assert!(!p.is_active());
    assert_eq!(p.total_requests(), 0);
}

// ── pipe_source ──────────────────────────────────────────────────────────────
#[test]
fn pipe_source_stdin_not_seekable() {
    use oximedia_io::pipe_source::PipeSource;
    let src = PipeSource::from_stdin();
    assert!(!src.is_seekable());
    assert_eq!(src.position(), 0);
}

// ── chunked_upload ───────────────────────────────────────────────────────────
#[test]
fn chunked_upload_split_and_reassemble() {
    use oximedia_io::chunked_upload::{ChunkedUploadReassembler, ChunkedUploadSplitter};
    let data = b"Hello, OxiMedia chunked upload test!";
    let splitter = ChunkedUploadSplitter::new(8);
    let chunks = splitter.split(data);
    assert!(!chunks.is_empty());
    assert_eq!(splitter.chunk_size(), 8);
    let reassembled = ChunkedUploadReassembler::reassemble(&chunks);
    assert_eq!(&reassembled, data);
}

// ── http_source ───────────────────────────────────────────────────────────────
#[test]
fn http_byte_range_header_bounded() {
    use oximedia_io::http_source::ByteRange;
    let r = ByteRange::new(0, 1023);
    assert_eq!(r.to_header_value(), "bytes=0-1023");
    assert_eq!(r.length(), Some(1024));
}

#[test]
fn http_byte_range_header_open_ended() {
    use oximedia_io::http_source::ByteRange;
    let r = ByteRange::from_offset(1024);
    assert_eq!(r.to_header_value(), "bytes=1024-");
    assert!(r.length().is_none());
}

#[test]
fn http_response_info_status_codes() {
    use oximedia_io::http_source::HttpResponseInfo;
    let ok = HttpResponseInfo {
        status: 200,
        content_length: Some(4096),
        accepts_ranges: false,
        content_type: None,
        etag: None,
    };
    assert!(ok.is_success());
    assert!(!ok.is_partial_content());

    let partial = HttpResponseInfo {
        status: 206,
        content_length: Some(1024),
        accepts_ranges: true,
        content_type: None,
        etag: None,
    };
    assert!(partial.is_success());
    assert!(partial.is_partial_content());
}

// ── s3_source ─────────────────────────────────────────────────────────────────
#[test]
fn s3_byte_range_http_header() {
    use oximedia_io::s3_source::ByteRange;
    // S3 ByteRange uses exclusive end: [0, 1024) → length=1024, header "bytes=0-1023"
    let r = ByteRange::new(0, 1024);
    let hdr = r.http_header();
    assert!(hdr.starts_with("bytes="), "unexpected header: {hdr}");
    assert_eq!(r.length(), 1024);
}

#[test]
fn s3_source_object_url_format() {
    use oximedia_io::s3_source::{S3Config, S3Source};
    let mut cfg = S3Config::default();
    cfg.bucket = "my-bucket".to_string();
    cfg.path_style = true;
    let src = S3Source::new(cfg, "video/test.mp4");
    let url = src.object_url();
    assert!(url.contains("my-bucket"), "url: {url}");
    assert!(url.contains("video/test.mp4"), "url: {url}");
    assert_eq!(src.key(), "video/test.mp4");
    assert_eq!(src.bucket(), "my-bucket");
}

// ── multipart_writer ──────────────────────────────────────────────────────────
#[test]
fn multipart_writer_config_default_positive() {
    use oximedia_io::multipart_writer::MultipartConfig;
    let cfg = MultipartConfig::default();
    assert!(cfg.part_size > 0);
    assert!(cfg.max_concurrency > 0);
    assert!(cfg.cleanup_on_drop);
}
