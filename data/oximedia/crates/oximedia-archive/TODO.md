# oximedia-archive TODO

## Current Status
- 30 source files across modules: `archive_verify`, `asset_manifest`, `audit_trail`, `batch_archive`, `catalog`, `catalog_export`, `checksum` (sqlite-gated), `dedup_archive`, `fixity` (sqlite-gated), `format_registry`, `indexing`, `ingest_log`, `integrity_scan`, `migration`, `preservation`, `preservation_policy`, `quarantine` (sqlite-gated), `report` (sqlite-gated), `restore_plan`, `retention_schedule`, `search_index`, `split_archive`, `streaming_compress`, `tape`, `validate`, `version_history`
- Multi-algorithm checksumming (BLAKE3, SHA-256, MD5, CRC32)
- Async `ArchiveVerifier` with SQLite-backed fixity checking and PREMIS event logging
- Parallel verification, auto-quarantine, BagIt support
- Dependencies: blake3, sha2, md-5, crc32fast, sqlx (optional), chrono, tokio, rayon, regex, csv

## Enhancements
- [x] Implement true parallel file verification in `verify_files` using tokio/rayon (currently sequential await loop) (verified 2026-05-16; src/parallel_checksum.rs:11 ParallelChecksumConfig, compute_parallel_batch:187 rayon file-level parallelism, compute_parallel:69)
- [x] Add incremental checksumming in `checksum` module (resume interrupted verification) (verified 2026-05-16; src/incremental_checksum.rs:24 IncrementalCheckpoint, checkpoint_interval_bytes:97, was_resumed:126, IncrementalChecksummer:85)
- [x] Implement configurable quarantine policies in `quarantine` (max size, auto-cleanup after N days) (verified 2026-05-16; src/quarantine_policy.rs:18 QuarantinePolicy, auto_cleanup_after_days:27, max_size:18, check_admission:54, cleanup_candidates:340)
- [x] Add `catalog_export` CSV and JSON export for archive manifests (verified 2026-05-16; src/catalog_export.rs:126 CatalogExporter, ExportFormat::Csv:25, NdJson:26, to_csv_row:88)
- [x] Implement hierarchical catalog organization in `catalog` (collections, sub-collections)
- [x] Add file format identification (magic bytes) in `validate` beyond extension checking
- [x] Implement `search_index` full-text search over metadata fields (verified 2026-05-16; src/search_index.rs:117 SearchIndex, SearchQuery:62, search:194, BoolOp And/Or field-scoped, 283 lines)
- [x] Add `retention_schedule` enforcement: automatic deletion/archival when retention period expires
- [x] Improve `migration` module with dry-run support and rollback capability
- [x] Add sidecar file generation with human-readable checksum manifests

## New Features
- [x] Add cloud storage backend support (S3-compatible) for `archive_verify` remote verification (verified 2026-05-16; src/cloud_backend.rs:404 RemoteVerificationResult, verify_remote_files:430)
- [x] Implement LTFS (Linear Tape File System) metadata support in `tape` module (verified 2026-05-16; src/ltfs.rs:46 LtfsVolumeLabel, LTFS_FORMAT_VERSION:34 v2.5.0, SNIA LTFSv2:21)
- [x] Add BagIt profile validation (verify bag meets specific profile requirements) (verified 2026-05-16; src/bagit_profile.rs:26 BagItProfile, validate_bag_against_profile:329, parse_profile:455)
- [x] Implement media-specific validation in `validate` (check MKV/FLAC/PNG structure integrity) (verified 2026-05-16; src/media_validate.rs:161 validate_mkv EBML magic, validate_flac:231, validate_png:333, magic-byte detection:111)
- [x] Add deduplication reporting in `dedup_archive` (space savings summary, duplicate file list) (verified 2026-05-16; src/dedup_report.rs:185 DedupReport, DedupStats:59, DuplicateGroup:24, DedupAnalyzer:106, build_report:129)
- [x] Implement notification system for fixity check failures (webhook/email integration points) (verified 2026-05-16; src/notification.rs:62 NotificationEvent, NotificationChannel:158 trait, WebhookChannel:229, NotificationDispatcher:400)
- [x] Add archive health dashboard data generation (summary stats, trend over time)
- [x] Implement `split_archive` with configurable split strategies (by size, date, collection)

## Performance
- [x] Use memory-mapped I/O for checksum computation on large files in `checksum` (verified 2026-06-04; src/mmap_checksum.rs:75 hash_via_mmap hashes &mmap[..] slice directly without to_vec)
- [x] Implement streaming hash computation (process file in chunks) to reduce memory footprint (verified 2026-06-04; src/checksum.rs:521 compute_checksums_sync chunked BufReader loop, no full-file allocation)
- [x] Add parallel checksum computation (compute BLAKE3 + SHA-256 + CRC32 simultaneously per file)
- [x] Optimize `integrity_scan` with file modification time checks to skip unchanged files (verified 2026-06-05; src/integrity_scan.rs:353 ScanCache, scan:406 skips files whose (size, mtime_ms) match cached FileScanRecord via matches_metadata:152, re-hashes changed/new via compute_checksums_mmap, rehash_count/skip_count reset per call; FileScanRecord::with_mtime:143 builder + mtime_ms field:95)
- [x] Use connection pooling and batch SQL inserts in `fixity` for high-volume verification (verified 2026-06-04; src/fixity.rs flush_fixity_rows batches N rows per sqlx transaction, FIXITY_BATCH_SIZE=100)
- [ ] Add `streaming_compress` with configurable buffer sizes for throughput tuning

## Testing
- [ ] Add round-trip test: ingest file -> compute checksums -> verify -> confirm match
- [ ] Test `quarantine` workflow: corrupt file -> auto-detect -> quarantine -> restore
- [ ] Test `fixity` scheduled checking with mock clock (verify interval enforcement)
- [ ] Add test for `batch_archive` with 100+ synthetic files
- [ ] Test `catalog` operations: add/remove/search/export
- [ ] Test `version_history` with multiple versions of the same file (verify diff tracking)

## Documentation
- [ ] Document sqlite vs non-sqlite feature gating and which modules require database
- [ ] Add OAIS reference model diagram mapping modules to OAIS functional entities
- [ ] Document preservation workflow: ingest -> checksum -> validate -> catalog -> fixity schedule
