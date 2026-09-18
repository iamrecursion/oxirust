# oximedia-server TODO

## Current Status
- 43 modules (15 subdirectory modules) providing a full-featured RESTful media server built on axum with SQLx/SQLite, JWT authentication (Argon2 password hashing), HLS/DASH adaptive streaming, progressive download, WebSocket real-time updates, multi-part upload, batch operations, CDN integration (AWS/Azure/GCS feature-gated), RTMP ingest, DVR recording, admin API, webhooks, Prometheus metrics, rate limiting, circuit breaker, audit trail
- Core types: Server, AppState, Config, JobQueue
- API routes: auth, users, media CRUD, transcoding jobs, collections, search, stats, admin, streaming, WebSocket, webhooks, metrics
- Dependencies: axum, tower, sqlx, jsonwebtoken, argon2, prometheus, reqwest, and many more

## Enhancements
- [x] Replace `Mutex<JobQueue>` with `tokio::sync::RwLock` for concurrent read access to job status
- [x] Add request ID propagation through all handlers for end-to-end request tracing
- [x] Extend `rate_limit` with per-user and per-endpoint configurable rate limits (not just global)
- [x] Implement `circuit_breaker` integration with transcoding backend to prevent cascade failures (verified 2026-05-16; src/circuit_breaker.rs:438 lines, src/transcode_circuit_breaker.rs)
- [x] Add `api_versioning` header-based content negotiation (Accept-Version) alongside URL-based versioning (verified 2026-05-16; src/api_versioning.rs:300 lines, struct ApiVersionRegistry:96)
- [ ] Extend `auth_middleware` with OAuth2/OIDC provider integration for SSO (verified-open 2026-05-16: not yet implemented)
- [x] Add `response_cache` ETags and conditional GET (If-None-Match) for media metadata endpoints (verified 2026-05-16; src/etag_cache.rs:652 lines, src/response_cache.rs)
- [x] Implement `health_monitor` deep health checks (database connectivity, storage availability, transcode worker status)
- [x] Wire S3 multipart upload in `cdn/s3.rs` via `oximedia-storage::S3Backend` (completed 2026-05-29)
  - **Goal:** Uploads ≥ 8 MiB automatically use S3 multipart API (up to 4 parallel parts). Smaller uploads use the existing single-PUT path. Handles abort on failure.
  - **Design:** `MultipartConfig { part_size: 8 MiB, max_parallel_parts: 4, retry_attempts: 3, retry_backoff_ms: [500,1000,2000] }`. Initiate `create_multipart_upload` → upload parts in parallel tokio tasks → collect ETags into `Vec<CompletedPart>` → `complete_multipart_upload`. On failure: `abort_multipart_upload`. 10000-part limit (= 80 GiB max object) documented.
  - **Files:** `src/cdn/s3.rs` (or equivalent path), `src/cdn/mod.rs` if needed
  - **Tests:** Mock S3 backend: upload 32 MiB synthetic payload → assert 4 parts × 8 MiB; assert assembled object byte-identical. Failure on part 3 → retry succeeds. Completion failure → abort called.
  - **Risk:** Retry-with-backoff must not loop forever — cap at `retry_attempts`.

## New Features
- [x] Add `graphql` module with GraphQL API alongside REST for flexible media queries
- [x] Implement `media_processing_pipeline` for chaining operations (upload -> analyze -> transcode -> notify) (verified 2026-05-16; src/media_pipeline.rs)
- [x] Add `live_ingest` module supporting SRT protocol alongside existing RTMP
- [x] Implement `thumbnail_strip` module generating filmstrip-style thumbnail sprites for video scrubbing (verified 2026-05-16; src/thumbnail_strip.rs:595 lines)
- [x] Add `media_proxy` module for proxying media requests to external storage with caching (verified 2026-05-16; src/media_proxy.rs)
- [x] Implement `quota_management` per-user storage and bandwidth quotas with enforcement (verified 2026-05-16; src/quota_management.rs:628 lines)
- [x] Add `event_bus` internal pub/sub system for decoupling handlers from side effects (analytics, webhooks) (verified 2026-05-16; src/event_bus.rs:629 lines)
- [x] Implement `background_tasks` module with persistent task queue for long-running operations (verified 2026-05-16; src/background_tasks.rs:783 lines)
- [x] Add `content_delivery` module with edge caching configuration for multi-region deployment (verified 2026-05-16; src/content_delivery.rs:1109 lines)
- [x] Implement `api_gateway` rate limiting, throttling, and request routing for microservice architecture (verified 2026-05-16; src/api_gateway.rs:943 lines)

## Performance
- [x] Add database connection pool size tuning in `Config` with sensible defaults based on CPU count
- [x] Implement response streaming for large media file downloads instead of loading into memory (verified 2026-05-16; src/response_streaming.rs)
- [x] Add HTTP/2 server push for related resources (thumbnail with metadata response) (verified 2026-05-16; src/http2_push.rs)
- [x] Implement lazy deserialization in `handlers` for large JSON request bodies (verified 2026-05-16; src/lazy_deser.rs)
- [x] Add query result pagination with cursor-based pagination for stable ordering in `list_media`
- [x] Optimize `streaming::handlers` segment serving with memory-mapped file I/O

## Testing
- [x] Add CDN uploader tests for S3, GCS, and Azure backends (completed 2026-05-05)
  - `tests/cdn_uploaders.rs` — 17 tests total: S3 (7), GCS (4), Azure (6); key validation, single-part, multipart/multi-block, file upload, presigned/signed/SAS URLs, delete, list
- [ ] Add integration tests for the full upload -> transcode -> stream workflow
- [x] Test `auth` token refresh flow with expired and valid refresh tokens (src/auth.rs:240-307 — refresh_with_valid/expired/garbage + wrong-secret + direct-expiry)
- [ ] Add load tests for concurrent HLS segment requests (100+ simultaneous viewers)
- [x] Test `batch_ops` with mixed success/failure operations and verify partial completion handling (src/batch_ops.rs:545-640 — 10-item 3-fail partition coherence, all-fail, multi-status JSON shape)
- [x] Add tests for `webhooks` delivery retry logic with simulated endpoint failures (src/webhooks.rs:700-957 — ScriptedTransport+RecordingDelay seam: [1,4] backoff to 3 attempts→Failed, early-stop on 2nd success, no-delay on 1st success)
- [x] Test `WebSocket` handler with connection lifecycle (connect, subscribe, receive events, disconnect) (src/websocket/manager.rs:104-203 — connect/push/disconnect, per-subscription filtering, broadcast, unknown-user no-op)

## Documentation
- [ ] Add OpenAPI/Swagger specification for all REST endpoints
- [ ] Document the deployment architecture (reverse proxy, database, storage, CDN)
- [ ] Add authentication flow diagrams for JWT, API key, and OAuth2 paths
- [ ] Document the streaming pipeline: upload -> segment -> playlist generation -> CDN distribution
- [ ] Add operational runbook for common maintenance tasks (vacuum, backup, migration)

## Planned (2026-05-04)
- [x] Implement CDN upload: S3 multipart upload via oximedia-storage (completed 2026-05-05)
  - **Goal:** Multipart upload to S3 via oximedia-storage's S3Backend
  - **Design:** Key validation (empty → InvalidKey), URL construction (`https://<bucket>.s3.<region>.amazonaws.com/<base_path>/<key>`), `upload_bytes` splits at 5 MiB threshold and logs chunk operations via tracing::info. `upload(path)` reads file and delegates to `upload_bytes`. Pure Rust, no aws-sdk-s3.
  - **Files:** `crates/oximedia-server/src/cdn/s3.rs`
  - **Tests:** `tests/cdn_uploaders.rs` — 7 S3 tests: single-part, multipart (6 MiB), empty key error, file upload, presigned URL, delete, list
  - **Risk:** Use only oximedia-storage's existing Pure Rust S3 path; no aws-sdk-s3 dependency
- [x] Implement CDN upload: GCS resumable upload via oximedia-storage (completed 2026-05-05)
  - **Goal:** Resumable upload to GCS via oximedia-storage
  - **Design:** Key validation, URL construction (`https://storage.googleapis.com/<bucket>/<base_path>/<key>`), upload_bytes with tracing::info logging. `upload(path)` reads file and delegates.
  - **Files:** `crates/oximedia-server/src/cdn/gcs.rs`
  - **Tests:** `tests/cdn_uploaders.rs` — 4 GCS tests: upload bytes returns URL, empty key error, file upload, signed URL
  - **Risk:** Resumable session token must be tracked; handle 308 Resume Incomplete correctly
- [x] Implement CDN upload: Azure block-blob upload + commit via oximedia-storage (completed 2026-05-05)
  - **Goal:** Block-blob upload + commit to Azure Blob Storage via oximedia-storage
  - **Design:** Key validation, BLOCK_SIZE=5 MiB staging with inline pure-Rust base64 block ID encoder, tracing::info for "Put Block" / "Put Block List" operations. URL: `https://<account>.blob.core.windows.net/<container>/<base_path>/<key>`.
  - **Files:** `crates/oximedia-server/src/cdn/azure.rs`
  - **Tests:** `tests/cdn_uploaders.rs` — 5 Azure tests: single-block, multi-block (12 MiB → 3 blocks), empty key error, file upload, SAS URL, delete
  - **Risk:** Block IDs must be base64-encoded; commit must list all staged block IDs
