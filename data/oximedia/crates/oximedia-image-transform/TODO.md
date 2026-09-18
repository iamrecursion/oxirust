# oximedia-image-transform TODO

## Current Status
- 8 modules implementing Cloudflare Images-compatible URL image transformation
- `transform.rs`: Strongly-typed params (resize, crop, quality, format, rotation, fit modes, border, padding, trim)
- `parser.rs`: Parse `/cdn-cgi/image/` paths, query strings, comma-separated transform strings
- `negotiation.rs`: Accept-header format negotiation (AVIF > WebP > JPEG/PNG) + client hints (DPR, Width, Viewport-Width)
- `processor.rs`: Image processing execution pipeline
- `quality.rs`: SSIM-guided quality auto-tuning with complexity analysis
- `security.rs`: Request validation, SSRF prevention, HMAC-SHA256 signed URL verification
- `watermark.rs`: Text and image watermark overlay with 9 positions, opacity, scaling
- `face_detect.rs`: Skin-tone + saliency-based face detection for smart gravity cropping
- Dependencies: thiserror only (lightweight crate)

## Enhancements
- [x] Add signed URL support to `security.rs` with HMAC-SHA256 verification to prevent abuse
- [x] Extend `negotiation.rs` with client-hints support (DPR, Viewport-Width, Width headers)
- [x] Add quality auto-tuning in `quality.rs` based on image complexity (SSIM-guided)
- [x] Implement progressive JPEG output option in `transform.rs` for faster perceived loading
- [x] Add cache-control header generation to `negotiation.rs` based on transform parameters
- [x] Extend `parser.rs` with named preset support (e.g., `/cdn-cgi/image/preset=thumbnail/photo.jpg`)
- [x] Add aspect ratio preservation enforcement in `transform.rs` when both width and height are set — `enforce_aspect_ratio(src_w, src_h, req_w, req_h, fit_mode)` added; handles Contain/ScaleDown (letterbox) and Cover/Crop (fill) modes with 10 unit tests

## New Features
- [x] Add a `watermark.rs` module for image watermark overlay (text and image) with configurable position and opacity
- [x] Implement a `face_detect.rs` module for smart cropping based on face detection
- [x] Add a `blur_region.rs` module for selective region blurring (privacy, NSFW)
- [x] Implement a `responsive.rs` module for srcset/picture element generation with multiple breakpoints (verified 2026-05-16; src/responsive.rs:74 Breakpoint, 569 lines)
- [x] Add an `animation.rs` module for GIF/WebP animation resize and optimization (verified 2026-05-16; src/animation.rs:591 lines)
- [x] Implement an `origin_fetch.rs` module for fetching source images from remote URLs with caching (verified 2026-05-16; src/origin_fetch.rs:59 FetchResponse, OriginFetcher, 512 lines)
- [x] Add a `metrics.rs` module for tracking transform request statistics (hit rate, latency, format distribution) (verified 2026-05-16; src/metrics.rs:58 TransformMetrics, 529 lines)
- [x] Implement a `batch_transform.rs` module for processing multiple transform variants in a single request (verified 2026-05-16; src/batch_transform.rs:137 BatchTransformRequest, 548 lines)
- [x] Add an `image_analysis.rs` module for dominant color extraction and blur hash generation (verified 2026-05-16; src/image_analysis.rs:19 BlurHashEncoder, DominantColorExtractor, 668 lines)

## Performance
- [x] Add response caching layer with content-addressable storage keyed on transform params hash — `src/response_cache.rs`: `ResponseCache` (FIFO eviction, LRU hit-count, content-address key via `DefaultHasher`); 15 unit tests
- [x] Implement streaming transform pipeline to avoid loading entire source image into memory — `src/processor/streaming.rs`: `StreamingProcessor` + `StreamingConfig` (tile_rows/overlap_rows); bilinear scale per tile strip; 10 unit tests
- [x] Add early termination in `processor.rs` when output dimensions are smaller than a threshold
- [x] Implement parallel transform execution for batch requests — `batch_transform.rs` `process_batch` now uses `rayon::par_iter`

## Testing
- [x] Add parser fuzz tests for malformed `/cdn-cgi/image/` URLs (empty params, invalid values, injection attempts) — `tests/parser_fuzz.rs`: 40 tests covering empty/blank, missing values, integer overflow, shell injection, path traversal, unicode, extreme lengths, boundary values
- [ ] Test `negotiation.rs` with all common browser Accept header combinations
- [x] Add round-trip tests: parse transform string -> serialize -> parse again -> compare — Wave 30, 2026-06-08: canonical `TransformParams::serialize()` (fixed field order, defaults omitted, true parser inverse; comma-free `x` geometry form so 4-side trim/border/pad survive the top-level comma split) + parse↔serialize idempotency suite in `tests/roundtrip_transform.rs`; parser extended with `progressive`/`output_quality` keys and `split_sides` helper; 30 tests
- [x] Test `security.rs` with path traversal attempts, oversized dimensions, and resource exhaustion vectors
- [ ] Add integration tests with actual image data through the full parse -> process pipeline
- [x] Test edge cases: zero width, negative quality, unknown format strings — Wave 30, 2026-06-08: edge-case suite in `tests/roundtrip_transform.rs` asserts defined behaviour without panic (zero width parses then fails `validate()`; negative/over-u8 quality → parse `Err`; in-range-but-invalid quality → validation `Err`; unknown format → parse `Err`; non-numeric dims, empty/bare/trailing-comma tokens, out-of-range focal point)

## Documentation
- [ ] Document all supported transform parameters with examples and default values
- [ ] Add a compatibility matrix showing which Cloudflare Images features are supported
- [ ] Document the content negotiation priority rules and fallback behavior
