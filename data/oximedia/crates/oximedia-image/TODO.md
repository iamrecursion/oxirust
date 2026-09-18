# oximedia-image TODO

## Current Status
- 40+ modules for professional image sequence I/O and processing
- Formats: DPX (SMPTE 268M), OpenEXR, TIFF/BigTIFF, DNG (demosaic, parser, reader, writer)
- Image processing: convolution, edge_detect, morphology, advanced_morphology, blend_mode, color_adjust, color_balance, color_science, segmentation, inpaint/inpainting, stitch, texture_synthesis, mosaic
- Frequency domain: frequency_domain, image_pyramid, pyramid
- Noise: noise_estimation, noise_gen
- Color/tone: tone_curve, histogram_ops, depth_map, lens_correct
- Pipeline: pixel_pipeline, filter, filters, format_detect, sequence (SequencePattern), thumbnail_cache
- Metadata: exif_parser, metadata_xmp, icc_embed, raw, raw_decode
- Pixel types: U8, U10, U12, U16, U32, F16, F32; 9 color spaces; 12 compression methods
- Dependencies: oximedia-core, byteorder, half, tiff, flate2, oxiarc-deflate, weezl, rayon

## Enhancements
- [x] Add multi-layer/multi-part support to `exr.rs` for compositing workflows (read/write individual layers) (verified-open 2026-05-16: no MultiLayer/MultiPart in exr.rs)
- [x] Extend `dpx.rs` with 10-bit packed pixel read/write (Method A and Method B packing) (verified 2026-05-16; src/dpx.rs:49 PackingMethod::MethodA/MethodB, 10-bit MSB/LSB packing:1043)
- [x] Implement full TIFF tag roundtrip in `tiff.rs` (preserve unknown tags during read-modify-write) — `TiffInfo::preserve_unknown_tags`/`extra_ifd_entries`/`set_tag`/`get_tag`; round-trip tests for 0xC4A5 and 0x8825; tiff split to `tiff/tests.rs` (0.1.8 Wave 5 Slice γ)
- [x] Add adaptive thresholding modes (Otsu, triangle) to `segmentation.rs` (verified 2026-05-16; src/segmentation.rs:129 ThresholdMethod::Otsu/Triangle, adaptive_threshold_segment:245)
- [x] Extend `convolution.rs` with separable kernel optimization (apply 1D horizontal then vertical) (verified 2026-05-16; src/convolution.rs:260 SeparableKernel, gaussian_separable_kernel:311)
- [x] Add bilateral filter implementation to `filters.rs` for edge-preserving denoising
- [x] Implement guided filter in `inpainting.rs` for structure-aware inpainting
- [x] Add sub-pixel alignment to `stitch.rs` for higher-accuracy panorama stitching

## New Features
- [x] Add a `webp.rs` module for WebP image decode/encode (lossy and lossless) (verified 2026-05-16; src/webp.rs:866 lines, fn read_webp:423, fn encode_webp_bytes:479)
- [x] Add a `png.rs` module for PNG image decode/encode with APNG animation support
  (~[~] already implemented: PngDecoder, PngEncoder, APNG, zlib via oxiarc_deflate)
- [x] Implement a `jpeg.rs` module for JPEG baseline decode/encode
  (~[~] already implemented: JpegDecoder, JpegEncoder, YCbCr, DCT, Huffman tables)
- [x] Add a `heif.rs` module for HEIF/AVIF still image container support (verified 2026-05-16; src/heif.rs:983 lines, struct HeifContainer:391)
- [x] Implement a `focus_stack.rs` module for focus stacking of image sequences (verified 2026-05-16; src/focus_stack.rs:743 lines)
- [x] Add a `hdr_bracket.rs` module for exposure bracketing and Debevec HDR merge (verified 2026-05-16; src/hdr_bracket.rs:752 lines)
- [x] Implement a `content_aware_resize.rs` module (seam carving) for intelligent resizing (verified 2026-05-16; src/content_aware_resize.rs:760 lines)
- [x] Add a `super_resolution.rs` module for image upscaling beyond simple interpolation (verified 2026-05-16; src/super_resolution.rs:594 lines)
- [x] Implement `color_lut.rs` for applying 3D LUTs directly to ImageData (integrate with oximedia-lut) (verified 2026-05-16; src/color_lut.rs:720 lines)

## Performance
- [x] Add SIMD-accelerated pixel format conversion (U8↔F32 batch) — `simd_convert.rs` with AVX2/NEON/scalar fallback (0.1.8 Wave 5 Slice γ)
- [x] Implement tiled parallel processing in `convolution.rs` for large images (convolve_tiled rayon par_iter over tiles, ±1 pixel-exact; 2026-05-30)
- [x] Add memory-mapped I/O path for DPX/EXR sequence reading to reduce allocation overhead (MmapImageReader memmap2, mmap_reader.rs 432L; 2026-05-30)
- [x] Optimize morphology dilation/erosion with van Herk–Gil–Werman O(1) sliding-window — `morphology_vhgw.rs` with dispatch in `advanced_morphology.rs` for flat rect and axis-aligned line SEs (0.1.8 Wave 5 Slice γ)
- [x] Add OxiFFT-accelerated 2D FFT to `frequency_domain.rs` replacing hand-rolled O(N²) DFT with oxifft::fft2d/ifft2d (0.1.8 Wave 13 Slice C)
- [ ] Add GPU-accelerated path for `frequency_domain.rs` FFT operations (GPU wgpu path deferred — Pure-Rust OxiFFT migration done)
- [x] Parallelize `stitch.rs` feature matching across image pairs with rayon (match_descriptors_parallel rayon; 2026-05-30)

## Testing
- [x] Add round-trip tests for DPX read/write at all supported bit depths (8, 10, 12, 16) — pack_10bit/unpack_10bit_packed Method A & B; 8 tests in wave13_tests.rs (0.1.8 Wave 13 Slice C)
- [x] Test EXR compression round-trips (None, Zip, Piz, B44) with tolerance checks — code/decode round-trips + ExrDocument API; 7 tests in wave13_tests.rs (0.1.8 Wave 13 Slice C)
- [x] Add DNG demosaic quality tests comparing against reference implementations — PSNR uniform sensor ≥60 dB, uniform RGB ≥25 dB, known-pixel exact; 4 tests in wave13_tests.rs (0.1.8 Wave 13 Slice C)
- [x] Test `edge_detect.rs` Sobel/Canny operators against known edge maps — step-edge peak ±1px, uniform→zero, Prewitt; 4 tests in wave13_tests.rs (0.1.8 Wave 13 Slice C)
- [x] Add `blend_mode.rs` tests verifying each mode against W3C compositing formulas — Multiply/Screen/Overlay/SoftLight/HardLight/Normal with hand-computed expected values; 12 tests in wave13_tests.rs (0.1.8 Wave 13 Slice C)
- [x] Test `sequence.rs` pattern matching with edge cases (missing frames, non-contiguous ranges) — gap detection, non-contiguous ranges, zero-padded format, empty dir, has_frame checks; 6 tests in wave13_tests.rs (0.1.8 Wave 13 Slice C)

## Documentation
- [ ] Document supported DPX header fields and their mapping to ImageData metadata
- [ ] Add pixel format conversion matrix showing supported PixelType conversions
- [ ] Document the color science pipeline (color_science -> color_adjust -> color_balance)

## 0.1.8 Wave 5 (completed 2026-05-29)
- [x] VHGW O(1) rectangular morphology (dilate/erode, 3 compares/pixel regardless of kernel size) (Slice γ)
  - File: src/morphology_vhgw.rs (395 lines)
- [x] TIFF unknown-tag round-trip (preserve_unknown_tags + extra_ifd_entries HashMap) (Slice γ)
  - File: src/tiff/mod.rs; tests extracted to src/tiff/tests.rs (594 lines)
- [x] SIMD U8↔F32 normalized conversion (AVX2/NEON/scalar fallback) (Slice γ)
  - File: src/simd_convert.rs (388 lines)
