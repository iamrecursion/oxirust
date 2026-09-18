# oximedia

Facade crate for the OxiMedia multimedia framework.

## Overview

`oximedia` is the main entry point for using the OxiMedia framework. It re-exports types from all component crates for convenient access:

- `oximedia-core` - Core types and traits
- `oximedia-io` - I/O and bit-level reading
- `oximedia-container` - Container demuxing/muxing

## Usage

### Basic Import

```rust
use oximedia::prelude::*;
```

### Format Probing

```rust
use oximedia::prelude::*;

fn probe_file(path: &str) -> OxiResult<()> {
    let data = std::fs::read(path)?;
    let result = probe_format(&data)?;

    println!("Format: {:?}", result.format);
    println!("Confidence: {:.1}%", result.confidence * 100.0);

    Ok(())
}
```

### Available Types

From `oximedia-core`:

| Type | Description |
|------|-------------|
| `Rational` | Exact rational numbers |
| `Timestamp` | Media timestamps with timebase |
| `PixelFormat` | Video pixel formats |
| `SampleFormat` | Audio sample formats |
| `CodecId` | Codec identifiers |
| `MediaType` | Media type (Video, Audio, Subtitle) |
| `OxiError` | Unified error type |

From `oximedia-io`:

| Type | Description |
|------|-------------|
| `BitReader` | Bit-level reading |
| `MediaSource` | Async media source trait |
| `FileSource` | File-based source |
| `MemorySource` | Memory-based source |

From `oximedia-container`:

| Type | Description |
|------|-------------|
| `ContainerFormat` | Container format enum |
| `Packet` | Compressed media packet |
| `PacketFlags` | Packet properties |
| `StreamInfo` | Stream metadata |
| `Demuxer` | Demuxer trait |
| `probe_format` | Format detection function |

## Prelude

The prelude module exports the most commonly used types:

```rust
pub use crate::{
    CodecId, MediaType, OxiError, OxiResult,
    PixelFormat, Rational, SampleFormat, Timestamp,
    ContainerFormat, Packet, PacketFlags,
    probe_format,
};

pub use oximedia_io::MediaSource;
pub use oximedia_container::Demuxer;
```

## Example

```rust
use oximedia::prelude::*;

#[tokio::main]
async fn main() -> OxiResult<()> {
    // Create a timestamp
    let ts = Timestamp::new(1000, Rational::new(1, 1000));
    println!("Timestamp: {} seconds", ts.to_seconds());

    // Check codec type
    let codec = CodecId::Av1;
    println!("AV1 is video: {}", codec.is_video());

    // Probe a format
    let data = vec![0x1A, 0x45, 0xDF, 0xA3]; // EBML header
    if let Ok(result) = probe_format(&data) {
        println!("Detected: {:?}", result.format);
    }

    Ok(())
}
```

## Feature matrix

Every optional feature gates in exactly one workspace crate (occasionally the
same crate exposes more than one feature, e.g. `oximedia-codec` backs
`video`, `mjpeg`, and `apv`). Enable only the features you need — the
`default` build is core-only (probing, demuxing, computer vision):

| Feature | Crate(s) enabled | Purpose |
|---------|---------------|---------|
| `audio` | `oximedia-audio` | Opus, Vorbis, FLAC, PCM codecs |
| `video` | `oximedia-codec` | AV1, VP9, VP8 video codecs |
| `graph` | `oximedia-graph` | Filter graph / processing pipeline |
| `effects` | `oximedia-effects` | Professional audio effects suite |
| `net` | `oximedia-net` | HLS, DASH, SRT, RTMP, WebRTC |
| `metering` | `oximedia-metering` | EBU R128, ATSC A/85 loudness |
| `normalize` | `oximedia-normalize` | Loudness normalization (implicitly enables `metering`) |
| `quality` | `oximedia-quality` | PSNR, SSIM, VMAF, NIQE |
| `metadata-ext` | `oximedia-metadata` | ID3v2, XMP, EXIF, IPTC |
| `timecode` | `oximedia-timecode` | SMPTE LTC/VITC timecode |
| `workflow` | `oximedia-workflow` | DAG workflow orchestration |
| `batch` | `oximedia-batch` | Batch job processing engine |
| `monitor` | `oximedia-monitor` | System monitoring and alerting |
| `lut` | `oximedia-lut` | 1D/3D LUT and HDR pipeline |
| `colormgmt` | `oximedia-colormgmt` | ICC, ACES, HDR color management |
| `transcode` | `oximedia-transcode` | Full transcoding pipeline |
| `subtitle` | `oximedia-subtitle` | SRT, ASS, WebVTT rendering |
| `captions` | `oximedia-captions` | Closed caption formats |
| `archive` | `oximedia-archive` | Archive verification & preservation |
| `dedup` | `oximedia-dedup` | Media deduplication |
| `search` | `oximedia-search` | Media search and indexing |
| `mam` | `oximedia-mam` | Media Asset Management system |
| `scene` | `oximedia-scene` | AI scene understanding |
| `shots` | `oximedia-shots` | Shot detection & classification |
| `scopes` | `oximedia-scopes` | Broadcast video scopes |
| `vfx` | `oximedia-vfx` | Visual effects and compositing |
| `image-ext` | `oximedia-image` | Advanced image processing (DPX, EXR, TIFF) |
| `watermark` | `oximedia-watermark` | Audio watermarking and forensic detection |
| `mir` | `oximedia-mir` | Music Information Retrieval |
| `recommend` | `oximedia-recommend` | Content recommendation engine |
| `playlist` | `oximedia-playlist` | Broadcast playlist management |
| `playout` | `oximedia-playout` | Broadcast playout server |
| `rights` | `oximedia-rights` | Digital rights management |
| `review` | `oximedia-review` | Collaborative media review |
| `restore` | `oximedia-restore` | Audio/video restoration |
| `repair` | `oximedia-repair` | Media file repair and recovery |
| `multicam` | `oximedia-multicam` | Multi-camera sync and switching |
| `stabilize` | `oximedia-stabilize` | Video stabilization |
| `cloud` | `oximedia-cloud` | Cloud storage abstraction (S3, Azure, GCS) |
| `edl` | `oximedia-edl` | EDL parsing and generation |
| `ndi` | `oximedia-ndi` | NDI protocol support |
| `imf` | `oximedia-imf` | IMF package support (SMPTE ST 2067) |
| `aaf` | `oximedia-aaf` | AAF interchange (SMPTE ST 377-1) |
| `timesync` | `oximedia-timesync` | PTP/NTP time synchronization |
| `forensics` | `oximedia-forensics` | Media forensics and tampering detection |
| `accel` | `oximedia-accel` | Hardware acceleration (Vulkan GPU, CPU fallback) |
| `simd` | `oximedia-simd` | SIMD-optimised media kernels (DCT, SAD, blending) |
| `switcher` | `oximedia-switcher` | Professional live video switcher |
| `timeline` | `oximedia-timeline` | Multi-track timeline editor |
| `optimize` | `oximedia-optimize` | Codec optimisation suite (RDO, psychovisual, AQ) |
| `profiler` | `oximedia-profiler` | Performance profiling tools |
| `renderfarm` | `oximedia-renderfarm` | Distributed render farm coordinator |
| `storage` | `oximedia-storage` | Cloud-agnostic object storage (S3, Azure, GCS) |
| `collab` | `oximedia-collab` | Real-time CRDT collaborative editing |
| `gaming` | `oximedia-gaming` | Game streaming and screen capture |
| `virtual-prod` | `oximedia-virtual` | Virtual production and LED wall tools |
| `access` | `oximedia-access` | Accessibility (audio description, captions, WCAG) |
| `conform` | `oximedia-conform` | Media conforming (EDL/XML/AAF matching) |
| `convert` | `oximedia-convert` | Media format conversion utilities |
| `automation` | `oximedia-automation` | Broadcast automation and master control |
| `clips` | `oximedia-clips` | Professional clip management and logging |
| `proxy` | `oximedia-proxy` | Proxy and offline editing workflows |
| `presets` | `oximedia-presets` | Encoding preset library (200+ presets) |
| `calibrate` | `oximedia-calibrate` | Color calibration and camera profiling |
| `denoise` | `oximedia-denoise` | Video denoising (spatial, temporal, hybrid) |
| `align` | `oximedia-align` | Multi-camera video alignment and registration |
| `analysis` | `oximedia-analysis` | Comprehensive media analysis and QA |
| `audiopost` | `oximedia-audiopost` | Audio post-production (ADR, Foley, mixing) |
| `qc` | `oximedia-qc` | Broadcast-grade quality control and validation |
| `jobs` | `oximedia-jobs` | Job queue and worker management |
| `auto` | `oximedia-auto` | Automated video editing and highlight detection |
| `edit` | `oximedia-edit` | Video timeline editor with effects |
| `routing` | `oximedia-routing` | Signal routing, NMOS IS-04/IS-05/IS-07 |
| `audio-analysis` | `oximedia-audio-analysis` | Spectral, voice, music, forensics analysis |
| `gpu` | `oximedia-gpu` | WGPU GPU compute (Vulkan, Metal, DX12, WebGPU) |
| `packager` | `oximedia-packager` | HLS/DASH adaptive streaming packaging |
| `drm` | `oximedia-drm` | CENC, Widevine, PlayReady, FairPlay DRM |
| `archive-pro` | `oximedia-archive-pro` | BagIt, OAIS, PREMIS digital preservation |
| `distributed` | `oximedia-distributed` | Distributed multi-node encoding |
| `farm` | `oximedia-farm` | Render farm coordinator |
| `dolbyvision` | `oximedia-dolbyvision` | Dolby Vision RPU metadata |
| `mixer` | `oximedia-mixer` | Professional digital audio mixer |
| `scaling` | `oximedia-scaling` | High-quality video scaling |
| `graphics` | `oximedia-graphics` | Broadcast graphics engine |
| `videoip` | `oximedia-videoip` | Video-over-IP protocol |
| `compat-ffmpeg` | `oximedia-compat-ffmpeg` | FFmpeg CLI compatibility layer |
| `plugin` | `oximedia-plugin` | Dynamic/static codec plugin system |
| `server` | `oximedia-server` | RESTful media server |
| `hdr` | `oximedia-hdr` | HDR video processing (PQ/HLG, tone mapping, HDR10+) |
| `spatial` | `oximedia-spatial` | Spatial audio (Ambisonics, binaural, room simulation) |
| `cache` | `oximedia-cache` | High-performance media caching (LRU, tiered, warming) |
| `stream` | `oximedia-stream` | Adaptive streaming pipeline, segment management, QoE |
| `video-proc` | `oximedia-video` | Scene detection, pulldown detection, temporal denoising, perceptual fingerprinting |
| `cdn` | `oximedia-cdn` | CDN edge management, cache invalidation, geographic routing, origin failover |
| `neural` | `oximedia-neural` | Lightweight neural network inference for media (tensor ops, conv2d, scene classification) |
| `vr360` | `oximedia-360` | 360° VR video: equirectangular/cubemap projections, fisheye, stereo 3D |
| `analytics` | `oximedia-analytics` | Media engagement analytics: sessions, retention curves, A/B testing, scoring |
| `caption-gen` | `oximedia-caption-gen` | Advanced caption generation: speech alignment, WCAG compliance, diarization |
| `image-transform` | `oximedia-image-transform` | Image transformations: affine, perspective, resize, crop, rotate, color conversion, lens distortion |
| `pipeline` | `oximedia-pipeline` | Declarative media processing DSL: typed filter graph, node composition, execution planning |
| `ml` | `oximedia-ml` | Sovereign ML pipelines (Pure-Rust ONNX inference); combine with `ml-scene-classifier`, `ml-shot-boundary`, `ml-aesthetic-score`, `ml-object-detector`, `ml-face-embedder`, `ml-onnx` |
| `mjpeg` | `oximedia-codec` (mjpeg) | Motion JPEG intra-frame video codec |
| `apv` | `oximedia-codec` (apv) | APV (Advanced Professional Video) intra-frame codec (ISO/IEC 23009-13) |
| `full` | all of the above | Everything enabled |

Convenience presets that bundle several of the rows above:

| Preset | Expands to |
|--------|-----------|
| `minimal` | `audio`, `video`, `metadata-ext` |
| `audio-stack` | `audio`, `effects`, `metering`, `normalize`, `audio-analysis`, `mixer`, `audiopost` |
| `broadcast-stack` | `automation`, `playout`, `playlist`, `switcher`, `routing`, `graphics`, `scopes` |
| `streaming-stack` | `net`, `packager`, `drm`, `stream`, `cdn`, `cache`, `server` |

## Cookbook

Worked examples live in the workspace-level `examples/` directory (built via
`cargo run --example <name> --features <flags>`):

| Example | Required features | What it demonstrates |
|---------|-------------------|----------------------|
| `probe_file` | *(none)* | Container format probing on a raw byte buffer |
| `corner_detection` | *(none)* | `oximedia-cv` Harris/FAST corner detection |
| `optical_flow` | *(none)* | `oximedia-cv` Lucas-Kanade optical flow |
| `face_detection` | *(none)* | `oximedia-cv` Haar-cascade face detection |
| `image_processing` | *(none)* | `oximedia-cv` filters and color conversion |
| `decode_video` | *(none)* | Decoding a video bitstream frame-by-frame |
| `audio_metering` | `metering` | EBU R128 / ATSC A/85 loudness measurement |
| `quality_assessment` | `quality` | PSNR/SSIM/VMAF quality scoring |
| `timecode_operations` | `timecode` | SMPTE LTC/VITC timecode arithmetic |
| `dedup_detection` | `dedup` | Multi-strategy duplicate media detection |
| `workflow_pipeline` | `workflow` | DAG-based media workflow orchestration |
| `video_scopes` | `scopes` | Waveform/vectorscope/histogram rendering |
| `shot_detection` | `shots` | Cut/dissolve/fade shot boundary detection |
| `nmos_registry` | `routing` | NMOS IS-04 registry discovery |
| `color_pipeline` | `colormgmt`, `lut` | ACES color pipeline with LUT application |
| `media_pipeline` | `quality`, `metering`, `transcode`, `timecode`, `workflow`, `archive` | End-to-end ingest → transcode → QC → archive tutorial |
| `nmos_server_demo` | `routing` | Running a local NMOS IS-04 registry server |
| `ml_scene_classify` | `ml`, `ml-scene-classifier` | Places365 scene classification via OxiONNX |
| `ml_auto_caption` | `ml` | Automatic caption generation with sovereign ML |
| `ml_model_zoo` | `ml` | Loading and running bundled ONNX model zoo entries |
| `ffmpeg_translate_demo` | `compat-ffmpeg` | Parses an FFmpeg command line and prints the translated `TranscodeJob`(s) |

Every example above except `ffmpeg_translate_demo` lives in the workspace-root
`examples/` directory; `ffmpeg_translate_demo` lives in `oximedia/examples/`.

## Policy

- No unsafe code (`#![forbid(unsafe_code)]`)
- No warnings
- Apache 2.0 license

## License

Apache-2.0
