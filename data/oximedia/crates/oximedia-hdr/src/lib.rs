//! HDR (High Dynamic Range) video processing for OxiMedia.
//!
//! Provides SMPTE ST 2084 (PQ), HLG, and SDR transfer functions,
//! Rec. 2020 / Rec. 2100 gamut handling, HDR10 and HDR10+ metadata,
//! and a suite of tone-mapping operators for HDR-to-SDR conversion.
//!
//! # HDR-to-SDR Conversion Pipeline
//!
//! The full pipeline proceeds in five stages.  Each stage maps to one or
//! more modules in this crate.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │               HDR-TO-SDR CONVERSION PIPELINE                         │
//! └──────────────────────────────────────────────────────────────────────┘
//!
//!   ┌────────────────────┐
//!   │   HDR Input Frame  │  ← PQ (SMPTE ST 2084) or HLG (BT.2100)
//!   │   + Metadata SEIs  │    signal encoded in BT.2020 colour gamut
//!   └─────────┬──────────┘
//!             │
//!             ▼
//!   ┌──────────────────────────────────────────────────────────────────┐
//!   │  STAGE 1 · Metadata Extraction                                   │
//!   │                                                                  │
//!   │  • Content Light Level (MaxCLL / MaxFALL)  [color_volume.rs]    │
//!   │    Parsed from SMPTE ST 2086 SEI payload                        │
//!   │  • Mastering Display Colour Volume (MDCV)  [color_volume.rs]    │
//!   │    Peak nits, black level, display primaries                     │
//!   │  • HDR10+ dynamic metadata                 [dynamic_metadata.rs]│
//!   │    Per-scene / per-frame luminance curves   [hdr10plus.rs]      │
//!   │  • Dolby Vision RPU                        [dolby_vision_profile.rs]
//!   │    Profile detection, RPU parse/generate,  [dovi_rpu.rs]        │
//!   │    per-frame trim-pass parameters                                │
//!   └─────────────────────────────┬────────────────────────────────────┘
//!                                 │ peak nits, dynamic curves
//!                                 ▼
//!   ┌──────────────────────────────────────────────────────────────────┐
//!   │  STAGE 2 · Transfer Function Decode (EOTF / OOTF)               │
//!   │                                                                  │
//!   │  • PQ EOTF   (SMPTE ST 2084)   [transfer_function.rs:pq_eotf]  │
//!   │    Compressed signal E' → absolute linear light in cd/m²        │
//!   │  • HLG OOTF  (BT.2100)         [hlg_advanced.rs:hlg_ootf]      │
//!   │    Scene-referred → display-referred with system-gamma adjust    │
//!   │  • LUT-accelerated fast path   [transfer_function.rs:PqEotfLut] │
//!   │    Pre-computed 1-D look-up tables for throughput                │
//!   │  • SIMD batch path             [pq_simd.rs:PqSimdProcessor]     │
//!   │    8-wide unrolled EOTF/OETF for modern CPUs                    │
//!   └─────────────────────────────┬────────────────────────────────────┘
//!                                 │ linear scene light (normalised 0–1)
//!                                 ▼
//!   ┌──────────────────────────────────────────────────────────────────┐
//!   │  STAGE 3 · Gamut Mapping  (BT.2020 → BT.709)                    │
//!   │                                                                  │
//!   │  • Linear 3×3 matrix           [gamut.rs:GamutConversionMatrix] │
//!   │    Rec.2020 primaries → Rec.709 primaries (cached per-pair)     │
//!   │  • Soft-clip perceptual map    [soft_clip_gamut.rs]             │
//!   │    Chromatic desaturation instead of hard out-of-gamut clamp    │
//!   │  • ICtCp perceptual domain     [color_volume_transform.rs]      │
//!   │    Optional: map in ICtCp (BT.2100 Note 5) uniform colour space │
//!   └─────────────────────────────┬────────────────────────────────────┘
//!                                 │ BT.709 linear light [0, 1]
//!                                 ▼
//!   ┌──────────────────────────────────────────────────────────────────┐
//!   │  STAGE 4 · Tone Mapping  (HDR nits → SDR [0, 1])                │
//!   │                                                                  │
//!   │  • BT.2446 Method A / C   [tone_mapping.rs:Bt2446MethodA/C]    │
//!   │    ITU-R BT.2446-1 reference operators (scene → display)        │
//!   │  • Reinhard / ACES / Hable  [tone_mapping.rs:ToneMappingOperator]│
//!   │    Classic creative filmic operators                             │
//!   │  • Scene-referred per-frame [tone_mapping.rs:SceneReferredToneMapper]
//!   │    Luminance analysis + dynamic operator selection per frame     │
//!   │  • Rayon parallel rows      [tone_mapping.rs:map_frame_parallel] │
//!   │    Multi-core throughput via per-row parallelism                 │
//!   └─────────────────────────────┬────────────────────────────────────┘
//!                                 │ SDR linear light [0, 1]
//!                                 ▼
//!   ┌──────────────────────────────────────────────────────────────────┐
//!   │  STAGE 5 · Transfer Function Encode (OETF / SDR gamma)          │
//!   │                                                                  │
//!   │  • BT.709 OETF / sRGB gamma   [transfer_function.rs:sdr_gamma] │
//!   │    Linear → display-referred signal in [0, 1]                   │
//!   │  • Display-model gamma         [display_model.rs]               │
//!   │    Target-display peak / black / gamma characterisation          │
//!   └─────────────────────────────┬────────────────────────────────────┘
//!                                 │
//!                                 ▼
//!   ┌────────────────────┐
//!   │   SDR Output Frame │  ← BT.709 / sRGB, display-ready signal
//!   └────────────────────┘
//! ```
//!
//! ## Mermaid diagram (GitHub / GitLab rendered)
//!
//! ```mermaid
//! flowchart TD
//!     A["HDR Input Frame + Metadata SEIs\n(PQ/HLG · BT.2020)"]
//!     B["Stage 1 · Metadata Extraction\nCLL · MDCV · HDR10+ · Dolby Vision RPU"]
//!     C["Stage 2 · Transfer Function Decode\nPQ EOTF or HLG OOTF → linear light\n(LUT / SIMD accelerated)"]
//!     D["Stage 3 · Gamut Mapping\nBT.2020 → BT.709\nsoft-clip · ICtCp perceptual"]
//!     E["Stage 4 · Tone Mapping\nBT.2446-A/C · Reinhard · ACES · Hable\nscene-referred · dynamic · parallel"]
//!     F["Stage 5 · SDR OETF\nlinear → BT.709 gamma / sRGB"]
//!     G["SDR Output Frame\n(BT.709 / sRGB)"]
//!
//!     A --> B
//!     B --> C
//!     C --> D
//!     D --> E
//!     E --> F
//!     F --> G
//! ```

pub mod ambient_light_adaptation;
pub mod color_volume;
pub mod color_volume_transform;
pub mod cuva_metadata;
pub mod display_db;
pub mod display_model;
pub mod dolby_vision_profile;
pub mod dovi_rpu;
pub mod dynamic_metadata;
pub mod dynamic_metadata_validator;
pub mod gamut;
pub mod hdr10plus;
pub mod hdr10plus_generator;
pub mod hdr_fingerprint;
pub mod hdr_grading_assistant;
pub mod hdr_histogram;
pub mod hdr_lut_pipeline;
pub mod hdr_metadata_extractor;
pub mod hdr_metadata_validator;
pub mod hdr_scene_analysis;
pub mod hdr_scopes;
pub mod hlg_advanced;
pub mod hlg_broadcast_constraints;
pub mod hlg_display_gamma;
pub mod hlg_reference_display;
pub mod hlg_to_pq;
pub mod luminance_stats;
pub mod metadata;
pub mod metadata_passthrough;
pub mod pq_hlg_convert;
pub mod pq_simd;
pub mod scene_grading;
pub mod sdr_to_hdr;
pub mod soft_clip_gamut;
pub mod st2094;
pub mod tone_mapping;
pub mod tone_mapping_ext;
pub mod transfer_function;
pub mod vivid_hdr;

pub use color_volume::{
    encode_cll_sei, encode_hdr10_sei, luminance_from_primaries, parse_cll_sei, parse_hdr10_sei,
    ContentLightLevel, Hdr10PlusMetadata, MasteringDisplayColorVolume, MaxRgbAnalyzer,
};
pub use color_volume_transform::{ictcp_to_rgb, rgb_to_ictcp, ICtCpFrame};
pub use dolby_vision_profile::{
    detect_profile, BlSignalCompatibility, CrossVersionDvMeta, DolbyVisionProfile, DvMetadata,
    DvRpuData,
};
pub use dynamic_metadata::{DynamicMetadataFrame, Hdr10PlusDynamicMetadata};
pub use gamut::{ColorGamut, GamutConversionMatrix};
pub use hdr_histogram::{
    HdrHistogram, HdrHistogramAnalyzer, HdrHistogramConfig, HistogramScale, LuminanceHistogram,
};
pub use hdr_scopes::{LumaStatistics, Vectorscope, WaveformAxis, WaveformMonitor};
pub use hlg_advanced::{
    hlg_adapted_system_gamma, hlg_ootf, hlg_system_for_display, hlg_system_gamma,
    hlg_to_pq_with_gamma, hlg_to_sdr_adaptive, HlgHdr10Convert, HlgSdrConvert, HlgSystem,
};
pub use st2094::{ExtBlock, ExtBlockType, St2094_10Metadata, St2094_40Metadata};
// Note: metadata::ContentLightLevel is superseded by color_volume::ContentLightLevel;
// use the metadata module's struct via its full path if needed.
pub use cuva_metadata::{CuvaMetadata, CuvaPictureType, CuvaToneMapParams, CuvaWhitepoint};
pub use display_model::{
    DisplayGamma, DisplayPrimaries, FullDisplayModel, HdrFormatDisplay, ToneMapDisplayParams,
    WhitePoint,
};
pub use hdr_metadata_extractor::{ContentLightLevelInfo, HdrMetadataExtractor};
pub use luminance_stats::{ContentLuminanceAnalyzer, FrameLuminanceStats};
pub use metadata::{HdrFormat, HdrMasteringMetadata};
pub use metadata_passthrough::{
    HdrMetadataPassthrough, HdrSeiInjector, HdrSeiPayload, SeiNaluExtractor,
};
pub use pq_hlg_convert::{
    effective_system_gamma, hlg_to_pq_convert, pq_to_hlg_convert, PqHlgConverterConfig,
};
pub use scene_grading::{ColorGrade, SceneGradingDatabase, SceneGradingMetadata, TrimPass};
pub use tone_mapping::{
    map_frame_parallel, tone_map_frame_rayon, BT2446MethodAToneMapper, BT2446MethodCToneMapper,
    FrameLuminanceAnalysis, InverseToneMapper, InverseToneMappingOperator, SceneReferredToneMapper,
    ToneMapper, ToneMappingConfig, ToneMappingOperator,
};
pub use tone_mapping_ext::{
    Bt2446MethodAForwardMapper, SdrToHdrConfig, SdrToHdrMapper, UpliftAlgorithm,
};
pub use transfer_function::{
    hlg_eotf, hlg_oetf, pq_eotf, pq_eotf_batch, pq_oetf, pq_oetf_batch, sdr_gamma, HlgEotfLut,
    PqEotfLut, PqOetfLut, TransferFunction,
};
pub use vivid_hdr::{VividEotf, VividHdrMetadata, VividToneMapParams};

#[derive(Debug, Clone, thiserror::Error)]
pub enum HdrError {
    #[error("invalid luminance value: {0}")]
    InvalidLuminance(f32),
    #[error("unsupported transfer function: {0}")]
    UnsupportedTransferFunction(String),
    #[error("gamut conversion error: {0}")]
    GamutConversionError(String),
    #[error("metadata parse error: {0}")]
    MetadataParseError(String),
    #[error("tone mapping error: {0}")]
    ToneMappingError(String),
}

pub type Result<T> = std::result::Result<T, HdrError>;
