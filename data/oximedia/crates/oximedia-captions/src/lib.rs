//! Professional closed captioning and subtitle authoring system for `OxiMedia`.
//!
//! This crate provides comprehensive tools for creating, editing, validating, and exporting
//! closed captions and subtitles in various professional formats.
//!
//! # Supported Formats
//!
//! ## Closed Captions
//! - CEA-608 (Line 21, NTSC, 2 channels)
//! - CEA-708 (ATSC, up to 8 services)
//! - Teletext (EBU, BBC standards)
//! - ARIB (Japan)
//!
//! ## Subtitle Formats
//! - SRT (`SubRip`)
//! - `WebVTT` (Web Video Text Tracks)
//! - ASS/SSA (Advanced `SubStation` Alpha)
//! - TTML (Timed Text Markup Language)
//! - DFXP (Distribution Format Exchange Profile)
//! - SCC (Scenarist Closed Captions)
//! - STL (EBU-STL, Spruce STL)
//! - iTunes Timed Text (iTT)
//!
//! ## Embedded Formats
//! - MPEG-TS DVB subtitles
//! - MP4 608/708 captions
//! - Matroska/WebM subtitles
//! - Blu-ray PGS (Presentation Graphic Stream)
//! - DVD `VobSub`
//!
//! # Features
//!
//! - Caption authoring and editing
//! - Frame-accurate timing
//! - Style and positioning
//! - FCC and WCAG compliance validation
//! - Multi-language support
//! - Translation workflow
//! - Quality control and reporting
//! - Template system
//! - Import/export between formats

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod accessibility;
pub mod asr;
pub mod authoring;
pub mod batch;
pub mod caption_diff;
pub mod caption_export;
pub mod caption_fingerprint;
pub mod caption_gap_analysis;
pub mod caption_merge;
pub mod caption_normalize;
pub mod caption_profiler;
pub mod caption_qc;
pub mod caption_rate_control;
pub mod caption_renderer;
pub mod caption_search;
pub mod caption_segmenter;
pub mod caption_stats;
pub mod caption_style;
pub mod caption_sync;
pub mod caption_timing;
pub mod caption_translate;
pub mod caption_validator;
pub mod effects;
pub mod embedding;
pub mod error;
pub mod export;
pub mod forced_narrative;
pub mod formats;
pub mod import;
pub mod imsc;
pub mod live_captions;

/// Backward-compatibility alias: all items from the former `live_caption`
/// module now live in [`live_captions`].
pub use live_captions as live_caption;
pub mod merge_split;
pub mod quality_scorer;
pub mod region_def;
pub mod rendering;
pub mod report;
pub mod shotchange;
pub mod speaker_diarization;

/// Backward-compatibility alias: all items from the former `speaker_diarize`
/// module now live in [`speaker_diarization`].
pub use speaker_diarization as speaker_diarize;
pub mod caption_localization;
pub mod caption_versioning;
pub mod karaoke_timing;
pub mod ocr_subtitle;
pub mod sdh_generator;
pub mod smpte_2052;
pub mod standards;
pub mod teletext_page_editor;
pub mod templates;
pub mod translation;
pub mod types;
pub mod utils;
pub mod validation;

pub use error::{CaptionError, Result};
pub use types::{
    Alignment, Caption, CaptionId, CaptionStyle, CaptionTrack, Color, Duration, Language, Metadata,
    Position, Timestamp,
};

/// Caption format identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CaptionFormat {
    /// `SubRip` (.srt)
    Srt,
    /// Web Video Text Tracks (.vtt)
    WebVtt,
    /// Advanced `SubStation` Alpha (.ass)
    Ass,
    /// `SubStation` Alpha (.ssa)
    Ssa,
    /// Timed Text Markup Language (.ttml)
    Ttml,
    /// Distribution Format Exchange Profile (.dfxp)
    Dfxp,
    /// Scenarist Closed Captions (.scc)
    Scc,
    /// EBU-STL (.stl)
    EbuStl,
    /// Spruce STL (.stl)
    SpruceStl,
    /// iTunes Timed Text (.itt)
    ITt,
    /// CEA-608
    Cea608,
    /// CEA-708
    Cea708,
    /// Teletext
    Teletext,
    /// ARIB
    Arib,
    /// DVB subtitles
    Dvb,
    /// Blu-ray PGS
    Pgs,
    /// DVD `VobSub`
    VobSub,
}

impl CaptionFormat {
    /// Get the typical file extension for this format
    #[must_use]
    pub fn extension(&self) -> &str {
        match self {
            Self::Srt => "srt",
            Self::WebVtt => "vtt",
            Self::Ass => "ass",
            Self::Ssa => "ssa",
            Self::Ttml | Self::Dfxp => "ttml",
            Self::Scc => "scc",
            Self::EbuStl | Self::SpruceStl => "stl",
            Self::ITt => "itt",
            Self::Cea608 | Self::Cea708 => "scc",
            Self::Teletext => "txt",
            Self::Arib => "arib",
            Self::Dvb => "sub",
            Self::Pgs => "sup",
            Self::VobSub => "sub",
        }
    }

    /// Check if this is a text-based format
    #[must_use]
    pub const fn is_text_based(&self) -> bool {
        matches!(
            self,
            Self::Srt
                | Self::WebVtt
                | Self::Ass
                | Self::Ssa
                | Self::Ttml
                | Self::Dfxp
                | Self::Scc
                | Self::ITt
        )
    }

    /// Check if this is a closed caption format
    #[must_use]
    pub const fn is_closed_caption(&self) -> bool {
        matches!(
            self,
            Self::Cea608 | Self::Cea708 | Self::Teletext | Self::Arib
        )
    }

    /// Check if this is a graphic subtitle format
    #[must_use]
    pub const fn is_graphic(&self) -> bool {
        matches!(self, Self::Pgs | Self::VobSub | Self::Dvb)
    }
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
