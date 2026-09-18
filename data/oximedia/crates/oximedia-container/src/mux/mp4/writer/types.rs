//! Shared types for the MP4 muxer writer sub-modules.
//!
//! Contains [`Mp4FragmentMode`], [`Mp4Config`], [`Mp4SampleEntry`], and
//! [`Mp4TrackState`] — the core data structures used across all sub-modules.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use oximedia_core::MediaType;

use crate::StreamInfo;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Default timescale for the movie header (ticks per second).
pub const MOVIE_TIMESCALE: u32 = 1000;

/// Maximum supported tracks.
pub const MAX_TRACKS: usize = 16;

// ─── Configuration ───────────────────────────────────────────────────────────

/// MP4 fragment/muxing mode.
///
/// Controls whether the output is a standard progressive MP4 (single `moov`+`mdat`)
/// or a fragmented MP4 (`moov` init segment + repeating `sidx`+`moof`+`mdat` fragments).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mp4FragmentMode {
    /// Progressive MP4: single `moov` + single `mdat`.
    ///
    /// All sample metadata is collected in memory and written to `moov` at finalize time.
    Progressive,
    /// Fragmented MP4 (fMP4/CMAF-compatible): `moov` init segment + `sidx`+`moof`+`mdat` fragments.
    ///
    /// The `moov` init segment contains empty `stbl` tables and an `mvex`/`trex` extension box.
    /// Fragments are split on keyframe boundaries, honouring the requested target duration.
    ///
    /// `fragment_duration_ms` is the target fragment duration in milliseconds.  A value of 0
    /// means "one fragment per keyframe boundary".
    Fragmented {
        /// Target fragment duration in milliseconds.
        fragment_duration_ms: u32,
    },
}

impl Default for Mp4FragmentMode {
    fn default() -> Self {
        Self::Progressive
    }
}

/// Backward-compatible type alias — prefer [`Mp4FragmentMode`] in new code.
pub type Mp4Mode = Mp4FragmentMode;

/// Configuration for the MP4 muxer.
#[derive(Debug, Clone)]
pub struct Mp4Config {
    /// Muxing mode (progressive or fragmented).
    ///
    /// For fragmented mode use `Mp4FragmentMode::Fragmented { fragment_duration_ms }`.
    pub mode: Mp4FragmentMode,
    /// Major brand for ftyp box.
    pub major_brand: [u8; 4],
    /// Minor version for ftyp box.
    pub minor_version: u32,
    /// Compatible brands for ftyp box.
    pub compatible_brands: Vec<[u8; 4]>,
    /// Creation time (seconds since 1904-01-01).
    pub creation_time: u64,
    /// Modification time (seconds since 1904-01-01).
    pub modification_time: u64,
}

impl Default for Mp4Config {
    fn default() -> Self {
        Self {
            mode: Mp4FragmentMode::Progressive,
            major_brand: *b"isom",
            minor_version: 0x200,
            compatible_brands: vec![*b"isom", *b"iso6", *b"mp41"],
            creation_time: 0,
            modification_time: 0,
        }
    }
}

impl Mp4Config {
    /// Creates a new `Mp4Config` with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the muxing mode.
    #[must_use]
    pub fn with_mode(mut self, mode: Mp4FragmentMode) -> Self {
        self.mode = mode;
        self
    }

    /// Convenience builder: configure fragmented mode with the given target
    /// fragment duration in milliseconds.
    ///
    /// Equivalent to `.with_mode(Mp4FragmentMode::Fragmented { fragment_duration_ms: ms })`.
    #[must_use]
    pub fn with_fragmented(mut self, fragment_duration_ms: u32) -> Self {
        self.mode = Mp4FragmentMode::Fragmented {
            fragment_duration_ms,
        };
        self
    }

    /// Sets the major brand.
    #[must_use]
    pub const fn with_major_brand(mut self, brand: [u8; 4]) -> Self {
        self.major_brand = brand;
        self
    }

    /// Sets the minor version.
    #[must_use]
    pub const fn with_minor_version(mut self, version: u32) -> Self {
        self.minor_version = version;
        self
    }

    /// Adds a compatible brand.
    #[must_use]
    pub fn with_compatible_brand(mut self, brand: [u8; 4]) -> Self {
        self.compatible_brands.push(brand);
        self
    }
}

// ─── Sample Entry ─────────────────────────────────────────────────────────────

/// Metadata for a single sample in a track.
#[derive(Debug, Clone)]
pub struct Mp4SampleEntry {
    /// Sample size in bytes.
    pub size: u32,
    /// Sample duration in the track's timescale.
    pub duration: u32,
    /// Composition time offset (PTS - DTS) in the track's timescale.
    pub composition_offset: i32,
    /// Whether this sample is a sync (key) sample.
    pub is_sync: bool,
    /// Chunk index this sample belongs to.
    pub chunk_index: u32,
}

// ─── Track State ──────────────────────────────────────────────────────────────

/// Per-track state maintained during muxing.
#[derive(Debug, Clone)]
pub struct Mp4TrackState {
    /// Stream information.
    pub stream_info: StreamInfo,
    /// Track ID (1-based).
    pub track_id: u32,
    /// Timescale for this track.
    pub timescale: u32,
    /// Accumulated sample entries.
    pub samples: Vec<Mp4SampleEntry>,
    /// Raw mdat data for this track.
    pub mdat_data: Vec<u8>,
    /// Current chunk boundaries (sample indices).
    pub chunk_boundaries: Vec<u32>,
    /// Number of samples in the current chunk.
    pub current_chunk_samples: u32,
    /// Maximum samples per chunk before starting a new one.
    pub max_samples_per_chunk: u32,
    /// Total duration in timescale units.
    pub total_duration: u64,
    /// Handler type identifier.
    pub handler_type: [u8; 4],
    /// Handler name.
    pub handler_name: String,
}

impl Mp4TrackState {
    pub(super) fn new(stream_info: StreamInfo, track_id: u32) -> Self {
        let (timescale, handler_type, handler_name) = match stream_info.media_type {
            MediaType::Video => {
                let ts = if stream_info.timebase.den != 0 {
                    // Use timebase denominator as timescale
                    stream_info.timebase.den as u32
                } else {
                    90000
                };
                (ts, *b"vide", "OxiMedia Video Handler".to_string())
            }
            MediaType::Audio => {
                let ts = stream_info.codec_params.sample_rate.unwrap_or(48000);
                (ts, *b"soun", "OxiMedia Audio Handler".to_string())
            }
            _ => (1000, *b"text", "OxiMedia Text Handler".to_string()),
        };

        Self {
            stream_info,
            track_id,
            timescale,
            samples: Vec::new(),
            mdat_data: Vec::new(),
            chunk_boundaries: vec![0],
            current_chunk_samples: 0,
            max_samples_per_chunk: 10,
            total_duration: 0,
            handler_type,
            handler_name,
        }
    }

    pub(super) fn add_sample(
        &mut self,
        data: &[u8],
        duration: u32,
        comp_offset: i32,
        is_sync: bool,
    ) {
        let chunk_index = self.chunk_boundaries.len().saturating_sub(1) as u32;
        self.samples.push(Mp4SampleEntry {
            size: data.len() as u32,
            duration,
            composition_offset: comp_offset,
            is_sync,
            chunk_index,
        });
        self.mdat_data.extend_from_slice(data);
        self.total_duration += u64::from(duration);
        self.current_chunk_samples += 1;

        if self.current_chunk_samples >= self.max_samples_per_chunk {
            self.chunk_boundaries.push(self.samples.len() as u32);
            self.current_chunk_samples = 0;
        }
    }
}
