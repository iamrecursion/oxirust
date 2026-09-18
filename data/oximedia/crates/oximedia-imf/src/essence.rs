//! MXF Essence file handling
//!
//! This module provides support for parsing and validating MXF essence files
//! used in IMF packages, including video, audio, and subtitle tracks.

use crate::{EditRate, ImfError, ImfResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Color space for video essence
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    /// Rec. 709 (HDTV)
    Rec709,
    /// Rec. 2020 (UHD)
    Rec2020,
    /// DCI-P3 (Digital Cinema)
    DciP3,
    /// Rec. 601 (SDTV)
    Rec601,
    /// sRGB
    Srgb,
    /// Adobe RGB
    AdobeRgb,
    /// Custom/Unknown
    Custom(String),
}

impl ColorSpace {
    /// Get the color space name
    pub fn as_str(&self) -> &str {
        match self {
            Self::Rec709 => "Rec.709",
            Self::Rec2020 => "Rec.2020",
            Self::DciP3 => "DCI-P3",
            Self::Rec601 => "Rec.601",
            Self::Srgb => "sRGB",
            Self::AdobeRgb => "Adobe RGB",
            Self::Custom(s) => s,
        }
    }

    /// Parse color space from string
    pub fn from_str(s: &str) -> Self {
        match s {
            "Rec.709" | "BT.709" | "ITU-R BT.709" => Self::Rec709,
            "Rec.2020" | "BT.2020" | "ITU-R BT.2020" => Self::Rec2020,
            "DCI-P3" | "DCDM" => Self::DciP3,
            "Rec.601" | "BT.601" | "ITU-R BT.601" => Self::Rec601,
            "sRGB" => Self::Srgb,
            "Adobe RGB" | "AdobeRGB" => Self::AdobeRgb,
            _ => Self::Custom(s.to_string()),
        }
    }
}

impl std::fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Audio channel configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioChannelConfig {
    /// Mono (1.0)
    Mono,
    /// Stereo (2.0)
    Stereo,
    /// 5.1 Surround
    Surround51,
    /// 7.1 Surround
    Surround71,
    /// Atmos (object-based)
    Atmos,
    /// Custom channel count
    Custom(u8),
}

impl AudioChannelConfig {
    /// Get the number of channels
    pub fn channel_count(&self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Atmos => 16, // Typical Atmos bed
            Self::Custom(n) => *n,
        }
    }

    /// Create from channel count
    pub fn from_channel_count(count: u8) -> Self {
        match count {
            1 => Self::Mono,
            2 => Self::Stereo,
            6 => Self::Surround51,
            8 => Self::Surround71,
            16 => Self::Atmos,
            n => Self::Custom(n),
        }
    }
}

impl std::fmt::Display for AudioChannelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mono => write!(f, "1.0 (Mono)"),
            Self::Stereo => write!(f, "2.0 (Stereo)"),
            Self::Surround51 => write!(f, "5.1"),
            Self::Surround71 => write!(f, "7.1"),
            Self::Atmos => write!(f, "Atmos"),
            Self::Custom(n) => write!(f, "{n} channels"),
        }
    }
}

/// Transfer characteristic (EOTF)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferCharacteristic {
    /// Rec. 709
    Rec709,
    /// SMPTE ST 2084 (PQ)
    Pq,
    /// Hybrid Log-Gamma (HLG)
    Hlg,
    /// Linear
    Linear,
    /// sRGB
    Srgb,
    /// Custom
    Custom,
}

impl TransferCharacteristic {
    /// Is this an HDR transfer characteristic
    pub fn is_hdr(&self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }
}

/// Video essence descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoDescriptor {
    width: u32,
    height: u32,
    frame_rate: EditRate,
    bit_depth: u8,
    color_space: ColorSpace,
    transfer_characteristic: TransferCharacteristic,
    aspect_ratio: (u32, u32),
    interlaced: bool,
    codec: String,
    profile: Option<String>,
}

impl VideoDescriptor {
    /// Create a new video descriptor
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: u32,
        height: u32,
        frame_rate: EditRate,
        bit_depth: u8,
        color_space: ColorSpace,
        transfer_characteristic: TransferCharacteristic,
        codec: String,
    ) -> Self {
        Self {
            width,
            height,
            frame_rate,
            bit_depth,
            color_space,
            transfer_characteristic,
            aspect_ratio: (16, 9),
            interlaced: false,
            codec,
            profile: None,
        }
    }

    /// Get width
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get height
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get frame rate
    pub fn frame_rate(&self) -> EditRate {
        self.frame_rate
    }

    /// Get bit depth
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// Get color space
    pub fn color_space(&self) -> ColorSpace {
        self.color_space.clone()
    }

    /// Get transfer characteristic
    pub fn transfer_characteristic(&self) -> TransferCharacteristic {
        self.transfer_characteristic
    }

    /// Get aspect ratio
    pub fn aspect_ratio(&self) -> (u32, u32) {
        self.aspect_ratio
    }

    /// Set aspect ratio
    pub fn set_aspect_ratio(&mut self, ratio: (u32, u32)) {
        self.aspect_ratio = ratio;
    }

    /// Is interlaced
    pub fn is_interlaced(&self) -> bool {
        self.interlaced
    }

    /// Set interlaced
    pub fn set_interlaced(&mut self, interlaced: bool) {
        self.interlaced = interlaced;
    }

    /// Get codec
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// Get profile
    pub fn profile(&self) -> Option<&str> {
        self.profile.as_deref()
    }

    /// Set profile
    pub fn set_profile(&mut self, profile: String) {
        self.profile = Some(profile);
    }

    /// Is this HDR content
    pub fn is_hdr(&self) -> bool {
        self.transfer_characteristic.is_hdr()
    }

    /// Get resolution as string (e.g., "1920x1080")
    pub fn resolution_string(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// Get frame rate as float
    pub fn frame_rate_float(&self) -> f64 {
        self.frame_rate.as_f64()
    }
}

/// Audio essence descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDescriptor {
    sample_rate: u32,
    bit_depth: u8,
    channel_config: AudioChannelConfig,
    codec: String,
    language: Option<String>,
    audio_type: String,
}

impl AudioDescriptor {
    /// Create a new audio descriptor
    pub fn new(
        sample_rate: u32,
        bit_depth: u8,
        channel_config: AudioChannelConfig,
        codec: String,
    ) -> Self {
        Self {
            sample_rate,
            bit_depth,
            channel_config,
            codec,
            language: None,
            audio_type: "Main".to_string(),
        }
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get bit depth
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// Get channel configuration
    pub fn channel_config(&self) -> AudioChannelConfig {
        self.channel_config
    }

    /// Get codec
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// Get language
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// Set language
    pub fn set_language(&mut self, language: String) {
        self.language = Some(language);
    }

    /// Get audio type
    pub fn audio_type(&self) -> &str {
        &self.audio_type
    }

    /// Set audio type
    pub fn set_audio_type(&mut self, audio_type: String) {
        self.audio_type = audio_type;
    }

    /// Get channel count
    pub fn channel_count(&self) -> u8 {
        self.channel_config.channel_count()
    }
}

/// Subtitle/Caption descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleDescriptor {
    language: String,
    format: String,
    closed_caption: bool,
}

impl SubtitleDescriptor {
    /// Create a new subtitle descriptor
    pub fn new(language: String, format: String) -> Self {
        Self {
            language,
            format,
            closed_caption: false,
        }
    }

    /// Get language
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Get format
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Is closed caption
    pub fn is_closed_caption(&self) -> bool {
        self.closed_caption
    }

    /// Set closed caption
    pub fn set_closed_caption(&mut self, closed_caption: bool) {
        self.closed_caption = closed_caption;
    }
}

/// Essence descriptor (video, audio, or subtitle)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EssenceDescriptor {
    /// Video essence
    Video(VideoDescriptor),
    /// Audio essence
    Audio(AudioDescriptor),
    /// Subtitle essence
    Subtitle(SubtitleDescriptor),
    /// Unknown essence type
    Unknown,
}

impl EssenceDescriptor {
    /// Is this a video descriptor
    pub fn is_video(&self) -> bool {
        matches!(self, Self::Video(_))
    }

    /// Is this an audio descriptor
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::Audio(_))
    }

    /// Is this a subtitle descriptor
    pub fn is_subtitle(&self) -> bool {
        matches!(self, Self::Subtitle(_))
    }

    /// Get video descriptor
    pub fn as_video(&self) -> Option<&VideoDescriptor> {
        if let Self::Video(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Get audio descriptor
    pub fn as_audio(&self) -> Option<&AudioDescriptor> {
        if let Self::Audio(a) = self {
            Some(a)
        } else {
            None
        }
    }

    /// Get subtitle descriptor
    pub fn as_subtitle(&self) -> Option<&SubtitleDescriptor> {
        if let Self::Subtitle(s) = self {
            Some(s)
        } else {
            None
        }
    }
}

/// Timecode track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimecodeTrack {
    start_timecode: String,
    frame_rate: EditRate,
    drop_frame: bool,
}

impl TimecodeTrack {
    /// Create a new timecode track
    pub fn new(start_timecode: String, frame_rate: EditRate, drop_frame: bool) -> Self {
        Self {
            start_timecode,
            frame_rate,
            drop_frame,
        }
    }

    /// Get start timecode
    pub fn start_timecode(&self) -> &str {
        &self.start_timecode
    }

    /// Get frame rate
    pub fn frame_rate(&self) -> EditRate {
        self.frame_rate
    }

    /// Is drop frame
    pub fn is_drop_frame(&self) -> bool {
        self.drop_frame
    }
}

/// Essence track (one track within an MXF file)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EssenceTrack {
    track_id: u32,
    track_number: u32,
    descriptor: EssenceDescriptor,
    duration: u64,
    edit_rate: EditRate,
}

impl EssenceTrack {
    /// Create a new essence track
    pub fn new(
        track_id: u32,
        track_number: u32,
        descriptor: EssenceDescriptor,
        duration: u64,
        edit_rate: EditRate,
    ) -> Self {
        Self {
            track_id,
            track_number,
            descriptor,
            duration,
            edit_rate,
        }
    }

    /// Get track ID
    pub fn track_id(&self) -> u32 {
        self.track_id
    }

    /// Get track number
    pub fn track_number(&self) -> u32 {
        self.track_number
    }

    /// Get descriptor
    pub fn descriptor(&self) -> &EssenceDescriptor {
        &self.descriptor
    }

    /// Get duration in frames
    pub fn duration(&self) -> u64 {
        self.duration
    }

    /// Get edit rate
    pub fn edit_rate(&self) -> EditRate {
        self.edit_rate
    }

    /// Get duration in seconds
    pub fn duration_seconds(&self) -> f64 {
        self.duration as f64 / self.edit_rate.as_f64()
    }
}

/// MXF essence file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MxfEssence {
    file_path: PathBuf,
    file_id: Uuid,
    tracks: Vec<EssenceTrack>,
    timecode_track: Option<TimecodeTrack>,
    metadata: HashMap<String, String>,
    operational_pattern: String,
}

impl MxfEssence {
    /// Create a new MXF essence
    pub fn new(file_path: PathBuf, file_id: Uuid) -> Self {
        Self {
            file_path,
            file_id,
            tracks: Vec::new(),
            timecode_track: None,
            metadata: HashMap::new(),
            operational_pattern: "OP1a".to_string(),
        }
    }

    /// Get file path
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// Get file ID
    pub fn file_id(&self) -> Uuid {
        self.file_id
    }

    /// Get tracks
    pub fn tracks(&self) -> &[EssenceTrack] {
        &self.tracks
    }

    /// Add a track
    pub fn add_track(&mut self, track: EssenceTrack) {
        self.tracks.push(track);
    }

    /// Get timecode track
    pub fn timecode_track(&self) -> Option<&TimecodeTrack> {
        self.timecode_track.as_ref()
    }

    /// Set timecode track
    pub fn set_timecode_track(&mut self, timecode: TimecodeTrack) {
        self.timecode_track = Some(timecode);
    }

    /// Get metadata
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get operational pattern
    pub fn operational_pattern(&self) -> &str {
        &self.operational_pattern
    }

    /// Set operational pattern
    pub fn set_operational_pattern(&mut self, pattern: String) {
        self.operational_pattern = pattern;
    }

    /// Get all video tracks
    pub fn video_tracks(&self) -> Vec<&EssenceTrack> {
        self.tracks
            .iter()
            .filter(|t| t.descriptor.is_video())
            .collect()
    }

    /// Get all audio tracks
    pub fn audio_tracks(&self) -> Vec<&EssenceTrack> {
        self.tracks
            .iter()
            .filter(|t| t.descriptor.is_audio())
            .collect()
    }

    /// Get all subtitle tracks
    pub fn subtitle_tracks(&self) -> Vec<&EssenceTrack> {
        self.tracks
            .iter()
            .filter(|t| t.descriptor.is_subtitle())
            .collect()
    }

    /// Get primary video track (first video track)
    pub fn primary_video_track(&self) -> Option<&EssenceTrack> {
        self.tracks.iter().find(|t| t.descriptor.is_video())
    }

    /// Get total duration (maximum duration across all tracks)
    pub fn total_duration(&self) -> u64 {
        self.tracks.iter().map(|t| t.duration).max().unwrap_or(0)
    }

    /// Parse MXF file by scanning KLV (Key-Length-Value) triplets per SMPTE 377M.
    ///
    /// Locates the Header Partition Pack, reads the Primer Pack to build the
    /// local-tag-to-UL map, then iterates Object Sets to extract:
    /// - Operational pattern from the partition pack
    /// - Edit rate and duration from MaterialPackage / TimelineTrack
    /// - Picture descriptor (width, height, bit depth, color primaries)
    /// - Sound descriptor (sample rate, bit depth, channel count)
    pub fn from_file(path: &Path) -> ImfResult<Self> {
        use std::fs;
        use std::io::Read;

        if !path.exists() {
            return Err(ImfError::FileNotFound(path.to_string_lossy().to_string()));
        }

        let file_id = Uuid::new_v4();
        let mut essence = Self::new(path.to_path_buf(), file_id);

        essence.set_metadata("format".to_string(), "MXF".to_string());
        essence.set_metadata(
            "filename".to_string(),
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        );

        let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        essence.set_metadata("file_size".to_string(), file_size.to_string());

        // Read up to 256 KB for header scanning (large files may have many metadata sets)
        let scan_size = file_size.min(262_144) as usize;
        let mut buf = vec![0u8; scan_size];
        {
            let mut file = fs::File::open(path).map_err(ImfError::Io)?;
            let n = file.read(&mut buf).map_err(ImfError::Io)?;
            buf.truncate(n);
        }

        // SMPTE UL designator prefix present in all MXF keys
        let ul_prefix = [0x06u8, 0x0E, 0x2B, 0x34];

        // ---- Step 1: Find Header Partition Pack ----
        // Key starts with: 06 0E 2B 34 02 05 01 01 0D 01 02 01 01 02 ...
        let header_pp_prefix = [
            0x06u8, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x02,
        ];
        if let Some(pos) = mxf_find_bytes(&buf, &header_pp_prefix) {
            if let Some((val_off, _)) = mxf_ber_length(&buf, pos + 16) {
                // Header Partition Pack value layout (SMPTE 377M Table 15):
                // MajorVersion(2) MinorVersion(2) KAGSize(4) ThisPartition(8)
                // PreviousPartition(8) FooterPartition(8) HeaderByteCount(8)
                // IndexByteCount(8) IndexSID(4) BodyOffset(8) BodySID(4)
                // OperationalPattern(16) EssenceContainers(BatchOf<UL>)
                // Total before OperationalPattern = 2+2+4+8+8+8+8+8+4+8+4 = 64 bytes
                let op_off = val_off + 64;
                if op_off + 16 <= buf.len() {
                    let op_ul = &buf[op_off..op_off + 16];
                    // OP UL: 06 0E 2B 34 04 01 01 XX 0D 01 02 01 XX XX XX XX
                    if op_ul[..4] == ul_prefix && op_ul[8] == 0x0D {
                        let op_item = op_ul[12];
                        let op_package = op_ul[13];
                        let op_str = match (op_item, op_package) {
                            (0x01, 0x01) => "OP1a",
                            (0x01, 0x02) => "OP1b",
                            (0x01, 0x03) => "OP1c",
                            (0x02, 0x01) => "OP2a",
                            (0x02, 0x02) => "OP2b",
                            (0x03, 0x01) => "OP3a",
                            (0x03, 0x02) => "OP3b",
                            (0x10, 0x00) => "OPAtom",
                            _ => "OP1a",
                        };
                        essence.set_operational_pattern(op_str.to_string());
                    }

                    // EssenceContainers batch: 4-byte item count + 4-byte item_len + N×16 ULs
                    let ec_off = op_off + 16;
                    if ec_off + 8 <= buf.len() {
                        let ec_count = mxf_read_u32_be(&buf, ec_off);
                        essence
                            .set_metadata("essence_containers".to_string(), ec_count.to_string());
                        if ec_count > 0 && ec_off + 8 + 16 <= buf.len() {
                            // First essence container UL encodes wrapped codec
                            let ec_ul = &buf[ec_off + 8..ec_off + 8 + 16];
                            if ec_ul[..4] == ul_prefix {
                                // Byte [13] identifies the mapping kind
                                let codec_name = match ec_ul[13] {
                                    0x01 => "MPEG-2",
                                    0x04 => "AVC/H.264",
                                    0x09 => "VC-3/DNxHD",
                                    0x0B | 0x10 => "JPEG 2000",
                                    0x15 => "VC-3/DNxHD",
                                    0x1B => "VC-1",
                                    0x1C => "ProRes",
                                    _ => "Unknown",
                                };
                                essence.set_metadata(
                                    "essence_codec".to_string(),
                                    codec_name.to_string(),
                                );
                            }
                        }
                    }
                }
            }
        }

        // ---- Step 2: Build Primer Pack local-tag → UL map ----
        // Primer Pack key: 06 0E 2B 34 02 05 01 01 0D 01 02 01 01 05 01 00
        let primer_key = [
            0x06u8, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x05,
            0x01, 0x00,
        ];
        let mut primer_map: HashMap<u16, [u8; 16]> = HashMap::new();
        if let Some(primer_pos) = mxf_find_bytes(&buf, &primer_key) {
            if let Some((val_off, val_len)) = mxf_ber_length(&buf, primer_pos + 16) {
                let val_end = (val_off + val_len).min(buf.len());
                if val_end >= val_off + 8 {
                    let item_count = mxf_read_u32_be(&buf, val_off) as usize;
                    // Each entry: 2-byte local tag + 16-byte UL = 18 bytes
                    for i in 0..item_count {
                        let e = val_off + 8 + i * 18;
                        if e + 18 > val_end {
                            break;
                        }
                        let local_tag = mxf_read_u16_be(&buf, e);
                        let mut ul = [0u8; 16];
                        ul.copy_from_slice(&buf[e + 2..e + 18]);
                        primer_map.insert(local_tag, ul);
                    }
                }
            }
        }
        essence.set_metadata("primer_entries".to_string(), primer_map.len().to_string());

        // ---- Step 3: Scan all KLV Object Sets for essence metadata ----
        let mut info = MxfParsedInfo::default();
        let mut scan_pos = 0usize;

        while scan_pos + 16 < buf.len() {
            // All MXF keys start with the SMPTE UL prefix
            if buf[scan_pos..scan_pos + 4] != ul_prefix {
                scan_pos += 1;
                continue;
            }

            let key16 = &buf[scan_pos..scan_pos + 16];
            if let Some((val_off, val_len)) = mxf_ber_length(&buf, scan_pos + 16) {
                let val_end = (val_off + val_len).min(buf.len());
                let val = &buf[val_off..val_end];

                // Object sets follow SMPTE 377M §8.6:
                // Key bytes [4..8] = 02 53 01 01 for "set" registry items
                // Key byte  [14]   = class identifier
                if key16[4] == 0x02 && key16[5] == 0x53 && key16[8] == 0x0D {
                    match key16[14] {
                        0x36 | 0x37 => {
                            // MaterialPackage or SourcePackage: extract edit rate + duration
                            mxf_parse_package(val, &primer_map, &mut info);
                        }
                        0x27..=0x29 => {
                            // GenericPicture / CDCI / RGBA essence descriptor
                            mxf_parse_picture(val, &primer_map, &mut info);
                        }
                        0x42 | 0x47 | 0x48 => {
                            // GenericSound / AES3 / WaveAudio essence descriptor
                            mxf_parse_sound(val, &primer_map, &mut info);
                        }
                        _ => {}
                    }
                }

                let next = val_off + val_len;
                scan_pos = if next > scan_pos { next } else { scan_pos + 1 };
            } else {
                scan_pos += 1;
            }
        }

        // ---- Step 4: Build EssenceTrack objects from collected info ----
        let edit_rate = if info.edit_rate_num > 0 && info.edit_rate_den > 0 {
            EditRate::new(info.edit_rate_num, info.edit_rate_den)
        } else {
            EditRate::new(25, 1)
        };
        let duration = info.duration.unwrap_or(0);

        if info.has_picture {
            let width = info.picture_width.unwrap_or(1920);
            let height = info.picture_height.unwrap_or(1080);
            let bit_depth = info.picture_bit_depth.unwrap_or(8);
            let codec = essence
                .get_metadata("essence_codec")
                .unwrap_or("JPEG 2000")
                .to_string();

            let color_space = match info.picture_color_primaries {
                1 => ColorSpace::Rec709,
                9 => ColorSpace::Rec2020,
                10 => ColorSpace::DciP3,
                _ => ColorSpace::Rec709,
            };
            let transfer = match info.picture_transfer_char {
                16 => TransferCharacteristic::Pq,
                18 => TransferCharacteristic::Hlg,
                _ => TransferCharacteristic::Rec709,
            };

            let mut video_desc = VideoDescriptor::new(
                width,
                height,
                edit_rate,
                bit_depth,
                color_space,
                transfer,
                codec,
            );
            if let (Some(asp_w), Some(asp_h)) = (info.aspect_num, info.aspect_den) {
                video_desc.set_aspect_ratio((asp_w, asp_h));
            }
            video_desc.set_interlaced(info.picture_interlaced);

            essence.add_track(EssenceTrack::new(
                1,
                1,
                EssenceDescriptor::Video(video_desc),
                duration,
                edit_rate,
            ));
        }

        if info.has_sound {
            let sample_rate = info.audio_sample_rate.unwrap_or(48000);
            let bit_depth = info.audio_bit_depth.unwrap_or(24);
            let channels = info.audio_channels.unwrap_or(1);

            let audio_desc = AudioDescriptor::new(
                sample_rate,
                bit_depth,
                AudioChannelConfig::from_channel_count(channels),
                "PCM".to_string(),
            );
            let audio_edit_rate = EditRate::new(sample_rate, 1);
            let audio_duration = if edit_rate.as_f64() > 0.0 {
                (duration as f64 / edit_rate.as_f64() * f64::from(sample_rate)) as u64
            } else {
                duration
            };

            essence.add_track(EssenceTrack::new(
                2,
                2,
                EssenceDescriptor::Audio(audio_desc),
                audio_duration,
                audio_edit_rate,
            ));
        }

        if duration > 0 && edit_rate.as_f64() > 0.0 {
            let secs = duration as f64 / edit_rate.as_f64();
            essence.set_metadata("duration_seconds".to_string(), format!("{secs:.3}"));
            essence.set_metadata("duration_frames".to_string(), duration.to_string());
        }
        essence.set_metadata(
            "edit_rate".to_string(),
            format!("{}/{}", edit_rate.numerator(), edit_rate.denominator()),
        );

        Ok(essence)
    }

    /// Validate essence file
    pub fn validate(&self) -> ImfResult<Vec<String>> {
        let mut warnings = Vec::new();

        // Check if file exists
        if !self.file_path.exists() {
            return Err(ImfError::FileNotFound(
                self.file_path.to_string_lossy().to_string(),
            ));
        }

        // Check if we have at least one track
        if self.tracks.is_empty() {
            warnings.push("No tracks found in essence file".to_string());
        }

        // Check video track consistency
        let video_tracks = self.video_tracks();
        if video_tracks.len() > 1 {
            warnings.push(format!(
                "Multiple video tracks found: {}",
                video_tracks.len()
            ));
        }

        // Check audio track consistency
        for (i, track) in self.audio_tracks().iter().enumerate() {
            if let Some(audio_desc) = track.descriptor.as_audio() {
                if audio_desc.sample_rate() != 48000 {
                    warnings.push(format!(
                        "Audio track {} has non-standard sample rate: {} Hz",
                        i,
                        audio_desc.sample_rate()
                    ));
                }
            }
        }

        Ok(warnings)
    }

    /// Get essence info as string
    pub fn info_string(&self) -> String {
        let mut info = String::new();

        info.push_str(&format!("File: {}\n", self.file_path.display()));
        info.push_str(&format!("ID: {}\n", self.file_id));
        info.push_str(&format!(
            "Operational Pattern: {}\n",
            self.operational_pattern
        ));
        info.push_str(&format!("Tracks: {}\n", self.tracks.len()));

        if let Some(video) = self.primary_video_track() {
            if let Some(desc) = video.descriptor.as_video() {
                info.push_str(&format!(
                    "  Video: {} @ {:.2} fps, {} bit, {}\n",
                    desc.resolution_string(),
                    desc.frame_rate_float(),
                    desc.bit_depth(),
                    desc.color_space()
                ));
            }
        }

        for (i, track) in self.audio_tracks().iter().enumerate() {
            if let Some(desc) = track.descriptor.as_audio() {
                info.push_str(&format!(
                    "  Audio {}: {} Hz, {} bit, {}\n",
                    i,
                    desc.sample_rate(),
                    desc.bit_depth(),
                    desc.channel_config()
                ));
            }
        }

        info
    }
}

/// Builder for creating MXF essence with tracks
#[allow(dead_code)]
pub struct MxfEssenceBuilder {
    essence: MxfEssence,
}

#[allow(dead_code)]
impl MxfEssenceBuilder {
    /// Create a new builder
    pub fn new(path: PathBuf, file_id: Uuid) -> Self {
        Self {
            essence: MxfEssence::new(path, file_id),
        }
    }

    /// Add a video track
    pub fn with_video_track(
        mut self,
        track_id: u32,
        descriptor: VideoDescriptor,
        duration: u64,
        edit_rate: EditRate,
    ) -> Self {
        let track = EssenceTrack::new(
            track_id,
            track_id,
            EssenceDescriptor::Video(descriptor),
            duration,
            edit_rate,
        );
        self.essence.add_track(track);
        self
    }

    /// Add an audio track
    pub fn with_audio_track(
        mut self,
        track_id: u32,
        descriptor: AudioDescriptor,
        duration: u64,
        edit_rate: EditRate,
    ) -> Self {
        let track = EssenceTrack::new(
            track_id,
            track_id,
            EssenceDescriptor::Audio(descriptor),
            duration,
            edit_rate,
        );
        self.essence.add_track(track);
        self
    }

    /// Add a subtitle track
    pub fn with_subtitle_track(
        mut self,
        track_id: u32,
        descriptor: SubtitleDescriptor,
        duration: u64,
        edit_rate: EditRate,
    ) -> Self {
        let track = EssenceTrack::new(
            track_id,
            track_id,
            EssenceDescriptor::Subtitle(descriptor),
            duration,
            edit_rate,
        );
        self.essence.add_track(track);
        self
    }

    /// Set timecode track
    pub fn with_timecode(mut self, timecode: TimecodeTrack) -> Self {
        self.essence.set_timecode_track(timecode);
        self
    }

    /// Set operational pattern
    pub fn with_operational_pattern(mut self, pattern: String) -> Self {
        self.essence.set_operational_pattern(pattern);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.essence.set_metadata(key, value);
        self
    }

    /// Build the essence
    pub fn build(self) -> MxfEssence {
        self.essence
    }
}

// ============================================================================
// MXF binary parsing helpers (internal to this module)
// ============================================================================

/// Accumulated metadata gathered while scanning MXF KLV object sets.
#[derive(Default)]
struct MxfParsedInfo {
    edit_rate_num: u32,
    edit_rate_den: u32,
    duration: Option<u64>,
    has_picture: bool,
    picture_width: Option<u32>,
    picture_height: Option<u32>,
    picture_bit_depth: Option<u8>,
    picture_color_primaries: u8,
    picture_transfer_char: u8,
    picture_interlaced: bool,
    aspect_num: Option<u32>,
    aspect_den: Option<u32>,
    has_sound: bool,
    audio_sample_rate: Option<u32>,
    audio_bit_depth: Option<u8>,
    audio_channels: Option<u8>,
}

/// Decode a BER-encoded length and return `(value_offset, value_length)`.
///
/// Returns `None` if `buf` is too short.
fn mxf_ber_length(buf: &[u8], pos: usize) -> Option<(usize, usize)> {
    let b = *buf.get(pos)?;
    if b == 0x80 {
        // Indefinite form — unsupported; treat as 0
        return Some((pos + 1, 0));
    }
    if b & 0x80 == 0 {
        // Short definite form: single byte
        Some((pos + 1, b as usize))
    } else {
        // Long definite form: lower 7 bits = number of length bytes
        let n = (b & 0x7F) as usize;
        if n == 0 || n > 8 || pos + 1 + n > buf.len() {
            return None;
        }
        let mut length = 0usize;
        for i in 0..n {
            length = (length << 8) | buf[pos + 1 + i] as usize;
        }
        Some((pos + 1 + n, length))
    }
}

/// Find the first occurrence of `needle` in `haystack`.
fn mxf_find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Read a big-endian u16 from `buf[pos..]`.
fn mxf_read_u16_be(buf: &[u8], pos: usize) -> u16 {
    if pos + 2 > buf.len() {
        return 0;
    }
    u16::from_be_bytes([buf[pos], buf[pos + 1]])
}

/// Read a big-endian u32 from `buf[pos..]`.
fn mxf_read_u32_be(buf: &[u8], pos: usize) -> u32 {
    if pos + 4 > buf.len() {
        return 0;
    }
    u32::from_be_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]])
}

/// Read a big-endian i64 from `buf[pos..]`.
fn mxf_read_i64_be(buf: &[u8], pos: usize) -> i64 {
    if pos + 8 > buf.len() {
        return 0;
    }
    i64::from_be_bytes([
        buf[pos],
        buf[pos + 1],
        buf[pos + 2],
        buf[pos + 3],
        buf[pos + 4],
        buf[pos + 5],
        buf[pos + 6],
        buf[pos + 7],
    ])
}

/// Parse an MXF "local set" (sequence of 2-byte tag + 2-byte length + value TLVs).
///
/// The `primer_map` is used for reference (not currently dereferenced here,
/// because well-known static tags are used directly per SMPTE 380M).
fn mxf_iter_local_set<F>(data: &[u8], mut callback: F)
where
    F: FnMut(u16, &[u8]),
{
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let tag = mxf_read_u16_be(data, pos);
        let len = mxf_read_u16_be(data, pos + 2) as usize;
        let val_start = pos + 4;
        let val_end = (val_start + len).min(data.len());
        callback(tag, &data[val_start..val_end]);
        pos = val_end;
    }
}

/// Extract edit rate and duration from a MaterialPackage or SourcePackage set.
fn mxf_parse_package(data: &[u8], _primer: &HashMap<u16, [u8; 16]>, info: &mut MxfParsedInfo) {
    mxf_iter_local_set(data, |tag, val| {
        match tag {
            // Package Edit Rate (static UL 06 0E 2B 34 01 01 01 01 07 02 01 01 01 03 00 00)
            // registered as local tag 0x4901 in many implementations
            0x4901 => {
                if val.len() >= 8 {
                    let num = mxf_read_u32_be(val, 0);
                    let den = mxf_read_u32_be(val, 4);
                    if num > 0 && den > 0 {
                        info.edit_rate_num = num;
                        info.edit_rate_den = den;
                    }
                }
            }
            // Package Duration (local tag 0x4202 is Component::Duration in many MXF files)
            0x4202 | 0x1501 => {
                if val.len() >= 8 {
                    let d = mxf_read_i64_be(val, 0);
                    if d > 0 {
                        info.duration = Some(d as u64);
                    }
                }
            }
            _ => {}
        }
    });
}

/// Extract picture essence descriptor fields.
///
/// Well-known local tags from SMPTE 380M / RP 210:
/// 0x3203 = StoredWidth, 0x3202 = StoredHeight,
/// 0x3301 = ComponentDepth, 0x3201 = FrameLayout,
/// 0x320E = AspectRatio,  0x7F01 = ColorPrimaries,
/// 0x7F02 = CodingEquations / Transfer Char.
fn mxf_parse_picture(data: &[u8], _primer: &HashMap<u16, [u8; 16]>, info: &mut MxfParsedInfo) {
    info.has_picture = true;
    mxf_iter_local_set(data, |tag, val| {
        match tag {
            0x3203 => {
                if val.len() >= 4 {
                    info.picture_width = Some(mxf_read_u32_be(val, 0));
                }
            }
            0x3202 => {
                if val.len() >= 4 {
                    info.picture_height = Some(mxf_read_u32_be(val, 0));
                }
            }
            0x3301 => {
                if !val.is_empty() {
                    // ComponentDepth is a u8 or u32 depending on encoding
                    info.picture_bit_depth = Some(if val.len() >= 4 {
                        mxf_read_u32_be(val, 0).min(255) as u8
                    } else {
                        val[0]
                    });
                }
            }
            // FrameLayout: 0 = FullFrame (progressive), 1/2/3/4 = interlaced variants
            0x3201 => {
                if !val.is_empty() {
                    let layout = if val.len() >= 4 {
                        mxf_read_u32_be(val, 0) as u8
                    } else {
                        val[0]
                    };
                    info.picture_interlaced = layout != 0;
                }
            }
            // AspectRatio: two u32 numerator/denominator
            0x320E => {
                if val.len() >= 8 {
                    let n = mxf_read_u32_be(val, 0);
                    let d = mxf_read_u32_be(val, 4);
                    if n > 0 && d > 0 {
                        info.aspect_num = Some(n);
                        info.aspect_den = Some(d);
                    }
                }
            }
            // ColorPrimaries UL (SMPTE 2067): 1=Rec709, 9=Rec2020, 10=P3
            0x7F01 => {
                if !val.is_empty() {
                    info.picture_color_primaries = val[val.len().saturating_sub(1)];
                }
            }
            // CaptureGamma / TransferCharacteristic
            0x7F02 | 0x3210 => {
                if !val.is_empty() {
                    info.picture_transfer_char = val[val.len().saturating_sub(1)];
                }
            }
            _ => {}
        }
    });
}

/// Extract sound essence descriptor fields.
///
/// Well-known local tags:
/// 0x3D03 = AudioSampleRate, 0x3D01 = QuantizationBits,
/// 0x3D07 = ChannelCount
fn mxf_parse_sound(data: &[u8], _primer: &HashMap<u16, [u8; 16]>, info: &mut MxfParsedInfo) {
    info.has_sound = true;
    mxf_iter_local_set(data, |tag, val| {
        match tag {
            // AudioSampleRate: two u32 (numerator / denominator)
            0x3D03 => {
                if val.len() >= 8 {
                    let num = mxf_read_u32_be(val, 0);
                    let _den = mxf_read_u32_be(val, 4);
                    if num > 0 {
                        info.audio_sample_rate = Some(num);
                    }
                } else if val.len() >= 4 {
                    let rate = mxf_read_u32_be(val, 0);
                    if rate > 0 {
                        info.audio_sample_rate = Some(rate);
                    }
                }
            }
            // QuantizationBits
            0x3D01 => {
                if !val.is_empty() {
                    info.audio_bit_depth = Some(if val.len() >= 4 {
                        mxf_read_u32_be(val, 0).min(255) as u8
                    } else {
                        val[0]
                    });
                }
            }
            // ChannelCount
            0x3D07 => {
                if !val.is_empty() {
                    info.audio_channels = Some(if val.len() >= 4 {
                        mxf_read_u32_be(val, 0).min(255) as u8
                    } else {
                        val[0]
                    });
                }
            }
            _ => {}
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_space() {
        assert_eq!(ColorSpace::Rec709.as_str(), "Rec.709");
        assert_eq!(ColorSpace::from_str("BT.709"), ColorSpace::Rec709);
        assert_eq!(ColorSpace::from_str("DCI-P3"), ColorSpace::DciP3);
    }

    #[test]
    fn test_audio_channel_config() {
        assert_eq!(AudioChannelConfig::Stereo.channel_count(), 2);
        assert_eq!(AudioChannelConfig::Surround51.channel_count(), 6);
        assert_eq!(
            AudioChannelConfig::from_channel_count(8),
            AudioChannelConfig::Surround71
        );
    }

    #[test]
    fn test_video_descriptor() {
        let desc = VideoDescriptor::new(
            1920,
            1080,
            EditRate::fps_24(),
            10,
            ColorSpace::Rec709,
            TransferCharacteristic::Rec709,
            "JPEG2000".to_string(),
        );

        assert_eq!(desc.width(), 1920);
        assert_eq!(desc.height(), 1080);
        assert_eq!(desc.bit_depth(), 10);
        assert_eq!(desc.resolution_string(), "1920x1080");
        assert!(!desc.is_hdr());
    }

    #[test]
    fn test_audio_descriptor() {
        let desc = AudioDescriptor::new(48000, 24, AudioChannelConfig::Stereo, "PCM".to_string());

        assert_eq!(desc.sample_rate(), 48000);
        assert_eq!(desc.bit_depth(), 24);
        assert_eq!(desc.channel_count(), 2);
    }

    #[test]
    fn test_essence_track() {
        let video_desc = VideoDescriptor::new(
            1920,
            1080,
            EditRate::fps_24(),
            10,
            ColorSpace::Rec709,
            TransferCharacteristic::Rec709,
            "JPEG2000".to_string(),
        );

        let track = EssenceTrack::new(
            1,
            1,
            EssenceDescriptor::Video(video_desc),
            1000,
            EditRate::fps_24(),
        );

        assert_eq!(track.track_id(), 1);
        assert_eq!(track.duration(), 1000);
        assert!(track.descriptor().is_video());
    }

    #[test]
    fn test_mxf_essence_builder() {
        let video_desc = VideoDescriptor::new(
            1920,
            1080,
            EditRate::fps_24(),
            10,
            ColorSpace::Rec709,
            TransferCharacteristic::Rec709,
            "JPEG2000".to_string(),
        );

        let audio_desc =
            AudioDescriptor::new(48000, 24, AudioChannelConfig::Stereo, "PCM".to_string());

        let essence = MxfEssenceBuilder::new(PathBuf::from("test.mxf"), Uuid::new_v4())
            .with_video_track(1, video_desc, 1000, EditRate::fps_24())
            .with_audio_track(2, audio_desc, 1000, EditRate::fps_24())
            .with_operational_pattern("OP1a".to_string())
            .build();

        assert_eq!(essence.tracks().len(), 2);
        assert_eq!(essence.video_tracks().len(), 1);
        assert_eq!(essence.audio_tracks().len(), 1);
        assert_eq!(essence.operational_pattern(), "OP1a");
    }

    #[test]
    fn test_transfer_characteristic_hdr() {
        assert!(TransferCharacteristic::Pq.is_hdr());
        assert!(TransferCharacteristic::Hlg.is_hdr());
        assert!(!TransferCharacteristic::Rec709.is_hdr());
    }
}
