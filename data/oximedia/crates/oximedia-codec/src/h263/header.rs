//! H.263 header structures and parsing.
//!
//! This module handles parsing of picture headers, GOB (Group of Blocks) headers,
//! and macroblock headers according to ITU-T H.263 specification.

use super::bitstream::BitReader;
use crate::CodecError;

/// Picture format (source format).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PictureFormat {
    /// Forbidden format.
    Forbidden,
    /// Sub-QCIF (128x96).
    SubQcif,
    /// QCIF (176x144).
    Qcif,
    /// CIF (352x288).
    Cif,
    /// 4CIF (704x576).
    FourCif,
    /// 16CIF (1408x1152).
    SixteenCif,
    /// Reserved format.
    Reserved,
    /// Extended PTYPE (custom format).
    Extended,
}

impl PictureFormat {
    /// Get dimensions for this format.
    ///
    /// # Returns
    ///
    /// (width, height) or None for extended/forbidden formats.
    #[must_use]
    pub const fn dimensions(&self) -> Option<(u32, u32)> {
        match self {
            Self::SubQcif => Some((128, 96)),
            Self::Qcif => Some((176, 144)),
            Self::Cif => Some((352, 288)),
            Self::FourCif => Some((704, 576)),
            Self::SixteenCif => Some((1408, 1152)),
            Self::Forbidden | Self::Reserved | Self::Extended => None,
        }
    }

    /// Create from 3-bit source format code.
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Forbidden,
            1 => Self::SubQcif,
            2 => Self::Qcif,
            3 => Self::Cif,
            4 => Self::FourCif,
            5 => Self::SixteenCif,
            6 => Self::Reserved,
            7 => Self::Extended,
            _ => Self::Forbidden,
        }
    }

    /// Get source format code.
    #[must_use]
    pub const fn to_code(&self) -> u8 {
        match self {
            Self::Forbidden => 0,
            Self::SubQcif => 1,
            Self::Qcif => 2,
            Self::Cif => 3,
            Self::FourCif => 4,
            Self::SixteenCif => 5,
            Self::Reserved => 6,
            Self::Extended => 7,
        }
    }
}

/// Picture coding type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PictureCodingType {
    /// Intra-coded frame (I-frame).
    Intra,
    /// Predicted frame (P-frame).
    Inter,
}

/// H.263 picture header.
#[derive(Clone, Debug)]
pub struct PictureHeader {
    /// Temporal reference (frame number modulo 256).
    pub temporal_reference: u8,
    /// Picture coding type (I or P frame).
    pub picture_type: PictureCodingType,
    /// Source format.
    pub source_format: PictureFormat,
    /// Picture width (for extended format).
    pub width: Option<u32>,
    /// Picture height (for extended format).
    pub height: Option<u32>,
    /// Quantizer parameter.
    pub quantizer: u8,
    /// Unrestricted motion vectors enabled.
    pub umv_mode: bool,
    /// Syntax-based Arithmetic Coding mode.
    pub sac_mode: bool,
    /// Advanced Prediction mode.
    pub ap_mode: bool,
    /// PB-frames mode.
    pub pb_frames_mode: bool,
    /// Continuous Presence Multipoint enabled.
    pub cpm_enabled: bool,
    /// Picture Sub Bitstream Indicator.
    pub psbi: Option<u8>,
    /// Temporal, SNR, and Spatial Scalability mode.
    pub tss_mode: bool,
}

impl PictureHeader {
    /// Parse picture header from bitstream.
    ///
    /// # Arguments
    ///
    /// * `reader` - Bit reader positioned at picture start code
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid or incomplete.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, CodecError> {
        // Picture Start Code (PSC): 0x0000 1000 0000 0000 0000 0000 (22 bits)
        let psc = reader.read_bits(22)?;
        if psc != 0x20 {
            return Err(CodecError::InvalidData(format!(
                "Invalid picture start code: 0x{psc:06X}"
            )));
        }

        // Temporal Reference (TR): 8 bits
        let temporal_reference = reader.read_bits(8)? as u8;

        // Type Information: at least 13 bits
        // PTYPE consists of multiple fields

        // Bit 1: Always 1
        let marker = reader.read_bit()?;
        if !marker {
            return Err(CodecError::InvalidData("PTYPE marker bit not set".into()));
        }

        // Bit 2: Always 0
        let split_screen = reader.read_bit()?;
        if split_screen {
            return Err(CodecError::InvalidData("Split screen not supported".into()));
        }

        // Bit 3: Document camera indicator (0)
        let _doc_camera = reader.read_bit()?;

        // Bit 4: Freeze Picture Release (0)
        let _freeze_picture = reader.read_bit()?;

        // Bits 5-7: Source Format (3 bits)
        let source_format_code = reader.read_bits(3)? as u8;
        let source_format = PictureFormat::from_code(source_format_code);

        // Bit 8: Picture Coding Type (0=Intra, 1=Inter)
        let picture_type = if reader.read_bit()? {
            PictureCodingType::Inter
        } else {
            PictureCodingType::Intra
        };

        // Bit 9: Unrestricted Motion Vector mode (0=off, 1=on)
        let umv_mode = reader.read_bit()?;

        // Bit 10: Syntax-based Arithmetic Coding mode (0=off, 1=on)
        let sac_mode = reader.read_bit()?;

        // Bit 11: Advanced Prediction mode (0=off, 1=on)
        let ap_mode = reader.read_bit()?;

        // Bit 12: PB-frames mode (0=off, 1=on)
        let pb_frames_mode = reader.read_bit()?;

        // Bit 13: Always 0
        let marker2 = reader.read_bit()?;
        if marker2 {
            return Err(CodecError::InvalidData(
                "PTYPE reserved bit not zero".into(),
            ));
        }

        // Parse extended PTYPE if source format is Extended
        let (width, height) = if source_format == PictureFormat::Extended {
            // PLUSPTYPE parsing
            // UUI: Update Full Extended PTYPE
            let _uui = reader.read_bits(3)?;

            // Source Format: 3 bits
            let _extended_source_format = reader.read_bits(3)?;

            // Custom Picture Format
            let custom_pcf = reader.read_bit()?;

            let (w, h) = if custom_pcf {
                // Custom Picture Format (CPM)
                // Pixel Aspect Ratio: 4 bits
                let _par = reader.read_bits(4)?;

                // Picture Width Indication: 9 bits (in units of 4 pixels)
                let pw = reader.read_bits(9)?;
                let width = (pw + 1) * 4;

                // Marker bit
                let _marker = reader.read_bit()?;

                // Picture Height Indication: 9 bits
                let ph = reader.read_bits(9)?;
                let height = ph * 4;

                (Some(width), Some(height))
            } else {
                (None, None)
            };

            (w, h)
        } else {
            (None, None)
        };

        // CPM (Continuous Presence Multipoint): 1 bit
        let cpm_enabled = reader.read_bit()?;

        // PSBI (Picture Sub Bitstream Indicator): 2 bits (if CPM enabled)
        let psbi = if cpm_enabled {
            Some(reader.read_bits(2)? as u8)
        } else {
            None
        };

        // TSS (Temporal, SNR, Spatial Scalability): not in baseline
        let tss_mode = false;

        // PQUANT: Quantizer (5 bits)
        let quantizer = reader.read_bits(5)? as u8;
        if quantizer == 0 {
            return Err(CodecError::InvalidData("Invalid quantizer value: 0".into()));
        }

        // CPM: 1 bit (already parsed if needed)
        // PSBI: 2 bits (already parsed if needed)

        // PEI (Extra Insertion Information): read until 0
        loop {
            let pei = reader.read_bit()?;
            if !pei {
                break;
            }
            // PSPARE: 8 bits of spare data
            let _pspare = reader.read_bits(8)?;
        }

        Ok(Self {
            temporal_reference,
            picture_type,
            source_format,
            width,
            height,
            quantizer,
            umv_mode,
            sac_mode,
            ap_mode,
            pb_frames_mode,
            cpm_enabled,
            psbi,
            tss_mode,
        })
    }

    /// Get picture dimensions.
    ///
    /// # Returns
    ///
    /// (width, height) or error if format is invalid.
    pub fn dimensions(&self) -> Result<(u32, u32), CodecError> {
        if let (Some(w), Some(h)) = (self.width, self.height) {
            return Ok((w, h));
        }

        self.source_format
            .dimensions()
            .ok_or_else(|| CodecError::InvalidData("Invalid source format".into()))
    }

    /// Check if this is an I-frame.
    #[must_use]
    pub const fn is_intra(&self) -> bool {
        matches!(self.picture_type, PictureCodingType::Intra)
    }

    /// Check if this is a P-frame.
    #[must_use]
    pub const fn is_inter(&self) -> bool {
        matches!(self.picture_type, PictureCodingType::Inter)
    }
}

/// GOB (Group of Blocks) header.
#[derive(Clone, Debug)]
pub struct GobHeader {
    /// GOB number (0-17 for CIF).
    pub gob_number: u8,
    /// GOB frame ID.
    pub gob_frame_id: u8,
    /// Quantizer for this GOB.
    pub quantizer: u8,
}

impl GobHeader {
    /// Parse GOB header from bitstream.
    ///
    /// # Arguments
    ///
    /// * `reader` - Bit reader
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, CodecError> {
        // GBSC (GOB Start Code): 17 bits (0x00001)
        let gbsc = reader.read_bits(17)?;
        if gbsc != 0x01 {
            return Err(CodecError::InvalidData(format!(
                "Invalid GOB start code: 0x{gbsc:05X}"
            )));
        }

        // GN (GOB Number): 5 bits
        let gob_number = reader.read_bits(5)? as u8;

        // GFID (GOB Frame ID): 2 bits
        let gob_frame_id = reader.read_bits(2)? as u8;

        // GQUANT (GOB Quantizer): 5 bits
        let quantizer = reader.read_bits(5)? as u8;

        Ok(Self {
            gob_number,
            gob_frame_id,
            quantizer,
        })
    }
}

/// Macroblock type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacroblockType {
    /// Inter (predicted from previous frame).
    Inter,
    /// Inter with quantizer update.
    InterQ,
    /// Inter with 4 motion vectors.
    Inter4V,
    /// Intra (coded without prediction).
    Intra,
    /// Intra with quantizer update.
    IntraQ,
    /// Skipped macroblock.
    Skipped,
}

impl MacroblockType {
    /// Check if macroblock is intra-coded.
    #[must_use]
    pub const fn is_intra(&self) -> bool {
        matches!(self, Self::Intra | Self::IntraQ)
    }

    /// Check if macroblock is inter-coded.
    #[must_use]
    pub const fn is_inter(&self) -> bool {
        matches!(self, Self::Inter | Self::InterQ | Self::Inter4V)
    }

    /// Check if macroblock has quantizer update.
    #[must_use]
    pub const fn has_quant(&self) -> bool {
        matches!(self, Self::InterQ | Self::IntraQ)
    }

    /// Check if macroblock uses 4 motion vectors.
    #[must_use]
    pub const fn has_4mv(&self) -> bool {
        matches!(self, Self::Inter4V)
    }
}

/// Macroblock header.
#[derive(Clone, Debug)]
pub struct MacroblockHeader {
    /// Macroblock type.
    pub mb_type: MacroblockType,
    /// Coded block pattern for chrominance (0-3).
    pub cbpc: u8,
    /// Coded block pattern for luminance (0-15).
    pub cbpy: u8,
    /// Quantizer update (if mb_type has quantizer update).
    pub dquant: Option<i8>,
    /// Motion vector (for inter macroblocks).
    pub mvd_x: Option<i16>,
    /// Motion vector (for inter macroblocks).
    pub mvd_y: Option<i16>,
    /// Additional motion vectors for Inter4V mode.
    pub mvd_additional: Option<[(i16, i16); 4]>,
}

impl MacroblockHeader {
    /// Create a new macroblock header.
    #[must_use]
    pub const fn new(mb_type: MacroblockType) -> Self {
        Self {
            mb_type,
            cbpc: 0,
            cbpy: 0,
            dquant: None,
            mvd_x: None,
            mvd_y: None,
            mvd_additional: None,
        }
    }

    /// Get full coded block pattern (6 bits: [Y0 Y1 Y2 Y3 Cb Cr]).
    #[must_use]
    pub const fn cbp(&self) -> u8 {
        (self.cbpy << 2) | self.cbpc
    }

    /// Check if a block is coded.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block index (0-5: Y0-Y3, Cb, Cr)
    #[must_use]
    pub const fn is_block_coded(&self, block_idx: usize) -> bool {
        if block_idx > 5 {
            return false;
        }

        let cbp = self.cbp();
        ((cbp >> (5 - block_idx)) & 1) != 0
    }
}

/// End of sequence header.
#[derive(Clone, Copy, Debug)]
pub struct EndOfSequence;

impl EndOfSequence {
    /// Parse end of sequence code.
    ///
    /// # Arguments
    ///
    /// * `reader` - Bit reader
    ///
    /// # Errors
    ///
    /// Returns error if code is invalid.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, CodecError> {
        // EOS code: 22 bits (0x00 00 1F)
        let eos = reader.read_bits(22)?;
        if eos != 0x1F {
            return Err(CodecError::InvalidData(format!(
                "Invalid EOS code: 0x{eos:06X}"
            )));
        }

        Ok(Self)
    }
}

/// Slice header (for error resilience).
#[derive(Clone, Debug)]
pub struct SliceHeader {
    /// Slice start MB address.
    pub mb_address: u16,
    /// Quantizer for slice.
    pub quantizer: u8,
}

impl SliceHeader {
    /// Parse slice header.
    ///
    /// # Arguments
    ///
    /// * `reader` - Bit reader
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, CodecError> {
        // Slice Start Code: 17 bits
        let ssc = reader.read_bits(17)?;
        if ssc != 0x01 {
            return Err(CodecError::InvalidData(format!(
                "Invalid slice start code: 0x{ssc:05X}"
            )));
        }

        // MBA (Macroblock Address): variable length
        let mb_address = reader.read_bits(9)? as u16;

        // SQUANT (Slice Quantizer): 5 bits
        let quantizer = reader.read_bits(5)? as u8;

        Ok(Self {
            mb_address,
            quantizer,
        })
    }
}

/// Reference picture selection information.
#[derive(Clone, Debug)]
pub struct ReferencePictureSelection {
    /// Temporal reference for reference picture.
    pub temporal_reference: u8,
    /// Picture number.
    pub picture_number: u16,
}

/// Advanced modes configuration.
#[derive(Clone, Debug, Default)]
pub struct AdvancedModes {
    /// Unrestricted motion vectors.
    pub umv: bool,
    /// Syntax-based arithmetic coding.
    pub sac: bool,
    /// Advanced prediction.
    pub ap: bool,
    /// PB-frames.
    pub pb_frames: bool,
    /// Deblocking filter.
    pub deblocking: bool,
    /// Slice structured mode.
    pub slice_structured: bool,
    /// Reference picture selection.
    pub rps: bool,
    /// Independent segment decoding.
    pub isd: bool,
    /// Alternative inter VLC.
    pub aiv: bool,
    /// Modified quantization.
    pub mq: bool,
}

impl AdvancedModes {
    /// Check if any advanced mode is enabled.
    #[must_use]
    pub const fn is_advanced(&self) -> bool {
        self.umv
            || self.sac
            || self.ap
            || self.pb_frames
            || self.deblocking
            || self.slice_structured
            || self.rps
            || self.isd
            || self.aiv
            || self.mq
    }
}
