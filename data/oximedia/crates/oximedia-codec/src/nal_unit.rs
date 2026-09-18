//! NAL unit handling for H.264/H.265.
//!
//! Provides NAL unit type classification, start code detection, RBSP
//! trailing-bit stripping, and a simple Annex B byte-stream parser.

#![allow(dead_code)]

/// H.264 (AVC) NAL unit type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H264NalType {
    /// Unspecified.
    Unspecified,
    /// Non-IDR slice (P- or B-frame).
    NonIdrSlice,
    /// Slice data partition A.
    SlicePartitionA,
    /// Slice data partition B.
    SlicePartitionB,
    /// Slice data partition C.
    SlicePartitionC,
    /// IDR slice (keyframe).
    IdrSlice,
    /// Supplemental enhancement information.
    Sei,
    /// Sequence parameter set.
    Sps,
    /// Picture parameter set.
    Pps,
    /// Access unit delimiter.
    AccessUnitDelimiter,
    /// End of sequence.
    EndOfSequence,
    /// End of stream.
    EndOfStream,
    /// Filler data.
    FillerData,
    /// Reserved or unknown type.
    Reserved(u8),
}

impl H264NalType {
    /// Parses the first byte of a NAL unit header (ignoring forbidden zero bit
    /// and NRI bits, extracting the 5-bit nal_unit_type).
    #[must_use]
    pub fn from_header_byte(byte: u8) -> Self {
        match byte & 0x1F {
            0 => Self::Unspecified,
            1 => Self::NonIdrSlice,
            2 => Self::SlicePartitionA,
            3 => Self::SlicePartitionB,
            4 => Self::SlicePartitionC,
            5 => Self::IdrSlice,
            6 => Self::Sei,
            7 => Self::Sps,
            8 => Self::Pps,
            9 => Self::AccessUnitDelimiter,
            10 => Self::EndOfSequence,
            11 => Self::EndOfStream,
            12 => Self::FillerData,
            n => Self::Reserved(n),
        }
    }

    /// Returns `true` if this NAL unit is a slice (IDR or non-IDR).
    #[must_use]
    pub fn is_slice(self) -> bool {
        matches!(self, Self::NonIdrSlice | Self::IdrSlice)
    }

    /// Returns `true` if this NAL unit is a parameter set (SPS or PPS).
    #[must_use]
    pub fn is_parameter_set(self) -> bool {
        matches!(self, Self::Sps | Self::Pps)
    }
}

/// H.265 (HEVC) NAL unit type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H265NalType {
    /// Trailing non-reference picture (B-frame).
    TrailN,
    /// Trailing reference picture (P-frame).
    TrailR,
    /// Instantaneous decoder refresh picture (IDR).
    IdrWRadl,
    /// IDR without leading pictures.
    IdrNLp,
    /// Video parameter set.
    Vps,
    /// Sequence parameter set.
    Sps,
    /// Picture parameter set.
    Pps,
    /// Access unit delimiter.
    Aud,
    /// Supplemental enhancement information (prefix).
    PrefixSei,
    /// Reserved or unknown type.
    Reserved(u8),
}

impl H265NalType {
    /// Parses from the 6-bit nal_unit_type field of an HEVC NAL header.
    #[must_use]
    pub fn from_nal_type(nal_type: u8) -> Self {
        match nal_type & 0x3F {
            0 => Self::TrailN,
            1 => Self::TrailR,
            19 => Self::IdrWRadl,
            20 => Self::IdrNLp,
            32 => Self::Vps,
            33 => Self::Sps,
            34 => Self::Pps,
            35 => Self::Aud,
            39 => Self::PrefixSei,
            n => Self::Reserved(n),
        }
    }

    /// Returns `true` if this is an IDR picture.
    #[must_use]
    pub fn is_idr(self) -> bool {
        matches!(self, Self::IdrWRadl | Self::IdrNLp)
    }
}

/// The 3-byte Annex B start code prefix `0x00 0x00 0x01`.
pub const START_CODE_3: [u8; 3] = [0x00, 0x00, 0x01];
/// The 4-byte Annex B start code prefix `0x00 0x00 0x00 0x01`.
pub const START_CODE_4: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

/// Returns `true` if `data` begins with a 3-byte or 4-byte Annex B start code.
#[must_use]
pub fn has_start_code(data: &[u8]) -> bool {
    data.starts_with(&START_CODE_4) || data.starts_with(&START_CODE_3)
}

/// Removes trailing RBSP stop bit and zero-padding bytes from a raw RBSP
/// payload, returning the trimmed slice.
#[must_use]
pub fn strip_rbsp_trailing(data: &[u8]) -> &[u8] {
    // Walk backwards, skipping 0x00 bytes until we hit the stop bit (0x80).
    let mut end = data.len();
    while end > 0 {
        end -= 1;
        let byte = data[end];
        if byte == 0x80 {
            return &data[..end];
        }
        if byte != 0x00 {
            // No trailing stop bit found; return original.
            return data;
        }
    }
    data
}

/// A parsed NAL unit extracted from an Annex B byte stream.
#[derive(Debug, Clone)]
pub struct NalUnit<'a> {
    /// Raw bytes of this NAL unit (without the leading start code).
    pub data: &'a [u8],
    /// Byte offset of the first byte of this NAL unit within the original stream.
    pub offset: usize,
}

impl NalUnit<'_> {
    /// Returns the NAL unit header byte (first byte of `data`).
    #[must_use]
    pub fn header_byte(&self) -> Option<u8> {
        self.data.first().copied()
    }

    /// Parses the H.264 NAL type from the header byte.
    #[must_use]
    pub fn h264_type(&self) -> Option<H264NalType> {
        self.header_byte().map(H264NalType::from_header_byte)
    }
}

/// Parses all NAL units from an Annex B byte stream.
///
/// Returns a `Vec` of [`NalUnit`] references into the original `data` slice.
#[must_use]
pub fn parse_annex_b(data: &[u8]) -> Vec<NalUnit<'_>> {
    let mut units = Vec::new();
    let mut i = 0usize;
    let len = data.len();

    while i < len {
        // Detect start code: try 4-byte first, then 3-byte.
        let start_code_len = if i + 4 <= len && data[i..i + 4] == START_CODE_4 {
            4
        } else if i + 3 <= len && data[i..i + 3] == START_CODE_3 {
            3
        } else {
            i += 1;
            continue;
        };

        let nal_start = i + start_code_len;
        // Find the end of this NAL unit (next start code or end of data).
        let mut j = nal_start;
        let mut found_next = false;
        while j + 3 <= len {
            if data[j..j + 3] == START_CODE_3 || (j + 4 <= len && data[j..j + 4] == START_CODE_4) {
                found_next = true;
                break;
            }
            j += 1;
        }
        if !found_next {
            j = len;
        }
        // Strip trailing zero bytes before the next start code.
        let mut nal_end = j;
        while nal_end > nal_start && data[nal_end - 1] == 0x00 {
            nal_end -= 1;
        }
        if nal_end > nal_start {
            units.push(NalUnit {
                data: &data[nal_start..nal_end],
                offset: nal_start,
            });
        }
        i = j;
    }
    units
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h264_idr_type() {
        assert_eq!(H264NalType::from_header_byte(0x65), H264NalType::IdrSlice);
    }

    #[test]
    fn test_h264_sps_type() {
        assert_eq!(H264NalType::from_header_byte(0x67), H264NalType::Sps);
    }

    #[test]
    fn test_h264_pps_type() {
        assert_eq!(H264NalType::from_header_byte(0x68), H264NalType::Pps);
    }

    #[test]
    fn test_h264_non_idr_slice() {
        assert_eq!(
            H264NalType::from_header_byte(0x41),
            H264NalType::NonIdrSlice
        );
    }

    #[test]
    fn test_h264_is_slice() {
        assert!(H264NalType::IdrSlice.is_slice());
        assert!(H264NalType::NonIdrSlice.is_slice());
        assert!(!H264NalType::Sps.is_slice());
    }

    #[test]
    fn test_h264_is_parameter_set() {
        assert!(H264NalType::Sps.is_parameter_set());
        assert!(H264NalType::Pps.is_parameter_set());
        assert!(!H264NalType::IdrSlice.is_parameter_set());
    }

    #[test]
    fn test_h265_idr_type() {
        assert_eq!(H265NalType::from_nal_type(19), H265NalType::IdrWRadl);
        assert!(H265NalType::IdrWRadl.is_idr());
    }

    #[test]
    fn test_h265_non_idr_not_idr() {
        assert!(!H265NalType::TrailR.is_idr());
    }

    #[test]
    fn test_has_start_code_4byte() {
        assert!(has_start_code(&[0x00, 0x00, 0x00, 0x01, 0x67]));
    }

    #[test]
    fn test_has_start_code_3byte() {
        assert!(has_start_code(&[0x00, 0x00, 0x01, 0x67]));
    }

    #[test]
    fn test_no_start_code() {
        assert!(!has_start_code(&[0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_strip_rbsp_trailing_removes_stop_bit() {
        let data = &[0xAB, 0xCD, 0x80, 0x00, 0x00];
        let stripped = strip_rbsp_trailing(data);
        assert_eq!(stripped, &[0xAB, 0xCD]);
    }

    #[test]
    fn test_strip_rbsp_no_stop_bit_returns_original() {
        let data = &[0x01, 0x02, 0x03];
        assert_eq!(strip_rbsp_trailing(data), data);
    }

    #[test]
    fn test_parse_annex_b_single_nal() {
        let stream = [0x00, 0x00, 0x00, 0x01, 0x67, 0xAB, 0xCD];
        let units = parse_annex_b(&stream);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].data[0], 0x67);
    }

    #[test]
    fn test_parse_annex_b_multiple_nals() {
        let stream = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x11, 0x00, 0x00, 0x01, 0x68, 0x22,
        ];
        let units = parse_annex_b(&stream);
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].h264_type(), Some(H264NalType::Sps));
        assert_eq!(units[1].h264_type(), Some(H264NalType::Pps));
    }
}
