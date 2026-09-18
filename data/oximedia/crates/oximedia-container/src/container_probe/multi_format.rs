//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DetailedContainerInfo, DetailedStreamInfo};
use crate::container_probe_parsers::{
    parse_ebml_for_info, parse_flac_streaminfo, parse_moov, parse_ogg_bos, parse_wav_chunks,
    read_u32_be, read_u64_be,
};

/// A stateless multi-format container prober that inspects raw byte slices.
///
/// Compared to [`super::types::ContainerProber`] (magic-byte only), `MultiFormatProber`
/// performs a shallow parse of the container structure to discover stream
/// count, codec, dimensions, duration, and basic metadata — all without
/// decoding any compressed data.
///
/// # Supported formats
///
/// | Format | Detection | Duration | Streams |
/// |--------|-----------|----------|---------|
/// | MPEG-TS | ✓ | from PTS | from PMT |
/// | MP4/MOV | ✓ | mvhd | trak/hdlr |
/// | MKV/WebM | ✓ | EBML Segment/Info | TrackEntry |
/// | Ogg | ✓ | BOS codec | codec header |
/// | WAV | ✓ | fmt chunk | PCM params |
/// | FLAC | ✓ | STREAMINFO | sample params |
#[derive(Debug, Default)]
pub struct MultiFormatProber;
impl MultiFormatProber {
    /// Creates a new `MultiFormatProber`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
    /// Probes `data` and returns all available container information.
    #[must_use]
    pub fn probe(data: &[u8]) -> DetailedContainerInfo {
        let mut info = DetailedContainerInfo {
            file_size_bytes: data.len() as u64,
            ..Default::default()
        };
        if data.len() < 8 {
            info.format = "unknown".into();
            return info;
        }
        if data[0] == 0x47 && (data.len() < 376 || data[188] == 0x47) {
            Self::probe_mpegts(data, &mut info);
        } else if data[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            Self::probe_mkv(data, &mut info);
        } else if data.len() >= 8 && &data[4..8] == b"ftyp" {
            Self::probe_mp4(data, &mut info);
        } else if &data[..4] == b"OggS" {
            Self::probe_ogg(data, &mut info);
        } else if &data[..4] == b"RIFF" {
            Self::probe_wav(data, &mut info);
        } else if &data[..4] == b"fLaC" {
            Self::probe_flac(data, &mut info);
        } else if data.len() >= 4 && &data[..4] == b"caff" {
            Self::probe_caf(data, &mut info);
        } else if data.len() >= 8 && data[0..2] == [0x49, 0x49] && data[2..4] == [0x2A, 0x00] {
            Self::probe_dng_tiff(data, &mut info);
        } else if data.len() >= 8 && data[0..2] == [0x4D, 0x4D] && data[2..4] == [0x00, 0x2A] {
            Self::probe_dng_tiff(data, &mut info);
        } else if data.len() >= 16
            && data[0..4] == [0x06, 0x0E, 0x2B, 0x34]
            && data[4..8] == [0x02, 0x05, 0x01, 0x01]
        {
            Self::probe_mxf(data, &mut info);
        } else {
            info.format = "unknown".into();
        }
        if let (Some(dur_ms), sz) = (info.duration_ms, info.file_size_bytes) {
            if let Some(bitrate) = sz.saturating_mul(8).checked_div(dur_ms) {
                info.bitrate_kbps = Some(bitrate as u32);
            }
        }
        info
    }
    /// Returns only the stream list from `data`.
    #[must_use]
    pub fn probe_streams_only(data: &[u8]) -> Vec<DetailedStreamInfo> {
        Self::probe(data).streams
    }
    fn probe_mpegts(data: &[u8], info: &mut DetailedContainerInfo) {
        use super::functions::mpegts_probe::*;
        info.format = "mpeg-ts".into();
        let (streams, duration_ms) = scan_mpegts(data);
        info.streams = streams;
        info.duration_ms = duration_ms;
    }
    fn probe_mp4(data: &[u8], info: &mut DetailedContainerInfo) {
        info.format = "mp4".into();
        let mut offset = 0usize;
        while offset + 8 <= data.len() {
            let box_size = u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            let fourcc = &data[offset + 4..offset + 8];
            if box_size < 8 || offset + box_size > data.len() {
                break;
            }
            if fourcc == b"moov" {
                parse_moov(&data[offset + 8..offset + box_size], info);
                break;
            }
            offset += box_size;
        }
    }
    fn probe_mkv(data: &[u8], info: &mut DetailedContainerInfo) {
        info.format = "mkv".into();
        parse_ebml_for_info(data, info);
    }
    fn probe_ogg(data: &[u8], info: &mut DetailedContainerInfo) {
        info.format = "ogg".into();
        parse_ogg_bos(data, info);
    }
    fn probe_wav(data: &[u8], info: &mut DetailedContainerInfo) {
        info.format = "wav".into();
        if data.len() >= 12 && &data[8..12] == b"WAVE" {
            parse_wav_chunks(data, info);
        }
    }
    fn probe_flac(data: &[u8], info: &mut DetailedContainerInfo) {
        info.format = "flac".into();
        parse_flac_streaminfo(data, info);
    }
    fn probe_caf(data: &[u8], info: &mut DetailedContainerInfo) {
        info.format = "caf".into();
        if data.len() < 8 {
            return;
        }
        let version = u16::from_be_bytes([data[4], data[5]]);
        info.metadata
            .insert("caf_version".into(), format!("{version}"));
        let mut offset = 8usize;
        while offset + 12 <= data.len() {
            let chunk_type = &data[offset..offset + 4];
            let chunk_size = read_u64_be(data, offset + 4);
            if chunk_type == b"desc" && chunk_size >= 32 && offset + 44 <= data.len() {
                let desc = &data[offset + 12..];
                let sr = f64::from_be_bytes([
                    desc[0], desc[1], desc[2], desc[3], desc[4], desc[5], desc[6], desc[7],
                ]);
                let codec = String::from_utf8_lossy(&desc[8..12]).trim().to_string();
                let ch = if desc.len() >= 28 {
                    read_u32_be(desc, 24)
                } else {
                    0
                };
                let mut s = DetailedStreamInfo {
                    index: 0,
                    stream_type: "audio".into(),
                    codec,
                    ..Default::default()
                };
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    s.sample_rate = Some(sr as u32);
                    if ch > 0 && ch < 256 {
                        s.channels = Some(ch as u8);
                    }
                }
                info.streams.push(s);
            }
            // `12 + chunk_size` can itself overflow usize before the checked
            // offset advance below (chunk_size is an untrusted u64 read from the
            // file). Fold the header constant into a checked add so an oversized
            // chunk_size stops the scan cleanly instead of wrapping/panicking.
            let Some(advance) = 12usize.checked_add(chunk_size as usize) else {
                break;
            };
            match offset.checked_add(advance) {
                Some(new_offset) => offset = new_offset,
                None => break,
            }
        }
    }
    fn probe_dng_tiff(data: &[u8], info: &mut DetailedContainerInfo) {
        let is_le = data[0] == 0x49;
        let ru16 = |off: usize| -> u16 {
            if off + 2 > data.len() {
                return 0;
            }
            if is_le {
                u16::from_le_bytes([data[off], data[off + 1]])
            } else {
                u16::from_be_bytes([data[off], data[off + 1]])
            }
        };
        let ru32 = |off: usize| -> u32 {
            if off + 4 > data.len() {
                return 0;
            }
            if is_le {
                u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            } else {
                u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            }
        };
        let ifd_offset = ru32(4) as usize;
        if ifd_offset + 2 > data.len() {
            info.format = "tiff".into();
            return;
        }
        let entry_count = ru16(ifd_offset) as usize;
        let (mut found_dng, mut width, mut height) = (false, 0u32, 0u32);
        for i in 0..entry_count {
            let off = ifd_offset + 2 + i * 12;
            if off + 12 > data.len() {
                break;
            }
            match ru16(off) {
                0xC612 => found_dng = true,
                0x0100 => width = ru32(off + 8),
                0x0101 => height = ru32(off + 8),
                _ => {}
            }
        }
        if found_dng {
            info.format = "dng".into();
            let mut s = DetailedStreamInfo {
                index: 0,
                stream_type: "video".into(),
                codec: "raw".into(),
                ..Default::default()
            };
            if width > 0 {
                s.width = Some(width);
            }
            if height > 0 {
                s.height = Some(height);
            }
            info.streams.push(s);
        } else {
            info.format = "tiff".into();
        }
    }
    fn probe_mxf(data: &[u8], info: &mut DetailedContainerInfo) {
        info.format = "mxf".into();
        if data.len() >= 16 {
            let pt = data[13];
            let label = match pt {
                0x02 => "header_partition",
                0x03 => "body_partition",
                0x04 => "footer_partition",
                _ => "unknown_partition",
            };
            info.metadata
                .insert("mxf_partition_type".into(), label.into());
        }
        if data.len() >= 12 && data[8] == 0x0D && data[9] == 0x01 {
            info.metadata
                .insert("mxf_registry".into(), "smpte_rdd".into());
        }
        if data.len() >= 64 {
            info.streams.push(DetailedStreamInfo {
                index: 0,
                stream_type: "video".into(),
                codec: "mxf_essence".into(),
                ..Default::default()
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Security regression (P1 #8): `probe_caf` advances by `12 + chunk_size`
    /// where `chunk_size` is an untrusted u64. A size near u64::MAX must not
    /// overflow that add (panic in debug / wrap in release).
    #[test]
    fn probe_caf_oversized_chunk_size_no_panic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"caff"); // magic → dispatches to probe_caf
        data.extend_from_slice(&[0, 1, 0, 0]); // version + flags
        data.extend_from_slice(b"free"); // a chunk type
        data.extend_from_slice(&u64::MAX.to_be_bytes()); // huge chunk_size
                                                         // Must not panic; the scan stops cleanly on the overflowing advance.
        let _ = MultiFormatProber::probe(&data);
    }
}
