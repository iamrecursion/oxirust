//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DetailedStreamInfo;

/// MPEG-TS scanning helper (kept in a sub-module to avoid name collisions).
pub(super) mod mpegts_probe {
    use super::DetailedStreamInfo;
    use crate::demux::mpegts_enhanced::TsDemuxer;
    /// Stream type byte → codec name mapping (patent-free only).
    fn stream_type_to_codec(st: u8) -> Option<&'static str> {
        match st {
            0x85 => Some("av1"),
            0x84 => Some("vp9"),
            0x83 => Some("vp8"),
            0x81 => Some("opus"),
            0x82 => Some("flac"),
            0x80 => Some("pcm"),
            0x06 => Some("private"),
            _ => None,
        }
    }
    fn stream_type_to_kind(st: u8) -> &'static str {
        match st {
            0x85 | 0x84 | 0x83 | 0x1B | 0x24 => "video",
            0x81 | 0x82 | 0x80 | 0x03 | 0x04 | 0x0F | 0x11 => "audio",
            _ => "data",
        }
    }
    /// Scans `data` for MPEG-TS packets, returning (streams, duration_ms).
    pub fn scan_mpegts(data: &[u8]) -> (Vec<DetailedStreamInfo>, Option<u64>) {
        let mut demux = TsDemuxer::new();
        let scan_end = data.len().min(2 * 1024 * 1024);
        demux.feed(&data[..scan_end]);
        let si = demux.stream_info();
        let duration_ms = demux.duration_ms();
        let mut streams: Vec<DetailedStreamInfo> = Vec::new();
        let mut idx = 0u32;
        for pmt in si.pmts.values() {
            for ps in &pmt.streams {
                let codec = stream_type_to_codec(ps.stream_type)
                    .unwrap_or("unknown")
                    .to_string();
                let kind = stream_type_to_kind(ps.stream_type).to_string();
                let pid_info = si.pids.get(&ps.elementary_pid);
                let mut s = DetailedStreamInfo {
                    index: idx,
                    stream_type: kind.clone(),
                    codec,
                    ..Default::default()
                };
                if let Some(pi) = pid_info {
                    if let (Some(f), Some(l)) = (pi.pts_first, pi.pts_last) {
                        if l > f {
                            s.duration_ms = Some((l - f) / 90);
                        }
                    }
                    if s.duration_ms.is_some() && pi.total_bytes > 0 {
                        let dur_s = s.duration_ms.unwrap_or(1) as u64;
                        if let Some(bitrate) = (pi.total_bytes * 8).checked_div(dur_s) {
                            s.bitrate_kbps = Some(bitrate as u32);
                        }
                    }
                }
                streams.push(s);
                idx += 1;
            }
        }
        (streams, duration_ms)
    }
}
