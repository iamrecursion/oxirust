#![allow(dead_code)]
//! Format-specific container parsers for the multi-format prober.
//!
//! Covers MP4/ISOBMFF, EBML/MKV/WebM, Ogg BOS, WAV/RIFF, and FLAC STREAMINFO.

use super::container_probe::{DetailedContainerInfo, DetailedStreamInfo};

// ─── MP4 / ISOBMFF box walker ─────────────────────────────────────────────────

pub(crate) fn parse_moov(data: &[u8], info: &mut DetailedContainerInfo) {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        let fourcc = &data[offset + 4..offset + 8];
        let body = &data[offset + 8..offset + box_size];

        match fourcc {
            b"mvhd" => {
                if !body.is_empty() {
                    let version = body[0];
                    if version == 0 && body.len() >= 20 {
                        let ts = read_u32_be(body, 12);
                        let dur = read_u32_be(body, 16);
                        if ts > 0 {
                            info.duration_ms = Some(u64::from(dur) * 1000 / u64::from(ts));
                        }
                    } else if version == 1 && body.len() >= 28 {
                        let ts = read_u32_be(body, 20);
                        let dur = read_u64_be(body, 24);
                        if ts > 0 {
                            info.duration_ms = Some(dur * 1000 / u64::from(ts));
                        }
                    }
                }
            }
            b"trak" => {
                let mut s = DetailedStreamInfo::default();
                s.index = info.streams.len() as u32;
                parse_trak(body, &mut s);
                if !s.stream_type.is_empty() {
                    info.streams.push(s);
                }
            }
            _ => {}
        }

        offset += box_size;
    }
}

fn parse_trak(data: &[u8], stream: &mut DetailedStreamInfo) {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        let fourcc = &data[offset + 4..offset + 8];
        let body = &data[offset + 8..offset + box_size];

        if fourcc == b"mdia" {
            parse_mdia(body, stream);
        }

        offset += box_size;
    }
}

fn parse_mdia(data: &[u8], stream: &mut DetailedStreamInfo) {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        let fourcc = &data[offset + 4..offset + 8];
        let body = &data[offset + 8..offset + box_size];

        match fourcc {
            b"hdlr" => {
                if body.len() >= 12 {
                    let handler = &body[8..12];
                    stream.stream_type = match handler {
                        b"vide" => "video".into(),
                        b"soun" => "audio".into(),
                        b"subt" | b"text" => "subtitle".into(),
                        _ => "data".into(),
                    };
                }
            }
            b"minf" => parse_minf(body, stream),
            _ => {}
        }

        offset += box_size;
    }
}

fn parse_minf(data: &[u8], stream: &mut DetailedStreamInfo) {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        let fourcc = &data[offset + 4..offset + 8];
        let body = &data[offset + 8..offset + box_size];

        if fourcc == b"stbl" {
            parse_stbl(body, stream);
        }

        offset += box_size;
    }
}

fn parse_stbl(data: &[u8], stream: &mut DetailedStreamInfo) {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let box_size = read_u32_be(data, offset) as usize;
        if box_size < 8 || offset + box_size > data.len() {
            break;
        }
        let fourcc = &data[offset + 4..offset + 8];
        let body = &data[offset + 8..offset + box_size];

        if fourcc == b"stsd" && body.len() >= 8 {
            let entry_body = &body[8..];
            if entry_body.len() >= 8 {
                let codec_bytes = &entry_body[4..8];
                stream.codec = String::from_utf8_lossy(codec_bytes).trim().to_lowercase();
            }
        }

        offset += box_size;
    }
}

// ─── EBML / MKV / WebM walker ─────────────────────────────────────────────────

fn read_ebml_element(data: &[u8], offset: usize) -> Option<(u64, usize, usize)> {
    if offset >= data.len() {
        return None;
    }
    let (id, id_len) = read_ebml_id(data, offset)?;
    let (size, size_len) = read_ebml_size(data, offset + id_len)?;
    Some((id, size, id_len + size_len))
}

fn read_ebml_id(data: &[u8], offset: usize) -> Option<(u64, usize)> {
    let first = *data.get(offset)?;
    let len = if first & 0x80 != 0 {
        1
    } else if first & 0x40 != 0 {
        2
    } else if first & 0x20 != 0 {
        3
    } else if first & 0x10 != 0 {
        4
    } else {
        return None;
    };
    if offset + len > data.len() {
        return None;
    }
    let mut id: u64 = 0;
    for i in 0..len {
        id = (id << 8) | u64::from(data[offset + i]);
    }
    Some((id, len))
}

fn read_ebml_size(data: &[u8], offset: usize) -> Option<(usize, usize)> {
    let first = *data.get(offset)?;
    let (len, mask) = if first & 0x80 != 0 {
        (1usize, 0x7Fu8)
    } else if first & 0x40 != 0 {
        (2, 0x3F)
    } else if first & 0x20 != 0 {
        (3, 0x1F)
    } else if first & 0x10 != 0 {
        (4, 0x0F)
    } else if first & 0x08 != 0 {
        (5, 0x07)
    } else if first & 0x04 != 0 {
        (6, 0x03)
    } else if first & 0x02 != 0 {
        (7, 0x01)
    } else if first & 0x01 != 0 {
        (8, 0x00)
    } else {
        return None;
    };
    if offset + len > data.len() {
        return None;
    }
    let mut size: usize = (data[offset] & mask) as usize;
    for i in 1..len {
        size = (size << 8) | data[offset + i] as usize;
    }
    Some((size, len))
}

/// EBML element IDs of interest.
const EBML_ID_SEGMENT: u64 = 0x18538067;
const EBML_ID_INFO: u64 = 0x1549A966;
const EBML_ID_DURATION: u64 = 0x4489;
const EBML_ID_TIMECODE_SCALE: u64 = 0x2AD7B1;
const EBML_ID_TRACKS: u64 = 0x1654AE6B;
const EBML_ID_TRACK_ENTRY: u64 = 0xAE;
const EBML_ID_TRACK_TYPE: u64 = 0x83;
const EBML_ID_CODEC_ID: u64 = 0x86;
const _EBML_ID_TRACK_NUMBER: u64 = 0xD7;
const EBML_ID_VIDEO: u64 = 0xE0;
const EBML_ID_PIXEL_WIDTH: u64 = 0xB0;
const EBML_ID_PIXEL_HEIGHT: u64 = 0xBA;
const EBML_ID_AUDIO_ELEM: u64 = 0xE1;
const EBML_ID_SAMPLING_FREQ: u64 = 0xB5;
const EBML_ID_CHANNELS_ELEM: u64 = 0x9F;
const EBML_ID_DOCTYPE: u64 = 0x4282;

pub(crate) fn parse_ebml_for_info(data: &[u8], info: &mut DetailedContainerInfo) {
    let mut offset = 0usize;
    while offset + 4 <= data.len() {
        let Some((id, size, hlen)) = read_ebml_element(data, offset) else {
            break;
        };
        let body_start = offset + hlen;
        let body_end = body_start + size;

        if id == EBML_ID_DOCTYPE && body_end <= data.len() {
            let doctype = String::from_utf8_lossy(&data[body_start..body_end]);
            if doctype == "webm" {
                info.format = "webm".into();
            }
        }

        if id == EBML_ID_SEGMENT {
            let seg_end = if body_end > data.len() {
                data.len()
            } else {
                body_end
            };
            parse_segment_body(&data[body_start..seg_end], info);
            break;
        }

        if size == 0 || body_end > data.len() {
            break;
        }
        offset += hlen + size;
    }

    if info.format.is_empty() {
        info.format = "mkv".into();
    }
}

fn parse_segment_body(data: &[u8], info: &mut DetailedContainerInfo) {
    let mut offset = 0usize;
    let mut timecode_scale_ns: u64 = 1_000_000;

    while offset + 2 <= data.len() {
        let Some((id, size, hlen)) = read_ebml_element(data, offset) else {
            break;
        };
        let body_start = offset + hlen;
        let body_end = (body_start + size).min(data.len());

        match id {
            EBML_ID_INFO => {
                parse_segment_info(&data[body_start..body_end], info, &mut timecode_scale_ns);
            }
            EBML_ID_TRACKS => {
                parse_tracks_body(&data[body_start..body_end], info);
            }
            _ => {}
        }

        if info.duration_ms.is_some() && !info.streams.is_empty() {
            break;
        }

        let advance = hlen + size;
        if advance == 0 {
            break;
        }
        offset += advance;
    }
}

fn parse_segment_info(data: &[u8], info: &mut DetailedContainerInfo, timecode_scale: &mut u64) {
    let mut offset = 0usize;
    let mut raw_duration: Option<f64> = None;

    while offset + 2 <= data.len() {
        let Some((id, size, hlen)) = read_ebml_element(data, offset) else {
            break;
        };
        let body_start = offset + hlen;
        let body_end = (body_start + size).min(data.len());
        let body = &data[body_start..body_end];

        match id {
            EBML_ID_TIMECODE_SCALE => {
                if !body.is_empty() {
                    let mut v: u64 = 0;
                    for &b in body {
                        v = (v << 8) | u64::from(b);
                    }
                    *timecode_scale = v;
                }
            }
            EBML_ID_DURATION => {
                if body.len() == 4 {
                    let bits = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                    raw_duration = Some(f64::from(f32::from_bits(bits)));
                } else if body.len() == 8 {
                    let bits = u64::from_be_bytes([
                        body[0], body[1], body[2], body[3], body[4], body[5], body[6], body[7],
                    ]);
                    raw_duration = Some(f64::from_bits(bits));
                }
            }
            _ => {}
        }

        let advance = hlen + size;
        if advance == 0 {
            break;
        }
        offset += advance;
    }

    if let Some(dur) = raw_duration {
        if *timecode_scale > 0 {
            #[allow(clippy::cast_precision_loss)]
            let ms = (dur * (*timecode_scale as f64)) / 1_000_000.0;
            info.duration_ms = Some(ms as u64);
        }
    }
}

fn parse_tracks_body(data: &[u8], info: &mut DetailedContainerInfo) {
    let mut offset = 0usize;
    while offset + 2 <= data.len() {
        let Some((id, size, hlen)) = read_ebml_element(data, offset) else {
            break;
        };
        let body_start = offset + hlen;
        let body_end = (body_start + size).min(data.len());

        if id == EBML_ID_TRACK_ENTRY {
            let mut s = DetailedStreamInfo::default();
            s.index = info.streams.len() as u32;
            parse_track_entry(&data[body_start..body_end], &mut s);
            if !s.stream_type.is_empty() {
                info.streams.push(s);
            }
        }

        let advance = hlen + size;
        if advance == 0 {
            break;
        }
        offset += advance;
    }
}

fn parse_track_entry(data: &[u8], stream: &mut DetailedStreamInfo) {
    let mut offset = 0usize;
    while offset + 2 <= data.len() {
        let Some((id, size, hlen)) = read_ebml_element(data, offset) else {
            break;
        };
        let body_start = offset + hlen;
        let body_end = (body_start + size).min(data.len());
        let body = &data[body_start..body_end];

        match id {
            EBML_ID_TRACK_TYPE => {
                let track_type = body.first().copied().unwrap_or(0);
                stream.stream_type = match track_type {
                    1 => "video".into(),
                    2 => "audio".into(),
                    0x11 => "subtitle".into(),
                    _ => "data".into(),
                };
            }
            EBML_ID_CODEC_ID => {
                let codec_str = String::from_utf8_lossy(body).to_string();
                stream.codec = codec_str
                    .trim_end_matches('\0')
                    .replace("V_VP9", "vp9")
                    .replace("V_VP8", "vp8")
                    .replace("V_AV1", "av1")
                    .replace("A_OPUS", "opus")
                    .replace("A_VORBIS", "vorbis")
                    .replace("A_FLAC", "flac")
                    .to_lowercase();
            }
            EBML_ID_VIDEO => {
                parse_video_track(body, stream);
            }
            EBML_ID_AUDIO_ELEM => {
                parse_audio_track(body, stream);
            }
            _ => {}
        }

        let advance = hlen + size;
        if advance == 0 {
            break;
        }
        offset += advance;
    }
}

fn parse_video_track(data: &[u8], stream: &mut DetailedStreamInfo) {
    let mut offset = 0usize;
    while offset + 2 <= data.len() {
        let Some((id, size, hlen)) = read_ebml_element(data, offset) else {
            break;
        };
        let body_start = offset + hlen;
        let body_end = (body_start + size).min(data.len());
        let body = &data[body_start..body_end];

        match id {
            EBML_ID_PIXEL_WIDTH => {
                stream.width = Some(read_ebml_uint(body) as u32);
            }
            EBML_ID_PIXEL_HEIGHT => {
                stream.height = Some(read_ebml_uint(body) as u32);
            }
            _ => {}
        }

        let advance = hlen + size;
        if advance == 0 {
            break;
        }
        offset += advance;
    }
}

fn parse_audio_track(data: &[u8], stream: &mut DetailedStreamInfo) {
    let mut offset = 0usize;
    while offset + 2 <= data.len() {
        let Some((id, size, hlen)) = read_ebml_element(data, offset) else {
            break;
        };
        let body_start = offset + hlen;
        let body_end = (body_start + size).min(data.len());
        let body = &data[body_start..body_end];

        match id {
            EBML_ID_SAMPLING_FREQ => {
                if body.len() == 4 {
                    let bits = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                    stream.sample_rate = Some(f32::from_bits(bits) as u32);
                } else if body.len() == 8 {
                    let bits = u64::from_be_bytes([
                        body[0], body[1], body[2], body[3], body[4], body[5], body[6], body[7],
                    ]);
                    stream.sample_rate = Some(f64::from_bits(bits) as u32);
                }
            }
            EBML_ID_CHANNELS_ELEM => {
                stream.channels = Some(read_ebml_uint(body) as u8);
            }
            _ => {}
        }

        let advance = hlen + size;
        if advance == 0 {
            break;
        }
        offset += advance;
    }
}

fn read_ebml_uint(data: &[u8]) -> u64 {
    let mut v: u64 = 0;
    for &b in data {
        v = (v << 8) | u64::from(b);
    }
    v
}

// ─── Ogg BOS parsing ─────────────────────────────────────────────────────────

pub(crate) fn parse_ogg_bos(data: &[u8], info: &mut DetailedContainerInfo) {
    let mut offset = 0usize;
    let mut stream_idx = 0u32;

    while offset + 27 <= data.len() {
        if &data[offset..offset + 4] != b"OggS" {
            offset += 1;
            continue;
        }
        let header_type = data[offset + 5];
        let is_bos = (header_type & 0x02) != 0;
        let n_segs = data[offset + 26] as usize;

        if offset + 27 + n_segs > data.len() {
            break;
        }
        let payload_offset = offset + 27 + n_segs;
        let payload_len: usize = data[offset + 27..offset + 27 + n_segs]
            .iter()
            .map(|&s| s as usize)
            .sum();

        if is_bos && payload_offset + 8 <= data.len() {
            let payload = &data[payload_offset..];
            let mut s = DetailedStreamInfo {
                index: stream_idx,
                ..Default::default()
            };

            if payload.len() >= 7 && &payload[1..7] == b"vorbis" {
                s.stream_type = "audio".into();
                s.codec = "vorbis".into();
                if payload.len() >= 16 {
                    s.channels = Some(payload[11]);
                    let sr =
                        u32::from_le_bytes([payload[12], payload[13], payload[14], payload[15]]);
                    s.sample_rate = Some(sr);
                }
            } else if payload.len() >= 8 && &payload[..8] == b"OpusHead" {
                s.stream_type = "audio".into();
                s.codec = "opus".into();
                if payload.len() >= 11 {
                    s.channels = Some(payload[9]);
                    let sr =
                        u32::from_le_bytes([payload[12], payload[13], payload[14], payload[15]]);
                    s.sample_rate = Some(sr);
                }
            } else if payload.len() >= 4 && &payload[..4] == b"fLaC" {
                s.stream_type = "audio".into();
                s.codec = "flac".into();
            } else if payload.len() >= 7 && &payload[1..7] == b"theora" {
                s.stream_type = "video".into();
                s.codec = "theora".into();
            }

            if !s.stream_type.is_empty() {
                info.streams.push(s);
                stream_idx += 1;
            }
        }

        offset += 27 + n_segs + payload_len;
    }
}

// ─── WAV / RIFF chunk parsing ─────────────────────────────────────────────────

pub(crate) fn parse_wav_chunks(data: &[u8], info: &mut DetailedContainerInfo) {
    let mut offset = 12usize;
    while offset + 8 <= data.len() {
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;

        if chunk_id == b"fmt " && chunk_size >= 16 && offset + 8 + chunk_size <= data.len() {
            let fmt = &data[offset + 8..offset + 8 + chunk_size];
            let channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            let sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
            let sample_fmt = match bits {
                8 => "u8",
                16 => "s16",
                24 => "s24",
                32 => "s32",
                _ => "unknown",
            };
            info.streams.push(DetailedStreamInfo {
                index: 0,
                stream_type: "audio".into(),
                codec: "pcm".into(),
                sample_rate: Some(sample_rate),
                channels: Some(channels as u8),
                sample_format: Some(sample_fmt.into()),
                ..Default::default()
            });
        } else if chunk_id == b"data" && !info.streams.is_empty() {
            if let Some(s) = info.streams.first() {
                if let (Some(sr), Some(ch)) = (s.sample_rate, s.channels) {
                    let bits = match s.sample_format.as_deref() {
                        Some("u8") => 8u64,
                        Some("s24") => 24,
                        Some("s32") => 32,
                        _ => 16,
                    };
                    let bytes_per_sample = bits / 8;
                    let total_samples = chunk_size as u64 / (bytes_per_sample * u64::from(ch));
                    if sr > 0 {
                        info.duration_ms = Some(total_samples * 1000 / u64::from(sr));
                    }
                }
            }
        }

        let advance = 8 + chunk_size + (chunk_size & 1);
        if advance == 0 {
            break;
        }
        offset += advance;
    }
}

// ─── FLAC STREAMINFO parsing ──────────────────────────────────────────────────

pub(crate) fn parse_flac_streaminfo(data: &[u8], info: &mut DetailedContainerInfo) {
    if data.len() < 42 {
        return;
    }
    let block_type = data[4] & 0x7F;
    if block_type != 0 {
        return;
    }
    let si = &data[8..];
    if si.len() < 34 {
        return;
    }

    let sample_rate =
        (u32::from(si[10]) << 12) | (u32::from(si[11]) << 4) | (u32::from(si[12]) >> 4);
    let channels = ((si[12] >> 1) & 0x07) + 1;
    let bits_per_sample = (((si[12] & 0x01) << 4) | (si[13] >> 4)) + 1;
    let total_samples: u64 = (u64::from(si[13] & 0x0F) << 32)
        | (u64::from(si[14]) << 24)
        | (u64::from(si[15]) << 16)
        | (u64::from(si[16]) << 8)
        | u64::from(si[17]);

    let sample_format = match bits_per_sample {
        8 => "u8",
        16 => "s16",
        24 => "s24",
        32 => "s32",
        _ => "unknown",
    };

    if sample_rate > 0 && total_samples > 0 {
        info.duration_ms = Some(total_samples * 1000 / u64::from(sample_rate));
    }

    info.streams.push(DetailedStreamInfo {
        index: 0,
        stream_type: "audio".into(),
        codec: "flac".into(),
        sample_rate: Some(sample_rate),
        channels: Some(channels),
        sample_format: Some(sample_format.into()),
        ..Default::default()
    });
}

// ─── Byte reading utilities ───────────────────────────────────────────────────

pub(crate) fn read_u32_be(data: &[u8], offset: usize) -> u32 {
    if offset + 4 > data.len() {
        return 0;
    }
    u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

pub(crate) fn read_u64_be(data: &[u8], offset: usize) -> u64 {
    if offset + 8 > data.len() {
        return 0;
    }
    u64::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}
