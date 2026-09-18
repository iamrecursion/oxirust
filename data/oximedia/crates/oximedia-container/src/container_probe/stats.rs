//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::multi_format::MultiFormatProber;
use super::types::{DetailedContainerInfo, DetailedStreamStats};

/// Compute detailed per-second bitrate and keyframe-interval statistics from
/// a raw container byte slice.
///
/// # Algorithm
///
/// 1. Runs [`MultiFormatProber::probe`] to obtain stream metadata (codec,
///    duration, type).
/// 2. For **MPEG-TS** data: replays the byte slice through the enhanced
///    [`crate::demux::mpegts_enhanced::TsDemuxer`] at packet granularity, bucketing payload bytes into 1-second
///    windows and collecting PTS timestamps of PUSI-marked video packets as
///    keyframe proxies.
/// 3. For all other formats: falls back to a single-bucket histogram derived
///    from the aggregate bitrate reported by the prober, and sets
///    `keyframe_intervals_s = None`.
///
/// # Errors
///
/// Returns an error only if `data` is empty.
pub fn probe_detailed(data: &[u8]) -> oximedia_core::OxiResult<Vec<DetailedStreamStats>> {
    if data.is_empty() {
        return Err(oximedia_core::OxiError::Parse {
            offset: 0,
            message: "probe_detailed: empty data".into(),
        });
    }
    let base = MultiFormatProber::probe(data);
    if base.format == "mpeg-ts" {
        probe_detailed_mpegts(data, &base)
    } else {
        probe_detailed_fallback(data, &base)
    }
}
fn probe_detailed_mpegts(
    data: &[u8],
    base: &DetailedContainerInfo,
) -> oximedia_core::OxiResult<Vec<DetailedStreamStats>> {
    use crate::demux::mpegts_enhanced::TsDemuxer;
    const WINDOW_S: f64 = 1.0;
    const TS_CLOCK: f64 = 90_000.0;
    let mut demux = TsDemuxer::new();
    let packets = demux.feed(data);
    let si = demux.stream_info();
    let mut pid_to_idx: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
    let mut pid_to_type: std::collections::HashMap<u16, &str> = std::collections::HashMap::new();
    {
        let mut stream_idx = 0usize;
        for pmt in si.pmts.values() {
            for ps in &pmt.streams {
                pid_to_idx.insert(ps.elementary_pid, stream_idx);
                pid_to_type.insert(ps.elementary_pid, stream_type_kind(ps.stream_type));
                stream_idx += 1;
            }
        }
    }
    let num_streams = pid_to_idx.len();
    let mut window_bits: Vec<Vec<u64>> = vec![Vec::new(); num_streams];
    let mut kf_pts: Vec<Option<Vec<f64>>> = (0..num_streams)
        .map(|i| {
            let pid = pid_to_idx
                .iter()
                .find(|(_, &v)| v == i)
                .map(|(&k, _)| k)
                .unwrap_or(u16::MAX);
            if pid_to_type.get(&pid).copied() == Some("video") {
                Some(Vec::new())
            } else {
                None
            }
        })
        .collect();
    for pkt in &packets {
        let idx = match pid_to_idx.get(&pkt.pid) {
            Some(&i) => i,
            None => continue,
        };
        let bits = 188u64 * 8;
        let window_idx = if let Some(pts) = pkt.pts {
            (pts as f64 / TS_CLOCK / WINDOW_S) as usize
        } else {
            window_bits[idx].len()
        };
        if window_idx >= window_bits[idx].len() {
            window_bits[idx].resize(window_idx + 1, 0);
        }
        window_bits[idx][window_idx] += bits;
        if pkt.payload_unit_start {
            if let Some(ref mut kf_list) = kf_pts[idx] {
                if let Some(pts) = pkt.pts {
                    kf_list.push(pts as f64 / TS_CLOCK);
                }
            }
        }
    }
    let mut stats: Vec<DetailedStreamStats> = Vec::with_capacity(num_streams);
    for stream_idx in 0..num_streams {
        let base_stream = base.streams.get(stream_idx);
        let codec_id = base_stream
            .map(|s| s.codec.clone())
            .unwrap_or_else(|| "unknown".into());
        let duration_s = base_stream
            .and_then(|s| s.duration_ms)
            .map(|ms| ms as f64 / 1000.0)
            .unwrap_or(0.0);
        let histogram = window_bits[stream_idx].clone();
        let (mean, p50, p95, max) = bitrate_percentiles(&histogram);
        let (kf_intervals, kf_mean, kf_p50, kf_p95, kf_max) =
            compute_kf_interval_stats(kf_pts[stream_idx].as_deref());
        stats.push(DetailedStreamStats {
            stream_index: stream_idx,
            codec_id,
            duration_s,
            bitrate_window_s: WINDOW_S,
            bitrate_histogram: histogram,
            bitrate_mean: mean,
            bitrate_p50: p50,
            bitrate_p95: p95,
            bitrate_max: max,
            keyframe_intervals_s: kf_intervals,
            keyframe_interval_mean: kf_mean,
            keyframe_interval_p50: kf_p50,
            keyframe_interval_p95: kf_p95,
            keyframe_interval_max: kf_max,
        });
    }
    Ok(stats)
}
fn probe_detailed_fallback(
    _data: &[u8],
    base: &DetailedContainerInfo,
) -> oximedia_core::OxiResult<Vec<DetailedStreamStats>> {
    let mut stats = Vec::with_capacity(base.streams.len());
    for (idx, stream) in base.streams.iter().enumerate() {
        let histogram = if let Some(kbps) = stream.bitrate_kbps {
            vec![u64::from(kbps) * 1000]
        } else if let (Some(kbps), Some(dur_ms)) = (base.bitrate_kbps, base.duration_ms) {
            let n_windows = ((dur_ms as f64 / 1000.0).ceil() as usize).max(1);
            vec![u64::from(kbps) * 1000; n_windows]
        } else {
            Vec::new()
        };
        let (mean, p50, p95, max) = bitrate_percentiles(&histogram);
        let duration_s = stream
            .duration_ms
            .or(base.duration_ms)
            .map(|ms| ms as f64 / 1000.0)
            .unwrap_or(0.0);
        stats.push(DetailedStreamStats {
            stream_index: idx,
            codec_id: stream.codec.clone(),
            duration_s,
            bitrate_window_s: 1.0,
            bitrate_histogram: histogram,
            bitrate_mean: mean,
            bitrate_p50: p50,
            bitrate_p95: p95,
            bitrate_max: max,
            keyframe_intervals_s: None,
            keyframe_interval_mean: None,
            keyframe_interval_p50: None,
            keyframe_interval_p95: None,
            keyframe_interval_max: None,
        });
    }
    Ok(stats)
}
/// Maps an ISO 13818-1 stream-type byte to `"video"`, `"audio"`, or `"data"`.
fn stream_type_kind(st: u8) -> &'static str {
    match st {
        0x85 | 0x84 | 0x83 | 0x1B | 0x24 => "video",
        0x81 | 0x82 | 0x80 | 0x03 | 0x04 | 0x0F | 0x11 => "audio",
        _ => "data",
    }
}
/// Computes (mean, p50, p95, max) of a histogram of `u64` bit-count values.
///
/// All output values are in the same unit as the input (bits per window).
/// Returns `(0.0, 0.0, 0.0, 0.0)` if `histogram` is empty.
#[allow(clippy::cast_precision_loss)]
fn bitrate_percentiles(histogram: &[u64]) -> (f64, f64, f64, f64) {
    if histogram.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mean = histogram.iter().sum::<u64>() as f64 / histogram.len() as f64;
    let max = *histogram.iter().max().unwrap_or(&0) as f64;
    let mut sorted: Vec<f64> = histogram.iter().map(|&v| v as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);
    (mean, p50, p95, max)
}
/// Derives keyframe interval statistics from an optional list of keyframe timestamps.
///
/// Returns `(intervals, mean, p50, p95, max)` — all `None` when `kf_timestamps`
/// is `None` or has fewer than two entries.
#[allow(clippy::cast_precision_loss)]
fn compute_kf_interval_stats(
    kf_timestamps: Option<&[f64]>,
) -> (
    Option<Vec<f64>>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
) {
    let Some(ts) = kf_timestamps else {
        return (None, None, None, None, None);
    };
    if ts.len() < 2 {
        return (Some(Vec::new()), None, None, None, None);
    }
    let mut intervals: Vec<f64> = ts.windows(2).map(|w| w[1] - w[0]).collect();
    intervals.retain(|&v| v >= 0.0);
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if intervals.is_empty() {
        return (Some(Vec::new()), None, None, None, None);
    }
    let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
    let max = intervals.last().copied();
    let p50 = percentile(&intervals, 0.50);
    let p95 = percentile(&intervals, 0.95);
    (Some(intervals), Some(mean), Some(p50), Some(p95), max)
}
/// Computes the `p`-th percentile (0.0–1.0) of a **pre-sorted** slice using
/// linear interpolation between neighbours.
///
/// Returns 0.0 for empty slices; returns `sorted[0]` for single-element slices.
#[allow(clippy::cast_precision_loss)]
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}
