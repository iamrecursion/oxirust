//! WebAssembly bindings for streaming manifest utilities from `oximedia-stream`.
//!
//! Provides lightweight HLS/DASH manifest parsing and info extraction for
//! browser-side adaptive streaming clients.  All operations are synchronous and
//! entirely in-memory.

use wasm_bindgen::prelude::*;

use oximedia_stream::manifest_builder::StreamVariant;

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    crate::utils::js_err(&format!("{msg}"))
}

// ---------------------------------------------------------------------------
// ManifestInfo
// ---------------------------------------------------------------------------

/// Summary information extracted from an HLS or DASH manifest.
///
/// # Example
///
/// ```javascript
/// const info = parse_manifest_info(manifestBytes);
/// console.log(`${info.segment_count} segments, ${info.duration_secs}s`);
/// ```
#[wasm_bindgen]
pub struct ManifestInfo {
    /// Total duration of the presentation in seconds.
    pub duration_secs: f32,
    /// Number of media segments in the playlist.
    pub segment_count: u32,
    /// Approximate target bitrate in bits per second (from the first variant).
    pub bitrate_bps: u32,
}

/// Parse a raw UTF-8 manifest byte slice and return summary information.
///
/// Supports basic HLS (`#EXTM3U`) and DASH (contains `<MPD`) manifests.
/// For DASH, `bitrate_bps` is extracted from the first `bandwidth=` attribute.
/// Duration is estimated from the sum of `#EXTINF` durations (HLS) or the
/// `mediaPresentationDuration` attribute (DASH).
///
/// # Errors
///
/// Returns an error if the bytes are not valid UTF-8 or the format is unrecognised.
#[wasm_bindgen]
pub fn parse_manifest_info(manifest_bytes: &[u8]) -> Result<ManifestInfo, JsValue> {
    let text =
        std::str::from_utf8(manifest_bytes).map_err(|e| js_err(format!("UTF-8 error: {e}")))?;

    if text.contains("#EXTM3U") {
        parse_hls(text)
    } else if text.contains("<MPD") {
        parse_dash(text)
    } else {
        Err(js_err(
            "unrecognised manifest format (expected HLS or DASH)",
        ))
    }
}

fn parse_hls(text: &str) -> Result<ManifestInfo, JsValue> {
    let mut duration_secs = 0.0_f32;
    let mut segment_count = 0u32;
    let mut bitrate_bps = 0u32;

    for line in text.lines() {
        let line = line.trim();
        // #EXTINF:<duration>[,<title>]
        if let Some(rest) = line.strip_prefix("#EXTINF:") {
            let dur_str = rest.split(',').next().unwrap_or("0");
            if let Ok(d) = dur_str.parse::<f32>() {
                duration_secs += d;
                segment_count += 1;
            }
        }
        // #EXT-X-STREAM-INF:BANDWIDTH=<bps>,...
        if let Some(rest) = line.strip_prefix("#EXT-X-STREAM-INF:") {
            if bitrate_bps == 0 {
                for attr in rest.split(',') {
                    if let Some(val) = attr.trim().strip_prefix("BANDWIDTH=") {
                        if let Ok(bps) = val.trim().parse::<u32>() {
                            bitrate_bps = bps;
                        }
                    }
                }
            }
        }
    }

    Ok(ManifestInfo {
        duration_secs,
        segment_count,
        bitrate_bps,
    })
}

fn parse_dash(text: &str) -> Result<ManifestInfo, JsValue> {
    // Parse mediaPresentationDuration="PT<hours>H<minutes>M<seconds>S"
    let duration_secs = extract_dash_duration(text);

    // Count <SegmentURL> or <S> elements as segments.
    let segment_count = text.matches("<SegmentURL").count() as u32
        + if text.matches("<SegmentURL").count() == 0 {
            text.matches("<S ").count() as u32
        } else {
            0
        };

    // Extract first bandwidth= attribute value.
    let bitrate_bps = extract_dash_bandwidth(text);

    Ok(ManifestInfo {
        duration_secs,
        segment_count,
        bitrate_bps,
    })
}

/// Extract `mediaPresentationDuration` from a DASH MPD string.
///
/// Parses the ISO 8601 duration format `PT<H>H<M>M<S>S`.
fn extract_dash_duration(text: &str) -> f32 {
    let tag = "mediaPresentationDuration=\"";
    let Some(start) = text.find(tag) else {
        return 0.0;
    };
    let rest = &text[start + tag.len()..];
    let Some(end) = rest.find('"') else {
        return 0.0;
    };
    let iso = &rest[..end]; // e.g. "PT1H30M45.5S"
    parse_iso8601_duration(iso)
}

/// Parse an ISO 8601 duration like `PT1H30M45.5S` into seconds.
fn parse_iso8601_duration(s: &str) -> f32 {
    let s = s.strip_prefix("PT").unwrap_or(s);
    let mut total = 0.0_f32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '0'..='9' | '.' => cur.push(ch),
            'H' => {
                if let Ok(v) = cur.parse::<f32>() {
                    total += v * 3600.0;
                }
                cur.clear();
            }
            'M' => {
                if let Ok(v) = cur.parse::<f32>() {
                    total += v * 60.0;
                }
                cur.clear();
            }
            'S' => {
                if let Ok(v) = cur.parse::<f32>() {
                    total += v;
                }
                cur.clear();
            }
            _ => cur.clear(),
        }
    }
    total
}

/// Extract the first `bandwidth="<bps>"` value from a DASH MPD.
fn extract_dash_bandwidth(text: &str) -> u32 {
    let tag = "bandwidth=\"";
    let Some(start) = text.find(tag) else {
        return 0;
    };
    let rest = &text[start + tag.len()..];
    let Some(end) = rest.find('"') else {
        return 0;
    };
    rest[..end].parse::<u32>().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// HLS manifest builder helpers (re-export from manifest_builder)
// ---------------------------------------------------------------------------

/// Build a master HLS playlist string from variant descriptions encoded as JSON.
///
/// `variants_json` must be an array of objects, each with:
/// - `bandwidth` (integer, bps)
/// - `codecs` (string, e.g. `"av01.0.04M.08"`)
/// - `uri` (string)
/// - optional `resolution` ([width, height])
/// - optional `frame_rate` (number)
///
/// # Errors
///
/// Returns an error if `variants_json` is not a valid JSON array.
#[wasm_bindgen]
pub fn wasm_build_master_playlist(variants_json: &str) -> Result<String, JsValue> {
    let parsed: serde_json::Value = serde_json::from_str(variants_json)
        .map_err(|e| js_err(format!("JSON parse error: {e}")))?;

    let arr = parsed
        .as_array()
        .ok_or_else(|| js_err("expected JSON array of variant objects"))?;

    let mut variants = Vec::with_capacity(arr.len());
    for item in arr {
        let bandwidth =
            item.get("bandwidth")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| js_err("variant missing 'bandwidth' (integer)"))? as u64;
        let codecs = item
            .get("codecs")
            .and_then(|v| v.as_str())
            .ok_or_else(|| js_err("variant missing 'codecs' (string)"))?
            .to_string();
        let uri = item
            .get("uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| js_err("variant missing 'uri' (string)"))?
            .to_string();
        let resolution = item.get("resolution").and_then(|v| {
            let arr = v.as_array()?;
            let w = arr.first()?.as_u64()? as u32;
            let h = arr.get(1)?.as_u64()? as u32;
            Some((w, h))
        });
        let frame_rate = item
            .get("frame_rate")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32);

        variants.push(StreamVariant {
            bandwidth,
            resolution,
            codecs,
            uri,
            frame_rate,
        });
    }

    Ok(oximedia_stream::manifest_builder::build_master_playlist(
        &variants,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HLS_SAMPLE: &str = r#"#EXTM3U
#EXT-X-VERSION:3
#EXT-X-STREAM-INF:BANDWIDTH=500000,CODECS="av01.0.00M.08"
low.m3u8
#EXTINF:6.0,
seg0.ts
#EXTINF:6.0,
seg1.ts
#EXTINF:4.0,
seg2.ts
#EXT-X-ENDLIST"#;

    #[test]
    fn parse_hls_manifest() {
        let info = parse_manifest_info(HLS_SAMPLE.as_bytes()).expect("parse ok");
        assert_eq!(info.segment_count, 3, "should count 3 segments");
        assert!(
            (info.duration_secs - 16.0).abs() < 0.01,
            "total duration should be 16s, got {}",
            info.duration_secs
        );
        assert_eq!(info.bitrate_bps, 500_000);
    }

    #[test]
    fn parse_invalid_utf8_errors() {
        let bad: &[u8] = &[0xFF, 0xFE, 0x00];
        assert!(parse_manifest_info(bad).is_err());
    }

    #[test]
    fn parse_unknown_format_errors() {
        let unknown = b"not a manifest";
        assert!(parse_manifest_info(unknown).is_err());
    }

    #[test]
    fn iso8601_duration_parsing() {
        assert!((parse_iso8601_duration("PT1H30M45S") - 5445.0).abs() < 0.01);
        assert!((parse_iso8601_duration("PT6.5S") - 6.5).abs() < 0.01);
        assert!((parse_iso8601_duration("PT0S")).abs() < 0.01);
    }

    #[test]
    fn build_master_playlist_valid() {
        let json = r#"[{"bandwidth":1000000,"codecs":"av01.0.04M.08","uri":"high.m3u8"}]"#;
        let pl = wasm_build_master_playlist(json).expect("should build");
        assert!(pl.contains("#EXTM3U"), "should contain HLS header");
        assert!(pl.contains("BANDWIDTH=1000000"), "should contain bandwidth");
    }
}
