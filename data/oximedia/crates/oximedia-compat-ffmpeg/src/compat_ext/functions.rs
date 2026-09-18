//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::arg_parser::FfmpegArgs;
use crate::codec_map::CodecMap;
use crate::filter_lex::FilterNode;
use std::collections::HashMap;

use super::types::{ContainerMapper, PixelFormatMapper, StreamMap, TranslationHint};

/// Extension methods on [`FfmpegArgs`] for accessing multi-input, metadata,
/// filter_complex, map specifications, and seeking/duration parameters.
pub trait FfmpegArgsExt {
    /// Return all input specifications (all `-i` arguments).
    fn inputs_all(&self) -> &[crate::arg_parser::InputSpec];
    /// Return the first `-filter_complex` / `-lavfi` string found across all
    /// output specifications, or `None` if not present.
    fn complex_filter(&self) -> Option<&str>;
    /// Return parsed [`StreamMap`] entries gathered from all output
    /// specifications, in order.
    fn stream_maps(&self) -> Vec<StreamMap>;
    /// Collect all `-metadata key=value` pairs from all output specifications
    /// into a single `HashMap`.  Later outputs overwrite earlier ones on key
    /// collision.
    fn all_metadata(&self) -> HashMap<String, String>;
    /// Return the first seek-start value (`-ss` in *post-input* position),
    /// parsed as seconds.  Returns `None` if not set or unparseable.
    fn seek_start(&self) -> Option<f64>;
    /// Return the first maximum duration value (`-t`), parsed as seconds.
    fn duration(&self) -> Option<f64>;
    /// Return the first end-time value (`-to`), parsed as seconds.
    ///
    /// FFmpeg's `-to` sets the *end* position; OxiMedia treats it separately
    /// from duration.
    fn to_time(&self) -> Option<f64>;
}
impl FfmpegArgsExt for FfmpegArgs {
    fn inputs_all(&self) -> &[crate::arg_parser::InputSpec] {
        &self.inputs
    }
    fn complex_filter(&self) -> Option<&str> {
        for out in &self.outputs {
            if let Some(ref fc) = out.filter_complex {
                return Some(fc.as_str());
            }
        }
        None
    }
    fn stream_maps(&self) -> Vec<StreamMap> {
        let mut result = Vec::new();
        for out in &self.outputs {
            for map_spec in &out.map {
                let parsed = StreamMap::from_map_spec(map_spec);
                result.push(parsed);
            }
        }
        result
    }
    fn all_metadata(&self) -> HashMap<String, String> {
        let mut merged = HashMap::new();
        for out in &self.outputs {
            for (k, v) in &out.metadata {
                merged.insert(k.clone(), v.clone());
            }
        }
        merged
    }
    fn seek_start(&self) -> Option<f64> {
        for out in &self.outputs {
            if let Some(ref s) = out.seek {
                if let Some(secs) = parse_time_str(s) {
                    return Some(secs);
                }
            }
        }
        None
    }
    fn duration(&self) -> Option<f64> {
        for out in &self.outputs {
            if let Some(ref s) = out.duration {
                if let Some(secs) = parse_time_str(s) {
                    return Some(secs);
                }
            }
        }
        None
    }
    fn to_time(&self) -> Option<f64> {
        for out in &self.outputs {
            for (k, v) in &out.extra_args {
                if k == "-to" {
                    if let Some(secs) = parse_time_str(v) {
                        return Some(secs);
                    }
                }
            }
        }
        None
    }
}
/// Static table: (extension/ffmpeg_name, oximedia_container)
pub(crate) static CONTAINER_TABLE: &[(&str, &str)] = &[
    ("mp4", "mp4"),
    ("m4v", "mp4"),
    ("m4a", "mp4"),
    ("3gp", "mp4"),
    ("3g2", "mp4"),
    ("f4v", "mp4"),
    ("mkv", "matroska"),
    ("matroska", "matroska"),
    ("mka", "matroska"),
    ("mks", "matroska"),
    ("webm", "webm"),
    ("mov", "quicktime"),
    ("qt", "quicktime"),
    ("avi", "avi"),
    ("flv", "flv"),
    ("ts", "mpegts"),
    ("mts", "mpegts"),
    ("m2ts", "mpegts"),
    ("m2t", "mpegts"),
    ("mpegts", "mpegts"),
    ("ogg", "ogg"),
    ("ogv", "ogg"),
    ("oga", "ogg"),
    ("opus", "ogg"),
    ("wav", "wav"),
    ("wave", "wav"),
    ("flac", "flac"),
    ("aiff", "aiff"),
    ("aif", "aiff"),
    ("mp3", "mp3"),
    ("aac", "adts"),
    ("mxf", "mxf"),
    ("gxf", "gxf"),
    ("rm", "rm"),
    ("rmvb", "rm"),
    ("asf", "asf"),
    ("wmv", "asf"),
    ("wma", "asf"),
    ("nut", "nut"),
];
/// Static table: (ffmpeg_pix_fmt, oximedia_pix_fmt)
pub(crate) static PIX_FMT_TABLE: &[(&str, &str)] = &[
    ("yuv420p", "yuv420p"),
    ("yuvj420p", "yuv420p"),
    ("yuv420p10le", "yuv420p10le"),
    ("yuv420p10be", "yuv420p10le"),
    ("yuv420p12le", "yuv420p12le"),
    ("yuv422p", "yuv422p"),
    ("yuvj422p", "yuv422p"),
    ("yuv422p10le", "yuv422p10le"),
    ("yuv422p10be", "yuv422p10le"),
    ("yuv444p", "yuv444p"),
    ("yuvj444p", "yuv444p"),
    ("yuv444p10le", "yuv444p10le"),
    ("yuv444p10be", "yuv444p10le"),
    ("nv12", "nv12"),
    ("nv21", "nv12"),
    ("nv16", "yuv422p"),
    ("nv24", "yuv444p"),
    ("p010le", "p010le"),
    ("p010be", "p010le"),
    ("p016le", "p016le"),
    ("rgb24", "rgb24"),
    ("bgr24", "rgb24"),
    ("rgba", "rgba"),
    ("bgra", "rgba"),
    ("rgb0", "rgb24"),
    ("bgr0", "rgb24"),
    ("argb", "rgba"),
    ("abgr", "rgba"),
    ("gray", "gray8"),
    ("gray8", "gray8"),
    ("gray10le", "gray10le"),
    ("gray12le", "gray12le"),
    ("gray16le", "gray16le"),
    ("rgb48le", "rgb48le"),
    ("rgb48be", "rgb48le"),
    ("rgba64le", "rgba64le"),
    ("rgba64be", "rgba64le"),
    ("gbrp", "gbrp"),
    ("gbrp10le", "gbrp10le"),
];
/// Parses a comma-separated filter chain string into an ordered list of
/// [`FilterNode`] values.
///
/// Commas inside `()` (parameter lists) are not treated as separators.
/// This allows filter arguments like `scale=iw/2:ih/2` to parse correctly.
///
/// ## Example
///
/// ```
/// use oximedia_compat_ffmpeg::compat_ext::parse_filter_chain;
///
/// let nodes = parse_filter_chain("scale=1280:720,setsar=1");
/// assert_eq!(nodes.len(), 2);
/// assert_eq!(nodes[0].name, "scale");
/// assert_eq!(nodes[1].name, "setsar");
/// ```
pub fn parse_filter_chain(chain_str: &str) -> Vec<FilterNode> {
    split_on_comma_chain(chain_str)
        .into_iter()
        .map(|segment| {
            let segment = segment.trim();
            parse_filter_node_raw(segment)
        })
        .collect()
}
/// Split a filter chain string on `,` while ignoring commas inside `()` or `[]`.
fn split_on_comma_chain(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut paren_depth: usize = 0;
    let mut bracket_depth: usize = 0;
    let mut start = 0usize;
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => paren_depth += 1,
            b')' if paren_depth > 0 => paren_depth -= 1,
            b'[' => bracket_depth += 1,
            b']' if bracket_depth > 0 => bracket_depth -= 1,
            b',' if paren_depth == 0 && bracket_depth == 0 => {
                result.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    result.push(&s[start..]);
    result
}
/// Parse a single raw filter expression like `scale=1280:720` or `hflip` into
/// a [`FilterNode`].
fn parse_filter_node_raw(s: &str) -> FilterNode {
    let s = s.trim();
    let s = strip_pad_labels(s);
    let (name, args_str) = match s.find('=') {
        Some(pos) => (s[..pos].trim(), s[pos + 1..].trim()),
        None => (s.trim(), ""),
    };
    let (positional_args, named_args) = parse_filter_args(args_str);
    FilterNode {
        inputs: Vec::new(),
        name: name.to_string(),
        positional_args,
        named_args,
        outputs: Vec::new(),
    }
}
/// Strip any leading `[label]` and trailing `[label]` pad markers from a
/// filter expression fragment.
fn strip_pad_labels(s: &str) -> &str {
    let s = s.trim();
    let s = if s.starts_with('[') {
        if let Some(end) = s.find(']') {
            s[end + 1..].trim_start()
        } else {
            s
        }
    } else {
        s
    };
    let s = if s.ends_with(']') {
        if let Some(start) = s.rfind('[') {
            s[..start].trim_end()
        } else {
            s
        }
    } else {
        s
    };
    s
}
/// Parse colon-separated filter arguments into positional and named lists.
fn parse_filter_args(args_str: &str) -> (Vec<String>, Vec<(String, String)>) {
    if args_str.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut positional = Vec::new();
    let mut named = Vec::new();
    for token in args_str.split(':') {
        let token = token.trim();
        if let Some(eq_pos) = token.find('=') {
            let key = token[..eq_pos].trim().to_string();
            let value = token[eq_pos + 1..].trim().to_string();
            named.push((key, value));
        } else if !token.is_empty() {
            positional.push(token.to_string());
        }
    }
    (positional, named)
}
/// Known lavfi source filter names.
pub(crate) static LAVFI_SOURCES: &[&str] = &[
    "color",
    "colour",
    "testsrc",
    "testsrc2",
    "smptebars",
    "smptehdbars",
    "sine",
    "anullsrc",
    "nullsrc",
    "rgbtestsrc",
    "mandelbrot",
    "life",
    "mptestsrc",
    "gradients",
    "haldclutsrc",
    "pal75bars",
    "pal100bars",
    "allrgb",
    "allyuv",
    "yuvtestsrc",
];
/// Format an f32 FPS value with minimal decimal places.
pub(crate) fn format_fps_f32(fps: f32) -> String {
    let s = format!("{:.3}", fps);
    let s = s.trim_end_matches('0');
    let s = s.trim_end_matches('.');
    s.to_string()
}
/// Generate a list of [`TranslationHint`] values for all translatable elements
/// in a parsed [`FfmpegArgs`] command.
///
/// The hints cover:
/// - Codec mappings (direct and patent-substituted)
/// - Container format mappings
/// - Pixel format mappings
pub fn generate_hints(args: &FfmpegArgs) -> Vec<TranslationHint> {
    use crate::codec_map::CodecCategory;
    let codec_map = CodecMap::new();
    let mut hints = Vec::new();
    for out in &args.outputs {
        for opt in &out.stream_options {
            if let Some(ref codec) = opt.codec {
                if codec == "copy" {
                    hints.push(TranslationHint::direct(codec.clone(), "copy"));
                    continue;
                }
                if let Some(entry) = codec_map.lookup(codec) {
                    match entry.category {
                        CodecCategory::DirectMatch => {
                            hints.push(TranslationHint::direct(codec.clone(), entry.oxi_name));
                        }
                        CodecCategory::PatentSubstituted => {
                            hints.push(TranslationHint::substituted(
                                codec.clone(),
                                entry.oxi_name,
                                format!(
                                    "'{}' is patent-encumbered; using '{}' (patent-free)",
                                    codec, entry.oxi_name
                                ),
                            ));
                        }
                        CodecCategory::Copy => {
                            hints.push(TranslationHint::direct(codec.clone(), "copy"));
                        }
                    }
                } else {
                    hints.push(TranslationHint {
                        original: codec.clone(),
                        translated: codec.clone(),
                        confidence: 0.1,
                        note: Some("Unknown codec; passing through as-is".to_string()),
                    });
                }
            }
        }
        if let Some(ref fmt) = out.format {
            if let Some(oxi) = ContainerMapper::ffmpeg_to_oximedia(fmt) {
                let confidence = if oxi == fmt { 1.0 } else { 0.8 };
                hints.push(TranslationHint {
                    original: fmt.clone(),
                    translated: oxi.to_string(),
                    confidence,
                    note: if oxi != fmt {
                        Some(format!("Container '{}' mapped to '{}'", fmt, oxi))
                    } else {
                        None
                    },
                });
            }
        }
        for opt in &out.stream_options {
            if let Some(ref pix) = opt.pixel_fmt {
                if let Some(oxi) = PixelFormatMapper::ffmpeg_to_oximedia(pix) {
                    let confidence = if oxi == pix.as_str() { 1.0 } else { 0.9 };
                    hints.push(TranslationHint {
                        original: pix.clone(),
                        translated: oxi.to_string(),
                        confidence,
                        note: if oxi != pix.as_str() {
                            Some(format!("Pixel format '{}' normalised to '{}'", pix, oxi))
                        } else {
                            None
                        },
                    });
                }
            }
        }
    }
    hints
}
/// Parse a time string into seconds.
///
/// Accepts formats:
/// - `"123.45"` — plain seconds (float)
/// - `"HH:MM:SS"` — hours:minutes:seconds
/// - `"HH:MM:SS.mmm"` — with fractional seconds
/// - `"MM:SS"` — minutes:seconds
pub(crate) fn parse_time_str(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    match parts.as_slice() {
        [mm, ss] => {
            let minutes = mm.parse::<f64>().ok()?;
            let seconds = ss.parse::<f64>().ok()?;
            Some(minutes * 60.0 + seconds)
        }
        [hh, mm, ss] => {
            let hours = hh.parse::<f64>().ok()?;
            let minutes = mm.parse::<f64>().ok()?;
            let seconds = ss.parse::<f64>().ok()?;
            Some(hours * 3600.0 + minutes * 60.0 + seconds)
        }
        _ => None,
    }
}
