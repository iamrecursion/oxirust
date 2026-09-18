//! Advanced FFmpeg filtergraph to OxiMedia pipeline translation.
//!
//! This module provides a richer, semantically-typed filtergraph AST than the
//! low-level lexer in [`crate::filter_lex`].  Where `filter_lex` preserves the
//! raw text for diagnostic purposes, this module commits to strongly-typed
//! variants so that downstream pipeline builders can act on them without
//! further string manipulation.
//!
//! ## Filtergraph syntax primer
//!
//! ```text
//! [in]scale=1920:1080[s];[s]fps=30000/1001,hflip[out]
//!  ^^^ input label    ^^^ output label
//!                        ^^^^^^^^^^^^^^^^^^^^ comma-separated filter chain
//!                                            ^^^^ chained filter
//! ```
//!
//! A *filtergraph* contains one or more *filter chains* separated by `;`.
//! Each chain may carry optional leading `[label]` and trailing `[label]`
//! tokens.  Filters within a chain are separated by `,`.

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A strongly-typed filter node in a filtergraph chain.
///
/// Each variant corresponds to a named FFmpeg filter with fully parsed
/// arguments.  Unknown or unsupported filters fall through to the
/// [`FilterGraphNode::Unknown`] variant so that round-trip diagnostics remain
/// possible without discarding information.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterGraphNode {
    /// `scale=W:H` or `scale=w=W:h=H` — resize video.
    ///
    /// `flags` carries the optional `flags` argument (e.g. `"lanczos"`).
    Scale {
        width: i32,
        height: i32,
        flags: String,
    },
    /// `crop=w:h:x:y` — crop a rectangular region.
    Crop { w: i32, h: i32, x: i32, y: i32 },
    /// `overlay=x:y` — overlay one video on top of another.
    ///
    /// `x` and `y` are kept as strings because FFmpeg allows expressions such
    /// as `"(W-w)/2"`.
    Overlay { x: String, y: String },
    /// `pad=w:h:x:y[:color]` — pad video with a solid-colour border.
    Pad {
        w: i32,
        h: i32,
        x: i32,
        y: i32,
        color: String,
    },
    /// `transpose=dir` — transpose (rotate + flip) video.
    ///
    /// | dir | effect              |
    /// |-----|---------------------|
    /// | 0   | CCW + vertical flip |
    /// | 1   | CW (90°)            |
    /// | 2   | CCW (270°)          |
    /// | 3   | CW + vertical flip  |
    Transpose { dir: u8 },
    /// `vflip` — vertical flip.
    Vflip,
    /// `hflip` — horizontal flip.
    Hflip,
    /// `rotate=angle` — rotate by an angle in radians.
    ///
    /// `fillcolor` is the background colour for exposed corners.
    Rotate { angle: f64, fillcolor: String },
    /// `trim=start=S:end=E:duration=D` — temporal trim of a video stream.
    Trim {
        start: Option<f64>,
        end: Option<f64>,
        duration: Option<f64>,
    },
    /// `fps=N` or `fps=N/D` — force a target frame rate.
    ///
    /// The rate is stored as a string to preserve expressions like
    /// `"30000/1001"` (NTSC drop-frame).
    Fps { fps: String },
    /// `setpts=expr` — modify presentation timestamps.
    ///
    /// Common expressions: `"PTS-STARTPTS"`, `"2.0*PTS"`.
    SetPts(String),
    /// `format=pix_fmt` — force a pixel format conversion.
    Format { pix_fmt: String },
    // ── Audio ─────────────────────────────────────────────────────────────
    /// `volume=V` — adjust audio volume.
    ///
    /// `volume` may be a linear factor (`"0.5"`) or a dB expression
    /// (`"-10dB"`).
    Volume { volume: String },
    /// `loudnorm=I=i:LRA=l:TP=t` — EBU R128 loudness normalisation.
    Loudnorm { i: f32, lra: f32, tp: f32 },
    /// `atrim=start=S:end=E` — temporal trim of an audio stream.
    Atrim {
        start: Option<f64>,
        end: Option<f64>,
    },
    /// `asetpts=expr` — modify audio presentation timestamps.
    Asetpts(String),
    /// `pan=layout|mapping` — manual channel routing.
    Pan(String),
    // ── Generic passthrough ───────────────────────────────────────────────
    /// A filter whose name is known but which has no dedicated variant.
    Unknown { name: String, args: String },
}

/// A single linear filter chain within a filtergraph.
///
/// Each chain may be annotated with named input/output pads (labels) that
/// connect it to other chains.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FilterChain {
    /// Optional leading pad label, e.g. `"in"` from `[in]scale=…`.
    pub input_label: Option<String>,
    /// Optional trailing pad label, e.g. `"out"` from `…scale=…[out]`.
    pub output_label: Option<String>,
    /// Ordered sequence of filters in this chain.
    pub nodes: Vec<FilterGraphNode>,
}

/// A complete filtergraph composed of one or more [`FilterChain`]s.
///
/// Chains are separated by `;` in FFmpeg's syntax.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FilterGraph {
    /// All chains, in parse order.
    pub chains: Vec<FilterChain>,
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterGraph implementation
// ─────────────────────────────────────────────────────────────────────────────

impl FilterGraph {
    /// Parse an FFmpeg filtergraph string into a [`FilterGraph`].
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` with a human-readable message if the string
    /// contains a structural syntax error (e.g. an empty filter name).
    /// Individual unknown filter names never cause an error — they become
    /// [`FilterGraphNode::Unknown`] variants instead.
    pub fn parse(filtergraph: &str) -> Result<Self, String> {
        let mut graph = FilterGraph::default();

        for raw_chain in filtergraph.split(';') {
            let raw_chain = raw_chain.trim();
            if raw_chain.is_empty() {
                continue;
            }
            let chain = parse_chain(raw_chain)?;
            graph.chains.push(chain);
        }

        Ok(graph)
    }

    /// Describe the filtergraph as an ordered list of OxiMedia pipeline step
    /// descriptions suitable for logging or display.
    ///
    /// Each line has the form `"[chain N] StepDescription"`.
    pub fn to_pipeline_description(&self) -> String {
        let mut lines: Vec<String> = Vec::new();

        for (ci, chain) in self.chains.iter().enumerate() {
            let prefix = match (&chain.input_label, &chain.output_label) {
                (Some(i), Some(o)) => format!("[{} → {}]", i, o),
                (Some(i), None) => format!("[{}→]", i),
                (None, Some(o)) => format!("[→{}]", o),
                (None, None) => format!("[chain {}]", ci),
            };
            for node in &chain.nodes {
                lines.push(format!("{} {}", prefix, node_description(node)));
            }
        }

        lines.join("\n")
    }

    /// Return `true` if any chain contains at least one audio filter.
    pub fn has_audio_filters(&self) -> bool {
        self.chains
            .iter()
            .any(|c| c.nodes.iter().any(is_audio_node))
    }

    /// Return `true` if any chain contains at least one video filter.
    pub fn has_video_filters(&self) -> bool {
        self.chains
            .iter()
            .any(|c| c.nodes.iter().any(is_video_node))
    }

    /// Return the target resolution from the first [`FilterGraphNode::Scale`]
    /// node found anywhere in the graph, or `None` if no scale filter exists.
    pub fn scale_target(&self) -> Option<(i32, i32)> {
        for chain in &self.chains {
            for node in &chain.nodes {
                if let FilterGraphNode::Scale { width, height, .. } = node {
                    return Some((*width, *height));
                }
            }
        }
        None
    }

    /// Collect all [`FilterGraphNode::Unknown`] nodes across all chains.
    ///
    /// Useful for emitting diagnostics about unsupported filters.
    pub fn unknown_nodes(&self) -> Vec<(&str, &str)> {
        let mut result = Vec::new();
        for chain in &self.chains {
            for node in &chain.nodes {
                if let FilterGraphNode::Unknown { name, args } = node {
                    result.push((name.as_str(), args.as_str()));
                }
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterGraphNode implementation
// ─────────────────────────────────────────────────────────────────────────────

impl FilterGraphNode {
    /// Parse a single `name[=args]` expression into a [`FilterGraphNode`].
    ///
    /// The expression must not contain leading/trailing pad labels (`[…]`) —
    /// those are stripped by the chain parser before calling this function.
    pub fn parse(s: &str) -> FilterGraphNode {
        let s = s.trim();

        // Split at first `=` to separate name from argument list.
        let (name, args_str) = match s.find('=') {
            Some(pos) => (&s[..pos], &s[pos + 1..]),
            None => (s, ""),
        };

        let name_lower = name.trim().to_lowercase();
        let args_str = args_str.trim();

        match name_lower.as_str() {
            "scale" => parse_scale_node(args_str),
            "crop" => parse_crop_node(args_str),
            "overlay" => parse_overlay_node(args_str),
            "pad" => parse_pad_node(args_str),
            "transpose" => parse_transpose_node(args_str),
            "vflip" => FilterGraphNode::Vflip,
            "hflip" => FilterGraphNode::Hflip,
            "rotate" => parse_rotate_node(args_str),
            "trim" => parse_trim_node(args_str),
            "fps" => FilterGraphNode::Fps {
                fps: args_str.to_string(),
            },
            "setpts" => FilterGraphNode::SetPts(args_str.to_string()),
            "format" => FilterGraphNode::Format {
                pix_fmt: args_str.to_string(),
            },
            "volume" => FilterGraphNode::Volume {
                volume: args_str.to_string(),
            },
            "loudnorm" => parse_loudnorm_node(args_str),
            "atrim" => parse_atrim_node(args_str),
            "asetpts" => FilterGraphNode::Asetpts(args_str.to_string()),
            "pan" => FilterGraphNode::Pan(args_str.to_string()),
            _ => FilterGraphNode::Unknown {
                name: name.trim().to_string(),
                args: args_str.to_string(),
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chain parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a single chain string (no `;` separators) into a [`FilterChain`].
fn parse_chain(raw: &str) -> Result<FilterChain, String> {
    // Extract leading input label.
    let (input_label, rest) = extract_leading_label(raw);
    // Extract trailing output label.
    let (body, output_label) = extract_trailing_label(rest);

    let mut chain = FilterChain {
        input_label,
        output_label,
        nodes: Vec::new(),
    };

    for segment in split_on_comma(body) {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        // Each segment may itself have pad labels (inner labels between filters
        // in a complex graph).  Strip them before parsing.
        let (_, after_label) = extract_leading_label(segment);
        let (filter_body, _) = extract_trailing_label(after_label);

        if filter_body.trim().is_empty() {
            return Err(format!("empty filter expression in chain: {:?}", raw));
        }

        let node = FilterGraphNode::parse(filter_body);
        chain.nodes.push(node);
    }

    Ok(chain)
}

/// Extract a single leading `[label]` from the start of `s`.
///
/// Returns `(Some(label), remainder)` when a label is found,
/// `(None, s)` otherwise.
fn extract_leading_label(s: &str) -> (Option<String>, &str) {
    let s = s.trim_start();
    if !s.starts_with('[') {
        return (None, s);
    }
    if let Some(end) = s.find(']') {
        let label = s[1..end].trim().to_string();
        let rest = s[end + 1..].trim_start();
        // Only treat this as a pad label if it appears before the filter name
        // (i.e. the character after `]` is not `[` or is the start of a name).
        (Some(label), rest)
    } else {
        (None, s)
    }
}

/// Extract a single trailing `[label]` from the end of `s`.
///
/// Returns `(body, Some(label))` when a label is found,
/// `(s, None)` otherwise.
fn extract_trailing_label(s: &str) -> (&str, Option<String>) {
    let s = s.trim_end();
    if !s.ends_with(']') {
        return (s, None);
    }
    if let Some(open) = s.rfind('[') {
        let label = s[open + 1..s.len() - 1].trim().to_string();
        let body = s[..open].trim_end();
        (body, Some(label))
    } else {
        (s, None)
    }
}

/// Split `s` on `,` while respecting bracket and quote nesting.
fn split_on_comma(s: &str) -> Vec<&str> {
    let mut result: Vec<&str> = Vec::new();
    let mut depth: usize = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut start: usize = 0;
    let bytes = s.as_bytes();

    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b'[' if !in_single && !in_double => depth += 1,
            b']' if !in_single && !in_double && depth > 0 => depth -= 1,
            b',' if !in_single && !in_double && depth == 0 => {
                result.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }

    result.push(&s[start..]);
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-filter argument parsers
// ─────────────────────────────────────────────────────────────────────────────

/// Parse colon-separated args into positional and named parts.
///
/// `key=value` tokens are named; bare tokens are positional (in order).
fn split_args(args_str: &str) -> (Vec<&str>, Vec<(&str, &str)>) {
    if args_str.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut positional: Vec<&str> = Vec::new();
    let mut named: Vec<(&str, &str)> = Vec::new();

    for part in args_str.split(':') {
        let part = part.trim();
        if let Some(eq) = part.find('=') {
            named.push((&part[..eq], &part[eq + 1..]));
        } else if !part.is_empty() {
            positional.push(part);
        }
    }

    (positional, named)
}

fn named_val<'a>(named: &[(&'a str, &'a str)], key: &str) -> Option<&'a str> {
    named
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| *v)
}

fn pos_or_named<'a>(
    positional: &[&'a str],
    named: &[(&'a str, &'a str)],
    key: &str,
    idx: usize,
) -> Option<&'a str> {
    named_val(named, key).or_else(|| positional.get(idx).copied())
}

fn parse_i32_expr(s: &str) -> Option<i32> {
    // Only parse pure integers; keep expressions (iw/2 etc.) as -1.
    let s = s.trim();
    s.parse::<i32>().ok()
}

fn parse_f64_opt(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

fn parse_scale_node(args_str: &str) -> FilterGraphNode {
    let (pos, named) = split_args(args_str);
    let width = pos_or_named(&pos, &named, "w", 0)
        .and_then(parse_i32_expr)
        .unwrap_or(-1);
    let height = pos_or_named(&pos, &named, "h", 1)
        .and_then(parse_i32_expr)
        .unwrap_or(-1);
    let flags = named_val(&named, "flags").unwrap_or("").to_string();
    FilterGraphNode::Scale {
        width,
        height,
        flags,
    }
}

fn parse_crop_node(args_str: &str) -> FilterGraphNode {
    let (pos, named) = split_args(args_str);
    let w = pos_or_named(&pos, &named, "w", 0)
        .and_then(parse_i32_expr)
        .unwrap_or(0);
    let h = pos_or_named(&pos, &named, "h", 1)
        .and_then(parse_i32_expr)
        .unwrap_or(0);
    let x = pos_or_named(&pos, &named, "x", 2)
        .and_then(parse_i32_expr)
        .unwrap_or(0);
    let y = pos_or_named(&pos, &named, "y", 3)
        .and_then(parse_i32_expr)
        .unwrap_or(0);
    FilterGraphNode::Crop { w, h, x, y }
}

fn parse_overlay_node(args_str: &str) -> FilterGraphNode {
    let (pos, named) = split_args(args_str);
    let x = pos_or_named(&pos, &named, "x", 0)
        .unwrap_or("0")
        .to_string();
    let y = pos_or_named(&pos, &named, "y", 1)
        .unwrap_or("0")
        .to_string();
    FilterGraphNode::Overlay { x, y }
}

fn parse_pad_node(args_str: &str) -> FilterGraphNode {
    let (pos, named) = split_args(args_str);
    let w = pos_or_named(&pos, &named, "w", 0)
        .and_then(parse_i32_expr)
        .unwrap_or(0);
    let h = pos_or_named(&pos, &named, "h", 1)
        .and_then(parse_i32_expr)
        .unwrap_or(0);
    let x = pos_or_named(&pos, &named, "x", 2)
        .and_then(parse_i32_expr)
        .unwrap_or(0);
    let y = pos_or_named(&pos, &named, "y", 3)
        .and_then(parse_i32_expr)
        .unwrap_or(0);
    let color = pos_or_named(&pos, &named, "color", 4)
        .unwrap_or("black")
        .to_string();
    FilterGraphNode::Pad { w, h, x, y, color }
}

fn parse_transpose_node(args_str: &str) -> FilterGraphNode {
    let (pos, named) = split_args(args_str);
    let dir = pos_or_named(&pos, &named, "dir", 0)
        .and_then(|s| s.trim().parse::<u8>().ok())
        .unwrap_or(1);
    FilterGraphNode::Transpose { dir }
}

fn parse_rotate_node(args_str: &str) -> FilterGraphNode {
    let (pos, named) = split_args(args_str);
    let angle = pos_or_named(&pos, &named, "angle", 0)
        .and_then(parse_f64_opt)
        .unwrap_or(0.0);
    let fillcolor = named_val(&named, "fillcolor")
        .or_else(|| named_val(&named, "c"))
        .unwrap_or("black")
        .to_string();
    FilterGraphNode::Rotate { angle, fillcolor }
}

fn parse_trim_node(args_str: &str) -> FilterGraphNode {
    let (_, named) = split_args(args_str);
    let start = named_val(&named, "start").and_then(parse_f64_opt);
    let end = named_val(&named, "end").and_then(parse_f64_opt);
    let duration = named_val(&named, "duration").and_then(parse_f64_opt);
    FilterGraphNode::Trim {
        start,
        end,
        duration,
    }
}

fn parse_loudnorm_node(args_str: &str) -> FilterGraphNode {
    let (_, named) = split_args(args_str);
    let i = named_val(&named, "I")
        .or_else(|| named_val(&named, "i"))
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(-23.0);
    let lra = named_val(&named, "LRA")
        .or_else(|| named_val(&named, "lra"))
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(7.0);
    let tp = named_val(&named, "TP")
        .or_else(|| named_val(&named, "tp"))
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(-2.0);
    FilterGraphNode::Loudnorm { i, lra, tp }
}

fn parse_atrim_node(args_str: &str) -> FilterGraphNode {
    let (_, named) = split_args(args_str);
    let start = named_val(&named, "start").and_then(parse_f64_opt);
    let end = named_val(&named, "end").and_then(parse_f64_opt);
    FilterGraphNode::Atrim { start, end }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline description helpers
// ─────────────────────────────────────────────────────────────────────────────

fn node_description(node: &FilterGraphNode) -> String {
    match node {
        FilterGraphNode::Scale {
            width,
            height,
            flags,
        } => {
            if flags.is_empty() {
                format!("Scale {}x{}", width, height)
            } else {
                format!("Scale {}x{} (flags: {})", width, height, flags)
            }
        }
        FilterGraphNode::Crop { w, h, x, y } => {
            format!("Crop {}x{} at ({},{})", w, h, x, y)
        }
        FilterGraphNode::Overlay { x, y } => format!("Overlay at ({},{})", x, y),
        FilterGraphNode::Pad { w, h, x, y, color } => {
            format!("Pad to {}x{} offset ({},{}) color={}", w, h, x, y, color)
        }
        FilterGraphNode::Transpose { dir } => {
            let label = match dir {
                0 => "CCW + vflip",
                1 => "CW 90°",
                2 => "CCW 90°",
                3 => "CW + vflip",
                _ => "unknown",
            };
            format!("Transpose ({})", label)
        }
        FilterGraphNode::Vflip => "Vertical flip".to_string(),
        FilterGraphNode::Hflip => "Horizontal flip".to_string(),
        FilterGraphNode::Rotate { angle, fillcolor } => {
            format!("Rotate {:.4}rad fillcolor={}", angle, fillcolor)
        }
        FilterGraphNode::Trim {
            start,
            end,
            duration,
        } => {
            let mut parts: Vec<String> = Vec::new();
            if let Some(s) = start {
                parts.push(format!("start={}", s));
            }
            if let Some(e) = end {
                parts.push(format!("end={}", e));
            }
            if let Some(d) = duration {
                parts.push(format!("duration={}", d));
            }
            format!("Trim ({})", parts.join(", "))
        }
        FilterGraphNode::Fps { fps } => format!("Force FPS={}", fps),
        FilterGraphNode::SetPts(expr) => format!("SetPTS expr={}", expr),
        FilterGraphNode::Format { pix_fmt } => format!("PixelFormat {}", pix_fmt),
        FilterGraphNode::Volume { volume } => format!("Volume level={}", volume),
        FilterGraphNode::Loudnorm { i, lra, tp } => {
            format!("LoudNorm I={}:LRA={}:TP={}", i, lra, tp)
        }
        FilterGraphNode::Atrim { start, end } => {
            let s = start.map_or("".to_string(), |v| format!("start={}", v));
            let e = end.map_or("".to_string(), |v| format!("end={}", v));
            format!("AudioTrim ({} {})", s, e)
        }
        FilterGraphNode::Asetpts(expr) => format!("AudioSetPTS expr={}", expr),
        FilterGraphNode::Pan(mapping) => format!("Pan mapping={}", mapping),
        FilterGraphNode::Unknown { name, args } => {
            if args.is_empty() {
                format!("Unknown filter: {}", name)
            } else {
                format!("Unknown filter: {}={}", name, args)
            }
        }
    }
}

fn is_audio_node(node: &FilterGraphNode) -> bool {
    matches!(
        node,
        FilterGraphNode::Volume { .. }
            | FilterGraphNode::Loudnorm { .. }
            | FilterGraphNode::Atrim { .. }
            | FilterGraphNode::Asetpts(_)
            | FilterGraphNode::Pan(_)
    )
}

fn is_video_node(node: &FilterGraphNode) -> bool {
    matches!(
        node,
        FilterGraphNode::Scale { .. }
            | FilterGraphNode::Crop { .. }
            | FilterGraphNode::Overlay { .. }
            | FilterGraphNode::Pad { .. }
            | FilterGraphNode::Transpose { .. }
            | FilterGraphNode::Vflip
            | FilterGraphNode::Hflip
            | FilterGraphNode::Rotate { .. }
            | FilterGraphNode::Trim { .. }
            | FilterGraphNode::Fps { .. }
            | FilterGraphNode::SetPts(_)
            | FilterGraphNode::Format { .. }
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── simple single-filter chains ──────────────────────────────────────────

    #[test]
    fn test_scale_positional() {
        let g = FilterGraph::parse("scale=1920:1080").expect("parse");
        assert_eq!(g.chains.len(), 1);
        let node = &g.chains[0].nodes[0];
        assert!(
            matches!(
                node,
                FilterGraphNode::Scale {
                    width: 1920,
                    height: 1080,
                    ..
                }
            ),
            "unexpected node: {:?}",
            node
        );
    }

    #[test]
    fn test_scale_named_args() {
        let g = FilterGraph::parse("scale=w=1280:h=720").expect("parse");
        assert!(matches!(
            &g.chains[0].nodes[0],
            FilterGraphNode::Scale {
                width: 1280,
                height: 720,
                ..
            }
        ));
    }

    #[test]
    fn test_scale_with_flags() {
        let g = FilterGraph::parse("scale=1920:1080:flags=lanczos").expect("parse");
        match &g.chains[0].nodes[0] {
            FilterGraphNode::Scale {
                width,
                height,
                flags,
            } => {
                assert_eq!(*width, 1920);
                assert_eq!(*height, 1080);
                assert_eq!(flags, "lanczos");
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_crop_node() {
        let g = FilterGraph::parse("crop=640:480:10:20").expect("parse");
        assert!(matches!(
            &g.chains[0].nodes[0],
            FilterGraphNode::Crop {
                w: 640,
                h: 480,
                x: 10,
                y: 20
            }
        ));
    }

    #[test]
    fn test_hflip_vflip() {
        let g = FilterGraph::parse("hflip,vflip").expect("parse");
        assert_eq!(g.chains[0].nodes.len(), 2);
        assert!(matches!(g.chains[0].nodes[0], FilterGraphNode::Hflip));
        assert!(matches!(g.chains[0].nodes[1], FilterGraphNode::Vflip));
    }

    #[test]
    fn test_transpose() {
        let g = FilterGraph::parse("transpose=1").expect("parse");
        assert!(matches!(
            g.chains[0].nodes[0],
            FilterGraphNode::Transpose { dir: 1 }
        ));
    }

    #[test]
    fn test_fps_ntsc() {
        let g = FilterGraph::parse("fps=30000/1001").expect("parse");
        match &g.chains[0].nodes[0] {
            FilterGraphNode::Fps { fps } => assert_eq!(fps, "30000/1001"),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_setpts() {
        let g = FilterGraph::parse("setpts=PTS-STARTPTS").expect("parse");
        assert!(matches!(&g.chains[0].nodes[0], FilterGraphNode::SetPts(s) if s == "PTS-STARTPTS"));
    }

    #[test]
    fn test_format() {
        let g = FilterGraph::parse("format=yuv420p").expect("parse");
        assert!(
            matches!(&g.chains[0].nodes[0], FilterGraphNode::Format { pix_fmt } if pix_fmt == "yuv420p")
        );
    }

    // ── audio filters ────────────────────────────────────────────────────────

    #[test]
    fn test_volume_linear() {
        let g = FilterGraph::parse("volume=0.5").expect("parse");
        assert!(
            matches!(&g.chains[0].nodes[0], FilterGraphNode::Volume { volume } if volume == "0.5")
        );
    }

    #[test]
    fn test_volume_db_string() {
        // volume string is preserved verbatim in this AST.
        let g = FilterGraph::parse("volume=-10dB").expect("parse");
        assert!(
            matches!(&g.chains[0].nodes[0], FilterGraphNode::Volume { volume } if volume == "-10dB")
        );
    }

    #[test]
    fn test_loudnorm_defaults() {
        let g = FilterGraph::parse("loudnorm").expect("parse");
        match &g.chains[0].nodes[0] {
            FilterGraphNode::Loudnorm { i, lra, tp } => {
                assert!((*i - -23.0_f32).abs() < 0.01, "default I should be -23");
                assert!((*lra - 7.0_f32).abs() < 0.01, "default LRA should be 7");
                assert!((*tp - -2.0_f32).abs() < 0.01, "default TP should be -2");
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_loudnorm_custom() {
        let g = FilterGraph::parse("loudnorm=I=-16:LRA=11:TP=-1.5").expect("parse");
        match &g.chains[0].nodes[0] {
            FilterGraphNode::Loudnorm { i, lra, tp } => {
                assert!((*i - -16.0_f32).abs() < 0.01);
                assert!((*lra - 11.0_f32).abs() < 0.01);
                assert!((*tp - -1.5_f32).abs() < 0.01);
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_pan_filter() {
        let g = FilterGraph::parse("pan=stereo|c0=c0|c1=c1").expect("parse");
        assert!(
            matches!(&g.chains[0].nodes[0], FilterGraphNode::Pan(s) if s == "stereo|c0=c0|c1=c1")
        );
    }

    // ── multi-filter chains ──────────────────────────────────────────────────

    #[test]
    fn test_multi_filter_chain() {
        let g = FilterGraph::parse("scale=1280:720,fps=30,hflip").expect("parse");
        assert_eq!(g.chains.len(), 1);
        assert_eq!(g.chains[0].nodes.len(), 3);
        assert!(matches!(
            &g.chains[0].nodes[0],
            FilterGraphNode::Scale {
                width: 1280,
                height: 720,
                ..
            }
        ));
        assert!(matches!(&g.chains[0].nodes[1], FilterGraphNode::Fps { .. }));
        assert!(matches!(&g.chains[0].nodes[2], FilterGraphNode::Hflip));
    }

    #[test]
    fn test_multi_chain_semicolon() {
        let g = FilterGraph::parse("[0:v]scale=1920:1080[s];[s]fps=24[out]").expect("parse");
        assert_eq!(g.chains.len(), 2);
        assert_eq!(g.chains[0].input_label.as_deref(), Some("0:v"));
        assert_eq!(g.chains[0].output_label.as_deref(), Some("s"));
        assert_eq!(g.chains[1].input_label.as_deref(), Some("s"));
        assert_eq!(g.chains[1].output_label.as_deref(), Some("out"));
    }

    // ── label parsing ────────────────────────────────────────────────────────

    #[test]
    fn test_input_output_labels() {
        let g = FilterGraph::parse("[in]scale=640:480[out]").expect("parse");
        let chain = &g.chains[0];
        assert_eq!(chain.input_label.as_deref(), Some("in"));
        assert_eq!(chain.output_label.as_deref(), Some("out"));
        assert!(matches!(
            &chain.nodes[0],
            FilterGraphNode::Scale {
                width: 640,
                height: 480,
                ..
            }
        ));
    }

    #[test]
    fn test_label_only_input() {
        let g = FilterGraph::parse("[src]hflip").expect("parse");
        assert_eq!(g.chains[0].input_label.as_deref(), Some("src"));
        assert_eq!(g.chains[0].output_label, None);
    }

    // ── helper methods ───────────────────────────────────────────────────────

    #[test]
    fn test_has_audio_filters_true() {
        let g = FilterGraph::parse("volume=0.8").expect("parse");
        assert!(g.has_audio_filters());
        assert!(!g.has_video_filters());
    }

    #[test]
    fn test_has_video_filters_true() {
        let g = FilterGraph::parse("scale=1280:720").expect("parse");
        assert!(g.has_video_filters());
        assert!(!g.has_audio_filters());
    }

    #[test]
    fn test_scale_target() {
        let g = FilterGraph::parse("scale=1920:1080,fps=30").expect("parse");
        assert_eq!(g.scale_target(), Some((1920, 1080)));
    }

    #[test]
    fn test_scale_target_none() {
        let g = FilterGraph::parse("hflip,vflip").expect("parse");
        assert_eq!(g.scale_target(), None);
    }

    #[test]
    fn test_pipeline_description_non_empty() {
        let g = FilterGraph::parse("scale=1920:1080,fps=30").expect("parse");
        let desc = g.to_pipeline_description();
        assert!(desc.contains("Scale 1920x1080"));
        assert!(desc.contains("Force FPS=30"));
    }

    #[test]
    fn test_unknown_node_preserved() {
        let g = FilterGraph::parse("someunknownfilter=x=1:y=2").expect("parse");
        let unknowns = g.unknown_nodes();
        assert_eq!(unknowns.len(), 1);
        assert_eq!(unknowns[0].0, "someunknownfilter");
    }

    #[test]
    fn test_trim_with_start_end() {
        let g = FilterGraph::parse("trim=start=2.5:end=10.0").expect("parse");
        match &g.chains[0].nodes[0] {
            FilterGraphNode::Trim {
                start,
                end,
                duration,
            } => {
                assert!((start.expect("start") - 2.5).abs() < 0.001);
                assert!((end.expect("end") - 10.0).abs() < 0.001);
                assert!(duration.is_none());
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_overlay_expressions() {
        let g = FilterGraph::parse("overlay=(W-w)/2:(H-h)/2").expect("parse");
        match &g.chains[0].nodes[0] {
            FilterGraphNode::Overlay { x, y } => {
                assert_eq!(x, "(W-w)/2");
                assert_eq!(y, "(H-h)/2");
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_complex_av_graph() {
        // Simulates: -filter_complex "[0:v]scale=1920:1080[v];[0:a]loudnorm=I=-23:LRA=7:TP=-2[a]"
        let g = FilterGraph::parse("[0:v]scale=1920:1080[v];[0:a]loudnorm=I=-23:LRA=7:TP=-2[a]")
            .expect("parse");
        assert_eq!(g.chains.len(), 2);
        assert!(g.has_video_filters());
        assert!(g.has_audio_filters());
        assert_eq!(g.scale_target(), Some((1920, 1080)));
    }
}
