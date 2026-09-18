//! Common LUT Format (CLF / ACES Look Transform Format / DLP) read/write.
//!
//! The Common LUT Format (CLF) is an XML-based interchange format for
//! describing colour transforms as sequences of processing nodes.  It is
//! specified in S-2014-006 by the Academy / ASC.
//!
//! This implementation covers the core processing nodes needed for colour
//! grading pipelines:
//!
//! - `<LUT1D>` — 1-D per-channel look-up table.
//! - `<LUT3D>` — 3-D RGB look-up table.
//! - `<Matrix>` — 3×3 or 3×4 colour matrix.
//! - `<ASC_CDL>` — ASC CDL slope/offset/power/saturation.
//! - `<Gamma>` — power-law or sRGB transfer function.
//! - `<Log>` — log / antilog transfer function.
//!
//! # Reading CLF
//!
//! ```
//! use oximedia_lut::clf::{ClfDocument, ClfError};
//!
//! let xml = r#"<?xml version="1.0" ?>
//! <ProcessList id="test" name="Identity">
//!   <LUT1D id="l1" inBitDepth="32f" outBitDepth="32f">
//!     <Array dim="3 3">
//!       0.0 0.0 0.0
//!       0.5 0.5 0.5
//!       1.0 1.0 1.0
//!     </Array>
//!   </LUT1D>
//! </ProcessList>"#;
//!
//! let doc = ClfDocument::from_xml(xml).expect("parse");
//! assert_eq!(doc.nodes.len(), 1);
//! ```
//!
//! # Writing CLF
//!
//! ```
//! use oximedia_lut::clf::{ClfDocument, ClfNode, ClfLut1d};
//!
//! let lut1d = ClfLut1d {
//!     id: "id0".to_string(),
//!     name: Some("identity".to_string()),
//!     size: 3,
//!     data: vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
//! };
//! let doc = ClfDocument::new("test", "Identity", vec![ClfNode::Lut1d(lut1d)]);
//! let xml = doc.to_xml();
//! assert!(xml.contains("LUT1D"));
//! ```

use crate::error::{LutError, LutResult};
use crate::Rgb;

// ============================================================================
// CLF Node types
// ============================================================================

/// A 1-D per-channel LUT node in CLF format.
#[derive(Clone, Debug)]
pub struct ClfLut1d {
    /// Node identifier.
    pub id: String,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// Number of entries per channel.
    pub size: usize,
    /// LUT data: `data[i] = [r, g, b]` at normalised input `i / (size-1)`.
    pub data: Vec<[f64; 3]>,
}

/// A 3-D LUT node in CLF format.
#[derive(Clone, Debug)]
pub struct ClfLut3d {
    /// Node identifier.
    pub id: String,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// Grid size per axis.
    pub size: usize,
    /// Flat lattice data in `[r][g][b]` order.
    pub data: Vec<Rgb>,
}

/// A 3×3 or 3×4 matrix node.
#[derive(Clone, Debug)]
pub struct ClfMatrix {
    /// Node identifier.
    pub id: String,
    /// Optional name.
    pub name: Option<String>,
    /// Row-major matrix values.  9 values for 3×3, 12 for 3×4.
    pub values: Vec<f64>,
    /// Whether the matrix includes an offset column (3×4).
    pub has_offset: bool,
}

/// ASC CDL parameters.
#[derive(Clone, Debug)]
pub struct ClfAscCdl {
    /// Node identifier.
    pub id: String,
    /// Optional name.
    pub name: Option<String>,
    /// Slope for each channel `[r, g, b]`.
    pub slope: [f64; 3],
    /// Offset for each channel.
    pub offset: [f64; 3],
    /// Power (gamma) for each channel.
    pub power: [f64; 3],
    /// Saturation scalar.
    pub saturation: f64,
}

impl Default for ClfAscCdl {
    fn default() -> Self {
        Self {
            id: "cdl0".to_string(),
            name: None,
            slope: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
            power: [1.0, 1.0, 1.0],
            saturation: 1.0,
        }
    }
}

/// Gamma / power-law node.
#[derive(Clone, Debug)]
pub struct ClfGamma {
    /// Node identifier.
    pub id: String,
    /// Optional name.
    pub name: Option<String>,
    /// Exponent (gamma) value per channel `[r, g, b]`.
    pub exponent: [f64; 3],
    /// Style: `"basicFwd"` (power) or `"sRGBFwd"` (sRGB).
    pub style: String,
}

/// Log / antilog node.
#[derive(Clone, Debug)]
pub struct ClfLog {
    /// Node identifier.
    pub id: String,
    /// Optional name.
    pub name: Option<String>,
    /// Style: `"log10"`, `"log2"`, `"antiLog10"`, `"antiLog2"`, `"linToLog"`, `"logToLin"`.
    pub style: String,
    /// Base of logarithm (default 10.0).
    pub base: f64,
    /// Optional linear slope used for `linToLog` / `logToLin`.
    pub lin_side_slope: Option<f64>,
    /// Optional linear offset.
    pub lin_side_offset: Option<f64>,
    /// Optional log slope.
    pub log_side_slope: Option<f64>,
    /// Optional log offset.
    pub log_side_offset: Option<f64>,
}

/// A single processing node within a CLF document.
#[derive(Clone, Debug)]
pub enum ClfNode {
    /// 1-D LUT.
    Lut1d(ClfLut1d),
    /// 3-D LUT.
    Lut3d(ClfLut3d),
    /// Matrix.
    Matrix(ClfMatrix),
    /// ASC CDL.
    AscCdl(ClfAscCdl),
    /// Gamma / power law.
    Gamma(ClfGamma),
    /// Log / antilog.
    Log(ClfLog),
}

impl ClfNode {
    /// Return the node's identifier string.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Lut1d(n) => &n.id,
            Self::Lut3d(n) => &n.id,
            Self::Matrix(n) => &n.id,
            Self::AscCdl(n) => &n.id,
            Self::Gamma(n) => &n.id,
            Self::Log(n) => &n.id,
        }
    }

    /// Apply this CLF node to an RGB pixel.
    #[must_use]
    pub fn apply(&self, input: Rgb) -> Rgb {
        match self {
            Self::Lut1d(n) => apply_clf_lut1d(n, &input),
            Self::Lut3d(n) => apply_clf_lut3d(n, &input),
            Self::Matrix(n) => apply_clf_matrix(n, &input),
            Self::AscCdl(n) => apply_clf_asc_cdl(n, &input),
            Self::Gamma(n) => apply_clf_gamma(n, &input),
            Self::Log(n) => apply_clf_log(n, &input),
        }
    }
}

// ============================================================================
// ClfDocument
// ============================================================================

/// A CLF document containing a sequence of processing nodes.
#[derive(Clone, Debug)]
pub struct ClfDocument {
    /// Unique identifier for the `ProcessList`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Ordered list of processing nodes.
    pub nodes: Vec<ClfNode>,
    /// Optional description string.
    pub description: Option<String>,
}

/// CLF-specific error type.
#[derive(Debug, Clone)]
pub struct ClfError(pub String);

impl std::fmt::Display for ClfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CLF error: {}", self.0)
    }
}

impl std::error::Error for ClfError {}

impl From<ClfError> for LutError {
    fn from(e: ClfError) -> Self {
        Self::Parse(e.0)
    }
}

impl ClfDocument {
    /// Create a new CLF document.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, nodes: Vec<ClfNode>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            nodes,
            description: None,
        }
    }

    /// Apply all nodes in sequence to an RGB pixel.
    #[must_use]
    pub fn apply(&self, input: Rgb) -> Rgb {
        let mut pixel = input;
        for node in &self.nodes {
            pixel = node.apply(pixel);
        }
        pixel
    }

    // -----------------------------------------------------------------------
    // XML serialisation
    // -----------------------------------------------------------------------

    /// Serialise this CLF document to an XML string.
    #[must_use]
    pub fn to_xml(&self) -> String {
        let mut s = String::with_capacity(4096);
        s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        s.push_str(&format!(
            "<ProcessList id=\"{}\" name=\"{}\" compCLFversion=\"3\">\n",
            xml_escape(&self.id),
            xml_escape(&self.name)
        ));
        if let Some(desc) = &self.description {
            s.push_str(&format!(
                "  <Description>{}</Description>\n",
                xml_escape(desc)
            ));
        }
        for node in &self.nodes {
            s.push_str(&serialise_node(node));
        }
        s.push_str("</ProcessList>\n");
        s
    }

    // -----------------------------------------------------------------------
    // XML deserialisation (minimal, hand-rolled to avoid extra deps)
    // -----------------------------------------------------------------------

    /// Parse a CLF document from an XML string.
    ///
    /// # Errors
    ///
    /// Returns a [`LutResult`] error if the XML is malformed or missing
    /// required fields.
    pub fn from_xml(xml: &str) -> LutResult<Self> {
        let id = xml_attr(xml, "ProcessList", "id").unwrap_or_else(|| "unknown".to_string());
        let name = xml_attr(xml, "ProcessList", "name").unwrap_or_else(|| "".to_string());

        let description = xml_inner(xml, "Description");

        let mut nodes = Vec::new();

        // Iterate through top-level child elements.
        let mut rest = xml;
        while !rest.is_empty() {
            if let Some(pos) = rest.find('<') {
                let tag_start = &rest[pos + 1..];
                // Determine the tag name.
                let tag_name: String = tag_start
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                match tag_name.as_str() {
                    "LUT1D" => {
                        if let Some((block, after)) = xml_element_block(rest, "LUT1D") {
                            nodes.push(ClfNode::Lut1d(parse_lut1d(&block)?));
                            rest = after;
                            continue;
                        }
                    }
                    "LUT3D" => {
                        if let Some((block, after)) = xml_element_block(rest, "LUT3D") {
                            nodes.push(ClfNode::Lut3d(parse_lut3d(&block)?));
                            rest = after;
                            continue;
                        }
                    }
                    "Matrix" => {
                        if let Some((block, after)) = xml_element_block(rest, "Matrix") {
                            nodes.push(ClfNode::Matrix(parse_matrix(&block)?));
                            rest = after;
                            continue;
                        }
                    }
                    "ASC_CDL" => {
                        if let Some((block, after)) = xml_element_block(rest, "ASC_CDL") {
                            nodes.push(ClfNode::AscCdl(parse_asc_cdl(&block)?));
                            rest = after;
                            continue;
                        }
                    }
                    "Gamma" => {
                        if let Some((block, after)) = xml_element_block(rest, "Gamma") {
                            nodes.push(ClfNode::Gamma(parse_gamma(&block)?));
                            rest = after;
                            continue;
                        }
                    }
                    "Log" => {
                        if let Some((block, after)) = xml_element_block(rest, "Log") {
                            nodes.push(ClfNode::Log(parse_log(&block)?));
                            rest = after;
                            continue;
                        }
                    }
                    _ => {}
                }
                // Skip past this '<'
                rest = &rest[pos + 1..];
            } else {
                break;
            }
        }

        Ok(Self {
            id,
            name,
            nodes,
            description,
        })
    }
}

// ============================================================================
// Node application helpers
// ============================================================================

fn apply_clf_lut1d(n: &ClfLut1d, input: &Rgb) -> Rgb {
    if n.size < 2 || n.data.is_empty() {
        return *input;
    }
    let scale = (n.size - 1) as f64;
    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        let v = input[ch].clamp(0.0, 1.0) * scale;
        let lo = v.floor() as usize;
        let hi = (lo + 1).min(n.size - 1);
        let t = v - lo as f64;
        let a = n.data.get(lo).map(|d| d[ch]).unwrap_or(0.0);
        let b_val = n.data.get(hi).map(|d| d[ch]).unwrap_or(a);
        out[ch] = a * (1.0 - t) + b_val * t;
    }
    out
}

fn apply_clf_lut3d(n: &ClfLut3d, input: &Rgb) -> Rgb {
    if n.size < 2 || n.data.len() != n.size * n.size * n.size {
        return *input;
    }
    let scale = (n.size - 1) as f64;
    let r = input[0].clamp(0.0, 1.0) * scale;
    let g = input[1].clamp(0.0, 1.0) * scale;
    let b = input[2].clamp(0.0, 1.0) * scale;
    let r0 = r.floor() as usize;
    let g0 = g.floor() as usize;
    let b0 = b.floor() as usize;
    let r1 = (r0 + 1).min(n.size - 1);
    let g1 = (g0 + 1).min(n.size - 1);
    let b1 = (b0 + 1).min(n.size - 1);
    let dr = r - r0 as f64;
    let dg = g - g0 as f64;
    let db = b - b0 as f64;

    let sz = n.size;
    let at = |ri: usize, gi: usize, bi: usize| n.data[ri * sz * sz + gi * sz + bi];

    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        let c000 = at(r0, g0, b0)[ch];
        let c100 = at(r1, g0, b0)[ch];
        let c010 = at(r0, g1, b0)[ch];
        let c110 = at(r1, g1, b0)[ch];
        let c001 = at(r0, g0, b1)[ch];
        let c101 = at(r1, g0, b1)[ch];
        let c011 = at(r0, g1, b1)[ch];
        let c111 = at(r1, g1, b1)[ch];
        out[ch] = c000 * (1.0 - dr) * (1.0 - dg) * (1.0 - db)
            + c100 * dr * (1.0 - dg) * (1.0 - db)
            + c010 * (1.0 - dr) * dg * (1.0 - db)
            + c110 * dr * dg * (1.0 - db)
            + c001 * (1.0 - dr) * (1.0 - dg) * db
            + c101 * dr * (1.0 - dg) * db
            + c011 * (1.0 - dr) * dg * db
            + c111 * dr * dg * db;
    }
    out
}

fn apply_clf_matrix(n: &ClfMatrix, input: &Rgb) -> Rgb {
    let v = n.values.as_slice();
    if n.has_offset && v.len() >= 12 {
        [
            v[0] * input[0] + v[1] * input[1] + v[2] * input[2] + v[3],
            v[4] * input[0] + v[5] * input[1] + v[6] * input[2] + v[7],
            v[8] * input[0] + v[9] * input[1] + v[10] * input[2] + v[11],
        ]
    } else if v.len() >= 9 {
        [
            v[0] * input[0] + v[1] * input[1] + v[2] * input[2],
            v[3] * input[0] + v[4] * input[1] + v[5] * input[2],
            v[6] * input[0] + v[7] * input[1] + v[8] * input[2],
        ]
    } else {
        *input
    }
}

fn apply_clf_asc_cdl(n: &ClfAscCdl, input: &Rgb) -> Rgb {
    // ASC CDL formula: out = clamp(slope * in + offset) ^ power
    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        let v = (n.slope[ch] * input[ch] + n.offset[ch]).clamp(0.0, 1.0);
        out[ch] = v.powf(n.power[ch]);
    }
    // Apply saturation
    if (n.saturation - 1.0).abs() > 1e-9 {
        let luma = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
        for ch in 0..3 {
            out[ch] = luma + n.saturation * (out[ch] - luma);
        }
    }
    out
}

fn apply_clf_gamma(n: &ClfGamma, input: &Rgb) -> Rgb {
    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        let v = input[ch].max(0.0);
        out[ch] = if n.style == "sRGBFwd" {
            if v <= 0.003_130_8 {
                v * 12.92
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            }
        } else if n.style == "sRGBRev" {
            if v <= 0.040_45 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        } else {
            // basicFwd: simple power law
            v.powf(n.exponent[ch])
        };
    }
    out
}

fn apply_clf_log(n: &ClfLog, input: &Rgb) -> Rgb {
    let base = n.base;
    let log_base = base.max(1e-300).ln();
    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        let v = input[ch];
        out[ch] = match n.style.as_str() {
            "log10" => {
                if v > 0.0 {
                    v.log10()
                } else {
                    -10.0
                }
            }
            "log2" => {
                if v > 0.0 {
                    v.log2()
                } else {
                    -50.0
                }
            }
            "antiLog10" => 10.0_f64.powf(v),
            "antiLog2" => 2.0_f64.powf(v),
            "linToLog" => {
                let a = n.lin_side_slope.unwrap_or(1.0);
                let b = n.lin_side_offset.unwrap_or(0.0);
                let c = n.log_side_slope.unwrap_or(1.0);
                let d = n.log_side_offset.unwrap_or(0.0);
                let lin_v = a * v + b;
                if lin_v > 0.0 {
                    c * (lin_v.ln() / log_base) + d
                } else {
                    d
                }
            }
            "logToLin" => {
                let a = n.lin_side_slope.unwrap_or(1.0);
                let b = n.lin_side_offset.unwrap_or(0.0);
                let c = n.log_side_slope.unwrap_or(1.0);
                let d = n.log_side_offset.unwrap_or(0.0);
                let log_v = (v - d) / c.max(1e-300);
                (base.powf(log_v) - b) / a.max(1e-300)
            }
            _ => v,
        };
    }
    out
}

// ============================================================================
// XML serialisation helpers
// ============================================================================

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn serialise_node(node: &ClfNode) -> String {
    match node {
        ClfNode::Lut1d(n) => serialise_lut1d(n),
        ClfNode::Lut3d(n) => serialise_lut3d(n),
        ClfNode::Matrix(n) => serialise_matrix(n),
        ClfNode::AscCdl(n) => serialise_asc_cdl(n),
        ClfNode::Gamma(n) => serialise_gamma(n),
        ClfNode::Log(n) => serialise_log(n),
    }
}

fn serialise_lut1d(n: &ClfLut1d) -> String {
    let name_attr = n
        .name
        .as_deref()
        .map(|nm| format!(" name=\"{}\"", xml_escape(nm)))
        .unwrap_or_default();
    let mut s = format!(
        "  <LUT1D id=\"{}\"{} inBitDepth=\"32f\" outBitDepth=\"32f\">\n",
        xml_escape(&n.id),
        name_attr
    );
    s.push_str(&format!("    <Array dim=\"{} 3\">\n", n.size));
    for entry in &n.data {
        s.push_str(&format!(
            "      {:.10} {:.10} {:.10}\n",
            entry[0], entry[1], entry[2]
        ));
    }
    s.push_str("    </Array>\n  </LUT1D>\n");
    s
}

fn serialise_lut3d(n: &ClfLut3d) -> String {
    let name_attr = n
        .name
        .as_deref()
        .map(|nm| format!(" name=\"{}\"", xml_escape(nm)))
        .unwrap_or_default();
    let mut s = format!(
        "  <LUT3D id=\"{}\"{} inBitDepth=\"32f\" outBitDepth=\"32f\">\n",
        xml_escape(&n.id),
        name_attr
    );
    s.push_str(&format!("    <Array dim=\"{0} {0} {0} 3\">\n", n.size));
    for entry in &n.data {
        s.push_str(&format!(
            "      {:.10} {:.10} {:.10}\n",
            entry[0], entry[1], entry[2]
        ));
    }
    s.push_str("    </Array>\n  </LUT3D>\n");
    s
}

fn serialise_matrix(n: &ClfMatrix) -> String {
    let name_attr = n
        .name
        .as_deref()
        .map(|nm| format!(" name=\"{}\"", xml_escape(nm)))
        .unwrap_or_default();
    let cols = if n.has_offset { 4 } else { 3 };
    let mut s = format!(
        "  <Matrix id=\"{}\"{} inBitDepth=\"32f\" outBitDepth=\"32f\">\n",
        xml_escape(&n.id),
        name_attr
    );
    s.push_str(&format!("    <Array dim=\"3 {cols}\">\n"));
    for row in 0..3 {
        let base = row * cols;
        let vals: Vec<String> = (0..cols)
            .map(|c| format!("{:.10}", n.values.get(base + c).copied().unwrap_or(0.0)))
            .collect();
        s.push_str(&format!("      {}\n", vals.join(" ")));
    }
    s.push_str("    </Array>\n  </Matrix>\n");
    s
}

fn serialise_asc_cdl(n: &ClfAscCdl) -> String {
    let name_attr = n
        .name
        .as_deref()
        .map(|nm| format!(" name=\"{}\"", xml_escape(nm)))
        .unwrap_or_default();
    format!(
        "  <ASC_CDL id=\"{}\"{} inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <SOPNode>\n      <Slope>{:.10} {:.10} {:.10}</Slope>\n      <Offset>{:.10} {:.10} {:.10}</Offset>\n      <Power>{:.10} {:.10} {:.10}</Power>\n    </SOPNode>\n    <SatNode>\n      <Saturation>{:.10}</Saturation>\n    </SatNode>\n  </ASC_CDL>\n",
        xml_escape(&n.id), name_attr,
        n.slope[0], n.slope[1], n.slope[2],
        n.offset[0], n.offset[1], n.offset[2],
        n.power[0], n.power[1], n.power[2],
        n.saturation
    )
}

fn serialise_gamma(n: &ClfGamma) -> String {
    let name_attr = n
        .name
        .as_deref()
        .map(|nm| format!(" name=\"{}\"", xml_escape(nm)))
        .unwrap_or_default();
    format!(
        "  <Gamma id=\"{}\"{} inBitDepth=\"32f\" outBitDepth=\"32f\" style=\"{}\">\n    <GammaParams gamma=\"{:.10} {:.10} {:.10}\"/>\n  </Gamma>\n",
        xml_escape(&n.id), name_attr, xml_escape(&n.style),
        n.exponent[0], n.exponent[1], n.exponent[2]
    )
}

fn serialise_log(n: &ClfLog) -> String {
    let name_attr = n
        .name
        .as_deref()
        .map(|nm| format!(" name=\"{}\"", xml_escape(nm)))
        .unwrap_or_default();
    format!(
        "  <Log id=\"{}\"{} inBitDepth=\"32f\" outBitDepth=\"32f\" style=\"{}\"/>\n",
        xml_escape(&n.id),
        name_attr,
        xml_escape(&n.style)
    )
}

// ============================================================================
// XML deserialisation helpers
// ============================================================================

/// Extract the content of the first `<tag>...</tag>` block found in `s`.
fn xml_inner(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let start = s.find(&open)?;
    let after_open = &s[start..];
    // Find end of opening tag.
    let tag_end = after_open.find('>')?;
    let inner_start = tag_end + 1;
    let inner = &after_open[inner_start..];
    let end = inner.find(&close)?;
    Some(inner[..end].trim().to_string())
}

/// Extract an attribute value from the first occurrence of `<tag ... attr="value"`.
fn xml_attr(s: &str, tag: &str, attr: &str) -> Option<String> {
    let open = format!("<{tag}");
    let pos = s.find(&open)?;
    let frag = &s[pos..];
    let end_of_tag = frag.find('>')?;
    let tag_frag = &frag[..end_of_tag];
    let attr_search = format!("{attr}=\"");
    let attr_pos = tag_frag.find(&attr_search)?;
    let val_start = attr_pos + attr_search.len();
    let val_frag = &tag_frag[val_start..];
    let val_end = val_frag.find('"')?;
    Some(val_frag[..val_end].to_string())
}

/// Find the first complete `<tag ...>...</tag>` block in `s`.
/// Returns `(block_contents_including_outer_tags, rest_of_string)`.
fn xml_element_block<'a>(s: &'a str, tag: &str) -> Option<(String, &'a str)> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let start = s.find(&open)?;
    let after_start = &s[start..];
    let close_pos = after_start.find(&close)?;
    let end_pos = close_pos + close.len();
    let block = after_start[..end_pos].to_string();
    Some((block, &s[start + end_pos..]))
}

/// Parse space/newline-separated f64 values from a string.
fn parse_floats(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

fn parse_lut1d(xml: &str) -> LutResult<ClfLut1d> {
    let id = xml_attr(xml, "LUT1D", "id").unwrap_or_else(|| "l1".to_string());
    let name = xml_attr(xml, "LUT1D", "name");

    let array_inner = xml_inner(xml, "Array")
        .ok_or_else(|| LutError::Parse("LUT1D missing <Array>".to_string()))?;
    let floats = parse_floats(&array_inner);

    if floats.len() % 3 != 0 {
        return Err(LutError::Parse(format!(
            "LUT1D array length {} not divisible by 3",
            floats.len()
        )));
    }
    let size = floats.len() / 3;
    let data: Vec<[f64; 3]> = floats.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();

    Ok(ClfLut1d {
        id,
        name,
        size,
        data,
    })
}

fn parse_lut3d(xml: &str) -> LutResult<ClfLut3d> {
    let id = xml_attr(xml, "LUT3D", "id").unwrap_or_else(|| "l3".to_string());
    let name = xml_attr(xml, "LUT3D", "name");

    let array_inner = xml_inner(xml, "Array")
        .ok_or_else(|| LutError::Parse("LUT3D missing <Array>".to_string()))?;
    let floats = parse_floats(&array_inner);

    if floats.len() % 3 != 0 {
        return Err(LutError::Parse(format!(
            "LUT3D array length {} not divisible by 3",
            floats.len()
        )));
    }
    let total = floats.len() / 3;
    // Compute size from cube root.
    let size = (total as f64).cbrt().round() as usize;
    if size * size * size != total {
        return Err(LutError::Parse(format!(
            "LUT3D array {total} entries is not a perfect cube"
        )));
    }
    let data: Vec<Rgb> = floats.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();

    Ok(ClfLut3d {
        id,
        name,
        size,
        data,
    })
}

fn parse_matrix(xml: &str) -> LutResult<ClfMatrix> {
    let id = xml_attr(xml, "Matrix", "id").unwrap_or_else(|| "m1".to_string());
    let name = xml_attr(xml, "Matrix", "name");

    let array_inner = xml_inner(xml, "Array")
        .ok_or_else(|| LutError::Parse("Matrix missing <Array>".to_string()))?;
    let values = parse_floats(&array_inner);
    let has_offset = values.len() == 12;

    Ok(ClfMatrix {
        id,
        name,
        values,
        has_offset,
    })
}

fn parse_asc_cdl(xml: &str) -> LutResult<ClfAscCdl> {
    let id = xml_attr(xml, "ASC_CDL", "id").unwrap_or_else(|| "cdl1".to_string());
    let name = xml_attr(xml, "ASC_CDL", "name");

    let slope_s = xml_inner(xml, "Slope").unwrap_or_else(|| "1.0 1.0 1.0".to_string());
    let offset_s = xml_inner(xml, "Offset").unwrap_or_else(|| "0.0 0.0 0.0".to_string());
    let power_s = xml_inner(xml, "Power").unwrap_or_else(|| "1.0 1.0 1.0".to_string());
    let sat_s = xml_inner(xml, "Saturation").unwrap_or_else(|| "1.0".to_string());

    let sv = parse_floats(&slope_s);
    let ov = parse_floats(&offset_s);
    let pv = parse_floats(&power_s);
    let saturation = parse_floats(&sat_s).first().copied().unwrap_or(1.0);

    Ok(ClfAscCdl {
        id,
        name,
        slope: [
            sv.first().copied().unwrap_or(1.0),
            sv.get(1).copied().unwrap_or(1.0),
            sv.get(2).copied().unwrap_or(1.0),
        ],
        offset: [
            ov.first().copied().unwrap_or(0.0),
            ov.get(1).copied().unwrap_or(0.0),
            ov.get(2).copied().unwrap_or(0.0),
        ],
        power: [
            pv.first().copied().unwrap_or(1.0),
            pv.get(1).copied().unwrap_or(1.0),
            pv.get(2).copied().unwrap_or(1.0),
        ],
        saturation,
    })
}

fn parse_gamma(xml: &str) -> LutResult<ClfGamma> {
    let id = xml_attr(xml, "Gamma", "id").unwrap_or_else(|| "g1".to_string());
    let name = xml_attr(xml, "Gamma", "name");
    let style = xml_attr(xml, "Gamma", "style").unwrap_or_else(|| "basicFwd".to_string());

    let gamma_s =
        xml_attr(xml, "GammaParams", "gamma").unwrap_or_else(|| "1.0 1.0 1.0".to_string());
    let gv = parse_floats(&gamma_s);

    Ok(ClfGamma {
        id,
        name,
        style,
        exponent: [
            gv.first().copied().unwrap_or(1.0),
            gv.get(1).copied().unwrap_or(1.0),
            gv.get(2).copied().unwrap_or(1.0),
        ],
    })
}

fn parse_log(xml: &str) -> LutResult<ClfLog> {
    let id = xml_attr(xml, "Log", "id").unwrap_or_else(|| "log1".to_string());
    let name = xml_attr(xml, "Log", "name");
    let style = xml_attr(xml, "Log", "style").unwrap_or_else(|| "log10".to_string());
    let base_s = xml_attr(xml, "Log", "base").unwrap_or_else(|| "10.0".to_string());
    let base = base_s.parse::<f64>().unwrap_or(10.0);

    Ok(ClfLog {
        id,
        name,
        style,
        base,
        lin_side_slope: None,
        lin_side_offset: None,
        log_side_slope: None,
        log_side_offset: None,
    })
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut3d(size: usize) -> Vec<Rgb> {
        let scale = (size - 1) as f64;
        let mut lut = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    lut.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        lut
    }

    #[test]
    fn test_clf_lut1d_apply_identity() {
        let node = ClfLut1d {
            id: "id".to_string(),
            name: None,
            size: 3,
            data: vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        };
        let doc = ClfDocument::new("test", "t", vec![ClfNode::Lut1d(node)]);
        let out = doc.apply([0.5, 0.5, 0.5]);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_clf_lut3d_apply_identity() {
        let lut = identity_lut3d(3);
        let node = ClfLut3d {
            id: "id".to_string(),
            name: None,
            size: 3,
            data: lut,
        };
        let doc = ClfDocument::new("test", "t", vec![ClfNode::Lut3d(node)]);
        let inp = [0.5, 0.3, 0.7];
        let out = doc.apply(inp);
        for ch in 0..3 {
            assert!((out[ch] - inp[ch]).abs() < 1e-6, "ch={ch}");
        }
    }

    #[test]
    fn test_clf_matrix_identity() {
        let node = ClfMatrix {
            id: "m".to_string(),
            name: None,
            values: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            has_offset: false,
        };
        let doc = ClfDocument::new("t", "t", vec![ClfNode::Matrix(node)]);
        let inp = [0.4, 0.6, 0.2];
        let out = doc.apply(inp);
        for ch in 0..3 {
            assert!((out[ch] - inp[ch]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_clf_matrix_with_offset() {
        // Matrix that adds 0.1 to each channel.
        let node = ClfMatrix {
            id: "m".to_string(),
            name: None,
            values: vec![1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.1],
            has_offset: true,
        };
        let doc = ClfDocument::new("t", "t", vec![ClfNode::Matrix(node)]);
        let out = doc.apply([0.0, 0.0, 0.0]);
        assert!((out[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_clf_asc_cdl_identity() {
        let node = ClfAscCdl::default();
        let doc = ClfDocument::new("t", "t", vec![ClfNode::AscCdl(node)]);
        let inp = [0.4, 0.6, 0.2];
        let out = doc.apply(inp);
        for ch in 0..3 {
            assert!((out[ch] - inp[ch]).abs() < 1e-9, "ch={ch}");
        }
    }

    #[test]
    fn test_clf_asc_cdl_saturation_zero() {
        let mut node = ClfAscCdl::default();
        node.saturation = 0.0;
        let doc = ClfDocument::new("t", "t", vec![ClfNode::AscCdl(node)]);
        let out = doc.apply([0.8, 0.3, 0.1]);
        // All channels should equal luma.
        let luma = 0.2126 * 0.8 + 0.7152 * 0.3 + 0.0722 * 0.1;
        for ch in 0..3 {
            assert!((out[ch] - luma).abs() < 1e-9, "ch={ch}");
        }
    }

    #[test]
    fn test_clf_gamma_power_law() {
        let node = ClfGamma {
            id: "g".to_string(),
            name: None,
            style: "basicFwd".to_string(),
            exponent: [2.2, 2.2, 2.2],
        };
        let doc = ClfDocument::new("t", "t", vec![ClfNode::Gamma(node)]);
        let out = doc.apply([0.5, 0.5, 0.5]);
        let expected = 0.5_f64.powf(2.2);
        assert!((out[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn test_clf_log_node() {
        let node = ClfLog {
            id: "l".to_string(),
            name: None,
            style: "antiLog10".to_string(),
            base: 10.0,
            lin_side_slope: None,
            lin_side_offset: None,
            log_side_slope: None,
            log_side_offset: None,
        };
        let doc = ClfDocument::new("t", "t", vec![ClfNode::Log(node)]);
        let out = doc.apply([0.0, 0.0, 0.0]); // antilog10(0) = 1
        assert!((out[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_clf_serialise_deserialise_lut1d() {
        let lut1d = ClfLut1d {
            id: "id0".to_string(),
            name: Some("test lut".to_string()),
            size: 3,
            data: vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        };
        let doc = ClfDocument::new("doc1", "Test", vec![ClfNode::Lut1d(lut1d)]);
        let xml = doc.to_xml();

        assert!(xml.contains("LUT1D"));
        assert!(xml.contains("ProcessList"));

        // Round-trip parse.
        let doc2 = ClfDocument::from_xml(&xml).expect("parse");
        assert_eq!(doc2.nodes.len(), 1);
        if let ClfNode::Lut1d(n) = &doc2.nodes[0] {
            assert_eq!(n.size, 3);
            assert!((n.data[1][0] - 0.5).abs() < 1e-6);
        } else {
            panic!("Expected LUT1D node");
        }
    }

    #[test]
    fn test_clf_serialise_deserialise_lut3d() {
        let lut3d = ClfLut3d {
            id: "l3".to_string(),
            name: None,
            size: 2,
            data: identity_lut3d(2),
        };
        let doc = ClfDocument::new("doc", "Test3d", vec![ClfNode::Lut3d(lut3d)]);
        let xml = doc.to_xml();
        let doc2 = ClfDocument::from_xml(&xml).expect("parse");
        assert_eq!(doc2.nodes.len(), 1);
        if let ClfNode::Lut3d(n) = &doc2.nodes[0] {
            assert_eq!(n.size, 2);
        } else {
            panic!("Expected LUT3D node");
        }
    }

    #[test]
    fn test_clf_serialise_deserialise_asc_cdl() {
        let cdl = ClfAscCdl {
            id: "cdl1".to_string(),
            name: Some("grade".to_string()),
            slope: [1.1, 0.9, 1.0],
            offset: [0.05, -0.05, 0.0],
            power: [0.9, 1.1, 1.0],
            saturation: 0.8,
        };
        let doc = ClfDocument::new("d", "n", vec![ClfNode::AscCdl(cdl)]);
        let xml = doc.to_xml();
        let doc2 = ClfDocument::from_xml(&xml).expect("parse");
        assert_eq!(doc2.nodes.len(), 1);
    }

    #[test]
    fn test_clf_multi_node_pipeline() {
        // Build a simple pipeline: gamma → identity LUT.
        let gamma_node = ClfGamma {
            id: "g0".to_string(),
            name: None,
            style: "basicFwd".to_string(),
            exponent: [1.0, 1.0, 1.0],
        };
        let lut = identity_lut3d(3);
        let lut_node = ClfLut3d {
            id: "l0".to_string(),
            name: None,
            size: 3,
            data: lut,
        };
        let doc = ClfDocument::new(
            "pipe",
            "pipeline",
            vec![ClfNode::Gamma(gamma_node), ClfNode::Lut3d(lut_node)],
        );
        let inp = [0.4, 0.6, 0.2];
        let out = doc.apply(inp);
        // Gamma 1.0 is identity, LUT is identity → result should be input.
        for ch in 0..3 {
            assert!((out[ch] - inp[ch]).abs() < 1e-5, "ch={ch}");
        }
    }

    #[test]
    fn test_clf_srgb_gamma_node() {
        let node = ClfGamma {
            id: "g".to_string(),
            name: None,
            style: "sRGBFwd".to_string(),
            exponent: [1.0, 1.0, 1.0],
        };
        let doc = ClfDocument::new("t", "t", vec![ClfNode::Gamma(node)]);
        let out = doc.apply([0.0, 1.0, 0.5]);
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!(out[2] > 0.5); // sRGB OETF of 0.5 is > 0.5
    }

    #[test]
    fn test_clf_node_id_access() {
        let node = ClfNode::Lut1d(ClfLut1d {
            id: "my_id".to_string(),
            name: None,
            size: 2,
            data: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        });
        assert_eq!(node.id(), "my_id");
    }
}
