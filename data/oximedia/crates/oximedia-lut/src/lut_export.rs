//! LUT export to additional colour-grading formats.
//!
//! Provides serialisation of `Lut3d` to:
//!
//! - **DaVinci Resolve `.drx`** — XML wrapping the raw LUT data
//! - **Nuke / Shake `.csp`** — `CSPLUTV100` text format
//! - **Iridas / Autodesk `.look`** — XML look format
//!
//! All methods accept a reference to a `Lut3d` and return an owned `String`.

#![allow(dead_code)]

use crate::lut3d::Lut3d;

// ============================================================================
// Public API
// ============================================================================

/// Exporter for `Lut3d` to various third-party colour-grading formats.
pub struct LutExporter;

impl LutExporter {
    /// Export a `Lut3d` as a DaVinci Resolve `.drx` XML document.
    ///
    /// The output embeds the raw 3-D LUT cube data as a flat list of
    /// space-separated float triples, one per line, in R-major order
    /// (outer loop = R, middle = G, inner = B).
    #[must_use]
    pub fn to_resolve_drx(lut: &Lut3d) -> String {
        let size = lut.size();
        let mut data_lines = Vec::with_capacity(size * size * size);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let [rv, gv, bv] = lut.get(r, g, b);
                    data_lines.push(format!("{rv:.6} {gv:.6} {bv:.6}"));
                }
            }
        }

        let data_str = data_lines.join(" ");

        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <resolve_davinci_resolve version=\"1.0\">\n\
             \x20 <lut3d size=\"{size}\">\n\
             \x20   <data>{data_str}</data>\n\
             \x20 </lut3d>\n\
             </resolve_davinci_resolve>\n"
        )
    }

    /// Export a `Lut3d` as a Nuke/Shake `.csp` file.
    ///
    /// The Cinespace format (`CSPLUTV100`) contains:
    /// - A metadata block with the title
    /// - Three pre-scale input arrays (N evenly-spaced values from 0.0 to 1.0)
    /// - A `LUT3D` data block with one RGB triple per line
    #[must_use]
    pub fn to_nuke_csp(lut: &Lut3d) -> String {
        let size = lut.size();

        // Build pre-scale arrays: N evenly spaced values 0..1
        let prescale: Vec<String> = (0..size)
            .map(|i| {
                let v = if size <= 1 {
                    0.0_f64
                } else {
                    i as f64 / (size - 1) as f64
                };
                format!("{v:.6}")
            })
            .collect();
        let prescale_line = prescale.join(" ");

        // Build LUT3D data block
        let mut lut_lines = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let [rv, gv, bv] = lut.get(r, g, b);
                    lut_lines.push(format!("{rv:.6} {gv:.6} {bv:.6}"));
                }
            }
        }

        let mut out = String::new();
        out.push_str("CSPLUTV100\n");
        out.push_str("3D\n");
        out.push_str("BEGIN METADATA\n");
        out.push_str("\"Title\" \"OxiMedia LUT\"\n");
        out.push_str("END METADATA\n");
        // Pre-scale arrays — one line per channel (R, G, B)
        out.push_str(&format!("{size} {size} {size}\n"));
        out.push_str(&prescale_line);
        out.push('\n');
        out.push_str(&prescale_line);
        out.push('\n');
        out.push_str(&prescale_line);
        out.push('\n');
        out.push_str("LUT3D\n");
        for line in &lut_lines {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("END\n");

        out
    }

    /// Export a `Lut3d` as an Iridas/Autodesk `.look` XML document.
    ///
    /// The `.look` format wraps the raw cube data in a simple XML structure
    /// understood by Autodesk Lustre and compatible tools.
    #[must_use]
    pub fn to_iridas_look(lut: &Lut3d) -> String {
        let size = lut.size();
        let mut data_parts = Vec::with_capacity(size * size * size);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let [rv, gv, bv] = lut.get(r, g, b);
                    data_parts.push(format!("{rv:.6} {gv:.6} {bv:.6}"));
                }
            }
        }

        let data_str = data_parts.join(" ");

        format!(
            "<?xml version=\"1.0\"?>\n\
             <look>\n\
             \x20 <LUT>\n\
             \x20   <size>{size}</size>\n\
             \x20   <data>{data_str}</data>\n\
             \x20 </LUT>\n\
             </look>\n"
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lut3d, LutSize};

    // ---- to_resolve_drx ----

    #[test]
    fn test_drx_contains_root_element() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_resolve_drx(&lut);
        assert!(out.contains("<resolve_davinci_resolve"));
    }

    #[test]
    fn test_drx_contains_version_attribute() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_resolve_drx(&lut);
        assert!(out.contains("version=\"1.0\""));
    }

    #[test]
    fn test_drx_contains_lut3d_element_with_size() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_resolve_drx(&lut);
        assert!(out.contains("<lut3d size=\"17\">"));
    }

    #[test]
    fn test_drx_contains_data_element() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_resolve_drx(&lut);
        assert!(out.contains("<data>"));
    }

    #[test]
    fn test_drx_identity_first_entry_is_black() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_resolve_drx(&lut);
        // First entry in identity LUT is 0,0,0 → 0.000000 0.000000 0.000000
        assert!(out.contains("0.000000 0.000000 0.000000"));
    }

    #[test]
    fn test_drx_identity_last_entry_is_white() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_resolve_drx(&lut);
        // Last entry in identity LUT is 1,1,1 → 1.000000 1.000000 1.000000
        assert!(out.contains("1.000000 1.000000 1.000000"));
    }

    // ---- to_nuke_csp ----

    #[test]
    fn test_csp_starts_with_csplutv100() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_nuke_csp(&lut);
        assert!(out.starts_with("CSPLUTV100\n"));
    }

    #[test]
    fn test_csp_contains_3d() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_nuke_csp(&lut);
        assert!(out.contains("3D\n"));
    }

    #[test]
    fn test_csp_contains_lut3d_marker() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_nuke_csp(&lut);
        assert!(out.contains("LUT3D\n"));
    }

    #[test]
    fn test_csp_contains_end_marker() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_nuke_csp(&lut);
        assert!(out.contains("END\n"));
    }

    #[test]
    fn test_csp_identity_first_lut_line_is_black() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_nuke_csp(&lut);
        // After LUT3D header the first line of data should be 0.000000 0.000000 0.000000
        let after_lut3d = out.split("LUT3D\n").nth(1).expect("LUT3D marker present");
        let first_line = after_lut3d.lines().next().expect("at least one data line");
        assert_eq!(first_line, "0.000000 0.000000 0.000000");
    }

    #[test]
    fn test_csp_contains_metadata_block() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_nuke_csp(&lut);
        assert!(out.contains("BEGIN METADATA\n"));
        assert!(out.contains("END METADATA\n"));
    }

    // ---- to_iridas_look ----

    #[test]
    fn test_look_contains_root_element() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_iridas_look(&lut);
        assert!(out.contains("<look>"));
    }

    #[test]
    fn test_look_contains_size_element() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_iridas_look(&lut);
        assert!(out.contains("<size>17</size>"));
    }

    #[test]
    fn test_look_contains_lut_element() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_iridas_look(&lut);
        assert!(out.contains("<LUT>"));
    }

    #[test]
    fn test_look_identity_has_data() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = LutExporter::to_iridas_look(&lut);
        assert!(out.contains("<data>"));
        assert!(out.contains("0.000000 0.000000 0.000000"));
    }
}
