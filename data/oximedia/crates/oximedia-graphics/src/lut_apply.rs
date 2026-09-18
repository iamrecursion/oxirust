#![allow(dead_code)]
//! LUT (Look-Up Table) application for colour grading.
//!
//! Supports 1-D per-channel LUTs and 3-D cube LUTs with trilinear or
//! tetrahedral interpolation.  Parses the Adobe `.cube` file format and
//! provides factory methods for common grading LUTs.

// ─── 1-D LUT ─────────────────────────────────────────────────────────────────

/// A one-dimensional per-channel LUT.
///
/// Each channel table maps an input value in `[input_min, input_max]` to an
/// output value through piecewise-linear interpolation.
#[derive(Debug, Clone)]
pub struct Lut1D {
    /// Descriptive name (e.g. the filename).
    pub name: String,
    /// Number of table entries per channel.
    pub size: usize,
    /// Red output values indexed 0 ..= size-1.
    pub red: Vec<f32>,
    /// Green output values indexed 0 ..= size-1.
    pub green: Vec<f32>,
    /// Blue output values indexed 0 ..= size-1.
    pub blue: Vec<f32>,
    /// Minimum input value (default `0.0`).
    pub input_min: f32,
    /// Maximum input value (default `1.0`).
    pub input_max: f32,
}

impl Lut1D {
    /// Create an identity LUT with `size` entries (output equals input).
    pub fn new(size: usize) -> Self {
        let size = size.max(2);
        let table: Vec<f32> = (0..size).map(|i| i as f32 / (size - 1) as f32).collect();
        Self {
            name: String::from("identity"),
            size,
            red: table.clone(),
            green: table.clone(),
            blue: table,
            input_min: 0.0,
            input_max: 1.0,
        }
    }

    /// Look up a single sample with linear interpolation.
    ///
    /// The input is normalised to `[0, size-1]` then interpolated between
    /// the two nearest table entries.
    fn lookup_channel(table: &[f32], v: f32, input_min: f32, input_max: f32) -> f32 {
        let size = table.len();
        if size == 0 {
            return v;
        }
        let span = (input_max - input_min).max(f32::EPSILON);
        let t = ((v - input_min) / span).clamp(0.0, 1.0);
        let scaled = t * (size - 1) as f32;
        let lo = scaled.floor() as usize;
        let hi = (lo + 1).min(size - 1);
        let frac = scaled - lo as f32;
        table[lo] * (1.0 - frac) + table[hi] * frac
    }

    /// Apply the LUT to an RGB triplet.
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        (
            Self::lookup_channel(&self.red, r, self.input_min, self.input_max),
            Self::lookup_channel(&self.green, g, self.input_min, self.input_max),
            Self::lookup_channel(&self.blue, b, self.input_min, self.input_max),
        )
    }

    /// Parse an Adobe `.cube` 1-D LUT from text.
    ///
    /// The `.cube` format for 1-D LUTs uses the `LUT_1D_SIZE` keyword and
    /// three values per line (R G B).
    pub fn parse_cube(text: &str) -> Result<Self, String> {
        let mut size: Option<usize> = None;
        let mut input_min = 0.0_f32;
        let mut input_max = 1.0_f32;
        let mut title = String::from("unnamed");
        let mut red = Vec::new();
        let mut green = Vec::new();
        let mut blue = Vec::new();

        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with("TITLE") {
                title = line
                    .trim_start_matches("TITLE")
                    .trim_matches('"')
                    .trim()
                    .to_string();
                continue;
            }
            if line.starts_with("LUT_1D_SIZE") {
                let tok = line
                    .split_whitespace()
                    .nth(1)
                    .ok_or_else(|| "Missing LUT_1D_SIZE value".to_string())?;
                size = Some(tok.parse::<usize>().map_err(|e| e.to_string())?);
                continue;
            }
            if line.starts_with("DOMAIN_MIN") {
                let mut parts = line.split_whitespace().skip(1);
                input_min = parts
                    .next()
                    .ok_or_else(|| "Missing DOMAIN_MIN value".to_string())?
                    .parse::<f32>()
                    .map_err(|e| e.to_string())?;
                continue;
            }
            if line.starts_with("DOMAIN_MAX") {
                let mut parts = line.split_whitespace().skip(1);
                input_max = parts
                    .next()
                    .ok_or_else(|| "Missing DOMAIN_MAX value".to_string())?
                    .parse::<f32>()
                    .map_err(|e| e.to_string())?;
                continue;
            }
            // Data line: three floats.
            let mut parts = line.split_whitespace();
            let rv = parts
                .next()
                .ok_or_else(|| format!("Bad data line: {line}"))?
                .parse::<f32>()
                .map_err(|e| e.to_string())?;
            let gv = parts
                .next()
                .ok_or_else(|| format!("Bad data line: {line}"))?
                .parse::<f32>()
                .map_err(|e| e.to_string())?;
            let bv = parts
                .next()
                .ok_or_else(|| format!("Bad data line: {line}"))?
                .parse::<f32>()
                .map_err(|e| e.to_string())?;
            red.push(rv);
            green.push(gv);
            blue.push(bv);
        }

        let n = size.ok_or_else(|| "Missing LUT_1D_SIZE keyword".to_string())?;
        if red.len() != n {
            return Err(format!("Expected {n} data rows, found {}", red.len()));
        }

        Ok(Self {
            name: title,
            size: n,
            red,
            green,
            blue,
            input_min,
            input_max,
        })
    }
}

// ─── 3-D LUT ─────────────────────────────────────────────────────────────────

/// A three-dimensional cube LUT.
///
/// The data array is indexed as `r + g * size + b * size * size`.
#[derive(Debug, Clone)]
pub struct Lut3D {
    /// Descriptive name (e.g. the filename).
    pub name: String,
    /// Cube edge dimension (e.g. 33 for a 33×33×33 LUT).
    pub size: usize,
    /// Output RGB triplets in R-fastest order.
    pub data: Vec<[f32; 3]>,
    /// Minimum input value (default `0.0`).
    pub input_min: f32,
    /// Maximum input value (default `1.0`).
    pub input_max: f32,
}

impl Lut3D {
    /// Create an identity cube LUT with the given edge size.
    pub fn new(size: usize) -> Self {
        let size = size.max(2);
        let total = size * size * size;
        let mut data = Vec::with_capacity(total);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let rv = r as f32 / (size - 1) as f32;
                    let gv = g as f32 / (size - 1) as f32;
                    let bv = b as f32 / (size - 1) as f32;
                    data.push([rv, gv, bv]);
                }
            }
        }
        Self {
            name: String::from("identity"),
            size,
            data,
            input_min: 0.0,
            input_max: 1.0,
        }
    }

    /// Retrieve a lattice node by integer (r, g, b) indices.
    fn node(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        let rc = r.min(self.size - 1);
        let gc = g.min(self.size - 1);
        let bc = b.min(self.size - 1);
        self.data[rc + gc * self.size + bc * self.size * self.size]
    }

    /// Normalise an input value to `[0, size-1]` float coordinates.
    fn normalise(&self, v: f32) -> f32 {
        let span = (self.input_max - self.input_min).max(f32::EPSILON);
        let t = ((v - self.input_min) / span).clamp(0.0, 1.0);
        t * (self.size - 1) as f32
    }

    /// Trilinear interpolation.
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let rf = self.normalise(r);
        let gf = self.normalise(g);
        let bf = self.normalise(b);

        let r0 = rf.floor() as usize;
        let g0 = gf.floor() as usize;
        let b0 = bf.floor() as usize;
        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);

        let fr = rf - r0 as f32;
        let fg = gf - g0 as f32;
        let fb = bf - b0 as f32;

        // 8 corner lattice nodes
        let c000 = self.node(r0, g0, b0);
        let c100 = self.node(r1, g0, b0);
        let c010 = self.node(r0, g1, b0);
        let c110 = self.node(r1, g1, b0);
        let c001 = self.node(r0, g0, b1);
        let c101 = self.node(r1, g0, b1);
        let c011 = self.node(r0, g1, b1);
        let c111 = self.node(r1, g1, b1);

        let interp_chan = |c: usize| {
            let c00 = c000[c] * (1.0 - fr) + c100[c] * fr;
            let c10 = c010[c] * (1.0 - fr) + c110[c] * fr;
            let c01 = c001[c] * (1.0 - fr) + c101[c] * fr;
            let c11 = c011[c] * (1.0 - fr) + c111[c] * fr;
            let c0 = c00 * (1.0 - fg) + c10 * fg;
            let c1 = c01 * (1.0 - fg) + c11 * fg;
            c0 * (1.0 - fb) + c1 * fb
        };

        (interp_chan(0), interp_chan(1), interp_chan(2))
    }

    /// Tetrahedral (Sakamoto) interpolation — higher quality than trilinear.
    ///
    /// Decomposes the unit cube into 6 tetrahedra depending on which fractional
    /// component is largest.
    pub fn apply_tetrahedral(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let rf = self.normalise(r);
        let gf = self.normalise(g);
        let bf = self.normalise(b);

        let r0 = rf.floor() as usize;
        let g0 = gf.floor() as usize;
        let b0 = bf.floor() as usize;

        let fr = rf - r0 as f32;
        let fg = gf - g0 as f32;
        let fb = bf - b0 as f32;

        let c000 = self.node(r0, g0, b0);
        let c100 = self.node((r0 + 1).min(self.size - 1), g0, b0);
        let c010 = self.node(r0, (g0 + 1).min(self.size - 1), b0);
        let c001 = self.node(r0, g0, (b0 + 1).min(self.size - 1));
        let c110 = self.node((r0 + 1).min(self.size - 1), (g0 + 1).min(self.size - 1), b0);
        let c101 = self.node((r0 + 1).min(self.size - 1), g0, (b0 + 1).min(self.size - 1));
        let c011 = self.node(r0, (g0 + 1).min(self.size - 1), (b0 + 1).min(self.size - 1));
        let c111 = self.node(
            (r0 + 1).min(self.size - 1),
            (g0 + 1).min(self.size - 1),
            (b0 + 1).min(self.size - 1),
        );

        // Tetrahedral decomposition (6 cases).
        let interp_chan = |ch: usize| {
            if fr >= fg && fg >= fb {
                // Tetrahedron 1: fr >= fg >= fb
                c000[ch] * (1.0 - fr) + c100[ch] * (fr - fg) + c110[ch] * (fg - fb) + c111[ch] * fb
            } else if fr >= fb && fb >= fg {
                // Tetrahedron 2: fr >= fb >= fg
                c000[ch] * (1.0 - fr) + c100[ch] * (fr - fb) + c101[ch] * (fb - fg) + c111[ch] * fg
            } else if fb >= fr && fr >= fg {
                // Tetrahedron 3: fb >= fr >= fg
                c000[ch] * (1.0 - fb) + c001[ch] * (fb - fr) + c101[ch] * (fr - fg) + c111[ch] * fg
            } else if fb >= fg && fg >= fr {
                // Tetrahedron 4: fb >= fg >= fr
                c000[ch] * (1.0 - fb) + c001[ch] * (fb - fg) + c011[ch] * (fg - fr) + c111[ch] * fr
            } else if fg >= fb && fb >= fr {
                // Tetrahedron 5: fg >= fb >= fr
                c000[ch] * (1.0 - fg) + c010[ch] * (fg - fb) + c011[ch] * (fb - fr) + c111[ch] * fr
            } else {
                // Tetrahedron 6: fg >= fr >= fb
                c000[ch] * (1.0 - fg) + c010[ch] * (fg - fr) + c110[ch] * (fr - fb) + c111[ch] * fb
            }
        };

        (interp_chan(0), interp_chan(1), interp_chan(2))
    }

    /// Parse an Adobe `.cube` 3-D LUT from text.
    pub fn parse_cube(text: &str) -> Result<Self, String> {
        let mut size: Option<usize> = None;
        let mut input_min = 0.0_f32;
        let mut input_max = 1.0_f32;
        let mut title = String::from("unnamed");
        let mut data: Vec<[f32; 3]> = Vec::new();

        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with("TITLE") {
                title = line
                    .trim_start_matches("TITLE")
                    .trim_matches('"')
                    .trim()
                    .to_string();
                continue;
            }
            if line.starts_with("LUT_3D_SIZE") {
                let tok = line
                    .split_whitespace()
                    .nth(1)
                    .ok_or_else(|| "Missing LUT_3D_SIZE value".to_string())?;
                size = Some(tok.parse::<usize>().map_err(|e| e.to_string())?);
                continue;
            }
            if line.starts_with("DOMAIN_MIN") {
                let mut parts = line.split_whitespace().skip(1);
                input_min = parts
                    .next()
                    .ok_or_else(|| "Missing DOMAIN_MIN value".to_string())?
                    .parse::<f32>()
                    .map_err(|e| e.to_string())?;
                continue;
            }
            if line.starts_with("DOMAIN_MAX") {
                let mut parts = line.split_whitespace().skip(1);
                input_max = parts
                    .next()
                    .ok_or_else(|| "Missing DOMAIN_MAX value".to_string())?
                    .parse::<f32>()
                    .map_err(|e| e.to_string())?;
                continue;
            }
            // Data line.
            let mut parts = line.split_whitespace();
            let rv = parts
                .next()
                .ok_or_else(|| format!("Bad data line: {line}"))?
                .parse::<f32>()
                .map_err(|e| e.to_string())?;
            let gv = parts
                .next()
                .ok_or_else(|| format!("Bad data line: {line}"))?
                .parse::<f32>()
                .map_err(|e| e.to_string())?;
            let bv = parts
                .next()
                .ok_or_else(|| format!("Bad data line: {line}"))?
                .parse::<f32>()
                .map_err(|e| e.to_string())?;
            data.push([rv, gv, bv]);
        }

        let n = size.ok_or_else(|| "Missing LUT_3D_SIZE keyword".to_string())?;
        let expected = n * n * n;
        if data.len() != expected {
            return Err(format!(
                "Expected {expected} data rows for size {n}, found {}",
                data.len()
            ));
        }

        Ok(Self {
            name: title,
            size: n,
            data,
            input_min,
            input_max,
        })
    }

    /// Factory: S-curve contrast LUT.
    ///
    /// `shadows` / `midtones` / `highlights` control the bend at 0.25, 0.5
    /// and 0.75 respectively.  All are offsets from the identity (positive =
    /// brighter in that zone).
    pub fn make_contrast(shadows: f32, midtones: f32, highlights: f32) -> Self {
        let size = 33_usize;
        let mut lut = Self::new(size);
        let s = size - 1;
        for idx in 0..lut.data.len() {
            let b = idx / (size * size);
            let g = (idx / size) % size;
            let r = idx % size;

            let apply_scurve = |v: f32| -> f32 {
                // Piecewise S-curve: 3 segments controlled by the three
                // parameters.
                let out = if v < 0.25 {
                    v + shadows * v * (0.25 - v) * 4.0
                } else if v < 0.75 {
                    v + midtones * (v - 0.25) * (0.75 - v) * 4.0
                } else {
                    v + highlights * (v - 0.75) * (1.0 - v) * 4.0
                };
                out.clamp(0.0, 1.0)
            };

            let rv = apply_scurve(r as f32 / s as f32);
            let gv = apply_scurve(g as f32 / s as f32);
            let bv = apply_scurve(b as f32 / s as f32);
            lut.data[idx] = [rv, gv, bv];
        }
        lut.name = format!("contrast_s{shadows:.2}_m{midtones:.2}_h{highlights:.2}");
        lut
    }

    /// Factory: saturation adjustment LUT.
    ///
    /// `factor` of `1.0` is identity; `0.0` converts to greyscale; `2.0`
    /// doubles saturation.
    pub fn make_saturation(factor: f32) -> Self {
        let size = 33_usize;
        let mut lut = Self::new(size);
        let s = (size - 1) as f32;
        for idx in 0..lut.data.len() {
            let b = (idx / (size * size)) as f32 / s;
            let g = ((idx / size) % size) as f32 / s;
            let r = (idx % size) as f32 / s;

            // BT.709 luminance.
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let rv = (luma + (r - luma) * factor).clamp(0.0, 1.0);
            let gv = (luma + (g - luma) * factor).clamp(0.0, 1.0);
            let bv = (luma + (b - luma) * factor).clamp(0.0, 1.0);
            lut.data[idx] = [rv, gv, bv];
        }
        lut.name = format!("saturation_{factor:.2}");
        lut
    }
}

// ─── Unified LUT enum ─────────────────────────────────────────────────────────

/// Either a 1-D or 3-D LUT.
pub enum Lut {
    /// Per-channel 1-D LUT.
    OneDimensional(Lut1D),
    /// 3-D cube LUT.
    ThreeDimensional(Lut3D),
}

impl Lut {
    /// Apply this LUT to an RGB-interleaved `f32` frame.
    ///
    /// Input pixels are expected in `[0, 1]`.  The returned buffer has the
    /// same length as `pixels`.
    pub fn apply_frame(&self, pixels: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(pixels.len());
        let chunks = pixels.chunks_exact(3);
        let remainder = chunks.remainder();
        for chunk in chunks {
            let r = chunk[0];
            let g = chunk[1];
            let b = chunk[2];
            let (ro, go, bo) = match self {
                Lut::OneDimensional(l) => l.apply(r, g, b),
                Lut::ThreeDimensional(l) => l.apply(r, g, b),
            };
            out.push(ro);
            out.push(go);
            out.push(bo);
        }
        // Pass through any trailing values that don't form a complete triplet.
        out.extend_from_slice(remainder);
        out
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1-D identity ─────────────────────────────────────────────────────────

    #[test]
    fn test_lut1d_identity_passthrough() {
        let lut = Lut1D::new(256);
        let (r, g, b) = lut.apply(0.5, 0.25, 0.75);
        assert!((r - 0.5).abs() < 0.01, "r={r}");
        assert!((g - 0.25).abs() < 0.01, "g={g}");
        assert!((b - 0.75).abs() < 0.01, "b={b}");
    }

    #[test]
    fn test_lut1d_identity_endpoints() {
        let lut = Lut1D::new(16);
        let (r0, g0, b0) = lut.apply(0.0, 0.0, 0.0);
        assert!(r0.abs() < 1e-5);
        assert!(g0.abs() < 1e-5);
        assert!(b0.abs() < 1e-5);
        let (r1, g1, b1) = lut.apply(1.0, 1.0, 1.0);
        assert!((r1 - 1.0).abs() < 1e-5);
        assert!((g1 - 1.0).abs() < 1e-5);
        assert!((b1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lut1d_known_value() {
        // A 3-entry table: [0, 0.5, 1] → input 0.5 maps to entry 1 (value 0.5).
        let lut = Lut1D::new(3); // identity, so output = input
        let (r, _, _) = lut.apply(0.5, 0.0, 0.0);
        assert!((r - 0.5).abs() < 0.01, "expected ~0.5, got {r}");
    }

    // ── 3-D identity ─────────────────────────────────────────────────────────

    #[test]
    fn test_lut3d_identity_passthrough_trilinear() {
        let lut = Lut3D::new(17);
        for (r, g, b) in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.3, 0.8)] {
            let (ro, go, bo) = lut.apply(r, g, b);
            assert!((ro - r).abs() < 0.02, "trilinear r: {ro} vs {r}");
            assert!((go - g).abs() < 0.02, "trilinear g: {go} vs {g}");
            assert!((bo - b).abs() < 0.02, "trilinear b: {bo} vs {b}");
        }
    }

    #[test]
    fn test_lut3d_identity_passthrough_tetrahedral() {
        let lut = Lut3D::new(17);
        for (r, g, b) in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.4, 0.7, 0.2)] {
            let (ro, go, bo) = lut.apply_tetrahedral(r, g, b);
            assert!((ro - r).abs() < 0.02, "tet r: {ro} vs {r}");
            assert!((go - g).abs() < 0.02, "tet g: {go} vs {g}");
            assert!((bo - b).abs() < 0.02, "tet b: {bo} vs {b}");
        }
    }

    #[test]
    fn test_trilinear_vs_tetrahedral_close() {
        // For an identity LUT, both methods should give near-identical results.
        let lut = Lut3D::new(33);
        for (r, g, b) in [(0.1, 0.6, 0.9), (0.3, 0.3, 0.3), (0.7, 0.2, 0.5)] {
            let (tr, tg, tb) = lut.apply(r, g, b);
            let (er, eg, eb) = lut.apply_tetrahedral(r, g, b);
            assert!((tr - er).abs() < 0.02, "r differs: tri={tr} tet={er}");
            assert!((tg - eg).abs() < 0.02, "g differs: tri={tg} tet={eg}");
            assert!((tb - eb).abs() < 0.02, "b differs: tri={tb} tet={eb}");
        }
    }

    // ── Cube parser — 1-D ────────────────────────────────────────────────────

    #[test]
    fn test_parse_cube_1d_identity() {
        let cube = "TITLE \"identity\"\nLUT_1D_SIZE 3\n0.0 0.0 0.0\n0.5 0.5 0.5\n1.0 1.0 1.0\n";
        let lut = Lut1D::parse_cube(cube).expect("parse should succeed");
        assert_eq!(lut.size, 3);
        let (r, g, b) = lut.apply(0.5, 0.5, 0.5);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.5).abs() < 0.01);
        assert!((b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_cube_1d_missing_keyword_errors() {
        let cube = "# no size keyword\n0.0 0.0 0.0\n";
        assert!(Lut1D::parse_cube(cube).is_err());
    }

    #[test]
    fn test_parse_cube_1d_wrong_row_count_errors() {
        let cube = "LUT_1D_SIZE 4\n0.0 0.0 0.0\n0.5 0.5 0.5\n";
        assert!(Lut1D::parse_cube(cube).is_err());
    }

    // ── Cube parser — 3-D ────────────────────────────────────────────────────

    #[test]
    fn test_parse_cube_3d_identity_2x2() {
        // 2×2×2 = 8 entries.
        let cube = concat!(
            "LUT_3D_SIZE 2\n",
            "0.0 0.0 0.0\n",
            "1.0 0.0 0.0\n",
            "0.0 1.0 0.0\n",
            "1.0 1.0 0.0\n",
            "0.0 0.0 1.0\n",
            "1.0 0.0 1.0\n",
            "0.0 1.0 1.0\n",
            "1.0 1.0 1.0\n",
        );
        let lut = Lut3D::parse_cube(cube).expect("parse should succeed");
        assert_eq!(lut.size, 2);
        assert_eq!(lut.data.len(), 8);
    }

    #[test]
    fn test_parse_cube_3d_wrong_count_errors() {
        let cube = "LUT_3D_SIZE 2\n0.0 0.0 0.0\n";
        assert!(Lut3D::parse_cube(cube).is_err());
    }

    // ── apply_frame ──────────────────────────────────────────────────────────

    #[test]
    fn test_apply_frame_identity_1d() {
        let lut = Lut::OneDimensional(Lut1D::new(256));
        let pixels: Vec<f32> = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let out = lut.apply_frame(&pixels);
        assert_eq!(out.len(), pixels.len());
        for (o, i) in out.iter().zip(pixels.iter()) {
            assert!((o - i).abs() < 0.01, "expected ~{i}, got {o}");
        }
    }

    #[test]
    fn test_apply_frame_identity_3d() {
        let lut = Lut::ThreeDimensional(Lut3D::new(17));
        let pixels: Vec<f32> = vec![0.2, 0.4, 0.6, 0.8, 0.1, 0.9];
        let out = lut.apply_frame(&pixels);
        assert_eq!(out.len(), pixels.len());
        for (o, i) in out.iter().zip(pixels.iter()) {
            assert!((o - i).abs() < 0.02, "expected ~{i}, got {o}");
        }
    }

    // ── Factory LUTs ─────────────────────────────────────────────────────────

    #[test]
    fn test_make_contrast_increases_midtones() {
        let lut = Lut3D::make_contrast(0.0, 0.5, 0.0);
        // A midpoint (0.5, 0.5, 0.5) should be brighter.
        let (r, g, b) = lut.apply(0.5, 0.5, 0.5);
        assert!(r > 0.5, "midtones should be pushed up: r={r}");
        assert!(g > 0.5, "midtones should be pushed up: g={g}");
        assert!(b > 0.5, "midtones should be pushed up: b={b}");
    }

    #[test]
    fn test_make_saturation_zero_gives_grey() {
        let lut = Lut3D::make_saturation(0.0);
        // A saturated red should become grey.
        let (r, g, b) = lut.apply(1.0, 0.0, 0.0);
        // Luma of pure red ≈ 0.2126.
        assert!((r - g).abs() < 0.01, "desaturated: r={r} g={g}");
        assert!((g - b).abs() < 0.01, "desaturated: g={g} b={b}");
    }

    #[test]
    fn test_make_saturation_identity() {
        let lut = Lut3D::make_saturation(1.0);
        let (r, g, b) = lut.apply(0.6, 0.3, 0.9);
        assert!((r - 0.6).abs() < 0.05, "r={r}");
        assert!((g - 0.3).abs() < 0.05, "g={g}");
        assert!((b - 0.9).abs() < 0.05, "b={b}");
    }
}
