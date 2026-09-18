//! Test chart analysis for color calibration.
//!
//! Provides analysis of standard test charts such as X-Rite `ColorChecker`,
//! DSC resolution charts, grey scale charts, and more.

#![allow(dead_code)]

/// Type of test chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestChart {
    /// X-Rite `ColorChecker` Classic (24 patches).
    ColorChecker24,
    /// X-Rite `ColorChecker` Passport (30 patches).
    ColorChecker30,
    /// DSC resolution test chart.
    Dsc,
    /// Resolution chart.
    Resolution,
    /// Grey scale chart.
    GreyScale,
    /// X-Rite chart.
    Xrite,
}

impl TestChart {
    /// Returns the number of color patches in this chart.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        match self {
            Self::ColorChecker24 => 24,
            Self::ColorChecker30 => 30,
            Self::Dsc => 12,
            Self::Resolution => 0,
            Self::GreyScale => 10,
            Self::Xrite => 24,
        }
    }

    /// Returns the grid layout (cols, rows) for this chart.
    #[must_use]
    pub fn grid_layout(&self) -> (usize, usize) {
        match self {
            Self::ColorChecker24 => (6, 4),
            Self::ColorChecker30 => (6, 5),
            Self::Dsc => (4, 3),
            Self::Resolution => (1, 1),
            Self::GreyScale => (10, 1),
            Self::Xrite => (6, 4),
        }
    }
}

/// A single color patch with reference values.
#[derive(Debug, Clone)]
pub struct ColorPatch {
    /// Human-readable name.
    pub name: String,
    /// Expected sRGB values (0.0–1.0).
    pub expected_rgb: [f32; 3],
    /// Expected CIELAB values (L*, a*, b*).
    pub expected_lab: [f32; 3],
}

/// Returns the standard `ColorChecker` 24 patch reference data.
/// Only the first 6 patches are fully defined; the remaining 18 are stubs.
#[must_use]
pub fn colorchecker_patches() -> Vec<ColorPatch> {
    let mut patches = vec![
        // Row 1 – natural colors
        ColorPatch {
            name: "Dark Skin".to_string(),
            expected_rgb: [0.400, 0.267, 0.200],
            expected_lab: [37.99, 13.56, 14.06],
        },
        ColorPatch {
            name: "Light Skin".to_string(),
            expected_rgb: [0.769, 0.588, 0.459],
            expected_lab: [65.71, 18.13, 17.81],
        },
        ColorPatch {
            name: "Blue Sky".to_string(),
            expected_rgb: [0.259, 0.404, 0.620],
            expected_lab: [49.93, -4.88, -21.93],
        },
        ColorPatch {
            name: "Foliage".to_string(),
            expected_rgb: [0.271, 0.373, 0.184],
            expected_lab: [43.14, -13.10, 21.91],
        },
        ColorPatch {
            name: "Blue Flower".to_string(),
            expected_rgb: [0.443, 0.478, 0.690],
            expected_lab: [55.11, 8.84, -25.40],
        },
        ColorPatch {
            name: "Bluish Green".to_string(),
            expected_rgb: [0.243, 0.690, 0.647],
            expected_lab: [70.72, -33.40, -0.20],
        },
    ];

    // Stubs for patches 7–24
    let stub_names = [
        "Orange",
        "Purplish Blue",
        "Moderate Red",
        "Purple",
        "Yellow Green",
        "Orange Yellow",
        "Blue",
        "Green",
        "Red",
        "Yellow",
        "Magenta",
        "Cyan",
        "White 9.5",
        "Neutral 8",
        "Neutral 6.5",
        "Neutral 5",
        "Neutral 3.5",
        "Black 2",
    ];

    for name in &stub_names {
        patches.push(ColorPatch {
            name: (*name).to_string(),
            expected_rgb: [0.5, 0.5, 0.5],
            expected_lab: [50.0, 0.0, 0.0],
        });
    }

    patches
}

/// A measured patch pairing a reference `ColorPatch` with its measured values.
#[derive(Debug, Clone)]
pub struct MeasuredPatch {
    /// Reference color patch.
    pub patch: ColorPatch,
    /// Measured sRGB values (0.0–1.0).
    pub measured_rgb: [f32; 3],
    /// ΔE76 error between reference and measured LAB.
    pub delta_e: f32,
}

/// Result of analysing an entire test chart.
#[derive(Debug, Clone)]
pub struct ChartAnalysis {
    /// Per-patch measurements.
    pub patches: Vec<MeasuredPatch>,
    /// Mean ΔE76 across all patches.
    pub delta_e_mean: f32,
    /// Maximum ΔE76 across all patches.
    pub delta_e_max: f32,
}

impl ChartAnalysis {
    /// Derive the accuracy grade from the mean ΔE.
    #[must_use]
    pub fn grade(&self) -> AccuracyGrade {
        AccuracyGrade::from_delta_e(self.delta_e_mean)
    }
}

/// Overall accuracy grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccuracyGrade {
    /// Mean ΔE < 1 – imperceptible difference.
    Excellent,
    /// Mean ΔE 1–2 – just perceptible.
    Good,
    /// Mean ΔE 2–4 – noticeable difference.
    Acceptable,
    /// Mean ΔE ≥ 4 – significant difference.
    Poor,
}

impl AccuracyGrade {
    /// Determine grade from a mean ΔE value.
    #[must_use]
    pub fn from_delta_e(mean_de: f32) -> Self {
        if mean_de < 1.0 {
            Self::Excellent
        } else if mean_de < 2.0 {
            Self::Good
        } else if mean_de < 4.0 {
            Self::Acceptable
        } else {
            Self::Poor
        }
    }
}

/// Analyses a test chart image and returns per-patch measurements.
pub struct ChartAnalyzer;

impl ChartAnalyzer {
    /// Analyse a float pixel buffer (sRGB, linear memory, R/G/B interleaved).
    ///
    /// Patches are sampled at the centres of the normalised grid cells.
    /// ΔE76 is computed in CIELAB.
    ///
    /// # Arguments
    /// * `chart_image` – slice of `f32` pixels (R, G, B triples).
    /// * `width` / `height` – image dimensions in pixels.
    /// * `chart_type` – which chart to expect.
    #[must_use]
    pub fn analyze(
        chart_image: &[f32],
        width: u32,
        height: u32,
        chart_type: TestChart,
    ) -> ChartAnalysis {
        let refs = colorchecker_patches();
        let count = chart_type.patch_count().min(refs.len());
        let (cols, rows) = chart_type.grid_layout();

        let mut measured_patches = Vec::with_capacity(count);

        for i in 0..count {
            let col = i % cols.max(1);
            let row = i / cols.max(1);

            // Normalised centre of each grid cell
            let nx = (col as f32 + 0.5) / rows.max(1) as f32;
            let ny = (row as f32 + 0.5) / rows.max(1) as f32;

            let measured_rgb = sample_pixel(chart_image, width, height, nx, ny);
            let measured_lab = srgb_to_lab(measured_rgb);
            let ref_patch = &refs[i];
            let de = delta_e76(ref_patch.expected_lab, measured_lab);

            measured_patches.push(MeasuredPatch {
                patch: ref_patch.clone(),
                measured_rgb,
                delta_e: de,
            });
        }

        let (mean, max) = if measured_patches.is_empty() {
            (0.0, 0.0)
        } else {
            let sum: f32 = measured_patches.iter().map(|p| p.delta_e).sum();
            let max = measured_patches
                .iter()
                .map(|p| p.delta_e)
                .fold(0.0f32, f32::max);
            (sum / measured_patches.len() as f32, max)
        };

        ChartAnalysis {
            patches: measured_patches,
            delta_e_mean: mean,
            delta_e_max: max,
        }
    }
}

// ── internal helpers ─────────────────────────────────────────────────────────

/// Sample the nearest pixel at normalised coordinates (0–1, 0–1).
fn sample_pixel(image: &[f32], width: u32, height: u32, nx: f32, ny: f32) -> [f32; 3] {
    if image.is_empty() || width == 0 || height == 0 {
        return [0.0, 0.0, 0.0];
    }
    let px = ((nx * width as f32) as u32).min(width - 1);
    let py = ((ny * height as f32) as u32).min(height - 1);
    let idx = ((py * width + px) * 3) as usize;
    if idx + 2 < image.len() {
        [image[idx], image[idx + 1], image[idx + 2]]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Convert sRGB (0–1) to CIELAB (D65 reference white).
fn srgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    // Linearise
    let linear: [f32; 3] = rgb.map(|c| {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    });

    // sRGB → XYZ (D65)
    let x = linear[0] * 0.4124 + linear[1] * 0.3576 + linear[2] * 0.1805;
    let y = linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
    let z = linear[0] * 0.0193 + linear[1] * 0.1192 + linear[2] * 0.9505;

    // D65 reference white
    let xn = 0.950_47f32;
    let yn = 1.000_00f32;
    let zn = 1.088_83f32;

    let fx = lab_f(x / xn);
    let fy = lab_f(y / yn);
    let fz = lab_f(z / zn);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    [l, a, b]
}

fn lab_f(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    if t > DELTA.powi(3) {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// Compute ΔE76 between two CIELAB values.
fn delta_e76(lab1: [f32; 3], lab2: [f32; 3]) -> f32 {
    let dl = lab1[0] - lab2[0];
    let da = lab1[1] - lab2[1];
    let db = lab1[2] - lab2[2];
    (dl * dl + da * da + db * db).sqrt()
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chart_patch_counts() {
        assert_eq!(TestChart::ColorChecker24.patch_count(), 24);
        assert_eq!(TestChart::ColorChecker30.patch_count(), 30);
        assert_eq!(TestChart::Dsc.patch_count(), 12);
        assert_eq!(TestChart::Resolution.patch_count(), 0);
        assert_eq!(TestChart::GreyScale.patch_count(), 10);
        assert_eq!(TestChart::Xrite.patch_count(), 24);
    }

    #[test]
    fn test_colorchecker_patches_count() {
        let patches = colorchecker_patches();
        assert_eq!(patches.len(), 24);
    }

    #[test]
    fn test_colorchecker_first_patch_name() {
        let patches = colorchecker_patches();
        assert_eq!(patches[0].name, "Dark Skin");
    }

    #[test]
    fn test_colorchecker_blue_sky() {
        let patches = colorchecker_patches();
        assert_eq!(patches[2].name, "Blue Sky");
        // L* should be around 50
        assert!((patches[2].expected_lab[0] - 49.93).abs() < 0.1);
    }

    #[test]
    fn test_accuracy_grade_excellent() {
        assert_eq!(AccuracyGrade::from_delta_e(0.5), AccuracyGrade::Excellent);
    }

    #[test]
    fn test_accuracy_grade_good() {
        assert_eq!(AccuracyGrade::from_delta_e(1.5), AccuracyGrade::Good);
    }

    #[test]
    fn test_accuracy_grade_acceptable() {
        assert_eq!(AccuracyGrade::from_delta_e(3.0), AccuracyGrade::Acceptable);
    }

    #[test]
    fn test_accuracy_grade_poor() {
        assert_eq!(AccuracyGrade::from_delta_e(5.0), AccuracyGrade::Poor);
    }

    #[test]
    fn test_accuracy_grade_boundary_1() {
        assert_eq!(AccuracyGrade::from_delta_e(1.0), AccuracyGrade::Good);
        assert_eq!(AccuracyGrade::from_delta_e(0.999), AccuracyGrade::Excellent);
    }

    #[test]
    fn test_accuracy_grade_boundary_4() {
        assert_eq!(AccuracyGrade::from_delta_e(4.0), AccuracyGrade::Poor);
        assert_eq!(
            AccuracyGrade::from_delta_e(3.999),
            AccuracyGrade::Acceptable
        );
    }

    #[test]
    fn test_delta_e76_identical() {
        let lab = [50.0, 10.0, -10.0];
        assert!((delta_e76(lab, lab) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_delta_e76_simple() {
        let a = [50.0, 0.0, 0.0];
        let b = [53.0, 4.0, 0.0];
        let expected = (9.0f32 + 16.0f32).sqrt();
        assert!((delta_e76(a, b) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_analyze_empty_chart() {
        // Resolution chart has 0 patches → empty analysis
        let image = vec![0.5f32; 100 * 100 * 3];
        let analysis = ChartAnalyzer::analyze(&image, 100, 100, TestChart::Resolution);
        assert_eq!(analysis.patches.len(), 0);
        assert!((analysis.delta_e_mean - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_analyze_grey_image_returns_patches() {
        // A flat grey image; we only care that we get a result, not a specific ΔE
        let image = vec![0.5f32; 200 * 200 * 3];
        let analysis = ChartAnalyzer::analyze(&image, 200, 200, TestChart::GreyScale);
        assert_eq!(analysis.patches.len(), TestChart::GreyScale.patch_count());
        assert!(analysis.delta_e_mean >= 0.0);
    }

    #[test]
    fn test_chart_analysis_grade() {
        let analysis = ChartAnalysis {
            patches: vec![],
            delta_e_mean: 0.8,
            delta_e_max: 1.2,
        };
        assert_eq!(analysis.grade(), AccuracyGrade::Excellent);
    }

    #[test]
    fn test_srgb_to_lab_white() {
        let lab = srgb_to_lab([1.0, 1.0, 1.0]);
        // L* for white should be ~100
        assert!((lab[0] - 100.0).abs() < 1.0, "L* = {}", lab[0]);
    }

    #[test]
    fn test_srgb_to_lab_black() {
        let lab = srgb_to_lab([0.0, 0.0, 0.0]);
        assert!((lab[0] - 0.0).abs() < 0.1, "L* = {}", lab[0]);
    }
}
