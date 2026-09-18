#![allow(dead_code)]
//! Edge detection algorithms for image analysis.
//!
//! Implements classic edge-detection operators used in computer vision,
//! VFX compositing, and quality-control workflows:
//!
//! - **Sobel** - 3x3 gradient operator for robust edge detection
//! - **Prewitt** - Simpler 3x3 gradient operator
//! - **Laplacian** - Second-derivative zero-crossing detector
//! - **Roberts Cross** - 2x2 diagonal gradient operator
//! - **Scharr** - Improved Sobel with better rotational symmetry
//! - **Canny-style** - Non-maximum suppression with hysteresis thresholds

/// Edge detection algorithm to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeOperator {
    /// Sobel 3x3 operator.
    Sobel,
    /// Prewitt 3x3 operator.
    Prewitt,
    /// Laplacian 3x3 operator.
    Laplacian,
    /// Roberts Cross 2x2 diagonal operator.
    RobertsCross,
    /// Scharr 3x3 operator (improved Sobel).
    Scharr,
}

/// Configuration for edge detection.
#[derive(Clone, Debug)]
pub struct EdgeDetectConfig {
    /// The operator to use.
    pub operator: EdgeOperator,
    /// Low threshold for hysteresis (0.0-1.0).
    pub threshold_low: f64,
    /// High threshold for hysteresis (0.0-1.0).
    pub threshold_high: f64,
    /// Whether to normalize the output to `[0, 1]`.
    pub normalize_output: bool,
}

impl Default for EdgeDetectConfig {
    fn default() -> Self {
        Self {
            operator: EdgeOperator::Sobel,
            threshold_low: 0.1,
            threshold_high: 0.3,
            normalize_output: true,
        }
    }
}

impl EdgeDetectConfig {
    /// Create a new config with the given operator.
    pub fn new(operator: EdgeOperator) -> Self {
        Self {
            operator,
            ..Default::default()
        }
    }

    /// Set the thresholds for hysteresis edge detection.
    pub fn with_thresholds(mut self, low: f64, high: f64) -> Self {
        self.threshold_low = low.clamp(0.0, 1.0);
        self.threshold_high = high.clamp(0.0, 1.0);
        self
    }

    /// Set whether to normalize output.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize_output = normalize;
        self
    }
}

/// A grayscale image buffer for edge detection operations.
#[derive(Clone, Debug)]
pub struct GrayImage {
    /// Pixel data in row-major order, values in `[0.0, 1.0]`.
    pub data: Vec<f64>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl GrayImage {
    /// Create a new grayscale image filled with zeros.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0.0; (width as usize) * (height as usize)],
            width,
            height,
        }
    }

    /// Create from existing data.
    pub fn from_data(width: u32, height: u32, data: Vec<f64>) -> Option<Self> {
        if data.len() == (width as usize) * (height as usize) {
            Some(Self {
                data,
                width,
                height,
            })
        } else {
            None
        }
    }

    /// Get pixel value at (x, y) with clamped boundary handling.
    pub fn get_clamped(&self, x: i32, y: i32) -> f64 {
        let cx = x.clamp(0, self.width as i32 - 1) as usize;
        let cy = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[cy * (self.width as usize) + cx]
    }

    /// Set pixel value at (x, y).
    pub fn set(&mut self, x: u32, y: u32, value: f64) {
        if x < self.width && y < self.height {
            self.data[(y as usize) * (self.width as usize) + (x as usize)] = value;
        }
    }

    /// Get pixel value at (x, y), returning None if out of bounds.
    pub fn get(&self, x: u32, y: u32) -> Option<f64> {
        if x < self.width && y < self.height {
            Some(self.data[(y as usize) * (self.width as usize) + (x as usize)])
        } else {
            None
        }
    }

    /// Return the total number of pixels.
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// Apply a 3x3 convolution kernel to a grayscale image.
fn convolve_3x3(image: &GrayImage, kernel: &[f64; 9]) -> GrayImage {
    let mut output = GrayImage::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let xi = x as i32;
            let yi = y as i32;
            let mut sum = 0.0;
            for ky in -1..=1_i32 {
                for kx in -1..=1_i32 {
                    let ki = ((ky + 1) * 3 + (kx + 1)) as usize;
                    sum += image.get_clamped(xi + kx, yi + ky) * kernel[ki];
                }
            }
            output.set(x, y, sum);
        }
    }
    output
}

/// Compute the gradient magnitude from horizontal and vertical components.
fn gradient_magnitude(gx: &GrayImage, gy: &GrayImage) -> GrayImage {
    let mut output = GrayImage::new(gx.width, gx.height);
    for i in 0..gx.data.len() {
        let mag = (gx.data[i].powi(2) + gy.data[i].powi(2)).sqrt();
        output.data[i] = mag;
    }
    output
}

/// Compute the gradient direction in radians from horizontal and vertical components.
fn gradient_direction(gx: &GrayImage, gy: &GrayImage) -> Vec<f64> {
    gx.data
        .iter()
        .zip(gy.data.iter())
        .map(|(&x, &y)| y.atan2(x))
        .collect()
}

/// Normalize a grayscale image to `[0.0, 1.0]`.
fn normalize_image(image: &mut GrayImage) {
    let min = image.data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = image.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range > f64::EPSILON {
        for v in &mut image.data {
            *v = (*v - min) / range;
        }
    }
}

/// Run edge detection on a grayscale image.
pub fn detect_edges(image: &GrayImage, config: &EdgeDetectConfig) -> GrayImage {
    let mut result = match config.operator {
        EdgeOperator::Sobel => {
            let kx: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
            let ky: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
            let gx = convolve_3x3(image, &kx);
            let gy = convolve_3x3(image, &ky);
            gradient_magnitude(&gx, &gy)
        }
        EdgeOperator::Prewitt => {
            let kx: [f64; 9] = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
            let ky: [f64; 9] = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            let gx = convolve_3x3(image, &kx);
            let gy = convolve_3x3(image, &ky);
            gradient_magnitude(&gx, &gy)
        }
        EdgeOperator::Scharr => {
            let kx: [f64; 9] = [-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0];
            let ky: [f64; 9] = [-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0];
            let gx = convolve_3x3(image, &kx);
            let gy = convolve_3x3(image, &ky);
            gradient_magnitude(&gx, &gy)
        }
        EdgeOperator::Laplacian => {
            let k: [f64; 9] = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
            let mut result = convolve_3x3(image, &k);
            // Take absolute value for Laplacian
            for v in &mut result.data {
                *v = v.abs();
            }
            result
        }
        EdgeOperator::RobertsCross => roberts_cross(image),
    };

    if config.normalize_output {
        normalize_image(&mut result);
    }

    result
}

/// Roberts Cross 2x2 diagonal gradient operator.
fn roberts_cross(image: &GrayImage) -> GrayImage {
    let mut output = GrayImage::new(image.width, image.height);
    for y in 0..image.height.saturating_sub(1) {
        for x in 0..image.width.saturating_sub(1) {
            let xi = x as i32;
            let yi = y as i32;
            let p00 = image.get_clamped(xi, yi);
            let p01 = image.get_clamped(xi + 1, yi);
            let p10 = image.get_clamped(xi, yi + 1);
            let p11 = image.get_clamped(xi + 1, yi + 1);
            let gx = p00 - p11;
            let gy = p01 - p10;
            output.set(x, y, (gx.powi(2) + gy.powi(2)).sqrt());
        }
    }
    output
}

/// Apply non-maximum suppression to thin edges.
pub fn non_maximum_suppression(magnitude: &GrayImage, gx: &GrayImage, gy: &GrayImage) -> GrayImage {
    let mut output = GrayImage::new(magnitude.width, magnitude.height);
    let directions = gradient_direction(gx, gy);

    for y in 1..magnitude.height.saturating_sub(1) {
        for x in 1..magnitude.width.saturating_sub(1) {
            let idx = (y as usize) * (magnitude.width as usize) + (x as usize);
            let angle = directions[idx];
            let xi = x as i32;
            let yi = y as i32;

            // Quantize angle to 4 directions
            let angle_deg = angle.to_degrees().rem_euclid(180.0);
            let (n1, n2) = if !(22.5..157.5).contains(&angle_deg) {
                // Horizontal edge -> compare with left/right
                (
                    magnitude.get_clamped(xi - 1, yi),
                    magnitude.get_clamped(xi + 1, yi),
                )
            } else if angle_deg < 67.5 {
                // 45-degree edge
                (
                    magnitude.get_clamped(xi - 1, yi - 1),
                    magnitude.get_clamped(xi + 1, yi + 1),
                )
            } else if angle_deg < 112.5 {
                // Vertical edge -> compare with above/below
                (
                    magnitude.get_clamped(xi, yi - 1),
                    magnitude.get_clamped(xi, yi + 1),
                )
            } else {
                // 135-degree edge
                (
                    magnitude.get_clamped(xi + 1, yi - 1),
                    magnitude.get_clamped(xi - 1, yi + 1),
                )
            };

            let current = magnitude.data[idx];
            if current >= n1 && current >= n2 {
                output.set(x, y, current);
            }
        }
    }
    output
}

/// Apply hysteresis thresholding to an edge magnitude image.
pub fn hysteresis_threshold(image: &GrayImage, low: f64, high: f64) -> GrayImage {
    let mut output = GrayImage::new(image.width, image.height);

    // First pass: mark strong and weak edges
    for y in 0..image.height {
        for x in 0..image.width {
            let val = image.get_clamped(x as i32, y as i32);
            if val >= high {
                output.set(x, y, 1.0);
            } else if val >= low {
                output.set(x, y, 0.5); // weak edge candidate
            }
        }
    }

    // Second pass: promote weak edges connected to strong edges
    let mut changed = true;
    while changed {
        changed = false;
        for y in 1..output.height.saturating_sub(1) {
            for x in 1..output.width.saturating_sub(1) {
                let val = output.get_clamped(x as i32, y as i32);
                if (val - 0.5).abs() < f64::EPSILON {
                    // Check 8-neighbors for strong edges
                    let has_strong_neighbor = (-1..=1_i32).any(|dy| {
                        (-1..=1_i32).any(|dx| {
                            if dx == 0 && dy == 0 {
                                return false;
                            }
                            let nv = output.get_clamped(x as i32 + dx, y as i32 + dy);
                            (nv - 1.0).abs() < f64::EPSILON
                        })
                    });
                    if has_strong_neighbor {
                        output.set(x, y, 1.0);
                        changed = true;
                    }
                }
            }
        }
    }

    // Final pass: remove remaining weak edges
    for v in &mut output.data {
        if (*v - 1.0).abs() > f64::EPSILON {
            *v = 0.0;
        }
    }

    output
}

/// Compute edge density (ratio of edge pixels to total pixels).
#[allow(clippy::cast_precision_loss)]
pub fn edge_density(edge_image: &GrayImage, threshold: f64) -> f64 {
    let total = edge_image.data.len();
    if total == 0 {
        return 0.0;
    }
    let edge_count = edge_image.data.iter().filter(|&&v| v >= threshold).count();
    edge_count as f64 / total as f64
}

/// Edge strength statistics.
#[derive(Clone, Debug)]
pub struct EdgeStats {
    /// Mean edge magnitude.
    pub mean_magnitude: f64,
    /// Maximum edge magnitude.
    pub max_magnitude: f64,
    /// Ratio of edge pixels above threshold.
    pub edge_ratio: f64,
    /// Total number of pixels.
    pub total_pixels: usize,
}

impl EdgeStats {
    /// Compute edge statistics from a magnitude image.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_magnitude(mag: &GrayImage, threshold: f64) -> Self {
        let total = mag.data.len();
        if total == 0 {
            return Self {
                mean_magnitude: 0.0,
                max_magnitude: 0.0,
                edge_ratio: 0.0,
                total_pixels: 0,
            };
        }
        let sum: f64 = mag.data.iter().sum();
        let max = mag.data.iter().copied().fold(0.0_f64, f64::max);
        let edges = mag.data.iter().filter(|&&v| v >= threshold).count();
        Self {
            mean_magnitude: sum / total as f64,
            max_magnitude: max,
            edge_ratio: edges as f64 / total as f64,
            total_pixels: total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image() -> GrayImage {
        // 5x5 image with a vertical edge in the middle
        let mut img = GrayImage::new(5, 5);
        for y in 0..5 {
            for x in 0..5 {
                let val = if x < 2 { 0.0 } else { 1.0 };
                img.set(x, y, val);
            }
        }
        img
    }

    #[test]
    fn test_gray_image_new() {
        let img = GrayImage::new(10, 10);
        assert_eq!(img.width, 10);
        assert_eq!(img.height, 10);
        assert_eq!(img.pixel_count(), 100);
    }

    #[test]
    fn test_gray_image_from_data() {
        let data = vec![0.5; 12];
        let img = GrayImage::from_data(4, 3, data);
        assert!(img.is_some());
        let img = img.expect("should succeed in test");
        assert_eq!(img.width, 4);

        // Wrong size
        let data = vec![0.5; 10];
        assert!(GrayImage::from_data(4, 3, data).is_none());
    }

    #[test]
    fn test_gray_image_clamped_access() {
        let mut img = GrayImage::new(4, 4);
        img.set(2, 2, 0.8);
        assert!((img.get_clamped(2, 2) - 0.8).abs() < f64::EPSILON);
        // Out of bounds clamps
        assert!((img.get_clamped(-1, 0) - img.get_clamped(0, 0)).abs() < f64::EPSILON);
        assert!((img.get_clamped(10, 0) - img.get_clamped(3, 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sobel_edge_detection() {
        let img = make_test_image();
        let config = EdgeDetectConfig::new(EdgeOperator::Sobel);
        let edges = detect_edges(&img, &config);
        assert_eq!(edges.width, 5);
        assert_eq!(edges.height, 5);
        // The edge column (x=2) should have high values
        let edge_val = edges.get(2, 2).expect("should succeed in test");
        let flat_val = edges.get(0, 2).expect("should succeed in test");
        assert!(
            edge_val > flat_val,
            "Edge pixel should be stronger than flat area"
        );
    }

    #[test]
    fn test_prewitt_edge_detection() {
        let img = make_test_image();
        let config = EdgeDetectConfig::new(EdgeOperator::Prewitt);
        let edges = detect_edges(&img, &config);
        let edge_val = edges.get(2, 2).expect("should succeed in test");
        assert!(edge_val > 0.0, "Prewitt should detect the edge");
    }

    #[test]
    fn test_laplacian_edge_detection() {
        let img = make_test_image();
        let config = EdgeDetectConfig::new(EdgeOperator::Laplacian);
        let edges = detect_edges(&img, &config);
        let edge_val = edges.get(2, 2).expect("should succeed in test");
        assert!(edge_val > 0.0, "Laplacian should detect the edge");
    }

    #[test]
    fn test_scharr_edge_detection() {
        let img = make_test_image();
        let config = EdgeDetectConfig::new(EdgeOperator::Scharr);
        let edges = detect_edges(&img, &config);
        let edge_val = edges.get(2, 2).expect("should succeed in test");
        assert!(edge_val > 0.0, "Scharr should detect the edge");
    }

    #[test]
    fn test_roberts_cross_edge_detection() {
        let img = make_test_image();
        let config = EdgeDetectConfig::new(EdgeOperator::RobertsCross);
        let edges = detect_edges(&img, &config);
        // Roberts uses 2x2, so edge detected near transition
        let edge_val = edges.get(1, 1).expect("should succeed in test");
        assert!(edge_val > 0.0, "Roberts should detect edge transition");
    }

    #[test]
    fn test_hysteresis_threshold() {
        let mut img = GrayImage::new(5, 5);
        // Strong edge in center
        img.set(2, 2, 0.9);
        // Weak edges adjacent
        img.set(2, 1, 0.2);
        img.set(2, 3, 0.2);
        // Below low threshold elsewhere
        let result = hysteresis_threshold(&img, 0.15, 0.5);
        // Strong edge should remain
        assert!((result.get(2, 2).expect("should succeed in test") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_density() {
        let mut img = GrayImage::new(10, 10);
        // Set 25 pixels above threshold
        for i in 0..25 {
            img.data[i] = 0.8;
        }
        let density = edge_density(&img, 0.5);
        assert!((density - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_stats() {
        let mut img = GrayImage::new(4, 4);
        img.data = vec![
            0.0, 0.1, 0.5, 0.8, 0.0, 0.2, 0.6, 0.9, 0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.1, 0.4,
        ];
        let stats = EdgeStats::from_magnitude(&img, 0.5);
        assert!((stats.max_magnitude - 0.9).abs() < f64::EPSILON);
        assert!(stats.edge_ratio > 0.0);
        assert_eq!(stats.total_pixels, 16);
    }

    #[test]
    fn test_edge_config_builder() {
        let config = EdgeDetectConfig::new(EdgeOperator::Sobel)
            .with_thresholds(0.05, 0.2)
            .with_normalize(false);
        assert!((config.threshold_low - 0.05).abs() < f64::EPSILON);
        assert!((config.threshold_high - 0.2).abs() < f64::EPSILON);
        assert!(!config.normalize_output);
    }

    #[test]
    fn test_uniform_image_no_edges() {
        let img = GrayImage {
            data: vec![0.5; 25],
            width: 5,
            height: 5,
        };
        let config = EdgeDetectConfig::new(EdgeOperator::Sobel);
        let edges = detect_edges(&img, &config);
        // Uniform image should have no edges
        for v in &edges.data {
            assert!(*v < f64::EPSILON, "Uniform image should have zero edges");
        }
    }

    #[test]
    fn test_non_maximum_suppression() {
        let mut mag = GrayImage::new(5, 5);
        let mut gx = GrayImage::new(5, 5);
        let gy = GrayImage::new(5, 5);
        // Create a horizontal gradient line
        mag.set(1, 2, 0.3);
        mag.set(2, 2, 0.8);
        mag.set(3, 2, 0.3);
        gx.set(2, 2, 0.8);
        let result = non_maximum_suppression(&mag, &gx, &gy);
        // Center should survive as local max
        assert!(result.get(2, 2).expect("should succeed in test") > 0.0);
    }
}
