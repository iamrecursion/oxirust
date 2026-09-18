//! Frame warping and transformation for video stabilization.
//!
//! This module provides algorithms for applying perspective transformations to video frames:
//!
//! - Perspective transformation
//! - Border handling (crop, replicate, reflect)
//! - Temporal interpolation for missing regions
//! - Bilinear interpolation for smooth warping

use crate::error::{CvError, CvResult};
use crate::stabilize::motion::TransformMatrix;
use bytes::Bytes;
use oximedia_codec::VideoFrame;

/// Border handling mode for frame warping.
///
/// Defines how to handle pixels outside the frame boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderMode {
    /// Crop the frame to avoid black borders.
    Crop,
    /// Replicate edge pixels.
    #[default]
    Replicate,
    /// Reflect pixels at the border.
    Reflect,
    /// Use constant border value (typically black).
    Constant,
    /// Wrap around to the opposite side.
    Wrap,
}

/// Frame warper for applying transformations.
///
/// Warps video frames using perspective transformations.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::FrameWarper;
///
/// let warper = FrameWarper::new();
/// ```
#[derive(Debug, Clone)]
pub struct FrameWarper {
    /// Use bilinear interpolation.
    use_bilinear: bool,
    /// Temporal smoothing window.
    temporal_window: usize,
}

impl FrameWarper {
    /// Create a new frame warper.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::FrameWarper;
    ///
    /// let warper = FrameWarper::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            use_bilinear: true,
            temporal_window: 3,
        }
    }

    /// Set whether to use bilinear interpolation.
    #[must_use]
    pub const fn with_bilinear(mut self, use_bilinear: bool) -> Self {
        self.use_bilinear = use_bilinear;
        self
    }

    /// Set temporal smoothing window.
    #[must_use]
    pub const fn with_temporal_window(mut self, window: usize) -> Self {
        self.temporal_window = window;
        self
    }

    /// Warp a video frame using a transformation matrix.
    ///
    /// # Arguments
    ///
    /// * `frame` - Input video frame
    /// * `transform` - Transformation to apply
    /// * `border_mode` - How to handle borders
    /// * `crop_ratio` - Ratio of frame to keep after warping
    ///
    /// # Errors
    ///
    /// Returns an error if warping fails.
    pub fn warp(
        &self,
        frame: &VideoFrame,
        transform: &TransformMatrix,
        border_mode: BorderMode,
        crop_ratio: f64,
    ) -> CvResult<VideoFrame> {
        // Create output frame
        let mut output = VideoFrame::new(frame.format, frame.width, frame.height);
        output.allocate();
        output.timestamp = frame.timestamp;
        output.frame_type = frame.frame_type;
        output.color_info = frame.color_info;

        // Compute inverse transformation for backward warping
        let inv_transform = transform.invert();

        // Compute cropping bounds
        let crop_width = (frame.width as f64 * crop_ratio) as u32;
        let crop_height = (frame.height as f64 * crop_ratio) as u32;
        let crop_x = (frame.width - crop_width) / 2;
        let crop_y = (frame.height - crop_height) / 2;

        // Warp each plane
        for plane_idx in 0..frame.planes.len() {
            let (plane_width, plane_height) = frame.plane_dimensions(plane_idx);

            // Get plane data
            let input_plane = &frame.planes[plane_idx];
            let output_plane = &mut output.planes[plane_idx];

            self.warp_plane(
                &input_plane.data,
                output_plane,
                plane_width,
                plane_height,
                &inv_transform,
                border_mode,
                crop_x,
                crop_y,
                crop_width,
                crop_height,
            )?;
        }

        Ok(output)
    }

    /// Warp a single plane of the frame.
    #[allow(clippy::too_many_arguments)]
    fn warp_plane(
        &self,
        input: &[u8],
        output: &mut oximedia_codec::Plane,
        width: u32,
        height: u32,
        transform: &TransformMatrix,
        border_mode: BorderMode,
        crop_x: u32,
        crop_y: u32,
        crop_width: u32,
        crop_height: u32,
    ) -> CvResult<()> {
        let mut output_data = vec![0u8; (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                // Compute source coordinates
                let (src_x, src_y) = self.transform_coordinates(
                    x as f64,
                    y as f64,
                    transform,
                    crop_x,
                    crop_y,
                    crop_width,
                    crop_height,
                );

                // Sample pixel with interpolation
                let pixel = if self.use_bilinear {
                    self.sample_bilinear(input, src_x, src_y, width, height, border_mode)
                } else {
                    self.sample_nearest(input, src_x, src_y, width, height, border_mode)
                };

                let idx = (y * width + x) as usize;
                output_data[idx] = pixel;
            }
        }

        output.data = output_data;
        Ok(())
    }

    /// Transform coordinates using the transformation matrix.
    #[allow(clippy::too_many_arguments)]
    fn transform_coordinates(
        &self,
        x: f64,
        y: f64,
        transform: &TransformMatrix,
        crop_x: u32,
        crop_y: u32,
        crop_width: u32,
        crop_height: u32,
    ) -> (f64, f64) {
        // Adjust for cropping
        let x_centered = x - crop_x as f64;
        let y_centered = y - crop_y as f64;

        // Apply transformation
        let cos_a = transform.angle.cos();
        let sin_a = transform.angle.sin();
        let s = transform.scale;

        let src_x = s * (cos_a * x_centered - sin_a * y_centered) + transform.tx + crop_x as f64;
        let src_y = s * (sin_a * x_centered + cos_a * y_centered) + transform.ty + crop_y as f64;

        (src_x, src_y)
    }

    /// Sample pixel using bilinear interpolation.
    fn sample_bilinear(
        &self,
        image: &[u8],
        x: f64,
        y: f64,
        width: u32,
        height: u32,
        border_mode: BorderMode,
    ) -> u8 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let fx = x - x0 as f64;
        let fy = y - y0 as f64;

        // Get four corner pixels
        let p00 = self.get_pixel_border(image, x0, y0, width, height, border_mode);
        let p10 = self.get_pixel_border(image, x1, y0, width, height, border_mode);
        let p01 = self.get_pixel_border(image, x0, y1, width, height, border_mode);
        let p11 = self.get_pixel_border(image, x1, y1, width, height, border_mode);

        // Bilinear interpolation
        let top = p00 as f64 * (1.0 - fx) + p10 as f64 * fx;
        let bottom = p01 as f64 * (1.0 - fx) + p11 as f64 * fx;
        let result = top * (1.0 - fy) + bottom * fy;

        result.round().clamp(0.0, 255.0) as u8
    }

    /// Sample pixel using nearest neighbor interpolation.
    fn sample_nearest(
        &self,
        image: &[u8],
        x: f64,
        y: f64,
        width: u32,
        height: u32,
        border_mode: BorderMode,
    ) -> u8 {
        let x_rounded = x.round() as i32;
        let y_rounded = y.round() as i32;
        self.get_pixel_border(image, x_rounded, y_rounded, width, height, border_mode)
    }

    /// Get pixel value with border handling.
    fn get_pixel_border(
        &self,
        image: &[u8],
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        border_mode: BorderMode,
    ) -> u8 {
        let (bx, by) = self.apply_border_mode(x, y, width, height, border_mode);

        if bx < 0 || by < 0 || bx >= width as i32 || by >= height as i32 {
            return 0; // Default to black for out-of-bounds
        }

        let idx = (by * width as i32 + bx) as usize;
        if idx < image.len() {
            image[idx]
        } else {
            0
        }
    }

    /// Apply border mode to coordinates.
    fn apply_border_mode(
        &self,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        border_mode: BorderMode,
    ) -> (i32, i32) {
        match border_mode {
            BorderMode::Crop | BorderMode::Constant => (x, y),
            BorderMode::Replicate => (x.clamp(0, width as i32 - 1), y.clamp(0, height as i32 - 1)),
            BorderMode::Reflect => {
                let bx = if x < 0 {
                    -x
                } else if x >= width as i32 {
                    2 * (width as i32 - 1) - x
                } else {
                    x
                };
                let by = if y < 0 {
                    -y
                } else if y >= height as i32 {
                    2 * (height as i32 - 1) - y
                } else {
                    y
                };
                (bx, by)
            }
            BorderMode::Wrap => (x.rem_euclid(width as i32), y.rem_euclid(height as i32)),
        }
    }
}

impl Default for FrameWarper {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporal frame interpolator.
///
/// Interpolates missing regions in warped frames using temporal information.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::transform::TemporalInterpolator;
///
/// let interpolator = TemporalInterpolator::new(3);
/// ```
#[derive(Debug, Clone)]
pub struct TemporalInterpolator {
    /// Number of frames to use for interpolation.
    window_size: usize,
}

impl TemporalInterpolator {
    /// Create a new temporal interpolator.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of frames to use
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::transform::TemporalInterpolator;
    ///
    /// let interpolator = TemporalInterpolator::new(3);
    /// ```
    #[must_use]
    pub const fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Interpolate missing regions in a frame.
    ///
    /// # Arguments
    ///
    /// * `frames` - Window of frames around the current frame
    /// * `current_idx` - Index of the current frame in the window
    ///
    /// # Errors
    ///
    /// Returns an error if interpolation fails.
    pub fn interpolate(&self, frames: &[VideoFrame], current_idx: usize) -> CvResult<VideoFrame> {
        if current_idx >= frames.len() {
            return Err(CvError::invalid_parameter(
                "current_idx",
                format!("{current_idx}"),
            ));
        }

        // For now, just return the current frame
        // A full implementation would detect and fill missing regions
        Ok(frames[current_idx].clone())
    }

    /// Detect missing regions in a frame.
    fn detect_missing_regions(&self, frame: &VideoFrame) -> Vec<Region> {
        let mut regions = Vec::new();

        // Simple detection: look for completely black regions
        if !frame.planes.is_empty() {
            let plane = &frame.planes[0];
            let (width, height) = frame.plane_dimensions(0);

            let mut in_black_region = false;
            let mut region_start = 0;

            for y in 0..height {
                let mut row_is_black = true;
                for x in 0..width {
                    let idx = (y * width + x) as usize;
                    if idx < plane.data.len() && plane.data[idx] > 10 {
                        row_is_black = false;
                        break;
                    }
                }

                if row_is_black && !in_black_region {
                    in_black_region = true;
                    region_start = y;
                } else if !row_is_black && in_black_region {
                    in_black_region = false;
                    regions.push(Region {
                        x: 0,
                        y: region_start,
                        width,
                        height: y - region_start,
                    });
                }
            }
        }

        regions
    }

    /// Fill missing regions using temporal information.
    fn fill_regions(
        &self,
        frame: &mut VideoFrame,
        regions: &[Region],
        reference_frames: &[VideoFrame],
    ) {
        for region in regions {
            // Average pixels from reference frames
            for ref_frame in reference_frames {
                if ref_frame.planes.is_empty() {
                    continue;
                }

                // Copy region from reference frame
                let src_plane = &ref_frame.planes[0];
                let dst_plane = &mut frame.planes[0];

                for y in region.y..(region.y + region.height) {
                    for x in region.x..(region.x + region.width) {
                        let idx = (y * region.width + x) as usize;
                        if idx < src_plane.data.len() {
                            // Note: This is simplified; real implementation would blend multiple frames
                            let src_data: &Vec<u8> = src_plane.data.as_ref();
                            let mut dst_data = dst_plane.data.clone();
                            if idx < dst_data.len() {
                                dst_data[idx] = src_data[idx];
                            }
                            dst_plane.data = dst_data;
                        }
                    }
                }
            }
        }
    }
}

/// Region in a frame.
#[derive(Debug, Clone, Copy)]
struct Region {
    /// X coordinate.
    x: u32,
    /// Y coordinate.
    y: u32,
    /// Width.
    width: u32,
    /// Height.
    height: u32,
}

/// Perspective warp parameters.
///
/// Defines the parameters for perspective transformation.
#[derive(Debug, Clone, Copy)]
pub struct PerspectiveWarp {
    /// Source points (4 corners).
    pub src_points: [(f64, f64); 4],
    /// Destination points (4 corners).
    pub dst_points: [(f64, f64); 4],
}

impl PerspectiveWarp {
    /// Create a new perspective warp.
    ///
    /// # Arguments
    ///
    /// * `src_points` - Source corner points
    /// * `dst_points` - Destination corner points
    #[must_use]
    pub const fn new(src_points: [(f64, f64); 4], dst_points: [(f64, f64); 4]) -> Self {
        Self {
            src_points,
            dst_points,
        }
    }

    /// Compute homography matrix from point correspondences.
    #[must_use]
    pub fn compute_homography(&self) -> [f64; 9] {
        // Simplified homography computation
        // Full implementation would solve the linear system
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }
}

/// Zoom/crop stabilization mode.
///
/// Applies zoom and crop to avoid black borders.
#[derive(Debug, Clone)]
pub struct ZoomCropStabilizer {
    /// Maximum zoom factor.
    max_zoom: f64,
    /// Minimum crop ratio.
    min_crop: f64,
}

impl ZoomCropStabilizer {
    /// Create a new zoom/crop stabilizer.
    ///
    /// # Arguments
    ///
    /// * `max_zoom` - Maximum zoom factor
    /// * `min_crop` - Minimum crop ratio
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::transform::ZoomCropStabilizer;
    ///
    /// let stabilizer = ZoomCropStabilizer::new(1.2, 0.9);
    /// ```
    #[must_use]
    pub fn new(max_zoom: f64, min_crop: f64) -> Self {
        Self { max_zoom, min_crop }
    }

    /// Compute optimal zoom and crop for a transformation.
    ///
    /// # Arguments
    ///
    /// * `transform` - Input transformation
    /// * `frame_width` - Frame width
    /// * `frame_height` - Frame height
    ///
    /// # Returns
    ///
    /// Returns (zoom_factor, crop_x, crop_y, crop_width, crop_height).
    #[must_use]
    pub fn compute_zoom_crop(
        &self,
        transform: &TransformMatrix,
        frame_width: u32,
        frame_height: u32,
    ) -> (f64, u32, u32, u32, u32) {
        // Compute required zoom to avoid black borders
        let zoom = self.compute_required_zoom(transform);
        let zoom_clamped = zoom.clamp(1.0, self.max_zoom);

        // Compute crop region
        let crop_width = (frame_width as f64 * self.min_crop) as u32;
        let crop_height = (frame_height as f64 * self.min_crop) as u32;
        let crop_x = (frame_width - crop_width) / 2;
        let crop_y = (frame_height - crop_height) / 2;

        (zoom_clamped, crop_x, crop_y, crop_width, crop_height)
    }

    /// Compute required zoom factor to avoid black borders.
    fn compute_required_zoom(&self, transform: &TransformMatrix) -> f64 {
        // Simplified computation based on translation and rotation
        let translation_mag = (transform.tx * transform.tx + transform.ty * transform.ty).sqrt();
        let rotation_mag = transform.angle.abs();

        // Compute zoom needed to cover the transformation
        let zoom_for_translation = 1.0 + translation_mag / 100.0;
        let zoom_for_rotation = 1.0 + rotation_mag / 0.5;

        zoom_for_translation.max(zoom_for_rotation)
    }
}

impl Default for ZoomCropStabilizer {
    fn default() -> Self {
        Self::new(1.2, 0.9)
    }
}

/// Image pyramid for multi-scale processing.
#[derive(Debug, Clone)]
pub struct ImagePyramid {
    /// Pyramid levels.
    levels: Vec<ImageLevel>,
}

impl ImagePyramid {
    /// Create a new image pyramid.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    /// * `num_levels` - Number of pyramid levels
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::transform::ImagePyramid;
    ///
    /// let image_data = vec![0u8; 640 * 480];
    /// let pyramid = ImagePyramid::build(&image_data, 640, 480, 3);
    /// ```
    #[must_use]
    pub fn build(image: &[u8], width: u32, height: u32, num_levels: usize) -> Self {
        let mut levels = Vec::with_capacity(num_levels);

        // First level is the original image
        levels.push(ImageLevel {
            data: image.to_vec(),
            width,
            height,
        });

        // Build subsequent levels by downsampling
        for i in 1..num_levels {
            let prev_level = &levels[i - 1];
            let next_level =
                Self::downsample(&prev_level.data, prev_level.width, prev_level.height);
            levels.push(next_level);
        }

        Self { levels }
    }

    /// Get a specific level of the pyramid.
    #[must_use]
    pub fn level(&self, index: usize) -> Option<&ImageLevel> {
        self.levels.get(index)
    }

    /// Get the number of levels.
    #[must_use]
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Downsample an image by a factor of 2.
    fn downsample(image: &[u8], width: u32, height: u32) -> ImageLevel {
        let new_width = width / 2;
        let new_height = height / 2;
        let mut data = vec![0u8; (new_width * new_height) as usize];

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = x * 2;
                let src_y = y * 2;

                // Average 2x2 block
                let mut sum = 0u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let idx = ((src_y + dy) * width + (src_x + dx)) as usize;
                        if idx < image.len() {
                            sum += image[idx] as u32;
                        }
                    }
                }

                let idx = (y * new_width + x) as usize;
                data[idx] = (sum / 4) as u8;
            }
        }

        ImageLevel {
            data,
            width: new_width,
            height: new_height,
        }
    }
}

/// Single level of an image pyramid.
#[derive(Debug, Clone)]
pub struct ImageLevel {
    /// Image data.
    pub data: Vec<u8>,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}
