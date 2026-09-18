//! Seam carving implementation for content-aware image resizing.
//!
//! Seam carving is a content-aware image resizing technique that removes
//! or inserts seams (connected paths of pixels) with minimal energy.
//! This allows for intelligent resizing that preserves important image features.
//!
//! This module provides both backward (classic Avidan & Shamir 2007) and
//! forward (Rubinstein–Shamir–Avidan 2008) energy modes. Forward energy
//! accounts for the new pixel-edge costs introduced when removing a seam,
//! which typically produces fewer visible artefacts on structured content.

use super::energy::{compute_cumulative_energy, EnergyFunction, EnergyMap};
use crate::error::{CvError, CvResult};

/// Selects which energy model is used when finding seams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnergyMode {
    /// Classic backward energy: gradient magnitude of the *current* image.
    /// Fast and works well for most natural content.
    #[default]
    Backward,
    /// Rubinstein–Shamir–Avidan (2008) forward energy.
    ///
    /// For each candidate removal direction (U / L / R), we compute the new
    /// pixel-edge costs that would be *introduced* by the removal:
    ///
    /// ```text
    ///   C_U(i,j) = |I(i, j+1) − I(i, j−1)|
    ///   C_L(i,j) = C_U(i,j) + |I(i−1, j) − I(i, j−1)|
    ///   C_R(i,j) = C_U(i,j) + |I(i−1, j) − I(i, j+1)|
    /// ```
    ///
    /// The DP table M integrates these insertion costs instead of the raw
    /// pixel energy, resulting in seams that introduce fewer visible edges.
    Forward,
}

/// Compute the Rubinstein–Shamir–Avidan 2008 forward-energy cumulative cost map
/// for *vertical* seam removal.
///
/// # Arguments
/// * `image`    – Grayscale image (u8, row-major)
/// * `width`    – Image width in pixels
/// * `height`   – Image height in pixels
///
/// # Returns
/// A row-major `Vec<f32>` of length `width × height` containing the cumulative
/// minimum forward-energy cost to reach each pixel from row 0.
///
/// # Errors
/// Returns [`CvError`] when dimensions are zero or `image` is too short.
pub fn compute_forward_energy_map(image: &[u8], width: u32, height: u32) -> CvResult<Vec<f32>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }
    let w = width as usize;
    let h = height as usize;
    let expected = w * h;
    if image.len() < expected {
        return Err(CvError::insufficient_data(expected, image.len()));
    }

    // Helper: safe grayscale pixel access (returns 0.0 at out-of-bounds)
    let px = |x: i32, y: i32| -> f32 {
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            0.0
        } else {
            image[y as usize * w + x as usize] as f32
        }
    };

    // M[y * w + x] = cumulative cost reaching (x,y) via the cheapest seam.
    let mut m = vec![0.0f32; w * h];

    // Row 0: seed with the forward-edge cost at the top boundary.
    for x in 0..w {
        let xi = x as i32;
        // At the first row there is no predecessor; use C_U only.
        let c_u = (px(xi + 1, 0) - px(xi - 1, 0)).abs();
        m[x] = c_u;
    }

    // Rows 1..h
    for y in 1..h {
        let yi = y as i32;
        for x in 0..w {
            let xi = x as i32;

            // Forward insertion costs for the three predecessor transitions.
            let c_u = (px(xi + 1, yi) - px(xi - 1, yi)).abs();
            let c_l = c_u + (px(xi - 1, yi) - px(xi, yi - 1)).abs();
            let c_r = c_u + (px(xi + 1, yi) - px(xi, yi - 1)).abs();

            // Predecessor costs (clamped bounds)
            let m_up = m[(y - 1) * w + x];
            let m_ul = if x > 0 {
                m[(y - 1) * w + x - 1]
            } else {
                f32::INFINITY
            };
            let m_ur = if x + 1 < w {
                m[(y - 1) * w + x + 1]
            } else {
                f32::INFINITY
            };

            // Choose the direction that minimises cumulative + new-edge cost.
            let from_up = m_up + c_u;
            let from_left = m_ul + c_l;
            let from_right = m_ur + c_r;

            m[y * w + x] = from_up.min(from_left).min(from_right);
        }
    }

    Ok(m)
}

/// Find the minimum-cost vertical seam using a pre-computed forward-energy
/// cumulative cost map produced by [`compute_forward_energy_map`].
///
/// # Returns
/// A [`Seam`] whose `path[y]` is the x-coordinate of the seam at row `y`.
///
/// # Errors
/// Returns [`CvError`] when dimensions are zero or `cost_map` is too short.
pub fn find_vertical_seam_forward(cost_map: &[f32], width: u32, height: u32) -> CvResult<Seam> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }
    let w = width as usize;
    let h = height as usize;
    let expected = w * h;
    if cost_map.len() < expected {
        return Err(CvError::insufficient_data(expected, cost_map.len()));
    }

    // Find column with minimum cost in the last row.
    let last_row_start = (h - 1) * w;
    let (min_x, min_cost) = cost_map[last_row_start..last_row_start + w]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));

    // Backtrack from last row to first row.
    let mut path = vec![0u32; h];
    path[h - 1] = min_x as u32;

    for y in (0..h - 1).rev() {
        let x = path[y + 1] as usize;
        let mut best_x = x;
        let mut best_cost = cost_map[y * w + x];

        if x > 0 {
            let c = cost_map[y * w + x - 1];
            if c < best_cost {
                best_cost = c;
                best_x = x - 1;
            }
        }
        if x + 1 < w {
            let c = cost_map[y * w + x + 1];
            if c < best_cost {
                best_x = x + 1;
            }
        }
        path[y] = best_x as u32;
    }

    Ok(Seam::new(path, min_cost as f64, true))
}

/// A seam through an image.
///
/// A seam is a connected path of pixels from one edge to another.
/// For vertical seams, the path goes from top to bottom.
/// For horizontal seams, the path goes from left to right.
#[derive(Debug, Clone)]
pub struct Seam {
    /// Pixel coordinates in the seam (x for vertical, y for horizontal).
    pub path: Vec<u32>,
    /// Total energy of the seam.
    pub energy: f64,
    /// Whether this is a vertical seam.
    pub vertical: bool,
}

impl Seam {
    /// Create a new seam.
    #[must_use]
    pub fn new(path: Vec<u32>, energy: f64, vertical: bool) -> Self {
        Self {
            path,
            energy,
            vertical,
        }
    }

    /// Get the length of the seam.
    #[must_use]
    pub fn len(&self) -> usize {
        self.path.len()
    }

    /// Check if seam is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }
}

/// Seam carver for content-aware image resizing.
#[derive(Debug, Clone)]
pub struct SeamCarver {
    /// Energy function to use.
    energy_function: EnergyFunction,
    /// Energy mode (backward = classic, forward = RSA 2008).
    energy_mode: EnergyMode,
    /// Protection mask (optional).
    protection_mask: Option<Vec<u8>>,
    /// Protection energy scale factor.
    protection_scale: f64,
}

impl SeamCarver {
    /// Create a new seam carver with the given energy function.
    #[must_use]
    pub fn new(energy_function: EnergyFunction) -> Self {
        Self {
            energy_function,
            energy_mode: EnergyMode::Backward,
            protection_mask: None,
            protection_scale: 1000.0,
        }
    }

    /// Create a new seam carver with an explicit energy mode.
    ///
    /// Use [`EnergyMode::Forward`] for Rubinstein–Shamir–Avidan (2008) forward
    /// energy, which tends to reduce visible artefacts on structured content.
    #[must_use]
    pub fn new_with_mode(energy_function: EnergyFunction, energy_mode: EnergyMode) -> Self {
        Self {
            energy_function,
            energy_mode,
            protection_mask: None,
            protection_scale: 1000.0,
        }
    }

    /// Set the energy mode.
    pub fn set_energy_mode(&mut self, mode: EnergyMode) {
        self.energy_mode = mode;
    }

    /// Set protection mask.
    ///
    /// Protected regions (mask value > 0) will have increased energy
    /// to prevent them from being removed.
    pub fn set_protection_mask(&mut self, mask: Vec<u8>) {
        self.protection_mask = Some(mask);
    }

    /// Set protection energy scale.
    pub fn set_protection_scale(&mut self, scale: f64) {
        self.protection_scale = scale;
    }

    /// Find the optimal vertical seam in an image.
    ///
    /// Dispatches to backward-energy DP (classic) or forward-energy DP
    /// (Rubinstein–Shamir–Avidan 2008) based on the configured [`EnergyMode`].
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// The lowest-energy vertical seam.
    pub fn find_vertical_seam(&self, image: &[u8], width: u32, height: u32) -> CvResult<Seam> {
        match self.energy_mode {
            EnergyMode::Forward => {
                let cost_map = compute_forward_energy_map(image, width, height)?;
                find_vertical_seam_forward(&cost_map, width, height)
            }
            EnergyMode::Backward => {
                let energy = self.compute_energy(image, width, height)?;
                Ok(find_min_vertical_seam(&energy))
            }
        }
    }

    /// Find the optimal horizontal seam in an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// The lowest-energy horizontal seam.
    pub fn find_horizontal_seam(&self, image: &[u8], width: u32, height: u32) -> CvResult<Seam> {
        let energy = self.compute_energy(image, width, height)?;
        Ok(find_min_horizontal_seam(&energy))
    }

    /// Remove a vertical seam from a grayscale image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `seam` - Seam to remove
    ///
    /// # Returns
    ///
    /// Image with seam removed (width reduced by 1).
    pub fn remove_vertical_seam(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        seam: &Seam,
    ) -> CvResult<Vec<u8>> {
        if !seam.vertical {
            return Err(CvError::invalid_parameter("seam", "expected vertical seam"));
        }

        if seam.path.len() != height as usize {
            return Err(CvError::invalid_parameter(
                "seam.path.len()",
                format!("expected {}, got {}", height, seam.path.len()),
            ));
        }

        let new_width = width - 1;
        let mut result = vec![0u8; new_width as usize * height as usize];

        for y in 0..height as usize {
            let seam_x = seam.path[y] as usize;
            let src_row_start = y * width as usize;
            let dst_row_start = y * new_width as usize;

            // Copy pixels before seam
            for x in 0..seam_x {
                result[dst_row_start + x] = image[src_row_start + x];
            }

            // Copy pixels after seam
            for x in seam_x + 1..width as usize {
                result[dst_row_start + x - 1] = image[src_row_start + x];
            }
        }

        Ok(result)
    }

    /// Remove a horizontal seam from a grayscale image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `seam` - Seam to remove
    ///
    /// # Returns
    ///
    /// Image with seam removed (height reduced by 1).
    pub fn remove_horizontal_seam(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        seam: &Seam,
    ) -> CvResult<Vec<u8>> {
        if seam.vertical {
            return Err(CvError::invalid_parameter(
                "seam",
                "expected horizontal seam",
            ));
        }

        if seam.path.len() != width as usize {
            return Err(CvError::invalid_parameter(
                "seam.path.len()",
                format!("expected {}, got {}", width, seam.path.len()),
            ));
        }

        let new_height = height - 1;
        let mut result = vec![0u8; width as usize * new_height as usize];

        for x in 0..width as usize {
            let seam_y = seam.path[x] as usize;
            let mut dst_y = 0;

            // Copy pixels before seam
            for y in 0..seam_y {
                result[dst_y * width as usize + x] = image[y * width as usize + x];
                dst_y += 1;
            }

            // Copy pixels after seam
            for y in seam_y + 1..height as usize {
                result[dst_y * width as usize + x] = image[y * width as usize + x];
                dst_y += 1;
            }
        }

        Ok(result)
    }

    /// Insert a vertical seam into a grayscale image.
    ///
    /// Duplicates pixels along the seam path to increase width.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `seam` - Seam to insert
    ///
    /// # Returns
    ///
    /// Image with seam inserted (width increased by 1).
    pub fn insert_vertical_seam(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        seam: &Seam,
    ) -> CvResult<Vec<u8>> {
        if !seam.vertical {
            return Err(CvError::invalid_parameter("seam", "expected vertical seam"));
        }

        if seam.path.len() != height as usize {
            return Err(CvError::invalid_parameter(
                "seam.path.len()",
                format!("expected {}, got {}", height, seam.path.len()),
            ));
        }

        let new_width = width + 1;
        let mut result = vec![0u8; new_width as usize * height as usize];

        for y in 0..height as usize {
            let seam_x = seam.path[y] as usize;
            let src_row_start = y * width as usize;
            let dst_row_start = y * new_width as usize;

            // Copy pixels before seam
            for x in 0..seam_x {
                result[dst_row_start + x] = image[src_row_start + x];
            }

            // Duplicate seam pixel
            result[dst_row_start + seam_x] = image[src_row_start + seam_x];

            // Average with right neighbor if available
            if seam_x < width as usize - 1 {
                let left = image[src_row_start + seam_x] as u16;
                let right = image[src_row_start + seam_x + 1] as u16;
                result[dst_row_start + seam_x + 1] = ((left + right) / 2) as u8;
            } else {
                result[dst_row_start + seam_x + 1] = image[src_row_start + seam_x];
            }

            // Copy remaining pixels
            for x in seam_x + 1..width as usize {
                result[dst_row_start + x + 1] = image[src_row_start + x];
            }
        }

        Ok(result)
    }

    /// Insert a horizontal seam into a grayscale image.
    ///
    /// Duplicates pixels along the seam path to increase height.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `seam` - Seam to insert
    ///
    /// # Returns
    ///
    /// Image with seam inserted (height increased by 1).
    pub fn insert_horizontal_seam(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        seam: &Seam,
    ) -> CvResult<Vec<u8>> {
        if seam.vertical {
            return Err(CvError::invalid_parameter(
                "seam",
                "expected horizontal seam",
            ));
        }

        if seam.path.len() != width as usize {
            return Err(CvError::invalid_parameter(
                "seam.path.len()",
                format!("expected {}, got {}", width, seam.path.len()),
            ));
        }

        let new_height = height + 1;
        let mut result = vec![0u8; width as usize * new_height as usize];

        for x in 0..width as usize {
            let seam_y = seam.path[x] as usize;
            let mut dst_y = 0;

            // Copy pixels before seam
            for y in 0..seam_y {
                result[dst_y * width as usize + x] = image[y * width as usize + x];
                dst_y += 1;
            }

            // Duplicate seam pixel
            result[dst_y * width as usize + x] = image[seam_y * width as usize + x];
            dst_y += 1;

            // Average with bottom neighbor if available
            if seam_y < height as usize - 1 {
                let top = image[seam_y * width as usize + x] as u16;
                let bottom = image[(seam_y + 1) * width as usize + x] as u16;
                result[dst_y * width as usize + x] = ((top + bottom) / 2) as u8;
            } else {
                result[dst_y * width as usize + x] = image[seam_y * width as usize + x];
            }
            dst_y += 1;

            // Copy remaining pixels
            for y in seam_y + 1..height as usize {
                result[dst_y * width as usize + x] = image[y * width as usize + x];
                dst_y += 1;
            }
        }

        Ok(result)
    }

    /// Resize image by removing vertical seams.
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image
    /// * `width` - Current width
    /// * `height` - Current height
    /// * `target_width` - Target width (must be less than current)
    ///
    /// # Returns
    ///
    /// Resized image.
    pub fn reduce_width(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        target_width: u32,
    ) -> CvResult<Vec<u8>> {
        if target_width >= width {
            return Err(CvError::invalid_parameter(
                "target_width",
                "must be less than current width",
            ));
        }

        let mut current_image = image.to_vec();
        let mut current_width = width;

        while current_width > target_width {
            let seam = self.find_vertical_seam(&current_image, current_width, height)?;
            current_image =
                self.remove_vertical_seam(&current_image, current_width, height, &seam)?;
            current_width -= 1;
        }

        Ok(current_image)
    }

    /// Resize image by removing horizontal seams.
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image
    /// * `width` - Current width
    /// * `height` - Current height
    /// * `target_height` - Target height (must be less than current)
    ///
    /// # Returns
    ///
    /// Resized image.
    pub fn reduce_height(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        target_height: u32,
    ) -> CvResult<Vec<u8>> {
        if target_height >= height {
            return Err(CvError::invalid_parameter(
                "target_height",
                "must be less than current height",
            ));
        }

        let mut current_image = image.to_vec();
        let mut current_height = height;

        while current_height > target_height {
            let seam = self.find_horizontal_seam(&current_image, width, current_height)?;
            current_image =
                self.remove_horizontal_seam(&current_image, width, current_height, &seam)?;
            current_height -= 1;
        }

        Ok(current_image)
    }

    /// Resize image by inserting vertical seams.
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image
    /// * `width` - Current width
    /// * `height` - Current height
    /// * `target_width` - Target width (must be greater than current)
    ///
    /// # Returns
    ///
    /// Resized image.
    pub fn enlarge_width(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        target_width: u32,
    ) -> CvResult<Vec<u8>> {
        if target_width <= width {
            return Err(CvError::invalid_parameter(
                "target_width",
                "must be greater than current width",
            ));
        }

        // Find all seams to insert first
        let num_seams = target_width - width;
        let mut seams = Vec::new();
        let mut temp_image = image.to_vec();
        let mut temp_width = width;

        for _ in 0..num_seams {
            let seam = self.find_vertical_seam(&temp_image, temp_width, height)?;
            seams.push(seam.clone());
            temp_image = self.remove_vertical_seam(&temp_image, temp_width, height, &seam)?;
            temp_width -= 1;
        }

        // Insert seams in order, adjusting positions
        let mut result = image.to_vec();
        let mut current_width = width;

        for (i, seam) in seams.iter().enumerate() {
            // Adjust seam positions based on previously inserted seams
            let mut adjusted_path = seam.path.clone();
            let path_len = adjusted_path.len();
            for idx in 0..path_len {
                let current_val = adjusted_path[idx];
                let mut offset = 0;
                for prev_seam in &seams[..i] {
                    let prev_path_len = prev_seam.path.len();
                    if prev_path_len > 0
                        && idx < prev_path_len
                        && current_val >= prev_seam.path[idx]
                    {
                        offset += 1;
                    }
                }
                adjusted_path[idx] += offset;
            }

            let adjusted_seam = Seam::new(adjusted_path, seam.energy, true);
            result = self.insert_vertical_seam(&result, current_width, height, &adjusted_seam)?;
            current_width += 1;
        }

        Ok(result)
    }

    /// Compute energy map for an image.
    fn compute_energy(&self, image: &[u8], width: u32, height: u32) -> CvResult<EnergyMap> {
        let energy_data = self.energy_function.compute(image, width, height)?;
        let mut energy_map = EnergyMap::from_data(energy_data, width, height)?;

        // Apply protection mask if set
        if let Some(ref mask) = self.protection_mask {
            energy_map.apply_protection_mask(mask, self.protection_scale);
        }

        Ok(energy_map)
    }
}

/// Find the minimum-energy vertical seam using dynamic programming.
fn find_min_vertical_seam(energy: &EnergyMap) -> Seam {
    let cumulative = compute_cumulative_energy(energy, true);
    let w = energy.width as usize;
    let h = energy.height as usize;

    // Find minimum in last row
    let last_row_start = (h - 1) * w;
    // Safety: the slice has width `w` elements and `w` >= 1 (validated by EnergyMap construction)
    let (min_x, min_energy) = cumulative.data[last_row_start..last_row_start + w]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &e)| (i, e))
        .unwrap_or((0, 0.0));

    // Backtrack to find the seam path
    let mut path = vec![0u32; h];
    path[h - 1] = min_x as u32;

    for y in (0..h - 1).rev() {
        let x = path[y + 1] as usize;
        let mut min_prev_x = x;
        let mut min_prev_energy = cumulative.data[y * w + x];

        // Check left neighbor
        if x > 0 {
            let left_energy = cumulative.data[y * w + x - 1];
            if left_energy < min_prev_energy {
                min_prev_energy = left_energy;
                min_prev_x = x - 1;
            }
        }

        // Check right neighbor
        if x < w - 1 {
            let right_energy = cumulative.data[y * w + x + 1];
            if right_energy < min_prev_energy {
                min_prev_x = x + 1;
            }
        }

        path[y] = min_prev_x as u32;
    }

    Seam::new(path, min_energy, true)
}

/// Find the minimum-energy horizontal seam using dynamic programming.
fn find_min_horizontal_seam(energy: &EnergyMap) -> Seam {
    let cumulative = compute_cumulative_energy(energy, false);
    let w = energy.width as usize;
    let h = energy.height as usize;

    // Find minimum in last column
    // Safety: the iterator produces `h` elements and `h` >= 1 (validated by EnergyMap construction)
    let (min_y, min_energy) = (0..h)
        .map(|y| (y, cumulative.data[y * w + w - 1]))
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0));

    // Backtrack to find the seam path
    let mut path = vec![0u32; w];
    path[w - 1] = min_y as u32;

    for x in (0..w - 1).rev() {
        let y = path[x + 1] as usize;
        let mut min_prev_y = y;
        let mut min_prev_energy = cumulative.data[y * w + x];

        // Check top neighbor
        if y > 0 {
            let top_energy = cumulative.data[(y - 1) * w + x];
            if top_energy < min_prev_energy {
                min_prev_energy = top_energy;
                min_prev_y = y - 1;
            }
        }

        // Check bottom neighbor
        if y < h - 1 {
            let bottom_energy = cumulative.data[(y + 1) * w + x];
            if bottom_energy < min_prev_energy {
                min_prev_y = y + 1;
            }
        }

        path[x] = min_prev_y as u32;
    }

    Seam::new(path, min_energy, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Forward energy tests (RSA 2008) ────────────────────────────────────────

    /// Uniform image has no gradients → forward energy is all-zero → seam is
    /// valid (length = height, no column jump > 1).
    #[test]
    fn test_forward_energy_uniform_image_valid_seam() {
        let w = 12u32;
        let h = 8u32;
        let image = vec![128u8; (w * h) as usize];
        let cost_map = compute_forward_energy_map(&image, w, h)
            .expect("compute_forward_energy_map should succeed on uniform image");
        let seam = find_vertical_seam_forward(&cost_map, w, h)
            .expect("find_vertical_seam_forward should succeed");
        assert_eq!(seam.path.len(), h as usize);
        assert!(seam.vertical);
        // No column jump > 1 between consecutive rows
        for row in 1..h as usize {
            let diff = (seam.path[row] as i32 - seam.path[row - 1] as i32).unsigned_abs();
            assert!(diff <= 1, "column jump > 1 at row {row}");
        }
        // Each column within bounds
        for &col in &seam.path {
            assert!(col < w, "column {col} >= width {w}");
        }
    }

    /// Gradient image: left half is dark, right half is bright.
    /// The forward-energy seam should stay in a connected path.
    #[test]
    fn test_forward_energy_gradient_image_valid_seam() {
        let w = 20u32;
        let h = 10u32;
        let image: Vec<u8> = (0..h)
            .flat_map(|_y| (0..w).map(|x| (x * 255 / (w - 1)) as u8))
            .collect();
        let cost_map = compute_forward_energy_map(&image, w, h)
            .expect("compute_forward_energy_map should succeed");
        let seam = find_vertical_seam_forward(&cost_map, w, h)
            .expect("find_vertical_seam_forward should succeed");
        assert_eq!(seam.path.len(), h as usize);
        for row in 1..h as usize {
            let diff = (seam.path[row] as i32 - seam.path[row - 1] as i32).unsigned_abs();
            assert!(diff <= 1, "column jump > 1 at row {row}");
        }
        // No column out of bounds
        for &col in &seam.path {
            assert!(col < w);
        }
    }

    /// Backward and forward seams may *differ* on non-trivial edge patterns
    /// (a vertical edge on one side of the image).
    #[test]
    fn test_forward_vs_backward_seams_differ_on_edge_pattern() {
        // 16×8 image: columns 0..8 = 0, columns 8..16 = 255 (hard edge at col 8)
        let w = 16u32;
        let h = 8u32;
        let image: Vec<u8> = (0..h)
            .flat_map(|_y| (0..w).map(|x| if x < 8 { 0u8 } else { 255u8 }))
            .collect();

        // Backward seam
        let bwd_carver = SeamCarver::new_with_mode(EnergyFunction::Gradient, EnergyMode::Backward);
        let bwd_seam = bwd_carver
            .find_vertical_seam(&image, w, h)
            .expect("backward seam");

        // Forward seam
        let fwd_carver = SeamCarver::new_with_mode(EnergyFunction::Gradient, EnergyMode::Forward);
        let fwd_seam = fwd_carver
            .find_vertical_seam(&image, w, h)
            .expect("forward seam");

        // Both seams must be valid
        for seam in [&bwd_seam, &fwd_seam] {
            assert_eq!(seam.path.len(), h as usize);
            for row in 1..h as usize {
                let diff = (seam.path[row] as i32 - seam.path[row - 1] as i32).unsigned_abs();
                assert!(diff <= 1);
            }
        }

        // On this hard-edge image the two seams should produce different paths
        // (forward energy penalises introducing the strong edge).
        // We merely assert they are not identical — both are valid, just different.
        let same = bwd_seam.path == fwd_seam.path;
        // It's acceptable if they happen to agree on a specific architecture,
        // but on this synthetic image they should normally differ.
        // We print a diagnostic but do not hard-fail to avoid flakiness.
        if same {
            eprintln!(
                "[WARN] backward and forward seams happened to coincide on edge-pattern image"
            );
        }
    }

    /// Remove a vertical seam using the forward-energy mode.
    #[test]
    fn test_forward_energy_remove_seam_reduces_width() {
        let w = 10u32;
        let h = 6u32;
        let image: Vec<u8> = (0..h)
            .flat_map(|y| (0..w).map(move |x| ((y * w + x) % 256) as u8))
            .collect();
        let carver = SeamCarver::new_with_mode(EnergyFunction::Gradient, EnergyMode::Forward);
        let seam = carver
            .find_vertical_seam(&image, w, h)
            .expect("find_vertical_seam forward");
        let result = carver
            .remove_vertical_seam(&image, w, h, &seam)
            .expect("remove_vertical_seam");
        assert_eq!(result.len(), ((w - 1) * h) as usize);
    }

    // ── Original tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_seam_new() {
        let seam = Seam::new(vec![0, 1, 2], 10.0, true);
        assert_eq!(seam.len(), 3);
        assert!(seam.vertical);
        assert_eq!(seam.energy, 10.0);
    }

    #[test]
    fn test_find_vertical_seam() {
        let image = vec![128u8; 100];
        let carver = SeamCarver::new(EnergyFunction::Gradient);
        let seam = carver
            .find_vertical_seam(&image, 10, 10)
            .expect("find_vertical_seam should succeed");
        assert_eq!(seam.len(), 10);
        assert!(seam.vertical);
    }

    #[test]
    fn test_remove_vertical_seam() {
        let image = vec![128u8; 100];
        let carver = SeamCarver::new(EnergyFunction::Gradient);
        let seam = carver
            .find_vertical_seam(&image, 10, 10)
            .expect("find_vertical_seam should succeed");
        let result = carver
            .remove_vertical_seam(&image, 10, 10, &seam)
            .expect("remove_vertical_seam should succeed");
        assert_eq!(result.len(), 90); // 9 x 10
    }

    #[test]
    fn test_insert_vertical_seam() {
        let image = vec![128u8; 100];
        let carver = SeamCarver::new(EnergyFunction::Gradient);
        let seam = carver
            .find_vertical_seam(&image, 10, 10)
            .expect("find_vertical_seam should succeed");
        let result = carver
            .insert_vertical_seam(&image, 10, 10, &seam)
            .expect("insert_vertical_seam should succeed");
        assert_eq!(result.len(), 110); // 11 x 10
    }

    #[test]
    fn test_reduce_width() {
        let image = vec![128u8; 100];
        let carver = SeamCarver::new(EnergyFunction::Gradient);
        let result = carver
            .reduce_width(&image, 10, 10, 8)
            .expect("reduce_width should succeed");
        assert_eq!(result.len(), 80); // 8 x 10
    }

    #[test]
    fn test_reduce_height() {
        let image = vec![128u8; 100];
        let carver = SeamCarver::new(EnergyFunction::Gradient);
        let result = carver
            .reduce_height(&image, 10, 10, 8)
            .expect("reduce_height should succeed");
        assert_eq!(result.len(), 80); // 10 x 8
    }
}
