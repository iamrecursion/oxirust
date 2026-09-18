//! Vorbis floor types.
//!
//! Floors encode the spectral envelope (rough shape) of the audio signal.
//! Vorbis defines two floor types:
//!
//! - **Floor 0** - LSP (Line Spectral Pair) based, legacy
//! - **Floor 1** - Piecewise linear, more common
//!
//! Floor 1 is used in almost all modern Vorbis encoders.

#![forbid(unsafe_code)]

/// Floor configuration (common interface).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Floor {
    /// Floor type 0 (LSP-based).
    Type0(FloorType0),
    /// Floor type 1 (piecewise linear).
    Type1(FloorType1),
}

impl Floor {
    /// Get floor type number.
    #[must_use]
    pub fn floor_type(&self) -> u8 {
        match self {
            Floor::Type0(_) => 0,
            Floor::Type1(_) => 1,
        }
    }

    /// Check if this is floor type 0.
    #[must_use]
    pub fn is_type0(&self) -> bool {
        matches!(self, Floor::Type0(_))
    }

    /// Check if this is floor type 1.
    #[must_use]
    pub fn is_type1(&self) -> bool {
        matches!(self, Floor::Type1(_))
    }
}

/// Floor type 0 configuration (LSP-based).
///
/// Floor 0 uses Line Spectral Pairs to represent the spectral envelope.
/// This is the older floor type, rarely used in practice.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct FloorType0 {
    /// Order of the LSP representation.
    pub order: u8,
    /// Rate factor for amplitude interpolation.
    pub rate: u16,
    /// Bark map size.
    pub bark_map_size: u16,
    /// Amplitude bits.
    pub amplitude_bits: u8,
    /// Amplitude offset.
    pub amplitude_offset: u8,
    /// Number of books.
    pub number_of_books: u8,
    /// Book list (codebook numbers).
    pub book_list: Vec<u8>,
}

impl FloorType0 {
    /// Create a new floor type 0 with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the LSP order.
    #[must_use]
    pub fn order(&self) -> u8 {
        self.order
    }

    /// Get bark map size.
    #[must_use]
    pub fn bark_size(&self) -> u16 {
        self.bark_map_size
    }

    /// Compute bark frequency (skeleton).
    #[must_use]
    #[allow(dead_code)]
    pub fn bark_frequency(frequency: f32) -> f32 {
        // Bark scale conversion
        13.1 * (0.00074 * frequency).atan() + 2.24 * (frequency * frequency * 1.85e-8).atan()
    }
}

/// Floor type 1 class.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct FloorClass {
    /// Class dimensions.
    pub dimensions: u8,
    /// Class subclass count.
    pub subclasses: u8,
    /// Master book (for subclass selection).
    pub masterbook: u8,
    /// Subclass books.
    pub subclass_books: Vec<i16>,
}

impl FloorClass {
    /// Create new floor class.
    #[must_use]
    pub fn new(dimensions: u8, subclasses: u8) -> Self {
        Self {
            dimensions,
            subclasses,
            masterbook: 0,
            subclass_books: Vec::new(),
        }
    }

    /// Get number of subclasses.
    #[must_use]
    pub fn subclass_count(&self) -> usize {
        1 << self.subclasses
    }
}

/// Floor type 1 configuration (piecewise linear).
///
/// Floor 1 represents the spectral envelope as a piecewise linear curve.
/// Points are specified at fixed positions (X coordinates) with variable
/// Y values decoded per frame.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct FloorType1 {
    /// Number of partitions.
    pub partitions: u8,
    /// Partition class list.
    pub partition_class_list: Vec<u8>,
    /// Class configurations.
    pub classes: Vec<FloorClass>,
    /// Multiplier (1-4, adds 1 to get actual multiplier).
    pub multiplier: u8,
    /// Range bits.
    pub rangebits: u8,
    /// X positions (sorted).
    pub x_list: Vec<u16>,
    /// Sort order for X positions.
    pub sort_order: Vec<usize>,
}

impl FloorType1 {
    /// Create a new floor type 1 with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the actual multiplier value.
    #[must_use]
    pub fn actual_multiplier(&self) -> u8 {
        self.multiplier.saturating_add(1)
    }

    /// Get number of X positions.
    #[must_use]
    pub fn x_count(&self) -> usize {
        self.x_list.len()
    }

    /// Get range (2^rangebits).
    #[must_use]
    pub fn range(&self) -> u32 {
        1u32 << self.rangebits
    }

    /// Get maximum Y value based on multiplier.
    #[must_use]
    pub fn max_y_value(&self) -> u16 {
        match self.multiplier {
            1 => 128,
            2 => 86,
            3 => 64,
            // 0 and any other values default to 256
            _ => 256,
        }
    }

    /// Add X position and maintain sort order.
    pub fn add_x_position(&mut self, x: u16) {
        let index = self.x_list.len();
        self.x_list.push(x);
        self.sort_order.push(index);
        self.update_sort_order();
    }

    /// Update sort order based on X positions.
    fn update_sort_order(&mut self) {
        let x_list = &self.x_list;
        self.sort_order.sort_by_key(|&i| x_list[i]);
    }

    /// Synthesize floor curve from Y values.
    ///
    /// Returns the floor curve as amplitude values.
    #[must_use]
    #[allow(dead_code, clippy::cast_precision_loss)]
    pub fn synthesize(&self, y_values: &[u16], n: usize) -> Vec<f32> {
        if y_values.is_empty() || self.x_list.is_empty() {
            return vec![0.0; n];
        }

        let mut curve = vec![0.0; n];
        let multiplier = f32::from(self.actual_multiplier());

        // Linear interpolation between floor points
        for i in 1..self.sort_order.len() {
            let idx0 = self.sort_order[i - 1];
            let idx1 = self.sort_order[i];

            if idx0 >= y_values.len() || idx1 >= y_values.len() {
                continue;
            }

            let x0 = self.x_list[idx0] as usize;
            let x1 = self.x_list[idx1] as usize;
            let y0 = f32::from(y_values[idx0]) * multiplier;
            let y1 = f32::from(y_values[idx1]) * multiplier;

            if x1 <= x0 || x0 >= n {
                continue;
            }

            let end_x = x1.min(n);
            for (i, curve_val) in curve.iter_mut().enumerate().take(end_x).skip(x0) {
                let t = (i - x0) as f32 / (x1 - x0) as f32;
                *curve_val = y0 + t * (y1 - y0);
            }
        }

        curve
    }
}

/// Decoded floor data.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub enum FloorData {
    /// Floor type 0 data (LSP coefficients).
    Type0 {
        /// Amplitude.
        amplitude: u32,
        /// LSP coefficients.
        coefficients: Vec<f32>,
    },
    /// Floor type 1 data (Y values).
    Type1 {
        /// Y values for each X position.
        y_values: Vec<u16>,
        /// Final flag (false = floor unused).
        final_y: bool,
    },
    /// Unused floor.
    #[default]
    Unused,
}

impl FloorData {
    /// Check if floor is unused.
    #[must_use]
    pub fn is_unused(&self) -> bool {
        matches!(self, FloorData::Unused)
    }

    /// Check if floor is type 0.
    #[must_use]
    pub fn is_type0(&self) -> bool {
        matches!(self, FloorData::Type0 { .. })
    }

    /// Check if floor is type 1.
    #[must_use]
    pub fn is_type1(&self) -> bool {
        matches!(self, FloorData::Type1 { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_type() {
        let floor0 = Floor::Type0(FloorType0::new());
        assert_eq!(floor0.floor_type(), 0);
        assert!(floor0.is_type0());
        assert!(!floor0.is_type1());

        let floor1 = Floor::Type1(FloorType1::new());
        assert_eq!(floor1.floor_type(), 1);
        assert!(!floor1.is_type0());
        assert!(floor1.is_type1());
    }

    #[test]
    fn test_floor_type0() {
        let floor = FloorType0::new();
        assert_eq!(floor.order(), 0);
        assert_eq!(floor.bark_size(), 0);
    }

    #[test]
    fn test_floor_class() {
        let class = FloorClass::new(3, 2);
        assert_eq!(class.dimensions, 3);
        assert_eq!(class.subclasses, 2);
        assert_eq!(class.subclass_count(), 4); // 2^2 = 4
    }

    #[test]
    fn test_floor_type1() {
        let mut floor = FloorType1::new();
        floor.multiplier = 1;
        floor.rangebits = 8;

        assert_eq!(floor.actual_multiplier(), 2);
        assert_eq!(floor.range(), 256);
        assert_eq!(floor.max_y_value(), 128);
    }

    #[test]
    fn test_floor_type1_x_positions() {
        let mut floor = FloorType1::new();
        floor.add_x_position(100);
        floor.add_x_position(50);
        floor.add_x_position(150);

        assert_eq!(floor.x_count(), 3);
        // Sort order should have index 1 first (x=50)
        assert_eq!(floor.sort_order[0], 1);
    }

    #[test]
    fn test_floor_type1_synthesize_empty() {
        let floor = FloorType1::new();
        let curve = floor.synthesize(&[], 100);
        assert_eq!(curve.len(), 100);
        assert!(curve.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_floor_data() {
        assert!(FloorData::Unused.is_unused());
        assert!(!FloorData::Unused.is_type0());
        assert!(!FloorData::Unused.is_type1());

        let type0 = FloorData::Type0 {
            amplitude: 100,
            coefficients: vec![1.0, 2.0],
        };
        assert!(type0.is_type0());
        assert!(!type0.is_unused());

        let type1 = FloorData::Type1 {
            y_values: vec![10, 20, 30],
            final_y: true,
        };
        assert!(type1.is_type1());
        assert!(!type1.is_unused());
    }

    #[test]
    fn test_floor_data_default() {
        let data = FloorData::default();
        assert!(data.is_unused());
    }
}
