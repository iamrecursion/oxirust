//! Location and setting analysis for scenes.
//!
//! This module provides tools for analyzing the location type (interior/exterior),
//! time of day, and grouping scenes by location similarity.

/// Whether the scene takes place indoors or outdoors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocationType {
    /// Indoor/interior location.
    Interior,
    /// Outdoor/exterior location.
    Exterior,
}

impl LocationType {
    /// Symbol character representing this location type.
    ///
    /// Returns `'I'` for Interior and `'E'` for Exterior.
    #[must_use]
    pub const fn symbol(&self) -> char {
        match self {
            Self::Interior => 'I',
            Self::Exterior => 'E',
        }
    }
}

/// Time of day classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeOfDay {
    /// Dawn / sunrise period.
    Dawn,
    /// Morning hours.
    Morning,
    /// Midday / full daylight.
    Day,
    /// Afternoon hours.
    Afternoon,
    /// Evening / twilight.
    Evening,
    /// Night / dark.
    Night,
    /// Cannot be determined.
    Unknown,
}

impl TimeOfDay {
    /// Estimate time of day from color temperature in Kelvin.
    ///
    /// Approximate ranges:
    /// - Dawn: 1800–2700 K (warm orange sunrise)
    /// - Morning: 2700–4000 K (warm to neutral)
    /// - Day: 4000–6500 K (neutral to cool daylight)
    /// - Afternoon: 3500–5000 K (slightly warm afternoon)
    /// - Evening: 2000–3200 K (warm golden hour)
    /// - Night: < 2000 K or artificial light (very warm or very cool)
    ///
    /// This is a simplified heuristic mapping.
    #[must_use]
    pub fn from_color_temperature(kelvin: f32) -> Self {
        if kelvin < 1800.0 {
            Self::Night
        } else if kelvin < 2500.0 {
            Self::Dawn
        } else if kelvin < 3500.0 {
            Self::Evening
        } else if kelvin < 4500.0 {
            Self::Morning
        } else if kelvin < 6000.0 {
            Self::Afternoon
        } else if kelvin < 8000.0 {
            Self::Day
        } else {
            Self::Unknown
        }
    }
}

/// A tagged description of the location/setting in a scene.
#[derive(Debug, Clone, PartialEq)]
pub struct LocationTag {
    /// Whether the scene is interior or exterior.
    pub location_type: LocationType,
    /// Estimated time of day.
    pub time_of_day: TimeOfDay,
    /// Dominant colors as RGB triplets.
    pub dominant_colors: Vec<[u8; 3]>,
    /// Mean brightness (0.0 = black, 1.0 = white).
    pub brightness: f32,
    /// True if the scene is essentially static (low motion).
    pub is_static: bool,
}

impl LocationTag {
    /// Create a new location tag.
    #[must_use]
    pub fn new(
        location_type: LocationType,
        time_of_day: TimeOfDay,
        dominant_colors: Vec<[u8; 3]>,
        brightness: f32,
        is_static: bool,
    ) -> Self {
        Self {
            location_type,
            time_of_day,
            dominant_colors,
            brightness,
            is_static,
        }
    }
}

/// Analyzes location/setting characteristics from frame statistics.
pub struct LocationAnalyzer;

impl LocationAnalyzer {
    /// Analyze location characteristics from frame-level statistics.
    ///
    /// Parameters:
    /// - `luma_mean`: Mean luma value (0.0–255.0).
    /// - `luma_variance`: Variance of luma values across the frame.
    /// - `color_temperature`: Estimated color temperature in Kelvin.
    /// - `motion_magnitude`: Mean motion magnitude (pixels/frame, 0.0 = static).
    ///
    /// Interior/Exterior decision is based on luma variance and brightness:
    /// - High variance + moderate brightness → Exterior (natural lighting varies)
    /// - Low variance or very high brightness → Interior (artificial even lighting)
    #[must_use]
    pub fn analyze(
        luma_mean: f32,
        luma_variance: f32,
        color_temperature: f32,
        motion_magnitude: f32,
    ) -> LocationTag {
        let brightness = (luma_mean / 255.0).clamp(0.0, 1.0);

        // Exterior heuristic: natural light tends to have higher variance and
        // moderate-to-high color temperature
        let location_type = if luma_variance > 1500.0 && color_temperature > 4000.0 {
            LocationType::Exterior
        } else {
            LocationType::Interior
        };

        let time_of_day = TimeOfDay::from_color_temperature(color_temperature);

        // Dominant color: simplified - derive a single representative color from brightness
        // In a real implementation this would come from k-means clustering of the frame
        let gray = (luma_mean.clamp(0.0, 255.0)) as u8;
        let dominant_colors = vec![[gray, gray, gray]];

        let is_static = motion_magnitude < 2.0;

        LocationTag {
            location_type,
            time_of_day,
            dominant_colors,
            brightness,
            is_static,
        }
    }
}

/// Groups scenes by location similarity.
pub struct LocationCluster;

impl LocationCluster {
    /// Group scene indices by dominant color similarity.
    ///
    /// Uses a simple threshold: scenes whose first dominant color channels
    /// are all within 40 units of each other are placed in the same cluster.
    ///
    /// Returns a list of groups, where each group is a list of indices into `tags`.
    #[must_use]
    pub fn group_by_similarity(tags: &[LocationTag]) -> Vec<Vec<usize>> {
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        const THRESHOLD: i32 = 40;

        'outer: for (i, tag) in tags.iter().enumerate() {
            let color = tag
                .dominant_colors
                .first()
                .copied()
                .unwrap_or([128, 128, 128]);

            for cluster in &mut clusters {
                // Compare against first element of cluster
                let rep_tag = &tags[cluster[0]];
                let rep_color = rep_tag
                    .dominant_colors
                    .first()
                    .copied()
                    .unwrap_or([128, 128, 128]);

                let dr = (color[0] as i32 - rep_color[0] as i32).abs();
                let dg = (color[1] as i32 - rep_color[1] as i32).abs();
                let db = (color[2] as i32 - rep_color[2] as i32).abs();

                if dr <= THRESHOLD && dg <= THRESHOLD && db <= THRESHOLD {
                    cluster.push(i);
                    continue 'outer;
                }
            }

            // No matching cluster found, start a new one
            clusters.push(vec![i]);
        }

        clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_type_symbol_interior() {
        assert_eq!(LocationType::Interior.symbol(), 'I');
    }

    #[test]
    fn test_location_type_symbol_exterior() {
        assert_eq!(LocationType::Exterior.symbol(), 'E');
    }

    #[test]
    fn test_time_of_day_from_temperature_night() {
        let tod = TimeOfDay::from_color_temperature(1500.0);
        assert_eq!(tod, TimeOfDay::Night);
    }

    #[test]
    fn test_time_of_day_from_temperature_dawn() {
        let tod = TimeOfDay::from_color_temperature(2000.0);
        assert_eq!(tod, TimeOfDay::Dawn);
    }

    #[test]
    fn test_time_of_day_from_temperature_evening() {
        let tod = TimeOfDay::from_color_temperature(3000.0);
        assert_eq!(tod, TimeOfDay::Evening);
    }

    #[test]
    fn test_time_of_day_from_temperature_morning() {
        let tod = TimeOfDay::from_color_temperature(4000.0);
        assert_eq!(tod, TimeOfDay::Morning);
    }

    #[test]
    fn test_time_of_day_from_temperature_afternoon() {
        let tod = TimeOfDay::from_color_temperature(5000.0);
        assert_eq!(tod, TimeOfDay::Afternoon);
    }

    #[test]
    fn test_time_of_day_from_temperature_day() {
        let tod = TimeOfDay::from_color_temperature(6500.0);
        assert_eq!(tod, TimeOfDay::Day);
    }

    #[test]
    fn test_time_of_day_from_temperature_unknown() {
        let tod = TimeOfDay::from_color_temperature(9000.0);
        assert_eq!(tod, TimeOfDay::Unknown);
    }

    #[test]
    fn test_location_analyzer_exterior() {
        // High variance + warm daylight temperature → exterior
        let tag = LocationAnalyzer::analyze(180.0, 2000.0, 5500.0, 1.0);
        assert_eq!(tag.location_type, LocationType::Exterior);
    }

    #[test]
    fn test_location_analyzer_interior() {
        // Low variance + warm interior temperature → interior
        let tag = LocationAnalyzer::analyze(150.0, 500.0, 3000.0, 0.5);
        assert_eq!(tag.location_type, LocationType::Interior);
    }

    #[test]
    fn test_location_analyzer_is_static() {
        let tag = LocationAnalyzer::analyze(120.0, 800.0, 3200.0, 0.5);
        assert!(tag.is_static);
    }

    #[test]
    fn test_location_analyzer_not_static() {
        let tag = LocationAnalyzer::analyze(120.0, 800.0, 3200.0, 10.0);
        assert!(!tag.is_static);
    }

    #[test]
    fn test_location_analyzer_brightness_clamped() {
        let tag = LocationAnalyzer::analyze(300.0, 0.0, 3000.0, 0.0);
        assert!((tag.brightness - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_location_cluster_single_group() {
        // All tags with very similar colors should end up in one cluster
        let tags: Vec<LocationTag> = (0..4)
            .map(|i| {
                LocationTag::new(
                    LocationType::Interior,
                    TimeOfDay::Day,
                    vec![[100 + i, 100 + i, 100 + i]],
                    0.5,
                    true,
                )
            })
            .collect();
        let clusters = LocationCluster::group_by_similarity(&tags);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 4);
    }

    #[test]
    fn test_location_cluster_multiple_groups() {
        // Tags with very different colors should form separate clusters
        let mut tags = Vec::new();
        tags.push(LocationTag::new(
            LocationType::Interior,
            TimeOfDay::Day,
            vec![[10, 10, 10]],
            0.1,
            true,
        ));
        tags.push(LocationTag::new(
            LocationType::Exterior,
            TimeOfDay::Day,
            vec![[200, 200, 200]],
            0.9,
            false,
        ));
        tags.push(LocationTag::new(
            LocationType::Interior,
            TimeOfDay::Day,
            vec![[10, 10, 10]],
            0.1,
            true,
        ));
        let clusters = LocationCluster::group_by_similarity(&tags);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_location_cluster_empty() {
        let clusters = LocationCluster::group_by_similarity(&[]);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_location_tag_no_dominant_colors() {
        let tag = LocationTag::new(LocationType::Interior, TimeOfDay::Night, vec![], 0.0, true);
        let clusters = LocationCluster::group_by_similarity(&[tag]);
        assert_eq!(clusters.len(), 1);
    }
}
