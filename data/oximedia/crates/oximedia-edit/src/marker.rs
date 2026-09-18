//! Timeline markers and regions.
//!
//! Markers are used to annotate specific points on the timeline, while regions
//! define ranges of interest.

use std::collections::BTreeMap;

/// Unique identifier for markers.
pub type MarkerId = u64;

/// A marker on the timeline.
#[derive(Clone, Debug)]
pub struct Marker {
    /// Unique marker identifier.
    pub id: MarkerId,
    /// Timeline position.
    pub position: i64,
    /// Marker type.
    pub marker_type: MarkerType,
    /// Marker name.
    pub name: String,
    /// Marker color (for UI).
    pub color: Option<[u8; 3]>,
    /// User notes.
    pub notes: Option<String>,
}

impl Marker {
    /// Create a new marker.
    #[must_use]
    pub fn new(id: MarkerId, position: i64, name: String) -> Self {
        Self {
            id,
            position,
            marker_type: MarkerType::Standard,
            name,
            color: None,
            notes: None,
        }
    }

    /// Create a chapter marker.
    #[must_use]
    pub fn chapter(id: MarkerId, position: i64, name: String) -> Self {
        Self {
            id,
            position,
            marker_type: MarkerType::Chapter,
            name,
            color: Some([0, 100, 200]),
            notes: None,
        }
    }

    /// Create a comment marker.
    #[must_use]
    pub fn comment(id: MarkerId, position: i64, name: String, comment: String) -> Self {
        Self {
            id,
            position,
            marker_type: MarkerType::Comment,
            name,
            color: Some([255, 200, 0]),
            notes: Some(comment),
        }
    }
}

/// Type of marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerType {
    /// Standard marker.
    Standard,
    /// Chapter marker (for video chapters).
    Chapter,
    /// Comment/note marker.
    Comment,
    /// In point marker.
    In,
    /// Out point marker.
    Out,
    /// Cue point marker.
    Cue,
    /// Beat marker (for music).
    Beat,
}

/// A region on the timeline (time range with metadata).
#[derive(Clone, Debug)]
pub struct Region {
    /// Unique region identifier.
    pub id: u64,
    /// Region start position.
    pub start: i64,
    /// Region end position.
    pub end: i64,
    /// Region name.
    pub name: String,
    /// Region color (for UI).
    pub color: Option<[u8; 3]>,
    /// User notes.
    pub notes: Option<String>,
    /// Region is locked.
    pub locked: bool,
}

impl Region {
    /// Create a new region.
    #[must_use]
    pub fn new(id: u64, start: i64, end: i64, name: String) -> Self {
        Self {
            id,
            start,
            end,
            name,
            color: None,
            notes: None,
            locked: false,
        }
    }

    /// Get region duration.
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.end - self.start
    }

    /// Check if region contains a position.
    #[must_use]
    pub fn contains(&self, position: i64) -> bool {
        position >= self.start && position < self.end
    }

    /// Check if region overlaps with another region.
    #[must_use]
    pub fn overlaps(&self, other: &Region) -> bool {
        !(self.end <= other.start || self.start >= other.end)
    }

    /// Check if region overlaps with a time range.
    #[must_use]
    pub fn overlaps_range(&self, start: i64, end: i64) -> bool {
        !(self.end <= start || self.start >= end)
    }
}

/// Manager for timeline markers.
#[derive(Debug, Default)]
pub struct MarkerManager {
    /// All markers indexed by position.
    markers: BTreeMap<i64, Vec<Marker>>,
    /// Next marker ID.
    next_id: MarkerId,
}

impl MarkerManager {
    /// Create a new marker manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            markers: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Add a marker.
    pub fn add(&mut self, mut marker: Marker) -> MarkerId {
        marker.id = self.next_id;
        self.next_id += 1;

        self.markers
            .entry(marker.position)
            .or_default()
            .push(marker.clone());

        marker.id
    }

    /// Add a marker at a specific position.
    pub fn add_at(&mut self, position: i64, name: String) -> MarkerId {
        let marker = Marker::new(self.next_id, position, name);
        self.add(marker)
    }

    /// Remove a marker by ID.
    pub fn remove(&mut self, id: MarkerId) -> Option<Marker> {
        for markers in self.markers.values_mut() {
            if let Some(pos) = markers.iter().position(|m| m.id == id) {
                return Some(markers.remove(pos));
            }
        }
        None
    }

    /// Get a marker by ID.
    #[must_use]
    pub fn get(&self, id: MarkerId) -> Option<&Marker> {
        self.markers
            .values()
            .flat_map(|v| v.iter())
            .find(|m| m.id == id)
    }

    /// Get markers at a specific position.
    #[must_use]
    pub fn get_at(&self, position: i64) -> Vec<&Marker> {
        self.markers
            .get(&position)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get all markers in a time range.
    #[must_use]
    pub fn get_in_range(&self, start: i64, end: i64) -> Vec<&Marker> {
        self.markers
            .range(start..end)
            .flat_map(|(_, markers)| markers.iter())
            .collect()
    }

    /// Get all markers.
    #[must_use]
    pub fn all(&self) -> Vec<&Marker> {
        self.markers.values().flat_map(|v| v.iter()).collect()
    }

    /// Get markers by type.
    #[must_use]
    pub fn get_by_type(&self, marker_type: MarkerType) -> Vec<&Marker> {
        self.all()
            .into_iter()
            .filter(|m| m.marker_type == marker_type)
            .collect()
    }

    /// Get chapter markers sorted by position.
    #[must_use]
    pub fn get_chapters(&self) -> Vec<&Marker> {
        self.get_by_type(MarkerType::Chapter)
    }

    /// Find nearest marker to a position.
    #[must_use]
    pub fn find_nearest(&self, position: i64) -> Option<&Marker> {
        let mut nearest: Option<&Marker> = None;
        let mut min_distance = i64::MAX;

        for marker in self.all() {
            let distance = (marker.position - position).abs();
            if distance < min_distance {
                min_distance = distance;
                nearest = Some(marker);
            }
        }

        nearest
    }

    /// Find next marker after position.
    #[must_use]
    pub fn find_next(&self, position: i64) -> Option<&Marker> {
        self.markers
            .range(position + 1..)
            .flat_map(|(_, markers)| markers.iter())
            .next()
    }

    /// Find previous marker before position.
    #[must_use]
    pub fn find_previous(&self, position: i64) -> Option<&Marker> {
        self.markers
            .range(..position)
            .rev()
            .flat_map(|(_, markers)| markers.iter())
            .next()
    }

    /// Clear all markers.
    pub fn clear(&mut self) {
        self.markers.clear();
    }

    /// Get total marker count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.markers.values().map(Vec::len).sum()
    }

    /// Check if there are no markers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }
}

/// Manager for timeline regions.
#[derive(Debug, Default)]
pub struct RegionManager {
    /// All regions.
    regions: Vec<Region>,
    /// Next region ID.
    next_id: u64,
}

impl RegionManager {
    /// Create a new region manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a region.
    pub fn add(&mut self, mut region: Region) -> u64 {
        region.id = self.next_id;
        self.next_id += 1;
        let id = region.id;
        self.regions.push(region);
        id
    }

    /// Add a region with start and end positions.
    pub fn add_range(&mut self, start: i64, end: i64, name: String) -> u64 {
        let region = Region::new(self.next_id, start, end, name);
        self.add(region)
    }

    /// Remove a region by ID.
    pub fn remove(&mut self, id: u64) -> Option<Region> {
        if let Some(pos) = self.regions.iter().position(|r| r.id == id) {
            Some(self.regions.remove(pos))
        } else {
            None
        }
    }

    /// Get a region by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Region> {
        self.regions.iter().find(|r| r.id == id)
    }

    /// Get mutable region by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Region> {
        self.regions.iter_mut().find(|r| r.id == id)
    }

    /// Get all regions.
    #[must_use]
    pub fn all(&self) -> Vec<&Region> {
        self.regions.iter().collect()
    }

    /// Get regions containing a position.
    #[must_use]
    pub fn get_at(&self, position: i64) -> Vec<&Region> {
        self.regions
            .iter()
            .filter(|r| r.contains(position))
            .collect()
    }

    /// Get regions overlapping a time range.
    #[must_use]
    pub fn get_in_range(&self, start: i64, end: i64) -> Vec<&Region> {
        self.regions
            .iter()
            .filter(|r| r.overlaps_range(start, end))
            .collect()
    }

    /// Clear all regions.
    pub fn clear(&mut self) {
        self.regions.clear();
    }

    /// Get total region count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Check if there are no regions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

/// In/Out points for timeline editing.
#[derive(Clone, Copy, Debug, Default)]
pub struct InOutPoints {
    /// In point (start of selection).
    pub in_point: Option<i64>,
    /// Out point (end of selection).
    pub out_point: Option<i64>,
}

impl InOutPoints {
    /// Create new in/out points.
    #[must_use]
    pub fn new() -> Self {
        Self {
            in_point: None,
            out_point: None,
        }
    }

    /// Set in point.
    pub fn set_in(&mut self, position: i64) {
        self.in_point = Some(position);
    }

    /// Set out point.
    pub fn set_out(&mut self, position: i64) {
        self.out_point = Some(position);
    }

    /// Clear in point.
    pub fn clear_in(&mut self) {
        self.in_point = None;
    }

    /// Clear out point.
    pub fn clear_out(&mut self) {
        self.out_point = None;
    }

    /// Clear both points.
    pub fn clear(&mut self) {
        self.in_point = None;
        self.out_point = None;
    }

    /// Check if both points are set.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.in_point.is_some() && self.out_point.is_some()
    }

    /// Get duration if both points are set.
    #[must_use]
    pub fn duration(&self) -> Option<i64> {
        match (self.in_point, self.out_point) {
            (Some(i), Some(o)) if o > i => Some(o - i),
            _ => None,
        }
    }

    /// Get the range as a tuple.
    #[must_use]
    pub fn range(&self) -> Option<(i64, i64)> {
        match (self.in_point, self.out_point) {
            (Some(i), Some(o)) if o > i => Some((i, o)),
            _ => None,
        }
    }

    /// Check if a position is within the in/out range.
    #[must_use]
    pub fn contains(&self, position: i64) -> bool {
        match (self.in_point, self.out_point) {
            (Some(i), Some(o)) => position >= i && position < o,
            _ => false,
        }
    }
}
