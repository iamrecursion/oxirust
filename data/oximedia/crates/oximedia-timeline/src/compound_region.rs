#![allow(dead_code)]
//! Compound region management for nested / grouped timeline segments.

/// How a compound region behaves in the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompoundRegionType {
    /// A fully editable nested sequence.
    Nested,
    /// A logical group that moves together but can be exploded.
    Group,
    /// A collapsed region shown as a single block; contents hidden.
    Collapsed,
}

impl CompoundRegionType {
    /// Returns `true` if this region type can be expanded to show its contents.
    #[must_use]
    pub fn can_expand(&self) -> bool {
        matches!(
            self,
            CompoundRegionType::Nested | CompoundRegionType::Collapsed
        )
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            CompoundRegionType::Nested => "Nested",
            CompoundRegionType::Group => "Group",
            CompoundRegionType::Collapsed => "Collapsed",
        }
    }
}

/// A compound region comprising multiple child clip ids.
#[derive(Debug, Clone)]
pub struct CompoundRegion {
    /// Unique region identifier.
    pub id: u64,
    /// Display name.
    pub name: String,
    /// Region type.
    pub region_type: CompoundRegionType,
    /// Ordered child clip ids.
    pub children: Vec<u64>,
    /// Frames-per-second rate used to interpret frame counts.
    pub fps_num: u32,
    /// Denominator for the fps rational.
    pub fps_den: u32,
}

impl CompoundRegion {
    /// Creates a new compound region.
    pub fn new(
        id: u64,
        name: impl Into<String>,
        region_type: CompoundRegionType,
        fps_num: u32,
        fps_den: u32,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            region_type,
            children: Vec::new(),
            fps_num,
            fps_den,
        }
    }

    /// Adds a child clip id.
    pub fn add_child(&mut self, clip_id: u64) {
        self.children.push(clip_id);
    }

    /// Number of direct children.
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Total duration in frames (sum of `durations` provided for each child, in order).
    /// `durations` must correspond 1-to-1 with `self.children`.
    #[must_use]
    pub fn duration_frames(&self, durations: &[u64]) -> u64 {
        durations.iter().take(self.children.len()).sum()
    }
}

/// Manages a collection of compound regions.
#[derive(Debug, Default)]
pub struct CompoundRegionManager {
    regions: Vec<CompoundRegion>,
    next_id: u64,
}

impl CompoundRegionManager {
    /// Creates a new manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            next_id: 1,
        }
    }

    /// Creates and registers a new compound region, returning its id.
    pub fn create(
        &mut self,
        name: impl Into<String>,
        region_type: CompoundRegionType,
        fps_num: u32,
        fps_den: u32,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.regions
            .push(CompoundRegion::new(id, name, region_type, fps_num, fps_den));
        id
    }

    /// Expands (converts to `Nested`) a `Collapsed` region by id.
    /// Returns `true` on success.
    pub fn expand(&mut self, id: u64) -> bool {
        if let Some(r) = self.regions.iter_mut().find(|r| r.id == id) {
            if r.region_type == CompoundRegionType::Collapsed {
                r.region_type = CompoundRegionType::Nested;
                return true;
            }
        }
        false
    }

    /// Collapses a `Nested` region by id.
    /// Returns `true` on success.
    pub fn collapse(&mut self, id: u64) -> bool {
        if let Some(r) = self.regions.iter_mut().find(|r| r.id == id) {
            if r.region_type == CompoundRegionType::Nested {
                r.region_type = CompoundRegionType::Collapsed;
                return true;
            }
        }
        false
    }

    /// Returns a reference to a region by id.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&CompoundRegion> {
        self.regions.iter().find(|r| r.id == id)
    }

    /// Number of managed regions.
    #[must_use]
    pub fn count(&self) -> usize {
        self.regions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nested_can_expand() {
        assert!(CompoundRegionType::Nested.can_expand());
    }

    #[test]
    fn test_group_cannot_expand() {
        assert!(!CompoundRegionType::Group.can_expand());
    }

    #[test]
    fn test_collapsed_can_expand() {
        assert!(CompoundRegionType::Collapsed.can_expand());
    }

    #[test]
    fn test_region_type_names() {
        assert_eq!(CompoundRegionType::Nested.name(), "Nested");
        assert_eq!(CompoundRegionType::Group.name(), "Group");
        assert_eq!(CompoundRegionType::Collapsed.name(), "Collapsed");
    }

    #[test]
    fn test_compound_region_child_count() {
        let mut r = CompoundRegion::new(1, "R", CompoundRegionType::Group, 24, 1);
        assert_eq!(r.child_count(), 0);
        r.add_child(10);
        r.add_child(20);
        assert_eq!(r.child_count(), 2);
    }

    #[test]
    fn test_compound_region_duration_frames() {
        let mut r = CompoundRegion::new(1, "R", CompoundRegionType::Nested, 24, 1);
        r.add_child(1);
        r.add_child(2);
        assert_eq!(r.duration_frames(&[48, 24]), 72);
    }

    #[test]
    fn test_duration_partial_durations() {
        let mut r = CompoundRegion::new(1, "R", CompoundRegionType::Nested, 30, 1);
        r.add_child(1);
        r.add_child(2);
        // Only one duration provided — partial sum
        assert_eq!(r.duration_frames(&[30]), 30);
    }

    #[test]
    fn test_manager_create_increments_id() {
        let mut mgr = CompoundRegionManager::new();
        let a = mgr.create("A", CompoundRegionType::Group, 25, 1);
        let b = mgr.create("B", CompoundRegionType::Nested, 25, 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
    }

    #[test]
    fn test_manager_get_by_id() {
        let mut mgr = CompoundRegionManager::new();
        let id = mgr.create("Test", CompoundRegionType::Collapsed, 24, 1);
        let r = mgr.get(id).expect("should succeed in test");
        assert_eq!(r.name, "Test");
    }

    #[test]
    fn test_manager_expand_collapsed() {
        let mut mgr = CompoundRegionManager::new();
        let id = mgr.create("C", CompoundRegionType::Collapsed, 24, 1);
        assert!(mgr.expand(id));
        assert_eq!(
            mgr.get(id).expect("should succeed in test").region_type,
            CompoundRegionType::Nested
        );
    }

    #[test]
    fn test_manager_expand_non_collapsed_fails() {
        let mut mgr = CompoundRegionManager::new();
        let id = mgr.create("G", CompoundRegionType::Group, 24, 1);
        assert!(!mgr.expand(id));
    }

    #[test]
    fn test_manager_collapse_nested() {
        let mut mgr = CompoundRegionManager::new();
        let id = mgr.create("N", CompoundRegionType::Nested, 24, 1);
        assert!(mgr.collapse(id));
        assert_eq!(
            mgr.get(id).expect("should succeed in test").region_type,
            CompoundRegionType::Collapsed
        );
    }

    #[test]
    fn test_manager_count() {
        let mut mgr = CompoundRegionManager::new();
        mgr.create("X", CompoundRegionType::Group, 30, 1);
        mgr.create("Y", CompoundRegionType::Group, 30, 1);
        assert_eq!(mgr.count(), 2);
    }
}
