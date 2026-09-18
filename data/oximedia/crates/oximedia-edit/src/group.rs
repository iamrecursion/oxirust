//! Clip grouping and linking.
//!
//! Groups allow multiple clips to be treated as a single unit, while links
//! maintain relationships between clips (e.g., video and audio from the same source).

use std::collections::{HashMap, HashSet};

use crate::clip::ClipId;
use crate::error::{EditError, EditResult};

/// Unique identifier for groups.
pub type GroupId = u64;

/// A group of clips that move together.
#[derive(Clone, Debug)]
pub struct ClipGroup {
    /// Unique group identifier.
    pub id: GroupId,
    /// Clips in this group.
    pub clips: HashSet<ClipId>,
    /// Group name.
    pub name: Option<String>,
    /// Group color (for UI).
    pub color: Option<[u8; 3]>,
    /// Group is locked.
    pub locked: bool,
}

impl ClipGroup {
    /// Create a new group.
    #[must_use]
    pub fn new(id: GroupId) -> Self {
        Self {
            id,
            clips: HashSet::new(),
            name: None,
            color: None,
            locked: false,
        }
    }

    /// Add a clip to the group.
    pub fn add_clip(&mut self, clip_id: ClipId) {
        self.clips.insert(clip_id);
    }

    /// Remove a clip from the group.
    pub fn remove_clip(&mut self, clip_id: ClipId) -> bool {
        self.clips.remove(&clip_id)
    }

    /// Check if group contains a clip.
    #[must_use]
    pub fn contains(&self, clip_id: ClipId) -> bool {
        self.clips.contains(&clip_id)
    }

    /// Check if group is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Get clip count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    /// Get all clip IDs.
    #[must_use]
    pub fn clip_ids(&self) -> Vec<ClipId> {
        self.clips.iter().copied().collect()
    }
}

/// Manager for clip groups.
#[derive(Debug, Default)]
pub struct GroupManager {
    /// All groups.
    groups: HashMap<GroupId, ClipGroup>,
    /// Clip to group mapping.
    clip_to_group: HashMap<ClipId, GroupId>,
    /// Next group ID.
    next_id: GroupId,
}

impl GroupManager {
    /// Create a new group manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            clip_to_group: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new group.
    pub fn create_group(&mut self) -> GroupId {
        let id = self.next_id;
        self.next_id += 1;
        self.groups.insert(id, ClipGroup::new(id));
        id
    }

    /// Create a group with clips.
    pub fn create_group_with_clips(&mut self, clips: Vec<ClipId>) -> EditResult<GroupId> {
        let id = self.create_group();
        for clip_id in clips {
            self.add_to_group(id, clip_id)?;
        }
        Ok(id)
    }

    /// Delete a group.
    pub fn delete_group(&mut self, group_id: GroupId) -> Option<ClipGroup> {
        if let Some(group) = self.groups.remove(&group_id) {
            // Remove clip mappings
            for clip_id in &group.clips {
                self.clip_to_group.remove(clip_id);
            }
            Some(group)
        } else {
            None
        }
    }

    /// Add a clip to a group.
    pub fn add_to_group(&mut self, group_id: GroupId, clip_id: ClipId) -> EditResult<()> {
        // Check if clip is already in another group
        if self.clip_to_group.contains_key(&clip_id) {
            return Err(EditError::InvalidEdit(
                "Clip already in a group".to_string(),
            ));
        }

        let group = self
            .groups
            .get_mut(&group_id)
            .ok_or_else(|| EditError::InvalidEdit("Group not found".to_string()))?;

        group.add_clip(clip_id);
        self.clip_to_group.insert(clip_id, group_id);

        Ok(())
    }

    /// Remove a clip from its group.
    pub fn remove_from_group(&mut self, clip_id: ClipId) -> Option<GroupId> {
        if let Some(&group_id) = self.clip_to_group.get(&clip_id) {
            if let Some(group) = self.groups.get_mut(&group_id) {
                group.remove_clip(clip_id);
                self.clip_to_group.remove(&clip_id);
                return Some(group_id);
            }
        }
        None
    }

    /// Get the group containing a clip.
    #[must_use]
    pub fn get_clip_group(&self, clip_id: ClipId) -> Option<&ClipGroup> {
        self.clip_to_group
            .get(&clip_id)
            .and_then(|&group_id| self.groups.get(&group_id))
    }

    /// Get a group by ID.
    #[must_use]
    pub fn get_group(&self, group_id: GroupId) -> Option<&ClipGroup> {
        self.groups.get(&group_id)
    }

    /// Get mutable group by ID.
    pub fn get_group_mut(&mut self, group_id: GroupId) -> Option<&mut ClipGroup> {
        self.groups.get_mut(&group_id)
    }

    /// Get all groups.
    #[must_use]
    pub fn all_groups(&self) -> Vec<&ClipGroup> {
        self.groups.values().collect()
    }

    /// Check if a clip is grouped.
    #[must_use]
    pub fn is_grouped(&self, clip_id: ClipId) -> bool {
        self.clip_to_group.contains_key(&clip_id)
    }

    /// Get all clips in the same group as a clip.
    #[must_use]
    pub fn get_group_members(&self, clip_id: ClipId) -> Vec<ClipId> {
        self.get_clip_group(clip_id)
            .map(ClipGroup::clip_ids)
            .unwrap_or_default()
    }

    /// Clear all groups.
    pub fn clear(&mut self) {
        self.groups.clear();
        self.clip_to_group.clear();
    }

    /// Get total group count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Check if there are no groups.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

/// Type of link between clips.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkType {
    /// Video and audio from the same source.
    VideoAudio,
    /// Synchronized clips (move together).
    Synchronized,
    /// Parent-child relationship.
    ParentChild,
    /// Custom link.
    Custom,
}

/// A link between two clips.
#[derive(Clone, Debug)]
pub struct ClipLink {
    /// Source clip.
    pub clip_a: ClipId,
    /// Destination clip.
    pub clip_b: ClipId,
    /// Link type.
    pub link_type: LinkType,
    /// Link is active.
    pub active: bool,
}

impl ClipLink {
    /// Create a new link.
    #[must_use]
    pub fn new(clip_a: ClipId, clip_b: ClipId, link_type: LinkType) -> Self {
        Self {
            clip_a,
            clip_b,
            link_type,
            active: true,
        }
    }

    /// Check if link involves a clip.
    #[must_use]
    pub fn involves(&self, clip_id: ClipId) -> bool {
        self.clip_a == clip_id || self.clip_b == clip_id
    }

    /// Get the other clip in the link.
    #[must_use]
    pub fn other_clip(&self, clip_id: ClipId) -> Option<ClipId> {
        if self.clip_a == clip_id {
            Some(self.clip_b)
        } else if self.clip_b == clip_id {
            Some(self.clip_a)
        } else {
            None
        }
    }
}

/// Manager for clip links.
#[derive(Debug, Default)]
pub struct LinkManager {
    /// All links.
    links: Vec<ClipLink>,
    /// Clip to links mapping.
    clip_links: HashMap<ClipId, Vec<usize>>,
}

impl LinkManager {
    /// Create a new link manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            links: Vec::new(),
            clip_links: HashMap::new(),
        }
    }

    /// Add a link between two clips.
    pub fn add_link(&mut self, clip_a: ClipId, clip_b: ClipId, link_type: LinkType) -> usize {
        let index = self.links.len();
        let link = ClipLink::new(clip_a, clip_b, link_type);

        self.links.push(link);
        self.clip_links.entry(clip_a).or_default().push(index);
        self.clip_links.entry(clip_b).or_default().push(index);

        index
    }

    /// Add a video-audio link.
    pub fn link_video_audio(&mut self, video_clip: ClipId, audio_clip: ClipId) -> usize {
        self.add_link(video_clip, audio_clip, LinkType::VideoAudio)
    }

    /// Remove a link by index.
    pub fn remove_link(&mut self, index: usize) -> Option<ClipLink> {
        if index >= self.links.len() {
            return None;
        }

        let link = self.links.remove(index);

        // Update clip_links mappings
        self.clip_links.values_mut().for_each(|links| {
            links.retain(|&i| i != index);
            // Adjust indices for links after the removed one
            for link_idx in links.iter_mut() {
                if *link_idx > index {
                    *link_idx -= 1;
                }
            }
        });

        Some(link)
    }

    /// Remove all links involving a clip.
    pub fn remove_clip_links(&mut self, clip_id: ClipId) -> Vec<ClipLink> {
        let link_indices: Vec<usize> = self.clip_links.get(&clip_id).cloned().unwrap_or_default();

        let mut removed = Vec::new();
        // Remove in reverse order to maintain indices
        for &index in link_indices.iter().rev() {
            if let Some(link) = self.remove_link(index) {
                removed.push(link);
            }
        }

        self.clip_links.remove(&clip_id);
        removed
    }

    /// Get all links involving a clip.
    #[must_use]
    pub fn get_clip_links(&self, clip_id: ClipId) -> Vec<&ClipLink> {
        self.clip_links
            .get(&clip_id)
            .map(|indices| indices.iter().filter_map(|&i| self.links.get(i)).collect())
            .unwrap_or_default()
    }

    /// Get linked clips of a specific type.
    #[must_use]
    pub fn get_linked_clips(&self, clip_id: ClipId, link_type: LinkType) -> Vec<ClipId> {
        self.get_clip_links(clip_id)
            .into_iter()
            .filter(|link| link.link_type == link_type && link.active)
            .filter_map(|link| link.other_clip(clip_id))
            .collect()
    }

    /// Check if two clips are linked.
    #[must_use]
    pub fn are_linked(&self, clip_a: ClipId, clip_b: ClipId) -> bool {
        self.get_clip_links(clip_a)
            .iter()
            .any(|link| link.involves(clip_b))
    }

    /// Get all active links.
    #[must_use]
    pub fn active_links(&self) -> Vec<&ClipLink> {
        self.links.iter().filter(|link| link.active).collect()
    }

    /// Clear all links.
    pub fn clear(&mut self) {
        self.links.clear();
        self.clip_links.clear();
    }

    /// Get total link count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// Check if there are no links.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }
}

/// Compound clip - a clip that contains other clips.
#[derive(Clone, Debug)]
pub struct CompoundClip {
    /// Unique identifier.
    pub id: ClipId,
    /// Nested clips.
    pub clips: Vec<ClipId>,
    /// Compound clip name.
    pub name: String,
    /// Duration (maximum end time of nested clips).
    pub duration: i64,
}

impl CompoundClip {
    /// Create a new compound clip.
    #[must_use]
    pub fn new(id: ClipId, name: String) -> Self {
        Self {
            id,
            clips: Vec::new(),
            name,
            duration: 0,
        }
    }

    /// Add a clip to the compound clip.
    pub fn add_clip(&mut self, clip_id: ClipId) {
        self.clips.push(clip_id);
    }

    /// Remove a clip from the compound clip.
    pub fn remove_clip(&mut self, clip_id: ClipId) -> bool {
        if let Some(pos) = self.clips.iter().position(|&id| id == clip_id) {
            self.clips.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if compound clip contains a clip.
    #[must_use]
    pub fn contains(&self, clip_id: ClipId) -> bool {
        self.clips.contains(&clip_id)
    }

    /// Check if compound clip is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Get clip count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.clips.len()
    }
}

/// Manager for compound clips.
#[derive(Debug, Default)]
pub struct CompoundClipManager {
    /// All compound clips.
    compounds: HashMap<ClipId, CompoundClip>,
    /// Clip to compound mapping.
    clip_to_compound: HashMap<ClipId, ClipId>,
}

impl CompoundClipManager {
    /// Create a new compound clip manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            compounds: HashMap::new(),
            clip_to_compound: HashMap::new(),
        }
    }

    /// Create a compound clip.
    pub fn create(&mut self, id: ClipId, name: String) -> ClipId {
        let compound = CompoundClip::new(id, name);
        self.compounds.insert(id, compound);
        id
    }

    /// Delete a compound clip.
    pub fn delete(&mut self, id: ClipId) -> Option<CompoundClip> {
        if let Some(compound) = self.compounds.remove(&id) {
            // Remove clip mappings
            for &clip_id in &compound.clips {
                self.clip_to_compound.remove(&clip_id);
            }
            Some(compound)
        } else {
            None
        }
    }

    /// Add a clip to a compound clip.
    pub fn add_to_compound(&mut self, compound_id: ClipId, clip_id: ClipId) -> EditResult<()> {
        // Check if clip is already in another compound
        if self.clip_to_compound.contains_key(&clip_id) {
            return Err(EditError::InvalidEdit(
                "Clip already in a compound".to_string(),
            ));
        }

        let compound = self
            .compounds
            .get_mut(&compound_id)
            .ok_or_else(|| EditError::InvalidEdit("Compound clip not found".to_string()))?;

        compound.add_clip(clip_id);
        self.clip_to_compound.insert(clip_id, compound_id);

        Ok(())
    }

    /// Remove a clip from its compound.
    pub fn remove_from_compound(&mut self, clip_id: ClipId) -> Option<ClipId> {
        if let Some(&compound_id) = self.clip_to_compound.get(&clip_id) {
            if let Some(compound) = self.compounds.get_mut(&compound_id) {
                compound.remove_clip(clip_id);
                self.clip_to_compound.remove(&clip_id);
                return Some(compound_id);
            }
        }
        None
    }

    /// Get the compound containing a clip.
    #[must_use]
    pub fn get_compound_for_clip(&self, clip_id: ClipId) -> Option<&CompoundClip> {
        self.clip_to_compound
            .get(&clip_id)
            .and_then(|&compound_id| self.compounds.get(&compound_id))
    }

    /// Get a compound clip by ID.
    #[must_use]
    pub fn get(&self, id: ClipId) -> Option<&CompoundClip> {
        self.compounds.get(&id)
    }

    /// Get mutable compound clip by ID.
    pub fn get_mut(&mut self, id: ClipId) -> Option<&mut CompoundClip> {
        self.compounds.get_mut(&id)
    }

    /// Get all compound clips.
    #[must_use]
    pub fn all(&self) -> Vec<&CompoundClip> {
        self.compounds.values().collect()
    }

    /// Clear all compound clips.
    pub fn clear(&mut self) {
        self.compounds.clear();
        self.clip_to_compound.clear();
    }

    /// Get total compound clip count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.compounds.len()
    }

    /// Check if there are no compound clips.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.compounds.is_empty()
    }
}
