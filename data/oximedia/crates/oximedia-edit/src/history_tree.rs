//! Branching history tree for undo/redo.
//!
//! Unlike a linear undo stack that discards redo history when a new
//! action is performed, the branching history tree preserves all
//! branches. Each branch represents an alternative editing path,
//! allowing the user to explore different edit choices and return
//! to any previous state.

use std::collections::HashMap;

/// Unique identifier for a history node.
pub type NodeId = u64;

/// Unique identifier for a history branch.
pub type BranchId = u64;

/// An action recorded in the history tree.
#[derive(Debug, Clone, PartialEq)]
pub enum HistoryAction {
    /// A clip was added.
    AddClip {
        /// Clip ID.
        clip_id: u64,
        /// Track index.
        track_index: usize,
    },
    /// A clip was removed.
    RemoveClip {
        /// Clip ID.
        clip_id: u64,
        /// Track index.
        track_index: usize,
    },
    /// A clip was moved.
    MoveClip {
        /// Clip ID.
        clip_id: u64,
        /// Old start position.
        old_start: i64,
        /// New start position.
        new_start: i64,
    },
    /// A clip was trimmed.
    TrimClip {
        /// Clip ID.
        clip_id: u64,
        /// Old in-point.
        old_in: i64,
        /// Old out-point.
        old_out: i64,
        /// New in-point.
        new_in: i64,
        /// New out-point.
        new_out: i64,
    },
    /// A clip was split.
    SplitClip {
        /// Original clip ID.
        original_id: u64,
        /// New clip ID.
        new_id: u64,
        /// Position of the split.
        position: i64,
    },
    /// A transition was added.
    AddTransition {
        /// Clip A ID.
        clip_a: u64,
        /// Clip B ID.
        clip_b: u64,
        /// Transition duration.
        duration: i64,
    },
    /// A compound/group action containing multiple sub-actions.
    Compound {
        /// Description of the compound action.
        description: String,
        /// Sub-actions in order.
        actions: Vec<HistoryAction>,
    },
}

impl HistoryAction {
    /// Returns a human-readable description.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::AddClip { .. } => "Add clip",
            Self::RemoveClip { .. } => "Remove clip",
            Self::MoveClip { .. } => "Move clip",
            Self::TrimClip { .. } => "Trim clip",
            Self::SplitClip { .. } => "Split clip",
            Self::AddTransition { .. } => "Add transition",
            Self::Compound { description, .. } => description,
        }
    }

    /// Returns the inverse action (for undo).
    #[must_use]
    pub fn inverse(&self) -> Self {
        match self.clone() {
            Self::AddClip {
                clip_id,
                track_index,
            } => Self::RemoveClip {
                clip_id,
                track_index,
            },
            Self::RemoveClip {
                clip_id,
                track_index,
            } => Self::AddClip {
                clip_id,
                track_index,
            },
            Self::MoveClip {
                clip_id,
                old_start,
                new_start,
            } => Self::MoveClip {
                clip_id,
                old_start: new_start,
                new_start: old_start,
            },
            Self::TrimClip {
                clip_id,
                old_in,
                old_out,
                new_in,
                new_out,
            } => Self::TrimClip {
                clip_id,
                old_in: new_in,
                old_out: new_out,
                new_in: old_in,
                new_out: old_out,
            },
            Self::SplitClip {
                original_id,
                new_id,
                position,
            } => Self::SplitClip {
                original_id,
                new_id,
                position,
            },
            Self::AddTransition {
                clip_a,
                clip_b,
                duration,
            } => Self::AddTransition {
                clip_a,
                clip_b,
                duration,
            },
            Self::Compound {
                description,
                actions,
            } => Self::Compound {
                description,
                actions: actions.into_iter().rev().map(|a| a.inverse()).collect(),
            },
        }
    }
}

/// A node in the history tree.
#[derive(Debug, Clone)]
pub struct HistoryNode {
    /// Unique node ID.
    pub id: NodeId,
    /// The action performed at this node.
    pub action: HistoryAction,
    /// Parent node (None for root).
    pub parent: Option<NodeId>,
    /// Child nodes (branches).
    pub children: Vec<NodeId>,
    /// Branch this node belongs to.
    pub branch_id: BranchId,
    /// Timestamp (monotonic counter) when this node was created.
    pub timestamp: u64,
}

/// Metadata about a branch.
#[derive(Debug, Clone)]
pub struct BranchInfo {
    /// Branch ID.
    pub id: BranchId,
    /// Human-readable name.
    pub name: String,
    /// Node where this branch diverged from the parent.
    pub fork_point: Option<NodeId>,
    /// The tip (latest) node of this branch.
    pub tip: NodeId,
    /// Whether this is the currently active branch.
    pub active: bool,
}

/// A branching history tree that preserves all edit paths.
///
/// The tree maintains a current position (`cursor`) that indicates
/// the latest applied action. Undo moves the cursor toward the root;
/// redo moves it forward along the active branch. New actions at a
/// non-tip position create a new branch.
#[derive(Debug)]
pub struct HistoryTree {
    /// All nodes in the tree.
    nodes: HashMap<NodeId, HistoryNode>,
    /// Branch metadata.
    branches: HashMap<BranchId, BranchInfo>,
    /// Current cursor position (last applied action).
    cursor: Option<NodeId>,
    /// Currently active branch.
    active_branch: BranchId,
    /// Next node ID.
    next_node_id: NodeId,
    /// Next branch ID.
    next_branch_id: BranchId,
    /// Monotonic timestamp counter.
    timestamp_counter: u64,
    /// Maximum total nodes before pruning oldest branches.
    max_nodes: usize,
}

impl HistoryTree {
    /// Create a new empty history tree.
    #[must_use]
    pub fn new(max_nodes: usize) -> Self {
        let mut tree = Self {
            nodes: HashMap::new(),
            branches: HashMap::new(),
            cursor: None,
            active_branch: 1,
            next_node_id: 1,
            next_branch_id: 2,
            timestamp_counter: 0,
            max_nodes,
        };
        // Create the main branch
        tree.branches.insert(
            1,
            BranchInfo {
                id: 1,
                name: "Main".to_string(),
                fork_point: None,
                tip: 0, // will be updated on first push
                active: true,
            },
        );
        tree
    }

    /// Push a new action onto the tree.
    ///
    /// If the cursor is at the tip of the active branch, the action
    /// is appended. If the cursor is behind the tip (i.e., after some
    /// undos), a new branch is created from the current position.
    pub fn push(&mut self, action: HistoryAction) -> NodeId {
        self.timestamp_counter += 1;

        // Determine if we need to create a new branch
        let is_at_tip = self.cursor.map_or(true, |c| {
            self.branches
                .get(&self.active_branch)
                .map_or(true, |b| b.tip == c || b.tip == 0)
        });

        let branch_id = if !is_at_tip && self.cursor.is_some() {
            // Create a new branch
            self.create_branch_from_cursor()
        } else {
            self.active_branch
        };

        let node_id = self.next_node_id;
        self.next_node_id += 1;

        let node = HistoryNode {
            id: node_id,
            action,
            parent: self.cursor,
            children: Vec::new(),
            branch_id,
            timestamp: self.timestamp_counter,
        };

        // Update parent's children
        if let Some(parent_id) = self.cursor {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.children.push(node_id);
            }
        }

        self.nodes.insert(node_id, node);

        // Update branch tip
        if let Some(branch) = self.branches.get_mut(&branch_id) {
            branch.tip = node_id;
        }

        self.cursor = Some(node_id);

        // Prune if necessary
        if self.nodes.len() > self.max_nodes {
            self.prune_oldest_branch();
        }

        node_id
    }

    /// Undo: move cursor to parent node.
    ///
    /// Returns the action that was undone (cloned).
    pub fn undo(&mut self) -> Option<HistoryAction> {
        let cursor = self.cursor?;
        let node = self.nodes.get(&cursor)?;
        let action = node.action.clone();
        self.cursor = node.parent;
        Some(action)
    }

    /// Redo: move cursor to the first child on the active branch.
    ///
    /// Returns the action that was redone (cloned).
    pub fn redo(&mut self) -> Option<HistoryAction> {
        let cursor_id = self.cursor;

        // Find the child node that belongs to the active branch
        let child_id = if let Some(cid) = cursor_id {
            let node = self.nodes.get(&cid)?;
            node.children
                .iter()
                .find(|&&child| {
                    self.nodes
                        .get(&child)
                        .map_or(false, |n| n.branch_id == self.active_branch)
                })
                .or_else(|| node.children.first())
                .copied()
        } else {
            // At root: find the first node with no parent on active branch
            self.nodes
                .values()
                .find(|n| n.parent.is_none() && n.branch_id == self.active_branch)
                .map(|n| n.id)
        };

        let child_id = child_id?;
        self.cursor = Some(child_id);
        self.nodes.get(&child_id).map(|n| n.action.clone())
    }

    /// Switch to a different branch.
    pub fn switch_branch(&mut self, branch_id: BranchId) -> bool {
        if !self.branches.contains_key(&branch_id) {
            return false;
        }

        // Deactivate current branch
        if let Some(branch) = self.branches.get_mut(&self.active_branch) {
            branch.active = false;
        }

        // Activate new branch
        self.active_branch = branch_id;
        if let Some(branch) = self.branches.get_mut(&branch_id) {
            branch.active = true;
            // Move cursor to the branch tip
            self.cursor = Some(branch.tip);
        }

        true
    }

    /// Get the current cursor position.
    #[must_use]
    pub fn cursor(&self) -> Option<NodeId> {
        self.cursor
    }

    /// Get the current action (at cursor).
    #[must_use]
    pub fn current_action(&self) -> Option<&HistoryAction> {
        self.cursor
            .and_then(|id| self.nodes.get(&id))
            .map(|n| &n.action)
    }

    /// Check if undo is available.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        self.cursor.is_some()
    }

    /// Check if redo is available.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        if let Some(cursor) = self.cursor {
            self.nodes
                .get(&cursor)
                .map_or(false, |n| !n.children.is_empty())
        } else {
            // At root, check if there are any root nodes
            self.nodes.values().any(|n| n.parent.is_none())
        }
    }

    /// Get all branch IDs and names.
    #[must_use]
    pub fn branch_list(&self) -> Vec<(BranchId, &str)> {
        self.branches
            .values()
            .map(|b| (b.id, b.name.as_str()))
            .collect()
    }

    /// Get the active branch ID.
    #[must_use]
    pub fn active_branch_id(&self) -> BranchId {
        self.active_branch
    }

    /// Get the total number of nodes in the tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the total number of branches.
    #[must_use]
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Get the path from root to cursor (list of actions).
    #[must_use]
    pub fn path_to_cursor(&self) -> Vec<&HistoryAction> {
        let mut path = Vec::new();
        let mut current = self.cursor;
        while let Some(id) = current {
            if let Some(node) = self.nodes.get(&id) {
                path.push(&node.action);
                current = node.parent;
            } else {
                break;
            }
        }
        path.reverse();
        path
    }

    /// Clear the entire tree.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.branches.clear();
        self.cursor = None;
        self.active_branch = 1;
        self.next_node_id = 1;
        self.next_branch_id = 2;
        self.timestamp_counter = 0;
        self.branches.insert(
            1,
            BranchInfo {
                id: 1,
                name: "Main".to_string(),
                fork_point: None,
                tip: 0,
                active: true,
            },
        );
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    fn create_branch_from_cursor(&mut self) -> BranchId {
        let branch_id = self.next_branch_id;
        self.next_branch_id += 1;

        let branch = BranchInfo {
            id: branch_id,
            name: format!("Branch {branch_id}"),
            fork_point: self.cursor,
            tip: self.cursor.unwrap_or(0),
            active: true,
        };

        // Deactivate old branch
        if let Some(old) = self.branches.get_mut(&self.active_branch) {
            old.active = false;
        }

        self.branches.insert(branch_id, branch);
        self.active_branch = branch_id;
        branch_id
    }

    fn prune_oldest_branch(&mut self) {
        // Find the branch with the oldest tip (lowest timestamp), excluding active
        let oldest_branch = self
            .branches
            .values()
            .filter(|b| b.id != self.active_branch)
            .min_by_key(|b| self.nodes.get(&b.tip).map_or(0, |n| n.timestamp))
            .map(|b| b.id);

        if let Some(branch_id) = oldest_branch {
            // Remove nodes that belong exclusively to this branch
            let branch_nodes: Vec<NodeId> = self
                .nodes
                .values()
                .filter(|n| n.branch_id == branch_id)
                .map(|n| n.id)
                .collect();

            for node_id in branch_nodes {
                // Remove from parent's children
                if let Some(node) = self.nodes.get(&node_id) {
                    let parent = node.parent;
                    if let Some(pid) = parent {
                        if let Some(parent_node) = self.nodes.get_mut(&pid) {
                            parent_node.children.retain(|&c| c != node_id);
                        }
                    }
                }
                self.nodes.remove(&node_id);
            }

            self.branches.remove(&branch_id);
        }
    }

    /// Create a named branch at the current cursor position.
    ///
    /// This does not push any action — it just forks a new named branch
    /// from wherever the cursor currently sits.
    pub fn create_named_branch(&mut self, name: &str) -> BranchId {
        let branch_id = self.next_branch_id;
        self.next_branch_id += 1;

        let branch = BranchInfo {
            id: branch_id,
            name: name.to_string(),
            fork_point: self.cursor,
            tip: self.cursor.unwrap_or(0),
            active: false,
        };
        self.branches.insert(branch_id, branch);
        branch_id
    }

    /// Rename an existing branch.
    pub fn rename_branch(&mut self, branch_id: BranchId, new_name: &str) -> bool {
        if let Some(branch) = self.branches.get_mut(&branch_id) {
            branch.name = new_name.to_string();
            true
        } else {
            false
        }
    }

    /// Get the path (sequence of actions) from root to the tip of a given branch.
    #[must_use]
    pub fn path_to_branch_tip(&self, branch_id: BranchId) -> Vec<&HistoryAction> {
        let tip = match self.branches.get(&branch_id) {
            Some(b) if b.tip != 0 => b.tip,
            _ => return Vec::new(),
        };
        let mut path = Vec::new();
        let mut current = Some(tip);
        while let Some(id) = current {
            if let Some(node) = self.nodes.get(&id) {
                path.push(&node.action);
                current = node.parent;
            } else {
                break;
            }
        }
        path.reverse();
        path
    }

    /// Diff two branches: returns actions unique to each branch.
    ///
    /// Returns `(only_in_a, only_in_b, common_prefix_len)`.
    #[must_use]
    pub fn diff_branches(
        &self,
        branch_a: BranchId,
        branch_b: BranchId,
    ) -> (Vec<&HistoryAction>, Vec<&HistoryAction>, usize) {
        let path_a = self.path_to_branch_tip(branch_a);
        let path_b = self.path_to_branch_tip(branch_b);

        let common_len = path_a
            .iter()
            .zip(path_b.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let only_a = path_a[common_len..].to_vec();
        let only_b = path_b[common_len..].to_vec();
        (only_a, only_b, common_len)
    }

    /// Conflict type when merging branches.
    fn detect_conflicts(
        actions_a: &[&HistoryAction],
        actions_b: &[&HistoryAction],
    ) -> Vec<MergeConflict> {
        let mut conflicts = Vec::new();
        for (i, a) in actions_a.iter().enumerate() {
            for (j, b) in actions_b.iter().enumerate() {
                if Self::actions_conflict(a, b) {
                    conflicts.push(MergeConflict {
                        branch_a_index: i,
                        branch_b_index: j,
                        description: format!(
                            "Conflict: '{}' vs '{}'",
                            a.description(),
                            b.description()
                        ),
                    });
                }
            }
        }
        conflicts
    }

    /// Check if two actions conflict (operate on the same clip).
    fn actions_conflict(a: &HistoryAction, b: &HistoryAction) -> bool {
        let clip_a = Self::affected_clip_id(a);
        let clip_b = Self::affected_clip_id(b);
        match (clip_a, clip_b) {
            (Some(ca), Some(cb)) => ca == cb,
            _ => false,
        }
    }

    /// Extract the primary clip ID affected by an action.
    fn affected_clip_id(action: &HistoryAction) -> Option<u64> {
        match action {
            HistoryAction::AddClip { clip_id, .. }
            | HistoryAction::RemoveClip { clip_id, .. }
            | HistoryAction::MoveClip { clip_id, .. }
            | HistoryAction::TrimClip { clip_id, .. } => Some(*clip_id),
            HistoryAction::SplitClip { original_id, .. } => Some(*original_id),
            HistoryAction::AddTransition { clip_a, .. } => Some(*clip_a),
            HistoryAction::Compound { actions, .. } => {
                actions.first().and_then(|a| Self::affected_clip_id(a))
            }
        }
    }

    /// Attempt to merge `source_branch` into the active branch.
    ///
    /// Returns the merge result with any detected conflicts.
    pub fn merge_branch(&mut self, source_branch: BranchId) -> MergeResult {
        let (only_active, only_source, common_len) =
            self.diff_branches(self.active_branch, source_branch);

        let conflicts = Self::detect_conflicts(&only_active, &only_source);

        if !conflicts.is_empty() {
            return MergeResult {
                success: false,
                merged_actions: 0,
                conflicts,
            };
        }

        // No conflicts — append source-only actions to active branch.
        let source_path = self.path_to_branch_tip(source_branch);
        let actions_to_merge: Vec<HistoryAction> =
            source_path.into_iter().skip(common_len).cloned().collect();

        let count = actions_to_merge.len();
        for action in actions_to_merge {
            self.push(action);
        }

        MergeResult {
            success: true,
            merged_actions: count,
            conflicts: Vec::new(),
        }
    }

    /// Prune a specific branch by ID (not the active branch).
    ///
    /// Returns `true` if the branch was removed.
    pub fn prune_branch(&mut self, branch_id: BranchId) -> bool {
        if branch_id == self.active_branch {
            return false;
        }
        if !self.branches.contains_key(&branch_id) {
            return false;
        }

        // Collect nodes exclusive to this branch (not shared by children of other branches).
        let shared_nodes: std::collections::HashSet<NodeId> = self
            .nodes
            .values()
            .filter(|n| n.branch_id != branch_id)
            .map(|n| n.id)
            .collect();

        let to_remove: Vec<NodeId> = self
            .nodes
            .values()
            .filter(|n| n.branch_id == branch_id && !shared_nodes.contains(&n.id))
            .map(|n| n.id)
            .collect();

        for node_id in &to_remove {
            if let Some(node) = self.nodes.get(node_id) {
                let parent = node.parent;
                if let Some(pid) = parent {
                    if let Some(parent_node) = self.nodes.get_mut(&pid) {
                        parent_node.children.retain(|&c| c != *node_id);
                    }
                }
            }
            self.nodes.remove(node_id);
        }

        self.branches.remove(&branch_id);
        true
    }

    /// Prune all branches except the active one and optionally a set of kept branches.
    pub fn prune_all_except(&mut self, keep: &[BranchId]) -> usize {
        let to_prune: Vec<BranchId> = self
            .branches
            .keys()
            .copied()
            .filter(|&id| id != self.active_branch && !keep.contains(&id))
            .collect();
        let count = to_prune.len();
        for id in to_prune {
            self.prune_branch(id);
        }
        count
    }

    /// Serialize the history tree to a compact binary format.
    ///
    /// Format: node_count(u64) then for each node:
    ///   id(u64), parent(u64, 0=none), branch_id(u64), timestamp(u64),
    ///   action_tag(u8), action-specific fields.
    #[must_use]
    pub fn serialize_compact(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let node_count = self.nodes.len() as u64;
        buf.extend_from_slice(&node_count.to_le_bytes());

        // Write branch count and branches.
        let branch_count = self.branches.len() as u64;
        buf.extend_from_slice(&branch_count.to_le_bytes());
        for branch in self.branches.values() {
            buf.extend_from_slice(&branch.id.to_le_bytes());
            Self::write_string(&mut buf, &branch.name);
            buf.extend_from_slice(&branch.fork_point.unwrap_or(0).to_le_bytes());
            buf.extend_from_slice(&branch.tip.to_le_bytes());
            buf.push(u8::from(branch.active));
        }

        // Write active branch and cursor.
        buf.extend_from_slice(&self.active_branch.to_le_bytes());
        buf.extend_from_slice(&self.cursor.unwrap_or(0).to_le_bytes());
        buf.extend_from_slice(&self.next_node_id.to_le_bytes());
        buf.extend_from_slice(&self.next_branch_id.to_le_bytes());
        buf.extend_from_slice(&self.timestamp_counter.to_le_bytes());
        buf.extend_from_slice(&(self.max_nodes as u64).to_le_bytes());

        // Write nodes.
        for node in self.nodes.values() {
            buf.extend_from_slice(&node.id.to_le_bytes());
            buf.extend_from_slice(&node.parent.unwrap_or(0).to_le_bytes());
            buf.extend_from_slice(&node.branch_id.to_le_bytes());
            buf.extend_from_slice(&node.timestamp.to_le_bytes());
            let child_count = node.children.len() as u64;
            buf.extend_from_slice(&child_count.to_le_bytes());
            for &child in &node.children {
                buf.extend_from_slice(&child.to_le_bytes());
            }
            Self::serialize_action(&mut buf, &node.action);
        }

        buf
    }

    /// Deserialize a history tree from compact binary format.
    pub fn deserialize_compact(data: &[u8]) -> Option<Self> {
        let mut pos = 0usize;

        let node_count = Self::read_u64(data, &mut pos)?;
        let branch_count = Self::read_u64(data, &mut pos)?;

        let mut branches = HashMap::new();
        for _ in 0..branch_count {
            let id = Self::read_u64(data, &mut pos)?;
            let name = Self::read_string(data, &mut pos)?;
            let fork_point_raw = Self::read_u64(data, &mut pos)?;
            let fork_point = if fork_point_raw == 0 {
                None
            } else {
                Some(fork_point_raw)
            };
            let tip = Self::read_u64(data, &mut pos)?;
            let active = Self::read_u8(data, &mut pos)? != 0;
            branches.insert(
                id,
                BranchInfo {
                    id,
                    name,
                    fork_point,
                    tip,
                    active,
                },
            );
        }

        let active_branch = Self::read_u64(data, &mut pos)?;
        let cursor_raw = Self::read_u64(data, &mut pos)?;
        let cursor = if cursor_raw == 0 {
            None
        } else {
            Some(cursor_raw)
        };
        let next_node_id = Self::read_u64(data, &mut pos)?;
        let next_branch_id = Self::read_u64(data, &mut pos)?;
        let timestamp_counter = Self::read_u64(data, &mut pos)?;
        let max_nodes = Self::read_u64(data, &mut pos)? as usize;

        let mut nodes = HashMap::new();
        for _ in 0..node_count {
            let id = Self::read_u64(data, &mut pos)?;
            let parent_raw = Self::read_u64(data, &mut pos)?;
            let parent = if parent_raw == 0 {
                None
            } else {
                Some(parent_raw)
            };
            let branch_id = Self::read_u64(data, &mut pos)?;
            let timestamp = Self::read_u64(data, &mut pos)?;
            let child_count = Self::read_u64(data, &mut pos)?;
            let mut children = Vec::new();
            for _ in 0..child_count {
                children.push(Self::read_u64(data, &mut pos)?);
            }
            let action = Self::deserialize_action(data, &mut pos)?;
            nodes.insert(
                id,
                HistoryNode {
                    id,
                    action,
                    parent,
                    children,
                    branch_id,
                    timestamp,
                },
            );
        }

        Some(Self {
            nodes,
            branches,
            cursor,
            active_branch,
            next_node_id,
            next_branch_id,
            timestamp_counter,
            max_nodes,
        })
    }

    // ── Serialization helpers ───────────────────────────────────────────

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(bytes);
    }

    fn read_string(data: &[u8], pos: &mut usize) -> Option<String> {
        let len = Self::read_u64(data, pos)? as usize;
        if *pos + len > data.len() {
            return None;
        }
        let s = std::str::from_utf8(&data[*pos..*pos + len])
            .ok()?
            .to_string();
        *pos += len;
        Some(s)
    }

    fn read_u64(data: &[u8], pos: &mut usize) -> Option<u64> {
        if *pos + 8 > data.len() {
            return None;
        }
        let val = u64::from_le_bytes(data[*pos..*pos + 8].try_into().ok()?);
        *pos += 8;
        Some(val)
    }

    fn read_i64(data: &[u8], pos: &mut usize) -> Option<i64> {
        if *pos + 8 > data.len() {
            return None;
        }
        let val = i64::from_le_bytes(data[*pos..*pos + 8].try_into().ok()?);
        *pos += 8;
        Some(val)
    }

    fn read_u8(data: &[u8], pos: &mut usize) -> Option<u8> {
        if *pos >= data.len() {
            return None;
        }
        let val = data[*pos];
        *pos += 1;
        Some(val)
    }

    fn serialize_action(buf: &mut Vec<u8>, action: &HistoryAction) {
        match action {
            HistoryAction::AddClip {
                clip_id,
                track_index,
            } => {
                buf.push(0);
                buf.extend_from_slice(&clip_id.to_le_bytes());
                buf.extend_from_slice(&(*track_index as u64).to_le_bytes());
            }
            HistoryAction::RemoveClip {
                clip_id,
                track_index,
            } => {
                buf.push(1);
                buf.extend_from_slice(&clip_id.to_le_bytes());
                buf.extend_from_slice(&(*track_index as u64).to_le_bytes());
            }
            HistoryAction::MoveClip {
                clip_id,
                old_start,
                new_start,
            } => {
                buf.push(2);
                buf.extend_from_slice(&clip_id.to_le_bytes());
                buf.extend_from_slice(&old_start.to_le_bytes());
                buf.extend_from_slice(&new_start.to_le_bytes());
            }
            HistoryAction::TrimClip {
                clip_id,
                old_in,
                old_out,
                new_in,
                new_out,
            } => {
                buf.push(3);
                buf.extend_from_slice(&clip_id.to_le_bytes());
                buf.extend_from_slice(&old_in.to_le_bytes());
                buf.extend_from_slice(&old_out.to_le_bytes());
                buf.extend_from_slice(&new_in.to_le_bytes());
                buf.extend_from_slice(&new_out.to_le_bytes());
            }
            HistoryAction::SplitClip {
                original_id,
                new_id,
                position,
            } => {
                buf.push(4);
                buf.extend_from_slice(&original_id.to_le_bytes());
                buf.extend_from_slice(&new_id.to_le_bytes());
                buf.extend_from_slice(&position.to_le_bytes());
            }
            HistoryAction::AddTransition {
                clip_a,
                clip_b,
                duration,
            } => {
                buf.push(5);
                buf.extend_from_slice(&clip_a.to_le_bytes());
                buf.extend_from_slice(&clip_b.to_le_bytes());
                buf.extend_from_slice(&duration.to_le_bytes());
            }
            HistoryAction::Compound {
                description,
                actions,
            } => {
                buf.push(6);
                Self::write_string(buf, description);
                buf.extend_from_slice(&(actions.len() as u64).to_le_bytes());
                for a in actions {
                    Self::serialize_action(buf, a);
                }
            }
        }
    }

    fn deserialize_action(data: &[u8], pos: &mut usize) -> Option<HistoryAction> {
        let tag = Self::read_u8(data, pos)?;
        match tag {
            0 => {
                let clip_id = Self::read_u64(data, pos)?;
                let track_index = Self::read_u64(data, pos)? as usize;
                Some(HistoryAction::AddClip {
                    clip_id,
                    track_index,
                })
            }
            1 => {
                let clip_id = Self::read_u64(data, pos)?;
                let track_index = Self::read_u64(data, pos)? as usize;
                Some(HistoryAction::RemoveClip {
                    clip_id,
                    track_index,
                })
            }
            2 => {
                let clip_id = Self::read_u64(data, pos)?;
                let old_start = Self::read_i64(data, pos)?;
                let new_start = Self::read_i64(data, pos)?;
                Some(HistoryAction::MoveClip {
                    clip_id,
                    old_start,
                    new_start,
                })
            }
            3 => {
                let clip_id = Self::read_u64(data, pos)?;
                let old_in = Self::read_i64(data, pos)?;
                let old_out = Self::read_i64(data, pos)?;
                let new_in = Self::read_i64(data, pos)?;
                let new_out = Self::read_i64(data, pos)?;
                Some(HistoryAction::TrimClip {
                    clip_id,
                    old_in,
                    old_out,
                    new_in,
                    new_out,
                })
            }
            4 => {
                let original_id = Self::read_u64(data, pos)?;
                let new_id = Self::read_u64(data, pos)?;
                let position = Self::read_i64(data, pos)?;
                Some(HistoryAction::SplitClip {
                    original_id,
                    new_id,
                    position,
                })
            }
            5 => {
                let clip_a = Self::read_u64(data, pos)?;
                let clip_b = Self::read_u64(data, pos)?;
                let duration = Self::read_i64(data, pos)?;
                Some(HistoryAction::AddTransition {
                    clip_a,
                    clip_b,
                    duration,
                })
            }
            6 => {
                let description = Self::read_string(data, pos)?;
                let count = Self::read_u64(data, pos)? as usize;
                let mut actions = Vec::with_capacity(count);
                for _ in 0..count {
                    actions.push(Self::deserialize_action(data, pos)?);
                }
                Some(HistoryAction::Compound {
                    description,
                    actions,
                })
            }
            _ => None,
        }
    }

    /// Get a node by ID.
    #[must_use]
    pub fn get_node(&self, node_id: NodeId) -> Option<&HistoryNode> {
        self.nodes.get(&node_id)
    }

    /// Get branch info by ID.
    #[must_use]
    pub fn get_branch(&self, branch_id: BranchId) -> Option<&BranchInfo> {
        self.branches.get(&branch_id)
    }
}

/// Conflict detected during a branch merge.
#[derive(Debug, Clone)]
pub struct MergeConflict {
    /// Index in branch-A's unique actions.
    pub branch_a_index: usize,
    /// Index in branch-B's unique actions.
    pub branch_b_index: usize,
    /// Human-readable description.
    pub description: String,
}

/// Result of a branch merge operation.
#[derive(Debug)]
pub struct MergeResult {
    /// Whether the merge succeeded.
    pub success: bool,
    /// Number of actions merged.
    pub merged_actions: usize,
    /// Conflicts that prevented merge (empty if success).
    pub conflicts: Vec<MergeConflict>,
}

impl Default for HistoryTree {
    fn default() -> Self {
        Self::new(1000)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn add_action() -> HistoryAction {
        HistoryAction::AddClip {
            clip_id: 1,
            track_index: 0,
        }
    }

    fn move_action(clip_id: u64) -> HistoryAction {
        HistoryAction::MoveClip {
            clip_id,
            old_start: 0,
            new_start: 1000,
        }
    }

    #[test]
    fn test_empty_tree() {
        let tree = HistoryTree::new(100);
        assert!(!tree.can_undo());
        assert!(!tree.can_redo());
        assert_eq!(tree.node_count(), 0);
        assert_eq!(tree.branch_count(), 1); // Main branch
    }

    #[test]
    fn test_push_and_undo() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        assert!(tree.can_undo());
        assert!(!tree.can_redo());

        let action = tree.undo();
        assert!(action.is_some());
        assert_eq!(
            action.as_ref().expect("should undo").description(),
            "Add clip"
        );
        assert!(!tree.can_undo());
        assert!(tree.can_redo());
    }

    #[test]
    fn test_undo_redo_cycle() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        tree.push(move_action(1));

        tree.undo(); // undo move
        tree.undo(); // undo add
        assert!(!tree.can_undo());

        tree.redo(); // redo add
        assert!(tree.can_undo());
        assert!(tree.can_redo());

        tree.redo(); // redo move
        assert!(!tree.can_redo());
    }

    #[test]
    fn test_branching() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        let _node2 = tree.push(move_action(1));

        // Undo to after first action
        tree.undo();

        // Push a different action - should create a new branch
        tree.push(HistoryAction::RemoveClip {
            clip_id: 1,
            track_index: 0,
        });

        assert_eq!(tree.branch_count(), 2);
        assert_eq!(tree.node_count(), 3);
    }

    #[test]
    fn test_switch_branch() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        tree.push(move_action(1));

        tree.undo();

        // This creates branch 2
        tree.push(HistoryAction::RemoveClip {
            clip_id: 1,
            track_index: 0,
        });

        // Switch back to branch 1
        assert!(tree.switch_branch(1));
        assert_eq!(tree.active_branch_id(), 1);

        // Switch to non-existent branch
        assert!(!tree.switch_branch(999));
    }

    #[test]
    fn test_path_to_cursor() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        tree.push(move_action(1));

        let path = tree.path_to_cursor();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].description(), "Add clip");
        assert_eq!(path[1].description(), "Move clip");
    }

    #[test]
    fn test_clear() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        tree.push(move_action(1));
        tree.clear();
        assert_eq!(tree.node_count(), 0);
        assert_eq!(tree.branch_count(), 1);
        assert!(!tree.can_undo());
    }

    #[test]
    fn test_action_descriptions() {
        assert_eq!(
            HistoryAction::AddClip {
                clip_id: 1,
                track_index: 0
            }
            .description(),
            "Add clip"
        );
        assert_eq!(
            HistoryAction::RemoveClip {
                clip_id: 1,
                track_index: 0
            }
            .description(),
            "Remove clip"
        );
        assert_eq!(
            HistoryAction::MoveClip {
                clip_id: 1,
                old_start: 0,
                new_start: 100
            }
            .description(),
            "Move clip"
        );
        assert_eq!(
            HistoryAction::TrimClip {
                clip_id: 1,
                old_in: 0,
                old_out: 100,
                new_in: 10,
                new_out: 90
            }
            .description(),
            "Trim clip"
        );
        assert_eq!(
            HistoryAction::SplitClip {
                original_id: 1,
                new_id: 2,
                position: 500
            }
            .description(),
            "Split clip"
        );
        assert_eq!(
            HistoryAction::AddTransition {
                clip_a: 1,
                clip_b: 2,
                duration: 30
            }
            .description(),
            "Add transition"
        );
    }

    #[test]
    fn test_action_inverse() {
        let add = HistoryAction::AddClip {
            clip_id: 1,
            track_index: 0,
        };
        let inv = add.inverse();
        assert_eq!(inv.description(), "Remove clip");

        let mv = HistoryAction::MoveClip {
            clip_id: 1,
            old_start: 0,
            new_start: 1000,
        };
        if let HistoryAction::MoveClip {
            old_start,
            new_start,
            ..
        } = mv.inverse()
        {
            assert_eq!(old_start, 1000);
            assert_eq!(new_start, 0);
        } else {
            panic!("Expected MoveClip inverse");
        }
    }

    #[test]
    fn test_compound_action() {
        let compound = HistoryAction::Compound {
            description: "Ripple delete".to_string(),
            actions: vec![
                HistoryAction::RemoveClip {
                    clip_id: 1,
                    track_index: 0,
                },
                HistoryAction::MoveClip {
                    clip_id: 2,
                    old_start: 5000,
                    new_start: 0,
                },
            ],
        };
        assert_eq!(compound.description(), "Ripple delete");

        let inv = compound.inverse();
        if let HistoryAction::Compound { actions, .. } = &inv {
            // Inverse reverses order and inverts each
            assert_eq!(actions.len(), 2);
            assert_eq!(actions[0].description(), "Move clip");
            assert_eq!(actions[1].description(), "Add clip");
        } else {
            panic!("Expected Compound inverse");
        }
    }

    #[test]
    fn test_max_nodes_pruning() {
        let mut tree = HistoryTree::new(5);
        for i in 0..3 {
            tree.push(HistoryAction::AddClip {
                clip_id: i,
                track_index: 0,
            });
        }
        // Undo and branch
        tree.undo();
        tree.undo();
        for i in 10..15 {
            tree.push(HistoryAction::AddClip {
                clip_id: i,
                track_index: 0,
            });
        }
        // Tree should have pruned to stay near max_nodes
        assert!(tree.node_count() <= 10); // generous bound
    }

    #[test]
    fn test_branch_list() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        tree.push(move_action(1));
        // Undo one step (cursor moves to first action, which is not the tip)
        tree.undo();
        // Push a different action -- cursor is behind tip, so a new branch is created
        tree.push(HistoryAction::RemoveClip {
            clip_id: 99,
            track_index: 0,
        });

        let branches = tree.branch_list();
        assert!(
            branches.len() >= 2,
            "expected at least 2 branches, got {}",
            branches.len()
        );
    }

    #[test]
    fn test_current_action() {
        let mut tree = HistoryTree::new(100);
        assert!(tree.current_action().is_none());
        tree.push(add_action());
        assert!(tree.current_action().is_some());
        assert_eq!(
            tree.current_action().expect("should exist").description(),
            "Add clip"
        );
    }

    #[test]
    fn test_default_tree() {
        let tree = HistoryTree::default();
        assert_eq!(tree.node_count(), 0);
    }

    // ── Named branch tests ─────────────────────────────────────────────

    #[test]
    fn test_create_named_branch() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        let bid = tree.create_named_branch("experiment");
        assert!(tree.get_branch(bid).is_some());
        assert_eq!(
            tree.get_branch(bid).map(|b| b.name.as_str()),
            Some("experiment")
        );
        assert!(tree.branch_count() >= 2);
    }

    #[test]
    fn test_rename_branch() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        let bid = tree.create_named_branch("old_name");
        assert!(tree.rename_branch(bid, "new_name"));
        assert_eq!(
            tree.get_branch(bid).map(|b| b.name.as_str()),
            Some("new_name")
        );
        assert!(!tree.rename_branch(999, "nope"));
    }

    #[test]
    fn test_switch_to_named_branch_and_push() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        let bid = tree.create_named_branch("alt");
        assert!(tree.switch_branch(bid));
        tree.push(move_action(2));
        assert_eq!(tree.node_count(), 2);
        let path = tree.path_to_cursor();
        assert_eq!(path.len(), 2);
    }

    // ── Diff branches tests ────────────────────────────────────────────

    #[test]
    fn test_diff_branches_identical() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        // Both branches share the same single node.
        let bid = tree.create_named_branch("copy");
        let (only_a, only_b, common) = tree.diff_branches(tree.active_branch_id(), bid);
        assert!(only_a.is_empty());
        assert!(only_b.is_empty());
        assert_eq!(common, 1);
    }

    #[test]
    fn test_diff_branches_diverged() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action()); // node 1, branch 1
        tree.push(move_action(1)); // node 2, branch 1

        // Undo and create diverging branch
        tree.undo();
        tree.push(HistoryAction::RemoveClip {
            clip_id: 1,
            track_index: 0,
        }); // node 3, branch 2

        let branch2 = tree.active_branch_id();
        let (only_1, only_2, common) = tree.diff_branches(1, branch2);
        assert_eq!(common, 1); // shared: AddClip
        assert_eq!(only_1.len(), 1); // MoveClip
        assert_eq!(only_2.len(), 1); // RemoveClip
    }

    // ── Merge tests ────────────────────────────────────────────────────

    #[test]
    fn test_merge_no_conflict() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action()); // clip 1
        tree.push(move_action(1)); // clip 1

        tree.undo(); // back to after add

        // Create branch with action on different clip
        tree.push(HistoryAction::AddClip {
            clip_id: 99,
            track_index: 1,
        });
        let source = tree.active_branch_id();

        // Switch back to main and merge
        tree.switch_branch(1);
        let result = tree.merge_branch(source);
        assert!(result.success);
        assert_eq!(result.merged_actions, 1);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn test_merge_with_conflict() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action()); // clip 1

        // Branch 1 moves clip 1
        tree.push(move_action(1));
        tree.undo();

        // Branch 2 also operates on clip 1
        tree.push(HistoryAction::TrimClip {
            clip_id: 1,
            old_in: 0,
            old_out: 100,
            new_in: 10,
            new_out: 90,
        });
        let source = tree.active_branch_id();

        tree.switch_branch(1);
        let result = tree.merge_branch(source);
        assert!(!result.success);
        assert!(!result.conflicts.is_empty());
    }

    // ── Prune tests ────────────────────────────────────────────────────

    #[test]
    fn test_prune_branch() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        tree.push(move_action(1));
        tree.undo();

        // Create branch 2
        tree.push(HistoryAction::RemoveClip {
            clip_id: 1,
            track_index: 0,
        });
        let branch2 = tree.active_branch_id();

        tree.switch_branch(1);
        let before = tree.node_count();
        assert!(tree.prune_branch(branch2));
        assert!(tree.node_count() < before);
        assert!(!tree.branches.contains_key(&branch2));
    }

    #[test]
    fn test_prune_active_branch_fails() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        assert!(!tree.prune_branch(tree.active_branch_id()));
    }

    #[test]
    fn test_prune_all_except() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        tree.push(move_action(1));
        tree.undo();
        tree.push(HistoryAction::RemoveClip {
            clip_id: 99,
            track_index: 0,
        });
        let b2 = tree.active_branch_id();
        tree.switch_branch(1);

        let pruned = tree.prune_all_except(&[]);
        assert!(pruned >= 1);
        assert!(!tree.branches.contains_key(&b2));
    }

    // ── Serialization tests ────────────────────────────────────────────

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        tree.push(move_action(1));
        tree.push(HistoryAction::TrimClip {
            clip_id: 1,
            old_in: 0,
            old_out: 100,
            new_in: 10,
            new_out: 90,
        });

        let data = tree.serialize_compact();
        let restored = HistoryTree::deserialize_compact(&data);
        assert!(restored.is_some());
        let restored = restored.expect("deserialize ok");
        assert_eq!(restored.node_count(), tree.node_count());
        assert_eq!(restored.branch_count(), tree.branch_count());
        assert_eq!(restored.active_branch_id(), tree.active_branch_id());
    }

    #[test]
    fn test_serialize_empty_tree() {
        let tree = HistoryTree::new(50);
        let data = tree.serialize_compact();
        let restored = HistoryTree::deserialize_compact(&data);
        assert!(restored.is_some());
        let restored = restored.expect("ok");
        assert_eq!(restored.node_count(), 0);
    }

    #[test]
    fn test_serialize_with_branches() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        tree.push(move_action(1));
        tree.undo();
        tree.push(HistoryAction::RemoveClip {
            clip_id: 2,
            track_index: 0,
        });

        let data = tree.serialize_compact();
        let restored = HistoryTree::deserialize_compact(&data).expect("ok");
        assert_eq!(restored.branch_count(), tree.branch_count());
        assert_eq!(restored.node_count(), tree.node_count());
    }

    #[test]
    fn test_serialize_compound_action() {
        let mut tree = HistoryTree::new(100);
        tree.push(HistoryAction::Compound {
            description: "batch".to_string(),
            actions: vec![add_action(), move_action(1)],
        });
        let data = tree.serialize_compact();
        let restored = HistoryTree::deserialize_compact(&data).expect("ok");
        assert_eq!(restored.node_count(), 1);
        let action = restored.current_action().expect("has action");
        assert_eq!(action.description(), "batch");
    }

    #[test]
    fn test_deserialize_invalid_data() {
        let result = HistoryTree::deserialize_compact(&[0, 1, 2]);
        assert!(result.is_none());
    }

    #[test]
    fn test_path_to_branch_tip() {
        let mut tree = HistoryTree::new(100);
        tree.push(add_action());
        tree.push(move_action(1));
        let path = tree.path_to_branch_tip(1);
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_get_node() {
        let mut tree = HistoryTree::new(100);
        let nid = tree.push(add_action());
        assert!(tree.get_node(nid).is_some());
        assert!(tree.get_node(999).is_none());
    }
}
