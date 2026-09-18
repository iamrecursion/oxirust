//! Edit history for undo/redo functionality.
//!
//! This module provides two complementary undo/redo systems:
//!
//! * [`EditHistory`] — a classic linear undo/redo stack keyed on [`EditAction`]
//!   values.  New actions erase the "future" (redo stack is cleared).
//!
//! * [`BranchingHistory`] — a generic, state-snapshot history *tree*.  Each
//!   [`push`] creates a child of the current node.  Calling [`undo`] and then
//!   [`push`] with a different value forks a new branch rather than discarding
//!   the old future.  This enables Vim-style "undo tree" / "git-for-edits"
//!   workflows where every edit path is preserved and revisitable.
//!
//! [`push`]: BranchingHistory::push
//! [`undo`]: BranchingHistory::undo

/// An action that was performed on the timeline and can be undone/redone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditAction {
    /// A clip was added to the timeline.
    AddClip {
        /// The ID of the clip that was added.
        clip_id: u64,
    },
    /// A clip was removed from the timeline.
    RemoveClip {
        /// The ID of the clip that was removed.
        clip_id: u64,
    },
    /// A clip was moved to a new position.
    MoveClip {
        /// The ID of the clip that was moved.
        clip_id: u64,
        /// The original start position (in timebase units).
        old_pos: u64,
        /// The new start position (in timebase units).
        new_pos: u64,
    },
    /// A clip was trimmed (in or out point changed).
    TrimClip {
        /// The ID of the clip that was trimmed.
        clip_id: u64,
        /// The original in-point.
        old_in: u64,
        /// The original out-point.
        old_out: u64,
        /// The new in-point.
        new_in: u64,
        /// The new out-point.
        new_out: u64,
    },
    /// A transition was added between two clips.
    AddTransition {
        /// The ID of the first clip (A-side).
        clip_a: u64,
        /// The ID of the second clip (B-side).
        clip_b: u64,
        /// The duration of the transition (in timebase units).
        duration: u64,
    },
}

impl EditAction {
    /// Returns a human-readable description of this action.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::AddClip { .. } => "Add clip",
            Self::RemoveClip { .. } => "Remove clip",
            Self::MoveClip { .. } => "Move clip",
            Self::TrimClip { .. } => "Trim clip",
            Self::AddTransition { .. } => "Add transition",
        }
    }

    /// Returns the inverse of this action (what undo would do).
    #[must_use]
    pub fn inverse(&self) -> Self {
        match *self {
            Self::AddClip { clip_id } => Self::RemoveClip { clip_id },
            Self::RemoveClip { clip_id } => Self::AddClip { clip_id },
            Self::MoveClip {
                clip_id,
                old_pos,
                new_pos,
            } => Self::MoveClip {
                clip_id,
                old_pos: new_pos,
                new_pos: old_pos,
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
            Self::AddTransition {
                clip_a,
                clip_b,
                duration,
            } => Self::AddTransition {
                clip_a,
                clip_b,
                duration,
            },
        }
    }
}

/// Manages the undo/redo history for editing operations.
///
/// Maintains two stacks: an undo stack (past actions) and a redo stack
/// (actions that were undone and can be redone). Each `push` of a new action
/// clears the redo stack.
#[derive(Debug, Clone)]
pub struct EditHistory {
    /// Stack of actions that can be undone (most recent at back).
    undo_stack: Vec<EditAction>,
    /// Stack of actions that can be redone (most recent at back).
    redo_stack: Vec<EditAction>,
    /// Maximum number of actions to keep in the undo stack.
    max_depth: usize,
}

impl EditHistory {
    /// Creates a new `EditHistory` with the given maximum undo depth.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - Maximum number of actions to remember. Once exceeded,
    ///   the oldest entries are dropped.
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_depth,
        }
    }

    /// Pushes a new action onto the undo stack and clears the redo stack.
    ///
    /// If the undo stack exceeds `max_depth`, the oldest action is dropped.
    pub fn push(&mut self, action: EditAction) {
        self.redo_stack.clear();
        self.undo_stack.push(action);
        if self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    /// Pops the most recent action from the undo stack and pushes it onto
    /// the redo stack. Returns the action to be undone.
    pub fn undo(&mut self) -> Option<EditAction> {
        let action = self.undo_stack.pop()?;
        self.redo_stack.push(action.clone());
        Some(action)
    }

    /// Pops the most recent action from the redo stack and pushes it onto
    /// the undo stack. Returns the action to be redone.
    pub fn redo(&mut self) -> Option<EditAction> {
        let action = self.redo_stack.pop()?;
        self.undo_stack.push(action.clone());
        Some(action)
    }

    /// Returns `true` if there are any actions that can be undone.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Returns `true` if there are any actions that can be redone.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Clears both the undo and redo stacks.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Returns the number of actions currently in the undo stack.
    #[must_use]
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Returns the number of actions currently in the redo stack.
    #[must_use]
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Returns the maximum undo depth for this history.
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Returns a slice of all actions in the undo stack (oldest first).
    #[must_use]
    pub fn undo_stack(&self) -> &[EditAction] {
        &self.undo_stack
    }

    /// Returns a slice of all actions in the redo stack (oldest first).
    #[must_use]
    pub fn redo_stack(&self) -> &[EditAction] {
        &self.redo_stack
    }
}

impl Default for EditHistory {
    fn default() -> Self {
        Self::new(100)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BranchingHistory — generic state-snapshot tree
// ─────────────────────────────────────────────────────────────────────────────

/// Identifier for a node in a [`BranchingHistory`] tree.
///
/// Node IDs are monotonically increasing integers starting at `0`.  The root
/// node always has `id == 0`.
pub type NodeId = usize;

/// A single node in the [`BranchingHistory`] tree.
///
/// Each node stores a complete snapshot of the state `T` at the moment it was
/// created, together with bookkeeping for tree traversal.
#[derive(Debug, Clone)]
pub struct HistoryNode<T> {
    /// Unique identifier for this node (index into the `nodes` arena).
    pub id: NodeId,
    /// State snapshot stored at this node.
    pub state: T,
    /// ID of the parent node, or `None` for the root node.
    pub parent: Option<NodeId>,
    /// IDs of child nodes (branches that diverged from this node).
    pub children: Vec<NodeId>,
    /// Human-readable label such as `"Cut clip"` or `"Add filter"`.
    pub label: String,
}

/// A branching undo/redo history tree that stores arbitrary state snapshots.
///
/// Unlike the linear [`EditHistory`] which discards the redo stack whenever a
/// new action is pushed, `BranchingHistory` preserves every alternative editing
/// path.  Calling [`undo`] and then [`push`] with a new state creates a sibling
/// branch rather than overwriting the old timeline.
///
/// # Type parameter
///
/// `T: Clone` — the state type.  A full snapshot is cloned and stored at each
/// node, so `T` should be reasonably inexpensive to clone for large histories.
///
/// # Example
///
/// ```
/// use oximedia_edit::history::BranchingHistory;
///
/// let mut h: BranchingHistory<u32> = BranchingHistory::new(0);
/// let n1 = h.push(1, "step 1");
/// let n2 = h.push(2, "step 2");
///
/// // Undo back to state 1, then push a different state — creates a fork.
/// h.undo();
/// let n2b = h.push(20, "alternate step 2");
///
/// // Undo again to land back on the fork point (state 1), where both
/// // branches are visible as children.
/// h.undo();
///
/// // The old branch (n2 = 2) and the new branch (n2b = 20) both exist.
/// assert_eq!(h.available_branches().len(), 2);
/// ```
///
/// [`undo`]: BranchingHistory::undo
/// [`push`]: BranchingHistory::push
#[derive(Debug, Clone)]
pub struct BranchingHistory<T: Clone> {
    /// Arena-style storage for all nodes.
    ///
    /// Index == node ID: `nodes[id]` is always `Some` for valid IDs.  We use
    /// `Vec<Option<…>>` so removal (if ever needed by a trimming strategy) can
    /// be done in O(1) without re-indexing.
    nodes: Vec<Option<HistoryNode<T>>>,
    /// The node that is currently "active" (most recently applied state).
    current: NodeId,
    /// ID of the root node (always `0`).
    root: NodeId,
}

impl<T: Clone> BranchingHistory<T> {
    /// Create a new branching history tree with the given initial state.
    ///
    /// The initial state is stored in the root node (id `0`) with label
    /// `"initial"`.  The root node has no parent and is the only node in the
    /// tree at construction time.
    #[must_use]
    pub fn new(initial_state: T) -> Self {
        let root_node = HistoryNode {
            id: 0,
            state: initial_state,
            parent: None,
            children: Vec::new(),
            label: "initial".to_string(),
        };
        Self {
            nodes: vec![Some(root_node)],
            current: 0,
            root: 0,
        }
    }

    /// Push a new state as a child of the current node.
    ///
    /// The new node becomes the current node.  If the current node already has
    /// children (i.e. the user has undone some steps and now pushes a different
    /// state), the new state becomes an *additional* branch — existing children
    /// are preserved.
    ///
    /// Returns the [`NodeId`] of the newly created node.
    pub fn push(&mut self, state: T, label: impl Into<String>) -> NodeId {
        let new_id = self.nodes.len();
        let node = HistoryNode {
            id: new_id,
            state,
            parent: Some(self.current),
            children: Vec::new(),
            label: label.into(),
        };
        // Register as a child of the current node.
        if let Some(Some(parent)) = self.nodes.get_mut(self.current) {
            parent.children.push(new_id);
        }
        self.nodes.push(Some(node));
        self.current = new_id;
        new_id
    }

    /// Move the cursor to the parent of the current node (undo one step).
    ///
    /// Returns a reference to the parent state, or `None` if the current node
    /// is the root (nothing to undo).
    pub fn undo(&mut self) -> Option<&T> {
        let parent_id = self.nodes.get(self.current)?.as_ref()?.parent?;
        self.current = parent_id;
        self.nodes.get(self.current)?.as_ref().map(|n| &n.state)
    }

    /// Move the cursor to the **first child** of the current node (redo).
    ///
    /// "First child" is the child with the lowest [`NodeId`] among this node's
    /// children — equivalent to following the most-recently-created branch.
    /// Use [`redo_to_branch`] for explicit branch selection.
    ///
    /// Returns a reference to the child state, or `None` if there are no
    /// children (nothing to redo).
    ///
    /// [`redo_to_branch`]: BranchingHistory::redo_to_branch
    pub fn redo(&mut self) -> Option<&T> {
        let first_child = self
            .nodes
            .get(self.current)?
            .as_ref()?
            .children
            .first()
            .copied()?;
        self.current = first_child;
        self.nodes.get(self.current)?.as_ref().map(|n| &n.state)
    }

    /// Move the cursor to a specific child of the current node.
    ///
    /// `child_id` must be one of the direct children of the current node
    /// (i.e., it must appear in [`available_branches`]).  Returns `None` if
    /// `child_id` is not a valid child.
    ///
    /// [`available_branches`]: BranchingHistory::available_branches
    pub fn redo_to_branch(&mut self, child_id: NodeId) -> Option<&T> {
        let is_valid_child = self
            .nodes
            .get(self.current)?
            .as_ref()?
            .children
            .contains(&child_id);

        if !is_valid_child {
            return None;
        }
        self.current = child_id;
        self.nodes.get(self.current)?.as_ref().map(|n| &n.state)
    }

    /// Return a reference to the state stored at the current node.
    #[must_use]
    pub fn current_state(&self) -> &T {
        // Safety: `current` always points to a valid, non-`None` node.
        self.nodes[self.current]
            .as_ref()
            .map(|n| &n.state)
            .expect("current node must always be valid")
    }

    /// Return the IDs of all direct children of the current node.
    ///
    /// An empty slice means there is nothing to redo.  Multiple entries mean
    /// the current node is a branching point.
    #[must_use]
    pub fn available_branches(&self) -> &[NodeId] {
        self.nodes
            .get(self.current)
            .and_then(|n| n.as_ref())
            .map(|n| n.children.as_slice())
            .unwrap_or(&[])
    }

    /// Return the sequence of node IDs from the root to the current node
    /// (inclusive), ordered root-first.
    ///
    /// This represents the "main timeline" of states that led to the current
    /// position in the tree.
    #[must_use]
    pub fn path_to_current(&self) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut cursor = Some(self.current);
        while let Some(id) = cursor {
            path.push(id);
            cursor = self
                .nodes
                .get(id)
                .and_then(|n| n.as_ref())
                .and_then(|n| n.parent);
        }
        path.reverse();
        path
    }

    /// Return the total number of nodes in the tree (root + all pushes).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }

    /// Return the id of the root node (always `0`).
    #[must_use]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Return the id of the current node.
    #[must_use]
    pub fn current_id(&self) -> NodeId {
        self.current
    }

    /// Borrow the node for the given ID, or `None` if the ID is invalid.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&HistoryNode<T>> {
        self.nodes.get(id)?.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_history_is_empty() {
        let h = EditHistory::new(50);
        assert!(!h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.undo_count(), 0);
        assert_eq!(h.redo_count(), 0);
    }

    #[test]
    fn test_push_enables_undo() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 1 });
        assert!(h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn test_undo_returns_action() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 42 });
        let action = h.undo();
        assert!(action.is_some());
        assert_eq!(
            action.expect("test expectation failed"),
            EditAction::AddClip { clip_id: 42 }
        );
    }

    #[test]
    fn test_undo_enables_redo() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 1 });
        h.undo();
        assert!(!h.can_undo());
        assert!(h.can_redo());
    }

    #[test]
    fn test_redo_returns_action() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::RemoveClip { clip_id: 7 });
        h.undo();
        let redone = h.redo();
        assert!(redone.is_some());
        assert_eq!(
            redone.expect("test expectation failed"),
            EditAction::RemoveClip { clip_id: 7 }
        );
    }

    #[test]
    fn test_push_clears_redo_stack() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 1 });
        h.undo();
        assert!(h.can_redo());
        h.push(EditAction::AddClip { clip_id: 2 });
        assert!(!h.can_redo(), "pushing a new action must clear redo stack");
    }

    #[test]
    fn test_max_depth_enforced() {
        let mut h = EditHistory::new(3);
        h.push(EditAction::AddClip { clip_id: 1 });
        h.push(EditAction::AddClip { clip_id: 2 });
        h.push(EditAction::AddClip { clip_id: 3 });
        h.push(EditAction::AddClip { clip_id: 4 }); // should evict clip_id 1
        assert_eq!(h.undo_count(), 3);
        // The oldest remaining action should be clip_id=2
        assert_eq!(h.undo_stack()[0], EditAction::AddClip { clip_id: 2 });
    }

    #[test]
    fn test_clear_resets_both_stacks() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 1 });
        h.push(EditAction::AddClip { clip_id: 2 });
        h.undo();
        h.clear();
        assert!(!h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn test_undo_empty_returns_none() {
        let mut h = EditHistory::new(50);
        assert!(h.undo().is_none());
    }

    #[test]
    fn test_redo_empty_returns_none() {
        let mut h = EditHistory::new(50);
        assert!(h.redo().is_none());
    }

    #[test]
    fn test_move_clip_action() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::MoveClip {
            clip_id: 5,
            old_pos: 100,
            new_pos: 200,
        });
        let action = h.undo().expect("action should be valid");
        assert_eq!(
            action,
            EditAction::MoveClip {
                clip_id: 5,
                old_pos: 100,
                new_pos: 200,
            }
        );
    }

    #[test]
    fn test_trim_clip_action() {
        let action = EditAction::TrimClip {
            clip_id: 10,
            old_in: 0,
            old_out: 1000,
            new_in: 100,
            new_out: 900,
        };
        let inv = action.inverse();
        assert_eq!(
            inv,
            EditAction::TrimClip {
                clip_id: 10,
                old_in: 100,
                old_out: 900,
                new_in: 0,
                new_out: 1000,
            }
        );
    }

    #[test]
    fn test_action_descriptions() {
        assert_eq!(EditAction::AddClip { clip_id: 1 }.description(), "Add clip");
        assert_eq!(
            EditAction::RemoveClip { clip_id: 1 }.description(),
            "Remove clip"
        );
        assert_eq!(
            EditAction::MoveClip {
                clip_id: 1,
                old_pos: 0,
                new_pos: 0
            }
            .description(),
            "Move clip"
        );
        assert_eq!(
            EditAction::TrimClip {
                clip_id: 1,
                old_in: 0,
                old_out: 0,
                new_in: 0,
                new_out: 0
            }
            .description(),
            "Trim clip"
        );
        assert_eq!(
            EditAction::AddTransition {
                clip_a: 1,
                clip_b: 2,
                duration: 30
            }
            .description(),
            "Add transition"
        );
    }

    #[test]
    fn test_multiple_undo_redo_cycle() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 1 });
        h.push(EditAction::AddClip { clip_id: 2 });
        h.push(EditAction::AddClip { clip_id: 3 });

        // Undo all three
        assert_eq!(
            h.undo().expect("undo should succeed"),
            EditAction::AddClip { clip_id: 3 }
        );
        assert_eq!(
            h.undo().expect("undo should succeed"),
            EditAction::AddClip { clip_id: 2 }
        );
        assert_eq!(
            h.undo().expect("undo should succeed"),
            EditAction::AddClip { clip_id: 1 }
        );
        assert!(!h.can_undo());

        // Redo all three
        assert_eq!(
            h.redo().expect("redo should succeed"),
            EditAction::AddClip { clip_id: 1 }
        );
        assert_eq!(
            h.redo().expect("redo should succeed"),
            EditAction::AddClip { clip_id: 2 }
        );
        assert_eq!(
            h.redo().expect("redo should succeed"),
            EditAction::AddClip { clip_id: 3 }
        );
        assert!(!h.can_redo());
    }

    #[test]
    fn test_default_max_depth() {
        let h = EditHistory::default();
        assert_eq!(h.max_depth(), 100);
    }

    // ─── BranchingHistory tests ──────────────────────────────────────────────

    /// Push 3 states, undo×2, redo×2 → back at state 3 (linear path, no fork).
    #[test]
    fn test_branching_history_linear() {
        let mut h: BranchingHistory<u32> = BranchingHistory::new(0);

        let _n1 = h.push(1, "step 1");
        let _n2 = h.push(2, "step 2");
        let n3 = h.push(3, "step 3");

        // Sanity: at node 3.
        assert_eq!(*h.current_state(), 3);
        assert_eq!(h.current_id(), n3);

        // Undo×2 — land on state 1.
        let after_undo1 = h.undo().expect("undo 1 from 3");
        assert_eq!(*after_undo1, 2);
        let after_undo2 = h.undo().expect("undo 2 from 2");
        assert_eq!(*after_undo2, 1);
        assert_eq!(*h.current_state(), 1);

        // Redo×2 — back to state 3.
        let after_redo1 = h.redo().expect("redo 1");
        assert_eq!(*after_redo1, 2);
        let after_redo2 = h.redo().expect("redo 2");
        assert_eq!(*after_redo2, 3);
        assert_eq!(*h.current_state(), 3);
        assert_eq!(h.current_id(), n3);
    }

    /// Push state2, undo, push state2b → two branches from state1.
    #[test]
    fn test_branching_history_fork() {
        let mut h: BranchingHistory<u32> = BranchingHistory::new(0);

        let _n1 = h.push(1, "step 1");
        let n2a = h.push(2, "step 2a");

        // Undo back to state 1.
        h.undo().expect("undo to state 1");
        assert_eq!(*h.current_state(), 1);

        // Push a different state — forks from state 1.
        let n2b = h.push(20, "step 2b (alternate)");

        // There are now 2 branches from node n1.
        // n1 (state=1) has children [n2a, n2b].
        let branches = h.available_branches();
        // Currently at n2b (most recently pushed), so available_branches are its children (none).
        assert_eq!(branches.len(), 0, "n2b has no children yet");

        // Go back to state 1 and confirm two children.
        h.undo().expect("undo to state 1 again");
        let branches_from_1 = h.available_branches();
        assert_eq!(
            branches_from_1.len(),
            2,
            "state 1 should have two branches: {n2a} and {n2b}"
        );
        assert!(branches_from_1.contains(&n2a));
        assert!(branches_from_1.contains(&n2b));
    }

    /// After a fork, redo_to_branch picks the correct child.
    #[test]
    fn test_branching_history_redo_to_branch() {
        let mut h: BranchingHistory<u32> = BranchingHistory::new(0);

        let _n1 = h.push(1, "step 1");
        let n2a = h.push(2, "step 2a");

        h.undo(); // back at state 1
        let n2b = h.push(20, "step 2b");

        h.undo(); // back at state 1 again

        // Redo to the first branch (n2a, state=2).
        let state_a = h
            .redo_to_branch(n2a)
            .expect("redo_to_branch n2a should succeed");
        assert_eq!(*state_a, 2, "expected state of n2a = 2");
        assert_eq!(h.current_id(), n2a);

        // Navigate back and redo to the second branch (n2b, state=20).
        h.undo(); // back at state 1
        let state_b = h
            .redo_to_branch(n2b)
            .expect("redo_to_branch n2b should succeed");
        assert_eq!(*state_b, 20, "expected state of n2b = 20");
        assert_eq!(h.current_id(), n2b);

        // Invalid child ID returns None.
        h.undo(); // back at state 1
        let bad = h.redo_to_branch(9999);
        assert!(bad.is_none(), "invalid child should return None");
    }

    /// path_to_current returns correct root-to-current sequence.
    #[test]
    fn test_branching_history_path_to_current() {
        let mut h: BranchingHistory<u32> = BranchingHistory::new(0);

        let n1 = h.push(1, "step 1");
        let n2 = h.push(2, "step 2");
        let n3 = h.push(3, "step 3");

        // Full path: root(0) → n1 → n2 → n3.
        let path = h.path_to_current();
        assert_eq!(path, vec![h.root(), n1, n2, n3]);

        // Undo×2 → path: root(0) → n1.
        h.undo();
        h.undo();
        let path_after_undo = h.path_to_current();
        assert_eq!(path_after_undo, vec![h.root(), n1]);

        // Undo to root → path: just root(0).
        h.undo();
        let path_at_root = h.path_to_current();
        assert_eq!(path_at_root, vec![h.root()]);
    }

    /// After a fork, available_branches at the branch point has 2 entries.
    #[test]
    fn test_branching_history_available_branches() {
        let mut h: BranchingHistory<u32> = BranchingHistory::new(0);

        h.push(1, "step 1");
        let n2a = h.push(2, "step 2a");

        h.undo(); // back at state 1
        let n2b = h.push(20, "step 2b");

        h.undo(); // back at state 1 (the fork point)
        let branches = h.available_branches();

        assert_eq!(
            branches.len(),
            2,
            "fork point should expose exactly 2 branches"
        );
        // Both n2a and n2b must be reachable.
        assert!(
            branches.contains(&n2a),
            "n2a ({n2a}) must be in available_branches"
        );
        assert!(
            branches.contains(&n2b),
            "n2b ({n2b}) must be in available_branches"
        );

        // node_count: root + step1 + step2a + step2b = 4.
        assert_eq!(h.node_count(), 4);
    }
}
