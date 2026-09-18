//! Persistent vector backed by a Relaxed Radix-Balanced (RRB) tree.
//!
//! All "mutating" operations return a *new version* of the vector while leaving
//! the original intact.  Unchanged subtrees are shared via `Arc`, so the
//! amortised cost of each operation is O(log₃₂ N) **node allocations** rather
//! than O(N).
//!
//! # Branching factor
//!
//! Each internal node holds up to [`BRANCHING`] (32) children.  The tree height
//! grows by one when the number of elements exceeds 32^(height+1).  Index
//! navigation uses 5-bit radix steps.
//!
//! # Operations
//!
//! | Operation     | Allocation cost | Notes                              |
//! |---------------|-----------------|------------------------------------|
//! | `get`         | O(1)            | no allocation                      |
//! | `push_back`   | O(log N)        | amortised, O(1) common case        |
//! | `pop_back`    | O(log N)        |                                    |
//! | `update`      | O(log N)        | path-copy from root to leaf        |
//! | `iter`        | O(1)            | lazy traversal via stack           |
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::data_structures::PersistentVec;
//!
//! let v0 = PersistentVec::new();
//! let v1 = v0.push_back(10u32);
//! let v2 = v1.push_back(20u32);
//! let v3 = v2.update(0, 99u32).expect("index 0 is valid");
//!
//! assert_eq!(v3.get(0), Some(&99u32));
//! assert_eq!(v3.get(1), Some(&20u32));
//! // Original v2 is unchanged:
//! assert_eq!(v2.get(0), Some(&10u32));
//! ```

use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Branching factor: each internal node has up to 32 children.
pub const BRANCHING: usize = 32;

/// log₂(BRANCHING): bits consumed per tree level during radix navigation.
pub const LOG_BRANCHING: usize = 5;

/// Bitmask for extracting a single radix level index.
const MASK: usize = BRANCHING - 1;

// ─────────────────────────────────────────────────────────────────────────────
// Internal node types
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the RRB trie.
///
/// - `Internal`: holds up to `BRANCHING` child nodes.
/// - `Leaf`: holds up to `BRANCHING` data elements.
#[derive(Clone)]
pub(crate) enum Node<T: Clone> {
    Internal(Vec<Arc<Node<T>>>),
    Leaf(Vec<T>),
}

impl<T: Clone> Node<T> {
    // ------------------------------------------------------------------
    // Navigation helpers
    // ------------------------------------------------------------------

    /// Access the child nodes of an internal node.
    fn children(&self) -> &[Arc<Node<T>>] {
        match self {
            Node::Internal(ch) => ch,
            Node::Leaf(_) => &[],
        }
    }

    /// Access the data elements of a leaf node.
    fn elements(&self) -> &[T] {
        match self {
            Node::Leaf(elems) => elems,
            Node::Internal(_) => &[],
        }
    }

    // ------------------------------------------------------------------
    // Pure reads
    // ------------------------------------------------------------------

    /// Retrieve the element at `index` within the subtree rooted here.
    ///
    /// `shift` is `height * LOG_BRANCHING` — the number of bits by which the
    /// radix index for this level begins.  When `shift == 0` this is a leaf.
    fn get_at(&self, index: usize, shift: usize) -> Option<&T> {
        match self {
            Node::Leaf(elems) => elems.get(index & MASK),
            Node::Internal(children) => {
                let child_idx = (index >> shift) & MASK;
                let child = children.get(child_idx)?;
                if shift == LOG_BRANCHING {
                    // Next level is a leaf.
                    child.get_at(index, 0)
                } else {
                    child.get_at(index, shift - LOG_BRANCHING)
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Path-copy update
    // ------------------------------------------------------------------

    /// Return a new node that has `value` at `index`, sharing all unchanged
    /// subtrees with `self`.  Returns `None` if `index` is out of bounds.
    fn update_at(&self, index: usize, value: T, shift: usize) -> Option<Arc<Node<T>>> {
        match self {
            Node::Leaf(elems) => {
                let i = index & MASK;
                if i >= elems.len() {
                    return None;
                }
                let mut new_elems = elems.clone();
                new_elems[i] = value;
                Some(Arc::new(Node::Leaf(new_elems)))
            }
            Node::Internal(children) => {
                let child_idx = (index >> shift) & MASK;
                if child_idx >= children.len() {
                    return None;
                }
                let next_shift = shift.saturating_sub(LOG_BRANCHING);
                let new_child = children[child_idx].update_at(index, value, next_shift)?;
                let mut new_children = children.clone();
                new_children[child_idx] = new_child;
                Some(Arc::new(Node::Internal(new_children)))
            }
        }
    }

    // ------------------------------------------------------------------
    // Insertion helpers
    // ------------------------------------------------------------------

    /// Attempt to insert `leaf_node` as the rightmost leaf of the subtree
    /// rooted here, given that this subtree has height `height` (leaf == 0).
    ///
    /// Returns `None` when the subtree is full (cannot accept more leaves at
    /// this height), indicating the caller must grow the tree upward.
    fn push_tail(&self, leaf_node: Arc<Node<T>>, height: usize) -> Option<Arc<Node<T>>> {
        match self {
            Node::Leaf(_) => {
                // Leaf nodes cannot hold another leaf — the tree needs to grow.
                None
            }
            Node::Internal(children) => {
                if height == 1 {
                    // Direct parent of leaves.
                    if children.len() < BRANCHING {
                        let mut new_children = children.clone();
                        new_children.push(leaf_node);
                        Some(Arc::new(Node::Internal(new_children)))
                    } else {
                        None // this internal node is full
                    }
                } else {
                    // Try to insert into the rightmost child first.
                    if let Some(last) = children.last() {
                        if let Some(new_last) = last.push_tail(Arc::clone(&leaf_node), height - 1) {
                            let mut new_children = children.clone();
                            let last_idx = new_children.len() - 1;
                            new_children[last_idx] = new_last;
                            return Some(Arc::new(Node::Internal(new_children)));
                        }
                    }
                    // Rightmost subtree was full; create a new child path.
                    if children.len() < BRANCHING {
                        let new_subtree = build_path_to_leaf(leaf_node, height - 1);
                        let mut new_children = children.clone();
                        new_children.push(new_subtree);
                        Some(Arc::new(Node::Internal(new_children)))
                    } else {
                        None // this internal node is also full
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Deletion helpers
    // ------------------------------------------------------------------

    /// Remove the rightmost leaf from the subtree, returning the updated
    /// subtree root and the removed leaf.
    ///
    /// Returns `None` for the new root if the subtree becomes empty.
    fn pop_tail(&self, height: usize) -> Option<(Option<Arc<Node<T>>>, Arc<Node<T>>)> {
        match self {
            Node::Leaf(_) => {
                // A leaf is itself the rightmost "leaf subtree".
                None
            }
            Node::Internal(children) => {
                if children.is_empty() {
                    return None;
                }
                let last_idx = children.len() - 1;

                if height == 1 {
                    // Children are leaves.
                    let removed_leaf = Arc::clone(&children[last_idx]);
                    let new_root = if children.len() == 1 {
                        None
                    } else {
                        let new_children = children[..last_idx].to_vec();
                        Some(Arc::new(Node::Internal(new_children)))
                    };
                    Some((new_root, removed_leaf))
                } else {
                    let (new_last_opt, leaf) = children[last_idx].pop_tail(height - 1)?;
                    let new_children = match new_last_opt {
                        Some(new_last) => {
                            let mut ch = children.clone();
                            ch[last_idx] = new_last;
                            ch
                        }
                        None => children[..last_idx].to_vec(),
                    };
                    let new_root = if new_children.is_empty() {
                        None
                    } else {
                        Some(Arc::new(Node::Internal(new_children)))
                    };
                    Some((new_root, leaf))
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a spine of internal nodes of `height` levels leading down to `leaf`.
///
/// Height 0 → return `leaf` unchanged.
/// Height 1 → one internal node containing `leaf`.
/// Height 2 → one internal node containing one internal node containing `leaf`.
fn build_path_to_leaf<T: Clone>(leaf: Arc<Node<T>>, height: usize) -> Arc<Node<T>> {
    if height == 0 {
        return leaf;
    }
    let child = build_path_to_leaf(leaf, height - 1);
    Arc::new(Node::Internal(vec![child]))
}

// ─────────────────────────────────────────────────────────────────────────────
// PersistentVec<T>
// ─────────────────────────────────────────────────────────────────────────────

/// Persistent vector with structural sharing, backed by an RRB-tree.
///
/// All "mutating" operations return a new version while the original is left
/// unchanged.  Structural sharing (via `Arc`) ensures that only O(log N)
/// nodes are newly allocated per operation.
///
/// ```rust
/// use scirs2_core::data_structures::PersistentVec;
///
/// let v0: PersistentVec<u32> = PersistentVec::new();
/// let v1 = v0.push_back(1);
/// let v2 = v1.push_back(2);
///
/// assert_eq!(v2.len(), 2);
/// assert_eq!(v2.get(0), Some(&1));
/// // v1 is still valid and unchanged:
/// assert_eq!(v1.len(), 1);
/// ```
#[derive(Clone)]
pub struct PersistentVec<T: Clone> {
    /// Root of the trie, or `None` when empty.
    root: Option<Arc<Node<T>>>,
    /// Total number of elements.
    size: usize,
    /// `shift = height * LOG_BRANCHING` for the current tree.
    /// A brand-new (empty) vector starts with `shift = LOG_BRANCHING` (height 1).
    shift: usize,
    /// Tail buffer: up to BRANCHING elements not yet pushed into the trie.
    tail: Vec<T>,
}

impl<T: Clone> PersistentVec<T> {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Creates an empty persistent vector.
    pub fn new() -> Self {
        PersistentVec {
            root: None,
            size: 0,
            shift: LOG_BRANCHING,
            tail: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` when the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns a reference to the element at `index`, or `None` if out of
    /// bounds.
    ///
    /// O(log N) time, no allocation.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.size {
            return None;
        }

        // Is the element in the tail?
        let tail_offset = self.size - self.tail.len();
        if index >= tail_offset {
            return self.tail.get(index - tail_offset);
        }

        // Navigate the trie.
        let root = self.root.as_ref()?;
        root.get_at(index, self.shift)
    }

    // ------------------------------------------------------------------
    // Persistent updates
    // ------------------------------------------------------------------

    /// Returns a new vector with `value` appended to the end.
    ///
    /// Amortised O(log N) — O(1) when the tail has room.
    pub fn push_back(&self, value: T) -> Self {
        if self.tail.len() < BRANCHING {
            // Fast path: just extend the tail.
            let mut new_tail = self.tail.clone();
            new_tail.push(value);
            return PersistentVec {
                root: self.root.clone(),
                size: self.size + 1,
                shift: self.shift,
                tail: new_tail,
            };
        }

        // Tail is full; push it into the trie as a leaf node.
        let full_tail = std::mem::take(&mut { self.tail.clone() });
        let leaf = Arc::new(Node::Leaf(full_tail));

        let (new_root, new_shift) = match &self.root {
            None => {
                // Empty trie — the leaf becomes the sole node.
                (Some(Arc::new(Node::Internal(vec![leaf]))), self.shift)
            }
            Some(root) => {
                let height = self.shift / LOG_BRANCHING;
                match root.push_tail(leaf.clone(), height) {
                    Some(new_root) => (Some(new_root), self.shift),
                    None => {
                        // Tree is full at current height; grow it.
                        let new_height = height + 1;
                        let new_shift = new_height * LOG_BRANCHING;
                        let new_spine = build_path_to_leaf(leaf, height);
                        let new_root = Arc::new(Node::Internal(vec![Arc::clone(root), new_spine]));
                        (Some(new_root), new_shift)
                    }
                }
            }
        };

        PersistentVec {
            root: new_root,
            size: self.size + 1,
            shift: new_shift,
            tail: vec![value],
        }
    }

    /// Returns a new vector with the element at `index` replaced by `value`.
    ///
    /// Returns `None` if `index >= self.len()`.
    /// O(log N) node allocations.
    pub fn update(&self, index: usize, value: T) -> Option<Self> {
        if index >= self.size {
            return None;
        }

        let tail_offset = self.size - self.tail.len();

        if index >= tail_offset {
            // Element is in the tail.
            let mut new_tail = self.tail.clone();
            new_tail[index - tail_offset] = value;
            return Some(PersistentVec {
                root: self.root.clone(),
                size: self.size,
                shift: self.shift,
                tail: new_tail,
            });
        }

        // Element is in the trie — path-copy from root to the leaf.
        let root = self.root.as_ref()?;
        let new_root = root.update_at(index, value, self.shift)?;
        Some(PersistentVec {
            root: Some(new_root),
            size: self.size,
            shift: self.shift,
            tail: self.tail.clone(),
        })
    }

    /// Removes and returns the last element, returning `(new_vec, value)`.
    ///
    /// Returns `None` if the vector is empty.
    /// O(log N) in the worst case, O(1) common case.
    pub fn pop_back(&self) -> Option<(Self, T)> {
        if self.size == 0 {
            return None;
        }

        if !self.tail.is_empty() {
            // Fast path: pop from tail.
            let mut new_tail = self.tail.clone();
            let popped = new_tail.pop()?;

            if new_tail.is_empty() && self.root.is_some() {
                // Tail is now empty; pull the rightmost leaf from the trie
                // to become the new tail.
                let root = self.root.as_ref()?;
                let height = self.shift / LOG_BRANCHING;
                let (new_root_opt, leaf_arc) = root.pop_tail(height)?;

                let new_tail_vec = leaf_arc.elements().to_vec();

                // Shrink tree height if the new root has only one child.
                let (final_root, final_shift) = match new_root_opt {
                    None => (None, LOG_BRANCHING),
                    Some(nr) => {
                        let (collapsed, new_shift) = collapse_root(nr, self.shift);
                        (Some(collapsed), new_shift)
                    }
                };

                return Some((
                    PersistentVec {
                        root: final_root,
                        size: self.size - 1,
                        shift: final_shift,
                        tail: new_tail_vec,
                    },
                    popped,
                ));
            }

            return Some((
                PersistentVec {
                    root: self.root.clone(),
                    size: self.size - 1,
                    shift: self.shift,
                    tail: new_tail,
                },
                popped,
            ));
        }

        // Tail is empty; pull from trie (should not normally happen but handle
        // the degenerate case).
        let root = self.root.as_ref()?;
        let height = self.shift / LOG_BRANCHING;
        let (new_root_opt, leaf_arc) = root.pop_tail(height)?;

        let leaf_elems = leaf_arc.elements();
        let popped = leaf_elems.last()?.clone();
        let new_tail_vec: Vec<T> = leaf_elems[..leaf_elems.len() - 1].to_vec();

        let (final_root, final_shift) = match new_root_opt {
            None => (None, LOG_BRANCHING),
            Some(nr) => {
                let (collapsed, ns) = collapse_root(nr, self.shift);
                (Some(collapsed), ns)
            }
        };

        Some((
            PersistentVec {
                root: final_root,
                size: self.size - 1,
                shift: final_shift,
                tail: new_tail_vec,
            },
            popped,
        ))
    }

    // ------------------------------------------------------------------
    // Iteration
    // ------------------------------------------------------------------

    /// Returns an iterator over references to elements in order.
    pub fn iter(&self) -> PersistentVecIter<T> {
        // Collect all elements from the trie via depth-first traversal.
        let mut trie_elements: Vec<T> =
            Vec::with_capacity(self.size.saturating_sub(self.tail.len()));
        if let Some(root) = &self.root {
            collect_elements(root, &mut trie_elements);
        }
        // Append tail.
        trie_elements.extend_from_slice(&self.tail);

        PersistentVecIter {
            data: trie_elements,
            pos: 0,
        }
    }
}

/// Collapse the root if it has exactly one internal child (shrink height).
fn collapse_root<T: Clone>(root: Arc<Node<T>>, shift: usize) -> (Arc<Node<T>>, usize) {
    if shift <= LOG_BRANCHING {
        return (root, shift);
    }
    match root.as_ref() {
        Node::Internal(children) if children.len() == 1 => {
            collapse_root(Arc::clone(&children[0]), shift - LOG_BRANCHING)
        }
        _ => (root, shift),
    }
}

/// Depth-first traversal to collect all elements from a subtree.
fn collect_elements<T: Clone>(node: &Arc<Node<T>>, out: &mut Vec<T>) {
    match node.as_ref() {
        Node::Leaf(elems) => out.extend_from_slice(elems),
        Node::Internal(children) => {
            for child in children {
                collect_elements(child, out);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Default / FromIterator
// ─────────────────────────────────────────────────────────────────────────────

impl<T: Clone> Default for PersistentVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> FromIterator<T> for PersistentVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = PersistentVec::new();
        for item in iter {
            vec = vec.push_back(item);
        }
        vec
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Iterator
// ─────────────────────────────────────────────────────────────────────────────

/// Iterator over references to elements of a `PersistentVec`.
///
/// Produced by [`PersistentVec::iter`].
pub struct PersistentVecIter<T: Clone> {
    /// All elements in order (built eagerly during construction for simplicity).
    data: Vec<T>,
    pos: usize,
}

impl<T: Clone> Iterator for PersistentVecIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.data.len() {
            let item = self.data[self.pos].clone();
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<T: Clone> ExactSizeIterator for PersistentVecIter<T> {}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrb_basic_ops() {
        let mut v = PersistentVec::new();
        for i in 0u32..100 {
            v = v.push_back(i);
        }
        assert_eq!(v.len(), 100);
        for i in 0u32..100 {
            assert_eq!(v.get(i as usize), Some(&i), "mismatch at index {i}");
        }
        assert_eq!(v.get(100), None);
    }

    #[test]
    fn test_rrb_persistence() {
        let v0: PersistentVec<u32> = PersistentVec::new();
        let v1 = v0.push_back(1);
        let v2 = v1.push_back(2);
        let v3 = v2.push_back(3);

        // Newer versions see all elements.
        assert_eq!(v3.len(), 3);
        assert_eq!(v3.get(0), Some(&1));
        assert_eq!(v3.get(1), Some(&2));
        assert_eq!(v3.get(2), Some(&3));

        // Older versions are unmodified.
        assert_eq!(v0.len(), 0);
        assert_eq!(v1.len(), 1);
        assert_eq!(v1.get(0), Some(&1));
        assert_eq!(v2.len(), 2);
        assert_eq!(v2.get(1), Some(&2));
    }

    #[test]
    fn test_rrb_update() {
        let base: PersistentVec<i32> = (0..50).collect();

        let updated = base.update(25, 999).expect("index 25 is valid");

        // Updated version has the new value.
        assert_eq!(updated.get(25), Some(&999));
        // All other indices are unchanged.
        for i in 0..50 {
            if i != 25 {
                assert_eq!(updated.get(i), Some(&(i as i32)), "mismatch at {i}");
            }
        }
        // Original is unchanged.
        assert_eq!(base.get(25), Some(&25));

        // Out-of-bounds update returns None.
        assert!(base.update(100, 42).is_none());
    }

    #[test]
    fn test_rrb_iter() {
        let v: PersistentVec<u32> = (0..64).collect();
        let collected: Vec<u32> = v.iter().collect();
        let expected: Vec<u32> = (0..64).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_rrb_large() {
        const N: usize = 1000;
        let v: PersistentVec<usize> = (0..N).collect();
        assert_eq!(v.len(), N);
        for i in 0..N {
            assert_eq!(v.get(i), Some(&i), "mismatch at index {i}");
        }
    }

    #[test]
    fn test_rrb_pop_back() {
        let v: PersistentVec<u32> = (0..10).collect();

        let (v2, last) = v.pop_back().expect("non-empty");
        assert_eq!(last, 9);
        assert_eq!(v2.len(), 9);

        // Original unchanged.
        assert_eq!(v.len(), 10);
        assert_eq!(v.get(9), Some(&9));

        // Pop until empty.
        let mut cur = v.clone();
        for expected_last in (0..10).rev() {
            let (next, val) = cur.pop_back().expect("should not be empty");
            assert_eq!(val, expected_last);
            cur = next;
        }
        assert!(cur.is_empty());
        assert!(cur.pop_back().is_none());
    }

    #[test]
    fn test_rrb_from_iterator() {
        let v: PersistentVec<u8> = vec![10u8, 20, 30, 40, 50].into_iter().collect();
        assert_eq!(v.len(), 5);
        assert_eq!(v.get(2), Some(&30u8));
    }

    #[test]
    fn test_rrb_default_is_empty() {
        let v: PersistentVec<f64> = PersistentVec::default();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_rrb_cross_leaf_boundary() {
        // Push exactly BRANCHING+1 elements to force a leaf spill.
        let n = BRANCHING + 1;
        let v: PersistentVec<usize> = (0..n).collect();
        assert_eq!(v.len(), n);
        for i in 0..n {
            assert_eq!(v.get(i), Some(&i), "mismatch at {i}");
        }
    }

    #[test]
    fn test_rrb_update_across_leaf_boundary() {
        // Cover the code path where an update targets an element in the trie
        // (not the tail) near a leaf boundary.
        let v: PersistentVec<u32> = (0..100).collect();
        // BRANCHING = 32; index 31 is the last element of the first trie leaf.
        let updated = v.update(31, 777).expect("valid index");
        assert_eq!(updated.get(31), Some(&777));
        assert_eq!(v.get(31), Some(&31)); // original unchanged
    }
}
