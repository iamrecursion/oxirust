//! Augmented interval tree for O(log n) schedule overlap detection.
//!
//! # Data Structure
//!
//! An **augmented red-black BST** (Cormen, Leiserson, Rivest, Stein — *Introduction
//! to Algorithms*, Chapter 14.3) keyed on interval *low endpoint* with an
//! additional `max` field per node storing the maximum high endpoint in the
//! subtree.  This enables the following operations in **O(log n)** time:
//!
//! | Operation            | Time       | Space  |
//! |----------------------|------------|--------|
//! | `insert`             | O(log n)   | O(n)   |
//! | `remove`             | O(log n)   | —      |
//! | `query_overlapping`  | O(log n)   | —      |
//! | `query_all_overlapping` | O(k log n) | O(k) |
//! | `iter_sorted`        | O(n)       | O(n)   |
//!
//! A plain `Vec` gives O(n) for overlap queries, which is acceptable for
//! tens of entries but degrades for schedules with hundreds of daily playlists.
//!
//! # Usage
//!
//! ```
//! use oximedia_playlist::interval_tree::{IntervalTree, Interval};
//!
//! let mut tree: IntervalTree<u64, &str> = IntervalTree::new();
//! tree.insert(Interval::new(100, 200), "morning show");
//! tree.insert(Interval::new(300, 400), "news");
//! tree.insert(Interval::new(150, 250), "overlap");
//!
//! // Does anything overlap [180, 220)?
//! let hits = tree.query_all_overlapping(180, 220);
//! assert_eq!(hits.len(), 2); // "morning show" and "overlap"
//! ```

use std::cmp::Ordering;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A half-open interval `[low, high)`.
///
/// Type `K` is the key type (e.g., `u64` for epoch milliseconds, `i64` for
/// signed offsets, `f64` for seconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval<K> {
    /// Low endpoint (inclusive).
    pub low: K,
    /// High endpoint (exclusive).
    pub high: K,
}

impl<K: PartialOrd + Copy> Interval<K> {
    /// Create a new interval.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `low >= high`.
    #[must_use]
    pub fn new(low: K, high: K) -> Self {
        debug_assert!(
            low < high,
            "Interval::new: low must be strictly less than high"
        );
        Self { low, high }
    }

    /// Returns `true` when `self` overlaps `[query_low, query_high)`.
    ///
    /// Two half-open intervals `[a, b)` and `[c, d)` overlap iff `a < d && c < b`.
    #[must_use]
    pub fn overlaps(&self, query_low: K, query_high: K) -> bool {
        self.low < query_high && query_low < self.high
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal node
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Color {
    Red,
    Black,
}

/// An index into the node arena.
type NodeIdx = usize;
const NIL: NodeIdx = 0; // sentinel

#[derive(Debug, Clone)]
struct Node<K, V> {
    interval: Interval<K>,
    value: V,
    /// Maximum `high` endpoint in this subtree (augmentation).
    max: K,
    color: Color,
    left: NodeIdx,
    right: NodeIdx,
    parent: NodeIdx,
}

impl<K: Copy, V> Node<K, V> {
    fn new(interval: Interval<K>, value: V, color: Color) -> Self {
        Self {
            max: interval.high,
            interval,
            value,
            color,
            left: NIL,
            right: NIL,
            parent: NIL,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IntervalTree
// ─────────────────────────────────────────────────────────────────────────────

/// Augmented red-black interval tree.
///
/// Keys are `Interval<K>` ordered by `low` endpoint.  Values are of type `V`.
///
/// The tree does **not** guarantee unique intervals; inserting the same
/// `Interval` twice stores two distinct entries with potentially different
/// values.
pub struct IntervalTree<K, V> {
    /// Arena of nodes.  Index 0 is the NIL sentinel.
    arena: Vec<Node<K, V>>,
    root: NodeIdx,
    len: usize,
}

impl<K, V> Default for IntervalTree<K, V>
where
    K: Copy + PartialOrd + Ord + Default,
    V: Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> IntervalTree<K, V>
where
    K: Copy + PartialOrd + Ord + Default,
    V: Clone + Default,
{
    /// Create an empty interval tree.
    ///
    /// Allocates one sentinel NIL node at index 0.
    #[must_use]
    pub fn new() -> Self {
        // Index 0 is the NIL sentinel: black, with max = K::default().
        let nil_node = Node {
            interval: Interval {
                low: K::default(),
                high: K::default(),
            },
            value: V::default(),
            max: K::default(),
            color: Color::Black,
            left: NIL,
            right: NIL,
            parent: NIL,
        };
        Self {
            arena: vec![nil_node],
            root: NIL,
            len: 0,
        }
    }

    /// Return the number of intervals stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return `true` if the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert `(interval, value)` into the tree.
    ///
    /// O(log n) amortised.
    pub fn insert(&mut self, interval: Interval<K>, value: V) {
        let z = self.alloc_node(Node::new(interval, value, Color::Red));
        self.rb_insert(z);
        self.len += 1;
    }

    /// Remove the first interval that equals `interval` exactly (comparing
    /// both `low` and `high`).  Returns the associated value, or `None` if not
    /// found.
    ///
    /// O(log n).
    pub fn remove(&mut self, interval: Interval<K>) -> Option<V> {
        let z = self.find_node(interval)?;
        let val = self.arena[z].value.clone();
        self.rb_delete(z);
        self.len -= 1;
        Some(val)
    }

    /// Return the value of any interval that overlaps `[low, high)`, or `None`
    /// if no such interval exists.
    ///
    /// O(log n) (CLRS §14.3 single-overlap search).
    #[must_use]
    pub fn query_overlapping(&self, low: K, high: K) -> Option<&V> {
        let node = self.search_any(self.root, low, high)?;
        Some(&self.arena[node].value)
    }

    /// Return the interval associated with the first overlapping node, if any.
    #[must_use]
    pub fn query_overlapping_interval(&self, low: K, high: K) -> Option<Interval<K>> {
        let node = self.search_any(self.root, low, high)?;
        Some(self.arena[node].interval)
    }

    /// Return **all** values whose intervals overlap `[low, high)`.
    ///
    /// O(k log n) where k is the number of results.
    #[must_use]
    pub fn query_all_overlapping(&self, low: K, high: K) -> Vec<&V> {
        let mut result = Vec::new();
        self.search_all(self.root, low, high, &mut result);
        result
    }

    /// Return all `(interval, value)` pairs whose intervals overlap `[low, high)`.
    #[must_use]
    pub fn query_all_overlapping_pairs(&self, low: K, high: K) -> Vec<(Interval<K>, &V)> {
        let mut result = Vec::new();
        self.search_all_pairs(self.root, low, high, &mut result);
        result
    }

    /// Return all stored `(interval, value)` pairs in sorted order by `low`.
    #[must_use]
    pub fn iter_sorted(&self) -> Vec<(Interval<K>, &V)> {
        let mut out = Vec::with_capacity(self.len);
        self.inorder(self.root, &mut out);
        out
    }

    // ─── private helpers ────────────────────────────────────────────────

    fn alloc_node(&mut self, n: Node<K, V>) -> NodeIdx {
        self.arena.push(n);
        self.arena.len() - 1
    }

    fn max_of(&self, idx: NodeIdx) -> K {
        self.arena[idx].max
    }

    fn update_max(&mut self, idx: NodeIdx) {
        if idx == NIL {
            return;
        }
        let l = self.arena[idx].left;
        let r = self.arena[idx].right;
        let mut m = self.arena[idx].interval.high;
        if l != NIL && self.arena[l].max > m {
            m = self.arena[l].max;
        }
        if r != NIL && self.arena[r].max > m {
            m = self.arena[r].max;
        }
        self.arena[idx].max = m;
    }

    // CLRS §13.2 left-rotate
    fn left_rotate(&mut self, x: NodeIdx) {
        let y = self.arena[x].right;
        debug_assert!(y != NIL, "left_rotate called on node with NIL right child");
        // Turn y's left sub-tree into x's right sub-tree.
        let y_left = self.arena[y].left;
        self.arena[x].right = y_left;
        if y_left != NIL {
            self.arena[y_left].parent = x;
        }
        // Link x's parent to y.
        let xp = self.arena[x].parent;
        self.arena[y].parent = xp;
        if xp == NIL {
            self.root = y;
        } else if x == self.arena[xp].left {
            self.arena[xp].left = y;
        } else {
            self.arena[xp].right = y;
        }
        self.arena[y].left = x;
        self.arena[x].parent = y;
        // Update max fields bottom-up.
        self.update_max(x);
        self.update_max(y);
    }

    // CLRS §13.2 right-rotate
    fn right_rotate(&mut self, y: NodeIdx) {
        let x = self.arena[y].left;
        debug_assert!(x != NIL, "right_rotate called on node with NIL left child");
        let x_right = self.arena[x].right;
        self.arena[y].left = x_right;
        if x_right != NIL {
            self.arena[x_right].parent = y;
        }
        let yp = self.arena[y].parent;
        self.arena[x].parent = yp;
        if yp == NIL {
            self.root = x;
        } else if y == self.arena[yp].left {
            self.arena[yp].left = x;
        } else {
            self.arena[yp].right = x;
        }
        self.arena[x].right = y;
        self.arena[y].parent = x;
        self.update_max(y);
        self.update_max(x);
    }

    // CLRS §13.3 RB-INSERT
    fn rb_insert(&mut self, z: NodeIdx) {
        let mut y = NIL;
        let mut x = self.root;
        while x != NIL {
            y = x;
            // Update max on the way down.
            let z_high = self.arena[z].interval.high;
            if z_high > self.arena[x].max {
                self.arena[x].max = z_high;
            }
            if self.arena[z].interval.low < self.arena[x].interval.low {
                x = self.arena[x].left;
            } else {
                x = self.arena[x].right;
            }
        }
        self.arena[z].parent = y;
        if y == NIL {
            self.root = z;
        } else if self.arena[z].interval.low < self.arena[y].interval.low {
            self.arena[y].left = z;
        } else {
            self.arena[y].right = z;
        }
        // Fixup colour.
        self.rb_insert_fixup(z);
    }

    fn rb_insert_fixup(&mut self, mut z: NodeIdx) {
        while self.arena[self.arena[z].parent].color == Color::Red {
            let zp = self.arena[z].parent;
            let zpp = self.arena[zp].parent;
            if zp == self.arena[zpp].left {
                let y = self.arena[zpp].right; // uncle
                if self.arena[y].color == Color::Red {
                    // Case 1
                    self.arena[zp].color = Color::Black;
                    self.arena[y].color = Color::Black;
                    self.arena[zpp].color = Color::Red;
                    z = zpp;
                } else {
                    if z == self.arena[zp].right {
                        // Case 2
                        z = zp;
                        self.left_rotate(z);
                    }
                    // Case 3
                    let zp2 = self.arena[z].parent;
                    let zpp2 = self.arena[zp2].parent;
                    self.arena[zp2].color = Color::Black;
                    self.arena[zpp2].color = Color::Red;
                    self.right_rotate(zpp2);
                }
            } else {
                let y = self.arena[zpp].left; // uncle
                if self.arena[y].color == Color::Red {
                    self.arena[zp].color = Color::Black;
                    self.arena[y].color = Color::Black;
                    self.arena[zpp].color = Color::Red;
                    z = zpp;
                } else {
                    if z == self.arena[zp].left {
                        z = zp;
                        self.right_rotate(z);
                    }
                    let zp2 = self.arena[z].parent;
                    let zpp2 = self.arena[zp2].parent;
                    self.arena[zp2].color = Color::Black;
                    self.arena[zpp2].color = Color::Red;
                    self.left_rotate(zpp2);
                }
            }
        }
        self.arena[self.root].color = Color::Black;
    }

    // CLRS §13.4 RB-TRANSPLANT
    fn transplant(&mut self, u: NodeIdx, v: NodeIdx) {
        let up = self.arena[u].parent;
        if up == NIL {
            self.root = v;
        } else if u == self.arena[up].left {
            self.arena[up].left = v;
        } else {
            self.arena[up].right = v;
        }
        self.arena[v].parent = up;
    }

    fn tree_minimum(&self, mut x: NodeIdx) -> NodeIdx {
        while self.arena[x].left != NIL {
            x = self.arena[x].left;
        }
        x
    }

    // CLRS §13.4 RB-DELETE
    fn rb_delete(&mut self, z: NodeIdx) {
        let mut y = z;
        let mut y_original_color = self.arena[y].color;
        let x: NodeIdx;

        if self.arena[z].left == NIL {
            x = self.arena[z].right;
            self.transplant(z, self.arena[z].right);
        } else if self.arena[z].right == NIL {
            x = self.arena[z].left;
            self.transplant(z, self.arena[z].left);
        } else {
            y = self.tree_minimum(self.arena[z].right);
            y_original_color = self.arena[y].color;
            x = self.arena[y].right;
            if self.arena[y].parent == z {
                self.arena[x].parent = y;
            } else {
                self.transplant(y, self.arena[y].right);
                let zr = self.arena[z].right;
                self.arena[y].right = zr;
                self.arena[zr].parent = y;
            }
            self.transplant(z, y);
            let zl = self.arena[z].left;
            self.arena[y].left = zl;
            self.arena[zl].parent = y;
            self.arena[y].color = self.arena[z].color;
            // Propagate max upwards from y.
            self.update_max(y);
        }

        // Propagate max upwards from parent of deleted/moved node.
        let mut cur = self.arena[x].parent;
        while cur != NIL {
            self.update_max(cur);
            cur = self.arena[cur].parent;
        }

        if y_original_color == Color::Black {
            self.rb_delete_fixup(x);
        }
    }

    fn rb_delete_fixup(&mut self, mut x: NodeIdx) {
        while x != self.root && self.arena[x].color == Color::Black {
            let xp = self.arena[x].parent;
            if x == self.arena[xp].left {
                let mut w = self.arena[xp].right;
                if self.arena[w].color == Color::Red {
                    self.arena[w].color = Color::Black;
                    self.arena[xp].color = Color::Red;
                    self.left_rotate(xp);
                    w = self.arena[self.arena[x].parent].right;
                }
                let wl = self.arena[w].left;
                let wr = self.arena[w].right;
                if self.arena[wl].color == Color::Black && self.arena[wr].color == Color::Black {
                    self.arena[w].color = Color::Red;
                    x = self.arena[x].parent;
                } else {
                    if self.arena[wr].color == Color::Black {
                        self.arena[wl].color = Color::Black;
                        self.arena[w].color = Color::Red;
                        self.right_rotate(w);
                        w = self.arena[self.arena[x].parent].right;
                    }
                    let xp2 = self.arena[x].parent;
                    self.arena[w].color = self.arena[xp2].color;
                    self.arena[xp2].color = Color::Black;
                    let wr2 = self.arena[w].right;
                    self.arena[wr2].color = Color::Black;
                    self.left_rotate(xp2);
                    x = self.root;
                }
            } else {
                let xp_cur = self.arena[x].parent;
                let mut w = self.arena[xp_cur].left;
                if self.arena[w].color == Color::Red {
                    self.arena[w].color = Color::Black;
                    let xp_cur2 = self.arena[x].parent;
                    self.arena[xp_cur2].color = Color::Red;
                    self.right_rotate(xp_cur2);
                    let xp_cur3 = self.arena[x].parent;
                    w = self.arena[xp_cur3].left;
                }
                let wl = self.arena[w].left;
                let wr = self.arena[w].right;
                if self.arena[wr].color == Color::Black && self.arena[wl].color == Color::Black {
                    self.arena[w].color = Color::Red;
                    x = self.arena[x].parent;
                } else {
                    if self.arena[wl].color == Color::Black {
                        self.arena[wr].color = Color::Black;
                        self.arena[w].color = Color::Red;
                        self.left_rotate(w);
                        w = self.arena[self.arena[x].parent].left;
                    }
                    let xp2 = self.arena[x].parent;
                    self.arena[w].color = self.arena[xp2].color;
                    self.arena[xp2].color = Color::Black;
                    let wl2 = self.arena[w].left;
                    self.arena[wl2].color = Color::Black;
                    self.right_rotate(xp2);
                    x = self.root;
                }
            }
        }
        self.arena[x].color = Color::Black;
    }

    /// CLRS §14.3 INTERVAL-SEARCH — finds any overlapping node.
    fn search_any(&self, mut x: NodeIdx, low: K, high: K) -> Option<NodeIdx> {
        while x != NIL {
            if self.arena[x].interval.overlaps(low, high) {
                return Some(x);
            }
            let l = self.arena[x].left;
            if l != NIL && self.max_of(l) > low {
                x = l;
            } else {
                x = self.arena[x].right;
            }
        }
        None
    }

    /// Find all overlapping nodes by performing recursive DFS.
    fn search_all<'a>(&'a self, x: NodeIdx, low: K, high: K, result: &mut Vec<&'a V>) {
        if x == NIL {
            return;
        }
        // Prune: if the max of this subtree is <= low, no overlap possible.
        if self.max_of(x) <= low {
            return;
        }
        // Descend left.
        self.search_all(self.arena[x].left, low, high, result);
        // Check current.
        if self.arena[x].interval.overlaps(low, high) {
            result.push(&self.arena[x].value);
        }
        // Descend right only if low endpoint of subtree doesn't exceed high.
        if self.arena[x].interval.low < high {
            self.search_all(self.arena[x].right, low, high, result);
        }
    }

    fn search_all_pairs<'a>(
        &'a self,
        x: NodeIdx,
        low: K,
        high: K,
        result: &mut Vec<(Interval<K>, &'a V)>,
    ) {
        if x == NIL {
            return;
        }
        if self.max_of(x) <= low {
            return;
        }
        self.search_all_pairs(self.arena[x].left, low, high, result);
        if self.arena[x].interval.overlaps(low, high) {
            result.push((self.arena[x].interval, &self.arena[x].value));
        }
        if self.arena[x].interval.low < high {
            self.search_all_pairs(self.arena[x].right, low, high, result);
        }
    }

    /// In-order traversal (yields entries sorted by `low`).
    fn inorder<'a>(&'a self, x: NodeIdx, out: &mut Vec<(Interval<K>, &'a V)>) {
        if x == NIL {
            return;
        }
        self.inorder(self.arena[x].left, out);
        out.push((self.arena[x].interval, &self.arena[x].value));
        self.inorder(self.arena[x].right, out);
    }

    /// Find the first node with exactly this interval.
    fn find_node(&self, interval: Interval<K>) -> Option<NodeIdx> {
        let mut x = self.root;
        while x != NIL {
            match interval.low.cmp(&self.arena[x].interval.low) {
                Ordering::Less => x = self.arena[x].left,
                Ordering::Greater => x = self.arena[x].right,
                Ordering::Equal => {
                    if self.arena[x].interval.high == interval.high {
                        return Some(x);
                    }
                    // There may be duplicates with same low; try right subtree.
                    x = self.arena[x].right;
                }
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ScheduleIntervalIndex — domain adapter for chrono::DateTime<Utc>
// ─────────────────────────────────────────────────────────────────────────────

use chrono::{DateTime, Utc};

/// Schedule identifier attached to each interval entry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScheduleEntry {
    /// Unique identifier for the scheduled playlist.
    pub playlist_id: String,
    /// Human-readable name of the playlist.
    pub playlist_name: String,
    /// Priority of the scheduled item (higher = more important).
    pub priority: u32,
}

/// An interval tree indexed by epoch milliseconds (`i64`) for broadcast
/// schedule overlap detection.
///
/// All `DateTime<Utc>` values are converted to milliseconds since UNIX epoch
/// via [`to_epoch_ms`] before storage.
///
/// # Complexity
///
/// - `insert`, `remove`, `has_overlap`: **O(log n)**
/// - `all_overlapping`: **O(k log n)**, k = result count
pub struct ScheduleIntervalIndex {
    tree: IntervalTree<i64, ScheduleEntry>,
}

impl Default for ScheduleIntervalIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl ScheduleIntervalIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tree: IntervalTree::new(),
        }
    }

    /// Return the number of scheduled entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// Return `true` if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Insert a schedule entry.
    pub fn insert(&mut self, start: DateTime<Utc>, end: DateTime<Utc>, entry: ScheduleEntry) {
        let lo = to_epoch_ms(start);
        let hi = to_epoch_ms(end);
        if lo < hi {
            self.tree.insert(Interval::new(lo, hi), entry);
        }
    }

    /// Remove a schedule entry by its exact `[start, end)` range.
    ///
    /// Returns the removed entry, or `None` if not found.
    pub fn remove(&mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Option<ScheduleEntry> {
        let lo = to_epoch_ms(start);
        let hi = to_epoch_ms(end);
        if lo < hi {
            self.tree.remove(Interval::new(lo, hi))
        } else {
            None
        }
    }

    /// Return `true` if any scheduled entry overlaps `[query_start, query_end)`.
    ///
    /// O(log n).
    #[must_use]
    pub fn has_overlap(&self, query_start: DateTime<Utc>, query_end: DateTime<Utc>) -> bool {
        let lo = to_epoch_ms(query_start);
        let hi = to_epoch_ms(query_end);
        self.tree.query_overlapping(lo, hi).is_some()
    }

    /// Return all schedule entries whose intervals overlap `[query_start, query_end)`.
    ///
    /// O(k log n), k = result count.
    #[must_use]
    pub fn all_overlapping(
        &self,
        query_start: DateTime<Utc>,
        query_end: DateTime<Utc>,
    ) -> Vec<&ScheduleEntry> {
        let lo = to_epoch_ms(query_start);
        let hi = to_epoch_ms(query_end);
        self.tree.query_all_overlapping(lo, hi)
    }

    /// Return all entries sorted by start time.
    #[must_use]
    pub fn iter_sorted(&self) -> Vec<(DateTime<Utc>, DateTime<Utc>, &ScheduleEntry)> {
        self.tree
            .iter_sorted()
            .into_iter()
            .map(|(iv, entry)| (from_epoch_ms(iv.low), from_epoch_ms(iv.high), entry))
            .collect()
    }
}

/// Convert `DateTime<Utc>` to milliseconds since UNIX epoch.
#[must_use]
pub fn to_epoch_ms(dt: DateTime<Utc>) -> i64 {
    dt.timestamp_millis()
}

/// Convert milliseconds since UNIX epoch back to `DateTime<Utc>`.
#[must_use]
pub fn from_epoch_ms(ms: i64) -> DateTime<Utc> {
    use chrono::TimeZone;
    let secs = ms / 1000;
    let nanos = ((ms % 1000) * 1_000_000) as u32;
    Utc.timestamp_opt(secs, nanos).single().unwrap_or_else(|| {
        Utc.timestamp_opt(0, 0)
            .single()
            .expect("epoch zero is valid")
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};

    fn epoch() -> DateTime<Utc> {
        Utc.timestamp_opt(0, 0).single().expect("epoch")
    }

    fn at_secs(s: i64) -> DateTime<Utc> {
        epoch() + Duration::seconds(s)
    }

    fn make_entry(id: &str) -> ScheduleEntry {
        ScheduleEntry {
            playlist_id: id.to_string(),
            playlist_name: id.to_string(),
            priority: 1,
        }
    }

    // ── Interval ──────────────────────────────────────────────────────────

    #[test]
    fn test_interval_overlaps_standard() {
        let iv = Interval::new(100u64, 200u64);
        assert!(iv.overlaps(150, 300), "[100,200) should overlap [150,300)");
        assert!(iv.overlaps(50, 150), "[100,200) should overlap [50,150)");
        assert!(
            iv.overlaps(100, 200),
            "[100,200) should overlap [100,200) (same)"
        );
        assert!(
            !iv.overlaps(200, 300),
            "[100,200) should NOT overlap [200,300) (adjacent)"
        );
        assert!(
            !iv.overlaps(0, 100),
            "[100,200) should NOT overlap [0,100) (adjacent)"
        );
        assert!(!iv.overlaps(0, 50), "[100,200) should NOT overlap [0,50)");
    }

    #[test]
    fn test_interval_overlaps_point() {
        let iv = Interval::new(5i64, 10i64);
        // A single-point interval [7, 8) is inside.
        assert!(iv.overlaps(7, 8));
        // [10, 11) is adjacent — no overlap.
        assert!(!iv.overlaps(10, 11));
    }

    // ── IntervalTree basic ────────────────────────────────────────────────

    #[test]
    fn test_tree_empty() {
        let tree: IntervalTree<u64, u32> = IntervalTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.query_overlapping(0, 100), None);
    }

    #[test]
    fn test_tree_insert_and_query() {
        let mut tree: IntervalTree<u64, &str> = IntervalTree::new();
        tree.insert(Interval::new(100, 200), "a");
        assert_eq!(tree.len(), 1);
        let v = tree.query_overlapping(150, 160);
        assert_eq!(v, Some(&"a"));
    }

    #[test]
    fn test_tree_no_overlap_returns_none() {
        let mut tree: IntervalTree<u64, &str> = IntervalTree::new();
        tree.insert(Interval::new(100, 200), "a");
        assert_eq!(tree.query_overlapping(200, 300), None);
        assert_eq!(tree.query_overlapping(0, 100), None);
    }

    #[test]
    fn test_tree_multiple_inserts_all_overlapping() {
        let mut tree: IntervalTree<u64, &str> = IntervalTree::new();
        tree.insert(Interval::new(100, 200), "a");
        tree.insert(Interval::new(300, 400), "b");
        tree.insert(Interval::new(150, 250), "c");

        let hits = tree.query_all_overlapping(180, 220);
        assert_eq!(hits.len(), 2, "should hit 'a' and 'c'");
        assert!(hits.contains(&&"a") && hits.contains(&&"c"));
    }

    #[test]
    fn test_tree_remove() {
        let mut tree: IntervalTree<u64, &str> = IntervalTree::new();
        tree.insert(Interval::new(100, 200), "x");
        assert_eq!(tree.len(), 1);
        let removed = tree.remove(Interval::new(100, 200));
        assert_eq!(removed, Some("x"));
        assert!(tree.is_empty());
        assert_eq!(tree.query_overlapping(100, 200), None);
    }

    #[test]
    fn test_tree_remove_nonexistent_returns_none() {
        let mut tree: IntervalTree<u64, &str> = IntervalTree::new();
        tree.insert(Interval::new(100, 200), "x");
        let removed = tree.remove(Interval::new(100, 201)); // wrong high
        assert_eq!(removed, None);
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_tree_iter_sorted() {
        let mut tree: IntervalTree<u64, &str> = IntervalTree::new();
        tree.insert(Interval::new(300, 400), "c");
        tree.insert(Interval::new(100, 200), "a");
        tree.insert(Interval::new(200, 300), "b");

        let sorted = tree.iter_sorted();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0.low, 100);
        assert_eq!(sorted[1].0.low, 200);
        assert_eq!(sorted[2].0.low, 300);
    }

    #[test]
    fn test_tree_large_insert_and_query() {
        let mut tree: IntervalTree<u64, usize> = IntervalTree::new();
        // Insert 100 non-overlapping intervals: [0,10), [10,20), …
        for i in 0u64..100 {
            tree.insert(Interval::new(i * 10, i * 10 + 10), i as usize);
        }
        assert_eq!(tree.len(), 100);

        // Query a known interval.
        let v = tree.query_overlapping(55, 65);
        assert!(v.is_some(), "55..65 should overlap [50,60) or [60,70)");

        // Query outside all intervals — no overlap.
        assert_eq!(tree.query_overlapping(1000, 2000), None);
    }

    #[test]
    fn test_tree_query_all_adjacent_intervals() {
        let mut tree: IntervalTree<u64, &str> = IntervalTree::new();
        tree.insert(Interval::new(0, 100), "a");
        tree.insert(Interval::new(100, 200), "b");
        // [0,100) and [100,200) are adjacent; a query [50,150) overlaps both.
        let hits = tree.query_all_overlapping(50, 150);
        assert_eq!(hits.len(), 2);
    }

    // ── ScheduleIntervalIndex ─────────────────────────────────────────────

    #[test]
    fn test_schedule_index_empty() {
        let idx = ScheduleIntervalIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(!idx.has_overlap(epoch(), at_secs(3600)));
    }

    #[test]
    fn test_schedule_index_no_overlap() {
        let mut idx = ScheduleIntervalIndex::new();
        idx.insert(at_secs(0), at_secs(3600), make_entry("morning"));
        // A later time window that doesn't overlap.
        assert!(!idx.has_overlap(at_secs(3600), at_secs(7200)));
    }

    #[test]
    fn test_schedule_index_overlap_detected() {
        let mut idx = ScheduleIntervalIndex::new();
        idx.insert(at_secs(0), at_secs(3600), make_entry("morning"));
        idx.insert(at_secs(7200), at_secs(10800), make_entry("evening"));

        // [1800, 5400) overlaps "morning" [0, 3600).
        assert!(idx.has_overlap(at_secs(1800), at_secs(5400)));
        // [5400, 7200) doesn't overlap anything.
        assert!(!idx.has_overlap(at_secs(5400), at_secs(7200)));
    }

    #[test]
    fn test_schedule_index_all_overlapping() {
        let mut idx = ScheduleIntervalIndex::new();
        idx.insert(at_secs(0), at_secs(3600), make_entry("a"));
        idx.insert(at_secs(1800), at_secs(5400), make_entry("b"));
        idx.insert(at_secs(7200), at_secs(10800), make_entry("c"));

        // Query [2700, 3601) overlaps both "a" and "b".
        let hits = idx.all_overlapping(at_secs(2700), at_secs(3601));
        assert_eq!(hits.len(), 2, "expected 2 overlapping entries");
        let ids: Vec<_> = hits.iter().map(|e| e.playlist_id.as_str()).collect();
        assert!(ids.contains(&"a") && ids.contains(&"b"));
    }

    #[test]
    fn test_schedule_index_remove() {
        let mut idx = ScheduleIntervalIndex::new();
        idx.insert(at_secs(0), at_secs(3600), make_entry("morning"));
        assert_eq!(idx.len(), 1);
        let removed = idx.remove(at_secs(0), at_secs(3600));
        assert!(removed.is_some());
        assert!(idx.is_empty());
        assert!(!idx.has_overlap(at_secs(0), at_secs(3600)));
    }

    #[test]
    fn test_schedule_index_iter_sorted() {
        let mut idx = ScheduleIntervalIndex::new();
        idx.insert(at_secs(7200), at_secs(10800), make_entry("evening"));
        idx.insert(at_secs(0), at_secs(3600), make_entry("morning"));
        idx.insert(at_secs(3600), at_secs(7200), make_entry("afternoon"));

        let sorted = idx.iter_sorted();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].2.playlist_id, "morning");
        assert_eq!(sorted[1].2.playlist_id, "afternoon");
        assert_eq!(sorted[2].2.playlist_id, "evening");
    }

    #[test]
    fn test_to_from_epoch_ms_roundtrip() {
        let dt = at_secs(86400); // exactly 1 day
        let ms = to_epoch_ms(dt);
        let back = from_epoch_ms(ms);
        assert_eq!(dt, back);
    }
}
