//! [`XdmMap`] — the XDM 3.1 map, a function item from atomic keys to sequences.

use crate::atomic::{AtomicValue, same_key, same_key_hash};
use crate::error::{ErrorCode, XdmError};
use crate::sequence::Sequence;
use std::fmt;
use std::sync::Arc;

/// What [`XdmMap::merge`] should do when both maps bind the same key.
///
/// These are the four options of the `duplicates` parameter of
/// [`map:merge`](https://www.w3.org/TR/xpath-functions-31/#func-map-merge).
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, DuplicateKeys, Sequence, XdmMap};
///
/// let first = XdmMap::from_iter([(AtomicValue::string("k"), Sequence::from(AtomicValue::integer(1_i64)))]);
/// let second = XdmMap::from_iter([(AtomicValue::string("k"), Sequence::from(AtomicValue::integer(2_i64)))]);
///
/// let kept = XdmMap::merge([first.clone(), second.clone()], DuplicateKeys::UseFirst)?;
/// assert_eq!(kept.get(&AtomicValue::string("k")).map(Sequence::len), Some(1));
///
/// let combined = XdmMap::merge([first.clone(), second.clone()], DuplicateKeys::Combine)?;
/// assert_eq!(combined.get(&AtomicValue::string("k")).map(Sequence::len), Some(2));
///
/// assert!(XdmMap::merge([first, second], DuplicateKeys::Reject).is_err());
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DuplicateKeys {
    /// Keep the whole entry — key and value — from the first map that supplied the key. This is
    /// `map:merge`'s default.
    #[default]
    UseFirst,
    /// Keep the whole entry — key and value — from the last map that supplied the key. Two keys
    /// that are the [`same_key`] can still be different values, so which one survives is
    /// observable through `map:keys`.
    UseLast,
    /// Keep any one of the entries; this implementation keeps the first, deterministically.
    UseAny,
    /// Concatenate all the bindings' sequences, in map order, under the first key.
    Combine,
    /// Raise [`ErrorCode::Xqdy0137`].
    Reject,
}

/// The number of hash bits one level of the trie consumes; the fan-out is `1 << BITS`.
const BITS: u32 = 5;
/// The fan-out of a branch node.
const WIDTH: u32 = 1 << BITS;
/// Masks off the bits one level consumes.
const MASK: u64 = (WIDTH - 1) as u64;

/// The slot `hash` occupies at the level starting at `shift`.
///
/// The levels at shifts `0, 5, …, 60` between them consume all 64 bits of the hash, so two
/// *different* hashes are guaranteed to take different slots at one of them. Levels past the
/// last one cannot occur; clamping rather than shifting by 64 keeps the function total.
const fn slot(hash: u64, shift: u32) -> u32 {
    if shift >= u64::BITS {
        0
    } else {
        ((hash >> shift) & MASK) as u32
    }
}

/// The single bit that marks `hash`'s slot in a branch's bitmap.
const fn slot_bit(hash: u64, shift: u32) -> u32 {
    1 << slot(hash, shift)
}

/// Where a branch's `bit` slot sits in its (compressed) child list: the number of occupied
/// slots below it.
const fn child_index(bitmap: u32, bit: u32) -> usize {
    (bitmap & (bit - 1)).count_ones() as usize
}

/// One binding, plus the rank that reproduces insertion order.
#[derive(Debug, Clone)]
struct Entry {
    key: AtomicValue,
    value: Sequence,
    /// The rank at which this key was *first* bound in this map. Replacing a binding keeps it,
    /// so `map:put` on an existing key does not move the entry.
    order: u64,
}

/// One node of the hash array-mapped trie.
///
/// The trie is *persistent*: an update copies only the nodes on one root-to-leaf path — and
/// only those of them that are actually shared with another map — so [`XdmMap::clone`] is a
/// single reference-count bump and `map:put`/`map:remove` touch `O(log₃₂ n)` nodes instead of
/// all `n` entries.
#[derive(Debug, Clone)]
enum Node {
    /// A 32-way branch, compressed: `bitmap` marks the occupied slots and `children` holds
    /// exactly those, in slot order.
    Branch {
        /// One bit per occupied slot.
        bitmap: u32,
        /// The occupied children, in increasing slot order.
        children: Vec<Arc<Node>>,
    },
    /// Exactly one entry, sitting at the shallowest depth at which its hash is unambiguous.
    Leaf {
        /// The entry key's [`same_key_hash`].
        hash: u64,
        /// The binding.
        entry: Entry,
    },
    /// Two or more entries whose full 64-bit hashes are equal but whose keys are not
    /// [`same_key`] — a genuine hash collision.
    Collision {
        /// The hash every entry here shares.
        hash: u64,
        /// The colliding bindings.
        entries: Vec<Entry>,
    },
}

impl Node {
    /// A branch with no children, used as the placeholder an entry is moved out through.
    const fn empty_branch() -> Self {
        Self::Branch {
            bitmap: 0,
            children: Vec::new(),
        }
    }

    /// The entry `key` is bound to, if any.
    fn find(&self, hash: u64, key: &AtomicValue) -> Option<&Entry> {
        let mut node = self;
        let mut shift = 0;
        loop {
            match node {
                Self::Leaf {
                    hash: leaf_hash,
                    entry,
                } => {
                    return (*leaf_hash == hash && same_key(&entry.key, key)).then_some(entry);
                }
                Self::Collision {
                    hash: node_hash,
                    entries,
                } => {
                    if *node_hash != hash {
                        return None;
                    }
                    return entries.iter().find(|entry| same_key(&entry.key, key));
                }
                Self::Branch { bitmap, children } => {
                    let bit = slot_bit(hash, shift);
                    if bitmap & bit == 0 {
                        return None;
                    }
                    node = children.get(child_index(*bitmap, bit))?;
                    shift += BITS;
                }
            }
        }
    }

    /// Visits every entry, in trie order, stopping as soon as `predicate` answers `false`.
    fn all<'a>(&'a self, predicate: &mut impl FnMut(&'a Entry) -> bool) -> bool {
        match self {
            Self::Branch { children, .. } => children.iter().all(|child| child.all(predicate)),
            Self::Leaf { entry, .. } => predicate(entry),
            Self::Collision { entries, .. } => entries.iter().all(predicate),
        }
    }
}

/// Builds the branch that separates an existing subtree from a new entry whose hash differs
/// from it, creating one level for each level the two hashes still agree on.
///
/// Terminates because the levels between them consume all 64 bits of the hash and the two
/// hashes are known to differ.
fn split(
    shift: u32,
    existing: Arc<Node>,
    existing_hash: u64,
    hash: u64,
    entry: Entry,
) -> Arc<Node> {
    let existing_bit = slot_bit(existing_hash, shift);
    let bit = slot_bit(hash, shift);
    if existing_bit == bit {
        let child = split(shift + BITS, existing, existing_hash, hash, entry);
        return Arc::new(Node::Branch {
            bitmap: bit,
            children: vec![child],
        });
    }
    let leaf = Arc::new(Node::Leaf { hash, entry });
    let children = if existing_bit < bit {
        vec![existing, leaf]
    } else {
        vec![leaf, existing]
    };
    Arc::new(Node::Branch {
        bitmap: existing_bit | bit,
        children,
    })
}

/// Binds `entry` under `hash` in the subtree `node`, returning the entry it replaced.
///
/// Nodes are copied through [`Arc::make_mut`], so a map that owns its trie outright is updated
/// in place and one that shares it copies only the path.
fn insert_into(node: &mut Arc<Node>, shift: u32, hash: u64, entry: Entry) -> Option<Entry> {
    // A leaf or collision node under a *different* hash has to become a branch. Deciding that
    // before taking the mutable borrow lets the existing node be shared into the new branch
    // rather than copied.
    let diverging = match node.as_ref() {
        Node::Leaf {
            hash: existing_hash,
            ..
        }
        | Node::Collision {
            hash: existing_hash,
            ..
        } if *existing_hash != hash => Some(*existing_hash),
        _ => None,
    };
    if let Some(existing_hash) = diverging {
        let existing = Arc::clone(node);
        *node = split(shift, existing, existing_hash, hash, entry);
        return None;
    }

    let target = Arc::make_mut(node);
    match target {
        Node::Branch { bitmap, children } => {
            let bit = slot_bit(hash, shift);
            let index = child_index(*bitmap, bit);
            if *bitmap & bit == 0 {
                if index <= children.len() {
                    children.insert(index, Arc::new(Node::Leaf { hash, entry }));
                    *bitmap |= bit;
                }
                return None;
            }
            match children.get_mut(index) {
                Some(child) => insert_into(child, shift + BITS, hash, entry),
                None => None,
            }
        }
        Node::Leaf {
            hash: leaf_hash,
            entry: existing,
        } => {
            if same_key(&existing.key, &entry.key) {
                // F&O 3.1 §17.1: two keys that are `op:same-key` need not be the same *value* —
                // `xs:double('NaN')` and `xs:float('NaN')`, or an `xs:anyURI` and the
                // `xs:string` with the same characters. The new binding replaces the old key as
                // well as its value, so `map:put` and `map:merge`'s `use-last` leave the key the
                // caller last supplied, whose type `map:keys` then reports — but the entry keeps
                // its position in insertion order.
                let mut entry = entry;
                entry.order = existing.order;
                return Some(std::mem::replace(existing, entry));
            }
            // Two different keys with the same 64-bit hash. This is the one place the
            // implementation copies an entry rather than sharing it, and only a genuine hash
            // collision reaches it.
            let collision = Node::Collision {
                hash: *leaf_hash,
                entries: vec![existing.clone(), entry],
            };
            *target = collision;
            None
        }
        Node::Collision { entries, .. } => {
            match entries
                .iter_mut()
                .find(|candidate| same_key(&candidate.key, &entry.key))
            {
                Some(existing) => {
                    let mut entry = entry;
                    entry.order = existing.order;
                    Some(std::mem::replace(existing, entry))
                }
                None => {
                    entries.push(entry);
                    None
                }
            }
        }
    }
}

/// The sequence `key` is bound to, ready to be replaced in place.
fn value_mut<'a>(
    node: &'a mut Arc<Node>,
    shift: u32,
    hash: u64,
    key: &AtomicValue,
) -> Option<&'a mut Sequence> {
    match Arc::make_mut(node) {
        Node::Leaf {
            hash: leaf_hash,
            entry,
        } => (*leaf_hash == hash && same_key(&entry.key, key)).then_some(&mut entry.value),
        Node::Collision {
            hash: node_hash,
            entries,
        } => {
            if *node_hash != hash {
                return None;
            }
            entries
                .iter_mut()
                .find(|entry| same_key(&entry.key, key))
                .map(|entry| &mut entry.value)
        }
        Node::Branch { bitmap, children } => {
            let bit = slot_bit(hash, shift);
            if *bitmap & bit == 0 {
                return None;
            }
            let index = child_index(*bitmap, bit);
            value_mut(children.get_mut(index)?, shift + BITS, hash, key)
        }
    }
}

/// What removing a key did to a subtree.
enum Removal {
    /// The key was not bound here.
    Missing,
    /// The entry is gone and the subtree still holds something.
    Removed(Entry),
    /// The entry is gone and the subtree is now empty; the caller must drop it.
    Emptied(Entry),
}

/// What [`remove_from`] must do once it has looked at a node, computed while nothing is
/// borrowed mutably.
enum Step {
    Missing,
    TakeLeaf,
    Collision,
    Descend(usize),
}

/// Moves the entry out of a leaf node, copying it only if another map shares the node.
fn take_leaf(node: &mut Arc<Node>) -> Option<Entry> {
    match std::mem::replace(Arc::make_mut(node), Node::empty_branch()) {
        Node::Leaf { entry, .. } => Some(entry),
        _ => None,
    }
}

/// The only child of `node`, when it is a branch with exactly one child that can stand on its
/// own (a leaf or a collision node, both of which carry their own hash).
fn lone_child(node: &Arc<Node>) -> Option<Arc<Node>> {
    let Node::Branch { children, .. } = node.as_ref() else {
        return None;
    };
    let [only] = children.as_slice() else {
        return None;
    };
    matches!(only.as_ref(), Node::Leaf { .. } | Node::Collision { .. }).then(|| Arc::clone(only))
}

/// Unbinds `key` from the subtree `node`.
fn remove_from(node: &mut Arc<Node>, shift: u32, hash: u64, key: &AtomicValue) -> Removal {
    let step = match node.as_ref() {
        Node::Leaf {
            hash: leaf_hash,
            entry,
        } => {
            if *leaf_hash == hash && same_key(&entry.key, key) {
                Step::TakeLeaf
            } else {
                Step::Missing
            }
        }
        Node::Collision {
            hash: node_hash,
            entries,
        } => {
            if *node_hash == hash && entries.iter().any(|entry| same_key(&entry.key, key)) {
                Step::Collision
            } else {
                Step::Missing
            }
        }
        Node::Branch { bitmap, .. } => {
            let bit = slot_bit(hash, shift);
            if bitmap & bit == 0 {
                Step::Missing
            } else {
                Step::Descend(child_index(*bitmap, bit))
            }
        }
    };

    match step {
        Step::Missing => Removal::Missing,
        Step::TakeLeaf => match take_leaf(node) {
            Some(entry) => Removal::Emptied(entry),
            None => Removal::Missing,
        },
        Step::Collision => {
            let Node::Collision { hash, entries } = Arc::make_mut(node) else {
                return Removal::Missing;
            };
            let Some(index) = entries.iter().position(|entry| same_key(&entry.key, key)) else {
                return Removal::Missing;
            };
            let removed = entries.remove(index);
            // A collision node that is down to one entry becomes an ordinary leaf again.
            let collapsed = (entries.len() == 1)
                .then(|| entries.pop())
                .flatten()
                .map(|entry| Node::Leaf { hash: *hash, entry });
            match collapsed {
                Some(leaf) => *Arc::make_mut(node) = leaf,
                None if entries.is_empty() => return Removal::Emptied(removed),
                None => {}
            }
            Removal::Removed(removed)
        }
        Step::Descend(index) => {
            let outcome = {
                let Node::Branch { bitmap, children } = Arc::make_mut(node) else {
                    return Removal::Missing;
                };
                let Some(child) = children.get_mut(index) else {
                    return Removal::Missing;
                };
                match remove_from(child, shift + BITS, hash, key) {
                    Removal::Missing => return Removal::Missing,
                    Removal::Removed(entry) => Removal::Removed(entry),
                    Removal::Emptied(entry) => {
                        children.remove(index);
                        *bitmap &= !slot_bit(hash, shift);
                        if children.is_empty() {
                            return Removal::Emptied(entry);
                        }
                        Removal::Removed(entry)
                    }
                }
            };
            // Pull a lone leaf up so that repeated removal cannot leave a chain of one-child
            // branches behind.
            if let Some(child) = lone_child(node) {
                *node = child;
            }
            outcome
        }
    }
}

/// Consumes a subtree into `out`, moving entries out of the nodes this map owns outright and
/// copying only those another map still shares.
fn drain(node: Arc<Node>, out: &mut Vec<Entry>) {
    match Arc::try_unwrap(node) {
        Ok(Node::Branch { children, .. }) => {
            for child in children {
                drain(child, out);
            }
        }
        Ok(Node::Leaf { entry, .. }) => out.push(entry),
        Ok(Node::Collision { entries, .. }) => out.extend(entries),
        Err(shared) => {
            shared.all(&mut |entry| {
                out.push(entry.clone());
                true
            });
        }
    }
}

/// An XDM 3.1 [map](https://www.w3.org/TR/xpath-datamodel-31/#dt-map): a function item binding
/// atomic keys to sequences.
///
/// # Key identity
///
/// Keys are compared with [`same_key`], XQuery's `op:same-key` relation, **not** with `==` on
/// [`AtomicValue`] and not with the `eq` operator. The consequences worth remembering:
///
/// * `xs:integer 1`, `xs:decimal 1.0` and `xs:double 1.0E0` are one key, not three;
/// * `NaN` is a usable key, and there is only one of it;
/// * `xs:string "x"`, `xs:anyURI "x"` and `xs:untypedAtomic "x"` are one key;
/// * values of unrelated types are simply different keys — looking one up never raises.
///
/// # Representation
///
/// The entries live in a *persistent* hash array-mapped trie keyed by [`same_key_hash`], with
/// every candidate confirmed by [`same_key`]. Two properties of that representation are part of
/// what callers can rely on, because a query language builds maps by repeatedly deriving new
/// ones from old:
///
/// * [`clone`](Clone::clone) is `O(1)` — one reference-count bump — so passing a map around,
///   or holding both the old and the new map after a `map:put`, costs nothing;
/// * [`put`](XdmMap::put) and [`remove`](XdmMap::remove) are `O(log n)` and share every node
///   they do not change with the map they came from.
///
/// # Order
///
/// XDM says a map is unordered, and
/// [`map:keys`](https://www.w3.org/TR/xpath-functions-31/#func-map-keys) returns its result "in
/// implementation-dependent order". This implementation nonetheless yields *insertion* order
/// from [`entries`](XdmMap::entries), [`keys`](XdmMap::keys) and
/// [`into_entries`](XdmMap::into_entries), which makes tests and serialization deterministic;
/// replacing the binding of an existing key keeps that key's original position, and re-adding a
/// removed key puts it at the end. Producing that order costs one sort, `O(n log n)`, per call
/// — iteration is not the operation this type is tuned for. [`PartialEq`] ignores order
/// entirely.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, Sequence, XdmMap};
///
/// let mut map = XdmMap::new();
/// map.insert(AtomicValue::integer(1_i64), Sequence::from(AtomicValue::string("one")));
///
/// // The same key spelled with a different numeric type finds the same entry.
/// assert!(map.contains_key(&AtomicValue::double(1.0)));
/// assert_eq!(
///     map.get(&AtomicValue::decimal(1_i64)).and_then(Sequence::first).map(ToString::to_string),
///     Some("one".to_owned())
/// );
///
/// // Insertion order does not affect equality.
/// let a = XdmMap::from_iter([
///     (AtomicValue::string("x"), Sequence::empty()),
///     (AtomicValue::string("y"), Sequence::empty()),
/// ]);
/// let b = XdmMap::from_iter([
///     (AtomicValue::string("y"), Sequence::empty()),
///     (AtomicValue::string("x"), Sequence::empty()),
/// ]);
/// assert_eq!(a, b);
/// ```
#[derive(Clone, Default)]
pub struct XdmMap {
    /// The trie, or `None` for the empty map.
    root: Option<Arc<Node>>,
    length: usize,
    /// The rank the next *new* key will be given.
    next_order: u64,
}

impl XdmMap {
    /// The empty map, `map {}`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Whether the map has no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    fn entry(&self, key: &AtomicValue) -> Option<&Entry> {
        self.root.as_ref()?.find(same_key_hash(key), key)
    }

    /// The sequence bound to `key`, or `None` if the map has no such key.
    #[must_use]
    pub fn get(&self, key: &AtomicValue) -> Option<&Sequence> {
        self.entry(key).map(|entry| &entry.value)
    }

    /// Whether the map binds `key`, as `map:contains` reports.
    #[must_use]
    pub fn contains_key(&self, key: &AtomicValue) -> bool {
        self.entry(key).is_some()
    }

    /// Binds `key` to `value`, returning the sequence it previously bound, if any.
    ///
    /// When an existing entry is replaced, the *new* key value is kept (as `map:put`
    /// specifies) and so is the old entry's position in insertion order.
    pub fn insert(&mut self, key: AtomicValue, value: Sequence) -> Option<Sequence> {
        let hash = same_key_hash(&key);
        let entry = Entry {
            key,
            value,
            order: self.next_order,
        };
        let replaced = match &mut self.root {
            Some(root) => insert_into(root, 0, hash, entry),
            None => {
                self.root = Some(Arc::new(Node::Leaf { hash, entry }));
                None
            }
        };
        match replaced {
            Some(previous) => Some(previous.value),
            None => {
                self.length += 1;
                self.next_order = self.next_order.saturating_add(1);
                None
            }
        }
    }

    /// Returns this map with `key` bound to `value`, as `map:put` does.
    ///
    /// The map this was derived from is unaffected and keeps sharing every part of the trie
    /// that did not change.
    #[must_use]
    pub fn put(mut self, key: AtomicValue, value: Sequence) -> Self {
        self.insert(key, value);
        self
    }

    /// Unbinds `key`, returning the sequence it bound, as `map:remove` does.
    ///
    /// Removing a key the map does not bind is not an error and changes nothing.
    pub fn remove(&mut self, key: &AtomicValue) -> Option<Sequence> {
        let hash = same_key_hash(key);
        let root = self.root.as_mut()?;
        let removed = match remove_from(root, 0, hash, key) {
            Removal::Missing => return None,
            Removal::Removed(entry) => entry,
            Removal::Emptied(entry) => {
                self.root = None;
                entry
            }
        };
        self.length = self.length.saturating_sub(1);
        Some(removed.value)
    }

    /// The sequence bound to `key`, ready to be replaced in place.
    fn get_mut(&mut self, key: &AtomicValue) -> Option<&mut Sequence> {
        let hash = same_key_hash(key);
        value_mut(self.root.as_mut()?, 0, hash, key)
    }

    /// Every entry, in insertion order.
    fn ordered(&self) -> Vec<&Entry> {
        let mut entries = Vec::with_capacity(self.length);
        if let Some(root) = &self.root {
            root.all(&mut |entry| {
                entries.push(entry);
                true
            });
        }
        entries.sort_unstable_by_key(|entry| entry.order);
        entries
    }

    /// Iterates over the entries in insertion order.
    ///
    /// Ordering the entries costs `O(n log n)`; [`get`](XdmMap::get) is the operation to reach
    /// for when only one entry is wanted.
    pub fn entries(&self) -> impl Iterator<Item = (&AtomicValue, &Sequence)> {
        self.ordered()
            .into_iter()
            .map(|entry| (&entry.key, &entry.value))
    }

    /// Iterates over the keys in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &AtomicValue> {
        self.ordered().into_iter().map(|entry| &entry.key)
    }

    /// Consumes the map, yielding its entries in insertion order.
    #[must_use]
    pub fn into_entries(self) -> Vec<(AtomicValue, Sequence)> {
        let mut entries = Vec::with_capacity(self.length);
        if let Some(root) = self.root {
            drain(root, &mut entries);
        }
        entries.sort_unstable_by_key(|entry| entry.order);
        entries
            .into_iter()
            .map(|entry| (entry.key, entry.value))
            .collect()
    }

    /// Merges several maps into one, resolving key collisions with `duplicates`, as
    /// `map:merge` does.
    ///
    /// Two keys that are the [`same_key`] need not be the same *value*, so which of them
    /// survives is observable — `map:keys` reports its type. `map:merge` is defined in terms of
    /// whole *entries*: with `use-first` the entry from the first map that has the key is
    /// retained, with `use-last` the one from the last, key included. (`combine` keeps the first
    /// key and concatenates the values, which is all the specification constrains.)
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Xqdy0137`] when `duplicates` is [`DuplicateKeys::Reject`] and two maps bind
    /// the same key.
    pub fn merge(
        maps: impl IntoIterator<Item = Self>,
        duplicates: DuplicateKeys,
    ) -> Result<Self, XdmError> {
        let mut merged = Self::new();
        for map in maps {
            for (key, value) in map.into_entries() {
                if !merged.contains_key(&key) {
                    merged.insert(key, value);
                    continue;
                }
                match duplicates {
                    DuplicateKeys::UseFirst | DuplicateKeys::UseAny => {}
                    DuplicateKeys::UseLast => {
                        // `insert` replaces the key as well as the value and keeps the entry
                        // where it was, which is exactly "the last entry wins".
                        merged.insert(key, value);
                    }
                    DuplicateKeys::Combine => {
                        if let Some(existing) = merged.get_mut(&key) {
                            existing.append(value);
                        }
                    }
                    DuplicateKeys::Reject => {
                        return Err(XdmError::new(
                            ErrorCode::Xqdy0137,
                            format!("duplicate map key `{key}`"),
                        ));
                    }
                }
            }
        }
        Ok(merged)
    }
}

impl fmt::Debug for XdmMap {
    /// The entries, in insertion order — the trie the map is stored in is an implementation
    /// detail.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.ordered().into_iter().map(|e| (&e.key, &e.value)))
            .finish()
    }
}

impl FromIterator<(AtomicValue, Sequence)> for XdmMap {
    /// Later entries overwrite earlier ones with the same key, keeping the earlier key's
    /// position.
    fn from_iter<T: IntoIterator<Item = (AtomicValue, Sequence)>>(iter: T) -> Self {
        let mut map = Self::new();
        for (key, value) in iter {
            map.insert(key, value);
        }
        map
    }
}

impl IntoIterator for XdmMap {
    type Item = (AtomicValue, Sequence);
    type IntoIter = std::vec::IntoIter<(AtomicValue, Sequence)>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_entries().into_iter()
    }
}

impl PartialEq for XdmMap {
    /// Two maps are equal when they bind the same keys — in the [`same_key`] sense — to equal
    /// sequences. Insertion order is irrelevant.
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }
        match &self.root {
            None => true,
            Some(root) => root.all(&mut |entry| other.get(&entry.key) == Some(&entry.value)),
        }
    }
}

impl Eq for XdmMap {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::item::Item;

    fn entry(key: &str, value: i64) -> (AtomicValue, Sequence) {
        (
            AtomicValue::string(key),
            Sequence::from(AtomicValue::integer(value)),
        )
    }

    fn key_names(map: &XdmMap) -> Vec<String> {
        map.keys().map(AtomicValue::string_value).collect()
    }

    #[test]
    fn lookup_uses_same_key_across_the_numeric_tower() {
        let mut map = XdmMap::new();
        map.insert(
            AtomicValue::integer(1_i64),
            Sequence::from(AtomicValue::string("one")),
        );
        assert!(map.contains_key(&AtomicValue::double(1.0)));
        assert!(map.contains_key(&AtomicValue::decimal(1_i64)));
        assert!(map.contains_key(&AtomicValue::float(1.0_f32)));
        assert!(!map.contains_key(&AtomicValue::string("1")));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn nan_is_a_usable_key() {
        let mut map = XdmMap::new();
        map.insert(
            AtomicValue::double(f64::NAN),
            Sequence::from(AtomicValue::string("nan")),
        );
        assert!(map.contains_key(&AtomicValue::double(f64::NAN)));
        assert_eq!(
            map.get(&AtomicValue::double(f64::NAN))
                .and_then(Sequence::first)
                .map(Item::to_string),
            Some("nan".to_owned())
        );
    }

    #[test]
    fn insert_replaces_in_place_and_reports_the_old_value() {
        let mut map = XdmMap::from_iter([entry("a", 1), entry("b", 2)]);
        let previous = map.insert(
            AtomicValue::any_uri("a"),
            Sequence::from(AtomicValue::integer(9_i64)),
        );
        assert_eq!(previous.map(|old| old.len()), Some(1));
        assert_eq!(map.len(), 2);
        assert_eq!(
            key_names(&map),
            ["a", "b"],
            "the replacing key takes the replaced one's position"
        );
    }

    #[test]
    fn remove_reindexes_correctly() {
        let mut map = XdmMap::from_iter([entry("a", 1), entry("b", 2), entry("c", 3)]);
        assert!(map.remove(&AtomicValue::string("b")).is_some());
        assert_eq!(map.len(), 2);
        assert!(map.contains_key(&AtomicValue::string("a")));
        assert!(map.contains_key(&AtomicValue::string("c")));
        assert!(!map.contains_key(&AtomicValue::string("b")));
        assert!(map.remove(&AtomicValue::string("b")).is_none());
        assert_eq!(key_names(&map), ["a", "c"]);
    }

    #[test]
    fn equality_ignores_insertion_order() {
        let a = XdmMap::from_iter([entry("x", 1), entry("y", 2)]);
        let b = XdmMap::from_iter([entry("y", 2), entry("x", 1)]);
        assert_eq!(a, b);

        let c = XdmMap::from_iter([entry("x", 1), entry("y", 3)]);
        assert_ne!(a, c);
        assert_ne!(a, XdmMap::from_iter([entry("x", 1)]));
    }

    #[test]
    fn merge_honours_every_duplicate_policy() -> Result<(), XdmError> {
        let first = XdmMap::from_iter([entry("k", 1)]);
        let second = XdmMap::from_iter([entry("k", 2), entry("other", 3)]);

        let use_first = XdmMap::merge([first.clone(), second.clone()], DuplicateKeys::UseFirst)?;
        assert_eq!(
            use_first,
            XdmMap::from_iter([entry("k", 1), entry("other", 3)])
        );

        let use_any = XdmMap::merge([first.clone(), second.clone()], DuplicateKeys::UseAny)?;
        assert_eq!(use_any, use_first);

        let use_last = XdmMap::merge([first.clone(), second.clone()], DuplicateKeys::UseLast)?;
        assert_eq!(
            use_last,
            XdmMap::from_iter([entry("k", 2), entry("other", 3)])
        );

        let combined = XdmMap::merge([first.clone(), second.clone()], DuplicateKeys::Combine)?;
        assert_eq!(
            combined.get(&AtomicValue::string("k")).map(Sequence::len),
            Some(2)
        );

        let error = XdmMap::merge([first, second], DuplicateKeys::Reject)
            .expect_err("duplicate keys are rejected");
        assert_eq!(error.code(), ErrorCode::Xqdy0137);
        assert_eq!(DuplicateKeys::default(), DuplicateKeys::UseFirst);
        Ok(())
    }

    /// `map:merge` resolves a duplicate by choosing an *entry*, and two keys that are the same
    /// key can still be different values — so the surviving key's type is observable.
    #[test]
    fn merge_chooses_a_whole_entry_not_just_a_value() -> Result<(), XdmError> {
        let spellings = [
            AtomicValue::untyped_atomic("abc"),
            AtomicValue::string("abc"),
            AtomicValue::any_uri("abc"),
        ];
        let maps = || {
            spellings.iter().map(|key| {
                XdmMap::from_iter([(key.clone(), Sequence::from(AtomicValue::integer(1_i64)))])
            })
        };

        let last = XdmMap::merge(maps(), DuplicateKeys::UseLast)?;
        assert_eq!(last.len(), 1);
        assert_eq!(
            last.keys().next().map(AtomicValue::atomic_type),
            Some(crate::AtomicType::AnyUri),
            "use-last keeps the last map's entry, key included"
        );

        let first = XdmMap::merge(maps(), DuplicateKeys::UseFirst)?;
        assert_eq!(
            first.keys().next().map(AtomicValue::atomic_type),
            Some(crate::AtomicType::UntypedAtomic),
            "use-first keeps the first map's entry"
        );

        // The surviving key does not move: it is still where the key was first seen.
        let ordered = XdmMap::merge(
            [
                XdmMap::from_iter([entry("k", 1), entry("z", 2)]),
                XdmMap::from_iter([(
                    AtomicValue::any_uri("k"),
                    Sequence::from(AtomicValue::integer(3_i64)),
                )]),
            ],
            DuplicateKeys::UseLast,
        )?;
        assert_eq!(key_names(&ordered), ["k", "z"]);
        Ok(())
    }

    #[test]
    fn iteration_and_conversion() {
        let map = XdmMap::from_iter([entry("a", 1), entry("b", 2)]);
        assert_eq!(map.entries().count(), 2);
        assert_eq!(map.clone().into_entries().len(), 2);
        assert_eq!(map.clone().into_iter().count(), 2);
        assert!(XdmMap::new().is_empty());
        assert_eq!(format!("{map:?}"), format!("{:?}", map.clone()));

        let put = XdmMap::new().put(AtomicValue::string("z"), Sequence::empty());
        assert!(put.contains_key(&AtomicValue::string("z")));
    }

    /// The trie has to survive many keys, deep paths, removals that empty whole branches, and
    /// re-insertion — all while keeping insertion order.
    #[test]
    fn many_keys_survive_insertion_removal_and_reinsertion() {
        let count = 2_000_i64;
        let mut map = XdmMap::new();
        for index in 0..count {
            map.insert(
                AtomicValue::integer(index),
                Sequence::from(AtomicValue::integer(index * 2)),
            );
        }
        assert_eq!(map.len(), count as usize);
        for index in 0..count {
            assert_eq!(
                map.get(&AtomicValue::integer(index))
                    .and_then(Sequence::first)
                    .map(Item::to_string),
                Some((index * 2).to_string()),
                "key {index}"
            );
        }

        // Removing every other key leaves the rest reachable and in order.
        let mut thinned = map.clone();
        for index in (0..count).step_by(2) {
            assert!(thinned.remove(&AtomicValue::integer(index)).is_some());
        }
        assert_eq!(thinned.len(), (count / 2) as usize);
        assert_eq!(map.len(), count as usize, "the original map is untouched");
        for index in 0..count {
            assert_eq!(
                thinned.contains_key(&AtomicValue::integer(index)),
                index % 2 == 1,
                "key {index}"
            );
        }
        let order: Vec<String> = thinned.keys().map(AtomicValue::string_value).collect();
        let expected: Vec<String> = (0..count)
            .filter(|index| index % 2 == 1)
            .map(|index| index.to_string())
            .collect();
        assert_eq!(order, expected);

        // Emptying the map completely leaves a usable empty map.
        for index in 0..count {
            thinned.remove(&AtomicValue::integer(index));
        }
        assert!(thinned.is_empty());
        assert_eq!(thinned.keys().count(), 0);
        thinned.insert(AtomicValue::integer(7), Sequence::empty());
        assert_eq!(thinned.len(), 1);
    }

    /// A key that is removed and put back goes to the end of the order, which is what a
    /// `Vec`-backed map did too.
    #[test]
    fn removal_and_reinsertion_moves_a_key_to_the_end() {
        let mut map = XdmMap::from_iter([entry("a", 1), entry("b", 2), entry("c", 3)]);
        map.remove(&AtomicValue::string("a"));
        map.insert(
            AtomicValue::string("a"),
            Sequence::from(AtomicValue::integer(9_i64)),
        );
        assert_eq!(key_names(&map), ["b", "c", "a"]);
    }

    /// Deriving one map from another must not disturb the original, at any size.
    #[test]
    fn derived_maps_do_not_disturb_their_source() {
        let source = XdmMap::from_iter((0..500_i64).map(|index| {
            (
                AtomicValue::integer(index),
                Sequence::from(AtomicValue::integer(index)),
            )
        }));

        let put = source
            .clone()
            .put(AtomicValue::integer(0), Sequence::empty());
        let mut removed = source.clone();
        assert!(removed.remove(&AtomicValue::integer(0)).is_some());

        assert_eq!(source.len(), 500);
        assert_eq!(put.len(), 500);
        assert_eq!(removed.len(), 499);
        assert_eq!(
            source
                .get(&AtomicValue::integer(0))
                .map(Sequence::len)
                .unwrap_or_default(),
            1,
            "the source still binds the key both derivations changed"
        );
        assert_eq!(
            put.get(&AtomicValue::integer(0)).map(Sequence::len),
            Some(0)
        );
        assert!(!removed.contains_key(&AtomicValue::integer(0)));
    }

    /// Keys that hash alike but are not the same key have to coexist. `same_key_hash` sends
    /// every `xs:string` through the same hasher, so a collision is forced here by hand.
    #[test]
    fn colliding_hashes_coexist() {
        // Two values of unrelated types are never the same key, and this pair hashes into the
        // same numeric bucket as far as the trie is concerned only by chance; the collision
        // node is exercised directly instead.
        let mut collision = Node::Collision {
            hash: 7,
            entries: vec![
                Entry {
                    key: AtomicValue::string("a"),
                    value: Sequence::from(AtomicValue::integer(1_i64)),
                    order: 0,
                },
                Entry {
                    key: AtomicValue::string("b"),
                    value: Sequence::from(AtomicValue::integer(2_i64)),
                    order: 1,
                },
            ],
        };
        assert!(collision.find(7, &AtomicValue::string("b")).is_some());
        assert!(collision.find(7, &AtomicValue::string("c")).is_none());
        assert!(collision.find(8, &AtomicValue::string("a")).is_none());

        let mut node = Arc::new(std::mem::replace(&mut collision, Node::empty_branch()));
        // Replacing inside a collision node keeps the entry's position.
        let replaced = insert_into(
            &mut node,
            0,
            7,
            Entry {
                key: AtomicValue::string("a"),
                value: Sequence::from(AtomicValue::integer(9_i64)),
                order: 99,
            },
        );
        assert!(replaced.is_some());
        assert_eq!(
            node.find(7, &AtomicValue::string("a")).map(|e| e.order),
            Some(0)
        );

        // A third key under the same hash joins the node; removing them empties it.
        assert!(
            insert_into(
                &mut node,
                0,
                7,
                Entry {
                    key: AtomicValue::string("c"),
                    value: Sequence::empty(),
                    order: 2,
                },
            )
            .is_none()
        );
        for key in ["a", "b"] {
            assert!(matches!(
                remove_from(&mut node, 0, 7, &AtomicValue::string(key)),
                Removal::Removed(_)
            ));
        }
        assert!(matches!(
            remove_from(&mut node, 0, 7, &AtomicValue::string("c")),
            Removal::Emptied(_)
        ));
        assert!(matches!(
            remove_from(&mut node, 0, 7, &AtomicValue::string("c")),
            Removal::Missing
        ));
    }

    /// A leaf that meets a different key with the same 64-bit hash becomes a collision node.
    #[test]
    fn a_leaf_becomes_a_collision_node() {
        let mut node = Arc::new(Node::Leaf {
            hash: 3,
            entry: Entry {
                key: AtomicValue::string("a"),
                value: Sequence::empty(),
                order: 0,
            },
        });
        assert!(
            insert_into(
                &mut node,
                0,
                3,
                Entry {
                    key: AtomicValue::integer(1_i64),
                    value: Sequence::empty(),
                    order: 1,
                },
            )
            .is_none()
        );
        assert!(matches!(node.as_ref(), Node::Collision { .. }));
        assert!(node.find(3, &AtomicValue::string("a")).is_some());
        assert!(node.find(3, &AtomicValue::integer(1_i64)).is_some());
    }

    #[test]
    fn slots_cover_every_hash_bit() {
        assert_eq!(
            slot(0xFFFF_FFFF_FFFF_FFFF, 60),
            15,
            "the last level has four bits"
        );
        assert_eq!(slot(1, 0), 1);
        assert_eq!(slot(1, 64), 0, "levels past the hash are clamped");
        // Slots 0, 1 and 3 are occupied, so slot 3 is the third child and slot 0 the first.
        assert_eq!(child_index(0b1011, 0b1000), 2);
        assert_eq!(child_index(0b1011, 0b0010), 1);
        assert_eq!(child_index(0b1011, 0b0001), 0);
    }
}
