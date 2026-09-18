//! `BitmapTriples` (SPO order) — the default HDT triples implementation.
//!
//! The RDF graph, expressed purely in dictionary IDs, is stored as a three-level tree
//! (Subject → Predicate → Object) with the top (subject) level left implicit: the wire format
//! never stores a subject *value* anywhere, only per-group predicate/object data, so a reader
//! can only recover subject `s`'s data by treating it as the `s`-th group -- which is exactly
//! right when the subject ID space is dense and `1`-based, as it always is for any ID set this
//! crate's own [`crate::dictionary::FourSectionDictionary`] produces. [`BitmapTriples::build`]
//! is additionally a `pub` function accepting an arbitrary `(subject, predicate, object)` ID
//! set, so a caller bypassing the dictionary may legally supply subject IDs with gaps; see
//! `subject_keys` on [`BitmapTriples`] for how that case is still resolved correctly. The two
//! lower levels are each represented as an *array* of child IDs plus a *bitmap* marking, for
//! every array element, whether it is the last child of its parent:
//!
//! ```text
//! ArrayY[i]    the i-th (subject, predicate) pair's predicate ID, subjects in ascending order
//! BitmapY[i]   1 if ArrayY[i] is the last predicate for its subject
//!
//! ArrayZ[j]    the j-th triple's object ID, grouped by the (subject, predicate) pair it
//!              belongs to (in the same order as ArrayY)
//! BitmapZ[j]   1 if ArrayZ[j] is the last object for its (subject, predicate) pair
//! ```
//!
//! Given a bitmap's `rank`/`select` operations (see [`crate::bitmap`]), the child range `[start,
//! end]` of the `k`-th (`1`-based) parent group is `start = select1(k-1) + 1` (or `0` if
//! `k == 1`), `end = select1(k)`. This crate's `group_range` helper applies that formula
//! uniformly to both levels: the `i`-th (`0`-based) predicate's object range is
//! `group_range(bitmap_z, i + 1)`, and -- when the subject ID space is dense -- subject `s`'s
//! predicate range is `group_range(bitmap_y, s)` directly; `BitmapTriples`'s own internal
//! `subject_group_range` helper is the general form that also handles the gapped case.
//!
//! # Wire format
//!
//! ```text
//! ControlInformation   type=Triples, format=<http://purl.org/HDT/hdt#triplesBitmap>,
//!                      properties: order=1 (SPO), numTriples=<count>
//! Bitmap               BitmapY  (see crate::bitmap)
//! Bitmap               BitmapZ
//! BitArray              ArrayY  (see crate::bitarray)
//! BitArray              ArrayZ
//! ```
//!
//! # Complexity
//!
//! The wire format only ever materializes the SPO order above. [`BitmapTriples`] additionally
//! builds, lazily and purely in memory, two supplementary indexes over the very same triples --
//! PSO (grouped by predicate, then subject, then object) and OSP (grouped by object, then
//! subject, then predicate) -- mirroring what other HDT implementations call their "FoQ"
//! (focus-on-querying) additional indexes, without persisting anything resembling their
//! `.hdt.index` companion file. [`BitmapTriples::matching_ids`] routes every one of the eight
//! triple-pattern shapes to whichever structure resolves it fastest:
//!
//! ```text
//! (S, P, O)   SPO, subject-bound iterator, checking predicate and object per triple  O(degree(S))
//! (S, P, _)   SPO, subject-bound iterator, checking predicate per triple             O(degree(S))
//! (S, _, O)   SPO, subject-bound iterator, checking object per triple                O(degree(S))
//! (S, _, _)   SPO, subject-bound iterator                                            O(degree(S))
//! (_, P, O)   PSO, predicate-bound iterator, checking object per triple              O(degree(P))
//! (_, P, _)   PSO, predicate-bound iterator                                          O(degree(P))
//! (_, _, O)   OSP, object-bound iterator                                             O(degree(O))
//! (_, _, _)   full SPO scan (nothing to bind)                                        O(numTriples)
//! ```
//!
//! Building either supplementary index re-derives its order from the SPO structure with one
//! `O(numTriples log numTriples)` sort and roughly doubles this structure's memory footprint per
//! index built (so up to ~3x the footprint once both are warm) -- paid once, the first time a
//! query needs that order, and kept for the lifetime of the [`BitmapTriples`] value (a
//! [`Clone`] copies whichever supplementary indexes already happen to be built, rather than
//! discarding them).
//!
//! Every `S`-bound row above pays an extra `O(log S)` (`S` = distinct subjects) the one time a
//! [`BitmapTriples`] was [`built`](BitmapTriples::build) from a subject ID space with gaps, to
//! resolve the bound subject to its group (an internal binary search, mirroring how PSO/OSP
//! resolve their own bound id above); a dense subject ID space -- what every file this crate's
//! own [`crate::writer::HdtWriter`] writes always has -- takes the plain `O(1)` lookup instead.
//! Because the wire format itself cannot represent a gapped subject ID space (see the module
//! docs above), [`BitmapTriples::write`] rejects one outright rather than silently renumbering
//! it on the next read.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use oxixml_model::Triple;

use crate::bitarray::BitArray;
use crate::bitmap::{Bitmap, BitmapBuilder};
use crate::control_info::{ControlInfoType, ControlInformation};
use crate::dictionary::FourSectionDictionary;
use crate::error::{HdtError, Result};

/// The section name used in diagnostics for this module.
const SECTION: &str = "triples";

/// The `BitmapTriples` format IRI, as written in the triples [`ControlInformation`]'s `format`
/// field.
const TRIPLES_FORMAT: &str = "<http://purl.org/HDT/hdt#triplesBitmap>";

/// The `order` property value for SPO — the only order this crate produces or accepts.
const ORDER_SPO: u64 = 1;

/// The RDF graph, encoded purely as dictionary IDs, using the `BitmapTriples` (SPO) layout.
#[derive(Debug, Clone, Default)]
pub struct BitmapTriples {
    num_triples: u64,
    bitmap_y: Bitmap,
    bitmap_z: Bitmap,
    array_y: BitArray,
    array_z: BitArray,
    /// The distinct subject IDs present, in ascending order -- but only when that sequence is
    /// *not* already dense and `1`-based, i.e. only when group ordinal `k` would otherwise be
    /// mistaken for subject ID `k`. `None` is the common case (every `BitmapTriples` this
    /// crate's own [`crate::writer::HdtWriter`] builds, and every one [`Self::read`] parses off
    /// the wire, since the wire format has no room to describe anything else): subject ID `s`
    /// simply *is* group ordinal `s`, with no lookup table to build or search. `Some(keys)`
    /// covers a [`Self::build`] call fed a subject ID space with gaps directly (`build` is a
    /// `pub` function over an arbitrary `BTreeSet`, so nothing stops a caller from bypassing
    /// [`FourSectionDictionary`](crate::dictionary::FourSectionDictionary), which never itself
    /// produces gaps); see [`Self::subject_group_range`] and [`Self::subject_value_for_ordinal`].
    subject_keys: Option<Vec<u32>>,
    /// The PSO (predicate, subject, object) supplementary index, built lazily the first time a
    /// predicate-bound query needs it. See the module-level docs.
    pso_index: OnceLock<GroupedIndex>,
    /// The OSP (object, subject, predicate) supplementary index, built lazily the first time an
    /// object-bound (but not predicate-bound) query needs it. See the module-level docs.
    osp_index: OnceLock<GroupedIndex>,
}

impl BitmapTriples {
    /// Builds a `BitmapTriples` structure from a deduplicated, `(subject, predicate, object)`
    /// ID triple set already sorted in ascending SPO order (as [`std::collections::BTreeSet`]
    /// naturally iterates).
    #[must_use]
    pub fn build(id_triples: &BTreeSet<(u32, u32, u32)>) -> Self {
        let ordered: Vec<(u32, u32, u32)> = id_triples.iter().copied().collect();
        let num_triples = ordered.len() as u64;
        let (bitmap_y, bitmap_z, array_y, array_z) = build_group_levels(&ordered);
        let subject_keys = subject_keys_if_gapped(&ordered);

        Self {
            num_triples,
            bitmap_y,
            bitmap_z,
            array_y,
            array_z,
            subject_keys,
            pso_index: OnceLock::new(),
            osp_index: OnceLock::new(),
        }
    }

    /// The total number of triples.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> u64 {
        self.num_triples
    }

    /// Returns `true` if this structure stores no triples.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.num_triples == 0
    }

    /// The number of distinct subjects with at least one triple.
    ///
    /// This is a count, not a bound: it does *not* imply subject IDs occupy `1..=subject_count()`
    /// unless this structure's subject ID space happens to be dense (as it always is for a
    /// dictionary-driven `BitmapTriples`; see [`Self::build`] and the module docs for the
    /// general case). Use [`Self::iter_ids`] or [`Self::matching_ids`] to enumerate or look up
    /// subjects -- both already resolve gapped subject IDs correctly -- rather than looping
    /// `1..=subject_count()` and treating each value as a subject ID directly.
    #[must_use]
    #[inline]
    pub fn subject_count(&self) -> u64 {
        self.bitmap_y.count_ones()
    }

    /// Iterates over every `(subject, predicate, object)` ID triple, in SPO order.
    #[must_use]
    pub fn iter_ids(&self) -> AllTriplesIdIter<'_> {
        AllTriplesIdIter {
            triples: self,
            subject: 1,
            subject_count: self.subject_count(),
            y: 0,
            y_end: 0,
            y_valid: false,
            z: 0,
            z_end: 0,
            z_valid: false,
        }
    }

    /// Iterates over every `(subject, predicate, object)` ID triple matching the given
    /// (optionally bound) pattern.
    ///
    /// Routes each of the eight possible bound/unbound shapes to whichever order -- the native
    /// SPO structure, or a lazily-built PSO/OSP supplementary index -- resolves it fastest; see
    /// the module-level docs for the per-shape complexity table.
    #[must_use]
    pub fn matching_ids(
        &self,
        subject: Option<u32>,
        predicate: Option<u32>,
        object: Option<u32>,
    ) -> Box<dyn Iterator<Item = (u32, u32, u32)> + '_> {
        if let Some(subject_id) = subject {
            let Some((y_start, y_end)) = self.subject_group_range(subject_id) else {
                return Box::new(std::iter::empty());
            };
            return Box::new(SubjectBoundIdIter {
                triples: self,
                subject_id,
                predicate,
                object,
                y: y_start,
                y_end,
                current: None,
            });
        }

        if let Some(predicate_id) = predicate {
            let index = self
                .pso_index
                .get_or_init(|| self.build_secondary_index(SecondaryOrder::Pso));
            let Some((mid_start, mid_end)) = index.group_range_for(predicate_id) else {
                return Box::new(std::iter::empty());
            };
            return Box::new(SecondaryBoundIdIter {
                index,
                order: SecondaryOrder::Pso,
                group_id: predicate_id,
                other_filter: object,
                mid: mid_start,
                mid_end,
                current: None,
            });
        }

        if let Some(object_id) = object {
            let index = self
                .osp_index
                .get_or_init(|| self.build_secondary_index(SecondaryOrder::Osp));
            let Some((mid_start, mid_end)) = index.group_range_for(object_id) else {
                return Box::new(std::iter::empty());
            };
            return Box::new(SecondaryBoundIdIter {
                index,
                order: SecondaryOrder::Osp,
                group_id: object_id,
                other_filter: predicate,
                mid: mid_start,
                mid_end,
                current: None,
            });
        }

        Box::new(self.iter_ids())
    }

    /// Builds the `order` supplementary index from this structure's own SPO-ordered triples.
    ///
    /// Called at most once per order (see the `pso_index`/`osp_index` fields): the result is
    /// cached in the corresponding [`OnceLock`], keyed off nothing but `self`'s own already-
    /// materialized SPO levels, so this never re-enters [`Self::matching_ids`].
    fn build_secondary_index(&self, order: SecondaryOrder) -> GroupedIndex {
        let mut ordered: Vec<(u32, u32, u32)> = self
            .iter_ids()
            .map(|(s, p, o)| order.encode(s, p, o))
            .collect();
        ordered.sort_unstable();
        GroupedIndex::build(&ordered)
    }

    /// Resolves `subject_id` to its `[start, end]` (`0`-based, inclusive) range into `array_y`
    /// -- the general form of `group_range(&self.bitmap_y, u64::from(subject_id))`, correct
    /// even when this structure's subject IDs are not dense. Returns `None` if `subject_id` has
    /// no triples in this structure.
    fn subject_group_range(&self, subject_id: u32) -> Option<(usize, usize)> {
        match &self.subject_keys {
            None => group_range(&self.bitmap_y, u64::from(subject_id)),
            Some(keys) => {
                let ordinal = keys.binary_search(&subject_id).ok()? as u64 + 1;
                group_range(&self.bitmap_y, ordinal)
            }
        }
    }

    /// Resolves a `1`-based `bitmap_y`/`array_y` group ordinal back to the subject ID it
    /// represents -- the inverse of [`Self::subject_group_range`]. `ordinal` must be in
    /// `1..=self.subject_count()`, which every caller in this module (walking `bitmap_y`'s own
    /// groups in order) already guarantees by construction.
    fn subject_value_for_ordinal(&self, ordinal: u64) -> u32 {
        match &self.subject_keys {
            None => ordinal as u32,
            Some(keys) => keys[(ordinal - 1) as usize],
        }
    }

    /// Serializes this structure (its control information and both bitmaps/arrays), appending
    /// to `out`.
    ///
    /// # Errors
    ///
    /// Returns [`HdtError::NonDenseSubjectIds`] if this structure was [`built`](Self::build)
    /// from a subject ID space with gaps: the wire format has no field for an explicit subject
    /// ID list, so writing one out and reading it back would silently renumber its subjects.
    /// Densify subject IDs before writing (see [`HdtError::NonDenseSubjectIds`]'s own docs).
    pub fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        if self.subject_keys.is_some() {
            return Err(HdtError::NonDenseSubjectIds);
        }
        let info = ControlInformation::new(ControlInfoType::Triples, TRIPLES_FORMAT)
            .with_property("order", ORDER_SPO.to_string())
            .with_property("numTriples", self.num_triples.to_string());
        info.write(out);
        self.bitmap_y.write(out);
        self.bitmap_z.write(out);
        self.array_y.write(out);
        self.array_z.write(out);
        Ok(())
    }

    /// Deserializes a structure previously written by [`write`](Self::write), starting at
    /// `*pos` and advancing it past it.
    ///
    /// # Errors
    ///
    /// Returns an [`HdtError`] if the control information is malformed, declares a `format` (e.g.
    /// the alternative `triplesList` encoding) or an order this crate does not implement, or any
    /// component fails to parse or is internally inconsistent.
    pub fn read(data: &[u8], pos: &mut usize) -> Result<Self> {
        let info = ControlInformation::read(data, pos, SECTION)?;
        if info.control_type != Some(ControlInfoType::Triples) {
            return Err(HdtError::UnexpectedControlType {
                section: SECTION,
                found: info.control_type.map_or(0, |type_| type_ as u8),
            });
        }
        if !info.format_matches(TRIPLES_FORMAT) {
            return Err(HdtError::UnsupportedFormat {
                section: SECTION,
                found: info.format.clone(),
            });
        }
        let order = info.property_u64("order");
        if order != 0 && order != ORDER_SPO {
            return Err(HdtError::UnsupportedFormat {
                section: SECTION,
                found: format!("triples order {order} (only SPO=1 is supported)"),
            });
        }
        let declared_num_triples = info.property_u64("numTriples");

        let bitmap_y = Bitmap::read(data, pos, "triples/bitmapY")?;
        let bitmap_z = Bitmap::read(data, pos, "triples/bitmapZ")?;
        let array_y = BitArray::read(data, pos, "triples/arrayY")?;
        let array_z = BitArray::read(data, pos, "triples/arrayZ")?;

        if array_y.len() != bitmap_y.len() {
            return Err(HdtError::Malformed(
                "triples: ArrayY and BitmapY have different lengths".to_owned(),
            ));
        }
        if array_z.len() != bitmap_z.len() {
            return Err(HdtError::Malformed(
                "triples: ArrayZ and BitmapZ have different lengths".to_owned(),
            ));
        }
        if array_z.len() as u64 != declared_num_triples {
            return Err(HdtError::Malformed(format!(
                "triples: numTriples={declared_num_triples} but ArrayZ has {} entries",
                array_z.len()
            )));
        }
        // The subject axis is only ever recovered as "group ordinal == subject ID" (see the
        // module docs), which is sound only because every group that ArrayY declares is
        // provably closed: `build_group_levels` always closes the very last (subject,
        // predicate) group as well as the very last (subject, predicate, object) leaf, and
        // BitmapZ always carries exactly one closing bit per ArrayY entry. None of that is
        // re-derivable once the four components are back to independently-framed bytes on the
        // wire, so a crafted file that violates any of these would otherwise leave some ArrayY
        // entries -- or the very last ArrayZ entry -- permanently unreachable through any
        // query, while `len()` (from `declared_num_triples`, already checked above) keeps
        // reporting the full, honest count.
        if bitmap_z.count_ones() != array_y.len() as u64 {
            return Err(HdtError::Malformed(format!(
                "triples: BitmapZ closes {} (subject, predicate) groups but ArrayY has {} entries",
                bitmap_z.count_ones(),
                array_y.len()
            )));
        }
        if !bitmap_y.is_empty() && !bitmap_y.get(bitmap_y.len() - 1) {
            return Err(HdtError::Malformed(
                "triples: BitmapY's last bit must be set (the last subject's predicate group is always closed)"
                    .to_owned(),
            ));
        }
        if !bitmap_z.is_empty() && !bitmap_z.get(bitmap_z.len() - 1) {
            return Err(HdtError::Malformed(
                "triples: BitmapZ's last bit must be set (the last (subject, predicate) group is always closed)"
                    .to_owned(),
            ));
        }

        Ok(Self {
            num_triples: declared_num_triples,
            bitmap_y,
            bitmap_z,
            array_y,
            array_z,
            // Read off the wire, subject IDs are always dense: the format has no field that
            // could describe anything else (see the module docs).
            subject_keys: None,
            pso_index: OnceLock::new(),
            osp_index: OnceLock::new(),
        })
    }
}

/// Builds a [`FourSectionDictionary`]-backed `(subject, predicate, object)` ID set from
/// `triples`, resolving every term through `dictionary`.
///
/// Returns the ID set (deduplicated and sorted, ready for [`BitmapTriples::build`]) so the
/// caller can build the dictionary and the triples structure from the very same set.
pub(crate) fn resolve_id_triples(
    triples: &[Triple],
    dictionary: &FourSectionDictionary,
) -> BTreeSet<(u32, u32, u32)> {
    triples
        .iter()
        .map(|triple| {
            let s = dictionary
                .subject_id(triple.subject.as_ref())
                .expect("dictionary was built from these very triples");
            let p = dictionary
                .predicate_id(triple.predicate.as_ref())
                .expect("dictionary was built from these very triples");
            let o = dictionary
                .object_id(triple.object.as_ref())
                .expect("dictionary was built from these very triples");
            (s, p, o)
        })
        .collect()
}

/// Builds a [`Bitmap`] from a slice of booleans (`true` = `1` bit).
fn build_bitmap(bits: &[bool]) -> Bitmap {
    let mut builder = BitmapBuilder::zeroed(bits.len());
    for (index, &bit) in bits.iter().enumerate() {
        if bit {
            builder.set_one(index);
        }
    }
    builder.finish()
}

/// Builds the two bitmap/array levels shared by every triple-order index in this module: given
/// `ordered` -- already sorted and deduplicated in the desired `(group, mid, leaf)` order --
/// returns `(bitmap_mid, bitmap_leaf, array_mid, array_leaf)`, ready to embed in either
/// [`BitmapTriples`] itself (`group` = subject, `mid` = predicate, `leaf` = object) or a
/// [`GroupedIndex`] (`group` = predicate or object).
///
/// `bitmap_mid`/`array_mid` record, for every distinct `(group, mid)` pair in ascending order,
/// the `mid` value and whether it is the last such pair for its `group`; `bitmap_leaf`/
/// `array_leaf` record, for every triple, the `leaf` value and whether it is the last leaf for
/// its `(group, mid)` pair. Grouping into `bitmap_leaf` ranges is purely by ordinal position in
/// `array_mid` (the `i`-th entry's leaves are the `(i + 1)`-th, `1`-based, group of
/// `bitmap_leaf`) -- it never needs `mid` values to be dense. Grouping into `bitmap_mid` ranges
/// by contrast *is* keyed by `group`'s own numeric value, so a caller whose `group` id space may
/// have gaps -- a [`GroupedIndex`], or [`BitmapTriples`] itself, since [`BitmapTriples::build`]
/// accepts an arbitrary `BTreeSet` -- must resolve `group` to a dense ordinal themselves before
/// indexing into `bitmap_mid` (see [`GroupedIndex::group_range_for`] and
/// [`BitmapTriples::subject_group_range`]).
fn build_group_levels(ordered: &[(u32, u32, u32)]) -> (Bitmap, Bitmap, BitArray, BitArray) {
    let mut array_mid_values: Vec<u64> = Vec::new();
    let mut array_leaf_values: Vec<u64> = Vec::with_capacity(ordered.len());
    let mut mid_last_bits: Vec<bool> = Vec::new();
    let mut leaf_last_bits: Vec<bool> = Vec::with_capacity(ordered.len());

    for (index, &(group, mid, leaf)) in ordered.iter().enumerate() {
        let starts_new_mid_group =
            index == 0 || (ordered[index - 1].0, ordered[index - 1].1) != (group, mid);
        if starts_new_mid_group {
            array_mid_values.push(u64::from(mid));
            mid_last_bits.push(false);
        }

        array_leaf_values.push(u64::from(leaf));
        let ends_mid_group = index + 1 == ordered.len()
            || (ordered[index + 1].0, ordered[index + 1].1) != (group, mid);
        leaf_last_bits.push(ends_mid_group);
        if ends_mid_group {
            let ends_group = index + 1 == ordered.len() || ordered[index + 1].0 != group;
            if ends_group {
                if let Some(last) = mid_last_bits.last_mut() {
                    *last = true;
                }
            }
        }
    }

    (
        build_bitmap(&mid_last_bits),
        build_bitmap(&leaf_last_bits),
        BitArray::from_values(&array_mid_values),
        BitArray::from_values(&array_leaf_values),
    )
}

/// Extracts the distinct `group` (first-tuple-field) values from `ordered` -- itself already
/// sorted in `(group, ..)` order -- in ascending order, one entry per distinct value. Shared by
/// [`GroupedIndex::build`] and [`subject_keys_if_gapped`], the two places this module needs the
/// exact list of top-level keys a `(group, mid, leaf)`-ordered triple sequence uses.
fn distinct_groups(ordered: &[(u32, u32, u32)]) -> Vec<u32> {
    let mut keys: Vec<u32> = Vec::new();
    for &(group, ..) in ordered {
        if keys.last() != Some(&group) {
            keys.push(group);
        }
    }
    keys
}

/// Computes [`BitmapTriples`]'s `subject_keys` field from `ordered` (already sorted in
/// `(subject, predicate, object)` order): `None` if the distinct subject values already form a
/// dense, `1`-based sequence (`1, 2, 3, ...`), in which case a group ordinal and its subject ID
/// coincide and no lookup table is needed; `Some(keys)` -- the distinct subject values, in
/// ascending order -- otherwise. Building `keys` to check density and then discarding it in the
/// dense case costs one allocation at `build` time; what it buys is that the overwhelmingly
/// common dense case pays no retained memory and no per-query lookup at all.
fn subject_keys_if_gapped(ordered: &[(u32, u32, u32)]) -> Option<Vec<u32>> {
    let keys = distinct_groups(ordered);
    let is_dense = keys
        .iter()
        .enumerate()
        .all(|(index, &subject)| subject as usize == index + 1);
    (!is_dense).then_some(keys)
}

/// Returns the `0`-based `[start, end]` (inclusive) child range of the `group`-th (`1`-based)
/// parent in `bitmap`, or `None` if `group` is `0` or exceeds the number of groups.
fn group_range(bitmap: &Bitmap, group: u64) -> Option<(usize, usize)> {
    let end = bitmap.select1(group)?;
    let start = if group == 1 {
        0
    } else {
        bitmap.select1(group - 1)? + 1
    };
    Some((start, end))
}

/// Iterates over every `(subject, predicate, object)` ID triple, in SPO order.
#[derive(Debug)]
pub struct AllTriplesIdIter<'a> {
    triples: &'a BitmapTriples,
    subject: u64,
    subject_count: u64,
    y: usize,
    y_end: usize,
    y_valid: bool,
    z: usize,
    z_end: usize,
    z_valid: bool,
}

impl Iterator for AllTriplesIdIter<'_> {
    type Item = (u32, u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.subject > self.subject_count {
                return None;
            }
            if !self.y_valid {
                match group_range(&self.triples.bitmap_y, self.subject) {
                    Some((start, end)) => {
                        self.y = start;
                        self.y_end = end;
                        self.y_valid = true;
                    }
                    None => {
                        self.subject += 1;
                        continue;
                    }
                }
            }
            if self.y > self.y_end {
                self.y_valid = false;
                self.subject += 1;
                continue;
            }
            if !self.z_valid {
                match group_range(&self.triples.bitmap_z, self.y as u64 + 1) {
                    Some((start, end)) => {
                        self.z = start;
                        self.z_end = end;
                        self.z_valid = true;
                    }
                    None => {
                        self.y += 1;
                        continue;
                    }
                }
            }
            if self.z > self.z_end {
                self.z_valid = false;
                self.y += 1;
                continue;
            }
            let subject = self.triples.subject_value_for_ordinal(self.subject);
            let predicate = self.triples.array_y.get(self.y) as u32;
            let object = self.triples.array_z.get(self.z) as u32;
            self.z += 1;
            return Some((subject, predicate, object));
        }
    }
}

/// Iterates over every `(subject, predicate, object)` ID triple for a single, bound subject,
/// optionally filtered by predicate and/or object.
#[derive(Debug)]
struct SubjectBoundIdIter<'a> {
    triples: &'a BitmapTriples,
    subject_id: u32,
    predicate: Option<u32>,
    object: Option<u32>,
    y: usize,
    y_end: usize,
    /// `(predicate_id, next_z, z_end)` for the (subject, predicate) group currently being
    /// walked, if any.
    current: Option<(u32, usize, usize)>,
}

impl Iterator for SubjectBoundIdIter<'_> {
    type Item = (u32, u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((predicate_id, z, z_end)) = self.current {
                if z <= z_end {
                    let object_id = self.triples.array_z.get(z) as u32;
                    self.current = Some((predicate_id, z + 1, z_end));
                    if self.object.is_some_and(|expected| expected != object_id) {
                        continue;
                    }
                    return Some((self.subject_id, predicate_id, object_id));
                }
                self.current = None;
            }

            if self.y > self.y_end {
                return None;
            }
            let predicate_id = self.triples.array_y.get(self.y) as u32;
            self.y += 1;
            if self
                .predicate
                .is_some_and(|expected| expected != predicate_id)
            {
                continue;
            }
            if let Some((z_start, z_end)) = group_range(&self.triples.bitmap_z, self.y as u64) {
                self.current = Some((predicate_id, z_start, z_end));
            }
        }
    }
}

/// Which supplementary triple order a [`GroupedIndex`] and its [`SecondaryBoundIdIter`] embody.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SecondaryOrder {
    /// Grouped by predicate (top/"group"), then subject (mid), then object (leaf).
    Pso,
    /// Grouped by object (top/"group"), then subject (mid), then predicate (leaf).
    Osp,
}

impl SecondaryOrder {
    /// Reorders a `(subject, predicate, object)` ID triple into this order's own
    /// `(group, mid, leaf)` field order.
    const fn encode(self, subject: u32, predicate: u32, object: u32) -> (u32, u32, u32) {
        match self {
            Self::Pso => (predicate, subject, object),
            Self::Osp => (object, subject, predicate),
        }
    }

    /// Reassembles this order's `(group, mid, leaf)` fields back into `(subject, predicate,
    /// object)` ID order.
    const fn decode(self, group: u32, mid: u32, leaf: u32) -> (u32, u32, u32) {
        match self {
            Self::Pso => (mid, group, leaf),
            Self::Osp => (mid, leaf, group),
        }
    }
}

/// A supplementary triple-order index: the same triple set as [`BitmapTriples`]'s own SPO
/// levels, grouped by a different `(group, mid, leaf)` permutation of `(subject, predicate,
/// object)` -- see [`SecondaryOrder`] for which permutation. Shares [`BitmapTriples`]'s own
/// bitmap+array level scheme (see [`build_group_levels`]) and, for the very same reason
/// [`BitmapTriples`] itself keeps a `subject_keys` lookup, cannot assume its own top level
/// (predicate for PSO, object for OSP) is dense: [`BitmapTriples::build`] is a `pub` function
/// accepting an arbitrary `BTreeSet<(u32, u32, u32)>`, so a caller building triples directly
/// (bypassing [`FourSectionDictionary`], which never itself produces gaps in any of the three ID
/// spaces) may legally supply predicate or object ids with gaps. Unlike `subject_keys`, `keys`
/// here is never optional: this index is only ever built once, lazily, so there is no
/// established fast path worth special-casing away. `keys` records exactly which top-level
/// values are present, in ascending order, so a query id is resolved to its group by exact
/// binary search rather than by assuming it names one.
#[derive(Debug, Clone, Default)]
struct GroupedIndex {
    /// The distinct top-level (`group`) key values actually present, in ascending order. The
    /// `k`-th entry (`0`-based) is the top-level id whose child range is the `(k + 1)`-th
    /// (`1`-based) group of `bitmap_mid`/`array_mid`.
    keys: Vec<u32>,
    bitmap_mid: Bitmap,
    bitmap_leaf: Bitmap,
    array_mid: BitArray,
    array_leaf: BitArray,
}

impl GroupedIndex {
    /// Builds an index from `ordered`, already sorted (and deduplicated, since reordering the
    /// fields of a deduplicated triple set cannot introduce a collision) into `(group, mid,
    /// leaf)` order.
    fn build(ordered: &[(u32, u32, u32)]) -> Self {
        let keys = distinct_groups(ordered);
        let (bitmap_mid, bitmap_leaf, array_mid, array_leaf) = build_group_levels(ordered);
        Self {
            keys,
            bitmap_mid,
            bitmap_leaf,
            array_mid,
            array_leaf,
        }
    }

    /// Returns the `[start, end]` (inclusive, into `array_mid`) child range for the group whose
    /// top-level key is `group_value`, or `None` if `group_value` has no triples in this index.
    fn group_range_for(&self, group_value: u32) -> Option<(usize, usize)> {
        let ordinal = self.keys.binary_search(&group_value).ok()? as u64 + 1;
        group_range(&self.bitmap_mid, ordinal)
    }
}

/// Iterates over every `(subject, predicate, object)` ID triple for a single, bound top-level
/// key of a [`GroupedIndex`] (predicate for PSO, object for OSP), optionally filtered by the
/// remaining unbound field (object for PSO, predicate for OSP).
#[derive(Debug)]
struct SecondaryBoundIdIter<'a> {
    index: &'a GroupedIndex,
    order: SecondaryOrder,
    /// The bound top-level id (a predicate for PSO, an object for OSP).
    group_id: u32,
    /// A filter on the leaf field (object for PSO, predicate for OSP), if also bound.
    other_filter: Option<u32>,
    mid: usize,
    mid_end: usize,
    /// `(mid_value, next_leaf, leaf_end)` for the `(group, mid)` pair currently being walked, if
    /// any.
    current: Option<(u32, usize, usize)>,
}

impl Iterator for SecondaryBoundIdIter<'_> {
    type Item = (u32, u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((mid_value, leaf, leaf_end)) = self.current {
                if leaf <= leaf_end {
                    let leaf_value = self.index.array_leaf.get(leaf) as u32;
                    self.current = Some((mid_value, leaf + 1, leaf_end));
                    if self
                        .other_filter
                        .is_some_and(|expected| expected != leaf_value)
                    {
                        continue;
                    }
                    return Some(self.order.decode(self.group_id, mid_value, leaf_value));
                }
                self.current = None;
            }

            if self.mid > self.mid_end {
                return None;
            }
            let mid_value = self.index.array_mid.get(self.mid) as u32;
            self.mid += 1;
            if let Some((leaf_start, leaf_end)) =
                group_range(&self.index.bitmap_leaf, self.mid as u64)
            {
                self.current = Some((mid_value, leaf_start, leaf_end));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> BTreeSet<(u32, u32, u32)> {
        [
            (1, 1, 2),
            (1, 1, 3),
            (1, 2, 3),
            (2, 1, 1),
            (3, 3, 1),
            (3, 3, 2),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn iter_ids_yields_all_in_spo_order() {
        let id_triples = sample();
        let triples = BitmapTriples::build(&id_triples);
        assert_eq!(triples.len(), 6);
        let collected: Vec<_> = triples.iter_ids().collect();
        let expected: Vec<_> = id_triples.into_iter().collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn matching_ids_subject_bound() {
        let triples = BitmapTriples::build(&sample());
        let mut matched: Vec<_> = triples.matching_ids(Some(1), None, None).collect();
        matched.sort_unstable();
        assert_eq!(matched, vec![(1, 1, 2), (1, 1, 3), (1, 2, 3)]);
    }

    #[test]
    fn matching_ids_subject_and_predicate_bound() {
        let triples = BitmapTriples::build(&sample());
        let matched: Vec<_> = triples.matching_ids(Some(1), Some(1), None).collect();
        assert_eq!(matched, vec![(1, 1, 2), (1, 1, 3)]);
    }

    #[test]
    fn matching_ids_fully_bound() {
        let triples = BitmapTriples::build(&sample());
        let matched: Vec<_> = triples.matching_ids(Some(1), Some(1), Some(3)).collect();
        assert_eq!(matched, vec![(1, 1, 3)]);
        let matched: Vec<_> = triples.matching_ids(Some(1), Some(1), Some(99)).collect();
        assert!(matched.is_empty());
    }

    #[test]
    fn matching_ids_unbound_subject_scans_and_filters() {
        let triples = BitmapTriples::build(&sample());
        let mut matched: Vec<_> = triples.matching_ids(None, Some(3), None).collect();
        matched.sort_unstable();
        assert_eq!(matched, vec![(3, 3, 1), (3, 3, 2)]);
    }

    #[test]
    fn matching_ids_unbound_subject_and_object() {
        let triples = BitmapTriples::build(&sample());
        let matched: Vec<_> = triples.matching_ids(None, None, Some(1)).collect();
        let mut matched = matched;
        matched.sort_unstable();
        assert_eq!(matched, vec![(2, 1, 1), (3, 3, 1)]);
    }

    #[test]
    fn nonexistent_subject_has_no_matches() {
        let triples = BitmapTriples::build(&sample());
        assert!(triples.matching_ids(Some(99), None, None).next().is_none());
    }

    #[test]
    fn empty_triples() {
        let triples = BitmapTriples::build(&BTreeSet::new());
        assert!(triples.is_empty());
        assert_eq!(triples.iter_ids().count(), 0);
        assert!(triples.matching_ids(Some(1), None, None).next().is_none());
        // The PSO/OSP paths must also build (empty) indexes cleanly rather than panicking on an
        // empty `bitmap_mid` (zero groups, so `count_ones() == 0`).
        assert!(triples.matching_ids(None, Some(1), None).next().is_none());
        assert!(triples.matching_ids(None, None, Some(1)).next().is_none());
    }

    /// A deliberately sparse predicate id space (`{1, 3, 7}`, gaps at `2`, `4`, `5`, `6`) and
    /// object id space (`{2, 5, 9}`, gaps everywhere else) distinguishes a correct
    /// [`GroupedIndex`] -- which resolves a bound predicate/object to its group via the `keys`
    /// list -- from a buggy one that assumed the supplementary index's top level was dense (see
    /// [`GroupedIndex`]'s docs; `matching_ids_handles_gapped_subject_ids` below is this same
    /// test for the *subject* axis). [`BitmapTriples::build`] is a `pub` function over an
    /// arbitrary `BTreeSet<(u32, u32, u32)>`, so nothing stops a caller from supplying ids like
    /// these directly (bypassing [`FourSectionDictionary`], which would never itself produce
    /// gaps).
    #[test]
    fn matching_ids_handles_gapped_predicate_and_object_ids() {
        let id_triples: BTreeSet<(u32, u32, u32)> = [(1, 1, 2), (1, 3, 5), (2, 7, 2), (3, 3, 9)]
            .into_iter()
            .collect();
        let triples = BitmapTriples::build(&id_triples);

        let naive = |s: Option<u32>, p: Option<u32>, o: Option<u32>| -> Vec<(u32, u32, u32)> {
            let mut result: Vec<_> = id_triples
                .iter()
                .copied()
                .filter(|&(ts, tp, to)| {
                    s.is_none_or(|expected| expected == ts)
                        && p.is_none_or(|expected| expected == tp)
                        && o.is_none_or(|expected| expected == to)
                })
                .collect();
            result.sort_unstable();
            result
        };

        // Every one of the eight bound/unbound shapes, plus a few "this id exists, but never in
        // this field" negative cases that only a gap-aware (rather than density-assuming) index
        // gets right.
        let cases: &[(Option<u32>, Option<u32>, Option<u32>)] = &[
            (None, None, None),
            (Some(1), None, None),
            (Some(2), None, None),
            (Some(99), None, None),
            (None, Some(3), None),
            (None, Some(7), None),
            (None, Some(2), None), // 2 is only ever a subject id -- must be empty.
            (None, None, Some(2)),
            (None, None, Some(9)),
            (None, None, Some(1)), // 1 is only ever a predicate id -- must be empty.
            (Some(1), Some(3), None),
            (Some(1), None, Some(2)),
            (None, Some(3), Some(9)),
            (None, Some(3), Some(2)),
            (Some(3), Some(3), Some(9)),
        ];

        for &(s, p, o) in cases {
            let mut actual: Vec<_> = triples.matching_ids(s, p, o).collect();
            actual.sort_unstable();
            assert_eq!(actual, naive(s, p, o), "pattern ({s:?}, {p:?}, {o:?})");
        }
    }

    /// The subject-axis counterpart to `matching_ids_handles_gapped_predicate_and_object_ids`
    /// above -- and the direct regression test for the bug this module's `subject_keys` field
    /// exists to close. Subjects `{0, 2, 5}` (gaps at `1`, `3`, `4`, plus `0` itself: a subject
    /// ID `group_range`'s `select1` treats as *always* invalid when used as a raw ordinal, so a
    /// pre-fix reader could never reach subject `0`'s data through any subject-bound query at
    /// all) exercise every place a `1`-based group ordinal must NOT be mistaken for a subject
    /// ID: [`BitmapTriples::iter_ids`] (a full unbound scan must report the true subject values,
    /// not their ordinal rank), [`BitmapTriples::matching_ids`]'s own subject-bound branch, and
    /// -- the case a fix touching only the subject-bound branch would miss -- the lazily-built
    /// PSO/OSP secondary indexes, which derive their own subject values from `iter_ids` (see
    /// [`BitmapTriples::build_secondary_index`]) and so silently inherit whatever `iter_ids`
    /// gets wrong.
    #[test]
    fn matching_ids_handles_gapped_subject_ids() {
        let id_triples: BTreeSet<(u32, u32, u32)> = [
            (0, 1, 100),
            (0, 2, 101),
            (2, 1, 102),
            (5, 1, 100),
            (5, 3, 103),
        ]
        .into_iter()
        .collect();
        let triples = BitmapTriples::build(&id_triples);
        assert_eq!(triples.subject_count(), 3);

        let naive = |s: Option<u32>, p: Option<u32>, o: Option<u32>| -> Vec<(u32, u32, u32)> {
            let mut result: Vec<_> = id_triples
                .iter()
                .copied()
                .filter(|&(ts, tp, to)| {
                    s.is_none_or(|expected| expected == ts)
                        && p.is_none_or(|expected| expected == tp)
                        && o.is_none_or(|expected| expected == to)
                })
                .collect();
            result.sort_unstable();
            result
        };

        // A full unbound scan must yield the true (gapped) subject values -- `0`, `2`, `5` --
        // never their `1`-based ordinal rank (`1`, `2`, `3`), which is exactly what the
        // pre-fix `AllTriplesIdIter` returned instead.
        let mut all: Vec<_> = triples.iter_ids().collect();
        all.sort_unstable();
        assert_eq!(all, naive(None, None, None));

        let cases: &[(Option<u32>, Option<u32>, Option<u32>)] = &[
            (None, None, None),
            (Some(0), None, None),
            (Some(2), None, None),
            (Some(5), None, None),
            // 1, 3, 4 are real gaps: no subject ever used these IDs. Pre-fix, querying `2`
            // here (the 2nd group ordinal) would have wrongly returned subject `2`'s own data
            // relabeled, rather than the correct empty result.
            (Some(1), None, None),
            (Some(3), None, None),
            (Some(4), None, None),
            (Some(99), None, None),
            (Some(0), Some(2), None),
            (Some(5), None, Some(103)),
            // Predicate- and object-bound shapes route through the PSO/OSP secondary indexes,
            // whose subject values are derived from `iter_ids` -- the discriminating case for
            // whether the fix reaches beyond `matching_ids`'s own subject-bound branch.
            (None, Some(1), None),
            (None, Some(2), None),
            (None, Some(3), None),
            (None, None, Some(100)),
            (None, None, Some(103)),
            (None, Some(1), Some(100)),
        ];
        for &(s, p, o) in cases {
            let mut actual: Vec<_> = triples.matching_ids(s, p, o).collect();
            actual.sort_unstable();
            assert_eq!(actual, naive(s, p, o), "pattern ({s:?}, {p:?}, {o:?})");
        }
    }

    /// [`BitmapTriples::write`] cannot faithfully persist a subject ID space with gaps: the wire
    /// format has no explicit subject-ID list (a conforming reader always treats the `k`-th
    /// `BitmapY` group as subject `k`), so serializing one and reading it back would silently
    /// renumber subjects -- exactly the failure `matching_ids_handles_gapped_subject_ids` above
    /// guards against for the in-memory structure. `write` must refuse instead, and must not
    /// emit any bytes before detecting that.
    #[test]
    fn write_rejects_gapped_subject_ids() {
        let id_triples: BTreeSet<(u32, u32, u32)> = [(1, 1, 1), (5, 1, 1)].into_iter().collect();
        let triples = BitmapTriples::build(&id_triples);
        let mut buffer = Vec::new();
        let error = triples
            .write(&mut buffer)
            .expect_err("gapped subject IDs cannot round-trip through the wire format");
        assert!(matches!(error, HdtError::NonDenseSubjectIds));
        assert!(
            buffer.is_empty(),
            "a rejected write must not emit partial bytes"
        );
    }

    #[test]
    fn wire_roundtrip() -> Result<()> {
        let id_triples = sample();
        let triples = BitmapTriples::build(&id_triples);
        let mut buffer = Vec::new();
        triples.write(&mut buffer)?;
        let mut pos = 0;
        let decoded = BitmapTriples::read(&buffer, &mut pos)?;
        assert_eq!(pos, buffer.len());
        assert_eq!(decoded.len(), triples.len());
        assert_eq!(
            decoded.iter_ids().collect::<Vec<_>>(),
            triples.iter_ids().collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn wire_roundtrip_empty() -> Result<()> {
        let triples = BitmapTriples::build(&BTreeSet::new());
        let mut buffer = Vec::new();
        triples.write(&mut buffer)?;
        let mut pos = 0;
        let decoded = BitmapTriples::read(&buffer, &mut pos)?;
        assert!(decoded.is_empty());
        Ok(())
    }

    #[test]
    fn rejects_inconsistent_num_triples() {
        let triples = BitmapTriples::build(&sample());
        let mut buffer = Vec::new();
        triples
            .write(&mut buffer)
            .expect("dense subject ids write cleanly");
        // Corrupt the "numTriples" ASCII digit inside the control-information properties.
        // The control info always precedes the arrays, and numTriples=6 appears as the ASCII
        // byte '6' -- flip it to '5' by finding the "numTriples=" marker.
        let marker = b"numTriples=6";
        let position = buffer
            .windows(marker.len())
            .position(|window| window == marker)
            .expect("numTriples marker present");
        buffer[position + marker.len() - 1] = b'5';
        // The CRC16 covering the control information will now fail first, which is itself a
        // legitimate corruption signal.
        let mut pos = 0;
        let error = BitmapTriples::read(&buffer, &mut pos).expect_err("expected an error");
        assert!(matches!(error, HdtError::CrcMismatch { .. }));
    }

    /// A triples section whose `ControlInformation` declares a `format` this crate does not
    /// implement (e.g. the alternative `triplesList` encoding) must be rejected with a clear
    /// [`HdtError::UnsupportedFormat`], rather than being handed to the `BitmapTriples` decoder
    /// it does not actually describe.
    #[test]
    fn read_rejects_an_unrecognized_format() {
        let mut buffer = Vec::new();
        ControlInformation::new(ControlInfoType::Triples, "<http://example.com/triplesList>")
            .with_property("order", ORDER_SPO.to_string())
            .write(&mut buffer);
        let mut pos = 0;
        let error = BitmapTriples::read(&buffer, &mut pos).expect_err("expected an error");
        assert!(matches!(
            error,
            HdtError::UnsupportedFormat {
                section: SECTION,
                ..
            }
        ));
    }

    /// Hand-assembles a triples section byte stream from its raw components, bypassing
    /// [`BitmapTriples::build`]/[`BitmapTriples::write`] entirely -- exactly what an
    /// adversarially crafted `.hdt` file can do, since `BitmapY`, `BitmapZ`, `ArrayY`, and
    /// `ArrayZ` are each independently framed (their own length fields and checksums) and
    /// nothing on the wire otherwise ties them together.
    fn assemble_triples_section(
        num_triples: u64,
        bitmap_y: &Bitmap,
        bitmap_z: &Bitmap,
        array_y_values: &[u64],
        array_z_values: &[u64],
    ) -> Vec<u8> {
        let mut buffer = Vec::new();
        ControlInformation::new(ControlInfoType::Triples, TRIPLES_FORMAT)
            .with_property("order", ORDER_SPO.to_string())
            .with_property("numTriples", num_triples.to_string())
            .write(&mut buffer);
        bitmap_y.write(&mut buffer);
        bitmap_z.write(&mut buffer);
        BitArray::from_values(array_y_values).write(&mut buffer);
        BitArray::from_values(array_z_values).write(&mut buffer);
        buffer
    }

    /// `BitmapZ` must close exactly as many `(subject, predicate)` groups (its own count of `1`
    /// bits) as `ArrayY` declares -- one per group, always, in an honestly built structure (see
    /// [`build_group_levels`]'s docs). A file that understates this leaves an `ArrayY` entry's
    /// object range permanently unreachable through any query while `numTriples` (validated
    /// just above this check) still reports the full, honest count: a silent-wrong path, not
    /// merely a missing-data one, so `read` must reject it outright.
    #[test]
    fn read_rejects_bitmap_z_group_count_mismatch() {
        // Two (subject, predicate) pairs (ArrayY has 2 entries; BitmapY closes both).
        let mut bitmap_y_builder = BitmapBuilder::zeroed(2);
        bitmap_y_builder.set_one(0);
        bitmap_y_builder.set_one(1);
        // BitmapZ has 2 entries (one object per pair) but only ever closes 1 group, not 2.
        let mut bitmap_z_builder = BitmapBuilder::zeroed(2);
        bitmap_z_builder.set_one(1);
        let buffer = assemble_triples_section(
            2,
            &bitmap_y_builder.finish(),
            &bitmap_z_builder.finish(),
            &[10, 20],
            &[100, 200],
        );
        let mut pos = 0;
        let error = BitmapTriples::read(&buffer, &mut pos).expect_err("expected an error");
        assert!(
            matches!(&error, HdtError::Malformed(message) if message.contains("BitmapZ closes")),
            "expected the BitmapZ/ArrayY group-count check specifically, got: {error}"
        );
    }

    /// `BitmapY`'s very last bit must always be `1`: the last subject's last predicate
    /// necessarily closes both its own `(subject, predicate)` group and the subject's own group
    /// (see [`build_group_levels`]'s docs). A conforming writer can never leave it unset, so a
    /// file that does can only be corrupted or adversarially crafted.
    #[test]
    fn read_rejects_bitmap_y_last_bit_unset() {
        // One (subject, predicate) pair, but BitmapY's only bit is left unset.
        let bitmap_y = BitmapBuilder::zeroed(1).finish();
        let mut bitmap_z_builder = BitmapBuilder::zeroed(1);
        bitmap_z_builder.set_one(0);
        let buffer =
            assemble_triples_section(1, &bitmap_y, &bitmap_z_builder.finish(), &[10], &[100]);
        let mut pos = 0;
        let error = BitmapTriples::read(&buffer, &mut pos).expect_err("expected an error");
        assert!(
            matches!(&error, HdtError::Malformed(message) if message.contains("BitmapY's last bit")),
            "expected the BitmapY last-bit check specifically, got: {error}"
        );
    }

    /// `BitmapZ`'s very last bit must likewise always be `1` (the very last object always
    /// closes its `(subject, predicate)` group). This is independent of the group-count check
    /// above: the count here still matches (`2` closed groups, `2` `ArrayY` entries), so only
    /// the last-bit check can catch the final object being left permanently unclosed.
    #[test]
    fn read_rejects_bitmap_z_last_bit_unset() {
        let mut bitmap_y_builder = BitmapBuilder::zeroed(2);
        bitmap_y_builder.set_one(0);
        bitmap_y_builder.set_one(1);
        // 3 objects, but BitmapZ only ever closes 2 groups and never the 3rd (its last bit).
        let mut bitmap_z_builder = BitmapBuilder::zeroed(3);
        bitmap_z_builder.set_one(0);
        bitmap_z_builder.set_one(1);
        let buffer = assemble_triples_section(
            3,
            &bitmap_y_builder.finish(),
            &bitmap_z_builder.finish(),
            &[10, 20],
            &[100, 101, 102],
        );
        let mut pos = 0;
        let error = BitmapTriples::read(&buffer, &mut pos).expect_err("expected an error");
        assert!(
            matches!(&error, HdtError::Malformed(message) if message.contains("BitmapZ's last bit")),
            "expected the BitmapZ last-bit check specifically, got: {error}"
        );
    }
}
