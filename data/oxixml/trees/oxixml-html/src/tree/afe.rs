//! The list of active formatting elements
//! ([HTML § 13.2.4.3](https://html.spec.whatwg.org/multipage/parsing.html#the-list-of-active-formatting-elements))
//! and the adoption agency algorithm
//! ([HTML § 13.2.6.4.7](https://html.spec.whatwg.org/multipage/parsing.html#adoptionAgency)).
//!
//! Together these are what makes a browser turn `<b>1<p>2</b>3<p>4` into a tree
//! in which the bold formatting is reopened inside each paragraph. The list
//! remembers which formatting elements were opened but not yet closed; the
//! adoption agency algorithm repairs the tree when a formatting element and a
//! block element are misnested.

use oxixml_dom::NodeId;

use super::{OpenElement, Scope, TreeBuilder};
use crate::error::codes;
use crate::ns::{Namespace, is_special};
use crate::token::Tag;

/// One entry of the list of active formatting elements.
#[derive(Debug, Clone)]
pub(crate) enum FormatEntry {
    /// A marker, inserted when entering an `applet`, `object`, `marquee`,
    /// `template`, `td`, `th` or `caption` element.
    Marker,
    /// A formatting element, remembered together with the token that created
    /// it so that it can be recreated.
    Element {
        /// The element currently standing for this entry.
        id: NodeId,
        /// The start tag token the entry was created from.
        tag: Tag,
    },
}

impl FormatEntry {
    /// The node identifier of an element entry.
    pub(crate) const fn id(&self) -> Option<NodeId> {
        match self {
            Self::Marker => None,
            Self::Element { id, .. } => Some(*id),
        }
    }
}

/// Compares two attribute lists as unordered sets, as the Noah's Ark clause
/// requires.
fn same_attributes(left: &Tag, right: &Tag) -> bool {
    left.attributes.len() == right.attributes.len()
        && left
            .attributes
            .iter()
            .all(|attribute| right.attribute(&attribute.name) == Some(attribute.value.as_str()))
}

impl TreeBuilder<'_> {
    /// The index of the last marker in the list, or `None` when there is none.
    fn last_marker(&self) -> Option<usize> {
        self.formatting
            .iter()
            .rposition(|entry| matches!(entry, FormatEntry::Marker))
    }

    /// "Push onto the list of active formatting elements", including the
    /// Noah's Ark clause that keeps at most three equal entries.
    ///
    /// Also enforces [`crate::ParseOptions::max_depth`], the same ceiling
    /// [`TreeBuilder::insert_element`] enforces on the stack
    /// of open elements, and the same way: by checking the ceiling *first*
    /// and only doing any further work when under it. That ordering matters
    /// here, not just for symmetry with [`TreeBuilder::insert_element`] --
    /// the Noah's Ark scan below is itself an `O(length)` walk (as is
    /// [`Self::last_marker`], which it needs first), so running it
    /// unconditionally on every call, the way an earlier version of this
    /// function did, would cost `O(length)` per formatting start tag for as
    /// long as a document keeps nesting past the ceiling -- exactly the
    /// `depth * max_depth` blow-up this ceiling exists to prevent, just
    /// relocated to this function instead of removed. Checking the ceiling
    /// first makes every call past it `O(1)`, at the cost of also giving up
    /// Noah's Ark's own pruning once at the ceiling (a duplicate that would
    /// otherwise have been swapped out for a fresher entry is left in
    /// place); that is not a loss of spec fidelity beyond what is already
    /// documented here, since [`TreeBuilder::insert_element`]'s own
    /// documentation already establishes that nothing past the ceiling is
    /// tracked with full fidelity, and the two costs are inherently coupled
    /// -- deciding whether to prune requires the same scan that is too
    /// expensive to repeat unconditionally.
    ///
    /// A document that opens enormously many *differently-attributed*
    /// formatting elements defeats Noah's Ark (which only bounds repeats of
    /// the *same* name and attributes) without ever growing the stack of
    /// open elements past its own ceiling, since a formatting element that
    /// stopped being tracked as open there would otherwise still reach this
    /// function unconditionally; the ceiling check above is what keeps that
    /// case bounded too.
    pub(crate) fn push_formatting_element(&mut self, id: NodeId, tag: Tag) {
        if self.formatting.len() >= self.max_depth {
            if !self.formatting_max_depth_exceeded {
                self.formatting_max_depth_exceeded = true;
                self.error(codes::MAX_DEPTH_EXCEEDED);
            }
            return;
        }
        let start = self.last_marker().map_or(0, |index| index + 1);
        let mut equal = Vec::new();
        for index in start..self.formatting.len() {
            #[cfg(test)]
            {
                self.noah_ark_comparisons += 1;
            }
            if let FormatEntry::Element {
                tag: candidate_tag, ..
            } = &self.formatting[index]
            {
                if candidate_tag.name == tag.name && same_attributes(candidate_tag, &tag) {
                    equal.push(index);
                }
            }
        }
        if equal.len() >= 3 {
            self.formatting.remove(equal[0]);
        }
        self.formatting.push(FormatEntry::Element { id, tag });
    }

    /// "Clear the list of active formatting elements up to the last marker".
    pub(crate) fn clear_formatting_to_marker(&mut self) {
        while let Some(entry) = self.formatting.pop() {
            if matches!(entry, FormatEntry::Marker) {
                return;
            }
        }
    }

    /// The index in the list of the last entry with this tag name that is after
    /// the last marker.
    pub(crate) fn formatting_entry_named(&self, name: &str) -> Option<usize> {
        let start = self.last_marker().map_or(0, |index| index + 1);
        (start..self.formatting.len()).rev().find(|&index| {
            matches!(&self.formatting[index], FormatEntry::Element { tag, .. } if tag.name == name)
        })
    }

    /// The index in the list of the entry for a particular element.
    pub(crate) fn formatting_index_of(&self, id: NodeId) -> Option<usize> {
        self.formatting
            .iter()
            .position(|entry| entry.id() == Some(id))
    }

    /// "Reconstruct the active formatting elements".
    ///
    /// Once [`TreeBuilder::max_depth`] has been reached, this becomes a
    /// no-op: reconstruction's whole job is to re-open a formatting element
    /// at its exact former position in the tree, which is precisely the
    /// structural fidelity [`TreeBuilder::insert_element`]'s
    /// documentation already gives up past that ceiling. Skipping it there
    /// is not just consistent with that trade-off, it is load-bearing:
    /// without it, every entry at the tail of the list of active formatting
    /// elements that fell off the (now frozen) stack of open elements has to
    /// be rewound past and recreated on *every subsequent call*, and that
    /// tail only ever grows once the stack has stopped tracking new
    /// elements — an unbounded document would otherwise make this function
    /// itself cost O(depth) per call, the same class of blow-up
    /// [`TreeBuilder::max_depth`] exists to prevent elsewhere.
    pub(crate) fn reconstruct_formatting_elements(&mut self) {
        if self.max_depth_exceeded {
            return;
        }
        let Some(last) = self.formatting.last() else {
            return;
        };
        match last {
            FormatEntry::Marker => return,
            FormatEntry::Element { id, .. } if self.index_of(*id).is_some() => return,
            FormatEntry::Element { .. } => {}
        }

        // Rewind to the first entry that is a marker or still open. Bounded
        // to `max_depth` steps as defence in depth: the stack of open
        // elements can hold at most that many entries, so a well-formed
        // list of active formatting elements never needs to rewind further
        // than that to find one of them (or a marker).
        let mut index = self.formatting.len() - 1;
        let mut steps = 0usize;
        while index > 0 && steps < self.max_depth {
            let previous = index - 1;
            let stop = match &self.formatting[previous] {
                FormatEntry::Marker => true,
                FormatEntry::Element { id, .. } => self.index_of(*id).is_some(),
            };
            if stop {
                break;
            }
            index = previous;
            steps += 1;
        }

        // Advance, recreating each entry.
        while index < self.formatting.len() {
            let FormatEntry::Element { tag, .. } = self.formatting[index].clone() else {
                index += 1;
                continue;
            };
            let element = self.insert_html_element(&tag);
            self.formatting[index] = FormatEntry::Element { id: element, tag };
            index += 1;
        }
    }

    /// The adoption agency algorithm, run for an end tag whose name is one of
    /// the formatting elements.
    ///
    /// Returns `true` when the algorithm handled the token, and `false` when
    /// the caller should fall back to the "any other end tag" steps.
    pub(crate) fn adoption_agency(&mut self, subject: &str) -> bool {
        // Step 2.
        if let Some(current) = self.current_node() {
            if current.is_html(subject) && self.formatting_index_of(current.id).is_none() {
                self.open.pop();
                return true;
            }
        }

        for _ in 0..8 {
            // Step 4.3.
            let Some(formatting_index) = self.formatting_entry_named(subject) else {
                return false;
            };
            let FormatEntry::Element {
                id: formatting_id,
                tag: formatting_tag,
            } = self.formatting[formatting_index].clone()
            else {
                return false;
            };

            // Step 4.4.
            let Some(formatting_stack_index) = self.index_of(formatting_id) else {
                self.error(codes::MISNESTED_TAG);
                self.formatting.remove(formatting_index);
                return true;
            };

            // Step 4.5.
            if !self.has_node_in_scope(Scope::Default, formatting_id) {
                self.error(codes::MISNESTED_TAG);
                return true;
            }

            // Step 4.6.
            if self.current_node().map(|node| node.id) != Some(formatting_id) {
                self.error(codes::MISNESTED_TAG);
            }

            // Step 4.7.
            let furthest_block = self.open[formatting_stack_index + 1..]
                .iter()
                .position(|node| is_special(node.namespace, &node.name))
                .map(|offset| formatting_stack_index + 1 + offset);

            // Step 4.8.
            let Some(mut furthest_index) = furthest_block else {
                self.open.truncate(formatting_stack_index);
                self.formatting.remove(formatting_index);
                return true;
            };
            let furthest_id = self.open[furthest_index].id;

            // Step 4.9.
            let common_ancestor = self.open[formatting_stack_index - 1].id;

            // Step 4.10.
            let mut bookmark = formatting_index;

            // Steps 4.11–4.13.
            let mut node_index = furthest_index;
            let mut last_node = furthest_id;
            let mut inner = 0_u32;
            loop {
                inner += 1;
                if node_index == 0 {
                    break;
                }
                node_index -= 1;
                let node = self.open[node_index].clone();
                if node.id == formatting_id {
                    break;
                }
                let mut entry = self.formatting_index_of(node.id);
                if inner > 3 {
                    if let Some(index) = entry {
                        self.formatting.remove(index);
                        if index < bookmark {
                            bookmark -= 1;
                        }
                        entry = None;
                    }
                }
                let Some(entry_index) = entry else {
                    self.open.remove(node_index);
                    if node_index < furthest_index {
                        furthest_index -= 1;
                    }
                    continue;
                };

                // Step 4.13.6: recreate the element.
                let FormatEntry::Element { tag, .. } = self.formatting[entry_index].clone() else {
                    continue;
                };
                let replacement = self.create_detached_html_element(&tag);
                self.formatting[entry_index] = FormatEntry::Element {
                    id: replacement,
                    tag,
                };
                self.open.replace(
                    node_index,
                    OpenElement::new(replacement, Namespace::Html, node.name.clone()),
                );

                // Step 4.13.7.
                if last_node == furthest_id {
                    bookmark = entry_index + 1;
                }

                // Step 4.13.8.
                let result = self.document.append_child(replacement, last_node);
                debug_assert!(
                    result.is_ok(),
                    "`replacement` was just created detached two lines above, and `last_node` \
                     is either `furthest_id` (open on the stack, so a real element already in \
                     this document) or an earlier iteration's `replacement` -- either way a \
                     plain element that cannot be an ancestor of the brand-new `replacement`, so \
                     this can only fail if oxixml_dom's own invariants are broken: {result:?}"
                );
                last_node = replacement;
            }

            // Step 4.14.
            let point = self.appropriate_place(Some(common_ancestor));
            self.insert_at(point, last_node);

            // Steps 4.15–4.17.
            let replacement = self.create_detached_html_element(&formatting_tag);
            let children: Vec<NodeId> = self.document.children(furthest_id).collect();
            for child in children {
                let result = self.document.append_child(replacement, child);
                debug_assert!(
                    result.is_ok(),
                    "`replacement` was just created detached above and `child` is currently a \
                     child of `furthest_id` (snapshotted right before this loop), so moving it \
                     under the new, still-childless-of-its-own `replacement` cannot form a cycle \
                     or violate a kind constraint (an element's allowed children are exactly a \
                     document fragment's): {result:?}"
                );
            }
            let result = self.document.append_child(furthest_id, replacement);
            debug_assert!(
                result.is_ok(),
                "`replacement` now holds `furthest_id`'s former children (moved above), not \
                 `furthest_id` itself, so `furthest_id` adopting `replacement` as a new child \
                 cannot create a cycle: {result:?}"
            );

            // Step 4.18.
            if let Some(index) = self.formatting_index_of(formatting_id) {
                self.formatting.remove(index);
                if index < bookmark {
                    bookmark -= 1;
                }
            }
            let bookmark = bookmark.min(self.formatting.len());
            self.formatting.insert(
                bookmark,
                FormatEntry::Element {
                    id: replacement,
                    tag: formatting_tag,
                },
            );

            // Step 4.19.
            self.remove_from_stack(formatting_id);
            let insert_at = self
                .index_of(furthest_id)
                .map_or(self.open.len(), |index| index + 1);
            self.open.insert(
                insert_at,
                OpenElement::new(replacement, Namespace::Html, subject.to_owned()),
            );
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ParseOptions;

    /// Pins [`TreeBuilder::push_formatting_element`]'s cost directly rather
    /// than through wall-clock time, the way `tests/api.rs`'s sibling
    /// regression test for this same fix has to: on a heavily loaded shared
    /// machine, wall-clock time for even a cheap, correctly-bounded
    /// operation can vary by an order of magnitude from one run to the
    /// next, but this count cannot.
    ///
    /// Before this function checked its ceiling first, every one of these
    /// `depth` distinctly-attributed `<b>` elements ran a full Noah's Ark
    /// scan over the (frozen, at-the-ceiling) list once the ceiling was
    /// first reached, costing `O(depth * max_depth)` comparisons overall
    /// (roughly 1.2 million at this depth); after, the scan is skipped
    /// entirely once frozen, so the only comparisons are the ramp-up to the
    /// ceiling, bounded by `O(max_depth^2)` regardless of `depth`.
    #[test]
    fn push_formatting_element_does_not_scan_once_the_ceiling_is_frozen() {
        let depth = 5_000;
        let mut source = String::with_capacity(depth * 14);
        for i in 0..depth {
            source.push_str(&format!("<b class=\"{i}\">"));
        }

        let max_depth = ParseOptions::DEFAULT_MAX_DEPTH;
        let mut builder = TreeBuilder::new(&source, false, max_depth);
        builder.run();

        // Generous even for the bounded cost: the ramp-up to the ceiling is
        // itself a triangular sum of at most `max_depth` comparisons, i.e.
        // `O(max_depth^2)`, plus slack for `depth` itself. A regression back
        // to scanning unconditionally costs `depth * max_depth`, which
        // overshoots this bound by well over an order of magnitude at this
        // depth.
        let bound = depth + max_depth * max_depth;
        assert!(
            builder.noah_ark_comparisons <= bound,
            "push_formatting_element performed {} Noah's Ark comparisons for {depth} \
             distinctly-attributed formatting elements (bound {bound}); expected the scan to \
             stop once the list of active formatting elements reaches its ceiling",
            builder.noah_ark_comparisons
        );
        assert!(
            builder.formatting_max_depth_exceeded,
            "test premise: {depth} distinctly-attributed elements must actually reach the \
             {max_depth}-entry ceiling for this to exercise the frozen state at all"
        );
    }
}
