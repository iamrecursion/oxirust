//! The "Customizable Select" reactions of
//! [HTML § the `option` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-option-element)
//! and
//! [HTML § the `selectedcontent` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-selectedcontent-element):
//! a `select`'s first-child `button` may hold a `selectedcontent` element,
//! which mirrors a deep clone of the currently selected `option`'s contents.
//!
//! This is a DOM-mutation feature layered *on top of* tree construction
//! proper — the standard's own words are that these algorithms run when an
//! `option` element is popped off the stack of open elements, or when a
//! `selectedcontent` element connects to the document — rather than being
//! part of the § 13.2.6 insertion-mode state machine. A full implementation
//! tracks live selectedness through script and user interaction; this parser
//! only ever runs forward through source text once, with scripting off, so
//! the observable subset that matters is: for each `select`, which `option`
//! ends up selected once every `option` up to a given point has been seen,
//! and what the enabled `selectedcontent` (if any) should contain once that
//! is known.
//!
//! Two standard algorithms are reproduced almost verbatim
//! ([`TreeBuilder::maybe_clone_option_into_selectedcontent`],
//! [`TreeBuilder::update_select_selectedcontent`]); the "selectedness setting
//! algorithm" itself is not — since options are only ever discovered once,
//! in tree order, on this parser's single forward pass, running its
//! full list-of-options rescan on every `option` insertion would be
//! quadratic in the size of the `select`. [`SelectState`] instead tracks the
//! two facts that algorithm ever changes incrementally, in O(1) per
//! insertion; [`TreeBuilder::selected_option`] documents why that gives
//! identical results.

use oxixml_dom::NodeId;

use super::TreeBuilder;
use crate::ns::HTML_NAMESPACE;

/// Per-`select` bookkeeping the `selectedcontent`-cloning reactions need
/// while parsing, keyed by the `select` element's node id in
/// [`TreeBuilder::select_state`].
///
/// A `select` with a `multiple` attribute never gets an entry: neither step
/// of the selectedness setting algorithm applies to one (both require
/// `multiple` to be absent), and a `multiple` select never has an "enabled
/// `selectedcontent`" either
/// (see [`TreeBuilder::select_enabled_selectedcontent`]), so tracking it
/// would only cost time for no observable effect.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SelectState {
    /// Whether `select`'s display size is 1 — the extra condition the
    /// selectedness setting algorithm's first step places on the "no option
    /// is selected yet" default-first-option rule. (Its second step, the
    /// "two or more are selected" demotion, only ever requires `multiple` to
    /// be absent, which every entry already satisfies.)
    allow_default_first: bool,
    /// The first non-disabled `option` seen so far, in tree order — the
    /// fallback selection when no `option` carries the `selected` attribute.
    first_enabled_option: Option<NodeId>,
    /// The most recently seen `option` that carried the `selected` attribute
    /// at creation time — "the last one wins".
    last_selected_attribute_option: Option<NodeId>,
}

impl SelectState {
    fn new(allow_default_first: bool) -> Self {
        Self {
            allow_default_first,
            first_enabled_option: None,
            last_selected_attribute_option: None,
        }
    }
}

impl TreeBuilder<'_> {
    // ------------------------------------------------------------ reactions

    /// Runs the reaction HTML's `option` element "insertion steps" trigger
    /// via "update an option's nearest ancestor select": this parser never
    /// moves or removes an `option` once inserted, so of that algorithm's
    /// three steps only "run the selectedness setting algorithm given
    /// `newSelect`" ever applies.
    ///
    /// Call this right after inserting an `option` element.
    pub(crate) fn note_option_inserted(&mut self, option: NodeId) {
        let Some(select) = self.option_nearest_ancestor_select(option) else {
            return;
        };
        if self.document.has_attribute(select, "multiple") {
            return;
        }
        let has_selected_attribute = self.document.has_attribute(option, "selected");
        let is_disabled = self.is_option_disabled(option);
        let allow_default_first = self.display_size(select) == 1;

        let state = self
            .select_state
            .entry(select)
            .or_insert_with(|| SelectState::new(allow_default_first));
        if has_selected_attribute {
            state.last_selected_attribute_option = Some(option);
        }
        if state.first_enabled_option.is_none() && !is_disabled {
            state.first_enabled_option = Some(option);
        }
    }

    /// "To maybe clone an `option` into `selectedcontent`, given an `option`
    /// `option`"
    /// ([HTML § the `option` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-option-element)).
    ///
    /// The standard has the HTML and XML parsers run this whenever an
    /// `option` element is popped off the stack of open elements — including
    /// via "stop parsing", when parsing ends with the element still open
    /// (see [`TreeBuilder::stop_parsing`]).
    pub(crate) fn maybe_clone_option_into_selectedcontent(&mut self, option: NodeId) {
        let Some(select) = self.option_nearest_ancestor_select(option) else {
            return;
        };
        if self.selected_option(select) != Some(option) {
            return;
        }
        let Some(selectedcontent) = self.select_enabled_selectedcontent(select) else {
            return;
        };
        self.clone_option_into_selectedcontent(option, selectedcontent);
    }

    /// The `selectedcontent` element's "HTML element post-connection steps"
    /// ([HTML § the `selectedcontent` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-selectedcontent-element)).
    ///
    /// The standard's generic "post-connection steps" run once a whole
    /// subtree has finished connecting; this parser's "insert an HTML
    /// element" always inserts exactly one node at a time (its children, if
    /// any, are parsed and inserted afterwards, each through their own call),
    /// so there is no separate queued step to reproduce — the reaction can
    /// run synchronously right after insertion.
    ///
    /// Call this right after inserting a `selectedcontent` element.
    pub(crate) fn note_selectedcontent_inserted(&mut self, selectedcontent: NodeId) {
        let (nearest_select, disabled) = self.selectedcontent_ancestry(selectedcontent);
        let Some(select) = nearest_select else {
            return;
        };
        if disabled || self.document.has_attribute(select, "multiple") {
            return;
        }
        self.update_select_selectedcontent(select);
        self.clear_select_non_primary_selectedcontent(select);
    }

    // --------------------------------------------------------- algorithms

    /// "To update a `select`'s `selectedcontent` given a `select` element
    /// `select`"
    /// ([HTML § the `selectedcontent` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-selectedcontent-element)).
    fn update_select_selectedcontent(&mut self, select: NodeId) {
        let Some(selectedcontent) = self.select_enabled_selectedcontent(select) else {
            return;
        };
        match self.selected_option(select) {
            Some(option) => self.clone_option_into_selectedcontent(option, selectedcontent),
            None => self.clear_selectedcontent(selectedcontent),
        }
    }

    /// "To clear a `select`'s non-primary `selectedcontent` elements, given a
    /// `select` element `select`"
    /// ([HTML § the `selectedcontent` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-selectedcontent-element)):
    /// every `selectedcontent` descendant but the first, in tree order, is
    /// cleared.
    fn clear_select_non_primary_selectedcontent(&mut self, select: NodeId) {
        let rest: Vec<NodeId> = self
            .document
            .descendants(select)
            .filter(|&node| self.is_html_named(node, "selectedcontent"))
            .skip(1)
            .collect();
        for selectedcontent in rest {
            self.clear_selectedcontent(selectedcontent);
        }
    }

    /// "To clone an `option` into a `selectedcontent`, given an `option`
    /// element `option` and a `selectedcontent` element `selectedcontent`"
    /// ([HTML § the `selectedcontent` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-selectedcontent-element)):
    /// replace all of `selectedcontent`'s children with a deep clone of every
    /// child of `option`.
    fn clone_option_into_selectedcontent(&mut self, option: NodeId, selectedcontent: NodeId) {
        self.clear_selectedcontent(selectedcontent);
        let mut child = self.document.first_child(option);
        while let Some(id) = child {
            let next = self.document.next_sibling(id);
            if let Ok(clone) = self.document.clone_node(id, true) {
                let result = self.document.append_child(selectedcontent, clone);
                debug_assert!(
                    result.is_ok(),
                    "`clone` is a deep clone just created above, so it is a brand-new node \
                     distinct from every existing one -- it cannot equal `selectedcontent` or be \
                     one of its ancestors, and an element's allowed children never depend on \
                     which specific element it is: {result:?}"
                );
            }
            child = next;
        }
    }

    /// "To clear a `selectedcontent` given a `selectedcontent` element
    /// `selectedcontent`"
    /// ([HTML § the `selectedcontent` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-selectedcontent-element)):
    /// "replace all with null", i.e. remove every child.
    fn clear_selectedcontent(&mut self, selectedcontent: NodeId) {
        while let Some(child) = self.document.first_child(selectedcontent) {
            let result = self.document.remove_child(selectedcontent, child);
            debug_assert!(
                result.is_ok(),
                "`child` was just looked up as `selectedcontent`'s first child on the line \
                 above, so it is a child of `selectedcontent` at this exact point: {result:?}"
            );
        }
    }

    /// "To get a `select`'s enabled `selectedcontent` given a `select`
    /// element `select`"
    /// ([HTML § the `selectedcontent` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-selectedcontent-element)).
    fn select_enabled_selectedcontent(&self, select: NodeId) -> Option<NodeId> {
        if self.document.has_attribute(select, "multiple") {
            return None;
        }
        let candidate = self
            .document
            .descendants(select)
            .find(|&node| self.is_html_named(node, "selectedcontent"))?;
        let (_, disabled) = self.selectedcontent_ancestry(candidate);
        if disabled { None } else { Some(candidate) }
    }

    /// The `option` in `select`'s list of options whose selectedness this
    /// parser currently computes as true: the "selectedness setting
    /// algorithm"
    /// ([HTML § the `select` element](https://html.spec.whatwg.org/multipage/form-elements.html))
    /// run after every `option` seen so far has been inserted converges to
    /// the last `option` in tree order that carries the `selected`
    /// attribute or, when none does and `select`'s display size is 1, the
    /// first enabled `option` — by induction on insertion order: inserting
    /// an `option` either leaves a lone previously-selected `option` alone
    /// (fewer than two are selected), promotes a fresh `selected`-attributed
    /// `option` over it (demoting the old one, since two are now selected
    /// and it is not the last), or — when nothing was selected yet — falls
    /// back to the default-first rule. [`SelectState`] tracks exactly the
    /// two facts that recurrence depends on, so this needs no rescan.
    fn selected_option(&self, select: NodeId) -> Option<NodeId> {
        let state = self.select_state.get(&select)?;
        state.last_selected_attribute_option.or({
            if state.allow_default_first {
                state.first_enabled_option
            } else {
                None
            }
        })
    }

    // ------------------------------------------------------------- lookups

    /// "To get the nearest ancestor `select` given an `Element` `element`"
    /// ([HTML § the `option` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-option-element)).
    fn option_nearest_ancestor_select(&self, element: NodeId) -> Option<NodeId> {
        let mut ancestor_optgroup: Option<NodeId> = None;
        for ancestor in self.document.ancestors(element) {
            if self.is_html_named(ancestor, "datalist")
                || self.is_html_named(ancestor, "hr")
                || self.is_html_named(ancestor, "option")
            {
                return None;
            }
            if self.is_html_named(ancestor, "optgroup") {
                if ancestor_optgroup.is_some() {
                    return None;
                }
                ancestor_optgroup = Some(ancestor);
            }
            if self.is_html_named(ancestor, "select") {
                return Some(ancestor);
            }
        }
        None
    }

    /// The ancestor walk shared by the `selectedcontent` element's "HTML
    /// element post-connection steps" and by
    /// [`Self::select_enabled_selectedcontent`]'s "disabled" check
    /// ([HTML § the `selectedcontent` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-selectedcontent-element)):
    /// the nearest `select` ancestor (if any), and whether `selectedcontent`
    /// is disabled — nested inside another `select`, or inside an `option`
    /// or `selectedcontent` element.
    ///
    /// Computed on demand from the current tree rather than cached at
    /// connection time: this parser never reparents a node once inserted, so
    /// an ancestor chain computed later is exactly the one the standard's
    /// post-connection steps would have seen.
    fn selectedcontent_ancestry(&self, selectedcontent: NodeId) -> (Option<NodeId>, bool) {
        let mut nearest_select = None;
        for ancestor in self.document.ancestors(selectedcontent) {
            if self.is_html_named(ancestor, "select") {
                if nearest_select.is_none() {
                    nearest_select = Some(ancestor);
                    continue;
                }
                return (nearest_select, true);
            }
            if self.is_html_named(ancestor, "option")
                || self.is_html_named(ancestor, "selectedcontent")
            {
                return (nearest_select, true);
            }
        }
        (nearest_select, false)
    }

    /// An `option` element is "disabled"
    /// ([HTML § the `option` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-option-element))
    /// if it carries the `disabled` attribute itself, or if its nearest
    /// ancestor among `select`, `hr`, `datalist`, `option` and `optgroup`
    /// elements is an `optgroup` that carries the `disabled` attribute.
    fn is_option_disabled(&self, option: NodeId) -> bool {
        if self.document.has_attribute(option, "disabled") {
            return true;
        }
        for ancestor in self.document.ancestors(option) {
            if self.is_html_named(ancestor, "select")
                || self.is_html_named(ancestor, "hr")
                || self.is_html_named(ancestor, "datalist")
                || self.is_html_named(ancestor, "option")
            {
                return false;
            }
            if self.is_html_named(ancestor, "optgroup") {
                return self.document.has_attribute(ancestor, "disabled");
            }
        }
        false
    }

    /// The "display size"
    /// ([HTML § the `select` element](https://html.spec.whatwg.org/multipage/form-elements.html))
    /// of a `select` element: its `size` attribute, parsed as a non-negative
    /// integer, if that succeeds; otherwise 4 if `multiple` is present, else
    /// 1.
    fn display_size(&self, select: NodeId) -> u32 {
        if let Some(raw) = self.document.get_attribute(select, "size") {
            if let Some(value) = parse_non_negative_integer(raw) {
                return value;
            }
        }
        if self.document.has_attribute(select, "multiple") {
            4
        } else {
            1
        }
    }

    /// Whether `id` is an HTML-namespace element with local name `name`.
    fn is_html_named(&self, id: NodeId, name: &str) -> bool {
        self.document.element(id).is_some_and(|data| {
            data.namespace_uri() == Some(HTML_NAMESPACE) && data.local_name() == name
        })
    }
}

/// The "rules for parsing non-negative integers", restricted to what the
/// `size` attribute needs: optional leading ASCII whitespace, an optional
/// `+` sign, then one or more ASCII digits (trailing garbage is ignored, as
/// the standard's own algorithm allows). A leading `-` is not itself
/// special-cased — like the standard's version, it simply fails to find any
/// digits immediately after, and parsing fails.
fn parse_non_negative_integer(value: &str) -> Option<u32> {
    let trimmed = value.trim_start_matches(|ch: char| ch.is_ascii_whitespace());
    let rest = trimmed.strip_prefix('+').unwrap_or(trimmed);
    let digits_len = rest
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits_len == 0 {
        return None;
    }
    rest[..digits_len].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::parse_non_negative_integer;

    #[test]
    fn parses_plain_digits() {
        assert_eq!(parse_non_negative_integer("3"), Some(3));
        assert_eq!(parse_non_negative_integer("42"), Some(42));
    }

    #[test]
    fn accepts_leading_whitespace_and_plus_ignores_trailing_garbage() {
        assert_eq!(parse_non_negative_integer("  7"), Some(7));
        assert_eq!(parse_non_negative_integer("+7"), Some(7));
        assert_eq!(parse_non_negative_integer(" +7abc"), Some(7));
        assert_eq!(parse_non_negative_integer("7.5"), Some(7));
    }

    #[test]
    fn rejects_negative_or_absent_digits() {
        assert_eq!(parse_non_negative_integer("-1"), None);
        assert_eq!(parse_non_negative_integer(""), None);
        assert_eq!(parse_non_negative_integer("abc"), None);
        assert_eq!(parse_non_negative_integer("   "), None);
    }
}
