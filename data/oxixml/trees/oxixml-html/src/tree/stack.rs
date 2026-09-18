//! The stack of open elements
//! ([HTML § 13.2.4.2](https://html.spec.whatwg.org/multipage/parsing.html#the-stack-of-open-elements))
//! and the six element scopes defined on it.

use std::collections::HashMap;
use std::ops::Deref;

use oxixml_dom::NodeId;

use super::TreeBuilder;
use crate::ns::{Namespace, SCOPE_HTML, SCOPE_MATHML, SCOPE_SVG};

/// One entry of the stack of open elements.
///
/// The element's namespace and local name are cached alongside its node
/// identifier because every tree construction algorithm tests them, and going
/// back to the DOM for each test would dominate the parser's running time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OpenElement {
    pub(crate) id: NodeId,
    pub(crate) namespace: Namespace,
    pub(crate) name: String,
}

impl OpenElement {
    pub(crate) fn new(id: NodeId, namespace: Namespace, name: impl Into<String>) -> Self {
        Self {
            id,
            namespace,
            name: name.into(),
        }
    }

    /// Returns `true` when this is an HTML element with the given local name.
    pub(crate) fn is_html(&self, name: &str) -> bool {
        self.namespace == Namespace::Html && self.name == name
    }

    /// Returns `true` when this is an HTML element with any of the given local
    /// names.
    pub(crate) fn is_html_any(&self, names: &[&str]) -> bool {
        self.namespace == Namespace::Html && names.contains(&self.name.as_str())
    }
}

/// The stack of open elements, with an index of the HTML local names it holds.
///
/// The index exists because the "have an element in scope" and "has an open
/// template" tests run on essentially every token and, when the element they
/// look for is *absent*, walk the whole stack before failing — the scope
/// boundaries (`html`, `table`, `button`, …) do not include the ordinary
/// containers a real document nests deeply. Counting names lets those tests fail
/// in O(1), which is what keeps tree construction linear rather than quadratic in
/// the nesting depth of the document.
///
/// Reads go through [`Deref`] to a plain slice; every *mutation* has to go
/// through the inherent methods below so the index cannot drift.
#[derive(Debug, Clone, Default)]
pub(crate) struct OpenElements {
    items: Vec<OpenElement>,
    /// How many currently-open *HTML* elements carry each local name.
    html_counts: HashMap<String, usize>,
}

impl Deref for OpenElements {
    type Target = [OpenElement];

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

impl OpenElements {
    fn record(&mut self, element: &OpenElement) {
        if element.namespace == Namespace::Html {
            *self.html_counts.entry(element.name.clone()).or_insert(0) += 1;
        }
    }

    fn forget(&mut self, element: &OpenElement) {
        if element.namespace != Namespace::Html {
            return;
        }
        if let Some(count) = self.html_counts.get_mut(&element.name) {
            *count -= 1;
            if *count == 0 {
                self.html_counts.remove(&element.name);
            }
        }
    }

    /// Returns whether any HTML element with this local name is open.
    pub(crate) fn contains_html(&self, name: &str) -> bool {
        self.html_counts.contains_key(name)
    }

    pub(crate) fn push(&mut self, element: OpenElement) {
        self.record(&element);
        self.items.push(element);
    }

    pub(crate) fn pop(&mut self) -> Option<OpenElement> {
        let element = self.items.pop()?;
        self.forget(&element);
        Some(element)
    }

    pub(crate) fn insert(&mut self, index: usize, element: OpenElement) {
        self.record(&element);
        self.items.insert(index, element);
    }

    pub(crate) fn remove(&mut self, index: usize) -> OpenElement {
        let element = self.items.remove(index);
        self.forget(&element);
        element
    }

    pub(crate) fn truncate(&mut self, len: usize) {
        while self.items.len() > len {
            let _ = self.pop();
        }
    }

    /// Replaces the entry at `index`, which `IndexMut` would not be able to do
    /// without letting the name index drift.
    pub(crate) fn replace(&mut self, index: usize, element: OpenElement) {
        let Some(slot) = self.items.get_mut(index) else {
            return;
        };
        let previous = std::mem::replace(slot, element);
        let replacement = self.items[index].clone();
        self.forget(&previous);
        self.record(&replacement);
    }
}

/// Which of the six scopes a "have an element in scope" test uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scope {
    /// The default scope.
    Default,
    /// The default scope plus `ol` and `ul`.
    ListItem,
    /// The default scope plus `button`.
    Button,
    /// `html`, `table` and `template` only.
    Table,
}

impl Scope {
    /// Returns `true` when `node` terminates the search for this scope.
    fn is_boundary(self, node: &OpenElement) -> bool {
        match self {
            Self::Table => node.is_html_any(&["html", "table", "template"]),
            Self::Default | Self::ListItem | Self::Button => {
                let base = match node.namespace {
                    Namespace::Html => SCOPE_HTML.contains(&node.name.as_str()),
                    Namespace::MathMl => {
                        SCOPE_MATHML.contains(&node.name.as_str()) || node.name == "annotation-xml"
                    }
                    Namespace::Svg => SCOPE_SVG.contains(&node.name.as_str()),
                };
                base || match self {
                    Self::ListItem => node.is_html_any(&["ol", "ul"]),
                    Self::Button => node.is_html("button"),
                    _ => false,
                }
            }
        }
    }
}

impl TreeBuilder<'_> {
    /// The current node: the bottommost node of the stack of open elements.
    pub(crate) fn current_node(&self) -> Option<&OpenElement> {
        self.open.last()
    }

    /// The adjusted current node, which in the fragment case is the context
    /// element while the stack holds only the root.
    pub(crate) fn adjusted_current_node(&self) -> Option<&OpenElement> {
        if self.fragment && self.open.len() == 1 {
            self.context.as_ref()
        } else {
            self.open.last()
        }
    }

    /// Returns `true` when the adjusted current node is a MathML or SVG
    /// element, which is what makes `<![CDATA[` meaningful to the tokenizer.
    pub(crate) fn adjusted_current_is_foreign(&self) -> bool {
        self.adjusted_current_node()
            .is_some_and(|node| node.namespace != Namespace::Html)
    }

    /// "Have an element in the given scope", testing HTML elements by name.
    pub(crate) fn has_html_in_scope(&self, scope: Scope, name: &str) -> bool {
        // An element that is not open at all is certainly not in scope, and the
        // name index answers that without walking the stack.
        self.open.contains_html(name) && self.has_in_scope(scope, |node| node.is_html(name))
    }

    /// "Have an element in the given scope", testing several HTML names.
    pub(crate) fn has_any_html_in_scope(&self, scope: Scope, names: &[&str]) -> bool {
        names.iter().any(|name| self.open.contains_html(name))
            && self.has_in_scope(scope, |node| node.is_html_any(names))
    }

    /// "Have a particular element in the given scope", testing node identity.
    pub(crate) fn has_node_in_scope(&self, scope: Scope, id: NodeId) -> bool {
        self.has_in_scope(scope, |node| node.id == id)
    }

    /// The generic "have an element in scope" algorithm: walk the stack from
    /// the bottom, succeed on a match and fail on a scope boundary.
    pub(crate) fn has_in_scope(
        &self,
        scope: Scope,
        matches: impl Fn(&OpenElement) -> bool,
    ) -> bool {
        for node in self.open.iter().rev() {
            if matches(node) {
                return true;
            }
            if scope.is_boundary(node) {
                return false;
            }
        }
        false
    }

    /// Pops the current node off the stack of open elements, running the
    /// reaction HTML defines for an `option` element being "popped off the
    /// stack of open elements"
    /// ([HTML § the `option` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-option-element)) —
    /// the only such per-element reaction this crate implements.
    ///
    /// Every helper below that pops more than the current node alone
    /// (`pop_until_html_named` and friends) goes through this, so an `option`
    /// closed implicitly by a later sibling or an enclosing end tag reacts
    /// exactly as one closed by its own end tag would.
    pub(crate) fn pop_open_element(&mut self) -> Option<OpenElement> {
        let popped = self.open.pop()?;
        if popped.is_html("option") {
            self.maybe_clone_option_into_selectedcontent(popped.id);
        }
        Some(popped)
    }

    /// Pops elements until one with the given HTML name has been popped.
    pub(crate) fn pop_until_html_named(&mut self, name: &str) {
        while let Some(node) = self.pop_open_element() {
            if node.is_html(name) {
                return;
            }
        }
    }

    /// Pops elements until one with any of the given HTML names has been
    /// popped.
    pub(crate) fn pop_until_any_html_named(&mut self, names: &[&str]) {
        while let Some(node) = self.pop_open_element() {
            if node.is_html_any(names) {
                return;
            }
        }
    }

    /// Pops elements until the element with the given identifier has been
    /// popped.
    pub(crate) fn pop_until_id(&mut self, id: NodeId) {
        while let Some(node) = self.pop_open_element() {
            if node.id == id {
                return;
            }
        }
    }

    /// The index of an element on the stack, if it is still open.
    ///
    /// Searched from the *back*: a node occurs at most once on the stack of open
    /// elements, so the result is the same either way, but essentially every
    /// caller (through `appropriate_place`, on every single node insertion) asks
    /// about the current node — the last entry. Scanning from the front instead
    /// made tree construction O(depth) per insertion, i.e. quadratic in the size
    /// of a merely deeply nested document.
    pub(crate) fn index_of(&self, id: NodeId) -> Option<usize> {
        self.open.iter().rposition(|node| node.id == id)
    }

    /// Removes an element from the stack wherever it is.
    pub(crate) fn remove_from_stack(&mut self, id: NodeId) {
        if let Some(index) = self.index_of(id) {
            self.open.remove(index);
        }
    }

    /// Returns `true` when a `template` element is open.
    pub(crate) fn has_open_template(&self) -> bool {
        self.open.contains_html("template")
    }

    /// "Clear the stack back to a table context".
    pub(crate) fn clear_stack_to_table_context(&mut self) {
        while let Some(node) = self.open.last() {
            if node.is_html_any(&["table", "template", "html"]) {
                return;
            }
            self.open.pop();
        }
    }

    /// "Clear the stack back to a table body context".
    pub(crate) fn clear_stack_to_table_body_context(&mut self) {
        while let Some(node) = self.open.last() {
            if node.is_html_any(&["tbody", "tfoot", "thead", "template", "html"]) {
                return;
            }
            self.open.pop();
        }
    }

    /// "Clear the stack back to a table row context".
    pub(crate) fn clear_stack_to_table_row_context(&mut self) {
        while let Some(node) = self.open.last() {
            if node.is_html_any(&["tr", "template", "html"]) {
                return;
            }
            self.open.pop();
        }
    }
}
