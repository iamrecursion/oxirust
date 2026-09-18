//! XSLT match patterns: the little language `xsl:template/@match`, `xsl:key/@match` and
//! `xsl:number/@from` are written in.
//!
//! [XSLT 3.0 §5.5.2](https://www.w3.org/TR/xslt-30/#pattern-syntax) defines a pattern as a
//! restricted path expression, and defines matching by reference to *expression* semantics: a
//! node matches a pattern when it is a member of the result of evaluating the pattern with some
//! ancestor-or-self of the node as the context item. Evaluating that definition literally is
//! quadratic, so this module compiles the pattern into a chain of steps and matches it
//! **right to left**, which is what every production processor does: test the node against the
//! last step, then walk up the parent chain testing the steps to its left.
//!
//! Predicates keep their full generality — each one is compiled as an ordinary XPath expression
//! and evaluated with the focus the specification prescribes, so `para[1]` means "the first
//! `para` child of its parent" and not "any `para`".

use crate::compile::StaticEnv;
use crate::error::{XsltError, XsltResult};
use oxixml_infoset::{ExpandedName, NodeKind, NodeRef, XdmNode};
use oxixml_xpath::{CompiledXPath, DynamicContext, Focus};

/// How one step of a pattern connects to the step on its left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Connector {
    /// `/` — the step on the left must match this node's parent.
    Parent,
    /// `//` — the step on the left must match some ancestor.
    Ancestor,
}

/// What anchors the left end of a path pattern.
#[derive(Debug)]
enum Anchor {
    /// A relative pattern: nothing further is required.
    Relative,
    /// A leading `/` or `//`: the chain must reach a document node.
    Root,
    /// A leading `id(…)` or `key(…)`: the chain must reach a node the call selects.
    Selection(CompiledXPath),
}

/// A node test inside a pattern step.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PatternTest {
    /// `*` — any node of the axis's principal kind.
    AnyName,
    /// A fully resolved `QName`.
    Name(ExpandedName),
    /// `prefix:*`.
    AnyLocal(String),
    /// `*:local`.
    AnyNamespace(String),
    /// `node()`.
    AnyNode,
    /// `text()`.
    Text,
    /// `comment()`.
    Comment,
    /// `processing-instruction()`, optionally with a target.
    ProcessingInstruction(Option<String>),
    /// `namespace-node()`.
    Namespace,
    /// `element(name?)`.
    Element(Option<ExpandedName>),
    /// `attribute(name?)`.
    Attribute(Option<ExpandedName>),
    /// `document-node(element(name?)?)`.
    DocumentNode(Option<Option<ExpandedName>>),
}

/// Which axis a pattern step walks. Patterns allow only these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatternAxis {
    Child,
    Attribute,
    Namespace,
    /// `.` and `self::`, which XSLT 3.0 allows in a pattern.
    Current,
}

#[derive(Debug)]
struct PatternStep {
    axis: PatternAxis,
    /// Whether the axis was *written*, rather than left abbreviated. The two differ for one
    /// test: `document-node()` written on its own matches a document node, while
    /// `child::document-node()` asks for a document node that is the child of something, which
    /// no node is — `match-048` writes it and expects the rule never to fire.
    written_axis: bool,
    test: PatternTest,
    predicates: Vec<CompiledXPath>,
    /// How this step connects to whatever is on its left.
    connector: Connector,
}

#[derive(Debug)]
struct Alternative {
    steps: Vec<PatternStep>,
    anchor: Anchor,
    priority: f64,
    /// A pattern the step-chain compiler cannot express — `except`, `intersect`, a
    /// parenthesized sub-pattern, a variable reference, an XPath comment — held as an ordinary
    /// expression and matched by the definition
    /// [XSLT 3.0 §5.5.3](https://www.w3.org/TR/xslt-30/#pattern-semantics) gives: a node
    /// matches when it is a member of the result of evaluating the expression with some
    /// ancestor-or-self of the node as the context item.
    expression: Option<CompiledXPath>,
    /// The same expression with `self::` in front of its first step, which is how the
    /// `child-or-top` axis of [XSLT 3.0 §5.5.3](https://www.w3.org/TR/xslt-30/#pattern-semantics)
    /// behaves at a parentless node: there the first step of a relative pattern selects the
    /// node itself as well as its children.
    top_expression: Option<CompiledXPath>,
    /// The pattern compiled as an expression, for the case where the item being matched is not
    /// a node. [XSLT 3.0 §5.5.3](https://www.w3.org/TR/xslt-30/#pattern-semantics) lets only a
    /// *predicate pattern* — `.` followed by predicates — match an atomic value, so this is
    /// set only for those.
    predicate_expression: Option<CompiledXPath>,
}

/// A compiled match pattern.
///
/// # Examples
///
/// A pattern is compiled through [`Stylesheet`](crate::Stylesheet); the type is exposed so that
/// the default priority a template gets can be inspected.
#[derive(Debug)]
pub struct Pattern {
    source: String,
    alternatives: Vec<Alternative>,
    /// Whether the alternatives are one template rule rather than several.
    ///
    /// [XSLT 3.0 §6.5](https://www.w3.org/TR/xslt-30/#conflict) splits a union pattern into one
    /// template rule per alternative so that each can carry its own default priority — and notes
    /// that "the splitting of a template rule into multiple rules occurs only if there is no
    /// explicit `priority` attribute". A declared priority governs the whole pattern, so the
    /// rule stays single: the alternatives are then tried together, and an `xsl:next-match`
    /// written inside the template moves on to another template rather than back into the same
    /// one through a second alternative.
    single_rule: bool,
}

impl Pattern {
    /// Compiles `source` in `env`.
    ///
    /// # Errors
    ///
    /// `XTSE0340` when the text is not a legal pattern, plus any static error the predicates
    /// raise.
    pub(crate) fn compile(source: &str, env: &StaticEnv) -> XsltResult<Self> {
        crate::pattern30::check(source)?;
        // Before anything tries to compile a predicate: a call to a function that is barred
        // from patterns is a static error in its own right, and reporting the XPath parser's
        // `XPST0017` instead would hide it.
        check_pattern_functions(source)?;
        let mut alternatives = Vec::new();
        for text in pattern_alternatives(source) {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Err(XsltError::stat1c(
                    "XTSE0340",
                    format!("`{source}` has an empty alternative"),
                ));
            }
            check_id_key_pattern(trimmed)?;
            match compile_alternative(trimmed, env) {
                Ok(mut alternative) => {
                    apply_predicate_pattern(&mut alternative, trimmed, env);
                    alternatives.push(alternative);
                }
                // The step-chain grammar rejects constructs the language allows — an axis it
                // does not model reads as an unbound prefix, for one — so the alternative is
                // retried as an expression. The original diagnosis stands if that fails too.
                Err(error) if error.is("XTSE0340") || error.is("XTSE0280") => {
                    match compile_expression_alternative(trimmed, env) {
                        Ok(mut alternative) => {
                            apply_predicate_pattern(&mut alternative, trimmed, env);
                            alternatives.push(alternative);
                        }
                        Err(_) => return Err(error),
                    }
                }
                Err(error) => return Err(error),
            }
        }
        if alternatives.is_empty() {
            return Err(XsltError::stat1c("XTSE0340", "a pattern may not be empty"));
        }
        Ok(Self {
            source: source.to_owned(),
            alternatives,
            single_rule: false,
        })
    }

    /// Makes the pattern one template rule over all its alternatives, as §6.5 requires when the
    /// `xsl:template` declares a `priority` of its own.
    pub(crate) fn into_single_rule(mut self) -> Self {
        self.single_rule = true;
        self
    }

    /// The pattern's source text.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The highest default priority of the pattern's alternatives.
    #[must_use]
    pub fn default_priority(&self) -> f64 {
        self.alternatives
            .iter()
            .map(|alternative| alternative.priority)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The default priority of each alternative, in order — a union pattern behaves as several
    /// templates, one per alternative — or one entry when a declared priority keeps the rule
    /// single.
    pub(crate) fn alternative_priorities(&self) -> Vec<f64> {
        if self.single_rule {
            return vec![self.default_priority()];
        }
        self.alternatives
            .iter()
            .map(|alternative| alternative.priority)
            .collect()
    }

    /// Whether `node` matches any alternative.
    ///
    /// [XSLT 3.0 §5.5.3](https://www.w3.org/TR/xslt-30/#pattern-semantics) lets a processor
    /// either report a dynamic error raised while matching a pattern or treat the pattern as
    /// not matching. An error the pattern's *own* value comparison raises — an `FO…` code, such
    /// as the `FORG0001` that `match="*[.=117]"` provokes on an element whose untyped content
    /// is not a number — is treated as a non-match, which is what the W3C suite's stylesheets
    /// assume. An error the stylesheet's own machinery raises, such as `XTDE0640` for a
    /// circular `xsl:key`, keeps its meaning and is reported.
    ///
    /// # Errors
    ///
    /// Any `XT…` dynamic error a predicate raises.
    pub(crate) fn matches(&self, node: &NodeRef, context: &DynamicContext) -> XsltResult<bool> {
        let _current = CurrentNode::push(node);
        let _output_uri = crate::run::ClearedOutputUri::enter();
        for alternative in &self.alternatives {
            match match_alternative(alternative, node, context) {
                Ok(true) => return Ok(true),
                Ok(false) => {}
                Err(error) if is_value_error(&error) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(false)
    }

    /// Whether an item that is not a node matches the alternative at `index`.
    ///
    /// # Errors
    ///
    /// Any dynamic error a predicate raises.
    pub(crate) fn matches_item_alternative(
        &self,
        index: usize,
        item: &oxixml_infoset::Item,
        context: &DynamicContext,
    ) -> XsltResult<bool> {
        // A rule a declared priority keeps single is matched against every alternative at once,
        // whichever of them the caller happens to name.
        if self.single_rule {
            return self.matches_item(item, context);
        }
        self.item_alternative_matches(index, item, context)
    }

    /// Whether an item that is not a node matches any alternative — the XSLT 3.0 pattern forms
    /// such as `.[. instance of xs:string]` or `.[. instance of map(*)]` that match items rather
    /// than nodes.
    ///
    /// # Errors
    ///
    /// Any dynamic error a predicate raises.
    pub(crate) fn matches_item(
        &self,
        item: &oxixml_infoset::Item,
        context: &DynamicContext,
    ) -> XsltResult<bool> {
        for index in 0..self.alternatives.len() {
            if self.item_alternative_matches(index, item, context)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether the alternative at `index` matches an item that is not a node, taking no notice of
    /// whether the alternatives are one rule or several.
    ///
    /// Only a *predicate pattern* can match such an item, and the predicates are evaluated with
    /// the item itself as the context item — which is as true of a map, an array or any other
    /// function item as it is of an atomic value.
    fn item_alternative_matches(
        &self,
        index: usize,
        item: &oxixml_infoset::Item,
        context: &DynamicContext,
    ) -> XsltResult<bool> {
        let Some(alternative) = self.alternatives.get(index) else {
            return Ok(false);
        };
        let Some(expression) = &alternative.predicate_expression else {
            return Ok(false);
        };
        let _output_uri = crate::run::ClearedOutputUri::enter();
        let focus = context.clone().with_context_item(item.clone());
        match expression.evaluate_in(&focus) {
            Ok(sequence) => Ok(!sequence.is_empty()),
            Err(_) => Ok(false),
        }
    }

    /// Whether `node` matches the alternative at `index`.
    ///
    /// # Errors
    ///
    /// Any dynamic error a predicate raises.
    pub(crate) fn matches_alternative(
        &self,
        index: usize,
        node: &NodeRef,
        context: &DynamicContext,
    ) -> XsltResult<bool> {
        if self.single_rule {
            return self.matches(node, context);
        }
        let _current = CurrentNode::push(node);
        let _output_uri = crate::run::ClearedOutputUri::enter();
        match self.alternatives.get(index) {
            // See [`Pattern::matches`] for why a value error is a non-match.
            Some(alternative) => match match_alternative(alternative, node, context) {
                Err(error) if is_value_error(&error) => Ok(false),
                other => other,
            },
            None => Ok(false),
        }
    }
}

/// Publishes the node being matched as the *current node* for the duration of the match, so
/// that `current()` inside a predicate means what
/// [XSLT 3.0 §18.2](https://www.w3.org/TR/xslt-30/#function-current) says it means.
struct CurrentNode(bool);

impl CurrentNode {
    fn push(node: &NodeRef) -> Self {
        match crate::run::current_runtime() {
            Some(runtime) => {
                runtime
                    .current
                    .borrow_mut()
                    .push(Some(oxixml_infoset::Item::Node(node.clone())));
                Self(true)
            }
            None => Self(false),
        }
    }
}

impl Drop for CurrentNode {
    fn drop(&mut self) {
        if self.0
            && let Some(runtime) = crate::run::current_runtime()
        {
            runtime.current.borrow_mut().pop();
        }
    }
}

// --- matching ---------------------------------------------------------------------------

/// Whether an error came from evaluating a value rather than from the stylesheet's own
/// machinery: the F&O codes (`FORG0001`, `FOCA0002`, `FOTY0012`, …) and the XPath type errors.
fn is_value_error(error: &XsltError) -> bool {
    error
        .code()
        .is_some_and(|code| code.starts_with("FO") || code.starts_with("XPTY"))
}

fn match_alternative(
    alternative: &Alternative,
    node: &NodeRef,
    context: &DynamicContext,
) -> XsltResult<bool> {
    if let Some(expression) = &alternative.expression {
        let mut anchor = Some(node.clone());
        while let Some(candidate) = anchor {
            let focus = context.clone().with_context_item(candidate.clone());
            if selection_contains(expression, node, &focus)? {
                return Ok(true);
            }
            // At a parentless node the first step of a relative pattern takes the
            // `child-or-top` axis, which selects the node itself as well as its children.
            if candidate.parent().is_none()
                && let Some(top) = &alternative.top_expression
                && selection_contains(top, node, &focus)?
            {
                return Ok(true);
            }
            anchor = candidate.parent();
        }
        return Ok(false);
    }
    if alternative.steps.is_empty() {
        return match &alternative.anchor {
            Anchor::Root => Ok(node.node_kind() == NodeKind::Document),
            Anchor::Relative => Ok(false),
            Anchor::Selection(selection) => selection_contains(
                selection,
                node,
                &context.clone().with_context_item(node.root()),
            ),
        };
    }
    match_step(alternative, alternative.steps.len() - 1, node, context)
}

fn match_step(
    alternative: &Alternative,
    index: usize,
    node: &NodeRef,
    context: &DynamicContext,
) -> XsltResult<bool> {
    let Some(step) = alternative.steps.get(index) else {
        return Ok(false);
    };
    if !step_matches(step, node, context)? {
        return Ok(false);
    }
    if index == 0 && matches!(alternative.anchor, Anchor::Relative) {
        return Ok(true);
    }
    match step.connector {
        Connector::Parent => match node.parent() {
            Some(parent) => match_left(alternative, index, &parent, context),
            None => Ok(false),
        },
        Connector::Ancestor => {
            let mut current = node.parent();
            while let Some(ancestor) = current {
                if match_left(alternative, index, &ancestor, context)? {
                    return Ok(true);
                }
                current = ancestor.parent();
            }
            Ok(false)
        }
    }
}

/// Continues matching to the left of `index` at `node`.
fn match_left(
    alternative: &Alternative,
    index: usize,
    node: &NodeRef,
    context: &DynamicContext,
) -> XsltResult<bool> {
    if index == 0 {
        return match &alternative.anchor {
            Anchor::Relative => Ok(true),
            Anchor::Root => Ok(node.node_kind() == NodeKind::Document),
            Anchor::Selection(selection) => selection_contains(
                selection,
                node,
                &context.clone().with_context_item(node.root()),
            ),
        };
    }
    match_step(alternative, index - 1, node, context)
}

fn selection_contains(
    selection: &CompiledXPath,
    node: &NodeRef,
    context: &DynamicContext,
) -> XsltResult<bool> {
    let nodes = selection.evaluate_to_nodes_in(context)?;
    Ok(nodes.iter().any(|candidate| candidate == node))
}

fn step_matches(step: &PatternStep, node: &NodeRef, context: &DynamicContext) -> XsltResult<bool> {
    if !axis_and_test_match(step, node) {
        return Ok(false);
    }
    if step.predicates.is_empty() {
        return Ok(true);
    }
    predicates_match(step, node, context)
}

fn axis_and_test_match(step: &PatternStep, node: &NodeRef) -> bool {
    let kind = node.node_kind();
    // A document-node() test is written on the child axis but can only match a document node,
    // which is never a child of anything.
    let document_test = matches!(step.test, PatternTest::DocumentNode(_));
    let axis_fits = match step.axis {
        PatternAxis::Child if document_test => !step.written_axis && kind == NodeKind::Document,
        PatternAxis::Child => !matches!(
            kind,
            NodeKind::Attribute | NodeKind::Namespace | NodeKind::Document
        ),
        PatternAxis::Attribute => kind == NodeKind::Attribute,
        PatternAxis::Namespace => kind == NodeKind::Namespace,
        PatternAxis::Current => true,
    };
    if !axis_fits {
        return false;
    }
    let principal = match step.axis {
        PatternAxis::Attribute => NodeKind::Attribute,
        PatternAxis::Namespace => NodeKind::Namespace,
        _ => NodeKind::Element,
    };
    test_matches(&step.test, node, principal)
}

fn test_matches(test: &PatternTest, node: &NodeRef, principal: NodeKind) -> bool {
    let kind = node.node_kind();
    match test {
        PatternTest::AnyName => kind == principal,
        PatternTest::Name(name) => kind == principal && node.node_name().as_ref() == Some(name),
        PatternTest::AnyLocal(namespace) => {
            kind == principal
                && node
                    .node_name()
                    .is_some_and(|name| match namespace.is_empty() {
                        // `Q{}*` names the elements in no namespace, which no prefix can be bound to.
                        true => name.namespace().is_none(),
                        false => name.namespace() == Some(namespace.as_str()),
                    })
        }
        PatternTest::AnyNamespace(local) => {
            kind == principal
                && node
                    .node_name()
                    .is_some_and(|name| name.local_name() == local)
        }
        PatternTest::AnyNode => true,
        PatternTest::Text => kind == NodeKind::Text,
        PatternTest::Comment => kind == NodeKind::Comment,
        PatternTest::ProcessingInstruction(target) => {
            kind == NodeKind::ProcessingInstruction
                && target.as_ref().is_none_or(|target| {
                    node.node_name()
                        .is_some_and(|name| name.local_name() == target)
                })
        }
        PatternTest::Namespace => kind == NodeKind::Namespace,
        PatternTest::Element(name) => {
            kind == NodeKind::Element
                && name
                    .as_ref()
                    .is_none_or(|name| node.node_name().as_ref() == Some(name))
        }
        PatternTest::Attribute(name) => {
            kind == NodeKind::Attribute
                && name
                    .as_ref()
                    .is_none_or(|name| node.node_name().as_ref() == Some(name))
        }
        PatternTest::DocumentNode(inner) => {
            kind == NodeKind::Document
                && match inner {
                    None => true,
                    Some(element) => node.children().iter().any(|child| {
                        child.node_kind() == NodeKind::Element
                            && element
                                .as_ref()
                                .is_none_or(|name| child.node_name().as_ref() == Some(name))
                    }),
                }
        }
    }
}

/// Evaluates the step's predicates with the focus the specification prescribes: the candidate
/// siblings selected by the same axis and node test, in document order.
fn predicates_match(
    step: &PatternStep,
    node: &NodeRef,
    context: &DynamicContext,
) -> XsltResult<bool> {
    let candidates: Vec<NodeRef> = match node.parent() {
        Some(parent) => match step.axis {
            PatternAxis::Attribute => parent.attributes(),
            PatternAxis::Namespace => parent.namespaces(),
            PatternAxis::Child | PatternAxis::Current => parent.children(),
        },
        // A parentless node is the whole of its own candidate sequence. XSLT 3.0 §5.5.3 defines
        // the match by asking whether the node is a member of the sequence the equivalent path
        // selects "from any possible node"; with no parent to select from, only the node itself
        // remains, so a positional predicate sees `position()` = `last()` = 1. A variable
        // declared `as="element(…)"` holds exactly such a node, and returning "no match" here
        // used to make every predicate pattern miss it.
        None => vec![node.clone()],
    };
    let mut current: Vec<NodeRef> = candidates
        .into_iter()
        .filter(|candidate| axis_and_test_match(step, candidate))
        .collect();
    for predicate in &step.predicates {
        let size = current.len();
        let mut kept = Vec::with_capacity(size);
        for (offset, candidate) in current.iter().enumerate() {
            let focus = Focus::new(candidate.clone(), offset + 1, size);
            let scoped = context.clone().with_focus(focus);
            // A predicate is applied to every candidate on the axis, because a positional
            // predicate needs the whole sequence — so a *sibling* whose value raises must not
            // abort the match. XSLT 3.0 §5.5.3 lets such an error be treated as a non-match.
            match predicate_holds(predicate, &scoped, offset + 1) {
                Ok(true) => kept.push(candidate.clone()),
                Ok(false) => {}
                Err(error) if is_value_error(&error) => {}
                Err(error) => return Err(error),
            }
        }
        current = kept;
        if !current.iter().any(|candidate| candidate == node) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A predicate is positional when it evaluates to a single number, and a truth value otherwise.
pub(crate) fn predicate_holds(
    predicate: &CompiledXPath,
    context: &DynamicContext,
    position: usize,
) -> XsltResult<bool> {
    let value = predicate.evaluate_value_in(context)?;
    if value.is_singleton()
        && let Some(atomic) = value.first().and_then(oxixml_xpath::XpathItem::as_atomic)
        && atomic.atomic_type().is_numeric()
    {
        let wanted = atomic.as_f64().unwrap_or(f64::NAN);
        return Ok((wanted - position as f64).abs() < f64::EPSILON);
    }
    Ok(value.effective_boolean_value()?)
}

// --- compiling --------------------------------------------------------------------------

fn compile_alternative(source: &str, env: &StaticEnv) -> XsltResult<Alternative> {
    let mut scanner = Scanner::new(source);
    let mut anchor = Anchor::Relative;
    let mut leading_descendant = false;
    let mut rooted = false;

    scanner.skip_whitespace();
    if scanner.eat("//") {
        anchor = Anchor::Root;
        rooted = true;
        leading_descendant = true;
    } else if scanner.eat("/") {
        anchor = Anchor::Root;
        rooted = true;
        scanner.skip_whitespace();
        if scanner.at_end() {
            return Ok(Alternative {
                steps: Vec::new(),
                anchor,
                // "If the pattern is a PathExprP taking the form `/`, then the priority is
                // −0.5" ([XSLT 3.0 §6.5](https://www.w3.org/TR/xslt-30/#conflict)): the lone
                // slash matches one node per document and is the least selective pattern there
                // is, not the most. `conflict-resolution-1601` writes three rules for `/` and
                // expects the one declaring −0.4 to outrank the one taking the default.
                priority: -0.5,
                expression: None,
                top_expression: None,
                predicate_expression: None,
            });
        }
    } else if let Some(call) = scanner.rooted_expression() {
        let compiled = env.compile_expression(&call)?;
        anchor = Anchor::Selection(compiled);
        scanner.skip_whitespace();
        if scanner.at_end() {
            return Ok(Alternative {
                steps: Vec::new(),
                anchor,
                priority: 0.5,
                expression: None,
                top_expression: None,
                predicate_expression: None,
            });
        }
        if scanner.eat("//") {
            leading_descendant = true;
        } else if !scanner.eat("/") {
            return Err(XsltError::stat1c(
                "XTSE0340",
                format!("`{source}` is not a valid pattern"),
            ));
        }
    }

    let mut steps = Vec::new();
    let mut connector = if leading_descendant {
        Connector::Ancestor
    } else {
        Connector::Parent
    };
    loop {
        scanner.skip_whitespace();
        let step = scanner.step(env, connector)?;
        steps.push(step);
        scanner.skip_whitespace();
        if scanner.eat("//") {
            connector = Connector::Ancestor;
        } else if scanner.eat("/") {
            connector = Connector::Parent;
        } else {
            break;
        }
    }
    scanner.skip_whitespace();
    if !scanner.at_end() {
        return Err(XsltError::stat1c(
            "XTSE0340",
            format!("`{source}` is not a valid pattern"),
        ));
    }

    let priority = default_priority(&steps, rooted, &anchor);
    Ok(Alternative {
        steps,
        anchor,
        priority,
        expression: None,
        top_expression: None,
        predicate_expression: None,
    })
}

/// Compiles a pattern alternative that the step-chain grammar does not cover, keeping it as an
/// expression to be matched by the specification's own definition.
///
/// # Errors
///
/// `XTSE0340` when the text is not even a well-formed expression.
fn compile_expression_alternative(source: &str, env: &StaticEnv) -> XsltResult<Alternative> {
    let expression = env.compile_expression(source).map_err(|error| {
        XsltError::stat1c(
            "XTSE0340",
            format!("`{source}` is not a valid pattern: {}", error.message()),
        )
    })?;
    let top_expression =
        child_or_top_form(source).and_then(|rewritten| env.compile_expression(&rewritten).ok());
    Ok(Alternative {
        steps: Vec::new(),
        anchor: Anchor::Relative,
        priority: 0.5,
        expression: Some(expression),
        top_expression,
        predicate_expression: None,
    })
}

/// `(function, error code)` for the functions a pattern may not call —
/// [XSLT 3.0 §5.5.2](https://www.w3.org/TR/xslt-30/#pattern-syntax).
const FUNCTIONS_BARRED_FROM_PATTERNS: &[(&str, &str)] = &[
    ("current-group", "XTSE1060"),
    ("current-grouping-key", "XTSE1070"),
    ("current-merge-group", "XTSE3470"),
    ("current-merge-key", "XTSE3500"),
];

/// Rejects a pattern that calls a function whose value depends on where it is evaluated.
///
/// # Errors
///
/// `XTSE1060`, `XTSE1070`, `XTSE3470` or `XTSE3500`, one per barred function.
fn check_pattern_functions(source: &str) -> XsltResult<()> {
    let stripped = strip_xpath_comments(source);
    for (function, code) in FUNCTIONS_BARRED_FROM_PATTERNS {
        let mut rest = stripped.as_str();
        while let Some(position) = rest.find(function) {
            let after = rest[position + function.len()..].trim_start();
            let before = rest[..position].chars().next_back();
            let boundary = before.is_none_or(|c| !c.is_alphanumeric() && c != '-' && c != '_');
            if boundary && after.starts_with('(') {
                return Err(XsltError::stat1c(
                    *code,
                    format!("`{function}()` may not be called in a pattern"),
                ));
            }
            rest = &rest[position + function.len()..];
        }
    }
    Ok(())
}

/// Checks the arguments an `id()` or `key()` call takes at the head of a pattern.
///
/// The pattern grammar of [XSLT 3.0
/// §5.5.2](https://www.w3.org/TR/xslt-30/#pattern-syntax) admits only fixed values there:
///
/// ```text
/// IdKeyPattern ::= "id" "(" IdValue ("," (VarRef | "(" ")"))? ")"
///                | "key" "(" StringLiteral "," KeyValue ("," (VarRef | "(" ")"))? ")"
/// IdValue      ::= StringLiteral | VarRef
/// KeyValue     ::= Literal | VarRef
/// ```
///
/// A pattern is matched against a node without a focus of its own, so an argument that would
/// have to be *evaluated* against one cannot be allowed: `key('k', 40+2)` is `XTSE0340` even
/// though the identical call is an ordinary expression anywhere else (match-079).
///
/// The check runs before the alternative is compiled, so that the fallback which retries a
/// rejected alternative as a plain expression cannot paper over it.
///
/// # Errors
///
/// `XTSE0340` when an argument is outside the grammar.
fn check_id_key_pattern(source: &str) -> XsltResult<()> {
    let stripped = strip_xpath_comments(source);
    let text = stripped.trim_start();
    let (name, rest) = if let Some(rest) = call_head(text, "key") {
        ("key", rest)
    } else if let Some(rest) = call_head(text, "id") {
        ("id", rest)
    } else {
        return Ok(());
    };
    let Some(arguments) = parenthesized(rest) else {
        return Ok(());
    };
    let arguments: Vec<String> = split_top_level(&arguments, ',')
        .into_iter()
        .map(|argument| argument.trim().to_owned())
        .collect();
    let reject = |what: &str| {
        Err(XsltError::stat1c(
            "XTSE0340",
            format!("`{name}()` at the head of a pattern {what}"),
        ))
    };
    // The trailing argument both calls may take names the document to search, and the grammar
    // restricts it to a variable reference or the empty sequence.
    let (leading, trailing): (&[String], Option<&String>) = match (name, arguments.len()) {
        ("id", 1) => (&arguments[..], None),
        ("id", 2) => (&arguments[..1], arguments.get(1)),
        ("key", 2) => (&arguments[..], None),
        ("key", 3) => (&arguments[..2], arguments.get(2)),
        _ => return reject("does not have the number of arguments the grammar allows"),
    };
    if let Some(last) = trailing
        && !is_variable_reference(last)
        && last.replace(char::is_whitespace, "") != "()"
    {
        return reject("may only name the document to search with a variable reference or `()`");
    }
    for (position, argument) in leading.iter().enumerate() {
        let admitted = if name == "key" && position == 0 {
            is_string_literal(argument)
        } else if name == "id" {
            is_string_literal(argument) || is_variable_reference(argument)
        } else {
            is_literal(argument) || is_variable_reference(argument)
        };
        if !admitted {
            return reject("may only be given a literal or a variable reference");
        }
    }
    Ok(())
}

/// The text after a function name at the start of `text`, when the name is followed by its
/// argument list rather than being the start of a longer name.
fn call_head<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(name)?.trim_start();
    rest.starts_with('(').then_some(rest)
}

/// The interior of the balanced parenthesized run at the start of `text`.
fn parenthesized(text: &str) -> Option<String> {
    let mut characters = text.chars();
    if characters.next()? != '(' {
        return None;
    }
    let mut depth = 1_usize;
    let mut quote: Option<char> = None;
    let mut inside = String::new();
    for character in characters {
        match quote {
            Some(open) => {
                inside.push(character);
                if character == open {
                    quote = None;
                }
            }
            None => match character {
                '\'' | '"' => {
                    quote = Some(character);
                    inside.push(character);
                }
                '(' | '[' => {
                    depth += 1;
                    inside.push(character);
                }
                ')' | ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(inside);
                    }
                    inside.push(character);
                }
                other => inside.push(other),
            },
        }
    }
    None
}

/// Whether `text` is a single string literal.
fn is_string_literal(text: &str) -> bool {
    let mut characters = text.chars();
    let Some(open @ ('\'' | '"')) = characters.next() else {
        return false;
    };
    let mut closed = false;
    for character in characters {
        if closed {
            // A doubled quotation mark escapes itself and the literal carries on.
            if character != open {
                return false;
            }
            closed = false;
            continue;
        }
        if character == open {
            closed = true;
        }
    }
    closed
}

/// Whether `text` is a string or numeric literal — XPath's `Literal` production.
fn is_literal(text: &str) -> bool {
    if is_string_literal(text) {
        return true;
    }
    let digits = text.strip_prefix(['+', '-']).unwrap_or(text);
    !digits.is_empty()
        && digits.chars().any(|character| character.is_ascii_digit())
        && digits.parse::<f64>().is_ok()
}

/// Whether `text` is a variable reference — `$` followed by a name and nothing else.
fn is_variable_reference(text: &str) -> bool {
    let Some(name) = text.strip_prefix('$') else {
        return false;
    };
    let name = name.trim_start();
    !name.is_empty()
        && name.chars().all(|character| {
            character.is_alphanumeric() || matches!(character, '-' | '_' | '.' | ':')
        })
}

/// Marks an alternative that is a *predicate pattern* — `.` followed by zero or more predicates,
/// which is the one pattern form that can match an item that is not a node — with the expression
/// that matches such an item against it.
///
/// A pattern that carries at least one predicate also takes the priority
/// [XSLT 3.0 §6.5](https://www.w3.org/TR/xslt-30/#conflict) gives it: half a point per predicate.
/// A bare `.` keeps the priority the node-test rules already gave it, `-0.5`. It matches every
/// item there is, so it is the least selective pattern the language has, and the specification's
/// own reference implementation of `fn:snapshot` — the `snapshot-equivalent.xsl` stylesheet the
/// W3C suite carries — turns on that: it dispatches nodes with `match="node()"`, `match="@*"` and
/// `match="namespace-node()"` and keeps `match="."` for "atomic values and function items", which
/// only holds when `.` does not outrank a kind test.
fn apply_predicate_pattern(alternative: &mut Alternative, source: &str, env: &StaticEnv) {
    let Some(predicates) = predicate_pattern_arity(source) else {
        return;
    };
    if predicates > 0 {
        alternative.priority = 0.5 * predicates as f64;
    }
    alternative.predicate_expression = env.compile_expression(source).ok();
}

/// The number of predicates in a pattern of the form `.[…][…]`, or `None` when the pattern is
/// not of that form.
///
/// [XSLT 3.0 §5.5.2](https://www.w3.org/TR/xslt-30/#pattern-syntax) spells the production as
/// `PredicatePattern ::= "." PredicateList` with `PredicateList ::= Predicate*`, so a bare `.` is
/// a predicate pattern of arity zero — one that every item matches — rather than no predicate
/// pattern at all.
fn predicate_pattern_arity(source: &str) -> Option<usize> {
    let stripped = strip_xpath_comments(source);
    let rest = stripped.trim().strip_prefix('.')?;
    let rest = rest.trim_start();
    if rest.is_empty() {
        return Some(0);
    }
    if !rest.starts_with('[') {
        return None;
    }
    let mut depth = 0_usize;
    let mut quote: Option<char> = None;
    let mut count = 0_usize;
    for character in rest.chars() {
        match quote {
            Some(open) => {
                if character == open {
                    quote = None;
                }
            }
            None => match character {
                '\'' | '"' => quote = Some(character),
                '[' | '(' => depth += 1,
                ']' | ')' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 && character == ']' {
                        count += 1;
                    }
                }
                _ => {}
            },
        }
    }
    (count > 0).then_some(count)
}

/// Removes the `(: … :)` comments XPath allows, so that a pattern's shape can be recognized.
fn strip_xpath_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let characters: Vec<char> = source.chars().collect();
    let mut index = 0;
    let mut depth = 0_usize;
    while index < characters.len() {
        if characters[index] == '(' && characters.get(index + 1) == Some(&':') {
            depth += 1;
            index += 2;
            continue;
        }
        if depth > 0 && characters[index] == ':' && characters.get(index + 1) == Some(&')') {
            depth -= 1;
            index += 2;
            continue;
        }
        if depth == 0 {
            out.push(characters[index]);
        }
        index += 1;
    }
    out
}

/// The `self::`-anchored form of a pattern whose first step takes the default `child` axis,
/// or `None` when the pattern does not start with such a step.
fn child_or_top_form(source: &str) -> Option<String> {
    let trimmed = source.trim();
    let first = trimmed.chars().next()?;
    if matches!(first, '/' | '(' | '$' | '@' | '.' | '\'' | '"') {
        return None;
    }
    let head: String = trimmed
        .chars()
        .take_while(|c| !matches!(c, '/' | '[' | '(' | ' ' | '\t' | '\n' | '\r'))
        .collect();
    // An explicit axis keeps its own meaning; only the implicit `child` axis widens.
    if head.contains("::") {
        return None;
    }
    Some(format!("self::{trimmed}"))
}

/// [XSLT 3.0 §6.5](https://www.w3.org/TR/xslt-30/#conflict) — the default priority.
fn default_priority(steps: &[PatternStep], rooted: bool, anchor: &Anchor) -> f64 {
    if rooted || matches!(anchor, Anchor::Selection(_)) || steps.len() != 1 {
        return 0.5;
    }
    let Some(step) = steps.first() else {
        return 0.5;
    };
    if !step.predicates.is_empty() {
        return 0.5;
    }
    match &step.test {
        PatternTest::Name(_) | PatternTest::Element(Some(_)) | PatternTest::Attribute(Some(_)) => {
            0.0
        }
        PatternTest::ProcessingInstruction(Some(_)) => 0.0,
        PatternTest::AnyLocal(_) | PatternTest::AnyNamespace(_) => -0.25,
        _ => -0.5,
    }
}

/// The alternatives a pattern is treated as, which each become a template rule of their own.
///
/// [XSLT 3.0 §6.5](https://www.w3.org/TR/xslt-30/#conflict) begins its default-priority rules
/// with the parentheses: "If the top-level pattern is a ParenthesizedExprP then the outer
/// parentheses are effectively stripped; these rules are applied recursively to the UnionExprP
/// contained in the ParenthesizedExprP." So `(doc|cod)` is two rules of priority 0, exactly as
/// `doc|cod` is, and not one rule of the 0.5 that everything else takes — which is what
/// `match-082a` and `-082c` are written to catch.
fn pattern_alternatives(source: &str) -> Vec<String> {
    let trimmed = source.trim();
    if let Some(inner) = strip_outer_parentheses(trimmed) {
        return pattern_alternatives(inner);
    }
    let parts = split_alternatives(trimmed);
    match parts.len() {
        0 | 1 => vec![trimmed.to_owned()],
        _ => parts
            .iter()
            .flat_map(|part| pattern_alternatives(part))
            .collect(),
    }
}

/// The text inside `source`'s outer parentheses, when a single pair encloses the whole of it.
fn strip_outer_parentheses(source: &str) -> Option<&str> {
    let inner = source.strip_prefix('(')?.strip_suffix(')')?;
    // Only a pair that encloses everything may go: in `(a|b)/c` and `(a)|(b)` the outer
    // characters are parentheses too, and neither is a parenthesized pattern.
    let mut depth = 0_usize;
    let mut quote: Option<char> = None;
    for character in inner.chars() {
        match quote {
            Some(open) => {
                if character == open {
                    quote = None;
                }
            }
            None => match character {
                '\'' | '"' => quote = Some(character),
                '(' | '[' => depth += 1,
                ')' | ']' => match depth {
                    0 => return None,
                    _ => depth -= 1,
                },
                _ => {}
            },
        }
    }
    Some(inner)
}

/// Splits a pattern into its alternatives, which may be separated by `|` or by the `union`
/// keyword.
fn split_alternatives(source: &str) -> Vec<String> {
    split_top_level(source, '|')
        .into_iter()
        .flat_map(|part| split_keyword(&part, "union"))
        .collect()
}

/// Splits on a keyword appearing at the top level, outside brackets and string literals.
fn split_keyword(source: &str, keyword: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0_usize;
    let mut quote: Option<char> = None;
    let characters: Vec<char> = source.chars().collect();
    let wanted: Vec<char> = keyword.chars().collect();
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if let Some(open) = quote {
            current.push(character);
            if character == open {
                quote = None;
            }
            index += 1;
            continue;
        }
        match character {
            '\'' | '"' => {
                quote = Some(character);
                current.push(character);
            }
            '(' | '[' => {
                depth += 1;
                current.push(character);
            }
            ')' | ']' => {
                depth = depth.saturating_sub(1);
                current.push(character);
            }
            _ => {
                let boundary_before = index == 0 || characters[index - 1].is_whitespace();
                let end = index + wanted.len();
                let boundary_after = end >= characters.len()
                    || characters.get(end).is_some_and(|c| c.is_whitespace());
                // A keyword operator following a lone `/` is not an operator at all: the
                // *leading-lone-slash* rule of [XPath 3.1 §3.3.1](https://www.w3.org/TR/xpath-31/#id-path-operator)
                // reads `/` as a complete expression only when what follows cannot begin a
                // relative path, and a name can. So `match="/ union /*"` is the path
                // `/union/*` — an element `*` whose parent is a `union` element at the top of
                // the tree — and not the union of `/` and `/*`, which is what `match-038` is
                // written to catch.
                let lone_slash = matches!(current.trim(), "/" | "//");
                if depth == 0
                    && !lone_slash
                    && boundary_before
                    && boundary_after
                    && characters[index..].starts_with(&wanted[..])
                {
                    parts.push(std::mem::take(&mut current));
                    index = end;
                    continue;
                }
                current.push(character);
            }
        }
        index += 1;
    }
    parts.push(current);
    parts
}

/// Splits `source` on `separator`, ignoring occurrences inside brackets, parentheses and
/// string literals.
fn split_top_level(source: &str, separator: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0_usize;
    let mut quote: Option<char> = None;
    for character in source.chars() {
        match quote {
            Some(open) => {
                current.push(character);
                if character == open {
                    quote = None;
                }
            }
            None => match character {
                '\'' | '"' => {
                    quote = Some(character);
                    current.push(character);
                }
                '(' | '[' => {
                    depth += 1;
                    current.push(character);
                }
                ')' | ']' => {
                    depth = depth.saturating_sub(1);
                    current.push(character);
                }
                other if other == separator && depth == 0 => {
                    parts.push(std::mem::take(&mut current));
                }
                other => current.push(other),
            },
        }
    }
    parts.push(current);
    parts
}

struct Scanner {
    characters: Vec<char>,
    position: usize,
}

impl Scanner {
    fn new(source: &str) -> Self {
        Self {
            characters: source.chars().collect(),
            position: 0,
        }
    }

    fn at_end(&self) -> bool {
        self.position >= self.characters.len()
    }

    fn peek(&self) -> Option<char> {
        self.characters.get(self.position).copied()
    }

    /// Skips whitespace, and the XPath comments that may stand wherever whitespace may.
    ///
    /// [XPath 3.1 §2.6](https://www.w3.org/TR/xpath-31/#comments) writes a comment as
    /// `(: … :)` and nests them, and a pattern is XPath text: `match-215` opens two of its rules
    /// with one. Without this the pattern reads as a parenthesized expression and falls back to
    /// being matched as an expression, which answers differently for a parentless node.
    fn skip_whitespace(&mut self) {
        loop {
            while self.peek().is_some_and(char::is_whitespace) {
                self.position += 1;
            }
            if !self.skip_comment() {
                return;
            }
        }
    }

    /// Consumes one nested `(: … :)` comment, reporting whether there was one.
    ///
    /// An unterminated comment is left where it stands rather than swallowing the rest of the
    /// pattern: the text is not a pattern at all, and the caller's own diagnosis names it.
    fn skip_comment(&mut self) -> bool {
        let start = self.position;
        if !self.eat("(:") {
            return false;
        }
        let mut depth = 1_usize;
        while depth > 0 {
            if self.at_end() {
                self.position = start;
                return false;
            }
            if self.eat("(:") {
                depth += 1;
            } else if self.eat(":)") {
                depth -= 1;
            } else {
                self.position += 1;
            }
        }
        true
    }

    fn eat(&mut self, text: &str) -> bool {
        let wanted: Vec<char> = text.chars().collect();
        if self.characters[self.position..].starts_with(&wanted[..]) {
            self.position += wanted.len();
            true
        } else {
            false
        }
    }

    /// Recognizes the expression a path pattern may be rooted at: an `id(…)` or `key(…)` call,
    /// or — as XSLT 3.0 allows — a variable reference.
    fn rooted_expression(&mut self) -> Option<String> {
        let start = self.position;
        for name in ["id", "key", "doc", "root", "element-with-id", "collection"] {
            if self.eat(name) {
                self.skip_whitespace();
                if self.peek() == Some('(') {
                    let arguments = self.balanced('(', ')')?;
                    return Some(format!("{name}({arguments})"));
                }
                self.position = start;
            }
        }
        if self.eat("$") {
            let name = self.name();
            if !name.is_empty() {
                return Some(format!("${name}"));
            }
            self.position = start;
        }
        None
    }

    /// Consumes a balanced `open`…`close` run and returns its interior.
    fn balanced(&mut self, open: char, close: char) -> Option<String> {
        if self.peek() != Some(open) {
            return None;
        }
        self.position += 1;
        let mut depth = 1_usize;
        let mut text = String::new();
        let mut quote: Option<char> = None;
        while let Some(character) = self.peek() {
            self.position += 1;
            match quote {
                Some(mark) => {
                    text.push(character);
                    if character == mark {
                        quote = None;
                    }
                }
                None => {
                    if character == '\'' || character == '"' {
                        quote = Some(character);
                        text.push(character);
                    } else if character == open {
                        depth += 1;
                        text.push(character);
                    } else if character == close {
                        depth -= 1;
                        if depth == 0 {
                            return Some(text);
                        }
                        text.push(character);
                    } else {
                        text.push(character);
                    }
                }
            }
        }
        None
    }

    fn name(&mut self) -> String {
        let start = self.position;
        // `EQName` also has the braced form `Q{uri}local` (XPath 3.1 §2.5.5), whose URI may hold
        // any character but a brace — colons and slashes among them, which is why it is consumed
        // whole here rather than left to the name characters below.
        if self.peek() == Some('Q') && self.characters.get(self.position + 1) == Some(&'{') {
            self.position += 2;
            while let Some(character) = self.peek() {
                self.position += 1;
                if character == '}' {
                    break;
                }
            }
        }
        while let Some(character) = self.peek() {
            if character.is_alphanumeric()
                || matches!(character, '_' | '-' | '.' | ':')
                || !character.is_ascii()
            {
                self.position += 1;
            } else {
                break;
            }
        }
        self.characters[start..self.position].iter().collect()
    }

    fn step(&mut self, env: &StaticEnv, connector: Connector) -> XsltResult<PatternStep> {
        if self.peek() == Some('.') && self.characters.get(self.position + 1) != Some(&'.') {
            self.position += 1;
            return Ok(PatternStep {
                axis: PatternAxis::Current,
                written_axis: false,
                test: PatternTest::AnyNode,
                predicates: self.predicates(env)?,
                connector,
            });
        }
        let mut written = true;
        let axis = if self.eat("@") {
            PatternAxis::Attribute
        } else if self.eat("self::") {
            PatternAxis::Current
        } else if self.eat("child::") {
            PatternAxis::Child
        } else if self.eat("attribute::") {
            PatternAxis::Attribute
        } else if self.eat("namespace::") {
            PatternAxis::Namespace
        } else {
            written = false;
            PatternAxis::Child
        };
        self.skip_whitespace();
        let test = self.node_test(env, axis)?;
        // [XPath 3.1 §3.3.2.1](https://www.w3.org/TR/xpath-31/#abbrev): the axis an abbreviated
        // step takes is `child`, "unless the NodeTest is an AttributeTest or
        // SchemaAttributeTest, in which case it is attribute", and likewise the namespace axis
        // for a NamespaceNodeTest. So an unprefixed `attribute()` step selects attributes and an
        // unprefixed `namespace-node()` step selects namespace nodes — on the child axis neither
        // could ever match anything. XSLT 3.0 §5.5.3 relies on the implicit form: its rewriting
        // of a pattern's first step speaks of a step that "uses the namespace axis (explicitly or
        // implicitly)". `key-087` writes `match="namespace-node()"` and expects it to match.
        let axis = match (written, &test) {
            (false, PatternTest::Attribute(_)) => PatternAxis::Attribute,
            (false, PatternTest::Namespace) => PatternAxis::Namespace,
            _ => axis,
        };
        let predicates = self.predicates(env)?;
        Ok(PatternStep {
            axis,
            written_axis: written,
            test,
            predicates,
            connector,
        })
    }

    /// Reads the `[…]` predicates that follow a step.
    fn predicates(&mut self, env: &StaticEnv) -> XsltResult<Vec<CompiledXPath>> {
        let mut predicates = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek() != Some('[') {
                break;
            }
            let Some(text) = self.balanced('[', ']') else {
                return Err(XsltError::stat1c(
                    "XTSE0340",
                    "a predicate in a pattern has no closing `]`",
                ));
            };
            predicates.push(env.compile_expression(&text)?);
        }
        Ok(predicates)
    }

    fn node_test(&mut self, env: &StaticEnv, axis: PatternAxis) -> XsltResult<PatternTest> {
        if self.eat("*") {
            self.skip_whitespace();
            if self.eat(":") {
                let local = self.name();
                return Ok(PatternTest::AnyNamespace(local));
            }
            return Ok(PatternTest::AnyName);
        }
        let name = self.name();
        if name.is_empty() {
            return Err(XsltError::stat1c(
                "XTSE0340",
                "a pattern step must have a node test",
            ));
        }
        let saved = self.position;
        self.skip_whitespace();
        if self.peek() == Some('(') {
            let inner = self.balanced('(', ')').unwrap_or_default();
            return kind_test(&name, inner.trim(), env);
        }
        self.position = saved;
        // `Q{uri}*` is the braced spelling of `prefix:*`, and carries the same default priority
        // of -0.25 — which is the whole of `match-261`.
        if let Some(rest) = name.strip_prefix("Q{")
            && let Some((namespace, local)) = rest.split_once('}')
        {
            if local.is_empty() && self.eat("*") {
                return Ok(PatternTest::AnyLocal(namespace.trim().to_owned()));
            }
            if local == "*" {
                return Ok(PatternTest::AnyLocal(namespace.trim().to_owned()));
            }
        }
        if let Some(prefix) = name.strip_suffix(":*") {
            let namespace = env.resolve_prefix(prefix)?;
            return Ok(PatternTest::AnyLocal(namespace));
        }
        if name.ends_with(':') {
            // `prefix:` followed by `*` with intervening whitespace.
            let prefix = name.trim_end_matches(':');
            self.skip_whitespace();
            if self.eat("*") {
                let namespace = env.resolve_prefix(prefix)?;
                return Ok(PatternTest::AnyLocal(namespace));
            }
        }
        let resolved = match axis {
            PatternAxis::Attribute => env.resolve_attribute_name(&name)?,
            _ => env.resolve_element_name(&name)?,
        };
        Ok(PatternTest::Name(resolved))
    }
}

fn kind_test(name: &str, inner: &str, env: &StaticEnv) -> XsltResult<PatternTest> {
    match name {
        "node" => Ok(PatternTest::AnyNode),
        "text" => Ok(PatternTest::Text),
        "comment" => Ok(PatternTest::Comment),
        "namespace-node" => Ok(PatternTest::Namespace),
        "processing-instruction" => {
            let target = inner.trim();
            if target.is_empty() {
                return Ok(PatternTest::ProcessingInstruction(None));
            }
            let unquoted = target
                .trim_matches(|c| c == '\'' || c == '"')
                .trim()
                .to_owned();
            Ok(PatternTest::ProcessingInstruction(Some(unquoted)))
        }
        "element" | "schema-element" => Ok(PatternTest::Element(optional_name(inner, env, false)?)),
        "attribute" | "schema-attribute" => {
            Ok(PatternTest::Attribute(optional_name(inner, env, true)?))
        }
        "document-node" => {
            let inner = inner.trim();
            if inner.is_empty() {
                return Ok(PatternTest::DocumentNode(None));
            }
            let stripped = inner
                .strip_prefix("element(")
                .and_then(|rest| rest.strip_suffix(')'))
                .unwrap_or(inner);
            Ok(PatternTest::DocumentNode(Some(optional_name(
                stripped, env, false,
            )?)))
        }
        other => Err(XsltError::stat1c(
            "XTSE0340",
            format!("`{other}()` is not a node test that a pattern may use"),
        )),
    }
}

fn optional_name(
    inner: &str,
    env: &StaticEnv,
    attribute: bool,
) -> XsltResult<Option<ExpandedName>> {
    let text = inner.split(',').next().unwrap_or("").trim();
    if text.is_empty() || text == "*" {
        return Ok(None);
    }
    let resolved = if attribute {
        env.resolve_attribute_name(text)?
    } else {
        env.resolve_element_name(text)?
    };
    Ok(Some(resolved))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternatives_split_on_the_bar_and_the_keyword() {
        assert_eq!(split_alternatives("a | b union c"), ["a ", " b ", " c"]);
        assert_eq!(split_alternatives("unionised"), ["unionised"]);
    }

    #[test]
    fn top_level_splitting_respects_nesting() {
        assert_eq!(split_top_level("a|b", '|'), ["a", "b"]);
        assert_eq!(split_top_level("a[b|c]|d", '|'), ["a[b|c]", "d"]);
        assert_eq!(split_top_level("'a|b'", '|'), ["'a|b'"]);
    }

    #[test]
    fn a_bare_dot_is_a_predicate_pattern_of_arity_zero() {
        assert_eq!(predicate_pattern_arity("."), Some(0));
        assert_eq!(predicate_pattern_arity("  .  "), Some(0));
        assert_eq!(predicate_pattern_arity(". (: comment :)"), Some(0));
        assert_eq!(predicate_pattern_arity(".[a]"), Some(1));
        assert_eq!(predicate_pattern_arity(".[a][b]"), Some(2));
        // A dot that only *starts* a path is not a predicate pattern.
        assert_eq!(predicate_pattern_arity(".."), None);
        assert_eq!(predicate_pattern_arity("./a"), None);
        assert_eq!(predicate_pattern_arity("a"), None);
    }

    #[test]
    fn balanced_scanning_handles_quotes() {
        let mut scanner = Scanner::new("('a)b', 2)rest");
        assert_eq!(scanner.balanced('(', ')').as_deref(), Some("'a)b', 2"));
        assert_eq!(scanner.name(), "rest");
    }
}
