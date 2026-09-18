//! The parser of the [RELAX NG compact syntax][rnc], §1.
//!
//! The grammar of the compact-syntax specification is annotated with the XML
//! fragment each production builds, and this parser follows those annotations
//! literally: every production returns the [`Element`](crate::tree::Element) that
//! the specification says it should, so the compact syntax and the XML syntax
//! meet at the data model and share the whole downstream pipeline.
//!
//! Annotations — the `[ … ]` groups, the `>>` following annotations and the `##`
//! documentation lines — translate to foreign attributes and elements, which
//! simplification rule 4.1 removes. They are therefore parsed for well-formedness
//! and then discarded.
//!
//! [rnc]: https://relaxng.org/compact-20021121.html

use std::rc::Rc;

use crate::datatype::XSD_LIBRARY;
use crate::error::{Position, SchemaError, syntax};
use crate::tree::{Child, Context, Element, NamespaceMap, RNG_NS};
use crate::uri::{parse_uri_reference, xlink_escape};

use super::lex::{Token, tokenize};

/// The XML namespace, which the `xml` prefix is bound to from the start.
const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";
/// The namespace of namespace-declaration attributes, per the XML Infoset.
const XMLNS_INFOSET: &str = "http://www.w3.org/2000/xmlns";

/// A namespace URI, or the special value `inherit`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum NsValue {
    Inherit,
    Uri(String),
}

/// The environment of §1: prefix bindings, the default namespace, the datatype
/// prefix bindings and the base URI.
#[derive(Debug, Clone)]
struct Env {
    prefixes: Vec<(String, NsValue)>,
    default_namespace: NsValue,
    datatypes: Vec<(String, String)>,
}

impl Env {
    fn initial() -> Self {
        Self {
            prefixes: vec![("xml".to_owned(), NsValue::Uri(XML_NAMESPACE.to_owned()))],
            default_namespace: NsValue::Inherit,
            datatypes: vec![("xsd".to_owned(), XSD_LIBRARY.to_owned())],
        }
    }

    fn lookup_prefix(&self, prefix: &str) -> Option<&NsValue> {
        self.prefixes
            .iter()
            .find(|(candidate, _)| candidate == prefix)
            .map(|(_, value)| value)
    }

    fn lookup_datatype_prefix(&self, prefix: &str) -> Option<&str> {
        self.datatypes
            .iter()
            .find(|(candidate, _)| candidate == prefix)
            .map(|(_, uri)| uri.as_str())
    }

    /// The namespace map of the context of every element the translation builds.
    fn namespace_map(&self) -> NamespaceMap {
        let mut map = NamespaceMap::default();
        for (prefix, value) in &self.prefixes {
            if let NsValue::Uri(uri) = value
                && !uri.is_empty()
            {
                map.bind(prefix, uri);
            }
        }
        map.set_default(RNG_NS);
        map
    }
}

/// The greatest depth of nested patterns, `div`/`include` grammar content, parenthesised name
/// classes or nested annotation elements this parser will follow before reporting a
/// [`SchemaError`] instead of exhausting the native call stack.
///
/// The compact syntax's grammar has no bound of its own on any of these — parentheses,
/// `element`/`attribute`/`mixed`/`list`/`grammar` bodies and datatype `except` clauses all
/// recurse back into a pattern; `div` and `include` bodies recurse back into grammar content;
/// name classes recurse through their own parentheses; and `[ … ]` annotations can nest inside
/// one another. Hostile input exploiting any of these (`((((…))))` tens of thousands of
/// parentheses deep, or the equivalent with `div { div { … } } }`) would otherwise crash the
/// process rather than fail to parse. No legitimate schema comes remotely close to this depth.
///
/// This mirrors the size of the workspace's other expression-grammar recursive-descent parsers
/// rather than the larger ceiling used for pure data-structure nesting (element trees, DOM
/// trees and the like): each level of *this* grammar's mutual recursion passes through several
/// stack frames carrying owned `String`/`Element` data, which costs measurably more native
/// stack per level than a plain tree walk. A cap sized for the cheaper case would not
/// consistently prevent a stack overflow on a reduced-size stack (a worker thread, for
/// instance, rather than a generously sized main thread).
const MAX_PARSE_NESTING_DEPTH: usize = 64;

/// Parses a compact-syntax schema into the RELAX NG data model.
pub(super) fn parse(source: &str, base: Option<&str>) -> Result<Element, SchemaError> {
    let tokens = tokenize(source)?;
    let mut parser = Parser {
        tokens,
        index: 0,
        env: Env::initial(),
        context: Context::new(base.map(Rc::from), Rc::new(NamespaceMap::initial())),
        base: base.map(Rc::from),
        depth: 0,
    };
    parser.top_level()
}

struct Parser {
    tokens: Vec<(Token, Position)>,
    index: usize,
    env: Env,
    context: Context,
    base: Option<Rc<str>>,
    /// How many of the mutually-recursive production families below are currently nested
    /// inside one another, checked against [`MAX_PARSE_NESTING_DEPTH`] by
    /// [`guard_depth`](Self::guard_depth).
    depth: usize,
}

impl Parser {
    // -----------------------------------------------------------------------
    // Token handling
    // -----------------------------------------------------------------------

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.index)
            .map_or(&Token::Eof, |(token, _)| token)
    }

    fn peek_at(&self, offset: usize) -> &Token {
        self.tokens
            .get(self.index + offset)
            .map_or(&Token::Eof, |(token, _)| token)
    }

    fn position(&self) -> Position {
        self.tokens
            .get(self.index)
            .map_or(Position::UNKNOWN, |(_, position)| *position)
    }

    fn bump(&mut self) -> Token {
        let token = self.peek().clone();
        if self.index < self.tokens.len() {
            self.index += 1;
        }
        token
    }

    fn expect(&mut self, expected: &Token) -> Result<(), SchemaError> {
        if self.peek() == expected {
            let _ = self.bump();
            return Ok(());
        }
        Err(syntax(format!(
            "expected {}, found {}",
            expected.describe(),
            self.peek().describe()
        ))
        .at(self.position()))
    }

    fn expect_keyword(&mut self, keyword: &str) -> Result<(), SchemaError> {
        self.expect(&Token::Keyword(keyword.to_owned()))
    }

    fn is_keyword(&self, keyword: &str) -> bool {
        matches!(self.peek(), Token::Keyword(name) if name == keyword)
    }

    fn error<T>(&self, message: impl Into<String>) -> Result<T, SchemaError> {
        Err(syntax(message).at(self.position()))
    }

    fn element(&self, name: &str) -> Element {
        Element::new(name, self.context.clone(), self.position())
    }

    /// Runs `body` — a call into one of this parser's four recursively-nestable productions
    /// ([`primary`](Self::primary), [`component`](Self::component),
    /// [`name_class_item`](Self::name_class_item) or
    /// [`annotation_element`](Self::annotation_element)) — with the shared nesting-depth
    /// counter raised by one for its duration, restoring it again once `body` returns
    /// (successfully or not) so that unrelated constructs later in the document are never
    /// charged for depth a sibling subtree already unwound.
    ///
    /// Every mutually-recursive production family in this grammar funnels its recursive step
    /// through one of those four call sites, so bounding depth there — rather than at every
    /// intermediate production — bounds every path a hostile document could use to exhaust the
    /// native call stack.
    fn guard_depth<T>(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<T, SchemaError>,
    ) -> Result<T, SchemaError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_NESTING_DEPTH {
            self.depth -= 1;
            return self.error(format!(
                "the schema nests patterns, grammar content, name classes or annotations more \
                 than {MAX_PARSE_NESTING_DEPTH} levels deep"
            ));
        }
        let result = body(self);
        self.depth -= 1;
        result
    }

    // -----------------------------------------------------------------------
    // Annotations, which are parsed and discarded
    // -----------------------------------------------------------------------

    /// `annotations ::= documentations | documentations "[" annotationAttributes
    /// annotationElements "]"`.
    ///
    /// Documentation lines are dropped by the lexer. The `[ … ]` group is parsed
    /// so that its constraints are checked, and then dropped: it translates to
    /// foreign attributes and elements, which simplification rule 4.1 removes.
    fn lead_annotations(&mut self) -> Result<(), SchemaError> {
        if *self.peek() != Token::LBracket {
            return Ok(());
        }
        let _ = self.bump();
        let mut seen: Vec<(String, String)> = Vec::new();
        while let Token::CName(prefix, local) = self.peek().clone() {
            if *self.peek_at(1) != Token::Equal {
                break;
            }
            let _ = self.bump();
            let _ = self.bump();
            let _ = self.literal()?;
            let namespace = self.foreign_attribute_namespace(&prefix)?;
            self.note_attribute(&mut seen, namespace, local)?;
        }
        while self.at_annotation_element_name() {
            self.annotation_element(false)?;
        }
        self.expect(&Token::RBracket)
    }

    /// `followAnnotations ::= (">>" annotationElement)*`, returning how many
    /// annotation elements were consumed.
    fn follow_annotations(&mut self) -> Result<usize, SchemaError> {
        let mut count = 0;
        while *self.peek() == Token::Follow {
            let _ = self.bump();
            self.annotation_element(false)?;
            count += 1;
        }
        Ok(count)
    }

    fn at_annotation_element_name(&self) -> bool {
        matches!(
            self.peek(),
            Token::Ident(_) | Token::Keyword(_) | Token::CName(_, _)
        )
    }

    /// `annotationElement ::= foreignElementName annotationAttributesContent`,
    /// and the `nestedAnnotationElement` production, which differs only in that
    /// its name may be in the RELAX NG namespace.
    /// Recurses into [`annotation_attributes_content`](Self::annotation_attributes_content),
    /// which calls back into this function for a nested annotation element, so this is the
    /// annotation family's guarded call site — see [`guard_depth`](Self::guard_depth).
    fn annotation_element(&mut self, nested: bool) -> Result<(), SchemaError> {
        self.guard_depth(|parser| parser.annotation_element_at_depth(nested))
    }

    fn annotation_element_at_depth(&mut self, nested: bool) -> Result<(), SchemaError> {
        match self.bump() {
            Token::Ident(_) | Token::Keyword(_) => {}
            Token::CName(prefix, _) => {
                let namespace = self.prefixed_name_namespace(&prefix)?;
                if !nested && namespace == RNG_NS {
                    return self
                        .error("an annotation element must not be in the RELAX NG namespace");
                }
            }
            other => {
                return self.error(format!(
                    "expected an annotation element name, found {}",
                    other.describe()
                ));
            }
        }
        self.annotation_attributes_content()
    }

    /// `annotationAttributesContent ::= "[" nestedAnnotationAttributes
    /// annotationContent "]"`.
    fn annotation_attributes_content(&mut self) -> Result<(), SchemaError> {
        self.expect(&Token::LBracket)?;
        let mut seen: Vec<(String, String)> = Vec::new();
        loop {
            let (namespace, local) = match self.peek().clone() {
                Token::CName(prefix, local) if *self.peek_at(1) == Token::Equal => {
                    let namespace = self.prefixed_name_namespace(&prefix)?;
                    if namespace == XMLNS_INFOSET {
                        return self
                            .error("an annotation attribute must not be a namespace declaration");
                    }
                    (namespace, local)
                }
                Token::Ident(local) | Token::Keyword(local) if *self.peek_at(1) == Token::Equal => {
                    if local == "xmlns" {
                        return self
                            .error("an annotation attribute must not be a namespace declaration");
                    }
                    (String::new(), local)
                }
                _ => break,
            };
            let _ = self.bump();
            let _ = self.bump();
            let _ = self.literal()?;
            self.note_attribute(&mut seen, namespace, local)?;
        }
        // `annotationContent ::= (nestedAnnotationElement | literal)*`
        loop {
            if let Token::Literal(_) = self.peek() {
                let _ = self.bump();
                continue;
            }
            if self.at_annotation_element_name() && *self.peek_at(1) == Token::LBracket {
                self.annotation_element(true)?;
                continue;
            }
            break;
        }
        self.expect(&Token::RBracket)
    }

    /// Resolves the prefix of a `foreignAttributeName` and checks the three
    /// constraints §1 attaches to it.
    fn foreign_attribute_namespace(&mut self, prefix: &str) -> Result<String, SchemaError> {
        let namespace = self.prefixed_name_namespace(prefix)?;
        if namespace.is_empty() {
            return self.error("an annotation attribute must have a namespace");
        }
        if namespace == XMLNS_INFOSET {
            return self.error("an annotation attribute must not be a namespace declaration");
        }
        if namespace == RNG_NS {
            return self.error("an annotation attribute must not be in the RELAX NG namespace");
        }
        Ok(namespace)
    }

    /// Resolves a `prefixedName`, applying the "annotation inherit" constraint.
    fn prefixed_name_namespace(&mut self, prefix: &str) -> Result<String, SchemaError> {
        match self.env.lookup_prefix(prefix) {
            Some(NsValue::Uri(uri)) => Ok(uri.clone()),
            Some(NsValue::Inherit) => {
                self.error("an annotation name must not use a prefix bound to `inherit`")
            }
            None => self.error(format!("the namespace prefix `{prefix}` is not declared")),
        }
    }

    fn note_attribute(
        &self,
        seen: &mut Vec<(String, String)>,
        namespace: String,
        local: String,
    ) -> Result<(), SchemaError> {
        if seen.iter().any(|(ns, ln)| *ns == namespace && *ln == local) {
            return self.error(format!(
                "the annotation attribute `{local}` is given more than once"
            ));
        }
        seen.push((namespace, local));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // topLevel
    // -----------------------------------------------------------------------

    fn top_level(&mut self) -> Result<Element, SchemaError> {
        self.preamble()?;
        self.context = Context::new(self.base.clone(), Rc::new(self.env.namespace_map()));
        self.lead_annotations()?;

        if self.starts_grammar() {
            let mut grammar = self.element("grammar");
            let members = self.grammar_content(&Token::Eof)?;
            grammar.set_child_elements(members);
            self.expect(&Token::Eof)?;
            return Ok(grammar);
        }
        let (pattern, trailing) = self.pattern_counted()?;
        if trailing > 0 {
            return self.error(
                "a top-level pattern must translate to a single element, but following \
                 annotations added more",
            );
        }
        self.expect(&Token::Eof)?;
        Ok(pattern)
    }

    fn starts_grammar(&self) -> bool {
        match self.peek() {
            Token::Eof => true,
            Token::Keyword(name) if matches!(name.as_str(), "start" | "div" | "include") => true,
            Token::Ident(_) => matches!(
                self.peek_at(1),
                Token::Equal | Token::PipeEqual | Token::AmpEqual | Token::LBracket
            ),
            _ => false,
        }
    }

    // -----------------------------------------------------------------------
    // preamble
    // -----------------------------------------------------------------------

    fn preamble(&mut self) -> Result<(), SchemaError> {
        loop {
            if self.is_keyword("namespace") {
                let _ = self.bump();
                let prefix = self.namespace_prefix()?;
                self.expect(&Token::Equal)?;
                let value = self.namespace_uri_literal()?;
                self.bind_prefix(prefix, value)?;
            } else if self.is_keyword("default") {
                let _ = self.bump();
                self.expect_keyword("namespace")?;
                if *self.peek() == Token::Equal {
                    let _ = self.bump();
                    let value = self.namespace_uri_literal()?;
                    self.check_xml_namespace(None, &value)?;
                    self.bind_default(value)?;
                } else {
                    let prefix = self.namespace_prefix()?;
                    self.expect(&Token::Equal)?;
                    let value = self.namespace_uri_literal()?;
                    self.bind_prefix(prefix, value.clone())?;
                    self.bind_default(value)?;
                }
            } else if self.is_keyword("datatypes") {
                let _ = self.bump();
                let prefix = self.identifier_or_keyword()?;
                self.expect(&Token::Equal)?;
                let uri = self.literal()?;
                self.bind_datatype_prefix(prefix, uri)?;
            } else {
                return Ok(());
            }
        }
    }

    fn namespace_prefix(&mut self) -> Result<String, SchemaError> {
        let prefix = self.identifier_or_keyword()?;
        if prefix == "xmlns" {
            return self.error("`xmlns` must not be used as a namespace prefix");
        }
        Ok(prefix)
    }

    fn namespace_uri_literal(&mut self) -> Result<NsValue, SchemaError> {
        if self.is_keyword("inherit") {
            let _ = self.bump();
            return Ok(NsValue::Inherit);
        }
        Ok(NsValue::Uri(self.literal()?))
    }

    fn check_xml_namespace(
        &self,
        prefix: Option<&str>,
        value: &NsValue,
    ) -> Result<(), SchemaError> {
        if let NsValue::Uri(uri) = value
            && uri == XML_NAMESPACE
            && prefix != Some("xml")
        {
            return self.error("the XML namespace may only be bound to the prefix `xml`");
        }
        if prefix == Some("xml") && *value != NsValue::Uri(XML_NAMESPACE.to_owned()) {
            return self.error("the prefix `xml` may only be bound to the XML namespace");
        }
        Ok(())
    }

    fn bind_prefix(&mut self, prefix: String, value: NsValue) -> Result<(), SchemaError> {
        self.check_xml_namespace(Some(&prefix), &value)?;
        if prefix != "xml" && self.env.prefixes.iter().any(|(name, _)| *name == prefix) {
            return self.error(format!("the prefix `{prefix}` is declared more than once"));
        }
        self.env.prefixes.retain(|(name, _)| *name != prefix);
        self.env.prefixes.push((prefix, value));
        Ok(())
    }

    fn bind_default(&mut self, value: NsValue) -> Result<(), SchemaError> {
        if self.env.default_namespace != NsValue::Inherit {
            return self.error("the default namespace is declared more than once");
        }
        self.env.default_namespace = value;
        Ok(())
    }

    fn bind_datatype_prefix(&mut self, prefix: String, uri: String) -> Result<(), SchemaError> {
        if prefix == "xsd" && uri != XSD_LIBRARY {
            return self
                .error("the datatype prefix `xsd` may only be bound to the XML Schema library");
        }
        if uri.is_empty() {
            return self.error("a datatypes declaration must not use the empty URI");
        }
        match parse_uri_reference(&xlink_escape(&uri)) {
            Some(info) if info.absolute && !info.has_fragment => {}
            _ => {
                return self.error(format!(
                    "`{uri}` is not a legal datatype library URI: it must be absolute and have no \
                     fragment identifier"
                ));
            }
        }
        if prefix != "xsd" && self.env.datatypes.iter().any(|(name, _)| *name == prefix) {
            return self.error(format!(
                "the datatype prefix `{prefix}` is declared more than once"
            ));
        }
        self.env.datatypes.retain(|(name, _)| *name != prefix);
        self.env.datatypes.push((prefix, uri));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // grammar content
    // -----------------------------------------------------------------------

    fn grammar_content(&mut self, terminator: &Token) -> Result<Vec<Element>, SchemaError> {
        let mut members = Vec::new();
        loop {
            self.lead_annotations()?;
            if self.peek() == terminator || *self.peek() == Token::Eof {
                return Ok(members);
            }
            if matches!(self.peek(), Token::Ident(_) | Token::CName(_, _))
                && *self.peek_at(1) == Token::LBracket
            {
                // `member ::= annotationElementNotKeyword`
                self.annotation_element(false)?;
                continue;
            }
            members.push(self.component()?);
        }
    }

    /// `member ::= …`. Recurses into [`grammar_content`](Self::grammar_content) for `div` and
    /// `include` bodies, so this is one of the two call sites guarded against pathological
    /// nesting — see [`guard_depth`](Self::guard_depth).
    fn component(&mut self) -> Result<Element, SchemaError> {
        self.guard_depth(Self::component_at_depth)
    }

    fn component_at_depth(&mut self) -> Result<Element, SchemaError> {
        if self.is_keyword("start") {
            let mut start = self.element("start");
            let _ = self.bump();
            let combine = self.assign_op()?;
            let pattern = self.pattern()?;
            if let Some(combine) = combine {
                start.set_attribute("combine", combine);
            }
            start.children.push(Child::Element(pattern));
            return Ok(start);
        }
        if self.is_keyword("div") {
            let mut div = self.element("div");
            let _ = self.bump();
            self.expect(&Token::LBrace)?;
            let members = self.grammar_content(&Token::RBrace)?;
            self.expect(&Token::RBrace)?;
            div.set_child_elements(members);
            return Ok(div);
        }
        if self.is_keyword("include") {
            return self.include();
        }
        let Token::Ident(name) = self.peek().clone() else {
            return self.error(format!(
                "expected a grammar component, found {}",
                self.peek().describe()
            ));
        };
        let mut define = self.element("define");
        let _ = self.bump();
        let combine = self.assign_op()?;
        let pattern = self.pattern()?;
        define.set_attribute("name", name);
        if let Some(combine) = combine {
            define.set_attribute("combine", combine);
        }
        define.children.push(Child::Element(pattern));
        Ok(define)
    }

    fn assign_op(&mut self) -> Result<Option<&'static str>, SchemaError> {
        match self.peek() {
            Token::Equal => {
                let _ = self.bump();
                Ok(None)
            }
            Token::PipeEqual => {
                let _ = self.bump();
                Ok(Some("choice"))
            }
            Token::AmpEqual => {
                let _ = self.bump();
                Ok(Some("interleave"))
            }
            other => Err(syntax(format!(
                "expected `=`, `|=` or `&=`, found {}",
                other.describe()
            ))
            .at(self.position())),
        }
    }

    fn include(&mut self) -> Result<Element, SchemaError> {
        let mut include = self.element("include");
        self.expect_keyword("include")?;
        let href = self.any_uri_literal()?;
        include.set_attribute("href", href);
        self.optional_inherit(&mut include)?;
        if *self.peek() == Token::LBrace {
            let _ = self.bump();
            let members = self.grammar_content(&Token::RBrace)?;
            self.expect(&Token::RBrace)?;
            include.set_child_elements(members);
        }
        Ok(include)
    }

    /// `optInherit`: adds the `ns` attribute that the referenced schema inherits.
    fn optional_inherit(&mut self, element: &mut Element) -> Result<(), SchemaError> {
        let value = if self.is_keyword("inherit") {
            let _ = self.bump();
            self.expect(&Token::Equal)?;
            let prefix = self.identifier_or_keyword()?;
            match self.env.lookup_prefix(&prefix) {
                Some(value) => value.clone(),
                None => return self.error(format!("the prefix `{prefix}` is not declared")),
            }
        } else {
            self.env.default_namespace.clone()
        };
        if let NsValue::Uri(uri) = value {
            element.set_attribute("ns", uri);
        }
        Ok(())
    }

    fn any_uri_literal(&mut self) -> Result<String, SchemaError> {
        let value = self.literal()?;
        if parse_uri_reference(&xlink_escape(&value)).is_none() {
            return self.error(format!("`{value}` is not a URI reference"));
        }
        Ok(value)
    }

    // -----------------------------------------------------------------------
    // patterns
    // -----------------------------------------------------------------------

    fn pattern(&mut self) -> Result<Element, SchemaError> {
        self.pattern_counted().map(|(pattern, _)| pattern)
    }

    /// Parses a pattern, also reporting how many *following annotations* trail
    /// it: those translate to sibling elements, which the top-level "single
    /// element" constraint forbids.
    fn pattern_counted(&mut self) -> Result<(Element, usize), SchemaError> {
        let (first, trailing) = self.particle()?;
        let (name, separator) = match self.peek() {
            Token::Pipe => ("choice", Token::Pipe),
            Token::Comma => ("group", Token::Comma),
            Token::Amp => ("interleave", Token::Amp),
            _ => return Ok((first, trailing)),
        };
        let mut combined = self.element(name);
        combined.children.push(Child::Element(first));
        while *self.peek() == separator {
            let _ = self.bump();
            let (next, _) = self.particle()?;
            combined.children.push(Child::Element(next));
        }
        if matches!(self.peek(), Token::Pipe | Token::Comma | Token::Amp) {
            return self
                .error("`|`, `,` and `&` must not be mixed at the same level; use parentheses");
        }
        Ok((combined, 0))
    }

    fn particle(&mut self) -> Result<(Element, usize), SchemaError> {
        self.lead_annotations()?;
        let mut pattern = self.primary()?;
        match self.peek() {
            Token::Star => {
                let _ = self.bump();
                pattern = self.wrap("zeroOrMore", pattern);
            }
            Token::Plus => {
                let _ = self.bump();
                pattern = self.wrap("oneOrMore", pattern);
            }
            Token::Question => {
                let _ = self.bump();
                pattern = self.wrap("optional", pattern);
            }
            _ => {}
        }
        let trailing = self.follow_annotations()?;
        Ok((pattern, trailing))
    }

    fn wrap(&self, name: &str, inner: Element) -> Element {
        let mut wrapper = Element::like(&inner, name);
        wrapper.children.push(Child::Element(inner));
        wrapper
    }

    /// Recurses into [`pattern`](Self::pattern) (parens, `element`/`attribute`/`mixed`/`list`
    /// bodies), [`grammar_content`](Self::grammar_content) (`grammar` bodies) and itself
    /// (datatype `except` clauses), so this is the other call site guarded against
    /// pathological nesting — see [`guard_depth`](Self::guard_depth).
    fn primary(&mut self) -> Result<Element, SchemaError> {
        self.guard_depth(Self::primary_at_depth)
    }

    fn primary_at_depth(&mut self) -> Result<Element, SchemaError> {
        match self.peek().clone() {
            Token::Keyword(keyword) => match keyword.as_str() {
                "element" | "attribute" => {
                    let is_element = keyword == "element";
                    let mut pattern = self.element(&keyword);
                    let _ = self.bump();
                    let name_class = self.name_class(is_element)?;
                    self.expect(&Token::LBrace)?;
                    let body = self.pattern()?;
                    self.expect(&Token::RBrace)?;
                    pattern.children.push(Child::Element(name_class));
                    pattern.children.push(Child::Element(body));
                    Ok(pattern)
                }
                "mixed" | "list" => {
                    let mut pattern = self.element(&keyword);
                    let _ = self.bump();
                    self.expect(&Token::LBrace)?;
                    let body = self.pattern()?;
                    self.expect(&Token::RBrace)?;
                    pattern.children.push(Child::Element(body));
                    Ok(pattern)
                }
                "empty" | "notAllowed" | "text" => {
                    let pattern = self.element(&keyword);
                    let _ = self.bump();
                    Ok(pattern)
                }
                "grammar" => {
                    let mut pattern = self.element("grammar");
                    let _ = self.bump();
                    self.expect(&Token::LBrace)?;
                    let members = self.grammar_content(&Token::RBrace)?;
                    self.expect(&Token::RBrace)?;
                    pattern.set_child_elements(members);
                    Ok(pattern)
                }
                "parent" => {
                    let mut pattern = self.element("parentRef");
                    let _ = self.bump();
                    let name = self.identifier()?;
                    pattern.set_attribute("name", name);
                    Ok(pattern)
                }
                "external" => {
                    let mut pattern = self.element("externalRef");
                    let _ = self.bump();
                    let href = self.any_uri_literal()?;
                    pattern.set_attribute("href", href);
                    self.optional_inherit(&mut pattern)?;
                    Ok(pattern)
                }
                "string" | "token" => self.datatype(String::new(), keyword),
                other => self.error(format!("`{other}` cannot start a pattern")),
            },
            Token::CName(prefix, local) => {
                let Some(library) = self.env.lookup_datatype_prefix(&prefix) else {
                    return self.error(format!("the datatype prefix `{prefix}` is not declared"));
                };
                let library = library.to_owned();
                self.datatype(library, local)
            }
            Token::Literal(value) => {
                let mut pattern = self.element("value");
                let _ = self.bump();
                pattern.children.push(Child::Text(value));
                Ok(pattern)
            }
            Token::Ident(name) => {
                let mut pattern = self.element("ref");
                let _ = self.bump();
                pattern.set_attribute("name", name);
                Ok(pattern)
            }
            Token::LParen => {
                let _ = self.bump();
                let inner = self.pattern()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            other => self.error(format!("{} cannot start a pattern", other.describe())),
        }
    }

    /// `datatypeName optParams ["-" leadAnnotatedPrimary]` or
    /// `datatypeName datatypeValue`.
    fn datatype(&mut self, library: String, type_name: String) -> Result<Element, SchemaError> {
        let _ = self.bump();
        if let Token::Literal(value) = self.peek().clone() {
            let mut pattern = self.element("value");
            let _ = self.bump();
            pattern.set_attribute("datatypeLibrary", library);
            pattern.set_attribute("type", type_name);
            pattern.children.push(Child::Text(value));
            return Ok(pattern);
        }
        let mut pattern = self.element("data");
        pattern.set_attribute("datatypeLibrary", library);
        pattern.set_attribute("type", type_name);
        if *self.peek() == Token::LBrace {
            let _ = self.bump();
            while *self.peek() != Token::RBrace {
                self.lead_annotations()?;
                if *self.peek() == Token::RBrace {
                    break;
                }
                let name = self.identifier_or_keyword()?;
                self.expect(&Token::Equal)?;
                let value = self.literal()?;
                let mut param = self.element("param");
                param.set_attribute("name", name);
                param.children.push(Child::Text(value));
                pattern.children.push(Child::Element(param));
            }
            self.expect(&Token::RBrace)?;
        }
        if *self.peek() == Token::Minus {
            let _ = self.bump();
            self.lead_annotations()?;
            let inner = self.primary()?;
            let mut except = self.element("except");
            except.children.push(Child::Element(inner));
            pattern.children.push(Child::Element(except));
        }
        Ok(pattern)
    }

    // -----------------------------------------------------------------------
    // name classes
    // -----------------------------------------------------------------------

    fn name_class(&mut self, is_element: bool) -> Result<Element, SchemaError> {
        let first = self.name_class_item(is_element)?;
        if *self.peek() != Token::Pipe {
            return Ok(first);
        }
        let mut choice = self.element("choice");
        choice.children.push(Child::Element(first));
        while *self.peek() == Token::Pipe {
            let _ = self.bump();
            let next = self.name_class_item(is_element)?;
            choice.children.push(Child::Element(next));
        }
        Ok(choice)
    }

    /// Recurses into [`name_class`](Self::name_class) (parens) and
    /// [`except_name_class`](Self::except_name_class) (which itself calls back into this
    /// function), so this is the name-class family's guarded call site — see
    /// [`guard_depth`](Self::guard_depth).
    fn name_class_item(&mut self, is_element: bool) -> Result<Element, SchemaError> {
        self.guard_depth(|parser| parser.name_class_item_at_depth(is_element))
    }

    fn name_class_item_at_depth(&mut self, is_element: bool) -> Result<Element, SchemaError> {
        self.lead_annotations()?;
        let class = match self.peek().clone() {
            Token::Star => {
                let mut any = self.element("anyName");
                let _ = self.bump();
                if *self.peek() == Token::Minus {
                    let _ = self.bump();
                    let except = self.except_name_class(is_element)?;
                    any.children.push(Child::Element(except));
                }
                any
            }
            Token::NsName(prefix) => {
                let mut ns_name = self.element("nsName");
                let _ = self.bump();
                self.set_prefix_namespace(&mut ns_name, &prefix)?;
                if *self.peek() == Token::Minus {
                    let _ = self.bump();
                    let except = self.except_name_class(is_element)?;
                    ns_name.children.push(Child::Element(except));
                }
                ns_name
            }
            Token::CName(prefix, local) => {
                let mut name = self.element("name");
                let _ = self.bump();
                self.set_prefix_namespace(&mut name, &prefix)?;
                name.children.push(Child::Text(local));
                name
            }
            Token::Ident(local) | Token::Keyword(local) => {
                let mut name = self.element("name");
                let _ = self.bump();
                if is_element {
                    if let NsValue::Uri(uri) = self.env.default_namespace.clone() {
                        name.set_attribute("ns", uri);
                    }
                } else {
                    name.set_attribute("ns", "");
                }
                name.children.push(Child::Text(local));
                name
            }
            Token::LParen => {
                let _ = self.bump();
                let inner = self.name_class(is_element)?;
                self.expect(&Token::RParen)?;
                inner
            }
            other => {
                return self.error(format!("{} cannot start a name class", other.describe()));
            }
        };
        let _ = self.follow_annotations()?;
        Ok(class)
    }

    fn except_name_class(&mut self, is_element: bool) -> Result<Element, SchemaError> {
        let mut except = self.element("except");
        self.lead_annotations()?;
        let inner = self.name_class_item(is_element)?;
        except.children.push(Child::Element(inner));
        Ok(except)
    }

    fn set_prefix_namespace(
        &mut self,
        element: &mut Element,
        prefix: &str,
    ) -> Result<(), SchemaError> {
        let Some(value) = self.env.lookup_prefix(prefix).cloned() else {
            return self.error(format!("the namespace prefix `{prefix}` is not declared"));
        };
        match value {
            NsValue::Inherit => {}
            NsValue::Uri(uri) => {
                if uri == XMLNS_INFOSET {
                    return self.error("a name class must not use the `xmlns` namespace");
                }
                element.set_attribute("ns", uri);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // terminals
    // -----------------------------------------------------------------------

    fn identifier(&mut self) -> Result<String, SchemaError> {
        match self.peek().clone() {
            Token::Ident(name) => {
                let _ = self.bump();
                Ok(name)
            }
            other => self.error(format!(
                "expected an identifier, found {}",
                other.describe()
            )),
        }
    }

    fn identifier_or_keyword(&mut self) -> Result<String, SchemaError> {
        match self.peek().clone() {
            Token::Ident(name) | Token::Keyword(name) => {
                let _ = self.bump();
                Ok(name)
            }
            other => self.error(format!(
                "expected an identifier or keyword, found {}",
                other.describe()
            )),
        }
    }

    fn literal(&mut self) -> Result<String, SchemaError> {
        match self.peek().clone() {
            Token::Literal(value) => {
                let _ = self.bump();
                Ok(value)
            }
            other => self.error(format!("expected a literal, found {}", other.describe())),
        }
    }
}
