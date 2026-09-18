//! Path expressions, steps and node tests.
//!
//! `RelativePathExpr` is *left*-associative, which matters: in `a/b/c` the context position seen
//! by `c` is its position within the merged, document-ordered result of `a/b`, not within the
//! result of `b` for one `a`. The tree this module builds preserves that grouping exactly.

use super::Parser;
use crate::ast::{AxisStep, Expr, NodeTest};
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::lexer::{Token, is_ncname};
use crate::types::{KindTest, NameTest};
use crate::version::XPathVersion;
use oxixml_infoset::{AtomicType, Axis, ExpandedName, NodeKind, XSD_NAMESPACE};

/// The kind tests XPath 1.0 already had. Everything else is 2.0 or later.
const XPATH10_KIND_TESTS: &[&str] = &["node", "text", "comment", "processing-instruction"];

/// Every name that introduces a kind test.
const KIND_TESTS: &[&str] = &[
    "attribute",
    "comment",
    "document-node",
    "element",
    "namespace-node",
    "node",
    "processing-instruction",
    "schema-attribute",
    "schema-element",
    "text",
];

/// Whether `name` introduces a kind test in `version`.
///
/// `namespace-node()` is the one late arrival: XPath 3.0 added it, so before that the name is an
/// ordinary (and unknown) function name, which is why QT3's `namespace-node(1)` under `XP20`
/// expects `XPST0017` rather than a syntax error.
fn is_kind_test(name: &str, version: XPathVersion) -> bool {
    if name == "namespace-node" {
        return version.at_least(XPathVersion::XPath30);
    }
    KIND_TESTS.contains(&name)
}

impl Parser<'_> {
    /// `PathExpr ::= ("/" RelativePathExpr?) | ("//" RelativePathExpr) | RelativePathExpr`
    pub(crate) fn parse_path(&mut self) -> XPathResult<Expr> {
        let mut left = match self.peek() {
            Some(Token::Slash) => {
                self.advance();
                if !self.starts_step() {
                    return Ok(Expr::Root);
                }
                let step = self.parse_step_expr()?;
                Expr::Path(Box::new(Expr::Root), Box::new(step))
            }
            Some(Token::DoubleSlash) => {
                self.advance();
                let step = self.parse_step_expr()?;
                let base = Expr::Path(Box::new(Expr::Root), Box::new(Expr::descendant_or_self()));
                Expr::Path(Box::new(base), Box::new(step))
            }
            _ => self.parse_step_expr()?,
        };
        loop {
            if self.eat(&Token::Slash) {
                let step = self.parse_step_expr()?;
                left = Expr::Path(Box::new(left), Box::new(step));
            } else if self.eat(&Token::DoubleSlash) {
                let step = self.parse_step_expr()?;
                let base = Expr::Path(Box::new(left), Box::new(Expr::descendant_or_self()));
                left = Expr::Path(Box::new(base), Box::new(step));
            } else {
                return Ok(left);
            }
        }
    }

    /// Whether the next token can begin a `StepExpr`.
    ///
    /// This is the disambiguation the specification calls for after a *leading lone slash*: `/`
    /// alone is the document root, and `/ * 5` is therefore read as the path `/*` followed by a
    /// stray `5` rather than as a division.
    fn starts_step(&self) -> bool {
        matches!(
            self.peek(),
            Some(
                Token::Dot
                    | Token::DotDot
                    | Token::At
                    | Token::Star
                    | Token::Name(_)
                    | Token::BracedName { .. }
                    | Token::BracedWildcard { .. }
                    | Token::LeftParen
                    | Token::LeftBracket
                    | Token::Dollar
                    | Token::Integer(_)
                    | Token::Decimal(_)
                    | Token::Double(_)
                    | Token::Str(_)
            )
        )
    }

    /// `StepExpr ::= PostfixExpr | AxisStep`
    fn parse_step_expr(&mut self) -> XPathResult<Expr> {
        if self.starts_axis_step() {
            return self.parse_axis_step();
        }
        self.parse_postfix()
    }

    fn starts_axis_step(&self) -> bool {
        match self.peek() {
            Some(Token::DotDot | Token::At) => true,
            Some(Token::Star) => true,
            Some(Token::BracedWildcard { .. }) => true,
            Some(Token::BracedName { .. }) => {
                !matches!(self.peek_at(1), Some(Token::LeftParen | Token::Hash))
            }
            Some(Token::Name(name)) => {
                if matches!(self.peek_at(1), Some(Token::ColonColon)) {
                    return true;
                }
                if matches!(self.peek_at(1), Some(Token::LeftParen)) {
                    return is_kind_test(name.as_str(), self.version);
                }
                !matches!(self.peek_at(1), Some(Token::Hash | Token::LeftBrace))
            }
            _ => false,
        }
    }

    /// `AxisStep ::= (ReverseStep | ForwardStep) PredicateList`
    fn parse_axis_step(&mut self) -> XPathResult<Expr> {
        let (axis, node_test) = self.parse_axis_and_test()?;
        let mut predicates = Vec::new();
        while self.eat(&Token::LeftBracket) {
            predicates.push(self.parse_expr()?);
            self.expect(&Token::RightBracket)?;
        }
        Ok(Expr::Step(Box::new(AxisStep {
            axis,
            node_test,
            predicates,
        })))
    }

    fn parse_axis_and_test(&mut self) -> XPathResult<(Axis, NodeTest)> {
        if self.eat(&Token::DotDot) {
            return Ok((Axis::Parent, NodeTest::Kind(KindTest::AnyKind)));
        }
        if self.eat(&Token::At) {
            let test = self.parse_node_test(Axis::Attribute)?;
            return Ok((Axis::Attribute, test));
        }
        if let Some(Token::Name(name)) = self.peek().cloned() {
            if matches!(self.peek_at(1), Some(Token::ColonColon)) {
                let offset = self.offset();
                let Some(axis) = Axis::from_name(&name) else {
                    return Err(XPathError::syntax(
                        offset,
                        format!("`{name}` is not an axis"),
                    ));
                };
                self.advance();
                self.advance();
                let test = self.parse_node_test(axis)?;
                return Ok((axis, test));
            }
        }
        // XPath 3.1 §3.3.2.1, `AbbrevForwardStep ::= "@"? NodeTest`: "if the NodeTest in an
        // AbbrevForwardStep is an AttributeTest or a SchemaAttributeTest, then the default axis
        // is attribute; if it is a NamespaceNodeTest, the default axis is namespace". Every
        // other abbreviated step defaults to the child axis. The namespace case is what makes
        // `/*/namespace-node()` select the element's namespace nodes (`Axes113`) rather than the
        // children it has none of — and, in XQuery, what makes the same step `XQST0134`.
        let axis = if self.starts_attribute_kind_test() {
            Axis::Attribute
        } else if self.starts_namespace_kind_test() {
            Axis::Namespace
        } else {
            Axis::Child
        };
        let test = self.parse_node_test(axis)?;
        Ok((axis, test))
    }

    /// Whether the next tokens are a `NamespaceNodeTest`.
    fn starts_namespace_kind_test(&self) -> bool {
        self.version.at_least(XPathVersion::XPath30)
            && matches!(self.peek(), Some(Token::Name(name)) if name == "namespace-node"
                && matches!(self.peek_at(1), Some(Token::LeftParen)))
    }

    /// Whether the next tokens are an `AttributeTest` or a `SchemaAttributeTest`.
    fn starts_attribute_kind_test(&self) -> bool {
        matches!(self.peek(), Some(Token::Name(name))
            if matches!(name.as_str(), "attribute" | "schema-attribute")
                && matches!(self.peek_at(1), Some(Token::LeftParen)))
    }

    /// `NodeTest ::= KindTest | NameTest`
    fn parse_node_test(&mut self, axis: Axis) -> XPathResult<NodeTest> {
        if let Some(Token::Name(name)) = self.peek().cloned() {
            if is_kind_test(name.as_str(), self.version)
                && matches!(self.peek_at(1), Some(Token::LeftParen))
            {
                return Ok(NodeTest::Kind(self.parse_kind_test()?));
            }
        }
        Ok(NodeTest::Name(self.parse_name_test(axis)?))
    }

    /// `NameTest ::= EQName | Wildcard`
    fn parse_name_test(&mut self, axis: Axis) -> XPathResult<NameTest> {
        let element_like = axis.principal_node_kind() == NodeKind::Element;
        match self.peek().cloned() {
            Some(Token::Star) => {
                // `Wildcard` is one terminal symbol (XPath 3.1 §A.2.4.1, `/* ws: explicit */`),
                // so `* :name` and `*(:comment:):name` are not wildcards at all: the `*` is a
                // complete name test and what follows is a syntax error.
                let joined = matches!(self.peek_at(1), Some(Token::Colon))
                    && matches!(self.peek_at(2), Some(Token::Name(_)))
                    && self.tokens_are_adjacent(3);
                self.advance();
                if joined {
                    self.require_version(XPathVersion::XPath20, "the wildcard `*:name`")?;
                    self.advance();
                    let Some(Token::Name(local)) = self.peek().cloned() else {
                        return Err(self.unexpected("a local name after `*:`"));
                    };
                    self.advance();
                    if local.contains(':') {
                        return Err(self.unexpected("an NCName after `*:`"));
                    }
                    return Ok(NameTest::WithLocal(local));
                }
                Ok(NameTest::Any)
            }
            Some(Token::BracedWildcard { uri }) => {
                self.require_version(XPathVersion::XPath30, "the wildcard `Q{uri}*`")?;
                self.advance();
                Ok(NameTest::InNamespace(uri))
            }
            Some(Token::BracedName { .. }) => {
                let name = self.parse_element_name()?;
                Ok(NameTest::Name(name))
            }
            Some(Token::Name(lexical)) => {
                if matches!(self.peek_at(1), Some(Token::Colon))
                    && matches!(self.peek_at(2), Some(Token::Star))
                    // One terminal symbol again: `pre :*` and `pre:(:comment:)*` are not the
                    // `NCName ":" "*"` wildcard.
                    && self.tokens_are_adjacent(3)
                    // Nor is `a:b:*`: the wildcard's first part is an `NCName`, so a lexical
                    // `QName` before the colon ends the longest match — `map{a:b:*}` is the key
                    // `a:b`, a colon, and the value `*` (`MapConstructor-030`).
                    && !lexical.contains(':')
                {
                    let offset = self.offset();
                    self.advance();
                    self.advance();
                    self.advance();
                    let Some(uri) = self.context.namespace(&lexical) else {
                        return Err(XPathError::new(
                            crate::error::XPathErrorCode::Xpst0081,
                            format!("the namespace prefix `{lexical}` is not bound"),
                        )
                        .at(offset));
                    };
                    return Ok(NameTest::InNamespace(uri.to_owned()));
                }
                let name = if element_like {
                    self.parse_element_name()?
                } else {
                    self.parse_unqualified_name()?
                };
                Ok(NameTest::Name(name))
            }
            _ => Err(self.unexpected("a node test")),
        }
    }

    /// `KindTest ::= DocumentTest | ElementTest | AttributeTest | SchemaElementTest
    ///             | SchemaAttributeTest | PITest | CommentTest | TextTest
    ///             | NamespaceNodeTest | AnyKindTest`
    pub(crate) fn parse_kind_test(&mut self) -> XPathResult<KindTest> {
        let offset = self.offset();
        let Some(Token::Name(name)) = self.peek().cloned() else {
            return Err(self.unexpected("a kind test"));
        };
        if !XPATH10_KIND_TESTS.contains(&name.as_str()) {
            self.require_version(XPathVersion::XPath20, &format!("the kind test `{name}()`"))?;
        }
        self.advance();
        self.expect(&Token::LeftParen)?;
        let test = match name.as_str() {
            "node" => KindTest::AnyKind,
            "text" => KindTest::Text,
            "comment" => KindTest::Comment,
            "namespace-node" => KindTest::NamespaceNode,
            "processing-instruction" => match self.peek().cloned() {
                Some(Token::Name(target)) => {
                    self.advance();
                    KindTest::ProcessingInstruction(Some(target))
                }
                // XPath 3.1 §2.5.5.4: in the string-literal form the literal "must have the form
                // of an NCName, after leading and trailing whitespace has been stripped, or a
                // type error [err:XPTY0004] is raised". So `processing-instruction("b ")` names
                // the target `b`, while `("123ncname")` and `("prefix:b")` are type errors —
                // *errors*, not tests that select nothing.
                Some(Token::Str(target)) => {
                    self.advance();
                    let trimmed = target.trim();
                    if !is_ncname(trimmed) {
                        return Err(XPathError::new(
                            XPathErrorCode::Xpty0004,
                            format!(
                                "`processing-instruction(\"{target}\")` needs an NCName as its \
                                 target"
                            ),
                        )
                        .at(offset));
                    }
                    KindTest::ProcessingInstruction(Some(trimmed.to_owned()))
                }
                _ => KindTest::ProcessingInstruction(None),
            },
            "document-node" => {
                if self.peek() == Some(&Token::RightParen) {
                    KindTest::Document(None)
                } else {
                    // Only `element()` or `schema-element()` may constrain a `document-node()`
                    // test. This is checked on the inner test's *name*, before parsing it, so
                    // that e.g. `schema-attribute()`'s own `XPST0008` cannot pre-empt the syntax
                    // error `document-node(schema-attribute(…))` is required to raise.
                    let inner_is_element = matches!(
                        self.peek(),
                        Some(Token::Name(inner)) if matches!(inner.as_str(), "element" | "schema-element")
                    );
                    if !inner_is_element {
                        return Err(XPathError::syntax(
                            offset,
                            "`document-node()` accepts only an element test",
                        ));
                    }
                    let inner = self.parse_kind_test()?;
                    KindTest::Document(Some(Box::new(inner)))
                }
            }
            "element" => self.parse_element_test()?,
            "attribute" => self.parse_attribute_test()?,
            "schema-element" => {
                let name = self.parse_element_name()?;
                // XPath 3.1 §2.5.5.3: `schema-element(N)` is `XPST0008` unless `N` is a global
                // element declaration in the in-scope schema definitions. This is a *basic*
                // processor, so no schema is ever imported and no such declaration exists.
                return Err(XPathError::new(
                    XPathErrorCode::Xpst0008,
                    format!(
                        "the element declaration `{name}` is not in scope; \
                         this processor imports no schema"
                    ),
                ));
            }
            "schema-attribute" => {
                let name = self.parse_unqualified_name()?;
                // The same rule for `schema-attribute(N)`; see above.
                return Err(XPathError::new(
                    XPathErrorCode::Xpst0008,
                    format!(
                        "the attribute declaration `{name}` is not in scope; \
                         this processor imports no schema"
                    ),
                ));
            }
            other => {
                return Err(XPathError::syntax(
                    offset,
                    format!("`{other}()` is not a kind test"),
                ));
            }
        };
        self.expect(&Token::RightParen)?;
        Ok(test)
    }

    fn parse_element_test(&mut self) -> XPathResult<KindTest> {
        if self.peek() == Some(&Token::RightParen) {
            return Ok(KindTest::Element {
                name: None,
                type_name: None,
                nillable: false,
            });
        }
        let name = self.parse_name_or_wildcard(true)?;
        let mut type_name = None;
        let mut nillable = false;
        if self.eat(&Token::Comma) {
            let named = self.parse_element_name()?;
            require_known_type(&named)?;
            type_name = Some(named);
            nillable = self.eat(&Token::Question);
        }
        Ok(KindTest::Element {
            name,
            type_name,
            nillable,
        })
    }

    fn parse_attribute_test(&mut self) -> XPathResult<KindTest> {
        if self.peek() == Some(&Token::RightParen) {
            return Ok(KindTest::Attribute {
                name: None,
                type_name: None,
            });
        }
        let name = self.parse_name_or_wildcard(false)?;
        let mut type_name = None;
        if self.eat(&Token::Comma) {
            let named = self.parse_element_name()?;
            require_known_type(&named)?;
            type_name = Some(named);
        }
        Ok(KindTest::Attribute { name, type_name })
    }

    fn parse_name_or_wildcard(&mut self, element_like: bool) -> XPathResult<Option<ExpandedName>> {
        if self.eat(&Token::Star) {
            return Ok(None);
        }
        let name = if element_like {
            self.parse_element_name()?
        } else {
            self.parse_unqualified_name()?
        };
        Ok(Some(name))
    }
}

/// Rejects a type name that is not in the in-scope schema definitions with `XPST0008`.
///
/// XPath 3.1 §2.5.5.3 requires an `ElementTest` or `AttributeTest` naming a type to raise
/// `XPST0008` when that type is not in scope. Without the Schema Import Feature the in-scope
/// definitions are exactly the built-in types of XML Schema, so anything else — a user type, a
/// misspelled `xs:` name, a name in another namespace — is out of scope.
fn require_known_type(name: &ExpandedName) -> XPathResult<()> {
    let known = name.namespace() == Some(XSD_NAMESPACE)
        && (AtomicType::from_local_name(name.local_name()).is_some()
            || matches!(
                name.local_name(),
                "untyped" | "anyType" | "anySimpleType" | "anyAtomicType"
            ));
    if known {
        return Ok(());
    }
    Err(XPathError::new(
        XPathErrorCode::Xpst0008,
        format!("the type `{name}` is not in scope; this processor imports no schema"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticContext;
    use crate::parser::parse;

    fn parse31(source: &str) -> XPathResult<Expr> {
        parse(source, XPathVersion::XPath31, &StaticContext::new())
    }

    fn step_of(expression: &Expr) -> &AxisStep {
        match expression {
            Expr::Step(step) => step,
            other => panic!("expected a step, found {other:?}"),
        }
    }

    #[test]
    fn every_axis_parses() -> XPathResult<()> {
        for axis in Axis::ALL {
            let source = format!("{}::node()", axis.name());
            let expression = parse31(&source)?;
            assert_eq!(step_of(&expression).axis, *axis);
        }
        Ok(())
    }

    #[test]
    fn abbreviated_steps() -> XPathResult<()> {
        assert_eq!(step_of(&parse31("..")?).axis, Axis::Parent);
        assert_eq!(step_of(&parse31("@x")?).axis, Axis::Attribute);
        assert_eq!(step_of(&parse31("x")?).axis, Axis::Child);
        assert!(matches!(parse31(".")?, Expr::ContextItem));
        assert!(matches!(parse31("/")?, Expr::Root));
        Ok(())
    }

    #[test]
    fn wildcards() -> XPathResult<()> {
        let context = StaticContext::new().with_namespace("p", "urn:p");
        let parse = |source: &str| parse(source, XPathVersion::XPath31, &context);
        assert_eq!(
            step_of(&parse("*")?).node_test,
            NodeTest::Name(NameTest::Any)
        );
        assert_eq!(
            step_of(&parse("p:*")?).node_test,
            NodeTest::Name(NameTest::InNamespace("urn:p".to_owned()))
        );
        assert_eq!(
            step_of(&parse("*:a")?).node_test,
            NodeTest::Name(NameTest::WithLocal("a".to_owned()))
        );
        assert_eq!(
            step_of(&parse("Q{urn:p}*")?).node_test,
            NodeTest::Name(NameTest::InNamespace("urn:p".to_owned()))
        );
        assert_eq!(
            step_of(&parse("Q{urn:p}a")?).node_test,
            NodeTest::Name(NameTest::Name(ExpandedName::new(Some("urn:p"), "a")))
        );
        assert!(parse("q:*").is_err());
        Ok(())
    }

    #[test]
    fn a_wildcard_admits_no_whitespace_or_comment_inside_it() {
        let context = StaticContext::new().with_namespace("p", "urn:p");
        let parse = |source: &str| parse(source, XPathVersion::XPath31, &context);
        // XPath 3.1 §A.2.4.1: `Wildcard` is one terminal symbol, and "comments are allowed
        // between terminals" only — never inside one.
        for source in [
            "* :a", "*: a", "*:(:c:)a", "*(:c:):a", "p :*", "p: *", "p:(:c:)*", "p(:c:):*",
        ] {
            let outcome = parse(source);
            assert!(
                outcome.is_err(),
                "`{source}` is not a wildcard, so it must not parse: {outcome:?}"
            );
        }
        assert!(parse("*:a").is_ok());
        assert!(parse("p:*").is_ok());
    }

    #[test]
    fn the_default_element_namespace_applies_only_to_element_names() -> XPathResult<()> {
        let context = StaticContext::new().with_default_element_namespace("urn:d");
        let parse = |source: &str| parse(source, XPathVersion::XPath31, &context);
        assert_eq!(
            step_of(&parse("a")?).node_test,
            NodeTest::Name(NameTest::Name(ExpandedName::new(Some("urn:d"), "a")))
        );
        assert_eq!(
            step_of(&parse("@a")?).node_test,
            NodeTest::Name(NameTest::Name(ExpandedName::new(None::<String>, "a")))
        );
        Ok(())
    }

    #[test]
    fn kind_tests() -> XPathResult<()> {
        assert_eq!(
            step_of(&parse31("node()")?).node_test,
            NodeTest::Kind(KindTest::AnyKind)
        );
        assert_eq!(
            step_of(&parse31("text()")?).node_test,
            NodeTest::Kind(KindTest::Text)
        );
        assert_eq!(
            step_of(&parse31("processing-instruction('go')")?).node_test,
            NodeTest::Kind(KindTest::ProcessingInstruction(Some("go".to_owned())))
        );
        assert_eq!(
            step_of(&parse31("processing-instruction(go)")?).node_test,
            NodeTest::Kind(KindTest::ProcessingInstruction(Some("go".to_owned())))
        );
        assert_eq!(
            step_of(&parse31("element(a, xs:string)")?).node_test,
            NodeTest::Kind(KindTest::Element {
                name: Some(ExpandedName::new(None::<String>, "a")),
                type_name: Some(ExpandedName::new(
                    Some("http://www.w3.org/2001/XMLSchema"),
                    "string"
                )),
                nillable: false,
            })
        );
        assert_eq!(
            step_of(&parse31("attribute()")?).node_test,
            NodeTest::Kind(KindTest::Attribute {
                name: None,
                type_name: None
            })
        );
        assert!(matches!(
            step_of(&parse31("document-node(element(a))")?).node_test,
            NodeTest::Kind(KindTest::Document(Some(_)))
        ));
        assert!(parse31("document-node(text())").is_err());
        Ok(())
    }

    #[test]
    fn xpath_10_has_only_four_kind_tests() {
        let context = StaticContext::new();
        for source in ["node()", "text()", "comment()", "processing-instruction()"] {
            assert!(parse(source, XPathVersion::XPath10, &context).is_ok());
        }
        for source in [
            "element()",
            "attribute()",
            "document-node()",
            "namespace-node()",
        ] {
            assert!(parse(source, XPathVersion::XPath10, &context).is_err());
        }
    }

    /// XPath 3.1 §3.3.2.1: an `AbbrevForwardStep` whose `NodeTest` is an `AttributeTest` or a
    /// `SchemaAttributeTest` defaults to the **attribute** axis, not the child axis.
    #[test]
    fn attribute_tests_default_to_the_attribute_axis() -> XPathResult<()> {
        for source in [
            "attribute()",
            "attribute(a)",
            "attribute(*)",
            "attribute(*, xs:string)",
            "attribute(a, xs:untypedAtomic)",
        ] {
            assert_eq!(
                step_of(&parse31(source)?).axis,
                Axis::Attribute,
                "`{source}` must default to the attribute axis"
            );
        }
        // Inside a path, and after a `//`, the same rule applies to the last step.
        let nested = parse31("a/b/attribute(c)")?;
        let Expr::Path(_, right) = &nested else {
            panic!("expected a path");
        };
        assert_eq!(step_of(right).axis, Axis::Attribute);
        let abbreviated = parse31("//attribute()")?;
        let Expr::Path(_, right) = &abbreviated else {
            panic!("expected a path");
        };
        assert_eq!(step_of(right).axis, Axis::Attribute);
        // A predicate does not disturb the defaulting.
        let filtered = parse31("attribute(a)[. = 'x']")?;
        let step = step_of(&filtered);
        assert_eq!(step.axis, Axis::Attribute);
        assert_eq!(step.predicates.len(), 1);
        // Every other kind test still defaults to the child axis.
        for source in ["element()", "node()", "text()", "comment()"] {
            assert_eq!(
                step_of(&parse31(source)?).axis,
                Axis::Child,
                "`{source}` must keep the child axis"
            );
        }
        // An explicit axis always wins over the default.
        assert_eq!(step_of(&parse31("self::attribute()")?).axis, Axis::SelfAxis);
        Ok(())
    }

    /// `schema-attribute(N)` is `XPST0008` in a processor that imports no schema — but it must
    /// reach that check as an *attribute* step, never be rejected as a stray function call.
    #[test]
    fn schema_attribute_steps_report_xpst0008() {
        for source in [
            "schema-attribute(a)",
            "a/schema-attribute(b)",
            "//schema-attribute(b)[1]",
        ] {
            let context = StaticContext::new();
            let error = parse(source, XPathVersion::XPath31, &context)
                .expect_err("schema-attribute needs an imported schema");
            assert_eq!(
                error.code(),
                Some(XPathErrorCode::Xpst0008),
                "for `{source}`"
            );
        }
    }

    #[test]
    fn paths_are_left_associative() -> XPathResult<()> {
        let expression = parse31("a/b/c")?;
        let Expr::Path(left, _) = &expression else {
            panic!("expected a path");
        };
        assert!(matches!(**left, Expr::Path(_, _)));
        Ok(())
    }

    #[test]
    fn predicates_attach_to_their_step() -> XPathResult<()> {
        let expression = parse31("a[1][2]")?;
        assert_eq!(step_of(&expression).predicates.len(), 2);
        // A predicate on a parenthesized expression is a filter, not a step predicate.
        assert!(matches!(parse31("(a)[1]")?, Expr::Filter(_, _)));
        Ok(())
    }
}
