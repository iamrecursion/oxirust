//! An XPath 1.0 engine, sized for XML Signature.
//!
//! XML Signature needs XPath in three places — the
//! [XPath filtering transform](https://www.w3.org/TR/xmldsig-core1/#sec-XPath),
//! the [XPath Filter 2.0 transform](https://www.w3.org/TR/xmldsig-filter2/) and
//! the `#xpointer(…)` forms of a same-document reference — and
//! [§4.4.3.2](https://www.w3.org/TR/xmldsig-core1/#sec-ReferenceProcessingModel)
//! is explicit that a full XPath implementation is one way to get there but not
//! the only one: "such applications could use a sufficiently functional
//! replacement to a node-set and implement only those XPath expression
//! behaviors REQUIRED by this specification".
//!
//! This module takes the first route anyway and implements the whole of XPath
//! 1.0 — thirteen axes, every node test, the complete core function library,
//! the full expression grammar with its context-sensitive tokenizer — because
//! the alternative is a pattern matcher that quietly selects the wrong nodes
//! for an expression nobody anticipated, and in a signature that is a security
//! bug rather than a missing feature.
//!
//! On top of XPath 1.0 it adds `here()`, the one function XML Signature defines.
//!
//! # Deliberate limits
//!
//! * **Variables.** `$x` always fails. XML Signature says so: "the set of
//!   variable bindings used when evaluating the XPath expression is empty, and
//!   use of a variable reference in the XPath expression results in an error".
//! * **Extension functions.** A prefixed function name always fails; there is
//!   no extension mechanism to bind one to.
//! * **`number()` to string.** The conversion defers to the ECMAScript
//!   algorithm, which agrees with XPath for every magnitude in `[1e-6, 1e21)`
//!   and uses exponent notation outside it where XPath wants a full decimal
//!   expansion.

pub(crate) mod ast;
pub(crate) mod eval;
pub(crate) mod functions;
pub(crate) mod lexer;
pub(crate) mod parser;
pub(crate) mod value;

use crate::error::{DsigError, DsigResult};

pub(crate) use eval::{Context, Focus, IdResolver, NamespaceContext, whole_document_node_set};
pub(crate) use value::{NodeSet, Value, XNode};

/// A parsed, reusable XPath expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Expression {
    parsed: ast::Expr,
    source: String,
}

impl Expression {
    /// Parses an expression.
    ///
    /// # Errors
    ///
    /// [`DsigError::XPath`] with the parser's message.
    pub(crate) fn parse(source: &str) -> DsigResult<Self> {
        let parsed = parser::parse(source)
            .map_err(|message| DsigError::XPath(format!("{message} (in `{}`)", source.trim())))?;
        Ok(Self {
            parsed,
            source: source.to_owned(),
        })
    }

    /// Evaluates the expression with the given context node.
    ///
    /// # Errors
    ///
    /// [`DsigError::XPath`] with the evaluator's message.
    pub(crate) fn evaluate(&self, context: &Context<'_>, focus: &Focus) -> DsigResult<Value> {
        eval::evaluate(&self.parsed, context, focus)
            .map_err(|message| DsigError::XPath(format!("{message} (in `{}`)", self.source.trim())))
    }

    /// Evaluates the expression and requires a node-set result.
    ///
    /// # Errors
    ///
    /// [`DsigError::XPath`] when evaluation fails or the result is not a
    /// node-set.
    pub(crate) fn evaluate_to_node_set(
        &self,
        context: &Context<'_>,
        focus: &Focus,
    ) -> DsigResult<NodeSet> {
        match self.evaluate(context, focus)? {
            Value::NodeSet(set) => Ok(set),
            _ => Err(DsigError::XPath(format!(
                "`{}` does not evaluate to a node-set",
                self.source.trim()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::{Document, NodeId, build};

    struct Ids<'a>(&'a Document);

    impl IdResolver for Ids<'_> {
        fn resolve(&self, id: &str) -> Option<NodeId> {
            self.0
                .nodes_in_document_order()
                .into_iter()
                .find(|node| self.0.get_attribute(*node, "id") == Some(id))
        }
    }

    fn evaluate(document: &Document, expression: &str) -> DsigResult<Value> {
        let ids = Ids(document);
        let context = Context {
            document,
            namespaces: NamespaceContext::empty().bind("p", "urn:p"),
            here: None,
            ids: &ids,
        };
        Expression::parse(expression)?
            .evaluate(&context, &Focus::single(XNode::Tree(document.root())))
    }

    fn count(document: &Document, expression: &str) -> DsigResult<usize> {
        match evaluate(document, expression)? {
            Value::NodeSet(set) => Ok(set.len()),
            other => Err(DsigError::XPath(format!("expected a node-set: {other:?}"))),
        }
    }

    fn sample() -> Document {
        build::parse_str(
            r#"<?pi go?><!-- top --><root xmlns:p="urn:p" id="r">
                 <a id="a1" k="1">alpha</a>
                 <p:b id="b1">beta</p:b>
                 <a id="a2" k="2">gamma<!-- inner --></a>
               </root>"#,
        )
        .expect("well-formed test input")
    }

    #[test]
    fn axes_and_node_tests() -> DsigResult<()> {
        let document = sample();
        assert_eq!(count(&document, "/root/a")?, 2);
        assert_eq!(count(&document, "//a")?, 2);
        assert_eq!(count(&document, "//p:b")?, 1);
        assert_eq!(count(&document, "//*")?, 4);
        assert_eq!(count(&document, "//@*")?, 6);
        assert_eq!(count(&document, "//comment()")?, 2);
        assert_eq!(count(&document, "//processing-instruction()")?, 1);
        assert_eq!(count(&document, "//processing-instruction('pi')")?, 1);
        assert_eq!(count(&document, "//processing-instruction('other')")?, 0);
        assert_eq!(count(&document, "/root/a[1]/following-sibling::*")?, 2);
        assert_eq!(count(&document, "/root/a[2]/preceding-sibling::*")?, 2);
        assert_eq!(count(&document, "//a[2]/ancestor::*")?, 1);
        assert_eq!(count(&document, "//a[2]/ancestor-or-self::*")?, 2);
        Ok(())
    }

    #[test]
    fn namespace_axis_sees_the_implicit_xml_binding() -> DsigResult<()> {
        let document = sample();
        // `p` and `xml` are in scope on <root>.
        assert_eq!(count(&document, "/root/namespace::*")?, 2);
        assert_eq!(count(&document, "//namespace::*")?, 8);
        Ok(())
    }

    #[test]
    fn namespace_declarations_are_not_on_the_attribute_axis() -> DsigResult<()> {
        let document = sample();
        // <root> has `id` but not `xmlns:p`.
        assert_eq!(count(&document, "/root/@*")?, 1);
        Ok(())
    }

    #[test]
    fn predicates_and_positions() -> DsigResult<()> {
        let document = sample();
        assert_eq!(count(&document, "//a[@k='2']")?, 1);
        assert_eq!(count(&document, "//a[position()=1]")?, 1);
        assert_eq!(count(&document, "//a[last()]")?, 1);
        assert_eq!(count(&document, "//a[@k > 1]")?, 1);
        assert_eq!(count(&document, "//a[not(@k)]")?, 0);
        // A reverse axis numbers from the context node outwards.
        assert_eq!(
            match evaluate(&document, "name(//a[2]/ancestor-or-self::*[1])")? {
                Value::Str(text) => text,
                other => panic!("expected a string, got {other:?}"),
            },
            "a"
        );
        Ok(())
    }

    #[test]
    fn the_core_function_library_behaves() -> DsigResult<()> {
        let document = sample();
        let string = |expression: &str| -> DsigResult<String> {
            Ok(evaluate(&document, expression)?.to_xpath_string(&document))
        };
        assert_eq!(string("concat('a', 'b', 'c')")?, "abc");
        assert_eq!(string("substring('12345', 2, 3)")?, "234");
        assert_eq!(string("substring('12345', 1.5, 2.6)")?, "234");
        assert_eq!(string("substring('12345', 0, 3)")?, "12");
        assert_eq!(string("substring-before('1999/04/01','/')")?, "1999");
        assert_eq!(string("substring-after('1999/04/01','/')")?, "04/01");
        assert_eq!(string("normalize-space('  a  b  ')")?, "a b");
        assert_eq!(string("translate('bar','abc','ABC')")?, "BAr");
        assert_eq!(string("translate('--aaa--','abc-','ABC')")?, "AAA");
        assert_eq!(string("string-length('hello')")?, "5");
        assert_eq!(string("round(0.5)")?, "1");
        assert_eq!(string("round(-0.5)")?, "0");
        assert_eq!(string("floor(-1.5)")?, "-2");
        assert_eq!(string("ceiling(-1.5)")?, "-1");
        assert_eq!(string("string(1 div 0)")?, "Infinity");
        assert_eq!(string("count(//a)")?, "2");
        assert_eq!(string("sum(//a/@k)")?, "3");
        assert_eq!(string("local-name(//p:b)")?, "b");
        assert_eq!(string("namespace-uri(//p:b)")?, "urn:p");
        assert_eq!(string("name(//p:b)")?, "p:b");
        assert_eq!(string("string(//a)")?, "alpha");
        assert_eq!(string("boolean(//nothing)")?, "false");
        assert_eq!(string("string(id('a2'))")?, "gamma");
        Ok(())
    }

    #[test]
    fn variables_and_extension_functions_are_errors() {
        let document = sample();
        assert!(matches!(
            evaluate(&document, "$x"),
            Err(DsigError::XPath(_))
        ));
        assert!(matches!(
            evaluate(&document, "p:custom()"),
            Err(DsigError::XPath(_))
        ));
        assert!(matches!(
            evaluate(&document, "nope()"),
            Err(DsigError::XPath(_))
        ));
    }

    #[test]
    fn an_unbound_prefix_is_an_error() {
        let document = sample();
        assert!(matches!(
            evaluate(&document, "//q:thing"),
            Err(DsigError::XPath(_))
        ));
    }

    #[test]
    fn here_is_the_element_bearing_the_expression() -> DsigResult<()> {
        let document = build::parse_str(
            r#"<Signature xmlns="urn:s"><Transform><XPath>here()/ancestor::Signature[1]</XPath></Transform></Signature>"#,
        )
        .expect("well-formed test input");
        let xpath = document
            .get_elements_by_tag_name_ns(document.root(), Some("urn:s"), "XPath")
            .next()
            .expect("an XPath element");
        let ids = Ids(&document);
        let expression = Expression::parse("here()/ancestor::s:Signature[1]")?;
        // `s` is not bound on <XPath>, so bind it explicitly for the test.
        let context = Context {
            document: &document,
            namespaces: NamespaceContext::from_element(&document, xpath).bind("s", "urn:s"),
            here: Some(xpath),
            ids: &ids,
        };
        let set = expression
            .evaluate_to_node_set(&context, &Focus::single(XNode::Tree(document.root())))?;
        assert_eq!(set.len(), 1);
        assert_eq!(
            set.first().and_then(XNode::as_tree),
            document.document_element()
        );
        Ok(())
    }

    #[test]
    fn the_whole_document_node_set_matches_the_canonical_expression() -> DsigResult<()> {
        let document = sample();
        let expected = count(&document, "(//. | //@* | //namespace::*)")?;
        assert_eq!(whole_document_node_set(&document).len(), expected);
        Ok(())
    }
}
