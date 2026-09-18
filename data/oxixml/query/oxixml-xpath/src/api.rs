//! [`XPath`] and [`CompiledXPath`] — the public entry points.

use crate::ast::Expr;
use crate::context::{DynamicContext, StaticContext};
use crate::error::XPathResult;
use crate::eval;
use crate::parser;
use crate::value::{XpathItem, XpathValue};
use crate::version::XPathVersion;
use oxixml_infoset::{NodeRef, Sequence};
use std::sync::Arc;

/// The compiler.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::Tree;
/// use oxixml_xpath::{StaticContext, XPath, XPathVersion};
///
/// let tree = Tree::parse_str("<catalog><book price='9'/><book price='12'/></catalog>")?;
/// let root = tree.root();
///
/// let compiled = XPath::compile(
///     "sum(//book/@price)",
///     XPathVersion::XPath31,
///     StaticContext::new(),
/// )?;
/// assert_eq!(compiled.evaluate_to_number(root)?, 21.0);
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct XPath;

impl XPath {
    /// Compiles `expression` under `version` against `static_context`.
    ///
    /// # Errors
    ///
    /// Any static error: `XPST0003` for a syntax error or a construct the version does not have,
    /// `XPST0008` for an out-of-scope variable, `XPST0017` for an unknown function, `XPST0051`
    /// for an unknown atomic type, `XPST0080` for an illegal cast target, and `XPST0081` for an
    /// unbound namespace prefix.
    pub fn compile(
        expression: &str,
        version: XPathVersion,
        static_context: StaticContext,
    ) -> XPathResult<CompiledXPath> {
        let parsed = parser::parse(expression, version, &static_context)?;
        Ok(CompiledXPath {
            source: expression.to_owned(),
            version,
            static_context: Arc::new(static_context),
            expression: parsed,
        })
    }

    /// Compiles `expression` as XPath 1.0 with the default static context.
    ///
    /// # Errors
    ///
    /// As [`XPath::compile`].
    pub fn compile_10(expression: &str) -> XPathResult<CompiledXPath> {
        Self::compile(expression, XPathVersion::XPath10, StaticContext::new())
    }

    /// Compiles `expression` as XPath 3.1 with the default static context.
    ///
    /// # Errors
    ///
    /// As [`XPath::compile`].
    pub fn compile_31(expression: &str) -> XPathResult<CompiledXPath> {
        Self::compile(expression, XPathVersion::XPath31, StaticContext::new())
    }
}

/// A compiled expression, ready to evaluate any number of times against any number of documents.
///
/// The type is `Send + Sync` and cheap to clone, so one compiled expression can be shared across
/// threads: everything mutable lives in the [`DynamicContext`] passed to each evaluation.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::Tree;
/// use oxixml_xpath::{StaticContext, XPath, XPathVersion};
///
/// let compiled = XPath::compile(
///     "count(//item)",
///     XPathVersion::XPath31,
///     StaticContext::new(),
/// )?;
///
/// for source in ["<r><item/></r>", "<r><item/><item/></r>"] {
///     let tree = Tree::parse_str(source)?;
///     let counted = compiled.evaluate_to_number(tree.root())?;
///     assert!(counted >= 1.0);
/// }
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Debug, Clone)]
pub struct CompiledXPath {
    source: String,
    version: XPathVersion,
    static_context: Arc<StaticContext>,
    expression: Expr,
}

impl CompiledXPath {
    /// The source text this expression was compiled from.
    #[must_use]
    pub fn expression(&self) -> &str {
        &self.source
    }

    /// The version the expression was compiled under.
    #[must_use]
    pub const fn version(&self) -> XPathVersion {
        self.version
    }

    /// The static context the expression was compiled against.
    #[must_use]
    pub const fn static_context(&self) -> &Arc<StaticContext> {
        &self.static_context
    }

    /// The expression tree, for a host language that needs to inspect or rewrite it.
    #[must_use]
    pub const fn syntax_tree(&self) -> &Expr {
        &self.expression
    }

    /// A dynamic context over this expression's static context, with `context_item` in focus.
    #[must_use]
    pub fn context_with(&self, context_item: impl Into<XpathItem>) -> DynamicContext {
        DynamicContext::new(Arc::clone(&self.static_context))
            .with_version(self.version)
            .with_context_item(context_item)
    }

    /// A dynamic context over this expression's static context, with no focus.
    #[must_use]
    pub fn context(&self) -> DynamicContext {
        DynamicContext::new(Arc::clone(&self.static_context)).with_version(self.version)
    }

    /// Evaluates with `context_item` in focus.
    ///
    /// # Errors
    ///
    /// Any dynamic error, plus `XPTY0004` when the result holds a function item, which the data
    /// model cannot represent — use [`CompiledXPath::evaluate_value_in`] for those.
    pub fn evaluate(&self, context_item: impl Into<XpathItem>) -> XPathResult<Sequence> {
        self.evaluate_in(&self.context_with(context_item))
    }

    /// Evaluates with no context item.
    ///
    /// # Errors
    ///
    /// As [`CompiledXPath::evaluate`], plus `XPDY0002` if the expression needs a context item.
    pub fn evaluate_without_context(&self) -> XPathResult<Sequence> {
        self.evaluate_in(&self.context())
    }

    /// Evaluates against a fully specified dynamic context.
    ///
    /// # Errors
    ///
    /// As [`CompiledXPath::evaluate`].
    pub fn evaluate_in(&self, context: &DynamicContext) -> XPathResult<Sequence> {
        self.evaluate_value_in(context)?.into_sequence()
    }

    /// Evaluates against a fully specified dynamic context, keeping function items.
    ///
    /// # Errors
    ///
    /// Any dynamic error.
    pub fn evaluate_value_in(&self, context: &DynamicContext) -> XPathResult<XpathValue> {
        let context = context.clone().with_version(self.version);
        eval::evaluate(&self.expression, &context)
    }

    /// `fn:string` of the result, with `context_item` in focus.
    ///
    /// Under XPath 1.0 a node-set yields the string value of its first node in document order;
    /// under XPath 2.0 and later the result must hold at most one item.
    ///
    /// # Errors
    ///
    /// Any dynamic error, plus `XPTY0004` under XPath 2.0 semantics when the result holds more
    /// than one item.
    pub fn evaluate_to_string(&self, context_item: impl Into<XpathItem>) -> XPathResult<String> {
        self.evaluate_to_string_in(&self.context_with(context_item))
    }

    /// `fn:string` of the result, with no context item.
    ///
    /// # Errors
    ///
    /// As [`CompiledXPath::evaluate_to_string`].
    pub fn evaluate_to_string_without_context(&self) -> XPathResult<String> {
        self.evaluate_to_string_in(&self.context())
    }

    /// `fn:string` of the result, against a fully specified dynamic context.
    ///
    /// # Errors
    ///
    /// As [`CompiledXPath::evaluate_to_string`].
    pub fn evaluate_to_string_in(&self, context: &DynamicContext) -> XPathResult<String> {
        let context = context.clone().with_version(self.version);
        let value = self.evaluate_value_in(&context)?;
        eval::string_of(&value, &context)
    }

    /// `fn:number` of the result, with `context_item` in focus.
    ///
    /// # Errors
    ///
    /// Any dynamic error, plus `XPTY0004` under XPath 2.0 semantics when the result atomizes to
    /// more than one value.
    pub fn evaluate_to_number(&self, context_item: impl Into<XpathItem>) -> XPathResult<f64> {
        self.evaluate_to_number_in(&self.context_with(context_item))
    }

    /// `fn:number` of the result, with no context item.
    ///
    /// # Errors
    ///
    /// As [`CompiledXPath::evaluate_to_number`].
    pub fn evaluate_to_number_without_context(&self) -> XPathResult<f64> {
        self.evaluate_to_number_in(&self.context())
    }

    /// `fn:number` of the result, against a fully specified dynamic context.
    ///
    /// # Errors
    ///
    /// As [`CompiledXPath::evaluate_to_number`].
    pub fn evaluate_to_number_in(&self, context: &DynamicContext) -> XPathResult<f64> {
        let context = context.clone().with_version(self.version);
        let value = self.evaluate_value_in(&context)?;
        eval::number_of(&value, &context)
    }

    /// The effective boolean value of the result, with `context_item` in focus.
    ///
    /// # Errors
    ///
    /// Any dynamic error, plus `FORG0006` when the result has no effective boolean value.
    pub fn evaluate_to_boolean(&self, context_item: impl Into<XpathItem>) -> XPathResult<bool> {
        self.evaluate_to_boolean_in(&self.context_with(context_item))
    }

    /// The effective boolean value of the result, with no context item.
    ///
    /// # Errors
    ///
    /// As [`CompiledXPath::evaluate_to_boolean`].
    pub fn evaluate_to_boolean_without_context(&self) -> XPathResult<bool> {
        self.evaluate_to_boolean_in(&self.context())
    }

    /// The effective boolean value of the result, against a fully specified dynamic context.
    ///
    /// # Errors
    ///
    /// As [`CompiledXPath::evaluate_to_boolean`].
    pub fn evaluate_to_boolean_in(&self, context: &DynamicContext) -> XPathResult<bool> {
        eval::effective_boolean_value(&self.evaluate_value_in(context)?)
    }

    /// The result's nodes, in document order and without duplicates, with `context_item` in
    /// focus.
    ///
    /// Non-node items are dropped, which is what an XPath 1.0 caller expects of a node-set.
    ///
    /// # Errors
    ///
    /// Any dynamic error.
    pub fn evaluate_to_nodes(
        &self,
        context_item: impl Into<XpathItem>,
    ) -> XPathResult<Vec<NodeRef>> {
        self.evaluate_to_nodes_in(&self.context_with(context_item))
    }

    /// The result's nodes, with no context item.
    ///
    /// # Errors
    ///
    /// As [`CompiledXPath::evaluate_to_nodes`].
    pub fn evaluate_to_nodes_without_context(&self) -> XPathResult<Vec<NodeRef>> {
        self.evaluate_to_nodes_in(&self.context())
    }

    /// The result's nodes, against a fully specified dynamic context.
    ///
    /// # Errors
    ///
    /// As [`CompiledXPath::evaluate_to_nodes`].
    pub fn evaluate_to_nodes_in(&self, context: &DynamicContext) -> XPathResult<Vec<NodeRef>> {
        Ok(self.evaluate_value_in(context)?.nodes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::{AtomicValue, Tree};

    #[test]
    fn compiled_expressions_are_reusable_and_shareable() -> XPathResult<()> {
        let compiled = XPath::compile_31("count(//a)")?;
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        assert_send_sync(&compiled);

        for (source, expected) in [("<r><a/></r>", 1.0), ("<r><a/><a/></r>", 2.0)] {
            let tree = Tree::parse_str(source)?;
            assert_eq!(compiled.evaluate_to_number(tree.root())?, expected);
        }
        Ok(())
    }

    #[test]
    fn the_convenience_readers_agree_with_the_sequence() -> XPathResult<()> {
        let tree = Tree::parse_str("<r><a>x</a><a>y</a></r>")?;
        let compiled = XPath::compile_31("//a")?;
        let nodes = compiled.evaluate_to_nodes(tree.root())?;
        assert_eq!(nodes.len(), 2);
        assert_eq!(compiled.evaluate(tree.root())?.len(), 2);
        assert!(compiled.evaluate_to_boolean(tree.root())?);
        // Two nodes have no single string value under XPath 2.0 semantics.
        assert!(compiled.evaluate_to_string(tree.root()).is_err());

        let ten = XPath::compile_10("//a")?;
        assert_eq!(ten.evaluate_to_string(tree.root())?, "x");
        Ok(())
    }

    #[test]
    fn a_dynamic_context_supplies_variables() -> XPathResult<()> {
        let static_context = StaticContext::new().with_variable("wanted");
        let compiled = XPath::compile("$wanted + 1", XPathVersion::XPath31, static_context)?;
        let context = compiled
            .context()
            .with_variable("wanted", AtomicValue::integer(41_i64));
        assert_eq!(compiled.evaluate_to_number_in(&context)?, 42.0);
        Ok(())
    }

    #[test]
    fn metadata_is_preserved() -> XPathResult<()> {
        let compiled = XPath::compile_10("1 + 1")?;
        assert_eq!(compiled.expression(), "1 + 1");
        assert_eq!(compiled.version(), XPathVersion::XPath10);
        assert!(matches!(compiled.syntax_tree(), Expr::Binary { .. }));
        assert!(!compiled.static_context().default_collation().is_empty());
        Ok(())
    }
}
