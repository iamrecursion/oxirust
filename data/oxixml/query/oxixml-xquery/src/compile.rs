//! The compiler: it turns the syntax tree into a [`Program`], delegating everything it can to
//! [`oxixml_xpath`].
//!
//! # The delegation rule
//!
//! Every node the parser produced carries the byte range it covers and a flag saying whether its
//! subtree contains XQuery-only syntax. The compiler follows one rule:
//!
//! * a subtree that is **pure XPath** is compiled by [`oxixml_xpath::XPath::compile`] from its
//!   source range and becomes [`Program::Xpath`] — so path expressions, comparisons, arithmetic,
//!   casting, type matching, the function library and higher-order functions all behave exactly
//!   as the XPath engine's own conformance suite says they do, with no second implementation to
//!   drift;
//! * a subtree that is XQuery-only in a *structural* way — FLWOR, a constructor, `typeswitch`,
//!   `try`, a path with a contaminated predicate — is compiled to a node of its own;
//! * anything else — a comparison, an operator, a function call, a sequence — is compiled as a
//!   **template**: the XQuery-only children are lifted out into fresh variables, `$_h0`,
//!   `$_h1`, …, spliced into the source range in their place, and the result is handed to the
//!   XPath compiler. Evaluating the node evaluates the lifted children, binds them, and runs the
//!   template.
//!
//! The lifting is sound because a template is only ever built for a form whose operands are
//! evaluated exactly once, in the same focus and with the same variables in scope. Every form
//! that binds a variable, changes the focus or short-circuits gets a node of its own instead.

use crate::ast::{
    CatchTest, Constructor, ContentItem, LookupKey, NameSpec, Postfix, Syntax, SyntaxKind,
};
use crate::context::StaticQueryContext;
use crate::error::{XQueryError, XQueryErrorCode, XQueryResult};
use crate::eval::program::{
    CompiledAttribute, CompiledCatch, CompiledConstructor, CompiledContent, CompiledElement,
    CompiledFlwor, CompiledForBinding, CompiledGrouping, CompiledLetBinding, CompiledName,
    CompiledOrderSpec, CompiledPostfix, CompiledStep, CompiledSwitch, CompiledTypeCase,
    CompiledTypeSwitch, CompiledWindow, CompiledWindowCondition, FlworStep, LookupSpec, Program,
    TypeCheck, UserFunction,
};
use oxixml_infoset::ExpandedName;
use oxixml_xpath::{SequenceType, StaticContext, XPath, XPathVersion};
use std::collections::BTreeMap;

/// The variable a type check binds its operand to.
pub(crate) const TYPED_VALUE: &str = "_oxixml_typed";

/// `fn:data`, written so that no prolog declaration can rebind it.
const FN_DATA: &str = "Q{http://www.w3.org/2005/xpath-functions}data";

/// `fn:error` and `fn:QName`, written the same rebinding-proof way as [`FN_DATA`].
const FN_ERROR: &str = "Q{http://www.w3.org/2005/xpath-functions}error";
const FN_QNAME: &str = "Q{http://www.w3.org/2005/xpath-functions}QName";

/// `fn:function-arity`, written the same rebinding-proof way as [`FN_DATA`].
const FN_ARITY: &str = "Q{http://www.w3.org/2005/xpath-functions}function-arity";

/// The variable the function-coercion wrapper binds the supplied function to.
const COERCED: &str = "_oxixml_coerced";

/// The namespace of the specification's own error codes.
const ERR_NAMESPACE: &str = "http://www.w3.org/2005/xqt-errors";

/// The atomic types a function conversion may promote *from*, per target.
fn promotions(target: &str) -> &'static [&'static str] {
    match target {
        "double" => &["xs:decimal", "xs:float"],
        "float" => &["xs:decimal"],
        "string" => &["xs:anyURI"],
        _ => &[],
    }
}

/// The built-in atomic types that have no constructor function.
const UNCONSTRUCTIBLE: &[&str] = &[
    "anyAtomicType",
    "anySimpleType",
    "anyType",
    "NOTATION",
    "untyped",
];

/// Compiles one module's expressions against a fixed XPath static context.
pub(crate) struct Compiler<'a> {
    source: &'a str,
    /// The module's static context with *no* namespaces and no default element namespace: the
    /// two components a direct element constructor's namespace declaration attributes change.
    base: StaticContext,
    statik: StaticContext,
    options: std::sync::Arc<crate::eval::RuntimeOptions>,
    namespaces: BTreeMap<String, String>,
    default_element_namespace: Option<String>,
    holes: usize,
}

impl<'a> Compiler<'a> {
    pub(crate) fn new(
        source: &'a str,
        base: StaticContext,
        namespaces: BTreeMap<String, String>,
        default_element_namespace: Option<String>,
        options: std::sync::Arc<crate::eval::RuntimeOptions>,
    ) -> Self {
        let statik =
            derive_static_context(&base, &namespaces, default_element_namespace.as_deref());
        Self {
            source,
            base,
            statik,
            options,
            namespaces,
            default_element_namespace,
            holes: 0,
        }
    }

    fn rebuild(&mut self) {
        self.statik = derive_static_context(
            &self.base,
            &self.namespaces,
            self.default_element_namespace.as_deref(),
        );
    }

    pub(crate) const fn static_context(&self) -> &StaticContext {
        &self.statik
    }

    fn text(&self, node: &Syntax) -> &'a str {
        self.source.get(node.start..node.end).unwrap_or_default()
    }

    /// Compiles a range of query text as an XPath 3.1 expression.
    pub(crate) fn compile_xpath(&self, source: &str) -> XQueryResult<Program> {
        let compiled = XPath::compile(source, XPathVersion::XPath31, self.statik.clone())?;
        Ok(Program::Xpath(Box::new(compiled)))
    }

    /// Resolves a lexical name in the variable and attribute name space.
    pub(crate) fn resolve_variable(&self, lexical: &str) -> XQueryResult<ExpandedName> {
        self.resolve(lexical, None)
    }

    fn resolve(&self, lexical: &str, default: Option<&str>) -> XQueryResult<ExpandedName> {
        resolve_name(&self.namespaces, default, lexical)
    }

    fn fresh_hole(&mut self) -> String {
        let name = format!("_oxixml_hole_{}", self.holes);
        self.holes += 1;
        name
    }

    /// Compiles a type declaration into the check that applies the function conversion rules.
    pub(crate) fn compile_type_check(&self, declared: &str) -> XQueryResult<TypeCheck> {
        let source = conversion_source(declared);
        let compiled = XPath::compile(&source, XPathVersion::XPath31, self.statik.clone())
            .map_err(|error| type_declaration_error(declared, &error))?;
        Ok(TypeCheck {
            convert: Box::new(compiled),
        })
    }

    /// Compiles a type declaration into the *SequenceType matching* check a variable binding
    /// applies.
    ///
    /// [XQuery 3.1 §3.12](https://www.w3.org/TR/xquery-31/#id-variable-declarations) is explicit
    /// that a `let`, `for`, `group by`, window, quantified or prolog variable declaration
    /// *matches* its declared type — it does not convert to it. `let $i as xs:integer :=
    /// xs:untypedAtomic("1")` is therefore the type error `XPTY0004`, not the integer 1. Only a
    /// function's parameters and result get the function conversion rules of
    /// [`Compiler::compile_type_check`].
    pub(crate) fn compile_type_match(&self, declared: &str) -> XQueryResult<TypeCheck> {
        let source = format!("${TYPED_VALUE} treat as {}", declared.trim());
        let compiled = XPath::compile(&source, XPathVersion::XPath31, self.statik.clone())
            .map_err(|error| type_declaration_error(declared, &error))?;
        Ok(TypeCheck {
            convert: Box::new(compiled),
        })
    }

    /// Reads a function's declared parameter and result types as sequence types.
    ///
    /// XQuery 3.1 §2.5.5.5 judges a function item against a `function(…) as …` test by its
    /// declared signature, with the parameters contravariant and the result covariant, so a
    /// function item built for a declared or inline function has to carry the types rather than
    /// only the conversions that enforce them. An omitted type is `item()*`, as the grammar
    /// says. A type this engine cannot read back gives `None`, and the test then compares
    /// arities alone rather than deciding wrongly.
    pub(crate) fn declared_signature(
        &self,
        parameters: &[Option<String>],
        result: Option<&str>,
    ) -> Option<(Vec<SequenceType>, SequenceType)> {
        let read = |declared: Option<&str>| match declared {
            Some(declared) => SequenceType::parse(declared, &self.statik).ok(),
            None => Some(SequenceType::any()),
        };
        let mut types = Vec::with_capacity(parameters.len());
        for declared in parameters {
            types.push(read(declared.as_deref())?);
        }
        Some((types, read(result)?))
    }

    /// Compiles `$v instance of T`, which is how `typeswitch` decides a case.
    fn compile_instance_of(&self, declared: &str) -> XQueryResult<oxixml_xpath::CompiledXPath> {
        let source = format!("${TYPED_VALUE} instance of {declared}");
        let compiled = XPath::compile(&source, XPathVersion::XPath31, self.statik.clone())
            .map_err(|error| {
                XQueryError::new(
                    XQueryErrorCode::Xpst0003,
                    format!("`{declared}` is not a valid sequence type: {error}"),
                )
            })?;
        Ok(compiled)
    }

    // --- expressions --------------------------------------------------------------------------

    /// Compiles one expression.
    pub(crate) fn compile(&mut self, node: &Syntax) -> XQueryResult<Program> {
        if matches!(node.kind, SyntaxKind::ContextItem) {
            return Ok(Program::ContextItem);
        }
        if !node.xquery_only {
            return self.compile_xpath(self.text(node));
        }
        match &node.kind {
            SyntaxKind::Generic(children) => self.compile_template(node, children),
            SyntaxKind::ContextItem => Ok(Program::ContextItem),
            SyntaxKind::EmptySequence => Ok(Program::EmptySequence),
            SyntaxKind::Literal(value) => Ok(Program::Literal(value.clone())),
            SyntaxKind::Validate => Err(XQueryError::new(
                XQueryErrorCode::Xqst0075,
                "`validate` needs the Schema Import Feature, which this processor does not have",
            )
            .at(node.start)),
            SyntaxKind::Wrap(inner) => Ok(Program::Wrap(Box::new(self.compile(inner)?))),
            SyntaxKind::Extension { pragmas, body } => {
                // A pragma this processor does not recognize is ignored, but its name must
                // still be a QName whose prefix is bound.
                for pragma in pragmas {
                    self.resolve(pragma, None)?;
                }
                Ok(Program::Wrap(Box::new(self.compile(body)?)))
            }
            SyntaxKind::If {
                condition,
                then_branch,
                else_branch,
            } => Ok(Program::If {
                condition: Box::new(self.compile(condition)?),
                then_branch: Box::new(self.compile(then_branch)?),
                else_branch: Box::new(self.compile(else_branch)?),
            }),
            SyntaxKind::Logical { or, left, right } => Ok(Program::Logical {
                or: *or,
                left: Box::new(self.compile(left)?),
                right: Box::new(self.compile(right)?),
            }),
            SyntaxKind::Binding {
                kind,
                bindings,
                body,
            } => {
                let mut compiled = Vec::with_capacity(bindings.len());
                for binding in bindings {
                    compiled.push(CompiledLetBinding {
                        name: self.resolve_variable(&binding.name)?,
                        check: match &binding.declared_type {
                            Some(declared) => Some(self.compile_type_match(declared)?),
                            None => None,
                        },
                        value: self.compile(&binding.value)?,
                    });
                }
                Ok(Program::Binding {
                    kind: *kind,
                    bindings: compiled,
                    body: Box::new(self.compile(body)?),
                })
            }
            SyntaxKind::Path(path) => {
                let mut steps = Vec::with_capacity(path.steps.len());
                for step in &path.steps {
                    let mut predicates = Vec::with_capacity(step.predicates.len());
                    for predicate in &step.predicates {
                        predicates.push(self.compile(predicate)?);
                    }
                    steps.push(CompiledStep {
                        descendant: step.descendant,
                        simple_map: step.simple_map,
                        reverse: step.reverse,
                        base: self.compile(&step.base)?,
                        predicates,
                    });
                }
                Ok(Program::Path {
                    rooted: path.rooted,
                    rooted_descendant: path.rooted_descendant,
                    steps,
                })
            }
            SyntaxKind::Postfix { primary, postfixes } => {
                let mut compiled = Vec::with_capacity(postfixes.len());
                for postfix in postfixes {
                    compiled.push(self.compile_postfix(postfix)?);
                }
                Ok(Program::Postfix {
                    primary: Box::new(self.compile(primary)?),
                    postfixes: compiled,
                })
            }
            SyntaxKind::Inline(inline) => {
                let mut parameters = Vec::with_capacity(inline.parameters.len());
                let mut checks = Vec::with_capacity(inline.parameters.len());
                for (name, declared) in &inline.parameters {
                    parameters.push(self.resolve_variable(name)?);
                    checks.push(match declared {
                        Some(declared) => Some(self.compile_type_check(declared)?),
                        None => None,
                    });
                }
                let result = match &inline.result_type {
                    Some(declared) => Some(self.compile_type_check(declared)?),
                    None => None,
                };
                let declared_signature = self.declared_signature(
                    &inline
                        .parameters
                        .iter()
                        .map(|(_, declared)| declared.clone())
                        .collect::<Vec<_>>(),
                    inline.result_type.as_deref(),
                );
                Ok(Program::Inline(std::sync::Arc::new(UserFunction {
                    name: None,
                    options: std::sync::Arc::clone(&self.options),
                    parameters,
                    parameter_types: checks,
                    result_type: result,
                    declared_signature,
                    body: self.compile(&inline.body)?,
                })))
            }
            SyntaxKind::StringTemplate(parts) => {
                let mut compiled = Vec::with_capacity(parts.len());
                for part in parts {
                    compiled.push(self.compile_content(part)?);
                }
                Ok(Program::StringTemplate(compiled))
            }
            SyntaxKind::Flwor(flwor) => self.compile_flwor(flwor),
            SyntaxKind::Construct(constructor) => self.compile_constructor(constructor),
            SyntaxKind::TypeSwitch(expr) => {
                let mut cases = Vec::with_capacity(expr.cases.len());
                for case in &expr.cases {
                    let mut tests = Vec::with_capacity(case.types.len());
                    for declared in &case.types {
                        tests.push(self.compile_instance_of(declared)?);
                    }
                    cases.push(CompiledTypeCase {
                        variable: match &case.variable {
                            Some(name) => Some(self.resolve_variable(name)?),
                            None => None,
                        },
                        tests,
                        body: self.compile(&case.body)?,
                    });
                }
                Ok(Program::TypeSwitch(Box::new(CompiledTypeSwitch {
                    operand: self.compile(&expr.operand)?,
                    cases,
                    default_variable: match &expr.default_variable {
                        Some(name) => Some(self.resolve_variable(name)?),
                        None => None,
                    },
                    default_body: self.compile(&expr.default_body)?,
                })))
            }
            SyntaxKind::Switch(expr) => {
                let mut cases = Vec::with_capacity(expr.cases.len());
                for (operands, body) in &expr.cases {
                    let mut compiled = Vec::with_capacity(operands.len());
                    for operand in operands {
                        compiled.push(self.compile(operand)?);
                    }
                    cases.push((compiled, self.compile(body)?));
                }
                Ok(Program::Switch(Box::new(CompiledSwitch {
                    operand: self.compile(&expr.operand)?,
                    cases,
                    default_body: self.compile(&expr.default_body)?,
                })))
            }
            SyntaxKind::Try(expr) => {
                let mut catches = Vec::with_capacity(expr.catches.len());
                for catch in &expr.catches {
                    let mut tests = Vec::with_capacity(catch.tests.len());
                    for test in &catch.tests {
                        tests.push(self.compile_catch_test(test)?);
                    }
                    catches.push(CompiledCatch {
                        tests,
                        body: self.compile(&catch.body)?,
                    });
                }
                Ok(Program::Try {
                    body: Box::new(self.compile(&expr.body)?),
                    catches,
                })
            }
        }
    }

    fn compile_catch_test(&self, test: &CatchTest) -> XQueryResult<CatchTest> {
        Ok(match test {
            CatchTest::Any => CatchTest::Any,
            CatchTest::Local(local) => CatchTest::Local(local.clone()),
            CatchTest::Namespace(Some(prefix)) => {
                let uri = if prefix.contains(':') || prefix.contains('/') {
                    prefix.clone()
                } else {
                    self.namespaces.get(prefix).cloned().ok_or_else(|| {
                        XQueryError::new(
                            XQueryErrorCode::Xpst0081,
                            format!("the namespace prefix `{prefix}` is not bound"),
                        )
                    })?
                };
                CatchTest::Namespace(Some(uri))
            }
            CatchTest::Namespace(None) => CatchTest::Namespace(None),
            CatchTest::Name(lexical) => {
                let resolved = self.resolve(lexical, None)?;
                CatchTest::Name(format!(
                    "Q{{{}}}{}",
                    resolved.namespace().unwrap_or_default(),
                    resolved.local_name()
                ))
            }
        })
    }

    fn compile_postfix(&mut self, postfix: &Postfix) -> XQueryResult<CompiledPostfix> {
        Ok(match postfix {
            Postfix::Predicate(expr) => CompiledPostfix::Predicate(self.compile(expr)?),
            Postfix::Arguments(arguments) => {
                let mut compiled = Vec::with_capacity(arguments.len());
                for argument in arguments {
                    compiled.push(match argument {
                        Some(expr) => Some(self.compile(expr)?),
                        None => None,
                    });
                }
                CompiledPostfix::Arguments(compiled)
            }
            Postfix::Lookup(key) => CompiledPostfix::Lookup(match key {
                LookupKey::Name(name) => LookupSpec::Name(name.clone()),
                LookupKey::Integer(value) => LookupSpec::Integer(*value),
                LookupKey::Wildcard => LookupSpec::Wildcard,
                LookupKey::Computed(expr) => LookupSpec::Computed(Box::new(self.compile(expr)?)),
            }),
        })
    }

    /// Lifts the XQuery-only children out of `node` and compiles the rest as one XPath template.
    fn compile_template(&mut self, node: &Syntax, children: &[Syntax]) -> XQueryResult<Program> {
        let mut template = String::new();
        let mut holes = Vec::new();
        let mut cursor = node.start;
        for child in children {
            if !child.xquery_only {
                continue;
            }
            if child.start < cursor || child.end > node.end {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xpst0003,
                    "the expression could not be decomposed",
                )
                .at(node.start));
            }
            template.push_str(self.source.get(cursor..child.start).unwrap_or_default());
            let name = self.fresh_hole();
            template.push('$');
            template.push_str(&name);
            cursor = child.end;
            let compiled = self.compile(child)?;
            holes.push((ExpandedName::new(None::<&str>, name), compiled));
        }
        template.push_str(self.source.get(cursor..node.end).unwrap_or_default());
        let compiled = XPath::compile(&template, XPathVersion::XPath31, self.statik.clone())?;
        Ok(Program::Spliced {
            template: Box::new(compiled),
            holes,
        })
    }

    // --- FLWOR --------------------------------------------------------------------------------

    fn compile_flwor(&mut self, flwor: &crate::ast::Flwor) -> XQueryResult<Program> {
        let mut steps = Vec::with_capacity(flwor.clauses.len());
        // §3.12.7: a grouping spec with no `:=` names a variable an *earlier clause of this
        // FLWOR expression* bound; naming one bound outside it is `XQST0094`.
        let mut in_scope: Vec<ExpandedName> = Vec::new();
        for clause in &flwor.clauses {
            if let crate::ast::FlworClause::GroupBy(specs) = clause {
                for spec in specs {
                    if spec.value.is_some() {
                        continue;
                    }
                    let name = self.resolve_variable(&spec.name)?;
                    if !in_scope.contains(&name) {
                        return Err(XQueryError::new(
                            XQueryErrorCode::Xqst0094,
                            format!("`${name}` is not bound by a clause of this FLWOR expression"),
                        ));
                    }
                }
            }
            steps.push(self.compile_clause(clause)?);
            self.collect_bound(clause, &mut in_scope)?;
        }
        Ok(Program::Flwor(Box::new(CompiledFlwor {
            steps,
            result: self.compile(&flwor.result)?,
        })))
    }

    /// Adds the variables `clause` binds to `names`.
    fn collect_bound(
        &self,
        clause: &crate::ast::FlworClause,
        names: &mut Vec<ExpandedName>,
    ) -> XQueryResult<()> {
        use crate::ast::FlworClause;
        let push = |lexical: &str, names: &mut Vec<ExpandedName>| -> XQueryResult<()> {
            names.push(self.resolve_variable(lexical)?);
            Ok(())
        };
        match clause {
            FlworClause::For(bindings) => {
                for binding in bindings {
                    push(&binding.name, names)?;
                    if let Some(positional) = &binding.positional {
                        push(positional, names)?;
                    }
                }
            }
            FlworClause::Let(bindings) => {
                for binding in bindings {
                    push(&binding.name, names)?;
                }
            }
            FlworClause::Count(name) => push(name, names)?,
            FlworClause::GroupBy(specs) => {
                for spec in specs {
                    push(&spec.name, names)?;
                }
            }
            FlworClause::Window(window) => {
                push(&window.name, names)?;
                for condition in [Some(&window.start), window.end.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    for name in [
                        condition.current.as_ref(),
                        condition.positional.as_ref(),
                        condition.previous.as_ref(),
                        condition.next.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        push(name, names)?;
                    }
                }
            }
            FlworClause::Where(_) | FlworClause::OrderBy(_) => {}
        }
        Ok(())
    }

    /// Rejects a `collation` this processor does not know, per `XQST0076`.
    fn check_collation(&self, uri: Option<&str>) -> XQueryResult<()> {
        let Some(uri) = uri else {
            return Ok(());
        };
        let resolved = match self.options.base_uri.as_deref() {
            Some(base) => oxixml_fn::strings::uri::resolve_uri_against(uri, base)
                .unwrap_or_else(|_| uri.to_owned()),
            None => uri.to_owned(),
        };
        if oxixml_fn::Collations::new().resolve(&resolved).is_err() {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0076,
                format!("the collation `{uri}` is not one this processor knows"),
            ));
        }
        Ok(())
    }

    fn compile_clause(&mut self, clause: &crate::ast::FlworClause) -> XQueryResult<FlworStep> {
        use crate::ast::FlworClause;
        Ok(match clause {
            FlworClause::For(bindings) => {
                let mut compiled = Vec::with_capacity(bindings.len());
                for binding in bindings {
                    compiled.push(CompiledForBinding {
                        name: self.resolve_variable(&binding.name)?,
                        check: match &binding.declared_type {
                            Some(declared) => Some(self.compile_type_match(declared)?),
                            None => None,
                        },
                        allowing_empty: binding.allowing_empty,
                        positional: match &binding.positional {
                            Some(name) => Some(self.resolve_variable(name)?),
                            None => None,
                        },
                        value: self.compile(&binding.value)?,
                    });
                }
                FlworStep::For(compiled)
            }
            FlworClause::Let(bindings) => {
                let mut compiled = Vec::with_capacity(bindings.len());
                for binding in bindings {
                    compiled.push(CompiledLetBinding {
                        name: self.resolve_variable(&binding.name)?,
                        check: match &binding.declared_type {
                            Some(declared) => Some(self.compile_type_match(declared)?),
                            None => None,
                        },
                        value: self.compile(&binding.value)?,
                    });
                }
                FlworStep::Let(compiled)
            }
            FlworClause::Where(expr) => FlworStep::Where(self.compile(expr)?),
            FlworClause::Count(name) => FlworStep::Count(self.resolve_variable(name)?),
            FlworClause::GroupBy(specs) => {
                let mut compiled = Vec::with_capacity(specs.len());
                for spec in specs {
                    self.check_collation(spec.collation.as_deref())?;
                    compiled.push(CompiledGrouping {
                        name: self.resolve_variable(&spec.name)?,
                        check: match &spec.declared_type {
                            Some(declared) => Some(self.compile_type_match(declared)?),
                            None => None,
                        },
                        value: match &spec.value {
                            Some(expr) => Some(self.compile(expr)?),
                            None => None,
                        },
                        collation: spec.collation.clone(),
                    });
                }
                FlworStep::GroupBy(compiled)
            }
            FlworClause::OrderBy(specs) => {
                let mut compiled = Vec::with_capacity(specs.len());
                for spec in specs {
                    self.check_collation(spec.collation.as_deref())?;
                    compiled.push(CompiledOrderSpec {
                        key: self.compile(&spec.key)?,
                        ascending: spec.ascending,
                        empty_greatest: spec.empty_greatest,
                        collation: spec.collation.clone(),
                    });
                }
                FlworStep::OrderBy(compiled)
            }
            FlworClause::Window(window) => {
                let mut names: Vec<&String> = vec![&window.name];
                for condition in [Some(&window.start), window.end.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    for name in [
                        condition.current.as_ref(),
                        condition.positional.as_ref(),
                        condition.previous.as_ref(),
                        condition.next.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        if names.contains(&name) {
                            return Err(XQueryError::new(
                                XQueryErrorCode::Xqst0103,
                                format!("a window clause binds `${name}` more than once"),
                            ));
                        }
                        names.push(name);
                    }
                }
                FlworStep::Window(Box::new(CompiledWindow {
                    tumbling: window.tumbling,
                    name: self.resolve_variable(&window.name)?,
                    check: match &window.declared_type {
                        Some(declared) => Some(self.compile_type_match(declared)?),
                        None => None,
                    },
                    value: self.compile(&window.value)?,
                    start: self.compile_window_condition(&window.start)?,
                    end: match &window.end {
                        Some(condition) => Some(self.compile_window_condition(condition)?),
                        None => None,
                    },
                    only: window.only,
                }))
            }
        })
    }

    fn compile_window_condition(
        &mut self,
        condition: &crate::ast::WindowCondition,
    ) -> XQueryResult<CompiledWindowCondition> {
        Ok(CompiledWindowCondition {
            current: match &condition.current {
                Some(name) => Some(self.resolve_variable(name)?),
                None => None,
            },
            positional: match &condition.positional {
                Some(name) => Some(self.resolve_variable(name)?),
                None => None,
            },
            previous: match &condition.previous {
                Some(name) => Some(self.resolve_variable(name)?),
                None => None,
            },
            next: match &condition.next {
                Some(name) => Some(self.resolve_variable(name)?),
                None => None,
            },
            when: match &condition.when {
                Some(expr) => Some(self.compile(expr)?),
                None => None,
            },
        })
    }

    // --- constructors -------------------------------------------------------------------------

    fn compile_content(&mut self, item: &ContentItem) -> XQueryResult<CompiledContent> {
        Ok(match item {
            ContentItem::Text(text) => CompiledContent::Text(text.clone()),
            ContentItem::BoundaryWhitespace(text) => {
                CompiledContent::BoundaryWhitespace(text.clone())
            }
            ContentItem::Enclosed(expr) => CompiledContent::Enclosed(self.compile(expr)?),
            ContentItem::Nested(expr) => CompiledContent::Nested(self.compile(expr)?),
        })
    }

    fn compile_name_spec(&mut self, name: &NameSpec) -> XQueryResult<CompiledName> {
        Ok(match name {
            NameSpec::Lexical(lexical) => CompiledName::Lexical(lexical.clone()),
            NameSpec::Computed(expr) => CompiledName::Computed(Box::new(self.compile(expr)?)),
        })
    }

    fn compile_constructor(&mut self, constructor: &Constructor) -> XQueryResult<Program> {
        let compiled = match constructor {
            Constructor::DirectElement(element) => {
                // A namespace declaration attribute extends the statically known namespaces for
                // the whole of the constructor's content, so the enclosed expressions must be
                // compiled against the extended context.
                let saved_namespaces = self.namespaces.clone();
                let saved_default = self.default_element_namespace.clone();
                for (prefix, uri) in &element.namespaces {
                    match prefix {
                        Some(prefix) => {
                            self.namespaces.insert(prefix.clone(), uri.clone());
                        }
                        None => {
                            self.default_element_namespace = (!uri.is_empty()).then(|| uri.clone());
                        }
                    }
                }
                if !element.namespaces.is_empty() {
                    self.rebuild();
                }
                let compiled = self.direct_element(element);
                if !element.namespaces.is_empty() {
                    self.namespaces = saved_namespaces;
                    self.default_element_namespace = saved_default;
                    self.rebuild();
                }
                compiled?
            }
            other => self.compile_constructor_body(other)?,
        };
        Ok(Program::Construct(Box::new(compiled)))
    }

    fn direct_element(
        &mut self,
        element: &crate::ast::DirectElement,
    ) -> XQueryResult<CompiledConstructor> {
        let mut attributes = Vec::with_capacity(element.attributes.len());
        {
            for attribute in &element.attributes {
                let mut value = Vec::with_capacity(attribute.value.len());
                for part in &attribute.value {
                    value.push(self.compile_content(part)?);
                }
                attributes.push(CompiledAttribute {
                    name: attribute.name.clone(),
                    value,
                });
            }
        }
        let mut content = Vec::with_capacity(element.content.len());
        for part in &element.content {
            content.push(self.compile_content(part)?);
        }
        Ok(CompiledConstructor::DirectElement(Box::new(
            CompiledElement {
                name: element.name.clone(),
                namespaces: element.namespaces.clone(),
                attributes,
                content,
            },
        )))
    }

    fn compile_constructor_body(
        &mut self,
        constructor: &Constructor,
    ) -> XQueryResult<CompiledConstructor> {
        Ok(match constructor {
            Constructor::DirectElement(_) => unreachable!("handled by the caller"),
            Constructor::DirectComment(text) => CompiledConstructor::Comment(Some(
                Program::Literal(oxixml_infoset::AtomicValue::string(text.clone())),
            )),
            Constructor::DirectProcessingInstruction { target, content } => {
                CompiledConstructor::ProcessingInstruction {
                    name: CompiledName::Lexical(target.clone()),
                    content: Some(Program::Literal(oxixml_infoset::AtomicValue::string(
                        content.clone(),
                    ))),
                    literal: true,
                }
            }
            Constructor::ComputedElement { name, content } => CompiledConstructor::Element {
                name: self.compile_name_spec(name)?,
                content: match content {
                    Some(expr) => Some(self.compile(expr)?),
                    None => None,
                },
            },
            Constructor::ComputedAttribute { name, content } => CompiledConstructor::Attribute {
                name: self.compile_name_spec(name)?,
                content: match content {
                    Some(expr) => Some(self.compile(expr)?),
                    None => None,
                },
            },
            Constructor::ComputedDocument(expr) => {
                CompiledConstructor::Document(self.compile(expr)?)
            }
            Constructor::ComputedText(content) => CompiledConstructor::Text(match content {
                Some(expr) => Some(self.compile(expr)?),
                None => None,
            }),
            Constructor::ComputedComment(content) => CompiledConstructor::Comment(match content {
                Some(expr) => Some(self.compile(expr)?),
                None => None,
            }),
            Constructor::ComputedProcessingInstruction { name, content } => {
                CompiledConstructor::ProcessingInstruction {
                    name: self.compile_name_spec(name)?,
                    content: match content {
                        Some(expr) => Some(self.compile(expr)?),
                        None => None,
                    },
                    literal: false,
                }
            }
            Constructor::ComputedNamespace { prefix, uri } => CompiledConstructor::Namespace {
                prefix: self.compile_name_spec(prefix)?,
                uri: match uri {
                    Some(expr) => Some(self.compile(expr)?),
                    None => None,
                },
            },
        })
    }
}

/// Builds the XPath static context a module (or a constructor scope) compiles against.
fn derive_static_context(
    base: &StaticContext,
    namespaces: &BTreeMap<String, String>,
    default_element_namespace: Option<&str>,
) -> StaticContext {
    // XQuery's braced URI literal admits entity and character references where XPath's does not,
    // and the expressions handed to the XPath engine are ranges of *query* text.
    let mut statik = base.clone().with_xquery_host_language();
    for prefix in statik
        .namespaces()
        .map(|(prefix, _)| prefix.to_owned())
        .collect::<Vec<_>>()
    {
        statik = statik.without_namespace(&prefix);
    }
    statik = statik.with_namespaces(
        namespaces
            .iter()
            .filter(|(_, uri)| !uri.is_empty())
            .map(|(prefix, uri)| (prefix.clone(), uri.clone())),
    );
    if let Some(uri) = default_element_namespace.filter(|uri| !uri.is_empty()) {
        statik = statik.with_default_element_namespace(uri.to_owned());
    }
    statik
}

/// Resolves a lexical `QName`, a `Q{uri}local` name or an `NCName` against `namespaces`.
pub(crate) fn resolve_name(
    namespaces: &BTreeMap<String, String>,
    default: Option<&str>,
    lexical: &str,
) -> XQueryResult<ExpandedName> {
    if let Some(rest) = lexical.strip_prefix("Q{") {
        // The `}` to split on is the *last* one, not the first: a `BracedURILiteral` may not
        // contain a literal brace, but it may spell one with a character reference, and
        // `attribute Q{&#x7b;&#x7d;}x {}` really does name the namespace `{}`. A local name can
        // never contain a `}`, so the last one always closes the URI.
        let Some((uri, local)) = rest.rsplit_once('}') else {
            return Err(XQueryError::new(
                XQueryErrorCode::Xpst0003,
                format!("`{lexical}` is not a valid name"),
            ));
        };
        let namespace = (!uri.is_empty()).then_some(uri);
        return Ok(ExpandedName::new(namespace, local));
    }
    match lexical.split_once(':') {
        None => Ok(ExpandedName::new(
            default.filter(|uri| !uri.is_empty()),
            lexical,
        )),
        // A prefix bound to a zero-length URI is not bound at all.
        Some((prefix, local)) => match namespaces.get(prefix).filter(|uri| !uri.is_empty()) {
            Some(uri) => Ok(ExpandedName::new(Some(uri.as_str()), local)),
            None => Err(XQueryError::new(
                XQueryErrorCode::Xpst0081,
                format!("the namespace prefix `{prefix}` is not bound"),
            )),
        },
    }
}

/// Turns the failure to compile a type declaration into the error the specification names.
///
/// A `SequenceType` that does not parse is `XPST0003`, but one that parses and then names
/// something the static context does not define keeps its own code: `XPST0051` for an unknown
/// atomic type (`declare function local:error() as none { … }`), `XPST0081` for an unbound
/// prefix, `XQST0052` for a type that is not in the in-scope schema types. Reporting all three
/// as `XPST0003` would lose the distinction the suite tests for.
fn type_declaration_error(declared: &str, error: &oxixml_xpath::XPathError) -> XQueryError {
    let code = match error.name().local_name() {
        "XPST0051" => XQueryErrorCode::Xpst0051,
        "XPST0081" => XQueryErrorCode::Xpst0081,
        "XQST0052" => XQueryErrorCode::Xqst0052,
        _ => XQueryErrorCode::Xpst0003,
    };
    if code == XQueryErrorCode::Xpst0003 {
        return XQueryError::new(
            code,
            format!("`{declared}` is not a valid sequence type: {error}"),
        );
    }
    XQueryError::new(code, error.message().to_owned())
}

/// A `TypedFunctionTest` — `function(P₁, …, Pₙ) as R` — taken apart.
struct FunctionSignature {
    /// The item type as written, without any occurrence indicator.
    item: String,
    parameters: Vec<String>,
    result: String,
}

/// Splits `declared` into its parts when it is a `TypedFunctionTest`, with an optional
/// occurrence indicator and optional surrounding parentheses.
///
/// `function(*)` — the `AnyFunctionTest` — and anything that is not a function test at all give
/// `None`, because neither needs the coercion below.
fn typed_function_test(declared: &str) -> Option<FunctionSignature> {
    let trimmed = declared.trim();
    // `(function(…) as …)?` — an item type in parentheses so that an occurrence indicator can
    // follow it without being read as part of the result type.
    let item = if trimmed.starts_with('(') {
        let close = closing_paren(trimmed)?;
        let rest = trimmed.get(close + 1..)?.trim();
        if !matches!(rest, "" | "?" | "*" | "+") {
            return None;
        }
        trimmed.get(1..close)?.trim()
    } else {
        trimmed
    };
    let after_keyword = item.strip_prefix("function")?;
    if !after_keyword.trim_start().starts_with('(') {
        return None;
    }
    let signature = after_keyword.trim_start();
    let close = closing_paren(signature)?;
    let inside = signature.get(1..close)?.trim();
    if inside == "*" {
        return None;
    }
    let after = signature.get(close + 1..)?.trim_start();
    let result = after.strip_prefix("as")?;
    if !result.starts_with([' ', '\t', '\r', '\n', '(']) {
        return None;
    }
    Some(FunctionSignature {
        item: item.to_owned(),
        parameters: split_arguments(inside),
        result: result.trim().to_owned(),
    })
}

/// The index of the `)` that closes the `(` `text` begins with, ignoring nested parentheses and
/// anything inside a string literal or a `Q{…}` braced URI.
fn closing_paren(text: &str) -> Option<usize> {
    let mut depth = 0_usize;
    let mut quote: Option<char> = None;
    let mut braced = false;
    for (index, character) in text.char_indices() {
        if let Some(open) = quote {
            if character == open {
                quote = None;
            }
            continue;
        }
        if braced {
            if character == '}' {
                braced = false;
            }
            continue;
        }
        match character {
            '\'' | '"' => quote = Some(character),
            '{' => braced = true,
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

/// Splits a parameter list on its top-level commas.
fn split_arguments(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut parts = Vec::new();
    let mut depth = 0_usize;
    let mut quote: Option<char> = None;
    let mut braced = false;
    let mut start = 0_usize;
    for (index, character) in text.char_indices() {
        if let Some(open) = quote {
            if character == open {
                quote = None;
            }
            continue;
        }
        if braced {
            if character == '}' {
                braced = false;
            }
            continue;
        }
        match character {
            '\'' | '"' => quote = Some(character),
            '{' => braced = true,
            '(' | '[' => depth += 1,
            ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(text.get(start..index).unwrap_or_default().trim().to_owned());
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(text.get(start..).unwrap_or_default().trim().to_owned());
    parts
}

/// Builds the XPath expression that applies *function coercion* for `declared`.
///
/// [XQuery 3.1 §3.1.5.2](https://www.w3.org/TR/xquery-31/#id-function-coercion) makes function
/// coercion part of the function conversion rules: when the expected type is a function type and
/// the supplied value is a function item of the same arity that does not already match, the
/// supplied function is *wrapped* in one that has the expected signature and applies the
/// conversion rules to the arguments and to the result. A declared `function($x) { … }`, whose
/// own type is `function(item()*) as item()*`, is therefore a perfectly good argument for a
/// parameter declared `function(xs:string) as xs:string` — it is the wrapper, not the original,
/// that has to satisfy the signature.
///
/// The wrapper is written as an `InlineFunctionExpr` with the expected parameter and result
/// types, which is exactly the specification's definition: an inline function already applies
/// the conversion rules to what it is given and to what it returns.
fn coercion_source(declared: &str, signature: &FunctionSignature) -> String {
    let arity = signature.parameters.len();
    let parameters = signature
        .parameters
        .iter()
        .enumerate()
        .map(|(index, declared)| format!("${COERCED}_a{index} as {declared}"))
        .collect::<Vec<_>>()
        .join(", ");
    let arguments = (0..arity)
        .map(|index| format!("${COERCED}_a{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let item = &signature.item;
    let result = &signature.result;
    format!(
        "(for ${COERCED} in ${TYPED_VALUE} return \
         (if (${COERCED} instance of {item}) then ${COERCED} \
          else if (${COERCED} instance of function(*) \
                   and {FN_ARITY}(${COERCED}) eq {arity}) \
          then function({parameters}) as {result} {{ ${COERCED}({arguments}) }} \
          else ${COERCED})) treat as {declared}"
    )
}

/// Builds the XPath expression that applies the function conversion rules for `declared`.
///
/// The conversion is the specification's, not a cast: an `xs:untypedAtomic` becomes the target
/// type, `xs:decimal` and `xs:float` promote to `xs:double`, `xs:decimal` promotes to `xs:float`,
/// `xs:anyURI` promotes to `xs:string`, and nothing else changes. Whatever the conversion did not
/// fix is caught by the closing `treat as`.
fn conversion_source(declared: &str) -> String {
    let trimmed = declared.trim();
    if let Some(signature) = typed_function_test(trimmed) {
        return coercion_source(trimmed, &signature);
    }
    let item = trimmed
        .strip_suffix(['?', '*', '+'])
        .unwrap_or(trimmed)
        .trim();
    let Some(local) = atomic_local_name(item) else {
        return format!("${TYPED_VALUE} treat as {trimmed}");
    };
    // XQuery 3.1 §2.5.5 / XPath 3.1 §3.1.5.2: `xs:QName` and `xs:NOTATION` are the
    // *namespace-sensitive* types, and converting an `xs:untypedAtomic` value to one of them is
    // `XPTY0117` — the lexical form would have to be resolved against namespace bindings the
    // callee does not have. Everything else about the declaration is left to the `treat as`.
    if matches!(local, "QName" | "NOTATION") {
        return format!(
            "(for $_oxixml_item in {FN_DATA}(${TYPED_VALUE}) return \
             (if ($_oxixml_item instance of xs:untypedAtomic) \
              then {FN_ERROR}({FN_QNAME}('{ERR_NAMESPACE}', 'XPTY0117'), \
                   'an xs:untypedAtomic value cannot be converted to the \
                    namespace-sensitive type {item}') \
              else $_oxixml_item)) treat as {trimmed}"
        );
    }
    if UNCONSTRUCTIBLE.contains(&local) {
        return format!("{FN_DATA}(${TYPED_VALUE}) treat as {trimmed}");
    }
    let mut guard = "$_oxixml_item instance of xs:untypedAtomic".to_owned();
    for promotable in promotions(local) {
        guard.push_str(&format!(" or $_oxixml_item instance of {promotable}"));
    }
    format!(
        "(for $_oxixml_item in {FN_DATA}(${TYPED_VALUE}) return \
         (if ($_oxixml_item instance of {item}) then $_oxixml_item \
          else if ({guard}) then {item}($_oxixml_item) \
          else $_oxixml_item)) treat as {trimmed}"
    )
}

/// The local name of `item` when it names a built-in XML Schema atomic type.
fn atomic_local_name(item: &str) -> Option<&str> {
    if let Some(rest) = item.strip_prefix("Q{http://www.w3.org/2001/XMLSchema}") {
        return valid_ncname(rest).then_some(rest);
    }
    let rest = item.strip_prefix("xs:")?;
    valid_ncname(rest).then_some(rest)
}

fn valid_ncname(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// The namespace no prefix may be bound to, and which binds no prefix but `xmlns`.
pub(crate) const XMLNS_NAMESPACE: &str = "http://www.w3.org/2000/xmlns/";

/// The rule a *prolog* namespace declaration, module import or schema import obeys:
/// [§4.2](https://www.w3.org/TR/xquery-31/#id-namespace-declaration) forbids the prefixes `xml`
/// and `xmlns` outright, and forbids either reserved namespace URI whatever prefix it is given.
pub(crate) fn is_reserved_prolog_binding(prefix: &str, namespace: &str) -> bool {
    prefix == "xml"
        || prefix == "xmlns"
        || namespace == XMLNS_NAMESPACE
        || namespace == oxixml_qname::XML_NAMESPACE
}

/// The namespace an unprefixed annotation name is in.
pub(crate) const XQUERY_NAMESPACE: &str = "http://www.w3.org/2012/xquery";

/// The annotations the specification itself defines.
const STANDARD_ANNOTATIONS: &[&str] = &[
    "public",
    "private",
    "deterministic",
    "nondeterministic",
    "assignable",
    "unassignable",
    "simple",
    "updating",
];

/// Whether an annotation name is in a namespace a query may not declare an annotation in.
pub(crate) fn is_reserved_annotation(name: &oxixml_infoset::ExpandedName) -> bool {
    match name.namespace() {
        None => false,
        Some(XQUERY_NAMESPACE) => !STANDARD_ANNOTATIONS.contains(&name.local_name()),
        Some(namespace) => {
            is_reserved_function_namespace(namespace)
                || namespace == oxixml_qname::XML_NAMESPACE
                || namespace == XMLNS_NAMESPACE
        }
    }
}

/// The namespaces a query may not declare a function in.
pub(crate) fn is_reserved_function_namespace(namespace: &str) -> bool {
    matches!(
        namespace,
        oxixml_xpath::XSD_NAMESPACE
            | oxixml_xpath::XSI_NAMESPACE
            | oxixml_xpath::FN_NAMESPACE
            | oxixml_xpath::MATH_NAMESPACE
            | oxixml_xpath::ARRAY_NAMESPACE
            | oxixml_xpath::MAP_NAMESPACE
            // The predeclared `xml` and `xmlns` namespaces are reserved for functions too:
            // XQuery 3.1 §4.18 lists every namespace of §2.1.1's predeclared bindings, and the
            // suite requires `declare function xml:foo() { 1 }` to be `XQST0045`.
            | oxixml_qname::XML_NAMESPACE
            | XMLNS_NAMESPACE
    )
}

/// Builds the XPath static context a query's expressions are compiled against.
pub(crate) fn xpath_static_context(
    context: &StaticQueryContext,
    namespaces: &BTreeMap<String, String>,
    functions: oxixml_fn::FunctionRegistry,
) -> StaticContext {
    let mut statik = StaticContext::new().with_undeclared_variables();
    for prefix in statik
        .namespaces()
        .map(|(prefix, _)| prefix.to_owned())
        .collect::<Vec<_>>()
    {
        statik = statik.without_namespace(&prefix);
    }
    statik = statik.with_namespaces(
        namespaces
            .iter()
            .map(|(prefix, uri)| (prefix.clone(), uri.clone())),
    );
    statik = statik
        .with_default_function_namespace(context.default_function_namespace().to_owned())
        .with_default_collation(context.default_collation().to_owned())
        .with_functions(functions);
    if let Some(base) = context.base_uri() {
        statik = statik.with_base_uri(base.to_owned());
    }
    statik
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_type_is_only_treated() {
        assert_eq!(
            conversion_source("element()*"),
            "$_oxixml_typed treat as element()*"
        );
    }

    #[test]
    fn an_atomic_type_gets_the_conversion_rules() {
        let source = conversion_source("xs:double");
        assert!(source.contains(&format!("{FN_DATA}($_oxixml_typed)")));
        assert!(source.contains("xs:double($_oxixml_item)"));
        assert!(source.contains("instance of xs:decimal"));
        assert!(source.ends_with("treat as xs:double"));
    }

    #[test]
    fn an_unconstructible_type_only_atomizes() {
        assert_eq!(
            conversion_source("xs:anyAtomicType?"),
            format!("{FN_DATA}($_oxixml_typed) treat as xs:anyAtomicType?")
        );
    }

    #[test]
    fn reserved_bindings_are_recognized() {
        assert!(is_reserved_prolog_binding("xml", "urn:nope"));
        assert!(is_reserved_prolog_binding(
            "xml",
            oxixml_qname::XML_NAMESPACE
        ));
        assert!(is_reserved_prolog_binding("xmlns", "urn:x"));
        assert!(is_reserved_prolog_binding("p", XMLNS_NAMESPACE));
        assert!(is_reserved_prolog_binding("p", oxixml_qname::XML_NAMESPACE));
        assert!(!is_reserved_prolog_binding("p", "urn:p"));
        assert!(is_reserved_function_namespace(oxixml_xpath::FN_NAMESPACE));
        assert!(!is_reserved_function_namespace("urn:mine"));
    }

    #[test]
    fn reserved_annotations_are_recognized() {
        use oxixml_infoset::ExpandedName;
        let standard = ExpandedName::new(Some(XQUERY_NAMESPACE), "private");
        let invented = ExpandedName::new(Some(XQUERY_NAMESPACE), "invented");
        let mine = ExpandedName::new(Some("urn:mine"), "flag");
        assert!(!is_reserved_annotation(&standard));
        assert!(is_reserved_annotation(&invented));
        assert!(!is_reserved_annotation(&mine));
        assert!(is_reserved_annotation(&ExpandedName::new(
            Some(oxixml_qname::XML_NAMESPACE),
            "x"
        )));
    }
}
