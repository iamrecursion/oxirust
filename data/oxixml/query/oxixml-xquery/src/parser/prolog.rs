//! The module declaration, the query prolog, and the query body.

use super::Parser;
use crate::ast::unexpected;
use crate::context::{BoundarySpace, ConstructionMode, CopyNamespaces, EmptyOrder, OrderingMode};
use crate::error::{XQueryError, XQueryErrorCode, XQueryResult};
use crate::lexer::Token;
use crate::prolog::{
    ContextItemDeclaration, DecimalFormatDeclaration, FunctionDeclaration, Module, ModuleImport,
    Prolog, SchemaImport, VariableDeclaration,
};

impl Parser<'_> {
    /// `Module ::= VersionDecl? (LibraryModule | MainModule)`
    pub(crate) fn module(&mut self) -> XQueryResult<Module> {
        let mut prolog = Prolog::default();
        self.version_declaration(&mut prolog)?;
        let module_namespace = self.module_declaration()?;
        self.prolog(&mut prolog)?;
        let body = if module_namespace.is_some() {
            None
        } else {
            Some(self.expr()?)
        };
        Ok(Module {
            module_namespace,
            prolog,
            body,
            seen: self.take_observations(),
        })
    }

    fn version_declaration(&mut self, prolog: &mut Prolog) -> XQueryResult<()> {
        if !self.at_keyword("xquery") {
            return Ok(());
        }
        if !(self.next_is_keyword("version")? || self.next_is_keyword("encoding")?) {
            return Ok(());
        }
        self.advance()?;
        if self.eat_keyword("version")? {
            let version = self.uri_literal("a version string")?;
            if version != "1.0" && version != "3.0" && version != "3.1" {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqst0031,
                    format!("this processor implements XQuery 3.1, not {version}"),
                ));
            }
            prolog.version = Some(version);
        }
        if self.eat_keyword("encoding")? {
            let encoding = self.uri_literal("an encoding name")?;
            if !is_encoding_name(&encoding) {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqst0087,
                    format!("`{encoding}` is not a valid encoding name"),
                ));
            }
            prolog.encoding = Some(encoding);
        }
        self.expect(&Token::Semicolon, "`;` after the version declaration")?;
        Ok(())
    }

    fn module_declaration(&mut self) -> XQueryResult<Option<(String, String)>> {
        if !(self.at_keyword("module") && self.next_is_keyword("namespace")?) {
            return Ok(None);
        }
        self.advance()?;
        self.advance()?;
        let prefix = self.ncname("a module prefix")?;
        self.expect(&Token::Equal, "`=`")?;
        let uri = collapse(&self.uri_literal("the module's target namespace")?);
        if uri.is_empty() {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0088,
                "a module's target namespace may not be a zero-length string",
            ));
        }
        self.expect(&Token::Semicolon, "`;` after the module declaration")?;
        Ok(Some((prefix, uri)))
    }

    /// `Prolog ::= ((DefaultNamespaceDecl | Setter | NamespaceDecl | Import) Separator)*
    /// ((ContextItemDecl | AnnotatedDecl | OptionDecl) Separator)*`
    ///
    /// The grammar is in two halves, and the second may not be followed by the first: once a
    /// variable, function, option or context-item declaration has been seen, a setter, namespace
    /// declaration or import is a syntax error.
    fn prolog(&mut self, prolog: &mut Prolog) -> XQueryResult<()> {
        loop {
            if self.at_keyword("import") && self.starts_import()? {
                self.close_check("an import")?;
                self.import_declaration(prolog)?;
            } else if self.at_keyword("declare") && self.starts_declaration()? {
                self.declaration(prolog)?;
            } else {
                return Ok(());
            }
            self.expect(&Token::Semicolon, "`;` after a prolog declaration")?;
        }
    }

    /// Whether an `import` at the cursor opens an `Import` rather than a name test.
    ///
    /// XQuery reserves no words ([§2.1](https://www.w3.org/TR/xquery-31/#id-terminal-delimitation)),
    /// so `import` is also a perfectly good element name and `import ne import` is a comparison
    /// of two name tests, not a malformed prolog. `Import ::= SchemaImport | ModuleImport` and
    /// both continue with a keyword, so one token of lookahead decides.
    fn starts_import(&mut self) -> XQueryResult<bool> {
        Ok(self.next_is_keyword("schema")? || self.next_is_keyword("module")?)
    }

    /// Whether a `declare` at the cursor opens a prolog declaration rather than a name test.
    ///
    /// Every declaration `declare` can open continues with one of a closed set of keywords, or
    /// with the `%` of an annotation. Anything else — `declare`, `declare ne gt`, `declare union
    /// …` — is the query body, beginning with the name test `declare`.
    fn starts_declaration(&mut self) -> XQueryResult<bool> {
        if self.next_is(&Token::Percent)? {
            return Ok(true);
        }
        for keyword in [
            "default",
            "boundary-space",
            "base-uri",
            "construction",
            "ordering",
            "copy-namespaces",
            "decimal-format",
            "namespace",
            "context",
            "variable",
            "function",
            "option",
        ] {
            if self.next_is_keyword(keyword)? {
                // `declare namespace(…)` and the like are function calls on a name test.
                return Ok(!self.third_is_left_paren()?);
            }
        }
        Ok(false)
    }

    /// Whether the token two places after the cursor is `(`.
    fn third_is_left_paren(&mut self) -> XQueryResult<bool> {
        let state = self.state();
        self.advance()?;
        let found = self.next_is(&Token::LeftParen)?;
        self.restore(state);
        Ok(found)
    }

    /// Rejects a first-half declaration that follows a second-half one.
    fn close_check(&mut self, what: &str) -> XQueryResult<()> {
        if self.prolog_closed() {
            return Err(self.error(format!(
                "{what} may not follow a variable, function, option or context-item declaration"
            )));
        }
        Ok(())
    }

    fn import_declaration(&mut self, prolog: &mut Prolog) -> XQueryResult<()> {
        self.advance()?;
        if self.eat_keyword("schema")? {
            let mut prefix = None;
            let mut default_element_namespace = false;
            if self.at_keyword("default") {
                self.advance()?;
                self.expect_keyword("element")?;
                self.expect_keyword("namespace")?;
                default_element_namespace = true;
            } else if self.at_keyword("namespace") {
                self.advance()?;
                prefix = Some(self.ncname("a schema prefix")?);
                self.expect(&Token::Equal, "`=`")?;
            }
            let target_namespace = collapse(&self.uri_literal("the schema's target namespace")?);
            self.at_locations()?;
            prolog.schema_imports.push(SchemaImport {
                prefix,
                default_element_namespace,
                target_namespace,
            });
            return Ok(());
        }
        self.expect_keyword("module")?;
        let mut prefix = None;
        if self.eat_keyword("namespace")? {
            prefix = Some(self.ncname("a module prefix")?);
            self.expect(&Token::Equal, "`=`")?;
        }
        let target_namespace = collapse(&self.uri_literal("the module's target namespace")?);
        if target_namespace.is_empty() {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0088,
                "an imported module's target namespace may not be a zero-length string",
            ));
        }
        let locations = self.at_locations()?;
        prolog.module_imports.push(ModuleImport {
            prefix,
            target_namespace,
            locations,
        });
        Ok(())
    }

    fn at_locations(&mut self) -> XQueryResult<Vec<String>> {
        let mut locations = Vec::new();
        if self.eat_keyword("at")? {
            loop {
                let at = self.offset();
                let location = self.uri_literal("a location hint")?;
                // XQuery 3.1 §4.11: a `URILiteral` that is not in the lexical space of
                // `xs:anyURI` is `XQST0046`, and this is checked before anything is imported —
                // `import schema "…" at "%gg"` is a bad URI whatever the processor's features.
                if !is_uri_lexical(&location) {
                    return Err(XQueryError::new(
                        XQueryErrorCode::Xqst0046,
                        format!("`{location}` is not a valid URI"),
                    )
                    .at(at));
                }
                locations.push(location);
                if !self.eat(&Token::Comma)? {
                    break;
                }
            }
        }
        Ok(locations)
    }

    fn declaration(&mut self, prolog: &mut Prolog) -> XQueryResult<()> {
        self.advance()?;
        if self.at_keyword("default") {
            self.close_check("a `declare default` declaration")?;
            return self.default_declaration(prolog);
        }
        if self.at_any_keyword(&[
            "boundary-space",
            "base-uri",
            "construction",
            "ordering",
            "copy-namespaces",
            "decimal-format",
        ]) || (self.at_keyword("namespace") && !self.next_is(&Token::LeftParen)?)
        {
            self.close_check("a setter or namespace declaration")?;
        }
        if self.at_keyword("boundary-space") {
            self.advance()?;
            let policy = if self.eat_keyword("preserve")? {
                BoundarySpace::Preserve
            } else {
                self.expect_keyword("strip")?;
                BoundarySpace::Strip
            };
            return Self::set_once(
                &mut prolog.boundary_space,
                policy,
                XQueryErrorCode::Xqst0068,
                "boundary-space",
            );
        }
        if self.at_keyword("base-uri") {
            self.advance()?;
            let uri = self.uri_literal("a base URI")?;
            return Self::set_once(
                &mut prolog.base_uri,
                uri,
                XQueryErrorCode::Xqst0032,
                "base-uri",
            );
        }
        if self.at_keyword("construction") {
            self.advance()?;
            let mode = if self.eat_keyword("preserve")? {
                ConstructionMode::Preserve
            } else {
                self.expect_keyword("strip")?;
                ConstructionMode::Strip
            };
            return Self::set_once(
                &mut prolog.construction,
                mode,
                XQueryErrorCode::Xqst0067,
                "construction",
            );
        }
        if self.at_keyword("ordering") {
            self.advance()?;
            let mode = if self.eat_keyword("ordered")? {
                OrderingMode::Ordered
            } else {
                self.expect_keyword("unordered")?;
                OrderingMode::Unordered
            };
            return Self::set_once(
                &mut prolog.ordering,
                mode,
                XQueryErrorCode::Xqst0065,
                "ordering",
            );
        }
        if self.at_keyword("copy-namespaces") {
            self.advance()?;
            let preserve = if self.eat_keyword("preserve")? {
                true
            } else {
                self.expect_keyword("no-preserve")?;
                false
            };
            self.expect(&Token::Comma, "`,`")?;
            let inherit = if self.eat_keyword("inherit")? {
                true
            } else {
                self.expect_keyword("no-inherit")?;
                false
            };
            return Self::set_once(
                &mut prolog.copy_namespaces,
                CopyNamespaces { preserve, inherit },
                XQueryErrorCode::Xqst0055,
                "copy-namespaces",
            );
        }
        if self.at_keyword("decimal-format") {
            self.advance()?;
            let (name, _) = self.eqname("a decimal format name")?;
            let properties = self.decimal_format_properties()?;
            prolog.decimal_formats.push(DecimalFormatDeclaration {
                name: Some(name),
                properties,
            });
            return Ok(());
        }
        if self.at_keyword("namespace") {
            self.advance()?;
            let prefix = self.ncname("a namespace prefix")?;
            self.expect(&Token::Equal, "`=`")?;
            let uri = collapse(&self.uri_literal("a namespace URI")?);
            prolog.namespaces.push((prefix, uri));
            return Ok(());
        }
        if self.at_keyword("context") && self.next_is_keyword("item")? {
            self.close_prolog();
            self.advance()?;
            self.advance()?;
            // `ContextItemDecl` takes an `ItemType`, not a `SequenceType`: the context item is
            // exactly one item, so an occurrence indicator is a syntax error.
            let declared_type = if self.eat_keyword("as")? {
                Some(self.item_type_source()?)
            } else {
                None
            };
            let mut value = None;
            let mut external = false;
            if self.eat_keyword("external")? {
                external = true;
                if self.eat(&Token::Assign)? {
                    value = Some(self.expr_single()?);
                }
            } else {
                self.expect(&Token::Assign, "`:=`")?;
                value = Some(self.expr_single()?);
            }
            if prolog.context_item.is_some() {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqst0099,
                    "the context item type is declared more than once",
                ));
            }
            prolog.context_item = Some(ContextItemDeclaration {
                declared_type,
                value,
                external,
            });
            return Ok(());
        }
        if self.at_keyword("option") {
            self.close_prolog();
            self.advance()?;
            let (name, _) = self.eqname("an option name")?;
            let value = self.uri_literal("an option value")?;
            prolog.options.push((name, value));
            return Ok(());
        }
        let annotations = self.annotations()?;
        if self.at_keyword("variable") {
            self.close_prolog();
            self.advance()?;
            return self.variable_declaration(prolog, annotations);
        }
        if self.at_keyword("function") {
            self.close_prolog();
            self.advance()?;
            return self.function_declaration(prolog, annotations);
        }
        Err(self.error(unexpected(self.peek(), "a prolog declaration")))
    }

    fn default_declaration(&mut self, prolog: &mut Prolog) -> XQueryResult<()> {
        self.advance()?;
        if self.at_keyword("element") {
            self.advance()?;
            self.expect_keyword("namespace")?;
            let uri = collapse(&self.uri_literal("a namespace URI")?);
            check_default_namespace(&uri)?;
            return Self::set_once(
                &mut prolog.default_element_namespace,
                uri,
                XQueryErrorCode::Xqst0066,
                "default element namespace",
            );
        }
        if self.at_keyword("function") {
            self.advance()?;
            self.expect_keyword("namespace")?;
            let uri = collapse(&self.uri_literal("a namespace URI")?);
            check_default_namespace(&uri)?;
            return Self::set_once(
                &mut prolog.default_function_namespace,
                uri,
                XQueryErrorCode::Xqst0066,
                "default function namespace",
            );
        }
        if self.at_keyword("collation") {
            self.advance()?;
            let uri = self.uri_literal("a collation URI")?;
            return Self::set_once(
                &mut prolog.default_collation,
                uri,
                XQueryErrorCode::Xqst0038,
                "default collation",
            );
        }
        if self.at_keyword("order") {
            self.advance()?;
            self.expect_keyword("empty")?;
            let order = if self.eat_keyword("greatest")? {
                EmptyOrder::Greatest
            } else {
                self.expect_keyword("least")?;
                EmptyOrder::Least
            };
            return Self::set_once(
                &mut prolog.empty_order,
                order,
                XQueryErrorCode::Xqst0069,
                "default order",
            );
        }
        if self.at_keyword("decimal-format") {
            self.advance()?;
            let properties = self.decimal_format_properties()?;
            prolog.decimal_formats.push(DecimalFormatDeclaration {
                name: None,
                properties,
            });
            return Ok(());
        }
        Err(self.error(unexpected(self.peek(), "a `declare default` declaration")))
    }

    fn decimal_format_properties(&mut self) -> XQueryResult<Vec<(String, String)>> {
        let mut properties = Vec::new();
        while matches!(self.peek(), Token::Name(_)) {
            let (name, _) = self.eqname("a decimal-format property")?;
            self.expect(&Token::Equal, "`=`")?;
            let value = self.uri_literal("a decimal-format property value")?;
            properties.push((name, value));
        }
        Ok(properties)
    }

    fn variable_declaration(
        &mut self,
        prolog: &mut Prolog,
        annotations: Vec<String>,
    ) -> XQueryResult<()> {
        self.expect(&Token::Dollar, "`$`")?;
        // Not a `binding_name`: a prolog variable is not a *binder* with an extent, it is a
        // declaration in the module's static context, and the compiler treats it as one. Giving
        // it a scope that runs to the end of the module would also make it shadow the one place
        // it is *not* in scope — its own initializing expression.
        let (name, _) = self.eqname("a variable name")?;
        let declared_type = self.optional_type_declaration()?;
        let mut value = None;
        let mut external = false;
        if self.eat_keyword("external")? {
            external = true;
            if self.eat(&Token::Assign)? {
                value = Some(self.expr_single()?);
            }
        } else {
            self.expect(&Token::Assign, "`:=`")?;
            value = Some(self.expr_single()?);
        }
        prolog.variables.push(VariableDeclaration {
            name,
            declared_type,
            value,
            external,
            annotations,
        });
        Ok(())
    }

    fn function_declaration(
        &mut self,
        prolog: &mut Prolog,
        annotations: Vec<String>,
    ) -> XQueryResult<()> {
        let at = self.offset();
        let (name, _) = self.eqname("a function name")?;
        if Self::is_reserved_function_name(&name) {
            return Err(XQueryError::syntax(
                format!("`{name}` is a reserved function name and may not be declared"),
                at,
            ));
        }
        self.expect(&Token::LeftParen, "`(`")?;
        // A parameter is in scope over this function's body and nowhere else (XQuery 3.1 §4.18).
        let parameter_mark = self.binding_mark();
        let mut parameters = Vec::new();
        if self.peek() != &Token::RightParen {
            loop {
                self.expect(&Token::Dollar, "`$`")?;
                let parameter = self.binding_name("a parameter name")?;
                let declared = self.optional_type_declaration()?;
                parameters.push((parameter, declared));
                if !self.eat(&Token::Comma)? {
                    break;
                }
            }
        }
        self.expect(&Token::RightParen, "`)`")?;
        let result_type = if self.eat_keyword("as")? {
            Some(self.sequence_type_source()?)
        } else {
            None
        };
        let mut body = None;
        let mut external = false;
        if self.eat_keyword("external")? {
            external = true;
        } else {
            self.expect(&Token::LeftBrace, "`{`")?;
            body = if self.peek() == &Token::RightBrace {
                Some(crate::ast::Syntax::empty(self.offset()))
            } else {
                Some(self.expr()?)
            };
            self.expect(&Token::RightBrace, "`}`")?;
        }
        self.close_bindings(parameter_mark, self.last_end());
        prolog.functions.push(FunctionDeclaration {
            name,
            parameters,
            result_type,
            body,
            external,
            annotations,
        });
        Ok(())
    }

    fn set_once<T>(
        slot: &mut Option<T>,
        value: T,
        code: XQueryErrorCode,
        what: &str,
    ) -> XQueryResult<()> {
        if slot.is_some() {
            return Err(XQueryError::new(
                code,
                format!("`{what}` is declared more than once"),
            ));
        }
        *slot = Some(value);
        Ok(())
    }
}

/// Whitespace-normalizes a `URILiteral`, as the `xs:anyURI` whitespace facet prescribes.
fn collapse(uri: &str) -> String {
    uri.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Rejects a default element or function namespace that names one of the two namespaces
/// [XQuery 3.1 §4.2](https://www.w3.org/TR/xquery-31/#id-namespace-declaration) reserves.
fn check_default_namespace(uri: &str) -> XQueryResult<()> {
    if uri == oxixml_qname::XML_NAMESPACE || uri == crate::compile::XMLNS_NAMESPACE {
        return Err(XQueryError::new(
            XQueryErrorCode::Xqst0070,
            format!("`{uri}` may not be a default namespace"),
        ));
    }
    Ok(())
}

/// Whether `name` matches the XML `EncName` production.
fn is_encoding_name(name: &str) -> bool {
    let mut characters = name.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    if !first.is_ascii_alphabetic() {
        return false;
    }
    characters.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Whether `uri` is in the lexical space of `xs:anyURI`.
///
/// The only thing RFC 3986 rules out that a `URILiteral` can otherwise contain is a malformed
/// percent-escape: a `%` must be followed by two hexadecimal digits.
fn is_uri_lexical(uri: &str) -> bool {
    let mut characters = uri.chars();
    while let Some(character) = characters.next() {
        if character != '%' {
            continue;
        }
        let (Some(high), Some(low)) = (characters.next(), characters.next()) else {
            return false;
        };
        if !high.is_ascii_hexdigit() || !low.is_ascii_hexdigit() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(source: &str) -> XQueryResult<Module> {
        let mut parser = Parser::new(source)?;
        parser.parse_module()
    }

    #[test]
    fn a_bare_expression_is_a_main_module() -> XQueryResult<()> {
        let parsed = module("1 + 1")?;
        assert!(parsed.module_namespace.is_none());
        assert!(parsed.body.is_some());
        Ok(())
    }

    #[test]
    fn the_version_declaration_is_checked() -> XQueryResult<()> {
        let parsed = module(r#"xquery version "3.1"; 1"#)?;
        assert_eq!(parsed.prolog.version.as_deref(), Some("3.1"));
        let failure = module(r#"xquery version "9.9"; 1"#).expect_err("unsupported version");
        assert!(failure.is(XQueryErrorCode::Xqst0031));
        Ok(())
    }

    #[test]
    fn setters_are_recorded() -> XQueryResult<()> {
        let parsed = module(
            r#"declare boundary-space preserve;
               declare construction strip;
               declare ordering unordered;
               declare default order empty greatest;
               declare copy-namespaces no-preserve, no-inherit;
               declare base-uri "urn:base";
               declare default element namespace "urn:e";
               declare namespace p = "urn:p";
               1"#,
        )?;
        let prolog = parsed.prolog;
        assert_eq!(prolog.boundary_space, Some(BoundarySpace::Preserve));
        assert_eq!(prolog.construction, Some(ConstructionMode::Strip));
        assert_eq!(prolog.ordering, Some(OrderingMode::Unordered));
        assert_eq!(prolog.empty_order, Some(EmptyOrder::Greatest));
        assert_eq!(
            prolog.copy_namespaces,
            Some(CopyNamespaces {
                preserve: false,
                inherit: false
            })
        );
        assert_eq!(prolog.base_uri.as_deref(), Some("urn:base"));
        assert_eq!(prolog.default_element_namespace.as_deref(), Some("urn:e"));
        assert_eq!(prolog.namespaces.len(), 1);
        Ok(())
    }

    #[test]
    fn a_repeated_setter_is_a_static_error() {
        let failure = module("declare ordering ordered; declare ordering unordered; 1")
            .expect_err("declared twice");
        assert!(failure.is(XQueryErrorCode::Xqst0065));
    }

    #[test]
    fn variables_and_functions_are_collected() -> XQueryResult<()> {
        let parsed = module(
            r#"declare variable $x as xs:integer := 1;
               declare variable $y external;
               declare %private function local:f($a as xs:integer) as xs:integer { $a + $x };
               declare function local:g() external;
               local:f($y)"#,
        )?;
        assert_eq!(parsed.prolog.variables.len(), 2);
        assert!(parsed.prolog.variables[1].external);
        assert_eq!(parsed.prolog.functions.len(), 2);
        assert_eq!(parsed.prolog.functions[0].annotations, vec!["private"]);
        assert!(parsed.prolog.functions[1].external);
        Ok(())
    }

    #[test]
    fn a_library_module_has_no_body() -> XQueryResult<()> {
        let parsed = module(
            r#"module namespace m = "urn:m";
               declare function m:f() { 1 };"#,
        )?;
        assert_eq!(
            parsed.module_namespace,
            Some(("m".to_owned(), "urn:m".to_owned()))
        );
        assert!(parsed.body.is_none());
        Ok(())
    }

    #[test]
    fn imports_are_recorded() -> XQueryResult<()> {
        let parsed = module(
            r#"import module namespace m = "urn:m" at "m.xq", "m2.xq";
               import schema namespace s = "urn:s";
               1"#,
        )?;
        assert_eq!(parsed.prolog.module_imports.len(), 1);
        assert_eq!(parsed.prolog.module_imports[0].locations.len(), 2);
        assert_eq!(parsed.prolog.schema_imports.len(), 1);
        Ok(())
    }

    #[test]
    fn decimal_formats_are_recorded() -> XQueryResult<()> {
        let parsed = module(
            r#"declare decimal-format m:euro decimal-separator = "," grouping-separator = ".";
               declare default decimal-format NaN = "n/a";
               1"#,
        )?;
        assert_eq!(parsed.prolog.decimal_formats.len(), 2);
        assert_eq!(parsed.prolog.decimal_formats[0].properties.len(), 2);
        assert!(parsed.prolog.decimal_formats[1].name.is_none());
        Ok(())
    }

    #[test]
    fn a_context_item_declaration_is_recorded() -> XQueryResult<()> {
        let parsed = module("declare context item as xs:integer := 7; . + 1")?;
        let declaration = parsed.prolog.context_item.expect("a declaration");
        assert_eq!(declaration.declared_type.as_deref(), Some("xs:integer"));
        assert!(!declaration.external);
        Ok(())
    }
}
