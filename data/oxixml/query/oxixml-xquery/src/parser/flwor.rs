//! FLWOR, quantified, `switch`, `typeswitch` and `try`/`catch` expressions.
//!
//! Every one of these begins with a name that XQuery does *not* reserve, so each entry point
//! looks far enough ahead to be sure — `for` must be followed by `$`, `tumbling` or `sliding`,
//! `switch` and `typeswitch` by a parenthesized operand and then a `case`, `try` by a block and
//! then a `catch` — and rewinds when the lookahead fails, leaving the name to be parsed as an
//! ordinary function call or name test.

use super::Parser;
use crate::ast::{
    BindingKind, CatchClause, CatchTest, Flwor, FlworClause, ForBinding, GroupingSpec, LetBinding,
    OrderSpec, SimpleBinding, SwitchExpr, Syntax, SyntaxKind, TryExpr, TypeCase, TypeSwitchExpr,
    WindowClause, WindowCondition, unexpected,
};
use crate::error::{XQueryError, XQueryErrorCode, XQueryResult};
use crate::lexer::Token;

impl Parser<'_> {
    /// A FLWOR expression, or `None` when `for`/`let` is not starting one.
    pub(crate) fn try_flwor(&mut self) -> XQueryResult<Option<Syntax>> {
        let state = self.state();
        self.advance()?;
        let starts = self.peek() == &Token::Dollar
            || self.at_keyword("tumbling")
            || self.at_keyword("sliding");
        self.restore(state);
        if !starts {
            return Ok(None);
        }
        let start = self.offset();
        // Every variable a FLWOR clause binds is in scope over the rest of that FLWOR
        // expression, and over nothing outside it (XQuery 3.1 §3.12).
        let flwor_mark = self.binding_mark();
        let mut clauses = Vec::new();
        loop {
            if self.at_keyword("for") {
                clauses.push(self.for_clause()?);
            } else if self.at_keyword("let") {
                clauses.push(self.let_clause()?);
            } else if self.at_keyword("where") {
                self.advance()?;
                clauses.push(FlworClause::Where(self.expr_single()?));
            } else if self.at_keyword("group") && self.next_is_keyword("by")? {
                self.advance()?;
                self.advance()?;
                clauses.push(FlworClause::GroupBy(self.grouping_specs()?));
            } else if self.at_keyword("stable") && self.next_is_keyword("order")? {
                self.advance()?;
                self.expect_keyword("order")?;
                self.expect_keyword("by")?;
                clauses.push(FlworClause::OrderBy(self.order_specs()?));
            } else if self.at_keyword("order") && self.next_is_keyword("by")? {
                self.advance()?;
                self.advance()?;
                clauses.push(FlworClause::OrderBy(self.order_specs()?));
            } else if self.at_keyword("count") && self.next_is(&Token::Dollar)? {
                self.advance()?;
                self.expect(&Token::Dollar, "`$`")?;
                let name = self.binding_name("a variable name")?;
                clauses.push(FlworClause::Count(name));
            } else {
                break;
            }
        }
        self.expect_keyword("return")?;
        let result = self.expr_single()?;
        let end = self.last_end();
        self.close_bindings(flwor_mark, end);

        // A single `for` or `let` clause with plain bindings is exactly XPath's own `for` or
        // `let` expression, so it stays delegatable.
        if clauses.len() == 1 {
            if let Some(simple) = Self::as_simple_binding(&clauses[0]) {
                let (kind, bindings) = simple;
                return Ok(Some(Syntax::new(
                    start,
                    end,
                    SyntaxKind::Binding {
                        kind,
                        bindings,
                        body: Box::new(result),
                    },
                )));
            }
        }
        Ok(Some(Syntax::native(
            start,
            end,
            SyntaxKind::Flwor(Box::new(Flwor { clauses, result })),
        )))
    }

    fn as_simple_binding(clause: &FlworClause) -> Option<(BindingKind, Vec<SimpleBinding>)> {
        match clause {
            FlworClause::For(bindings) => {
                let plain = bindings.iter().all(|binding| {
                    binding.declared_type.is_none()
                        && !binding.allowing_empty
                        && binding.positional.is_none()
                });
                plain.then(|| {
                    (
                        BindingKind::For,
                        bindings
                            .iter()
                            .map(|binding| SimpleBinding {
                                name: binding.name.clone(),
                                declared_type: None,
                                value: binding.value.clone(),
                            })
                            .collect(),
                    )
                })
            }
            FlworClause::Let(bindings) => {
                let plain = bindings
                    .iter()
                    .all(|binding| binding.declared_type.is_none());
                plain.then(|| {
                    (
                        BindingKind::Let,
                        bindings
                            .iter()
                            .map(|binding| SimpleBinding {
                                name: binding.name.clone(),
                                declared_type: None,
                                value: binding.value.clone(),
                            })
                            .collect(),
                    )
                })
            }
            _ => None,
        }
    }

    fn for_clause(&mut self) -> XQueryResult<FlworClause> {
        self.expect_keyword("for")?;
        if self.at_keyword("tumbling") || self.at_keyword("sliding") {
            return Ok(FlworClause::Window(Box::new(self.window_clause()?)));
        }
        let mut bindings = Vec::new();
        loop {
            self.expect(&Token::Dollar, "`$`")?;
            let name = self.binding_name("a variable name")?;
            let declared_type = self.optional_type_declaration()?;
            let allowing_empty = if self.at_keyword("allowing") && self.next_is_keyword("empty")? {
                self.advance()?;
                self.advance()?;
                true
            } else {
                false
            };
            let positional = if self.at_keyword("at") {
                self.advance()?;
                self.expect(&Token::Dollar, "`$`")?;
                let at = self.offset();
                let positional = self.binding_name("a positional variable name")?;
                // §3.12.2: a `for` binding and its positional variable must have distinct names.
                if positional == name {
                    return Err(XQueryError::new(
                        XQueryErrorCode::Xqst0089,
                        format!("`${name}` is also the positional variable of its own binding"),
                    )
                    .at(at));
                }
                Some(positional)
            } else {
                None
            };
            self.expect_keyword("in")?;
            let value = self.expr_single()?;
            bindings.push(ForBinding {
                name,
                declared_type,
                allowing_empty,
                positional,
                value,
            });
            if !self.eat(&Token::Comma)? {
                break;
            }
        }
        Ok(FlworClause::For(bindings))
    }

    fn let_clause(&mut self) -> XQueryResult<FlworClause> {
        self.expect_keyword("let")?;
        let mut bindings = Vec::new();
        loop {
            self.expect(&Token::Dollar, "`$`")?;
            let name = self.binding_name("a variable name")?;
            let declared_type = self.optional_type_declaration()?;
            self.expect(&Token::Assign, "`:=`")?;
            let value = self.expr_single()?;
            bindings.push(LetBinding {
                name,
                declared_type,
                value,
            });
            if !self.eat(&Token::Comma)? {
                break;
            }
        }
        Ok(FlworClause::Let(bindings))
    }

    fn window_clause(&mut self) -> XQueryResult<WindowClause> {
        let tumbling = self.at_keyword("tumbling");
        self.advance()?;
        self.expect_keyword("window")?;
        self.expect(&Token::Dollar, "`$`")?;
        let name = self.binding_name("a window variable name")?;
        let declared_type = self.optional_type_declaration()?;
        self.expect_keyword("in")?;
        let value = self.expr_single()?;
        self.expect_keyword("start")?;
        let start = self.window_condition()?;
        let mut only = false;
        let mut end = None;
        if self.at_keyword("only") && self.next_is_keyword("end")? {
            self.advance()?;
            self.advance()?;
            only = true;
            end = Some(self.window_condition()?);
        } else if self.at_keyword("end") {
            self.advance()?;
            end = Some(self.window_condition()?);
        }
        Ok(WindowClause {
            tumbling,
            name,
            declared_type,
            value,
            start,
            end,
            only,
        })
    }

    fn window_condition(&mut self) -> XQueryResult<WindowCondition> {
        let mut condition = WindowCondition::default();
        if self.peek() == &Token::Dollar {
            self.advance()?;
            condition.current = Some(self.binding_name("a variable name")?);
        }
        if self.at_keyword("at") {
            self.advance()?;
            self.expect(&Token::Dollar, "`$`")?;
            condition.positional = Some(self.binding_name("a variable name")?);
        }
        if self.at_keyword("previous") {
            self.advance()?;
            self.expect(&Token::Dollar, "`$`")?;
            condition.previous = Some(self.binding_name("a variable name")?);
        }
        if self.at_keyword("next") {
            self.advance()?;
            self.expect(&Token::Dollar, "`$`")?;
            condition.next = Some(self.binding_name("a variable name")?);
        }
        if self.at_keyword("when") {
            self.advance()?;
            condition.when = Some(self.expr_single()?);
        }
        Ok(condition)
    }

    fn grouping_specs(&mut self) -> XQueryResult<Vec<GroupingSpec>> {
        let mut specs = Vec::new();
        loop {
            self.expect(&Token::Dollar, "`$`")?;
            let name = self.binding_name("a grouping variable name")?;
            let declared_type = self.optional_type_declaration()?;
            // `GroupingSpec ::= GroupingVariable (TypeDeclaration? ":=" ExprSingle)?
            //  ("collation" URILiteral)?` — a type declaration only exists inside the group that
            //  the `:=` closes, so one without a binding expression does not parse at all.
            let value = if self.eat(&Token::Assign)? {
                Some(self.expr_single()?)
            } else {
                if declared_type.is_some() {
                    return Err(XQueryError::syntax(
                        "a `group by` type declaration must be followed by `:=`",
                        self.offset(),
                    ));
                }
                None
            };
            let collation = if self.at_keyword("collation") {
                self.advance()?;
                Some(self.uri_literal("a collation URI")?)
            } else {
                None
            };
            specs.push(GroupingSpec {
                name,
                declared_type,
                value,
                collation,
            });
            if !self.eat(&Token::Comma)? {
                break;
            }
        }
        Ok(specs)
    }

    fn order_specs(&mut self) -> XQueryResult<Vec<OrderSpec>> {
        let mut specs = Vec::new();
        loop {
            let key = self.expr_single()?;
            let mut ascending = true;
            if self.at_keyword("ascending") {
                self.advance()?;
            } else if self.at_keyword("descending") {
                self.advance()?;
                ascending = false;
            }
            let mut empty_greatest = None;
            if self.at_keyword("empty") {
                self.advance()?;
                if self.at_keyword("greatest") {
                    self.advance()?;
                    empty_greatest = Some(true);
                } else {
                    self.expect_keyword("least")?;
                    empty_greatest = Some(false);
                }
            }
            let collation = if self.at_keyword("collation") {
                self.advance()?;
                Some(self.uri_literal("a collation URI")?)
            } else {
                None
            };
            specs.push(OrderSpec {
                key,
                ascending,
                empty_greatest,
                collation,
            });
            if !self.eat(&Token::Comma)? {
                break;
            }
        }
        Ok(specs)
    }

    /// `some`/`every`, or `None` when the name is not starting one.
    pub(crate) fn try_quantified(&mut self) -> XQueryResult<Option<Syntax>> {
        if !self.next_is(&Token::Dollar)? {
            return Ok(None);
        }
        let start = self.offset();
        let every = self.at_keyword("every");
        self.advance()?;
        // A quantified expression's variables are in scope over the rest of the expression only.
        let quantified_mark = self.binding_mark();
        let mut bindings = Vec::new();
        let mut typed = false;
        loop {
            self.expect(&Token::Dollar, "`$`")?;
            let name = self.binding_name("a variable name")?;
            let declared_type = self.optional_type_declaration()?;
            typed |= declared_type.is_some();
            self.expect_keyword("in")?;
            let value = self.expr_single()?;
            bindings.push(SimpleBinding {
                name,
                declared_type,
                value,
            });
            if !self.eat(&Token::Comma)? {
                break;
            }
        }
        self.expect_keyword("satisfies")?;
        let body = self.expr_single()?;
        let end = self.last_end();
        self.close_bindings(quantified_mark, end);
        let kind = SyntaxKind::Binding {
            kind: if every {
                BindingKind::Every
            } else {
                BindingKind::Some
            },
            bindings,
            body: Box::new(body),
        };
        Ok(Some(if typed {
            Syntax::native(start, end, kind)
        } else {
            Syntax::new(start, end, kind)
        }))
    }

    /// `switch (E) case … default return …`, or `None` when `switch` is an ordinary name.
    pub(crate) fn try_switch(&mut self) -> XQueryResult<Option<Syntax>> {
        let state = self.state();
        let start = self.offset();
        self.advance()?;
        self.expect(&Token::LeftParen, "`(`")?;
        let operand = match self.expr() {
            Ok(operand) => operand,
            Err(_) => {
                self.restore(state);
                return Ok(None);
            }
        };
        if self.peek() != &Token::RightParen {
            self.restore(state);
            return Ok(None);
        }
        self.advance()?;
        if !self.at_keyword("case") {
            self.restore(state);
            return Ok(None);
        }
        let mut cases = Vec::new();
        while self.at_keyword("case") {
            let mut operands = Vec::new();
            while self.eat_keyword("case")? {
                operands.push(self.expr_single()?);
            }
            self.expect_keyword("return")?;
            cases.push((operands, self.expr_single()?));
        }
        self.expect_keyword("default")?;
        self.expect_keyword("return")?;
        let default_body = self.expr_single()?;
        let end = self.last_end();
        Ok(Some(Syntax::native(
            start,
            end,
            SyntaxKind::Switch(Box::new(SwitchExpr {
                operand,
                cases,
                default_body,
            })),
        )))
    }

    /// `typeswitch (E) case … default …`, or `None` when `typeswitch` is an ordinary name.
    pub(crate) fn try_typeswitch(&mut self) -> XQueryResult<Option<Syntax>> {
        let state = self.state();
        let start = self.offset();
        self.advance()?;
        self.expect(&Token::LeftParen, "`(`")?;
        let operand = match self.expr() {
            Ok(operand) => operand,
            Err(_) => {
                self.restore(state);
                return Ok(None);
            }
        };
        if self.peek() != &Token::RightParen {
            self.restore(state);
            return Ok(None);
        }
        self.advance()?;
        if !self.at_keyword("case") {
            self.restore(state);
            return Ok(None);
        }
        let mut cases = Vec::new();
        while self.eat_keyword("case")? {
            // A case clause's variable is in scope over *that* clause's return expression only
            // (XQuery 3.1 §3.13.2).
            let case_mark = self.binding_mark();
            let mut variable = None;
            if self.peek() == &Token::Dollar {
                self.advance()?;
                variable = Some(self.binding_name("a variable name")?);
                self.expect_keyword("as")?;
            }
            let mut types = vec![self.sequence_type_source()?];
            while self.eat(&Token::Pipe)? {
                types.push(self.sequence_type_source()?);
            }
            self.expect_keyword("return")?;
            let body = self.expr_single()?;
            self.close_bindings(case_mark, self.last_end());
            cases.push(TypeCase {
                variable,
                types,
                body,
            });
        }
        self.expect_keyword("default")?;
        let default_mark = self.binding_mark();
        let mut default_variable = None;
        if self.peek() == &Token::Dollar {
            self.advance()?;
            default_variable = Some(self.binding_name("a variable name")?);
        }
        self.expect_keyword("return")?;
        let default_body = self.expr_single()?;
        self.close_bindings(default_mark, self.last_end());
        let end = self.last_end();
        Ok(Some(Syntax::native(
            start,
            end,
            SyntaxKind::TypeSwitch(Box::new(TypeSwitchExpr {
                operand,
                cases,
                default_variable,
                default_body,
            })),
        )))
    }

    /// `try { … } catch … { … }`, or `None` when `try` is an ordinary name.
    pub(crate) fn try_catch(&mut self) -> XQueryResult<Option<Syntax>> {
        let state = self.state();
        let start = self.offset();
        self.advance()?;
        self.expect(&Token::LeftBrace, "`{`")?;
        let body = if self.peek() == &Token::RightBrace {
            Syntax::empty(self.offset())
        } else {
            match self.expr() {
                Ok(body) => body,
                Err(_) => {
                    self.restore(state);
                    return Ok(None);
                }
            }
        };
        if self.peek() != &Token::RightBrace {
            self.restore(state);
            return Ok(None);
        }
        self.advance()?;
        if !self.at_keyword("catch") {
            self.restore(state);
            return Ok(None);
        }
        let mut catches = Vec::new();
        while self.eat_keyword("catch")? {
            let mut tests = vec![self.catch_test()?];
            while self.eat(&Token::Pipe)? {
                tests.push(self.catch_test()?);
            }
            self.expect(&Token::LeftBrace, "`{`")?;
            let handler = if self.peek() == &Token::RightBrace {
                Syntax::empty(self.offset())
            } else {
                self.expr()?
            };
            self.expect(&Token::RightBrace, "`}`")?;
            catches.push(CatchClause {
                tests,
                body: handler,
            });
        }
        let end = self.last_end();
        Ok(Some(Syntax::native(
            start,
            end,
            SyntaxKind::Try(Box::new(TryExpr { body, catches })),
        )))
    }

    fn catch_test(&mut self) -> XQueryResult<CatchTest> {
        match self.peek().clone() {
            Token::Star => {
                self.advance()?;
                if self.peek() == &Token::Colon {
                    self.advance()?;
                    let local = self.ncname("a local name after `*:`")?;
                    return Ok(CatchTest::Local(local));
                }
                Ok(CatchTest::Any)
            }
            Token::Name(name) => {
                self.advance()?;
                if self.peek() == &Token::Colon {
                    self.advance()?;
                    self.expect(&Token::Star, "`*` after `prefix:`")?;
                    return Ok(CatchTest::Namespace(Some(name)));
                }
                Ok(CatchTest::Name(name))
            }
            Token::BracedName { uri, local } => {
                self.advance()?;
                Ok(CatchTest::Name(format!("Q{{{uri}}}{local}")))
            }
            Token::BracedWildcard { uri } => {
                self.advance()?;
                Ok(CatchTest::Namespace(Some(uri)))
            }
            other => Err(self.error(unexpected(&other, "an error name test"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> XQueryResult<Syntax> {
        let mut parser = Parser::new(source)?;
        let parsed = parser.expr()?;
        assert_eq!(parser.peek(), &Token::Eof, "trailing input in `{source}`");
        Ok(parsed)
    }

    #[test]
    fn a_single_clause_flwor_stays_an_xpath_binding() -> XQueryResult<()> {
        let parsed = parse("for $x in (1, 2) return $x")?;
        assert!(matches!(parsed.kind, SyntaxKind::Binding { .. }));
        assert!(!parsed.xquery_only);
        Ok(())
    }

    #[test]
    fn every_extra_clause_makes_it_a_flwor() -> XQueryResult<()> {
        for source in [
            "for $x in (1, 2) let $y := $x return $y",
            "for $x in (1, 2) where $x > 1 return $x",
            "for $x in (1, 2) order by $x descending return $x",
            "for $x in (1, 2) count $c return $c",
            "for $x in (1, 2) group by $k := $x return $k",
            "for $x as xs:integer in (1, 2) return $x",
            "for $x at $i in (1, 2) return $i",
            "for $x allowing empty in () return 0",
            "for tumbling window $w in (1, 2, 3) start $s when true() return $w",
            "for sliding window $w in (1 to 5) start $s at $i when true() only end $e when true() return $w",
            "let $x := 1 stable order by $x return $x",
        ] {
            let parsed = parse(source)?;
            assert!(
                matches!(parsed.kind, SyntaxKind::Flwor(_)),
                "`{source}` should be a FLWOR"
            );
            assert!(parsed.xquery_only, "`{source}`");
        }
        Ok(())
    }

    #[test]
    fn typeswitch_and_switch_parse() -> XQueryResult<()> {
        let parsed = parse(
            "typeswitch (1) case $n as xs:integer | xs:decimal return 'number' default $d return 'other'",
        )?;
        match &parsed.kind {
            SyntaxKind::TypeSwitch(expr) => {
                assert_eq!(expr.cases.len(), 1);
                assert_eq!(expr.cases[0].types.len(), 2);
                assert_eq!(expr.default_variable.as_deref(), Some("d"));
            }
            other => panic!("expected a typeswitch, got {other:?}"),
        }
        let parsed = parse("switch (1) case 1 case 2 return 'low' default return 'high'")?;
        match &parsed.kind {
            SyntaxKind::Switch(expr) => assert_eq!(expr.cases[0].0.len(), 2),
            other => panic!("expected a switch, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn try_catch_parses_its_name_tests() -> XQueryResult<()> {
        let parsed = parse("try { 1 } catch err:FOAR0001 | *:XPTY0004 | p:* | * { 2 }")?;
        match &parsed.kind {
            SyntaxKind::Try(expr) => {
                assert_eq!(expr.catches.len(), 1);
                assert_eq!(expr.catches[0].tests.len(), 4);
            }
            other => panic!("expected a try, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn a_name_that_only_looks_like_a_keyword_is_a_function_call() -> XQueryResult<()> {
        for source in ["count(1)", "for(1)"] {
            let parsed = parse(source)?;
            assert!(!parsed.xquery_only, "`{source}`");
        }
        Ok(())
    }

    #[test]
    fn a_typed_quantifier_needs_this_evaluator() -> XQueryResult<()> {
        let parsed = parse("some $x as xs:integer in (1, 2) satisfies $x > 1")?;
        assert!(parsed.xquery_only);
        Ok(())
    }
}
