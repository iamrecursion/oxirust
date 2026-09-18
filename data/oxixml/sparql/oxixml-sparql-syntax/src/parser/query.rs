//! Query-level productions: prologue, the four query forms, dataset and solution modifiers.

use super::build::{
    Selection, SelectionMember, SelectionOption, SelectionVariables, are_variables_bound, new_join,
    not_empty_fold,
};
use super::{Parser, R, alt};
use crate::algebra::{Expression, GraphPattern, OrderExpression, QueryDataset};
use crate::query::{AskQuery, ConstructQuery, DescribeQuery, Query, SelectQuery};
use crate::term::{NamedNodePattern, TriplePattern, Variable};
use oxixml_iri::Iri;
use oxixml_model::OxString;
use std::collections::HashSet;

impl Parser<'_> {
    // --- entry --------------------------------------------------------------

    pub(crate) fn query_unit(&mut self) -> R<Query> {
        self.ws();
        self.prologue()?;
        self.ws();
        let q = alt!(
            self,
            self.select_query(),
            self.construct_query(),
            self.describe_query(),
            self.ask_query(),
        )?;
        self.ws();
        Ok(q)
    }

    /// Matches the `VERSION` keyword of the SPARQL 1.2 prologue, and nothing else.
    fn try_version_keyword(&mut self) -> bool {
        let save = self.pos;
        if self.try_kw("VERSION") {
            true
        } else {
            self.pos = save;
            false
        }
    }

    pub(crate) fn prologue(&mut self) -> R<()> {
        loop {
            self.ws();
            if self.try_kw("BASE") {
                self.ws();
                let iri = self.iriref()?;
                let parsed = Iri::parse(OxString::new_owned(&iri));
                match parsed {
                    Ok(v) => self.base_iri = Some(v),
                    Err(_) => return self.fatal("Invalid BASE IRI"),
                }
            } else if self.try_kw("PREFIX") {
                self.ws();
                let ns = self.pname_ns()?;
                self.ws();
                let iri = self.iriref()?;
                self.prefixes.insert(ns, iri);
            } else if cfg!(feature = "sparql-12") && self.try_version_keyword() {
                #[cfg(feature = "sparql-12")]
                self.version_decl()?;
            } else {
                break;
            }
        }
        Ok(())
    }

    // --- SELECT -------------------------------------------------------------

    fn select_query(&mut self) -> R<Query> {
        let select = self.select_clause()?;
        self.ws();
        let dataset = self.dataset_clauses()?;
        self.ws();
        let where_pattern = self.where_clause()?;
        let modifiers = self.solution_modifiers()?;
        let pattern = self.build_select(select, where_pattern, modifiers)?;
        Ok(Query::Select(SelectQuery {
            dataset,
            pattern,
            base_iri: self.base_iri.clone(),
        }))
    }

    pub(crate) fn sub_select(&mut self) -> R<GraphPattern> {
        let select = self.select_clause()?;
        self.ws();
        let where_pattern = self.where_clause()?;
        let modifiers = self.solution_modifiers()?;
        self.build_select(select, where_pattern, modifiers)
    }

    fn select_clause(&mut self) -> R<Selection> {
        self.kw("SELECT")?;
        self.aggregates.push(Vec::new());
        self.ws();
        let option = if self.try_kw("DISTINCT") {
            SelectionOption::Distinct
        } else if self.try_kw("REDUCED") {
            SelectionOption::Reduced
        } else {
            SelectionOption::Default
        };
        self.ws();
        let variables = if self.try_str("*") {
            SelectionVariables::Star
        } else {
            let mut members = Vec::new();
            loop {
                let save = self.pos;
                match self.select_member() {
                    Ok(m) => {
                        members.push(m);
                        self.ws();
                    }
                    Err(e) if e.fatal => return Err(e),
                    Err(_) => {
                        self.pos = save;
                        break;
                    }
                }
            }
            if members.is_empty() {
                return self.mk_err("expected projection");
            }
            SelectionVariables::Explicit(members)
        };
        Ok(Selection { option, variables })
    }

    fn select_member(&mut self) -> R<SelectionMember> {
        if self.peek_char() == Some('(') {
            self.eat("(")?;
            self.ws();
            let e = self.expression()?;
            self.ws();
            self.kw("AS")?;
            self.ws();
            let v = self.var()?;
            self.ws();
            self.eat(")")?;
            Ok(SelectionMember::Expression(e, v))
        } else {
            let v = self.var()?;
            Ok(SelectionMember::Variable(v))
        }
    }

    // --- CONSTRUCT ----------------------------------------------------------

    fn construct_query(&mut self) -> R<Query> {
        self.kw("CONSTRUCT")?;
        self.ws();
        if self.peek_char() == Some('{') {
            // Blank node labels of a `CONSTRUCT` template live in their own scope: the very same
            // label may be reused by the `WHERE` clause, exactly like in `spargebra`.
            let outer_used = std::mem::take(&mut self.used_bnodes);
            let outer_current = std::mem::take(&mut self.currently_used_bnodes);
            let template = self.construct_template();
            self.used_bnodes = outer_used;
            self.currently_used_bnodes = outer_current;
            let template = template?;
            self.ws();
            let dataset = self.dataset_clauses()?;
            self.ws();
            let where_pattern = self.where_clause()?;
            let modifiers = self.solution_modifiers()?;
            let pattern = self.build_select(Selection::no_op(), where_pattern, modifiers)?;
            Ok(Query::Construct(ConstructQuery {
                template,
                dataset,
                pattern,
                base_iri: self.base_iri.clone(),
            }))
        } else {
            let dataset = self.dataset_clauses()?;
            self.ws();
            self.kw("WHERE")?;
            self.ws();
            self.eat("{")?;
            self.ws();
            let mut template = Vec::new();
            let save = self.pos;
            if self.triples_template(&mut template).is_err() {
                self.pos = save;
                template.clear();
            }
            self.ws();
            self.eat("}")?;
            let modifiers = self.solution_modifiers()?;
            let pattern = self.build_select(
                Selection::no_op(),
                GraphPattern::Bgp {
                    patterns: template.clone(),
                },
                modifiers,
            )?;
            Ok(Query::Construct(ConstructQuery {
                template,
                dataset,
                pattern,
                base_iri: self.base_iri.clone(),
            }))
        }
    }

    fn construct_template(&mut self) -> R<Vec<TriplePattern>> {
        self.eat("{")?;
        self.ws();
        let mut patterns = Vec::new();
        loop {
            let save = self.pos;
            // `ConstructTemplate ::= '{' TriplesTemplate? '}'` makes the whole template optional,
            // so even the *first* iteration backtracks to an empty template on an ordinary,
            // non-fatal parse failure here — unlike `triples_template`'s own loop, whose leading
            // statement is mandatory. A *fatal* error (for example `enter_recursion` rejecting a
            // template nested too deeply) is a different matter either way: it must still abort
            // the parse instead of being silently reinterpreted as "this template is empty".
            match self.triples_same_subject(&mut patterns) {
                Ok(()) => {}
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    break;
                }
            }
            self.ws();
            if self.try_str(".") {
                self.ws();
            } else {
                break;
            }
        }
        self.ws();
        self.eat("}")?;
        Ok(patterns)
    }

    // --- DESCRIBE -----------------------------------------------------------

    fn describe_query(&mut self) -> R<Query> {
        self.kw("DESCRIBE")?;
        self.aggregates.push(Vec::new());
        self.ws();
        let selection = if self.try_str("*") {
            Selection::no_op()
        } else {
            let mut items = Vec::new();
            loop {
                let save = self.pos;
                match self.var_or_iri() {
                    Ok(item) => {
                        items.push(item);
                        self.ws();
                    }
                    Err(e) if e.fatal => return Err(e),
                    Err(_) => {
                        self.pos = save;
                        break;
                    }
                }
            }
            if items.is_empty() {
                return self.mk_err("expected DESCRIBE target");
            }
            let members = items
                .into_iter()
                .map(|item| match item {
                    NamedNodePattern::NamedNode(n) => {
                        SelectionMember::Expression(n.into(), self.fresh_var())
                    }
                    NamedNodePattern::Variable(v) => SelectionMember::Variable(v),
                })
                .collect();
            Selection {
                option: SelectionOption::Default,
                variables: SelectionVariables::Explicit(members),
            }
        };
        self.ws();
        let dataset = self.dataset_clauses()?;
        self.ws();
        let where_pattern = self.opt_where_clause()?;
        let modifiers = self.solution_modifiers()?;
        let pattern = self.build_select(selection, where_pattern, modifiers)?;
        Ok(Query::Describe(DescribeQuery {
            dataset,
            pattern,
            base_iri: self.base_iri.clone(),
        }))
    }

    // --- ASK ----------------------------------------------------------------

    fn ask_query(&mut self) -> R<Query> {
        self.kw("ASK")?;
        self.ws();
        let dataset = self.dataset_clauses()?;
        self.ws();
        let where_pattern = self.where_clause()?;
        let modifiers = self.solution_modifiers()?;
        let pattern = self.build_select(Selection::no_op(), where_pattern, modifiers)?;
        Ok(Query::Ask(AskQuery {
            dataset,
            pattern,
            base_iri: self.base_iri.clone(),
        }))
    }

    // --- dataset ------------------------------------------------------------

    fn dataset_clauses(&mut self) -> R<Option<QueryDataset>> {
        let mut default = Vec::new();
        let mut named = Vec::new();
        let mut any = false;
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_kw("FROM") {
                self.pos = save;
                break;
            }
            self.ws();
            if self.try_kw("NAMED") {
                self.ws();
                named.push(self.iri()?);
            } else {
                default.push(self.iri()?);
            }
            any = true;
        }
        if any {
            Ok(Some(QueryDataset {
                default,
                named: Some(named),
            }))
        } else {
            Ok(None)
        }
    }

    // --- WHERE --------------------------------------------------------------

    fn where_clause(&mut self) -> R<GraphPattern> {
        self.ws();
        self.try_kw("WHERE");
        self.ws();
        self.group_graph_pattern()
    }

    fn opt_where_clause(&mut self) -> R<GraphPattern> {
        self.ws();
        let save = self.pos;
        self.try_kw("WHERE");
        self.ws();
        if self.peek_char() == Some('{') {
            self.group_graph_pattern()
        } else {
            self.pos = save;
            Ok(GraphPattern::default())
        }
    }

    // --- solution modifiers -------------------------------------------------

    pub(crate) fn solution_modifiers(&mut self) -> R<Modifiers> {
        self.ws();
        let group = self.group_clause()?;
        self.ws();
        let having = self.having_clause()?;
        self.ws();
        let order_by = self.order_clause()?;
        self.ws();
        let offset_limit = self.limit_offset_clauses()?;
        self.ws();
        let values = self.values_clause()?;
        Ok(Modifiers {
            group,
            having,
            order_by,
            offset_limit,
            values,
        })
    }

    #[allow(clippy::type_complexity)]
    fn group_clause(&mut self) -> R<Option<(Vec<Variable>, Vec<(Expression, Variable)>)>> {
        let save = self.pos;
        if !self.try_kw("GROUP") {
            self.pos = save;
            return Ok(None);
        }
        self.ws();
        self.kw("BY")?;
        self.ws();
        let mut projections: Vec<(Expression, Variable)> = Vec::new();
        let mut clauses = Vec::new();
        let mut count = 0;
        loop {
            let save = self.pos;
            match self.group_condition() {
                Ok((e, vo)) => {
                    count += 1;
                    // `(Expr AS Var)` always binds `Var`, even when `Expr` is a bare variable:
                    // `GROUP BY (?x AS ?y)` groups on `?y`, see SPARQL 1.1 §18.2.4.1.
                    match (vo, e) {
                        (Some(v), e) => {
                            projections.push((e, v.clone()));
                            clauses.push(v);
                        }
                        (None, Expression::Variable(v)) => clauses.push(v),
                        (None, e) => {
                            let v = self.fresh_var();
                            projections.push((e, v.clone()));
                            clauses.push(v);
                        }
                    }
                    self.ws();
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    break;
                }
            }
        }
        if count == 0 {
            return self.mk_err("expected group condition");
        }
        Ok(Some((clauses, projections)))
    }

    fn group_condition(&mut self) -> R<(Expression, Option<Variable>)> {
        // `GROUP BY` expressions are evaluated below the `Group` node, so they may not contain
        // aggregates.
        let saved = std::mem::replace(&mut self.aggregates_allowed, false);
        let r = self.group_condition_inner();
        self.aggregates_allowed = saved;
        r
    }

    fn group_condition_inner(&mut self) -> R<(Expression, Option<Variable>)> {
        if self.peek_char() == Some('(') {
            self.eat("(")?;
            self.ws();
            let e = self.expression()?;
            self.ws();
            let v = if self.try_kw("AS") {
                self.ws();
                let v = self.var()?;
                self.ws();
                Some(v)
            } else {
                None
            };
            self.eat(")")?;
            return Ok((e, v));
        }
        alt!(
            self,
            self.builtin_call().map(|e| (e, None)),
            self.function_call().map(|e| (e, None)),
            self.var().map(|v| (v.into(), None)),
        )
    }

    /// `HavingClause ::= 'HAVING' HavingCondition+`
    ///
    /// A flat list of `n` conditions folds into a tree `n - 1` levels deep, so — exactly like a
    /// genuinely nested construct — every condition after the first charges
    /// [`Parser::enter_recursion`] before it is parsed. See [`Parser::leave_recursion_by`] for why
    /// the charge happens before, not after, and why it is released once for the whole clause
    /// rather than per condition.
    fn having_clause(&mut self) -> R<Option<Expression>> {
        let save = self.pos;
        if !self.try_kw("HAVING") {
            self.pos = save;
            return Ok(None);
        }
        self.ws();
        let mut conditions = Vec::new();
        let mut charged = 0_u32;
        loop {
            let save = self.pos;
            let is_first = conditions.is_empty();
            if !is_first {
                self.enter_recursion()?;
            }
            match self.constraint() {
                Ok(c) => {
                    if !is_first {
                        charged += 1;
                    }
                    conditions.push(c);
                    self.ws();
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    if !is_first {
                        self.leave_recursion();
                    }
                    self.pos = save;
                    break;
                }
            }
        }
        let result = not_empty_fold(conditions.into_iter(), |a, b| {
            Expression::And(Box::new(a), Box::new(b))
        });
        self.leave_recursion_by(charged);
        match result {
            Ok(e) => Ok(Some(e)),
            Err(_) => self.mk_err("expected having condition"),
        }
    }

    fn order_clause(&mut self) -> R<Option<Vec<OrderExpression>>> {
        let save = self.pos;
        if !self.try_kw("ORDER") {
            self.pos = save;
            return Ok(None);
        }
        self.ws();
        self.kw("BY")?;
        self.ws();
        let mut conditions = Vec::new();
        loop {
            let save = self.pos;
            match self.order_condition() {
                Ok(c) => {
                    conditions.push(c);
                    self.ws();
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    break;
                }
            }
        }
        if conditions.is_empty() {
            return self.mk_err("expected order condition");
        }
        Ok(Some(conditions))
    }

    fn order_condition(&mut self) -> R<OrderExpression> {
        if self.try_kw("ASC") {
            self.ws();
            let e = self.bracketed_expression()?;
            Ok(OrderExpression::Asc(e))
        } else if self.try_kw("DESC") {
            self.ws();
            let e = self.bracketed_expression()?;
            Ok(OrderExpression::Desc(e))
        } else {
            alt!(
                self,
                self.constraint().map(OrderExpression::Asc),
                self.var()
                    .map(|v| OrderExpression::Asc(Expression::from(v))),
            )
        }
    }

    fn limit_offset_clauses(&mut self) -> R<Option<(usize, Option<usize>)>> {
        let save = self.pos;
        if self.try_kw("LIMIT") {
            self.ws();
            let limit = self.read_uint()?;
            self.ws();
            let offset = if self.try_kw("OFFSET") {
                self.ws();
                self.read_uint()?
            } else {
                0
            };
            Ok(Some((offset, Some(limit))))
        } else if self.try_kw("OFFSET") {
            self.ws();
            let offset = self.read_uint()?;
            self.ws();
            let limit = if self.try_kw("LIMIT") {
                self.ws();
                Some(self.read_uint()?)
            } else {
                None
            };
            Ok(Some((offset, limit)))
        } else {
            self.pos = save;
            Ok(None)
        }
    }

    fn read_uint(&mut self) -> R<usize> {
        let start = self.pos;
        while self.pos < self.input.len() && self.input.as_bytes()[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return self.mk_err("expected non-negative integer");
        }
        match self.input[start..self.pos].parse::<usize>() {
            Ok(v) => Ok(v),
            Err(_) => self.mk_err("integer out of range"),
        }
    }

    fn values_clause(&mut self) -> R<Option<GraphPattern>> {
        let save = self.pos;
        if self.try_kw("VALUES") {
            self.ws();
            Ok(Some(self.data_block()?))
        } else {
            self.pos = save;
            Ok(None)
        }
    }

    // --- build_select (§18.2.4) --------------------------------------------

    pub(crate) fn build_select(
        &mut self,
        select: Selection,
        r#where: GraphPattern,
        modifiers: Modifiers,
    ) -> R<GraphPattern> {
        let Modifiers {
            mut group,
            having,
            order_by,
            offset_limit,
            values,
        } = modifiers;
        let mut p = r#where;
        let mut with_aggregate = false;

        let aggregates = self.aggregates.pop().unwrap_or_default();
        if group.is_none() && !aggregates.is_empty() {
            group = Some((vec![], vec![]));
        }

        if let Some((clauses, binds)) = group {
            for (expression, variable) in binds {
                p = GraphPattern::Extend {
                    inner: Box::new(p),
                    variable,
                    expression,
                };
            }
            p = GraphPattern::Group {
                inner: Box::new(p),
                variables: clauses,
                aggregates,
            };
            with_aggregate = true;
        }

        if let Some(expr) = having {
            p = GraphPattern::Filter {
                expr,
                inner: Box::new(p),
            };
        }

        if let Some(data) = values {
            p = new_join(p, data);
        }

        let mut pv: Vec<Variable> = Vec::new();
        let with_project = match select.variables {
            SelectionVariables::Explicit(members) => {
                let mut visible = HashSet::new();
                p.on_in_scope_variable(|v| {
                    visible.insert(v.clone());
                });
                for member in members {
                    let v = match member {
                        SelectionMember::Variable(v) => {
                            if with_aggregate && !visible.contains(&v) {
                                return self.fatal("The SELECT contains an unbound variable");
                            }
                            v
                        }
                        SelectionMember::Expression(expression, variable) => {
                            if visible.contains(&variable) {
                                return self
                                    .fatal("The SELECT overrides an existing variable with AS");
                            }
                            if with_aggregate && !are_variables_bound(&expression, &visible) {
                                return self
                                    .fatal("The SELECT expression uses an unbound variable");
                            }
                            p = GraphPattern::Extend {
                                inner: Box::new(p),
                                variable: variable.clone(),
                                expression,
                            };
                            variable
                        }
                    };
                    if pv.contains(&v) {
                        return self.fatal("Duplicated variable in SELECT");
                    }
                    pv.push(v);
                }
                true
            }
            SelectionVariables::Star => {
                if with_aggregate {
                    return self.fatal("SELECT * is not allowed with GROUP BY");
                }
                p.on_in_scope_variable(|v| {
                    if !pv.contains(v) {
                        pv.push(v.clone());
                    }
                });
                pv.sort();
                true
            }
            SelectionVariables::Everything => false,
        };

        let mut m = p;

        if let Some(expression) = order_by {
            m = GraphPattern::OrderBy {
                inner: Box::new(m),
                expression,
            };
        }

        if with_project {
            m = GraphPattern::Project {
                inner: Box::new(m),
                variables: pv,
            };
        }
        match select.option {
            SelectionOption::Distinct => m = GraphPattern::Distinct { inner: Box::new(m) },
            SelectionOption::Reduced => m = GraphPattern::Reduced { inner: Box::new(m) },
            SelectionOption::Default => (),
        }

        if let Some((start, length)) = offset_limit {
            m = GraphPattern::Slice {
                inner: Box::new(m),
                start,
                length,
            };
        }
        Ok(m)
    }
}

/// The parsed solution modifiers of a query.
pub(crate) struct Modifiers {
    #[allow(clippy::type_complexity)]
    pub(crate) group: Option<(Vec<Variable>, Vec<(Expression, Variable)>)>,
    pub(crate) having: Option<Expression>,
    pub(crate) order_by: Option<Vec<OrderExpression>>,
    pub(crate) offset_limit: Option<(usize, Option<usize>)>,
    pub(crate) values: Option<GraphPattern>,
}

impl Parser<'_> {
    /// `VarOrIri` — a variable or an IRI.
    pub(crate) fn var_or_iri(&mut self) -> R<NamedNodePattern> {
        alt!(
            self,
            self.var().map(NamedNodePattern::Variable),
            self.iri().map(NamedNodePattern::NamedNode),
        )
    }
}
