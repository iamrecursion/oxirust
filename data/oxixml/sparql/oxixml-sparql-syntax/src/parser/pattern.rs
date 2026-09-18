//! Graph pattern productions: group graph patterns, triples blocks, property paths, collections
//! and inline data.

use super::build::{
    PartialGraphPattern, TripleOrPathPattern, VariableOrPropertyPath, build_bgp, new_join,
    not_empty_fold,
};
use super::{Parser, R, alt};
use crate::algebra::{Expression, GraphPattern, PropertyPathExpression};
use crate::term::{GroundTerm, NamedNode, NamedNodePattern, TermPattern, TriplePattern};
use oxixml_model::vocab::rdf;

impl Parser<'_> {
    // --- group graph patterns ----------------------------------------------

    pub(crate) fn group_graph_pattern(&mut self) -> R<GraphPattern> {
        self.eat("{")?;
        self.ws();
        self.promote_bnodes();
        self.enter_recursion()?;
        // A nested group graph pattern (`EXISTS { … }`, a sub-`SELECT`, …) opens a fresh
        // aggregate scope, even inside a `FILTER` or `BIND` where aggregates are forbidden.
        let saved = std::mem::replace(&mut self.aggregates_allowed, true);
        let p = alt!(self, self.ggp_sub_close(), self.ggp_subselect_close());
        self.aggregates_allowed = saved;
        self.leave_recursion();
        p
    }

    fn ggp_sub_close(&mut self) -> R<GraphPattern> {
        let p = self.group_graph_pattern_sub()?;
        self.promote_bnodes();
        self.ws();
        self.eat("}")?;
        Ok(p)
    }

    fn ggp_subselect_close(&mut self) -> R<GraphPattern> {
        let p = self.sub_select()?;
        self.promote_bnodes();
        self.ws();
        self.eat("}")?;
        Ok(p)
    }

    fn promote_bnodes(&mut self) {
        let taken = std::mem::take(&mut self.currently_used_bnodes);
        self.used_bnodes.extend(taken);
    }

    /// `GroupGraphPatternSub ::= TriplesBlock? ( GraphPatternNotTriples '.'? TriplesBlock? )*`
    ///
    /// Every `OPTIONAL`/`MINUS`/`BIND`/nested pattern wraps the graph pattern built so far one
    /// level deeper (`LeftJoin`/`Minus`/`Extend`/`Join`), every `FILTER` extends a separate `And`
    /// chain, and a `TriplesBlock` interleaved between them folds through [`build_bgp`] — so a
    /// group with very many such clauses is, exactly like the other flat-chain productions, a flat
    /// sequence that folds into a tree one level deeper per clause. Each growth point charges
    /// [`Parser::enter_recursion`] right before it runs, so the tree this group builds — walked
    /// natively by `on_in_scope_variable` below on every `BIND`, and later by `Display`/`fmt_sse`/
    /// `Drop` — never grows deep enough to overflow the stack.
    fn group_graph_pattern_sub(&mut self) -> R<GraphPattern> {
        let mut filter: Option<Expression> = None;
        let mut g = match self.opt_triples_block()? {
            Some(v) => build_bgp(self, v)?,
            None => GraphPattern::default(),
        };
        let mut charged = 0_u32;
        loop {
            self.ws();
            let save = self.pos;
            match self.graph_pattern_not_triples() {
                Ok(part) => {
                    match part {
                        PartialGraphPattern::Optional(p, f) => {
                            self.enter_recursion()?;
                            charged += 1;
                            g = GraphPattern::LeftJoin {
                                left: Box::new(g),
                                right: Box::new(p),
                                expression: f,
                            }
                        }
                        PartialGraphPattern::Minus(p) => {
                            self.enter_recursion()?;
                            charged += 1;
                            g = GraphPattern::Minus {
                                left: Box::new(g),
                                right: Box::new(p),
                            }
                        }
                        PartialGraphPattern::Bind(expression, variable) => {
                            let mut contains = false;
                            g.on_in_scope_variable(|v| {
                                if *v == variable {
                                    contains = true;
                                }
                            });
                            if contains {
                                return self.fatal("BIND overrides an existing variable");
                            }
                            self.enter_recursion()?;
                            charged += 1;
                            g = GraphPattern::Extend {
                                inner: Box::new(g),
                                variable,
                                expression,
                            }
                        }
                        PartialGraphPattern::Filter(expr) => {
                            filter = Some(if let Some(f) = filter {
                                self.enter_recursion()?;
                                charged += 1;
                                Expression::And(Box::new(f), Box::new(expr))
                            } else {
                                expr
                            })
                        }
                        PartialGraphPattern::Other(e) => {
                            self.enter_recursion()?;
                            charged += 1;
                            g = new_join(g, e)
                        }
                    }
                    self.ws();
                    self.try_str(".");
                    self.ws();
                    if let Some(tb) = self.opt_triples_block()? {
                        self.enter_recursion()?;
                        charged += 1;
                        g = new_join(g, build_bgp(self, tb)?);
                    }
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    break;
                }
            }
        }
        self.leave_recursion_by(charged);
        Ok(if let Some(expr) = filter {
            GraphPattern::Filter {
                expr,
                inner: Box::new(g),
            }
        } else {
            g
        })
    }

    fn opt_triples_block(&mut self) -> R<Option<Vec<TripleOrPathPattern>>> {
        let save = self.pos;
        let mut all = Vec::new();
        match self.triples_same_subject_path(&mut all) {
            Ok(()) => {}
            Err(e) if e.fatal => return Err(e),
            Err(_) => {
                self.pos = save;
                return Ok(None);
            }
        }
        loop {
            self.ws();
            let save2 = self.pos;
            if !self.try_str(".") {
                self.pos = save2;
                break;
            }
            self.ws();
            let save3 = self.pos;
            match self.triples_same_subject_path(&mut all) {
                Ok(()) => {}
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save3;
                    break;
                }
            }
        }
        Ok(Some(all))
    }

    fn graph_pattern_not_triples(&mut self) -> R<PartialGraphPattern> {
        alt!(
            self,
            self.group_or_union_graph_pattern(),
            self.optional_graph_pattern(),
            self.minus_graph_pattern(),
            self.graph_graph_pattern(),
            self.service_graph_pattern(),
            self.filter_pattern(),
            self.bind_pattern(),
            self.inline_data(),
        )
    }

    /// `GroupOrUnionGraphPattern ::= GroupGraphPattern ( 'UNION' GroupGraphPattern )*`
    ///
    /// A flat chain of `n + 1` `UNION`-joined arms folds into a tree `n` levels deep, so — exactly
    /// like a genuinely nested construct — each additional arm charges [`Parser::enter_recursion`]
    /// before it is parsed. See [`Parser::leave_recursion_by`] for why the charge happens before,
    /// not after, and why it is released once for the whole chain rather than per arm.
    fn group_or_union_graph_pattern(&mut self) -> R<PartialGraphPattern> {
        let mut patterns = vec![self.group_graph_pattern()?];
        let mut charged = 0_u32;
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_kw("UNION") {
                self.pos = save;
                break;
            }
            self.ws();
            if let Err(err) = self.enter_recursion() {
                self.leave_recursion_by(charged);
                return Err(err);
            }
            charged += 1;
            match self.group_graph_pattern() {
                Ok(p) => patterns.push(p),
                Err(err) => {
                    self.leave_recursion_by(charged);
                    return Err(err);
                }
            }
        }
        let result = not_empty_fold(patterns.into_iter(), |a, b| GraphPattern::Union {
            left: Box::new(a),
            right: Box::new(b),
        });
        self.leave_recursion_by(charged);
        match result {
            Ok(g) => Ok(PartialGraphPattern::Other(g)),
            Err(_) => self.mk_err("expected group pattern"),
        }
    }

    fn optional_graph_pattern(&mut self) -> R<PartialGraphPattern> {
        self.kw("OPTIONAL")?;
        self.ws();
        let p = self.group_graph_pattern()?;
        Ok(if let GraphPattern::Filter { expr, inner } = p {
            PartialGraphPattern::Optional(*inner, Some(expr))
        } else {
            PartialGraphPattern::Optional(p, None)
        })
    }

    fn graph_graph_pattern(&mut self) -> R<PartialGraphPattern> {
        self.kw("GRAPH")?;
        self.ws();
        let name = self.var_or_iri()?;
        self.ws();
        let inner = self.group_graph_pattern()?;
        Ok(PartialGraphPattern::Other(GraphPattern::Graph {
            name,
            inner: Box::new(inner),
        }))
    }

    fn service_graph_pattern(&mut self) -> R<PartialGraphPattern> {
        self.kw("SERVICE")?;
        self.ws();
        let silent = self.try_kw("SILENT");
        self.ws();
        let name = self.var_or_iri()?;
        self.ws();
        let inner = self.group_graph_pattern()?;
        Ok(PartialGraphPattern::Other(GraphPattern::Service {
            name,
            inner: Box::new(inner),
            silent,
        }))
    }

    fn minus_graph_pattern(&mut self) -> R<PartialGraphPattern> {
        self.kw("MINUS")?;
        self.ws();
        let p = self.group_graph_pattern()?;
        Ok(PartialGraphPattern::Minus(p))
    }

    fn filter_pattern(&mut self) -> R<PartialGraphPattern> {
        self.kw("FILTER")?;
        self.ws();
        let c = self.constraint_without_aggregates()?;
        Ok(PartialGraphPattern::Filter(c))
    }

    fn bind_pattern(&mut self) -> R<PartialGraphPattern> {
        self.kw("BIND")?;
        self.ws();
        self.eat("(")?;
        self.ws();
        let e = self.expression_without_aggregates()?;
        self.ws();
        self.kw("AS")?;
        self.ws();
        let v = self.var()?;
        self.ws();
        self.eat(")")?;
        Ok(PartialGraphPattern::Bind(e, v))
    }

    fn inline_data(&mut self) -> R<PartialGraphPattern> {
        self.kw("VALUES")?;
        self.ws();
        Ok(PartialGraphPattern::Other(self.data_block()?))
    }

    // --- VALUES / data block ------------------------------------------------

    pub(crate) fn data_block(&mut self) -> R<GraphPattern> {
        alt!(self, self.inline_data_one_var(), self.inline_data_full())
    }

    fn inline_data_one_var(&mut self) -> R<GraphPattern> {
        let var = self.var()?;
        self.ws();
        self.eat("{")?;
        self.ws();
        let mut bindings = Vec::new();
        loop {
            let save = self.pos;
            match self.data_block_value() {
                Ok(v) => {
                    bindings.push(vec![v]);
                    self.ws();
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    break;
                }
            }
        }
        self.eat("}")?;
        Ok(GraphPattern::Values {
            variables: vec![var],
            bindings,
        })
    }

    fn inline_data_full(&mut self) -> R<GraphPattern> {
        // '(' Var* ')' '{' ( '(' DataBlockValue* ')' )* '}'
        self.eat("(")?;
        self.ws();
        let mut variables: Vec<crate::term::Variable> = Vec::new();
        loop {
            let save = self.pos;
            match self.var() {
                Ok(v) => {
                    if variables.contains(&v) {
                        // A solution mapping is a partial function from variables to terms, so a
                        // row may not bind the same variable twice.
                        return self.fatal("A variable is repeated in the VALUES variable list");
                    }
                    variables.push(v);
                    self.ws();
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    break;
                }
            }
        }
        self.eat(")")?;
        self.ws();
        self.eat("{")?;
        self.ws();
        let mut bindings = Vec::new();
        loop {
            let save = self.pos;
            if !self.try_str("(") {
                self.pos = save;
                break;
            }
            self.ws();
            let mut row = Vec::new();
            loop {
                let save2 = self.pos;
                match self.data_block_value() {
                    Ok(v) => {
                        row.push(v);
                        self.ws();
                    }
                    Err(e) if e.fatal => return Err(e),
                    Err(_) => {
                        self.pos = save2;
                        break;
                    }
                }
            }
            self.eat(")")?;
            self.ws();
            if row.len() != variables.len() {
                return self.fatal("VALUES row arity does not match the variable count");
            }
            bindings.push(row);
        }
        self.eat("}")?;
        Ok(GraphPattern::Values {
            variables,
            bindings,
        })
    }

    fn data_block_value(&mut self) -> R<Option<GroundTerm>> {
        if self.try_kw("UNDEF") {
            return Ok(None);
        }
        #[cfg(feature = "sparql-12")]
        if self.at_triple_term() {
            return self.triple_term_data().map(|t| Some(GroundTerm::from(t)));
        }
        alt!(
            self,
            self.iri().map(|i| Some(GroundTerm::from(i))),
            self.rdf_literal().map(|l| Some(GroundTerm::from(l))),
            self.numeric_literal().map(|l| Some(GroundTerm::from(l))),
            self.boolean_literal().map(|l| Some(GroundTerm::from(l))),
        )
    }

    // --- triples with property paths ---------------------------------------

    fn triples_same_subject_path(&mut self, patterns: &mut Vec<TripleOrPathPattern>) -> R<()> {
        let save = self.pos;
        // A) VarOrTerm PropertyListPathNotEmpty
        match self.var_or_term() {
            Ok(subject) => {
                self.ws();
                match self.property_list_path_not_empty(&subject, patterns) {
                    Ok(()) => return Ok(()),
                    Err(e) if e.fatal => return Err(e),
                    Err(_) => self.pos = save,
                }
            }
            Err(e) if e.fatal => return Err(e),
            Err(_) => self.pos = save,
        }
        // B) ReifiedTripleBlockPath ::= ReifiedTriple PropertyListPath
        #[cfg(feature = "sparql-12")]
        if self.at_reified_triple() {
            let subject = self.reified_triple_path(patterns)?;
            self.ws();
            return self.property_list_path(&subject, patterns);
        }
        // C) TriplesNodePath PropertyListPath
        let subject = self.triples_node_path(patterns)?;
        self.ws();
        self.property_list_path(&subject, patterns)
    }

    fn property_list_path(
        &mut self,
        subject: &TermPattern,
        patterns: &mut Vec<TripleOrPathPattern>,
    ) -> R<()> {
        let save = self.pos;
        match self.property_list_path_not_empty(subject, patterns) {
            Ok(()) => Ok(()),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                Ok(())
            }
        }
    }

    pub(crate) fn property_list_path_not_empty(
        &mut self,
        subject: &TermPattern,
        patterns: &mut Vec<TripleOrPathPattern>,
    ) -> R<()> {
        let verb = self.verb_path_or_simple()?;
        self.ws();
        self.object_list_path(subject, &verb, patterns)?;
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_str(";") {
                self.pos = save;
                break;
            }
            self.ws();
            let save2 = self.pos;
            match self.verb_path_or_simple() {
                Ok(v) => {
                    self.ws();
                    self.object_list_path(subject, &v, patterns)?;
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save2;
                }
            }
        }
        Ok(())
    }

    fn verb_path_or_simple(&mut self) -> R<VariableOrPropertyPath> {
        alt!(
            self,
            self.path().map(VariableOrPropertyPath::from),
            self.var().map(VariableOrPropertyPath::from),
        )
    }

    fn object_list_path(
        &mut self,
        subject: &TermPattern,
        verb: &VariableOrPropertyPath,
        patterns: &mut Vec<TripleOrPathPattern>,
    ) -> R<()> {
        loop {
            let object = self.graph_node_path(patterns)?;
            #[cfg(feature = "sparql-12")]
            self.annotation_path(subject, verb, &object, patterns)?;
            self.add_to_triple_or_path(subject.clone(), verb.clone(), object, patterns)?;
            self.ws();
            let save = self.pos;
            if !self.try_str(",") {
                self.pos = save;
                break;
            }
            self.ws();
        }
        Ok(())
    }

    fn add_to_triple_or_path(
        &mut self,
        subject: TermPattern,
        predicate: VariableOrPropertyPath,
        object: TermPattern,
        patterns: &mut Vec<TripleOrPathPattern>,
    ) -> R<()> {
        match predicate {
            VariableOrPropertyPath::Variable(p) => {
                patterns.push(TripleOrPathPattern::Triple(TriplePattern::new(
                    subject, p, object,
                )));
            }
            VariableOrPropertyPath::PropertyPath(p) => match p {
                PropertyPathExpression::NamedNode(p) => {
                    patterns.push(TripleOrPathPattern::Triple(TriplePattern::new(
                        subject, p, object,
                    )));
                }
                PropertyPathExpression::Reverse(p) => {
                    self.add_to_triple_or_path(object, (*p).into(), subject, patterns)?;
                }
                PropertyPathExpression::Sequence(a, b) => {
                    let middle = TermPattern::from(self.fresh_bnode());
                    self.add_to_triple_or_path(subject, (*a).into(), middle.clone(), patterns)?;
                    self.add_to_triple_or_path(middle, (*b).into(), object, patterns)?;
                }
                path => patterns.push(TripleOrPathPattern::Path {
                    subject,
                    path,
                    object,
                }),
            },
        }
        Ok(())
    }

    fn triples_node_path(&mut self, patterns: &mut Vec<TripleOrPathPattern>) -> R<TermPattern> {
        self.enter_recursion()?;
        let t = alt!(
            self,
            self.collection_path(patterns),
            self.blank_node_property_list_path(patterns),
        );
        self.leave_recursion();
        t
    }

    fn blank_node_property_list_path(
        &mut self,
        patterns: &mut Vec<TripleOrPathPattern>,
    ) -> R<TermPattern> {
        self.eat("[")?;
        self.ws();
        let subject = TermPattern::from(self.fresh_bnode());
        self.property_list_path_not_empty(&subject, patterns)?;
        self.ws();
        self.eat("]")?;
        Ok(subject)
    }

    fn collection_path(&mut self, patterns: &mut Vec<TripleOrPathPattern>) -> R<TermPattern> {
        self.eat("(")?;
        self.ws();
        let mut objects = Vec::new();
        loop {
            let save = self.pos;
            match self.graph_node_path(patterns) {
                Ok(o) => {
                    objects.push(o);
                    self.ws();
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    break;
                }
            }
        }
        if objects.is_empty() {
            return self.mk_err("empty collection");
        }
        self.eat(")")?;
        let mut current = TermPattern::from(rdf::NIL.into_owned());
        for object in objects.into_iter().rev() {
            let node = TermPattern::from(self.fresh_bnode());
            patterns.push(TripleOrPathPattern::Triple(TriplePattern::new(
                node.clone(),
                rdf::FIRST.into_owned(),
                object,
            )));
            patterns.push(TripleOrPathPattern::Triple(TriplePattern::new(
                node.clone(),
                rdf::REST.into_owned(),
                current,
            )));
            current = node;
        }
        Ok(current)
    }

    fn graph_node_path(&mut self, patterns: &mut Vec<TripleOrPathPattern>) -> R<TermPattern> {
        #[cfg(feature = "sparql-12")]
        if self.at_reified_triple() {
            return self.reified_triple_path(patterns);
        }
        let save = self.pos;
        match self.var_or_term() {
            Ok(t) => Ok(t),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                self.triples_node_path(patterns)
            }
        }
    }

    // --- triples without paths (CONSTRUCT template, quads) ------------------

    pub(crate) fn triples_template(&mut self, patterns: &mut Vec<TriplePattern>) -> R<()> {
        self.triples_same_subject(patterns)?;
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_str(".") {
                self.pos = save;
                break;
            }
            self.ws();
            let save2 = self.pos;
            match self.triples_same_subject(patterns) {
                Ok(()) => {}
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save2;
                    break;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn triples_same_subject(&mut self, patterns: &mut Vec<TriplePattern>) -> R<()> {
        let save = self.pos;
        match self.var_or_term() {
            Ok(subject) => {
                self.ws();
                match self.property_list_not_empty(&subject, patterns) {
                    Ok(()) => return Ok(()),
                    Err(e) if e.fatal => return Err(e),
                    Err(_) => self.pos = save,
                }
            }
            Err(e) if e.fatal => return Err(e),
            Err(_) => self.pos = save,
        }
        #[cfg(feature = "sparql-12")]
        if self.at_reified_triple() {
            let subject = self.reified_triple(patterns)?;
            self.ws();
            return self.property_list(&subject, patterns);
        }
        let subject = self.triples_node(patterns)?;
        self.ws();
        self.property_list(&subject, patterns)
    }

    fn property_list(&mut self, subject: &TermPattern, patterns: &mut Vec<TriplePattern>) -> R<()> {
        let save = self.pos;
        match self.property_list_not_empty(subject, patterns) {
            Ok(()) => Ok(()),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                Ok(())
            }
        }
    }

    pub(crate) fn property_list_not_empty(
        &mut self,
        subject: &TermPattern,
        patterns: &mut Vec<TriplePattern>,
    ) -> R<()> {
        let verb = self.verb()?;
        self.ws();
        self.object_list(subject, &verb, patterns)?;
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_str(";") {
                self.pos = save;
                break;
            }
            self.ws();
            let save2 = self.pos;
            match self.verb() {
                Ok(v) => {
                    self.ws();
                    self.object_list(subject, &v, patterns)?;
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save2;
                }
            }
        }
        Ok(())
    }

    fn verb(&mut self) -> R<NamedNodePattern> {
        let save = self.pos;
        match self.var_or_iri() {
            Ok(v) => Ok(v),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                if self.try_a_keyword() {
                    Ok(NamedNodePattern::NamedNode(rdf::TYPE.into_owned()))
                } else {
                    self.mk_err("expected verb")
                }
            }
        }
    }

    fn object_list(
        &mut self,
        subject: &TermPattern,
        verb: &NamedNodePattern,
        patterns: &mut Vec<TriplePattern>,
    ) -> R<()> {
        loop {
            let object = self.graph_node(patterns)?;
            #[cfg(feature = "sparql-12")]
            self.annotation(subject, verb, &object, patterns)?;
            patterns.push(TriplePattern::new(subject.clone(), verb.clone(), object));
            self.ws();
            let save = self.pos;
            if !self.try_str(",") {
                self.pos = save;
                break;
            }
            self.ws();
        }
        Ok(())
    }

    fn triples_node(&mut self, patterns: &mut Vec<TriplePattern>) -> R<TermPattern> {
        self.enter_recursion()?;
        let t = alt!(
            self,
            self.collection(patterns),
            self.blank_node_property_list(patterns),
        );
        self.leave_recursion();
        t
    }

    fn blank_node_property_list(&mut self, patterns: &mut Vec<TriplePattern>) -> R<TermPattern> {
        self.eat("[")?;
        self.ws();
        let subject = TermPattern::from(self.fresh_bnode());
        self.property_list_not_empty(&subject, patterns)?;
        self.ws();
        self.eat("]")?;
        Ok(subject)
    }

    fn collection(&mut self, patterns: &mut Vec<TriplePattern>) -> R<TermPattern> {
        self.eat("(")?;
        self.ws();
        let mut objects = Vec::new();
        loop {
            let save = self.pos;
            match self.graph_node(patterns) {
                Ok(o) => {
                    objects.push(o);
                    self.ws();
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    break;
                }
            }
        }
        if objects.is_empty() {
            return self.mk_err("empty collection");
        }
        self.eat(")")?;
        let mut current = TermPattern::from(rdf::NIL.into_owned());
        for object in objects.into_iter().rev() {
            let node = TermPattern::from(self.fresh_bnode());
            patterns.push(TriplePattern::new(
                node.clone(),
                rdf::FIRST.into_owned(),
                object,
            ));
            patterns.push(TriplePattern::new(
                node.clone(),
                rdf::REST.into_owned(),
                current,
            ));
            current = node;
        }
        Ok(current)
    }

    fn graph_node(&mut self, patterns: &mut Vec<TriplePattern>) -> R<TermPattern> {
        #[cfg(feature = "sparql-12")]
        if self.at_reified_triple() {
            return self.reified_triple(patterns);
        }
        let save = self.pos;
        match self.var_or_term() {
            Ok(t) => Ok(t),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                self.triples_node(patterns)
            }
        }
    }

    // --- terms --------------------------------------------------------------

    pub(crate) fn var_or_term(&mut self) -> R<TermPattern> {
        let save = self.pos;
        match self.var() {
            Ok(v) => Ok(v.into()),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                self.graph_term()
            }
        }
    }

    fn graph_term(&mut self) -> R<TermPattern> {
        // NIL must be tried before blank node's '(' handling of collections is not here.
        let save = self.pos;
        if self.nil().is_ok() {
            return Ok(TermPattern::from(rdf::NIL.into_owned()));
        }
        self.pos = save;
        #[cfg(feature = "sparql-12")]
        if self.at_triple_term() {
            return self.triple_term().map(TermPattern::from);
        }
        alt!(
            self,
            self.iri().map(TermPattern::from),
            self.rdf_literal().map(TermPattern::from),
            self.numeric_literal().map(TermPattern::from),
            self.boolean_literal().map(TermPattern::from),
            self.blank_node().map(TermPattern::from),
        )
    }

    /// Consumes the `a` abbreviation of `rdf:type`, leaving the cursor untouched on failure.
    ///
    /// The SPARQL tokenizer follows the longest-token rule, so `a:` is the `PNAME_NS` of the
    /// prefix `a` and never `a` followed by `:`; likewise `ab` is a `PN_PREFIX` fragment rather
    /// than the keyword `a` followed by `b`.
    pub(crate) fn try_a_keyword(&mut self) -> bool {
        let save = self.pos;
        if !self.try_str("a") {
            return false;
        }
        let continues_name = matches!(
            self.peek_char(),
            Some(c) if super::terminal::is_pn_chars(c) || c == ':'
        );
        if continues_name {
            self.pos = save;
            false
        } else {
            true
        }
    }

    // --- property paths -----------------------------------------------------

    fn path(&mut self) -> R<PropertyPathExpression> {
        // `PathPrimary ::= … | '(' Path ')'` recurses back here, so the depth must be bounded.
        self.enter_recursion()?;
        let p = self.path_alternative();
        self.leave_recursion();
        p
    }

    /// `PathAlternative ::= PathSequence ( '|' PathSequence )*`
    ///
    /// A flat chain of `n + 1` `|`-joined elements folds into a tree `n` levels deep, so — exactly
    /// like a genuinely nested construct — each additional element charges
    /// [`Parser::enter_recursion`] before it is parsed. See [`Parser::leave_recursion_by`] for why
    /// the charge happens before, not after, and why it is released once for the whole chain
    /// rather than per element.
    fn path_alternative(&mut self) -> R<PropertyPathExpression> {
        let mut items = vec![self.path_sequence()?];
        let mut charged = 0_u32;
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_str("|") {
                self.pos = save;
                break;
            }
            self.ws();
            if let Err(err) = self.enter_recursion() {
                self.leave_recursion_by(charged);
                return Err(err);
            }
            charged += 1;
            match self.path_sequence() {
                Ok(p) => items.push(p),
                Err(err) => {
                    self.leave_recursion_by(charged);
                    return Err(err);
                }
            }
        }
        let result = not_empty_fold(items.into_iter(), |a, b| {
            PropertyPathExpression::Alternative(Box::new(a), Box::new(b))
        });
        self.leave_recursion_by(charged);
        match result {
            Ok(p) => Ok(p),
            Err(_) => self.mk_err("expected path"),
        }
    }

    /// `PathSequence ::= PathEltOrInverse ( '/' PathEltOrInverse )*`
    ///
    /// See [`Parser::path_alternative`]: the same left-deep fold, charged the same way. The
    /// resulting bound also protects [`Parser::add_to_triple_or_path`], the only other place that
    /// walks a `PropertyPathExpression::Sequence` chain (natively, recursively, while expanding it
    /// into triple/path patterns): it can only recurse as deep as the chain built here, which this
    /// bound now keeps shallow.
    fn path_sequence(&mut self) -> R<PropertyPathExpression> {
        let mut items = vec![self.path_elt_or_inverse()?];
        let mut charged = 0_u32;
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_str("/") {
                self.pos = save;
                break;
            }
            self.ws();
            if let Err(err) = self.enter_recursion() {
                self.leave_recursion_by(charged);
                return Err(err);
            }
            charged += 1;
            match self.path_elt_or_inverse() {
                Ok(p) => items.push(p),
                Err(err) => {
                    self.leave_recursion_by(charged);
                    return Err(err);
                }
            }
        }
        let result = not_empty_fold(items.into_iter(), |a, b| {
            PropertyPathExpression::Sequence(Box::new(a), Box::new(b))
        });
        self.leave_recursion_by(charged);
        match result {
            Ok(p) => Ok(p),
            Err(_) => self.mk_err("expected path"),
        }
    }

    fn path_elt_or_inverse(&mut self) -> R<PropertyPathExpression> {
        if self.try_str("^") {
            self.ws();
            let p = self.path_elt()?;
            Ok(PropertyPathExpression::Reverse(Box::new(p)))
        } else {
            self.path_elt()
        }
    }

    fn path_elt(&mut self) -> R<PropertyPathExpression> {
        let p = self.path_primary()?;
        self.ws();
        if self.try_str("*") {
            Ok(PropertyPathExpression::ZeroOrMore(Box::new(p)))
        } else if self.try_str("+") {
            Ok(PropertyPathExpression::OneOrMore(Box::new(p)))
        } else if self.peek_char() == Some('?') && !self.question_is_variable() {
            self.pos += 1;
            Ok(PropertyPathExpression::ZeroOrOne(Box::new(p)))
        } else {
            Ok(p)
        }
    }

    /// True when a `?` is the start of a variable rather than the zero-or-one path operator.
    fn question_is_variable(&self) -> bool {
        matches!(
            self.rest().as_bytes().get(1),
            Some(c) if c.is_ascii_digit() || *c == b'_' || c.is_ascii_alphabetic() || *c >= 0x80
        )
    }

    fn path_primary(&mut self) -> R<PropertyPathExpression> {
        if self.peek_char() == Some('(') {
            self.eat("(")?;
            self.ws();
            let p = self.path()?;
            self.ws();
            self.eat(")")?;
            Ok(p)
        } else if self.try_str("!") {
            self.ws();
            self.path_negated_property_set()
        } else if self.try_a_keyword() {
            Ok(PropertyPathExpression::NamedNode(rdf::TYPE.into_owned()))
        } else {
            self.iri().map(PropertyPathExpression::NamedNode)
        }
    }

    fn path_negated_property_set(&mut self) -> R<PropertyPathExpression> {
        if self.peek_char() == Some('(') {
            self.eat("(")?;
            self.ws();
            let mut direct = Vec::new();
            let mut inverse = Vec::new();
            let save = self.pos;
            if let Ok((rev, iri)) = self.path_one_in_property_set() {
                if rev {
                    inverse.push(iri);
                } else {
                    direct.push(iri);
                }
                loop {
                    self.ws();
                    let s = self.pos;
                    if !self.try_str("|") {
                        self.pos = s;
                        break;
                    }
                    self.ws();
                    let (rev, iri) = self.path_one_in_property_set()?;
                    if rev {
                        inverse.push(iri);
                    } else {
                        direct.push(iri);
                    }
                }
            } else {
                self.pos = save;
            }
            self.ws();
            self.eat(")")?;
            Ok(build_negated(direct, inverse))
        } else {
            let (rev, iri) = self.path_one_in_property_set()?;
            Ok(if rev {
                PropertyPathExpression::Reverse(Box::new(
                    PropertyPathExpression::NegatedPropertySet(vec![iri]),
                ))
            } else {
                PropertyPathExpression::NegatedPropertySet(vec![iri])
            })
        }
    }

    fn path_one_in_property_set(&mut self) -> R<(bool, NamedNode)> {
        let reverse = self.try_str("^");
        if reverse {
            self.ws();
        }
        if self.try_a_keyword() {
            Ok((reverse, rdf::TYPE.into_owned()))
        } else {
            let iri = self.iri()?;
            Ok((reverse, iri))
        }
    }
}

fn build_negated(direct: Vec<NamedNode>, inverse: Vec<NamedNode>) -> PropertyPathExpression {
    if inverse.is_empty() {
        PropertyPathExpression::NegatedPropertySet(direct)
    } else if direct.is_empty() {
        PropertyPathExpression::Reverse(Box::new(PropertyPathExpression::NegatedPropertySet(
            inverse,
        )))
    } else {
        PropertyPathExpression::Alternative(
            Box::new(PropertyPathExpression::NegatedPropertySet(direct)),
            Box::new(PropertyPathExpression::Reverse(Box::new(
                PropertyPathExpression::NegatedPropertySet(inverse),
            ))),
        )
    }
}
