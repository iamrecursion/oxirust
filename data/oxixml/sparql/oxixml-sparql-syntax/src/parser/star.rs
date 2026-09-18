//! SPARQL 1.2 (RDF-star) productions: triple terms, reified triples, reifiers and annotation
//! blocks.
//!
//! Everything in this module is gated behind the `sparql-12` feature and follows the
//! [SPARQL 1.2 Query](https://www.w3.org/TR/sparql12-query/) grammar productions `[59]`,
//! `[70]`–`[71]`, `[110]`–`[124]` and `[137]`–`[139]`.
//!
//! The translation to the algebra is the one mandated by the specification: a reifier `~ R`
//! attached to the triple `S P O` contributes the extra pattern
//! `R rdf:reifies <<( S P O )>>`, and an annotation block `{| ... |}` attaches its property list
//! to that same reifier.

use super::build::{TripleOrPathPattern, VariableOrPropertyPath};
use super::{Parser, R, alt};
use crate::algebra::{Expression, PropertyPathExpression};
use crate::term::{GroundTerm, GroundTriple, NamedNodePattern, TermPattern, TriplePattern};
use crate::vocab::sparql;
use oxixml_model::vocab::rdf;
use oxixml_model::{BaseDirection, Literal, NamedNode};

/// One parsed `Annotation` / `AnnotationPath` element.
enum AnnotationPart {
    /// `Reifier ::= '~' VarOrReifierId?`
    Reifier(Option<TermPattern>),
    /// `AnnotationBlock ::= '{|' PropertyListNotEmpty '|}'` — the cursor sits on `{|`.
    Block,
}

impl Parser<'_> {
    // --- terminals ----------------------------------------------------------

    /// The direction suffix of a `LANG_DIR` token, i.e. the `--ltr` of `@en--ltr`.
    pub(crate) fn base_direction(&mut self) -> Option<BaseDirection> {
        let save = self.pos;
        if !self.try_str("--") {
            self.pos = save;
            return None;
        }
        if self.try_str("ltr") {
            Some(BaseDirection::Ltr)
        } else if self.try_str("rtl") {
            Some(BaseDirection::Rtl)
        } else {
            self.pos = save;
            None
        }
    }

    /// `VersionDecl ::= 'VERSION' VersionSpecifier`, where the specifier is a short string.
    ///
    /// The version is validated but, exactly like `spargebra`, is not kept in the algebra.
    pub(crate) fn version_decl(&mut self) -> R<()> {
        self.ws();
        let value = self.short_string_only()?;
        if value.is_empty() {
            return self.fatal("The VERSION specifier must not be empty");
        }
        Ok(())
    }

    // --- triple terms -------------------------------------------------------

    /// `TripleTerm ::= '<<(' TripleTermSubject Verb TripleTermObject ')>>'`
    ///
    /// A triple term nested in subject or object position (`<<( ... <<( ... )>> ... )>>`)
    /// recurses back into this same production through
    /// [`Parser::triple_term_subject_or_object`], so — exactly like a parenthesized expression or
    /// a nested group graph pattern — the depth must be bounded to keep this parser's own call
    /// stack (and the native `Display`/`Drop` glue that later walks the resulting, just as deeply
    /// nested, [`TriplePattern`]) within a safe depth.
    pub(crate) fn triple_term(&mut self) -> R<TriplePattern> {
        self.enter_recursion()?;
        let result = self.triple_term_body();
        self.leave_recursion();
        result
    }

    fn triple_term_body(&mut self) -> R<TriplePattern> {
        self.eat("<<(")?;
        self.ws();
        let subject = self.triple_term_subject_or_object()?;
        self.ws();
        let predicate = self.reified_verb()?;
        self.ws();
        let object = self.triple_term_subject_or_object()?;
        self.ws();
        self.eat(")>>")?;
        Ok(TriplePattern::new(subject, predicate, object))
    }

    /// `TripleTermSubject` / `TripleTermObject`.
    fn triple_term_subject_or_object(&mut self) -> R<TermPattern> {
        alt!(
            self,
            self.var().map(TermPattern::from),
            self.iri().map(TermPattern::from),
            self.rdf_literal().map(TermPattern::from),
            self.numeric_literal().map(TermPattern::from),
            self.boolean_literal().map(TermPattern::from),
            self.blank_node().map(TermPattern::from),
            self.triple_term().map(TermPattern::from),
        )
    }

    /// `TripleTermData ::= '<<(' TripleTermDataSubject ( iri | 'a' ) TripleTermDataObject ')>>'`
    ///
    /// Used by `VALUES`, `INSERT DATA` and `DELETE DATA`, where no variable may appear.
    ///
    /// See [`Parser::triple_term`]: [`Parser::triple_term_data_object`] recurses back into this
    /// same production for a nested triple term, so the depth is bounded the same way.
    pub(crate) fn triple_term_data(&mut self) -> R<GroundTriple> {
        self.enter_recursion()?;
        let result = self.triple_term_data_body();
        self.leave_recursion();
        result
    }

    fn triple_term_data_body(&mut self) -> R<GroundTriple> {
        self.eat("<<(")?;
        self.ws();
        let subject = self.iri()?;
        self.ws();
        let predicate = self.iri_or_a()?;
        self.ws();
        let object = self.triple_term_data_object()?;
        self.ws();
        self.eat(")>>")?;
        Ok(GroundTriple {
            subject,
            predicate,
            object,
        })
    }

    /// `TripleTermDataObject ::= iri | RDFLiteral | NumericLiteral | BooleanLiteral | TripleTermData`
    fn triple_term_data_object(&mut self) -> R<GroundTerm> {
        alt!(
            self,
            self.iri().map(GroundTerm::from),
            self.rdf_literal().map(GroundTerm::from),
            self.numeric_literal().map(GroundTerm::from),
            self.boolean_literal().map(GroundTerm::from),
            self.triple_term_data().map(GroundTerm::from),
        )
    }

    /// `ExprTripleTerm ::= '<<(' ExprTripleTermSubject Verb ExprTripleTermObject ')>>'`
    ///
    /// It translates to a call of the `TRIPLE` built-in, matching `spargebra`.
    ///
    /// See [`Parser::triple_term`]: [`Parser::expr_triple_term_object`] recurses back into this
    /// same production for a nested triple term, so the depth is bounded the same way — bounding
    /// in turn the depth of the `TRIPLE(…, …, TRIPLE(…))` [`Expression::FunctionCall`] chain this
    /// builds, which is walked just as natively by `Display`/`fmt_sse`/`Drop`.
    pub(crate) fn expr_triple_term(&mut self) -> R<Expression> {
        self.enter_recursion()?;
        let result = self.expr_triple_term_body();
        self.leave_recursion();
        result
    }

    fn expr_triple_term_body(&mut self) -> R<Expression> {
        self.eat("<<(")?;
        self.ws();
        let subject = alt!(
            self,
            self.iri().map(Expression::from),
            self.var().map(Expression::from),
        )?;
        self.ws();
        let predicate = self.reified_verb()?;
        self.ws();
        let object = self.expr_triple_term_object()?;
        self.ws();
        self.eat(")>>")?;
        Ok(Expression::FunctionCall(
            sparql::TRIPLE,
            vec![subject, Expression::from(predicate), object],
        ))
    }

    /// `ExprTripleTermObject ::= iri | RDFLiteral | NumericLiteral | BooleanLiteral | Var | ExprTripleTerm`
    fn expr_triple_term_object(&mut self) -> R<Expression> {
        alt!(
            self,
            self.iri().map(Expression::from),
            self.rdf_literal().map(Expression::from),
            self.numeric_literal().map(Expression::from),
            self.boolean_literal().map(Expression::from),
            self.var().map(Expression::from),
            self.expr_triple_term(),
        )
    }

    // --- reified triples ----------------------------------------------------

    /// `ReifiedTriple ::= '<<' ReifiedTripleSubject Verb ReifiedTripleObject Reifier? '>>'`
    ///
    /// Returns the reifier term, which stands for the reified triple in the enclosing pattern,
    /// and pushes `reifier rdf:reifies <<( subject predicate object )>>`.
    ///
    /// See [`Parser::triple_term`]: [`Parser::reified_triple_subject_or_object`] recurses back
    /// into this same production for a reified triple nested in subject or object position
    /// (`<< << … >> p o >>`), so the depth is bounded the same way.
    pub(crate) fn reified_triple(&mut self, patterns: &mut Vec<TriplePattern>) -> R<TermPattern> {
        self.enter_recursion()?;
        let result = self.reified_triple_body(patterns);
        self.leave_recursion();
        result
    }

    fn reified_triple_body(&mut self, patterns: &mut Vec<TriplePattern>) -> R<TermPattern> {
        self.eat("<<")?;
        // `<<(` opens a triple term, never a reified triple.
        if self.peek_char() == Some('(') {
            return self.mk_err("expected a reified triple");
        }
        self.ws();
        let subject = self.reified_triple_subject_or_object(patterns)?;
        self.ws();
        let predicate = self.reified_verb()?;
        self.ws();
        let object = self.reified_triple_subject_or_object(patterns)?;
        self.ws();
        // An absent or anonymous reifier both stand for a fresh blank node.
        let reifier = match self.reifier()?.flatten() {
            Some(id) => id,
            None => TermPattern::from(self.fresh_bnode()),
        };
        self.ws();
        self.eat(">>")?;
        patterns.push(TriplePattern::new(
            reifier.clone(),
            rdf::REIFIES.into_owned(),
            TriplePattern::new(subject, predicate, object),
        ));
        Ok(reifier)
    }

    /// The `Vec<TripleOrPathPattern>` flavour of [`Parser::reified_triple`].
    pub(crate) fn reified_triple_path(
        &mut self,
        patterns: &mut Vec<TripleOrPathPattern>,
    ) -> R<TermPattern> {
        let mut inner = Vec::new();
        let term = self.reified_triple(&mut inner)?;
        patterns.extend(inner.into_iter().map(TripleOrPathPattern::Triple));
        Ok(term)
    }

    /// `ReifiedTripleSubject` / `ReifiedTripleObject`.
    fn reified_triple_subject_or_object(
        &mut self,
        patterns: &mut Vec<TriplePattern>,
    ) -> R<TermPattern> {
        let save = self.pos;
        match self.var() {
            Ok(v) => return Ok(v.into()),
            Err(e) if e.fatal => return Err(e),
            Err(_) => self.pos = save,
        }
        let simple = alt!(
            self,
            self.iri().map(TermPattern::from),
            self.rdf_literal().map(TermPattern::from),
            self.numeric_literal().map(TermPattern::from),
            self.boolean_literal().map(TermPattern::from),
            self.blank_node().map(TermPattern::from),
            self.triple_term().map(TermPattern::from),
        );
        match simple {
            Ok(t) => Ok(t),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                let before = patterns.len();
                match self.reified_triple(patterns) {
                    Ok(t) => Ok(t),
                    Err(e) => {
                        patterns.truncate(before);
                        Err(e)
                    }
                }
            }
        }
    }

    /// `Reifier ::= '~' VarOrReifierId?`
    ///
    /// `Ok(None)` means there was no reifier at all, `Ok(Some(None))` an anonymous one.
    fn reifier(&mut self) -> R<Option<Option<TermPattern>>> {
        let save = self.pos;
        if !self.try_str("~") {
            self.pos = save;
            return Ok(None);
        }
        self.ws();
        let inner = self.pos;
        match self.var_or_reifier_id() {
            Ok(id) => Ok(Some(Some(id))),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = inner;
                Ok(Some(None))
            }
        }
    }

    /// `VarOrReifierId ::= Var | iri | BlankNode`
    fn var_or_reifier_id(&mut self) -> R<TermPattern> {
        alt!(
            self,
            self.var().map(TermPattern::from),
            self.iri().map(TermPattern::from),
            self.blank_node().map(TermPattern::from),
        )
    }

    /// `Verb ::= VarOrIri | 'a'` restricted to what a triple term may hold.
    fn reified_verb(&mut self) -> R<NamedNodePattern> {
        let save = self.pos;
        match self.var_or_iri() {
            Ok(v) => Ok(v),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                if self.try_a_keyword() {
                    Ok(NamedNodePattern::NamedNode(rdf::TYPE.into_owned()))
                } else {
                    self.mk_err("expected a verb")
                }
            }
        }
    }

    /// `iri | 'a'`, for the ground `TripleTermData` production.
    fn iri_or_a(&mut self) -> R<NamedNode> {
        let save = self.pos;
        match self.iri() {
            Ok(v) => Ok(v),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                if self.try_a_keyword() {
                    Ok(rdf::TYPE.into_owned())
                } else {
                    self.mk_err("expected a verb")
                }
            }
        }
    }

    // --- annotations --------------------------------------------------------

    /// `Annotation ::= ( Reifier | AnnotationBlock )*`, in a plain (path-free) triples block.
    ///
    /// An annotation block nested inside another (`{| ... {| ... |} ... |}`) recurses back into
    /// this same production through [`Parser::property_list_not_empty`]'s own object list, so —
    /// exactly like [`Parser::triple_term`] and the other self-recursive `sparql-12` productions
    /// — the depth must be bounded. Unlike those, an `Annotation` is not itself recursive (the
    /// loop above walks a flat, same-level run of reifiers and blocks); only entering a block's
    /// own body is, so only that one call is guarded below.
    pub(crate) fn annotation(
        &mut self,
        subject: &TermPattern,
        predicate: &NamedNodePattern,
        object: &TermPattern,
        patterns: &mut Vec<TriplePattern>,
    ) -> R<()> {
        let mut current: Option<TermPattern> = None;
        loop {
            self.ws();
            let Some(part) = self.annotation_part()? else {
                break;
            };
            let to_emit = match part {
                AnnotationPart::Reifier(id) => {
                    let previous = current.take();
                    current = Some(id.unwrap_or_else(|| self.fresh_bnode().into()));
                    previous
                }
                AnnotationPart::Block => {
                    let reifier = match current.take() {
                        Some(r) => r,
                        None => self.fresh_bnode().into(),
                    };
                    self.ws();
                    self.enter_recursion()?;
                    let result = self.property_list_not_empty(&reifier, patterns);
                    self.leave_recursion();
                    result?;
                    self.ws();
                    self.eat("|}")?;
                    Some(reifier)
                }
            };
            if let Some(reifier) = to_emit {
                patterns.push(reifies(reifier, subject, predicate, object));
            }
        }
        if let Some(reifier) = current {
            patterns.push(reifies(reifier, subject, predicate, object));
        }
        Ok(())
    }

    /// `AnnotationPath ::= ( Reifier | AnnotationBlockPath )*`, in a property-path triples block.
    ///
    /// See [`Parser::annotation`]: the same recursion, through
    /// [`Parser::property_list_path_not_empty`] here, guarded the same way.
    pub(crate) fn annotation_path(
        &mut self,
        subject: &TermPattern,
        predicate: &VariableOrPropertyPath,
        object: &TermPattern,
        patterns: &mut Vec<TripleOrPathPattern>,
    ) -> R<()> {
        // Peek first, so that a plain SPARQL 1.1 triples block never pays for the lookup.
        self.ws();
        if !self.at_annotation() {
            return Ok(());
        }
        let predicate = match predicate {
            VariableOrPropertyPath::Variable(v) => NamedNodePattern::Variable(v.clone()),
            VariableOrPropertyPath::PropertyPath(PropertyPathExpression::NamedNode(p)) => {
                NamedNodePattern::NamedNode(p.clone())
            }
            VariableOrPropertyPath::PropertyPath(_) => {
                return self
                    .fatal("Reifiers can only be used on triples and not on property paths");
            }
        };
        let mut current: Option<TermPattern> = None;
        loop {
            self.ws();
            let Some(part) = self.annotation_part()? else {
                break;
            };
            let to_emit = match part {
                AnnotationPart::Reifier(id) => {
                    let previous = current.take();
                    current = Some(id.unwrap_or_else(|| self.fresh_bnode().into()));
                    previous
                }
                AnnotationPart::Block => {
                    let reifier = match current.take() {
                        Some(r) => r,
                        None => self.fresh_bnode().into(),
                    };
                    self.ws();
                    self.enter_recursion()?;
                    let result = self.property_list_path_not_empty(&reifier, patterns);
                    self.leave_recursion();
                    result?;
                    self.ws();
                    self.eat("|}")?;
                    Some(reifier)
                }
            };
            if let Some(reifier) = to_emit {
                patterns.push(TripleOrPathPattern::Triple(reifies(
                    reifier, subject, &predicate, object,
                )));
            }
        }
        if let Some(reifier) = current {
            patterns.push(TripleOrPathPattern::Triple(reifies(
                reifier, subject, &predicate, object,
            )));
        }
        Ok(())
    }

    /// Whether an annotation (a reifier or an annotation block) starts at the cursor.
    fn at_annotation(&self) -> bool {
        matches!(self.peek_char(), Some('~')) || self.rest().starts_with("{|")
    }

    /// Parses one `Reifier` or the opening `{|` of an annotation block.
    fn annotation_part(&mut self) -> R<Option<AnnotationPart>> {
        if self.try_str("{|") {
            return Ok(Some(AnnotationPart::Block));
        }
        match self.reifier()? {
            Some(id) => Ok(Some(AnnotationPart::Reifier(id))),
            None => Ok(None),
        }
    }

    // --- helpers shared with the SPARQL 1.1 productions ---------------------

    /// Builds a directional language-tagged literal.
    pub(crate) fn directional_literal(
        value: String,
        language: String,
        direction: BaseDirection,
    ) -> Literal {
        Literal::new_directional_language_tagged_literal_unchecked(value, language, direction)
    }

    /// Whether a variable-free triple term (`<<(`) starts at the cursor.
    pub(crate) fn at_triple_term(&self) -> bool {
        self.rest().starts_with("<<(")
    }

    /// Whether a reified triple (`<<` not followed by `(`) starts at the cursor.
    pub(crate) fn at_reified_triple(&self) -> bool {
        self.rest().starts_with("<<") && !self.at_triple_term()
    }
}

/// `reifier rdf:reifies <<( subject predicate object )>>`
fn reifies(
    reifier: TermPattern,
    subject: &TermPattern,
    predicate: &NamedNodePattern,
    object: &TermPattern,
) -> TriplePattern {
    TriplePattern::new(
        reifier,
        rdf::REIFIES.into_owned(),
        TriplePattern::new(subject.clone(), predicate.clone(), object.clone()),
    )
}
