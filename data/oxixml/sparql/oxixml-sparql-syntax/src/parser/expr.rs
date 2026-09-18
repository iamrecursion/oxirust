//! Expression productions: the precedence chain, built-in calls and aggregates.

use super::{Parser, R, alt};
use crate::algebra::{AggregateExpression, Expression};
use crate::term::{NamedNode, Variable};
use crate::vocab::sparql;
use oxixml_model::OxString;
use std::collections::BTreeMap;

impl Parser<'_> {
    // --- precedence chain ---------------------------------------------------

    pub(crate) fn expression(&mut self) -> R<Expression> {
        // Every nested expression (parentheses, function and built-in arguments) goes through
        // this entry point, so guarding it bounds the whole precedence chain's recursion.
        self.enter_recursion()?;
        let e = self.conditional_or_expression();
        self.leave_recursion();
        e
    }

    /// Parses an expression in a context where aggregates are forbidden (`FILTER`, `BIND` and
    /// `GROUP BY`).
    pub(crate) fn expression_without_aggregates(&mut self) -> R<Expression> {
        let saved = std::mem::replace(&mut self.aggregates_allowed, false);
        let e = self.expression();
        self.aggregates_allowed = saved;
        e
    }

    /// Parses a `Constraint` in a context where aggregates are forbidden (`FILTER`).
    pub(crate) fn constraint_without_aggregates(&mut self) -> R<Expression> {
        let saved = std::mem::replace(&mut self.aggregates_allowed, false);
        let e = self.constraint();
        self.aggregates_allowed = saved;
        e
    }

    /// `ConditionalOrExpression ::= ConditionalAndExpression ( '||' ConditionalAndExpression )*`
    ///
    /// A flat chain of `n + 1` `||`-joined operands folds into a tree `n` levels deep (`Or(Or(Or(
    /// a, b), c), d)`, …), so — exactly like a genuinely nested construct — each additional level
    /// charges [`Parser::enter_recursion`] before it is parsed. See
    /// [`Parser::leave_recursion_by`] for why the charge happens before, not after, and why it is
    /// released once for the whole chain rather than per element.
    fn conditional_or_expression(&mut self) -> R<Expression> {
        let mut e = self.conditional_and_expression()?;
        let mut charged = 0_u32;
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_str("||") {
                self.pos = save;
                break;
            }
            self.ws();
            if let Err(err) = self.enter_recursion() {
                self.leave_recursion_by(charged);
                return Err(err);
            }
            charged += 1;
            match self.conditional_and_expression() {
                Ok(r) => e = Expression::Or(Box::new(e), Box::new(r)),
                Err(err) => {
                    self.leave_recursion_by(charged);
                    return Err(err);
                }
            }
        }
        self.leave_recursion_by(charged);
        Ok(e)
    }

    /// `ConditionalAndExpression ::= ValueLogical ( '&&' ValueLogical )*`
    ///
    /// See [`Parser::conditional_or_expression`]: the same left-deep fold, charged the same way.
    fn conditional_and_expression(&mut self) -> R<Expression> {
        let mut e = self.value_logical()?;
        let mut charged = 0_u32;
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_str("&&") {
                self.pos = save;
                break;
            }
            self.ws();
            if let Err(err) = self.enter_recursion() {
                self.leave_recursion_by(charged);
                return Err(err);
            }
            charged += 1;
            match self.value_logical() {
                Ok(r) => e = Expression::And(Box::new(e), Box::new(r)),
                Err(err) => {
                    self.leave_recursion_by(charged);
                    return Err(err);
                }
            }
        }
        self.leave_recursion_by(charged);
        Ok(e)
    }

    fn value_logical(&mut self) -> R<Expression> {
        self.relational_expression()
    }

    fn relational_expression(&mut self) -> R<Expression> {
        let a = self.additive_expression()?;
        self.ws();
        let save = self.pos;
        if self.try_str("=") {
            self.ws();
            let b = self.additive_expression()?;
            return Ok(Expression::FunctionCall(sparql::EQUALS, vec![a, b]));
        }
        if self.try_str("!=") {
            self.ws();
            let b = self.additive_expression()?;
            return Ok(Expression::FunctionCall(sparql::NOT_EQUALS, vec![a, b]));
        }
        // The longest-token rule: when a full `IRIREF` starts here, `<` is not an operator.
        if !self.at_iriref() {
            if self.try_str("<=") {
                self.ws();
                let b = self.additive_expression()?;
                return Ok(Expression::FunctionCall(
                    sparql::LESS_THAN_OR_EQUAL,
                    vec![a, b],
                ));
            }
            if self.try_str("<") {
                self.ws();
                let b = self.additive_expression()?;
                return Ok(Expression::FunctionCall(sparql::LESS_THAN, vec![a, b]));
            }
        }
        if self.try_str(">=") {
            self.ws();
            let b = self.additive_expression()?;
            return Ok(Expression::FunctionCall(
                sparql::GREATER_THAN_OR_EQUAL,
                vec![a, b],
            ));
        }
        if self.try_str(">") {
            self.ws();
            let b = self.additive_expression()?;
            return Ok(Expression::FunctionCall(sparql::GREATER_THAN, vec![a, b]));
        }
        self.pos = save;
        if self.try_kw("IN") {
            self.ws();
            let list = self.expression_list()?;
            return Ok(Expression::In(Box::new(a), list));
        }
        let save = self.pos;
        if self.try_kw("NOT") {
            self.ws();
            if self.try_kw("IN") {
                self.ws();
                let list = self.expression_list()?;
                return Ok(Expression::FunctionCall(
                    sparql::LOGICAL_NOT,
                    vec![Expression::In(Box::new(a), list)],
                ));
            }
            self.pos = save;
        }
        Ok(a)
    }

    /// `AdditiveExpression ::= MultiplicativeExpression ( '+' MultiplicativeExpression | '-'
    /// MultiplicativeExpression | … )*`
    ///
    /// See [`Parser::conditional_or_expression`]: the same left-deep fold, charged the same way.
    fn additive_expression(&mut self) -> R<Expression> {
        let mut e = self.multiplicative_expression()?;
        let mut charged = 0_u32;
        loop {
            self.ws();
            let save = self.pos;
            let name = if self.try_str("+") {
                sparql::ADD
            } else if self.try_str("-") {
                sparql::SUBTRACT
            } else {
                self.pos = save;
                break;
            };
            self.ws();
            if let Err(err) = self.enter_recursion() {
                self.leave_recursion_by(charged);
                return Err(err);
            }
            charged += 1;
            match self.multiplicative_expression() {
                Ok(r) => e = Expression::FunctionCall(name, vec![e, r]),
                Err(err) => {
                    self.leave_recursion_by(charged);
                    return Err(err);
                }
            }
        }
        self.leave_recursion_by(charged);
        Ok(e)
    }

    /// `MultiplicativeExpression ::= UnaryExpression ( '*' UnaryExpression | '/'
    /// UnaryExpression | … )*`
    ///
    /// See [`Parser::conditional_or_expression`]: the same left-deep fold, charged the same way.
    fn multiplicative_expression(&mut self) -> R<Expression> {
        let mut e = self.unary_expression()?;
        let mut charged = 0_u32;
        loop {
            self.ws();
            let save = self.pos;
            let name = if self.try_str("*") {
                sparql::MULTIPLY
            } else if self.try_str("/") {
                sparql::DIVIDE
            } else {
                self.pos = save;
                break;
            };
            self.ws();
            if let Err(err) = self.enter_recursion() {
                self.leave_recursion_by(charged);
                return Err(err);
            }
            charged += 1;
            match self.unary_expression() {
                Ok(r) => e = Expression::FunctionCall(name, vec![e, r]),
                Err(err) => {
                    self.leave_recursion_by(charged);
                    return Err(err);
                }
            }
        }
        self.leave_recursion_by(charged);
        Ok(e)
    }

    fn unary_expression(&mut self) -> R<Expression> {
        if self.try_str("!") {
            self.ws();
            let e = self.primary_expression()?;
            Ok(Expression::FunctionCall(sparql::LOGICAL_NOT, vec![e]))
        } else if self.try_str("+") {
            self.ws();
            let e = self.primary_expression()?;
            Ok(Expression::FunctionCall(sparql::UNARY_PLUS, vec![e]))
        } else if self.try_str("-") {
            self.ws();
            let e = self.primary_expression()?;
            Ok(Expression::FunctionCall(sparql::UNARY_MINUS, vec![e]))
        } else {
            self.primary_expression()
        }
    }

    fn primary_expression(&mut self) -> R<Expression> {
        #[cfg(feature = "sparql-12")]
        if self.at_triple_term() {
            return self.expr_triple_term();
        }
        alt!(
            self,
            self.bracketed_expression(),
            self.iri_or_function(),
            self.var().map(Expression::from),
            self.rdf_literal().map(Expression::from),
            self.numeric_literal().map(Expression::from),
            self.boolean_literal().map(Expression::from),
            self.builtin_call(),
        )
    }

    pub(crate) fn bracketed_expression(&mut self) -> R<Expression> {
        self.eat("(")?;
        self.ws();
        let e = self.expression()?;
        self.ws();
        self.eat(")")?;
        Ok(e)
    }

    fn iri_or_function(&mut self) -> R<Expression> {
        let iri = self.iri()?;
        self.ws();
        let save = self.pos;
        match self.arg_list(&iri) {
            Ok((distinct, args)) => self.build_function_call(iri, distinct, args),
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                Ok(Expression::NamedNode(iri))
            }
        }
    }

    pub(crate) fn function_call(&mut self) -> R<Expression> {
        let iri = self.iri()?;
        self.ws();
        let (distinct, args) = self.arg_list(&iri)?;
        self.build_function_call(iri, distinct, args)
    }

    /// Turns a parsed `iri ArgList` into either a plain function call or, when the IRI was
    /// registered with [`SparqlParser::with_custom_aggregate_function`], an aggregate.
    ///
    /// [`SparqlParser::with_custom_aggregate_function`]: crate::SparqlParser::with_custom_aggregate_function
    fn build_function_call(
        &mut self,
        iri: NamedNode,
        distinct: bool,
        mut args: Vec<Expression>,
    ) -> R<Expression> {
        if !self.custom_aggregate_functions.contains(&iri) {
            return Ok(Expression::FunctionCall(iri, args));
        }
        if args.len() != 1 {
            return self.fatal("A custom aggregate function takes exactly one argument");
        }
        let Some(expr) = args.pop() else {
            return self.fatal("A custom aggregate function takes exactly one argument");
        };
        let aggregate = AggregateExpression::FunctionCall {
            name: iri,
            expr,
            distinct,
            scalarvals: BTreeMap::new(),
        };
        let variable = self.new_aggregation(aggregate)?;
        Ok(Expression::Variable(variable))
    }

    /// `ArgList ::= NIL | '(' 'DISTINCT'? Expression ( ',' Expression )* ')'`
    ///
    /// Returns whether `DISTINCT` was given together with the arguments.
    fn arg_list(&mut self, name: &NamedNode) -> R<(bool, Vec<Expression>)> {
        let save = self.pos;
        if self.nil().is_ok() {
            return Ok((false, Vec::new()));
        }
        self.pos = save;
        self.eat("(")?;
        self.ws();
        let distinct = self.try_kw("DISTINCT");
        self.ws();
        // The arguments of a custom aggregate are aggregated expressions, so they may not
        // themselves contain an aggregate.
        let aggregated = self.in_aggregate || self.custom_aggregate_functions.contains(name);
        let outer = std::mem::replace(&mut self.in_aggregate, aggregated);
        let args = self.arg_list_items();
        self.in_aggregate = outer;
        let args = args?;
        self.ws();
        self.eat(")")?;
        Ok((distinct, args))
    }

    /// The comma-separated argument expressions of an `ArgList`.
    fn arg_list_items(&mut self) -> R<Vec<Expression>> {
        let mut args = vec![self.expression()?];
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_str(",") {
                self.pos = save;
                break;
            }
            self.ws();
            args.push(self.expression()?);
        }
        Ok(args)
    }

    fn expression_list(&mut self) -> R<Vec<Expression>> {
        let save = self.pos;
        if self.nil().is_ok() {
            return Ok(Vec::new());
        }
        self.pos = save;
        self.eat("(")?;
        self.ws();
        let mut args = vec![self.expression()?];
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_str(",") {
                self.pos = save;
                break;
            }
            self.ws();
            args.push(self.expression()?);
        }
        self.ws();
        self.eat(")")?;
        Ok(args)
    }

    pub(crate) fn constraint(&mut self) -> R<Expression> {
        alt!(
            self,
            self.bracketed_expression(),
            self.function_call(),
            self.builtin_call(),
        )
    }

    // --- built-in calls -----------------------------------------------------

    pub(crate) fn builtin_call(&mut self) -> R<Expression> {
        // Aggregates take precedence (COUNT/SUM/...).
        let save = self.pos;
        match self.aggregate() {
            Ok(agg) => {
                if self.in_aggregate {
                    // `COUNT(SUM(?x))` and friends: aggregates may not be nested.
                    return self.fatal("Aggregate functions cannot be nested");
                }
                let v = self.new_aggregation(agg)?;
                return Ok(Expression::Variable(v));
            }
            Err(e) if e.fatal => return Err(e),
            Err(_) => self.pos = save,
        }
        alt!(
            self,
            self.exists_func(),
            self.not_exists_func(),
            self.bound_call(),
            self.if_call(),
            self.coalesce_call(),
            self.regex_call(),
            self.substr_call(),
            self.replace_call(),
            self.concat_call(),
            self.unary_builtin("STR", sparql::STR),
            self.binary_builtin("LANGMATCHES", sparql::LANG_MATCHES),
            self.unary_builtin("LANG", sparql::LANG),
            self.unary_builtin("DATATYPE", sparql::DATATYPE),
            self.unary_builtin("IRI", sparql::IRI),
            self.unary_builtin("URI", sparql::URI),
            self.bnode_call(),
            self.nullary_builtin("RAND", sparql::RAND),
            self.unary_builtin("ABS", sparql::ABS),
            self.unary_builtin("CEIL", sparql::CEIL),
            self.unary_builtin("FLOOR", sparql::FLOOR),
            self.unary_builtin("ROUND", sparql::ROUND),
            self.unary_builtin("STRLEN", sparql::STRLEN),
            self.unary_builtin("UCASE", sparql::UCASE),
            self.unary_builtin("LCASE", sparql::LCASE),
            self.unary_builtin("ENCODE_FOR_URI", sparql::ENCODE_FOR_URI),
            self.binary_builtin("CONTAINS", sparql::CONTAINS),
            self.binary_builtin("STRSTARTS", sparql::STRSTARTS),
            self.binary_builtin("STRENDS", sparql::STRENDS),
            self.binary_builtin("STRBEFORE", sparql::STRBEFORE),
            self.binary_builtin("STRAFTER", sparql::STRAFTER),
            self.unary_builtin("YEAR", sparql::YEAR),
            self.unary_builtin("MONTH", sparql::MONTH),
            self.unary_builtin("DAY", sparql::DAY),
            self.unary_builtin("HOURS", sparql::HOURS),
            self.unary_builtin("MINUTES", sparql::MINUTES),
            self.unary_builtin("SECONDS", sparql::SECONDS),
            self.unary_builtin("TIMEZONE", sparql::TIMEZONE),
            self.unary_builtin("TZ", sparql::TZ),
            self.nullary_builtin("NOW", sparql::NOW),
            self.nullary_builtin("UUID", sparql::UUID),
            self.nullary_builtin("STRUUID", sparql::STRUUID),
            self.unary_builtin("MD5", sparql::MD5),
            self.unary_builtin("SHA1", sparql::SHA1),
            self.unary_builtin("SHA256", sparql::SHA256),
            self.unary_builtin("SHA384", sparql::SHA384),
            self.unary_builtin("SHA512", sparql::SHA512),
            self.binary_builtin("STRLANG", sparql::STRLANG),
            self.binary_builtin("STRDT", sparql::STRDT),
            self.binary_builtin("sameTerm", sparql::SAME_TERM),
            self.unary_builtin("isIRI", sparql::IS_IRI),
            self.unary_builtin("isURI", sparql::IS_URI),
            self.unary_builtin("isBLANK", sparql::IS_BLANK),
            self.unary_builtin("isLITERAL", sparql::IS_LITERAL),
            self.unary_builtin("isNUMERIC", sparql::IS_NUMERIC),
            self.sparql_12_builtin(),
        )
    }

    /// The built-ins added by SPARQL 1.2, all of them no-ops without the `sparql-12` feature.
    #[cfg(feature = "sparql-12")]
    fn sparql_12_builtin(&mut self) -> R<Expression> {
        alt!(
            self,
            self.unary_builtin("LANGDIR", sparql::LANGDIR),
            self.ternary_builtin("STRLANGDIR", sparql::STRLANGDIR),
            self.unary_builtin("hasLANGDIR", sparql::HAS_LANGDIR),
            self.unary_builtin("hasLANG", sparql::HAS_LANG),
            self.unary_builtin("isTRIPLE", sparql::IS_TRIPLE),
            self.ternary_builtin("TRIPLE", sparql::TRIPLE),
            self.unary_builtin("SUBJECT", sparql::SUBJECT),
            self.unary_builtin("PREDICATE", sparql::PREDICATE),
            self.unary_builtin("OBJECT", sparql::OBJECT),
        )
    }

    #[cfg(not(feature = "sparql-12"))]
    fn sparql_12_builtin(&mut self) -> R<Expression> {
        self.mk_err("expected a built-in call")
    }

    /// A three-argument built-in such as `TRIPLE(?s, ?p, ?o)`.
    #[cfg(feature = "sparql-12")]
    fn ternary_builtin(&mut self, kw: &str, name: NamedNode) -> R<Expression> {
        self.kw(kw)?;
        self.ws();
        self.eat("(")?;
        self.ws();
        let a = self.expression()?;
        self.ws();
        self.eat(",")?;
        self.ws();
        let b = self.expression()?;
        self.ws();
        self.eat(",")?;
        self.ws();
        let c = self.expression()?;
        self.ws();
        self.eat(")")?;
        Ok(Expression::FunctionCall(name, vec![a, b, c]))
    }

    fn unary_builtin(&mut self, kw: &str, name: NamedNode) -> R<Expression> {
        self.kw(kw)?;
        self.ws();
        self.eat("(")?;
        self.ws();
        let e = self.expression()?;
        self.ws();
        self.eat(")")?;
        Ok(Expression::FunctionCall(name, vec![e]))
    }

    fn binary_builtin(&mut self, kw: &str, name: NamedNode) -> R<Expression> {
        self.kw(kw)?;
        self.ws();
        self.eat("(")?;
        self.ws();
        let a = self.expression()?;
        self.ws();
        self.eat(",")?;
        self.ws();
        let b = self.expression()?;
        self.ws();
        self.eat(")")?;
        Ok(Expression::FunctionCall(name, vec![a, b]))
    }

    fn nullary_builtin(&mut self, kw: &str, name: NamedNode) -> R<Expression> {
        self.kw(kw)?;
        self.ws();
        self.nil()?;
        Ok(Expression::FunctionCall(name, Vec::new()))
    }

    fn bound_call(&mut self) -> R<Expression> {
        self.kw("BOUND")?;
        self.ws();
        self.eat("(")?;
        self.ws();
        let v = self.var()?;
        self.ws();
        self.eat(")")?;
        Ok(Expression::Bound(v))
    }

    fn bnode_call(&mut self) -> R<Expression> {
        self.kw("BNODE")?;
        self.ws();
        let save = self.pos;
        if self.nil().is_ok() {
            return Ok(Expression::FunctionCall(sparql::BNODE, Vec::new()));
        }
        self.pos = save;
        self.eat("(")?;
        self.ws();
        let e = self.expression()?;
        self.ws();
        self.eat(")")?;
        Ok(Expression::FunctionCall(sparql::BNODE, vec![e]))
    }

    fn if_call(&mut self) -> R<Expression> {
        self.kw("IF")?;
        self.ws();
        self.eat("(")?;
        self.ws();
        let a = self.expression()?;
        self.ws();
        self.eat(",")?;
        self.ws();
        let b = self.expression()?;
        self.ws();
        self.eat(",")?;
        self.ws();
        let c = self.expression()?;
        self.ws();
        self.eat(")")?;
        Ok(Expression::If(Box::new(a), Box::new(b), Box::new(c)))
    }

    fn coalesce_call(&mut self) -> R<Expression> {
        self.kw("COALESCE")?;
        self.ws();
        let args = self.expression_list()?;
        Ok(Expression::Coalesce(args))
    }

    fn concat_call(&mut self) -> R<Expression> {
        self.kw("CONCAT")?;
        self.ws();
        let args = self.expression_list()?;
        Ok(Expression::FunctionCall(sparql::CONCAT, args))
    }

    fn substr_call(&mut self) -> R<Expression> {
        self.kw("SUBSTR")?;
        self.ws();
        self.eat("(")?;
        self.ws();
        let a = self.expression()?;
        self.ws();
        self.eat(",")?;
        self.ws();
        let b = self.expression()?;
        self.ws();
        let mut args = vec![a, b];
        if self.try_str(",") {
            self.ws();
            args.push(self.expression()?);
            self.ws();
        }
        self.eat(")")?;
        Ok(Expression::FunctionCall(sparql::SUBSTR, args))
    }

    fn replace_call(&mut self) -> R<Expression> {
        self.kw("REPLACE")?;
        self.ws();
        self.eat("(")?;
        self.ws();
        let a = self.expression()?;
        self.ws();
        self.eat(",")?;
        self.ws();
        let b = self.expression()?;
        self.ws();
        self.eat(",")?;
        self.ws();
        let c = self.expression()?;
        self.ws();
        let mut args = vec![a, b, c];
        if self.try_str(",") {
            self.ws();
            args.push(self.expression()?);
            self.ws();
        }
        self.eat(")")?;
        Ok(Expression::FunctionCall(sparql::REPLACE, args))
    }

    fn regex_call(&mut self) -> R<Expression> {
        self.kw("REGEX")?;
        self.ws();
        self.eat("(")?;
        self.ws();
        let a = self.expression()?;
        self.ws();
        self.eat(",")?;
        self.ws();
        let b = self.expression()?;
        self.ws();
        let mut args = vec![a, b];
        if self.try_str(",") {
            self.ws();
            args.push(self.expression()?);
            self.ws();
        }
        self.eat(")")?;
        Ok(Expression::FunctionCall(sparql::REGEX, args))
    }

    fn exists_func(&mut self) -> R<Expression> {
        self.kw("EXISTS")?;
        self.ws();
        let p = self.group_graph_pattern()?;
        Ok(Expression::Exists(Box::new(p)))
    }

    fn not_exists_func(&mut self) -> R<Expression> {
        self.kw("NOT")?;
        self.ws();
        self.kw("EXISTS")?;
        self.ws();
        let p = self.group_graph_pattern()?;
        Ok(Expression::FunctionCall(
            sparql::LOGICAL_NOT,
            vec![Expression::Exists(Box::new(p))],
        ))
    }

    // --- aggregates ---------------------------------------------------------

    fn aggregate(&mut self) -> R<AggregateExpression> {
        if self.try_kw("COUNT") {
            self.ws();
            self.eat("(")?;
            self.ws();
            let distinct = self.try_kw("DISTINCT");
            self.ws();
            if self.try_str("*") {
                self.ws();
                self.eat(")")?;
                return Ok(AggregateExpression::CountSolutions { distinct });
            }
            let expr = self.aggregated_expression()?;
            self.ws();
            self.eat(")")?;
            return Ok(AggregateExpression::FunctionCall {
                name: sparql::AGG_COUNT,
                expr,
                distinct,
                scalarvals: BTreeMap::new(),
            });
        }
        if let Some(name) = self.simple_aggregate_name() {
            self.ws();
            self.eat("(")?;
            self.ws();
            let distinct = self.try_kw("DISTINCT");
            self.ws();
            let expr = self.aggregated_expression()?;
            self.ws();
            self.eat(")")?;
            return Ok(AggregateExpression::FunctionCall {
                name,
                expr,
                distinct,
                scalarvals: BTreeMap::new(),
            });
        }
        if self.try_kw("GROUP_CONCAT") {
            self.ws();
            self.eat("(")?;
            self.ws();
            let distinct = self.try_kw("DISTINCT");
            self.ws();
            let expr = self.aggregated_expression()?;
            self.ws();
            let mut scalarvals = BTreeMap::new();
            if self.try_str(";") {
                self.ws();
                self.kw("SEPARATOR")?;
                self.ws();
                self.eat("=")?;
                self.ws();
                let sep = self.string()?;
                self.ws();
                scalarvals.insert(OxString::from("separator".to_owned()), OxString::from(sep));
            }
            self.eat(")")?;
            return Ok(AggregateExpression::FunctionCall {
                name: sparql::AGG_GROUP_CONCAT,
                expr,
                distinct,
                scalarvals,
            });
        }
        self.mk_err("expected aggregate")
    }

    /// Parses the argument of an aggregate, forbidding further aggregates inside it.
    fn aggregated_expression(&mut self) -> R<Expression> {
        let outer = std::mem::replace(&mut self.in_aggregate, true);
        let result = self.expression();
        self.in_aggregate = outer;
        result
    }

    fn simple_aggregate_name(&mut self) -> Option<NamedNode> {
        if self.try_kw("SUM") {
            Some(sparql::AGG_SUM)
        } else if self.try_kw("MIN") {
            Some(sparql::AGG_MIN)
        } else if self.try_kw("MAX") {
            Some(sparql::AGG_MAX)
        } else if self.try_kw("AVG") {
            Some(sparql::AGG_AVG)
        } else if self.try_kw("SAMPLE") {
            Some(sparql::AGG_SAMPLE)
        } else {
            None
        }
    }

    fn new_aggregation(&mut self, agg: AggregateExpression) -> R<Variable> {
        if !self.aggregates_allowed {
            // Hoisting the aggregate into the enclosing `Group` would put its variable below the
            // node defining it, so such queries must be rejected instead.
            return self.fatal("Aggregate functions cannot be used in FILTER, BIND or GROUP BY");
        }
        let existing = match self.aggregates.last() {
            Some(aggregates) => aggregates
                .iter()
                .find_map(|(v, a)| (a == &agg).then_some(v))
                .cloned(),
            None => return self.fatal("Unexpected aggregate outside of a query projection"),
        };
        if let Some(v) = existing {
            return Ok(v);
        }
        let new_var = self.fresh_var();
        if let Some(aggregates) = self.aggregates.last_mut() {
            aggregates.push((new_var.clone(), agg));
        }
        Ok(new_var)
    }
}
