//! A recursive-descent parser for the XPath 1.0 grammar.
//!
//! The productions are taken straight from
//! [XPath 1.0 Appendix A](https://www.w3.org/TR/1999/REC-xpath-19991116/#exprlex),
//! bottom-up: `OrExpr` → `AndExpr` → `EqualityExpr` → `RelationalExpr` →
//! `AdditiveExpr` → `MultiplicativeExpr` → `UnaryExpr` → `UnionExpr` →
//! `PathExpr` → `FilterExpr` → `PrimaryExpr`, with `LocationPath` folded into
//! `PathExpr`.
//!
//! Abbreviations are expanded here rather than in the evaluator: `//` becomes
//! `/descendant-or-self::node()/`, `.` becomes `self::node()`, `..` becomes
//! `parent::node()`, and a step with no axis gets `child::`.

use crate::xpath::ast::{Axis, BinaryOp, Expr, Name, NodeTest, Step};
use crate::xpath::lexer::{Token, tokenize};

/// How deeply an expression may nest before the parser gives up.
///
/// The grammar is parsed by recursive descent, so a `(((…)))` chain — or a
/// chain of predicates, or of function arguments — costs one run of stack
/// frames per level. A `<ds:Transform>` carries attacker-supplied text, and an
/// unbounded descent turns a kilobyte of parentheses into a stack overflow,
/// which aborts the process rather than raising an error. Every expression the
/// XML Signature Recommendations use nests a handful of levels at most, so a
/// bound this generous costs nothing real.
pub(crate) const MAX_DEPTH: usize = 128;

/// Parses an XPath expression.
///
/// # Errors
///
/// Returns a human-readable message describing where the expression stops
/// being valid XPath, or says that it nests deeper than [`MAX_DEPTH`].
pub(crate) fn parse(input: &str) -> Result<Expr, String> {
    let tokens = tokenize(input)?;
    let mut parser = Parser {
        tokens: &tokens,
        position: 0,
        depth: 0,
    };
    let expression = parser.parse_expr()?;
    if parser.position < parser.tokens.len() {
        return Err(format!(
            "unexpected `{}` after a complete XPath expression",
            parser.peek_description()
        ));
    }
    Ok(expression)
}

struct Parser<'a> {
    tokens: &'a [Token],
    position: usize,
    /// How many nested productions are currently on the stack.
    depth: usize,
}

impl Parser<'_> {
    /// Runs `descend` one level deeper, refusing to go past [`MAX_DEPTH`].
    fn nested<T>(
        &mut self,
        descend: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.depth >= MAX_DEPTH {
            return Err(format!(
                "the XPath expression nests more than {MAX_DEPTH} levels deep"
            ));
        }
        self.depth += 1;
        let result = descend(self);
        self.depth -= 1;
        result
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn peek_description(&self) -> String {
        self.peek()
            .map_or_else(|| "end of expression".to_owned(), Token::to_string)
    }

    fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.position);
        if token.is_some() {
            self.position += 1;
        }
        token
    }

    fn eat(&mut self, expected: &Token) -> bool {
        if self.peek() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        if self.eat(expected) {
            Ok(())
        } else {
            Err(format!(
                "expected `{expected}`, found `{}`",
                self.peek_description()
            ))
        }
    }

    // OrExpr ::= AndExpr | OrExpr 'or' AndExpr
    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.nested(Self::parse_or)
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_and()?;
        while self.eat(&Token::OperatorName("or")) {
            let right = self.parse_and()?;
            left = Expr::Binary(BinaryOp::Or, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_equality()?;
        while self.eat(&Token::OperatorName("and")) {
            let right = self.parse_equality()?;
            left = Expr::Binary(BinaryOp::And, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_relational()?;
        loop {
            let operator = match self.peek() {
                Some(Token::Equal) => BinaryOp::Equal,
                Some(Token::NotEqual) => BinaryOp::NotEqual,
                _ => return Ok(left),
            };
            self.position += 1;
            let right = self.parse_relational()?;
            left = Expr::Binary(operator, Box::new(left), Box::new(right));
        }
    }

    fn parse_relational(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_additive()?;
        loop {
            let operator = match self.peek() {
                Some(Token::Less) => BinaryOp::Less,
                Some(Token::LessOrEqual) => BinaryOp::LessOrEqual,
                Some(Token::Greater) => BinaryOp::Greater,
                Some(Token::GreaterOrEqual) => BinaryOp::GreaterOrEqual,
                _ => return Ok(left),
            };
            self.position += 1;
            let right = self.parse_additive()?;
            left = Expr::Binary(operator, Box::new(left), Box::new(right));
        }
    }

    fn parse_additive(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let operator = match self.peek() {
                Some(Token::Plus) => BinaryOp::Add,
                Some(Token::Minus) => BinaryOp::Subtract,
                _ => return Ok(left),
            };
            self.position += 1;
            let right = self.parse_multiplicative()?;
            left = Expr::Binary(operator, Box::new(left), Box::new(right));
        }
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_unary()?;
        loop {
            let operator = match self.peek() {
                Some(Token::Star) => BinaryOp::Multiply,
                Some(Token::OperatorName("div")) => BinaryOp::Divide,
                Some(Token::OperatorName("mod")) => BinaryOp::Modulo,
                _ => return Ok(left),
            };
            self.position += 1;
            let right = self.parse_unary()?;
            left = Expr::Binary(operator, Box::new(left), Box::new(right));
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if self.eat(&Token::Minus) {
            // `- - - …` recurses on itself, so it is bounded too.
            let inner = self.nested(Self::parse_unary)?;
            return Ok(Expr::Negate(Box::new(inner)));
        }
        self.parse_union()
    }

    fn parse_union(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_path()?;
        while self.eat(&Token::Pipe) {
            let right = self.parse_path()?;
            left = Expr::Union(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// `PathExpr ::= LocationPath | FilterExpr (('/' | '//') RelativeLocationPath)?`
    fn parse_path(&mut self) -> Result<Expr, String> {
        if self.starts_filter_expr() {
            let primary = self.parse_primary()?;
            let predicates = self.parse_predicates()?;
            let mut steps = Vec::new();
            if self.eat(&Token::SlashSlash) {
                steps.push(Expr::descendant_or_self_step());
                steps.extend(self.parse_relative_path()?);
            } else if self.eat(&Token::Slash) {
                steps.extend(self.parse_relative_path()?);
            }
            // With nothing filtering it, a `PrimaryExpr` is the expression: a
            // parenthesised sub-expression, a literal and a function call all
            // keep their own type rather than being coerced to a node-set.
            if predicates.is_empty() && steps.is_empty() {
                return Ok(primary);
            }
            return Ok(Expr::Filter {
                primary: Box::new(primary),
                predicates,
                steps,
            });
        }
        self.parse_location_path()
    }

    /// Whether the next token starts a `PrimaryExpr` rather than a step.
    ///
    /// `FunctionName ::= QName - NodeType` in
    /// [XPath §3.7](https://www.w3.org/TR/1999/REC-xpath-19991116/#exprlex): the
    /// four node type names are reserved, so `comment()` is always a node test
    /// and never a call to a function called `comment`.
    fn starts_filter_expr(&self) -> bool {
        match self.peek() {
            Some(
                Token::Variable { .. } | Token::LeftParen | Token::Literal(_) | Token::Number(_),
            ) => true,
            Some(Token::FunctionName { prefix, local }) => {
                prefix.is_some()
                    || !matches!(
                        local.as_str(),
                        "node" | "text" | "comment" | "processing-instruction"
                    )
            }
            _ => false,
        }
    }

    fn parse_location_path(&mut self) -> Result<Expr, String> {
        if self.eat(&Token::SlashSlash) {
            let mut steps = vec![Expr::descendant_or_self_step()];
            steps.extend(self.parse_relative_path()?);
            return Ok(Expr::Path {
                absolute: true,
                steps,
            });
        }
        if self.eat(&Token::Slash) {
            // `/` alone selects the root node.
            let steps = if self.starts_step() {
                self.parse_relative_path()?
            } else {
                Vec::new()
            };
            return Ok(Expr::Path {
                absolute: true,
                steps,
            });
        }
        Ok(Expr::Path {
            absolute: false,
            steps: self.parse_relative_path()?,
        })
    }

    fn starts_step(&self) -> bool {
        matches!(
            self.peek(),
            Some(
                Token::Dot
                    | Token::DotDot
                    | Token::At
                    | Token::AxisName(_)
                    | Token::Name { .. }
                    | Token::NameTestStar
                    | Token::NameTestPrefixStar(_)
                    | Token::FunctionName { .. }
            )
        )
    }

    fn parse_relative_path(&mut self) -> Result<Vec<Step>, String> {
        let mut steps = vec![self.parse_step()?];
        loop {
            if self.eat(&Token::SlashSlash) {
                steps.push(Expr::descendant_or_self_step());
                steps.push(self.parse_step()?);
            } else if self.eat(&Token::Slash) {
                steps.push(self.parse_step()?);
            } else {
                return Ok(steps);
            }
        }
    }

    fn parse_step(&mut self) -> Result<Step, String> {
        if self.eat(&Token::Dot) {
            return Ok(Expr::self_step());
        }
        if self.eat(&Token::DotDot) {
            return Ok(Step {
                axis: Axis::Parent,
                node_test: NodeTest::AnyNode,
                predicates: Vec::new(),
            });
        }
        let axis = if self.eat(&Token::At) {
            Axis::Attribute
        } else if let Some(Token::AxisName(name)) = self.peek() {
            let axis =
                Axis::from_name(name).ok_or_else(|| format!("`{name}` is not an XPath axis"))?;
            self.position += 1;
            self.expect(&Token::ColonColon)?;
            axis
        } else {
            Axis::Child
        };
        let node_test = self.parse_node_test()?;
        let predicates = self.parse_predicates()?;
        Ok(Step {
            axis,
            node_test,
            predicates,
        })
    }

    fn parse_node_test(&mut self) -> Result<NodeTest, String> {
        match self.advance().cloned() {
            Some(Token::NameTestStar) => Ok(NodeTest::AnyName),
            Some(Token::NameTestPrefixStar(prefix)) => Ok(NodeTest::AnyNameInNamespace(prefix)),
            Some(Token::Name { prefix, local }) => Ok(NodeTest::Name(Name { prefix, local })),
            Some(Token::FunctionName {
                prefix: None,
                local,
            }) => {
                self.expect(&Token::LeftParen)?;
                let test = match local.as_str() {
                    "node" => NodeTest::AnyNode,
                    "text" => NodeTest::Text,
                    "comment" => NodeTest::Comment,
                    "processing-instruction" => {
                        let target = match self.peek().cloned() {
                            Some(Token::Literal(target)) => {
                                self.position += 1;
                                Some(target)
                            }
                            _ => None,
                        };
                        NodeTest::ProcessingInstruction(target)
                    }
                    other => return Err(format!("`{other}()` is not an XPath node type")),
                };
                self.expect(&Token::RightParen)?;
                Ok(test)
            }
            other => Err(format!(
                "expected a node test, found `{}`",
                other.map_or_else(|| "end of expression".to_owned(), |token| token.to_string())
            )),
        }
    }

    fn parse_predicates(&mut self) -> Result<Vec<Expr>, String> {
        let mut predicates = Vec::new();
        while self.eat(&Token::LeftBracket) {
            predicates.push(self.parse_expr()?);
            self.expect(&Token::RightBracket)?;
        }
        Ok(predicates)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.advance().cloned() {
            Some(Token::Variable { prefix, local }) => Ok(Expr::Variable(Name { prefix, local })),
            Some(Token::Literal(text)) => Ok(Expr::Literal(text)),
            Some(Token::Number(value)) => Ok(Expr::Number(value)),
            Some(Token::LeftParen) => {
                let inner = self.parse_expr()?;
                self.expect(&Token::RightParen)?;
                Ok(inner)
            }
            Some(Token::FunctionName { prefix, local }) => {
                self.expect(&Token::LeftParen)?;
                let mut arguments = Vec::new();
                if !self.eat(&Token::RightParen) {
                    loop {
                        arguments.push(self.parse_expr()?);
                        if self.eat(&Token::Comma) {
                            continue;
                        }
                        self.expect(&Token::RightParen)?;
                        break;
                    }
                }
                Ok(Expr::Function {
                    name: Name { prefix, local },
                    arguments,
                })
            }
            other => Err(format!(
                "expected a primary expression, found `{}`",
                other.map_or_else(|| "end of expression".to_owned(), |token| token.to_string())
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_canonical_document_expression_parses() -> Result<(), String> {
        let expression = parse("(//. | //@* | //namespace::*)")?;
        assert!(matches!(expression, Expr::Union(_, _)));
        Ok(())
    }

    #[test]
    fn abbreviations_expand() -> Result<(), String> {
        let Expr::Path { absolute, steps } = parse("//a/../@b")? else {
            panic!("expected a location path");
        };
        assert!(absolute);
        assert_eq!(steps.len(), 4);
        assert_eq!(
            steps.first().map(|step| step.axis),
            Some(Axis::DescendantOrSelf)
        );
        assert_eq!(steps.get(1).map(|step| step.axis), Some(Axis::Child));
        assert_eq!(steps.get(2).map(|step| step.axis), Some(Axis::Parent));
        assert_eq!(steps.get(3).map(|step| step.axis), Some(Axis::Attribute));
        Ok(())
    }

    #[test]
    fn a_lone_slash_is_the_root() -> Result<(), String> {
        assert_eq!(
            parse("/")?,
            Expr::Path {
                absolute: true,
                steps: Vec::new()
            }
        );
        Ok(())
    }

    #[test]
    fn precedence_follows_the_grammar() -> Result<(), String> {
        // `1 + 2 * 3` must group as `1 + (2 * 3)`.
        let expression = parse("1 + 2 * 3")?;
        let Expr::Binary(BinaryOp::Add, left, right) = expression else {
            panic!("expected an addition at the top");
        };
        assert_eq!(*left, Expr::Number(1.0));
        assert!(matches!(*right, Expr::Binary(BinaryOp::Multiply, _, _)));

        // `a or b and c` must group as `a or (b and c)`.
        let expression = parse("a or b and c")?;
        assert!(matches!(expression, Expr::Binary(BinaryOp::Or, _, _)));
        Ok(())
    }

    #[test]
    fn the_filter2_enveloped_example_parses() -> Result<(), String> {
        let expression = parse("here()/ancestor::dsig:Signature[1]")?;
        let Expr::Filter {
            primary,
            predicates,
            steps,
        } = expression
        else {
            panic!("expected a filter expression");
        };
        assert!(matches!(*primary, Expr::Function { .. }));
        assert!(predicates.is_empty());
        assert_eq!(steps.len(), 1);
        assert_eq!(steps.first().map(|step| step.predicates.len()), Some(1));
        Ok(())
    }

    #[test]
    fn node_tests_parse() -> Result<(), String> {
        assert!(parse("processing-instruction('php')").is_ok());
        assert!(parse("child::text()").is_ok());
        assert!(parse("comment()").is_ok());
        assert!(parse("ds:*").is_ok());
        Ok(())
    }

    /// A `<ds:Transform>` may carry any text at all, so a parenthesis chain must
    /// come back as an error rather than as a stack overflow — an overflow
    /// aborts the process and cannot be caught by a verifying service.
    #[test]
    fn deeply_nested_parentheses_are_rejected_instead_of_overflowing() {
        for levels in [700_usize, 10_000, 100_000] {
            let expression = format!("{}true(){}", "(".repeat(levels), ")".repeat(levels));
            let message = parse(&expression).expect_err("a chain that deep must be refused");
            assert!(
                message.contains("nests more than"),
                "unexpected message: {message}"
            );
        }
    }

    #[test]
    fn other_nesting_forms_are_bounded_too() {
        let predicates = format!("a{}{}", "[a".repeat(1000), "]".repeat(1000));
        assert!(parse(&predicates).is_err());
        let arguments = format!("{}true(){}", "boolean(".repeat(1000), ")".repeat(1000));
        assert!(parse(&arguments).is_err());
        let minuses = format!("{}1", "-".repeat(100_000));
        assert!(parse(&minuses).is_err());
    }

    #[test]
    fn nesting_just_inside_the_bound_still_parses() -> Result<(), String> {
        let levels = MAX_DEPTH - 8;
        let expression = format!("{}true(){}", "(".repeat(levels), ")".repeat(levels));
        assert!(matches!(parse(&expression)?, Expr::Function { .. }));
        Ok(())
    }

    #[test]
    fn malformed_expressions_are_rejected() {
        assert!(parse("a[").is_err());
        assert!(parse("()").is_err());
        assert!(parse("a b").is_err());
        assert!(parse("foo()bar").is_err());
        assert!(parse("child::bogus()").is_err());
    }
}
