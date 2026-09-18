//! CTL (Color Transform Language) parser.
//!
//! Consumes a token stream produced by [`crate::ctl_lexer::tokenize`] and
//! builds an abstract syntax tree (AST) suitable for evaluation by
//! [`crate::ctl_interpreter`].
//!
//! ## Grammar (simplified)
//!
//! ```text
//! program       ::= function_def*
//! function_def  ::= type ident '(' param_list ')' block
//! param_list    ::= (param (',' param)*)?
//! param         ::= ('input'|'output') 'varying' type ident
//! block         ::= '{' statement* '}'
//! statement     ::= var_decl | assignment | if_stmt | return_stmt | expr_stmt
//! var_decl      ::= type ident ('=' expr)? ';'
//! assignment    ::= ident '=' expr ';'
//! if_stmt       ::= 'if' '(' expr ')' block ('else' block)?
//! return_stmt   ::= 'return' expr? ';'
//! expr          ::= or_expr
//! or_expr       ::= and_expr ('||' and_expr)*
//! and_expr      ::= comparison ('&&' comparison)*
//! comparison    ::= additive (('==' | '!=' | '<' | '>' | '<=' | '>=') additive)?
//! additive      ::= multiplicative (('+' | '-') multiplicative)*
//! multiplicative::= unary (('*' | '/') unary)*
//! unary         ::= ('-' | '!') unary | primary
//! primary       ::= float_lit | int_lit | bool_lit | ident | call | '(' expr ')'
//! call          ::= ident '(' arg_list ')'
//! ```

use crate::ctl_lexer::CtlToken;
use crate::error::ColorError;

// ─────────────────────────────────────────────────────────────────────────────
// AST node types
// ─────────────────────────────────────────────────────────────────────────────

/// Binary operator variants.
#[derive(Debug, Clone, PartialEq)]
pub enum CtlBinOp {
    /// Addition (`+`)
    Add,
    /// Subtraction (`-`)
    Sub,
    /// Multiplication (`*`)
    Mul,
    /// Division (`/`)
    Div,
    /// Equal (`==`)
    Eq,
    /// Not-equal (`!=`)
    Ne,
    /// Less-than (`<`)
    Lt,
    /// Greater-than (`>`)
    Gt,
    /// Less-than-or-equal (`<=`)
    Le,
    /// Greater-than-or-equal (`>=`)
    Ge,
    /// Logical AND (`&&`)
    And,
    /// Logical OR (`||`)
    Or,
}

/// Unary operator variants.
#[derive(Debug, Clone, PartialEq)]
pub enum CtlUnOp {
    /// Numeric negation (`-`)
    Neg,
    /// Logical NOT (`!`)
    Not,
}

/// A CTL AST expression node.
#[derive(Debug, Clone)]
pub enum CtlExpr {
    /// A floating-point literal.
    FloatLit(f64),
    /// A boolean literal.
    BoolLit(bool),
    /// A variable read.
    Var(String),
    /// Binary operation.
    BinOp {
        /// Operator.
        op: CtlBinOp,
        /// Left-hand side.
        lhs: Box<CtlExpr>,
        /// Right-hand side.
        rhs: Box<CtlExpr>,
    },
    /// Unary operation.
    UnOp {
        /// Operator.
        op: CtlUnOp,
        /// Operand.
        operand: Box<CtlExpr>,
    },
    /// Function call.
    Call {
        /// Callee name (built-in or user-defined).
        name: String,
        /// Argument expressions.
        args: Vec<CtlExpr>,
    },
    /// Variable assignment statement (`ident = expr`).
    Assign {
        /// Target variable name.
        target: String,
        /// Value expression.
        value: Box<CtlExpr>,
    },
    /// Variable declaration (`float x = expr` or `float x`).
    VarDecl {
        /// Variable name.
        name: String,
        /// Optional initializer.
        init: Option<Box<CtlExpr>>,
    },
    /// If/else statement.
    If {
        /// Condition expression.
        cond: Box<CtlExpr>,
        /// Then-branch block.
        then_body: Box<CtlExpr>,
        /// Optional else-branch block.
        else_body: Option<Box<CtlExpr>>,
    },
    /// Return statement with optional value.
    Return(Option<Box<CtlExpr>>),
    /// A block of statements.
    Block(Vec<CtlExpr>),
}

/// A CTL function parameter descriptor.
#[derive(Debug, Clone)]
pub struct CtlParam {
    /// Parameter name.
    pub name: String,
    /// `true` if declared as `output`, `false` if `input`.
    pub is_output: bool,
}

/// The parsed representation of a CTL `main()` function.
#[derive(Debug, Clone)]
pub struct CtlMainFunc {
    /// Ordered parameter list.
    pub params: Vec<CtlParam>,
    /// Body of the function as a [`CtlExpr::Block`].
    pub body: CtlExpr,
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser
// ─────────────────────────────────────────────────────────────────────────────

struct Parser<'t> {
    tokens: &'t [CtlToken],
    pos: usize,
}

impl<'t> Parser<'t> {
    fn new(tokens: &'t [CtlToken]) -> Self {
        Self { tokens, pos: 0 }
    }

    // ── Token stream primitives ───────────────────────────────────────────────

    fn peek(&self) -> Option<&CtlToken> {
        self.tokens.get(self.pos)
    }

    fn peek2(&self) -> Option<&CtlToken> {
        self.tokens.get(self.pos + 1)
    }

    fn advance(&mut self) -> Option<&CtlToken> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    /// Consume the next token and return it; error if end-of-tokens.
    #[allow(dead_code)]
    fn expect_any(&mut self) -> Result<&CtlToken, ColorError> {
        self.advance()
            .ok_or_else(|| ColorError::Parse("unexpected end of token stream".to_string()))
    }

    /// Consume the next token if it matches `expected`; otherwise error.
    fn expect(&mut self, expected: &CtlToken) -> Result<(), ColorError> {
        match self.peek() {
            Some(tok) if tok == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(tok) => Err(ColorError::Parse(format!(
                "expected {expected:?}, got {tok:?}"
            ))),
            None => Err(ColorError::Parse(format!(
                "expected {expected:?}, got end of input"
            ))),
        }
    }

    /// Consume and return the identifier at the current position; error otherwise.
    fn expect_ident(&mut self) -> Result<String, ColorError> {
        match self.advance() {
            Some(CtlToken::Ident(name)) => Ok(name.clone()),
            Some(tok) => Err(ColorError::Parse(format!(
                "expected identifier, got {tok:?}"
            ))),
            None => Err(ColorError::Parse(
                "expected identifier, got end of input".to_string(),
            )),
        }
    }

    /// Return `true` and consume if the next token matches `tok`.
    fn eat(&mut self, tok: &CtlToken) -> bool {
        if self.peek() == Some(tok) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    // ── Type parsing ─────────────────────────────────────────────────────────

    /// Consume a type keyword and return its string. Currently supports
    /// `float`, `int`, `bool`, `void`.
    fn parse_type(&mut self) -> Result<String, ColorError> {
        match self.peek() {
            Some(CtlToken::KwFloat) => {
                self.pos += 1;
                Ok("float".to_string())
            }
            Some(CtlToken::KwInt) => {
                self.pos += 1;
                Ok("int".to_string())
            }
            Some(CtlToken::KwBool) => {
                self.pos += 1;
                Ok("bool".to_string())
            }
            Some(CtlToken::KwVoid) => {
                self.pos += 1;
                Ok("void".to_string())
            }
            Some(tok) => Err(ColorError::Parse(format!("expected type, got {tok:?}"))),
            None => Err(ColorError::Parse(
                "expected type, got end of input".to_string(),
            )),
        }
    }

    // ── Parameter list ────────────────────────────────────────────────────────

    fn parse_param_list(&mut self) -> Result<Vec<CtlParam>, ColorError> {
        let mut params = Vec::new();
        self.expect(&CtlToken::LParen)?;

        // Empty parameter list
        if self.eat(&CtlToken::RParen) {
            return Ok(params);
        }

        loop {
            // Each parameter looks like: [input|output] [varying|uniform] type ident
            // The qualifiers are optional in general CTL but we parse them if present.
            let is_output = match self.peek() {
                Some(CtlToken::KwOutput) => {
                    self.pos += 1;
                    true
                }
                Some(CtlToken::KwInput) => {
                    self.pos += 1;
                    false
                }
                _ => false,
            };

            // Optional `varying` / `uniform`
            if matches!(self.peek(), Some(CtlToken::KwVarying | CtlToken::KwUniform)) {
                self.pos += 1;
            }

            // Type (we don't use it in the interpreter, but must consume)
            self.parse_type()?;

            let name = self.expect_ident()?;
            params.push(CtlParam { name, is_output });

            if !self.eat(&CtlToken::Comma) {
                break;
            }
        }

        self.expect(&CtlToken::RParen)?;
        Ok(params)
    }

    // ── Block ─────────────────────────────────────────────────────────────────

    fn parse_block(&mut self) -> Result<CtlExpr, ColorError> {
        self.expect(&CtlToken::LBrace)?;
        let mut stmts = Vec::new();
        while self.peek() != Some(&CtlToken::RBrace) {
            if self.peek().is_none() {
                return Err(ColorError::Parse(
                    "unexpected end of input inside block".to_string(),
                ));
            }
            stmts.push(self.parse_statement()?);
        }
        self.expect(&CtlToken::RBrace)?;
        Ok(CtlExpr::Block(stmts))
    }

    // ── Statement ─────────────────────────────────────────────────────────────

    fn parse_statement(&mut self) -> Result<CtlExpr, ColorError> {
        match self.peek() {
            // Variable declaration: type ident ...
            Some(CtlToken::KwFloat | CtlToken::KwInt | CtlToken::KwBool) => self.parse_var_decl(),
            // if statement
            Some(CtlToken::KwIf) => self.parse_if_stmt(),
            // return statement
            Some(CtlToken::KwReturn) => self.parse_return_stmt(),
            // Assignment: ident = expr ;  or expression statement
            Some(CtlToken::Ident(_)) => {
                // Look-ahead: if next-next is `=` (not `==`), it's an assignment.
                if matches!(self.peek2(), Some(CtlToken::Assign)) {
                    self.parse_assignment()
                } else {
                    // Expression statement
                    let expr = self.parse_expr()?;
                    self.expect(&CtlToken::Semi)?;
                    Ok(expr)
                }
            }
            Some(tok) => Err(ColorError::Parse(format!(
                "unexpected token at start of statement: {tok:?}"
            ))),
            None => Err(ColorError::Parse(
                "unexpected end of input in statement".to_string(),
            )),
        }
    }

    fn parse_var_decl(&mut self) -> Result<CtlExpr, ColorError> {
        // Consume type keyword
        self.parse_type()?;
        let name = self.expect_ident()?;
        let init = if self.eat(&CtlToken::Assign) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        self.expect(&CtlToken::Semi)?;
        Ok(CtlExpr::VarDecl { name, init })
    }

    fn parse_assignment(&mut self) -> Result<CtlExpr, ColorError> {
        let target = self.expect_ident()?;
        self.expect(&CtlToken::Assign)?;
        let value = self.parse_expr()?;
        self.expect(&CtlToken::Semi)?;
        Ok(CtlExpr::Assign {
            target,
            value: Box::new(value),
        })
    }

    fn parse_if_stmt(&mut self) -> Result<CtlExpr, ColorError> {
        self.expect(&CtlToken::KwIf)?;
        self.expect(&CtlToken::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(&CtlToken::RParen)?;
        let then_body = self.parse_block()?;
        let else_body = if self.eat(&CtlToken::KwElse) {
            Some(Box::new(self.parse_block()?))
        } else {
            None
        };
        Ok(CtlExpr::If {
            cond: Box::new(cond),
            then_body: Box::new(then_body),
            else_body,
        })
    }

    fn parse_return_stmt(&mut self) -> Result<CtlExpr, ColorError> {
        self.expect(&CtlToken::KwReturn)?;
        let val = if self.peek() == Some(&CtlToken::Semi) {
            None
        } else {
            Some(Box::new(self.parse_expr()?))
        };
        self.expect(&CtlToken::Semi)?;
        Ok(CtlExpr::Return(val))
    }

    // ── Expressions (recursive descent) ──────────────────────────────────────

    fn parse_expr(&mut self) -> Result<CtlExpr, ColorError> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> Result<CtlExpr, ColorError> {
        let mut lhs = self.parse_and_expr()?;
        while self.eat(&CtlToken::Or) {
            let rhs = self.parse_and_expr()?;
            lhs = CtlExpr::BinOp {
                op: CtlBinOp::Or,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_and_expr(&mut self) -> Result<CtlExpr, ColorError> {
        let mut lhs = self.parse_comparison()?;
        while self.eat(&CtlToken::And) {
            let rhs = self.parse_comparison()?;
            lhs = CtlExpr::BinOp {
                op: CtlBinOp::And,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_comparison(&mut self) -> Result<CtlExpr, ColorError> {
        let lhs = self.parse_additive()?;
        let op = match self.peek() {
            Some(CtlToken::Eq) => CtlBinOp::Eq,
            Some(CtlToken::Ne) => CtlBinOp::Ne,
            Some(CtlToken::Lt) => CtlBinOp::Lt,
            Some(CtlToken::Gt) => CtlBinOp::Gt,
            Some(CtlToken::Le) => CtlBinOp::Le,
            Some(CtlToken::Ge) => CtlBinOp::Ge,
            _ => return Ok(lhs),
        };
        self.pos += 1;
        let rhs = self.parse_additive()?;
        Ok(CtlExpr::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        })
    }

    fn parse_additive(&mut self) -> Result<CtlExpr, ColorError> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Some(CtlToken::Plus) => CtlBinOp::Add,
                Some(CtlToken::Minus) => CtlBinOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_multiplicative()?;
            lhs = CtlExpr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<CtlExpr, ColorError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(CtlToken::Star) => CtlBinOp::Mul,
                Some(CtlToken::Slash) => CtlBinOp::Div,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_unary()?;
            lhs = CtlExpr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<CtlExpr, ColorError> {
        match self.peek() {
            Some(CtlToken::Minus) => {
                self.pos += 1;
                let operand = self.parse_unary()?;
                Ok(CtlExpr::UnOp {
                    op: CtlUnOp::Neg,
                    operand: Box::new(operand),
                })
            }
            Some(CtlToken::Bang) => {
                self.pos += 1;
                let operand = self.parse_unary()?;
                Ok(CtlExpr::UnOp {
                    op: CtlUnOp::Not,
                    operand: Box::new(operand),
                })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<CtlExpr, ColorError> {
        match self.peek().cloned() {
            Some(CtlToken::FloatLit(v)) => {
                self.pos += 1;
                Ok(CtlExpr::FloatLit(v))
            }
            Some(CtlToken::IntLit(v)) => {
                self.pos += 1;
                // Promote int literals to float in expressions
                Ok(CtlExpr::FloatLit(v as f64))
            }
            Some(CtlToken::BoolLit(b)) => {
                self.pos += 1;
                Ok(CtlExpr::BoolLit(b))
            }
            Some(CtlToken::Ident(name)) => {
                self.pos += 1;
                // Is this a function call?
                if self.eat(&CtlToken::LParen) {
                    let mut args = Vec::new();
                    if self.peek() != Some(&CtlToken::RParen) {
                        args.push(self.parse_expr()?);
                        while self.eat(&CtlToken::Comma) {
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(&CtlToken::RParen)?;
                    Ok(CtlExpr::Call { name, args })
                } else {
                    Ok(CtlExpr::Var(name))
                }
            }
            Some(CtlToken::LParen) => {
                self.pos += 1;
                let inner = self.parse_expr()?;
                self.expect(&CtlToken::RParen)?;
                Ok(inner)
            }
            Some(tok) => Err(ColorError::Parse(format!(
                "expected expression, got {tok:?}"
            ))),
            None => Err(ColorError::Parse(
                "expected expression, got end of input".to_string(),
            )),
        }
    }

    // ── Top-level: find and parse `main` function ─────────────────────────────

    /// Scan through the token stream looking for a function named `main`,
    /// parse its parameter list and body, and return a [`CtlMainFunc`].
    ///
    /// Any functions before `main` are skipped (their bodies are consumed but
    /// not evaluated).
    fn parse_main(&mut self) -> Result<CtlMainFunc, ColorError> {
        loop {
            // Expect a return type
            self.parse_type()?;
            // Expect a function name
            let name = self.expect_ident()?;
            // Parse parameter list
            let params = self.parse_param_list()?;

            if name == "main" {
                let body = self.parse_block()?;
                return Ok(CtlMainFunc { params, body });
            }

            // Not `main` — skip its body by brace-counting
            self.skip_braced_body()?;
        }
    }

    /// Skip a `{ ... }` body (possibly nested) without building an AST.
    fn skip_braced_body(&mut self) -> Result<(), ColorError> {
        self.expect(&CtlToken::LBrace)?;
        let mut depth = 1usize;
        loop {
            match self.advance() {
                None => {
                    return Err(ColorError::Parse(
                        "unexpected end of input while skipping function body".to_string(),
                    ))
                }
                Some(CtlToken::LBrace) => depth += 1,
                Some(CtlToken::RBrace) => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a sequence of CTL tokens and locate the `main()` function.
///
/// Returns a [`CtlMainFunc`] containing the parameter list and the body AST.
///
/// # Errors
///
/// Returns [`ColorError::Parse`] if the token stream does not contain a
/// syntactically valid `main` function, or if any parse rule is violated.
pub fn parse_main_function(tokens: &[CtlToken]) -> Result<CtlMainFunc, ColorError> {
    let mut parser = Parser::new(tokens);
    parser
        .parse_main()
        .map_err(|e| ColorError::Parse(format!("CTL parse error: {e}")))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctl_lexer::tokenize;

    fn parse(src: &str) -> CtlMainFunc {
        let toks = tokenize(src).expect("lexer failed");
        parse_main_function(&toks).expect("parser failed")
    }

    #[test]
    fn test_empty_main() {
        let src = "void main() {}";
        let func = parse(src);
        assert!(func.params.is_empty());
        assert!(matches!(func.body, CtlExpr::Block(ref v) if v.is_empty()));
    }

    #[test]
    fn test_main_with_params() {
        let src = "void main(output varying float rOut, input varying float rIn) {}";
        let func = parse(src);
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.params[0].name, "rOut");
        assert!(func.params[0].is_output);
        assert_eq!(func.params[1].name, "rIn");
        assert!(!func.params[1].is_output);
    }

    #[test]
    fn test_simple_assignment() {
        let src = "void main() { rOut = rIn; }";
        let func = parse(src);
        match &func.body {
            CtlExpr::Block(stmts) => {
                assert_eq!(stmts.len(), 1);
                assert!(matches!(&stmts[0], CtlExpr::Assign { target, .. } if target == "rOut"));
            }
            _ => panic!("expected block"),
        }
    }

    #[test]
    fn test_var_decl_with_init() {
        let src = "void main() { float x = 1.0; }";
        let func = parse(src);
        match &func.body {
            CtlExpr::Block(stmts) => {
                assert!(
                    matches!(&stmts[0], CtlExpr::VarDecl { name, init: Some(_) } if name == "x")
                );
            }
            _ => panic!("expected block"),
        }
    }

    #[test]
    fn test_if_else() {
        let src = "void main() { if (x > 0.0) { rOut = 1.0; } else { rOut = 0.0; } }";
        let func = parse(src);
        match &func.body {
            CtlExpr::Block(stmts) => {
                assert!(matches!(
                    &stmts[0],
                    CtlExpr::If {
                        else_body: Some(_),
                        ..
                    }
                ));
            }
            _ => panic!("expected block"),
        }
    }

    #[test]
    fn test_function_call() {
        let src = "void main() { rOut = clamp(rIn, 0.0, 1.0); }";
        let func = parse(src);
        match &func.body {
            CtlExpr::Block(stmts) => match &stmts[0] {
                CtlExpr::Assign { value, .. } => {
                    assert!(
                        matches!(value.as_ref(), CtlExpr::Call { name, args } if name == "clamp" && args.len() == 3)
                    );
                }
                _ => panic!("expected assign"),
            },
            _ => panic!("expected block"),
        }
    }

    #[test]
    fn test_arithmetic_precedence() {
        // 2.0 + 3.0 * 4.0 should parse as 2.0 + (3.0 * 4.0)
        let src = "void main() { rOut = 2.0 + 3.0 * 4.0; }";
        let func = parse(src);
        match &func.body {
            CtlExpr::Block(stmts) => match &stmts[0] {
                CtlExpr::Assign { value, .. } => {
                    assert!(matches!(
                        value.as_ref(),
                        CtlExpr::BinOp {
                            op: CtlBinOp::Add,
                            ..
                        }
                    ));
                }
                _ => panic!("expected assign"),
            },
            _ => panic!("expected block"),
        }
    }

    #[test]
    fn test_unary_negation() {
        let src = "void main() { rOut = -rIn; }";
        let func = parse(src);
        match &func.body {
            CtlExpr::Block(stmts) => match &stmts[0] {
                CtlExpr::Assign { value, .. } => {
                    assert!(matches!(
                        value.as_ref(),
                        CtlExpr::UnOp {
                            op: CtlUnOp::Neg,
                            ..
                        }
                    ));
                }
                _ => panic!("expected assign"),
            },
            _ => panic!("expected block"),
        }
    }

    #[test]
    fn test_skip_helper_function_before_main() {
        let src = r#"
            float helper(float x) {
                return x;
            }
            void main() { rOut = 1.0; }
        "#;
        let func = parse(src);
        assert!(matches!(func.body, CtlExpr::Block(_)));
    }

    #[test]
    fn test_return_statement() {
        let src = "void main() { return; }";
        let func = parse(src);
        match &func.body {
            CtlExpr::Block(stmts) => {
                assert!(matches!(&stmts[0], CtlExpr::Return(None)));
            }
            _ => panic!("expected block"),
        }
    }

    #[test]
    fn test_missing_main_error() {
        let toks = tokenize("float helper() {}").expect("tokenize helper fn");
        let result = parse_main_function(&toks);
        assert!(result.is_err());
    }
}
