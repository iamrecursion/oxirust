//! The recursive-descent parser for the XPath 3.1 grammar.
//!
//! One function per production, in the order the specification numbers them, with the operator
//! precedence encoded in the call chain rather than in a table:
//!
//! ```text
//! ExprSingle → OrExpr → AndExpr → ComparisonExpr → StringConcatExpr → RangeExpr
//!            → AdditiveExpr → MultiplicativeExpr → UnionExpr → IntersectExceptExpr
//!            → InstanceofExpr → TreatExpr → CastableExpr → CastExpr → ArrowExpr
//!            → UnaryExpr → SimpleMapExpr → PathExpr → StepExpr
//! ```
//!
//! Because XPath has no reserved words, every keyword operator is recognized by *position*: the
//! parser asks "is the next token the name `div`?" only where an operator may appear, so
//! `child::div`, `a div b` and `div(1)` all resolve correctly with no lexer state.
//!
//! The parser also does the whole of the crate's static analysis, since XPath has no separate
//! static-typing pass: it resolves every lexical `QName` against the
//! [`StaticContext`](crate::StaticContext) (raising `XPST0081` for an unbound prefix), checks
//! that every `$name` is in scope (`XPST0008`), checks that every statically named function
//! exists at the arity it is called with (`XPST0017`), and rejects every construct that is not
//! part of the [`XPathVersion`] being compiled (`XPST0003`).

mod steps;
mod typing;

use crate::ast::{
    BinaryOp, Expr, GeneralOp, InlineFunction, KeySpecifier, NodeOp, ValueOp, VarBinding,
};
use crate::context::StaticContext;
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::lexer::{Spanned, Token, is_ncname, tokenize_with_references};
use crate::parser::typing::CastTarget;
use crate::types::{ItemType, Occurrence, SequenceType, SingleType};
use crate::version::XPathVersion;
use oxixml_infoset::{AtomicType, AtomicValue, ExpandedName};

/// The namespace of the `xmlns` attributes, which no name may be in.
const XMLNS_NAMESPACE: &str = "http://www.w3.org/2000/xmlns/";

/// The sequence type `xs:numeric`, with `?` when the cast target carried one.
fn numeric_sequence_type(optional: bool) -> SequenceType {
    SequenceType::new(
        ItemType::Numeric,
        if optional {
            Occurrence::ZeroOrOne
        } else {
            Occurrence::One
        },
    )
}
use std::sync::Arc;

/// The names that may not be used as a function name in a `FunctionCall`, because the grammar
/// gives them another meaning followed by `(`.
///
/// XPath 2.0 §3.1.5 lists thirteen; a name only becomes reserved with the version that gives it
/// its other meaning, so `map()` and `array()` are ordinary (unknown, hence `XPST0017`) function
/// calls before 3.1 and `function()`, `switch()` and `namespace-node()` are before 3.0.
const RESERVED_FUNCTION_NAMES: &[&str] = &[
    "attribute",
    "comment",
    "document-node",
    "element",
    "empty-sequence",
    "if",
    "item",
    "node",
    "processing-instruction",
    "schema-attribute",
    "schema-element",
    "text",
    "typeswitch",
];

/// The names XPath 3.0 added to [`RESERVED_FUNCTION_NAMES`].
const RESERVED_IN_30: &[&str] = &["function", "namespace-node", "switch"];

/// The names XPath 3.1 added to [`RESERVED_FUNCTION_NAMES`].
const RESERVED_IN_31: &[&str] = &["array", "map"];

/// Whether `name` may not be used as a function name in `version`.
fn is_reserved_function_name(name: &str, version: XPathVersion) -> bool {
    RESERVED_FUNCTION_NAMES.contains(&name)
        || (version.at_least(XPathVersion::XPath30) && RESERVED_IN_30.contains(&name))
        || (version.at_least(XPathVersion::XPath31) && RESERVED_IN_31.contains(&name))
}

/// Parses `source` into an expression tree.
///
/// # Errors
///
/// Any of the static error codes: `XPST0003` for a syntax error or a construct the version does
/// not have, `XPST0008` for an out-of-scope variable, `XPST0017` for an unknown function,
/// `XPST0051` for an unknown atomic type, `XPST0080` for an illegal cast target, and `XPST0081`
/// for an unbound namespace prefix.
pub(crate) fn parse(
    source: &str,
    version: XPathVersion,
    context: &StaticContext,
) -> XPathResult<Expr> {
    let tokens = tokenize_with_references(source, context.is_xquery_host_language())?;
    let mut parser = Parser {
        tokens,
        index: 0,
        end: source.len(),
        version,
        context,
        scopes: Vec::new(),
        depth: 0,
    };
    let expression = parser.parse_expr()?;
    if let Some(spanned) = parser.peek_spanned() {
        return Err(XPathError::syntax(
            spanned.offset,
            format!(
                "unexpected {} after a complete expression",
                spanned.token.describe()
            ),
        ));
    }
    Ok(expression)
}

/// Parses a `SequenceType` on its own, for a host that declares types outside an expression.
///
/// # Errors
///
/// `XPST0003` when the text is not a sequence type or is followed by anything else, and the
/// static errors [`parse`] lists for the names inside it.
pub(crate) fn parse_type(source: &str, context: &StaticContext) -> XPathResult<SequenceType> {
    let tokens = tokenize_with_references(source, context.is_xquery_host_language())?;
    let mut parser = Parser {
        tokens,
        index: 0,
        end: source.len(),
        version: XPathVersion::XPath31,
        context,
        scopes: Vec::new(),
        depth: 0,
    };
    let declared = parser.parse_sequence_type()?;
    if let Some(spanned) = parser.peek_spanned() {
        return Err(XPathError::syntax(
            spanned.offset,
            format!(
                "unexpected {} after a complete sequence type",
                spanned.token.describe()
            ),
        ));
    }
    Ok(declared)
}

/// How deeply one expression may nest inside another.
///
/// The grammar is parsed by recursive descent, so a parenthesized expression, a predicate or an
/// argument list re-enters the whole precedence chain and every level costs several stack
/// frames. Without a ceiling a few hundred bytes of `((((…1…))))` overflows the stack, which is
/// a process abort (`SIGABRT`) rather than a catchable Rust panic — so a host that compiles
/// externally supplied query text could not defend against it. This mirrors the evaluator's own
/// `XPDY0130` call-depth guard.
pub(crate) const MAX_NESTING_DEPTH: usize = 64;

pub(crate) struct Parser<'a> {
    tokens: Vec<Spanned>,
    index: usize,
    end: usize,
    pub(crate) version: XPathVersion,
    pub(crate) context: &'a StaticContext,
    scopes: Vec<ExpandedName>,
    /// How many expressions are currently open around the one being parsed.
    depth: usize,
}

impl<'a> Parser<'a> {
    // --- token plumbing ------------------------------------------------------------------

    pub(crate) fn peek_spanned(&self) -> Option<&Spanned> {
        self.tokens.get(self.index)
    }

    pub(crate) fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.index).map(|spanned| &spanned.token)
    }

    pub(crate) fn peek_at(&self, ahead: usize) -> Option<&Token> {
        self.tokens
            .get(self.index + ahead)
            .map(|spanned| &spanned.token)
    }

    pub(crate) fn offset(&self) -> usize {
        self.tokens
            .get(self.index)
            .map_or(self.end, |spanned| spanned.offset)
    }

    /// The offset of the token `ahead` positions on, if there is one.
    pub(crate) fn offset_at(&self, ahead: usize) -> Option<usize> {
        self.tokens
            .get(self.index + ahead)
            .map(|spanned| spanned.offset)
    }

    /// Whether the next `count` tokens are written with nothing between them.
    ///
    /// The lexer discards whitespace and comments, so a production the grammar marks `/* ws:
    /// explicit */` — `Wildcard` and `ArgumentPlaceholder` among them — has to be recognized by
    /// comparing offsets. `pre:*` and `pre :*` produce the same three tokens; only the first
    /// spells a single `Wildcard` token, and XPath 3.1 §A.2.4.1 says a comment "may not occur
    /// within a terminal symbol".
    pub(crate) fn tokens_are_adjacent(&self, count: usize) -> bool {
        let mut expected = match self.offset_at(0) {
            Some(offset) => offset,
            None => return false,
        };
        for ahead in 0..count {
            let (Some(offset), Some(token)) = (self.offset_at(ahead), self.peek_at(ahead)) else {
                return false;
            };
            if offset != expected {
                return false;
            }
            expected = offset + token.width();
        }
        true
    }

    pub(crate) fn advance(&mut self) {
        self.index += 1;
    }

    pub(crate) fn eat(&mut self, token: &Token) -> bool {
        if self.peek() == Some(token) {
            self.advance();
            return true;
        }
        false
    }

    pub(crate) fn expect(&mut self, token: &Token) -> XPathResult<()> {
        if self.eat(token) {
            return Ok(());
        }
        Err(self.unexpected(&format!("`{}`", token.spelling())))
    }

    /// Opens one level of grammatical nesting, refusing to go past [`MAX_NESTING_DEPTH`].
    ///
    /// # Errors
    ///
    /// `XPST0003` once the ceiling is reached.
    pub(crate) fn enter_nesting(&mut self) -> XPathResult<()> {
        if self.depth >= MAX_NESTING_DEPTH {
            return Err(XPathError::syntax(
                self.offset(),
                format!("the expression nests more than {MAX_NESTING_DEPTH} levels deep"),
            ));
        }
        self.depth += 1;
        Ok(())
    }

    /// Closes one level of grammatical nesting.
    pub(crate) fn leave_nesting(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    pub(crate) fn unexpected(&self, wanted: &str) -> XPathError {
        let offset = self.offset();
        match self.peek() {
            Some(token) => XPathError::syntax(
                offset,
                format!("expected {wanted} but found {}", token.describe()),
            ),
            None => XPathError::syntax(
                offset,
                format!("expected {wanted} but the expression ends here"),
            ),
        }
    }

    /// Whether the next token is the name `word`.
    pub(crate) fn at_word(&self, word: &str) -> bool {
        matches!(self.peek(), Some(Token::Name(name)) if name == word)
    }

    /// Whether the token `ahead` positions on is the name `word`.
    pub(crate) fn at_word_ahead(&self, ahead: usize, word: &str) -> bool {
        matches!(self.peek_at(ahead), Some(Token::Name(name)) if name == word)
    }

    /// Consumes the next token when it is the name `word`.
    pub(crate) fn eat_word(&mut self, word: &str) -> bool {
        if self.at_word(word) {
            self.advance();
            return true;
        }
        false
    }

    pub(crate) fn expect_word(&mut self, word: &str) -> XPathResult<()> {
        if self.eat_word(word) {
            return Ok(());
        }
        Err(self.unexpected(&format!("`{word}`")))
    }

    /// Rejects a construct that does not exist in the version being compiled.
    pub(crate) fn require_version(&self, minimum: XPathVersion, what: &str) -> XPathResult<()> {
        if self.version.at_least(minimum) {
            return Ok(());
        }
        Err(XPathError::syntax(
            self.offset(),
            format!(
                "{what} was introduced in XPath {}, but this expression is compiled as {}",
                minimum.as_str(),
                self.version
            ),
        ))
    }

    // --- name resolution -----------------------------------------------------------------

    /// Reads an `EQName` and resolves it as an element or type name.
    pub(crate) fn parse_element_name(&mut self) -> XPathResult<ExpandedName> {
        self.parse_eqname(|context, lexical| context.resolve_element_name(lexical))
    }

    /// Reads an `EQName` and resolves it as an attribute or variable name.
    pub(crate) fn parse_unqualified_name(&mut self) -> XPathResult<ExpandedName> {
        self.parse_eqname(|context, lexical| context.resolve_unqualified_name(lexical))
    }

    /// Reads an `EQName` and resolves it as a function name.
    pub(crate) fn parse_function_name(&mut self) -> XPathResult<ExpandedName> {
        self.parse_eqname(|context, lexical| context.resolve_function_name(lexical))
    }

    fn parse_eqname(
        &mut self,
        resolve: impl Fn(&StaticContext, &str) -> XPathResult<ExpandedName>,
    ) -> XPathResult<ExpandedName> {
        let offset = self.offset();
        match self.peek().cloned() {
            Some(Token::Name(lexical)) => {
                self.advance();
                resolve(self.context, &lexical).map_err(|error| error.at(offset))
            }
            Some(Token::BracedName { uri, local }) => {
                self.require_version(XPathVersion::XPath30, "a `Q{uri}local` name")?;
                self.advance();
                // XQuery 3.1 §4.1 / XPath 3.1 §2.1.1 reserve the namespace of the `xmlns`
                // attributes: no name may be in it, however it is written. A braced URI literal
                // is the one spelling that could name it without a namespace declaration, and
                // whitespace normalization does not save it (`eqname-910`).
                if uri == XMLNS_NAMESPACE {
                    return Err(XPathError::new(
                        XPathErrorCode::Xqst0070,
                        format!("`{uri}` may not be used as the namespace of a name"),
                    )
                    .at(offset));
                }
                let namespace = (!uri.is_empty()).then_some(uri);
                Ok(ExpandedName::new(namespace, local))
            }
            _ => Err(self.unexpected("a name")),
        }
    }

    // --- variable scoping ----------------------------------------------------------------

    fn push_scope(&mut self, name: ExpandedName) {
        self.scopes.push(name);
    }

    fn pop_scopes(&mut self, count: usize) {
        self.scopes.truncate(self.scopes.len() - count);
    }

    fn check_variable(&self, name: &ExpandedName, offset: usize) -> XPathResult<()> {
        if self.context.allows_undeclared_variables()
            || self.scopes.contains(name)
            || self.context.has_variable(name)
        {
            return Ok(());
        }
        Err(XPathError::new(
            XPathErrorCode::Xpst0008,
            format!("the variable `${name}` is not in scope"),
        )
        .at(offset))
    }

    // --- productions ---------------------------------------------------------------------

    /// `Expr ::= ExprSingle ("," ExprSingle)*`
    pub(crate) fn parse_expr(&mut self) -> XPathResult<Expr> {
        let first = self.parse_expr_single()?;
        if self.peek() != Some(&Token::Comma) {
            return Ok(first);
        }
        self.require_version(XPathVersion::XPath20, "the sequence operator `,`")?;
        let mut items = vec![first];
        while self.eat(&Token::Comma) {
            items.push(self.parse_expr_single()?);
        }
        Ok(Expr::Sequence(items))
    }

    /// `ExprSingle ::= ForExpr | LetExpr | QuantifiedExpr | IfExpr | OrExpr`
    ///
    /// Every recursive re-entry into the grammar — a parenthesized expression, a predicate, an
    /// argument, a branch of `if` — passes through here, which makes it the one place the
    /// nesting depth has to be counted.
    pub(crate) fn parse_expr_single(&mut self) -> XPathResult<Expr> {
        self.enter_nesting()?;
        let parsed = self.parse_expr_single_inner();
        self.leave_nesting();
        parsed
    }

    fn parse_expr_single_inner(&mut self) -> XPathResult<Expr> {
        if self.at_word("for") && matches!(self.peek_at(1), Some(Token::Dollar)) {
            return self.parse_for_or_let(false);
        }
        if self.at_word("let") && matches!(self.peek_at(1), Some(Token::Dollar)) {
            return self.parse_for_or_let(true);
        }
        if (self.at_word("some") || self.at_word("every"))
            && matches!(self.peek_at(1), Some(Token::Dollar))
        {
            return self.parse_quantified();
        }
        if self.at_word("if") && matches!(self.peek_at(1), Some(Token::LeftParen)) {
            return self.parse_if();
        }
        self.parse_or()
    }

    fn parse_for_or_let(&mut self, is_let: bool) -> XPathResult<Expr> {
        if is_let {
            self.require_version(XPathVersion::XPath30, "a `let` expression")?;
        } else {
            self.require_version(XPathVersion::XPath20, "a `for` expression")?;
        }
        self.advance();
        let mut bindings = Vec::new();
        loop {
            self.expect(&Token::Dollar)?;
            let name = self.parse_unqualified_name()?;
            if is_let {
                self.expect(&Token::Assign)?;
            } else {
                self.expect_word("in")?;
            }
            let value = self.parse_expr_single()?;
            // A `for` binding is visible to the following bindings and to the body; a `let`
            // binding likewise. Both are pushed as soon as their value expression is parsed.
            self.push_scope(name.clone());
            bindings.push(VarBinding { name, value });
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect_word("return")?;
        let body = Box::new(self.parse_expr_single()?);
        self.pop_scopes(bindings.len());
        Ok(if is_let {
            Expr::Let { bindings, body }
        } else {
            Expr::For { bindings, body }
        })
    }

    fn parse_quantified(&mut self) -> XPathResult<Expr> {
        self.require_version(XPathVersion::XPath20, "a quantified expression")?;
        let every = self.at_word("every");
        self.advance();
        let mut bindings = Vec::new();
        loop {
            self.expect(&Token::Dollar)?;
            let name = self.parse_unqualified_name()?;
            self.expect_word("in")?;
            let value = self.parse_expr_single()?;
            self.push_scope(name.clone());
            bindings.push(VarBinding { name, value });
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect_word("satisfies")?;
        let satisfies = Box::new(self.parse_expr_single()?);
        self.pop_scopes(bindings.len());
        Ok(Expr::Quantified {
            every,
            bindings,
            satisfies,
        })
    }

    fn parse_if(&mut self) -> XPathResult<Expr> {
        self.require_version(XPathVersion::XPath20, "a conditional expression")?;
        self.advance();
        self.expect(&Token::LeftParen)?;
        let condition = Box::new(self.parse_expr()?);
        self.expect(&Token::RightParen)?;
        self.expect_word("then")?;
        let then_branch = Box::new(self.parse_expr_single()?);
        self.expect_word("else")?;
        let else_branch = Box::new(self.parse_expr_single()?);
        Ok(Expr::If {
            condition,
            then_branch,
            else_branch,
        })
    }

    /// `OrExpr ::= AndExpr ( "or" AndExpr )*`
    fn parse_or(&mut self) -> XPathResult<Expr> {
        let mut left = self.parse_and()?;
        while self.eat_word("or") {
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// `AndExpr ::= ComparisonExpr ( "and" ComparisonExpr )*`
    fn parse_and(&mut self) -> XPathResult<Expr> {
        let mut left = self.parse_comparison()?;
        while self.eat_word("and") {
            let right = self.parse_comparison()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// `ComparisonExpr ::= StringConcatExpr ( (ValueComp | GeneralComp | NodeComp) StringConcatExpr )?`
    fn parse_comparison(&mut self) -> XPathResult<Expr> {
        let left = self.parse_string_concat()?;
        if let Some(op) = self.peek_general_comparison() {
            self.advance();
            let right = self.parse_string_concat()?;
            return Ok(Expr::General {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        if let Some(op) = self.peek_node_comparison() {
            self.require_version(XPathVersion::XPath20, "a node comparison")?;
            self.advance();
            let right = self.parse_string_concat()?;
            return Ok(Expr::Node {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        if let Some(op) = self.peek_value_comparison() {
            self.require_version(XPathVersion::XPath20, "a value comparison")?;
            self.advance();
            let right = self.parse_string_concat()?;
            return Ok(Expr::Value {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn peek_general_comparison(&self) -> Option<GeneralOp> {
        match self.peek()? {
            Token::Equal => Some(GeneralOp::Equal),
            Token::NotEqual => Some(GeneralOp::NotEqual),
            Token::Less => Some(GeneralOp::Less),
            Token::LessEqual => Some(GeneralOp::LessEqual),
            Token::Greater => Some(GeneralOp::Greater),
            Token::GreaterEqual => Some(GeneralOp::GreaterEqual),
            _ => None,
        }
    }

    fn peek_node_comparison(&self) -> Option<NodeOp> {
        match self.peek()? {
            Token::Precedes => Some(NodeOp::Precedes),
            Token::Follows => Some(NodeOp::Follows),
            Token::Name(name) if name == "is" => Some(NodeOp::Is),
            _ => None,
        }
    }

    fn peek_value_comparison(&self) -> Option<ValueOp> {
        let Token::Name(name) = self.peek()? else {
            return None;
        };
        match name.as_str() {
            "eq" => Some(ValueOp::Equal),
            "ne" => Some(ValueOp::NotEqual),
            "lt" => Some(ValueOp::Less),
            "le" => Some(ValueOp::LessEqual),
            "gt" => Some(ValueOp::Greater),
            "ge" => Some(ValueOp::GreaterEqual),
            _ => None,
        }
    }

    /// `StringConcatExpr ::= RangeExpr ( "||" RangeExpr )*`
    fn parse_string_concat(&mut self) -> XPathResult<Expr> {
        let mut left = self.parse_range()?;
        while self.peek() == Some(&Token::Concat) {
            self.require_version(
                XPathVersion::XPath30,
                "the string concatenation operator `||`",
            )?;
            self.advance();
            let right = self.parse_range()?;
            left = Expr::Binary {
                op: BinaryOp::Concat,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// `RangeExpr ::= AdditiveExpr ( "to" AdditiveExpr )?`
    fn parse_range(&mut self) -> XPathResult<Expr> {
        let left = self.parse_additive()?;
        if self.at_word("to") {
            self.require_version(XPathVersion::XPath20, "the range operator `to`")?;
            self.advance();
            let right = self.parse_additive()?;
            return Ok(Expr::Binary {
                op: BinaryOp::Range,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    /// `AdditiveExpr ::= MultiplicativeExpr ( ("+" | "-") MultiplicativeExpr )*`
    fn parse_additive(&mut self) -> XPathResult<Expr> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Some(Token::Plus) => BinaryOp::Add,
                Some(Token::Minus) => BinaryOp::Subtract,
                _ => return Ok(left),
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
    }

    /// `MultiplicativeExpr ::= UnionExpr ( ("*" | "div" | "idiv" | "mod") UnionExpr )*`
    fn parse_multiplicative(&mut self) -> XPathResult<Expr> {
        let mut left = self.parse_union()?;
        loop {
            let op = match self.peek() {
                Some(Token::Star) => BinaryOp::Multiply,
                Some(Token::Name(name)) if name == "div" => BinaryOp::Divide,
                Some(Token::Name(name)) if name == "mod" => BinaryOp::Modulus,
                Some(Token::Name(name)) if name == "idiv" => {
                    self.require_version(XPathVersion::XPath20, "the operator `idiv`")?;
                    BinaryOp::IntegerDivide
                }
                _ => return Ok(left),
            };
            self.advance();
            let right = self.parse_union()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
    }

    /// `UnionExpr ::= IntersectExceptExpr ( ("union" | "|") IntersectExceptExpr )*`
    fn parse_union(&mut self) -> XPathResult<Expr> {
        let mut left = self.parse_intersect_except()?;
        loop {
            if self.peek() == Some(&Token::Pipe) || self.at_word("union") {
                self.advance();
            } else {
                return Ok(left);
            }
            let right = self.parse_intersect_except()?;
            left = Expr::Binary {
                op: BinaryOp::Union,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
    }

    /// `IntersectExceptExpr ::= InstanceofExpr ( ("intersect" | "except") InstanceofExpr )*`
    fn parse_intersect_except(&mut self) -> XPathResult<Expr> {
        let mut left = self.parse_instance_of()?;
        loop {
            let op = if self.at_word("intersect") {
                BinaryOp::Intersect
            } else if self.at_word("except") {
                BinaryOp::Except
            } else {
                return Ok(left);
            };
            self.require_version(XPathVersion::XPath20, "the operator `intersect`/`except`")?;
            self.advance();
            let right = self.parse_instance_of()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
    }

    /// `InstanceofExpr ::= TreatExpr ( "instance" "of" SequenceType )?`
    fn parse_instance_of(&mut self) -> XPathResult<Expr> {
        let left = self.parse_treat()?;
        if self.at_word("instance") && self.at_word_ahead(1, "of") {
            self.require_version(XPathVersion::XPath20, "the `instance of` operator")?;
            self.advance();
            self.advance();
            let required = self.parse_sequence_type()?;
            return Ok(Expr::InstanceOf(Box::new(left), required));
        }
        Ok(left)
    }

    /// `TreatExpr ::= CastableExpr ( "treat" "as" SequenceType )?`
    fn parse_treat(&mut self) -> XPathResult<Expr> {
        let left = self.parse_castable()?;
        if self.at_word("treat") && self.at_word_ahead(1, "as") {
            self.require_version(XPathVersion::XPath20, "the `treat as` operator")?;
            self.advance();
            self.advance();
            let required = self.parse_sequence_type()?;
            return Ok(Expr::TreatAs(Box::new(left), required));
        }
        Ok(left)
    }

    /// `CastableExpr ::= CastExpr ( "castable" "as" SingleType )?`
    fn parse_castable(&mut self) -> XPathResult<Expr> {
        let left = self.parse_cast()?;
        if self.at_word("castable") && self.at_word_ahead(1, "as") {
            self.require_version(XPathVersion::XPath20, "the `castable as` operator")?;
            self.advance();
            self.advance();
            return Ok(match self.parse_cast_target()? {
                CastTarget::Single(target) => Expr::Castable(Box::new(left), target),
                CastTarget::List(target) => Expr::CastableList(Box::new(left), target),
                // XPath 3.1 §19.3: a value is castable to a union type when it is already an
                // instance of one of its member types, or castable to one of them. `xs:double`
                // is the widest of `xs:numeric`'s three members and the one its cast produces.
                CastTarget::Numeric(optional) => Expr::Or(
                    Box::new(Expr::InstanceOf(
                        Box::new(left.clone()),
                        numeric_sequence_type(optional),
                    )),
                    Box::new(Expr::Castable(
                        Box::new(left),
                        SingleType::new(AtomicType::Double, optional),
                    )),
                ),
            });
        }
        Ok(left)
    }

    /// `CastExpr ::= ArrowExpr ( "cast" "as" SingleType )?`
    fn parse_cast(&mut self) -> XPathResult<Expr> {
        let left = self.parse_arrow()?;
        if self.at_word("cast") && self.at_word_ahead(1, "as") {
            self.require_version(XPathVersion::XPath20, "the `cast as` operator")?;
            self.advance();
            self.advance();
            return Ok(match self.parse_cast_target()? {
                CastTarget::Single(target) => Expr::Cast(Box::new(left), target),
                CastTarget::List(target) => Expr::CastList(Box::new(left), target),
                // Casting to the union type `xs:numeric` leaves a value that is already one of
                // its member types — or a subtype, so `xs:short(256)` stays an `xs:short` —
                // untouched, and otherwise casts to `xs:double`.
                CastTarget::Numeric(optional) => Expr::If {
                    condition: Box::new(Expr::InstanceOf(
                        Box::new(left.clone()),
                        numeric_sequence_type(optional),
                    )),
                    then_branch: Box::new(left.clone()),
                    else_branch: Box::new(Expr::Cast(
                        Box::new(left),
                        SingleType::new(AtomicType::Double, optional),
                    )),
                },
            });
        }
        Ok(left)
    }

    /// `ArrowExpr ::= UnaryExpr ( "=>" ArrowFunctionSpecifier ArgumentList )*`
    fn parse_arrow(&mut self) -> XPathResult<Expr> {
        let mut left = self.parse_unary()?;
        while self.peek() == Some(&Token::Arrow) {
            self.require_version(XPathVersion::XPath30, "the arrow operator `=>`")?;
            self.advance();
            left = self.parse_arrow_application(left)?;
        }
        Ok(left)
    }

    fn parse_arrow_application(&mut self, operand: Expr) -> XPathResult<Expr> {
        match self.peek() {
            Some(Token::Dollar) => {
                self.advance();
                let offset = self.offset();
                let name = self.parse_unqualified_name()?;
                self.check_variable(&name, offset)?;
                let arguments = self.parse_argument_list()?;
                Ok(dynamic_call(Expr::VarRef(name), operand, arguments))
            }
            Some(Token::LeftParen) => {
                self.advance();
                let target = self.parse_expr()?;
                self.expect(&Token::RightParen)?;
                let arguments = self.parse_argument_list()?;
                Ok(dynamic_call(target, operand, arguments))
            }
            _ => {
                let offset = self.offset();
                let name = self.parse_function_name()?;
                let arguments = self.parse_argument_list()?;
                let mut full = vec![Some(operand)];
                full.extend(arguments);
                self.build_static_call(name, full, offset)
            }
        }
    }

    /// `UnaryExpr ::= ("-" | "+")* ValueExpr`
    fn parse_unary(&mut self) -> XPathResult<Expr> {
        let mut negate = false;
        let mut signed = false;
        loop {
            match self.peek() {
                Some(Token::Minus) => {
                    negate = !negate;
                    signed = true;
                    self.advance();
                }
                Some(Token::Plus) => {
                    signed = true;
                    self.advance();
                }
                _ => break,
            }
        }
        let operand = self.parse_simple_map()?;
        if !signed {
            return Ok(operand);
        }
        Ok(Expr::Unary {
            negate,
            operand: Box::new(operand),
        })
    }

    /// `SimpleMapExpr ::= PathExpr ("!" PathExpr)*`
    fn parse_simple_map(&mut self) -> XPathResult<Expr> {
        let mut left = self.parse_path()?;
        while self.peek() == Some(&Token::Bang) {
            self.require_version(XPathVersion::XPath30, "the simple map operator `!`")?;
            self.advance();
            let right = self.parse_path()?;
            left = Expr::SimpleMap(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // --- primary expressions -------------------------------------------------------------

    /// `PostfixExpr ::= PrimaryExpr (Predicate | ArgumentList | Lookup)*`
    pub(crate) fn parse_postfix(&mut self) -> XPathResult<Expr> {
        let mut expression = self.parse_primary()?;
        loop {
            match self.peek() {
                Some(Token::LeftBracket) => {
                    self.advance();
                    let predicate = self.parse_expr()?;
                    self.expect(&Token::RightBracket)?;
                    expression = Expr::Filter(Box::new(expression), Box::new(predicate));
                }
                Some(Token::LeftParen) => {
                    self.require_version(XPathVersion::XPath30, "a dynamic function call")?;
                    let arguments = self.parse_argument_list()?;
                    expression = Expr::DynamicCall {
                        target: Box::new(expression),
                        arguments,
                    };
                }
                Some(Token::Question) => {
                    self.require_version(XPathVersion::XPath31, "the lookup operator `?`")?;
                    self.advance();
                    let key = self.parse_key_specifier()?;
                    expression = Expr::Lookup {
                        base: Some(Box::new(expression)),
                        key,
                    };
                }
                _ => return Ok(expression),
            }
        }
    }

    /// `PrimaryExpr ::= Literal | VarRef | ParenthesizedExpr | ContextItemExpr | FunctionCall
    ///                | FunctionItemExpr | MapConstructor | ArrayConstructor | UnaryLookup`
    fn parse_primary(&mut self) -> XPathResult<Expr> {
        let offset = self.offset();
        match self.peek().cloned() {
            Some(Token::Integer(text)) => {
                self.advance();
                Ok(self.numeric_expression(AtomicType::Integer, &text, offset)?)
            }
            Some(Token::Decimal(text)) => {
                self.advance();
                Ok(Expr::Literal(self.numeric_literal(
                    AtomicType::Decimal,
                    &text,
                    offset,
                )?))
            }
            Some(Token::Double(text)) => {
                self.advance();
                Ok(Expr::Literal(self.numeric_literal(
                    AtomicType::Double,
                    &text,
                    offset,
                )?))
            }
            Some(Token::Str(text)) => {
                self.advance();
                Ok(Expr::Literal(AtomicValue::string(text)))
            }
            Some(Token::Dollar) => {
                self.advance();
                let offset = self.offset();
                let name = self.parse_unqualified_name()?;
                self.check_variable(&name, offset)?;
                Ok(Expr::VarRef(name))
            }
            Some(Token::LeftParen) => {
                self.advance();
                if self.eat(&Token::RightParen) {
                    return Ok(Expr::empty_sequence());
                }
                let inner = self.parse_expr()?;
                self.expect(&Token::RightParen)?;
                Ok(inner)
            }
            Some(Token::Dot) => {
                self.advance();
                Ok(Expr::ContextItem)
            }
            Some(Token::LeftBracket) => {
                self.require_version(XPathVersion::XPath31, "an array constructor")?;
                self.advance();
                let mut members = Vec::new();
                if !self.eat(&Token::RightBracket) {
                    loop {
                        members.push(self.parse_expr_single()?);
                        if !self.eat(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(&Token::RightBracket)?;
                }
                Ok(Expr::ArrayConstructor {
                    curly: false,
                    members,
                })
            }
            Some(Token::Question) => {
                self.require_version(XPathVersion::XPath31, "the lookup operator `?`")?;
                self.advance();
                let key = self.parse_key_specifier()?;
                Ok(Expr::Lookup { base: None, key })
            }
            Some(Token::Name(name)) => self.parse_named_primary(&name, offset),
            Some(Token::BracedName { .. }) => {
                let name = self.parse_function_name()?;
                self.parse_call_or_reference(name, offset)
            }
            _ => Err(self.unexpected("an expression")),
        }
    }

    fn parse_named_primary(&mut self, name: &str, offset: usize) -> XPathResult<Expr> {
        // `function` only becomes a keyword with the inline function expressions of XPath 3.0;
        // before that it is an ordinary function name, so `function()` is a call to a function
        // that does not exist (`XPST0017`), not a malformed inline function.
        if name == "function"
            && self.version.has_function_items()
            && matches!(self.peek_at(1), Some(Token::LeftParen))
        {
            return self.parse_inline_function();
        }
        if name == "map" && matches!(self.peek_at(1), Some(Token::LeftBrace)) {
            return self.parse_map_constructor();
        }
        if name == "array" && matches!(self.peek_at(1), Some(Token::LeftBrace)) {
            return self.parse_curly_array();
        }
        // A reserved function name is reserved in a *named function reference* too: XPath 3.1
        // §3.1.6 forbids `if#0` and `element#1` exactly as it forbids `if(…)`.
        if is_reserved_function_name(name, self.version)
            && matches!(self.peek_at(1), Some(Token::LeftParen | Token::Hash))
        {
            return Err(XPathError::syntax(
                offset,
                format!("`{name}` is a reserved function name and cannot be called"),
            ));
        }
        let resolved = self.parse_function_name()?;
        self.parse_call_or_reference(resolved, offset)
    }

    fn parse_call_or_reference(&mut self, name: ExpandedName, offset: usize) -> XPathResult<Expr> {
        if self.peek() == Some(&Token::Hash) {
            self.require_version(XPathVersion::XPath30, "a named function reference")?;
            self.advance();
            let arity_offset = self.offset();
            let Some(Token::Integer(text)) = self.peek().cloned() else {
                return Err(self.unexpected("an arity"));
            };
            self.advance();
            // `NamedFunctionRef ::= EQName "#" IntegerLiteral`, and an `IntegerLiteral` is an
            // unbounded `xs:integer`: `fn:concat#340282366920938463463374607431768211456` is
            // *syntactically* a named function reference, not a syntax error. No function item
            // of that arity can be built, and F&O 3.1 gives an out-of-range numeric value its
            // own code — `FOAR0002`, which is what the suite accepts here.
            let arity: usize = text.parse().map_err(|_| {
                if text.chars().all(|digit| digit.is_ascii_digit()) {
                    XPathError::new(
                        XPathErrorCode::Foar0002,
                        format!(
                            "the arity {text} is larger than any function item this processor \
                             can build"
                        ),
                    )
                    .at(arity_offset)
                } else {
                    XPathError::syntax(arity_offset, format!("`{text}` is not a valid arity"))
                }
            })?;
            self.check_function_exists(&name, arity, offset)?;
            return Ok(Expr::NamedFunctionRef { name, arity });
        }
        if self.peek() == Some(&Token::LeftParen) {
            let arguments = self.parse_argument_list()?;
            return self.build_static_call(name, arguments, offset);
        }
        Err(XPathError::syntax(
            offset,
            format!("the name `{name}` is used where an expression is expected"),
        ))
    }

    fn build_static_call(
        &mut self,
        name: ExpandedName,
        arguments: Vec<Option<Expr>>,
        offset: usize,
    ) -> XPathResult<Expr> {
        self.check_function_exists(&name, arguments.len(), offset)?;
        if arguments.iter().any(Option::is_none) {
            self.require_version(XPathVersion::XPath30, "a partial function application")?;
            return Ok(Expr::PartialCall { name, arguments });
        }
        Ok(Expr::FunctionCall {
            name,
            arguments: arguments.into_iter().flatten().collect(),
        })
    }

    /// `ArgumentList ::= "(" (Argument ("," Argument)*)? ")"`, with `?` placeholders.
    fn parse_argument_list(&mut self) -> XPathResult<Vec<Option<Expr>>> {
        self.expect(&Token::LeftParen)?;
        let mut arguments = Vec::new();
        if self.eat(&Token::RightParen) {
            return Ok(arguments);
        }
        loop {
            if self.peek() == Some(&Token::Question)
                && matches!(self.peek_at(1), Some(Token::Comma | Token::RightParen))
            {
                self.require_version(XPathVersion::XPath30, "an argument placeholder `?`")?;
                self.advance();
                arguments.push(None);
            } else {
                arguments.push(Some(self.parse_expr_single()?));
            }
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect(&Token::RightParen)?;
        Ok(arguments)
    }

    fn parse_key_specifier(&mut self) -> XPathResult<KeySpecifier> {
        match self.peek().cloned() {
            // `KeySpecifier ::= NCName | IntegerLiteral | ParenthesizedExpr | "*"`: the name form
            // is an `NCName`, so a lexical `QName` such as `?xs:integer` is a syntax error rather
            // than a lookup of the key `xs:integer`.
            Some(Token::Name(name)) => {
                let offset = self.offset();
                self.advance();
                if !is_ncname(&name) {
                    return Err(XPathError::syntax(
                        offset,
                        format!("a lookup key must be an NCName, and `{name}` is not"),
                    ));
                }
                Ok(KeySpecifier::Name(name))
            }
            Some(Token::Integer(text)) => {
                let offset = self.offset();
                self.advance();
                let value: i64 = text.parse().map_err(|_| {
                    XPathError::syntax(offset, format!("`{text}` is not a valid lookup key"))
                })?;
                Ok(KeySpecifier::Integer(value))
            }
            Some(Token::Star) => {
                self.advance();
                Ok(KeySpecifier::Wildcard)
            }
            Some(Token::LeftParen) => {
                self.advance();
                if self.eat(&Token::RightParen) {
                    return Ok(KeySpecifier::Computed(Box::new(Expr::empty_sequence())));
                }
                let inner = self.parse_expr()?;
                self.expect(&Token::RightParen)?;
                Ok(KeySpecifier::Computed(Box::new(inner)))
            }
            _ => Err(self.unexpected("a lookup key")),
        }
    }

    fn parse_inline_function(&mut self) -> XPathResult<Expr> {
        self.require_version(XPathVersion::XPath30, "an inline function expression")?;
        self.advance();
        self.expect(&Token::LeftParen)?;
        let mut parameters = Vec::new();
        if !self.eat(&Token::RightParen) {
            loop {
                self.expect(&Token::Dollar)?;
                let name = self.parse_unqualified_name()?;
                let declared = if self.at_word("as") {
                    self.advance();
                    Some(self.parse_sequence_type()?)
                } else {
                    None
                };
                parameters.push((name, declared));
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
            self.expect(&Token::RightParen)?;
        }
        // XPath 3.1 §3.1.7: "the parameters of an inline function expression must have distinct
        // names, or a static error [err:XQST0039] is raised".
        for (index, (name, _)) in parameters.iter().enumerate() {
            if parameters[..index].iter().any(|(seen, _)| seen == name) {
                return Err(XPathError::new(
                    XPathErrorCode::Xqst0039,
                    format!("the parameter `${name}` is declared twice"),
                )
                .at(self.offset()));
            }
        }
        let result = if self.at_word("as") {
            self.advance();
            Some(self.parse_sequence_type()?)
        } else {
            None
        };
        for (name, _) in &parameters {
            self.push_scope(name.clone());
        }
        self.expect(&Token::LeftBrace)?;
        // XPath 3.0 spelled the body `EnclosedExpr ::= "{" Expr "}"`, so an empty one was a
        // syntax error; 3.1 relaxed it to `"{" Expr? "}"` and made `function(){}` the empty
        // sequence. QT3 pins both readings — `inline-fn-006` (`XP30`) against `inline-fn-007`
        // (`XP31+`), and `FunctionCall-044` against `-045`.
        let body = if self.peek() == Some(&Token::RightBrace) {
            if !self.version.at_least(XPathVersion::XPath31) {
                return Err(self.unexpected("an expression as the function's body"));
            }
            Expr::empty_sequence()
        } else {
            self.parse_expr()?
        };
        self.expect(&Token::RightBrace)?;
        self.pop_scopes(parameters.len());
        Ok(Expr::InlineFunction(Arc::new(InlineFunction {
            parameters,
            result,
            body,
        })))
    }

    fn parse_map_constructor(&mut self) -> XPathResult<Expr> {
        self.require_version(XPathVersion::XPath31, "a map constructor")?;
        self.advance();
        self.expect(&Token::LeftBrace)?;
        let mut entries = Vec::new();
        if self.eat(&Token::RightBrace) {
            return Ok(Expr::MapConstructor(entries));
        }
        loop {
            let key = self.parse_expr_single()?;
            self.expect(&Token::Colon)?;
            let value = self.parse_expr_single()?;
            entries.push((key, value));
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect(&Token::RightBrace)?;
        Ok(Expr::MapConstructor(entries))
    }

    fn parse_curly_array(&mut self) -> XPathResult<Expr> {
        self.require_version(XPathVersion::XPath31, "an array constructor")?;
        self.advance();
        self.expect(&Token::LeftBrace)?;
        if self.eat(&Token::RightBrace) {
            return Ok(Expr::ArrayConstructor {
                curly: true,
                members: Vec::new(),
            });
        }
        let inner = self.parse_expr()?;
        self.expect(&Token::RightBrace)?;
        Ok(Expr::ArrayConstructor {
            curly: true,
            members: vec![inner],
        })
    }

    /// A numeric literal as an *expression*: one whose lexical form is well-formed but whose
    /// value this implementation cannot hold becomes a [`Expr::Deferred`] `FOAR0002`, so that a
    /// `try`/`catch` around it sees the overflow the specification describes.
    fn numeric_expression(
        &self,
        atomic_type: AtomicType,
        text: &str,
        offset: usize,
    ) -> XPathResult<Expr> {
        match self.numeric_literal(atomic_type, text, offset) {
            Ok(value) => Ok(Expr::Literal(value)),
            Err(error) if text.chars().all(|c| c.is_ascii_digit()) => {
                let _ = error;
                Ok(Expr::Deferred {
                    code: XPathErrorCode::Foar0002,
                    message: format!("the integer literal `{text}` overflows xs:integer"),
                })
            }
            Err(error) => Err(error),
        }
    }

    fn numeric_literal(
        &self,
        atomic_type: AtomicType,
        text: &str,
        offset: usize,
    ) -> XPathResult<AtomicValue> {
        // XPath 1.0 has exactly one numeric type, and it is the IEEE double.
        let effective = if self.version.is_xpath_10() {
            AtomicType::Double
        } else {
            atomic_type
        };
        AtomicValue::parse(effective, text)
            .map_err(|error| XPathError::syntax(offset, error.message().to_owned()))
    }

    fn check_function_exists(
        &self,
        name: &ExpandedName,
        arity: usize,
        offset: usize,
    ) -> XPathResult<()> {
        if crate::builtins::is_known(self.context, name, arity, self.version) {
            return Ok(());
        }
        Err(XPathError::new(
            XPathErrorCode::Xpst0017,
            format!("no function `{name}` of arity {arity} is in scope"),
        )
        .at(offset))
    }
}

fn dynamic_call(target: Expr, operand: Expr, arguments: Vec<Option<Expr>>) -> Expr {
    let mut full = vec![Some(operand)];
    full.extend(arguments);
    Expr::DynamicCall {
        target: Box::new(target),
        arguments: full,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::NodeTest;
    use crate::types::NameTest;
    use oxixml_infoset::Axis;

    fn parse31(source: &str) -> XPathResult<Expr> {
        parse(source, XPathVersion::XPath31, &StaticContext::new())
    }

    fn parse_at(source: &str, version: XPathVersion) -> XPathResult<Expr> {
        parse(source, version, &StaticContext::new())
    }

    /// A name is only reserved once the version gives it another meaning, and a function only
    /// exists once the version adds it — both make an earlier version report `XPST0017` where
    /// 3.1 reports something else entirely.
    #[test]
    fn the_function_catalogue_and_the_reserved_names_follow_the_version() {
        for source in ["map()", "array()", "switch()", "function()"] {
            let failure = parse_at(source, XPathVersion::XPath20)
                .expect_err("not a function anyone has declared");
            assert_eq!(
                failure.code(),
                Some(XPathErrorCode::Xpst0017),
                "for `{source}` under XPath 2.0"
            );
        }
        assert_eq!(
            parse_at("map()", XPathVersion::XPath31)
                .expect_err("reserved in 3.1")
                .code(),
            Some(XPathErrorCode::Xpst0003)
        );
        assert!(parse_at("map{1:2}", XPathVersion::XPath31).is_ok());

        for source in ["format-number(3, '0000')", "round(1, 2)", "data()"] {
            assert_eq!(
                parse_at(source, XPathVersion::XPath20)
                    .expect_err("added after 2.0")
                    .code(),
                Some(XPathErrorCode::Xpst0017),
                "for `{source}`"
            );
            assert!(parse_at(source, XPathVersion::XPath31).is_ok(), "{source}");
        }
        // `namespace-node()` is a kind test from 3.0 only; before that it is a function name.
        assert_eq!(
            parse_at("namespace-node(1)", XPathVersion::XPath20)
                .expect_err("no such function")
                .code(),
            Some(XPathErrorCode::Xpst0017)
        );
    }

    #[test]
    fn an_inline_function_may_not_repeat_a_parameter_name() {
        let failure = parse31("function($a, $a) { $a }").expect_err("a repeated parameter");
        assert_eq!(failure.code(), Some(XPathErrorCode::Xqst0039));
        assert!(parse31("function($a, $b) { $a }").is_ok());
        // The body may be empty from 3.1 onwards, and not before.
        assert!(parse31("function($a) {}").is_ok());
        assert_eq!(
            parse_at("function($a) {}", XPathVersion::XPath30)
                .expect_err("3.0 requires an expression")
                .code(),
            Some(XPathErrorCode::Xpst0003)
        );
    }

    #[test]
    fn abbreviations_normalize() -> XPathResult<()> {
        let expression = parse31("//b")?;
        let Expr::Path(left, right) = &expression else {
            panic!("expected a path");
        };
        assert!(matches!(**left, Expr::Path(_, _)));
        let Expr::Step(step) = &**right else {
            panic!("expected a step");
        };
        assert_eq!(step.axis, Axis::Child);
        assert_eq!(
            step.node_test,
            NodeTest::Name(NameTest::Name(ExpandedName::new(None::<String>, "b")))
        );
        Ok(())
    }

    #[test]
    fn precedence_binds_as_the_grammar_says() -> XPathResult<()> {
        // `1 + 2 * 3` is `1 + (2 * 3)`.
        let expression = parse31("1 + 2 * 3")?;
        let Expr::Binary { op, right, .. } = &expression else {
            panic!("expected a binary expression");
        };
        assert_eq!(*op, BinaryOp::Add);
        assert!(matches!(
            **right,
            Expr::Binary {
                op: BinaryOp::Multiply,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn keywords_are_only_operators_in_operator_position() -> XPathResult<()> {
        assert!(matches!(parse31("child::div")?, Expr::Step(_)));
        assert!(matches!(
            parse31("a div b")?,
            Expr::Binary {
                op: BinaryOp::Divide,
                ..
            }
        ));
        assert!(matches!(parse31("/div")?, Expr::Path(_, _)));
        Ok(())
    }

    #[test]
    fn version_gates_reject_later_constructs() {
        let context = StaticContext::new();
        for source in [
            "1 to 3",
            "1 idiv 2",
            "a intersect b",
            "a eq b",
            "a is b",
            "'a' || 'b'",
            "a ! b",
            "for $x in 1 return $x",
            "if (1) then 2 else 3",
            "1 instance of xs:integer",
            "1 cast as xs:string",
            "map { 1 : 2 }",
            "[1, 2]",
            "fn:abs#1",
            "function($x) { $x }",
        ] {
            assert!(
                parse(source, XPathVersion::XPath10, &context).is_err(),
                "`{source}` must not parse as XPath 1.0"
            );
            assert!(
                parse(source, XPathVersion::XPath31, &context).is_ok(),
                "`{source}` must parse as XPath 3.1"
            );
        }
    }

    #[test]
    fn static_errors_are_reported() {
        let context = StaticContext::new();
        assert_eq!(
            parse("$nope", XPathVersion::XPath31, &context)
                .expect_err("undeclared")
                .code(),
            Some(XPathErrorCode::Xpst0008)
        );
        assert_eq!(
            parse("nope:x", XPathVersion::XPath31, &context)
                .expect_err("unbound prefix")
                .code(),
            Some(XPathErrorCode::Xpst0081)
        );
        assert_eq!(
            parse("fn:no-such-function()", XPathVersion::XPath31, &context)
                .expect_err("unknown function")
                .code(),
            Some(XPathErrorCode::Xpst0017)
        );
        assert_eq!(
            parse("1 +", XPathVersion::XPath31, &context)
                .expect_err("truncated")
                .code(),
            Some(XPathErrorCode::Xpst0003)
        );
        assert!(parse("1 2", XPathVersion::XPath31, &context).is_err());
    }

    #[test]
    fn bindings_are_in_scope_for_later_bindings_and_the_body() -> XPathResult<()> {
        parse31("for $x in (1, 2), $y in $x return $x + $y")?;
        parse31("let $x := 1, $y := $x + 1 return $y")?;
        parse31("some $x in (1, 2) satisfies $x = 1")?;
        parse31("function($x) { $x + 1 }")?;
        assert!(parse31("(for $x in 1 return $x) + $x").is_err());
        Ok(())
    }

    #[test]
    fn placeholders_become_partial_applications() -> XPathResult<()> {
        assert!(matches!(
            parse31("fn:substring(?, 2)")?,
            Expr::PartialCall { .. }
        ));
        assert!(matches!(
            parse31("fn:substring('abc', 2)")?,
            Expr::FunctionCall { .. }
        ));
        // `?a` inside an argument list is a unary lookup, not a placeholder.
        assert!(matches!(
            parse31("fn:count(?a)")?,
            Expr::FunctionCall { .. }
        ));
        Ok(())
    }

    #[test]
    fn the_arrow_operator_rewrites_to_a_call() -> XPathResult<()> {
        let expression = parse31("'abc' => fn:upper-case()")?;
        let Expr::FunctionCall { name, arguments } = &expression else {
            panic!("expected a function call");
        };
        assert_eq!(name.local_name(), "upper-case");
        assert_eq!(arguments.len(), 1);
        Ok(())
    }

    #[test]
    fn comments_and_whitespace_are_ignored() -> XPathResult<()> {
        assert_eq!(parse31("1 (: two :) + 2")?, parse31("1+2")?);
        Ok(())
    }
}
