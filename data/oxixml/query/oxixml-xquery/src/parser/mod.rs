//! The XQuery 3.1 recursive-descent parser.
//!
//! The parser is deliberately *span-preserving*: every expression it returns records the byte
//! range of query text it was parsed from, and whether that range contains any XQuery-only
//! syntax. That is what lets [`crate::compile`] hand a pure-XPath range straight to
//! [`oxixml_xpath`] instead of duplicating an evaluator for it.
//!
//! XQuery needs unbounded lookahead in several places — `element` may open a constructor, a kind
//! test or a function call, `<` may open a comparison or a direct element constructor, and
//! `declare` may introduce any of a dozen declarations — so the parser saves a [`State`] and
//! rewinds when an alternative fails.

mod construct;
mod flwor;
mod prolog;
mod types;

use crate::ast::{LookupKey, PathExpr, PathStep, Postfix, Syntax, SyntaxKind, unexpected};
use crate::error::{XQueryError, XQueryErrorCode, XQueryResult};
use crate::lexer::{Lexer, Spanned, Token};
use crate::prolog::Module;

/// The kind tests, whose names are followed by `(` in a step but which are not function calls.
/// The axes whose steps run from the context node backwards. Their node tests and predicates see
/// that order, so `ancestor::*[1]` is the *nearest* ancestor even though the step's value is in
/// document order.
const REVERSE_AXES: &[&str] = &[
    "ancestor",
    "ancestor-or-self",
    "parent",
    "preceding",
    "preceding-sibling",
];

const KIND_TESTS: &[&str] = &[
    "element",
    "attribute",
    "text",
    "comment",
    "processing-instruction",
    "document-node",
    "node",
    "schema-element",
    "schema-attribute",
    "namespace-node",
];

/// The names that may open a computed node constructor.
pub(crate) const COMPUTED_CONSTRUCTORS: &[&str] = &[
    "element",
    "attribute",
    "text",
    "comment",
    "processing-instruction",
    "document",
    "namespace",
];

/// The names that open a curly-brace primary expression rather than a name test.
const CURLY_PRIMARIES: &[&str] = &["ordered", "unordered", "map", "array"];

/// The unprefixed names
/// [XQuery 3.1 A.3](https://www.w3.org/TR/xquery-31/#id-reserved-fn-names) reserves, because
/// expression syntax takes precedence over a function call spelled with one of them.
pub(crate) const RESERVED_FUNCTION_NAMES: &[&str] = &[
    "array",
    "attribute",
    "comment",
    "document-node",
    "element",
    "empty-sequence",
    "function",
    "if",
    "item",
    "map",
    "namespace-node",
    "node",
    "processing-instruction",
    "schema-attribute",
    "schema-element",
    "switch",
    "text",
    "typeswitch",
];

/// A saved scanner position, for backtracking.
#[derive(Debug, Clone)]
pub(crate) struct State {
    current: Spanned,
    position: usize,
    last_end: usize,
    blanked: usize,
    references: usize,
    bindings: usize,
    type_annotations: usize,
    calls: usize,
    cast_targets: usize,
}

/// One variable binding, and the region of the query its name is in scope over.
///
/// XQuery 3.1 §5.1 gives every binder a *scope*: a function's parameters are in scope over that
/// function's body and nowhere else, a `typeswitch` case variable over that case's return clause
/// only, a FLWOR variable over the rest of its FLWOR expression. Recording the extent is what
/// lets [`crate::api`] decide `XPST0008` binder by binder instead of module-wide.
#[derive(Debug, Clone)]
pub(crate) struct Binding {
    /// Where the binding's name starts.
    pub(crate) at: usize,
    /// One past the last byte the binding is in scope over. [`usize::MAX`] means "to the end of
    /// the module", which is the scope of a prolog variable declaration.
    pub(crate) end: usize,
    /// The lexical name bound.
    pub(crate) name: String,
}

/// What the parser noticed while scanning, over and above the syntax tree itself.
#[derive(Debug, Clone, Default)]
pub(crate) struct Observations {
    /// Byte ranges whose text must never reach the XPath compiler, because XPath cannot parse
    /// them: the annotation assertions of a `FunctionTest`. Blanking a range rather than
    /// deleting it keeps every other offset in the query intact.
    pub(crate) blanked: Vec<(usize, usize)>,
    /// Every variable *reference* the parser consumed, as `(offset, lexical name)`.
    pub(crate) references: Vec<(usize, String)>,
    /// Every variable *binding* the parser consumed, with the extent of its scope.
    pub(crate) bindings: Vec<Binding>,
    /// The annotation names used as annotation assertions inside a sequence type.
    pub(crate) type_annotations: Vec<String>,
    /// Every function *call* the parser consumed, as `(offset, lexical name, arity)`.
    pub(crate) calls: Vec<(usize, String, usize)>,
    /// The `SingleType` of every `cast as` and `castable as`, as `(offset, lexical name)` with
    /// any trailing `?` removed. XQuery names errors for these that XPath does not — `XQST0052`
    /// for a target that is not a simple type in the in-scope schema types — so they are checked
    /// here rather than left to the delegated XPath compilation.
    pub(crate) cast_targets: Vec<(usize, String)>,
}

/// How deeply one expression may nest inside another.
///
/// The grammar is parsed by recursive descent, so every nesting level costs several stack
/// frames and an unbounded nesting would overflow the stack — a `SIGABRT`, not a catchable
/// error. The ceiling matches the one the XPath parser applies.
pub(crate) const MAX_NESTING_DEPTH: usize = 64;

/// The parser.
pub(crate) struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Spanned,
    last_end: usize,
    seen: Observations,
    /// Set once a `ContextItemDecl`, `AnnotatedDecl` or `OptionDecl` has been parsed, after
    /// which the grammar allows no further setter, namespace declaration or import.
    prolog_closed: bool,
    /// How many expressions are currently open around the one being parsed.
    depth: usize,
    /// How many direct constructors are currently open around the one being scanned.
    ///
    /// Constructor nesting is counted separately from expression nesting because a nested
    /// constructor is scanned from the raw text and never passes through [`Parser::expr_single`],
    /// and because what it bounds is the depth of an element tree rather than of an expression.
    pub(crate) constructor_depth: usize,
}

impl<'a> Parser<'a> {
    /// A parser positioned at the first token of `source`.
    pub(crate) fn new(source: &'a str) -> XQueryResult<Self> {
        let mut lexer = Lexer::new(source);
        let current = lexer.next_token()?;
        Ok(Self {
            lexer,
            current,
            last_end: 0,
            seen: Observations::default(),
            prolog_closed: false,
            depth: 0,
            constructor_depth: 0,
        })
    }

    /// What the parser noticed, taken out of it once parsing is done.
    pub(crate) fn take_observations(&mut self) -> Observations {
        std::mem::take(&mut self.seen)
    }

    /// Records that `start..end` must be blanked before the text reaches the XPath compiler.
    pub(crate) fn blank(&mut self, start: usize, end: usize) {
        self.seen.blanked.push((start, end));
    }

    /// Records a variable reference.
    pub(crate) fn record_reference(&mut self, at: usize, name: String) {
        self.seen.references.push((at, name));
    }

    /// Records a variable binding whose scope has not been delimited yet.
    pub(crate) fn record_binding(&mut self, at: usize, name: String) {
        self.seen.bindings.push(Binding {
            at,
            end: usize::MAX,
            name,
        });
    }

    /// How many bindings have been recorded, for [`Parser::close_bindings`].
    pub(crate) fn binding_mark(&self) -> usize {
        self.seen.bindings.len()
    }

    /// Ends the scope of every binding recorded since `mark` that is still open at `end`.
    ///
    /// A nested binder closes its own bindings first, so an enclosing construct only ever
    /// delimits the bindings it introduced itself.
    pub(crate) fn close_bindings(&mut self, mark: usize, end: usize) {
        for binding in self.seen.bindings.iter_mut().skip(mark) {
            if binding.end == usize::MAX {
                binding.end = end;
            }
        }
    }

    /// Records a function call.
    pub(crate) fn record_call(&mut self, at: usize, name: String, arity: usize) {
        self.seen.calls.push((at, name, arity));
    }

    /// Records the target type of a `cast as` or `castable as`.
    pub(crate) fn record_cast_target(&mut self, at: usize, source: &str) {
        let name = source.trim().trim_end_matches('?').trim_end();
        self.seen.cast_targets.push((at, name.to_owned()));
    }

    /// Whether the prolog has moved on to its second half — see [`Parser::prolog`].
    pub(crate) const fn prolog_closed(&self) -> bool {
        self.prolog_closed
    }

    /// Records that the prolog has moved on to its second half.
    pub(crate) const fn close_prolog(&mut self) {
        self.prolog_closed = true;
    }

    /// The blank ranges recorded so far.
    pub(crate) fn blanked_ranges(&self) -> &[(usize, usize)] {
        &self.seen.blanked
    }

    /// Records annotation names met while scanning a sequence type, for the reserved-namespace
    /// check the compiler runs once it knows the module's namespaces.
    pub(crate) fn note_type_annotations(&mut self, names: Vec<String>) {
        self.seen.type_annotations.extend(names);
    }

    /// Consumes an `EQName` that *binds* a variable, and records the binding.
    pub(crate) fn binding_name(&mut self, what: &str) -> XQueryResult<String> {
        let at = self.offset();
        let (name, _) = self.eqname(what)?;
        self.record_binding(at, name.clone());
        Ok(name)
    }

    // --- scanner plumbing -------------------------------------------------------------------

    pub(crate) const fn source(&self) -> &'a str {
        self.lexer.source()
    }

    pub(crate) fn text(&self, start: usize, end: usize) -> &'a str {
        self.lexer.source().get(start..end).unwrap_or_default()
    }

    pub(crate) const fn peek(&self) -> &Token {
        &self.current.token
    }

    /// The byte offset the current token starts at.
    pub(crate) const fn offset(&self) -> usize {
        self.current.start
    }

    /// The byte offset just past the most recently consumed token.
    pub(crate) const fn last_end(&self) -> usize {
        self.last_end
    }

    /// The byte offset just past the *current* token, without consuming it.
    ///
    /// The raw scanners of [`super::construct`] use this to close an enclosed expression
    /// without letting the expression lexer look at the XML that follows, where a `&` or a `<`
    /// would not be a token at all.
    pub(crate) const fn current_end(&self) -> usize {
        self.current.end
    }

    /// Checks that the current token closes an enclosed expression and returns the offset just
    /// past it, leaving the scanner where it is.
    pub(crate) fn close_brace(&mut self) -> XQueryResult<usize> {
        if self.peek() == &Token::RightBrace {
            return Ok(self.current_end());
        }
        Err(self.error(unexpected(self.peek(), "`}`")))
    }

    pub(crate) fn state(&self) -> State {
        State {
            current: self.current.clone(),
            position: self.lexer.position(),
            last_end: self.last_end,
            blanked: self.seen.blanked.len(),
            references: self.seen.references.len(),
            bindings: self.seen.bindings.len(),
            type_annotations: self.seen.type_annotations.len(),
            calls: self.seen.calls.len(),
            cast_targets: self.seen.cast_targets.len(),
        }
    }

    pub(crate) fn restore(&mut self, state: State) {
        self.lexer.seek(state.position);
        self.current = state.current;
        self.last_end = state.last_end;
        // Whatever the abandoned alternative noticed did not happen.
        self.seen.blanked.truncate(state.blanked);
        self.seen.references.truncate(state.references);
        self.seen.bindings.truncate(state.bindings);
        self.seen.type_annotations.truncate(state.type_annotations);
        self.seen.calls.truncate(state.calls);
        self.seen.cast_targets.truncate(state.cast_targets);
    }

    /// Restarts tokenizing at `position`, which is how the raw scanners hand control back.
    pub(crate) fn resume_at(&mut self, position: usize) -> XQueryResult<()> {
        self.lexer.seek(position);
        self.last_end = position;
        self.current = self.lexer.next_token()?;
        Ok(())
    }

    /// Consumes the current token and returns it.
    pub(crate) fn advance(&mut self) -> XQueryResult<Spanned> {
        let taken = self.current.clone();
        self.last_end = taken.end;
        self.current = self.lexer.next_token()?;
        Ok(taken)
    }

    /// Consumes the current token when it is `token`.
    pub(crate) fn eat(&mut self, token: &Token) -> XQueryResult<bool> {
        if &self.current.token == token {
            self.advance()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Consumes `token`, or fails with `XPST0003`.
    pub(crate) fn expect(&mut self, token: &Token, wanted: &str) -> XQueryResult<Spanned> {
        if &self.current.token == token {
            return self.advance();
        }
        Err(XQueryError::syntax(
            unexpected(&self.current.token, wanted),
            self.current.start,
        ))
    }

    /// Whether the current token is the keyword `name`.
    pub(crate) fn at_keyword(&self, name: &str) -> bool {
        matches!(&self.current.token, Token::Name(found) if found == name)
    }

    /// Whether the current token is any of `names`.
    pub(crate) fn at_any_keyword(&self, names: &[&str]) -> bool {
        matches!(&self.current.token, Token::Name(found) if names.contains(&found.as_str()))
    }

    /// Consumes the keyword `name` if it is current.
    pub(crate) fn eat_keyword(&mut self, name: &str) -> XQueryResult<bool> {
        if self.at_keyword(name) {
            self.advance()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Consumes the keyword `name`, or fails with `XPST0003`.
    pub(crate) fn expect_keyword(&mut self, name: &str) -> XQueryResult<Spanned> {
        if self.at_keyword(name) {
            return self.advance();
        }
        Err(XQueryError::syntax(
            unexpected(&self.current.token, &format!("`{name}`")),
            self.current.start,
        ))
    }

    /// Whether the *next* token after the current one is the keyword `name`.
    pub(crate) fn next_is_keyword(&mut self, name: &str) -> XQueryResult<bool> {
        let state = self.state();
        self.advance()?;
        let found = self.at_keyword(name);
        self.restore(state);
        Ok(found)
    }

    /// Whether the *next* token after the current one is `token`.
    pub(crate) fn next_is(&mut self, token: &Token) -> XQueryResult<bool> {
        let state = self.state();
        self.advance()?;
        let found = &self.current.token == token;
        self.restore(state);
        Ok(found)
    }

    pub(crate) fn error(&self, message: impl Into<String>) -> XQueryError {
        XQueryError::syntax(message, self.current.start)
    }

    // --- entry points -----------------------------------------------------------------------

    /// Parses a whole module and checks that the query text is exhausted.
    pub(crate) fn parse_module(&mut self) -> XQueryResult<Module> {
        let module = self.module()?;
        if self.peek() != &Token::Eof {
            return Err(self.error(unexpected(self.peek(), "the end of the query")));
        }
        validate_output_declarations(&module)?;
        Ok(module)
    }

    // --- Expr -------------------------------------------------------------------------------

    /// `Expr ::= ExprSingle ("," ExprSingle)*`
    pub(crate) fn expr(&mut self) -> XQueryResult<Syntax> {
        let first = self.expr_single()?;
        if self.peek() != &Token::Comma {
            return Ok(first);
        }
        let start = first.start;
        let mut items = vec![first];
        while self.eat(&Token::Comma)? {
            items.push(self.expr_single()?);
        }
        let end = self.last_end();
        Ok(Syntax::new(start, end, SyntaxKind::Generic(items)))
    }

    /// `ExprSingle`
    ///
    /// Every recursive re-entry into the grammar passes through here, so this is where the
    /// nesting depth is counted: without a ceiling a few hundred nested parentheses overflow
    /// the stack, which aborts the process rather than raising a catchable error.
    pub(crate) fn expr_single(&mut self) -> XQueryResult<Syntax> {
        if self.depth >= MAX_NESTING_DEPTH {
            return Err(self.error(format!(
                "the expression nests more than {MAX_NESTING_DEPTH} levels deep"
            )));
        }
        self.depth += 1;
        let parsed = self.expr_single_inner();
        self.depth -= 1;
        parsed
    }

    fn expr_single_inner(&mut self) -> XQueryResult<Syntax> {
        if self.at_keyword("for") || self.at_keyword("let") {
            if let Some(parsed) = self.try_flwor()? {
                return Ok(parsed);
            }
        }
        if self.at_keyword("some") || self.at_keyword("every") {
            if let Some(parsed) = self.try_quantified()? {
                return Ok(parsed);
            }
        }
        if self.at_keyword("if") && self.next_is(&Token::LeftParen)? {
            return self.if_expr();
        }
        if self.at_keyword("switch") && self.next_is(&Token::LeftParen)? {
            if let Some(parsed) = self.try_switch()? {
                return Ok(parsed);
            }
        }
        if self.at_keyword("typeswitch") && self.next_is(&Token::LeftParen)? {
            if let Some(parsed) = self.try_typeswitch()? {
                return Ok(parsed);
            }
        }
        if self.at_keyword("try") && self.next_is(&Token::LeftBrace)? {
            if let Some(parsed) = self.try_catch()? {
                return Ok(parsed);
            }
        }
        self.or_expr()
    }

    fn if_expr(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        self.expect_keyword("if")?;
        self.expect(&Token::LeftParen, "`(`")?;
        let condition = self.expr()?;
        self.expect(&Token::RightParen, "`)`")?;
        self.expect_keyword("then")?;
        let then_branch = self.expr_single()?;
        self.expect_keyword("else")?;
        let else_branch = self.expr_single()?;
        let end = self.last_end();
        Ok(Syntax::new(
            start,
            end,
            SyntaxKind::If {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            },
        ))
    }

    fn or_expr(&mut self) -> XQueryResult<Syntax> {
        let mut left = self.and_expr()?;
        while self.at_keyword("or") {
            self.advance()?;
            let right = self.and_expr()?;
            left = Syntax::new(
                left.start,
                right.end,
                SyntaxKind::Logical {
                    or: true,
                    left: Box::new(left),
                    right: Box::new(right),
                },
            );
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> XQueryResult<Syntax> {
        let mut left = self.comparison_expr()?;
        while self.at_keyword("and") {
            self.advance()?;
            let right = self.comparison_expr()?;
            left = Syntax::new(
                left.start,
                right.end,
                SyntaxKind::Logical {
                    or: false,
                    left: Box::new(left),
                    right: Box::new(right),
                },
            );
        }
        Ok(left)
    }

    fn comparison_expr(&mut self) -> XQueryResult<Syntax> {
        let left = self.string_concat_expr()?;
        let is_comparison = matches!(
            self.peek(),
            Token::Equal
                | Token::NotEqual
                | Token::Less
                | Token::LessEqual
                | Token::Greater
                | Token::GreaterEqual
                | Token::Precedes
                | Token::Follows
        ) || self.at_any_keyword(&["eq", "ne", "lt", "le", "gt", "ge", "is"]);
        if !is_comparison {
            return Ok(left);
        }
        self.advance()?;
        let right = self.string_concat_expr()?;
        Ok(Syntax::new(
            left.start,
            right.end,
            SyntaxKind::Generic(vec![left, right]),
        ))
    }

    fn string_concat_expr(&mut self) -> XQueryResult<Syntax> {
        let mut operands = vec![self.range_expr()?];
        while self.eat(&Token::Concat)? {
            operands.push(self.range_expr()?);
        }
        Ok(Self::fold_generic(operands))
    }

    fn range_expr(&mut self) -> XQueryResult<Syntax> {
        let left = self.additive_expr()?;
        if !self.at_keyword("to") {
            return Ok(left);
        }
        self.advance()?;
        let right = self.additive_expr()?;
        Ok(Syntax::new(
            left.start,
            right.end,
            SyntaxKind::Generic(vec![left, right]),
        ))
    }

    fn additive_expr(&mut self) -> XQueryResult<Syntax> {
        let mut operands = vec![self.multiplicative_expr()?];
        while matches!(self.peek(), Token::Plus | Token::Minus) {
            self.advance()?;
            operands.push(self.multiplicative_expr()?);
        }
        Ok(Self::fold_generic(operands))
    }

    fn multiplicative_expr(&mut self) -> XQueryResult<Syntax> {
        let mut operands = vec![self.union_expr()?];
        while self.peek() == &Token::Star || self.at_any_keyword(&["div", "idiv", "mod"]) {
            self.advance()?;
            operands.push(self.union_expr()?);
        }
        Ok(Self::fold_generic(operands))
    }

    fn union_expr(&mut self) -> XQueryResult<Syntax> {
        let mut operands = vec![self.intersect_except_expr()?];
        while self.peek() == &Token::Pipe || self.at_keyword("union") {
            self.advance()?;
            operands.push(self.intersect_except_expr()?);
        }
        Ok(Self::fold_generic(operands))
    }

    fn intersect_except_expr(&mut self) -> XQueryResult<Syntax> {
        let mut operands = vec![self.instance_of_expr()?];
        while self.at_any_keyword(&["intersect", "except"]) {
            self.advance()?;
            operands.push(self.instance_of_expr()?);
        }
        Ok(Self::fold_generic(operands))
    }

    fn instance_of_expr(&mut self) -> XQueryResult<Syntax> {
        let operand = self.treat_expr()?;
        if !(self.at_keyword("instance") && self.next_is_keyword("of")?) {
            return Ok(operand);
        }
        self.advance()?;
        self.advance()?;
        self.sequence_type_source()?;
        let (start, end) = (operand.start, self.last_end());
        Ok(Syntax::new(start, end, SyntaxKind::Generic(vec![operand])))
    }

    fn treat_expr(&mut self) -> XQueryResult<Syntax> {
        let operand = self.castable_expr()?;
        if !(self.at_keyword("treat") && self.next_is_keyword("as")?) {
            return Ok(operand);
        }
        self.advance()?;
        self.advance()?;
        self.sequence_type_source()?;
        let (start, end) = (operand.start, self.last_end());
        Ok(Syntax::new(start, end, SyntaxKind::Generic(vec![operand])))
    }

    fn castable_expr(&mut self) -> XQueryResult<Syntax> {
        let operand = self.cast_expr()?;
        if !(self.at_keyword("castable") && self.next_is_keyword("as")?) {
            return Ok(operand);
        }
        self.advance()?;
        self.advance()?;
        let at = self.offset();
        let target = self.single_type_source()?;
        self.record_cast_target(at, &target);
        let (start, end) = (operand.start, self.last_end());
        Ok(Syntax::new(start, end, SyntaxKind::Generic(vec![operand])))
    }

    fn cast_expr(&mut self) -> XQueryResult<Syntax> {
        let operand = self.arrow_expr()?;
        if !(self.at_keyword("cast") && self.next_is_keyword("as")?) {
            return Ok(operand);
        }
        self.advance()?;
        self.advance()?;
        let at = self.offset();
        let target = self.single_type_source()?;
        self.record_cast_target(at, &target);
        let (start, end) = (operand.start, self.last_end());
        Ok(Syntax::new(start, end, SyntaxKind::Generic(vec![operand])))
    }

    fn arrow_expr(&mut self) -> XQueryResult<Syntax> {
        let mut left = self.unary_expr()?;
        while self.eat(&Token::Arrow)? {
            let mut operands = vec![left];
            match self.peek().clone() {
                Token::Name(_) | Token::BracedName { .. } => {
                    self.advance()?;
                }
                Token::Dollar => {
                    self.advance()?;
                    let at = self.offset();
                    let (name, _) = self.eqname("a variable name")?;
                    self.record_reference(at, name);
                }
                Token::LeftParen => {
                    let inner = self.parenthesized()?;
                    operands.push(inner);
                }
                other => {
                    return Err(self.error(unexpected(&other, "a function name after `=>`")));
                }
            }
            let (arguments, end) = self.argument_list()?;
            for argument in arguments.into_iter().flatten() {
                operands.push(argument);
            }
            let start = operands.first().map_or(0, |first| first.start);
            left = Syntax::new(start, end, SyntaxKind::Generic(operands));
        }
        Ok(left)
    }

    fn unary_expr(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        let mut signed = false;
        while matches!(self.peek(), Token::Plus | Token::Minus) {
            self.advance()?;
            signed = true;
        }
        let operand = self.value_expr()?;
        if !signed {
            return Ok(operand);
        }
        let end = operand.end;
        Ok(Syntax::new(start, end, SyntaxKind::Generic(vec![operand])))
    }

    /// `ValueExpr ::= ValidateExpr | ExtensionExpr | SimpleMapExpr`
    fn value_expr(&mut self) -> XQueryResult<Syntax> {
        if self.at_keyword("validate") {
            if let Some(parsed) = self.try_validate()? {
                return Ok(parsed);
            }
        }
        if self.peek() == &Token::PragmaOpen {
            return self.extension_expr();
        }
        self.simple_map_expr()
    }

    fn try_validate(&mut self) -> XQueryResult<Option<Syntax>> {
        let state = self.state();
        let start = self.offset();
        self.advance()?;
        if self.at_any_keyword(&["lax", "strict"]) {
            self.advance()?;
        } else if self.at_keyword("type") {
            self.advance()?;
            self.eqname("a type name")?;
        }
        if self.peek() != &Token::LeftBrace {
            self.restore(state);
            return Ok(None);
        }
        self.advance()?;
        self.expr()?;
        let end = self.expect(&Token::RightBrace, "`}`")?.end;
        Ok(Some(Syntax::native(start, end, SyntaxKind::Validate)))
    }

    fn extension_expr(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        let mut pragmas = Vec::new();
        while self.peek() == &Token::PragmaOpen {
            pragmas.push(self.skip_pragma()?);
        }
        self.expect(&Token::LeftBrace, "`{` after a pragma")?;
        if self.peek() == &Token::RightBrace {
            return Err(XQueryError::new(
                crate::error::XQueryErrorCode::Xqst0079,
                "an extension expression with no recognized pragma needs an expression",
            )
            .at(start));
        }
        let body = self.expr()?;
        let end = self.expect(&Token::RightBrace, "`}`")?.end;
        Ok(Syntax::native(
            start,
            end,
            SyntaxKind::Extension {
                pragmas,
                body: Box::new(body),
            },
        ))
    }

    /// Skips `Pragma ::= "(#" S? EQName (S PragmaContents)? "#)"`, whose contents the expression
    /// lexer does not tokenize, and returns the pragma's lexical name.
    ///
    /// The grammar is exact about the whitespace: the name must be an `EQName`, and anything
    /// after it must be separated from it by whitespace, so `(#p:name(x)#)` is a syntax error
    /// rather than a pragma called `p:name(x)`.
    fn skip_pragma(&mut self) -> XQueryResult<String> {
        let start = self.offset();
        let source = self.source();
        let Some(relative) = source.get(start..).and_then(|rest| rest.find("#)")) else {
            return Err(XQueryError::syntax("the pragma is never closed", start));
        };
        let close = start + relative;
        let body = self.text(start + 2, close);
        let leading = body.len() - body.trim_start().len();
        let trimmed = body.trim_start();
        let (name, consumed) = scan_pragma_name(trimmed)
            .ok_or_else(|| XQueryError::syntax("a pragma needs an EQName", start))?;
        let rest = trimmed.get(consumed..).unwrap_or_default();
        if !rest.is_empty() && !rest.starts_with([' ', '\t', '\r', '\n']) {
            return Err(XQueryError::syntax(
                "a pragma's contents must be separated from its name by whitespace",
                start + 2 + leading + consumed,
            ));
        }
        self.resume_at(close + 2)?;
        Ok(name)
    }

    /// `SimpleMapExpr ::= PathExpr ("!" PathExpr)*`
    fn simple_map_expr(&mut self) -> XQueryResult<Syntax> {
        let first = self.path_expr()?;
        if self.peek() != &Token::Bang {
            return Ok(first);
        }
        let start = first.start;
        let mut steps = vec![PathStep {
            descendant: false,
            simple_map: false,
            reverse: false,
            base: first,
            predicates: Vec::new(),
        }];
        while self.eat(&Token::Bang)? {
            let operand = self.path_expr()?;
            steps.push(PathStep {
                descendant: false,
                simple_map: true,
                reverse: false,
                base: operand,
                predicates: Vec::new(),
            });
        }
        let end = self.last_end();
        Ok(Syntax::new(
            start,
            end,
            SyntaxKind::Path(Box::new(PathExpr {
                rooted: false,
                rooted_descendant: false,
                steps,
            })),
        ))
    }

    /// `PathExpr`
    fn path_expr(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        let (rooted, rooted_descendant) = match self.peek() {
            Token::Slash => {
                self.advance()?;
                (true, false)
            }
            Token::DoubleSlash => {
                self.advance()?;
                (true, true)
            }
            _ => (false, false),
        };
        if rooted && !rooted_descendant && !self.starts_step() {
            // A lone `/` is the root of the tree containing the context node.
            return Ok(Syntax::new(
                start,
                self.last_end(),
                SyntaxKind::Generic(Vec::new()),
            ));
        }
        let mut steps = Vec::new();
        let mut descendant = false;
        loop {
            steps.push(self.step_expr(descendant)?);
            match self.peek() {
                Token::Slash => {
                    self.advance()?;
                    descendant = false;
                }
                Token::DoubleSlash => {
                    self.advance()?;
                    descendant = true;
                }
                _ => break,
            }
        }
        let end = self.last_end();
        if !rooted && steps.len() == 1 && !steps[0].descendant && steps[0].predicates.is_empty() {
            return Ok(steps.remove(0).base);
        }
        Ok(Syntax::new(
            start,
            end,
            SyntaxKind::Path(Box::new(PathExpr {
                rooted,
                rooted_descendant,
                steps,
            })),
        ))
    }

    /// Whether the current token can begin a step.
    ///
    /// This decides the *leading lone slash*: `PathExpr ::= ("/" RelativePathExpr?) | …`, so a
    /// `/` is a complete expression — the root of the tree containing the context node — unless
    /// what follows can begin a `RelativePathExpr`
    /// ([XQuery 3.1 §3.3.1](https://www.w3.org/TR/xquery-31/#id-path-expressions), and the
    /// grammar note that a `/` followed by something that cannot start a relative path is the
    /// whole expression). The set below is therefore what a `StepExpr` — a `PostfixExpr` over a
    /// `PrimaryExpr`, or an `AxisStep` — can start with, and nothing else: an operator such as
    /// `-`, `=`, `!` or `|` ends the path rather than continuing it, which is what makes
    /// `/ - 5`, `/ = $a`, `/ ! works` and `/ | //*` the expressions the specification says they
    /// are.
    fn starts_step(&self) -> bool {
        matches!(
            self.peek(),
            // `AxisStep`: an axis or abbreviation, a name test, a wildcard, a kind test.
            Token::At
                | Token::DotDot
                | Token::Star
                | Token::Name(_)
                | Token::BracedName { .. }
                | Token::BracedWildcard { .. }
                // `PrimaryExpr`: `.`, a literal, `$v`, `(…)`, a direct constructor, a string
                // constructor, an inline function's annotation, a unary lookup.
                | Token::Dot
                | Token::Integer(_)
                | Token::Decimal(_)
                | Token::Double(_)
                | Token::Str { .. }
                | Token::Dollar
                | Token::LeftParen
                | Token::LeftBracket
                | Token::Less
                | Token::Backtick
                | Token::Percent
                | Token::Question
        )
    }

    /// `StepExpr ::= PostfixExpr | AxisStep`
    fn step_expr(&mut self, descendant: bool) -> XQueryResult<PathStep> {
        if let Some(step) = self.try_axis_step(descendant)? {
            return Ok(step);
        }
        let base = self.postfix_expr()?;
        Ok(PathStep {
            descendant,
            simple_map: false,
            reverse: false,
            base,
            predicates: Vec::new(),
        })
    }

    /// Parses an axis step when the current token can only begin one.
    fn try_axis_step(&mut self, descendant: bool) -> XQueryResult<Option<PathStep>> {
        let start = self.offset();
        let mut reverse = false;
        let scanned = match self.peek().clone() {
            Token::At => {
                self.advance()?;
                self.node_test()?;
                true
            }
            Token::DotDot => {
                self.advance()?;
                reverse = true;
                true
            }
            Token::Star | Token::BracedWildcard { .. } => {
                self.node_test()?;
                true
            }
            Token::Name(name) => {
                if self.next_is(&Token::ColonColon)? {
                    // XQuery has no namespace axis: `namespace::*` is not in its grammar at all
                    // (XQuery 3.1 §3.3.2.1), so it is a syntax error rather than a step that
                    // selects nothing.
                    if name == "namespace" {
                        return Err(XQueryError::syntax(
                            "the namespace axis is not part of the XQuery grammar",
                            self.offset(),
                        ));
                    }
                    reverse = REVERSE_AXES.contains(&name.as_str());
                    self.advance()?;
                    self.advance()?;
                    self.node_test()?;
                    true
                } else if self.next_is(&Token::LeftParen)? {
                    if KIND_TESTS.contains(&name.as_str()) {
                        self.reject_namespace_node_test(&name)?;
                        self.advance()?;
                        self.balanced_parens()?;
                        true
                    } else {
                        false
                    }
                } else if self.next_is(&Token::Colon)? {
                    self.node_test()?;
                    true
                } else if self.next_is(&Token::Hash)? || self.next_is(&Token::LeftBrace)? {
                    false
                } else if CURLY_PRIMARIES.contains(&name.as_str())
                    || COMPUTED_CONSTRUCTORS.contains(&name.as_str())
                {
                    // `element foo { … }` is a constructor; a bare `element` is a name test.
                    if self.starts_computed_constructor(&name)? {
                        false
                    } else {
                        self.advance()?;
                        true
                    }
                } else {
                    self.advance()?;
                    true
                }
            }
            Token::BracedName { .. } => {
                if self.next_is(&Token::LeftParen)? || self.next_is(&Token::Hash)? {
                    false
                } else {
                    self.advance()?;
                    true
                }
            }
            _ => false,
        };
        if !scanned {
            return Ok(None);
        }
        let base = Syntax::new(start, self.last_end(), SyntaxKind::Generic(Vec::new()));
        let predicates = self.predicates()?;
        Ok(Some(PathStep {
            descendant,
            simple_map: false,
            reverse,
            base,
            predicates,
        }))
    }

    /// An *abbreviated* forward step whose node test is `namespace-node()` runs on the namespace
    /// axis, which XQuery does not have; XQuery 3.1 §3.3.2.1 gives that its own code, `XQST0134`
    /// (`Axes112`). Written with an axis of its own the test is perfectly legal — `Axes115`
    /// expects `attribute::namespace-node()` to select nothing and `Axes117` expects
    /// `self::namespace-node()` to select the namespace node it is applied to — and so it is in
    /// a `SequenceType`, where it merely names the kind of an item.
    fn reject_namespace_node_test(&self, name: &str) -> XQueryResult<()> {
        if name == "namespace-node" {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0134,
                "the namespace axis is not supported in XQuery, so `namespace-node()` cannot be \
                 the node test of an axis step",
            )
            .at(self.offset()));
        }
        Ok(())
    }

    /// `NodeTest ::= KindTest | NameTest`, with the axis already consumed.
    ///
    /// `Wildcard ::= "*" | (NCName ":" "*") | ("*" ":" NCName) | (BracedURILiteral "*")` carries
    /// the grammar note `ws: explicit` ([XQuery 3.1 §A.2.4.1](https://www.w3.org/TR/xquery-31/#DefaultWhitespaceHandling)),
    /// so no whitespace or comment may stand inside one, and `*:*` is not one of its four forms.
    /// A `:` that fails either test is a *different* terminal — most often the one that separates
    /// a map constructor's key from its value, which is why `map{b:2}`, `map{* :b}` and
    /// `map{*:*}` are well-formed maps with a name-test key rather than syntax errors.
    fn node_test(&mut self) -> XQueryResult<()> {
        match self.peek().clone() {
            Token::Star => {
                let star = self.advance()?;
                if self.peek() == &Token::Colon && self.offset() == star.end {
                    let state = self.state();
                    let colon = self.advance()?;
                    if let Token::Name(name) = self.peek().clone() {
                        let local_start = self.offset();
                        if local_start == colon.end {
                            // Only one NCName belongs to the wildcard: `*:b:b` is `*:b`, then
                            // the map constructor's `:`, then the name test `b`. The lexer
                            // scanned the longest `QName` it could, so the rest is handed back.
                            let local = name.split(':').next().unwrap_or(name.as_str());
                            if local.len() == name.len() {
                                self.advance()?;
                            } else {
                                self.resume_at(local_start + local.len())?;
                            }
                            return Ok(());
                        }
                    }
                    self.restore(state);
                }
                Ok(())
            }
            Token::BracedWildcard { .. } | Token::BracedName { .. } => {
                self.advance()?;
                Ok(())
            }
            Token::Name(name) => {
                if KIND_TESTS.contains(&name.as_str()) && self.next_is(&Token::LeftParen)? {
                    self.advance()?;
                    self.balanced_parens()?;
                    return Ok(());
                }
                let scanned = self.advance()?;
                // A name the lexer already joined across a `:` is a `QName`, never the prefix of
                // a wildcard, so `map{a:b:c}` is the key `a:b` and the value `c`.
                if !name.contains(':')
                    && self.peek() == &Token::Colon
                    && self.offset() == scanned.end
                {
                    let state = self.state();
                    let colon = self.advance()?;
                    if self.peek() == &Token::Star && self.offset() == colon.end {
                        self.advance()?;
                    } else {
                        self.restore(state);
                    }
                }
                Ok(())
            }
            other => Err(self.error(unexpected(&other, "a node test"))),
        }
    }

    /// Consumes a `(`-delimited run with balanced parentheses and returns its end offset.
    pub(crate) fn balanced_parens(&mut self) -> XQueryResult<usize> {
        self.expect(&Token::LeftParen, "`(`")?;
        let mut depth = 1_usize;
        loop {
            match self.peek() {
                Token::LeftParen => depth += 1,
                Token::RightParen => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(self.advance()?.end);
                    }
                }
                Token::Eof => return Err(self.error("unbalanced parentheses")),
                _ => {}
            }
            self.advance()?;
        }
    }

    fn predicates(&mut self) -> XQueryResult<Vec<Syntax>> {
        let mut predicates = Vec::new();
        while self.peek() == &Token::LeftBracket {
            self.advance()?;
            predicates.push(self.expr()?);
            self.expect(&Token::RightBracket, "`]`")?;
        }
        Ok(predicates)
    }

    /// `PostfixExpr ::= PrimaryExpr (Predicate | ArgumentList | Lookup)*`
    fn postfix_expr(&mut self) -> XQueryResult<Syntax> {
        let primary = self.primary_expr()?;
        let start = primary.start;
        let mut end = primary.end;
        let mut postfixes = Vec::new();
        loop {
            match self.peek() {
                Token::LeftBracket => {
                    self.advance()?;
                    let predicate = self.expr()?;
                    end = self.expect(&Token::RightBracket, "`]`")?.end;
                    postfixes.push(Postfix::Predicate(predicate));
                }
                Token::LeftParen => {
                    let (arguments, at) = self.argument_list()?;
                    end = at;
                    postfixes.push(Postfix::Arguments(arguments));
                }
                Token::Question => {
                    self.advance()?;
                    let (key, at) = self.lookup_key()?;
                    end = at;
                    postfixes.push(Postfix::Lookup(key));
                }
                _ => break,
            }
        }
        if postfixes.is_empty() {
            return Ok(primary);
        }
        Ok(Syntax::new(
            start,
            end,
            SyntaxKind::Postfix {
                primary: Box::new(primary),
                postfixes,
            },
        ))
    }

    fn lookup_key(&mut self) -> XQueryResult<(LookupKey, usize)> {
        match self.peek().clone() {
            Token::Star => {
                let end = self.advance()?.end;
                Ok((LookupKey::Wildcard, end))
            }
            Token::Integer(text) => {
                let end = self.advance()?.end;
                let value = text
                    .parse::<i64>()
                    .map_err(|_| self.error("the lookup key is not an integer"))?;
                Ok((LookupKey::Integer(value), end))
            }
            Token::Name(name) => {
                let end = self.advance()?.end;
                Ok((LookupKey::Name(name), end))
            }
            Token::LeftParen => {
                let start = self.offset();
                self.advance()?;
                // `KeySpecifier ::= … | ParenthesizedExpr`, and `()` is a ParenthesizedExpr: a
                // lookup on the empty sequence of keys selects nothing (QT3 `Lookup-025`).
                if self.peek() == &Token::RightParen {
                    let end = self.advance()?.end;
                    let empty = Syntax::new(start, end, SyntaxKind::Generic(Vec::new()));
                    return Ok((LookupKey::Computed(Box::new(empty)), end));
                }
                let inner = self.expr()?;
                let end = self.expect(&Token::RightParen, "`)`")?.end;
                Ok((LookupKey::Computed(Box::new(inner)), end))
            }
            other => Err(self.error(unexpected(&other, "a lookup key"))),
        }
    }

    /// `ArgumentList ::= "(" (Argument ("," Argument)*)? ")"`, where `?` is a placeholder.
    pub(crate) fn argument_list(&mut self) -> XQueryResult<(Vec<Option<Syntax>>, usize)> {
        self.expect(&Token::LeftParen, "`(`")?;
        let mut arguments = Vec::new();
        if self.peek() == &Token::RightParen {
            let end = self.advance()?.end;
            return Ok((arguments, end));
        }
        loop {
            if self.peek() == &Token::Question
                && (self.next_is(&Token::Comma)? || self.next_is(&Token::RightParen)?)
            {
                self.advance()?;
                arguments.push(None);
            } else {
                arguments.push(Some(self.expr_single()?));
            }
            if !self.eat(&Token::Comma)? {
                break;
            }
        }
        let end = self.expect(&Token::RightParen, "`)`")?.end;
        Ok((arguments, end))
    }

    fn parenthesized(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        self.expect(&Token::LeftParen, "`(`")?;
        if self.peek() == &Token::RightParen {
            let end = self.advance()?.end;
            return Ok(Syntax::new(start, end, SyntaxKind::Generic(Vec::new())));
        }
        let inner = self.expr()?;
        let end = self.expect(&Token::RightParen, "`)`")?.end;
        Ok(Syntax::new(start, end, SyntaxKind::Generic(vec![inner])))
    }

    /// `PrimaryExpr`
    fn primary_expr(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        match self.peek().clone() {
            Token::Integer(_) | Token::Decimal(_) | Token::Double(_) | Token::Dot => {
                let end = self.advance()?.end;
                Ok(Syntax::new(start, end, SyntaxKind::Generic(Vec::new())))
            }
            Token::Str { value, expanded } => {
                let end = self.advance()?.end;
                if expanded {
                    return Ok(Syntax::native(
                        start,
                        end,
                        SyntaxKind::Literal(oxixml_infoset::AtomicValue::string(value)),
                    ));
                }
                Ok(Syntax::new(start, end, SyntaxKind::Generic(Vec::new())))
            }
            Token::Dollar => {
                self.advance()?;
                let at = self.offset();
                let (name, _) = self.eqname("a variable name")?;
                self.record_reference(at, name);
                Ok(Syntax::new(
                    start,
                    self.last_end(),
                    SyntaxKind::Generic(Vec::new()),
                ))
            }
            Token::LeftParen => self.parenthesized(),
            Token::LeftBracket => self.square_array(),
            Token::Question => {
                self.advance()?;
                let (key, end) = self.lookup_key()?;
                Ok(Syntax::new(
                    start,
                    end,
                    SyntaxKind::Postfix {
                        primary: Box::new(Syntax::new(start, start, SyntaxKind::ContextItem)),
                        postfixes: vec![Postfix::Lookup(key)],
                    },
                ))
            }
            Token::Percent => self.inline_function(),
            Token::Less => self.direct_constructor(),
            Token::Backtick => self.string_constructor(),
            Token::Name(name) => self.named_primary(&name),
            Token::BracedName { .. } => {
                if self.next_is(&Token::Hash)? {
                    self.advance()?;
                    self.advance()?;
                    let end = self.advance()?.end;
                    return Ok(Syntax::new(start, end, SyntaxKind::Generic(Vec::new())));
                }
                self.function_call()
            }
            other => Err(self.error(unexpected(&other, "an expression"))),
        }
    }

    fn square_array(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        self.expect(&Token::LeftBracket, "`[`")?;
        let mut members = Vec::new();
        if self.peek() != &Token::RightBracket {
            loop {
                members.push(self.expr_single()?);
                if !self.eat(&Token::Comma)? {
                    break;
                }
            }
        }
        let end = self.expect(&Token::RightBracket, "`]`")?.end;
        Ok(Syntax::new(start, end, SyntaxKind::Generic(members)))
    }

    /// A primary expression that begins with a name.
    fn named_primary(&mut self, name: &str) -> XQueryResult<Syntax> {
        let start = self.offset();
        if self.next_is(&Token::Hash)? {
            // XQuery 3.1 §3.1.6: a reserved function name may not head a named function
            // reference either, so `if#0` and `element#1` are `XPST0003`.
            if Self::is_reserved_function_name(name) {
                return Err(self.error(format!(
                    "`{name}` is a reserved function name and cannot be referenced"
                )));
            }
            self.advance()?;
            self.advance()?;
            let arity = self.advance()?;
            // A `NamedFunctionRef` names a function just as a call does, and the same static
            // rules apply to it — a `%private` function of an imported library module may not
            // be reached through `defs:f#0` either.
            if let Token::Integer(digits) = &arity.token {
                if let Ok(arity) = digits.parse::<usize>() {
                    self.record_call(start, name.to_owned(), arity);
                }
            }
            let end = arity.end;
            return Ok(Syntax::new(start, end, SyntaxKind::Generic(Vec::new())));
        }
        match name {
            "function" if self.next_is(&Token::LeftParen)? => return self.inline_function(),
            "ordered" | "unordered" if self.next_is(&Token::LeftBrace)? => {
                self.advance()?;
                self.advance()?;
                // `EnclosedExpr ::= "{" Expr? "}"`, so `ordered {}` is the empty sequence.
                let body = if self.peek() == &Token::RightBrace {
                    Syntax::empty(self.offset())
                } else {
                    self.expr()?
                };
                let end = self.expect(&Token::RightBrace, "`}`")?.end;
                return Ok(Syntax::native(start, end, SyntaxKind::Wrap(Box::new(body))));
            }
            "map" if self.next_is(&Token::LeftBrace)? => return self.map_constructor(),
            "array" if self.next_is(&Token::LeftBrace)? => return self.curly_array(),
            _ => {}
        }
        if COMPUTED_CONSTRUCTORS.contains(&name) {
            if let Some(parsed) = self.try_computed_constructor(name)? {
                return Ok(parsed);
            }
        }
        if self.next_is(&Token::LeftParen)? {
            return self.function_call();
        }
        Err(self.error(unexpected(self.peek(), "an expression")))
    }

    fn map_constructor(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        self.advance()?;
        self.expect(&Token::LeftBrace, "`{`")?;
        let mut parts = Vec::new();
        if self.peek() != &Token::RightBrace {
            loop {
                parts.push(self.expr_single()?);
                self.expect(&Token::Colon, "`:` in a map constructor")?;
                parts.push(self.expr_single()?);
                if !self.eat(&Token::Comma)? {
                    break;
                }
            }
        }
        let end = self.expect(&Token::RightBrace, "`}`")?.end;
        Ok(Syntax::new(start, end, SyntaxKind::Generic(parts)))
    }

    fn curly_array(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        self.advance()?;
        self.expect(&Token::LeftBrace, "`{`")?;
        let mut members = Vec::new();
        if self.peek() != &Token::RightBrace {
            members.push(self.expr()?);
        }
        let end = self.expect(&Token::RightBrace, "`}`")?.end;
        Ok(Syntax::new(start, end, SyntaxKind::Generic(members)))
    }

    fn function_call(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        let name = match self.peek().clone() {
            Token::Name(name) => Some(name),
            Token::BracedName { uri, local } => Some(format!("Q{{{uri}}}{local}")),
            _ => None,
        };
        self.advance()?;
        let (arguments, end) = self.argument_list()?;
        if let Some(name) = name {
            self.record_call(start, name, arguments.len());
        }
        let children = arguments.into_iter().flatten().collect();
        Ok(Syntax::new(start, end, SyntaxKind::Generic(children)))
    }

    /// `InlineFunctionExpr ::= Annotation* "function" "(" ParamList? ")" ("as" SequenceType)?
    /// FunctionBody`
    fn inline_function(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        let annotations = self.annotations()?;
        // XQuery 3.1 §3.1.7: `%public` and `%private` describe a *declaration*'s visibility, so
        // neither may annotate an inline function, which has no name to be visible under.
        if let Some(visibility) = annotations
            .iter()
            .find(|name| matches!(name.as_str(), "public" | "private"))
        {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0125,
                format!("`%{visibility}` may not annotate an inline function"),
            ));
        }
        self.expect_keyword("function")?;
        self.expect(&Token::LeftParen, "`(`")?;
        let mut parameters = Vec::new();
        if self.peek() != &Token::RightParen {
            loop {
                self.expect(&Token::Dollar, "`$`")?;
                let name = self.binding_name("a parameter name")?;
                let declared = self.optional_type_declaration()?;
                parameters.push((name, declared));
                if !self.eat(&Token::Comma)? {
                    break;
                }
            }
        }
        self.expect(&Token::RightParen, "`)`")?;
        let mut result_type = None;
        if self.at_keyword("as") {
            self.advance()?;
            result_type = Some(self.sequence_type_source()?);
        }
        self.expect(&Token::LeftBrace, "`{`")?;
        let body = if self.peek() == &Token::RightBrace {
            Syntax::empty(self.offset())
        } else {
            self.expr()?
        };
        let end = self.expect(&Token::RightBrace, "`}`")?.end;
        let inline = crate::ast::InlineFunction {
            parameters,
            result_type,
            body,
        };
        let kind = SyntaxKind::Inline(Box::new(inline));
        if annotations.is_empty() {
            Ok(Syntax::new(start, end, kind))
        } else {
            Ok(Syntax::native(start, end, kind))
        }
    }

    /// `Annotation* ::= ("%" EQName ("(" Literal ("," Literal)* ")")?)*`
    ///
    /// Every annotation's source range is recorded for blanking, so that an annotation that sits
    /// inside a form the XPath compiler will be handed — an inline function, or the annotation
    /// assertions of a `FunctionTest` — never reaches it. Annotations carry no meaning this
    /// processor acts on beyond the checks of §4.15, so eliding them is conformant.
    pub(crate) fn annotations(&mut self) -> XQueryResult<Vec<String>> {
        let mut annotations = Vec::new();
        while self.peek() == &Token::Percent {
            let start = self.offset();
            self.advance()?;
            let (name, _) = self.eqname("an annotation name")?;
            if self.eat(&Token::LeftParen)? {
                loop {
                    // An annotation's parameters are literals and nothing else.
                    match self.peek() {
                        Token::Integer(_)
                        | Token::Decimal(_)
                        | Token::Double(_)
                        | Token::Str { .. } => {
                            self.advance()?;
                        }
                        other => {
                            return Err(
                                self.error(unexpected(other, "a literal annotation parameter"))
                            );
                        }
                    }
                    if !self.eat(&Token::Comma)? {
                        break;
                    }
                }
                self.expect(&Token::RightParen, "`)`")?;
            }
            let end = self.last_end();
            self.blank(start, end);
            annotations.push(name);
        }
        Ok(annotations)
    }

    /// Whether a lexical function name is one the grammar reserves.
    pub(crate) fn is_reserved_function_name(name: &str) -> bool {
        !name.contains(':') && RESERVED_FUNCTION_NAMES.contains(&name)
    }

    fn fold_generic(mut operands: Vec<Syntax>) -> Syntax {
        if operands.len() == 1 {
            return operands.remove(0);
        }
        let start = operands.first().map_or(0, |first| first.start);
        let end = operands.last().map_or(start, |last| last.end);
        Syntax::new(start, end, SyntaxKind::Generic(operands))
    }
}

/// Scans the `EQName` that opens a pragma, returning it and how many bytes it took.
fn scan_pragma_name(text: &str) -> Option<(String, usize)> {
    if let Some(rest) = text.strip_prefix("Q{") {
        let close = rest.find('}')?;
        let uri = rest.get(..close)?;
        let after = rest.get(close + 1..)?;
        let (local, taken) = scan_ncname_prefix(after)?;
        return Some((format!("Q{{{uri}}}{local}"), 2 + close + 1 + taken));
    }
    let (first, mut taken) = scan_ncname_prefix(text)?;
    let mut name = first;
    if text.get(taken..).is_some_and(|rest| rest.starts_with(':')) {
        let (local, more) = scan_ncname_prefix(text.get(taken + 1..)?)?;
        name.push(':');
        name.push_str(&local);
        taken += 1 + more;
    }
    Some((name, taken))
}

/// Scans the `NCName` at the start of `text`.
fn scan_ncname_prefix(text: &str) -> Option<(String, usize)> {
    let mut end = 0;
    for (offset, c) in text.char_indices() {
        let first = offset == 0;
        if (first && !crate::lexer::is_name_start(c)) || (!first && !crate::lexer::is_name_char(c))
        {
            break;
        }
        end = offset + c.len_utf8();
    }
    (end > 0).then(|| (text.get(..end).unwrap_or_default().to_owned(), end))
}

/// The namespace `declare option output:…` lives in.
const SERIALIZATION_NAMESPACE: &str = "http://www.w3.org/2010/xslt-xquery-serialization";

/// The serialization parameters an *output declaration* may name.
///
/// [XQuery 3.1 §2.2.4](https://www.w3.org/TR/xquery-31/#id-serialization-declarations) says an
/// option in the serialization namespace whose local name is not one of these — or is
/// `use-character-maps`, which the XQuery serialization option syntax cannot express — is
/// `XQST0109`. The list is the parameter set of Serialization 3.1 §3.
const OUTPUT_PARAMETERS: &[&str] = &[
    "allow-duplicate-names",
    "byte-order-mark",
    "cdata-section-elements",
    "doctype-public",
    "doctype-system",
    "encoding",
    "escape-uri-attributes",
    "html-version",
    "include-content-type",
    "indent",
    "item-separator",
    "json-node-output-method",
    "media-type",
    "method",
    "normalization-form",
    "omit-xml-declaration",
    "parameter-document",
    "standalone",
    "suppress-indentation",
    "undeclare-prefixes",
    "version",
];

/// Checks the module's `declare option` declarations that are in the serialization namespace.
///
/// * `XQST0108` — an output declaration in a *library* module, where it has no meaning;
/// * `XQST0109` — a parameter name outside [`OUTPUT_PARAMETERS`];
/// * `XQST0110` — the same parameter declared twice.
///
/// Options in any other namespace are, per §4.16, ignored by a processor that does not
/// recognise them.
fn validate_output_declarations(module: &Module) -> XQueryResult<()> {
    let resolve = |lexical: &str| -> Option<(String, String)> {
        if let Some(rest) = lexical.strip_prefix("Q{") {
            let (uri, local) = rest.split_once('}')?;
            return Some((uri.to_owned(), local.to_owned()));
        }
        let (prefix, local) = lexical.split_once(':')?;
        let uri = module
            .prolog
            .namespaces
            .iter()
            .rev()
            .find(|(bound, _)| bound == prefix)
            .map(|(_, uri)| uri.clone())?;
        Some((uri, local.to_owned()))
    };
    let mut seen: Vec<String> = Vec::new();
    for (lexical, _) in &module.prolog.options {
        let Some((namespace, local)) = resolve(lexical) else {
            continue;
        };
        if namespace != SERIALIZATION_NAMESPACE {
            continue;
        }
        if module.module_namespace.is_some() {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0108,
                "an output declaration may not appear in a library module",
            ));
        }
        if !OUTPUT_PARAMETERS.contains(&local.as_str()) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0109,
                format!("`{local}` is not a serialization parameter an output declaration may set"),
            ));
        }
        if seen.contains(&local) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0110,
                format!("the serialization parameter `{local}` is declared more than once"),
            ));
        }
        seen.push(local);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> XQueryResult<Syntax> {
        let mut parser = Parser::new(source)?;
        let expr = parser.expr()?;
        assert_eq!(parser.peek(), &Token::Eof, "trailing input in `{source}`");
        Ok(expr)
    }

    #[test]
    fn pure_xpath_stays_clean() -> XQueryResult<()> {
        for source in [
            "1 + 2",
            "//book[@year < 1900]/title",
            "for $x in 1 to 3 return $x * $x",
            "let $x := 1 return $x",
            "some $x in (1, 2) satisfies $x > 1",
            "if (true()) then 1 else 2",
            "map { 'a': 1 }?a",
            "array { 1, 2 }(1)",
            "'a' || 'b'",
            "$x instance of xs:integer",
            "1 cast as xs:double",
            "a/b/c[1]",
            "(1, 2, 3)[. > 1]",
            "function($n) { $n }",
            "fn:concat#2",
            "@id",
            "..",
            "descendant-or-self::node()",
            "*:local",
            "p:*",
            "processing-instruction(x)",
            "1 ! (. + 1)",
        ] {
            let parsed = parse(source)?;
            assert!(!parsed.xquery_only, "`{source}` should be pure XPath");
            assert_eq!(parsed.start, 0, "`{source}` start");
            assert_eq!(parsed.end, source.len(), "`{source}` end");
        }
        Ok(())
    }

    #[test]
    fn xquery_only_syntax_is_flagged() -> XQueryResult<()> {
        for source in [
            "<a/>",
            "element a { 1 }",
            "for $x in 1 to 3 where $x > 1 return $x",
            "for $x at $i in (1, 2) return $i",
            "try { 1 } catch * { 2 }",
            "typeswitch (1) case xs:integer return 'i' default return 'o'",
            "switch (1) case 1 return 'one' default return 'other'",
            "ordered { 1 }",
            "'a&amp;b'",
            "text { 'x' }",
            "document { <a/> }",
        ] {
            let parsed = parse(source)?;
            assert!(parsed.xquery_only, "`{source}` should need this evaluator");
        }
        Ok(())
    }

    #[test]
    fn contamination_climbs_to_the_root() -> XQueryResult<()> {
        let parsed = parse("1 + count(<a/>)")?;
        assert!(parsed.xquery_only);
        Ok(())
    }

    #[test]
    fn syntax_errors_carry_an_offset() {
        let failure = parse("1 +").expect_err("a truncated expression");
        assert!(failure.is_static());
        assert_eq!(failure.offset(), Some(3));
    }
}
