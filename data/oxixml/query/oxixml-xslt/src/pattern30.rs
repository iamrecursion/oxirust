//! The `Pattern30` production of
//! [XSLT 3.0 §5.5.2](https://www.w3.org/TR/xslt-30/#pattern-syntax), checked syntactically.
//!
//! A pattern is *not* simply an XPath expression that a processor happens to evaluate
//! right-to-left: the specification gives it its own grammar, deliberately narrower than
//! `Expr`. `a/preceding-sibling::b` parses perfectly well as an expression and is nonetheless
//! `XTSE0340`, because `ReverseStepP` admits only the `parent` axis. Handing a pattern
//! straight to the XPath parser therefore accepts a language larger than XSLT defines, so this
//! module walks the pattern against the grammar itself before anything else looks at it.
//!
//! The grammar implemented here, verbatim from the specification:
//!
//! ```text
//! Pattern              ::= PredicatePattern | UnionExprP
//! PredicatePattern     ::= "." PredicateList
//! UnionExprP           ::= IntersectExceptExprP ( ("|" | "union") IntersectExceptExprP )*
//! IntersectExceptExprP ::= PathExprP ( ("intersect" | "except") PathExprP )*
//! PathExprP            ::= RootedPath | ("/" RelativePathExprP?) | ("//" RelativePathExprP)
//!                        | RelativePathExprP
//! RootedPath           ::= (VarRef | FunctionCallP) PredicateList
//!                          (("/" | "//") RelativePathExprP)?
//! FunctionCallP        ::= OuterFunctionName ArgumentList
//! OuterFunctionName    ::= "doc" | "id" | "element-with-id" | "key" | "root"
//! RelativePathExprP    ::= StepExprP (("/" | "//") StepExprP)*
//! StepExprP            ::= PostfixExprP | AxisStepP
//! PostfixExprP         ::= ParenthesizedExprP PredicateList
//! ParenthesizedExprP   ::= "(" UnionExprP ")"
//! AxisStepP            ::= (ForwardStepP | ReverseStepP) PredicateList
//! ForwardStepP         ::= (ForwardAxisP "::" NodeTest) | AbbrevForwardStep
//! ForwardAxisP         ::= "child" | "descendant" | "attribute" | "self"
//!                        | "descendant-or-self" | "namespace"
//! ReverseStepP         ::= ("parent" "::" NodeTest) | ".."
//! ```
//!
//! The contents of a predicate and the arguments of a `FunctionCallP` are full `Expr`s, which
//! the XPath compiler validates in its turn; here they are only skipped over as balanced
//! bracket runs.

use crate::error::{XsltError, XsltResult};

/// One lexical token of a pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    /// A name: an `NCName`, a `QName`, a `Q{uri}local`, or a wildcard form such as `*:local`.
    Name(String),
    /// A numeric or string literal.
    Literal,
    /// A balanced `( … )` run that follows a kind test or an `OuterFunctionName`, holding the
    /// text between the parentheses.
    Args(String),
    /// Punctuation or an operator, held verbatim.
    Punct(String),
}

/// The axes a `ForwardStepP` may name.
const FORWARD_AXES: &[&str] = &[
    "child",
    "descendant",
    "attribute",
    "self",
    "descendant-or-self",
    "namespace",
];

/// The kind tests a `NodeTest` may be.
const KIND_TESTS: &[&str] = &[
    "document-node",
    "element",
    "attribute",
    "schema-element",
    "schema-attribute",
    "processing-instruction",
    "comment",
    "text",
    "namespace-node",
    "node",
];

/// The functions `OuterFunctionName` admits, with and without the `fn:` prefix.
const OUTER_FUNCTIONS: &[&str] = &[
    "doc",
    "id",
    "element-with-id",
    "key",
    "root",
    "fn:doc",
    "fn:id",
    "fn:element-with-id",
    "fn:key",
    "fn:root",
];

/// Checks `source` against the `Pattern30` production.
///
/// # Errors
///
/// `XTSE0340` when the text does not match the production. A pattern this function accepts is
/// not thereby known to compile — the predicates it contains are still ordinary expressions.
pub(crate) fn check(source: &str) -> XsltResult<()> {
    let Some(tokens) = tokenize(source) else {
        // The text does not even lex — the XPath compiler will report the detail, and its
        // diagnosis is more useful than this module's.
        return Ok(());
    };
    let mut parser = Parser {
        tokens: &tokens,
        at: 0,
    };
    let matched = parser.pattern() && parser.at == tokens.len();
    if matched {
        return Ok(());
    }
    Err(XsltError::stat1c(
        "XTSE0340",
        format!("`{}` does not match the Pattern production", source.trim()),
    ))
}

struct Parser<'a> {
    tokens: &'a [Token],
    at: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at)
    }

    fn peek_punct(&self, text: &str) -> bool {
        matches!(self.peek(), Some(Token::Punct(found)) if found == text)
    }

    fn peek_name(&self, text: &str) -> bool {
        matches!(self.peek(), Some(Token::Name(found)) if found == text)
    }

    fn eat_punct(&mut self, text: &str) -> bool {
        let found = self.peek_punct(text);
        if found {
            self.at += 1;
        }
        found
    }

    /// `Pattern ::= PredicatePattern | UnionExprP`
    fn pattern(&mut self) -> bool {
        let start = self.at;
        if self.predicate_pattern() && self.at == self.tokens.len() {
            return true;
        }
        self.at = start;
        self.union()
    }

    /// `PredicatePattern ::= "." PredicateList`
    fn predicate_pattern(&mut self) -> bool {
        if !self.eat_punct(".") {
            return false;
        }
        self.predicate_list()
    }

    /// `UnionExprP ::= IntersectExceptExprP (("|" | "union") IntersectExceptExprP)*`
    fn union(&mut self) -> bool {
        if !self.intersect_except() {
            return false;
        }
        while self.peek_punct("|") || self.peek_name("union") {
            self.at += 1;
            if !self.intersect_except() {
                return false;
            }
        }
        true
    }

    /// `IntersectExceptExprP ::= PathExprP (("intersect" | "except") PathExprP)*`
    fn intersect_except(&mut self) -> bool {
        if !self.path() {
            return false;
        }
        while self.peek_name("intersect") || self.peek_name("except") {
            self.at += 1;
            if !self.path() {
                return false;
            }
        }
        true
    }

    /// `PathExprP ::= RootedPath | ("/" RelativePathExprP?) | ("//" RelativePathExprP)
    ///              | RelativePathExprP`
    fn path(&mut self) -> bool {
        let start = self.at;
        if self.rooted_path() {
            return true;
        }
        self.at = start;
        if self.eat_punct("//") {
            return self.relative_path();
        }
        if self.eat_punct("/") {
            let after = self.at;
            if self.relative_path() {
                return true;
            }
            // A lone `/` is a whole pattern; anything that follows it has to be a step, so
            // `/[doc]` is rejected rather than read as a filtered root.
            self.at = after;
            return true;
        }
        self.at = start;
        self.relative_path()
    }

    /// `RootedPath ::= (VarRef | FunctionCallP) PredicateList (("/" | "//") RelativePathExprP)?`
    fn rooted_path(&mut self) -> bool {
        if !(self.var_ref() || self.function_call()) {
            return false;
        }
        if !self.predicate_list() {
            return false;
        }
        if self.eat_punct("//") || self.eat_punct("/") {
            return self.relative_path();
        }
        true
    }

    fn var_ref(&mut self) -> bool {
        if !self.peek_punct("$") {
            return false;
        }
        self.at += 1;
        match self.peek() {
            Some(Token::Name(_)) => {
                self.at += 1;
                true
            }
            _ => false,
        }
    }

    /// `FunctionCallP ::= OuterFunctionName ArgumentList`
    fn function_call(&mut self) -> bool {
        let Some(Token::Name(name)) = self.peek() else {
            return false;
        };
        if !OUTER_FUNCTIONS.contains(&name.as_str()) {
            return false;
        }
        if !matches!(self.tokens.get(self.at + 1), Some(Token::Args(_))) {
            return false;
        }
        self.at += 2;
        true
    }

    /// `RelativePathExprP ::= StepExprP (("/" | "//") StepExprP)*`
    fn relative_path(&mut self) -> bool {
        if !self.step() {
            return false;
        }
        while self.peek_punct("/") || self.peek_punct("//") {
            self.at += 1;
            if !self.step() {
                return false;
            }
        }
        true
    }

    /// `StepExprP ::= PostfixExprP | AxisStepP`
    fn step(&mut self) -> bool {
        let start = self.at;
        if self.postfix() {
            return true;
        }
        self.at = start;
        self.axis_step()
    }

    /// `PostfixExprP ::= ParenthesizedExprP PredicateList`
    fn postfix(&mut self) -> bool {
        if !self.eat_punct("(") {
            return false;
        }
        if !self.union() {
            return false;
        }
        if !self.eat_punct(")") {
            return false;
        }
        self.predicate_list()
    }

    /// `AxisStepP ::= (ForwardStepP | ReverseStepP) PredicateList`
    ///
    /// An axis name followed by `::` is committed to as an axis: an axis the production does
    /// not list — every reverse axis but `parent` — fails the whole step rather than being
    /// re-read as the name test of an `AbbrevForwardStep`.
    fn axis_step(&mut self) -> bool {
        if let Some(Token::Name(name)) = self.peek().cloned()
            && matches!(self.tokens.get(self.at + 1), Some(Token::Punct(p)) if p == "::")
        {
            if !(FORWARD_AXES.contains(&name.as_str()) || name == "parent") {
                return false;
            }
            self.at += 2;
            return self.node_test() && self.predicate_list();
        }
        // `AbbrevReverseStep ::= ".."`
        if self.eat_punct("..") {
            return self.predicate_list();
        }
        // `AbbrevForwardStep ::= "@"? NodeTest`
        self.eat_punct("@");
        self.node_test() && self.predicate_list()
    }

    /// `NodeTest ::= KindTest | NameTest`
    fn node_test(&mut self) -> bool {
        match self.peek().cloned() {
            Some(Token::Name(name)) => {
                if let Some(Token::Args(inner)) = self.tokens.get(self.at + 1).cloned() {
                    if !KIND_TESTS.contains(&name.as_str()) {
                        return false;
                    }
                    self.at += 2;
                    return name != "processing-instruction" || is_pi_target(&inner);
                }
                self.at += 1;
                true
            }
            Some(Token::Punct(text)) if text == "*" => {
                self.at += 1;
                true
            }
            _ => false,
        }
    }

    /// `PredicateList ::= Predicate*`, each predicate already lexed as a single balanced token.
    fn predicate_list(&mut self) -> bool {
        while self.peek_punct("[]") {
            self.at += 1;
        }
        true
    }
}

/// Lexes a pattern, collapsing every balanced `[…]` predicate and `(…)` argument list that
/// belongs to a function call into a single token, and returning `None` when the text does not
/// lex at all.
fn tokenize(source: &str) -> Option<Vec<Token>> {
    let characters: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if character.is_whitespace() {
            index += 1;
            continue;
        }
        // An XPath comment, which may nest.
        if character == '(' && characters.get(index + 1) == Some(&':') {
            let mut depth = 1;
            index += 2;
            while index < characters.len() && depth > 0 {
                if characters[index] == '(' && characters.get(index + 1) == Some(&':') {
                    depth += 1;
                    index += 2;
                } else if characters[index] == ':' && characters.get(index + 1) == Some(&')') {
                    depth -= 1;
                    index += 2;
                } else {
                    index += 1;
                }
            }
            continue;
        }
        if character == '\'' || character == '"' {
            index = skip_string(&characters, index)?;
            tokens.push(Token::Literal);
            continue;
        }
        if character.is_ascii_digit() {
            while index < characters.len()
                && (characters[index].is_ascii_digit()
                    || characters[index] == '.'
                    || characters[index] == 'e'
                    || characters[index] == 'E'
                    || ((characters[index] == '+' || characters[index] == '-')
                        && matches!(
                            index.checked_sub(1).and_then(|back| characters.get(back)),
                            Some('e' | 'E')
                        )))
            {
                index += 1;
            }
            tokens.push(Token::Literal);
            continue;
        }
        if character == '[' {
            index = skip_balanced(&characters, index, '[', ']')?;
            tokens.push(Token::Punct("[]".to_owned()));
            continue;
        }
        if character == '(' {
            // A parenthesis that opens a kind test or an `OuterFunctionName`'s argument list is
            // collapsed with its partner, because its contents are an ordinary `Expr` that the
            // XPath compiler validates. Every other parenthesis stays visible: it either opens
            // a `ParenthesizedExprP`, or it belongs to a general function call, which is not a
            // `FunctionCallP` and must therefore fail.
            let collapse = matches!(
                tokens.last(),
                Some(Token::Name(name))
                    if KIND_TESTS.contains(&name.as_str())
                        || OUTER_FUNCTIONS.contains(&name.as_str())
            );
            if collapse {
                let end = skip_balanced(&characters, index, '(', ')')?;
                tokens.push(Token::Args(characters[index + 1..end - 1].iter().collect()));
                index = end;
                continue;
            }
            index += 1;
            tokens.push(Token::Punct("(".to_owned()));
            continue;
        }
        if character == 'Q' && characters.get(index + 1) == Some(&'{') {
            let end = skip_balanced(&characters, index + 1, '{', '}')?;
            let mut name: String = characters[index..end].iter().collect();
            index = end;
            while index < characters.len() && is_name_character(characters[index]) {
                name.push(characters[index]);
                index += 1;
            }
            if characters.get(index) == Some(&'*') {
                name.push('*');
                index += 1;
            }
            tokens.push(Token::Name(name));
            continue;
        }
        if is_name_start(character) {
            let start = index;
            while index < characters.len() && is_name_character(characters[index]) {
                index += 1;
            }
            let mut name: String = characters[start..index].iter().collect();
            // A `QName` or a `prefix:*` wildcard, but never the `::` of an axis.
            if characters.get(index) == Some(&':') && characters.get(index + 1) != Some(&':') {
                index += 1;
                name.push(':');
                if characters.get(index) == Some(&'*') {
                    name.push('*');
                    index += 1;
                } else {
                    while index < characters.len() && is_name_character(characters[index]) {
                        name.push(characters[index]);
                        index += 1;
                    }
                }
            }
            tokens.push(Token::Name(name));
            continue;
        }
        if character == '*' {
            // `*:local` is a name test, a bare `*` is the wildcard.
            if characters.get(index + 1) == Some(&':') {
                let start = index;
                index += 2;
                while index < characters.len() && is_name_character(characters[index]) {
                    index += 1;
                }
                tokens.push(Token::Name(characters[start..index].iter().collect()));
            } else {
                index += 1;
                tokens.push(Token::Punct("*".to_owned()));
            }
            continue;
        }
        let two: String = characters[index..(index + 2).min(characters.len())]
            .iter()
            .collect();
        if two == "//" || two == "::" || two == ".." {
            index += 2;
            tokens.push(Token::Punct(two));
            continue;
        }
        index += 1;
        tokens.push(Token::Punct(character.to_string()));
    }
    Some(tokens)
}

fn skip_string(characters: &[char], start: usize) -> Option<usize> {
    let quote = characters[start];
    let mut index = start + 1;
    while index < characters.len() {
        if characters[index] == quote {
            if characters.get(index + 1) == Some(&quote) {
                index += 2;
                continue;
            }
            return Some(index + 1);
        }
        index += 1;
    }
    None
}

/// The index just past the `close` that balances the `open` at `start`, skipping over string
/// literals so that a bracket inside one does not count.
fn skip_balanced(characters: &[char], start: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0usize;
    let mut index = start;
    while index < characters.len() {
        let character = characters[index];
        if character == '\'' || character == '"' {
            index = skip_string(characters, index)?;
            continue;
        }
        if character == open {
            depth += 1;
        } else if character == close {
            depth -= 1;
            if depth == 0 {
                return Some(index + 1);
            }
        }
        index += 1;
    }
    None
}

/// Whether the argument of a `processing-instruction()` test is a legal target: an `NCName` or
/// a string literal. `error-0340c` pins the `NCName` requirement down — a target written with a
/// colon, as `proc:inst-2`, is `XTSE0340` because a processing instruction has no namespace.
fn is_pi_target(inner: &str) -> bool {
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return true;
    }
    if (trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() > 1)
        || (trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() > 1)
    {
        return true;
    }
    oxixml_qname::is_ncname(trimmed)
}

fn is_name_start(character: char) -> bool {
    character.is_alphabetic() || character == '_'
}

fn is_name_character(character: char) -> bool {
    character.is_alphanumeric() || character == '_' || character == '-' || character == '.'
}

#[cfg(test)]
mod tests {
    use super::check;

    #[test]
    fn the_ordinary_forms_are_accepted() {
        for pattern in [
            "/",
            "*",
            "node()",
            "a",
            "a/b",
            "a//b",
            "/a/b",
            "//a",
            "@id",
            "attribute::id",
            "child::a/descendant::b",
            "parent::a/b",
            "../a",
            "a[1]",
            "a[b/c = 'x'][2]",
            "a | b",
            "a union b",
            "a intersect b except c",
            "processing-instruction('x')",
            "element(a, b)",
            "document-node(element(a))",
            "prefix:a/*:b/c:*",
            "Q{http://example.com/}a",
            ".",
            ".[. instance of xs:integer]",
            "$v/a/b",
            "key('k', 'v')",
            "id('x')/a",
            "doc('x.xml')//a",
            "(a | b)[1]/c",
            "text()",
            "@* except @q except @xml:id",
            "* except (xxx | yyy)",
            "* except Q{http://my.com/}z",
            "processing-instruction(proc)",
            "processing-instruction('proc')",
        ] {
            assert!(check(pattern).is_ok(), "`{pattern}` should be a pattern");
        }
    }

    #[test]
    fn the_restricted_forms_are_rejected() {
        for pattern in [
            // `ReverseStepP` admits only the `parent` axis.
            "a/preceding-sibling::b[1]",
            "following-sibling::a",
            "ancestor::a",
            // A filtered root is not a `PathExprP`.
            "/[doc]",
            // A predicate pattern is a whole pattern, never a branch of a union or the body of
            // a parenthesized expression.
            "element(foo) | .[. instance of xs:integer]",
            "(.[. instance of xs:integer])",
            // A literal is not a `StepExprP`.
            "name/1223",
            "'abc'",
            // A general function call is not a `FunctionCallP`.
            "my:test(.)",
            // A processing-instruction target is an `NCName`, never a `QName`.
            "processing-instruction(proc:inst-2)",
        ] {
            assert!(check(pattern).is_err(), "`{pattern}` is not a pattern");
        }
    }
}
