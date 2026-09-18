//! Node constructors: the direct XML-like syntax, the computed constructors, and the
//! `` `[ … ]` `` string constructor.
//!
//! Direct constructors are scanned from the raw query text rather than from tokens, because
//! their content is XML, not XQuery: `<a>1 < 2</a>` is not a comparison, `{{` is a literal brace
//! and `&#x20;` is a character reference. The scanner hands control back to the token stream
//! whenever it meets an enclosed expression, and takes it again at the closing brace — which is
//! exactly the "lexical state" split
//! [XQuery 3.1 A.2.4](https://www.w3.org/TR/xquery-31/#DirectConstructors) describes.
//!
//! The split has to hold for *nested* constructors too. `<a><b/>&amp;'</a>` leaves element
//! content when `<b/>` opens and re-enters it when `<b/>` closes, so the `&` and the apostrophe
//! after it are content characters — not the start of a token and not the start of a string
//! literal. Only [`Parser::direct_constructor`], the entry point the expression grammar reaches
//! through, resumes tokenizing; the raw scanners call [`Parser::scan_constructor`] instead and
//! stay in the XML state until the outermost constructor ends.

use super::Parser;
use crate::ast::{
    Constructor, ContentItem, DirectAttribute, DirectElement, NameSpec, Syntax, SyntaxKind,
};
use crate::error::{XQueryError, XQueryErrorCode, XQueryResult};
use crate::lexer::{Token, XML_SPACE, is_name_char, is_name_start, resolve_reference};

/// How deeply direct constructors may nest inside one another.
///
/// A nested constructor is scanned from the raw text rather than through
/// [`Parser::expr_single`](super::Parser::expr_single), so the expression grammar's own ceiling
/// never sees it and an unbounded `<a><a><a>…` would recurse until the thread's stack is gone —
/// a process abort rather than a catchable error. The figure is the workspace's ceiling on
/// *structural* nesting: an element tree, not an expression, is what is being counted here.
///
/// It bounds the *logical* depth, so the deepest input it still admits needs a thread with room
/// for it, exactly as the expression grammar's [`MAX_NESTING_DEPTH`](super::MAX_NESTING_DEPTH)
/// does — a scan level costs several kilobytes of native stack in an unoptimized build.
pub(crate) const MAX_CONSTRUCTOR_DEPTH: usize = 256;

/// A literal run of a constructor's content, and whether it may be boundary whitespace.
#[derive(Debug, Default)]
struct TextRun {
    text: String,
    /// Set when a character or entity reference, or a CDATA section, contributed to the run,
    /// which stops it from ever being boundary whitespace.
    protected: bool,
}

impl TextRun {
    fn flush(&mut self, into: &mut Vec<ContentItem>) {
        if self.text.is_empty() {
            // An empty run has nothing to protect — an empty CDATA section contributes no
            // characters, so it cannot stop the *next* run from being boundary whitespace.
            self.protected = false;
            return;
        }
        let text = std::mem::take(&mut self.text);
        let protected = std::mem::take(&mut self.protected);
        if !protected && text.chars().all(char::is_whitespace) {
            into.push(ContentItem::BoundaryWhitespace(text));
        } else {
            into.push(ContentItem::Text(text));
        }
    }
}

impl<'a> Parser<'a> {
    /// Scans a `<`-introduced constructor at the current token and resumes tokenizing after it.
    ///
    /// This is the expression grammar's entry point, so what follows the constructor really is
    /// an expression again. A constructor nested in another one's content goes through
    /// [`Parser::scan_constructor`], which leaves the scanner alone.
    pub(crate) fn direct_constructor(&mut self) -> XQueryResult<Syntax> {
        let parsed = self.scan_constructor(self.offset())?;
        let end = parsed.end;
        self.resume_at(end)?;
        Ok(parsed)
    }

    /// Scans a `<`-introduced constructor — an element, a comment or a processing instruction —
    /// from the raw text at `start`, without touching the token stream.
    fn scan_constructor(&mut self, start: usize) -> XQueryResult<Syntax> {
        let source = self.source();
        if source
            .get(start..)
            .is_some_and(|rest| rest.starts_with("<!--"))
        {
            return self.direct_comment(start);
        }
        if source
            .get(start..)
            .is_some_and(|rest| rest.starts_with("<?"))
        {
            return self.direct_processing_instruction(start);
        }
        let (element, end) = self.direct_element(start)?;
        Ok(Syntax::native(
            start,
            end,
            SyntaxKind::Construct(Box::new(Constructor::DirectElement(Box::new(element)))),
        ))
    }

    fn direct_comment(&mut self, start: usize) -> XQueryResult<Syntax> {
        let source = self.source();
        let body_start = start + 4;
        let Some(relative) = source.get(body_start..).and_then(|rest| rest.find("-->")) else {
            return Err(XQueryError::syntax("the comment is never closed", start));
        };
        let content = normalize_line_endings(
            source
                .get(body_start..body_start + relative)
                .unwrap_or_default(),
        );
        if content.contains("--") || content.ends_with('-') {
            return Err(XQueryError::syntax(
                "a direct comment constructor may not contain `--` or end with `-`",
                start,
            ));
        }
        let end = body_start + relative + 3;
        Ok(Syntax::native(
            start,
            end,
            SyntaxKind::Construct(Box::new(Constructor::DirectComment(content))),
        ))
    }

    fn direct_processing_instruction(&mut self, start: usize) -> XQueryResult<Syntax> {
        let source = self.source();
        let Some((target, after)) = scan_ncname(source, start + 2) else {
            return Err(XQueryError::syntax(
                "a processing instruction needs a target",
                start,
            ));
        };
        if target.eq_ignore_ascii_case("xml") {
            return Err(XQueryError::syntax(
                "`xml` is a reserved processing-instruction target",
                start,
            ));
        }
        // `DirPIConstructor ::= "<?" PITarget (S DirPIContents)? "?>"`: the target has to be a
        // whole NCName, so anything but whitespace or `?>` after it is a syntax error.
        let tail = source.get(after..).unwrap_or_default();
        if !tail.starts_with("?>") && !tail.starts_with(XML_SPACE) {
            return Err(XQueryError::syntax(
                "a processing-instruction target must be an NCName",
                after,
            ));
        }
        let Some(relative) = source.get(after..).and_then(|rest| rest.find("?>")) else {
            return Err(XQueryError::syntax(
                "the processing instruction is never closed",
                start,
            ));
        };
        let raw = source.get(after..after + relative).unwrap_or_default();
        // `DirPIConstructor ::= "<?" PITarget (S DirPIContents)? "?>"`: the whole `S` run
        // separates the target from the content, so none of it is content.
        let content = normalize_line_endings(raw.trim_start_matches(XML_SPACE));
        let end = after + relative + 2;
        Ok(Syntax::native(
            start,
            end,
            SyntaxKind::Construct(Box::new(Constructor::DirectProcessingInstruction {
                target,
                content,
            })),
        ))
    }

    /// Scans `<name attrs>content</name>` starting at `start`, returning the end offset.
    fn direct_element(&mut self, start: usize) -> XQueryResult<(DirectElement, usize)> {
        let source = self.source();
        let Some((name, mut position)) = scan_qname(source, start + 1) else {
            return Err(XQueryError::syntax(
                "an element name must follow `<`",
                start,
            ));
        };
        let mut namespaces = Vec::new();
        let mut attributes = Vec::new();
        loop {
            let after_space = skip_xml_space(source, position);
            let saw_space = after_space > position;
            position = after_space;
            match source.get(position..).and_then(|rest| rest.chars().next()) {
                Some('/') => {
                    if !source.get(position..).is_some_and(|r| r.starts_with("/>")) {
                        return Err(XQueryError::syntax("expected `/>`", position));
                    }
                    return Ok((
                        DirectElement {
                            name,
                            namespaces,
                            attributes,
                            content: Vec::new(),
                        },
                        position + 2,
                    ));
                }
                Some('>') => {
                    position += 1;
                    break;
                }
                Some(_) if saw_space => {
                    position =
                        self.direct_attribute(source, position, &mut namespaces, &mut attributes)?;
                }
                _ => return Err(XQueryError::syntax("expected `>` or `/>`", position)),
            }
        }
        let (content, position) = self.direct_content(position, &name)?;
        Ok((
            DirectElement {
                name,
                namespaces,
                attributes,
                content,
            },
            position,
        ))
    }

    fn direct_attribute(
        &mut self,
        source: &'a str,
        start: usize,
        namespaces: &mut Vec<(Option<String>, String)>,
        attributes: &mut Vec<DirectAttribute>,
    ) -> XQueryResult<usize> {
        let Some((name, mut position)) = scan_qname(source, start) else {
            return Err(XQueryError::syntax("expected an attribute name", start));
        };
        position = skip_xml_space(source, position);
        if !source.get(position..).is_some_and(|r| r.starts_with('=')) {
            return Err(XQueryError::syntax(
                "expected `=` after an attribute name",
                position,
            ));
        }
        position = skip_xml_space(source, position + 1);
        let quote = source
            .get(position..)
            .and_then(|rest| rest.chars().next())
            .filter(|c| *c == '"' || *c == '\'')
            .ok_or_else(|| XQueryError::syntax("expected a quoted attribute value", position))?;
        let (value, position) = self.attribute_value(source, position + 1, quote)?;

        let is_namespace = name == "xmlns" || name.starts_with("xmlns:");
        if is_namespace {
            let literal = match value.as_slice() {
                [] => String::new(),
                [ContentItem::Text(text)] => text.clone(),
                [ContentItem::BoundaryWhitespace(text)] => text.clone(),
                _ => {
                    return Err(XQueryError::new(
                        XQueryErrorCode::Xqst0022,
                        "the value of a namespace declaration attribute must be a literal",
                    )
                    .at(start));
                }
            };
            let prefix = name.strip_prefix("xmlns:").map(str::to_owned);
            namespaces.push((prefix, literal));
        } else {
            attributes.push(DirectAttribute { name, value });
        }
        Ok(position)
    }

    fn attribute_value(
        &mut self,
        source: &'a str,
        mut position: usize,
        quote: char,
    ) -> XQueryResult<(Vec<ContentItem>, usize)> {
        let mut parts = Vec::new();
        let mut run = TextRun::default();
        loop {
            let rest = source.get(position..).unwrap_or_default();
            let Some(c) = rest.chars().next() else {
                return Err(XQueryError::syntax(
                    "the attribute value is never closed",
                    position,
                ));
            };
            if c == quote {
                if rest
                    .get(c.len_utf8()..)
                    .is_some_and(|tail| tail.starts_with(quote))
                {
                    run.text.push(quote);
                    position += 2 * c.len_utf8();
                    continue;
                }
                run.flush(&mut parts);
                return Ok((parts, position + c.len_utf8()));
            }
            match c {
                '{' if rest.starts_with("{{") => {
                    run.text.push('{');
                    position += 2;
                }
                '}' if rest.starts_with("}}") => {
                    run.text.push('}');
                    position += 2;
                }
                '{' => {
                    run.flush(&mut parts);
                    let (expr, after) = self.enclosed_expression(position)?;
                    parts.push(ContentItem::Enclosed(expr));
                    position = after;
                }
                '}' => {
                    return Err(XQueryError::syntax(
                        "a lone `}` must be written `}}`",
                        position,
                    ));
                }
                '<' => {
                    return Err(XQueryError::syntax(
                        "`<` is not allowed in an attribute value",
                        position,
                    ));
                }
                '&' => {
                    let (text, after) = scan_reference_at(source, position)?;
                    run.text.push_str(&text);
                    run.protected = true;
                    position = after;
                }
                '\t' | '\n' => {
                    run.text.push(' ');
                    position += 1;
                }
                '\r' => {
                    run.text.push(' ');
                    position += if rest.starts_with("\r\n") { 2 } else { 1 };
                }
                _ => {
                    run.text.push(c);
                    position += c.len_utf8();
                }
            }
        }
    }

    /// Scans element content up to and including `</name>`.
    fn direct_content(
        &mut self,
        mut position: usize,
        name: &str,
    ) -> XQueryResult<(Vec<ContentItem>, usize)> {
        let source = self.source();
        let mut parts = Vec::new();
        let mut run = TextRun::default();
        loop {
            let rest = source.get(position..).unwrap_or_default();
            let Some(c) = rest.chars().next() else {
                return Err(XQueryError::syntax(
                    format!("the element `{name}` is never closed"),
                    position,
                ));
            };
            if rest.starts_with("</") {
                run.flush(&mut parts);
                let Some((closing, after)) = scan_qname(source, position + 2) else {
                    return Err(XQueryError::syntax(
                        "expected an element name after `</`",
                        position,
                    ));
                };
                if closing != name {
                    return Err(XQueryError::new(
                        XQueryErrorCode::Xqst0118,
                        format!("`</{closing}>` does not close `<{name}>`"),
                    )
                    .at(position));
                }
                let after = skip_xml_space(source, after);
                if !source.get(after..).is_some_and(|r| r.starts_with('>')) {
                    return Err(XQueryError::syntax("expected `>`", after));
                }
                return Ok((parts, after + 1));
            }
            if rest.starts_with("<![CDATA[") {
                let body = position + 9;
                let Some(relative) = source.get(body..).and_then(|tail| tail.find("]]>")) else {
                    return Err(XQueryError::syntax(
                        "the CDATA section is never closed",
                        position,
                    ));
                };
                let text =
                    normalize_line_endings(source.get(body..body + relative).unwrap_or_default());
                // §3.9.1.4: characters generated by a CDataSection are not whitespace characters
                // for the purpose of finding boundary whitespace, so a CDATA section joins the
                // surrounding literal run and protects the whole of it.
                run.text.push_str(&text);
                run.protected = true;
                position = body + relative + 3;
                continue;
            }
            if c == '<' {
                run.flush(&mut parts);
                let nested = self.nested_constructor(position)?;
                position = nested.1;
                parts.push(ContentItem::Nested(nested.0));
                continue;
            }
            match c {
                '{' if rest.starts_with("{{") => {
                    run.text.push('{');
                    position += 2;
                }
                '}' if rest.starts_with("}}") => {
                    run.text.push('}');
                    position += 2;
                }
                '{' => {
                    run.flush(&mut parts);
                    let (expr, after) = self.enclosed_expression(position)?;
                    parts.push(ContentItem::Enclosed(expr));
                    position = after;
                }
                '}' => {
                    return Err(XQueryError::syntax(
                        "a lone `}` must be written `}}`",
                        position,
                    ));
                }
                '&' => {
                    let (text, after) = scan_reference_at(source, position)?;
                    run.text.push_str(&text);
                    run.protected = true;
                    position = after;
                }
                '\r' => {
                    run.text.push('\n');
                    position += if rest.starts_with("\r\n") { 2 } else { 1 };
                }
                _ => {
                    run.text.push(c);
                    position += c.len_utf8();
                }
            }
        }
    }

    /// Parses a nested `<`-introduced constructor inside element content.
    ///
    /// The scanner stays in the XML lexical state throughout: what follows the nested
    /// constructor is more of the enclosing element's content, where `&`, `'` and `"` are
    /// ordinary characters rather than the start of a reference or of a string literal.
    fn nested_constructor(&mut self, at: usize) -> XQueryResult<(Syntax, usize)> {
        if self.constructor_depth >= MAX_CONSTRUCTOR_DEPTH {
            return Err(XQueryError::syntax(
                format!(
                    "direct constructors nest more than {MAX_CONSTRUCTOR_DEPTH} levels deep, \
                     which is this implementation's limit"
                ),
                at,
            ));
        }
        self.constructor_depth += 1;
        let parsed = self.scan_constructor(at);
        self.constructor_depth -= 1;
        let parsed = parsed?;
        let end = parsed.end;
        Ok((parsed, end))
    }

    /// Parses `{ Expr? }` starting at the brace, returning the expression and the offset just
    /// past the closing brace.
    pub(crate) fn enclosed_expression(&mut self, at: usize) -> XQueryResult<(Syntax, usize)> {
        self.resume_at(at)?;
        self.expect(&Token::LeftBrace, "`{`")?;
        let expr = if self.peek() == &Token::RightBrace {
            Syntax::empty(self.offset())
        } else {
            self.expr()?
        };
        let end = self.close_brace()?;
        Ok((expr, end))
    }

    /// Parses `{ Expr? }` at the current token.
    fn enclosed_here(&mut self) -> XQueryResult<Option<Syntax>> {
        self.expect(&Token::LeftBrace, "`{`")?;
        if self.peek() == &Token::RightBrace {
            self.advance()?;
            return Ok(None);
        }
        let expr = self.expr()?;
        self.expect(&Token::RightBrace, "`}`")?;
        Ok(Some(expr))
    }

    /// `CompElementConstructor` and friends, when `name` really does open one.
    pub(crate) fn try_computed_constructor(&mut self, name: &str) -> XQueryResult<Option<Syntax>> {
        if !self.starts_computed_constructor(name)? {
            return Ok(None);
        }
        let start = self.offset();
        self.advance()?;
        let constructor = match name {
            "document" => {
                let content = self.enclosed_here()?;
                Constructor::ComputedDocument(
                    content.unwrap_or_else(|| Syntax::empty(self.last_end())),
                )
            }
            "text" => Constructor::ComputedText(self.enclosed_here()?),
            "comment" => Constructor::ComputedComment(self.enclosed_here()?),
            "element" => {
                let name = self.name_specifier()?;
                Constructor::ComputedElement {
                    name,
                    content: self.enclosed_here()?,
                }
            }
            "attribute" => {
                let name = self.name_specifier()?;
                Constructor::ComputedAttribute {
                    name,
                    content: self.enclosed_here()?,
                }
            }
            "processing-instruction" => {
                // `CompPIConstructor ::= "processing-instruction" (NCName | ("{" Expr "}"))
                // EnclosedExpr` — a *lexical* PI target is an `NCName`, never a `QName`, so
                // `processing-instruction foo:pi {…}` does not parse at all.
                let name = self.name_specifier()?;
                if let NameSpec::Lexical(lexical) = &name {
                    if lexical.contains(':') {
                        return Err(self.error(format!(
                            "`{lexical}` is not an NCName; a processing instruction's target \
                             has no namespace"
                        )));
                    }
                }
                Constructor::ComputedProcessingInstruction {
                    name,
                    content: self.enclosed_here()?,
                }
            }
            "namespace" => {
                let prefix = self.name_specifier()?;
                Constructor::ComputedNamespace {
                    prefix,
                    uri: self.enclosed_here()?,
                }
            }
            _ => return Ok(None),
        };
        let end = self.last_end();
        Ok(Some(Syntax::native(
            start,
            end,
            SyntaxKind::Construct(Box::new(constructor)),
        )))
    }

    fn name_specifier(&mut self) -> XQueryResult<NameSpec> {
        if self.peek() == &Token::LeftBrace {
            self.advance()?;
            let expr = self.expr()?;
            self.expect(&Token::RightBrace, "`}`")?;
            return Ok(NameSpec::Computed(Box::new(expr)));
        }
        let (name, _) = self.eqname("a constructor name")?;
        Ok(NameSpec::Lexical(name))
    }

    /// `` `[ … ]` `` — the string constructor of XQuery 3.1.
    pub(crate) fn string_constructor(&mut self) -> XQueryResult<Syntax> {
        let start = self.offset();
        let source = self.source();
        if !source
            .get(start..)
            .is_some_and(|rest| rest.starts_with("``["))
        {
            return Err(XQueryError::syntax(
                "a string constructor starts with `` ``[ ``",
                start,
            ));
        }
        let mut position = start + 3;
        let mut parts = Vec::new();
        let mut text = String::new();
        loop {
            let rest = source.get(position..).unwrap_or_default();
            if rest.starts_with("]``") {
                if !text.is_empty() {
                    parts.push(ContentItem::Text(std::mem::take(&mut text)));
                }
                position += 3;
                break;
            }
            if rest.starts_with("`{") {
                if !text.is_empty() {
                    parts.push(ContentItem::Text(std::mem::take(&mut text)));
                }
                self.resume_at(position + 1)?;
                self.expect(&Token::LeftBrace, "`{`")?;
                let expr = if self.peek() == &Token::RightBrace {
                    Syntax::empty(self.offset())
                } else {
                    self.expr()?
                };
                let after = self.close_brace()?;
                if !source
                    .get(after..)
                    .is_some_and(|tail| tail.starts_with('`'))
                {
                    return Err(XQueryError::syntax(
                        "an interpolation must be closed with `` }` ``",
                        after,
                    ));
                }
                parts.push(ContentItem::Enclosed(expr));
                position = after + 1;
                continue;
            }
            let Some(c) = rest.chars().next() else {
                return Err(XQueryError::syntax(
                    "the string constructor is never closed",
                    start,
                ));
            };
            text.push(c);
            position += c.len_utf8();
        }
        self.resume_at(position)?;
        Ok(Syntax::native(
            start,
            position,
            SyntaxKind::StringTemplate(parts),
        ))
    }
}

// --- raw scanning helpers ---------------------------------------------------------------------

fn scan_ncname(source: &str, start: usize) -> Option<(String, usize)> {
    let rest = source.get(start..)?;
    let mut end = start;
    for (offset, c) in rest.char_indices() {
        let first = offset == 0;
        if (first && !is_name_start(c)) || (!first && !is_name_char(c)) {
            break;
        }
        end = start + offset + c.len_utf8();
    }
    if end == start {
        return None;
    }
    Some((source.get(start..end)?.to_owned(), end))
}

fn scan_qname(source: &str, start: usize) -> Option<(String, usize)> {
    let (first, mut end) = scan_ncname(source, start)?;
    let mut name = first;
    if source.get(end..).is_some_and(|rest| rest.starts_with(':')) {
        if let Some((local, after)) = scan_ncname(source, end + 1) {
            name.push(':');
            name.push_str(&local);
            end = after;
        }
    }
    Some((name, end))
}

fn skip_xml_space(source: &str, start: usize) -> usize {
    let mut position = start;
    while let Some(c) = source.get(position..).and_then(|rest| rest.chars().next()) {
        if matches!(c, ' ' | '\t' | '\n' | '\r') {
            position += c.len_utf8();
        } else {
            break;
        }
    }
    position
}

fn scan_reference_at(source: &str, start: usize) -> XQueryResult<(String, usize)> {
    let rest = source.get(start + 1..).unwrap_or_default();
    let Some(relative) = rest.find(';') else {
        return Err(XQueryError::syntax(
            "the entity or character reference is never closed",
            start,
        ));
    };
    let body = rest.get(..relative).unwrap_or_default();
    let text = resolve_reference(body, start)?;
    Ok((text, start + 1 + relative + 1))
}

fn normalize_line_endings(text: &str) -> String {
    if !text.contains('\r') {
        return text.to_owned();
    }
    text.replace("\r\n", "\n").replace('\r', "\n")
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

    fn element_of(parsed: &Syntax) -> &DirectElement {
        match &parsed.kind {
            SyntaxKind::Construct(constructor) => match constructor.as_ref() {
                Constructor::DirectElement(element) => element,
                other => panic!("expected a direct element, got {other:?}"),
            },
            other => panic!("expected a constructor, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_element_parses() -> XQueryResult<()> {
        let parsed = parse("<a/>")?;
        let element = element_of(&parsed);
        assert_eq!(element.name, "a");
        assert!(element.content.is_empty());
        assert_eq!(parsed.end, 4);
        Ok(())
    }

    #[test]
    fn attributes_and_namespaces_are_separated() -> XQueryResult<()> {
        let parsed = parse(r#"<a xmlns:p="urn:p" x="1" y="{2}"/>"#)?;
        let element = element_of(&parsed);
        assert_eq!(
            element.namespaces,
            vec![(Some("p".to_owned()), "urn:p".to_owned())]
        );
        assert_eq!(element.attributes.len(), 2);
        assert!(matches!(
            element.attributes[1].value.as_slice(),
            [ContentItem::Enclosed(_)]
        ));
        Ok(())
    }

    #[test]
    fn boundary_whitespace_is_marked() -> XQueryResult<()> {
        let parsed = parse("<a> <b/> </a>")?;
        let element = element_of(&parsed);
        assert!(matches!(
            element.content.as_slice(),
            [
                ContentItem::BoundaryWhitespace(_),
                ContentItem::Nested(_),
                ContentItem::BoundaryWhitespace(_)
            ]
        ));
        Ok(())
    }

    #[test]
    fn a_reference_protects_its_run() -> XQueryResult<()> {
        let parsed = parse("<a>&#x20;</a>")?;
        let element = element_of(&parsed);
        assert!(matches!(
            element.content.as_slice(),
            [ContentItem::Text(text)] if text == " "
        ));
        Ok(())
    }

    #[test]
    fn braces_double_to_escape() -> XQueryResult<()> {
        let parsed = parse("<a>{{x}}</a>")?;
        let element = element_of(&parsed);
        assert!(matches!(
            element.content.as_slice(),
            [ContentItem::Text(text)] if text == "{x}"
        ));
        Ok(())
    }

    #[test]
    fn cdata_is_literal_text() -> XQueryResult<()> {
        let parsed = parse("<a><![CDATA[1 < 2 & {x}]]></a>")?;
        let element = element_of(&parsed);
        assert!(matches!(
            element.content.as_slice(),
            [ContentItem::Text(text)] if text == "1 < 2 & {x}"
        ));
        Ok(())
    }

    #[test]
    fn nested_elements_and_expressions_interleave() -> XQueryResult<()> {
        let parsed = parse("<a>x{1 + 1}<b y='2'>z</b></a>")?;
        let element = element_of(&parsed);
        assert_eq!(element.content.len(), 3);
        Ok(())
    }

    #[test]
    fn a_mismatched_end_tag_is_a_syntax_error() {
        assert!(parse("<a></b>").is_err());
    }

    #[test]
    fn comments_and_processing_instructions_parse() -> XQueryResult<()> {
        let parsed = parse("<!-- hi -->")?;
        assert!(matches!(
            &parsed.kind,
            SyntaxKind::Construct(c) if matches!(c.as_ref(), Constructor::DirectComment(text) if text == " hi ")
        ));
        let parsed = parse("<?php echo ?>")?;
        assert!(matches!(
            &parsed.kind,
            SyntaxKind::Construct(c)
                if matches!(c.as_ref(), Constructor::DirectProcessingInstruction { target, content }
                    if target == "php" && content == "echo ")
        ));
        Ok(())
    }

    #[test]
    fn a_direct_comment_with_a_double_hyphen_is_a_syntax_error() {
        let failure = parse("<!-- a -- b -->").expect_err("`--` is illegal");
        assert!(failure.is(XQueryErrorCode::Xpst0003));
    }

    #[test]
    fn a_direct_processing_instruction_may_not_target_xml() {
        let failure = parse("<?XmL data?>").expect_err("`xml` is reserved");
        assert!(failure.is(XQueryErrorCode::Xpst0003));
    }

    #[test]
    fn computed_constructors_parse() -> XQueryResult<()> {
        for source in [
            "element a { 1 }",
            "element { 'a' } { 1 }",
            "attribute a { 1 }",
            "document { <a/> }",
            "text { 'x' }",
            "comment { 'x' }",
            "processing-instruction p { 'x' }",
            "namespace p { 'urn:x' }",
        ] {
            let parsed = parse(source)?;
            assert!(
                matches!(parsed.kind, SyntaxKind::Construct(_)),
                "`{source}`"
            );
            assert_eq!(parsed.end, source.len(), "`{source}`");
        }
        Ok(())
    }

    #[test]
    fn a_string_constructor_interleaves_text_and_expressions() -> XQueryResult<()> {
        let parsed = parse("``[a`{1 + 1}`b]``")?;
        match &parsed.kind {
            SyntaxKind::StringTemplate(parts) => assert_eq!(parts.len(), 3),
            other => panic!("expected a string template, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn namespace_attributes_must_be_literal() {
        let failure = parse(r#"<a xmlns:p="{1}"/>"#).expect_err("not a literal");
        assert!(failure.is(XQueryErrorCode::Xqst0022));
    }

    /// The literal text of an element's content, with enclosed expressions left out.
    fn literal_content(element: &DirectElement) -> String {
        element
            .content
            .iter()
            .filter_map(|item| match item {
                ContentItem::Text(text) | ContentItem::BoundaryWhitespace(text) => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn content_after_a_nested_constructor_is_still_content() -> XQueryResult<()> {
        // Closing a nested constructor returns to the XML lexical state, so none of these
        // characters starts a token: an apostrophe or a quotation mark is a literal character,
        // `&…;` is a reference and `}}` is an escaped brace.
        for (source, expected) in [
            ("<a><b/>'</a>", "'"),
            ("<a><b/>\"</a>", "\""),
            ("<a><b/>&amp;</a>", "&"),
            ("<a><b/>&apos;x&quot;</a>", "'x\""),
            ("<a><b/>}}</a>", "}"),
            ("<a><b/>{{</a>", "{"),
            ("<a><b/>></a>", ">"),
            ("<a><b/>#</a>", "#"),
            ("<a><b/><![CDATA[']]></a>", "'"),
            ("<a><!-- c -->'&amp;</a>", "'&"),
            ("<a><?p data?>'&amp;</a>", "'&"),
            ("<a><b>'</b>'</a>", "'"),
        ] {
            let parsed = parse(source)?;
            let element = element_of(&parsed);
            assert_eq!(literal_content(element), expected, "`{source}`");
            assert_eq!(parsed.end, source.len(), "`{source}`");
        }
        Ok(())
    }

    #[test]
    fn a_script_element_may_hold_quoted_markup() -> XQueryResult<()> {
        // The `ser/method-html` catalogue writes a script body that mixes apostrophes, entity
        // references and a nested element; the apostrophes are content, not a string literal.
        let source = "<html><head><script language=\"Jack&amp;Jill\">go &amp;&amp; \
                      write('<p class=\"Bill&amp;Ben\">Jekyll&amp;Hyde</p>');</script></head>\
                      <body><p>Hi!</p></body></html>";
        let parsed = parse(source)?;
        assert_eq!(parsed.end, source.len());
        Ok(())
    }

    #[test]
    fn enclosed_expressions_still_hand_the_scanner_back() -> XQueryResult<()> {
        // An enclosed expression *is* an expression, so a string literal inside one is a string
        // literal — and what follows the closing brace is content again.
        let parsed = parse("<a>{'x'}'{\"y\"}&amp;</a>")?;
        let element = element_of(&parsed);
        assert_eq!(literal_content(element), "'&");
        assert_eq!(element.content.len(), 4);
        let parsed = parse("<a>{ <b>{ 'x' }</b> }'</a>")?;
        assert_eq!(literal_content(element_of(&parsed)), "'");
        Ok(())
    }

    #[test]
    fn an_attribute_value_may_follow_an_enclosed_expression() -> XQueryResult<()> {
        let parsed = parse(r#"<a x="{1}&amp;{{y}}"><b/>'</a>"#)?;
        let element = element_of(&parsed);
        assert_eq!(literal_content(element), "'");
        assert_eq!(element.attributes.len(), 1);
        Ok(())
    }

    fn nest(levels: usize) -> String {
        let mut source = String::with_capacity(levels * 7);
        for _ in 0..levels {
            source.push_str("<a>");
        }
        for _ in 0..levels {
            source.push_str("</a>");
        }
        source
    }

    #[test]
    fn nesting_is_bounded_rather_than_exhausting_the_stack() {
        // Deep-but-legal nesting costs a stack frame per level, so the check runs where there is
        // room for the deepest input the ceiling still admits — the same arrangement the
        // expression grammar's own nesting test uses.
        let outcome = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                // The outermost constructor is not itself nested, so the ceiling admits one more
                // level than it counts.
                let accepted = parse(&nest(MAX_CONSTRUCTOR_DEPTH + 1)).is_ok();
                let refused = parse(&nest(MAX_CONSTRUCTOR_DEPTH + 2))
                    .err()
                    .is_some_and(|failure| failure.is(XQueryErrorCode::Xpst0003));
                // A million levels is the case the ceiling exists for: without it the scan
                // recurses until the thread's stack is gone, which aborts the process.
                let bounded = parse(&nest(1_000_000))
                    .err()
                    .is_some_and(|failure| failure.is(XQueryErrorCode::Xpst0003));
                (accepted, refused, bounded)
            })
            .expect("the helper thread starts")
            .join()
            .expect("the helper thread finishes");
        assert_eq!(outcome, (true, true, true));
    }
}
