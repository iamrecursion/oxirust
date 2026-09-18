//! Terminal productions: IRIs, prefixed names, literals, variables, blank nodes and escapes.

use super::{Parser, R};
use crate::term::{BlankNode, Literal, NamedNode, Variable};
use oxixml_iri::{Iri, IriRef};
use oxixml_model::vocab::xsd;

impl Parser<'_> {
    // --- IRIs ---------------------------------------------------------------

    /// `iri` — an `IRIREF` or a `PrefixedName`, returning a resolved [`NamedNode`].
    pub(crate) fn iri(&mut self) -> R<NamedNode> {
        if self.peek_char() == Some('<') {
            let s = self.iriref()?;
            Ok(NamedNode::new_unchecked(s))
        } else {
            let s = self.prefixed_name()?;
            Ok(NamedNode::new_unchecked(s))
        }
    }

    /// `IRIREF` — returns the resolved absolute IRI string.
    pub(crate) fn iriref(&mut self) -> R<String> {
        self.eat("<")?;
        let start = self.pos;
        let bytes = self.input.as_bytes();
        while self.pos < bytes.len() && bytes[self.pos] != b'>' {
            // Reject the characters the IRIREF production forbids explicitly.
            match bytes[self.pos] {
                b'<' | b'"' | b'{' | b'}' | b'|' | b'^' | b'`' => {
                    return self.mk_err("invalid IRI");
                }
                0..=0x20 => return self.mk_err("invalid IRI"),
                _ => self.pos += 1,
            }
        }
        if self.pos >= bytes.len() {
            return self.mk_err("unterminated IRI");
        }
        let raw = &self.input[start..self.pos];
        self.pos += 1; // consume '>'
        let unescaped = match unescape_iriref(raw) {
            Ok(v) => v,
            Err(_) => return self.mk_err("invalid IRI escape"),
        };
        match self.resolve_iri(unescaped) {
            Ok(v) => Ok(v),
            Err(()) => self.mk_err("IRI resolution failed"),
        }
    }

    /// Whether a complete `IRIREF` token starts at the cursor, without consuming anything.
    ///
    /// SPARQL tokenizes with the longest-token rule, so a `<` that opens a well-formed `IRIREF`
    /// is never the "less than" operator. This is what makes `FILTER (?x<?a&&?b>?y)` a syntax
    /// error instead of `(?x < ?a) && (?b > ?y)`.
    pub(crate) fn at_iriref(&self) -> bool {
        let bytes = self.input.as_bytes();
        if bytes.get(self.pos) != Some(&b'<') {
            return false;
        }
        let mut i = self.pos + 1;
        while let Some(&b) = bytes.get(i) {
            match b {
                b'>' => return true,
                b'<' | b'"' | b'{' | b'}' | b'|' | b'^' | b'`' | 0..=0x20 => return false,
                _ => i += 1,
            }
        }
        false
    }

    fn resolve_iri(&self, iri: String) -> Result<String, ()> {
        if let Some(base) = &self.base_iri {
            let reference = IriRef::parse(iri).map_err(|_| ())?;
            base.resolve(&reference)
                .map(|i| i.into_inner())
                .map_err(|_| ())
        } else {
            Iri::parse(iri.clone()).map(|_| iri).map_err(|_| ())
        }
    }

    /// `PrefixedName` — `PNAME_LN` or `PNAME_NS`, resolved against declared prefixes.
    pub(crate) fn prefixed_name(&mut self) -> R<String> {
        let ns = self.pname_ns()?;
        // Optional local part.
        let local_start = self.pos;
        let local_len = scan_pn_local(self.rest());
        if local_len > 0 {
            let local = &self.input[local_start..local_start + local_len];
            self.pos += local_len;
            let Some(base) = self.prefixes.get(&ns) else {
                return self.mk_err("undefined prefix");
            };
            let mut iri = String::with_capacity(base.len() + local.len());
            iri.push_str(base);
            for chunk in local.split('\\') {
                iri.push_str(chunk);
            }
            match Iri::parse(iri) {
                Ok(v) => Ok(v.into_inner()),
                Err(_) => self.mk_err("invalid IRI"),
            }
        } else {
            let Some(base) = self.prefixes.get(&ns) else {
                return self.mk_err("undefined prefix");
            };
            match Iri::parse(base.clone()) {
                Ok(v) => Ok(v.into_inner()),
                Err(_) => self.mk_err("invalid IRI"),
            }
        }
    }

    /// `PNAME_NS` — an optional prefix followed by `:`. Returns the prefix text.
    pub(crate) fn pname_ns(&mut self) -> R<String> {
        let start = self.pos;
        let len = scan_pn_prefix(self.rest());
        let prefix = self.input[start..start + len].to_owned();
        self.pos += len;
        self.eat(":")?;
        Ok(prefix)
    }

    // --- variables ----------------------------------------------------------

    /// `Var` — `?name` or `$name`.
    pub(crate) fn var(&mut self) -> R<Variable> {
        match self.peek_char() {
            Some('?') | Some('$') => {
                self.pos += 1;
                let start = self.pos;
                let len = scan_varname(self.rest());
                if len == 0 {
                    return self.mk_err("expected variable name");
                }
                let name = self.input[start..start + len].to_owned();
                self.pos += len;
                Ok(Variable::new_unchecked(name))
            }
            _ => self.mk_err("expected variable"),
        }
    }

    // --- blank nodes --------------------------------------------------------

    /// `BlankNode` — `_:label` or `[]`.
    pub(crate) fn blank_node(&mut self) -> R<BlankNode> {
        if self.rest().starts_with("_:") {
            self.pos += 2;
            let start = self.pos;
            let len = scan_blank_node_label(self.rest());
            if len == 0 {
                return self.mk_err("expected blank node label");
            }
            let label = self.input[start..start + len].to_owned();
            self.pos += len;
            if self.used_bnodes.contains(&label) {
                return self.fatal("Re-used blank node label across basic graph patterns");
            }
            self.currently_used_bnodes.insert(label.clone());
            Ok(BlankNode::new_unchecked(label))
        } else {
            // ANON: '[' WS* ']'
            self.eat("[")?;
            self.ws_only();
            self.eat("]")?;
            Ok(self.fresh_bnode())
        }
    }

    /// Consumes only whitespace (no comments), used inside `NIL` and `ANON`.
    fn ws_only(&mut self) {
        let bytes = self.input.as_bytes();
        while self.pos < bytes.len() {
            match bytes[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    /// `NIL` — `'(' WS* ')'`.
    pub(crate) fn nil(&mut self) -> R<()> {
        self.eat("(")?;
        self.ws_only();
        self.eat(")")
    }

    // --- literals -----------------------------------------------------------

    /// `RDFLiteral` — a string with optional datatype or language tag.
    pub(crate) fn rdf_literal(&mut self) -> R<Literal> {
        let value = self.string()?;
        let save = self.pos;
        self.ws();
        if self.try_str("^^") {
            self.ws();
            let datatype = self.iri()?;
            return Ok(Literal::new_typed_literal(value, datatype));
        }
        self.pos = save;
        if self.peek_char() == Some('@') {
            let lang = self.langtag()?;
            #[cfg(feature = "sparql-12")]
            if let Some(direction) = self.base_direction() {
                return Ok(Self::directional_literal(value, lang, direction));
            }
            return Ok(Literal::new_language_tagged_literal_unchecked(value, lang));
        }
        Ok(Literal::new_simple_literal(value))
    }

    /// `BooleanLiteral` — `true` or `false`.
    pub(crate) fn boolean_literal(&mut self) -> R<Literal> {
        if self.try_str("true") {
            Ok(Literal::new_typed_literal("true", xsd::BOOLEAN))
        } else if self.try_str("false") {
            Ok(Literal::new_typed_literal("false", xsd::BOOLEAN))
        } else {
            self.mk_err("expected boolean literal")
        }
    }

    /// `NumericLiteral` — integer, decimal or double, optionally signed.
    pub(crate) fn numeric_literal(&mut self) -> R<Literal> {
        let rest = self.rest();
        // Optional sign.
        let (sign_len, body) = match rest.as_bytes().first() {
            Some(b'+') | Some(b'-') => (1, &rest[1..]),
            _ => (0, rest),
        };
        let dl = scan_double(body);
        if dl > 0 {
            let total = sign_len + dl;
            let text = self.input[self.pos..self.pos + total].to_owned();
            self.pos += total;
            return Ok(Literal::new_typed_literal(text, xsd::DOUBLE));
        }
        let del = scan_decimal(body);
        if del > 0 {
            let total = sign_len + del;
            let text = self.input[self.pos..self.pos + total].to_owned();
            self.pos += total;
            return Ok(Literal::new_typed_literal(text, xsd::DECIMAL));
        }
        let il = scan_integer(body);
        if il > 0 {
            let total = sign_len + il;
            let text = self.input[self.pos..self.pos + total].to_owned();
            self.pos += total;
            return Ok(Literal::new_typed_literal(text, xsd::INTEGER));
        }
        self.mk_err("expected numeric literal")
    }

    /// `LANGTAG` — `@` followed by a BCP47-shaped tag. Returns the lower-cased tag.
    pub(crate) fn langtag(&mut self) -> R<String> {
        self.eat("@")?;
        let start = self.pos;
        let bytes = self.input.as_bytes();
        let mut i = self.pos;
        let mut count = 0;
        while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
            i += 1;
            count += 1;
        }
        if count == 0 {
            return self.mk_err("expected language tag");
        }
        loop {
            if i < bytes.len() && bytes[i] == b'-' {
                let mut j = i + 1;
                let mut sub = 0;
                while j < bytes.len() && bytes[j].is_ascii_alphanumeric() {
                    j += 1;
                    sub += 1;
                }
                if sub == 0 {
                    break;
                }
                i = j;
            } else {
                break;
            }
        }
        let tag = self.input[start..i].to_ascii_lowercase();
        self.pos = i;
        Ok(tag)
    }

    /// `String` — the four SPARQL string literal forms.
    pub(crate) fn string(&mut self) -> R<String> {
        if self.rest().starts_with("'''") {
            self.long_string("'''")
        } else if self.rest().starts_with("\"\"\"") {
            self.long_string("\"\"\"")
        } else if self.peek_char() == Some('\'') {
            self.short_string('\'')
        } else if self.peek_char() == Some('"') {
            self.short_string('"')
        } else {
            self.mk_err("expected string literal")
        }
    }

    /// `STRING_LITERAL1 | STRING_LITERAL2` — the short string forms only.
    #[cfg(feature = "sparql-12")]
    pub(crate) fn short_string_only(&mut self) -> R<String> {
        match self.peek_char() {
            Some('\'') => self.short_string('\''),
            Some('"') => self.short_string('"'),
            _ => self.mk_err("expected a short string literal"),
        }
    }

    fn short_string(&mut self, quote: char) -> R<String> {
        self.pos += quote.len_utf8();
        let start = self.pos;
        let bytes = self.input.as_bytes();
        let q = quote as u8;
        while self.pos < bytes.len() {
            match bytes[self.pos] {
                b'\n' | b'\r' => return self.mk_err("unterminated string"),
                b'\\' => {
                    self.pos += 1;
                    if self.pos < bytes.len() {
                        self.pos += 1;
                    }
                }
                c if c == q => {
                    let raw = &self.input[start..self.pos];
                    self.pos += 1;
                    return match unescape_string(raw) {
                        Ok(v) => Ok(v),
                        Err(_) => self.mk_err("invalid string escape"),
                    };
                }
                _ => self.pos += 1,
            }
        }
        self.mk_err("unterminated string")
    }

    fn long_string(&mut self, delim: &str) -> R<String> {
        self.pos += delim.len();
        let start = self.pos;
        let bytes = self.input.as_bytes();
        let db = delim.as_bytes();
        while self.pos < bytes.len() {
            if bytes[self.pos] == b'\\' {
                self.pos += 1;
                if self.pos < bytes.len() {
                    self.pos += 1;
                }
                continue;
            }
            if self.rest().as_bytes().starts_with(db) {
                let raw = &self.input[start..self.pos];
                self.pos += db.len();
                return match unescape_string(raw) {
                    Ok(v) => Ok(v),
                    Err(_) => self.mk_err("invalid string escape"),
                };
            }
            self.pos += 1;
        }
        self.mk_err("unterminated string")
    }
}

// --- escape handling --------------------------------------------------------

fn unescape_iriref(mut input: &str) -> Result<String, ()> {
    let mut output = String::with_capacity(input.len());
    while let Some((before, after)) = input.split_once('\\') {
        output.push_str(before);
        let mut chars = after.chars();
        let (escape, rest) = match chars.next() {
            Some('u') => read_hex_char::<4>(chars.as_str())?,
            Some('U') => read_hex_char::<8>(chars.as_str())?,
            _ => return Err(()),
        };
        output.push(escape);
        input = rest;
    }
    output.push_str(input);
    Ok(output)
}

fn unescape_string(mut input: &str) -> Result<String, ()> {
    let mut output = String::with_capacity(input.len());
    while let Some((before, after)) = input.split_once('\\') {
        output.push_str(before);
        let mut chars = after.chars();
        let (escape, rest) = match chars.next() {
            Some('t') => ('\u{0009}', chars.as_str()),
            Some('b') => ('\u{0008}', chars.as_str()),
            Some('n') => ('\u{000A}', chars.as_str()),
            Some('r') => ('\u{000D}', chars.as_str()),
            Some('f') => ('\u{000C}', chars.as_str()),
            Some('"') => ('\u{0022}', chars.as_str()),
            Some('\'') => ('\u{0027}', chars.as_str()),
            Some('\\') => ('\u{005C}', chars.as_str()),
            Some('u') => read_hex_char::<4>(chars.as_str())?,
            Some('U') => read_hex_char::<8>(chars.as_str())?,
            _ => return Err(()),
        };
        output.push(escape);
        input = rest;
    }
    output.push_str(input);
    Ok(output)
}

fn read_hex_char<const SIZE: usize>(input: &str) -> Result<(char, &str), ()> {
    if let Some(escape) = input.get(..SIZE) {
        if let Some(c) = u32::from_str_radix(escape, 16)
            .ok()
            .and_then(char::from_u32)
        {
            Ok((c, &input[SIZE..]))
        } else {
            Err(())
        }
    } else {
        Err(())
    }
}

// --- character classes ------------------------------------------------------

pub(crate) fn is_pn_chars_base(c: char) -> bool {
    matches!(c,
        'A'..='Z' | 'a'..='z'
        | '\u{00C0}'..='\u{00D6}' | '\u{00D8}'..='\u{00F6}' | '\u{00F8}'..='\u{02FF}'
        | '\u{0370}'..='\u{037D}' | '\u{037F}'..='\u{1FFF}' | '\u{200C}'..='\u{200D}'
        | '\u{2070}'..='\u{218F}' | '\u{2C00}'..='\u{2FEF}' | '\u{3001}'..='\u{D7FF}'
        | '\u{F900}'..='\u{FDCF}' | '\u{FDF0}'..='\u{FFFD}' | '\u{10000}'..='\u{EFFFF}')
}

fn is_pn_chars_u(c: char) -> bool {
    c == '_' || is_pn_chars_base(c)
}

pub(crate) fn is_pn_chars(c: char) -> bool {
    is_pn_chars_u(c)
        || matches!(c,
            '-' | '0'..='9' | '\u{00B7}' | '\u{0300}'..='\u{036F}' | '\u{203F}'..='\u{2040}')
}

fn is_varname_start(c: char) -> bool {
    is_pn_chars_u(c) || c.is_ascii_digit()
}

fn is_varname_rest(c: char) -> bool {
    is_pn_chars_u(c)
        || matches!(c,
            '0'..='9' | '\u{00B7}' | '\u{0300}'..='\u{036F}' | '\u{203F}'..='\u{2040}')
}

/// Scans a `VARNAME`, returning the byte length matched.
fn scan_varname(input: &str) -> usize {
    let mut chars = input.char_indices();
    let Some((_, first)) = chars.next() else {
        return 0;
    };
    if !is_varname_start(first) {
        return 0;
    }
    let mut len = first.len_utf8();
    for (i, c) in chars {
        if is_varname_rest(c) {
            len = i + c.len_utf8();
        } else {
            break;
        }
    }
    len
}

/// Scans a `PN_PREFIX`, returning the byte length matched.
fn scan_pn_prefix(input: &str) -> usize {
    let mut chars = input.char_indices();
    let Some((_, first)) = chars.next() else {
        return 0;
    };
    if !is_pn_chars_base(first) {
        return 0;
    }
    let mut last_non_dot = first.len_utf8();
    for (i, c) in chars {
        if c == '.' {
            continue;
        } else if is_pn_chars(c) {
            last_non_dot = i + c.len_utf8();
        } else {
            break;
        }
    }
    last_non_dot
}

/// Scans a `BLANK_NODE_LABEL` body (after `_:`), returning the byte length matched.
fn scan_blank_node_label(input: &str) -> usize {
    let mut chars = input.char_indices();
    let Some((_, first)) = chars.next() else {
        return 0;
    };
    if !(is_pn_chars_u(first) || first.is_ascii_digit()) {
        return 0;
    }
    let mut last_non_dot = first.len_utf8();
    for (i, c) in chars {
        if c == '.' {
            continue;
        } else if is_pn_chars(c) {
            last_non_dot = i + c.len_utf8();
        } else {
            break;
        }
    }
    last_non_dot
}

/// Scans a `PN_LOCAL`, returning the byte length matched (including `\`-escapes).
fn scan_pn_local(input: &str) -> usize {
    let bytes = input.as_bytes();
    // First character.
    let first = match first_local_char(input, 0) {
        Some(l) => l,
        None => return 0,
    };
    let mut len = first;
    let mut last_significant = first;
    loop {
        if len >= bytes.len() {
            break;
        }
        if bytes[len] == b'.' {
            // A dot is only allowed if a valid char follows.
            if first_local_char(input, len + 1).is_some() || dot_followed(input, len + 1) {
                len += 1;
                continue;
            } else {
                break;
            }
        }
        match local_char(input, len) {
            Some(next) => {
                len = next;
                last_significant = next;
            }
            None => break,
        }
    }
    last_significant
}

fn dot_followed(input: &str, at: usize) -> bool {
    local_char(input, at).is_some() || input.as_bytes().get(at) == Some(&b'.')
}

/// A `PN_LOCAL` first character: `PN_CHARS_U | ':' | [0-9] | PLX`.
fn first_local_char(input: &str, at: usize) -> Option<usize> {
    let bytes = input.as_bytes();
    if at >= bytes.len() {
        return None;
    }
    if let Some(len) = plx_len(input, at) {
        return Some(at + len);
    }
    let c = input[at..].chars().next()?;
    if is_pn_chars_u(c) || c == ':' || c.is_ascii_digit() {
        Some(at + c.len_utf8())
    } else {
        None
    }
}

/// A `PN_LOCAL` interior character: `PN_CHARS | ':' | PLX`.
fn local_char(input: &str, at: usize) -> Option<usize> {
    let bytes = input.as_bytes();
    if at >= bytes.len() {
        return None;
    }
    if let Some(len) = plx_len(input, at) {
        return Some(at + len);
    }
    let c = input[at..].chars().next()?;
    if is_pn_chars(c) || c == ':' {
        Some(at + c.len_utf8())
    } else {
        None
    }
}

/// `PLX` — `PERCENT` (`%HH`) or `PN_LOCAL_ESC` (`\` + one reserved char).
fn plx_len(input: &str, at: usize) -> Option<usize> {
    let bytes = input.as_bytes();
    match bytes.get(at) {
        Some(b'%') => {
            if is_hex(bytes.get(at + 1)) && is_hex(bytes.get(at + 2)) {
                Some(3)
            } else {
                None
            }
        }
        Some(b'\\') => match bytes.get(at + 1) {
            Some(c) if is_pn_local_esc(*c) => Some(2),
            _ => None,
        },
        _ => None,
    }
}

fn is_hex(b: Option<&u8>) -> bool {
    matches!(b, Some(c) if c.is_ascii_hexdigit())
}

fn is_pn_local_esc(c: u8) -> bool {
    matches!(
        c,
        b'_' | b'~'
            | b'.'
            | b'-'
            | b'!'
            | b'$'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b';'
            | b'='
            | b'/'
            | b'?'
            | b'#'
            | b'@'
            | b'%'
    )
}

// --- numeric scanners -------------------------------------------------------

fn scan_integer(input: &str) -> usize {
    input.bytes().take_while(u8::is_ascii_digit).count()
}

/// `DECIMAL` — `[0-9]* '.' [0-9]+`.
fn scan_decimal(input: &str) -> usize {
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'.' {
        return 0;
    }
    i += 1;
    let frac_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == frac_start { 0 } else { i }
}

/// `DOUBLE` — digits/decimal followed by an exponent.
fn scan_double(input: &str) -> usize {
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut has_digits = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        has_digits = true;
    }
    let mut has_dot = false;
    if i < bytes.len() && bytes[i] == b'.' {
        has_dot = true;
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
            has_digits = true;
        }
    }
    // Must have at least one digit somewhere before the exponent.
    if !has_digits {
        return 0;
    }
    // Exponent is mandatory for DOUBLE.
    if i >= bytes.len() || (bytes[i] != b'e' && bytes[i] != b'E') {
        return 0;
    }
    let _ = has_dot;
    i += 1;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let exp_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == exp_start { 0 } else { i }
}
