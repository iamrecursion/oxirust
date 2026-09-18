//! `SequenceType`, `ItemType` and `SingleType` — the type syntax of `instance of`, `treat as`,
//! `cast as`, `castable as` and an inline function's signature.

use super::Parser;
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::lexer::Token;
use crate::types::{ItemType, ListType, Occurrence, SequenceType, SingleType};
use crate::version::XPathVersion;
use oxixml_infoset::{AtomicType, ExpandedName, XSD_NAMESPACE};

/// What `cast as` and `castable as` were given as their target type.
pub(crate) enum CastTarget {
    /// An ordinary `SingleType`.
    Single(SingleType),
    /// `xs:numeric`, with `true` when it carried a `?`.
    Numeric(bool),
    /// One of the three built-in list types.
    List(ListType),
}

/// The item type of the list type `local` names, for the three XML Schema declares.
const fn list_member(local: &str) -> Option<AtomicType> {
    match local.as_bytes() {
        b"IDREFS" => Some(AtomicType::Idref),
        b"NMTOKENS" => Some(AtomicType::Nmtoken),
        b"ENTITIES" => Some(AtomicType::Entity),
        _ => None,
    }
}

impl Parser<'_> {
    /// `SequenceType ::= ("empty-sequence" "(" ")") | (ItemType OccurrenceIndicator?)`
    pub(crate) fn parse_sequence_type(&mut self) -> XPathResult<SequenceType> {
        if self.at_word("empty-sequence") && matches!(self.peek_at(1), Some(Token::LeftParen)) {
            self.advance();
            self.expect(&Token::LeftParen)?;
            self.expect(&Token::RightParen)?;
            return Ok(SequenceType::Empty);
        }
        let item_type = self.parse_item_type()?;
        let occurrence = match self.peek() {
            Some(Token::Question) => {
                self.advance();
                Occurrence::ZeroOrOne
            }
            Some(Token::Star) => {
                self.advance();
                Occurrence::ZeroOrMore
            }
            Some(Token::Plus) => {
                self.advance();
                Occurrence::OneOrMore
            }
            _ => Occurrence::One,
        };
        Ok(SequenceType::new(item_type, occurrence))
    }

    /// `ItemType ::= KindTest | ("item" "(" ")") | FunctionTest | MapTest | ArrayTest
    ///             | AtomicOrUnionType | ParenthesizedItemType`
    pub(crate) fn parse_item_type(&mut self) -> XPathResult<ItemType> {
        // A parenthesized item type, a function test and a map/array test all recurse back
        // into this production, so it needs the same ceiling the expression grammar has.
        self.enter_nesting()?;
        let parsed = self.parse_item_type_inner();
        self.leave_nesting();
        parsed
    }

    fn parse_item_type_inner(&mut self) -> XPathResult<ItemType> {
        if self.eat(&Token::LeftParen) {
            let inner = self.parse_item_type()?;
            self.expect(&Token::RightParen)?;
            return Ok(inner);
        }
        if let Some(Token::Name(name)) = self.peek().cloned() {
            let followed_by_paren = matches!(self.peek_at(1), Some(Token::LeftParen));
            if followed_by_paren {
                match name.as_str() {
                    "item" => {
                        self.advance();
                        self.expect(&Token::LeftParen)?;
                        self.expect(&Token::RightParen)?;
                        return Ok(ItemType::Item);
                    }
                    "function" => return self.parse_function_test(),
                    "map" => return self.parse_map_test(),
                    "array" => return self.parse_array_test(),
                    "attribute"
                    | "comment"
                    | "document-node"
                    | "element"
                    | "namespace-node"
                    | "node"
                    | "processing-instruction"
                    | "schema-attribute"
                    | "schema-element"
                    | "text" => {
                        return Ok(ItemType::Kind(self.parse_kind_test()?));
                    }
                    // A name followed by `(` in item-type position must be one of the kind tests
                    // or the parametric `item`/`function`/`map`/`array` tests handled above. An
                    // atomic or union type is a bare name, never a call, so `document(…)` and the
                    // like are a syntax error (`XPST0003`), not an unknown-type error.
                    other => {
                        let offset = self.offset();
                        return Err(XPathError::new(
                            XPathErrorCode::Xpst0003,
                            format!("`{other}(` is not an item type"),
                        )
                        .at(offset));
                    }
                }
            }
        }
        let offset = self.offset();
        let name = self.parse_element_name()?;
        if name.namespace() == Some(XSD_NAMESPACE) && name.local_name() == "numeric" {
            return Ok(ItemType::Numeric);
        }
        Ok(ItemType::Atomic(self.atomic_type(&name, offset)?))
    }

    /// `SingleType ::= SimpleTypeName "?"?`, where the name may also be `xs:numeric`.
    ///
    /// `xs:numeric` is the union of `xs:double`, `xs:float` and `xs:decimal` that XPath 3.1
    /// §2.5.4 defines. It is not an [`AtomicType`], so it cannot be carried by a [`SingleType`];
    /// the caller gets [`CastTarget::Numeric`] instead and rewrites the expression into the
    /// union's casting rule (XPath 3.1 §19.3).
    pub(crate) fn parse_cast_target(&mut self) -> XPathResult<CastTarget> {
        if self.peek_is_numeric_type() {
            self.parse_element_name()?;
            return Ok(CastTarget::Numeric(self.eat(&Token::Question)));
        }
        if let Some(member) = self.peek_list_member() {
            self.parse_element_name()?;
            return Ok(CastTarget::List(ListType::new(
                member,
                self.eat(&Token::Question),
            )));
        }
        Ok(CastTarget::Single(self.parse_single_type()?))
    }

    /// The item type of the built-in list type the next token names, if it names one.
    ///
    /// `xs:IDREFS`, `xs:NMTOKENS` and `xs:ENTITIES` are list types rather than atomic ones, so a
    /// [`SingleType`] cannot carry them and [`Self::parse_single_type`] would report `XPST0051`.
    /// They are none the less in scope in every processor — XML Schema declares them — and
    /// [XPath 3.1 §19.3.4](https://www.w3.org/TR/xpath-31/#casting-to-list-types) defines a cast
    /// to them.
    fn peek_list_member(&self) -> Option<AtomicType> {
        let named = match self.peek()? {
            Token::BracedName { uri, local } => {
                return (uri == XSD_NAMESPACE).then(|| list_member(local)).flatten();
            }
            Token::Name(lexical) => lexical,
            _ => return None,
        };
        let name = self.context.resolve_element_name(named).ok()?;
        (name.namespace() == Some(XSD_NAMESPACE))
            .then(|| list_member(name.local_name()))
            .flatten()
    }

    /// Whether the next token spells `xs:numeric`, in either the prefixed or the braced form.
    fn peek_is_numeric_type(&self) -> bool {
        let named = match self.peek() {
            Some(Token::BracedName { uri, local }) => {
                return uri == XSD_NAMESPACE && local == "numeric";
            }
            Some(Token::Name(lexical)) => lexical,
            _ => return false,
        };
        self.context.resolve_element_name(named).is_ok_and(|name| {
            name.namespace() == Some(XSD_NAMESPACE) && name.local_name() == "numeric"
        })
    }

    /// `SingleType ::= SimpleTypeName "?"?`
    pub(crate) fn parse_single_type(&mut self) -> XPathResult<SingleType> {
        let offset = self.offset();
        let name = self.parse_element_name()?;
        // The two host languages name the same mistake differently. XPath 3.1 reports an unknown
        // cast target as `XPST0051`, the code it gives an unknown `AtomicOrUnionType`; XQuery 3.0
        // gave the cast its own, `XQST0052`, and the suite pins both readings on the same query
        // (`K-SeqExprCast-9` under `XQ10`, `K-SeqExprCast-9a` under `XQ30+`).
        let atomic_type = self.atomic_type(&name, offset).map_err(|error| {
            if self.context.is_xquery_host_language()
                && error.code() == Some(XPathErrorCode::Xpst0051)
            {
                return XPathError::new(XPathErrorCode::Xqst0052, error.message().to_owned())
                    .at(offset);
            }
            error
        })?;
        if matches!(
            atomic_type,
            AtomicType::AnyAtomicType | AtomicType::Notation
        ) {
            return Err(XPathError::new(
                XPathErrorCode::Xpst0080,
                format!(
                    "`xs:{}` is not a legal cast target",
                    atomic_type.local_name()
                ),
            )
            .at(offset));
        }
        let optional = self.eat(&Token::Question);
        Ok(SingleType::new(atomic_type, optional))
    }

    fn parse_function_test(&mut self) -> XPathResult<ItemType> {
        self.require_version(XPathVersion::XPath30, "a function test")?;
        self.advance();
        self.expect(&Token::LeftParen)?;
        if self.eat(&Token::Star) {
            self.expect(&Token::RightParen)?;
            return Ok(ItemType::AnyFunction);
        }
        let mut parameters = Vec::new();
        if !self.eat(&Token::RightParen) {
            loop {
                parameters.push(self.parse_sequence_type()?);
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
            self.expect(&Token::RightParen)?;
        }
        self.expect_word("as")?;
        let result = Box::new(self.parse_sequence_type()?);
        Ok(ItemType::TypedFunction { parameters, result })
    }

    fn parse_map_test(&mut self) -> XPathResult<ItemType> {
        self.require_version(XPathVersion::XPath31, "a map test")?;
        self.advance();
        self.expect(&Token::LeftParen)?;
        if self.eat(&Token::Star) {
            self.expect(&Token::RightParen)?;
            return Ok(ItemType::AnyMap);
        }
        let offset = self.offset();
        let name = self.parse_element_name()?;
        let key = self.atomic_type(&name, offset)?;
        self.expect(&Token::Comma)?;
        let value = Box::new(self.parse_sequence_type()?);
        self.expect(&Token::RightParen)?;
        Ok(ItemType::TypedMap { key, value })
    }

    fn parse_array_test(&mut self) -> XPathResult<ItemType> {
        self.require_version(XPathVersion::XPath31, "an array test")?;
        self.advance();
        self.expect(&Token::LeftParen)?;
        if self.eat(&Token::Star) {
            self.expect(&Token::RightParen)?;
            return Ok(ItemType::AnyArray);
        }
        let member = Box::new(self.parse_sequence_type()?);
        self.expect(&Token::RightParen)?;
        Ok(ItemType::TypedArray(member))
    }

    /// Maps an expanded name onto a built-in atomic type.
    ///
    /// This engine has no schema, so the only named types are the built-ins of the XML Schema
    /// namespace; anything else is `XPST0051`.
    fn atomic_type(&self, name: &ExpandedName, offset: usize) -> XPathResult<AtomicType> {
        if name.namespace() == Some(XSD_NAMESPACE) {
            if let Some(atomic_type) = AtomicType::from_local_name(name.local_name()) {
                return Ok(atomic_type);
            }
        }
        Err(XPathError::new(
            XPathErrorCode::Xpst0051,
            format!("`{name}` is not a known atomic type"),
        )
        .at(offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr;
    use crate::context::StaticContext;
    use crate::parser::parse;
    use crate::types::KindTest;

    fn sequence_type(source: &str) -> XPathResult<SequenceType> {
        let expression = parse(
            &format!("() instance of {source}"),
            XPathVersion::XPath31,
            &StaticContext::new(),
        )?;
        match &expression {
            Expr::InstanceOf(_, required) => Ok(required.clone()),
            other => panic!("expected an instance-of expression, found {other:?}"),
        }
    }

    /// XPath 3.1 §2.5.4/§19.3: `xs:numeric` is a union type, so a cast to it is rewritten into
    /// "keep it if it already is one of the members, otherwise cast to `xs:double`" rather than
    /// carried by a `SingleType`, which can only hold an atomic type.
    #[test]
    fn a_cast_to_xs_numeric_is_rewritten_into_the_union_rule() -> XPathResult<()> {
        let context = StaticContext::new();
        assert!(matches!(
            parse("17 cast as xs:numeric", XPathVersion::XPath31, &context)?,
            Expr::If { .. }
        ));
        assert!(matches!(
            parse(
                "'1' castable as xs:numeric",
                XPathVersion::XPath31,
                &context
            )?,
            Expr::Or(_, _)
        ));
        // An ordinary target is untouched.
        assert!(matches!(
            parse("'1' cast as xs:integer", XPathVersion::XPath31, &context)?,
            Expr::Cast(_, _)
        ));
        Ok(())
    }

    #[test]
    fn occurrence_indicators_parse() -> XPathResult<()> {
        assert_eq!(sequence_type("empty-sequence()")?, SequenceType::Empty);
        assert_eq!(
            sequence_type("item()")?,
            SequenceType::new(ItemType::Item, Occurrence::One)
        );
        assert_eq!(
            sequence_type("item()?")?,
            SequenceType::new(ItemType::Item, Occurrence::ZeroOrOne)
        );
        assert_eq!(
            sequence_type("item()*")?,
            SequenceType::new(ItemType::Item, Occurrence::ZeroOrMore)
        );
        assert_eq!(
            sequence_type("item()+")?,
            SequenceType::new(ItemType::Item, Occurrence::OneOrMore)
        );
        Ok(())
    }

    #[test]
    fn item_types_cover_every_form() -> XPathResult<()> {
        assert_eq!(
            sequence_type("xs:integer")?,
            SequenceType::new(ItemType::Atomic(AtomicType::Integer), Occurrence::One)
        );
        assert_eq!(
            sequence_type("element(a)")?,
            SequenceType::new(
                ItemType::Kind(KindTest::Element {
                    name: Some(ExpandedName::new(None::<String>, "a")),
                    type_name: None,
                    nillable: false
                }),
                Occurrence::One
            )
        );
        assert_eq!(
            sequence_type("function(*)")?,
            SequenceType::new(ItemType::AnyFunction, Occurrence::One)
        );
        assert!(matches!(
            sequence_type("function(xs:integer) as xs:integer")?,
            SequenceType::Items {
                item_type: ItemType::TypedFunction { .. },
                ..
            }
        ));
        assert_eq!(
            sequence_type("map(*)")?,
            SequenceType::new(ItemType::AnyMap, Occurrence::One)
        );
        assert!(matches!(
            sequence_type("map(xs:string, xs:integer*)")?,
            SequenceType::Items {
                item_type: ItemType::TypedMap { .. },
                ..
            }
        ));
        assert_eq!(
            sequence_type("array(*)")?,
            SequenceType::new(ItemType::AnyArray, Occurrence::One)
        );
        assert!(matches!(
            sequence_type("array(xs:integer)")?,
            SequenceType::Items {
                item_type: ItemType::TypedArray(_),
                ..
            }
        ));
        assert_eq!(
            sequence_type("(xs:integer)")?,
            SequenceType::new(ItemType::Atomic(AtomicType::Integer), Occurrence::One)
        );
        Ok(())
    }

    #[test]
    fn unknown_type_names_are_xpst0051() {
        let failure = sequence_type("xs:nope").expect_err("no such type");
        assert_eq!(failure.code(), Some(XPathErrorCode::Xpst0051));
    }

    #[test]
    fn illegal_cast_targets_are_xpst0080() {
        let context = StaticContext::new();
        for source in ["1 cast as xs:anyAtomicType", "1 cast as xs:NOTATION"] {
            let failure = parse(source, XPathVersion::XPath31, &context).expect_err("illegal");
            assert_eq!(failure.code(), Some(XPathErrorCode::Xpst0080));
        }
    }

    #[test]
    fn a_name_with_parentheses_that_is_not_a_kind_test_is_xpst0003() {
        // `document(…)` is not a type: a name followed by `(` in item-type position must be a
        // kind test or a parametric `item`/`function`/`map`/`array` test, so this is a syntax
        // error rather than an unknown-type error (QT3 `K2-NodeTest-12`, `K2-NodeTest-13`).
        for source in ["document()", "document(*)", "notatype()"] {
            let failure = sequence_type(source).expect_err("not an item type");
            assert_eq!(
                failure.code(),
                Some(XPathErrorCode::Xpst0003),
                "for `{source}`"
            );
        }
    }

    #[test]
    fn document_node_accepts_only_an_element_test() {
        // `schema-attribute()` inside `document-node()` is a syntax error, and it must be caught
        // as such before `schema-attribute`'s own `XPST0008` can pre-empt it (QT3
        // `K2-NodeTest-18`).
        let context = StaticContext::new();
        let failure = parse(
            "1 instance of document-node(schema-attribute(ncname))",
            XPathVersion::XPath31,
            &context,
        )
        .expect_err("schema-attribute is not a valid document-node() inner test");
        assert_eq!(failure.code(), Some(XPathErrorCode::Xpst0003));
    }
}
