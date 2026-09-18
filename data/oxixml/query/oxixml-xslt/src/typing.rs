//! Required types: the `as` attribute of `xsl:variable`, `xsl:param`, `xsl:template`,
//! `xsl:function` and friends, checked at run time.
//!
//! [XSLT 3.0 §5.4.2](https://www.w3.org/TR/xslt-30/#sequence-type-matching) says the value a
//! declaration produces is *converted to the required type using the function conversion
//! rules*, and that a failure is a type error whose code depends on where the `as` was
//! written — `XTTE0570` for a variable, `XTTE0590` for a parameter, `XTTE0505` for the body of
//! a template, `XTTE0780` for the body of a function.
//!
//! The `SequenceType` grammar belongs to XPath, so rather than re-implement it this module
//! borrows the one [`oxixml_xpath`] already parses: the type text is compiled inside an
//! `instance of` expression and lifted straight out of the resulting syntax tree.

use crate::compile::StaticEnv;
use crate::error::{XsltError, XsltResult};
use oxixml_infoset::{AtomicType, AtomicValue};
use oxixml_xpath::ast::Expr;
use oxixml_xpath::{ItemType, Occurrence, SequenceType, XpathValue};

/// A compiled `as` attribute.
#[derive(Debug)]
pub(crate) struct RequiredType {
    /// The text as it was written, for diagnostics.
    text: String,
    /// The parsed type, or `None` when the text is a well-formed `SequenceType` naming something
    /// this processor does not know — an unrecognized *name* is left unchecked rather than
    /// turned into a static error, which is what forwards-compatible processing asks for. Text
    /// that is not a `SequenceType` at all does not reach here: it is rejected at compile time.
    matcher: Option<SequenceType>,
}

impl RequiredType {
    /// Compiles the `as` attribute written on `element`, in the namespaces of `env`.
    ///
    /// # Errors
    ///
    /// `XPST0003` when the text is not a `SequenceType` at all.
    pub(crate) fn compile(env: &StaticEnv, text: &str) -> XsltResult<Self> {
        Ok(Self {
            text: text.to_owned(),
            matcher: parse(env, text)?,
        })
    }

    /// The type as it was written.
    #[cfg(test)]
    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// Whether the type was understood well enough to be enforced.
    pub(crate) const fn is_checked(&self) -> bool {
        self.matcher.is_some()
    }

    /// The parsed type, for a caller that has to *publish* it rather than check against it —
    /// the signature an `xsl:function` contributes to the static context.
    pub(crate) const fn sequence_type(&self) -> Option<&SequenceType> {
        self.matcher.as_ref()
    }

    /// The type `item()*`, which every `as`-less declaration has.
    pub(crate) const fn any_sequence() -> SequenceType {
        SequenceType::Items {
            item_type: ItemType::Item,
            occurrence: Occurrence::ZeroOrMore,
        }
    }

    /// Whether `value` matches this type *as it stands*, with no conversion at all.
    ///
    /// [XSLT 3.0 §10.1.1](https://www.w3.org/TR/xslt-30/#context-item) needs exactly this for
    /// `xsl:context-item`: "No attempt is made to convert the context item to the required type
    /// (using the function conversion rules or otherwise)." A type this processor could not
    /// parse is left unenforced, as in [`RequiredType::convert`].
    pub(crate) fn matches(&self, value: &XpathValue) -> bool {
        self.matcher
            .as_ref()
            .is_none_or(|matcher| matcher.matches(value))
    }

    /// Whether the empty sequence is a valid instance of this type.
    ///
    /// [XSLT 3.0 §9.3](https://www.w3.org/TR/xslt-30/#parameters) turns on this question: a
    /// parameter whose implicit default (the empty sequence) "cannot be converted to the
    /// required type … is implicitly mandatory".
    pub(crate) fn permits_empty(&self) -> bool {
        self.matches(&XpathValue::from(oxixml_infoset::Sequence::empty()))
    }

    /// The occurrence indicator the type was written with, or `None` when the type could not be
    /// parsed. `xsl:context-item/@as` is an `ItemType`, not a `SequenceType`, so anything other
    /// than [`Occurrence::One`] there is a static error.
    pub(crate) fn occurrence(&self) -> Option<Occurrence> {
        match self.matcher.as_ref()? {
            SequenceType::Empty => None,
            SequenceType::Items { occurrence, .. } => Some(*occurrence),
        }
    }

    /// Applies the function conversion rules and checks the outcome, raising `code` on
    /// failure.
    ///
    /// # Errors
    ///
    /// `code` when the value cannot be converted to this type.
    pub(crate) fn convert(
        &self,
        value: XpathValue,
        code: &str,
        what: &str,
    ) -> XsltResult<XpathValue> {
        let Some(matcher) = &self.matcher else {
            return Ok(value);
        };
        let converted = match matcher {
            SequenceType::Items {
                item_type: ItemType::Atomic(target),
                ..
            } => atomize_and_promote(&value, Some(*target))?,
            SequenceType::Items {
                item_type: ItemType::Numeric,
                ..
            } => atomize_and_promote(&value, None)?,
            _ => value,
        };
        if matcher.matches(&converted) {
            return Ok(converted);
        }
        Err(XsltError::dynamic(
            code,
            format!(
                "{what} does not match the required type `{}` (the value holds {} item(s))",
                self.text,
                converted.len()
            ),
        ))
    }
}

/// Atomizes `value` and applies the numeric/URI promotions the function conversion rules
/// allow. `target` is `None` for `xs:numeric`, which promotes nothing but untyped atomics.
fn atomize_and_promote(value: &XpathValue, target: Option<AtomicType>) -> XsltResult<XpathValue> {
    let mut out = Vec::new();
    for atomic in value.atomize()? {
        out.push(promote(atomic, target)?);
    }
    Ok(XpathValue::from_items(out.into_iter().map(Into::into)))
}

fn promote(value: AtomicValue, target: Option<AtomicType>) -> XsltResult<AtomicValue> {
    let Some(target) = target else {
        // `xs:numeric` in a sequence type is the union of the three numeric types; an untyped
        // atomic is cast to `xs:double`, and nothing else is touched.
        return Ok(if value.atomic_type() == AtomicType::UntypedAtomic {
            cast(&value, AtomicType::Double)?
        } else {
            value
        });
    };
    if value.atomic_type().is_derived_from(target) {
        return Ok(value);
    }
    let promotable = value.atomic_type() == AtomicType::UntypedAtomic
        || (target == AtomicType::Double && value.is_numeric())
        || (target == AtomicType::Float
            && value.atomic_type().is_derived_from(AtomicType::Decimal))
        || (target == AtomicType::String && value.atomic_type() == AtomicType::AnyUri);
    if !promotable {
        return Ok(value);
    }
    // A failed cast leaves the value alone: the type check that follows turns it into the
    // caller's error code, which is what the spec asks for.
    Ok(cast(&value, target).unwrap_or(value))
}

fn cast(value: &AtomicValue, target: AtomicType) -> XsltResult<AtomicValue> {
    AtomicValue::parse(target, &value.string_value()).map_err(|error| {
        XsltError::dynamic(
            "XPTY0004",
            format!(
                "`{}` is not a valid `xs:{}`: {}",
                value.string_value(),
                target.local_name(),
                error
            ),
        )
    })
}

/// Lifts a `SequenceType` out of the XPath parser by compiling `. instance of TYPE`.
///
/// Text the parser rejects — `as="xs:integer((("`, `as="as as as"`, a type naming a prefix
/// nothing binds — is not a `SequenceType`, and [XSLT 3.0
/// §5.4.1](https://www.w3.org/TR/xslt-30/#sequence-types) makes that a static error at the
/// declaration: leaving it unchecked would silently discard exactly the check the author asked
/// for. The one place it stays soft is forwards-compatible processing, where a module states a
/// later version than this processor implements and §3.9 requires a construct it has never heard
/// of to be tolerated rather than rejected.
fn parse(env: &StaticEnv, text: &str) -> XsltResult<Option<SequenceType>> {
    let trimmed = text.trim();
    if trimmed.is_empty() && env.forwards_compatible() {
        return Ok(None);
    }
    let compiled = match env.compile_expression(&format!(". instance of {trimmed}")) {
        Ok(compiled) => compiled,
        Err(_) if env.forwards_compatible() => return Ok(None),
        Err(error) => {
            return Err(XsltError::stat1c(
                error.code().unwrap_or("XPST0003"),
                format!("`{trimmed}` is not a sequence type"),
            ));
        }
    };
    Ok(match compiled.syntax_tree() {
        Expr::InstanceOf(_, sequence_type) => Some(sequence_type.clone()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::XsltVersion;
    use oxixml_infoset::Sequence;

    fn env() -> StaticEnv {
        let mut env = StaticEnv::new(XsltVersion::V30);
        env.declare_namespaces([(Some("xs"), "http://www.w3.org/2001/XMLSchema")]);
        env
    }

    #[test]
    fn a_sequence_type_is_lifted_out_of_the_parser() -> XsltResult<()> {
        let required = RequiredType::compile(&env(), "xs:integer")?;
        assert!(required.is_checked());
        assert_eq!(required.text(), "xs:integer");
        assert!(RequiredType::compile(&env(), "element()*")?.is_checked());
        Ok(())
    }

    #[test]
    fn a_mismatch_carries_the_caller_s_code() -> XsltResult<()> {
        let required = RequiredType::compile(&env(), "xs:integer")?;
        let value = XpathValue::from(AtomicValue::boolean(true));
        let error = required
            .convert(value, "XTTE0570", "the variable `v`")
            .expect_err("a boolean is not an integer");
        assert!(error.is("XTTE0570"));
        Ok(())
    }

    #[test]
    fn untyped_atomics_are_cast_and_numbers_promoted() -> XsltResult<()> {
        let required = RequiredType::compile(&env(), "xs:double")?;
        let value = XpathValue::from(AtomicValue::untyped_atomic("2.5"));
        let converted = required.convert(value, "XTTE0570", "v")?;
        assert_eq!(converted.len(), 1);

        let integers = RequiredType::compile(&env(), "xs:double*")?;
        let value = XpathValue::from_items([AtomicValue::integer(3).into()]);
        assert_eq!(integers.convert(value, "XTTE0570", "v")?.len(), 1);
        Ok(())
    }

    #[test]
    fn cardinality_is_enforced() -> XsltResult<()> {
        let required = RequiredType::compile(&env(), "xs:integer")?;
        let value = XpathValue::from(Sequence::empty());
        assert!(required.convert(value, "XTTE0590", "p").is_err());
        Ok(())
    }

    /// Rejecting text that is not a sequence type must not reject the ordinary type surface
    /// along with it: every one of these is a type an `as` attribute legitimately carries.
    /// (`schema-element(…)` and `schema-attribute(…)` are a separate matter — they name a
    /// declaration in a schema, which a processor without schema awareness has none of, and
    /// `XPST0008` for them is what the specification asks for.)
    #[test]
    fn the_ordinary_item_types_are_all_accepted() {
        for text in [
            "item()",
            "empty-sequence()",
            "xs:integer",
            "xs:integer*",
            "xs:integer?",
            "xs:integer+",
            "xs:dayTimeDuration",
            "xs:anyAtomicType",
            "node()",
            "element()",
            "element(a)",
            "element(*)",
            "attribute()",
            "attribute(a)",
            "document-node()",
            "document-node(element(a))",
            "text()",
            "comment()",
            "processing-instruction()",
            "namespace-node()",
            "map(*)",
            "map(xs:string, item())",
            "array(*)",
            "array(item())",
            "function(*)",
            "function(item()) as item()",
            "xs:numeric",
            "xs:untypedAtomic",
            "Q{http://www.w3.org/2001/XMLSchema}integer",
            "element(a, xs:string)",
            "element(*, xs:untyped)",
            "attribute(*, xs:string)",
        ] {
            let compiled = RequiredType::compile(&env(), text);
            assert!(
                compiled.is_ok(),
                "`{text}` was rejected: {:?}",
                compiled.err()
            );
        }
    }

    #[test]
    fn text_that_is_not_a_sequence_type_is_a_static_error() {
        for text in [
            "not a type at all",
            "xs:integer(((",
            "as as as",
            "xs:nonesuch",
            "",
        ] {
            let error = RequiredType::compile(&env(), text)
                .expect_err("the text does not hold a sequence type");
            assert_eq!(error.phase(), crate::error::ErrorPhase::Static);
        }
    }

    #[test]
    fn a_type_is_left_unchecked_in_forwards_compatible_mode() -> XsltResult<()> {
        // A module written for a later version than this one may name a type this processor
        // has never heard of, and §3.9 requires that to be tolerated rather than rejected.
        let mut env = env();
        env.set_version_text("4.0");
        let required = RequiredType::compile(&env, "xs:nonesuch")?;
        assert!(!required.is_checked());
        let value = XpathValue::from(AtomicValue::boolean(true));
        assert_eq!(required.convert(value, "XTTE0570", "v")?.len(), 1);
        Ok(())
    }
}
