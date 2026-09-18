//! Lexical checking of attribute values against their declared types.
//!
//! Implements the per-value half of the validity constraints of XML 1.0 §3.3.1
//! — `ID`, `IDREF`, `IDREFS`, `ENTITY`, `ENTITIES`, `NMTOKEN`, `NMTOKENS`,
//! `NOTATION` and enumerations. The *cross-document* half (every `ID` unique,
//! every `IDREF` matched) lives in [`Validator`](super::Validator), which is the
//! only place that can see the whole document.

use oxixml_qname::{is_name, is_nmtoken};

use crate::model::{AttributeType, Dtd};

/// Checks `value` — already normalized per XML 1.0 §3.3.3 — against `att_type`.
///
/// Returns `None` when the value is acceptable, or a human-readable reason why
/// it is not.
///
/// # Examples
///
/// ```
/// use oxixml_dtd::{AttributeType, Dtd};
/// use oxixml_dtd::validate::lexical_error;
///
/// let dtd = Dtd::new();
/// assert_eq!(lexical_error(&dtd, &AttributeType::Id, "chapter1"), None);
/// assert!(lexical_error(&dtd, &AttributeType::Id, "1chapter").is_some());
/// assert_eq!(lexical_error(&dtd, &AttributeType::NmToken, "1chapter"), None);
/// ```
#[must_use]
pub fn lexical_error(dtd: &Dtd, att_type: &AttributeType, value: &str) -> Option<String> {
    check(Some(dtd), att_type, value)
}

/// Checks `value` against `att_type` *syntactically only*.
///
/// This is the weaker test that XML 1.0 §3.3.2 (as corrected by erratum E9a)
/// applies to a declared **default** value: it must meet the lexical
/// constraints of its type, but an `ENTITY` default need not name an entity
/// that is actually declared, because a default that is never used is never
/// checked against the stronger VC: Entity Name.
///
/// # Examples
///
/// ```
/// use oxixml_dtd::AttributeType;
/// use oxixml_dtd::validate::syntactic_error;
///
/// // No entity named `logo` is declared anywhere, and that is fine here.
/// assert_eq!(syntactic_error(&AttributeType::Entity, "logo"), None);
/// assert!(syntactic_error(&AttributeType::Entity, "1logo").is_some());
/// ```
#[must_use]
pub fn syntactic_error(att_type: &AttributeType, value: &str) -> Option<String> {
    check(None, att_type, value)
}

/// Shared implementation: `dtd` is `None` for the syntax-only check.
fn check(dtd: Option<&Dtd>, att_type: &AttributeType, value: &str) -> Option<String> {
    match att_type {
        AttributeType::Cdata => None,
        AttributeType::Id | AttributeType::IdRef => {
            (!is_name(value)).then(|| format!("`{value}` is not a Name"))
        }
        AttributeType::IdRefs => check_list(value, |token| {
            is_name(token)
                .then_some(())
                .ok_or_else(|| format!("`{token}` is not a Name"))
        }),
        AttributeType::NmToken => {
            (!is_nmtoken(value)).then(|| format!("`{value}` is not an Nmtoken"))
        }
        AttributeType::NmTokens => check_list(value, |token| {
            is_nmtoken(token)
                .then_some(())
                .ok_or_else(|| format!("`{token}` is not an Nmtoken"))
        }),
        AttributeType::Entity => check_unparsed_entity(dtd, value),
        AttributeType::Entities => {
            check_list(value, |token| match check_unparsed_entity(dtd, token) {
                Some(reason) => Err(reason),
                None => Ok(()),
            })
        }
        AttributeType::Notation(tokens) | AttributeType::Enumeration(tokens) => {
            if tokens.iter().any(|token| token == value) {
                None
            } else {
                Some(format!("`{value}` is not one of {}", tokens.join(", ")))
            }
        }
    }
}

/// Applies `check` to every whitespace-separated token, requiring at least one.
fn check_list(value: &str, check: impl Fn(&str) -> Result<(), String>) -> Option<String> {
    let mut seen = false;
    for token in value.split(' ').filter(|token| !token.is_empty()) {
        seen = true;
        if let Err(reason) = check(token) {
            return Some(reason);
        }
    }
    (!seen).then(|| "the value must contain at least one token".to_owned())
}

/// VC: Entity Name (XML 1.0 §3.3.1) — the value must name a declared *unparsed*
/// entity. With no `dtd`, only the `Name` production is checked.
fn check_unparsed_entity(dtd: Option<&Dtd>, value: &str) -> Option<String> {
    if !is_name(value) {
        return Some(format!("`{value}` is not a Name"));
    }
    let dtd = dtd?;
    match dtd.general_entity(value) {
        Some(entity) if entity.is_unparsed() => None,
        Some(_) => Some(format!("`{value}` is a parsed entity, not an unparsed one")),
        None => Some(format!("`{value}` is not a declared unparsed entity")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{EntityDecl, EntityDefinition, EntityKind, ExternalId};

    fn dtd_with_unparsed_entity() -> Dtd {
        let mut dtd = Dtd::new();
        dtd.add_entity(EntityDecl::new(
            "logo",
            EntityKind::General,
            EntityDefinition::External {
                id: ExternalId::system("logo.gif"),
                notation: Some("gif".to_owned()),
            },
            false,
        ));
        dtd.add_entity(EntityDecl::new(
            "text",
            EntityKind::General,
            EntityDefinition::Internal("hello".to_owned()),
            false,
        ));
        dtd
    }

    #[test]
    fn cdata_accepts_anything() {
        assert_eq!(
            lexical_error(&Dtd::new(), &AttributeType::Cdata, "< > & \" '"),
            None
        );
    }

    #[test]
    fn id_requires_a_name() {
        let dtd = Dtd::new();
        assert_eq!(lexical_error(&dtd, &AttributeType::Id, "a1"), None);
        assert!(lexical_error(&dtd, &AttributeType::Id, "1a").is_some());
        assert!(lexical_error(&dtd, &AttributeType::Id, "a b").is_some());
    }

    #[test]
    fn idrefs_requires_at_least_one_name() {
        let dtd = Dtd::new();
        assert_eq!(lexical_error(&dtd, &AttributeType::IdRefs, "a b c"), None);
        assert!(lexical_error(&dtd, &AttributeType::IdRefs, "").is_some());
        assert!(lexical_error(&dtd, &AttributeType::IdRefs, "a 1b").is_some());
    }

    #[test]
    fn entity_values_must_name_unparsed_entities() {
        let dtd = dtd_with_unparsed_entity();
        assert_eq!(lexical_error(&dtd, &AttributeType::Entity, "logo"), None);
        assert!(lexical_error(&dtd, &AttributeType::Entity, "text").is_some());
        assert!(lexical_error(&dtd, &AttributeType::Entity, "missing").is_some());
        assert_eq!(
            lexical_error(&dtd, &AttributeType::Entities, "logo logo"),
            None
        );
    }

    #[test]
    fn enumerations_check_membership() {
        let dtd = Dtd::new();
        let att_type = AttributeType::Enumeration(vec!["a".to_owned(), "b".to_owned()]);
        assert_eq!(lexical_error(&dtd, &att_type, "a"), None);
        assert!(lexical_error(&dtd, &att_type, "c").is_some());
    }

    #[test]
    fn nmtokens_accept_leading_digits() {
        let dtd = Dtd::new();
        assert_eq!(lexical_error(&dtd, &AttributeType::NmTokens, "1 2 3"), None);
        assert!(lexical_error(&dtd, &AttributeType::NmTokens, "a$b").is_some());
    }
}
