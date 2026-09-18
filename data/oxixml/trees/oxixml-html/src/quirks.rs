//! Quirks-mode detection from the DOCTYPE token.
//!
//! The rules are those of the
//! ["initial" insertion mode](https://html.spec.whatwg.org/multipage/parsing.html#the-initial-insertion-mode),
//! which is the only place the standard ever sets the document's mode. The
//! long lists of public identifiers below are reproduced from that section
//! verbatim; they name the DTDs of pre-standard browsers, whose documents must
//! keep being laid out the way those browsers laid them out.

use crate::token::Doctype;

/// The three CSS compatibility modes a document can be in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QuirksMode {
    /// Standards mode: what a modern `<!DOCTYPE html>` selects.
    #[default]
    NoQuirks,
    /// "Almost standards" mode, selected by the transitional and frameset
    /// DTDs of HTML 4.01 and XHTML 1.0.
    LimitedQuirks,
    /// Quirks mode, selected by a missing, malformed or pre-HTML-4 DOCTYPE.
    Quirks,
}

impl QuirksMode {
    /// Returns `true` for [`QuirksMode::Quirks`].
    #[must_use]
    pub const fn is_quirks(self) -> bool {
        matches!(self, Self::Quirks)
    }
}

/// Public identifiers that select quirks mode outright.
const QUIRKY_PUBLIC_IDS: &[&str] = &[
    "-//W3O//DTD W3 HTML Strict 3.0//EN//",
    "-/W3C/DTD HTML 4.0 Transitional/EN",
    "HTML",
];

/// The one system identifier that selects quirks mode outright.
const QUIRKY_SYSTEM_ID: &str = "http://www.ibm.com/data/dtd/v11/ibmxhtml1-transitional.dtd";

/// Public identifier prefixes that select quirks mode.
const QUIRKY_PUBLIC_ID_PREFIXES: &[&str] = &[
    "+//Silmaril//dtd html Pro v0r11 19970101//",
    "-//AS//DTD HTML 3.0 asWedit + extensions//",
    "-//AdvaSoft Ltd//DTD HTML 3.0 asWedit + extensions//",
    "-//IETF//DTD HTML 2.0 Level 1//",
    "-//IETF//DTD HTML 2.0 Level 2//",
    "-//IETF//DTD HTML 2.0 Strict Level 1//",
    "-//IETF//DTD HTML 2.0 Strict Level 2//",
    "-//IETF//DTD HTML 2.0 Strict//",
    "-//IETF//DTD HTML 2.0//",
    "-//IETF//DTD HTML 2.1E//",
    "-//IETF//DTD HTML 3.0//",
    "-//IETF//DTD HTML 3.2 Final//",
    "-//IETF//DTD HTML 3.2//",
    "-//IETF//DTD HTML 3//",
    "-//IETF//DTD HTML Level 0//",
    "-//IETF//DTD HTML Level 1//",
    "-//IETF//DTD HTML Level 2//",
    "-//IETF//DTD HTML Level 3//",
    "-//IETF//DTD HTML Strict Level 0//",
    "-//IETF//DTD HTML Strict Level 1//",
    "-//IETF//DTD HTML Strict Level 2//",
    "-//IETF//DTD HTML Strict Level 3//",
    "-//IETF//DTD HTML Strict//",
    "-//IETF//DTD HTML//",
    "-//Metrius//DTD Metrius Presentational//",
    "-//Microsoft//DTD Internet Explorer 2.0 HTML Strict//",
    "-//Microsoft//DTD Internet Explorer 2.0 HTML//",
    "-//Microsoft//DTD Internet Explorer 2.0 Tables//",
    "-//Microsoft//DTD Internet Explorer 3.0 HTML Strict//",
    "-//Microsoft//DTD Internet Explorer 3.0 HTML//",
    "-//Microsoft//DTD Internet Explorer 3.0 Tables//",
    "-//Netscape Comm. Corp.//DTD HTML//",
    "-//Netscape Comm. Corp.//DTD Strict HTML//",
    "-//O'Reilly and Associates//DTD HTML 2.0//",
    "-//O'Reilly and Associates//DTD HTML Extended 1.0//",
    "-//O'Reilly and Associates//DTD HTML Extended Relaxed 1.0//",
    "-//SQ//DTD HTML 2.0 HoTMetaL + extensions//",
    "-//SoftQuad Software//DTD HoTMetaL PRO 6.0::19990601::extensions to HTML 4.0//",
    "-//SoftQuad//DTD HoTMetaL PRO 4.0::19971010::extensions to HTML 4.0//",
    "-//Spyglass//DTD HTML 2.0 Extended//",
    "-//Sun Microsystems Corp.//DTD HotJava HTML//",
    "-//Sun Microsystems Corp.//DTD HotJava Strict HTML//",
    "-//W3C//DTD HTML 3 1995-03-24//",
    "-//W3C//DTD HTML 3.2 Draft//",
    "-//W3C//DTD HTML 3.2 Final//",
    "-//W3C//DTD HTML 3.2//",
    "-//W3C//DTD HTML 3.2S Draft//",
    "-//W3C//DTD HTML 4.0 Frameset//",
    "-//W3C//DTD HTML 4.0 Transitional//",
    "-//W3C//DTD HTML Experimental 19960712//",
    "-//W3C//DTD HTML Experimental 970421//",
    "-//W3C//DTD W3 HTML//",
    "-//W3O//DTD W3 HTML 3.0//",
    "-//WebTechs//DTD Mozilla HTML 2.0//",
    "-//WebTechs//DTD Mozilla HTML//",
];

/// Public identifier prefixes that select quirks mode when the system
/// identifier is missing and limited-quirks mode when it is present.
const CONDITIONAL_PREFIXES: &[&str] = &[
    "-//W3C//DTD HTML 4.01 Frameset//",
    "-//W3C//DTD HTML 4.01 Transitional//",
];

/// Public identifier prefixes that always select limited-quirks mode.
const LIMITED_QUIRKS_PREFIXES: &[&str] = &[
    "-//W3C//DTD XHTML 1.0 Frameset//",
    "-//W3C//DTD XHTML 1.0 Transitional//",
];

/// ASCII case-insensitive `starts_with`.
fn starts_with_ignore_ascii_case(haystack: &str, prefix: &str) -> bool {
    haystack.len() >= prefix.len()
        && haystack.is_char_boundary(prefix.len())
        && haystack[..prefix.len()].eq_ignore_ascii_case(prefix)
}

/// Decides the quirks mode a DOCTYPE token selects.
///
/// `parser_cannot_change_the_mode` corresponds to the flag the standard sets
/// for `srcdoc` documents and for fragment parsing; when it is `true` the mode
/// is always [`QuirksMode::NoQuirks`].
pub(crate) fn quirks_mode_for(
    doctype: &Doctype,
    parser_cannot_change_the_mode: bool,
) -> QuirksMode {
    if parser_cannot_change_the_mode {
        return QuirksMode::NoQuirks;
    }
    let public = doctype.public_id.as_deref();
    let system = doctype.system_id.as_deref();

    if doctype.force_quirks
        || doctype.name.as_deref() != Some("html")
        || public.is_some_and(|id| {
            QUIRKY_PUBLIC_IDS
                .iter()
                .any(|known| id.eq_ignore_ascii_case(known))
        })
        || system.is_some_and(|id| id.eq_ignore_ascii_case(QUIRKY_SYSTEM_ID))
        || public.is_some_and(|id| {
            QUIRKY_PUBLIC_ID_PREFIXES
                .iter()
                .any(|prefix| starts_with_ignore_ascii_case(id, prefix))
        })
        || (system.is_none()
            && public.is_some_and(|id| {
                CONDITIONAL_PREFIXES
                    .iter()
                    .any(|prefix| starts_with_ignore_ascii_case(id, prefix))
            }))
    {
        return QuirksMode::Quirks;
    }

    if public.is_some_and(|id| {
        LIMITED_QUIRKS_PREFIXES
            .iter()
            .any(|prefix| starts_with_ignore_ascii_case(id, prefix))
    }) || (system.is_some()
        && public.is_some_and(|id| {
            CONDITIONAL_PREFIXES
                .iter()
                .any(|prefix| starts_with_ignore_ascii_case(id, prefix))
        }))
    {
        return QuirksMode::LimitedQuirks;
    }

    QuirksMode::NoQuirks
}

#[cfg(test)]
mod tests {
    use super::{QuirksMode, quirks_mode_for};
    use crate::token::Doctype;

    fn doctype(name: &str, public: Option<&str>, system: Option<&str>) -> Doctype {
        Doctype {
            name: Some(name.to_owned()),
            public_id: public.map(str::to_owned),
            system_id: system.map(str::to_owned),
            force_quirks: false,
        }
    }

    #[test]
    fn html5_doctype_is_standards_mode() {
        assert_eq!(
            quirks_mode_for(&doctype("html", None, None), false),
            QuirksMode::NoQuirks
        );
    }

    #[test]
    fn html4_transitional_depends_on_the_system_identifier() {
        let public = Some("-//W3C//DTD HTML 4.01 Transitional//EN");
        assert_eq!(
            quirks_mode_for(&doctype("html", public, None), false),
            QuirksMode::Quirks
        );
        assert_eq!(
            quirks_mode_for(
                &doctype("html", public, Some("http://www.w3.org/TR/html4/loose.dtd")),
                false
            ),
            QuirksMode::LimitedQuirks
        );
    }

    #[test]
    fn legacy_public_identifiers_and_names_force_quirks() {
        assert_eq!(
            quirks_mode_for(&doctype("HTML", Some("HTML"), None), false),
            QuirksMode::Quirks
        );
        assert_eq!(
            quirks_mode_for(&doctype("xhtml", None, None), false),
            QuirksMode::Quirks
        );
        assert_eq!(
            quirks_mode_for(
                &doctype("html", Some("-//IETF//DTD HTML 3.0//EN"), None),
                false
            ),
            QuirksMode::Quirks
        );
    }

    #[test]
    fn xhtml1_transitional_is_limited_quirks() {
        assert_eq!(
            quirks_mode_for(
                &doctype("html", Some("-//W3C//DTD XHTML 1.0 Transitional//EN"), None),
                false
            ),
            QuirksMode::LimitedQuirks
        );
    }

    #[test]
    fn the_mode_is_frozen_for_fragments() {
        assert_eq!(
            quirks_mode_for(&doctype("nonsense", None, None), true),
            QuirksMode::NoQuirks
        );
    }
}
