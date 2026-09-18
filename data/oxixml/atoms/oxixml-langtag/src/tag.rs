//! The [`LanguageTag`] type itself: storage, parsing entry points, and component accessors.

use alloc::borrow::ToOwned;
use alloc::string::String;
use core::ops::Deref;

use crate::error::LanguageTagParseError;
use crate::extension_iter::ExtensionSubtags;
use crate::parse::{parse_langtag, parse_positions, parse_privateuse_only};
use crate::positions::{TagElementsPositions, TagShape};

/// A [BCP 47](https://www.rfc-editor.org/info/bcp47) / [RFC 5646](https://www.rfc-editor.org/rfc/rfc5646) language tag.
///
/// `LanguageTag` is generic over its backing storage `T` (anything that derefs to `str`), so it
/// can borrow (`LanguageTag<&str>`), own (`LanguageTag<String>`), or share (`LanguageTag<Rc<str>>`)
/// the tag text without committing callers to one representation. [`LanguageTag::parse`] performs
/// no heap allocation of its own: it stores `tag` as given and records only a handful of byte
/// offsets alongside it.
///
/// This checks **well-formedness** only ([RFC 5646 §2.2.9](https://www.rfc-editor.org/rfc/rfc5646#section-2.2.9)):
/// syntactic conformance to the `langtag` / `privateuse` / `grandfathered` grammar, plus
/// rejection of duplicate variant subtags and duplicate extension singletons (which RFC 5646
/// groups under **valid**, but which this crate treats as parse failures because such a tag has
/// no unambiguous interpretation). It does *not* check subtags against the IANA Language Subtag
/// Registry — a well-formed tag like `qq-Zzzz` parses successfully even though neither `qq` nor
/// `Zzzz` is an assigned subtag.
///
/// ```
/// use oxixml_langtag::LanguageTag;
///
/// let tag = LanguageTag::parse("zh-cmn-Hans-CN").expect("well-formed BCP 47 tag");
/// assert_eq!(tag.primary_language(), "zh");
/// assert_eq!(tag.extended_language(), Some("cmn"));
/// assert_eq!(tag.script(), Some("Hans"));
/// assert_eq!(tag.region(), Some("CN"));
/// assert_eq!(tag.as_str(), "zh-cmn-Hans-CN");
/// ```
#[derive(Clone, Copy)]
pub struct LanguageTag<T> {
    pub(crate) tag: T,
    pub(crate) positions: TagElementsPositions,
}

impl<T: Deref<Target = str>> LanguageTag<T> {
    /// Parses `tag` as a BCP 47 language tag, checking [well-formedness](Self) only.
    ///
    /// The original `tag` value is stored unchanged (in particular, its casing is preserved) —
    /// use [`LanguageTag::parse_and_normalize`] instead if case-normalized output is wanted.
    ///
    /// # Errors
    ///
    /// Returns [`LanguageTagParseError`] if `tag` does not match the `Language-Tag` ABNF
    /// production, or has a duplicate variant subtag or a duplicate extension singleton.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// assert!(LanguageTag::parse("en-US").is_ok());
    /// assert!(LanguageTag::parse("en-US".to_string()).is_ok());
    /// assert!(LanguageTag::parse("a-DE").is_err()); // single-letter primary language
    /// ```
    pub fn parse(tag: T) -> Result<Self, LanguageTagParseError> {
        let positions = parse_positions(tag.as_bytes())?;
        Ok(Self { tag, positions })
    }

    /// Returns the tag as originally given to [`LanguageTag::parse`] (or the normalized string,
    /// for a tag built by [`LanguageTag::parse_and_normalize`]).
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// assert_eq!(LanguageTag::parse("en-us").expect("valid").as_str(), "en-us");
    /// ```
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.tag
    }

    /// Consumes `self`, returning the underlying storage of type `T`.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag: String = LanguageTag::parse("en-us".to_string())
    ///     .expect("valid")
    ///     .into_inner();
    /// assert_eq!(tag, "en-us");
    /// ```
    #[inline]
    pub fn into_inner(self) -> T {
        self.tag
    }

    /// Returns the [primary language subtag](https://www.rfc-editor.org/rfc/rfc5646#section-2.2.1).
    ///
    /// For a standalone private-use tag (`x-...`) or a grandfathered tag, this returns the
    /// entire tag, since neither production has a separate primary-language component.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("zh-cmn-Hans-CN").expect("valid");
    /// assert_eq!(tag.primary_language(), "zh");
    /// ```
    #[inline]
    pub fn primary_language(&self) -> &str {
        &self.tag[..self.positions.language_end]
    }

    /// Returns the [extended language subtag(s)](https://www.rfc-editor.org/rfc/rfc5646#section-2.2.2),
    /// as one hyphen-joined string, or `None` if the tag has none.
    ///
    /// A well-formed tag has at most one *canonical* extended-language subtag but may carry up
    /// to three (permanently reserved for compatibility); use
    /// [`LanguageTag::extended_language_subtags`] to iterate them individually.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("zh-cmn-Hans-CN").expect("valid");
    /// assert_eq!(tag.extended_language(), Some("cmn"));
    /// assert_eq!(LanguageTag::parse("zh").expect("valid").extended_language(), None);
    /// ```
    #[inline]
    pub fn extended_language(&self) -> Option<&str> {
        self.positions
            .extlang_end
            .map(|end| &self.tag[self.positions.language_end + 1..end])
    }

    /// Iterates over the individual extended language subtags, in order.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("zh-cmn-Hans-CN").expect("valid");
    /// assert_eq!(tag.extended_language_subtags().collect::<Vec<_>>(), vec!["cmn"]);
    /// ```
    #[inline]
    pub fn extended_language_subtags(&self) -> impl Iterator<Item = &str> {
        self.extended_language()
            .unwrap_or_default()
            .split_terminator('-')
    }

    /// Returns the primary language subtag together with its extended language subtag(s), as
    /// they appear in the tag (i.e. [`primary_language`](Self::primary_language) plus
    /// [`extended_language`](Self::extended_language), hyphen-joined).
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("zh-cmn-Hans-CN").expect("valid");
    /// assert_eq!(tag.full_language(), "zh-cmn");
    /// ```
    #[inline]
    pub fn full_language(&self) -> &str {
        &self.tag[..self.positions.lang_block_end()]
    }

    /// Returns the [script subtag](https://www.rfc-editor.org/rfc/rfc5646#section-2.2.3), if
    /// present.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("zh-cmn-Hans-CN").expect("valid");
    /// assert_eq!(tag.script(), Some("Hans"));
    /// assert_eq!(LanguageTag::parse("zh").expect("valid").script(), None);
    /// ```
    #[inline]
    pub fn script(&self) -> Option<&str> {
        self.positions
            .script_end
            .map(|end| &self.tag[self.positions.lang_block_end() + 1..end])
    }

    /// Returns the [region subtag](https://www.rfc-editor.org/rfc/rfc5646#section-2.2.4), if
    /// present.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("zh-cmn-Hans-CN").expect("valid");
    /// assert_eq!(tag.region(), Some("CN"));
    /// assert_eq!(LanguageTag::parse("es-419").expect("valid").region(), Some("419"));
    /// ```
    #[inline]
    pub fn region(&self) -> Option<&str> {
        self.positions
            .region_end
            .map(|end| &self.tag[self.positions.script_block_end() + 1..end])
    }

    /// Returns every [variant subtag](https://www.rfc-editor.org/rfc/rfc5646#section-2.2.5), as
    /// one hyphen-joined string, or `None` if the tag has none.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("sl-rozaj-biske").expect("valid");
    /// assert_eq!(tag.variant(), Some("rozaj-biske"));
    /// ```
    #[inline]
    pub fn variant(&self) -> Option<&str> {
        self.positions
            .variant_end
            .map(|end| &self.tag[self.positions.region_block_end() + 1..end])
    }

    /// Iterates over the individual variant subtags, in order.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("sl-rozaj-biske").expect("valid");
    /// assert_eq!(tag.variant_subtags().collect::<Vec<_>>(), vec!["rozaj", "biske"]);
    /// ```
    #[inline]
    pub fn variant_subtags(&self) -> impl Iterator<Item = &str> {
        self.variant().unwrap_or_default().split_terminator('-')
    }

    /// Returns every [extension sequence](https://www.rfc-editor.org/rfc/rfc5646#section-2.2.6)
    /// (every singleton and its values), as one hyphen-joined string, or `None` if the tag has
    /// none.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("de-DE-u-co-phonebk").expect("valid");
    /// assert_eq!(tag.extension(), Some("u-co-phonebk"));
    /// ```
    #[inline]
    pub fn extension(&self) -> Option<&str> {
        self.positions
            .extension_end
            .map(|end| &self.tag[self.positions.variant_block_end() + 1..end])
    }

    /// Iterates over `(singleton, values)` pairs, one per extension sequence, in order.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("de-DE-u-co-phonebk").expect("valid");
    /// assert_eq!(
    ///     tag.extension_subtags().collect::<Vec<_>>(),
    ///     vec![('u', "co-phonebk")]
    /// );
    /// ```
    #[inline]
    pub fn extension_subtags(&self) -> impl Iterator<Item = (char, &str)> {
        ExtensionSubtags::new(self.extension().unwrap_or_default())
    }

    /// Returns the [private use sequence](https://www.rfc-editor.org/rfc/rfc5646#section-2.2.7),
    /// including its leading `x-` introducer, or `None` if the tag has none.
    ///
    /// For a standalone private-use tag (one that has *no* primary language subtag at all, e.g.
    /// `x-whatever`), this returns the whole tag.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("de-x-foo-bar").expect("valid");
    /// assert_eq!(tag.private_use(), Some("x-foo-bar"));
    ///
    /// let standalone = LanguageTag::parse("x-fr-ch").expect("valid");
    /// assert_eq!(standalone.private_use(), Some("x-fr-ch"));
    /// ```
    #[inline]
    pub fn private_use(&self) -> Option<&str> {
        if self.positions.shape == TagShape::PrivateUseOnly {
            return Some(&self.tag);
        }
        let start = self.positions.extension_block_end();
        if start == self.tag.len() {
            None
        } else {
            Some(&self.tag[start + 1..])
        }
    }

    /// Iterates over the individual private-use value subtags (i.e. everything after the `x-`
    /// introducer), in order.
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse("de-x-foo-bar").expect("valid");
    /// assert_eq!(tag.private_use_subtags().collect::<Vec<_>>(), vec!["foo", "bar"]);
    /// ```
    #[inline]
    pub fn private_use_subtags(&self) -> impl Iterator<Item = &str> {
        self.private_use()
            .map(|sequence| &sequence[2..])
            .unwrap_or_default()
            .split_terminator('-')
    }
}

impl LanguageTag<String> {
    /// Parses `tag` and normalizes its casing per
    /// [RFC 5646 §2.1.1](https://www.rfc-editor.org/rfc/rfc5646#section-2.1.1): every subtag is
    /// lowercased, except the script subtag (Titlecased: first letter upper, rest lower) and the
    /// region subtag (UPPERCASED). Grandfathered tags normalize to the fixed canonical casing
    /// RFC 5646 lists for each of them (e.g. `en-GB-oed`, `sgn-BE-FR`), regardless of the casing
    /// they were given in.
    ///
    /// This allocates: it always produces an owned, freshly cased `String`, even when the input
    /// was already in canonical form.
    ///
    /// # Errors
    ///
    /// Returns [`LanguageTagParseError`] under the same conditions as [`LanguageTag::parse`].
    ///
    /// ```
    /// use oxixml_langtag::LanguageTag;
    ///
    /// let tag = LanguageTag::parse_and_normalize("mn-cyrl-mn").expect("well-formed");
    /// assert_eq!(tag.as_str(), "mn-Cyrl-MN");
    ///
    /// let tag = LanguageTag::parse_and_normalize("en-us").expect("well-formed");
    /// assert_eq!(tag.as_str(), "en-US");
    /// ```
    pub fn parse_and_normalize(tag: &str) -> Result<Self, LanguageTagParseError> {
        if let Some(canonical) = crate::grandfathered::canonical_grandfathered(tag.as_bytes()) {
            return Ok(Self {
                tag: canonical.to_owned(),
                positions: TagElementsPositions::whole_tag(
                    TagShape::Grandfathered,
                    canonical.len(),
                ),
            });
        }
        if crate::parse::starts_privateuse(tag.as_bytes()) {
            let positions = parse_privateuse_only(tag.as_bytes())?;
            let mut normalized = tag.to_owned();
            normalized.make_ascii_lowercase();
            return Ok(Self {
                tag: normalized,
                positions,
            });
        }

        let positions = parse_langtag(tag.as_bytes())?;
        let mut normalized = tag.to_owned();

        let lang_end = positions.lang_block_end();
        normalized[..lang_end].make_ascii_lowercase();

        if let Some(script_end) = positions.script_end {
            let script_start = lang_end + 1;
            normalized[script_start..script_start + 1].make_ascii_uppercase();
            normalized[script_start + 1..script_end].make_ascii_lowercase();
        }

        if let Some(region_end) = positions.region_end {
            let region_start = positions.script_block_end() + 1;
            normalized[region_start..region_end].make_ascii_uppercase();
        }

        let tail_start = positions.region_block_end();
        if tail_start < normalized.len() {
            normalized[tail_start + 1..].make_ascii_lowercase();
        }

        Ok(Self {
            tag: normalized,
            positions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn parse_preserves_original_casing() -> Result<(), LanguageTagParseError> {
        let tag = LanguageTag::parse("EN-us")?;
        assert_eq!(tag.as_str(), "EN-us");
        Ok(())
    }

    #[test]
    fn accessors_on_full_tag() -> Result<(), LanguageTagParseError> {
        let tag = LanguageTag::parse("zh-cmn-Hans-CN-pinyin-u-co-phonebk-x-priv")?;
        assert_eq!(tag.primary_language(), "zh");
        assert_eq!(tag.extended_language(), Some("cmn"));
        assert_eq!(tag.full_language(), "zh-cmn");
        assert_eq!(tag.script(), Some("Hans"));
        assert_eq!(tag.region(), Some("CN"));
        assert_eq!(tag.variant(), Some("pinyin"));
        assert_eq!(tag.extension(), Some("u-co-phonebk"));
        assert_eq!(tag.private_use(), Some("x-priv"));
        Ok(())
    }

    #[test]
    fn into_inner_roundtrips() -> Result<(), LanguageTagParseError> {
        let tag = LanguageTag::parse("en-us".to_string())?;
        assert_eq!(tag.into_inner(), "en-us");
        Ok(())
    }

    #[test]
    fn normalize_grandfathered() -> Result<(), LanguageTagParseError> {
        assert_eq!(
            LanguageTag::parse_and_normalize("SGN-be-FR")?.as_str(),
            "sgn-BE-FR"
        );
        Ok(())
    }

    #[test]
    fn normalize_privateuse_only() -> Result<(), LanguageTagParseError> {
        assert_eq!(
            LanguageTag::parse_and_normalize("X-Fr-CH")?.as_str(),
            "x-fr-ch"
        );
        Ok(())
    }
}
