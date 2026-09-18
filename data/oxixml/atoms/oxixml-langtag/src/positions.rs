//! Byte-offset bookkeeping that lets [`crate::LanguageTag`] slice its accessor substrings out of
//! the single string it stores, without re-parsing or allocating.

/// Which top-level ABNF production a successfully parsed tag matched.
///
/// Component accessors ([`primary_language`](crate::LanguageTag::primary_language),
/// [`script`](crate::LanguageTag::script), …) are computed purely from byte offsets and do not
/// need to know the shape — except [`private_use`](crate::LanguageTag::private_use), which
/// returns the *entire* tag for a standalone `privateuse` production (there is no language
/// subtag to exclude), so that one case is tracked explicitly instead of being inferred by
/// re-inspecting the string's casing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TagShape {
    /// A `langtag` production: `language ["-" script] ["-" region] *("-" variant)
    /// *("-" extension) ["-" privateuse]`.
    Standard,
    /// A standalone `privateuse` production (`x-...`), carrying no language subtag at all.
    PrivateUseOnly,
    /// One of the fixed `grandfathered` tags, kept and compared as a whole.
    Grandfathered,
}

/// The end offsets (exclusive, in bytes) of each component of a parsed tag, relative to the
/// start of the tag string.
///
/// Every field after `language_end` is `None` when that component is absent from the tag.
/// Accessors reconstruct "the offset right before component X would start" by walking back
/// through whichever earlier components *are* present — see the `*_block_end` helpers below —
/// rather than pre-filling every field with a repeated sentinel, which keeps "is this component
/// present" and "where does the tag's prefix through the previous component end" as two
/// separate, individually obvious questions.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TagElementsPositions {
    pub(crate) shape: TagShape,
    /// End offset of the primary language subtag alone (never absent once parsing succeeds).
    pub(crate) language_end: usize,
    /// End offset of the last extended-language (`extlang`) subtag, if any were present.
    pub(crate) extlang_end: Option<usize>,
    /// End offset of the script subtag, if present.
    pub(crate) script_end: Option<usize>,
    /// End offset of the region subtag, if present.
    pub(crate) region_end: Option<usize>,
    /// End offset of the last variant subtag, if one or more were present.
    pub(crate) variant_end: Option<usize>,
    /// End offset of the last extension value subtag, if one or more extensions were present.
    pub(crate) extension_end: Option<usize>,
}

impl TagElementsPositions {
    /// Builds the position set used for tags with no internal structure to report — a
    /// standalone private-use tag or a grandfathered tag — where every accessor other than
    /// [`primary_language`](crate::LanguageTag::primary_language) (and, for private use,
    /// [`private_use`](crate::LanguageTag::private_use)) returns `None`.
    pub(crate) const fn whole_tag(shape: TagShape, len: usize) -> Self {
        Self {
            shape,
            language_end: len,
            extlang_end: None,
            script_end: None,
            region_end: None,
            variant_end: None,
            extension_end: None,
        }
    }

    /// End offset through the language and extended-language subtags.
    pub(crate) fn lang_block_end(&self) -> usize {
        self.extlang_end.unwrap_or(self.language_end)
    }

    /// End offset through the language, extended-language, and script subtags.
    pub(crate) fn script_block_end(&self) -> usize {
        self.script_end.unwrap_or_else(|| self.lang_block_end())
    }

    /// End offset through the language, extended-language, script, and region subtags.
    pub(crate) fn region_block_end(&self) -> usize {
        self.region_end.unwrap_or_else(|| self.script_block_end())
    }

    /// End offset through every component up to and including the variant subtags.
    pub(crate) fn variant_block_end(&self) -> usize {
        self.variant_end.unwrap_or_else(|| self.region_block_end())
    }

    /// End offset through every component up to and including the extensions.
    pub(crate) fn extension_block_end(&self) -> usize {
        self.extension_end
            .unwrap_or_else(|| self.variant_block_end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_tag_collapses_every_block_end_to_the_language_end() {
        let positions = TagElementsPositions::whole_tag(TagShape::PrivateUseOnly, 7);
        assert_eq!(positions.language_end, 7);
        assert_eq!(positions.lang_block_end(), 7);
        assert_eq!(positions.script_block_end(), 7);
        assert_eq!(positions.region_block_end(), 7);
        assert_eq!(positions.variant_block_end(), 7);
        assert_eq!(positions.extension_block_end(), 7);
    }

    #[test]
    fn block_end_chain_falls_back_through_absent_components() {
        let positions = TagElementsPositions {
            shape: TagShape::Standard,
            language_end: 2,
            extlang_end: None,
            script_end: None,
            region_end: Some(5),
            variant_end: None,
            extension_end: None,
        };
        assert_eq!(positions.lang_block_end(), 2);
        assert_eq!(positions.script_block_end(), 2);
        assert_eq!(positions.region_block_end(), 5);
        assert_eq!(positions.variant_block_end(), 5);
        assert_eq!(positions.extension_block_end(), 5);
    }
}
