//! Processor-wide options.

/// Options controlling [`process_document_with`](crate::process_document_with).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XIncludeOptions {
    max_depth: usize,
}

impl XIncludeOptions {
    /// The default maximum inclusion depth (see [`Self::max_depth`]).
    pub const DEFAULT_MAX_DEPTH: usize = 100;

    /// Creates the default options.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_depth: Self::DEFAULT_MAX_DEPTH,
        }
    }

    /// Sets the maximum number of nested resource *acquisitions* (not plain element nesting)
    /// the processor will follow before giving up with
    /// [`XIncludeError::MaxDepthExceeded`](crate::XIncludeError::MaxDepthExceeded).
    ///
    /// This is a defensive guard on top of the mandatory
    /// [loop detection](https://www.w3.org/TR/xinclude/#loops) XInclude itself requires
    /// (which this crate always performs, regardless of this setting): loop detection alone
    /// cannot bound the depth of a legal, non-repeating chain of distinct resources, or of a
    /// self-referencing inclusion that walks deeper into the same document with a different
    /// `xpointer` at every step.
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Returns the configured maximum inclusion depth.
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }
}

impl Default for XIncludeOptions {
    fn default() -> Self {
        Self::new()
    }
}
