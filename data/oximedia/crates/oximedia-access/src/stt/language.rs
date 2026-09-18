//! STT language model selection.

use serde::{Deserialize, Serialize};

/// STT language model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttLanguageModel {
    /// Language code.
    pub language: String,
    /// Domain-specific vocabulary.
    pub domain: Option<SttDomain>,
    /// Custom vocabulary phrases.
    pub custom_vocabulary: Vec<String>,
}

/// Domain for specialized vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SttDomain {
    /// General conversation.
    General,
    /// Medical terminology.
    Medical,
    /// Legal terminology.
    Legal,
    /// Technical/IT terminology.
    Technical,
    /// Finance and business.
    Finance,
}

impl SttLanguageModel {
    /// Create a new language model.
    #[must_use]
    pub fn new(language: String) -> Self {
        Self {
            language,
            domain: None,
            custom_vocabulary: Vec::new(),
        }
    }

    /// Set domain.
    #[must_use]
    pub const fn with_domain(mut self, domain: SttDomain) -> Self {
        self.domain = Some(domain);
        self
    }

    /// Add custom vocabulary.
    #[must_use]
    pub fn with_vocabulary(mut self, words: Vec<String>) -> Self {
        self.custom_vocabulary = words;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_model() {
        let model = SttLanguageModel::new("en".to_string()).with_domain(SttDomain::Medical);

        assert_eq!(model.language, "en");
        assert_eq!(model.domain, Some(SttDomain::Medical));
    }
}
