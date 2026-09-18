//! Transformer tokenizers.
//!
//! This module provides production-quality tokenizer implementations for
//! modern transformer architectures:
//!
//! - [`bert`]: BERT-style WordPiece tokenizer with `[CLS]` / `[SEP]` / `[MASK]`
//!   special tokens, batch encoding with padding/truncation, pair encoding with
//!   `token_type_ids`.
//!
//! - [`roberta`]: RoBERTa byte-level BPE tokenizer with `<s>` / `</s>` special
//!   tokens and GPT-2 compatible byte encoding.

/// BERT-style WordPiece tokenizer.
pub mod bert;
/// Enhanced BPE tokenizer with LLM-compatible features (special tokens, chat templates).
pub mod bpe_enhanced;
/// HuggingFace tokenizers JSON serialization format.
pub mod hf_json;
/// RoBERTa byte-level BPE tokenizer.
pub mod roberta;
/// Language-agnostic Unicode tokenizer (CJK, accents, punctuation).
pub mod unicode;

pub use bert::{BatchEncoding, BertEncoding, BertTokenizer};
pub use bpe_enhanced::{
    BpeError, BpeVocab as EnhancedBpeVocab, ByteLevelBpe, ChatStyle, ChatTemplate, Message,
    SpecialTokens,
};
pub use hf_json::{HfAddedToken, HfNormalizerConfig, HfTokenizerJson};
pub use roberta::RobertaTokenizer;
pub use unicode::{UnicodeTokenizer, UnicodeTokenizerConfig};
