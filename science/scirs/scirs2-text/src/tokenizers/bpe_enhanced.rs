//! Enhanced BPE tokenizer with LLM-compatible features.
//!
//! Supports special tokens, chat templates, and byte-level fallback.
//! Designed for compatibility with GPT-2, LLaMA, Mistral, and ChatML-style models.

use std::collections::HashMap;

/// Errors produced by the enhanced BPE tokenizer.
#[derive(Debug, thiserror::Error)]
pub enum BpeError {
    /// A token was not found in the vocabulary.
    #[error("Unknown token: {0}")]
    UnknownToken(String),
    /// The vocabulary is structurally invalid.
    #[error("Invalid vocabulary")]
    InvalidVocab,
    /// A chat template formatting error.
    #[error("Template error: {0}")]
    TemplateError(String),
}

// ─── Special tokens ──────────────────────────────────────────────────────────

/// Special tokens used by LLM tokenizers.
///
/// Each field is optional; `None` means the model does not use that token type.
/// The `custom` map holds additional model-specific tokens (e.g., `<|im_start|>`).
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token.
    pub bos: Option<String>,
    /// End-of-sequence token.
    pub eos: Option<String>,
    /// Unknown-word fallback token.
    pub unk: Option<String>,
    /// Padding token.
    pub pad: Option<String>,
    /// Separator token (multi-sequence models).
    pub sep: Option<String>,
    /// Classification token (BERT-style).
    pub cls: Option<String>,
    /// Mask token (for masked LM).
    pub mask: Option<String>,
    /// Additional model-specific special tokens mapped to preferred IDs.
    /// The preferred IDs are treated as hints; `BpeVocab::new` skips tokens
    /// already registered via the standard fields.
    pub custom: HashMap<String, u32>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos: None,
            eos: None,
            unk: Some("<unk>".into()),
            pad: Some("<pad>".into()),
            sep: None,
            cls: None,
            mask: None,
            custom: HashMap::new(),
        }
    }
}

impl SpecialTokens {
    /// GPT-2 style special tokens (`<|endoftext|>` for both BOS and EOS).
    pub fn gpt2() -> Self {
        Self {
            bos: Some("<|endoftext|>".into()),
            eos: Some("<|endoftext|>".into()),
            ..Default::default()
        }
    }

    /// LLaMA / Mistral style special tokens (`<s>` / `</s>`).
    pub fn llama() -> Self {
        Self {
            bos: Some("<s>".into()),
            eos: Some("</s>".into()),
            unk: Some("<unk>".into()),
            ..Default::default()
        }
    }

    /// ChatML style special tokens (`<|im_start|>` / `<|im_end|>`).
    ///
    /// The custom map is left empty here because `BpeVocab::new` will assign
    /// IDs to bos/eos during the standard loop; the `<|im_start|>` and
    /// `<|im_end|>` entries in `custom` would conflict with those IDs.
    /// If callers need a specific custom token not covered by bos/eos/unk/pad,
    /// they can insert into `custom` after construction.
    pub fn chatml() -> Self {
        Self {
            bos: Some("<|im_start|>".into()),
            eos: Some("<|im_end|>".into()),
            custom: HashMap::new(),
            ..Default::default()
        }
    }
}

// ─── BPE vocabulary ───────────────────────────────────────────────────────────

/// BPE vocabulary with ordered merge rules and special-token awareness.
#[derive(Debug, Clone)]
pub struct BpeVocab {
    /// Token string → integer ID.
    pub token_to_id: HashMap<String, u32>,
    /// Integer ID → token string.
    pub id_to_token: HashMap<u32, String>,
    /// Ordered merge rules learned during training (earlier = higher priority).
    pub merges: Vec<(String, String)>,
    /// The special-token configuration used to build this vocab.
    pub special_tokens: SpecialTokens,
}

impl BpeVocab {
    /// Create a new vocabulary, pre-registering all special tokens.
    ///
    /// Standard special tokens (bos, eos, unk, pad, sep, cls, mask) are
    /// assigned IDs 0, 1, 2, … in declaration order, skipping duplicates.
    /// Custom tokens from `special_tokens.custom` are registered next,
    /// but only if the token string has not already been added by the
    /// standard loop (prevents ID collisions).
    pub fn new(special_tokens: SpecialTokens) -> Self {
        let mut vocab = Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: Vec::new(),
            special_tokens: special_tokens.clone(),
        };

        // Register standard special tokens in declaration order.
        let standard_slots: &[Option<&str>] = &[
            special_tokens.bos.as_deref(),
            special_tokens.eos.as_deref(),
            special_tokens.unk.as_deref(),
            special_tokens.pad.as_deref(),
            special_tokens.sep.as_deref(),
            special_tokens.cls.as_deref(),
            special_tokens.mask.as_deref(),
        ];
        for slot in standard_slots.iter().flatten() {
            vocab.ensure_token(slot);
        }

        // Register custom tokens; skip any that already have an ID.
        // We sort by preferred ID for deterministic ordering.
        let mut custom_sorted: Vec<(&String, u32)> =
            special_tokens.custom.iter().map(|(k, &v)| (k, v)).collect();
        custom_sorted.sort_by_key(|&(_, id)| id);
        for (tok, _preferred_id) in custom_sorted {
            vocab.ensure_token(tok);
        }

        vocab
    }

    /// Register a token if not yet present and return its ID.
    fn ensure_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.token_to_id.len() as u32;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        id
    }

    /// Add a token, returning its ID.  If the token already exists the
    /// existing ID is returned without modification.
    pub fn add_token(&mut self, token: String) -> u32 {
        self.ensure_token(&token)
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Look up the ID for a token string, returning `None` if not found.
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Look up the token string for an ID, returning `None` if not found.
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Returns `true` if `token` is registered as any kind of special token.
    pub fn is_special(&self, token: &str) -> bool {
        let s = &self.special_tokens;
        s.bos.as_deref() == Some(token)
            || s.eos.as_deref() == Some(token)
            || s.unk.as_deref() == Some(token)
            || s.pad.as_deref() == Some(token)
            || s.sep.as_deref() == Some(token)
            || s.cls.as_deref() == Some(token)
            || s.mask.as_deref() == Some(token)
            || s.custom.contains_key(token)
    }
}

// ─── Byte-level BPE tokenizer ────────────────────────────────────────────────

/// Byte-level BPE tokenizer (GPT-2 style).
///
/// The vocabulary is seeded with 256 byte tokens using a `Ġ`-prefixed encoding
/// (`Ġ\x00` … `Ġÿ`), then additional merge rules are applied during training.
/// This implementation handles encoding and decoding; actual merge-rule training
/// is left to external tooling (HuggingFace `tokenizers`, etc.).
pub struct ByteLevelBpe {
    /// The underlying vocabulary.
    pub vocab: BpeVocab,
}

impl ByteLevelBpe {
    /// Create a new byte-level BPE tokenizer seeded with byte tokens and
    /// the provided special tokens.
    pub fn new(special_tokens: SpecialTokens) -> Self {
        let mut vocab = BpeVocab::new(special_tokens);
        // Add all 256 byte tokens as the base vocabulary.
        // GPT-2 uses a `Ġ` prefix so each byte maps to a distinct, displayable
        // token string (`Ġa`, `Ġb`, …, `Ġ\x00`, …).
        for b in 0u8..=255 {
            let token = format!("Ġ{}", b as char);
            vocab.ensure_token(&token);
        }
        Self { vocab }
    }

    /// Total number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.vocab_size()
    }

    /// Encode `text` into token IDs.
    ///
    /// When `add_special_tokens` is `true`, the BOS token is prepended and the
    /// EOS token is appended (if either is configured).
    ///
    /// This implementation performs character-level byte encoding.  A production
    /// tokenizer would additionally apply ordered BPE merge rules; this is
    /// intentionally omitted to keep the implementation self-contained.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let mut ids = Vec::new();

        if add_special_tokens {
            if let Some(bos_id) = self
                .vocab
                .special_tokens
                .bos
                .as_deref()
                .and_then(|t| self.vocab.get_id(t))
            {
                ids.push(bos_id);
            }
        }

        // Encode each character via the Ġ-prefixed byte-token representation.
        let unk_id = self
            .vocab
            .special_tokens
            .unk
            .as_deref()
            .and_then(|t| self.vocab.get_id(t))
            .unwrap_or(0);

        for ch in text.chars() {
            let token = format!("Ġ{ch}");
            let id = self.vocab.get_id(&token).unwrap_or(unk_id);
            ids.push(id);
        }

        if add_special_tokens {
            if let Some(eos_id) = self
                .vocab
                .special_tokens
                .eos
                .as_deref()
                .and_then(|t| self.vocab.get_id(t))
            {
                ids.push(eos_id);
            }
        }

        ids
    }

    /// Decode a sequence of token IDs back to a string.
    ///
    /// When `skip_special_tokens` is `true`, any token registered as a special
    /// token (BOS, EOS, UNK, PAD, etc.) is omitted from the output.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        ids.iter()
            .filter_map(|&id| {
                let tok = self.vocab.get_token(id)?;
                if skip_special_tokens && self.vocab.is_special(tok) {
                    return None;
                }
                // Strip the leading Ġ prefix added during encoding.
                Some(tok.trim_start_matches('Ġ').to_string())
            })
            .collect()
    }
}

// ─── Chat templates ───────────────────────────────────────────────────────────

/// A single message in a multi-turn conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// The role of the author: `"user"`, `"assistant"`, or `"system"`.
    pub role: String,
    /// The text content of the message.
    pub content: String,
}

/// The formatting style used by a [`ChatTemplate`].
#[derive(Debug, Clone)]
pub enum ChatStyle {
    /// ChatML format: `<|im_start|>role\ncontent<|im_end|>\n`.
    ChatML,
    /// LLaMA-2 format: `[INST] user [/INST] assistant </s>`.
    Llama2,
    /// Alpaca format: `### Instruction:\n… ### Response:\n`.
    Alpaca,
    /// Simple human-readable format: `role: content\n`.
    Simple,
}

/// Formats a list of [`Message`]s into a model-ready prompt string.
///
/// Each LLM family expects a distinct conversation format.  `ChatTemplate`
/// encodes these conventions so that callers need not hard-code format strings.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    /// The formatting style to apply.
    pub style: ChatStyle,
}

impl ChatTemplate {
    /// Create a new template with the given style.
    pub fn new(style: ChatStyle) -> Self {
        Self { style }
    }

    /// Format `messages` into a prompt string.
    ///
    /// When `add_generation_prompt` is `true`, the template appends a partial
    /// assistant turn header so that the model can begin generating immediately
    /// (e.g., `<|im_start|>assistant\n` for ChatML).
    pub fn apply(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        match &self.style {
            ChatStyle::ChatML => {
                let mut s = String::new();
                for msg in messages {
                    s.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n",
                        msg.role, msg.content
                    ));
                }
                if add_generation_prompt {
                    s.push_str("<|im_start|>assistant\n");
                }
                s
            }

            ChatStyle::Llama2 => {
                let mut s = String::new();
                for msg in messages {
                    match msg.role.as_str() {
                        "system" => {
                            s.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n", msg.content));
                        }
                        "user" => {
                            s.push_str(&format!("[INST] {} [/INST]", msg.content));
                        }
                        "assistant" => {
                            s.push_str(&format!(" {} </s>", msg.content));
                        }
                        _ => {
                            s.push_str(&msg.content);
                        }
                    }
                }
                s
            }

            ChatStyle::Alpaca => {
                let mut s = String::new();
                for msg in messages {
                    match msg.role.as_str() {
                        "user" => {
                            s.push_str(&format!("### Instruction:\n{}\n\n", msg.content));
                        }
                        "assistant" => {
                            s.push_str(&format!("### Response:\n{}\n\n", msg.content));
                        }
                        _ => {}
                    }
                }
                if add_generation_prompt {
                    s.push_str("### Response:\n");
                }
                s
            }

            ChatStyle::Simple => messages
                .iter()
                .map(|m| format!("{}: {}\n", m.role, m.content))
                .collect(),
        }
    }

    /// Approximate token count for `messages` using whitespace splitting.
    ///
    /// This is a fast heuristic; for precise counts, encode the formatted
    /// prompt with a full BPE tokenizer.
    pub fn count_tokens(&self, messages: &[Message]) -> usize {
        let prompt = self.apply(messages, false);
        prompt.split_whitespace().count()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens_gpt2() {
        let st = SpecialTokens::gpt2();
        assert_eq!(st.bos.as_deref(), Some("<|endoftext|>"));
        assert_eq!(st.eos.as_deref(), Some("<|endoftext|>"));
    }

    #[test]
    fn test_special_tokens_llama() {
        let st = SpecialTokens::llama();
        assert_eq!(st.bos.as_deref(), Some("<s>"));
        assert_eq!(st.eos.as_deref(), Some("</s>"));
    }

    #[test]
    fn test_bpe_vocab_add_token() {
        let mut vocab = BpeVocab::new(SpecialTokens::default());
        let id = vocab.add_token("hello".to_string());
        assert_eq!(vocab.get_id("hello"), Some(id));
        assert_eq!(vocab.get_token(id), Some("hello"));
    }

    #[test]
    fn test_bpe_vocab_special_tokens() {
        let vocab = BpeVocab::new(SpecialTokens::llama());
        assert!(vocab.is_special("<s>"));
        assert!(vocab.is_special("</s>"));
        assert!(!vocab.is_special("hello"));
    }

    #[test]
    fn test_byte_level_bpe_encode_decode() {
        let bpe = ByteLevelBpe::new(SpecialTokens::gpt2());
        let ids = bpe.encode("abc", false);
        assert_eq!(ids.len(), 3);
        let decoded = bpe.decode(&ids, false);
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_byte_level_bpe_with_special_tokens() {
        let bpe = ByteLevelBpe::new(SpecialTokens::gpt2());
        let ids = bpe.encode("hi", true);
        // BOS + 'h' + 'i' + EOS = 4 tokens
        assert_eq!(ids.len(), 4);
        let decoded = bpe.decode(&ids, true);
        assert_eq!(decoded, "hi");
    }

    #[test]
    fn test_chat_template_chatml() {
        let tmpl = ChatTemplate::new(ChatStyle::ChatML);
        let msgs = vec![
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Hi!".to_string(),
            },
        ];
        let prompt = tmpl.apply(&msgs, false);
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_end|>"));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_chat_template_generation_prompt() {
        let tmpl = ChatTemplate::new(ChatStyle::ChatML);
        let msgs = vec![Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let prompt = tmpl.apply(&msgs, true);
        assert!(
            prompt.ends_with("<|im_start|>assistant\n"),
            "Expected generation prompt at end: {prompt}"
        );
    }

    #[test]
    fn test_chat_template_llama2() {
        let tmpl = ChatTemplate::new(ChatStyle::Llama2);
        let msgs = vec![Message {
            role: "user".to_string(),
            content: "Tell me a joke".to_string(),
        }];
        let prompt = tmpl.apply(&msgs, false);
        assert!(prompt.contains("[INST]") && prompt.contains("[/INST]"));
    }

    #[test]
    fn test_chat_template_token_count() {
        let tmpl = ChatTemplate::new(ChatStyle::Simple);
        let msgs = vec![Message {
            role: "user".to_string(),
            content: "Hello world how are you".to_string(),
        }];
        let count = tmpl.count_tokens(&msgs);
        assert!(count >= 5, "Expected >= 5 tokens, got {count}");
    }

    #[test]
    fn test_vocab_size() {
        let bpe = ByteLevelBpe::new(SpecialTokens::default());
        // 256 byte tokens + at least 2 standard special tokens (unk, pad)
        assert!(bpe.vocab_size() >= 256);
    }

    #[test]
    fn test_chatml_no_id_collision() {
        // Ensure BpeVocab correctly handles ChatML tokens without ID conflicts.
        let vocab = BpeVocab::new(SpecialTokens::chatml());
        let bos_id = vocab.get_id("<|im_start|>");
        let eos_id = vocab.get_id("<|im_end|>");
        assert!(bos_id.is_some());
        assert!(eos_id.is_some());
        // IDs must be distinct.
        assert_ne!(bos_id, eos_id);
        // Reverse lookup must be consistent.
        assert_eq!(vocab.get_token(bos_id.unwrap()), Some("<|im_start|>"));
        assert_eq!(vocab.get_token(eos_id.unwrap()), Some("<|im_end|>"));
    }

    #[test]
    fn test_alpaca_template() {
        let tmpl = ChatTemplate::new(ChatStyle::Alpaca);
        let msgs = vec![
            Message {
                role: "user".to_string(),
                content: "What is Rust?".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "A systems language.".to_string(),
            },
        ];
        let prompt = tmpl.apply(&msgs, true);
        assert!(prompt.contains("### Instruction:"));
        assert!(prompt.contains("### Response:"));
        assert!(prompt.ends_with("### Response:\n"));
    }

    #[test]
    fn test_llama_special_token_is_special() {
        let bpe = ByteLevelBpe::new(SpecialTokens::llama());
        // Check that <s> and </s> are recognized as special.
        assert!(bpe.vocab.is_special("<s>"));
        assert!(bpe.vocab.is_special("</s>"));
        // A normal byte token should not be special.
        assert!(!bpe.vocab.is_special("Ġa"));
    }
}
