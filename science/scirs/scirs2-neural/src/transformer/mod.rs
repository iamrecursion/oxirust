//! Transformer models implementation
//!
//! This module provides implementation of transformer models as described
//! in "Attention Is All You Need" by Vaswani et al., including encoder and
//! decoder layers and full transformer architectures.

mod decoder;
mod encoder;
mod model;
// Re-export from layers to provide compatibility
pub use crate::layers::{MultiHeadAttention, SelfAttention};
// Re-export positional encoding from utils
pub use crate::utils::positional_encoding::{
    LearnedPositionalEncoding, PositionalEncoding, PositionalEncodingFactory,
    PositionalEncodingType, RelativePositionalEncoding, RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
};
// Re-export transformer components
pub use decoder::{TransformerDecoder, TransformerDecoderLayer};
pub use encoder::{FeedForward, TransformerEncoder, TransformerEncoderLayer};
pub use model::{Transformer, TransformerConfig};
