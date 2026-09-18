//! CEA-608 and CEA-708 closed caption encoding and embedding.
//!
//! This module provides complete support for encoding closed captions
//! according to CEA-608, CEA-708, and ATSC A/53 standards.

pub mod embed;
pub mod encoder;

// Re-export main types
pub use embed::{
    A53Validator, CaptionEmbedder, FrameRate, FrameRateAdapter, Line21Encoder,
    Mpeg2UserDataBuilder, SeiNalBuilder, TimecodeCalculator, VideoField, VideoFormat,
};

pub use encoder::{
    get_framerate_code, Cea608Attributes, Cea608Channel, Cea608Color, Cea608Command, Cea608Encoder,
    Cea608ExtendedChar, Cea608MidRowCode, Cea608Mode, Cea608Pac, Cea608SpecialChar, Cea708Color,
    Cea708Command, Cea708Encoder, Cea708FontStyle, Cea708Opacity, Cea708PenAttributes,
    Cea708PenColor, Cea708PenSize, Cea708ServiceNumber, Cea708WindowAnchor, Cea708WindowAttributes,
    Cea708WindowId,
};
