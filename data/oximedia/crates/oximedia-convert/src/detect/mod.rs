// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Media format and codec detection.

pub mod codec;
pub mod format;
pub mod properties;

pub use codec::CodecDetector;
pub use format::FormatDetector;
pub use properties::MediaProperties;
