// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Audio extraction and track selection.

pub mod extract;
pub mod track;

pub use extract::{AudioExtractor, AudioFormat};
pub use track::{AudioTrackInfo, AudioTrackSelector};
