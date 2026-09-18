// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! File splitting by time, size, or chapters.

pub mod chapter;
pub mod size;
pub mod time;

pub use chapter::ChapterSplitter;
pub use size::SizeSplitter;
pub use time::TimeSplitter;
