//! Chapter marker support.
//!
//! Provides chapter support for Matroska and MP4 containers.

#![forbid(unsafe_code)]

pub mod generator;
pub mod matroska;
pub mod mp4;

pub use generator::{ChapterGenerator, ChapterGeneratorConfig};
pub use matroska::{MatroskaChapter, MatroskaChapters, MatroskaChaptersBuilder, MatroskaEdition};
pub use mp4::{Mp4Chapter, Mp4ChapterTrack};
