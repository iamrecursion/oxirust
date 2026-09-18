//! OpenCV `cv2` API compatibility layer for OxiMedia.
//!
//! Provides a pure-Rust `Mat` type (BGR channel ordering by default per OpenCV convention),
//! ~150 OpenCV constants, and a curated function surface dispatching into OxiMedia crates.
//!
//! # Quick Start
//! ```no_run
//! use oximedia_compat_cv2::{imread, imwrite, IMREAD_COLOR};
//!
//! let mat = imread("input.png", IMREAD_COLOR).expect("imread failed");
//! imwrite("output.png", &mat).expect("imwrite failed");
//! ```

pub mod constants;

/// Auto-generated, build-time reflection of [`constants`].
///
/// `build.rs` syn-parses `src/constants.rs` and emits a single static
/// table — `LIST_CONSTANTS` — listing every `pub const` declaration as a
/// `(category, name, type, value)` tuple. Use this from tools (e.g.
/// the `oximedia-cv2` CLI's `--list-constants` flag) instead of
/// hand-maintaining a parallel listing.
pub mod constants_list {
    include!(concat!(env!("OUT_DIR"), "/constants_list.rs"));
}

pub mod error;
pub mod image_io;
pub mod mat;

pub mod arithmetic;
pub mod color;
pub mod connected;
pub mod contour;
#[cfg(feature = "dnn")]
pub mod dnn;
pub mod drawing;
pub mod edge;
pub mod features;
pub mod filter;
pub mod geometry;
pub mod hershey_font;
pub mod histogram;
pub mod hough;
pub mod morphology;
pub mod optical_flow;
pub mod template;
pub mod threshold;

pub use constants::*;
pub use error::{Cv2Error, Cv2Result};
pub use hershey_font::put_text_hershey;
pub use image_io::{imdecode, imencode, imread, imwrite};
pub use mat::{Mat, MatType, Point, Point2f, Rect, Scalar, Size};
