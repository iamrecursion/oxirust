//! Machine learning module for ONNX Runtime integration.
//!
//! This module provides infrastructure for running ONNX models including:
//! - ONNX Runtime session management
//! - Tensor operations and conversions
//! - Image preprocessing for ML models
//! - Post-processing utilities (NMS, softmax, etc.)
//!
//! # Example
//!
//! ```no_run
//! use oximedia_cv::ml::{OnnxRuntime, Tensor};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create runtime and load model
//! let runtime = OnnxRuntime::new()?;
//! let session = runtime.load_model("model.onnx")?;
//!
//! // Run inference
//! // let input_tensor = Tensor::zeros(&[1, 3, 224, 224]);
//! // let outputs = session.run(&[input_tensor])?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::unused_self)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::too_many_arguments)]

pub mod postprocessing;
pub mod preprocessing;
pub mod runtime;
pub mod tensor;

// Re-export commonly used items
pub use postprocessing::{
    confidence_threshold, decode_yolo_boxes, nms, sigmoid, soft_nms, softmax,
};
pub use preprocessing::{
    normalize, normalize_imagenet, pad_to_size, resize_to_fit, ImagePreprocessor,
};
pub use runtime::{DeviceType, OnnxRuntime, Session};
pub use tensor::{DataLayout, Tensor, TensorData};
