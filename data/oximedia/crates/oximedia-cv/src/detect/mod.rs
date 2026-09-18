//! Detection algorithms.
//!
//! This module provides various detection algorithms including:
//!
//! - [`face`]: Face detection using Haar cascades and CNN models
//! - [`face_align`]: Face alignment based on landmarks
//! - [`motion`]: Motion detection and optical flow
//! - [`object`]: Generic object detection with NMS
//! - `yolo`: YOLO object detection (YOLOv5/YOLOv8)
//! - [`yolo_utils`]: Utility functions for YOLO
//! - [`corner`]: Corner detection (Harris, Shi-Tomasi, FAST)
//!
//! # Example
//!
//! ```
//! use oximedia_cv::detect::{BoundingBox, Detection};
//!
//! let bbox = BoundingBox::new(10.0, 20.0, 100.0, 150.0);
//! let detection = Detection::new(bbox, 0, 0.95);
//! assert!(detection.confidence > 0.9);
//! ```

pub mod corner;
pub mod face;
pub mod face_align;
pub mod face_multiscale;
pub mod motion;
pub mod object;
pub mod object_detect;
#[cfg(feature = "onnx")]
pub mod yolo;
pub mod yolo_utils;

// Re-export commonly used items
pub use corner::{Corner, CornerDetector, FastDetector, HarrisDetector, ShiTomasiDetector};
pub use face::{DetectionResult, FaceDetector, FaceRegion, HaarCascade, IntegralImage};
pub use motion::{MotionDetector, MotionRegion, OpticalFlowLK};
pub use object::{BoundingBox, Detection, ObjectDetector};
#[cfg(feature = "onnx")]
pub use yolo::{coco_class_names, InputResolution, YoloConfig, YoloDetector, YoloVersion};
pub use yolo_utils::{letterbox_resize, LetterboxParams};
