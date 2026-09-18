//! ML framework converters
#![allow(dead_code)]

use crate::error::Result;
use crate::ml_framework::types::{MLFramework, MLModel, MLTensor};
use std::path::Path;

pub mod coreml;
pub mod huggingface;
pub mod jax;
pub mod mxnet;
pub mod onnx;
pub mod pytorch;
pub mod safetensors;
pub mod tensorflow;

pub use coreml::CoreMLConverter;
pub use huggingface::HuggingFaceConverter;
pub use jax::JAXConverter;
pub use mxnet::MXNetConverter;
pub use onnx::ONNXConverter;
pub use pytorch::PyTorchConverter;
pub use safetensors::SafeTensorsConverter;
pub use tensorflow::TensorFlowConverter;

/// Trait for ML framework converters
pub trait MLFrameworkConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()>;
    fn load_model(&self, path: &Path) -> Result<MLModel>;
    #[allow(dead_code)]
    fn save_tensor(&self, tensor: &MLTensor, path: &Path) -> Result<()>;
    #[allow(dead_code)]
    fn load_tensor(&self, path: &Path) -> Result<MLTensor>;
}

/// Get appropriate converter for framework
pub fn get_converter(framework: MLFramework) -> Box<dyn MLFrameworkConverter> {
    match framework {
        MLFramework::PyTorch => Box::new(PyTorchConverter),
        MLFramework::ONNX => Box::new(ONNXConverter),
        MLFramework::SafeTensors => Box::new(SafeTensorsConverter),
        MLFramework::TensorFlow => Box::new(TensorFlowConverter),
        MLFramework::JAX => Box::new(JAXConverter),
        MLFramework::MXNet => Box::new(MXNetConverter),
        MLFramework::CoreML => Box::new(CoreMLConverter),
        MLFramework::HuggingFace => Box::new(HuggingFaceConverter),
    }
}
