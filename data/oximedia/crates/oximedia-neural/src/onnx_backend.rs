//! ONNX inference backend powered by [`oxionnx`].
//!
//! When the `onnx` Cargo feature is enabled, this module exposes [`OnnxBackend`],
//! a struct that wraps an [`oxionnx::Session`] to provide full-graph ONNX inference
//! using the pure-Rust oxionnx engine.
//!
//! # Example
//!
//! ```rust,no_run
//! use std::collections::HashMap;
//! use std::path::Path;
//! use oximedia_neural::OnnxBackend;
//! use oximedia_neural::tensor::Tensor;
//!
//! let backend = OnnxBackend::load(Path::new("model.onnx")).unwrap();
//! let mut inputs = HashMap::new();
//! inputs.insert("input".to_string(), Tensor::from_data(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
//! let outputs = backend.run(&inputs).unwrap();
//! ```

use std::collections::HashMap;
use std::path::Path;

use crate::error::NeuralError;
use crate::tensor::Tensor;

/// Full-graph ONNX inference backend wrapping an [`oxionnx::Session`].
///
/// `OnnxBackend` provides actual neural network inference (not just model
/// introspection) by delegating execution to the pure-Rust oxionnx engine.
/// It bridges between oximedia's [`Tensor`] type and oxionnx's tensor format.
///
/// # Thread safety
///
/// `OnnxBackend` is `Send + Sync` as long as the underlying `oxionnx::Session`
/// is, which it is by default.
pub struct OnnxBackend {
    session: oxionnx::Session,
}

impl OnnxBackend {
    /// Load an ONNX model from a file path and create a new `OnnxBackend`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Io`] if the file cannot be read or the ONNX
    /// protobuf is malformed.
    pub fn load(path: &Path) -> Result<Self, NeuralError> {
        let session = oxionnx::Session::from_file(path)
            .map_err(|e| NeuralError::Io(format!("oxionnx session load failed: {e}")))?;
        Ok(Self { session })
    }

    /// Load an ONNX model from raw protobuf bytes and create a new `OnnxBackend`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Io`] if the bytes cannot be parsed as a valid ONNX
    /// model protobuf.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, NeuralError> {
        let session = oxionnx::Session::from_bytes(bytes)
            .map_err(|e| NeuralError::Io(format!("oxionnx session load failed: {e}")))?;
        Ok(Self { session })
    }

    /// Run inference on the loaded ONNX model.
    ///
    /// Inputs are provided as a map from input tensor name to [`Tensor`] value.
    /// The return value maps output tensor names to their computed [`Tensor`] values.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Io`] if inference fails (e.g., shape mismatch,
    /// unsupported operator, or missing input tensor).
    pub fn run(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, NeuralError> {
        // Convert oximedia Tensors → oxionnx Tensors
        let onnx_inputs: HashMap<&str, oxionnx::Tensor> = inputs
            .iter()
            .map(|(name, t)| {
                let onnx_t = oxionnx::Tensor::new(t.data().to_vec(), t.shape().to_vec());
                (name.as_str(), onnx_t)
            })
            .collect();

        // Run inference via oxionnx session
        let raw_outputs = self
            .session
            .run(&onnx_inputs)
            .map_err(|e| NeuralError::Io(format!("oxionnx inference failed: {e}")))?;

        // Convert oxionnx Tensors → oximedia Tensors
        let mut outputs = HashMap::with_capacity(raw_outputs.len());
        for (name, ot) in raw_outputs {
            let t = Tensor::from_data(ot.data, ot.shape)
                .map_err(|e| NeuralError::Io(format!("output tensor conversion failed: {e}")))?;
            outputs.insert(name, t);
        }
        Ok(outputs)
    }

    /// Return the names of the model's graph inputs.
    pub fn input_names(&self) -> &[String] {
        self.session.input_names()
    }

    /// Return the names of the model's graph outputs.
    pub fn output_names(&self) -> &[String] {
        self.session.output_names()
    }
}
