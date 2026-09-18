//! Error handling for Python bindings.

use pyo3::{create_exception, exceptions::PyException, prelude::*};

create_exception!(oximedia, OxiMediaError, PyException);

/// Convert `OxiMedia` core error to Python exception.
pub fn from_oxi_error(err: oximedia_core::OxiError) -> PyErr {
    OxiMediaError::new_err(err.to_string())
}

/// Convert codec error to Python exception.
pub fn from_codec_error(err: oximedia_codec::CodecError) -> PyErr {
    OxiMediaError::new_err(err.to_string())
}

/// Convert container error to Python exception.
pub fn from_container_error(err: &str) -> PyErr {
    OxiMediaError::new_err(err.to_string())
}

/// Convert audio error to Python exception.
pub fn from_audio_error(err: oximedia_audio::AudioError) -> PyErr {
    OxiMediaError::new_err(err.to_string())
}

/// Result type alias for Python operations.
pub type PyOxiResult<T> = Result<T, PyErr>;
