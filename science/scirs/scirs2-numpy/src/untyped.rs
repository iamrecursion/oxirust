//! Runtime-typed array: a byte buffer with a runtime dtype descriptor.
//!
//! [`UntypedArray`] stores elements without a compile-time type parameter.
//! This is useful in generic entry-point functions that need to inspect
//! the dtype of an incoming array before deciding how to process it,
//! analogous to `numpy.ndarray` with an unspecified element type.
//!
//! For the statically-typed `PyUntypedArray` wrapper around NumPy C-API
//! objects, see `untyped_array`.

use pyo3::prelude::*;

/// A runtime-typed multi-dimensional array backed by a flat byte buffer.
///
/// Elements can be read and written as `f64` regardless of the underlying
/// storage dtype — the conversion is applied automatically.
#[pyclass(name = "UntypedArray")]
pub struct UntypedArray {
    /// Raw backing buffer; length is `n_elements * itemsize`.
    data: Vec<u8>,
    /// Canonical dtype name (e.g. `"float64"`, `"int32"`).
    dtype_name: String,
    /// Logical shape.
    shape: Vec<usize>,
    /// Byte size of a single element.
    itemsize: usize,
}

#[pymethods]
impl UntypedArray {
    /// Construct a zero-filled untyped array.
    ///
    /// # Arguments
    /// * `shape`      – logical dimensions.
    /// * `dtype_name` – dtype string; one of: `"float32"`, `"f32"`, `"float64"`, `"f64"`,
    ///                  `"int32"`, `"i32"`, `"int64"`, `"i64"`, `"bool"`, `"b"`,
    ///                  `"uint8"`, `"u8"`, `"int8"`, `"i8"`.
    #[new]
    pub fn new(shape: Vec<usize>, dtype_name: String) -> PyResult<Self> {
        let itemsize = resolve_itemsize(&dtype_name)?;
        let n: usize = shape.iter().product::<usize>().max(1);
        Ok(Self {
            data: vec![0u8; n * itemsize],
            dtype_name,
            shape,
            itemsize,
        })
    }

    /// Return the canonical dtype name.
    pub fn dtype_name(&self) -> &str {
        &self.dtype_name
    }

    /// Return the byte size of a single element.
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }

    /// Return the logical shape.
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Return the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Return the total byte size of the backing buffer.
    pub fn nbytes(&self) -> usize {
        self.data.len()
    }

    /// Return the total number of elements (product of shape).
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Return `true` if the element dtype is a floating-point type.
    pub fn is_floating(&self) -> bool {
        matches!(
            self.dtype_name.as_str(),
            "float32" | "f32" | "float64" | "f64"
        )
    }

    /// Return `true` if the element dtype is an integer type.
    pub fn is_integer(&self) -> bool {
        matches!(
            self.dtype_name.as_str(),
            "int32" | "i32" | "int64" | "i64" | "int8" | "i8" | "uint8" | "u8"
        )
    }

    /// Read element at `flat_index` and cast it to `f64`.
    ///
    /// Returns `PyIndexError` if the index is out of bounds.
    pub fn read_as_f64(&self, flat_index: usize) -> PyResult<f64> {
        let offset = flat_index * self.itemsize;
        if offset + self.itemsize > self.data.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "flat_index {flat_index} is out of bounds"
            )));
        }
        let value = match self.dtype_name.as_str() {
            "float32" | "f32" => {
                let bytes: [u8; 4] = self.data[offset..offset + 4].try_into().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("slice conversion error (f32)")
                })?;
                f32::from_le_bytes(bytes) as f64
            }
            "float64" | "f64" => {
                let bytes: [u8; 8] = self.data[offset..offset + 8].try_into().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("slice conversion error (f64)")
                })?;
                f64::from_le_bytes(bytes)
            }
            "int32" | "i32" => {
                let bytes: [u8; 4] = self.data[offset..offset + 4].try_into().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("slice conversion error (i32)")
                })?;
                i32::from_le_bytes(bytes) as f64
            }
            "int64" | "i64" => {
                let bytes: [u8; 8] = self.data[offset..offset + 8].try_into().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("slice conversion error (i64)")
                })?;
                i64::from_le_bytes(bytes) as f64
            }
            "int8" | "i8" => self.data[offset] as i8 as f64,
            "uint8" | "u8" | "bool" | "b" => self.data[offset] as f64,
            _ => 0.0,
        };
        Ok(value)
    }

    /// Write `value` (as `f64`) to element at `flat_index`, casting to the array's dtype.
    ///
    /// Returns `PyIndexError` if the index is out of bounds.
    pub fn write_f64(&mut self, flat_index: usize, value: f64) -> PyResult<()> {
        let offset = flat_index * self.itemsize;
        if offset + self.itemsize > self.data.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "flat_index {flat_index} is out of bounds"
            )));
        }
        match self.dtype_name.as_str() {
            "float32" | "f32" => {
                self.data[offset..offset + 4].copy_from_slice(&(value as f32).to_le_bytes());
            }
            "float64" | "f64" => {
                self.data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            }
            "int32" | "i32" => {
                self.data[offset..offset + 4].copy_from_slice(&(value as i32).to_le_bytes());
            }
            "int64" | "i64" => {
                self.data[offset..offset + 8].copy_from_slice(&(value as i64).to_le_bytes());
            }
            "int8" | "i8" => {
                self.data[offset] = value as i8 as u8;
            }
            "uint8" | "u8" => {
                self.data[offset] = value as u8;
            }
            "bool" | "b" => {
                self.data[offset] = if value != 0.0 { 1u8 } else { 0u8 };
            }
            _ => {}
        }
        Ok(())
    }
}

/// Resolve the byte size of a dtype name string.
///
/// Returns `PyValueError` for unsupported dtype strings.
fn resolve_itemsize(dtype_name: &str) -> PyResult<usize> {
    match dtype_name {
        "float32" | "f32" => Ok(4),
        "float64" | "f64" => Ok(8),
        "int32" | "i32" => Ok(4),
        "int64" | "i64" => Ok(8),
        "bool" | "b" => Ok(1),
        "uint8" | "u8" => Ok(1),
        "int8" | "i8" => Ok(1),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported dtype '{dtype_name}'; supported: float32, f32, float64, f64, \
             int32, i32, int64, i64, bool, b, uint8, u8, int8, i8"
        ))),
    }
}

/// Register the untyped array class into a PyO3 module.
///
/// Call this from your `#[pymodule]` init function to expose [`UntypedArray`].
pub fn register_untyped_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UntypedArray>()?;
    Ok(())
}
