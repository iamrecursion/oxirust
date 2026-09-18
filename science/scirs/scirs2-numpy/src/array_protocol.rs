//! NumPy array protocol (`__array__` and `__array_interface__`) support.
//!
//! The NumPy array protocol enables Python objects to be converted to NumPy
//! arrays via two mechanisms:
//!
//! - `__array__(dtype=None)` — a method that returns a `numpy.ndarray`.
//! - `__array_interface__` — a property returning a Python dict describing
//!   the underlying buffer (shape, dtype string, raw pointer, etc.).
//!
//! The array interface dictionary format follows:
//! <https://numpy.org/doc/stable/reference/arrays.interface.html>

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use thiserror::Error;

// ─── Error types ────────────────────────────────────────────────────────────

/// Errors produced by the array-protocol layer.
#[derive(Debug, Error)]
pub enum ArrayProtocolError {
    /// The requested element dtype is not supported.
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),

    /// The numpy typestr string could not be parsed.
    #[error("invalid typestr: {0}")]
    InvalidTypestr(String),

    /// A Python API call failed.
    #[error("python error: {0}")]
    PythonError(String),
}

impl From<PyErr> for ArrayProtocolError {
    fn from(e: PyErr) -> Self {
        Self::PythonError(e.to_string())
    }
}

impl From<ArrayProtocolError> for PyErr {
    fn from(e: ArrayProtocolError) -> Self {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    }
}

// ─── parse_typestr ───────────────────────────────────────────────────────────

/// Parse a NumPy type-string into `(kind_char, byte_count)`.
///
/// NumPy typestrings have the format `<endian><kind><bytes>`, where:
/// - endianness: `'<'` (little), `'>'` (big), `'='` (native), `'|'` (n/a)
/// - kind: `'f'` float, `'i'` signed int, `'u'` unsigned int, `'b'` bool,
///   `'c'` complex, etc.
/// - bytes: decimal byte count, e.g. `8` for 64-bit.
///
/// Returns `(kind, byte_count)`.
///
/// # Examples
///
/// ```
/// use scirs2_numpy::array_protocol::parse_typestr;
/// let (kind, bytes) = parse_typestr("<f8").unwrap();
/// assert_eq!(kind, 'f');
/// assert_eq!(bytes, 8);
/// ```
pub fn parse_typestr(typestr: &str) -> Result<(char, usize), ArrayProtocolError> {
    if typestr.len() < 3 {
        return Err(ArrayProtocolError::InvalidTypestr(format!(
            "too short: {typestr:?}"
        )));
    }
    let mut chars = typestr.chars();
    let endian = chars
        .next()
        .ok_or_else(|| ArrayProtocolError::InvalidTypestr(format!("empty typestr: {typestr:?}")))?;
    // Validate endianness character.
    if !matches!(endian, '<' | '>' | '=' | '|') {
        return Err(ArrayProtocolError::InvalidTypestr(format!(
            "unknown endianness character {endian:?} in {typestr:?}"
        )));
    }
    let kind = chars.next().ok_or_else(|| {
        ArrayProtocolError::InvalidTypestr(format!("missing kind in {typestr:?}"))
    })?;
    if !kind.is_ascii_alphabetic() {
        return Err(ArrayProtocolError::InvalidTypestr(format!(
            "invalid kind character {kind:?} in {typestr:?}"
        )));
    }
    let size_str: String = chars.collect();
    let byte_count = size_str.parse::<usize>().map_err(|_| {
        ArrayProtocolError::InvalidTypestr(format!(
            "invalid byte count {size_str:?} in {typestr:?}"
        ))
    })?;
    if byte_count == 0 {
        return Err(ArrayProtocolError::InvalidTypestr(format!(
            "byte count must be > 0 in {typestr:?}"
        )));
    }
    Ok((kind, byte_count))
}

// ─── ArrayProtocol trait ────────────────────────────────────────────────────

/// Mixin trait for types that support the NumPy array interface protocol.
///
/// Implementors must provide shape, stride, type-string, and a raw data
/// pointer, from which a complete [`ArrayInterfaceDict`] can be assembled.
pub trait ArrayProtocol {
    /// Returns the populated [`ArrayInterfaceDict`] for this object.
    fn array_interface(&self) -> ArrayInterfaceDict;

    /// Returns the NumPy dtype type-string (e.g. `"<f8"` for little-endian f64).
    fn dtype_str(&self) -> &'static str;

    /// Returns the logical shape of the array.
    fn shape(&self) -> Vec<usize>;

    /// Returns the strides of the array **in bytes**.
    fn strides(&self) -> Vec<usize>;

    /// Returns a raw pointer to the first byte of element data.
    fn data_ptr(&self) -> *const u8;

    /// Returns the total number of bytes occupied by the element buffer.
    fn nbytes(&self) -> usize;
}

// ─── ArrayInterfaceDict ──────────────────────────────────────────────────────

/// Data for the `__array_interface__` protocol dictionary.
///
/// See: <https://numpy.org/doc/stable/reference/arrays.interface.html>
pub struct ArrayInterfaceDict {
    /// Logical shape of the array.
    pub shape: Vec<usize>,
    /// NumPy dtype typestr (e.g. `"<f8"`).
    pub typestr: String,
    /// Raw pointer to element data, encoded as a Python integer.
    pub data_ptr: usize,
    /// Whether the buffer should be treated as read-only.
    pub readonly: bool,
    /// Optional per-dimension strides in bytes.
    pub strides: Option<Vec<usize>>,
    /// Protocol version; always 3.
    pub version: u8,
}

impl ArrayInterfaceDict {
    /// Serialize this descriptor into a Python dict suitable for `__array_interface__`.
    ///
    /// The resulting dict has the keys `shape`, `typestr`, `data`, `version`,
    /// and optionally `strides`.
    pub fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        // shape — tuple of usize
        let shape_tuple = PyTuple::new(py, self.shape.iter().copied())?;
        dict.set_item("shape", shape_tuple)?;

        // typestr — str
        dict.set_item("typestr", &self.typestr)?;

        // data — (ptr_as_int, readonly_bool)
        let data_tuple = PyTuple::new(py, [self.data_ptr, self.readonly as usize])?;
        dict.set_item("data", data_tuple)?;

        // version — always 3
        dict.set_item("version", self.version)?;

        // strides — optional tuple of usize
        if let Some(ref strides) = self.strides {
            let strides_tuple = PyTuple::new(py, strides.iter().copied())?;
            dict.set_item("strides", strides_tuple)?;
        }

        Ok(dict)
    }
}

// ─── NdArrayWrapper ──────────────────────────────────────────────────────────

/// A concrete array type implementing the NumPy `__array__` and
/// `__array_interface__` protocols.
///
/// Wraps an owned flat `Vec<f64>` buffer with a logical shape, and exposes
/// it to NumPy via the array interface protocol.
#[pyclass(name = "NdArrayWrapper")]
pub struct NdArrayWrapper {
    /// Flat element buffer in C (row-major) order.
    data: Vec<f64>,
    /// Logical shape.
    shape: Vec<usize>,
    /// Per-dimension strides **in bytes** (C-contiguous by default).
    strides: Vec<usize>,
    /// NumPy dtype typestr.
    dtype: String,
}

#[pymethods]
impl NdArrayWrapper {
    /// Construct a new `NdArrayWrapper` with C-contiguous strides.
    ///
    /// # Arguments
    /// * `data`  – flat element buffer; must have `shape.iter().product::<usize>()` elements.
    /// * `shape` – logical dimensions.
    #[new]
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> PyResult<Self> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "data length {} does not match shape product {}",
                data.len(),
                n
            )));
        }
        let strides = compute_c_strides_bytes(&shape, std::mem::size_of::<f64>());
        Ok(Self {
            data,
            shape,
            strides,
            dtype: "<f8".to_owned(),
        })
    }

    /// Return a Python representation suitable for numpy consumption.
    ///
    /// Calls `numpy.array(list_of_floats).reshape(shape)` so that consumers
    /// that call `np.asarray(obj)` or `np.array(obj.__array__())` obtain the
    /// correct array.
    ///
    /// Note: requires NumPy to be installed in the active Python environment.
    #[pyo3(name = "__array__")]
    pub fn array_method(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let np = py.import("numpy").map_err(|e| {
            pyo3::exceptions::PyImportError::new_err(format!("numpy not available: {e}"))
        })?;
        // Build a flat Python list from the data buffer.
        let flat_list = PyList::new(py, &self.data)?;
        // numpy.array(flat_list, dtype='f8')
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "f8")?;
        let arr = np.call_method("array", (flat_list,), Some(&kwargs))?;
        // Reshape to logical shape.
        let shape_tuple = PyTuple::new(py, self.shape.iter().copied())?;
        let reshaped = arr.call_method1("reshape", (shape_tuple,))?;
        Ok(reshaped.unbind())
    }

    /// The `__array_interface__` property, returning a dict describing the buffer.
    #[getter]
    pub fn array_interface(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let desc = ArrayInterfaceDict {
            shape: self.shape.clone(),
            typestr: self.dtype.clone(),
            data_ptr: self.data.as_ptr() as usize,
            readonly: true,
            strides: Some(self.strides.clone()),
            version: 3,
        };
        let dict = desc.to_py_dict(py)?;
        Ok(dict.into_any().unbind())
    }

    /// Return the shape as a Python tuple.
    pub fn shape_tuple(&self, py: Python<'_>) -> Py<PyAny> {
        PyTuple::new(py, self.shape.iter().copied())
            .map(|t| t.into_any().unbind())
            .unwrap_or_else(|_| py.None())
    }

    /// Return the dtype typestr (e.g. `"<f8"`).
    pub fn dtype_str(&self) -> &str {
        &self.dtype
    }

    /// Return a flat copy of the data buffer.
    pub fn data(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Return the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

impl ArrayProtocol for NdArrayWrapper {
    fn array_interface(&self) -> ArrayInterfaceDict {
        ArrayInterfaceDict {
            shape: self.shape.clone(),
            typestr: self.dtype.clone(),
            data_ptr: self.data.as_ptr() as usize,
            readonly: true,
            strides: Some(self.strides.clone()),
            version: 3,
        }
    }

    fn dtype_str(&self) -> &'static str {
        "<f8"
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn strides(&self) -> Vec<usize> {
        self.strides.clone()
    }

    fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr() as *const u8
    }

    fn nbytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f64>()
    }
}

// ─── Register ───────────────────────────────────────────────────────────────

/// Register array-protocol classes into a PyO3 module.
pub fn register_array_protocol_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NdArrayWrapper>()?;
    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Compute C-contiguous (row-major) strides in bytes for a given shape.
///
/// The last dimension has stride `elem_size`; each preceding dimension has stride
/// equal to the product of all following dimensions multiplied by `elem_size`.
fn compute_c_strides_bytes(shape: &[usize], elem_size: usize) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return Vec::new();
    }
    let mut strides = vec![elem_size; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_typestr ---

    #[test]
    fn test_parse_typestr_f64_le() {
        let (kind, bytes) = parse_typestr("<f8").expect("parse_typestr failed");
        assert_eq!(kind, 'f');
        assert_eq!(bytes, 8);
    }

    #[test]
    fn test_parse_typestr_i32_be() {
        let (kind, bytes) = parse_typestr(">i4").expect("parse_typestr failed");
        assert_eq!(kind, 'i');
        assert_eq!(bytes, 4);
    }

    #[test]
    fn test_parse_typestr_u16_native() {
        let (kind, bytes) = parse_typestr("=u2").expect("parse_typestr failed");
        assert_eq!(kind, 'u');
        assert_eq!(bytes, 2);
    }

    #[test]
    fn test_parse_typestr_bool_noendian() {
        let (kind, bytes) = parse_typestr("|b1").expect("parse_typestr failed");
        assert_eq!(kind, 'b');
        assert_eq!(bytes, 1);
    }

    #[test]
    fn test_parse_typestr_error_too_short() {
        assert!(parse_typestr("<f").is_err());
        assert!(parse_typestr("").is_err());
        assert!(parse_typestr("<").is_err());
    }

    #[test]
    fn test_parse_typestr_error_bad_endian() {
        assert!(parse_typestr("?f8").is_err());
    }

    #[test]
    fn test_parse_typestr_error_zero_bytes() {
        assert!(parse_typestr("<f0").is_err());
    }

    // --- ArrayInterfaceDict ---

    #[test]
    fn test_array_interface_dict_version() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let wrapper = NdArrayWrapper::new(data, vec![2, 2]).expect("NdArrayWrapper::new failed");
        let iface = ArrayProtocol::array_interface(&wrapper);
        assert_eq!(iface.version, 3, "version must be 3");
    }

    #[test]
    fn test_array_interface_dict_shape() {
        let data = vec![1.0_f64; 6];
        let wrapper = NdArrayWrapper::new(data, vec![2, 3]).expect("NdArrayWrapper::new failed");
        let iface = ArrayProtocol::array_interface(&wrapper);
        assert_eq!(iface.shape, vec![2, 3]);
    }

    #[test]
    fn test_array_interface_dict_typestr() {
        let data = vec![0.0_f64; 4];
        let wrapper = NdArrayWrapper::new(data, vec![4]).expect("NdArrayWrapper::new failed");
        let iface = ArrayProtocol::array_interface(&wrapper);
        assert_eq!(iface.typestr, "<f8");
    }

    #[test]
    fn test_array_interface_dict_data_ptr_nonzero() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let wrapper = NdArrayWrapper::new(data, vec![3]).expect("NdArrayWrapper::new failed");
        let iface = ArrayProtocol::array_interface(&wrapper);
        assert_ne!(iface.data_ptr, 0, "data pointer must be non-null");
    }

    // --- NdArrayWrapper construction ---

    #[test]
    fn test_ndarray_wrapper_shape_mismatch() {
        // data has 4 elements but shape says 6
        let result = NdArrayWrapper::new(vec![1.0; 4], vec![2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ndarray_wrapper_scalar() {
        // 0-d equivalent: shape = [1]
        let wrapper = NdArrayWrapper::new(vec![42.0], vec![1]).expect("scalar failed");
        assert_eq!(wrapper.ndim(), 1);
        assert_eq!(wrapper.data(), vec![42.0]);
    }

    #[test]
    fn test_ndarray_wrapper_strides_c_order() {
        // shape [3, 4] → strides [32, 8] (in bytes, f64=8)
        let data = vec![0.0_f64; 12];
        let wrapper = NdArrayWrapper::new(data, vec![3, 4]).expect("NdArrayWrapper::new failed");
        let strides = ArrayProtocol::strides(&wrapper);
        assert_eq!(strides, vec![32, 8]);
    }

    // --- Python-GIL tests (require auto-initialize feature) ---

    #[test]
    fn test_array_interface_py_dict_keys() {
        Python::attach(|py| {
            let data = vec![1.0_f64, 2.0, 3.0, 4.0];
            let wrapper =
                NdArrayWrapper::new(data, vec![2, 2]).expect("NdArrayWrapper::new failed");
            let iface = ArrayProtocol::array_interface(&wrapper);
            let dict = iface.to_py_dict(py).expect("to_py_dict failed");

            assert!(dict
                .get_item("shape")
                .expect("shape lookup failed")
                .is_some());
            assert!(dict
                .get_item("typestr")
                .expect("typestr lookup failed")
                .is_some());
            assert!(dict.get_item("data").expect("data lookup failed").is_some());
            assert!(dict
                .get_item("version")
                .expect("version lookup failed")
                .is_some());
        });
    }

    #[test]
    fn test_array_interface_py_dict_shape_values() {
        Python::attach(|py| {
            let data = vec![0.0_f64; 6];
            let wrapper =
                NdArrayWrapper::new(data, vec![2, 3]).expect("NdArrayWrapper::new failed");
            let iface = ArrayProtocol::array_interface(&wrapper);
            let dict = iface.to_py_dict(py).expect("to_py_dict failed");

            let shape_obj = dict
                .get_item("shape")
                .expect("shape lookup failed")
                .expect("shape missing");
            let shape_tuple = shape_obj.cast::<PyTuple>().expect("shape is not a tuple");
            assert_eq!(shape_tuple.len(), 2);
            let v0: usize = shape_tuple
                .get_item(0)
                .expect("item 0")
                .extract()
                .expect("extract[0]");
            let v1: usize = shape_tuple
                .get_item(1)
                .expect("item 1")
                .extract()
                .expect("extract[1]");
            assert_eq!(v0, 2);
            assert_eq!(v1, 3);
        });
    }
}
