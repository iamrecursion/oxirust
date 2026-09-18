//! Structured dtype support — record arrays with named, typed fields.
//!
//! Mirrors `numpy.dtype([("name", "f64"), ...])` and `numpy.recarray`.
//! A [`StructuredDtype`] describes the layout of a single record, and a
//! [`StructuredArray`] stores `n` such records as a flat byte buffer.

use pyo3::prelude::*;

/// Descriptor for a single named field within a structured dtype.
#[derive(Debug, Clone)]
#[pyclass(name = "DtypeField", from_py_object)]
pub struct DtypeField {
    /// Field name.
    #[pyo3(get)]
    pub name: String,
    /// Element type string, e.g. `"f64"`, `"i32"`, `"bool"`.
    #[pyo3(get)]
    pub dtype: String,
    /// Byte offset of this field within one record.
    #[pyo3(get)]
    pub offset: usize,
}

/// A structured dtype: a record type composed of multiple named fields.
///
/// Mirrors `numpy.dtype` with compound (structured) field specifications.
#[pyclass(name = "StructuredDtype")]
pub struct StructuredDtype {
    /// Ordered list of field descriptors.
    fields: Vec<DtypeField>,
    /// Total byte size of one record.
    itemsize: usize,
}

#[pymethods]
impl StructuredDtype {
    /// Build a structured dtype from a list of `(name, dtype_str)` pairs.
    ///
    /// Fields are laid out sequentially with no padding.  The first field is at
    /// byte offset 0; each subsequent field starts immediately after the previous.
    #[new]
    pub fn new(field_specs: Vec<(String, String)>) -> PyResult<Self> {
        let mut offset = 0usize;
        let mut fields = Vec::with_capacity(field_specs.len());
        for (name, dtype) in field_specs {
            let size = dtype_size(&dtype)?;
            fields.push(DtypeField {
                name,
                dtype,
                offset,
            });
            offset += size;
        }
        Ok(Self {
            fields,
            itemsize: offset,
        })
    }

    /// Return an ordered list of field names.
    pub fn names(&self) -> Vec<String> {
        self.fields.iter().map(|f| f.name.clone()).collect()
    }

    /// Return the total byte size of one record.
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }

    /// Return the byte offset of each field (in field order).
    pub fn offsets(&self) -> Vec<usize> {
        self.fields.iter().map(|f| f.offset).collect()
    }

    /// Return the number of fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

/// Return the byte size for a dtype name string.
///
/// Returns a `PyValueError` for unrecognised dtype strings.
fn dtype_size(dtype: &str) -> PyResult<usize> {
    match dtype {
        "f32" | "float32" => Ok(4),
        "f64" | "float64" => Ok(8),
        "i32" | "int32" => Ok(4),
        "i64" | "int64" => Ok(8),
        "u32" | "uint32" => Ok(4),
        "u64" | "uint64" => Ok(8),
        "bool" => Ok(1),
        "i8" | "int8" => Ok(1),
        "u8" | "uint8" => Ok(1),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown dtype '{dtype}'; supported: f32, f64, i32, i64, u32, u64, bool, i8, u8"
        ))),
    }
}

/// A structured array: a flat byte buffer interpreted as `n_records` records,
/// each described by a [`StructuredDtype`].
#[pyclass(name = "StructuredArray")]
pub struct StructuredArray {
    /// Dtype describing the layout of each record.
    dtype: StructuredDtype,
    /// Raw backing buffer; length is `n_records * dtype.itemsize`.
    data: Vec<u8>,
    /// Number of records stored.
    n_records: usize,
}

#[pymethods]
impl StructuredArray {
    /// Construct a zero-initialised structured array.
    ///
    /// # Arguments
    /// * `n_records`   – number of records to allocate.
    /// * `field_specs` – list of `(name, dtype_str)` pairs defining the record layout.
    #[new]
    pub fn new_empty(n_records: usize, field_specs: Vec<(String, String)>) -> PyResult<Self> {
        let dtype = StructuredDtype::new(field_specs)?;
        let data = vec![0u8; n_records * dtype.itemsize];
        Ok(Self {
            dtype,
            data,
            n_records,
        })
    }

    /// Return the number of records.
    pub fn n_records(&self) -> usize {
        self.n_records
    }

    /// Return the byte size of one record.
    pub fn itemsize(&self) -> usize {
        self.dtype.itemsize
    }

    /// Read all values for an `f64` field as a `Vec<f64>`.
    ///
    /// Returns `PyKeyError` if `field_name` is not found, or `PyTypeError`
    /// if the field dtype is not `"f64"` / `"float64"`.
    pub fn get_field_f64(&self, field_name: &str) -> PyResult<Vec<f64>> {
        let field = self
            .dtype
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("field '{field_name}' not found"))
            })?;
        if field.dtype != "f64" && field.dtype != "float64" {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "field '{field_name}' has dtype '{}', not f64",
                field.dtype
            )));
        }
        let mut result = Vec::with_capacity(self.n_records);
        for i in 0..self.n_records {
            let byte_offset = i * self.dtype.itemsize + field.offset;
            let bytes: [u8; 8] = self.data[byte_offset..byte_offset + 8]
                .try_into()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("slice conversion error"))?;
            result.push(f64::from_le_bytes(bytes));
        }
        Ok(result)
    }

    /// Write all values for an `f64` field from a `Vec<f64>`.
    ///
    /// Returns `PyValueError` if `values.len() != n_records`, `PyKeyError` if the
    /// field is not found, or `PyTypeError` if the field dtype is not `"f64"` / `"float64"`.
    pub fn set_field_f64(&mut self, field_name: &str, values: Vec<f64>) -> PyResult<()> {
        if values.len() != self.n_records {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "values length {} does not match n_records {}",
                values.len(),
                self.n_records
            )));
        }
        let field = self
            .dtype
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("field '{field_name}' not found"))
            })?;
        if field.dtype != "f64" && field.dtype != "float64" {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "field '{field_name}' has dtype '{}', not f64",
                field.dtype
            )));
        }
        let field_offset = field.offset;
        let itemsize = self.dtype.itemsize;
        for (i, &v) in values.iter().enumerate() {
            let byte_offset = i * itemsize + field_offset;
            self.data[byte_offset..byte_offset + 8].copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }

    /// Return the names of all fields.
    pub fn field_names(&self) -> Vec<String> {
        self.dtype.names()
    }
}

/// Register structured dtype classes into a PyO3 module.
///
/// Call this from your `#[pymodule]` init function to expose `DtypeField`,
/// `StructuredDtype`, and `StructuredArray`.
pub fn register_structured_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DtypeField>()?;
    m.add_class::<StructuredDtype>()?;
    m.add_class::<StructuredArray>()?;
    Ok(())
}
