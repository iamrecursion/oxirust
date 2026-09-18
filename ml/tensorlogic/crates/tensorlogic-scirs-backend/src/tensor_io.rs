//! Tensor binary serialization and deserialization.
//!
//! Saves and loads `ArrayD<f64>` tensors in a simple binary format:
//! `[magic(4)] [version(1)] [ndim(4)] [shape(ndim*8)] [data(nelems*8)]`
//!
//! For multi-tensor files:
//! `[count(4)] [name_len(4)][name(bytes)][tensor]...`

use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use thiserror::Error;

/// Magic bytes identifying the TensorLogic Tensor Format.
const MAGIC: &[u8; 4] = b"TLTF";

/// Current format version.
const VERSION: u8 = 1;

/// Errors that can occur during tensor I/O operations.
#[derive(Debug, Error)]
pub enum TensorIoError {
    /// An underlying I/O error occurred.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The file does not start with the expected magic bytes.
    #[error("Invalid magic bytes")]
    InvalidMagic,

    /// The file version is not supported by this implementation.
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u8),

    /// The number of elements implied by the shape does not match the data.
    #[error("Shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
}

/// Header metadata for a serialized tensor.
#[derive(Debug, Clone)]
pub struct TensorHeader {
    /// Number of dimensions.
    pub ndim: usize,
    /// Shape of each dimension.
    pub shape: Vec<usize>,
    /// Total number of elements (product of shape).
    pub element_count: usize,
    /// Size of the data section in bytes (`element_count * 8`).
    pub size_bytes: usize,
}

impl TensorHeader {
    /// Create a header from an existing tensor.
    pub fn from_tensor(tensor: &ArrayD<f64>) -> Self {
        let shape: Vec<usize> = tensor.shape().to_vec();
        let element_count = tensor.len();
        Self {
            ndim: shape.len(),
            shape,
            element_count,
            size_bytes: element_count * 8,
        }
    }
}

/// Save a tensor to a binary file at the given path.
pub fn save_tensor(path: &Path, tensor: &ArrayD<f64>) -> Result<(), TensorIoError> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_tensor(&mut writer, tensor)?;
    writer.flush()?;
    Ok(())
}

/// Load a tensor from a binary file at the given path.
pub fn load_tensor(path: &Path) -> Result<ArrayD<f64>, TensorIoError> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    read_tensor(&mut reader)
}

/// Write a tensor to any [`Write`] implementation.
pub fn write_tensor<W: Write>(writer: &mut W, tensor: &ArrayD<f64>) -> Result<(), TensorIoError> {
    // Magic
    writer.write_all(MAGIC)?;
    // Version
    writer.write_all(&[VERSION])?;

    let shape = tensor.shape();
    let ndim = shape.len() as u32;
    // ndim as little-endian u32
    writer.write_all(&ndim.to_le_bytes())?;

    // shape: each dimension as little-endian u64
    for &dim in shape {
        writer.write_all(&(dim as u64).to_le_bytes())?;
    }

    // Data: iterate in standard (row-major) order, write each f64 as little-endian
    for &value in tensor.iter() {
        writer.write_all(&value.to_le_bytes())?;
    }

    Ok(())
}

/// Read a tensor from any [`Read`] implementation.
pub fn read_tensor<R: Read>(reader: &mut R) -> Result<ArrayD<f64>, TensorIoError> {
    let header = read_header(reader)?;

    // Read data
    let mut data = vec![0u8; header.element_count * 8];
    reader.read_exact(&mut data)?;

    let values: Vec<f64> = data
        .chunks_exact(8)
        .map(|chunk| {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(chunk);
            f64::from_le_bytes(bytes)
        })
        .collect();

    if values.len() != header.element_count {
        return Err(TensorIoError::ShapeMismatch {
            expected: header.element_count,
            got: values.len(),
        });
    }

    let tensor = ArrayD::from_shape_vec(IxDyn(&header.shape), values).map_err(|_| {
        TensorIoError::ShapeMismatch {
            expected: header.element_count,
            got: 0,
        }
    })?;

    Ok(tensor)
}

/// Read just the header from a reader without consuming the data section.
pub fn read_header<R: Read>(reader: &mut R) -> Result<TensorHeader, TensorIoError> {
    // Magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(TensorIoError::InvalidMagic);
    }

    // Version
    let mut ver = [0u8; 1];
    reader.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(TensorIoError::UnsupportedVersion(ver[0]));
    }

    // ndim
    let mut ndim_bytes = [0u8; 4];
    reader.read_exact(&mut ndim_bytes)?;
    let ndim = u32::from_le_bytes(ndim_bytes) as usize;

    // shape
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let mut dim_bytes = [0u8; 8];
        reader.read_exact(&mut dim_bytes)?;
        shape.push(u64::from_le_bytes(dim_bytes) as usize);
    }

    let element_count = shape.iter().copied().product::<usize>().max(1);
    // For 0-d tensors (scalar), element_count is 1
    let element_count = if ndim == 0 { 1 } else { element_count };

    Ok(TensorHeader {
        ndim,
        shape,
        element_count,
        size_bytes: element_count * 8,
    })
}

/// Save multiple named tensors to a single binary file.
///
/// Format: `[count(4)] [name_len(4)][name(bytes)][tensor]...`
pub fn save_tensors(path: &Path, tensors: &[(&str, &ArrayD<f64>)]) -> Result<(), TensorIoError> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    let count = tensors.len() as u32;
    writer.write_all(&count.to_le_bytes())?;

    for &(name, tensor) in tensors {
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u32;
        writer.write_all(&name_len.to_le_bytes())?;
        writer.write_all(name_bytes)?;
        write_tensor(&mut writer, tensor)?;
    }

    writer.flush()?;
    Ok(())
}

/// Load all named tensors from a multi-tensor binary file.
pub fn load_tensors(path: &Path) -> Result<Vec<(String, ArrayD<f64>)>, TensorIoError> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut count_bytes = [0u8; 4];
    reader.read_exact(&mut count_bytes)?;
    let count = u32::from_le_bytes(count_bytes) as usize;

    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        // Read name
        let mut name_len_bytes = [0u8; 4];
        reader.read_exact(&mut name_len_bytes)?;
        let name_len = u32::from_le_bytes(name_len_bytes) as usize;

        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let tensor = read_tensor(&mut reader)?;
        result.push((name, tensor));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr0, Array, Array1, Array2};
    use std::io::Cursor;

    /// Helper to create a unique temp file path.
    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("tensorlogic_test_{name}_{}", std::process::id()))
    }

    #[test]
    fn test_header_from_tensor() {
        let tensor = Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|x| x as f64).collect())
            .expect("failed to create tensor");
        let header = TensorHeader::from_tensor(&tensor);
        assert_eq!(header.ndim, 3);
        assert_eq!(header.shape, vec![2, 3, 4]);
        assert_eq!(header.element_count, 24);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let tensor = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("failed to create tensor");
        let path = temp_path("roundtrip.tltf");
        save_tensor(&path, &tensor).expect("save failed");
        let loaded = load_tensor(&path).expect("load failed");
        assert_eq!(tensor, loaded);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_scalar() {
        let tensor = arr0(42.5).into_dyn();
        let path = temp_path("scalar.tltf");
        save_tensor(&path, &tensor).expect("save failed");
        let loaded = load_tensor(&path).expect("load failed");
        assert_eq!(tensor, loaded);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_1d() {
        let tensor = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]).into_dyn();
        let path = temp_path("1d.tltf");
        save_tensor(&path, &tensor).expect("save failed");
        let loaded = load_tensor(&path).expect("load failed");
        assert_eq!(tensor, loaded);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_2d() {
        let tensor = Array2::from_shape_vec((3, 4), (0..12).map(|x| x as f64).collect())
            .expect("failed to create tensor")
            .into_dyn();
        let path = temp_path("2d.tltf");
        save_tensor(&path, &tensor).expect("save failed");
        let loaded = load_tensor(&path).expect("load failed");
        assert_eq!(tensor, loaded);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_3d() {
        let tensor = Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|x| x as f64).collect())
            .expect("failed to create tensor");
        let path = temp_path("3d.tltf");
        save_tensor(&path, &tensor).expect("save failed");
        let loaded = load_tensor(&path).expect("load failed");
        assert_eq!(tensor, loaded);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_large() {
        let data: Vec<f64> = (0..10_000).map(|x| x as f64 * 0.001).collect();
        let tensor =
            Array::from_shape_vec(IxDyn(&[100, 100]), data).expect("failed to create tensor");
        let path = temp_path("large.tltf");
        save_tensor(&path, &tensor).expect("save failed");
        let loaded = load_tensor(&path).expect("load failed");
        assert_eq!(tensor, loaded);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_read_in_memory() {
        let tensor = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0])
            .expect("failed to create tensor");
        let mut buf = Vec::new();
        write_tensor(&mut buf, &tensor).expect("write failed");
        let mut cursor = Cursor::new(&buf);
        let loaded = read_tensor(&mut cursor).expect("read failed");
        assert_eq!(tensor, loaded);
    }

    #[test]
    fn test_read_invalid_magic() {
        let data = b"BADMxxxxxxxx";
        let mut cursor = Cursor::new(data.as_slice());
        let result = read_tensor(&mut cursor);
        assert!(result.is_err());
        match result {
            Err(TensorIoError::InvalidMagic) => {}
            other => panic!("Expected InvalidMagic, got {other:?}"),
        }
    }

    #[test]
    fn test_read_header_only() {
        let tensor = Array::from_shape_vec(IxDyn(&[3, 5]), (0..15).map(|x| x as f64).collect())
            .expect("failed to create tensor");
        let mut buf = Vec::new();
        write_tensor(&mut buf, &tensor).expect("write failed");
        let mut cursor = Cursor::new(&buf);
        let header = read_header(&mut cursor).expect("header read failed");
        assert_eq!(header.ndim, 2);
        assert_eq!(header.shape, vec![3, 5]);
        assert_eq!(header.element_count, 15);
    }

    #[test]
    fn test_save_load_tensors_multi() {
        let t1 = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();
        let t2 = Array2::from_shape_vec((2, 2), vec![4.0, 5.0, 6.0, 7.0])
            .expect("failed to create tensor")
            .into_dyn();
        let t3 = arr0(99.0).into_dyn();

        let path = temp_path("multi.tltf");
        save_tensors(&path, &[("alpha", &t1), ("beta", &t2), ("gamma", &t3)]).expect("save failed");
        let loaded = load_tensors(&path).expect("load failed");
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].0, "alpha");
        assert_eq!(loaded[0].1, t1);
        assert_eq!(loaded[1].0, "beta");
        assert_eq!(loaded[1].1, t2);
        assert_eq!(loaded[2].0, "gamma");
        assert_eq!(loaded[2].1, t3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_tensors_empty_list() {
        let path = temp_path("empty_multi.tltf");
        save_tensors(&path, &[]).expect("save failed");
        let loaded = load_tensors(&path).expect("load failed");
        assert!(loaded.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_tensors_names_preserved() {
        let t = Array1::from(vec![1.0]).into_dyn();
        let names = ["weights", "bias", "running_mean"];
        let tensors: Vec<(&str, &ArrayD<f64>)> = names.iter().map(|n| (*n, &t)).collect();
        let path = temp_path("names.tltf");
        save_tensors(&path, &tensors).expect("save failed");
        let loaded = load_tensors(&path).expect("load failed");
        let loaded_names: Vec<&str> = loaded.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(loaded_names, names.to_vec());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_tensor_io_error_display() {
        let e1 = TensorIoError::InvalidMagic;
        assert!(!format!("{e1}").is_empty());

        let e2 = TensorIoError::UnsupportedVersion(99);
        assert!(format!("{e2}").contains("99"));

        let e3 = TensorIoError::ShapeMismatch {
            expected: 10,
            got: 5,
        };
        let msg = format!("{e3}");
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_header_size_bytes() {
        let tensor = Array::from_shape_vec(IxDyn(&[4, 5]), (0..20).map(|x| x as f64).collect())
            .expect("failed to create tensor");
        let header = TensorHeader::from_tensor(&tensor);
        assert_eq!(header.size_bytes, header.element_count * 8);
        assert_eq!(header.size_bytes, 160);
    }

    #[test]
    fn test_save_load_negative_values() {
        let tensor = Array::from_shape_vec(IxDyn(&[4]), vec![-1.0, -100.5, -0.0, -f64::MAX])
            .expect("failed to create tensor");
        let path = temp_path("negative.tltf");
        save_tensor(&path, &tensor).expect("save failed");
        let loaded = load_tensor(&path).expect("load failed");
        assert_eq!(tensor, loaded);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_special_values() {
        let tensor = Array::from_shape_vec(
            IxDyn(&[4]),
            vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0],
        )
        .expect("failed to create tensor");
        let path = temp_path("special.tltf");
        save_tensor(&path, &tensor).expect("save failed");
        let loaded = load_tensor(&path).expect("load failed");
        // NaN != NaN, so compare bitwise
        for (orig, load) in tensor.iter().zip(loaded.iter()) {
            assert_eq!(orig.to_bits(), load.to_bits());
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_nonexistent_dir() {
        let path = std::path::PathBuf::from("/nonexistent_dir_xyz/tensor.tltf");
        let tensor = arr0(1.0).into_dyn();
        let result = save_tensor(&path, &tensor);
        assert!(result.is_err());
    }
}
