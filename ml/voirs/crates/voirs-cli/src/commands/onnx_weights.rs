//! Pure-Rust extraction of ONNX weights for `voirs convert-model`.
//!
//! Reads the ONNX protobuf with `oxionnx-proto` and copies every constant
//! tensor -- graph initializers and the `value` tensor of `Constant` nodes --
//! into SafeTensors *with its original element type*. `oxionnx_core::Tensor`
//! is f32-only, so the `TensorProto`s are decoded directly here instead:
//!
//! - `raw_data` (little-endian, tightly packed) is copied verbatim;
//! - the typed fields (`float_data`, `int32_data`, `int64_data`,
//!   `double_data`, `uint64_data`) are re-encoded little-endian, honouring
//!   the ONNX rule that 8/16-bit integers, `BOOL`, `FLOAT16` and `BFLOAT16`
//!   are stored bit-wise inside `int32_data`;
//! - external data (`data_location = EXTERNAL`) is read from the sidecar file
//!   named by `location` (resolved below the model directory with path
//!   traversal rejected), honouring `offset` / `length`.
//!
//! Element types SafeTensors cannot represent (`STRING`, complex, `FLOAT8*`,
//! 4-bit) are reported as skipped rather than silently dropped; malformed
//! data (wrong byte count, missing external file) is an error.

use oxionnx_proto::{dtype_code, external_entry, is_external, resolve_external_path, TensorProto};
use safetensors::tensor::Dtype;
use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// One constant tensor, ready to be written to SafeTensors.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedTensor {
    /// ONNX tensor name (initializer name or `Constant` output name).
    pub name: String,
    /// Element type, identical to the ONNX `data_type`.
    pub dtype: Dtype,
    /// Tensor shape (`[]` for scalars).
    pub shape: Vec<usize>,
    /// Little-endian, tightly packed element bytes.
    pub data: Vec<u8>,
}

/// A constant tensor that was not converted, with the reason.
#[derive(Debug, Clone, PartialEq)]
pub struct SkippedTensor {
    /// ONNX tensor name.
    pub name: String,
    /// Why it was skipped (e.g. unsupported element type).
    pub reason: String,
}

/// Everything `convert-model` needs from an ONNX file.
#[derive(Debug, Default)]
pub struct OnnxWeights {
    /// Converted tensors, in graph order (initializers first).
    pub tensors: Vec<ExtractedTensor>,
    /// Constant tensors with an element type SafeTensors cannot hold.
    pub skipped: Vec<SkippedTensor>,
    /// Number of graph nodes.
    pub node_count: usize,
    /// Number of real graph inputs (initializers listed as inputs excluded).
    pub input_count: usize,
    /// Number of graph outputs.
    pub output_count: usize,
}

/// SafeTensors dtype and element width for an ONNX `data_type`, or `None` if
/// SafeTensors cannot represent it.
fn safetensors_dtype(data_type: i32) -> Option<(Dtype, usize)> {
    Some(match data_type {
        dtype_code::FLOAT32 => (Dtype::F32, 4),
        dtype_code::UINT8 => (Dtype::U8, 1),
        dtype_code::INT8 => (Dtype::I8, 1),
        dtype_code::UINT16 => (Dtype::U16, 2),
        dtype_code::INT16 => (Dtype::I16, 2),
        dtype_code::INT32 => (Dtype::I32, 4),
        dtype_code::INT64 => (Dtype::I64, 8),
        dtype_code::BOOL => (Dtype::BOOL, 1),
        dtype_code::FLOAT16 => (Dtype::F16, 2),
        dtype_code::DOUBLE => (Dtype::F64, 8),
        dtype_code::UINT32 => (Dtype::U32, 4),
        dtype_code::UINT64 => (Dtype::U64, 8),
        dtype_code::BFLOAT16 => (Dtype::BF16, 2),
        _ => return None,
    })
}

/// Validated shape and element count of a tensor.
fn shape_of(tensor: &TensorProto) -> Result<(Vec<usize>, usize), String> {
    let mut shape = Vec::with_capacity(tensor.dims.len());
    let mut count = 1_usize;
    for &dim in &tensor.dims {
        let dim = usize::try_from(dim)
            .map_err(|_| format!("tensor '{}' has a negative dimension {dim}", tensor.name))?;
        count = count
            .checked_mul(dim)
            .ok_or_else(|| format!("tensor '{}' has an overflowing shape", tensor.name))?;
        shape.push(dim);
    }
    Ok((shape, count))
}

/// Check that a typed field holds exactly `count` values.
fn expect_len(tensor: &TensorProto, field: &str, len: usize, count: usize) -> Result<(), String> {
    if len == count {
        Ok(())
    } else {
        Err(format!(
            "tensor '{}': {field} holds {len} values, shape needs {count}",
            tensor.name
        ))
    }
}

/// Re-encode the typed protobuf fields as little-endian bytes.
fn typed_field_bytes(tensor: &TensorProto, count: usize) -> Result<Vec<u8>, String> {
    let name = &tensor.name;
    let bytes = match tensor.data_type {
        dtype_code::FLOAT32 => {
            expect_len(tensor, "float_data", tensor.float_data.len(), count)?;
            tensor
                .float_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        }
        dtype_code::DOUBLE => {
            expect_len(tensor, "double_data", tensor.double_data.len(), count)?;
            tensor
                .double_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        }
        dtype_code::INT64 => {
            expect_len(tensor, "int64_data", tensor.int64_data.len(), count)?;
            tensor
                .int64_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        }
        dtype_code::INT32 => {
            expect_len(tensor, "int32_data", tensor.int32_data.len(), count)?;
            tensor
                .int32_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        }
        // Stored bit-wise in int32_data: keep the low byte / low 16 bits.
        dtype_code::INT8 | dtype_code::UINT8 => {
            expect_len(tensor, "int32_data", tensor.int32_data.len(), count)?;
            tensor.int32_data.iter().map(|&v| v as u8).collect()
        }
        dtype_code::BOOL => {
            expect_len(tensor, "int32_data", tensor.int32_data.len(), count)?;
            tensor
                .int32_data
                .iter()
                .map(|&v| u8::from(v != 0))
                .collect()
        }
        dtype_code::INT16 | dtype_code::UINT16 | dtype_code::FLOAT16 | dtype_code::BFLOAT16 => {
            expect_len(tensor, "int32_data", tensor.int32_data.len(), count)?;
            tensor
                .int32_data
                .iter()
                .flat_map(|&v| (v as u16).to_le_bytes())
                .collect()
        }
        dtype_code::UINT32 => {
            expect_len(tensor, "uint64_data", tensor.uint64_data.len(), count)?;
            tensor
                .uint64_data
                .iter()
                .flat_map(|&v| (v as u32).to_le_bytes())
                .collect()
        }
        dtype_code::UINT64 => {
            expect_len(tensor, "uint64_data", tensor.uint64_data.len(), count)?;
            tensor
                .uint64_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        }
        other => return Err(format!("tensor '{name}': unsupported data_type {other}")),
    };
    Ok(bytes)
}

/// Read an external-data tensor from its sidecar file.
fn external_bytes(
    tensor: &TensorProto,
    base_dir: &Path,
    byte_len: usize,
) -> Result<Vec<u8>, String> {
    let name = &tensor.name;
    let location = external_entry(tensor, "location")
        .ok_or_else(|| format!("tensor '{name}' is external but has no 'location'"))?;
    let path = resolve_external_path(base_dir, location, name).map_err(|e| e.to_string())?;

    let parse = |key: &str| -> Result<Option<u64>, String> {
        external_entry(tensor, key)
            .map(|v| {
                v.trim()
                    .parse::<u64>()
                    .map_err(|_| format!("tensor '{name}': invalid external '{key}' value '{v}'"))
            })
            .transpose()
    };
    let offset = parse("offset")?.unwrap_or(0);
    if let Some(length) = parse("length")? {
        if length != byte_len as u64 {
            return Err(format!(
                "tensor '{name}': external length {length} does not match the {byte_len} bytes its shape needs"
            ));
        }
    }

    let mut file = std::fs::File::open(&path)
        .map_err(|e| format!("tensor '{name}': cannot open {}: {e}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("tensor '{name}': cannot seek to {offset}: {e}"))?;
    let mut data = vec![0_u8; byte_len];
    file.read_exact(&mut data).map_err(|e| {
        format!(
            "tensor '{name}': {} is too short for {byte_len} bytes at offset {offset}: {e}",
            path.display()
        )
    })?;
    Ok(data)
}

/// Convert one `TensorProto`: `Ok(Err(skip))` for dtypes SafeTensors cannot
/// hold, `Err` for malformed tensors.
fn convert_tensor(
    tensor: &TensorProto,
    name: &str,
    base_dir: &Path,
) -> Result<Result<ExtractedTensor, SkippedTensor>, String> {
    let Some((dtype, width)) = safetensors_dtype(tensor.data_type) else {
        return Ok(Err(SkippedTensor {
            name: name.to_string(),
            reason: format!("unsupported ONNX data_type {}", tensor.data_type),
        }));
    };
    let (shape, count) = shape_of(tensor)?;
    let byte_len = count
        .checked_mul(width)
        .ok_or_else(|| format!("tensor '{name}' is too large"))?;

    let data = if is_external(tensor) {
        external_bytes(tensor, base_dir, byte_len)?
    } else if !tensor.raw_data.is_empty() {
        if tensor.raw_data.len() != byte_len {
            return Err(format!(
                "tensor '{name}': raw_data holds {} bytes, shape needs {byte_len}",
                tensor.raw_data.len()
            ));
        }
        tensor.raw_data.clone()
    } else {
        typed_field_bytes(tensor, count)?
    };

    Ok(Ok(ExtractedTensor {
        name: name.to_string(),
        dtype,
        shape,
        data,
    }))
}

/// Extract every constant tensor of the ONNX model in `model_bytes`.
///
/// `base_dir` is the directory external-data locations are resolved against
/// (the model file's directory). A tensor name seen twice keeps its first
/// occurrence.
pub fn extract_onnx_weights(model_bytes: &[u8], base_dir: &Path) -> Result<OnnxWeights, String> {
    let model = oxionnx_proto::parse_model(model_bytes)
        .map_err(|e| format!("failed to parse ONNX model: {e}"))?;
    let graph = &model.graph;

    let initializer_names: HashSet<&str> =
        graph.initializers.iter().map(|t| t.name.as_str()).collect();
    let mut weights = OnnxWeights {
        node_count: graph.nodes.len(),
        input_count: graph
            .inputs
            .iter()
            .filter(|name| !initializer_names.contains(name.as_str()))
            .count(),
        output_count: graph.outputs.len(),
        ..OnnxWeights::default()
    };

    let constants = graph.nodes.iter().filter_map(|node| {
        let is_constant =
            node.op_type == "Constant" && (node.domain.is_empty() || node.domain == "ai.onnx");
        if !is_constant {
            return None;
        }
        let tensor = node
            .attributes
            .iter()
            .find(|attr| attr.name == "value")
            .and_then(|attr| attr.value.t.as_ref())?;
        let name = node
            .outputs
            .first()
            .filter(|name| !name.is_empty())
            .unwrap_or(&node.name);
        Some((name.as_str(), tensor))
    });

    let mut seen = HashSet::new();
    let sources = graph
        .initializers
        .iter()
        .map(|t| (t.name.as_str(), t))
        .chain(constants);
    for (name, tensor) in sources {
        if name.is_empty() || !seen.insert(name.to_string()) {
            continue;
        }
        match convert_tensor(tensor, name, base_dir)? {
            Ok(extracted) => weights.tensors.push(extracted),
            Err(skipped) => weights.skipped.push(skipped),
        }
    }

    Ok(weights)
}

/// Write `tensors` to a SafeTensors file at `output`, byte-for-byte as
/// extracted (no dtype conversion).
pub fn write_safetensors(
    tensors: &[ExtractedTensor],
    metadata: std::collections::HashMap<String, String>,
    output: &Path,
) -> Result<(), String> {
    use safetensors::tensor::TensorView;

    let mut views = std::collections::HashMap::with_capacity(tensors.len());
    for tensor in tensors {
        let view = TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data)
            .map_err(|e| format!("tensor '{}': {e}", tensor.name))?;
        views.insert(tensor.name.clone(), view);
    }
    safetensors::serialize_to_file(&views, Some(metadata), output)
        .map_err(|e| format!("failed to write {}: {e}", output.display()))
}

#[cfg(test)]
#[path = "onnx_weights_tests.rs"]
mod tests;
