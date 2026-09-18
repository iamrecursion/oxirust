//! Tests for ONNX → SafeTensors weight extraction. Models are encoded
//! in-test with a minimal protobuf writer, so no fixture files are needed.

use super::*;
use safetensors::tensor::Dtype;
use std::collections::HashMap;

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

// ---- minimal protobuf encoder ------------------------------------------------

fn varint(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn field_varint(field: u32, value: u64, out: &mut Vec<u8>) {
    varint(u64::from(field) << 3, out);
    varint(value, out);
}

fn field_bytes(field: u32, bytes: &[u8], out: &mut Vec<u8>) {
    varint((u64::from(field) << 3) | 2, out);
    varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

fn packed_varints(values: impl IntoIterator<Item = u64>) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        varint(v, &mut out);
    }
    out
}

/// How a test tensor stores its data.
enum Storage {
    Raw(Vec<u8>),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    Int32s(Vec<i32>),
    Int64s(Vec<i64>),
    Uint64s(Vec<u64>),
    Strings(Vec<&'static str>),
    External {
        location: String,
        offset: u64,
        length: u64,
    },
}

fn tensor_proto(name: &str, data_type: i32, dims: &[i64], storage: &Storage) -> Vec<u8> {
    let mut t = Vec::new();
    field_bytes(1, &packed_varints(dims.iter().map(|&d| d as u64)), &mut t);
    field_varint(2, data_type as u64, &mut t);
    match storage {
        Storage::Raw(bytes) => field_bytes(9, bytes, &mut t),
        Storage::Floats(v) => {
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            field_bytes(4, &b, &mut t);
        }
        Storage::Doubles(v) => {
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            field_bytes(10, &b, &mut t);
        }
        // Protobuf int32/int64 are varints of the two's-complement u64.
        Storage::Int32s(v) => field_bytes(
            5,
            &packed_varints(v.iter().map(|&x| i64::from(x) as u64)),
            &mut t,
        ),
        Storage::Int64s(v) => field_bytes(7, &packed_varints(v.iter().map(|&x| x as u64)), &mut t),
        Storage::Uint64s(v) => field_bytes(11, &packed_varints(v.iter().copied()), &mut t),
        Storage::Strings(v) => {
            for s in v {
                field_bytes(6, s.as_bytes(), &mut t);
            }
        }
        Storage::External {
            location,
            offset,
            length,
        } => {
            for (key, value) in [
                ("location", location.clone()),
                ("offset", offset.to_string()),
                ("length", length.to_string()),
            ] {
                let mut entry = Vec::new();
                field_bytes(1, key.as_bytes(), &mut entry);
                field_bytes(2, value.as_bytes(), &mut entry);
                field_bytes(13, &entry, &mut t);
            }
            field_varint(14, 1, &mut t);
        }
    }
    field_bytes(8, name.as_bytes(), &mut t);
    t
}

fn value_info(name: &str) -> Vec<u8> {
    let mut v = Vec::new();
    field_bytes(1, name.as_bytes(), &mut v);
    v
}

fn constant_node(output: &str, tensor: &[u8]) -> Vec<u8> {
    let mut attr = Vec::new();
    field_bytes(1, b"value", &mut attr);
    field_bytes(5, tensor, &mut attr);
    field_varint(20, 4, &mut attr); // AttributeType::TENSOR
    let mut node = Vec::new();
    field_bytes(2, output.as_bytes(), &mut node);
    field_bytes(3, b"const_node", &mut node);
    field_bytes(4, b"Constant", &mut node);
    field_bytes(5, &attr, &mut node);
    node
}

fn identity_node(input: &str, output: &str) -> Vec<u8> {
    let mut node = Vec::new();
    field_bytes(1, input.as_bytes(), &mut node);
    field_bytes(2, output.as_bytes(), &mut node);
    field_bytes(4, b"Identity", &mut node);
    node
}

fn model(
    initializers: &[Vec<u8>],
    nodes: &[Vec<u8>],
    inputs: &[&str],
    outputs: &[&str],
) -> Vec<u8> {
    let mut graph = Vec::new();
    for node in nodes {
        field_bytes(1, node, &mut graph);
    }
    field_bytes(2, b"test_graph", &mut graph);
    for init in initializers {
        field_bytes(5, init, &mut graph);
    }
    for input in inputs {
        field_bytes(11, &value_info(input), &mut graph);
    }
    for output in outputs {
        field_bytes(12, &value_info(output), &mut graph);
    }
    let mut opset = Vec::new();
    field_varint(2, 17, &mut opset);
    let mut m = Vec::new();
    field_varint(1, 8, &mut m);
    field_bytes(7, &graph, &mut m);
    field_bytes(8, &opset, &mut m);
    m
}

fn le<T: Copy, const N: usize>(values: &[T], to: fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&v| to(v)).collect()
}

fn temp_dir(tag: &str) -> std::io::Result<std::path::PathBuf> {
    let dir = std::env::temp_dir().join(format!("voirs_onnx_weights_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// One conversion case: name, ONNX data_type, dims, storage, expected
/// SafeTensors dtype, expected bytes.
type DtypeCase = (&'static str, i32, Vec<i64>, Storage, Dtype, Vec<u8>);

// ---- tests -------------------------------------------------------------------

/// Every representable dtype, in both raw_data and typed-field storage,
/// lands in SafeTensors with the same dtype, shape and bytes.
#[test]
fn test_onnx_to_safetensors_preserves_dtypes_byte_for_byte() -> TestResult {
    use dtype_code as d;
    // (name, data_type, dims, storage, expected dtype, expected bytes)
    let cases: Vec<DtypeCase> = vec![
        (
            "f32_raw",
            d::FLOAT32,
            vec![2],
            Storage::Raw(le(&[1.5_f32, -2.0], f32::to_le_bytes)),
            Dtype::F32,
            le(&[1.5_f32, -2.0], f32::to_le_bytes),
        ),
        (
            "f32_typed",
            d::FLOAT32,
            vec![1, 2],
            Storage::Floats(vec![3.25, 0.5]),
            Dtype::F32,
            le(&[3.25_f32, 0.5], f32::to_le_bytes),
        ),
        (
            "f32_scalar",
            d::FLOAT32,
            vec![],
            Storage::Floats(vec![9.0]),
            Dtype::F32,
            le(&[9.0_f32], f32::to_le_bytes),
        ),
        (
            "f64_typed",
            d::DOUBLE,
            vec![2],
            Storage::Doubles(vec![0.1, -0.2]),
            Dtype::F64,
            le(&[0.1_f64, -0.2], f64::to_le_bytes),
        ),
        (
            "i64_typed",
            d::INT64,
            vec![3],
            Storage::Int64s(vec![-5, 7, 1 << 40]),
            Dtype::I64,
            le(&[-5_i64, 7, 1 << 40], i64::to_le_bytes),
        ),
        (
            "i32_raw",
            d::INT32,
            vec![2],
            Storage::Raw(le(&[-1_i32, 123_456], i32::to_le_bytes)),
            Dtype::I32,
            le(&[-1_i32, 123_456], i32::to_le_bytes),
        ),
        (
            "i32_typed",
            d::INT32,
            vec![2],
            Storage::Int32s(vec![-7, 8]),
            Dtype::I32,
            le(&[-7_i32, 8], i32::to_le_bytes),
        ),
        (
            "i8_typed",
            d::INT8,
            vec![2],
            Storage::Int32s(vec![-3, 4]),
            Dtype::I8,
            vec![0xfd, 0x04],
        ),
        (
            "u8_raw",
            d::UINT8,
            vec![3],
            Storage::Raw(vec![0, 128, 255]),
            Dtype::U8,
            vec![0, 128, 255],
        ),
        (
            "bool_typed",
            d::BOOL,
            vec![3],
            Storage::Int32s(vec![0, 5, 1]),
            Dtype::BOOL,
            vec![0, 1, 1],
        ),
        (
            "f16_typed",
            d::FLOAT16,
            vec![2],
            Storage::Int32s(vec![0x3c00, 0xc000]),
            Dtype::F16,
            vec![0x00, 0x3c, 0x00, 0xc0],
        ),
        (
            "bf16_raw",
            d::BFLOAT16,
            vec![1],
            Storage::Raw(vec![0x80, 0x3f]),
            Dtype::BF16,
            vec![0x80, 0x3f],
        ),
        (
            "i16_typed",
            d::INT16,
            vec![1],
            Storage::Int32s(vec![-2]),
            Dtype::I16,
            le(&[-2_i16], i16::to_le_bytes),
        ),
        (
            "u16_typed",
            d::UINT16,
            vec![1],
            Storage::Int32s(vec![65_535]),
            Dtype::U16,
            vec![0xff, 0xff],
        ),
        (
            "u32_typed",
            d::UINT32,
            vec![1],
            Storage::Uint64s(vec![4_000_000_000]),
            Dtype::U32,
            le(&[4_000_000_000_u32], u32::to_le_bytes),
        ),
        (
            "u64_typed",
            d::UINT64,
            vec![1],
            Storage::Uint64s(vec![u64::MAX]),
            Dtype::U64,
            le(&[u64::MAX], u64::to_le_bytes),
        ),
    ];

    let mut initializers: Vec<Vec<u8>> = cases
        .iter()
        .map(|(name, dt, dims, storage, _, _)| tensor_proto(name, *dt, dims, storage))
        .collect();
    initializers.push(tensor_proto(
        "labels",
        d::STRING,
        &[1],
        &Storage::Strings(vec!["x"]),
    ));
    let constant = tensor_proto("", d::INT64, &[2], &Storage::Int64s(vec![11, -12]));
    let bytes = model(
        &initializers,
        &[
            constant_node("const_out", &constant),
            identity_node("input", "output"),
        ],
        // Old-style ONNX lists initializers as inputs too; only "input" is real.
        &["input", "f32_raw"],
        &["output"],
    );

    let dir = temp_dir("dtypes")?;
    let weights = extract_onnx_weights(&bytes, &dir)?;
    assert_eq!(weights.node_count, 2);
    assert_eq!(weights.input_count, 1);
    assert_eq!(weights.output_count, 1);
    assert_eq!(weights.skipped.len(), 1);
    assert_eq!(weights.skipped[0].name, "labels");

    let output = dir.join("model.safetensors");
    write_safetensors(&weights.tensors, HashMap::new(), &output)?;
    let file = std::fs::read(&output)?;
    let reloaded = safetensors::SafeTensors::deserialize(&file)?;
    assert_eq!(reloaded.names().len(), cases.len() + 1);

    for (name, _, dims, _, dtype, expected) in &cases {
        let view = reloaded.tensor(name)?;
        assert_eq!(view.dtype(), *dtype, "{name}");
        let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        assert_eq!(view.shape(), shape.as_slice(), "{name}");
        assert_eq!(view.data(), expected.as_slice(), "{name}");
    }
    let constant = reloaded.tensor("const_out")?;
    assert_eq!(constant.dtype(), Dtype::I64);
    assert_eq!(
        constant.data(),
        le(&[11_i64, -12], i64::to_le_bytes).as_slice()
    );

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// External data is read from the sidecar at the given offset; traversal
/// outside the model directory is rejected.
#[test]
fn test_onnx_external_data_is_read_and_confined() -> TestResult {
    let dir = temp_dir("external")?;
    let mut sidecar = vec![0xaa_u8; 4];
    sidecar.extend(le(&[2.5_f32, -1.0], f32::to_le_bytes));
    std::fs::write(dir.join("weights.bin"), &sidecar)?;

    let external = Storage::External {
        location: "weights.bin".to_string(),
        offset: 4,
        length: 8,
    };
    let bytes = model(
        &[tensor_proto("ext", dtype_code::FLOAT32, &[2], &external)],
        &[],
        &[],
        &[],
    );
    let weights = extract_onnx_weights(&bytes, &dir)?;
    assert_eq!(weights.tensors.len(), 1);
    assert_eq!(
        weights.tensors[0].data,
        le(&[2.5_f32, -1.0], f32::to_le_bytes)
    );

    let escaping = Storage::External {
        location: "../weights.bin".to_string(),
        offset: 0,
        length: 8,
    };
    let bytes = model(
        &[tensor_proto("ext", dtype_code::FLOAT32, &[2], &escaping)],
        &[],
        &[],
        &[],
    );
    assert!(
        extract_onnx_weights(&bytes, &dir).is_err(),
        "`..` must be rejected"
    );

    let wrong_length = Storage::External {
        location: "weights.bin".to_string(),
        offset: 4,
        length: 4,
    };
    let bytes = model(
        &[tensor_proto(
            "ext",
            dtype_code::FLOAT32,
            &[2],
            &wrong_length,
        )],
        &[],
        &[],
        &[],
    );
    assert!(
        extract_onnx_weights(&bytes, &dir).is_err(),
        "length mismatch must be rejected"
    );

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// Data that does not match the declared shape is an error, not a silently
/// truncated tensor.
#[test]
fn test_onnx_malformed_tensor_is_rejected() -> TestResult {
    let dir = temp_dir("malformed")?;
    let short_raw = tensor_proto("w", dtype_code::FLOAT32, &[3], &Storage::Raw(vec![0; 8]));
    assert!(extract_onnx_weights(&model(&[short_raw], &[], &[], &[]), &dir).is_err());

    let short_typed = tensor_proto("w", dtype_code::INT64, &[2], &Storage::Int64s(vec![1]));
    assert!(extract_onnx_weights(&model(&[short_typed], &[], &[], &[]), &dir).is_err());

    assert!(
        extract_onnx_weights(b"\xff\xff\xff", &dir).is_err(),
        "garbage must not parse"
    );

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
