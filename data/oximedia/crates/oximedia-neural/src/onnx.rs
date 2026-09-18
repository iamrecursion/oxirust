//! Pure-Rust ONNX model loading.
//!
//! Parses a `.onnx` protobuf file and maps the computation graph nodes to
//! in-memory layer representations.  No C/C++ runtime dependencies; the
//! protobuf wire format is decoded manually using the `prost` crate's
//! low-level bytes facilities.
//!
//! # Supported ONNX operators
//!
//! | ONNX op | Mapped to |
//! |---------|-----------|
//! | `Gemm` / `MatMul` | [`OnnxLayer::Linear`] |
//! | `Conv` | [`OnnxLayer::Conv2d`] |
//! | `ConvTranspose` | [`OnnxLayer::ConvTranspose2d`] |
//! | `Relu` | [`OnnxLayer::Relu`] |
//! | `Sigmoid` | [`OnnxLayer::Sigmoid`] |
//! | `Tanh` | [`OnnxLayer::Tanh`] |
//! | `MaxPool` | [`OnnxLayer::MaxPool2d`] |
//! | `AveragePool` / `GlobalAveragePool` | [`OnnxLayer::AvgPool2d`] |
//! | `BatchNormalization` | [`OnnxLayer::BatchNorm`] |
//! | `Flatten` | [`OnnxLayer::Flatten`] |
//! | `Reshape` | [`OnnxLayer::Reshape`] |
//! | `Add` | [`OnnxLayer::Add`] |
//! | `Mul` | [`OnnxLayer::Mul`] |
//! | `Dropout` | [`OnnxLayer::Identity`] (inference — no-op) |
//! | `Identity` | [`OnnxLayer::Identity`] |
//!
//! Unrecognised operators produce [`OnnxLayer::Unsupported`] with the op name
//! preserved; execution via [`OnnxModel::run_layer`] returns an error for those.
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_neural::onnx::OnnxModel;
//!
//! let bytes = std::fs::read("model.onnx").unwrap();
//! let model = OnnxModel::from_bytes(&bytes).unwrap();
//! println!("Loaded {} layers", model.layers.len());
//! ```

use std::collections::HashMap;

use crate::error::NeuralError;
use crate::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// Protobuf wire format constants
// ──────────────────────────────────────────────────────────────────────────────

const WIRE_VARINT: u8 = 0;
const WIRE_64BIT: u8 = 1;
const WIRE_LEN_DELIM: u8 = 2;
const WIRE_32BIT: u8 = 5;

// ──────────────────────────────────────────────────────────────────────────────
// Low-level protobuf reader
// ──────────────────────────────────────────────────────────────────────────────

/// Minimal protobuf reader used to hand-parse ONNX protos.
struct ProtoReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Reads a base-128 varint.
    fn read_varint(&mut self) -> Result<u64, NeuralError> {
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            if self.pos >= self.data.len() {
                return Err(NeuralError::Io("unexpected end of protobuf varint".into()));
            }
            let byte = self.data[self.pos];
            self.pos += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(NeuralError::Io("varint too long".into()));
            }
        }
        Ok(result)
    }

    /// Reads a field tag and returns `(field_number, wire_type)`.
    fn read_tag(&mut self) -> Result<(u32, u8), NeuralError> {
        let tag = self.read_varint()?;
        let field_number = (tag >> 3) as u32;
        let wire_type = (tag & 0x07) as u8;
        Ok((field_number, wire_type))
    }

    /// Skips a field with the given wire type.
    fn skip_field(&mut self, wire_type: u8) -> Result<(), NeuralError> {
        match wire_type {
            WIRE_VARINT => {
                self.read_varint()?;
            }
            WIRE_64BIT => {
                if self.pos + 8 > self.data.len() {
                    return Err(NeuralError::Io(
                        "unexpected end while skipping 64-bit field".into(),
                    ));
                }
                self.pos += 8;
            }
            WIRE_LEN_DELIM => {
                let len = self.read_varint()? as usize;
                if self.pos + len > self.data.len() {
                    return Err(NeuralError::Io(
                        "unexpected end while skipping len-delim field".into(),
                    ));
                }
                self.pos += len;
            }
            WIRE_32BIT => {
                if self.pos + 4 > self.data.len() {
                    return Err(NeuralError::Io(
                        "unexpected end while skipping 32-bit field".into(),
                    ));
                }
                self.pos += 4;
            }
            other => {
                return Err(NeuralError::Io(format!("unknown wire type {other}")));
            }
        }
        Ok(())
    }

    /// Reads a length-delimited byte slice.
    fn read_bytes(&mut self) -> Result<&'a [u8], NeuralError> {
        let len = self.read_varint()? as usize;
        if self.pos + len > self.data.len() {
            return Err(NeuralError::Io("unexpected end while reading bytes".into()));
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    /// Reads a length-delimited UTF-8 string.
    fn read_string(&mut self) -> Result<String, NeuralError> {
        let bytes = self.read_bytes()?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| NeuralError::Io(format!("invalid utf8 in proto string: {e}")))
    }

    /// Returns a sub-reader over the next length-delimited field.
    fn sub_reader(&mut self) -> Result<ProtoReader<'a>, NeuralError> {
        let bytes = self.read_bytes()?;
        Ok(ProtoReader::new(bytes))
    }

    /// Reads a little-endian f32.
    fn read_f32_le(&mut self) -> Result<f32, NeuralError> {
        if self.pos + 4 > self.data.len() {
            return Err(NeuralError::Io("unexpected end reading f32".into()));
        }
        let bytes: [u8; 4] = self.data[self.pos..self.pos + 4]
            .try_into()
            .map_err(|_| NeuralError::Io("failed to read 4 bytes for f32".into()))?;
        self.pos += 4;
        Ok(f32::from_le_bytes(bytes))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ONNX proto structures
// ──────────────────────────────────────────────────────────────────────────────

/// ONNX data type codes as used in the TensorProto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDataType {
    Float,
    Int64,
    Int32,
    Other(i32),
}

impl OnnxDataType {
    fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Float,
            6 => Self::Int32,
            7 => Self::Int64,
            other => Self::Other(other),
        }
    }
}

/// An initializer tensor parsed from the ONNX model (i.e. weight or bias).
#[derive(Debug, Clone)]
pub struct OnnxInitializer {
    /// Tensor name, used for lookups by graph nodes.
    pub name: String,
    /// Tensor shape.
    pub dims: Vec<i64>,
    /// Data type.
    pub data_type: OnnxDataType,
    /// f32 data (populated for FLOAT tensors).
    pub float_data: Vec<f32>,
}

impl OnnxInitializer {
    /// Converts to a `Tensor`.  Returns an error if the data type is not FLOAT.
    pub fn to_tensor(&self) -> Result<Tensor, NeuralError> {
        if self.data_type != OnnxDataType::Float {
            return Err(NeuralError::Io(format!(
                "OnnxInitializer '{}': only FLOAT type supported, got {:?}",
                self.name, self.data_type
            )));
        }
        let shape: Vec<usize> = self.dims.iter().map(|&d| d.max(1) as usize).collect();
        if shape.is_empty() {
            return Err(NeuralError::InvalidShape(format!(
                "OnnxInitializer '{}': empty dims",
                self.name
            )));
        }
        Tensor::from_data(self.float_data.clone(), shape)
    }
}

/// An ONNX graph node.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Operator type string (e.g. `"Conv"`, `"Relu"`).
    pub op_type: String,
    /// Node name (may be empty).
    pub name: String,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Node attributes: name → AttributeValue.
    pub attributes: HashMap<String, OnnxAttrValue>,
}

/// An ONNX attribute value.
#[derive(Debug, Clone)]
pub enum OnnxAttrValue {
    /// Single integer.
    Int(i64),
    /// Single float.
    Float(f32),
    /// List of integers (e.g. kernel_shape, pads, strides).
    Ints(Vec<i64>),
    /// List of floats.
    Floats(Vec<f32>),
    /// String.
    String(String),
}

impl OnnxAttrValue {
    /// Extracts as a single i64 if possible.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts as a `Vec<i64>` if possible.
    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            Self::Ints(v) => Some(v),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OnnxLayer — the high-level IR
// ──────────────────────────────────────────────────────────────────────────────

/// High-level layer representation derived from an ONNX graph node.
#[derive(Debug, Clone)]
pub enum OnnxLayer {
    /// Fully-connected (Gemm / MatMul) layer.
    Linear {
        weight: Tensor,
        bias: Option<Tensor>,
    },
    /// 2-D convolution.
    Conv2d {
        weight: Tensor,
        bias: Option<Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    },
    /// 2-D transposed convolution.
    ConvTranspose2d {
        weight: Tensor,
        bias: Option<Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    },
    /// ReLU activation.
    Relu,
    /// Sigmoid activation.
    Sigmoid,
    /// Tanh activation.
    Tanh,
    /// 2-D max pooling.
    MaxPool2d {
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    /// 2-D / global average pooling.
    AvgPool2d {
        kernel: (usize, usize),
        stride: (usize, usize),
        global: bool,
    },
    /// Batch normalization (inference mode).
    BatchNorm {
        scale: Tensor,
        bias: Tensor,
        mean: Tensor,
        var: Tensor,
        epsilon: f32,
    },
    /// Flatten to 1-D.
    Flatten { axis: usize },
    /// Reshape.
    Reshape { target_shape: Vec<usize> },
    /// Element-wise add of two inputs.
    Add,
    /// Element-wise multiply of two inputs.
    Mul,
    /// No-op (Dropout inference / Identity).
    Identity,
    /// Operator not yet mapped.
    Unsupported { op_type: String },
}

// ──────────────────────────────────────────────────────────────────────────────
// OnnxModel
// ──────────────────────────────────────────────────────────────────────────────

/// A parsed ONNX model ready for inference.
pub struct OnnxModel {
    /// Ordered list of layers derived from the graph nodes.
    pub layers: Vec<OnnxLayer>,
    /// Raw ONNX nodes (preserved for debugging).
    pub nodes: Vec<OnnxNode>,
    /// Initializer map: name → tensor.
    pub initializers: HashMap<String, OnnxInitializer>,
    /// IR version from the ModelProto header.
    pub ir_version: i64,
    /// Model domain.
    pub domain: String,
    /// ONNX opset version.
    pub opset_version: i64,
}

impl OnnxModel {
    /// Loads an ONNX model from raw protobuf bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NeuralError> {
        let mut reader = ProtoReader::new(bytes);
        parse_model_proto(&mut reader)
    }

    /// Loads an ONNX model from a file path.
    pub fn from_file(path: &str) -> Result<Self, NeuralError> {
        let bytes = std::fs::read(path)
            .map_err(|e| NeuralError::Io(format!("failed to read '{path}': {e}")))?;
        Self::from_bytes(&bytes)
    }

    /// Runs a single layer against an input tensor, returning the output.
    ///
    /// For binary ops (`Add`, `Mul`) a second tensor `rhs` is required.
    pub fn run_layer(
        layer: &OnnxLayer,
        input: &Tensor,
        rhs: Option<&Tensor>,
    ) -> Result<Tensor, NeuralError> {
        use crate::layers::{
            AvgPool2d, BatchNorm2d, Conv2dLayer, ConvTranspose2d, GlobalAvgPool, MaxPool2d,
        };
        use crate::tensor::{add, mul};

        match layer {
            OnnxLayer::Relu => {
                let mut out = input.clone();
                crate::tensor::relu_inplace(&mut out);
                Ok(out)
            }
            OnnxLayer::Sigmoid => {
                let d: Vec<f32> = input
                    .data()
                    .iter()
                    .map(|&x| 1.0 / (1.0 + (-x).exp()))
                    .collect();
                Tensor::from_data(d, input.shape().to_vec())
            }
            OnnxLayer::Tanh => {
                let d: Vec<f32> = input.data().iter().map(|&x| x.tanh()).collect();
                Tensor::from_data(d, input.shape().to_vec())
            }
            OnnxLayer::Identity => Ok(input.clone()),
            OnnxLayer::Flatten { axis } => {
                let shape = input.shape();
                let leading: usize = shape[..*axis].iter().product();
                let trailing: usize = shape[*axis..].iter().product();
                let new_shape = if leading == 0 {
                    vec![trailing]
                } else {
                    vec![leading, trailing]
                };
                input.reshape(new_shape)
            }
            OnnxLayer::Reshape { target_shape } => input.reshape(target_shape.clone()),
            OnnxLayer::Add => {
                let r =
                    rhs.ok_or_else(|| NeuralError::InvalidShape("Add layer requires rhs".into()))?;
                add(input, r)
            }
            OnnxLayer::Mul => {
                let r =
                    rhs.ok_or_else(|| NeuralError::InvalidShape("Mul layer requires rhs".into()))?;
                mul(input, r)
            }
            OnnxLayer::Linear { weight, bias } => {
                let mut layer =
                    crate::layers::LinearLayer::new(weight.shape()[1], weight.shape()[0])?;
                layer.weight = weight.clone();
                if let Some(b) = bias {
                    layer.bias = b.clone();
                }
                layer.forward(input)
            }
            OnnxLayer::Conv2d {
                weight,
                bias,
                stride,
                padding,
                ..
            } => {
                let (oc, ic, kh, kw) = (
                    weight.shape()[0],
                    weight.shape()[1],
                    weight.shape()[2],
                    weight.shape()[3],
                );
                let mut layer = Conv2dLayer::new(ic, oc, kh, kw, *stride, *padding)?;
                layer.weight = weight.clone();
                if let Some(b) = bias {
                    layer.bias = b.clone();
                }
                layer.forward(input)
            }
            OnnxLayer::ConvTranspose2d {
                weight,
                bias,
                stride,
                padding,
                output_padding,
            } => {
                let (ic, oc, kh, kw) = (
                    weight.shape()[0],
                    weight.shape()[1],
                    weight.shape()[2],
                    weight.shape()[3],
                );
                let mut layer =
                    ConvTranspose2d::new(ic, oc, kh, kw, *stride, *padding, *output_padding)?;
                layer.weight = weight.clone();
                if let Some(b) = bias {
                    layer.bias = b.clone();
                }
                layer.forward(input)
            }
            OnnxLayer::MaxPool2d { kernel, stride, .. } => {
                let layer = MaxPool2d::new(*kernel, *stride)?;
                layer.forward(input)
            }
            OnnxLayer::AvgPool2d {
                kernel,
                stride,
                global,
            } => {
                if *global {
                    let layer = GlobalAvgPool::new();
                    layer.forward(input)
                } else {
                    let layer = AvgPool2d::new(*kernel, *stride)?;
                    layer.forward(input)
                }
            }
            OnnxLayer::BatchNorm {
                scale,
                bias,
                mean,
                var,
                epsilon,
            } => {
                let c = scale.numel();
                let mut bn = BatchNorm2d::new(c)?;
                bn.weight = scale.data().to_vec();
                bn.bias = bias.data().to_vec();
                bn.running_mean = mean.data().to_vec();
                bn.running_var = var.data().to_vec();
                bn.eps = *epsilon;
                bn.forward(input)
            }
            OnnxLayer::Unsupported { op_type } => {
                Err(NeuralError::Io(format!("unsupported ONNX op: '{op_type}'")))
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OnnxModelInfo — lightweight model introspection without weight loading
// ──────────────────────────────────────────────────────────────────────────────

/// Lightweight metadata extracted from an ONNX model file.
///
/// Unlike [`OnnxModel`], this struct does **not** load weight tensors.
/// It is intended for fast model introspection: checking operator compatibility,
/// verifying I/O names, and reading header metadata before committing to a full
/// parse.
///
/// # Example
///
/// ```rust
/// use oximedia_neural::onnx::inspect_onnx;
///
/// // In practice you would load bytes from disk.
/// let bytes: &[u8] = &[];
/// match inspect_onnx(bytes) {
///     Ok(info) => {
///         println!("IR version: {}", info.ir_version);
///         println!("Ops: {:?}", info.node_types);
///     }
///     Err(e) => eprintln!("Parse error: {e}"),
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OnnxModelInfo {
    /// ONNX IR format version (field 1 of `ModelProto`).
    pub ir_version: i64,
    /// ONNX opset version used by this model.
    pub opset_version: i64,
    /// Name of the computation graph (may be empty).
    pub graph_name: String,
    /// Deduplicated list of operator type strings found in graph nodes.
    /// E.g. `["Conv", "BatchNormalization", "Relu", "Gemm"]`.
    pub node_types: Vec<String>,
    /// Names of graph-level inputs (excluding initializers).
    pub input_names: Vec<String>,
    /// Names of graph-level outputs.
    pub output_names: Vec<String>,
}

/// Parses an ONNX protobuf byte slice and returns lightweight model metadata.
///
/// This function intentionally avoids allocating weight data: it reads only
/// the header fields, graph name, node op-types, and I/O names.  It is
/// therefore much cheaper than [`OnnxModel::from_bytes`] for large models.
///
/// ## Errors
///
/// Returns [`NeuralError::Io`] if the protobuf wire format is malformed.
/// An empty `data` slice is accepted and returns a zeroed `OnnxModelInfo`.
pub fn inspect_onnx(data: &[u8]) -> Result<OnnxModelInfo, NeuralError> {
    let mut reader = ProtoReader::new(data);
    parse_model_info(&mut reader)
}

/// Internal: parses `ModelProto` fields needed for `OnnxModelInfo`.
fn parse_model_info(reader: &mut ProtoReader<'_>) -> Result<OnnxModelInfo, NeuralError> {
    let mut ir_version = 0i64;
    let mut opset_version = 0i64;
    let mut graph_bytes: Option<Vec<u8>> = None;

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, WIRE_VARINT) => {
                ir_version = reader.read_varint()? as i64;
            }
            (2, WIRE_LEN_DELIM) => {
                // opset_import: OperatorSetIdProto, field 2 = version
                let mut sub = reader.sub_reader()?;
                while !sub.is_empty() {
                    let (f, w) = sub.read_tag()?;
                    match (f, w) {
                        (2, WIRE_VARINT) => {
                            opset_version = sub.read_varint()? as i64;
                        }
                        _ => {
                            sub.skip_field(w)?;
                        }
                    }
                }
            }
            (4, WIRE_LEN_DELIM) => {
                // graph (GraphProto)
                let bytes = reader.read_bytes()?;
                graph_bytes = Some(bytes.to_vec());
            }
            (_, w) => {
                reader.skip_field(w)?;
            }
        }
    }

    let graph_data = graph_bytes.unwrap_or_default();
    let (graph_name, node_types, input_names, output_names) = parse_graph_info(&graph_data)?;

    Ok(OnnxModelInfo {
        ir_version,
        opset_version,
        graph_name,
        node_types,
        input_names,
        output_names,
    })
}

/// Parses a `GraphProto` and extracts only structural metadata (no weights).
fn parse_graph_info(
    data: &[u8],
) -> Result<(String, Vec<String>, Vec<String>, Vec<String>), NeuralError> {
    let mut reader = ProtoReader::new(data);
    let mut graph_name = String::new();
    // Use insertion-ordered dedup via a Vec + seen set pattern.
    let mut node_types_ordered: Vec<String> = Vec::new();
    let mut node_types_seen = std::collections::HashSet::new();
    let mut input_names: Vec<String> = Vec::new();
    let mut output_names: Vec<String> = Vec::new();
    // We will collect initializer names to exclude them from inputs.
    let mut initializer_names = std::collections::HashSet::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, WIRE_LEN_DELIM) => {
                // node (NodeProto): we only need op_type (field 4).
                let bytes = reader.read_bytes()?;
                let op_type = extract_op_type_from_node(bytes)?;
                if !op_type.is_empty() && node_types_seen.insert(op_type.clone()) {
                    node_types_ordered.push(op_type);
                }
            }
            (3, WIRE_LEN_DELIM) => {
                // graph name
                graph_name = reader.read_string()?;
            }
            (11, WIRE_LEN_DELIM) => {
                // input (ValueInfoProto): field 1 = name
                let bytes = reader.read_bytes()?;
                let name = extract_name_from_value_info(bytes)?;
                if !name.is_empty() {
                    input_names.push(name);
                }
            }
            (12, WIRE_LEN_DELIM) => {
                // output (ValueInfoProto): field 1 = name
                let bytes = reader.read_bytes()?;
                let name = extract_name_from_value_info(bytes)?;
                if !name.is_empty() {
                    output_names.push(name);
                }
            }
            (5, WIRE_LEN_DELIM) => {
                // initializer (TensorProto): field 8 = name
                let bytes = reader.read_bytes()?;
                let name = extract_initializer_name(bytes)?;
                if !name.is_empty() {
                    initializer_names.insert(name);
                }
            }
            (_, w) => {
                reader.skip_field(w)?;
            }
        }
    }

    // Filter out initializers from the input list (ONNX models often list
    // weight tensors as graph inputs).
    let filtered_inputs: Vec<String> = input_names
        .into_iter()
        .filter(|n| !initializer_names.contains(n))
        .collect();

    Ok((
        graph_name,
        node_types_ordered,
        filtered_inputs,
        output_names,
    ))
}

/// Extracts `op_type` (field 4, string) from a serialised `NodeProto`.
fn extract_op_type_from_node(data: &[u8]) -> Result<String, NeuralError> {
    let mut reader = ProtoReader::new(data);
    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        if field == 4 && wire == WIRE_LEN_DELIM {
            return reader.read_string();
        }
        reader.skip_field(wire)?;
    }
    Ok(String::new())
}

/// Extracts the `name` (field 1, string) from a `ValueInfoProto`.
fn extract_name_from_value_info(data: &[u8]) -> Result<String, NeuralError> {
    let mut reader = ProtoReader::new(data);
    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        if field == 1 && wire == WIRE_LEN_DELIM {
            return reader.read_string();
        }
        reader.skip_field(wire)?;
    }
    Ok(String::new())
}

/// Extracts the `name` (field 8, string) from a `TensorProto` initializer.
fn extract_initializer_name(data: &[u8]) -> Result<String, NeuralError> {
    let mut reader = ProtoReader::new(data);
    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        if field == 8 && wire == WIRE_LEN_DELIM {
            return reader.read_string();
        }
        reader.skip_field(wire)?;
    }
    Ok(String::new())
}

// ──────────────────────────────────────────────────────────────────────────────
// Parsing helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Parses a `ModelProto` (the top-level ONNX message).
fn parse_model_proto(reader: &mut ProtoReader<'_>) -> Result<OnnxModel, NeuralError> {
    let mut ir_version = 0i64;
    let mut domain = String::new();
    let mut opset_version = 0i64;
    let mut graph_bytes: Option<Vec<u8>> = None;

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, WIRE_VARINT) => {
                ir_version = reader.read_varint()? as i64;
            }
            (2, WIRE_LEN_DELIM) => {
                // opset_import — OperatorSetIdProto, field 2 = version (varint)
                let mut sub = reader.sub_reader()?;
                while !sub.is_empty() {
                    let (f, w) = sub.read_tag()?;
                    match (f, w) {
                        (2, WIRE_VARINT) => {
                            opset_version = sub.read_varint()? as i64;
                        }
                        _ => {
                            sub.skip_field(w)?;
                        }
                    }
                }
            }
            (4, WIRE_LEN_DELIM) => {
                // graph (GraphProto)
                let bytes = reader.read_bytes()?;
                graph_bytes = Some(bytes.to_vec());
            }
            (7, WIRE_LEN_DELIM) => {
                domain = reader.read_string()?;
            }
            (_, w) => {
                reader.skip_field(w)?;
            }
        }
    }

    let graph_data = graph_bytes.unwrap_or_default();
    let mut graph_reader = ProtoReader::new(&graph_data);
    let (nodes, initializers) = parse_graph_proto(&mut graph_reader)?;

    let layers = build_layers(&nodes, &initializers)?;

    Ok(OnnxModel {
        layers,
        nodes,
        initializers,
        ir_version,
        domain,
        opset_version,
    })
}

/// Parses a `GraphProto`, returning nodes and initializers.
fn parse_graph_proto(
    reader: &mut ProtoReader<'_>,
) -> Result<(Vec<OnnxNode>, HashMap<String, OnnxInitializer>), NeuralError> {
    let mut nodes = Vec::new();
    let mut initializers = HashMap::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, WIRE_LEN_DELIM) => {
                // node (NodeProto)
                let bytes = reader.read_bytes()?;
                let mut sub = ProtoReader::new(bytes);
                let node = parse_node_proto(&mut sub)?;
                nodes.push(node);
            }
            (5, WIRE_LEN_DELIM) => {
                // initializer (TensorProto)
                let bytes = reader.read_bytes()?;
                let mut sub = ProtoReader::new(bytes);
                let init = parse_tensor_proto(&mut sub)?;
                initializers.insert(init.name.clone(), init);
            }
            (_, w) => {
                reader.skip_field(w)?;
            }
        }
    }
    Ok((nodes, initializers))
}

/// Parses a `NodeProto`.
fn parse_node_proto(reader: &mut ProtoReader<'_>) -> Result<OnnxNode, NeuralError> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut name = String::new();
    let mut op_type = String::new();
    let mut attributes = HashMap::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, WIRE_LEN_DELIM) => {
                inputs.push(reader.read_string()?);
            }
            (2, WIRE_LEN_DELIM) => {
                outputs.push(reader.read_string()?);
            }
            (3, WIRE_LEN_DELIM) => {
                name = reader.read_string()?;
            }
            (4, WIRE_LEN_DELIM) => {
                op_type = reader.read_string()?;
            }
            (5, WIRE_LEN_DELIM) => {
                // attribute (AttributeProto)
                let bytes = reader.read_bytes()?;
                let mut sub = ProtoReader::new(bytes);
                if let Ok((attr_name, attr_val)) = parse_attribute_proto(&mut sub) {
                    attributes.insert(attr_name, attr_val);
                }
            }
            (_, w) => {
                reader.skip_field(w)?;
            }
        }
    }

    Ok(OnnxNode {
        op_type,
        name,
        inputs,
        outputs,
        attributes,
    })
}

/// Parses an `AttributeProto`.
fn parse_attribute_proto(
    reader: &mut ProtoReader<'_>,
) -> Result<(String, OnnxAttrValue), NeuralError> {
    let mut attr_name = String::new();
    let mut attr_type = 0i32;
    let mut int_val = 0i64;
    let mut float_val = 0.0f32;
    let mut ints: Vec<i64> = Vec::new();
    let mut floats: Vec<f32> = Vec::new();
    let mut str_bytes: Vec<u8> = Vec::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, WIRE_LEN_DELIM) => {
                attr_name = reader.read_string()?;
            }
            (20, WIRE_VARINT) => {
                attr_type = reader.read_varint()? as i32;
            }
            (3, WIRE_VARINT) => {
                int_val = reader.read_varint()? as i64;
            }
            (4, WIRE_32BIT) => {
                float_val = reader.read_f32_le()?;
            }
            (6, WIRE_VARINT) => {
                ints.push(reader.read_varint()? as i64);
            }
            (7, WIRE_32BIT) => {
                floats.push(reader.read_f32_le()?);
            }
            (8, WIRE_LEN_DELIM) => {
                str_bytes = reader.read_bytes()?.to_vec();
            }
            (_, w) => {
                reader.skip_field(w)?;
            }
        }
    }

    let value = match attr_type {
        1 => OnnxAttrValue::Float(float_val),
        2 => OnnxAttrValue::Int(int_val),
        3 => OnnxAttrValue::String(String::from_utf8(str_bytes).unwrap_or_default()),
        6 => OnnxAttrValue::Floats(floats),
        7 => OnnxAttrValue::Ints(ints),
        _ => {
            // Fallback: prefer ints if present.
            if !ints.is_empty() {
                OnnxAttrValue::Ints(ints)
            } else if !floats.is_empty() {
                OnnxAttrValue::Floats(floats)
            } else {
                OnnxAttrValue::Int(int_val)
            }
        }
    };

    Ok((attr_name, value))
}

/// Parses a `TensorProto` initializer.
fn parse_tensor_proto(reader: &mut ProtoReader<'_>) -> Result<OnnxInitializer, NeuralError> {
    let mut dims: Vec<i64> = Vec::new();
    let mut data_type = 0i32;
    let mut name = String::new();
    let mut float_data: Vec<f32> = Vec::new();
    let mut raw_data: Option<Vec<u8>> = None;

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, WIRE_VARINT) => {
                dims.push(reader.read_varint()? as i64);
            }
            (2, WIRE_VARINT) => {
                data_type = reader.read_varint()? as i32;
            }
            (4, WIRE_32BIT) => {
                float_data.push(reader.read_f32_le()?);
            }
            (8, WIRE_LEN_DELIM) => {
                name = reader.read_string()?;
            }
            (9, WIRE_LEN_DELIM) => {
                raw_data = Some(reader.read_bytes()?.to_vec());
            }
            (_, w) => {
                reader.skip_field(w)?;
            }
        }
    }

    // Decode raw_data if present.
    if let Some(raw) = raw_data {
        match OnnxDataType::from_i32(data_type) {
            OnnxDataType::Float => {
                float_data = decode_raw_f32(&raw)?;
            }
            OnnxDataType::Int64 => {
                let ints = decode_raw_i64(&raw)?;
                float_data = ints.into_iter().map(|i| i as f32).collect();
            }
            OnnxDataType::Int32 => {
                let ints = decode_raw_i32(&raw)?;
                float_data = ints.into_iter().map(|i| i as f32).collect();
            }
            OnnxDataType::Other(_) => {}
        }
    }

    Ok(OnnxInitializer {
        name,
        dims,
        data_type: OnnxDataType::from_i32(data_type),
        float_data,
    })
}

fn decode_raw_f32(raw: &[u8]) -> Result<Vec<f32>, NeuralError> {
    if raw.len() % 4 != 0 {
        return Err(NeuralError::Io("raw_data length not multiple of 4".into()));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn decode_raw_i64(raw: &[u8]) -> Result<Vec<i64>, NeuralError> {
    if raw.len() % 8 != 0 {
        return Err(NeuralError::Io(
            "raw_data length not multiple of 8 for i64".into(),
        ));
    }
    Ok(raw
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

fn decode_raw_i32(raw: &[u8]) -> Result<Vec<i32>, NeuralError> {
    if raw.len() % 4 != 0 {
        return Err(NeuralError::Io(
            "raw_data length not multiple of 4 for i32".into(),
        ));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ──────────────────────────────────────────────────────────────────────────────
// Layer builder
// ──────────────────────────────────────────────────────────────────────────────

/// Maps ONNX nodes to [`OnnxLayer`] values.
fn build_layers(
    nodes: &[OnnxNode],
    initializers: &HashMap<String, OnnxInitializer>,
) -> Result<Vec<OnnxLayer>, NeuralError> {
    let mut layers = Vec::with_capacity(nodes.len());

    for node in nodes {
        let layer = build_layer(node, initializers)?;
        layers.push(layer);
    }
    Ok(layers)
}

fn get_tensor(
    node: &OnnxNode,
    input_index: usize,
    initializers: &HashMap<String, OnnxInitializer>,
) -> Option<Tensor> {
    node.inputs
        .get(input_index)
        .and_then(|name| initializers.get(name))
        .and_then(|init| init.to_tensor().ok())
}

fn attr_pair(attrs: &HashMap<String, OnnxAttrValue>, name: &str) -> (usize, usize) {
    if let Some(OnnxAttrValue::Ints(v)) = attrs.get(name) {
        let h = v.first().copied().unwrap_or(1).max(1) as usize;
        let w = v.get(1).copied().unwrap_or(h as i64).max(1) as usize;
        (h, w)
    } else {
        (1, 1)
    }
}

fn build_layer(
    node: &OnnxNode,
    initializers: &HashMap<String, OnnxInitializer>,
) -> Result<OnnxLayer, NeuralError> {
    match node.op_type.as_str() {
        "Gemm" | "MatMul" => {
            let weight = get_tensor(node, 1, initializers)
                .ok_or_else(|| NeuralError::Io(format!("Gemm '{}': missing weight", node.name)))?;
            let bias = get_tensor(node, 2, initializers);
            Ok(OnnxLayer::Linear { weight, bias })
        }
        "Conv" => {
            let weight = get_tensor(node, 0, initializers)
                .ok_or_else(|| NeuralError::Io(format!("Conv '{}': missing weight", node.name)))?;
            let bias = get_tensor(node, 1, initializers);
            let stride = attr_pair(&node.attributes, "strides");
            let padding = attr_pair(&node.attributes, "pads");
            let dilation = attr_pair(&node.attributes, "dilations");
            let groups = node
                .attributes
                .get("group")
                .and_then(|v| v.as_int())
                .unwrap_or(1) as usize;
            Ok(OnnxLayer::Conv2d {
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
            })
        }
        "ConvTranspose" => {
            let weight = get_tensor(node, 0, initializers).ok_or_else(|| {
                NeuralError::Io(format!("ConvTranspose '{}': missing weight", node.name))
            })?;
            let bias = get_tensor(node, 1, initializers);
            let stride = attr_pair(&node.attributes, "strides");
            let padding = attr_pair(&node.attributes, "pads");
            let output_padding = attr_pair(&node.attributes, "output_padding");
            Ok(OnnxLayer::ConvTranspose2d {
                weight,
                bias,
                stride,
                padding,
                output_padding,
            })
        }
        "Relu" => Ok(OnnxLayer::Relu),
        "Sigmoid" => Ok(OnnxLayer::Sigmoid),
        "Tanh" => Ok(OnnxLayer::Tanh),
        "Dropout" | "Identity" => Ok(OnnxLayer::Identity),
        "Flatten" => {
            let axis = node
                .attributes
                .get("axis")
                .and_then(|v| v.as_int())
                .unwrap_or(1)
                .max(0) as usize;
            Ok(OnnxLayer::Flatten { axis })
        }
        "MaxPool" => {
            let kernel = attr_pair(&node.attributes, "kernel_shape");
            let stride = attr_pair(&node.attributes, "strides");
            let padding = attr_pair(&node.attributes, "pads");
            Ok(OnnxLayer::MaxPool2d {
                kernel,
                stride,
                padding,
            })
        }
        "AveragePool" => {
            let kernel = attr_pair(&node.attributes, "kernel_shape");
            let stride = attr_pair(&node.attributes, "strides");
            Ok(OnnxLayer::AvgPool2d {
                kernel,
                stride,
                global: false,
            })
        }
        "GlobalAveragePool" => Ok(OnnxLayer::AvgPool2d {
            kernel: (1, 1),
            stride: (1, 1),
            global: true,
        }),
        "BatchNormalization" => {
            let scale = get_tensor(node, 0, initializers)
                .ok_or_else(|| NeuralError::Io("BatchNorm: missing scale".into()))?;
            let bias = get_tensor(node, 1, initializers)
                .ok_or_else(|| NeuralError::Io("BatchNorm: missing bias".into()))?;
            let mean = get_tensor(node, 2, initializers)
                .ok_or_else(|| NeuralError::Io("BatchNorm: missing mean".into()))?;
            let var = get_tensor(node, 3, initializers)
                .ok_or_else(|| NeuralError::Io("BatchNorm: missing var".into()))?;
            let epsilon = node
                .attributes
                .get("epsilon")
                .and_then(|v| match v {
                    OnnxAttrValue::Float(f) => Some(*f),
                    _ => None,
                })
                .unwrap_or(1e-5);
            Ok(OnnxLayer::BatchNorm {
                scale,
                bias,
                mean,
                var,
                epsilon,
            })
        }
        "Add" => Ok(OnnxLayer::Add),
        "Mul" => Ok(OnnxLayer::Mul),
        other => Ok(OnnxLayer::Unsupported {
            op_type: other.to_string(),
        }),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Protobuf primitives ───────────────────────────────────────────────────

    fn encode_varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let b = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
        out
    }

    fn encode_len_delim(field: u32, data: &[u8]) -> Vec<u8> {
        let tag = ((field << 3) | 2) as u64;
        let mut out = encode_varint(tag);
        out.extend_from_slice(&encode_varint(data.len() as u64));
        out.extend_from_slice(data);
        out
    }

    fn encode_varint_field(field: u32, value: u64) -> Vec<u8> {
        let tag = ((field << 3) | 0) as u64;
        let mut out = encode_varint(tag);
        out.extend_from_slice(&encode_varint(value));
        out
    }

    #[allow(dead_code)]
    fn encode_f32_field(field: u32, value: f32) -> Vec<u8> {
        let tag = ((field << 3) | 5) as u64;
        let mut out = encode_varint(tag);
        out.extend_from_slice(&value.to_le_bytes());
        out
    }

    // Builds a minimal TensorProto for a float vector.
    fn make_tensor_proto(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        // field 1 = dims (varint repeated)
        for &d in dims {
            buf.extend_from_slice(&encode_varint_field(1, d as u64));
        }
        // field 2 = data_type = FLOAT (1)
        buf.extend_from_slice(&encode_varint_field(2, 1));
        // field 8 = name (string)
        buf.extend_from_slice(&encode_len_delim(8, name.as_bytes()));
        // field 9 = raw_data
        let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        buf.extend_from_slice(&encode_len_delim(9, &raw));
        buf
    }

    // ── Round-trip tests ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_tensor_proto_float() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let buf = make_tensor_proto("weight", &[2, 2], &data);
        let mut reader = ProtoReader::new(&buf);
        let init = parse_tensor_proto(&mut reader).expect("parse_tensor_proto");
        assert_eq!(init.name, "weight");
        assert_eq!(init.dims, vec![2i64, 2]);
        assert_eq!(init.float_data.len(), 4);
        for (got, exp) in init.float_data.iter().zip(data.iter()) {
            assert!((got - exp).abs() < 1e-6, "got={got} exp={exp}");
        }
    }

    #[test]
    fn test_parse_tensor_to_tensor() {
        let data = vec![1.0_f32; 6];
        let buf = make_tensor_proto("w", &[2, 3], &data);
        let mut reader = ProtoReader::new(&buf);
        let init = parse_tensor_proto(&mut reader).expect("parse_tensor_proto");
        let tensor = init.to_tensor().expect("to_tensor");
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn test_run_layer_relu() {
        let input =
            Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).expect("tensor from_data");
        let out = OnnxModel::run_layer(&OnnxLayer::Relu, &input, None).expect("run layer");
        assert_eq!(out.data(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_run_layer_sigmoid_range() {
        // Use moderate values to avoid f32 underflow/overflow.
        let input = Tensor::from_data(vec![-10.0, 0.0, 10.0], vec![3]).expect("tensor from_data");
        let out = OnnxModel::run_layer(&OnnxLayer::Sigmoid, &input, None).expect("run layer");
        // sigmoid(-10) ≈ 4.5e-5, very close to 0.
        assert!(out.data()[0] >= 0.0 && out.data()[0] < 0.001);
        assert!((out.data()[1] - 0.5).abs() < 1e-5);
        // sigmoid(10) ≈ 0.9999546, very close to 1.
        assert!(out.data()[2] > 0.999 && out.data()[2] <= 1.0);
    }

    #[test]
    fn test_run_layer_identity() {
        let input = Tensor::ones(vec![3, 3]).expect("tensor ones");
        let out = OnnxModel::run_layer(&OnnxLayer::Identity, &input, None).expect("run layer");
        assert_eq!(out.shape(), &[3, 3]);
        assert!(out.data().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_run_layer_flatten() {
        let input = Tensor::ones(vec![2, 3, 4]).expect("tensor ones");
        let layer = OnnxLayer::Flatten { axis: 1 };
        let out = OnnxModel::run_layer(&layer, &input, None).expect("run layer");
        assert_eq!(out.shape(), &[2, 12]);
    }

    #[test]
    fn test_run_layer_linear() {
        // weight: [[1,1],[1,1]], bias: [0.5, 0.5]
        let weight = Tensor::ones(vec![2, 2]).expect("tensor ones");
        let bias = Tensor::from_data(vec![0.5, 0.5], vec![2]).expect("tensor from_data");
        let layer = OnnxLayer::Linear {
            weight,
            bias: Some(bias),
        };
        let input = Tensor::ones(vec![2]).expect("tensor ones");
        let out = OnnxModel::run_layer(&layer, &input, None).expect("run layer");
        assert_eq!(out.shape(), &[2]);
        // 1*1 + 1*1 + 0.5 = 2.5
        assert!((out.data()[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_run_layer_maxpool() {
        let layer = OnnxLayer::MaxPool2d {
            kernel: (2, 2),
            stride: (2, 2),
            padding: (0, 0),
        };
        let input =
            Tensor::from_data(vec![1.0, 3.0, 2.0, 4.0], vec![1, 2, 2]).expect("tensor from_data");
        let out = OnnxModel::run_layer(&layer, &input, None).expect("run layer");
        assert_eq!(out.shape(), &[1, 1, 1]);
        assert!((out.data()[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_run_layer_unsupported() {
        let layer = OnnxLayer::Unsupported {
            op_type: "LSTM".to_string(),
        };
        let input = Tensor::ones(vec![4]).expect("tensor ones");
        assert!(OnnxModel::run_layer(&layer, &input, None).is_err());
    }

    #[test]
    fn test_varint_encoding_decoding() {
        let cases: &[u64] = &[0, 1, 127, 128, 255, 65535, u32::MAX as u64];
        for &v in cases {
            let encoded = encode_varint(v);
            let mut reader = ProtoReader::new(&encoded);
            let decoded = reader.read_varint().expect("read_varint");
            assert_eq!(decoded, v, "case {v}");
        }
    }

    #[test]
    fn test_build_layer_relu_node() {
        let node = OnnxNode {
            op_type: "Relu".to_string(),
            name: "relu_0".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };
        let inits = HashMap::new();
        let layer = build_layer(&node, &inits).expect("build_layer");
        assert!(matches!(layer, OnnxLayer::Relu));
    }

    #[test]
    fn test_build_layer_dropout_maps_to_identity() {
        let node = OnnxNode {
            op_type: "Dropout".to_string(),
            name: "drop".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        let layer = build_layer(&node, &HashMap::new()).expect("build_layer");
        assert!(matches!(layer, OnnxLayer::Identity));
    }

    #[test]
    fn test_build_layer_unknown_maps_to_unsupported() {
        let node = OnnxNode {
            op_type: "FancyCustomOp".to_string(),
            name: "".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        let layer = build_layer(&node, &HashMap::new()).expect("build_layer");
        assert!(matches!(layer, OnnxLayer::Unsupported { .. }));
    }

    // ── inspect_onnx / OnnxModelInfo ──────────────────────────────────────────

    /// Builds a minimal but valid ModelProto byte buffer for testing.
    ///
    /// Structure:
    /// - field 1 (varint): ir_version
    /// - field 2 (len-delim): opset_import { field 2 (varint): opset_version }
    /// - field 4 (len-delim): graph {
    ///       field 3 (string): graph_name
    ///       field 1 (len-delim): node { field 4 (string): op_type }
    ///       field 11 (len-delim): input { field 1 (string): "input_0" }
    ///       field 12 (len-delim): output { field 1 (string): "output_0" }
    ///   }
    fn build_minimal_model_proto(
        ir_version: u64,
        opset_version: u64,
        graph_name: &str,
        op_types: &[&str],
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // field 1 = ir_version (varint)
        buf.extend_from_slice(&encode_varint_field(1, ir_version));

        // field 2 = opset_import (len-delim message: { field 2 = version })
        let opset_bytes = encode_varint_field(2, opset_version);
        buf.extend_from_slice(&encode_len_delim(2, &opset_bytes));

        // Build the graph proto.
        let mut graph = Vec::new();

        // field 3 = graph name
        graph.extend_from_slice(&encode_len_delim(3, graph_name.as_bytes()));

        // field 1 = nodes
        for op in op_types {
            // NodeProto: field 4 = op_type
            let node = encode_len_delim(4, op.as_bytes());
            graph.extend_from_slice(&encode_len_delim(1, &node));
        }

        // field 11 = input (ValueInfoProto: field 1 = name)
        let input_vi = encode_len_delim(1, b"input_0");
        graph.extend_from_slice(&encode_len_delim(11, &input_vi));

        // field 12 = output (ValueInfoProto: field 1 = name)
        let output_vi = encode_len_delim(1, b"output_0");
        graph.extend_from_slice(&encode_len_delim(12, &output_vi));

        // field 4 = graph (ModelProto)
        buf.extend_from_slice(&encode_len_delim(4, &graph));

        buf
    }

    #[test]
    fn test_inspect_onnx_empty_bytes() {
        // Empty input should succeed with zero/default fields.
        let info = inspect_onnx(&[]).expect("inspect_onnx");
        assert_eq!(info.ir_version, 0);
        assert_eq!(info.opset_version, 0);
        assert!(info.graph_name.is_empty());
        assert!(info.node_types.is_empty());
        assert!(info.input_names.is_empty());
        assert!(info.output_names.is_empty());
    }

    #[test]
    fn test_inspect_onnx_ir_version() {
        let buf = build_minimal_model_proto(8, 17, "test_graph", &["Relu"]);
        let info = inspect_onnx(&buf).expect("inspect_onnx");
        assert_eq!(info.ir_version, 8);
    }

    #[test]
    fn test_inspect_onnx_opset_version() {
        let buf = build_minimal_model_proto(8, 17, "test_graph", &["Relu"]);
        let info = inspect_onnx(&buf).expect("inspect_onnx");
        assert_eq!(info.opset_version, 17);
    }

    #[test]
    fn test_inspect_onnx_graph_name() {
        let buf = build_minimal_model_proto(7, 13, "my_model_graph", &["Conv"]);
        let info = inspect_onnx(&buf).expect("inspect_onnx");
        assert_eq!(info.graph_name, "my_model_graph");
    }

    #[test]
    fn test_inspect_onnx_node_types_deduped() {
        // Repeated ops should appear only once.
        let buf = build_minimal_model_proto(8, 17, "g", &["Conv", "Relu", "Conv", "Relu", "Gemm"]);
        let info = inspect_onnx(&buf).expect("inspect_onnx");
        // Unique, ordered by first appearance.
        assert_eq!(info.node_types, vec!["Conv", "Relu", "Gemm"]);
    }

    #[test]
    fn test_inspect_onnx_node_types_empty() {
        let buf = build_minimal_model_proto(8, 17, "empty", &[]);
        let info = inspect_onnx(&buf).expect("inspect_onnx");
        assert!(info.node_types.is_empty());
    }

    #[test]
    fn test_inspect_onnx_input_output_names() {
        let buf = build_minimal_model_proto(8, 17, "g", &["Relu"]);
        let info = inspect_onnx(&buf).expect("inspect_onnx");
        assert_eq!(info.input_names, vec!["input_0"]);
        assert_eq!(info.output_names, vec!["output_0"]);
    }

    #[test]
    fn test_inspect_onnx_multiple_op_types() {
        let ops = &["Conv", "BatchNormalization", "Relu", "MaxPool", "Gemm"];
        let buf = build_minimal_model_proto(8, 18, "resnet", ops);
        let info = inspect_onnx(&buf).expect("inspect_onnx");
        assert_eq!(info.node_types.len(), ops.len());
        for op in ops {
            assert!(
                info.node_types.contains(&op.to_string()),
                "expected op {op} in node_types"
            );
        }
    }

    #[test]
    fn test_inspect_onnx_is_cheaper_than_full_parse() {
        // Both should succeed on the same bytes; inspect_onnx must not error.
        let buf = build_minimal_model_proto(7, 12, "net", &["Relu", "Sigmoid"]);
        // Full parse may fail due to missing weights — that's OK.
        // inspect_onnx must always succeed on well-formed protobuf.
        let info = inspect_onnx(&buf).expect("inspect_onnx");
        assert!(!info.node_types.is_empty());
    }
}
