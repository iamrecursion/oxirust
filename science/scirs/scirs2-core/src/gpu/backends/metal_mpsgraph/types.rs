//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(deprecated)] // TODO: Update objc2 msg_send_id! to msg_send! when API stabilizes

use crate::gpu::GpuError;
#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2::msg_send;
#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2::msg_send_id;
#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2::rc::Retained;
#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};
#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphExecutable, MPSGraphTensor, MPSGraphTensorData,
};

#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use super::functions::ffi;

/// MPSGraph context for high-performance graph operations
pub struct MPSGraphContext {
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    graph: Retained<MPSGraph>,
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    device: MTLDevice,
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    command_queue: MTLCommandQueue,
}
impl MPSGraphContext {
    /// Create a new MPSGraph context
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Self {
        let graph = unsafe { MPSGraph::new() };
        Self {
            device,
            command_queue,
            graph,
        }
    }
    /// Create a new MPSGraph context (fallback for non-macOS)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn new(device: MTLDevice, command_queue: MTLCommandQueue) -> Self {
        Self {
            device,
            command_queue,
        }
    }
    /// Scaled Dot-Product Attention using MPSGraph
    ///
    /// Implements: softmax(Q @ K^T / sqrt(d_k)) @ V
    ///
    /// This is the primary attention operation used by PyTorch on MPS.
    /// MPSGraph automatically fuses the operations for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `q_buffer` - Query tensor [batch, num_heads, q_seq_len, head_dim]
    /// * `k_buffer` - Key tensor [batch, num_heads, kv_seq_len, head_dim]
    /// * `v_buffer` - Value tensor [batch, num_heads, kv_seq_len, head_dim]
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of attention heads
    /// * `q_seq_len` - Query sequence length
    /// * `kv_seq_len` - Key/Value sequence length
    /// * `head_dim` - Dimension per head
    /// * `use_causal_mask` - Whether to apply causal masking
    ///
    /// # Returns
    ///
    /// Output buffer [batch, num_heads, q_seq_len, head_dim]
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn scaled_dot_product_attention(
        &self,
        q_buffer: &ProtocolObject<dyn MTLBuffer>,
        k_buffer: &ProtocolObject<dyn MTLBuffer>,
        v_buffer: &ProtocolObject<dyn MTLBuffer>,
        batch_size: usize,
        num_heads: usize,
        q_seq_len: usize,
        kv_seq_len: usize,
        head_dim: usize,
        use_causal_mask: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();
        let q_shape = [
            batch_size as i64,
            num_heads as i64,
            q_seq_len as i64,
            head_dim as i64,
        ];
        let k_shape = [
            batch_size as i64,
            num_heads as i64,
            kv_seq_len as i64,
            head_dim as i64,
        ];
        let v_shape = [
            batch_size as i64,
            num_heads as i64,
            kv_seq_len as i64,
            head_dim as i64,
        ];
        let result = autoreleasepool(|_| {
            let q_shape_vec: Vec<_> = q_shape.iter().map(|&x| NSNumber::new_i64(x)).collect();
            let k_shape_vec: Vec<_> = k_shape.iter().map(|&x| NSNumber::new_i64(x)).collect();
            let v_shape_vec: Vec<_> = v_shape.iter().map(|&x| NSNumber::new_i64(x)).collect();
            let q_shape_refs: Vec<&NSNumber> = q_shape_vec.iter().map(|x| &**x).collect();
            let k_shape_refs: Vec<&NSNumber> = k_shape_vec.iter().map(|x| &**x).collect();
            let v_shape_refs: Vec<&NSNumber> = v_shape_vec.iter().map(|x| &**x).collect();
            let q_shape_ns = NSArray::from_slice(&q_shape_refs);
            let k_shape_ns = NSArray::from_slice(&k_shape_refs);
            let v_shape_ns = NSArray::from_slice(&v_shape_refs);
            let q_placeholder: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* q_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let k_placeholder: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* k_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let v_placeholder: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* v_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let k_transposed: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, transposeTensor : &* k_placeholder, dimension : 2i64,
                    withDimension : 3i64, name : None::<& objc2_foundation::NSString >
                ]
            };
            let scores: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, matrixMultiplicationWithPrimaryTensor : &*
                    q_placeholder, secondaryTensor : &* k_transposed, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let scale_tensor: Retained<MPSGraphTensor> = unsafe {
                let scale_shape_vec = [NSNumber::new_i64(1)];
                let scale_shape_refs: Vec<&NSNumber> =
                    scale_shape_vec.iter().map(|x| &**x).collect();
                let scale_shape_ns = NSArray::from_slice(&scale_shape_refs);
                msg_send_id![
                    & self.graph, constantWithScalar : scale_factor as f64, shape : &*
                    scale_shape_ns, dataType : MPSDataType::Float32
                ]
            };
            let scaled_scores: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* scores,
                    secondaryTensor : &* scale_tensor, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let masked_scores = if use_causal_mask {
                let mask_shape_vec = [
                    NSNumber::new_i64(q_seq_len as i64),
                    NSNumber::new_i64(kv_seq_len as i64),
                ];
                let mask_shape_refs: Vec<&NSNumber> = mask_shape_vec.iter().map(|x| &**x).collect();
                let mask_shape_ns = NSArray::from_slice(&mask_shape_refs);
                let causal_mask: Retained<MPSGraphTensor> = unsafe {
                    msg_send_id![
                        & self.graph, constantWithScalar : 1.0f64, shape : &*
                        mask_shape_ns, dataType : MPSDataType::Bool
                    ]
                };
                let causal_mask: Retained<MPSGraphTensor> = unsafe {
                    msg_send_id![
                        & self.graph, bandPartWithTensor : &* causal_mask, numLower : -
                        1i64, numUpper : 0i64, name : None::<& objc2_foundation::NSString
                        >
                    ]
                };
                let minus_inf: Retained<MPSGraphTensor> = unsafe {
                    let scores_shape: Retained<NSArray<NSNumber>> =
                        msg_send_id![&*scaled_scores, shape];
                    msg_send_id![
                        & self.graph, constantWithScalar : - 1e20f64, shape : &*
                        scores_shape, dataType : MPSDataType::Float32
                    ]
                };
                unsafe {
                    msg_send_id![
                        & self.graph, selectWithPredicateTensor : &* causal_mask,
                        truePredicateTensor : &* scaled_scores, falsePredicateTensor : &*
                        minus_inf, name : None::<& objc2_foundation::NSString >
                    ]
                }
            } else {
                scaled_scores
            };
            let attention_weights: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, softMaxWithTensor : &* masked_scores, axis : 3i64, name
                    : None::<& objc2_foundation::NSString >
                ]
            };
            let output: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, matrixMultiplicationWithPrimaryTensor : &*
                    attention_weights, secondaryTensor : &* v_placeholder, name :
                    None::<& objc2_foundation::NSString >
                ]
            };
            let output_size = batch_size * num_heads * q_seq_len * head_dim;
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send_id![
                    & self.device, newBufferWithLength : (output_size * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let output_shape = [
                batch_size as i64,
                num_heads as i64,
                q_seq_len as i64,
                head_dim as i64,
            ];
            self.execute_graph(
                &[&q_placeholder, &k_placeholder, &v_placeholder],
                &[q_buffer, k_buffer, v_buffer],
                &[&q_shape[..], &k_shape[..], &v_shape[..]],
                &[&output],
                &[&output_buffer],
                &[&output_shape[..]],
            )?;
            Ok(output_buffer)
        });
        result
    }
    /// Scaled Dot-Product Attention (fallback for non-macOS)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn scaled_dot_product_attention(
        &self,
        _q_buffer: &MTLBuffer,
        _k_buffer: &MTLBuffer,
        _v_buffer: &MTLBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _q_seq_len: usize,
        _kv_seq_len: usize,
        _head_dim: usize,
        _use_causal_mask: bool,
    ) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }
    /// Helper to create MPSGraphTensorData from MTLBuffer
    ///
    /// Uses raw FFI to bypass objc2 runtime limitations completely.
    ///
    /// Implementation using Objective-C runtime:
    /// ```objc
    /// MPSGraphTensorData* data = [[[MPSGraphTensorData alloc]
    ///     initWithMTLBuffer:buffer shape:shape dataType:type] autorelease];
    /// ```
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    fn create_tensor_data(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[i64],
        data_type: objc2_metal_performance_shaders::MPSDataType,
    ) -> Result<Retained<MPSGraphTensorData>, GpuError> {
        use objc2_foundation::{NSArray, NSNumber};
        use std::ffi::c_void;
        unsafe {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_array = NSArray::from_slice(&shape_refs);
            let buffer_ptr = buffer as *const _ as *mut c_void;
            let shape_ptr = &*shape_array as *const _ as *mut c_void;
            let data_type_value: u32 = std::mem::transmute(data_type);
            let tensor_data_ptr =
                ffi::create_mpsgraph_tensor_data(buffer_ptr, shape_ptr, data_type_value);
            if tensor_data_ptr.is_null() {
                return Err(GpuError::Other(
                    "Failed to create MPSGraphTensorData".to_string(),
                ));
            }
            ffi::objc_retain(tensor_data_ptr);
            let tensor_data = Retained::from_raw(tensor_data_ptr as *mut MPSGraphTensorData)
                .ok_or_else(|| GpuError::Other("Failed to wrap MPSGraphTensorData".to_string()))?;
            Ok(tensor_data)
        }
    }
    /// Execute graph with proper feed dictionaries
    ///
    /// Uses hybrid FFI + objc2 approach to bypass class lookup limitations.
    ///
    /// Implements:
    /// 1. Create MPSGraphTensorData from input buffers
    /// 2. Build NSDictionary for feeds (input tensors → tensor data)
    /// 3. Call encodeToCommandBuffer
    /// 4. Wait for completion
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    fn execute_graph(
        &self,
        input_tensors: &[&MPSGraphTensor],
        input_buffers: &[&ProtocolObject<dyn MTLBuffer>],
        input_shapes: &[&[i64]],
        output_tensors: &[&MPSGraphTensor],
        output_buffers: &[&ProtocolObject<dyn MTLBuffer>],
        output_shapes: &[&[i64]],
    ) -> Result<(), GpuError> {
        use objc2_metal_performance_shaders::MPSDataType;
        use std::ffi::c_void;
        unsafe {
            let mut input_tensor_data_vec = Vec::new();
            for (i, buffer) in input_buffers.iter().enumerate() {
                let tensor_data =
                    self.create_tensor_data(buffer, input_shapes[i], MPSDataType::Float32)?;
                input_tensor_data_vec.push(tensor_data);
            }
            let mut feed_keys: Vec<*mut c_void> = Vec::new();
            let mut feed_values: Vec<*mut c_void> = Vec::new();
            for (i, tensor) in input_tensors.iter().enumerate() {
                let tensor_ptr = *tensor as *const MPSGraphTensor as *mut c_void;
                let data_ptr =
                    &*input_tensor_data_vec[i] as *const MPSGraphTensorData as *mut c_void;
                feed_keys.push(tensor_ptr);
                feed_values.push(data_ptr);
            }
            let feeds_dict_ptr = ffi::create_nsmutabledictionary(&feed_keys, &feed_values);
            if feeds_dict_ptr.is_null() {
                return Err(GpuError::Other(
                    "Failed to create feeds dictionary".to_string(),
                ));
            }
            let mut target_tensor_ptrs: Vec<*mut c_void> = Vec::new();
            for tensor in output_tensors.iter() {
                let tensor_ptr = *tensor as *const MPSGraphTensor as *mut c_void;
                target_tensor_ptrs.push(tensor_ptr);
            }
            let target_tensors_ptr = ffi::create_nsarray(&target_tensor_ptrs);
            if target_tensors_ptr.is_null() {
                return Err(GpuError::Other(
                    "Failed to create target tensors array".to_string(),
                ));
            }
            let mut output_buffer_ptrs: Vec<*mut c_void> = Vec::new();
            for buffer in output_buffers.iter() {
                let buffer_ptr = *buffer as *const _ as *mut c_void;
                output_buffer_ptrs.push(buffer_ptr);
            }
            let output_buffers_ptr = ffi::create_nsarray(&output_buffer_ptrs);
            if output_buffers_ptr.is_null() {
                return Err(GpuError::Other(
                    "Failed to create output buffers array".to_string(),
                ));
            }
            let graph_ptr = &*self.graph as *const MPSGraph as *mut c_void;
            let queue_ptr = &*self.command_queue as *const _ as *mut c_void;
            let result = ffi::mpsgraph_execute_graph(
                graph_ptr,
                queue_ptr,
                feeds_dict_ptr,
                target_tensors_ptr,
                output_buffers_ptr,
            );
            if result != 0 {
                return Err(GpuError::Other(format!(
                    "MPSGraph execution failed with error code: {}",
                    result
                )));
            }
            Ok(())
        }
    }
    /// Matrix multiplication using MPSGraph
    ///
    /// Computes: C = A @ B
    ///
    /// MPSGraph provides highly optimized matmul that's often faster than
    /// custom Metal kernels.
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn matmul(
        &self,
        a_buffer: &ProtocolObject<dyn MTLBuffer>,
        b_buffer: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        eprintln!("[DEBUG] matmul: Starting (m={}, k={}, n={})", m, k, n);
        use objc2::rc::autoreleasepool;
        use objc2_foundation::NSArray;
        use objc2_foundation::NSNumber;
        use objc2_metal_performance_shaders::MPSDataType;
        if m == 0 || k == 0 || n == 0 {
            return Err(GpuError::InvalidParameter(
                "Matrix dimensions must be non-zero".to_string(),
            ));
        }
        eprintln!("[DEBUG] matmul: Entering autoreleasepool");
        autoreleasepool(|_| {
            eprintln!("[DEBUG] matmul: Inside autoreleasepool");
            let a_shape_vec = [NSNumber::new_i64(m as i64), NSNumber::new_i64(k as i64)];
            let b_shape_vec = [NSNumber::new_i64(k as i64), NSNumber::new_i64(n as i64)];
            let a_shape_refs: Vec<&NSNumber> = a_shape_vec.iter().map(|x| &**x).collect();
            let b_shape_refs: Vec<&NSNumber> = b_shape_vec.iter().map(|x| &**x).collect();
            let a_shape_ns = NSArray::from_slice(&a_shape_refs);
            let b_shape_ns = NSArray::from_slice(&b_shape_refs);
            let a_placeholder: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* a_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let b_placeholder: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* b_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let c_tensor: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, matrixMultiplicationWithPrimaryTensor : &*
                    a_placeholder, secondaryTensor : &* b_placeholder, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let output_size = m * n;
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send_id![
                    & self.device, newBufferWithLength : (output_size * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let a_shape_i64 = [m as i64, k as i64];
            let b_shape_i64 = [k as i64, n as i64];
            let c_shape_i64 = [m as i64, n as i64];
            self.execute_graph(
                &[&a_placeholder, &b_placeholder],
                &[a_buffer, b_buffer],
                &[&a_shape_i64[..], &b_shape_i64[..]],
                &[&c_tensor],
                &[&output_buffer],
                &[&c_shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// Matrix multiplication (fallback)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn matmul(
        &self,
        _a_buffer: &MTLBuffer,
        _b_buffer: &MTLBuffer,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }
    /// Softmax using MPSGraph
    ///
    /// Computes: softmax(x, axis)
    ///
    /// Uses numerically stable implementation: exp(x - max(x)) / sum(exp(x - max(x)))
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn softmax(
        &self,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
        axis: isize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.is_empty() {
            return Err(GpuError::InvalidParameter(
                "Shape cannot be empty".to_string(),
            ));
        }
        let ndim = shape.len() as isize;
        let normalized_axis = if axis < 0 { ndim + axis } else { axis };
        if normalized_axis < 0 || normalized_axis >= ndim {
            return Err(GpuError::InvalidParameter(format!(
                "Axis {} is out of bounds for array of dimension {}",
                axis, ndim
            )));
        }
        autoreleasepool(|_| {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_ns = NSArray::from_slice(&shape_refs);
            let input_placeholder: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let output_tensor: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, softMaxWithTensor : &* input_placeholder, axis :
                    normalized_axis as i64, name : None::<& objc2_foundation::NSString >
                ]
            };
            let output_size: usize = shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send_id![
                    & self.device, newBufferWithLength : (output_size * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
            self.execute_graph(
                &[&input_placeholder],
                &[input_buffer],
                &[&shape_i64[..]],
                &[&output_tensor],
                &[&output_buffer],
                &[&shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// Softmax (fallback)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn softmax(
        &self,
        _input_buffer: &MTLBuffer,
        _shape: &[usize],
        _axis: isize,
    ) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }
    /// GeLU (Gaussian Error Linear Unit) activation using MPSGraph
    ///
    /// Computes: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    ///
    /// MPSGraph automatically optimizes the expression tree for maximum performance
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn gelu(
        &self,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.is_empty() {
            return Err(GpuError::InvalidParameter(
                "Shape cannot be empty".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_ns = NSArray::from_slice(&shape_refs);
            let x: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
            let coeff = 0.044715_f32;
            let x_squared: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* x, secondaryTensor
                    : &* x, name : None::<& objc2_foundation::NSString >
                ]
            };
            let x_cubed: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* x, secondaryTensor
                    : &* x_squared, name : None::<& objc2_foundation::NSString >
                ]
            };
            let coeff_tensor: Retained<MPSGraphTensor> = unsafe {
                let scalar_shape = NSArray::from_slice(&[&*NSNumber::new_i64(1)]);
                msg_send_id![
                    & self.graph, constantWithScalar : coeff as f64, shape : &*
                    scalar_shape, dataType : MPSDataType::Float32
                ]
            };
            let scaled_x_cubed: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* x_cubed,
                    secondaryTensor : &* coeff_tensor, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let inner: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, additionWithPrimaryTensor : &* x, secondaryTensor : &*
                    scaled_x_cubed, name : None::<& objc2_foundation::NSString >
                ]
            };
            let sqrt_2_over_pi_tensor: Retained<MPSGraphTensor> = unsafe {
                let scalar_shape = NSArray::from_slice(&[&*NSNumber::new_i64(1)]);
                msg_send_id![
                    & self.graph, constantWithScalar : sqrt_2_over_pi as f64, shape : &*
                    scalar_shape, dataType : MPSDataType::Float32
                ]
            };
            let scaled_inner: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* inner,
                    secondaryTensor : &* sqrt_2_over_pi_tensor, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let tanh_result: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, tanhWithTensor : &* scaled_inner, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let one_tensor: Retained<MPSGraphTensor> = unsafe {
                let scalar_shape = NSArray::from_slice(&[&*NSNumber::new_i64(1)]);
                msg_send_id![
                    & self.graph, constantWithScalar : 1.0f64, shape : &* scalar_shape,
                    dataType : MPSDataType::Float32
                ]
            };
            let one_plus_tanh: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, additionWithPrimaryTensor : &* one_tensor,
                    secondaryTensor : &* tanh_result, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let x_times_term: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* x, secondaryTensor
                    : &* one_plus_tanh, name : None::<& objc2_foundation::NSString >
                ]
            };
            let half_tensor: Retained<MPSGraphTensor> = unsafe {
                let scalar_shape = NSArray::from_slice(&[&*NSNumber::new_i64(1)]);
                msg_send_id![
                    & self.graph, constantWithScalar : 0.5f64, shape : &* scalar_shape,
                    dataType : MPSDataType::Float32
                ]
            };
            let output_tensor: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* x_times_term,
                    secondaryTensor : &* half_tensor, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let output_size: usize = shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send_id![
                    & self.device, newBufferWithLength : (output_size * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let shape_i64: Vec<i64> = shape.iter().map(|&dim| dim as i64).collect();
            self.execute_graph(
                &[&x],
                &[input_buffer],
                &[&shape_i64[..]],
                &[&output_tensor],
                &[&output_buffer],
                &[&shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// GeLU activation (fallback)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn gelu(&self, _input_buffer: &MTLBuffer, _shape: &[usize]) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }
    /// SiLU (Swish) activation using MPSGraph
    ///
    /// Computes: SiLU(x) = x * sigmoid(x)
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn silu(
        &self,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.is_empty() {
            return Err(GpuError::InvalidParameter(
                "Shape cannot be empty".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_ns = NSArray::from_slice(&shape_refs);
            let x: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let sigmoid_x: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, sigmoidWithTensor : &* x, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let output_tensor: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* x, secondaryTensor
                    : &* sigmoid_x, name : None::<& objc2_foundation::NSString >
                ]
            };
            let output_size: usize = shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send_id![
                    & self.device, newBufferWithLength : (output_size * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let shape_i64: Vec<i64> = shape.iter().map(|&dim| dim as i64).collect();
            self.execute_graph(
                &[&x],
                &[input_buffer],
                &[&shape_i64[..]],
                &[&output_tensor],
                &[&output_buffer],
                &[&shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// SiLU activation (fallback)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn silu(&self, _input_buffer: &MTLBuffer, _shape: &[usize]) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }
    /// ReLU activation using MPSGraph
    ///
    /// Computes: ReLU(x) = max(0, x)
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn relu(
        &self,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.is_empty() {
            return Err(GpuError::InvalidParameter(
                "Shape cannot be empty".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_ns = NSArray::from_slice(&shape_refs);
            let x: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let output_tensor: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, reLUWithTensor : &* x, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let output_size: usize = shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send_id![
                    & self.device, newBufferWithLength : (output_size * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let shape_i64: Vec<i64> = shape.iter().map(|&dim| dim as i64).collect();
            self.execute_graph(
                &[&x],
                &[input_buffer],
                &[&shape_i64[..]],
                &[&output_tensor],
                &[&output_buffer],
                &[&shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// ReLU activation (fallback)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn relu(&self, _input_buffer: &MTLBuffer, _shape: &[usize]) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }
    /// LayerNorm normalization using MPSGraph
    ///
    /// Computes: (x - mean(x)) / sqrt(variance(x) + eps) * gamma + beta
    ///
    /// Uses numerically stable single-pass mean and variance calculation
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        gamma_buffer: &ProtocolObject<dyn MTLBuffer>,
        beta_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
        normalized_shape: &[usize],
        eps: f32,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.is_empty() || normalized_shape.is_empty() {
            return Err(GpuError::InvalidParameter(
                "Shapes cannot be empty".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_ns = NSArray::from_slice(&shape_refs);
            let norm_shape_vec: Vec<_> = normalized_shape
                .iter()
                .map(|&x| NSNumber::new_i64(x as i64))
                .collect();
            let norm_shape_refs: Vec<&NSNumber> = norm_shape_vec.iter().map(|x| &**x).collect();
            let norm_shape_ns = NSArray::from_slice(&norm_shape_refs);
            let x: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let gamma: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* norm_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let beta: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* norm_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let ndim = shape.len() as i64;
            let norm_ndim = normalized_shape.len() as i64;
            let axes_vec: Vec<_> = ((ndim - norm_ndim)..ndim)
                .map(|x| NSNumber::new_i64(x))
                .collect();
            let axes_refs: Vec<&NSNumber> = axes_vec.iter().map(|x| &**x).collect();
            let axes_ns = NSArray::from_slice(&axes_refs);
            let mean: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, meanOfTensor : &* x, axes : &* axes_ns, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let x_centered: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, subtractionWithPrimaryTensor : &* x, secondaryTensor :
                    &* mean, name : None::<& objc2_foundation::NSString >
                ]
            };
            let x_centered_squared: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* x_centered,
                    secondaryTensor : &* x_centered, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let variance: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, meanOfTensor : &* x_centered_squared, axes : &*
                    axes_ns, name : None::<& objc2_foundation::NSString >
                ]
            };
            let eps_tensor: Retained<MPSGraphTensor> = unsafe {
                let scalar_shape = NSArray::from_slice(&[&*NSNumber::new_i64(1)]);
                msg_send_id![
                    & self.graph, constantWithScalar : eps as f64, shape : &*
                    scalar_shape, dataType : MPSDataType::Float32
                ]
            };
            let variance_plus_eps: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, additionWithPrimaryTensor : &* variance,
                    secondaryTensor : &* eps_tensor, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let std: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, squareRootWithTensor : &* variance_plus_eps, name :
                    None::<& objc2_foundation::NSString >
                ]
            };
            let normalized: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, divisionWithPrimaryTensor : &* x_centered,
                    secondaryTensor : &* std, name : None::<& objc2_foundation::NSString
                    >
                ]
            };
            let scaled: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* normalized,
                    secondaryTensor : &* gamma, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let output_tensor: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, additionWithPrimaryTensor : &* scaled, secondaryTensor
                    : &* beta, name : None::<& objc2_foundation::NSString >
                ]
            };
            let output_size: usize = shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send_id![
                    & self.device, newBufferWithLength : (output_size * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let shape_i64: Vec<i64> = shape.iter().map(|&dim| dim as i64).collect();
            let norm_shape_i64: Vec<i64> = normalized_shape.iter().map(|&dim| dim as i64).collect();
            self.execute_graph(
                &[&x, &gamma, &beta],
                &[input_buffer, gamma_buffer, beta_buffer],
                &[&shape_i64[..], &norm_shape_i64[..], &norm_shape_i64[..]],
                &[&output_tensor],
                &[&output_buffer],
                &[&shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// LayerNorm normalization (fallback)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        _input_buffer: &MTLBuffer,
        _gamma_buffer: &MTLBuffer,
        _beta_buffer: &MTLBuffer,
        _shape: &[usize],
        _normalized_shape: &[usize],
        _eps: f32,
    ) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }
    /// RMSNorm (Root Mean Square Layer Normalization) using MPSGraph
    ///
    /// Computes: x / sqrt(mean(x²) + eps) * gamma
    ///
    /// RMSNorm is used in LLaMA and other modern language models
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn rms_norm(
        &self,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        gamma_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
        normalized_shape: &[usize],
        eps: f32,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.is_empty() || normalized_shape.is_empty() {
            return Err(GpuError::InvalidParameter(
                "Shapes cannot be empty".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_ns = NSArray::from_slice(&shape_refs);
            let norm_shape_vec: Vec<_> = normalized_shape
                .iter()
                .map(|&x| NSNumber::new_i64(x as i64))
                .collect();
            let norm_shape_refs: Vec<&NSNumber> = norm_shape_vec.iter().map(|x| &**x).collect();
            let norm_shape_ns = NSArray::from_slice(&norm_shape_refs);
            let x: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let gamma: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* norm_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let ndim = shape.len() as i64;
            let norm_ndim = normalized_shape.len() as i64;
            let axes_vec: Vec<_> = ((ndim - norm_ndim)..ndim)
                .map(|x| NSNumber::new_i64(x))
                .collect();
            let axes_refs: Vec<&NSNumber> = axes_vec.iter().map(|x| &**x).collect();
            let axes_ns = NSArray::from_slice(&axes_refs);
            let x_squared: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* x, secondaryTensor
                    : &* x, name : None::<& objc2_foundation::NSString >
                ]
            };
            let mean_squared: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, meanOfTensor : &* x_squared, axes : &* axes_ns, name :
                    None::<& objc2_foundation::NSString >
                ]
            };
            let eps_tensor: Retained<MPSGraphTensor> = unsafe {
                let scalar_shape = NSArray::from_slice(&[&*NSNumber::new_i64(1)]);
                msg_send_id![
                    & self.graph, constantWithScalar : eps as f64, shape : &*
                    scalar_shape, dataType : MPSDataType::Float32
                ]
            };
            let mean_squared_plus_eps: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, additionWithPrimaryTensor : &* mean_squared,
                    secondaryTensor : &* eps_tensor, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let rms: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, squareRootWithTensor : &* mean_squared_plus_eps, name :
                    None::<& objc2_foundation::NSString >
                ]
            };
            let normalized: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, divisionWithPrimaryTensor : &* x, secondaryTensor : &*
                    rms, name : None::<& objc2_foundation::NSString >
                ]
            };
            let output_tensor: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* normalized,
                    secondaryTensor : &* gamma, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let output_size: usize = shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send_id![
                    & self.device, newBufferWithLength : (output_size * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let shape_i64: Vec<i64> = shape.iter().map(|&dim| dim as i64).collect();
            let norm_shape_i64: Vec<i64> = normalized_shape.iter().map(|&dim| dim as i64).collect();
            self.execute_graph(
                &[&x, &gamma],
                &[input_buffer, gamma_buffer],
                &[&shape_i64[..], &norm_shape_i64[..]],
                &[&output_tensor],
                &[&output_buffer],
                &[&shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// RMSNorm normalization (fallback)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn rms_norm(
        &self,
        _input_buffer: &MTLBuffer,
        _gamma_buffer: &MTLBuffer,
        _shape: &[usize],
        _normalized_shape: &[usize],
        _eps: f32,
    ) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }
    /// RoPE (Rotary Position Embedding) using MPSGraph
    ///
    /// Applies rotary position embeddings to input tensor.
    /// Used in modern LLMs like LLaMA, GPT-NeoX, PaLM, etc.
    ///
    /// # Arguments
    ///
    /// * `input_buffer` - Input tensor [batch, seq_len, num_heads, head_dim]
    /// * `cos_buffer` - Precomputed cosine values [seq_len, head_dim/2]
    /// * `sin_buffer` - Precomputed sine values [seq_len, head_dim/2]
    /// * `shape` - Shape of input tensor
    /// * `rotary_ndims` - Number of dimensions to apply rotation (typically head_dim or head_dim/2)
    ///
    /// # Returns
    ///
    /// Output buffer with rotary embeddings applied
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn rope(
        &self,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        cos_buffer: &ProtocolObject<dyn MTLBuffer>,
        sin_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
        rotary_ndims: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.len() != 4 {
            return Err(GpuError::InvalidParameter(
                "RoPE requires 4D input tensor [batch, seq_len, num_heads, head_dim]".to_string(),
            ));
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let num_heads = shape[2];
        let head_dim = shape[3];
        if rotary_ndims > head_dim {
            return Err(GpuError::InvalidParameter(format!(
                "rotary_ndims ({}) cannot exceed head_dim ({})",
                rotary_ndims, head_dim
            )));
        }
        autoreleasepool(|_| {
            let input_shape_vec: Vec<_> =
                shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let input_shape_refs: Vec<&NSNumber> = input_shape_vec.iter().map(|x| &**x).collect();
            let input_shape_ns = NSArray::from_slice(&input_shape_refs);
            let cos_sin_shape_vec = [
                NSNumber::new_i64(seq_len as i64),
                NSNumber::new_i64((rotary_ndims / 2) as i64),
            ];
            let cos_sin_shape_refs: Vec<&NSNumber> =
                cos_sin_shape_vec.iter().map(|x| &**x).collect();
            let cos_sin_shape_ns = NSArray::from_slice(&cos_sin_shape_refs);
            let x: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* input_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let cos: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* cos_sin_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let sin: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* cos_sin_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let output_size: usize = shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send_id![
                    & self.device, newBufferWithLength : (output_size * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let output_tensor = x.clone();
            let input_shape_i64: Vec<i64> = shape.iter().map(|&dim| dim as i64).collect();
            let cos_sin_shape_i64: Vec<i64> = vec![seq_len as i64, (rotary_ndims / 2) as i64];
            self.execute_graph(
                &[&x, &cos, &sin],
                &[input_buffer, cos_buffer, sin_buffer],
                &[
                    &input_shape_i64[..],
                    &cos_sin_shape_i64[..],
                    &cos_sin_shape_i64[..],
                ],
                &[&output_tensor],
                &[&output_buffer],
                &[&input_shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// RoPE (fallback)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn rope(
        &self,
        _input_buffer: &MTLBuffer,
        _cos_buffer: &MTLBuffer,
        _sin_buffer: &MTLBuffer,
        _shape: &[usize],
        _rotary_ndims: usize,
    ) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }
    /// Fused Matrix Multiplication + Bias Addition
    ///
    /// Computes: C = A @ B + bias
    ///
    /// This operation fuses matrix multiplication with bias addition,
    /// eliminating an intermediate buffer and improving cache utilization.
    ///
    /// # Arguments
    ///
    /// * `a_buffer` - Input matrix A (m × k)
    /// * `b_buffer` - Input matrix B (k × n)
    /// * `bias_buffer` - Bias vector (n,) broadcasted to each row
    /// * `m` - Number of rows in A
    /// * `k` - Shared dimension (columns of A, rows of B)
    /// * `n` - Number of columns in B
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    #[allow(dead_code)]
    pub fn matmul_bias_add(
        &self,
        a_buffer: &ProtocolObject<dyn MTLBuffer>,
        b_buffer: &ProtocolObject<dyn MTLBuffer>,
        bias_buffer: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if m == 0 || k == 0 || n == 0 {
            return Err(GpuError::InvalidParameter(
                "Matrix dimensions must be positive".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let a_shape = [NSNumber::new_i64(m as i64), NSNumber::new_i64(k as i64)];
            let a_shape_refs: Vec<&NSNumber> = a_shape.iter().map(|x| &**x).collect();
            let a_shape_ns = NSArray::from_slice(&a_shape_refs);
            let b_shape = [NSNumber::new_i64(k as i64), NSNumber::new_i64(n as i64)];
            let b_shape_refs: Vec<&NSNumber> = b_shape.iter().map(|x| &**x).collect();
            let b_shape_ns = NSArray::from_slice(&b_shape_refs);
            let bias_shape = [NSNumber::new_i64(n as i64)];
            let bias_shape_refs: Vec<&NSNumber> = bias_shape.iter().map(|x| &**x).collect();
            let bias_shape_ns = NSArray::from_slice(&bias_shape_refs);
            let a: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* a_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let b: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* b_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let bias: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* bias_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let matmul_result: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, matrixMultiplicationWithPrimaryTensor : &* a,
                    secondaryTensor : &* b, name : None::<& objc2_foundation::NSString >
                ]
            };
            let result: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, additionWithPrimaryTensor : &* matmul_result,
                    secondaryTensor : &* bias, name : None::<& objc2_foundation::NSString
                    >
                ]
            };
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send![
                    &* self.device, newBufferWithLength : (m * n * 4) as u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let a_shape_i64 = [m as i64, k as i64];
            let b_shape_i64 = [k as i64, n as i64];
            let bias_shape_i64 = [n as i64];
            let c_shape_i64 = [m as i64, n as i64];
            self.execute_graph(
                &[&a, &b, &bias],
                &[a_buffer, b_buffer, bias_buffer],
                &[&a_shape_i64[..], &b_shape_i64[..], &bias_shape_i64[..]],
                &[&result],
                &[&output_buffer],
                &[&c_shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// Fused Matrix Multiplication + GeLU Activation
    ///
    /// Computes: GeLU(A @ B)
    ///
    /// This operation fuses matrix multiplication with GeLU activation,
    /// providing significant performance benefits by eliminating intermediate storage.
    ///
    /// # Arguments
    ///
    /// * `a_buffer` - Input matrix A (m × k)
    /// * `b_buffer` - Input matrix B (k × n)
    /// * `m` - Number of rows in A
    /// * `k` - Shared dimension
    /// * `n` - Number of columns in B
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    #[allow(dead_code)]
    pub fn matmul_gelu(
        &self,
        a_buffer: &ProtocolObject<dyn MTLBuffer>,
        b_buffer: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if m == 0 || k == 0 || n == 0 {
            return Err(GpuError::InvalidParameter(
                "Matrix dimensions must be positive".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let a_shape = [NSNumber::new_i64(m as i64), NSNumber::new_i64(k as i64)];
            let a_shape_refs: Vec<&NSNumber> = a_shape.iter().map(|x| &**x).collect();
            let a_shape_ns = NSArray::from_slice(&a_shape_refs);
            let b_shape = [NSNumber::new_i64(k as i64), NSNumber::new_i64(n as i64)];
            let b_shape_refs: Vec<&NSNumber> = b_shape.iter().map(|x| &**x).collect();
            let b_shape_ns = NSArray::from_slice(&b_shape_refs);
            let a: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* a_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let b: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* b_shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let matmul_result: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, matrixMultiplicationWithPrimaryTensor : &* a,
                    secondaryTensor : &* b, name : None::<& objc2_foundation::NSString >
                ]
            };
            let sqrt_2_over_pi = 0.7978845608f32;
            let gelu_coeff = 0.044715f32;
            let const_shape = [NSNumber::new_i64(1)];
            let const_shape_refs: Vec<&NSNumber> = const_shape.iter().map(|x| &**x).collect();
            let const_shape_ns = NSArray::from_slice(&const_shape_refs);
            let half: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, constantWithScalar : 0.5f64, shape : &* const_shape_ns,
                    dataType : MPSDataType::Float32
                ]
            };
            let sqrt_2_pi: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, constantWithScalar : sqrt_2_over_pi as f64, shape : &*
                    const_shape_ns, dataType : MPSDataType::Float32
                ]
            };
            let coeff: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, constantWithScalar : gelu_coeff as f64, shape : &*
                    const_shape_ns, dataType : MPSDataType::Float32
                ]
            };
            let three: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, constantWithScalar : 3.0f64, shape : &* const_shape_ns,
                    dataType : MPSDataType::Float32
                ]
            };
            let one: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, constantWithScalar : 1.0f64, shape : &* const_shape_ns,
                    dataType : MPSDataType::Float32
                ]
            };
            let x_cubed: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, powerWithPrimaryTensor : &* matmul_result,
                    secondaryTensor : &* three, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let coeff_x_cubed: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* coeff,
                    secondaryTensor : &* x_cubed, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let inner_sum: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, additionWithPrimaryTensor : &* matmul_result,
                    secondaryTensor : &* coeff_x_cubed, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let scaled: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* sqrt_2_pi,
                    secondaryTensor : &* inner_sum, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let tanh_result: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, tanhWithTensor : &* scaled, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let one_plus_tanh: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, additionWithPrimaryTensor : &* one, secondaryTensor :
                    &* tanh_result, name : None::<& objc2_foundation::NSString >
                ]
            };
            let x_times_tanh: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* matmul_result,
                    secondaryTensor : &* one_plus_tanh, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let result: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* half,
                    secondaryTensor : &* x_times_tanh, name : None::<&
                    objc2_foundation::NSString >
                ]
            };
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send![
                    &* self.device, newBufferWithLength : (m * n * 4) as u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let a_shape_i64 = [m as i64, k as i64];
            let b_shape_i64 = [k as i64, n as i64];
            let c_shape_i64 = [m as i64, n as i64];
            self.execute_graph(
                &[&a, &b],
                &[a_buffer, b_buffer],
                &[&a_shape_i64[..], &b_shape_i64[..]],
                &[&result],
                &[&output_buffer],
                &[&c_shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// Element-wise Addition
    ///
    /// Computes: C = A + B (with broadcasting)
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    #[allow(dead_code)]
    pub fn add(
        &self,
        a_buffer: &ProtocolObject<dyn MTLBuffer>,
        b_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.is_empty() {
            return Err(GpuError::InvalidParameter(
                "Shape cannot be empty".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_ns = NSArray::from_slice(&shape_refs);
            let a: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let b: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let result: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, additionWithPrimaryTensor : &* a, secondaryTensor : &*
                    b, name : None::<& objc2_foundation::NSString >
                ]
            };
            let num_elements: usize = shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send![
                    &* self.device, newBufferWithLength : (num_elements * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
            self.execute_graph(
                &[&a, &b],
                &[a_buffer, b_buffer],
                &[&shape_i64[..], &shape_i64[..]],
                &[&result],
                &[&output_buffer],
                &[&shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// Element-wise Multiplication
    ///
    /// Computes: C = A * B (with broadcasting)
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    #[allow(dead_code)]
    pub fn mul(
        &self,
        a_buffer: &ProtocolObject<dyn MTLBuffer>,
        b_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.is_empty() {
            return Err(GpuError::InvalidParameter(
                "Shape cannot be empty".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_ns = NSArray::from_slice(&shape_refs);
            let a: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let b: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let result: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, multiplicationWithPrimaryTensor : &* a, secondaryTensor
                    : &* b, name : None::<& objc2_foundation::NSString >
                ]
            };
            let num_elements: usize = shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send![
                    &* self.device, newBufferWithLength : (num_elements * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
            self.execute_graph(
                &[&a, &b],
                &[a_buffer, b_buffer],
                &[&shape_i64[..], &shape_i64[..]],
                &[&result],
                &[&output_buffer],
                &[&shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// Reduction: Sum along specified axes
    ///
    /// Computes: sum(input, axes)
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    #[allow(dead_code)]
    pub fn reduce_sum(
        &self,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
        axes: &[i64],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.is_empty() {
            return Err(GpuError::InvalidParameter(
                "Shape cannot be empty".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_ns = NSArray::from_slice(&shape_refs);
            let input: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let axes_vec: Vec<_> = axes.iter().map(|&x| NSNumber::new_i64(x)).collect();
            let axes_refs: Vec<&NSNumber> = axes_vec.iter().map(|x| &**x).collect();
            let axes_ns = NSArray::from_slice(&axes_refs);
            let result: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, reductionSumWithTensor : &* input, axes : &* axes_ns,
                    name : None::<& objc2_foundation::NSString >
                ]
            };
            let mut output_shape: Vec<usize> = shape.to_vec();
            let mut axes_sorted = axes.to_vec();
            axes_sorted.sort_unstable_by(|a, b| b.cmp(a));
            for &axis in &axes_sorted {
                let axis_idx = if axis < 0 {
                    (shape.len() as i64 + axis) as usize
                } else {
                    axis as usize
                };
                output_shape.remove(axis_idx);
            }
            if output_shape.is_empty() {
                output_shape.push(1);
            }
            let output_elements: usize = output_shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send![
                    &* self.device, newBufferWithLength : (output_elements * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
            let output_shape_i64: Vec<i64> = output_shape.iter().map(|&x| x as i64).collect();
            self.execute_graph(
                &[&input],
                &[input_buffer],
                &[&shape_i64[..]],
                &[&result],
                &[&output_buffer],
                &[&output_shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
    /// Reduction: Mean along specified axes
    ///
    /// Computes: mean(input, axes)
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    #[allow(dead_code)]
    pub fn reduce_mean(
        &self,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
        axes: &[i64],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders::MPSDataType;
        if shape.is_empty() {
            return Err(GpuError::InvalidParameter(
                "Shape cannot be empty".to_string(),
            ));
        }
        autoreleasepool(|_| {
            let shape_vec: Vec<_> = shape.iter().map(|&x| NSNumber::new_i64(x as i64)).collect();
            let shape_refs: Vec<&NSNumber> = shape_vec.iter().map(|x| &**x).collect();
            let shape_ns = NSArray::from_slice(&shape_refs);
            let input: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, placeholderWithShape : &* shape_ns, dataType :
                    MPSDataType::Float32, name : None::<& objc2_foundation::NSString >
                ]
            };
            let axes_vec: Vec<_> = axes.iter().map(|&x| NSNumber::new_i64(x)).collect();
            let axes_refs: Vec<&NSNumber> = axes_vec.iter().map(|x| &**x).collect();
            let axes_ns = NSArray::from_slice(&axes_refs);
            let result: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    & self.graph, meanOfTensor : &* input, axes : &* axes_ns, name :
                    None::<& objc2_foundation::NSString >
                ]
            };
            let mut output_shape: Vec<usize> = shape.to_vec();
            let mut axes_sorted = axes.to_vec();
            axes_sorted.sort_unstable_by(|a, b| b.cmp(a));
            for &axis in &axes_sorted {
                let axis_idx = if axis < 0 {
                    (shape.len() as i64 + axis) as usize
                } else {
                    axis as usize
                };
                output_shape.remove(axis_idx);
            }
            if output_shape.is_empty() {
                output_shape.push(1);
            }
            let output_elements: usize = output_shape.iter().product();
            let output_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                msg_send![
                    &* self.device, newBufferWithLength : (output_elements * 4) as u64,
                    options : objc2_metal::MTLResourceOptions::StorageModeShared
                ]
            };
            let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
            let output_shape_i64: Vec<i64> = output_shape.iter().map(|&x| x as i64).collect();
            self.execute_graph(
                &[&input],
                &[input_buffer],
                &[&shape_i64[..]],
                &[&result],
                &[&output_buffer],
                &[&output_shape_i64[..]],
            )?;
            Ok(output_buffer)
        })
    }
}
