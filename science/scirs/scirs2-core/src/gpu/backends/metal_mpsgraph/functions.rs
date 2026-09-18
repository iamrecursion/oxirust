//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(deprecated)] // TODO: Update objc2 msg_send_id! to msg_send! when API stabilizes

use crate::gpu::GpuError;
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

use super::types::MPSGraphContext;

#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
pub(crate) mod ffi {
    use std::ffi::{c_void, CString};
    use std::os::raw::{c_char, c_int};
    #[allow(clippy::duplicated_attributes)]
    #[link(name = "Foundation", kind = "framework")]
    #[link(name = "Metal", kind = "framework")]
    #[link(name = "MetalPerformanceShadersGraph", kind = "framework")]
    extern "C" {
        pub fn objc_getClass(name: *const c_char) -> *mut c_void;
        pub fn sel_registerName(name: *const c_char) -> *mut c_void;
        pub fn objc_retain(obj: *mut c_void) -> *mut c_void;
        pub fn objc_release(obj: *mut c_void);
        pub fn objc_autorelease(obj: *mut c_void) -> *mut c_void;
    }
    extern "C" {
        /// Execute MPSGraph using the simpler runWithMTLCommandQueue API
        ///
        /// Parameters:
        /// - graph_ptr: MPSGraph instance
        /// - queue_ptr: MTLCommandQueue
        /// - feeds_ptr: NSDictionary mapping input MPSGraphTensor -> MPSGraphTensorData
        /// - target_tensors_ptr: NSArray of MPSGraphTensor (output tensors to compute)
        /// - output_buffers_ptr: NSArray of MTLBuffer (pre-allocated output buffers)
        ///
        /// Returns: 0 on success, negative on error
        pub fn mpsgraph_execute_graph(
            graph_ptr: *mut c_void,
            queue_ptr: *mut c_void,
            feeds_ptr: *mut c_void,
            target_tensors_ptr: *mut c_void,
            output_buffers_ptr: *mut c_void,
        ) -> c_int;
    }
    #[allow(clashing_extern_declarations)]
    extern "C" {
        #[link_name = "objc_msgSend"]
        pub fn objc_msgSend_noargs(receiver: *mut c_void, selector: *mut c_void) -> *mut c_void;
        #[link_name = "objc_msgSend"]
        pub fn objc_msgSend_init_tensor_data(
            receiver: *mut c_void,
            selector: *mut c_void,
            buffer: *mut c_void,
            shape: *mut c_void,
            data_type: u32,
        ) -> *mut c_void;
        #[link_name = "objc_msgSend"]
        pub fn objc_msgSend_encode_graph(
            receiver: *mut c_void,
            selector: *mut c_void,
            command_buffer: *mut c_void,
            feeds: *mut c_void,
            target_ops: *mut c_void,
            results_dict: *mut c_void,
            exec_desc: *mut c_void,
        );
        #[link_name = "objc_msgSend"]
        pub fn objc_msgSend_dict_create(
            receiver: *mut c_void,
            selector: *mut c_void,
            objects: *const *mut c_void,
            keys: *const *mut c_void,
            count: u64,
        ) -> *mut c_void;
        #[link_name = "objc_msgSend"]
        pub fn objc_msgSend_array_create(
            receiver: *mut c_void,
            selector: *mut c_void,
            objects: *const *mut c_void,
            count: u64,
        ) -> *mut c_void;
        #[link_name = "objc_msgSend"]
        pub fn objc_msgSend_init_with_longlong(
            receiver: *mut c_void,
            selector: *mut c_void,
            value: i64,
        ) -> *mut c_void;
        #[link_name = "objc_msgSend"]
        pub fn objc_msgSend_init_with_uint(
            receiver: *mut c_void,
            selector: *mut c_void,
            value: u32,
        ) -> *mut c_void;
        #[link_name = "objc_msgSend"]
        pub fn objc_msgSend_set_object_for_key(
            receiver: *mut c_void,
            selector: *mut c_void,
            object: *mut c_void,
            key: *mut c_void,
        );
    }
    /// Helper to get Objective-C class by name
    pub unsafe fn get_class(name: &str) -> *mut c_void {
        let name_cstr = CString::new(name).expect("Operation failed");
        objc_getClass(name_cstr.as_ptr())
    }
    /// Helper to get selector by name
    pub unsafe fn get_selector(name: &str) -> *mut c_void {
        let name_cstr = CString::new(name).expect("Operation failed");
        sel_registerName(name_cstr.as_ptr())
    }
    /// Create NSNumber from i64
    pub unsafe fn create_nsnumber_i64(value: i64) -> *mut c_void {
        let ns_number_class = get_class("NSNumber");
        let alloc_sel = get_selector("alloc");
        let init_sel = get_selector("initWithLongLong:");
        let obj = objc_msgSend_noargs(ns_number_class, alloc_sel);
        objc_msgSend_init_with_longlong(obj, init_sel, value)
    }
    /// Create NSNumber from u32
    pub unsafe fn create_nsnumber_u32(value: u32) -> *mut c_void {
        let ns_number_class = get_class("NSNumber");
        let alloc_sel = get_selector("alloc");
        let init_sel = get_selector("initWithUnsignedInt:");
        let obj = objc_msgSend_noargs(ns_number_class, alloc_sel);
        objc_msgSend_init_with_uint(obj, init_sel, value)
    }
    /// Create NSArray from objects
    /// Retains objects to ensure proper reference counting
    pub unsafe fn create_nsarray(objects: &[*mut c_void]) -> *mut c_void {
        for obj in objects.iter() {
            if !obj.is_null() {
                objc_retain(*obj);
            }
        }
        let ns_array_class = get_class("NSArray");
        let array_sel = get_selector("arrayWithObjects:count:");
        objc_msgSend_array_create(
            ns_array_class,
            array_sel,
            objects.as_ptr(),
            objects.len() as u64,
        )
    }
    /// Create NSMutableDictionary using setObject:forKey:
    /// This approach doesn't require keys to implement NSCopying
    pub unsafe fn create_nsmutabledictionary(
        keys: &[*mut c_void],
        objects: &[*mut c_void],
    ) -> *mut c_void {
        assert_eq!(keys.len(), objects.len());
        let dict_class = get_class("NSMutableDictionary");
        let alloc_sel = get_selector("alloc");
        let init_sel = get_selector("init");
        let set_sel = get_selector("setObject:forKey:");
        let dict = objc_msgSend_noargs(dict_class, alloc_sel);
        let dict = objc_msgSend_noargs(dict, init_sel);
        if dict.is_null() {
            return std::ptr::null_mut();
        }
        for i in 0..keys.len() {
            objc_msgSend_set_object_for_key(dict, set_sel, objects[i], keys[i]);
        }
        dict
    }
    /// Create NSDictionary from keys and objects
    /// Retains keys and objects to ensure proper reference counting
    pub unsafe fn create_nsdictionary(
        keys: &[*mut c_void],
        objects: &[*mut c_void],
    ) -> *mut c_void {
        assert_eq!(keys.len(), objects.len());
        for key in keys.iter() {
            if !key.is_null() {
                objc_retain(*key);
            }
        }
        for obj in objects.iter() {
            if !obj.is_null() {
                objc_retain(*obj);
            }
        }
        let ns_dict_class = get_class("NSDictionary");
        let dict_sel = get_selector("dictionaryWithObjects:forKeys:count:");
        objc_msgSend_dict_create(
            ns_dict_class,
            dict_sel,
            objects.as_ptr(),
            keys.as_ptr(),
            keys.len() as u64,
        )
    }
    /// Get MPSGraphTensorData class (bypassing objc2 class lookup issues)
    pub unsafe fn get_mpsgraph_tensor_data_class() -> *mut c_void {
        get_class("MPSGraphTensorData")
    }
    /// Create MPSGraphTensorData from MTLBuffer
    pub unsafe fn create_mpsgraph_tensor_data(
        buffer: *mut c_void,
        shape: *mut c_void,
        data_type: u32,
    ) -> *mut c_void {
        eprintln!("[DEBUG] create_mpsgraph_tensor_data: Starting");
        let tensor_data_class = get_mpsgraph_tensor_data_class();
        if tensor_data_class.is_null() {
            eprintln!("[DEBUG] MPSGraphTensorData class is NULL!");
            return std::ptr::null_mut();
        }
        eprintln!(
            "[DEBUG] Got MPSGraphTensorData class: {:?}",
            tensor_data_class
        );
        let alloc_sel = get_selector("alloc");
        eprintln!("[DEBUG] Got alloc selector: {:?}", alloc_sel);
        let alloc_obj = objc_msgSend_noargs(tensor_data_class, alloc_sel);
        if alloc_obj.is_null() {
            eprintln!("[DEBUG] Allocation failed!");
            return std::ptr::null_mut();
        }
        eprintln!("[DEBUG] Allocated object: {:?}", alloc_obj);
        let init_sel = get_selector("initWithMTLBuffer:shape:dataType:");
        eprintln!("[DEBUG] Got init selector: {:?}", init_sel);
        eprintln!(
            "[DEBUG] Calling init with buffer={:?}, shape={:?}, data_type={}",
            buffer, shape, data_type
        );
        let result = objc_msgSend_init_tensor_data(alloc_obj, init_sel, buffer, shape, data_type);
        eprintln!("[DEBUG] Init returned: {:?}", result);
        result
    }
}
#[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
type MTLDevice = ();
#[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
type MTLCommandQueue = ();
#[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
type MTLBuffer = ();
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    fn test_mpsgraph_context_creation() {
        use objc2_metal::MTLCreateSystemDefaultDevice;
        unsafe {
            if let Some(device) = MTLCreateSystemDefaultDevice() {
                let command_queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>> =
                    msg_send_id![&device, newCommandQueue];
                let _context = MPSGraphContext::new(device, command_queue);
            }
        }
    }
    #[test]
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    fn test_matmul_api() {
        use objc2_metal::MTLCreateSystemDefaultDevice;
        unsafe {
            if let Some(device) = MTLCreateSystemDefaultDevice() {
                let command_queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>> =
                    msg_send_id![&device, newCommandQueue];
                let context = MPSGraphContext::new(device.clone(), command_queue);
                let m = 2;
                let k = 3;
                let n = 2;
                let a_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = msg_send_id![
                    & device, newBufferWithLength : (m * k * 4) as u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ];
                let b_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = msg_send_id![
                    & device, newBufferWithLength : (k * n * 4) as u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ];
                let result = context.matmul(&a_buffer, &b_buffer, m, k, n);
                assert!(result.is_ok(), "Matmul should succeed");
            }
        }
    }
    #[test]
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    fn test_matmul_validation() {
        use objc2_metal::MTLCreateSystemDefaultDevice;
        unsafe {
            if let Some(device) = MTLCreateSystemDefaultDevice() {
                let command_queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>> =
                    msg_send_id![&device, newCommandQueue];
                let context = MPSGraphContext::new(device.clone(), command_queue);
                let dummy_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = msg_send_id![
                    & device, newBufferWithLength : 64u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ];
                let result = context.matmul(&dummy_buffer, &dummy_buffer, 0, 1, 1);
                assert!(result.is_err(), "Should reject zero dimensions");
                assert!(matches!(result.unwrap_err(), GpuError::InvalidParameter(_)));
            }
        }
    }
    #[test]
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    fn test_softmax_api() {
        use objc2_metal::MTLCreateSystemDefaultDevice;
        unsafe {
            if let Some(device) = MTLCreateSystemDefaultDevice() {
                let command_queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>> =
                    msg_send_id![&device, newCommandQueue];
                let context = MPSGraphContext::new(device.clone(), command_queue);
                let shape = [2, 3, 4];
                let size: usize = shape.iter().product();
                let input_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = msg_send_id![
                    & device, newBufferWithLength : (size * 4) as u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ];
                let result = context.softmax(&input_buffer, &shape, -1);
                assert!(result.is_ok(), "Softmax should succeed");
                let result = context.softmax(&input_buffer, &shape, 10);
                assert!(result.is_err(), "Should reject out-of-bounds axis");
            }
        }
    }
    #[test]
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    fn test_gelu_api() {
        use objc2_metal::MTLCreateSystemDefaultDevice;
        unsafe {
            if let Some(device) = MTLCreateSystemDefaultDevice() {
                let command_queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>> =
                    msg_send_id![&device, newCommandQueue];
                let context = MPSGraphContext::new(device.clone(), command_queue);
                let shape = [2, 512, 768];
                let size: usize = shape.iter().product();
                let input_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = msg_send_id![
                    & device, newBufferWithLength : (size * 4) as u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ];
                let result = context.gelu(&input_buffer, &shape);
                assert!(result.is_ok(), "GeLU should succeed");
            }
        }
    }
    #[test]
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    fn test_layer_norm_api() {
        use objc2_metal::MTLCreateSystemDefaultDevice;
        unsafe {
            if let Some(device) = MTLCreateSystemDefaultDevice() {
                let command_queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>> =
                    msg_send_id![&device, newCommandQueue];
                let context = MPSGraphContext::new(device.clone(), command_queue);
                let shape = [2, 512, 768];
                let normalized_shape = [768];
                let size: usize = shape.iter().product();
                let norm_size: usize = normalized_shape.iter().product();
                let input_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = msg_send_id![
                    & device, newBufferWithLength : (size * 4) as u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ];
                let gamma_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = msg_send_id![
                    & device, newBufferWithLength : (norm_size * 4) as u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ];
                let beta_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = msg_send_id![
                    & device, newBufferWithLength : (norm_size * 4) as u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ];
                let result = context.layer_norm(
                    &input_buffer,
                    &gamma_buffer,
                    &beta_buffer,
                    &shape,
                    &normalized_shape,
                    1e-5,
                );
                assert!(result.is_ok(), "LayerNorm should succeed");
            }
        }
    }
    #[test]
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    fn test_rope_validation() {
        use objc2_metal::MTLCreateSystemDefaultDevice;
        unsafe {
            if let Some(device) = MTLCreateSystemDefaultDevice() {
                let command_queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>> =
                    msg_send_id![&device, newCommandQueue];
                let context = MPSGraphContext::new(device.clone(), command_queue);
                let dummy_buffer: Retained<ProtocolObject<dyn MTLBuffer>> = msg_send_id![
                    & device, newBufferWithLength : 1024u64, options :
                    objc2_metal::MTLResourceOptions::StorageModeShared
                ];
                let result = context.rope(&dummy_buffer, &dummy_buffer, &dummy_buffer, &[2, 3], 64);
                assert!(result.is_err(), "Should reject non-4D input");
                let shape = [1, 10, 12, 64];
                let result = context.rope(&dummy_buffer, &dummy_buffer, &dummy_buffer, &shape, 128);
                assert!(result.is_err(), "Should reject rotary_ndims > head_dim");
            }
        }
    }
}
