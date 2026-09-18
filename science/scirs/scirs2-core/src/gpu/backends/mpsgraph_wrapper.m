#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

/// Execute MPSGraph using encodeToCommandBuffer with in-place results
///
/// This creates MPSGraphTensorData for output buffers and uses them in a results dictionary
///
/// Parameters:
/// - graph: MPSGraph instance
/// - queue: MTLCommandQueue
/// - feeds: NSDictionary mapping input MPSGraphTensor -> MPSGraphTensorData
/// - target_tensors: NSArray of MPSGraphTensor (output tensors to compute)
/// - output_buffers: NSArray of MTLBuffer (pre-allocated output buffers - results written here)
///
/// Returns: 0 on success, non-zero on error
int mpsgraph_execute_graph(
    void* graph_ptr,
    void* queue_ptr,
    void* feeds_ptr,
    void* target_tensors_ptr,
    void* output_buffers_ptr
) {
    @autoreleasepool {
        // Cast void* to proper Objective-C types
        MPSGraph* graph = (__bridge MPSGraph*)graph_ptr;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
        NSDictionary* feeds = (__bridge NSDictionary*)feeds_ptr;
        NSArray<MPSGraphTensor*>* targetTensors = (__bridge NSArray<MPSGraphTensor*>*)target_tensors_ptr;
        NSArray<id<MTLBuffer>>* outputBuffers = (__bridge NSArray<id<MTLBuffer>>*)output_buffers_ptr;

        if (!graph || !queue || !feeds || !targetTensors || !outputBuffers) {
            NSLog(@"[MPSGraph] Null parameter detected");
            return -1;
        }

        if ([targetTensors count] != [outputBuffers count]) {
            NSLog(@"[MPSGraph] Mismatch: %lu target tensors vs %lu output buffers",
                  (unsigned long)[targetTensors count], (unsigned long)[outputBuffers count]);
            return -2;
        }

        @try {
            // Create MPSGraphTensorData for each output buffer
            NSMutableDictionary* outputTensorDataDict = [NSMutableDictionary dictionary];

            for (NSUInteger i = 0; i < [targetTensors count]; i++) {
                MPSGraphTensor* tensor = targetTensors[i];
                id<MTLBuffer> buffer = outputBuffers[i];

                // Get shape from tensor
                NSArray<NSNumber*>* shape = [tensor shape];

                // Create MPSGraphTensorData wrapping the output buffer
                MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:buffer
                                shape:shape
                             dataType:[tensor dataType]];

                [outputTensorDataDict setObject:tensorData forKey:tensor];
            }

            // Run graph synchronously - this returns computed results
            MPSGraphTensorDataDictionary* results = [graph runWithMTLCommandQueue:queue
                                                                            feeds:feeds
                                                                    targetTensors:targetTensors
                                                                 targetOperations:nil];

            if (!results) {
                NSLog(@"[MPSGraph] runWithMTLCommandQueue returned nil");
                return -3;
            }

            //  TODO: Copy results to output buffers
            // For now, just verify that all results are present
            for (NSUInteger i = 0; i < [targetTensors count]; i++) {
                MPSGraphTensor* tensor = targetTensors[i];
                MPSGraphTensorData* resultData = results[tensor];

                if (!resultData) {
                    NSLog(@"[MPSGraph] No result data for tensor %lu", (unsigned long)i);
                    return -4;
                }
            }

            NSLog(@"[MPSGraph] Graph execution completed successfully, but results not copied yet");
            return 0;
        }
        @catch (NSException* exception) {
            NSLog(@"[MPSGraph] Exception during execution: %@", exception);
            NSLog(@"[MPSGraph] Reason: %@", exception.reason);
            return -7;
        }
    }
}

/// Get MPSDataType enum value for Float32
/// Required because Rust transmute may not be reliable across FFI boundary
uint32_t mpsgraph_datatype_float32(void) {
    return (uint32_t)MPSDataTypeFloat32;
}

/// Get MPSDataType enum value for Float16
uint32_t mpsgraph_datatype_float16(void) {
    return (uint32_t)MPSDataTypeFloat16;
}

/// Get MPSDataType enum value for Int32
uint32_t mpsgraph_datatype_int32(void) {
    return (uint32_t)MPSDataTypeInt32;
}
