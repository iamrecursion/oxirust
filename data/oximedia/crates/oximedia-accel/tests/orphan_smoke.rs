//! Smoke tests verifying newly wired orphan modules compile and expose
//! at least one public item from each module.

use oximedia_accel::{
    async_compute::AsyncComputeQueue,
    buffer_ring::{BufferingMode, FrameBufferRing},
    compute_backend::{DispatchParams, VulkanComputeBackend},
    gpu_ops::compute_histogram_gpu,
    memory_access::{AccessPattern, MemoryAccessProfile},
    metal_backend::MetalDeviceInfo,
    multi_gpu::MultiGpuStrategy,
    pipeline_cache::{PipelineCache, PipelineDescriptor},
    vectorize::VectorizationStatus,
    work_item::{WorkItem, WorkItemKind},
};

#[test]
fn test_async_compute_queue_new() {
    let queue = AsyncComputeQueue::new();
    let stats = queue.stats().expect("stats should succeed");
    assert_eq!(stats.submitted, 0);
}

#[test]
fn test_frame_buffer_ring_double_buffering() {
    let ring = FrameBufferRing::new(BufferingMode::Double, 1024);
    assert_eq!(ring.slot_count(), 2);
    assert_eq!(ring.slot_capacity(), 1024);
    assert!(ring.is_empty());
}

#[test]
fn test_vulkan_compute_backend_new() {
    let backend = VulkanComputeBackend::new();
    // A freshly constructed backend should be usable.
    let _ = backend;
}

#[test]
fn test_dispatch_params_1d() {
    let p = DispatchParams::new_1d(64);
    assert_eq!(p.groups_x, 64);
    assert_eq!(p.groups_y, 1);
    assert_eq!(p.groups_z, 1);
}

#[test]
fn test_compute_histogram_gpu_empty() {
    // Zero-size frame returns zeroed histogram.
    let hist = compute_histogram_gpu(&[], 0, 0);
    assert_eq!(hist.len(), 256);
    assert!(hist.iter().all(|&v| v == 0));
}

#[test]
fn test_compute_histogram_gpu_single_white_pixel() {
    // Single white RGB24 pixel: luma ≈ 255.
    let frame = [255u8, 255, 255];
    let hist = compute_histogram_gpu(&frame, 1, 1);
    assert_eq!(hist[255], 1);
}

#[test]
fn test_access_pattern_sequential_efficiency() {
    let pattern = AccessPattern::Sequential;
    assert!((pattern.cache_efficiency() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_memory_access_profile_new() {
    let profile = MemoryAccessProfile::new();
    assert_eq!(profile.region_count(), 0);
}

#[test]
fn test_metal_device_info_stub() {
    let info = MetalDeviceInfo::stub();
    assert!(!info.name.is_empty());
}

#[test]
fn test_multi_gpu_strategy_round_robin() {
    let strategy = MultiGpuStrategy::RoundRobin;
    assert_eq!(strategy, MultiGpuStrategy::RoundRobin);
}

#[test]
fn test_pipeline_cache_new() {
    let cache = PipelineCache::new(64);
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_pipeline_descriptor_cache_key() {
    let desc = PipelineDescriptor::new("scale_kernel");
    // Cache key is deterministic.
    assert_eq!(desc.cache_key(), desc.cache_key());
}

#[test]
fn test_vectorization_status_scalar_not_vectorized() {
    let status = VectorizationStatus::Scalar;
    assert!(!status.is_vectorized());
}

#[test]
fn test_vectorization_status_fully_vectorized() {
    let status = VectorizationStatus::FullyVectorized { width: 8 };
    assert!(status.is_vectorized());
    assert_eq!(status.simd_width(), 8);
}

#[test]
fn test_work_item_kind_labels() {
    assert_eq!(WorkItemKind::Compute.label(), "compute");
    assert_eq!(WorkItemKind::Scale.label(), "scale");
    assert!(!WorkItemKind::Transfer.is_compute());
}

#[test]
fn test_work_item_new() {
    let item = WorkItem::new(1, WorkItemKind::Scale, 1920, 1080);
    assert_eq!(item.id, 1);
    assert_eq!(item.width, 1920);
    assert_eq!(item.height, 1080);
}
