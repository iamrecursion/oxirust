//! Motion estimation kernel implementations.

use crate::buffer::BufferManager;
use crate::error::{AccelError, AccelResult};
use crate::shaders::motion::block_sad;
use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::pipeline::{
    compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::sync::GpuFuture;

/// Motion estimation kernel.
pub struct MotionKernel {
    device: Arc<Device>,
    queue: Arc<Queue>,
    buffer_manager: BufferManager,
    descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
    pipeline: Arc<ComputePipeline>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionPushConstants {
    width: u32,
    height: u32,
    block_size: u32,
    search_range: u32,
}

impl MotionKernel {
    /// Creates a new motion estimation kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline creation fails.
    ///
    /// # Panics
    ///
    /// Panics if the shader entry point "main" is not found.
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        buffer_manager: BufferManager,
    ) -> AccelResult<Self> {
        let descriptor_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        // Create pipeline
        let shader = block_sad::load(device.clone()).map_err(|e| {
            AccelError::ShaderCompilation(format!("Motion estimation shader: {e:?}"))
        })?;

        let stage =
            PipelineShaderStageCreateInfo::new(shader.entry_point("main").ok_or_else(|| {
                AccelError::ShaderCompilation(
                    "Motion estimation shader: entry point 'main' not found".to_string(),
                )
            })?);

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
                .into_pipeline_layout_create_info(device.clone())
                .map_err(|e| AccelError::PipelineCreation(format!("Motion layout: {e:?}")))?,
        )
        .map_err(|e| AccelError::PipelineCreation(format!("Motion layout creation: {e:?}")))?;

        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .map_err(|e| AccelError::PipelineCreation(format!("Motion pipeline: {e:?}")))?;

        Ok(Self {
            device,
            queue,
            buffer_manager,
            descriptor_allocator,
            pipeline,
        })
    }

    /// Performs block-based motion estimation.
    ///
    /// # Errors
    ///
    /// Returns an error if the motion estimation fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn estimate(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        block_size: u32,
    ) -> AccelResult<Vec<(i16, i16)>> {
        let expected_size = (width * height) as usize;
        if reference.len() != expected_size || current.len() != expected_size {
            return Err(AccelError::BufferSizeMismatch {
                expected: expected_size,
                actual: reference.len().min(current.len()),
            });
        }

        // Create input buffers
        let ref_buffer = self
            .buffer_manager
            .create_device_buffer(reference.len() as u64, BufferUsage::STORAGE_BUFFER)?;
        let cur_buffer = self
            .buffer_manager
            .create_device_buffer(current.len() as u64, BufferUsage::STORAGE_BUFFER)?;

        self.buffer_manager.upload_data(reference, &ref_buffer)?;
        self.buffer_manager.upload_data(current, &cur_buffer)?;

        // Calculate output size (one motion vector per block)
        let blocks_wide = width.div_ceil(block_size);
        let blocks_high = height.div_ceil(block_size);
        let mv_count = blocks_wide * blocks_high;

        // Create motion vector buffer (2 x i16 per block = 4 bytes per block)
        let mv_buffer = self
            .buffer_manager
            .create_device_buffer(u64::from(mv_count * 4), BufferUsage::STORAGE_BUFFER)?;

        // Create descriptor set
        let layout = self
            .pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or_else(|| AccelError::PipelineCreation("No descriptor set layout".to_string()))?;

        let descriptor_set = DescriptorSet::new(
            self.descriptor_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, ref_buffer),
                WriteDescriptorSet::buffer(1, cur_buffer),
                WriteDescriptorSet::buffer(2, mv_buffer.clone()),
            ],
            [],
        )
        .map_err(|e| AccelError::PipelineCreation(format!("Descriptor set: {e:?}")))?;

        // Create command buffer
        let mut builder = self.buffer_manager.create_command_buffer()?;

        // Use a reasonable search range (8 pixels)
        let push_constants = MotionPushConstants {
            width,
            height,
            block_size,
            search_range: 8,
        };

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .map_err(|e| AccelError::CommandBuffer(format!("Bind pipeline: {e:?}")))?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .map_err(|e| AccelError::CommandBuffer(format!("Bind descriptor sets: {e:?}")))?
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .map_err(|e| AccelError::CommandBuffer(format!("Push constants: {e:?}")))?;
        unsafe {
            builder
                .dispatch([blocks_wide.div_ceil(8), blocks_high.div_ceil(8), 1])
                .map_err(|e| AccelError::Dispatch(format!("Dispatch: {e:?}")))?;
        }

        let command_buffer = builder
            .build()
            .map_err(|e| AccelError::CommandBuffer(format!("Build: {e:?}")))?;

        // Execute and wait
        vulkano::sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|e| AccelError::Dispatch(format!("Execute: {e:?}")))?
            .then_signal_fence_and_flush()
            .map_err(|e| AccelError::Dispatch(format!("Flush: {e:?}")))?
            .wait(None)
            .map_err(|e| AccelError::Synchronization(format!("Wait: {e:?}")))?;

        // Download results
        let mv_data = self.buffer_manager.download_data(&mv_buffer)?;

        // Convert bytes to motion vectors
        let mut motion_vectors = Vec::with_capacity(mv_count as usize);
        for chunk in mv_data.chunks_exact(4) {
            let dx = i16::from_le_bytes([chunk[0], chunk[1]]);
            let dy = i16::from_le_bytes([chunk[2], chunk[3]]);
            motion_vectors.push((dx, dy));
        }

        Ok(motion_vectors)
    }
}
