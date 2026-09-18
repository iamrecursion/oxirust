//! Image scaling kernel implementations.

use crate::buffer::BufferManager;
use crate::error::{AccelError, AccelResult};
use crate::shaders::scale::{bilinear, nearest};
use crate::traits::ScaleFilter;
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

/// Image scaling kernel.
pub struct ScaleKernel {
    device: Arc<Device>,
    queue: Arc<Queue>,
    buffer_manager: BufferManager,
    descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
    bilinear_pipeline: Arc<ComputePipeline>,
    nearest_pipeline: Arc<ComputePipeline>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalePushConstants {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
}

impl ScaleKernel {
    /// Creates a new scaling kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline creation fails.
    /// Creates a new image scaling kernel.
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

        // Create bilinear pipeline
        let bilinear_shader = bilinear::load(device.clone())
            .map_err(|e| AccelError::ShaderCompilation(format!("Bilinear shader: {e:?}")))?;

        let bilinear_stage =
            PipelineShaderStageCreateInfo::new(bilinear_shader.entry_point("main").ok_or_else(
                || AccelError::ShaderCompilation("Shader entry point 'main' not found".to_string()),
            )?);

        let bilinear_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(
                &bilinear_stage,
            ))
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| AccelError::PipelineCreation(format!("Bilinear layout: {e:?}")))?,
        )
        .map_err(|e| AccelError::PipelineCreation(format!("Bilinear layout creation: {e:?}")))?;

        let bilinear_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bilinear_stage, bilinear_layout),
        )
        .map_err(|e| AccelError::PipelineCreation(format!("Bilinear pipeline: {e:?}")))?;

        // Create nearest pipeline
        let nearest_shader = nearest::load(device.clone())
            .map_err(|e| AccelError::ShaderCompilation(format!("Nearest shader: {e:?}")))?;

        let nearest_stage =
            PipelineShaderStageCreateInfo::new(nearest_shader.entry_point("main").ok_or_else(
                || AccelError::ShaderCompilation("Shader entry point 'main' not found".to_string()),
            )?);

        let nearest_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(
                &nearest_stage,
            ))
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| AccelError::PipelineCreation(format!("Nearest layout: {e:?}")))?,
        )
        .map_err(|e| AccelError::PipelineCreation(format!("Nearest layout creation: {e:?}")))?;

        let nearest_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(nearest_stage, nearest_layout),
        )
        .map_err(|e| AccelError::PipelineCreation(format!("Nearest pipeline: {e:?}")))?;

        Ok(Self {
            device,
            queue,
            buffer_manager,
            descriptor_allocator,
            bilinear_pipeline,
            nearest_pipeline,
        })
    }

    /// Scales an image using the specified filter.
    ///
    /// # Errors
    ///
    /// Returns an error if the scaling operation fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn scale(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        channels: u32,
        filter: ScaleFilter,
    ) -> AccelResult<Vec<u8>> {
        // Validate input
        let expected_size = (src_width * src_height * channels) as usize;
        if input.len() != expected_size {
            return Err(AccelError::BufferSizeMismatch {
                expected: expected_size,
                actual: input.len(),
            });
        }

        // Create input buffer
        let input_buffer = self
            .buffer_manager
            .create_device_buffer(input.len() as u64, BufferUsage::STORAGE_BUFFER)?;
        self.buffer_manager.upload_data(input, &input_buffer)?;

        // Create output buffer
        let output_size = (dst_width * dst_height * channels) as usize;
        let output_buffer = self
            .buffer_manager
            .create_device_buffer(output_size as u64, BufferUsage::STORAGE_BUFFER)?;

        // Select pipeline
        let pipeline = match filter {
            ScaleFilter::Nearest => &self.nearest_pipeline,
            ScaleFilter::Bilinear | ScaleFilter::Bicubic | ScaleFilter::Lanczos => {
                &self.bilinear_pipeline
            }
        };

        // Create descriptor set
        let layout =
            pipeline.layout().set_layouts().first().ok_or_else(|| {
                AccelError::PipelineCreation("No descriptor set layout".to_string())
            })?;

        let descriptor_set = DescriptorSet::new(
            self.descriptor_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_buffer.clone()),
                WriteDescriptorSet::buffer(1, output_buffer.clone()),
            ],
            [],
        )
        .map_err(|e| AccelError::PipelineCreation(format!("Descriptor set: {e:?}")))?;

        // Create command buffer
        let mut builder = self.buffer_manager.create_command_buffer()?;

        // Push constants
        let push_constants = ScalePushConstants {
            src_width,
            src_height,
            dst_width,
            dst_height,
            channels,
        };

        builder
            .bind_pipeline_compute(pipeline.clone())
            .map_err(|e| AccelError::CommandBuffer(format!("Bind pipeline: {e:?}")))?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .map_err(|e| AccelError::CommandBuffer(format!("Bind descriptor sets: {e:?}")))?
            .push_constants(pipeline.layout().clone(), 0, push_constants)
            .map_err(|e| AccelError::CommandBuffer(format!("Push constants: {e:?}")))?;
        unsafe {
            builder
                .dispatch([dst_width.div_ceil(16), dst_height.div_ceil(16), 1])
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

        // Download result
        self.buffer_manager.download_data(&output_buffer)
    }
}
