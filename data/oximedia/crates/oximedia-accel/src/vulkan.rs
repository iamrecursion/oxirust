//! Vulkan-based hardware acceleration implementation.

use crate::buffer::BufferManager;
use crate::device::{DeviceSelector, VulkanDevice};
use crate::error::{AccelError, AccelResult};
use crate::kernels::{color::ColorKernel, motion::MotionKernel, scale::ScaleKernel};
use crate::traits::{HardwareAccel, ScaleFilter};
use oximedia_core::PixelFormat;
use std::sync::Arc;

/// Vulkan-based hardware acceleration backend.
pub struct VulkanAccel {
    device: Arc<VulkanDevice>,
    scale_kernel: ScaleKernel,
    color_kernel: ColorKernel,
    motion_kernel: MotionKernel,
}

impl VulkanAccel {
    /// Creates a new Vulkan acceleration backend.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vulkan initialization fails
    /// - No suitable device is found
    /// - Kernel creation fails
    pub fn new(selector: &DeviceSelector) -> AccelResult<Self> {
        let device = Arc::new(selector.select()?);

        let scale_kernel = ScaleKernel::new(
            device.device.clone(),
            device.queue.clone(),
            BufferManager::new(device.allocator.clone(), device.queue.clone()),
        )?;

        let color_kernel = ColorKernel::new(
            device.device.clone(),
            device.queue.clone(),
            BufferManager::new(device.allocator.clone(), device.queue.clone()),
        )?;

        let motion_kernel = MotionKernel::new(
            device.device.clone(),
            device.queue.clone(),
            BufferManager::new(device.allocator.clone(), device.queue.clone()),
        )?;

        Ok(Self {
            device,
            scale_kernel,
            color_kernel,
            motion_kernel,
        })
    }

    /// Returns the name of the GPU device.
    #[must_use]
    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Returns the device type.
    #[must_use]
    pub fn device_type(&self) -> vulkano::device::physical::PhysicalDeviceType {
        self.device.device_type()
    }

    /// Returns the total device memory in bytes.
    #[must_use]
    pub fn total_memory(&self) -> u64 {
        self.device.total_memory()
    }

    fn channels_for_format(format: PixelFormat) -> AccelResult<u32> {
        match format {
            PixelFormat::Rgb24 => Ok(3),
            PixelFormat::Rgba32 => Ok(4),
            PixelFormat::Gray8 => Ok(1),
            _ => Err(AccelError::InvalidFormat(format!(
                "Unsupported format for GPU operations: {format:?}"
            ))),
        }
    }
}

impl HardwareAccel for VulkanAccel {
    fn scale_image(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        format: PixelFormat,
        filter: ScaleFilter,
    ) -> AccelResult<Vec<u8>> {
        let channels = Self::channels_for_format(format)?;

        self.scale_kernel.scale(
            input, src_width, src_height, dst_width, dst_height, channels, filter,
        )
    }

    fn convert_color(
        &self,
        input: &[u8],
        width: u32,
        height: u32,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> AccelResult<Vec<u8>> {
        self.color_kernel
            .convert(input, width, height, src_format, dst_format)
    }

    fn motion_estimation(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        block_size: u32,
    ) -> AccelResult<Vec<(i16, i16)>> {
        self.motion_kernel
            .estimate(reference, current, width, height, block_size)
    }
}
