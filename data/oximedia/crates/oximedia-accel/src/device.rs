//! GPU device enumeration and selection.
//!
//! # Device Requirements
//!
//! ## Required
//!
//! | Requirement                  | Minimum value                       |
//! |------------------------------|-------------------------------------|
//! | Vulkan API version           | `Version::V1_2` (for full feature set, including `VkPhysicalDeviceSubgroupProperties` queries) |
//! | Compute-capable queue family | At least one queue with `QueueFlags::COMPUTE` |
//!
//! The `DeviceSelector` defaults to `Version::V1_0` so that the crate can run on older
//! drivers, but callers that rely on subgroup operations (see [`crate::subgroup`]) should
//! raise the minimum to `Version::V1_2`:
//!
//! ```no_run
//! use oximedia_accel::device::{DeviceSelector, DevicePreference};
//! use vulkano::Version;
//!
//! let device = DeviceSelector::new()
//!     .with_min_api_version(Version::V1_2)
//!     .select();
//! ```
//!
//! ## Optional / Recommended Extensions
//!
//! | Extension / Feature                           | Effect when present                                              |
//! |-----------------------------------------------|------------------------------------------------------------------|
//! | Subgroup operations (`VK_KHR_shader_subgroup_*`) | Enables [`crate::subgroup::SubgroupCapabilities`]; required for parallel reductions and ballot operations in compute shaders |
//! | Buffer device address (`VK_KHR_buffer_device_address`) | Needed for indirect-draw / indirect-compute dispatch chains; pass via `required_extensions` |
//! | Cooperative matrix (`VK_KHR_cooperative_matrix`) | Improves ML / matrix-multiplication throughput on supported hardware |
//!
//! None of these are enabled by default.  Enable them by setting
//! `DeviceSelector::required_extensions` before calling `select()`:
//!
//! ```no_run
//! use oximedia_accel::device::DeviceSelector;
//! use vulkano::device::DeviceExtensions;
//!
//! let exts = DeviceExtensions {
//!     khr_buffer_device_address: true,
//!     ..DeviceExtensions::empty()
//! };
//!
//! let device = DeviceSelector {
//!     required_extensions: exts,
//!     ..DeviceSelector::new()
//! }.select();
//! ```

use crate::error::{AccelError, AccelResult};
use std::sync::Arc;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::{Version, VulkanLibrary};

/// Strategy used by [`DeviceSelector`] to rank candidate Vulkan physical devices.
///
/// Each variant maps to a distinct scoring function applied inside [`DeviceSelector::select`].
/// Devices are sorted by score (descending) after filtering by `min_api_version` and
/// `required_extensions`, and the highest-scoring device is chosen.
///
/// ## Scoring table
///
/// | Variant        | Score formula                                                       |
/// |----------------|---------------------------------------------------------------------|
/// | `Discrete`     | 1 000 if `PhysicalDeviceType::DiscreteGpu`, else 0                  |
/// | `Integrated`   | 1 000 if `PhysicalDeviceType::IntegratedGpu`, else 0                |
/// | `Any`          | 100 unconditionally                                                 |
/// | `MostMemory`   | Sum of all memory-heap sizes in MiB (clamped to `u32::MAX`)        |
/// | `MostCompute`  | `properties.max_compute_work_group_count[0]` (X-dimension count)   |
///
/// **Universal discrete bonus:** regardless of which variant is active, every
/// `DiscreteGpu` device receives an additional **+100** points.  This means that
/// when two devices tie on the primary criterion (e.g., both have similar memory),
/// the discrete GPU is still preferred — even under `Integrated` or `MostMemory`
/// preferences.  The bonus is intentionally small (100) so it cannot override a
/// large memory or compute lead from an integrated device.
///
/// ## Overriding the preference
///
/// ```no_run
/// use oximedia_accel::device::{DeviceSelector, DevicePreference};
///
/// // Explicitly pick the device with the most VRAM (useful for large-model inference)
/// let device = DeviceSelector::new()
///     .with_preference(DevicePreference::MostMemory)
///     .select();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    /// Prefer discrete GPUs (dedicated graphics cards).
    ///
    /// Scores `DiscreteGpu` at 1 000, everything else at 0.
    /// This is the default because discrete GPUs typically have larger VRAM,
    /// higher compute throughput, and dedicated memory bandwidth.
    #[default]
    Discrete,
    /// Prefer integrated GPUs (shared-memory, on-die).
    ///
    /// Scores `IntegratedGpu` at 1 000, everything else at 0.
    /// Useful when power consumption or thermal constraints matter more than
    /// peak throughput, or when targeting iGPU-only systems.
    Integrated,
    /// Select the first available device (no topology preference).
    ///
    /// All devices score 100, so the one returned first by the Vulkan driver
    /// enumeration wins (driver-defined order, usually index 0).
    Any,
    /// Select device with the most aggregate video memory.
    ///
    /// Score = sum of all `MemoryHeap::size` values divided by 1 MiB, clamped to
    /// `u32::MAX`.  Favours devices with large VRAM which is important for
    /// texture-heavy workloads and large intermediate buffers.
    MostMemory,
    /// Select device with the highest compute work-group X-dimension limit.
    ///
    /// Score = `max_compute_work_group_count[0]`.  On most hardware this is
    /// `65535` for both discrete and integrated GPUs, so this preference
    /// effectively degrades to the universal discrete bonus.  It is most
    /// useful when a specific compute extension raises this limit.
    MostCompute,
}

/// Configures and executes Vulkan physical-device selection.
///
/// Build a selector with the desired constraints, then call [`DeviceSelector::select`]
/// to obtain a fully initialised [`VulkanDevice`]:
///
/// ```no_run
/// use oximedia_accel::device::{DeviceSelector, DevicePreference};
/// use vulkano::Version;
///
/// let vulkan_device = DeviceSelector::new()
///     .with_preference(DevicePreference::MostMemory)
///     .with_min_api_version(Version::V1_2)
///     .select()
///     .expect("no suitable Vulkan device found");
///
/// println!("Selected: {}", vulkan_device.name());
/// ```
///
/// ## Scoring summary
///
/// Internally `select` uses a private `score_device` function.  See [`DevicePreference`]
/// for the full scoring table.  Key point: every `DiscreteGpu` always receives an
/// additional **+100** bonus on top of the preference score, so a discrete GPU is
/// preferred when the primary scores are equal.
///
/// ## Device requirements
///
/// See the [module-level documentation](self) for the required Vulkan API version,
/// mandatory queue flags, and optional extensions.
#[derive(Debug, Clone)]
pub struct DeviceSelector {
    /// Device preference strategy.
    pub preference: DevicePreference,
    /// Minimum Vulkan API version required.
    pub min_api_version: Version,
    /// Required device extensions.
    pub required_extensions: DeviceExtensions,
}

impl Default for DeviceSelector {
    fn default() -> Self {
        Self {
            preference: DevicePreference::default(),
            min_api_version: Version::V1_0,
            required_extensions: DeviceExtensions::empty(),
        }
    }
}

impl DeviceSelector {
    /// Creates a new device selector with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the device preference.
    #[must_use]
    pub fn with_preference(mut self, preference: DevicePreference) -> Self {
        self.preference = preference;
        self
    }

    /// Sets the minimum API version.
    #[must_use]
    pub fn with_min_api_version(mut self, version: Version) -> Self {
        self.min_api_version = version;
        self
    }

    /// Selects a Vulkan device based on the configured preferences.
    ///
    /// The selection pipeline is:
    ///
    /// 1. Load the Vulkan library (fails fast if Vulkan is not installed).
    /// 2. Create a `VkInstance`.
    /// 3. Enumerate all physical devices.
    /// 4. **Filter** by `api_version >= min_api_version` AND `supported_extensions ⊇ required_extensions`.
    /// 5. **Score** each remaining device with the `score_device` heuristic (see [`DevicePreference`]).
    /// 6. Pick the highest-scoring device (`max_by_key`).
    /// 7. Locate a compute-capable queue family (`QueueFlags::COMPUTE`).
    /// 8. Create a logical device with one compute queue and return a [`VulkanDevice`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vulkan library cannot be loaded
    /// - No suitable instance can be created
    /// - No suitable device is found after filtering (returns [`crate::error::AccelError::NoDevice`])
    /// - No compute-capable queue family exists on the selected device
    /// - Device creation fails
    pub fn select(&self) -> AccelResult<VulkanDevice> {
        let library = VulkanLibrary::new()
            .map_err(|e| AccelError::VulkanInit(format!("Failed to load Vulkan library: {e:?}")))?;

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                application_name: Some("OxiMedia".to_string()),
                application_version: Version::V1_0,
                ..Default::default()
            },
        )
        .map_err(|e| AccelError::VulkanInit(format!("Failed to create instance: {e:?}")))?;

        let physical_device = instance
            .enumerate_physical_devices()
            .map_err(|e| {
                AccelError::DeviceSelection(format!("Failed to enumerate devices: {e:?}"))
            })?
            .filter(|dev| {
                dev.api_version() >= self.min_api_version
                    && dev
                        .supported_extensions()
                        .contains(&self.required_extensions)
            })
            .max_by_key(|dev| self.score_device(dev))
            .ok_or(AccelError::NoDevice)?;

        tracing::info!(
            "Selected GPU: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|q| q.queue_flags.contains(QueueFlags::COMPUTE))
            .ok_or_else(|| {
                AccelError::DeviceSelection("No compute queue family found".to_string())
            })?;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: self.required_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    #[allow(clippy::cast_possible_truncation)]
                    queue_family_index: queue_family_index as u32,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(|e| AccelError::DeviceSelection(format!("Failed to create device: {e:?}")))?;

        let queue = queues.next().ok_or_else(|| {
            AccelError::DeviceSelection("Failed to get compute queue".to_string())
        })?;

        let allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        Ok(VulkanDevice {
            physical_device,
            device,
            queue,
            allocator,
        })
    }

    fn score_device(&self, device: &vulkano::device::physical::PhysicalDevice) -> u32 {
        let props = device.properties();
        let device_type = props.device_type;

        let mut score = match self.preference {
            DevicePreference::Discrete => {
                if device_type == vulkano::device::physical::PhysicalDeviceType::DiscreteGpu {
                    1000
                } else {
                    0
                }
            }
            DevicePreference::Integrated => {
                if device_type == vulkano::device::physical::PhysicalDeviceType::IntegratedGpu {
                    1000
                } else {
                    0
                }
            }
            DevicePreference::Any => 100,
            DevicePreference::MostMemory => {
                // Score based on available memory (in MB)
                let memory = device
                    .memory_properties()
                    .memory_heaps
                    .iter()
                    .map(|heap| heap.size / (1024 * 1024))
                    .sum::<u64>();
                #[allow(clippy::cast_possible_truncation)]
                {
                    memory.min(u64::from(u32::MAX)) as u32
                }
            }
            DevicePreference::MostCompute => {
                // Score based on compute capability approximation
                props.max_compute_work_group_count[0]
            }
        };

        // Add bonus for discrete GPUs regardless of preference
        if device_type == vulkano::device::physical::PhysicalDeviceType::DiscreteGpu {
            score += 100;
        }

        score
    }
}

/// Represents a selected Vulkan device with its associated resources.
pub struct VulkanDevice {
    /// The physical device.
    pub physical_device: Arc<vulkano::device::physical::PhysicalDevice>,
    /// The logical device.
    pub device: Arc<Device>,
    /// The compute queue.
    pub queue: Arc<Queue>,
    /// Memory allocator.
    pub allocator: Arc<StandardMemoryAllocator>,
}

impl VulkanDevice {
    /// Returns the device name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.physical_device.properties().device_name
    }

    /// Returns the device type.
    #[must_use]
    pub fn device_type(&self) -> vulkano::device::physical::PhysicalDeviceType {
        self.physical_device.properties().device_type
    }

    /// Returns the total device memory in bytes.
    #[must_use]
    pub fn total_memory(&self) -> u64 {
        self.physical_device
            .memory_properties()
            .memory_heaps
            .iter()
            .map(|heap| heap.size)
            .sum()
    }
}
