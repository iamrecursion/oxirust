#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
//! Node capability modelling for render farm workers.
//!
//! Tracks what rendering features each worker node supports and
//! validates that a job's requirements can be satisfied before dispatch.
//! Also includes GPU resource tracking: VRAM, compute units, and GPU temperature.

use std::collections::HashSet;

/// A discrete capability that a render node may possess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderCapability {
    /// NVIDIA CUDA acceleration.
    CudaGpu,
    /// AMD `ROCm` acceleration.
    RocmGpu,
    /// Apple Metal acceleration.
    MetalGpu,
    /// CPU-only software rendering.
    CpuRender,
    /// High-memory node (≥ 128 GiB RAM).
    HighMemory,
    /// NVMe-backed fast local scratch storage.
    FastLocalStorage,
    /// Access to shared network-attached storage.
    NetworkStorage,
    /// Ability to run Blender render engine.
    BlenderEngine,
    /// Ability to run Arnold render engine.
    ArnoldEngine,
    /// Ability to run V-Ray render engine.
    VRayEngine,
    /// Ability to run `RenderMan` render engine.
    RenderManEngine,
    /// Hardware-accelerated video encoding (NVENC/VCE/etc.).
    HardwareVideoEncode,
    /// Support for 3D LUT colour management.
    LutColorManagement,
    /// Support for distributed tile rendering.
    TileRendering,
}

impl RenderCapability {
    /// Returns a human-readable name for the capability.
    pub fn capability_name(self) -> &'static str {
        match self {
            Self::CudaGpu => "CUDA GPU",
            Self::RocmGpu => "ROCm GPU",
            Self::MetalGpu => "Metal GPU",
            Self::CpuRender => "CPU Render",
            Self::HighMemory => "High Memory",
            Self::FastLocalStorage => "Fast Local Storage",
            Self::NetworkStorage => "Network Storage",
            Self::BlenderEngine => "Blender Engine",
            Self::ArnoldEngine => "Arnold Engine",
            Self::VRayEngine => "V-Ray Engine",
            Self::RenderManEngine => "RenderMan Engine",
            Self::HardwareVideoEncode => "Hardware Video Encode",
            Self::LutColorManagement => "LUT Color Management",
            Self::TileRendering => "Tile Rendering",
        }
    }

    /// Returns `true` if this capability relates to GPU acceleration.
    pub fn is_gpu(self) -> bool {
        matches!(self, Self::CudaGpu | Self::RocmGpu | Self::MetalGpu)
    }

    /// Returns `true` if this capability relates to a specific render engine.
    pub fn is_render_engine(self) -> bool {
        matches!(
            self,
            Self::BlenderEngine | Self::ArnoldEngine | Self::VRayEngine | Self::RenderManEngine
        )
    }
}

// ── GpuResourceInfo ────────────────────────────────────────────────────────

/// Detailed GPU resource information for a render node.
///
/// Tracks VRAM, compute unit count, and current GPU temperature
/// so the scheduler can route memory-heavy or thermally-sensitive tasks
/// to appropriate nodes.
#[derive(Debug, Clone)]
pub struct GpuResourceInfo {
    /// GPU device name / model (e.g. "NVIDIA RTX 4090").
    pub device_name: String,
    /// Total VRAM in mebibytes (MiB).
    pub vram_total_mib: u64,
    /// Currently available VRAM in mebibytes (MiB).
    pub vram_available_mib: u64,
    /// Number of compute units (CUDA cores, CUs, or Execution Units depending on vendor).
    pub compute_units: u32,
    /// Current GPU core clock speed in MHz, if available.
    pub clock_mhz: Option<u32>,
    /// Current GPU die temperature in degrees Celsius, if available.
    pub temperature_celsius: Option<f32>,
    /// GPU utilisation in the range [0.0, 1.0], if available.
    pub utilisation: Option<f32>,
}

impl GpuResourceInfo {
    /// Create a new `GpuResourceInfo` with the mandatory fields.
    ///
    /// Optional metrics (`clock_mhz`, `temperature_celsius`, `utilisation`) default to `None`.
    pub fn new(device_name: impl Into<String>, vram_total_mib: u64, compute_units: u32) -> Self {
        Self {
            device_name: device_name.into(),
            vram_total_mib,
            vram_available_mib: vram_total_mib,
            compute_units,
            clock_mhz: None,
            temperature_celsius: None,
            utilisation: None,
        }
    }

    /// Builder: set available VRAM.
    pub fn with_vram_available(mut self, mib: u64) -> Self {
        self.vram_available_mib = mib;
        self
    }

    /// Builder: set current clock speed in MHz.
    pub fn with_clock_mhz(mut self, mhz: u32) -> Self {
        self.clock_mhz = Some(mhz);
        self
    }

    /// Builder: set current temperature in degrees Celsius.
    pub fn with_temperature(mut self, celsius: f32) -> Self {
        self.temperature_celsius = Some(celsius);
        self
    }

    /// Builder: set GPU utilisation in [0.0, 1.0].
    ///
    /// Values outside the range are clamped.
    pub fn with_utilisation(mut self, util: f32) -> Self {
        self.utilisation = Some(util.clamp(0.0, 1.0));
        self
    }

    /// Returns the used VRAM in MiB (saturating at 0 if available > total).
    pub fn vram_used_mib(&self) -> u64 {
        self.vram_total_mib.saturating_sub(self.vram_available_mib)
    }

    /// Returns `true` if the GPU appears to be overheating based on `threshold_celsius`.
    pub fn is_overheating(&self, threshold_celsius: f32) -> bool {
        self.temperature_celsius
            .map(|t| t >= threshold_celsius)
            .unwrap_or(false)
    }

    /// Returns `true` if at least `required_mib` MiB of VRAM are free.
    pub fn has_vram_for(&self, required_mib: u64) -> bool {
        self.vram_available_mib >= required_mib
    }
}

// ── NodeCapabilitySet ──────────────────────────────────────────────────────

/// The complete set of capabilities advertised by one render node.
#[derive(Debug, Clone, Default)]
pub struct NodeCapabilitySet {
    caps: HashSet<RenderCapability>,
    node_id: String,
    /// Optional GPU resource information (present when at least one GPU capability is set).
    gpu_info: Option<Vec<GpuResourceInfo>>,
}

impl NodeCapabilitySet {
    /// Creates an empty capability set for the given node.
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            caps: HashSet::new(),
            node_id: node_id.into(),
            gpu_info: None,
        }
    }

    /// Returns the node identifier.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Adds a capability to this node's set.
    pub fn add(&mut self, cap: RenderCapability) {
        self.caps.insert(cap);
    }

    /// Returns `true` if the node has the requested capability.
    pub fn has(&self, cap: RenderCapability) -> bool {
        self.caps.contains(&cap)
    }

    /// Returns the number of capabilities the node advertises.
    pub fn count(&self) -> usize {
        self.caps.len()
    }

    /// Returns `true` if this node satisfies all capabilities in `requirements`.
    pub fn meets_requirements(&self, requirements: &[RenderCapability]) -> bool {
        requirements.iter().all(|r| self.has(*r))
    }

    /// Returns an iterator over the capabilities.
    pub fn iter(&self) -> impl Iterator<Item = &RenderCapability> {
        self.caps.iter()
    }

    // ── GPU resource tracking ──────────────────────────────────────────────

    /// Attach GPU resource information for one or more GPUs on this node.
    ///
    /// Replaces any previously set GPU info.
    pub fn set_gpu_info(&mut self, info: Vec<GpuResourceInfo>) {
        self.gpu_info = if info.is_empty() { None } else { Some(info) };
    }

    /// Returns a slice of all GPU resource descriptors, or an empty slice if none.
    pub fn gpu_info(&self) -> &[GpuResourceInfo] {
        self.gpu_info.as_deref().unwrap_or(&[])
    }

    /// Returns the total VRAM across all GPUs in MiB.
    pub fn total_vram_mib(&self) -> u64 {
        self.gpu_info().iter().map(|g| g.vram_total_mib).sum()
    }

    /// Returns the available VRAM across all GPUs in MiB.
    pub fn available_vram_mib(&self) -> u64 {
        self.gpu_info().iter().map(|g| g.vram_available_mib).sum()
    }

    /// Returns the total compute unit count across all GPUs.
    pub fn total_compute_units(&self) -> u32 {
        self.gpu_info().iter().map(|g| g.compute_units).sum()
    }

    /// Returns `true` if any GPU on this node is above `threshold_celsius`.
    pub fn any_gpu_overheating(&self, threshold_celsius: f32) -> bool {
        self.gpu_info()
            .iter()
            .any(|g| g.is_overheating(threshold_celsius))
    }

    /// Returns the hottest GPU temperature in Celsius, or `None` if no GPU
    /// temperature data is available.
    pub fn max_gpu_temperature(&self) -> Option<f32> {
        self.gpu_info()
            .iter()
            .filter_map(|g| g.temperature_celsius)
            .reduce(f32::max)
    }

    /// Returns `true` if the node can satisfy a request for `required_vram_mib`
    /// across its available GPU VRAM.
    pub fn can_allocate_vram(&self, required_vram_mib: u64) -> bool {
        self.available_vram_mib() >= required_vram_mib
    }
}

// ── CapabilityRequirement ──────────────────────────────────────────────────

/// A named set of capability requirements that a job demands.
#[derive(Debug, Clone, Default)]
pub struct CapabilityRequirement {
    required: Vec<RenderCapability>,
    /// Optional human-readable label for diagnostics.
    pub label: String,
}

impl CapabilityRequirement {
    /// Creates a new, empty requirement set with a label.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            required: Vec::new(),
            label: label.into(),
        }
    }

    /// Adds a required capability.
    pub fn require(&mut self, cap: RenderCapability) {
        if !self.required.contains(&cap) {
            self.required.push(cap);
        }
    }

    /// Returns `true` if the given node satisfies **all** requirements.
    pub fn all_met(&self, node: &NodeCapabilitySet) -> bool {
        node.meets_requirements(&self.required)
    }

    /// Returns the list of capabilities that are **not** met by `node`.
    pub fn unmet_by(&self, node: &NodeCapabilitySet) -> Vec<RenderCapability> {
        self.required
            .iter()
            .filter(|&&c| !node.has(c))
            .copied()
            .collect()
    }

    /// Returns the number of required capabilities.
    pub fn len(&self) -> usize {
        self.required.len()
    }

    /// Returns `true` if no capabilities are required.
    pub fn is_empty(&self) -> bool {
        self.required.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn full_node() -> NodeCapabilitySet {
        let mut n = NodeCapabilitySet::new("node-01");
        n.add(RenderCapability::CudaGpu);
        n.add(RenderCapability::BlenderEngine);
        n.add(RenderCapability::HighMemory);
        n.add(RenderCapability::TileRendering);
        n.add(RenderCapability::NetworkStorage);
        n
    }

    #[test]
    fn test_capability_name_not_empty() {
        let caps = [
            RenderCapability::CudaGpu,
            RenderCapability::RocmGpu,
            RenderCapability::CpuRender,
            RenderCapability::BlenderEngine,
        ];
        for cap in caps {
            assert!(!cap.capability_name().is_empty());
        }
    }

    #[test]
    fn test_is_gpu_flags() {
        assert!(RenderCapability::CudaGpu.is_gpu());
        assert!(RenderCapability::MetalGpu.is_gpu());
        assert!(!RenderCapability::CpuRender.is_gpu());
    }

    #[test]
    fn test_is_render_engine_flags() {
        assert!(RenderCapability::BlenderEngine.is_render_engine());
        assert!(RenderCapability::ArnoldEngine.is_render_engine());
        assert!(!RenderCapability::CudaGpu.is_render_engine());
    }

    #[test]
    fn test_node_has_added_capability() {
        let n = full_node();
        assert!(n.has(RenderCapability::CudaGpu));
        assert!(n.has(RenderCapability::BlenderEngine));
    }

    #[test]
    fn test_node_missing_capability() {
        let n = full_node();
        assert!(!n.has(RenderCapability::VRayEngine));
    }

    #[test]
    fn test_node_count() {
        let n = full_node();
        assert_eq!(n.count(), 5);
    }

    #[test]
    fn test_node_id() {
        let n = NodeCapabilitySet::new("render-box-42");
        assert_eq!(n.node_id(), "render-box-42");
    }

    #[test]
    fn test_meets_requirements_all_present() {
        let n = full_node();
        assert!(n.meets_requirements(&[RenderCapability::CudaGpu, RenderCapability::BlenderEngine]));
    }

    #[test]
    fn test_meets_requirements_one_missing() {
        let n = full_node();
        assert!(!n.meets_requirements(&[
            RenderCapability::CudaGpu,
            RenderCapability::VRayEngine // not present
        ]));
    }

    #[test]
    fn test_capability_requirement_all_met() {
        let n = full_node();
        let mut req = CapabilityRequirement::new("blender-cuda");
        req.require(RenderCapability::CudaGpu);
        req.require(RenderCapability::BlenderEngine);
        assert!(req.all_met(&n));
    }

    #[test]
    fn test_capability_requirement_unmet_by() {
        let n = full_node();
        let mut req = CapabilityRequirement::new("vray-job");
        req.require(RenderCapability::VRayEngine);
        req.require(RenderCapability::CudaGpu);
        let unmet = req.unmet_by(&n);
        assert_eq!(unmet, vec![RenderCapability::VRayEngine]);
    }

    #[test]
    fn test_capability_requirement_empty() {
        let req = CapabilityRequirement::new("any");
        assert!(req.is_empty());
        let n = NodeCapabilitySet::new("x");
        assert!(req.all_met(&n)); // vacuously true
    }

    #[test]
    fn test_add_deduplicates() {
        let mut n = NodeCapabilitySet::new("n");
        n.add(RenderCapability::CpuRender);
        n.add(RenderCapability::CpuRender);
        assert_eq!(n.count(), 1);
    }

    #[test]
    fn test_require_deduplicates() {
        let mut req = CapabilityRequirement::new("dup");
        req.require(RenderCapability::HighMemory);
        req.require(RenderCapability::HighMemory);
        assert_eq!(req.len(), 1);
    }

    // ── GpuResourceInfo tests ─────────────────────────────────────────────

    #[test]
    fn test_gpu_resource_info_new() {
        let gpu = GpuResourceInfo::new("RTX 4090", 24576, 16384);
        assert_eq!(gpu.device_name, "RTX 4090");
        assert_eq!(gpu.vram_total_mib, 24576);
        assert_eq!(gpu.vram_available_mib, 24576);
        assert_eq!(gpu.compute_units, 16384);
        assert!(gpu.temperature_celsius.is_none());
        assert!(gpu.utilisation.is_none());
    }

    #[test]
    fn test_gpu_resource_info_builder() {
        let gpu = GpuResourceInfo::new("RTX 3080", 10240, 8704)
            .with_vram_available(8192)
            .with_clock_mhz(1800)
            .with_temperature(72.5)
            .with_utilisation(0.85);
        assert_eq!(gpu.vram_available_mib, 8192);
        assert_eq!(gpu.clock_mhz, Some(1800));
        assert_eq!(gpu.temperature_celsius, Some(72.5));
        assert_eq!(gpu.utilisation, Some(0.85));
    }

    #[test]
    fn test_gpu_vram_used_mib() {
        let gpu = GpuResourceInfo::new("A100", 40960, 6912).with_vram_available(20480);
        assert_eq!(gpu.vram_used_mib(), 20480);
    }

    #[test]
    fn test_gpu_is_overheating() {
        let hot = GpuResourceInfo::new("GPU", 8192, 2048).with_temperature(85.0);
        let cool = GpuResourceInfo::new("GPU", 8192, 2048).with_temperature(65.0);
        assert!(hot.is_overheating(80.0));
        assert!(!cool.is_overheating(80.0));
    }

    #[test]
    fn test_gpu_is_overheating_no_temp() {
        let gpu = GpuResourceInfo::new("GPU", 8192, 2048);
        // No temperature data → never overheating
        assert!(!gpu.is_overheating(70.0));
    }

    #[test]
    fn test_gpu_has_vram_for() {
        let gpu = GpuResourceInfo::new("GPU", 16384, 4096).with_vram_available(8192);
        assert!(gpu.has_vram_for(8192));
        assert!(gpu.has_vram_for(4096));
        assert!(!gpu.has_vram_for(8193));
    }

    #[test]
    fn test_gpu_utilisation_clamped() {
        let gpu = GpuResourceInfo::new("GPU", 8192, 2048).with_utilisation(1.5);
        assert_eq!(gpu.utilisation, Some(1.0));
        let gpu2 = GpuResourceInfo::new("GPU", 8192, 2048).with_utilisation(-0.5);
        assert_eq!(gpu2.utilisation, Some(0.0));
    }

    // ── NodeCapabilitySet GPU tracking tests ──────────────────────────────

    #[test]
    fn test_node_set_and_get_gpu_info() {
        let mut n = NodeCapabilitySet::new("gpu-node-01");
        n.add(RenderCapability::CudaGpu);
        let gpus = vec![
            GpuResourceInfo::new("RTX 4090", 24576, 16384).with_temperature(60.0),
            GpuResourceInfo::new("RTX 4090", 24576, 16384).with_temperature(65.0),
        ];
        n.set_gpu_info(gpus);
        assert_eq!(n.gpu_info().len(), 2);
    }

    #[test]
    fn test_node_total_vram_mib() {
        let mut n = NodeCapabilitySet::new("n");
        n.set_gpu_info(vec![
            GpuResourceInfo::new("A", 8192, 2048),
            GpuResourceInfo::new("B", 16384, 4096),
        ]);
        assert_eq!(n.total_vram_mib(), 24576);
    }

    #[test]
    fn test_node_available_vram_mib() {
        let mut n = NodeCapabilitySet::new("n");
        n.set_gpu_info(vec![
            GpuResourceInfo::new("A", 8192, 2048).with_vram_available(4096),
            GpuResourceInfo::new("B", 16384, 4096).with_vram_available(12288),
        ]);
        assert_eq!(n.available_vram_mib(), 16384);
    }

    #[test]
    fn test_node_total_compute_units() {
        let mut n = NodeCapabilitySet::new("n");
        n.set_gpu_info(vec![
            GpuResourceInfo::new("A", 8192, 3584),
            GpuResourceInfo::new("B", 16384, 6912),
        ]);
        assert_eq!(n.total_compute_units(), 10496);
    }

    #[test]
    fn test_node_any_gpu_overheating() {
        let mut n = NodeCapabilitySet::new("n");
        n.set_gpu_info(vec![
            GpuResourceInfo::new("A", 8192, 2048).with_temperature(70.0),
            GpuResourceInfo::new("B", 8192, 2048).with_temperature(85.0),
        ]);
        assert!(n.any_gpu_overheating(80.0));
        assert!(!n.any_gpu_overheating(90.0));
    }

    #[test]
    fn test_node_max_gpu_temperature() {
        let mut n = NodeCapabilitySet::new("n");
        n.set_gpu_info(vec![
            GpuResourceInfo::new("A", 8192, 2048).with_temperature(70.0),
            GpuResourceInfo::new("B", 8192, 2048).with_temperature(82.0),
        ]);
        assert_eq!(n.max_gpu_temperature(), Some(82.0));
    }

    #[test]
    fn test_node_max_gpu_temperature_none_when_no_gpu() {
        let n = NodeCapabilitySet::new("cpu-only");
        assert!(n.max_gpu_temperature().is_none());
    }

    #[test]
    fn test_node_can_allocate_vram() {
        let mut n = NodeCapabilitySet::new("n");
        n.set_gpu_info(vec![
            GpuResourceInfo::new("A", 24576, 16384).with_vram_available(20480)
        ]);
        assert!(n.can_allocate_vram(20480));
        assert!(!n.can_allocate_vram(20481));
    }

    #[test]
    fn test_node_set_empty_gpu_info_removes_gpus() {
        let mut n = NodeCapabilitySet::new("n");
        n.set_gpu_info(vec![GpuResourceInfo::new("A", 8192, 2048)]);
        n.set_gpu_info(vec![]); // clear
        assert_eq!(n.gpu_info().len(), 0);
        assert_eq!(n.total_vram_mib(), 0);
    }
}
