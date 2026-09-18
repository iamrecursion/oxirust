//! Device management for hardware accelerators

use crate::error::Result;
use crate::hardware::accelerator::AcceleratorFactory;
use crate::hardware::{Accelerator, AcceleratorType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device ID
    pub id: usize,
    /// Device type
    pub device_type: AcceleratorType,
    /// Device name
    pub name: String,
    /// Compute capability
    pub compute_capability: (u32, u32),
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Is device available
    pub is_available: bool,
    /// Additional properties
    pub properties: HashMap<String, String>,
}
/// Device manager for handling multiple accelerators
pub struct DeviceManager {
    devices: Arc<Mutex<HashMap<(AcceleratorType, usize), Arc<dyn Accelerator>>>>,
    device_info: Arc<Mutex<Vec<DeviceInfo>>>,
    default_device: Arc<Mutex<Option<(AcceleratorType, usize)>>>,
impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Result<Self> {
        let manager = Self {
            devices: Arc::new(Mutex::new(HashMap::new())),
            device_info: Arc::new(Mutex::new(Vec::new())),
            default_device: Arc::new(Mutex::new(None)),
        };
        // Discover available devices
        manager.discover_devices()?;
        Ok(manager)
    }
    /// Discover available devices
    fn discover_devices(&self) -> Result<()> {
        let mut devices = self.devices.lock().expect("Operation failed");
        let mut device_info = self.device_info.lock().expect("Operation failed");
        // Discover CPU devices (always available)
        let cpu_device = AcceleratorFactory::create(AcceleratorType::CPU)?;
        devices.insert((AcceleratorType::CPU, 0), cpu_device.clone());
        device_info.push(DeviceInfo {
            id: 0,
            device_type: AcceleratorType::CPU,
            name: "CPU".to_string(),
            compute_capability: (1, 0),
            total_memory: cpu_device.capabilities().total_memory,
            available_memory: cpu_device.capabilities().total_memory,
            is_available: true,
            properties: HashMap::new(),
        });
        // Discover CUDA devices
        self.discover_cuda_devices(&mut devices, &mut device_info)?;
        // Discover other device types
        self.discover_metal_devices(&mut devices, &mut device_info)?;
        self.discover_rocm_devices(&mut devices, &mut device_info)?;
        // Set default device
        if !device_info.is_empty() {
            let mut default = self.default_device.lock().expect("Operation failed");
            *default = Some((device_info[0].device_type, device_info[0].id));
        }
        Ok(())
    /// Discover CUDA devices
    fn discover_cuda_devices(
        &self,
        devices: &mut HashMap<(AcceleratorType, usize), Arc<dyn Accelerator>>,
        device_info: &mut Vec<DeviceInfo>,
    ) -> Result<()> {
        // Check if CUDA is available
        if !Self::is_cuda_available() {
            return Ok(());
        // Simulate CUDA device discovery
        // In real implementation, this would use CUDA runtime API
        let num_cuda_devices = Self::get_cuda_device_count();
        for i in 0..num_cuda_devices {
            if let Ok(cuda_device) = AcceleratorFactory::create(AcceleratorType::CUDA) {
                devices.insert((AcceleratorType::CUDA, i), cuda_device.clone());
                let mut properties = HashMap::new();
                properties.insert("cuda_version".to_string(), "11.8".to_string());
                properties.insert("driver_version".to_string(), "520.61.05".to_string());
                device_info.push(DeviceInfo {
                    id: i,
                    device_type: AcceleratorType::CUDA,
                    name: format!("NVIDIA GPU {}", i),
                    compute_capability: (7, 5), // Example: RTX 2080
                    total_memory: 11 * 1024 * 1024 * 1024, // 11GB
                    available_memory: 10 * 1024 * 1024 * 1024, // 10GB
                    is_available: true,
                    properties,
                });
            }
    /// Discover Metal devices (macOS)
    fn discover_metal_devices(
        #[cfg(target_os = "macos")]
        {
            if let Ok(metal_device) = AcceleratorFactory::create(AcceleratorType::Metal) {
                devices.insert((AcceleratorType::Metal, 0), metal_device.clone());
                properties.insert("metal_version".to_string(), "3.0".to_string());
                    id: 0,
                    device_type: AcceleratorType::Metal,
                    name: "Apple GPU".to_string(),
                    compute_capability: (3, 0),
                    total_memory: 8 * 1024 * 1024 * 1024, // 8GB
                    available_memory: 7 * 1024 * 1024 * 1024, // 7GB
    /// Discover ROCm devices (AMD)
    fn discover_rocm_devices(
        _devices: &mut HashMap<(AcceleratorType, usize), Arc<dyn Accelerator>>, _device_info: &mut Vec<DeviceInfo>,
        // ROCm device discovery would go here
    /// List all available devices
    pub fn list_devices(&self) -> Vec<DeviceInfo> {
        self.device_info.lock().expect("Operation failed").clone()
    /// Get a specific device
    pub fn get_device(
        device_type: AcceleratorType,
        device_id: usize,
    ) -> Result<Arc<dyn Accelerator>> {
        let devices = self.devices.lock().expect("Operation failed");
        devices
            .get(&(device_type, device_id))
            .cloned()
            .ok_or_else(|| {
                crate::error::NeuralError::DeviceNotFound(format!(
                    "{:?} device {} not found",
                    device_type, device_id
                ))
            })
    /// Get default device
    pub fn get_default_device(&self) -> Result<Arc<dyn Accelerator>> {
        let default = self.default_device.lock().expect("Operation failed");
        if let Some((device_type, device_id)) = *default {
            self.get_device(device_type, device_id)
        } else {
            Err(crate::error::NeuralError::DeviceNotFound(
                "No default device available".to_string(),
            ))
    /// Set default device
    pub fn set_default_device(&self, device_type: AcceleratorType, deviceid: usize) -> Result<()> {
        // Verify device exists
        let _ = self.get_device(device_type, device_id)?;
        let mut default = self.default_device.lock().expect("Operation failed");
        *default = Some((device_type, device_id));
    /// Get device by capabilities
    pub fn get_device_by_capabilities(
        selector: &DeviceSelector,
        let device_info = self.device_info.lock().expect("Operation failed");
        // Find best matching device
        let best_device = device_info
            .iter()
            .filter(|info| selector.matches(info))
            .max_by_key(|info| selector.score(info));
        if let Some(device) = best_device {
            self.get_device(device.device_type, device.id)
                "No device matching requirements found".to_string(),
    /// Check if CUDA is available
    fn is_cuda_available() -> bool {
        // Simplified check
        std::env::var("CUDA_HOME").is_ok()
    /// Get CUDA device count
    fn get_cuda_device_count() -> usize {
        // In real implementation, use cudaGetDeviceCount
        if Self::is_cuda_available() {
            1 // Simulate 1 GPU
            0
    /// Synchronize all devices
    pub fn synchronize_all(&self) -> Result<()> {
        for device in devices.values() {
            device.synchronize()?;
    /// Get total memory across all devices
    pub fn total_memory(&self) -> usize {
        device_info.iter().map(|info| info.total_memory).sum()
    /// Get available memory across all devices
    pub fn available_memory(&self) -> usize {
        device_info.iter().map(|info| info.available_memory).sum()
/// Device selector for finding devices by capabilities
pub struct DeviceSelector {
    /// Minimum memory required
    pub min_memory: Option<usize>,
    /// Required device type
    pub device_type: Option<AcceleratorType>,
    /// Minimum compute capability
    pub min_compute_capability: Option<(u32, u32)>,
    /// Required features
    pub required_features: Vec<String>,
    /// Prefer device with most memory
    pub prefer_memory: bool,
    /// Prefer device with highest compute capability
    pub prefer_compute: bool,
impl Default for DeviceSelector {
    fn default() -> Self {
        Self {
            min_memory: None,
            device_type: None,
            min_compute_capability: None,
            required_features: Vec::new(),
            prefer_memory: true,
            prefer_compute: false,
impl DeviceSelector {
    /// Check if device matches selector criteria
    pub fn matches(&self, info: &DeviceInfo) -> bool {
        // Check device type
        if let Some(device_type) = self.device_type {
            if info.device_type != device_type {
                return false;
        // Check memory
        if let Some(min_memory) = self.min_memory {
            if info.available_memory < min_memory {
        // Check compute capability
        if let Some((min_major, min_minor)) = self.min_compute_capability {
            let (major, minor) = info.compute_capability;
            if major < min_major || (major == min_major && minor < min_minor) {
        // Check required features
        for feature in &self.required_features {
            if !info.properties.contains_key(feature) {
        true
    /// Score device for selection
    pub fn score(&self, info: &DeviceInfo) -> u64 {
        let mut score = 0u64;
        if self.prefer_memory {
            score += info.available_memory as u64;
        if self.prefer_compute {
            score += (major as u64 * 1000 + minor as u64) * 1_000_000_000;
        score
/// Multi-device execution context
pub struct MultiDeviceContext {
    devices: Vec<Arc<dyn Accelerator>>,
    distribution_strategy: DistributionStrategy,
impl MultiDeviceContext {
    /// Create a new multi-device context
    pub fn new(devices: Vec<Arc<dyn Accelerator>>, strategy: DistributionStrategy) -> Self {
            devices,
            distribution_strategy: strategy,
    /// Distribute work across _devices
    pub fn distribute_work<F>(&self, work_items: usize, workfn: F) -> Result<()>
    where
        F: Fn(usize, &dyn Accelerator) -> Result<()> + Send + Sync,
    {
        match self.distribution_strategy {
            DistributionStrategy::RoundRobin => {
                for (i, item) in (0..work_items).enumerate() {
                    let device_idx = i % self._devices.len();
                    work_fn(item, &*self._devices[device_idx])?;
                }
            DistributionStrategy::LoadBalanced => {
                // Simple load balancing based on available memory
                // In practice, would use more sophisticated scheduling
                let mut device_loads = vec![0usize; self.devices.len()];
                for item in 0..work_items {
                    let min_load_idx = device_loads
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, &load)| load)
                        .map(|(idx_)| idx)
                        .unwrap_or(0);
                    work_fn(item, &*self.devices[min_load_idx])?;
                    device_loads[min_load_idx] += 1;
            DistributionStrategy::DataParallel => {
                // Split work evenly across devices
                let items_per_device = (work_items + self.devices.len() - 1) / self.devices.len();
                for (device_idx, device) in self.devices.iter().enumerate() {
                    let start = device_idx * items_per_device;
                    let end = ((device_idx + 1) * items_per_device).min(work_items);
                    for item in start..end {
                        work_fn(item, &**device)?;
                    }
        for device in &self.devices {
/// Work distribution strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistributionStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Load-balanced distribution
    LoadBalanced,
    /// Data-parallel distribution
    DataParallel,
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_device_manager() {
        let manager = DeviceManager::new().expect("Operation failed");
        let devices = manager.list_devices();
        assert!(!devices.is_empty());
        assert!(devices
            .any(|d| d.device_type == AcceleratorType::CPU));
    fn test_device_selector() {
        let selector = DeviceSelector {
            min_memory: Some(1024 * 1024 * 1024), // 1GB
            device_type: Some(AcceleratorType::CPU),
            ..Default::default()
        let device_info = DeviceInfo {
            total_memory: 16 * 1024 * 1024 * 1024,
            available_memory: 8 * 1024 * 1024 * 1024,
        assert!(selector.matches(&device_info));
    fn test_multi_device_context() {
        let cpu1 = AcceleratorFactory::create(AcceleratorType::CPU).expect("Operation failed");
        let cpu2 = AcceleratorFactory::create(AcceleratorType::CPU).expect("Operation failed");
        let context = MultiDeviceContext::new(vec![cpu1, cpu2], DistributionStrategy::RoundRobin);
        let mut work_distribution = vec![0; 2];
        context
            .distribute_work(10, |item, device| {
                let device_idx = if device.accelerator_type() == AcceleratorType::CPU {
                    0
                } else {
                    1
                };
                work_distribution[device_idx] += 1;
                Ok(())
            .expect("Operation failed");
        // Round-robin should distribute evenly
        assert_eq!(work_distribution[0], 10); // All work goes to first device since both are CPU
