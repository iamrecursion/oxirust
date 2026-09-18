//! Synthetic tests for the `DeviceType` / `DeviceCapabilities` probing
//! layer. Every test must pass on a plain CI worker with **no GPU and no
//! CUDA driver** — probing is infallible and CPU is always reported as
//! available.

use oximedia_ml::{DeviceCapabilities, DeviceType};

#[test]
fn auto_returns_valid_variant() {
    let device = DeviceType::auto();
    assert!(
        matches!(
            device,
            DeviceType::Cpu | DeviceType::Cuda | DeviceType::WebGpu | DeviceType::DirectMl
        ),
        "auto() must never return CoreMl and must be a real variant (got {device:?})"
    );
    // Whatever was chosen must genuinely be available.
    assert!(device.is_available());
}

#[test]
fn auto_is_idempotent() {
    let a = DeviceType::auto();
    let b = DeviceType::auto();
    let c = DeviceType::auto();
    assert_eq!(a, b);
    assert_eq!(b, c);
}

#[test]
fn cpu_is_always_available() {
    assert!(DeviceType::Cpu.is_available());
    assert_eq!(DeviceType::Cpu.name(), "cpu");
    assert_eq!(DeviceType::Cpu.display_name(), "CPU");
}

#[test]
fn probe_caps_cpu_returns_sane_defaults() {
    let caps = DeviceType::Cpu.probe_caps();
    assert_eq!(caps.device_type, DeviceType::Cpu);
    assert!(caps.is_available);
    assert!(caps.supports_int8, "CPU reference kernels must cover int8");
    // CPU kernels in oxionnx are fp32 today; these must stay conservative.
    assert!(!caps.supports_fp16);
    assert!(!caps.supports_bf16);
    assert!(caps.compute_capability.is_none());
    assert!(caps.device_name.starts_with("CPU"));
}

#[test]
fn list_available_always_contains_cpu() {
    let list = DeviceType::list_available();
    assert!(list.contains(&DeviceType::Cpu), "CPU must be enumerated");
    // CoreMl must never be enumerated until a coreml feature exists.
    assert!(!list.contains(&DeviceType::CoreMl));
    // Every listed device must actually probe as available.
    for d in &list {
        assert!(d.is_available(), "{d:?} listed but reports unavailable");
    }
}

#[test]
fn probe_all_returns_five_entries() {
    let all = DeviceCapabilities::probe_all();
    assert_eq!(all.len(), 5);

    let variants: Vec<DeviceType> = all.iter().map(|c| c.device_type).collect();
    assert_eq!(variants, DeviceType::all_variants().to_vec());

    // CoreMl is always present but never available.
    let coreml = all
        .iter()
        .find(|c| c.device_type == DeviceType::CoreMl)
        .expect("CoreMl entry must exist");
    assert!(!coreml.is_available);
}

#[test]
fn probe_never_panics() {
    // Drive every probe on every variant repeatedly; catch_unwind would
    // still let the test fail, so an unwrap-free run is success.
    for _ in 0..4 {
        for device in DeviceType::all_variants() {
            let _ = device.is_available();
            let _ = device.probe_caps();
        }
    }
}

#[test]
fn display_name_is_stable() {
    assert_eq!(DeviceType::Cpu.display_name(), "CPU");
    assert_eq!(DeviceType::Cuda.display_name(), "CUDA");
    assert_eq!(DeviceType::WebGpu.display_name(), "WebGPU");
    assert_eq!(DeviceType::DirectMl.display_name(), "DirectML");
    assert_eq!(DeviceType::CoreMl.display_name(), "CoreML");
}

#[test]
fn best_available_matches_auto() {
    let caps = DeviceCapabilities::best_available();
    assert_eq!(caps.device_type, DeviceType::auto());
    assert!(caps.is_available);
}

#[test]
fn name_and_display_are_distinct_spellings() {
    // The lowercase programmatic name and the human-facing label intentionally differ.
    for d in DeviceType::all_variants() {
        assert_ne!(
            d.name(),
            d.display_name(),
            "name vs display_name collided for {d:?}",
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_probe_does_not_panic_without_gpu() {
    // We do not assert a specific outcome — only that probing is
    // infallible. Some CI runners have a CUDA driver, most do not.
    let _ = DeviceType::Cuda.is_available();
    let caps = DeviceType::Cuda.probe_caps();
    assert_eq!(caps.device_type, DeviceType::Cuda);
}
