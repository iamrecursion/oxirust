//! Bilateral filter — spatial noise reduction with edge preservation.
//!
//! The bilateral filter weights each neighbour by both spatial proximity
//! (Gaussian over distance) **and** photometric similarity (Gaussian over
//! intensity difference).  Pixels separated by a strong edge contribute
//! little, so the filter smooths flat regions while keeping sharp transitions.
//!
//! This module provides:
//! - [`bilateral_cpu`] — single-channel scalar reference implementation
//!   (always available, deterministic, used by tests).
//! - `BilateralGpu` — WebGPU dispatcher gated on `feature = "webgpu"` that
//!   submits the [`crate::shaders::bilateral::BILATERAL_WGSL`] compute shader.
//!
//! Both paths take an f32 single-channel buffer.  Multi-channel images should
//! be processed plane-by-plane (or repacked to grayscale before noise
//! reduction).
//!
//! The GPU kernel uses cooperative shared-memory tiling: every 16×16
//! workgroup loads a 32×32 tile (16 + 2×8 halo) into workgroup memory before
//! a `workgroupBarrier()`.  Maximum kernel radius is **8 pixels** to keep all
//! filter taps inside the halo; the dispatcher rejects larger radii.

use crate::error::{AccelError, AccelResult};

/// Maximum supported bilateral kernel radius (set by the WGSL shader halo).
pub const MAX_BILATERAL_RADIUS: u32 = 8;

/// Bilateral filter parameters.
#[derive(Debug, Clone, Copy)]
pub struct BilateralParams {
    /// Standard deviation of the photometric (intensity) Gaussian.
    pub sigma_color: f32,
    /// Standard deviation of the spatial Gaussian.
    pub sigma_space: f32,
    /// Kernel radius in pixels; the actual window is `(2r+1)²`.
    pub kernel_radius: u32,
}

impl Default for BilateralParams {
    fn default() -> Self {
        Self {
            sigma_color: 0.1,
            sigma_space: 2.0,
            kernel_radius: 2,
        }
    }
}

impl BilateralParams {
    /// Validate that the parameters can be dispatched.
    ///
    /// # Errors
    /// Returns [`AccelError::InvalidDimensions`] when `kernel_radius` exceeds
    /// [`MAX_BILATERAL_RADIUS`] or when either sigma is non-finite/non-positive.
    pub fn validate(&self) -> AccelResult<()> {
        if self.kernel_radius > MAX_BILATERAL_RADIUS {
            return Err(AccelError::InvalidDimensions(format!(
                "bilateral kernel_radius={} exceeds maximum {MAX_BILATERAL_RADIUS}",
                self.kernel_radius
            )));
        }
        if !self.sigma_color.is_finite() || self.sigma_color <= 0.0 {
            return Err(AccelError::InvalidDimensions(format!(
                "bilateral sigma_color must be positive and finite, got {}",
                self.sigma_color
            )));
        }
        if !self.sigma_space.is_finite() || self.sigma_space <= 0.0 {
            return Err(AccelError::InvalidDimensions(format!(
                "bilateral sigma_space must be positive and finite, got {}",
                self.sigma_space
            )));
        }
        Ok(())
    }
}

/// CPU reference implementation of the bilateral filter.
///
/// Produces the same arithmetic as the WGSL shader (operation order: sum and
/// wsum accumulated separately, then divided), so the GPU test can match
/// within ±0.01 LSB on a healthy float pipeline.
///
/// # Errors
/// Returns [`AccelError::BufferSizeMismatch`] when `input.len() != width*height`,
/// or [`AccelError::InvalidDimensions`] when `params.validate()` fails.
pub fn bilateral_cpu(
    input: &[f32],
    width: u32,
    height: u32,
    params: BilateralParams,
) -> AccelResult<Vec<f32>> {
    params.validate()?;
    let n = (width as usize) * (height as usize);
    if input.len() != n {
        return Err(AccelError::BufferSizeMismatch {
            expected: n,
            actual: input.len(),
        });
    }
    let r = params.kernel_radius as i32;
    let w = width as i32;
    let h = height as i32;
    let inv2sc2 = 1.0_f32 / (2.0 * params.sigma_color * params.sigma_color);
    let inv2ss2 = 1.0_f32 / (2.0 * params.sigma_space * params.sigma_space);

    let mut out = vec![0.0_f32; n];

    for y in 0..h {
        for x in 0..w {
            let center = input[(y * w + x) as usize];
            let mut sum: f32 = 0.0;
            let mut wsum: f32 = 0.0;
            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = (x + dx).clamp(0, w - 1);
                    let ny = (y + dy).clamp(0, h - 1);
                    let neighbor = input[(ny * w + nx) as usize];
                    let dc = center - neighbor;
                    let color_w = (-(dc * dc) * inv2sc2).exp();
                    let space_w = (-((dx * dx + dy * dy) as f32) * inv2ss2).exp();
                    let w_total = color_w * space_w;
                    sum += neighbor * w_total;
                    wsum += w_total;
                }
            }
            out[(y * w + x) as usize] = sum / wsum;
        }
    }

    Ok(out)
}

// ── GPU path (feature = "webgpu") ───────────────────────────────────────────

#[cfg(feature = "webgpu")]
pub use gpu_impl::BilateralGpu;

#[cfg(feature = "webgpu")]
mod gpu_impl {
    use super::*;
    use crate::shaders::BILATERAL_WGSL;
    use std::sync::Arc;
    use wgpu::util::DeviceExt;

    /// GPU dispatcher for the bilateral filter compute shader.
    ///
    /// Caller supplies a shared [`Arc<wgpu::Device>`] / [`Arc<wgpu::Queue>`] —
    /// these are kept inside so the pipeline can be reused across frames
    /// without re-compiling the shader.
    pub struct BilateralGpu {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
    }

    /// Uniform buffer layout matching `BilateralParams` in WGSL.
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct BilateralParamsUbo {
        width: u32,
        height: u32,
        sigma_color: f32,
        sigma_space: f32,
        kernel_radius: i32,
        _pad0: i32,
        _pad1: i32,
        _pad2: i32,
    }

    // Safety: plain old data, no Drop, no references.
    unsafe impl bytemuck::Zeroable for BilateralParamsUbo {}
    unsafe impl bytemuck::Pod for BilateralParamsUbo {}

    impl BilateralGpu {
        /// Compile the shader and build the pipeline.
        ///
        /// # Errors
        /// Returns [`AccelError::PipelineCreation`] on shader/pipeline errors.
        pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> AccelResult<Self> {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bilateral-wgsl"),
                source: wgpu::ShaderSource::Wgsl(BILATERAL_WGSL.into()),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bilateral-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bilateral-pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bilateral-pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("bilateral"),
                compilation_options: Default::default(),
                cache: None,
            });

            Ok(Self {
                device,
                queue,
                pipeline,
                bind_group_layout: bgl,
            })
        }

        /// Dispatch the bilateral filter on `input` and return the filtered f32 buffer.
        ///
        /// # Errors
        /// Returns [`AccelError`] variants for buffer-size mismatch, dispatch
        /// failures, or readback errors.
        pub fn process(
            &self,
            input: &[f32],
            width: u32,
            height: u32,
            params: BilateralParams,
        ) -> AccelResult<Vec<f32>> {
            params.validate()?;
            let n = (width as usize) * (height as usize);
            if input.len() != n {
                return Err(AccelError::BufferSizeMismatch {
                    expected: n,
                    actual: input.len(),
                });
            }
            if width == 0 || height == 0 {
                return Ok(Vec::new());
            }

            let bytes_in: &[u8] = bytemuck::cast_slice(input);
            let out_bytes_len = (n * std::mem::size_of::<f32>()) as u64;

            let src_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bilateral-src"),
                    contents: bytes_in,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

            let dst_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bilateral-dst"),
                size: out_bytes_len,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let ubo = BilateralParamsUbo {
                width,
                height,
                sigma_color: params.sigma_color,
                sigma_space: params.sigma_space,
                kernel_radius: params.kernel_radius as i32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let uniform_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bilateral-ubo"),
                    contents: bytemuck::bytes_of(&ubo),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bilateral-bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bilateral-encoder"),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bilateral-cpass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let dx = width.div_ceil(16);
                let dy = height.div_ceil(16);
                cpass.dispatch_workgroups(dx, dy, 1);
            }

            // Readback
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bilateral-readback"),
                size: out_bytes_len,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&dst_buf, 0, &readback, 0, out_bytes_len);
            self.queue.submit(Some(encoder.finish()));

            let buf_slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buf_slice.map_async(wgpu::MapMode::Read, move |v| {
                let _ = tx.send(v);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| {
                    AccelError::Synchronization(format!("bilateral poll failed: {e:?}"))
                })?;
            rx.recv()
                .map_err(|_| {
                    AccelError::Synchronization("bilateral channel recv error".to_string())
                })?
                .map_err(|e| AccelError::Synchronization(format!("bilateral map failed: {e:?}")))?;

            let data = buf_slice
                .get_mapped_range()
                .map_err(|e| AccelError::MemoryMap(format!("bilateral buffer map failed: {e}")))?;
            let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
            drop(data);
            readback.unmap();
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference output for a tiny constant input must equal the input
    /// (smoothing a flat plane is a no-op).
    #[test]
    fn test_bilateral_cpu_matches_reference() {
        let w = 8u32;
        let h = 8u32;
        let input: Vec<f32> = vec![0.5; (w * h) as usize];
        let params = BilateralParams {
            sigma_color: 0.1,
            sigma_space: 1.5,
            kernel_radius: 2,
        };
        let out = bilateral_cpu(&input, w, h, params).expect("bilateral_cpu");
        assert_eq!(out.len(), input.len());
        for (idx, v) in out.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "constant input idx {idx} should be unchanged, got {v}"
            );
        }
    }

    /// A synthetic step edge — left half 0.0, right half 1.0 — must be
    /// preserved by the bilateral filter (gradient remains > 0.5 across the
    /// boundary).
    #[test]
    fn test_bilateral_preserves_edges() {
        let w = 16u32;
        let h = 8u32;
        let mut input = vec![0.0_f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                input[(y * w + x) as usize] = if x < w / 2 { 0.0 } else { 1.0 };
            }
        }
        let params = BilateralParams {
            sigma_color: 0.05, // very sharp colour kernel preserves edges
            sigma_space: 2.0,
            kernel_radius: 3,
        };
        let out = bilateral_cpu(&input, w, h, params).expect("bilateral_cpu");
        // Sample gradient across the step (x = w/2 - 1 → w/2)
        let mid_y = h / 2;
        let left = out[(mid_y * w + (w / 2 - 1)) as usize];
        let right = out[(mid_y * w + (w / 2)) as usize];
        let grad = right - left;
        assert!(
            grad > 0.5,
            "edge gradient should remain > 0.5 but was {grad} (left={left}, right={right})"
        );
    }

    #[test]
    fn test_bilateral_cpu_rejects_oversize_radius() {
        let input = vec![0.0_f32; 16];
        let params = BilateralParams {
            sigma_color: 0.1,
            sigma_space: 1.0,
            kernel_radius: MAX_BILATERAL_RADIUS + 1,
        };
        assert!(bilateral_cpu(&input, 4, 4, params).is_err());
    }

    #[test]
    fn test_bilateral_cpu_rejects_size_mismatch() {
        let input = vec![0.0_f32; 15]; // 4×4 expected = 16
        let params = BilateralParams::default();
        let err = bilateral_cpu(&input, 4, 4, params).expect_err("should mismatch");
        assert!(matches!(err, AccelError::BufferSizeMismatch { .. }));
    }

    #[test]
    fn test_bilateral_params_default_is_valid() {
        let p = BilateralParams::default();
        assert!(p.validate().is_ok());
    }

    /// GPU path tests — skip when no adapter is available.
    #[cfg(feature = "webgpu")]
    fn try_init_device() -> Option<(std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>)> {
        use std::sync::Arc;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            // native/trusted context; limit-bucketing is only a browser-fingerprinting mitigation
            apply_limit_buckets: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bilateral-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: Default::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
        }))
        .ok()?;
        Some((Arc::new(device), Arc::new(queue)))
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn test_bilateral_gpu_matches_cpu_64x64() {
        let Some((device, queue)) = try_init_device() else {
            eprintln!("no gpu adapter — skipping test_bilateral_gpu_matches_cpu_64x64");
            return;
        };
        let w = 64u32;
        let h = 64u32;
        let mut input = vec![0.0_f32; (w * h) as usize];
        // Deterministic checkerboard + ramp pattern.
        for y in 0..h {
            for x in 0..w {
                let v = ((x ^ y) & 0xF) as f32 / 16.0;
                input[(y * w + x) as usize] = v;
            }
        }
        let params = BilateralParams {
            sigma_color: 0.15,
            sigma_space: 2.0,
            kernel_radius: 3,
        };
        let cpu_out = bilateral_cpu(&input, w, h, params).expect("cpu");
        let gpu = BilateralGpu::new(device, queue).expect("bilateral gpu pipeline");
        let gpu_out = gpu.process(&input, w, h, params).expect("gpu process");
        assert_eq!(gpu_out.len(), cpu_out.len());
        let mut max_diff = 0.0_f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        // Allow a generous tolerance for f32 GPU vs f32 CPU exp() drift.
        assert!(
            max_diff < 0.02,
            "GPU bilateral output deviates from CPU by {max_diff}"
        );
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn test_bilateral_gpu_constant_input() {
        let Some((device, queue)) = try_init_device() else {
            eprintln!("no gpu adapter — skipping test_bilateral_gpu_constant_input");
            return;
        };
        let w = 32u32;
        let h = 32u32;
        let input = vec![0.42_f32; (w * h) as usize];
        let params = BilateralParams {
            sigma_color: 0.1,
            sigma_space: 1.0,
            kernel_radius: 2,
        };
        let gpu = BilateralGpu::new(device, queue).expect("pipeline");
        let out = gpu.process(&input, w, h, params).expect("process");
        for v in &out {
            assert!(
                (v - 0.42).abs() < 1e-3,
                "constant input must be preserved, got {v}"
            );
        }
    }
}
