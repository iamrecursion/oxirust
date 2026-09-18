//! Temporal noise reduction with motion-adaptive IIR alpha blending.
//!
//! The temporal NR filter blends the current frame with up to **two** prior
//! frames.  For each pixel the absolute difference between current and prior
//! is normalised against a `motion_threshold`; in static areas the blend
//! weight is high (heavy averaging → noise suppression), while moving areas
//! receive little prior contribution (preserve sharpness).
//!
//! Per-pixel weight:
//! ```text
//! w_i = (1 - clamp(|cur - prev_i| / motion_threshold, 0, 1)) * strength
//! out = (cur + Σ prev_i * w_i) / (1 + Σ w_i)
//! ```
//!
//! This module exposes:
//! - [`temporal_nr_cpu`] — scalar reference (always-on, deterministic).
//! - `TemporalNr` — WebGPU dispatcher (feature `webgpu`) backed by a ring
//!   buffer of `wgpu::Buffer`s holding the previous frames.  The ring uses a
//!   `VecDeque` that pops the oldest buffer when capacity is exceeded — the
//!   dropped `wgpu::Buffer` releases its GPU allocation immediately, giving
//!   bounded memory regardless of how long the pipeline runs.

use crate::error::{AccelError, AccelResult};

/// Temporal NR parameters (motion sensitivity + blending strength).
#[derive(Debug, Clone, Copy)]
pub struct TemporalNrParams {
    /// Maximum |Δ| treated as "static" — larger differences fall to weight 0.
    pub motion_threshold: f32,
    /// Scaling applied to the static-area weight (0–1 typically).
    pub strength: f32,
}

impl Default for TemporalNrParams {
    fn default() -> Self {
        Self {
            motion_threshold: 0.1,
            strength: 0.8,
        }
    }
}

impl TemporalNrParams {
    /// Validate the parameters.
    ///
    /// # Errors
    /// Returns [`AccelError::InvalidDimensions`] when values are non-finite or
    /// when `motion_threshold <= 0` or `strength` is outside [0, 4].
    pub fn validate(&self) -> AccelResult<()> {
        if !self.motion_threshold.is_finite() || self.motion_threshold <= 0.0 {
            return Err(AccelError::InvalidDimensions(format!(
                "temporal NR motion_threshold must be positive and finite, got {}",
                self.motion_threshold
            )));
        }
        if !self.strength.is_finite() || !(0.0..=4.0).contains(&self.strength) {
            return Err(AccelError::InvalidDimensions(format!(
                "temporal NR strength must be in [0, 4], got {}",
                self.strength
            )));
        }
        Ok(())
    }
}

/// CPU reference for temporal noise reduction.
///
/// Operation order matches the WGSL shader (sum and wsum accumulated
/// separately, then divided), so GPU-vs-CPU comparison is well-conditioned.
///
/// `prev_frames` length controls how many priors are blended (capped at 2 —
/// extra entries are ignored).
///
/// # Errors
/// - [`AccelError::BufferSizeMismatch`] if any buffer length ≠ `width*height`.
/// - [`AccelError::InvalidDimensions`] if `params.validate()` fails.
pub fn temporal_nr_cpu(
    current: &[f32],
    prev_frames: &[&[f32]],
    width: u32,
    height: u32,
    params: TemporalNrParams,
) -> AccelResult<Vec<f32>> {
    params.validate()?;
    let n = (width as usize) * (height as usize);
    if current.len() != n {
        return Err(AccelError::BufferSizeMismatch {
            expected: n,
            actual: current.len(),
        });
    }
    for (i, p) in prev_frames.iter().enumerate() {
        if p.len() != n {
            return Err(AccelError::BufferSizeMismatch {
                expected: n,
                actual: p.len(),
            });
        }
        // Limit at two prev frames (mirrors GPU shader bindings).
        if i >= 2 {
            break;
        }
    }

    let limit = prev_frames.len().min(2);
    let mut out = vec![0.0_f32; n];
    for idx in 0..n {
        let c = current[idx];
        let mut sum = c;
        let mut wsum: f32 = 1.0;
        for prev in &prev_frames[..limit] {
            let p = prev[idx];
            let d = (c - p).abs();
            let normalised = (d / params.motion_threshold).clamp(0.0, 1.0);
            let w = (1.0 - normalised) * params.strength;
            sum += p * w;
            wsum += w;
        }
        out[idx] = sum / wsum;
    }
    Ok(out)
}

// ── GPU path (feature = "webgpu") ───────────────────────────────────────────

#[cfg(feature = "webgpu")]
pub use gpu_impl::TemporalNr;

#[cfg(feature = "webgpu")]
mod gpu_impl {
    use super::*;
    use crate::shaders::TEMPORAL_NR_WGSL;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use wgpu::util::DeviceExt;

    /// Uniform layout matching `TemporalParams` in WGSL (32 bytes).
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct TemporalParamsUbo {
        width: u32,
        height: u32,
        motion_threshold: f32,
        strength: f32,
        num_prev_frames: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }
    unsafe impl bytemuck::Zeroable for TemporalParamsUbo {}
    unsafe impl bytemuck::Pod for TemporalParamsUbo {}

    /// GPU temporal noise reduction with a ring buffer of previous frames.
    pub struct TemporalNr {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
        /// Ring buffer of the **N most recent** previous-frame GPU buffers.
        /// Bounded by `capacity`; oldest entry is popped on overflow.
        ring: VecDeque<wgpu::Buffer>,
        capacity: usize,
        /// Persistent zero buffer used as a placeholder when fewer than two
        /// previous frames are available (WGSL requires all bindings).
        zero_buffer: Option<wgpu::Buffer>,
        zero_buffer_len: u64,
    }

    impl TemporalNr {
        /// Build a new temporal NR pipeline.
        ///
        /// `capacity` is the maximum number of previous frames retained in
        /// GPU memory (1, 2, or more — the kernel uses up to 2).
        ///
        /// # Errors
        /// Returns [`AccelError::InvalidDimensions`] if `capacity == 0`, or
        /// pipeline-related errors from wgpu.
        pub fn new(
            capacity: usize,
            device: Arc<wgpu::Device>,
            queue: Arc<wgpu::Queue>,
        ) -> AccelResult<Self> {
            if capacity == 0 {
                return Err(AccelError::InvalidDimensions(
                    "temporal NR capacity must be > 0".to_string(),
                ));
            }
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("temporal-nr-wgsl"),
                source: wgpu::ShaderSource::Wgsl(TEMPORAL_NR_WGSL.into()),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("temporal-nr-bgl"),
                entries: &[
                    // cur
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
                    // prev0
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // prev1
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // output
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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
                label: Some("temporal-nr-pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("temporal-nr-pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("temporal_nr"),
                compilation_options: Default::default(),
                cache: None,
            });

            Ok(Self {
                device,
                queue,
                pipeline,
                bind_group_layout: bgl,
                ring: VecDeque::with_capacity(capacity),
                capacity,
                zero_buffer: None,
                zero_buffer_len: 0,
            })
        }

        /// Number of previous frames currently retained.
        pub fn ring_len(&self) -> usize {
            self.ring.len()
        }

        /// Maximum number of previous frames the ring may hold.
        pub fn capacity(&self) -> usize {
            self.capacity
        }

        /// Reset the ring (drop all retained previous-frame buffers).
        pub fn reset(&mut self) {
            self.ring.clear();
        }

        /// Process the current frame.
        ///
        /// On return the input has been appended to the ring (oldest popped
        /// if capacity exceeded) so subsequent calls can use it as a prior.
        ///
        /// # Errors
        /// Buffer-size mismatch, dispatch failure, or readback error.
        pub fn process(
            &mut self,
            current: &[f32],
            width: u32,
            height: u32,
            params: TemporalNrParams,
        ) -> AccelResult<Vec<f32>> {
            params.validate()?;
            let n = (width as usize) * (height as usize);
            if current.len() != n {
                return Err(AccelError::BufferSizeMismatch {
                    expected: n,
                    actual: current.len(),
                });
            }
            if width == 0 || height == 0 {
                return Ok(Vec::new());
            }

            let bytes_in: &[u8] = bytemuck::cast_slice(current);
            let buf_len = (n * std::mem::size_of::<f32>()) as u64;

            // Ensure persistent zero buffer matches the current frame size.
            // If size changed we must reallocate.
            if self.zero_buffer.is_none() || self.zero_buffer_len != buf_len {
                let zero_bytes = vec![0u8; buf_len as usize];
                let zb = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("temporal-nr-zero"),
                        contents: &zero_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
                self.zero_buffer = Some(zb);
                self.zero_buffer_len = buf_len;
                // Ring size assumptions break if frame size changed mid-stream.
                self.ring.clear();
            }

            // Upload current frame into a new GPU buffer.
            let cur_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("temporal-nr-cur"),
                    contents: bytes_in,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

            // Determine prev bindings.
            // Ring is in chronological order (oldest at front), so the most
            // recent prior is `back()`, second-most-recent is the entry before it.
            let zero = self
                .zero_buffer
                .as_ref()
                .ok_or_else(|| AccelError::BufferAllocation("zero buffer missing".to_string()))?;
            let ring_len = self.ring.len();
            let prev0 = if ring_len >= 1 {
                self.ring.get(ring_len - 1).unwrap_or(zero)
            } else {
                zero
            };
            let prev1 = if ring_len >= 2 {
                self.ring.get(ring_len - 2).unwrap_or(zero)
            } else {
                zero
            };
            let num_prev_frames = ring_len.min(2) as u32;

            // Output buffer.
            let dst_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("temporal-nr-dst"),
                size: buf_len,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let ubo = TemporalParamsUbo {
                width,
                height,
                motion_threshold: params.motion_threshold,
                strength: params.strength,
                num_prev_frames,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let uniform_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("temporal-nr-ubo"),
                    contents: bytemuck::bytes_of(&ubo),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("temporal-nr-bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cur_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: prev0.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: prev1.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("temporal-nr-encoder"),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("temporal-nr-cpass"),
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
                label: Some("temporal-nr-readback"),
                size: buf_len,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&dst_buf, 0, &readback, 0, buf_len);
            self.queue.submit(Some(encoder.finish()));

            let buf_slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buf_slice.map_async(wgpu::MapMode::Read, move |v| {
                let _ = tx.send(v);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| {
                    AccelError::Synchronization(format!("temporal NR poll failed: {e:?}"))
                })?;
            rx.recv()
                .map_err(|_| {
                    AccelError::Synchronization("temporal NR channel recv error".to_string())
                })?
                .map_err(|e| {
                    AccelError::Synchronization(format!("temporal NR map failed: {e:?}"))
                })?;

            let data = buf_slice.get_mapped_range().map_err(|e| {
                AccelError::MemoryMap(format!("temporal NR buffer map failed: {e}"))
            })?;
            let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
            drop(data);
            readback.unmap();

            // Push current onto the ring; pop oldest if at capacity.
            if self.ring.len() == self.capacity {
                let _drop_oldest = self.ring.pop_front();
            }
            self.ring.push_back(cur_buf);

            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 10 identical frames must yield identical output (no motion → stable
    /// blending; with equal inputs all weights are maximal).
    #[test]
    fn test_temporal_nr_static_input() {
        let w = 8u32;
        let h = 8u32;
        let frame = vec![0.5_f32; (w * h) as usize];
        let params = TemporalNrParams {
            motion_threshold: 0.1,
            strength: 0.8,
        };
        // Simulate a ring of two prior frames identical to current.
        let prev: Vec<&[f32]> = vec![&frame, &frame];
        let out = temporal_nr_cpu(&frame, &prev, w, h, params).expect("cpu");
        for v in &out {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "static input must be preserved, got {v}"
            );
        }
    }

    /// Motion-adaptive behaviour: pixels that match the prior receive heavy
    /// averaging (output is near the average), while pixels with a strong
    /// difference are kept close to the current value.
    #[test]
    fn test_temporal_nr_motion_adaptive() {
        let w = 4u32;
        let h = 4u32;
        let n = (w * h) as usize;
        let cur = vec![0.5_f32; n];
        let mut prev = vec![0.5_f32; n];
        // Introduce strong motion at pixel 0 only.
        prev[0] = 0.0;
        let params = TemporalNrParams {
            motion_threshold: 0.05, // 0.5 difference is way above threshold
            strength: 1.0,
        };
        let out = temporal_nr_cpu(&cur, &[&prev], w, h, params).expect("cpu");
        // Pixel 0 has high motion → should stay near current 0.5.
        assert!(
            (out[0] - 0.5).abs() < 1e-4,
            "moving pixel must stay sharp (cur=0.5), got {}",
            out[0]
        );
        // Pixel 1 has no motion (cur == prev == 0.5) → exact average is 0.5.
        assert!(
            (out[1] - 0.5).abs() < 1e-6,
            "static pixel must stay at 0.5, got {}",
            out[1]
        );
    }

    /// Heavy averaging of a noisy frame with a clean prior must move the
    /// output closer to the prior in static areas.
    #[test]
    fn test_temporal_nr_noise_attenuation_in_static_area() {
        let w = 4u32;
        let h = 4u32;
        let cur = vec![0.6_f32; (w * h) as usize];
        let prev = vec![0.4_f32; (w * h) as usize];
        let params = TemporalNrParams {
            // 0.2 delta is below threshold → strong weight
            motion_threshold: 1.0,
            strength: 1.0,
        };
        let out = temporal_nr_cpu(&cur, &[&prev], w, h, params).expect("cpu");
        // Weight w = (1 - 0.2/1.0) * 1.0 = 0.8 → out = (0.6 + 0.4*0.8)/(1+0.8)
        // = (0.6 + 0.32) / 1.8 = 0.5111…
        let expected = (0.6 + 0.4 * 0.8) / 1.8;
        for v in &out {
            assert!((v - expected).abs() < 1e-4, "expected ~{expected}, got {v}");
        }
    }

    #[test]
    fn test_temporal_nr_cpu_rejects_size_mismatch() {
        let cur = vec![0.0_f32; 15];
        let prev = vec![0.0_f32; 16];
        let params = TemporalNrParams::default();
        let err =
            temporal_nr_cpu(&cur, &[&prev], 4, 4, params).expect_err("expected size mismatch");
        assert!(matches!(err, AccelError::BufferSizeMismatch { .. }));
    }

    #[test]
    fn test_temporal_nr_params_default_is_valid() {
        assert!(TemporalNrParams::default().validate().is_ok());
    }

    #[test]
    fn test_temporal_nr_cpu_caps_at_two_priors() {
        let w = 2u32;
        let h = 2u32;
        let cur = vec![0.5_f32; 4];
        let p0 = vec![0.5_f32; 4];
        let p1 = vec![0.5_f32; 4];
        let p2 = vec![0.5_f32; 4]; // Should be ignored.
        let params = TemporalNrParams {
            motion_threshold: 1.0,
            strength: 1.0,
        };
        let out_three = temporal_nr_cpu(&cur, &[&p0, &p1, &p2], w, h, params).expect("cpu");
        let out_two = temporal_nr_cpu(&cur, &[&p0, &p1], w, h, params).expect("cpu");
        for (a, b) in out_three.iter().zip(out_two.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "third prior must be ignored, diff {}",
                (a - b).abs()
            );
        }
    }

    /// Ring capacity smoke test using the GPU dispatcher.  Push 20 frames
    /// into a ring of capacity 3 → assert ring length never exceeds 3.
    #[cfg(feature = "webgpu")]
    #[test]
    fn test_temporal_nr_ring_capacity_bounds() {
        let Some((device, queue)) = try_init_device() else {
            eprintln!("no gpu adapter — skipping test_temporal_nr_ring_capacity_bounds");
            return;
        };
        let mut tnr = TemporalNr::new(3, device, queue).expect("pipeline");
        let w = 8u32;
        let h = 8u32;
        let params = TemporalNrParams::default();
        for i in 0..20 {
            let frame = vec![i as f32 * 0.01; (w * h) as usize];
            let _ = tnr.process(&frame, w, h, params).expect("process");
            assert!(
                tnr.ring_len() <= 3,
                "ring overflow at iteration {i} → len {}",
                tnr.ring_len()
            );
        }
        assert_eq!(tnr.ring_len(), 3, "ring should be saturated at capacity");
    }

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
            label: Some("temporal-nr-test"),
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
    fn test_temporal_nr_gpu_static_input() {
        let Some((device, queue)) = try_init_device() else {
            eprintln!("no gpu adapter — skipping test_temporal_nr_gpu_static_input");
            return;
        };
        let mut tnr = TemporalNr::new(3, device, queue).expect("pipeline");
        let w = 16u32;
        let h = 16u32;
        let frame = vec![0.5_f32; (w * h) as usize];
        let params = TemporalNrParams::default();
        // Push 5 identical frames; final output should equal the input.
        let mut last = Vec::new();
        for _ in 0..5 {
            last = tnr.process(&frame, w, h, params).expect("process");
        }
        for v in &last {
            assert!(
                (v - 0.5).abs() < 1e-3,
                "static GPU output must preserve constant, got {v}"
            );
        }
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn test_temporal_nr_gpu_capacity_zero_rejected() {
        let Some((device, queue)) = try_init_device() else {
            eprintln!("no gpu adapter — skipping test_temporal_nr_gpu_capacity_zero_rejected");
            return;
        };
        let err = TemporalNr::new(0, device, queue);
        assert!(err.is_err());
    }
}
