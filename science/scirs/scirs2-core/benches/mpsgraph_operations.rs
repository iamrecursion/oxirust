//! MPSGraph Operations Performance Benchmarks
//!
//! Comprehensive benchmarks measuring MPSGraph operation performance on Apple Silicon.
//!
//! ## Benchmark Categories
//!
//! 1. **Matrix Operations**: matmul with various sizes (small, medium, large, transformer)
//! 2. **Attention Operations**: Scaled Dot-Product Attention with causal masking
//! 3. **Activation Functions**: GeLU, SiLU, ReLU
//! 4. **Normalization**: LayerNorm, RMSNorm
//! 5. **End-to-End**: Transformer block simulation (rinna-1b GPT-2 sizes)
//!
//! ## Performance Targets (from MPSGRAPH.md)
//!
//! - SDPA: 10-30x speedup over basic Metal
//! - Matmul: 5-10x speedup
//! - GeLU: 10-20x speedup (operator stitching)
//! - End-to-end: 50-200x for transformer models
//!
//! ## Running Benchmarks
//!
//! ```bash
//! # All MPSGraph benchmarks
//! cargo bench --bench mpsgraph_operations --features mpsgraph
//!
//! # Specific operation
//! cargo bench --bench mpsgraph_operations matmul --features mpsgraph
//! ```
//!
//! ## Current Status
//!
//! ⚠️ **Note**: Graph execution is currently stubbed due to objc2 runtime limitations.
//! These benchmarks measure graph construction overhead. Once execution is enabled,
//! they will measure actual GPU performance and verify the 50-200x speedup targets.

#![cfg(all(feature = "mpsgraph", target_os = "macos"))]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
use scirs2_core::gpu::backends::metal_mpsgraph::MPSGraphContext;
use std::hint::black_box;

/// Type alias for Metal context setup return type
#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
type MetalContextSetup = (
    Retained<ProtocolObject<dyn MTLDevice>>,
    Retained<ProtocolObject<dyn MTLCommandQueue>>,
    MPSGraphContext,
);

/// Initialize Metal device and MPSGraph context (shared across benchmarks)
#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
fn setup_context() -> MetalContextSetup {
    unsafe {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device required");
        let queue: Retained<ProtocolObject<dyn MTLCommandQueue>> =
            msg_send![&device, newCommandQueue];
        let ctx = MPSGraphContext::new(device.clone(), queue.clone());
        (device, queue, ctx)
    }
}

/// Create a Metal buffer with specified size in bytes
#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
fn create_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    size_bytes: u64,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        msg_send![
            device,
            newBufferWithLength: size_bytes,
            options: objc2_metal::MTLResourceOptions::StorageModeShared
        ]
    }
}

// ============================================================================
// Matrix Multiplication Benchmarks
// ============================================================================

fn bench_matmul_sizes(c: &mut Criterion) {
    let (device, _queue, ctx) = setup_context();

    let mut group = c.benchmark_group("mpsgraph_matmul");

    // Test various matrix sizes relevant to transformer models
    let sizes = vec![
        ("small_64x64", 64, 64, 64),
        ("medium_256x256", 256, 256, 256),
        ("large_1024x1024", 1024, 1024, 1024),
        ("transformer_qk", 2048, 64, 2048), // Q @ K^T in attention
        ("transformer_av", 2048, 2048, 64), // Attn @ V
        ("rinna_ffn", 2048, 768, 3072),     // rinna-1b feedforward
    ];

    for (name, m, k, n) in sizes {
        let throughput = Throughput::Elements((m * k + k * n + m * n) as u64);
        group.throughput(throughput);

        let a_buffer = create_buffer(&device, (m * k * 4) as u64);
        let b_buffer = create_buffer(&device, (k * n * 4) as u64);

        group.bench_with_input(BenchmarkId::from_parameter(name), &(m, k, n), |b, _| {
            b.iter(|| {
                ctx.matmul(
                    black_box(&a_buffer),
                    black_box(&b_buffer),
                    black_box(m),
                    black_box(k),
                    black_box(n),
                )
                .expect("matmul failed")
            });
        });
    }

    group.finish();
}

// ============================================================================
// Scaled Dot-Product Attention Benchmarks
// ============================================================================

fn bench_sdpa_sizes(c: &mut Criterion) {
    let (device, _queue, ctx) = setup_context();

    let mut group = c.benchmark_group("mpsgraph_sdpa");

    // SDPA configurations: (batch, num_heads, seq_len, head_dim)
    let configs = vec![
        ("gpt2_base", 1, 12, 128, 64),
        ("gpt2_medium", 1, 16, 256, 64),
        ("rinna_1b", 1, 16, 512, 64),
        ("llama_style", 1, 32, 1024, 128),
    ];

    for (name, batch, num_heads, seq_len, head_dim) in configs {
        let qkv_elements = (batch * num_heads * seq_len * head_dim) as u64;
        let throughput = Throughput::Elements(qkv_elements * 3); // Q, K, V
        group.throughput(throughput);

        let qkv_size = (batch * num_heads * seq_len * head_dim * 4) as u64;
        let q_buffer = create_buffer(&device, qkv_size);
        let k_buffer = create_buffer(&device, qkv_size);
        let v_buffer = create_buffer(&device, qkv_size);

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(batch, num_heads, seq_len, head_dim),
            |b, _| {
                b.iter(|| {
                    ctx.scaled_dot_product_attention(
                        black_box(&q_buffer),
                        black_box(&k_buffer),
                        black_box(&v_buffer),
                        black_box(batch),
                        black_box(num_heads),
                        black_box(seq_len),
                        black_box(seq_len),
                        black_box(head_dim),
                        black_box(true), // use_causal_mask
                    )
                    .expect("SDPA failed")
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Activation Function Benchmarks
// ============================================================================

fn bench_activation_functions(c: &mut Criterion) {
    let (device, _queue, ctx) = setup_context();

    let mut group = c.benchmark_group("mpsgraph_activations");

    // Test with transformer-relevant sizes
    let shapes = vec![
        ("small", vec![1, 128, 768]),
        ("medium", vec![1, 512, 768]),
        ("rinna_ffn", vec![1, 2048, 3072]),
    ];

    for (size_name, shape) in shapes {
        let num_elements: usize = shape.iter().product();
        let throughput = Throughput::Elements(num_elements as u64);
        group.throughput(throughput);

        let buffer = create_buffer(&device, (num_elements * 4) as u64);

        // GeLU benchmark
        group.bench_with_input(BenchmarkId::new("gelu", size_name), &shape, |b, shape| {
            b.iter(|| {
                ctx.gelu(black_box(&buffer), black_box(shape))
                    .expect("GeLU failed")
            });
        });

        // SiLU benchmark
        group.bench_with_input(BenchmarkId::new("silu", size_name), &shape, |b, shape| {
            b.iter(|| {
                ctx.silu(black_box(&buffer), black_box(shape))
                    .expect("SiLU failed")
            });
        });

        // ReLU benchmark
        group.bench_with_input(BenchmarkId::new("relu", size_name), &shape, |b, shape| {
            b.iter(|| {
                ctx.relu(black_box(&buffer), black_box(shape))
                    .expect("ReLU failed")
            });
        });
    }

    group.finish();
}

// ============================================================================
// Normalization Benchmarks
// ============================================================================

fn bench_normalization(c: &mut Criterion) {
    let (device, _queue, ctx) = setup_context();

    let mut group = c.benchmark_group("mpsgraph_normalization");

    // LayerNorm configurations
    let configs = vec![
        ("gpt2_base", vec![1, 128, 768], vec![768]),
        ("rinna_1b", vec![1, 2048, 768], vec![768]),
        ("llama_style", vec![1, 1024, 4096], vec![4096]),
    ];

    for (name, input_shape, norm_shape) in configs {
        let num_elements: usize = input_shape.iter().product();
        let norm_elements: usize = norm_shape.iter().product();
        let throughput = Throughput::Elements(num_elements as u64);
        group.throughput(throughput);

        let input_buffer = create_buffer(&device, (num_elements * 4) as u64);
        let gamma_buffer = create_buffer(&device, (norm_elements * 4) as u64);
        let beta_buffer = create_buffer(&device, (norm_elements * 4) as u64);

        // LayerNorm benchmark
        group.bench_with_input(
            BenchmarkId::new("layer_norm", name),
            &(&input_shape, &norm_shape),
            |b, (input_shape, norm_shape)| {
                b.iter(|| {
                    ctx.layer_norm(
                        black_box(&input_buffer),
                        black_box(&gamma_buffer),
                        black_box(&beta_buffer),
                        black_box(input_shape),
                        black_box(norm_shape),
                        black_box(1e-5),
                    )
                    .expect("LayerNorm failed")
                });
            },
        );

        // RMSNorm benchmark
        group.bench_with_input(
            BenchmarkId::new("rms_norm", name),
            &(&input_shape, &norm_shape),
            |b, (input_shape, norm_shape)| {
                b.iter(|| {
                    ctx.rms_norm(
                        black_box(&input_buffer),
                        black_box(&gamma_buffer),
                        black_box(input_shape),
                        black_box(norm_shape),
                        black_box(1e-5),
                    )
                    .expect("RMSNorm failed")
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Softmax Benchmarks
// ============================================================================

fn bench_softmax(c: &mut Criterion) {
    let (device, _queue, ctx) = setup_context();

    let mut group = c.benchmark_group("mpsgraph_softmax");

    let shapes = vec![
        ("attention_scores", vec![1, 12, 128, 128]), // GPT-2 base attention
        ("rinna_attention", vec![1, 16, 512, 512]),  // rinna-1b attention
        ("large_vocab", vec![1, 50257]),             // GPT-2 vocabulary
    ];

    for (name, shape) in shapes {
        let num_elements: usize = shape.iter().product();
        let throughput = Throughput::Elements(num_elements as u64);
        group.throughput(throughput);

        let buffer = create_buffer(&device, (num_elements * 4) as u64);

        group.bench_with_input(BenchmarkId::from_parameter(name), &shape, |b, shape| {
            b.iter(|| {
                ctx.softmax(black_box(&buffer), black_box(shape), black_box(-1))
                    .expect("Softmax failed")
            });
        });
    }

    group.finish();
}

// ============================================================================
// End-to-End Transformer Block Benchmark
// ============================================================================

fn bench_transformer_block(c: &mut Criterion) {
    let (device, _queue, ctx) = setup_context();

    let mut group = c.benchmark_group("mpsgraph_transformer_block");

    // Simulates a single transformer block forward pass (rinna-1b configuration)
    // Includes: LayerNorm → SDPA → LayerNorm → FFN (matmul + GeLU + matmul)
    let batch = 1;
    let seq_len = 512;
    let hidden_dim = 768;
    let num_heads = 16;
    let head_dim = hidden_dim / num_heads;
    let ffn_dim = 3072;

    let throughput = Throughput::Elements((batch * seq_len * hidden_dim) as u64);
    group.throughput(throughput);

    // Pre-allocate all buffers
    let hidden_buffer = create_buffer(&device, (batch * seq_len * hidden_dim * 4) as u64);
    let qkv_buffer = create_buffer(&device, (batch * num_heads * seq_len * head_dim * 4) as u64);
    let gamma_buffer = create_buffer(&device, (hidden_dim * 4) as u64);
    let beta_buffer = create_buffer(&device, (hidden_dim * 4) as u64);
    let ffn1_buffer = create_buffer(&device, (hidden_dim * ffn_dim * 4) as u64);
    let ffn2_buffer = create_buffer(&device, (ffn_dim * hidden_dim * 4) as u64);

    group.bench_function("rinna_1b_block", |b| {
        b.iter(|| {
            // Pre-attention LayerNorm
            ctx.layer_norm(
                black_box(&hidden_buffer),
                black_box(&gamma_buffer),
                black_box(&beta_buffer),
                black_box(&[batch, seq_len, hidden_dim]),
                black_box(&[hidden_dim]),
                black_box(1e-5),
            )
            .expect("LayerNorm 1 failed");

            // Self-attention (SDPA)
            ctx.scaled_dot_product_attention(
                black_box(&qkv_buffer),
                black_box(&qkv_buffer),
                black_box(&qkv_buffer),
                black_box(batch),
                black_box(num_heads),
                black_box(seq_len),
                black_box(seq_len),
                black_box(head_dim),
                black_box(true),
            )
            .expect("SDPA failed");

            // Post-attention LayerNorm
            ctx.layer_norm(
                black_box(&hidden_buffer),
                black_box(&gamma_buffer),
                black_box(&beta_buffer),
                black_box(&[batch, seq_len, hidden_dim]),
                black_box(&[hidden_dim]),
                black_box(1e-5),
            )
            .expect("LayerNorm 2 failed");

            // FFN: matmul + GeLU
            ctx.matmul(
                black_box(&hidden_buffer),
                black_box(&ffn1_buffer),
                black_box(batch * seq_len),
                black_box(hidden_dim),
                black_box(ffn_dim),
            )
            .expect("FFN matmul 1 failed");

            ctx.gelu(
                black_box(&hidden_buffer),
                black_box(&[batch, seq_len, ffn_dim]),
            )
            .expect("GeLU failed");

            // FFN: matmul back to hidden_dim
            ctx.matmul(
                black_box(&hidden_buffer),
                black_box(&ffn2_buffer),
                black_box(batch * seq_len),
                black_box(ffn_dim),
                black_box(hidden_dim),
            )
            .expect("FFN matmul 2 failed");
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    mpsgraph_benches,
    bench_matmul_sizes,
    bench_sdpa_sizes,
    bench_activation_functions,
    bench_normalization,
    bench_softmax,
    bench_transformer_block,
);

criterion_main!(mpsgraph_benches);

#[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
fn main() {
    println!("MPSGraph benchmarks require macOS with mpsgraph feature enabled");
    println!("Run with: cargo bench --bench mpsgraph_operations --features mpsgraph");
}
