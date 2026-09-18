//! Wave 29 / Slice 4 — deterministic CPU-path known-answer tests for
//! `oximedia-gpu`.
//!
//! These tests exercise the CPU-testable surface of the GPU crate ONLY — no
//! physical GPU device, no `wgpu` adapter, no shader execution. Every oracle
//! here is derived by reading the corresponding `src/*.rs` implementation:
//!
//! * `shader_cache` — FNV-1a hashing, in-process LRU eviction, disk roundtrip.
//! * `workgroup`    — workgroup auto-tuning + shared-memory layout sizing.
//! * `color_convert_kernel` — RGB↔YUV buffer conversion (BT.709), range expand.
//! * `indirect_dispatch` — little-endian `[u32; 3]` (de)serialisation.
//!
//! Test files may use `.expect()` (never `unwrap()` in production code, but
//! these are tests). Pedantic cast lints are pre-allowed below for the few
//! deliberate numeric casts in oracle assertions.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::unreadable_literal)]

use oximedia_gpu::color_convert_kernel::{ColorConvertKernel, ColorStandard, RangeMode};
use oximedia_gpu::indirect_dispatch::IndirectDispatchArgs;
use oximedia_gpu::shader_cache::{
    hash_source, CompiledShader, DiskShaderCache, EvictionPolicy, GpuShaderCache, ShaderVersion,
};
use oximedia_gpu::workgroup::{
    DeviceLimits, SharedMemoryLayout, WorkgroupAutoTuner, WorkgroupSize,
};

// =============================================================================
// (a) shader_cache
// =============================================================================

/// FNV-1a 64-bit golden vectors, computed independently and matched against the
/// `hash_source` constants (`FNV_OFFSET = 14695981039346656037`,
/// `FNV_PRIME = 1099511628211`).
#[test]
fn fnv1a_known_answers() {
    // Empty input returns the FNV offset basis unchanged.
    assert_eq!(hash_source(b""), 14695981039346656037);
    assert_eq!(hash_source(b"abc"), 16654208175385433931);
    assert_eq!(hash_source(b"hello world shader"), 10622985475639379208);
}

#[test]
fn fnv1a_deterministic_and_distinct() {
    let a = b"// shader source version one\nvoid main() {}";
    let b = b"// shader source version two\nvoid main() { discard; }";

    // Determinism: same bytes -> same hash, every time.
    assert_eq!(hash_source(a), hash_source(a));
    assert_eq!(hash_source(b), hash_source(b));

    // Distinctness: different bytes -> (overwhelmingly likely) different hash.
    assert_ne!(hash_source(a), hash_source(b));

    // A single-byte whitespace change must perturb the hash.
    assert_ne!(
        hash_source(b"void main(){}"),
        hash_source(b"void main() {}")
    );
}

/// Helper: build a `CompiledShader` keyed only by `source_hash` (backend fixed).
fn shader_with_hash(hash: u64, size: usize) -> CompiledShader {
    CompiledShader::new(vec![0u8; size], ShaderVersion::new(hash, "vulkan", 0))
}

/// LRU eviction: with `max_entries = 3`, insert 3 entries, *touch* the first to
/// bump its recency, then insert a 4th. The least-recently-used survivor logic
/// means the just-bumped entry (1) must remain, and the genuinely
/// least-recently-used entry (2) must be evicted.
///
/// NOTE on the implementation: `needs_eviction` triggers when
/// `entry_count >= max_entries`, so inserting the 4th entry (when 3 are present)
/// evicts exactly one before storing. The LRU victim is chosen by the oldest
/// `last_access` timestamp.
#[test]
fn lru_eviction_evicts_least_recently_used() {
    let mut cache = GpuShaderCache::new(usize::MAX, 3, EvictionPolicy::Lru);

    let v1 = ShaderVersion::new(1, "vulkan", 0);
    let v2 = ShaderVersion::new(2, "vulkan", 0);
    let v3 = ShaderVersion::new(3, "vulkan", 0);
    let v4 = ShaderVersion::new(4, "vulkan", 0);

    // Insert 1, 2, 3 in order. Access order (oldest -> newest): 1, 2, 3.
    cache.insert(shader_with_hash(1, 16));
    cache.insert(shader_with_hash(2, 16));
    cache.insert(shader_with_hash(3, 16));
    assert_eq!(cache.len(), 3);

    // Touch entry 1 -> it becomes the most-recently-used. Order: 2, 3, 1.
    assert!(
        cache.get(&v1).is_some(),
        "entry 1 must be present before touch"
    );

    // Insert entry 4 -> cache is full, one LRU victim is evicted. Victim = 2.
    cache.insert(shader_with_hash(4, 16));
    assert_eq!(cache.len(), 3, "len must stay at the entry cap");

    assert!(cache.contains(&v1), "touched entry 1 must survive");
    assert!(
        !cache.contains(&v2),
        "least-recently-used entry 2 must be evicted"
    );
    assert!(cache.contains(&v3), "entry 3 must survive");
    assert!(
        cache.contains(&v4),
        "freshly-inserted entry 4 must be present"
    );

    assert!(
        cache.stats().evictions >= 1,
        "at least one eviction must have been recorded"
    );
}

/// LFU eviction: the lowest `hit_count` entry is removed. After inserting 1, 2,
/// 3 (cap 3) and hitting 2 and 3, inserting a 4th must evict entry 1 (zero hits).
#[test]
fn lfu_eviction_evicts_least_frequently_used() {
    let mut cache = GpuShaderCache::new(usize::MAX, 3, EvictionPolicy::Lfu);
    let v1 = ShaderVersion::new(1, "vulkan", 0);
    let v2 = ShaderVersion::new(2, "vulkan", 0);
    let v3 = ShaderVersion::new(3, "vulkan", 0);

    cache.insert(shader_with_hash(1, 16));
    cache.insert(shader_with_hash(2, 16));
    cache.insert(shader_with_hash(3, 16));

    // Raise hit counts for 2 and 3; entry 1 keeps hit_count == 0.
    let _ = cache.get(&v2);
    let _ = cache.get(&v3);

    cache.insert(shader_with_hash(4, 16));
    assert_eq!(cache.len(), 3);
    assert!(
        !cache.contains(&v1),
        "zero-hit entry 1 must be the LFU victim"
    );
    assert!(cache.contains(&v2));
    assert!(cache.contains(&v3));
}

/// Disk roundtrip: `put` then `get` returns byte-identical bytecode, and the
/// stats counters advance. Uses `std::env::temp_dir()` and cleans up.
#[test]
fn disk_cache_roundtrip_byte_identical() {
    let dir = std::env::temp_dir().join("oximedia_gpu_w29_disk_roundtrip");
    let _ = std::fs::remove_dir_all(&dir); // clean slate

    let mut disk = DiskShaderCache::open(&dir).expect("open disk shader cache");
    let version = ShaderVersion::new(0xDEAD_BEEF_u64, "metal", 7);
    let bytecode: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
    let shader = CompiledShader::new(bytecode.clone(), version.clone());

    disk.put(&shader);
    let read_back = disk
        .get(&version)
        .expect("stored shader must be retrievable");

    assert_eq!(read_back, bytecode, "disk roundtrip must be byte-identical");
    assert_eq!(disk.stats().disk_writes, 1, "exactly one write recorded");
    assert_eq!(disk.stats().disk_hits, 1, "exactly one hit recorded");

    // A non-existent version is a miss, not a hit.
    let other = ShaderVersion::new(0x1234_u64, "metal", 7);
    assert!(disk.get(&other).is_none(), "unknown version must miss");
    assert_eq!(disk.stats().disk_misses, 1, "exactly one miss recorded");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Disk cache `.shd` / `.meta` file pair is actually written to disk under the
/// expected key, and `clear` removes both. Confirms the on-disk layout matches
/// the documented `<hex_hash>_<backend>_<flags>.{shd,meta}` scheme.
#[test]
fn disk_cache_writes_file_pair_and_clears() {
    let dir = std::env::temp_dir().join("oximedia_gpu_w29_disk_filepair");
    let _ = std::fs::remove_dir_all(&dir);

    let mut disk = DiskShaderCache::open(&dir).expect("open disk shader cache");
    let version = ShaderVersion::new(0xABCD_u64, "dx12", 3);
    disk.put(&CompiledShader::new(vec![9u8; 32], version));

    let shd = dir.join("000000000000abcd_dx12_3.shd");
    let meta = dir.join("000000000000abcd_dx12_3.meta");
    assert!(
        shd.exists(),
        "bytecode .shd file must exist at expected key"
    );
    assert!(
        meta.exists(),
        "metadata .meta file must exist at expected key"
    );

    disk.clear();
    assert!(!shd.exists(), ".shd must be removed by clear()");
    assert!(!meta.exists(), ".meta must be removed by clear()");

    let _ = std::fs::remove_dir_all(&dir);
}

// =============================================================================
// (b) workgroup
// =============================================================================

/// 1D auto-tune for a one-million-element problem. With default device limits
/// (max total 1024, subgroup 32) the tuner picks a 256-wide workgroup, and the
/// dispatch covers all elements: ceil(1_000_000 / 256) = 3907.
#[test]
fn tune_1d_one_million_elements() {
    let tuner = WorkgroupAutoTuner::new(DeviceLimits::default());
    let (wg, dispatch) = tuner.tune_1d(1_000_000);

    assert_eq!(wg.x, 256, "expected 256-wide workgroup");
    assert_eq!(wg.y, 1);
    assert_eq!(wg.z, 1);
    assert_eq!(dispatch.groups_x, 3907, "ceil(1_000_000 / 256) == 3907");
    assert_eq!(dispatch.groups_y, 1);
    assert_eq!(dispatch.groups_z, 1);

    // Invariants: valid, warp-aligned, fully covering the problem.
    assert!(wg.is_valid());
    assert!(wg.is_warp_aligned());
    assert!(
        u64::from(dispatch.groups_x) * u64::from(wg.x) >= 1_000_000,
        "dispatch must cover all elements"
    );
}

/// 2D auto-tune for 1080p. The efficiency-maximising candidate is (32, 8):
/// it achieves a perfect 1.0 efficiency (60*135*256 == 1920*1080) versus
/// (16,16)'s 0.9926, so the tuner prefers it. Dispatch = (60, 135, 1).
#[test]
fn tune_2d_1080p_picks_32x8() {
    let tuner = WorkgroupAutoTuner::new(DeviceLimits::default());
    let (wg, dispatch) = tuner.tune_2d(1920, 1080);

    assert_eq!(wg.x, 32, "expected 32-wide workgroup");
    assert_eq!(wg.y, 8, "expected 8-tall workgroup");
    assert_eq!(wg.z, 1);
    assert_eq!(dispatch.groups_x, 60, "ceil(1920 / 32) == 60");
    assert_eq!(dispatch.groups_y, 135, "ceil(1080 / 8) == 135");
    assert_eq!(dispatch.groups_z, 1);

    // Invariants.
    assert!(wg.is_valid());
    assert!(wg.is_warp_aligned());
    assert!(u64::from(dispatch.groups_x) * u64::from(wg.x) >= 1920);
    assert!(u64::from(dispatch.groups_y) * u64::from(wg.y) >= 1080);
}

/// `WorkgroupSize::is_valid` boundary conditions against the 1024-total /
/// 1024-per-dim limits.
#[test]
fn workgroup_is_valid_boundaries() {
    // Total exactly 1024, every dim within limits -> valid.
    assert!(
        WorkgroupSize::new(1024, 1, 1).is_valid(),
        "total 1024 is valid"
    );
    assert!(
        WorkgroupSize::new(32, 32, 1).is_valid(),
        "32*32 == 1024 is valid"
    );

    // x exceeds MAX_WORKGROUP_DIM (1024) -> invalid (also total 1025).
    let over_dim = WorkgroupSize::new(1025, 1, 1);
    assert_eq!(over_dim.total(), 1025);
    assert!(
        !over_dim.is_valid(),
        "1025 exceeds per-dim and total limits"
    );

    // Each dim within the per-dim cap but total 2048 -> invalid by total.
    let over_total = WorkgroupSize::new(32, 64, 1);
    assert_eq!(over_total.total(), 2048);
    assert!(!over_total.is_valid(), "total 2048 exceeds the 1024 cap");

    // Zero dimension -> invalid.
    assert!(
        !WorkgroupSize::new(0, 1, 1).is_valid(),
        "zero dim is invalid"
    );
}

/// `SharedMemoryLayout::new(element_count, element_size, alignment)` computes
/// `size_bytes = element_count * round_up(element_size, alignment)`.
#[test]
fn shared_memory_layout_sizing() {
    // 256 elements * round_up(4, 4)=4 = 1024 bytes.
    let l1 = SharedMemoryLayout::new(256, 4, 4);
    assert_eq!(l1.size_bytes, 1024);
    assert_eq!(l1.element_count, 256);
    assert_eq!(l1.element_size, 4);
    assert_eq!(l1.alignment, 4);
    assert!(l1.fits_in_shared_memory());

    // 100 elements * round_up(6, 8)=8 = 800 bytes (6-byte element padded to 8).
    let l2 = SharedMemoryLayout::new(100, 6, 8);
    assert_eq!(l2.size_bytes, 800);

    // 50000 elements * round_up(4, 4)=4 = 200000 bytes -> exceeds 48 KiB limit.
    let l3 = SharedMemoryLayout::new(50_000, 4, 4);
    assert_eq!(l3.size_bytes, 200_000);
    assert!(!l3.fits_in_shared_memory(), "200000 bytes exceeds 48 KiB");
}

// =============================================================================
// (c) color_convert_kernel
// =============================================================================

/// Build a 1×1 RGBA pixel buffer.
fn rgba1(r: u8, g: u8, b: u8) -> Vec<u8> {
    vec![r, g, b, 255]
}

/// BT.709 RGB→YUV known answers for full and limited range, 1×1 pixel.
/// White and black map to fixed Y values; chroma stays neutral at 128.
#[test]
fn rgb_to_yuv_bt709_white_black_known_answers() {
    // ── FULL range ──
    let full = ColorConvertKernel::new(ColorStandard::Bt709, RangeMode::Full);

    let white = rgba1(255, 255, 255);
    let mut dst = vec![0u8; 4];
    full.convert_rgb_to_yuv(&white, &mut dst, 1, 1)
        .expect("full white conversion");
    assert_eq!((dst[0], dst[1], dst[2]), (255, 128, 128), "FULL white");
    assert_eq!(dst[3], 255, "alpha pass-through");

    let black = rgba1(0, 0, 0);
    let mut dst = vec![0u8; 4];
    full.convert_rgb_to_yuv(&black, &mut dst, 1, 1)
        .expect("full black conversion");
    assert_eq!((dst[0], dst[1], dst[2]), (0, 128, 128), "FULL black");

    // ── LIMITED (studio) range ──
    let limited = ColorConvertKernel::new(ColorStandard::Bt709, RangeMode::Limited);

    let mut dst = vec![0u8; 4];
    limited
        .convert_rgb_to_yuv(&rgba1(255, 255, 255), &mut dst, 1, 1)
        .expect("limited white conversion");
    assert_eq!((dst[0], dst[1], dst[2]), (235, 128, 128), "LIMITED white");

    let mut dst = vec![0u8; 4];
    limited
        .convert_rgb_to_yuv(&rgba1(0, 0, 0), &mut dst, 1, 1)
        .expect("limited black conversion");
    assert_eq!((dst[0], dst[1], dst[2]), (16, 128, 128), "LIMITED black");
}

/// `expand_limited_to_full` (static fn) maps studio-swing Y/Cb/Cr to full range.
/// Limited black [16,128,128] -> [0,128,128]; limited white [235,240,240] ->
/// [255,255,255]. (Cb/Cr at 240 is the limited chroma maximum.)
#[test]
fn expand_limited_to_full_known_answers() {
    let mut dst = vec![0u8; 4];
    ColorConvertKernel::expand_limited_to_full(&[16, 128, 128, 255], &mut dst, 1, 1)
        .expect("expand limited black");
    assert_eq!(
        (dst[0], dst[1], dst[2]),
        (0, 128, 128),
        "limited black -> full"
    );
    assert_eq!(dst[3], 255, "alpha pass-through");

    let mut dst = vec![0u8; 4];
    ColorConvertKernel::expand_limited_to_full(&[235, 240, 240, 255], &mut dst, 1, 1)
        .expect("expand limited white");
    assert_eq!(
        (dst[0], dst[1], dst[2]),
        (255, 255, 255),
        "limited white -> full"
    );
}

/// RGB→YUV→RGB roundtrip is exact for white/black and within ±1 for a mid color
/// (BT.709 full range). Confirms the forward/inverse matrices are consistent.
#[test]
fn rgb_yuv_roundtrip_bt709_full() {
    let std = ColorStandard::Bt709;
    let range = RangeMode::Full;

    // Exact roundtrip for pure white and black.
    for &(r, g, b) in &[(255u8, 255u8, 255u8), (0u8, 0u8, 0u8)] {
        let src = rgba1(r, g, b);
        let mut yuv = vec![0u8; 4];
        ColorConvertKernel::rgb_to_yuv(&src, &mut yuv, 1, 1, std, range).expect("fwd");
        let mut back = vec![0u8; 4];
        ColorConvertKernel::yuv_to_rgb(&yuv, &mut back, 1, 1, std, range).expect("inv");
        assert_eq!(
            (back[0], back[1], back[2]),
            (r, g, b),
            "exact roundtrip for ({r},{g},{b})"
        );
    }

    // Mid color: roundtrip within ±1 LSB on each channel.
    let mid = rgba1(120, 90, 200);
    let mut yuv = vec![0u8; 4];
    ColorConvertKernel::rgb_to_yuv(&mid, &mut yuv, 1, 1, std, range).expect("fwd mid");
    let mut back = vec![0u8; 4];
    ColorConvertKernel::yuv_to_rgb(&yuv, &mut back, 1, 1, std, range).expect("inv mid");
    for i in 0..3 {
        let diff = (i32::from(mid[i]) - i32::from(back[i])).abs();
        assert!(
            diff <= 1,
            "channel {i}: |{}-{}| = {diff} > 1",
            mid[i],
            back[i]
        );
    }
}

// =============================================================================
// (d) indirect_dispatch
// =============================================================================

/// `IndirectDispatchArgs::to_bytes` serialises three little-endian u32 values.
#[test]
fn indirect_args_to_bytes_little_endian() {
    let bytes = IndirectDispatchArgs::new(1, 2, 3).to_bytes();
    assert_eq!(bytes, [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
}

/// `from_bytes` roundtrips a 12-byte buffer to an equal value, and rejects any
/// slice shorter than 12 bytes by returning `None`.
#[test]
fn indirect_args_from_bytes_roundtrip_and_short_reject() {
    let original = IndirectDispatchArgs::new(0xDEAD, 0xBEEF, 0x1234);
    let bytes = original.to_bytes();
    let restored = IndirectDispatchArgs::from_bytes(&bytes).expect("12 valid bytes deserialise");
    assert_eq!(restored, original, "roundtrip must be lossless");

    // An 11-byte slice is too short.
    assert!(
        IndirectDispatchArgs::from_bytes(&[0u8; 11]).is_none(),
        "11 bytes must be rejected"
    );
    // Empty slice is too short.
    assert!(
        IndirectDispatchArgs::from_bytes(&[]).is_none(),
        "empty slice must be rejected"
    );
}
