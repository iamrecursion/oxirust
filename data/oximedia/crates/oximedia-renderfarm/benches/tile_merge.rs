//! Criterion benchmark for `TileCompositor::merge` on a synthetic 8K frame.
use criterion::{criterion_group, criterion_main, Criterion};
use oximedia_renderfarm::tile_render::{Rect, TileCompositor, TilePixels};
use std::hint::black_box;

fn bench_tile_merge_8k(c: &mut Criterion) {
    const W: u32 = 7680;
    const H: u32 = 4320;
    const BPP: u32 = 4;
    const TILE_W: u32 = 480;
    const TILE_H: u32 = 270;

    let compositor = TileCompositor::new(W, H, BPP);

    let tiles: Vec<TilePixels> = (0..16u32)
        .flat_map(|row| {
            (0..16u32).map(move |col| {
                let x = col * TILE_W;
                let y = row * TILE_H;
                let w = TILE_W.min(W.saturating_sub(x));
                let h = TILE_H.min(H.saturating_sub(y));
                TilePixels {
                    rect: Rect::new(x, y, w, h),
                    data: vec![128u8; (w * h * BPP) as usize],
                }
            })
        })
        .collect();

    c.bench_function("tile_merge_8k_256tiles", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            compositor
                .merge(black_box(&tiles), &mut output)
                .expect("merge ok");
            black_box(output);
        });
    });
}

criterion_group!(benches, bench_tile_merge_8k);
criterion_main!(benches);
