use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_analytics::{
    event_buffer::{Event, EventBuffer},
    session::{analyze_sessions_batch, PlaybackEvent, ViewerSession},
};
use std::hint::black_box;

fn bench_event_buffer_push_drain(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_buffer");

    for n in [1_000usize, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("push_drain", n), &n, |b, &n| {
            b.iter(|| {
                let mut buf = EventBuffer::new(n + 1).expect("buffer");
                for i in 0..n {
                    let ev = Event::new(
                        black_box("play"),
                        black_box(format!("sess-{i}")),
                        black_box(i as u64 * 100),
                    );
                    buf.push(black_box(ev)).expect("push");
                }
                let drained = buf.drain();
                black_box(drained.len())
            });
        });
    }
    group.finish();
}

fn make_sessions(n: usize) -> Vec<ViewerSession> {
    let content_id = "content-bench-001";
    (0..n)
        .map(|i| {
            let mut s = ViewerSession::new(
                format!("sess-{i}"),
                Some(format!("user-{}", i % 100)),
                content_id,
                (i as i64) * 1000,
            );
            s.push_event(PlaybackEvent::Play {
                timestamp_ms: (i as i64) * 1000,
            });
            s.push_event(PlaybackEvent::BufferEnd {
                position_ms: 30_000,
                duration_ms: 500,
            });
            s.push_event(PlaybackEvent::Seek {
                from_ms: 60_000,
                to_ms: 300_000,
            });
            s.push_event(PlaybackEvent::Pause {
                timestamp_ms: (i as i64) * 1000 + 600_000,
                position_ms: 600_000,
            });
            s.push_event(PlaybackEvent::End {
                position_ms: 600_000,
                watch_duration_ms: 600_000,
            });
            s
        })
        .collect()
}

fn bench_session_analysis_batch(c: &mut Criterion) {
    let content_ms = 3_600_000u64; // 1 hour
    let mut group = c.benchmark_group("session_analysis");

    for n in [100usize, 500, 1_000] {
        let sessions = make_sessions(n);
        group.bench_with_input(BenchmarkId::new("batch", n), &n, |b, _| {
            b.iter(|| {
                let metrics = analyze_sessions_batch(black_box(&sessions), black_box(content_ms));
                black_box(metrics.len())
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_event_buffer_push_drain,
    bench_session_analysis_batch
);
criterion_main!(benches);
