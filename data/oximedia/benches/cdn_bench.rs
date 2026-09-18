use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_cdn::{
    edge_manager::{EdgeManager, EdgeNode},
    geo_routing::{haversine_km, EdgeNodeGeo, GeoLocation, GeoRouter},
    origin_failover::{OriginPool, OriginServer, OriginStrategy},
};
use std::hint::black_box;

fn bench_haversine_distance(c: &mut Criterion) {
    // 1 000 lat/lon point pairs → 1 000 computations per iteration
    let points: Vec<(f64, f64)> = (0..1_000)
        .map(|i| {
            let lat = -80.0 + (i as f64 * 160.0 / 1_000.0);
            let lon = -180.0 + (i as f64 * 360.0 / 1_000.0);
            (lat, lon)
        })
        .collect();

    c.bench_function("haversine_1k_pairs", |b| {
        b.iter(|| {
            let mut total = 0.0f64;
            for (i, &(lat1, lon1)) in points.iter().enumerate() {
                let (lat2, lon2) = points[(i + 1) % points.len()];
                total += haversine_km(
                    black_box(lat1),
                    black_box(lon1),
                    black_box(lat2),
                    black_box(lon2),
                );
            }
            black_box(total);
        });
    });
}

fn bench_geo_router_assign_edge(c: &mut Criterion) {
    let mut group = c.benchmark_group("geo_router_assign_edge");

    for n_nodes in [10usize, 50, 200] {
        let mut router = GeoRouter::new();
        for i in 0..n_nodes {
            let lat = -80.0 + (i as f64 * 160.0 / n_nodes as f64);
            let lon = -180.0 + (i as f64 * 360.0 / n_nodes as f64);
            let geo = EdgeNodeGeo::new(format!("node-{i}"), GeoLocation::new(lat, lon, "US"));
            router.add_node(geo);
        }
        let client = GeoLocation::new(35.6762, 139.6503, "JP");

        group.bench_with_input(BenchmarkId::new("nodes", n_nodes), &n_nodes, |b, _| {
            b.iter(|| {
                let result = router.assign_edge(black_box(&client));
                black_box(result.map(|id| id.0.len()).unwrap_or(0))
            });
        });
    }
    group.finish();
}

fn bench_origin_pool_select(c: &mut Criterion) {
    let mut pool = OriginPool::new(OriginStrategy::WeightedRoundRobin);
    for i in 0..10 {
        let server = OriginServer::new(
            &format!("origin-{i}"),
            &format!("https://origin{i}.example.com"),
            1,
            0,
        );
        pool.add_server_owned(server);
    }

    c.bench_function("origin_pool_select_wrr_10k", |b| {
        b.iter(|| {
            let mut last_len = 0usize;
            for _ in 0u64..10_000 {
                if let Some(s) = pool.select() {
                    last_len = s.url.len();
                }
            }
            black_box(last_len);
        });
    });
}

fn bench_edge_manager_best_node(c: &mut Criterion) {
    let mut mgr = EdgeManager::new();
    for i in 0..50usize {
        let node = EdgeNode::new(
            &format!("edge-{i}"),
            "cloudflare",
            &format!("us-east-{}", i % 4 + 1),
            &format!("edge{i}.cf.example.com"),
        );
        mgr.add_node(node);
    }

    c.bench_function("edge_manager_best_node_10k", |b| {
        b.iter(|| {
            let mut count = 0usize;
            for i in 0u64..10_000 {
                let region = format!("us-east-{}", (i % 4) + 1);
                if let Some(node) = mgr.best_node_for(black_box(&region), black_box(&[])) {
                    count += node.id.len();
                }
            }
            black_box(count);
        });
    });
}

criterion_group!(
    benches,
    bench_haversine_distance,
    bench_geo_router_assign_edge,
    bench_origin_pool_select,
    bench_edge_manager_best_node,
);
criterion_main!(benches);
