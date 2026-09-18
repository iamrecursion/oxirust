// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Benchmarks for render farm scheduler.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_renderfarm::job::{JobId, Priority};
use oximedia_renderfarm::scheduler::{Scheduler, SchedulingAlgorithm, Task};
use oximedia_renderfarm::worker::{Worker, WorkerRegistration};
use std::collections::HashMap;
use std::hint::black_box;
use std::net::{IpAddr, Ipv4Addr};

fn create_test_worker(hostname: &str) -> Worker {
    let registration = WorkerRegistration {
        hostname: hostname.to_string(),
        ip_address: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
        port: 8080,
        capabilities: Default::default(),
        location: None,
        tags: HashMap::new(),
    };
    Worker::new(registration)
}

fn benchmark_scheduler_enqueue(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_enqueue");

    for size in &[10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let scheduler = Scheduler::new(SchedulingAlgorithm::Priority);
            b.iter(|| {
                for i in 0..size {
                    let task = Task::new(JobId::new(), i as u32, Priority::Normal);
                    scheduler.enqueue(black_box(task));
                }
            });
        });
    }

    group.finish();
}

fn benchmark_scheduler_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_schedule");

    for algo in &[
        SchedulingAlgorithm::FCFS,
        SchedulingAlgorithm::Priority,
        SchedulingAlgorithm::Deadline,
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{algo:?}")),
            algo,
            |b, algo| {
                let scheduler = Scheduler::new(*algo);
                let worker = create_test_worker("worker01");

                // Pre-fill queue
                for i in 0..100 {
                    let task = Task::new(JobId::new(), i, Priority::Normal);
                    scheduler.enqueue(task);
                }

                b.iter(|| {
                    let _ = scheduler.schedule(black_box(&worker));
                });
            },
        );
    }

    group.finish();
}

fn benchmark_scheduler_assign(c: &mut Criterion) {
    c.bench_function("scheduler_assign", |b| {
        let scheduler = Scheduler::new(SchedulingAlgorithm::Priority);
        let worker = create_test_worker("worker01");

        b.iter(|| {
            let task = Task::new(JobId::new(), 1, Priority::Normal);
            let _ = scheduler.assign(black_box(worker.id), black_box(task));
        });
    });
}

fn benchmark_task_urgency(c: &mut Criterion) {
    c.bench_function("task_urgency_calculation", |b| {
        let task = Task::new(JobId::new(), 1, Priority::High);

        b.iter(|| {
            black_box(task.urgency_score());
        });
    });
}

fn benchmark_fair_share_scheduling(c: &mut Criterion) {
    c.bench_function("fair_share_scheduling", |b| {
        let scheduler = Scheduler::new(SchedulingAlgorithm::FairShare);
        let worker = create_test_worker("worker01");

        // Create tasks for multiple jobs
        for _job_idx in 0..10 {
            let job_id = JobId::new();
            for frame in 0..10 {
                let task = Task::new(job_id, frame, Priority::Normal);
                scheduler.enqueue(task);
            }
        }

        b.iter(|| {
            let _ = scheduler.schedule(black_box(&worker));
        });
    });
}

fn benchmark_backfill_scheduling(c: &mut Criterion) {
    c.bench_function("backfill_scheduling", |b| {
        let scheduler = Scheduler::new(SchedulingAlgorithm::Backfill);
        let worker = create_test_worker("worker01");

        // Create tasks with varying estimated times
        for i in 0..50 {
            let mut task = Task::new(JobId::new(), i, Priority::Normal);
            task.estimated_time = (f64::from(i) * 10.0).min(3600.0);
            scheduler.enqueue(task);
        }

        b.iter(|| {
            let _ = scheduler.schedule(black_box(&worker));
        });
    });
}

criterion_group!(
    benches,
    benchmark_scheduler_enqueue,
    benchmark_scheduler_schedule,
    benchmark_scheduler_assign,
    benchmark_task_urgency,
    benchmark_fair_share_scheduling,
    benchmark_backfill_scheduling,
);

criterion_main!(benches);
