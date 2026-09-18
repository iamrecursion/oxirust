// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Execution policy of the per-particle gather loops.
//!
//! Every pair sum of the weakly-compressible SPH pipeline (density summation,
//! pressure, viscosity and surface-tension accelerations, XSPH, Shepard/MLS
//! corrections, particle shifting, δ-SPH diffusion, ...) is a *gather*: the value
//! of particle `i` is formed from the neighbours of `i` alone, in the order of its
//! neighbour list, and nothing is shared between particles.  The loops are written
//! as one closure per particle that returns that particle's value
//! ([`collect_per_particle`]); the callers then write the values back by particle
//! index.
//!
//! With the `parallel` feature the particles are split into contiguous index ranges
//! and handed to the rayon pool.  Each value is still computed by one thread with
//! the very operations (and in the very order) of the serial loop and no value
//! depends on another, so there is no reduction whose order could change: the
//! result is bitwise identical to the serial evaluation for every thread count and
//! every schedule.

/// Smallest number of particles one worker task takes: below this the cost of
/// handing out work exceeds the gather itself.
#[cfg(feature = "parallel")]
const MIN_PARTICLES_PER_TASK: usize = 128;

/// How [`collect_per_particle`] evaluates the per-particle closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Execution {
    /// One thread, particle after particle.
    Serial,
    /// The rayon pool (the calling thread joins it), contiguous index ranges.
    #[cfg(feature = "parallel")]
    Parallel,
}

impl Execution {
    /// Policy of the public entry points for a loop over `count` particles.
    ///
    /// Parallel when the `parallel` feature is enabled and `count` can be split
    /// into at least two tasks of `MIN_PARTICLES_PER_TASK` particles; handing a
    /// smaller loop to the pool could not use a second thread and would only add
    /// the cost of waking one.
    #[cfg(feature = "parallel")]
    pub(crate) fn for_count(count: usize) -> Self {
        if count >= 2 * MIN_PARTICLES_PER_TASK {
            Self::Parallel
        } else {
            Self::Serial
        }
    }

    /// Policy of the public entry points: serial, the `parallel` feature is off.
    #[cfg(not(feature = "parallel"))]
    pub(crate) fn for_count(_count: usize) -> Self {
        Self::Serial
    }
}

/// `(0..count).map(f).collect()`, evaluated according to `execution`.
///
/// Element `i` of the result is `f(i)` and `f` is called exactly once per index,
/// whatever the policy.  The bounds are those of the parallel evaluation so that
/// code that compiles without the `parallel` feature also compiles with it.
pub(crate) fn collect_per_particle<T, F>(execution: Execution, count: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    match execution {
        Execution::Serial => (0..count).map(f).collect(),
        #[cfg(feature = "parallel")]
        Execution::Parallel => {
            use rayon::prelude::*;
            (0..count)
                .into_par_iter()
                .with_min_len(MIN_PARTICLES_PER_TASK)
                .map(f)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_collects_in_index_order() {
        let values = collect_per_particle(Execution::Serial, 6, |i| i * i);
        assert_eq!(values, vec![0, 1, 4, 9, 16, 25]);
        assert!(collect_per_particle(Execution::Serial, 0, |i| i).is_empty());
    }

    #[test]
    fn default_policy_is_serial_for_small_loops() {
        assert_eq!(Execution::for_count(0), Execution::Serial);
        assert_eq!(Execution::for_count(255), Execution::Serial);
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn default_policy_is_serial_without_the_feature() {
        assert_eq!(Execution::for_count(1_000_000), Execution::Serial);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn default_policy_is_parallel_for_large_loops_with_the_feature() {
        assert_eq!(Execution::for_count(255), Execution::Serial);
        assert_eq!(Execution::for_count(256), Execution::Parallel);
        assert_eq!(Execution::for_count(1_000_000), Execution::Parallel);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_collects_the_same_values_in_the_same_order() {
        // Sums that are sensitive to the order of the operations: any change of
        // the evaluation order of one element would change its bits.
        let value = |i: usize| -> f64 {
            (0..50).fold(0.0_f64, |acc, k| {
                acc + 1.0 / (1.0 + (i * 31 + k * 7) as f64).sqrt()
            })
        };
        for count in [0, 1, 255, 256, 1000, 4097] {
            let serial = collect_per_particle(Execution::Serial, count, value);
            for threads in [1, 2, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("thread pool");
                let parallel =
                    pool.install(|| collect_per_particle(Execution::Parallel, count, value));
                assert_eq!(serial.len(), parallel.len());
                for (i, (s, p)) in serial.iter().zip(&parallel).enumerate() {
                    assert_eq!(s.to_bits(), p.to_bits(), "count {count}, element {i}");
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_calls_the_closure_once_per_index() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let values = collect_per_particle(Execution::Parallel, 3000, |i| {
            calls.fetch_add(1, Ordering::Relaxed);
            i
        });
        assert_eq!(calls.load(Ordering::Relaxed), 3000);
        assert!(values.iter().enumerate().all(|(i, &v)| i == v));
    }
}
