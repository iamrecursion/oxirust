//! Quick and easy random generation for rapid prototyping
//!
//! This module provides the simplest possible interface for common random generation tasks.
//! Perfect for quick scripts, prototypes, and when you just need "some random numbers fast".
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::random::quick::*;
//!
//! // Generate random numbers with minimal setup
//! let x = random_f64();              // Random number [0, 1)
//! let n = random_int(1, 100);        // Random integer [1, 100]
//! let b = random_bool();             // Random boolean
//!
//! // Quick arrays
//! let data = random_vector(1000);    // 1000 random f64s
//! let matrix = random_matrix(10, 10); // 10x10 random matrix
//!
//! // Quick sampling
//! let items = vec!["A", "B", "C", "D"];
//! let choice = pick_one(&items);      // Pick random item
//! let sample = pick_many(&items, 2);  // Pick 2 random items
//! ```

use crate::random::{random_normal_array, random_uniform_array, thread_rng};
use ::ndarray::{Array2, Ix2};
use rand_distr::{Normal, Uniform};

/// Generate a random f64 in [0, 1)
pub fn random_f64() -> f64 {
    crate::random::convenience::uniform()
}

/// Generate a random f32 in [0, 1)
pub fn random_f32() -> f32 {
    random_f64() as f32
}

/// Generate a random integer in the given range (inclusive)
pub fn random_int(min: i64, max: i64) -> i64 {
    crate::random::convenience::integer(min, max)
}

/// Generate a random usize in the given range (inclusive)
pub fn random_usize(min: usize, max: usize) -> usize {
    random_int(min as i64, max as i64) as usize
}

/// Generate a random boolean
pub fn random_bool() -> bool {
    crate::random::convenience::boolean()
}

/// Generate a random boolean with given probability of true
pub fn random_bool_with_prob(prob: f64) -> bool {
    random_f64() < prob
}

/// Generate a vector of random f64s in [0, 1)
pub fn random_vector(size: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    (0..size)
        .map(|_| rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed")))
        .collect()
}

/// Generate a vector of random integers in the given range
pub fn random_int_vector(size: usize, min: i64, max: i64) -> Vec<i64> {
    (0..size).map(|_| random_int(min, max)).collect()
}

/// Generate a 2D matrix of random f64s in [0, 1)
pub fn random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = thread_rng();
    random_uniform_array(Ix2(rows, cols), &mut rng)
}

/// Generate a 2D matrix of random normal values
pub fn random_normal_matrix(rows: usize, cols: usize, mean: f64, std: f64) -> Array2<f64> {
    let mut rng = thread_rng();
    random_normal_array(Ix2(rows, cols), mean, std, &mut rng)
}

/// Pick a random element from a slice
pub fn pick_one<T>(items: &[T]) -> Option<&T> {
    if items.is_empty() {
        None
    } else {
        let index = random_usize(0, items.len() - 1);
        Some(&items[index])
    }
}

/// Pick multiple random elements from a slice (with replacement)
pub fn pick_many<T: Clone>(items: &[T], count: usize) -> Vec<T> {
    (0..count)
        .filter_map(|_| pick_one(items))
        .cloned()
        .collect()
}

/// Shuffle a vector in place
pub fn shuffle<T>(items: &mut Vec<T>) {
    use crate::random::seq::SliceRandom;
    let mut rng = crate::random::thread_rng();
    items.shuffle(&mut rng);
}

/// Create a shuffled copy of a vector
pub fn shuffled<T: Clone>(items: &[T]) -> Vec<T> {
    let mut result = items.to_vec();
    shuffle(&mut result);
    result
}

/// Generate random text of given length (alphanumeric)
pub fn random_text(length: usize) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    (0..length)
        .map(|_| {
            let idx = random_usize(0, CHARS.len() - 1);
            CHARS[idx] as char
        })
        .collect()
}

/// Generate a random hex string of given byte length
pub fn random_hex(byte_length: usize) -> String {
    (0..byte_length)
        .map(|_| format!("{:02x}", random_usize(0, 255)))
        .collect()
}

/// Coin flip - returns true or false with 50% probability
pub fn coin_flip() -> bool {
    random_bool()
}

/// Dice roll - returns 1-6
pub fn dice_roll() -> i64 {
    random_int(1, 6)
}

/// Roll multiple dice and return the sum
pub fn dice_roll_sum(count: usize) -> i64 {
    (0..count).map(|_| dice_roll()).sum()
}

/// Generate a random percentage (0.0 to 100.0)
pub fn random_percentage() -> f64 {
    random_f64() * 100.0
}

/// Random choice with weights (simplified interface)
pub fn weighted_choice<T: Clone>(items: &[T], weights: &[f64]) -> Option<T> {
    if items.len() != weights.len() || items.is_empty() {
        return None;
    }

    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }

    let mut cumulative = 0.0;
    let target = random_f64() * total;

    for (item, &weight) in items.iter().zip(weights.iter()) {
        cumulative += weight;
        if target <= cumulative {
            return Some(item.clone());
        }
    }

    // Fallback to last item
    items.last().cloned()
}
