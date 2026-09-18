//! Real-time audio processing utilities.
//!
//! This module provides lock-free primitives designed for real-time audio
//! processing where bounded latency and wait-free operation are critical.

pub mod ring;

pub use ring::AudioRingBuffer;
