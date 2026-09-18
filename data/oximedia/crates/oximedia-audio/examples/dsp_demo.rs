//! DSP demonstration example.
//!
//! This example demonstrates the usage of the DSP modules in oximedia-audio.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use oximedia_audio::dsp::{
    Compressor, CompressorConfig, EqBand, Equalizer, EqualizerConfig, Reverb, ReverbConfig,
};

fn main() {
    println!("OxiMedia Audio DSP Demo");
    println!("=======================\n");

    demo_equalizer();
    demo_compressor();
    demo_reverb();
}

fn demo_equalizer() {
    println!("1. Parametric Equalizer");
    println!("-----------------------");

    let config = EqualizerConfig::three_band(3.0, 0.0, -2.0);
    let mut eq = Equalizer::new(config, 48000.0, 2);

    let mut samples = vec![vec![0.5_f64; 1024]; 2];

    eq.process_planar(&mut samples);

    println!("Created 3-band EQ with +3dB bass, 0dB mid, -2dB treble");
    println!("Processed {} samples per channel\n", samples[0].len());
}

fn demo_compressor() {
    println!("2. Dynamics Compressor");
    println!("----------------------");

    let config = CompressorConfig::new(-20.0, 4.0)
        .with_timing(10.0, 100.0)
        .with_soft_knee(6.0)
        .with_auto_makeup();

    let mut compressor = Compressor::new(config, 48000.0, 2);

    let mut samples = vec![vec![0.8_f64; 1024]; 2];

    compressor.process_planar(&mut samples);

    println!("Created compressor with -20dB threshold, 4:1 ratio");
    println!("Attack: 10ms, Release: 100ms, Soft knee: 6dB");
    println!("Gain reduction: {:.2} dB\n", compressor.gain_reduction_db());
}

fn demo_reverb() {
    println!("3. Schroeder Reverb");
    println!("-------------------");

    let config = ReverbConfig::large_hall();
    let mut reverb = Reverb::new(config, 48000.0);

    let mut left = vec![0.6_f64; 1024];
    let mut right = vec![0.6_f64; 1024];

    reverb.process_stereo_planar(&mut left, &mut right);

    println!("Created large hall reverb preset");
    println!("Processed {} sample frames\n", left.len());
}

fn demo_custom_eq() {
    let config = EqualizerConfig::new()
        .add_band(EqBand::high_pass(80.0, 0.707))
        .add_band(EqBand::peaking(250.0, 3.0, 1.0))
        .add_band(EqBand::peaking(1000.0, -2.0, 1.5))
        .add_band(EqBand::low_pass(16000.0, 0.707));

    let mut eq = Equalizer::new(config, 44100.0, 2);

    let mut samples = vec![vec![0.5_f64; 2048]; 2];
    eq.process_planar(&mut samples);

    println!("Custom 4-band EQ configuration processed");
}

fn demo_graphic_eq() {
    let config =
        EqualizerConfig::graphic_10_band(0.0, 1.0, 2.0, 3.0, 2.0, 0.0, -1.0, -2.0, -1.0, 0.0);

    let mut eq = Equalizer::new(config, 48000.0, 2);

    let mut samples = vec![vec![0.5_f64; 4096]; 2];
    eq.process_planar(&mut samples);

    println!("10-band graphic EQ processed");
}

fn demo_limiter() {
    let config = CompressorConfig::new(-6.0, 20.0)
        .with_timing(0.1, 50.0)
        .with_hard_knee();

    let mut limiter = Compressor::new(config, 48000.0, 2);

    let mut samples = vec![vec![0.9_f64; 1024]; 2];
    limiter.process_planar(&mut samples);

    println!("Limiter (high ratio compressor) processed");
}
