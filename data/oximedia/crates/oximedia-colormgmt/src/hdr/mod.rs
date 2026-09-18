//! HDR (High Dynamic Range) processing and tone mapping.

pub mod tonemapping;

pub use tonemapping::{ToneCurve, ToneMapper, ToneMappingOperator};

use oximedia_core::hdr::TransferCharacteristic;

/// Converts HDR to SDR using tone mapping.
///
/// # Arguments
///
/// * `hdr_rgb` - Input HDR RGB in linear light [0, inf)
/// * `peak_nits` - Peak luminance of HDR content in nits
/// * `target_nits` - Target peak luminance in nits (typically 100 for SDR)
/// * `operator` - Tone mapping operator to use
#[must_use]
pub fn hdr_to_sdr(
    hdr_rgb: [f64; 3],
    peak_nits: f64,
    target_nits: f64,
    operator: ToneMappingOperator,
) -> [f64; 3] {
    let mapper = ToneMapper::new(operator, peak_nits, target_nits);
    mapper.apply(hdr_rgb)
}

/// Converts SDR to HDR using inverse tone mapping.
///
/// # Arguments
///
/// * `sdr_rgb` - Input SDR RGB in linear light [0, 1]
/// * `source_nits` - Peak luminance of SDR content (typically 100 nits)
/// * `target_nits` - Target peak luminance in nits (e.g., 1000 for HDR)
#[must_use]
pub fn sdr_to_hdr(sdr_rgb: [f64; 3], source_nits: f64, target_nits: f64) -> [f64; 3] {
    let scale = target_nits / source_nits;
    [sdr_rgb[0] * scale, sdr_rgb[1] * scale, sdr_rgb[2] * scale]
}

/// Applies PQ (ST.2084) EOTF to convert signal to linear light.
///
/// # Arguments
///
/// * `pq_signal` - PQ encoded signal [0, 1]
///
/// # Returns
///
/// Linear light normalized to [0, 1] where 1.0 = 10000 nits
#[must_use]
pub fn pq_to_linear(pq_signal: [f64; 3]) -> [f64; 3] {
    let transfer = TransferCharacteristic::Pq;
    [
        transfer.eotf(pq_signal[0]),
        transfer.eotf(pq_signal[1]),
        transfer.eotf(pq_signal[2]),
    ]
}

/// Applies PQ (ST.2084) inverse EOTF to convert linear light to signal.
///
/// # Arguments
///
/// * `linear` - Linear light [0, 1] where 1.0 = 10000 nits
///
/// # Returns
///
/// PQ encoded signal [0, 1]
#[must_use]
pub fn linear_to_pq(linear: [f64; 3]) -> [f64; 3] {
    let transfer = TransferCharacteristic::Pq;
    [
        transfer.oetf(linear[0]),
        transfer.oetf(linear[1]),
        transfer.oetf(linear[2]),
    ]
}

/// Applies HLG EOTF to convert signal to linear light.
///
/// # Arguments
///
/// * `hlg_signal` - HLG encoded signal [0, 1]
///
/// # Returns
///
/// Linear light [0, 1]
#[must_use]
pub fn hlg_to_linear(hlg_signal: [f64; 3]) -> [f64; 3] {
    let transfer = TransferCharacteristic::Hlg;
    [
        transfer.eotf(hlg_signal[0]),
        transfer.eotf(hlg_signal[1]),
        transfer.eotf(hlg_signal[2]),
    ]
}

/// Applies HLG inverse EOTF to convert linear light to signal.
///
/// # Arguments
///
/// * `linear` - Linear light [0, 1]
///
/// # Returns
///
/// HLG encoded signal [0, 1]
#[must_use]
pub fn linear_to_hlg(linear: [f64; 3]) -> [f64; 3] {
    let transfer = TransferCharacteristic::Hlg;
    [
        transfer.oetf(linear[0]),
        transfer.oetf(linear[1]),
        transfer.oetf(linear[2]),
    ]
}

/// Converts PQ to HLG.
#[must_use]
pub fn pq_to_hlg(pq: [f64; 3], pq_peak_nits: f64, hlg_peak_nits: f64) -> [f64; 3] {
    // Convert PQ to linear
    let linear = pq_to_linear(pq);

    // Scale from PQ range to HLG range
    let scale = hlg_peak_nits / pq_peak_nits;
    let scaled = [linear[0] * scale, linear[1] * scale, linear[2] * scale];

    // Convert to HLG
    linear_to_hlg(scaled)
}

/// Converts HLG to PQ.
#[must_use]
pub fn hlg_to_pq(hlg: [f64; 3], hlg_peak_nits: f64, pq_peak_nits: f64) -> [f64; 3] {
    // Convert HLG to linear
    let linear = hlg_to_linear(hlg);

    // Scale from HLG range to PQ range
    let scale = pq_peak_nits / hlg_peak_nits;
    let scaled = [linear[0] * scale, linear[1] * scale, linear[2] * scale];

    // Convert to PQ
    linear_to_pq(scaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_roundtrip() {
        let linear = [0.5, 0.3, 0.7];
        let pq = linear_to_pq(linear);
        let linear2 = pq_to_linear(pq);

        assert!((linear2[0] - linear[0]).abs() < 1e-6);
        assert!((linear2[1] - linear[1]).abs() < 1e-6);
        assert!((linear2[2] - linear[2]).abs() < 1e-6);
    }

    #[test]
    fn test_hlg_roundtrip() {
        let linear = [0.5, 0.3, 0.7];
        let hlg = linear_to_hlg(linear);
        let linear2 = hlg_to_linear(hlg);

        assert!((linear2[0] - linear[0]).abs() < 1e-6);
        assert!((linear2[1] - linear[1]).abs() < 1e-6);
        assert!((linear2[2] - linear[2]).abs() < 1e-6);
    }

    #[test]
    fn test_sdr_to_hdr() {
        let sdr = [0.5, 0.5, 0.5];
        let hdr = sdr_to_hdr(sdr, 100.0, 1000.0);

        // Should be 10x brighter
        assert!((hdr[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_hdr_to_sdr() {
        let hdr = [5.0, 3.0, 2.0];
        let sdr = hdr_to_sdr(hdr, 1000.0, 100.0, ToneMappingOperator::Reinhard);

        // Should be tone mapped to [0, 1] range
        assert!(sdr[0] >= 0.0 && sdr[0] <= 1.0);
        assert!(sdr[1] >= 0.0 && sdr[1] <= 1.0);
        assert!(sdr[2] >= 0.0 && sdr[2] <= 1.0);
    }
}
