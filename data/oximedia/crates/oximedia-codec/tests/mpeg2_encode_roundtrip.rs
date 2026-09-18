//! Integration tests for the MPEG-2 I-frame encoder (`mpeg2` feature).
//!
//! These tests exercise the full forward path (`Mpeg2Encoder`) and verify that
//! the **existing** [`Mpeg2Decoder`] reconstructs the image within the bounded
//! quantiser / IDCT error, plus targeted round-trips of the individual
//! building blocks (FDCT/IDCT, forward/inverse quant, VLC encode/decode).

#![cfg(feature = "mpeg2")]

use oximedia_codec::mpeg2::bitreader::BitReader;
use oximedia_codec::mpeg2::bitwriter::BitWriter;
use oximedia_codec::mpeg2::dequant::{dequantize_intra, DEFAULT_INTRA_MATRIX};
use oximedia_codec::mpeg2::entropy::decode_ac;
use oximedia_codec::mpeg2::fdct::fdct_8x8;
use oximedia_codec::mpeg2::idct::idct_8x8;
use oximedia_codec::mpeg2::quantize_fwd::quantize_intra;
use oximedia_codec::mpeg2::vlc_encode::{encode_ac_run_level, encode_eob};
use oximedia_codec::mpeg2::vlc_tables::{AC_TABLE_B14, EOB_RUN, ESCAPE_RUN};
use oximedia_codec::mpeg2::zigzag::SCAN_PROGRESSIVE;
use oximedia_codec::mpeg2::{Mpeg2Decoder, Mpeg2Encoder, Mpeg2EncoderConfig};

/// Mean absolute error between two equal-length byte planes.
fn mean_abs_error(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let total: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (i32::from(x) - i32::from(y)).unsigned_abs() as u64)
        .sum();
    total as f64 / a.len() as f64
}

/// Max absolute error between two equal-length byte planes.
fn max_abs_error(a: &[u8], b: &[u8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (i32::from(x) - i32::from(y)).abs())
        .max()
        .unwrap_or(0)
}

#[test]
fn roundtrip_constant_grey_16x16() {
    // A flat mid-grey frame is the near-exact case: zero AC, zero DC
    // differential, so reconstruction is within a couple of LSB.
    let w = 16u32;
    let h = 16u32;
    let cfg = Mpeg2EncoderConfig::new(w, h, 4);
    let enc = Mpeg2Encoder::new(cfg).expect("encoder");

    let y = vec![128u8; (w * h) as usize];
    let c = vec![128u8; ((w / 2) * (h / 2)) as usize];

    let stream = enc.encode_planes(&y, &c, &c).expect("encode");
    let frame = Mpeg2Decoder::new().decode(&stream).expect("decode");

    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    assert_eq!(frame.y.len(), (w * h) as usize);

    assert!(
        max_abs_error(&frame.y, &y) <= 2,
        "grey luma max error {} > 2",
        max_abs_error(&frame.y, &y)
    );
    assert!(max_abs_error(&frame.cb, &c) <= 2, "grey cb error too large");
    assert!(max_abs_error(&frame.cr, &c) <= 2, "grey cr error too large");
}

#[test]
fn roundtrip_grey_multiple_values() {
    // Several flat values across the legal 8-bit range, each near-exact.
    for value in [0u8, 16, 64, 128, 200, 235] {
        let w = 16u32;
        let h = 16u32;
        let enc = Mpeg2Encoder::new(Mpeg2EncoderConfig::new(w, h, 2)).expect("enc");
        let y = vec![value; (w * h) as usize];
        let c = vec![value; ((w / 2) * (h / 2)) as usize];
        let stream = enc.encode_planes(&y, &c, &c).expect("encode");
        let frame = Mpeg2Decoder::new().decode(&stream).expect("decode");
        assert!(
            max_abs_error(&frame.y, &y) <= 2,
            "value {value}: max luma error {}",
            max_abs_error(&frame.y, &y)
        );
    }
}

#[test]
fn roundtrip_gradient_64x48() {
    // A smooth horizontal+vertical gradient. With a low qscale the mean abs
    // error must be small (the FDCT concentrates energy in low frequencies the
    // quantiser preserves well).
    let w = 64u32;
    let h = 48u32;
    let cfg = Mpeg2EncoderConfig::new(w, h, 2); // q_scale = 4 (linear)
    let enc = Mpeg2Encoder::new(cfg).expect("encoder");

    let cw = (w / 2) as usize;
    let ch = (h / 2) as usize;

    // Luma: smooth ramp 16..=235.
    let mut y = vec![0u8; (w * h) as usize];
    for row in 0..h as usize {
        for col in 0..w as usize {
            let v = 16 + ((col * 180) / w as usize + (row * 40) / h as usize);
            y[row * w as usize + col] = v.min(235) as u8;
        }
    }
    // Chroma: gentle ramps around 128.
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = (110 + (col * 30) / cw) as u8;
            cr[row * cw + col] = (140 - (row * 20) / ch) as u8;
        }
    }

    let stream = enc.encode_planes(&y, &cb, &cr).expect("encode");
    let frame = Mpeg2Decoder::new().decode(&stream).expect("decode");

    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);

    let y_mae = mean_abs_error(&frame.y, &y);
    let cb_mae = mean_abs_error(&frame.cb, &cb);
    let cr_mae = mean_abs_error(&frame.cr, &cr);

    assert!(y_mae < 3.0, "luma mean abs error {y_mae} too large");
    assert!(cb_mae < 3.0, "cb mean abs error {cb_mae} too large");
    assert!(cr_mae < 3.0, "cr mean abs error {cr_mae} too large");
}

#[test]
fn roundtrip_non_multiple_of_16_dimensions() {
    // 20×20 → padded to 32×32 (2×2 macroblocks). The decoder crops back to
    // 20×20; the visible region must round-trip within tolerance.
    let w = 20u32;
    let h = 20u32;
    let enc = Mpeg2Encoder::new(Mpeg2EncoderConfig::new(w, h, 2)).expect("enc");

    let cw = (w as usize).div_ceil(2);
    let ch = (h as usize).div_ceil(2);
    let y = vec![100u8; (w * h) as usize];
    let cb = vec![120u8; cw * ch];
    let cr = vec![140u8; cw * ch];

    let stream = enc.encode_planes(&y, &cb, &cr).expect("encode");
    let frame = Mpeg2Decoder::new().decode(&stream).expect("decode");
    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    assert_eq!(frame.y.len(), (w * h) as usize);
    assert!(max_abs_error(&frame.y, &y) <= 3, "luma error too large");
}

#[test]
fn fdct_idct_identity_dc() {
    // FDCT then IDCT of a DC-only spatial block reconstructs the flat value.
    for value in [0i32, 8, 64, 128, 200, 255] {
        let block = [value; 64];
        let freq = fdct_8x8(&block);
        // DC coefficient ≈ 8 · value.
        assert!(
            (freq[0] - 8 * value).abs() <= 1,
            "FDCT DC for {value}: {}",
            freq[0]
        );
        let spatial = idct_8x8(&freq);
        for (i, &v) in spatial.iter().enumerate() {
            assert!(
                (v - value).abs() <= 2,
                "FDCT→IDCT [{i}] for value {value}: got {v}"
            );
        }
    }
}

#[test]
fn forward_quant_inverse_quant_roundtrip() {
    // Quantise then dequantise; each AC coefficient must come back within one
    // quant step (W·q_scale/16), DC exact for multiples of intra_dc_mult.
    let q_scale = 8; // q_scale_type linear, code 4
    let coeffs: [i32; 64] = std::array::from_fn(|i| {
        if i == 0 {
            1024 // multiple of 8 → DC exact at precision 0
        } else {
            (((i * 37) % 23) as i32 - 11) * 8
        }
    });
    let qf = quantize_intra(&coeffs, &DEFAULT_INTRA_MATRIX, 0, q_scale);
    let recon = dequantize_intra(&qf, &DEFAULT_INTRA_MATRIX, 0, q_scale);

    assert_eq!(recon[0], 1024, "DC must be exact");
    for i in 1..64 {
        let step = (i32::from(DEFAULT_INTRA_MATRIX[i]) * q_scale) / 16 + 1;
        // F[63] may additionally be toggled ±1 by mismatch control (§7.4.4).
        let tol = if i == 63 { step + 1 } else { step };
        assert!(
            (recon[i] - coeffs[i]).abs() <= tol,
            "coeff[{i}]={} recon={} step={step}",
            coeffs[i],
            recon[i]
        );
    }
}

#[test]
fn vlc_encode_decode_each_table_entry() {
    // Every (run, level) entry in B-14 must encode and decode back to itself,
    // for both signs. Uses the public bit writer + decoder AC path.
    for &(_, _, run, level) in AC_TABLE_B14 {
        if run == EOB_RUN || run == ESCAPE_RUN {
            continue;
        }
        for signed in [i32::from(level), -i32::from(level)] {
            let mut w = BitWriter::new();
            encode_ac_run_level(&mut w, AC_TABLE_B14, run, signed).expect("encode ac");
            encode_eob(&mut w, AC_TABLE_B14).expect("encode eob");
            w.write_bits(0, 24); // padding for the reader
            let bytes = w.into_bytes();

            let mut r = BitReader::new(&bytes);
            let mut block = [0i32; 64];
            decode_ac(&mut r, &mut block, false, false).expect("decode ac");

            let scan_index = run as usize + 1;
            let raster = SCAN_PROGRESSIVE[scan_index];
            assert_eq!(
                block[raster], signed,
                "(run={run}, level={signed}) decoded wrong at raster {raster}"
            );
            // No stray coefficients.
            for (i, &v) in block.iter().enumerate() {
                if i != raster {
                    assert_eq!(v, 0, "stray coeff at raster {i}");
                }
            }
        }
    }
}

#[test]
fn encoded_stream_parses_with_decoder() {
    // The encoder's headers must parse cleanly and yield a frame of exactly the
    // configured dimensions.
    let w = 48u32;
    let h = 32u32;
    let enc = Mpeg2Encoder::new(Mpeg2EncoderConfig::new(w, h, 6)).expect("enc");
    let cw = (w / 2) as usize;
    let ch = (h / 2) as usize;
    let y = vec![90u8; (w * h) as usize];
    let cb = vec![128u8; cw * ch];
    let cr = vec![128u8; cw * ch];

    let stream = enc.encode_planes(&y, &cb, &cr).expect("encode");

    // Sanity on framing: starts with sequence header, ends with sequence end.
    assert_eq!(&stream[0..4], &[0x00, 0x00, 0x01, 0xB3]);
    assert_eq!(&stream[stream.len() - 4..], &[0x00, 0x00, 0x01, 0xB7]);

    let frame = Mpeg2Decoder::new().decode(&stream).expect("decode");
    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    assert_eq!(frame.y.len(), (w * h) as usize);
    assert_eq!(frame.cb.len(), cw * ch);
    assert_eq!(frame.cr.len(), cw * ch);
}

#[test]
fn encoder_video_frame_trait_roundtrip() {
    // Exercise the VideoEncoder trait surface (send_frame / receive_packet).
    use oximedia_codec::frame::{Plane, VideoFrame};
    use oximedia_codec::traits::VideoEncoder;
    use oximedia_core::PixelFormat;

    let w = 32u32;
    let h = 16u32;
    let mut enc = Mpeg2Encoder::new(Mpeg2EncoderConfig::new(w, h, 4)).expect("enc");

    let cw = (w / 2) as usize;
    let ch = (h / 2) as usize;
    let y = vec![128u8; (w * h) as usize];
    let c = vec![128u8; cw * ch];

    let mut frame = VideoFrame::new(PixelFormat::Yuv420p, w, h);
    frame.planes = vec![
        Plane::with_dimensions(y.clone(), w as usize, w, h),
        Plane::with_dimensions(c.clone(), cw, w / 2, h),
        Plane::with_dimensions(c.clone(), cw, w / 2, h),
    ];

    enc.send_frame(&frame).expect("send_frame");
    let pkt = enc
        .receive_packet()
        .expect("receive_packet")
        .expect("packet present");
    assert!(pkt.keyframe);

    let decoded = Mpeg2Decoder::new().decode(&pkt.data).expect("decode");
    assert_eq!(decoded.width, w);
    assert_eq!(decoded.height, h);
    assert!(max_abs_error(&decoded.y, &y) <= 2);

    // Second receive yields nothing.
    assert!(enc.receive_packet().expect("ok").is_none());
}

// ── Wave 10: 4:2:2 and 4:4:4 chroma format round-trips ─────────────────────

/// Compute PSNR (dB) of one byte plane vs. its reference.
fn psnr_u8(reconstructed: &[u8], original: &[u8]) -> f64 {
    assert_eq!(reconstructed.len(), original.len());
    if reconstructed.is_empty() {
        return f64::INFINITY;
    }
    let mse: f64 = reconstructed
        .iter()
        .zip(original.iter())
        .map(|(&r, &o)| {
            let d = f64::from(i32::from(r) - i32::from(o));
            d * d
        })
        .sum::<f64>()
        / reconstructed.len() as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

#[test]
fn roundtrip_constant_grey_16x16_yuv422() {
    // 4:2:2 chroma: per-MB chroma plane is 8×16, so two stacked 8×8 chroma
    // blocks. Flat input means zero AC, zero DC differential → near-exact.
    let w = 16u32;
    let h = 16u32;
    let cfg = Mpeg2EncoderConfig::yuv422p(w, h, 4);
    let enc = Mpeg2Encoder::new(cfg).expect("encoder");

    let y = vec![128u8; (w * h) as usize];
    let cw = (w / 2) as usize;
    let ch = h as usize;
    let c = vec![128u8; cw * ch];

    let stream = enc.encode_planes(&y, &c, &c).expect("encode");
    let frame = Mpeg2Decoder::new().decode(&stream).expect("decode");

    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    assert_eq!(frame.chroma_format, 2);
    assert_eq!(frame.y.len(), (w * h) as usize);
    assert_eq!(frame.cb.len(), cw * ch);
    assert_eq!(frame.cr.len(), cw * ch);

    assert!(
        max_abs_error(&frame.y, &y) <= 2,
        "y22 grey max error {} > 2",
        max_abs_error(&frame.y, &y)
    );
    assert!(
        max_abs_error(&frame.cb, &c) <= 2,
        "cb22 grey max error {} > 2",
        max_abs_error(&frame.cb, &c)
    );
    assert!(
        max_abs_error(&frame.cr, &c) <= 2,
        "cr22 grey max error {} > 2",
        max_abs_error(&frame.cr, &c)
    );
}

#[test]
fn roundtrip_constant_grey_16x16_yuv444() {
    // 4:4:4 chroma: per-MB chroma is 16×16 — four 8×8 chroma blocks.
    let w = 16u32;
    let h = 16u32;
    let cfg = Mpeg2EncoderConfig::yuv444p(w, h, 4);
    let enc = Mpeg2Encoder::new(cfg).expect("encoder");

    let y = vec![128u8; (w * h) as usize];
    let c = vec![128u8; (w * h) as usize];

    let stream = enc.encode_planes(&y, &c, &c).expect("encode");
    let frame = Mpeg2Decoder::new().decode(&stream).expect("decode");

    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    assert_eq!(frame.chroma_format, 3);
    assert_eq!(frame.y.len(), (w * h) as usize);
    assert_eq!(frame.cb.len(), (w * h) as usize);
    assert_eq!(frame.cr.len(), (w * h) as usize);

    assert!(
        max_abs_error(&frame.y, &y) <= 2,
        "y44 grey max error {} > 2",
        max_abs_error(&frame.y, &y)
    );
    assert!(
        max_abs_error(&frame.cb, &c) <= 2,
        "cb44 grey max error {} > 2",
        max_abs_error(&frame.cb, &c)
    );
    assert!(
        max_abs_error(&frame.cr, &c) <= 2,
        "cr44 grey max error {} > 2",
        max_abs_error(&frame.cr, &c)
    );
}

#[test]
fn roundtrip_gradient_yuv422() {
    // Vertical + horizontal gradient in Yuv422p; PSNR ≥ 40 dB at qscale = 2.
    let w = 32u32;
    let h = 32u32;
    let cfg = Mpeg2EncoderConfig::yuv422p(w, h, 2);
    let enc = Mpeg2Encoder::new(cfg).expect("encoder");

    let cw = (w / 2) as usize;
    let ch = h as usize;

    let mut y = vec![0u8; (w * h) as usize];
    for row in 0..h as usize {
        for col in 0..w as usize {
            // Mostly horizontal ramp 16..=235.
            let v = 16 + ((col * 180) / w as usize + (row * 20) / h as usize);
            y[row * w as usize + col] = v.min(235) as u8;
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = (110 + (col * 30) / cw) as u8;
            cr[row * cw + col] = (140 - (row * 20) / ch) as u8;
        }
    }

    let stream = enc.encode_planes(&y, &cb, &cr).expect("encode");
    let frame = Mpeg2Decoder::new().decode(&stream).expect("decode");
    assert_eq!(frame.chroma_format, 2);

    let y_psnr = psnr_u8(&frame.y, &y);
    let cb_psnr = psnr_u8(&frame.cb, &cb);
    let cr_psnr = psnr_u8(&frame.cr, &cr);
    assert!(y_psnr >= 40.0, "yuv422 luma psnr {y_psnr} dB < 40");
    assert!(cb_psnr >= 40.0, "yuv422 cb psnr {cb_psnr} dB < 40");
    assert!(cr_psnr >= 40.0, "yuv422 cr psnr {cr_psnr} dB < 40");
}

#[test]
fn roundtrip_gradient_yuv444() {
    // Vertical + horizontal gradient in Yuv444p; PSNR ≥ 40 dB at qscale = 2.
    let w = 32u32;
    let h = 32u32;
    let cfg = Mpeg2EncoderConfig::yuv444p(w, h, 2);
    let enc = Mpeg2Encoder::new(cfg).expect("encoder");

    let cw = w as usize;
    let ch = h as usize;

    let mut y = vec![0u8; (w * h) as usize];
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..h as usize {
        for col in 0..w as usize {
            let v = 16 + ((col * 180) / w as usize + (row * 20) / h as usize);
            y[row * w as usize + col] = v.min(235) as u8;
            cb[row * cw + col] = (110 + (col * 30) / cw) as u8;
            cr[row * cw + col] = (140 - (row * 20) / ch) as u8;
        }
    }

    let stream = enc.encode_planes(&y, &cb, &cr).expect("encode");
    let frame = Mpeg2Decoder::new().decode(&stream).expect("decode");
    assert_eq!(frame.chroma_format, 3);

    let y_psnr = psnr_u8(&frame.y, &y);
    let cb_psnr = psnr_u8(&frame.cb, &cb);
    let cr_psnr = psnr_u8(&frame.cr, &cr);
    assert!(y_psnr >= 40.0, "yuv444 luma psnr {y_psnr} dB < 40");
    assert!(cb_psnr >= 40.0, "yuv444 cb psnr {cb_psnr} dB < 40");
    assert!(cr_psnr >= 40.0, "yuv444 cr psnr {cr_psnr} dB < 40");
}

#[test]
fn sequence_extension_writes_chroma_format() {
    // For each of chroma_format in {1, 2, 3}: encode a tiny stream, then
    // re-parse the sequence_extension via the public Mpeg2Decoder pipeline and
    // confirm `frame.chroma_format` matches.
    for cf in 1u8..=3u8 {
        let w = 16u32;
        let h = 16u32;
        let mut cfg = Mpeg2EncoderConfig::new(w, h, 4);
        cfg.chroma_format = cf;
        let enc = Mpeg2Encoder::new(cfg).expect("encoder");

        let y_len = (w * h) as usize;
        let (cw, ch) = match cf {
            1 => ((w / 2) as usize, (h / 2) as usize),
            2 => ((w / 2) as usize, h as usize),
            _ => (w as usize, h as usize),
        };
        let y = vec![128u8; y_len];
        let c = vec![128u8; cw * ch];

        let stream = enc.encode_planes(&y, &c, &c).expect("encode");
        let frame = Mpeg2Decoder::new().decode(&stream).expect("decode");
        assert_eq!(
            frame.chroma_format, cf,
            "expected chroma_format {cf}, got {}",
            frame.chroma_format
        );
    }
}

#[test]
fn decoder_accepts_4_2_2_header() {
    // The encoder writes chroma_format=2 in sequence_extension; the decoder
    // must parse it without rejecting.
    let w = 32u32;
    let h = 16u32;
    let cfg = Mpeg2EncoderConfig::yuv422p(w, h, 4);
    let enc = Mpeg2Encoder::new(cfg).expect("encoder");

    let y = vec![128u8; (w * h) as usize];
    let cw = (w / 2) as usize;
    let ch = h as usize;
    let c = vec![128u8; cw * ch];

    let stream = enc.encode_planes(&y, &c, &c).expect("encode");

    // Stream framing sanity.
    assert_eq!(&stream[0..4], &[0x00, 0x00, 0x01, 0xB3]);
    assert_eq!(&stream[stream.len() - 4..], &[0x00, 0x00, 0x01, 0xB7]);

    let frame = Mpeg2Decoder::new()
        .decode(&stream)
        .expect("decoder accepts 4:2:2 header");
    assert_eq!(frame.chroma_format, 2);
    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    assert_eq!(frame.cb.len(), cw * ch);
    assert_eq!(frame.cr.len(), cw * ch);
}
