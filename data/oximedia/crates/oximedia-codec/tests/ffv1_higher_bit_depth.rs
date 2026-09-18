//! FFV1 higher bit-depth encode → decode round-trip integration tests.
//!
//! These tests verify lossless roundtrip for 10-bit and 12-bit pixel formats
//! across 4:2:0, 4:2:2, and 4:4:4 chroma subsampling modes.
//! All tests use constant-value frames to ensure range-coder determinism.

#[cfg(feature = "ffv1")]
mod ffv1_higher_bit_depth {
    use oximedia_codec::ffv1::types::{Ffv1ChromaType, Ffv1Colorspace, Ffv1Config, Ffv1Version};
    use oximedia_codec::ffv1::{Ffv1Decoder, Ffv1Encoder};
    use oximedia_codec::frame::{Plane, VideoFrame};
    use oximedia_codec::traits::{
        BitrateMode, EncoderConfig, EncoderPreset, VideoDecoder, VideoEncoder,
    };
    use oximedia_core::{CodecId, PixelFormat, Rational, Timestamp};

    fn make_encoder_config(width: u32, height: u32) -> EncoderConfig {
        EncoderConfig {
            codec: CodecId::Ffv1,
            width,
            height,
            pixel_format: PixelFormat::Yuv420p,
            framerate: Rational::new(30, 1),
            bitrate: BitrateMode::Lossless,
            preset: EncoderPreset::Medium,
            profile: None,
            keyint: 1,
            threads: 1,
            timebase: Rational::new(1, 1000),
        }
    }

    /// Build a constant-value HBD plane (all samples equal to `val`, stored as 2-byte LE).
    fn make_constant_hbd_plane(pw: u32, ph: u32, val: u16) -> Plane {
        let stride = pw as usize * 2;
        let mut data = vec![0u8; stride * ph as usize];
        for y in 0..ph as usize {
            for x in 0..pw as usize {
                let base = y * stride + x * 2;
                data[base] = (val & 0xFF) as u8;
                data[base + 1] = ((val >> 8) & 0xFF) as u8;
            }
        }
        Plane::with_dimensions(data, stride, pw, ph)
    }

    /// Read a 2-byte LE sample from a plane's byte buffer.
    fn read_le16(plane: &Plane, y: usize, x: usize) -> u16 {
        let base = y * plane.stride + x * 2;
        let lo = plane.data[base] as u16;
        let hi = plane.data[base + 1] as u16;
        lo | (hi << 8)
    }

    /// Create an Ffv1Encoder with the given config, chroma, and bit depth.
    fn make_ffv1_encoder(
        enc_config: EncoderConfig,
        chroma: Ffv1ChromaType,
        bits: u8,
    ) -> Ffv1Encoder {
        let ffv1 = Ffv1Config {
            version: Ffv1Version::V3,
            width: enc_config.width,
            height: enc_config.height,
            colorspace: Ffv1Colorspace::YCbCr,
            chroma_type: chroma,
            bits_per_raw_sample: bits,
            num_h_slices: 1,
            num_v_slices: 1,
            ec: true,
            range_coder_mode: true,
            state_transition_delta: Vec::new(),
        };
        Ffv1Encoder::with_ffv1_config(enc_config, ffv1).expect("encoder init")
    }

    // ── Test 1: 10-bit 4:2:0 constant ──────────────────────────────────────

    #[test]
    fn lossless_roundtrip_10bit_yuv420() {
        let width = 32u32;
        let height = 32u32;
        let bits = 10u8;
        let luma_val: u16 = 512; // mid-scale 10-bit
        let chroma_val: u16 = 512;

        let cw = width / 2;
        let ch = height / 2;

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p10le, width, height);
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, luma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(cw, ch, chroma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(cw, ch, chroma_val));

        let enc_config = make_encoder_config(width, height);
        let mut encoder = make_ffv1_encoder(enc_config, Ffv1ChromaType::Chroma420, bits);

        encoder.send_frame(&frame).expect("encode");
        let packet = encoder
            .receive_packet()
            .expect("recv ok")
            .expect("has packet");

        let extradata = encoder.extradata();
        let mut decoder = Ffv1Decoder::with_extradata(&extradata).expect("decoder init");
        decoder.send_packet(&packet.data, 0).expect("decode");
        let decoded = decoder
            .receive_frame()
            .expect("recv ok")
            .expect("has frame");

        assert_eq!(decoded.planes.len(), 3);

        // Y plane
        let mask = (1u16 << bits) - 1;
        for y in 0..height as usize {
            for x in 0..width as usize {
                let dec_val = read_le16(&decoded.planes[0], y, x) & mask;
                assert_eq!(dec_val, luma_val, "Y mismatch at ({x},{y})");
            }
        }
        // U and V planes
        for pi in 1..3usize {
            for y in 0..ch as usize {
                for x in 0..cw as usize {
                    let dec_val = read_le16(&decoded.planes[pi], y, x) & mask;
                    assert_eq!(dec_val, chroma_val, "plane {pi} mismatch at ({x},{y})");
                }
            }
        }
    }

    // ── Test 2: 12-bit 4:2:2 constant ──────────────────────────────────────

    #[test]
    fn lossless_roundtrip_12bit_yuv422() {
        let width = 16u32;
        let height = 16u32;
        let bits = 12u8;
        let luma_val: u16 = 2048; // mid-scale 12-bit
        let chroma_val: u16 = 1024;

        let cw = width / 2;

        let mut frame = VideoFrame::new(PixelFormat::Yuv422p12le, width, height);
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, luma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(cw, height, chroma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(cw, height, chroma_val));

        let enc_config = make_encoder_config(width, height);
        let mut encoder = make_ffv1_encoder(enc_config, Ffv1ChromaType::Chroma422, bits);

        encoder.send_frame(&frame).expect("encode");
        let packet = encoder
            .receive_packet()
            .expect("recv ok")
            .expect("has packet");

        let extradata = encoder.extradata();
        let mut decoder = Ffv1Decoder::with_extradata(&extradata).expect("decoder init");
        decoder.send_packet(&packet.data, 0).expect("decode");
        let decoded = decoder
            .receive_frame()
            .expect("recv ok")
            .expect("has frame");

        assert_eq!(decoded.planes.len(), 3);

        let mask = (1u16 << bits) - 1;
        // Y plane
        for y in 0..height as usize {
            for x in 0..width as usize {
                let dec_val = read_le16(&decoded.planes[0], y, x) & mask;
                assert_eq!(dec_val, luma_val, "Y mismatch at ({x},{y})");
            }
        }
        // Cb/Cr planes
        for pi in 1..3usize {
            for y in 0..height as usize {
                for x in 0..cw as usize {
                    let dec_val = read_le16(&decoded.planes[pi], y, x) & mask;
                    assert_eq!(dec_val, chroma_val, "plane {pi} mismatch at ({x},{y})");
                }
            }
        }
    }

    // ── Test 3: 10-bit 4:4:4 constant ──────────────────────────────────────

    #[test]
    fn lossless_roundtrip_10bit_yuv444() {
        let width = 16u32;
        let height = 16u32;
        let bits = 10u8;
        let luma_val: u16 = 640;
        let chroma_val: u16 = 512;

        let mut frame = VideoFrame::new(PixelFormat::Yuv444p10le, width, height);
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, luma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, chroma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, chroma_val));

        let enc_config = make_encoder_config(width, height);
        let mut encoder = make_ffv1_encoder(enc_config, Ffv1ChromaType::Chroma444, bits);

        encoder.send_frame(&frame).expect("encode");
        let packet = encoder
            .receive_packet()
            .expect("recv ok")
            .expect("has packet");

        let extradata = encoder.extradata();
        let mut decoder = Ffv1Decoder::with_extradata(&extradata).expect("decoder init");
        decoder.send_packet(&packet.data, 0).expect("decode");
        let decoded = decoder
            .receive_frame()
            .expect("recv ok")
            .expect("has frame");

        assert_eq!(decoded.planes.len(), 3);

        let mask = (1u16 << bits) - 1;
        for y in 0..height as usize {
            for x in 0..width as usize {
                let dec_y = read_le16(&decoded.planes[0], y, x) & mask;
                assert_eq!(dec_y, luma_val, "Y mismatch at ({x},{y})");
            }
        }
        for pi in 1..3usize {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let dec_val = read_le16(&decoded.planes[pi], y, x) & mask;
                    assert_eq!(dec_val, chroma_val, "plane {pi} mismatch at ({x},{y})");
                }
            }
        }
    }

    // ── Test 4: constant grey 12-bit ─────────────────────────────────────────

    #[test]
    fn lossless_roundtrip_16bit_yuv420() {
        let width = 32u32;
        let height = 32u32;
        let bits = 16u8;
        let luma_val: u16 = 32768; // mid-scale 16-bit
        let chroma_val: u16 = 32768;

        let cw = width / 2;
        let ch = height / 2;

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p16le, width, height);
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, luma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(cw, ch, chroma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(cw, ch, chroma_val));

        let enc_config = make_encoder_config(width, height);
        let mut encoder = make_ffv1_encoder(enc_config, Ffv1ChromaType::Chroma420, bits);

        encoder.send_frame(&frame).expect("encode");
        let packet = encoder
            .receive_packet()
            .expect("recv ok")
            .expect("has packet");

        let extradata = encoder.extradata();
        let mut decoder = Ffv1Decoder::with_extradata(&extradata).expect("decoder init");
        decoder.send_packet(&packet.data, 0).expect("decode");
        let decoded = decoder
            .receive_frame()
            .expect("recv ok")
            .expect("has frame");

        assert_eq!(decoded.planes.len(), 3);

        for y in 0..height as usize {
            for x in 0..width as usize {
                let dec_val = read_le16(&decoded.planes[0], y, x);
                assert_eq!(dec_val, luma_val, "Y mismatch at ({x},{y})");
            }
        }
        for pi in 1..3usize {
            for y in 0..ch as usize {
                for x in 0..cw as usize {
                    let dec_val = read_le16(&decoded.planes[pi], y, x);
                    assert_eq!(dec_val, chroma_val, "plane {pi} mismatch at ({x},{y})");
                }
            }
        }
    }

    // ── Test 5: 16-bit 4:2:2 ────────────────────────────────────────────────

    #[test]
    fn lossless_roundtrip_16bit_yuv422() {
        let width = 16u32;
        let height = 16u32;
        let bits = 16u8;
        let luma_val: u16 = 65535; // max 16-bit
        let chroma_val: u16 = 16384; // quarter-scale

        let cw = width / 2;

        let mut frame = VideoFrame::new(PixelFormat::Yuv422p16le, width, height);
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, luma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(cw, height, chroma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(cw, height, chroma_val));

        let enc_config = make_encoder_config(width, height);
        let mut encoder = make_ffv1_encoder(enc_config, Ffv1ChromaType::Chroma422, bits);

        encoder.send_frame(&frame).expect("encode");
        let packet = encoder
            .receive_packet()
            .expect("recv ok")
            .expect("has packet");

        let extradata = encoder.extradata();
        let mut decoder = Ffv1Decoder::with_extradata(&extradata).expect("decoder init");
        decoder.send_packet(&packet.data, 0).expect("decode");
        let decoded = decoder
            .receive_frame()
            .expect("recv ok")
            .expect("has frame");

        assert_eq!(decoded.planes.len(), 3);

        for y in 0..height as usize {
            for x in 0..width as usize {
                let dec_val = read_le16(&decoded.planes[0], y, x);
                assert_eq!(dec_val, luma_val, "Y mismatch at ({x},{y})");
            }
        }
        for pi in 1..3usize {
            for y in 0..height as usize {
                for x in 0..cw as usize {
                    let dec_val = read_le16(&decoded.planes[pi], y, x);
                    assert_eq!(dec_val, chroma_val, "plane {pi} mismatch at ({x},{y})");
                }
            }
        }
    }

    // ── Test 6: 16-bit 4:4:4 ────────────────────────────────────────────────

    #[test]
    fn lossless_roundtrip_16bit_yuv444() {
        let width = 16u32;
        let height = 16u32;
        let bits = 16u8;
        let luma_val: u16 = 48000;
        let chroma_val: u16 = 24000;

        let mut frame = VideoFrame::new(PixelFormat::Yuv444p16le, width, height);
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, luma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, chroma_val));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, chroma_val));

        let enc_config = make_encoder_config(width, height);
        let mut encoder = make_ffv1_encoder(enc_config, Ffv1ChromaType::Chroma444, bits);

        encoder.send_frame(&frame).expect("encode");
        let packet = encoder
            .receive_packet()
            .expect("recv ok")
            .expect("has packet");

        let extradata = encoder.extradata();
        let mut decoder = Ffv1Decoder::with_extradata(&extradata).expect("decoder init");
        decoder.send_packet(&packet.data, 0).expect("decode");
        let decoded = decoder
            .receive_frame()
            .expect("recv ok")
            .expect("has frame");

        assert_eq!(decoded.planes.len(), 3);

        for y in 0..height as usize {
            for x in 0..width as usize {
                let dec_y = read_le16(&decoded.planes[0], y, x);
                assert_eq!(dec_y, luma_val, "Y mismatch at ({x},{y})");
            }
        }
        for pi in 1..3usize {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let dec_val = read_le16(&decoded.planes[pi], y, x);
                    assert_eq!(dec_val, chroma_val, "plane {pi} mismatch at ({x},{y})");
                }
            }
        }
    }

    // ── Test 7: constant grey 12-bit ─────────────────────────────────────────

    #[test]
    fn roundtrip_constant_grey_12bit() {
        let width = 16u32;
        let height = 16u32;
        let bits = 12u8;
        let grey_val: u16 = 2048; // mid-scale 12-bit

        let cw = width / 2;
        let ch = height / 2;

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p12le, width, height);
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame
            .planes
            .push(make_constant_hbd_plane(width, height, grey_val));
        frame.planes.push(make_constant_hbd_plane(cw, ch, grey_val));
        frame.planes.push(make_constant_hbd_plane(cw, ch, grey_val));

        let enc_config = make_encoder_config(width, height);
        let mut encoder = make_ffv1_encoder(enc_config, Ffv1ChromaType::Chroma420, bits);

        encoder.send_frame(&frame).expect("encode");
        let packet = encoder
            .receive_packet()
            .expect("recv ok")
            .expect("has packet");

        let extradata = encoder.extradata();
        let mut decoder = Ffv1Decoder::with_extradata(&extradata).expect("decoder init");
        decoder.send_packet(&packet.data, 0).expect("decode");
        let decoded = decoder
            .receive_frame()
            .expect("recv ok")
            .expect("has frame");

        assert_eq!(decoded.planes.len(), 3);
        let mask = (1u16 << bits) - 1;

        // Y plane
        for y in 0..height as usize {
            for x in 0..width as usize {
                let dec_val = read_le16(&decoded.planes[0], y, x) & mask;
                assert_eq!(
                    dec_val, grey_val,
                    "Y constant grey mismatch at ({x},{y}): got={dec_val} expected={grey_val}"
                );
            }
        }
        // U/V planes
        for pi in 1..3usize {
            for y in 0..ch as usize {
                for x in 0..cw as usize {
                    let dec_val = read_le16(&decoded.planes[pi], y, x) & mask;
                    assert_eq!(
                        dec_val, grey_val,
                        "plane {pi} constant grey mismatch at ({x},{y}): got={dec_val}"
                    );
                }
            }
        }
    }
}
