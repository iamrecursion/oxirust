//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use crate::arg_parser::FfmpegArgs;
    use crate::compat_ext::functions::{
        generate_hints, parse_filter_chain, parse_time_str, FfmpegArgsExt,
    };
    use crate::compat_ext::types::{
        ArgumentBuilder, ContainerMapper, FfmpegCompatDiagnostics, FfmpegDiagnostics,
        FilterGraphParser, FilterGraphValidator, MapStreamType, PixelFormatMapper, StreamMap,
    };
    fn s(v: &str) -> String {
        v.to_string()
    }
    #[test]
    fn test_multi_input_finds_all_inputs() {
        let args = vec![
            s("-i"),
            s("video.mp4"),
            s("-i"),
            s("audio.flac"),
            s("-i"),
            s("subtitle.srt"),
            s("-c:v"),
            s("copy"),
            s("output.mkv"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs_all().len(), 3);
        assert_eq!(parsed.inputs_all()[0].path, "video.mp4");
        assert_eq!(parsed.inputs_all()[1].path, "audio.flac");
        assert_eq!(parsed.inputs_all()[2].path, "subtitle.srt");
    }
    #[test]
    fn test_single_input_via_accessor() {
        let args = vec![s("-i"), s("input.mkv"), s("output.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs_all().len(), 1);
        assert_eq!(parsed.inputs_all()[0].path, "input.mkv");
    }
    #[test]
    fn test_complex_filter_extracted() {
        let args = vec![
            s("-i"),
            s("a.mp4"),
            s("-i"),
            s("b.mp4"),
            s("-filter_complex"),
            s("[0:v][1:v]overlay=W/2:0[out]"),
            s("-map"),
            s("[out]"),
            s("output.mp4"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(
            parsed.complex_filter(),
            Some("[0:v][1:v]overlay=W/2:0[out]")
        );
    }
    #[test]
    fn test_complex_filter_none_when_absent() {
        let args = vec![s("-i"), s("a.mp4"), s("out.mp4")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert!(parsed.complex_filter().is_none());
    }
    #[test]
    fn test_stream_maps_parsed() {
        let args = vec![
            s("-i"),
            s("in.mp4"),
            s("-map"),
            s("0:v:0"),
            s("-map"),
            s("0:a:1"),
            s("out.mkv"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let maps = parsed.stream_maps();
        assert_eq!(maps.len(), 2);
        assert_eq!(maps[0].input_idx, 0);
        assert_eq!(maps[0].stream_type, Some(MapStreamType::Video));
        assert_eq!(maps[0].stream_idx, Some(0));
        assert_eq!(maps[1].stream_type, Some(MapStreamType::Audio));
        assert_eq!(maps[1].stream_idx, Some(1));
    }
    #[test]
    fn test_metadata_extracted() {
        let args = vec![
            s("-i"),
            s("in.mp4"),
            s("-metadata"),
            s("title=Hello World"),
            s("-metadata"),
            s("artist=COOLJAPAN"),
            s("out.mp4"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let meta = parsed.all_metadata();
        assert_eq!(meta.get("title").map(|s| s.as_str()), Some("Hello World"));
        assert_eq!(meta.get("artist").map(|s| s.as_str()), Some("COOLJAPAN"));
    }
    #[test]
    fn test_seek_start_seconds() {
        let args = vec![s("-i"), s("in.mp4"), s("-ss"), s("30.5"), s("out.mp4")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let ss = parsed.seek_start().expect("should have seek");
        assert!((ss - 30.5).abs() < 0.001);
    }
    #[test]
    fn test_duration_seconds() {
        let args = vec![s("-i"), s("in.mp4"), s("-t"), s("120.0"), s("out.mp4")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let t = parsed.duration().expect("should have duration");
        assert!((t - 120.0).abs() < 0.001);
    }
    #[test]
    fn test_seek_start_timecode() {
        let args = vec![s("-i"), s("in.mp4"), s("-ss"), s("00:01:30"), s("out.mp4")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let ss = parsed.seek_start().expect("should have seek");
        assert!((ss - 90.0).abs() < 0.001);
    }
    #[test]
    fn test_container_mp4() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("mp4"), Some("mp4"));
    }
    #[test]
    fn test_container_mkv_matroska() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("mkv"), Some("matroska"));
        assert_eq!(
            ContainerMapper::ffmpeg_to_oximedia("matroska"),
            Some("matroska")
        );
    }
    #[test]
    fn test_container_webm() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("webm"), Some("webm"));
    }
    #[test]
    fn test_container_mov_quicktime() {
        assert_eq!(
            ContainerMapper::ffmpeg_to_oximedia("mov"),
            Some("quicktime")
        );
    }
    #[test]
    fn test_container_ts_mpegts() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("ts"), Some("mpegts"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("mts"), Some("mpegts"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("m2ts"), Some("mpegts"));
    }
    #[test]
    fn test_container_ogg_wav_flac() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("ogg"), Some("ogg"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("wav"), Some("wav"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("flac"), Some("flac"));
    }
    #[test]
    fn test_container_avi_flv() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("avi"), Some("avi"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("flv"), Some("flv"));
    }
    #[test]
    fn test_container_unknown() {
        assert!(ContainerMapper::ffmpeg_to_oximedia("xyz_unknown").is_none());
    }
    #[test]
    fn test_container_case_insensitive() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("MP4"), Some("mp4"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("MKV"), Some("matroska"));
    }
    #[test]
    fn test_pixel_fmt_yuv420p() {
        assert_eq!(
            PixelFormatMapper::ffmpeg_to_oximedia("yuv420p"),
            Some("yuv420p")
        );
    }
    #[test]
    fn test_pixel_fmt_yuv422p() {
        assert_eq!(
            PixelFormatMapper::ffmpeg_to_oximedia("yuv422p"),
            Some("yuv422p")
        );
    }
    #[test]
    fn test_pixel_fmt_yuv444p() {
        assert_eq!(
            PixelFormatMapper::ffmpeg_to_oximedia("yuv444p"),
            Some("yuv444p")
        );
    }
    #[test]
    fn test_pixel_fmt_nv12() {
        assert_eq!(PixelFormatMapper::ffmpeg_to_oximedia("nv12"), Some("nv12"));
    }
    #[test]
    fn test_pixel_fmt_p010le() {
        assert_eq!(
            PixelFormatMapper::ffmpeg_to_oximedia("p010le"),
            Some("p010le")
        );
    }
    #[test]
    fn test_pixel_fmt_rgb24_rgba() {
        assert_eq!(
            PixelFormatMapper::ffmpeg_to_oximedia("rgb24"),
            Some("rgb24")
        );
        assert_eq!(PixelFormatMapper::ffmpeg_to_oximedia("rgba"), Some("rgba"));
    }
    #[test]
    fn test_pixel_fmt_unknown() {
        assert!(PixelFormatMapper::ffmpeg_to_oximedia("xyz_unknown_fmt").is_none());
    }
    #[test]
    fn test_filter_chain_two_nodes() {
        let nodes = parse_filter_chain("scale=1280:720,setsar=1");
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].name, "scale");
        assert_eq!(nodes[1].name, "setsar");
    }
    #[test]
    fn test_filter_chain_single_node() {
        let nodes = parse_filter_chain("hflip");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "hflip");
    }
    #[test]
    fn test_filter_chain_args_parsed() {
        let nodes = parse_filter_chain("scale=1920:1080");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "scale");
        assert_eq!(nodes[0].positional_args, vec!["1920", "1080"]);
    }
    #[test]
    fn test_filter_chain_named_args() {
        let nodes = parse_filter_chain("scale=w=1280:h=720");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "scale");
        assert!(nodes[0]
            .named_args
            .iter()
            .any(|(k, v)| k == "w" && v == "1280"));
        assert!(nodes[0]
            .named_args
            .iter()
            .any(|(k, v)| k == "h" && v == "720"));
    }
    #[test]
    fn test_filter_chain_multiple_nodes() {
        let nodes = parse_filter_chain("scale=1280:720,vflip,hflip");
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].name, "scale");
        assert_eq!(nodes[1].name, "vflip");
        assert_eq!(nodes[2].name, "hflip");
    }
    #[test]
    fn test_lavfi_color() {
        assert!(FilterGraphParser::is_lavfi_source("color"));
        assert!(FilterGraphParser::is_lavfi_source("colour"));
    }
    #[test]
    fn test_lavfi_testsrc() {
        assert!(FilterGraphParser::is_lavfi_source("testsrc"));
        assert!(FilterGraphParser::is_lavfi_source("testsrc2"));
    }
    #[test]
    fn test_lavfi_smptebars() {
        assert!(FilterGraphParser::is_lavfi_source("smptebars"));
        assert!(FilterGraphParser::is_lavfi_source("smptehdbars"));
    }
    #[test]
    fn test_lavfi_sine() {
        assert!(FilterGraphParser::is_lavfi_source("sine"));
    }
    #[test]
    fn test_lavfi_anullsrc_nullsrc() {
        assert!(FilterGraphParser::is_lavfi_source("anullsrc"));
        assert!(FilterGraphParser::is_lavfi_source("nullsrc"));
    }
    #[test]
    fn test_lavfi_not_scale() {
        assert!(!FilterGraphParser::is_lavfi_source("scale"));
        assert!(!FilterGraphParser::is_lavfi_source("hflip"));
        assert!(!FilterGraphParser::is_lavfi_source("overlay"));
    }
    #[test]
    fn test_lavfi_with_params_stripped() {
        assert!(FilterGraphParser::is_lavfi_source(
            "color=c=red:size=1280x720"
        ));
    }
    #[test]
    fn test_validator_valid_graph() {
        use crate::filter_graph::FilterGraph;
        let graph = FilterGraph::parse("scale=1280:720").expect("parse");
        let problems = FilterGraphValidator::check_connections(&graph);
        assert!(
            problems.is_empty(),
            "simple chain with no labels: {:?}",
            problems
        );
    }
    #[test]
    fn test_validator_internal_connection_valid() {
        use crate::filter_graph::FilterGraph;
        let graph = FilterGraph::parse("[a]scale=1280:720[b];[b]hflip[c]").expect("parse");
        let problems = FilterGraphValidator::check_connections(&graph);
        assert_eq!(problems.len(), 2, "external I/O labels: {:?}", problems);
    }
    #[test]
    fn test_validator_unconnected_internal_label() {
        use crate::filter_graph::FilterGraph;
        let graph = FilterGraph::parse("[in]scale=1280:720[out1];[in]hflip[out2]").expect("parse");
        let problems = FilterGraphValidator::check_connections(&graph);
        assert!(!problems.is_empty(), "should detect unmatched pads");
    }
    #[test]
    fn test_builder_codec_video_audio() {
        let args = ArgumentBuilder::new()
            .input("input.mp4")
            .codec_video("av1")
            .crf(30)
            .codec_audio("opus")
            .output("output.webm")
            .build();
        assert!(args.contains(&"-c:v".to_string()));
        assert!(args.contains(&"av1".to_string()));
        assert!(args.contains(&"-c:a".to_string()));
        assert!(args.contains(&"opus".to_string()));
        assert!(args.contains(&"-crf".to_string()));
        assert!(args.contains(&"30".to_string()));
        assert_eq!(args.last(), Some(&"output.webm".to_string()));
    }
    #[test]
    fn test_builder_bitrate_video_audio() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .bitrate_video("4M")
            .bitrate_audio("192k")
            .output("out.mp4")
            .build();
        let bv_idx = args.iter().position(|a| a == "-b:v").expect("-b:v");
        assert_eq!(args[bv_idx + 1], "4M");
        let ba_idx = args.iter().position(|a| a == "-b:a").expect("-b:a");
        assert_eq!(args[ba_idx + 1], "192k");
    }
    #[test]
    fn test_builder_scale() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .scale(1280, 720)
            .output("out.mp4")
            .build();
        let vf_idx = args.iter().position(|a| a == "-vf").expect("-vf");
        assert_eq!(args[vf_idx + 1], "scale=1280:720");
    }
    #[test]
    fn test_builder_fps() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .fps(29.97)
            .output("out.mp4")
            .build();
        let r_idx = args.iter().position(|a| a == "-r").expect("-r");
        assert_eq!(args[r_idx + 1], "29.97");
    }
    #[test]
    fn test_builder_seek_duration() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .seek(30.0)
            .duration(60.0)
            .output("out.mp4")
            .build();
        let ss_idx = args.iter().position(|a| a == "-ss").expect("-ss");
        assert_eq!(args[ss_idx + 1], "30");
        let t_idx = args.iter().position(|a| a == "-t").expect("-t");
        assert_eq!(args[t_idx + 1], "60");
    }
    #[test]
    fn test_builder_metadata() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .metadata("title", "Test Video")
            .output("out.mp4")
            .build();
        let m_idx = args
            .iter()
            .position(|a| a == "-metadata")
            .expect("-metadata");
        assert_eq!(args[m_idx + 1], "title=Test Video");
    }
    #[test]
    fn test_builder_preset() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .preset("medium")
            .output("out.mp4")
            .build();
        let p_idx = args.iter().position(|a| a == "-preset").expect("-preset");
        assert_eq!(args[p_idx + 1], "medium");
    }
    #[test]
    fn test_builder_input_before_output() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .codec_video("av1")
            .output("out.webm")
            .build();
        let i_idx = args.iter().position(|a| a == "-i").expect("-i");
        let out_idx = args.iter().position(|a| a == "out.webm").expect("out");
        assert!(i_idx < out_idx);
    }
    #[test]
    fn test_builder_multi_input() {
        let args = ArgumentBuilder::new()
            .input("video.mp4")
            .input("audio.flac")
            .codec_video("copy")
            .codec_audio("copy")
            .output("output.mkv")
            .build();
        let i_positions: Vec<usize> = args
            .iter()
            .enumerate()
            .filter_map(|(i, a)| if a == "-i" { Some(i) } else { None })
            .collect();
        assert_eq!(i_positions.len(), 2);
        assert_eq!(args[i_positions[0] + 1], "video.mp4");
        assert_eq!(args[i_positions[1] + 1], "audio.flac");
    }
    #[test]
    fn test_deprecated_vcodec() {
        let warnings = FfmpegDiagnostics::check_deprecated_options(&["-vcodec", "libx264"]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].deprecated_flag, "-vcodec");
        assert_eq!(warnings[0].replacement, "-c:v");
    }
    #[test]
    fn test_deprecated_acodec() {
        let warnings = FfmpegDiagnostics::check_deprecated_options(&["-acodec", "aac"]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].deprecated_flag, "-acodec");
        assert_eq!(warnings[0].replacement, "-c:a");
    }
    #[test]
    fn test_deprecated_ab() {
        let warnings = FfmpegDiagnostics::check_deprecated_options(&["-ab", "128k"]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].deprecated_flag, "-ab");
        assert_eq!(warnings[0].replacement, "-b:a");
    }
    #[test]
    fn test_deprecated_none_for_modern_flags() {
        let warnings = FfmpegDiagnostics::check_deprecated_options(&[
            "-c:v", "av1", "-c:a", "opus", "-b:a", "128k",
        ]);
        assert!(warnings.is_empty());
    }
    #[test]
    fn test_deprecated_multiple() {
        let warnings = FfmpegDiagnostics::check_deprecated_options(&[
            "-vcodec", "libx264", "-acodec", "aac", "-ab", "128k",
        ]);
        assert_eq!(warnings.len(), 3);
    }
    #[test]
    fn test_compat_score_known_codecs() {
        let args_strs = vec![
            s("-i"),
            s("in.mkv"),
            s("-c:v"),
            s("libaom-av1"),
            s("-c:a"),
            s("libopus"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args_strs).expect("parse");
        let score = FfmpegCompatDiagnostics::score(&parsed);
        assert!(
            score >= 0.9,
            "known patent-free codecs should score near 1.0, got {}",
            score
        );
    }
    #[test]
    fn test_compat_score_unknown_codec_deducts() {
        let args_strs = vec![
            s("-i"),
            s("in.mkv"),
            s("-c:v"),
            s("completely_unknown_video_codec_xyz"),
            s("out.mkv"),
        ];
        let parsed = FfmpegArgs::parse(&args_strs).expect("parse");
        let score = FfmpegCompatDiagnostics::score(&parsed);
        assert!(score < 1.0, "unknown codec should deduct from score");
    }
    #[test]
    fn test_compat_score_bounded_zero_to_one() {
        let args_strs = vec![
            s("-i"),
            s("in.mkv"),
            s("-c:v"),
            s("unk1"),
            s("-c:a"),
            s("unk2"),
            s("-f"),
            s("xyz_unknown_format"),
            s("out.mkv"),
        ];
        let parsed = FfmpegArgs::parse(&args_strs).expect("parse");
        let score = FfmpegCompatDiagnostics::score(&parsed);
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }
    #[test]
    fn test_hint_direct_av1() {
        let args_strs = vec![
            s("-i"),
            s("in.mkv"),
            s("-c:v"),
            s("libaom-av1"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args_strs).expect("parse");
        let hints = generate_hints(&parsed);
        let av1_hint = hints.iter().find(|h| h.original == "libaom-av1");
        assert!(av1_hint.is_some(), "should have a hint for libaom-av1");
        let h = av1_hint.expect("tested above");
        assert_eq!(h.translated, "av1");
        assert!(
            (h.confidence - 1.0).abs() < 0.001,
            "direct match = confidence 1.0"
        );
    }
    #[test]
    fn test_hint_patent_substituted() {
        let args_strs = vec![s("-i"), s("in.mp4"), s("-c:v"), s("libx265"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args_strs).expect("parse");
        let hints = generate_hints(&parsed);
        let hint = hints.iter().find(|h| h.original == "libx265");
        assert!(hint.is_some());
        let h = hint.expect("tested above");
        assert_eq!(h.translated, "av1");
        assert!(
            (h.confidence - 0.5).abs() < 0.001,
            "substitution = confidence 0.5"
        );
        assert!(h.note.is_some(), "substitution should have a note");
    }
    #[test]
    fn test_hint_codec_libx265_av1() {
        use crate::codec_map::CodecMap;
        let cm = CodecMap::new();
        let entry = cm.lookup("libx265").expect("libx265 should exist");
        assert_eq!(entry.oxi_name, "av1");
    }
    #[test]
    fn test_stream_map_parse_video() {
        let sm = StreamMap::parse("0:v:0").expect("parse");
        assert_eq!(sm.input_idx, 0);
        assert_eq!(sm.stream_type, Some(MapStreamType::Video));
        assert_eq!(sm.stream_idx, Some(0));
        assert!(!sm.negative);
    }
    #[test]
    fn test_stream_map_parse_audio() {
        let sm = StreamMap::parse("0:a:1").expect("parse");
        assert_eq!(sm.input_idx, 0);
        assert_eq!(sm.stream_type, Some(MapStreamType::Audio));
        assert_eq!(sm.stream_idx, Some(1));
    }
    #[test]
    fn test_stream_map_parse_negative() {
        let sm = StreamMap::parse("-0:s").expect("parse");
        assert!(sm.negative);
        assert_eq!(sm.stream_type, Some(MapStreamType::Subtitle));
    }
    #[test]
    fn test_stream_map_parse_all_streams() {
        let sm = StreamMap::parse("1").expect("parse");
        assert_eq!(sm.input_idx, 1);
        assert!(sm.stream_type.is_none());
        assert!(sm.stream_idx.is_none());
    }
    #[test]
    fn test_parse_time_float() {
        assert!((parse_time_str("123.45").expect("ok") - 123.45).abs() < 0.001);
    }
    #[test]
    fn test_parse_time_hhmmss() {
        assert!((parse_time_str("01:30:00").expect("ok") - 5400.0).abs() < 0.001);
    }
    #[test]
    fn test_parse_time_mmss() {
        assert!((parse_time_str("02:30").expect("ok") - 150.0).abs() < 0.001);
    }
}
