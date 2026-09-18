//! Smoke tests for newly-wired orphan modules in oximedia-compat-ffmpeg.
#[test]
fn test_batch_mode_error() {
    let _ = std::any::type_name::<oximedia_compat_ffmpeg::batch_mode::BatchError>();
}
#[test]
fn test_compat_ext_stream_map() {
    let _ = std::any::type_name::<oximedia_compat_ffmpeg::compat_ext::StreamMap>();
}
#[test]
fn test_concat_compat_timestamp() {
    let _ = std::any::type_name::<oximedia_compat_ffmpeg::concat_compat::Timestamp>();
}
#[test]
fn test_lavfi_compat_param_value() {
    use oximedia_compat_ffmpeg::lavfi_compat::ParamValue;
    let pv = ParamValue::parse("1920");
    let _ = pv;
}
#[test]
fn test_preset_translator_options() {
    let _ = std::any::type_name::<
        oximedia_compat_ffmpeg::preset_translator::TranslatedEncoderOptions,
    >();
}
#[test]
fn test_two_pass_config() {
    let _ = std::any::type_name::<oximedia_compat_ffmpeg::two_pass::PassLogConfig>();
}
