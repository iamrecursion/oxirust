fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Pure-Rust protobuf codegen: `protox` parses the .proto file directly
    // instead of shelling out to an external `protoc` binary, so no
    // system-level protoc installation is required to build this crate.
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("cargo:rerun-if-changed=proto/farm.proto");
        let file_descriptor_set = protox::compile(["proto/farm.proto"], ["proto"])?;
        tonic_prost_build::configure().compile_fds(file_descriptor_set)?;
    }
    Ok(())
}
