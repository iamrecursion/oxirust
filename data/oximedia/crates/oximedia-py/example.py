#!/usr/bin/env python3
"""
Example usage of OxiMedia Python bindings.

This example demonstrates:
1. Video decoding (AV1, VP9, VP8)
2. Video encoding (AV1)
3. Audio decoding (Opus)
4. Container demuxing (Matroska/WebM)
5. Container muxing (Matroska/WebM)
"""

import oximedia


def decode_video_example():
    """Example of decoding AV1 video."""
    print("=== Video Decoding Example ===")

    # Create AV1 decoder
    decoder = oximedia.Av1Decoder()

    # Example packet data (normally from a demuxer)
    # packet_data = b"..."

    # Send packet to decoder
    # decoder.send_packet(packet_data, pts=0)

    # Receive decoded frame
    # frame = decoder.receive_frame()
    # if frame:
    #     print(f"Decoded frame: {frame.width}x{frame.height}")
    #     print(f"Format: {frame.format}")
    #     print(f"PTS: {frame.pts}")
    #
    #     # Access plane data
    #     y_plane = frame.plane_data(0)  # Y plane
    #     u_plane = frame.plane_data(1)  # U plane
    #     v_plane = frame.plane_data(2)  # V plane

    print("Decoder created successfully")


def encode_video_example():
    """Example of encoding video to AV1."""
    print("\n=== Video Encoding Example ===")

    # Create encoder configuration
    config = oximedia.EncoderConfig(
        width=1920,
        height=1080,
        framerate=(30, 1),  # 30 fps
        crf=28.0,           # Quality (lower = better, 18-28 typical)
        preset="medium",    # Speed/quality tradeoff
        keyint=250          # Keyframe every 250 frames
    )

    print(f"Config: {config}")

    # Create AV1 encoder
    encoder = oximedia.Av1Encoder(config)

    # Create a test frame
    frame = oximedia.VideoFrame(
        1920, 1080,
        oximedia.PixelFormat("yuv420p")
    )
    frame.set_pts(0)

    # Send frame to encoder
    # encoder.send_frame(frame)

    # Receive encoded packet
    # packet = encoder.receive_packet()
    # if packet:
    #     print(f"Encoded packet:")
    #     print(f"  Size: {len(packet['data'])} bytes")
    #     print(f"  PTS: {packet['pts']}")
    #     print(f"  Keyframe: {packet['keyframe']}")

    print("Encoder created successfully")


def decode_audio_example():
    """Example of decoding Opus audio."""
    print("\n=== Audio Decoding Example ===")

    # Create Opus decoder
    decoder = oximedia.OpusDecoder(
        sample_rate=48000,  # 48 kHz
        channels=2          # Stereo
    )

    print(f"Decoder: {decoder}")

    # Decode packet (normally from a demuxer)
    # packet_data = b"..."
    # audio_frame = decoder.decode_packet(packet_data)

    # print(f"Audio frame:")
    # print(f"  Samples: {audio_frame.sample_count}")
    # print(f"  Sample rate: {audio_frame.sample_rate}Hz")
    # print(f"  Channels: {audio_frame.channels}")
    # print(f"  Format: {audio_frame.format}")
    # print(f"  Duration: {audio_frame.duration_seconds():.3f}s")

    # # Get samples as float32 or int16
    # samples_f32 = audio_frame.to_f32()
    # samples_i16 = audio_frame.to_i16()


def demux_container_example():
    """Example of demuxing a Matroska/WebM file."""
    print("\n=== Container Demuxing Example ===")

    # This example requires an actual .mkv or .webm file
    # Uncomment to use with a real file:

    # demuxer = oximedia.MatroskaDemuxer("video.mkv")
    # demuxer.probe()
    #
    # # Get stream information
    # streams = demuxer.streams()
    # print(f"Found {len(streams)} streams:")
    #
    # for stream in streams:
    #     print(f"\nStream {stream.index}:")
    #     print(f"  Codec: {stream.codec}")
    #     print(f"  Timebase: {stream.timebase}")
    #
    #     if stream.width:
    #         print(f"  Video: {stream.width}x{stream.height}")
    #
    #     if stream.sample_rate:
    #         print(f"  Audio: {stream.sample_rate}Hz, {stream.channels} channels")
    #
    # # Read packets
    # packet_count = 0
    # while True:
    #     try:
    #         packet = demuxer.read_packet()
    #         packet_count += 1
    #
    #         if packet_count % 100 == 0:
    #             print(f"Read {packet_count} packets...")
    #
    #     except StopIteration:
    #         print(f"End of file. Total packets: {packet_count}")
    #         break

    print("Demuxer example code ready (requires input file)")


def mux_container_example():
    """Example of muxing packets to a Matroska/WebM file."""
    print("\n=== Container Muxing Example ===")

    # This example shows how to create a muxer
    # Uncomment to use:

    # # Create muxer
    # muxer = oximedia.MatroskaMuxer(
    #     "output.mkv",
    #     title="My Video"
    # )
    #
    # # Add streams (you'd get these from a demuxer or create manually)
    # # video_stream = ...  # StreamInfo object
    # # audio_stream = ...  # StreamInfo object
    #
    # # video_index = muxer.add_stream(video_stream)
    # # audio_index = muxer.add_stream(audio_stream)
    #
    # # Write header
    # muxer.write_header()
    #
    # # Write packets (interleaved from different streams)
    # # for packet in packets:
    # #     muxer.write_packet(packet)
    #
    # # Finalize file
    # muxer.write_trailer()
    #
    # print("File written successfully!")

    print("Muxer example code ready (requires stream info and packets)")


def pixel_format_example():
    """Example of working with pixel formats."""
    print("\n=== Pixel Format Example ===")

    # Create pixel formats
    yuv420 = oximedia.PixelFormat("yuv420p")
    print(f"YUV 4:2:0: {yuv420}")
    print(f"  Is planar: {yuv420.is_planar()}")
    print(f"  Plane count: {yuv420.plane_count()}")

    yuv444 = oximedia.PixelFormat("yuv444p")
    print(f"\nYUV 4:4:4: {yuv444}")
    print(f"  Plane count: {yuv444.plane_count()}")


def sample_format_example():
    """Example of working with audio sample formats."""
    print("\n=== Sample Format Example ===")

    # Create sample formats
    f32_format = oximedia.SampleFormat("f32")
    print(f"F32: {f32_format}")
    print(f"  Sample size: {f32_format.sample_size()} bytes")

    i16_format = oximedia.SampleFormat("i16")
    print(f"\nI16: {i16_format}")
    print(f"  Sample size: {i16_format.sample_size()} bytes")


def rational_example():
    """Example of working with rational numbers (frame rates, timebases)."""
    print("\n=== Rational Number Example ===")

    # Frame rate: 30 fps
    framerate = oximedia.Rational(30, 1)
    print(f"Frame rate: {framerate} = {framerate.to_float()} fps")

    # Frame rate: 23.976 fps (24000/1001)
    framerate_ntsc = oximedia.Rational(24000, 1001)
    print(f"NTSC frame rate: {framerate_ntsc} = {framerate_ntsc.to_float():.3f} fps")

    # Timebase: 1/1000 (milliseconds)
    timebase = oximedia.Rational(1, 1000)
    print(f"Timebase: {timebase} = {timebase.to_float()} seconds")


def main():
    """Run all examples."""
    print("OxiMedia Python Bindings Examples")
    print("=" * 50)

    try:
        decode_video_example()
        encode_video_example()
        decode_audio_example()
        demux_container_example()
        mux_container_example()
        pixel_format_example()
        sample_format_example()
        rational_example()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nNote: Some examples are commented out and require")
        print("actual media files to run. Uncomment the code to use them.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
