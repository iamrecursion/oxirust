//! `oximedia-cv2` — Pure-Rust OpenCV cv2 API drop-in CLI.
//!
//! Exposes the `oximedia-compat-cv2` function surface as named-arg subcommands.
//! Mirrors the OpenCV Python API naming but uses idiomatic CLI flags.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "dnn")]
use oximedia_compat_cv2::dnn::{blob_from_image, read_net_from_onnx};
use oximedia_compat_cv2::mat::{Mat, MatType, Size};
use oximedia_compat_cv2::{
    color::cvt_color,
    edge::{canny, convert_scale_abs, laplacian, sobel},
    features::{corner_harris, fast_feature_detector, Orb},
    filter::{bilateral_filter, gaussian_blur, median_blur},
    geometry::{flip, resize, rotate},
    histogram::equalize_hist,
    image_io::{imread, imwrite},
    morphology::{dilate, erode, get_structuring_element, morphology_ex},
    threshold::{adaptive_threshold, threshold},
    ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C, COLOR_BGR2GRAY, COLOR_BGR2HSV,
    COLOR_BGR2RGB, COLOR_RGB2BGR, IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED, INTER_CUBIC,
    INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST, MORPH_BLACKHAT, MORPH_CLOSE, MORPH_CROSS,
    MORPH_DILATE, MORPH_ELLIPSE, MORPH_ERODE, MORPH_GRADIENT, MORPH_OPEN, MORPH_RECT, MORPH_TOPHAT,
    ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, THRESH_BINARY, THRESH_BINARY_INV,
    THRESH_OTSU,
};
use std::path::PathBuf;

// ── CLI Definition ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "oximedia-cv2",
    version,
    about = "Pure-Rust OpenCV cv2 drop-in — OxiMedia compat layer",
    long_about = "OpenCV cv2 API compatibility CLI. Dispatches to oximedia-compat-cv2.\n\
                  Use --list-functions to see all supported functions.\n\
                  Use --list-constants to see all ~134 OpenCV constants."
)]
struct Cli {
    /// Print all supported cv2 functions and exit.
    #[arg(long, global = true)]
    list_functions: bool,

    /// Print all OpenCV constants (~134) grouped by category, then exit.
    #[arg(long, global = true)]
    list_constants: bool,

    /// Print dispatch details for the given subcommand without executing, then exit.
    #[arg(long, global = true)]
    explain: bool,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ImreadFlags {
    Color,
    Grayscale,
    Unchanged,
}

impl ImreadFlags {
    fn to_i32(self) -> i32 {
        match self {
            Self::Color => IMREAD_COLOR,
            Self::Grayscale => IMREAD_GRAYSCALE,
            Self::Unchanged => IMREAD_UNCHANGED,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ColorCodeArg {
    Bgr2rgb,
    Rgb2bgr,
    Bgr2gray,
    Bgr2hsv,
    Hsv2bgr,
}

impl ColorCodeArg {
    fn to_i32(self) -> i32 {
        match self {
            Self::Bgr2rgb => COLOR_BGR2RGB,
            Self::Rgb2bgr => COLOR_RGB2BGR,
            Self::Bgr2gray => COLOR_BGR2GRAY,
            Self::Bgr2hsv => COLOR_BGR2HSV,
            Self::Hsv2bgr => oximedia_compat_cv2::COLOR_HSV2BGR,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum InterpolationArg {
    Nearest,
    Linear,
    Cubic,
    Lanczos4,
}

impl InterpolationArg {
    fn to_i32(self) -> i32 {
        match self {
            Self::Nearest => INTER_NEAREST,
            Self::Linear => INTER_LINEAR,
            Self::Cubic => INTER_CUBIC,
            Self::Lanczos4 => INTER_LANCZOS4,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ThresholdKindArg {
    Binary,
    BinaryInv,
    Otsu,
}

impl ThresholdKindArg {
    fn to_i32(self) -> i32 {
        match self {
            Self::Binary => THRESH_BINARY,
            Self::BinaryInv => THRESH_BINARY_INV,
            Self::Otsu => THRESH_BINARY | THRESH_OTSU,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum MorphKernelArg {
    Rect,
    Ellipse,
    Cross,
}

impl MorphKernelArg {
    fn to_i32(self) -> i32 {
        match self {
            Self::Rect => MORPH_RECT,
            Self::Ellipse => MORPH_ELLIPSE,
            Self::Cross => MORPH_CROSS,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FlipCodeArg {
    /// Flip around the x-axis (vertical flip).
    X,
    /// Flip around the y-axis (horizontal flip).
    Y,
    /// Flip around both axes (180° rotation equivalent).
    Both,
}

impl FlipCodeArg {
    fn to_i32(self) -> i32 {
        // cv2.flip semantics: 0 = vertical, 1 = horizontal, -1 = both.
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Both => -1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RotateCodeArg {
    Cw90,
    Rot180,
    Ccw90,
}

impl RotateCodeArg {
    fn to_i32(self) -> i32 {
        match self {
            Self::Cw90 => ROTATE_90_CLOCKWISE,
            Self::Rot180 => ROTATE_180,
            Self::Ccw90 => ROTATE_90_COUNTERCLOCKWISE,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum AdaptiveMethodArg {
    Mean,
    Gaussian,
}

impl AdaptiveMethodArg {
    fn to_i32(self) -> i32 {
        match self {
            Self::Mean => ADAPTIVE_THRESH_MEAN_C,
            Self::Gaussian => ADAPTIVE_THRESH_GAUSSIAN_C,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum AdaptiveThreshKindArg {
    Binary,
    BinaryInv,
}

impl AdaptiveThreshKindArg {
    fn to_i32(self) -> i32 {
        match self {
            Self::Binary => THRESH_BINARY,
            Self::BinaryInv => THRESH_BINARY_INV,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum MorphOpArg {
    Erode,
    Dilate,
    Open,
    Close,
    Gradient,
    Tophat,
    Blackhat,
}

impl MorphOpArg {
    fn to_i32(self) -> i32 {
        match self {
            Self::Erode => MORPH_ERODE,
            Self::Dilate => MORPH_DILATE,
            Self::Open => MORPH_OPEN,
            Self::Close => MORPH_CLOSE,
            Self::Gradient => MORPH_GRADIENT,
            Self::Tophat => MORPH_TOPHAT,
            Self::Blackhat => MORPH_BLACKHAT,
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Read image file and write to output (format conversion via extension).
    Imread {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "color")]
        flags: ImreadFlags,
    },

    /// Convert color space (e.g. bgr2rgb, bgr2gray, bgr2hsv).
    CvtColor {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        code: ColorCodeArg,
    },

    /// Resize image to given dimensions.
    Resize {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        width: u32,
        #[arg(long)]
        height: u32,
        #[arg(long, default_value = "linear")]
        interpolation: InterpolationArg,
    },

    /// Apply Gaussian blur.
    GaussianBlur {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "5")]
        ksize: i32,
        #[arg(long, default_value = "1.4")]
        sigma: f64,
    },

    /// Canny edge detection.
    Canny {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "100")]
        threshold1: f64,
        #[arg(long, default_value = "200")]
        threshold2: f64,
    },

    /// Apply threshold to grayscale image.
    Threshold {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "128")]
        thresh: f64,
        #[arg(long, default_value = "binary")]
        kind: ThresholdKindArg,
    },

    /// Morphological erosion.
    Erode {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "3")]
        ksize: u32,
        #[arg(long, default_value = "rect")]
        shape: MorphKernelArg,
    },

    /// Morphological dilation.
    Dilate {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "3")]
        ksize: u32,
        #[arg(long, default_value = "rect")]
        shape: MorphKernelArg,
    },

    /// Histogram equalization (grayscale only).
    EqualizeHist { input: PathBuf, output: PathBuf },

    /// Median blur filter.
    MedianBlur {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "3")]
        ksize: i32,
    },

    /// Bilateral filter (edge-preserving).
    BilateralFilter {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "9")]
        d: i32,
        #[arg(long, default_value = "75")]
        sigma_color: f64,
        #[arg(long, default_value = "75")]
        sigma_space: f64,
    },

    /// Print image metadata (rows, cols, channels, dtype).
    Probe { input: PathBuf },

    /// Flip image around the x-axis, y-axis, or both.
    Flip {
        input: PathBuf,
        output: PathBuf,
        /// Axis to flip around: `x`, `y`, or `both`.
        #[arg(long, default_value = "y")]
        code: FlipCodeArg,
    },

    /// Rotate image by a multiple of 90 degrees.
    Rotate {
        input: PathBuf,
        output: PathBuf,
        /// `cw90` (90° clockwise), `rot180` (180°), `ccw90` (90° counter-clockwise).
        #[arg(long, default_value = "cw90")]
        code: RotateCodeArg,
    },

    /// Sobel first-order gradient. Output is `convertScaleAbs`-normalised so
    /// it can be saved as a PNG/JPEG.
    Sobel {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "1")]
        dx: i32,
        #[arg(long, default_value = "0")]
        dy: i32,
        #[arg(long, default_value = "3")]
        ksize: i32,
    },

    /// Laplacian second derivative. Output is `convertScaleAbs`-normalised so
    /// it can be saved as a PNG/JPEG.
    Laplacian {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "3")]
        ksize: i32,
    },

    /// Adaptive thresholding (mean-C or Gaussian-C).
    AdaptiveThreshold {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "255")]
        max_value: f64,
        #[arg(long, default_value = "mean")]
        method: AdaptiveMethodArg,
        #[arg(long = "type", default_value = "binary")]
        kind: AdaptiveThreshKindArg,
        #[arg(long, default_value = "11")]
        block_size: i32,
        #[arg(long, default_value = "2")]
        c: f64,
    },

    /// Morphology compound operation (open/close/gradient/tophat/blackhat/...).
    MorphologyEx {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "open")]
        op: MorphOpArg,
        #[arg(long, default_value = "3")]
        ksize: u32,
        #[arg(long, default_value = "rect")]
        shape: MorphKernelArg,
    },

    /// FAST corner detection. Writes a visualisation with each corner drawn
    /// as a small box on top of the input image.
    FastCorners {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "10")]
        threshold: i32,
        #[arg(long = "nonmax-suppression", default_value_t = true)]
        nonmax_suppression: bool,
    },

    /// Harris corner detection. Writes a `convertScaleAbs`-normalised
    /// response map (grayscale).
    HarrisCorners {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "2")]
        block_size: i32,
        #[arg(long, default_value = "3")]
        ksize: i32,
        #[arg(long, default_value = "0.04")]
        k: f64,
    },

    /// ORB keypoint + descriptor detection (oriented FAST + rotated BRIEF).
    OrbDetect {
        input: PathBuf,
        #[arg(long, default_value = "500")]
        num_features: usize,
        /// Optional output PNG with detected keypoints drawn as small boxes.
        #[arg(long)]
        draw: Option<PathBuf>,
    },

    /// Run a forward pass through an ONNX classifier and report top-K logits.
    /// Requires `--features dnn`.
    #[cfg(feature = "dnn")]
    DnnForward {
        /// Path to the ONNX model file.
        #[arg(long)]
        model: PathBuf,
        /// Input image to classify.
        #[arg(long)]
        input: PathBuf,
        /// Resize target as `WxH` (e.g. `224x224`).
        #[arg(long, default_value = "224x224")]
        resize: String,
        /// Pixel scale factor (e.g. `0.00392156862` for 1/255).
        #[arg(long, default_value = "0.00392156862")]
        scale: f32,
        /// Per-channel mean as `R,G,B` (or post-swap channel order).
        #[arg(long, default_value = "0,0,0")]
        mean: String,
        /// Swap red and blue channels (BGR ↔ RGB).
        #[arg(long)]
        swap_rb: bool,
        /// Specific output tensor name; defaults to first model output.
        #[arg(long)]
        output_tensor: Option<String>,
        /// Number of top-K logits to print (clamped to length of output).
        #[arg(long, default_value = "5")]
        top_k: usize,
    },
}

// ── Main ───────────────────────────────────────────────────────────────────────

fn main() {
    // Install the Pure-Rust `rustls-rustcrypto` crypto provider as the
    // process-wide default before any TLS connection can be opened. See
    // `oximedia_net::tls_provider` for details. Idempotent.
    oximedia_net::install_default_crypto_provider();

    let cli = Cli::parse();

    if cli.list_functions {
        print_functions();
        return;
    }
    if cli.list_constants {
        print_constants();
        return;
    }

    let Some(cmd) = cli.cmd else {
        eprintln!(
            "No subcommand given. Use --help for usage or --list-functions for available functions."
        );
        std::process::exit(1);
    };

    if cli.explain {
        print_explain(&cmd);
        return;
    }

    if let Err(e) = run(cmd) {
        eprintln!("oximedia-cv2: error: {e:#}");
        std::process::exit(1);
    }
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Imread {
            input,
            output,
            flags,
        } => {
            let mat = imread(&input, flags.to_i32())
                .with_context(|| format!("imread failed for {}", input.display()))?;
            imwrite(&output, &mat)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!(
                "{}x{} {} ch -> {}",
                mat.cols,
                mat.rows,
                mat.channels(),
                output.display()
            );
        }

        Cmd::CvtColor {
            input,
            output,
            code,
        } => {
            let mat = imread(&input, IMREAD_COLOR)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let out = cvt_color(&mat, code.to_i32()).with_context(|| "cvtColor failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!(
                "{}x{} -> {} ch -> {}",
                out.cols,
                out.rows,
                out.channels(),
                output.display()
            );
        }

        Cmd::Resize {
            input,
            output,
            width,
            height,
            interpolation,
        } => {
            let mat = imread(&input, IMREAD_COLOR)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let out = resize(
                &mat,
                Size {
                    width: width as usize,
                    height: height as usize,
                },
                interpolation.to_i32(),
            )
            .with_context(|| "resize failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!("{}x{} -> {}", out.cols, out.rows, output.display());
        }

        Cmd::GaussianBlur {
            input,
            output,
            ksize,
            sigma,
        } => {
            let mat = imread(&input, IMREAD_COLOR)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let out =
                gaussian_blur(&mat, ksize, sigma, sigma).with_context(|| "gaussianBlur failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!(
                "gaussianBlur ksize={ksize} sigma={sigma} -> {}",
                output.display()
            );
        }

        Cmd::Canny {
            input,
            output,
            threshold1,
            threshold2,
        } => {
            let mat = imread(&input, IMREAD_GRAYSCALE)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let out =
                canny(&mat, threshold1, threshold2, 3, false).with_context(|| "canny failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!(
                "canny t1={threshold1} t2={threshold2} -> {}",
                output.display()
            );
        }

        Cmd::Threshold {
            input,
            output,
            thresh,
            kind,
        } => {
            let mat = imread(&input, IMREAD_GRAYSCALE)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let (effective_t, out) = threshold(&mat, thresh, 255.0, kind.to_i32())
                .with_context(|| "threshold failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!(
                "threshold effective_t={effective_t:.1} -> {}",
                output.display()
            );
        }

        Cmd::Erode {
            input,
            output,
            ksize,
            shape,
        } => {
            let mat = imread(&input, IMREAD_GRAYSCALE)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let k = ksize as i32;
            let kernel = get_structuring_element(shape.to_i32(), k)
                .with_context(|| "getStructuringElement failed")?;
            let out = erode(&mat, &kernel, 1).with_context(|| "erode failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!("erode ksize={ksize} -> {}", output.display());
        }

        Cmd::Dilate {
            input,
            output,
            ksize,
            shape,
        } => {
            let mat = imread(&input, IMREAD_GRAYSCALE)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let k = ksize as i32;
            let kernel = get_structuring_element(shape.to_i32(), k)
                .with_context(|| "getStructuringElement failed")?;
            let out = dilate(&mat, &kernel, 1).with_context(|| "dilate failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!("dilate ksize={ksize} -> {}", output.display());
        }

        Cmd::EqualizeHist { input, output } => {
            let mat = imread(&input, IMREAD_GRAYSCALE)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let out = equalize_hist(&mat).with_context(|| "equalizeHist failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!("equalizeHist -> {}", output.display());
        }

        Cmd::MedianBlur {
            input,
            output,
            ksize,
        } => {
            let mat = imread(&input, IMREAD_COLOR)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let out = median_blur(&mat, ksize).with_context(|| "medianBlur failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!("medianBlur ksize={ksize} -> {}", output.display());
        }

        Cmd::BilateralFilter {
            input,
            output,
            d,
            sigma_color,
            sigma_space,
        } => {
            let mat = imread(&input, IMREAD_COLOR)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let out = bilateral_filter(&mat, d, sigma_color, sigma_space)
                .with_context(|| "bilateralFilter failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!(
                "bilateralFilter d={d} sigma_c={sigma_color} sigma_s={sigma_space} -> {}",
                output.display()
            );
        }

        Cmd::Probe { input } => {
            let mat = imread(&input, IMREAD_UNCHANGED)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            println!("file:     {}", input.display());
            println!("rows:     {}", mat.rows);
            println!("cols:     {}", mat.cols);
            println!("channels: {}", mat.channels());
            println!("dtype:    {:?}", mat.mat_type);
            println!("bytes:    {}", mat.data.len());
        }

        Cmd::Flip {
            input,
            output,
            code,
        } => {
            let mat = imread(&input, IMREAD_UNCHANGED)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let out = flip(&mat, code.to_i32()).with_context(|| "flip failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!("flip code={} -> {}", code.to_i32(), output.display());
        }

        Cmd::Rotate {
            input,
            output,
            code,
        } => {
            let mat = imread(&input, IMREAD_UNCHANGED)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let out = rotate(&mat, code.to_i32()).with_context(|| "rotate failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!(
                "rotate code={} -> {}x{} {}",
                code.to_i32(),
                out.cols,
                out.rows,
                output.display()
            );
        }

        Cmd::Sobel {
            input,
            output,
            dx,
            dy,
            ksize,
        } => {
            let mat = imread(&input, IMREAD_GRAYSCALE)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let f32_out = sobel(&mat, dx, dy, ksize).with_context(|| "sobel failed")?;
            // Underlying op returns CV_32FC1; the PNG/JPEG encoder only
            // accepts u8, so map through convertScaleAbs to land in u8 space.
            let displayable = convert_scale_abs(&f32_out)
                .with_context(|| "convertScaleAbs after sobel failed")?;
            imwrite(&output, &displayable)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!(
                "sobel dx={dx} dy={dy} ksize={ksize} -> {}",
                output.display()
            );
        }

        Cmd::Laplacian {
            input,
            output,
            ksize,
        } => {
            let mat = imread(&input, IMREAD_GRAYSCALE)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let f32_out = laplacian(&mat, ksize).with_context(|| "laplacian failed")?;
            let displayable = convert_scale_abs(&f32_out)
                .with_context(|| "convertScaleAbs after laplacian failed")?;
            imwrite(&output, &displayable)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!("laplacian ksize={ksize} -> {}", output.display());
        }

        Cmd::AdaptiveThreshold {
            input,
            output,
            max_value,
            method,
            kind,
            block_size,
            c,
        } => {
            let mat = imread(&input, IMREAD_GRAYSCALE)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let out = adaptive_threshold(
                &mat,
                max_value,
                method.to_i32(),
                kind.to_i32(),
                block_size,
                c,
            )
            .with_context(|| "adaptiveThreshold failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            // Count threshold-passed pixels for an at-a-glance sanity check.
            let on_count = out.data.iter().filter(|&&v| v != 0).count();
            println!(
                "adaptiveThreshold block={block_size} c={c} -> {} (on_pixels={on_count})",
                output.display()
            );
        }

        Cmd::MorphologyEx {
            input,
            output,
            op,
            ksize,
            shape,
        } => {
            let mat = imread(&input, IMREAD_UNCHANGED)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let kernel = get_structuring_element(shape.to_i32(), ksize as i32)
                .with_context(|| "getStructuringElement failed")?;
            let out =
                morphology_ex(&mat, op.to_i32(), &kernel).with_context(|| "morphologyEx failed")?;
            imwrite(&output, &out)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!(
                "morphologyEx op={} ksize={ksize} -> {}",
                op.to_i32(),
                output.display()
            );
        }

        Cmd::FastCorners {
            input,
            output,
            threshold,
            nonmax_suppression,
        } => {
            let mat = imread(&input, IMREAD_GRAYSCALE)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let kps = fast_feature_detector(&mat, threshold, nonmax_suppression)
                .with_context(|| "FAST detection failed")?;
            // Visualise on a 3-channel canvas so the markers are visible.
            let mut canvas = grayscale_to_bgr(&mat);
            for kp in &kps {
                draw_marker_3x3(&mut canvas, kp.pt.x, kp.pt.y);
            }
            imwrite(&output, &canvas)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            println!(
                "fastCorners threshold={threshold} nms={nonmax_suppression} corners={} -> {}",
                kps.len(),
                output.display()
            );
        }

        Cmd::HarrisCorners {
            input,
            output,
            block_size,
            ksize,
            k,
        } => {
            let mat = imread(&input, IMREAD_GRAYSCALE)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let response =
                corner_harris(&mat, block_size, ksize, k).with_context(|| "cornerHarris failed")?;
            // Normalise the f32 response map into u8 [0,255] so it is
            // round-trippable as PNG/JPEG (and also more visually useful).
            let displayable = harris_response_to_gray(&response)
                .with_context(|| "Harris response normalisation failed")?;
            imwrite(&output, &displayable)
                .with_context(|| format!("imwrite failed for {}", output.display()))?;
            // Count 'strong' responders for a quick sanity check.
            let strong = displayable.data.iter().filter(|&&v| v > 64).count();
            println!(
                "harrisCorners block={block_size} ksize={ksize} k={k} strong_pixels={strong} -> {}",
                output.display()
            );
        }

        Cmd::OrbDetect {
            input,
            num_features,
            draw,
        } => {
            let mat = imread(&input, IMREAD_UNCHANGED)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let detector = Orb::new(num_features);
            let (kps, descriptors) = detector
                .detect_and_compute(&mat, None)
                .with_context(|| "ORB detect_and_compute failed")?;
            println!(
                "orb keypoints={} descriptors_rows={}",
                kps.len(),
                descriptors.rows
            );
            for (i, kp) in kps.iter().take(10).enumerate() {
                println!(
                    "  kp[{i}] x={:.2} y={:.2} size={:.2} angle={:.2} response={:.4}",
                    kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response
                );
            }
            if let Some(out_path) = draw {
                let mut canvas = if mat.mat_type == MatType::CV_8UC1 {
                    grayscale_to_bgr(&mat)
                } else {
                    mat.clone()
                };
                for kp in &kps {
                    draw_marker_3x3(&mut canvas, kp.pt.x, kp.pt.y);
                }
                imwrite(&out_path, &canvas)
                    .with_context(|| format!("imwrite failed for {}", out_path.display()))?;
                println!("draw -> {}", out_path.display());
            }
        }

        #[cfg(feature = "dnn")]
        Cmd::DnnForward {
            model,
            input,
            resize: resize_arg,
            scale,
            mean,
            swap_rb,
            output_tensor,
            top_k,
        } => {
            let (target_w, target_h) = parse_size_wxh(&resize_arg)
                .with_context(|| format!("invalid --resize value: {resize_arg:?}"))?;
            let (mean_a, mean_b, mean_c) =
                parse_triplet(&mean).with_context(|| format!("invalid --mean value: {mean:?}"))?;

            let image = imread(&input, IMREAD_COLOR)
                .with_context(|| format!("imread failed for {}", input.display()))?;
            let blob = blob_from_image(
                &image,
                scale,
                (target_w, target_h),
                (mean_a, mean_b, mean_c),
                swap_rb,
                false,
            )
            .with_context(|| "blob_from_image failed")?;

            let net = read_net_from_onnx(&model)
                .with_context(|| format!("read_net_from_onnx failed for {}", model.display()))?;
            let logits = match output_tensor.as_deref() {
                Some(name) => net
                    .forward_named(&blob, name)
                    .with_context(|| format!("forward_named({name:?}) failed"))?,
                None => net.forward(&blob).with_context(|| "forward failed")?,
            };
            print_dnn_output(&logits, top_k);
        }
    }
    Ok(())
}

// ── Helpers shared by the new subcommands ────────────────────────────────────

/// Convert a `CV_8UC1` Mat to a `CV_8UC3` BGR Mat by replicating the luma into
/// each colour plane.  Used by `fast-corners`, `harris-corners`, and
/// `orb-detect` to render visualisations on top of grayscale inputs.
fn grayscale_to_bgr(src: &Mat) -> Mat {
    if src.mat_type != MatType::CV_8UC1 {
        // Defensive: only used after an `IMREAD_GRAYSCALE` so this is never
        // hit in practice, but we keep the caller-side guarantee by cloning.
        return src.clone();
    }
    let mut bgr = Vec::with_capacity(src.data.len() * 3);
    for &g in &src.data {
        bgr.extend_from_slice(&[g, g, g]);
    }
    Mat::from_bgr_bytes(bgr, src.rows, src.cols)
}

/// Draw a small 3×3 red marker centred at `(x, y)` on a `CV_8UC3` BGR canvas.
/// Out-of-bounds pixels are silently clipped.
fn draw_marker_3x3(canvas: &mut Mat, x: f32, y: f32) {
    if canvas.mat_type != MatType::CV_8UC3 {
        return;
    }
    let cx = x.round() as i32;
    let cy = y.round() as i32;
    let cols = canvas.cols as i32;
    let rows = canvas.rows as i32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let px = cx + dx;
            let py = cy + dy;
            if px < 0 || py < 0 || px >= cols || py >= rows {
                continue;
            }
            let off = (py as usize * canvas.step) + (px as usize * 3);
            // BGR → red marker.
            if off + 3 <= canvas.data.len() {
                canvas.data[off] = 0; // B
                canvas.data[off + 1] = 0; // G
                canvas.data[off + 2] = 255; // R
            }
        }
    }
}

/// Decode a `CV_32FC1` Harris response Mat into an 8-bit grayscale Mat scaled
/// to `[0, 255]` for visualisation.  The min and max of the response are
/// mapped to 0 and 255 respectively (linear stretch).
fn harris_response_to_gray(response: &Mat) -> Result<Mat> {
    if response.mat_type != MatType::CV_32FC1 {
        anyhow::bail!(
            "harris_response_to_gray: expected CV_32FC1, got {:?}",
            response.mat_type
        );
    }
    let n = response.rows * response.cols;
    if response.data.len() != n * 4 {
        anyhow::bail!(
            "harris_response_to_gray: byte-length mismatch (have {}, expected {})",
            response.data.len(),
            n * 4
        );
    }

    let mut values = Vec::with_capacity(n);
    for chunk in response.data.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().context("f32 chunk decode failed")?;
        values.push(f32::from_le_bytes(arr));
    }

    let (mut min_v, mut max_v) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &values {
        if v.is_finite() {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        return Ok(Mat::from_gray_bytes(
            vec![0u8; n],
            response.rows,
            response.cols,
        ));
    }
    let span = (max_v - min_v).max(f32::EPSILON);
    let bytes: Vec<u8> = values
        .into_iter()
        .map(|v| (((v - min_v) / span) * 255.0).clamp(0.0, 255.0) as u8)
        .collect();
    Ok(Mat::from_gray_bytes(bytes, response.rows, response.cols))
}

/// Parse a `WIDTHxHEIGHT` string (e.g. `"224x224"`) into `(u32, u32)`.
#[cfg(feature = "dnn")]
fn parse_size_wxh(s: &str) -> Result<(u32, u32)> {
    let (w_str, h_str) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| anyhow::anyhow!("expected WIDTHxHEIGHT, got {s:?}"))?;
    let w: u32 = w_str
        .trim()
        .parse()
        .with_context(|| format!("width is not a u32: {w_str:?}"))?;
    let h: u32 = h_str
        .trim()
        .parse()
        .with_context(|| format!("height is not a u32: {h_str:?}"))?;
    if w == 0 || h == 0 {
        anyhow::bail!("resize dimensions must be non-zero, got {w}x{h}");
    }
    Ok((w, h))
}

/// Parse a comma-separated triple of floats (e.g. `"0.485,0.456,0.406"`).
#[cfg(feature = "dnn")]
fn parse_triplet(s: &str) -> Result<(f32, f32, f32)> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        anyhow::bail!(
            "expected 3 comma-separated values, got {} ({:?})",
            parts.len(),
            s
        );
    }
    let parse_f32 = |p: &str| -> Result<f32> {
        p.trim()
            .parse::<f32>()
            .with_context(|| format!("not a float: {p:?}"))
    };
    Ok((
        parse_f32(parts[0])?,
        parse_f32(parts[1])?,
        parse_f32(parts[2])?,
    ))
}

/// Decode a `CV_32FC1` logits Mat into top-K `(class_index, score)` lines.
/// Falls back to printing shape + first-N values when the Mat is non-2-D.
#[cfg(feature = "dnn")]
fn print_dnn_output(out: &Mat, top_k: usize) {
    let elem_size = out.mat_type.elem_size();
    let depth = out.mat_type.depth_bytes();
    if depth != 4 {
        // Non-f32 outputs are printed as raw shape only.
        println!(
            "dnn output: rows={} cols={} channels={} dtype={:?} bytes={}",
            out.rows,
            out.cols,
            out.channels(),
            out.mat_type,
            out.data.len()
        );
        return;
    }
    let total = out.data.len() / 4;
    let mut floats = Vec::with_capacity(total);
    for chunk in out.data.chunks_exact(4) {
        if let Ok(arr) = <[u8; 4]>::try_from(chunk) {
            floats.push(f32::from_ne_bytes(arr));
        }
    }
    println!(
        "dnn output: rows={} cols={} channels={} dtype={:?} elem_size={elem_size}",
        out.rows,
        out.cols,
        out.channels(),
        out.mat_type
    );

    // For the `[N, classes]` (rank 2 logits) case `rows==1` is the typical
    // setup.  We pick the *first row* and rank by score for top-K display.
    if out.cols > 0 && out.rows > 0 {
        let row_len = out.cols;
        let row: Vec<f32> = floats.iter().take(row_len).copied().collect();
        let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(indexed.len()).max(1);
        println!("top-{k}:");
        for (rank, (idx, score)) in indexed.iter().take(k).enumerate() {
            println!("  #{rank} class={idx} score={score:.6}");
        }
    } else {
        // Fallback: dump first 8 values for whatever shape we got.
        let preview: Vec<String> = floats.iter().take(8).map(|v| format!("{v:.6}")).collect();
        println!("first values: [{}]", preview.join(", "));
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn print_explain(cmd: &Cmd) {
    let (name, dispatch) = match cmd {
        Cmd::Imread { .. } => ("imread", "oximedia_compat_cv2::image_io::imread -> imwrite"),
        Cmd::CvtColor { .. } => ("cvt-color", "oximedia_compat_cv2::color::cvt_color"),
        Cmd::Resize { .. } => (
            "resize",
            "oximedia_compat_cv2::geometry::resize -> oximedia-scaling",
        ),
        Cmd::GaussianBlur { .. } => (
            "gaussian-blur",
            "oximedia_compat_cv2::filter::gaussian_blur",
        ),
        Cmd::Canny { .. } => (
            "canny",
            "oximedia_compat_cv2::edge::canny -> oximedia-image::canny",
        ),
        Cmd::Threshold { .. } => (
            "threshold",
            "oximedia_compat_cv2::threshold::threshold (Otsu/Triangle auto-select)",
        ),
        Cmd::Erode { .. } => ("erode", "oximedia_compat_cv2::morphology::erode"),
        Cmd::Dilate { .. } => ("dilate", "oximedia_compat_cv2::morphology::dilate"),
        Cmd::EqualizeHist { .. } => (
            "equalize-hist",
            "oximedia_compat_cv2::histogram::equalize_hist",
        ),
        Cmd::MedianBlur { .. } => ("median-blur", "oximedia_compat_cv2::filter::median_blur"),
        Cmd::BilateralFilter { .. } => (
            "bilateral-filter",
            "oximedia_compat_cv2::filter::bilateral_filter",
        ),
        Cmd::Probe { .. } => (
            "probe",
            "oximedia_compat_cv2::image_io::imread (metadata only)",
        ),
        Cmd::Flip { .. } => ("flip", "oximedia_compat_cv2::geometry::flip"),
        Cmd::Rotate { .. } => ("rotate", "oximedia_compat_cv2::geometry::rotate"),
        Cmd::Sobel { .. } => (
            "sobel",
            "oximedia_compat_cv2::edge::sobel -> convertScaleAbs",
        ),
        Cmd::Laplacian { .. } => (
            "laplacian",
            "oximedia_compat_cv2::edge::laplacian -> convertScaleAbs",
        ),
        Cmd::AdaptiveThreshold { .. } => (
            "adaptive-threshold",
            "oximedia_compat_cv2::threshold::adaptive_threshold",
        ),
        Cmd::MorphologyEx { .. } => (
            "morphology-ex",
            "oximedia_compat_cv2::morphology::morphology_ex",
        ),
        Cmd::FastCorners { .. } => (
            "fast-corners",
            "oximedia_compat_cv2::features::fast_feature_detector",
        ),
        Cmd::HarrisCorners { .. } => (
            "harris-corners",
            "oximedia_compat_cv2::features::corner_harris -> normalise -> u8",
        ),
        Cmd::OrbDetect { .. } => (
            "orb-detect",
            "oximedia_compat_cv2::features::Orb::detect_and_compute",
        ),
        #[cfg(feature = "dnn")]
        Cmd::DnnForward { .. } => (
            "dnn-forward",
            "oximedia_compat_cv2::dnn::{read_net_from_onnx, blob_from_image, Net::forward}",
        ),
    };
    println!("subcommand:  {name}");
    println!("dispatch:    {dispatch}");
    println!("(--explain: no output file written)");
}

fn print_functions() {
    let fns: &[(&str, &str)] = &[
        ("imread", "Read image file -> Mat"),
        ("imwrite", "Write Mat -> image file"),
        ("imdecode", "Decode in-memory bytes -> Mat"),
        ("imencode", "Encode Mat -> bytes"),
        (
            "cvt_color",
            "Color space conversion (BGR<->RGB/Gray/HSV/HLS/Lab/YUV)",
        ),
        ("resize", "Resize image (nearest/bilinear/bicubic/Lanczos4)"),
        ("flip", "Flip image horizontally / vertically / both"),
        ("rotate", "Rotate 90/180/270 degrees"),
        ("warp_affine", "Apply affine 2x3 transformation matrix"),
        (
            "get_rotation_matrix_2d",
            "Compute 2x3 rotation matrix from angle/center/scale",
        ),
        (
            "copy_make_border",
            "Add padding around image (replicate/reflect/constant)",
        ),
        ("gaussian_blur", "Gaussian blur with separable convolution"),
        ("blur", "Normalized box (averaging) filter"),
        ("box_filter", "Box filter with optional normalization"),
        ("median_blur", "Median filter (salt-and-pepper removal)"),
        ("bilateral_filter", "Edge-preserving bilateral filter"),
        (
            "filter_2d",
            "General 2D convolution with user-supplied kernel",
        ),
        ("pyramid_down", "Gaussian pyramid down-sample (x0.5)"),
        ("pyramid_up", "Gaussian pyramid up-sample (x2)"),
        (
            "canny",
            "Canny edge detection (dispatches to oximedia-image)",
        ),
        ("sobel", "Sobel gradient in X and/or Y direction"),
        ("laplacian", "Laplacian second derivative"),
        ("scharr", "Scharr gradient (alternative to Sobel)"),
        (
            "convert_scale_abs",
            "Scale + offset then take absolute value (for Sobel->u8)",
        ),
        (
            "threshold",
            "Fixed threshold with Otsu/Triangle auto-select",
        ),
        (
            "adaptive_threshold",
            "Local adaptive threshold (mean-C or Gaussian-C)",
        ),
        ("erode", "Morphological erosion"),
        ("dilate", "Morphological dilation"),
        (
            "morphology_ex",
            "Open/Close/Gradient/TopHat/BlackHat/HitMiss",
        ),
        (
            "get_structuring_element",
            "Build morphological kernel (rect/ellipse/cross)",
        ),
        ("good_features_to_track", "Shi-Tomasi corner detection"),
        (
            "corner_harris",
            "Harris corner response map (CV_32FC1 output)",
        ),
        (
            "fast_feature_detector",
            "FAST keypoint detection with optional NMS",
        ),
        (
            "orb_create",
            "ORB feature detector (oriented FAST + rotated BRIEF descriptor + Hamming matcher)",
        ),
        (
            "orb-detect",
            "Run ORB detector on an image and report keypoints",
        ),
        (
            "dnn-forward",
            "ONNX classifier forward pass with cv2.dnn API (requires --features dnn)",
        ),
        ("flip", "CLI: flip subcommand (cv2.flip)"),
        ("rotate", "CLI: rotate subcommand (cv2.rotate)"),
        (
            "adaptive-threshold",
            "CLI: adaptive-threshold subcommand (cv2.adaptiveThreshold)",
        ),
        (
            "morphology-ex",
            "CLI: morphology-ex subcommand (cv2.morphologyEx)",
        ),
        ("fast-corners", "CLI: fast-corners subcommand (cv2.FAST)"),
        (
            "harris-corners",
            "CLI: harris-corners subcommand (cv2.cornerHarris)",
        ),
        (
            "calc_optical_flow_pyr_lk",
            "Lucas-Kanade sparse optical flow",
        ),
        (
            "find_contours",
            "Find contours from binary mask (RETR_LIST/TREE/EXTERNAL)",
        ),
        ("draw_contours", "Draw contours onto Mat"),
        (
            "bounding_rect",
            "Axis-aligned bounding rectangle of a contour",
        ),
        ("contour_area", "Signed area of a contour"),
        ("arc_length", "Perimeter of a contour"),
        ("approx_poly_dp", "Douglas-Peucker contour approximation"),
        ("hough_lines", "Standard Hough line transform"),
        ("hough_lines_p", "Probabilistic Hough line transform"),
        ("hough_circles", "Hough circle transform"),
        ("equalize_hist", "Histogram equalization (grayscale)"),
        ("calc_hist", "Compute histogram for one or more channels"),
        (
            "compare_hist",
            "Compare two histograms (correlation/intersection/chi^2/Bhatt.)",
        ),
        ("normalize", "Normalize Mat to given range"),
        (
            "match_template",
            "Template matching (SQDIFF/CCORR/CCOEFF, +/-NORMED)",
        ),
        ("min_max_loc", "Find global min and max and their locations"),
        ("line", "Draw a Bresenham line"),
        ("rectangle", "Draw an axis-aligned rectangle"),
        ("circle", "Draw a circle (Bresenham midpoint algorithm)"),
        ("ellipse", "Draw an ellipse or ellipse arc"),
        ("polylines", "Draw a polyline"),
        ("fill_poly", "Fill a polygon"),
        ("put_text", "Render text with Hershey font"),
        ("add", "Per-element addition with saturation"),
        ("subtract", "Per-element subtraction with saturation"),
        ("multiply", "Per-element multiplication with saturation"),
        ("divide", "Per-element division with saturation"),
        (
            "add_weighted",
            "Weighted per-element addition (alpha-blend)",
        ),
        ("abs_diff", "Per-element absolute difference"),
        ("bitwise_and", "Per-element bitwise AND"),
        ("bitwise_or", "Per-element bitwise OR"),
        ("bitwise_xor", "Per-element bitwise XOR"),
        ("bitwise_not", "Per-element bitwise NOT"),
        ("in_range", "Test each pixel against [lower, upper] bounds"),
        ("compare", "Per-element comparison -> binary mask"),
        (
            "connected_components",
            "Two-pass union-find connected-component labeling",
        ),
        (
            "connected_components_with_stats",
            "Connected components with bounding-box + area + centroid",
        ),
    ];

    println!("{:<40} Description", "Function");
    println!("{}", "-".repeat(80));
    for (name, desc) in fns {
        println!("{:<40} {}", name, desc);
    }
    println!("\nTotal: {} functions", fns.len());
}

fn print_constants() {
    // Constants are reflected at build time from
    // `oximedia-compat-cv2/src/constants.rs` via `build.rs`. Each entry is
    // `(category, name, type, value)`. We group by category for display
    // (matching the prior hand-maintained format), with no-category items
    // listed under `[uncategorized]` at the end.
    let entries = oximedia_compat_cv2::constants_list::LIST_CONSTANTS;

    // Stable bucketing in original sort order so the output is deterministic.
    let mut categories: Vec<&str> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for (cat, _, _, _) in entries.iter() {
        if seen.insert(cat) {
            categories.push(cat);
        }
    }

    let mut total = 0usize;
    for category in &categories {
        let display = if category.is_empty() {
            "uncategorized"
        } else {
            category
        };
        println!("\n[{display}]");
        for (cat, name, _, val) in entries.iter() {
            if cat == category {
                println!("  {:<40} = {}", name, val);
                total += 1;
            }
        }
    }
    println!("\nTotal: {total} constants");
}
