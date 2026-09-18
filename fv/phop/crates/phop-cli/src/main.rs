//! `phop` — command-line symbolic discovery.
//!
//! ```text
//! phop discover data.csv --top-k 5 --format latex
//! ```

use clap::{Parser, Subcommand, ValueEnum};
use phop_core::{AnySolution, Backend, Config, DataSet, Discoverer};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Candidate-pool cap for the `--method rich` engine (matches the benchmark harness).
const RICH_CAND_CAP: usize = 2000;

#[derive(Parser)]
#[command(
    name = "phop",
    version,
    about = "Differentiable symbolic discovery on the EML operator"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Discover closed-form expressions from a CSV dataset (last column = target).
    Discover {
        /// Path to the CSV file (with header row).
        data: PathBuf,
        /// Population / candidate budget.
        #[arg(long, default_value_t = 256)]
        population: usize,
        /// Maximum tree depth.
        #[arg(long, default_value_t = 3)]
        max_depth: usize,
        /// Maximum optimization epochs per candidate.
        #[arg(long, default_value_t = 1000)]
        max_epochs: usize,
        /// Number of Pareto solutions to report.
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        /// Adam learning rate.
        #[arg(long, default_value_t = 0.05)]
        learning_rate: f64,
        /// RNG seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Output format.
        #[arg(long, value_enum, default_value_t = Format::Table)]
        format: Format,
        /// Search method: structural enumeration or differentiable Gumbel-Softmax topology.
        #[arg(long, value_enum, default_value_t = Method::Enumerate)]
        method: Method,
        /// Target column: a 0-based column index OR a header name.
        /// When absent, the last column is used.
        #[arg(long)]
        target: Option<String>,
        /// Comma-separated feature columns to use (indices or header names), e.g. `--features a,c`
        /// or `--features 0,2`. When absent, all non-target columns are features.
        #[arg(long, value_delimiter = ',')]
        features: Option<Vec<String>>,
        /// Complexity penalty weight (used by `--method gumbel`).
        #[arg(long)]
        lambda_complexity: Option<f64>,
        /// Sparsity penalty weight (used by `--method gumbel`).
        #[arg(long)]
        lambda_sparsity: Option<f64>,
        /// Parsimony penalty weight (used by `--method gumbel`).
        #[arg(long)]
        lambda_parsimony: Option<f64>,
        /// Suppress the informational "phop: loaded ..." line on stderr.
        #[arg(long)]
        quiet: bool,
        /// Print the resolved configuration to stderr (overrides `--quiet`).
        #[arg(long)]
        verbose: bool,
        /// Write formatted results to FILE instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Compute backend for constant fitting. `cuda` requires a build with
        /// `--features gpu-cuda` and an NVIDIA device; `metal` likewise requires
        /// `--features gpu-metal` and a Metal device (macOS). Otherwise it falls back to CPU.
        #[arg(long, value_enum, default_value_t = Gpu::Cpu)]
        gpu: Gpu,
        /// After discovery, symbolically analyze the best law via the oxieml CAS (derivative,
        /// antiderivative, Maclaurin series, +∞ limit) and print it to stderr.
        #[arg(long)]
        analyze: bool,
        /// After discovery, print a CERTIFIED range enclosure of the best law over the data's
        /// bounding box plus a certified root search along x0 (interval Newton/Krawczyk) to stderr.
        #[arg(long)]
        certify: bool,
        /// Buckingham-π reduction: give each feature's SI dimension as 7 comma-separated integer
        /// exponents `[T,L,M,Θ,I,N,J]`, features separated by `;` — e.g.
        /// `--units "0,1,0,0,0,0,0;1,0,0,0,0,0,0"`. Inputs are reduced to dimensionless π-groups
        /// before discovery.
        #[arg(long)]
        units: Option<String>,
    },
    /// Apply a previously discovered law to new data.
    ///
    /// MODEL is a JSON file produced by `phop discover --format json` (or a bare serialized model);
    /// DATA is a CSV with the same columns as the training data (target last, or via `--target`).
    Predict {
        /// Path to the model JSON.
        model: PathBuf,
        /// Path to the input CSV (header row; same columns as training).
        data: PathBuf,
        /// Target column (index or name) to compare against; when present, prints R² to stderr.
        #[arg(long)]
        target: Option<String>,
        /// Feature columns to use (indices or names); defaults to all non-target columns.
        #[arg(long, value_delimiter = ',')]
        features: Option<Vec<String>>,
        /// Write predictions to FILE instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Gpu {
    /// CPU (default).
    Cpu,
    /// NVIDIA CUDA GPU.
    Cuda,
    /// Apple Metal GPU (macOS).
    Metal,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Method {
    /// Exact structural enumeration with per-candidate constant fitting (shallow, precise).
    Enumerate,
    /// Differentiable Gumbel-Softmax topology search (Layer B).
    Gumbel,
    /// Differentiable tree-shape search with per-node expand/terminate gates (depth learning).
    Gated,
    /// Gated search warm-started from the enumeration seed (robust deeper recovery).
    GatedWarm,
    /// Run enumerate + gumbel + gated and merge their Pareto fronts (best across methods).
    Auto,
    /// Rich-leaf engine: affine (`Σaᵢxᵢ+b`) and log-linear (`Σaᵢ ln xᵢ+b`, i.e. monomial) leaves —
    /// recovers products / power-laws / ratios that the bare-leaf search cannot. Own representation.
    Rich,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Format {
    /// Aligned table of complexity, MSE, and the pretty EML form.
    Table,
    /// LaTeX math for each solution.
    Latex,
    /// A standalone Rust function per solution.
    Rust,
    /// JSON array, one object per ranked solution.
    Json,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("phop: error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Read the CSV header row.
fn read_headers(path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("failed to open CSV '{}': {e}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("failed to read CSV header of '{}': {e}", path.display()))?;
    Ok(headers.iter().map(str::to_string).collect())
}

/// Resolve a column token (a 0-based index or a header name) into a column index.
fn resolve_one(token: &str, headers: &[String]) -> Result<usize, Box<dyn std::error::Error>> {
    if let Ok(idx) = token.parse::<usize>() {
        if idx < headers.len() {
            return Ok(idx);
        }
        return Err(format!(
            "column index {idx} out of range ({} columns)",
            headers.len()
        )
        .into());
    }
    headers.iter().position(|h| h == token).ok_or_else(|| {
        format!(
            "unknown column '{token}'; available: [{}]",
            headers.join(", ")
        )
        .into()
    })
}

#[allow(clippy::too_many_arguments)]
fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Discover {
            data,
            population,
            max_depth,
            max_epochs,
            top_k,
            learning_rate,
            seed,
            format,
            method,
            target,
            features,
            lambda_complexity,
            lambda_sparsity,
            lambda_parsimony,
            quiet,
            verbose,
            output,
            gpu,
            analyze,
            certify,
            units,
        } => {
            let mut ds = load_dataset(&data, &target, &features)?;

            // Buckingham-π reduction: collapse dimensioned inputs to dimensionless groups first.
            if let Some(spec) = &units {
                let dims = parse_units(spec, ds.n_vars())?;
                let (reduced, groups) = ds.to_dimensionless(&dims)?;
                if verbose || !quiet {
                    eprintln!(
                        "phop: reduced {} feature(s) to {} dimensionless π-group(s): {groups:?}",
                        ds.n_vars(),
                        groups.len()
                    );
                }
                ds = reduced;
            }

            if verbose || !quiet {
                eprintln!(
                    "phop: loaded {} rows, {} feature(s) [{}] -> {}",
                    ds.len(),
                    ds.n_vars(),
                    ds.feature_names.join(", "),
                    ds.target_name
                );
            }

            let backend = match gpu {
                Gpu::Cpu => Backend::Cpu,
                Gpu::Cuda => Backend::Cuda,
                Gpu::Metal => Backend::Metal,
            };
            if gpu == Gpu::Cuda && !cfg!(feature = "gpu-cuda") {
                eprintln!(
                    "phop: note: --gpu cuda requested but this binary was built without the \
                     `gpu-cuda` feature; falling back to CPU"
                );
            }
            if gpu == Gpu::Metal && !cfg!(feature = "gpu-metal") {
                eprintln!(
                    "phop: note: --gpu metal requested but this binary was built without the \
                     `gpu-metal` feature; falling back to CPU"
                );
            }

            let mut cfg = Config::default()
                .population(population)
                .max_depth(max_depth)
                .max_epochs(max_epochs)
                .learning_rate(learning_rate)
                .seed(seed)
                .top_k(top_k)
                .backend(backend);
            if let Some(v) = lambda_complexity {
                cfg.lambda_complexity = v;
            }
            if let Some(v) = lambda_sparsity {
                cfg.lambda_sparsity = v;
            }
            if let Some(v) = lambda_parsimony {
                cfg.lambda_parsimony = v;
            }

            if verbose {
                eprintln!(
                    "phop: config method={} population={} max_depth={} max_epochs={} top_k={} \
                     learning_rate={} seed={} lambda_complexity={} lambda_sparsity={} \
                     lambda_parsimony={}",
                    match method {
                        Method::Enumerate => "enumerate",
                        Method::Gumbel => "gumbel",
                        Method::Gated => "gated",
                        Method::GatedWarm => "gated-warm",
                        Method::Auto => "auto",
                        Method::Rich => "rich",
                    },
                    cfg.population,
                    cfg.max_depth,
                    cfg.max_epochs,
                    cfg.top_k,
                    cfg.learning_rate,
                    cfg.seed,
                    cfg.lambda_complexity,
                    cfg.lambda_sparsity,
                    cfg.lambda_parsimony,
                );
            }

            // The rich-leaf engine uses its own (ANode) representation — affine + log-linear leaves —
            // so it is handled on a separate path rather than the EmlTree Pareto front. Both `rich`
            // (affine only) and `auto` (the merged EML + affine front) render through `render_any`.
            if method == Method::Rich {
                let max_internal = max_depth.clamp(1, 5);
                let mut sols: Vec<AnySolution> =
                    phop_core::discover_affine_pareto(&ds.x, &ds.y, max_internal, RICH_CAND_CAP)
                        .into_iter()
                        .map(AnySolution::Affine)
                        .collect();
                sols.sort_by(|a, b| {
                    a.mse()
                        .partial_cmp(&b.mse())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                sols.truncate(top_k);
                let rows = rows_with_r2(&sols, &ds);
                write_out(output.as_deref(), &render_any(format, &rows))?;
                if analyze {
                    eprintln!("phop: note: --analyze is not available for --method rich (its leaves are not an oxieml EmlTree)");
                }
                if certify {
                    eprintln!("phop: note: --certify is not available for --method rich (its leaves are not an oxieml EmlTree)");
                }
                return Ok(());
            }

            // `auto` now runs the FULL meta-ensemble — the EML-tree searches *and* the rich-leaf
            // affine engine — merged into one Pareto front, so the default search reaches the
            // product/power-law recovery the shallow EML core can't assemble.
            if method == Method::Auto {
                let max_internal = max_depth.clamp(1, 5);
                let mut sols =
                    phop_core::discover_auto_all(&ds, &cfg, max_internal, RICH_CAND_CAP)?;
                sols.truncate(top_k);
                let rows = rows_with_r2(&sols, &ds);
                write_out(output.as_deref(), &render_any(format, &rows))?;
                if analyze {
                    match sols.iter().find_map(AnySolution::as_eml) {
                        Some(best_eml) => print_analysis(best_eml),
                        None => eprintln!(
                            "phop: note: the best laws are rich-leaf (affine) forms; --analyze needs an oxieml EmlTree"
                        ),
                    }
                }
                if certify {
                    match sols.iter().find_map(AnySolution::as_eml) {
                        Some(best_eml) => print_certification(best_eml, &ds),
                        None => eprintln!(
                            "phop: note: the best laws are rich-leaf (affine) forms; --certify needs an oxieml EmlTree"
                        ),
                    }
                }
                return Ok(());
            }

            let front = match method {
                Method::Enumerate => Discoverer::new(cfg).fit(&ds)?,
                Method::Gumbel => phop_core::discover_gumbel(&ds, &cfg)?,
                Method::Gated => phop_core::discover_gated(&ds, &cfg)?,
                Method::GatedWarm => phop_core::discover_gated_warm(&ds, &cfg)?,
                Method::Auto => unreachable!("auto handled above"),
                Method::Rich => unreachable!("rich handled above"),
            };
            let top = front.pareto_top(top_k);

            let rendered = render(format, &top);
            write_out(output.as_deref(), &rendered)?;

            // Discover → analyze: symbolic CAS analysis of the best law (oxieml).
            if analyze {
                if let Some(best) = front.best() {
                    print_analysis(best);
                }
            }
            // Discover → certify: a guaranteed range enclosure + certified root of the best law.
            if certify {
                if let Some(best) = front.best() {
                    print_certification(best, &ds);
                }
            }
            Ok(())
        }
        Command::Predict {
            model,
            data,
            target,
            features,
            output,
        } => {
            let content = std::fs::read_to_string(&model)
                .map_err(|e| format!("failed to read model '{}': {e}", model.display()))?;
            let model_json = extract_model_json(&content)?;
            let sol = phop_core::Solution::from_model_json(&model_json)?;
            let ds = load_dataset(&data, &target, &features)?;
            let preds = sol.predict(&ds.x)?;

            let mut out = String::from("prediction\n");
            for v in preds.iter() {
                out.push_str(&format!("{v}\n"));
            }
            write_out(output.as_deref(), &out)?;

            // If the input carried a real target column, report the model's R² on this data.
            let r2 = r2_of(
                preds.as_slice().unwrap_or(&[]),
                ds.y.as_slice().unwrap_or(&[]),
            );
            if r2.is_finite() {
                eprintln!("phop: R² of the model on this data = {r2:.6}");
            }
            Ok(())
        }
    }
}

/// Load a dataset from a CSV, resolving the target column (index/name, default last) and an optional
/// feature-column subset. Shared by `discover` and `predict`.
fn load_dataset(
    data: &Path,
    target: &Option<String>,
    features: &Option<Vec<String>>,
) -> Result<DataSet, Box<dyn std::error::Error>> {
    let headers = read_headers(data)?;
    let target_idx = match target {
        Some(t) => resolve_one(t, &headers)?,
        None => headers.len().saturating_sub(1),
    };
    let ds = match features {
        Some(tokens) => {
            let feats = tokens
                .iter()
                .map(|t| resolve_one(t, &headers))
                .collect::<Result<Vec<usize>, _>>()?;
            DataSet::from_csv_columns(data, &feats, target_idx)?
        }
        None => DataSet::from_csv_with_target(data, Some(target_idx))?,
    };
    Ok(ds)
}

/// Parse a `--units` spec (features separated by `;`, each 7 comma-separated integer SI exponents
/// `[T,L,M,Θ,I,N,J]`) into one [`phop_core::Dimension`] per feature.
fn parse_units(
    spec: &str,
    n_vars: usize,
) -> Result<Vec<phop_core::Dimension>, Box<dyn std::error::Error>> {
    let dims = spec
        .split(';')
        .map(|tok| {
            let nums = tok
                .split(',')
                .map(|s| s.trim().parse::<i32>())
                .collect::<Result<Vec<i32>, _>>()
                .map_err(|e| format!("--units: not an integer exponent: {e}"))?;
            let arr: phop_core::Dimension = nums.clone().try_into().map_err(|_| {
                format!(
                    "--units: each feature needs 7 exponents, got {}",
                    nums.len()
                )
            })?;
            Ok::<_, Box<dyn std::error::Error>>(arr)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dims.len() != n_vars {
        return Err(format!(
            "--units: expected {n_vars} dimension vector(s) (one per feature), got {}",
            dims.len()
        )
        .into());
    }
    Ok(dims)
}

/// Extract the serialized EML model string from a model file: a `discover --format json` array
/// (rank-1's `model` field), an object with a `model` field, or a bare serialized tree.
fn extract_model_json(content: &str) -> Result<String, Box<dyn std::error::Error>> {
    let v: serde_json::Value =
        serde_json::from_str(content).map_err(|e| format!("model file is not valid JSON: {e}"))?;
    if let Some(arr) = v.as_array() {
        if let Some(m) = arr
            .iter()
            .find_map(|item| item.get("model").and_then(serde_json::Value::as_str))
        {
            return Ok(m.to_string());
        }
        return Err("no EML 'model' field in the JSON array (rich-leaf/affine laws can't be reloaded yet — re-run with --method enumerate/gumbel/gated/auto)".into());
    }
    if let Some(m) = v.get("model").and_then(serde_json::Value::as_str) {
        return Ok(m.to_string());
    }
    if v.get("root").is_some() {
        return Ok(content.to_string());
    }
    Err("could not find a serialized EML model in the file".into())
}

/// Print a certified range enclosure of `best` over the data's bounding box and a certified root
/// search along x0 (interval Newton/Krawczyk), to stderr.
fn print_certification(best: &phop_core::Solution, ds: &DataSet) {
    let nv = ds.n_vars();
    let domain: Vec<(f64, f64)> = (0..nv)
        .map(|j| {
            let col = ds.x.column(j);
            let lo = col.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (lo, hi)
        })
        .collect();
    let (lo, hi) = best.certified_range(&domain);
    eprintln!("phop: certified range over the data box: f(x) ∈ [{lo:.6}, {hi:.6}]");
    if let Some(&(x0lo, x0hi)) = domain.first() {
        let others: Vec<f64> = domain.iter().skip(1).map(|(a, b)| 0.5 * (a + b)).collect();
        match best.certified_root(0, &others, x0lo, x0hi) {
            Ok(cert) => {
                eprintln!("phop: certified root search on x0 ∈ [{x0lo:.4}, {x0hi:.4}]: {cert:?}")
            }
            Err(e) => eprintln!("phop: certified root search failed: {e}"),
        }
    }
}

/// Write `rendered` to `output` (a file) or stdout.
fn write_out(output: Option<&Path>, rendered: &str) -> Result<(), Box<dyn std::error::Error>> {
    match output {
        Some(path) => {
            let mut file = File::create(path)
                .map_err(|e| format!("failed to create '{}': {e}", path.display()))?;
            file.write_all(rendered.as_bytes())
                .map_err(|e| format!("failed to write '{}': {e}", path.display()))?;
        }
        None => print!("{rendered}"),
    }
    Ok(())
}

/// Pair each [`AnySolution`] with an R² computed uniformly on the loaded data, so the display metric
/// is comparable across engines (the EML core stores only MSE; the affine engine stores its own R²,
/// but recomputing here keeps one consistent definition for every row).
fn rows_with_r2<'a>(sols: &'a [AnySolution], ds: &DataSet) -> Vec<(&'a AnySolution, f64)> {
    let y = ds.y.as_slice().unwrap_or(&[]);
    sols.iter()
        .map(|s| {
            let r2 = s
                .predict(&ds.x)
                .ok()
                .map(|p| r2_of(p.as_slice().unwrap_or(&[]), y))
                .unwrap_or(f64::NAN);
            (s, r2)
        })
        .collect()
}

/// Coefficient of determination of `pred` vs `y`.
fn r2_of(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || y.is_empty() {
        return f64::NAN;
    }
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
    let ss_res: f64 = pred.iter().zip(y).map(|(p, v)| (p - v).powi(2)).sum();
    if ss_tot <= 0.0 {
        return f64::NAN;
    }
    1.0 - ss_res / ss_tot
}

/// Render a heterogeneous (EML + affine) front in the requested format, with a `source` and
/// `symbolic` column. Used by both `--method auto` and `--method rich`.
fn render_any(format: Format, rows: &[(&AnySolution, f64)]) -> String {
    let mut out = String::new();
    match format {
        Format::Table => {
            out.push_str(&format!(
                "{:>4}  {:>7}  {:>10}  {:>12}  {:>8}  {:>4}  expression\n",
                "rank", "source", "complexity", "mse", "r2", "sym"
            ));
            for (i, (s, r2)) in rows.iter().enumerate() {
                out.push_str(&format!(
                    "{:>4}  {:>7}  {:>10}  {:>12.4e}  {:>8.4}  {:>4}  {}\n",
                    i + 1,
                    s.source(),
                    s.complexity(),
                    s.mse(),
                    r2,
                    if s.is_symbolic() { "yes" } else { "no" },
                    s.expr(),
                ));
            }
        }
        Format::Latex => {
            for (i, (s, r2)) in rows.iter().enumerate() {
                out.push_str(&format!(
                    "% rank {} (source={}, complexity={}, mse={:.4e}, r2={:.4}, symbolic={})\n",
                    i + 1,
                    s.source(),
                    s.complexity(),
                    s.mse(),
                    r2,
                    s.is_symbolic()
                ));
                out.push_str(&s.latex());
                out.push('\n');
            }
        }
        Format::Rust => {
            // EML members emit canonical Rust; affine (rich-leaf) members have no native codegen yet,
            // so they are emitted as a commented expression.
            for (i, (s, _)) in rows.iter().enumerate() {
                match s.as_eml() {
                    Some(e) => {
                        out.push_str(&format!(
                            "// rank {} (source=eml, complexity={}, mse={:.4e})\n",
                            i + 1,
                            s.complexity(),
                            s.mse()
                        ));
                        out.push_str(&e.rust_code());
                        out.push('\n');
                    }
                    None => out.push_str(&format!(
                        "// rank {} (source=affine, complexity={}, mse={:.4e}, symbolic={}) — rich-leaf form\n// {}\n",
                        i + 1,
                        s.complexity(),
                        s.mse(),
                        s.is_symbolic(),
                        s.expr()
                    )),
                }
            }
        }
        Format::Json => {
            let items: Vec<serde_json::Value> = rows
                .iter()
                .enumerate()
                .map(|(i, (s, r2))| {
                    let mut obj = serde_json::json!({
                        "rank": i + 1,
                        "source": s.source(),
                        "complexity": s.complexity(),
                        "mse": s.mse(),
                        "r2": r2,
                        "symbolic": s.is_symbolic(),
                        "latex": s.latex(),
                        "pretty": s.expr(),
                    });
                    if let Some(e) = s.as_eml() {
                        obj["rust"] = serde_json::Value::String(e.rust_code());
                        obj["numpy"] = serde_json::Value::String(e.numpy_code());
                        obj["sympy"] = serde_json::Value::String(e.sympy_code());
                        if let Ok(m) = e.to_model_json() {
                            obj["model"] = serde_json::Value::String(m);
                        }
                    }
                    obj
                })
                .collect();
            out.push_str(
                &serde_json::to_string_pretty(&serde_json::Value::Array(items))
                    .unwrap_or_else(|_| "[]".to_string()),
            );
            out.push('\n');
        }
    }
    out
}

/// Print the oxieml-CAS analysis (derivative / antiderivative / Maclaurin / `+∞` limit) of `best`.
fn print_analysis(best: &phop_core::Solution) {
    let a = best.analyze(0, 5);
    eprintln!("phop: analysis of best law (w.r.t. x0):");
    eprintln!("  f           = {}", a.latex);
    eprintln!("  d/dx0       = {}", a.derivative);
    eprintln!(
        "  ∫ f dx0     = {}",
        a.antiderivative.as_deref().unwrap_or("(no closed form)")
    );
    eprintln!(
        "  maclaurin   = {}",
        a.maclaurin.as_deref().unwrap_or("(unavailable)")
    );
    match a.limit_pos_inf {
        Some(v) => eprintln!("  lim x0→+∞   = {v}"),
        None => eprintln!("  lim x0→+∞   = (not finite)"),
    }
}

/// Render the ranked solutions in the requested format, returning a String.
fn render(format: Format, top: &[&phop_core::Solution]) -> String {
    let mut out = String::new();
    match format {
        Format::Table => {
            out.push_str(&format!(
                "{:>4}  {:>10}  {:>12}  expression\n",
                "rank", "complexity", "mse"
            ));
            for (i, s) in top.iter().enumerate() {
                out.push_str(&format!(
                    "{:>4}  {:>10}  {:>12.4e}  {}\n",
                    i + 1,
                    s.complexity,
                    s.mse,
                    s.pretty()
                ));
            }
        }
        Format::Latex => {
            for (i, s) in top.iter().enumerate() {
                out.push_str(&format!(
                    "% rank {} (complexity={}, mse={:.4e})\n",
                    i + 1,
                    s.complexity,
                    s.mse
                ));
                out.push_str(&s.latex());
                out.push('\n');
            }
        }
        Format::Rust => {
            for (i, s) in top.iter().enumerate() {
                out.push_str(&format!(
                    "// rank {} (complexity={}, mse={:.4e})\n",
                    i + 1,
                    s.complexity,
                    s.mse
                ));
                out.push_str(&s.rust_code());
                out.push('\n');
            }
        }
        Format::Json => {
            out.push_str(&render_json(top));
            out.push('\n');
        }
    }
    out
}

/// Emit a JSON array, one object per ranked solution, via `serde_json`.
fn render_json(top: &[&phop_core::Solution]) -> String {
    let items: Vec<serde_json::Value> = top
        .iter()
        .enumerate()
        .map(|(i, s)| {
            serde_json::json!({
                "rank": i + 1,
                "complexity": s.complexity,
                "mse": s.mse,
                "latex": s.latex(),
                "pretty": s.pretty(),
                "rust": s.rust_code(),
                "numpy": s.numpy_code(),
                "sympy": s.sympy_code(),
                "model": s.to_model_json().ok(),
            })
        })
        .collect();
    serde_json::to_string_pretty(&serde_json::Value::Array(items))
        .unwrap_or_else(|_| "[]".to_string())
}
