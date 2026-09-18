# phop-cli

The `phop` command-line interface for
[phop](https://github.com/cool-japan/phop) — the first differentiable
symbolic-discovery engine: it learns expression *topology* and *numeric
parameters* by gradient descent over a tensorized population of homogeneous EML
trees (`eml(x, y) = exp(x) − ln(y)`, Odrzywołek 2026), end-to-end in pure Rust.

## Install

```bash
cargo install phop-cli
```

## Usage

```bash
phop discover data.csv --method auto --format table
```

The `auto` method runs the full meta-ensemble (EML-tree searches plus the
rich-leaf affine engine) and prints a merged Pareto front with `source` and
`symbolic` columns.

- Methods: `enumerate | gumbel | gated | gated-warm | auto | rich`
- Formats: `table | latex | rust | json`
- Flags: `--target`, `--features`, `--top-k`, `--analyze`, `--gpu cuda`

Part of the [phop](https://github.com/cool-japan/phop) project.

## License

Apache-2.0
