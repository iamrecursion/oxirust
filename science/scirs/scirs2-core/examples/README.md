# Example Code

This directory contains example code for demonstrating the functionality of the scirs2-core library.

## Important Notes

- Some examples may contain experimental code that demonstrates API usage but may not compile with the latest version of the library.
- Examples are intended for educational purposes and may contain intentional errors or warnings for demonstration.
- When building the project with `cargo test` or `cargo build`, it's recommended to use the `--lib` flag to focus on the library code and ignore examples.

## Running Examples

To run specific examples:

```bash
cargo run --example <example_name> --features <required_features>
```

See the crate's Cargo.toml for the list of required features for each example.

## Testing Strategy

Due to the evolving nature of the API, many examples may have warnings or even compilation errors. This is expected and does not affect the functionality of the library itself. 

Our testing strategy prioritizes:
1. Zero warnings in library code
2. Passing tests in core functionality 
3. Documentation of known issues in examples

For clean builds, use:

```bash
# Build only the library code
cargo build -p scirs2-core --lib

# Run only the library tests
cargo test -p scirs2-core --lib
```