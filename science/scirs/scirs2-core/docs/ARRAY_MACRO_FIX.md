# Array Macro Re-export Fix

## Problem Solved

Previously, projects using `scirs2-core` struggled with accessing the `array!` macro for creating ndarray arrays. Users had to import it from `scirs2_autograd::ndarray::array`, which was inconvenient and created unnecessary dependencies.

## Solution

The `array!` macro is now re-exported directly from `scirs2-core` for convenient access.

## Usage

### Before (Inconvenient)
```rust
// Users had to remember this specific import path
use scirs2_autograd::ndarray::array;

let matrix = array![[1, 2, 3], [4, 5, 6]];
```

### After (Convenient) ✅
```rust
// Now users can import directly from scirs2-core
use scirs2_core::array;

let matrix = array![[1, 2, 3], [4, 5, 6]];
```

## Alternative Import Paths

The `array!` macro is now available through multiple convenient paths:

1. **Direct from scirs2-core** (recommended):
   ```rust
   use scirs2_core::array;
   ```

2. **From ndarray_ext module**:
   ```rust
   use scirs2_core::ndarray_ext::array;
   ```

3. **Original path** (still works):
   ```rust
   use scirs2_autograd::ndarray::array;
   ```

## Implementation Details

- Added re-export in `scirs2-core/src/ndarray_ext/mod.rs`
- Added top-level re-export in `scirs2-core/src/lib.rs`
- Comprehensive documentation and examples added
- Full backward compatibility maintained

## Examples

See `examples/array_macro_usage.rs` for comprehensive usage examples.

## Testing

```bash
# Run the example to test the array! macro
cargo run --example array_macro_usage

# Run tests
cargo test array_macro_test
```

This fix addresses a common pain point reported by projects using scirs2-core in production environments.