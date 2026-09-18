# Error Handling Best Practices

This document provides guidelines for using the error handling system in scirs2.

## Overview

The scirs2-core crate provides a comprehensive error handling system that supports:

- Rich error contexts with location information
- Error chaining for preserving error history
- Convenient error creation macros
- Type conversions between different error types

## Core Error Type

The `CoreError` enum is the base error type for scirs2-core. It includes variants for common error categories:

```rust
pub enum CoreError {
    /// I/O error
    IoError(ErrorContext),
    /// Value error
    ValueError(ErrorContext),
    /// Index error
    IndexError(ErrorContext),
    /// Shape error
    ShapeError(ErrorContext),
    /// Type error
    TypeError(ErrorContext),
    /// Runtime error
    RuntimeError(ErrorContext),
    /// Not implemented error
    NotImplementedError(ErrorContext),
}
```

Each variant contains an `ErrorContext` that provides additional information about the error.

## Error Context

The `ErrorContext` struct provides detailed information about an error:

```rust
pub struct ErrorContext {
    /// Error message
    pub message: String,
    /// Location where the error occurred
    pub location: Option<ErrorLocation>,
    /// Cause of the error
    pub cause: Option<Box<CoreError>>,
}
```

An `ErrorContext` can include:
- A descriptive message
- The file and line where the error occurred
- An optional cause (for error chaining)

## Creating Errors

### Using Macros

For simple errors, use the provided macros:

```rust
// Create a value error
let err = value_err!("Invalid value: {}", value);

// Create an error with location information
let err = value_err_loc!("Invalid value: {}", value);

// Create an error with a cause
let err = value_err_with_cause!("Operation failed", cause);
```

### Manually Creating Errors

For more control, create errors manually:

```rust
let context = ErrorContext::new("Invalid value")
    .with_location(ErrorLocation::new(file!(), line!()))
    .with_cause(some_other_error);

let error = CoreError::ValueError(context);
```

## Error Chaining

Error chaining allows preserving the original cause of an error:

```rust
fn process_data() -> Result<(), CoreError> {
    let result = load_data().map_err(|err| {
        value_err_with_cause!("Failed to process data", err)
    })?;
    
    // Continue processing...
    Ok(())
}
```

This creates a chain of errors that can be inspected to understand the full error path.

## Converting Between Error Types

Each module in scirs2 defines its own error type, but all module errors can be converted to `CoreError`:

```rust
// Convert an I/O error to CoreError
let file = File::open("data.txt").map_err(|err| {
    CoreError::from_io_error(err, "Failed to open data file")
})?;
```

For custom error types, implement the `From` trait:

```rust
impl From<MyModuleError> for CoreError {
    fn from(err: MyModuleError) -> Self {
        CoreError::RuntimeError(
            ErrorContext::new(format!("Module error: {}", err))
        )
    }
}
```

## Error Handling Patterns

### Result Propagation with Context

Use the `?` operator with error mapping to propagate errors with additional context:

```rust
fn process_file(path: &str) -> CoreResult<Data> {
    let file = File::open(path).map_err(|err| {
        CoreError::from_io_error(err, &format!("Failed to open file: {}", path))
    })?;
    
    let data = parse_file(file).map_err(|err| {
        runtime_err_with_cause!(&format!("Failed to parse file: {}", path), err)
    })?;
    
    Ok(data)
}
```

### Validating Inputs

Use validation functions from the `validation` module to check inputs:

```rust
use scirs2_core::validation::{check_in_bounds, check_positive, check_shape};

fn process_data(data: &Array2<f64>, alpha: f64) -> CoreResult<Array2<f64>> {
    // Validate inputs
    check_positive(alpha, "alpha")?;
    check_shape(&data, &[10, 10], "data")?;
    
    // Process data...
    Ok(result)
}
```

### When to Use Error Context

- Add detailed error messages that explain what went wrong
- Include parameter values that caused the error
- Provide suggestions for fixing the error when possible
- Add location information in library code, but not in application code

### Handling Errors

When handling errors, extract useful information:

```rust
match result {
    Ok(value) => {
        // Process value
    }
    Err(err) => {
        // Get the error message
        let message = err.to_string();
        
        // Check the error type
        match err {
            CoreError::ValueError(_) => {
                // Handle value errors
            }
            CoreError::IoError(context) => {
                // Handle I/O errors
                if let Some(cause) = context.cause {
                    // Handle the cause
                }
            }
            _ => {
                // Handle other errors
            }
        }
    }
}
```

## Best Practices

1. **Be Specific**: Use the most specific error variant that applies
2. **Add Context**: Include enough information to understand and fix the error
3. **Chain Errors**: Preserve the original cause when wrapping errors
4. **Location Information**: Include file and line information in library code
5. **Validate Early**: Check inputs at function boundaries
6. **Handle Gracefully**: Provide helpful error messages to users

By following these guidelines, you can create a robust error handling system that helps diagnose and fix issues quickly.