# SciRS2-WASM Development Guide

Complete guide for building, testing, and deploying SciRS2 WebAssembly bindings.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building](#building)
- [Testing](#testing)
- [Optimization](#optimization)
- [Deployment](#deployment)
- [Browser Compatibility](#browser-compatibility)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

1. **Rust** (1.70+)
   ```bash
   rustup install stable
   rustup target add wasm32-unknown-unknown
   ```

2. **wasm-pack**
   ```bash
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

3. **Node.js** (18+)
   ```bash
   # Using nvm
   nvm install 18
   nvm use 18
   ```

4. **wasm-opt** (optional, for optimization)
   ```bash
   # macOS
   brew install binaryen

   # Ubuntu/Debian
   sudo apt-get install binaryen

   # From source
   git clone https://github.com/WebAssembly/binaryen
   cd binaryen && cmake . && make
   ```

## Building

### Standard Build

```bash
# Build for bundlers (webpack, rollup, parcel)
wasm-pack build --target bundler

# Build for web (ES modules)
wasm-pack build --target web

# Build for Node.js
wasm-pack build --target nodejs

# Build all targets
npm run build:all
```

### Release Build (Optimized)

```bash
# Release build with size optimization
wasm-pack build --release --target bundler

# Or use npm script
npm run build:release
```

### SIMD Build (Advanced)

For browsers with WASM SIMD support:

```bash
# Build with SIMD instructions
RUSTFLAGS='-C target-feature=+simd128' \
  wasm-pack build --target bundler --release -- --features simd

# Or use npm script
npm run build:simd
```

### Custom Features

```bash
# Build with specific features
wasm-pack build --release -- \
  --features "linalg,stats,fft,signal"

# Build with all modules
wasm-pack build --release -- --features all-modules
```

## Testing

### Browser Tests

```bash
# Firefox (headless)
wasm-pack test --headless --firefox

# Chrome (headless)
wasm-pack test --headless --chrome

# Safari (requires macOS)
wasm-pack test --headless --safari

# All browsers
npm test
```

### Node.js Tests

```bash
# Run tests in Node.js
wasm-pack test --node

# Or use npm script
npm run test:node
```

### Interactive Testing

```bash
# Serve the example web app
cd www
npm install
npm start

# Open browser to http://localhost:8080
```

### Manual Testing

```bash
# Build and run Node.js example
wasm-pack build --target nodejs
node examples/node_example.js
```

## Optimization

### Size Optimization

1. **Build with size optimization**
   ```bash
   wasm-pack build --release --target bundler
   ```

2. **Run wasm-opt**
   ```bash
   cd pkg
   wasm-opt -Oz -o scirs2_wasm_bg_opt.wasm scirs2_wasm_bg.wasm

   # Or use npm script
   npm run optimize
   ```

3. **Compare sizes**
   ```bash
   ls -lh pkg/*.wasm
   ```

### Profile Configuration

Edit `Cargo.toml` for different optimization profiles:

```toml
[profile.release]
opt-level = "z"     # "z" for size, "3" for speed
lto = true          # Link-time optimization
codegen-units = 1   # Better optimization
panic = "abort"     # Smaller binary
strip = true        # Remove debug symbols
```

### Feature Gating

Build only what you need:

```bash
# Minimal build (array operations only)
wasm-pack build --release --no-default-features

# With specific features
wasm-pack build --release --features "linalg,stats"
```

## Deployment

### NPM Package

1. **Build for NPM**
   ```bash
   wasm-pack build --release --target bundler
   ```

2. **Publish** (requires npm account)
   ```bash
   cd pkg
   npm publish
   ```

3. **Install in projects**
   ```bash
   npm install scirs2-wasm
   ```

### CDN Deployment

#### Using unpkg

```html
<script type="module">
  import init, * as scirs2 from 'https://unpkg.com/scirs2-wasm/scirs2_wasm.js';

  async function main() {
    await init();
    // Use scirs2...
  }

  main();
</script>
```

#### Using jsDelivr

```html
<script type="module">
  import init, * as scirs2 from 'https://cdn.jsdelivr.net/npm/scirs2-wasm/scirs2_wasm.js';
  // ...
</script>
```

### Self-Hosting

1. **Copy WASM files to your server**
   ```bash
   cp pkg/*.wasm public/
   cp pkg/*.js public/
   cp pkg/*.d.ts public/
   ```

2. **Configure MIME types**
   ```nginx
   # nginx
   location ~ \.wasm$ {
       types { application/wasm wasm; }
   }
   ```

3. **Enable compression**
   ```nginx
   # nginx
   gzip on;
   gzip_types application/wasm;
   ```

## Browser Compatibility

### Standard WASM

- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 16+
- Node.js 8+

### WASM SIMD (simd128)

- Chrome 91+
- Firefox 89+
- Safari 16.4+ (limited support)
- Edge 91+
- Node.js 20+

### Feature Detection

```javascript
// Check WASM support
if (typeof WebAssembly === 'undefined') {
  console.error('WebAssembly is not supported');
}

// Check SIMD support
const simdSupported = scirs2.has_simd_support();
console.log('SIMD support:', simdSupported);
```

## Performance Tips

### 1. Reuse Arrays

```javascript
// ❌ Bad: Creates new arrays each iteration
for (let i = 0; i < 1000; i++) {
  const arr = new scirs2.WasmArray([1, 2, 3]);
  // process arr...
}

// ✅ Good: Reuse array
const arr = new scirs2.WasmArray([1, 2, 3]);
for (let i = 0; i < 1000; i++) {
  // process arr...
}
```

### 2. Batch Operations

```javascript
// ❌ Bad: Multiple small operations
for (let i = 0; i < data.length; i++) {
  result[i] = scirs2.compute(data[i]);
}

// ✅ Good: Single batch operation
const result = scirs2.compute_batch(data);
```

### 3. Use Typed Arrays

```javascript
// ✅ Better performance with TypedArrays
const data = new Float64Array([1, 2, 3, 4]);
const arr = new scirs2.WasmArray(data);
```

### 4. Minimize JS<->WASM Crossing

```javascript
// ❌ Bad: Frequent JS<->WASM calls
for (let i = 0; i < arr.len(); i++) {
  const val = arr.get(i);  // WASM call
  console.log(val);
}

// ✅ Good: Single bulk operation
const data = arr.to_array();  // One WASM call
for (let i = 0; i < data.length; i++) {
  console.log(data[i]);
}
```

### 5. Use Web Workers

```javascript
// worker.js
importScripts('scirs2_wasm.js');

self.onmessage = async (e) => {
  const { data } = e;
  await wasm_bindgen('scirs2_wasm_bg.wasm');

  const arr = new wasm_bindgen.WasmArray(data);
  const result = wasm_bindgen.compute(arr);

  self.postMessage(result.to_array());
};
```

### 6. Enable SIMD

```bash
# Build with SIMD
RUSTFLAGS='-C target-feature=+simd128' \
  wasm-pack build --release --features simd
```

## Troubleshooting

### Build Errors

#### "wasm32-unknown-unknown target not found"

```bash
rustup target add wasm32-unknown-unknown
```

#### "wasm-pack not found"

```bash
cargo install wasm-pack
```

#### "memory access out of bounds"

Increase stack size in `.cargo/config.toml`:

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "link-arg=-zstack-size=2097152"]
```

### Runtime Errors

#### "Module not found" in Node.js

```bash
# Rebuild for Node.js target
wasm-pack build --target nodejs
```

#### "MIME type error" in browser

Configure server to serve `.wasm` files with correct MIME type:

```javascript
// Express.js
app.use(express.static('public', {
  setHeaders: (res, path) => {
    if (path.endsWith('.wasm')) {
      res.set('Content-Type', 'application/wasm');
    }
  }
}));
```

#### Memory issues

```javascript
// Create arrays in smaller chunks
const CHUNK_SIZE = 10000;
for (let i = 0; i < totalSize; i += CHUNK_SIZE) {
  const chunk = data.slice(i, i + CHUNK_SIZE);
  processChunk(chunk);
}
```

### Performance Issues

1. **Enable release mode**
   ```bash
   wasm-pack build --release
   ```

2. **Run wasm-opt**
   ```bash
   wasm-opt -O3 input.wasm -o output.wasm
   ```

3. **Profile with Chrome DevTools**
   - Open DevTools → Performance
   - Record and analyze

4. **Check bundle size**
   ```bash
   ls -lh pkg/*.wasm
   ```

## Advanced Topics

### Custom Memory Management

```rust
// In Rust code
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn allocate_buffer(size: usize) -> Vec<f64> {
    vec![0.0; size]
}
```

### Streaming Compilation

```javascript
// For large WASM files
async function loadWasm() {
  const response = await fetch('scirs2_wasm_bg.wasm');
  const module = await WebAssembly.compileStreaming(response);
  // ...
}
```

### Multi-threading (Experimental)

Requires SharedArrayBuffer and Atomics:

```javascript
// Check support
const supportsThreads = typeof SharedArrayBuffer !== 'undefined';
```

## Resources

- [wasm-bindgen documentation](https://rustwasm.github.io/wasm-bindgen/)
- [wasm-pack documentation](https://rustwasm.github.io/wasm-pack/)
- [WebAssembly MDN](https://developer.mozilla.org/en-US/docs/WebAssembly)
- [Rust WASM Book](https://rustwasm.github.io/docs/book/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/cool-japan/scirs/issues
- Discussions: https://github.com/cool-japan/scirs/discussions

## License

Apache-2.0
