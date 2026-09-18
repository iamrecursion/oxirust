# phop (Python bindings)

Python bindings for [phop](https://github.com/cool-japan/phop) — differentiable
symbolic discovery on the EML operator.

```python
import numpy as np
import phop

x = np.array([[1.0], [2.0], [3.0]])
y = np.array([2.0, 4.0, 6.0])

disco = phop.Discoverer(population=256, max_depth=3, max_epochs=1000, seed=0)
result = disco.fit(x, y)
print(result.top_latex(5))
```

## Build

```bash
maturin develop -m crates/phop-py/Cargo.toml
```
