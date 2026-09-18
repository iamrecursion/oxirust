"""Generate identical (X, y) CSVs for the real Feynman-100 equations (M6).

Reads the canonical FeynmanEquations.csv (Filename, Formula, # variables, and per-variable
[low, high] ranges — from the AI-Feynman / Feynman Symbolic Regression Database) and, for each
equation, samples `N` uniform points per variable in its range, evaluates the formula, and writes
`<out_dir>/<id>.csv` with a header `v1,...,vk,y` (last column = target). Both the phop `feynman` bin
and `pysr_feynman.py` consume these same files, so the comparison is apples-to-apples.

Usage:
    python3 gen_feynman.py FeynmanEquations.csv <out_dir> [N]
"""
import csv, json, sys, re
import numpy as np

NS = {
    "exp": np.exp, "sqrt": np.sqrt, "ln": np.log, "log": np.log,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "abs": np.abs, "pi": np.pi,
}

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "FeynmanEquations.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "feynman_data"
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    import os
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    manifest = []
    written = 0
    for r in rows:
        formula = (r.get("Formula") or "").strip()
        nvars = (r.get("# variables") or "").strip()
        if not formula or not nvars.isdigit():
            continue
        nvars = int(nvars)
        ident = re.sub(r"[^A-Za-z0-9]+", "_", (r.get("Filename") or f"eq{written}").strip())

        # Collect (name, low, high) per variable.
        vars_ = []
        ok = True
        for i in range(1, nvars + 1):
            name = (r.get(f"v{i}_name") or "").strip()
            lo, hi = r.get(f"v{i}_low"), r.get(f"v{i}_high")
            try:
                lo, hi = float(lo), float(hi)
            except (TypeError, ValueError):
                ok = False
                break
            if not name:
                ok = False
                break
            vars_.append((name, lo, hi))
        if not ok or not vars_:
            continue

        env = dict(NS)
        cols = []
        for name, lo, hi in vars_:
            x = rng.uniform(lo, hi, size=N)
            env[name] = x
            cols.append(x)
        try:
            y = eval(formula, {"__builtins__": {}}, env)  # controlled, local file
            y = np.asarray(y, dtype=float) * np.ones(N)
        except Exception as e:
            print(f"skip {ident}: eval failed ({e})", file=sys.stderr)
            continue

        data = np.column_stack(cols + [y])
        finite = np.all(np.isfinite(data), axis=1)
        data = data[finite]
        if data.shape[0] < 10:
            print(f"skip {ident}: too few finite rows", file=sys.stderr)
            continue

        header = [n for n, _, _ in vars_] + ["y"]
        path = f"{out_dir}/{ident}.csv"
        np.savetxt(path, data, delimiter=",", header=",".join(header), comments="", fmt="%.10g")
        manifest.append({"id": ident, "formula": formula, "nvars": nvars,
                         "vars": [n for n, _, _ in vars_], "rows": int(data.shape[0])})
        written += 1

    json.dump(manifest, open(f"{out_dir}/manifest.json", "w"), indent=1)
    print(f"wrote {written} equation CSVs to {out_dir}/ (N={N})")

if __name__ == "__main__":
    main()
