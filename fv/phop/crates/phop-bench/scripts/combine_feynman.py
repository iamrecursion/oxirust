"""Combine the phop and PySR Feynman-100 results into one comparison + summary (M6).

Usage:
    python3 combine_feynman.py phop_feynman.csv pysr_feynman.json [manifest.json]
"""
import csv, json, sys, statistics as st

def main():
    phop_csv = sys.argv[1] if len(sys.argv) > 1 else "phop_feynman.csv"
    pysr_json = sys.argv[2] if len(sys.argv) > 2 else "pysr_feynman.json"

    phop = {}
    for r in csv.DictReader(open(phop_csv)):
        phop[r["name"]] = r
    pysr = json.load(open(pysr_json))

    names = sorted(set(phop) | set(pysr))
    print(f"{'equation':24s} {'phop_r2':>8} {'phop_ms':>8} {'phop':>5}   {'pysr_r2':>8} {'pysr_s':>7} {'pysr':>5}")
    print("-" * 78)
    p_rec = q_rec = 0
    p_ms, q_s = [], []
    both_p_ms, both_q_s = [], []
    for n in names:
        pr = phop.get(n, {})
        qr = pysr.get(n, {})
        p_r2 = float(pr.get("r2", "nan") or "nan")
        p_ms_v = float(pr.get("ms", "nan") or "nan")
        p_ok = pr.get("recovered", "") == "true"
        q_r2 = qr.get("r2", float("nan"))
        q_s_v = qr.get("time_s", float("nan"))
        q_ok = bool(qr.get("recovered", False))
        p_rec += int(p_ok); q_rec += int(q_ok)
        if p_ms_v == p_ms_v: p_ms.append(p_ms_v)
        if q_s_v == q_s_v: q_s.append(q_s_v)
        if p_ms_v == p_ms_v and q_s_v == q_s_v:
            both_p_ms.append(p_ms_v); both_q_s.append(q_s_v)
        print(f"{n:24s} {p_r2:8.4f} {p_ms_v:8.0f} {('YES' if p_ok else '-'):>5}   "
              f"{q_r2:8.4f} {q_s_v:7.1f} {('YES' if q_ok else '-'):>5}")

    print("-" * 78)
    nt = len(names)
    print(f"\nRecovered (R2 >= 0.999):  phop {p_rec}/{nt}   PySR {q_rec}/{nt}")
    if p_ms:
        print(f"phop time/eq:  median {st.median(p_ms):.0f} ms,  total {sum(p_ms)/1000:.1f} s")
    if q_s:
        print(f"PySR time/eq:  median {st.median(q_s):.1f} s,   total {sum(q_s):.0f} s")
    if both_p_ms:
        spd = st.median([ (q*1000.0)/p for p, q in zip(both_p_ms, both_q_s) if p > 0 ])
        print(f"Median per-eq speedup (PySR_ms / phop_ms): ~{spd:.0f}x  (phop faster)")

if __name__ == "__main__":
    main()
