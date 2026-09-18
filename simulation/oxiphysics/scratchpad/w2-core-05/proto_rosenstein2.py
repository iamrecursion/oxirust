import numpy as np, time
from scipy.spatial import cKDTree
from proto_rosenstein import lorenz_orbit, slope

def rosenstein_kd(data, horizon, window, restricted=True, K=400):
    n = len(data)
    steps = min(horizon, n//2)
    min_sep = max(window, 1)
    followable = n - steps + 1 if restricted else n
    pts = data[:followable]
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=min(K, followable))
    nn = np.full(followable, -1)
    ii = np.arange(followable)
    for col in range(idx.shape[1]):
        cand = idx[:, col]
        ok = (np.abs(cand - ii) >= min_sep) & (nn < 0)
        nn[ok] = cand[ok]
    valid = nn >= 0
    i = ii[valid]; j = nn[valid]
    out = np.zeros(steps)
    for t in range(steps):
        if not restricted:
            sel = (i + t < n) & (j + t < n)
            ii_t, jj_t = i[sel], j[sel]
        else:
            ii_t, jj_t = i, j
        d = np.sqrt(((data[ii_t+t] - data[jj_t+t])**2).sum(-1))
        out[t] = np.log(d[d > 1e-300]).mean()
    return out

if __name__ == "__main__":
    dt = 0.01
    for n in (5000, 20000):
        data = lorenz_orbit(n, dt)
        for window in (100, 1):
            c = rosenstein_kd(data, 300, window, True)
            print(f"N={n} window={window}")
            for a, b in [(0, 10), (0, 30), (0, 50), (10, 50), (10, 100), (20, 100), (30, 150), (50, 200), (100, 250)]:
                print(f"   slope[{a},{b}]/dt = {slope(c, a, b)/dt:.3f}", end="")
            print()
            print("   curve samples", np.round(c[::25], 3))
