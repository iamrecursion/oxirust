import numpy as np
from scipy.spatial.distance import pdist

def corr_dim_pairs(data, n_radii, window=1):
    n = len(data)
    d = pdist(data)
    # all pairs (window 1)
    d = d[d >= 0]
    dmin = d[d > 0].min(); dmax = d.max()
    r_min = 1.1*dmin; r_max = 0.9*dmax
    radii = np.array([r_min * (r_max/r_min)**(k / max(n_radii-1, 1)) for k in range(n_radii)])
    ds = np.sort(d)
    counts = np.searchsorted(ds, radii, side='right')
    C = counts / len(ds)
    ok = C > 0
    return np.log(radii[ok]), np.log(C[ok])

def henon(n):
    a, b = 1.4, 0.3
    x, y = 0.1, 0.1
    for _ in range(200):
        x, y = 1 - a*x*x + y, b*x
    pts = []
    for _ in range(n):
        x, y = 1 - a*x*x + y, b*x
        pts.append([x, y])
    return np.array(pts)

lr, lc = corr_dim_pairs(henon(2500), 40)
sel = (lr >= -4.5) & (lr <= -2.0)
print("henon: n sel", sel.sum(), "slope", np.polyfit(lr[sel], lc[sel], 1)[0])
for n_pts in (1500, 2500):
    lr, lc = corr_dim_pairs(henon(n_pts), 40)
    for lo, hi in [(-4.5, -2.0), (-4.0, -2.0), (-5, -2.5)]:
        sel = (lr >= lo) & (lr <= hi)
        print(n_pts, (lo, hi), sel.sum(), np.polyfit(lr[sel], lc[sel], 1)[0])

# Kronecker
a, b = 0.754877666246693, 0.569840290998053
n = 2000
k = np.arange(n) + 1.0
sq = np.stack([(0.5 + a*k) % 1.0, (0.5 + b*k) % 1.0], 1)
lr, lc = corr_dim_pairs(sq, 60)
sel = (lr >= -3.5) & (lr <= -2.3)
print("square: n sel", sel.sum(), "slope", np.polyfit(lr[sel], lc[sel], 1)[0])
g = 0.618033988749895
n = 1500
k = np.arange(n) + 1.0
ln = ((0.5 + g*k) % 1.0).reshape(-1, 1)
lr, lc = corr_dim_pairs(ln, 60)
sel = (lr >= -3.5) & (lr <= -2.3)
print("line: n sel", sel.sum(), "slope", np.polyfit(lr[sel], lc[sel], 1)[0])
