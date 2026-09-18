import numpy as np
import math

def rp_matrix(sig, thr):
    s = np.asarray(sig)
    return np.abs(s[:, None] - s[None, :]) <= thr

def diag_lengths(R, min_line):
    n = R.shape[0]
    min_line = max(min_line, 1)
    lengths = []
    for k in range(-(n-1), n):
        if k == 0: continue
        d = np.diagonal(R, offset=k)
        run = 0
        for v in d:
            if v: run += 1
            else:
                if run >= min_line: lengths.append(run)
                run = 0
        if run >= min_line: lengths.append(run)
    return lengths

def vert_lengths(R, min_line):
    n = R.shape[0]
    min_line = max(min_line, 1)
    lengths = []
    for j in range(n):
        run = 0
        for i in range(n):
            if R[i, j]: run += 1
            else:
                if run >= min_line: lengths.append(run)
                run = 0
        if run >= min_line: lengths.append(run)
    return lengths

def rqa(R, min_line):
    n = R.shape[0]
    total = int(R.sum()); ident = int(np.trace(R))
    L = diag_lengths(R, min_line)
    det = (sum(L)/(total-ident)) if total-ident > 0 else 0.0
    V = vert_lengths(R, min_line)
    lam = sum(V)/total if total else 0.0
    return dict(rr=total/(n*n), det=det, lmax=max(L) if L else 0, lmean=(sum(L)/len(L)) if L else 0.0, lam=lam, nlines=len(L))

# sine test
n = 200
sig = [math.sin(2*math.pi*i/25.0) for i in range(n)]
R = rp_matrix(sig, 0.1)
for ml in (2,):
    print("sine n=200 period 25 thr 0.1 min_line", ml, rqa(R, ml))

# noise test (u64 LCG, seed 7)
def lcg_noise(n, seed):
    state = seed; out = []
    M = (1 << 64) - 1
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & M
        out.append((state >> 11) / float(1 << 53))
    return out
noise = lcg_noise(300, 7)
R = rp_matrix(noise, 0.15)
for ml in (2, 5):
    r = rqa(R, ml); print("noise seed 7 thr 0.15 min_line", ml, r)

# default theiler window for period-40 sine (400 samples), noise 500 seed 99
def first_zero(x):
    x = np.asarray(x, float); n = len(x)
    c = x - x.mean()
    var = (c*c).sum()
    if var <= 0: return None
    for lag in range(1, n//2):
        if (c[:-lag] * c[lag:]).sum() <= 0: return lag
    return None
s40 = [math.sin(2*math.pi*i/40.0) for i in range(400)]
print("period-40 sine window:", first_zero(s40))
print("noise 500 seed 99 window:", first_zero(lcg_noise(500, 99)))

print("---- structured plot checks")
def plot(n, cells):
    R = np.zeros((n, n), bool)
    for i, j in cells: R[i, j] = True
    return R
cells = [(i, i) for i in range(8)]
for (i, j) in [(0,2),(1,3),(2,4),(0,4),(1,5),(0,6)]:
    cells += [(i, j), (j, i)]
R = plot(8, cells)
for ml in (1, 2, 3, 4):
    print("min_line", ml, rqa(R, ml))
L2 = diag_lengths(R, 2)
from collections import Counter
cnt = Counter(L2); nl = len(L2)
ent = -sum((c/nl)*math.log(c/nl) for c in cnt.values())
print("entropy min_line=2:", ent, "expected ln2 =", math.log(2))
# small LAM example
cells = [(i, i) for i in range(4)] + [(1, 0), (2, 0)]
R = plot(4, cells)
print("lam example", rqa(R, 2), "vert", vert_lengths(R, 2))
R = np.ones((8, 8), bool)
print("constant", rqa(R, 2))
cells = [(0,1),(1,2),(2,3),(4,0)]
R = plot(5, cells)
print("no identity:", rqa(R, 2))
