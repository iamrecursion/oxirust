import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def harmonic(a, b):
    return np.where(a == b, a, 2 * a * b / (a + b))

def homogenize(phase_map, k0, k1):
    pm = np.array(phase_map)
    ny, nx = pm.shape
    k = np.where(pm == 0, k0, k1).astype(float)
    n = nx * ny
    idx = np.arange(n).reshape(ny, nx)
    faces = []  # (low, high, k, axis)
    ke = harmonic(k, np.roll(k, -1, axis=1))  # east neighbor (x)
    kn = harmonic(k, np.roll(k, -1, axis=0))  # north neighbor (y)
    low = idx.ravel()
    east = np.roll(idx, -1, axis=1).ravel()
    north = np.roll(idx, -1, axis=0).ravel()
    F = [(low, east, ke.ravel(), 0), (low, north, kn.ravel(), 1)]
    chi = []
    for j in (0, 1):
        rows, cols, vals = [], [], []
        b = np.zeros(n)
        for (lo, hi, kf, ax) in F:
            m = lo != hi
            lo_, hi_, kf_ = lo[m], hi[m], kf[m]
            rows += [lo_, hi_, lo_, hi_]
            cols += [lo_, hi_, hi_, lo_]
            vals += [kf_, kf_, -kf_, -kf_]
            if ax == j:
                np.add.at(b, lo_, kf_)
                np.add.at(b, hi_, -kf_)
        L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        x = np.zeros(n)
        if np.linalg.norm(b) > 1e-13 * np.abs(b).sum() and n > 1:
            Lr = L[1:, 1:]
            br = b[1:]
            sol = spla.spsolve(Lr.tocsc(), br)
            x[1:] = sol
        chi.append(x)
    T = np.zeros((2, 2))
    for (lo, hi, kf, ax) in F:
        for j in (0, 1):
            T[ax, j] += np.sum(kf * ((1.0 if ax == j else 0.0) + chi[j][hi] - chi[j][lo]))
    T /= n
    kxy = 0.5 * (T[0, 1] + T[1, 0])
    return np.array([T[0, 0], kxy, kxy, T[1, 1]])

def block_checker(blocks, side):
    n = blocks * side
    return [[((r // side) + (c // side)) % 2 for c in range(n)] for r in range(n)]

if __name__ == "__main__":
    k0, k1 = 1.0, 4.0
    exact = (k0 * k1) ** 0.5
    for side in (4, 8, 16, 32):
        K = homogenize(block_checker(2, side), k0, k1)
        print(side, K, abs(K[0] - exact) / exact)
    # one-pixel checkerboard
    pm = [[(r + c) % 2 for c in range(4)] for r in range(4)]
    print("1px", homogenize(pm, 1.0, 4.0), 2 * 1.0 * 4.0 / 5.0)
    # layers along x (phase depends on row)
    pm = [[1 if r >= 3 else 0 for c in range(8)] for r in range(8)]
    K = homogenize(pm, 1.0, 10.0)
    f0, f1 = 3 / 8, 5 / 8
    print("layers", K, f0 * 1 + f1 * 10, 1 / (f0 / 1 + f1 / 10))
    # square inclusion
    n = 32
    pm = [[1 if (8 <= r < 24 and 8 <= c < 24) else 0 for c in range(n)] for r in range(n)]
    K = homogenize(pm, 1.0, 10.0)
    f1 = 0.25; f0 = 0.75; k0 = 1.0; k1 = 10.0
    voigt = f0 * k0 + f1 * k1
    reuss = 1 / (f0 / k0 + f1 / k1)
    hs_lo = k0 + f1 / (1 / (k1 - k0) + f0 / (2 * k0))
    hs_hi = k1 + f0 / (1 / (k0 - k1) + f1 / (2 * k1))
    print("square", K, "voigt", voigt, "reuss", reuss, "HS", hs_lo, hs_hi)
    # anisotropic rectangle
    pm = [[1 if (6 <= r < 10 and 1 <= c < 15) else 0 for c in range(16)] for r in range(16)]
    K = homogenize(pm, 1.0, 20.0)
    print("rect", K)
