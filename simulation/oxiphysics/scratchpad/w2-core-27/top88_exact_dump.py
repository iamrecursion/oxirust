import sys
sys.path.insert(0, ".")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from top88_ref import top88

def run(exact):
    nelx, nely, volfrac, penal, rmin = 60, 20, 0.5, 3.0, 1.5
    if not exact:
        x, xp, hist, KE = top88(nelx, nely, volfrac, penal, rmin, 1, verbose=False)
        return xp, [h[1] for h in hist]
    E0, Emin, nu = 1.0, 1e-9, 0.3
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], float)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], float)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], float)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], float)
    top = np.vstack([np.hstack([A11, A12]), np.hstack([A12.T, A11])])
    bot = np.vstack([np.hstack([B11, B12]), np.hstack([B12.T, B11])])
    KE = 1.0 / (1 - nu**2) / 24.0 * (top + nu * bot)
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape((1 + nely, 1 + nelx), order="F")
    edofVec = (2 * nodenrs[:-1, :-1] + 1).reshape((nelx * nely, 1), order="F")
    offs = np.array([0, 1] + [2 * nely + 2, 2 * nely + 3, 2 * nely + 0, 2 * nely + 1] + [-2, -1])
    edofMat = edofVec + offs[None, :]
    iK = np.kron(edofMat, np.ones((8, 1))).T.reshape(64 * nelx * nely, order="F").astype(int) - 1
    jK = np.kron(edofMat, np.ones((1, 8))).T.reshape(64 * nelx * nely, order="F").astype(int) - 1
    ndof = 2 * (nely + 1) * (nelx + 1)
    F = np.zeros(ndof); F[1] = -1.0
    U = np.zeros(ndof)
    fixeddofs = np.union1d(np.arange(0, 2 * (nely + 1), 2), [ndof - 1])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    cr = int(np.ceil(rmin)) - 1
    iH, jH, sH = [], [], []
    for i1 in range(1, nelx + 1):
        for j1 in range(1, nely + 1):
            e1 = (i1 - 1) * nely + j1
            for i2 in range(max(i1 - cr, 1), min(i1 + cr, nelx) + 1):
                for j2 in range(max(j1 - cr, 1), min(j1 + cr, nely) + 1):
                    e2 = (i2 - 1) * nely + j2
                    iH.append(e1 - 1); jH.append(e2 - 1)
                    sH.append(max(0.0, rmin - np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)))
    H = sp.csr_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely))
    Hs = np.asarray(H.sum(axis=1)).ravel()
    x = np.full(nelx * nely, volfrac)
    change = 1.0
    hist = []
    ed0 = edofMat - 1
    while change > 0.01 and len(hist) < 500:
        sK = (KE.reshape(-1, 1) * (Emin + x[None, :] ** penal * (E0 - Emin))).reshape(-1, order="F")
        K = sp.csc_matrix((sK, (iK, jK)), shape=(ndof, ndof)); K = (K + K.T) / 2
        U[:] = 0.0
        U[freedofs] = spla.spsolve(K[freedofs][:, freedofs].tocsc(), F[freedofs])
        Ue = U[ed0]
        ce = np.sum((Ue @ KE) * Ue, axis=1)
        c = np.sum((Emin + x**penal * (E0 - Emin)) * ce)
        dc = -penal * (E0 - Emin) * x ** (penal - 1) * ce
        dc = (H @ (x * dc)) / Hs / np.maximum(1e-3, x)
        gain = np.maximum(-dc, 0)
        move = 0.2
        lo, hi = np.log(gain.max()) - 60 * np.log(10), np.log(gain.max()) + 30 * np.log(10)
        def dens(lam):
            lower = np.maximum(x - move, 0); upper = np.minimum(x + move, 1)
            return np.minimum(np.maximum(x * np.sqrt(gain / lam), lower), upper)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi: break
            if dens(np.exp(mid)).mean() > volfrac: lo = mid
            else: hi = mid
        xnew = dens(np.exp(0.5 * (lo + hi)))
        change = np.max(np.abs(xnew - x))
        hist.append(c)
        x = xnew
    return x.reshape((nely, nelx), order="F"), hist

xa, ha = run(False)
xb, hb = run(True)
xa = xa[::-1, :]; xb = xb[::-1, :]
print("max |density difference| over all elements:", np.abs(xa - xb).max(), "mean", np.abs(xa - xb).mean())
for (r, c) in [(0, 0), (0, 59), (19, 0), (19, 59), (10, 30), (5, 5), (15, 10), (10, 50), (2, 40), (18, 3)]:
    print((r, c), round(float(xa[r, c]), 5), round(float(xb[r, c]), 5))
print("final compliance (last history entry):", ha[-1], hb[-1], len(ha), len(hb))
