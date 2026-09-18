#!/usr/bin/env python3
"""Faithful NumPy/SciPy transcription of top88.m (Andreassen et al. 2011).

Used only to produce reference numbers for the Rust regression tests; the
transcription follows the published MATLAB code line by line (MBB half beam,
E0 = 1, Emin = 1e-9, nu = 0.3, linear-scale lambda bisection with the 1e-3
relative tolerance, change > 0.01 stopping rule).
"""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def top88(nelx, nely, volfrac, penal, rmin, ft, verbose=True, max_loop=2000):
    E0, Emin, nu = 1.0, 1e-9, 0.3
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], float)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], float)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], float)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], float)
    top = np.vstack([np.hstack([A11, A12]), np.hstack([A12.T, A11])])
    bot = np.vstack([np.hstack([B11, B12]), np.hstack([B12.T, B11])])
    KE = 1.0 / (1 - nu**2) / 24.0 * (top + nu * bot)

    # node numbers: column-major (1+nely) x (1+nelx), 1-based like MATLAB
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape((1 + nely, 1 + nelx), order="F")
    edofVec = (2 * nodenrs[:-1, :-1] + 1).reshape((nelx * nely, 1), order="F")
    offs = np.array([0, 1] + [2 * nely + 2, 2 * nely + 3, 2 * nely + 0, 2 * nely + 1] + [-2, -1])
    edofMat = edofVec + offs[None, :]  # 1-based dofs
    iK = np.kron(edofMat, np.ones((8, 1))).T.reshape(64 * nelx * nely, order="F").astype(int) - 1
    jK = np.kron(edofMat, np.ones((1, 8))).T.reshape(64 * nelx * nely, order="F").astype(int) - 1
    ndof = 2 * (nely + 1) * (nelx + 1)
    F = np.zeros(ndof)
    F[1] = -1.0  # sparse(2,1,-1,...)
    U = np.zeros(ndof)
    fixeddofs = np.union1d(np.arange(0, 2 * (nely + 1), 2), [ndof - 1])
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    # filter
    cr = int(np.ceil(rmin)) - 1
    iH, jH, sH = [], [], []
    for i1 in range(1, nelx + 1):
        for j1 in range(1, nely + 1):
            e1 = (i1 - 1) * nely + j1
            for i2 in range(max(i1 - cr, 1), min(i1 + cr, nelx) + 1):
                for j2 in range(max(j1 - cr, 1), min(j1 + cr, nely) + 1):
                    e2 = (i2 - 1) * nely + j2
                    iH.append(e1 - 1)
                    jH.append(e2 - 1)
                    sH.append(max(0.0, rmin - np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)))
    H = sp.csr_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely))
    Hs = np.asarray(H.sum(axis=1)).ravel()

    x = np.full(nelx * nely, volfrac)  # column-major flattening == MATLAB x(:)
    xPhys = x.copy()
    loop = 0
    change = 1.0
    history = []
    ed0 = edofMat - 1
    while change > 0.01 and loop < max_loop:
        loop += 1
        sK = (KE.reshape(-1, 1) * (Emin + xPhys[None, :] ** penal * (E0 - Emin))).reshape(-1, order="F")
        K = sp.csc_matrix((sK, (iK, jK)), shape=(ndof, ndof))
        K = (K + K.T) / 2
        Kff = K[freedofs][:, freedofs].tocsc()
        U[:] = 0.0
        U[freedofs] = spla.spsolve(Kff, F[freedofs])
        Ue = U[ed0]  # (nel, 8)
        ce = np.sum((Ue @ KE) * Ue, axis=1)
        c = np.sum((Emin + xPhys**penal * (E0 - Emin)) * ce)
        dc = -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce
        dv = np.ones(nelx * nely)
        if ft == 1:
            dc = (H @ (x * dc)) / Hs / np.maximum(1e-3, x)
        elif ft == 2:
            dc = H @ (dc / Hs)
            dv = H @ (dv / Hs)
        l1, l2, move = 0.0, 1e9, 0.2
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0, np.maximum(x - move, np.minimum(1, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
            if ft == 1:
                xPhys = xnew.copy()
            elif ft == 2:
                xPhys = (H @ xnew) / Hs
            if xPhys.sum() > volfrac * nelx * nely:
                l1 = lmid
            else:
                l2 = lmid
        change = np.max(np.abs(xnew - x))
        x = xnew
        history.append((loop, c, xPhys.mean(), change))
        if verbose:
            print(f" It.:{loop:5d} Obj.:{c:11.6f} Vol.:{xPhys.mean():7.3f} ch.:{change:7.3f}")
    return x.reshape((nely, nelx), order="F"), xPhys.reshape((nely, nelx), order="F"), history, KE


if __name__ == "__main__":
    nelx = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    nely = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    volfrac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    penal = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    rmin = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5
    ft = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    x, xp, hist, KE = top88(nelx, nely, volfrac, penal, rmin, ft)
    np.set_printoptions(linewidth=200, precision=17)
    print("final iterations:", len(hist), "final compliance:", repr(hist[-1][1]))
    print("KE[0,0]=", KE[0, 0], "KE[0,1]=", KE[0, 1])
