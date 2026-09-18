import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ls_proto import hj_step, reinit_step

NU = 0.3
def ke_gauss(nu):
    scale = 1 - nu * nu
    D = np.array([[1/scale, nu/scale, 0], [nu/scale, 1/scale, 0], [0, 0, 0.5*(1-nu)/scale]])
    g = (1/3) ** 0.5
    nodes = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ke = np.zeros((8, 8))
    for xi, eta in [(-g, -g), (g, -g), (g, g), (-g, g)]:
        B = np.zeros((3, 8))
        for a, (xa, ya) in enumerate(nodes):
            dndx = 0.5*xa*(1+ya*eta); dndy = 0.5*ya*(1+xa*xi)
            B[0, 2*a] = dndx; B[1, 2*a+1] = dndy; B[2, 2*a] = dndy; B[2, 2*a+1] = dndx
        ke += 0.25 * B.T @ D @ B
    return ke
KE = ke_gauss(NU)

def rust_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def heaviside(x, eps=1.0):
    out = np.where(x > eps, 1.0, np.where(x < -eps, 1e-3, 0.0))
    mid = (np.abs(x) <= eps)
    smooth = 0.5 * (1 + x/eps + np.sin(x*np.pi/eps)/np.pi)
    val = np.clip(1e-3 + (1 - 1e-3) * smooth, 1e-3, 1.0)
    return np.where(mid, val, out)

def element_density(phi, nx, ny):
    avg = 0.25 * (phi[:-1, :-1] + phi[:-1, 1:] + phi[1:, :-1] + phi[1:, 1:])   # (ny, nx), row 0 bottom
    return heaviside(avg)

class FE:
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        nn = (nx+1)*(ny+1)
        self.ndof = 2*nn
        edofs = []
        for row in range(ny):
            for col in range(nx):
                n0 = row*(nx+1) + col
                nodes = [n0, n0+1, n0+nx+2, n0+nx+1]
                edofs.append([d for n in nodes for d in (2*n, 2*n+1)])
        self.edofs = np.array(edofs)
        self.iK = np.repeat(self.edofs, 8, axis=1).ravel()
        self.jK = np.tile(self.edofs, (1, 8)).ravel()
        loads = np.zeros(self.ndof)
        loads[2*(ny*(nx+1)) + 1] = -1.0
        fixed = [2*(row*(nx+1)) for row in range(ny+1)] + [2*nx + 1]
        self.loads = loads
        self.fixed = np.array(sorted(set(fixed)))
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
    def solve(self, E):
        sK = (KE.ravel()[None, :] * E[:, None]).ravel()
        K = sp.csc_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        u = np.zeros(self.ndof)
        u[self.free] = spla.spsolve(K[self.free][:, self.free].tocsc(), self.loads[self.free])
        ue = u[self.edofs]
        se = np.einsum('ei,ij,ej->e', ue, KE, ue)
        return u, se, float(self.loads @ u)

def hole_array(nx, ny, spacing, radius):
    rows, cols = np.mgrid[0:ny+1, 0:nx+1].astype(float)
    cx = rust_round((cols - spacing/2)/spacing)*spacing + spacing/2
    cy = rust_round((rows - spacing/2)/spacing)*spacing + spacing/2
    return np.sqrt((cols-cx)**2 + (rows-cy)**2) - radius

def to_nodes(field, nx, ny):
    s = np.zeros((ny+1, nx+1)); c = np.zeros((ny+1, nx+1))
    f = field.reshape(ny, nx)
    s[:-1, :-1] += f; s[:-1, 1:] += f; s[1:, :-1] += f; s[1:, 1:] += f
    c[:-1, :-1] += 1; c[:-1, 1:] += 1; c[1:, :-1] += 1; c[1:, 1:] += 1
    return s / c

def cfl_dt(dt, v):
    m = np.abs(v).max()
    return 0.5 / m if m * dt > 0.5 else dt

def run(nx, ny, vf, dt, phi, n_iter, bracket=(-2.0, 3.0), reinit_first=True, interval=5, steps=10, verbose=False):
    fe = FE(nx, ny)
    comp, vols = [], []
    def reinit(phi):
        s = phi / np.sqrt(phi**2 + 1.0)
        p = phi
        for _ in range(steps):
            p = reinit_step(p, s, 0.3)
        return p
    for it in range(n_iter):
        if reinit_first and interval > 0 and it > 0 and it % interval == 0:
            phi = reinit(phi)
        rho = element_density(phi, nx, ny).ravel()
        E = 1e-9 + rho * (1 - 1e-9)
        u, se, c = fe.solve(E)
        comp.append(c)
        w = to_nodes(E * se, nx, ny)
        w = w / w.max()
        def vol_after(lam):
            v = lam - w
            p = hj_step(phi, v, cfl_dt(dt, v))
            return element_density(p, nx, ny).mean()
        lo, hi = bracket
        if vol_after(lo) <= vf:
            lam = lo
        elif vol_after(hi) >= vf:
            lam = hi
        else:
            for _ in range(60):
                mid = 0.5*(lo+hi)
                if vol_after(mid) > vf: lo = mid
                else: hi = mid
            lam = 0.5*(lo+hi)
        v = lam - w
        phi = hj_step(phi, v, cfl_dt(dt, v))
        if (not reinit_first) and interval > 0 and (it + 1) % interval == 0:
            phi = reinit(phi)
        vols.append(element_density(phi, nx, ny).mean())
    rho = element_density(phi, nx, ny).ravel()
    E = 1e-9 + rho * (1 - 1e-9)
    u, se, c = fe.solve(E)
    return comp, vols, c, rho, phi

if __name__ == "__main__":
    nx, ny = 30, 10
    phi0 = hole_array(nx, ny, 5.0, 1.2)
    print("== old settings (bracket [0,1], reinit at the end), expect Rust numbers 150.854 170.921 202.539 214.049 205.912 259.735")
    comp, vols, c, rho, phi = run(nx, ny, 0.5, 0.5, phi0.copy(), 12, bracket=(0.0, 1.0), reinit_first=False)
    print([round(x, 3) for x in comp[:8]], [round(x, 4) for x in vols[:8]])
    print("== new settings")
    comp, vols, c, rho, phi = run(nx, ny, 0.5, 0.5, phi0.copy(), 80)
    print("final compliance", c, "final volume", vols[-1])
    print("compliance history", [round(x, 1) for x in comp])
    print("volumes", [round(x, 4) for x in vols])
    print("gray elements", int(((rho > 0.1) & (rho < 0.9)).sum()))
