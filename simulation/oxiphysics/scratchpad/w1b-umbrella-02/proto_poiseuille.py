import numpy as np, sys
sys.path.insert(0, '.')
from proto_lbm import *

def guo_collide(f, omega, Fx, solid=None):
    rho, j = moments(f)[0], None
    rho = f.sum(0)
    mom = np.einsum('qd,qxyz->dxyz', E.astype(float), f)
    u = np.stack([(mom[0] + 0.5*Fx)/rho, mom[1]/rho, mom[2]/rho])
    fe = feq(rho, u)
    F = np.array([Fx, 0.0, 0.0])
    eu = np.einsum('qd,dxyz->qxyz', E.astype(float), u)
    ef = (E @ F)  # (19,)
    uf = (u * F[:, None, None, None]).sum(0)
    G = W[:, None, None, None]*(eu*ef[:, None, None, None]/CS2**2 - uf[None]/CS2 + ef[:, None, None, None]/CS2)
    out = f - omega*(f - fe) + (1 - 0.5*omega)*G
    if solid is not None:
        out = np.where(solid[None], f, out)
    return out

def poiseuille(nx, ny, nz, tau, Fx, steps, mode):
    omega = 1/tau
    nu = (tau - 0.5)/3
    f = feq(np.ones((nx, ny, nz)), np.zeros((3, nx, ny, nz)))
    solid = np.zeros((nx, ny, nz), bool)
    if mode in ('flagged', 'flagged_rows'):
        solid[:, 0, :] = True; solid[:, ny-1, :] = True
    rows = np.zeros((nx, ny, nz), bool); rows[:, 0, :] = True; rows[:, ny-1, :] = True
    for s in range(steps):
        f = guo_collide(f, omega, Fx, solid if mode != 'unflagged' else None)
        f = stream(f)
        if mode == 'flagged':
            f = np.where(solid[None], f[OPP], f)
        else:
            f = np.where(rows[None], f[OPP], f)
    rho = f.sum(0)
    mom = np.einsum('qd,qxyz->dxyz', E.astype(float), f)
    ux = (mom[0] + 0.5*Fx*(~solid))/rho
    prof = ux[nx//2, :, nz//2]
    H = ny - 2
    y = np.arange(ny) - 0.5
    ana = Fx/(2*nu)*y*(H - y)
    ana[0] = ana[-1] = 0
    sl = slice(1, ny-1)
    l2 = np.sqrt(((prof[sl] - ana[sl])**2).sum()/ (ana[sl]**2).sum())
    return l2, prof[ny//2], ana[ny//2]

for mode in ('flagged', 'unflagged'):
    for steps in (6000, 20000):
        l2, c, a = poiseuille(4, 21, 4, 0.8, 1e-6, steps, mode)
        print(f"{mode} steps={steps}: rel L2={l2:.3e} centre={c:.6e} analytic={a:.6e}")
