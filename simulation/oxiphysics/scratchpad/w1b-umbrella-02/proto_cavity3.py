import numpy as np, sys
sys.path.insert(0, '.')
from proto_lbm import *
def go(N, u_lid, tau, steps, restrict=True, lid_all=False, corner_moving=False, nz=3):
    nx = ny = N + 2
    omega = 1/tau
    solid = np.zeros((nx, ny, nz), bool)
    solid[0, :, :] = True; solid[nx-1, :, :] = True; solid[:, 0, :] = True; solid[:, ny-1, :] = True
    lid = np.zeros_like(solid)
    if corner_moving:
        lid[:, ny-1, :] = True
    else:
        lid[1:nx-1, ny-1, :] = True
    f = feq(np.ones((nx, ny, nz)), np.zeros((3, nx, ny, nz)))
    uw = np.array([u_lid, 0, 0]); corr = 6.0*W*(E @ uw)
    if restrict:
        corr = np.where(EY < 0, corr, 0.0)
    fluid = ~solid
    out = []
    for step in range(1, steps+1):
        fc = collide_bgk(f, omega)
        f = np.where(solid[None], f, fc)
        f = stream(f)
        f = np.where(solid[None], f[OPP], f)
        rho_l = f.sum(0)
        f = np.where(lid[None], f + corr[:, None, None, None]*rho_l[None], f)
        if step % 100 == 0:
            rho, u = moments(f)
            out.append((step, np.abs(u[:, fluid]).max(), rho[fluid].min(), rho[fluid].max()))
    return out
for (u_lid, tau, label) in [(0.0, 0.8, "lid at rest"), (0.001, 0.8, "u_lid=0.001 tau=0.8"), (0.01, 0.8, "u_lid=0.01 tau=0.8"), (0.1, 0.8, "u_lid=0.1 tau=0.8"), (0.1, 1.0, "u_lid=0.1 tau=1.0")]:
    print(label)
    for r in go(16, u_lid, tau, 1000)[::3]:
        print("  ", r)
