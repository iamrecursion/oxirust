import numpy as np, sys
sys.path.insert(0, '.')
from proto_lbm import *
N = 24; u_lid = 0.1; Re = 100.0; nz = 3
nx = ny = N + 2
nu = u_lid*N/Re; tau = 3*nu + 0.5; omega = 1/tau
solid = np.zeros((nx, ny, nz), bool)
solid[0, :, :] = True; solid[nx-1, :, :] = True; solid[:, 0, :] = True; solid[:, ny-1, :] = True
lid = np.zeros_like(solid); lid[1:nx-1, ny-1, :] = True
f = feq(np.ones((nx, ny, nz)), np.zeros((3, nx, ny, nz)))
uw = np.array([u_lid, 0, 0]); corr = 6.0*W*(E @ uw)
fluid = ~solid
for step in range(1, 12001):
    rho, u = moments(f)
    fc = f - omega*(f - feq(rho, u))
    f = np.where(solid[None], f, fc)
    f = stream(f)
    f = np.where(solid[None], f[OPP], f)
    rho_l = f.sum(0)
    f = np.where(lid[None], f + corr[:, None, None, None]*rho_l[None], f)
    if step % 500 == 0:
        rho, u = moments(f)
        um = np.sqrt((u**2).sum(0))[fluid].max()
        print(step, "max|u| fluid", um, "rho min/max fluid", rho[fluid].min(), rho[fluid].max(), "max |f| solid", np.abs(f[:, solid]).max())
        if not np.isfinite(um): break
