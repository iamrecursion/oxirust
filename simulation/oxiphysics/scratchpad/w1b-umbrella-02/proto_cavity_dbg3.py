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
uw = np.array([u_lid, 0, 0]); corr = 6.0*W*(E @ uw); corr = np.where(EY < 0, corr, 0.0)
fluid = ~solid
for step in range(1, 401):
    rho, u = moments(f)
    fc = f - omega*(f - feq(rho, u))
    f = np.where(solid[None], f, fc)
    f = stream(f)
    f = np.where(solid[None], f[OPP], f)
    rho_l = f.sum(0)
    f = np.where(lid[None], f + corr[:, None, None, None]*rho_l[None], f)
    if step in (1, 2, 3, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400):
        rho, u = moments(f)
        sp = np.sqrt((u**2).sum(0))
        sp_f = np.where(fluid, sp, 0)
        idx = np.unravel_index(np.argmax(sp_f), sp.shape)
        print(step, "max|u|", sp_f.max(), "at", idx, " rho fluid min/max", rho[fluid].min(), rho[fluid].max(), "rho min location", np.unravel_index(np.argmin(np.where(fluid, rho, 9)), rho.shape))
