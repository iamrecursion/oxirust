import numpy as np, sys
sys.path.insert(0, '.')
from proto_lbm import *
def go(N, u_lid, tau, steps, ramp=0, restrict=True, corner_moving=False, nz=3, rho_const=False):
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
    fluid = ~solid
    out = []
    for step in range(1, steps+1):
        ul = u_lid*min(1.0, step/ramp) if ramp else u_lid
        uw = np.array([ul, 0, 0]); corr = 6.0*W*(E @ uw)
        if restrict:
            corr = np.where(EY < 0, corr, 0.0)
        fc = collide_bgk(f, omega)
        f = np.where(solid[None], f, fc)
        f = stream(f)
        f = np.where(solid[None], f[OPP], f)
        rho_l = np.ones_like(f[0]) if rho_const else f.sum(0)
        f = np.where(lid[None], f + corr[:, None, None, None]*rho_l[None], f)
        if step % 200 == 0:
            rho, u = moments(f)
            out.append((step, round(float(np.abs(u[:, fluid]).max()), 5), round(float(rho[fluid].min()), 5), round(float(rho[fluid].max()), 5)))
            if not np.isfinite(out[-1][1]) or out[-1][1] > 5: break
    return out
for kw in [dict(), dict(ramp=1000), dict(corner_moving=True), dict(rho_const=True), dict(restrict=False)]:
    print(kw)
    for r in go(16, 0.1, 0.8, 2000, **kw)[::2]:
        print("  ", r)
