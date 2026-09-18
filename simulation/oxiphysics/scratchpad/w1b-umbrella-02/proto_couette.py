import numpy as np, sys
sys.path.insert(0, '.')
from proto_lbm import *
nx, ny, nz = 3, 22, 3
H = ny - 2
u_w = 0.05
tau = 0.8; omega = 1/tau
solid = np.zeros((nx, ny, nz), bool); solid[:, 0, :] = True; solid[:, ny-1, :] = True
lid = np.zeros_like(solid); lid[:, ny-1, :] = True
f = feq(np.ones((nx, ny, nz)), np.zeros((3, nx, ny, nz)))
uw = np.array([u_w, 0, 0]); corr = 6.0*W*(E @ uw)
for step in range(1, 6001):
    rho, u = moments(f)
    fc = f - omega*(f - feq(rho, u))
    f = np.where(solid[None], f, fc)
    f = stream(f)
    f = np.where(solid[None], f[OPP], f)
    rho_l = f.sum(0)
    f = np.where(lid[None], f + corr[:, None, None, None]*rho_l[None], f)
rho, u = moments(f)
prof = u[0, 1, 1:ny-1, 1]
y = np.arange(1, ny-1) - 0.5   # distance from bottom wall plane at y=0.5
exact = u_w*y/H
print("max |u - exact|/u_w:", np.abs(prof - exact).max()/u_w)
print(prof[:5], exact[:5])
print("max|f| solid:", np.abs(f[:, solid]).max(), " rho fluid min/max", rho[:,1:ny-1].min(), rho[:,1:ny-1].max())
