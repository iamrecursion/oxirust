import numpy as np, sys
sys.path.insert(0, '.')
from proto_lbm import *
N = 32; k = 2*np.pi/N
x, y, z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
for U0 in (0.002, 0.01, 0.03):
    for tau in (0.8,):
        u = U0*np.stack([np.sin(k*x)*np.cos(k*y)*np.cos(k*z), -np.cos(k*x)*np.sin(k*y)*np.cos(k*z), 0*x])
        f = feq(np.ones((N, N, N)), u)
        nu = (tau - 0.5)/3
        ke1 = None
        for t in range(1, 221):
            f = collide_bgk(f, 1/tau)
            f = stream(f)
            if t == 20: ke1 = ke(f)
        ke2 = ke(f)
        nu_meas = -np.log(ke2/ke1)/(6*k*k*200)
        print(f"U0={U0} tau={tau}: nu={nu:.5f} measured (Stokes 6 nu k^2)={nu_meas:.5f} rel err={(nu_meas-nu)/nu:+.3%}")
