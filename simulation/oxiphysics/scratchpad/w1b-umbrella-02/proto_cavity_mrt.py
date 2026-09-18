import numpy as np, sys, time
sys.path.insert(0, '.')
import proto_mrt
from proto_mrt import collide_mrt, from_viscosity
from proto_lbm import *
from proto_cavity2 import evaluate

def run(N, u_lid, Re, nz, max_steps, check_every=250, tol=1e-6):
    nx = ny = N + 2
    nu = u_lid*N/Re; tau = 3*nu + 0.5
    s = from_viscosity(nu)
    solid = np.zeros((nx, ny, nz), bool)
    solid[0, :, :] = True; solid[nx-1, :, :] = True; solid[:, 0, :] = True; solid[:, ny-1, :] = True
    lid = np.zeros_like(solid); lid[1:nx-1, ny-1, :] = True
    f = feq(np.ones((nx, ny, nz)), np.zeros((3, nx, ny, nz)))
    uw = np.array([u_lid, 0, 0]); corr = np.where(EY < 0, 6.0*W*(E @ uw), 0.0)
    fluid = ~solid
    prev = None; delta = None
    for step in range(1, max_steps+1):
        fc = collide_mrt(f, s)
        f = np.where(solid[None], f, fc)
        f = stream(f)
        f = np.where(solid[None], f[OPP], f)
        f = np.where(lid[None], f + corr[:, None, None, None], f)
        if step % check_every == 0:
            rho, u = moments(f)
            if not np.isfinite(u[:, fluid]).all() or np.abs(u[:, fluid]).max() > 1.0:
                return None, step, None, nu, tau
            prof = u[0, nx//2, :, nz//2].copy()
            if prev is not None:
                delta = np.abs(prof - prev).max()/u_lid
                if delta < tol: return f, step, delta, nu, tau
            prev = prof
    return f, step, delta, nu, tau

N = int(sys.argv[1])
t0 = time.time()
f, step, delta, nu, tau = run(N, 0.1, 100.0, 3, 20000)
print(f"MRT N={N} tau={tau:.4f} steps={step} delta={delta} time={time.time()-t0:.1f}s")
if f is not None:
    res = evaluate(f, N, 0.1)
    print("worst abs err:", max(abs(m-r) for (_, m, r) in res), "fails:", sum(abs(m-r) > 0.1*abs(r)+0.01 for (_, m, r) in res))
else:
    print("UNSTABLE")
