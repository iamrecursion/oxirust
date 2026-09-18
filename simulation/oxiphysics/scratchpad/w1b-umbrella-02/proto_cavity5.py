import numpy as np, sys, time
sys.path.insert(0, '.')
from proto_lbm import *
from proto_cavity import GHIA, lagrange_interp

def run(N, u_lid, Re, nz, max_steps, op='bgk', lam=3/16, restrict=True, check_every=250, tol=1e-6, verbose=False, mrt=None):
    nx = ny = N + 2
    nu = u_lid*N/Re; tau = 3*nu + 0.5; omega = 1/tau
    tm = lam/(tau - 0.5) + 0.5; om_minus = 1/tm
    solid = np.zeros((nx, ny, nz), bool)
    solid[0, :, :] = True; solid[nx-1, :, :] = True; solid[:, 0, :] = True; solid[:, ny-1, :] = True
    lid = np.zeros_like(solid); lid[1:nx-1, ny-1, :] = True
    f = feq(np.ones((nx, ny, nz)), np.zeros((3, nx, ny, nz)))
    uw = np.array([u_lid, 0, 0]); corr0 = 6.0*W*(E @ uw)
    # per-direction mask: link from a lid node along e_i ends in a fluid node
    corr_mask = np.zeros((19, nx, ny, nz))
    for i in range(19):
        dest_fluid = ~np.roll(solid, shift=(-EX[i], -EY[i], -EZ[i]), axis=(0, 1, 2))  # dest(x)=x+e_i -> solid[x+e_i]
        corr_mask[i] = np.where(lid & dest_fluid, corr0[i], 0.0)
    corr = None
    prev = None
    delta = None
    fluid = ~solid
    for step in range(1, max_steps+1):
        if op == 'bgk':
            fc = collide_bgk(f, omega)
        elif op == 'trt':
            fc = collide_trt(f, omega, om_minus)
        f = np.where(solid[None], f, fc)
        f = stream(f)
        f = np.where(solid[None], f[OPP], f)
        f = f + corr_mask
        if step % check_every == 0:
            rho, u = moments(f)
            if not np.isfinite(u[:, fluid]).all() or np.abs(u[:, fluid]).max() > 1.0:
                return None, step, float('nan'), nu, tau
            prof = u[0, nx//2, :, nz//2].copy()
            if prev is not None:
                delta = np.abs(prof - prev).max()/u_lid
                if verbose:
                    print(step, delta)
                if delta < tol:
                    return f, step, delta, nu, tau
            prev = prof
    return f, step, delta, nu, tau

def evaluate(f, N, u_lid, order=4):
    nx = ny = N + 2
    rho, u = moments(f)
    ys = (np.arange(1, ny-1) - 0.5)/N
    xs = (np.arange(1, nx-1) - 0.5)/N
    zc = f.shape[3]//2
    col_u = u[0, 1:nx-1, 1:ny-1, zc]
    prof = np.array([lagrange_interp(xs, col_u[:, j], 0.5, order) for j in range(N)])/u_lid
    out = []
    for (yl, ref) in GHIA:
        if yl <= 0 or yl >= 1:
            continue
        out.append((yl, lagrange_interp(ys, prof, yl, order), ref))
    return out

if __name__ == '__main__':
    N = int(sys.argv[1]); steps = int(sys.argv[2]); op = sys.argv[3]; nz = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    lam = float(sys.argv[5]) if len(sys.argv) > 5 else 3/16
    t0 = time.time()
    f, step, delta, nu, tau = run(N, 0.1, 100.0, nz, steps, op, lam)
    print(f"N={N} op={op} nz={nz} tau={tau:.4f} nu={nu:.5f} steps={step} last delta={delta}  time={time.time()-t0:.1f}s")
    if f is None:
        print("UNSTABLE"); sys.exit(0)
    res = evaluate(f, N, 0.1)
    worst_abs = 0; nfail = 0
    for (yl, m, r) in res:
        err = abs(m - r); tol = 0.1*abs(r) + 0.01
        flag = '' if err <= tol else '  <-- FAIL'
        nfail += err > tol
        print(f"  y/L={yl:.4f} measured={m:+.5f} ref={r:+.5f} abs err={err:.5f} tol={tol:.5f}{flag}")
        worst_abs = max(worst_abs, err)
    print("worst abs err:", worst_abs, "fails:", nfail)
