import numpy as np, sys, time
sys.path.insert(0, '.')
from proto_lbm import *

GHIA = [(1.0000, 1.00000), (0.9766, 0.84123), (0.9688, 0.78871), (0.9609, 0.73722), (0.9531, 0.68717),
        (0.8516, 0.23151), (0.7344, 0.00332), (0.6172, -0.13641), (0.5000, -0.20581), (0.4531, -0.21090),
        (0.2813, -0.15662), (0.1719, -0.10150), (0.1016, -0.06434), (0.0703, -0.04775), (0.0625, -0.04192),
        (0.0547, -0.03717), (0.0000, 0.00000)]

def lagrange_interp(xs, ys, x, order=4):
    # local polynomial interpolation using the `order` nearest nodes
    idx = np.argsort(np.abs(xs - x))[:order]
    idx = np.sort(idx)
    val = 0.0
    for a in idx:
        term = ys[a]
        for b in idx:
            if a != b:
                term *= (x - xs[b])/(xs[a] - xs[b])
        val += term
    return val

def run(N, u_lid, Re, nz, max_steps, check_every=200, tol=1e-6):
    nx = ny = N + 2
    nu = u_lid*N/Re
    tau = 3*nu + 0.5
    omega = 1/tau
    solid = np.zeros((nx, ny, nz), bool)
    solid[0, :, :] = True; solid[nx-1, :, :] = True; solid[:, 0, :] = True; solid[:, ny-1, :] = True
    lid = np.zeros_like(solid)
    lid[1:nx-1, ny-1, :] = True
    f = feq(np.ones((nx, ny, nz)), np.zeros((3, nx, ny, nz)))
    uw = np.array([u_lid, 0.0, 0.0])
    corr = 6.0*W*(E @ uw)  # per-direction 6 w_i (e_i . u_w)
    prev = None
    for step in range(1, max_steps+1):
        # collide non-solid
        rho, u = moments(f)
        fc = f - omega*(f - feq(rho, u))
        f = np.where(solid[None], f, fc)
        f = stream(f)
        # bounce-back on solid: swap with opposite
        fs = f[OPP]
        f = np.where(solid[None], fs, f)
        # lid correction
        rho_l = f.sum(0)
        add = corr[:, None, None, None]*rho_l[None]
        f = np.where(lid[None], f + add, f)
        if step % check_every == 0:
            rho, u = moments(f)
            prof = u[0, nx//2, :, nz//2].copy()
            if prev is not None:
                delta = np.abs(prof - prev).max()/u_lid
                if delta < tol:
                    return f, step, delta, nu, tau
            prev = prof
    return f, step, delta, nu, tau

def evaluate(f, N, u_lid, order=4):
    nx = ny = N + 2
    rho, u = moments(f)
    # fluid nodes j=1..N at y_phys=(j-0.5)/N, x = nx//2 line (use average of two central columns if N even)
    # we sample at x central: pick x = (nx)//2; its physical coordinate is (x-0.5)/N
    # sample the centerline x=0.5: interpolate between two columns if necessary
    ys = (np.arange(1, ny-1) - 0.5)/N
    xs = (np.arange(1, nx-1) - 0.5)/N
    zc = f.shape[3]//2
    # u(x=0.5, y) by interpolating in x of columns
    col_u = u[0, 1:nx-1, 1:ny-1, zc]  # (N, N) [x,y]
    prof = np.array([lagrange_interp(xs, col_u[:, j], 0.5, order) for j in range(N)])/u_lid
    out = []
    for (yl, ref) in GHIA:
        if yl <= 0 or yl >= 1:
            continue
        out.append((yl, lagrange_interp(ys, prof, yl, order), ref))
    return out

if __name__ == '__main__':
    N = int(sys.argv[1]); steps = int(sys.argv[2]); nz = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    t0 = time.time()
    f, step, delta, nu, tau = run(N, 0.1, 100.0, nz, steps)
    print(f"N={N} nz={nz} tau={tau:.4f} nu={nu:.5f} steps={step} last delta={delta:.3e}  time={time.time()-t0:.1f}s")
    res = evaluate(f, N, 0.1)
    worst = 0; worst_abs = 0
    for (yl, m, r) in res:
        err = abs(m - r)
        tol = 0.1*abs(r) + 0.01
        flag = '' if err <= tol else '  <-- FAIL'
        print(f"  y/L={yl:.4f} measured={m:+.5f} ref={r:+.5f} abs err={err:.5f} tol={tol:.5f}{flag}")
        worst_abs = max(worst_abs, err)
    print("worst abs err:", worst_abs)
