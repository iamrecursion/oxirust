import numpy as np

EX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0])
EY = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1])
EZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1])
W = np.array([1/3] + [1/18]*6 + [1/36]*12)
E = np.stack([EX, EY, EZ], 1)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15])
for i in range(19):
    assert (E[OPP[i]] == -E[i]).all()
CS2 = 1/3

def feq(rho, u):
    # rho: (nx,ny,nz), u: (3,nx,ny,nz) -> (19,nx,ny,nz)
    eu = np.einsum('qd,dxyz->qxyz', E.astype(float), u)
    usq = (u**2).sum(0)
    return W[:, None, None, None] * rho[None] * (1 + eu/CS2 + eu**2/(2*CS2**2) - usq[None]/(2*CS2))

def moments(f):
    rho = f.sum(0)
    u = np.einsum('qd,qxyz->dxyz', E.astype(float), f) / rho[None]
    return rho, u

def stream(f):
    out = np.empty_like(f)
    for i in range(19):
        out[i] = np.roll(f[i], shift=(EX[i], EY[i], EZ[i]), axis=(0, 1, 2))
    return out

def collide_bgk(f, omega):
    rho, u = moments(f)
    return f - omega*(f - feq(rho, u))

def collide_trt(f, wp, wm):
    rho, u = moments(f)
    fe = feq(rho, u)
    fo = f[OPP]
    feo = fe[OPP]
    fs = 0.5*(f + fo); fa = 0.5*(f - fo)
    fes = 0.5*(fe + feo); fea = 0.5*(fe - feo)
    return f - wp*(fs - fes) - wm*(fa - fea)

def ke(f):
    rho, u = moments(f)
    return 0.5*(rho*(u**2).sum(0)).sum()

def abc_field(N, U0, k):
    x, y, z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
    A = B = C = 1.0
    ux = U0*(A*np.sin(k*z) + C*np.cos(k*y))
    uy = U0*(B*np.sin(k*x) + A*np.cos(k*z))
    uz = U0*(C*np.sin(k*y) + B*np.cos(k*x))
    return np.stack([ux, uy, uz])

def run_decay(N, tau, U0, op, t1, t2):
    k = 2*np.pi/N
    u = abc_field(N, U0, k)
    rho = np.ones((N, N, N))
    f = feq(rho, u)
    omega = 1/tau
    nu = (tau - 0.5)/3
    ke1 = None
    for t in range(1, t2+1):
        if op == 'bgk':
            f = collide_bgk(f, omega)
        else:
            lam = 3/16
            tm = lam/(tau - 0.5) + 0.5
            f = collide_trt(f, omega, 1/tm)
        f = stream(f)
        if t == t1:
            ke1 = ke(f)
    ke2 = ke(f)
    nu_meas = -np.log(ke2/ke1)/(2*k*k*(t2 - t1))
    return nu, nu_meas

if __name__ == '__main__':
    import sys, time
    for op in ('bgk', 'trt'):
        for tau in (0.6, 0.8, 1.0):
            t0 = time.time()
            nu, nm = run_decay(32, tau, 0.01, op, 20, 220)
            print(f"{op} tau={tau} nu={nu:.5f} measured={nm:.5f} rel err={(nm-nu)/nu:+.4%}  ({time.time()-t0:.1f}s)")
