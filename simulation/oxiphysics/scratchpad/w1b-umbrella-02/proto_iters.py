import numpy as np, sys
sys.path.insert(0, '.')
from proto_gyro import *
I = np.diag([0.020833333333333332, 0.010833333333333334, 0.016666666666666666])
Iinv = np.linalg.inv(I)
def rhs(wb): return Iinv @ np.cross(I @ wb, wb)
def ref_flip(wb0):
    dt = 1e-4; wb = np.array(wb0, float); t = 0.0
    for k in range(int(40/dt)):
        k1 = rhs(wb); k2 = rhs(wb + 0.5*dt*k1); k3 = rhs(wb + 0.5*dt*k2); k4 = rhs(wb + dt*k3)
        new = wb + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        if wb[2] > 0 >= new[2]: return t + dt*wb[2]/(wb[2] - new[2])
        wb = new; t += dt
def world_step(w, q, dt, iters):
    R = quat_to_mat(q); Iw = R @ I @ R.T
    w0 = w.copy(); w1 = w0.copy()
    for _ in range(iters):
        f = Iw @ (w1 - w0)/dt + np.cross(w1, Iw @ w1)
        J = Iw/dt + skew(w1) @ Iw - skew(Iw @ w1)
        w1 = w1 + np.linalg.solve(J, -f)
    return w1
for iters in (1, 2, 3):
    # test A: asymmetric box ω=(1,0.1,0), dt=0.01, T=10
    q = np.array([0, 0, 0, 1.0]); w = np.array([1.0, 0.1, 0.0])
    R = quat_to_mat(q); L0 = R @ I @ R.T @ w; E0 = 0.5*w @ (R @ I @ R.T @ w)
    maxL = 0; maxE = 0
    for k in range(1000):
        w = world_step(w, q, 0.01, iters)
        q = quat_mul(exp_quat(w, 0.01), q); q /= np.linalg.norm(q)
        R = quat_to_mat(q); L = R @ I @ R.T @ w; E = 0.5*w @ (R @ I @ R.T @ w)
        maxL = max(maxL, np.linalg.norm(L - L0)/np.linalg.norm(L0)); maxE = max(maxE, abs(E-E0)/E0)
    # test B: Dzhanibekov spin 3, dt 0.005
    spin = 3.0; w0 = np.array([1e-3, 0, spin]); t_ref = ref_flip(w0)
    q = np.array([0, 0, 0, 1.0]); w = w0.copy(); L0 = I @ w
    flip = None; mincos = 1; maxL2 = 0; minE = 1
    E0 = 0.5*w @ (I @ w)
    for k in range(int(round(20/0.005))):
        w = world_step(w, q, 0.005, iters)
        q = quat_mul(exp_quat(w, 0.005), q); q /= np.linalg.norm(q)
        R = quat_to_mat(q); wb = R.T @ w; L = R @ I @ wb
        maxL2 = max(maxL2, np.linalg.norm(L - L0)/np.linalg.norm(L0))
        mincos = min(mincos, (R[:, 2] @ L)/np.linalg.norm(L)); minE = min(minE, 0.5*wb @ (I @ wb)/E0)
        if flip is None and wb[2] < 0: flip = (k+1)*0.005
    print(f"iters={iters}: A: maxL={maxL:.3e} maxE={maxE:.3e} | B: ref flip {t_ref:.4f} engine {flip:.4f} ({(flip-t_ref)/t_ref:+.3%}) mincos={mincos:.4f} maxL={maxL2:.3e} minE={minE:.5f}")
