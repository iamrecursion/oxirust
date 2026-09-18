import numpy as np, sys
sys.path.insert(0, '.')
from proto_gyro import *
I = np.diag([0.020833333333333332, 0.010833333333333334, 0.016666666666666666])
Iinv = np.linalg.inv(I)
def rhs(wb): return Iinv @ np.cross(I @ wb, wb)
def rk4(wb0, dt, T):
    wb = np.array(wb0, float); t = 0.0
    prev = wb.copy(); n = int(round(T/dt))
    for k in range(n):
        k1 = rhs(wb); k2 = rhs(wb + 0.5*dt*k1); k3 = rhs(wb + 0.5*dt*k2); k4 = rhs(wb + dt*k3)
        new = wb + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        if wb[2] > 0 >= new[2]:
            # linear interpolation of the crossing
            return t + dt*wb[2]/(wb[2] - new[2])
        wb = new; t += dt
    return None
for spin, dt in ((2.0, 0.005), (2.0, 0.01), (3.0, 0.005)):
    for pert in (1e-3,):
        w0 = np.array([pert, 0.0, spin])
        t_ref = rk4(w0, 1e-4, 30.0)
        q = np.array([0,0,0,1.0]); w = w0.copy()
        L0 = I @ w
        E0 = 0.5*w @ (I @ w)
        n = int(round(20.0/dt)); flip = None; mincos = 1.0; maxL = 0; minE = 1.0; maxE = 1.0
        for k in range(n):
            w = gyro_step(w, q, I, dt)
            q = quat_mul(exp_quat(w, dt), q); q /= np.linalg.norm(q)
            R = quat_to_mat(q)
            wb = R.T @ w
            L = R @ I @ wb
            maxL = max(maxL, np.linalg.norm(L - L0)/np.linalg.norm(L0))
            E = 0.5*wb @ (I @ wb); minE = min(minE, E/E0); maxE = max(maxE, E/E0)
            c = (R[:, 2] @ L)/np.linalg.norm(L); mincos = min(mincos, c)
            if flip is None and wb[2] < 0: flip = (k+1)*dt
        print(f"spin={spin} dt={dt} pert={pert}: ref flip={t_ref:.4f}  engine flip={flip}  rel diff={(flip-t_ref)/t_ref:+.3%}  mincos={mincos:.4f}  maxLdrift={maxL:.3e}  E/E0 in [{minE:.5f},{maxE:.6f}]")
