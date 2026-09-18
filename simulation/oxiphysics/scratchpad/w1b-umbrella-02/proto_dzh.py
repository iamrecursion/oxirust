import numpy as np
import sys
sys.path.insert(0, '.')
from proto_gyro import *

# box 0.2 x 0.4 x 0.3 -> Ixx=0.0208 (largest), Iyy=0.0108 (smallest), Izz=0.01667 (intermediate)
I = np.diag([0.020833333333333332, 0.010833333333333334, 0.016666666666666666])
Iinv = np.linalg.inv(I)

def euler_rhs(wb):
    # I wb' = (I wb) x wb
    return Iinv @ np.cross(I @ wb, wb)

def rk4_body(wb0, dt, T):
    n = int(round(T/dt))
    wb = np.array(wb0, float)
    out = [(0.0, wb.copy())]
    for k in range(n):
        k1 = euler_rhs(wb)
        k2 = euler_rhs(wb + 0.5*dt*k1)
        k3 = euler_rhs(wb + 0.5*dt*k2)
        k4 = euler_rhs(wb + dt*k3)
        wb = wb + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        out.append(((k+1)*dt, wb.copy()))
    return out

for spin in (2.0, 3.0, 5.0):
    pert = 1e-3
    w0 = np.array([pert*spin, pert*spin*0.5, spin])  # perturb in x, y; spin about z (intermediate)
    ref = rk4_body(w0, 1e-4, 20.0)
    # first flip: wz changes sign
    t_flip = next((t for t, wb in ref if wb[2] < 0), None)
    lam = spin*np.sqrt((I[2,2]-I[1,1])*(I[0,0]-I[2,2])/(I[1,1]*I[0,0]))
    print(f"spin={spin} growth rate lambda={lam:.4f}; ref flip time (wz<0)={t_flip}")
    for dt in (0.01, 0.005, 1/240):
        # engine simulation
        q = np.array([0,0,0,1.0]); w = w0.copy()
        n = int(round(20.0/dt))
        L0 = quat_to_mat(q) @ I @ quat_to_mat(q).T @ w
        flip_t = None
        min_dot = 1.0
        maxvec = 0.0
        for k in range(n):
            w = gyro_step(w, q, I, dt)
            q = quat_mul(exp_quat(w, dt), q); q /= np.linalg.norm(q)
            R = quat_to_mat(q)
            L = R @ I @ R.T @ w
            maxvec = max(maxvec, np.linalg.norm(L-L0)/np.linalg.norm(L0))
            e2 = R[:, 2]
            d = e2 @ L / np.linalg.norm(L)
            min_dot = min(min_dot, d)
            wb = R.T @ w
            if flip_t is None and wb[2] < 0:
                flip_t = (k+1)*dt
        print(f"   dt={dt:.5f}: flip time={flip_t}, min cos(e2,L)={min_dot:.4f}, max L vec drift={maxvec:.3e}")
    # no gyro
    q = np.array([0,0,0,1.0]); w = w0.copy(); dt = 0.01
    min_dot = 1.0
    L0 = I @ w
    for k in range(2000):
        q = quat_mul(exp_quat(w, dt), q); q /= np.linalg.norm(q)
        R = quat_to_mat(q)
        L = R @ I @ R.T @ w
        d = R[:, 2] @ L / np.linalg.norm(L)
        min_dot = min(min_dot, d)
    print(f"   no gyro dt=0.01: min cos(e2,L)={min_dot:.6f}")
