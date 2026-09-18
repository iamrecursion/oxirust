import numpy as np

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def quat_mul(a, b):
    # (x,y,z,w) convention
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ])

def quat_to_mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])

def exp_quat(omega, dt):
    n = np.linalg.norm(omega)
    if n < 1e-10:
        return np.array([0, 0, 0, 1.0])
    half = n * dt * 0.5
    ax = omega / n
    s = np.sin(half)
    return np.array([ax[0]*s, ax[1]*s, ax[2]*s, np.cos(half)])

def gyro_step(omega_w, q, I, dt, iters=2):
    R = quat_to_mat(q)
    w0 = R.T @ omega_w
    w1 = w0.copy()
    for _ in range(iters):
        f = I @ (w1 - w0) / dt + np.cross(w1, I @ w1)
        J = I / dt + skew(w1) @ I - skew(I @ w1)
        w1 = w1 + np.linalg.solve(J, -f)
    return R @ w1

def run(I, omega0, dt, T, gyro=True, iters=2, record=False):
    q = np.array([0, 0, 0, 1.0])
    w = np.array(omega0, float)
    n = int(round(T / dt))
    traj = []
    R = quat_to_mat(q)
    L0 = R @ I @ R.T @ w
    maxvec = 0.0
    maxmag = 0.0
    E0 = 0.5 * w @ (R @ I @ R.T @ w)
    maxE = 0.0
    for k in range(n):
        if gyro:
            w = gyro_step(w, q, I, dt, iters)
        dq = exp_quat(w, dt)
        q = quat_mul(dq, q)
        q /= np.linalg.norm(q)
        R = quat_to_mat(q)
        L = R @ I @ R.T @ w
        maxvec = max(maxvec, np.linalg.norm(L - L0) / np.linalg.norm(L0))
        maxmag = max(maxmag, abs(np.linalg.norm(L) - np.linalg.norm(L0)) / np.linalg.norm(L0))
        E = 0.5 * w @ (R @ I @ R.T @ w)
        maxE = max(maxE, abs(E - E0) / E0)
        if record:
            traj.append((k * dt + dt, q.copy(), w.copy(), L.copy()))
    return maxvec, maxmag, maxE, traj

I = np.diag([0.020833333333333332, 0.010833333333333334, 0.016666666666666666])
w0 = [1.0, 0.1, 0.0]
for gyro in (False, True):
    for dt in (0.01, 0.005, 1/240):
        mv, mm, mE, _ = run(I, w0, dt, 10.0, gyro=gyro)
        print(f"gyro={gyro} dt={dt:.5f} max vec drift={mv:.4e} max mag drift={mm:.4e} max E drift={mE:.4e}")
