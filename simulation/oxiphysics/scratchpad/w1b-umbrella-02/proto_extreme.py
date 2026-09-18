import numpy as np
rng = np.random.default_rng(1)
def skew(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def step(omega, Iw, dt, iters=2, energy_guard=False):
    res = lambda w1: Iw @ (w1 - omega)/dt + np.cross(w1, Iw @ w1)
    r0 = np.linalg.norm(res(omega))
    if not (r0 > 0 and np.isfinite(r0)): return omega, 'noop'
    w1 = omega.copy()
    for _ in range(iters):
        f = res(w1)
        J = Iw/dt + skew(w1) @ Iw - skew(Iw @ w1)
        try:
            w1 = w1 - np.linalg.inv(J) @ f
        except np.linalg.LinAlgError:
            return omega, 'singular'
    ok = np.isfinite(w1).all() and np.linalg.norm(res(w1)) < r0
    if ok and energy_guard:
        ok = (w1 @ Iw @ w1) <= (omega @ Iw @ omega)*(1 + 1e-12)
    return (w1, 'ok') if ok else (omega, 'rejected')
worst = 0; cnt = {'ok': 0, 'rejected': 0, 'noop': 0, 'singular': 0}; energy_gain = 0
for trial in range(20000):
    A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)
    d = np.exp(rng.uniform(-3, 3, size=3))
    Iw = Q @ np.diag(d) @ Q.T
    omega = rng.normal(size=3) * 10**rng.uniform(-3, 4)
    dt = 10**rng.uniform(-6, 2)
    w1, st = step(omega, Iw, dt)
    cnt[st] += 1
    e0 = omega @ Iw @ omega; e1 = w1 @ Iw @ w1
    if e1 > e0*(1 + 1e-9):
        energy_gain += 1
        worst = max(worst, e1/e0)
print(cnt, "energy gained in", energy_gain, "cases; worst ratio", worst)
