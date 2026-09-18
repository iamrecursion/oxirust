import numpy as np, time

def lorenz_rhs(s, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = s
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

def rk4(s, h):
    k1 = lorenz_rhs(s); k2 = lorenz_rhs(s+0.5*h*k1); k3 = lorenz_rhs(s+0.5*h*k2); k4 = lorenz_rhs(s+h*k3)
    return s + h/6*(k1+2*k2+2*k3+k4)

def lorenz_orbit(n, dt):
    s = np.array([1.0, 1.0, 1.0])
    h = dt/4
    for _ in range(1000):
        for _ in range(4): s = rk4(s, h)
    out = []
    for _ in range(n):
        for _ in range(4): s = rk4(s, h)
        out.append(s.copy())
    return np.array(out)

def rosenstein(data, horizon, window, restricted=True):
    n = len(data)
    steps = min(horizon, n//2)
    min_sep = max(window, 1)
    followable = n - steps + 1 if restricted else n
    sum_log = np.zeros(steps); counts = np.zeros(steps)
    pts = data[:followable]
    # pairwise squared distances
    d2 = ((pts[:, None, :] - pts[None, :, :])**2).sum(-1)
    idx = np.arange(followable)
    mask = np.abs(idx[:, None] - idx[None, :]) < min_sep
    d2[mask] = np.inf
    nn = d2.argmin(1)
    ok = np.isfinite(d2[idx, nn])
    for t in range(steps):
        i = idx[ok]; j = nn[ok]
        if not restricted:
            sel = (i + t < n) & (j + t < n)
            i = i[sel]; j = j[sel]
        d = np.sqrt(((data[i+t] - data[j+t])**2).sum(-1))
        good = d > 1e-300
        sum_log[t] += np.log(d[good]).sum(); counts[t] += good.sum()
    return sum_log / counts

def slope(curve, a, b):
    t = np.arange(a, b+1)
    return np.polyfit(t, curve[a:b+1], 1)[0]

if __name__ == "__main__":
    dt = 0.01
    t0 = time.time()
    data = lorenz_orbit(5000, dt)
    print("orbit", time.time()-t0)
    for restricted in (True, False):
        c = rosenstein(data, 150, 100, restricted)
        print("restricted", restricted, "window=100 slope[10,100]/dt =", slope(c, 10, 100)/dt, "slope[0,150]", slope(c,0,149)/dt, "slope [20,120]", slope(c,20,120)/dt)
        c1 = rosenstein(data, 150, 1, restricted)
        print("   window=1 slope[10,100]/dt =", slope(c1, 10, 100)/dt)
    # logistic map
    x = 0.3; orbit = [x]
    for _ in range(3200):
        x = 4.0*x*(1-x); orbit.append(x)
    orbit = np.array(orbit)
    emb = np.stack([orbit[200:3200][:-1], orbit[200:3200][1:]], 1)
    for w in (1, 2, 3):
        for restricted in (True, False):
            c = rosenstein(emb, 8, w, restricted)
            print("logistic window", w, "restricted", restricted, "slope[0,4] =", slope(c, 0, 4), "curve", np.round(c, 3))
