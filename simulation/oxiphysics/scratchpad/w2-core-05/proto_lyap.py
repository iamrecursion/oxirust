import numpy as np, time

SIGMA, RHO, BETA = 10.0, 28.0, 8.0/3.0

def f(x):
    return np.array([SIGMA*(x[1]-x[0]), x[0]*(RHO-x[2])-x[1], x[0]*x[1]-BETA*x[2]])

def jac(x):
    return np.array([[-SIGMA, SIGMA, 0.0], [RHO-x[2], -1.0, -x[0]], [x[1], x[0], -BETA]])

def rk4(x, h):
    k1 = f(x); k2 = f(x+0.5*h*k1); k3 = f(x+0.5*h*k2); k4 = f(x+h*k3)
    return x + h/6*(k1+2*k2+2*k3+k4)

def step_aug(x, Q, h):
    k1 = f(x); K1 = jac(x) @ Q
    x2 = x + 0.5*h*k1; k2 = f(x2); K2 = jac(x2) @ (Q + 0.5*h*K1)
    x3 = x + 0.5*h*k2; k3 = f(x3); K3 = jac(x3) @ (Q + 0.5*h*K2)
    x4 = x + h*k3; k4 = f(x4); K4 = jac(x4) @ (Q + h*K3)
    return x + h/6*(k1+2*k2+2*k3+k4), Q + h/6*(K1+2*K2+2*K3+K4)

def run(x0, h, n_steps, renorm=10):
    x = np.array(x0, float)
    for _ in range(2000): x = rk4(x, h)
    # QR spectrum
    Q = np.eye(3); sums = np.zeros(3)
    xx = x.copy()
    t0 = time.time()
    for _ in range(n_steps):
        xx, M = step_aug(xx, Q, h)
        Qn, R = np.linalg.qr(M)   # Householder QR; MGS equivalent up to signs
        sums += np.log(np.abs(np.diag(R)))
        Q = Qn
    spec = np.sort(sums / (h*n_steps))[::-1]
    # MLE single vector
    xx = x.copy(); v = np.array([[1.0],[0.0],[0.0]]); s = 0.0; cnt = 0
    for step in range(n_steps):
        xx, v = step_aug(xx, v, h)
        if (step+1) % renorm == 0:
            nrm = np.linalg.norm(v); s += np.log(nrm); v = v/nrm; cnt += 1
    mle = s/(cnt*renorm)/h
    return spec, mle, time.time()-t0

for x0 in ([1,1,1], [0.5,-1.0,12.0], [-3.0, 2.0, 25.0]):
    spec, mle, dt = run(x0, 0.01, 100000)
    print(x0, "spectrum", np.round(spec, 4), "sum", round(spec.sum(),4), "MLE", round(mle,4), f"({dt:.1f}s)")
