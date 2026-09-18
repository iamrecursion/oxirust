import re, numpy as np, sys
sys.path.insert(0, '.')
from proto_lbm import *

src = open('/Users/kitasan/work/oxiphysics/crates/oxiphysics-lbm/src/mrt3d.rs').read()
blk = src[src.index('pub const M: [[f64; 19]; 19] = ['):]
blk = blk[:blk.index('];\n')]
rows = []
for line in blk.splitlines():
    line = line.strip()
    if line.startswith('[') and line.endswith('],'):
        nums = [float(x) for x in re.findall(r'-?\d+\.', line)]
        if len(nums) == 19:
            rows.append(nums)
M = np.array(rows)
assert M.shape == (19, 19), M.shape
Minv = np.linalg.inv(M)
# check orthogonality property: M W M^T diag
print("M orth check:", np.abs((M @ M.T) - np.diag(np.diag(M @ M.T))).max())

def collide_mrt(f, s):
    rho, u = moments(f)
    fe = feq(rho, u)
    sh = f.shape[1:]
    m = (M @ f.reshape(19, -1))
    me = (M @ fe.reshape(19, -1))
    s = np.array(s).copy()
    for c in (0, 3, 5, 7):
        s[c] = 0.0
    m = m - s[:, None]*(m - me)
    return (Minv @ m).reshape((19,) + sh)

def from_viscosity(nu):
    omega = 1.0/(3*nu + 0.5)
    s = [0.0, 1.19, 1.4, 0.0, 1.2, 0.0, 1.2, 0.0, 1.2, omega, 1.4, omega, 1.4, omega, omega, omega, 1.98, 1.98, 1.98]
    return s

def run_decay_mrt(N, tau, U0, t1, t2):
    k = 2*np.pi/N
    u = abc_field(N, U0, k)
    rho = np.ones((N, N, N))
    f = feq(rho, u)
    nu = (tau - 0.5)/3
    s = from_viscosity(nu)
    ke1 = None
    for t in range(1, t2+1):
        f = collide_mrt(f, s)
        f = stream(f)
        if t == t1:
            ke1 = ke(f)
    ke2 = ke(f)
    return nu, -np.log(ke2/ke1)/(2*k*k*(t2 - t1))

for tau in (0.6, 0.8, 1.0):
    nu, nm = run_decay_mrt(32, tau, 0.01, 20, 220)
    print(f"mrt tau={tau} nu={nu:.5f} measured={nm:.5f} rel err={(nm-nu)/nu:+.4%}")
