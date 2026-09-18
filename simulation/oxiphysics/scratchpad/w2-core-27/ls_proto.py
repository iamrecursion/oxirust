import numpy as np

def one_sided(phi):
    ny1, nx1 = phi.shape
    dxm = np.zeros_like(phi); dxp = np.zeros_like(phi)
    dym = np.zeros_like(phi); dyp = np.zeros_like(phi)
    dxm[:, 1:] = phi[:, 1:] - phi[:, :-1]
    dxp[:, :-1] = phi[:, 1:] - phi[:, :-1]
    dxm[:, 0] = dxp[:, 0]
    dxp[:, -1] = dxm[:, -1]
    dym[1:, :] = phi[1:, :] - phi[:-1, :]
    dyp[:-1, :] = phi[1:, :] - phi[:-1, :]
    dym[0, :] = dyp[0, :]
    dyp[-1, :] = dym[-1, :]
    return dxm, dxp, dym, dyp

def hj_step(phi, v, dt):
    dxm, dxp, dym, dyp = one_sided(phi)
    gp = np.sqrt(np.maximum(dxm, 0)**2 + np.minimum(dxp, 0)**2 + np.maximum(dym, 0)**2 + np.minimum(dyp, 0)**2)
    gm = np.sqrt(np.maximum(dxp, 0)**2 + np.minimum(dxm, 0)**2 + np.maximum(dyp, 0)**2 + np.minimum(dym, 0)**2)
    return phi - dt * (np.maximum(v, 0) * gp + np.minimum(v, 0) * gm)

def reinit_step(phi, sign, dtau):
    dxm, dxp, dym, dyp = one_sided(phi)
    gpos = np.sqrt(np.maximum(np.maximum(dxm, 0)**2, np.minimum(dxp, 0)**2) + np.maximum(np.maximum(dym, 0)**2, np.minimum(dyp, 0)**2))
    gneg = np.sqrt(np.maximum(np.minimum(dxm, 0)**2, np.maximum(dxp, 0)**2) + np.maximum(np.minimum(dym, 0)**2, np.maximum(dyp, 0)**2))
    g = np.where(sign > 0, gpos, gneg)
    return phi - dtau * sign * (g - 1.0)

n = 40
yy, xx = np.mgrid[0:n+1, 0:n+1].astype(float)
dist = np.sqrt((xx - 20)**2 + (yy - 20)**2)

def front_radius(phi):
    row = phi[20]
    for col in range(20, n):
        a, b = row[col], row[col+1]
        if a > 0 and b <= 0:
            return (col - 20) + a / (a - b)
    return float('nan')

# (a) inward v=+1, dt=0.2, 20 steps
phi = 12 - dist
for _ in range(20):
    phi = hj_step(phi, np.ones_like(phi), 0.2)
print("inward radius", front_radius(phi), "expected 8")
band = np.abs(8 - dist) < 3
exact = 8 - dist
print("max err in band", np.abs(phi - exact)[band & (dist > 1)].max())
dyc, dxc = np.gradient(phi)
g = np.sqrt(dxc**2 + dyc**2)
print("grad range in band", g[band & (dist > 2)].min(), g[band & (dist > 2)].max())

# (b) outward
phi = 12 - dist
for _ in range(20):
    phi = hj_step(phi, -np.ones_like(phi), 0.2)
print("outward radius", front_radius(phi), "expected 16")
exact = 16 - dist
band = np.abs(exact) < 3
print("max err in band", np.abs(phi - exact)[band & (dist > 1)].max())
dyc, dxc = np.gradient(phi)
g = np.sqrt(dxc**2 + dyc**2)
print("grad range in band", g[band & (dist > 2)].min(), g[band & (dist > 2)].max())

# (j) reinit of 2*SDF
phi0 = 2 * (12 - dist)
phi = phi0.copy()
sign = phi0 / np.sqrt(phi0**2 + 1.0)
for k in range(100):
    phi = reinit_step(phi, sign, 0.3)
print("reinit front radius", front_radius(phi), "expected 12")
dyc, dxc = np.gradient(phi)
g = np.sqrt(dxc**2 + dyc**2)
mask = (np.abs(phi) <= 5) & (dist > 2)
print("reinit grad range", g[mask].min(), g[mask].max())
print("max |phi - sdf| in band", np.abs(phi - (12 - dist))[mask].max())

# SDF kept
phi = (12 - dist)
s0 = phi / np.sqrt(phi**2 + 1)
p = phi.copy()
for k in range(30):
    p = reinit_step(p, s0, 0.3)
mask = (np.abs(phi) < 4) & (dist > 2)
print("sdf drift", np.abs(p - phi)[mask].max())
print("sdf front", front_radius(p))

print("---- off-grid circle")
dist2 = np.sqrt((xx - 20.3)**2 + (yy - 19.6)**2)
def front_radius2(phi, cx=20.3, row=20):
    # radius from crossing along the grid row nearest to the centre: report crossing x-position
    r = phi[row]
    for col in range(21, n):
        a, b = r[col], r[col+1]
        if a > 0 and b <= 0:
            return col + a / (a - b)
    return float('nan')
phi0 = 2 * (12.4 - dist2)
phi = phi0.copy()
sign = phi0 / np.sqrt(phi0**2 + 1.0)
for k in range(100):
    phi = reinit_step(phi, sign, 0.3)
exact_cross = 20.3 + np.sqrt(12.4**2 - (20 - 19.6)**2)
print("reinit crossing", front_radius2(phi), "exact", exact_cross, "initial", front_radius2(phi0))
dyc, dxc = np.gradient(phi)
g = np.sqrt(dxc**2 + dyc**2)
mask = (np.abs(phi) <= 5) & (dist2 > 2)
print("grad range", g[mask].min(), g[mask].max())
print("max |phi - sdf|", np.abs(phi - (12.4 - dist2))[mask].max())
# SDF kept
phis = 12.4 - dist2
s0 = phis / np.sqrt(phis**2 + 1)
p = phis.copy()
for k in range(30):
    p = reinit_step(p, s0, 0.3)
m = (np.abs(phis) < 4) & (dist2 > 2)
print("sdf drift", np.abs(p - phis)[m].max(), "crossing", front_radius2(p), "exact", exact_cross)
# 10 steps
p = phis.copy()
for k in range(10):
    p = reinit_step(p, s0, 0.3)
print("sdf drift after 10", np.abs(p - phis)[m].max())
