import numpy as np, itertools, math

def dwdr(r, h):
    sigma = 1.0/(math.pi*h**3)
    q = r/h
    if q >= 2.0 or q < 1e-14:
        return 0.0
    if q >= 1.0:
        t = 2.0 - q
        return sigma*(-0.75*t*t)/h
    return sigma*(-3.0*q + 2.25*q*q)/h

def w(r, h):
    sigma = 1.0/(math.pi*h**3)
    q = r/h
    if q >= 2: return 0.0
    if q >= 1:
        t = 2-q; return sigma*0.25*t**3
    return sigma*(1-1.5*q*q+0.75*q**3)

def lam_ratio(h_over_dx, k_dx, eps=0.01):
    dx = 1.0
    h = h_over_dx*dx
    k = k_dx/dx
    rng = int(math.ceil(2*h/dx))+1
    s = 0.0
    dens = 0.0
    for i in range(-rng, rng+1):
        for j in range(-rng, rng+1):
            for l in range(-rng, rng+1):
                r = math.sqrt(i*i+j*j+l*l)*dx
                dens += w(r, h)*dx**3
                if r < 1e-12 or r >= 2*h:
                    continue
                g = -r*dwdr(r, h)/(r*r + eps*h*h)
                s += g*(1-math.cos(k*i*dx))
    lam_over_nu = 2*dx**3*s
    return lam_over_nu/(k*k), dens*1.0  # density in units rho0*dx^3 per volume -> dens*rho0 ... (m=rho0 dx^3)

for h_over_dx in (1.0, 1.2, 1.3, 1.5):
    for n in (8, 12, 16, 20, 32):
        kdx = 2*math.pi/n
        ratio, dens = lam_ratio(h_over_dx, kdx)
        print(f"h/dx={h_over_dx:4.2f} N={n:3d} k*dx={kdx:.3f} lam/(nu k^2)={ratio:.4f} density/rho0={dens:.5f}")
