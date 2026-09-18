import math

def rates(k, g, ki, gi):
    p = (3*k + 4*g) / (3*ki + 4*g)
    den = 3*k*(3*g + 2*gi) + 4*g*(2*g + 3*gi)
    q = 5*g*(3*k + 4*g) / den
    return (ki - k) * p, (gi - g) * q

def ds(km, gm, ki, gi, f, n):
    if f == 0: return km, gm
    if f == 1: return ki, gi
    tend = -math.log1p(-f)
    dt = tend / n
    k, g = km, gm
    for _ in range(n):
        dk1, dg1 = rates(k, g, ki, gi)
        dk2, dg2 = rates(k + .5*dt*dk1, g + .5*dt*dg1, ki, gi)
        dk3, dg3 = rates(k + .5*dt*dk2, g + .5*dt*dg2, ki, gi)
        dk4, dg4 = rates(k + dt*dk3, g + dt*dg3, ki, gi)
        k += dt/6*(dk1 + 2*dk2 + 2*dk3 + dk4)
        g += dt/6*(dg1 + 2*dg2 + 2*dg3 + dg4)
    return k, g

def rel(a, b): return abs(a-b)/abs(b)

# voids nu=0.2
gm = 30e9; km = 4/3*gm
for f in (0.1, 0.3, 0.5):
    k, g = ds(km, gm, 0, 0, f, 200)
    s = (1-f)**2
    print("voids", f, rel(k, km*s), rel(g, gm*s))

# Roscoe
gm = 1e9; km = 1e6*gm; rigid = 1e9*gm
for f in (0.1, 0.3, 0.5):
    k, g = ds(km, gm, rigid, rigid, f, 400)
    print("roscoe", f, rel(g, gm*(1-f)**-2.5))

# slopes
km, gm, ki, gi = 50e9, 30e9, 300e9, 150e9
f = 1e-6
k, g = ds(km, gm, ki, gi, f, 4)
ks = (ki-km)*(km+4*gm/3)/(ki+4*gm/3)
zeta = gm*(9*km+8*gm)/(6*(km+2*gm))
gs = (gi-gm)*(gm+zeta)/(gi+zeta)
print("slopes", rel((k-km)/f, ks), rel((g-gm)/f, gs))

# HS bounds
def hs_bulk_lower(k1, g1, k2, f1, f2):
    denom = 1/(k2-k1) + 3*f1/(3*k1+4*g1)
    return k1 + f2/denom
def hs_bulk_upper(k2, g2, k1, f1, f2):
    denom = 1/(k1-k2) + 3*f2/(3*k2+4*g2)
    return k2 + f1/denom
def hs_shear_lower(k1, g1, g2, f1, f2):
    a1 = 6*f1*(k1+2*g1)/(5*g1*(3*k1+4*g1))
    return g1 + f2/(1/(g2-g1)+a1)
def hs_shear_upper(k2, g2, g1, f1, f2):
    a2 = 6*f2*(k2+2*g2)/(5*g2*(3*k2+4*g2))
    return g2 + f1/(1/(g1-g2)+a2)
for fi in (0.1, 0.3, 0.6):
    fm = 1-fi
    k, g = ds(km, gm, ki, gi, fi, 200)
    klo = hs_bulk_lower(km, gm, ki, fm, fi); khi = hs_bulk_upper(ki, gi, km, fm, fi)
    glo = hs_shear_lower(km, gm, gi, fm, fi); ghi = hs_shear_upper(ki, gi, gm, fm, fi)
    kv = fm*km + fi*ki; kr = 1/(fm/km + fi/ki)
    print("HS", fi, klo <= k <= khi, glo <= g <= ghi, kr < k < kv, (k-klo)/klo, (khi-k)/khi, (g-glo)/glo, (ghi-g)/ghi)

# rigid
km, gm = 50e9, 30e9
rigid = 1e3*km
f = 0.3
k, g = ds(km, gm, rigid, rigid, f, 400)
voigt = (1-f)*km + f*rigid
tangent = km + f*(km + 4*gm/3)
print("rigid", k, voigt, tangent, k < 0.1*voigt, k > tangent*1.05)

# RK4 order
km, gm, ki, gi, f = 50e9, 30e9, 300e9, 150e9, 0.5
kref, gref = ds(km, gm, ki, gi, f, 4000)
def err(n):
    k, g = ds(km, gm, ki, gi, f, n)
    return max(rel(k, kref), rel(g, gref))
e = [err(n) for n in (2, 4, 8)]
print("order", e, e[0]/e[1], e[1]/e[2])
