import math
R_T, R_C, S21, P_T, P_C = 62e6, 200e6, 92e6, 0.30, 0.25

def puck(s2, t):
    if s2 >= 0:
        shear = t / S21
        tension = (1 - P_T * R_T / S21) * s2 / R_T
        return math.sqrt(shear**2 + tension**2) + P_T * s2 / S21
    s = math.sqrt(1 + 2 * P_C * R_C / S21)
    ra = R_C / (1 + s)
    tc = S21 * math.sqrt(s)
    if -s2 * tc <= ra * abs(t):
        w = P_C * s2
        return (math.sqrt(t * t + w * w) + w) / S21
    shear = t / ((1 + s) * S21)
    comp = s2 / R_C
    return (shear**2 + comp**2) * R_C / (-s2)

print(puck(R_T, 0), puck(0, S21), puck(0, -S21), puck(-1e-3, S21), puck(-R_C, 0), puck(-0.5 * R_C, 0))
s = math.sqrt(1 + 2 * P_C * R_C / S21)
ra = R_C / (1 + s); tc = S21 * math.sqrt(s)
print("corner", puck(-ra, tc), ra, tc)
for tau in (20e6, 55e6, 90e6):
    b = -ra / tc * tau
    print("straddle", tau, puck(b * (1 - 1e-9), tau), puck(b * (1 + 1e-9), tau), abs(puck(b*(1-1e-9), tau) - puck(b*(1+1e-9), tau)))
states = [(40e6, 30e6), (-20e6, 60e6), (-120e6, 20e6), (-60e6, 40e6), (-150e6, 0.0)]
for s2, t in states:
    base = puck(s2, t)
    for a in (0.37, 1.0, 2.9):
        sc = puck(a * s2, a * t)
        assert abs(sc - a * base) < 1e-12 * a * base, (s2, t, a, sc, a*base)
    print("hom", s2, t, base)
print(puck(20e6, 50e6), puck(0, 50e6), puck(-20e6, 50e6))
# grid non-negativity & monotone
for s2 in (-250e6, -100e6, -20e6, 0.0, 20e6, 60e6):
    for t in (0.0, 25e6, 80e6):
        f = puck(s2, t)
        assert f >= 0 and puck(1.5*s2, 1.5*t) >= f - 1e-15
print("ok")
