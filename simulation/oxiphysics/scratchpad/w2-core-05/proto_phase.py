import numpy as np

def analytic(x):
    n = len(x)
    n2 = 1 << (n-1).bit_length() if n > 1 else 1
    X = np.fft.fft(np.concatenate([x, np.zeros(n2-n)]))
    X[1:n2//2] *= 2
    X[n2//2+1:] = 0
    return np.fft.ifft(X)[:n]

def inst_phase(sig):
    n = len(sig)
    if n < 2: return np.zeros(n)
    c = sig - sig.mean()
    z = analytic(c)
    w = np.arctan2(z.imag, z.real)
    out = np.empty(n); off = 0.0; prev = None
    for i, wi in enumerate(w):
        if prev is not None:
            off -= 2*np.pi*np.round((wi-prev)/(2*np.pi))
        out[i] = wi + off; prev = wi
    return out

n = 1024
k = np.arange(n)
sig = np.cos(2*np.pi*5*k/n + 0.7)
p = inst_phase(sig)
slope = 2*np.pi*5/n
steps = np.diff(p)[100:924]
print("cos: min step/slope", steps.min()/slope, "max", steps.max()/slope, "max abs err vs ramp", np.abs(p - (0.7 + slope*k)).max(), "last", p[-1])

n = 1000; fs = 1000.0
k = np.arange(n)
sig = np.sin(2*np.pi*5*k/fs)
p = inst_phase(sig)
slope = 2*np.pi*5/fs
steps = np.diff(p)[100:900]
print("sin1000: min step/slope", steps.min()/slope, "max", steps.max()/slope, "mean slope ratio", ((p[900]-p[100])/800)/slope, "mid offset", p[500]-slope*500, " -pi/2 =", -np.pi/2)

n = 1024
k = np.arange(n)
a = np.sin(2*np.pi*6*k/n)
locked = 0.4*np.sin(2*np.pi*6*k/n + 1.1)
drift = np.sin(2*np.pi*9.3*k/n)
pa, pl, pd = inst_phase(a), inst_phase(locked), inst_phase(drift)
R = lambda x, y: abs(np.mean(np.exp(1j*(x-y))))
print("R locked", R(pa, pl), "R drifting", R(pa, pd))
# two sample
print(inst_phase(np.array([1.0, -1.0])))
