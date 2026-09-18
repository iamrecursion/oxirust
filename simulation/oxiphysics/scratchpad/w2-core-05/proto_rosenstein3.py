import numpy as np, time
from proto_rosenstein import lorenz_orbit, slope
from proto_rosenstein2 import rosenstein_kd

dt = 0.01
big = lorenz_orbit(20000, dt)
rng = [(0, 5), (0, 10), (0, 20), (30, 150), (50, 200), (20, 120)]
for n in (3000, 5000, 8000, 10000, 20000):
    for start in (0,):
        data = big[start:start+n]
        for window in (100,):
            c = rosenstein_kd(data, 300, window, True)
            print(f"N={n} w={window}: " + "  ".join(f"[{a},{b}]={slope(c,a,b)/dt:.3f}" for a, b in rng))
# different segments of the same length to see sampling variance
print("variance across segments, N=5000, w=100")
for start in range(0, 15001, 5000):
    data = big[start:start+5000]
    c = rosenstein_kd(data, 300, 100, True)
    print(f" seg {start}: " + "  ".join(f"[{a},{b}]={slope(c,a,b)/dt:.3f}" for a, b in rng))
