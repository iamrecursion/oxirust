import numpy as np
from proto_rosenstein import lorenz_orbit, slope
from proto_rosenstein2 import rosenstein_kd

for dt, n, windows in [(0.002, 5000, (1, 250, 500)), (0.005, 5000, (1, 100, 200))]:
    data = lorenz_orbit(n, dt)
    for w in windows:
        c = rosenstein_kd(data, 400, w, True)
        print(f"dt={dt} N={n} w={w}: curve[::50]={np.round(c[::50],3)}  slope[0,20]/dt={slope(c,0,20)/dt:.3f}  [20,100]={slope(c,20,100)/dt:.3f} [50,200]={slope(c,50,200)/dt:.3f} [100,300]={slope(c,100,300)/dt:.3f}")
# autocorrelation first zero crossing of x for dt=0.01
dt = 0.01
data = lorenz_orbit(5000, dt)
x = data[:,0] - data[:,0].mean()
ac = [np.dot(x[:-k], x[k:]) if k else np.dot(x, x) for k in range(0, 600)]
fz = next(k for k in range(1, 600) if ac[k] <= 0)
print("first zero crossing of autocorrelation of x at lag", fz, "samples =", fz*dt, "time units")
