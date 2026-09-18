import sys
sys.path.insert(0, ".")
from top88_ref import top88
import numpy as np
x, xp, hist, KE = top88(60, 20, 0.5, 3.0, 1.5, 1, verbose=False)
for it, c, vol, ch in hist[:12]:
    print(it, repr(float(c)), repr(float(vol)), repr(float(ch)))
print("...")
for it, c, vol, ch in hist[-3:]:
    print(it, repr(float(c)), repr(float(vol)), repr(float(ch)))
print("n_iter", len(hist))
# some density samples of the final design (row-major (row, col) with row=0 at the TOP in top88 -> flip to bottom-up)
xf = xp[::-1, :]  # row 0 = bottom
print("final volume", float(xp.mean()))
for (r, c) in [(0, 0), (0, 59), (19, 0), (19, 59), (10, 30), (5, 5), (15, 10), (10, 50), (2, 40), (18, 3)]:
    print("bottom-up (row, col) =", (r, c), repr(float(xf[r, c])))
