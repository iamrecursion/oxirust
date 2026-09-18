import numpy as np
from top88_ref import top88
def ke_gauss(nu):
    scale = 1 - nu*nu
    D = np.array([[1/scale, nu/scale, 0],[nu/scale, 1/scale, 0],[0,0,0.5*(1-nu)/scale]])
    g = (1/3)**0.5
    nodes = [(-1,-1),(1,-1),(1,1),(-1,1)]
    ke = np.zeros((8,8))
    for xi, eta in [(-g,-g),(g,-g),(g,g),(-g,g)]:
        B = np.zeros((3,8))
        for a,(xa,ya) in enumerate(nodes):
            dndx = 0.5*xa*(1+ya*eta)
            dndy = 0.5*ya*(1+xa*xi)
            B[0,2*a]=dndx; B[1,2*a+1]=dndy; B[2,2*a]=dndy; B[2,2*a+1]=dndx
        ke += 0.25 * B.T @ D @ B
    return ke
def ke_top88(nu):
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], float)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], float)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], float)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], float)
    top = np.vstack([np.hstack([A11, A12]), np.hstack([A12.T, A11])])
    bot = np.vstack([np.hstack([B11, B12]), np.hstack([B12.T, B11])])
    return 1.0 / (1 - nu**2) / 24.0 * (top + nu * bot)
for nu in (0.0, 0.3, 0.45):
    print(nu, np.abs(ke_gauss(nu) - ke_top88(nu)).max())
