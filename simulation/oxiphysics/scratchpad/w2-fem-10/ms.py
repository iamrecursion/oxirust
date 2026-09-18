import numpy as np

def stiff(E, nu):
    lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
    C = np.zeros((6,6))
    for i in range(3):
        for j in range(3):
            C[i,j] = lam
        C[i,i] = lam + 2*mu
    for i in range(3,6): C[i,i] = mu
    return C, lam, mu

VP = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]

def gamma0(xi, tau6, lam0, mu0):
    n2 = xi @ xi
    if n2 == 0: return np.zeros(6)
    n = xi/np.sqrt(n2)
    t = np.zeros((3,3))
    for v,(i,j) in enumerate(VP):
        t[i,j] = tau6[v]; t[j,i] = tau6[v]
    tn = t @ n
    ntn = n @ tn
    a = 1/(2*mu0)
    long = (lam0+mu0)/(mu0*(lam0+2*mu0))
    # eps_ij = a*(n_j tn_i + n_i tn_j)... Gamma0 : tau
    e = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            e[i,j] = a*(n[j]*tn[i] + n[i]*tn[j]) - long*n[i]*n[j]*ntn
    return np.array([e[0,0], e[1,1], e[2,2], e[1,2]+e[2,1], e[0,2]+e[2,0], e[0,1]+e[1,0]])

def solve(Cfield, dims, mean_strain, lam0, mu0, tol=1e-11, maxit=20000):
    nx, ny, nz = dims
    N = nx*ny*nz
    strain = np.tile(mean_strain, (N,1))
    freq = [np.fft.fftfreq(n, 1.0/n) for n in dims]
    # wrapped freq like materials: i<=n/2 -> i else i-n  (Nyquist positive)
    def wf(n): return np.array([i if i <= n//2 else i-n for i in range(n)], float)
    kx, ky, kz = wf(nx), wf(ny), wf(nz)
    sh = np.zeros((6, nx, ny, nz), complex)
    eh = np.zeros((6, nx, ny, nz), complex)
    for m in range(6):
        eh[m,0,0,0] = mean_strain[m]*N
    for it in range(maxit):
        sig = np.einsum('nij,nj->ni', Cfield, strain)
        for m in range(6):
            sh[m] = np.fft.fftn(sig[:,m].reshape(nx,ny,nz))
        res2 = 0.0; mean2 = sum(abs(sh[m,0,0,0])**2 for m in range(6))
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    if ix==0 and iy==0 and iz==0: continue
                    xi = np.array([kx[ix], ky[iy], kz[iz]])
                    t = sh[:,ix,iy,iz]
                    T = np.zeros((3,3),complex)
                    for v,(i,j) in enumerate(VP):
                        T[i,j]=t[v]; T[j,i]=t[v]
                    d = T @ xi
                    res2 += np.sum(np.abs(d)**2)
                    er = gamma0(xi, t.real, lam0, mu0); ei = gamma0(xi, t.imag, lam0, mu0)
                    eh[:,ix,iy,iz] -= er + 1j*ei
        for m in range(6):
            eh[m,0,0,0] = mean_strain[m]*N
        for m in range(6):
            strain[:,m] = np.fft.ifftn(eh[m]).real.reshape(N)
        r = np.sqrt(res2)/np.sqrt(mean2) if mean2>0 else np.sqrt(res2)
        if r <= tol: return strain, it+1
    return strain, maxit

def ceff(Cfield, dims, lam0, mu0, **kw):
    N = len(Cfield)
    C = np.zeros((6,6))
    its = []
    for m in range(6):
        e = np.zeros(6); e[m] = 1
        strain, it = solve(Cfield, dims, e, lam0, mu0, **kw)
        its.append(it)
        sig = np.einsum('nij,nj->ni', Cfield, strain)
        C[:,m] = sig.mean(axis=0)
    return C, its

def backus(layers):
    # layers: list of (w, lam, mu)
    mean = lambda f: sum(w*f(l,m) for w,l,m in layers)
    inv_m = mean(lambda l,m: 1/(l+2*m)); l_over_m = mean(lambda l,m: l/(l+2*m))
    mu_mean = mean(lambda l,m: m); inv_mu = mean(lambda l,m: 1/m)
    c11 = mean(lambda l,m: 4*m*(l+m)/(l+2*m)) + l_over_m**2/inv_m
    c12 = mean(lambda l,m: 2*l*m/(l+2*m)) + l_over_m**2/inv_m
    c13 = l_over_m/inv_m; c33 = 1/inv_m; c44 = 1/inv_mu
    C = np.zeros((6,6))
    C[0,0]=C[1,1]=c11; C[0,1]=C[1,0]=c12; C[0,2]=C[2,0]=C[1,2]=C[2,1]=c13; C[2,2]=c33
    C[3,3]=C[4,4]=c44; C[5,5]=mu_mean
    return C

if __name__ == "__main__":
    A, lamA, muA = stiff(10e9, .30)
    B, lamB, muB = stiff(60e9, .25)
    dims = (4,4,8)
    field = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(8):
                field.append(B if iz >= 4 else A)
    field = np.array(field)
    # reference: basic scheme with 1.01*max K,G
    KA = 10e9/(3*(1-.6)); KB = 60e9/(3*(1-.5)); GA = muA; GB = muB
    k0 = 1.01*max(KA,KB); g0 = 1.01*max(GA,GB)
    lam0 = k0 - 2*g0/3
    C, its = ceff(field, dims, lam0, g0, tol=1e-12)
    ex = backus([(.5, lamA, muA), (.5, lamB, muB)])
    print(its)
    print(np.max(np.abs(C-ex))/np.max(np.abs(ex)))
