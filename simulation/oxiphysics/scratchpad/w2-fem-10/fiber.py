def rom(e_f, e_m, nu_f, nu_m, g_f, g_m, vf):
    vm = 1 - vf
    e1 = vf*e_f + vm*e_m
    nu12 = vf*nu_f + vm*nu_m
    eta_g = (g_f/g_m - 1)/(g_f/g_m + 1)
    g12 = g_m*(1 + eta_g*vf)/(1 - eta_g*vf)
    g23 = g12
    k_f = e_f/(3*(1-2*nu_f)) + g_f/3
    k_m = e_m/(3*(1-2*nu_m)) + g_m/3
    dk = k_f - k_m
    k23 = k_m if abs(dk) < 1e-12*abs(k_m) else k_m + vf/(1/dk + vm/(k_m+g_m))
    m = 1 + 4*k23*nu12*nu12/e1
    nu23 = (k23 - m*g23)/(k23 + m*g23)
    e2 = 4*k23*g23/(k23 + m*g23)
    return e1, e2, nu12, nu23, g12, g23, k23

iso = lambda e, nu: e/(2*(1+nu))
print("identical", rom(70e9, 70e9, .3, .3, iso(70e9,.3), iso(70e9,.3), .4))
ef, nuf, em, num = 230e9, .2, 3.5e9, .35
r = rom(ef, em, nuf, num, iso(ef,nuf), iso(em,num), .6)
print("carbon", r, "G23 check", r[5] - r[1]/(2*(1+r[3])))
e1, e2, nu12, nu23, g12, g23, k23 = r
print("k back", e2/(2*(1 - nu23 - 2*nu12**2*e2/e1)), k23)
k_f = ef/(3*(1-2*nuf)) + iso(ef,nuf)/3; k_m = em/(3*(1-2*num)) + iso(em,num)/3
vf = .6
print("wiener", 1/(vf/k_f+(1-vf)/k_m), k23, vf*k_f+(1-vf)*k_m)
ef, nuf, em, num = 72e9, .22, 3e9, .38
prev = num
for vf in (.1,.2,.3,.4,.5,.6,.7):
    r = rom(ef, em, nuf, num, iso(ef,nuf), iso(em,num), vf)
    print("glass", vf, r[3], abs(r[3]-prev))
    prev = r[3]
for vf in (0.0, 1.0):
    ef, nuf, em, num = 230e9, .2, 3.5e9, .35
    print("lim", vf, rom(ef, em, nuf, num, iso(ef,nuf), iso(em,num), vf))
