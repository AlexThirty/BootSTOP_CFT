import numpy as np
from scipy import special as sp
from values_BPS import Curvature
from matplotlib import pyplot as plt

def get_C(g):
    if g<1:
        C = 4*g**4 - (24*sp.zeta(3)+16*np.pi**2/3)*g**6 + (64*np.pi**2*sp.zeta(3)/3 + 360*sp.zeta(5)+64*np.pi**4/9)*g**8
        C = C - (112*np.pi**4*sp.zeta(3)/5 + 272*np.pi**2*sp.zeta(5) + 4816*sp.zeta(7) + 416*np.pi**6/45)*g**10
        C = C + (3488*np.pi**6*sp.zeta(3)/135 + 2192*np.pi**4*sp.zeta(5)/9 + 9184*np.pi**2*sp.zeta(7)/3 + 63504*sp.zeta(9) + 176*np.pi**8/15)*g**12
        
    else:
        C = (2*np.pi**2-3)*g/(6*np.pi**3) + (-24*sp.zeta(3)+5-4*np.pi**2)/(32*np.pi**4) + (11+2*np.pi**2)/(256*np.pi**5*g) + (96*sp.zeta(3)+75+8*np.pi**2)/(4096*np.pi**6*g**2)
        C = C + 3*(408*sp.zeta(3)-240*sp.zeta(5)+213+14*np.pi**2)/(65536*np.pi**7*g**3) + 3*(315*sp.zeta(3)-240*sp.zeta(5)+149+6*np.pi**2)/(65536*np.pi**8*g**4)
        #C = C + 3.0440129037e-7/g**5 + 8.00851627e-8/g**6 + 2-1258348e-8/g**7
        
    return C

gs = np.concatenate((np.arange(start=0.01, stop=0.25, step=0.01),
                     np.arange(start=0.25, stop=4.05, step=0.05),
                     #np.arange(start=4.25, stop=5.25, step=0.25)
                     ))
gs = np.around(gs, decimals=2)


mathematica_C = np.zeros(len(gs))
expansion_C = np.zeros(len(gs))

for i, g in enumerate(gs):
    mathematica_C[i] = Curvature[str(g)]
    expansion_C[i] = get_C(g)
    
plt.figure()
plt.plot(gs, mathematica_C, color='red', label='Mathematica')
plt.plot(gs, expansion_C, color='blue', label='python expansion')
plt.legend()
plt.ylim((0,0.01))
plt.xlim((0,0.25))
plt.show()