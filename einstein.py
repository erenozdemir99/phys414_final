import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate

#PART A
k = 100
length_unit = 1.477  # in km


def TOV(r, f):
    """returns the system of ODEs for Tolman-
Oppenheimer-Volkoff (TOV) equations: f = [m ,v, p, m_p]"""
    df = np.zeros(4)

    if r == 0:
        df = np.array([0,0,0,0])
    else:
        rho = np.sqrt(f[2]/k)
        df[0] = 4*np.pi*r**2 * rho
        df[1] = 2* (f[0]+4*np.pi*r**3 *f[2])/(r*(r-2*f[0]))
        df[2] = (-1/2) * (f[2] + rho) * df[1]
        df[3] = 4*np.pi*np.sqrt(1/(1-2*(f[0]/r))) * r**2 * rho
    return df


num_iter = 50 #number of iterations
M_vec = [] #mass vector
E_fb_vec = [] #fractional binding energy vector
R_vec = [] #Radius vector
for i in range(num_iter):
    tol = 1e-8 #tolerance
    p0 = (1e-4) + i*0.0001
    r0 = 10
    f0 = np.array([0,0,p0,0])

    while p0 > tol:
        sol = integrate.solve_ivp(TOV, t_span =[0, r0], y0=f0, method='RK45')
        p0 = sol.y[2,-1] #Last value of our solution
        if p0 > 0:
            r0 *= 1.0002 #Increasing r0
        elif p0 < 0:
            r0 *= 0.9998 #Decreasing r0
        else:
            break

    M = sol.y[0,-1] #mass
    M_p = sol.y[3,-1] #baryonic mass
    delta = -(M- M_p)/M #Fractional binding energy
    E_fb_vec.append(delta)
    M_vec.append(M)
    R = sol.t[-1] #Radius
    R = R*length_unit
    R_vec.append(R)

plt.title('Neutron Star M-R Dependence solution of TOV')
plt.xlabel('Radius in km')
plt.ylabel('Mass in solar mass')
plt.plot(R_vec, M_vec)
plt.show()

plt.figure()
plt.title('Neutron Star fractional binding energy vs. Radius')
plt.xlabel('Radius in km')
plt.ylabel('Fractional Binding Energy')
plt.plot(R_vec, E_fb_vec)
plt.show()