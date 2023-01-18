import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate

#PART A and B
k = 100 #constant of the Polytrope
length_unit = 1.477  # in km
solar_mass = 1.989 * 1e30 #Solar mass in kg

def TOV(r, f):
    """returns the system of ODEs for Tolman-
Oppenheimer-Volkoff (TOV) equations: f = [m ,v, p, m_p]"""
    df = np.zeros(4)

    if r == 0: # If we are at the center, we should arrange the initial conditions manually
        df = np.array([0,0,0,0])
    else:
        rho = np.sqrt(f[2]/k)
        # TOV equations
        df[0] = 4*np.pi*r**2 * rho
        df[1] = 2* (f[0]+4*np.pi*r**3 *f[2])/(r*(r-2*f[0]))
        df[2] = (-1/2) * (f[2] + rho) * df[1]
        df[3] = 4*np.pi*np.sqrt(1/(1-2*(f[0]/r))) * r**2 * rho
    return df


num_iter = 50 #number of iterations
M_vec = [] #mass vector
E_fb_vec = [] #fractional binding energy vector
R_vec = [] #Radius vector
rho_c_vec =[] #Central density
for i in range(num_iter):
    tol = 1e-8 #tolerance
    p0 = (1e-4) + i*0.0001 # Initial pressure
    r0 = 10 #Our guess for the R
    f0 = np.array([0,0,p0,0]) #Initial conditions

    ## Applying the shooting method
    while p0 > tol:
        sol = integrate.solve_ivp(TOV, t_span =[0, r0], y0=f0, method='RK45')
        p0 = sol.y[2,-1] #Last value of our solution
        if p0 > 0:
            r0 *= 1.002 #Increasing r0
        elif p0 < 0:
            r0 *= 0.998 #Decreasing r0
        else:
            break
    P = sol.y[2,0] #Pressure at 0
    rho_c = np.sqrt(P/k) #Central density
    rho_c = rho_c * (solar_mass/(length_unit * 10**3)**3)#unit conversion
    rho_c_vec.append(rho_c)
    M = sol.y[0,-1] #mass
    M_p = sol.y[3,-1] #baryonic mass
    delta = -(M- M_p)/M #Fractional binding energy
    E_fb_vec.append(delta)
    M_vec.append(M)
    R = sol.t[-1] #Radius
    R = R*length_unit
    R_vec.append(R)

#plotting the results
plt.title('Neutron Star M-R Dependence solution of TOV')
plt.xlabel('Radius in km')
plt.ylabel('Mass in solar mass')
plt.plot(R_vec, M_vec)
plt.show()

#plotting the results
plt.figure()
plt.title('Neutron Star fractional binding energy vs. Radius')
plt.xlabel('Radius in km')
plt.ylabel('Fractional Binding Energy')
plt.plot(R_vec, E_fb_vec)
plt.show()

#PART C
#plotting the results
# Find the stable region
index = 1
#Checking were the derivative is 0
for i in range(1,len(rho_c_vec)):
    dM = M_vec[i+1]-M_vec[i]
    index = i
    if dM <tol:
        break

print(M_vec[index])
#Dividing solutions into stable and unstable parts
M_vec_unstable = M_vec[0:index+1]
M_vec_stable = M_vec[index:]
rho_c_vec_unstable = rho_c_vec[0:index+1]
rho_c_vec_stable = rho_c_vec[index:]

###Plotting Mass vs. Central Density
plt.figure()
plt.title('Neutron Star Mass vs. Central Density')
plt.xlabel('Central Density (kg/m^3')
plt.ylabel('Mass in solar mass')
plt.plot(rho_c_vec_stable, M_vec_stable, label='stable')
plt.plot(rho_c_vec_unstable, M_vec_unstable, '--', label='unstable')
plt.legend()
plt.show()
