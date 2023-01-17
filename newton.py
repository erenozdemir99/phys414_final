import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate



# PART B

def read_csv_data(filename):
    """reads the white_dwarf.csv file and returns the logg and mass """
    mass = []
    logg = []
    with open(filename, 'r') as file:
        data = csv.reader(file)
        header = next(data)
        for row in data:
            mass.append(float(row[2]))
            logg.append(float(row[1]))

    mass = np.array(mass)
    logg = np.array(logg)
    return logg, mass

def lane_emden(ksi, f):
    """Lane-emden equation written in differential form. f = [theta, ksi]"""
    df = np.zeros(2)
    if ksi == 0:
        df = np.array([0,1])
    else:
        df[0] = f[1]
        df[1] = (-2/ksi) * f[1] - f[0]**(3/2)
    return df

# Reading the file
file = "white_dwarf_data.csv"
logg, mass = read_csv_data(file)

n = len(logg)
R = np.zeros(n)
G = 6.67430 * 1e-11 #in CSV units
solar_mass = 1.9891 * 1e30 # solarmass
G_solar = G * solar_mass # G in scaled units
radius_earth = 6.38 * 10**6
for i in range(n):
    radius = np.sqrt(G_solar * mass[i]/(10**(logg[i]-2)))
    radius = radius/radius_earth

    R[i] = radius
print(R)

plt.scatter(R, mass)
plt.title('M-R graph')
plt.xlabel('Radius (in average Earth radius)')
plt.ylabel('Mass (in Solar mass)')
plt.show()




#PART C




# Curve fitting
# We take logR = 0.5 as our cutoff value, so R = 1.64

R_new = []

mass_new = []
for i in range(len(R)):
    if R[i] >= 1.64:
        R_new.append(np.log(R[i]))
        mass_new.append(np.log(mass[i]))
R_new = np.array(R_new)
mass_new = np.array(mass_new)


p = np.polyfit(R_new, mass_new, 1)

x = np.linspace(0.5, 1, 100)
poly = np.polyval(p,x)
slope = poly[1] #slope of the fit
n = (slope - 3)/(slope - 1) # n value is solved from the slope
y_intercept = poly[0] # y intersection


#Plotting logM vs. logR
plt.figure()
plt.title('logM-logR graph')
plt.xlabel('logR')
plt.ylabel('logM')
plt.scatter(np.log(R), np.log(mass))
plt.plot(x, poly, 'r')
plt.show()

##Applying shooting method to solve Lane-Emden
#First solution is guessed as 2.45 from taylor expansion

tol = 1e-3 #tolerance
ksi0 = 2.45 #Initial guess
f0 = np.array([1,0])
theta = 1 #at the center
while theta > tol:
    sol = integrate.solve_ivp(lane_emden, t_span =[0, ksi0], y0=f0, method='RK45')
    theta = sol.y[0,-1] #Last value of our solution
    if theta > 0:
        ksi0 *= 1.02 #Increasing ksi0
    elif theta < 0:
        ksi0 *= 0.98 #Decreasing ksi0
    else:
        break



z = sol.y[1,-1] # z at the ksi_final (z =dtheta/dksi)
N = ((4*np.pi)**(1/n)/(n+1)) * (-ksi0**2 * z)**((1-n)/n) * ksi0**((n-3)/n)
print(N)
K = np.exp(y_intercept * ((n-1)/n)) * (G_solar/(radius_earth)**3) * N
print(K)


