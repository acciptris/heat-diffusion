########################################################

# Code can be run by running the command 'python solve.py'

# Packages required: Numpy, Matplotlib

########################################################

import matplotlib.pyplot as plt
import numpy as np

import functions

# Length of edge of square domain
L = 1

# Number of cells on each side
N = 50

# North boundary condition
T_b2 = 200

# East, West and South boundary condition
T_b1 = 100

# Initial value at real cells
T_ic = 150

# Plate material properties
k = 16.2
Cp = 500
rho = 7750

# Convergence criteria
C = 1e-5


analyticSolution = functions.solveAnalyticSolution(L, N, T_b1, T_b2)
numericalSolution = functions.solve(
    L, N, T_ic=T_ic, T_b1=T_b1, T_b2=T_b2, convergence_criteria=C, k=k, Cp=Cp, rho=rho
)


############# POST PROCESSING ###############


# Plotting contour plots of solution
contour_levels = np.linspace(50, 250, 20)

latest_time = numericalSolution["latest_time"]
T_history = numericalSolution["T_history"]
T = T_history[latest_time]

plt.figure()
plt.contourf(T.transpose(), levels=contour_levels)
plt.colorbar()
plt.savefig("figures/Temperature Contour plot.png")


# Plotting solution at y=0.5
plt.figure()
deltaX = L / N
x = np.linspace(deltaX / 2, 1 - (deltaX / 2), N)

plt.plot(x, T[:, N // 2], label=f"N={N}")
plt.plot(
    x, analyticSolution[:, N // 2], label=f"N={N} (Analytic solution)",
)
plt.xlabel("x")
plt.ylabel("Temperature")
plt.legend()
plt.savefig("figures/y_0.5.png")


# Plotting solution at x=0.5
plt.figure()
deltaY = L / N
y = np.linspace(deltaY / 2, 1 - (deltaY / 2), N)

plt.plot(y, T[N // 2, :], label=f"N={N}")
plt.plot(
    y, analyticSolution[N // 2, :], label=f"N={N} (Analytic solution)",
)
plt.xlabel("y")
plt.ylabel("Temperature")
plt.legend()
plt.savefig("figures/x_0.5.png")
