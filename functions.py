import math

import numpy as np


def solve(
    L=1,
    N=50,
    deltaT=5,
    T_ic=150,
    T_b1=100,
    T_b2=200,
    convergence_criteria=1e-5,
    k=16.2,
    Cp=500,
    rho=7750,
):
    # Boundary conditions
    T_bN = T_b2  # North boundary condition
    T_bE = T_bW = T_bS = T_b1  # East, West and South boundary condition

    alpha = k / (rho * Cp)

    deltaX = L / N
    deltaY = L / N

    T0 = np.zeros((N + 2, N + 2))
    T1 = np.zeros((N + 2, N + 2))

    # Initialize real cells with fixed value
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            T0[i, j] = T_ic

    # Initialize values in fictitious cells
    for i in range(1, N + 1):
        T0[i, N + 1] = 2 * T_bN - T0[i, N]
        T0[i, 0] = 2 * T_bS - T0[i, 1]
        T0[0, i] = 2 * T_bW - T0[1, i]
        T0[N + 1, i] = 2 * T_bE - T0[N, i]

    # variable to keep track of time
    time = 0
    # Dictionary to store temperature solution as value and time as key
    T_history = {time: T0}

    # Dictionary to store temperature solution with fictitious cells as value and time as key
    T_history_with_fictitious = {}

    # Fix T_rms with a value greater than the convergence criteria
    T_rms = convergence_criteria + 1

    # for i in range(1000):
    while T_rms > convergence_criteria:
        time += deltaT
        print("time = {}".format(time), end=" ")

        # Finding T n+1 for real cells
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                T1[i, j] = T0[i, j] + alpha * deltaT * (
                    ((T0[i + 1, j] + T0[i - 1, j] - 2 * T0[i, j]) / (deltaX ** 2))
                    + ((T0[i, j + 1] + T0[i, j - 1] - 2 * T0[i, j]) / (deltaY ** 2))
                )

        # Finding T n+1 for fictitious cells
        for i in range(1, N + 1):
            T1[i, N + 1] = 2 * T_bN - T1[i, N]
            T1[i, 0] = 2 * T_bS - T1[i, 1]
            T1[0, i] = 2 * T_bW - T1[1, i]
            T1[N + 1, i] = 2 * T_bE - T1[N, i]

        # Find T_rms from all real cells
        T_rms = 0
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                T_rms += (T0[i, j] - T1[i, j]) ** 2
        T_rms /= N ** 2
        T_rms = T_rms ** 0.5
        print("T_rms = {}".format(T_rms))

        # Replacing values at previous time step with the new values for next iteration
        T0 = T1.copy()
        # Storing solution of the real cells at current time level
        T_history[time] = T1[1 : N + 1, 1 : N + 1].copy()
        T_history_with_fictitious[time] = T1.copy()

        if time > 100000 * deltaT:
            print("Convergence criteria not met, stopping at time = {}".format(time))
            break

    output = {
        "T_history": T_history,
        "T_history_with_fictitious": T_history_with_fictitious,
        "latest_time": time,
        "N": N,
    }
    return output


def solveAnalyticSolution(L=1, N=10, T_b1=100, T_b2=200):
    pi = math.pi

    # x[i,j] and y[i,j] provide the position coordinates of (i,j) point
    x = np.zeros((N, N))
    y = np.zeros((N, N))
    for i in range(N):
        x[i, :] = i * (L / (N - 1))
    for j in range(N):
        y[:, j] = j * (L / (N - 1))

    # Number of series term to calculate analytical solution
    analytical_T_nmax = 100

    T = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            x_pos = x[i, j]
            y_pos = y[i, j]

            series_sum = 0
            for n in range(1, analytical_T_nmax):
                term1 = (((-1) ** (n + 1)) + 1) / n
                term2 = math.sin((n * pi * x_pos) / L)
                term3 = math.sinh((n * pi * y_pos) / L) / math.sinh((n * pi * L) / L)
                series_sum = series_sum + term1 * term2 * term3
            T[i, j] = T_b1 + (T_b2 - T_b1) * (2 / math.pi) * series_sum

    return T
