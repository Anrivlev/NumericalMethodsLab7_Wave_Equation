import numpy as np
import matplotlib.pyplot as plt


def least_squares(x, y):
    n = len(x)

    sumx = x.sum()
    sumy = y.sum()
    xy = x * y
    sumxy = xy.sum()
    xx = x * x
    sumxx = xx.sum()

    b = (n * sumxy - sumx*sumy) / (n * sumxx - sumx**2)
    a = (sumy - b * sumx) / n
    return a, b


def next_layer():
    u = np.array()

    return u

def main():
    x_min = 0.
    x_max = 1.
    t_min = 0.
    t_max = 10.
    h = 0.01
    C = 1.
    a = np.sqrt(1/2)
    tau = C * h / a

    u0 = lambda x, t: t + x**2 + np.arcsin(t*(x-1) / 2)  # exact solution
    ux0 = lambda x: x**2  # first initial condition, u(x, 0)
    utx0 = lambda x: (x + 1) / 2  # second initial condition, u_t(x, 0)
    # u_tt = a**2 * u_xx + f(x, t)
    f = lambda x, t: -2 + ((2*t*((x-1)**3) - (t**3)*(x-1)) / ((4 - (t**2)*((x - 1)**2))**(3/2)))
    # alpha[0] * u(0, t) + beta[0] * u_x(0, t) = mu[0](t)
    # alpha[1] * u(1, t) + beta[1] * u_x(1, t) = mu[1](t)
    alpha = np.array([1, 1])
    beta = np.array([0, 2])
    mu = np.array([lambda t: t - np.arcsin(t/2), lambda t: 5 + 2*t])


if __name__ == '__main__':
    print('PyCharm')
