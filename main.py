import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


def next_layer(u_prev, f,  alpha, beta, gamma, a,  h, tau, t_now):
    u = np.zeros(len(u_prev[0]))
    for i in range(1, len(u) - 1):
        u[i] = 2 * u_prev[1, i] - u_prev[0, i] + ((a * tau / h)**2) * (u_prev[1, i + 1] - 2 * u_prev[1, i] + u_prev[1, i - 1]) + tau**2 * f(i * h, t_now)
    u[0] = (gamma[0](t_now + tau) - beta[0] * u[1] / h) / (alpha[0] - beta[0] / h)
    u[-1] = (gamma[1](t_now + tau) + beta[1] * u[-2] / h) / (alpha[1] + beta[1] / h)
    return u


def first_order_solve():
    print("Hello world")


def second_order_solve():
    print("Hello world")


def main():
    x_min = 0.
    x_max = 1.
    t_min = 0.
    t_max = 10.
    h = 0.01
    N = int((x_max - x_min) // h + 1)  # number of points
    x_range = np.linspace(x_min, x_max, N)
    C = 1.
    a = np.sqrt(1/2)
    tau = C * h / a

    u0 = lambda x, t: t + x**2 + np.arcsin(t*(x-1) / 2)  # exact solution
    phi = lambda x: x**2  # first initial condition, u(x, 0)
    psi = lambda x: (x + 1) / 2  # second initial condition, u_t(x, 0)
    # u_tt = a**2 * u_xx + f(x, t)
    f = lambda x, t: -2 + ((2*t*((x-1)**3) - (t**3)*(x-1)) / ((4 - (t**2)*((x - 1)**2))**(3/2)))
    # alpha[0] * u(0, t) + beta[0] * u_x(0, t) = mu[0](t)
    # alpha[1] * u(1, t) + beta[1] * u_x(1, t) = mu[1](t)
    alpha = np.array([1, 1])
    beta = np.array([0, 2])
    gamma = np.array([lambda t: t - np.arcsin(t/2), lambda t: 5 + 2*t])

    u = np.zeros((3, N))
    u[0] = phi(x_range)
    u[1] = u[0] + tau * psi(x_range)
    # u[0] + tau * psi(x_range) + ((tau**2) / 2) * (a**2 * phi2(xrange) + f(xrange, 0))
    u[2] = next_layer(u[0:2], f, alpha, beta, gamma, a, h, tau, t_min + 2 * tau)

    fig = plt.figure()
    ax = plt.axes(xlim=(x_min, x_max), ylim=(-3, 3))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = x_range
        t_now = t_min + 2 * tau
        if i in range(3):
            y = u[i]
        else:
            t_now += tau
            u[0] = u[1]
            u[1] = u[2]
            u[2] = next_layer(u[0:2], f, alpha, beta, gamma, a, h, tau, t_now)
            y = u[2]
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=int((t_max - t_min) // tau), interval=20, blit=True)

    anim.save('solution.gif', writer='imagemagick')


if __name__ == '__main__':
    main()
