import numpy as np


class InteriorPointSolver:
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c

    def solve(self, eps=1e-3, theta=0.5):
        A = self.A
        c = self.c
        b = self.b

        m, n = A.shape
        x = np.ones((n, 1))
        y = np.ones((m, 1))
        s = np.ones((n, 1))

        while np.abs(x.T @ s) > eps:
            mu = theta * x.T @ s / n
            system_size = m + n + n
            A_all = np.zeros((system_size, system_size))
            A_all[:m, m:m + n] = A
            A_all[m:m + n, :m] = A.T
            A_all[m:m + n, m + n:] = np.eye(n)
            A_all[m + n:, m: m + n] = np.diag(s.reshape((n, )))
            A_all[m + n:, m + n:] = np.diag(x.reshape((n, )))

            b_all = np.zeros((system_size, 1))
            b_all[:m, ] = b - A @ x
            b_all[m:m + n, ] = c - s - A.T @ y
            b_all[m + n:] = mu * np.ones((n, 1)) - x * s
            res = np.linalg.solve(A_all, b_all)
            dy = res[:m, ]
            dx = res[m: m + n, ]
            ds = res[m + n:, ]

            alpha = 1
            if (dx < 0).any():
                w = - np.nan_to_num(x / dx)
                alpha = np.min([alpha, w[w > 0].min()])
            if (ds < 0).any():
                w = - np.nan_to_num(s / ds)
                alpha = np.min([alpha, w[w > 0].min()])
            y += alpha * dy
            x += alpha * dx
            s += alpha * ds

        return x, c.T @ x, None
