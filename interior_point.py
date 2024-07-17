import numpy as np


class InteriorPointSolver:
    """
    Implementation of the interior point method in
    matrix form for solving LP problems such as:
    c^T x -> min, Ax = b, x >= 0
    """
    
    def __init__(self, A, b, c):
        """
        Solver initialization
        :param A: constraint matrix
        :param b: constraint vector
        :param c: vector of coefficients of the objective function
        :return: None
        """
        
        self.A = A
        self.b = b
        self.c = c

    def solve(self, eps=1e-3, theta=0.5):
        """
        Function for finding a solution
        :param eps: parameter for computational stability,
                    values less than eps are taken equal to zero
        :param theta: parameter defining the displacement value
                      for the Newton method
        :return: the optimal value of the variables and the objective function,
                 as well as None to match the format with the simplex method
        """

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
