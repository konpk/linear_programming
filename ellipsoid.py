import numpy as np
import matplotlib.pyplot as plt


class EllipsoidMethodSolver:
    """
    Implementation of the ellipsoid method for searching for
    a point belonging to a set, which is given in the form:
    Cx <= d.
    Solving LP problems is implemented using the
    sliding objective optimization
    """

    def __init__(self, C, d, c=None, eps=1e-5, start_r=None, n_iter=None):
        """
        Solver initialization
        :param C: constraint matrix
        :param d: constraint vector
        :param c: vector of coefficients of the objective function
        :param eps: parameter for computational stability,
                    values less than eps are taken equal to zero
        :param start_r: radius of the initial ellipsoid
        :param n_iter: number of iterations in the method
        :return: None
        """

        self.C = C
        self.d = d
        self.c = c
        self.eps = eps
        self.start_r = start_r
        self.n_iter = n_iter

    def draw_2d(self, A, x):
        """
        Method for drawing an ellipsoid in a two-dimensional case.
        The ellipsoid is given in the form:
        (y - x)^T A^{-1} (y - x) <= 1
        :param A: matrix defining the ellipsoid
        :param x: ellipsoid center
        :return: None
        """

        _, D, V = np.linalg.svd(A)
        a = 1 / np.sqrt(D[0])
        b = 1 / np.sqrt(D[1])

        theta = np.linspace(0, 2 * np.pi, 100)
        S = np.zeros((2, 100))
        S[0, :] = a * np.cos(theta)
        S[1, :] = b * np.sin(theta)
        S = V @ S
        S[0, :] += x[0]
        S[1, :] += x[1]

        plt.plot(S[0, :], S[1, :])

    def find_encoding_length(self, elements):
        """
        A function that determines the encoding length for
        a system of matrices and/or vectors
        :param elements: iterable object containing matrices and/or vectors
        :return: encoding length value
        """

        res = 0

        for e in elements:
            res += np.sum(np.round(np.log2(np.abs(e) + 1)) + 1)

        return res

    def solve(self, is_opt=False):
        """
        Function for finding a solution
        :is_opt: flag indicating what problem needs
                 to be solved (optimization or point search)
        :return: a point from a set or the optimal value of
                 variables depending on the value of the flag
        """

        C = self.C
        d = self.d
        current_opt = None
        current_opt_x = None

        _, n = C.shape

        n_iter = self.n_iter
        if n_iter is None:
            n_iter = 50 * (n + 1) ** 2 * self.find_encoding_length([C, d])
            n_iter = round(n_iter)
        start_r = self.start_r
        if start_r is None:
            r = n * 2 ** (2 * (self.find_encoding_length([C, d]) - n ** 2))
            start_r = max(r, 1)

        A = start_r * np.eye(n)
        x = np.zeros((n, 1))

        rho = 1 / (n + 1)
        sigma = n ** 2 / (n ** 2 - 1)
        tau = 2 / (n + 1)
        xi = 1 + 1 / (4 * (n + 1) ** 2)

        for _ in range(n_iter):
            if A.shape[0] == 2 and not is_opt:
                self.draw_2d(np.linalg.inv(A), x)

            c = None

            if np.all(C @ x <= d):
                if not is_opt:
                    plt.show()
                    return x
                else:
                    c = -self.c
                    if np.sqrt(c.T @ A @ c) > self.eps:
                        val = self.c.T @ x
                        if current_opt is None or val > current_opt:
                            current_opt = val
                            current_opt_x = x
                    else:
                        return x
            else:
                j = np.where(C @ x > d)[0][0]
                c = C[j, :].reshape((-1, 1))

            divider = c.T @ A @ c
            if np.isnan(np.sqrt(divider)) or divider < self.eps:
                if is_opt:
                    return current_opt_x
                else:
                    return x

            g = c / np.sqrt(divider)
            b = A @ g
            x = x - rho * b
            A = xi * sigma * ((A - tau * (b @ b.T)))

        if is_opt:
            return current_opt_x

        raise Exception
