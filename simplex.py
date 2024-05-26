import numpy as np


class SimplexSolver:
    def __init__(self, A, b, c, eps=1e-5, is_slack=False):
        self.A = A
        self.b = b
        self.c = c
        self.eps = eps
        self.is_slack = is_slack

    def solve(self):
        A = self.A
        c = self.c
        b = self.b

        c = c.reshape(-1)
        b = b.reshape(-1)

        m, n = A.shape
        basic_indices, non_basic_indices = [], []

        if self.is_slack:
            basic_indices = np.array([i for i in range(n - m, n)])
            non_basic_indices = np.array([i for i in range(n - m)])
        else:
            A_new = np.concatenate((A, np.eye(m)), axis=1)
            c_new = np.zeros(c.shape[0] + m)
            c_new[-m:] = 1
            prev_problem = SimplexSolver(
                A_new,
                b.reshape(-1, 1),
                c_new.reshape(-1, 1),
                is_slack=True
                )
            _, val, res_idxs = prev_problem.solve()

            if np.abs(np.sum(val)) > self.eps:
                raise Exception

            basic_indices = np.array(res_idxs)
            basic_s = set(basic_indices)
            non_basic_indices = [i for i in range(n) if i not in basic_s]
            non_basic_indices = np.array(non_basic_indices)

        while True:
            B_inv = np.linalg.inv(A[:, basic_indices])
            d = B_inv @ b
            c_tilde = np.zeros(n)
            D = c[basic_indices] @ B_inv @ A[:, non_basic_indices]
            c_tilde[non_basic_indices] = c[non_basic_indices] - D
            c_min = np.min(c_tilde)
            j_min = np.argmin(c_tilde)

            if c_min >= -self.eps:
                if self.is_slack:
                    while np.max(basic_indices) >= n - m:
                        j_min = np.max(basic_indices)
                        B_inv = np.linalg.inv(A[:, basic_indices])
                        W = B_inv @ A
                        i_min = -1
                        for i, e in enumerate(W[basic_indices == j_min, :][0]):
                            if np.abs(e) > self.eps and i < n - m:
                                i_min = i
                                break
                        k = j_min
                        basic_indices[basic_indices == j_min] = i_min
                        non_basic_indices[non_basic_indices == i_min] = k

                x_res = np.zeros((n, 1))
                x_res[basic_indices, 0] = d
                return x_res, c @ x_res, basic_indices
            w = B_inv @ A[:, j_min]
            good_idxs = np.where(w > self.eps)[0]
            next_i = np.argmin(d[good_idxs] / w[good_idxs])
            i_min = good_idxs[next_i]
            k = basic_indices[i_min]
            basic_indices[i_min] = j_min
            non_basic_indices[non_basic_indices == j_min] = k
