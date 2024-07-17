import time
import numpy as np
from numpy.random import choice
import pickle as pkl
import interior_point
import ellipsoid
import simplex
import sympy
from networkx.algorithms.bipartite import random_graph, maximum_matching
from networkx.linalg import incidence_matrix
import networkx as nx
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import optimize
from scipy.io.matlab import loadmat


def generate_bipartite_graph(n, p, count):
    """
    Generates a random bipartite graphs and constructs their adjacency matrix.
    Saves the result in pickle format.
    :param n: number of vertices in one part
    :param p: probability of an edge between two vertices
    :param count: number of graphs for generation
    :return: None
    """

    matrices = []
    graphs = []

    for _ in range(count):
        g = random_graph(n, n, p)
        A = incidence_matrix(g).toarray()
        A = A[~np.all(A == 0, axis=1)]
        matrices.append(A)
        graphs.append(g)

    with open(f"./bipartite_examples/bg_{n}.pickle", "wb") as f:
        pkl.dump({"matrices": matrices, "graphs": graphs}, f)


def calculate_maximum_matching_time(vertexes_count, method="simplex"):
    """
    Calculates the average time to solve a relaxation of
    maximum matching problem for a set of graphs of a certain size
    :param vertexes_count: number of vertices in one part
    :param method: method to solve (one of the simplex, ip or ellipsoid)
    :return: None
    """

    matrices = None
    graphs = []

    with open(f"./bipartite_examples/bg_{vertexes_count}.pickle", "rb") as f:
        d = pkl.load(f)
        matrices = d["matrices"]
        graphs = d["graphs"]

    ts = []
    opt_res = own_res = 0
    for i, A in enumerate(matrices):
        g = graphs[i]
        u = [k for k in g.nodes if g.nodes[k]['bipartite'] == 0]
        opt_res = maximum_matching(g, top_nodes=u)
        m, n = A.shape
        problem = None
        if method == "ellipsoid":
            A = np.concatenate((A, -np.eye(n)), axis=0)
            c = np.array([1] * n)
            c = c.reshape((-1, 1))
            b = np.array([1] * m + [0] * n)
            b = b.reshape((-1, 1))
            problem = ellipsoid.EllipsoidMethodSolver(A, b, c=c)
            start = time.time()
            x = problem.solve(is_opt=True)
            finish = time.time()
            ts.append(finish - start)
            own_res = (c.T @ x)[0][0]
            print(f"Own method result on {i} iter: {own_res}")
            print(f"Optimum value on {i} iter: {len(opt_res) // 2}")
        else:
            A = np.concatenate((A, np.eye(m)), axis=1)
            b = np.ones((m, 1))
            c = np.array([-1] * n + [0] * m)
            c = c.reshape((m + n, 1))
            if method == "simplex":
                problem = simplex.SimplexSolver(A, b, c)
            elif method == "ip":
                problem = interior_point.InteriorPointSolver(A, b, c)
            start = time.time()
            _, val, _ = problem.solve()
            finish = time.time()
            ts.append(finish - start)
            print(f"Own method result on {i} iter: {-val.sum()}")
            print(f"Optimum value on {i} itee: {len(opt_res) // 2}")

    ts = np.array(ts)
    res = None

    with open(f"maximum_matching_{method}_res.pickle", "rb") as f:
        res = pkl.load(f)["res"]

    res[vertexes_count] = ts

    with open(f"maximum_matching_{method}_res.pickle", "wb") as f:
        pkl.dump({"res": res}, f)

    print(f"Mean time {ts.mean()}")
    print(f"Time std: {ts.std()}")


def generate_flow_graph(n, p, count):
    """
    Generates a random directed graphs for max flow
    problem and constructs their adjacency matrix.
    Saves the result in pickle format.
    :param n: number of vertices in graph
    :param p: probability of an edge between two vertices,
              except first and last
    :param count: number of graphs for generation
    :return: None
    """

    matrices = []
    graphs = []
    c_vecs = []

    for _ in range(count):
        vertexes = [i for i in range(n)]
        edges = []
        costs = [i for i in range(5, 11)]
        c_vec = []
        g = nx.DiGraph()
        for v in vertexes[:-1]:
            for w in vertexes[v + 1:]:
                prob = p
                if v == 0 or w == n - 1:
                    prob += 0.2
                var = choice([0, 1], p=[1 - prob, prob])
                if var == 1:
                    c = choice(costs)
                    c_vec.append(c)
                    g.add_edge(f"{v}", f"{w}", capacity=c)
                    edges.append((f"{v}", f"{w}"))

        A = incidence_matrix(
            g,
            nodelist=[f"{i}" for i in range(n)],
            edgelist=edges, oriented=True
            ).toarray()

        matrices.append(A)
        graphs.append(g)
        c_vecs.append(c_vec)

    with open(f"./flow_examples/fg_{n}.pickle", "wb") as f:
        pkl.dump({"matrices": matrices, "graphs": graphs, "c_vecs": c_vecs}, f)


def calculate_maximum_flow_time(vertexes_count, method="simplex"):
    """
    Calculates the average time to solve a relaxation of maximum flow problem
    for a set of graphs of a certain size
    :param vertexes_count: number of vertices in graph
    :param method: method to solve (one of the simplex, ip or ellipsoid)
    :return: None
    """

    matrices = None
    graphs = None
    c_vecs = None

    with open(f"./flow_examples/fg_{vertexes_count}.pickle", "rb") as f:
        d = pkl.load(f)
        matrices = d["matrices"]
        graphs = d["graphs"]
        c_vecs = d["c_vecs"]

    ts = []
    opt_res = 0
    for i, A in enumerate(matrices[:20]):
        g = graphs[i]
        try:
            opt_res, _ = nx.maximum_flow(g, "0", f"{vertexes_count - 1}")
        except Exception:
            pass
        m, n = A.shape
        problem = None
        if method == "ellipsoid":
            f_0 = np.zeros(m)
            f_0[0] = 1
            f_0[-1] = -1
            f_0 = f_0.reshape(-1, 1)
            A = np.concatenate((A, f_0), axis=1)
            A = np.delete(A, m - 1, 0)
            E = np.concatenate((np.eye(n), np.zeros(n).reshape(-1, 1)), axis=1)
            A = np.concatenate((A, E), axis=0)
            A = np.concatenate((A, -np.eye(n + 1)), axis=0)
            b = np.zeros(m + 2 * n)
            b[:m - 1] = 1e-3
            b[m - 1:-n - 1] = c_vecs[i]
            b = b.reshape(-1, 1)
            c = np.zeros(n + 1)
            c[n] = 1
            c = c.reshape(-1, 1)
            problem = ellipsoid.EllipsoidMethodSolver(A, b, c=c)
            start = time.time()
            x = problem.solve(is_opt=True)
            finish = time.time()
            ts.append(finish - start)
            print(f"Own method result on {i} iter: {c.T @ x}")
            print(f"Optimum value on {i} iter: {opt_res}")
        else:
            f_0 = np.zeros(m)
            f_0[0] = 1
            f_0[-1] = -1
            f_0 = f_0.reshape(-1, 1)
            A = np.concatenate((A, f_0), axis=1)
            E = np.eye(n)
            Z = np.zeros((m, n))
            Z = np.concatenate((Z, E), axis=0)
            E = np.concatenate((E, np.zeros(n).reshape(-1, 1)), axis=1)
            A = np.concatenate((A, E), axis=0)
            A = np.concatenate((A, Z), axis=1)
            b = np.zeros(m + n)
            b[m:] = c_vecs[i]
            b = b.reshape(-1, 1)
            c = np.zeros(2 * n + 1)
            c[n] = -1
            c = c.reshape(-1, 1)
            A = np.delete(A, m - 1, 0)
            b = np.delete(b, m - 1, 0)
            if method == "simplex":
                problem = simplex.SimplexSolver(A, b, c)
            elif method == "ip":
                problem = interior_point.InteriorPointSolver(A, b, c)
            start = time.time()
            _, val, _ = problem.solve()
            finish = time.time()
            ts.append(finish - start)
            print(f"Own method result on {i} iter: {-val}")
            print(f"Optimum value on {i} iter: {opt_res}")

    ts = np.array(ts)
    res = None

    with open(f"maximum_flow_{method}_res.pickle", "rb") as f:
        res = pkl.load(f)["res"]

    res[vertexes_count] = ts

    with open(f"maximum_flow_{method}_res.pickle", "wb") as f:
        pkl.dump({"res": res}, f)

    print(f"Mean time {ts.mean()}")
    print(f"Time std: {ts.std()}")


def netlib_calculate_time(path, method="simplex", count=5):
    """
    Calculates the average time to solve problem from netlib.
    Problems have .mat format and look like
    min   c'x
    s.t.  Ax = b,
    lbounds <= x <= ubounds

    :param path: path to the .mat file
    :param method: method to solve (one of the simplex, ip,
                   highs-ip or highs-ds)
    :param count: number of runs to evaluate
    :return: None
    """

    problem_statement = loadmat(path)
    A = problem_statement["A"].toarray().astype('float64')
    b = problem_statement['b'].astype('float64')
    c = problem_statement['c'].astype('float64')
    lb = problem_statement['lbounds']
    ub = problem_statement['ubounds']

    eps = 1e-5
    data_inf = 1e32
    m, n = A.shape

    relationships_plus = []
    b_plus = []
    for i, var in enumerate(lb):
        if var > eps:
            print(f"lb {var}")
            r = [0] * n
            r[i] = -1
            relationships_plus.append(r)
            b_plus.append(-var)
    for i, var in enumerate(ub):
        if np.abs(var - data_inf) > eps:
            print(f"ub {var}")
            r = [0] * n
            r[i] = 1
            relationships_plus.append(r)
            b_plus.append(var)

    if relationships_plus:
        relationships_plus = np.array(relationships_plus)
        b_plus = np.array(b_plus)
        k, _ = relationships_plus.shape
        relationships_plus = np.concatenate(
            (relationships_plus, np.eye(k)),
            axis=1
            )
        A = np.concatenate((A, np.zeros((m, k))), axis=1)
        A = np.concatenate((A, relationships_plus), axis=0)
        b = np.concatenate((b, b_plus), axis=0)
        c = np.concatenate((c, np.array([[0] for _ in range(k)])), axis=0)

    ts = []

    if "highs" in method:
        for i in range(count):
            start = time.time()
            res = optimize.linprog(
                c=c.reshape(-1),
                A_eq=A,
                b_eq=b.reshape(-1),
                bounds=(0, None),
                method=method
            )
            finish = time.time()
            ts.append(finish - start)
            if i == 0:
                print(f"{method} result: {res}")
        ts = np.array(ts)
        print(f"Mean {ts.mean()}")
        print(f"Std: {ts.std()}")
        return

    print(f"negative components in b: {(b < 0).sum()}")
    A[(b < 0).reshape(-1), :] *= -1.0
    b[b < 0] *= -1.0

    print(f"A shape: {A.shape}; rank(A) = {np.linalg.matrix_rank(A)}")
    if A.shape[0] > np.linalg.matrix_rank(A):
        _, inds = sympy.Matrix(A).T.rref()
        A = A[inds, :]
        b = b[inds, :]

    for i in range(count):
        problem = None
        if method == "simplex":
            problem = simplex.SimplexSolver(A, b, c)
        elif method == "ip":
            problem = interior_point.InteriorPointSolver(A, b, c)
        start = time.time()
        _, val, _ = problem.solve()
        finish = time.time()
        ts.append(finish - start)
        if i == 0:
            print(f"{method} result: {val.sum()}")

    ts = np.array(ts)
    print(f"Mean {ts.mean()}")
    print(f"Std: {ts.std()}")


def classification_example():
    """
    An example of classification using SVM as an LP problem
    """

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"sklearn SVM accuracy: {accuracy_score(y_test, y_pred)}")

    m, n = X_train.shape
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    A = np.zeros((m, 2 * m + 2 * n + 2))
    for i in range(m):
        A[i, :n] = X_train[i] * y_train[i]
        A[i, n: 2 * n] = -X_train[i] * y_train[i]
        A[i, 2 * n] = y_train[i]
        A[i, 2 * n + 1] = -y_train[i]
    A[:, 2 * n + 2: 2 * n + m + 2] = -np.eye(m)
    A[:, 2 * n + m + 2:] = np.eye(m)
    b = np.ones((m, 1))
    c = np.zeros((2 * m + 2 * n + 2, 1))
    c[2 * n + m + 2:, 0] = np.ones(m)
    problem = simplex.SimplexSolver(A, b, c)
    sol, _, _ = problem.solve()
    w = sol[:n, 0] - sol[n: 2 * n, 0]
    b = sol[2 * n, 0] - sol[2 * n + 1, 0]
    y_pred = np.ones(y_test.shape)
    y_pred[X_test @ w + b < 0] = -1
    print(f"LP SVM accuracy: {accuracy_score(y_test, y_pred)}")


def main():
    classification_example()


if __name__ == '__main__':
    main()
