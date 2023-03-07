'''
Computes the simulation capacity of the general manifold

Based on original code by SueYeon Chung
'''

import numpy as np
import warnings

from cvxopt import solvers, matrix


def manifold_simcap_analysis(XtotT, n_rep, seed=0):
    '''
    Computes the simulation capacity of the given data

    Args:
        XtotT: Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
            of the space, and P_i is the number of sampled points for the i_th manifold.
        n_rep: Number of random draws to try at each feature dimension
        seed: Random seed

    Returns:
        asim: Simulation capacity
        P: Number of objects in XtotT
        Nc0: Number of features to separate with 0.5 chance
        N_vec: Values of N used in bisection search
        p_vec: Fraction of separable trials at each value of N
    '''
    # Get the number of objects and the total number of features
    P, N = len(XtotT), XtotT[0].shape[0]
    # Concatenate all the data and compute the global mean
    # Xori = np.concatenate(XtotT, axis=1)
    # global_mean = np.mean(Xori, axis=1, keepdims=True)
    # Subtract the global mean
    # Xtot0 = [x - global_mean for x in XtotT]
    Xtot0 = XtotT
    # Find the number of features for separability with 0.5 chance
    Nc, N_vec, p_vec = bisection_Nc_general(Xtot0, n_rep, 2, N, 0.05, seed)
    # Check if there was a solution, if so interpolate betweeen the boundaries for the capacity
    if Nc is np.nan:
        asim0 = np.nan
        Nc0 = np.nan
    else:
        # Find the boundary points
        below = np.max([i for i, p in enumerate(p_vec) if p < 0.5])
        above = np.min([i for i, p in enumerate(p_vec) if p >= 0.5])
        bounds = [below, above]
        # Interpolate between the bounds for the number of features where p=0.5
        N_vals = [N_vec[i] for i in bounds]
        p_vals = [p_vec[i] for i in bounds]
        Nc0 = np.interp(0.5, p_vals, N_vals)
        asim0 = P/Nc0
    return asim0, P, Nc0, N_vec, p_vec


def bisection_Nc_general(Xtot, n_rep, Nmin, Nmax, p_tol, seed, verbose=False):
    '''
    Performs a bisection search for the number of features such that the probability that the data
    is linearly separable is 0.5.  Implements the flag_n = 2 case from the original matlab code.

    Args:
        Xtot: Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
            of the space, and P_i is the number of sampled points for the i_th manifold.
        n_rep: Number of random draws to try at each feature number N
        Nmin: Minimum number of features to try
        Nmax: Maximum number of features to try
        p_tol:
        seed: Random seed

    Returns:
        Ncur: Number of features at the end of the bisection search
        Nall_vec: Every value of N tried during the search
        pall_vec: Computed separation probability at each value of N
    '''
    # Get the number of input objects
    P = len(Xtot)
    # Configure the separability check
    def create_f_pdiff(Xtot, n_rep, seed):
        def f_pdiff(N):
            return compute_sep_Nc_general(Xtot, N, n_rep=n_rep, seed=seed) - 0.5
        return f_pdiff
    f_pdiff = create_f_pdiff(Xtot, n_rep, seed)
    # Initialize the bisection search
    fmin = f_pdiff(Nmin)
    fmax = f_pdiff(Nmax)
    pmin_vec = [fmin + 0.5]
    pmax_vec = [fmax + 0.5]
    Nmin_vec = [Nmin]
    Nmax_vec = [Nmax]
    Ncur = int((Nmin + Nmax)/2 + 0.5)

    # Check that there is something to search over
    if pmax_vec[0] == 0:
        warnings.warn("Maximum N gives zero separability. Need more neurons.")
        Ncur = np.nan
        Nall_vec = np.nan
        pall_vec = np.nan

    # Check that the target value is between the max and the min
    if fmin * fmax > 0:
        warnings.warn("Wrong choice of Nmin and Nmax")
        print('fmin, fmax = ', fmin, fmax)
        Ncur = np.nan
        Nall_vec = np.nan
        pall_vec = np.nan

    # If there is something to seach over, do the bisection search
    if Ncur is not np.nan:
        # Check separability at this N
        fcur = f_pdiff(Ncur)
        # Set up ending conditions for the search
        err = np.abs(fcur)
        kk = 0
        dN = 1000
        # Search for the target value of Ncur
        Ncur_vec = []
        pcur_vec = []
        while err > p_tol and dN > 1 and kk < 100:
            kk += 1
            if verbose:
                print("{}th bisection run, P={}, Ncur={}, Nmin={}, pmin={}, Nmax={}, pmax={}".format(kk, P, Ncur, Nmin, fmin + 0.5, Nmax, fmax + 0.5))
            # Check that the target value is between the max and the current N value
            # Adjust the bounds of the search appropriately
            if fmin * fcur < 0:
                Nmax = Ncur
                fmax = fcur
            else:
                Nmin = Ncur
                fmin = fcur
            # Store results of this step
            pmin_vec.append(fmin + 0.5)
            pmax_vec.append(fmax + 0.5)
            Nmin_vec.append(Nmin)
            Nmax_vec.append(Nmax)
            # Get the next N to check
            Ncur = int((Nmin + Nmax)/2 + 0.5)
            fcur = f_pdiff(Ncur)
            err = np.abs(fcur)
            if verbose:
                print("err={}, p_tol={}".format(err, p_tol))
            dN = Nmax - Nmin
            Ncur_vec.append(Ncur)
            pcur_vec.append(fcur + 0.5)

        # Get the final quantities
        combined_quantities = [(n, pcur_vec[i]) for i, n in enumerate(Ncur_vec)]
        combined_quantities += [(n, pmin_vec[i]) for i, n in enumerate(Nmin_vec)]
        combined_quantities += [(n, pmax_vec[i]) for i, n in enumerate(Nmax_vec)]
        sorted_quantities = sorted(combined_quantities, key=lambda x: x[0])
        Nall_vec = [q[0] for q in sorted_quantities]
        pall_vec = [q[1] for q in sorted_quantities]
    return Ncur, Nall_vec, pall_vec


def compute_sep_Nc_general(Xtot, N_cur, n_rep, seed, reduced=True):
    '''
    Computes the separability of the input data using N_cur features. Only implements the
    flag_n = 2 case from the original matlab code.

    Args:
        Xtot: Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
            of the space, and P_i is the number of sampled points for the i_th manifold.
        N_cur: Number of features to use when checking linear separability
        n_rep: Number of random label assignments to try
        seed: Random seed
        reduced: Optionally use a smaller number of repetitions for large numbers of features.

    Returns
        p_conv: Fraction of the n_rep runs that were separable
    '''
    # Set the random seed
    np.random.seed(seed=seed)
    # Get the number of manifolds and dimensionality of data
    P, N = len(Xtot), Xtot[0].shape[0]
    # Use a smaller number of runs if the current number of features is high
    if N_cur > 1500 and reduced:
        n_rep = 5
    # Pick P/2 random objects to assign a positive label to for each repetition
    indpAll = [np.random.choice(range(P), size=P//2, replace=False) for i in range(n_rep)]
    # For each repetition, compute the separability of the randomly labeled data
    sep_vec = []
    for i in range(n_rep):
        # Create the label array
        indp = indpAll[i]
        labels00 = - np.ones((P))
        labels00[indp] = 1
        # Create a (normalized) random projection from N dimensions to N_cur dimensions
        try:
            W = np.random.randn(N, N_cur)
            W = W / np.sqrt(np.sum(np.square(W), axis=0, keepdims=True))
            # Project the data for each manifold into the lower dimensional space
            Xsub = [np.matmul(W.T, X) for X in Xtot]
            # Check separability in this subspace
            sep0, w0, bias0, margin0 = check_data_separability_general(Xsub, labels00)
            sep_vec.append(sep0)
        except ValueError as e:
            warnings.warn('Could not find solution')
            sep_vec.append(False)
    p_conv = np.mean(sep_vec)
    return p_conv


def check_data_separability_general(X, labels):
    '''
    Checks if a dichotomy of X given by labels is linearly separable.

    Args:
        X: Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
            of the space, and P_i is the number of sampled points for the i_th manifold.
        labels: Labels (+1 or -1). Should be a 1D array of shape (P) where P is number of manifolds.

    Returns:
        sep: Whether or not the dichotomy is linearly separable
        w: Weights of the optimal hyperplane
        bias: Bias for the separating plane
        margin: Size of margin
    '''
    # Get the indicies of the positive and negative labels
    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == -1]
    # Get the number of classes and feature dimensin
    P, N = len(X), X[0].shape[0]
    # Combine the data and labels
    X_tot = np.concatenate(X, axis=1)
    y_tot = np.concatenate([labels[i] * np.ones(x.shape[1]) for i, x in enumerate(X)])
    y_tot = y_tot.reshape(1, -1)
    assert X_tot.shape[1] == y_tot.shape[1]

    # Initialize weights and biases to zero
    w_ini = np.zeros((N, 1))
    bias_ini = 0
    # Set margin to zero
    kappa = 0
    # Set tolerance for solver
    tolerance = 1e-8
    # Find the optimal hyperplane
    sep, w, margin, flag, u, bias = find_svm_sep_primal_wb(X_tot, y_tot, w_ini, kappa=kappa, tolerance=tolerance, flag_wb=1)
    return sep, w, bias, margin


def find_svm_sep_primal_wb(X, y, w_ini, kappa=0, tolerance=1e-8, flag_wb=0):
    '''
    Finds the optimal separating hyperplane for data X given the dichotomy specified by y.
    The plane is defined by the vector w and is found by minimizing
        1/2 * w.T * w
    Subject to the constraint
        y * (x.T * w + b) >= 1
    For all data points, and an optional bias b.

    Args:
        X: Data matrix of shape (N, M) where N is the number of features, and M is the number of data points.
        y: Matrix of shape (1, M) containing the label for each of the M data points. Labels must be +1 or -1
        flag_wb: Option to include a bias.  Uses a bias if set to 1.

    Returns:
        sep: Whether or not the dichotomy is linearly separable
        w: Weights of the optimal hyperplane
        margin: Size of margin
        flag: Not used. 
        u: Unormalized weights of the optimal hyperplane
        bias: Bias for the separating plane
    '''
    # Configure the solver
    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 1000000
    solvers.options['feastol'] = tolerance
    solvers.options['abstol'] = tolerance
    solvers.options['reltol'] = tolerance

    # Get the shape of X
    M, N = X.shape[1], X.shape[0]
    # Verify there are the right number of labels and that they are +/- 1
    assert M == y.shape[1]
    assert all([np.abs(l[0]) == 1 for l in y])

    # Optionally add a constant component to X, otherwise plane is constrained to pass through the origin
    if flag_wb == 1:
        offset = np.ones((1, M))
    else:
        offset = np.zeros((1, M))
    Xb = np.concatenate([X, offset], axis=0)

    # Construct the input to the solver
    # Want to minimize 1/2 * w.T * P * w subject to the constrant that -y * X.T * w <= -1
    # P ignores the component of w that corresponds to offset, the constraint does not.

    # P should be identity with the final component set to zero
    P = np.identity(N + 1)
    P[-1, -1] = 0
    P = matrix(P)

    # q should be zero, (no term like q.T * w)
    q = np.zeros(N + 1)
    q = matrix(q)

    # Specify the constraint.  Ab is -y * X.T, bb is a vector of -1s
    Ab = - y * Xb # (N, M)
    Ab = matrix(Ab.T) # (M, N)
    bb = - np.ones(M)
    bb = matrix(bb)

    # Solve using cvxopt
    output = solvers.qp(P, q, Ab, bb)
    ub = np.array(output['x'])
    # Separate the bias
    u = ub[0:-1, 0]
    b = ub[-1, 0]
    # Normalize the outputs
    u_norm = np.linalg.norm(u)
    b /= u_norm
    w = u/u_norm
    # Compute the margin
    Pr = (np.matmul(w.T, X) + b)/np.linalg.norm(w.T)
    margin = np.min(y * Pr )
    # Check seperability
    seperable = np.all(np.sign(Pr) == y)
    return seperable, w, margin, 1, u, b
