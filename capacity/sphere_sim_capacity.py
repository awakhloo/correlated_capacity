import numpy as np
from cvxopt import solvers, matrix
from tqdm import tqdm 
import warnings
from scipy.linalg import qr, cholesky

'''
This is a script for numerically calculating the capacity using SY Chung's MCP (simple) algorithm.
Here we let
P = number of manifolds 
N = ambient dim 
K = intrinsic dim of the spheres 
'''

def get_axes(N, P, K, lam0, lam1, psi0, psi1): 
    '''
    make some sphere axes with homogenous correlations
    '''
    # axes
    ax_mat = (lam0 - lam1) * np.eye(P) + lam1 * np.ones((P,P))
    L = cholesky(ax_mat, lower = True)
    all_axes = [] 
    Phis = 1/np.sqrt(N) * np.random.randn(P, N, K+1) 
    for k in range(K): 
        Phi = Phis[..., k]
        ax = L @ Phi
        all_axes.append(ax) 
    # centers
    c_mat = (psi0 - psi1) * np.eye(P) + psi1 * np.ones((P, P))
    L_cent = cholesky(c_mat, lower = True)
    Phi = Phis[..., K] 
    cents = L_cent @ Phi
    all_axes.append(cents)
    all_axes = np.stack(all_axes, axis = -1)
    assert all_axes.shape == (P, N, K+1)
    return all_axes

def sphere_simcap(sphere_axes, n_rep, seed=0):
    '''
    Computes the simulation capacity of the given data

    Args:
        sphere_axes: P X N X K+1 array of sphere axes, with the final rightmost dimension corresponding to centroid
        seed: Random seed

    Returns:
        asim: Simulation capacity
        P: Number of objects in XtotT
        Nc0: Number of features to separate with 0.5 chance
        N_vec: Values of N used in bisection search
        p_vec: Fraction of separable trials at each value of N
    '''
    # Get the number of objects and the total number of features
    P, N, K = sphere_axes.shape
    # Find the number of features for separability with 0.5 chance
    Nc, N_vec, p_vec = bisection_Nc_general(sphere_axes, n_rep, 2, N, 0.05, seed)
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


def bisection_Nc_general(sphere_axes, n_rep, Nmin, Nmax, p_tol, seed, verbose=False):
    '''
    Performs a bisection search for the number of features such that the probability that the data
    is linearly separable is 0.5.  Implements the flag_n = 2 case from the original matlab code.

    Args:
        sphere_axes: P X N X K+1 array of sphere axes, with the final rightmost dimension corresponding to centroid
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
    P, N, K1 = sphere_axes.shape 
    # Configure the separability check
    def create_f_pdiff(sphere_axes, n_rep, seed):
        def f_pdiff(N):
            return compute_sep_Nc_general(sphere_axes, N, n_rep=n_rep, seed=seed) - 0.5
        return f_pdiff
    f_pdiff = create_f_pdiff(sphere_axes, n_rep, seed)
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


def compute_sep_Nc_general(sphere_axes, N_cur, n_rep, seed, reduced=False):
    '''
    Computes the separability of the input data using N_cur features. Only implements the
    flag_n = 2 case from the original matlab code.

    Args:
        sphere_axes: P X N X K+1 array of sphere axes, with the final rightmost dimension corresponding to centroid
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
    P, N, K1 = sphere_axes.shape 
    # Use a smaller number of runs if the current number of features is high
    if N_cur > 1500 and reduced:
        n_rep = 5
    # Pick P/2 random objects to assign a positive label to for each repetition
    indpAll = [np.random.choice(range(P), size=P//2, replace=False) for i in range(n_rep)]
    # For each repetition, compute the separability of the randomly labeled data
    sep_vec = []
    print('On ', N_cur, ' Features') 
    for i in tqdm(range(n_rep)):
        # Create the label array
        indp = indpAll[i]
        labels00 = - np.ones((P))
        labels00[indp] = 1
        # Create a (normalized) random projection from N dimensions to N_cur dimensions
        try:
            sep0 = check_separability(sphere_axes, N_cur, labels00)
            sep_vec.append(sep0)
        except ValueError as e:
            warnings.warn('Could not find solution')
            sep_vec.append(False)
    p_conv = np.mean(sep_vec)
    return p_conv

def check_separability(sphere_axes, N_cur, labels, num_init_samples=20): 
    '''
    Project sphere axes into a lower dimensional space and check the resulting separability
    '''
    P, N, K1 = sphere_axes.shape 
    K = K1-1
    W = np.random.randn(N, N_cur)
    W = W / np.sqrt(np.sum(np.square(W), axis=0, keepdims=True))
    sphere_axes = np.einsum('ijk, jl -> ilk', sphere_axes, W)
    cent, ax = sphere_axes[..., -1], sphere_axes[..., :-1]
    # draw some samples from the surface of the sphere: 
    spheres = np.random.randn(P, K, num_init_samples)
    spheres = spheres/np.sqrt(np.sum(spheres**2, 1, keepdims = True))
    # make the manifold samples and reformat as a dict for later ragged arrays:
    samples = cent[..., None] + np.sum(spheres[:, None, :, :] * ax[..., None], axis = 2) # sum over intrinsic dim. shape is now P, N, #samples 
    assert samples.shape == (P, N_cur, num_init_samples)
    samples = {i : samples[i] for i in range(P)}
    return sep_spheres(sphere_axes, labels, samples, kappa=0., max_iter=300, tol = 1e-6)
    
def sep_spheres(sphere_axes, labels, samples, kappa, max_iter, tol=0): 
    '''
    Given some sphere axes and initial samples, check if a random dichotomy is separable using SYC's MCP algo
    Args: 
    - sphere_axes: array of shape P x N x (K+1) with the centroids as the last dimension 
    - labels: random vector of manifold labels
    - samples: initial batch of samples. A dictionary with keys=np.arange(P) and values of arrays of shape 
    - kappa: margin
    
    returns: 
    - is_separable: boolean of separability 
    '''
    for i in range(max_iter): 
        feasible, w, b = get_w(labels, samples, kappa)
        if i % 10 == 0: 
            print(f'On iter {i}')
        if not feasible: 
            return False #, w, b
        sat, viol_mfs, viol_samples, margins = check_constraint(w, b, labels, sphere_axes, kappa, tol)
        if sat: 
            return True #, viol_mfs, viol_samples, margins, w
        # augment the samples if we didn't separate within tolerance and try again
        for i, mu in enumerate(viol_mfs): 
            samples[mu] = np.concatenate([samples[mu], viol_samples[i].reshape(-1, 1)], axis = 1)
    warnings.warn('Max iter reached. Defaulting to separable')
    return True
     
    
def get_w(labels, samples, kap):
    '''
    Given some samples of the spheres, try to calculate a solution vector 
    Args: 
    - labels: vector of shape P with values in {±1}
    - samples: dictionary with keys from 1 to P and with array values of shape N X #Samples 
    Returns: 
    - separating vector
    '''
    ## sort samples into plus or minus labels 
    plus, minus = [], []
    num_s = [] 
    N = samples[0].shape[0] 
    for mu in range(len(labels)):
        if labels[mu] == 1 : 
            plus.append(samples[mu].reshape(N, -1))
        elif labels[mu] == -1: 
            minus.append(samples[mu].reshape(N, -1))
        else: 
            raise RuntimeError("labels aren't in ±1")
        num_s.append(samples[mu].shape[-1])
    plus = np.concatenate(plus, axis = -1)
    minus = np.concatenate(minus, axis = -1)
    X = np.concatenate([plus, minus], axis = -1)
    # make a new vector of labels for this arrangement of data 
    num_plus, num_minus = plus.shape[-1], minus.shape[-1]
    lab = np.concatenate([np.ones(num_plus), -np.ones(num_minus)]).reshape(1, -1)
    feasible, w, margin, u, b = find_svm_sep_primal_wb(X, lab, tolerance=1e-8, flag_wb=1)
    return feasible, w, b

def find_svm_sep_primal_wb(X, y, tolerance=1e-8, flag_wb=1):
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
    solvers.options['maxiters'] = 3000
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
    # Check separability
    separable = np.all(np.sign(Pr) == y)
    return separable, w, margin, u, b
    
def check_constraint(w, b, labels, axes, kappa, tol): 
    '''
    Check whether the constraint y^\mu<w, m> > kappa - tol is violated. 
    Args: 
    - w: candidate solution in R^Nr
    - b: bias term
    - labels, axes, kappa, tol same as above
    Returns: 
    - constraint violation boolean flag
    - viol_mfs: a vector manifold indices which are violating the constraint
    - viol_samples: an array of shape (P, N) of samples with the largest margin from each manifold
    '''
    # get the signed fields on the axes and centroids without the center dim
    ax, cent = axes[..., :-1], axes[..., -1]
    T = np.einsum('j, ijk -> ik', w, ax)
    C = np.einsum('j, ij -> i', w, cent)
    T_norms = np.linalg.norm(T, axis = 1)
    margins = labels * (C - labels * T_norms + b)/np.linalg.norm(w)
    constraint = margins >= kappa - tol 
    # check which manifolds violate the constraint 
    viol_mfs = np.argwhere(~constraint).reshape(-1)
    # get the biggest constraint-violation per viol_mf 
    shape_coords = -labels[viol_mfs, None] * T[viol_mfs]
    shape_coords = shape_coords / np.linalg.norm(shape_coords, axis = -1, keepdims=True) # normalize onto the sphere. Shape is num_viol_mfs x K
    # make samples by summing up axes and adding back in the centroids 
    viol_samples =  np.einsum('ijk, ik -> ij', ax[viol_mfs], shape_coords)
    viol_samples = cent[viol_mfs] + viol_samples    
    return np.all(constraint), viol_mfs, viol_samples, margins
