import numpy as np
from scipy.linalg import qr, cholesky, solve_triangular
from cvxopt import solvers, matrix
from tqdm import tqdm
from collections import defaultdict

# Configure cvxopt solvers
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 20000
solvers.options['abstol'] = 1e-12
solvers.options['reltol'] = 1e-12
solvers.options['feastol'] = 1e-12

def manifold_analysis_corr(XtotT, kappa, n_t):
    '''
    Carry out the analysis on multiple manifolds.
    Args:
        XtotT: Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
                of the space, and P_i is the number of sampled points for the i_th manifold.
        kappa: Margin size to use in the analysis (scalar, kappa > 0)
        n_t: Number of gaussian vectors to sample per manifold
    Returns:
        capacity: capacity estimate for the collection of manifolds
        norms: value of the convex minimization objective at its final value
        exits: cvxopt exit statuses
        all_stats: a dictionary containing statistics which jointly describe the correlations' effect on the capacity
        axes: manifold axes
    '''
    # Number of manifolds to analyze
    num_manifolds = len(XtotT)
    N, P = XtotT[0].shape[0], num_manifolds 
    
    # note that we do not subtract the global mean. This is done to preserve all linear independence properties which will be important when inverting covariance matrices. 
    XtotInput, centers = [], []
    for i in range(num_manifolds):
        # separate the manifolds into (1) centroids and (2) the manifold points with the centroid subtracted out. 
        Xr = XtotT[i]
        # Compute mean of the data 
        Xr0 = np.mean(Xr, axis=1) 
        centers.append(Xr0)
        # Center the data
        M = (Xr - Xr0.reshape(-1, 1))
        XtotInput.append(M)
    centers = np.stack(centers, axis=1) # Centers is an array of shape (N, P) for P manifolds
    
    # Make the D+1 dimensional data
    sD1, axes = [], []
    for i in range(num_manifolds):
        # Get the manifold points in the manifold-axis basis (i.e., NOT the standard N-dimensional basis) 
        S_r = XtotInput[i]
        center = centers[:, i] 
        # Get the axes Q, and the manifold vectors in the orthogonal axes basis: Q.T @ S_r 
        Q, R = qr(S_r, mode='economic') # Q is shape ambient_dim x m 
        S_r = Q.T @ S_r 
        # Get the new sizes
        D, m = S_r.shape
        # Add the center dimension to the axes and manifold points
        sD1_p = np.concatenate([S_r, np.ones((1, m))], axis=0) # (D+1, m)
        sD1.append(sD1_p)
        # Save the axes to calculate correlations later on: 
        Q = np.concatenate([Q, center[:, None]], axis = -1)     
        assert np.linalg.matrix_rank(Q) == m + 1, f'Axes for manifold {i} have a lower rank than they should'
        axes.append(Q)
        
    # Get the covariance tensors from the axes: 
    axes = np.stack(axes, axis = 0) # shape is (P, N, D+1)
    assert axes.shape == (num_manifolds, N, D+1)
    C, L = covariance_tensors(axes.transpose(0, 2, 1))
    
    # Draw the t-vectors and labels
    sD1 = np.stack(sD1, axis = 0) # shape is (P, D+1, m)
    t_vecs = np.random.randn(P, D+1, n_t)
    labels = np.random.choice([-1,1], size=(n_t, num_manifolds))
    norms, exits = [] , []
    
    # calculate the capacity: 
    print('Sampling...')
    all_stats = defaultdict(list) 
    for i in tqdm(range(n_t)): 
        t = t_vecs[..., i]
        v_f, vt_f, exitflag, alphar, normvt2, supp_stats = minimize_quad_form(t, L, C, sD1, kappa, labels[i])
        norms.append(normvt2)
        exits.append(exitflag)
        for key, val in supp_stats.items(): 
            all_stats[key].append(val)
            
    # reformat the dictionary of results for ease of access. Calculate the capacity
    for key, val in all_stats.items():
        all_stats[key] = np.stack(val)
    cap_unc = np.mean(norms)/num_manifolds
    
    return 1/cap_unc, norms, exits, all_stats, axes

def minimize_quad_form(t, L, C, sD1, kappa, labels) :
    '''
    This function carries out the constrained minimization described in the overleaf doc, equation (10)
    min \sum_μ ||v_μ - t_μ||^2 subject to min_s y<Lv, s> ≥ kappa 
    Instead of minimizing F = ||V-T||^2, The actual function that is minimized will be
        F' = 0.5 * V^2 - T * V
    Args:
        t: A single T vector encoded as a 2D array of shape (P, D+1) where P=num_manifolds
        L: Cholesky factorization of covariance tensor of shape (P, M, P, M) where M=num_manifold_axes
        sD1: 3D array of shape (P, D+1, m) where m is number of manifold points
        kappa: Size of margin (default 0)
    Returns:
        v_f: D+1 dimensional solution vector encoded as a 2D array of shape (D+1, 1)
        vt_f: Final value of the objective function (which does not include T^2). May be negative.
        exitflag: Not used, but equal to 1 if a local minimum is found.
        alphar: Vector of lagrange multipliers at the solution. 
        normvt2: Final value of ||V-T||^2 at the solution.
        supp_stats: A dictionary containing relevant statistics of the support vectors (e.g., Q^{cent,cent}, the support vectors themselves, etc.)
    '''    
    # t is shape P, D+1 so we unroll it:  
    P, D1, m = sD1.shape
    t = t.reshape(-1)
    
    # Construct the matrices needed for F' = 0.5 * V' * P * V - q' * V.
    # We will need P = Identity, and q = -T
    q = - t.astype(np.double)
    q = matrix(q)
    
    # Construct the constraints.  We need <yLV, S> - k > 0.
    # This means G = -(sD1  yy^T o L)  and h = -kappa
    constraint = get_constraint_matrix(L, labels, sD1) # shape is (m*P, P*D1)    
    G = constraint.astype(np.double)
    G = matrix(G)

    h =  - np.ones(m*P) * kappa
    h = h.T.astype(np.double)
    h = matrix(h)
    
    # The matrix of the quadratic form is simply the identity: 
    A = matrix(np.eye(D1 * P))
    
    # Carry out the constrained minimization
    output = solvers.qp(A, q, G, h)

    # Format the output
    v_f = np.array(output['x']).reshape(-1)
    vt_f = output['primal objective']
    if output['status'] == 'optimal':
        exitflag = 1
    else:
        exitflag = 0
    alphar = np.array(output['z'])
    # Compute the true value of the objective function
    normvt2 = np.square(v_f - t).sum()
    
    # get the support statistics: 
    supp_stats = support_statistics(v_f, t, labels, L, C)
    return v_f, vt_f, exitflag, alphar, normvt2, supp_stats

def support_statistics(v, t, labels, L, C): 
    '''
    Calculate the relevant statistics of the support vectors.
    Args:
    - v: the solution to the constrained optimization problem. A vector of shape P*D1
    - t: Gaussian vector; same shape as v
    - labels: vector of labels of shape P 
    - L: Cholesky factorization of shape P, D1, P, D1
    Returns
     dictionary containing all support statistics 
    ''' 
    P, D1, P, D1 = L.shape
    # solve the triangular linear system Cx = L(V - T) for x as described in equation (17) of the overleaf doc 
    L = L.reshape(P*D1, P*D1)
    L, v, t = L.astype(np.double), v.astype(np.double), t.astype(np.double)
    ylamS = solve_cholesky(L@(v-t), L).reshape(P, D1) # x = ylambda * S 
    
    # calculate the contributions from each of the correlations using Einstein summation: 
    all_terms = np.einsum('mi, nj, minj -> minj', ylamS, ylamS, C)
    cent_cent = np.sum(all_terms[:, -1, :, -1] - np.diag(np.diag(all_terms[:, -1, :, -1])))
    ax_ax = np.sum(all_terms[:, :-1, :, :-1]) - np.sum(all_terms[np.arange(P), :-1, np.arange(P), :-1])
    cent_ax = 2 * (np.sum(all_terms[:, :-1, :, -1]) - np.sum(all_terms[np.arange(P), :-1, np.arange(P), -1]))
    
    # contributions to the capacity from each manifold individually:
    individual_contribs = np.sum(all_terms[np.arange(P), :, np.arange(P), :])
    outdct = {'cent_cent' : cent_cent,
              'ax_ax' : ax_ax,
              'cent_ax' : cent_ax,
              'individual_contribs' : individual_contribs, 
              'cap_hat' : all_terms.sum(),
              'ylamS' : ylamS,
              'labels' : labels,
              'v' : v} 
    return outdct
    
def solve_cholesky(b, L):
    '''
    Solve the linear system LL.T x = b using forward and backward substitution. L is assumed to be lower triangular.
    '''
    assert np.allclose(L, np.tril(L)), 'Cholesky factorization is not lower triangular. Check axes transformations/reshapes.'
    y = solve_triangular(L, b, lower = True)
    return solve_triangular(L.T, y, lower = False) 
    
def covariance_tensors(axes): 
    '''
    Compute the covariance tensor C^{μ, i}_{ν, j} = <u^μ_i, u^ν_j> and its Cholesky factorization.
    Args: 
    - axes: A tensor of shape (num_manifolds, num_axes, ambient_dim) 
    Returns: 
    - Covariance tensor of shape (P, num_axes)^2 
    - Cholesky factorization of the above
    ''' 
    P, D1, N = axes.shape
    axes = axes.reshape(P*D1, N).T
    if N > P*D1:
        # In the high dimensional regime, we avoid ever explicitly forming C to calculate its Cholesky decomp -- just use the QR decomp + orthogonality of Q: 
        Q, R = qr(axes, mode = 'economic') # get full rank matrices to account for the case in which N < P*D1 
        assert np.all(R.shape == (P*D1, P*D1)), 'Need more samples per manifold'
        L = R.T 
        print('Shape and rank of manifold axes is: ', axes.shape, np.linalg.matrix_rank(axes))
        print('Shape and rank of covariance matrix is: ', L.shape, np.linalg.matrix_rank(L)) # note we use the fact that rank(L)=rank(C) 
        C = L @ L.T
    else: 
        print('Approximating capacity by forcing positive definiteness of correlation tensor. Need more neurons for exact calculation')
        C = axes.T @ axes
        eigs = np.linalg.eig(C)[0]
        C = C + np.eye(C.shape[0]) * 1e-3 # compensate for any negative or zero eigenvalues
        L = cholesky(C, lower=True)
    return C.reshape(P, D1, P, D1), L.reshape(P, D1, P, D1)
    
def get_constraint_matrix(L, labels, sD1): 
    '''
    Build the constraint matrix for the constrained optimization.
    Args: 
    - L: Cholesky factorization of C in the tensor form (num_manifolds, num_axes, num_manifolds, num_axes)
    - C: the covariance tensor; same shape as L
    - labels: A vector of labels in {+1, -1}^(num_manifolds)
    - sD1: An array of manifold points in the shape (num_manifolds, num_axes, num_samples) 
    Returns: 
    - constraint matrix in shape (P*m, P*(D+1)) 
    '''
    assert len(labels.shape) == 1
    assert labels.shape[0] == L.shape[0] 
    P, D1, m = sD1.shape    
    Y = labels
    G = np.einsum('m, minj -> minj', Y, L)
    # the constraint is given by s L v ≥ kappa for all manifold points s. Therefore, we carry out the sum (sL). 
    constraint = np.einsum('mis, minj -> msnj', sD1, G).reshape(P * m, P * D1)    
    return constraint
    
def get_null_constraint_matrix(sD1): 
    '''
    A function to unit test get_constraint_matrix on uncorrelated manifolds. 
    args: 
    sD1: an array of manifold points of shape P, D1, m
    ''' 
    P, D1, m = sD1.shape 
    A = np.zeros((P * m, P*D1))
    for mu in range(P): 
        for s in range(m): 
            A[s + mu * m, mu * D1 : (mu+1) * D1] = sD1[mu, :, s]
    return A 
