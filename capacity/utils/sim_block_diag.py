import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.linalg import block_diag, sqrtm, cholesky

def closest_ancestor(P, level_sizes): 
    '''
    Build a matrix whose entries M_{i,j} are the level of the closest ancestor of the ith and jth manifolds.
    Args: 
    - P: number of manifolds. 
    - level_sizes: how many manifolds occupy each hierarchy.
    '''
    assert P % np.prod(level_sizes)==0, 'Supplied graph does not cleanly partition classes' #check that the supplied graph is feasible
    
    # we proceed by adding block matrices of ones at each level. 
    # first we get the size of the block matrix associated with each level. 
    rng_d = {level : int(P/np.prod(level_sizes[:level+1])) for level in range(len(level_sizes))}
    # form block matrices: we do this by forming nested block diagonal matrices of ones and summing. 
    blocks = [block_diag(*[np.ones((rng, rng)) for i in range(P // rng)])
              for rng in rng_d.values()]
    ## combine all block levels
    return np.sum(np.stack(blocks), 0)
    
def build_corr_mat(ancs, corrs, sig = 0.1):
    '''
    Construct the average manifold center correlation matrix.
    Args:
    - ancs: A matrix whose entries are the level of the closest ancestor along the graphical hierarchy (higher values indicate
    closer ancestors). That is, this is essentially the output of "closest_ancestor"
    - corrs: the average correlation for nodes at each level of the hierarchy –e.g. with two levels of blocks, this should be 
    length 3, with the first entry corresponding to the base correlation among all nodes.
    '''
    P = ancs.shape[0]
    # make sure we've got an array
    if type(corrs) != np.ndarray : 
        corrs=np.array(corrs) 
    assert np.max(np.abs(corrs)) < 1., 'The maximum correlation magnitude is 1.' 
    ancs = ancs.astype('int')
    # set up indexing matrix – one for each level in the hierarchy – and slice out correct correlation value 
    mat = np.ones((P, P, len(corrs))) * corrs # along each of the last axes we have constant matrices of value corrs[i]
    mat = mat[np.arange(P), np.arange(P), ancs] # use the ancestor matrix to slice out the correct correlation value
    # threshold and set diagonal to 1. avoid singularities
    eps = 0.005
    mat = np.maximum(np.minimum(mat, 1.-eps), -1+eps) 
    mat[np.arange(P), np.arange(P)] = 1.
    return mat
    
def sample_centroids(base_cm, dim_N, eps = 1e-10): 
    '''
    Use spectral decomp to sample a set of manifold centroids, given a base correlation matrix. 
    Args: 
    - base_cm: average correlation matrix
    - dim_N: dimension of ambient space
    '''
    assert (base_cm == base_cm.T).all(), 'base correlation matrix is not symetric' 
    eigval, eigvec = np.linalg.eigh(base_cm)
    P = eigval.shape[0]
    Lam = np.eye(P) * eigval
    assert np.sum((base_cm - eigvec @ Lam @ eigvec.T)**2) < eps, 'failiure in spectral decomposition.' 
    #cholesky decomp 
    L = eigvec @ sqrtm(Lam)
    assert (L.imag == np.zeros(L.shape)).all()
    Phi = 1/np.sqrt(dim_N) * np.random.randn(P, dim_N) 
    return L @ Phi, {'corr_mat' :L, 'base_mat': Phi}  # now we can scale by centroid norms 
    
def draw_spheres(P, dim_S, dim_N, samples, radius=1., eps=1e-10, sample_surface=True, axes=None): 
    '''
    Sample points uniformly on spheres of dimension dim_S in an ambient space: \R^(dim_N). 
    The dim_S many axes have Gaussian i.i.d. components. 
    Args: 
    - P: number of manifolds to sample.
    - dim_S: dimension of the sphere. 
    - dim_N: dimension of the ambient space.
    - samples: number of points to sample on sphere. 
    - radius: radius of S-dimensional sphere in R^(dim_N)
    Returns: 
    - uniform samples from spheres in an array of shape (P, samples, N)
    ''' 
    # draw from spherically symetric dist
    spheres = np.random.randn(P, samples, dim_S)
    # normalize so that we're sampling uniformly from the surface of the 'dim_S'-dimensional unit sphere
    spheres = spheres/np.sqrt(np.sum(spheres**2, -1, keepdims = True))
    if sample_surface is False:
        # randomly rescale over [0,radius] so that we're randomly sampling from the volume of the sphere
        radial_sample = np.random.uniform(low=0., high = radius, size = (P, samples, 1))
        spheres *= radial_sample
    else: 
        # rescale to the desired radius
        spheres *= radius 
    # generate manifold axes and normalize using random unit-norm axes or supplied axes:
    if axes is None:
        axes = 1/np.sqrt(dim_N) * np.random.randn(dim_S, P, dim_N)
        axes /= np.sqrt(np.sum(axes**2, axis = -1, keepdims=True))
    # project spheres onto their respective axes
    spheres = np.einsum('ijk,kin->ijn', spheres, axes) 
    return spheres 

def make_manifolds(P, S, N, samples, level_sizes, corrs, radius=1., 
                   return_proj=False, center_radius = 1., sample_surface=True): 
    '''make sphereical manifolds.
    Args:
    - P : number of manifolds
    - S : intrinsic dimensionality 
    - N : ambient dimensionality
    - samples: number of samples to draw
    - level_sizes: number of blocks of correlated centroids at each level, relative to the previous level; must satisfy prod(level_sizes) mod P = 0  
    - corrs: the correlations at each level, starting from baseline shared correlation; a list of length len(level_sizes)+1
    - radius: spherical radius
    - return_proj: whether to return centroid correlation matrix data
    - center_radius: radius of manifold centroids
    - sample_surface: whether to sample from the surface of the spheres (true) or their interior (false)'''
    spheres = draw_spheres(P, S, N, samples, radius=radius, sample_surface=sample_surface) # shape is (P x smpls x N) 
    # form correlated centers
    anc = closest_ancestor(P, level_sizes) 
    # build base correlation matrix and sample
    base_corr_mat = build_corr_mat(anc, corrs) 
    centers, c_dct = sample_centroids(base_corr_mat, N) # shape is (P x N) so we insert new axis below
    manifolds = spheres + center_radius * centers[:, None, :] 
    if return_proj is False : 
        return manifolds
    else: 
        return manifolds, c_dct
    
    
def make_corr_spheres(P, S, N, samples, level_sizes_cent, corrs_cent, level_sizes_ax, corrs_ax, radius=1.,
                      center_radius=1., sample_surface=True): 
    '''
    Make spherical manifolds with correlations running between both centers and axes in a block diagonal fashion. 
    Returns a (P, samples, N)-shaped array containing samples from the surface of correlated spheres
    '''
    ## get the centers: 
    anc = closest_ancestor(P, level_sizes_cent) 
    base_corr_mat = build_corr_mat(anc, corrs_cent) 
    centers, c_dct = sample_centroids(base_corr_mat, N) # shape is (P x N) so we insert new axis below
    ## get the axes: 
    axs = [] 
    for s in range(S): 
        anc = closest_ancestor(P, level_sizes_ax) 
        base_corr_mat = build_corr_mat(anc, corrs_ax) 
        ax, c_dct = sample_centroids(base_corr_mat, N)
        axs.append(ax) 
    axs = np.stack(axs) # shape is (S x P x N), as required
    spheres = draw_spheres(P, S, N, samples, radius=radius, sample_surface=sample_surface, axes=axs)
    manifolds = spheres + center_radius * centers[:, None, :]
    return manifolds
        
        
def gaussian_clouds(P, N, samples, level_sizes_cent, corrs_cent, level_sizes_pt, corrs_pt, radius=1.,
                      center_radius=1.): 
    '''
    Returns a (P, samples, N)-shaped array containing gaussian point cloud samples
    '''
    ## get the centers: 
    anc = closest_ancestor(P, level_sizes_cent) 
    base_corr_mat = build_corr_mat(anc, corrs_cent) 
    centers, c_dct = sample_centroids(base_corr_mat, N) # shape is (P x N) so we insert new axis below
    ## get the cloud: 
    anc = closest_ancestor(P, level_sizes_pt) 
    base_corr_mat = build_corr_mat(anc, corrs_pt) # a P x P correlation matrix 
    L = cholesky(base_corr_mat, lower=True)
    cloud = 1/np.sqrt(N) * np.random.randn(P, samples, N)
    cloud = radius * np.einsum('mn, nsk -> msk', L, cloud)
    return center_radius * centers[:, None, :] + cloud         
                      
                      
                      