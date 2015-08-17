r"""
Composition Statistics (:mod:`skbio.stats.composition`)
=======================================================

.. currentmodule:: skbio.stats.composition

This module provides functions for compositional data analysis.

Many 'omics datasets are inherently compositional - meaning that they
are best interpreted as proportions or percentages rather than
absolute counts.

Formally, :math:`x` is a composition if :math:`\sum_{i=0}^D x_{i} = c`
and :math:`x_{i} > 0`, :math:`1 \leq i \leq D` and :math:`c` is a real
valued constant and there are :math:`D` components for each
composition. In this module :math:`c=1`. Compositional data can be
analyzed using Aitchison geometry. [1]_

However, in this framework, standard real Euclidean operations such as
addition and multiplication no longer apply. Only operations such as
perturbation and power can be used to manipulate this data.

This module allows two styles of manipulation of compositional data.
Compositional data can be analyzed using perturbation and power
operations, which can be useful for simulation studies. The
alternative strategy is to transform compositional data into the real
space.  Right now, the centre log ratio transform (clr) and
the isometric log ratio transform (ilr) [2]_ can be used to accomplish
this. This transform can be useful for performing standard statistical
tools such as parametric hypothesis testing, regressions and more.

The major caveat of using this framework is dealing with zeros.  In
the Aitchison geometry, only compositions with nonzero components can
be considered. The multiplicative replacement technique [3]_ can be
used to substitute these zeros with small pseudocounts without
introducing major distortions to the data.

Functions
---------

.. autosummary::
   :toctree: generated/

   closure
   multiplicative_replacement
   perturb
   perturb_inv
   power
   inner
   clr
   clr_inv
   ilr
   ilr_inv
   centralize

References
----------
.. [1] V. Pawlowsky-Glahn. "Lecture Notes on Compositional Data Analysis"

.. [2] J. J. Egozcue. "Isometric Logratio Transformations for
   Compositional Data Analysis"

.. [3] J. A. Martin-Fernandez. "Dealing With Zeros and Missing Values in
   Compositional Data Sets Using Nonparametric Imputation"


Examples
--------

>>> import numpy as np

Consider a very simple environment with only 3 species. The species
in the environment are equally distributed and their proportions are
equivalent:

>>> otus = np.array([1./3, 1./3., 1./3])

Suppose that an antibiotic kills off half of the population for the
first two species, but doesn't harm the third species. Then the
perturbation vector would be as follows

>>> antibiotic = np.array([1./2, 1./2, 1])

And the resulting perturbation would be

>>> perturb(otus, antibiotic)
array([ 0.25,  0.25,  0.5 ])

"""

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.stats as ss
from skbio.diversity.alpha import lladser_pe


def closure(mat):
    """
    Performs closure to ensure that all elements add up to 1.

    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components

    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import closure
    >>> X = np.array([[2, 2, 6], [4, 4, 2]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ 0.4,  0.4,  0.2]])

    """
    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()


def multiplicative_replacement(mat, delta=None):
    r"""Replace all zeros with small non-zero values

    It uses the multiplicative replacement strategy [1]_ ,
    replacing zeros with a small positive :math:`\delta`
    and ensuring that the compositions still add up to 1.


    Parameters
    ----------
    mat: array_like
       a matrix of proportions where
       rows = compositions and
       columns = components
    delta: float, optional
       a small number to be used to replace zeros
       If delta is not specified, then the default delta is
       :math:`\delta = \frac{1}{N^2}` where :math:`N`
       is the number of components

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    References
    ----------
    .. [1] J. A. Martin-Fernandez. "Dealing With Zeros and Missing Values in
       Compositional Data Sets Using Nonparametric Imputation"


    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import multiplicative_replacement
    >>> X = np.array([[.2,.4,.4, 0],[0,.5,.5,0]])
    >>> multiplicative_replacement(X)
    array([[ 0.1875,  0.375 ,  0.375 ,  0.0625],
           [ 0.0625,  0.4375,  0.4375,  0.0625]])

    """
    mat = closure(mat)
    z_mat = (mat == 0)

    num_feats = mat.shape[-1]
    tot = z_mat.sum(axis=-1, keepdims=True)

    if delta is None:
        delta = (1. / num_feats)**2

    zcnts = 1 - tot * delta
    mat = np.where(z_mat, delta, zcnts * mat)
    return mat.squeeze()


def coverage_replacement(count_mat, uncovered_estimator=lladser_pe):
    r"""Replace all zeros with small non-zero values
    using a coverage estimator

    It uses the multiplicative replacement strategy [1]_ ,
    replacing zeros with a small positive :math:`\delta`
    and ensuring that the compositions still add up to 1.
    However, :math:`\delta` is determined using a coverage
    estimator such that all of the non-zero values add up
    to the coverage probability

    Parameters
    ----------
    count_mat: array_like
       a matrix of counts where
       rows = samples and
       columns = components
    uncovered_estimator : function, optional
       function to estimate the uncovered probability

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    """
    mat = closure(count_mat)
    z_mat = (mat == 0)

    tot = z_mat.sum(axis=-1)

    def func(x):
        up = uncovered_estimator(x)
        if up > 1:
            return 1 - 1 / sum(x)
        else:
            return up

    p_unobs = np.apply_along_axis(func,
                                  -1, count_mat)
    delta = p_unobs / tot

    p_obs = 1 - p_unobs
    p_obs = np.repeat(p_obs[np.newaxis, :],
                      mat.shape[-1], 0).T

    delta = np.repeat(delta[np.newaxis, :],
                      mat.shape[-1], 0).T

    rounded_zeros = np.multiply(z_mat, delta)
    non_zeros = np.multiply(mat, p_obs)

    mat = rounded_zeros + non_zeros
    return mat.squeeze()


def perturb(x, y):
    r"""
    Performs the perturbation operation.

    This operation is defined as

    .. math::
        x \oplus y = C[x_1 y_1, \ldots, x_D y_D]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]

    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
    y : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import perturb
    >>> x = np.array([.1,.3,.4, .2])
    >>> y = np.array([1./6,1./6,1./3,1./3])
    >>> perturb(x,y)
    array([ 0.0625,  0.1875,  0.5   ,  0.25  ])

    """
    x, y = closure(x), closure(y)
    return closure(x * y)


def perturb_inv(x, y):
    r"""
    Performs the inverse perturbation operation.

    This operation is defined as

    .. math::
        x \ominus y = C[x_1 y_1^{-1}, \ldots, x_D y_D^{-1}]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]


    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components
    y : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import perturb_inv
    >>> x = np.array([.1,.3,.4, .2])
    >>> y = np.array([1./6,1./6,1./3,1./3])
    >>> perturb_inv(x,y)
    array([ 0.14285714,  0.42857143,  0.28571429,  0.14285714])
    """
    x, y = closure(x), closure(y)
    return closure(x / y)


def power(x, a):
    r"""
    Performs the power operation.

    This operation is defined as follows

    .. math::
        `x \odot a = C[x_1^a, \ldots, x_D^a]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]

    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
    a : float
        a scalar float

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import power
    >>> x = np.array([.1,.3,.4, .2])
    >>> power(x, .1)
    array([ 0.23059566,  0.25737316,  0.26488486,  0.24714631])

    """
    x = closure(x)
    return closure(x**a).squeeze()


def inner(x, y):
    r"""
    Calculates the Aitchson inner product.

    This inner product is defined as follows

    .. math::
        \langle x, y \rangle_a =
        \frac{1}{2D} \sum\limits_{i=1}^{D} \sum\limits_{j=1}^{D}
        \ln\left(\frac{x_i}{x_j}\right) \ln\left(\frac{y_i}{y_j}\right)

    Parameters
    ----------
    x : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components
    y : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components

    Returns
    -------
    numpy.ndarray
         inner product result

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import inner
    >>> x = np.array([.1, .3, .4, .2])
    >>> y = np.array([.2, .4, .2, .2])
    >>> inner(x, y)
    0.21078524737545556
    """
    x = closure(x)
    y = closure(y)
    a, b = clr(x), clr(y)
    return a.dot(b.T)


def norm(x):
    """
    Calculates the Aitchison norm

    The norm is calculated as follows

    .. math::
        \norm{x}_a = \sqrt{\langle x, x \rangle_a}

    Parameters
    ----------
    x : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components

    Returns
    -------
    numpy.ndarray
        list of norms
    """
    return np.sqrt(np.diag(inner(x, x)))


def distance(x, y):
    """
    Calculates the Aitchison distance.  This is a measure
    of distance or dissimiliarity between two compositions

    The norm is calculated as follows

    .. math::
        d_a(x, y) = \norm{ x \ominus y }

    Parameters
    ----------
    x : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components

    y : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components

    Returns
    -------
    numpy.ndarray
        list of distances
    """
    return norm(perturb_inv(x, y))


def clr(mat):
    r"""
    Performs centre log ratio transformation.

    This function transforms compositions from Aitchison geometry to
    the real space. The :math:`clr` transform is both an isometry and an
    isomorphism defined on the following spaces

    :math:`clr: S^D \rightarrow U`

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    It is defined for a composition :math:`x` as follows:

    .. math::
        clr(x) = \ln\left[\frac{x_1}{g_m(x)}, \ldots, \frac{x_D}{g_m(x)}\right]

    where :math:`g_m(x) = (\prod\limits_{i=1}^{D} x_i)^{1/D}` is the geometric
    mean of :math:`x`.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         clr transformed matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import clr
    >>> x = np.array([.1, .3, .4, .2])
    >>> clr(x)
    array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])

    Notes
    -----
    If there are zeros present, only the nonzero components are considered
    """
    mat = closure(mat)
    lmat = np.atleast_2d(np.log(mat))

    # If zeros are present, only consider the nonzero components
    idx = (lmat != -np.inf).astype(np.int)
    lmat[lmat == -np.inf] = 0
    gm = np.diag(lmat.dot(idx.T) / idx.sum(axis=1))
    gm = np.atleast_2d(gm).T
    res = lmat - gm
    res[mat == 0] = 0
    return (res).squeeze()


def clr_inv(mat):
    r"""
    Performs inverse centre log ratio transformation.

    This function transforms compositions from the real space to
    Aitchison geometry. The :math:`clr^{-1}` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`clr^{-1}: U \rightarrow S^D`

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    This transformation is defined as follows

    .. math::
        clr^{-1}(x) = C[\exp( x_1, \ldots, x_D)]

    Parameters
    ----------
    mat : numpy.ndarray, float
       a matrix of real values where
       rows = transformed compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         inverse clr transformed matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import clr_inv
    >>> x = np.array([.1, .3, .4, .2])
    >>> clr_inv(x)
    array([ 0.21383822,  0.26118259,  0.28865141,  0.23632778])

    """
    return closure(np.exp(mat))


def ilr(mat, basis=None, check=True):
    r"""
    Performs isometric log ratio transformation.

    This function transforms compositions from Aitchison simplex to
    the real space. The :math: ilr` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`ilr: S^D \rightarrow \mathbb{R}^{D-1}`

    The ilr transformation is defined as follows

    .. math::
        ilr(x) =
        [\langle x, e_1 \rangle_a, \ldots, \langle x, e_{D-1} \rangle_a]

    where :math:`[e_1,\ldots,e_{D-1}]` is an orthonormal basis in the simplex.

    If an orthornormal basis isn't specified, the J. J. Egozcue orthonormal
    basis derived from Gram-Schmidt orthogonalization will be used by
    default.

    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of proportions where
       rows = compositions and
       columns = components

    basis: numpy.ndarray, optional
        orthonormal basis for Aitchison simplex
        defaults to J.J.Egozcue orthonormal basis

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import ilr
    >>> x = np.array([.1, .3, .4, .2])
    >>> ilr(x)
    array([-0.7768362 , -0.68339802,  0.11704769])

    """
    mat = closure(mat)
    if basis is None:
        basis = clr_inv(_gram_schmidt_basis(mat.shape[-1]))
    elif check:
        _check_orthogonality(basis)
    return inner(mat, basis)


def ilr_inv(mat, basis=None, check=True):
    r"""
    Performs inverse isometric log ratio transform.

    This function transforms compositions from the real space to
    Aitchison geometry. The :math:`ilr^{-1}` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`ilr^{-1}: \mathbb{R}^{D-1} \rightarrow S^D`

    The inverse ilr transformation is defined as follows

    .. math::
        ilr^{-1}(x) = \bigoplus\limits_{i=1}^{D-1} x \odot e_i

    where :math:`[e_1,\ldots, e_{D-1}]` is an orthonormal basis in the simplex.

    If an orthornormal basis isn't specified, the J. J. Egozcue orthonormal
    basis derived from Gram-Schmidt orthogonalization will be used by
    default.


    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of transformed proportions where
       rows = compositions and
       columns = components

    basis: numpy.ndarray, optional
        orthonormal basis for Aitchison simplex
        defaults to J.J.Egozcue orthonormal basis

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import ilr
    >>> x = np.array([.1, .3, .6,])
    >>> ilr_inv(x)
    array([ 0.34180297,  0.29672718,  0.22054469,  0.14092516])

    """

    if basis is None:
        basis = _gram_schmidt_basis(mat.shape[-1] + 1)
    elif check:
        _check_orthogonality(basis)
    return clr_inv(np.dot(mat, basis))


def centralize(mat):
    r"""Center data around its geometric average.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         centered composition matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import centralize
    >>> X = np.array([[.1,.3,.4, .2],[.2,.2,.2,.4]])
    >>> centralize(X)
    array([[ 0.17445763,  0.30216948,  0.34891526,  0.17445763],
           [ 0.32495488,  0.18761279,  0.16247744,  0.32495488]])

    """
    mat = closure(mat)
    cen = ss.gmean(mat, axis=0)
    return perturb_inv(mat, cen)


def phylogenetic_basis(treenode):
    """
    Determines the basis based on phylogenetic tree

    Parameters
    ----------
    treenode : skbio.TreeNode
        Phylogenetic tree.  MUST be a bifurcating tree
    Returns
    -------
    basis : dict, {str, np.array}
        Returns a set of orthonormal bases in the Aitchison simplex
        corresponding to the phylogenetic tree. The order of the
        basis is index by the level order of the internal nodes

    Raises
    ------
    ValueError
        The tree doesn't contain two branches
    ValueError
        The tree doesn't have unique node names
    Examples
    --------
    >>> from skbio.stats.composition import phylogenetic_basis
    >>> from six import StringIO
    >>> from skbio imoprt TreeNode
    >>> tree = "((b,c)a, d)root;"
    >>> t = TreeNode.read(StringIO(tree))
    >>> phylogenetic_basis(t)
    array([[ 0.62985567,  0.18507216,  0.18507216],
           [ 0.28399541,  0.57597535,  0.14002925]])
    """
    nodes = [n for n in treenode.levelorder(include_self=True)]

    D = len(nodes)
    n_tips = sum([n.is_tip() for n in nodes])

    # keeps track of k, r, s, t for all of the internal nodes
    history = np.zeros((4, D-1))
    basis = np.zeros((n_tips-1, n_tips))

    # Fill in r and s for all of the nodes
    for i in range(1, D):
        j = D-i
        # left or right child
        child_idx = int(nodes[j].parent.children[0] == nodes[j])
        parent_idx = (j+1)//2-1
        if len(nodes[j].children) == 0:
            history[child_idx+1, parent_idx] = 1
        else:
            # number of tips in child node
            parent_history = history[1, j] + history[2, j]
            history[child_idx+1, parent_idx] = parent_history

    # Fill in k and t for all of the nodes
    # and find the basis
    idx = 0
    for n in nodes:
        if len(n.children) == 0:
            idx += 1
            continue
        if len(n.children) != 2:
            raise ValueError("Not a bifurcating tree!")

        parent_idx = (j+1)//2-1

        r = history[1, idx]
        s = history[2, idx]

        # get parent values
        _k = history[0, parent_idx]
        _r = history[1, parent_idx]
        _s = history[2, parent_idx]
        _t = history[3, parent_idx]

        a = np.sqrt(s / (r*(r+s)))
        b = -1*np.sqrt(r / (s*(r+s)))

        if n.parent is None:
            basis[idx, :] = clr_inv([a]*r + [b]*s)
            idx += 1
            continue

        if n.parent.children[0] == n:  # right child
            k = _r + _k
            t = _t
        else:  # left child
            k = _k
            t = _s + _t
        basis[idx, :] = clr_inv([0]*k + [a]*r + [b]*s + [0]*t)
        history[0, idx] = k
        history[1, idx] = r
        history[2, idx] = s
        history[3, idx] = t
        idx += 1
    return basis


def _merge_two_dicts(x, y):
    '''
    Given two dicts, merge them into a new dict as a shallow copy.
    '''
    z = x.copy()
    z.update(y)

    if len(z) < len(x) + len(y):
        raise ValueError("Non unique node names!")

    return z


def _gram_schmidt_basis(n):
    """
    Builds clr transformed basis derived from
    gram schmidt orthogonalization

    Parameters
    ----------
    n : int
        Dimension of the Aitchison simplex
    """
    basis = np.zeros((n, n-1))
    for j in range(n-1):
        i = j + 1
        e = np.array([(1/i)]*i + [-1] +
                     [0]*(n-i-1))*np.sqrt(i/(i+1))
        basis[:, j] = e
    return basis.T


def _check_orthogonality(basis):
    """
    Checks to see if basis is truly orthonormal in the
    Aitchison simplex

    Parameters
    ----------
    basis: numpy.ndarray
        basis in the Aitchison simplex
    """
    if not np.allclose(inner(basis, basis), np.identity(len(basis)),
                       rtol=1e-4, atol=1e-6):
        raise ValueError("Aitchison basis is not orthonormal")
