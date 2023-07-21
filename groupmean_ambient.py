import numpy as np
import iisignature
from iisignature import sigcombine


def siginversion(sig, channels, depth):
    r"""
    Compute the inverse of an element :math:`a` of the signature Lie group with
    formula:

    .. math::
        a^{-1} = \sum_{k=0}^m(1-a)^{\otimes k}

    with :math:`m` is the signature depth.
    Computation uses that: :math:`1+x+x^2+x^3=1+x(1+x(1+x))` where we replace
    :math:`x` with :math:`(1-a)` (Horner's method).
    """
    rightsig = sig.copy()
    # Horner's iterations
    it = 0
    while it < depth:
        rightsig = sigcombine(sig, -rightsig, channels, depth) + rightsig
        it += 1
    rightsig = -rightsig
    return rightsig


def depth_inds(channels, depth, with_zero=True, scalar=False):
    """
    Most libraries computing the signature transform output the signature as a
    vector. This function outputs the indices corresponding to first value of
    each signature depth in this vector. Example: with depth=4 and channels=2,
    returns [0, 2, 6, 14, 30] (or [0, 1, 3, 7, 15, 31] if scalar=True).
    
    Parameters
    ----------
    scalar : boolean
        Presence of scalar as first value of the signature coordinates. For
        instance, `iisignature` returns signature vectors without the scalar
        value.
    """
    if with_zero:
        return np.concatenate(([0], np.cumsum(channels**np.arange(1-scalar, depth+1))))
    return np.cumsum(channels**np.arange(1-scalar, depth+1))



def mean(SX, channels, depth, weights=None):
    """
    Compute explicit solution :math:`m` of
    .. math::
        \sum_{i=1}^n \log(m^{-1} x_i) = 0
    where :math:`x_i` are signatures and :math:`\log` is the logsignature
    transform.

    Parameters
    ----------
    SX : (batch, 1 + channels**1 + ... + channels**depth) numpy.ndarray
        Signatures to average (scalar value included).

    depth : int
        Depth of signature.

    channels : int
        Number of space dimensions.

    weights : (batch) numpy.ndarray
        Weights of the barycenter. Must sum to one.

    Returns
    -------
    sigbarycenter : (channels**1 + ... + channels**depth) numpy.ndarray
        A signature which is the barycenter of the signatures in SX.

    Example
    -------
    >>> batch = 5     # number of time series
    >>> stream = 30   # number of timestamps for each time series
    >>> channels = 2  # number of dimensions
    >>> depth = 3     # depth (order) of truncation of the signature
    >>> weights = 1./batch*np.ones(batch)
    >>> X = np.random.rand(batch, stream, channels)
    >>> SX = iisignature.sig(X, depth)
    >>> m = mean(SX, channels, depth, weights)

    Reference
    ---------
    The barycenter in free nilpotent Lie groups and its application to 
    iterated-integrals signatures https://arxiv.org/abs/2305.18996
    """
    batch = len(SX)
    dinds = depth_inds(channels, depth, with_zero=True, scalar=False)
    if weights is None:
        weights = 1./batch*np.ones(batch)
    a = np.zeros(dinds[-1])
    if not SX.shape[1] == dinds[-1]:
        raise ValueError(f"Wrong number of signature coordinates. Got "
                         f"{SX.shape[1]} instead of {dinds[-1]}.")
    b = SX.copy()
    p = np.zeros((batch, dinds[-1]))
    q = np.zeros((batch, dinds[-1]))
    v = np.zeros((batch, depth, dinds[-1]))
    # add dimension to store powers of vi.
    # vi shape =(batch, powers, sigterms)
    # /!\ careful to not add it as 3rd dim (but 2nd dim)

    p_coeffs = np.power(-np.ones(depth-1), np.arange(1, depth))*1/np.arange(2, depth+1)

    # CASE K=1
    a[dinds[0]:dinds[1]] = -np.mean(b[:, dinds[0]:dinds[1]], axis=0)

    # CASE K = 2, 3, ...
    K = 2 
    while K < depth+1:
        left, right = dinds[K-1], dinds[K]
        i = 0
        while i < batch:
            l2, r2 = dinds[K-2], dinds[K-1]
            v[i, 0, l2:r2] = q[i, l2:r2] + a[l2:r2] + b[i, l2:r2]

            # update q
            q[i, left:right] = sigcombine(
                a[:right], 
                b[i, :right], 
                channels, K)[left:right] - a[left:right] - b[i, left:right]
            for j in range(K-1):
                # compute powers of v
                v[i, j+1, left:right] = sigcombine(
                    v[i, 0, :right],
                    v[i, j, :right], 
                    channels, K)[left:right] \
                    - v[i, 0, left:right] - v[i, j, left:right]
                # update p
                p[i, left:right] += p_coeffs[j]*v[i, j+1, left:right]
            i += 1
        # update a
        a[left:right] = -np.sum(
            (weights*(b[:, left:right] + q[:, left:right] + p[:, left:right]).transpose()).transpose(),
            axis=0
        )  # double transpose : take advantage of numpy broadcasting
        K += 1
    return(siginversion(a, channels, depth))