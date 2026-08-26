"""Spherical harmonic transforms between alms and HEALPix maps, via ducc0.

Each transform comes with its adjoint, which the CG solvers need: the adjoint of a transform is
not its inverse, and using one for the other silently breaks the symmetry of the linear operator.
`pseudo_alm_to_map_inverse` is the exception - an actual (pseudo-)inverse, for initialization.
"""
import os

import ducc0
import numpy as np
from numpy.typing import NDArray

from commander4.math_utils.arithmetic import inplace_arr_add
from commander4.diagnostics.performance import benchmark



# Cache for geom_info objects ... pretty small, each entry has a size of O(nside)
# This will be mainly beneficial for small SHTs with high nthreads
hp_geominfos = {}

def _prep_input(arr_in, arr_out, nside, spin):
    ndim_in = arr_in.ndim
    if spin == 0 and ndim_in == 1:
        arr_in = arr_in.reshape((1,-1))
        if arr_out is not None:
            arr_out = arr_out.reshape((1,-1))

    if arr_in.ndim !=2 or (arr_out is not None and arr_out.ndim != 2):
        raise RuntimeError("bad array dimensionality") 

    if nside not in hp_geominfos:
        hp_geominfos[nside] = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()

    return arr_in, arr_out, ndim_in


def alm_to_map(alm: NDArray, nside: int, lmax: int, *, spin: int=0,
               nthreads: int|None=None, out=None, acc: bool=False) -> NDArray:
    with benchmark("alm2map"):
        nthreads = int(os.environ.get("OMP_NUM_THREADS", "1")) if nthreads is None else nthreads
        use_theta_interpol = nside >= 2048
        alm, out, ndim_in = _prep_input(alm, out, nside, spin)
        if acc:
            if out is None:
                raise RuntimeError("Can not accumulate to None output")
            tmp_out = np.copy(out)
        out = ducc0.sht.synthesis(alm=alm, map=out, lmax=lmax, spin=spin,
                                nthreads=nthreads, **hp_geominfos[nside],
                                theta_interpol=use_theta_interpol)
        if acc:
            inplace_arr_add(out, tmp_out)
    return out if ndim_in == 2 else out.reshape((-1,))


def alm_to_map_adjoint(mp: NDArray, nside: int, lmax: int, *, spin: int=0,
                       nthreads: int|None=None, out=None, acc: bool=False) -> NDArray:
    with benchmark("alm2map_adj"):
        nthreads = int(os.environ.get("OMP_NUM_THREADS", "1")) if nthreads is None else nthreads
        use_theta_interpol = nside >= 2048
        mp, out, ndim_in = _prep_input(mp, out, nside, spin)
        if acc:
            if out is None:
                raise RuntimeError("Can not accumulate to None output")
            tmp_out = np.copy(out)
        out = ducc0.sht.adjoint_synthesis(map=mp, alm=out, lmax=lmax, spin=spin,
                                        nthreads=nthreads, **hp_geominfos[nside],
                                        theta_interpol=use_theta_interpol)
        if acc:
            inplace_arr_add(out, tmp_out)
    return out if ndim_in == 2 else out.reshape((-1,))


def map_to_alm(mp: NDArray, nside: int, lmax: int, *, spin: int=0,
                       nthreads: int|None=None, out=None, acc: bool=False) -> NDArray:
    """ Spherical harmonic analysis (inverse synthesis; Y^-1), using only the scalar normalization
        factor 4pi/npix, not any further processing.
        See `pseudo_alm_to_map_inverse` for an equivalent to healpys iterative map2alm.
    """
    with benchmark("map2alm"):
        nthreads = int(os.environ.get("OMP_NUM_THREADS", "1")) if nthreads is None else nthreads
        use_theta_interpol = nside >= 2048
        mp, out, ndim_in = _prep_input(mp, out, nside, spin)
        if acc:
            if out is None:
                raise RuntimeError("Can not accumulate to None output")
            tmp_out = np.copy(out)
        out = ducc0.sht.adjoint_synthesis(map=mp, alm=out, lmax=lmax, spin=spin,
                                        nthreads=nthreads, **hp_geominfos[nside],
                                        theta_interpol=use_theta_interpol)
        out *= 4*np.pi/(12*nside**2)
        if acc:
            inplace_arr_add(out, tmp_out)
    return out if ndim_in == 2 else out.reshape((-1,))


def map_to_alm_adjoint(alm: NDArray, nside: int, lmax: int, *, spin: int=0,
               nthreads: int|None=None, out=None, acc: bool=False) -> NDArray:
    with benchmark("map2alm_adj"):
        nthreads = int(os.environ.get("OMP_NUM_THREADS", "1")) if nthreads is None else nthreads
        use_theta_interpol = nside >= 2048
        alm, out, ndim_in = _prep_input(alm, out, nside, spin)
        if acc:
            if out is None:
                raise RuntimeError("Can not accumulate to None output")
            tmp_out = np.copy(out)
        out = ducc0.sht.synthesis(alm=alm, map=out, lmax=lmax, spin=spin,
                                nthreads=nthreads, **hp_geominfos[nside],
                                theta_interpol=use_theta_interpol)
        out *= 4*np.pi/(12*nside**2)
        if acc:
            inplace_arr_add(out, tmp_out)
    return out if ndim_in == 2 else out.reshape((-1,))


def pseudo_alm_to_map_inverse(map: NDArray, nside: int, lmax: int, *, spin: int=0,
               nthreads: int|None=None, out=None, epsilon: float, maxiter: int,
               return_info: bool=False) -> NDArray:
    """Tries to extract spherical harmonic coefficients from (sets of) one or two maps
    by using the iterative LSMR algorithm.
    
    Parameters
    ----------
    map: numpy.ndarray(([ncomp,] 12*nside**2), dtype=numpy.float32 or numpy.float64
    nside: int
        nside parameter of the Healpix map
    lmax: int >= 0
        the maximum l moment of the transform (inclusive).
    spin: int >= 0
        the spin to use for the transform.
        If spin==0, ncomp must be 1, otherwise 2
    nthreads: None or int >= 0
        the number of threads to use for the computation. Defaults to `None`, which yields to the
        number specified in OMP_NUM_THREADS. nthreads=0 uses all threads on the system
    out: None or numpy.ndarray([ncomp,] (lmax+1)*(lmax+2)//2),
         dtype=numpy.complex of same precision as `map`)
        the set of spherical harmonic coefficients.
        if `None`, a new suitable array is allocated
    epsilon: float > 0
        the relative tolerance used as a stopping criterion
    maxiter: int >= 0
        the maximum number of iterations before stopping the algorithm
    
    Returns
    -------
    numpy.ndarray(([ncomp,] (lmax+1)*(lmax+2)//2), dtype=numpy.complex of same accuracy as `map`)
        the set of spherical harmonic coefficients.
        If `out` was supplied, this will be the same object
    
    if `return_info` is True (default False), also returns a (5,) tuple containing:
        int:
            the reason for stopping the iteration
            1: approximate solution to the equation system found
            2: approximate least-squares solution found
            3: condition number of the equation system too large
            7: maximum number of iterations reached
        
        int:
            the iteration count
        
        float:
            the residual norm, divided by the norm of `map`
        
        float:
            the quality of the least-squares solution
    """
    with benchmark("alm_to_map_inv"):
        nthreads = int(os.environ.get("OMP_NUM_THREADS", "1")) if nthreads is None else nthreads
        map, out, ndim_in = _prep_input(map, out, nside, spin)
        res = ducc0.sht.pseudo_analysis(map=map, alm=out, lmax=lmax, spin=spin,
                                        nthreads=nthreads, **hp_geominfos[nside],
                                        epsilon=epsilon, maxiter=maxiter)
        out = res[0] if ndim_in == 2 else res[0].reshape((-1,))
    if return_info:
        return out, res
    else:
        return out
