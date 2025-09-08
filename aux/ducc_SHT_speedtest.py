import numpy as np
from numpy.typing import NDArray
import ducc0
from time import time


# Cache for geom_info objects ... pretty small, each entry has a size of O(nside)
# This will be mainly beneficial for small SHTs with high nthreads
hp_geominfos = {}

def alm_to_map(alm: NDArray, nside: int, lmax: int, nthreads=1, out=None) -> NDArray:
    if nside not in hp_geominfos:
        hp_geominfos[nside] = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    out = None if out is None else out.reshape((1,-1))
    return ducc0.sht.synthesis(alm=alm.reshape((1,-1)),
                               map=out,
                               lmax=lmax,
                               spin=0,
                               nthreads=nthreads,
                               **hp_geominfos[nside]).reshape((-1,))


def alm_to_map_adjoint(mp: NDArray, nside: int, lmax: int, nthreads=1, out=None) -> NDArray:
    if nside not in hp_geominfos:
        hp_geominfos[nside] = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    out = None if out is None else out.reshape((1,-1))
    return ducc0.sht.adjoint_synthesis(map=mp.reshape((1,-1)),
                                       alm=out,
                                       lmax=lmax,
                                       spin=0,
                                       nthreads=nthreads,
                                       **hp_geominfos[nside]).reshape((-1,))


print(f"ducc0 version: {ducc0.__version__}")
ftype = np.float32
nside_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
niter_list = [100, 100, 100, 100, 50, 25, 10, 4, 2, 1, 1, 1]
nthreads_lists = [
  [ 3, 6, 12, 24, 48, 96, 192, 384, 768 ],
  [ 3, 6, 12, 24, 48, 96, 192, 384, 768 ],
  [ 3, 6, 12, 24, 48, 96, 192, 384, 768 ],
  [ 3, 6, 12, 24, 48, 96, 192, 384, 768 ],
  [ 3, 6, 12, 24, 48, 96, 192, 384, 768 ],
  [ 3, 6, 12, 24, 48, 96, 192, 384, 768 ],
  [ 3, 6, 12, 24, 48, 96, 192, 384, 768 ],
  [ 6, 12, 24, 48, 96, 192, 384, 768 ],
  [ 24, 48, 96, 192, 384, 768 ],
  [ 96, 192, 384, 768 ],
  [ 192, 384, 768 ],
  [ 192, 384, 768 ]]
ducc0.misc.resize_thread_pool(nthreads_lists[-1][-1])
ducc0.misc.print_diagnostics()
print(f"{'nside':^5s} {'nthreads':^8s} {'time [s]':^12s} {'fwd time [s]':^12s} {'adj time [s]':^12s}")
#ducc0.misc.preallocate_memory(8)
for nside, nthreads_list, niter in zip(nside_list, nthreads_lists, niter_list):
    npix = 12*nside**2
    lmax = 3*nside
    # filling with random numbers takes forever ...
    m = np.random.normal(0, 1, npix).astype(ftype)
    #m = ducc0.misc.empty_noncritical((npix,),dtype=ftype, nthreads=nthreads_list[-1])
    #m[()] = np.ones(npix, dtype=ftype)
    # Warmup
    alms = alm_to_map_adjoint(m, nside, lmax, nthreads_list[-1])
    m2 = alm_to_map(alms, nside, lmax, nthreads_list[-1])
    prevtime = 1.0
    for nthreads in nthreads_list:
        #ducc0.misc.resize_thread_pool(nthreads)
        t_fwd = t_adj = 0
        for i in range(niter):
            t0 = time()
            alms = alm_to_map_adjoint(m, nside, lmax, nthreads, out=alms)
            t1 = time()
            t_adj += t1-t0
            m2 = alm_to_map(alms, nside, lmax, nthreads, out=m2)
            t_fwd += time()-t1
        t_fwd /= niter
        t_adj /= niter
        if nthreads == nthreads_list[0]:
            reference_t_fwd, reference_t_adj = t_fwd, t_adj
        print(f"{nside:>5d} {nthreads:>8d} {t_fwd+t_adj:>12.3f} {t_fwd:>12.3f} {t_adj:>12.3f} {prevtime/(t_fwd+t_adj)}")
        prevtime = t_fwd+t_adj
