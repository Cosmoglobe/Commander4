import numpy as np
from numpy.typing import NDArray
import ducc0
from time import time


# Cache for geom_info objects ... pretty small, each entry has a size of O(nside)
# This will be mainly beneficial for small SHTs with high nthreads
hp_geominfos = {}

def alm_to_map(alm: NDArray, nside: int, lmax: int, nthreads=1, out=None) -> NDArray:
    spin = 2 if alm.shape[0] == 2 else 0
    if nside not in hp_geominfos:
        hp_geominfos[nside] = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    out = ducc0.sht.synthesis(alm=alm, map=out, lmax=lmax, spin=spin,
                              nthreads=nthreads, **hp_geominfos[nside])
    return out 


def alm_to_map_adjoint(mp: NDArray, nside: int, lmax: int, nthreads=1, out=None) -> NDArray:
    spin = 2 if mp.shape[0] == 2 else 0
    if nside not in hp_geominfos:
        hp_geominfos[nside] = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    out = None if out is None else out
    out = ducc0.sht.adjoint_synthesis(map=mp, alm=out, lmax=lmax, spin=spin,
                                      nthreads=nthreads, **hp_geominfos[nside])
    return out



print(f"ducc0 version: {ducc0.__version__}")
ftype = np.float32 #np.float64
lmax_nside_factor = 3.0


# nside_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
nside_list = [2048, 4096, 8192, 16384, 32768]
# nside_list = [16384]
# nthreads_lists = [
#   [ 1, 3, 6, 12, 24, 48, 96, 192, 384],  # 128
#   [ 1, 3, 6, 12, 24, 48, 96, 192, 384],  # 256
#   [ 1, 3, 6, 12, 24, 48, 96, 192, 384],  # 512
#   [ 1, 3, 6, 12, 24, 48, 96, 192, 384],  # 1024
#   [ 1, 3, 6, 12, 24, 48, 96, 192, 384],  # 2048
#   [ 1, 3, 6, 12, 24, 48, 96, 192, 384],  # 4096
#   [ 1, 3, 6, 12, 24, 48, 96, 192, 384],  # 8192
#   [ 1, 3, 6, 12, 24, 48, 96, 192, 384],  # 16k
#   [ 384],  # 32k
#   [ 384]]  # 64k
nthreads_lists = [[192, 384],
                  [192, 384],
                  [192, 384],
                  [192, 384],
                  [192, 384],]

# nthreads_lists = [
#   [ 1, 2, 4, 8, 16, 32, 64, 128],  # 128
#   [ 1, 2, 4, 8, 16, 32, 64, 128],  # 256
#   [ 1, 2, 4, 8, 16, 32, 64, 128],  # 512
#   [ 1, 2, 4, 8, 16, 32, 64, 128],  # 1024
#   [ 1, 2, 4, 8, 16, 32, 64, 128],  # 2048
#   [ 1, 2, 4, 8, 16, 32, 64, 128],  # 4096
#   [ 1, 2, 4, 8, 16, 32, 64, 128],  # 8192
#   [ 1, 2, 4, 8, 16, 32, 64, 128],  # 16k
#   [128],  # 32k
#   [128]]  # 64k

# ducc0.misc.resize_thread_pool(nthreads_lists[-1][-1])
ducc0.misc.print_diagnostics()
print(f"{'nside':^5s} {'nthreads':^8s} {'time [s]':^12s} {'fwd time [s]':^12s} {'adj time [s]':^12s}")
#ducc0.misc.preallocate_memory(8)


# npol = 2
for npol in [1, 2]:
    print(f"npol = {npol}, dtype = {ftype}, lmax = {lmax_nside_factor} x nside")
    print(ducc0.__file__)
    print(nside_list)
    for nside, nthreads_list in zip(nside_list, nthreads_lists):
        npix = 12*nside**2
        lmax = int(lmax_nside_factor*nside)
        # filling with random numbers takes forever ...
        m = np.random.normal(0, 1, (npol, npix)).astype(ftype)
        #m = ducc0.misc.empty_noncritical((npix,),dtype=ftype, nthreads=nthreads_list[-1])
        #m[()] = np.ones(npix, dtype=ftype)
        # Warmup
        alms = alm_to_map_adjoint(m, nside, lmax, nthreads_list[-1])
        m2 = alm_to_map(alms, nside, lmax, nthreads_list[-1])
        print(m.dtype, m2.dtype, alms.dtype)
        prevtime = 1.0
        for nthreads in nthreads_list:
            #ducc0.misc.resize_thread_pool(nthreads)
            t_fwd = t_adj = 0
            # for i in range(niter):
            niter = 0
            while t_fwd + t_adj < 90:
                t0 = time()
                alms = alm_to_map_adjoint(m, nside, lmax, nthreads, out=alms)
                t1 = time()
                t_adj += t1-t0
                m2 = alm_to_map(alms, nside, lmax, nthreads, out=m2)
                t_fwd += time()-t1
                niter += 1
            t_fwd /= niter
            t_adj /= niter
            if nthreads == nthreads_list[0]:
                reference_t_fwd, reference_t_adj = t_fwd, t_adj
            print(f"{nside:>5d} {nthreads:>8d} {t_fwd+t_adj:>12.5f} {t_fwd:>12.5f} {t_adj:>12.5f} {niter:>12d} {prevtime/(t_fwd+t_adj):>12.5f}")
            prevtime = t_fwd+t_adj
