import time
t0 = time.time()
# Lots of imports to test startup time:
from mpi4py import MPI
import numpy as np
import os
# import matplotlib.pyplot as plt
# import healpy as hp
# import ducc0
# import pixell
# import cProfile
# import pstats
# import logging
# import os
# import yaml

comm = MPI.COMM_WORLD
size, rank = comm.Get_size(), comm.Get_rank()
comm.Barrier()
if rank == 0:
    print(f"World size: {size}. Startup time: {time.time() - t0:.3f}s")
t0 = time.time()
print(f"{os.sched_getaffinity(0)}")

# N = int(1e9//8)  # 1GB per task.
# 100 MB, 1GB, 10 GB
Ns = [int(1e8//8), int(1e9//8), int(1e10//8)]
N_str = ["100 MB", "1 GB", "10 GB"]

for N, N_str in zip(Ns, N_str):
    if rank == 0:
        print(f"\nTransfer size (per rank): {N_str}.")
    data = np.random.randn(N)
    comm.Barrier()
    t_used = time.time() - t0
    if rank == 0:
        print(f"RNG time:       {t_used:.3f} s")
    t0 = time.time()

    if rank != 0:
        comm.Send(data, dest=0)
    else:
        for i in range(1, size):
            comm.Recv(data, source=i)

    comm.Barrier()
    t_used = time.time() - t0
    GB_s = 8e-9*(size-1)*N/t_used
    if rank == 0:
        print(f"Send/Recv time: {t_used:.3f} s  {GB_s:.3f} GB/s.")
    t0 = time.time()

    if rank == 0:
        data_recv = np.zeros_like(data)
        comm.Reduce(data, data_recv, root=0)
    else:
        comm.Reduce(data, None, root=0)

    comm.Barrier()
    t_used = time.time() - t0
    GB_s = 8e-9*(size-1)*N/t_used
    if rank == 0:
        print(f"Reduce time:    {t_used:.3f} s  {GB_s:.3f} GB/s.")
    t0 = time.time()

