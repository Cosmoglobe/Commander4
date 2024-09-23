import numpy as np
from mpi4py import MPI
from time import time

def collectiveBench(comm, size):
    """Tests performance of collective communications of the given size
       on the given communicator. Uses the buffer protocol-based mpi4py
       scheme, i.e. requires array-like objects."""
    if comm.rank == 0 and size < 1e6:
        print("Warning: small message size. You may not get good estimates for actual performance")

    if comm.rank == 0:
        print(f"Testing buffer-based collective communications on {comm.size} tasks with a message size of {size/1e9}GB")

    # warming up
    for i in range(10):
        comm.Barrier()
    t0 = time()
    comm.Barrier()
    t1 = time()
    if comm.rank == 0:
        print(f"Barrier time: {t1-t0}s")
    buf = np.ones(int(size/8), dtype=np.float64)
    recvbuf = buf.copy()
    t0 = time()
    comm.Bcast(buf, 0)
    comm.Barrier()
    t1 = time()
    if comm.rank == 0:
        print(f"Broadcast: {t1-t0}s, {1e-9*size*(comm.size-1)/(t1-t0)}GB/s")
    comm.Barrier()
    t0 = time()
    comm.Reduce(buf, recvbuf, MPI.SUM, 0)
    comm.Barrier()
    t1 = time()
    if comm.rank == 0:
        print(f"Reduce to master: {t1-t0}s, {1e-9*size*(comm.size-1)/(t1-t0)}GB/s")
    comm.Barrier()
    t0 = time()
    comm.Allreduce(buf, recvbuf, MPI.SUM)
    comm.Barrier()
    t1 = time()
    if comm.rank == 0:
        print(f"Allreduce: {t1-t0}s, {1e-9*size*(2*(comm.size-1))/(t1-t0)}GB/s")

def collectiveBenchSimple(comm, size):
    """Tests performance of collective communications of the given size
       on the given communicator. Uses the "simple" mpi4py scheme, i.e. without
       relying on the buffer protocol."""
    if comm.rank == 0 and size < 1e6:
        print("Warning: small message size. You may not get good estimates for actual performance")

    if comm.rank == 0:
        print(f"Testing object-based collective communications on {comm.size} tasks with a message size of {size/1e9}GB")

    # warming up
    for i in range(10):
        comm.Barrier()
    t0 = time()
    comm.Barrier()
    t1 = time()
    if comm.rank == 0:
        print(f"Barrier time: {t1-t0}s")
    buf = np.ones(int(size/8), dtype=np.float64)
    t0 = time()
    recvbuf = comm.bcast(buf, 0)
    comm.Barrier()
    t1 = time()
    del recvbuf
    if comm.rank == 0:
        print(f"Broadcast: {t1-t0}s, {1e-9*size*(comm.size-1)/(t1-t0)}GB/s")
    comm.Barrier()
    t0 = time()
    recvbuf = comm.reduce(buf, MPI.SUM, 0)
    comm.Barrier()
    t1 = time()
    del recvbuf
    if comm.rank == 0:
        print(f"Reduce to master: {t1-t0}s, {1e-9*size*(comm.size-1)/(t1-t0)}GB/s")
    comm.Barrier()
    t0 = time()
    recvbuf = comm.allreduce(buf, MPI.SUM)
    comm.Barrier()
    t1 = time()
    del recvbuf
    if comm.rank == 0:
        print(f"Allreduce: {t1-t0}s, {1e-9*size*(2*(comm.size-1))/(t1-t0)}GB/s")

collectiveBench(MPI.COMM_WORLD, 1e8)
collectiveBenchSimple(MPI.COMM_WORLD, 1e8)
