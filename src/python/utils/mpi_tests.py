import numpy as np
from mpi4py import MPI
from time import time
import argparse


def do_benchmark(comm, work, name, msg_size):
    comm.Barrier()
    t0 = time()
    work()
    t1 = time()
    tmax = comm.allreduce(t1-t0, MPI.MAX)
    if comm.rank == 0:
        print(f"{name}: {tmax}s, {1e-9*msg_size/tmax}GB/s")


def collectiveBench(comm, size):
    """Tests performance of collective communications of the given size
       on the given communicator. Uses the buffer protocol-based mpi4py
       scheme, i.e. requires array-like objects.

    Parameters
    ----------
    comm : MPI comunicator to use
    size : float
        message size in bytes
    """
    if comm.rank == 0 and size < 1e6:
        print("Warning: small message size. You may not get good estimates for actual performance")

    if comm.rank == 0:
        print(f"Testing buffer-based collective communications on {comm.size} tasks with a message size of {size/1e9}GB")

    # warming up
    for i in range(10):
        comm.Barrier()
    do_benchmark(comm, lambda : comm.Barrier(), "Barrier", 0)
    buf = np.ones(int(size/8), dtype=np.float64)
    recvbuf = buf.copy()
    do_benchmark(comm, lambda : comm.Bcast(buf, 0), "Bcast", size*(comm.size-1))
    do_benchmark(comm, lambda : comm.Reduce(buf, recvbuf, MPI.SUM, 0), "Reduce", size*(comm.size-1))
    do_benchmark(comm, lambda : comm.Allreduce(buf, recvbuf, MPI.SUM), "Allreduce", size*2*(comm.size-1))


def collectiveBenchSimple(comm, size):
    """Tests performance of collective communications of the given size
       on the given communicator. Uses the "simple" mpi4py scheme, i.e. without
       relying on the buffer protocol.

    Parameters
    ----------
    comm : MPI comunicator to use
    size : float
        message size in bytes
    """
    if comm.rank == 0 and size < 1e6:
        print("Warning: small message size. You may not get good estimates for actual performance")

    if comm.rank == 0:
        print(f"Testing object-based collective communications on {comm.size} tasks with a message size of {size/1e9}GB")

    # warming up
    for i in range(10):
        comm.Barrier()

    do_benchmark(comm, lambda : comm.Barrier(), "Barrier", 0)
    buf = np.ones(int(size/8), dtype=np.float64)
    do_benchmark(comm, lambda : comm.bcast(buf, 0), "bcast", size*(comm.size-1))
    do_benchmark(comm, lambda : comm.reduce(buf, MPI.SUM, 0), "reduce", size*(comm.size-1))
    do_benchmark(comm, lambda : comm.allreduce(buf, MPI.SUM), "allreduce", size*2*(comm.size-1))


parser = argparse.ArgumentParser(
                    prog='mpi_tests',
                    description='Simple MPI benchmarks')
parser.add_argument('message_size', type=float, help="message size in bytes")
args = parser.parse_args()

collectiveBench(MPI.COMM_WORLD, args.message_size)
collectiveBenchSimple(MPI.COMM_WORLD, args.message_size)
