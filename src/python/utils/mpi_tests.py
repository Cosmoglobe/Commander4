import numpy as np
from mpi4py import MPI
from time import time
import argparse
from numpy.testing import assert_ as myassert


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
    myassert((buf==1).all(), "Bcast problem")
    do_benchmark(comm, lambda : comm.Reduce(buf, recvbuf if comm.rank == 0 else None, MPI.SUM, 0), "Reduce", size*(comm.size-1))
    myassert((buf==1).all(), "Bcast problem")
    if comm.rank == 0:
        myassert((recvbuf==comm.size).all(), "Reduce problem")
    do_benchmark(comm, lambda : comm.Allreduce(buf, recvbuf, MPI.SUM), "Allreduce", size*2*(comm.size-1))
    myassert((buf==1).all(), "Bcast problem")
    myassert((recvbuf==comm.size).all(), "Allreduce problem")


def collectiveBenchInplace(comm, size):
    """Tests performance of collective communications of the given size
    on the given communicator. Uses the buffer protocol-based mpi4py
    scheme, i.e. requires array-like objects.
    Where possible, in-place communication is used.

    Parameters
    ----------
    comm : MPI comunicator to use
    size : float
        message size in bytes
    """
    if comm.rank == 0 and size < 1e6:
        print("Warning: small message size. You may not get good estimates for actual performance")

    if comm.rank == 0:
        print(f"Testing in-place buffer-based collective communications on {comm.size} tasks with a message size of {size/1e9}GB")

    # warming up
    for i in range(10):
        comm.Barrier()
    do_benchmark(comm, lambda : comm.Barrier(), "Barrier", 0)
    buf = np.ones(int(size/8), dtype=np.float64)
    do_benchmark(comm, lambda : comm.Bcast(buf, 0), "Bcast", size*(comm.size-1))
    myassert((buf==1).all(), "Bcast problem")
    do_benchmark(comm, lambda : comm.Reduce(MPI.IN_PLACE if comm.rank==0 else buf, buf if comm.rank==0 else None, MPI.SUM, 0), "Reduce", size*(comm.size-1))
    myassert((buf==(comm.size if comm.rank == 0 else 1)).all(), "Reduce problem")
    do_benchmark(comm, lambda : comm.Allreduce(MPI.IN_PLACE, buf, MPI.SUM), "Allreduce", size*2*(comm.size-1))
    myassert((buf==2*comm.size-1).all(), "Reduce problem")


def collectiveBenchPersistent(comm, size):
    """Tests performance of collective communications of the given size
    on the given communicator. Uses the buffer protocol-based mpi4py
    scheme, i.e. requires array-like objects.
    This routine uses persistent collective, a feature introcuced in the
    MPI 4 standard. It might not work wir most current MPI libraries.

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
    req = comm.Bcast_init(buf,0)
    do_benchmark(comm, lambda : req.Start(), "Bcast", size*(comm.size-1))
    del req
    req = comm.Reduce_init(buf, recvbuf, MPI.SUM, 0)
    do_benchmark(comm, lambda : req.Start(), "Reduce", size*(comm.size-1))
    del req
    req = comm.Allreduce_init(buf, recvbuf, MPI.SUM)
    do_benchmark(comm, lambda : req.Start(), "Allreduce", size*2*(comm.size-1))
    del req


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
    myassert((buf==1).all(), "Bcast problem")
    do_benchmark(comm, lambda : comm.reduce(buf, MPI.SUM, 0), "reduce", size*(comm.size-1))
    myassert((buf==1).all(), "Bcast problem")
    do_benchmark(comm, lambda : comm.allreduce(buf, MPI.SUM), "allreduce", size*2*(comm.size-1))
    myassert((buf==1).all(), "Bcast problem")


parser = argparse.ArgumentParser(
                    prog='mpi_tests',
                    description='Simple MPI benchmarks')
parser.add_argument('message_size', type=float, help="message size in bytes")
args = parser.parse_args()

collectiveBench(MPI.COMM_WORLD, args.message_size)
collectiveBenchSimple(MPI.COMM_WORLD, args.message_size)
collectiveBenchInplace(MPI.COMM_WORLD, args.message_size)
#collectiveBenchPersistent(MPI.COMM_WORLD, args.message_size)
