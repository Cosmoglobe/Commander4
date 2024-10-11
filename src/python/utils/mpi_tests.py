# On OpenMPI, run with
# mpirun -np <number of nodes> --map-by node --bind-to none python3 mpi_tests.py 1e9 <number of cores per node>


import numpy as np
from mpi4py import MPI
from time import time
import argparse
import math
from numpy.testing import assert_ as myassert
import ducc0


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, spin, ncomp, rng):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res

def do_benchmark(comm, work, name, msg_size, msg_size_per_comm):
    # warm-up
    comm.Barrier()
    work()
    comm.Barrier()
    t0 = time()
    work()
    t1 = time()
    tmax = comm.allreduce(t1-t0, MPI.MAX)
    if comm.rank == 0:
        print(f"{name:12s}: {tmax:12.3f}s {1e-9*msg_size/tmax:12.3f}GB/s {1e-9*msg_size_per_comm/tmax:12.3f}GB/s")


def alm_to_map(alm: np.array, map: np.array, nside: int, lmax: int, nthreads=1) -> np.array:
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    return ducc0.sht.synthesis(alm=alm.reshape((1,-1)),
                               map=map,
                               lmax=lmax,
                               spin=0,
                               nthreads=nthreads, **geom).reshape((-1,))


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
    num_comm_rounds = math.ceil(np.log2(comm.Get_size()))

    if comm.rank == 0 and size < 1e6:
        print("Warning: small message size. You may not get good estimates for actual performance. We recommend using 1e6 or higher.")

    if comm.rank == 0:
        print(f"Testing buffer-based collective communications on {comm.size} tasks with a message size of {size/1e9}GB")
        print(f"For 1-to-all or all-to-1 communication, {num_comm_rounds} 'rounds' of communications are needed.")
        print(f"{'':12s}  {'Time':>12s} {'Tot. speed':>16s} {'Speed/comm-round':>19s}")

    # warming up
    for i in range(10):
        comm.Barrier()
    do_benchmark(comm, lambda : comm.Barrier(), "Barrier", 0, 1)
    buf = np.ones(int(size/8), dtype=np.float64)
    recvbuf = buf.copy()
    do_benchmark(comm, lambda : comm.Send(buf, 1) if comm.rank == 0 else comm.Recv(recvbuf, 0) if comm.rank == 1 else time, "Send/Recv", size, size)
    myassert((buf==1).all(), "Send/Recv problem")
    do_benchmark(comm, lambda : comm.Bcast(buf, 0), "Bcast", size*(comm.size-1), num_comm_rounds*size)
    myassert((buf==1).all(), "Bcast problem")
    do_benchmark(comm, lambda : comm.Reduce(buf, recvbuf if comm.rank == 0 else None, MPI.SUM, 0), "Reduce", size*(comm.size-1), num_comm_rounds*size)
    myassert((buf==1).all(), "Bcast problem")
    if comm.rank == 0:
        myassert((recvbuf==comm.size).all(), "Reduce problem")
    do_benchmark(comm, lambda : comm.Allreduce(buf, recvbuf, MPI.SUM), "Allreduce", size*2*(comm.size-1), num_comm_rounds*size*2)
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
    num_comm_rounds = math.ceil(np.log2(comm.Get_size()))

    if comm.rank == 0 and size < 1e6:
        print("Warning: small message size. You may not get good estimates for actual performance")

    if comm.rank == 0:
        print(f"Testing in-place buffer-based collective communications on {comm.size} tasks with a message size of {size/1e9}GB")
    # warming up
    for i in range(10):
        comm.Barrier()
    do_benchmark(comm, lambda : comm.Barrier(), "Barrier", 0, 1)
    buf = np.ones(int(size/8), dtype=np.float64)
    myassert((buf==1).all(), "Send/Recv problem")
    do_benchmark(comm, lambda : comm.Bcast(buf, 0), "Bcast", size*(comm.size-1), num_comm_rounds*size)
    myassert((buf==1).all(), "Bcast problem")
    do_benchmark(comm, lambda : comm.Reduce(MPI.IN_PLACE if comm.rank==0 else buf, buf if comm.rank==0 else None, MPI.SUM, 0), "Reduce", size*(comm.size-1), num_comm_rounds*size)
 #   myassert((buf==(comm.size if comm.rank == 0 else 1)).all(), "Reduce problem")
    do_benchmark(comm, lambda : comm.Allreduce(MPI.IN_PLACE, buf, MPI.SUM), "Allreduce", size*2*(comm.size-1), num_comm_rounds*size)
 #   myassert((buf==2*comm.size-1).all(), "Reduce problem")


def collectiveBenchPersistent(comm, size):
    """Tests performance of collective communications of the given size
    on the given communicator. Uses the buffer protocol-based mpi4py
    scheme, i.e. requires array-like objects.
    This routine uses persistent collective, a feature introcuced in the
    MPI 4 standard. It might not work with most current MPI libraries.

    Parameters
    ----------
    comm : MPI comunicator to use
    size : float
        message size in bytes
    """
    num_comm_rounds = math.ceil(np.log2(comm.Get_size()))

    if comm.rank == 0 and size < 1e6:
        print("Warning: small message size. You may not get good estimates for actual performance")

    if comm.rank == 0:
        print(f"Testing buffer-based collective communications on {comm.size} tasks with a message size of {size/1e9}GB")

    # warming up
    for i in range(10):
        comm.Barrier()
    do_benchmark(comm, lambda : comm.Barrier(), "Barrier", 0, 1)
    buf = np.ones(int(size/8), dtype=np.float64)
    recvbuf = buf.copy()
    req = comm.Bcast_init(buf,0)
    do_benchmark(comm, lambda : req.Start(), "Bcast", size*(comm.size-1), num_comm_rounds*size)
    del req
    req = comm.Reduce_init(buf, recvbuf, MPI.SUM, 0)
    do_benchmark(comm, lambda : req.Start(), "Reduce", size*(comm.size-1), num_comm_rounds*size)
    del req
    req = comm.Allreduce_init(buf, recvbuf, MPI.SUM)
    do_benchmark(comm, lambda : req.Start(), "Allreduce", size*2*(comm.size-1), num_comm_rounds*size*2)
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
    num_comm_rounds = math.ceil(np.log2(comm.Get_size()))

    if comm.rank == 0 and size < 1e6:
        print("Warning: small message size. You may not get good estimates for actual performance")

    if comm.rank == 0:
        print(f"Testing object-based collective communications on {comm.size} tasks with a message size of {size/1e9}GB")

    # warming up
    for i in range(10):
        comm.Barrier()

    do_benchmark(comm, lambda : comm.Barrier(), "Barrier", 0, 1)
    buf = np.ones(int(size/8), dtype=np.float64)
    # do_benchmark(comm, lambda : comm.send(buf, 1) if comm.rank == 0 else comm.recv(0) if comm.rank == 1 else time, "send/recv", size, 1)
    myassert((buf==1).all(), "send/recv problem")
    do_benchmark(comm, lambda : comm.bcast(buf, 0), "bcast", size*(comm.size-1), num_comm_rounds*size)
    myassert((buf==1).all(), "bcast problem")
    do_benchmark(comm, lambda : comm.reduce(buf, MPI.SUM, 0), "reduce", size*(comm.size-1), num_comm_rounds*size)
    myassert((buf==1).all(), "reduce problem")
    do_benchmark(comm, lambda : comm.allreduce(buf, MPI.SUM), "allreduce", size*2*(comm.size-1), num_comm_rounds*size*2)
    myassert((buf==1).all(), "allreduce problem")


def bench_SHT(comm, nside, max_nthreads):
    if comm.rank == 0:
        print(f"Benchmarking independent SHTs on every task with 1<=nthreads<={max_nthreads}.")
        print(f"{'nthreads':>10s}   {'tmin':>11s}  {'tmax':>11s}")
    lmax = 2*nside
    rng = np.random.default_rng(48)
    alm = random_alm(lmax, lmax, 0, 1, rng)
    map = np.ones((1,12*nside**2))
    nthreads = 1
    while nthreads <= max_nthreads:
        comm.Barrier()
        t0 = time()
        alm_to_map(alm, map, nside, lmax, nthreads=nthreads)
        t0 = time() - t0
        tmin, tmax = comm.allreduce(t0, MPI.MIN), comm.allreduce(t0, MPI.MAX)
        if comm.rank == 0:
            print(f"{nthreads:10d}: {tmin:12.3f}s{tmax:12.3f}s")
        nthreads *= 2

def report_status(comm):
    for i in range(comm.size):
        if i == comm.rank:
            print(f"task #{comm.rank}:")
            print(f"running on node '{MPI.Get_processor_name()}'")
            import sys
            print("Python", sys.version)
            vers = MPI.Get_version()
            print(f"mpi4py {vers[0]}.{vers[1]}")
            print("MPI:", MPI.Get_library_version())
        comm.Barrier()
   

def one_task_per_node_comm(comm_in):
    # get MPI processor name (will be unique for each compute node)
    procname = MPI.Get_processor_name()
    # get unique list of MPI processor names
    names = list(set(comm_in.allgather(procname)))
    # we need to make sure that "names" has the same ordering everywhere,
    # so we broadcast the version on the root 
    names = comm_in.bcast(names)
    # determine unique index of this task's node
    mynode = names.index(procname)
    # new communicator containing all tasks on this node
    nodecomm = comm_in.Split(mynode, comm_in.rank)
    # am I the master of nodecomm?
    local_master = 1 if nodecomm.rank == 0 else 0
    # split comm_in into a communicator containing all "node masters"
    # and another containing all remaining tasks (which we will ignore)
    mastercomm = comm_in.Split(local_master)
    # if we belong to the remaining tasks, return COMM_NULL
    if not local_master:
        mastercomm = MPI.COMM_NULL
    return mastercomm


if not MPI.Is_initialized():
    MPI.Init()

comm2 = one_task_per_node_comm(MPI.COMM_WORLD)
if comm2 != MPI.COMM_NULL:
#    report_status(comm2)
    parser = argparse.ArgumentParser(
                        prog='mpi_tests',
                        description='Simple MPI benchmarks')
    parser.add_argument('message_size', type=float, help="message size in bytes")
    parser.add_argument('threads_per_node', type=int, help="number of threads to use per node")
    args = parser.parse_args()

    collectiveBench(comm2, args.message_size)
    collectiveBenchSimple(comm2, args.message_size)
    collectiveBenchInplace(comm2, args.message_size)
    #collectiveBenchPersistent(comm2, args.message_size)

    bench_SHT(comm2, 2048, args.threads_per_node)
