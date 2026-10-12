# Thread-parallel TOD processing: prototype results

> **Status.** Prototype on branch `no_GIL_test`, built 2026-10-12 on owl39 (AMD EPYC 9965, 384
> cores, 8 NUMA domains of 48 cores). It implements the design in
> `thread_parallel_tod_design.md`. Measured with Planck HFI 143 GHz on Python 3.15.0t
> (free-threaded, `PYTHON_GIL=0`), with Python 3.15.0 (normal build) for comparison. Run logs are
> in `c4_chains/threads_test/`.

## 1. Summary

**It runs, safely and stably, today.** On free-threaded Python 3.15t with `PYTHON_GIL=0`, the
scan loops of all HFI steps run on worker threads. After two fixes (§5.1, §5.2), all 14 threaded
runs since have done their work without errors, on 2 to 24 ranks with 8 to 96 workers per rank.
One of them then hung while the process exited, in the interpreter's memory allocator (§5.5).

**The results are identical.** With the same ranks, a run with 8 workers per rank and one without
workers give chain files that agree to the last bit: all 34 datasets, maps included, over three
Gibbs iterations. Changing the number of ranks changes the order of the MPI sums; the chains then
differ by about 0.01 posterior standard deviations after three iterations (§4.1).

**It is faster: 28% on the full 143 GHz band**, with 8 ranks per node (one per NUMA domain), each
running 24 workers of 2 cores, and the scans dealt to the ranks round-robin: 1021–1026 s per chain
pass against 1422–1426 s for today's 192 ranks of 2 cores (§4.4). The gain comes almost entirely
from waiting that disappears, not from faster computing:

- Within a rank, workers take the next scan as they become free, so a rank never waits for one
  slow scan. Between ranks the waiting stays unless every rank gets a similar mix of scans; with
  1000+ scans per rank, dealing them round-robin achieves that (waiting 6 s instead of 259 s).
- With the scans in blocks, as today's split gives with these file lists, threads give only 10%
  on the full band (1279–1285 s) and 20–25% on a 2000-scan subset.
- The computing itself does not get faster. The mapmaker is limited by the node's memory
  bandwidth: once every core is busy, a scan costs the same whichever thread runs it.
- Larger ranks also make the far-sidelobe setup (built by one rank per node with its own cores)
  and the map gathering faster, and need less node memory: 100 GB against 177 GB on the subset
  (not measured on the full band).
- 192 × 2 is the right baseline: today's code is 5–50% slower with 384 ranks of 1 core, and no
  faster with 128 × 3 or larger ranks (§4.2).
- On the normal (GIL) build the threaded layout is slower than today's (149–155 s against
  123–135 s per pass on the subset).

## 2. What the prototype changes

| Piece | Where | What |
|---|---|---|
| Worker pool | `threads.py` (new) | One pool per rank; `run_in_order` runs tasks on it and returns results in task order, with at most 4 tasks per worker in flight. `library_nthreads()` gives a library call the rank's cores, or a worker's share of them. |
| `TODView.map` | `tod/view.py` | Runs a per-scan function on every detector-scan, each on its own fresh view, and yields the results in scan order. Adds each scan's time to `scan_runtime`. |
| Per-scan random numbers | `TODView.rng` | Seeded from run seed, chain, iteration, step name, scan ID and detector. Passed down to the correlated-noise sampler, gap filling, PSD sampler and gain gap fill as `rng`. The numba gap filler gets a seed drawn from it. |
| Converted loops | `gain.py`, `hfi_demodulation.py`, `tod2map.py`, `jumps/sampling.py`, `pixel_domain.py`, `base_reader.py` | Absolute, relative and temporal gain (one shared per-scan function), HFI baselines and modulation phase, the mapmaking loop, jump detection, the pixel-domain setup, and reading. The glitch sampler is still a placeholder and is not converted; nor is the CG mapmaker's operator loop. |
| Map binning | `mapmaking/binned.py`, `mapmaker_pymod.cc` | Each worker bins its scan into small maps over the scan's own pixels (`bin_scan`, new kernel `distinct_pixels`). The main thread adds these to the rank's maps in scan order (`add_scan`, new kernel `add_to_pixels`). The C++ kernels release the GIL. |
| Benchmark timer | `diagnostics/performance.py` | One stack of open timers per thread, a lock on the totals. Workers file their entries under their caller's open tags; their times are summed over the workers. |
| Settings | `mpi/setup.py` | New `resources.tod.scan_threads` (default 1: no workers, as before). TOD ranks get one BLAS thread (§5.3). |
| Library threads | `fft.py`, `pointing.py`, `sky_projection.py`, `sidelobe_deconvolve.py` | Read `library_nthreads()` instead of `OMP_NUM_THREADS`. Workers set numba's thread count for themselves. The far-sidelobe projector gets a second ducc0 plan with a worker's thread count (§5.2). |
| numba | `threads.py` | A lock around numba's compilation (§5.1). |

The source changes are about 700 lines added and 430 removed in 20 files, plus 86 lines in
`threads.py`. All 778 tests pass on 3.15t and on 3.15; `tests/test_scan_threads.py` is new.

A converted loop looks like this (absolute gain):

```python
for _, _, d_term, s_term in scan_view.map(_calibration_terms, "abs", gain_cfg,
                                          step="abs-gain", accepted_only=True):
    sum_s_T_N_inv_d += d_term
    sum_s_T_N_inv_s += s_term
```

## 3. Test setup

- Data: HFI 143 GHz, all 11 detectors. Most tests use scans 51–2050 (1984 scans, 21,800
  detector-scans); the full-band test uses scans 51–23938 (23,311 scans). Same settings as
  `param_PlanckHFI_143GHz.yml` (binned mapmaker, gains, correlated noise, far sidelobes, HFI
  baselines), TOD only, no component separation.
- Two parameter variants: three Gibbs iterations, where gain, correlated noise and far sidelobes
  start in iteration 2 (as in production); and one iteration with every step on from the start,
  used for the layout comparisons. Each iteration runs both chains, so the timings below are per
  chain pass.
- File lists: hke's `filelist_143.txt`, which has no ntod column and no pointing-plane normals.
  The scans are therefore split between ranks in equal blocks of consecutive scans.
- One owl node, nothing else running. Each run starts from a frozen copy of the code.
- "a × b" means a ranks of b cores; "s" is the number of workers per rank (`scan_threads`).

## 4. Results

### 4.1 Identical results

**Same ranks, 1 against 8 workers: identical to the last bit.** Two runs of three Gibbs iterations
on 24 × 16, one with 8 workers per rank and one with none (`scan_threads: 1`, the plain serial
loop). Every dataset of the chain files, in all three iterations, is identical: gains, noise
parameters, baselines, χ², power spectra, and all maps (34 of 34 datasets; the timing
`scan_runtime` left out). This needed the fixes in §5.2 and §5.3; before the BLAS fix, the two
runs differed from iteration 2 on, at the level of the noise CG's tolerance (§5.3).

**Different numbers of ranks: the same to round-off at first, then slightly apart.** 192 × 2
against 24 × 16 with 8 workers:

| Quantity | Iteration 1 | Iteration 2 | Iteration 3 |
|---|---|---|---|
| Absolute gain, relative difference | 2e-16 | 1.7e-6 | 1.4e-6 |
| Noise parameters, largest relative difference | identical | 6e-8 | 8e-6 |
| Weight map (`cov`), largest relative difference | 9e-16 | 4e-6 | 3e-6 |
| Sky map, largest relative difference | 9e-10 | 2e-6 | 8e-6 |

The MPI sums over ranks add in another order, so the first differences are at round-off. They
then grow, but I have not traced where: the absolute gain of iteration 2 already differs by
1.7e-6, although everything stored from iteration 1 agrees to round-off and no noise CG (§5.3)
has run before it. (The test data were being rewritten when this was found, so it could not be
rerun.) Still, the largest gain difference is about 0.01 posterior standard deviations (relative
1.7e-6, against a posterior width of 1.5e-4). With today's code the random numbers depend on the
rank, so two layouts differ by about one standard deviation; the per-scan random streams remove
that.

### 4.2 Speed

One chain pass, 2000 scans, every step on (seconds; rank averages from the performance reports,
second pass):

| Layout | Workers × cores | Pass | Gain steps | Far-sidelobe setup | Mapmaker | ... of which waiting | Map gathering |
|---|---|---|---|---|---|---|---|
| 192 × 2, today's code | — | 122.7 | 25.6 | 29.2 | 65.4 | 10.5 | 4.2 |
| 192 × 2, new code, no workers | — | 123.7 | 25.2 | 29.9 | 66.1 | 11.4 | 4.2 |
| 384 × 1, today's code | — | 186.9 | 24.7 | 71.1 | 88.3 | 15.7 | 6.0 |
| 384 × 1, new code, no workers | — | 190.7 | 25.1 | 73.5 | 89.3 | 17.0 | 6.0 |
| **24 × 16** | 8 × 2 | **98.8** | 20.6 | 12.1 | 63.8 | 9.5 | 2.6 |
| 16 × 24 | 12 × 2 | 107.7 | 20.2 | 16.5 | 68.8 | 13.0 | 2.5 |
| 8 × 48 | 24 × 2 | 118.2 | 20.7 | 30.7 | 64.6 | 7.4 | 2.6 |
| 8 × 48 | 16 × 3 | 107.6 | 25.7 | 20.8 | 58.8 | 5.0 | 2.6 |
| 8 × 48, scans round-robin | 24 × 2 | 103.7 | 19.2 | 20.8 | 61.4 | 0.8 | 5.7 |
| 8 × 48, normal GIL build | 24 × 2 | 149.4 | 26.6 | 20.8 | 99.8 | 9.2 | 2.8 |

"Gain steps" are HFI baselines plus absolute and relative gain. The first pass of each run is
13 s (today's code) to 3 s (threaded) slower, from first-time setup.

- **The new code without workers is as fast as today's.** The `map` loops, per-scan random numbers
  and two-step binning cost nothing measurable.
- **A fully populated node is the weaker baseline.** Today's code on 384 ranks of 1 core is 50%
  slower than on 192 ranks of 2 cores (187 against 123 s per pass). Correlated-noise sampling costs
  0.95 core-seconds per scan on 1-core ranks against 0.67 on 2-core ranks, and the far-sidelobe
  cubes are built with 1 core each. The comparisons therefore use 192 × 2.
- **Two cores per worker beat one**, for the same reason. With an earlier version of the binning,
  8 × 48 took 123–125 s per pass with 40 single-core workers, and 108–109 s with 24 workers of 2
  cores. One scan per core puts twice as many scans in flight, and their FFT arrays (about 10 MB
  each) no longer fit in the 32 MB of L3 cache that 16 cores share. Each scan then waits longer on
  memory. The second core itself helps the FFTs only a little: ducc0 splits a long 1-D FFT (over
  10,000 points) into two factors and shares the sub-FFTs between threads, which makes an
  800,000-point DCT 1.2× faster on 2 threads and 1.6× on 4.
- **The far-sidelobe setup depends on the size of one rank**: one rank per node builds all 11
  detector cubes with its own cores, and the other ranks share them. With 1 core it takes 70 s,
  with 2 cores 29 s, with 16 cores 12 s, and with 48 cores 19–21 s, where ducc0 apparently stops
  scaling.
- **Map gathering is faster with fewer ranks** (4.2 s → 2.6 s).
- **The gain steps are 20–25% faster**: the ranks no longer wait for each other at the gain
  barrier (2.7 s → 1.1 s), and the per-scan work is somewhat cheaper.

Today's code in other layouts (seconds per pass, first / second):

| Layout | Pass | Gain steps | Far-sidelobe setup | Mapmaker |
|---|---|---|---|---|
| 384 × 1 | 212.7 / 186.9 | 24.7 | 71.1 | 88.3 |
| 192 × 2 | 135.6 / 123.7 | 25.6 | 30.4 | 65.2 |
| 128 × 3 | 132.1 / 126.0 | 31.9 | 21.9 | 70.0 |
| 96 × 4 | 139.0 / 134.4 | 36.4 | 18.1 | 77.6 |
| 48 × 8 | 218.6 / 214.2 | 64.1 | 13.8 | 134.0 |

192 × 2 and 128 × 3 are today's fastest layouts. With fewer, larger ranks the per-scan Python
and numpy work runs on one core while the others wait; with 1-core ranks too many scans compete
for cache and memory bandwidth.

Node memory, the highest use during the run minus the idle node, sampled every 5 s:

| Layout | Node memory |
|---|---|
| Today's code, 192 × 2 | 177 GB |
| Today's code, 96 × 4 | 122 GB |
| Today's code, 48 × 8 | 101 GB |
| 24 × 16, 8 × 2 workers | 100 GB |

The threaded layout needs 57% of the memory of today's 192 × 2, about what today's code needs
with 48 large ranks. (The performance reports' per-rank memory, summed over ranks, overstates
layouts with many ranks: it counts shared memory, such as the 26 GB of far-sidelobe cubes and the
libraries, once per process.)

Garbage collection is negligible: about 18 collections per rank and run, 0.2 s in total, the
longest 0.05 s.

### 4.3 Waiting between ranks

Threads remove the waiting *within* a rank: a rank's workers take the next scan as they become
free. They do not remove differences *between* ranks. With these file lists every rank gets an
equal block of consecutive scans, and some blocks cost more, for example because they cross the
Galactic plane, where more samples are masked and the noise CG needs more iterations. The total
correlated-noise time of the 24 ranks of 24 × 16 ranges from 265 to 405 thread-seconds; so the
fastest rank waits 17 s for the slowest.

Dealing the scans round-robin in time (scan i to rank i mod 8) gives every rank the same mix of
sky. With 248 scans per rank the sums then agree within 3%, and the waiting falls from 7.4 s to
0.8 s. The price: every rank's maps cover the whole observed sky, so map gathering takes 5.7 s
instead of 2.6 s and memory grows 30%; and reading took 47 s instead of 18 s, since each rank
opens files from the whole period. For 192 ranks of 2 cores, round-robin does not help: 10 scans
per rank are too few to average out, the waiting only falls from 11.4 to 7.9 s, and map
gathering grows from 4.2 to 12.4 s.

With proper file lists, the split goes by sky position and number of samples
(`split_by_sky_position`), which removes the time-block effect but keeps the sky-region one.

### 4.4 Full 143 GHz band

All 23,311 scans, one iteration with every step on, so two chain passes per run (seconds, rank
averages from the second pass):

| | Today's code, 192 × 2 | 24 × 16, 8 × 2 workers | 8 × 48, 24 × 2 workers, round-robin |
|---|---|---|---|
| Pass (chain 1 / chain 2) | 1426 / 1422 | 1285 / 1279 | **1026 / 1021** |
| HFI baselines | 149 | 123 | 94 |
| Absolute gain (of which waiting) | 181 (45) | 168 (51) | 100 (0.4) |
| Relative gain | 156 | 144 | 100 |
| Far-sidelobe setup | 29 | 13 | 19 |
| Mapmaker | 904 | 828 | 706 |
| ... of which waiting at the barrier (average / most) | 259 / 577 | 203 / 333 | 6 / 11 |
| Map gathering | 17 | 11 | 18 |
| Node memory in the mapmaker (largest rank) | 916 GB (43 GB) | 720 GB (60 GB) | 816 GB (118 GB) |
| Reading, once per run | 104 | 322 | 619 |

- **Threads with an even scan split are 28% faster** than today's code. Nearly all of the gain is
  waiting that disappears: in today's run the ranks wait 259 s on average in the mapmaker and 45 s
  at the absolute-gain barrier, and the slowest rank of each step sets its time. The even split
  needs the large ranks: on the subset, round-robin made today's 192 ranks slower (§4.3).
- **Threads alone, with the scans in blocks, give 10%.** With these file lists the blocks hold
  equal numbers of scans, but full-mission scans differ a lot in length: the correlated-noise
  work of the 24 ranks ranges from 2834 to 4897 thread-seconds. Today's 192 ranks see the same
  effect, more strongly.
- With proper file lists (ntod and normals), the split weighs scans by their number of samples,
  which removes most of the length effect for both layouts; what is left is the cost difference
  between sky regions (§4.3).
- Round-robin costs reading time (619 s against 104 s, about 1% of a 20-iteration run) and memory
  per rank, since every rank's maps cover the whole sky.

### 4.5 Reading

Reading uses the workers too, but h5py runs one HDF5 call at a time per process. For 2000 scans:
12.5 s with 192 ranks, 52–60 s with 8 ranks reading one scan at a time, and 16–32 s with 8 ranks
and workers (the spread comes from the file cache). The full band (23,311 scans) took 322 s with
24 × 16. This happens once per run.

## 5. Problems found

### 5.1 numba compiles unsafely from several threads

The first call of a numba kernel compiles it. When several workers make that first call at the same
time, numba 0.68 can crash in its compile-timing code (`AttributeError: 'TimingListener' object has
no attribute '_ts'`; it happened in the sixth threaded run). numba keeps these timers in one
global registry, which it changes outside its own compiler lock. **Fix:** a lock around numba's
`Dispatcher.compile` (`threads._serialize_numba_compilation`). A kernel compiles once per argument
types, so the lock costs nothing afterwards. This should be reported to numba.

### 5.2 ducc0 picks a different far-sidelobe kernel for different thread counts

ducc0's `ConvolverPlan` chooses the cube's oversampling σ and the interpolation kernel's width by
minimizing an estimated cost of building the cube plus interpolating the samples. The estimate
assumes that interpolation scales perfectly with the thread count and building at most 6×
(`nufft/nufft_common.h`), so with more threads it picks a smaller cube with a wider kernel. For
lmax = mmax = 256 and ε = 1e-4: σ = 1.6 up to 32 threads (25.9 GiB of cubes for 11 detectors),
1.4 at 48 (17.2 GiB) and about 1.25 from 96 on (12.4 GiB). A cube can only be read with the kernel
it was built for. The prototype built the cubes with the rank's 48 threads and gave the workers a
second plan with 1 thread (a plan's thread count is fixed, and 24 workers each asking for 48
threads would swamp the cores), so the two plans disagreed. Without a
check, every far-sidelobe value would have been silently wrong; with the check the run stopped.
**Today's code is correct, but its results depend on `num_threads`:** with 48 or more threads per
rank, the far-sidelobe TOD is computed with another kernel than with 2 threads. Both meet the
requested accuracy: built from the same random sky and beam, the σ = 1.6 and σ = 1.4 cubes give
values that differ by 2.7e-5 (relative L2 difference; at most 1.6e-4 of the RMS).
**Fix:** pin the oversampling to 1.6 (`sigma_min=1.6, sigma_max=1.6+1e-6`), which reproduces
today's 2-thread results exactly, for any thread count.

### 5.3 OpenBLAS changes results with its thread count

OpenBLAS splits a long dot product between its threads, which changes the order of the additions:
the same 800,000-sample `np.dot` gives `...214909` on 1 thread and `...21491` on 16, a last-digit
difference. The correlated-noise CG takes its step sizes from such dot products (pixell's `CG`
uses `a.dot(b)`), but in float64 that alone does no harm: with the CG's filter in float64, runs on
1 and 16 BLAS threads stay within 6e-15 of each other over all 64 iterations of a test scan.

What spreads the difference is the CG's Fourier filter, which works in float32, as in
Commander3. Each iteration rounds the 118,432 gap samples of the test scan to float32. While the
two runs differ by about 1e-14, that rounding hides the difference, except for a sample that
happens to lie on a float32 rounding boundary. At iteration 7 one did: the runs then differed by a
full float32 step (6e-8) in that sample, which more samples followed every iteration. After 64
iterations the solutions differed by 3.5e-6, below the CG's own tolerance (1e-5). On 8 test scans
the noise parameters differed by up to 6e-8. Both answers are equally valid; the float32 filter
was chosen knowing its error. **Fix:** TOD ranks always get one BLAS thread, so the float64
inputs to the filter are identical. ducc0's FFTs and interpolation, numba's per-sample loops and
the map kernels give identical results for any thread count (checked).

### 5.4 Map binning in the main thread became the bottleneck

The first version kept the map binning in the main thread, in scan order, as the design proposed
(option C). It took 20 ms per detector-scan, the same as in today's ranks, but now one thread did
it for 2725 scans per rank: 57 s, longer than the workers needed. Splitting the binning kernel over
pixel blocks gave the same bits for any thread count, but needed cores the workers were using, and
running more threads than cores slowed everything. **The working design:** each worker bins its
own scan into small maps over the scan's own pixels (`bin_scan`); the main thread only adds these
to the rank's maps in scan order (`add_scan`), 5 ms per scan. The sums are added in the same order
whatever the thread count, so the maps stay identical, and no per-thread full-size maps are needed.
This replaces options A–C of the design note (§3.7).

### 5.5 Smaller points

- **Memory bandwidth limits the mapmaker.** Per scan, correlated-noise sampling costs 0.27 s with
  2 cores when the workers are held back, but 0.38 s when all cores are busy; today's ranks take
  0.33 s. Fewer, larger ranks do not change this.
- **The normal (GIL) build gains nothing.** The numba kernels and the Python code hold the GIL;
  only the ducc0 FFTs and the C++ kernels run in parallel.
- **Ranks across NUMA domains are slow.** Two ranks of 192 cores (one per socket) took 144 s per
  pass, and the first pass 379 s.
- **`np.random` in numba is per thread.** The gap filler now seeds numba's generator from the
  scan's `rng`; before, its draws were never seeded, so no run could be repeated exactly.
- **A process hung at exit.** After the full-band run on 24 × 16 had finished and written
  everything, one of the 24 processes spun for over an hour at exit, inside the free-threaded
  interpreter's memory allocator (mimalloc's `mi_process_done`, walking its heap pages). The
  prototype's launcher now finalizes MPI and leaves with `os._exit`, skipping that cleanup; the
  later full-band run exited normally. A Commander4 release on 3.15t would need the same, or a
  fixed CPython.

## 6. Consequences for the design

- **Layout:** one rank per NUMA domain (8 per owl node), each with 24 workers of 2 cores, and an
  even scan split was fastest on the full band. With scans in blocks, ranks of 16 cores did best
  on the subset. Workers should have 2 cores, not 1.
- **The round-robin split is not on the branch.** It was a test change of one line in
  `read_tods_from_file` (`scan_ranks = np.arange(len(scans)) % nranks`). How to split scans for
  large ranks, by time, sky or measured cost, belongs with the separate load-balancing work.
- **Map accumulation:** per-scan binning plus ordered adding (§5.4), not atomics or per-thread maps.
- **Determinism needs one BLAS thread, a fixed ducc0 kernel, and seeded numba draws** (§5.2–5.3,
  §5.5). With them, runs with the same rank layout agree to the last bit; changing the number of
  ranks changes the order of the MPI sums, and those differences then grow through the chain.
- **The remaining waiting is between ranks**, so the scan split matters more than threads. With
  large ranks, dealing scans evenly over time or sky becomes affordable.
- **Still to convert:** the CG mapmaker's operator loop and the glitch sampler, once it is real.
- **Blockers left before production:** numba's thread safety (§5.1), ducc0 and the other
  dependencies declaring free-threading support, healpy wheels for 3.15.

## 7. How to reproduce

```yaml
resources:
  tod:
    num_threads: 48
    scan_threads: 24
```

```bash
source ~/my_venvs/.venv_py315t/bin/activate
export PYTHON_GIL=0 I_MPI_PIN_DOMAIN=48:compact   # one rank per NUMA domain
mpirun -n 8 commander4 -p params.yml
```

The C++ backend must be rebuilt for this branch (`pip install --no-deps -e .`), in every venv
that uses it; the 3.13 venv `.com4_main_venv` has not been rebuilt. Long runs may hang at exit
(§5.5); the benchmark runs used a small launcher that ends with `MPI.Finalize()` and `os._exit`.
The scripts used (launcher, run, compare chain files, summarize reports) are in the session's
scratch directory and can be added to `aux/` on request.
