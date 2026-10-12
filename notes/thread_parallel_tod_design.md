# Thread-parallel TOD processing (design proposal)

> **Status.** Proposal, not implemented. The target is roughly a year from now (late 2027), when
> free-threaded Python and our dependencies are expected to be ready. Measurements come from two HFI
> runs with 1344 TOD ranks of 2 threads each, on 7 owl nodes: one before load balancing
> (2026-10-10, `run-20261010T115422.964767+0200-3oq8.log` in `c4_chains/HFI/logs/`) and one after
> (2026-10-11, `run-20261011T140032.584426+0200-ha5m.log` in
> `c4_chains/HFI_balanced-ntod_band-auto-balanced/logs/`), and from small tests on owl39 with
> Python 3.15.0 and 3.15.0t (Appendix A). Line numbers refer to `0d87c9a` and will drift.
>
> **Prototype (2026-10-12).** A working prototype is on branch `no_GIL_test`; its results are in
> `thread_prototype_results.md`. It changed three parts of this proposal: map accumulation (§3.7:
> per-scan binning, not options A–C), cores per worker (§3.5: 2, not 1), and three new sources of
> run-to-run differences (§4, points 11–13).

## 1. Summary

Today every TOD rank processes its scans one after the other. Node memory limits how many ranks we
can start, so we usually give each rank 2–4 cores. The extra cores only work inside ducc0 FFTs and
numba loops; the rest of the time they sit idle.

The proposal is to run fewer, larger ranks (one per NUMA domain, or 4–8 cores per rank) and spread
each rank's detector-scans over a pool of threads. A NUMA domain is a group of cores with its own
nearby memory; an owl node has 8 of them.

1. **One way to loop over scans.** A new method `TODView.map(func, ...)` replaces the
   `for view in scan_view.iter_focused():` loops. The science stays in a per-scan function.
   All thread handling lives inside `map` and four support pieces: per-scan random numbers, a
   library-thread helper, a thread-safe benchmark timer and a thread-safe map accumulator.
2. **Gains.** All cores work all the time; scans are shared out within a rank as threads become
   free, which evens out the load imbalance that remains after splitting scans by number of samples
   (ranks still wait about 16% of the mapmaker step, §2.2); and heavy and light work share the
   rank's memory instead of every rank being sized for the worst case.
3. **Main risks.** Our dependencies do not yet support free-threaded Python; some code is safe today
   only because the GIL serializes it (§4); every scan loop that is not converted becomes slower
   (§5.2); and HDF5 reading cannot be parallelized within one process (§3.9).
4. **Most of the work can start now.** The support pieces and the loop conversions change nothing
   with 1 thread. On the normal (GIL) Python build they already give part of the speedup, because
   ducc0, numpy and our Huffman decoder release the GIL (§2.3).

## 2. Starting point

### 2.1 Hardware

An owl node (AMD EPYC 9965) has 384 cores in 2 sockets, split into 8 NUMA domains of 48 cores, and
1.5 TB of RAM: about 3.9 GB per core.

### 2.2 What the HFI runs show

Two runs with the same setup. In the balanced run, the scans of each band were split between its
ranks by number of samples and sky position (`split_by_sky_position`, `file_io/tod_reader.py:48`),
and the ranks were split between the bands by detectors × samples (`band_weight: auto`). Numbers
for iteration 2 of chain 1, from the performance reports in the logs:

| | Before | Balanced |
|---|---|---|
| Mapmaker step, average per rank (all bands) | 1377 s | 1049 s |
| ... of which waiting at the barrier after the scan loop (`MPI-sync`) | 630 s | 173 s |
| Absolute-gain step / waiting at its barrier, average | 255 s / 117 s | 103 s / 9 s |
| Correlated-noise sampling per rank, 143 GHz: fastest / slowest | 75 s / 1255 s | 424 s / 875 s |
| Correlated-noise sampling per rank, 353 GHz: fastest / slowest | 62 s / 1251 s | 386 s / 855 s |
| Memory per rank in the mapmaker: lowest / average / highest | 0.11 / 3.52 / 41.2 GB | 1.81 / 3.14 / 40.2 GB |

- Balancing removed most of the waiting. The wait after the mapmaking loop fell from 46% to 16%
  of the step, and the step became 24% faster.
- What is left comes from scans with the same number of samples but different cost: correlated-noise
  sampling still takes twice as long on the slowest rank of a band as on the fastest (424 s against
  875 s at 143 GHz). The number of CG iterations and the amount of masking differ between scans,
  and a split made before the run cannot predict them. This remaining wait, about 170–215 s per
  band in the mapmaker plus about 45 s at the barrier that ends each TOD iteration, is what threads
  remove within a rank (§5.2).
- The same split also balanced the gain steps: the wait at the absolute-gain barrier fell from
  117 s to 9 s, since their cost follows the number of samples closely.
- Memory: no rank is nearly empty any more, so the lowest now uses about 1.5–2 GB. Before balancing,
  nearly empty ranks used 0.29 GB, roughly what a Python process with Commander4's imports costs.
  So each extra process costs little: 192 ranks per node is about 56 GB, or 4% of the node's
  memory. The memory problem is the largest ranks, which still use 38–40 GB at 143 and 353 GHz,
  about 13× the average. Threads help because a rank's threads share its memory (§5.4).
- Reading the TOD (`fileread-tod`) took 330–425 s in the recent runs (425 s in the balanced one),
  once per run.
- Outside the scan loops, the TOD side does little work: far-sidelobe setup about 42 s, map
  gathering about 12 s and file writing about 1 s per iteration (balanced run).

### 2.3 Which code lets threads run in parallel today

The GIL (global interpreter lock) lets only one thread run Python code at a time. Compiled code can
release it while it works, and other threads then run in parallel. Measured on 4 threads with 3.15
(Appendix A), and from reading the code:

| Code | Releases the GIL? | Speedup on 4 threads, normal build |
|---|---|---|
| Huffman decoding (`compression_pymod.cc:24`) | yes | (not measured) |
| ducc0 (FFTs, `Healpix_Base`) | yes | 3.8–4.5× |
| numpy on large arrays | mostly | (not measured) |
| numba kernels (sky projection, sigma0, gap filling, arithmetic) | **no**: none uses `nogil=True` | 0.8–0.9× |
| numba kernel after adding `nogil=True` | yes | 2.7× |
| C++ mapmaker kernels (`mapmaker_pymod.cc`) | **no** | (not measured) |
| Python code (`TODView` logic, loop bookkeeping) | no | — |

On the free-threaded build with the GIL off, the same numba kernel gives 2.2–2.9× without any code
change. (This kernel reads the map at random positions, so memory speed limits it below 4×.)

### 2.4 Free-threaded Python in October 2026

Free-threaded Python ("3.15t") has no GIL. A package must declare that it is safe without the GIL,
and Python turns the GIL back on when it imports one that has not.

- **Turn the GIL back on:** mpi4py, ducc0, healpy, astropy (`astropy.io.fits`), numpy-quaternion,
  ephem. Running with `PYTHON_GIL=0` forces it off, at our own risk.
- **Safe:** numpy, scipy, h5py, pyyaml, numba, matplotlib, pixell, camb, qpoint and our own
  `_cmdr4_backend`.
- **healpy** has no wheels for 3.15. A plain source build installs but fails on import
  (healpy issue #949); it works only after repairing the wheel with `auditwheel`.

## 3. Design

### 3.1 Process layout

- One MPI rank per NUMA domain (8 per owl node, 48 threads each), or smaller ranks of 4–8 cores if
  memory allows. Ranks no larger than a NUMA domain keep each thread's memory close to it.
- Each rank is pinned to its cores (Intel MPI: `I_MPI_PIN_DOMAIN=numa` or an explicit core list).
  Its threads inherit the pinning.
- `resources.tod.num_threads` keeps its name but now means "threads per rank". Library calls get all
  of them outside `map` and 1 inside (§3.5).
- Scans are still split between ranks before the run, by number of samples and sky position, and
  ranks between bands by `band_weight` (§2.2). Within a rank, threads take the next scan as they
  become free.

### 3.2 One way to loop over scans: `TODView.map`

`map` runs a per-scan function for every detector-scan and returns the results in scan order:

```python
# tod/view.py (sketch)
_worker = threading.local()      # _worker.active is True inside a pool thread
_pool = None                     # one thread pool per rank, made on first use


def _start_worker():
    _worker.active = True
    numba.set_num_threads(1)     # numba's thread count is set per calling thread


class TODView:
    def map(self, func, *args, step: str, accepted_only: bool = False) -> list:
        """Call `func(view, *args)` for every detector-scan and return the results in scan order.

        Each call gets its own freshly focused view. `step` names the Gibbs step; it keys the
        per-scan random numbers (§3.4) and the benchmark entry. With one thread this is a plain loop
        in the calling thread, so print, pdb and tracebacks behave as usual.
        """
        accept = self.tod_samples.accept if accepted_only else None
        tasks = list(self.experiment_data.iter_detector_scans(accept))

        def run_one(task):
            iscan, det = task
            view = TODView(self.experiment_data, self.tod_samples,
                           compsep_output=self.compsep_output,
                           downsample_factor=self._downsample_factor,
                           proc_mask_type=self._proc_mask_type,
                           mask_threshold=self._mask_threshold, step=step).focus(iscan, det)
            t0 = time.perf_counter()
            result = func(view, *args)
            return result, time.perf_counter() - t0

        with benchmark(step):
            if NUM_THREADS == 1:
                outputs = [run_one(task) for task in tasks]
            else:
                global _pool
                if _pool is None:
                    _pool = ThreadPoolExecutor(NUM_THREADS, initializer=_start_worker)
                outputs = list(_pool.map(run_one, tasks))
        # Added here, in the main thread: two detectors of the same scan may run at the same time.
        for (iscan, _), (_, runtime) in zip(tasks, outputs):
            self.tod_samples.scan_runtime[iscan] += runtime
        return [result for result, _ in outputs]
```

- **One pool per rank**, created once. Starting threads for every loop would cost time for nothing.
- **A fresh view per scan.** `iter_focused` re-focuses one shared view object
  (`view.py:125-147`), which threads cannot share. A `TODView` only stores references, so creating
  one costs nothing.
- **Results in input order.** Sums done in the main thread over the returned list are then the same,
  to the last bit, as in the serial loop, for any number of threads. This is what makes the
  equality test in §4 strict: any difference between 1 and N threads is a bug, not rounding. It is
  the main reason to prefer `map` over the alternatives in §3.12.
- **Only the per-scan functions run in parallel.** `map` returns when all of them have finished, and
  the loop over its results then runs in the main thread, one result at a time, in scan order.
  That loop is serial, so it must stay cheap: adding numbers, storing values. (A variant of `map`
  can hand out results in scan order as they become ready, so the combining loop runs while the
  threads still work. This avoids holding all results at once when they are large.)
- **Load balancing.** Tasks can be handed out longest first, using the scans' numbers of samples or
  the run times in `scan_runtime`, while results are still returned in scan order. Long scans then
  do not end up last.
- **Errors.** An exception in a worker is raised in the main thread, with its traceback, when the
  results are collected. Other workers finish their current scan first.
- `scan_runtime` is now measured inside the worker, instead of as the time between loop passes
  (`tod2map.py:94-100`). It is still written to the chain files, though the split between ranks no
  longer uses it.

### 3.3 What a converted step looks like

The absolute-gain loop (`gain.py:123-136`) today:

```python
for view in scan_view.iter_focused(accepted_only=True):
    calib = view.get_calib_tod("abs", gain_cfg.calibrate_against,
                               gap_fill_method=gain_cfg.gap_fill_method)
    ...
    N_inv_s = experiment_data.apply_N_inv(s_cal, view.noise_params, samprate=gain_samprate)
    sum_s_T_N_inv_d += np.dot(N_inv_s, residual_tod)
    sum_s_T_N_inv_s += np.dot(s_cal, N_inv_s)
```

After conversion (the `...` stands for the same unchanged lines):

```python
def _abs_gain_scan_terms(view, gain_cfg):
    """s^T N^-1 d and s^T N^-1 s for one detector-scan."""
    calib = view.get_calib_tod("abs", gain_cfg.calibrate_against,
                               gap_fill_method=gain_cfg.gap_fill_method, rng=view.rng)
    ...
    N_inv_s = view.experiment_data.apply_N_inv(s_cal, view.noise_params, samprate=gain_samprate)
    return np.dot(N_inv_s, residual_tod), np.dot(s_cal, N_inv_s)

for d_term, s_term in scan_view.map(_abs_gain_scan_terms, gain_cfg, step="abs-gain",
                                    accepted_only=True):
    sum_s_T_N_inv_d += d_term
    sum_s_T_N_inv_s += s_term
```

The function body is the old loop body, with four changes:

1. The two `+=` lines become one `return` of the two terms; the sums move to the loop below.
2. `experiment_data` is reached through the view (`view.experiment_data`), since the function no
   longer sees the variables of `sample_absolute_gain`.
3. `get_calib_tod` gets `rng=view.rng`, so the gap filling draws from the scan's own random stream
   (§3.4) instead of the global `np.random`.
4. `step="abs-gain"` names the step for the random streams and the benchmark entry.

The per-scan function sits right above the loop, in the same file. The loop that combines the
results reads like the old loop. Every variable created inside the per-scan function belongs to its
thread; in OpenMP each such variable must be declared `private`, and forgetting one is a classic
silent bug.

### 3.4 Per-scan random numbers

Every detector-scan gets its own random generator, made from a fixed list of numbers:

```python
@property
def rng(self) -> np.random.Generator:
    """Random numbers for this detector-scan and Gibbs step, whichever rank or thread runs it."""
    if self._rng is None:
        step_id = zlib.crc32(self._step.encode())
        self._rng = np.random.default_rng([self.tod_samples.seed, self.tod_samples.chain,
                                           self.tod_samples.iteration, step_id,
                                           self.tod_samples.scan_ids[self.iscan], self.idet])
    return self._rng
```

`TODSamples` has `chain` and `scan_ids` today; `seed` (a run seed from the parameter file) and
`iteration` would be new, as are the view's `_step` and `_rng`.

- **The step name must be part of the seed.** Without it, two Gibbs steps in the same iteration
  would draw the same numbers for the same scan.
- `tod/` and `data_models/` use the global `np.random` in 16 places. The 13 in per-scan code must
  take a generator instead: `sample_ncorr.py:95-97, 176-178, 247, 363`, `psd.py:59`, `view.py:610`,
  `hfi_demodulation.py:46` and the two numba draws below. The three draws done once per rank in the
  main thread (`gain.py:154`, `393`, `396`) can stay.
- **numba draws.** Inside numba functions, `np.random` uses numba's own random state, one per thread
  (`gap_filling.py:78, 88`). Which thread runs a scan would then decide its random numbers. Fix:
  pass in a seed drawn from `view.rng` and call `np.random.seed(seed)` at the start of the numba
  function. This seeds only the calling thread's state (tested, Appendix A).

### 3.5 Library threads inside workers

ducc0 and numba take their thread count from `OMP_NUM_THREADS` today. Inside a worker they must use
1 thread, or a rank with 8 workers runs 64 threads.

- A single helper replaces the 15 places that read `OMP_NUM_THREADS` directly: 10 in code that
  per-scan TOD work calls (`math_utils/fft.py`, `data_models/pointing.py`,
  `tod/sky_projection.py`, `tod/sidelobe_deconvolve.py`) and 5 in `math_utils/sht.py`:

  ```python
  def inner_nthreads() -> int:
      """Threads a library call may use: 1 inside a scan worker, all of the rank's outside."""
      return 1 if getattr(_worker, "active", False) else NUM_THREADS
  ```
- `numba.set_num_threads(1)` in each worker at start-up (§3.2). The setting applies only to the
  calling thread (tested, Appendix A).
- BLAS threads (OpenBLAS inside numpy) are shared by the whole process and cannot be set per thread.
  Set `OPENBLAS_NUM_THREADS=1` for TOD ranks; the TOD code makes little use of BLAS.
- Large single operations outside `map`, such as the far-sidelobe convolution, keep using all of the
  rank's threads through ducc0.

> **Prototype:** workers of 2 cores were clearly faster than workers of 1 (108 against 124 s per
> pass). With one scan per core, twice as many FFT arrays compete for the L3 cache and the memory
> bandwidth. The prototype's setting `scan_threads` gives each worker `num_threads // scan_threads`
> cores.

### 3.6 Benchmark timer

`PerfLogger` keeps one global stack of open timers (`diagnostics/performance.py:72`). Code inside
per-scan functions opens timers (`huffman`, `pointing`, `FFT`, ...), so threads would mix up each
other's entries.

- Give each thread its own stack (`threading.local`). When a worker starts a task, its stack starts
  with the caller's current path, so entries still appear under `abs-gain`, `mapmaker` and so on.
- Protect the totals with a lock.
- Inside `map`, the reported times are summed over threads. Mark them as such in the report
  (e.g. "thread-s"), or they will look larger than the wall time of their parent.

### 3.7 Map accumulation

The binned mapmaker (`binned.accumulate`, `tod2map.py:220`) and the CG mapmaker
(`cg.accum_to_RHS`, `tod2map.py:227`; the operator loop at `cg.py:294`) add every scan into one map
per rank. Three options:

| Option | Memory | Speed | Same result every run? |
|---|---|---|---|
| **A. Shared map, thread-safe additions in C++.** The kernels release the GIL and use atomic additions, i.e. additions that cannot lose updates when two threads hit the same pixel. | one map | atomic additions are slower than plain ones; measure (`map-binning` was 26.5 s per iteration on average) | no: the order of additions varies, so values differ in the last bits |
| **B. One map per thread**, summed after the loop. | one map per thread | plain additions | only if each thread gets a fixed list of scans, which gives up load balancing |
| **C. Main thread adds results in scan order**, as workers return them. | in-flight results | serial: with 4× more scans per rank, about 4 × 26.5 ≈ 106 s per iteration | yes |

Recommendation: A for production, with C as an optional `deterministic` mode for debugging.

> **Prototype:** C made the main thread the bottleneck (20 ms per scan, 57 s per iteration for
> 2725 scans per rank). What works is a fourth option: each worker bins its scan into small maps
> over the scan's own pixels, and the main thread adds these to the rank's maps in scan order
> (5 ms per scan). Identical for any number of threads, and with no per-thread maps. See
> `thread_prototype_results.md`, §5.4.

- C++20's `std::atomic_ref<double>::fetch_add` is the simplest way to write A. The backend is
  C++17 today (`CMakeLists.txt:10`), so either move to C++20 (both icpx and gcc 11 support it) or
  write the atomic addition as a compare-and-swap loop.
- The kernels must release the GIL (`nb::gil_scoped_release`), as the Huffman decoder already does.

### 3.8 MPI

- Only the main thread calls MPI. The per-scan functions never do.
- mpi4py starts MPI in its "multiple threads may call MPI" mode by default. Asking for the
  main-thread-only mode (`mpi4py.rc.thread_level = "funneled"`, before importing `mpi4py.MPI`)
  states our rule and can make MPI calls cheaper.
- With fewer ranks, the reductions and gathers at the end of each step involve fewer participants
  and get faster.

### 3.9 Reading the TOD

`BaseReader.read` (`file_io/experiments/base_reader.py:133`) loops over scans and opens one HDF5
file per scan. Each scan can become a task. However, h5py runs all HDF5 calls of a process one at a
time, behind its own lock, and it keeps doing so on free-threaded Python. Within a rank:

- reading bytes from the files stays serial;
- decoding the flags at construction (`detector_tod.py:150-152`, Huffman decoding plus `packbits`)
  and building the Python objects run in parallel.

The 330–425 s per run are split between these in an unknown way. **Measure that first.** If HDF5
calls dominate, the options are: read all detectors of a scan in fewer, larger calls; or accept a
slower start if the Gibbs loop gains more than reading loses.

### 3.10 The loops to convert

| Loop | Location | Runs | Notes |
|---|---|---|---|
| HFI baselines | `hfi_demodulation.py:49` | every iteration | return the `stats` terms instead of adding them up in the loop |
| HFI modulation phase | `hfi_demodulation.py:112` | first iteration | return the `rejected` count |
| Absolute gain | `gain.py:123` | every iteration | §3.3 |
| Relative gain | `gain.py:209` | every iteration | return per-scan sums; add them into the per-detector arrays in the main thread |
| Temporal gain | `gain.py:323` | every iteration | as relative gain; the per-detector solves (`gain.py:351`) could become a second `map`, over detectors |
| Mapmaking | `tod2map.py:96` | every iteration | §3.11 |
| Glitches (3 passes) | `glitches/sampling.py:71, 77, 84` | every iteration | already write only per-scan slots |
| Jumps | `jumps/sampling.py:52` | every iteration | return counts and offsets instead of appending to lists |
| CG operator | `cg.py:294` | every CG step | map with the shared accumulator; create `out_local` before the map |
| Pixel domain | `pixel_domain.py:130` | once | per-scan pixel lists, combined in the main thread |
| Reading | `base_reader.py:133` | once | §3.9 |
| Bookkeeping | `tod_samples.py`, `gain.py:453`, `sidelobe_deconvolve.py:52` | — | cheap; leave serial |

### 3.11 The mapmaking loop

The loop body in `tod2map.py:96-227` is the largest. It mixes per-scan science with shared state:

- `tod2map.py:146-153`: lists and counters for the correlated-noise summary (`sampled_params`,
  `residuals`, `niters`, `num_failed_convergences_ncorr`, ...). These become fields of the per-scan
  result and are combined after `map`.
- `tod2map.py:220-227`: map accumulation, through the thread-safe accumulator (§3.7).
- `tod2map.py:94-100`: scan timing, now inside `map` (§3.2).
- The writes to `tod_samples` (`accept`, `noise_params`, `ncorr_*`, `good_fraction`, `chisq_z`) go
  to `[view.iscan, view.idet]` slots and can stay as they are.
- The data-selection checks only stop processing of their own detector-scan. They depend on nothing
  from other scans, so running the scans in any order is fine.

### 3.12 Alternatives considered: keeping the explicit loops

A plain `for` loop cannot be kept as it is. Python always runs a loop body in the thread that runs
the loop, and it has nothing like `#pragma omp`. To give work to another thread, the work must be a
function. The only choice is where that function boundary goes.

**Around one scan:** `map`, as proposed above.

**Around the whole loop.** Each thread runs today's loop, unchanged, on its own share of the scans,
with its own sums. A short step afterwards combines the per-thread sums. This is the closest match
to `!$omp parallel do reduction(+:...)`:

```python
def _abs_gain_sums(views, gain_cfg):
    """Sum s^T N^-1 d and s^T N^-1 s over the detector-scans given to this thread."""
    sum_s_T_N_inv_d = 0.0
    sum_s_T_N_inv_s = 0.0
    for view in views:
        calib = view.get_calib_tod("abs", gain_cfg.calibrate_against,
                                   gap_fill_method=gain_cfg.gap_fill_method, rng=view.rng)
        ...
        N_inv_s = view.experiment_data.apply_N_inv(s_cal, view.noise_params, samprate=gain_samprate)
        sum_s_T_N_inv_d += np.dot(N_inv_s, residual_tod)
        sum_s_T_N_inv_s += np.dot(s_cal, N_inv_s)
    return sum_s_T_N_inv_d, sum_s_T_N_inv_s

per_thread = scan_view.run_on_threads(_abs_gain_sums, gain_cfg)   # one result per thread
sum_s_T_N_inv_d = sum(p[0] for p in per_thread)
sum_s_T_N_inv_s = sum(p[1] for p in per_thread)
```

The loop and its `+=` lines stay where they are, but:
- The combining step is still needed. OpenMP's `reduction` clause merges each thread's private
  copies automatically; Python cannot.
- **Results change with the number of threads**, because the per-thread sums are added in a
  different order. When idle threads take the next free scan (needed for load balancing), results
  can also differ between runs. The equality test (§4) would then need tolerances for every
  quantity, so a race that changes results slightly would pass unnoticed.
- Lists built inside the loop come out in thread order, not scan order.
- The same rule against races applies: the loop may only change variables created inside the
  function, and its own `[view.iscan, view.idet]` slots.

**Preparing data ahead, loops untouched.** Worker threads prepare the next few scans
(decompression, pointing, sky projection) while the main thread runs the unchanged loop body. Only
the preparation gets faster; the expensive science in the body, such as the correlated-noise CG,
stays serial. The speedup is therefore limited.

**Rejected: rewriting loops into functions automatically.** Tools that do this hide what really
runs, which makes errors hard to understand for anyone who is not a programmer. (The library
`pymp` gets OpenMP-like syntax by copying the whole process at a `with` line; threads cannot do
that.)

`map` is preferred because its exact agreement with the serial loop keeps the equality test strict.
Running the whole loop per thread remains a reasonable choice for loops with many counters, such as
the one in `tod2map`, if keeping the loop matters more there.

## 4. Where it can go wrong

All of these are races: two threads touch the same data and the result depends on timing. They
usually give **wrong numbers, not crashes**.

1. **Counters, sums and lists in loop bodies.** Examples: `gain.py:135-136`,
   `hfi_demodulation.py:92-96`, `jumps/sampling.py:65-69`, `tod2map.py:146-153`. `x += y` reads,
   adds and writes back, and two threads can lose one of the updates. Fix: return the values and
   combine them in the main thread.
2. **Slots shared by several detector-scans.** Arrays indexed by scan only or by detector only:
   `scan_runtime[iscan]` (`tod2map.py:99`), `local_s_T_N_inv_s[view.idet]` (`gain.py:223`). Two
   detectors of the same scan, or the same detector in two scans, run at the same time. Fix: as 1.
   **Rule: a per-scan function writes only to `[view.iscan, view.idet]` slots.**
3. **Buffers created on first use on a shared object.** `CGMapmaker` creates `_rhs_loca_map` on the
   first call to `accum_to_RHS` (`cg.py:236`, `266`; similar at `cg.py:103`, `111`). If two threads
   arrive together, both create a buffer and one thread's additions are lost. Fix: create such
   buffers before the `map`.
4. **Code that is safe today only because it holds the GIL.** The C++ map accumulators
   (`mapmaker_pymod.cc`) and any numba kernel that writes to a shared array cannot run at the same
   time on the normal build. With the GIL off they can, and additions into the same pixel are lost.
   **This kind of bug appears without any code change, just by switching the interpreter.** Fix:
   §3.7, and a review of every kernel that writes to an array it did not create.
5. **Random numbers.** (a) The global `np.random` is protected against corruption, but the order of
   draws depends on timing, so runs cannot be repeated. (b) `np.random.seed(...)` in a worker
   (`sample_ncorr.py:95`, `176`) reseeds the generator of all threads. (c) numba draws come from
   the running thread's own state (§3.4). Fix: `view.rng` everywhere.
6. **The reused view.** A view from `iter_focused` must never be passed to another thread; it is
   re-focused on the next scan. `map` creates fresh views.
7. **The benchmark timer** (§3.6).
8. **Module-level caches.** `uK_CMB_to_uK_RJ_dict` (`sky_projection.py:28`) is filled on first use
   through pysm3/astropy units. Two threads might then call astropy at the same time, and astropy
   has not declared that this is safe. Fix: fill it before the `map`. (`math_utils/sht.py:20` is on
   the compsep side.)
9. **Packages running with the GIL forced off.** All six packages from §2.4 are loaded when
   Commander4 starts; numpy-quaternion and ephem come in through `pixell.coordsys`
   (`data_models/pointing.py:12`). Which of them per-scan code calls:
   - **ducc0** (FFTs, `Healpix_Base`), for all experiments. ducc0 is built for use from many
     threads, so the risk is low.
   - **pixell.coordsys and numpy-quaternion**, for experiments with boresight pointing (Simons
     Observatory): `ScanBoresightPointing.get_det_point` (`pointing.py:97-116`) multiplies
     quaternions for every detector-scan. This has not been checked for use from several threads.
     Test it under the equality test (below) before trusting it, or compute the pointing before
     the `map`.
   - healpy is used only in `detector_map.py`, per band and in the main thread; mpi4py only in the
     main thread; astropy only through point 8.
10. **Too many threads.** Not an error, but slow (§3.5).

Found by the prototype (`thread_prototype_results.md`, §5):

11. **numba compiles unsafely from several threads.** Workers making the first call of a kernel at
    the same time can crash numba 0.68. Fix: a lock around numba's compilation.
12. **ducc0's far-sidelobe kernel depends on the thread count.** A plan made with 48 threads uses
    another oversampling than one made with 1, and today's code is affected too. Fix: pin the
    oversampling.
13. **OpenBLAS results depend on its thread count**, through the order of the additions in long
    dot products. Fix: one BLAS thread for TOD ranks.

**Catching these:**
- **Equality test.** Run each Gibbs step on a small data set with 1 and with 4 threads, on both
  Python builds. Everything except the maps must be identical to the last bit; the maps must agree
  to about 1e-12 (option A) or exactly (option C). This catches most of points 1–7. It can demand
  exact agreement only because `map` returns results in scan order (§3.2); with one loop per thread
  (§3.12) it would need tolerances everywhere.
- A check in the tests that per-scan code in `tod/` does not use `np.random.` directly.
- Run the full test suite on 3.15t with `PYTHON_GIL=0` and more than one thread.

## 5. Consequences

### 5.1 For the code and the people writing it

Rules for contributors:

1. Write what happens to one detector-scan as a function `f(view, ...)`, placed right above the loop
   that uses it.
2. Inside it, write only to `[view.iscan, view.idet]` slots of `tod_samples`. Return everything
   else.
3. Draw random numbers only from `view.rng`.
4. Do not call MPI inside it.
5. Combine the results after `map`, in the main thread.

- With `num_threads: 1` everything runs as a plain loop in the main thread, so debugging works as
  today.
- Long loop bodies become named functions with clear inputs and outputs. For `tod2map`'s ~130-line
  loop this also makes the per-scan science testable on its own.
- The cost is one function per loop, plus the rules above. The rules are the same ones OpenMP
  hides behind `#pragma omp parallel for`.

### 5.2 For runtime

**Scan loops that are not converted get slower.** Example: going from 2 to 8 cores per rank with the
same total number of cores means 4× fewer ranks and 4× more scans per rank.

| Code | Time per iteration vs today |
|---|---|
| Converted scan loop: 4× the scans on 8 threads (today's second core is mostly idle) | about 0.5× |
| Scan loop that was not converted | **4×** |
| Work that does not grow with scans per rank (reductions, master-only work, file writes) | unchanged |
| Large single operations using library threads (far-sidelobe convolution) | faster |

| Share of TOD time in converted loops | Speedup vs today |
|---|---|
| 90% | 1.2× |
| 95% | 1.5× |
| 99% | 1.9× |

These numbers assume the converted loops scale perfectly to 8 threads, and they leave out the gain
from better load balancing. In the HFI runs, nearly all non-waiting time is inside scan loops
(§2.2), so 95% or more is reachable, but only if nearly every scan loop is converted. A loop someone
forgot shows up in the benchmark report: its time grows about 4× when the threads per rank go from 2
to 8.

Other runtime effects to measure:

- **Load balance.** Within a rank, threads take the next scan as they become free. In the balanced
  run, ranks still waited about 16% of the mapmaker step, plus about 45 s at the end of each TOD
  iteration (§2.2). With 8–48 threads per rank and longest-first ordering, little of that should
  remain inside a rank. Larger ranks also hold more scans, so the cost differences between scans
  average out better between ranks.
- **Free-threaded Python is somewhat slower per thread**, because of the extra safety work inside
  the interpreter. Compare 3.15 and 3.15t with one thread.
- **Garbage collection pauses all threads** on the free-threaded build. Commander4 holds one Python
  object per detector-scan plus their attributes, and the collector checks them all.
  `gc.freeze()` after reading takes them out of its view. Measure the pause times.
- **Many threads touching the same Python objects** (`experiment_data`, configs) slow each other
  down a little, because each use changes the object's reference count. Probably small; measure.
- **Startup.** Fewer processes import the packages and compile the numba functions.
- **Reading** may get slower (§3.9).

### 5.3 For reproducibility

- With per-scan random streams, the chains no longer depend on how many ranks or threads are used,
  or on which rank holds which scan. Today each rank draws from its own global generator, so any
  change in how scans are split (another number of ranks, other band weights, other file lists)
  changes the draws. This is a gain even without threads.
- With map option A, maps differ between runs in the last bits.

### 5.4 For memory

- **Per-thread scratch.** An HFI 143 GHz scan has about 800,000 samples (`HFI_143_000091.h5`, scan
  `000000`: 803,721); one float64 copy is 6.4 MB. Correlated-noise sampling works on mirrored arrays
  of twice that length and holds a few tens of such arrays at its peak. **Estimate: 100–300 MB per
  thread; measure with `log_memory`.** Today one scan per rank is in flight, i.e. one per 2 cores.
  With threads it is one per core: about 40 GB more per node at 384 threads, for 200 MB scans.
- **Pooling.** A rank's threads share its memory, so a heavy scan on one thread uses room that the
  others are not using. Only the rank's total must fit, not every thread's worst case.
- **Fewer copies.** Each removed rank saves the cost of a Python process (about 0.29 GB, §2.2), plus
  its own copy of the band's read-only data (sky model, processing masks) and its local map buffers.
- Before choosing the layout, find out what fills the ranks that still use 38–40 GB after balancing
  (§8).

## 6. Blockers (October 2026)

| Blocker | Matters for | What unblocks it |
|---|---|---|
| mpi4py, ducc0, healpy, astropy, numpy-quaternion, ephem turn the GIL back on | free-threaded build | new releases declaring support; until then `PYTHON_GIL=0`. ducc0 and numpy-quaternion matter most, since per-scan code calls them (§4, point 9) |
| healpy has no 3.15 wheels; source builds are broken (healpy #949) | 3.15 builds | official wheels; until then `auditwheel repair` by hand |
| numba kernels hold the GIL | normal build only | add `nogil=True` (one word per kernel) |
| C++ mapmaker kernels hold the GIL and are not thread-safe | both builds | §3.7 |
| h5py runs HDF5 calls one at a time | reading, both builds | §3.9; no fix expected from h5py |
| Commander4's own shared state (timer, random numbers, buffers, caches) | both builds | §3.4, §3.6, §4 |
| Unknown scaling of free-threaded Python at 8–48 threads in our code | free-threaded build | measure (§7, phase 2) |

## 7. Suggested order of work

**Phase 0: now, no change in behaviour, useful anyway.**
- Per-scan random streams (`view.rng`). This already makes the chains independent of the scan
  distribution.
- The `inner_nthreads()` helper.
- The thread-safe `PerfLogger`.
- `TODView.map` running with 1 thread.
- Convert loops to `map` when they are next touched for other reasons.
- Create lazily created buffers up front (§4, point 3).

**Phase 1: normal Python build, partial speedup.**
- `nogil=True` on the numba kernels.
- Release the GIL in the C++ kernels that do not touch Python objects.
- Thread-safe map accumulation (§3.7).
- Allow `num_threads > 1`. Add the equality test. Measure HFI at 2, 4 and 8 cores per rank.

**Phase 2: free-threaded build (~2027).**
- Build a 3.15t or 3.16t environment. Use `PYTHON_GIL=0` until the dependencies declare support.
- Measure scaling at 4, 8, 16 and 48 threads per rank, the garbage-collection pauses and the
  per-thread memory.
- Redesign reading if §3.9 shows it is needed.

**Phase 3: production layout.**
- Choose ranks per node from the phase 2 numbers, and update the job scripts (pinning,
  `OPENBLAS_NUM_THREADS=1`, thread level).

## 8. Open questions

- What fills the ranks that still use 38–40 GB after balancing?
- How much of the 330–425 s of reading is spent in HDF5 calls?
- Map accumulation: is option A fast enough, and is a `deterministic` mode wanted?
- How much scratch memory does one thread really use on long HFI and LiteBIRD scans?
- Should `map` live on `TODView` (proposed, since it builds views) or on `DetectorGroupTOD`?

## Appendix A. Small tests behind §2.3, §3.4 and §3.5

Run on owl39 on 2026-10-10, with Python 3.15.0 (normal build) and 3.15.0t (`PYTHON_GIL=0`). The
numbers varied by about ±20% between repeats on the shared node.

**GIL release.** 16 calls, run one after the other and then on 4 threads, each library call using
one thread itself:

```python
@numba.njit(fastmath=True)          # and a copy with nogil=True
def sky_kernel(skymap, pix, n_repeat):
    out = np.empty(pix.size, dtype=np.float32)
    for _ in range(n_repeat):
        for i in range(pix.size):
            out[i] = skymap[pix[i]]
    return out

tod = rng.standard_normal(2 * 803721)                          # one mirrored HFI scan
skymap = rng.standard_normal(12 * 1024**2).astype(np.float32)
pix = rng.integers(0, skymap.size, 803721)
tasks = {"ducc0 FFT": lambda _: ducc0.fft.r2c(tod, nthreads=1),
         "numba kernel": lambda _: sky_kernel(skymap, pix, 5)}
```

| | normal build | free-threaded, GIL off |
|---|---|---|
| ducc0 FFT | 3.8–4.5× | 3.4–3.8× |
| numba kernel without `nogil` | 0.8–0.9× | 2.2–2.9× |
| numba kernel with `nogil=True` | 2.7× | 2.9× |

**numba thread count.** After `numba.set_num_threads(1)` in a second thread,
`numba.get_num_threads()` in the main thread still returned 384.

**numba random state.** Two threads each calling a numba function that runs `np.random.seed(42)`
and then `np.random.randn()` got the same number, so seeding inside a numba function affects only
the calling thread.
