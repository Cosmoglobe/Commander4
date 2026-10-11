"""Write a copy of a band's filelist with what a run measured about each scan.

Every band chain file holds the wall time the mapmaking loop spent on each scan (`scan_runtime`),
each scan's sky position (`scan_sky_position`) and number of samples (`scan_ntod`), and the IDs of
the scans the reader kept nothing from (`dropped_scan_ids`: no data in the file, or every detector
flagged or cut). This tool writes them into the filelist's columns:

    scan_id "path" weight theta phi ntod read

- weight: the mean run time over the chain files given, in seconds per iteration.
- theta, phi: the sky position, in radians.
- ntod: the number of samples, or -1 if not known. The parameter `scan_weights` decides whether
  Commander4 splits the scans between ranks by weight or by ntod.
- read: 1 if the scan is to be read, 0 if a run found no usable data in it. Commander4 leaves out
  the scans marked 0 before it splits the scans over its ranks. The mark reflects the read-time cuts
  of the runs the chain files come from (bad-data bitmask, `min_unmasked_fraction`, ...); after
  loosening them, run the tool with `--clear-read-marks`.

    c4-scan-weights filelist_143.txt chains_bands/PlanckHFI_Planck143GHz_chain*_iter*.h5 \\
        -o filelist_143_weighted.txt

The rows keep their order and any further columns. A scan that a chain file kept is marked 1, and
failing that, one that a chain file dropped is marked 0 and gets weight 0. The other rows (bad
scans, scans outside the run's scan-ID range) get the median weight and keep their theta, phi, ntod
and read columns, so the tool can be run again on its own output. Chains written before a dataset
existed leave its columns as they were. Scans in the chain files that the filelist does not list,
e.g. from a run over a wider scan-ID range, are ignored.
"""
import argparse

import h5py
import numpy as np


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filelist", help="The band's filelist, as the run read it.")
    parser.add_argument("chain_files", nargs="+",
                        help="Band chain files of that band, from any iterations, chains and runs.")
    parser.add_argument("-o", "--output", required=True, help="The filelist to write.")
    parser.add_argument("--clear-read-marks", action="store_true",
                        help="Mark every scan to be read, ignoring the scans the chains' runs "
                             "dropped and any 0 already in the filelist, e.g. after the read-time "
                             "cuts have been loosened.")
    args = parser.parse_args(argv)

    with open(args.filelist) as infile:
        infile.readline()  # The first line holds the number of scans.
        rows = [line.split() for line in infile if line.strip()]
    row_of_scan = {int(columns[0]): irow for irow, columns in enumerate(rows)}

    # Sum each row's run time over the chain files, and count the files that timed it.
    runtime_sum = np.zeros(len(rows))
    runtime_count = np.zeros(len(rows), dtype=np.int64)
    sky_position = np.full((len(rows), 2), np.nan)
    scan_ntod = np.full(len(rows), -1, dtype=np.int64)
    dropped = np.zeros(len(rows), dtype=bool)
    ignored_scan_ids = set()
    for path in args.chain_files:
        with h5py.File(path, "r") as chain:
            scan_ids = chain["scan_ids"][()]
            runtime = chain["scan_runtime"][()]
            position = chain["scan_sky_position"][()] if "scan_sky_position" in chain else None
            ntod = chain["scan_ntod"][()] if "scan_ntod" in chain else None
            dropped_ids = chain["dropped_scan_ids"][()] if "dropped_scan_ids" in chain else []
        # Scans the filelist does not list, e.g. from a run over a wider scan-ID range, are ignored.
        listed = np.array([scan_id in row_of_scan for scan_id in scan_ids], dtype=bool)
        if not listed.any():
            raise ValueError(f"{path} holds none of the scans in {args.filelist}. Is it a chain "
                             "file of another band?")
        ignored_scan_ids.update(scan_ids[~listed].tolist())
        irows = np.array([row_of_scan[scan_id] for scan_id in scan_ids[listed]], dtype=np.int64)
        runtime_sum[irows] += runtime[listed]  # A chain file holds each scan once.
        runtime_count[irows] += 1
        # Static, so every chain file holds the same values.
        if position is not None:
            sky_position[irows] = position[listed]
        if ntod is not None:
            scan_ntod[irows] = ntod[listed]
        dropped[np.array([row_of_scan[scan_id] for scan_id in dropped_ids
                          if scan_id in row_of_scan], dtype=np.int64)] = True

    timed = runtime_count > 0
    read = np.array([len(columns) < 7 or columns[6] != "0" for columns in rows])
    read[dropped] = False
    read[timed] = True
    if args.clear_read_marks:
        read[:] = True
    median_weight = np.median(runtime_sum[timed]/runtime_count[timed])
    weight = np.full(len(rows), median_weight)
    weight[timed] = runtime_sum[timed]/runtime_count[timed]
    weight[~read] = 0.0
    with open(args.output, "w") as outfile:
        outfile.write(f"{len(rows)}\n")
        for irow, columns in enumerate(rows):
            theta_phi = columns[3:5] if len(columns) > 4 else ["0.0", "0.0"]
            if np.isfinite(sky_position[irow]).all():
                theta_phi = [f"{angle:.6f}" for angle in sky_position[irow]]
            ntod = columns[5] if len(columns) > 5 else "-1"
            if scan_ntod[irow] >= 0:
                ntod = str(scan_ntod[irow])
            outfile.write(" ".join([columns[0], columns[1], f"{weight[irow]:.4g}", *theta_phi,
                                    ntod, "1" if read[irow] else "0", *columns[7:]]) + "\n")
    print(f"Wrote {args.output}. {len(args.chain_files)} chain files timed {timed.sum()} of "
          f"{len(rows)} scans, {weight[timed].sum():.0f} s per iteration in total. "
          f"{(~read).sum()} scans are marked not to be read (weight 0), and the remaining "
          f"{(read & ~timed).sum()} got the median weight, {median_weight:.4g} s. From the chains, "
          f"{np.isfinite(sky_position[:, 0]).sum()} scans got a sky position and "
          f"{(scan_ntod >= 0).sum()} got their ntod. {len(ignored_scan_ids)} scans in the chains "
          "are not in the filelist and were ignored.")
