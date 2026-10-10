"""Write a copy of a band's filelist with each scan's measured run time as its weight.

Every band chain file holds the wall time the mapmaking loop spent on each scan (`scan_runtime`)
and each scan's sky position (`scan_sky_position`). This tool averages the run times over the chain
files it is given and writes them to the filelist's weight column, in seconds per iteration. It
writes the sky positions to the theta and phi columns. A later run can then split the scans over its
ranks so that every rank gets about the same total run time.

    c4-scan-weights filelist_143.txt chains_bands/PlanckHFI_Planck143GHz_chain*_iter*.h5 \\
        -o filelist_143_weighted.txt

The rows keep their order and all other columns. Rows that no chain file covers
(bad scans, rows outside the run's filelist slice, scans the reader dropped) get the median weight
and keep their theta and phi. Chains written before `scan_sky_position` existed leave theta and phi
as they were.
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
    args = parser.parse_args(argv)

    with open(args.filelist) as infile:
        infile.readline()  # The first line holds the number of scans.
        rows = [line.split() for line in infile if line.strip()]
    row_of_scan = {int(columns[0]): irow for irow, columns in enumerate(rows)}

    # Sum each row's run time over the chain files, and count the files that timed it.
    runtime_sum = np.zeros(len(rows))
    runtime_count = np.zeros(len(rows), dtype=np.int64)
    sky_position = np.full((len(rows), 2), np.nan)
    for path in args.chain_files:
        with h5py.File(path, "r") as chain:
            scan_ids = chain["scan_ids"][()]
            runtime = chain["scan_runtime"][()]
            position = chain["scan_sky_position"][()] if "scan_sky_position" in chain else None
        unknown = [scan_id for scan_id in scan_ids if scan_id not in row_of_scan]
        if unknown:
            raise ValueError(f"{path} holds {len(unknown)} scans that are not in {args.filelist}, "
                             f"e.g. scan {unknown[0]}. Is it a chain file of another band?")
        irows = np.array([row_of_scan[scan_id] for scan_id in scan_ids], dtype=np.int64)
        runtime_sum[irows] += runtime  # A chain file holds each scan once.
        runtime_count[irows] += 1
        if position is not None:
            sky_position[irows] = position  # Static, so every chain file holds the same value.

    timed = runtime_count > 0
    median_weight = np.median(runtime_sum[timed]/runtime_count[timed])
    weight = np.full(len(rows), median_weight)
    weight[timed] = runtime_sum[timed]/runtime_count[timed]
    with open(args.output, "w") as outfile:
        outfile.write(f"{len(rows)}\n")
        for irow, columns in enumerate(rows):
            theta_phi = columns[3:5]
            if np.isfinite(sky_position[irow]).all():
                theta_phi = [f"{angle:.6f}" for angle in sky_position[irow]]
            outfile.write(" ".join([columns[0], columns[1], f"{weight[irow]:.4g}", *theta_phi,
                                    *columns[5:]]) + "\n")
    print(f"Wrote {args.output}. {len(args.chain_files)} chain files timed {timed.sum()} of "
          f"{len(rows)} scans, {weight[timed].sum():.0f} s per iteration in total; the other scans "
          f"got the median weight, {median_weight:.4g} s. "
          f"{np.isfinite(sky_position[:, 0]).sum()} scans got a sky position from the chains.")
