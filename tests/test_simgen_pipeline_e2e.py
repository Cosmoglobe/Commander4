"""End-to-end simgen run exercising the bolometer transfer function through the full pipeline.

This drives the real ``simgen.pipeline.run`` (single MPI rank): a CMB (CAMB) + thermal-dust (PySM3)
sky, the analytic Planck scan (astropy) with the orbital dipole (ducc0), written to HDF5 and read
back. It therefore needs ``camb``, ``pysm3``, ``ducc0`` and ``astropy`` importable -- heavier than
``test_simgen_transfer.py`` (pure numpy), which covers the filter math and the pipeline wiring.

The band has two shared-boresight detectors with identical pointing, polarization angle and gain, and
zero noise: ``det_tf`` carries a single-pole time constant, ``det_id`` opts out. The filtered
detector's written TOD must equal the plain detector's TOD passed through the same ``SinglePole``.
"""
import os

import numpy as np
import pytest


@pytest.mark.parametrize("compress", [False, True])
def test_end_to_end_transfer_function(tmp_path, compress):
    pytest.importorskip("camb")
    pytest.importorskip("pysm3")
    pytest.importorskip("ducc0")
    pytest.importorskip("astropy")
    import h5py
    import yaml
    from simgen import pipeline
    from simgen.transfer import SinglePole
    from commander4.backend import utils as cpp_utils

    tau_ms, fsamp, nside = 15.0, 12.0, 16
    out_dir = str(tmp_path / "sim")
    params = {
        "general": {"nside": nside, "units": "uK_RJ", "double_precision": False,
                    "seed": 3, "output_dir": out_dir},
        "components": {
            "CMB": {"enabled": True, "component_class": "CMB",
                    "params": {"polarization": "IQU", "shortname": "cmb", "lmax": 47,
                               "solar_dipole": True}},
            "ThermalDust": {"enabled": True, "component_class": "ThermalDust",
                            "params": {"polarization": "IQU", "shortname": "dust", "beta": 1.54,
                                       "T": 20, "nu_ref": 857, "lmax": 47,
                                       "template": {"source": "pysm3", "preset": "d0"}}}},
        "simulation": {
            "nscans": 1, "scan_duration_sec": 8, "npsi": 256,
            "orbital_dipole": True, "compress": compress,
            "pointing": {"strategy": "planck_scan", "los_angle": 85.0, "spin_angle_tilt": 7.5,
                         "spin_rate": 1.0, "mission_start": "2009-08-13T00:00:00",
                         "precession_period_days": 1.0},
            "noise": {"white": True},   # sigma0 = 0 below -> zero-noise, deterministic TOD
        },
        "experiments": {"Sat": {"enabled": True, "bands": {"B100": {
            "enabled": True, "freq": 100.0, "fwhm": 40.0, "fsamp": fsamp,
            "eval_nside": nside, "data_nside": nside, "sigma0": 0.0, "polarization": "I",
            "detectors": {
                "det_tf": {"psi_offset_deg": 0.0, "tau_ms": tau_ms},
                "det_id": {"psi_offset_deg": 0.0, "transfer_function": {"enabled": False}}}}}}},
    }
    param_path = str(tmp_path / "param.yml")
    with open(param_path, "w") as f:
        yaml.dump(params, f, sort_keys=False)

    assert pipeline.run(param_path) == 0

    scan_file = os.path.join(out_dir, "B100", "scan_000001.h5")
    with h5py.File(scan_file, "r") as f:
        ntod = int(f["000001/common/ntod"][0])
        assert np.isclose(f["common/fsamp"][0], fsamp)

        def read_tod(det):
            return f[f"000001/{det}/tod"][:].astype(np.float64)

        def read_pix(det):
            raw = f[f"000001/{det}/pix"][()]
            if not compress:
                return raw.astype(np.int64)
            tree = f["000001/common/hufftree"][:]
            symb = f["000001/common/huffsymb"][:]
            enc = np.frombuffer(raw.tobytes(), dtype=np.uint8)
            diffs = cpp_utils.huffman_decode(enc, tree, symb, np.empty(ntod, dtype=np.int64))
            return np.cumsum(diffs)   # pointing is diff-encoded before Huffman

        tod_tf, tod_id = read_tod("det_tf"), read_tod("det_id")
        # Shared boresight => identical pointing for both detectors (sanity check on the setup).
        np.testing.assert_array_equal(read_pix("det_tf"), read_pix("det_id"))

    # The plain detector's TOD is the clean signal (zero noise); the filtered one must match it
    # passed through the single-pole response baked in from `tau_ms`.
    expected = SinglePole(tau_ms * 1e-3).apply(tod_id, fsamp)
    np.testing.assert_allclose(tod_tf, expected, rtol=1e-4, atol=1e-4)
    assert not np.allclose(tod_tf, tod_id)          # the transfer function changed the TOD
    # DC (calibration) preserved up to the small mirrored-boundary effect over a short scan.
    assert np.isclose(tod_tf.mean(), tod_id.mean(), rtol=2e-2)
