"""Current-format chain discovery and plotting."""

from pathlib import Path

import h5py
import healpy as hp
import numpy as np

from commander4.diagnostics import plotting
from commander4.standalone_tools.plot_chain import (
    ChainFile,
    _parse_int_set,
    _read_detector_summary,
    _resample_map,
    main,
)


def _write_band_file(run_dir: Path, iteration: int) -> None:
    path = run_dir / "chains_bands" / f"Exp_Band_chain01_iter{iteration:04d}.h5"
    nscan, ndet, nbin = 3, 2, 4
    with h5py.File(path, "w") as handle:
        handle["metadata/band_unit"] = "uK_RJ"
        handle["metadata/map_fwhm_arcmin"] = 30.0
        handle["scan_ids"] = [10, 20, 30]
        handle["scan_start_time"] = [1.0, 2.0, 3.0]
        handle["det_names"] = ["d0", "d1"]
        handle["abs_gain"] = 1.0 + iteration
        handle["detrel_gain"] = [0.1, -0.1]
        handle["gain_prior"] = np.ones((ndet, 3))
        handle["present"] = np.ones((nscan, ndet), dtype=np.int8)
        handle["accept"] = np.ones((nscan, ndet), dtype=np.int8)
        handle["temporal_gain"] = np.full((nscan, ndet), iteration, dtype=float)
        handle["noise_params"] = np.ones((nscan, ndet, 3))
        handle["good_fraction"] = np.full((nscan, ndet), 0.9)
        handle["chisq_z"] = np.zeros((nscan, ndet))
        handle["ncorr_cg_residual"] = np.full((nscan, ndet), 1e-5)
        handle["ncorr_cg_niter"] = np.full((nscan, ndet), 5)
        handle["ncorr_converged"] = np.ones((nscan, ndet), dtype=np.int8)
        handle["jump_counts"] = np.zeros((nscan, ndet), dtype=int)
        handle["orbital_velocity"] = np.ones((nscan, ndet, 3))
        freqs = np.geomspace(0.1, 10.0, nbin)
        handle["tod_ps_freqs"] = np.broadcast_to(freqs, (nscan, ndet, nbin))
        for dataset_name in (
            "tod_ps_raw",
            "tod_ps_ncorr",
            "tod_ps_ncorrsub",
            "tod_ps_residual",
        ):
            handle[dataset_name] = np.ones((nscan, ndet, nbin))

        if iteration == 1:
            npix = hp.nside2npix(1)
            handle["maps/observed_sky"] = np.arange(3 * npix).reshape(3, npix)
            handle["maps/rms"] = np.ones((3, npix))
            handle["maps/nhit"] = np.ones(npix, dtype=int)
            handle["maps/cov"] = np.ones((6, npix))
            handle["maps/corrnoise"] = np.ones((3, npix))
            handle["maps/skymodel"] = np.ones((3, npix))
            handle["maps/res"] = np.zeros((3, npix))
            handle["maps/orbdipole"] = np.full((3, npix), np.nan)


def _write_compsep_file(run_dir: Path, iteration: int) -> None:
    path = run_dir / "chains_compsep" / f"chain01_iter{iteration:04d}.h5"
    lmax = 2
    alm_size = hp.Alm.getsize(lmax)
    with h5py.File(path, "w") as handle:
        handle["chi2/total"] = 100.0 + iteration
        handle["chi2/ndof"] = 100.0
        handle["chi2/reduced"] = 1.0 + iteration / 100.0
        handle["chi2/z"] = iteration / 10.0
        handle["chi2/bands/Band_I/chi2"] = 50.0
        handle["chi2/bands/Band_I/ndof"] = 50.0
        handle["chi2/bands/Band_I/reduced"] = 1.0
        handle["chi2/bands/Band_I/nu"] = 30.0
        handle["chi2/map"] = np.ones((3, hp.nside2npix(1)))
        handle["residuals/Band_QU"] = np.ones((2, hp.nside2npix(1)))
        residuals = np.geomspace(1.0, 1e-5, 5)
        handle["amplitude_groups/amps/I/cg_residuals"] = residuals
        handle["amplitude_groups/amps/I/n_iter"] = residuals.size
        handle["amplitude_groups/amps/QU/cg_residuals"] = residuals[:3]
        handle["amplitude_groups/amps/QU/n_iter"] = 3
        alms = np.zeros((3, alm_size), dtype=np.complex128)
        alms[:, 0] = 1.0
        alms[:, hp.Alm.getidx(lmax, 2, 0)] = 0.2
        handle["comps/cmb/alms"] = alms
        handle["comps/cmb/lmax"] = lmax
        handle["comps/cmb/sigma_l"] = np.ones((3, lmax + 1))
        handle["comps/cmb/amp_fwhm_arcmin"] = 0.0
        handle["comps/cmb/sed/nu_ref"] = 30.0
        handle["comps/cmb/sed/sample_param"] = float(iteration)
        handle["comps/cmb/mixing/Band"] = 1.0


def _make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "chains_bands").mkdir(parents=True)
    (run_dir / "chains_compsep").mkdir()
    for iteration in (1, 2):
        _write_band_file(run_dir, iteration)
        _write_compsep_file(run_dir, iteration)
    return run_dir


def test_parse_int_set_handles_lists_and_inclusive_ranges():
    assert _parse_int_set("1,3-5") == {1, 3, 4, 5}
    assert _parse_int_set("3-1") == {1, 2, 3}
    assert _parse_int_set("all") is None


def test_resample_map_preserves_counts_and_combines_rms_as_inverse_variance():
    counts = np.ones(hp.nside2npix(2))
    degraded_counts = _resample_map(counts, 1, "count")[0]
    np.testing.assert_allclose(degraded_counts, 4.0)

    rms = np.full(hp.nside2npix(2), 2.0)
    degraded_rms = _resample_map(rms, 1, "rms")[0]
    np.testing.assert_allclose(degraded_rms, 2.0)


def test_detector_summary_reads_large_detector_axes_in_chunks(tmp_path: Path):
    path = tmp_path / "Exp_Band_chain01_iter0001.h5"
    detector_count = 300
    with h5py.File(path, "w") as handle:
        handle["det_names"] = [f"d{index}" for index in range(detector_count)]
        handle["present"] = np.ones((2, detector_count), dtype=np.int8)
        accept = np.ones((2, detector_count), dtype=np.int8)
        accept[0, 129] = 0
        handle["accept"] = accept
        noise_params = np.ones((2, detector_count, 3))
        noise_params[:, :, 0] = np.arange(detector_count)[None, :]
        handle["noise_params"] = noise_params

    entry = ChainFile(str(path), chain=1, iteration=1, band="Exp_Band")
    detector_names, metrics = _read_detector_summary(
        entry, {"d0", "d129", "d299"}
    )

    assert detector_names == ["d0", "d129", "d299"]
    np.testing.assert_allclose(metrics["noise_sigma0"], [0.0, 129.0, 299.0])
    np.testing.assert_allclose(metrics["accept_fraction"], [1.0, 0.5, 1.0])


def test_main_does_not_create_an_output_directory_when_there_are_no_plots(tmp_path: Path):
    run_dir = tmp_path / "empty_run"
    (run_dir / "chains_bands").mkdir(parents=True)
    (run_dir / "chains_compsep").mkdir()
    output_dir = tmp_path / "empty_plots"

    assert main([str(run_dir), "--output-dir", str(output_dir)]) == 0
    assert not output_dir.exists()


def test_main_writes_filtered_current_format_plots(tmp_path: Path, monkeypatch):
    run_dir = _make_run(tmp_path)
    output_dir = tmp_path / "plots"
    captured_series = {}
    original_plot_chain_line_panels = plotting.plot_chain_line_panels

    def capture_plot_chain_line_panels(filename, title, xlabel, panels):
        for panel in panels:
            key = f"{title}:{panel.title}"
            captured_series[key] = [label for label, _, _ in panel.series]
        return original_plot_chain_line_panels(filename, title, xlabel, panels)

    monkeypatch.setattr(plotting, "plot_chain_line_panels", capture_plot_chain_line_panels)

    result = main([
        str(run_dir),
        "--output-dir", str(output_dir),
        "--plots", "all",
        "--iter", "1",
        "--band", "Exp_Band",
        "--detector", "d0",
        "--nside", "1",
    ])

    assert result == 0
    assert (output_dir / "maps_bands/observed_sky/chain01_Exp_Band_iter0001.png").is_file()
    assert (output_dir / "maps_bands/cov/chain01_Exp_Band_iter0001.png").is_file()
    assert (output_dir / "maps_bands/nhit/chain01_Exp_Band.png").is_file()
    assert (output_dir / "maps_bands/corrnoise/chain01_Exp_Band_iter0001.png").is_file()
    assert (output_dir / "maps_bands/skymodel/chain01_Exp_Band_iter0001.png").is_file()
    assert not (output_dir / "maps_bands/res").exists()
    assert not (output_dir / "maps_bands/orbdipole").exists()
    assert not (output_dir / "maps_bands/observed_sky/chain01_Exp_Band_iter0002.png").exists()
    residual = output_dir / "maps_compsep/residuals/chain01_residual_Band_QU_iter0001.png"
    assert residual.is_file()
    noise_dir = output_dir / "tod_scans/noise_params"
    assert (noise_dir / "chain01_Exp_Band_d0.png").is_file()
    assert not (noise_dir / "chain01_Exp_Band_d1.png").exists()
    assert (output_dir / "tod_scans/data_quality/chain01_Exp_Band_d0.png").is_file()
    assert (output_dir / "tod_scans/ncorr_solver/chain01_Exp_Band_d0.png").is_file()
    assert not (output_dir / "tod_scans/chisq_z").exists()
    assert not (output_dir / "tod_scans/orbital_velocity").exists()
    assert not (output_dir / "tod_scans/scan_start_time").exists()
    assert not (output_dir / "tod_traces/gain_prior").exists()
    density_dir = output_dir / "tod_scans/noise_params_fknee_alpha"
    assert (density_dir / "chain01_Exp_Band_d0_iter0001.png").is_file()
    assert not (density_dir / "chain01_Exp_Band_d0_iter0002.png").exists()
    chi2_trace = output_dir / "compsep_traces/chi2/chain01_per_band_reduced_I.png"
    assert chi2_trace.is_file()
    assert (output_dir / "compsep_cg/amps/chain01_iter0001.png").is_file()
    assert not (output_dir / "compsep_cg/amps/I").exists()
    assert (output_dir / "maps_components/cmb/chain01_iter0001.png").is_file()
    assert not (output_dir / "maps_components/cmb/chain01_iter0002.png").exists()
    assert (output_dir / "spectra_components/cmb/full_sky/chain01_Cl.png").is_file()
    assert (output_dir / "spectra_components/cmb/galactic_cut/chain01_Cl.png").is_file()
    assert not (output_dir / "spectra_components/dust/galactic_cut").exists()
    assert not (output_dir / "source_amplitudes").exists()
    sampled_trace = output_dir / "component_traces/cmb/sed/chain01_comps_cmb_sed_sample_param.png"
    assert sampled_trace.is_file()
    assert not (output_dir / "component_traces/cmb/sed/chain01_comps_cmb_sed_nu_ref.png").exists()
    assert not (output_dir / "compsep_traces/chi2/chain01_chi2_total.png").exists()
    assert not any(path.is_dir() for path in output_dir.rglob("chain*"))
    noise_key = "Exp_Band d0, chain 1: noise parameters:sigma0"
    assert captured_series[noise_key] == ["iter 1", "iter 2"]
    assert captured_series["cmb realized spectra; chain 1:E"] == ["iter 1", "iter 2"]
    cut_key = "cmb pseudo-spectra, |b| >= 20 deg; chain 1:T"
    assert captured_series[cut_key] == ["iter 1", "iter 2"]


def test_summary_mode_replaces_individual_detector_plots(tmp_path: Path, monkeypatch):
    run_dir = _make_run(tmp_path)
    output_dir = tmp_path / "summary_plots"
    path = run_dir / "chains_bands/Exp_Band_chain01_iter0001.h5"
    with h5py.File(path, "r+") as handle:
        values = np.asarray(handle["noise_params"][()])
        values[:, 0, 0] = [1.0, 1.0, 100.0]
        values[:, 1, 0] = 10.0
        handle["noise_params"][...] = values

    captured_medians = {}
    original_plot_chain_line_panels = plotting.plot_chain_line_panels

    def capture_plot_chain_line_panels(filename, title, xlabel, panels):
        for panel in panels:
            for label, x_values, y_values in panel.series:
                if label == "median":
                    captured_medians[panel.title] = (x_values, y_values)
        return original_plot_chain_line_panels(filename, title, xlabel, panels)

    monkeypatch.setattr(plotting, "plot_chain_line_panels", capture_plot_chain_line_panels)

    result = main([
        str(run_dir),
        "--output-dir", str(output_dir),
        "--plots", "tod",
        "--detector-plots", "summary",
        "--iter", "1",
    ])

    assert result == 0
    summary_root = output_dir / "tod_summaries"
    assert (summary_root / "gain/chain01_Exp_Band.png").is_file()
    assert (summary_root / "noise/chain01_Exp_Band.png").is_file()
    assert (summary_root / "data_quality/chain01_Exp_Band.png").is_file()
    assert (summary_root / "ncorr_solver/chain01_Exp_Band.png").is_file()
    density = summary_root / "noise_params_fknee_alpha/chain01_Exp_Band_iter0001.png"
    assert density.is_file()
    assert not (
        summary_root / "noise_params_fknee_alpha/chain01_Exp_Band_iter0002.png"
    ).exists()
    assert (summary_root / "power_spectra/chain01_Exp_Band_iter0001.png").is_file()
    assert not (summary_root / "power_spectra/chain01_Exp_Band_iter0002.png").exists()
    table = summary_root / "detector_tables/chain01_Exp_Band_iter0002.csv"
    assert table.is_file()
    table_detectors = {line.split(",", 1)[0] for line in table.read_text().splitlines()[1:]}
    assert table_detectors == {"d0", "d1"}

    assert (output_dir / "tod_traces/abs_gain/chain01_Exp_Band.png").is_file()
    assert not (output_dir / "tod_traces/detrel_gain").exists()
    assert not (output_dir / "tod_scans/noise_params").exists()
    assert not (output_dir / "tod_scans/noise_params_fknee_alpha").exists()
    assert not (output_dir / "tod_power_spectra").exists()
    # Per-detector scan medians are [1, 10], whose equally detector-weighted median is 5.5. Pooling
    # all detector-scans directly would instead give 10 because d1 contributes three tens.
    iterations, sigma0_medians = captured_medians["median sigma0"]
    assert iterations[0] == 1
    assert sigma0_medians[0] == 5.5


def test_none_mode_keeps_only_non_detector_tod_plots(tmp_path: Path):
    run_dir = _make_run(tmp_path)
    output_dir = tmp_path / "no_detector_plots"

    result = main([
        str(run_dir),
        "--output-dir", str(output_dir),
        "--plots", "tod",
        "--detector-plots", "none",
    ])

    assert result == 0
    assert (output_dir / "tod_traces/abs_gain/chain01_Exp_Band.png").is_file()
    assert not (output_dir / "tod_traces/detrel_gain").exists()
    assert not (output_dir / "tod_scans").exists()
    assert not (output_dir / "tod_summaries").exists()
