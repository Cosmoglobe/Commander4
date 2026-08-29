"""The parameter file's top-level shape, and the quantities Commander4 derives from it.

Three things here would otherwise fail late or silently: a parameter file still written against the
old `general` layout (every key would resolve to something else, or to nothing), the MPI task counts
that used to be stated and are now computed, and which mapmaker a band ends up using.

The `optimize` / `sample_amplitudes` polarity is pinned in test_cg_amplitude_sampling.py, next to
the solver whose behaviour it actually changes.
"""
import pytest
from pixell.bunch import Bunch

import numpy as np

from commander4.file_io.experiments.read_utils import apply_noise_priors
from commander4.tod.noise.psd import NoisePSDOof
from commander4.parameters.schema import (TOP_LEVEL_BLOCKS, validate_param_schema, compsep_enabled,
                                     derive_task_counts, task_count_breakdown, resolve_param,
                                     resolve_band_lmax)


def _band(num_tasks=1, enabled=True, **kw):
    return Bunch(enabled=enabled, num_tasks=num_tasks, **kw)


def _params(compsep_bands=(), tod_bands=(("BandA", 4), ("BandB", 2)), groups=True, **compsep_kw):
    """A minimal params Bunch: one enabled experiment, plus whatever compsep config is asked for.

    `groups` adds one enabled CG sampling group, since compsep ranks are only allocated when there
    is something to sample. Pass `groups=False` for the "nothing enabled" case.
    """
    compsep = Bunch(bands=Bunch({name: Bunch(enabled=en, polarization=pol)
                                 for name, pol, en in compsep_bands}), **compsep_kw)
    if groups:
        compsep.cg_sampling_groups = Bunch(amps=Bunch(enabled=True))
    return Bunch(
        experiments=Bunch(EXP=Bunch(enabled=True, bands=Bunch(
            {name: _band(n) for name, n in tod_bands}))),
        compsep=compsep,
    )


# --------------------------------------------------------------------------------------
# Rejecting the old schema
# --------------------------------------------------------------------------------------
def test_the_replaced_general_block_is_refused():
    """A pre-restructure file must fail loudly, naming where its settings went."""
    with pytest.raises(ValueError) as excinfo:
        validate_param_schema({"general": {"niter_gibbs": 4}, "components": {}})
    message = str(excinfo.value)
    assert "general" in message
    # Specifically the "your file is the old layout" message, not the generic unknown-key one:
    # it has to say where each group of settings went, not just that something is unrecognised.
    assert "was replaced by" in message
    assert "Gibbs loop control" in message and "thread counts" in message
    for block in TOP_LEVEL_BLOCKS:
        assert block in message
    assert "proposed_param_layout.yml" in message


def test_an_unknown_top_level_block_is_refused():
    with pytest.raises(ValueError) as excinfo:
        validate_param_schema({"gibbs": {}, "CompSep_bands": {}, "typo_here": {}})
    message = str(excinfo.value)
    assert "CompSep_bands" in message and "typo_here" in message
    assert "compsep" in message   # lists the valid blocks


def test_the_seven_blocks_are_accepted():
    validate_param_schema({block: {} for block in TOP_LEVEL_BLOCKS})
    validate_param_schema({"gibbs": {}})   # a partial file is a different problem, not this one


def test_the_removed_log_filename_is_refused():
    params = {
        "output": {
            "logging": {
                "file": {"level": "info", "filename": "custom.log"},
            },
        },
    }
    with pytest.raises(ValueError, match="run-<run-id>.log"):
        validate_param_schema(params)


def test_the_renamed_float_precision_is_refused():
    """A stale 'single' would be truthy under the new name, i.e. silently select double."""
    with pytest.raises(ValueError, match="double_precision"):
        validate_param_schema({"compsep": {"float_precision": "single"}})


@pytest.mark.parametrize("value", ["false", "single", 0, 1])
def test_a_non_boolean_double_precision_is_refused(value):
    with pytest.raises(ValueError, match="must be true or false"):
        validate_param_schema({"compsep": {"double_precision": value}})


def test_double_precision_accepts_booleans_and_may_be_omitted():
    validate_param_schema({"compsep": {"double_precision": True}})
    validate_param_schema({"compsep": {"double_precision": False}})
    validate_param_schema({"compsep": {}})


# --------------------------------------------------------------------------------------
# Derived MPI task counts
# --------------------------------------------------------------------------------------
def test_task_counts_are_derived_from_the_band_configuration():
    params = _params(compsep_bands=(("BandA", "IQU", True), ("BandB", "I", True)))
    counts = derive_task_counts(params)
    assert (counts.tod, counts.compsep_I, counts.compsep_QU) == (6, 2, 1)
    assert counts.total == 9


def test_disabled_bands_and_experiments_contribute_nothing():
    params = _params(compsep_bands=(("BandA", "IQU", True), ("BandB", "IQU", False)))
    params.experiments.EXP.bands.BandB.enabled = False
    counts = derive_task_counts(params)
    assert (counts.tod, counts.compsep_I, counts.compsep_QU, counts.total) == (4, 1, 1, 6)

    params.experiments.EXP.enabled = False
    assert derive_task_counts(params).tod == 0


def test_tod_only_mode_allocates_no_compsep_ranks():
    """`compsep.enabled: false` is how TOD-only runs are spelled now that the counts are derived."""
    bands = (("BandA", "IQU", True), ("BandB", "IQU", True))
    assert derive_task_counts(_params(compsep_bands=bands)).total == 6 + 4
    counts = derive_task_counts(_params(compsep_bands=bands, enabled=False))
    assert (counts.compsep_I, counts.compsep_QU, counts.total) == (0, 0, 6)
    assert compsep_enabled(_params(compsep_bands=bands)) is True       # defaults to on
    assert compsep_enabled(_params(compsep_bands=bands, enabled=False)) is False


def test_no_enabled_sampling_group_allocates_no_compsep_ranks():
    """Compsep ranks exist to run sampling groups; with none enabled they would sample nothing.

    Allocating them anyway reserves nodes and memory just to forward maps and evaluate a
    chi-squared, when TOD-only reaches the same fixed sky model on its own.
    """
    bands = (("BandA", "IQU", True), ("BandB", "IQU", True))
    assert derive_task_counts(_params(compsep_bands=bands)).total == 6 + 4

    counts = derive_task_counts(_params(compsep_bands=bands, groups=False))
    assert (counts.compsep_I, counts.compsep_QU, counts.total) == (0, 0, 6)
    assert compsep_enabled(_params(compsep_bands=bands, groups=False)) is False

    # A group present but switched off counts as none enabled.
    params = _params(compsep_bands=bands, groups=False)
    params.compsep.mcmc_sampling_groups = Bunch(specind=Bunch(enabled=False))
    assert compsep_enabled(params) is False
    params.compsep.mcmc_sampling_groups.specind.enabled = True
    assert compsep_enabled(params) is True


def test_a_sampling_group_with_compsep_switched_off_is_refused():
    """`compsep.enabled: false` plus an enabled group is contradictory, and must not run silently."""
    import commander4.diagnostics.log  # noqa: F401  -- registers the SUMMARY/VERBOSE levels.
    from commander4.mpi.setup import init_mpi

    params = _params(compsep_bands=(("BandA", "IQU", True),), enabled=False)
    params.resources = Bunch(tod=Bunch(num_threads=1), compsep=Bunch(num_threads=1))
    with pytest.raises(RuntimeError, match="compsep.enabled' is false"):
        init_mpi(params)


def test_the_breakdown_names_every_term():
    """The 'wrong mpirun -n' message must be actionable without hand-counting bands."""
    counts = derive_task_counts(_params(compsep_bands=(("BandA", "IQU", True),)))
    text = task_count_breakdown(counts)
    assert "8" in text and "6" in text and "TOD" in text and "CompSep-I" in text


# --------------------------------------------------------------------------------------
# Scoped parameter lookup (`resolve_param`)
# --------------------------------------------------------------------------------------
MAPMAKER_SCOPES = ("experiments.EXP.bands.BAND", "experiments.EXP", "tod_processing")


def _mapmaker_params(global_mm=None, exp_mm=None, band_mm=None):
    band = Bunch(enabled=True, num_tasks=1)
    if band_mm is not None:
        band.mapmaker = band_mm
    experiment = Bunch(enabled=True, bands=Bunch(BAND=band))
    if exp_mm is not None:
        experiment.mapmaker = exp_mm
    tod = Bunch()
    if global_mm is not None:
        tod.mapmaker = global_mm
    return Bunch(experiments=Bunch(EXP=experiment), tod_processing=tod)


def _mapmaker(**kw):
    return resolve_param(_mapmaker_params(**kw), "mapmaker", MAPMAKER_SCOPES)


def test_the_first_scope_that_defines_the_key_wins():
    """Mapmaker precedence: band, then experiment, then the global default."""
    assert _mapmaker(global_mm="bin") == "bin"
    assert _mapmaker(global_mm="bin", exp_mm="CG") == "CG"
    assert _mapmaker(global_mm="bin", exp_mm="CG", band_mm="bin") == "bin"
    # An experiment override still beats the global default when the band says nothing.
    assert _mapmaker(global_mm="CG", exp_mm="bin") == "bin"


def test_a_scope_that_does_not_exist_is_an_error_by_default():
    """There is no good reason to list a scope that isn't there."""
    params = Bunch(tod_processing=Bunch(mapmaker="bin"))
    with pytest.raises(ValueError, match="no.such.scope"):
        resolve_param(params, "mapmaker", ("no.such.scope", "tod_processing"))
    # Even when a later scope would have answered, and even with a default available.
    with pytest.raises(ValueError, match="no.such.scope"):
        resolve_param(params, "mapmaker", ("no.such.scope", "tod_processing"), default="CG")


def test_an_optional_scope_may_be_absent_without_complaint(caplog):
    """A per-band override block most bands do not carry is not a mistake, so opting out of the
    error must not just move the noise to the log: it drops to debug."""
    params = Bunch(tod_processing=Bunch(mapmaker="bin"))
    with caplog.at_level("ERROR", logger="commander4.parameters.schema"):
        assert resolve_param(params, "mapmaker", ("no.such.scope", "tod_processing"),
                             raise_on_missing_scope=False) == "bin"
    assert caplog.text == ""
    with caplog.at_level("DEBUG", logger="commander4.parameters.schema"):
        resolve_param(params, "mapmaker", ("no.such.scope", "tod_processing"),
                      raise_on_missing_scope=False)
    assert "no.such.scope" in caplog.text


def test_legal_values_rejects_a_mistyped_value():
    params = Bunch(tod_processing=Bunch(gap_fill_method="wn"))
    assert resolve_param(params, "gap_fill_method", ("tod_processing",),
                         legal_values=("wn", "fallback")) == "wn"
    with pytest.raises(ValueError) as excinfo:
        resolve_param(params, "gap_fill_method", ("tod_processing",),
                      legal_values=("fallback", "full_cg"))
    message = str(excinfo.value)
    assert "'wn'" in message and "fallback" in message and "tod_processing" in message


def test_legal_values_also_checks_the_default():
    """A bad default is a programming error, and just as worth catching as a bad file value."""
    with pytest.raises(ValueError, match="the default"):
        resolve_param(Bunch(a=Bunch()), "x", ("a",), default="bogus", legal_values=("ok",))


def test_a_key_defined_nowhere_raises_and_names_every_scope_searched():
    with pytest.raises(ValueError) as excinfo:
        resolve_param(_mapmaker_params(), "mapmaker", MAPMAKER_SCOPES)
    message = str(excinfo.value)
    assert "mapmaker" in message
    for scope in MAPMAKER_SCOPES:
        assert scope in message


def test_a_default_is_returned_instead_of_raising():
    assert resolve_param(_mapmaker_params(), "mapmaker", MAPMAKER_SCOPES, default="bin") == "bin"
    # A configured value still wins over the default.
    assert resolve_param(_mapmaker_params(exp_mm="CG"), "mapmaker", MAPMAKER_SCOPES,
                         default="bin") == "CG"


def test_none_is_a_usable_default():
    """The 'no default' sentinel must be distinct from None, or `default=None` would raise."""
    assert resolve_param(_mapmaker_params(), "mapmaker", MAPMAKER_SCOPES, default=None) is None
    # ... and a falsy default is returned as-is rather than falling through to the error.
    assert resolve_param(_mapmaker_params(), "mapmaker", MAPMAKER_SCOPES, default=0) == 0


def test_falsy_values_still_count_as_defined():
    """`0`/`False`/`""` are answers, not absences, so a narrow scope setting one must win."""
    params = Bunch(a=Bunch(x=0), b=Bunch(x=5))
    assert resolve_param(params, "x", ("a", "b")) == 0
    params = Bunch(a=Bunch(x=False), b=Bunch(x=True))
    assert resolve_param(params, "x", ("a", "b")) is False


def test_where_the_value_came_from_is_logged(caplog):
    """An overridden setting is otherwise invisible in a chain's log."""
    with caplog.at_level("DEBUG", logger="commander4.parameters.schema"):
        _mapmaker(global_mm="bin", exp_mm="CG")
    assert "mapmaker" in caplog.text and "experiments.EXP" in caplog.text


# --- band lmax ------------------------------------------------------------------------------
# A band's lmax is the ceiling on what any component can be fitted against; C3 states it per band
# (BAND_LMAX) and so do we, with the full HEALPix bandlimit as the fallback.

def _lmax_params(band_lmax=None, exp_lmax=None, file_lmax=None):
    band = Bunch(enabled=True)
    if band_lmax is not None:
        band.lmax = band_lmax
    experiment = Bunch(enabled=True, bands=Bunch(BandA=band))
    if exp_lmax is not None:
        experiment.lmax = exp_lmax
    compsep_band = Bunch(enabled=True, get_from="file")
    if file_lmax is not None:
        compsep_band.lmax = file_lmax
    return Bunch(experiments=Bunch(EXP=experiment),
                 compsep=Bunch(bands=Bunch(BandA=compsep_band)))


def test_band_lmax_defaults_to_the_full_healpix_bandlimit():
    """3*nside-1 is everything a map at this resolution can carry, so nothing is left unconstrained
    merely by the band's own lmax."""
    assert resolve_band_lmax(_lmax_params(), "BandA", "EXP", 64) == 191
    assert resolve_band_lmax(_lmax_params(), "BandA", "EXP", 512) == 1535


def test_band_lmax_prefers_the_band_over_the_experiment():
    assert resolve_band_lmax(_lmax_params(exp_lmax=100), "BandA", "EXP", 64) == 100
    assert resolve_band_lmax(_lmax_params(band_lmax=150, exp_lmax=100), "BandA", "EXP", 64) == 150


def test_a_file_band_takes_its_lmax_from_its_compsep_entry():
    """`get_from: file` bands have no experiment block; freq/fwhm/nside already live here."""
    assert resolve_band_lmax(_lmax_params(file_lmax=80), "BandA", None, 64) == 80
    assert resolve_band_lmax(_lmax_params(), "BandA", None, 64) == 191


# ===================================================================
# Noise-PSD prior bounds (experiments/read_utils.apply_noise_priors)
# ===================================================================

def _noise_params(band=None, experiment=None, tod_processing=None):
    """A parameter Bunch carrying `noise_prior_bounds` at zero or more of the three scopes."""
    band_block = Bunch(enabled=True)
    if band is not None:
        band_block.noise_prior_bounds = Bunch(**band)
    exp_block = Bunch(bands=Bunch(BandA=band_block))
    if experiment is not None:
        exp_block.noise_prior_bounds = Bunch(**experiment)
    tod_block = Bunch()
    if tod_processing is not None:
        tod_block.noise_prior_bounds = Bunch(**tod_processing)
    return Bunch(experiments=Bunch(EXP=exp_block), tod_processing=tod_block)


def test_noise_prior_bounds_left_alone_when_the_parameter_file_is_silent():
    """A reader's instrument-appropriate defaults must survive an unconfigured parameter file."""
    model = NoisePSDOof(P_uni=[[np.nan, np.nan], [0.03, 40.0], [-4.0, -2.0]])
    apply_noise_priors(model, _noise_params(), "EXP", "BandA")
    np.testing.assert_allclose(model.P_uni[1], [0.03, 40.0])
    np.testing.assert_allclose(model.P_uni[2], [-4.0, -2.0])


def test_noise_prior_bounds_override_only_the_named_parameters():
    model = NoisePSDOof(P_uni=[[np.nan, np.nan], [0.03, 40.0], [-4.0, -2.0]])
    apply_noise_priors(model, _noise_params(band={"fknee": [0.001, 5.0]}), "EXP", "BandA")
    np.testing.assert_allclose(model.P_uni[1], [0.001, 5.0])
    np.testing.assert_allclose(model.P_uni[2], [-4.0, -2.0])  # untouched


def test_noise_prior_bounds_prefer_the_band_over_wider_scopes():
    model = NoisePSDOof()
    apply_noise_priors(model, _noise_params(band={"fknee": [1.0, 2.0]},
                                                  experiment={"fknee": [3.0, 4.0]},
                                                  tod_processing={"fknee": [5.0, 6.0]}),
                             "EXP", "BandA")
    np.testing.assert_allclose(model.P_uni[1], [1.0, 2.0])

    model = NoisePSDOof()
    apply_noise_priors(model, _noise_params(experiment={"fknee": [3.0, 4.0]},
                                                  tod_processing={"fknee": [5.0, 6.0]}),
                             "EXP", "BandA")
    np.testing.assert_allclose(model.P_uni[1], [3.0, 4.0])


def test_noise_prior_bounds_reject_an_unknown_parameter_name():
    """A typo here would otherwise leave the default bounds silently in force."""
    model = NoisePSDOof()
    with pytest.raises(ValueError, match="fkne"):
        apply_noise_priors(model, _noise_params(band={"fkne": [0.1, 1.0]}), "EXP", "BandA")


def test_noise_prior_sets_the_informative_prior_from_the_parameter_file():
    """`noise_prior: {name: [mean, rms]}` must land in P_active, which the sampler reads."""
    model = NoisePSDOof()
    params = _noise_params()
    params.experiments.EXP.bands.BandA.noise_prior = Bunch(fknee=[5.0, 0.25], alpha=[-2.0, 0.4])
    apply_noise_priors(model, params, "EXP", "BandA")

    np.testing.assert_allclose(model.P_active[1], [5.0, 0.25])
    np.testing.assert_allclose(model.P_active[2], [-2.0, 0.4])
    # The prior is now actually consulted: at the mean it beats a value four rms away.
    assert model.log_prior(2, -2.0) > model.log_prior(2, -3.6)


def test_a_nonpositive_prior_rms_in_the_parameter_file_freezes_that_parameter():
    """The parameter-file route must reach C3's "hold this one fixed" switch, not just the prior."""
    model = NoisePSDOof()
    params = _noise_params()
    params.experiments.EXP.bands.BandA.noise_prior = Bunch(fknee=[10.0, -1.0])
    apply_noise_priors(model, params, "EXP", "BandA")

    assert not model.is_sampled(1)
    assert model.is_sampled(2)


def test_noise_prior_rejects_an_unknown_parameter_name():
    model = NoisePSDOof()
    params = _noise_params()
    params.experiments.EXP.bands.BandA.noise_prior = Bunch(alfa=[1.0, 1.0])
    with pytest.raises(ValueError, match="alfa"):
        apply_noise_priors(model, params, "EXP", "BandA")
