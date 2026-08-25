"""The modular TOD simulator installed with Commander4 as ``c4-simgen``.

Generates per-detector time-ordered data in the ``litebird_sim`` HDF5 format, read back by the
main program under ``experiment_id: "general"``, with swappable pointing strategies, sky
components, noise models and TOD modifiers. See ``simgen/README.md`` and ``simgen/params/``.
"""
# NB: no submodule imports here on purpose, so that importing ``simgen.config``/``simgen.writers``
# does not pull in the heavy sky stack (pysm3/ducc0/camb). Use ``from simgen.pipeline import run``.
