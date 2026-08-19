"""simgen -- a modular TOD simulator for Commander4.

Generates per-detector time-ordered data in the ``litebird_sim`` HDF5 format (directly readable by
the main program with ``replace_tod_with_sim: false``), with swappable pointing strategies, sky
components, noise models and TOD modifiers. See ``README.md`` and ``example_param.yml``.
"""
# NB: no submodule imports here on purpose -- importing ``simgen.config``/``simgen.writers`` should
# not pull in the heavy sky stack (pysm3/ducc0/camb). Use ``from simgen.pipeline import run``.
