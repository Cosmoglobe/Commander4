"""Sky components: the model that component separation solves for.

The component classes are re-exported here so that a parameter file's `component_class:` name can
be resolved with a plain attribute lookup (`getattr(commander4.sky, "ThermalDust")`), which is what
`CompList`, the chain plotter and the simulator all do. Adding a new component means adding it to
the import below; nothing else needs to know where it lives.

Note this is the one package in Commander4 whose `__init__.py` imports anything: everywhere else,
imports state the full module path.
"""
from commander4.sky.component import Component, TemplateComponent, CMBRelQuad
from commander4.sky.diffuse_components import (DiffuseComponent, CMB, ThermalDust, Synchrotron,
                                               FreeFree, SpinningDust)
from commander4.sky.point_sources import PointSourcesComponent, RadioSources
