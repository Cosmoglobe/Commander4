"""Template components: a fixed spatial template with a sampled overall scaling.

Both classes here are placeholders awaiting the template sampler; nothing constructs them yet.
"""
from commander4.sky.component import Component


class TemplateComponent(Component):
    """Placeholder for a component with a fixed spatial template and a sampled scaling."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} is not implemented.")


class CMBRelQuad(TemplateComponent):
    """Placeholder for the CMB relativistic quadrupole template. Not implemented yet."""
