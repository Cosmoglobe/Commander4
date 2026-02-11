from pixell.bunch import Bunch

class Params(Bunch):
    """
    An extension of the pixell.bunch.Bunch class. The Bunch class can be seen as an extension of
    Python dictionaries, but allows for `mydict.key` syntax, in addition to mydict['key'] syntax.
    The 'Params' class further improves upon Bunch in two ways:
    1. It allows nested dictionary input, such that an element in the Bunch can itself be a Bunch.
    2. There is a special `.name` property, which contains the key of the Bunch. Say that 'bands'
       is a Params object, and that you assign `myband = bands.Planck30GHz`. By doing `myband.name`,
       or `str(myband)`, you are able to recover the fact that `Planck30GHz` was the key used, a
       functionality that does not exist neither in Python dictionaries or pixell.bunch.Bunch.

    Args:
        *args: Dictionaries or iterables (passed to Bunch).
        name (str): The name/ID of this specific node (e.g., 'Planck30GHz').
        **kwargs: Arbitrary key-value pairs (passed to Bunch).
    """
    def __init__(self, *args, name=None, **kwargs):
        # 1. Store the name using object.__setattr__ to bypass Bunch's 
        #    custom __setattr__ (which would otherwise add it as a dict key).
        object.__setattr__(self, "_name", name)
        
        # 2. Initialize the parent Bunch (behaves like a dict)
        super().__init__(*args, **kwargs)
        
        # 3. Recursively populate the data
        for key, val in list(self.items()):
            if isinstance(val, dict):
                # RECURSION: Pass the child dict AND the key as the name
                self[key] = Params(val, name=key)
        # We don't need to handle the non-recursive "leaf" nodes explicitly, as this is
        # handled by pixell.Bunch, which we have already called the constructor of.

    @property
    def name(self):
        """Read-only access to the node name."""
        return self._name

    def __str__(self):
        """Returns the name if present, otherwise default string representation."""
        return self._name if self._name else super().__str__()

    def __repr__(self):
        """Clean debugging representation showing the name."""
        base_repr = super().__repr__()
        if self._name:
            return f"<Params '{self._name}': {base_repr}>"
        return base_repr