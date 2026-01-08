"""This package, "cmdr4_support", is a wrapper for the  C++ "_cmd4_backend" package.
It doesn't add any functionality, but simply makes "cmdr4_support" the entry point for all
compiled code. This separation could prove useful in the future.
"""

from commander4._cmdr4_backend import *