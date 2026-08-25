#include <pybind11/pybind11.h>
#include "utils_pymod.cc"

using namespace cmdr4;

PYBIND11_MODULE(PKGNAME, m)
  {
#define CMDR4_XSTRINGIFY(s) CMDR4_STRINGIFY(s)
#define CMDR4_STRINGIFY(s) #s
  m.attr("__version__") = CMDR4_XSTRINGIFY(PKGVERSION);
#undef CMDR4_STRINGIFY
#undef CMDR4_XSTRINGIFY

  add_utils(m);
  }
