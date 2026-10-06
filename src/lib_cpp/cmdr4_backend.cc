#include "ducc0/bindings/pybind_utils.h"
#include "compression_pymod.cc"
#include "mapmaker_pymod.cc"
#include "compsep_pymod.cc"

using namespace cmdr4;

NB_MODULE(PKGNAME, m)
  {
#define CMDR4_XSTRINGIFY(s) CMDR4_STRINGIFY(s)
#define CMDR4_STRINGIFY(s) #s
  m.attr("__version__") = CMDR4_XSTRINGIFY(PKGVERSION);
#undef CMDR4_STRINGIFY
#undef CMDR4_XSTRINGIFY

  add_compression(m);
  add_mapmaker(m);
  add_compsep(m);
  }
