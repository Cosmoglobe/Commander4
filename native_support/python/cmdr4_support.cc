#include "ducc0/infra/string_utils.cc"
#include "ducc0/infra/threading.cc"
#include "ducc0/infra/mav.cc"
//#include "ducc0/math/pointing.cc"
//#include "ducc0/math/geom_utils.cc"
//#include "ducc0/math/space_filling.cc"
//#include "ducc0/math/gl_integrator.cc"
//#include "ducc0/math/gridding_kernel.cc"
//#include "ducc0/math/wigner3j.cc"
//#include "ducc0/sht/sht.cc"
//#include "ducc0/healpix/healpix_tables.cc"
//#include "ducc0/healpix/healpix_base.cc"

#include <pybind11/pybind11.h>
#include "python/utils_pymod.cc"

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
