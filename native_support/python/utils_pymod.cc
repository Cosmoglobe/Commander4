#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>
#include <cmath>
#include <complex>

#include "ducc0/infra/mav.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/math/constants.h"
#include "ducc0/bindings/pybind_utils.h"

namespace cmdr4 {

namespace detail_pymodule_utils {

using namespace std;
using namespace ducc0;
namespace py = pybind11;
auto None = py::none();


py::array Py_amplitude_sampling_per_pix_helper (
  const py::array &map_sky_,
  const py::array &map_rms_,
  const py::array &M_,
  py::object &comp_maps__,
  size_t nthreads)
  {
  const auto map_sky = to_cmav<double,2>(map_sky_);
  size_t nband = map_sky.shape(0);
  size_t npix = map_sky.shape(1);
  const auto map_rms = to_cmav<double,2>(map_rms_);
  MR_assert(map_rms.shape()==map_sky.shape(), "map shape mismatch");
  const auto M = to_cmav<double,2>(M_);
  size_t ncomp = M.shape(1);
  MR_assert(M.shape(0)==nband, "M shape mismatch");
  auto comp_maps_ = get_optional_Pyarr<double>(comp_maps__, {ncomp, npix});
  auto comp_maps = to_vmav<double,2>(comp_maps_);
  {
  py::gil_scoped_release release;

/* do computations here */

  }
  return comp_maps_;
  }

void add_utils(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("utils");
//  m.doc() = utils_DS;

  m.def("amplitude_sampling_per_pix_helper",
        Py_amplitude_sampling_per_pix_helper,
//        Py_amplitude_sampling_per_pix_helper_DS,
        "map_sky"_a,
        "map_rms"_a,
        "M"_a,
        "comp_maps"_a=None,
        "nthreads"_a=1);
  }

}

using detail_pymodule_utils::add_utils;

}
