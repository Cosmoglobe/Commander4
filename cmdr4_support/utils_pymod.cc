#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>
#include <algorithm>
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

// exception type for signalling a (nearly) singular matrix)
struct SingularError {};

// LU decomposition code taken from https://www.johnloomis.org/ece538/notes/2008/Matrix/ludcmp.html

//!  Find pivot element
/*!
*   The function pivot finds the largest element for a pivot in "jcol"
*   of Matrix "a", performs interchanges of the appropriate
*   rows in "a", and also interchanges the corresponding elements in
*   the order vector.
*
*
*  \param     a      -  n by n Matrix of coefficients
*  \param   order  - integer vector to hold row ordering
*  \param    jcol   - column of "a" being searched for pivot element
*
*/
int pivot(vmav<double,2> &a, vmav<int,1> &order, int jcol)
  {
  constexpr double TINY=1e-20;
	 int n = a.shape(0);

 	/*
  	*  Find biggest element on or below diagonal.
	  *  This will be the pivot row.
	  */

 	int ipvt = jcol;
	 double big = std::abs(a(ipvt,ipvt));
	 for (int i = ipvt+1; i<n; i++)
    {
		  double anext = std::abs(a(i,jcol));
	  	if (anext>big)
      {
			   big = anext;
			   ipvt = i;
		    }
	   }
  if (std::abs(big)<TINY)
    throw SingularError();

	 /* Interchange pivot row (ipvt) with current row (jcol). */
 	if (ipvt==jcol) return 0;
	 //a.swaprows(jcol,ipvt);
  for (int i=0; i<n; ++i)
    swap(a(jcol,i), a(ipvt,i));
  swap(order(jcol), order(ipvt));
 	return 1;
  }

int ludcmp(vmav<double,2> &a, vmav<int,1> &order)
  {
	 int flag = 1;    /* changes sign with each row interchange */

 	int n = a.shape(0);
	 MR_assert(a.shape(1)==size_t(n));
  MR_assert(order.shape(0)==size_t(n));

	 /* establish initial ordering in order vector */
	 for (int i=0; i<n; i++)
    order(i) = i;

	 /* do pivoting for first column and check for singularity */
	 if (pivot(a,order,0))
    flag = -flag;
	 double diag = 1.0/a(0,0);
	 for (int i=1; i<n; i++)
    a(0,i) *= diag;
	
 	/* Now complete the computing of L and U elements.
 	 * The general plan is to compute a column of L's, then
  	* call pivot to interchange rows, and then compute
  	* a row of U's. */
 	int nm1 = n - 1;
 	for (int j=1; j<nm1; j++)
    {
	  	/* column of L's */
	  	for (int i=j; i<n; i++)
      {
		   	double sum = 0.0;
		   	for (int k=0; k<j; k++)
        sum += a(i,k)*a(k,j);
      a(i,j) -= sum;
		    }
		  /* pivot, and check for singularity */
		  if (pivot(a,order,j))
      flag = -flag;
		  /* row of U's */
		  diag = 1.0/a(j,j);
		  for (int k=j+1; k<n; k++)
      {
			   double sum = 0.0;
		   	for (int i=0; i<j; i++)
        sum += a(j,i)*a(i,k);
	   		a(j,k) = (a(j,k)-sum)*diag;
		    }
	   }

 	/* still need to get last element in L Matrix */

	 double sum = 0.0;
	 for (int k=0; k<nm1; k++)
    sum += a(nm1,k)*a(k,nm1);
	 a(nm1,nm1) -= sum;
	 return flag;
  }


//!  This function is used to find the solution to a system of equations,
/*!   A x = b, after LU decomposition of A has been found.
*    Within this routine, the elements of b are rearranged in the same way
*    that the rows of a were interchanged, using the order vector.
*    The solution is returned in x.
*
*
*  \param  a     - the LU decomposition of the original coefficient Matrix.
*  \param  b     - the vector of right-hand sides
*  \param       x     - the solution vector
*  \param    order - integer array of row order as arranged during pivoting
*
*/
void solvlu(const cmav<double,2> &a, const cmav<double,1> &b, vmav<double,1> &x, const cmav<int,1> &order)
  {
	 int n = a.shape(0);

	 /* rearrange the elements of the b vector. x is used to hold them. */
 	for (int i=0; i<n; i++)
  		x(i) = b(order(i));

 	/* do forward substitution, replacing x vector. */
  x(0) /= a(0,0);
 	for (int i=1; i<n; i++)
    {
		  double sum = 0.0;
		  for (int j=0; j<i; j++)
      sum += a(i,j)*x(j);
	  	x(i) = (x(i)-sum)/a(i,i);
	   }

 	/* now get the solution vector, x[n-1] is already done */
 	for (int i=n-2; i>=0; i--)
    {
		  double sum = 0.0;
	  	for (int j=i+1; j<n; j++)
      sum += a(i,j) * x(j);
  		x(i) -= sum;
	   }
  }

NpArr Py_amplitude_sampling_per_pix_helper (
  const CNpArr &map_sky_,
  const CNpArr &map_rms_,
  const CNpArr &M_,
  const CNpArr &random_,
  const OptNpArr &comp_maps__,
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
  const auto random = to_cmav<double,2>(random_);
  MR_assert(random.shape(0)==npix, "random numbers shape mismatch");
  MR_assert(random.shape(1)==nband, "random numbers shape mismatch");
  auto [comp_maps_, comp_maps] = get_OptNpArr_and_vmav<double,2>(comp_maps__, {ncomp, npix}, "comp_maps");
  {
  py::gil_scoped_release release;

  execStatic(npix, nthreads, 0,[&](auto &sched)
    {
    vmav<double,1> x({ncomp}), xmap({nband}), tmap({nband}), comp({ncomp});
    vmav<int,1> order({ncomp});
    vmav<double,2> A({ncomp,ncomp});
    while (auto rng=sched.getNext())
      for (size_t it=rng.lo; it<rng.hi; ++it)
        {
        for (size_t i=0; i<nband; ++i)
          {
          xmap(i) = 1./map_rms(i,it);
          tmap(i) = xmap(i)*(xmap(i)*map_sky(i,it) + random(it,i));
          }
        for (size_t i=0; i<ncomp; ++i)
          {
          x(i) = 0;
          for (size_t j=0; j<nband; ++j)
            x(i) += M(j,i)*tmap(j);
          }
        for (size_t i=0; i<ncomp; ++i)
          for (size_t j=0; j<ncomp; ++j)
            {
            A(i,j) = 0;
            for (size_t k=0; k<nband; ++k)
              A(i,j) += M(k,i)*M(k,j)*xmap(k)*xmap(k);
            }
        try
          {
          ludcmp(A, order);
          solvlu(A, x, comp, order);
          for (size_t i=0; i<ncomp; ++i)
            comp_maps(i,it) = comp(i);
          }
        catch (SingularError)
          {
          for (size_t i=0; i<ncomp; ++i)
            comp_maps(i,it) = 0;
          }
        }
    });
  }
  return comp_maps_;
  }

constexpr const char *Py_amplitude_sampling_per_pix_helper_DS = R"""(
Helper function for amplitude_ampling_per_pix.

Parameters
----------
map_sky: numpy.ndarray((nband, npix), dtype=np.float64)
    the sky maps for all bands
map_rms: numpy.ndarray((nband, npix), dtype=np.float64)
    the RMS maps for all bands
M: numpy.ndarray((nband, ncomp), dtype=np.float64)
    the mixing matrix
random: numpy.ndarray((npix, nband), dtype=np.float64)
    normal-distributed random numbers
    NOTE: the transposed shape is only for performance reasons
comp_maps: numpy.ndarray((npix, ncomp), dtype=np.float64) or None
    If provided, the results will be stored in this array

Returns
-------
numpy.ndarray((npix, ncomp), dtype=np.float64)
    The component maps
    If comp_maps was provided, this array is identical to comp_maps
)""";

template<typename T> static NpArr Py2_huffman_decode(const CNpArr &bytes_,
  const CNpArr &tree_, const CNpArr &symb_)
  {
  auto bytes = to_cmav<uint8_t,1>(bytes_);
  auto tree = to_cmav<int64_t,1>(tree_);
  auto symb = to_cmav<T,1>(symb_);
  vector<T> out;
  {
  py::gil_scoped_release release;
  MR_assert((tree.shape(0)&1)==1, "bad tree size");
  size_t n_internal = (tree.shape(0)-1)/2;
  cmav<int64_t,2> lrnodes(&tree(1), {2, n_internal},
    {tree.stride(0)*ptrdiff_t(n_internal),tree.stride(0)});
  size_t nsymb = symb.shape(0);
  size_t startnode = nsymb + n_internal;
  size_t nbits = bytes.shape(0)*8 - 8 - bytes(0);
  size_t node = startnode;
  for (size_t i=8; i<nbits+8; ++i)
    {
    size_t bit = (bytes(i/8) >> (7-(i%8))) & 1;
    node = lrnodes(bit, node-nsymb-1);
    if (node <= nsymb)
      {
      out.push_back(symb(node-1));
      node = startnode;
      }
    }
  }
  auto [res_, res] = make_Pyarr_and_vmav<T,1>({out.size()});
  for (size_t i=0; i<out.size(); ++i)
    res(i) = out[i];
  return res_;
  }

static NpArr Py_huffman_decode(const CNpArr &bytes,
  const CNpArr &tree, const CNpArr &symb)
  {
  if (isPyarr<int64_t>(symb))
    return Py2_huffman_decode<int64_t> (bytes, tree, symb);
  MR_fail("type matching failed: 'symb' has neither type 'i8' nor 'xxx'");
  }

constexpr const char *Py_huffman_decode_DS = R"""(
Decode a Commander3-style Huffman-compressed bitstream.

Parameters
----------
bytes: numpy.ndarray((nbytes,), dtype=np.uint8)
    the bit stream
tree: numpy.ndarray((ntree,), dtype=np.int64)
    the tree array
symb: numpy.ndarray((nsymb,), dtype=np.int64 or TBD)
    the array of possible symbols in the stream

Returns
-------
numpy.ndarray(ndata,), dtype identical to that of symb)
    the uncopressed data array
)""";

void add_utils(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("utils");
//  m.doc() = utils_DS;

  m.def("amplitude_sampling_per_pix_helper",
        Py_amplitude_sampling_per_pix_helper,
        Py_amplitude_sampling_per_pix_helper_DS,
        "map_sky"_a,
        "map_rms"_a,
        "M"_a,
        "random"_a,
        "comp_maps"_a=None,
        "nthreads"_a=1);

  m.def("huffman_decode", Py_huffman_decode, Py_huffman_decode_DS, "bytes"_a,
        "tree"_a, "symb"_a);
  }

}

using detail_pymodule_utils::add_utils;

}
