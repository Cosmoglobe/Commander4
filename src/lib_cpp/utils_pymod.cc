#include "ducc0/bindings/pybind_utils.h"  // must be the first include

#include "ducc0/infra/mav.h"

namespace cmdr4 {

namespace detail_pymodule_utils {

using namespace std;
using namespace ducc0;
// Note that this script inherits (from ducc0) py:: as the nanobind namespace.

template<typename T> static NpArr Py2_huffman_decode(const CNpArr &bytes_,
  const CNpArr &tree_, const CNpArr &symb_, const NpArr &out_)
  {
  auto bytes = to_cmav<uint8_t,1>(bytes_);
  auto tree = to_cmav<int64_t,1>(tree_);
  auto symb = to_cmav<T,1>(symb_);
  auto out = to_vmav<T,1>(out_);
  {
  py::gil_scoped_release release;
  MR_assert((tree.shape(0)&1)==1, "bad tree size");
  size_t n_internal = (tree.shape(0)-1)/2;
  cmav<int64_t,2> lrnodes(&tree(1), {2, n_internal},
    {tree.stride(0)*ptrdiff_t(n_internal),tree.stride(0)});
  size_t nsymb = symb.shape(0);
  size_t nout = out.shape(0);
  size_t startnode = nsymb + n_internal;
  size_t nbits = bytes.shape(0)*8 - 8 - bytes(0);
  size_t node = startnode;
  size_t pos=0;
  for (size_t i=8; i<nbits+8; ++i)
    {
    size_t bit = (bytes(i/8) >> (7-(i%8))) & 1;
    node = lrnodes(bit, node-nsymb-1);
    if (node <= nsymb)
      {
      MR_assert(pos<nout, "overflow");
      out(pos) = symb(node-1);
      ++pos;
      node = startnode;
      }
    }
  MR_assert(pos==nout, "out array is too large");
  }
  return out_;
  }

template<typename T> static NpArr Py3_huffman_decode(const CNpArr &bytes,
  const CNpArr &tree, const CNpArr &symb, const NpArr &out,
  const char *dtype_descr)
  {
  MR_assert(isPyarr<T>(out), "type mismatch: 'out' must have the same dtype as 'symb' (",
    dtype_descr, ")");
  return Py2_huffman_decode<T>(bytes, tree, symb, out);
  }

static NpArr Py_huffman_decode(const CNpArr &bytes,
  const CNpArr &tree, const CNpArr &symb, const NpArr &out)
  {
  if (isPyarr<int8_t>(symb))
    return Py3_huffman_decode<int8_t>(bytes, tree, symb, out, "i1");
  if (isPyarr<uint8_t>(symb))
    return Py3_huffman_decode<uint8_t>(bytes, tree, symb, out, "u1");
  if (isPyarr<int16_t>(symb))
    return Py3_huffman_decode<int16_t>(bytes, tree, symb, out, "i2");
  if (isPyarr<uint16_t>(symb))
    return Py3_huffman_decode<uint16_t>(bytes, tree, symb, out, "u2");
  if (isPyarr<int32_t>(symb))
    return Py3_huffman_decode<int32_t>(bytes, tree, symb, out, "i4");
  if (isPyarr<uint32_t>(symb))
    return Py3_huffman_decode<uint32_t>(bytes, tree, symb, out, "u4");
  if (isPyarr<int64_t>(symb))
    return Py3_huffman_decode<int64_t>(bytes, tree, symb, out, "i8");
  if (isPyarr<uint64_t>(symb))
    return Py3_huffman_decode<uint64_t>(bytes, tree, symb, out, "u8");
  MR_fail("type mismatch: 'symb' must have an integer dtype among 'i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', or 'u8'");
  }

constexpr const char *Py_huffman_decode_DS = R"""(
Decode a Commander3-style Huffman-compressed bitstream.

Parameters
----------
bytes: numpy.ndarray((nbytes,), dtype=np.uint8)
    the bit stream
tree: numpy.ndarray((ntree,), dtype=np.int64)
    the tree array
symb: numpy.ndarray((nsymb,), dtype any signed or unsigned 8/16/32/64-bit integer type)
    the array of possible symbols in the stream
out: numpy.ndarray((ndata,), dtype identical to that of symb)
    the array into which the uncopressed data is written
  The size of this array *must* match the number of decoded symbols!
  The dtype of this array *must* be identical to that of symb.

Returns
-------
numpy.ndarray(ndata,), dtype identical to that of symb)
    the uncopressed data array, identical to `out`
)""";

void add_utils(py::module_ &msup)
  {
  using namespace py::literals;
  auto m = msup.def_submodule("utils");

  m.def("huffman_decode", Py_huffman_decode, Py_huffman_decode_DS, "bytes"_a,
        "tree"_a, "symb"_a, "out"_a);
  }

}  // ends `namespace detail_pymodule_utils`

using detail_pymodule_utils::add_utils;

}  // ends `namespace cmdr4`
