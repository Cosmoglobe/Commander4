#include "ducc0/bindings/pybind_utils.h"  // must be the first include

#include "ducc0/infra/mav.h"

namespace cmdr4 {

namespace detail_pymodule_compression {

using namespace std;
using namespace ducc0;
// Note that this script inherits (from ducc0) py:: as the nanobind namespace.

/** Decodes the bitstream into out, for symbols (and out) of integer type T. */
template<typename T> static void huffman_decode_T(const CNpArr &bytes_,
  const CNpArr &tree_, const CNpArr &symb_, const NpArr &out_, const char *dtype_descr)
  {
  MR_assert(isPyarr<T>(out_), "type mismatch: 'out' must have the same dtype as 'symb' (",
    dtype_descr, ")");
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
  }

static NpArr Py_huffman_decode(const CNpArr &bytes,
  const CNpArr &tree, const CNpArr &symb, const NpArr &out)
  {
  if (isPyarr<int8_t>(symb))
    huffman_decode_T<int8_t>(bytes, tree, symb, out, "i1");
  else if (isPyarr<uint8_t>(symb))
    huffman_decode_T<uint8_t>(bytes, tree, symb, out, "u1");
  else if (isPyarr<int16_t>(symb))
    huffman_decode_T<int16_t>(bytes, tree, symb, out, "i2");
  else if (isPyarr<uint16_t>(symb))
    huffman_decode_T<uint16_t>(bytes, tree, symb, out, "u2");
  else if (isPyarr<int32_t>(symb))
    huffman_decode_T<int32_t>(bytes, tree, symb, out, "i4");
  else if (isPyarr<uint32_t>(symb))
    huffman_decode_T<uint32_t>(bytes, tree, symb, out, "u4");
  else if (isPyarr<int64_t>(symb))
    huffman_decode_T<int64_t>(bytes, tree, symb, out, "i8");
  else if (isPyarr<uint64_t>(symb))
    huffman_decode_T<uint64_t>(bytes, tree, symb, out, "u8");
  else
    MR_fail("type mismatch: 'symb' must have an integer dtype among 'i1', 'u1', 'i2', 'u2', ",
            "'i4', 'u4', 'i8', or 'u8'");
  return out;
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

void add_compression(py::module_ &msup)
  {
  using namespace py::literals;
  auto m = msup.def_submodule("compression");

  m.def("huffman_decode", Py_huffman_decode, Py_huffman_decode_DS, "bytes"_a,
        "tree"_a, "symb"_a, "out"_a);
  }

}  // ends `namespace detail_pymodule_compression`

using detail_pymodule_compression::add_compression;

}  // ends `namespace cmdr4`
