#include "ducc0/bindings/pybind_utils.h"  // must be the first include

#include <cmath>
#include <limits>

namespace cmdr4 {

namespace detail_pymodule_mapmaker {

using namespace std;
using namespace ducc0;

// All maps are float64: a pixel can collect millions of samples, and a float32 running sum loses
// precision long before that. The TOD may be float32 or float64; each sample is used as a double.
// Maps have shape (3, npix) with rows I, Q, U, or (1, npix) for intensity alone, and the per-pixel
// symmetric 3x3 matrices (6, npix), stored as their 6 unique elements (II, IQ, IU, QQ, QU, UU).
// A sample's pointing row is [response_I, response_P*cos(2 psi), response_P*sin(2 psi)], or
// [response_I] for an intensity map. When only intensity is measured, psi is never read.
// Every pixel index is checked to lie in [0, npix), since an index outside the map would silently
// read or write other memory. Casting to size_t turns a negative index into a huge one, so a single
// comparison covers both ends. The check only reads pix, which the loop reads anyway, so it adds
// no memory traffic and its cost was not measurable even with all cores of a node busy.

/** Adds one hit per sample into the hit map: hits[pix[i]] += 1. */
void Py_hit_accumulator(const NpArr &hits_, const CNpArr &pix_){
    auto hits = to_vmav<double,1>(hits_, "hits");
    auto pix = to_cmav<int64_t,1>(pix_, "pix");
    const size_t npix = hits.shape(0);
    for (size_t i = 0; i < pix.shape(0); i++){
        const int64_t p = pix(i);
        MR_assert(size_t(p) < npix, "pixel index out of range");
        hits(p) += 1.0;
    }
}


/** Accumulates one detector-scan into all binned maps of a band at once.
 *
 * Adds the normal matrix P^T N^-1 P (LHS of Eq. 77 in BP01) into weights, and the right-hand side
 * P^T N^-1 d of each TOD row tods[j] into maps[j], all with the same per-sample weight. Doing all
 * maps in one pass reads the pointing once and touches each pixel once per sample, instead of
 * once per map. For an IQU band each sample's pointing row is
 * [response_I, response_P*cos(2 psi), response_P*sin(2 psi)], weights is (6, npix) and maps is
 * (nmaps, 3, npix). An I-only band has the pointing row [response_I], weights (1, npix) and maps
 * (nmaps, 1, npix), and psi is never read.
 */
template<typename Ttod>
void binned_map_accumulator_T(const NpArr &weights_, const NpArr &maps_, const CNpArr &tods_,
                              double weight, const CNpArr &pix_, const CNpArr &psi_,
                              double response_I, double response_P){
    auto weights = to_vmav<double,2>(weights_, "weights");
    auto maps = to_vmav<double,3>(maps_, "maps");
    auto tods = to_cmav<Ttod,2>(tods_, "tods");
    auto pix = to_cmav<int64_t,1>(pix_, "pix");
    const size_t ntod = pix.shape(0), npix = weights.shape(1), nmaps = maps.shape(0);
    const bool intensity_band = weights.shape(0) == 1;
    MR_assert(intensity_band || weights.shape(0) == 6,
              "weights must have shape (6, npix) or (1, npix)");
    MR_assert(maps.shape(1) == (intensity_band ? 1 : 3) && maps.shape(2) == npix,
              "maps must have shape (nmaps, 3, npix), or (nmaps, 1, npix) with (1, npix) weights");
    MR_assert(tods.shape(0) == nmaps && tods.shape(1) == ntod,
              "tods must have shape (nmaps, ntod)");

    const double weight_I = weight * response_I;
    const double weight_P = weight * response_P;
    const double weight_II = weight * response_I * response_I;
    const double weight_IP = weight * response_I * response_P;
    const double weight_PP = weight * response_P * response_P;
    if (response_I == 0.0 && response_P == 0.0) return;
    if (intensity_band || response_P == 0.0){
        // Only intensity is measured (an I-only band, or a detector without polarization response),
        // so only I and II are non-zero, and psi is never read.
        for (size_t i = 0; i < ntod; i++){
            const int64_t p = pix(i);
            MR_assert(size_t(p) < npix, "pixel index out of range");
            weights(0, p) += weight_II;
            for (size_t j = 0; j < nmaps; j++)
                maps(j, 0, p) += weight_I * tods(j, i);
        }
        return;
    }
    auto psi = to_cmav<double,1>(psi_, "psi");
    MR_assert(psi.shape(0) == ntod, "pix and psi lengths differ");
    for (size_t i = 0; i < ntod; i++){
        // Read the sample into locals before writing: the compiler cannot rule out that a write
        // to the maps changes psi or tods, so it would re-read them and compute cos and sin again.
        const int64_t p = pix(i);
        MR_assert(size_t(p) < npix, "pixel index out of range");
        const double cos2psi = std::cos(2.0 * psi(i));
        const double sin2psi = std::sin(2.0 * psi(i));
        weights(0, p) += weight_II;                      // II
        weights(1, p) += weight_IP * cos2psi;            // IQ
        weights(2, p) += weight_IP * sin2psi;            // IU
        weights(3, p) += weight_PP * cos2psi * cos2psi;  // QQ
        weights(4, p) += weight_PP * sin2psi * cos2psi;  // QU
        weights(5, p) += weight_PP * sin2psi * sin2psi;  // UU
        for (size_t j = 0; j < nmaps; j++){
            const double d = tods(j, i);
            maps(j, 0, p) += weight_I * d;
            maps(j, 1, p) += weight_P * d * cos2psi;
            maps(j, 2, p) += weight_P * d * sin2psi;
        }
    }
}

void Py_binned_map_accumulator(const NpArr &weights, const NpArr &maps, const CNpArr &tods,
                               double weight, const CNpArr &pix, const CNpArr &psi,
                               double response_I, double response_P){
    if (isPyarr<float>(tods))
        binned_map_accumulator_T<float>(weights, maps, tods, weight, pix, psi,
                                        response_I, response_P);
    else if (isPyarr<double>(tods))
        binned_map_accumulator_T<double>(weights, maps, tods, weight, pix, psi,
                                         response_I, response_P);
    else
        MR_fail("tods must be float32 or float64");
}


/** Reads a map along the pointing (the pointing matrix P): tod = P map.
 *
 * map is (3, npix) or (1, npix); see the pointing rows at the top of this file.
 */
void Py_map2tod(const CNpArr &map_, const NpArr &tod_, const CNpArr &pix_, const CNpArr &psi_,
                double response_I, double response_P){
    auto map = to_cmav<double,2>(map_, "map");
    auto tod = to_vmav<double,1>(tod_, "tod");
    auto pix = to_cmav<int64_t,1>(pix_, "pix");
    const size_t ntod = tod.shape(0), npix = map.shape(1);
    MR_assert(map.shape(0) == 1 || map.shape(0) == 3, "map must have shape (3, npix) or (1, npix)");
    MR_assert(pix.shape(0) == ntod, "tod and pix lengths differ");
    if (map.shape(0) == 1 || response_P == 0.0){
        // Only intensity is measured, so psi is never read.
        for (size_t i = 0; i < ntod; i++){
            const int64_t p = pix(i);
            MR_assert(size_t(p) < npix, "pixel index out of range");
            tod(i) = response_I * map(0, p);
        }
        return;
    }
    auto psi = to_cmav<double,1>(psi_, "psi");
    MR_assert(psi.shape(0) == ntod, "tod and psi lengths differ");
    for (size_t i = 0; i < ntod; i++){
        const int64_t p = pix(i);
        MR_assert(size_t(p) < npix, "pixel index out of range");
        tod(i) = response_I * map(0, p)
               + response_P * map(1, p) * std::cos(2.0 * psi(i))
               + response_P * map(2, p) * std::sin(2.0 * psi(i));
    }
}


/** Adds a TOD into a map along the pointing (the transpose of map2tod): map += P^T tod.
 *
 * map is (3, npix) or (1, npix), as for map2tod. Unlike binned_map_accumulator this applies no
 * weight: the CG mapmaker weights the TOD itself (N^-1) before calling it.
 */
template<typename Ttod>
void tod2map_T(const NpArr &map_, const CNpArr &tod_, const CNpArr &pix_, const CNpArr &psi_,
               double response_I, double response_P){
    auto map = to_vmav<double,2>(map_, "map");
    auto tod = to_cmav<Ttod,1>(tod_, "tod");
    auto pix = to_cmav<int64_t,1>(pix_, "pix");
    const size_t ntod = tod.shape(0), npix = map.shape(1);
    MR_assert(map.shape(0) == 1 || map.shape(0) == 3, "map must have shape (3, npix) or (1, npix)");
    MR_assert(pix.shape(0) == ntod, "tod and pix lengths differ");
    if (map.shape(0) == 1 || response_P == 0.0){
        // Only intensity is measured, so psi is never read.
        for (size_t i = 0; i < ntod; i++){
            const int64_t p = pix(i);
            MR_assert(size_t(p) < npix, "pixel index out of range");
            map(0, p) += response_I * tod(i);
        }
        return;
    }
    auto psi = to_cmav<double,1>(psi_, "psi");
    MR_assert(psi.shape(0) == ntod, "tod and psi lengths differ");
    for (size_t i = 0; i < ntod; i++){
        // Read the sample into locals before writing: the compiler cannot rule out that a write
        // to map changes psi or tod, so it would re-read them and compute cos and sin separately.
        const int64_t p = pix(i);
        MR_assert(size_t(p) < npix, "pixel index out of range");
        const double d = tod(i);
        const double cos2psi = std::cos(2.0 * psi(i));
        const double sin2psi = std::sin(2.0 * psi(i));
        map(0, p) += response_I * d;
        map(1, p) += response_P * d * cos2psi;
        map(2, p) += response_P * d * sin2psi;
    }
}

void Py_tod2map(const NpArr &map, const CNpArr &tod, const CNpArr &pix, const CNpArr &psi,
                double response_I, double response_P){
    if (isPyarr<float>(tod))
        tod2map_T<float>(map, tod, pix, psi, response_I, response_P);
    else if (isPyarr<double>(tod))
        tod2map_T<double>(map, tod, pix, psi, response_I, response_P);
    else
        MR_fail("tod must be float32 or float64");
}


/** Inverts a symmetric positive definite 3x3 matrix, given and returned as its 6 unique elements.
 *
 * Returns false (leaving the outputs unset) if the matrix is singular or ill-conditioned.
 */
inline bool invert_SPD_3x3(const double a00, const double a01, const double a02,
                           const double a11, const double a12, const double a22,
                           double &inv00, double &inv01, double &inv02,
                           double &inv11, double &inv12, double &inv22){
    const double det = a00 * (a11 * a22 - a12 * a12)
                     - a01 * (a01 * a22 - a02 * a12)
                     + a02 * (a01 * a12 - a02 * a11);

    const double diag_prod = a00 * a11 * a22;
    // If the diagonal product is zero, we are singular, or matrix is not actually SPD.
    if (diag_prod <= std::numeric_limits<double>::min()) return false;

    // Heuristic for Condition Number: det / (a00 * a11 * a22) roughly approximates
    // 1/condition_number assuming the matrix is positive definite.
    // If det is < 1e-12 of the diagonal product, the matrix is ill-conditioned.
    if (det <= 1e-12 * diag_prod) return false;

    // Directly calculate the elements of the inverse of A:
    const double inv_det = 1.0 / det;
    inv00 = (a11 * a22 - a12 * a12) * inv_det;
    inv01 = (a02 * a12 - a01 * a22) * inv_det;
    inv02 = (a01 * a12 - a02 * a11) * inv_det;
    inv11 = (a00 * a22 - a02 * a02) * inv_det;
    inv12 = (a01 * a02 - a00 * a12) * inv_det;
    inv22 = (a00 * a11 - a01 * a01) * inv_det;
    return true;
}


/** Solves the per-pixel 3x3 IQU system norm_map x = map_rhs into map_out (3, npix).
 *
 * Singular or ill-conditioned pixels are set to zero.
 */
void Py_map_solve_IQU(const NpArr &map_out_, const CNpArr &map_rhs_, const CNpArr &norm_map_){
    auto map_out = to_vmav<double,2>(map_out_, "map_out");
    auto map_rhs = to_cmav<double,2>(map_rhs_, "map_rhs");
    auto norm_map = to_cmav<double,2>(norm_map_, "norm_map");
    const size_t num_pix = map_out.shape(1);
    MR_assert(map_out.shape(0)==3 && map_rhs.shape(0)==3 && norm_map.shape(0)==6,
              "need shapes (3,npix), (3,npix), (6,npix)");
    MR_assert(map_rhs.shape(1)==num_pix && norm_map.shape(1)==num_pix, "pixel counts differ");

    for (size_t ipix = 0; ipix < num_pix; ipix++){
        double inv00, inv01, inv02, inv11, inv12, inv22;
        if (!invert_SPD_3x3(norm_map(0, ipix), norm_map(1, ipix), norm_map(2, ipix),
                            norm_map(3, ipix), norm_map(4, ipix), norm_map(5, ipix),
                            inv00, inv01, inv02, inv11, inv12, inv22)){
            map_out(0, ipix) = 0.0;
            map_out(1, ipix) = 0.0;
            map_out(2, ipix) = 0.0;
            continue;
        }
        const double b0 = map_rhs(0, ipix);
        const double b1 = map_rhs(1, ipix);
        const double b2 = map_rhs(2, ipix);
        map_out(0, ipix) = inv00 * b0 + inv01 * b1 + inv02 * b2;
        map_out(1, ipix) = inv01 * b0 + inv11 * b1 + inv12 * b2;
        map_out(2, ipix) = inv02 * b0 + inv12 * b1 + inv22 * b2;
    }
}


/** Computes the inverse noise variance of I, Q and U, 1/diag(A^-1), into inv_var_out (3, npix).
 *
 * The variance of one Stokes parameter, with the other two fitted at the same time, is the matching
 * diagonal element of A^-1. Singular or ill-conditioned pixels get zero inverse variance.
 */
void Py_map_inv_var_IQU(const NpArr &inv_var_out_, const CNpArr &norm_map_){
    auto inv_var_out = to_vmav<double,2>(inv_var_out_, "inv_var_out");
    auto norm_map = to_cmav<double,2>(norm_map_, "norm_map");
    const size_t num_pix = inv_var_out.shape(1);
    MR_assert(inv_var_out.shape(0)==3 && norm_map.shape(0)==6, "need shapes (3,npix) and (6,npix)");
    MR_assert(norm_map.shape(1)==num_pix, "pixel counts differ");

    for (size_t ipix = 0; ipix < num_pix; ipix++){
        double inv00, inv01, inv02, inv11, inv12, inv22;
        if (!invert_SPD_3x3(norm_map(0, ipix), norm_map(1, ipix), norm_map(2, ipix),
                            norm_map(3, ipix), norm_map(4, ipix), norm_map(5, ipix),
                            inv00, inv01, inv02, inv11, inv12, inv22)){
            inv_var_out(0, ipix) = 0.0;
            inv_var_out(1, ipix) = 0.0;
            inv_var_out(2, ipix) = 0.0;
            continue;
        }
        // invert_SPD_3x3 does not fully test for positive definiteness, so guard against a
        // negative variance.
        inv_var_out(0, ipix) = inv00 > 0.0 ? 1.0 / inv00 : 0.0;
        inv_var_out(1, ipix) = inv11 > 0.0 ? 1.0 / inv11 : 0.0;
        inv_var_out(2, ipix) = inv22 > 0.0 ? 1.0 / inv22 : 0.0;
    }
}


/** Inverts each pixel's symmetric 3x3 matrix into inv_out (6, npix), as its 6 unique elements.
 *
 * Used for the block preconditioner of the CG mapmaker. Singular or ill-conditioned pixels get an
 * all-zero inverse, by the same test as map_solve_IQU, so both mapmakers drop the same pixels.
 */
void Py_map_invert_IQU(const NpArr &inv_out_, const CNpArr &norm_map_){
    auto inv_out = to_vmav<double,2>(inv_out_, "inv_out");
    auto norm_map = to_cmav<double,2>(norm_map_, "norm_map");
    const size_t num_pix = inv_out.shape(1);
    MR_assert(inv_out.shape(0)==6 && norm_map.shape(0)==6, "need shapes (6,npix) and (6,npix)");
    MR_assert(norm_map.shape(1)==num_pix, "pixel counts differ");

    for (size_t ipix = 0; ipix < num_pix; ipix++){
        // invert_SPD_3x3 leaves these at zero when it cannot invert the matrix.
        double inv00 = 0.0, inv01 = 0.0, inv02 = 0.0, inv11 = 0.0, inv12 = 0.0, inv22 = 0.0;
        invert_SPD_3x3(norm_map(0, ipix), norm_map(1, ipix), norm_map(2, ipix),
                       norm_map(3, ipix), norm_map(4, ipix), norm_map(5, ipix),
                       inv00, inv01, inv02, inv11, inv12, inv22);
        inv_out(0, ipix) = inv00;
        inv_out(1, ipix) = inv01;
        inv_out(2, ipix) = inv02;
        inv_out(3, ipix) = inv11;
        inv_out(4, ipix) = inv12;
        inv_out(5, ipix) = inv22;
    }
}


/** Multiplies each pixel's IQU vector by its symmetric 3x3 matrix: map_out = inv_N_map map_in.
 *
 * Used by the block preconditioner of the CG mapmaker.
 */
void Py_apply_invN_to_map_IQU(const CNpArr &map_in_, const NpArr &map_out_,
                              const CNpArr &inv_N_map_){
    auto map_in = to_cmav<double,2>(map_in_, "map_in");
    auto map_out = to_vmav<double,2>(map_out_, "map_out");
    auto inv_N = to_cmav<double,2>(inv_N_map_, "inv_N_map");
    const size_t num_pix = map_out.shape(1);
    MR_assert(map_in.shape(0)==3 && map_out.shape(0)==3 && inv_N.shape(0)==6,
              "need shapes (3,npix), (3,npix), (6,npix)");
    MR_assert(map_in.shape(1)==num_pix && inv_N.shape(1)==num_pix, "pixel counts differ");

    for (size_t ipix = 0; ipix < num_pix; ipix++){
        const double I = map_in(0, ipix), Q = map_in(1, ipix), U = map_in(2, ipix);
        map_out(0, ipix) = inv_N(0, ipix) * I + inv_N(1, ipix) * Q + inv_N(2, ipix) * U;
        map_out(1, ipix) = inv_N(1, ipix) * I + inv_N(3, ipix) * Q + inv_N(4, ipix) * U;
        map_out(2, ipix) = inv_N(2, ipix) * I + inv_N(4, ipix) * Q + inv_N(5, ipix) * U;
    }
}


// Hash table from global pixel index to a rank's local pixel index (used by PixelDomain). It is
// open addressing with linear probing in one power-of-two array of uint64 slots, at least four
// times the number of local pixels. Each slot holds the global pixel in its high 32 bits and the
// local index in its low 32 bits, so a lookup reads one 8-byte word and usually one cache line. An
// empty slot has all bits set. Global pixels must lie below 2^32 - 1, i.e. nside <= 16384.
// A pixel is stored in the first free slot at or after its hash slot (pixel_hash_slot), so a lookup
// starts at the hash slot and steps forward until it finds the pixel. The table is at most a
// quarter full, so the first slot is nearly always the right one.
constexpr uint64_t EMPTY_SLOT = ~uint64_t(0);

/** The table slot where the search for global pixel `pix` starts, in a table of 2^bits slots.
 *
 * Fibonacci hashing: multiply by 2^64 divided by the golden ratio and keep the top `bits` bits.
 * This spreads runs of neighbouring pixel numbers evenly over the table.
 */
inline size_t pixel_hash_slot(uint64_t pix, int bits){
    return size_t((pix * 0x9E3779B97F4A7C15ULL) >> (64 - bits));
}

/** Builds the hash table that maps local_pix[j] to j. */
NpArr Py_build_pixel_hash(const CNpArr &local_pix_){
    auto local_pix = to_cmav<int64_t,1>(local_pix_, "local_pix");
    const size_t nlocal = local_pix.shape(0);
    MR_assert(nlocal < (size_t(1) << 32), "too many local pixels for 32-bit local indices");
    int bits = 2;
    while ((size_t(1) << bits) < 4*nlocal) bits++;
    const size_t mask = (size_t(1) << bits) - 1;
    auto table_ = make_Pyarr<uint64_t>({mask + 1});
    auto table = to_vmav<uint64_t,1>(table_);
    for (size_t slot = 0; slot <= mask; slot++)
        table(slot) = EMPTY_SLOT;
    for (size_t j = 0; j < nlocal; j++){
        const uint64_t p = uint64_t(local_pix(j));  // a negative index becomes huge and fails below
        MR_assert(p < (EMPTY_SLOT >> 32), "global pixel index out of range");
        // Step forward from the hash slot to the first free slot; `& mask` wraps around at the end.
        size_t slot = pixel_hash_slot(p, bits);
        while (table(slot) != EMPTY_SLOT){
            MR_assert((table(slot) >> 32) != p, "duplicate pixel in local_pix");
            slot = (slot + 1) & mask;
        }
        table(slot) = (p << 32) | uint64_t(j);  // global pixel in the high half, local in the low
    }
    return table_;
}

/** Looks up the local index of every global pixel in pix; fails if one is not in the table. */
NpArr Py_global_to_local(const CNpArr &table_, const CNpArr &pix_){
    auto table = to_cmav<uint64_t,1>(table_, "table");
    auto pix = to_cmav<int64_t,1>(pix_, "pix");
    const size_t mask = table.shape(0) - 1;
    int bits = 0;
    while ((size_t(1) << bits) < table.shape(0)) bits++;
    auto out_ = make_Pyarr<int64_t>({pix.shape(0)});
    auto out = to_vmav<int64_t,1>(out_);
    for (size_t i = 0; i < pix.shape(0); i++){
        const uint64_t p = uint64_t(pix(i));
        MR_assert(p < (EMPTY_SLOT >> 32), "global pixel index out of range");
        // Step forward from the hash slot until the slot holding p. Reaching an empty slot means p
        // was never inserted.
        size_t slot = pixel_hash_slot(p, bits);
        while ((table(slot) >> 32) != p){
            MR_assert(table(slot) != EMPTY_SLOT, "pixel not in the local pixel domain");
            slot = (slot + 1) & mask;
        }
        out(i) = int64_t(table(slot) & 0xFFFFFFFFu);  // the low half is the local index
    }
    return out_;
}


void add_mapmaker(py::module_ &msup)
  {
  using namespace py::literals;
  auto m = msup.def_submodule("mapmaker");
  m.doc() = "Compiled mapmaking kernels. Maps are float64; the TOD may be float32 or float64.";

  m.def("hit_accumulator", Py_hit_accumulator,
        "Add one hit per sample into the hit map (npix,): `hits[pix] += 1`.",
        "hits"_a, "pix"_a);
  m.def("binned_map_accumulator", Py_binned_map_accumulator,
        "Add one scan to all binned maps of a band in one pass: the normal matrix P^T N^-1 P into "
        "weights (6, npix), and P^T N^-1 d of each TOD row into maps (nmaps, 3, npix). An I-only "
        "band uses weights (1, npix) and maps (nmaps, 1, npix).",
        "weights"_a, "maps"_a, "tods"_a, "weight"_a, "pix"_a, "psi"_a, "response_I"_a=1.0,
        "response_P"_a=1.0);
  m.def("map2tod", Py_map2tod,
        "Read a map (3, npix) or (1, npix) along the pointing into tod: `tod = P map`, where a "
        "sample's pointing row is [response_I, response_P cos(2 psi), response_P sin(2 psi)], or "
        "[response_I] for a (1, npix) map.",
        "map"_a, "tod"_a, "pix"_a, "psi"_a, "response_I"_a=1.0, "response_P"_a=1.0);
  m.def("tod2map", Py_tod2map,
        "Add tod into a map (3, npix) or (1, npix) along the pointing, the transpose of map2tod: "
        "`map += P^T tod`.",
        "map"_a, "tod"_a, "pix"_a, "psi"_a, "response_I"_a=1.0, "response_P"_a=1.0);
  m.def("map_solve_IQU", Py_map_solve_IQU,
        "Solve the per-pixel 3x3 IQU system into map_out (3, npix); unsolvable pixels get 0.",
        "map_out"_a, "map_rhs"_a, "norm_map"_a);
  m.def("map_inv_var_IQU", Py_map_inv_var_IQU,
        "Write the inverse variance `1/diag(A^-1)` per pixel into inv_var_out (3, npix); "
        "unsolvable pixels get 0.",
        "inv_var_out"_a, "norm_map"_a);
  m.def("map_invert_IQU", Py_map_invert_IQU,
        "Write each pixel's inverse 3x3 matrix into inv_out (6, npix), as its 6 unique elements; "
        "unsolvable pixels get 0.",
        "inv_out"_a, "norm_map"_a);
  m.def("apply_invN_to_map_IQU", Py_apply_invN_to_map_IQU,
        "Multiply each pixel's IQU vector by its 3x3 matrix: `map_out = inv_N_map map_in`.",
        "map_in"_a, "map_out"_a, "inv_N_map"_a);
  m.def("build_pixel_hash", Py_build_pixel_hash,
        "Build the hash table (a uint64 array) that maps global pixel `local_pix[j]` to `j`.",
        "local_pix"_a);
  m.def("global_to_local", Py_global_to_local,
        "Return the local index (int64) of every global pixel in pix, using a table from "
        "`build_pixel_hash`. Fails if a pixel is not in the table.",
        "table"_a, "pix"_a);
  }


}  // ends `namespace detail_pymodule_mapmaker`

using detail_pymodule_mapmaker::add_mapmaker;

}  // ends `namespace cmdr4`
