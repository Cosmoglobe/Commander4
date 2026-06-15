#include <cmath>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define CMDR4_RESTRICT __restrict__
#else
#define CMDR4_RESTRICT
#endif

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884197;
constexpr double kInvHalfPi = 2.0 / kPi;
constexpr double kTwothird = 2.0 / 3.0;
constexpr double kPoleSwitch = 0.01;
constexpr double kSouthPoleSwitch = kPi - kPoleSwitch;
constexpr double kCosPoly9C1 = -1.570796299748726144;
constexpr double kCosPoly9C3 = 0.6459635104511287640;
constexpr double kCosPoly9C5 = -0.07968909439095754610;
constexpr double kCosPoly9C7 = 0.004673138064893530078;
constexpr double kCosPoly9C9 = -0.0001512642762782153870;

inline double normalize_tt(const double phi) {
	double tt = phi * kInvHalfPi;
	if ((tt >= 4.0) || (tt < 0.0)) {
		tt -= 4.0 * std::floor(tt * 0.25);
		if (tt >= 4.0) {
			tt = 0.0;
		}
	}
	return tt;
}

inline int64_t loc2pix_ring_tt(
	const double z,
	const double tt,
	const double sth,
	const bool have_sth,
	const int64_t nside
) {
	const double za = std::abs(z);
	const int64_t nl4 = 4 * nside;
	const int64_t ncap = 2 * nside * (nside - 1);
	const int64_t npix = 12 * nside * nside;

	if (za <= kTwothird) {
		const double temp1 = static_cast<double>(nside) * (0.5 + tt);
		const double temp2 = static_cast<double>(nside) * z * 0.75;
		const int64_t jp = static_cast<int64_t>(temp1 - temp2);
		const int64_t jm = static_cast<int64_t>(temp1 + temp2);
		const int64_t ir = nside + 1 + jp - jm;
		const int64_t kshift = 1 - (ir & 1LL);
		const int64_t t1 = jp + jm - nside + kshift + 1 + nl4 + nl4;
		const int64_t ip = (t1 >> 1) & (nl4 - 1);
		return ncap + (ir - 1) * nl4 + ip;
	}

	const double tp = tt - static_cast<int64_t>(tt);
	const double tmp = (have_sth && (za >= 0.99))
		? static_cast<double>(nside) * sth / std::sqrt((1.0 + za) / 3.0)
		: static_cast<double>(nside) * std::sqrt(3.0 * (1.0 - za));

	const int64_t jp = static_cast<int64_t>(tp * tmp);
	const int64_t jm = static_cast<int64_t>((1.0 - tp) * tmp);
	const int64_t ir = jp + jm + 1;
	const int64_t ip = static_cast<int64_t>(tt * static_cast<double>(ir));
	if (z > 0.0) {
		return 2 * ir * (ir - 1) + ip;
	}
	return npix - 2 * ir * (ir + 1) + ip;
}

inline int effective_threads(const int requested_threads) {
	if (requested_threads > 0) {
		return requested_threads;
	}
#ifdef _OPENMP
	return omp_get_max_threads();
#else
	return 1;
#endif
}

template<bool PhiInRange>
inline double phi_to_tt(const double phi) {
	if constexpr (PhiInRange) {
		return phi * kInvHalfPi;
	}
	return normalize_tt(phi);
}

inline double approx_cos_theta(const double angle) {
	const double u = angle * kInvHalfPi - 1.0;
	const double u2 = u * u;
	return u * (kCosPoly9C1 + u2 * (kCosPoly9C3 + u2 * (kCosPoly9C5 + u2 * (kCosPoly9C7 + u2 * kCosPoly9C9))));
}

inline double distance_to_nearest_integer(const double value) {
	const double frac = value - std::floor(value);
	return std::min(frac, 1.0 - frac);
}

// Small z errors only matter when they push the scalar formulas across an integer boundary.
inline bool poly_needs_exact_boundary_refinement_from_ztt(
	const double z,
	const double tt,
	const int64_t nside,
	const double boundary_tol
) {
	if (boundary_tol <= 0.0) {
		return false;
	}

	const double za = std::abs(z);
	const double z_boundary_tol = (4.0 * boundary_tol) / (3.0 * static_cast<double>(nside));
	if (std::abs(za - kTwothird) < z_boundary_tol) {
		return true;
	}

	if (za <= kTwothird) {
		const double temp1 = static_cast<double>(nside) * (0.5 + tt);
		const double temp2 = static_cast<double>(nside) * z * 0.75;
		return (distance_to_nearest_integer(temp1 - temp2) < boundary_tol)
			|| (distance_to_nearest_integer(temp1 + temp2) < boundary_tol);
	}

	const double tp = tt - static_cast<double>(static_cast<int64_t>(tt));
	const double tmp = static_cast<double>(nside) * std::sqrt(3.0 * (1.0 - za));
	return (distance_to_nearest_integer(tp * tmp) < boundary_tol)
		|| (distance_to_nearest_integer((1.0 - tp) * tmp) < boundary_tol);
}

// The rare pole refinement is deferred to a cleanup pass so this hot loop can vectorize.
template<bool PhiInRange, typename T>
void ang2pix_ring_theta_core(
	const T *CMDR4_RESTRICT theta,
	const T *CMDR4_RESTRICT phi,
	int64_t *CMDR4_RESTRICT out,
	const int64_t size,
	const int64_t nside,
	const int nthreads
) {
	if (nthreads <= 1) {
	#ifdef _OPENMP
		#pragma omp simd
	#endif
		for (int64_t index = 0; index < size; ++index) {
			const double angle = static_cast<double>(theta[index]);
			const double tt = phi_to_tt<PhiInRange>(static_cast<double>(phi[index]));
			out[index] = loc2pix_ring_tt(std::cos(angle), tt, 0.0, false, nside);
		}
		return;
	}

#ifdef _OPENMP
	#pragma omp parallel for simd schedule(static) num_threads(nthreads)
	for (int64_t index = 0; index < size; ++index) {
		const double angle = static_cast<double>(theta[index]);
		const double tt = phi_to_tt<PhiInRange>(static_cast<double>(phi[index]));
		out[index] = loc2pix_ring_tt(std::cos(angle), tt, 0.0, false, nside);
	}
#else
	for (int64_t index = 0; index < size; ++index) {
		const double angle = static_cast<double>(theta[index]);
		const double tt = phi_to_tt<PhiInRange>(static_cast<double>(phi[index]));
		out[index] = loc2pix_ring_tt(std::cos(angle), tt, 0.0, false, nside);
	}
#endif
}

template<bool PhiInRange, typename T>
void ang2pix_ring_theta_impl(
	const T *CMDR4_RESTRICT theta,
	const T *CMDR4_RESTRICT phi,
	int64_t *CMDR4_RESTRICT out,
	const int64_t size,
	const int64_t nside,
	const bool refine_poles,
	const int nthreads
) {
	ang2pix_ring_theta_core<PhiInRange>(theta, phi, out, size, nside, nthreads);
	if (!refine_poles) {
		return;
	}

	for (int64_t index = 0; index < size; ++index) {
		const double angle = static_cast<double>(theta[index]);
		if ((angle < kPoleSwitch) || (angle > kSouthPoleSwitch)) {
			const double tt = phi_to_tt<PhiInRange>(static_cast<double>(phi[index]));
			out[index] = loc2pix_ring_tt(std::cos(angle), tt, std::sin(angle), true, nside);
		}
	}
}

template<typename T>
void ang2pix_ring_theta_dispatch(
	const T *CMDR4_RESTRICT theta,
	const T *CMDR4_RESTRICT phi,
	int64_t *CMDR4_RESTRICT out,
	const int64_t size,
	const int64_t nside,
	const bool refine_poles,
	const bool phi_in_range,
	const int nthreads
) {
	if (phi_in_range) {
		ang2pix_ring_theta_impl<true>(theta, phi, out, size, nside, refine_poles, nthreads);
	} else {
		ang2pix_ring_theta_impl<false>(theta, phi, out, size, nside, refine_poles, nthreads);
	}
}

template<bool PhiInRange, typename T>
void ang2pix_ring_poly_core(
	const T *CMDR4_RESTRICT theta,
	const T *CMDR4_RESTRICT phi,
	int64_t *CMDR4_RESTRICT out,
	unsigned char *CMDR4_RESTRICT refine_mask,
	const int64_t size,
	const int64_t nside,
	const double boundary_tol,
	const int nthreads
) {
	const bool track_boundary_mask = (refine_mask != nullptr) && (boundary_tol > 0.0);
	if (nthreads <= 1) {
	#ifdef _OPENMP
		#pragma omp simd
	#endif
		for (int64_t index = 0; index < size; ++index) {
			const double angle = static_cast<double>(theta[index]);
			const double tt = phi_to_tt<PhiInRange>(static_cast<double>(phi[index]));
			const double z = approx_cos_theta(angle);
			out[index] = loc2pix_ring_tt(z, tt, 0.0, false, nside);
			if (track_boundary_mask) {
				refine_mask[index] = static_cast<unsigned char>(
					poly_needs_exact_boundary_refinement_from_ztt(z, tt, nside, boundary_tol)
				);
			}
		}
		return;
	}

#ifdef _OPENMP
	#pragma omp parallel for simd schedule(static) num_threads(nthreads)
	for (int64_t index = 0; index < size; ++index) {
		const double angle = static_cast<double>(theta[index]);
		const double tt = phi_to_tt<PhiInRange>(static_cast<double>(phi[index]));
		const double z = approx_cos_theta(angle);
		out[index] = loc2pix_ring_tt(z, tt, 0.0, false, nside);
		if (track_boundary_mask) {
			refine_mask[index] = static_cast<unsigned char>(
				poly_needs_exact_boundary_refinement_from_ztt(z, tt, nside, boundary_tol)
			);
		}
	}
#else
	for (int64_t index = 0; index < size; ++index) {
		const double angle = static_cast<double>(theta[index]);
		const double tt = phi_to_tt<PhiInRange>(static_cast<double>(phi[index]));
		const double z = approx_cos_theta(angle);
		out[index] = loc2pix_ring_tt(z, tt, 0.0, false, nside);
		if (track_boundary_mask) {
			refine_mask[index] = static_cast<unsigned char>(
				poly_needs_exact_boundary_refinement_from_ztt(z, tt, nside, boundary_tol)
			);
		}
	}
#endif
}

template<bool PhiInRange, typename T>
void ang2pix_ring_poly_impl(
	const T *CMDR4_RESTRICT theta,
	const T *CMDR4_RESTRICT phi,
	int64_t *CMDR4_RESTRICT out,
	const int64_t size,
	const int64_t nside,
	const double boundary_tol,
	const bool refine_poles,
	const int nthreads
) {
	std::vector<unsigned char> refine_mask;
	if (boundary_tol > 0.0) {
		refine_mask.resize(static_cast<std::size_t>(size));
	}

	ang2pix_ring_poly_core<PhiInRange>(
		theta,
		phi,
		out,
		refine_mask.empty() ? nullptr : refine_mask.data(),
		size,
		nside,
		boundary_tol,
		nthreads
	);
	if ((!refine_poles) && refine_mask.empty()) {
		return;
	}

	for (int64_t index = 0; index < size; ++index) {
		const double angle = static_cast<double>(theta[index]);
		const bool refine_pole = refine_poles && ((angle < kPoleSwitch) || (angle > kSouthPoleSwitch));
		const bool refine_boundary = (!refine_pole) && (!refine_mask.empty())
			&& (refine_mask[static_cast<std::size_t>(index)] != 0);
		if (refine_pole || refine_boundary) {
			const double tt = phi_to_tt<PhiInRange>(static_cast<double>(phi[index]));
			out[index] = loc2pix_ring_tt(
				std::cos(angle),
				tt,
				refine_pole ? std::sin(angle) : 0.0,
				refine_pole,
				nside
			);
		}
	}
}

template<typename T>
void ang2pix_ring_poly_dispatch(
	const T *CMDR4_RESTRICT theta,
	const T *CMDR4_RESTRICT phi,
	int64_t *CMDR4_RESTRICT out,
	const int64_t size,
	const int64_t nside,
	const double boundary_tol,
	const bool refine_poles,
	const bool phi_in_range,
	const int nthreads
) {
	if (phi_in_range) {
		ang2pix_ring_poly_impl<true>(theta, phi, out, size, nside, boundary_tol, refine_poles, nthreads);
	} else {
		ang2pix_ring_poly_impl<false>(theta, phi, out, size, nside, boundary_tol, refine_poles, nthreads);
	}
}

}  // namespace

extern "C" {

void ang2pix_ring_theta_f64(
	const double *theta,
	const double *phi,
	int64_t *out,
	const int64_t size,
	const int64_t nside,
	const int refine_poles,
	const int phi_in_range,
	const int nthreads
) {
	ang2pix_ring_theta_dispatch(
		theta,
		phi,
		out,
		size,
		nside,
		refine_poles != 0,
		phi_in_range != 0,
		effective_threads(nthreads)
	);
}

void ang2pix_ring_theta_f32(
	const float *theta,
	const float *phi,
	int64_t *out,
	const int64_t size,
	const int64_t nside,
	const int refine_poles,
	const int phi_in_range,
	const int nthreads
) {
	ang2pix_ring_theta_dispatch(
		theta,
		phi,
		out,
		size,
		nside,
		refine_poles != 0,
		phi_in_range != 0,
		effective_threads(nthreads)
	);
}

void ang2pix_ring_poly_f64(
	const double *theta,
	const double *phi,
	int64_t *out,
	const int64_t size,
	const int64_t nside,
	const double boundary_tol,
	const int refine_poles,
	const int phi_in_range,
	const int nthreads
) {
	ang2pix_ring_poly_dispatch(
		theta,
		phi,
		out,
		size,
		nside,
		boundary_tol,
		refine_poles != 0,
		phi_in_range != 0,
		effective_threads(nthreads)
	);
}

void ang2pix_ring_poly_f32(
	const float *theta,
	const float *phi,
	int64_t *out,
	const int64_t size,
	const int64_t nside,
	const double boundary_tol,
	const int refine_poles,
	const int phi_in_range,
	const int nthreads
) {
	ang2pix_ring_poly_dispatch(
		theta,
		phi,
		out,
		size,
		nside,
		boundary_tol,
		refine_poles != 0,
		phi_in_range != 0,
		effective_threads(nthreads)
	);
}

}
