// Compile as:
// g++ -shared -O3 -fPIC mapmaker.cpp -o mapmaker.so
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>
#include <limits>

/** Simple serial mapmaker which accumulates the values of the TOD, weigthed by specified value, into a map, but leaves the map
 *  unnormalized (with respect to the weights), meaning it is supposed to be called multiple times on the same map, and the normalized.
 * 
 *  Args:
 *      map (OUTPUT) -- 1D array of length 'num_pix', representing the signal map, which will be populated by this function.
 *      tod -- 1D array, containing the TOD of length 'scan_len'.
 *      weight -- scalar, representing the weights to apply to the TOD.
 *      pix -- 1D array, containing the pixel pointing index of each element in tod.
 *      scan_len -- Length of the scan as an int.
 *      num_pix -- Number of pixels in map and map_rms.
 */
template<typename T>
void _map_accumulator_T(T *map, const T *tod, const T weight, int64_t *pix, int64_t scan_len){
    for(int64_t itod=0; itod<scan_len; itod++){
        map[pix[itod]] += tod[itod] * weight;
    }
}


/** Simple serial mapmaker accumulating the weights (typically inverse-variance weights) for the above "map_accumulator".
 * 
 *  Args:
 *      map (OUTPUT) -- 1D array of length 'num_pix', representing the accumulated weight map.
 *      weight -- scalar, representing the weight to bin up.
 *      pix -- 1D array, containing the pixel pointing index of each element in tod.
 *      scan_len -- Length of the scan as an int.
 *      num_pix -- Number of pixels in map and map_rms.
 */
template<typename T>
void _map_weight_accumulator_T(T *map, const T weight, int64_t *pix, int64_t scan_len){
    for(int64_t itod=0; itod<scan_len; itod++){
        map[pix[itod]] += weight;
    }
}


/** Simple serial mapmaker which accumulates the values of the TOD, weigthed by specified value, into a map, but leaves the map
 *  unnormalized (with respect to the weights), meaning it is supposed to be called multiple times on the same map, and then normalized.
 * 
 *  Args:
 *      map (OUTPUT) -- 1D array of length 'num_pix', representing the signal map, which will be populated by this function.
 *      tod -- 1D array, containing the TOD of length 'scan_len'.
 *      weight -- scalar, representing the weights to apply to the TOD.
 *      pix -- 1D array, containing the pixel pointing index of each element in tod.
 *      scan_len -- Length of the scan as an int.
 *      num_pix -- Number of pixels in map and map_rms.
 *      Note: this is RHS of Eqn. (77) in BP01
 */
template<typename T>
void _map_accumulator_IQU_T(T *map, const T *tod, const T weight, int64_t *pix, const double *psi, int64_t scan_len, int64_t num_pix){
    for(int64_t itod=0; itod<scan_len; itod++){
        const T cos2psi = static_cast<T>(std::cos(2.0 * psi[itod]));
        const T sin2psi = static_cast<T>(std::sin(2.0 * psi[itod]));
        map[pix[itod]]             += tod[itod] * weight;                // I
        map[pix[itod] +   num_pix] += tod[itod] * cos2psi * weight;       // Q
        map[pix[itod] + 2*num_pix] += tod[itod] * sin2psi * weight;       // U
    }
}

/** Simple serial transpose of the mapmaker operator, which accumulates the values on the TOD, given a certain pointing on a map and angle psi.
 * 
 *  Notes: 
 *      - the mapmaking operator is not unitary, i.e. P^T != P^-1. Meaning that applying thi function on the output of _map_accumulator_IQU_T will
 *          not give the original TODs.
 *      - no weights are considered here.
 * 
 *  Args:
 *      map (OUTPUT) -- 1D array of length 'num_pix', representing the signal map, which will be populated by this function.
 *      tod -- 1D array, containing the TOD of length 'scan_len'.
 *      pix -- 1D array, containing the pixel pointing index of each element in tod.
 *      scan_len -- Length of the scan as an int.
 *      num_pix -- Number of pixels in map.
 */
template<typename T>
void _map2tod_IQU_T(T *tod, const T *map, int64_t *pix, const double *psi, int64_t scan_len, int64_t num_pix){
    for(int64_t itod=0; itod<scan_len; itod++){
        const T cos2psi = static_cast<T>(std::cos(2.0 * psi[itod]));
        const T sin2psi = static_cast<T>(std::sin(2.0 * psi[itod]));
        tod[itod] = map[pix[itod]]                      //I
            + map[pix[itod] +   num_pix] * cos2psi      //Q
            + map[pix[itod] + 2*num_pix] * sin2psi;     //U
    }
}

/** Simple serial mapmaker accumulating the weights (typically inverse-variance weights) for the above "map_accumulator".
 * 
 *  Args:
 *      map (OUTPUT) -- 1D array of length 'num_pix', representing the accumulated weight map.
 *      weight -- scalar, representing the weight to bin up.
 *      pix -- 1D array, containing the pixel pointing index of each element in tod.
 *      scan_len -- Length of the scan as an int.
 *      num_pix -- Number of pixels in map and map_rms.
 *      Note: this is LHS of Eqn. (77) in BP01
 */
template<typename T>
void _map_weight_accumulator_IQU_T(T *map, const T weight, int64_t *pix, const double *psi, int64_t scan_len, int64_t num_pix){
    for(int64_t itod=0; itod<scan_len; itod++){
        const T cos2psi = static_cast<T>(std::cos(2.0 * psi[itod]));
        const T sin2psi = static_cast<T>(std::sin(2.0 * psi[itod]));
        map[pix[itod]]             += weight;                 // II
        map[pix[itod] +   num_pix] += weight*cos2psi;         // IQ
        map[pix[itod] + 2*num_pix] += weight*sin2psi;         // IU
        map[pix[itod] + 3*num_pix] += weight*cos2psi*cos2psi; // QQ
        map[pix[itod] + 4*num_pix] += weight*sin2psi*cos2psi; // QU
        map[pix[itod] + 5*num_pix] += weight*sin2psi*sin2psi; // UU
    }
}


/** Invert a symmetric positive definite (SPD) 3x3 matrix.
 * Accepts the 6 (out of 9) unique elements of such a matrix as arguments,
 * as well as the 6 unique elements of the inverse output. Also checks for singularity.
 *
 * Args:
 *   a00_in, a01_in, a02_in, a11_in, a12_in, a22_in -- INPUT unique elements of A.
 *   inv00, inv01, inv02, inv11, inv12, inv22       -- OUTPUT unique inverse elements of A.
 *
 * Returns:
 *   true if inversion succeeded, false if matrix is singular/ill-conditioned.
 */
template<typename T>
inline bool _invert_SPD_3x3(const T a00, const T a01, const T a02,
                            const T a11, const T a12, const T a22,
                            T &inv00, T &inv01, T &inv02,
                            T &inv11, T &inv12, T &inv22){
    const T det = a00 * (a11 * a22 - a12 * a12)
                - a01 * (a01 * a22 - a02 * a12)
                + a02 * (a01 * a12 - a02 * a11);

    const T diag_prod = a00 * a11 * a22;
    // If the diagonal product is zero, we are singular, or matrix is not actually SPD.
    if (diag_prod <= std::numeric_limits<T>::min()) return false; 
    
    // Heuristic for Condition Number: det / (a00 * a11 * a22) roughly approximates
    // 1/condition_number assuming the matrix is positive definite.
    // If det is < 1e-12 of the diagonal product, the matrix is ill-conditioned.
    const T rcond_threshold = static_cast<T>(1e-12); 
    if (det <= rcond_threshold * diag_prod) return false;

    const T inv_det = static_cast<T>(1) / det;
    // Directly calculate the elements of the inverse of A:
    inv00 =  (a11 * a22 - a12 * a12) * inv_det;
    inv01 =  (a02 * a12 - a01 * a22) * inv_det;
    inv02 =  (a01 * a12 - a02 * a11) * inv_det;
    inv11 =  (a00 * a22 - a02 * a02) * inv_det;
    inv12 =  (a01 * a02 - a00 * a12) * inv_det;
    inv22 =  (a00 * a11 - a01 * a01) * inv_det;

    // Return true if inversion was successful (not singular or ill-conditioned matrix)
    return true;
}


/** Solve IQU mapmaking per pixel using explicit 3x3 inversion.
 *
 * Args:
 *   map_out (OUTPUT) -- 2D array [3, num_pix] for I,Q,U solution.
 *   map_rhs -- 2D array [3, num_pix] for b (accumulated map).
 *   norm_map -- 2D array [6, num_pix] with unique A elements (II, IQ, IU, QQ, QU, UU).
 *   num_pix -- Number of pixels.
 *
 * Singular or ill-conditioned pixels are zeroed.
 */
template<typename T>
void _map_solve_IQU_T(T *map_out, const T *map_rhs, const T *norm_map, int64_t num_pix){
    for (int64_t ipix = 0; ipix < num_pix; ipix++) {
        // Find the array elements that form the 3x3 A matrix to be inverted.
        const T a00 = norm_map[ipix];
        const T a01 = norm_map[num_pix + ipix];
        const T a02 = norm_map[2 * num_pix + ipix];
        const T a11 = norm_map[3 * num_pix + ipix];
        const T a12 = norm_map[4 * num_pix + ipix];
        const T a22 = norm_map[5 * num_pix + ipix];

        // Define the 6 unique elements of the inverse 3x3 matrix, and call the solver.
        T inv00, inv01, inv02, inv11, inv12, inv22;
        bool successful_invert = _invert_SPD_3x3(a00, a01, a02, a11, a12, a22,
                                                 inv00, inv01, inv02, inv11, inv12, inv22);
        if (!successful_invert){
            // If solver failed (singular matrix) set values to zero.
            map_out[ipix] = static_cast<T>(0);
            map_out[num_pix + ipix] = static_cast<T>(0);
            map_out[2 * num_pix + ipix] = static_cast<T>(0);
            continue;
        }

        const T b0 = map_rhs[ipix];
        const T b1 = map_rhs[num_pix + ipix];
        const T b2 = map_rhs[2 * num_pix + ipix];

        map_out[ipix] = inv00 * b0 + inv01 * b1 + inv02 * b2;
        map_out[num_pix + ipix] = inv01 * b0 + inv11 * b1 + inv12 * b2;
        map_out[2 * num_pix + ipix] = inv02 * b0 + inv12 * b1 + inv22 * b2;
    }
}

/** Compute RMS maps from inverse diagonal of per-pixel 3x3 covariances.
 *
 * Args:
 *   rms_out (OUTPUT) -- 2D array [3, num_pix] for rms(I), rms(Q), rms(U).
 *   norm_map -- 2D array [6, num_pix] with unique A elements (II, IQ, IU, QQ, QU, UU).
 *   num_pix -- Number of pixels.
 *
 * Singular or ill-conditioned pixels are zeroed.
 */
template<typename T>
void _map_invdiag_IQU_T(T *rms_out, const T *norm_map, int64_t num_pix){
    for (int64_t ipix = 0; ipix < num_pix; ipix++) {
        // Find the array elements that form the 3x3 A matrix to be inverted.
        const T a00 = norm_map[ipix];
        const T a01 = norm_map[num_pix + ipix];
        const T a02 = norm_map[2 * num_pix + ipix];
        const T a11 = norm_map[3 * num_pix + ipix];
        const T a12 = norm_map[4 * num_pix + ipix];
        const T a22 = norm_map[5 * num_pix + ipix];

        // Define the 6 unique elements of the inverse 3x3 matrix, and call the solver.
        T inv00, inv01, inv02, inv11, inv12, inv22;
        bool successful_invert = _invert_SPD_3x3(a00, a01, a02, a11, a12, a22,
                                                 inv00, inv01, inv02, inv11, inv12, inv22);
        if (!successful_invert){
            // If solver failed (singular matrix) set values to zero.
            rms_out[ipix] = 0.0;
            rms_out[num_pix + ipix] = 0.0;
            rms_out[2 * num_pix + ipix] = 0.0;
            continue;
        }
        
        // Rms values are the sqrt of the diagonal. However, set them to zero if diag is negative.
        rms_out[ipix] = inv00 > static_cast<T>(0) ? std::sqrt(inv00) : static_cast<T>(0);
        rms_out[num_pix + ipix] = inv11 > static_cast<T>(0) ? std::sqrt(inv11) : static_cast<T>(0);
        rms_out[2 * num_pix + ipix] = inv22 > static_cast<T>(0) ? std::sqrt(inv22) : static_cast<T>(0);
    }
}

/**
 * Below are the functions exposed to the user, meant to be used by Ctypes.
 * They are all wrapped in 'extern C' to be usable with Ctypes (not have their names scrambled).
 * The documentation for the various functions can be found in the template functions.
 */

extern "C"
void map_accumulator_f32(float *map, float *tod, float weight, int64_t *pix, int64_t scan_len){
    _map_accumulator_T<float>(map, tod, weight, pix, scan_len);
}

extern "C"
void map_accumulator_f64(double *map, double *tod, double weight, int64_t *pix, int64_t scan_len){
    _map_accumulator_T<double>(map, tod, weight, pix, scan_len);
}

extern "C"
void map_weight_accumulator_f32(float *map, float weight, int64_t *pix, int64_t scan_len){
    _map_weight_accumulator_T<float>(map, weight, pix, scan_len);
}

extern "C"
void map_weight_accumulator_f64(double *map, double weight, int64_t *pix, int64_t scan_len){
    _map_weight_accumulator_T<double>(map, weight, pix, scan_len);
}

extern "C"
void map_accumulator_IQU_f32(float *map, float *tod, float weight, int64_t *pix, double *psi, int64_t scan_len, int64_t num_pix){
    _map_accumulator_IQU_T<float>(map, tod, weight, pix, psi, scan_len, num_pix);
}

extern "C"
void map_accumulator_IQU_f64(double *map, double *tod, double weight, int64_t *pix, double *psi, int64_t scan_len, int64_t num_pix){
    _map_accumulator_IQU_T<double>(map, tod, weight, pix, psi, scan_len, num_pix);
}

extern "C"
void map2tod_IQU_f64(double *tod, double *map, int64_t *pix, double *psi, int64_t scan_len, int64_t num_pix){
    _map2tod_IQU_T<double>(tod, map, pix, psi, scan_len, num_pix);
}

extern "C"
void map2tod_IQU_f32(float *tod, float *map, int64_t *pix, double *psi, int64_t scan_len, int64_t num_pix){
    _map2tod_IQU_T<float>(tod, map, pix, psi, scan_len, num_pix);
}

extern "C"
void map_weight_accumulator_IQU_f32(float *map, float weight, int64_t *pix, double *psi, int64_t scan_len, int64_t num_pix){
    _map_weight_accumulator_IQU_T<float>(map, weight, pix, psi, scan_len, num_pix);
}

extern "C"
void map_weight_accumulator_IQU_f64(double *map, double weight, int64_t *pix, double *psi, int64_t scan_len, int64_t num_pix){
    _map_weight_accumulator_IQU_T<double>(map, weight, pix, psi, scan_len, num_pix);
}

extern "C"
void map_solve_IQU_f32(float *map_out, float *map_rhs, float *norm_map, int64_t num_pix){
    _map_solve_IQU_T<float>(map_out, map_rhs, norm_map, num_pix);
}

extern "C"
void map_solve_IQU_f64(double *map_out, double *map_rhs, double *norm_map, int64_t num_pix){
    _map_solve_IQU_T<double>(map_out, map_rhs, norm_map, num_pix);
}

extern "C"
void map_invdiag_IQU_f32(float *rms_out, float *norm_map, int64_t num_pix){
    _map_invdiag_IQU_T<float>(rms_out, norm_map, num_pix);
}

extern "C"
void map_invdiag_IQU_f64(double *rms_out, double *norm_map, int64_t num_pix){
    _map_invdiag_IQU_T<double>(rms_out, norm_map, num_pix);
}

