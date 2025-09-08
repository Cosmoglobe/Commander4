// Compile as:
// g++ -shared -O3 mapmaker.cpp -o mapmaker.so
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

template<typename T_map, typename T_tod, typename T_weight>
void _map_accumulator_T(T_map *map, T_tod *tod, T_weight weight, int64_t *pix, int64_t scan_len, int64_t num_pix){
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

    for(int64_t itod=0; itod<scan_len; itod++){
        map[pix[itod]] += tod[itod]*weight;
    }
}

template<typename T_map, typename T_weight>
void _map_weight_accumulator_T(T_map *map, T_weight weight, int64_t *pix, int64_t scan_len, int64_t num_pix){
    /** Simple serial mapmaker accumulating the weights (typically inverse-variance weights) for the above "map_accumulator".
     * 
     *  Args:
     *      map (OUTPUT) -- 1D array of length 'num_pix', representing the accumulated weight map.
     *      weight -- scalar, representing the weight to bin up.
     *      pix -- 1D array, containing the pixel pointing index of each element in tod.
     *      scan_len -- Length of the scan as an int.
     *      num_pix -- Number of pixels in map and map_rms.
     */

    for(int64_t itod=0; itod<scan_len; itod++){
        map[pix[itod]] += weight;
    }
}

template<typename T_map, typename T_tod, typename T_weight>
void _map_accumulator_IQU_T(T_map *map, T_tod *tod, T_weight weight, int64_t *pix, double *psi, int64_t scan_len, int64_t num_pix){
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
     *      Note: this is RHS of Eqn. (77) in BP01
     */

    for(int64_t itod=0; itod<scan_len; itod++){
        map[pix[itod]]             += tod[itod]*weight;                  // I
        map[pix[itod] +   num_pix] += tod[itod]*cos(2*psi[itod])*weight; // Q
        map[pix[itod] + 2*num_pix] += tod[itod]*sin(2*psi[itod])*weight; // U
    }
}

template<typename T_map, typename T_weight>
void _map_weight_accumulator_IQU_T(T_map *map, T_weight weight, int64_t *pix, double *psi, int64_t scan_len, int64_t num_pix){
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

    for(int64_t itod=0; itod<scan_len; itod++){
        double cos2psi = cos(2*psi[itod]);
        double sin2psi = sin(2*psi[itod]);
        map[pix[itod]]             += weight;                 // II
        map[pix[itod] +   num_pix] += weight*cos2psi;         // IQ
        map[pix[itod] + 2*num_pix] += weight*sin2psi;         // IU
        map[pix[itod] + 3*num_pix] += weight*cos2psi*cos2psi; // QQ
        map[pix[itod] + 4*num_pix] += weight*sin2psi*cos2psi; // QU
        map[pix[itod] + 5*num_pix] += weight*sin2psi*sin2psi; // UU
    }
}

/**
 * Below are the functions exposed to the user, meant to be used by Ctypes.
 * They are all wrapped in 'extern C' to be usable with Ctypes (not have their names scrambled).
 */

extern "C"
void map_accumulator_f32(float *map, float *tod, float weight, int64_t *pix, int64_t scan_len, int64_t num_pix){
    _map_accumulator_T<float, float, float>(map, tod, weight, pix, scan_len, num_pix);
}

extern "C"
void map_weight_accumulator_f32(float *map, float weight, int64_t *pix, int64_t scan_len, int64_t num_pix){
    _map_weight_accumulator_T<float, float>(map, weight, pix, scan_len, num_pix);
}

extern "C"
void map_accumulator_IQU_f32(float *map, float *tod, float weight, int64_t *pix, double *psi, int64_t scan_len, int64_t num_pix){
    _map_accumulator_IQU_T<float, float, float>(map, tod, weight, pix, psi, scan_len, num_pix);
}

extern "C"
void map_weight_accumulator_IQU_f32(float *map, float weight, int64_t *pix, double *psi, int64_t scan_len, int64_t num_pix){
    _map_weight_accumulator_IQU_T<float, float>(map, weight, pix, psi, scan_len, num_pix);
}

