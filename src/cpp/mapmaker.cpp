// Compile as:
// g++ -shared -O3 mapmaker.cpp -o mapmaker.so
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

extern "C"
void map_accumulator(double *map, double *tod, double weight, int64_t *pix, int64_t scan_len, int64_t num_pix){
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

extern "C"
void map_weight_accumulator(double *map, double weight, int64_t *pix, int64_t scan_len, int64_t num_pix){
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