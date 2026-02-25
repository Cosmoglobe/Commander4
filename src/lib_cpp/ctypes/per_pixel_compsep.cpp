// Compile as:
// g++ -shared -O3 -fPIC mapmaker.cpp -o mapmaker.so

#include <cmath>
#include <vector>
#include <omp.h>

/**
 * Simple Cholesky decomposition and solver for small matrices (A x = b)
 * A is symmetric positive definite, stored in row-major order
 * Returns true on success, false if matrix is not positive definite
 */
bool solve_cholesky(int n, std::vector<double>& A, std::vector<double>& b, std::vector<double>& x) {
    std::vector<double> L(n * n, 0.0);

    // Cholesky Decomposition: A = L * L^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }

            if (i == j) {
                double val = A[i * n + i] - sum;
                if (val <= 0) return false; // Not positive definite
                L[i * n + j] = std::sqrt(val);
            } else {
                L[i * n + j] = (1.0 / L[j * n + j]) * (A[i * n + j] - sum);
            }
        }
    }

    // Forward substitution: L * y = b
    std::vector<double> y(n);
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i * n + i];
    }

    // Backward substitution: L^T * x = y
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += L[j * n + i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i * n + i];
    }
    
    return true;
}


extern "C" {

/**
 * Perform pixel-wise component separation.
 * * Dimensions:
 * npix: Number of pixels
 * nband: Number of frequency bands
 * ncomp: Number of components
 * * Pointers (Standard C-Contiguous Numpy Arrays):
 * maps_sky: (nband, npix)
 * maps_rms: (nband, npix)
 * M:        (nband, ncomp) (Mixing matrix)
 * rand:     (npix, nband)  (Note: Python code implies this shape)
 * out_map:  (ncomp, npix)  (Output)
 */
void solve_compsep(int npix, int nband, int ncomp, const double* maps_sky, const double* maps_rms,
    const double* M, const double* rand, double* out_map){

    // Parallelize over pixels
    #pragma omp parallel
    {
        // Pre-allocate thread-local workspace to avoid malloc overhead in loop
        std::vector<double> A(ncomp * ncomp);
        std::vector<double> rhs(ncomp);
        std::vector<double> sol(ncomp);

        #pragma omp for schedule(static)
        for (int i = 0; i < npix; i++) {
            
            // 1. Reset A and RHS
            std::fill(A.begin(), A.end(), 0.0);
            std::fill(rhs.begin(), rhs.end(), 0.0);

            // 2. Build A (LHS) and b (RHS)
            // A = M.T * diag(w) * M
            // b = M.T * (w*sky + sqrt(w)*rand)
            // where w = 1/rms^2
            
            for (int b = 0; b < nband; b++) {
                // Access logic based on array shapes
                // maps_sky is (nband, npix) -> idx = b * npix + i
                // maps_rms is (nband, npix) -> idx = b * npix + i
                // rand     is (npix, nband) -> idx = i * nband + b
                
                double rms_val = maps_rms[b * npix + i];
                double inv_rms = 1.0 / rms_val;
                double inv_var = inv_rms * inv_rms; // w
                
                double sky_val = maps_sky[b * npix + i];
                double rnd_val = rand[i * nband + b];
                
                // The term in the parenthesis from python: 
                // (inv_rms_map**2 * maps_sky) + (rand * inv_rms_map)
                // = (inv_var * sky) + (rnd * inv_rms)
                double weighted_d = (inv_var * sky_val) + (rnd_val * inv_rms);

                for (int c1 = 0; c1 < ncomp; c1++) {
                    double m_val_1 = M[b * ncomp + c1];
                    
                    // Accumulate RHS
                    rhs[c1] += m_val_1 * weighted_d;

                    // Accumulate LHS (Upper triangle only is sufficient for Cholesky 
                    // but we fill full for simplicity or if we swap solvers later)
                    for (int c2 = 0; c2 < ncomp; c2++) {
                        double m_val_2 = M[b * ncomp + c2];
                        A[c1 * ncomp + c2] += m_val_1 * inv_var * m_val_2;
                    }
                }
            }

            // 3. Solve Ax = b
            bool success = solve_cholesky(ncomp, A, rhs, sol);

            // 4. Write to output
            // out_map is (ncomp, npix) -> idx = c * npix + i
            if (success) {
                for (int c = 0; c < ncomp; c++) {
                    out_map[c * npix + i] = sol[c];
                }
            } else {
                // Fallback for singular matrices (optional: set to 0 or NaN)
                for (int c = 0; c < ncomp; c++) {
                    out_map[c * npix + i] = 0.0;
                }
            }
        }
    }
}

}