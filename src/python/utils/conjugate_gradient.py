import numpy as np
from math_operations import alm_dot_product

def conjugate_gradient(A, b, x0, M_inv, NPIX=1, pix_obs=[False], tol=1e-5, max_iter=1000):
    # Conjugate Gradient Method for solving the linear system Ax = b.
    # Can use NPIX and pix_obs if solving for a masked map in map space
    #
    # A - The coefficient operator/matrix (square, symmetric, positive-definite). 
    # b - The right-hand side vector.
    # x0 - Initial guess for the solution.
    # M_inv - The inverse of the preconditioner matrix.
    # NPIX - The number of pixels (required for masked formats)
    # pix_obs - Observed pixels (required for masked formats)
    # tol - Tolerance for convergence.
    # max_iter - Maximum number of iterations.
    #
    # return The approximate solution x and the rms 


    experiment_name = 'test'

    x = x0
    r = b - A(x)
    d = M_inv.dot(r)
    delta = np.dot(r, d)
    delta0 = delta
    file = open('delta_CG_{0}.txt'.format(experiment_name), 'w')
    file.write('{0} {1}\n'.format(0, 1))
    file.close()

    # if the data is reduced
    if all(pix_obs):
        x_map = np.zeros(NPIX)
        r_map = np.zeros(NPIX)

    for k in range(max_iter):
        q = A(d)
        alpha = delta / np.dot(d, q)
        x += alpha * d
        r -= alpha * q

        s = M_inv.dot(r)
        delta_new = np.dot(r, s)

        file = open('delta_CG_{0}.txt'.format(experiment_name), 'a')
        file.write('{0} {1}\n'.format(k+1, delta_new/delta0))
        file.close()

        if delta_new < tol*tol*delta0:
            break
        beta = delta_new / delta
        d = s + beta * d
        delta = delta_new

    if all(pix_obs):
        x_map[pix_obs] = x
        r_map[pix_obs] = r
    else:
        x_map = x
        r_map = r

    return x_map, r_map


def conjugate_gradient_alm(A, b, x0, M_inv, tol=1e-5, max_iter=1000, write_output=True):
    # Conjugate Gradient Method for solving the linear system Ax = b.
    #
    # A - The coefficient operator/matrix (square, symmetric, positive-definite).
    # b - The right-hand side vector.
    # x0 - Initial guess for the solution.
    # M_inv - The inverse of the preconditioner matrix.
    # tol - Tolerance for convergence.
    # max_iter - Maximum number of iterations.
    # write_output - whether to write convergance info into a txt file
    #
    # return The approximate solution x and the rms


    experiment_name = 'test'

    x = x0
    r = b - A(x)
    d = alm_dot_product(M_inv, r)
    delta = alm_dot_product(r, d)
    delta0 = delta

    if write_output:
        file = open('delta_CG_{0}.txt'.format(experiment_name), 'w')
        file.write('{0} {1}\n'.format(0, 1))
        file.close()

    for k in range(max_iter):
        q = A(d)
        alpha = delta / alm_dot_product(d, q)
        x += alpha * d
        r -= alpha * q

        s = alm_dot_product(M_inv, r)
        delta_new = alm_dot_product(r, s)

        if write_output:
            file = open('delta_CG_{0}.txt'.format(experiment_name), 'a')
            file.write('{0} {1}\n'.format(k+1, delta_new/delta0))
            file.close()

        if delta_new < tol*tol*delta0:
            break
        beta = delta_new / delta
        d = s + beta * d
        delta = delta_new

    return x_map, r_map
