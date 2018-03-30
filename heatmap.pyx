cimport cython
import numpy as np
cimport numpy as np

ctypedef np.float_t DTYPE_t

def putGaussianMaps(np.ndarray[DTYPE_t, ndim = 2] entry, DTYPE_t rows,
                    DTYPE_t cols, DTYPE_t center_x, DTYPE_t center_y,  DTYPE_t stride,
                    int grid_x, int grid_y, DTYPE_t sigma):
    cdef DTYPE_t start = stride / 2.0 - 0.5
    cdef DTYPE_t x, y, d2

    for g_y in range(grid_y):
        for g_x in range(grid_x):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)
            exponent = d2 / 2.0 / sigma / sigma
            if (exponent > 4.6052):
                continue
            entry[g_y, g_x] += np.exp(-exponent)
            if (entry[g_y, g_x] > 1):
                entry[g_y, g_x] = 1
    return entry


def putVecMaps(np.ndarray[DTYPE_t, ndim = 2] entryX, np.ndarray[DTYPE_t, ndim = 2]  entryY,
               np.ndarray[DTYPE_t, ndim = 2] count, DTYPE_t center1_x, DTYPE_t center1_y,
               DTYPE_t center2_x, DTYPE_t center2_y, DTYPE_t stride, DTYPE_t grid_x, DTYPE_t grid_y,
               DTYPE_t sigma, DTYPE_t thre):

    cdef DTYPE_t centerA_x = center1_x / stride
    cdef DTYPE_t centerA_y = center1_y / stride

    cdef DTYPE_t centerB_x = center2_x / stride
    cdef DTYPE_t centerB_y = center2_y / stride

    cdef DTYPE_t bc_x = centerB_x - centerA_x
    cdef DTYPE_t bc_y = centerB_y - centerA_y

    cdef int min_x = max(int(round(min(centerA_x, centerB_x)) - thre), 0)
    cdef int max_x = min(int(round(max(centerA_x, centerB_x))) + thre, grid_x)
    cdef int min_y = max(int(round(min(centerA_y, centerB_y) - thre)), 0)
    cdef int max_y = min(int(round(max(centerA_y, centerB_y) + thre)), grid_y)

    cdef DTYPE_t norm_bc = np.sqrt(bc_x * bc_x + bc_y * bc_y)
    if norm_bc == 0:
        return

    bc_x = bc_x / norm_bc
    bc_y = bc_y / norm_bc

    cdef DTYPE_t ba_x, ba_y
    cdef DTYPE_t dist
    cdef DTYPE_t cnt

    for g_y in range(min_y, max_y):
        for g_x in range(min_x, max_x):
            ba_x = g_x - centerA_x
            ba_y = g_y - centerA_y
            dist = np.absolute(ba_x * bc_y - ba_y * bc_x)

            if (dist <= thre):
                cnt = count[g_y, g_x]
                if (cnt == 0):
                    entryX[g_y, g_x] = bc_x
                    entryY[g_y, g_x] = bc_y
                else:
                    entryX[g_y, g_x] = (entryX[g_y, g_x] * cnt + bc_x) / (cnt + 1)
                    entryY[g_y, g_x] = (entryY[g_y, g_x] * cnt + bc_y) / (cnt + 1)
                    count[g_y, g_x] = cnt + 1
