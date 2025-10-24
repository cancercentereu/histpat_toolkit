# Import necessary modules
import numpy as np
cimport numpy as np

cpdef np.ndarray[np.int32_t, ndim=2] get_labels_info(np.ndarray[np.uint8_t, ndim=2] tissue_mask, np.ndarray[np.int32_t, ndim=2] labels):
    cdef np.ndarray[np.int32_t, ndim=2] labels_info = np.ones((labels.max() + 1, 4), dtype=np.int32) * -1

    for y in range(tissue_mask.shape[0]):
        for x in range(tissue_mask.shape[1]):
            if tissue_mask[y, x] == 0:
                continue
            val = labels[y, x]
            if val == 0:
                continue

            if labels_info[val][0] == -1:
                labels_info[val, 0] = x
                labels_info[val, 1] = x
                labels_info[val, 2] = y
                labels_info[val, 3] = y
            else:
                labels_info[val, 0] = min(x, labels_info[val, 0])
                labels_info[val, 1] = max(x, labels_info[val, 1])
                labels_info[val, 2] = min(y, labels_info[val, 2])
                labels_info[val, 3] = max(y, labels_info[val, 3])
    
    return labels_info