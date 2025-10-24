# Import necessary modules
import numpy as np
cimport numpy as np

cpdef np.ndarray[np.int32_t, ndim=2] cython_bfs_nearest_labeling(np.ndarray[np.uint8_t, ndim=2] tissue_mask, np.ndarray[np.int32_t, ndim=2] labels):
    cdef np.ndarray[np.int32_t, ndim=2] queue = np.zeros((tissue_mask.size, 3), dtype=np.int32)
    cdef int queue_start = 0
    cdef int queue_end = 0

    cdef np.ndarray[np.intp_t, ndim=1] where_anchors_y, where_anchors_x
    where_anchors_y, where_anchors_x = np.where(labels)
    for y, x in zip(where_anchors_y, where_anchors_x):
        val = labels[y, x]
        if val == 0:
            continue

        queue[queue_end, 0] = x
        queue[queue_end, 1] = y
        queue[queue_end, 2] = val
        queue_end += 1
    
    while queue_start != queue_end:
        x, y, val = queue[queue_start, 0], queue[queue_start, 1], queue[queue_start, 2]
        queue_start += 1

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= tissue_mask.shape[1] or ny < 0 or ny >= tissue_mask.shape[0]:
                    continue
                if labels[ny, nx] != 0:
                    continue
                if tissue_mask[ny, nx] == 0:
                    continue

                labels[ny, nx] = val
                queue[queue_end, 0] = nx
                queue[queue_end, 1] = ny
                queue[queue_end, 2] = val
                queue_end += 1
    
    return labels